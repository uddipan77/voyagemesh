"""Deterministic day-by-day itinerary construction.

The planner is a scheduling algorithm, not a prompt. It places attractions into days while
respecting travel time, meal breaks, opening-hour uncertainty, accessibility needs, and
weather suitability. The LLM's only role downstream is to *describe* the resulting plan.

Feasibility is guaranteed twice over: the planner reserves travel time between consecutive
slots, and :class:`~vm_contracts.itinerary.ItineraryDay` independently rejects any schedule
whose gaps are too small. If the planner ever produced an impossible day, construction
would raise rather than the plan reaching a user.

Weather handling
----------------
On a day forecast to be unsuitable for outdoor activity, indoor attractions are preferred.
Outdoor ones that still get scheduled carry an explicit ``weather_warning`` — we surface
the risk rather than silently dropping the highlight of the trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from vm_contracts.common import AccessibilityNeed, DataOrigin, Money
from vm_contracts.destination import Attraction, WeatherDay, WeatherSummary
from vm_contracts.itinerary import ActivitySlot, Itinerary, ItineraryDay, SlotKind
from vm_contracts.offers import AccommodationOffer, TransportOffer
from vm_contracts.trip import NormalizedTripRequest
from vm_domain.geo import estimate_travel_minutes

__all__ = ["ItineraryPlanner", "PlanningWindow"]

# Sightseeing day boundaries, in minutes from midnight.
DAY_START = 9 * 60  # 09:00
DAY_END = 19 * 60  # 19:00
LUNCH_START = 12 * 60 + 30  # 12:30
LUNCH_MINUTES = 60
DINNER_START = 19 * 60  # 19:00
DINNER_MINUTES = 75

CHECK_IN_MINUTE = 15 * 60  # 15:00 — near-universal hotel policy
CHECK_OUT_MINUTE = 11 * 60  # 11:00

MIN_SLOT_MINUTES = 20
_ARRIVAL_BUFFER_MINUTES = 45
"""Time from arrival at the destination before sightseeing can realistically begin."""


@dataclass(frozen=True, slots=True)
class PlanningWindow:
    """The usable part of one day, once travel and check-in are accounted for."""

    day_number: int
    day_date: date
    earliest_minute: int
    latest_minute: int

    @property
    def available_minutes(self) -> int:
        return max(0, self.latest_minute - self.earliest_minute)


class ItineraryPlanner:
    """Builds a feasible itinerary from attractions, weather, and trip constraints."""

    def __init__(self, *, max_attractions_per_day: int = 4) -> None:
        self._max_per_day = max(1, max_attractions_per_day)

    def plan(
        self,
        request: NormalizedTripRequest,
        *,
        attractions: list[Attraction],
        weather: WeatherSummary | None = None,
        accommodation: AccommodationOffer | None = None,
        outbound: TransportOffer | None = None,
        inbound: TransportOffer | None = None,
    ) -> Itinerary:
        """Produce a day-by-day plan.

        Args:
            request: Supplies dates, interests, and accessibility needs.
            attractions: Candidate points of interest.
            weather: Forecast used for outdoor suitability. ``None`` means unknown, which
                is treated as "no warning available" rather than "fine".
            accommodation: Anchors travel-time estimates and check-in/out slots.
            outbound: Constrains when day 1 can start.
            inbound: Constrains when the final day must end.

        Returns:
            An :class:`Itinerary`. Never raises for ordinary shortages — a destination with
            no attractions yields days containing meals and transfers plus a warning.
        """
        warnings: list[str] = []
        eligible, excluded = self._filter_accessible(attractions, request.accessibility_needs)
        if excluded:
            warnings.append(
                f"{len(excluded)} attraction(s) were excluded because their accessibility "
                f"is unsuitable or unknown for the stated needs."
            )
        if not eligible:
            warnings.append(
                "No attractions matched this destination and the stated preferences. "
                "The itinerary covers travel and meals only."
            )

        ordered = self._prioritise(eligible, request.interests)
        windows = self._build_windows(request, outbound=outbound, inbound=inbound)

        low_walking = AccessibilityNeed.LOW_WALKING_DISTANCE in request.accessibility_needs

        remaining = list(ordered)
        days: list[ItineraryDay] = []

        for window in windows:
            day_weather = weather.for_date(window.day_date) if weather is not None else None
            outdoor_ok = day_weather.is_outdoor_friendly if day_weather is not None else None

            chosen = self._select_for_day(remaining, outdoor_ok=outdoor_ok, window=window)
            for attraction in chosen:
                remaining.remove(attraction)

            slots = self._build_slots(
                window=window,
                attractions=chosen,
                request=request,
                low_walking=low_walking,
                outdoor_ok=outdoor_ok,
                accommodation=accommodation,
                outbound=outbound if window.day_number == 1 else None,
                inbound=inbound if window.day_number == len(windows) else None,
            )

            days.append(
                ItineraryDay(
                    day_number=window.day_number,
                    day_date=window.day_date,
                    summary=self._summarise(window, chosen),
                    slots=slots,
                    weather_note=self._weather_note(day_weather),
                    total_walking_km=None,
                )
            )

        return Itinerary(
            days=days,
            overview=self._overview(request, days),
            unscheduled_suggestions=[a.name for a in remaining[:10]],
            warnings=warnings[:20],
        )

    # -- selection ---------------------------------------------------------
    def _filter_accessible(
        self, attractions: list[Attraction], needs: list[AccessibilityNeed]
    ) -> tuple[list[Attraction], list[Attraction]]:
        """Split into usable and excluded.

        When accessibility is *unknown* and needs were stated, the attraction is excluded.
        Scheduling a site we cannot confirm is step-free for someone who requires step-free
        access would be worse than omitting it — and the omission is reported.
        """
        if not needs:
            return list(attractions), []
        eligible: list[Attraction] = []
        excluded: list[Attraction] = []
        for attraction in attractions:
            (eligible if attraction.is_accessible_for(needs) is True else excluded).append(
                attraction
            )
        return eligible, excluded

    def _prioritise(self, attractions: list[Attraction], interests: list[str]) -> list[Attraction]:
        """Order by interest relevance, then deterministically by ID.

        The ID tiebreak matters: without it, two equally-relevant attractions could swap
        order between runs and break the cache and the evaluation suite.
        """

        def rank(attraction: Attraction) -> tuple[int, str]:
            tags = {attraction.category.value, *(t.lower() for t in attraction.interest_tags)}
            matches = sum(1 for interest in interests if interest.lower() in tags)
            return (-matches, attraction.attraction_id)

        return sorted(attractions, key=rank)

    def _select_for_day(
        self, candidates: list[Attraction], *, outdoor_ok: bool | None, window: PlanningWindow
    ) -> list[Attraction]:
        """Choose attractions that fit the day's usable time and weather."""
        if not candidates or window.available_minutes < MIN_SLOT_MINUTES:
            return []

        pool = list(candidates)
        if outdoor_ok is False:
            # Indoor first, but outdoor options stay available rather than being discarded:
            # a wet day with nothing scheduled is a worse outcome than a damp walk.
            pool.sort(key=lambda a: (a.is_outdoor, a.attraction_id))

        selected: list[Attraction] = []
        # Reserve lunch plus a nominal transfer allowance so selection cannot claim the
        # entire window and leave the scheduler unable to place a meal.
        budget = window.available_minutes - LUNCH_MINUTES - 30

        for attraction in pool:
            if len(selected) >= self._max_per_day:
                break
            cost = attraction.typical_visit_minutes + 20  # visit plus nominal transfer
            if cost <= budget:
                selected.append(attraction)
                budget -= cost

        return selected

    # -- scheduling --------------------------------------------------------
    def _build_windows(
        self,
        request: NormalizedTripRequest,
        *,
        outbound: TransportOffer | None,
        inbound: TransportOffer | None,
    ) -> list[PlanningWindow]:
        windows: list[PlanningWindow] = []
        for offset in range(request.days):
            day_date = request.departure_date + timedelta(days=offset)
            earliest, latest = DAY_START, DAY_END

            if offset == 0 and outbound is not None:
                arrival = outbound.arrival_at
                earliest = max(
                    DAY_START, arrival.hour * 60 + arrival.minute + _ARRIVAL_BUFFER_MINUTES
                )
            if offset == request.days - 1 and inbound is not None:
                departure = inbound.departure_at
                latest = min(DAY_END, departure.hour * 60 + departure.minute - 90)

            windows.append(
                PlanningWindow(
                    day_number=offset + 1,
                    day_date=day_date,
                    earliest_minute=min(earliest, DAY_END),
                    latest_minute=max(latest, min(earliest, DAY_END)),
                )
            )
        return windows

    def _build_slots(
        self,
        *,
        window: PlanningWindow,
        attractions: list[Attraction],
        request: NormalizedTripRequest,
        low_walking: bool,
        outdoor_ok: bool | None,
        accommodation: AccommodationOffer | None,
        outbound: TransportOffer | None,
        inbound: TransportOffer | None,
    ) -> list[ActivitySlot]:
        """Schedule one day.

        Fixed commitments (arrival, check-in, meals, departure) are placed first, then
        attractions are fitted into the gaps *between* them.

        An earlier version advanced a single cursor for attractions while appending fixed
        slots independently, which let a 15:00 check-in collide with an attraction already
        occupying that time. ItineraryDay's overlap validator caught it rather than a user
        seeing an impossible schedule.
        """
        fixed = self._fixed_slots(
            window=window,
            request=request,
            accommodation=accommodation,
            outbound=outbound,
            inbound=inbound,
        )
        free = self._free_intervals(fixed, window)
        placed = self._place_attractions(
            attractions,
            free_intervals=free,
            accommodation=accommodation,
            low_walking=low_walking,
            outdoor_ok=outdoor_ok,
        )
        return sorted([*fixed, *placed], key=lambda slot: slot.start_minute)

    def _fixed_slots(
        self,
        *,
        window: PlanningWindow,
        request: NormalizedTripRequest,
        accommodation: AccommodationOffer | None,
        outbound: TransportOffer | None,
        inbound: TransportOffer | None,
    ) -> list[ActivitySlot]:
        """Commitments whose times are dictated by transport schedules or convention."""
        slots: list[ActivitySlot] = []

        if outbound is not None:
            arrival_minute = outbound.arrival_at.hour * 60 + outbound.arrival_at.minute
            end = min(1440, arrival_minute + 30)
            if end > arrival_minute:
                slots.append(
                    ActivitySlot(
                        kind=SlotKind.ARRIVAL,
                        title=f"Arrive in {request.destination.title()}",
                        start_minute=max(0, arrival_minute),
                        end_minute=end,
                        description=(
                            f"Arriving on the {outbound.legs[-1].mode.value} service at "
                            f"{outbound.arrival_at.strftime('%H:%M')}."
                        ),
                    )
                )

        if accommodation is not None and window.day_number == 1:
            check_in = max(CHECK_IN_MINUTE, window.earliest_minute)
            if check_in + 30 <= 1440:
                slots.append(
                    ActivitySlot(
                        kind=SlotKind.CHECK_IN,
                        title=f"Check in - {accommodation.name}",
                        start_minute=check_in,
                        end_minute=check_in + 30,
                        description=accommodation.address,
                    )
                )

        if inbound is not None:
            departure_minute = inbound.departure_at.hour * 60 + inbound.departure_at.minute
            start = max(0, departure_minute - 60)
            if departure_minute > start:
                slots.append(
                    ActivitySlot(
                        kind=SlotKind.DEPARTURE,
                        title=f"Depart for {request.origin.title()}",
                        start_minute=start,
                        end_minute=min(1440, departure_minute),
                        description=(
                            f"Departing on the {inbound.legs[0].mode.value} service at "
                            f"{inbound.departure_at.strftime('%H:%M')}."
                        ),
                    )
                )

        # The window during which the traveller is actually at the destination. Meals are
        # only meaningful inside it: an earlier version scheduled lunch and dinner on the
        # final day *after* an 11:45 departure, cheerfully feeding someone already on a
        # coach home.
        present_from = 0
        present_until = 1440
        for slot in slots:
            if slot.kind is SlotKind.ARRIVAL:
                present_from = max(present_from, slot.start_minute)
            if slot.kind is SlotKind.DEPARTURE:
                present_until = min(present_until, slot.start_minute)

        # Meals are otherwise a requirement, not a nicety the scheduler may drop when the
        # day is busy. Placing them at fixed times forces attractions to work around them
        # rather than the reverse.
        for title, start, length in (
            ("Lunch", LUNCH_START, LUNCH_MINUTES),
            ("Dinner", DINNER_START, DINNER_MINUTES),
        ):
            end = start + length
            if start < present_from or end > present_until:
                continue
            collides = any(slot.start_minute < end and start < slot.end_minute for slot in slots)
            if end <= 1440 and not collides:
                slots.append(
                    ActivitySlot(
                        kind=SlotKind.MEAL,
                        title=title,
                        start_minute=start,
                        end_minute=end,
                        description=f"{title} break.",
                    )
                )

        return sorted(slots, key=lambda slot: slot.start_minute)

    def _free_intervals(
        self, fixed: list[ActivitySlot], window: PlanningWindow
    ) -> list[tuple[int, int]]:
        """Usable gaps inside the planning window, once fixed slots are removed."""
        intervals: list[tuple[int, int]] = []
        cursor = window.earliest_minute

        for slot in sorted(fixed, key=lambda s: s.start_minute):
            if slot.start_minute > cursor:
                intervals.append((cursor, min(slot.start_minute, window.latest_minute)))
            cursor = max(cursor, slot.end_minute)

        if cursor < window.latest_minute:
            intervals.append((cursor, window.latest_minute))

        return [(a, b) for a, b in intervals if b - a >= MIN_SLOT_MINUTES]

    def _place_attractions(
        self,
        attractions: list[Attraction],
        *,
        free_intervals: list[tuple[int, int]],
        accommodation: AccommodationOffer | None,
        low_walking: bool,
        outdoor_ok: bool | None,
    ) -> list[ActivitySlot]:
        """Fit attractions into the free gaps, reserving travel time before each."""
        slots: list[ActivitySlot] = []
        pending = list(attractions)
        previous_location = accommodation.location if accommodation is not None else None

        for interval_start, interval_end in free_intervals:
            cursor = interval_start
            while pending:
                attraction = pending[0]
                travel = estimate_travel_minutes(
                    previous_location,
                    attraction.location,
                    low_walking_distance=low_walking,
                )
                start = cursor + travel
                end = start + attraction.typical_visit_minutes
                if end > interval_end:
                    break  # does not fit here; try the next gap

                pending.pop(0)
                warning = None
                if outdoor_ok is False and attraction.is_outdoor:
                    warning = (
                        "Rain or extreme temperatures are estimated for this day; this is "
                        "an outdoor site."
                    )

                slots.append(
                    ActivitySlot(
                        kind=SlotKind.ATTRACTION,
                        title=attraction.name,
                        start_minute=start,
                        end_minute=end,
                        attraction_id=attraction.attraction_id,
                        description=attraction.description,
                        estimated_cost=attraction.admission_price,
                        cost_is_unknown=attraction.provenance.origin
                        in (DataOrigin.LIVE, DataOrigin.CACHED)
                        and attraction.admission_price is None
                        and not attraction.is_free,
                        is_outdoor=attraction.is_outdoor,
                        weather_warning=warning,
                        accessibility_note=(
                            None
                            if attraction.accessibility_is_known
                            else "Accessibility information is not available for this site."
                        ),
                        # The fixture corpus deliberately omits opening hours, so every
                        # fixture-sourced timing is an estimate and is flagged as such.
                        is_uncertain=not attraction.opening_hours,
                    )
                )
                cursor = end
                previous_location = attraction.location

        return slots

    # -- narration ---------------------------------------------------------
    def _summarise(self, window: PlanningWindow, attractions: list[Attraction]) -> str:
        if not attractions:
            return f"Day {window.day_number}: travel, meals, and free time."
        names = ", ".join(a.name for a in attractions[:3])
        extra = f" and {len(attractions) - 3} more" if len(attractions) > 3 else ""
        return f"Day {window.day_number}: {names}{extra}."

    def _weather_note(self, day: WeatherDay | None) -> str | None:
        if day is None:
            return None
        return (
            f"{day.condition.title()}, up to {day.temperature_max_c:.0f} °C, "
            f"{day.precipitation_mm:.1f} mm precipitation."
        )

    def _overview(self, request: NormalizedTripRequest, days: list[ItineraryDay]) -> str:
        scheduled = sum(1 for day in days for slot in day.slots if slot.kind is SlotKind.ATTRACTION)
        return (
            f"{request.days}-day itinerary for {request.destination.title()} with "
            f"{scheduled} scheduled attraction(s) across {len(days)} day(s). "
            f"Timings are estimates and allow for travel between locations."
        )


def total_admission_cost(itinerary: Itinerary) -> Money | None:
    """Convenience re-export of the itinerary's own priced total."""
    return itinerary.estimated_activity_cost()
