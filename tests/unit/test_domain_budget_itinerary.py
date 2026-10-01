"""Budget arithmetic and itinerary feasibility.

Together these cover the project's central claim: every number and every schedule the user
sees is produced by code that can be reasoned about and re-run, not by a model.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from itertools import pairwise

import pytest

from vm_contracts import (
    AccessibilityNeed,
    AccommodationOffer,
    AccommodationType,
    Attraction,
    AttractionCategory,
    BudgetCategory,
    BudgetStatus,
    Currency,
    GeoPoint,
    Money,
    NormalizedTripRequest,
    Provenance,
    SlotKind,
    TransportLeg,
    TransportMode,
    TransportOffer,
    TripRequest,
    WeatherDay,
    WeatherSummary,
)
from vm_domain.budget import (
    calculate_budget,
    default_allowances,
    remaining_accommodation_allowance,
)
from vm_domain.itinerary import ItineraryPlanner
from vm_domain.providers.base import POIQuery, WeatherQuery
from vm_domain.providers.fixtures import FixturePOIProvider, FixtureWeatherProvider

pytestmark = pytest.mark.unit

EUR = Currency.EUR
DEPART = date(2026, 8, 10)
RETURN = date(2026, 8, 13)
PRAGUE = GeoPoint(latitude=50.0875, longitude=14.4213)


def request_for(**overrides) -> NormalizedTripRequest:
    defaults = {
        "origin": "Nuremberg",
        "destination": "Prague",
        "departure_date": DEPART,
        "return_date": RETURN,
        "travellers": 1,
        "max_budget": Money.of(350, EUR),
        "accommodation_preference": AccommodationType.HOSTEL,
        "interests": ["history", "architecture"],
    }
    return NormalizedTripRequest.from_request(TripRequest(**{**defaults, **overrides}))


def transport_offer(price: str = "60", *, arrive_hour: int = 12) -> TransportOffer:
    departure = datetime(2026, 8, 10, 8, 0, tzinfo=UTC)
    arrival = departure.replace(hour=arrive_hour)
    return TransportOffer(
        offer_id="t1",
        legs=[
            TransportLeg(
                mode=TransportMode.TRAIN,
                carrier="Test Rail",
                from_location="A",
                to_location="B",
                departure_at=departure,
                arrival_at=arrival,
            )
        ],
        total_price=Money.of(price, EUR),
        price_per_traveller=Money.of(price, EUR),
        travellers=1,
        provenance=Provenance.mocked("test"),
    )


def stay_offer(nightly: str = "30", nights: int = 3) -> AccommodationOffer:
    return AccommodationOffer(
        offer_id="a1",
        name="Test Hostel",
        accommodation_type=AccommodationType.HOSTEL,
        price_per_night=Money.of(nightly, EUR),
        total_price=Money.of(nightly, EUR).times(nights),
        nights=nights,
        guests=1,
        location=PRAGUE,
        provenance=Provenance.mocked("test"),
    )


class TestBudget:
    def test_sums_all_components(self):
        summary = calculate_budget(
            request_for(), transport=transport_offer("60"), accommodation=stay_offer("30")
        )
        assert summary.amount_for(BudgetCategory.TRANSPORT) == Money.of(60, EUR)
        assert summary.amount_for(BudgetCategory.ACCOMMODATION) == Money.of(90, EUR)
        assert summary.total == sum(
            (line.amount for line in summary.lines[1:]), start=summary.lines[0].amount
        )

    def test_estimates_are_labelled_and_explained(self):
        summary = calculate_budget(
            request_for(), transport=transport_offer(), accommodation=stay_offer()
        )
        estimates = [line for line in summary.lines if line.is_estimate]
        quotes = [line for line in summary.lines if not line.is_estimate]
        assert {line.category for line in estimates} == {
            BudgetCategory.FOOD,
            BudgetCategory.LOCAL_TRANSPORT,
        }
        assert all(line.basis for line in estimates + quotes), "every line must explain itself"

    def test_estimates_scale_with_travellers_and_days(self):
        one = calculate_budget(
            request_for(travellers=1), transport=None, accommodation=None, include_estimates=True
        )
        four = calculate_budget(
            request_for(travellers=4), transport=None, accommodation=None, include_estimates=True
        )
        assert four.amount_for(BudgetCategory.FOOD) == one.amount_for(BudgetCategory.FOOD).times(4)

    def test_allowances_are_tiered_by_accommodation_preference(self):
        hostel = default_allowances(AccommodationType.HOSTEL)
        hotel = default_allowances(AccommodationType.HOTEL)
        assert hotel.food_per_day > hostel.food_per_day

    def test_over_budget_is_detected_and_quantified(self):
        summary = calculate_budget(
            request_for(max_budget=Money.of(100, EUR)),
            transport=transport_offer("60"),
            accommodation=stay_offer("30"),
        )
        assert summary.status is BudgetStatus.OVER_BUDGET
        assert summary.overspend.amount > 0
        assert summary.remaining == Money.zero(EUR)
        assert any("exceeds" in note for note in summary.notes)

    def test_missing_transport_makes_the_budget_incomplete(self):
        """A total that silently omits transport must not be presented as final."""
        summary = calculate_budget(request_for(), transport=None, accommodation=stay_offer())
        assert summary.status is BudgetStatus.INCOMPLETE
        assert BudgetCategory.TRANSPORT in summary.missing_components
        assert summary.is_within_budget is False

    def test_missing_accommodation_makes_the_budget_incomplete(self):
        summary = calculate_budget(request_for(), transport=transport_offer(), accommodation=None)
        assert summary.status is BudgetStatus.INCOMPLETE
        assert BudgetCategory.ACCOMMODATION in summary.missing_components

    def test_day_trip_needs_no_accommodation_and_stays_complete(self):
        summary = calculate_budget(
            request_for(return_date=DEPART), transport=transport_offer(), accommodation=None
        )
        assert summary.status is not BudgetStatus.INCOMPLETE
        assert BudgetCategory.ACCOMMODATION not in summary.missing_components
        assert any("Day trip" in note for note in summary.notes)

    def test_currency_mismatch_raises_rather_than_converting(self):
        gbp = transport_offer().model_copy(
            update={
                "total_price": Money.of(60, Currency.GBP),
                "price_per_traveller": Money.of(60, Currency.GBP),
            }
        )
        with pytest.raises(ValueError, match="does not convert currencies"):
            calculate_budget(request_for(), transport=gbp, accommodation=stay_offer())

    def test_produces_a_valid_summary_when_nothing_can_be_priced(self):
        summary = calculate_budget(
            request_for(), transport=None, accommodation=None, include_estimates=False
        )
        assert summary.status is BudgetStatus.INCOMPLETE
        assert summary.total == Money.zero(EUR)

    def test_exclude_estimates_omits_only_the_heuristic_lines(self):
        summary = calculate_budget(
            request_for(),
            transport=transport_offer("60"),
            accommodation=stay_offer("30"),
            include_estimates=False,
        )
        assert summary.total == Money.of(150, EUR)
        assert not any(line.is_estimate for line in summary.lines)

    def test_budget_is_reproducible(self):
        args = {"transport": transport_offer(), "accommodation": stay_offer()}
        first = calculate_budget(request_for(), **args)
        for _ in range(10):
            assert calculate_budget(request_for(), **args).total == first.total


class TestRemainingAllowance:
    def test_deducts_transport_and_estimates(self):
        allowance = remaining_accommodation_allowance(
            request_for(), transport=transport_offer("60")
        )
        assert allowance.amount < Decimal("350")
        assert allowance.amount > 0

    def test_floors_at_zero_rather_than_going_negative(self):
        allowance = remaining_accommodation_allowance(
            request_for(max_budget=Money.of(20, EUR)), transport=transport_offer("500")
        )
        assert allowance == Money.zero(EUR)

    def test_larger_party_leaves_less_for_accommodation(self):
        one = remaining_accommodation_allowance(
            request_for(travellers=1), transport=transport_offer("60")
        )
        four = remaining_accommodation_allowance(
            request_for(travellers=4), transport=transport_offer("60")
        )
        assert four.amount < one.amount


class TestItineraryPlanner:
    def _attractions(self, count: int = 6) -> list[Attraction]:
        return [
            Attraction(
                attraction_id=f"poi-{index}",
                name=f"Attraction {index}",
                category=AttractionCategory.HISTORIC,
                location=GeoPoint(
                    latitude=PRAGUE.latitude + index * 0.004, longitude=PRAGUE.longitude
                ),
                typical_visit_minutes=60,
                is_outdoor=index % 2 == 0,
                accessibility_is_known=True,
                accessibility_features=[AccessibilityNeed.STEP_FREE_ACCESS],
                interest_tags=["history"],
                provenance=Provenance.fixture("test"),
            )
            for index in range(count)
        ]

    def _weather(self, *, wet_day: date | None = None) -> WeatherSummary:
        days = []
        current = DEPART
        while current <= RETURN:
            wet = current == wet_day
            days.append(
                WeatherDay(
                    forecast_date=current,
                    temperature_min_c=14.0,
                    temperature_max_c=24.0,
                    precipitation_mm=20.0 if wet else 0.0,
                    condition="heavy rain" if wet else "clear",
                )
            )
            current += timedelta(days=1)
        return WeatherSummary(location="Prague", days=days, provenance=Provenance.mocked("test"))

    def test_produces_one_day_per_trip_day(self):
        itinerary = ItineraryPlanner().plan(request_for(), attractions=self._attractions())
        assert itinerary.day_count == 4

    def test_no_day_contains_overlapping_slots(self):
        """The ItineraryDay validator enforces this, so construction succeeding is the
        assertion — but check explicitly so the guarantee is visible in the test names."""
        itinerary = ItineraryPlanner().plan(
            request_for(),
            attractions=self._attractions(10),
            accommodation=stay_offer(),
            outbound=transport_offer(),
        )
        for day in itinerary.days:
            for earlier, later in pairwise(day.slots):
                assert not earlier.overlaps(later), f"day {day.day_number}: {earlier} / {later}"

    def test_check_in_does_not_collide_with_sightseeing(self):
        """Regression: a fixed 15:00 check-in used to be appended independently of the
        attraction cursor and could land inside an already-scheduled visit."""
        itinerary = ItineraryPlanner().plan(
            request_for(),
            attractions=self._attractions(10),
            accommodation=stay_offer(),
            outbound=transport_offer(arrive_hour=10),
        )
        day_one = itinerary.days[0]
        check_ins = [s for s in day_one.slots if s.kind is SlotKind.CHECK_IN]
        assert len(check_ins) == 1
        for slot in day_one.slots:
            if slot is not check_ins[0]:
                assert not slot.overlaps(check_ins[0])

    def test_meals_are_scheduled(self):
        itinerary = ItineraryPlanner().plan(
            request_for(), attractions=self._attractions(), accommodation=stay_offer()
        )
        assert any(day.has_meal_break for day in itinerary.days)

    def test_no_meals_after_departure(self):
        """Regression: lunch and dinner were scheduled on the final day after an 11:45
        departure, feeding a traveller already on the coach home."""
        inbound = transport_offer().model_copy(
            update={
                "legs": [
                    TransportLeg(
                        mode=TransportMode.TRAIN,
                        carrier="Test Rail",
                        from_location="B",
                        to_location="A",
                        departure_at=datetime(2026, 8, 13, 11, 45, tzinfo=UTC),
                        arrival_at=datetime(2026, 8, 13, 15, 45, tzinfo=UTC),
                    )
                ]
            }
        )
        itinerary = ItineraryPlanner().plan(
            request_for(),
            attractions=self._attractions(),
            accommodation=stay_offer(),
            inbound=inbound,
        )
        final = itinerary.days[-1]
        departures = [s for s in final.slots if s.kind is SlotKind.DEPARTURE]
        assert departures, "a departure slot should exist on the final day"
        for slot in final.slots:
            if slot.kind is SlotKind.MEAL:
                assert slot.start_minute < departures[0].start_minute

    def test_travel_time_is_reserved_between_attractions(self):
        itinerary = ItineraryPlanner().plan(
            request_for(), attractions=self._attractions(8), accommodation=stay_offer()
        )
        for day in itinerary.days:
            visits = [s for s in day.slots if s.kind is SlotKind.ATTRACTION]
            for earlier, later in pairwise(visits):
                assert later.start_minute > earlier.end_minute, "no time allowed for travel"

    def test_attractions_are_never_scheduled_twice(self):
        itinerary = ItineraryPlanner().plan(
            request_for(), attractions=self._attractions(8), accommodation=stay_offer()
        )
        ids = [
            slot.attraction_id for day in itinerary.days for slot in day.slots if slot.attraction_id
        ]
        assert len(ids) == len(set(ids))

    def test_wet_day_prefers_indoor_and_warns_about_outdoor(self):
        itinerary = ItineraryPlanner().plan(
            request_for(),
            attractions=self._attractions(8),
            weather=self._weather(wet_day=DEPART + timedelta(days=1)),
            accommodation=stay_offer(),
        )
        wet_day = itinerary.days[1]
        for slot in wet_day.slots:
            if slot.kind is SlotKind.ATTRACTION and slot.is_outdoor:
                assert slot.weather_warning is not None

    def test_inaccessible_attractions_are_excluded_and_reported(self):
        attractions = self._attractions(4)
        attractions.append(
            Attraction(
                attraction_id="poi-unknown",
                name="Unknown Accessibility Site",
                category=AttractionCategory.HISTORIC,
                location=PRAGUE,
                provenance=Provenance.fixture("test"),
                accessibility_is_known=False,
            )
        )
        itinerary = ItineraryPlanner().plan(
            request_for(accessibility_needs=[AccessibilityNeed.STEP_FREE_ACCESS]),
            attractions=attractions,
        )
        scheduled = {
            slot.attraction_id for day in itinerary.days for slot in day.slots if slot.attraction_id
        }
        # Unknown accessibility is excluded, not optimistically assumed suitable.
        assert "poi-unknown" not in scheduled
        assert any("accessibility" in warning.lower() for warning in itinerary.warnings)

    def test_no_attractions_yields_a_valid_itinerary_with_a_warning(self):
        itinerary = ItineraryPlanner().plan(request_for(), attractions=[])
        assert itinerary.day_count == 4
        assert any("No attractions" in warning for warning in itinerary.warnings)

    def test_unscheduled_attractions_are_offered_not_discarded(self):
        itinerary = ItineraryPlanner().plan(
            request_for(return_date=DEPART), attractions=self._attractions(20)
        )
        assert itinerary.unscheduled_suggestions

    def test_planning_is_reproducible(self):
        args = {"attractions": self._attractions(8), "accommodation": stay_offer()}

        def fingerprint():
            plan = ItineraryPlanner().plan(request_for(), **args)
            return [
                (day.day_number, slot.title, slot.start_minute)
                for day in plan.days
                for slot in day.slots
            ]

        first = fingerprint()
        for _ in range(5):
            assert fingerprint() == first

    def test_fixture_timings_are_flagged_uncertain(self):
        """The fixture corpus omits opening hours, so no timing may be presented as fact."""
        attractions = asyncio.run(FixturePOIProvider().search(POIQuery(destination="Prague"))).items
        itinerary = ItineraryPlanner().plan(
            request_for(), attractions=attractions, accommodation=stay_offer()
        )
        visits = [
            slot for day in itinerary.days for slot in day.slots if slot.kind is SlotKind.ATTRACTION
        ]
        assert visits
        assert all(slot.is_uncertain for slot in visits)


class TestFixtureProviders:
    def test_poi_provider_returns_curated_real_places(self):
        result = asyncio.run(FixturePOIProvider().search(POIQuery(destination="Prague")))
        assert result.origin.value == "fixture"
        assert any(a.name == "Charles Bridge" for a in result.items)

    def test_unknown_city_returns_unavailable_not_invented_data(self):
        result = asyncio.run(FixturePOIProvider().search(POIQuery(destination="Atlantis")))
        assert result.origin.value == "unavailable"
        assert result.items == []
        assert result.error is not None

    def test_interest_filter_falls_back_rather_than_returning_nothing(self):
        result = asyncio.run(
            FixturePOIProvider().search(
                POIQuery(destination="Prague", interests=("underwater basket weaving",))
            )
        )
        assert result.items, "a niche interest should still surface the highlights"

    def test_fixture_weather_is_labelled_mocked_with_a_note(self):
        """A synthesised forecast must never be presented as a real one."""
        result = asyncio.run(
            FixtureWeatherProvider().forecast(
                WeatherQuery(location=PRAGUE, start_date=DEPART, end_date=RETURN)
            )
        )
        summary = result.items[0]
        assert summary.provenance.origin.value == "mocked"
        assert summary.note is not None and "not a forecast" in summary.note
        assert summary.is_complete is False

    def test_fixture_weather_is_deterministic(self):
        query = WeatherQuery(location=PRAGUE, start_date=DEPART, end_date=RETURN)
        first = asyncio.run(FixtureWeatherProvider().forecast(query)).items[0]
        second = asyncio.run(FixtureWeatherProvider().forecast(query)).items[0]
        assert [d.temperature_max_c for d in first.days] == [
            d.temperature_max_c for d in second.days
        ]

    def test_weather_covers_every_requested_day(self):
        result = asyncio.run(
            FixtureWeatherProvider().forecast(
                WeatherQuery(location=PRAGUE, start_date=DEPART, end_date=RETURN)
            )
        )
        assert len(result.items[0].days) == 4

    def test_reversed_dates_are_reported_not_silently_accepted(self):
        result = asyncio.run(
            FixtureWeatherProvider().forecast(
                WeatherQuery(location=PRAGUE, start_date=RETURN, end_date=DEPART)
            )
        )
        assert result.ok is False
        assert result.items == []
