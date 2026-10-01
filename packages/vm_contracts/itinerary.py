"""Day-by-day itinerary.

Slots use minute-of-day integers rather than ``datetime`` objects. The itinerary is
inherently local-time and date-relative, and integers make overlap detection exact and
timezone-free — the alternative invites a whole class of DST and offset bugs for no gain.

Overlap prevention is enforced *in the model*, not merely checked afterwards: an
``ItineraryDay`` containing overlapping slots cannot be constructed at all.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Self

from pydantic import Field, computed_field, model_validator

from vm_contracts.common import Money, StrictModel

__all__ = [
    "ActivitySlot",
    "Itinerary",
    "ItineraryDay",
    "SlotKind",
    "format_minute_of_day",
]


def format_minute_of_day(minute: int) -> str:
    """``545`` → ``"09:05"``."""
    return f"{minute // 60:02d}:{minute % 60:02d}"


class SlotKind(StrEnum):
    ATTRACTION = "attraction"
    MEAL = "meal"
    REST = "rest"
    TRANSIT = "transit"
    CHECK_IN = "check_in"
    CHECK_OUT = "check_out"
    FREE_TIME = "free_time"
    DEPARTURE = "departure"
    ARRIVAL = "arrival"


class ActivitySlot(StrictModel):
    """One scheduled block within a day."""

    kind: SlotKind
    title: Annotated[str, Field(min_length=1, max_length=200)]
    start_minute: Annotated[int, Field(ge=0, le=1439)]
    end_minute: Annotated[int, Field(ge=1, le=1440)]

    attraction_id: Annotated[str | None, Field(max_length=120)] = None
    description: Annotated[str | None, Field(max_length=600)] = None
    estimated_cost: Money | None = None
    cost_is_unknown: bool = False
    """An unpriced admission is not a free activity and leaves the budget incomplete."""
    travel_minutes_to_next: Annotated[int, Field(ge=0, le=480)] = 0
    """Transit time from this slot to the next. The day validator requires the following
    slot to start at least this many minutes later — this is what makes the schedule
    physically feasible rather than merely non-overlapping."""

    is_outdoor: bool = False
    weather_warning: Annotated[str | None, Field(max_length=300)] = None
    accessibility_note: Annotated[str | None, Field(max_length=300)] = None
    is_uncertain: bool = False
    """True when timing rests on estimated opening hours. Rendered with a caveat."""

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.end_minute <= self.start_minute:
            raise ValueError(
                f"slot '{self.title}' ends at {format_minute_of_day(self.end_minute)} which is "
                f"not after its start {format_minute_of_day(self.start_minute)}"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_minutes(self) -> int:
        return self.end_minute - self.start_minute

    @computed_field  # type: ignore[prop-decorator]
    @property
    def display_time(self) -> str:
        return f"{format_minute_of_day(self.start_minute)}–{format_minute_of_day(self.end_minute)}"

    def overlaps(self, other: ActivitySlot) -> bool:
        """Half-open interval overlap: a slot ending at 12:00 and one starting at 12:00
        do not overlap."""
        return self.start_minute < other.end_minute and other.start_minute < self.end_minute


class ItineraryDay(StrictModel):
    """A single day of the trip."""

    day_number: Annotated[int, Field(ge=1, le=31)]
    day_date: date
    summary: Annotated[str, Field(min_length=1, max_length=400)]
    slots: Annotated[list[ActivitySlot], Field(max_length=20)] = Field(default_factory=list)
    weather_note: Annotated[str | None, Field(max_length=300)] = None
    total_walking_km: Annotated[float | None, Field(ge=0, le=100)] = None

    @model_validator(mode="after")
    def _validate_schedule(self) -> Self:
        ordered = sorted(self.slots, key=lambda s: s.start_minute)
        if [s.start_minute for s in self.slots] != [s.start_minute for s in ordered]:
            raise ValueError(f"day {self.day_number}: slots must be in chronological order")

        for earlier, later in pairwise(ordered):
            if earlier.overlaps(later):
                raise ValueError(
                    f"day {self.day_number}: '{earlier.title}' ({earlier.display_time}) overlaps "
                    f"'{later.title}' ({later.display_time})"
                )
            gap = later.start_minute - earlier.end_minute
            if gap < earlier.travel_minutes_to_next:
                raise ValueError(
                    f"day {self.day_number}: only {gap} min between '{earlier.title}' and "
                    f"'{later.title}', but {earlier.travel_minutes_to_next} min of travel is "
                    f"required — this schedule is not physically possible"
                )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_meal_break(self) -> bool:
        return any(s.kind is SlotKind.MEAL for s in self.slots)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def scheduled_minutes(self) -> int:
        return sum(s.duration_minutes for s in self.slots)

    def estimated_cost(self) -> Money | None:
        """Sum of slot costs, or ``None`` when nothing in the day is priced."""
        costs = [s.estimated_cost for s in self.slots if s.estimated_cost is not None]
        if not costs:
            return None
        total = costs[0]
        for extra in costs[1:]:
            total = total + extra  # Money.__add__ enforces a single currency
        return total


class Itinerary(StrictModel):
    """The complete day-by-day plan."""

    days: Annotated[list[ItineraryDay], Field(max_length=31)] = Field(default_factory=list)
    overview: Annotated[str, Field(max_length=1500)] = ""
    unscheduled_suggestions: Annotated[list[str], Field(max_length=15)] = Field(
        default_factory=list
    )
    """Attractions that fitted the interests but not the schedule. Offering these is more
    useful than silently discarding them."""

    warnings: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        numbers = [d.day_number for d in self.days]
        if numbers != sorted(numbers):
            raise ValueError("itinerary days must be in ascending day_number order")
        if len(numbers) != len(set(numbers)):
            raise ValueError("duplicate day_number in itinerary")
        dates = [d.day_date for d in self.days]
        if len(dates) != len(set(dates)):
            raise ValueError("duplicate day_date in itinerary")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def day_count(self) -> int:
        return len(self.days)

    def estimated_activity_cost(self) -> Money | None:
        totals = [c for c in (d.estimated_cost() for d in self.days) if c is not None]
        if not totals:
            return None
        total = totals[0]
        for extra in totals[1:]:
            total = total + extra
        return total
