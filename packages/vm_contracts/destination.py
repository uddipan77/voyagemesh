"""Destination knowledge: attractions, weather, and retrieved reference documents."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, computed_field, model_validator

from vm_contracts.common import (
    AccessibilityNeed,
    GeoPoint,
    Money,
    Provenance,
    StrictModel,
)

__all__ = [
    "Attraction",
    "AttractionCategory",
    "OpeningHours",
    "RetrievedDocument",
    "WeatherDay",
    "WeatherSummary",
]


class AttractionCategory(StrEnum):
    HISTORIC = "historic"
    ARCHITECTURE = "architecture"
    MUSEUM = "museum"
    GALLERY = "gallery"
    PARK = "park"
    VIEWPOINT = "viewpoint"
    RELIGIOUS = "religious"
    FOOD = "food"
    MARKET = "market"
    NIGHTLIFE = "nightlife"
    SHOPPING = "shopping"
    NATURE = "nature"
    ENTERTAINMENT = "entertainment"
    OTHER = "other"


class OpeningHours(StrictModel):
    """Opening times for one weekday.

    ``is_estimated`` exists because opening hours are the single most commonly hallucinated
    travel fact. When a provider does not publish hours we mark the record estimated and
    the UI renders it with a caveat rather than as fact — we never guess a plausible time
    and present it plainly.
    """

    weekday: Annotated[int, Field(ge=0, le=6)]
    """0 = Monday, 6 = Sunday."""

    opens_minute_of_day: Annotated[int | None, Field(ge=0, le=1439)] = None
    closes_minute_of_day: Annotated[int | None, Field(ge=0, le=1440)] = None
    closed: bool = False
    is_estimated: bool = True

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.closed:
            return self
        if self.opens_minute_of_day is None or self.closes_minute_of_day is None:
            raise ValueError("opening hours must specify both open and close, or be marked closed")
        if self.closes_minute_of_day <= self.opens_minute_of_day:
            raise ValueError(
                f"closing time ({self.closes_minute_of_day}) must be after opening "
                f"({self.opens_minute_of_day})"
            )
        return self


class Attraction(StrictModel):
    """A point of interest at the destination."""

    attraction_id: Annotated[str, Field(min_length=1, max_length=120)]
    name: Annotated[str, Field(min_length=1, max_length=200)]
    category: AttractionCategory
    location: GeoPoint | None = None
    description: Annotated[str | None, Field(max_length=1000)] = None

    typical_visit_minutes: Annotated[int, Field(ge=5, le=600)] = 60
    admission_price: Money | None = None
    is_free: bool = False
    is_outdoor: bool = False
    """Drives weather suitability — an outdoor site on a heavy-rain day is flagged."""

    accessibility_features: list[AccessibilityNeed] = Field(default_factory=list)
    accessibility_is_known: bool = False
    """False means we have no accessibility information — which is *not* the same as
    'not accessible'. The UI must distinguish 'unknown' from 'unsuitable'."""

    opening_hours: list[OpeningHours] = Field(default_factory=list)
    interest_tags: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)
    provenance: Provenance
    source_url: Annotated[str | None, Field(max_length=500)] = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.is_free and self.admission_price is not None:
            raise ValueError("an attraction marked free must not carry an admission price")
        weekdays = [h.weekday for h in self.opening_hours]
        if len(weekdays) != len(set(weekdays)):
            raise ValueError("duplicate weekday entries in opening_hours")
        return self

    def matches_interests(self, interests: list[str]) -> bool:
        """True when the attraction relates to any stated interest.

        An empty interest list matches everything: a user who stated no preferences
        should see the destination's highlights, not nothing.
        """
        if not interests:
            return True
        haystack = {self.category.value, *(t.lower() for t in self.interest_tags)}
        return any(
            interest in haystack or any(interest in tag for tag in haystack)
            for interest in (i.lower() for i in interests)
        )

    def is_accessible_for(self, needs: list[AccessibilityNeed]) -> bool | None:
        """Tri-state: ``True`` suitable, ``False`` unsuitable, ``None`` unknown."""
        if not needs:
            return True
        if not self.accessibility_is_known:
            return None
        return all(need in self.accessibility_features for need in needs)


class WeatherDay(StrictModel):
    """Forecast for a single day at the destination."""

    forecast_date: date
    temperature_min_c: Annotated[float, Field(ge=-90.0, le=60.0)]
    temperature_max_c: Annotated[float, Field(ge=-90.0, le=60.0)]
    precipitation_mm: Annotated[float, Field(ge=0.0, le=1000.0)]
    precipitation_probability: Annotated[float | None, Field(ge=0.0, le=100.0)] = None
    wind_speed_kmh: Annotated[float | None, Field(ge=0.0, le=500.0)] = None
    condition: Annotated[str, Field(min_length=1, max_length=80)]

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.temperature_max_c < self.temperature_min_c:
            raise ValueError(
                f"max temperature ({self.temperature_max_c}) is below min "
                f"({self.temperature_min_c})"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_outdoor_friendly(self) -> bool:
        """Whether outdoor activities are reasonable.

        Thresholds are deliberate and documented rather than model-judged: >5 mm of rain,
        below 0 °C, or above 35 °C makes a full outdoor day unpleasant for most people.
        """
        return (
            self.precipitation_mm <= 5.0
            and self.temperature_max_c >= 0.0
            and self.temperature_max_c <= 35.0
        )


class WeatherSummary(StrictModel):
    """Forecast covering the trip window."""

    location: Annotated[str, Field(min_length=1, max_length=120)]
    coordinates: GeoPoint | None = None
    days: Annotated[list[WeatherDay], Field(max_length=31)] = Field(default_factory=list)
    provenance: Provenance
    note: Annotated[str | None, Field(max_length=300)] = None
    """Set when the forecast is incomplete — e.g. a departure beyond the 16-day horizon."""

    @model_validator(mode="after")
    def _validate(self) -> Self:
        dates = [d.forecast_date for d in self.days]
        if len(dates) != len(set(dates)):
            raise ValueError("duplicate forecast dates")
        if dates != sorted(dates):
            raise ValueError("forecast days must be in chronological order")
        return self

    def for_date(self, day: date) -> WeatherDay | None:
        return next((d for d in self.days if d.forecast_date == day), None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_complete(self) -> bool:
        return bool(self.days) and self.note is None


class RetrievedDocument(StrictModel):
    """A chunk retrieved from the pgvector RAG corpus.

    Content is **untrusted input**. It is inserted into prompts inside an explicit
    untrusted-data delimiter, and any instruction it contains must be ignored — see
    ``vm_guardrails.rag``.
    """

    document_id: Annotated[str, Field(min_length=1, max_length=120)]
    chunk_id: Annotated[str, Field(min_length=1, max_length=120)]
    title: Annotated[str, Field(min_length=1, max_length=200)]
    content: Annotated[str, Field(min_length=1, max_length=8000)]
    similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    citation_id: Annotated[str, Field(min_length=1, max_length=40)]
    """Short handle (``[D3]``) the narrative uses to cite this source."""

    provenance: Provenance
    tags: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)
