"""The trip request — the single input to the whole system.

Two forms exist deliberately. :class:`TripRequest` is what a user submits: lenient about
casing and spacing, validated for internal consistency. :class:`NormalizedTripRequest` is
what the workflow operates on: canonical, with derived values (nights, trip length)
computed once so that no downstream node recomputes them differently.

The normalised form is also what the cache key hashes. Normalising *before* hashing is
what makes "Prague" and "  prague " the same cache entry.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Self

from pydantic import Field, computed_field, field_validator, model_validator

from vm_contracts.common import (
    AccessibilityNeed,
    AccommodationType,
    Currency,
    Money,
    RankingStrategy,
    StrictModel,
    TransportMode,
)

__all__ = ["NormalizedTripRequest", "TripRequest"]

MAX_INTERESTS = 12
MAX_TRIP_NIGHTS = 30
MAX_TRAVELLERS = 12

# A place name is letters, single spaces, and the small set of punctuation that occurs in
# real toponyms (Baden-Baden, 's-Hertogenbosch, Saint-Étienne, Frankfurt (Oder)).
#
# The whitespace class is a literal space, NOT `\s`: `\s` matches newlines, which would
# have admitted "Prague\nIgnore previous instructions" — a multi-line injection payload
# wearing a valid place name as a prefix. Horizontal space is all a toponym needs.
#
# The typographic apostrophe U+2019 is accepted because real names and normalised input
# both use it (N'Djamena, 's-Hertogenbosch).
# The first character must be a letter or an apostrophe — the latter because
# 's-Hertogenbosch is a real city. Anchoring it this way is what rejects a leading dot or
# slash ("../../etc/passwd") while leaving genuine names alone.
_PLACE_PATTERN = r"^(?:[^\W\d_]|['’])[\w \-'’.,()/]{1,79}$"

# Rejected outright regardless of the pattern above. `/` and `.` are legitimate in names
# like "Frankfurt/Main" and "St. Gallen", but a traversal-shaped sequence never is.
_PLACE_FORBIDDEN_SEQUENCES = ("..", "//", "./", "/.")


class TripRequest(StrictModel):
    """A user's trip-planning request, as submitted."""

    origin: Annotated[str, Field(min_length=2, max_length=80, pattern=_PLACE_PATTERN)]
    destination: Annotated[str, Field(min_length=2, max_length=80, pattern=_PLACE_PATTERN)]

    departure_date: date
    return_date: date

    travellers: Annotated[int, Field(ge=1, le=MAX_TRAVELLERS)] = 1
    guest_nationality: Annotated[str | None, Field(pattern=r"^[A-Z]{2}$")] = None
    """ISO country code required for live hotel rates. Never inferred from destination."""

    max_budget: Money
    """Total budget for the whole trip and all travellers — not per person, not per night.
    Ambiguity here is the most expensive kind of misunderstanding in this domain, so the
    field name and this docstring are explicit."""

    accommodation_preference: AccommodationType = AccommodationType.ANY
    transport_preference: TransportMode = TransportMode.ANY

    interests: Annotated[list[str], Field(max_length=MAX_INTERESTS)] = Field(default_factory=list)

    max_transport_duration_hours: Annotated[float, Field(gt=0, le=48)] = 12.0
    max_transfers: Annotated[int, Field(ge=0, le=5)] = 2
    accessibility_needs: list[AccessibilityNeed] = Field(default_factory=list)
    ranking_strategy: RankingStrategy = RankingStrategy.BALANCED

    notes: Annotated[str | None, Field(max_length=500)] = None
    """Free-text. Passed through prompt-injection screening; never used to build a URL,
    a query, or a path."""

    @field_validator("origin", "destination")
    @classmethod
    def _normalise_place(cls, value: str) -> str:
        for sequence in _PLACE_FORBIDDEN_SEQUENCES:
            if sequence in value:
                raise ValueError(
                    f"place name contains the sequence '{sequence}', which no real toponym "
                    f"uses: {value[:40]}"
                )
        return " ".join(value.split())

    @field_validator("interests")
    @classmethod
    def _clean_interests(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in values:
            item = " ".join(raw.split()).lower()
            if not item:
                continue
            if len(item) > 60:
                raise ValueError(f"interest too long ({len(item)} chars, max 60): {item[:30]}…")
            if item not in seen:
                seen.add(item)
                cleaned.append(item)
        return cleaned

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        if self.return_date < self.departure_date:
            raise ValueError(
                f"return_date ({self.return_date}) must not precede departure_date "
                f"({self.departure_date})"
            )
        nights = (self.return_date - self.departure_date).days
        if nights > MAX_TRIP_NIGHTS:
            raise ValueError(
                f"trip length {nights} nights exceeds the {MAX_TRIP_NIGHTS}-night maximum"
            )
        if self.max_budget.amount <= 0:
            raise ValueError("max_budget must be greater than zero")
        if self.origin.casefold() == self.destination.casefold():
            raise ValueError(
                f"origin and destination are the same place ({self.origin}); "
                f"there is no journey to plan"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def nights(self) -> int:
        """Nights of accommodation required. A same-day return needs zero."""
        return (self.return_date - self.departure_date).days

    @computed_field  # type: ignore[prop-decorator]
    @property
    def days(self) -> int:
        """Calendar days the itinerary must cover, inclusive of both endpoints."""
        return self.nights + 1

    @property
    def currency(self) -> Currency:
        return self.max_budget.currency

    def is_day_trip(self) -> bool:
        return self.nights == 0


class NormalizedTripRequest(StrictModel):
    """Canonical form of a :class:`TripRequest`.

    Produced once by the orchestrator's ``normalize_trip_request`` node. Everything
    downstream — agents, cache key, persistence — uses this, so there is exactly one
    interpretation of the user's intent in the system.
    """

    origin: Annotated[str, Field(min_length=2, max_length=80)]
    destination: Annotated[str, Field(min_length=2, max_length=80)]
    origin_normalized: Annotated[str, Field(min_length=2, max_length=80)]
    """Casefolded, whitespace-collapsed. This is the form that enters the cache key."""

    destination_normalized: Annotated[str, Field(min_length=2, max_length=80)]

    departure_date: date
    return_date: date
    nights: Annotated[int, Field(ge=0, le=MAX_TRIP_NIGHTS)]
    days: Annotated[int, Field(ge=1)]

    travellers: Annotated[int, Field(ge=1, le=MAX_TRAVELLERS)]
    guest_nationality: Annotated[str | None, Field(pattern=r"^[A-Z]{2}$")] = None
    max_budget: Money
    currency: Currency

    accommodation_preference: AccommodationType
    transport_preference: TransportMode
    interests: list[str]
    max_transport_duration_hours: Annotated[float, Field(gt=0, le=48)]
    max_transfers: Annotated[int, Field(ge=0, le=5)]
    accessibility_needs: list[AccessibilityNeed]
    ranking_strategy: RankingStrategy

    budget_bucket: Annotated[str, Field(min_length=1, max_length=20)]
    """Coarse budget band (e.g. ``"300-400"``).

    The cache key uses the bucket rather than the exact amount: a €349 and a €351 request
    for the same route on the same dates want the same set of options, and keying on the
    exact figure would make the cache almost never hit.
    """

    @classmethod
    def from_request(cls, request: TripRequest, *, bucket_size: int = 100) -> NormalizedTripRequest:
        """Derive the canonical form. Pure — no I/O, fully deterministic."""
        amount = int(request.max_budget.amount)
        lower = (amount // bucket_size) * bucket_size
        return cls(
            origin=request.origin,
            destination=request.destination,
            origin_normalized=request.origin.casefold(),
            destination_normalized=request.destination.casefold(),
            departure_date=request.departure_date,
            return_date=request.return_date,
            nights=request.nights,
            days=request.days,
            travellers=request.travellers,
            guest_nationality=request.guest_nationality,
            max_budget=request.max_budget,
            currency=request.max_budget.currency,
            accommodation_preference=request.accommodation_preference,
            transport_preference=request.transport_preference,
            # Sorted so that ["food", "history"] and ["history", "food"] — the same request
            # expressed in a different order — produce the same cache key.
            interests=sorted(request.interests),
            max_transport_duration_hours=request.max_transport_duration_hours,
            max_transfers=request.max_transfers,
            accessibility_needs=sorted(request.accessibility_needs, key=lambda n: n.value),
            ranking_strategy=request.ranking_strategy,
            budget_bucket=f"{lower}-{lower + bucket_size}",
        )

    def cache_fingerprint(self) -> dict[str, object]:
        """The exact fields that determine cache identity.

        Kept as an explicit dict rather than dumping the whole model, so that adding a
        non-semantic field later cannot silently invalidate every cache entry.
        """
        return {
            "origin": self.origin_normalized,
            "destination": self.destination_normalized,
            "departure": self.departure_date.isoformat(),
            "return": self.return_date.isoformat(),
            "travellers": self.travellers,
            "guest_nationality": self.guest_nationality,
            "budget_bucket": self.budget_bucket,
            "currency": self.currency.value,
            "ranking": self.ranking_strategy.value,
            "interests": self.interests,
            "accessibility": [n.value for n in self.accessibility_needs],
            "accommodation": self.accommodation_preference.value,
            "transport": self.transport_preference.value,
            "max_duration_h": self.max_transport_duration_hours,
            "max_transfers": self.max_transfers,
        }
