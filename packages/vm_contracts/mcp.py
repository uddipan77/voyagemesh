"""Shared MCP tool contracts.

Tool argument and result schemas live here, in the contracts package, rather than beside
each server. That is what makes a contract test meaningful: the agent and the server both
import *these* models, so a change that would break the pair fails at import or validation
time instead of at runtime in another container (brief §8).

Every tool in VoyageMesh is **read-only**. There is no write, delete, or execute tool
anywhere, which is the simplest possible answer to "what can a hijacked model do with
these?" — nothing but look things up.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any

from pydantic import Field, model_validator

from vm_contracts.common import (
    AccessibilityNeed,
    AccommodationType,
    Currency,
    GeoPoint,
    RankingStrategy,
    StrictModel,
    TransportMode,
)

__all__ = [
    "MCP_PROTOCOL_VERSION",
    "AccommodationSearchArgs",
    "DistanceArgs",
    "GeocodeArgs",
    "KnowledgeArgs",
    "POISearchArgs",
    "ScoreTransportArgs",
    "StayCostArgs",
    "ToolError",
    "ToolResult",
    "TransportSearchArgs",
    "ValidateScheduleArgs",
    "WeatherArgs",
]

MCP_PROTOCOL_VERSION = "1.0"

# Shared field bounds. Defined once so the agent, the server, and the tests cannot drift
# apart on what "too many" means.
MAX_TRAVELLERS = 12
MAX_RESULTS = 25
MAX_NIGHTS = 30


class ToolError(StrictModel):
    """A sanitised tool failure.

    Tools return this rather than raising, so an agent's ReAct loop sees a structured
    result it can reason about instead of an exception it must catch. The message never
    contains a stack trace, an internal hostname, or a provider's raw error body.
    """

    code: Annotated[str, Field(min_length=1, max_length=60)]
    message: Annotated[str, Field(min_length=1, max_length=400)]
    retryable: bool = False


class ToolResult(StrictModel):
    """Envelope returned by every MCP tool.

    Carrying ``origin`` and ``source_name`` on the envelope — not just inside the payload —
    means an agent can label data correctly without having to understand the shape of every
    tool's result.
    """

    ok: bool
    origin: Annotated[str, Field(min_length=1, max_length=20)]
    source_name: Annotated[str, Field(min_length=1, max_length=120)]
    data: dict[str, Any] = Field(default_factory=dict)
    warnings: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)
    error: ToolError | None = None

    @model_validator(mode="after")
    def _validate(self) -> ToolResult:
        if not self.ok and self.error is None:
            raise ValueError("a failed tool result must carry an error")
        if self.ok and self.error is not None:
            raise ValueError("a successful tool result must not carry an error")
        return self

    @classmethod
    def success(
        cls,
        *,
        origin: str,
        source_name: str,
        data: dict[str, Any],
        warnings: list[str] | None = None,
    ) -> ToolResult:
        return cls(
            ok=True,
            origin=origin,
            source_name=source_name,
            data=data,
            warnings=warnings or [],
        )

    @classmethod
    def failure(
        cls,
        *,
        code: str,
        message: str,
        source_name: str = "voyagemesh",
        retryable: bool = False,
    ) -> ToolResult:
        return cls(
            ok=False,
            origin="unavailable",
            source_name=source_name,
            error=ToolError(code=code, message=message, retryable=retryable),
        )


# ---------------------------------------------------------------------------
# Transport MCP
# ---------------------------------------------------------------------------
class TransportSearchArgs(StrictModel):
    """Arguments for ``search_ground_transport`` and ``search_flights``."""

    origin: Annotated[str, Field(min_length=2, max_length=80)]
    destination: Annotated[str, Field(min_length=2, max_length=80)]
    departure_date: date
    return_date: date | None = None
    travellers: Annotated[int, Field(ge=1, le=MAX_TRAVELLERS)] = 1
    currency: Currency = Currency.EUR
    preferred_mode: TransportMode = TransportMode.ANY
    max_duration_hours: Annotated[float, Field(gt=0, le=48)] = 12.0
    max_transfers: Annotated[int, Field(ge=0, le=5)] = 2
    accessibility_needs: Annotated[list[AccessibilityNeed], Field(max_length=8)] = Field(
        default_factory=list
    )
    ranking_strategy: RankingStrategy = RankingStrategy.BALANCED
    limit: Annotated[int, Field(ge=1, le=MAX_RESULTS)] = 10

    @model_validator(mode="after")
    def _validate(self) -> TransportSearchArgs:
        if self.return_date is not None and self.return_date < self.departure_date:
            raise ValueError("return_date must not precede departure_date")
        if self.origin.strip().casefold() == self.destination.strip().casefold():
            raise ValueError("origin and destination are the same place")
        return self


class ScoreTransportArgs(StrictModel):
    """Arguments for ``calculate_transport_score``.

    Offers are passed as opaque dicts so the tool can validate them itself. Accepting
    pre-validated domain objects would let a malformed offer skip the check that the tool
    exists to perform.
    """

    offers: Annotated[list[dict[str, Any]], Field(min_length=1, max_length=MAX_RESULTS)]
    ranking_strategy: RankingStrategy = RankingStrategy.BALANCED


# ---------------------------------------------------------------------------
# Lodging MCP
# ---------------------------------------------------------------------------
class AccommodationSearchArgs(StrictModel):
    """Arguments for ``search_accommodation``."""

    destination: Annotated[str, Field(min_length=2, max_length=80)]
    check_in: date
    check_out: date
    guests: Annotated[int, Field(ge=1, le=MAX_TRAVELLERS)] = 1
    guest_nationality: Annotated[str | None, Field(pattern=r"^[A-Z]{2}$")] = None
    currency: Currency = Currency.EUR
    accommodation_type: AccommodationType = AccommodationType.ANY
    max_total_price: Annotated[float | None, Field(ge=0)] = None
    accessibility_needs: Annotated[list[AccessibilityNeed], Field(max_length=8)] = Field(
        default_factory=list
    )
    ranking_strategy: RankingStrategy = RankingStrategy.BALANCED
    limit: Annotated[int, Field(ge=1, le=MAX_RESULTS)] = 10

    @property
    def nights(self) -> int:
        return (self.check_out - self.check_in).days

    @model_validator(mode="after")
    def _validate(self) -> AccommodationSearchArgs:
        if self.check_out <= self.check_in:
            raise ValueError("check_out must be after check_in")
        if (self.check_out - self.check_in).days > MAX_NIGHTS:
            raise ValueError(f"stay exceeds the {MAX_NIGHTS}-night maximum")
        return self


class StayCostArgs(StrictModel):
    """Arguments for ``calculate_total_stay_cost``."""

    price_per_night: Annotated[float, Field(ge=0, le=100_000)]
    nights: Annotated[int, Field(ge=1, le=MAX_NIGHTS)]
    currency: Currency = Currency.EUR
    rooms: Annotated[int, Field(ge=1, le=10)] = 1


class DistanceArgs(StrictModel):
    """Arguments for ``calculate_distance_score`` and ``calculate_location_distance``."""

    origin: GeoPoint
    destination: GeoPoint
    low_walking_distance: bool = False


# ---------------------------------------------------------------------------
# Destination MCP
# ---------------------------------------------------------------------------
class GeocodeArgs(StrictModel):
    """Arguments for ``geocode_destination``."""

    place: Annotated[str, Field(min_length=2, max_length=100)]


class POISearchArgs(StrictModel):
    """Arguments for ``search_points_of_interest``."""

    destination: Annotated[str, Field(min_length=2, max_length=80)]
    interests: Annotated[list[str], Field(max_length=12)] = Field(default_factory=list)
    limit: Annotated[int, Field(ge=1, le=50)] = 30
    accessibility_needs: Annotated[list[AccessibilityNeed], Field(max_length=8)] = Field(
        default_factory=list
    )


class WeatherArgs(StrictModel):
    """Arguments for ``get_weather_forecast``."""

    location: GeoPoint
    start_date: date
    end_date: date
    location_name: Annotated[str, Field(max_length=120)] = ""

    @model_validator(mode="after")
    def _validate(self) -> WeatherArgs:
        if self.end_date < self.start_date:
            raise ValueError("end_date must not precede start_date")
        if (self.end_date - self.start_date).days > MAX_NIGHTS:
            raise ValueError(f"forecast window exceeds {MAX_NIGHTS} days")
        return self


class KnowledgeArgs(StrictModel):
    """Arguments for ``retrieve_destination_knowledge``.

    The retrieved text is untrusted and is wrapped before it enters any prompt — see
    ``vm_llm.wrap_untrusted`` and threat T-1.
    """

    destination: Annotated[str, Field(min_length=2, max_length=80)]
    query: Annotated[str, Field(min_length=2, max_length=300)]
    limit: Annotated[int, Field(ge=1, le=10)] = 4


class ValidateScheduleArgs(StrictModel):
    """Arguments for ``validate_daily_schedule``."""

    slots: Annotated[list[dict[str, Any]], Field(max_length=20)]
    day_number: Annotated[int, Field(ge=1, le=31)] = 1
