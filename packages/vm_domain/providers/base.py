"""Provider protocols — the seam between the domain and the outside world.

Every external data source implements one of these ``Protocol`` classes. Agents and MCP
tools depend on the protocol, never on a concrete provider, which is what allows the whole
system to run offline against deterministic mocks and against real free APIs with the same
code path (ADR-010).

Providers must **never raise** for an ordinary failure. A timeout, a rate limit, or an
empty result is expressed as a :class:`ProviderResult` with ``origin=UNAVAILABLE`` and a
populated ``error``. Raising would force every caller to write the same try/except, and a
missed one would turn a degraded dependency into a failed user request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

from vm_contracts.common import (
    AccessibilityNeed,
    AccommodationType,
    Currency,
    DataOrigin,
    GeoPoint,
    TransportMode,
)
from vm_contracts.destination import Attraction, WeatherSummary
from vm_contracts.offers import AccommodationOffer, TransportOffer

__all__ = [
    "AccommodationProvider",
    "AccommodationQuery",
    "GeocodingProvider",
    "POIProvider",
    "POIQuery",
    "ProviderResult",
    "TransportProvider",
    "TransportQuery",
    "WeatherProvider",
    "WeatherQuery",
]


@dataclass(frozen=True, slots=True)
class ProviderResult[T]:
    """What a provider returns: the data, where it came from, and what went wrong.

    Carrying the origin alongside the payload is what lets the UI label every record
    honestly without the caller having to remember which provider it used.
    """

    items: list[T]
    origin: DataOrigin
    source_name: str
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def is_empty(self) -> bool:
        return not self.items

    @classmethod
    def unavailable(cls, source_name: str, error: str) -> ProviderResult[T]:
        """A failed lookup. Note ``items`` is empty — never a fabricated substitute."""
        return cls(items=[], origin=DataOrigin.UNAVAILABLE, source_name=source_name, error=error)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TransportQuery:
    origin: str
    destination: str
    departure_date: date
    return_date: date | None
    travellers: int
    currency: Currency
    preferred_mode: TransportMode = TransportMode.ANY
    max_duration_hours: float = 12.0
    max_transfers: int = 2
    accessibility_needs: tuple[AccessibilityNeed, ...] = ()


@dataclass(frozen=True, slots=True)
class AccommodationQuery:
    destination: str
    check_in: date
    check_out: date
    guests: int
    currency: Currency
    guest_nationality: str | None = None
    accommodation_type: AccommodationType = AccommodationType.ANY
    max_total_price: float | None = None
    accessibility_needs: tuple[AccessibilityNeed, ...] = ()

    @property
    def nights(self) -> int:
        return (self.check_out - self.check_in).days


@dataclass(frozen=True, slots=True)
class POIQuery:
    destination: str
    location: GeoPoint | None = None
    interests: tuple[str, ...] = ()
    radius_metres: int = 5000
    limit: int = 30


@dataclass(frozen=True, slots=True)
class WeatherQuery:
    location: GeoPoint
    start_date: date
    end_date: date
    location_name: str = ""


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
class TransportProvider(Protocol):
    """A source of transport offers."""

    name: str

    async def search(self, query: TransportQuery) -> ProviderResult[TransportOffer]:
        """Find journeys matching ``query``. Must not raise for an ordinary failure."""
        ...


class AccommodationProvider(Protocol):
    """A source of accommodation offers."""

    name: str

    async def search(self, query: AccommodationQuery) -> ProviderResult[AccommodationOffer]: ...


class POIProvider(Protocol):
    """A source of points of interest."""

    name: str

    async def search(self, query: POIQuery) -> ProviderResult[Attraction]: ...


class WeatherProvider(Protocol):
    """A source of weather forecasts."""

    name: str

    async def forecast(self, query: WeatherQuery) -> ProviderResult[WeatherSummary]: ...


class GeocodingProvider(Protocol):
    """Resolves a place name to coordinates."""

    name: str

    async def geocode(self, place: str) -> ProviderResult[GeoPoint]: ...
