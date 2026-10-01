"""Pure domain logic: scoring, budgets, itinerary planning, and data providers.

This package performs **no I/O of its own** beyond reading repository fixture files, holds
no global mutable state, and imports no service. Every calculation the user ultimately
sees — every price, duration, score, and schedule — is produced here by a deterministic
function that can be re-run and reasoned about (ADR-009).

The practical test of that claim: the entire package can be exercised without a network,
a database, an LLM, or a running service, and identical inputs always produce identical
outputs.
"""

from vm_domain.budget import (
    DailyAllowances,
    calculate_budget,
    default_allowances,
    remaining_accommodation_allowance,
)
from vm_domain.geo import distance_km, estimate_travel_minutes, normalise
from vm_domain.itinerary import ItineraryPlanner, PlanningWindow
from vm_domain.providers.base import (
    AccommodationProvider,
    AccommodationQuery,
    GeocodingProvider,
    POIProvider,
    POIQuery,
    ProviderResult,
    TransportProvider,
    TransportQuery,
    WeatherProvider,
    WeatherQuery,
)
from vm_domain.providers.fixtures import (
    FixturePOIProvider,
    FixtureWeatherProvider,
    available_fixture_cities,
)
from vm_domain.providers.mock_accommodation import MockAccommodationProvider
from vm_domain.providers.mock_transport import MockTransportProvider
from vm_domain.scoring import (
    TOP_N,
    AccommodationWeights,
    TransportWeights,
    accommodation_weights_for,
    filter_accommodation,
    filter_transport,
    rank_accommodation,
    rank_transport,
    schedule_inconvenience,
    transport_weights_for,
)

__all__ = [
    "TOP_N",
    "AccommodationProvider",
    "AccommodationQuery",
    "AccommodationWeights",
    "DailyAllowances",
    "FixturePOIProvider",
    "FixtureWeatherProvider",
    "GeocodingProvider",
    "ItineraryPlanner",
    "MockAccommodationProvider",
    "MockTransportProvider",
    "POIProvider",
    "POIQuery",
    "PlanningWindow",
    "ProviderResult",
    "TransportProvider",
    "TransportQuery",
    "TransportWeights",
    "WeatherProvider",
    "WeatherQuery",
    "accommodation_weights_for",
    "available_fixture_cities",
    "calculate_budget",
    "default_allowances",
    "distance_km",
    "estimate_travel_minutes",
    "filter_accommodation",
    "filter_transport",
    "normalise",
    "rank_accommodation",
    "rank_transport",
    "remaining_accommodation_allowance",
    "schedule_inconvenience",
    "transport_weights_for",
]
