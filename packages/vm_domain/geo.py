"""Geographic calculations.

Pure functions over :class:`~vm_contracts.common.GeoPoint`. No I/O, no external routing
service — a routing API would be more accurate but would make itinerary feasibility depend
on a network call that can fail, and the walking estimates here only need to be good enough
to prevent an impossible schedule.

Travel times are deliberately conservative. Under-estimating produces an itinerary that
looks fine and cannot actually be walked, which is the failure mode that matters.
"""

from __future__ import annotations

import math

from vm_contracts.common import GeoPoint

__all__ = [
    "AVERAGE_WALK_KMH",
    "URBAN_TRANSIT_KMH",
    "distance_km",
    "estimate_travel_minutes",
    "normalise",
]

EARTH_RADIUS_KM = 6371.0088

AVERAGE_WALK_KMH = 4.2
"""Comfortable sightseeing pace, not a brisk commute. Tourists stop, look, and photograph."""

URBAN_TRANSIT_KMH = 18.0
"""Effective door-to-door speed for tram/metro including waiting and walking to the stop."""

WALK_THRESHOLD_KM = 1.8
"""Beyond this, assume public transport rather than walking."""

_STREET_FACTOR = 1.25
"""Straight-line distance under-states real walking distance because streets are a grid,
not a line. 1.25 is the conventional detour factor for European city centres."""

_TRANSIT_OVERHEAD_MINUTES = 7.0
"""Fixed cost of using transit: finding the stop, waiting, and the final walk."""


def distance_km(origin: GeoPoint, destination: GeoPoint) -> float:
    """Great-circle distance in kilometres (haversine).

    Haversine rather than a planar approximation because the latter degrades badly at
    high latitudes, and this project happily plans trips to Tromsø.
    """
    lat1, lon1 = math.radians(origin.latitude), math.radians(origin.longitude)
    lat2, lon2 = math.radians(destination.latitude), math.radians(destination.longitude)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    # Clamp before asin: floating-point error can push `a` a hair above 1.0 for antipodal
    # points, which would raise ValueError from an otherwise valid calculation.
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


def estimate_travel_minutes(
    origin: GeoPoint | None,
    destination: GeoPoint | None,
    *,
    low_walking_distance: bool = False,
) -> int:
    """Conservative door-to-door travel time between two points, in whole minutes.

    Returns a default of 20 minutes when either coordinate is unknown: assuming zero would
    let the itinerary validator accept a schedule that cannot be walked, and the whole
    point of that validator is to catch exactly that.

    Args:
        origin: Start point, or ``None`` if not geocoded.
        destination: End point, or ``None`` if not geocoded.
        low_walking_distance: When the traveller has stated a low-walking-distance
            accessibility need, transit is assumed above a much shorter threshold.
    """
    if origin is None or destination is None:
        return 20

    straight_line = distance_km(origin, destination)
    if straight_line < 0.05:
        return 0

    street_distance = straight_line * _STREET_FACTOR
    walk_limit = 0.6 if low_walking_distance else WALK_THRESHOLD_KM

    if street_distance <= walk_limit:
        minutes = (street_distance / AVERAGE_WALK_KMH) * 60.0
    else:
        minutes = (street_distance / URBAN_TRANSIT_KMH) * 60.0 + _TRANSIT_OVERHEAD_MINUTES

    return max(1, math.ceil(minutes))


def normalise(value: float, low: float, high: float) -> float:
    """Scale ``value`` into ``[0.0, 1.0]`` given the observed range.

    Returns ``0.0`` when ``low == high``. That case means every candidate scored
    identically on this attribute, so it carries no information and must not influence the
    ranking — returning 0.5 instead would silently shift every total score.
    """
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))
