"""Fixture-backed providers for points of interest and weather.

These are the fallback path used when a live API is unavailable, unkeyed, or the system is
running in ``PROVIDER_MODE=mock``.

Note the difference in labelling. Attractions come from a curated file of **real places**
and are labelled ``FIXTURE`` — the names, coordinates, and visit durations are accurate.
Weather, by contrast, cannot be curated: a forecast for a future date is either live or it
is a guess. The fixture weather provider therefore labels its output ``MOCKED`` and marks
the summary with an explicit note, because presenting a synthesised forecast as a real one
would be exactly the kind of confident fabrication this project refuses to make.

File access is deliberately constrained: paths are resolved against a fixed base directory
and a static filename, never from user input (threat T-11).
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from vm_contracts.common import (
    AccessibilityNeed,
    DataOrigin,
    GeoPoint,
    Provenance,
)
from vm_contracts.destination import (
    Attraction,
    AttractionCategory,
    WeatherDay,
    WeatherSummary,
)
from vm_domain.providers.base import POIQuery, ProviderResult, WeatherQuery

__all__ = [
    "FIXTURE_DIR",
    "FixturePOIProvider",
    "FixtureWeatherProvider",
    "available_fixture_cities",
]

# Resolved once from this file's location. User input never contributes to a path.
FIXTURE_DIR = Path(__file__).resolve().parents[3] / "data" / "fixtures"
_ATTRACTIONS_FILE = "attractions.json"

POI_SOURCE_NAME = "VoyageMesh curated destination fixtures"
WEATHER_SOURCE_NAME = "VoyageMesh synthetic climatology (not a forecast)"


@lru_cache(maxsize=1)
def _load_attractions() -> dict[str, Any]:
    """Load and cache the curated attraction corpus.

    Cached because it is read on nearly every itinerary request and never changes at
    runtime. A missing or malformed file yields an empty corpus rather than raising — the
    Itinerary Agent must degrade to "no attractions found", not fail the whole trip.
    """
    path = FIXTURE_DIR / _ATTRACTIONS_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    cities = raw.get("cities")
    return cities if isinstance(cities, dict) else {}


def available_fixture_cities() -> list[str]:
    return sorted(_load_attractions().keys())


class FixturePOIProvider:
    """Serves curated real-world attractions from the repository."""

    name = "fixture_poi"

    async def search(self, query: POIQuery) -> ProviderResult[Attraction]:
        cities = _load_attractions()
        city = cities.get(query.destination.strip().casefold())

        if city is None:
            # Honest emptiness. Inventing attractions for an unknown city would be the
            # single worst thing this provider could do.
            return ProviderResult(
                items=[],
                origin=DataOrigin.UNAVAILABLE,
                source_name=POI_SOURCE_NAME,
                error=(
                    f"No curated attraction data for '{query.destination}'. "
                    f"Available: {', '.join(available_fixture_cities()) or 'none'}."
                ),
            )

        attractions: list[Attraction] = []
        for entry in city.get("attractions", []):
            attraction = _parse_attraction(entry)
            if attraction is not None:
                attractions.append(attraction)

        if query.interests:
            interests = list(query.interests)
            matched = [a for a in attractions if a.matches_interests(interests)]
            # Falling back to the full set beats returning nothing: a traveller whose
            # niche interest matches no local attraction still wants to see the highlights.
            attractions = matched or attractions

        return ProviderResult(
            items=attractions[: query.limit],
            origin=DataOrigin.FIXTURE,
            source_name=POI_SOURCE_NAME,
            warnings=[
                "Attraction details come from a curated fixture file. Opening hours and "
                "admission prices are deliberately not included, because stale values "
                "presented as current would be misleading."
            ],
        )


def _parse_attraction(entry: dict[str, Any]) -> Attraction | None:
    """Convert one fixture record, skipping anything malformed.

    Skipping rather than raising keeps one bad hand-edited record from taking down the
    whole destination.
    """
    try:
        latitude = entry.get("latitude")
        longitude = entry.get("longitude")
        location = (
            GeoPoint(latitude=float(latitude), longitude=float(longitude))
            if latitude is not None and longitude is not None
            else None
        )
        return Attraction(
            attraction_id=str(entry["attraction_id"]),
            name=str(entry["name"]),
            category=AttractionCategory(entry.get("category", "other")),
            location=location,
            description=entry.get("description"),
            typical_visit_minutes=int(entry.get("typical_visit_minutes", 60)),
            is_free=bool(entry.get("is_free", False)),
            is_outdoor=bool(entry.get("is_outdoor", False)),
            accessibility_is_known=bool(entry.get("accessibility_is_known", False)),
            accessibility_features=[
                AccessibilityNeed(value) for value in entry.get("accessibility_features", [])
            ],
            interest_tags=[str(tag) for tag in entry.get("interest_tags", [])],
            provenance=Provenance.fixture(
                POI_SOURCE_NAME, source_id=str(entry.get("attraction_id", ""))
            ),
        )
    except (KeyError, ValueError, TypeError):
        return None


class FixtureWeatherProvider:
    """Synthesises a seasonal weather pattern when no live forecast is available.

    This is **not a forecast** and does not pretend to be one. Values follow a smooth
    seasonal curve for central-European latitudes with deterministic day-to-day variation,
    labelled ``MOCKED``, and the returned summary always carries a note saying so.

    It exists so the itinerary planner's weather-suitability logic remains exercisable
    offline and in tests — not to tell anyone what to pack.
    """

    name = "fixture_weather"

    async def forecast(self, query: WeatherQuery) -> ProviderResult[WeatherSummary]:
        if query.end_date < query.start_date:
            return ProviderResult.unavailable(WEATHER_SOURCE_NAME, "end_date precedes start_date")

        days: list[WeatherDay] = []
        current = query.start_date
        while current <= query.end_date and len(days) < 31:
            days.append(_synthetic_day(current, query.location))
            current += timedelta(days=1)

        return ProviderResult(
            items=[
                WeatherSummary(
                    location=query.location_name or "destination",
                    coordinates=query.location,
                    days=days,
                    provenance=Provenance.mocked(WEATHER_SOURCE_NAME),
                    note=(
                        "Synthetic seasonal estimate, not a forecast. Live weather was "
                        "unavailable; do not rely on these values."
                    ),
                )
            ],
            origin=DataOrigin.MOCKED,
            source_name=WEATHER_SOURCE_NAME,
            warnings=["Weather is a synthetic seasonal estimate, not a real forecast."],
        )


def _synthetic_day(day: date, location: GeoPoint) -> WeatherDay:
    """A plausible seasonal day, deterministic in the date and location."""
    # Seasonal sine peaking in late July (day 202).
    day_of_year = day.timetuple().tm_yday
    seasonal = math.sin((day_of_year - 111) / 365.0 * 2 * math.pi)

    # Cooler with latitude: roughly 0.6 °C per degree north of 45.
    latitude_adjustment = (location.latitude - 45.0) * 0.6
    mean_temp = 12.0 + seasonal * 11.0 - latitude_adjustment

    variation = _stable_unit(f"{day.isoformat()}:{location.latitude:.2f}")
    temp_max = round(mean_temp + 4.0 + variation * 5.0, 1)
    temp_min = round(mean_temp - 4.0 + variation * 2.0, 1)

    rain_roll = _stable_unit(f"rain:{day.isoformat()}:{location.longitude:.2f}")
    precipitation = round(max(0.0, (rain_roll - 0.62) * 24.0), 1)

    if precipitation > 8.0:
        condition = "heavy rain"
    elif precipitation > 2.0:
        condition = "showers"
    elif precipitation > 0.2:
        condition = "light rain"
    elif variation > 0.6:
        condition = "clear"
    else:
        condition = "partly cloudy"

    return WeatherDay(
        forecast_date=day,
        temperature_min_c=min(temp_min, temp_max),
        temperature_max_c=max(temp_min, temp_max),
        precipitation_mm=precipitation,
        precipitation_probability=round(min(100.0, rain_roll * 100.0), 1),
        wind_speed_kmh=round(6.0 + variation * 18.0, 1),
        condition=condition,
    )


def _stable_unit(key: str) -> float:
    """Deterministic value in ``[0.0, 1.0)`` derived from ``key``."""
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:4], "big") / 0x100000000
