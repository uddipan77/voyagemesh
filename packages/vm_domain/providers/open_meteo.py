"""Open-Meteo weather and geocoding — a genuinely live, free data source.

Open-Meteo requires no API key, permits non-commercial use freely, and returns real
forecasts and coordinates, independently of the credentialed travel providers.

Failure is degradation, never an exception. Both providers return a ``ProviderResult`` with
``origin=UNAVAILABLE`` when the API cannot be reached, so the Itinerary Agent can drop
weather from the plan and say so, rather than failing the whole trip (brief §23).

Every field is bounds-checked before it becomes a domain object. A provider returning a
temperature of 9999 is a bug or an attack, and either way it must not reach the itinerary
planner (threat T-5).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from vm_contracts.common import DataOrigin, GeoPoint, Provenance
from vm_contracts.destination import WeatherDay, WeatherSummary
from vm_domain.providers.base import ProviderResult, WeatherQuery
from vm_resilience.safe_http import SafeHTTPClient

__all__ = [
    "GEOCODING_HOST",
    "WEATHER_HOST",
    "OpenMeteoGeocodingProvider",
    "OpenMeteoWeatherProvider",
]

logger = logging.getLogger(__name__)

WEATHER_HOST = "api.open-meteo.com"
GEOCODING_HOST = "geocoding-api.open-meteo.com"

_FORECAST_URL = f"https://{WEATHER_HOST}/v1/forecast"
_GEOCODING_URL = f"https://{GEOCODING_HOST}/v1/search"

SOURCE_NAME = "Open-Meteo"
LICENSE_NOTE = "Open-Meteo, CC BY 4.0 — free for non-commercial use"

# Open-Meteo publishes about 16 days of forecast. Beyond that the API returns nothing
# rather than erroring, so the gap is detected and reported instead of silently omitted.
FORECAST_HORIZON_DAYS = 16

# WMO weather interpretation codes, condensed to what a traveller actually needs.
_WMO_CONDITIONS: dict[int, str] = {
    0: "clear",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


class OpenMeteoWeatherProvider:
    """Live daily forecasts from Open-Meteo."""

    name = "open_meteo_weather"

    def __init__(self, client: SafeHTTPClient) -> None:
        self._client = client

    async def forecast(self, query: WeatherQuery) -> ProviderResult[WeatherSummary]:
        if query.end_date < query.start_date:
            return ProviderResult.unavailable(SOURCE_NAME, "end_date precedes start_date")

        params = {
            "latitude": round(query.location.latitude, 4),
            "longitude": round(query.location.longitude, 4),
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "precipitation_probability_max",
                    "wind_speed_10m_max",
                ]
            ),
            "start_date": query.start_date.isoformat(),
            "end_date": query.end_date.isoformat(),
            "timezone": "auto",
        }

        try:
            payload = await self._client.get_json(_FORECAST_URL, params=params)
        except Exception as exc:
            # Degrade, never raise: the Itinerary Agent must be able to drop weather and
            # report it rather than failing the whole trip.
            logger.warning("open_meteo_weather_unavailable", extra={"error": type(exc).__name__})
            return ProviderResult.unavailable(
                SOURCE_NAME, f"weather lookup failed ({type(exc).__name__})"
            )

        if not isinstance(payload, dict):
            return ProviderResult.unavailable(SOURCE_NAME, "unexpected response shape")

        days, warnings = _parse_daily(payload.get("daily"))
        if not days:
            return ProviderResult.unavailable(
                SOURCE_NAME, "response contained no usable forecast days"
            )

        requested = (query.end_date - query.start_date).days + 1
        note: str | None = None
        if len(days) < requested:
            missing = requested - len(days)
            note = (
                f"Forecast covers {len(days)} of {requested} requested days. Open-Meteo "
                f"publishes about {FORECAST_HORIZON_DAYS} days ahead, so the final "
                f"{missing} day(s) of this trip have no forecast yet."
            )

        return ProviderResult(
            items=[
                WeatherSummary(
                    location=query.location_name or "destination",
                    coordinates=query.location,
                    days=days,
                    provenance=Provenance.live(
                        SOURCE_NAME,
                        source_url=_FORECAST_URL,
                        license_note=LICENSE_NOTE,
                    ),
                    note=note,
                )
            ],
            origin=DataOrigin.LIVE,
            source_name=SOURCE_NAME,
            warnings=warnings,
        )


class OpenMeteoGeocodingProvider:
    """Resolves a place name to coordinates using Open-Meteo's geocoding API."""

    name = "open_meteo_geocoding"

    def __init__(self, client: SafeHTTPClient) -> None:
        self._client = client

    async def geocode(self, place: str) -> ProviderResult[GeoPoint]:
        cleaned = " ".join(place.split())[:100]
        if not cleaned:
            return ProviderResult.unavailable(SOURCE_NAME, "empty place name")

        try:
            payload = await self._client.get_json(
                _GEOCODING_URL,
                params={"name": cleaned, "count": 1, "language": "en", "format": "json"},
            )
        except Exception as exc:
            logger.warning("open_meteo_geocoding_unavailable", extra={"error": type(exc).__name__})
            return ProviderResult.unavailable(
                SOURCE_NAME, f"geocoding failed ({type(exc).__name__})"
            )

        if not isinstance(payload, dict):
            return ProviderResult.unavailable(SOURCE_NAME, "unexpected response shape")

        results = payload.get("results")
        if not isinstance(results, list) or not results:
            # A place we cannot locate is reported honestly. Guessing a coordinate would
            # produce a confidently wrong itinerary.
            return ProviderResult(
                items=[],
                origin=DataOrigin.UNAVAILABLE,
                source_name=SOURCE_NAME,
                error=f"no coordinates found for '{cleaned}'",
            )

        first = results[0]
        if not isinstance(first, dict):
            return ProviderResult.unavailable(SOURCE_NAME, "unexpected result shape")

        latitude = _coerce_float(first.get("latitude"), low=-90.0, high=90.0)
        longitude = _coerce_float(first.get("longitude"), low=-180.0, high=180.0)
        if latitude is None or longitude is None:
            return ProviderResult.unavailable(
                SOURCE_NAME, "result did not contain valid coordinates"
            )

        label = str(first.get("name", cleaned))[:120]
        country = first.get("country")
        return ProviderResult(
            items=[GeoPoint(latitude=latitude, longitude=longitude)],
            origin=DataOrigin.LIVE,
            source_name=SOURCE_NAME,
            warnings=[f"Resolved '{cleaned}' to {label}" + (f", {country}" if country else "")],
        )


def _parse_daily(daily: Any) -> tuple[list[WeatherDay], list[str]]:
    """Convert Open-Meteo's column-oriented arrays into validated day records.

    Malformed or out-of-range entries are dropped with a warning rather than raising: one
    bad day must not cost the traveller the whole forecast (threat T-5).
    """
    if not isinstance(daily, dict):
        return [], ["weather response contained no daily block"]

    dates = daily.get("time")
    if not isinstance(dates, list) or not dates:
        return [], ["weather response contained no dates"]

    codes = _column(daily, "weather_code", len(dates))
    highs = _column(daily, "temperature_2m_max", len(dates))
    lows = _column(daily, "temperature_2m_min", len(dates))
    rain = _column(daily, "precipitation_sum", len(dates))
    rain_probability = _column(daily, "precipitation_probability_max", len(dates))
    wind = _column(daily, "wind_speed_10m_max", len(dates))

    days: list[WeatherDay] = []
    warnings: list[str] = []
    skipped = 0

    for index, raw_date in enumerate(dates):
        try:
            forecast_date = date.fromisoformat(str(raw_date))
        except ValueError:
            skipped += 1
            continue

        high = _coerce_float(highs[index], low=-90.0, high=60.0)
        low = _coerce_float(lows[index], low=-90.0, high=60.0)
        if high is None or low is None:
            skipped += 1
            continue

        precipitation = _coerce_float(rain[index], low=0.0, high=1000.0) or 0.0
        code = codes[index]
        condition = _WMO_CONDITIONS.get(int(code), "unknown") if _is_number(code) else "unknown"

        try:
            days.append(
                WeatherDay(
                    forecast_date=forecast_date,
                    # Open-Meteo has been observed to return max < min at high latitudes
                    # around the poles; ordering them keeps the model constructible.
                    temperature_min_c=min(low, high),
                    temperature_max_c=max(low, high),
                    precipitation_mm=precipitation,
                    precipitation_probability=_coerce_float(
                        rain_probability[index], low=0.0, high=100.0
                    ),
                    wind_speed_kmh=_coerce_float(wind[index], low=0.0, high=500.0),
                    condition=condition,
                )
            )
        except ValueError:
            skipped += 1

    if skipped:
        warnings.append(f"{skipped} forecast day(s) were dropped as malformed or out of range")

    return days, warnings


def _column(daily: dict[str, Any], key: str, length: int) -> list[Any]:
    """Return a column padded to ``length`` so index access is always safe."""
    values = daily.get(key)
    if not isinstance(values, list):
        return [None] * length
    if len(values) < length:
        return [*values, *([None] * (length - len(values)))]
    return values[:length]


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _coerce_float(value: Any, *, low: float, high: float) -> float | None:
    """Convert to a float inside ``[low, high]``, or ``None``.

    Out-of-range values are rejected rather than clamped: a temperature of 9999 means the
    response is wrong, and clamping it to 60 would launder a bug into plausible data.
    """
    if not _is_number(value):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):  # NaN / infinity
        return None
    if number < low or number > high:
        return None
    return number
