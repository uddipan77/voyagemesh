"""Open-Meteo provider parsing and degradation, exercised offline with respx.

The live smoke test proves the integration reaches the real API. These tests prove what
that cannot cheaply prove: that a malformed day is dropped rather than crashing the
forecast, that an out-of-range temperature is refused rather than laundered into plausible
data (threat T-5), and that every failure degrades to an honest UNAVAILABLE result rather
than an exception (brief §23).
"""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from vm_domain.providers.base import WeatherQuery
from vm_domain.providers.open_meteo import (
    GEOCODING_HOST,
    WEATHER_HOST,
    OpenMeteoGeocodingProvider,
    OpenMeteoWeatherProvider,
)
from vm_resilience import SafeHTTPClient

pytestmark = pytest.mark.unit

FORECAST_URL = f"https://{WEATHER_HOST}/v1/forecast"
GEOCODE_URL = f"https://{GEOCODING_HOST}/v1/search"

PRAGUE = {"latitude": 50.0875, "longitude": 14.4213}


@pytest.fixture
async def client():
    instance = SafeHTTPClient(allowed_hosts=[WEATHER_HOST, GEOCODING_HOST], max_retries=0)
    try:
        yield instance
    finally:
        await instance.aclose()


def _daily(days: int = 3, **overrides):
    base = {
        "time": [f"2026-08-{10 + i:02d}" for i in range(days)],
        "weather_code": [0, 61, 3][:days],
        "temperature_2m_max": [26.0, 21.0, 24.0][:days],
        "temperature_2m_min": [15.0, 14.0, 16.0][:days],
        "precipitation_sum": [0.0, 4.8, 0.2][:days],
        "precipitation_probability_max": [0, 60, 20][:days],
        "wind_speed_10m_max": [6.0, 18.0, 12.0][:days],
    }
    base.update(overrides)
    return base


class TestWeatherParsing:
    async def test_parses_a_valid_forecast(self, client, respx_mock):
        respx_mock.get(FORECAST_URL).mock(
            return_value=httpx.Response(200, json={"daily": _daily(3)})
        )
        from vm_contracts.common import GeoPoint

        result = await OpenMeteoWeatherProvider(client).forecast(
            WeatherQuery(
                location=GeoPoint(**PRAGUE),
                start_date=date(2026, 8, 10),
                end_date=date(2026, 8, 12),
                location_name="Prague",
            )
        )
        assert result.ok
        assert result.origin.value == "live"
        summary = result.items[0]
        assert len(summary.days) == 3
        assert summary.provenance.origin.value == "live"
        assert summary.provenance.retrieved_at is not None
        assert summary.days[0].condition == "clear"
        assert summary.days[1].condition == "light rain"

    async def test_out_of_range_temperature_is_dropped_not_laundered(self, client, respx_mock):
        """A temperature of 9999 means the response is wrong; clamping it would launder a
        bug into plausible data."""
        from vm_contracts.common import GeoPoint

        bad = _daily(3)
        bad["temperature_2m_max"] = [26.0, 9999.0, 24.0]
        respx_mock.get(FORECAST_URL).mock(return_value=httpx.Response(200, json={"daily": bad}))

        result = await OpenMeteoWeatherProvider(client).forecast(
            WeatherQuery(
                location=GeoPoint(**PRAGUE),
                start_date=date(2026, 8, 10),
                end_date=date(2026, 8, 12),
            )
        )
        assert result.ok
        assert len(result.items[0].days) == 2, "the impossible day should be dropped"
        # The warning lives on the ProviderResult envelope, not on the WeatherSummary.
        assert any("dropped" in w for w in result.warnings)

    async def test_missing_columns_are_padded_not_crashed(self, client, respx_mock):
        from vm_contracts.common import GeoPoint

        partial = _daily(3)
        del partial["wind_speed_10m_max"]  # provider omitted a column entirely
        respx_mock.get(FORECAST_URL).mock(return_value=httpx.Response(200, json={"daily": partial}))
        result = await OpenMeteoWeatherProvider(client).forecast(
            WeatherQuery(
                location=GeoPoint(**PRAGUE),
                start_date=date(2026, 8, 10),
                end_date=date(2026, 8, 12),
            )
        )
        assert result.ok
        assert result.items[0].days[0].wind_speed_kmh is None

    async def test_short_forecast_reports_the_gap(self, client, respx_mock):
        """Requesting 5 days but the API returning 3 must be flagged, not silently truncated."""
        from vm_contracts.common import GeoPoint

        respx_mock.get(FORECAST_URL).mock(
            return_value=httpx.Response(200, json={"daily": _daily(3)})
        )
        result = await OpenMeteoWeatherProvider(client).forecast(
            WeatherQuery(
                location=GeoPoint(**PRAGUE),
                start_date=date(2026, 8, 10),
                end_date=date(2026, 8, 14),  # 5 days
            )
        )
        assert result.ok
        note = result.items[0].note
        assert note is not None and "3 of 5" in note
        assert result.items[0].is_complete is False

    async def test_empty_daily_block_degrades_honestly(self, client, respx_mock):
        from vm_contracts.common import GeoPoint

        respx_mock.get(FORECAST_URL).mock(
            return_value=httpx.Response(200, json={"daily": {"time": []}})
        )
        result = await OpenMeteoWeatherProvider(client).forecast(
            WeatherQuery(
                location=GeoPoint(**PRAGUE),
                start_date=date(2026, 8, 10),
                end_date=date(2026, 8, 12),
            )
        )
        assert result.ok is False
        assert result.origin.value == "unavailable"

    async def test_http_failure_degrades_never_raises(self, client, respx_mock):
        from vm_contracts.common import GeoPoint

        respx_mock.get(FORECAST_URL).mock(return_value=httpx.Response(500))
        result = await OpenMeteoWeatherProvider(client).forecast(
            WeatherQuery(
                location=GeoPoint(**PRAGUE),
                start_date=date(2026, 8, 10),
                end_date=date(2026, 8, 12),
            )
        )
        assert result.ok is False
        assert result.origin.value == "unavailable"
        assert result.error is not None

    async def test_reversed_dates_are_rejected_before_any_call(self, client):
        from vm_contracts.common import GeoPoint

        result = await OpenMeteoWeatherProvider(client).forecast(
            WeatherQuery(
                location=GeoPoint(**PRAGUE),
                start_date=date(2026, 8, 12),
                end_date=date(2026, 8, 10),
            )
        )
        assert result.ok is False


class TestGeocoding:
    async def test_resolves_a_place(self, client, respx_mock):
        respx_mock.get(GEOCODE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "name": "Prague",
                            "country": "Czechia",
                            "latitude": 50.08804,
                            "longitude": 14.42076,
                        }
                    ]
                },
            )
        )
        result = await OpenMeteoGeocodingProvider(client).geocode("Prague")
        assert result.ok
        assert result.origin.value == "live"
        point = result.items[0]
        assert 50 < point.latitude < 51

    async def test_unknown_place_reports_not_found_not_a_guess(self, client, respx_mock):
        """Guessing a coordinate for an unlocatable place would produce a confidently wrong
        itinerary."""
        respx_mock.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json={"results": []}))
        result = await OpenMeteoGeocodingProvider(client).geocode("Zzyzxville")
        assert result.ok is False
        assert result.items == []
        assert result.error is not None

    async def test_out_of_range_coordinates_are_refused(self, client, respx_mock):
        respx_mock.get(GEOCODE_URL).mock(
            return_value=httpx.Response(
                200, json={"results": [{"name": "Bad", "latitude": 999.0, "longitude": 0.0}]}
            )
        )
        result = await OpenMeteoGeocodingProvider(client).geocode("Somewhere")
        assert result.ok is False

    async def test_empty_place_name_short_circuits(self, client):
        result = await OpenMeteoGeocodingProvider(client).geocode("   ")
        assert result.ok is False

    async def test_http_failure_degrades(self, client, respx_mock):
        respx_mock.get(GEOCODE_URL).mock(return_value=httpx.Response(503))
        result = await OpenMeteoGeocodingProvider(client).geocode("Prague")
        assert result.ok is False
        assert result.origin.value == "unavailable"
