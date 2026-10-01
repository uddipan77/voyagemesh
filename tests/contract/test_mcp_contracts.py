"""MCP contract tests.

These exist because the agent and the server run in different containers. They import the
*same* argument models from ``vm_contracts.mcp`` and drive the servers in-process, so a
change that would break the agent/server pair fails here rather than at runtime across a
network boundary (brief §8).

Every tool is exercised through ``call_tool`` — the real MCP dispatch path — not by calling
the handler directly, so the argument-validation and result-envelope wrapping are part of
what is tested.
"""

from __future__ import annotations

from typing import Any

import pytest

from vm_config.settings import Settings
from vm_contracts.mcp import (
    AccommodationSearchArgs,
    DistanceArgs,
    GeocodeArgs,
    KnowledgeArgs,
    POISearchArgs,
    StayCostArgs,
    ToolResult,
    TransportSearchArgs,
    ValidateScheduleArgs,
    WeatherArgs,
)
from vm_destination_mcp.server import build_server as build_destination
from vm_lodging_mcp.server import build_server as build_lodging
from vm_transport_mcp.server import build_server as build_transport

pytestmark = [pytest.mark.contract]


def _mock_settings() -> Settings:
    """Settings that keep every server offline — no live API calls in the contract suite.

    ``Settings.for_testing`` rather than ``Settings(PROVIDER_MODE=...)``: a kwarg on the
    root never reaches the nested ``providers`` group, so the latter would silently leave
    the mode at its ``auto`` default and every destination server would build a live HTTP
    client. That is the exact bug this helper exists to prevent.
    """
    return Settings.for_testing(PROVIDER_MODE="mock", ENVIRONMENT="test")


def _unwrap(result: Any) -> dict[str, Any]:
    """`call_tool` returns `(content_blocks, structured_dict)`; we assert on the structured
    dict, which is what an A2A caller consumes."""
    structured = result[1] if isinstance(result, tuple) else result
    assert isinstance(structured, dict)
    return structured


async def _call(server: Any, name: str, args: Any) -> ToolResult:
    payload = args.model_dump(mode="json") if hasattr(args, "model_dump") else args
    raw = await server.call_tool(name, payload)
    # Re-validate through the shared contract so the test fails if the server ever returns
    # a shape the agent could not parse.
    return ToolResult.model_validate(_unwrap(raw))


# ---------------------------------------------------------------------------
# Advertised surface
# ---------------------------------------------------------------------------
class TestAdvertisedTools:
    async def test_transport_advertises_its_five_tools(self):
        server = build_transport(_mock_settings())
        names = {t.name for t in await server.list_tools()}
        assert names == {
            "search_ground_transport",
            "search_flights",
            "normalize_transport_offer",
            "calculate_transport_score",
            "validate_transport_offer",
        }

    async def test_lodging_advertises_its_five_tools(self):
        server = build_lodging(_mock_settings())
        names = {t.name for t in await server.list_tools()}
        assert names == {
            "search_accommodation",
            "calculate_total_stay_cost",
            "calculate_distance_score",
            "normalize_accommodation_offer",
            "validate_accommodation_offer",
        }

    async def test_destination_advertises_its_six_tools(self):
        server = build_destination(_mock_settings())
        names = {t.name for t in await server.list_tools()}
        assert names == {
            "geocode_destination",
            "search_points_of_interest",
            "get_weather_forecast",
            "retrieve_destination_knowledge",
            "calculate_location_distance",
            "validate_daily_schedule",
        }

    async def test_tools_advertise_typed_parameters_not_opaque_kwargs(self):
        """Regression: a **kwargs wrapper advertised a single 'kwargs' field and every
        call failed validation before reaching the handler."""
        server = build_transport(_mock_settings())
        tool = next(t for t in await server.list_tools() if t.name == "search_ground_transport")
        properties = set(tool.inputSchema.get("properties", {}))
        assert {"origin", "destination", "departure_date"} <= properties
        assert "kwargs" not in properties


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
class TestTransportContract:
    async def test_search_returns_labelled_ranked_offers(self):
        server = build_transport(_mock_settings())
        result = await _call(
            server,
            "search_ground_transport",
            TransportSearchArgs(
                origin="Nuremberg",
                destination="Prague",
                departure_date="2026-08-10",
                max_duration_hours=8,
                ranking_strategy="balanced",
            ),
        )
        assert result.ok
        assert result.origin == "mocked"
        offers = result.data["offers"]
        assert offers
        scores = [o["score"] for o in offers]
        assert scores == sorted(scores), "offers must come back ranked best-first"

    async def test_invalid_arguments_return_a_structured_failure(self):
        """A bad call must produce a ToolResult an agent can reason about, not an exception."""
        server = build_transport(_mock_settings())
        raw = await server.call_tool(
            "search_ground_transport",
            {"origin": "X", "destination": "Prague", "departure_date": "2026-08-10"},
        )
        result = ToolResult.model_validate(_unwrap(raw))
        assert result.ok is False
        assert result.error is not None
        assert result.error.code == "invalid_arguments"

    async def test_scoring_is_deterministic_across_servers(self):
        """Two independently built servers must rank an identical offer set identically."""
        server_a = build_transport(_mock_settings())
        server_b = build_transport(_mock_settings())
        search = await _call(
            server_a,
            "search_ground_transport",
            TransportSearchArgs(
                origin="Nuremberg", destination="Prague", departure_date="2026-08-10"
            ),
        )
        offers = search.data["offers"]

        args = {"offers": offers, "ranking_strategy": "cheapest"}
        first = ToolResult.model_validate(
            _unwrap(await server_a.call_tool("calculate_transport_score", args))
        )
        second = ToolResult.model_validate(
            _unwrap(await server_b.call_tool("calculate_transport_score", args))
        )
        assert [o["offer_id"] for o in first.data["offers"]] == [
            o["offer_id"] for o in second.data["offers"]
        ]

    async def test_malformed_offers_are_dropped_not_fatal(self):
        server = build_transport(_mock_settings())
        raw = await server.call_tool(
            "validate_transport_offer", {"offers": [{"nonsense": True}, {"also": "bad"}]}
        )
        result = ToolResult.model_validate(_unwrap(raw))
        assert result.ok
        assert result.data["valid_count"] == 0
        assert result.data["invalid_count"] == 2


# ---------------------------------------------------------------------------
# Lodging
# ---------------------------------------------------------------------------
class TestLodgingContract:
    async def test_search_returns_stay_priced_offers(self):
        server = build_lodging(_mock_settings())
        result = await _call(
            server,
            "search_accommodation",
            AccommodationSearchArgs(
                destination="Prague",
                check_in="2026-08-10",
                check_out="2026-08-13",
                guests=1,
                accommodation_type="hostel",
            ),
        )
        assert result.ok
        offer = result.data["offers"][0]
        assert offer["nights"] == 3
        # total must equal nightly x nights — the tool, not a prompt, guarantees this
        assert float(offer["total_price"]["amount"]) == pytest.approx(
            float(offer["price_per_night"]["amount"]) * 3
        )

    async def test_total_stay_cost_is_pure_arithmetic(self):
        server = build_lodging(_mock_settings())
        result = await _call(
            server,
            "calculate_total_stay_cost",
            StayCostArgs(price_per_night=42.5, nights=3, rooms=2),
        )
        assert result.ok
        assert result.origin == "computed"
        assert float(result.data["total_price"]["amount"]) == pytest.approx(42.5 * 2 * 3)

    async def test_distance_score_bounds(self):
        server = build_lodging(_mock_settings())
        result = await _call(
            server,
            "calculate_distance_score",
            DistanceArgs(
                origin={"latitude": 50.0875, "longitude": 14.4213},
                destination={"latitude": 50.0875, "longitude": 14.4213},
            ),
        )
        assert result.ok
        assert result.data["distance_km"] == pytest.approx(0.0, abs=1e-6)
        assert result.data["distance_penalty"] == 0.0

    async def test_normalize_and_validate_round_trip_offers(self):
        """An offer produced by search must survive normalize and validate — the exact
        serialise/re-parse path an agent walks between tools."""
        server = build_lodging(_mock_settings())
        search = await _call(
            server,
            "search_accommodation",
            AccommodationSearchArgs(
                destination="Prague", check_in="2026-08-10", check_out="2026-08-13"
            ),
        )
        offers = search.data["offers"]

        normalized = await _call(server, "normalize_accommodation_offer", {"offers": offers})
        assert normalized.ok
        assert normalized.data["rejected_count"] == 0

        validated = await _call(server, "validate_accommodation_offer", {"offers": offers})
        assert validated.ok
        assert validated.data["valid_count"] == len(offers)
        assert validated.data["invalid_count"] == 0


# ---------------------------------------------------------------------------
# Destination
# ---------------------------------------------------------------------------
class TestDestinationContract:
    async def test_poi_search_returns_curated_attractions(self):
        server = build_destination(_mock_settings())
        result = await _call(
            server,
            "search_points_of_interest",
            POISearchArgs(destination="Prague", interests=["history"]),
        )
        assert result.ok
        assert result.origin == "fixture"
        assert result.data["count"] > 0

    async def test_geocode_falls_back_to_a_curated_centre_in_mock_mode(self):
        server = build_destination(_mock_settings())
        result = await _call(server, "geocode_destination", GeocodeArgs(place="Prague"))
        assert result.ok
        assert 49 < result.data["latitude"] < 51

    async def test_weather_in_mock_mode_is_labelled_not_a_forecast(self):
        server = build_destination(_mock_settings())
        result = await _call(
            server,
            "get_weather_forecast",
            WeatherArgs(
                location={"latitude": 50.0875, "longitude": 14.4213},
                start_date="2026-08-10",
                end_date="2026-08-13",
            ),
        )
        assert result.ok
        assert result.data["forecast"]["provenance"]["origin"] == "mocked"
        assert "not a forecast" in (result.data["forecast"]["note"] or "").lower()

    async def test_knowledge_returns_honest_empty_until_rag_is_wired(self):
        server = build_destination(_mock_settings())
        result = await _call(
            server,
            "retrieve_destination_knowledge",
            KnowledgeArgs(destination="Prague", query="accessibility of the old town"),
        )
        assert result.ok
        assert result.data["documents"] == []
        assert result.warnings

    async def test_validate_schedule_detects_overlap(self):
        server = build_destination(_mock_settings())
        result = await _call(
            server,
            "validate_daily_schedule",
            ValidateScheduleArgs(
                slots=[
                    {"title": "A", "start_minute": 540, "end_minute": 720},
                    {"title": "B", "start_minute": 660, "end_minute": 780},
                ]
            ),
        )
        assert result.ok
        assert result.data["feasible"] is False
        assert any("overlap" in p for p in result.data["problems"])

    async def test_validate_schedule_detects_missing_travel_time(self):
        server = build_destination(_mock_settings())
        result = await _call(
            server,
            "validate_daily_schedule",
            ValidateScheduleArgs(
                slots=[
                    {
                        "title": "A",
                        "start_minute": 540,
                        "end_minute": 600,
                        "travel_minutes_to_next": 45,
                    },
                    {"title": "B", "start_minute": 610, "end_minute": 700},
                ]
            ),
        )
        assert result.ok
        assert result.data["feasible"] is False

    async def test_validate_schedule_accepts_a_feasible_day(self):
        server = build_destination(_mock_settings())
        result = await _call(
            server,
            "validate_daily_schedule",
            ValidateScheduleArgs(
                slots=[
                    {
                        "title": "A",
                        "start_minute": 540,
                        "end_minute": 600,
                        "travel_minutes_to_next": 15,
                    },
                    {"title": "B", "start_minute": 620, "end_minute": 700},
                ]
            ),
        )
        assert result.ok
        assert result.data["feasible"] is True

    async def test_location_distance_reports_km_and_travel_time(self):
        server = build_destination(_mock_settings())
        result = await _call(
            server,
            "calculate_location_distance",
            DistanceArgs(
                origin={"latitude": 50.0875, "longitude": 14.4213},
                destination={"latitude": 50.0900, "longitude": 14.4000},
            ),
        )
        assert result.ok
        assert result.data["distance_km"] > 0
        assert result.data["travel_minutes"] >= 1

    async def test_geocode_unknown_place_in_mock_mode_reports_not_found(self):
        server = build_destination(_mock_settings())
        raw = await server.call_tool("geocode_destination", {"place": "Zzyzxville"})
        result = ToolResult.model_validate(_unwrap(raw))
        # No live API and no curated centre — an honest failure, never a guessed coordinate.
        assert result.ok is False
        assert result.error is not None


class TestDestinationLiveMode:
    """The live provider path, exercised offline with respx.

    In mock mode the live branches never run, so they sat untested. Here the destination
    server is built in ``auto`` mode with the Open-Meteo hosts mocked, so the live geocode
    and weather branches execute against a scripted response rather than the network.
    """

    def _live_settings(self) -> Settings:
        return Settings.for_testing(PROVIDER_MODE="auto", ENVIRONMENT="test")

    async def _close(self, server: Any) -> None:
        client = getattr(server, "voyagemesh_live_client", None)
        if client is not None:
            await client.aclose()

    async def test_live_geocode_path(self, respx_mock):
        import httpx

        respx_mock.get("https://geocoding-api.open-meteo.com/v1/search").mock(
            return_value=httpx.Response(
                200,
                json={"results": [{"name": "Prague", "latitude": 50.088, "longitude": 14.42}]},
            )
        )
        server = build_destination(self._live_settings())
        try:
            result = await _call(server, "geocode_destination", GeocodeArgs(place="Prague"))
            assert result.ok
            assert result.origin == "live"
            assert 50 < result.data["latitude"] < 51
        finally:
            await self._close(server)

    async def test_live_weather_path(self, respx_mock):
        import httpx

        daily = {
            "time": ["2026-08-10", "2026-08-11"],
            "weather_code": [0, 61],
            "temperature_2m_max": [26.0, 21.0],
            "temperature_2m_min": [15.0, 14.0],
            "precipitation_sum": [0.0, 4.8],
            "precipitation_probability_max": [0, 60],
            "wind_speed_10m_max": [6.0, 18.0],
        }
        respx_mock.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json={"daily": daily})
        )
        server = build_destination(self._live_settings())
        try:
            result = await _call(
                server,
                "get_weather_forecast",
                WeatherArgs(
                    location={"latitude": 50.0875, "longitude": 14.4213},
                    start_date="2026-08-10",
                    end_date="2026-08-11",
                    location_name="Prague",
                ),
            )
            assert result.ok
            assert result.data["forecast"]["provenance"]["origin"] == "live"
        finally:
            await self._close(server)


# ---------------------------------------------------------------------------
# Cross-cutting guarantees
# ---------------------------------------------------------------------------
class TestSharedGuarantees:
    async def test_every_result_validates_against_the_shared_envelope(self):
        """The contract itself: an agent parses every result as ToolResult and it works."""
        server = build_transport(_mock_settings())
        for name, payload in [
            (
                "search_ground_transport",
                {"origin": "Nuremberg", "destination": "Prague", "departure_date": "2026-08-10"},
            ),
            (
                "search_flights",
                {"origin": "Nuremberg", "destination": "Prague", "departure_date": "2026-08-10"},
            ),
        ]:
            result = ToolResult.model_validate(_unwrap(await server.call_tool(name, payload)))
            assert result.origin
            assert result.source_name

    async def test_tool_budget_bounds_runaway_calls(self):
        """A tool cannot be called without limit — the per-session budget stops a loop."""
        settings = Settings.for_testing(PROVIDER_MODE="mock")
        settings.mcp.max_tool_calls_per_task = 1  # budget becomes 1 * 8 = 8
        server = build_transport(settings)
        registry = server.voyagemesh_registry

        payload = {"origin": "Nuremberg", "destination": "Prague", "departure_date": "2026-08-10"}
        results = [
            ToolResult.model_validate(
                _unwrap(await server.call_tool("search_ground_transport", payload))
            )
            for _ in range(10)
        ]
        assert results[0].ok
        assert results[-1].ok is False
        assert results[-1].error is not None
        assert results[-1].error.code == "tool_budget_exceeded"
        assert registry.call_count("search_ground_transport") == 10
