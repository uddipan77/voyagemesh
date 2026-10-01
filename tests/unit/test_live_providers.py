"""Provider contracts through real HTTP serialization, with offline supplier responses."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import date, timedelta

import httpx
import pytest

from vm_config.settings import ProviderSettings, Settings
from vm_contracts.common import AccommodationType, Currency, DataOrigin, GeoPoint, Money, utc_now
from vm_contracts.mcp import ToolResult
from vm_contracts.trip import NormalizedTripRequest, TripRequest
from vm_destination_mcp.server import build_server as build_destination
from vm_domain.providers.base import AccommodationQuery, POIQuery, TransportQuery
from vm_domain.providers.duffel import DuffelTransportProvider
from vm_domain.providers.geoapify import GeoapifyPOIProvider
from vm_domain.providers.liteapi import LiteAPIAccommodationProvider
from vm_domain.providers.live_support import make_provider_client
from vm_domain.scoring import filter_transport
from vm_harness import InProcessToolClient
from vm_lodging_mcp.server import build_server as build_lodging
from vm_resilience.safe_http import ResponseTooLargeError, SafeHTTPClient, UnsafeURLError
from vm_transport_mcp.server import build_server as build_transport

pytestmark = pytest.mark.unit

FLIGHT_QUERY = TransportQuery(
    origin="LHR",
    destination="CDG",
    departure_date=date(2030, 5, 10),
    return_date=date(2030, 5, 13),
    travellers=2,
    currency=Currency.EUR,
)
STAY_QUERY = AccommodationQuery(
    destination="Paris",
    check_in=date(2030, 5, 10),
    check_out=date(2030, 5, 13),
    guests=2,
    currency=Currency.EUR,
    guest_nationality="IN",
)


@pytest.fixture
async def client():
    instance = make_provider_client(ProviderSettings(_env_file=None))
    yield instance
    await instance.aclose()


def key_file(tmp_path, value):
    path = tmp_path / "test.key"
    path.write_text(value, encoding="utf-8")
    return str(path)


def flight_payload():
    london = {"name": "Heathrow", "iata_code": "LHR", "time_zone": "Europe/London"}
    paris = {"name": "Charles de Gaulle", "iata_code": "CDG", "time_zone": "Europe/Paris"}

    def segment(origin, destination, start, end):
        return {
            "origin": origin,
            "destination": destination,
            "departing_at": start,
            "arriving_at": end,
            "operating_carrier": {"name": "Example Airline", "iata_code": "XX"},
            "operating_carrier_flight_number": "12",
        }

    return {
        "data": {
            "live_mode": True,
            "offers": [
                {
                    "id": "off_contract_example",
                    "live_mode": True,
                    "total_amount": "301.01",
                    "total_currency": "EUR",
                    "expires_at": (utc_now() + timedelta(hours=1)).isoformat(),
                    "passengers": [{"id": "pas_1"}, {"id": "pas_2"}],
                    "slices": [
                        {
                            "segments": [
                                segment(london, paris, "2030-05-10T10:00:00", "2030-05-10T12:15:00")
                            ]
                        },
                        {
                            "segments": [
                                segment(paris, london, "2030-05-13T17:00:00", "2030-05-13T17:20:00")
                            ]
                        },
                    ],
                }
            ],
        }
    }


def hotel_payload():
    return {
        "sandbox": False,
        "hotels": [
            {
                "id": "lp-test",
                "name": "Example Hotel",
                "latitude": 48.85,
                "longitude": 2.35,
                "rating": 8.5,
                "address": "Example address",
            }
        ],
        "data": [
            {
                "hotelId": "lp-test",
                "roomTypes": [
                    {
                        "offerId": "large-offer-token" * 80,
                        "rates": [
                            {
                                "adultCount": 2,
                                "childCount": 0,
                                "retailRate": {
                                    "total": [{"amount": 300, "currency": "EUR"}],
                                    "taxesAndFees": [
                                        {"included": True, "amount": 10, "currency": "EUR"},
                                        {"included": False, "amount": 15, "currency": "EUR"},
                                    ],
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }


def mock_geocode(respx_mock):
    return respx_mock.get("https://geocoding-api.open-meteo.com/v1/search").respond(
        200,
        json={
            "results": [
                {"latitude": 48.85, "longitude": 2.35, "name": "Paris", "country": "France"}
            ]
        },
    )


async def test_duffel_roundtrip_party_total_currency_and_timezones(client, tmp_path, respx_mock):
    token = "duffel_live_contract_test"
    route = respx_mock.post("https://api.duffel.com/air/offer_requests").respond(
        200, json=flight_payload()
    )
    result = await DuffelTransportProvider(client, api_key_file=key_file(tmp_path, token)).search(
        FLIGHT_QUERY
    )
    assert result.ok and result.origin is DataOrigin.LIVE
    offer = result.items[0]
    assert offer.total_price == Money.of("301.01", Currency.EUR)
    assert offer.price_per_traveller == Money.of("150.51", Currency.EUR)
    assert offer.total_duration_minutes == 75
    assert offer.return_duration_minutes == 80
    assert offer.return_transfer_count == offer.transfer_count == 0
    assert offer.expires_at > utc_now()
    request = route.calls.last.request
    assert request.headers["Authorization"] == f"Bearer {token}"
    assert request.headers["Duffel-Version"] == "v2"
    assert token not in str(request.url)
    body = json.loads(request.content)["data"]
    assert len(body["slices"]) == 2 and len(body["passengers"]) == 2
    assert body["slices"][1]["departure_date"] == "2030-05-13"
    # Constraints apply to each journey, not the three days spent at the destination.
    assert filter_transport([offer], max_duration_hours=2, max_transfers=0, accessibility_needs=[])[
        0
    ]
    assert not filter_transport(
        [offer], max_duration_hours=1.3, max_transfers=0, accessibility_needs=[]
    )[0]


async def test_duffel_resolves_city_using_provider_suggestions(client, tmp_path, respx_mock):
    respx_mock.get("https://api.duffel.com/places/suggestions").respond(
        200,
        json={
            "data": [
                {"type": "airport", "name": "Charles de Gaulle", "iata_code": "CDG"},
                {"type": "city", "name": "Paris", "iata_code": "PAR"},
            ]
        },
    )
    route = respx_mock.post("https://api.duffel.com/air/offer_requests").respond(
        200, json=flight_payload()
    )
    result = await DuffelTransportProvider(
        client, api_key_file=key_file(tmp_path, "duffel_live_test")
    ).search(replace(FLIGHT_QUERY, destination="Paris"))
    assert result.ok
    assert json.loads(route.calls.last.request.content)["data"]["slices"][0]["destination"] == "PAR"


@pytest.mark.parametrize("bad", ["currency", "expired", "sandbox", "passengers", "return"])
async def test_duffel_rejects_unusable_quotes(client, tmp_path, respx_mock, bad):
    payload = flight_payload()
    raw = payload["data"]["offers"][0]
    if bad == "currency":
        raw["total_currency"] = "USD"
    if bad == "expired":
        raw["expires_at"] = (utc_now() - timedelta(seconds=1)).isoformat()
    if bad == "sandbox":
        raw["live_mode"] = False
    if bad == "passengers":
        raw["passengers"] = []
    if bad == "return":
        raw["slices"].pop()
    respx_mock.post("https://api.duffel.com/air/offer_requests").respond(200, json=payload)
    result = await DuffelTransportProvider(
        client, api_key_file=key_file(tmp_path, "duffel_live_test")
    ).search(FLIGHT_QUERY)
    assert not result.items and result.warnings


async def test_liteapi_includes_property_fees_once_for_full_stay(client, tmp_path, respx_mock):
    mock_geocode(respx_mock)
    route = respx_mock.post("https://api.liteapi.travel/v3.0/hotels/rates").respond(
        200, json=hotel_payload()
    )
    result = await LiteAPIAccommodationProvider(
        client, api_key_file=key_file(tmp_path, "prod_test")
    ).search(STAY_QUERY)
    assert result.ok and result.origin is DataOrigin.LIVE
    offer = result.items[0]
    assert offer.total_price == Money.of(315, Currency.EUR)
    assert offer.price_per_night == Money.of(105, Currency.EUR)
    assert offer.guests == 2 and offer.nights == 3
    assert offer.cancellation is None and not offer.accessibility_features
    assert len(offer.offer_id) <= 120
    body = json.loads(route.calls.last.request.content)
    assert body["guestNationality"] == "IN" and body["occupancies"] == [{"adults": 2}]
    assert body["currency"] == "EUR" and body["includeHotelData"] is True


async def test_liteapi_discovers_hotel_type_ids(client, tmp_path, respx_mock):
    mock_geocode(respx_mock)
    respx_mock.get("https://api.liteapi.travel/v3.0/data/hotelTypes").respond(
        200, json={"data": [{"id": 201, "name": "Hostel"}, {"id": 204, "name": "Hotel"}]}
    )
    route = respx_mock.post("https://api.liteapi.travel/v3.0/hotels/rates").respond(
        200, json=hotel_payload()
    )
    result = await LiteAPIAccommodationProvider(
        client, api_key_file=key_file(tmp_path, "prod_test")
    ).search(replace(STAY_QUERY, accommodation_type=AccommodationType.HOSTEL))
    assert result.items[0].accommodation_type is AccommodationType.HOSTEL
    assert json.loads(route.calls.last.request.content)["hotelTypeIds"] == [201]


@pytest.mark.parametrize("bad", ["currency", "fee_currency", "occupancy", "metadata", "sandbox"])
async def test_liteapi_rejects_misleading_quotes(client, tmp_path, respx_mock, bad):
    mock_geocode(respx_mock)
    payload = hotel_payload()
    rate = payload["data"][0]["roomTypes"][0]["rates"][0]
    if bad == "currency":
        rate["retailRate"]["total"][0]["currency"] = "USD"
    if bad == "fee_currency":
        rate["retailRate"]["taxesAndFees"][1]["currency"] = "USD"
    if bad == "occupancy":
        rate["adultCount"] = 1
    if bad == "metadata":
        payload["hotels"] = []
    if bad == "sandbox":
        payload["sandbox"] = True
    respx_mock.post("https://api.liteapi.travel/v3.0/hotels/rates").respond(200, json=payload)
    result = await LiteAPIAccommodationProvider(
        client, api_key_file=key_file(tmp_path, "prod_test")
    ).search(STAY_QUERY)
    assert not result.items


async def test_liteapi_requires_nationality_before_calling_api(client, tmp_path, respx_mock):
    result = await LiteAPIAccommodationProvider(
        client, api_key_file=key_file(tmp_path, "prod_test")
    ).search(replace(STAY_QUERY, guest_nationality=None))
    assert not result.ok and "nationality" in result.error
    assert not respx_mock.calls


async def test_geoapify_real_places_unknown_prices_and_header_auth(client, tmp_path, respx_mock):
    feature = {
        "properties": {
            "name": "Example Museum",
            "place_id": "id" * 200,
            "lat": 48.85,
            "lon": 2.35,
            "categories": ["entertainment.museum"],
        }
    }
    route = respx_mock.post("https://api.geoapify.com/v2/places").respond(
        200,
        json={
            "features": [feature, copy.deepcopy(feature), {"properties": {"name": "incomplete"}}],
        },
    )
    token = "geoapify_contract_key"
    result = await GeoapifyPOIProvider(client, api_key_file=key_file(tmp_path, token)).search(
        POIQuery(
            destination="Paris",
            location=GeoPoint(latitude=48.85, longitude=2.35),
            interests=("history",),
        )
    )
    assert result.ok and len(result.items) == 1
    poi = result.items[0]
    assert poi.provenance.origin is DataOrigin.LIVE
    assert poi.admission_price is None and not poi.is_free and not poi.opening_hours
    assert "OpenStreetMap" in poi.provenance.license_note
    assert route.calls.last.request.headers["x-api-key"] == token
    assert token not in str(route.calls.last.request.url)
    assert "incomplete" in " ".join(result.warnings)


@pytest.mark.parametrize(
    "provider,query",
    [
        (DuffelTransportProvider, FLIGHT_QUERY),
        (LiteAPIAccommodationProvider, STAY_QUERY),
        (GeoapifyPOIProvider, POIQuery(destination="Paris")),
    ],
)
async def test_missing_keys_do_not_call_network_or_fabricate(client, respx_mock, provider, query):
    result = await provider(client, api_key_file=None).search(query)
    assert not result.ok and not result.items and result.origin is DataOrigin.UNAVAILABLE
    assert not respx_mock.calls


@pytest.mark.parametrize(
    "provider,query,token",
    [
        (DuffelTransportProvider, FLIGHT_QUERY, "duffel_test_example"),
        (LiteAPIAccommodationProvider, STAY_QUERY, "sand_example"),
    ],
)
async def test_sandbox_keys_are_refused(client, tmp_path, respx_mock, provider, query, token):
    result = await provider(client, api_key_file=key_file(tmp_path, token)).search(query)
    assert not result.ok and not respx_mock.calls


@pytest.mark.parametrize("status", [401, 403, 429, 500])
async def test_error_bodies_and_keys_are_not_exposed_or_retried(
    client, tmp_path, respx_mock, status
):
    token = "duffel_live_secret_that_must_not_leak"
    route = respx_mock.post("https://api.duffel.com/air/offer_requests").respond(status, text=token)
    result = await DuffelTransportProvider(client, api_key_file=key_file(tmp_path, token)).search(
        FLIGHT_QUERY
    )
    assert not result.ok and token not in result.error
    assert route.call_count == 1


@pytest.mark.parametrize(
    "build,tool,args",
    [
        (
            build_transport,
            "search_flights",
            {"origin": "LHR", "destination": "CDG", "departure_date": "2030-05-10"},
        ),
        (
            build_transport,
            "search_ground_transport",
            {"origin": "Berlin", "destination": "Paris", "departure_date": "2030-05-10"},
        ),
        (
            build_lodging,
            "search_accommodation",
            {"destination": "Paris", "check_in": "2030-05-10", "check_out": "2030-05-13"},
        ),
        (build_destination, "search_points_of_interest", {"destination": "Prague"}),
    ],
)
@pytest.mark.parametrize("mode", ["live", "auto"])
async def test_live_mcp_never_falls_back_to_mock(build, tool, args, mode, respx_mock):
    settings = Settings.for_testing(PROVIDER_MODE=mode, DB_ENABLED=False)
    server = build(settings)
    app = server.streamable_http_app()
    async with app.router.lifespan_context(app):
        result = await InProcessToolClient(server).call(tool, args)
    assert isinstance(result, ToolResult)
    assert not result.ok and result.origin == "unavailable"
    assert not respx_mock.calls


@pytest.mark.parametrize(
    "tool,args,url",
    [
        (
            "geocode_destination",
            {"place": "Prague"},
            "https://geocoding-api.open-meteo.com/v1/search",
        ),
        (
            "get_weather_forecast",
            {
                "location": {"latitude": 50, "longitude": 14},
                "start_date": "2030-05-10",
                "end_date": "2030-05-13",
            },
            "https://api.open-meteo.com/v1/forecast",
        ),
    ],
)
async def test_live_destination_failures_never_return_fixture(tool, args, url, respx_mock):
    respx_mock.get(url).respond(503)
    server = build_destination(Settings.for_testing(PROVIDER_MODE="live", DB_ENABLED=False))
    app = server.streamable_http_app()
    async with app.router.lifespan_context(app):
        result = await InProcessToolClient(server).call(tool, args)
    assert not result.ok and result.origin == "unavailable"


@pytest.mark.filterwarnings("ignore:Unclosed <MemoryObjectReceiveStream:ResourceWarning")
async def test_provider_pool_survives_http_initialization_and_multiple_tool_calls(
    client,
    monkeypatch,
    respx_mock,
):
    """Regression for the actual stateless MCP HTTP path, including discovery.

    FastMCP executes its protocol lifespan for each HTTP request. It must not close
    the provider pool after initialize/list_tools or after the first tool call.
    """
    import gc

    from vm_destination_mcp import server as destination_module

    monkeypatch.setattr(destination_module, "make_provider_client", lambda settings: client)
    route = mock_geocode(respx_mock)
    server = build_destination(Settings.for_testing(PROVIDER_MODE="live", DB_ENABLED=False))
    app = server.streamable_http_app()
    headers = {"Accept": "application/json, text/event-stream"}
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "lifespan-regression", "version": "1.0"},
            },
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        *[
            {
                "jsonrpc": "2.0",
                "id": i,
                "method": "tools/call",
                "params": {
                    "name": "geocode_destination",
                    "arguments": {"place": "Paris"},
                },
            }
            for i in (3, 4)
        ],
    ]
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost:8130",
        ) as http,
    ):
        for message in messages:
            response = await http.post("/mcp", json=message, headers=headers)
            assert response.status_code == 200
            if message["method"] == "tools/call":
                event = next(
                    line[6:] for line in response.text.splitlines() if line.startswith("data: ")
                )
                result = ToolResult.model_validate(json.loads(event)["result"]["structuredContent"])
                assert result.ok and result.origin == "live"
                assert result.data["latitude"] == 48.85
            assert not client._client.is_closed
    assert client._client.is_closed
    assert route.call_count == 2
    # The installed MCP SDK leaves stateless request receive streams to GC. Scope
    # that SDK warning to this HTTP test; the provider pool must close explicitly.
    gc.collect()


async def test_post_has_same_ssrf_redirect_size_guards_and_no_retries(respx_mock):
    async with SafeHTTPClient(
        allowed_hosts=["api.duffel.com"], max_response_bytes=100, max_retries=2
    ) as client:
        with pytest.raises(UnsafeURLError):
            await client.post_json("https://localhost/private", json={})
        route = respx_mock.post("https://api.duffel.com/search")
        route.respond(302, headers={"Location": "https://localhost/private"})
        with pytest.raises(UnsafeURLError):
            await client.post_json("https://api.duffel.com/search", json={})
        route.respond(200, text="x" * 101)
        with pytest.raises(ResponseTooLargeError):
            await client.post_json("https://api.duffel.com/search", json={})
        route.respond(500)
        with pytest.raises(httpx.HTTPStatusError):
            await client.post_json("https://api.duffel.com/search", json={})
        assert route.call_count == 3


def test_nationality_is_carried_and_distinguishes_cache_identity():
    request = TripRequest(
        origin="Berlin",
        destination="Paris",
        departure_date="2030-05-10",
        return_date="2030-05-13",
        max_budget=Money.of(500, Currency.EUR),
        guest_nationality="IN",
    )
    normalized = NormalizedTripRequest.from_request(request)
    other = NormalizedTripRequest.from_request(
        request.model_copy(update={"guest_nationality": "DE"})
    )
    assert normalized.guest_nationality == "IN"
    assert normalized.cache_fingerprint() != other.cache_fingerprint()


async def test_live_quotes_survive_agents_orchestrator_and_budget(tmp_path, respx_mock):
    """Exercise discovery, MCP, all agents and final source labels, with HTTP fixtures."""
    from contextlib import AsyncExitStack

    from vm_contracts.budget import BudgetCategory, BudgetStatus
    from vm_harness import InProcessAgentClient
    from vm_itinerary_agent.agent import ItineraryAgent
    from vm_llm.mock_provider import MockLLMProvider
    from vm_orchestrator import OrchestratorDependencies, TravelOrchestrator
    from vm_stay_agent.agent import StayAgent
    from vm_transport_agent.agent import TransportAgent

    for provider, token in [
        ("duffel", "duffel_live_test"),
        ("liteapi", "prod_test"),
        ("geoapify", "geo_test"),
    ]:
        (tmp_path / f"{provider}.key").write_text(token, encoding="utf-8")
    settings = Settings.for_testing(
        PROVIDER_MODE="live",
        DB_ENABLED=False,
        PROVIDER_DUFFEL_API_KEY_FILE=str(tmp_path / "duffel.key"),
        PROVIDER_LITEAPI_API_KEY_FILE=str(tmp_path / "liteapi.key"),
        PROVIDER_GEOAPIFY_API_KEY_FILE=str(tmp_path / "geoapify.key"),
    )
    mock_geocode(respx_mock)
    respx_mock.post("https://api.duffel.com/air/offer_requests").respond(200, json=flight_payload())
    respx_mock.post("https://api.liteapi.travel/v3.0/hotels/rates").respond(
        200, json=hotel_payload()
    )
    respx_mock.post("https://api.geoapify.com/v2/places").respond(
        200,
        json={
            "features": [
                {
                    "properties": {
                        "name": "Example Museum",
                        "place_id": "place_example",
                        "lat": 48.85,
                        "lon": 2.35,
                        "categories": ["entertainment.museum"],
                    }
                }
            ]
        },
    )
    respx_mock.get("https://api.open-meteo.com/v1/forecast").respond(503)
    servers = [build_transport(settings), build_lodging(settings), build_destination(settings)]
    async with AsyncExitStack() as stack:
        for server in servers:
            app = server.streamable_http_app()
            await stack.enter_async_context(app.router.lifespan_context(app))
        agents = [
            kind(tool_client=InProcessToolClient(server), llm=MockLLMProvider())
            for kind, server in zip(
                (TransportAgent, StayAgent, ItineraryAgent), servers, strict=True
            )
        ]
        deps = OrchestratorDependencies(
            settings=settings,
            transport_client=InProcessAgentClient(agents[0]),
            stay_client=InProcessAgentClient(agents[1]),
            itinerary_client=InProcessAgentClient(agents[2]),
        )
        request = TripRequest(
            origin="LHR",
            destination="CDG",
            departure_date="2030-05-10",
            return_date="2030-05-13",
            travellers=2,
            guest_nationality="IN",
            max_budget=Money.of(1000, Currency.EUR),
            transport_preference="flight",
            interests=["history"],
        )
        plan = await TravelOrchestrator(deps).plan_trip(request)
    assert plan.transport and plan.transport.recommended
    assert plan.transport.recommended.return_legs
    assert plan.transport.recommended.total_price == Money.of("301.01", Currency.EUR)
    assert plan.accommodation.recommended.total_price == Money.of(315, Currency.EUR)
    sources = {source.component: source for source in plan.data_sources}
    assert sources["transport"].source_name == "Duffel Flights"
    assert sources["accommodation"].source_name == "LiteAPI Hotels"
    assert sources["itinerary"].source_name == "Geoapify Places"
    assert all(
        sources[k].origin is DataOrigin.LIVE for k in ("transport", "accommodation", "itinerary")
    )
    assert plan.budget.status is BudgetStatus.INCOMPLETE
    assert BudgetCategory.ACTIVITIES in plan.budget.missing_components
    assert plan.replans == 0  # Missing admission prices must not repeat paid searches.
    assert any("OpenStreetMap" in w for w in plan.warnings)
    assert not plan.validation_errors
