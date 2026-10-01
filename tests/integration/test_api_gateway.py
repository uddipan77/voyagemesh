"""The API gateway, driven through a real ASGI stack.

Composes the gateway over an in-process orchestrator (real graph + real agents + mock LLM), a
fakeredis-backed rate limiter / idempotency store, and a real RS256 JWT validator with a
self-signed key. No network, no Docker — but every request crosses the whole gateway:
middleware, auth, rate limit, idempotency, planning, and the contract-shaped error handlers.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fakeredis.aioredis import FakeRedis
from httpx import ASGITransport, AsyncClient

from vm_api_gateway import GatewayServices, create_app
from vm_auth import JwksClient, JwtValidator
from vm_auth.fastapi_deps import AuthDependencies
from vm_caching import RedisClient
from vm_config.settings import Settings
from vm_contracts.common import AccommodationType, Currency, Money, RankingStrategy
from vm_contracts.plan import PlanStatus, TripPlan
from vm_contracts.trip import TripRequest
from vm_destination_mcp.server import build_server as build_destination
from vm_harness import InProcessAgentClient, InProcessToolClient
from vm_itinerary_agent.agent import ItineraryAgent
from vm_llm.mock_provider import MockLLMProvider
from vm_lodging_mcp.server import build_server as build_lodging
from vm_orchestrator import OrchestratorDependencies, TravelOrchestrator
from vm_stay_agent.agent import StayAgent
from vm_transport_agent.agent import TransportAgent
from vm_transport_mcp.server import build_server as build_transport

pytestmark = pytest.mark.integration

ISSUER = "http://keycloak:8080/realms/voyagemesh"
AUDIENCE = "voyagemesh-api"
KID = "gw-key-1"


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture(scope="module")
def keypair():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def jwk(keypair):
    from jwt.algorithms import RSAAlgorithm

    j = RSAAlgorithm.to_jwk(keypair.public_key(), as_dict=True)
    j["kid"] = KID
    return j


def _token(keypair, *, roles=("traveller",), sub="user-1"):
    now = int(time.time())
    return jwt.encode(
        {
            "sub": sub,
            "iss": ISSUER,
            "aud": AUDIENCE,
            "exp": now + 3600,
            "iat": now,
            "preferred_username": "alice",
            "realm_access": {"roles": list(roles)},
        },
        keypair,
        algorithm="RS256",
        headers={"kid": KID},
    )


def _settings(**overrides: Any) -> Settings:
    return Settings.for_testing(
        PROVIDER_MODE="mock",
        ENVIRONMENT="test",
        AUTH_ISSUER=ISSUER,
        AUTH_AUDIENCE=AUDIENCE,
        OTEL_ENABLED="false",  # no live span export to a collector during tests
        **overrides,
    )


def _orchestrator(settings: Settings, cache: Any = None) -> TravelOrchestrator:
    llm = MockLLMProvider()
    deps = OrchestratorDependencies(
        settings=settings,
        transport_client=InProcessAgentClient(
            TransportAgent(tool_client=InProcessToolClient(build_transport(settings)), llm=llm)
        ),
        stay_client=InProcessAgentClient(
            StayAgent(tool_client=InProcessToolClient(build_lodging(settings)), llm=llm)
        ),
        itinerary_client=InProcessAgentClient(
            ItineraryAgent(tool_client=InProcessToolClient(build_destination(settings)), llm=llm)
        ),
        cache=cache,
    )
    return TravelOrchestrator(deps)


def _services(settings: Settings, jwk, *, redis: RedisClient | None = None) -> GatewayServices:
    redis = redis or RedisClient(settings.redis, redis=FakeRedis())
    validator = JwtValidator(settings.auth, jwks_client=JwksClient("x", static_keys=[jwk]))
    return GatewayServices(
        settings=settings,
        orchestrator=_orchestrator(settings),
        redis=redis,
        auth=AuthDependencies(settings, validator=validator),
        database=None,
    )


class _Harness:
    def __init__(self, settings: Settings, services: GatewayServices, keypair):
        self.app = create_app(settings=settings, services=services)
        self.services = services
        self._keypair = keypair

    def client(self) -> AsyncClient:
        return AsyncClient(transport=ASGITransport(app=self.app), base_url="http://gw")

    def auth(self, *, roles=("traveller",)) -> dict[str, str]:
        return {"Authorization": f"Bearer {_token(self._keypair, roles=roles)}"}


@pytest.fixture
def harness(jwk, keypair):
    settings = _settings()
    return _Harness(settings, _services(settings, jwk), keypair)


def _body(**overrides: Any) -> dict[str, Any]:
    request = TripRequest(
        origin="Nuremberg",
        destination="Prague",
        departure_date=date(2026, 8, 10),
        return_date=date(2026, 8, 13),
        travellers=1,
        max_budget=Money.of(350, Currency.EUR),
        accommodation_preference=AccommodationType.HOSTEL,
        interests=["history", "architecture"],
        ranking_strategy=RankingStrategy.BALANCED,
        **overrides,
    )
    return request.model_dump(mode="json")


class TestHealth:
    async def test_liveness_needs_no_auth(self, harness):
        async with harness.client() as c:
            r = await c.get("/health/live")
        assert r.status_code == 200
        assert r.json()["status"] == "alive"

    async def test_readiness_reports_components(self, harness):
        async with harness.client() as c:
            r = await c.get("/health/ready")
        assert r.status_code == 200
        assert "redis" in r.json()["components"]


class TestAuthentication:
    async def test_planning_requires_a_token(self, harness):
        async with harness.client() as c:
            r = await c.post("/api/v1/trips", json=_body())
        assert r.status_code == 401
        assert r.json()["code"] == "unauthenticated"

    async def test_a_valid_token_plans(self, harness):
        async with harness.client() as c:
            r = await c.post("/api/v1/trips", json=_body(), headers=harness.auth())
        assert r.status_code == 200
        plan = TripPlan.model_validate(r.json())
        assert plan.status is PlanStatus.COMPLETE
        assert r.headers["X-Trip-Id"] == plan.trip_id

    async def test_a_forged_token_is_rejected(self, harness):
        attacker = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        forged = _token(attacker, roles=["admin"])
        async with harness.client() as c:
            r = await c.post(
                "/api/v1/trips", json=_body(), headers={"Authorization": f"Bearer {forged}"}
            )
        assert r.status_code == 401

    async def test_wrong_role_is_forbidden(self, jwk, keypair):
        # A token without the traveller role cannot submit a plan.
        settings = _settings()
        harness = _Harness(settings, _services(settings, jwk), keypair)
        async with harness.client() as c:
            r = await c.post(
                "/api/v1/trips", json=_body(), headers=harness.auth(roles=("evaluator",))
            )
        assert r.status_code == 403
        assert r.json()["code"] == "insufficient_role"


class TestValidation:
    async def test_malformed_body_is_a_422_with_problems(self, harness):
        async with harness.client() as c:
            r = await c.post("/api/v1/trips", json={"origin": "X"}, headers=harness.auth())
        assert r.status_code == 422
        payload = r.json()
        assert payload["code"] == "validation_failed"
        assert payload["problems"]

    async def test_a_trace_id_is_echoed(self, harness):
        async with harness.client() as c:
            r = await c.get("/health/live")
        assert r.headers.get("x-request-id")
        assert r.headers.get("traceparent")


class TestIdempotency:
    async def test_a_repeated_key_replays_the_same_plan(self, harness):
        headers = {**harness.auth(), "Idempotency-Key": "abc-123"}
        async with harness.client() as c:
            first = await c.post("/api/v1/trips", json=_body(), headers=headers)
            second = await c.post("/api/v1/trips", json=_body(), headers=headers)
        assert first.status_code == second.status_code == 200
        assert first.json()["trip_id"] == second.json()["trip_id"]
        assert second.headers.get("Idempotency-Replayed") == "true"

    async def test_different_keys_produce_independent_plans(self, harness):
        async with harness.client() as c:
            a = await c.post(
                "/api/v1/trips", json=_body(), headers={**harness.auth(), "Idempotency-Key": "a"}
            )
            b = await c.post(
                "/api/v1/trips", json=_body(), headers={**harness.auth(), "Idempotency-Key": "b"}
            )
        assert a.json()["trip_id"] != b.json()["trip_id"]


class TestRateLimiting:
    async def test_exhausting_the_burst_yields_429(self, jwk, keypair):
        settings = _settings(LIMIT_RATE_LIMIT_REQUESTS_PER_MINUTE=1, LIMIT_RATE_LIMIT_BURST=2)
        harness = _Harness(settings, _services(settings, jwk), keypair)
        limited = None
        async with harness.client() as c:
            for _ in range(6):
                r = await c.post("/api/v1/trips", json=_body(), headers=harness.auth())
                if r.status_code == 429:
                    limited = r
                    break
        assert limited is not None, "burst of 2 should be exhausted within 6 requests"
        assert limited.json()["code"] == "rate_limited"
        assert limited.headers.get("Retry-After")


class TestStatusEndpoint:
    async def test_unknown_trip_is_404(self, harness):
        async with harness.client() as c:
            r = await c.get("/api/v1/trips/trip_does_not_exist", headers=harness.auth())
        assert r.status_code == 404
        assert r.json()["code"] == "trip_not_found"

    async def test_status_requires_auth(self, harness):
        async with harness.client() as c:
            r = await c.get("/api/v1/trips/trip_x")
        assert r.status_code == 401


class TestBodySizeGuard:
    async def test_an_oversized_body_is_413(self, jwk, keypair):
        # A tiny ceiling makes the ordinary request body exceed it; httpx sets a truthful
        # Content-Length, which the guard reads before the body is deserialised.
        settings = _settings(LIMIT_MAX_REQUEST_BYTES=1024)
        harness = _Harness(settings, _services(settings, jwk), keypair)
        # Inflate the raw JSON past the ceiling with filler. The guard reads Content-Length
        # and rejects before the body is ever deserialised, so the extra key never reaches
        # (strict) validation.
        big = {**_body(), "_filler": "x" * 2000}
        async with harness.client() as c:
            r = await c.post("/api/v1/trips", json=big, headers=harness.auth())
        assert r.status_code == 413
        assert r.json()["code"] == "request_too_large"


class TestOpenAPI:
    async def test_schema_is_served_and_names_the_routes(self, harness):
        async with harness.client() as c:
            r = await c.get("/openapi.json")
        assert r.status_code == 200
        paths = r.json()["paths"]
        assert "/api/v1/trips" in paths
        assert "/api/v1/trips/{trip_id}" in paths
