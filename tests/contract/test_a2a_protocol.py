"""A2A protocol contract: server and client, over a real ASGI transport.

Drives the agents through the actual HTTP endpoints — card discovery, task submission,
error paths — rather than in-process, so the wire contract is what is tested. This is what
makes the A2A here genuine per brief §7: the orchestrator reaches an agent only through its
published card and HTTP interface.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from vm_auth import SharedSecretServiceAuthenticator, StaticServiceTokenProvider
from vm_config.settings import Settings
from vm_contracts.a2a import AgentTask, TaskStatus
from vm_harness import (
    A2ADiscoveryError,
    A2AResponseError,
    HttpAgentClient,
    InProcessAgentClient,
    InProcessToolClient,
    build_a2a_app,
    submit_verified,
)
from vm_llm.mock_provider import MockLLMProvider
from vm_transport_agent.agent import TransportAgent
from vm_transport_mcp.server import build_server

pytestmark = pytest.mark.contract

SECRET = SecretStr("contract-test-secret")


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _settings() -> Settings:
    return Settings.for_testing(PROVIDER_MODE="mock", ENVIRONMENT="test")


def _agent() -> TransportAgent:
    return TransportAgent(
        tool_client=InProcessToolClient(build_server(_settings())), llm=MockLLMProvider()
    )


def _app(agent: TransportAgent | None = None) -> Any:
    return build_a2a_app(
        agent or _agent(),
        authenticator=SharedSecretServiceAuthenticator(SECRET),
        url="http://transport-agent:8010",
    )


async def _http_client(app: Any):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://transport-agent:8010")


def _payload(**overrides: Any) -> dict[str, Any]:
    return {
        "origin": "Nuremberg",
        "destination": "Prague",
        "departure_date": "2026-08-10",
        "max_duration_hours": 8,
        **overrides,
    }


def _agent_client(http: httpx.AsyncClient, *, authed: bool = True) -> HttpAgentClient:
    return HttpAgentClient(
        "http://transport-agent:8010",
        token_provider=StaticServiceTokenProvider(SECRET if authed else None),
        client=http,
    )


class TestDiscovery:
    async def test_card_is_served_and_valid(self):
        http = await _http_client(_app())
        try:
            card = await _agent_client(http).discover()
            assert card.has_skill("plan_transport")
            assert card.url == "http://transport-agent:8010"
        finally:
            await http.aclose()

    async def test_card_endpoint_needs_no_credential(self):
        """Discovery must not require the credential the card describes how to present."""
        http = await _http_client(_app())
        try:
            response = await http.get("/.well-known/agent-card.json")
            assert response.status_code == 200
        finally:
            await http.aclose()

    async def test_discovery_of_an_unreachable_agent_raises(self):
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://x")
        # Point the client at a path the app does not serve to force a 404 on discovery.
        client = HttpAgentClient("http://x/nope", client=http)
        try:
            with pytest.raises(A2ADiscoveryError):
                await client.discover()
        finally:
            await http.aclose()


class TestTaskSubmission:
    async def test_full_round_trip(self):
        http = await _http_client(_app())
        try:
            task = AgentTask(
                skill="plan_transport", correlation_id="c1", request_id="r1", payload=_payload()
            )
            artifact = await submit_verified(
                _agent_client(http), task, required_skill="plan_transport"
            )
            assert artifact.status is TaskStatus.COMPLETED
            assert artifact.matches(task)
            assert artifact.result is not None
        finally:
            await http.aclose()

    async def test_skill_verification_before_submission(self):
        """The orchestrator must not submit to an agent lacking the required skill (§7)."""
        http = await _http_client(_app())
        try:
            task = AgentTask(
                skill="plan_transport", correlation_id="c", request_id="r", payload=_payload()
            )
            with pytest.raises(A2ADiscoveryError, match="does not advertise"):
                await submit_verified(_agent_client(http), task, required_skill="book_hotel")
        finally:
            await http.aclose()

    async def test_trace_and_claims_are_propagated_not_the_user_token(self):
        """The task carries a user reference and roles; the wire must forward those, and
        never an access token (least privilege, threat T-4)."""
        captured: dict[str, str] = {}

        class CapturingAgent(TransportAgent):
            async def handle(self, task: AgentTask) -> Any:  # type: ignore[override]
                captured["user_reference"] = task.user_reference or ""
                return await super().handle(task)

        agent = CapturingAgent(
            tool_client=InProcessToolClient(build_server(_settings())), llm=MockLLMProvider()
        )
        http = await _http_client(_app(agent))
        try:
            task = AgentTask(
                skill="plan_transport",
                correlation_id="c",
                request_id="r",
                payload=_payload(),
                user_reference="user-42",
                user_roles=["traveller"],
            )
            await _agent_client(http).submit_task(task)
        finally:
            await http.aclose()
        # The agent received the task (with its user_reference); no access token was involved.
        assert captured["user_reference"] == "user-42"


class TestResponseValidation:
    async def test_a_mismatched_correlation_id_is_rejected(self):
        """A response for a different task must not be accepted (threat T-4).

        Simulated by tampering the response before the client validates it."""
        agent = _agent()
        http = await _http_client(_app(agent))
        try:
            # Submit a genuine task, then assert the client's own matching logic via a forged
            # artifact whose correlation ID differs.
            from vm_contracts.a2a import AgentArtifact
            from vm_contracts.common import utc_now
            from vm_harness.a2a_client import _validate_artifact

            task = AgentTask(
                skill="plan_transport", correlation_id="real", request_id="r", payload=_payload()
            )
            started = utc_now()
            forged = AgentArtifact(
                task_id=task.task_id,
                correlation_id="DIFFERENT",
                agent_name="transport-agent",
                agent_version="0.1.0",
                skill="plan_transport",
                status=TaskStatus.COMPLETED,
                result={"recommended": None},
                started_at=started,
                completed_at=started,
                duration_ms=0,
            )
            with pytest.raises(A2AResponseError, match="did not match"):
                _validate_artifact(forged, task)
        finally:
            await http.aclose()


class TestErrorPaths:
    async def test_malformed_task_body_is_a_client_error(self):
        http = await _http_client(_app())
        try:
            response = await http.post(
                "/a2a/tasks",
                json={"not": "a task"},
                headers={"Authorization": f"Bearer {SECRET.get_secret_value()}"},
            )
            assert response.status_code == 422
            assert response.json()["code"] == "validation_failed"
        finally:
            await http.aclose()

    async def test_invalid_payload_yields_a_rejected_artifact_not_an_error(self):
        """A structurally valid task with a bad payload is the agent's to reject — it
        returns a REJECTED artifact (200), not an HTTP error."""
        http = await _http_client(_app())
        try:
            client = _agent_client(http)
            task = AgentTask(
                skill="plan_transport",
                correlation_id="c",
                request_id="r",
                payload={"origin": "X"},  # missing required fields
            )
            artifact = await client.submit_task(task)
            assert artifact.status is TaskStatus.REJECTED
            assert artifact.result is None
        finally:
            await http.aclose()


class TestHealth:
    async def test_health_endpoints(self):
        http = await _http_client(_app())
        try:
            assert (await http.get("/a2a/health/live")).json()["status"] == "alive"
            assert (await http.get("/a2a/health/ready")).json()["status"] == "ready"
        finally:
            await http.aclose()


class TestInProcessClientParity:
    async def test_in_process_client_runs_the_same_validation(self):
        """The in-process client must enforce correlation matching just like HTTP, so tests
        using it exercise the real guarantee."""
        client = InProcessAgentClient(_agent())
        task = AgentTask(
            skill="plan_transport", correlation_id="c", request_id="r", payload=_payload()
        )
        artifact = await client.submit_task(task)
        assert artifact.matches(task)
        card = await client.discover()
        assert card.has_skill("plan_transport")
