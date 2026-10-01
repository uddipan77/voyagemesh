"""Service-factory auth selection and A2A client retry/error paths.

The `select_service_authenticator` production guard is security-relevant: it is what stops a
service starting unauthenticated in production. It gets explicit tests rather than being
covered incidentally.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from vm_auth import (
    InsecureServiceAuthenticator,
    SharedSecretServiceAuthenticator,
)
from vm_config.settings import Settings
from vm_contracts.a2a import AgentTask, TaskStatus
from vm_harness import A2AResponseError, HttpAgentClient
from vm_harness.service_factory import build_agent_service, select_service_authenticator
from vm_transport_agent.agent import build_agent

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


class TestServiceAuthSelection:
    def test_a_configured_secret_enables_shared_secret_auth(self, tmp_path):
        secret_file = tmp_path / "service.secret"
        secret_file.write_text("a-real-service-secret", encoding="utf-8")
        settings = Settings.for_testing(
            ENVIRONMENT="local",
            AUTH_SERVICE_CLIENT_SECRET_FILE=str(secret_file),
        )
        assert isinstance(select_service_authenticator(settings), SharedSecretServiceAuthenticator)

    def test_no_secret_outside_production_falls_back_to_insecure(self):
        settings = Settings.for_testing(ENVIRONMENT="local")
        assert isinstance(select_service_authenticator(settings), InsecureServiceAuthenticator)

    def test_no_secret_in_production_refuses_to_start(self):
        """The load-bearing security guard: production must not run unauthenticated."""
        settings = Settings.for_testing(ENVIRONMENT="production")
        with pytest.raises(RuntimeError, match="refusing to start unauthenticated"):
            select_service_authenticator(settings)


class TestBuildAgentService:
    def test_builds_a_working_a2a_app(self):
        """The factory wires an agent into a service without touching the MCP server.

        Mock LLM + mock providers, so the build needs no network; the HTTP MCP client is
        constructed but never called here."""
        settings = Settings.for_testing(
            PROVIDER_MODE="mock", ENVIRONMENT="test", LLM_PROVIDER="mock"
        )
        app = build_agent_service(
            build_agent,
            settings=settings,
            url="http://transport:8010",
            mcp_url="http://transport-mcp:8110",
        )
        assert app.title == "VoyageMesh transport-agent"

    async def test_built_service_serves_the_card(self):
        settings = Settings.for_testing(
            PROVIDER_MODE="mock", ENVIRONMENT="test", LLM_PROVIDER="mock"
        )
        app = build_agent_service(
            build_agent,
            settings=settings,
            url="http://transport:8010",
            mcp_url="http://transport-mcp:8110",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://transport:8010"
        ) as http:
            card = (await http.get("/.well-known/agent-card.json")).json()
            assert card["name"] == "transport-agent"


class TestHttpClientResilience:
    """The client's retry and error mapping, without a live server."""

    def _settings(self) -> Settings:
        return Settings.for_testing(PROVIDER_MODE="mock", ENVIRONMENT="test")

    def _valid_artifact_json(self, task: AgentTask) -> dict:
        from vm_contracts.common import utc_now

        started = utc_now()
        from vm_contracts.a2a import AgentArtifact

        return AgentArtifact(
            task_id=task.task_id,
            correlation_id=task.correlation_id,
            agent_name="transport-agent",
            agent_version="0.1.0",
            skill=task.skill,
            status=TaskStatus.COMPLETED,
            result={"recommended": None},
            started_at=started,
            completed_at=started,
            duration_ms=1,
        ).model_dump(mode="json")

    @pytest.fixture(autouse=True)
    def _instant_backoff(self, monkeypatch):
        import asyncio

        async def instant(_seconds: float) -> None:
            return None

        monkeypatch.setattr(asyncio, "sleep", instant)

    async def test_server_error_is_retried_then_succeeds(self):
        task = AgentTask(skill="plan_transport", correlation_id="c", request_id="r", payload={})
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(503)
            return httpx.Response(200, json=self._valid_artifact_json(task))

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = HttpAgentClient("http://agent", client=http, max_retries=2)
        try:
            artifact = await client.submit_task(task)
            assert artifact.status is TaskStatus.COMPLETED
            assert calls["n"] == 2
        finally:
            await http.aclose()

    async def test_client_error_is_not_retried(self):
        task = AgentTask(skill="plan_transport", correlation_id="c", request_id="r", payload={})
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(422)

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = HttpAgentClient("http://agent", client=http, max_retries=2)
        try:
            with pytest.raises(A2AResponseError, match="rejected"):
                await client.submit_task(task)
            assert calls["n"] == 1
        finally:
            await http.aclose()

    async def test_persistent_server_error_raises_after_retries(self):
        task = AgentTask(skill="plan_transport", correlation_id="c", request_id="r", payload={})

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = HttpAgentClient("http://agent", client=http, max_retries=2)
        try:
            with pytest.raises(A2AResponseError):
                await client.submit_task(task)
        finally:
            await http.aclose()

    async def test_connection_failure_maps_to_response_error(self):
        task = AgentTask(skill="plan_transport", correlation_id="c", request_id="r", payload={})

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = HttpAgentClient("http://agent", client=http, max_retries=1)
        try:
            with pytest.raises(A2AResponseError, match="unreachable"):
                await client.submit_task(task)
        finally:
            await http.aclose()

    async def test_owns_its_client_when_not_injected(self):
        client = HttpAgentClient("http://agent")
        # A client it created must be closeable via aclose without error.
        await client.aclose()
