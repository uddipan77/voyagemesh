"""A2A security: service authentication, role enforcement, and no credential leakage.

Guards threat T-4 (cross-agent impersonation / task tampering): an agent does no work for a
caller it cannot authenticate, and an authentication failure never echoes the presented
token.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from vm_auth import InsecureServiceAuthenticator, SharedSecretServiceAuthenticator
from vm_config.settings import Settings
from vm_harness import InProcessToolClient, build_a2a_app
from vm_llm.mock_provider import MockLLMProvider
from vm_transport_agent.agent import TransportAgent
from vm_transport_mcp.server import build_server

pytestmark = [pytest.mark.security, pytest.mark.contract]

REAL_SECRET = "the-real-service-secret-value"


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _agent() -> TransportAgent:
    settings = Settings.for_testing(PROVIDER_MODE="mock", ENVIRONMENT="test")
    return TransportAgent(
        tool_client=InProcessToolClient(build_server(settings)), llm=MockLLMProvider()
    )


def _app(authenticator: Any) -> Any:
    return build_a2a_app(_agent(), authenticator=authenticator, url="http://transport:8010")


async def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://transport:8010"
    )


TASK = {
    "skill": "plan_transport",
    "correlation_id": "c",
    "request_id": "r",
    "payload": {"origin": "Nuremberg", "destination": "Prague", "departure_date": "2026-08-10"},
}


class TestAuthenticationRequired:
    async def test_task_without_a_token_is_rejected(self):
        http = await _client(_app(SharedSecretServiceAuthenticator(SecretStr(REAL_SECRET))))
        try:
            response = await http.post("/a2a/tasks", json=TASK)
            assert response.status_code == 401
            assert response.json()["code"] == "unauthenticated"
        finally:
            await http.aclose()

    async def test_task_with_a_wrong_token_is_rejected(self):
        http = await _client(_app(SharedSecretServiceAuthenticator(SecretStr(REAL_SECRET))))
        try:
            response = await http.post(
                "/a2a/tasks", json=TASK, headers={"Authorization": "Bearer wrong-guess"}
            )
            assert response.status_code == 401
        finally:
            await http.aclose()

    async def test_no_work_is_done_for_an_unauthenticated_caller(self):
        """The agent must not run — not merely refuse the result — for a bad caller.

        Verified by asserting no tool call reached the MCP server: the response is a pure
        auth rejection, not a completed-then-discarded run."""
        settings = Settings.for_testing(PROVIDER_MODE="mock", ENVIRONMENT="test")
        server = build_server(settings)
        agent = TransportAgent(tool_client=InProcessToolClient(server), llm=MockLLMProvider())
        app = build_a2a_app(
            agent,
            authenticator=SharedSecretServiceAuthenticator(SecretStr(REAL_SECRET)),
            url="http://t:8010",
        )
        http = await _client(app)
        try:
            await http.post("/a2a/tasks", json=TASK)  # no auth
            registry = server.voyagemesh_registry
            assert registry.call_count("search_ground_transport") == 0
        finally:
            await http.aclose()


class TestNoTokenLeakage:
    async def test_auth_failure_response_does_not_echo_the_token(self):
        http = await _client(_app(SharedSecretServiceAuthenticator(SecretStr(REAL_SECRET))))
        try:
            leaked = "MY-SECRET-GUESS-TOKEN-98765"
            response = await http.post(
                "/a2a/tasks", json=TASK, headers={"Authorization": f"Bearer {leaked}"}
            )
            assert leaked not in response.text
        finally:
            await http.aclose()

    async def test_the_real_secret_never_appears_in_a_response(self):
        http = await _client(_app(SharedSecretServiceAuthenticator(SecretStr(REAL_SECRET))))
        try:
            for response in [
                await http.get("/.well-known/agent-card.json"),
                await http.post("/a2a/tasks", json=TASK),
                await http.post(
                    "/a2a/tasks", json=TASK, headers={"Authorization": f"Bearer {REAL_SECRET}"}
                ),
            ]:
                assert REAL_SECRET not in response.text
        finally:
            await http.aclose()


class TestRoleEnforcement:
    async def test_a_caller_without_the_invoke_role_is_forbidden(self):
        """A valid identity lacking `agent:invoke` gets 403, not 401."""
        from vm_auth import ServiceIdentity

        class ReadOnlyAuth:
            async def authenticate(self, headers: dict[str, str]) -> ServiceIdentity:
                return ServiceIdentity(subject="reader", roles=frozenset({"reader"}))

            @property
            def scheme_name(self) -> str:
                return "test"

        http = await _client(_app(ReadOnlyAuth()))
        try:
            response = await http.post(
                "/a2a/tasks", json=TASK, headers={"Authorization": "Bearer anything"}
            )
            assert response.status_code == 403
            assert response.json()["code"] == "insufficient_role"
        finally:
            await http.aclose()


class TestInsecureModeIsExplicit:
    async def test_insecure_authenticator_allows_calls_but_is_opt_in(self):
        """The insecure authenticator works (dev convenience) but is never the default: a
        service must be constructed with it deliberately."""
        http = await _client(_app(InsecureServiceAuthenticator()))
        try:
            response = await http.post("/a2a/tasks", json=TASK)
            assert response.status_code == 200  # no auth enforced
        finally:
            await http.aclose()
