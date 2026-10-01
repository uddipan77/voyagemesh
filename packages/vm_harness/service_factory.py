"""Builds a runnable A2A service for a specialist agent from configuration.

Each agent's ``app.py`` is a few-line module that calls this. Centralising the wiring —
provider selection, MCP client, service authenticator, insecure-mode guard — means all
three agents make the same safe choices, and a change to those choices happens once.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from vm_auth.service_auth import (
    InsecureServiceAuthenticator,
    ServiceAuthenticator,
    ServiceTokenProvider,
    SharedSecretServiceAuthenticator,
    StaticServiceTokenProvider,
)
from vm_config.settings import Environment, Settings
from vm_harness.a2a_server import build_a2a_app
from vm_harness.agent import SpecialistAgent
from vm_harness.mcp_client import HttpMCPToolClient, ToolClient
from vm_llm.base import LLMProvider
from vm_llm.factory import build_llm_provider
from vm_telemetry import CONTENT_TYPE_LATEST, metrics_payload

__all__ = [
    "AgentBuilder",
    "build_agent_service",
    "build_service_token_provider",
    "select_service_authenticator",
]

logger = logging.getLogger(__name__)

# An agent's factory function: `build_agent(*, tool_client, llm) -> SpecialistAgent`.
AgentBuilder = Callable[..., SpecialistAgent]


def select_service_authenticator(settings: Settings) -> ServiceAuthenticator:
    """Choose how inbound A2A calls are authenticated.

    Selection order:

    1. **Keycloak** when ``AUTH_SERVICE_AUTH=keycloak`` — the production scheme (Phase 11):
       a service caller presents a Keycloak client-credentials token, validated exactly like
       a user token but requiring the ``agent:invoke`` role.
    2. **Shared secret** when a service-client secret is configured — the dev scheme (Phase 7),
       genuinely enforced but simpler than Keycloak.
    3. **Insecure** only outside production, and only when nothing else is configured.

    In production, an unconfigured service auth is fatal here rather than silently
    unauthenticated.
    """
    if settings.auth.service_auth == "keycloak":
        from vm_auth import KeycloakServiceAuthenticator

        return KeycloakServiceAuthenticator(settings.auth)

    secret = settings.auth.load_client_secret()
    if secret is not None:
        return SharedSecretServiceAuthenticator(secret)

    if settings.environment is Environment.PRODUCTION:
        raise RuntimeError(
            "no service authentication configured (set AUTH_SERVICE_AUTH=keycloak or "
            "AUTH_SERVICE_CLIENT_SECRET_FILE) but ENVIRONMENT=production; refusing to start "
            "unauthenticated"
        )
    logger.warning(
        "service_auth_insecure_fallback",
        extra={"reason": "no Keycloak or shared secret configured"},
    )
    return InsecureServiceAuthenticator()


def build_service_token_provider(settings: Settings) -> ServiceTokenProvider:
    """Choose how *outbound* A2A calls present a credential — the mirror of the inbound
    selector above.

    ``keycloak`` mints a client-credentials token; a configured shared secret presents that
    secret; otherwise calls go out unauthenticated (only reached outside production, where the
    receiving agent runs its own insecure authenticator).
    """
    if settings.auth.service_auth == "keycloak":
        secret = settings.auth.load_client_secret()
        if secret is not None:
            from vm_auth import KeycloakTokenProvider

            return KeycloakTokenProvider(settings.auth, client_secret=secret)
        if settings.environment is Environment.PRODUCTION:
            raise RuntimeError(
                "AUTH_SERVICE_AUTH=keycloak but no AUTH_SERVICE_CLIENT_SECRET_FILE is set; "
                "the orchestrator cannot obtain a service token"
            )
    secret = settings.auth.load_client_secret()
    return StaticServiceTokenProvider(secret)


def build_agent_service(
    build_agent: AgentBuilder,
    *,
    settings: Settings,
    url: str,
    mcp_url: str,
) -> FastAPI:
    """Wire a specialist agent into a runnable A2A FastAPI service.

    Args:
        build_agent: The agent's ``build_agent(*, tool_client, llm)`` factory.
        settings: Root settings — selects the LLM provider, MCP transport, and auth scheme.
        url: The agent's externally-reachable base URL (advertised in its card).
        mcp_url: The base URL of this agent's MCP server.
    """
    from vm_llm.observability import InstrumentedLLMProvider
    from vm_telemetry import configure_telemetry, instrument_app

    # Agents are separate processes, so each needs its own tracer provider — configuring it in
    # the gateway does nothing for them. Without this an A2A trace ends at the gateway's outbound
    # call and the agent's own MCP/LLM/HTTP spans are lost (brief §24, criterion 17).
    configure_telemetry(settings)

    tool_client: ToolClient = HttpMCPToolClient(
        mcp_url, timeout_seconds=settings.mcp.timeout_seconds
    )
    # Wrap the provider so every LLM call this agent makes emits metrics and a span. Done here
    # (not in the factory) so build_llm_provider keeps returning the bare provider its tests
    # expect, while every running service is instrumented.
    llm: LLMProvider = InstrumentedLLMProvider(build_llm_provider(settings))
    agent = build_agent(
        tool_client=tool_client,
        llm=llm,
        max_tool_calls=settings.limits.max_tool_calls_per_agent,
        max_llm_calls=settings.llm.max_calls_per_agent,
        max_tool_retries=settings.mcp.max_retries,
        default_deadline_seconds=settings.limits.max_agent_task_duration_seconds,
    )
    authenticator = select_service_authenticator(settings)
    app = build_a2a_app(agent, authenticator=authenticator, url=url)

    # Prometheus scrapes every agent on /metrics (infra/prometheus/prometheus.yml, job "agents").
    # The counters are recorded process-wide by vm_telemetry, so without this route the samples
    # exist but are unreachable and the scrape target stays permanently down.
    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(content=metrics_payload(), media_type=CONTENT_TYPE_LATEST)

    # Auto-instrument last, once the app and its routes exist (same ordering as the gateway).
    instrument_app(app, settings=settings)
    return app
