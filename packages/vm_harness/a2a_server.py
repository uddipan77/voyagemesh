"""A2A server harness.

Turns a :class:`~vm_harness.agent.SpecialistAgent` into an independent HTTP service that
publishes an Agent Card and accepts structured tasks. This is what makes the A2A in this
project *genuine* (brief §7): the orchestrator will reach an agent only through these HTTP
endpoints and its published card, never by importing its Python.

Endpoints:

* ``GET /.well-known/agent-card.json`` — the agent's capabilities, schemas, and auth
  requirements. Discovery reads this before delegating.
* ``POST /a2a/tasks`` — submit an :class:`~vm_contracts.a2a.AgentTask`, receive an
  :class:`~vm_contracts.a2a.AgentArtifact`. Requires a valid service identity with the
  ``agent:invoke`` role.
* ``GET /a2a/health/live`` and ``/a2a/health/ready`` — for Compose health checks.

Security posture:

* Every task call is authenticated by a :class:`~vm_auth.ServiceAuthenticator`. A caller
  without a valid credential is rejected before any work is done (threat T-4).
* The correlation ID on the returned artifact is bound to the submitted task, so a response
  cannot be mistaken for the answer to a different request.
* Errors are sanitised: an authentication failure returns a fixed 401/403 with no echo of
  the presented token; an unexpected error returns a generic 500 with a trace ID.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vm_auth.service_auth import INVOKE_ROLE, ServiceAuthenticator, ServiceAuthError
from vm_contracts.a2a import AGENT_CARD_PATH, AgentTask
from vm_contracts.errors import ErrorCode, ErrorResponse
from vm_contracts.tracing import HEADER_REQUEST_ID, TraceContext
from vm_harness.agent import SpecialistAgent

__all__ = ["build_a2a_app"]

logger = logging.getLogger(__name__)


def build_a2a_app(
    agent: SpecialistAgent,
    *,
    authenticator: ServiceAuthenticator,
    url: str,
) -> FastAPI:
    """Build the FastAPI app that serves ``agent`` over A2A.

    Args:
        agent: The specialist to expose.
        authenticator: Validates the calling service's credentials.
        url: The externally-reachable base URL, advertised in the Agent Card.
    """
    app = FastAPI(
        title=f"VoyageMesh {agent.agent_name}",
        version=agent.version,
        docs_url=None,  # no interactive docs on an internal service
        redoc_url=None,
    )
    card = agent.agent_card(url=url)

    @app.get(AGENT_CARD_PATH)
    async def agent_card() -> dict[str, Any]:
        # The card is public within the internal network — discovery must not require the
        # very credential the card describes how to obtain.
        return card.model_dump(mode="json")

    @app.get("/a2a/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive", "agent": agent.agent_name, "version": agent.version}

    @app.get("/a2a/health/ready")
    async def ready() -> dict[str, str]:
        return {"status": "ready", "agent": agent.agent_name}

    @app.post("/a2a/tasks")
    async def submit_task(request: Request) -> JSONResponse:
        headers = dict(request.headers.items())
        trace = TraceContext.from_traceparent(
            headers.get("traceparent"), request_id=headers.get(HEADER_REQUEST_ID)
        )

        # 1. Authenticate the calling service. A caller we cannot verify does no work.
        try:
            identity = await authenticator.authenticate(headers)
        except ServiceAuthError:
            logger.warning(
                "a2a_auth_failed",
                extra={"agent": agent.agent_name, **trace.log_fields()},
            )
            return _error(ErrorCode.UNAUTHENTICATED, "service authentication failed", trace, 401)

        if not identity.has_role(INVOKE_ROLE):
            return _error(
                ErrorCode.INSUFFICIENT_ROLE,
                f"the '{INVOKE_ROLE}' role is required to submit tasks",
                trace,
                403,
            )

        # 2. Parse the task. A malformed body is a client error, not a server crash.
        try:
            body = await request.json()
            task = AgentTask.model_validate(body)
        except Exception:
            return _error(
                ErrorCode.VALIDATION_FAILED, "request body was not a valid A2A task", trace, 422
            )

        # 3. Run the agent. `handle` never raises — it returns an honest artifact for every
        # outcome — so there is no failure path that could leak an internal error here.
        artifact = await agent.handle(task)

        logger.info(
            "a2a_task_handled",
            extra={
                "agent": agent.agent_name,
                "skill": task.skill,
                "status": artifact.status.value,
                "caller": identity.subject,
                **trace.log_fields(),
            },
        )
        return JSONResponse(content=artifact.model_dump(mode="json"))

    return app


def _error(code: ErrorCode, detail: str, trace: TraceContext, status: int) -> JSONResponse:
    payload = ErrorResponse.of(code, detail, request_id=trace.request_id, trace_id=trace.trace_id)
    return JSONResponse(status_code=status, content=payload.model_dump(mode="json"))
