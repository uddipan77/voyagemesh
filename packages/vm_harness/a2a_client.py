"""A2A client — the orchestrator's side of the protocol.

The :class:`AgentClient` protocol is the seam the brief asks for (§7): the domain layer
depends on it, never on HTTP, so a protocol-version change touches an adapter and nothing
else. Two implementations back it:

* :class:`HttpAgentClient` reaches a remote agent over HTTP — discovery via the well-known
  card, submission via ``POST /a2a/tasks``, with a service credential attached.
* :class:`InProcessAgentClient` drives a :class:`SpecialistAgent` object directly, so the
  orchestrator and its tests can run without a network.

The orchestrator's obligations from §7 live here, not in the caller: verify the agent
advertises the required skill before submitting; retry transient failures within a bounded
budget; reject a response whose schema, correlation ID, or task ID does not match what was
sent (threat T-4).
"""

from __future__ import annotations

import asyncio
import logging
from types import TracebackType
from typing import Protocol, Self

import httpx

from vm_auth.service_auth import PropagatedClaims, ServiceTokenProvider, StaticServiceTokenProvider
from vm_contracts.a2a import AGENT_CARD_PATH, AgentArtifact, AgentCard, AgentTask, TaskStatus
from vm_contracts.tracing import TraceContext
from vm_harness.agent import SpecialistAgent

__all__ = [
    "A2AClientError",
    "A2ADiscoveryError",
    "A2AResponseError",
    "AgentClient",
    "HttpAgentClient",
    "InProcessAgentClient",
    "submit_verified",
]

logger = logging.getLogger(__name__)


class A2AClientError(Exception):
    """Base class for A2A client failures. Messages are sanitised — no URL, no token."""


class A2ADiscoveryError(A2AClientError):
    """An agent's card could not be fetched or did not advertise a required skill."""


class A2AResponseError(A2AClientError):
    """An agent's response was unreachable, malformed, or answered the wrong task."""


class AgentClient(Protocol):
    """The orchestrator's view of a specialist agent."""

    async def discover(self) -> AgentCard:
        """Fetch and validate the agent's card. Raises :class:`A2ADiscoveryError`."""
        ...

    async def submit_task(self, task: AgentTask) -> AgentArtifact:
        """Submit a task and return the validated artifact.

        Raises:
            A2AResponseError: The agent was unreachable after retries, or its response did
                not match the task (schema, task ID, or correlation ID).
        """
        ...


def _validate_artifact(artifact: AgentArtifact, task: AgentTask) -> AgentArtifact:
    """Reject a response that does not genuinely answer ``task``.

    A mismatch is either a bug or an attempt to inject a result for a different request;
    either way the orchestrator must not accept it (threat T-4). Runs on every response,
    from both the HTTP and in-process clients.
    """
    if not artifact.matches(task):
        raise A2AResponseError(
            "agent response did not match the submitted task (task ID or correlation ID mismatch)"
        )
    return artifact


class HttpAgentClient:
    """Reaches a remote agent over HTTP."""

    def __init__(
        self,
        base_url: str,
        *,
        token_provider: ServiceTokenProvider | None = None,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._tokens = token_provider or StaticServiceTokenProvider(None)
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds), follow_redirects=False
        )
        self._owns_client = client is None

    async def discover(self) -> AgentCard:
        url = f"{self._base_url}{AGENT_CARD_PATH}"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return AgentCard.model_validate(response.json())
        except httpx.HTTPError as exc:
            raise A2ADiscoveryError(
                f"could not fetch the agent card ({type(exc).__name__})"
            ) from None
        except Exception as exc:
            raise A2ADiscoveryError(
                f"agent card was not a valid AgentCard ({type(exc).__name__})"
            ) from None

    async def submit_task(self, task: AgentTask) -> AgentArtifact:
        url = f"{self._base_url}/a2a/tasks"
        headers = await self._headers(task)
        body = task.model_dump(mode="json")

        last: Exception | None = None
        for attempt in range(1, self._max_retries + 2):
            try:
                response = await self._client.post(url, json=body, headers=headers)
            except httpx.HTTPError as exc:
                last = A2AResponseError(f"agent unreachable ({type(exc).__name__})")
                await self._backoff(attempt)
                continue

            if response.status_code >= 500:
                # A server error may be transient; retry within budget.
                last = A2AResponseError(f"agent returned {response.status_code}")
                await self._backoff(attempt)
                continue
            if response.status_code >= 400:
                # A client error will not improve on retry.
                raise A2AResponseError(f"agent rejected the task ({response.status_code})")

            try:
                artifact = AgentArtifact.model_validate(response.json())
            except Exception as exc:
                raise A2AResponseError(
                    f"agent response was not a valid artifact ({type(exc).__name__})"
                ) from None
            return _validate_artifact(artifact, task)

        raise last or A2AResponseError("agent unreachable")

    async def _headers(self, task: AgentTask) -> dict[str, str]:
        headers = await self._tokens.authorization_header()
        # Propagate trace + minimal claims. NOT the user's access token (least privilege).
        trace = TraceContext.from_traceparent(task.traceparent, request_id=task.request_id)
        headers.update(trace.to_headers())
        headers.update(
            PropagatedClaims(
                user_reference=task.user_reference,
                roles=tuple(task.user_roles),
                request_id=task.request_id,
                trace_id=task.trace_id,
            ).to_headers()
        )
        return headers

    async def _backoff(self, attempt: int) -> None:
        await asyncio.sleep(min(0.2 * (2 ** (attempt - 1)), 4.0))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class InProcessAgentClient:
    """Drives a specialist agent directly, without a network hop.

    Runs the same response validation as the HTTP client, so a test using this exercises the
    orchestrator's correlation-matching guarantees exactly as production would.
    """

    def __init__(self, agent: SpecialistAgent, *, url: str = "http://in-process") -> None:
        self._agent = agent
        self._url = url

    async def discover(self) -> AgentCard:
        return self._agent.agent_card(url=self._url)

    async def submit_task(self, task: AgentTask) -> AgentArtifact:
        artifact = await self._agent.handle(task)
        return _validate_artifact(artifact, task)


async def submit_verified(
    client: AgentClient, task: AgentTask, *, required_skill: str
) -> AgentArtifact:
    """Discover, verify the skill, then submit — the orchestrator's full §7 obligation.

    Kept as a free function so the orchestrator can call it against any ``AgentClient``.
    """
    card = await client.discover()
    if not card.has_skill(required_skill):
        raise A2ADiscoveryError(
            f"agent '{card.name}' does not advertise the required skill '{required_skill}'"
        )
    artifact = await client.submit_task(task)
    if artifact.status is TaskStatus.REJECTED:
        logger.warning(
            "a2a_task_rejected",
            extra={"agent": card.name, "skill": required_skill, "task": task.task_id},
        )
    return artifact
