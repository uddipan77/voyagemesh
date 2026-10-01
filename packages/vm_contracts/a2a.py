"""A2A — the agent-to-agent protocol contracts.

These types define the wire format between the orchestrator and the specialist agents.
They are the reason A2A here is genuine rather than simulated: the orchestrator knows an
agent only through its published :class:`AgentCard` and this task schema, never through a
Python import.

The whole module is intentionally free of domain types beyond a generic payload. Keeping
the protocol layer ignorant of trips and offers is what allows an A2A version change to
touch this file and its adapter alone — the requirement in brief §7.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import Field, computed_field, model_validator

from vm_contracts.common import StrictModel, utc_now

__all__ = [
    "AgentArtifact",
    "AgentCapabilities",
    "AgentCard",
    "AgentSkill",
    "AgentTask",
    "AuthScheme",
    "TaskError",
    "TaskStatus",
    "new_task_id",
]

A2A_PROTOCOL_VERSION = "1.0"
AGENT_CARD_PATH = "/.well-known/agent-card.json"


def new_task_id() -> str:
    return f"task_{uuid.uuid4().hex[:20]}"


class TaskStatus(StrEnum):
    """Lifecycle of an A2A task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    """Produced a usable but incomplete result — e.g. offers found but weather missing.
    Distinct from FAILED so the orchestrator can use the data while flagging degradation."""

    REJECTED = "rejected"
    """Refused before execution: bad schema, missing skill, or failed authorisation."""

    TIMEOUT = "timeout"

    @property
    def is_terminal(self) -> bool:
        return self is not TaskStatus.PENDING and self is not TaskStatus.RUNNING

    @property
    def is_usable(self) -> bool:
        """Whether the artifact carries data worth consuming."""
        return self in (TaskStatus.COMPLETED, TaskStatus.PARTIAL)


class AuthScheme(StrEnum):
    NONE = "none"
    """Development only. An agent advertising this must be running with the insecure flag."""

    BEARER_JWT = "bearer_jwt"
    OAUTH2_CLIENT_CREDENTIALS = "oauth2_client_credentials"


class AgentSkill(StrictModel):
    """A capability an agent advertises.

    The orchestrator verifies a required skill is present *before* submitting a task, so a
    capability mismatch surfaces as an explicit discovery failure rather than a confusing
    schema error deep inside the agent.
    """

    name: Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")]
    description: Annotated[str, Field(min_length=1, max_length=500)]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    tags: Annotated[list[str], Field(max_length=10)] = Field(default_factory=list)
    max_duration_seconds: Annotated[float, Field(gt=0, le=300)] = 30.0


class AgentCapabilities(StrictModel):
    """Non-skill properties of an agent."""

    streaming: bool = False
    push_notifications: bool = False
    supports_partial_results: bool = True
    max_concurrent_tasks: Annotated[int, Field(ge=1, le=100)] = 8
    mock_mode: bool = False
    """True when the agent is serving deterministic mock data. Propagated into the
    response so the UI's MOCKED badge is driven by the agent's own declaration."""


class AgentCard(StrictModel):
    """An agent's self-description, served at :data:`AGENT_CARD_PATH`."""

    protocol_version: Annotated[str, Field(min_length=1, max_length=20)] = A2A_PROTOCOL_VERSION
    name: Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_-]*$")]
    display_name: Annotated[str, Field(min_length=1, max_length=120)]
    description: Annotated[str, Field(min_length=1, max_length=1000)]
    version: Annotated[str, Field(min_length=1, max_length=20)]
    url: Annotated[str, Field(min_length=1, max_length=300)]

    skills: Annotated[list[AgentSkill], Field(min_length=1, max_length=20)]
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)

    auth_scheme: AuthScheme = AuthScheme.BEARER_JWT
    required_roles: Annotated[list[str], Field(max_length=10)] = Field(default_factory=list)
    documentation_url: Annotated[str | None, Field(max_length=300)] = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        names = [s.name for s in self.skills]
        if len(names) != len(set(names)):
            raise ValueError("duplicate skill names in agent card")
        return self

    def has_skill(self, name: str) -> bool:
        return any(s.name == name for s in self.skills)

    def skill(self, name: str) -> AgentSkill:
        for candidate in self.skills:
            if candidate.name == name:
                return candidate
        raise KeyError(
            f"agent '{self.name}' does not advertise skill '{name}'; it offers: "
            f"{', '.join(s.name for s in self.skills)}"
        )


class AgentTask(StrictModel):
    """A unit of work submitted to an agent."""

    task_id: Annotated[str, Field(min_length=1, max_length=60)] = Field(default_factory=new_task_id)
    skill: Annotated[str, Field(min_length=1, max_length=80)]
    payload: dict[str, Any]

    correlation_id: Annotated[str, Field(min_length=1, max_length=60)]
    """Binds the task to the originating trip request. The orchestrator discards any
    artifact whose correlation ID does not match what it sent — see threat T-4."""

    request_id: Annotated[str, Field(min_length=1, max_length=60)]
    trace_id: Annotated[str | None, Field(max_length=60)] = None
    traceparent: Annotated[str | None, Field(max_length=100)] = None
    """W3C trace context header, propagated verbatim so one trace spans every service."""

    user_reference: Annotated[str | None, Field(max_length=100)] = None
    """A privacy-safe user handle. The user's access token is never forwarded — only the
    minimum claims an agent needs (brief §15B, least privilege)."""

    user_roles: Annotated[list[str], Field(max_length=10)] = Field(default_factory=list)
    deadline_seconds: Annotated[float, Field(gt=0, le=300)] = 25.0
    attempt: Annotated[int, Field(ge=1, le=5)] = 1
    created_at: datetime = Field(default_factory=utc_now)
    idempotency_key: Annotated[str | None, Field(max_length=100)] = None


class TaskError(StrictModel):
    """A sanitised failure description.

    Never carries a stack trace, an internal hostname, or a provider's raw error body —
    those go to the structured log with the trace ID, not across a service boundary.
    """

    code: Annotated[str, Field(min_length=1, max_length=60)]
    message: Annotated[str, Field(min_length=1, max_length=500)]
    retryable: bool = False
    details: dict[str, str] = Field(default_factory=dict)


class AgentArtifact(StrictModel):
    """An agent's response to a task."""

    task_id: Annotated[str, Field(min_length=1, max_length=60)]
    correlation_id: Annotated[str, Field(min_length=1, max_length=60)]
    agent_name: Annotated[str, Field(min_length=1, max_length=80)]
    agent_version: Annotated[str, Field(min_length=1, max_length=20)]
    skill: Annotated[str, Field(min_length=1, max_length=80)]

    status: TaskStatus
    result: dict[str, Any] | None = None
    error: TaskError | None = None

    warnings: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)
    action_summaries: Annotated[list[str], Field(max_length=30)] = Field(default_factory=list)
    """Concise, auditable descriptions of what the agent did ("searched 12 rail options,
    filtered 4 exceeding the duration limit"). Deliberately *not* raw reasoning — brief
    §11 forbids exposing private chain-of-thought, and storing it would be a liability."""

    started_at: datetime
    completed_at: datetime
    duration_ms: Annotated[int, Field(ge=0)]
    tool_call_count: Annotated[int, Field(ge=0)] = 0
    llm_call_count: Annotated[int, Field(ge=0)] = 0
    retry_count: Annotated[int, Field(ge=0)] = 0
    degraded: bool = False
    used_mock_data: bool = False

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.status.is_usable and self.result is None:
            raise ValueError(
                f"status={self.status.value} claims a usable result but result is None"
            )
        if self.status in (TaskStatus.FAILED, TaskStatus.REJECTED, TaskStatus.TIMEOUT):
            if self.error is None:
                raise ValueError(f"status={self.status.value} requires an error description")
            if self.result is not None:
                raise ValueError(
                    f"status={self.status.value} must not carry a result; a failed task "
                    f"returning data is exactly how fabricated output leaks into a plan"
                )
        if self.status is TaskStatus.PARTIAL and not self.warnings:
            raise ValueError(
                "status=partial requires at least one warning explaining what is missing"
            )
        if self.completed_at < self.started_at:
            raise ValueError("completed_at precedes started_at")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_usable(self) -> bool:
        return self.status.is_usable

    def matches(self, task: AgentTask) -> bool:
        """Whether this artifact genuinely answers ``task``.

        Checked by the orchestrator on every response. A mismatch means either a bug or an
        attempt to inject a result for a different request.
        """
        return (
            self.task_id == task.task_id
            and self.correlation_id == task.correlation_id
            and self.skill == task.skill
        )
