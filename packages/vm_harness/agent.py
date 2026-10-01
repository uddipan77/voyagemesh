"""Base class for the specialist agents.

A specialist agent receives an :class:`~vm_contracts.a2a.AgentTask`, runs its deterministic
tool plan under the harness, and returns an :class:`~vm_contracts.a2a.AgentArtifact`. All of
the A2A envelope handling — status selection, timing, trace propagation, failure
sanitisation — lives here so a subclass writes only its domain logic.

The status the base assigns is honest by construction:

* ``COMPLETED`` — the agent produced its primary result.
* ``PARTIAL`` — usable output with a gap it names in the warnings (brief §23).
* ``FAILED`` — no usable result; carries a sanitised error and, crucially, *no* result
  payload, so a failure can never be mistaken for data.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from vm_contracts.a2a import AgentArtifact, AgentCard, AgentTask, TaskError, TaskStatus
from vm_contracts.common import utc_now
from vm_contracts.tracing import TraceContext
from vm_harness.harness import AgentHarness
from vm_harness.mcp_client import ToolClient, ToolClientError
from vm_harness.trace import AgentRunContext, StoppingConditionReached
from vm_llm.base import LLMProvider

__all__ = ["AgentOutcome", "SpecialistAgent"]

logger = logging.getLogger(__name__)


class AgentOutcome(BaseModel):
    """What a subclass returns from :meth:`SpecialistAgent.execute`.

    Keeping this separate from the wire-level :class:`AgentArtifact` means a subclass never
    has to think about task IDs, timing, or correlation — it says *what happened*, and the
    base translates that into the protocol.
    """

    model_config = {"arbitrary_types_allowed": True}

    status: TaskStatus
    result: dict[str, Any] | None = None
    error: TaskError | None = None


class SpecialistAgent(ABC):
    """Common runtime for a single-skill agent."""

    def __init__(
        self,
        *,
        agent_name: str,
        version: str,
        tool_client: ToolClient,
        llm: LLMProvider,
        max_tool_calls: int = 8,
        max_llm_calls: int = 6,
        max_tool_retries: int = 2,
        default_deadline_seconds: float = 25.0,
    ) -> None:
        self.agent_name = agent_name
        self.version = version
        self._default_deadline = default_deadline_seconds
        self._max_tool_calls = max_tool_calls
        self._max_llm_calls = max_llm_calls
        self.harness = AgentHarness(
            agent_name=agent_name,
            tool_client=tool_client,
            llm=llm,
            max_tool_retries=max_tool_retries,
        )

    # -- to be provided by each specialist -------------------------------
    @property
    @abstractmethod
    def skill(self) -> str:
        """The single skill this agent advertises and accepts."""

    @abstractmethod
    def agent_card(self, *, url: str) -> AgentCard:
        """The Agent Card this specialist publishes (used by A2A discovery in Phase 7)."""

    @abstractmethod
    async def execute(self, task: AgentTask, context: AgentRunContext) -> AgentOutcome:
        """Run the domain logic. May assume the task's skill has already been checked."""

    # -- shared runtime --------------------------------------------------
    async def handle(self, task: AgentTask) -> AgentArtifact:
        """Execute ``task`` and return a well-formed artifact.

        Never raises: every failure path — wrong skill, exhausted budget, unreachable tool,
        unexpected exception — is converted into an honest artifact. An agent that raised
        across the A2A boundary would force every caller to catch, and a missed catch would
        turn a degraded dependency into a failed request.
        """
        started = utc_now()
        trace = self._trace_for(task)
        context = AgentRunContext(
            agent_name=self.agent_name,
            trace=trace,
            max_tool_calls=self._max_tool_calls,
            max_llm_calls=self._max_llm_calls,
            deadline_seconds=min(task.deadline_seconds, self._default_deadline),
        )

        if task.skill != self.skill:
            return self._artifact(
                task,
                started,
                context,
                TaskStatus.REJECTED,
                error=TaskError(
                    code="unsupported_skill",
                    message=(f"this agent provides '{self.skill}', not '{task.skill}'"),
                ),
            )

        try:
            outcome = await self.execute(task, context)
        except StoppingConditionReached as stop:
            context.warn(f"Stopped early: {stop.detail}")
            outcome = AgentOutcome(
                status=TaskStatus.FAILED,
                error=TaskError(code=stop.reason.value, message=stop.detail, retryable=False),
            )
        except ToolClientError:
            outcome = AgentOutcome(
                status=TaskStatus.FAILED,
                error=TaskError(
                    code="tool_unavailable",
                    message="a required tool server was unreachable",
                    retryable=True,
                ),
            )
        except Exception:
            # The message is fixed and generic; the real detail goes to the log with the
            # trace ID, never across the boundary (threat T-2).
            logger.exception("agent_unhandled_error", extra={"agent": self.agent_name})
            outcome = AgentOutcome(
                status=TaskStatus.FAILED,
                error=TaskError(
                    code="internal_error",
                    message="the agent encountered an unexpected error",
                    retryable=False,
                ),
            )

        return self._artifact(
            task, started, context, outcome.status, result=outcome.result, error=outcome.error
        )

    def _artifact(
        self,
        task: AgentTask,
        started: Any,
        context: AgentRunContext,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None = None,
        error: TaskError | None = None,
    ) -> AgentArtifact:
        completed = utc_now()
        # PARTIAL must carry at least one warning; if the agent degraded but recorded none,
        # add a generic one so the invariant holds and the caller sees the gap.
        warnings = list(context.warnings)
        if status is TaskStatus.PARTIAL and not warnings:
            warnings.append("Some information was unavailable; this result is partial.")

        return AgentArtifact(
            task_id=task.task_id,
            correlation_id=task.correlation_id,
            agent_name=self.agent_name,
            agent_version=self.version,
            skill=task.skill,
            status=status,
            result=result if status.is_usable else None,
            error=error,
            warnings=warnings[:20],
            action_summaries=context.action_summaries()[:30],
            started_at=started,
            completed_at=completed,
            duration_ms=int((completed - started).total_seconds() * 1000),
            tool_call_count=context.tool_calls,
            llm_call_count=context.llm_calls,
            retry_count=sum(a.retry_count for a in context.actions),
            degraded=context.degraded or status is TaskStatus.PARTIAL,
            used_mock_data=context.used_mock_data,
        )

    def _trace_for(self, task: AgentTask) -> TraceContext:
        base = TraceContext.from_traceparent(task.traceparent, request_id=task.request_id)
        return base.model_copy(
            update={
                "correlation_id": task.correlation_id,
                "agent_name": self.agent_name,
                "agent_task_id": task.task_id,
                "user_reference": task.user_reference,
            }
        )
