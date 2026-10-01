"""Agent run context: stopping conditions, action trace, and accounting.

This is where the brief's non-negotiables about controlled reasoning live (§11, §12):

* **Every loop is bounded.** Tool calls, LLM calls, and wall-clock time each have a hard
  ceiling. :meth:`AgentRunContext.check_can_continue` is called before every step and
  raises :class:`StoppingConditionReached` when a budget is spent — there is no path by
  which an agent runs unbounded.
* **The action trace is auditable, not confessional.** Each step records *what the agent
  did* — tool, redacted arguments, status, latency, result count — never the model's hidden
  reasoning. Storing chain-of-thought would be both a privacy liability and, per the brief,
  forbidden.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from vm_contracts.tracing import TraceContext
from vm_llm.types import TokenUsage

__all__ = [
    "ActionRecord",
    "ActionStatus",
    "AgentRunContext",
    "StopReason",
    "StoppingConditionReached",
]


class ActionStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRIED = "retried"


class StopReason(StrEnum):
    """Why an agent stopped. Every value except ``COMPLETED`` is a guard firing."""

    COMPLETED = "completed"
    MAX_TOOL_CALLS = "max_tool_calls"
    MAX_LLM_CALLS = "max_llm_calls"
    DEADLINE = "deadline"
    NO_NEW_INFORMATION = "no_new_information"


class StoppingConditionReached(Exception):  # noqa: N818 - a control signal, not an error condition
    """Raised when a run may not take another step.

    Carrying the :class:`StopReason` lets the agent produce an honest partial result that
    says *why* it stopped, rather than a bare failure.
    """

    def __init__(self, reason: StopReason, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(slots=True)
class ActionRecord:
    """One auditable step in an agent's run.

    Deliberately records outcomes, not reasoning. ``arguments_redacted`` is a shallow,
    scrubbed view of the tool arguments — enough to see what was asked, never a credential.
    """

    step: int
    action: str
    selected_tool: str | None
    arguments_redacted: dict[str, Any]
    status: ActionStatus
    latency_ms: int
    result_count: int = 0
    retry_count: int = 0
    decision_summary: str = ""
    """A concise, human-readable note — 'searched 12 options, 4 exceeded the duration
    limit'. Never the model's raw chain-of-thought."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "selected_tool": self.selected_tool,
            "arguments": self.arguments_redacted,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "result_count": self.result_count,
            "retry_count": self.retry_count,
            "decision_summary": self.decision_summary,
        }


_REDACT_KEYS = frozenset(
    {"token", "authorization", "api_key", "apikey", "password", "secret", "cookie"}
)
_REDACTED = "***"


def _redact(arguments: dict[str, Any], *, max_len: int = 120) -> dict[str, Any]:
    """A shallow, truncated, secret-scrubbed view of tool arguments for the trace."""
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        if key.lower() in _REDACT_KEYS:
            out[key] = _REDACTED
            continue
        if isinstance(value, str) and len(value) > max_len:
            out[key] = value[:max_len] + "…"
        elif isinstance(value, list):
            out[key] = f"[{len(value)} items]"
        elif isinstance(value, dict):
            out[key] = f"{{{len(value)} keys}}"
        else:
            out[key] = value
    return out


@dataclass(slots=True)
class AgentRunContext:
    """Tracks budgets, timing, tokens, and the action trace for one agent task."""

    agent_name: str
    trace: TraceContext
    max_tool_calls: int = 8
    max_llm_calls: int = 6
    deadline_seconds: float = 25.0

    _started_at: float = field(default_factory=time.monotonic, init=False)
    tool_calls: int = field(default=0, init=False)
    llm_calls: int = field(default=0, init=False)
    usage: TokenUsage = field(default_factory=TokenUsage, init=False)
    actions: list[ActionRecord] = field(default_factory=list, init=False)
    warnings: list[str] = field(default_factory=list, init=False)
    used_mock_data: bool = field(default=False, init=False)
    degraded: bool = field(default=False, init=False)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started_at

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_seconds - self.elapsed_seconds)

    def check_can_continue(self) -> None:
        """Raise if another step would breach a budget. Call before every step."""
        if self.tool_calls >= self.max_tool_calls:
            raise StoppingConditionReached(
                StopReason.MAX_TOOL_CALLS,
                f"reached the {self.max_tool_calls}-tool-call limit",
            )
        if self.llm_calls >= self.max_llm_calls:
            raise StoppingConditionReached(
                StopReason.MAX_LLM_CALLS, f"reached the {self.max_llm_calls}-LLM-call limit"
            )
        if self.elapsed_seconds >= self.deadline_seconds:
            raise StoppingConditionReached(
                StopReason.DEADLINE,
                f"exceeded the {self.deadline_seconds:g}s task deadline",
            )

    def record_tool(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        status: ActionStatus,
        latency_ms: int,
        result_count: int = 0,
        retry_count: int = 0,
        summary: str = "",
    ) -> ActionRecord:
        self.tool_calls += 1
        record = ActionRecord(
            step=len(self.actions) + 1,
            action="tool_call",
            selected_tool=tool,
            arguments_redacted=_redact(arguments),
            status=status,
            latency_ms=latency_ms,
            result_count=result_count,
            retry_count=retry_count,
            decision_summary=summary,
        )
        self.actions.append(record)
        return record

    def record_llm(
        self, *, usage: TokenUsage, latency_ms: int, summary: str, status: ActionStatus
    ) -> ActionRecord:
        self.llm_calls += 1
        self.usage = self.usage + usage
        record = ActionRecord(
            step=len(self.actions) + 1,
            action="llm_call",
            selected_tool=None,
            arguments_redacted={},
            status=status,
            latency_ms=latency_ms,
            decision_summary=summary,
        )
        self.actions.append(record)
        return record

    def note(self, message: str) -> None:
        """Record a step that was neither a tool nor an LLM call (a decision, a skip)."""
        self.actions.append(
            ActionRecord(
                step=len(self.actions) + 1,
                action="note",
                selected_tool=None,
                arguments_redacted={},
                status=ActionStatus.OK,
                latency_ms=0,
                decision_summary=message,
            )
        )

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def action_summaries(self) -> list[str]:
        """The concise, shareable trace. Safe to return in an A2A artifact."""
        return [a.decision_summary for a in self.actions if a.decision_summary]

    def trace_records(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self.actions]
