"""The reusable agent harness.

Every specialist agent shares this, so retries, timeouts, tracing, token accounting,
stopping rules, and structured-output validation are implemented once (brief §13). An agent
subclass supplies only its domain logic — which tools to call and how to assemble the
result — and inherits the guarantees.

Design position: **tool selection is deterministic, tool execution is instrumented.** The
agents do not let the LLM choose the next tool from an open set; they follow a fixed,
auditable plan, and the harness enforces ReAct-style bounds around it. This is the brief's
"deterministic workflow control" applied at the agent level — reproducible, testable, and
immune to a prompt-injection attempt steering the tool sequence (ADR-009, threat T-1). The
LLM's only job is to *narrate* the deterministic result.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from vm_contracts.mcp import ToolResult
from vm_harness.mcp_client import ToolClient, ToolClientError
from vm_harness.trace import ActionStatus, AgentRunContext
from vm_llm.base import LLMProvider
from vm_llm.types import LLMError, Message

__all__ = ["AgentHarness"]


class AgentHarness:
    """Runs instrumented tool calls and structured LLM narration for one agent."""

    def __init__(
        self,
        *,
        agent_name: str,
        tool_client: ToolClient,
        llm: LLMProvider,
        max_tool_retries: int = 2,
    ) -> None:
        self.agent_name = agent_name
        self._tools = tool_client
        self._llm = llm
        self._max_tool_retries = max_tool_retries

    async def call_tool(
        self,
        context: AgentRunContext,
        *,
        tool: str,
        arguments: dict[str, Any],
        result_counter: str | None = None,
        summary: str = "",
        retry_transient: bool = True,
    ) -> ToolResult:
        """Call an MCP tool under the run's budget, recording an audit entry.

        Args:
            context: The run context whose budget and trace this call consumes.
            tool: Tool name.
            arguments: Tool arguments (redacted before they enter the trace).
            result_counter: A ``data`` key whose length is the meaningful "result count"
                for the trace (e.g. ``"offers"``).
            summary: Concise, auditable description of the intent.
            retry_transient: Whether to retry when the tool reports ``retryable``.

        Returns:
            The :class:`ToolResult`. A tool that ran and failed returns ``ok=False``; only a
            transport failure after the retry budget raises.

        Raises:
            StoppingConditionReached: A budget was spent before the call.
            ToolClientError: The server was unreachable after retries.
        """
        context.check_can_continue()
        started = time.perf_counter()
        attempts = 0
        last_error: ToolClientError | None = None

        # Bounded retry — brief §12 caps MCP retries at 2. A retry counts as one tool call
        # against the budget, so a flapping dependency cannot be retried into a budget
        # overrun.
        while attempts <= self._max_tool_retries:
            attempts += 1
            try:
                result = await self._tools.call(tool, arguments)
            except ToolClientError as exc:
                last_error = exc
                if attempts > self._max_tool_retries:
                    break
                continue

            if not result.ok and result.error and result.error.retryable and retry_transient:
                if attempts > self._max_tool_retries:
                    pass  # fall through and record the failure below
                else:
                    continue

            latency = int((time.perf_counter() - started) * 1000)
            count = _result_count(result, result_counter)
            context.record_tool(
                tool=tool,
                arguments=arguments,
                status=ActionStatus.OK if result.ok else ActionStatus.FAILED,
                latency_ms=latency,
                result_count=count,
                retry_count=attempts - 1,
                summary=summary or _default_summary(tool, result, count),
            )
            for warning in result.warnings:
                context.warn(warning)
            if not result.ok and result.error is not None:
                context.warn(result.error.message)
            if result.origin == "mocked":
                context.used_mock_data = True
            return result

        # Transport failure after exhausting retries. Record it and re-raise so the agent
        # can decide whether the task can still produce a partial result.
        latency = int((time.perf_counter() - started) * 1000)
        context.record_tool(
            tool=tool,
            arguments=arguments,
            status=ActionStatus.FAILED,
            latency_ms=latency,
            retry_count=attempts - 1,
            summary=f"'{tool}' unreachable after {attempts} attempt(s)",
        )
        context.degraded = True
        raise last_error or ToolClientError(f"'{tool}' failed")

    async def narrate[T: BaseModel](
        self,
        context: AgentRunContext,
        *,
        system: str,
        user: str,
        response_model: type[T],
        temperature: float = 0.0,
    ) -> T | None:
        """Produce a structured narrative from the LLM under the run's budget.

        Returns ``None`` — never raises — when the LLM is unavailable or its output will not
        validate. Narration is a presentation concern; a plan with correct numbers and no
        prose is still a usable plan, so a narration failure degrades rather than fails the
        task (brief §23).
        """
        try:
            context.check_can_continue()
        except Exception:
            context.warn("Reasoning summary skipped: agent budget was exhausted.")
            return None

        started = time.perf_counter()
        try:
            response = await self._llm.generate_structured(
                [Message.system(system), Message.user(user)],
                response_model,
                temperature=temperature,
                timeout_seconds=max(2.0, context.remaining_seconds),
            )
        except LLMError as exc:
            context.record_llm(
                usage=_zero_usage(),
                latency_ms=int((time.perf_counter() - started) * 1000),
                summary=f"narration unavailable ({type(exc).__name__})",
                status=ActionStatus.FAILED,
            )
            context.warn("A human-readable reasoning summary could not be generated.")
            context.degraded = True
            return None

        context.record_llm(
            usage=response.usage,
            latency_ms=response.latency_ms,
            summary="generated the reasoning summary",
            status=ActionStatus.OK,
        )
        if response.repaired:
            context.warn("The reasoning summary required output repair.")
        return response.value


def _result_count(result: ToolResult, key: str | None) -> int:
    if key is None:
        return 0
    value = result.data.get(key)
    return len(value) if isinstance(value, list) else 0


def _default_summary(tool: str, result: ToolResult, count: int) -> str:
    if not result.ok:
        code = result.error.code if result.error else "unknown"
        return f"{tool} failed ({code})"
    if count:
        return f"{tool} returned {count} result(s) from {result.source_name}"
    return f"{tool} completed ({result.origin})"


def _zero_usage() -> Any:
    from vm_llm.types import TokenUsage

    return TokenUsage()
