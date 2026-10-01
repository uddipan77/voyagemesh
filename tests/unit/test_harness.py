"""Agent harness: stopping conditions, action trace, tool client, and narration.

The stopping-condition tests are the load-bearing ones: they are what make "no agent runs
unbounded" (brief §12) a proven property rather than a claim.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from vm_contracts.mcp import ToolError, ToolResult
from vm_contracts.tracing import TraceContext
from vm_harness import (
    ActionStatus,
    AgentHarness,
    AgentRunContext,
    InProcessToolClient,
    StoppingConditionReached,
    StopReason,
    ToolClientError,
)
from vm_harness.mcp_client import _parse_tool_output
from vm_llm.mock_provider import MockLLMProvider
from vm_llm.types import LLMTimeoutError

pytestmark = pytest.mark.unit


def _context(**kwargs: Any) -> AgentRunContext:
    return AgentRunContext(agent_name="test-agent", trace=TraceContext.new(), **kwargs)


class Narrative(BaseModel):
    model_config = {"extra": "forbid"}
    headline: Annotated[str, Field(min_length=3, max_length=80)]
    detail: Annotated[str, Field(min_length=5, max_length=300)]


class FakeToolClient:
    """A scripted tool client: maps tool name to a ToolResult or an exception."""

    def __init__(self, script: dict[str, Any]) -> None:
        self.script = script
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append((tool, arguments))
        outcome = self.script.get(tool)
        if isinstance(outcome, list):  # a sequence: one per successive call
            outcome = outcome[min(len(self.calls) - 1, len(outcome) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            raise ToolClientError(f"no script for '{tool}'")
        return outcome

    async def list_tools(self) -> list[str]:
        return list(self.script)


def _ok(**data: Any) -> ToolResult:
    return ToolResult.success(origin="mocked", source_name="test", data=data)


def _fail(*, retryable: bool) -> ToolResult:
    return ToolResult(
        ok=False,
        origin="unavailable",
        source_name="test",
        error=ToolError(code="boom", message="failed", retryable=retryable),
    )


class TestStoppingConditions:
    def test_tool_call_budget_is_enforced(self):
        context = _context(max_tool_calls=2)
        context.tool_calls = 2
        with pytest.raises(StoppingConditionReached) as exc:
            context.check_can_continue()
        assert exc.value.reason is StopReason.MAX_TOOL_CALLS

    def test_llm_call_budget_is_enforced(self):
        context = _context(max_llm_calls=1)
        context.llm_calls = 1
        with pytest.raises(StoppingConditionReached) as exc:
            context.check_can_continue()
        assert exc.value.reason is StopReason.MAX_LLM_CALLS

    def test_deadline_is_enforced(self):
        context = _context(deadline_seconds=0.0)
        with pytest.raises(StoppingConditionReached) as exc:
            context.check_can_continue()
        assert exc.value.reason is StopReason.DEADLINE

    def test_within_budget_does_not_raise(self):
        _context(max_tool_calls=8).check_can_continue()  # no exception


class TestActionTrace:
    def test_records_tool_calls_with_a_summary(self):
        context = _context()
        context.record_tool(
            tool="search",
            arguments={"q": "x"},
            status=ActionStatus.OK,
            latency_ms=5,
            result_count=3,
            summary="searched",
        )
        assert context.tool_calls == 1
        assert context.action_summaries() == ["searched"]

    def test_trace_redacts_credential_shaped_arguments(self):
        """The action trace is auditable but must never carry a secret."""
        context = _context()
        context.record_tool(
            tool="call",
            arguments={"token": "gsk_secret", "authorization": "Bearer abc", "origin": "Prague"},
            status=ActionStatus.OK,
            latency_ms=1,
        )
        recorded = context.trace_records()[0]["arguments"]
        assert recorded["token"] == "***"
        assert recorded["authorization"] == "***"
        assert recorded["origin"] == "Prague"  # ordinary values pass through

    def test_trace_summarises_large_arguments(self):
        context = _context()
        context.record_tool(
            tool="rank",
            arguments={"offers": [{}] * 20, "note": "x" * 500},
            status=ActionStatus.OK,
            latency_ms=1,
        )
        recorded = context.trace_records()[0]["arguments"]
        assert recorded["offers"] == "[20 items]"
        assert recorded["note"].endswith("…")

    def test_action_summaries_contain_no_reasoning_tokens(self):
        """Brief §11: store concise action summaries, never raw chain-of-thought.

        The trace records only outcomes; there is no field that could hold model reasoning.
        """
        context = _context()
        context.record_llm(
            usage=_zero(),
            latency_ms=10,
            summary="generated the reasoning summary",
            status=ActionStatus.OK,
        )
        record = context.trace_records()[0]
        assert set(record) == {
            "step",
            "action",
            "selected_tool",
            "arguments",
            "status",
            "latency_ms",
            "result_count",
            "retry_count",
            "decision_summary",
        }
        # No key that could carry hidden reasoning.
        assert "reasoning_tokens" not in record
        assert "chain_of_thought" not in record


def _zero():
    from vm_llm.types import TokenUsage

    return TokenUsage()


class TestHarnessToolCalls:
    async def test_successful_call_records_and_counts(self):
        client = FakeToolClient({"search": _ok(offers=[1, 2, 3])})
        harness = AgentHarness(agent_name="a", tool_client=client, llm=MockLLMProvider())
        context = _context()
        result = await harness.call_tool(
            context, tool="search", arguments={"q": "x"}, result_counter="offers"
        )
        assert result.ok
        assert context.tool_calls == 1
        assert context.actions[0].result_count == 3

    async def test_mock_origin_flags_used_mock_data(self):
        client = FakeToolClient({"search": _ok(offers=[])})
        harness = AgentHarness(agent_name="a", tool_client=client, llm=MockLLMProvider())
        context = _context()
        await harness.call_tool(context, tool="search", arguments={})
        assert context.used_mock_data is True

    async def test_retryable_failure_is_retried_then_recorded(self):
        # Fails twice (retryable), then succeeds.
        client = FakeToolClient(
            {"search": [_fail(retryable=True), _fail(retryable=True), _ok(offers=[1])]}
        )
        harness = AgentHarness(agent_name="a", tool_client=client, llm=MockLLMProvider())
        context = _context()
        result = await harness.call_tool(context, tool="search", arguments={})
        assert result.ok
        assert len(client.calls) == 3  # 1 + 2 retries

    async def test_non_retryable_failure_is_not_retried(self):
        client = FakeToolClient({"search": _fail(retryable=False)})
        harness = AgentHarness(agent_name="a", tool_client=client, llm=MockLLMProvider())
        context = _context()
        result = await harness.call_tool(context, tool="search", arguments={})
        assert result.ok is False
        assert len(client.calls) == 1

    async def test_transport_failure_raises_after_retries_and_marks_degraded(self):
        client = FakeToolClient({"search": ToolClientError("unreachable")})
        harness = AgentHarness(agent_name="a", tool_client=client, llm=MockLLMProvider())
        context = _context()
        with pytest.raises(ToolClientError):
            await harness.call_tool(context, tool="search", arguments={})
        assert context.degraded is True

    async def test_a_call_over_budget_raises_before_dispatch(self):
        client = FakeToolClient({"search": _ok()})
        harness = AgentHarness(agent_name="a", tool_client=client, llm=MockLLMProvider())
        context = _context(max_tool_calls=1)
        context.tool_calls = 1
        with pytest.raises(StoppingConditionReached):
            await harness.call_tool(context, tool="search", arguments={})
        assert client.calls == [], "no dispatch once the budget is spent"


class TestHarnessNarration:
    async def test_narration_returns_a_validated_model(self):
        harness = AgentHarness(
            agent_name="a", tool_client=FakeToolClient({}), llm=MockLLMProvider()
        )
        context = _context()
        result = await harness.narrate(context, system="s", user="u", response_model=Narrative)
        assert isinstance(result, Narrative)
        assert context.llm_calls == 1

    async def test_narration_failure_degrades_rather_than_raises(self):
        """A plan with correct numbers and no prose is still usable (brief §23)."""
        harness = AgentHarness(
            agent_name="a",
            tool_client=FakeToolClient({}),
            llm=MockLLMProvider(fail_with=LLMTimeoutError("slow")),
        )
        context = _context()
        result = await harness.narrate(context, system="s", user="u", response_model=Narrative)
        assert result is None
        assert context.degraded is True
        assert any("reasoning summary" in w for w in context.warnings)

    async def test_narration_is_skipped_when_budget_is_exhausted(self):
        harness = AgentHarness(
            agent_name="a", tool_client=FakeToolClient({}), llm=MockLLMProvider()
        )
        context = _context(max_llm_calls=1)
        context.llm_calls = 1
        result = await harness.narrate(context, system="s", user="u", response_model=Narrative)
        assert result is None


class TestParseToolOutput:
    def test_accepts_the_structured_tuple_form(self):
        raw = (["content"], _ok(offers=[1]).model_dump(mode="json"))
        result = _parse_tool_output(raw, tool="t")
        assert result.ok

    def test_rejects_a_non_object_result(self):
        with pytest.raises(ToolClientError):
            _parse_tool_output(("content", 42), tool="t")

    def test_rejects_a_contract_violating_shape(self):
        with pytest.raises(ToolClientError, match="contract"):
            _parse_tool_output({"unexpected": "shape"}, tool="t")


class TestInProcessClientErrorHandling:
    async def test_dispatch_failure_becomes_a_tool_client_error(self):
        class Broken:
            async def call_tool(self, *a: Any, **k: Any) -> Any:
                raise RuntimeError("dispatch broke")

        client = InProcessToolClient(Broken())
        with pytest.raises(ToolClientError):
            await client.call("x", {})

    async def test_list_tools_maps_names(self):
        class Server:
            async def list_tools(self) -> Any:
                return [type("T", (), {"name": "search"})(), type("T", (), {"name": "rank"})()]

        assert await InProcessToolClient(Server()).list_tools() == ["search", "rank"]

    async def test_aclose_is_a_noop(self):
        await InProcessToolClient(object()).aclose()  # must not raise


class TestHttpMCPToolClient:
    """The HTTP client's happy path needs a live server (covered by E2E in Phase 16). These
    cover construction and the failure mapping, which are the parts that can go wrong
    without a network."""

    def test_base_url_is_normalised_to_the_mcp_endpoint(self):
        from vm_harness import HttpMCPToolClient

        client = HttpMCPToolClient("http://transport-agent:8110/")
        assert client._base_url == "http://transport-agent:8110/mcp"

    async def test_unreachable_server_raises_tool_client_error(self):
        from vm_harness import HttpMCPToolClient

        # A port nothing is listening on: the connection attempt must surface as a
        # ToolClientError, not a raw transport exception.
        client = HttpMCPToolClient("http://127.0.0.1:1/", timeout_seconds=1.0)
        with pytest.raises(ToolClientError, match=r"could not reach|no structured content"):
            await client.call("search", {})

    async def test_aclose_is_a_noop(self):
        from vm_harness import HttpMCPToolClient

        async with HttpMCPToolClient("http://x:8000") as client:
            assert client is not None
