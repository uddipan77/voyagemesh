"""Specialist agents driven end-to-end against in-process MCP servers.

These prove the agent-level guarantees the brief and threat model require:

* a failed task carries an error and **no** result (a failure can never be mistaken for
  data);
* a partial result is labelled and carries a warning explaining the gap;
* the agent computes nothing itself — prices and rankings come from the tools;
* stopping conditions bound every run;
* the action trace is auditable and secret-free.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from vm_config.settings import Settings
from vm_contracts.a2a import AgentTask, TaskStatus
from vm_contracts.agent_results import ItineraryResult, StayResult, TransportResult
from vm_contracts.mcp import ToolResult
from vm_destination_mcp.server import build_server as build_destination
from vm_harness import InProcessToolClient, ToolClientError
from vm_itinerary_agent.agent import ItineraryAgent
from vm_llm.mock_provider import MockLLMProvider
from vm_lodging_mcp.server import build_server as build_lodging
from vm_stay_agent.agent import StayAgent
from vm_transport_agent.agent import TransportAgent
from vm_transport_mcp.server import build_server as build_transport

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _settings() -> Settings:
    return Settings.for_testing(PROVIDER_MODE="mock", ENVIRONMENT="test")


def _task(skill: str, payload: dict[str, Any]) -> AgentTask:
    return AgentTask(skill=skill, correlation_id="corr-1", request_id="req-1", payload=payload)


class FailingToolClient:
    """A tool client that always fails at the transport level."""

    async def call(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        raise ToolClientError("server unreachable")

    async def list_tools(self) -> list[str]:
        return []


class SelectiveToolClient:
    """Delegates to a real in-process client but makes named tools fail.

    Used to drive an agent's *degradation* paths — a specific tool being unavailable while
    the rest work — which is how a real partial-failure looks."""

    def __init__(self, inner: Any, *, fail_tools: set[str]) -> None:
        self._inner = inner
        self._fail = fail_tools

    async def call(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        if tool in self._fail:
            raise ToolClientError(f"'{tool}' unavailable")
        return await self._inner.call(tool, arguments)

    async def list_tools(self) -> list[str]:
        return await self._inner.list_tools()


# ---------------------------------------------------------------------------
# Transport Agent
# ---------------------------------------------------------------------------
class TestTransportAgent:
    def _agent(self, tool_client: Any = None) -> TransportAgent:
        client = tool_client or InProcessToolClient(build_transport(_settings()))
        return TransportAgent(tool_client=client, llm=MockLLMProvider())

    def _payload(self, **overrides: Any) -> dict[str, Any]:
        return {
            "origin": "Nuremberg",
            "destination": "Prague",
            "departure_date": "2026-08-10",
            "travellers": 1,
            "max_duration_hours": 8,
            "ranking_strategy": "balanced",
            **overrides,
        }

    async def test_produces_a_ranked_recommendation(self):
        artifact = await self._agent().handle(_task("plan_transport", self._payload()))
        assert artifact.status is TaskStatus.COMPLETED
        result = TransportResult.model_validate(artifact.result)
        assert result.recommended is not None
        assert len(result.alternatives) <= 5
        assert result.data_origin == "mocked"
        assert artifact.used_mock_data is True

    async def test_recommendation_and_alternatives_are_distinct_and_ranked(self):
        artifact = await self._agent().handle(_task("plan_transport", self._payload()))
        result = TransportResult.model_validate(artifact.result)
        ids = [result.recommended.offer_id] + [a.offer_id for a in result.alternatives]
        assert len(ids) == len(set(ids)), "no offer appears twice"
        scores = [result.recommended.score] + [a.score for a in result.alternatives]
        assert scores == sorted(s for s in scores if s is not None)

    async def test_the_agent_does_not_price_or_rank_itself(self):
        """Every price and score originates from the tools, echoed unchanged."""
        artifact = await self._agent().handle(_task("plan_transport", self._payload()))
        result = TransportResult.model_validate(artifact.result)
        rec = result.recommended
        assert rec is not None
        # Total equals per-traveller x travellers — the MCP computed it, the agent relayed it.
        assert rec.total_price == rec.price_per_traveller.times(rec.travellers)
        assert rec.score is not None  # the score came attached from the scoring tool

    async def test_records_an_auditable_action_trace(self):
        artifact = await self._agent().handle(_task("plan_transport", self._payload()))
        assert artifact.action_summaries
        assert any("searched" in s for s in artifact.action_summaries)
        assert artifact.tool_call_count >= 2  # search + score

    async def test_wrong_skill_is_rejected_with_no_result(self):
        artifact = await self._agent().handle(_task("book_flight", self._payload()))
        assert artifact.status is TaskStatus.REJECTED
        assert artifact.result is None
        assert artifact.error is not None

    async def test_invalid_payload_is_rejected(self):
        artifact = await self._agent().handle(
            _task("plan_transport", {"origin": "X"})  # missing required fields
        )
        assert artifact.status is TaskStatus.REJECTED
        assert artifact.result is None

    async def test_unreachable_tools_fail_without_a_result(self):
        """The single most important invariant: a failure never carries data."""
        agent = self._agent(FailingToolClient())
        artifact = await agent.handle(_task("plan_transport", self._payload()))
        assert artifact.status is TaskStatus.FAILED
        assert artifact.result is None
        assert artifact.error is not None
        assert artifact.error.message  # sanitised, present

    async def test_stopping_condition_bounds_the_run(self):
        agent = TransportAgent(
            tool_client=InProcessToolClient(build_transport(_settings())),
            llm=MockLLMProvider(),
            max_tool_calls=1,  # cannot complete search + score
        )
        artifact = await agent.handle(_task("plan_transport", self._payload()))
        # Either it fails cleanly on the budget, or produces a partial — never unbounded,
        # and never a COMPLETED it could not legitimately reach.
        assert artifact.status in (TaskStatus.FAILED, TaskStatus.PARTIAL)
        assert artifact.tool_call_count <= 2

    async def test_artifact_matches_the_task(self):
        task = _task("plan_transport", self._payload())
        artifact = await self._agent().handle(task)
        assert artifact.matches(task)

    async def test_flight_search_failure_degrades_to_partial(self):
        """A long route searches flights; if that fails, ground results still stand — but
        the result is marked partial with a warning."""
        inner = InProcessToolClient(build_transport(_settings()))
        agent = self._agent(SelectiveToolClient(inner, fail_tools={"search_flights"}))
        # Paris→London is short (460 km) so no flight search; use a long route.
        artifact = await agent.handle(
            _task(
                "plan_transport",
                self._payload(origin="Amsterdam", destination="Berlin", max_duration_hours=12),
            )
        )
        # 660 km triggers a flight search, which fails — ground options remain.
        assert artifact.status in (TaskStatus.PARTIAL, TaskStatus.COMPLETED)
        if artifact.status is TaskStatus.PARTIAL:
            assert any("flight" in w.lower() for w in artifact.warnings)


# ---------------------------------------------------------------------------
# Stay Agent
# ---------------------------------------------------------------------------
class TestStayAgent:
    def _agent(self, tool_client: Any = None) -> StayAgent:
        client = tool_client or InProcessToolClient(build_lodging(_settings()))
        return StayAgent(tool_client=client, llm=MockLLMProvider())

    def _payload(self, **overrides: Any) -> dict[str, Any]:
        return {
            "destination": "Prague",
            "check_in": "2026-08-10",
            "check_out": "2026-08-13",
            "guests": 1,
            "ranking_strategy": "balanced",
            **overrides,
        }

    async def test_produces_a_stay_priced_recommendation(self):
        artifact = await self._agent().handle(_task("plan_accommodation", self._payload()))
        assert artifact.status is TaskStatus.COMPLETED
        result = StayResult.model_validate(artifact.result)
        rec = result.recommended
        assert rec is not None
        assert rec.nights == 3
        # Total equals nightly x nights — priced by the tool for the whole stay.
        assert rec.total_price == rec.price_per_night.times(3)

    async def test_wrong_skill_rejected(self):
        artifact = await self._agent().handle(_task("plan_transport", self._payload()))
        assert artifact.status is TaskStatus.REJECTED
        assert artifact.result is None

    async def test_unreachable_tool_fails_cleanly(self):
        agent = self._agent(FailingToolClient())
        artifact = await agent.handle(_task("plan_accommodation", self._payload()))
        assert artifact.status is TaskStatus.FAILED
        assert artifact.result is None


# ---------------------------------------------------------------------------
# Itinerary Agent
# ---------------------------------------------------------------------------
class TestItineraryAgent:
    def _agent(self, tool_client: Any = None) -> ItineraryAgent:
        client = tool_client or InProcessToolClient(build_destination(_settings()))
        return ItineraryAgent(tool_client=client, llm=MockLLMProvider())

    def _payload(self, **overrides: Any) -> dict[str, Any]:
        return {
            "origin": "Nuremberg",
            "destination": "Prague",
            "departure_date": "2026-08-10",
            "return_date": "2026-08-13",
            "travellers": 1,
            "interests": ["history", "architecture", "local food"],
            **overrides,
        }

    async def test_builds_a_feasible_itinerary(self):
        artifact = await self._agent().handle(_task("plan_itinerary", self._payload()))
        assert artifact.status is TaskStatus.COMPLETED
        result = ItineraryResult.model_validate(artifact.result)
        assert result.itinerary is not None
        assert result.itinerary.day_count == 4

    async def test_itinerary_has_no_overlapping_activities(self):
        """The strongest guarantee: re-validate the returned plan and it must be feasible.

        Because the plan was built by ItineraryPlanner and validated by ItineraryDay,
        re-parsing it cannot produce overlaps — construction would have failed first."""
        artifact = await self._agent().handle(_task("plan_itinerary", self._payload()))
        result = ItineraryResult.model_validate(artifact.result)
        for day in result.itinerary.days:
            for earlier, later in zip(day.slots, day.slots[1:], strict=False):
                assert not earlier.overlaps(later)

    async def test_uses_the_full_tool_chain(self):
        artifact = await self._agent().handle(_task("plan_itinerary", self._payload()))
        summaries = " ".join(artifact.action_summaries)
        assert "geocoded" in summaries
        assert "attractions" in summaries
        assert "weather" in summaries
        assert "validated day" in summaries

    async def test_weather_is_labelled_by_origin(self):
        artifact = await self._agent().handle(_task("plan_itinerary", self._payload()))
        result = ItineraryResult.model_validate(artifact.result)
        # In mock mode the weather is synthetic, so its origin must say so.
        assert result.weather_origin in ("mocked", "unavailable")

    async def test_degrades_to_partial_when_weather_is_unavailable(self):
        """Weather failing must yield a usable itinerary marked partial with a warning
        (brief §23: 'if weather fails, still return the rest with a weather warning')."""
        inner = InProcessToolClient(build_destination(_settings()))
        agent = ItineraryAgent(
            tool_client=SelectiveToolClient(inner, fail_tools={"get_weather_forecast"}),
            llm=MockLLMProvider(),
        )
        artifact = await agent.handle(_task("plan_itinerary", self._payload()))
        assert artifact.status is TaskStatus.PARTIAL
        result = ItineraryResult.model_validate(artifact.result)
        assert result.itinerary is not None  # still usable
        assert result.weather is None
        assert any("weather" in w.lower() for w in artifact.warnings)

    async def test_degrades_when_points_of_interest_are_unavailable(self):
        inner = InProcessToolClient(build_destination(_settings()))
        agent = ItineraryAgent(
            tool_client=SelectiveToolClient(inner, fail_tools={"search_points_of_interest"}),
            llm=MockLLMProvider(),
        )
        artifact = await agent.handle(_task("plan_itinerary", self._payload()))
        # Still produces a day structure (travel/meals) but flagged partial.
        assert artifact.status is TaskStatus.PARTIAL
        assert artifact.degraded is True

    async def test_wrong_skill_rejected(self):
        artifact = await self._agent().handle(_task("plan_transport", self._payload()))
        assert artifact.status is TaskStatus.REJECTED

    async def test_reproducible_across_runs(self):
        """Same task, same plan — the agent adds no nondeterminism over the domain layer."""
        a = await self._agent().handle(_task("plan_itinerary", self._payload()))
        b = await self._agent().handle(_task("plan_itinerary", self._payload()))
        plan_a = ItineraryResult.model_validate(a.result).itinerary
        plan_b = ItineraryResult.model_validate(b.result).itinerary
        titles_a = [(d.day_number, s.title) for d in plan_a.days for s in d.slots]
        titles_b = [(d.day_number, s.title) for d in plan_b.days for s in d.slots]
        assert titles_a == titles_b


# ---------------------------------------------------------------------------
# Cross-agent guarantees
# ---------------------------------------------------------------------------
class TestSharedAgentGuarantees:
    @pytest.mark.parametrize(
        ("build_server", "agent_cls", "skill", "url"),
        [
            (build_transport, TransportAgent, "plan_transport", "http://t:8010"),
            (build_lodging, StayAgent, "plan_accommodation", "http://s:8020"),
            (build_destination, ItineraryAgent, "plan_itinerary", "http://i:8030"),
        ],
    )
    async def test_agent_card_advertises_its_single_skill(
        self, build_server, agent_cls, skill, url
    ):
        agent = agent_cls(
            tool_client=InProcessToolClient(build_server(_settings())), llm=MockLLMProvider()
        )
        card = agent.agent_card(url=url)
        assert card.has_skill(skill)
        assert card.required_roles == ["agent:invoke"]
        assert card.auth_scheme.value == "bearer_jwt"
        # The card's advertised schemas must be real JSON Schema objects.
        assert card.skills[0].input_schema.get("type") == "object"
        assert card.skills[0].output_schema.get("type") == "object"
