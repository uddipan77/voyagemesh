"""Agent-level contract tests.

The orchestrator (Phase 7-8) will validate each agent's artifact against the shared result
models and discover each agent through its card. These tests lock in that compatibility now:
an agent's advertised output schema must match what it actually returns, and its artifact
must round-trip through the shared result model — the exact operation the orchestrator
performs across the A2A boundary (brief §7).
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from vm_config.settings import Settings
from vm_contracts.a2a import AgentArtifact, AgentCard, AgentTask, TaskStatus
from vm_contracts.agent_results import ItineraryResult, StayResult, TransportResult
from vm_destination_mcp.server import build_server as build_destination
from vm_harness import InProcessToolClient
from vm_itinerary_agent.agent import ItineraryAgent
from vm_llm.mock_provider import MockLLMProvider
from vm_lodging_mcp.server import build_server as build_lodging
from vm_stay_agent.agent import StayAgent
from vm_transport_agent.agent import TransportAgent
from vm_transport_mcp.server import build_server as build_transport

pytestmark = pytest.mark.contract


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _settings() -> Settings:
    return Settings.for_testing(PROVIDER_MODE="mock", ENVIRONMENT="test")


CASES = [
    (
        build_transport,
        TransportAgent,
        TransportResult,
        "plan_transport",
        {
            "origin": "Nuremberg",
            "destination": "Prague",
            "departure_date": "2026-08-10",
            "max_duration_hours": 8,
        },
    ),
    (
        build_lodging,
        StayAgent,
        StayResult,
        "plan_accommodation",
        {"destination": "Prague", "check_in": "2026-08-10", "check_out": "2026-08-13"},
    ),
    (
        build_destination,
        ItineraryAgent,
        ItineraryResult,
        "plan_itinerary",
        {
            "origin": "Nuremberg",
            "destination": "Prague",
            "departure_date": "2026-08-10",
            "return_date": "2026-08-13",
            "interests": ["history"],
        },
    ),
]


def _build_agent(build_server: Any, agent_cls: Any) -> Any:
    return agent_cls(
        tool_client=InProcessToolClient(build_server(_settings())), llm=MockLLMProvider()
    )


class TestAgentCards:
    @pytest.mark.parametrize(("build_server", "agent_cls", "_result", "skill", "_p"), CASES)
    def test_card_is_a_valid_agent_card(self, build_server, agent_cls, _result, skill, _p):
        card = _build_agent(build_server, agent_cls).agent_card(url="http://agent:8000")
        # Re-validate through the shared AgentCard model — the orchestrator will do this on
        # discovery, so it must succeed.
        restored = AgentCard.model_validate(card.model_dump(mode="json"))
        assert restored.has_skill(skill)
        assert restored.protocol_version

    @pytest.mark.parametrize(("build_server", "agent_cls", "_result", "skill", "_p"), CASES)
    def test_advertised_output_schema_matches_the_real_result_model(
        self, build_server, agent_cls, _result, skill, _p
    ):
        """The card's output schema must be the result model the agent actually returns."""
        card = _build_agent(build_server, agent_cls).agent_card(url="http://agent:8000")
        advertised = card.skill(skill).output_schema
        assert advertised == _result.model_json_schema()


class TestArtifactContract:
    @pytest.mark.parametrize(
        ("build_server", "agent_cls", "result_model", "skill", "payload"), CASES
    )
    async def test_artifact_result_round_trips_through_the_shared_model(
        self, build_server, agent_cls, result_model, skill, payload
    ):
        """The core A2A handoff: the orchestrator re-validates the artifact's result.

        This is exactly the operation that failed before StrictModel learned to drop its own
        computed fields — so it is worth asserting per agent."""
        agent = _build_agent(build_server, agent_cls)
        task = AgentTask(skill=skill, correlation_id="c", request_id="r", payload=payload)
        artifact = await agent.handle(task)

        assert artifact.status is TaskStatus.COMPLETED
        # 1. The artifact itself round-trips through the wire model.
        restored = AgentArtifact.model_validate(artifact.model_dump(mode="json"))
        assert restored.matches(task)
        # 2. Its result round-trips through the shared domain result model.
        result = result_model.model_validate(restored.result)
        assert result.has_result

    @pytest.mark.parametrize(("build_server", "agent_cls", "_result", "skill", "payload"), CASES)
    async def test_artifact_reports_honest_accounting(
        self, build_server, agent_cls, _result, skill, payload
    ):
        agent = _build_agent(build_server, agent_cls)
        artifact = await agent.handle(
            AgentTask(skill=skill, correlation_id="c", request_id="r", payload=payload)
        )
        assert artifact.tool_call_count >= 1
        assert artifact.duration_ms >= 0
        assert artifact.agent_name
        assert artifact.used_mock_data is True  # mock provider mode
        # Action summaries are the shareable trace; none may be empty.
        assert all(s.strip() for s in artifact.action_summaries)
