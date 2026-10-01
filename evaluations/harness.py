"""Builds an orchestrator for one evaluation case, applying its injected fault.

Deliberately mirrors the integration harness: the real graph, the three real agents in-process
over their real MCP servers, and the mock LLM — fully offline and deterministic. Faults are
injected by swapping one component (a downed agent, a slow agent, a dead cache, a failing LLM),
so a case exercises the *pipeline's* behaviour under that condition, not a stub of it.
"""

from __future__ import annotations

import asyncio
from typing import Any

from vm_config.settings import Settings
from vm_contracts.a2a import AgentArtifact, AgentCard, AgentSkill, AgentTask
from vm_destination_mcp.server import build_server as build_destination
from vm_harness import (
    A2ADiscoveryError,
    A2AResponseError,
    InProcessAgentClient,
    InProcessToolClient,
)
from vm_itinerary_agent.agent import ItineraryAgent
from vm_llm.mock_provider import MockLLMProvider
from vm_llm.types import LLMRateLimitError, StructuredOutputError
from vm_lodging_mcp.server import build_server as build_lodging
from vm_orchestrator import OrchestratorDependencies, TravelOrchestrator
from vm_stay_agent.agent import StayAgent
from vm_transport_agent.agent import TransportAgent
from vm_transport_mcp.server import build_server as build_transport

__all__ = ["build_orchestrator_for_fault", "settings_for_fault"]

# A short deadline used for the "slow agent" fault, so the hung agent is cut by the request
# deadline (an honest partial) rather than the test waiting on it.
_SLOW_DEADLINE_SECONDS = 0.5


class _FailingAgentClient:
    """An agent whose calls always fail — a downed A2A service."""

    async def discover(self) -> Any:
        raise A2ADiscoveryError("agent unreachable")

    async def submit_task(self, task: AgentTask) -> AgentArtifact:
        raise A2AResponseError("agent unreachable")


class _SlowAgentClient:
    """An agent that hangs past the deadline — a timed-out upstream."""

    def __init__(self, skill: str) -> None:
        self._skill = skill

    async def discover(self) -> AgentCard:
        return AgentCard(
            name="slow",
            display_name="Slow",
            description="hangs",
            version="0.1.0",
            url="http://slow:8000",
            skills=[
                AgentSkill(
                    name=self._skill,
                    description="slow",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                )
            ],
        )

    async def submit_task(self, task: AgentTask) -> AgentArtifact:
        await asyncio.sleep(5)
        raise AssertionError("should have been cancelled by the deadline")


def settings_for_fault(base: Settings, fault: str | None) -> Settings:
    """Return settings adjusted for the fault (e.g. a short deadline for a slow agent)."""
    if fault and fault.startswith("agent_slow"):
        return Settings.for_testing(
            PROVIDER_MODE="mock",
            ENVIRONMENT="test",
            OTEL_ENABLED="false",
            LIMIT_MAX_REQUEST_DURATION_SECONDS=_SLOW_DEADLINE_SECONDS,
        )
    return base


def build_orchestrator_for_fault(settings: Settings, fault: str | None) -> TravelOrchestrator:
    llm: Any = MockLLMProvider()
    if fault == "llm_rate_limit":
        llm = MockLLMProvider(fail_with=LLMRateLimitError("rate limited"), fail_times=999)
    elif fault == "llm_malformed":
        llm = MockLLMProvider(fail_with=StructuredOutputError("schema mismatch"), fail_times=999)

    transport: Any = InProcessAgentClient(
        TransportAgent(tool_client=InProcessToolClient(build_transport(settings)), llm=llm)
    )
    stay: Any = InProcessAgentClient(
        StayAgent(tool_client=InProcessToolClient(build_lodging(settings)), llm=llm)
    )
    itinerary: Any = InProcessAgentClient(
        ItineraryAgent(tool_client=InProcessToolClient(build_destination(settings)), llm=llm)
    )

    if fault == "agent_down:transport":
        transport = _FailingAgentClient()
    elif fault == "agent_down:stay":
        stay = _FailingAgentClient()
    elif fault == "agent_down:itinerary":
        itinerary = _FailingAgentClient()
    elif fault == "agent_slow:transport":
        transport = _SlowAgentClient("plan_transport")

    cache = _build_cache(settings, fault)
    deps = OrchestratorDependencies(
        settings=settings,
        transport_client=transport,
        stay_client=stay,
        itinerary_client=itinerary,
        cache=cache,
    )
    return TravelOrchestrator(deps)


def _build_cache(settings: Settings, fault: str | None) -> Any:
    """A fakeredis-backed plan cache, or a dead-Redis one for the redis_down fault (fail-open)."""
    from vm_caching import RedisClient, RedisPlanCache

    if fault == "redis_down":
        # Point at an unreachable Redis so every operation exercises the fail-open path.
        dead = RedisClient(Settings.for_testing(REDIS_URL="redis://127.0.0.1:1/0").redis)
        return RedisPlanCache(dead, settings.cache_ttl)

    from fakeredis.aioredis import FakeRedis

    client = RedisClient(settings.redis, redis=FakeRedis())
    return RedisPlanCache(client, settings.cache_ttl)
