"""The orchestrator's injected dependencies.

Bundling agent clients, the cache, and configuration behind one object keeps the nodes as
closures over a single ``deps`` and makes the whole orchestrator trivially testable: a test
supplies in-process agent clients and a no-op cache, and the same graph runs offline.
"""

from __future__ import annotations

from typing import Protocol

from vm_config.settings import Settings
from vm_contracts.a2a import AgentArtifact, AgentTask
from vm_contracts.plan import TripPlan
from vm_contracts.trip import NormalizedTripRequest
from vm_harness import AgentClient, submit_verified

__all__ = ["NoOpPlanCache", "OrchestratorDependencies", "PlanCache"]


class PlanCache(Protocol):
    """A cache of finished plans keyed by the normalised request (brief §17)."""

    async def get_plan(self, normalized: NormalizedTripRequest) -> TripPlan | None:
        """Return a fresh cached plan, or ``None`` on a miss. Must not raise."""
        ...

    async def put_plan(self, normalized: NormalizedTripRequest, plan: TripPlan) -> None:
        """Store a plan. Must not raise — a cache failure degrades to a miss (brief §17)."""
        ...


class NoOpPlanCache:
    """A cache that never hits. The real Redis implementation lands in Phase 10.

    Using a no-op rather than ``None`` means the ``check_cache`` node needs no special-casing
    and the cache-miss path is exercised exactly as it will be in production.
    """

    async def get_plan(self, normalized: NormalizedTripRequest) -> TripPlan | None:
        return None

    async def put_plan(self, normalized: NormalizedTripRequest, plan: TripPlan) -> None:
        return None


class OrchestratorDependencies:
    """Everything the graph nodes need, injected once."""

    def __init__(
        self,
        *,
        settings: Settings,
        transport_client: AgentClient,
        stay_client: AgentClient,
        itinerary_client: AgentClient,
        cache: PlanCache | None = None,
    ) -> None:
        self.settings = settings
        self.cache: PlanCache = cache or NoOpPlanCache()
        self._clients: dict[str, AgentClient] = {
            "transport": transport_client,
            "stay": stay_client,
            "itinerary": itinerary_client,
        }

    def agent_client(self, agent: str) -> AgentClient:
        return self._clients[agent]

    async def submit(
        self, client: AgentClient, task: AgentTask, *, required_skill: str
    ) -> AgentArtifact:
        """Discover-verify-submit, the orchestrator's full §7 obligation.

        Indirected through ``deps`` so a test can stub it, but the default is the real
        ``submit_verified`` that verifies the skill and validates the response.
        """
        return await submit_verified(client, task, required_skill=required_skill)
