"""The LangGraph orchestrator's typed state.

LangGraph merges the partial dict each node returns into this state. Fields that *accumulate*
across parallel branches (warnings, degraded services) use an ``Annotated`` reducer so a
concurrent transport and stay node cannot clobber each other's contributions; everything
else is last-write-wins, which is safe because no two nodes write the same scalar field.

The state carries the fourteen sections' inputs as they are gathered, plus the audit fields
the brief lists (§10): iteration count, agent task IDs, cache status, degraded services.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from vm_contracts.agent_results import ItineraryResult, StayResult, TransportResult
from vm_contracts.budget import BudgetSummary
from vm_contracts.plan import DataSourceInfo, PlanStatus, TripPlan
from vm_contracts.trip import NormalizedTripRequest, TripRequest

__all__ = ["OrchestratorState", "unique_extend"]


def unique_extend(existing: list[Any], incoming: list[Any]) -> list[Any]:
    """Reducer that appends without duplicating — for warnings and degraded-service lists.

    ``operator.add`` would duplicate a warning added on a replan pass; this keeps the lists
    honest across the loop.
    """
    result = list(existing)
    for item in incoming:
        if item not in result:
            result.append(item)
    return result


class OrchestratorState(TypedDict, total=False):
    """State threaded through the planning graph."""

    # --- identity / correlation ---
    request_id: str
    trace_id: str
    trip_id: str
    user_reference: str | None
    user_roles: list[str]

    # --- request ---
    trip_request: TripRequest
    normalized: NormalizedTripRequest

    # --- gathered results ---
    transport: TransportResult | None
    accommodation: StayResult | None
    itinerary: ItineraryResult | None
    budget: BudgetSummary | None
    data_sources: Annotated[list[DataSourceInfo], operator.add]

    # --- control ---
    cached_plan: TripPlan | None
    """A plan served from cache; when present the graph short-circuits and the orchestrator
    returns it directly rather than re-assembling one."""

    iteration_count: int
    """Number of replans performed. Bounded by ``LIMIT_MAX_REPLANS``."""

    max_replans: int
    accommodation_ceiling: float | None
    """A tightened per-stay budget set by the replan node; ``None`` on the first pass."""

    agent_task_ids: Annotated[list[str], operator.add]
    cache_status: str

    # --- accumulating diagnostics (parallel-safe reducers) ---
    warnings: Annotated[list[str], unique_extend]
    degraded_services: Annotated[list[str], unique_extend]
    validation_errors: Annotated[list[str], unique_extend]

    # --- outcome ---
    final_status: PlanStatus
    trade_off_explanation: str
    reasoning_summary: str
