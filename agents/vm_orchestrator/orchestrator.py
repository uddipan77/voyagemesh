"""The Travel Orchestrator.

Owns the compiled graph and the agent clients, runs a trip request through the workflow, and
assembles the final :class:`~vm_contracts.plan.TripPlan` — including the output guardrail pass
that is the last check before a plan is returned (brief §6A, §16C).

The orchestrator enforces the overall request deadline (brief §12): the whole graph run is
wrapped in a timeout, so a hung agent cannot make a request exceed its budget. A timeout
yields an honest partial or failed plan, never a hang.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time

from vm_config.settings import Settings
from vm_contracts.common import ConfidenceLevel
from vm_contracts.plan import PlanStatus, TripPlan
from vm_contracts.tracing import TraceContext, new_request_id
from vm_contracts.trip import TripRequest
from vm_guardrails import check_plan
from vm_orchestrator.dependencies import OrchestratorDependencies
from vm_orchestrator.graph import build_graph
from vm_orchestrator.state import OrchestratorState

__all__ = ["TravelOrchestrator"]

logger = logging.getLogger(__name__)


def _new_trip_id() -> str:
    return f"trip_{secrets.token_hex(10)}"


class TravelOrchestrator:
    """Runs the LangGraph planning workflow for a trip request."""

    def __init__(self, deps: OrchestratorDependencies) -> None:
        self._deps = deps
        self._graph = build_graph(deps)
        self._settings: Settings = deps.settings

    async def plan_trip(
        self,
        request: TripRequest,
        *,
        request_id: str | None = None,
        trace: TraceContext | None = None,
        user_reference: str | None = None,
        user_roles: list[str] | None = None,
    ) -> TripPlan:
        """Plan a trip end to end. Never raises — every failure becomes an honest plan."""
        trace = trace or TraceContext.new(request_id=request_id or new_request_id())
        trip_id = _new_trip_id()

        initial: OrchestratorState = {
            "request_id": trace.request_id,
            "trace_id": trace.trace_id,
            "trip_id": trip_id,
            "user_reference": user_reference,
            "user_roles": user_roles or [],
            "trip_request": request,
            "iteration_count": 0,
            "max_replans": self._settings.limits.max_replans,
            "accommodation_ceiling": None,
            "data_sources": [],
            "agent_task_ids": [],
            "warnings": [],
            "degraded_services": [],
            "validation_errors": [],
            "cache_status": "miss",
        }

        deadline = self._settings.limits.max_request_duration_seconds
        started = time.monotonic()
        try:
            final_state: OrchestratorState = await asyncio.wait_for(
                self._graph.ainvoke(initial), timeout=deadline
            )
        except TimeoutError:
            logger.warning("orchestrator_deadline_exceeded", extra=_log(trace, trip_id))
            return self._timeout_plan(trace, trip_id)
        except Exception:
            # LangGraph runs nodes concurrently and can wrap the cancellation raised when
            # `wait_for` times out into its own exception type, so a genuine deadline breach
            # does not always surface as TimeoutError. If we are at (or past) the deadline,
            # treat it as the timeout it is; otherwise it is a real crash. Either way the
            # request produces an honest plan rather than propagating.
            if time.monotonic() - started >= deadline * 0.9:
                logger.warning("orchestrator_deadline_exceeded", extra=_log(trace, trip_id))
                return self._timeout_plan(trace, trip_id)
            logger.exception("orchestrator_unhandled_error", extra=_log(trace, trip_id))
            return self._error_plan(trace, trip_id, "an unexpected error occurred while planning")

        # A cache hit short-circuits: the stored plan is already validated and
        # guardrail-checked from its original run. Return it under this request's id.
        cached = final_state.get("cached_plan")
        if final_state.get("cache_status") == "hit" and cached is not None:
            return cached.model_copy(
                update={"request_id": trace.request_id, "trip_id": trip_id, "cache_status": "hit"}
            )

        plan = self._assemble(final_state, trace, trip_id)

        # Output guardrails — the last gate (brief §16C). A blocking failure downgrades the
        # plan rather than returning content that violates a hard rule.
        guard = check_plan(plan)
        if guard.should_block:
            from vm_telemetry import record_guardrail_failure

            record_guardrail_failure("output", len(guard.blocking_failures))
            logger.warning(
                "output_guardrail_blocked",
                extra={"failures": list(guard.blocking_failures), **_log(trace, trip_id)},
            )
            return plan.model_copy(
                update={
                    "status": PlanStatus.FAILED,
                    "validation_errors": [*plan.validation_errors, *guard.blocking_failures],
                    "trade_off_explanation": "",
                    "reasoning_summary": "",
                }
            )
        if guard.warnings:
            plan = plan.model_copy(update={"warnings": _dedupe([*plan.warnings, *guard.warnings])})

        # Cache the finished plan (no-op until Phase 10 wires Redis).
        if plan.is_usable and final_state.get("normalized") is not None:
            await self._deps.cache.put_plan(final_state["normalized"], plan)

        return plan

    # -- assembly --------------------------------------------------------
    def _assemble(self, state: OrchestratorState, trace: TraceContext, trip_id: str) -> TripPlan:
        status = state.get("final_status", PlanStatus.FAILED)
        transport = state.get("transport")
        accommodation = state.get("accommodation")
        itinerary = state.get("itinerary")
        budget = state.get("budget")

        available = {
            "transport": transport is not None and transport.has_result,
            "accommodation": accommodation is not None and accommodation.has_result,
            "itinerary": itinerary is not None and itinerary.has_result,
            "budget": budget is not None,
        }
        completeness = 100.0 * sum(available.values()) / len(available)

        return TripPlan(
            request_id=trace.request_id,
            trip_id=trip_id,
            status=status,
            transport=transport,
            accommodation=accommodation,
            itinerary=itinerary,
            budget=budget,
            trade_off_explanation=state.get("trade_off_explanation", ""),
            reasoning_summary=state.get("reasoning_summary", ""),
            data_sources=state.get("data_sources", []),
            confidence=_confidence(status, state),
            completeness_percent=round(completeness, 1),
            degraded_services=state.get("degraded_services", []),
            warnings=state.get("warnings", []),
            unavailable_sections=[k for k, present in available.items() if not present],
            replans=state.get("iteration_count", 0),
            cache_status=state.get("cache_status", "miss"),
            validation_errors=state.get("validation_errors", []),
        )

    def _timeout_plan(self, trace: TraceContext, trip_id: str) -> TripPlan:
        return self._error_plan(
            trace,
            trip_id,
            f"planning exceeded the {self._settings.limits.max_request_duration_seconds:g}s "
            f"deadline",
            status=PlanStatus.PARTIAL,
        )

    def _error_plan(
        self,
        trace: TraceContext,
        trip_id: str,
        detail: str,
        *,
        status: PlanStatus = PlanStatus.FAILED,
    ) -> TripPlan:
        return TripPlan(
            request_id=trace.request_id,
            trip_id=trip_id,
            status=status,
            confidence=ConfidenceLevel.NONE,
            completeness_percent=0.0,
            validation_errors=[detail],
            warnings=[detail],
        )


def _confidence(status: PlanStatus, state: OrchestratorState) -> ConfidenceLevel:
    if status is PlanStatus.FAILED or status is PlanStatus.NO_VIABLE_PLAN:
        return ConfidenceLevel.NONE
    if status is PlanStatus.PARTIAL or state.get("degraded_services"):
        return ConfidenceLevel.LOW
    # All-mocked transport/stay caps confidence at medium — the data is plausible, not real.
    transport = state.get("transport")
    if transport is not None and transport.data_origin == "mocked":
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.HIGH


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))[:30]


def _log(trace: TraceContext, trip_id: str) -> dict[str, str]:
    return {"request_id": trace.request_id, "trace_id": trace.trace_id, "trip_id": trip_id}
