"""The orchestrator's graph nodes.

Each node is an ``async`` function taking the state and returning a *partial* dict that
LangGraph merges in. Nodes are deliberately thin: the real work lives in the domain layer
(budget arithmetic), the agents (search and ranking), and the guardrails. A node's job is to
call those, record what happened, and route.

The two rules that shape every node:

* **The orchestrator never calls a travel provider.** It speaks only to agents over A2A
  (brief §6A). Transport, stay, and itinerary data arrive as agent artifacts.
* **The budget is computed by code, here.** ``calculate_total_budget`` runs
  ``vm_domain.calculate_budget``; no agent and no LLM decides whether the plan fits (ADR-009).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vm_contracts.a2a import AgentArtifact, AgentTask, TaskStatus
from vm_contracts.agent_results import ItineraryResult, StayResult, TransportResult
from vm_contracts.budget import BudgetStatus
from vm_contracts.common import DataOrigin, Provenance, utc_now
from vm_contracts.plan import DataSourceInfo, PlanStatus
from vm_contracts.tracing import TraceContext
from vm_domain.budget import calculate_budget, remaining_accommodation_allowance
from vm_guardrails import check_trip_request
from vm_orchestrator.state import OrchestratorState

if TYPE_CHECKING:
    from vm_orchestrator.dependencies import OrchestratorDependencies

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node factory: nodes are closures over the orchestrator's dependencies (agent
# clients, cache, config), so the graph stays a pure description of control flow.
# ---------------------------------------------------------------------------
def make_nodes(deps: OrchestratorDependencies) -> dict[str, object]:
    """Build the node functions bound to ``deps``."""

    async def validate_input(state: OrchestratorState) -> dict[str, object]:
        """Input guardrails: injection screening and configurable limits (brief §16A).

        The request is already schema-valid (the gateway validated it); this adds the checks
        a schema cannot express. A failure ends the graph with validation errors rather than
        planning a request that should have been refused.
        """
        request = state["trip_request"]
        result = check_trip_request(
            request,
            detect_injection=deps.settings.guardrails.enable_prompt_injection_detection,
            max_interests=deps.settings.limits.max_interests,
        )
        if not result.passed:
            logger.warning(
                "input_guardrail_rejected",
                extra={"injection": result.rejected_for_injection, **_log(state)},
            )
            return {
                "final_status": PlanStatus.FAILED,
                "validation_errors": list(result.errors),
            }
        return {"iteration_count": 0, "cache_status": "miss"}

    async def normalize_trip_request(state: OrchestratorState) -> dict[str, object]:
        """Canonicalise the request once, so every downstream node shares one interpretation."""
        from vm_contracts.trip import NormalizedTripRequest

        normalized = NormalizedTripRequest.from_request(state["trip_request"])
        return {"normalized": normalized}

    async def check_cache(state: OrchestratorState) -> dict[str, object]:
        """Look up a cached plan by the normalised fingerprint (brief §17).

        The cache is an injected interface; the real Redis implementation lands in Phase 10.
        A hit short-circuits the graph via the conditional edge; a miss continues to planning.
        """
        cached = await deps.cache.get_plan(state["normalized"])
        if cached is not None:
            logger.info("cache_hit", extra=_log(state))
            return {"cache_status": "hit", "cached_plan": cached}
        return {"cache_status": "miss"}

    async def request_transport_agent(state: OrchestratorState) -> dict[str, object]:
        """Delegate transport planning over A2A (parallel with stay)."""
        normalized = state["normalized"]
        payload = {
            "origin": normalized.origin,
            "destination": normalized.destination,
            "departure_date": normalized.departure_date.isoformat(),
            "return_date": normalized.return_date.isoformat(),
            "travellers": normalized.travellers,
            "currency": normalized.currency.value,
            "preferred_mode": normalized.transport_preference.value,
            "max_duration_hours": normalized.max_transport_duration_hours,
            "max_transfers": normalized.max_transfers,
            "accessibility_needs": [n.value for n in normalized.accessibility_needs],
            "ranking_strategy": normalized.ranking_strategy.value,
        }
        artifact, task_id = await _delegate(
            deps, state, skill="plan_transport", payload=payload, agent="transport"
        )
        if artifact is None or not artifact.is_usable or artifact.result is None:
            return {
                "transport": None,
                "degraded_services": ["transport"],
                "warnings": [
                    "Transport planning was unavailable.",
                    *(artifact.warnings if artifact else []),
                ],
                "agent_task_ids": [task_id],
            }
        result = TransportResult.model_validate(artifact.result)
        return {
            "transport": result,
            "agent_task_ids": [task_id],
            "data_sources": [
                _source(
                    "transport",
                    result.data_origin,
                    result.recommended.provenance if result.recommended else None,
                )
            ],
            **_degradation_from(artifact, "transport"),
        }

    async def request_stay_agent(state: OrchestratorState) -> dict[str, object]:
        """Delegate accommodation planning over A2A (parallel with transport)."""
        normalized = state["normalized"]
        if normalized.nights == 0:
            return {"accommodation": None, "warnings": ["Day trip — no accommodation needed."]}

        ceiling = state.get("accommodation_ceiling")
        payload = {
            "destination": normalized.destination,
            "check_in": normalized.departure_date.isoformat(),
            "check_out": normalized.return_date.isoformat(),
            "guests": normalized.travellers,
            "guest_nationality": normalized.guest_nationality,
            "currency": normalized.currency.value,
            "accommodation_type": normalized.accommodation_preference.value,
            "accessibility_needs": [n.value for n in normalized.accessibility_needs],
            "ranking_strategy": normalized.ranking_strategy.value,
        }
        if ceiling is not None:
            payload["max_total_price"] = ceiling

        artifact, task_id = await _delegate(
            deps, state, skill="plan_accommodation", payload=payload, agent="stay"
        )
        if artifact is None or not artifact.is_usable or artifact.result is None:
            return {
                "accommodation": None,
                "degraded_services": ["accommodation"],
                "warnings": [
                    "Accommodation planning was unavailable.",
                    *(artifact.warnings if artifact else []),
                ],
                "agent_task_ids": [task_id],
            }
        result = StayResult.model_validate(artifact.result)
        return {
            "accommodation": result,
            "agent_task_ids": [task_id],
            "data_sources": [
                _source(
                    "accommodation",
                    result.data_origin,
                    result.recommended.provenance if result.recommended else None,
                )
            ],
            **_degradation_from(artifact, "accommodation"),
        }

    async def request_itinerary_agent(state: OrchestratorState) -> dict[str, object]:
        """Delegate itinerary planning, anchored to the chosen accommodation's location.

        Runs after the parallel pair because it needs the recommended stay's coordinates to
        compute realistic walking times (brief §6D).
        """
        normalized = state["normalized"]
        accommodation = state.get("accommodation")
        location = None
        if accommodation and accommodation.recommended and accommodation.recommended.location:
            loc = accommodation.recommended.location
            location = {"latitude": loc.latitude, "longitude": loc.longitude}

        payload: dict[str, object] = {
            "origin": normalized.origin,
            "destination": normalized.destination,
            "departure_date": normalized.departure_date.isoformat(),
            "return_date": normalized.return_date.isoformat(),
            "travellers": normalized.travellers,
            "interests": normalized.interests,
            "accessibility_needs": [n.value for n in normalized.accessibility_needs],
            "accommodation_type": normalized.accommodation_preference.value,
        }
        if location is not None:
            payload["accommodation_location"] = location

        artifact, task_id = await _delegate(
            deps, state, skill="plan_itinerary", payload=payload, agent="itinerary"
        )
        if artifact is None or not artifact.is_usable or artifact.result is None:
            return {
                "itinerary": None,
                "degraded_services": ["itinerary"],
                "warnings": ["Itinerary planning was unavailable."],
                "agent_task_ids": [task_id],
            }
        result = ItineraryResult.model_validate(artifact.result)
        sources = [
            _source(
                "itinerary",
                result.poi_origin,
                result.considered_attractions[0].provenance
                if result.considered_attractions
                else None,
            )
        ]
        if result.weather is not None:
            sources.append(_source("weather", result.weather_origin, result.weather.provenance))
        return {
            "itinerary": result,
            "agent_task_ids": [task_id],
            "data_sources": sources,
            **_degradation_from(artifact, "itinerary"),
        }

    async def calculate_total_budget(state: OrchestratorState) -> dict[str, object]:
        """Compute the trip budget by code — the one place cost is decided (ADR-009)."""
        normalized = state["normalized"]
        transport = state.get("transport")
        accommodation = state.get("accommodation")
        itinerary = state.get("itinerary")

        summary = calculate_budget(
            normalized,
            transport=transport.recommended if transport else None,
            accommodation=accommodation.recommended if accommodation else None,
            itinerary=itinerary.itinerary if itinerary else None,
        )
        return {
            "budget": summary,
            "data_sources": [_source("budget", "computed")],
        }

    async def replan(state: OrchestratorState) -> dict[str, object]:
        """Tighten constraints and loop back to planning (brief §10, §12).

        The strategy is honest: it does not relax the user's budget. It computes how much is
        left for accommodation once transport and estimates are covered, and re-runs the stay
        search under that ceiling — the cheapest lever that keeps the user's hard limit
        intact. Bounded by ``max_replans``.
        """
        normalized = state["normalized"]
        transport = state.get("transport")
        ceiling = remaining_accommodation_allowance(
            normalized, transport=transport.recommended if transport else None
        )
        iteration = state.get("iteration_count", 0) + 1
        logger.info("replanning", extra={"iteration": iteration, **_log(state)})
        return {
            "iteration_count": iteration,
            "accommodation_ceiling": float(ceiling.amount),
            "warnings": [
                f"Replanned (pass {iteration}) with a tighter accommodation budget of "
                f"{ceiling} to stay within the total budget."
            ],
        }

    async def finalize(state: OrchestratorState) -> dict[str, object]:
        """Assemble the trade-off explanation and reasoning summary from settled facts.

        Narratives come from the agents' own reasoning summaries, stitched together — no new
        LLM call is made here, so finalisation cannot introduce an unvalidated claim.
        """
        parts: list[str] = []
        transport = state.get("transport")
        accommodation = state.get("accommodation")
        itinerary = state.get("itinerary")

        if transport and transport.reasoning:
            parts.append(f"Transport: {transport.reasoning.explanation}")
        if accommodation and accommodation.reasoning:
            parts.append(f"Accommodation: {accommodation.reasoning.explanation}")
        if itinerary and itinerary.reasoning:
            parts.append(f"Itinerary: {itinerary.reasoning.explanation}")

        budget = state.get("budget")
        trade_off = ""
        if budget is not None:
            trade_off = (
                f"Estimated total {budget.total} against a {budget.max_budget} budget "
                f"({budget.utilisation_percent:.0f}% used). "
            )
            if budget.status is BudgetStatus.INCOMPLETE:
                trade_off += "Some costs are unpriced; the total budget cannot be confirmed. "
            elif not budget.is_within_budget:
                trade_off += f"This exceeds the budget by {budget.overspend}. "

        status = _derive_status(state)
        return {
            "final_status": status,
            "reasoning_summary": " ".join(parts)[:2000],
            "trade_off_explanation": trade_off[:2000],
        }

    return {
        "validate_input": validate_input,
        "normalize_trip_request": normalize_trip_request,
        "check_cache": check_cache,
        "request_transport_agent": request_transport_agent,
        "request_stay_agent": request_stay_agent,
        "request_itinerary_agent": request_itinerary_agent,
        "calculate_total_budget": calculate_total_budget,
        "replan": replan,
        "finalize": finalize,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _delegate(
    deps: OrchestratorDependencies,
    state: OrchestratorState,
    *,
    skill: str,
    payload: dict[str, object],
    agent: str,
) -> tuple[AgentArtifact | None, str]:
    """Submit a task to an agent, returning ``(artifact_or_None, task_id)``.

    Never raises — an unreachable agent yields ``None`` so the caller degrades. The
    correlation ID binds every task to this trip request (threat T-4).
    """
    from vm_harness import A2AClientError

    trace = TraceContext(
        request_id=state["request_id"],
        trace_id=state["trace_id"],
        span_id=state["trace_id"][:16],
    )
    task = AgentTask(
        skill=skill,
        payload=payload,
        correlation_id=state["trip_id"],
        request_id=state["request_id"],
        trace_id=state["trace_id"],
        traceparent=trace.to_traceparent(),
        user_reference=state.get("user_reference"),
        user_roles=state.get("user_roles", []),
        deadline_seconds=deps.settings.a2a.timeout_seconds,
    )
    client = deps.agent_client(agent)
    try:
        artifact = await deps.submit(client, task, required_skill=skill)
    except A2AClientError as exc:
        logger.warning(
            "agent_delegation_failed",
            extra={"agent": agent, "error": type(exc).__name__, **_log(state)},
        )
        return None, task.task_id
    return artifact, task.task_id


def _degradation_from(artifact: AgentArtifact, service: str) -> dict[str, object]:
    """Extract degradation and warnings from a partial artifact."""
    out: dict[str, object] = {"warnings": list(artifact.warnings)}
    if artifact.status is TaskStatus.PARTIAL:
        out["degraded_services"] = [service]
        out["warnings"] = list(artifact.warnings)
    return out


def _source(component: str, origin: str, provenance: Provenance | None = None) -> DataSourceInfo:
    try:
        data_origin = DataOrigin(origin)
    except ValueError:
        data_origin = DataOrigin.UNAVAILABLE
    retrieved = utc_now() if data_origin in (DataOrigin.LIVE, DataOrigin.CACHED) else None
    return DataSourceInfo(
        component=component,
        source_name=provenance.source_name if provenance else component,
        origin=data_origin,
        retrieved_at=provenance.retrieved_at if provenance else retrieved,
        note=provenance.license_note if provenance else None,
    )


def _derive_status(state: OrchestratorState) -> PlanStatus:
    transport = state.get("transport")
    budget = state.get("budget")

    # No transport and (for an overnight trip) no accommodation means nothing usable.
    if transport is None or not transport.has_result:
        return PlanStatus.NO_VIABLE_PLAN

    if (
        budget is not None
        and not budget.is_within_budget
        and state.get("iteration_count", 0) >= (state.get("max_replans", 3))
    ):
        # Over budget after exhausting replans: honest, not hidden.
        return PlanStatus.PARTIAL

    if state.get("degraded_services"):
        return PlanStatus.PARTIAL
    if budget is not None and budget.status is BudgetStatus.INCOMPLETE:
        return PlanStatus.PARTIAL
    return PlanStatus.COMPLETE


def _log(state: OrchestratorState) -> dict[str, str]:
    return {
        "request_id": state.get("request_id", "?"),
        "trip_id": state.get("trip_id", "?"),
        "trace_id": state.get("trace_id", "?"),
    }
