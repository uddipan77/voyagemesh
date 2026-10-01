"""The LangGraph planning workflow.

The graph is a pure description of control flow; all behaviour lives in the nodes. The shape
follows the brief's diagram (§10): validate → normalise → cache check → parallel fan-out to
transport and stay → join → itinerary → budget → constraint check → (replan | finalize).

Conditional edges are deterministic functions of the state — cache hit, constraints
violated, replans remaining — never model decisions (brief §10, "deterministic conditional
edges").
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from vm_contracts.budget import BudgetStatus
from vm_contracts.plan import PlanStatus
from vm_orchestrator.dependencies import OrchestratorDependencies
from vm_orchestrator.nodes import make_nodes
from vm_orchestrator.state import OrchestratorState

__all__ = ["build_graph"]


def build_graph(deps: OrchestratorDependencies) -> Any:
    """Compile the planning graph bound to ``deps``."""
    nodes = make_nodes(deps)
    graph = StateGraph(OrchestratorState)

    for name, fn in nodes.items():
        # `fn` is an async node closure; LangGraph's node type is broad and not worth
        # threading through the closure factory, so cast at this single boundary.
        graph.add_node(name, cast("Any", fn))

    graph.add_edge(START, "validate_input")

    # After input validation: stop on rejection, otherwise normalise.
    graph.add_conditional_edges(
        "validate_input",
        _after_validate,
        {"normalize_trip_request": "normalize_trip_request", "end": END},
    )

    graph.add_edge("normalize_trip_request", "check_cache")

    # After the cache check: a fresh hit ends the graph; a miss fans out to BOTH agents at
    # once. Returning a list is what makes them run in parallel; gating both behind this one
    # conditional is what stops the stay agent running on a cache hit.
    graph.add_conditional_edges(
        "check_cache",
        _after_cache,
        ["request_transport_agent", "request_stay_agent", END],
    )

    # Both agents feed the itinerary node, which LangGraph runs once after both parents
    # complete in the super-step — a natural join on a shared successor.
    graph.add_edge("request_transport_agent", "request_itinerary_agent")
    graph.add_edge("request_stay_agent", "request_itinerary_agent")
    graph.add_edge("request_itinerary_agent", "calculate_total_budget")

    # After budget: satisfied → finalize; over budget with replans left → replan; over budget
    # with none left → finalize (honest over-budget partial).
    graph.add_conditional_edges(
        "calculate_total_budget",
        _after_budget,
        {"replan": "replan", "finalize": "finalize"},
    )

    # Replan re-fans-out to BOTH agents together (transport re-runs unchanged; stay re-runs
    # under the tighter ceiling). Triggering both keeps the itinerary join unambiguous — its
    # two parents are always active together, never one alone.
    graph.add_edge("replan", "request_transport_agent")
    graph.add_edge("replan", "request_stay_agent")
    graph.add_edge("finalize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Conditional edges — deterministic functions of state (brief §10)
# ---------------------------------------------------------------------------
def _after_validate(state: OrchestratorState) -> str:
    if state.get("final_status") is PlanStatus.FAILED:
        return "end"
    return "normalize_trip_request"


def _after_cache(state: OrchestratorState) -> list[str] | str:
    if state.get("cache_status") == "hit":
        return END
    # Fan out to both agents in parallel.
    return ["request_transport_agent", "request_stay_agent"]


def _after_budget(state: OrchestratorState) -> str:
    """Replan only when the plan is over budget AND replans remain."""
    budget = state.get("budget")
    if budget is None or budget.is_within_budget or budget.status is BudgetStatus.INCOMPLETE:
        return "finalize"
    iteration = state.get("iteration_count", 0)
    max_replans = state.get("max_replans", 3)
    if iteration >= max_replans:
        return "finalize"  # exhausted; report an honest over-budget partial
    return "replan"
