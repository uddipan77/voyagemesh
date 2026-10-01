"""Prometheus metrics for VoyageMesh (brief §21).

Every metric named in the brief is defined here, once, on the default registry, with the
recording helpers that keep call sites free of Prometheus imports. Labels are kept low-
cardinality on purpose: paths use the route *template* (``/api/v1/trips/{trip_id}``), never the
concrete id, so a metric never blows up into a series per trip.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

__all__ = [
    "CONTENT_TYPE_LATEST",
    "metrics_payload",
    "observe_llm",
    "record_api_request",
    "record_cache",
    "record_guardrail_failure",
    "record_trip_planning",
]

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)

# --- API / gateway ---------------------------------------------------------
api_requests_total = Counter(
    "api_requests_total", "HTTP requests handled by the gateway.", ["method", "path", "status"]
)
api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "Gateway request latency.",
    ["method", "path"],
    buckets=_LATENCY_BUCKETS,
)

# --- Trip planning ---------------------------------------------------------
trip_planning_requests_total = Counter(
    "trip_planning_requests_total", "Trip-planning requests by final status.", ["status"]
)
trip_planning_duration_seconds = Histogram(
    "trip_planning_duration_seconds", "End-to-end planning latency.", buckets=_LATENCY_BUCKETS
)
plan_constraint_violation_total = Counter(
    "plan_constraint_violation_total", "Plans that violated a hard constraint."
)
plan_replanning_total = Counter("plan_replanning_total", "Replanning iterations performed.")
plans_within_budget_total = Counter(
    "plans_within_budget_total", "Usable plans that fit the stated budget."
)
degraded_responses_total = Counter(
    "degraded_responses_total", "Responses returned in a degraded/partial state."
)
guardrail_failures_total = Counter(
    "guardrail_failures_total", "Guardrail failures by stage.", ["stage"]
)

# --- Agents / A2A ----------------------------------------------------------
agent_requests_total = Counter("agent_requests_total", "Agent invocations.", ["agent"])
agent_failures_total = Counter("agent_failures_total", "Agent invocation failures.", ["agent"])
agent_duration_seconds = Histogram(
    "agent_duration_seconds", "Agent handling latency.", ["agent"], buckets=_LATENCY_BUCKETS
)
a2a_requests_total = Counter("a2a_requests_total", "Outbound A2A task submissions.", ["agent"])
a2a_failures_total = Counter("a2a_failures_total", "Outbound A2A failures.", ["agent"])
a2a_duration_seconds = Histogram(
    "a2a_duration_seconds", "A2A round-trip latency.", ["agent"], buckets=_LATENCY_BUCKETS
)

# --- MCP tools -------------------------------------------------------------
mcp_tool_calls_total = Counter("mcp_tool_calls_total", "MCP tool calls.", ["server", "tool"])
mcp_tool_failures_total = Counter(
    "mcp_tool_failures_total", "MCP tool call failures.", ["server", "tool"]
)
mcp_tool_duration_seconds = Histogram(
    "mcp_tool_duration_seconds", "MCP tool latency.", ["server", "tool"], buckets=_LATENCY_BUCKETS
)

# --- LLM -------------------------------------------------------------------
llm_requests_total = Counter("llm_requests_total", "LLM requests.", ["provider", "model"])
llm_failures_total = Counter("llm_failures_total", "LLM request failures.", ["provider", "model"])
llm_duration_seconds = Histogram(
    "llm_duration_seconds", "LLM request latency.", ["provider", "model"], buckets=_LATENCY_BUCKETS
)
llm_input_tokens_total = Counter(
    "llm_input_tokens_total", "LLM input tokens consumed.", ["provider", "model"]
)
llm_output_tokens_total = Counter(
    "llm_output_tokens_total", "LLM output tokens produced.", ["provider", "model"]
)
structured_output_failures_total = Counter(
    "structured_output_failures_total", "LLM structured-output validation failures."
)

# --- Cache / external ------------------------------------------------------
cache_hits_total = Counter("cache_hits_total", "Cache hits.", ["kind"])
cache_misses_total = Counter("cache_misses_total", "Cache misses.", ["kind"])
external_api_requests_total = Counter(
    "external_api_requests_total", "Outbound external-provider requests.", ["provider"]
)
external_api_failures_total = Counter(
    "external_api_failures_total", "Outbound external-provider failures.", ["provider"]
)


# --- Recording helpers -----------------------------------------------------
def record_api_request(method: str, path: str, status: int, duration_s: float) -> None:
    api_requests_total.labels(method=method, path=path, status=str(status)).inc()
    api_request_duration_seconds.labels(method=method, path=path).observe(duration_s)


def record_trip_planning(
    *, status: str, duration_s: float, replans: int, within_budget: bool | None, degraded: bool
) -> None:
    trip_planning_requests_total.labels(status=status).inc()
    trip_planning_duration_seconds.observe(duration_s)
    if replans:
        plan_replanning_total.inc(replans)
    if within_budget:
        plans_within_budget_total.inc()
    if degraded:
        degraded_responses_total.inc()
    if status == "no_viable_plan":
        plan_constraint_violation_total.inc()


def record_guardrail_failure(stage: str, count: int = 1) -> None:
    if count:
        guardrail_failures_total.labels(stage=stage).inc(count)


def record_cache(hit: bool, kind: str = "plan") -> None:
    (cache_hits_total if hit else cache_misses_total).labels(kind=kind).inc()


def observe_llm(
    *,
    provider: str,
    model: str,
    duration_s: float,
    input_tokens: int,
    output_tokens: int,
    failed: bool = False,
) -> None:
    llm_requests_total.labels(provider=provider, model=model).inc()
    llm_duration_seconds.labels(provider=provider, model=model).observe(duration_s)
    if failed:
        llm_failures_total.labels(provider=provider, model=model).inc()
        return
    if input_tokens:
        llm_input_tokens_total.labels(provider=provider, model=model).inc(input_tokens)
    if output_tokens:
        llm_output_tokens_total.labels(provider=provider, model=model).inc(output_tokens)


def metrics_payload() -> bytes:
    """The current metrics in Prometheus text exposition format, for the /metrics endpoint."""
    return generate_latest()
