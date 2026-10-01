"""Generate the eight VoyageMesh Grafana dashboards (brief §21).

Hand-writing Grafana JSON is error-prone and repetitive, so the dashboards are generated from a
compact spec here. Run this to (re)emit the JSON files that Grafana provisions:

    python infra/grafana/generate_dashboards.py

Each dashboard maps to one of the brief's eight required views and is built from the metrics
defined in ``vm_telemetry.metrics``. The output is deterministic — no timestamps, stable panel
ids — so re-running produces a clean diff only when the spec changes.
"""

from __future__ import annotations

import json
from pathlib import Path

_DS = {"type": "prometheus", "uid": "prometheus"}
_OUT = Path(__file__).resolve().parent / "dashboards"


def _target(expr: str, legend: str, ref: str = "A") -> dict:
    return {"expr": expr, "legendFormat": legend, "refId": ref, "datasource": _DS}


def _panel(
    pid: int,
    title: str,
    targets: list[dict],
    *,
    x: int,
    y: int,
    unit: str = "short",
    ptype: str = "timeseries",
    w: int = 12,
    h: int = 8,
) -> dict:
    return {
        "id": pid,
        "title": title,
        "type": ptype,
        "datasource": _DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}}
        if ptype == "timeseries"
        else {"reduceOptions": {"calcs": ["lastNotNull"]}},
        "targets": targets,
    }


def _dashboard(uid: str, title: str, panels: list[dict]) -> dict:
    return {
        "uid": uid,
        "title": title,
        "tags": ["voyagemesh"],
        "schemaVersion": 39,
        "version": 1,
        "editable": True,
        "refresh": "30s",
        "time": {"from": "now-6h", "to": "now"},
        "timepicker": {},
        "templating": {"list": []},
        "annotations": {"list": []},
        "panels": panels,
    }


def _grid(specs: list[tuple]) -> list[dict]:
    """Lay panels out two-per-row from (title, targets, unit, ptype) tuples."""
    panels = []
    for i, (title, targets, unit, ptype) in enumerate(specs):
        x = 0 if i % 2 == 0 else 12
        y = (i // 2) * 8
        panels.append(_panel(i + 1, title, targets, x=x, y=y, unit=unit, ptype=ptype))
    return panels


DASHBOARDS = {
    "vm-system-health": (
        "1 · System Health",
        [
            ("Service up", [_target("up", "{{job}}")], "short", "timeseries"),
            (
                "API p95 latency (s)",
                [
                    _target(
                        "histogram_quantile(0.95, sum(rate(api_request_duration_seconds_bucket[5m])) by (le))",
                        "p95",
                    )
                ],
                "s",
                "timeseries",
            ),
            (
                "Degraded responses (rate)",
                [_target("rate(degraded_responses_total[5m])", "degraded")],
                "short",
                "timeseries",
            ),
            (
                "Guardrail failures (rate)",
                [_target("sum(rate(guardrail_failures_total[5m])) by (stage)", "{{stage}}")],
                "short",
                "timeseries",
            ),
        ],
    ),
    "vm-api-orchestration": (
        "2 · API & Orchestration",
        [
            (
                "Requests by status (rate)",
                [_target("sum(rate(api_requests_total[5m])) by (status)", "{{status}}")],
                "reqps",
                "timeseries",
            ),
            (
                "API latency p50/p95 (s)",
                [
                    _target(
                        "histogram_quantile(0.5, sum(rate(api_request_duration_seconds_bucket[5m])) by (le))",
                        "p50",
                        "A",
                    ),
                    _target(
                        "histogram_quantile(0.95, sum(rate(api_request_duration_seconds_bucket[5m])) by (le))",
                        "p95",
                        "B",
                    ),
                ],
                "s",
                "timeseries",
            ),
            (
                "Planning outcomes (rate)",
                [_target("sum(rate(trip_planning_requests_total[5m])) by (status)", "{{status}}")],
                "short",
                "timeseries",
            ),
            (
                "Planning p95 (s)",
                [
                    _target(
                        "histogram_quantile(0.95, sum(rate(trip_planning_duration_seconds_bucket[5m])) by (le))",
                        "p95",
                    )
                ],
                "s",
                "timeseries",
            ),
        ],
    ),
    "vm-agent-a2a": (
        "3 · Agent & A2A Performance",
        [
            (
                "Agent invocations (rate)",
                [_target("sum(rate(agent_requests_total[5m])) by (agent)", "{{agent}}")],
                "short",
                "timeseries",
            ),
            (
                "Agent failures (rate)",
                [_target("sum(rate(agent_failures_total[5m])) by (agent)", "{{agent}}")],
                "short",
                "timeseries",
            ),
            (
                "A2A round-trip p95 (s)",
                [
                    _target(
                        "histogram_quantile(0.95, sum(rate(a2a_duration_seconds_bucket[5m])) by (le, agent))",
                        "{{agent}}",
                    )
                ],
                "s",
                "timeseries",
            ),
            (
                "A2A failures (rate)",
                [_target("sum(rate(a2a_failures_total[5m])) by (agent)", "{{agent}}")],
                "short",
                "timeseries",
            ),
        ],
    ),
    "vm-mcp-provider": (
        "4 · MCP & Provider Performance",
        [
            (
                "MCP tool calls (rate)",
                [
                    _target(
                        "sum(rate(mcp_tool_calls_total[5m])) by (server, tool)",
                        "{{server}}/{{tool}}",
                    )
                ],
                "short",
                "timeseries",
            ),
            (
                "MCP tool failures (rate)",
                [_target("sum(rate(mcp_tool_failures_total[5m])) by (server)", "{{server}}")],
                "short",
                "timeseries",
            ),
            (
                "MCP tool p95 (s)",
                [
                    _target(
                        "histogram_quantile(0.95, sum(rate(mcp_tool_duration_seconds_bucket[5m])) by (le, tool))",
                        "{{tool}}",
                    )
                ],
                "s",
                "timeseries",
            ),
            (
                "External API failures (rate)",
                [
                    _target(
                        "sum(rate(external_api_failures_total[5m])) by (provider)", "{{provider}}"
                    )
                ],
                "short",
                "timeseries",
            ),
        ],
    ),
    "vm-llm-usage": (
        "5 · LLM Usage",
        [
            (
                "LLM requests (rate)",
                [_target("sum(rate(llm_requests_total[5m])) by (model)", "{{model}}")],
                "short",
                "timeseries",
            ),
            (
                "LLM p95 latency (s)",
                [
                    _target(
                        "histogram_quantile(0.95, sum(rate(llm_duration_seconds_bucket[5m])) by (le, model))",
                        "{{model}}",
                    )
                ],
                "s",
                "timeseries",
            ),
            (
                "Token throughput (rate)",
                [
                    _target("sum(rate(llm_input_tokens_total[5m]))", "input", "A"),
                    _target("sum(rate(llm_output_tokens_total[5m]))", "output", "B"),
                ],
                "short",
                "timeseries",
            ),
            (
                "Structured-output failures (rate)",
                [_target("rate(structured_output_failures_total[5m])", "failures")],
                "short",
                "timeseries",
            ),
        ],
    ),
    "vm-cache-database": (
        "6 · Cache & Database",
        [
            (
                "Cache hits vs misses (rate)",
                [
                    _target("sum(rate(cache_hits_total[5m]))", "hits", "A"),
                    _target("sum(rate(cache_misses_total[5m]))", "misses", "B"),
                ],
                "short",
                "timeseries",
            ),
            (
                "Cache hit ratio",
                [
                    _target(
                        "sum(rate(cache_hits_total[5m])) / clamp_min(sum(rate(cache_hits_total[5m])) + sum(rate(cache_misses_total[5m])), 1)",
                        "hit ratio",
                    )
                ],
                "percentunit",
                "timeseries",
            ),
            (
                "DB query p95 (s) — via OTel/SQLAlchemy",
                [
                    _target(
                        "histogram_quantile(0.95, sum(rate(db_client_operation_duration_seconds_bucket[5m])) by (le))",
                        "p95",
                    )
                ],
                "s",
                "timeseries",
            ),
            (
                "Cache total (last)",
                [_target("sum(cache_hits_total) + sum(cache_misses_total)", "lookups")],
                "short",
                "stat",
            ),
        ],
    ),
    "vm-product-quality": (
        "7 · Product Quality",
        [
            (
                "Plans within budget (rate)",
                [_target("rate(plans_within_budget_total[5m])", "within budget")],
                "short",
                "timeseries",
            ),
            (
                "Constraint violations (rate)",
                [_target("rate(plan_constraint_violation_total[5m])", "no viable plan")],
                "short",
                "timeseries",
            ),
            (
                "Replanning iterations (rate)",
                [_target("rate(plan_replanning_total[5m])", "replans")],
                "short",
                "timeseries",
            ),
            (
                "Degraded responses (total)",
                [_target("degraded_responses_total", "degraded")],
                "short",
                "stat",
            ),
        ],
    ),
    "vm-security-guardrails": (
        "8 · Security & Guardrail Events",
        [
            (
                "Guardrail failures by stage (rate)",
                [_target("sum(rate(guardrail_failures_total[5m])) by (stage)", "{{stage}}")],
                "short",
                "timeseries",
            ),
            (
                "Auth failures 401/403 (rate)",
                [_target('sum(rate(api_requests_total{status=~"401|403"}[5m]))', "auth failures")],
                "short",
                "timeseries",
            ),
            (
                "Rate-limited 429 (rate)",
                [_target('sum(rate(api_requests_total{status="429"}[5m]))', "rate limited")],
                "short",
                "timeseries",
            ),
            (
                "Structured-output failures (total)",
                [_target("structured_output_failures_total", "failures")],
                "short",
                "stat",
            ),
        ],
    ),
}


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    for uid, (title, specs) in DASHBOARDS.items():
        dashboard = _dashboard(uid, title, _grid(specs))
        path = _OUT / f"{uid}.json"
        path.write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(_OUT.parent)}")


if __name__ == "__main__":
    main()
