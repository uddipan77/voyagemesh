"""Observability: OpenTelemetry tracing and Prometheus metrics (brief §21).

Tracing and metrics are both fail-safe optionals — a missing collector or SDK degrades to a
no-op, never a broken service. `configure_telemetry` is the single startup call; `metrics` holds
every named metric plus recording helpers; `traced` opens manual spans.
"""

from vm_config.settings import Settings
from vm_telemetry import metrics
from vm_telemetry.metrics import (
    CONTENT_TYPE_LATEST,
    metrics_payload,
    observe_llm,
    record_api_request,
    record_cache,
    record_guardrail_failure,
    record_trip_planning,
)
from vm_telemetry.tracing import configure_tracing, get_tracer, instrument_app, traced

__all__ = [
    "CONTENT_TYPE_LATEST",
    "configure_telemetry",
    "configure_tracing",
    "get_tracer",
    "instrument_app",
    "metrics",
    "metrics_payload",
    "observe_llm",
    "record_api_request",
    "record_cache",
    "record_guardrail_failure",
    "record_trip_planning",
    "traced",
]


def configure_telemetry(settings: Settings) -> None:
    """Set up tracing for the process. Metrics are registered at import and need no setup."""
    configure_tracing(settings)
