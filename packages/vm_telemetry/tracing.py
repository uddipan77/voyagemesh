"""OpenTelemetry tracing setup and helpers.

`configure_tracing` builds one `TracerProvider` per process — a service-named resource, an OTLP
HTTP exporter to the collector, and a ratio sampler — and is idempotent. Auto-instrumentation
for FastAPI, httpx, Redis, and SQLAlchemy is applied by `instrument_app`, which is how incoming
HTTP, A2A calls, MCP calls, external HTTP, Redis operations, and DB operations all get spans
without hand-written instrumentation. `traced` is the context manager for the manual spans that
matter — planning, replanning, guardrails, LLM calls.

Everything here fails safe: if the SDK or an instrumentor is unavailable, tracing degrades to a
no-op rather than breaking the service. Observability is never a hard dependency.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

from vm_config.settings import Settings

__all__ = ["configure_tracing", "get_tracer", "instrument_app", "traced"]

logger = logging.getLogger(__name__)

_configured = False


def configure_tracing(settings: Settings) -> None:
    """Install the global tracer provider. Idempotent and fail-safe."""
    global _configured
    if _configured or not settings.telemetry.enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        resource = Resource.create(
            {
                "service.name": settings.service_name,
                "service.version": settings.service_version,
                "deployment.environment": settings.environment.value,
            }
        )
        provider = TracerProvider(
            resource=resource,
            sampler=TraceIdRatioBased(settings.telemetry.traces_sampler_ratio),
        )
        exporter = OTLPSpanExporter(
            endpoint=f"{settings.telemetry.exporter_otlp_endpoint.rstrip('/')}/v1/traces"
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        if settings.telemetry.console_export:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        _configured = True
        logger.info(
            "tracing_configured", extra={"endpoint": settings.telemetry.exporter_otlp_endpoint}
        )
    except Exception:
        logger.warning("tracing_setup_failed_running_without_traces", exc_info=False)


def get_tracer(name: str = "voyagemesh") -> Any:
    from opentelemetry import trace

    return trace.get_tracer(name)


def instrument_app(app: Any, *, settings: Settings) -> None:
    """Apply auto-instrumentation to a FastAPI app and the shared clients it uses."""
    if not settings.telemetry.enabled:
        return
    _try(lambda: _instrument_fastapi(app))
    _try(_instrument_httpx)
    _try(_instrument_redis)
    _try(_instrument_sqlalchemy)


@contextlib.contextmanager
def traced(name: str, **attributes: Any) -> Iterator[Any]:
    """Open a manual span named ``name`` with ``attributes``. A no-op if tracing is off."""
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("voyagemesh")
        with tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
            yield span
    except Exception:
        yield None


def _instrument_fastapi(app: Any) -> None:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)


def _instrument_httpx() -> None:
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    HTTPXClientInstrumentor().instrument()


def _instrument_redis() -> None:
    from opentelemetry.instrumentation.redis import RedisInstrumentor

    RedisInstrumentor().instrument()


def _instrument_sqlalchemy() -> None:
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    SQLAlchemyInstrumentor().instrument()


def _try(fn: Any) -> None:
    try:
        fn()
    except Exception:
        logger.debug("instrumentation_skipped", exc_info=False)
