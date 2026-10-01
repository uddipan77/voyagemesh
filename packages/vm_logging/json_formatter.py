"""A structured JSON log formatter with trace correlation and redaction built in.

Every record becomes one JSON object carrying: timestamp, level, logger, message, the bound
correlation context (request_id / trip_id / agent / …), the active OpenTelemetry trace and span
IDs (so a log line links to its span), and any ``extra`` fields — all passed through the
redactor first. There is no code path by which a raw ``extra`` reaches the sink unscrubbed.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any

from vm_logging.context import get_log_context
from vm_logging.redaction import REDACTED, is_sensitive_key, redact_text, redact_value

__all__ = ["JsonFormatter"]

# Standard LogRecord attributes we do not want to duplicate into the JSON as "extra".
_RESERVED = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


def _otel_ids() -> dict[str, str]:
    """Current trace/span IDs from OpenTelemetry, or empty if no active/valid span."""
    try:
        # `import x.y as y`, not `from x import y`: `opentelemetry` is a namespace package split
        # across distributions, so mypy resolves the parent but cannot see `trace` as an
        # attribute of it. Runtime behaviour is identical.
        import opentelemetry.trace as trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            return {
                "trace_id": format(ctx.trace_id, "032x"),
                "span_id": format(ctx.span_id, "016x"),
            }
    except Exception:
        return {}
    return {}


class JsonFormatter(logging.Formatter):
    def __init__(self, *, service: str = "voyagemesh") -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _dt.datetime.fromtimestamp(record.created, tz=_dt.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "service": self._service,
            "message": redact_text(record.getMessage()),
        }
        # Correlation context, then live OTel IDs (which win if both present).
        payload.update(get_log_context())
        payload.update(_otel_ids())

        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = REDACTED if is_sensitive_key(key) else redact_value(value)

        if record.exc_info:
            # The type and a redacted message only — never the full traceback in the JSON
            # field (it can carry connection strings). Stack goes to the console formatter.
            exc_type = record.exc_info[0]
            exc_val = record.exc_info[1]
            payload["error_type"] = exc_type.__name__ if exc_type else "Exception"
            payload["error"] = redact_text(str(exc_val)) if exc_val else ""

        return json.dumps(payload, default=str, ensure_ascii=False)
