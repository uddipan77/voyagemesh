"""Structured JSON logging with trace correlation and secret redaction (brief §21).

One formatter turns every log record into a searchable JSON object carrying the request's
correlation context and its OpenTelemetry trace/span IDs, and no secret ever reaches the sink —
sensitive fields and credential-shaped text are scrubbed by construction, not by remembering to.
"""

from vm_logging.context import (
    bind_log_context,
    clear_log_context,
    get_log_context,
    log_context,
)
from vm_logging.json_formatter import JsonFormatter
from vm_logging.redaction import REDACTED, is_sensitive_key, redact_mapping, redact_text
from vm_logging.setup import configure_logging

__all__ = [
    "REDACTED",
    "JsonFormatter",
    "bind_log_context",
    "clear_log_context",
    "configure_logging",
    "get_log_context",
    "is_sensitive_key",
    "log_context",
    "redact_mapping",
    "redact_text",
]
