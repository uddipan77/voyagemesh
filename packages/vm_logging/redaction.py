"""Log redaction — the last line of defence against a secret reaching a log sink.

Brief §21 mandates that API keys, JWTs, authorization headers, passwords, cookies, provider
secrets, and sensitive user fields never appear in logs. This module scrubs two things:

* **structured fields** — any log ``extra`` whose key looks sensitive is replaced wholesale;
* **free text** — a message or string value that contains a bearer token, an ``Authorization``
  header, or a long base64/JWT-shaped blob has that span replaced.

The philosophy matches the rest of the codebase: when in doubt, redact. A false positive costs
a little log readability; a false negative leaks a credential.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["REDACTED", "is_sensitive_key", "redact_mapping", "redact_text"]

REDACTED = "***REDACTED***"

# Field names (case-insensitive, substring match) whose values are replaced entirely.
_SENSITIVE_KEY_MARKERS = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "api-key",
    "authorization",
    "auth_header",
    "cookie",
    "session",
    "credential",
    "private_key",
    "client_secret",
)

# Field names that are safe despite matching a marker above (avoid over-redaction).
_ALLOWLIST = frozenset(
    {"token_count", "input_tokens", "output_tokens", "total_tokens", "tokens", "session_count"}
)

# Text patterns that indicate a credential regardless of the field name.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)\bauthorization\b\s*[:=]\s*\S+"),
    # A JWT: three base64url segments separated by dots.
    re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
    re.compile(r"(?i)\b(api[_-]?key|secret|password)\b\s*[:=]\s*\S+"),
    # Groq-style keys.
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}"),
)


def is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in _ALLOWLIST:
        return False
    return any(marker in lowered for marker in _SENSITIVE_KEY_MARKERS)


def redact_text(text: str) -> str:
    """Replace any credential-shaped span in ``text``."""
    redacted = text
    for pattern in _PATTERNS:
        redacted = pattern.sub(REDACTED, redacted)
    return redacted


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return redact_mapping(value)
    if isinstance(value, list | tuple):
        return type(value)(redact_value(v) for v in value)
    return value


def redact_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``data`` with sensitive keys replaced and string values scrubbed."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(key, str) and is_sensitive_key(key):
            out[key] = REDACTED
        else:
            out[key] = redact_value(value)
    return out
