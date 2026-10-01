"""Secrets must never reach a log sink (brief §21, threat T-9).

These tests assert the redaction layer scrubs every credential class the brief names, including
the *actual* Groq API key value read from the key file — the strongest possible check that the
one real secret in this repository cannot be logged.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from vm_logging import JsonFormatter, redact_mapping, redact_text

pytestmark = [pytest.mark.security, pytest.mark.unit]

_KEY_FILE = Path(__file__).resolve().parents[2] / "apikeys" / "api.key"


def _log_line(**extra) -> str:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(f"redaction.{id(buf)}")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.info("event", extra=extra)
    return buf.getvalue()


class TestTextRedaction:
    @pytest.mark.parametrize(
        "text",
        [
            "Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.payloadpart.signaturepart",
            "api_key=gsk_" + ("0" * 30),  # Synthetic key shape; never a usable credential.
            "password: hunter2secret",
            "here is a bearer abcdefghijklmnop token",
        ],
    )
    def test_credential_shaped_text_is_scrubbed(self, text):
        out = redact_text(text)
        assert "***REDACTED***" in out

    def test_a_jwt_is_removed(self):
        jwt = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImsxIn0.eyJzdWIiOiJhIn0.abcDEF123_-"
        assert jwt not in redact_text(f"token={jwt}")


class TestMappingRedaction:
    def test_sensitive_keys_are_replaced(self):
        out = redact_mapping({"authorization": "Bearer x", "client_secret": "s", "city": "Prague"})
        assert out["authorization"] == "***REDACTED***"
        assert out["client_secret"] == "***REDACTED***"
        assert out["city"] == "Prague"  # non-sensitive survives

    def test_nested_values_are_scrubbed(self):
        out = redact_mapping({"headers": {"Authorization": "Bearer secret"}})
        assert "secret" not in json.dumps(out)


class TestTheRealGroqKeyIsNeverLogged:
    def test_the_key_value_does_not_survive_redaction(self):
        if not _KEY_FILE.exists():
            pytest.skip("no api.key present in this environment")
        key = _KEY_FILE.read_text(encoding="utf-8").strip()
        if not key:
            pytest.skip("api.key is empty")
        # Whether it arrives as a field value, inside a message, or as an Authorization header,
        # the real key must never appear in the emitted JSON.
        assert key not in _log_line(api_key=key)
        assert key not in redact_text(f"calling groq with key {key}")
        assert key not in redact_text(f"Authorization: Bearer {key}")
