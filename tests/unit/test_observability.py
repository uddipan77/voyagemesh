"""Structured logging, redaction, and Prometheus metrics."""

from __future__ import annotations

import io
import json
import logging

import pytest

from vm_logging import JsonFormatter, bind_log_context, clear_log_context, log_context
from vm_telemetry import metrics, metrics_payload, observe_llm, record_api_request, traced

pytestmark = pytest.mark.unit


def _emit(record_fn, *, service="vm") -> dict:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter(service=service))
    logger = logging.getLogger(f"test.{id(buf)}")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    record_fn(logger)
    return json.loads(buf.getvalue().strip())


class TestStructuredLogging:
    def test_message_and_level_are_json(self):
        obj = _emit(lambda lg: lg.info("hello world"))
        assert obj["level"] == "INFO"
        assert obj["message"] == "hello world"
        assert obj["service"] == "vm"

    def test_bound_context_appears_on_every_line(self):
        def emit(lg):
            with log_context(request_id="req-9", trip_id="trip-3"):
                lg.info("planning")

        obj = _emit(emit)
        assert obj["request_id"] == "req-9"
        assert obj["trip_id"] == "trip-3"

    def test_context_is_restored_after_the_block(self):
        clear_log_context()
        with log_context(request_id="a"):
            bind_log_context(trip_id="b")
        obj = _emit(lambda lg: lg.info("after"))
        assert "request_id" not in obj and "trip_id" not in obj

    def test_extra_fields_are_included(self):
        obj = _emit(lambda lg: lg.info("counted", extra={"count": 7, "agent": "transport"}))
        assert obj["count"] == 7
        assert obj["agent"] == "transport"


class TestRedaction:
    def test_sensitive_extra_keys_are_redacted(self):
        obj = _emit(
            lambda lg: lg.info(
                "auth",
                extra={"api_key": "gsk_abc", "authorization": "Bearer x", "password": "hunter2"},
            )
        )
        assert obj["api_key"] == "***REDACTED***"
        assert obj["authorization"] == "***REDACTED***"
        assert obj["password"] == "***REDACTED***"

    def test_credential_shaped_message_is_scrubbed(self):
        obj = _emit(lambda lg: lg.info("token is Bearer abc.def.ghi please"))
        assert "abc.def.ghi" not in obj["message"]
        assert "***REDACTED***" in obj["message"]

    def test_token_count_is_not_over_redacted(self):
        # A key that merely contains "token" but is safe must survive.
        obj = _emit(lambda lg: lg.info("usage", extra={"input_tokens": 42}))
        assert obj["input_tokens"] == 42


class TestMetrics:
    def test_api_metric_appears_in_payload(self):
        record_api_request("POST", "/api/v1/trips", 200, 0.1)
        payload = metrics_payload().decode()
        assert "api_requests_total" in payload
        assert 'path="/api/v1/trips"' in payload

    def test_llm_tokens_are_counted(self):
        before = _counter_value(metrics.llm_input_tokens_total, provider="groq", model="m")
        observe_llm(provider="groq", model="m", duration_s=0.2, input_tokens=10, output_tokens=4)
        after = _counter_value(metrics.llm_input_tokens_total, provider="groq", model="m")
        assert after - before == 10

    def test_cache_hit_and_miss_are_distinct_series(self):
        from vm_telemetry import record_cache

        record_cache(hit=True, kind="plan")
        record_cache(hit=False, kind="plan")
        payload = metrics_payload().decode()
        assert "cache_hits_total" in payload
        assert "cache_misses_total" in payload


class TestTracing:
    async def test_traced_is_a_noop_without_a_provider(self):
        # With no configured provider, traced must not raise and must yield.
        entered = False
        with traced("unit.span", **{"vm.k": "v"}):
            entered = True
        assert entered


def _counter_value(counter, **labels) -> float:
    return counter.labels(**labels)._value.get()
