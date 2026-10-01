"""Correlation identifiers carried across every service boundary.

:class:`TraceContext` is the envelope that makes a single request followable from the
browser through the gateway, orchestrator, agents, MCP servers, and out to providers. It
serialises to W3C ``traceparent`` for OpenTelemetry interop, plus VoyageMesh-specific
correlation headers that the structured logger attaches to every event.
"""

from __future__ import annotations

import re
import secrets
from typing import Annotated, Self

from pydantic import Field, model_validator

from vm_contracts.common import StrictModel

__all__ = [
    "HEADER_CORRELATION_ID",
    "HEADER_REQUEST_ID",
    "HEADER_TRACEPARENT",
    "HEADER_TRIP_ID",
    "TraceContext",
    "new_request_id",
]

HEADER_TRACEPARENT = "traceparent"
HEADER_REQUEST_ID = "x-request-id"
HEADER_CORRELATION_ID = "x-correlation-id"
HEADER_TRIP_ID = "x-trip-id"

_TRACEPARENT_RE = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")


def new_request_id() -> str:
    return f"req_{secrets.token_hex(10)}"


class TraceContext(StrictModel):
    """Correlation identity for one request."""

    request_id: Annotated[str, Field(min_length=1, max_length=60)] = Field(
        default_factory=new_request_id
    )
    trace_id: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    span_id: Annotated[str, Field(pattern=r"^[0-9a-f]{16}$")]
    trace_flags: Annotated[str, Field(pattern=r"^[0-9a-f]{2}$")] = "01"

    correlation_id: Annotated[str | None, Field(max_length=60)] = None
    trip_id: Annotated[str | None, Field(max_length=60)] = None
    user_reference: Annotated[str | None, Field(max_length=100)] = None
    """A privacy-safe user handle — a Keycloak subject ID, never an email address. This
    value reaches logs and traces, so it must not be personally identifying."""

    agent_name: Annotated[str | None, Field(max_length=80)] = None
    agent_task_id: Annotated[str | None, Field(max_length=60)] = None

    @model_validator(mode="after")
    def _reject_all_zero_ids(self) -> Self:
        # The W3C spec defines all-zero IDs as invalid; accepting one produces traces that
        # silently merge unrelated requests in the backend.
        if self.trace_id == "0" * 32:
            raise ValueError("trace_id must not be all zeros")
        if self.span_id == "0" * 16:
            raise ValueError("span_id must not be all zeros")
        return self

    @classmethod
    def new(cls, *, request_id: str | None = None, sampled: bool = True) -> TraceContext:
        """Start a fresh trace at the edge of the system."""
        return cls(
            request_id=request_id or new_request_id(),
            trace_id=secrets.token_hex(16),
            span_id=secrets.token_hex(8),
            trace_flags="01" if sampled else "00",
        )

    @classmethod
    def from_traceparent(
        cls, traceparent: str | None, *, request_id: str | None = None
    ) -> TraceContext:
        """Continue an inbound trace, or start a new one if the header is absent or invalid.

        Malformed headers start a new trace rather than raising: a broken upstream should
        degrade observability, never fail the user's request.
        """
        if traceparent:
            match = _TRACEPARENT_RE.match(traceparent.strip().lower())
            if match:
                trace_id, span_id, flags = match.groups()
                if trace_id != "0" * 32 and span_id != "0" * 16:
                    return cls(
                        request_id=request_id or new_request_id(),
                        trace_id=trace_id,
                        span_id=span_id,
                        trace_flags=flags,
                    )
        return cls.new(request_id=request_id)

    def to_traceparent(self) -> str:
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags}"

    def child(
        self, *, agent_name: str | None = None, agent_task_id: str | None = None
    ) -> TraceContext:
        """Derive a context for a downstream call: same trace, new span."""
        return self.model_copy(
            update={
                "span_id": secrets.token_hex(8),
                "agent_name": agent_name or self.agent_name,
                "agent_task_id": agent_task_id or self.agent_task_id,
            }
        )

    def to_headers(self) -> dict[str, str]:
        """Headers to attach to an outbound A2A or MCP request."""
        headers = {
            HEADER_TRACEPARENT: self.to_traceparent(),
            HEADER_REQUEST_ID: self.request_id,
        }
        if self.correlation_id:
            headers[HEADER_CORRELATION_ID] = self.correlation_id
        if self.trip_id:
            headers[HEADER_TRIP_ID] = self.trip_id
        return headers

    def log_fields(self) -> dict[str, str]:
        """Fields bound to every structured log line emitted while this context is active.

        Deliberately excludes anything sensitive: no tokens, no email, no raw user input.
        """
        fields = {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }
        for key, value in (
            ("correlation_id", self.correlation_id),
            ("trip_id", self.trip_id),
            ("user_ref", self.user_reference),
            ("agent", self.agent_name),
            ("agent_task_id", self.agent_task_id),
        ):
            if value:
                fields[key] = value
        return fields

    @property
    def is_sampled(self) -> bool:
        return bool(int(self.trace_flags, 16) & 0x01)
