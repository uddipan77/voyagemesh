"""Typed, sanitised error responses (RFC 9457 problem details).

Every error the API returns is constructed here. There is no path by which an unhandled
exception's message or traceback reaches a client: the gateway's exception handler
converts anything unrecognised into :meth:`ErrorResponse.internal`, which carries a fixed
message and the trace ID. The real detail goes to the structured log, correlated by that
same trace ID.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field

from vm_contracts.common import StrictModel

__all__ = ["ErrorCode", "ErrorResponse", "ValidationProblem"]


class ErrorCode(StrEnum):
    """Stable, machine-readable error identifiers.

    Clients branch on these; the human-readable ``detail`` may be reworded freely without
    breaking anything.
    """

    # 400 family
    VALIDATION_FAILED = "validation_failed"
    INVALID_DATE_RANGE = "invalid_date_range"
    BUDGET_INVALID = "budget_invalid"
    UNSUPPORTED_CURRENCY = "unsupported_currency"
    UNSUPPORTED_DESTINATION = "unsupported_destination"
    REQUEST_TOO_LARGE = "request_too_large"
    PROMPT_INJECTION_DETECTED = "prompt_injection_detected"

    # 401 / 403
    UNAUTHENTICATED = "unauthenticated"
    # These name a *failure mode*, not a credential — see this class's docstring.
    TOKEN_EXPIRED = "token_expired"  # nosec B105
    TOKEN_INVALID = "token_invalid"  # nosec B105
    INSUFFICIENT_ROLE = "insufficient_role"

    # 404 / 409
    TRIP_NOT_FOUND = "trip_not_found"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"

    # 429
    RATE_LIMITED = "rate_limited"

    # 5xx
    INTERNAL_ERROR = "internal_error"
    AGENT_UNAVAILABLE = "agent_unavailable"
    LLM_UNAVAILABLE = "llm_unavailable"
    UPSTREAM_TIMEOUT = "upstream_timeout"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    NO_VIABLE_PLAN = "no_viable_plan"
    """No option satisfied the constraints. Reported honestly rather than by relaxing the
    user's budget or fabricating a cheaper offer."""


_STATUS_BY_CODE: dict[ErrorCode, int] = {
    ErrorCode.VALIDATION_FAILED: 422,
    ErrorCode.INVALID_DATE_RANGE: 422,
    ErrorCode.BUDGET_INVALID: 422,
    ErrorCode.UNSUPPORTED_CURRENCY: 422,
    ErrorCode.UNSUPPORTED_DESTINATION: 422,
    ErrorCode.REQUEST_TOO_LARGE: 413,
    ErrorCode.PROMPT_INJECTION_DETECTED: 400,
    ErrorCode.UNAUTHENTICATED: 401,
    ErrorCode.TOKEN_EXPIRED: 401,
    ErrorCode.TOKEN_INVALID: 401,
    ErrorCode.INSUFFICIENT_ROLE: 403,
    ErrorCode.TRIP_NOT_FOUND: 404,
    ErrorCode.IDEMPOTENCY_CONFLICT: 409,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.AGENT_UNAVAILABLE: 503,
    ErrorCode.LLM_UNAVAILABLE: 503,
    ErrorCode.UPSTREAM_TIMEOUT: 504,
    ErrorCode.GUARDRAIL_BLOCKED: 422,
    ErrorCode.NO_VIABLE_PLAN: 422,
}


class ValidationProblem(StrictModel):
    """A single field-level validation failure."""

    field: Annotated[str, Field(min_length=1, max_length=120)]
    message: Annotated[str, Field(min_length=1, max_length=300)]
    provided: Annotated[str | None, Field(max_length=200)] = None
    """Echo of the offending value, truncated. Never populated for a field that could
    hold a credential."""


class ErrorResponse(StrictModel):
    """RFC 9457-style problem details."""

    type: Annotated[str, Field(min_length=1, max_length=200)] = "about:blank"
    title: Annotated[str, Field(min_length=1, max_length=200)]
    status: Annotated[int, Field(ge=100, le=599)]
    code: ErrorCode
    detail: Annotated[str, Field(min_length=1, max_length=1000)]
    """Safe for display. Never contains a stack trace, internal hostname, SQL fragment,
    or provider error body."""

    request_id: Annotated[str | None, Field(max_length=60)] = None
    trace_id: Annotated[str | None, Field(max_length=60)] = None
    """Given to the user so a support request can be correlated with the server-side log
    that holds the real detail."""

    problems: Annotated[list[ValidationProblem], Field(max_length=50)] = Field(default_factory=list)
    retry_after_seconds: Annotated[int | None, Field(ge=0)] = None

    @classmethod
    def of(
        cls,
        code: ErrorCode,
        detail: str,
        *,
        title: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
        problems: list[ValidationProblem] | None = None,
        retry_after_seconds: int | None = None,
    ) -> ErrorResponse:
        return cls(
            title=title or code.value.replace("_", " ").title(),
            status=_STATUS_BY_CODE[code],
            code=code,
            detail=detail,
            request_id=request_id,
            trace_id=trace_id,
            problems=problems or [],
            retry_after_seconds=retry_after_seconds,
        )

    @classmethod
    def internal(
        cls, *, request_id: str | None = None, trace_id: str | None = None
    ) -> ErrorResponse:
        """The only response permitted for an unexpected exception.

        The message is a fixed string by design — a generic error that reveals nothing is
        strictly better than a helpful one that leaks a connection string.
        """
        return cls.of(
            ErrorCode.INTERNAL_ERROR,
            "An unexpected error occurred. Quote the trace ID when reporting this.",
            request_id=request_id,
            trace_id=trace_id,
        )

    @staticmethod
    def status_for(code: ErrorCode) -> int:
        return _STATUS_BY_CODE[code]
