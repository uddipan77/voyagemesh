"""Exception handlers — the single place a failure becomes an HTTP response.

Every error the gateway returns is an :class:`~vm_contracts.errors.ErrorResponse` (RFC 9457
problem details), carrying a stable ``code``, a safe ``detail``, and the ``trace_id`` the
client can quote. There is no path by which an unhandled exception's message or traceback
reaches a client: :func:`unhandled_exception_handler` logs the real detail (correlated by
trace ID) and returns a fixed generic 500 (threat T-9).
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from vm_contracts.errors import ErrorCode, ErrorResponse, ValidationProblem
from vm_contracts.tracing import TraceContext

__all__ = ["GatewayError", "install_error_handlers"]

logger = logging.getLogger(__name__)

# Field names that must never be echoed back in a validation problem, even truncated.
_SENSITIVE_FIELDS = frozenset({"authorization", "token", "password", "secret", "api_key"})


class GatewayError(Exception):
    """A deliberate, already-classified failure a handler can raise.

    Carrying an :class:`ErrorCode` means the HTTP status and shape are decided by the contract,
    not re-invented at each raise site.
    """

    def __init__(
        self,
        code: ErrorCode,
        detail: str,
        *,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.retry_after_seconds = retry_after_seconds


def _trace(request: Request) -> TraceContext:
    existing = getattr(request.state, "trace", None)
    return existing if isinstance(existing, TraceContext) else TraceContext.new()


def _response(error: ErrorResponse) -> JSONResponse:
    headers = {}
    if error.retry_after_seconds is not None:
        headers["Retry-After"] = str(error.retry_after_seconds)
    return JSONResponse(
        status_code=error.status,
        content=jsonable_encoder(error, exclude_none=True),
        headers=headers or None,
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(GatewayError)
    async def _gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
        trace = _trace(request)
        return _response(
            ErrorResponse.of(
                exc.code,
                exc.detail,
                request_id=trace.request_id,
                trace_id=trace.trace_id,
                retry_after_seconds=exc.retry_after_seconds,
            )
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        trace = _trace(request)
        problems: list[ValidationProblem] = []
        for err in exc.errors()[:50]:
            field = ".".join(str(p) for p in err.get("loc", ()) if p != "body") or "body"
            leaf = field.rsplit(".", 1)[-1].lower()
            problems.append(
                ValidationProblem(
                    field=field[:120],
                    message=str(err.get("msg", "invalid"))[:300],
                    # Never echo the offending value for a field that could hold a credential.
                    provided=None if leaf in _SENSITIVE_FIELDS else _short(err.get("input")),
                )
            )
        return _response(
            ErrorResponse.of(
                ErrorCode.VALIDATION_FAILED,
                "the request did not match the expected schema",
                request_id=trace.request_id,
                trace_id=trace.trace_id,
                problems=problems,
            )
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        trace = _trace(request)
        code = {
            401: ErrorCode.UNAUTHENTICATED,
            403: ErrorCode.INSUFFICIENT_ROLE,
            404: ErrorCode.TRIP_NOT_FOUND,
            413: ErrorCode.REQUEST_TOO_LARGE,
            429: ErrorCode.RATE_LIMITED,
        }.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
        detail = exc.detail if isinstance(exc.detail, str) else code.value.replace("_", " ")
        return _response(
            ErrorResponse.of(code, detail, request_id=trace.request_id, trace_id=trace.trace_id)
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        trace = _trace(request)
        # The real detail goes to the log, correlated by trace ID — never to the client.
        logger.exception("gateway_unhandled_error", extra=trace.log_fields())
        return _response(
            ErrorResponse.internal(request_id=trace.request_id, trace_id=trace.trace_id)
        )


def _short(value: object) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else repr(value)
    return text[:200]
