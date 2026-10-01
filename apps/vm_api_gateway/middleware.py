"""Cross-cutting request middleware.

`TraceContextMiddleware` establishes one correlation context per request — continuing an
inbound ``traceparent`` / ``x-request-id`` if the caller supplied one, minting a fresh one
otherwise — stashes it on ``request.state.trace``, binds it into the structured-logging context
so every log line emitted while handling the request carries the same IDs, and records the API
Prometheus metrics. It echoes the ID back on the response so a client can quote it.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from vm_contracts.tracing import HEADER_REQUEST_ID, HEADER_TRACEPARENT, TraceContext
from vm_logging import bind_log_context, clear_log_context
from vm_telemetry import record_api_request

__all__ = ["TraceContextMiddleware"]


class TraceContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        trace = TraceContext.from_traceparent(
            request.headers.get(HEADER_TRACEPARENT),
            request_id=request.headers.get(HEADER_REQUEST_ID),
        )
        request.state.trace = trace
        bind_log_context(request_id=trace.request_id, trace_id=trace.trace_id)

        started = time.monotonic()
        try:
            response = await call_next(request)
        finally:
            duration = time.monotonic() - started
        # Use the matched route template (never the concrete id) to keep label cardinality low.
        route = request.scope.get("route")
        path = getattr(route, "path", None) or "unmatched"
        record_api_request(request.method, path, response.status_code, duration)

        response.headers[HEADER_REQUEST_ID] = trace.request_id
        response.headers[HEADER_TRACEPARENT] = trace.to_traceparent()
        clear_log_context()
        return response
