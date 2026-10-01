"""The API gateway application.

``create_app`` assembles the gateway: middleware (trace context, CORS, body-size guard),
versioned routes, health probes, and the contract-shaped error handlers. It builds its
:class:`GatewayServices` lazily in the lifespan so importing this module never opens a socket —
which is what lets tests import ``create_app`` and inject their own services.

Run standalone::

    uv run uvicorn vm_api_gateway.app:app --port 8000
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from vm_api_gateway.errors import install_error_handlers
from vm_api_gateway.middleware import TraceContextMiddleware
from vm_api_gateway.routes import health_router, v1_router
from vm_api_gateway.services import GatewayServices
from vm_config.settings import Settings, get_settings
from vm_contracts.errors import ErrorCode, ErrorResponse
from vm_logging import configure_logging
from vm_telemetry import configure_telemetry, instrument_app

__all__ = ["create_app"]

logger = logging.getLogger(__name__)

_DESCRIPTION = (
    "Observable, secure, budget-aware multi-agent travel planner. Submit a trip request and "
    "receive a deterministic, honestly-labelled plan assembled by specialist agents over A2A."
)


def create_app(
    *, settings: Settings | None = None, services: GatewayServices | None = None
) -> FastAPI:
    """Build the gateway. Pass ``services`` to inject a pre-wired composition root (tests);
    otherwise it is built from ``settings`` at startup."""
    settings = settings or get_settings()
    configure_logging(settings)
    configure_telemetry(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # An injected composition root is attached eagerly below (so a test client that does
        # not run lifespan events still has it); only the default path builds here.
        if services is None:
            app.state.services = GatewayServices.build_default(settings)
        if settings.auth_bypass_active:
            logger.warning(
                "gateway_auth_bypass_active",
                extra={"detail": "Requests are NOT authenticated. Development only."},
            )
        try:
            yield
        finally:
            # Only tear down services we created; an injected one is the caller's to close.
            if services is None:
                await app.state.services.aclose()

    app = FastAPI(
        title="VoyageMesh API",
        version=settings.service_version,
        description=_DESCRIPTION,
        lifespan=lifespan,
    )
    if services is not None:
        app.state.services = services

    app.add_middleware(_BodySizeLimitMiddleware, max_bytes=settings.limits.max_request_bytes)
    app.add_middleware(TraceContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "traceparent"],
    )

    install_error_handlers(app)
    app.include_router(health_router)
    app.include_router(v1_router)

    # Auto-instrument last, once the app and its routes exist.
    instrument_app(app, settings=settings)
    return app


class _BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject an over-large body from its ``Content-Length`` before it is read into memory —
    a cheap first line against a resource-exhaustion payload (brief §12, threat T-7)."""

    def __init__(self, app: object, *, max_bytes: int) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._max = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > self._max:
            error = ErrorResponse.of(
                ErrorCode.REQUEST_TOO_LARGE,
                f"request body exceeds the {self._max} byte limit",
            )
            return JSONResponse(status_code=error.status, content=error.model_dump(mode="json"))
        return await call_next(request)


# The ASGI app uvicorn imports. Building it here (not at module import of create_app's body)
# keeps `import vm_api_gateway.app` side-effect free until this line runs.
app = create_app()
