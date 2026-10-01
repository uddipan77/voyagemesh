"""Versioned HTTP routes — the public surface of VoyageMesh.

Everything a browser or API client touches lives under ``/api/v1`` so a later ``/api/v2`` can
change shapes without breaking existing clients. The two health probes sit at the root, outside
the version prefix, because orchestration (Compose, Kubernetes) expects them at a fixed path.

The trip-submission handler is where the request-lifecycle concerns compose in order:
authentication → rate limit → size guard → idempotency → plan → persist. Each is a small,
independently-tested step; none is skippable.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Header, Request, Response, status
from fastapi.responses import PlainTextResponse

from vm_api_gateway.errors import GatewayError
from vm_api_gateway.schemas import LivenessResponse, ReadinessResponse
from vm_api_gateway.services import GatewayServices
from vm_auth.user_auth import AuthenticatedUser, Role, TokenError
from vm_contracts.errors import ErrorCode
from vm_contracts.plan import TripPlan
from vm_contracts.tracing import TraceContext
from vm_contracts.trip import NormalizedTripRequest, TripRequest
from vm_logging import bind_log_context
from vm_telemetry import CONTENT_TYPE_LATEST, metrics_payload, record_trip_planning, traced

__all__ = ["health_router", "v1_router"]

health_router = APIRouter(tags=["health"])
v1_router = APIRouter(prefix="/api/v1", tags=["trips"])


def _services(request: Request) -> GatewayServices:
    return request.app.state.services  # type: ignore[no-any-return]


def _trace(request: Request) -> TraceContext:
    trace = getattr(request.state, "trace", None)
    return trace if isinstance(trace, TraceContext) else TraceContext.new()


async def _require_traveller(request: Request) -> AuthenticatedUser:
    """Authenticate and require the ``traveller`` role — planning is not anonymous.

    ``require_user`` already maps a bad/absent token to a 401; the role check maps a missing
    role to a contract-shaped 403 rather than letting a raw ``TokenError`` escape.
    """
    services = _services(request)
    user = await services.auth.require_user(request)
    try:
        user.require_role(Role.TRAVELLER)
    except TokenError as exc:
        raise GatewayError(ErrorCode.INSUFFICIENT_ROLE, exc.message) from None
    return user


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@health_router.get("/health/live", response_model=LivenessResponse)
async def live(request: Request) -> LivenessResponse:
    settings = _services(request).settings
    return LivenessResponse(service=settings.service_name, version=settings.service_version)


@health_router.get("/health/ready", response_model=ReadinessResponse)
async def ready(request: Request) -> ReadinessResponse:
    components = await _services(request).readiness()
    return ReadinessResponse(components=components)


@health_router.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus scrape endpoint. Unauthenticated by design — it is reachable only inside the
    internal network (Compose/Kubernetes), and it exposes counters, never secrets or payloads."""
    return PlainTextResponse(content=metrics_payload(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Trips
# ---------------------------------------------------------------------------
@v1_router.post("/trips", response_model=TripPlan, status_code=status.HTTP_200_OK)
async def submit_trip(
    request: Request,
    trip: TripRequest,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> TripPlan:
    """Plan a trip end to end.

    Synchronous by design: the orchestrator runs the whole LangGraph workflow under the global
    deadline and returns a complete, guardrail-checked plan. The plan's own ``status`` conveys
    the outcome — a degraded or no-viable-plan result is a 200 carrying an honest plan, not an
    HTTP error, because the request itself succeeded.
    """
    services = _services(request)
    trace = _trace(request)
    user = await _require_traveller(request)

    # 1. Rate limit, per authenticated user (fail-open if Redis is down).
    decision = await services.rate_limiter.check(user.reference)
    if not decision.allowed:
        raise GatewayError(
            ErrorCode.RATE_LIMITED,
            "too many requests; retry after the indicated delay",
            retry_after_seconds=max(1, round(decision.retry_after_seconds)),
        )

    # 2. Idempotency: a repeated key returns the first response, never a second plan.
    if idempotency_key:
        record = await services.idempotency.reserve(_idem_scope(user, idempotency_key))
        if record.conflict:
            raise GatewayError(
                ErrorCode.IDEMPOTENCY_CONFLICT,
                "a request with this Idempotency-Key is still in flight",
            )
        if record.stored_result is not None:
            response.headers["Idempotency-Replayed"] = "true"
            return TripPlan.model_validate_json(record.stored_result)

    normalized: NormalizedTripRequest | None = None
    try:
        normalized = NormalizedTripRequest.from_request(trip)
    except Exception:
        normalized = None  # normalisation is best-effort context for persistence only

    # 3. Plan. The orchestrator never raises — every failure is an honest plan.
    started = time.monotonic()
    with traced("trip.plan", **{"vm.user_ref": user.reference}):
        plan = await services.orchestrator.plan_trip(
            trip,
            request_id=trace.request_id,
            trace=trace,
            user_reference=user.reference,
            user_roles=sorted(user.roles),
        )
    duration_s = time.monotonic() - started
    duration_ms = round(duration_s * 1000)
    bind_log_context(trip_id=plan.trip_id)
    record_trip_planning(
        status=plan.status.value,
        duration_s=duration_s,
        replans=plan.replans,
        within_budget=plan.within_budget,
        degraded=bool(plan.degraded_services) or plan.status.value == "partial",
    )

    # 4. Persist (best-effort) and finalise idempotency.
    await services.persist_plan(plan, request=trip, normalized=normalized, duration_ms=duration_ms)
    if idempotency_key:
        await services.idempotency.complete(
            _idem_scope(user, idempotency_key), plan.model_dump_json()
        )

    response.headers["X-Trip-Id"] = plan.trip_id
    response.headers["X-Cache"] = plan.cache_status
    return plan


@v1_router.get("/trips/{trip_id}", response_model=TripPlan)
async def get_trip(request: Request, trip_id: str) -> TripPlan:
    """Retrieve a previously planned trip. Requires authentication; the opaque ``trip_id`` is
    the capability — a caller who does not hold it cannot enumerate trips."""
    services = _services(request)
    await services.auth.require_user(request)  # any authenticated user
    plan = await services.get_stored_plan(trip_id)
    if plan is None:
        raise GatewayError(ErrorCode.TRIP_NOT_FOUND, f"no trip with id {trip_id!r}")
    return plan


def _idem_scope(user: AuthenticatedUser, key: str) -> str:
    """Namespace the idempotency key by user so two users' identical keys never collide."""
    return f"{user.reference}:{key}"[:200]
