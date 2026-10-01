"""The gateway's composition root.

`GatewayServices` owns every long-lived dependency the request handlers need — the
orchestrator, Redis (cache / rate-limiter / idempotency), the database, and JWT auth — and is
built once at startup and closed once at shutdown. Handlers receive it through
``request.app.state.services``; nothing in a route constructs its own client.

Two ways to build it:

* :meth:`GatewayServices.build_default` — production wiring from :class:`Settings`: HTTP agent
  clients over A2A, a real Redis-backed cache, a real Postgres pool.
* the constructor directly — tests inject in-process agent clients, a fake Redis, and no
  database, and drive the exact same handlers.

Every dependency here is *fail-open* (brief §17): Redis or Postgres being down degrades the
service (no cache, no persistence, no dedup) but never fails a plan request.
"""

from __future__ import annotations

import logging

from vm_auth import JwtValidator
from vm_auth.fastapi_deps import AuthDependencies
from vm_caching import IdempotencyStore, RedisClient, RedisPlanCache, TokenBucketRateLimiter
from vm_config.settings import ProviderMode, Settings
from vm_contracts.plan import TripPlan
from vm_contracts.trip import NormalizedTripRequest, TripRequest
from vm_database import Database, TripRepository
from vm_harness import HttpAgentClient, build_service_token_provider
from vm_orchestrator.dependencies import NoOpPlanCache, OrchestratorDependencies
from vm_orchestrator.orchestrator import TravelOrchestrator

__all__ = ["GatewayServices"]

logger = logging.getLogger(__name__)


class GatewayServices:
    """Long-lived dependencies for the API gateway."""

    def __init__(
        self,
        *,
        settings: Settings,
        orchestrator: TravelOrchestrator,
        redis: RedisClient,
        auth: AuthDependencies,
        database: Database | None = None,
    ) -> None:
        self.settings = settings
        self.orchestrator = orchestrator
        self.redis = redis
        self.auth = auth
        self.database = database
        self.rate_limiter = TokenBucketRateLimiter(redis, settings.limits)
        self.idempotency = IdempotencyStore(
            redis, ttl_seconds=settings.cache_ttl.idempotency_seconds
        )

    @classmethod
    def build_default(cls, settings: Settings) -> GatewayServices:
        """Wire the production gateway from configuration."""
        redis = RedisClient(settings.redis)
        token_provider = build_service_token_provider(settings)

        def _client(url: str) -> HttpAgentClient:
            return HttpAgentClient(
                url,
                token_provider=token_provider,
                timeout_seconds=settings.a2a.timeout_seconds,
                max_retries=settings.a2a.max_retries,
            )

        deps = OrchestratorDependencies(
            settings=settings,
            transport_client=_client(settings.a2a.transport_agent_url),
            stay_client=_client(settings.a2a.stay_agent_url),
            itinerary_client=_client(settings.a2a.itinerary_agent_url),
            # Fresh supplier quotes on every live search. This also prevents a switch
            # from mock to live from returning an earlier mock plan from Redis.
            cache=(
                RedisPlanCache(redis, settings.cache_ttl)
                if settings.providers.mode is ProviderMode.MOCK
                else NoOpPlanCache()
            ),
        )
        orchestrator = TravelOrchestrator(deps)

        database = Database(settings.database) if settings.database.enabled else None
        validator = JwtValidator(settings.auth)
        auth = AuthDependencies(settings, validator=validator)
        return cls(
            settings=settings,
            orchestrator=orchestrator,
            redis=redis,
            auth=auth,
            database=database,
        )

    # -- persistence (best-effort; a DB outage never fails a request) ----------
    async def persist_plan(
        self,
        plan: TripPlan,
        *,
        request: TripRequest,
        normalized: NormalizedTripRequest | None,
        duration_ms: int,
    ) -> None:
        """Save a finished plan. Swallows any storage error — persistence is not on the
        critical path of returning a plan to the user (brief §17)."""
        if self.database is None:
            return
        try:
            async with self.database.session() as session:
                await TripRepository(session).save_plan(
                    plan, request=request, normalized=normalized, duration_ms=duration_ms
                )
        except Exception:
            logger.warning("plan_persist_failed", extra={"trip_id": plan.trip_id}, exc_info=False)

    async def get_stored_plan(self, trip_id: str) -> TripPlan | None:
        """Fetch a previously planned trip, or ``None`` if unknown or the DB is unavailable."""
        if self.database is None:
            return None
        try:
            async with self.database.session() as session:
                return await TripRepository(session).get_plan(trip_id)
        except Exception:
            logger.warning("plan_fetch_failed", extra={"trip_id": trip_id}, exc_info=False)
            return None

    async def readiness(self) -> dict[str, str]:
        """Report each dependency's state. All are fail-open, so a down dependency reads
        ``degraded`` rather than making the gateway unready."""
        components = {"redis": "disabled", "database": "disabled"}
        if self.settings.redis.enabled:
            components["redis"] = "up" if await self.redis.ping() else "degraded"
        if self.database is not None:
            components["database"] = "up" if await self.database.health_check() else "degraded"
        return components

    async def aclose(self) -> None:
        if self.database is not None:
            await self.database.aclose()
        await self.redis.aclose()
