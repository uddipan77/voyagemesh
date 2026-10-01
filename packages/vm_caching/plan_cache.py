"""Redis-backed trip-plan cache.

Implements the orchestrator's ``PlanCache`` protocol (Phase 8) with real Redis. On a hit the
stored plan is re-validated against the current ``TripPlan`` schema before being served — a
payload that no longer validates (a schema change, or a poisoned entry) is treated as a miss
and evicted, so a bad deploy or a cache-poisoning attempt cannot serve broken results (threat
T-8).

Every operation fails open through :class:`RedisClient`, so Redis being down degrades to a
cache-free system, never a failed request.
"""

from __future__ import annotations

import logging

from vm_caching.client import RedisClient
from vm_caching.keys import plan_cache_key
from vm_config.settings import CacheTTLSettings
from vm_contracts.common import DataOrigin
from vm_contracts.plan import PlanStatus, TripPlan
from vm_contracts.trip import NormalizedTripRequest
from vm_telemetry import record_cache

__all__ = ["RedisPlanCache"]

logger = logging.getLogger(__name__)


class RedisPlanCache:
    """A trip-plan cache backed by Redis. Fail-open."""

    def __init__(self, client: RedisClient, ttl: CacheTTLSettings) -> None:
        self._client = client
        self._ttl = ttl

    async def get_plan(self, normalized: NormalizedTripRequest) -> TripPlan | None:
        """Return a fresh cached plan, or ``None`` on a miss. Never raises."""
        key = self._client.key(plan_cache_key(normalized))

        raw = await self._client.call(lambda r: r.get(key), fallback=None, label="plan_cache_get")
        if raw is None:
            record_cache(hit=False, kind="plan")
            return None

        try:
            plan = TripPlan.model_validate_json(raw)
        except Exception:
            # A stored plan that no longer validates is poison or drift — evict and miss.
            logger.warning("plan_cache_evict_invalid", extra={"key": key})
            await self._client.call(lambda r: r.delete(key), fallback=0, label="plan_cache_evict")
            record_cache(hit=False, kind="plan")
            return None

        # Re-label the served plan as cache-sourced so the UI badge is honest.
        record_cache(hit=True, kind="plan")
        return plan.model_copy(update={"cache_status": "hit"})

    async def put_plan(self, normalized: NormalizedTripRequest, plan: TripPlan) -> None:
        """Store a plan under the shortest applicable TTL. Never raises.

        Only usable plans are cached — a failed or no-viable-plan result is not worth
        serving again, and caching it could mask a since-resolved transient failure.
        """
        if plan.status not in (PlanStatus.COMPLETE, PlanStatus.PARTIAL):
            return

        key = self._client.key(plan_cache_key(normalized))
        ttl = self._ttl_for(plan)
        payload = plan.model_dump_json()

        await self._client.call(
            lambda r: r.set(key, payload, ex=ttl), fallback=None, label="plan_cache_put"
        )

    async def invalidate(self, normalized: NormalizedTripRequest) -> None:
        key = self._client.key(plan_cache_key(normalized))
        await self._client.call(lambda r: r.delete(key), fallback=0, label="plan_cache_invalidate")

    def _ttl_for(self, plan: TripPlan) -> int:
        """The plan inherits the shortest TTL of its inputs (brief §17).

        A plan is only as fresh as its most volatile component. Transport and accommodation
        (15 min) are the shortest; if the plan contains live weather that would be shorter
        still, but weather is not the binding constraint here.
        """
        ttls = [
            self._ttl.plan_seconds,
            self._ttl.transport_seconds,
            self._ttl.accommodation_seconds,
        ]
        # A fully-mocked plan is deterministic, so it may be cached for the plan TTL; a plan
        # with any live component takes the volatile floor.
        has_live = any(
            source.origin in (DataOrigin.LIVE, DataOrigin.CACHED) for source in plan.data_sources
        )
        if has_live:
            ttls.append(self._ttl.weather_seconds)
        return max(1, min(ttls))
