"""Redis caching, rate limiting, idempotency, and locks — against fakeredis.

fakeredis (with lupa for Lua) covers the logic offline and deterministically. The fail-open
behaviour — every operation degrading gracefully when Redis is down — is the most important
property here and is tested with a client pointed at nothing.
"""

from __future__ import annotations

from datetime import date

import pytest
from fakeredis.aioredis import FakeRedis

from vm_caching import (
    RedisClient,
    RedisPlanCache,
    TokenBucketRateLimiter,
    plan_cache_key,
)
from vm_caching.idempotency import DistributedLock, IdempotencyStore
from vm_config.settings import (
    CacheTTLSettings,
    LimitSettings,
    RedisSettings,
)
from vm_contracts.common import Currency, Money
from vm_contracts.plan import PlanStatus, TripPlan
from vm_contracts.trip import NormalizedTripRequest, TripRequest

pytestmark = pytest.mark.unit


@pytest.fixture
async def client():
    c = RedisClient(RedisSettings(_env_file=None), redis=FakeRedis())
    try:
        yield c
    finally:
        await c.aclose()


def _normalized(**overrides) -> NormalizedTripRequest:
    defaults = {
        "origin": "Nuremberg",
        "destination": "Prague",
        "departure_date": date(2026, 8, 10),
        "return_date": date(2026, 8, 13),
        "max_budget": Money.of(350, Currency.EUR),
    }
    return NormalizedTripRequest.from_request(TripRequest(**{**defaults, **overrides}))


def _plan(status: PlanStatus = PlanStatus.COMPLETE) -> TripPlan:
    return TripPlan(request_id="r", trip_id="t", status=status)


class TestCacheKeys:
    def test_key_is_deterministic(self):
        assert plan_cache_key(_normalized()) == plan_cache_key(_normalized())

    def test_key_is_case_and_whitespace_insensitive(self):
        assert plan_cache_key(_normalized(origin="Nuremberg")) == plan_cache_key(
            _normalized(origin="  nuremberg ")
        )

    def test_semantically_different_requests_differ(self):
        assert plan_cache_key(_normalized()) != plan_cache_key(_normalized(destination="Vienna"))

    def test_key_is_namespaced_and_versioned(self):
        key = plan_cache_key(_normalized())
        assert key.startswith("trip:v1:")


class TestPlanCache:
    async def test_miss_then_hit(self, client):
        cache = RedisPlanCache(client, CacheTTLSettings(_env_file=None))
        normalized = _normalized()
        assert await cache.get_plan(normalized) is None  # miss
        await cache.put_plan(normalized, _plan())
        hit = await cache.get_plan(normalized)
        assert hit is not None
        assert hit.cache_status == "hit"  # re-labelled honestly

    async def test_failed_plans_are_not_cached(self, client):
        cache = RedisPlanCache(client, CacheTTLSettings(_env_file=None))
        normalized = _normalized()
        await cache.put_plan(normalized, _plan(PlanStatus.FAILED))
        assert await cache.get_plan(normalized) is None

    async def test_partial_plans_are_cached(self, client):
        cache = RedisPlanCache(client, CacheTTLSettings(_env_file=None))
        normalized = _normalized()
        await cache.put_plan(normalized, _plan(PlanStatus.PARTIAL))
        assert await cache.get_plan(normalized) is not None

    async def test_invalid_stored_payload_is_evicted(self, client):
        """A poisoned or schema-drifted entry must not be served (threat T-8)."""
        cache = RedisPlanCache(client, CacheTTLSettings(_env_file=None))
        normalized = _normalized()
        key = client.key(plan_cache_key(normalized))
        await client.raw.set(key, b'{"not": "a valid plan"}')
        assert await cache.get_plan(normalized) is None
        # And it was evicted, not left to fail again.
        assert await client.raw.get(key) is None

    async def test_invalidate_removes_the_entry(self, client):
        cache = RedisPlanCache(client, CacheTTLSettings(_env_file=None))
        normalized = _normalized()
        await cache.put_plan(normalized, _plan())
        await cache.invalidate(normalized)
        assert await cache.get_plan(normalized) is None


class TestRateLimiter:
    def _limiter(self, client, *, per_minute=60, burst=3, now_ms=1000):
        return TokenBucketRateLimiter(
            client,
            LimitSettings(
                _env_file=None,
                rate_limit_requests_per_minute=per_minute,
                rate_limit_burst=burst,
            ),
            now_ms=now_ms,
        )

    async def test_allows_up_to_the_burst_then_denies(self, client):
        limiter = self._limiter(client, burst=3)
        outcomes = [(await limiter.check("user")).allowed for _ in range(5)]
        assert outcomes == [True, True, True, False, False]

    async def test_reports_remaining_tokens(self, client):
        limiter = self._limiter(client, burst=3)
        assert (await limiter.check("user")).remaining == 2

    async def test_denied_request_reports_retry_after(self, client):
        limiter = self._limiter(client, per_minute=60, burst=1)
        await limiter.check("user")
        decision = await limiter.check("user")
        assert decision.allowed is False
        assert decision.retry_after_seconds > 0

    async def test_refills_over_time(self, client):
        limiter = self._limiter(client, per_minute=60, burst=1, now_ms=1000)  # 1 token/sec
        assert (await limiter.check("user")).allowed is True
        assert (await limiter.check("user")).allowed is False
        # Advance 2 seconds — the bucket refills.
        later = self._limiter(client, per_minute=60, burst=1, now_ms=3000)
        assert (await later.check("user")).allowed is True

    async def test_identities_are_independent(self, client):
        limiter = self._limiter(client, burst=1)
        assert (await limiter.check("alice")).allowed is True
        assert (await limiter.check("bob")).allowed is True  # bob has his own bucket


class TestIdempotency:
    async def test_first_reservation_is_new(self, client):
        store = IdempotencyStore(client)
        assert (await store.reserve("key-1")).is_new is True

    async def test_concurrent_reservation_conflicts(self, client):
        store = IdempotencyStore(client)
        await store.reserve("key-1")
        second = await store.reserve("key-1")
        assert second.is_new is False
        assert second.conflict is True

    async def test_completed_key_replays_the_result(self, client):
        store = IdempotencyStore(client)
        await store.reserve("key-1")
        await store.complete("key-1", "STORED-RESULT")
        replay = await store.reserve("key-1")
        assert replay.stored_result == "STORED-RESULT"

    async def test_released_key_can_be_retried(self, client):
        store = IdempotencyStore(client)
        await store.reserve("key-1")
        await store.release("key-1")
        assert (await store.reserve("key-1")).is_new is True


class TestDistributedLock:
    async def test_acquire_and_block(self, client):
        lock = DistributedLock(client)
        token = await lock.acquire("resource")
        assert token is not None
        assert await lock.acquire("resource") is None  # held

    async def test_release_frees_the_lock(self, client):
        lock = DistributedLock(client)
        token = await lock.acquire("resource")
        await lock.release("resource", token)
        assert await lock.acquire("resource") is not None

    async def test_release_with_wrong_token_is_a_noop(self, client):
        """A lock that expired and was re-acquired must not be released by the old holder."""
        lock = DistributedLock(client)
        await lock.acquire("resource")
        await lock.release("resource", "wrong-token")
        assert await lock.acquire("resource") is None  # still held


class TestFailOpen:
    """The most important property: Redis being down never fails a request (brief §17)."""

    @pytest.fixture
    async def broken_client(self):
        # A real client pointed at a port nothing listens on — every op will fail.
        c = RedisClient(RedisSettings(_env_file=None, url="redis://127.0.0.1:1/0"))
        try:
            yield c
        finally:
            await c.aclose()

    async def test_cache_degrades_to_a_miss(self, broken_client):
        cache = RedisPlanCache(broken_client, CacheTTLSettings(_env_file=None))
        normalized = _normalized()
        await cache.put_plan(normalized, _plan())  # silently no-ops
        assert await cache.get_plan(normalized) is None  # miss, not an error

    async def test_rate_limiter_allows(self, broken_client):
        limiter = TokenBucketRateLimiter(broken_client, LimitSettings(_env_file=None))
        assert (await limiter.check("user")).allowed is True  # fail-open

    async def test_idempotency_treats_every_request_as_new(self, broken_client):
        store = IdempotencyStore(broken_client)
        assert (await store.reserve("k")).is_new is True
        assert (await store.reserve("k")).is_new is True  # no dedup, but never blocked

    async def test_lock_always_acquires(self, broken_client):
        lock = DistributedLock(broken_client)
        assert await lock.acquire("r") is not None  # proceed unserialised

    async def test_disabled_redis_is_never_available(self):
        client = RedisClient(RedisSettings(_env_file=None, enabled=False))
        assert client.available is False
        assert await client.health_check() is False
