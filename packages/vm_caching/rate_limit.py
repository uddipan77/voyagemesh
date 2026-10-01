"""Distributed token-bucket rate limiting (brief §17, §24).

A per-identity token bucket in Redis, evaluated atomically by a small Lua script so a burst of
concurrent requests cannot each read-then-write and overshoot the limit. Fail-open: if Redis
is unavailable the limiter allows the request, because a rate limiter that fails *closed* would
turn a Redis outage into a full outage.
"""

from __future__ import annotations

from dataclasses import dataclass

from vm_caching.client import RedisClient
from vm_config.settings import LimitSettings

__all__ = ["RateLimitDecision", "TokenBucketRateLimiter"]

# Atomic token bucket. KEYS[1] = bucket key; ARGV = rate_per_sec, capacity, now_ms, cost.
# Returns {allowed(0/1), remaining_tokens, retry_after_ms}. Refills continuously based on the
# elapsed time since the last update, so there is no separate refill job.
_BUCKET_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then
  tokens = capacity
  ts = now
end

local elapsed = math.max(0, now - ts) / 1000.0
tokens = math.min(capacity, tokens + elapsed * rate)

local allowed = 0
local retry_after = 0
if tokens >= cost then
  allowed = 1
  tokens = tokens - cost
else
  retry_after = math.ceil(((cost - tokens) / rate) * 1000)
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('PEXPIRE', key, math.ceil((capacity / rate) * 1000) + 1000)
return {allowed, math.floor(tokens), retry_after}
"""


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_seconds: float


class TokenBucketRateLimiter:
    """Per-identity distributed rate limiter."""

    def __init__(
        self, client: RedisClient, limits: LimitSettings, *, now_ms: int | None = None
    ) -> None:
        self._client = client
        self._rate = limits.rate_limit_requests_per_minute / 60.0  # tokens per second
        self._capacity = float(limits.rate_limit_burst)
        self._fixed_now = now_ms  # for deterministic tests

    async def check(self, identity: str, *, cost: int = 1) -> RateLimitDecision:
        """Consume ``cost`` tokens for ``identity``. Allows the request if Redis is down."""
        import time

        now = self._fixed_now if self._fixed_now is not None else int(time.time() * 1000)
        key = self._client.key("ratelimit", identity)

        async def run(r):  # type: ignore[no-untyped-def]
            result = await r.eval(_BUCKET_LUA, 1, key, self._rate, self._capacity, now, cost)
            allowed, remaining, retry_ms = result
            return RateLimitDecision(
                allowed=bool(allowed),
                remaining=int(remaining),
                retry_after_seconds=round(int(retry_ms) / 1000.0, 3),
            )

        # Fail-open: an unreachable Redis allows the request rather than blocking everyone.
        return await self._client.call(
            run,
            fallback=RateLimitDecision(
                allowed=True, remaining=int(self._capacity), retry_after_seconds=0.0
            ),
            label="rate_limit",
        )
