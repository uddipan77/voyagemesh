"""Redis: trip caching, rate limiting, idempotency, and locks — all fail-open.

Every operation tolerates Redis being unavailable (brief §17): a connection failure degrades
to a cache miss, an allowed request, an un-deduplicated call, or an unheld lock — never a
failed user request. Redis is an optimisation and a safety rail, not a hard dependency.
"""

from vm_caching.client import RedisClient
from vm_caching.idempotency import (
    DistributedLock,
    IdempotencyRecord,
    IdempotencyStore,
)
from vm_caching.keys import WORKFLOW_VERSION, plan_cache_key
from vm_caching.plan_cache import RedisPlanCache
from vm_caching.rate_limit import RateLimitDecision, TokenBucketRateLimiter

__all__ = [
    "WORKFLOW_VERSION",
    "DistributedLock",
    "IdempotencyRecord",
    "IdempotencyStore",
    "RateLimitDecision",
    "RedisClient",
    "RedisPlanCache",
    "TokenBucketRateLimiter",
    "plan_cache_key",
]
