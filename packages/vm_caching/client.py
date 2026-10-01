"""Redis client wrapper with graceful degradation.

Every Redis operation in VoyageMesh goes through this wrapper, and every one of them tolerates
Redis being unavailable (brief §17). A connection failure never fails the user's request: the
cache degrades to a miss, the rate limiter to "allow", the lock to "proceed", the idempotency
store to "no dedup". Redis is an optimisation and a safety rail, never a hard dependency.

The ``call`` helper is the single place that catches Redis errors and substitutes a fallback,
so no call site has to remember to. It also short-circuits when Redis is disabled by
configuration, so ``REDIS_ENABLED=false`` runs the whole system cache-free.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from vm_config.settings import RedisSettings

if TYPE_CHECKING:
    from redis.asyncio import Redis as _Redis

    Redis = _Redis[bytes]

__all__ = ["RedisClient"]

logger = logging.getLogger(__name__)


class RedisClient:
    """A thin, fail-open wrapper around ``redis.asyncio.Redis``."""

    def __init__(self, settings: RedisSettings, *, redis: Redis | None = None) -> None:
        self._settings = settings
        self._prefix = settings.key_prefix
        self._enabled = settings.enabled
        self._degraded = False

        if redis is not None:
            self._redis: Redis | None = redis
        elif not settings.enabled:
            self._redis = None
        else:
            from redis.asyncio import Redis as AsyncRedis

            self._redis = AsyncRedis.from_url(
                settings.url,
                socket_timeout=settings.socket_timeout_seconds,
                socket_connect_timeout=settings.connect_timeout_seconds,
                max_connections=settings.max_connections,
                decode_responses=False,
            )

    @property
    def available(self) -> bool:
        """Whether Redis is enabled and has not recently failed."""
        return self._enabled and self._redis is not None and not self._degraded

    def key(self, *parts: str) -> str:
        """Build a namespaced key: ``vm:part1:part2``."""
        return ":".join([self._prefix, *parts])

    async def call[T](
        self,
        operation: Callable[[Redis], Awaitable[T]],
        *,
        fallback: T,
        label: str = "redis_op",
    ) -> T:
        """Run a Redis operation, returning ``fallback`` on any failure.

        This is the fail-open boundary. A single failure marks the client degraded so a
        storm of operations against a down Redis does not each pay the connect timeout;
        :meth:`reset` (or a successful ``ping`` via :meth:`health_check`) clears it.
        """
        if not self.available or self._redis is None:
            return fallback
        try:
            return await operation(self._redis)
        except Exception as exc:
            if not self._degraded:
                logger.warning("redis_degraded", extra={"op": label, "error": type(exc).__name__})
            self._degraded = True
            return fallback

    async def health_check(self) -> bool:
        """Ping Redis. A success clears the degraded flag; a failure sets it. Never raises."""
        if not self._enabled or self._redis is None:
            return False
        try:
            await self._redis.ping()
        except Exception:
            self._degraded = True
            return False
        self._degraded = False
        return True

    def reset(self) -> None:
        """Clear the degraded flag (e.g. after a manual recovery)."""
        self._degraded = False

    async def ping(self) -> bool:
        """A live round-trip for readiness checks. Fail-open: returns ``False`` rather than
        raising when Redis is disabled or unreachable, so a health endpoint stays responsive."""
        if not self._enabled or self._redis is None:
            return False
        return await self.call(lambda r: r.ping(), fallback=False, label="ping")

    async def aclose(self) -> None:
        # `aclose` on redis.asyncio is present at runtime but missing from some stub
        # versions; call it defensively and suppress a dead-socket error on shutdown.
        if self._redis is not None:
            close = getattr(self._redis, "aclose", None) or self._redis.close
            with contextlib.suppress(Exception):
                await close()

    @property
    def raw(self) -> Any:
        """The underlying client, for advanced operations. May be ``None``."""
        return self._redis
