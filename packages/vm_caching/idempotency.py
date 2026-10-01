"""Idempotency keys and request deduplication (brief §17, §20).

Lets a client safely retry a plan request: the first request with a given ``Idempotency-Key``
runs and its result is stored; a retry with the same key returns the stored result instead of
planning again. Also provides a simple distributed lock for cache-stampede protection — when
many identical requests arrive at once, one computes and the rest wait or fall through.

Fail-open, like everything else here: if Redis is down, idempotency is simply not enforced
(every request runs), because dropping the dedup is far better than dropping the request.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from vm_caching.client import RedisClient

__all__ = ["DistributedLock", "IdempotencyRecord", "IdempotencyStore"]


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    """The outcome of reserving an idempotency key."""

    is_new: bool
    """True when this caller won the reservation and should do the work."""

    stored_result: str | None = None
    """The previously stored result, when a completed key is replayed."""

    conflict: bool = False
    """True when the key is reserved but not yet complete — a concurrent in-flight request."""


class IdempotencyStore:
    """Stores idempotency reservations and their results."""

    _RESERVED = b"__reserved__"

    def __init__(self, client: RedisClient, *, ttl_seconds: int = 900) -> None:
        self._client = client
        self._ttl = ttl_seconds

    async def reserve(self, key: str) -> IdempotencyRecord:
        """Attempt to reserve ``key``.

        Returns ``is_new=True`` if this caller should do the work; ``stored_result`` if the
        key already completed; ``conflict=True`` if another request holds it in-flight.
        """
        redis_key = self._client.key("idem", key)

        async def run(r):  # type: ignore[no-untyped-def]
            # SET NX makes the reservation atomic — exactly one caller wins.
            won = await r.set(redis_key, self._RESERVED, nx=True, ex=self._ttl)
            if won:
                return IdempotencyRecord(is_new=True)
            existing = await r.get(redis_key)
            if existing == self._RESERVED:
                return IdempotencyRecord(is_new=False, conflict=True)
            decoded = existing.decode("utf-8") if isinstance(existing, bytes) else existing
            return IdempotencyRecord(is_new=False, stored_result=decoded)

        # Fail-open: without Redis, treat every request as new (no dedup, but never blocked).
        return await self._client.call(
            run, fallback=IdempotencyRecord(is_new=True), label="idem_reserve"
        )

    async def complete(self, key: str, result: str) -> None:
        """Store the result for ``key`` so a later retry replays it."""
        redis_key = self._client.key("idem", key)
        await self._client.call(
            lambda r: r.set(redis_key, result, ex=self._ttl), fallback=None, label="idem_complete"
        )

    async def release(self, key: str) -> None:
        """Release a reservation without a result (e.g. the work failed), so a retry may run."""
        redis_key = self._client.key("idem", key)
        await self._client.call(lambda r: r.delete(redis_key), fallback=0, label="idem_release")


class DistributedLock:
    """A best-effort distributed lock for cache-stampede protection.

    Acquire returns a token; release only deletes the key if the token still matches, so a
    lock that expired and was re-acquired by someone else is not released out from under them.
    Fail-open: without Redis, ``acquire`` succeeds so work proceeds unserialised.
    """

    def __init__(self, client: RedisClient, *, ttl_seconds: int = 30) -> None:
        self._client = client
        self._ttl = ttl_seconds

    async def acquire(self, name: str) -> str | None:
        """Return a lock token if acquired, else ``None``. Succeeds if Redis is down."""
        token = secrets.token_hex(12)
        key = self._client.key("lock", name)

        async def run(r):  # type: ignore[no-untyped-def]
            won = await r.set(key, token, nx=True, ex=self._ttl)
            return token if won else None

        # Fail-open: an unreachable Redis returns the token so the caller proceeds.
        return await self._client.call(run, fallback=token, label="lock_acquire")

    async def release(self, name: str, token: str) -> None:
        key = self._client.key("lock", name)
        lua = (
            "if redis.call('get', KEYS[1]) == ARGV[1] then "
            "return redis.call('del', KEYS[1]) else return 0 end"
        )
        await self._client.call(
            lambda r: r.eval(lua, 1, key, token),  # type: ignore[no-untyped-call]
            fallback=0,
            label="lock_release",
        )
