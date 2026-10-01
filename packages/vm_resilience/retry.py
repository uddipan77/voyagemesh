"""Retry with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

__all__ = ["RetryPolicy", "retry_async"]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Backoff configuration.

    Jitter is deliberate: without it, N callers that fail together retry together, and the
    recovering dependency is hit by a synchronised thundering herd at each interval.
    """

    max_attempts: int = 3
    base_delay_seconds: float = 0.2
    max_delay_seconds: float = 4.0
    jitter_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_seconds <= 0:
            raise ValueError("base_delay_seconds must be positive")

    def delay_for(self, attempt: int, *, seed: str = "") -> float:
        """Delay before ``attempt`` (1-based; the delay *after* attempt 1 is ``delay_for(1)``).

        Jitter is derived from a hash of ``seed`` and the attempt number rather than from
        :mod:`random`, so a test can assert exact delays without patching global RNG state.
        """
        exponential: float = min(
            self.base_delay_seconds * (2.0 ** (attempt - 1)), self.max_delay_seconds
        )
        if self.jitter_ratio <= 0:
            return exponential
        digest = hashlib.sha256(f"{seed}:{attempt}".encode()).digest()
        unit = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF  # [0, 1]
        offset = (unit * 2 - 1) * self.jitter_ratio  # [-ratio, +ratio]
        return float(max(0.0, exponential * (1 + offset)))


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    retry_on: Callable[[BaseException], bool] | None = None,
    seed: str = "",
) -> T:
    """Run ``operation``, retrying transient failures.

    Args:
        operation: The zero-argument coroutine function to run.
        policy: Backoff configuration.
        retry_on: Predicate deciding whether an exception is worth retrying. Defaults to
            retrying everything except :class:`asyncio.CancelledError`.
        seed: Stabilises jitter; pass the URL or task ID.

    Returns:
        Whatever ``operation`` returns.

    Raises:
        The last exception raised by ``operation`` once attempts are exhausted.
    """
    policy = policy or RetryPolicy()
    should_retry = retry_on or _default_retry_on
    last: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await operation()
        except asyncio.CancelledError:
            # Cancellation is a control-flow signal, never a transient fault. Retrying it
            # would defeat the caller's timeout.
            raise
        except Exception as exc:
            last = exc
            if attempt == policy.max_attempts or not should_retry(exc):
                raise
            await asyncio.sleep(policy.delay_for(attempt, seed=seed))

    # Unreachable in practice: the final attempt always returns or re-raises above. The
    # raise is here only so the function has no implicit `return None` path — `max_attempts`
    # is validated >= 1, so `last` is always set if control reaches here.
    raise last from None  # type: ignore[misc]  # pragma: no cover


def _default_retry_on(exc: BaseException) -> bool:
    return not isinstance(exc, asyncio.CancelledError)
