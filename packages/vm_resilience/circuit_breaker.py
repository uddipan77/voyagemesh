"""Circuit breaker.

Stops a failing dependency from consuming the caller's time budget. After
``failure_threshold`` consecutive failures the circuit opens and subsequent calls fail
immediately; after ``reset_timeout`` it half-opens and lets a limited number of probes
through to test recovery.

The value here is specific: with a 45-second request deadline and three agents, a provider
that hangs for 8 seconds on every call can exhaust the budget on retries alone. An open
circuit converts that into an instant, honest "degraded" result.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum

__all__ = ["CircuitBreaker", "CircuitBreakerOpenError", "CircuitState"]


class CircuitState(StrEnum):
    CLOSED = "closed"
    """Normal operation."""

    OPEN = "open"
    """Failing fast. No calls reach the dependency."""

    HALF_OPEN = "half_open"
    """Probing recovery with a limited number of trial calls."""


class CircuitBreakerOpenError(Exception):
    """Raised instead of calling a dependency whose circuit is open."""

    def __init__(self, name: str, *, retry_after_seconds: float) -> None:
        super().__init__(
            f"circuit '{name}' is open; not calling the dependency for another "
            f"{retry_after_seconds:.1f}s"
        )
        self.name = name
        self.retry_after_seconds = retry_after_seconds


class CircuitBreaker:
    """An async circuit breaker guarding one dependency.

    Instances are safe to share across concurrent tasks: state transitions are guarded by
    an :class:`asyncio.Lock`, so a burst of concurrent failures cannot race the threshold
    check and let far more calls through than intended.
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        reset_timeout_seconds: float = 30.0,
        half_open_max_calls: int = 2,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        self.name = name
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout_seconds
        self._half_open_max = max(1, half_open_max_calls)
        # monotonic by default: a wall-clock jump (NTP correction, DST) must not make a
        # circuit appear to have been open for hours.
        self._now = time_source

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._half_open_calls = 0
        self._half_open_successes = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    async def call[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run ``operation`` under the breaker.

        Raises:
            CircuitBreakerOpenError: The circuit is open, or half-open and already at its
                probe limit.
        """
        await self._before_call()
        try:
            result = await operation()
        except Exception:
            await self._record_failure()
            raise
        await self._record_success()
        return result

    async def _before_call(self) -> None:
        async with self._lock:
            if self._state is CircuitState.OPEN:
                elapsed = self._now() - self._opened_at
                if elapsed < self._reset_timeout:
                    raise CircuitBreakerOpenError(
                        self.name, retry_after_seconds=self._reset_timeout - elapsed
                    )
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._half_open_successes = 0

            if self._state is CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max:
                    # Probes are in flight; hold everyone else back rather than
                    # stampeding a dependency that may still be unwell.
                    raise CircuitBreakerOpenError(self.name, retry_after_seconds=0.0)
                self._half_open_calls += 1

    async def _record_success(self) -> None:
        async with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self._half_open_max:
                    self._close()
                return
            self._consecutive_failures = 0

    async def _record_failure(self) -> None:
        async with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                # A failure during probing means the dependency is still unwell. Reopen
                # immediately rather than waiting for the threshold again.
                self._open()
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._threshold:
                self._open()

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._now()
        self._half_open_calls = 0
        self._half_open_successes = 0

    def _close(self) -> None:
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._half_open_calls = 0
        self._half_open_successes = 0

    def reset(self) -> None:
        """Force the circuit closed. For tests and administrative recovery."""
        self._close()

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name!r}, state={self._state.value}, "
            f"failures={self._consecutive_failures})"
        )
