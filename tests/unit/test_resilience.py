"""Circuit breaker and retry behaviour.

These guard the request's time budget: with three agents and a 45-second deadline, a
provider that hangs on every call can exhaust the budget on retries alone. The breaker
converts that into an instant honest failure.
"""

from __future__ import annotations

import asyncio

import pytest

from vm_resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    RetryPolicy,
    retry_async,
)

pytestmark = pytest.mark.unit


class _Clock:
    """A hand-advanced monotonic clock, so breaker timing is asserted without sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def _fail() -> None:
    raise RuntimeError("boom")


async def _succeed() -> str:
    return "ok"


class TestCircuitBreaker:
    async def test_passes_calls_through_while_closed(self):
        breaker = CircuitBreaker("dep", failure_threshold=3)
        assert await breaker.call(_succeed) == "ok"
        assert breaker.state is CircuitState.CLOSED

    async def test_opens_after_threshold_consecutive_failures(self):
        breaker = CircuitBreaker("dep", failure_threshold=3)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(_fail)
        assert breaker.state is CircuitState.OPEN

    async def test_open_circuit_fails_fast_without_calling_the_dependency(self):
        breaker = CircuitBreaker("dep", failure_threshold=2)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(_fail)

        called = False

        async def tracked() -> str:
            nonlocal called
            called = True
            return "ok"

        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(tracked)
        assert called is False, "an open circuit must not reach the dependency"

    async def test_a_success_resets_the_failure_count(self):
        breaker = CircuitBreaker("dep", failure_threshold=3)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(_fail)
        await breaker.call(_succeed)
        assert breaker.consecutive_failures == 0
        # Two more failures should not open it — the counter was reset.
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(_fail)
        assert breaker.state is CircuitState.CLOSED

    async def test_half_opens_after_the_reset_timeout(self):
        clock = _Clock()
        breaker = CircuitBreaker(
            "dep", failure_threshold=2, reset_timeout_seconds=30, time_source=clock
        )
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(_fail)
        assert breaker.state is CircuitState.OPEN

        clock.advance(31)
        # The next call is allowed through as a probe (half-open).
        assert await breaker.call(_succeed) == "ok"

    async def test_recovers_to_closed_after_enough_successful_probes(self):
        clock = _Clock()
        breaker = CircuitBreaker(
            "dep",
            failure_threshold=2,
            reset_timeout_seconds=30,
            half_open_max_calls=2,
            time_source=clock,
        )
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(_fail)
        clock.advance(31)
        await breaker.call(_succeed)
        await breaker.call(_succeed)
        assert breaker.state is CircuitState.CLOSED

    async def test_a_failed_probe_reopens_immediately(self):
        clock = _Clock()
        breaker = CircuitBreaker(
            "dep", failure_threshold=2, reset_timeout_seconds=30, time_source=clock
        )
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(_fail)
        clock.advance(31)
        with pytest.raises(RuntimeError):
            await breaker.call(_fail)  # probe fails
        assert breaker.state is CircuitState.OPEN
        # And it stays closed to traffic without waiting for the threshold again.
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(_succeed)

    async def test_concurrent_failures_do_not_overshoot_the_threshold(self):
        """The lock must stop a burst of concurrent failures racing the threshold check."""
        breaker = CircuitBreaker("dep", failure_threshold=5)
        results = await asyncio.gather(
            *(breaker.call(_fail) for _ in range(20)), return_exceptions=True
        )
        # Once open, further calls raise CircuitBreakerOpenError rather than RuntimeError.
        open_errors = sum(isinstance(r, CircuitBreakerOpenError) for r in results)
        assert breaker.state is CircuitState.OPEN
        assert open_errors > 0, (
            "some concurrent calls should have been rejected by the open circuit"
        )

    def test_threshold_must_be_positive(self):
        with pytest.raises(ValueError):
            CircuitBreaker("dep", failure_threshold=0)

    async def test_reset_forces_closed(self):
        breaker = CircuitBreaker("dep", failure_threshold=1)
        with pytest.raises(RuntimeError):
            await breaker.call(_fail)
        assert breaker.state is CircuitState.OPEN
        breaker.reset()
        assert breaker.state is CircuitState.CLOSED


class TestRetryPolicy:
    def test_delay_grows_exponentially(self):
        policy = RetryPolicy(base_delay_seconds=0.1, jitter_ratio=0.0)
        assert policy.delay_for(1) == pytest.approx(0.1)
        assert policy.delay_for(2) == pytest.approx(0.2)
        assert policy.delay_for(3) == pytest.approx(0.4)

    def test_delay_is_capped(self):
        policy = RetryPolicy(base_delay_seconds=1.0, max_delay_seconds=3.0, jitter_ratio=0.0)
        assert policy.delay_for(10) == pytest.approx(3.0)

    def test_jitter_is_deterministic_for_a_seed(self):
        policy = RetryPolicy(jitter_ratio=0.25)
        assert policy.delay_for(2, seed="url-a") == policy.delay_for(2, seed="url-a")

    def test_jitter_varies_by_seed(self):
        policy = RetryPolicy(jitter_ratio=0.25)
        assert policy.delay_for(2, seed="url-a") != policy.delay_for(2, seed="url-b")

    def test_jitter_stays_within_bounds(self):
        policy = RetryPolicy(base_delay_seconds=1.0, jitter_ratio=0.25, max_delay_seconds=100)
        for seed in ("a", "b", "c", "d", "e"):
            delay = policy.delay_for(1, seed=seed)
            assert 0.75 <= delay <= 1.25

    def test_rejects_invalid_configuration(self):
        with pytest.raises(ValueError):
            RetryPolicy(max_attempts=0)
        with pytest.raises(ValueError):
            RetryPolicy(base_delay_seconds=0)


class TestRetryAsync:
    @pytest.fixture(autouse=True)
    def _instant(self, monkeypatch):
        async def instant(_seconds: float) -> None:
            return None

        monkeypatch.setattr(asyncio, "sleep", instant)

    async def test_returns_on_first_success(self):
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        assert await retry_async(op, policy=RetryPolicy(max_attempts=3)) == "ok"
        assert calls == 1

    async def test_retries_until_success(self):
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("transient")
            return "ok"

        assert await retry_async(op, policy=RetryPolicy(max_attempts=5)) == "ok"
        assert calls == 3

    async def test_raises_after_exhausting_attempts(self):
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise RuntimeError("always")

        with pytest.raises(RuntimeError, match="always"):
            await retry_async(op, policy=RetryPolicy(max_attempts=3))
        assert calls == 3

    async def test_respects_the_retry_on_predicate(self):
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            await retry_async(
                op,
                policy=RetryPolicy(max_attempts=5),
                retry_on=lambda exc: not isinstance(exc, ValueError),
            )
        assert calls == 1, "a non-retryable error must fail immediately"

    async def test_cancellation_is_never_retried(self):
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await retry_async(op, policy=RetryPolicy(max_attempts=5))
        assert calls == 1
