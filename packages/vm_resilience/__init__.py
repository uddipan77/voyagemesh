"""Resilience primitives: retries, circuit breakers, and SSRF-safe HTTP.

All outbound provider traffic passes through :class:`SafeHTTPClient`, which lives inside
the MCP servers — so the system's entire egress surface is auditable in one place.
"""

from vm_resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)
from vm_resilience.retry import RetryPolicy, retry_async
from vm_resilience.safe_http import (
    ResponseTooLargeError,
    SafeHTTPClient,
    UnsafeURLError,
)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitState",
    "ResponseTooLargeError",
    "RetryPolicy",
    "SafeHTTPClient",
    "UnsafeURLError",
    "retry_async",
]
