"""SSRF-resistant HTTP client for outbound provider calls.

Every external request in VoyageMesh goes through this client, and it lives inside the
three MCP servers — so the entire system's egress surface is auditable by reading one file
(ADR-004, threat T-3).

The guarantees, and why each is needed
--------------------------------------
* **Host allowlist.** The hostname must appear in ``allowed_hosts``. Checked before a
  socket is opened.
* **No redirects.** An allowlisted host must not be able to bounce the client to an
  internal one. ``follow_redirects=False`` is not a default we inherit — it is the point.
* **Literal-IP refusal.** Requests to raw IPs are refused, which closes the
  ``169.254.169.254`` cloud-metadata path and loopback access without needing to resolve
  and inspect every DNS answer.
* **Userinfo refusal.** ``http://allowed.example@evil.com/`` has hostname ``evil.com``;
  Python's parser gets this right, but a human reviewer reading a log line may not, so
  credentials-in-authority are rejected outright.
* **Response-size cap.** Streamed and aborted past the limit, so a hostile or broken
  provider cannot exhaust memory.
* **Timeouts, retries, circuit breaker.** Applied per host.

What this deliberately does not do
----------------------------------
It does not resolve DNS and check the resulting IPs against private ranges. That defends
against DNS-rebinding, which needs a hostile allowlisted host — and the allowlist is a
short, static list of well-known public APIs. The check is noted here rather than silently
omitted; see threat T-3's residual risk.
"""

from __future__ import annotations

import functools
import ipaddress
import logging
import ssl
from types import TracebackType
from typing import Any, Self
from urllib.parse import urlsplit

import certifi
import httpx

from vm_resilience.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from vm_resilience.retry import RetryPolicy, retry_async

__all__ = [
    "ResponseTooLargeError",
    "SafeHTTPClient",
    "UnsafeURLError",
]

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = frozenset({"https", "http"})


@functools.lru_cache(maxsize=1)
def _shared_ssl_context() -> ssl.SSLContext:
    """Build the TLS context once per process and share it across clients.

    Constructing an ``httpx.AsyncClient`` builds an ``SSLContext`` by default, which loads
    the system CA bundle — measured at roughly **one second per client on Windows**. That
    is invisible until something constructs clients per request, at which point it is a
    second of latency inside a 45-second budget, on every call.

    Sharing one context is safe: it is read-only after construction and `ssl` supports
    concurrent use across connections.
    """
    return ssl.create_default_context(cafile=certifi.where())


class UnsafeURLError(Exception):
    """A URL was refused before any network activity took place.

    The message names the *reason* and the offending host, never the full URL, which may
    contain query parameters echoing user input.
    """


class ResponseTooLargeError(Exception):
    """A response exceeded the configured size cap and was aborted mid-stream."""


class SafeHTTPClient:
    """An ``httpx.AsyncClient`` wrapper enforcing the guarantees above."""

    def __init__(
        self,
        *,
        allowed_hosts: list[str],
        timeout_seconds: float = 8.0,
        max_response_bytes: int = 2_000_000,
        user_agent: str = "VoyageMesh/0.1",
        max_retries: int = 2,
        breaker_failure_threshold: int = 5,
        breaker_reset_timeout_seconds: float = 30.0,
        allow_http: bool = False,
    ) -> None:
        """Construct the client.

        Args:
            allowed_hosts: Exact hostnames permitted. Empty means *nothing* is permitted —
                fail closed, so a misconfiguration blocks egress rather than opening it.
            allow_http: Permit plaintext ``http://``. Off by default; some local test
                fixtures need it.
        """
        self._allowed = {host.strip().lower() for host in allowed_hosts if host.strip()}
        self._max_bytes = max_response_bytes
        self._allow_http = allow_http
        self._retry = RetryPolicy(max_attempts=max_retries + 1)
        self._breakers: dict[str, CircuitBreaker] = {}
        self._breaker_config = (breaker_failure_threshold, breaker_reset_timeout_seconds)

        self._client = httpx.AsyncClient(
            verify=_shared_ssl_context(),
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    @property
    def allowed_hosts(self) -> frozenset[str]:
        return frozenset(self._allowed)

    def check_url(self, url: str) -> str:
        """Validate ``url`` and return its hostname.

        Raises:
            UnsafeURLError: The URL is malformed, uses a disallowed scheme, targets a
                literal IP, carries userinfo, or names a host outside the allowlist.
        """
        try:
            parts = urlsplit(url)
        except ValueError as exc:
            raise UnsafeURLError(f"malformed URL: {exc}") from None

        scheme = (parts.scheme or "").lower()
        if scheme not in _ALLOWED_SCHEMES:
            raise UnsafeURLError(f"scheme '{scheme or '(none)'}' is not permitted")
        if scheme == "http" and not self._allow_http:
            raise UnsafeURLError("plaintext http is not permitted for provider calls")

        if parts.username is not None or parts.password is not None:
            # `http://allowed.example@evil.com/` resolves to evil.com. Reject rather than
            # rely on every future reader of a log line parsing it correctly.
            raise UnsafeURLError("URLs containing credentials are not permitted")

        hostname = (parts.hostname or "").lower()
        if not hostname:
            raise UnsafeURLError("URL has no host")

        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            pass  # a name, not a literal address — the normal case
        else:
            raise UnsafeURLError(
                f"requests to literal IP addresses are not permitted (host '{hostname}')"
            )

        if hostname not in self._allowed:
            raise UnsafeURLError(
                f"host '{hostname}' is not in the provider allowlist; add it to "
                f"PROVIDER_ALLOWED_HOSTS if this source is intended"
            )

        return hostname

    async def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | list[Any]:
        """GET ``url`` and parse the response as JSON.

        Raises:
            UnsafeURLError: The URL failed validation.
            ResponseTooLargeError: The body exceeded the size cap.
            CircuitBreakerOpenError: The host's circuit is open.
            httpx.HTTPError: A transport or status error after retries.
        """
        return await self._request_json("GET", url, params=params, headers=headers)

    async def post_json(
        self,
        url: str,
        *,
        json: dict[str, Any],
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | list[Any]:
        """POST a search payload, with the same egress protections as GET.

        POST is deliberately not retried: even search calls can be metered, and a timeout
        does not prove the provider did not process the request. No booking endpoints use
        this client. Authentication belongs in headers, never URLs.
        """
        return await self._request_json("POST", url, params=params, headers=headers, body=json)

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        host = self.check_url(url)
        breaker = self._breaker_for(host)

        async def attempt() -> dict[str, Any] | list[Any]:
            return await breaker.call(
                lambda: self._request_json_once(method, url, params, headers, body)
            )

        if method == "POST":
            return await attempt()
        return await retry_async(attempt, policy=self._retry, retry_on=_is_transient)

    async def _request_json_once(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | list[Any]:
        body = bytearray()
        async with self._client.stream(
            method, url, params=params, headers=headers, json=payload
        ) as response:
            if response.is_redirect:
                # An allowlisted host must not be able to redirect us elsewhere.
                raise UnsafeURLError(
                    f"provider returned a redirect ({response.status_code}), which is not "
                    f"followed for provider calls"
                )
            response.raise_for_status()
            if response.status_code == 204:
                return {}  # Some search APIs use 204 for no availability.

            declared = response.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > self._max_bytes:
                raise ResponseTooLargeError(
                    f"response declares {declared} bytes, exceeding the "
                    f"{self._max_bytes} byte limit"
                )

            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > self._max_bytes:
                    # Abort mid-stream: a provider that lies about content-length must not
                    # be able to exhaust memory.
                    raise ResponseTooLargeError(
                        f"response exceeded the {self._max_bytes} byte limit while streaming"
                    )

        import json

        try:
            parsed: Any = json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise httpx.HTTPError(f"provider returned invalid JSON: {exc}") from None

        if not isinstance(parsed, dict | list):
            raise httpx.HTTPError("provider returned a JSON scalar; expected object or array")
        return parsed

    def _breaker_for(self, host: str) -> CircuitBreaker:
        breaker = self._breakers.get(host)
        if breaker is None:
            threshold, reset = self._breaker_config
            breaker = CircuitBreaker(host, failure_threshold=threshold, reset_timeout_seconds=reset)
            self._breakers[host] = breaker
        return breaker

    def breaker_state(self, host: str) -> str:
        return self._breaker_for(host.lower()).state.value

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


def _is_transient(exc: BaseException) -> bool:
    """Whether a failure is worth retrying.

    A refused URL or an oversized response will not improve on a second attempt, and an
    open circuit exists precisely to avoid more calls — so none of those are retried.
    """
    if isinstance(exc, UnsafeURLError | ResponseTooLargeError | CircuitBreakerOpenError):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status in (408, 425, 429) or 500 <= status < 600
    return isinstance(exc, httpx.TransportError)
