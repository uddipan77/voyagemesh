"""Threat T-3: SSRF and arbitrary outbound requests.

`SafeHTTPClient` is the whole system's egress surface, so it gets adversarial tests rather
than happy-path ones. Every case here is a URL that a naive allowlist check would let
through.
"""

from __future__ import annotations

import httpx
import pytest

from vm_resilience import (
    CircuitBreakerOpenError,
    ResponseTooLargeError,
    SafeHTTPClient,
    UnsafeURLError,
)

pytestmark = [pytest.mark.security, pytest.mark.unit]

ALLOWED = ["api.open-meteo.com", "geocoding-api.open-meteo.com"]


@pytest.fixture(autouse=True)
def _instant_backoff(monkeypatch):
    """Exercise the retry paths without paying their wall-clock cost.

    Backoff timing is asserted directly in the retry unit tests; here it is pure latency.
    """
    import asyncio

    async def instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant)


@pytest.fixture
async def client():
    instance = SafeHTTPClient(allowed_hosts=ALLOWED)
    try:
        yield instance
    finally:
        await instance.aclose()


class TestAllowlist:
    def test_permits_an_allowlisted_host(self, client):
        assert client.check_url("https://api.open-meteo.com/v1/forecast") == "api.open-meteo.com"

    @pytest.mark.parametrize(
        "url",
        [
            "https://evil.example.com/steal",
            "https://api.open-meteo.com.evil.com/",  # suffix confusion
            "https://notapi.open-meteo.com/",  # prefix confusion
            "https://open-meteo.com/",  # parent domain is not the allowlisted host
        ],
    )
    def test_refuses_hosts_outside_the_allowlist(self, client, url):
        with pytest.raises(UnsafeURLError, match="allowlist"):
            client.check_url(url)

    def test_host_matching_is_case_insensitive(self, client):
        assert client.check_url("https://API.Open-Meteo.COM/v1/forecast")

    def test_empty_allowlist_fails_closed(self):
        """A misconfiguration must block egress, never open it."""
        closed = SafeHTTPClient(allowed_hosts=[])
        with pytest.raises(UnsafeURLError):
            closed.check_url("https://api.open-meteo.com/")


class TestInternalTargets:
    @pytest.mark.parametrize(
        "url",
        [
            "https://169.254.169.254/latest/meta-data/",  # cloud metadata
            "https://127.0.0.1/admin",
            "https://10.0.0.5/internal",
            "https://192.168.1.1/",
            "https://[::1]/admin",
            "https://0.0.0.0/",
        ],
    )
    def test_refuses_literal_ip_addresses(self, client, url):
        """Blocks the metadata endpoint and loopback without resolving every DNS answer."""
        with pytest.raises(UnsafeURLError):
            client.check_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://keycloak:8080/realms/master",
            "http://postgres:5432/",
            "http://orchestrator:8001/internal",
        ],
    )
    def test_refuses_internal_service_names(self, client, url):
        with pytest.raises(UnsafeURLError):
            client.check_url(url)


class TestAuthorityTricks:
    @pytest.mark.parametrize(
        "url",
        [
            "https://api.open-meteo.com@evil.com/",
            "https://user:pass@evil.com/",
            "https://api.open-meteo.com:pass@evil.com/path",
        ],
    )
    def test_refuses_credentials_in_the_authority(self, client, url):
        """`https://allowed@evil.com/` targets evil.com — rejected outright."""
        with pytest.raises(UnsafeURLError, match=r"credentials|allowlist"):
            client.check_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://api.open-meteo.com/",
            "gopher://api.open-meteo.com/",
            "data:text/plain;base64,SGVsbG8=",
            "javascript:alert(1)",
            "//api.open-meteo.com/no-scheme",
        ],
    )
    def test_refuses_non_http_schemes(self, client, url):
        with pytest.raises(UnsafeURLError):
            client.check_url(url)

    def test_refuses_plaintext_http_by_default(self, client):
        with pytest.raises(UnsafeURLError, match="plaintext"):
            client.check_url("http://api.open-meteo.com/v1/forecast")

    def test_http_can_be_enabled_explicitly_for_local_fixtures(self):
        permissive = SafeHTTPClient(allowed_hosts=ALLOWED, allow_http=True)
        assert permissive.check_url("http://api.open-meteo.com/v1/forecast")

    @pytest.mark.parametrize("url", ["", "   ", "https://", "not a url at all", "https:///path"])
    def test_refuses_malformed_urls(self, client, url):
        with pytest.raises(UnsafeURLError):
            client.check_url(url)


class TestRedirects:
    async def test_redirect_is_refused_not_followed(self, client, respx_mock):
        """An allowlisted host must not be able to bounce us to an internal one."""
        respx_mock.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(302, headers={"Location": "http://169.254.169.254/"})
        )
        with pytest.raises(UnsafeURLError, match="redirect"):
            await client.get_json("https://api.open-meteo.com/v1/forecast")


class TestResponseLimits:
    async def test_oversized_declared_body_is_refused(self, respx_mock):
        small = SafeHTTPClient(allowed_hosts=ALLOWED, max_response_bytes=1000)
        respx_mock.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, headers={"content-length": "99999"}, json={"ok": True})
        )
        with pytest.raises(ResponseTooLargeError, match="declares"):
            await small.get_json("https://api.open-meteo.com/v1/forecast")
        await small.aclose()

    async def test_oversized_streamed_body_is_aborted(self, respx_mock):
        """A provider that *lies* about content-length must not exhaust memory.

        The declared-size check is the cheap first line; this asserts the second one, which
        is the only defence when the header is absent or dishonest.
        """
        small = SafeHTTPClient(allowed_hosts=ALLOWED, max_response_bytes=500)
        respx_mock.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, headers={"content-length": "10"}, content=b"x" * 5000)
        )
        with pytest.raises(ResponseTooLargeError, match="while streaming"):
            await small.get_json("https://api.open-meteo.com/v1/forecast")
        await small.aclose()

    async def test_invalid_json_is_reported_not_guessed(self, client, respx_mock):
        respx_mock.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, content=b"<html>not json</html>")
        )
        with pytest.raises(httpx.HTTPError, match="invalid JSON"):
            await client.get_json("https://api.open-meteo.com/v1/forecast")

    async def test_json_scalar_is_rejected(self, client, respx_mock):
        respx_mock.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, content=b"42")
        )
        with pytest.raises(httpx.HTTPError, match="scalar"):
            await client.get_json("https://api.open-meteo.com/v1/forecast")


class TestHappyPathAndResilience:
    async def test_successful_json_fetch(self, client, respx_mock):
        respx_mock.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(200, json={"temperature": 21.5})
        )
        assert await client.get_json("https://api.open-meteo.com/v1/forecast") == {
            "temperature": 21.5
        }

    async def test_server_errors_are_retried(self, respx_mock):
        instance = SafeHTTPClient(allowed_hosts=ALLOWED, max_retries=2)
        route = respx_mock.get("https://api.open-meteo.com/v1/forecast")
        route.side_effect = [
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        ]
        assert await instance.get_json("https://api.open-meteo.com/v1/forecast") == {"ok": True}
        assert route.call_count == 2
        await instance.aclose()

    async def test_client_errors_are_not_retried(self, respx_mock):
        """A 400 will not become a 200; retrying wastes the caller's time budget."""
        instance = SafeHTTPClient(allowed_hosts=ALLOWED, max_retries=2)
        route = respx_mock.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(400)
        )
        with pytest.raises(httpx.HTTPStatusError):
            await instance.get_json("https://api.open-meteo.com/v1/forecast")
        assert route.call_count == 1
        await instance.aclose()

    async def test_circuit_opens_after_repeated_failures(self, respx_mock):
        instance = SafeHTTPClient(allowed_hosts=ALLOWED, max_retries=0, breaker_failure_threshold=3)
        route = respx_mock.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(503)
        )
        for _ in range(3):
            with pytest.raises(httpx.HTTPStatusError):
                await instance.get_json("https://api.open-meteo.com/v1/forecast")

        assert instance.breaker_state("api.open-meteo.com") == "open"

        calls_before = route.call_count
        with pytest.raises(CircuitBreakerOpenError):
            await instance.get_json("https://api.open-meteo.com/v1/forecast")
        assert route.call_count == calls_before, "open circuit must not reach the network"
        await instance.aclose()

    async def test_breakers_are_isolated_per_host(self, respx_mock):
        instance = SafeHTTPClient(allowed_hosts=ALLOWED, max_retries=0, breaker_failure_threshold=2)
        respx_mock.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(503)
        )
        respx_mock.get("https://geocoding-api.open-meteo.com/v1/search").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        for _ in range(2):
            with pytest.raises(httpx.HTTPStatusError):
                await instance.get_json("https://api.open-meteo.com/v1/forecast")

        assert instance.breaker_state("api.open-meteo.com") == "open"
        assert instance.breaker_state("geocoding-api.open-meteo.com") == "closed"
        # One failing provider must not disable an unrelated one.
        assert await instance.get_json("https://geocoding-api.open-meteo.com/v1/search") == {
            "results": []
        }
        await instance.aclose()

    async def test_refused_urls_are_not_retried(self, client):
        """A blocked URL will never become allowed; retrying it is pure waste."""
        with pytest.raises(UnsafeURLError):
            await client.get_json("https://evil.example.com/")


class TestErrorMessagesLeakNothing:
    def test_message_names_the_host_not_the_full_url(self, client):
        with pytest.raises(UnsafeURLError) as exc:
            client.check_url("https://evil.example.com/path?token=SECRET123&user=alice")
        message = str(exc.value)
        assert "evil.example.com" in message
        assert "SECRET123" not in message
        assert "alice" not in message
