"""Shared boundaries for external search adapters. Never expose provider error bodies."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import httpx

from vm_config.settings import ProviderSettings
from vm_resilience.safe_http import SafeHTTPClient


def make_provider_client(settings: ProviderSettings) -> SafeHTTPClient:
    return SafeHTTPClient(
        allowed_hosts=settings.allowed_hosts,
        timeout_seconds=settings.http_timeout_seconds,
        max_response_bytes=settings.max_response_bytes,
        user_agent=settings.user_agent,
        max_retries=0,  # The end-to-end trip has a bounded deadline.
    )


def provider_lifespan(
    client: SafeHTTPClient | None,
) -> Callable[[Any], AbstractAsyncContextManager[dict[str, object]]]:
    @asynccontextmanager
    async def lifespan(_server: Any) -> AsyncIterator[dict[str, object]]:
        try:
            yield {}
        finally:
            if client is not None:
                await client.aclose()

    return lifespan


def object_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("expected a provider object")
    return value


def records(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("expected a provider array")
    return [object_payload(item) for item in value]


def failure_message(provider: str, exc: Exception) -> str:
    """Exceptions can contain credentials, query strings or entire supplier responses."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            return f"{provider}: authentication or account access was refused. Check the key."
        if status == 429:
            return f"{provider}: rate limit reached. Try again later."
        return f"{provider}: search failed (HTTP {status})."
    if isinstance(exc, httpx.TimeoutException):
        return f"{provider}: search timed out."
    return f"{provider}: search unavailable or response could not be validated."
