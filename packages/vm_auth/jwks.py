"""JWKS client — fetches and caches Keycloak's signing keys.

The gateway validates a user's JWT against Keycloak's published JSON Web Key Set. Keys are
cached for a configurable TTL and re-fetched on a cache miss (a key ID we have not seen),
which handles Keycloak's key rotation without a restart.

The fetch uses a plain httpx client, not the SSRF-safe provider client: the JWKS endpoint is
an internal, configured Keycloak URL, not a model- or user-supplied one.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

__all__ = ["JwksClient", "JwksError"]

logger = logging.getLogger(__name__)


class JwksError(Exception):
    """The signing keys could not be fetched or a key ID was not found."""


class JwksClient:
    """Fetches and caches JWKS signing keys by ``kid``."""

    def __init__(
        self,
        jwks_url: str,
        *,
        cache_seconds: int = 300,
        timeout_seconds: float = 5.0,
        static_keys: list[dict[str, Any]] | None = None,
        time_source: Any = time.monotonic,
    ) -> None:
        self._url = jwks_url
        self._cache_seconds = cache_seconds
        self._timeout = timeout_seconds
        self._now = time_source
        # ``static_keys`` bypasses the network entirely — used by tests, which build their own
        # keypair and JWK so JWT validation can be exercised without a running Keycloak.
        self._static = static_keys
        self._keys: dict[str, dict[str, Any]] = {}
        self._fetched_at = 0.0

    async def get_signing_key(self, kid: str) -> dict[str, Any]:
        """Return the JWK for ``kid``, refreshing the cache if it is missing or stale."""
        if kid in self._keys and not self._is_stale():
            return self._keys[kid]

        await self._refresh()
        if kid not in self._keys:
            # A refresh that still lacks the kid means the token was signed by a key we do not
            # trust — reject rather than guess.
            raise JwksError(f"no signing key found for kid '{kid}'")
        return self._keys[kid]

    def _is_stale(self) -> bool:
        if self._static is not None:
            return False
        return bool((self._now() - self._fetched_at) >= self._cache_seconds)

    async def _refresh(self) -> None:
        if self._static is not None:
            self._keys = {k["kid"]: k for k in self._static if "kid" in k}
            return
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(self._url)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            raise JwksError(f"could not fetch JWKS ({type(exc).__name__})") from None

        keys = payload.get("keys") if isinstance(payload, dict) else None
        if not isinstance(keys, list):
            raise JwksError("JWKS response did not contain a key list")
        self._keys = {k["kid"]: k for k in keys if isinstance(k, dict) and "kid" in k}
        self._fetched_at = self._now()
