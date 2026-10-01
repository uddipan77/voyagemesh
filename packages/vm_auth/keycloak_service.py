"""Keycloak client-credentials service authentication (brief §15B).

The production implementation of the Phase-7 `ServiceAuthenticator` / `ServiceTokenProvider`
interfaces. The orchestrator obtains a service token from Keycloak via the client-credentials
grant and presents it on A2A calls; each agent validates that token exactly like a user token
(same JWKS, same issuer/audience checks) but requires the `agent:invoke` role.

Because it implements the same interfaces as the dev shared-secret authenticator, switching
to it is one line of wiring in `select_service_authenticator` (Phase 12) — the agents and the
orchestrator are unchanged.
"""

from __future__ import annotations

import logging
import time

import httpx
from pydantic import SecretStr

from vm_auth.jwks import JwksClient
from vm_auth.service_auth import ServiceAuthError, ServiceIdentity
from vm_auth.user_auth import JwtValidator, TokenError
from vm_config.settings import AuthSettings

__all__ = ["KeycloakServiceAuthenticator", "KeycloakTokenProvider"]

logger = logging.getLogger(__name__)


class KeycloakServiceAuthenticator:
    """Validates a service caller's Keycloak access token.

    Reuses :class:`JwtValidator` — a service token is a JWT like any other — but maps the
    result to a :class:`ServiceIdentity` and requires the invoke role. Rejecting a caller
    without a valid token is the same guarantee as the shared-secret authenticator, now backed
    by real client credentials.
    """

    def __init__(self, settings: AuthSettings, *, jwks_client: JwksClient | None = None) -> None:
        self._validator = JwtValidator(settings, jwks_client=jwks_client)

    async def authenticate(self, headers: dict[str, str]) -> ServiceIdentity:
        token = _bearer(headers)
        if token is None:
            raise ServiceAuthError("missing or malformed Authorization: Bearer header")
        try:
            user = await self._validator.validate(token)
        except TokenError as exc:
            # Do not leak which check failed to a service caller beyond a generic reason.
            raise ServiceAuthError(f"service token rejected ({exc.code.value})") from None
        return ServiceIdentity(subject=user.subject, roles=user.roles)

    @property
    def scheme_name(self) -> str:
        return "keycloak_client_credentials"


class KeycloakTokenProvider:
    """Fetches and caches a service access token via the client-credentials grant.

    The token is cached until shortly before it expires, so a burst of A2A calls shares one
    token rather than hammering Keycloak. The client secret is loaded from a file at
    construction and held only as a :class:`SecretStr`.
    """

    def __init__(
        self,
        settings: AuthSettings,
        *,
        client_secret: SecretStr,
        timeout_seconds: float = 5.0,
        time_source: object = time.monotonic,
    ) -> None:
        self._token_url = settings.resolved_token_url
        self._client_id = settings.service_client_id
        self._secret = client_secret
        self._timeout = timeout_seconds
        self._now = time_source
        self._cached_token: str | None = None
        self._expires_at = 0.0

    async def authorization_header(self) -> dict[str, str]:
        token = await self._get_token()
        return {"Authorization": f"Bearer {token}"} if token else {}

    async def _get_token(self) -> str | None:
        now = self._now()  # type: ignore[operator]
        if self._cached_token is not None and now < self._expires_at:
            return self._cached_token

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self._client_id,
                        "client_secret": self._secret.get_secret_value(),
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            # Fail-safe: a token fetch failure means the outbound call goes without a
            # credential and the agent will reject it — a clean, honest failure.
            logger.warning("service_token_fetch_failed", extra={"error": type(exc).__name__})
            return None

        token = payload.get("access_token")
        expires_in = float(payload.get("expires_in", 60))
        if not isinstance(token, str):
            return None
        # Refresh 10 s before expiry to avoid using a token that dies mid-request.
        self._cached_token = token
        self._expires_at = now + max(1.0, expires_in - 10.0)
        return token


def _bearer(headers: dict[str, str]) -> str | None:
    for key, value in headers.items():
        if key.lower() == "authorization":
            parts = value.split(None, 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1].strip()
            return None
    return None
