"""User authentication — Keycloak OIDC JWT validation (brief §15A).

Validates a bearer JWT presented by the frontend: signature (RS256 via JWKS), issuer,
audience, and expiry, then extracts the user's roles. The gateway depends on
:class:`JwtValidator`; a dev-only bypass exists but is refused in production by the config
validator (see :mod:`vm_config.settings`).

Security posture (threat T-6):

* **Algorithm is pinned to RS256.** ``alg: none`` and HS256-confusion are structurally
  impossible — a symmetric algorithm is never accepted, so an attacker cannot re-sign a token
  with the public key.
* **Issuer and audience are verified.** A token from another realm or minted for another
  audience is rejected.
* **Expiry is enforced** with a small configurable leeway for clock skew.
* Every failure is a specific, sanitised error — never a stack trace, never the token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import jwt

from vm_auth.jwks import JwksClient, JwksError
from vm_config.settings import AuthSettings

__all__ = [
    "AuthenticatedUser",
    "JwtValidator",
    "Role",
    "TokenError",
    "TokenErrorCode",
]


class Role(StrEnum):
    """The application roles (brief §15A)."""

    TRAVELLER = "traveller"
    EVALUATOR = "evaluator"
    ADMIN = "admin"


class TokenErrorCode(StrEnum):
    MISSING = "unauthenticated"
    EXPIRED = "token_expired"
    INVALID = "token_invalid"
    INSUFFICIENT_ROLE = "insufficient_role"


class TokenError(Exception):
    """A JWT was missing, malformed, expired, or otherwise unacceptable.

    Carries a machine-readable code and a message safe for a sanitised 401/403. Never
    contains the token.
    """

    def __init__(self, code: TokenErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    """A validated end user."""

    subject: str
    username: str | None = None
    roles: frozenset[str] = field(default_factory=frozenset)

    def has_role(self, role: str | Role) -> bool:
        return str(role) in self.roles

    def require_role(self, role: str | Role) -> None:
        if not self.has_role(role):
            raise TokenError(
                TokenErrorCode.INSUFFICIENT_ROLE,
                f"the '{role}' role is required for this operation",
            )

    @property
    def reference(self) -> str:
        """A privacy-safe handle for logging/tracing — the subject, never an email."""
        return self.subject


class JwtValidator:
    """Validates Keycloak OIDC access tokens."""

    def __init__(self, settings: AuthSettings, *, jwks_client: JwksClient | None = None) -> None:
        self._settings = settings
        self._jwks = jwks_client or JwksClient(
            settings.resolved_jwks_url, cache_seconds=settings.jwks_cache_seconds
        )

    async def validate(self, token: str) -> AuthenticatedUser:
        """Validate ``token`` and return the authenticated user.

        Raises :class:`TokenError` for any failure, with a specific code.
        """
        if not token or not token.strip():
            raise TokenError(TokenErrorCode.MISSING, "no bearer token was presented")

        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise TokenError(TokenErrorCode.INVALID, "the token header could not be read") from None

        kid = header.get("kid")
        if not kid:
            raise TokenError(TokenErrorCode.INVALID, "the token has no key id")

        try:
            jwk = await self._jwks.get_signing_key(kid)
        except JwksError:
            raise TokenError(
                TokenErrorCode.INVALID, "the token was signed by an untrusted key"
            ) from None

        try:
            public_key = jwt.algorithms.RSAAlgorithm.from_jwk(jwk)
            claims = jwt.decode(
                token,
                key=public_key,  # type: ignore[arg-type]
                algorithms=self._settings.algorithms,  # pinned — no HS256, no 'none'
                audience=self._settings.audience,
                issuer=self._settings.issuer,
                leeway=self._settings.leeway_seconds,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except jwt.ExpiredSignatureError:
            raise TokenError(TokenErrorCode.EXPIRED, "the token has expired") from None
        except jwt.InvalidIssuerError:
            raise TokenError(TokenErrorCode.INVALID, "the token issuer is not trusted") from None
        except jwt.InvalidAudienceError:
            raise TokenError(
                TokenErrorCode.INVALID, "the token was not issued for this service"
            ) from None
        except jwt.PyJWTError:
            raise TokenError(TokenErrorCode.INVALID, "the token is invalid") from None

        return AuthenticatedUser(
            subject=str(claims["sub"]),
            username=claims.get("preferred_username"),
            roles=_extract_roles(claims, self._settings.audience),
        )


def _extract_roles(claims: dict[str, Any], audience: str) -> frozenset[str]:
    """Pull realm roles and this client's roles from a Keycloak token.

    Keycloak places realm-wide roles under ``realm_access.roles`` and per-client roles under
    ``resource_access.<client>.roles``. We take both, so a role granted either way is honoured.
    """
    roles: set[str] = set()
    realm = claims.get("realm_access")
    if isinstance(realm, dict) and isinstance(realm.get("roles"), list):
        roles.update(str(r) for r in realm["roles"])

    resource = claims.get("resource_access")
    if isinstance(resource, dict):
        client = resource.get(audience)
        if isinstance(client, dict) and isinstance(client.get("roles"), list):
            roles.update(str(r) for r in client["roles"])

    return frozenset(roles)
