"""Service-to-service authentication for A2A calls.

This is the *hook* the brief asks for in Phase 7 (§7, §15B); the Keycloak client-credentials
implementation lands in Phase 11. Everything downstream depends on the
:class:`ServiceAuthenticator` protocol, so swapping the dev authenticator for the Keycloak
one changes one line of wiring and nothing in the agents.

Two rules are enforced here regardless of the concrete authenticator:

* **Least privilege across the boundary.** The user's access token is *never* forwarded to
  an agent. Only the minimal claims an agent needs — a privacy-safe user reference, roles,
  request ID, and trace ID — are propagated (threat T-4).
* **Every service call carries a verifiable identity.** An agent will not do work for a
  caller it cannot authenticate, unless the insecure dev bypass is explicitly enabled — and
  that bypass is loud and refused in production.
"""

from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import SecretStr

from vm_contracts.tracing import TraceContext

__all__ = [
    "InsecureServiceAuthenticator",
    "PropagatedClaims",
    "ServiceAuthError",
    "ServiceAuthenticator",
    "ServiceIdentity",
    "ServiceTokenProvider",
    "SharedSecretServiceAuthenticator",
    "StaticServiceTokenProvider",
]

logger = logging.getLogger(__name__)

INVOKE_ROLE = "agent:invoke"


class ServiceAuthError(Exception):
    """Authentication of a service caller failed.

    The message is safe to log and to surface as a sanitised 401/403; it never contains the
    presented token.
    """


@dataclass(frozen=True, slots=True)
class ServiceIdentity:
    """The authenticated identity of a calling service."""

    subject: str
    roles: frozenset[str] = field(default_factory=frozenset)

    def has_role(self, role: str) -> bool:
        return role in self.roles

    @property
    def can_invoke_agents(self) -> bool:
        return INVOKE_ROLE in self.roles


@dataclass(frozen=True, slots=True)
class PropagatedClaims:
    """The minimal, non-sensitive claims forwarded from the user request to an agent.

    Deliberately *not* the user's access token. An agent needs to know who the trip is for
    and how to correlate its work — nothing more (least privilege, threat T-4).
    """

    user_reference: str | None = None
    roles: tuple[str, ...] = ()
    request_id: str | None = None
    trace_id: str | None = None

    def to_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.user_reference:
            headers["x-vm-user-ref"] = self.user_reference
        if self.roles:
            headers["x-vm-user-roles"] = ",".join(self.roles)
        if self.request_id:
            headers["x-request-id"] = self.request_id
        if self.trace_id:
            headers["x-vm-trace-id"] = self.trace_id
        return headers

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> PropagatedClaims:
        lower = {k.lower(): v for k, v in headers.items()}
        roles = lower.get("x-vm-user-roles", "")
        return cls(
            user_reference=lower.get("x-vm-user-ref") or None,
            roles=tuple(r for r in roles.split(",") if r) if roles else (),
            request_id=lower.get("x-request-id") or None,
            trace_id=lower.get("x-vm-trace-id") or None,
        )

    @classmethod
    def from_trace(cls, trace: TraceContext, *, roles: tuple[str, ...] = ()) -> PropagatedClaims:
        return cls(
            user_reference=trace.user_reference,
            roles=roles,
            request_id=trace.request_id,
            trace_id=trace.trace_id,
        )


class ServiceAuthenticator(Protocol):
    """Validates a service caller's credentials from a request's headers."""

    async def authenticate(self, headers: dict[str, str]) -> ServiceIdentity:
        """Return the authenticated identity, or raise :class:`ServiceAuthError`."""
        ...

    @property
    def scheme_name(self) -> str:
        """Human-readable name of the scheme, for the Agent Card and logs."""
        ...


class ServiceTokenProvider(Protocol):
    """Supplies the credential an outbound A2A call presents."""

    async def authorization_header(self) -> dict[str, str]:
        """Return the ``Authorization`` header (possibly empty) for an outbound call."""
        ...


# ---------------------------------------------------------------------------
# Dev implementations (Phase 11 adds the Keycloak ones)
# ---------------------------------------------------------------------------
def _extract_bearer(headers: dict[str, str]) -> str | None:
    for key, value in headers.items():
        if key.lower() == "authorization":
            parts = value.split(None, 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1].strip()
            return None
    return None


class SharedSecretServiceAuthenticator:
    """Validates a bearer token against a shared secret loaded from a file.

    A stand-in for real client credentials that is nonetheless genuinely enforced: a caller
    without the secret is rejected. The comparison is constant-time to avoid leaking the
    secret through timing. Phase 11 replaces this with Keycloak JWKS validation.
    """

    def __init__(self, secret: SecretStr, *, service_subject: str = "voyagemesh-service") -> None:
        self._secret = secret
        self._subject = service_subject

    async def authenticate(self, headers: dict[str, str]) -> ServiceIdentity:
        presented = _extract_bearer(headers)
        if presented is None:
            raise ServiceAuthError("missing or malformed Authorization: Bearer header")
        # Constant-time compare so a wrong token cannot be discovered byte-by-byte.
        if not hmac.compare_digest(presented, self._secret.get_secret_value()):
            raise ServiceAuthError("service credential rejected")
        return ServiceIdentity(subject=self._subject, roles=frozenset({INVOKE_ROLE}))

    @property
    def scheme_name(self) -> str:
        return "shared_secret"


class InsecureServiceAuthenticator:
    """Allows every caller. Development only.

    Exists so the stack can run without configured service credentials during early
    development, but it is never silent: it logs a warning on every construction, and the
    A2A server refuses to use it when the environment is production.
    """

    def __init__(self) -> None:
        logger.warning(
            "insecure_service_auth_enabled",
            extra={"detail": "A2A calls are NOT authenticated. Development only."},
        )

    async def authenticate(self, headers: dict[str, str]) -> ServiceIdentity:
        return ServiceIdentity(subject="insecure-dev-caller", roles=frozenset({INVOKE_ROLE}))

    @property
    def scheme_name(self) -> str:
        return "insecure_none"


class StaticServiceTokenProvider:
    """Presents a fixed bearer token (the shared secret) on outbound calls."""

    def __init__(self, secret: SecretStr | None) -> None:
        self._secret = secret

    async def authorization_header(self) -> dict[str, str]:
        if self._secret is None:
            return {}
        return {"Authorization": f"Bearer {self._secret.get_secret_value()}"}
