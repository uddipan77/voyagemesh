"""FastAPI authentication dependencies for the API gateway (brief §15A, §20).

Turns the :class:`JwtValidator` into request dependencies the gateway routes declare:
``require_user`` for any authenticated caller, ``require_role`` for role-gated endpoints. A
failure raises an ``HTTPException`` with a sanitised problem-details body — never a stack
trace, never the token.

A development bypass (`AUTH_DEV_INSECURE_ALLOW_UNAUTHENTICATED`) injects a synthetic user so
the stack can run without Keycloak, but it is refused in production by the config validator and
logs a warning banner at startup.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from fastapi import Depends, HTTPException, Request, status

from vm_auth.user_auth import AuthenticatedUser, JwtValidator, Role, TokenError, TokenErrorCode
from vm_config.settings import Settings

__all__ = ["AuthDependencies"]

logger = logging.getLogger(__name__)

_DEV_USER = AuthenticatedUser(
    subject="dev-insecure-user",
    username="dev",
    roles=frozenset({Role.TRAVELLER.value, Role.EVALUATOR.value, Role.ADMIN.value}),
)

_STATUS_BY_CODE = {
    TokenErrorCode.MISSING: status.HTTP_401_UNAUTHORIZED,
    TokenErrorCode.EXPIRED: status.HTTP_401_UNAUTHORIZED,
    TokenErrorCode.INVALID: status.HTTP_401_UNAUTHORIZED,
    TokenErrorCode.INSUFFICIENT_ROLE: status.HTTP_403_FORBIDDEN,
}


class AuthDependencies:
    """Builds request dependencies bound to a validator and settings."""

    def __init__(self, settings: Settings, *, validator: JwtValidator | None = None) -> None:
        self._settings = settings
        self._bypass = settings.auth_bypass_active
        self._validator = validator or JwtValidator(settings.auth)
        if self._bypass:
            logger.warning(
                "auth_bypass_active",
                extra={"detail": "Requests are NOT authenticated. Development only."},
            )

    async def require_user(self, request: Request) -> AuthenticatedUser:
        """Authenticate the request. Raises 401 if no valid token is present."""
        if self._bypass:
            return _DEV_USER

        token = _bearer(request)
        try:
            return await self._validator.validate(token or "")
        except TokenError as exc:
            raise _http_error(exc) from None

    def require_role(self, role: Role | str) -> Callable[[Request], Awaitable[AuthenticatedUser]]:
        """A dependency factory that requires ``role`` in addition to authentication."""

        async def dependency(request: Request) -> AuthenticatedUser:
            user = await self.require_user(request)
            try:
                user.require_role(role)
            except TokenError as exc:
                raise _http_error(exc) from None
            return user

        return dependency

    # A ready-made dependency object for the common "any authenticated user" case.
    @property
    def user(self) -> Callable[..., Awaitable[AuthenticatedUser]]:
        async def dependency(
            request: Request, _self: AuthDependencies = Depends(lambda: self)
        ) -> AuthenticatedUser:
            return await self.require_user(request)

        return dependency


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if not header:
        return None
    parts = header.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def _http_error(exc: TokenError) -> HTTPException:
    return HTTPException(
        status_code=_STATUS_BY_CODE.get(exc.code, status.HTTP_401_UNAUTHORIZED),
        detail={"code": exc.code.value, "message": exc.message},
        headers={"WWW-Authenticate": "Bearer"}
        if exc.code is not TokenErrorCode.INSUFFICIENT_ROLE
        else None,
    )
