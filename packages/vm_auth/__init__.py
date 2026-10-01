"""Authentication: user (Keycloak OIDC, Phase 11) and service-to-service (A2A).

Phase 7 provides the service-auth *hook* — the `ServiceAuthenticator` protocol and dev
implementations — so agents enforce a verifiable caller identity and forward only minimal
claims across the A2A boundary (least privilege, threat T-4). Phase 11 adds the Keycloak
implementations behind the same interfaces.
"""

from vm_auth.fastapi_deps import AuthDependencies
from vm_auth.jwks import JwksClient, JwksError
from vm_auth.keycloak_service import KeycloakServiceAuthenticator, KeycloakTokenProvider
from vm_auth.service_auth import (
    INVOKE_ROLE,
    InsecureServiceAuthenticator,
    PropagatedClaims,
    ServiceAuthenticator,
    ServiceAuthError,
    ServiceIdentity,
    ServiceTokenProvider,
    SharedSecretServiceAuthenticator,
    StaticServiceTokenProvider,
)
from vm_auth.user_auth import (
    AuthenticatedUser,
    JwtValidator,
    Role,
    TokenError,
    TokenErrorCode,
)

__all__ = [
    "INVOKE_ROLE",
    "AuthDependencies",
    "AuthenticatedUser",
    "InsecureServiceAuthenticator",
    "JwksClient",
    "JwksError",
    "JwtValidator",
    "KeycloakServiceAuthenticator",
    "KeycloakTokenProvider",
    "PropagatedClaims",
    "Role",
    "ServiceAuthError",
    "ServiceAuthenticator",
    "ServiceIdentity",
    "ServiceTokenProvider",
    "SharedSecretServiceAuthenticator",
    "StaticServiceTokenProvider",
    "TokenError",
    "TokenErrorCode",
]
