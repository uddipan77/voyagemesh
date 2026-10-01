"""JWT validation security (threat T-6): every way a bad token must be rejected.

Uses a self-signed RS256 keypair and a hand-built JWKS, so the full validation path — issuer,
audience, expiry, algorithm pinning, signature — is exercised without a running Keycloak.
"""

from __future__ import annotations

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from vm_auth import (
    JwksClient,
    JwtValidator,
    KeycloakServiceAuthenticator,
    Role,
    ServiceAuthError,
    TokenError,
    TokenErrorCode,
)
from vm_config.settings import AuthSettings

pytestmark = [pytest.mark.security, pytest.mark.unit]

ISSUER = "http://keycloak:8080/realms/voyagemesh"
AUDIENCE = "voyagemesh-api"
KID = "test-key-1"


@pytest.fixture(scope="module")
def keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key


@pytest.fixture(scope="module")
def jwk(keypair):
    """The public key as a JWK, as Keycloak would publish it."""
    from jwt.algorithms import RSAAlgorithm

    public_jwk = RSAAlgorithm.to_jwk(keypair.public_key(), as_dict=True)
    public_jwk["kid"] = KID
    public_jwk["alg"] = "RS256"
    return public_jwk


@pytest.fixture
def validator(jwk):
    settings = AuthSettings(_env_file=None, issuer=ISSUER, audience=AUDIENCE)
    return JwtValidator(settings, jwks_client=JwksClient("unused", static_keys=[jwk]))


def _token(
    keypair,
    *,
    issuer: str = ISSUER,
    audience: str = AUDIENCE,
    subject: str = "user-123",
    exp_offset: int = 3600,
    roles: list[str] | None = None,
    kid: str | None = KID,
    algorithm: str = "RS256",
    key=None,
    extra: dict | None = None,
) -> str:
    now = int(time.time())
    claims = {
        "sub": subject,
        "iss": issuer,
        "aud": audience,
        "iat": now,
        "exp": now + exp_offset,
        "preferred_username": "alice",
    }
    if roles is not None:
        claims["realm_access"] = {"roles": roles}
    if extra:
        claims.update(extra)
    headers = {"kid": kid} if kid else {}
    signing_key = key if key is not None else keypair
    return jwt.encode(claims, signing_key, algorithm=algorithm, headers=headers)


class TestValidTokens:
    async def test_a_valid_token_is_accepted(self, validator, keypair):
        user = await validator.validate(_token(keypair, roles=["traveller"]))
        assert user.subject == "user-123"
        assert user.username == "alice"
        assert user.has_role(Role.TRAVELLER)

    async def test_client_roles_are_extracted(self, validator, keypair):
        token = _token(
            keypair,
            extra={"resource_access": {AUDIENCE: {"roles": ["evaluator"]}}},
        )
        user = await validator.validate(token)
        assert user.has_role(Role.EVALUATOR)

    async def test_reference_is_the_subject_not_an_email(self, validator, keypair):
        user = await validator.validate(_token(keypair))
        assert user.reference == "user-123"
        assert "@" not in user.reference


class TestRejectedTokens:
    async def test_missing_token_is_rejected(self, validator):
        with pytest.raises(TokenError) as exc:
            await validator.validate("")
        assert exc.value.code is TokenErrorCode.MISSING

    async def test_expired_token_is_rejected(self, validator, keypair):
        with pytest.raises(TokenError) as exc:
            await validator.validate(_token(keypair, exp_offset=-3600))
        assert exc.value.code is TokenErrorCode.EXPIRED

    async def test_wrong_issuer_is_rejected(self, validator, keypair):
        with pytest.raises(TokenError) as exc:
            await validator.validate(_token(keypair, issuer="http://evil/realms/x"))
        assert exc.value.code is TokenErrorCode.INVALID

    async def test_wrong_audience_is_rejected(self, validator, keypair):
        with pytest.raises(TokenError) as exc:
            await validator.validate(_token(keypair, audience="some-other-service"))
        assert exc.value.code is TokenErrorCode.INVALID

    async def test_alg_none_is_rejected(self, validator):
        """The single most important JWT attack: an unsigned token must never validate."""
        now = int(time.time())
        unsigned = jwt.encode(
            {"sub": "attacker", "iss": ISSUER, "aud": AUDIENCE, "exp": now + 3600},
            key="",
            algorithm="none",
            headers={"kid": KID},
        )
        with pytest.raises(TokenError) as exc:
            await validator.validate(unsigned)
        assert exc.value.code is TokenErrorCode.INVALID

    async def test_token_signed_by_a_different_key_is_rejected(self, validator):
        """HS256-confusion / wrong-key: a token signed by anything but the trusted key fails."""
        attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        forged = _token(None, key=attacker_key, roles=["admin"])
        with pytest.raises(TokenError) as exc:
            await validator.validate(forged)
        assert exc.value.code is TokenErrorCode.INVALID

    async def test_unknown_kid_is_rejected(self, validator, keypair):
        with pytest.raises(TokenError) as exc:
            await validator.validate(_token(keypair, kid="some-other-kid"))
        assert exc.value.code is TokenErrorCode.INVALID

    async def test_token_with_no_kid_is_rejected(self, validator, keypair):
        with pytest.raises(TokenError) as exc:
            await validator.validate(_token(keypair, kid=None))
        assert exc.value.code is TokenErrorCode.INVALID

    async def test_garbage_is_rejected(self, validator):
        with pytest.raises(TokenError):
            await validator.validate("not.a.jwt")

    async def test_a_tampered_payload_is_rejected(self, validator, keypair):
        """Changing a claim after signing breaks the signature."""
        token = _token(keypair, roles=["traveller"])
        header, payload, signature = token.split(".")
        import base64
        import json

        decoded = json.loads(base64.urlsafe_b64decode(payload + "=="))
        decoded["realm_access"] = {"roles": ["admin"]}  # privilege escalation attempt
        tampered_payload = base64.urlsafe_b64encode(json.dumps(decoded).encode()).rstrip(b"=")
        tampered = f"{header}.{tampered_payload.decode()}.{signature}"
        with pytest.raises(TokenError):
            await validator.validate(tampered)

    async def test_error_messages_never_contain_the_token(self, validator, keypair):
        token = _token(keypair, exp_offset=-3600, subject="SENSITIVE-SUBJECT")
        with pytest.raises(TokenError) as exc:
            await validator.validate(token)
        assert token not in str(exc.value)


class TestRoleChecks:
    async def test_require_role_passes_for_a_held_role(self, validator, keypair):
        user = await validator.validate(_token(keypair, roles=["admin"]))
        user.require_role(Role.ADMIN)  # no exception

    async def test_require_role_raises_for_a_missing_role(self, validator, keypair):
        user = await validator.validate(_token(keypair, roles=["traveller"]))
        with pytest.raises(TokenError) as exc:
            user.require_role(Role.ADMIN)
        assert exc.value.code is TokenErrorCode.INSUFFICIENT_ROLE


class TestKeycloakServiceAuth:
    """The Keycloak service authenticator reuses JWT validation and requires the invoke role."""

    @pytest.fixture
    def authenticator(self, jwk):
        settings = AuthSettings(_env_file=None, issuer=ISSUER, audience=AUDIENCE)
        return KeycloakServiceAuthenticator(
            settings, jwks_client=JwksClient("unused", static_keys=[jwk])
        )

    async def test_valid_service_token_authenticates(self, authenticator, keypair):
        token = _token(keypair, subject="voyagemesh-orchestrator", roles=["agent:invoke"])
        identity = await authenticator.authenticate({"authorization": f"Bearer {token}"})
        assert identity.can_invoke_agents
        assert identity.subject == "voyagemesh-orchestrator"

    async def test_missing_token_is_rejected(self, authenticator):
        with pytest.raises(ServiceAuthError):
            await authenticator.authenticate({})

    async def test_expired_service_token_is_rejected(self, authenticator, keypair):
        token = _token(keypair, exp_offset=-10, roles=["agent:invoke"])
        with pytest.raises(ServiceAuthError, match="rejected"):
            await authenticator.authenticate({"authorization": f"Bearer {token}"})

    async def test_service_auth_error_does_not_leak_the_token(self, authenticator, keypair):
        token = _token(keypair, issuer="http://evil/x")
        with pytest.raises(ServiceAuthError) as exc:
            await authenticator.authenticate({"authorization": f"Bearer {token}"})
        assert token not in str(exc.value)
