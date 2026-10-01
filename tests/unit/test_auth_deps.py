"""FastAPI auth dependencies and service-authenticator selection."""

from __future__ import annotations

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient

from vm_auth import (
    AuthDependencies,
    JwksClient,
    JwtValidator,
    KeycloakServiceAuthenticator,
    Role,
)
from vm_auth.service_auth import InsecureServiceAuthenticator, SharedSecretServiceAuthenticator
from vm_config.settings import AuthSettings, Settings
from vm_harness.service_factory import select_service_authenticator

pytestmark = pytest.mark.unit

ISSUER = "http://keycloak:8080/realms/voyagemesh"
AUDIENCE = "voyagemesh-api"
KID = "k1"


@pytest.fixture(scope="module")
def keypair():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def jwk(keypair):
    from jwt.algorithms import RSAAlgorithm

    j = RSAAlgorithm.to_jwk(keypair.public_key(), as_dict=True)
    j["kid"] = KID
    return j


def _token(keypair, *, roles, sub="user-1"):
    now = int(time.time())
    return jwt.encode(
        {
            "sub": sub,
            "iss": ISSUER,
            "aud": AUDIENCE,
            "exp": now + 3600,
            "iat": now,
            "preferred_username": "alice",
            "realm_access": {"roles": roles},
        },
        keypair,
        algorithm="RS256",
        headers={"kid": KID},
    )


def _app(settings: Settings, jwk) -> FastAPI:
    validator = JwtValidator(settings.auth, jwks_client=JwksClient("x", static_keys=[jwk]))
    deps = AuthDependencies(settings, validator=validator)
    app = FastAPI()

    @app.get("/me")
    async def me(user=Depends(deps.require_user)):
        return {"subject": user.subject, "roles": sorted(user.roles)}

    @app.get("/admin")
    async def admin(user=Depends(deps.require_role(Role.ADMIN))):
        return {"ok": True}

    return app


async def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestRequireUser:
    async def test_valid_token_authenticates(self, jwk, keypair):
        settings = Settings.for_testing(
            AUTH_ISSUER=ISSUER, AUTH_AUDIENCE=AUDIENCE, ENVIRONMENT="test"
        )
        http = await _client(_app(settings, jwk))
        try:
            token = _token(keypair, roles=["traveller"])
            r = await http.get("/me", headers={"Authorization": f"Bearer {token}"})
            assert r.status_code == 200
            assert r.json()["subject"] == "user-1"
        finally:
            await http.aclose()

    async def test_missing_token_is_401(self, jwk, keypair):
        settings = Settings.for_testing(AUTH_ISSUER=ISSUER, AUTH_AUDIENCE=AUDIENCE)
        http = await _client(_app(settings, jwk))
        try:
            r = await http.get("/me")
            assert r.status_code == 401
            assert r.json()["detail"]["code"] == "unauthenticated"
        finally:
            await http.aclose()

    async def test_role_gate_forbids_without_the_role(self, jwk, keypair):
        settings = Settings.for_testing(AUTH_ISSUER=ISSUER, AUTH_AUDIENCE=AUDIENCE)
        http = await _client(_app(settings, jwk))
        try:
            token = _token(keypair, roles=["traveller"])
            r = await http.get("/admin", headers={"Authorization": f"Bearer {token}"})
            assert r.status_code == 403
            assert r.json()["detail"]["code"] == "insufficient_role"
        finally:
            await http.aclose()

    async def test_role_gate_allows_with_the_role(self, jwk, keypair):
        settings = Settings.for_testing(AUTH_ISSUER=ISSUER, AUTH_AUDIENCE=AUDIENCE)
        http = await _client(_app(settings, jwk))
        try:
            token = _token(keypair, roles=["admin"])
            r = await http.get("/admin", headers={"Authorization": f"Bearer {token}"})
            assert r.status_code == 200
        finally:
            await http.aclose()


class TestDevBypass:
    async def test_bypass_injects_a_synthetic_user(self, jwk, keypair):
        settings = Settings.for_testing(
            ENVIRONMENT="local", AUTH_DEV_INSECURE_ALLOW_UNAUTHENTICATED="true"
        )
        http = await _client(_app(settings, jwk))
        try:
            r = await http.get("/me")  # no token
            assert r.status_code == 200
            assert r.json()["subject"] == "dev-insecure-user"
        finally:
            await http.aclose()

    def test_bypass_is_refused_in_production(self):
        with pytest.raises(ValueError, match="not permitted when ENVIRONMENT=production"):
            Settings.for_testing(
                ENVIRONMENT="production", AUTH_DEV_INSECURE_ALLOW_UNAUTHENTICATED="true"
            )


class TestServiceAuthSelection:
    def test_keycloak_is_selected_when_configured(self):
        settings = Settings.for_testing(ENVIRONMENT="test", AUTH_SERVICE_AUTH="keycloak")
        assert isinstance(select_service_authenticator(settings), KeycloakServiceAuthenticator)

    def test_shared_secret_when_a_secret_is_configured(self, tmp_path):
        secret = tmp_path / "s.secret"
        secret.write_text("dev-secret", encoding="utf-8")
        settings = Settings.for_testing(
            ENVIRONMENT="local", AUTH_SERVICE_CLIENT_SECRET_FILE=str(secret)
        )
        assert isinstance(select_service_authenticator(settings), SharedSecretServiceAuthenticator)

    def test_insecure_only_outside_production(self):
        settings = Settings.for_testing(ENVIRONMENT="local")
        assert isinstance(select_service_authenticator(settings), InsecureServiceAuthenticator)

    def test_production_without_auth_refuses_to_start(self):
        settings = Settings.for_testing(ENVIRONMENT="production")
        with pytest.raises(RuntimeError, match="refusing to start unauthenticated"):
            select_service_authenticator(settings)


class TestKeycloakServiceRoundTrip:
    async def test_service_token_authenticates_with_invoke_role(self, jwk, keypair):
        settings = AuthSettings(_env_file=None, issuer=ISSUER, audience=AUDIENCE)
        auth = KeycloakServiceAuthenticator(
            settings, jwks_client=JwksClient("x", static_keys=[jwk])
        )
        token = _token(keypair, roles=["agent:invoke"], sub="orchestrator")
        identity = await auth.authenticate({"authorization": f"Bearer {token}"})
        assert identity.can_invoke_agents
