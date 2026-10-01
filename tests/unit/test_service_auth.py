"""Service-to-service authentication and claim propagation."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from vm_auth import (
    INVOKE_ROLE,
    InsecureServiceAuthenticator,
    PropagatedClaims,
    ServiceAuthError,
    ServiceIdentity,
    SharedSecretServiceAuthenticator,
    StaticServiceTokenProvider,
)
from vm_contracts.tracing import TraceContext

pytestmark = pytest.mark.unit

SECRET = SecretStr("s3cr3t-service-token")


class TestSharedSecretAuthenticator:
    async def test_accepts_the_correct_bearer_token(self):
        auth = SharedSecretServiceAuthenticator(SECRET)
        identity = await auth.authenticate({"authorization": "Bearer s3cr3t-service-token"})
        assert identity.can_invoke_agents
        assert identity.subject == "voyagemesh-service"

    async def test_rejects_a_wrong_token(self):
        auth = SharedSecretServiceAuthenticator(SECRET)
        with pytest.raises(ServiceAuthError, match="rejected"):
            await auth.authenticate({"authorization": "Bearer wrong"})

    async def test_rejects_a_missing_header(self):
        auth = SharedSecretServiceAuthenticator(SECRET)
        with pytest.raises(ServiceAuthError, match="missing"):
            await auth.authenticate({})

    @pytest.mark.parametrize(
        "value",
        ["Basic abc", "Bearer", "s3cr3t-service-token", "bearer"],  # malformed forms
    )
    async def test_rejects_malformed_authorization(self, value):
        auth = SharedSecretServiceAuthenticator(SECRET)
        with pytest.raises(ServiceAuthError):
            await auth.authenticate({"authorization": value})

    async def test_error_never_contains_the_presented_token(self):
        auth = SharedSecretServiceAuthenticator(SECRET)
        with pytest.raises(ServiceAuthError) as exc:
            await auth.authenticate({"authorization": "Bearer SUPERSECRETGUESS12345"})
        assert "SUPERSECRETGUESS12345" not in str(exc.value)

    async def test_case_insensitive_header_name(self):
        auth = SharedSecretServiceAuthenticator(SECRET)
        identity = await auth.authenticate({"Authorization": "Bearer s3cr3t-service-token"})
        assert identity.can_invoke_agents


class TestInsecureAuthenticator:
    async def test_allows_any_caller(self):
        identity = await InsecureServiceAuthenticator().authenticate({})
        assert identity.can_invoke_agents


class TestServiceIdentity:
    def test_role_checks(self):
        identity = ServiceIdentity(subject="s", roles=frozenset({INVOKE_ROLE, "reader"}))
        assert identity.has_role(INVOKE_ROLE)
        assert identity.can_invoke_agents
        assert not ServiceIdentity(subject="s").can_invoke_agents


class TestPropagatedClaims:
    def test_round_trips_through_headers(self):
        claims = PropagatedClaims(
            user_reference="user-123", roles=("traveller",), request_id="req-1", trace_id="a" * 32
        )
        restored = PropagatedClaims.from_headers(claims.to_headers())
        assert restored == claims

    def test_omits_absent_fields_from_headers(self):
        assert PropagatedClaims().to_headers() == {}

    def test_never_carries_an_access_token(self):
        """Least privilege (threat T-4): only minimal claims cross the boundary."""
        headers = PropagatedClaims(user_reference="user-123", roles=("traveller",)).to_headers()
        joined = " ".join(f"{k}={v}" for k, v in headers.items()).lower()
        assert "authorization" not in joined
        assert "access_token" not in joined
        assert "bearer" not in joined

    def test_derives_from_a_trace_context(self):
        trace = TraceContext.new().model_copy(update={"user_reference": "user-9"})
        claims = PropagatedClaims.from_trace(trace, roles=("evaluator",))
        assert claims.user_reference == "user-9"
        assert claims.request_id == trace.request_id
        assert claims.trace_id == trace.trace_id
        assert claims.roles == ("evaluator",)


class TestTokenProvider:
    async def test_static_provider_presents_the_secret(self):
        header = await StaticServiceTokenProvider(SECRET).authorization_header()
        assert header == {"Authorization": "Bearer s3cr3t-service-token"}

    async def test_static_provider_with_no_secret_presents_nothing(self):
        assert await StaticServiceTokenProvider(None).authorization_header() == {}
