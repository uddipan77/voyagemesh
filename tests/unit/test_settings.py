"""Configuration precedence, coercion, and safety guards."""

from __future__ import annotations

import pytest

from vm_config.settings import (
    Environment,
    LLMSettings,
    ProviderMode,
    Settings,
    get_settings,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate each test from the developer's real environment and .env file."""
    for var in (
        "LLM_PROVIDER",
        "GROQ_MODEL",
        "GROQ_API_KEY_FILE",
        "ENVIRONMENT",
        "CORS_ALLOWED_ORIGINS",
        "AUTH_DEV_INSECURE_ALLOW_UNAUTHENTICATED",
        "DB_URL",
        "PROVIDER_MODE",
        "LIMIT_MAX_REPLANS",
    ):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _settings() -> Settings:
    # _env_file=None ignores any developer .env so defaults are deterministic in CI.
    return Settings(_env_file=None)


def test_defaults_are_sane():
    s = _settings()
    assert s.environment is Environment.LOCAL
    assert s.llm.provider == "groq"
    assert s.limits.max_replans == 3
    assert s.limits.max_tool_calls_per_agent == 8
    assert s.limits.max_request_duration_seconds == 45.0
    assert s.a2a.max_retries == 2
    assert s.mcp.max_retries == 2
    assert s.auth.dev_insecure_allow_unauthenticated is False


def test_model_is_configurable_not_hardcoded(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    assert LLMSettings(_env_file=None).groq_model == "openai/gpt-oss-120b"


def test_env_overrides_apply(monkeypatch):
    monkeypatch.setenv("LIMIT_MAX_REPLANS", "1")
    monkeypatch.setenv("PROVIDER_MODE", "mock")
    s = _settings()
    assert s.limits.max_replans == 1
    assert s.providers.mode is ProviderMode.MOCK


def test_cors_origins_accept_comma_separated_string(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://a.local, http://b.local")
    assert _settings().cors_allowed_origins == ["http://a.local", "http://b.local"]


def test_sync_postgres_url_is_upgraded_to_asyncpg(monkeypatch):
    monkeypatch.setenv("DB_URL", "postgresql://u:p@localhost:5432/db")
    assert _settings().database.url.startswith("postgresql+asyncpg://")


def test_auth_bypass_rejected_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("AUTH_DEV_INSECURE_ALLOW_UNAUTHENTICATED", "true")
    with pytest.raises(ValueError, match="not permitted when ENVIRONMENT=production"):
        _settings()


def test_wildcard_cors_rejected_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
    with pytest.raises(ValueError, match="CORS_ALLOWED_ORIGINS"):
        _settings()


def test_auth_bypass_allowed_but_flagged_in_local(monkeypatch):
    monkeypatch.setenv("AUTH_DEV_INSECURE_ALLOW_UNAUTHENTICATED", "true")
    s = _settings()
    assert s.auth_bypass_active is True


def test_jwks_and_token_urls_derive_from_issuer():
    s = _settings()
    assert s.auth.resolved_jwks_url.endswith("/protocol/openid-connect/certs")
    assert s.auth.resolved_token_url.endswith("/protocol/openid-connect/token")


def test_settings_repr_does_not_expose_secret_paths_content(tmp_path, monkeypatch):
    key = tmp_path / "api.key"
    key.write_text("gsk_secret_value_123", encoding="utf-8")
    monkeypatch.setenv("GROQ_API_KEY_FILE", str(key))
    s = _settings()
    assert s.llm.api_key_available() is True
    assert "gsk_secret_value_123" not in repr(s)


def test_api_key_available_false_when_missing(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY_FILE", "/nonexistent/path/api.key")
    assert _settings().llm.api_key_available() is False


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
