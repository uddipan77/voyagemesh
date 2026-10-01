"""Centralised, environment-driven configuration.

Each concern is an independent ``BaseSettings`` subclass with its own ``env_prefix``, so
environment variable names stay flat and predictable (``GROQ_MODEL``, ``REDIS_URL``,
``A2A_TRANSPORT_AGENT_URL``) rather than deeply nested. :class:`Settings` composes them.

Nothing here reads a secret *value* from the environment: secrets are referenced by file
path (``*_FILE``) and loaded via :mod:`vm_config.secrets`. This keeps secrets out of
``docker inspect``, process listings, and crash dumps.
"""

from __future__ import annotations

import functools
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from vm_config.secrets import read_secret_file, try_read_secret_file

# pydantic-settings JSON-decodes complex types straight from the environment, which makes
# `FOO=a,b` a hard SettingsError before any validator runs. `NoDecode` hands us the raw
# string so `_split_csv` can accept both `a,b` (Compose-friendly) and `["a","b"]` (JSON).
CsvList = Annotated[list[str], NoDecode]


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            import json

            try:
                return json.loads(text)
            except ValueError:
                return [text]
        return [part.strip() for part in text.split(",") if part.strip()]
    return value


__all__ = [
    "A2ASettings",
    "AuthSettings",
    "CacheTTLSettings",
    "DatabaseSettings",
    "Environment",
    "GuardrailSettings",
    "LLMSettings",
    "LimitSettings",
    "MCPSettings",
    "ProviderMode",
    "ProviderSettings",
    "RedisSettings",
    "ResilienceSettings",
    "Settings",
    "TelemetrySettings",
    "get_settings",
]


class Environment(StrEnum):
    """Deployment profile. Controls which unsafe developer shortcuts are permitted."""

    LOCAL = "local"
    TEST = "test"
    PRODUCTION = "production"


class ProviderMode(StrEnum):
    """How external travel data is sourced."""

    MOCK = "mock"
    """Deterministic in-repo fixtures only. Fully offline and reproducible."""

    LIVE = "live"
    """Use external APIs. Missing credentials or failed sources return unavailable."""

    AUTO = "auto"
    """Use configured external APIs and free APIs; never substitute mock prices."""


class _Base(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
class LLMSettings(_Base):
    """Groq / LLM provider configuration.

    The model is chosen by configuration, never hard-coded at a call site.
    """

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    provider: Literal["groq", "mock"] = Field(default="groq", alias="LLM_PROVIDER")
    timeout_seconds: float = Field(default=30.0, gt=0, le=300, alias="LLM_TIMEOUT_SECONDS")
    max_retries: int = Field(default=2, ge=0, le=5, alias="LLM_MAX_RETRIES")
    max_calls_per_agent: int = Field(default=6, ge=1, le=50, alias="LLM_MAX_CALLS_PER_AGENT")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, alias="LLM_TEMPERATURE")
    max_output_tokens: int = Field(default=2048, ge=64, le=32768, alias="LLM_MAX_OUTPUT_TOKENS")

    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_judge_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_JUDGE_MODEL")
    groq_api_key_file: str | None = Field(default=None, alias="GROQ_API_KEY_FILE")
    groq_base_url: str = Field(default="https://api.groq.com", alias="GROQ_BASE_URL")

    # Cost accounting (USD per 1M tokens). Configurable because vendor pricing changes;
    # used only for local cost *estimation* dashboards, never for billing.
    cost_per_1m_input_usd: float = Field(default=0.59, ge=0, alias="LLM_COST_PER_1M_INPUT_USD")
    cost_per_1m_output_usd: float = Field(default=0.79, ge=0, alias="LLM_COST_PER_1M_OUTPUT_USD")

    def load_api_key(self) -> SecretStr:
        """Load the Groq API key from disk. Raises ``SecretLoadError`` if unavailable."""
        if not self.groq_api_key_file:
            from vm_config.secrets import SecretLoadError

            raise SecretLoadError(
                "groq api key: GROQ_API_KEY_FILE is not set. Point it at a file containing "
                "only the key (host default: apikeys/api.key, container: "
                "/run/secrets/groq_api_key)."
            )
        return read_secret_file(self.groq_api_key_file, name="groq api key")

    def api_key_available(self) -> bool:
        """True when a Groq key can actually be loaded — used to decide mock fallback."""
        return try_read_secret_file(self.groq_api_key_file, name="groq api key") is not None


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
class AuthSettings(_Base):
    """Keycloak OIDC settings for user and service-to-service authentication."""

    model_config = SettingsConfigDict(env_prefix="AUTH_", extra="ignore")

    issuer: str = Field(default="http://localhost:8080/realms/voyagemesh")
    """Expected ``iss`` claim. Tokens from any other issuer are rejected."""

    jwks_url: str | None = Field(default=None)
    """Overrides the issuer-derived JWKS URL (needed when the container-internal Keycloak
    hostname differs from the browser-facing issuer)."""

    audience: str = Field(default="voyagemesh-api")
    algorithms: CsvList = Field(default_factory=lambda: ["RS256"])
    jwks_cache_seconds: int = Field(default=300, ge=0)
    leeway_seconds: int = Field(default=10, ge=0, le=120)

    service_client_id: str = Field(default="voyagemesh-service")
    service_client_secret_file: str | None = Field(default=None)
    token_url: str | None = Field(default=None)

    # How inbound A2A calls are authenticated: 'shared_secret' (dev, Phase 7) or 'keycloak'
    # (production client-credentials, Phase 11).
    service_auth: Literal["shared_secret", "keycloak"] = Field(default="shared_secret")

    dev_insecure_allow_unauthenticated: bool = Field(default=False)
    """Development-only bypass. Guarded by :meth:`Settings.validate_safety` — it is
    rejected outright when ``ENVIRONMENT=production`` and logs a loud warning otherwise."""

    @property
    def resolved_jwks_url(self) -> str:
        return self.jwks_url or f"{self.issuer.rstrip('/')}/protocol/openid-connect/certs"

    @property
    def resolved_token_url(self) -> str:
        return self.token_url or f"{self.issuer.rstrip('/')}/protocol/openid-connect/token"

    _split_algorithms = field_validator("algorithms", mode="before")(_split_csv)

    def load_client_secret(self) -> SecretStr | None:
        return try_read_secret_file(self.service_client_secret_file, name="service client secret")


# ---------------------------------------------------------------------------
# Data stores
# ---------------------------------------------------------------------------
class RedisSettings(_Base):
    model_config = SettingsConfigDict(env_prefix="REDIS_", extra="ignore")

    url: str = Field(default="redis://localhost:6379/0")
    enabled: bool = Field(default=True)
    socket_timeout_seconds: float = Field(default=2.0, gt=0)
    connect_timeout_seconds: float = Field(default=2.0, gt=0)
    max_connections: int = Field(default=20, ge=1)
    key_prefix: str = Field(default="vm")


class CacheTTLSettings(_Base):
    """Per-domain cache lifetimes, in seconds.

    Volatile data (prices, schedules) expires quickly; stable data (points of interest)
    lives for days. The final plan inherits the shortest TTL of its inputs.
    """

    model_config = SettingsConfigDict(env_prefix="CACHE_TTL_", extra="ignore")

    transport_seconds: int = Field(default=900, ge=0)  # 15 min
    accommodation_seconds: int = Field(default=900, ge=0)  # 15 min
    weather_seconds: int = Field(default=7200, ge=0)  # 2 h
    poi_seconds: int = Field(default=259200, ge=0)  # 3 d
    plan_seconds: int = Field(default=900, ge=0)
    idempotency_seconds: int = Field(default=900, ge=0)  # 15 min — an idempotency-key window
    stale_grace_seconds: int = Field(default=300, ge=0)
    """Extra window during which a stale entry may be served while it is refreshed."""


class DatabaseSettings(_Base):
    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")

    url: str = Field(default="postgresql+asyncpg://voyagemesh:voyagemesh@localhost:5432/voyagemesh")
    enabled: bool = Field(default=True)
    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=10, ge=0)
    pool_timeout_seconds: float = Field(default=10.0, gt=0)
    statement_timeout_ms: int = Field(default=10_000, ge=100)
    echo: bool = Field(default=False)
    embedding_dimensions: int = Field(default=384, ge=8, le=4096)

    @field_validator("url")
    @classmethod
    def _require_async_driver(cls, v: str) -> str:
        if v.startswith("postgresql://"):
            # asyncpg is required by the async SQLAlchemy engine; silently accepting the
            # sync URL produces a confusing runtime error much later.
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v


# ---------------------------------------------------------------------------
# Service topology
# ---------------------------------------------------------------------------
class A2ASettings(_Base):
    """Locations of the specialist agents reachable over the A2A protocol."""

    model_config = SettingsConfigDict(env_prefix="A2A_", extra="ignore")

    transport_agent_url: str = Field(default="http://localhost:8010")
    stay_agent_url: str = Field(default="http://localhost:8020")
    itinerary_agent_url: str = Field(default="http://localhost:8030")

    timeout_seconds: float = Field(default=20.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    max_response_bytes: int = Field(default=2_000_000, ge=1024)


class MCPSettings(_Base):
    """Locations of the per-agent MCP servers."""

    model_config = SettingsConfigDict(env_prefix="MCP_", extra="ignore")

    transport_url: str = Field(default="http://localhost:8110")
    lodging_url: str = Field(default="http://localhost:8120")
    destination_url: str = Field(default="http://localhost:8130")

    timeout_seconds: float = Field(default=10.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    max_response_bytes: int = Field(default=1_000_000, ge=1024)
    max_tool_calls_per_task: int = Field(default=8, ge=1, le=50)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class ProviderSettings(_Base):
    """External travel-data provider policy."""

    model_config = SettingsConfigDict(env_prefix="PROVIDER_", extra="ignore")

    mode: ProviderMode = Field(default=ProviderMode.AUTO)
    http_timeout_seconds: float = Field(default=8.0, gt=0)
    max_response_bytes: int = Field(default=2_000_000, ge=1024)
    user_agent: str = Field(default="VoyageMesh/0.1 (local portfolio project)")

    allowed_hosts: CsvList = Field(
        default_factory=lambda: [
            "api.open-meteo.com",
            "geocoding-api.open-meteo.com",
            "api.opentripmap.com",
            "nominatim.openstreetmap.org",
            "api.duffel.com",
            "api.liteapi.travel",
            "api.geoapify.com",
        ]
    )
    """SSRF allowlist. Any outbound provider request to a host outside this list is
    refused by the safe HTTP client before a socket is opened."""

    _split_allowed_hosts = field_validator("allowed_hosts", mode="before")(_split_csv)

    opentripmap_api_key_file: str | None = Field(default=None)
    duffel_api_key_file: str | None = None
    liteapi_api_key_file: str | None = None
    geoapify_api_key_file: str | None = None
    search_timeout_seconds: int = Field(default=6, ge=2, le=12)
    """Provider-side search deadline; keep below HTTP/MCP/task timeouts."""

    def load_opentripmap_key(self) -> SecretStr | None:
        return try_read_secret_file(self.opentripmap_api_key_file, name="opentripmap key")


# ---------------------------------------------------------------------------
# Resilience & limits
# ---------------------------------------------------------------------------
class ResilienceSettings(_Base):
    model_config = SettingsConfigDict(env_prefix="RESILIENCE_", extra="ignore")

    retry_base_delay_seconds: float = Field(default=0.2, gt=0)
    retry_max_delay_seconds: float = Field(default=4.0, gt=0)
    retry_jitter: float = Field(default=0.1, ge=0)

    breaker_failure_threshold: int = Field(default=5, ge=1)
    breaker_reset_timeout_seconds: float = Field(default=30.0, gt=0)
    breaker_half_open_max_calls: int = Field(default=2, ge=1)

    bulkhead_max_concurrency: int = Field(default=16, ge=1)


class LimitSettings(_Base):
    """Hard stopping conditions. These exist to guarantee termination."""

    model_config = SettingsConfigDict(env_prefix="LIMIT_", extra="ignore")

    max_replans: int = Field(default=3, ge=0, le=10)
    max_tool_calls_per_agent: int = Field(default=8, ge=1, le=50)
    max_request_duration_seconds: float = Field(default=45.0, gt=0, le=600)
    max_agent_task_duration_seconds: float = Field(default=25.0, gt=0)
    max_request_bytes: int = Field(default=64_000, ge=1024)
    max_interests: int = Field(default=12, ge=1)
    max_travellers: int = Field(default=12, ge=1)
    max_trip_nights: int = Field(default=30, ge=1)

    rate_limit_requests_per_minute: int = Field(default=30, ge=1)
    rate_limit_burst: int = Field(default=10, ge=1)


class GuardrailSettings(_Base):
    model_config = SettingsConfigDict(env_prefix="GUARDRAIL_", extra="ignore")

    enable_input_guardrails: bool = Field(default=True)
    enable_output_guardrails: bool = Field(default=True)
    enable_prompt_injection_detection: bool = Field(default=True)
    budget_tolerance_ratio: float = Field(default=0.0, ge=0.0, le=0.25)
    """Fraction by which a plan may exceed the stated budget before it is treated as a
    constraint violation. Defaults to zero — the user's budget is a hard limit."""

    max_free_text_length: int = Field(default=2000, ge=10)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
class TelemetrySettings(_Base):
    model_config = SettingsConfigDict(env_prefix="OTEL_", extra="ignore")

    enabled: bool = Field(default=True, alias="OTEL_ENABLED")
    exporter_otlp_endpoint: str = Field(
        default="http://localhost:4318", alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    traces_sampler_ratio: float = Field(default=1.0, ge=0.0, le=1.0, alias="OTEL_SAMPLER_RATIO")
    console_export: bool = Field(default=False, alias="OTEL_CONSOLE_EXPORT")
    metrics_enabled: bool = Field(default=True, alias="OTEL_METRICS_ENABLED")


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
class Settings(_Base):
    """Root configuration object. Obtain via :func:`get_settings`."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore", env_file=".env")

    service_name: str = Field(default="voyagemesh", alias="SERVICE_NAME")
    service_version: str = Field(default="0.1.0", alias="SERVICE_VERSION")
    environment: Environment = Field(default=Environment.LOCAL, alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: Literal["json", "console"] = Field(default="json", alias="LOG_FORMAT")
    # Containers must bind all interfaces to be reachable across the Compose network.
    # Exposure is controlled by Compose port publishing, not by the bind address.
    host: str = Field(default="0.0.0.0", alias="HOST")  # noqa: S104  # nosec B104
    port: int = Field(default=8000, ge=1, le=65535, alias="PORT")

    cors_allowed_origins: CsvList = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"],
        alias="CORS_ALLOWED_ORIGINS",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    cache_ttl: CacheTTLSettings = Field(default_factory=CacheTTLSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    a2a: A2ASettings = Field(default_factory=A2ASettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)
    limits: LimitSettings = Field(default_factory=LimitSettings)
    guardrails: GuardrailSettings = Field(default_factory=GuardrailSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)

    _split_origins = field_validator("cors_allowed_origins", mode="before")(_split_csv)

    @model_validator(mode="after")
    def validate_safety(self) -> Settings:
        """Refuse configurations that would be unsafe outside development."""
        if self.environment is Environment.PRODUCTION:
            if self.auth.dev_insecure_allow_unauthenticated:
                raise ValueError(
                    "AUTH_DEV_INSECURE_ALLOW_UNAUTHENTICATED=true is not permitted when "
                    "ENVIRONMENT=production. This flag disables all request authentication."
                )
            if "*" in self.cors_allowed_origins:
                raise ValueError(
                    "CORS_ALLOWED_ORIGINS must not be '*' when ENVIRONMENT=production."
                )
        return self

    @classmethod
    def for_testing(cls, **overrides: object) -> Settings:
        """Build settings for programmatic use, routing flat overrides to nested groups.

        There is a genuine footgun this exists to close: ``Settings(PROVIDER_MODE="mock")``
        **silently does nothing**. Each nested settings group (``providers``, ``redis``, …)
        is created by its own ``default_factory`` that reads ``os.environ`` independently,
        so a keyword passed to the *root* never reaches it. Under Docker this is invisible
        because real environment variables reach every group correctly — but a test or
        script that constructs ``Settings`` with kwargs would think it had disabled live
        mode and would not have.

        This helper accepts the same flat, prefixed names the environment uses
        (``PROVIDER_MODE``, ``REDIS_ENABLED``, root fields like ``ENVIRONMENT``) and applies
        each to the correct group, so programmatic construction matches env-var behaviour.
        """
        prefixed = {
            "providers": ("PROVIDER_", ProviderSettings),
            "redis": ("REDIS_", RedisSettings),
            "database": ("DB_", DatabaseSettings),
            "llm": ("", LLMSettings),
            "auth": ("AUTH_", AuthSettings),
            "a2a": ("A2A_", A2ASettings),
            "mcp": ("MCP_", MCPSettings),
            "limits": ("LIMIT_", LimitSettings),
            "guardrails": ("GUARDRAIL_", GuardrailSettings),
            "cache_ttl": ("CACHE_TTL_", CacheTTLSettings),
            "resilience": ("RESILIENCE_", ResilienceSettings),
            "telemetry": ("OTEL_", TelemetrySettings),
        }
        group_kwargs: dict[str, dict[str, Any]] = {name: {} for name in prefixed}
        root_kwargs: dict[str, Any] = {}

        for key, value in overrides.items():
            for group, (prefix, _model) in prefixed.items():
                if prefix and key.upper().startswith(prefix):
                    group_kwargs[group][key.upper().removeprefix(prefix).lower()] = value
                    break
            else:
                root_kwargs[key] = value

        built: dict[str, Any] = {"_env_file": None}
        for group, (_prefix, model) in prefixed.items():
            if group_kwargs[group]:
                built[group] = model(_env_file=None, **group_kwargs[group])
        built.update(root_kwargs)
        return cls(**built)

    @property
    def is_test(self) -> bool:
        return self.environment is Environment.TEST

    @property
    def auth_bypass_active(self) -> bool:
        """True when unauthenticated requests are accepted. Always surfaced in logs."""
        return self.auth.dev_insecure_allow_unauthenticated


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that every module observes identical configuration. Tests clear the cache
    via ``get_settings.cache_clear()`` after patching the environment.
    """
    return Settings()
