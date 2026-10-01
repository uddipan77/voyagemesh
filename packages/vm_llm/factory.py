"""Provider selection.

One place decides which :class:`~vm_llm.base.LLMProvider` a service uses, so the choice —
and the degradation policy around it — is not re-implemented at each call site.
"""

from __future__ import annotations

import logging

from vm_config.settings import Environment, LLMSettings, Settings
from vm_llm.base import LLMProvider
from vm_llm.mock_provider import MockLLMProvider

__all__ = ["build_llm_provider"]

logger = logging.getLogger(__name__)


def build_llm_provider(
    settings: Settings | None = None,
    *,
    llm_settings: LLMSettings | None = None,
    environment: Environment = Environment.LOCAL,
    allow_mock_fallback: bool | None = None,
) -> LLMProvider:
    """Return the configured provider.

    Args:
        settings: Root settings; supplies both LLM config and environment.
        llm_settings: LLM config, when the root object is not to hand.
        environment: Deployment profile, used only when ``settings`` is omitted.
        allow_mock_fallback: Whether a missing/broken Groq key may silently degrade to the
            mock provider. Defaults to ``True`` outside production.

    Returns:
        A ready provider.

    Raises:
        LLMUnavailableError: Groq is configured, unavailable, and fallback is not allowed.

    Falling back to a mock in production would mean serving synthesised narrative text as
    though a model had produced it, so the fallback is refused there — the service reports
    a dependency failure instead.
    """
    config = llm_settings or (settings.llm if settings else LLMSettings())
    env = settings.environment if settings else environment

    if allow_mock_fallback is None:
        allow_mock_fallback = env is not Environment.PRODUCTION

    if config.provider == "mock":
        logger.info("llm_provider_selected", extra={"provider": "mock", "reason": "configured"})
        return MockLLMProvider()

    from vm_llm.groq_provider import GroqLLMProvider

    try:
        provider = GroqLLMProvider(config)
    except Exception:
        if not allow_mock_fallback:
            raise
        # Log without the exception text: it may name a filesystem path.
        logger.warning(
            "llm_provider_degraded_to_mock",
            extra={"provider": "mock", "reason": "groq_unavailable", "environment": env.value},
        )
        return MockLLMProvider(model=f"{config.groq_model}-MOCK-FALLBACK")

    logger.info("llm_provider_selected", extra={"provider": "groq", "model": provider.model})
    return provider
