"""LLM provider abstraction and structured-output handling.

Every LLM call in VoyageMesh passes through :class:`~vm_llm.base.LLMProvider`. There is no
free-text path: output is always validated into a Pydantic model at the boundary, so the
rest of the codebase only ever handles typed objects (ADR-018).

The provider is chosen by configuration (ADR-008), which is what allows the entire test
suite to run offline and deterministically against :class:`MockLLMProvider`.
"""

from vm_llm.base import (
    UNTRUSTED_DATA_INSTRUCTION,
    LLMProvider,
    build_schema_instruction,
    wrap_untrusted,
)
from vm_llm.factory import build_llm_provider
from vm_llm.json_repair import extract_json, repair_json
from vm_llm.mock_provider import MockLLMProvider
from vm_llm.types import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
    Message,
    Role,
    StructuredOutputError,
    TokenUsage,
)

__all__ = [
    "UNTRUSTED_DATA_INSTRUCTION",
    "LLMAuthenticationError",
    "LLMError",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "Message",
    "MockLLMProvider",
    "Role",
    "StructuredOutputError",
    "TokenUsage",
    "build_llm_provider",
    "build_schema_instruction",
    "extract_json",
    "repair_json",
    "wrap_untrusted",
]
