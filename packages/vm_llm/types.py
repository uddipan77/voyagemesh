"""LLM-layer types: messages, usage accounting, and structured errors.

Nothing here is Groq-specific. The rest of the codebase only ever sees these types, which
is what makes the provider genuinely swappable rather than nominally abstracted (ADR-008).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "LLMError",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "Message",
    "Role",
    "StructuredOutputError",
    "TokenUsage",
]


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of a conversation."""

    role: Role
    content: str

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(role=Role.ASSISTANT, content=content)

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts and an estimated cost for one call.

    Cost is an *estimate* computed from configured per-million rates. It drives local
    dashboards only and is never presented as a billing figure.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            estimated_cost_usd=round(self.estimated_cost_usd + other.estimated_cost_usd, 8),
        )


@dataclass(frozen=True, slots=True)
class LLMResponse[T]:
    """A validated structured response plus the metadata needed for observability."""

    value: T
    model: str
    usage: TokenUsage
    latency_ms: int
    provider: str
    attempts: int = 1
    repaired: bool = False
    """True when the raw output needed JSON repair before it would validate. Surfaced as a
    metric — a rising repair rate is an early warning of prompt or model drift."""

    finish_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class LLMError(Exception):
    """Base class for LLM failures.

    Messages are written to be safe for logs and for a sanitised API response: they never
    contain the API key, the full prompt, or a provider's raw error body.
    """

    retryable: bool = False


class LLMTimeoutError(LLMError):
    retryable = True


class LLMRateLimitError(LLMError):
    """Provider rejected the call for rate limiting."""

    retryable = True

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMUnavailableError(LLMError):
    """The provider could not be reached, or is not configured."""

    retryable = True


class LLMAuthenticationError(LLMUnavailableError):
    """The provider rejected the credentials.

    Subclasses :class:`LLMUnavailableError` so existing handling still catches it, but is
    **not retryable**: a 401 will not become a 200 on the second attempt, and retrying
    burns the caller's latency budget for nothing.
    """

    retryable = False


class StructuredOutputError(LLMError):
    """The model's output did not validate against the requested schema.

    Retryable because feeding the validation error back to the model frequently fixes it.
    After the retry budget is exhausted this propagates: the system reports a failure
    rather than passing unvalidated data downstream.
    """

    retryable = True

    def __init__(self, message: str, *, raw_output: str | None = None) -> None:
        super().__init__(message)
        # Retained in memory for the repair retry, but deliberately never included in
        # ``str(self)`` — model output can echo prompt content into a log line.
        self.raw_output = raw_output
