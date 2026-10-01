"""Groq implementation of :class:`~vm_llm.base.LLMProvider`.

Handles the operational realities the rest of the codebase should not have to know about:
retries with exponential backoff, rate-limit signalling, timeouts, JSON repair, schema
validation, token accounting, and error sanitisation.

Secret handling
---------------
The API key is loaded from disk at construction time via :mod:`vm_config.secrets`, held
only as a :class:`~pydantic.SecretStr`, and passed to the client exactly once. It is never
logged, never placed in an exception message, and never attached to a span. Errors raised
from here are scrubbed by :func:`_sanitise` before they escape.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, cast

from pydantic import BaseModel, SecretStr, ValidationError

from vm_config.settings import LLMSettings
from vm_llm.base import build_schema_instruction
from vm_llm.json_repair import repair_json
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

__all__ = ["GroqLLMProvider"]

# Anything matching these is redacted from an error before it can reach a log or a
# response. Belt-and-braces: the key should never be in a provider message in the first
# place, but a 401 body has been known to echo the credential back.
_SECRET_PATTERNS = (
    re.compile(r"gsk_[A-Za-z0-9]{10,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{10,}"),
    re.compile(r"(?i)(api[-_]?key[\"'\s:=]+)[A-Za-z0-9._\-]{10,}"),
    re.compile(r"(?i)(authorization[\"'\s:=]+)[A-Za-z0-9._\-\s]{10,}"),
)

_MAX_ERROR_CHARS = 300


def _sanitise(text: str) -> str:
    """Redact anything credential-shaped and truncate.

    Applied to every provider error before it is wrapped in an :class:`LLMError`.
    """
    cleaned = text
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub(
            lambda m: (m.group(1) + "[REDACTED]") if m.groups() else "[REDACTED]", cleaned
        )
    if len(cleaned) > _MAX_ERROR_CHARS:
        cleaned = cleaned[:_MAX_ERROR_CHARS] + "…"
    return cleaned


class GroqLLMProvider:
    """Structured-output provider backed by Groq's chat completions API."""

    name = "groq"

    def __init__(
        self,
        settings: LLMSettings,
        *,
        api_key: SecretStr | None = None,
        model: str | None = None,
    ) -> None:
        """Construct the provider.

        Args:
            settings: LLM configuration (model, retries, cost rates).
            api_key: Overrides the key file. Used by tests; production loads from disk.
            model: Overrides ``settings.groq_model``.

        Raises:
            LLMUnavailableError: The ``groq`` package is missing or no key could be loaded.
        """
        self._settings = settings
        self.model = model or settings.groq_model

        try:
            from groq import AsyncGroq
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise LLMUnavailableError("the 'groq' package is not installed; run `uv sync`") from exc

        key = api_key
        if key is None:
            try:
                key = settings.load_api_key()
            except Exception:
                # Deliberately unchained: a SecretLoadError message carries a filesystem
                # path, which belongs in the local log rather than in a propagated error.
                raise LLMUnavailableError(
                    "Groq API key unavailable. Set GROQ_API_KEY_FILE to a file containing "
                    "only the key, or set LLM_PROVIDER=mock to run without one."
                ) from None

        self._client = AsyncGroq(
            api_key=key.get_secret_value(),
            base_url=settings.groq_base_url or None,
            max_retries=0,  # retries are handled here so backoff and metrics are ours
        )

    async def generate_structured[T: BaseModel](
        self,
        messages: list[Message],
        response_model: type[T],
        *,
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
        max_output_tokens: int | None = None,
    ) -> LLMResponse[T]:
        """Generate and validate a structured completion."""
        payload = self._build_messages(messages, response_model)
        max_attempts = self._settings.max_retries + 1

        started = time.perf_counter()
        usage = TokenUsage()
        last_error: LLMError | None = None
        validation_feedback: str | None = None

        for attempt in range(1, max_attempts + 1):
            attempt_messages = list(payload)
            if validation_feedback is not None:
                # Feed the validation failure back rather than blindly retrying: the model
                # usually corrects a shape error when told precisely what was wrong.
                attempt_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not match the required schema:\n"
                            f"{validation_feedback}\n"
                            "Respond again with a single valid JSON object only."
                        ),
                    }
                )

            try:
                raw, call_usage, finish_reason = await self._call(
                    attempt_messages,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                    max_output_tokens=max_output_tokens or self._settings.max_output_tokens,
                )
                usage = usage + call_usage
            except LLMError as error:
                last_error = error
                if not error.retryable or attempt == max_attempts:
                    raise
                await self._backoff(attempt, error)
                continue

            value, repaired, problem = self._parse(raw, response_model)
            if value is not None:
                return LLMResponse(
                    value=value,
                    model=self.model,
                    usage=usage,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    provider=self.name,
                    attempts=attempt,
                    repaired=repaired,
                    finish_reason=finish_reason,
                )

            last_error = StructuredOutputError(
                f"response did not match {response_model.__name__}", raw_output=raw
            )
            validation_feedback = problem
            if attempt == max_attempts:
                # Report the failure rather than passing unvalidated data downstream.
                raise last_error
            await self._backoff(attempt, last_error)

        raise last_error or LLMUnavailableError("no response from provider")

    def _build_messages(
        self, messages: list[Message], response_model: type[BaseModel]
    ) -> list[dict[str, str]]:
        """Prepend the schema instruction to the system message."""
        instruction = build_schema_instruction(response_model)
        payload = [message.to_dict() for message in messages]

        for entry in payload:
            if entry["role"] == Role.SYSTEM.value:
                entry["content"] = f"{entry['content']}\n\n{instruction}"
                return payload

        return [{"role": Role.SYSTEM.value, "content": instruction}, *payload]

    async def _call(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        timeout_seconds: float,
        max_output_tokens: int,
    ) -> tuple[str, TokenUsage, str | None]:
        """One API call, with provider exceptions mapped to our error hierarchy."""
        # The SDK types `messages` as a union of per-role TypedDicts. Our messages are
        # built from a validated Role enum, so the runtime shape is correct; the cast
        # keeps that assertion in one place rather than scattering ignores.
        payload = cast("Any", messages)
        try:
            completion = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self.model,
                    messages=payload,
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    response_format={"type": "json_object"},
                ),
                timeout=timeout_seconds,
            )
        except TimeoutError as exc:
            raise LLMTimeoutError(f"Groq call exceeded {timeout_seconds:g}s") from exc
        except Exception as exc:
            raise self._map_error(exc) from None

        choice = completion.choices[0]
        content = choice.message.content or ""
        raw_usage = getattr(completion, "usage", None)
        usage = TokenUsage(
            input_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
        )
        return content, self._with_cost(usage), choice.finish_reason

    def _map_error(self, exc: Exception) -> LLMError:
        """Translate a provider exception into a sanitised domain error."""
        message = _sanitise(str(exc))
        status = getattr(exc, "status_code", None)
        name = type(exc).__name__

        if status == 429 or "rate limit" in message.lower() or name == "RateLimitError":
            return LLMRateLimitError(
                f"Groq rate limit reached: {message}",
                retry_after_seconds=self._retry_after(exc),
            )
        if status in (408, 504) or name in ("APITimeoutError", "APIConnectionError"):
            return LLMTimeoutError(f"Groq request failed to complete: {message}")
        if status in (401, 403):
            # Never echo the credential-bearing detail, even redacted. Non-retryable:
            # a rejected credential will not be accepted on the second attempt.
            return LLMAuthenticationError(
                "Groq rejected the credentials. Check that the key file contains a valid, "
                "current API key."
            )
        if status is not None and 500 <= int(status) < 600:
            return LLMUnavailableError(f"Groq service error: {message}")
        return LLMUnavailableError(f"Groq call failed: {message}")

    def _retry_after(self, exc: Exception) -> float | None:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if not headers:
            return None
        try:
            value = headers.get("retry-after")
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _parse[T: BaseModel](
        self, raw: str, response_model: type[T]
    ) -> tuple[T | None, bool, str | None]:
        """Validate ``raw`` into ``response_model``, repairing JSON if needed.

        Returns ``(value, was_repaired, problem_description)``.
        """
        try:
            return response_model.model_validate_json(raw), False, None
        except ValidationError as exc:
            direct_problem = self._describe(exc)
        except ValueError:
            direct_problem = "response was not valid JSON"

        recovered: Any = repair_json(raw)
        if recovered is None:
            return None, False, direct_problem or "response was not valid JSON"

        try:
            return response_model.model_validate(recovered), True, None
        except ValidationError as exc:
            return None, True, self._describe(exc)

    def _describe(self, exc: ValidationError) -> str:
        """Compact, model-readable summary of what failed validation.

        Only field locations and error types are included — never the submitted values,
        which could echo prompt content into a log.
        """
        problems = [
            f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['type']}"
            for error in exc.errors()[:8]
        ]
        return "; ".join(problems)

    def _with_cost(self, usage: TokenUsage) -> TokenUsage:
        cost = (
            usage.input_tokens / 1_000_000 * self._settings.cost_per_1m_input_usd
            + usage.output_tokens / 1_000_000 * self._settings.cost_per_1m_output_usd
        )
        return TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            estimated_cost_usd=round(cost, 8),
        )

    async def _backoff(self, attempt: int, error: LLMError) -> None:
        """Exponential backoff, honouring ``Retry-After`` when the provider supplies it."""
        if isinstance(error, LLMRateLimitError) and error.retry_after_seconds:
            await asyncio.sleep(min(error.retry_after_seconds, 30.0))
            return
        await asyncio.sleep(min(0.25 * (2 ** (attempt - 1)), 8.0))

    async def health_check(self) -> bool:
        """Cheap liveness probe. Never raises."""
        try:
            await asyncio.wait_for(self._client.models.list(), timeout=5.0)
        except Exception:
            return False
        return True

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool.

        Services call this on shutdown. It matters more than it looks: the SDK's client is
        bound to the event loop that created it, and letting it be finalised after that
        loop closes raises "Event loop is closed" from the garbage collector — noise that
        masks real errors in tests and in shutdown logs.
        """
        close = getattr(self._client, "close", None)
        if close is None:
            return
        try:
            result = close()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            # Shutdown must not fail because a socket was already gone.
            return

    async def __aenter__(self) -> GroqLLMProvider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        # No key material, not even a prefix.
        return f"GroqLLMProvider(model={self.model!r})"
