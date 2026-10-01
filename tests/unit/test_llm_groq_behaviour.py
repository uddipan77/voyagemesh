"""Groq provider operational behaviour, exercised offline with a stubbed client.

The live suite (`tests/integration/test_groq_live.py`) proves the integration works. It
cannot cheaply prove what happens on the *third* rate-limit, or when the model returns
fenced JSON, or when output never validates — those paths need determinism and would
otherwise sit untested at 29% coverage while looking fine.

Here the SDK call is replaced with a scripted stub, so retries, backoff, error mapping,
JSON repair, and the validation-feedback loop are all covered without a network.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field, SecretStr

from vm_config.settings import LLMSettings
from vm_llm.groq_provider import GroqLLMProvider
from vm_llm.types import (
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
    Message,
    StructuredOutputError,
)

pytestmark = pytest.mark.unit

FAKE_KEY = SecretStr("gsk_" + "T3st" * 10)


class Answer(BaseModel):
    model_config = {"extra": "forbid"}

    headline: Annotated[str, Field(min_length=3, max_length=80)]
    confidence: Literal["high", "medium", "low"]
    score: Annotated[float, Field(ge=0.0, le=1.0)]


VALID = '{"headline": "Balanced coach option", "confidence": "high", "score": 0.21}'


class _Usage:
    def __init__(self, prompt: int = 100, completion: int = 20) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Completion:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.choices = [_Choice(content, finish_reason)]
        self.usage = _Usage()


class ScriptedClient:
    """Replays a scripted sequence of responses or exceptions."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[list[dict[str, str]]] = []
        self.chat = self  # type: ignore[assignment]
        self.completions = self

    async def create(self, **kwargs: Any) -> _Completion:
        self.calls.append(kwargs["messages"])
        if not self.script:
            raise AssertionError("scripted client exhausted — more calls than expected")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return _Completion(step)


class _ProviderError(Exception):
    """Stands in for an SDK error carrying a status code."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def build(script: list[Any], **setting_overrides: Any) -> tuple[GroqLLMProvider, ScriptedClient]:
    settings = LLMSettings(_env_file=None, **setting_overrides)
    provider = GroqLLMProvider(settings, api_key=FAKE_KEY)
    client = ScriptedClient(script)
    provider._client = client  # type: ignore[assignment]
    return provider, client


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """Backoff is real code we want exercised, but not real waiting."""
    import asyncio

    async def instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant)


class TestHappyPath:
    async def test_valid_response_is_parsed_first_time(self):
        provider, _ = build([VALID])
        response = await provider.generate_structured([Message.user("x")], Answer)
        assert response.value.headline == "Balanced coach option"
        assert response.attempts == 1
        assert response.repaired is False

    async def test_schema_instruction_is_appended_to_the_system_message(self):
        provider, client = build([VALID])
        await provider.generate_structured(
            [Message.system("You explain rankings."), Message.user("x")], Answer
        )
        system = client.calls[0][0]
        assert system["role"] == "system"
        assert "You explain rankings." in system["content"]
        assert "headline" in system["content"], "the schema must reach the model"

    async def test_system_message_is_created_when_absent(self):
        provider, client = build([VALID])
        await provider.generate_structured([Message.user("x")], Answer)
        assert client.calls[0][0]["role"] == "system"

    async def test_token_usage_and_cost_are_reported(self):
        provider, _ = build([VALID])
        response = await provider.generate_structured([Message.user("x")], Answer)
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 20
        assert response.usage.estimated_cost_usd > 0


class TestJsonRepairPath:
    async def test_markdown_fenced_output_is_repaired(self):
        provider, _ = build([f"```json\n{VALID}\n```"])
        response = await provider.generate_structured([Message.user("x")], Answer)
        assert response.value.confidence == "high"
        assert response.repaired is True, "repair should be reported for observability"

    async def test_prose_wrapped_output_is_repaired(self):
        provider, _ = build([f"Sure! Here you go:\n{VALID}\nLet me know if you need more."])
        response = await provider.generate_structured([Message.user("x")], Answer)
        assert response.repaired is True

    async def test_truncated_output_is_closed_and_parsed(self):
        truncated = '{"headline": "Balanced coach option", "confidence": "high", "score": 0.21'
        provider, _ = build([truncated])
        response = await provider.generate_structured([Message.user("x")], Answer)
        assert response.value.score == pytest.approx(0.21)


class TestValidationFeedbackLoop:
    async def test_wrong_shape_is_retried_with_the_error_fed_back(self):
        """A shape error is usually corrected when the model is told what was wrong."""
        provider, client = build(['{"headline": "hi"}', VALID])
        response = await provider.generate_structured([Message.user("x")], Answer)

        assert response.attempts == 2
        retry_messages = client.calls[1]
        feedback = retry_messages[-1]["content"]
        assert "did not match the required schema" in feedback
        assert "confidence" in feedback, "the feedback must name the offending fields"

    async def test_feedback_never_echoes_submitted_values(self):
        """Validation detail must not leak prompt content into a log line."""
        provider, client = build(['{"headline": "SENSITIVE-VALUE", "score": 9.9}', VALID])
        await provider.generate_structured([Message.user("x")], Answer)
        assert "SENSITIVE-VALUE" not in client.calls[1][-1]["content"]

    async def test_persistent_invalid_output_raises_after_the_retry_budget(self):
        provider, client = build(['{"nope": 1}'] * 3, LLM_MAX_RETRIES=2)
        with pytest.raises(StructuredOutputError):
            await provider.generate_structured([Message.user("x")], Answer)
        assert len(client.calls) == 3, "should try once plus max_retries"

    async def test_unparseable_output_raises_rather_than_guessing(self):
        provider, _ = build(["complete nonsense, no json"] * 3, LLM_MAX_RETRIES=2)
        with pytest.raises(StructuredOutputError):
            await provider.generate_structured([Message.user("x")], Answer)

    async def test_usage_accumulates_across_retries(self):
        """Cost accounting must charge for the failed attempts too."""
        provider, _ = build(['{"nope": 1}', VALID])
        response = await provider.generate_structured([Message.user("x")], Answer)
        assert response.usage.input_tokens == 200  # two calls at 100 each


class TestRetryAndErrorMapping:
    async def test_rate_limit_is_retried_then_succeeds(self):
        provider, client = build([_ProviderError("rate limit exceeded", 429), VALID])
        response = await provider.generate_structured([Message.user("x")], Answer)
        assert response.value.confidence == "high"
        assert len(client.calls) == 2

    async def test_rate_limit_raises_after_budget(self):
        provider, _ = build([_ProviderError("rate limit exceeded", 429)] * 3, LLM_MAX_RETRIES=2)
        with pytest.raises(LLMRateLimitError):
            await provider.generate_structured([Message.user("x")], Answer)

    async def test_server_error_is_retried(self):
        provider, client = build([_ProviderError("internal error", 503), VALID])
        await provider.generate_structured([Message.user("x")], Answer)
        assert len(client.calls) == 2

    async def test_auth_failure_is_not_retried(self):
        """Retrying a rejected credential wastes time and quota to no purpose."""
        provider, client = build([_ProviderError("invalid api key", 401)] * 3)
        with pytest.raises(LLMUnavailableError):
            await provider.generate_structured([Message.user("x")], Answer)
        assert len(client.calls) == 1

    async def test_auth_failure_message_reveals_nothing(self):
        provider, _ = build([_ProviderError(f"bad key: {FAKE_KEY.get_secret_value()}", 401)])
        with pytest.raises(LLMUnavailableError) as exc:
            await provider.generate_structured([Message.user("x")], Answer)
        assert "gsk_" not in str(exc.value)

    async def test_provider_error_text_is_sanitised_before_propagating(self):
        leaky = _ProviderError(
            f"failed with Authorization: Bearer {FAKE_KEY.get_secret_value()}", 500
        )
        provider, _ = build([leaky], LLM_MAX_RETRIES=0)
        with pytest.raises(LLMUnavailableError) as exc:
            await provider.generate_structured([Message.user("x")], Answer)
        assert FAKE_KEY.get_secret_value() not in str(exc.value)
        assert "[REDACTED]" in str(exc.value)

    async def test_timeout_maps_to_timeout_error(self):
        provider, _ = build([TimeoutError()] * 3, LLM_MAX_RETRIES=0)
        with pytest.raises(LLMTimeoutError):
            await provider.generate_structured([Message.user("x")], Answer)

    async def test_retry_after_header_is_honoured(self):
        class _RateLimitedError(_ProviderError):
            def __init__(self) -> None:
                super().__init__("rate limited", 429)
                self.response = type("R", (), {"headers": {"retry-after": "2"}})()

        provider, _ = build([_RateLimitedError(), VALID])
        response = await provider.generate_structured([Message.user("x")], Answer)
        assert response.attempts == 2

    async def test_zero_retries_fails_immediately(self):
        provider, client = build([_ProviderError("boom", 500)], LLM_MAX_RETRIES=0)
        with pytest.raises(LLMUnavailableError):
            await provider.generate_structured([Message.user("x")], Answer)
        assert len(client.calls) == 1


class TestLifecycle:
    async def test_aclose_is_safe_to_call_twice(self):
        class _Closable(ScriptedClient):
            def __init__(self) -> None:
                super().__init__([])
                self.closed = 0

            async def close(self) -> None:
                self.closed += 1

        provider, _ = build([VALID])
        closable = _Closable()
        provider._client = closable  # type: ignore[assignment]
        await provider.aclose()
        await provider.aclose()
        assert closable.closed == 2

    async def test_aclose_swallows_errors(self):
        """Shutdown must not fail because a socket was already gone."""

        class _Broken(ScriptedClient):
            def __init__(self) -> None:
                super().__init__([])

            async def close(self) -> None:
                raise RuntimeError("Event loop is closed")

        provider, _ = build([VALID])
        provider._client = _Broken()  # type: ignore[assignment]
        await provider.aclose()  # must not raise

    async def test_async_context_manager_closes(self):
        provider, _ = build([VALID])
        async with provider as p:
            assert p is provider

    async def test_health_check_returns_false_on_error(self):
        class _NoModels(ScriptedClient):
            def __init__(self) -> None:
                super().__init__([])

            @property
            def models(self) -> Any:
                raise RuntimeError("unreachable")

        provider, _ = build([VALID])
        provider._client = _NoModels()  # type: ignore[assignment]
        assert await provider.health_check() is False
