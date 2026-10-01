"""Live Groq integration.

Skipped unless a real API key is present, so the default suite stays offline and free:

    uv run pytest -m requires_groq        # opt in
    uv run pytest -m "not requires_groq"  # default CI behaviour

These tests deliberately assert on *contract* properties — that a response validates,
that usage is reported, that errors are sanitised — and never on the model's wording.
Asserting on generated text would make the suite flaky for no benefit, and the whole
architecture exists so that nothing depends on what the model says.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, SecretStr

from vm_config.settings import LLMSettings
from vm_llm.groq_provider import GroqLLMProvider
from vm_llm.types import LLMError, LLMUnavailableError, Message

REPO_ROOT = Path(__file__).resolve().parents[2]
KEY_FILE = REPO_ROOT / "apikeys" / "api.key"


def _key_available() -> bool:
    try:
        return KEY_FILE.is_file() and bool(KEY_FILE.read_text(encoding="utf-8").strip())
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_groq,
    pytest.mark.skipif(not _key_available(), reason="no Groq API key at apikeys/api.key"),
]


class TradeOffExplanation(BaseModel):
    """Mirrors the real narration task: explain an already-decided ranking."""

    model_config = {"extra": "forbid"}

    headline: Annotated[str, Field(min_length=10, max_length=120)]
    reasoning: Annotated[str, Field(min_length=40, max_length=600)]
    confidence: Literal["high", "medium", "low"]
    factors: Annotated[list[str], Field(min_length=1, max_length=5)]


SYSTEM = (
    "You are the explanation component of VoyageMesh, a travel planner. The option below "
    "was ALREADY selected by deterministic scoring code. Explain the trade-off for the "
    "traveller. Never invent prices, times, or options. Never claim anything is booked."
)

USER = (
    "Selected: Coach, 19.51 EUR, 3h50m, 0 transfers, departs 07:45.\n"
    "Alternatives: Train 42.50 EUR 2h51m; Coach 14.11 EUR 5h04m with 2 transfers.\n"
    "Strategy: balanced. Explain why the selected option won."
)


@pytest.fixture
async def provider():
    """Function-scoped, and explicitly closed.

    The Groq SDK holds an httpx client bound to the event loop that created it, and
    pytest-asyncio gives each test a fresh loop. Sharing one provider across tests — or
    leaving its pool for the garbage collector — raises "Event loop is closed" during
    teardown. Both were observed before this fixture was narrowed and given an explicit
    close, which is also why GroqLLMProvider grew an aclose().
    """
    settings = LLMSettings(_env_file=None, GROQ_API_KEY_FILE=str(KEY_FILE))
    instance = GroqLLMProvider(settings)
    try:
        yield instance
    finally:
        await instance.aclose()


async def test_health_check_succeeds(provider):
    assert await provider.health_check() is True


async def test_structured_output_validates_against_the_schema(provider):
    response = await provider.generate_structured(
        [Message.system(SYSTEM), Message.user(USER)],
        TradeOffExplanation,
        temperature=0.0,
        timeout_seconds=30.0,
    )
    value = response.value
    assert isinstance(value, TradeOffExplanation)
    assert value.confidence in ("high", "medium", "low")
    assert 1 <= len(value.factors) <= 5
    assert len(value.headline) >= 10


async def test_usage_and_latency_are_reported(provider):
    """Token accounting must work against the real API, not just the mock."""
    response = await provider.generate_structured(
        [Message.system(SYSTEM), Message.user(USER)], TradeOffExplanation, timeout_seconds=30.0
    )
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.usage.estimated_cost_usd > 0
    assert response.latency_ms > 0
    assert response.provider == "groq"
    assert response.attempts >= 1


async def test_narrative_makes_no_booking_claim(provider):
    """Guards T-10. A booking claim is the most consumer-harmful thing this could emit."""
    response = await provider.generate_structured(
        [Message.system(SYSTEM), Message.user(USER)], TradeOffExplanation, timeout_seconds=30.0
    )
    text = f"{response.value.headline} {response.value.reasoning}".lower()
    for claim in ("i have booked", "your booking is confirmed", "reservation confirmed"):
        assert claim not in text


async def test_invalid_credentials_produce_a_sanitised_error(provider):
    """A rejected key must not appear in the error that propagates."""
    fake = "gsk_" + "b" * 40
    bad = GroqLLMProvider(
        LLMSettings(_env_file=None, GROQ_API_KEY_FILE=str(KEY_FILE)),
        api_key=SecretStr(fake),
    )
    try:
        with pytest.raises(LLMError) as exc:
            await bad.generate_structured(
                [Message.user("hello")], TradeOffExplanation, timeout_seconds=20.0
            )
    finally:
        await bad.aclose()
    message = str(exc.value)
    assert fake not in message
    assert "gsk_" not in message
    assert isinstance(exc.value, LLMUnavailableError)


async def test_short_timeout_is_enforced(provider):
    """A 1 ms budget cannot be met by a network round trip."""
    from vm_llm.types import LLMTimeoutError

    with pytest.raises((LLMTimeoutError, LLMError)):
        await provider.generate_structured(
            [Message.system(SYSTEM), Message.user(USER)],
            TradeOffExplanation,
            timeout_seconds=0.001,
        )
