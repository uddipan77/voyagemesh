"""Verify the Groq integration against the live API.

Run:

    uv run python scripts/verify_groq.py

Makes a small number of real calls to confirm the provider works end to end: key loading,
structured output, schema validation, token accounting, and error sanitisation.

The key is read from ``GROQ_API_KEY_FILE`` (or the repo default) at runtime and is never
printed. This script reports only the *shape* of what it loaded — length and a masked
prefix — never the value.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from vm_config.settings import LLMSettings
from vm_llm.groq_provider import GroqLLMProvider
from vm_llm.types import LLMError, Message

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEY_FILE = REPO_ROOT / "apikeys" / "api.key"


class TradeOffExplanation(BaseModel):
    """A realistic VoyageMesh task: narrate an already-computed ranking.

    Note what the model is *not* asked for. No prices, no durations, no choice of winner —
    those are decided by ``vm_domain`` before this call. The model only turns settled
    facts into prose, which is exactly the division of labour ADR-009 requires.
    """

    model_config = {"extra": "forbid"}

    headline: Annotated[str, Field(min_length=10, max_length=120)]
    reasoning: Annotated[str, Field(min_length=40, max_length=600)]
    key_tradeoff: Annotated[str, Field(min_length=10, max_length=200)]
    confidence: Literal["high", "medium", "low"]
    suits_traveller_type: Annotated[list[str], Field(min_length=1, max_length=4)]


SYSTEM = (
    "You are the explanation component of VoyageMesh, a travel planner. You are given a "
    "transport option that has ALREADY been selected by deterministic scoring code. "
    "Your only job is to explain the trade-off in clear prose for the traveller. "
    "Never invent prices, times, or options. Never claim anything is booked."
)

USER = """The deterministic ranker selected this option for a Nuremberg to Prague trip
(3 nights, EUR 350 total budget, 'balanced' ranking strategy, traveller interested in
history, architecture, and local food):

SELECTED: Coach, 19.51 EUR, 3h50m door-to-door, 0 transfers, departs 07:45
ALTERNATIVES CONSIDERED:
  - Train, 42.50 EUR, 2h51m, 0 transfers, departs 07:45
  - Coach, 14.11 EUR, 5h04m, 2 transfers, departs 15:45

Score components for the selected option (lower is better):
  price 0.000, duration 0.106, transfers 0.000, inconvenience 0.103

Explain why this option was chosen over the alternatives."""


def describe_key(path: Path) -> str:
    """Report the key's shape without revealing it."""
    if not path.exists():
        return f"MISSING at {path}"
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        return "EMPTY"
    return f"{len(value)} chars, starts {value[:4]}..., ends ...{value[-2:]}"


async def main() -> int:
    key_file = DEFAULT_KEY_FILE
    print("=" * 72)
    print("VoyageMesh - live Groq integration check")
    print("=" * 72)
    print(f"key file : {key_file}")
    print(f"key shape: {describe_key(key_file)}")

    settings = LLMSettings(_env_file=None, GROQ_API_KEY_FILE=str(key_file))
    print(f"model    : {settings.groq_model}")
    print(f"retries  : {settings.max_retries}   timeout: {settings.timeout_seconds}s")
    print()

    try:
        provider = GroqLLMProvider(settings)
    except LLMError as exc:
        print(f"FAILED to construct provider: {exc}")
        return 1

    print(f"provider : {provider!r}")
    print("(note the repr carries no key material)")
    print()

    print("-" * 72)
    print("1. health_check()")
    print("-" * 72)
    healthy = await provider.health_check()
    print(f"   reachable: {healthy}")
    if not healthy:
        print("   Could not reach Groq. Check network and key validity.")
        return 1
    print()

    print("-" * 72)
    print("2. generate_structured() - real call, real schema validation")
    print("-" * 72)
    try:
        response = await provider.generate_structured(
            [Message.system(SYSTEM), Message.user(USER)],
            TradeOffExplanation,
            temperature=0.0,
            timeout_seconds=30.0,
        )
    except LLMError as exc:
        print(f"   FAILED: {type(exc).__name__}: {exc}")
        return 1

    result = response.value
    print(f"   model       : {response.model}")
    print(f"   latency     : {response.latency_ms} ms")
    print(f"   attempts    : {response.attempts}")
    print(f"   json repair : {response.repaired}")
    print(f"   finish      : {response.finish_reason}")
    print(
        f"   tokens      : {response.usage.input_tokens} in / "
        f"{response.usage.output_tokens} out "
        f"(est. ${response.usage.estimated_cost_usd:.6f})"
    )
    print()
    print(f"   headline    : {result.headline}")
    print(f"   confidence  : {result.confidence}")
    print(f"   suits       : {', '.join(result.suits_traveller_type)}")
    print(f"   trade-off   : {result.key_tradeoff}")
    print()
    print("   reasoning:")
    for line in _wrap(result.reasoning, 66):
        print(f"     {line}")
    print()

    print("-" * 72)
    print("3. Determinism at temperature 0.0 (second identical call)")
    print("-" * 72)
    second = await provider.generate_structured(
        [Message.system(SYSTEM), Message.user(USER)],
        TradeOffExplanation,
        temperature=0.0,
        timeout_seconds=30.0,
    )
    identical = second.value.headline == result.headline
    print(f"   headline identical: {identical}")
    if not identical:
        print("   (Groq does not guarantee determinism even at temperature 0 -")
        print("    this is why no VoyageMesh assertion depends on LLM output.)")
    print()

    print("-" * 72)
    print("4. Error sanitisation - bad key must not leak")
    print("-" * 72)
    from pydantic import SecretStr

    bad = GroqLLMProvider(settings, api_key=SecretStr("gsk_" + "b" * 40))
    try:
        await bad.generate_structured(
            [Message.user("hi")], TradeOffExplanation, timeout_seconds=15.0
        )
        print("   unexpected: call succeeded with an invalid key")
    except LLMError as exc:
        text = f"{type(exc).__name__}: {exc}"
        leaked = "gsk_bbbb" in text
        print(f"   raised  : {text}")
        print(f"   leaks key: {leaked}")
        if leaked:
            print("   FAILED: the key appeared in the error message")
            return 1
    print()

    print("=" * 72)
    print("RESULT: live Groq integration verified")
    print("=" * 72)
    return 0


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
