"""One live-Groq agent run, opt-in.

Proves the whole agent path works with a real model producing the narrative: the mock
suite covers structure, this covers that Groq's output validates as a ``ReasoningSummary``
and makes no booking claim (threat T-10). Skipped without a key so the default suite stays
offline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from vm_config.settings import LLMSettings, Settings
from vm_contracts.a2a import AgentTask, TaskStatus
from vm_contracts.agent_results import TransportResult
from vm_harness import InProcessToolClient
from vm_llm.groq_provider import GroqLLMProvider
from vm_transport_agent.agent import TransportAgent
from vm_transport_mcp.server import build_server as build_transport

KEY_FILE = Path(__file__).resolve().parents[2] / "apikeys" / "api.key"


def _key_available() -> bool:
    try:
        return KEY_FILE.is_file() and bool(KEY_FILE.read_text(encoding="utf-8").strip())
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_groq,
    pytest.mark.skipif(not _key_available(), reason="no Groq API key"),
]


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


async def test_transport_agent_produces_a_real_narrative():
    settings = Settings.for_testing(PROVIDER_MODE="mock", ENVIRONMENT="test")
    provider = GroqLLMProvider(LLMSettings(_env_file=None, GROQ_API_KEY_FILE=str(KEY_FILE)))
    agent = TransportAgent(tool_client=InProcessToolClient(build_transport(settings)), llm=provider)
    try:
        artifact = await agent.handle(
            AgentTask(
                skill="plan_transport",
                correlation_id="c",
                request_id="r",
                payload={
                    "origin": "Nuremberg",
                    "destination": "Prague",
                    "departure_date": "2026-08-10",
                    "max_duration_hours": 8,
                    "ranking_strategy": "balanced",
                },
            )
        )
    finally:
        await provider.aclose()

    assert artifact.status is TaskStatus.COMPLETED
    assert artifact.llm_call_count == 1
    result = TransportResult.model_validate(artifact.result)
    assert result.reasoning is not None, "the live model should have produced a narrative"

    # The narrative is prose about a decided result — it must not claim a booking (T-10).
    text = f"{result.reasoning.headline} {result.reasoning.explanation}".lower()
    for claim in ("i have booked", "booking confirmed", "reservation confirmed"):
        assert claim not in text
