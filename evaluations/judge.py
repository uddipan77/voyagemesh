"""Optional LLM-as-judge scoring (brief §22).

Advisory only. The judge scores qualitative dimensions the deterministic checks cannot — is the
trade-off explanation coherent? does the plan match the stated preferences? — but it **never**
gates a run: a failed deterministic check fails the case regardless of any judge score, and the
judge is off unless explicitly enabled. When enabled it uses whatever provider is configured;
with the mock provider it returns neutral scores, so a run stays deterministic.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from evaluations.models import JudgeScores
from vm_contracts.plan import TripPlan
from vm_llm.base import LLMProvider, build_schema_instruction
from vm_llm.types import Message

__all__ = ["JudgeVerdict", "score_plan"]

logger = logging.getLogger(__name__)


class JudgeVerdict(BaseModel):
    relevance: float = Field(ge=0, le=5)
    usefulness: float = Field(ge=0, le=5)
    coherence: float = Field(ge=0, le=5)
    preference_alignment: float = Field(ge=0, le=5)
    tradeoff_quality: float = Field(ge=0, le=5)
    comment: str = Field(default="", max_length=300)


_SYSTEM = (
    "You are a strict travel-plan reviewer. Score the plan from 0 to 5 on each dimension. "
    "Judge only what is present; do not reward confident-sounding but ungrounded content. "
    "You are advisory — deterministic checks decide correctness."
)


async def score_plan(plan: TripPlan, provider: LLMProvider) -> JudgeScores:
    """Score a plan. Never raises — a judge failure yields neutral scores, never a run failure."""
    summary = (
        f"status={plan.status.value}; within_budget={plan.within_budget}; "
        f"transport={'yes' if plan.transport else 'no'}; "
        f"accommodation={'yes' if plan.accommodation else 'no'}; "
        f"itinerary={'yes' if plan.itinerary else 'no'}; "
        f"trade_off={plan.trade_off_explanation[:400]!r}; "
        f"reasoning={plan.reasoning_summary[:400]!r}"
    )
    messages = [
        Message.system(_SYSTEM + "\n\n" + build_schema_instruction(JudgeVerdict)),
        Message.user(f"Plan under review:\n{summary}"),
    ]
    try:
        response = await provider.generate_structured(messages, JudgeVerdict, temperature=0.0)
        v = response.value
        return JudgeScores(
            relevance=v.relevance,
            usefulness=v.usefulness,
            coherence=v.coherence,
            preference_alignment=v.preference_alignment,
            tradeoff_quality=v.tradeoff_quality,
            comment=v.comment,
        )
    except Exception:
        logger.warning("judge_unavailable_returning_neutral", exc_info=False)
        return JudgeScores(comment="judge unavailable")
