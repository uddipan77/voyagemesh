"""Stay Agent.

Skill: ``plan_accommodation``. Searches accommodation via the Lodging MCP, takes the
deterministically-ranked, stay-priced result, and returns the recommendation plus up to
five alternatives with a reasoning summary.

As with Transport, the agent computes nothing: the MCP prices the whole stay and ranks the
options (ADR-009). The agent orchestrates, budgets, and narrates.
"""

from __future__ import annotations

from typing import Any

from vm_contracts.a2a import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    AgentTask,
    AuthScheme,
    TaskError,
    TaskStatus,
)
from vm_contracts.agent_results import ReasoningSummary, StayResult
from vm_contracts.mcp import AccommodationSearchArgs
from vm_contracts.offers import AccommodationOffer
from vm_harness.agent import AgentOutcome, SpecialistAgent
from vm_harness.mcp_client import ToolClient
from vm_harness.trace import AgentRunContext
from vm_llm.base import LLMProvider

SKILL = "plan_accommodation"
AGENT_NAME = "stay-agent"
VERSION = "0.1.0"

_NARRATION_SYSTEM = (
    "You are the Stay Agent's explanation component for VoyageMesh. You are given an "
    "accommodation option ALREADY selected by deterministic scoring code, plus the "
    "alternatives it beat. Explain the choice for the traveller in plain language. Never "
    "invent prices, names, ratings, or properties. Never claim anything is booked. Report "
    "only what the data shows."
)


class StayAgent(SpecialistAgent):
    """Searches and recommends accommodation."""

    def __init__(self, *, tool_client: ToolClient, llm: LLMProvider, **kwargs: Any) -> None:
        super().__init__(
            agent_name=AGENT_NAME, version=VERSION, tool_client=tool_client, llm=llm, **kwargs
        )

    @property
    def skill(self) -> str:
        return SKILL

    def agent_card(self, *, url: str) -> AgentCard:
        return AgentCard(
            name=AGENT_NAME,
            display_name="Stay Agent",
            description=(
                "Searches accommodation for a stay, applies the traveller's budget, type, "
                "and accessibility constraints, and returns a deterministically ranked "
                "recommendation priced for the whole stay, with up to five alternatives."
            ),
            version=VERSION,
            url=url,
            skills=[
                AgentSkill(
                    name=SKILL,
                    description="Search and rank accommodation for a destination and dates.",
                    input_schema=AccommodationSearchArgs.model_json_schema(),
                    output_schema=StayResult.model_json_schema(),
                    tags=["accommodation", "lodging", "ranking"],
                    max_duration_seconds=25.0,
                )
            ],
            capabilities=AgentCapabilities(supports_partial_results=True),
            auth_scheme=AuthScheme.BEARER_JWT,
            required_roles=["agent:invoke"],
        )

    async def execute(self, task: AgentTask, context: AgentRunContext) -> AgentOutcome:
        try:
            args = AccommodationSearchArgs.model_validate(task.payload)
        except Exception:
            return AgentOutcome(
                status=TaskStatus.REJECTED,
                error=TaskError(
                    code="invalid_payload",
                    message="the task payload did not match the accommodation search schema",
                ),
            )

        context.note(
            f"Planning accommodation in {args.destination}, "
            f"{args.check_in} to {args.check_out}, {args.guests} guest(s)"
        )

        search = await self.harness.call_tool(
            context,
            tool="search_accommodation",
            arguments=args.model_dump(mode="json"),
            result_counter="offers",
            summary=f"searched accommodation in {args.destination}",
        )

        if not search.ok:
            return AgentOutcome(
                status=TaskStatus.FAILED,
                error=TaskError(
                    code="accommodation_unavailable",
                    message="accommodation search was unavailable for these dates",
                    retryable=True,
                ),
            )

        offers = _parse_offers(search.data.get("offers", []))
        if not offers:
            context.warn("No accommodation satisfied the stated constraints.")
            return AgentOutcome(
                status=TaskStatus.FAILED,
                error=TaskError(
                    code="no_viable_accommodation",
                    message="no accommodation matched the dates, budget, and constraints",
                    retryable=False,
                ),
            )

        recommended = offers[0]
        alternatives = offers[1:6]
        reasoning = await self._narrate(context, args, recommended, alternatives)

        result = StayResult(
            recommended=recommended,
            alternatives=alternatives,
            total_found=int(search.data.get("total_found", len(offers))),
            excluded_by_constraints=int(search.data.get("excluded_by_constraints", 0)),
            data_origin=search.origin,
            reasoning=reasoning,
        )
        status = TaskStatus.PARTIAL if context.degraded else TaskStatus.COMPLETED
        return AgentOutcome(status=status, result=result.model_dump(mode="json"))

    async def _narrate(
        self,
        context: AgentRunContext,
        args: AccommodationSearchArgs,
        recommended: AccommodationOffer,
        alternatives: list[AccommodationOffer],
    ) -> ReasoningSummary | None:
        return await self.harness.narrate(
            context,
            system=_NARRATION_SYSTEM,
            user=_narration_prompt(args, recommended, alternatives),
            response_model=ReasoningSummary,
        )


def _parse_offers(raw: list[dict[str, Any]]) -> list[AccommodationOffer]:
    from vm_contracts.mcp_runtime import strip_computed_fields

    offers: list[AccommodationOffer] = []
    for item in raw:
        try:
            offers.append(
                AccommodationOffer.model_validate(strip_computed_fields(item, AccommodationOffer))
            )
        except Exception:  # noqa: S112  # nosec B112 - one malformed record must not lose the rest
            continue
    return offers


def _narration_prompt(
    args: AccommodationSearchArgs,
    recommended: AccommodationOffer,
    alternatives: list[AccommodationOffer],
) -> str:
    def describe(offer: AccommodationOffer) -> str:
        distance = (
            f", {offer.distance_to_centre_km:.1f} km from centre"
            if offer.distance_to_centre_km is not None
            else ""
        )
        rating = f", rated {offer.rating}/10" if offer.rating is not None else ""
        cancel = (
            "free cancellation"
            if offer.cancellation and offer.cancellation.free_cancellation
            else "non-refundable"
        )
        return (
            f"{offer.accommodation_type.value}, {offer.total_price} total "
            f"({offer.price_per_night}/night x {offer.nights}){distance}{rating}, {cancel}"
        )

    lines = [
        f"Destination: {args.destination}, {args.nights} night(s), {args.guests} guest(s).",
        f"Ranking strategy: {args.ranking_strategy.value}.",
        f"SELECTED: {describe(recommended)}.",
    ]
    if alternatives:
        lines.append("ALTERNATIVES CONSIDERED:")
        lines.extend(f"  - {describe(alt)}" for alt in alternatives)
    lines.append("Explain why the selected option fits the strategy better than the alternatives.")
    return "\n".join(lines)


def build_agent(*, tool_client: ToolClient, llm: LLMProvider, **kwargs: Any) -> StayAgent:
    return StayAgent(tool_client=tool_client, llm=llm, **kwargs)
