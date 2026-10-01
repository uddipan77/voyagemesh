"""Transport Agent.

Skill: ``plan_transport``. Given a route, dates, and constraints, it searches transport via
the Transport MCP, takes the deterministically-ranked result, and returns the recommended
option plus up to five alternatives with a reasoning summary.

The agent does no ranking or pricing itself — the MCP tools do (ADR-009). Its job is to
orchestrate the tool plan, enforce the budget, decide whether flights are worth searching,
and turn the settled result into a structured, honestly-labelled artifact.
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
from vm_contracts.agent_results import ReasoningSummary, TransportResult
from vm_contracts.mcp import TransportSearchArgs
from vm_contracts.offers import TransportOffer
from vm_harness.agent import AgentOutcome, SpecialistAgent
from vm_harness.mcp_client import ToolClient
from vm_harness.trace import AgentRunContext
from vm_llm.base import LLMProvider

SKILL = "plan_transport"
AGENT_NAME = "transport-agent"
VERSION = "0.1.0"

# Below this straight-line distance, searching flights wastes a tool call: airport
# transfers make short flights slower than rail, and the MCP returns nothing anyway.
_FLIGHT_WORTH_IT_KM = 500

_NARRATION_SYSTEM = (
    "You are the Transport Agent's explanation component for VoyageMesh. You are given a "
    "transport option ALREADY selected by deterministic scoring code, plus the alternatives "
    "it beat. Explain the choice for the traveller in plain language. Never invent prices, "
    "times, carriers, or options. Never claim anything is booked. Report only what the data "
    "shows."
)


class TransportAgent(SpecialistAgent):
    """Searches and recommends transport options."""

    def __init__(self, *, tool_client: ToolClient, llm: LLMProvider, **kwargs: Any) -> None:
        super().__init__(
            agent_name=AGENT_NAME,
            version=VERSION,
            tool_client=tool_client,
            llm=llm,
            **kwargs,
        )

    @property
    def skill(self) -> str:
        return SKILL

    def agent_card(self, *, url: str) -> AgentCard:
        return AgentCard(
            name=AGENT_NAME,
            display_name="Transport Agent",
            description=(
                "Searches ground and air transport between two cities, applies the "
                "traveller's duration/transfer/accessibility constraints, and returns a "
                "deterministically ranked recommendation with up to five alternatives."
            ),
            version=VERSION,
            url=url,
            skills=[
                AgentSkill(
                    name=SKILL,
                    description="Plan and rank transport between an origin and destination.",
                    input_schema=TransportSearchArgs.model_json_schema(),
                    output_schema=TransportResult.model_json_schema(),
                    tags=["transport", "search", "ranking"],
                    max_duration_seconds=25.0,
                )
            ],
            capabilities=AgentCapabilities(supports_partial_results=True),
            auth_scheme=AuthScheme.BEARER_JWT,
            required_roles=["agent:invoke"],
        )

    async def execute(self, task: AgentTask, context: AgentRunContext) -> AgentOutcome:
        try:
            args = TransportSearchArgs.model_validate(task.payload)
        except Exception:
            return AgentOutcome(
                status=TaskStatus.REJECTED,
                error=TaskError(
                    code="invalid_payload",
                    message="the task payload did not match the transport search schema",
                ),
            )

        context.note(
            f"Planning transport {args.origin} → {args.destination} on {args.departure_date}"
        )

        offers, origin, found, excluded = await self._gather_offers(context, args)

        if not offers:
            context.warn("No transport option satisfied the stated constraints.")
            return AgentOutcome(
                status=TaskStatus.FAILED,
                error=TaskError(
                    code="no_viable_transport",
                    message="no transport option matched the route, dates, and constraints",
                    retryable=False,
                ),
            )

        recommended = offers[0]
        alternatives = offers[1:6]
        reasoning = await self._narrate(context, args, recommended, alternatives)

        result = TransportResult(
            recommended=recommended,
            alternatives=alternatives,
            total_found=found,
            excluded_by_constraints=excluded,
            data_origin=origin,
            reasoning=reasoning,
        )

        # PARTIAL when we produced a recommendation but something degraded along the way —
        # a failed flight search, or a missing narrative. The result is usable; the caller
        # is told it is incomplete.
        status = TaskStatus.PARTIAL if context.degraded else TaskStatus.COMPLETED
        return AgentOutcome(status=status, result=result.model_dump(mode="json"))

    async def _gather_offers(
        self, context: AgentRunContext, args: TransportSearchArgs
    ) -> tuple[list[TransportOffer], str, int, int]:
        """Run the deterministic tool plan: ground always, air when it could win.

        The two searches are merged and the *combined* set is re-ranked, so a flight and a
        coach compete on the same footing rather than being ranked within their own mode.
        """
        collected: list[dict[str, Any]] = []
        origin = "unavailable"
        found = 0
        excluded = 0

        from vm_contracts.common import TransportMode

        if args.preferred_mode is not TransportMode.FLIGHT:
            ground = await self.harness.call_tool(
                context,
                tool="search_ground_transport",
                arguments=args.model_dump(mode="json"),
                result_counter="offers",
                summary=f"searched ground transport {args.origin}→{args.destination}",
            )
            if ground.ok:
                collected.extend(ground.data.get("offers", []))
                origin = ground.origin
                found += int(ground.data.get("total_found", 0))
                excluded += int(ground.data.get("excluded_by_constraints", 0))

        if self._flights_worth_searching(args) or (
            not collected and args.preferred_mode is TransportMode.ANY
        ):
            try:
                air = await self.harness.call_tool(
                    context,
                    tool="search_flights",
                    arguments=args.model_dump(mode="json"),
                    result_counter="offers",
                    summary=f"searched flights {args.origin}→{args.destination}",
                )
                if air.ok:
                    if not collected:
                        origin = air.origin
                    collected.extend(air.data.get("offers", []))
                    found += int(air.data.get("total_found", 0))
                    excluded += int(air.data.get("excluded_by_constraints", 0))
            except Exception:
                # A flight-search failure is not fatal — ground options may suffice.
                context.warn("Flight search was unavailable; results cover ground transport only.")
                context.degraded = True

        if not collected:
            return [], origin, found, excluded

        # Re-rank the combined set through the scoring tool so the winner is chosen across
        # all modes, deterministically, by code.
        ranked = await self.harness.call_tool(
            context,
            tool="calculate_transport_score",
            arguments={"offers": collected, "ranking_strategy": args.ranking_strategy.value},
            result_counter="offers",
            summary=f"ranked {len(collected)} combined option(s) by {args.ranking_strategy.value}",
        )
        if not ranked.ok:
            # Scoring failed but we still have offers; fall back to the ground order rather
            # than losing everything.
            context.warn("Combined ranking was unavailable; using the ground-transport order.")
            offers = _parse_offers(collected)
        else:
            offers = _parse_offers(ranked.data.get("offers", []))

        return offers, origin, found or len(offers), excluded

    def _flights_worth_searching(self, args: TransportSearchArgs) -> bool:
        from vm_contracts.common import TransportMode

        if args.preferred_mode is TransportMode.FLIGHT:
            return True
        if args.preferred_mode not in (TransportMode.ANY,):
            return False  # a specific non-flight mode was requested
        return _route_distance_km(args.origin, args.destination) >= _FLIGHT_WORTH_IT_KM

    async def _narrate(
        self,
        context: AgentRunContext,
        args: TransportSearchArgs,
        recommended: TransportOffer,
        alternatives: list[TransportOffer],
    ) -> ReasoningSummary | None:
        user = _narration_prompt(args, recommended, alternatives)
        return await self.harness.narrate(
            context,
            system=_NARRATION_SYSTEM,
            user=user,
            response_model=ReasoningSummary,
        )


def _parse_offers(raw: list[dict[str, Any]]) -> list[TransportOffer]:
    """Re-hydrate offers from tool output, dropping computed fields and any that no longer
    validate. The MCP already validated them; this is defence in depth at the boundary."""
    from vm_contracts.mcp_runtime import strip_computed_fields

    offers: list[TransportOffer] = []
    for item in raw:
        try:
            offers.append(
                TransportOffer.model_validate(strip_computed_fields(item, TransportOffer))
            )
        except Exception:  # noqa: S112  # nosec B112 - one malformed record must not lose the rest
            continue
    return offers


def _route_distance_km(origin: str, destination: str) -> int:
    """A coarse, deterministic distance used only to decide whether to search flights.

    Reuses the mock provider's distance table so the decision matches the offers actually
    generated. This is a routing *hint*, never a figure shown to the user.
    """
    from vm_domain.providers.mock_transport import _distance_km

    return _distance_km(origin, destination)


def _narration_prompt(
    args: TransportSearchArgs,
    recommended: TransportOffer,
    alternatives: list[TransportOffer],
) -> str:
    def describe(offer: TransportOffer) -> str:
        hours, minutes = divmod(offer.total_duration_minutes, 60)
        modes = "/".join(m.value for m in offer.modes)
        return (
            f"{modes}, {offer.total_price}, {hours}h{minutes:02d}m, "
            f"{offer.transfer_count} transfer(s), departs "
            f"{offer.departure_at.strftime('%H:%M')}"
        )

    lines = [
        f"Route: {args.origin} to {args.destination}, {args.travellers} traveller(s).",
        f"Ranking strategy: {args.ranking_strategy.value}.",
        f"SELECTED: {describe(recommended)}.",
    ]
    if alternatives:
        lines.append("ALTERNATIVES CONSIDERED:")
        lines.extend(f"  - {describe(alt)}" for alt in alternatives)
    lines.append("Explain why the selected option fits the strategy better than the alternatives.")
    return "\n".join(lines)


def build_agent(*, tool_client: ToolClient, llm: LLMProvider, **kwargs: Any) -> TransportAgent:
    return TransportAgent(tool_client=tool_client, llm=llm, **kwargs)
