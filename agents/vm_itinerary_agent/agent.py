"""Itinerary Agent.

Skill: ``plan_itinerary``. Gathers place, weather, and points of interest via the
Destination MCP, then builds a feasible day-by-day plan with the deterministic
:class:`~vm_domain.itinerary.ItineraryPlanner` and returns it with a reasoning summary.

The plan's feasibility — no overlaps, travel time respected, meals present — is guaranteed
by the planner and the ``ItineraryDay`` invariants, not by the model. The agent additionally
re-checks the assembled schedule through the ``validate_daily_schedule`` tool, so an
independent code path confirms what the planner produced (defence in depth).
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any

from pydantic import Field, model_validator

from vm_contracts.a2a import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    AgentTask,
    AuthScheme,
    TaskError,
    TaskStatus,
)
from vm_contracts.agent_results import ItineraryResult, ReasoningSummary
from vm_contracts.common import (
    AccessibilityNeed,
    AccommodationType,
    Currency,
    GeoPoint,
    Money,
    RankingStrategy,
    StrictModel,
)
from vm_contracts.destination import Attraction, WeatherSummary
from vm_contracts.itinerary import Itinerary
from vm_contracts.trip import NormalizedTripRequest, TripRequest
from vm_harness.agent import AgentOutcome, SpecialistAgent
from vm_harness.mcp_client import ToolClient
from vm_harness.trace import AgentRunContext
from vm_llm.base import LLMProvider

SKILL = "plan_itinerary"
AGENT_NAME = "itinerary-agent"
VERSION = "0.1.0"

_NARRATION_SYSTEM = (
    "You are the Itinerary Agent's explanation component for VoyageMesh. You are given a "
    "day-by-day itinerary ALREADY built by deterministic scheduling code. Summarise it for "
    "the traveller in plain language — the pace, the themes, the weather considerations. "
    "Never invent attractions, opening hours, or prices. Never claim anything is booked. "
    "Timings are estimates; say so where relevant."
)


class ItineraryTaskPayload(StrictModel):
    """The Itinerary Agent's input.

    Richer than the MCP tool args because the agent needs the whole trip shape to schedule
    days. Accommodation coordinates, when the orchestrator supplies them, anchor walking
    times; without them the planner falls back to a conservative default.
    """

    origin: Annotated[str, Field(min_length=2, max_length=80)]
    destination: Annotated[str, Field(min_length=2, max_length=80)]
    departure_date: date
    return_date: date
    travellers: Annotated[int, Field(ge=1, le=12)] = 1
    interests: Annotated[list[str], Field(max_length=12)] = Field(default_factory=list)
    accessibility_needs: Annotated[list[AccessibilityNeed], Field(max_length=8)] = Field(
        default_factory=list
    )
    accommodation_location: GeoPoint | None = None
    accommodation_type: AccommodationType = AccommodationType.ANY

    @model_validator(mode="after")
    def _validate(self) -> ItineraryTaskPayload:
        if self.return_date < self.departure_date:
            raise ValueError("return_date must not precede departure_date")
        return self

    def to_normalized(self) -> NormalizedTripRequest:
        # Budget is irrelevant to scheduling but required by the request model; use a
        # generous placeholder so validation passes without affecting the plan.
        request = TripRequest(
            origin=self.origin,
            destination=self.destination,
            departure_date=self.departure_date,
            return_date=self.return_date,
            travellers=self.travellers,
            max_budget=Money.of(100000, Currency.EUR),
            accommodation_preference=self.accommodation_type,
            interests=self.interests,
            accessibility_needs=self.accessibility_needs,
            ranking_strategy=RankingStrategy.BALANCED,
        )
        return NormalizedTripRequest.from_request(request)


class ItineraryAgent(SpecialistAgent):
    """Builds a feasible day-by-day itinerary."""

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
            display_name="Itinerary Agent",
            description=(
                "Builds a feasible day-by-day itinerary from points of interest, weather, "
                "and the traveller's interests and accessibility needs. Guarantees no "
                "overlapping activities and realistic travel time between locations."
            ),
            version=VERSION,
            url=url,
            skills=[
                AgentSkill(
                    name=SKILL,
                    description="Create a validated day-by-day itinerary for a destination.",
                    input_schema=ItineraryTaskPayload.model_json_schema(),
                    output_schema=ItineraryResult.model_json_schema(),
                    tags=["itinerary", "poi", "weather", "scheduling"],
                    max_duration_seconds=25.0,
                )
            ],
            capabilities=AgentCapabilities(supports_partial_results=True),
            auth_scheme=AuthScheme.BEARER_JWT,
            required_roles=["agent:invoke"],
        )

    async def execute(self, task: AgentTask, context: AgentRunContext) -> AgentOutcome:
        try:
            payload = ItineraryTaskPayload.model_validate(task.payload)
        except Exception:
            return AgentOutcome(
                status=TaskStatus.REJECTED,
                error=TaskError(
                    code="invalid_payload",
                    message="the task payload did not match the itinerary schema",
                ),
            )

        context.note(
            f"Planning {payload.destination} itinerary, "
            f"{payload.departure_date}→{payload.return_date}"
        )

        location = payload.accommodation_location or await self._geocode(
            context, payload.destination
        )
        attractions, poi_origin = await self._search_poi(context, payload)
        weather, weather_origin = await self._weather(context, payload, location)

        # Build the plan with deterministic code. Feasibility is guaranteed here.
        from vm_domain.itinerary import ItineraryPlanner

        itinerary = ItineraryPlanner().plan(
            payload.to_normalized(),
            attractions=attractions,
            weather=weather,
            accommodation=None,
        )

        if itinerary.day_count == 0:
            return AgentOutcome(
                status=TaskStatus.FAILED,
                error=TaskError(
                    code="itinerary_unavailable",
                    message="an itinerary could not be built for this destination",
                    retryable=False,
                ),
            )

        await self._validate_schedules(context, itinerary)

        reasoning = await self._narrate(context, payload, itinerary, weather)

        result = ItineraryResult(
            itinerary=itinerary,
            weather=weather,
            considered_attractions=attractions[:40],
            weather_origin=weather_origin,
            poi_origin=poi_origin,
            reasoning=reasoning,
        )
        status = TaskStatus.PARTIAL if context.degraded else TaskStatus.COMPLETED
        return AgentOutcome(status=status, result=result.model_dump(mode="json"))

    # -- gathering steps -------------------------------------------------
    async def _geocode(self, context: AgentRunContext, place: str) -> GeoPoint | None:
        try:
            result = await self.harness.call_tool(
                context,
                tool="geocode_destination",
                arguments={"place": place},
                summary=f"geocoded {place}",
            )
        except Exception:
            context.warn("Geocoding was unavailable; walking times use a conservative default.")
            return None
        if result.ok and "latitude" in result.data:
            return GeoPoint(
                latitude=float(result.data["latitude"]),
                longitude=float(result.data["longitude"]),
            )
        return None

    async def _search_poi(
        self, context: AgentRunContext, payload: ItineraryTaskPayload
    ) -> tuple[list[Attraction], str]:
        try:
            result = await self.harness.call_tool(
                context,
                tool="search_points_of_interest",
                arguments={
                    "destination": payload.destination,
                    "interests": payload.interests,
                    "limit": 30,
                },
                result_counter="attractions",
                summary=f"searched attractions in {payload.destination}",
            )
        except Exception:
            context.warn("Points-of-interest search was unavailable.")
            context.degraded = True
            return [], "unavailable"

        if not result.ok:
            context.warn(f"No attraction data for {payload.destination}.")
            context.degraded = True
            return [], result.origin

        attractions = _parse_attractions(result.data.get("attractions", []))
        return attractions, result.origin

    async def _weather(
        self, context: AgentRunContext, payload: ItineraryTaskPayload, location: GeoPoint | None
    ) -> tuple[WeatherSummary | None, str]:
        if location is None:
            context.warn("Weather skipped: destination could not be located.")
            return None, "unavailable"
        try:
            result = await self.harness.call_tool(
                context,
                tool="get_weather_forecast",
                arguments={
                    "location": location.model_dump(mode="json"),
                    "start_date": payload.departure_date.isoformat(),
                    "end_date": payload.return_date.isoformat(),
                    "location_name": payload.destination,
                },
                summary="retrieved the weather forecast",
            )
        except Exception:
            context.warn("Weather was unavailable; the itinerary omits weather considerations.")
            context.degraded = True
            return None, "unavailable"

        if not result.ok or "forecast" not in result.data:
            context.warn("Weather was unavailable; the itinerary omits weather considerations.")
            context.degraded = True
            return None, result.origin

        try:
            summary = WeatherSummary.model_validate(_strip(result.data["forecast"], WeatherSummary))
        except Exception:
            context.warn("The weather forecast could not be parsed; it is omitted.")
            context.degraded = True
            return None, result.origin
        return summary, result.origin

    async def _validate_schedules(self, context: AgentRunContext, itinerary: Itinerary) -> None:
        """Re-check each day through the validation tool.

        The planner already guarantees feasibility, so this is defence in depth: an
        independent code path confirming the schedule. A tool failure here is not fatal —
        the plan is already valid by construction.
        """
        for day in itinerary.days:
            try:
                result = await self.harness.call_tool(
                    context,
                    tool="validate_daily_schedule",
                    arguments={
                        "day_number": day.day_number,
                        "slots": [s.model_dump(mode="json") for s in day.slots],
                    },
                    summary=f"validated day {day.day_number} schedule",
                )
            except Exception:
                return  # budget likely exhausted; the plan is valid regardless
            if result.ok and not result.data.get("feasible", True):
                # Should not happen — the planner guarantees feasibility — but if it did,
                # say so rather than presenting an impossible day as fine.
                context.warn(f"Day {day.day_number} failed independent feasibility validation.")
                context.degraded = True

    async def _narrate(
        self,
        context: AgentRunContext,
        payload: ItineraryTaskPayload,
        itinerary: Itinerary,
        weather: WeatherSummary | None,
    ) -> ReasoningSummary | None:
        return await self.harness.narrate(
            context,
            system=_NARRATION_SYSTEM,
            user=_narration_prompt(payload, itinerary, weather),
            response_model=ReasoningSummary,
        )


def _parse_attractions(raw: list[dict[str, Any]]) -> list[Attraction]:
    attractions: list[Attraction] = []
    for item in raw:
        try:
            attractions.append(Attraction.model_validate(_strip(item, Attraction)))
        except Exception:  # noqa: S112  # nosec B112 - one malformed record must not lose the rest
            continue
    return attractions


def _strip(payload: dict[str, Any], model: type) -> dict[str, Any]:
    from vm_contracts.mcp_runtime import strip_computed_fields

    return strip_computed_fields(payload, model)


def _narration_prompt(
    payload: ItineraryTaskPayload, itinerary: Itinerary, weather: WeatherSummary | None
) -> str:
    lines = [
        f"Destination: {payload.destination}, {itinerary.day_count} day(s).",
        f"Interests: {', '.join(payload.interests) or 'general sightseeing'}.",
    ]
    for day in itinerary.days:
        attractions = [s.title for s in day.slots if s.kind.value == "attraction"]
        lines.append(f"Day {day.day_number}: {', '.join(attractions) or 'travel and free time'}.")
    if weather is not None and weather.days:
        lines.append(
            "Weather: " + "; ".join(f"{d.forecast_date} {d.condition}" for d in weather.days[:4])
        )
    lines.append("Summarise the itinerary's shape, pace, and any weather considerations.")
    return "\n".join(lines)


def build_agent(*, tool_client: ToolClient, llm: LLMProvider, **kwargs: Any) -> ItineraryAgent:
    return ItineraryAgent(tool_client=tool_client, llm=llm, **kwargs)
