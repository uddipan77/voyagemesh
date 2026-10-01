"""Structured results the specialist agents return inside their A2A artifacts.

These are the typed payloads that go into ``AgentArtifact.result``. Defining them here, in
the shared contracts package, is what lets the orchestrator validate an agent's response
against the *same* model the agent used to build it — the schema-compatibility check the
brief requires of A2A (§7).

Each result separates the *recommended* option from the *alternatives* (top five, per the
brief) and carries the concise reasoning summary. It never carries hidden reasoning.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from vm_contracts.common import ConfidenceLevel, StrictModel
from vm_contracts.destination import Attraction, WeatherSummary
from vm_contracts.itinerary import Itinerary
from vm_contracts.offers import AccommodationOffer, TransportOffer

__all__ = [
    "ItineraryResult",
    "ReasoningSummary",
    "StayResult",
    "TransportResult",
]


class ReasoningSummary(StrictModel):
    """A human-readable explanation of a deterministic result.

    This is the shape the narration LLM must produce. It is prose *about* facts the code
    already decided — it contains no prices, schedules, or choices of its own. The output
    guardrail (Phase 8) rejects any narrative that makes a booking claim.
    """

    headline: Annotated[str, Field(min_length=5, max_length=160)]
    explanation: Annotated[str, Field(min_length=20, max_length=800)]
    key_tradeoff: Annotated[str, Field(max_length=300)] = ""
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM


class TransportResult(StrictModel):
    """The Transport Agent's structured output."""

    recommended: TransportOffer | None
    alternatives: Annotated[list[TransportOffer], Field(max_length=5)] = Field(default_factory=list)
    total_found: Annotated[int, Field(ge=0)] = 0
    excluded_by_constraints: Annotated[int, Field(ge=0)] = 0
    data_origin: Annotated[str, Field(min_length=1, max_length=20)] = "unavailable"
    reasoning: ReasoningSummary | None = None

    @property
    def has_result(self) -> bool:
        return self.recommended is not None


class StayResult(StrictModel):
    """The Stay Agent's structured output."""

    recommended: AccommodationOffer | None
    alternatives: Annotated[list[AccommodationOffer], Field(max_length=5)] = Field(
        default_factory=list
    )
    total_found: Annotated[int, Field(ge=0)] = 0
    excluded_by_constraints: Annotated[int, Field(ge=0)] = 0
    data_origin: Annotated[str, Field(min_length=1, max_length=20)] = "unavailable"
    reasoning: ReasoningSummary | None = None

    @property
    def has_result(self) -> bool:
        return self.recommended is not None


class ItineraryResult(StrictModel):
    """The Itinerary Agent's structured output."""

    itinerary: Itinerary | None
    weather: WeatherSummary | None = None
    considered_attractions: Annotated[list[Attraction], Field(max_length=40)] = Field(
        default_factory=list
    )
    weather_origin: Annotated[str, Field(max_length=20)] = "unavailable"
    poi_origin: Annotated[str, Field(max_length=20)] = "unavailable"
    reasoning: ReasoningSummary | None = None

    @property
    def has_result(self) -> bool:
        return self.itinerary is not None and self.itinerary.day_count > 0
