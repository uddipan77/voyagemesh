"""The final trip plan — the structured response the whole system produces.

This is what a user ultimately receives. It composes the specialist agents' results with
the code-computed budget and the orchestrator's honesty metadata, covering the fourteen
mandated sections (brief §4) in one validated object.

Two properties are load-bearing and enforced here rather than trusted:

* **Every data source is accounted for.** :attr:`TripPlan.data_sources` lists the origin and
  retrieval time of every input, so the UI can label `LIVE` / `CACHED` / `MOCKED` / `FIXTURE`
  honestly (§4 items 10-13).
* **Degradation is visible.** :attr:`TripPlan.degraded_services` and the confidence /
  completeness fields say plainly what was unavailable, rather than presenting a partial plan
  as complete (§4 item 12, brief §23).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import Field, computed_field

from vm_contracts.agent_results import ItineraryResult, StayResult, TransportResult
from vm_contracts.budget import BudgetSummary
from vm_contracts.common import ConfidenceLevel, DataOrigin, StrictModel, utc_now

__all__ = [
    "DataSourceInfo",
    "PlanStatus",
    "TripPlan",
]


class PlanStatus(StrEnum):
    """Overall outcome of a planning request."""

    COMPLETE = "complete"
    """Every section produced with no degraded dependency."""

    PARTIAL = "partial"
    """A usable plan with one or more named gaps (brief §23)."""

    NO_VIABLE_PLAN = "no_viable_plan"
    """Constraints could not be satisfied — reported honestly, not by relaxing the budget."""

    FAILED = "failed"
    """The request could not be processed. Carries validation errors, not a fake plan."""


class DataSourceInfo(StrictModel):
    """Provenance of one input to the plan, for the 'data-source information' section."""

    component: Annotated[str, Field(min_length=1, max_length=60)]
    source_name: Annotated[str, Field(min_length=1, max_length=120)]
    origin: DataOrigin
    retrieved_at: datetime | None = None
    note: Annotated[str | None, Field(max_length=300)] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def label(self) -> str:
        return self.origin.value.upper()


class TripPlan(StrictModel):
    """The complete, validated trip plan."""

    # --- identity ---
    request_id: Annotated[str, Field(min_length=1, max_length=60)]
    trip_id: Annotated[str, Field(min_length=1, max_length=60)]
    status: PlanStatus
    generated_at: datetime = Field(default_factory=utc_now)

    # --- the sections (brief §4) ---
    transport: TransportResult | None = None  # sections 1-2
    accommodation: StayResult | None = None  # sections 3-4
    itinerary: ItineraryResult | None = None  # sections 5-6 (itinerary + weather)
    budget: BudgetSummary | None = None  # sections 7-8

    trade_off_explanation: Annotated[str, Field(max_length=2000)] = ""  # section 9
    reasoning_summary: Annotated[str, Field(max_length=2000)] = ""  # section 14

    data_sources: Annotated[list[DataSourceInfo], Field(max_length=20)] = Field(
        default_factory=list
    )  # sections 10-11

    # --- honesty metadata (sections 12-13) ---
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    completeness_percent: Annotated[float, Field(ge=0.0, le=100.0)] = 0.0
    degraded_services: Annotated[list[str], Field(max_length=10)] = Field(default_factory=list)
    warnings: Annotated[list[str], Field(max_length=30)] = Field(default_factory=list)
    unavailable_sections: Annotated[list[str], Field(max_length=10)] = Field(default_factory=list)

    # --- audit ---
    replans: Annotated[int, Field(ge=0, le=10)] = 0
    cache_status: Annotated[str, Field(max_length=20)] = "miss"
    validation_errors: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_usable(self) -> bool:
        return self.status in (PlanStatus.COMPLETE, PlanStatus.PARTIAL)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def within_budget(self) -> bool | None:
        """Whether the plan fits the budget, or ``None`` if the budget is incomplete."""
        if self.budget is None:
            return None
        return self.budget.is_within_budget

    def section_availability(self) -> dict[str, bool]:
        """Which of the headline sections are present. Drives the UI's section tabs."""
        return {
            "transport": self.transport is not None and self.transport.has_result,
            "accommodation": self.accommodation is not None and self.accommodation.has_result,
            "itinerary": self.itinerary is not None and self.itinerary.has_result,
            "budget": self.budget is not None,
        }
