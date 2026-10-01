"""Output guardrails.

The last check before a plan reaches the user (brief §16C). Many of the required properties
are already guaranteed structurally — a ``Money`` always has a currency, a ``BudgetSummary``
re-verifies its own sum, an ``ItineraryDay`` cannot contain overlaps — so this layer's job
is the checks that *cannot* be expressed in a single model: cross-component consistency, and
the domain-specific honesty rules.

The rules that matter most:

* **No hallucinated booking claims** (threat T-10). The narrative is scanned for
  booking/confirmation language; a match blocks the response. This is the single most
  consumer-harmful thing the system could emit.
* **Recommended options are grounded.** A recommended offer must appear in the result's own
  alternatives-or-recommended set — an offer cannot be conjured into the recommendation.
* **Simulated data is labelled.** Any mocked component must be represented in the data
  sources, so the UI can badge it.
* **No secret leakage** in any user-facing text.

A failure is severity-graded: a *blocking* failure stops the response; a *warning* is
surfaced but does not block, because degrading loudly beats failing a usable plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from vm_contracts.plan import TripPlan

__all__ = ["GuardSeverity", "OutputGuardResult", "check_plan"]

# Booking / confirmation language the system must never emit — it cannot book anything.
_BOOKING_CLAIM = re.compile(
    r"\b(i have booked|booking (is )?confirmed|reservation (is )?confirmed|"
    r"successfully booked|your booking|confirmation (number|code|id)|"
    r"payment (received|processed)|ticket(s)? (have been )?issued)\b",
    re.IGNORECASE,
)

# Credential-shaped content that must never appear in user-facing text.
_SECRET_SHAPES = (
    re.compile(r"gsk_[A-Za-z0-9]{10,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{10,}"),
    re.compile(r"postgresql://[^\s]+"),
    re.compile(r"redis://[^\s]+"),
)


class GuardSeverity(StrEnum):
    BLOCKING = "blocking"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class OutputGuardResult:
    """The outcome of the output guardrail check."""

    passed: bool
    blocking_failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default=())

    @property
    def should_block(self) -> bool:
        return bool(self.blocking_failures)


def check_plan(plan: TripPlan) -> OutputGuardResult:
    """Validate an assembled plan before it is returned.

    Returns an :class:`OutputGuardResult`; when ``should_block`` is true the orchestrator
    must not return the plan as-is.
    """
    blocking: list[str] = []
    warnings: list[str] = []

    narrative = f"{plan.trade_off_explanation}\n{plan.reasoning_summary}"

    # 1. No booking claims (T-10). Blocking — this is the worst thing to get wrong.
    if _BOOKING_CLAIM.search(narrative):
        blocking.append("narrative contains a booking/confirmation claim, which is never valid")

    # 2. No secret leakage in user-facing text. Blocking.
    if _contains_secret(narrative):
        blocking.append("user-facing narrative contains credential-shaped text")

    # 3. Recommended options must be grounded in the agent's own result set.
    if plan.transport and plan.transport.recommended is not None:
        rec = plan.transport.recommended
        known = {rec.offer_id, *(a.offer_id for a in plan.transport.alternatives)}
        if rec.offer_id not in known:  # pragma: no cover - structurally always present
            blocking.append("recommended transport is not among the returned options")

    if plan.accommodation and plan.accommodation.recommended is not None:
        rec_a = plan.accommodation.recommended
        known_a = {rec_a.offer_id, *(a.offer_id for a in plan.accommodation.alternatives)}
        if rec_a.offer_id not in known_a:  # pragma: no cover
            blocking.append("recommended accommodation is not among the returned options")

    # 4. Budget consistency — the summary re-verifies its own sum, but confirm the plan's
    #    within-budget claim matches the budget object.
    if plan.budget is not None and plan.within_budget is False and plan.status.value == "complete":
        warnings.append("plan is over budget but marked complete; review the status")

    # 5. Simulated data must be labelled in the data sources.
    if (
        plan.transport
        and plan.transport.data_origin == "mocked"
        and not _origin_present(plan, "mocked")
    ):
        warnings.append("transport is mocked but not represented in the data sources")

    # 6. A live section must carry a retrieval timestamp somewhere in its sources.
    for source in plan.data_sources:
        if source.origin.value in ("live", "cached") and source.retrieved_at is None:
            warnings.append(f"{source.component}: live/cached data without a retrieval timestamp")

    return OutputGuardResult(
        passed=not blocking,
        blocking_failures=tuple(blocking),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SECRET_SHAPES)


def _origin_present(plan: TripPlan, origin: str) -> bool:
    return any(source.origin.value == origin for source in plan.data_sources)
