"""Budget summary.

:class:`BudgetSummary` is the one place in the system where the trip's cost is decided,
and it is constructed exclusively by ``vm_domain.budget`` from Python arithmetic. The
model re-verifies its own totals in a validator, so a summary whose components do not add
up to its total cannot exist — which makes the "component totals equal the total"
output guardrail structurally guaranteed rather than a check someone might forget to run.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, computed_field, model_validator

from vm_contracts.common import Currency, Money, StrictModel

__all__ = ["BudgetCategory", "BudgetLine", "BudgetStatus", "BudgetSummary"]


class BudgetCategory(StrEnum):
    TRANSPORT = "transport"
    ACCOMMODATION = "accommodation"
    ACTIVITIES = "activities"
    FOOD = "food"
    LOCAL_TRANSPORT = "local_transport"
    CONTINGENCY = "contingency"


class BudgetStatus(StrEnum):
    WITHIN_BUDGET = "within_budget"
    AT_BUDGET = "at_budget"
    OVER_BUDGET = "over_budget"
    INCOMPLETE = "incomplete"
    """One or more components could not be priced — the total is a lower bound, and the
    UI must not present it as a complete figure."""


class BudgetLine(StrictModel):
    """One costed component of the trip."""

    category: BudgetCategory
    label: Annotated[str, Field(min_length=1, max_length=120)]
    amount: Money
    is_estimate: bool = False
    """True for heuristic figures (food, local transport) rather than quoted prices."""

    basis: Annotated[str, Field(max_length=200)] = ""
    """How the figure was derived, e.g. ``"3 nights x 42.00 EUR"``. Shown to the user so
    no number in the budget is unexplained."""


class BudgetSummary(StrictModel):
    """The complete trip budget, computed and self-verifying."""

    currency: Currency
    max_budget: Money
    lines: Annotated[list[BudgetLine], Field(min_length=1, max_length=20)]
    total: Money

    travellers: Annotated[int, Field(ge=1)]
    nights: Annotated[int, Field(ge=0)]
    status: BudgetStatus
    missing_components: list[BudgetCategory] = Field(default_factory=list)
    notes: Annotated[list[str], Field(max_length=10)] = Field(default_factory=list)

    @model_validator(mode="after")
    def _verify_arithmetic(self) -> Self:
        for line in self.lines:
            if line.amount.currency is not self.currency:
                raise ValueError(
                    f"budget line '{line.label}' is in {line.amount.currency.value} but the "
                    f"budget is in {self.currency.value}; convert before summing"
                )
        if self.max_budget.currency is not self.currency:
            raise ValueError("max_budget currency must match the budget currency")
        if self.total.currency is not self.currency:
            raise ValueError("total currency must match the budget currency")

        computed = sum((line.amount.amount for line in self.lines), start=self.total.amount * 0)
        if computed != self.total.amount:
            raise ValueError(
                f"budget does not add up: lines sum to {computed} but total is "
                f"{self.total.amount} {self.currency.value}"
            )

        # INCOMPLETE is validated on its own terms rather than being compared against
        # _derive_status(). Folding it into the derivation made the missing_components
        # check unreachable and produced a misleading "figures imply within_budget"
        # message for a summary whose real problem was an unpriced component.
        if self.status is BudgetStatus.INCOMPLETE:
            if not self.missing_components:
                raise ValueError(
                    "status=incomplete requires missing_components to say what is absent"
                )
            return self

        if self.missing_components:
            raise ValueError(
                f"missing_components names {len(self.missing_components)} unpriced component(s) "
                f"but status is {self.status.value}; a total that omits a component must be "
                f"reported as incomplete rather than presented as final"
            )

        expected = self._derive_status()
        if self.status is not expected:
            raise ValueError(
                f"status is {self.status.value} but the figures imply {expected.value} "
                f"(total {self.total}, budget {self.max_budget})"
            )
        return self

    def _derive_status(self) -> BudgetStatus:
        """Status implied purely by the arithmetic. Assumes completeness."""
        if self.total.amount > self.max_budget.amount:
            return BudgetStatus.OVER_BUDGET
        if self.total.amount == self.max_budget.amount:
            return BudgetStatus.AT_BUDGET
        return BudgetStatus.WITHIN_BUDGET

    @computed_field  # type: ignore[prop-decorator]
    @property
    def remaining(self) -> Money:
        """Budget left over. Zero when at or over budget — never negative."""
        return self.max_budget.difference_or_zero(self.total)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def overspend(self) -> Money:
        """Amount by which the plan exceeds the budget. Zero when within budget.

        Expressed as a separate non-negative figure rather than a negative ``remaining``,
        so no consumer can accidentally render "-€45 remaining"."""
        return self.total.difference_or_zero(self.max_budget)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_within_budget(self) -> bool:
        return self.status in (BudgetStatus.WITHIN_BUDGET, BudgetStatus.AT_BUDGET)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def utilisation_percent(self) -> float:
        if self.max_budget.amount == 0:
            return 0.0
        return round(float(self.total.amount / self.max_budget.amount) * 100.0, 1)

    def amount_for(self, category: BudgetCategory) -> Money:
        total = Money.zero(self.currency)
        for line in self.lines:
            if line.category is category:
                total = total + line.amount
        return total
