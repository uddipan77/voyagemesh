"""Trip budget calculation.

The single place where a trip's cost is decided, and it is arithmetic — no LLM is involved
at any point (ADR-009). :class:`~vm_contracts.budget.BudgetSummary` re-verifies the sum in
its own validator, so a mistake here fails loudly rather than reaching the user.

Estimates versus quotes
-----------------------
Transport and accommodation come from provider offers and are exact for what they quote.
Food and local transport are *heuristics* — nobody quotes them — so every such line is
marked ``is_estimate=True`` and carries a ``basis`` string explaining how it was derived.
The UI renders estimates differently from quoted prices.

Omission is never silent. If accommodation could not be priced, the summary is
``INCOMPLETE`` and names the missing category, rather than quietly reporting a total that
happens to exclude the largest expense.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from vm_contracts.budget import BudgetCategory, BudgetLine, BudgetStatus, BudgetSummary
from vm_contracts.common import AccommodationType, Currency, Money
from vm_contracts.itinerary import Itinerary
from vm_contracts.offers import AccommodationOffer, TransportOffer
from vm_contracts.trip import NormalizedTripRequest

__all__ = [
    "DailyAllowances",
    "calculate_budget",
    "default_allowances",
    "remaining_accommodation_allowance",
]


@dataclass(frozen=True, slots=True)
class DailyAllowances:
    """Per-traveller, per-day estimates for costs no provider quotes.

    Figures are conservative mid-range European city estimates. They are configuration,
    not facts, and are labelled as estimates wherever they appear.
    """

    food_per_day: Decimal
    local_transport_per_day: Decimal

    def __post_init__(self) -> None:
        if self.food_per_day < 0 or self.local_transport_per_day < 0:
            raise ValueError("allowances must be non-negative")


# Tiered by accommodation preference, which is the best available signal of how the
# traveller intends to spend generally: someone booking a hostel is not usually planning
# EUR 60 dinners.
_ALLOWANCES_BY_TYPE: dict[AccommodationType, DailyAllowances] = {
    AccommodationType.HOSTEL: DailyAllowances(
        food_per_day=Decimal("22"), local_transport_per_day=Decimal("6")
    ),
    AccommodationType.BUDGET_HOTEL: DailyAllowances(
        food_per_day=Decimal("30"), local_transport_per_day=Decimal("7")
    ),
    AccommodationType.GUESTHOUSE: DailyAllowances(
        food_per_day=Decimal("30"), local_transport_per_day=Decimal("7")
    ),
    AccommodationType.APARTMENT: DailyAllowances(
        food_per_day=Decimal("26"), local_transport_per_day=Decimal("7")
    ),
    AccommodationType.HOTEL: DailyAllowances(
        food_per_day=Decimal("45"), local_transport_per_day=Decimal("9")
    ),
    AccommodationType.ANY: DailyAllowances(
        food_per_day=Decimal("30"), local_transport_per_day=Decimal("7")
    ),
}


def default_allowances(accommodation_type: AccommodationType) -> DailyAllowances:
    return _ALLOWANCES_BY_TYPE[accommodation_type]


def calculate_budget(
    request: NormalizedTripRequest,
    *,
    transport: TransportOffer | None,
    accommodation: AccommodationOffer | None,
    itinerary: Itinerary | None = None,
    allowances: DailyAllowances | None = None,
    include_estimates: bool = True,
) -> BudgetSummary:
    """Compute the full trip budget.

    Args:
        request: The normalised trip request; supplies currency, travellers, and nights.
        transport: The recommended transport offer, or ``None`` if none was found.
        accommodation: The recommended accommodation, or ``None``.
        itinerary: Used to cost admissions. Activity costs are only included when the
            itinerary actually prices them.
        allowances: Override the per-day estimates.
        include_estimates: When ``False``, food and local transport are omitted entirely.
            Useful for comparing like-for-like against a provider quote.

    Returns:
        A :class:`BudgetSummary`. Status is ``INCOMPLETE`` whenever a required component
        could not be priced.

    Raises:
        ValueError: If an offer's currency differs from the request's. Currency conversion
            is deliberately not attempted — silently converting at an unstated rate would
            put an invented number in front of the user.
    """
    currency: Currency = request.currency
    allowances = allowances or default_allowances(request.accommodation_preference)

    lines: list[BudgetLine] = []
    missing: list[BudgetCategory] = []
    notes: list[str] = []

    # --- Transport -------------------------------------------------------
    if transport is not None:
        _require_currency(transport.total_price, currency, "transport offer")
        lines.append(
            BudgetLine(
                category=BudgetCategory.TRANSPORT,
                label=_transport_label(transport),
                amount=transport.total_price,
                is_estimate=False,
                basis=(
                    f"{transport.price_per_traveller} x {transport.travellers} "
                    f"traveller(s), {transport.transfer_count} transfer(s)"
                ),
            )
        )
    else:
        missing.append(BudgetCategory.TRANSPORT)
        notes.append("No transport option could be priced for this route and date.")

    # --- Accommodation ---------------------------------------------------
    if request.nights == 0:
        notes.append("Day trip — no accommodation required.")
    elif accommodation is not None:
        _require_currency(accommodation.total_price, currency, "accommodation offer")
        lines.append(
            BudgetLine(
                category=BudgetCategory.ACCOMMODATION,
                label=accommodation.name,
                amount=accommodation.total_price,
                is_estimate=False,
                basis=(
                    f"{accommodation.price_per_night} x {accommodation.nights} night(s), "
                    f"{accommodation.guests} guest(s)"
                ),
            )
        )
    else:
        missing.append(BudgetCategory.ACCOMMODATION)
        notes.append("No accommodation could be priced for these dates.")

    # --- Activities ------------------------------------------------------
    # Only included when the itinerary actually carries prices. A zero line would claim
    # the activities are free, which is a different and false statement from "unpriced".
    if itinerary is not None:
        if any(slot.cost_is_unknown for day in itinerary.days for slot in day.slots):
            missing.append(BudgetCategory.ACTIVITIES)
            notes.append("Some admission prices are unknown; the total excludes those costs.")
        activity_cost = itinerary.estimated_activity_cost()
        if activity_cost is not None:
            _require_currency(activity_cost, currency, "itinerary activities")
            lines.append(
                BudgetLine(
                    category=BudgetCategory.ACTIVITIES,
                    label="Admissions and activities",
                    amount=activity_cost.times(request.travellers),
                    is_estimate=False,
                    basis=f"{activity_cost} x {request.travellers} traveller(s)",
                )
            )

    # --- Estimated day-to-day costs --------------------------------------
    if include_estimates:
        traveller_days = request.days * request.travellers

        food_total = Money.of(allowances.food_per_day * traveller_days, currency)
        lines.append(
            BudgetLine(
                category=BudgetCategory.FOOD,
                label="Food (estimated)",
                amount=food_total,
                is_estimate=True,
                basis=(
                    f"{Money.of(allowances.food_per_day, currency)}/day x {request.days} day(s) "
                    f"x {request.travellers} traveller(s)"
                ),
            )
        )

        local_total = Money.of(allowances.local_transport_per_day * traveller_days, currency)
        lines.append(
            BudgetLine(
                category=BudgetCategory.LOCAL_TRANSPORT,
                label="Local transport (estimated)",
                amount=local_total,
                is_estimate=True,
                basis=(
                    f"{Money.of(allowances.local_transport_per_day, currency)}/day "
                    f"x {request.days} day(s) x {request.travellers} traveller(s)"
                ),
            )
        )

    # A summary must have at least one line. A request that priced nothing still needs a
    # representable answer, so emit an explicit zero line and let the status say why.
    if not lines:
        lines.append(
            BudgetLine(
                category=BudgetCategory.TRANSPORT,
                label="No priced components",
                amount=Money.zero(currency),
                is_estimate=False,
                basis="Nothing could be priced for this request.",
            )
        )

    total = Money.zero(currency)
    for line in lines:
        total = total + line.amount

    status = _derive_status(total, request.max_budget, missing)
    if status is BudgetStatus.OVER_BUDGET:
        notes.append(
            f"Plan exceeds the stated budget by {total.difference_or_zero(request.max_budget)}."
        )

    return BudgetSummary(
        currency=currency,
        max_budget=request.max_budget,
        lines=lines,
        total=total,
        travellers=request.travellers,
        nights=request.nights,
        status=status,
        missing_components=missing,
        notes=notes[:10],
    )


def remaining_accommodation_allowance(
    request: NormalizedTripRequest,
    *,
    transport: TransportOffer | None,
    allowances: DailyAllowances | None = None,
) -> Money:
    """How much is left for accommodation once fixed and estimated costs are covered.

    Used by the Stay Agent to cap its search, and by the replanning node to tighten
    constraints after an over-budget result. Floors at zero — a negative allowance is
    reported as "nothing available", never as a negative price.
    """
    allowances = allowances or default_allowances(request.accommodation_preference)
    currency = request.currency

    committed = Money.zero(currency)
    if transport is not None:
        _require_currency(transport.total_price, currency, "transport offer")
        committed = committed + transport.total_price

    traveller_days = request.days * request.travellers
    committed = committed + Money.of(allowances.food_per_day * traveller_days, currency)
    committed = committed + Money.of(allowances.local_transport_per_day * traveller_days, currency)

    return request.max_budget.difference_or_zero(committed)


def _derive_status(total: Money, max_budget: Money, missing: list[BudgetCategory]) -> BudgetStatus:
    if missing:
        return BudgetStatus.INCOMPLETE
    if total.amount > max_budget.amount:
        return BudgetStatus.OVER_BUDGET
    if total.amount == max_budget.amount:
        return BudgetStatus.AT_BUDGET
    return BudgetStatus.WITHIN_BUDGET


def _transport_label(offer: TransportOffer) -> str:
    modes = "/".join(mode.value for mode in offer.modes)
    carriers = ", ".join(dict.fromkeys(leg.carrier for leg in offer.legs))
    return f"{modes.title()} — {carriers}"


def _require_currency(amount: Money, expected: Currency, what: str) -> None:
    if amount.currency is not expected:
        raise ValueError(
            f"{what} is priced in {amount.currency.value} but the trip budget is in "
            f"{expected.value}. VoyageMesh does not convert currencies: an unstated "
            f"exchange rate would put an invented number in the user's budget."
        )
