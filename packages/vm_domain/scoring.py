"""Deterministic ranking of transport and accommodation offers.

This module is the concrete expression of ADR-009: **the LLM does not rank anything.**
Ranking here is a pure function of the candidate set. Identical inputs produce an
identical order, forever, and every weight is visible and testable.

The scoring model
-----------------
Each attribute is min-max normalised *across the candidate set*, then combined with
strategy-specific weights::

    score = Σ (weight_i x normalised_attribute_i)

Lower is better; ``0.0`` means an option was best-in-set on every attribute. Normalising
within the set rather than against absolute thresholds is deliberate — "expensive" only
has meaning relative to the alternatives actually available for this route.

Two consequences worth understanding:

* A single candidate always scores ``0.0``. With nothing to compare against, no attribute
  carries information.
* Adding a very cheap option changes every other option's price score. That is correct:
  the others really did just become relatively worse value.

Hard constraints are *not* part of the score. An option violating the traveller's maximum
duration, transfer limit, or accessibility needs is filtered out entirely — see
:func:`filter_transport`. Scoring a step-free-access requirement against price would be
wrong in a way no weight could fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from vm_contracts.common import AccessibilityNeed, RankingStrategy
from vm_contracts.offers import AccommodationOffer, TransportOffer
from vm_domain.geo import normalise

__all__ = [
    "AccommodationWeights",
    "TransportWeights",
    "accommodation_weights_for",
    "filter_accommodation",
    "filter_transport",
    "rank_accommodation",
    "rank_transport",
    "schedule_inconvenience",
    "transport_weights_for",
]

TOP_N = 5
"""Number of alternatives returned to the user, per the brief."""


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TransportWeights:
    """Relative importance of each transport attribute. Must sum to 1.0."""

    price: float
    duration: float
    transfers: float
    inconvenience: float

    def __post_init__(self) -> None:
        total = self.price + self.duration + self.transfers + self.inconvenience
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"transport weights must sum to 1.0, got {total}")


@dataclass(frozen=True, slots=True)
class AccommodationWeights:
    """Relative importance of each accommodation attribute. Must sum to 1.0."""

    price: float
    distance: float
    rating: float
    flexibility: float

    def __post_init__(self) -> None:
        total = self.price + self.distance + self.rating + self.flexibility
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"accommodation weights must sum to 1.0, got {total}")


# These encode an opinion about what travellers want. The opinion is stated here, in one
# place, where it can be argued with and changed — rather than being an opaque judgement
# made differently on every request.
_TRANSPORT_WEIGHTS: dict[RankingStrategy, TransportWeights] = {
    # Price dominates, but not absolutely: a €2 saving is not worth six extra hours.
    RankingStrategy.CHEAPEST: TransportWeights(
        price=0.82, duration=0.10, transfers=0.05, inconvenience=0.03
    ),
    # Speed dominates; transfers still matter because each is a chance to miss a connection.
    RankingStrategy.FASTEST: TransportWeights(
        price=0.08, duration=0.74, transfers=0.10, inconvenience=0.08
    ),
    # Price leads but does not dominate. Inconvenience is weighted meaningfully here
    # because a 04:50 departure genuinely ruins a short trip.
    RankingStrategy.BALANCED: TransportWeights(
        price=0.40, duration=0.30, transfers=0.15, inconvenience=0.15
    ),
    RankingStrategy.FEWEST_TRANSFERS: TransportWeights(
        price=0.15, duration=0.15, transfers=0.65, inconvenience=0.05
    ),
}

_ACCOMMODATION_WEIGHTS: dict[RankingStrategy, AccommodationWeights] = {
    RankingStrategy.CHEAPEST: AccommodationWeights(
        price=0.80, distance=0.10, rating=0.07, flexibility=0.03
    ),
    # "Fastest" has no direct lodging meaning, so it is read as "minimise time lost
    # commuting" — distance to the centre becomes the dominant term.
    RankingStrategy.FASTEST: AccommodationWeights(
        price=0.20, distance=0.55, rating=0.20, flexibility=0.05
    ),
    RankingStrategy.BALANCED: AccommodationWeights(
        price=0.42, distance=0.25, rating=0.23, flexibility=0.10
    ),
    # Likewise, "fewest transfers" reads as "stay central so you need fewer journeys".
    RankingStrategy.FEWEST_TRANSFERS: AccommodationWeights(
        price=0.30, distance=0.45, rating=0.20, flexibility=0.05
    ),
}


def transport_weights_for(strategy: RankingStrategy) -> TransportWeights:
    return _TRANSPORT_WEIGHTS[strategy]


def accommodation_weights_for(strategy: RankingStrategy) -> AccommodationWeights:
    return _ACCOMMODATION_WEIGHTS[strategy]


# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------
EARLY_DEPARTURE_MINUTE = 6 * 60
LATE_ARRIVAL_MINUTE = 22 * 60


def schedule_inconvenience(offer: TransportOffer) -> float:
    """How antisocial the timing is, in ``[0.0, 1.0]``.

    Penalises departures before 06:00 and arrivals after 22:00, scaling with how far
    outside the window the time falls. A 04:00 departure scores worse than 05:30.

    Kept separate from duration because they are genuinely different complaints: a
    ten-hour daytime journey and a four-hour overnight one are unpleasant for
    unrelated reasons.
    """
    departure_minute = offer.departure_at.hour * 60 + offer.departure_at.minute

    # Arrival is measured relative to the *departure date*, not as a wall-clock time.
    # Taking the clock time alone made a journey arriving at 00:30 the following morning
    # look like an enviably early arrival (minute 30) instead of a brutal one (minute
    # 1470) — overnight coaches escaped the penalty entirely.
    day_offset = (offer.arrival_at.date() - offer.departure_at.date()).days
    arrival_minute = offer.arrival_at.hour * 60 + offer.arrival_at.minute + day_offset * 24 * 60

    early_penalty = 0.0
    if departure_minute < EARLY_DEPARTURE_MINUTE:
        early_penalty = (EARLY_DEPARTURE_MINUTE - departure_minute) / EARLY_DEPARTURE_MINUTE

    late_penalty = 0.0
    if arrival_minute > LATE_ARRIVAL_MINUTE:
        late_penalty = (arrival_minute - LATE_ARRIVAL_MINUTE) / (24 * 60 - LATE_ARRIVAL_MINUTE)

    return min(1.0, early_penalty + late_penalty)


def _flexibility_penalty(offer: AccommodationOffer) -> float:
    """``0.0`` for free cancellation, ``1.0`` for none. Unknown terms are treated as
    inflexible — assuming the better case would overstate what we actually know."""
    if offer.cancellation is None:
        return 1.0
    return 0.0 if offer.cancellation.free_cancellation else 1.0


def _rating_penalty(offer: AccommodationOffer, *, unrated_default: float) -> float:
    """Rating inverted into a penalty, since lower scores are better here.

    Unrated properties receive a mid-range default rather than the best or worst possible
    value: treating "no reviews yet" as either excellent or terrible would both be
    fabrications.
    """
    if offer.rating is None:
        return unrated_default
    return 1.0 - (offer.rating / 10.0)


# ---------------------------------------------------------------------------
# Filtering — hard constraints
# ---------------------------------------------------------------------------
def filter_transport(
    offers: list[TransportOffer],
    *,
    max_duration_hours: float,
    max_transfers: int,
    accessibility_needs: list[AccessibilityNeed],
) -> tuple[list[TransportOffer], list[str]]:
    """Split offers into those meeting every hard constraint and reasons for the rest.

    Returns the surviving offers plus human-readable rejection reasons, so the user can be
    told *why* only two options came back rather than being left to guess.
    """
    kept: list[TransportOffer] = []
    reasons: list[str] = []

    for offer in offers:
        longest_journey = max(offer.total_duration_minutes, offer.return_duration_minutes)
        most_transfers = max(offer.transfer_count, offer.return_transfer_count)
        if longest_journey > max_duration_hours * 60:
            hours = longest_journey / 60
            reasons.append(
                f"{offer.offer_id}: {hours:.1f} h journey exceeds the "
                f"{max_duration_hours:g} h limit"
            )
            continue
        if most_transfers > max_transfers:
            reasons.append(
                f"{offer.offer_id}: {most_transfers} transfers exceeds the limit of {max_transfers}"
            )
            continue
        missing = [n for n in accessibility_needs if n not in offer.accessibility_features]
        if missing:
            reasons.append(
                f"{offer.offer_id}: does not provide {', '.join(n.value for n in missing)}"
            )
            continue
        kept.append(offer)

    return kept, reasons


def filter_accommodation(
    offers: list[AccommodationOffer],
    *,
    accessibility_needs: list[AccessibilityNeed],
    max_total_price: Decimal | None = None,
) -> tuple[list[AccommodationOffer], list[str]]:
    """As :func:`filter_transport`, for accommodation."""
    kept: list[AccommodationOffer] = []
    reasons: list[str] = []

    for offer in offers:
        missing = [n for n in accessibility_needs if n not in offer.accessibility_features]
        if missing:
            reasons.append(f"{offer.name}: does not provide {', '.join(n.value for n in missing)}")
            continue
        if max_total_price is not None and offer.total_price.amount > max_total_price:
            reasons.append(f"{offer.name}: {offer.total_price} exceeds the accommodation allowance")
            continue
        kept.append(offer)

    return kept, reasons


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def rank_transport(offers: list[TransportOffer], strategy: RankingStrategy) -> list[TransportOffer]:
    """Score and order transport offers, best first.

    Returns new offers carrying ``score`` and ``score_breakdown``; the inputs are frozen
    and unmodified. The breakdown is surfaced to the user as the trade-off explanation, so
    the ranking is inspectable rather than asserted.
    """
    if not offers:
        return []

    weights = transport_weights_for(strategy)

    prices = [float(o.total_price.amount) for o in offers]
    durations = [float(o.total_duration_minutes + o.return_duration_minutes) for o in offers]
    transfers = [float(o.transfer_count + o.return_transfer_count) for o in offers]
    inconveniences = [schedule_inconvenience(o) for o in offers]

    price_lo, price_hi = min(prices), max(prices)
    duration_lo, duration_hi = min(durations), max(durations)
    transfer_lo, transfer_hi = min(transfers), max(transfers)
    inconvenience_lo, inconvenience_hi = min(inconveniences), max(inconveniences)

    scored: list[TransportOffer] = []
    for offer, price, duration, transfer, inconvenience in zip(
        offers, prices, durations, transfers, inconveniences, strict=True
    ):
        breakdown = {
            "price": round(weights.price * normalise(price, price_lo, price_hi), 6),
            "duration": round(weights.duration * normalise(duration, duration_lo, duration_hi), 6),
            "transfers": round(
                weights.transfers * normalise(transfer, transfer_lo, transfer_hi), 6
            ),
            "inconvenience": round(
                weights.inconvenience
                * normalise(inconvenience, inconvenience_lo, inconvenience_hi),
                6,
            ),
        }
        total = round(sum(breakdown.values()), 6)
        scored.append(offer.model_copy(update={"score": total, "score_breakdown": breakdown}))

    # offer_id breaks ties so that equally-scored options keep a stable, reproducible
    # order — without it, two identical offers could swap places between runs.
    return sorted(scored, key=lambda o: (o.score if o.score is not None else 1.0, o.offer_id))


def rank_accommodation(
    offers: list[AccommodationOffer], strategy: RankingStrategy
) -> list[AccommodationOffer]:
    """Score and order accommodation offers, best first."""
    if not offers:
        return []

    weights = accommodation_weights_for(strategy)

    prices = [float(o.total_price.amount) for o in offers]
    # An unknown distance is treated as the worst observed rather than zero: an unlocated
    # property must not win the distance term by default.
    known_distances = [
        o.distance_to_centre_km for o in offers if o.distance_to_centre_km is not None
    ]
    worst_distance = max(known_distances) if known_distances else 0.0
    distances = [
        o.distance_to_centre_km if o.distance_to_centre_km is not None else worst_distance
        for o in offers
    ]

    rated = [o.rating for o in offers if o.rating is not None]
    unrated_default = 1.0 - ((sum(rated) / len(rated)) / 10.0) if rated else 0.5
    ratings = [_rating_penalty(o, unrated_default=unrated_default) for o in offers]
    flexibilities = [_flexibility_penalty(o) for o in offers]

    price_lo, price_hi = min(prices), max(prices)
    distance_lo, distance_hi = min(distances), max(distances)
    rating_lo, rating_hi = min(ratings), max(ratings)
    flex_lo, flex_hi = min(flexibilities), max(flexibilities)

    scored: list[AccommodationOffer] = []
    for offer, price, distance, rating, flexibility in zip(
        offers, prices, distances, ratings, flexibilities, strict=True
    ):
        breakdown = {
            "price": round(weights.price * normalise(price, price_lo, price_hi), 6),
            "distance": round(weights.distance * normalise(distance, distance_lo, distance_hi), 6),
            "rating": round(weights.rating * normalise(rating, rating_lo, rating_hi), 6),
            "flexibility": round(weights.flexibility * normalise(flexibility, flex_lo, flex_hi), 6),
        }
        total = round(sum(breakdown.values()), 6)
        scored.append(offer.model_copy(update={"score": total, "score_breakdown": breakdown}))

    return sorted(scored, key=lambda o: (o.score if o.score is not None else 1.0, o.offer_id))
