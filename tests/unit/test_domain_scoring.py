"""Deterministic scoring, filtering, and geo calculations.

The reproducibility tests are the important ones: they are what make ADR-009's claim —
that ranking is a pure, auditable function rather than an LLM judgement — actually true
rather than merely asserted.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta

import pytest

from vm_contracts import (
    AccessibilityNeed,
    AccommodationOffer,
    AccommodationType,
    CancellationPolicy,
    Currency,
    GeoPoint,
    Money,
    Provenance,
    RankingStrategy,
    TransportLeg,
    TransportMode,
    TransportOffer,
)
from vm_domain.geo import (
    distance_km,
    estimate_travel_minutes,
    normalise,
)
from vm_domain.providers.base import AccommodationQuery, TransportQuery
from vm_domain.providers.mock_accommodation import MockAccommodationProvider
from vm_domain.providers.mock_transport import MockTransportProvider
from vm_domain.scoring import (
    accommodation_weights_for,
    filter_accommodation,
    filter_transport,
    rank_accommodation,
    rank_transport,
    schedule_inconvenience,
    transport_weights_for,
)

pytestmark = pytest.mark.unit

EUR = Currency.EUR
BASE = datetime(2026, 8, 10, 8, 0, tzinfo=UTC)

PRAGUE = GeoPoint(latitude=50.0875, longitude=14.4213)
NUREMBERG = GeoPoint(latitude=49.4539, longitude=11.0775)
CHARLES_BRIDGE = GeoPoint(latitude=50.0865, longitude=14.4114)


def transport(
    offer_id: str,
    *,
    price: str,
    hours: float,
    transfers: int = 0,
    depart_hour: int = 9,
    accessibility: list[AccessibilityNeed] | None = None,
) -> TransportOffer:
    """Build a transport offer with an exact price/duration/transfer profile."""
    departure = BASE.replace(hour=depart_hour)
    total_minutes = int(hours * 60)
    legs: list[TransportLeg] = []
    segments = transfers + 1
    per_segment = total_minutes // segments
    cursor = departure
    for index in range(segments):
        arrival = cursor + timedelta(minutes=per_segment)
        legs.append(
            TransportLeg(
                mode=TransportMode.TRAIN,
                carrier="Test Rail",
                from_location=f"Stop {index}",
                to_location=f"Stop {index + 1}",
                departure_at=cursor,
                arrival_at=arrival,
            )
        )
        cursor = arrival
    # Stretch the final leg so total duration is exact regardless of integer division.
    if legs[-1].arrival_at != departure + timedelta(minutes=total_minutes):
        legs[-1] = legs[-1].model_copy(
            update={"arrival_at": departure + timedelta(minutes=total_minutes)}
        )
    return TransportOffer(
        offer_id=offer_id,
        legs=legs,
        total_price=Money.of(price, EUR),
        price_per_traveller=Money.of(price, EUR),
        travellers=1,
        provenance=Provenance.mocked("test"),
        accessibility_features=accessibility or [],
    )


def stay(
    offer_id: str,
    *,
    nightly: str,
    nights: int = 3,
    distance: float | None = 1.0,
    rating: float | None = 8.0,
    free_cancellation: bool = True,
    accessibility: list[AccessibilityNeed] | None = None,
) -> AccommodationOffer:
    return AccommodationOffer(
        offer_id=offer_id,
        name=f"Stay {offer_id}",
        accommodation_type=AccommodationType.HOSTEL,
        price_per_night=Money.of(nightly, EUR),
        total_price=Money.of(nightly, EUR).times(nights),
        nights=nights,
        guests=1,
        distance_to_centre_km=distance,
        rating=rating,
        cancellation=CancellationPolicy(
            free_cancellation=free_cancellation,
            free_cancellation_until=BASE if free_cancellation else None,
        ),
        accessibility_features=accessibility or [],
        provenance=Provenance.mocked("test"),
    )


class TestGeo:
    def test_known_distance_is_accurate(self):
        """Nuremberg to Prague is roughly 250 km as the crow flies."""
        assert 240 < distance_km(NUREMBERG, PRAGUE) < 260

    def test_distance_is_symmetric_and_zero_for_identical_points(self):
        assert distance_km(PRAGUE, NUREMBERG) == pytest.approx(distance_km(NUREMBERG, PRAGUE))
        assert distance_km(PRAGUE, PRAGUE) == pytest.approx(0.0, abs=1e-9)

    def test_antipodal_points_do_not_raise(self):
        """Floating-point error at the antipodes must not escape as a ValueError."""
        north = GeoPoint(latitude=90.0, longitude=0.0)
        south = GeoPoint(latitude=-90.0, longitude=0.0)
        assert distance_km(north, south) == pytest.approx(20015.0, rel=0.01)

    def test_short_hop_is_walked(self):
        minutes = estimate_travel_minutes(PRAGUE, CHARLES_BRIDGE)
        assert 1 <= minutes <= 20

    def test_unknown_coordinates_return_a_conservative_default(self):
        """Zero would let the day validator accept a schedule that cannot be walked."""
        assert estimate_travel_minutes(None, PRAGUE) == 20
        assert estimate_travel_minutes(PRAGUE, None) == 20

    def test_low_walking_need_switches_to_transit_sooner(self):
        far = GeoPoint(latitude=50.1000, longitude=14.4213)
        normal = estimate_travel_minutes(PRAGUE, far)
        assisted = estimate_travel_minutes(PRAGUE, far, low_walking_distance=True)
        assert assisted != normal

    def test_normalise_returns_zero_when_all_values_identical(self):
        """An attribute with no variance carries no information and must not shift scores."""
        assert normalise(5.0, 5.0, 5.0) == 0.0

    def test_normalise_clamps_and_scales(self):
        assert normalise(0.0, 0.0, 10.0) == 0.0
        assert normalise(10.0, 0.0, 10.0) == 1.0
        assert normalise(5.0, 0.0, 10.0) == 0.5
        assert normalise(99.0, 0.0, 10.0) == 1.0


class TestWeights:
    @pytest.mark.parametrize("strategy", list(RankingStrategy))
    def test_all_weight_sets_sum_to_one(self, strategy):
        transport_weights_for(strategy)  # constructor validates the sum
        accommodation_weights_for(strategy)

    def test_strategies_have_genuinely_different_priorities(self):
        cheapest = transport_weights_for(RankingStrategy.CHEAPEST)
        fastest = transport_weights_for(RankingStrategy.FASTEST)
        assert cheapest.price > fastest.price
        assert fastest.duration > cheapest.duration


class TestScheduleInconvenience:
    def test_civilised_times_score_zero(self):
        assert schedule_inconvenience(transport("t", price="50", hours=4, depart_hour=10)) == 0.0

    def test_early_departure_is_penalised_proportionally(self):
        at_five = schedule_inconvenience(transport("t", price="50", hours=2, depart_hour=5))
        at_four = schedule_inconvenience(transport("t", price="50", hours=2, depart_hour=4))
        assert 0 < at_five < at_four <= 1.0

    def test_late_arrival_is_penalised(self):
        late = transport("t", price="50", hours=3, depart_hour=20)  # arrives 23:00
        assert schedule_inconvenience(late) > 0

    def test_overnight_arrival_after_midnight_is_penalised_not_rewarded(self):
        """Regression: clock-time-only arithmetic read a 00:30 arrival as minute 30,
        making an overnight coach look like a pleasantly early arrival."""
        overnight = transport("t", price="50", hours=5, depart_hour=19)  # arrives 00:00
        deeper = transport("t", price="50", hours=8, depart_hour=19)  # arrives 03:00
        assert schedule_inconvenience(overnight) > 0
        assert schedule_inconvenience(deeper) >= schedule_inconvenience(overnight)

    def test_overnight_arrival_scores_worse_than_same_day_evening(self):
        evening = transport("t", price="50", hours=3, depart_hour=18)  # arrives 21:00
        overnight = transport("t", price="50", hours=7, depart_hour=18)  # arrives 01:00
        assert schedule_inconvenience(overnight) > schedule_inconvenience(evening)

    def test_result_is_bounded(self):
        awful = transport("t", price="50", hours=19, depart_hour=4)
        assert 0.0 <= schedule_inconvenience(awful) <= 1.0


class TestRankTransport:
    def _candidates(self) -> list[TransportOffer]:
        return [
            transport("cheap-slow", price="15", hours=9, transfers=2, depart_hour=5),
            transport("fast-expensive", price="120", hours=3, transfers=0, depart_hour=9),
            transport("middle", price="45", hours=5, transfers=1, depart_hour=8),
        ]

    def test_cheapest_strategy_picks_the_cheapest(self):
        assert rank_transport(self._candidates(), RankingStrategy.CHEAPEST)[0].offer_id == (
            "cheap-slow"
        )

    def test_fastest_strategy_picks_the_fastest(self):
        assert rank_transport(self._candidates(), RankingStrategy.FASTEST)[0].offer_id == (
            "fast-expensive"
        )

    def test_fewest_transfers_picks_the_direct_option(self):
        winner = rank_transport(self._candidates(), RankingStrategy.FEWEST_TRANSFERS)[0]
        assert winner.transfer_count == 0

    def test_balanced_avoids_both_extremes(self):
        """Balanced should reject the 9-hour 05:00 slog and the 120 EUR premium."""
        assert rank_transport(self._candidates(), RankingStrategy.BALANCED)[0].offer_id == "middle"

    def test_scores_are_attached_and_breakdown_sums_to_total(self):
        for offer in rank_transport(self._candidates(), RankingStrategy.BALANCED):
            assert offer.score is not None
            assert 0.0 <= offer.score <= 1.0
            assert sum(offer.score_breakdown.values()) == pytest.approx(offer.score, abs=1e-6)

    def test_ranking_is_reproducible(self):
        """ADR-009's core claim: identical inputs, identical order, every time."""
        candidates = self._candidates()
        first = [o.offer_id for o in rank_transport(candidates, RankingStrategy.BALANCED)]
        for _ in range(20):
            assert [
                o.offer_id for o in rank_transport(candidates, RankingStrategy.BALANCED)
            ] == first

    def test_ranking_is_independent_of_input_order(self):
        candidates = self._candidates()
        forward = [o.offer_id for o in rank_transport(candidates, RankingStrategy.BALANCED)]
        backward = [
            o.offer_id for o in rank_transport(list(reversed(candidates)), RankingStrategy.BALANCED)
        ]
        assert forward == backward

    def test_identical_offers_are_tie_broken_stably_by_id(self):
        pair = [
            transport("bbb", price="50", hours=4),
            transport("aaa", price="50", hours=4),
        ]
        assert [o.offer_id for o in rank_transport(pair, RankingStrategy.BALANCED)] == [
            "aaa",
            "bbb",
        ]

    def test_single_candidate_scores_zero(self):
        """Nothing to compare against means no attribute carries information."""
        only = rank_transport([transport("only", price="999", hours=20)], RankingStrategy.BALANCED)
        assert only[0].score == 0.0

    def test_empty_input_returns_empty(self):
        assert rank_transport([], RankingStrategy.BALANCED) == []

    def test_inputs_are_not_mutated(self):
        candidates = self._candidates()
        rank_transport(candidates, RankingStrategy.BALANCED)
        assert all(offer.score is None for offer in candidates)


class TestFilterTransport:
    def test_duration_limit_excludes_and_explains(self):
        kept, reasons = filter_transport(
            [transport("long", price="20", hours=10), transport("ok", price="60", hours=4)],
            max_duration_hours=8,
            max_transfers=3,
            accessibility_needs=[],
        )
        assert [o.offer_id for o in kept] == ["ok"]
        assert "long" in reasons[0] and "exceeds" in reasons[0]

    def test_transfer_limit_excludes(self):
        kept, reasons = filter_transport(
            [transport("many", price="20", hours=4, transfers=3)],
            max_duration_hours=12,
            max_transfers=1,
            accessibility_needs=[],
        )
        assert kept == []
        assert "transfers" in reasons[0]

    def test_accessibility_is_a_hard_filter_not_a_penalty(self):
        accessible = transport(
            "yes", price="90", hours=6, accessibility=[AccessibilityNeed.STEP_FREE_ACCESS]
        )
        inaccessible = transport("no", price="10", hours=3)
        kept, reasons = filter_transport(
            [inaccessible, accessible],
            max_duration_hours=12,
            max_transfers=3,
            accessibility_needs=[AccessibilityNeed.STEP_FREE_ACCESS],
        )
        # The cheap, fast option is removed entirely rather than merely ranked lower.
        assert [o.offer_id for o in kept] == ["yes"]
        assert "step_free_access" in reasons[0]

    def test_boundary_duration_is_inclusive(self):
        kept, _ = filter_transport(
            [transport("exact", price="20", hours=8)],
            max_duration_hours=8,
            max_transfers=3,
            accessibility_needs=[],
        )
        assert len(kept) == 1


class TestRankAccommodation:
    def _candidates(self) -> list[AccommodationOffer]:
        return [
            stay("budget-far", nightly="18", distance=6.0, rating=7.0),
            stay("central-pricey", nightly="95", distance=0.3, rating=9.2),
            stay("middle", nightly="42", distance=1.5, rating=8.4),
        ]

    def test_cheapest_picks_lowest_total(self):
        assert rank_accommodation(self._candidates(), RankingStrategy.CHEAPEST)[0].offer_id == (
            "budget-far"
        )

    def test_fastest_prioritises_proximity_to_centre(self):
        assert rank_accommodation(self._candidates(), RankingStrategy.FASTEST)[0].offer_id == (
            "central-pricey"
        )

    def test_unknown_distance_does_not_win_by_default(self):
        """An unlocated property must not score best on distance simply for lacking data."""
        candidates = [
            stay("located", nightly="40", distance=0.2),
            stay("unlocated", nightly="40", distance=None),
        ]
        ranked = rank_accommodation(candidates, RankingStrategy.FASTEST)
        assert ranked[0].offer_id == "located"

    def test_unrated_property_is_treated_as_average_not_excellent(self):
        candidates = [
            stay("great", nightly="40", rating=9.5),
            stay("unrated", nightly="40", rating=None),
            stay("poor", nightly="40", rating=6.0),
        ]
        ranked = [o.offer_id for o in rank_accommodation(candidates, RankingStrategy.BALANCED)]
        assert ranked.index("unrated") == 1

    def test_free_cancellation_is_preferred_all_else_equal(self):
        candidates = [
            stay("rigid", nightly="40", free_cancellation=False),
            stay("flexible", nightly="40", free_cancellation=True),
        ]
        assert rank_accommodation(candidates, RankingStrategy.BALANCED)[0].offer_id == "flexible"

    def test_ranking_is_reproducible(self):
        candidates = self._candidates()
        first = [o.offer_id for o in rank_accommodation(candidates, RankingStrategy.BALANCED)]
        for _ in range(20):
            assert [
                o.offer_id for o in rank_accommodation(candidates, RankingStrategy.BALANCED)
            ] == first


class TestFilterAccommodation:
    def test_price_ceiling_excludes(self):
        from decimal import Decimal

        kept, reasons = filter_accommodation(
            [stay("pricey", nightly="90"), stay("ok", nightly="20")],
            accessibility_needs=[],
            max_total_price=Decimal("100"),
        )
        assert [o.offer_id for o in kept] == ["ok"]
        assert "allowance" in reasons[0]

    def test_accessibility_filter(self):
        kept, _ = filter_accommodation(
            [
                stay("plain", nightly="20"),
                stay(
                    "accessible",
                    nightly="20",
                    accessibility=[AccessibilityNeed.WHEELCHAIR_ACCESSIBLE],
                ),
            ],
            accessibility_needs=[AccessibilityNeed.WHEELCHAIR_ACCESSIBLE],
        )
        assert [o.offer_id for o in kept] == ["accessible"]


class TestMockProviderDeterminism:
    def _query(self, **overrides) -> TransportQuery:
        defaults = {
            "origin": "Nuremberg",
            "destination": "Prague",
            "departure_date": date(2026, 8, 10),
            "return_date": date(2026, 8, 13),
            "travellers": 1,
            "currency": EUR,
        }
        return TransportQuery(**{**defaults, **overrides})

    def test_same_query_produces_identical_offers(self):
        provider = MockTransportProvider()
        first = asyncio.run(provider.search(self._query()))
        second = asyncio.run(provider.search(self._query()))
        assert [o.offer_id for o in first.items] == [o.offer_id for o in second.items]
        assert [o.total_price for o in first.items] == [o.total_price for o in second.items]
        assert [o.departure_at for o in first.items] == [o.departure_at for o in second.items]

    def test_a_fresh_provider_instance_produces_the_same_offers(self):
        """No hidden instance state — determinism must survive a process restart."""
        first = asyncio.run(MockTransportProvider().search(self._query()))
        second = asyncio.run(MockTransportProvider().search(self._query()))
        assert [o.total_price for o in first.items] == [o.total_price for o in second.items]

    @pytest.mark.parametrize(
        "change",
        [
            {"destination": "Vienna"},
            {"departure_date": date(2026, 9, 1)},
            {"travellers": 3},
        ],
    )
    def test_different_queries_produce_different_offers(self, change):
        base = asyncio.run(MockTransportProvider().search(self._query()))
        other = asyncio.run(MockTransportProvider().search(self._query(**change)))
        assert [o.offer_id for o in base.items] != [o.offer_id for o in other.items]

    def test_every_offer_is_labelled_mocked_and_unbookable(self):
        result = asyncio.run(MockTransportProvider().search(self._query()))
        assert result.origin.value == "mocked"
        assert result.warnings
        for offer in result.items:
            assert offer.provenance.is_synthetic is True
            assert offer.provenance.display_label == "MOCKED"
            # A booking URL on fabricated inventory would be the worst possible lie.
            assert offer.booking_reference_url is None

    def test_party_pricing_scales_with_travellers(self):
        single = asyncio.run(MockTransportProvider().search(self._query(travellers=1)))
        family = asyncio.run(MockTransportProvider().search(self._query(travellers=4)))
        for offer in family.items:
            assert offer.total_price == offer.price_per_traveller.times(4)
        assert single.items[0].total_price != family.items[0].total_price

    def test_offers_are_internally_consistent(self):
        """Every generated offer must satisfy the contract's own validators."""
        result = asyncio.run(MockTransportProvider().search(self._query()))
        assert result.items
        for offer in result.items:
            assert offer.total_duration_minutes > 0
            assert offer.transfer_count == len(offer.legs) - 1
            assert offer.arrival_at > offer.departure_at


class TestMockAccommodationProvider:
    def _query(self, **overrides) -> AccommodationQuery:
        defaults = {
            "destination": "Prague",
            "check_in": date(2026, 8, 10),
            "check_out": date(2026, 8, 13),
            "guests": 1,
            "currency": EUR,
        }
        return AccommodationQuery(**{**defaults, **overrides})

    def test_same_query_produces_identical_offers(self):
        first = asyncio.run(MockAccommodationProvider().search(self._query()))
        second = asyncio.run(MockAccommodationProvider().search(self._query()))
        assert [o.offer_id for o in first.items] == [o.offer_id for o in second.items]
        assert [o.total_price for o in first.items] == [o.total_price for o in second.items]

    @pytest.mark.parametrize(
        "change",
        [{"destination": "Vienna"}, {"check_in": date(2026, 9, 1)}, {"guests": 3}],
    )
    def test_different_queries_produce_different_offers(self, change):
        base = asyncio.run(MockAccommodationProvider().search(self._query()))
        other = asyncio.run(MockAccommodationProvider().search(self._query(**change)))
        assert [o.offer_id for o in base.items] != [o.offer_id for o in other.items]

    def test_every_offer_is_labelled_mocked_and_unbookable(self):
        result = asyncio.run(MockAccommodationProvider().search(self._query()))
        assert result.origin.value == "mocked"
        assert result.warnings
        for offer in result.items:
            assert offer.provenance.is_synthetic is True
            assert offer.booking_reference_url is None

    def test_property_names_are_obviously_synthetic(self):
        """A realistic-looking name would undermine the MOCKED badge in the UI."""
        result = asyncio.run(MockAccommodationProvider().search(self._query()))
        assert all(offer.name.startswith("Mesh ") for offer in result.items)

    def test_total_price_matches_nightly_rate_times_nights(self):
        result = asyncio.run(MockAccommodationProvider().search(self._query()))
        assert result.items
        for offer in result.items:
            assert offer.nights == 3
            assert offer.total_price == offer.price_per_night.times(3)

    def test_day_trip_returns_no_offers_with_an_explanation(self):
        result = asyncio.run(
            MockAccommodationProvider().search(
                self._query(check_in=date(2026, 8, 10), check_out=date(2026, 8, 10))
            )
        )
        assert result.items == []
        assert result.warnings

    def test_type_preference_is_respected(self):
        result = asyncio.run(
            MockAccommodationProvider().search(
                self._query(accommodation_type=AccommodationType.HOSTEL)
            )
        )
        assert result.items
        assert all(o.accommodation_type is AccommodationType.HOSTEL for o in result.items)

    def test_unconstrained_search_spans_price_tiers(self):
        """A single tier would leave the ranking nothing meaningful to discriminate on."""
        result = asyncio.run(MockAccommodationProvider().search(self._query()))
        types = {offer.accommodation_type for offer in result.items}
        assert len(types) > 1

    def test_offers_are_placed_near_the_real_city_centre(self):
        result = asyncio.run(MockAccommodationProvider().search(self._query()))
        prague_centre = GeoPoint(latitude=50.0875, longitude=14.4213)
        for offer in result.items:
            assert offer.location is not None
            assert distance_km(prague_centre, offer.location) < 15

    def test_unknown_destination_still_produces_stable_offers(self):
        """The generator must work for any input, not just the curated demo cities."""
        first = asyncio.run(MockAccommodationProvider().search(self._query(destination="Zzyzx")))
        second = asyncio.run(MockAccommodationProvider().search(self._query(destination="Zzyzx")))
        assert first.items
        assert [o.total_price for o in first.items] == [o.total_price for o in second.items]
