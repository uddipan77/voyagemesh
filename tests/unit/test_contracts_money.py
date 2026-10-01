"""Money arithmetic and the currency-safety invariants budgets depend on."""

from __future__ import annotations

from decimal import Decimal

import pytest

from vm_contracts import Currency, Money

pytestmark = pytest.mark.unit

EUR = Currency.EUR
GBP = Currency.GBP


class TestConstruction:
    def test_quantizes_to_cents(self):
        assert Money.of("120.005", EUR).amount == Decimal("120.01")
        assert Money.of("120.004", EUR).amount == Decimal("120.00")

    def test_equality_ignores_trailing_precision(self):
        assert Money.of("10.5", EUR) == Money.of("10.50", EUR)

    def test_accepts_int_and_str(self):
        assert Money.of(30, EUR).amount == Decimal("30.00")
        assert Money.of("30", EUR).amount == Decimal("30.00")

    def test_zero_helper(self):
        assert Money.zero(EUR).amount == Decimal("0.00")

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            Money(amount=Decimal("-1.00"), currency=EUR)

    def test_different_currencies_are_not_equal(self):
        assert Money.of(10, EUR) != Money.of(10, GBP)

    def test_is_hashable(self):
        """Money is used in sets and as dict keys during de-duplication."""
        assert len({Money.of("10.50", EUR), Money.of("10.5", EUR)}) == 1


class TestArithmetic:
    def test_add_and_multiply(self):
        assert (Money.of(100, EUR) + Money.of("20.50", EUR)).amount == Decimal("120.50")
        assert Money.of("42.50", EUR).times(3).amount == Decimal("127.50")

    def test_times_zero_is_valid(self):
        """A zero-night stay costs nothing — this is a real day-trip case."""
        assert Money.of(50, EUR).times(0) == Money.zero(EUR)

    def test_subtract_within_range(self):
        assert (Money.of(100, EUR) - Money.of(30, EUR)).amount == Decimal("70.00")

    def test_subtract_below_zero_raises_and_explains_alternative(self):
        with pytest.raises(ValueError, match="difference_or_zero"):
            Money.of(30, EUR) - Money.of(100, EUR)

    def test_difference_or_zero_floors(self):
        assert Money.of(30, EUR).difference_or_zero(Money.of(100, EUR)) == Money.zero(EUR)

    def test_decimal_precision_is_exact(self):
        """The float-arithmetic bug this type exists to prevent: 0.1+0.2 != 0.3."""
        total = Money.of("0.10", EUR) + Money.of("0.20", EUR)
        assert total.amount == Decimal("0.30")
        assert total == Money.of("0.30", EUR)

    def test_hundred_additions_do_not_drift(self):
        total = Money.zero(EUR)
        for _ in range(100):
            total = total + Money.of("0.01", EUR)
        assert total.amount == Decimal("1.00")


class TestCurrencySafety:
    @pytest.mark.parametrize(
        "operation",
        [
            lambda a, b: a + b,
            lambda a, b: a - b,
            lambda a, b: a.exceeds(b),
            lambda a, b: a < b,
            lambda a, b: a.difference_or_zero(b),
        ],
    )
    def test_mixed_currency_operations_all_raise(self, operation):
        with pytest.raises(ValueError, match=r"EUR|GBP"):
            operation(Money.of(10, EUR), Money.of(10, GBP))

    def test_error_names_both_currencies(self):
        with pytest.raises(ValueError) as exc:
            Money.of(10, EUR) + Money.of(10, GBP)
        assert "EUR" in str(exc.value) and "GBP" in str(exc.value)


class TestComparison:
    def test_exceeds_is_strict(self):
        budget = Money.of(350, EUR)
        assert Money.of("350.01", EUR).exceeds(budget) is True
        assert Money.of(350, EUR).exceeds(budget) is False

    def test_ordering(self):
        assert Money.of(10, EUR) < Money.of(20, EUR)
        assert Money.of(10, EUR) <= Money.of(10, EUR)

    def test_str_always_includes_currency(self):
        """Guardrail requirement: no price is ever rendered without its currency."""
        assert str(Money.of("42.5", EUR)) == "42.50 EUR"


class TestImmutability:
    def test_money_is_frozen(self):
        amount = Money.of(10, EUR)
        with pytest.raises(ValueError):
            amount.amount = Decimal("999.00")

    def test_arithmetic_returns_new_instances(self):
        original = Money.of(10, EUR)
        assert (original + Money.of(5, EUR)) is not original
        assert original.amount == Decimal("10.00")
