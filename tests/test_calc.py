"""Tests for the deterministic banking calculators in bank.calc."""

from datetime import date

import pytest

from bank.calc import (
    calendar_cashflow_forecast,
    cashflow_forecast,
    deposit_interest,
    early_withdrawal,
    installment_quote,
    prepay_quote,
)


def test_installment_quote_example() -> None:
    quote = installment_quote(1_200_000, 12, 0.6)
    assert quote["periods"] == 12
    assert quote["payment_fen"] == 107_200
    assert quote["total_fee_fen"] == 86_400
    assert quote["nominal_annual_pct"] == 7.2
    assert 12.9 <= quote["apr_pct"] <= 13.3


def test_installment_quote_zero_fee() -> None:
    quote = installment_quote(1_200_000, 12, 0.0)
    assert quote["payment_fen"] == 100_000
    assert quote["total_fee_fen"] == 0
    assert quote["nominal_annual_pct"] == 0.0
    assert quote["apr_pct"] == 0.0


def test_installment_quote_single_period_apr() -> None:
    quote = installment_quote(100_000, 1, 1.0)
    assert quote["payment_fen"] == 101_000
    assert quote["apr_pct"] == 12.0


@pytest.mark.parametrize(
    ("principal", "periods", "fee"),
    [(0, 12, 0.6), (1_000, 0, 0.6), (1_000, 61, 0.6), (1_000, 12, -0.1)],
)
def test_installment_quote_validation(principal: int, periods: int, fee: float) -> None:
    with pytest.raises(ValueError):
        installment_quote(principal, periods, fee)


def test_deposit_interest_example() -> None:
    assert deposit_interest(10_000_000, 1.55, 360) == 155_000
    assert deposit_interest(100_000, 1.55, 180) == 775


@pytest.mark.parametrize(
    ("principal", "rate", "days"),
    [(-1, 1.55, 360), (10_000, -1.0, 360), (10_000, 1.55, -1)],
)
def test_deposit_interest_validation(principal: int, rate: float, days: int) -> None:
    with pytest.raises(ValueError):
        deposit_interest(principal, rate, days)


def test_early_withdrawal_example() -> None:
    result = early_withdrawal(10_000_000, 2.0, 0.2, 100, 360)
    assert result["if_held_fen"] == 200_000
    assert result["if_withdrawn_now_fen"] == 5_556
    assert result["lost_fen"] == 194_444


def test_early_withdrawal_validation() -> None:
    with pytest.raises(ValueError):
        early_withdrawal(10_000_000, 2.0, 0.2, 400, 360)
    with pytest.raises(ValueError):
        early_withdrawal(10_000_000, 2.0, 0.2, -1, 360)


def test_prepay_lower_payment() -> None:
    quote = prepay_quote(100_000_000, 4.0, 240, 20_000_000, "lower_payment")
    assert quote["mode"] == "lower_payment"
    assert quote["months_before"] == 240
    assert quote["months_after"] == 240
    assert quote["payment_after_fen"] < quote["payment_before_fen"]
    assert quote["interest_saved_fen"] > 0


def test_prepay_shorten_term() -> None:
    lower = prepay_quote(100_000_000, 4.0, 240, 20_000_000, "lower_payment")
    shorter = prepay_quote(100_000_000, 4.0, 240, 20_000_000, "shorten_term")
    assert shorter["months_after"] < 240
    assert shorter["payment_after_fen"] <= shorter["payment_before_fen"]
    assert shorter["interest_saved_fen"] > lower["interest_saved_fen"]


def test_prepay_zero_rate() -> None:
    lower = prepay_quote(12_000_000, 0.0, 12, 6_000_000, "lower_payment")
    assert lower["payment_before_fen"] == 1_000_000
    assert lower["payment_after_fen"] == 500_000
    assert lower["interest_saved_fen"] == 0

    shorter = prepay_quote(12_000_000, 0.0, 12, 6_000_000, "shorten_term")
    assert shorter["months_after"] == 6
    assert shorter["payment_after_fen"] == 1_000_000
    assert shorter["interest_saved_fen"] == 0


@pytest.mark.parametrize(
    ("prepay", "months", "mode"),
    [(0, 240, "lower_payment"), (100_000_000, 240, "lower_payment"),
     (20_000_000, 0, "lower_payment"), (20_000_000, 240, "other")],
)
def test_prepay_validation(prepay: int, months: int, mode: str) -> None:
    with pytest.raises(ValueError):
        prepay_quote(100_000_000, 4.0, months, prepay, mode)


def test_cashflow_salary_and_rent() -> None:
    result = cashflow_forecast(
        500_000,
        [(25, 1_500_000), (1, -300_000)],
        15,
        24,
    )
    assert result["end_balance_fen"] == 1_700_000
    assert result["lowest_balance_fen"] == 500_000
    assert result["lowest_on_day_offset"] == 0
    assert result["goes_negative"] is False


def test_cashflow_goes_negative() -> None:
    result = cashflow_forecast(100_000, [(1, -150_000)], 10, 24)
    assert result["end_balance_fen"] == -50_000
    assert result["lowest_balance_fen"] == -50_000
    assert result["lowest_on_day_offset"] == 5
    assert result["goes_negative"] is True


def test_cashflow_repeats_monthly() -> None:
    result = cashflow_forecast(1_000_000, [(1, -100_000)], 60, 1)
    assert result["end_balance_fen"] == 800_000
    assert result["lowest_balance_fen"] == 800_000
    assert result["lowest_on_day_offset"] == 56
    assert result["goes_negative"] is False


def test_calendar_cashflow_one_off_today_counts() -> None:
    result = calendar_cashflow_forecast(
        100_000,
        [],
        [(date(2026, 10, 17), -40_000)],
        date(2026, 10, 17),
        0,
    )
    assert result["end_balance_fen"] == 60_000
    assert result["lowest_balance_fen"] == 60_000
    assert result["lowest_on_day_offset"] == 0


def test_calendar_cashflow_actual_30_and_31_day_months() -> None:
    october = calendar_cashflow_forecast(
        100_000,
        [],
        [(date(2026, 10, 31), -1_000)],
        date(2026, 10, 1),
        30,
    )
    assert october["end_balance_fen"] == 99_000
    assert october["lowest_on_day_offset"] == 30

    november = calendar_cashflow_forecast(
        100_000,
        [],
        [(date(2026, 11, 30), -1_000)],
        date(2026, 11, 1),
        30,
    )
    assert november["end_balance_fen"] == 99_000
    assert november["lowest_on_day_offset"] == 29


@pytest.mark.parametrize(
    ("today", "due", "days"),
    [
        (date(2024, 2, 1), date(2024, 2, 29), 28),
        (date(2025, 2, 1), date(2025, 2, 28), 27),
    ],
)
def test_calendar_cashflow_february_leap_and_non_leap(
    today: date, due: date, days: int
) -> None:
    result = calendar_cashflow_forecast(100_000, [], [(due, -1_000)], today, days)
    assert result["end_balance_fen"] == 99_000
    assert result["lowest_on_day_offset"] == (due - today).days


def test_calendar_cashflow_one_off_bill_is_not_repeated() -> None:
    result = calendar_cashflow_forecast(
        1_000_000,
        [],
        [(date(2026, 1, 15), -300_000)],
        date(2026, 1, 1),
        60,
    )
    assert result["end_balance_fen"] == 700_000


def test_calendar_cashflow_recurring_day_31_clamps_to_month_end() -> None:
    result = calendar_cashflow_forecast(
        1_000_000, [(31, -100_000)], [], date(2026, 1, 1), 59
    )
    # Jan 31 and Feb 28 fall inside the window; Mar 31 does not.
    assert result["end_balance_fen"] == 800_000
    assert result["lowest_on_day_offset"] == 58


def test_calendar_cashflow_recurring_waits_until_after_today() -> None:
    result = calendar_cashflow_forecast(
        1_000_000, [(15, -100_000)], [], date(2026, 1, 15), 30
    )
    # The next 15th is Feb 15, past Jan 15 + 30 days.
    assert result["end_balance_fen"] == 1_000_000


def test_calendar_cashflow_validation() -> None:
    today = date(2026, 1, 1)
    with pytest.raises(ValueError):
        calendar_cashflow_forecast(100_000, [], [], today, -1)
    with pytest.raises(ValueError):
        calendar_cashflow_forecast(100_000, [(0, -1_000)], [], today, 10)
    with pytest.raises(ValueError):
        calendar_cashflow_forecast(100_000, [(1, -1.5)], [], today, 10)
    with pytest.raises(ValueError):
        calendar_cashflow_forecast(
            100_000, [], [(today, "not-fen")], today, 10
        )
