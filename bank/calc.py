"""Deterministic banking calculators: instalments, deposits and cashflow."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta


def installment_quote(
    principal_fen: int,
    periods: int,
    fee_rate_per_period_pct: float,
) -> dict:
    """Quote a Chinese credit-card instalment plan.

    The fee is charged on the original principal every period, so the real
    annual rate (APR) is well above the nominal one.
    """
    if principal_fen <= 0:
        raise ValueError("principal_fen must be > 0")
    if not 1 <= periods <= 60:
        raise ValueError("periods must be between 1 and 60")
    if fee_rate_per_period_pct < 0:
        raise ValueError("fee_rate_per_period_pct must be >= 0")

    fee_per_period = principal_fen * fee_rate_per_period_pct / 100
    payment = round(principal_fen / periods + fee_per_period)
    total_fee = round(fee_per_period * periods)

    lo, hi = 0.0, 1.0
    while hi - lo > 1e-10:
        mid = (lo + hi) / 2
        present = sum(payment / (1 + mid) ** k for k in range(1, periods + 1))
        if present > principal_fen:
            lo = mid
        else:
            hi = mid
    monthly_rate = (lo + hi) / 2

    return {
        "periods": periods,
        "payment_fen": payment,
        "total_fee_fen": total_fee,
        "nominal_annual_pct": round(fee_rate_per_period_pct * 12, 2),
        "apr_pct": round(monthly_rate * 12 * 100, 2),
    }


def deposit_interest(principal_fen: int, annual_rate_pct: float, days: int) -> int:
    """Simple interest using a 360-day year, rounded to the nearest fen."""
    if principal_fen < 0 or annual_rate_pct < 0 or days < 0:
        raise ValueError("principal, rate and days must be >= 0")
    return round(principal_fen * annual_rate_pct / 100 * days / 360)


def early_withdrawal(
    principal_fen: int,
    term_rate_pct: float,
    demand_rate_pct: float,
    days_held: int,
    term_days: int,
) -> dict:
    """Compare a matured deposit against withdrawing early at the demand rate."""
    if days_held < 0 or term_days < 0:
        raise ValueError("days must be >= 0")
    if days_held > term_days:
        raise ValueError("days_held cannot exceed term_days")

    if_held = deposit_interest(principal_fen, term_rate_pct, term_days)
    if_withdrawn = deposit_interest(principal_fen, demand_rate_pct, days_held)
    return {
        "if_held_fen": if_held,
        "if_withdrawn_now_fen": if_withdrawn,
        "lost_fen": if_held - if_withdrawn,
    }


def _payment(balance_fen: int, monthly_rate: float, months: int) -> float:
    """Equal-instalment (等额本息) payment for ``balance_fen`` over ``months``."""
    if monthly_rate == 0:
        return balance_fen / months
    return balance_fen * monthly_rate / (1 - (1 + monthly_rate) ** -months)


def prepay_quote(
    balance_fen: int,
    annual_rate_pct: float,
    months_left: int,
    prepay_fen: int,
    mode: str,
) -> dict:
    """Quote the effect of an early partial repayment on an equal-instalment loan."""
    if prepay_fen <= 0 or prepay_fen >= balance_fen:
        raise ValueError("prepay_fen must be > 0 and < balance_fen")
    if months_left < 1:
        raise ValueError("months_left must be >= 1")
    if mode not in ("lower_payment", "shorten_term"):
        raise ValueError("mode must be 'lower_payment' or 'shorten_term'")

    monthly_rate = annual_rate_pct / 12 / 100
    payment_before = _payment(balance_fen, monthly_rate, months_left)
    total_before = payment_before * months_left - balance_fen

    remaining = balance_fen - prepay_fen
    if mode == "lower_payment":
        months_after = months_left
        payment_after = _payment(remaining, monthly_rate, months_after)
    else:
        if monthly_rate == 0:
            months_after = math.ceil(remaining / payment_before)
        else:
            months_after = math.ceil(
                -math.log(1 - remaining * monthly_rate / payment_before)
                / math.log(1 + monthly_rate)
            )
        payment_after = _payment(remaining, monthly_rate, months_after)

    total_after = payment_after * months_after - remaining
    return {
        "mode": mode,
        "payment_before_fen": round(payment_before),
        "payment_after_fen": round(payment_after),
        "months_before": months_left,
        "months_after": months_after,
        "interest_saved_fen": round(total_before - total_after),
    }


def cashflow_forecast(
    balance_fen: int,
    recurring: list[tuple[int, int]],
    days: int,
    today_day_of_month: int,
) -> dict:
    """Forecast a balance over ``days`` using a 28-day day-of-month cycle."""
    by_day: dict[int, int] = {}
    for day_of_month, amount_fen in recurring:
        by_day[day_of_month] = by_day.get(day_of_month, 0) + amount_fen

    balance = balance_fen
    lowest = balance_fen
    lowest_offset = 0
    day = today_day_of_month

    for offset in range(1, days + 1):
        day = day % 28 + 1
        balance += by_day.get(day, 0)
        if balance < lowest:
            lowest = balance
            lowest_offset = offset

    return {
        "end_balance_fen": balance,
        "lowest_balance_fen": lowest,
        "lowest_on_day_offset": lowest_offset,
        "goes_negative": lowest < 0,
    }


def _require_fen(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("amounts must be integer fen")
    return value


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        first_next = date(year + 1, 1, 1)
    else:
        first_next = date(year, month + 1, 1)
    return (first_next - timedelta(days=1)).day


def calendar_cashflow_forecast(
    balance_fen: int,
    recurring: list[tuple[int, int]],
    one_off: list[tuple[date, int]],
    today: date,
    days: int,
) -> dict:
    """Forecast a balance using real calendar dates.

    ``recurring`` holds ``(day_of_month, amount_fen)`` items that occur once per
    month strictly after ``today`` and no later than ``today + days``.  A day
    past the end of a month (e.g. 31 in February) falls on the actual month end.
    ``one_off`` holds ``(date, amount_fen)`` items and is counted once on its
    date, including ``today``.  Offsets are whole days from ``today``.
    """
    _require_fen(balance_fen)
    if not isinstance(days, int) or isinstance(days, bool) or days < 0:
        raise ValueError("days must be a nonnegative integer")
    if isinstance(today, datetime):
        today = today.date()
    if not isinstance(today, date):
        raise ValueError("today must be a date")

    recurring_by_day: dict[int, int] = {}
    for day_of_month, amount_fen in recurring:
        if (
            not isinstance(day_of_month, int)
            or isinstance(day_of_month, bool)
            or not 1 <= day_of_month <= 31
        ):
            raise ValueError("day_of_month must be an integer between 1 and 31")
        amount = _require_fen(amount_fen)
        recurring_by_day[day_of_month] = (
            recurring_by_day.get(day_of_month, 0) + amount
        )

    one_off_by_date: dict[date, int] = {}
    for when, amount_fen in one_off:
        if isinstance(when, datetime):
            when = when.date()
        if not isinstance(when, date):
            raise ValueError("one-off dates must be date objects")
        amount = _require_fen(amount_fen)
        one_off_by_date[when] = one_off_by_date.get(when, 0) + amount

    balance = balance_fen
    lowest = balance_fen
    lowest_offset = 0
    end = today + timedelta(days=days)

    current = today
    offset = 0
    while current <= end:
        change = one_off_by_date.get(current, 0)
        if offset:
            month_end = _days_in_month(current.year, current.month)
            for day_of_month, amount_fen in recurring_by_day.items():
                if current.day == min(day_of_month, month_end):
                    change += amount_fen
        balance += change
        if balance < lowest:
            lowest = balance
            lowest_offset = offset
        current += timedelta(days=1)
        offset += 1

    return {
        "end_balance_fen": balance,
        "lowest_balance_fen": lowest,
        "lowest_on_day_offset": lowest_offset,
        "goes_negative": lowest < 0,
    }
