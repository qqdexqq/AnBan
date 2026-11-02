"""Read-only projection of ledger debits into the sandbox's minimal schema."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from bank.categories import CATEGORIES


def analysis_rows(bank: Any, start: date | datetime, end: date | datetime) -> list[dict]:
    """Return spending rows, excluding income and transfers to owned savings.

    Counterparty is examined in trusted application code and never copied to the
    sandbox. Use the bank's unbounded customer-scoped query, not its default
    recent-transactions limit. No writes or clock reads occur here.
    """
    result = []
    for row in bank.list_transactions(start, end, limit=None):
        if row.amount_fen >= 0:
            continue
        if row.category == "transfer" and row.counterparty == "我的储蓄账户":
            continue
        result.append({
            "day": row.ts.date().isoformat(),
            "weekday": row.ts.weekday(),
            "hour": row.ts.hour,
            "amount_fen": -row.amount_fen,
            "category": row.category if row.category in CATEGORIES else "other",
        })
    return result
