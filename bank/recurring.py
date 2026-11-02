"""Read-only detection of recurring subscriptions and charges."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from statistics import median

_CATEGORY_KEYWORDS = (
    ("video", ("腾讯视频", "爱奇艺", "优酷", "b站", "bilibili", "芒果tv",
               "netflix", "youtube", "disney", "视频")),
    ("music", ("qq音乐", "网易云", "spotify", "apple music", "音乐")),
    ("cloud", ("网盘", "icloud", "百度网盘", "夸克", "onedrive", "dropbox",
               "云盘", "云存储")),
    ("game", ("王者荣耀", "游戏", "原神", "月卡", "steam", "game")),
)

_MIN_GAP_DAYS = 25
_MAX_GAP_DAYS = 35
_AMOUNT_TOLERANCE = 0.10


def subscription_category(merchant: str) -> str:
    """Classify a merchant name into a coarse subscription category."""
    text = merchant.casefold()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return category
    return "other"


def _as_datetime(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, datetime.min.time())


def _debits(
    conn: sqlite3.Connection,
    customer_id: str,
    now: datetime,
    lookback_days: int,
) -> list[sqlite3.Row]:
    start = now - timedelta(days=lookback_days)
    return conn.execute(
        """
        SELECT ts, amount_fen, counterparty
        FROM transactions
        WHERE customer_id = ?
          AND amount_fen < 0
          AND category != 'transfer'
          AND channel NOT IN ('errand', 'transfer', 'bill')
          AND ts >= ?
          AND ts <= ?
        ORDER BY ts
        """,
        (customer_id, start.isoformat(), now.isoformat()),
    ).fetchall()


def _merchants_match(left: str, right: str) -> bool:
    a = left.casefold()
    b = right.casefold()
    return a == b or a in b or b in a


def detect_recurring(
    conn: sqlite3.Connection,
    customer_id: str,
    now: datetime,
    lookback_days: int = 180,
) -> list[dict]:
    """Return recurring debit series for one customer."""
    now_dt = _as_datetime(now)
    groups: dict[str, list[sqlite3.Row]] = {}
    for row in _debits(conn, customer_id, now_dt, lookback_days):
        groups.setdefault(row["counterparty"], []).append(row)

    results: list[dict] = []
    for merchant, rows in groups.items():
        if len(rows) < 2:
            continue
        rows.sort(key=lambda row: row["ts"])
        dates = [datetime.fromisoformat(row["ts"]).date() for row in rows]
        gaps = [
            (dates[index] - dates[index - 1]).days for index in range(1, len(dates))
        ]
        if any(gap < _MIN_GAP_DAYS or gap > _MAX_GAP_DAYS for gap in gaps):
            continue
        amounts = [abs(int(row["amount_fen"])) for row in rows]
        middle = median(amounts)
        if any(
            abs(amount - middle) > middle * _AMOUNT_TOLERANCE for amount in amounts
        ):
            continue
        last_charged = dates[-1]
        next_expected = last_charged + timedelta(days=round(median(gaps)))
        results.append(
            {
                "merchant": merchant,
                "amount_fen": int(round(middle)),
                "occurrences": len(rows),
                "last_charged": last_charged.isoformat(),
                "next_expected": next_expected.isoformat(),
                "category": subscription_category(merchant),
            }
        )

    results.sort(key=lambda item: item["merchant"])
    return results


def last_charge(
    conn: sqlite3.Connection,
    customer_id: str,
    merchant: str,
    now: datetime,
    lookback_days: int = 180,
) -> dict | None:
    """Return the most recent debit matching a merchant, if any."""
    now_dt = _as_datetime(now)
    rows = _debits(conn, customer_id, now_dt, lookback_days)
    matches = [
        row for row in rows if _merchants_match(row["counterparty"], merchant)
    ]
    if not matches:
        return None
    latest = max(matches, key=lambda row: row["ts"])
    return {
        "date": datetime.fromisoformat(latest["ts"]).date().isoformat(),
        "amount_fen": abs(int(latest["amount_fen"])),
    }


def recurring_overview(
    conn: sqlite3.Connection,
    customer_id: str,
    now: datetime,
) -> list[dict]:
    """Merge registered subscriptions with detected recurring charges."""
    now_dt = _as_datetime(now)
    detected = detect_recurring(conn, customer_id, now_dt)
    subscriptions = conn.execute(
        """
        SELECT * FROM subscriptions
        WHERE customer_id = ?
        ORDER BY merchant
        """,
        (customer_id,),
    ).fetchall()

    matched: set[int] = set()
    items: list[dict] = []
    for subscription in subscriptions:
        merchant = subscription["merchant"]
        sources: list[int] = []
        for index, group in enumerate(detected):
            if _merchants_match(merchant, group["merchant"]):
                sources.append(index)
                matched.add(index)
        latest = last_charge(conn, customer_id, merchant, now_dt)
        last_used_at = subscription["last_used_at"]
        items.append(
            {
                "subscription_id": subscription["id"],
                "merchant": merchant,
                "amount_fen": int(subscription["amount_fen"]),
                "status": subscription["status"],
                "category": subscription_category(merchant),
                "source": "both" if sources else "registered",
                "last_charged": latest["date"] if latest else None,
                "unused_days": (
                    now_dt.date() - date.fromisoformat(last_used_at)
                ).days,
            }
        )

    for index, group in enumerate(detected):
        if index in matched:
            continue
        items.append(
            {
                "subscription_id": None,
                "merchant": group["merchant"],
                "amount_fen": group["amount_fen"],
                "status": "detected",
                "category": group["category"],
                "source": "detected",
                "last_charged": group["last_charged"],
                "unused_days": None,
            }
        )

    items.sort(
        key=lambda item: (
            0 if item["subscription_id"] is not None else 1,
            item["merchant"].casefold(),
        )
    )
    return items
