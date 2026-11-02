"""Customer-visible payment habits and gentle nudges. SIMULATED.

The module only derives patterns from local history.  Nothing is used by a
payment until the customer confirms it, and deleted memories are never
suggested again.
"""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from typing import Any

MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    customer_id TEXT NOT NULL,
    id TEXT NOT NULL,
    kind TEXT NOT NULL,
    contact_id TEXT,
    amount_fen INTEGER,
    day_of_month INTEGER,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (customer_id, id)
);

CREATE TABLE IF NOT EXISTS nudge_dismissals (
    customer_id TEXT NOT NULL,
    key TEXT NOT NULL,
    until TEXT NOT NULL,
    PRIMARY KEY (customer_id, key)
);
"""

# Seed rent rows use category "housing"; app transfers use "transfer".
_OUTGOING_CATEGORIES = ("transfer", "housing")
# The seed calls salary "income"; newer data/other tools use "salary".
_OPENING_COUNTERPARTY = "期初余额"


def _ensure(conn: sqlite3.Connection) -> None:
    conn.executescript(MEMORY_SCHEMA)


def _as_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _as_datetime(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    return datetime.fromisoformat(str(value))


def _row_date(value: str) -> date:
    return date.fromisoformat(value[:10])


def _is_salary_row(row: sqlite3.Row) -> bool:
    category = row["category"]
    if category == "salary":
        return True
    # Fall back to the seeded income rows, but never the opening balance.
    return category == "income" and row["counterparty"] != _OPENING_COUNTERPARTY


def list_items(
    conn: sqlite3.Connection, customer_id: str, include_deleted: bool = False
) -> list[dict[str, Any]]:
    """Return this customer's memories, ordered by kind and id."""

    _ensure(conn)
    sql = "SELECT * FROM memory_items WHERE customer_id = ?"
    if not include_deleted:
        sql += " AND status != 'deleted'"
    sql += " ORDER BY kind, id"
    rows = conn.execute(sql, (customer_id,)).fetchall()
    return [dict(row) for row in rows]


def _payments(
    conn: sqlite3.Connection, customer_id: str, start: date, end: date
) -> list[tuple[str, datetime, int, str]]:
    """Outgoing payments to contacts as (contact_id, when, amount, source)."""

    contacts = conn.execute(
        "SELECT id, name FROM contacts WHERE customer_id = ?", (customer_id,)
    ).fetchall()
    by_name: dict[str, list[str]] = defaultdict(list)
    for contact in contacts:
        by_name[contact["name"]].append(contact["id"])

    start_ts = start.isoformat()
    end_ts = (end + timedelta(days=1)).isoformat()
    payments: list[tuple[str, datetime, int, str]] = []

    transfer_rows = conn.execute(
        """
        SELECT id, to_contact_id, amount_fen, created_at
        FROM transfers
        WHERE customer_id = ? AND status != 'cancelled'
          AND created_at >= ? AND created_at < ?
        """,
        (customer_id, start_ts, end_ts),
    ).fetchall()
    transfer_signatures: set[tuple[str, date, int]] = set()
    for row in transfer_rows:
        when = datetime.fromisoformat(row["created_at"])
        amount = int(row["amount_fen"])
        contact_id = row["to_contact_id"]
        transfer_signatures.add((contact_id, when.date(), amount))
        payments.append((contact_id, when, amount, "transfer"))

    txn_rows = conn.execute(
        """
        SELECT id, ts, amount_fen, counterparty, category
        FROM transactions
        WHERE customer_id = ? AND amount_fen < 0
          AND ts >= ? AND ts < ?
        """,
        (customer_id, start_ts, end_ts),
    ).fetchall()
    for row in txn_rows:
        if row["category"] not in _OUTGOING_CATEGORIES:
            continue
        when = datetime.fromisoformat(row["ts"])
        amount = -int(row["amount_fen"])
        for contact_id in by_name.get(row["counterparty"], ()):
            signature = (contact_id, when.date(), amount)
            if signature in transfer_signatures:
                # The same payment is already represented by a transfers row.
                continue
            payments.append((contact_id, when, amount, "transaction"))

    return payments


def _usual_candidate(
    payments: list[tuple[str, datetime, int, str]]
) -> tuple[int, int] | None:
    """Return (amount_fen, occurrences) for the most common amount."""

    if len(payments) < 2:
        return None
    counts = Counter(amount for _, _, amount, _ in payments)
    best_count = max(counts.values())
    if best_count < 2:
        return None
    latest: dict[int, datetime] = {}
    for _, when, amount, _ in payments:
        if amount not in latest or when > latest[amount]:
            latest[amount] = when
    tied = [amount for amount, count in counts.items() if count == best_count]
    amount = max(tied, key=lambda value: latest[value])
    return amount, best_count


def _median_day(days: list[int]) -> int:
    ordered = sorted(days)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        value = float(ordered[middle])
    else:
        value = (ordered[middle - 1] + ordered[middle]) / 2
    return max(1, min(int(round(value)), 28))


def _monthly_candidate(
    payments: list[tuple[str, datetime, int, str]]
) -> tuple[int, int] | None:
    """Return (latest amount, median day) for a monthly payment pattern."""

    if len(payments) < 2:
        return None
    best: tuple[
        tuple[int, int, datetime], list[tuple[str, datetime, int, str]]
    ] | None = None
    for centre in {when.day for _, when, _, _ in payments}:
        cluster = [
            payment
            for payment in payments
            if abs(payment[1].day - centre) <= 3
        ]
        months = {(when.year, when.month) for _, when, _, _ in cluster}
        if len(months) < 2:
            continue
        latest = max(when for _, when, _, _ in cluster)
        score = (len(months), len(cluster), latest)
        if best is None or score > best[0]:
            best = (score, cluster)
    if best is None:
        return None
    cluster = best[1]
    latest_payment = max(cluster, key=lambda payment: payment[1])
    return latest_payment[2], _median_day([when.day for _, when, _, _ in cluster])


def _salary_rows(
    conn: sqlite3.Connection, customer_id: str, start: date, end: date
) -> list[sqlite3.Row]:
    start_ts = start.isoformat()
    end_ts = (end + timedelta(days=1)).isoformat()
    rows = conn.execute(
        """
        SELECT ts, amount_fen, category, counterparty
        FROM transactions
        WHERE customer_id = ? AND amount_fen > 0
          AND category IN ('salary', 'income')
          AND ts >= ? AND ts < ?
        ORDER BY ts, id
        """,
        (customer_id, start_ts, end_ts),
    ).fetchall()
    explicit = [row for row in rows if row["category"] == "salary"]
    if explicit:
        return explicit
    return [row for row in rows if _is_salary_row(row)]


def _save_candidate(
    conn: sqlite3.Connection,
    customer_id: str,
    item_id: str,
    kind: str,
    contact_id: str | None,
    amount_fen: int,
    day_of_month: int | None,
    updated_at: str,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO memory_items
            (customer_id, id, kind, contact_id, amount_fen, day_of_month,
             status, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 'suggested', ?)
        """,
        (
            customer_id,
            item_id,
            kind,
            contact_id,
            int(amount_fen),
            day_of_month,
            updated_at,
        ),
    )
    conn.execute(
        """
        UPDATE memory_items
        SET contact_id = ?, amount_fen = ?, day_of_month = ?, updated_at = ?
        WHERE customer_id = ? AND id = ? AND status = 'suggested'
        """,
        (
            contact_id,
            int(amount_fen),
            day_of_month,
            updated_at,
            customer_id,
            item_id,
        ),
    )


def learn(
    conn: sqlite3.Connection, customer_id: str, now: date | datetime
) -> list[dict[str, Any]]:
    """Derive suggested memories from the last 120 days of history."""

    _ensure(conn)
    today = _as_date(now)
    start = today - timedelta(days=120)
    updated_at = now.isoformat()
    payments = _payments(conn, customer_id, start, today)

    by_contact: dict[str, list[tuple[str, datetime, int, str]]] = defaultdict(list)
    for payment in payments:
        by_contact[payment[0]].append(payment)

    candidates: list[tuple[str, str, str | None, int, int | None]] = []
    for contact_id, contact_payments in by_contact.items():
        usual = _usual_candidate(contact_payments)
        if usual is not None:
            candidates.append(
                (
                    f"usual:{contact_id}",
                    "usual_amount",
                    contact_id,
                    usual[0],
                    None,
                )
            )
        monthly = _monthly_candidate(contact_payments)
        if monthly is not None:
            candidates.append(
                (
                    f"monthly:{contact_id}",
                    "monthly_transfer",
                    contact_id,
                    monthly[0],
                    monthly[1],
                )
            )

    salaries = _salary_rows(conn, customer_id, start, today)
    salary_days = [_row_date(row["ts"]) for row in salaries]
    if len({(day.year, day.month) for day in salary_days}) >= 2:
        latest = salaries[-1]
        latest_day = salary_days[-1]
        candidates.append(
            (
                "payday",
                "payday",
                None,
                int(latest["amount_fen"]),
                max(1, min(latest_day.day, 28)),
            )
        )

    with conn:
        for item_id, kind, contact_id, amount_fen, day_of_month in candidates:
            _save_candidate(
                conn,
                customer_id,
                item_id,
                kind,
                contact_id,
                amount_fen,
                day_of_month,
                updated_at,
            )
    return list_items(conn, customer_id)


def _item_row(
    conn: sqlite3.Connection, customer_id: str, item_id: str
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM memory_items WHERE customer_id = ? AND id = ?",
        (customer_id, item_id),
    ).fetchone()
    if row is None:
        raise KeyError(item_id)
    return row


def confirm(
    conn: sqlite3.Connection, customer_id: str, item_id: str, now: date | datetime
) -> dict[str, Any]:
    """Confirm one suggested memory."""

    _ensure(conn)
    with conn:
        _item_row(conn, customer_id, item_id)
        conn.execute(
            """
            UPDATE memory_items SET status = 'confirmed', updated_at = ?
            WHERE customer_id = ? AND id = ?
            """,
            (now.isoformat(), customer_id, item_id),
        )
        row = _item_row(conn, customer_id, item_id)
    return dict(row)


def delete(
    conn: sqlite3.Connection, customer_id: str, item_id: str, now: date | datetime
) -> dict[str, Any]:
    """Delete one memory so it is never suggested again."""

    _ensure(conn)
    with conn:
        _item_row(conn, customer_id, item_id)
        conn.execute(
            """
            UPDATE memory_items SET status = 'deleted', updated_at = ?
            WHERE customer_id = ? AND id = ?
            """,
            (now.isoformat(), customer_id, item_id),
        )
        row = _item_row(conn, customer_id, item_id)
    return dict(row)


def usual_amount(
    conn: sqlite3.Connection, customer_id: str, contact_id: str
) -> int | None:
    """The confirmed usual amount for a contact, if the customer confirmed one."""

    _ensure(conn)
    row = conn.execute(
        """
        SELECT amount_fen FROM memory_items
        WHERE customer_id = ? AND id = ? AND kind = 'usual_amount'
          AND status = 'confirmed'
        """,
        (customer_id, f"usual:{contact_id}"),
    ).fetchone()
    return int(row["amount_fen"]) if row is not None else None


def _has_recent_salary(
    conn: sqlite3.Connection, customer_id: str, today: date
) -> bool:
    yesterday = today - timedelta(days=1)
    end_ts = (today + timedelta(days=1)).isoformat()
    rows = conn.execute(
        """
        SELECT ts, category, counterparty
        FROM transactions
        WHERE customer_id = ? AND amount_fen > 0
          AND ts >= ? AND ts < ?
        """,
        (customer_id, yesterday.isoformat(), end_ts),
    ).fetchall()
    return any(_is_salary_row(row) for row in rows)


def _dismissed(
    conn: sqlite3.Connection, customer_id: str, now: datetime
) -> set[str]:
    rows = conn.execute(
        "SELECT key, until FROM nudge_dismissals WHERE customer_id = ?",
        (customer_id,),
    ).fetchall()
    dismissed: set[str] = set()
    for row in rows:
        if datetime.fromisoformat(row["until"]) > now:
            dismissed.add(row["key"])
    return dismissed


def nudges(
    conn: sqlite3.Connection, customer_id: str, now: date | datetime
) -> list[dict[str, Any]]:
    """Return confirmed-pattern payday transfer and bill nudges."""

    _ensure(conn)
    now_dt = _as_datetime(now)
    today = now_dt.date()
    result: list[dict[str, Any]] = []

    payday = conn.execute(
        """
        SELECT * FROM memory_items
        WHERE customer_id = ? AND id = 'payday' AND kind = 'payday'
          AND status = 'confirmed'
        """,
        (customer_id,),
    ).fetchone()
    if payday is not None and _has_recent_salary(conn, customer_id, today):
        month_start = today.replace(day=1)
        paid_contacts = {
            payment[0]
            for payment in _payments(conn, customer_id, month_start, today)
        }
        monthly = conn.execute(
            """
            SELECT * FROM memory_items
            WHERE customer_id = ? AND kind = 'monthly_transfer'
              AND status = 'confirmed'
            ORDER BY id
            """,
            (customer_id,),
        ).fetchall()
        for item in monthly:
            contact_id = item["contact_id"]
            if contact_id is None or contact_id in paid_contacts:
                continue
            result.append(
                {
                    "kind": "payday_transfer",
                    "contact_id": contact_id,
                    "amount_fen": int(item["amount_fen"]),
                    "item_id": item["id"],
                    "key": f"payday_transfer:{contact_id}",
                }
            )

    bills_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'bills'"
    ).fetchone()
    if bills_table is not None:
        bills = conn.execute(
            """
            SELECT category, amount_due_fen, due_date
            FROM bills
            WHERE customer_id = ? AND status = 'due'
            ORDER BY due_date, category
            """,
            (customer_id,),
        ).fetchall()
        for bill in bills:
            due = date.fromisoformat(bill["due_date"])
            if abs((due - today).days) <= 3:
                result.append(
                    {
                        "kind": "bill_due",
                        "category": bill["category"],
                        "amount_due_fen": int(bill["amount_due_fen"]),
                        "due_date": bill["due_date"],
                        "key": f"bill_due:{bill['category']}",
                    }
                )

    dismissed = _dismissed(conn, customer_id, now_dt)
    return [nudge for nudge in result if nudge["key"] not in dismissed]


def dismiss(
    conn: sqlite3.Connection,
    customer_id: str,
    key: str,
    now: date | datetime,
    days: int = 30,
) -> dict[str, Any]:
    """Silence one nudge key for ``days`` days."""

    _ensure(conn)
    until = _as_datetime(now) + timedelta(days=days)
    with conn:
        conn.execute(
            """
            INSERT INTO nudge_dismissals (customer_id, key, until)
            VALUES (?, ?, ?)
            ON CONFLICT(customer_id, key) DO UPDATE SET until = excluded.until
            """,
            (customer_id, key, until.isoformat()),
        )
    return {"key": key, "until": until.isoformat()}
