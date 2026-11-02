"""Utility bills (生活缴费) for the demo bank. SIMULATED.

Each seeded customer gets one unpaid bill per utility.  Paying a bill moves
money out of their current account (the balance the app shows) and appends a
ledger row; nothing
leaves the mock bank.
"""

from __future__ import annotations

import random
import secrets
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any

from bank.api import BankError

CATEGORIES = ("water", "electricity", "gas", "mobile", "broadband", "property")

_AMOUNTS: dict[str, tuple[int, int]] = {
    "water": (3000, 12000),
    "electricity": (8000, 40000),
    "gas": (3000, 15000),
    "mobile": (5000, 20000),
    "broadband": (9900, 19900),
    "property": (15000, 60000),
}

_BILLERS: dict[str, tuple[str, str]] = {
    "water": ("市自来水公司", "City Water Co."),
    "electricity": ("国家电网", "State Grid"),
    "gas": ("市燃气公司", "City Gas Co."),
    "mobile": ("中国移动", "China Mobile"),
    "broadband": ("中国电信宽带", "China Telecom Broadband"),
    "property": ("物业服务中心", "Property Management"),
}

BILLS_SCHEMA = """
CREATE TABLE IF NOT EXISTS bills (
    customer_id TEXT NOT NULL,
    id TEXT NOT NULL,
    category TEXT NOT NULL,
    biller TEXT NOT NULL,
    biller_en TEXT NOT NULL,
    account_no TEXT NOT NULL,
    amount_due_fen INTEGER NOT NULL,
    due_date TEXT NOT NULL,
    status TEXT NOT NULL,
    paid_at TEXT,
    PRIMARY KEY (customer_id, id)
);
"""


def _ensure(conn: sqlite3.Connection) -> None:
    conn.executescript(BILLS_SCHEMA)


def _as_dict(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    out["simulated"] = True
    return out


def seed_bills(conn: sqlite3.Connection, customer_id: str, today: date) -> None:
    """Insert one unpaid bill per category, once per customer."""

    _ensure(conn)
    with conn:
        existing = conn.execute(
            "SELECT 1 FROM bills WHERE customer_id = ? LIMIT 1", (customer_id,)
        ).fetchone()
        if existing is not None:
            return
        rng = random.Random(customer_id)
        rows = []
        for category in CATEGORIES:
            low, high = _AMOUNTS[category]
            biller, biller_en = _BILLERS[category]
            amount = int(rng.randint(low, high))
            due = today + timedelta(days=rng.randint(3, 20))
            account_no = "".join(str(rng.randint(0, 9)) for _ in range(10))
            rows.append(
                (
                    customer_id,
                    f"bill_{category}",
                    category,
                    biller,
                    biller_en,
                    account_no,
                    amount,
                    due.isoformat(),
                    "due",
                    None,
                )
            )
        conn.executemany(
            """
            INSERT INTO bills (customer_id, id, category, biller, biller_en,
                account_no, amount_due_fen, due_date, status, paid_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def due_amount(
    conn: sqlite3.Connection, customer_id: str, category: str, today: date
) -> int | None:
    """What paying this category would cost now; None when nothing is due."""

    if category not in CATEGORIES:
        return None
    seed_bills(conn, customer_id, today)
    row = conn.execute(
        "SELECT amount_due_fen FROM bills"
        " WHERE customer_id = ? AND category = ? AND status = 'due'",
        (customer_id, category),
    ).fetchone()
    return int(row["amount_due_fen"]) if row is not None else None


def list_bills(
    conn: sqlite3.Connection, customer_id: str, status: str | None = "due"
) -> list[dict[str, Any]]:
    _ensure(conn)
    if status is None:
        rows = conn.execute(
            "SELECT * FROM bills WHERE customer_id = ? ORDER BY due_date, id",
            (customer_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM bills
            WHERE customer_id = ? AND status = ?
            ORDER BY due_date, id
            """,
            (customer_id, status),
        ).fetchall()
    return [_as_dict(row) for row in rows]


def upcoming_bills(
    conn: sqlite3.Connection, customer_id: str, today: date, days: int = 14
) -> list[dict[str, Any]]:
    _ensure(conn)
    cutoff = (today + timedelta(days=days)).isoformat()
    rows = conn.execute(
        """
        SELECT * FROM bills
        WHERE customer_id = ? AND status = 'due' AND due_date <= ?
        ORDER BY due_date, id
        """,
        (customer_id, cutoff),
    ).fetchall()
    return [_as_dict(row) for row in rows]


def pay_bill(
    conn: sqlite3.Connection, customer_id: str, category: str, now: datetime
) -> dict[str, Any]:
    if category not in CATEGORIES:
        raise BankError("invalid_state", "unknown bill category")
    _ensure(conn)
    with conn:
        bill = conn.execute(
            """
            SELECT * FROM bills
            WHERE customer_id = ? AND category = ? AND status = 'due'
            """,
            (customer_id, category),
        ).fetchone()
        if bill is None:
            raise BankError("not_found", "no due bill for category")
        account = conn.execute(
            """
            SELECT * FROM accounts
            WHERE customer_id = ? AND type = 'current'
            ORDER BY id
            LIMIT 1
            """,
            (customer_id,),
        ).fetchone()
        if account is None:
            raise BankError("not_found", "account not found")
        amount = int(bill["amount_due_fen"])
        if int(account["balance_fen"]) < amount:
            raise BankError("insufficient_funds", "balance too low")
        conn.execute(
            """
            UPDATE accounts SET balance_fen = balance_fen - ?
            WHERE customer_id = ? AND id = ?
            """,
            (amount, customer_id, account["id"]),
        )
        conn.execute(
            """
            UPDATE bills SET status = 'paid', paid_at = ?
            WHERE customer_id = ? AND id = ?
            """,
            (now.isoformat(), customer_id, bill["id"]),
        )
        conn.execute(
            """
            INSERT INTO transactions
                (customer_id, id, account_id, ts, amount_fen, counterparty,
                 category, memo, channel, country, card_present)
            VALUES (?, ?, ?, ?, ?, ?, 'utilities', ?, 'bill', 'CN', 0)
            """,
            (
                customer_id,
                "txn_bill_" + secrets.token_hex(6),
                account["id"],
                now.isoformat(),
                -amount,
                bill["biller"],
                bill["id"],
            ),
        )
        updated = conn.execute(
            "SELECT * FROM bills WHERE customer_id = ? AND id = ?",
            (customer_id, bill["id"]),
        ).fetchone()
    out = _as_dict(updated)
    out["account_id"] = account["id"]
    return out
