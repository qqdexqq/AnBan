"""Recall requests (退汇/追回申请) for transfers that have already left. SIMULATED.

Once a transfer is released no bank can pull it back, freeze the payee or tell
the sender who holds the account.  What a real bank does is ask the receiving
bank to request a voluntary return, and give the customer an evidence pack for
96110/110 or a 不当得利 claim (民法典 art. 122).  Here the ticket is only a row;
nothing is sent anywhere, and the reply must never promise the money back.
"""

from __future__ import annotations

import secrets
import sqlite3
from datetime import datetime
from typing import Any

from bank.api import BankError

RECALL_SCHEMA = """
CREATE TABLE IF NOT EXISTS recall_requests (
    customer_id TEXT NOT NULL,
    id TEXT NOT NULL,
    transfer_id TEXT NOT NULL,
    amount_fen INTEGER NOT NULL,
    to_contact_id TEXT NOT NULL,
    sent_at TEXT NOT NULL,
    released_at TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (customer_id, id),
    UNIQUE (customer_id, transfer_id)
);
"""


def _ensure(conn: sqlite3.Connection) -> None:
    conn.executescript(RECALL_SCHEMA)


def _as_dict(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    out["simulated"] = True
    return out


def open_recall(
    conn: sqlite3.Connection, customer_id: str, transfer_id: str, now: datetime
) -> dict[str, Any]:
    """Open (or return the existing) recall ticket for a released transfer."""

    _ensure(conn)
    with conn:
        row = conn.execute(
            "SELECT * FROM transfers WHERE customer_id = ? AND id = ?",
            (customer_id, transfer_id),
        ).fetchone()
        if row is None:
            raise BankError("not_found", "transfer not found")
        if row["status"] == "cancelled":
            raise BankError("invalid_state", "transfer was undone")
        if row["status"] == "pending" and now < datetime.fromisoformat(row["release_at"]):
            # Still inside the hold: the honest answer is Undo, not a recall.
            raise BankError("invalid_state", "still undoable")
        existing = conn.execute(
            "SELECT * FROM recall_requests WHERE customer_id = ? AND transfer_id = ?",
            (customer_id, transfer_id),
        ).fetchone()
        if existing is not None:
            return _as_dict(existing)
        ticket = "RC" + secrets.token_hex(4).upper()
        conn.execute(
            """
            INSERT INTO recall_requests (customer_id, id, transfer_id, amount_fen,
                to_contact_id, sent_at, released_at, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'submitted', ?)
            """,
            (
                customer_id,
                ticket,
                transfer_id,
                int(row["amount_fen"]),
                row["to_contact_id"],
                row["created_at"],
                row["release_at"],
                now.isoformat(),
            ),
        )
        created = conn.execute(
            "SELECT * FROM recall_requests WHERE customer_id = ? AND id = ?",
            (customer_id, ticket),
        ).fetchone()
    return _as_dict(created)


def recalls(conn: sqlite3.Connection, customer_id: str) -> list[dict[str, Any]]:
    _ensure(conn)
    rows = conn.execute(
        "SELECT * FROM recall_requests WHERE customer_id = ? ORDER BY created_at DESC",
        (customer_id,),
    ).fetchall()
    return [_as_dict(row) for row in rows]
