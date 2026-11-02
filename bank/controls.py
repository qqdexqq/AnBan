"""Card controls and self-imposed merchant blocks. SIMULATED.

Tightening a control applies at once; loosening is delayed (the agent layer
adds its own extra checks before a loosening action is accepted).  Merchant
blocks lift only after a 48-hour cooling-off request, and any spending in the
blocked category is still blocked while that request is pending.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any

from bank.api import BankError

CONTROLS = ("online", "abroad", "atm", "contactless")
BLOCK_CATEGORIES = ("gambling", "crypto", "lottery", "adult", "cash_like")
UNBLOCK_DELAY = timedelta(hours=48)

CONTROLS_SCHEMA = """
CREATE TABLE IF NOT EXISTS card_controls (
    customer_id TEXT NOT NULL,
    card_id TEXT NOT NULL,
    control TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (customer_id, card_id, control)
);

CREATE TABLE IF NOT EXISTS merchant_blocks (
    customer_id TEXT NOT NULL,
    category TEXT NOT NULL,
    created_at TEXT NOT NULL,
    unblock_requested_at TEXT,
    unblock_at TEXT,
    PRIMARY KEY (customer_id, category)
);
"""


def _ensure(conn: sqlite3.Connection) -> None:
    conn.executescript(CONTROLS_SCHEMA)


def _require_card(conn: sqlite3.Connection, customer_id: str, card_id: str) -> None:
    row = conn.execute(
        "SELECT id FROM cards WHERE customer_id = ? AND id = ?",
        (customer_id, card_id),
    ).fetchone()
    if row is None:
        raise BankError("not_found", "card not found")


def get_controls(
    conn: sqlite3.Connection, customer_id: str, card_id: str
) -> dict[str, bool]:
    """Return every control for a card; a missing row means enabled."""

    _ensure(conn)
    _require_card(conn, customer_id, card_id)
    rows = conn.execute(
        """
        SELECT control, enabled FROM card_controls
        WHERE customer_id = ? AND card_id = ?
        """,
        (customer_id, card_id),
    ).fetchall()
    overrides = {row["control"]: bool(row["enabled"]) for row in rows}
    return {control: overrides.get(control, True) for control in CONTROLS}


def set_control(
    conn: sqlite3.Connection,
    customer_id: str,
    card_id: str,
    control: str,
    enabled: bool,
    now: datetime,
) -> dict[str, Any]:
    """Tighten or loosen one control; the loosening gate lives above this layer."""

    if control not in CONTROLS:
        raise BankError("invalid_state", "unknown control")
    _ensure(conn)
    _require_card(conn, customer_id, card_id)
    previous = get_controls(conn, customer_id, card_id)[control]
    with conn:
        conn.execute(
            """
            INSERT INTO card_controls
                (customer_id, card_id, control, enabled, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (customer_id, card_id, control) DO UPDATE SET
                enabled = excluded.enabled,
                updated_at = excluded.updated_at
            """,
            (customer_id, card_id, control, int(bool(enabled)), now.isoformat()),
        )
    return {
        "card_id": card_id,
        "control": control,
        "enabled": bool(enabled),
        "previous": previous,
    }


def block_category(
    conn: sqlite3.Connection, customer_id: str, category: str, now: datetime
) -> dict[str, Any]:
    """Block a spending category; re-blocking also cancels any pending unblock."""

    if category not in BLOCK_CATEGORIES:
        raise BankError("invalid_state", "unknown category")
    _ensure(conn)
    with conn:
        conn.execute(
            """
            INSERT INTO merchant_blocks
                (customer_id, category, created_at, unblock_requested_at, unblock_at)
            VALUES (?, ?, ?, NULL, NULL)
            ON CONFLICT (customer_id, category) DO UPDATE SET
                unblock_requested_at = NULL,
                unblock_at = NULL
            """,
            (customer_id, category, now.isoformat()),
        )
    return {"category": category, "blocked": True}


def request_unblock(
    conn: sqlite3.Connection, customer_id: str, category: str, now: datetime
) -> dict[str, Any]:
    """Start the cooling-off clock, or return the already pending request."""

    _ensure(conn)
    with conn:
        row = conn.execute(
            "SELECT * FROM merchant_blocks WHERE customer_id = ? AND category = ?",
            (customer_id, category),
        ).fetchone()
        if row is None:
            raise BankError("not_found", "block not found")
        unblock_at = row["unblock_at"]
        if unblock_at is None:
            unblock_at = (now + UNBLOCK_DELAY).isoformat()
            conn.execute(
                """
                UPDATE merchant_blocks
                SET unblock_requested_at = ?, unblock_at = ?
                WHERE customer_id = ? AND category = ?
                """,
                (now.isoformat(), unblock_at, customer_id, category),
            )
    return {"category": category, "unblock_at": unblock_at}


def apply_due_unblocks(
    conn: sqlite3.Connection, customer_id: str, now: datetime
) -> list[str]:
    """Delete blocks whose cooling-off has expired and return their categories."""

    _ensure(conn)
    rows = conn.execute(
        """
        SELECT category, unblock_at FROM merchant_blocks
        WHERE customer_id = ? AND unblock_at IS NOT NULL
        """,
        (customer_id,),
    ).fetchall()
    due = [
        row["category"]
        for row in rows
        if datetime.fromisoformat(row["unblock_at"]) <= now
    ]
    if due:
        with conn:
            conn.executemany(
                "DELETE FROM merchant_blocks WHERE customer_id = ? AND category = ?",
                [(customer_id, category) for category in due],
            )
    return due


def blocked_categories(
    conn: sqlite3.Connection, customer_id: str, now: datetime
) -> list[str]:
    """Sorted still-blocked categories; a pending unblock is still blocked."""

    apply_due_unblocks(conn, customer_id, now)
    rows = conn.execute(
        "SELECT category FROM merchant_blocks WHERE customer_id = ?",
        (customer_id,),
    ).fetchall()
    return sorted(row["category"] for row in rows)


def is_blocked(
    conn: sqlite3.Connection, customer_id: str, category: str, now: datetime
) -> bool:
    apply_due_unblocks(conn, customer_id, now)
    row = conn.execute(
        "SELECT 1 FROM merchant_blocks WHERE customer_id = ? AND category = ?",
        (customer_id, category),
    ).fetchone()
    return row is not None
