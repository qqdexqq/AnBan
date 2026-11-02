"""Trusted one-time-card payment gateway for the online errand agent.

One errand ticket owns exactly one virtual card.  A card can be charged at most
once, and every limit is checked at authorization time, before any money moves.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Iterator

from .api import Bank, BankError
from .redemption import _as_now, _current_account, _next_id

CATEGORIES = frozenset({"flowers", "cake"})
MAX_BUDGET_FEN = 100_000
MAX_TEXT = 64
DAILY_LIMIT_FEN = 100_000
MAX_EXPIRY = timedelta(hours=4)
_TICKET_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

_ERRAND_TICKETS_TABLE = """
CREATE TABLE IF NOT EXISTS errand_tickets (
    customer_id TEXT NOT NULL,
    ticket_id TEXT NOT NULL,
    card_id TEXT NOT NULL,
    category TEXT NOT NULL,
    budget_fen INTEGER NOT NULL,
    deliver_by TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('issued', 'charged', 'closed')),
    order_ref TEXT,
    charged_fen INTEGER,
    merchant TEXT,
    txn_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (customer_id, ticket_id),
    UNIQUE (customer_id, card_id)
)
"""


@contextmanager
def _transaction(conn: sqlite3.Connection, savepoint: str) -> Iterator[None]:
    """Run a block in one transaction, mirroring the redemption module."""
    began_transaction = not conn.in_transaction
    if began_transaction:
        conn.execute("BEGIN IMMEDIATE")
    sp = None if began_transaction else savepoint
    if sp is not None:
        conn.execute(f"SAVEPOINT {sp}")

    try:
        conn.execute(_ERRAND_TICKETS_TABLE)
        yield
    except BaseException:
        if sp is not None:
            conn.execute(f"ROLLBACK TO {sp}")
            conn.execute(f"RELEASE {sp}")
        elif began_transaction:
            conn.rollback()
        raise
    else:
        if sp is not None:
            conn.execute(f"RELEASE {sp}")
        elif began_transaction:
            conn.commit()


def _ticket(conn: sqlite3.Connection, customer_id: str, ticket_id: str) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT * FROM errand_tickets
        WHERE customer_id = ? AND ticket_id = ?
        """,
        (customer_id, ticket_id),
    ).fetchone()
    if row is None:
        raise BankError("not_found", "ticket not found")
    return row


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_TEXT:
        raise BankError("invalid_state", f"{field} must be 1-{MAX_TEXT} characters")
    return value


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _check_not_blocked(
    conn: sqlite3.Connection, customer_id: str, now_dt: datetime
) -> None:
    if _table_exists(conn, "security_lockouts"):
        row = conn.execute(
            "SELECT locked_until FROM security_lockouts WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
        if row is not None and _as_now(row["locked_until"]) > now_dt:
            raise BankError("R_LOCKED", "customer is locked out")
    if _table_exists(conn, "security_scam_freezes"):
        row = conn.execute(
            "SELECT frozen_until FROM security_scam_freezes WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
        if row is not None and _as_now(row["frozen_until"]) > now_dt:
            raise BankError("R_SCAM", "customer account is frozen for suspected scam")


def reserved_budget(conn: sqlite3.Connection, customer_id: str, now_dt: datetime, *, exclude_ticket: str = "") -> int:
    """Live ticket exposure, including a ticket crossing local midnight."""
    if not _table_exists(conn, "errand_tickets"):
        return 0
    return int(conn.execute(
        "SELECT COALESCE(SUM(budget_fen),0) FROM errand_tickets "
        "WHERE customer_id=? AND status='issued' AND expires_at>? AND ticket_id<>?",
        (customer_id, now_dt.isoformat(), exclude_ticket),
    ).fetchone()[0])


def _check_exposure(conn: sqlite3.Connection, customer_id: str, now_dt: datetime,
                    amount_fen: int, *, exclude_ticket: str = "") -> None:
    # The same policy configuration decides the ordinary/elder ceiling. All
    # settings reads are transaction-preserving and occur under the debit lock.
    from security.policy import PolicyConfig, _daily_limit
    bank = Bank(conn, customer_id)
    limit = min(DAILY_LIMIT_FEN, _daily_limit(bank, PolicyConfig()))
    exposure = bank.daily_outgoing_total(now_dt) + reserved_budget(
        conn, customer_id, now_dt, exclude_ticket=exclude_ticket)
    if exposure + amount_fen > limit:
        raise BankError("over_limit", "daily payment limit reached")


def open_errand(
    conn: sqlite3.Connection,
    customer_id: str,
    *,
    ticket_id: str,
    category: str,
    budget_fen: int,
    deliver_by: date,
    expires_at: datetime,
    now: date | datetime | str,
) -> dict[str, object]:
    """Issue a one-time virtual card for one errand ticket."""
    if not isinstance(ticket_id, str) or not _TICKET_ID_RE.fullmatch(ticket_id):
        raise BankError("invalid_state", "ticket_id is invalid")
    if category not in CATEGORIES:
        raise BankError("invalid_state", "category must be flowers or cake")
    if (
        not _is_int(budget_fen)
        or budget_fen < 1
        or budget_fen > MAX_BUDGET_FEN
    ):
        raise BankError(
            "invalid_amount", "budget_fen must be an integer from 1 to 100000"
        )
    now_dt = _as_now(now)
    if not isinstance(deliver_by, date) or isinstance(deliver_by, datetime):
        raise BankError("invalid_state", "deliver_by must be a date")
    if deliver_by < now_dt.date():
        raise BankError("invalid_state", "deliver_by cannot be in the past")
    if (
        not isinstance(expires_at, datetime)
        or expires_at.tzinfo is not None
        or not now_dt < expires_at <= now_dt + MAX_EXPIRY
    ):
        raise BankError(
            "invalid_state", "expires_at must be a naive datetime within 4 hours"
        )
    now_iso = now_dt.isoformat()

    with _transaction(conn, "bank_errand_open_sp"):
        existing = conn.execute(
            """
            SELECT 1 FROM errand_tickets
            WHERE customer_id = ? AND ticket_id = ?
            """,
            (customer_id, ticket_id),
        ).fetchone()
        if existing is not None:
            raise BankError("invalid_state", "ticket already exists")

        _check_not_blocked(conn, customer_id, now_dt)

        _check_exposure(conn, customer_id, now_dt, budget_fen)

        account = _current_account(conn, customer_id)
        if int(account["balance_fen"]) < budget_fen:
            raise BankError("insufficient_funds", "budget exceeds current balance")

        card_id = _next_id(conn, "card")
        conn.execute(
            """
            INSERT INTO cards
                (customer_id, id, kind, last4, status, online_limit_fen, account_id)
            VALUES (?, ?, 'virtual', ?, 'active', ?, ?)
            """,
            (customer_id, card_id, card_id[-4:], budget_fen, account["id"]),
        )
        conn.execute(
            """
            INSERT INTO errand_tickets
                (customer_id, ticket_id, card_id, category, budget_fen,
                 deliver_by, expires_at, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'issued', ?, ?)
            """,
            (
                customer_id,
                ticket_id,
                card_id,
                category,
                budget_fen,
                deliver_by.isoformat(),
                expires_at.isoformat(),
                now_iso,
                now_iso,
            ),
        )

    return {
        "ticket_id": ticket_id,
        "card_id": card_id,
        "budget_fen": budget_fen,
        "status": "issued",
    }


def authorize_charge(
    conn: sqlite3.Connection,
    customer_id: str,
    *,
    ticket_id: str,
    card_id: str,
    amount_fen: int,
    merchant: str,
    order_ref: str,
    now: date | datetime | str,
) -> dict[str, object]:
    """Authorize at most one charge against the ticket's virtual card."""
    now_dt = _as_now(now)

    with _transaction(conn, "bank_errand_charge_sp"):
        ticket = _ticket(conn, customer_id, ticket_id)
        status = ticket["status"]

        if status == "charged":
            if (
                order_ref == ticket["order_ref"]
                and _is_int(amount_fen)
                and amount_fen == ticket["charged_fen"]
                and card_id == ticket["card_id"]
            ):
                return {
                    "ticket_id": ticket_id,
                    "order_ref": ticket["order_ref"],
                    "charged_fen": int(ticket["charged_fen"]),
                    "merchant": ticket["merchant"],
                    "txn_id": ticket["txn_id"],
                    "replayed": True,
                }
            raise BankError("ticket_used", "ticket has already been charged")
        _check_not_blocked(conn, customer_id, now_dt)
        if now_dt >= _as_now(ticket["expires_at"]):
            raise BankError("ticket_expired", "ticket has expired")
        if status == "closed":
            raise BankError("ticket_closed", "ticket has been closed")

        if card_id != ticket["card_id"]:
            raise BankError("wrong_card", "card does not belong to this ticket")
        card = conn.execute(
            """
            SELECT * FROM cards
            WHERE customer_id = ? AND id = ?
            """,
            (customer_id, card_id),
        ).fetchone()
        if (
            card is None
            or card["kind"] != "virtual"
            or card["status"] != "active"
        ):
            raise BankError("card_inactive", "virtual card is not active")

        if not _is_int(amount_fen) or amount_fen <= 0:
            raise BankError("invalid_amount", "amount_fen must be a positive integer")
        if amount_fen > int(ticket["budget_fen"]) or amount_fen > int(
            card["online_limit_fen"]
        ):
            raise BankError("over_limit", "amount exceeds the ticket budget")

        merchant = _check_text(merchant, "merchant")
        order_ref = _check_text(order_ref, "order_ref")

        # Opening a ticket does not reserve account funds. Another approved
        # payment or stricter elder mode may have consumed its daily headroom.
        _check_exposure(conn, customer_id, now_dt, amount_fen, exclude_ticket=ticket_id)

        account = _current_account(conn, customer_id)
        if int(account["balance_fen"]) < amount_fen:
            raise BankError("insufficient_funds", "amount exceeds current balance")

        conn.execute(
            """
            UPDATE accounts SET balance_fen = balance_fen - ?
            WHERE customer_id = ? AND id = ?
            """,
            (amount_fen, customer_id, account["id"]),
        )
        txn_id = _next_id(conn, "txn")
        conn.execute(
            """
            INSERT INTO transactions
                (customer_id, id, account_id, ts, amount_fen, counterparty,
                 category, memo, channel, country, card_present)
            VALUES (?, ?, ?, ?, ?, ?, 'shopping', ?, 'errand', 'CN', 0)
            """,
            (
                customer_id,
                txn_id,
                account["id"],
                now_dt.isoformat(),
                -amount_fen,
                merchant,
                order_ref,
            ),
        )
        conn.execute(
            """
            UPDATE cards SET status = 'closed'
            WHERE customer_id = ? AND id = ?
            """,
            (customer_id, card_id),
        )
        cursor = conn.execute(
            """
            UPDATE errand_tickets
            SET status = 'charged', order_ref = ?, charged_fen = ?,
                merchant = ?, txn_id = ?, updated_at = ?
            WHERE customer_id = ? AND ticket_id = ? AND status = 'issued'
            """,
            (
                order_ref,
                amount_fen,
                merchant,
                txn_id,
                now_dt.isoformat(),
                customer_id,
                ticket_id,
            ),
        )
        if cursor.rowcount != 1:
            raise BankError("ticket_used", "ticket is no longer available")

    return {
        "ticket_id": ticket_id,
        "order_ref": order_ref,
        "charged_fen": amount_fen,
        "merchant": merchant,
        "txn_id": txn_id,
        "replayed": False,
    }


def close_errand(
    conn: sqlite3.Connection,
    customer_id: str,
    *,
    ticket_id: str,
    now: date | datetime | str,
) -> dict[str, object]:
    """Close an unused ticket and its card; charged tickets need a refund."""
    now_dt = _as_now(now)

    with _transaction(conn, "bank_errand_close_sp"):
        ticket = _ticket(conn, customer_id, ticket_id)
        status = ticket["status"]
        if status == "charged":
            raise BankError(
                "ticket_used", "charged tickets need a refund, not a close"
            )
        if status == "issued":
            conn.execute(
                """
                UPDATE errand_tickets SET status = 'closed', updated_at = ?
                WHERE customer_id = ? AND ticket_id = ? AND status = 'issued'
                """,
                (now_dt.isoformat(), customer_id, ticket_id),
            )
            conn.execute(
                """
                UPDATE cards SET status = 'closed'
                WHERE customer_id = ? AND id = ?
                """,
                (customer_id, ticket["card_id"]),
            )
            status = "closed"

    return {"ticket_id": ticket_id, "status": status}


def errand_status(
    conn: sqlite3.Connection, customer_id: str, ticket_id: str
) -> dict[str, object]:
    """Read-only view of one ticket; only the table may be created if missing."""
    conn.execute(_ERRAND_TICKETS_TABLE)
    ticket = _ticket(conn, customer_id, ticket_id)
    charged_fen = ticket["charged_fen"]
    return {
        "ticket_id": ticket["ticket_id"],
        "status": ticket["status"],
        "card_id": ticket["card_id"],
        "budget_fen": int(ticket["budget_fen"]),
        "charged_fen": int(charged_fen) if charged_fen is not None else None,
        "order_ref": ticket["order_ref"],
        "txn_id": ticket["txn_id"],
        "merchant": ticket["merchant"],
    }
