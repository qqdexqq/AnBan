"""Mock-only full redemption of the declared redeemable fund holdings.

This module is deliberately narrow: only the mock funds in
``REDEEMABLE_PRODUCTS`` can be redeemed, only in full, and only at the value the
bank computes itself.  There is no real market price, settlement or maturity
handling here.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, time, timedelta, timezone

from .api import BankError

# p1 is the term product: its maturity is not modelled, so it is refused.
TERM_PRODUCT = "p1"
REDEEMABLE_PRODUCTS = frozenset({"p2", "p3", "p4"})

# China has no DST, so a fixed +08:00 offset is exact.
_LOCAL_TZ = timezone(timedelta(hours=8))

_REDEMPTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS bank_redemptions (
    customer_id TEXT NOT NULL,
    id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    product_id TEXT NOT NULL,
    quoted_amount_fen INTEGER NOT NULL,
    original_principal_fen INTEGER NOT NULL,
    original_value_fen INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (customer_id, id),
    UNIQUE (customer_id, idempotency_key)
)
"""


def _to_local_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(_LOCAL_TZ).replace(tzinfo=None)


def _as_now(value: date | datetime | str) -> datetime:
    if isinstance(value, datetime):
        return _to_local_naive(value)
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if isinstance(value, str):
        try:
            return _to_local_naive(datetime.fromisoformat(value))
        except ValueError as exc:
            raise BankError("invalid_state", "invalid datetime") from exc
    raise BankError("invalid_state", "invalid datetime")


def _next_id(conn: sqlite3.Connection, prefix: str) -> str:
    row = conn.execute("SELECT value FROM meta WHERE key = 'id_seq'").fetchone()
    seq = int(row[0]) + 1 if row is not None else 1
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('id_seq', ?)", (str(seq),)
    )
    return f"{prefix}_{seq:06d}"


def _check_product(product_id: str) -> None:
    if not isinstance(product_id, str) or not product_id:
        raise BankError("invalid_state", "product_id is required")
    if product_id == TERM_PRODUCT:
        raise BankError(
            "invalid_state",
            "term product cannot be redeemed because maturity is not modelled",
        )
    if product_id not in REDEEMABLE_PRODUCTS:
        raise BankError("not_found", "product is not a redeemable mock fund")


def _current_account(conn: sqlite3.Connection, customer_id: str) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT * FROM accounts
        WHERE customer_id = ? AND type = 'current'
        ORDER BY id LIMIT 1
        """,
        (customer_id,),
    ).fetchone()
    if row is None:
        raise BankError("invalid_state", "current account not found")
    return row


def _receipt(row: sqlite3.Row) -> dict[str, object]:
    return {
        "product_id": row["product_id"],
        "amount_fen": int(row["quoted_amount_fen"]),
        "status": "redeemed",
        "simulated": True,
        "receipt_id": row["id"],
        "created_at": row["created_at"],
    }


def quote(
    conn: sqlite3.Connection, customer_id: str, product_id: str
) -> dict[str, object]:
    """Return the bank-computed full-redemption quote for one owned holding."""
    _check_product(product_id)
    row = conn.execute(
        """
        SELECT principal_fen, value_fen FROM holdings
        WHERE customer_id = ? AND product_id = ?
        """,
        (customer_id, product_id),
    ).fetchone()
    if row is None:
        raise BankError("not_found", "holding not found")
    principal_fen = int(row["principal_fen"])
    value_fen = int(row["value_fen"])
    if principal_fen <= 0 or value_fen <= 0:
        raise BankError("invalid_state", "holding has no redeemable value")
    return {
        "product_id": product_id,
        "principal_fen": principal_fen,
        "value_fen": value_fen,
        "payout_fen": value_fen,
    }


def redeem(
    conn: sqlite3.Connection,
    customer_id: str,
    product_id: str,
    quoted_amount_fen: int,
    idempotency_key: str,
    now: date | datetime | str,
) -> dict[str, object]:
    """Redeem an owned redeemable fund holding in full, atomically.

    The credit is always the bank's current ``holding.value_fen``.  ``now`` is
    injected by the caller; this function never reads the wall clock.
    """
    _check_product(product_id)
    if not isinstance(idempotency_key, str) or not idempotency_key.strip():
        raise BankError("invalid_state", "idempotency_key is required")
    if (
        not isinstance(quoted_amount_fen, int)
        or isinstance(quoted_amount_fen, bool)
        or quoted_amount_fen <= 0
    ):
        raise BankError(
            "invalid_amount", "quoted_amount_fen must be a positive integer in fen"
        )
    now_dt = _as_now(now)

    began_transaction = not conn.in_transaction
    if began_transaction:
        conn.execute("BEGIN IMMEDIATE")
    savepoint = None if began_transaction else "bank_redeem_sp"
    if savepoint is not None:
        conn.execute(f"SAVEPOINT {savepoint}")

    try:
        conn.execute(_REDEMPTIONS_TABLE)

        existing = conn.execute(
            """
            SELECT * FROM bank_redemptions
            WHERE customer_id = ? AND idempotency_key = ?
            """,
            (customer_id, idempotency_key),
        ).fetchone()
        if existing is not None:
            if (
                existing["product_id"] != product_id
                or int(existing["quoted_amount_fen"]) != quoted_amount_fen
            ):
                raise BankError(
                    "invalid_state",
                    "idempotency_key already used with different arguments",
                )
            result = _receipt(existing)
        else:
            current = quote(conn, customer_id, product_id)
            if quoted_amount_fen != current["payout_fen"]:
                raise BankError(
                    "invalid_state",
                    "quoted amount no longer matches the current valuation",
                )

            account = _current_account(conn, customer_id)
            value_fen = int(current["value_fen"])
            principal_fen = int(current["principal_fen"])

            conn.execute(
                """
                UPDATE accounts SET balance_fen = balance_fen + ?
                WHERE customer_id = ? AND id = ?
                """,
                (value_fen, customer_id, account["id"]),
            )
            conn.execute(
                """
                UPDATE holdings SET principal_fen = 0, value_fen = 0
                WHERE customer_id = ? AND product_id = ?
                """,
                (customer_id, product_id),
            )

            product = conn.execute(
                "SELECT name FROM products WHERE id = ?", (product_id,)
            ).fetchone()
            counterparty = product["name"] if product is not None else product_id
            conn.execute(
                """
                INSERT INTO transactions
                    (customer_id, id, account_id, ts, amount_fen, counterparty,
                     category, memo, channel, country, card_present)
                VALUES (?, ?, ?, ?, ?, ?, 'redemption', ?, 'app', 'CN', 0)
                """,
                (
                    customer_id,
                    _next_id(conn, "txn"),
                    account["id"],
                    now_dt.isoformat(),
                    value_fen,
                    counterparty,
                    "模拟赎回",
                ),
            )

            receipt_id = _next_id(conn, "rdm")
            created_at = now_dt.isoformat()
            conn.execute(
                """
                INSERT INTO bank_redemptions
                    (customer_id, id, idempotency_key, product_id,
                     quoted_amount_fen, original_principal_fen,
                     original_value_fen, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    customer_id,
                    receipt_id,
                    idempotency_key,
                    product_id,
                    value_fen,
                    principal_fen,
                    value_fen,
                    created_at,
                ),
            )
            result = {
                "product_id": product_id,
                "amount_fen": value_fen,
                "status": "redeemed",
                "simulated": True,
                "receipt_id": receipt_id,
                "created_at": created_at,
            }
    except BaseException:
        if savepoint is not None:
            conn.execute(f"ROLLBACK TO {savepoint}")
            conn.execute(f"RELEASE {savepoint}")
        elif began_transaction:
            conn.rollback()
        raise
    else:
        if savepoint is not None:
            conn.execute(f"RELEASE {savepoint}")
        elif began_transaction:
            conn.commit()

    return result
