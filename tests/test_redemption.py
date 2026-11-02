"""Tests for the mock-only full-redemption capability."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta

import pytest

from bank.api import Bank, BankError
from bank.db import connect, create_schema, load_state
from bank.redemption import quote, redeem
from bank.seed import generate

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42


def _current_balance(conn: sqlite3.Connection, customer_id: str) -> int:
    return int(
        conn.execute(
            """
            SELECT balance_fen FROM accounts
            WHERE customer_id = ? AND type = 'current'
            """,
            (customer_id,),
        ).fetchone()[0]
    )


def _holding(
    conn: sqlite3.Connection, customer_id: str, product_id: str
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT principal_fen, value_fen FROM holdings
        WHERE customer_id = ? AND product_id = ?
        """,
        (customer_id, product_id),
    ).fetchone()


def _total_owned(conn: sqlite3.Connection, customer_id: str) -> int:
    accounts = conn.execute(
        "SELECT COALESCE(SUM(balance_fen), 0) FROM accounts WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()[0]
    pots = conn.execute(
        "SELECT COALESCE(SUM(balance_fen), 0) FROM pots WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()[0]
    holdings = conn.execute(
        "SELECT COALESCE(SUM(value_fen), 0) FROM holdings WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()[0]
    return int(accounts + pots + holdings)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        is not None
    )


def _count(conn: sqlite3.Connection, table: str, customer_id: str) -> int:
    if not _table_exists(conn, table):
        return 0
    return int(
        conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE customer_id = ?", (customer_id,)
        ).fetchone()[0]
    )


@pytest.fixture
def env():
    state = generate("young_pro", SEED, TODAY)
    conn = connect()
    create_schema(conn)
    load_state(conn, state)
    try:
        yield conn, state
    finally:
        conn.close()


@pytest.fixture
def multi_env():
    state_a = generate("young_pro", SEED, TODAY)
    state_b = generate("retiree", SEED, TODAY)
    conn = connect()
    create_schema(conn)
    load_state(conn, state_a)
    load_state(conn, state_b)
    try:
        yield conn, state_a, state_b
    finally:
        conn.close()


def test_own_funds_credit_conservation(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    before_balance = _current_balance(conn, customer_id)
    before_total = _total_owned(conn, customer_id)
    before_holdings = _count(conn, "holdings", customer_id)

    quoted = quote(conn, customer_id, "p3")
    assert quoted["product_id"] == "p3"
    assert quoted["principal_fen"] == 1_000_000
    assert quoted["value_fen"] == 1_120_000
    assert quoted["payout_fen"] == 1_120_000

    receipt = redeem(
        conn,
        customer_id,
        "p3",
        quoted["payout_fen"],
        "redeem-p3",
        NOW,
    )

    assert receipt["product_id"] == "p3"
    assert receipt["amount_fen"] == 1_120_000
    assert receipt["status"] == "redeemed"
    assert receipt["simulated"] is True
    assert isinstance(receipt["receipt_id"], str) and receipt["receipt_id"]
    assert receipt["created_at"] == NOW.isoformat()
    assert _current_balance(conn, customer_id) == before_balance + 1_120_000
    assert _total_owned(conn, customer_id) == before_total

    # The holding row stays, zeroed, rather than being deleted.
    assert _count(conn, "holdings", customer_id) == before_holdings
    holding = _holding(conn, customer_id, "p3")
    assert holding is not None
    assert holding["principal_fen"] == 0
    assert holding["value_fen"] == 0

    ledger = conn.execute(
        """
        SELECT amount_fen, category FROM transactions
        WHERE customer_id = ? AND category = 'redemption'
        """,
        (customer_id,),
    ).fetchall()
    assert len(ledger) == 1
    assert ledger[0]["amount_fen"] == 1_120_000
    assert ledger[0]["amount_fen"] > 0


def test_original_holding_row_is_not_deleted(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    quoted = quote(conn, customer_id, "p3")

    redeem(conn, customer_id, "p3", quoted["payout_fen"], "keep-row", NOW)

    rows = conn.execute(
        """
        SELECT principal_fen, value_fen FROM holdings
        WHERE customer_id = ? AND product_id = 'p3'
        """,
        (customer_id,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["principal_fen"] == 0
    assert rows[0]["value_fen"] == 0


def test_refuses_unowned_term_and_zero_holdings(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    before_balance = _current_balance(conn, customer_id)
    before_total = _total_owned(conn, customer_id)

    with pytest.raises(BankError) as exc:
        quote(conn, customer_id, "p2")
    assert exc.value.code == "not_found"

    with pytest.raises(BankError) as exc:
        quote(conn, customer_id, "p1")
    assert exc.value.code == "invalid_state"

    with pytest.raises(BankError) as exc:
        redeem(conn, customer_id, "p1", 1, "term", NOW)
    assert exc.value.code == "invalid_state"

    with pytest.raises(BankError) as exc:
        redeem(conn, customer_id, "p2", 1, "unowned", NOW)
    assert exc.value.code == "not_found"

    assert _current_balance(conn, customer_id) == before_balance
    assert _total_owned(conn, customer_id) == before_total

    quoted = quote(conn, customer_id, "p3")
    redeem(conn, customer_id, "p3", quoted["payout_fen"], "zero-after", NOW)
    with pytest.raises(BankError) as exc:
        quote(conn, customer_id, "p3")
    assert exc.value.code == "invalid_state"


def test_stale_quote_is_refused_without_mutation(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    quoted = quote(conn, customer_id, "p3")

    conn.execute(
        """
        UPDATE holdings SET value_fen = value_fen + 1
        WHERE customer_id = ? AND product_id = 'p3'
        """,
        (customer_id,),
    )
    conn.commit()

    balance_before = _current_balance(conn, customer_id)
    holding_before = tuple(_holding(conn, customer_id, "p3"))
    txn_before = _count(conn, "transactions", customer_id)
    receipt_before = _count(conn, "bank_redemptions", customer_id)

    with pytest.raises(BankError) as exc:
        redeem(
            conn,
            customer_id,
            "p3",
            quoted["payout_fen"],
            "stale",
            NOW,
        )
    assert exc.value.code == "invalid_state"

    assert _current_balance(conn, customer_id) == balance_before
    assert tuple(_holding(conn, customer_id, "p3")) == holding_before
    assert _count(conn, "transactions", customer_id) == txn_before
    assert _count(conn, "bank_redemptions", customer_id) == receipt_before


def test_same_key_replay_credits_once(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    quoted = quote(conn, customer_id, "p3")
    before_balance = _current_balance(conn, customer_id)

    first = redeem(conn, customer_id, "p3", quoted["payout_fen"], "same-key", NOW)
    second = redeem(
        conn,
        customer_id,
        "p3",
        quoted["payout_fen"],
        "same-key",
        NOW + timedelta(hours=1),
    )

    assert second == first
    assert _current_balance(conn, customer_id) == before_balance + 1_120_000
    assert _count(conn, "bank_redemptions", customer_id) == 1
    assert (
        conn.execute(
            """
            SELECT COUNT(*) FROM transactions
            WHERE customer_id = ? AND category = 'redemption'
            """,
            (customer_id,),
        ).fetchone()[0]
        == 1
    )

    with pytest.raises(BankError) as exc:
        redeem(conn, customer_id, "p3", quoted["payout_fen"] + 1, "same-key", NOW)
    assert exc.value.code == "invalid_state"
    assert _current_balance(conn, customer_id) == before_balance + 1_120_000


def test_cross_customer_idempotency_isolation(multi_env) -> None:
    conn, state_a, state_b = multi_env
    customer_a = state_a.customer.id
    customer_b = state_b.customer.id
    quote_a = quote(conn, customer_a, "p3")
    quote_b = quote(conn, customer_b, "p2")
    balance_a = _current_balance(conn, customer_a)
    balance_b = _current_balance(conn, customer_b)

    receipt_a = redeem(conn, customer_a, "p3", quote_a["payout_fen"], "shared", NOW)
    receipt_b = redeem(conn, customer_b, "p2", quote_b["payout_fen"], "shared", NOW)

    assert receipt_a["receipt_id"] != receipt_b["receipt_id"]
    assert _current_balance(conn, customer_a) == balance_a + quote_a["value_fen"]
    assert _current_balance(conn, customer_b) == balance_b + quote_b["value_fen"]
    assert _count(conn, "bank_redemptions", customer_a) == 1
    assert _count(conn, "bank_redemptions", customer_b) == 1

    assert (
        redeem(conn, customer_a, "p3", quote_a["payout_fen"], "shared", NOW)
        == receipt_a
    )
    assert (
        redeem(conn, customer_b, "p2", quote_b["payout_fen"], "shared", NOW)
        == receipt_b
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM bank_redemptions WHERE idempotency_key = 'shared'"
        ).fetchone()[0]
        == 2
    )


def test_failed_receipt_insert_rolls_back_everything(env) -> None:
    conn, state = env
    customer_id = state.customer.id

    # Create the receipt table with one successful redemption first.
    bank = Bank(conn, customer_id)
    bank.subscribe_product("p2", 20_000, NOW)
    p2_quote = quote(conn, customer_id, "p2")
    redeem(conn, customer_id, "p2", p2_quote["payout_fen"], "create-table", NOW)

    conn.execute(
        """
        CREATE TRIGGER fail_redemption_receipt
        BEFORE INSERT ON bank_redemptions
        BEGIN
            SELECT RAISE(ABORT, 'forced failure');
        END;
        """
    )
    conn.commit()

    p3_quote = quote(conn, customer_id, "p3")
    balance_before = _current_balance(conn, customer_id)
    holding_before = tuple(_holding(conn, customer_id, "p3"))
    txn_before = _count(conn, "transactions", customer_id)
    receipt_before = _count(conn, "bank_redemptions", customer_id)

    with pytest.raises(sqlite3.IntegrityError):
        redeem(conn, customer_id, "p3", p3_quote["payout_fen"], "forced", NOW)

    assert _current_balance(conn, customer_id) == balance_before
    assert tuple(_holding(conn, customer_id, "p3")) == holding_before
    assert _count(conn, "transactions", customer_id) == txn_before
    assert _count(conn, "bank_redemptions", customer_id) == receipt_before
