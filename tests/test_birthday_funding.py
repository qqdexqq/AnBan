"""Reserved-pot funding for merchant orders (birthday reserve scenario)."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from inspect import Parameter, signature

import pytest

from bank.api import Bank, BankError
from bank.db import connect, create_schema, load_state
from bank.seed import generate

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42
POT_NAME = "birthday"


@pytest.fixture
def env():
    state = generate("young_pro", SEED, TODAY)
    conn = connect()
    create_schema(conn)
    load_state(conn, state)
    bank = Bank(conn, state.customer.id)
    try:
        yield bank, state, conn
    finally:
        conn.close()


def _pot_balance(conn: sqlite3.Connection, pot_id: str) -> int:
    return conn.execute(
        "SELECT balance_fen FROM pots WHERE id = ?", (pot_id,)
    ).fetchone()[0]


def _order_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM merchant_orders").fetchone()[0]


def _has_funding_column(conn: sqlite3.Connection) -> bool:
    return "funding_pot_id" in {
        row[1] for row in conn.execute("PRAGMA table_info(merchant_orders)")
    }


def _legacy_conn(current_fen: int) -> tuple[sqlite3.Connection, str]:
    """Connection whose ``merchant_orders`` table predates funding_pot_id."""
    customer_id = "cust_legacy"
    conn = connect()
    conn.execute(
        "CREATE TABLE customers (id TEXT PRIMARY KEY, persona_id TEXT NOT NULL, "
        "name TEXT NOT NULL, phone TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE accounts (customer_id TEXT NOT NULL, id TEXT NOT NULL, "
        "type TEXT NOT NULL, balance_fen INTEGER NOT NULL, "
        "PRIMARY KEY (customer_id, id))"
    )
    conn.execute(
        "CREATE TABLE pots (customer_id TEXT NOT NULL, id TEXT NOT NULL, "
        "name TEXT NOT NULL, balance_fen INTEGER NOT NULL, "
        "PRIMARY KEY (customer_id, id))"
    )
    conn.execute(
        "CREATE TABLE merchant_orders (customer_id TEXT NOT NULL, id TEXT NOT NULL, "
        "merchant TEXT NOT NULL, item TEXT NOT NULL, amount_fen INTEGER NOT NULL, "
        "deliver_on TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, "
        "PRIMARY KEY (customer_id, id))"
    )
    conn.execute(
        "INSERT INTO customers (id, persona_id, name, phone) VALUES (?, ?, ?, ?)",
        (customer_id, "legacy", "Legacy", "13000000000"),
    )
    conn.execute(
        "INSERT INTO accounts (customer_id, id, type, balance_fen) "
        "VALUES (?, 'acc_legacy_current', 'current', ?)",
        (customer_id, current_fen),
    )
    conn.commit()
    return conn, customer_id


def test_birthday_orders_debit_pot_and_cancels_refund_it(env) -> None:
    bank, _, conn = env
    current_before = bank.get_balance()

    pot = bank.move_to_pot(POT_NAME, 100_000)
    assert pot.balance_fen == 100_000
    assert bank.get_balance() == current_before - 100_000

    bouquet = bank.merchant_order(
        "花店",
        "bouquet_m",
        TODAY + timedelta(days=1),
        NOW,
        funding_pot_id=pot.id,
    )
    cake = bank.merchant_order(
        "蛋糕店",
        "cake_8in",
        TODAY + timedelta(days=2),
        NOW,
        funding_pot_id=pot.id,
    )

    assert (bouquet.amount_fen, cake.amount_fen) == (29_900, 26_800)
    assert bouquet.funding_pot_id == pot.id
    assert cake.funding_pot_id == pot.id

    # The pot, not the current account, pays for both orders.
    assert bank.get_balance() == current_before - 100_000
    assert _pot_balance(conn, pot.id) == 43_300
    assert (
        (current_before) - (bank.get_balance() + _pot_balance(conn, pot.id))
        == 56_700
    )

    assert bank.cancel_order(bouquet.id, NOW).status == "cancelled"
    assert bank.cancel_order(cake.id, NOW).status == "cancelled"
    assert _pot_balance(conn, pot.id) == 100_000
    assert bank.get_balance() == current_before - 100_000

    recovered = bank.move_from_pot(POT_NAME, 100_000)
    assert recovered.balance_fen == 0
    assert bank.get_balance() == current_before


def test_default_order_still_debits_current(env) -> None:
    bank, _, conn = env
    before = bank.get_balance()

    order = bank.merchant_order(
        "花店", "bouquet_m", TODAY + timedelta(days=1), NOW
    )

    assert order.funding_pot_id is None
    assert bank.get_balance() == before - 29_900
    row = conn.execute(
        "SELECT funding_pot_id FROM merchant_orders WHERE id = ?", (order.id,)
    ).fetchone()
    assert row["funding_pot_id"] is None


def test_funding_pot_is_keyword_only_and_optional() -> None:
    parameter = signature(Bank.merchant_order).parameters["funding_pot_id"]
    assert parameter.kind is Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_insufficient_pot_is_rejected_without_mutation(env) -> None:
    bank, _, conn = env
    pot = bank.move_to_pot(POT_NAME, 10_000)
    current = bank.get_balance()
    orders_before = _order_count(conn)

    with pytest.raises(BankError) as exc:
        bank.merchant_order(
            "花店",
            "bouquet_m",
            TODAY + timedelta(days=1),
            NOW,
            funding_pot_id=pot.id,
        )

    assert exc.value.code == "insufficient_funds"
    assert _pot_balance(conn, pot.id) == 10_000
    assert bank.get_balance() == current
    assert _order_count(conn) == orders_before


def test_cross_customer_pot_is_rejected_without_mutation() -> None:
    state_a = generate("young_pro", SEED, TODAY)
    state_b = generate("retiree", SEED, TODAY)
    conn = connect()
    create_schema(conn)
    load_state(conn, state_a)
    load_state(conn, state_b)
    bank_a = Bank(conn, state_a.customer.id)
    bank_b = Bank(conn, state_b.customer.id)
    try:
        pot_b = bank_b.move_to_pot(POT_NAME, 50_000)
        current_a = bank_a.get_balance()
        pot_b_balance = _pot_balance(conn, pot_b.id)
        orders_before = _order_count(conn)

        with pytest.raises(BankError) as exc:
            bank_a.merchant_order(
                "花店",
                "bouquet_m",
                TODAY + timedelta(days=1),
                NOW,
                funding_pot_id=pot_b.id,
            )

        assert exc.value.code == "not_found"
        assert bank_a.get_balance() == current_a
        assert _pot_balance(conn, pot_b.id) == pot_b_balance
        assert _order_count(conn) == orders_before
    finally:
        conn.close()


def test_funded_order_insert_failure_rolls_back_pot_debit(env) -> None:
    bank, _, conn = env
    pot = bank.move_to_pot(POT_NAME, 100_000)
    conn.execute(
        """
        CREATE TRIGGER fail_order_insert
        BEFORE INSERT ON merchant_orders
        BEGIN
            SELECT RAISE(ABORT, 'forced failure');
        END;
        """
    )
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError):
        bank.merchant_order(
            "花店",
            "bouquet_m",
            TODAY + timedelta(days=1),
            NOW,
            funding_pot_id=pot.id,
        )

    assert _pot_balance(conn, pot.id) == 100_000
    assert _order_count(conn) == 0


def test_funded_order_keeps_existing_cancel_window(env) -> None:
    bank, _, conn = env
    pot = bank.move_to_pot(POT_NAME, 100_000)
    order = bank.merchant_order(
        "蛋糕店", "cake_6in", TODAY, NOW, funding_pot_id=pot.id
    )

    with pytest.raises(BankError) as exc:
        bank.cancel_order(order.id, NOW)

    assert exc.value.code == "window_closed"
    assert _pot_balance(conn, pot.id) == 100_000 - 16_800


def test_legacy_rows_without_funding_column_refund_current() -> None:
    conn, customer_id = _legacy_conn(1_000_000 - 29_900)
    try:
        assert not _has_funding_column(conn)
        conn.execute(
            """
            INSERT INTO merchant_orders
                (customer_id, id, merchant, item, amount_fen, deliver_on,
                 status, created_at)
            VALUES (?, 'ord_legacy', '花店', 'bouquet_m', 29900, ?, 'open', ?)
            """,
            (
                customer_id,
                (TODAY + timedelta(days=1)).isoformat(),
                NOW.isoformat(),
            ),
        )
        conn.commit()

        bank = Bank(conn, customer_id)
        assert _has_funding_column(conn)
        before = bank.get_balance()

        cancelled = bank.cancel_order("ord_legacy", NOW)

        assert cancelled.funding_pot_id is None
        assert bank.get_balance() == before + 29_900
    finally:
        conn.close()


def test_additive_ddl_does_not_commit_callers_outer_transaction() -> None:
    conn, customer_id = _legacy_conn(1_000_000)
    try:
        assert not _has_funding_column(conn)
        balance_sql = (
            "SELECT balance_fen FROM accounts "
            "WHERE customer_id = ? AND type = 'current'"
        )
        before = conn.execute(balance_sql, (customer_id,)).fetchone()[0]

        conn.execute("BEGIN")
        conn.execute(
            "UPDATE accounts SET balance_fen = balance_fen + 7 "
            "WHERE customer_id = ? AND type = 'current'",
            (customer_id,),
        )

        Bank(conn, customer_id)

        # The DDL joined the caller's transaction and is visible inside it.
        assert _has_funding_column(conn)
        conn.rollback()

        # Neither the caller's update nor the schema change was committed.
        assert not _has_funding_column(conn)
        assert conn.execute(balance_sql, (customer_id,)).fetchone()[0] == before
    finally:
        conn.close()
