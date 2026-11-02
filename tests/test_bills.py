"""Tests for utility bills."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta

import pytest

from bank.api import BankError
from bank.bills import CATEGORIES, list_bills, pay_bill, seed_bills, upcoming_bills
from bank.db import connect, create_schema, load_state
from bank.seed import generate

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42


@pytest.fixture
def env():
    state = generate("young_pro", SEED, TODAY)
    conn = connect()
    create_schema(conn)
    load_state(conn, state)
    seed_bills(conn, state.customer.id, TODAY)
    try:
        yield state.customer.id, conn
    finally:
        conn.close()


def _all(conn: sqlite3.Connection, customer_id: str) -> list[dict]:
    return list_bills(conn, customer_id, None)


def _top_account(conn: sqlite3.Connection, customer_id: str) -> sqlite3.Row:
    return conn.execute(
        """
        SELECT * FROM accounts WHERE customer_id = ? AND type = 'current'
        ORDER BY id LIMIT 1
        """,
        (customer_id,),
    ).fetchone()


def test_seed_creates_one_bill_per_category(env) -> None:
    customer_id, conn = env
    bills = _all(conn, customer_id)
    assert len(bills) == 6
    assert {bill["category"] for bill in bills} == set(CATEGORIES)
    for bill in bills:
        assert bill["id"] == f"bill_{bill['category']}"
        assert bill["status"] == "due"
        assert bill["paid_at"] is None
        assert bill["biller"] and bill["biller_en"]
        assert len(bill["account_no"]) == 10 and bill["account_no"].isdigit()
        low, high = {
            "water": (3000, 12000),
            "electricity": (8000, 40000),
            "gas": (3000, 15000),
            "mobile": (5000, 20000),
            "broadband": (9900, 19900),
            "property": (15000, 60000),
        }[bill["category"]]
        assert low <= bill["amount_due_fen"] <= high
        due = date.fromisoformat(bill["due_date"])
        assert TODAY + timedelta(days=3) <= due <= TODAY + timedelta(days=20)


def test_seed_is_idempotent_and_deterministic(env) -> None:
    customer_id, conn = env
    first = _all(conn, customer_id)
    seed_bills(conn, customer_id, TODAY)
    assert _all(conn, customer_id) == first

    other = connect()
    try:
        create_schema(other)
        state = generate("young_pro", SEED, TODAY)
        load_state(other, state)
        seed_bills(other, state.customer.id, TODAY)
        assert _all(other, state.customer.id) == first
    finally:
        other.close()


def test_list_bills_status_filter(env) -> None:
    customer_id, conn = env
    bill = list_bills(conn, customer_id)[0]
    pay_bill(conn, customer_id, bill["category"], NOW)

    assert all(row["status"] == "due" for row in list_bills(conn, customer_id))
    assert bill["id"] not in {row["id"] for row in list_bills(conn, customer_id)}
    paid = list_bills(conn, customer_id, "paid")
    assert [row["id"] for row in paid] == [bill["id"]]
    assert len(_all(conn, customer_id)) == 6


def test_upcoming_bills_filters_by_days(env) -> None:
    customer_id, conn = env
    horizon = (TODAY + timedelta(days=10)).isoformat()
    result = upcoming_bills(conn, customer_id, TODAY, 10)
    expected = [
        bill for bill in list_bills(conn, customer_id) if bill["due_date"] <= horizon
    ]
    assert [bill["id"] for bill in result] == [bill["id"] for bill in expected]
    assert all(bill["due_date"] <= horizon for bill in result)
    assert len(upcoming_bills(conn, customer_id, TODAY, 30)) == 6


def test_pay_bill_reduces_balance_and_marks_paid(env) -> None:
    customer_id, conn = env
    bill = list_bills(conn, customer_id)[0]
    account = _top_account(conn, customer_id)
    before = int(account["balance_fen"])

    paid = pay_bill(conn, customer_id, bill["category"], NOW)

    assert paid["id"] == bill["id"]
    assert paid["status"] == "paid"
    assert paid["paid_at"] == NOW.isoformat()
    assert paid["account_id"] == account["id"]
    balance = conn.execute(
        "SELECT balance_fen FROM accounts WHERE customer_id = ? AND id = ?",
        (customer_id, account["id"]),
    ).fetchone()[0]
    assert balance == before - bill["amount_due_fen"]

    txn = conn.execute(
        "SELECT * FROM transactions WHERE customer_id = ? AND category = 'utilities'",
        (customer_id,),
    ).fetchone()
    assert txn is not None
    assert txn["amount_fen"] == -bill["amount_due_fen"]
    assert txn["counterparty"] == bill["biller"]
    assert txn["ts"] == NOW.isoformat()


def test_double_pay_raises_not_found(env) -> None:
    customer_id, conn = env
    bill = list_bills(conn, customer_id)[0]
    account = _top_account(conn, customer_id)
    pay_bill(conn, customer_id, bill["category"], NOW)
    after_first = conn.execute(
        "SELECT balance_fen FROM accounts WHERE customer_id = ? AND id = ?",
        (customer_id, account["id"]),
    ).fetchone()[0]

    with pytest.raises(BankError) as exc:
        pay_bill(conn, customer_id, bill["category"], NOW)
    assert exc.value.code == "not_found"
    assert conn.execute(
        "SELECT balance_fen FROM accounts WHERE customer_id = ? AND id = ?",
        (customer_id, account["id"]),
    ).fetchone()[0] == after_first


def test_insufficient_funds_changes_nothing(env) -> None:
    customer_id, conn = env
    with conn:
        conn.execute(
            "UPDATE accounts SET balance_fen = 0 WHERE customer_id = ?",
            (customer_id,),
        )
    before = _all(conn, customer_id)

    with pytest.raises(BankError) as exc:
        pay_bill(conn, customer_id, before[0]["category"], NOW)
    assert exc.value.code == "insufficient_funds"

    assert _all(conn, customer_id) == before
    assert all(
        row[0] == 0
        for row in conn.execute(
            "SELECT balance_fen FROM accounts WHERE customer_id = ?", (customer_id,)
        )
    )
    assert conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE customer_id = ? AND category = 'utilities'",
        (customer_id,),
    ).fetchone()[0] == 0


def test_invalid_category_raises_invalid_state(env) -> None:
    customer_id, conn = env
    before = _all(conn, customer_id)
    with pytest.raises(BankError) as exc:
        pay_bill(conn, customer_id, "rent", NOW)
    assert exc.value.code == "invalid_state"
    assert _all(conn, customer_id) == before
