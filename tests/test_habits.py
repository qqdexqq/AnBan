"""Tests for customer-visible habits and nudges."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta

import pytest

from agent.habits import (
    confirm,
    delete,
    dismiss,
    learn,
    list_items,
    nudges,
    usual_amount,
)
from bank.bills import seed_bills
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
    try:
        yield state, conn
    finally:
        conn.close()


def _account_id(state) -> str:
    return state.accounts[0].id


def _txn(
    conn: sqlite3.Connection,
    state,
    txn_id: str,
    when: datetime,
    amount_fen: int,
    counterparty: str,
    category: str = "transfer",
) -> None:
    conn.execute(
        """
        INSERT INTO transactions
            (customer_id, id, account_id, ts, amount_fen, counterparty,
             category, memo, channel, country, card_present)
        VALUES (?, ?, ?, ?, ?, ?, ?, '', 'transfer', 'CN', 0)
        """,
        (
            state.customer.id,
            txn_id,
            _account_id(state),
            when.isoformat(),
            amount_fen,
            counterparty,
            category,
        ),
    )


def _transfer(
    conn: sqlite3.Connection,
    state,
    transfer_id: str,
    contact_id: str,
    when: datetime,
    amount_fen: int,
    status: str = "released",
) -> None:
    conn.execute(
        """
        INSERT INTO transfers
            (customer_id, id, to_contact_id, amount_fen, note,
             idempotency_key, status, created_at, release_at, account_id)
        VALUES (?, ?, ?, ?, '', ?, ?, ?, ?, ?)
        """,
        (
            state.customer.id,
            transfer_id,
            contact_id,
            amount_fen,
            transfer_id,
            status,
            when.isoformat(),
            (when + timedelta(hours=2)).isoformat(),
            _account_id(state),
        ),
    )


def _contact(state, relation: str):
    return next(contact for contact in state.contacts if contact.relation == relation)


def test_learn_suggests_seed_habits(env) -> None:
    state, conn = env
    landlord = _contact(state, "landlord")

    items = learn(conn, state.customer.id, NOW)
    by_id = {item["id"]: item for item in items}

    monthly = by_id[f"monthly:{landlord.id}"]
    assert monthly["kind"] == "monthly_transfer"
    assert monthly["contact_id"] == landlord.id
    assert monthly["amount_fen"] == 350_000
    assert monthly["day_of_month"] == 1
    assert monthly["status"] == "suggested"

    usual = by_id[f"usual:{landlord.id}"]
    assert usual["kind"] == "usual_amount"
    assert usual["amount_fen"] == 350_000
    assert usual["day_of_month"] is None

    # Generated salary months fall back to the seed's "income" category.
    assert by_id["payday"]["kind"] == "payday"
    assert by_id["payday"]["status"] == "suggested"

    assert [item["id"] for item in items] == sorted(by_id)


def test_confirmation_is_required_and_delete_sticks(env) -> None:
    state, conn = env
    landlord = _contact(state, "landlord")
    customer_id = state.customer.id
    item_id = f"usual:{landlord.id}"

    learn(conn, customer_id, NOW)
    assert usual_amount(conn, customer_id, landlord.id) is None

    confirmed = confirm(conn, customer_id, item_id, NOW)
    assert confirmed["status"] == "confirmed"
    assert usual_amount(conn, customer_id, landlord.id) == 350_000

    deleted = delete(conn, customer_id, item_id, NOW)
    assert deleted["status"] == "deleted"
    assert usual_amount(conn, customer_id, landlord.id) is None
    assert item_id not in {item["id"] for item in list_items(conn, customer_id)}
    assert item_id in {
        item["id"] for item in list_items(conn, customer_id, include_deleted=True)
    }

    learn(conn, customer_id, NOW)
    after = {
        item["id"]: item
        for item in list_items(conn, customer_id, include_deleted=True)
    }
    assert after[item_id]["status"] == "deleted"


def test_unknown_items_raise_key_error(env) -> None:
    state, conn = env
    customer_id = state.customer.id
    with pytest.raises(KeyError):
        confirm(conn, customer_id, "missing", NOW)
    with pytest.raises(KeyError):
        delete(conn, customer_id, "missing", NOW)


def test_monthly_transfer_uses_median_day_and_latest_amount(env) -> None:
    state, conn = env
    mother = _contact(state, "parent")
    _txn(conn, state, "m1", datetime(2026, 8, 7, 9), -1_000, mother.name, "housing")
    _txn(conn, state, "m2", datetime(2026, 9, 8, 9), -2_000, mother.name, "housing")
    _txn(conn, state, "m3", datetime(2026, 10, 9, 9), -3_000, mother.name, "housing")
    conn.commit()

    item = next(
        item
        for item in learn(conn, state.customer.id, NOW)
        if item["id"] == f"monthly:{mother.id}"
    )
    assert item["amount_fen"] == 3_000
    assert item["day_of_month"] == 8
    assert all(
        item["id"] != f"usual:{mother.id}"
        for item in list_items(conn, state.customer.id)
    )


def test_usual_amount_mode_tie_uses_latest(env) -> None:
    state, conn = env
    mother = _contact(state, "parent")
    _txn(conn, state, "u1", datetime(2026, 9, 1, 9), -10_000, mother.name)
    _txn(conn, state, "u2", datetime(2026, 9, 2, 9), -20_000, mother.name)
    _txn(conn, state, "u3", datetime(2026, 10, 1, 9), -10_000, mother.name)
    _txn(conn, state, "u4", datetime(2026, 10, 2, 9), -20_000, mother.name)
    conn.commit()

    item = next(
        item
        for item in learn(conn, state.customer.id, NOW)
        if item["id"] == f"usual:{mother.id}"
    )
    assert item["amount_fen"] == 20_000


def test_learn_refreshes_suggested_but_not_confirmed(env) -> None:
    state, conn = env
    mother = _contact(state, "parent")
    customer_id = state.customer.id

    _txn(conn, state, "r1", datetime(2026, 9, 1, 9), -10_000, mother.name)
    _txn(conn, state, "r2", datetime(2026, 9, 2, 9), -10_000, mother.name)
    conn.commit()
    first = next(
        item
        for item in learn(conn, customer_id, NOW)
        if item["id"] == f"usual:{mother.id}"
    )
    assert first["amount_fen"] == 10_000

    _txn(conn, state, "r3", datetime(2026, 9, 3, 9), -20_000, mother.name)
    _txn(conn, state, "r4", datetime(2026, 9, 4, 9), -20_000, mother.name)
    _txn(conn, state, "r5", datetime(2026, 9, 5, 9), -20_000, mother.name)
    conn.commit()
    refreshed = next(
        item
        for item in learn(conn, customer_id, NOW)
        if item["id"] == f"usual:{mother.id}"
    )
    assert refreshed["amount_fen"] == 20_000

    confirm(conn, customer_id, f"usual:{mother.id}", NOW)
    _txn(conn, state, "r6", datetime(2026, 9, 6, 9), -30_000, mother.name)
    _txn(conn, state, "r7", datetime(2026, 9, 7, 9), -30_000, mother.name)
    _txn(conn, state, "r8", datetime(2026, 9, 8, 9), -30_000, mother.name)
    conn.commit()
    learn(conn, customer_id, NOW)
    assert usual_amount(conn, customer_id, mother.id) == 20_000


def test_transfers_table_is_a_payment_source(env) -> None:
    state, conn = env
    friend = state.contacts[1]
    _transfer(conn, state, "t1", friend.id, datetime(2026, 9, 1, 9), 5_000)
    _transfer(conn, state, "t2", friend.id, datetime(2026, 9, 10, 9), 5_000)
    _transfer(
        conn, state, "t3", friend.id, datetime(2026, 9, 12, 9), 7_000, "cancelled"
    )
    conn.commit()

    item = next(
        item
        for item in learn(conn, state.customer.id, NOW)
        if item["id"] == f"usual:{friend.id}"
    )
    assert item["amount_fen"] == 5_000


def _confirm_payday_and_monthly(conn, state, contact_id, day: int, amount: int) -> str:
    customer_id = state.customer.id
    item_id = f"monthly:{contact_id}"
    learn(conn, customer_id, NOW)
    confirm(conn, customer_id, "payday", NOW)
    confirm(conn, customer_id, item_id, NOW)
    return item_id


def test_payday_transfer_nudge_lifecycle(env) -> None:
    state, conn = env
    friend = state.contacts[1]
    customer_id = state.customer.id
    # Monthly pattern in the previous two months; nothing paid this month.
    _txn(conn, state, "p1", datetime(2026, 8, 15, 9), -10_000, friend.name)
    _txn(conn, state, "p2", datetime(2026, 9, 15, 9), -12_000, friend.name)
    conn.commit()

    item_id = _confirm_payday_and_monthly(conn, state, friend.id, 15, 12_000)
    assert nudges(conn, customer_id, NOW) == []

    _txn(
        conn,
        state,
        "salary1",
        datetime(2026, 10, 16, 9),
        200_000,
        "公司",
        "salary",
    )
    conn.commit()

    found = nudges(conn, customer_id, NOW)
    assert found == [
        {
            "kind": "payday_transfer",
            "contact_id": friend.id,
            "amount_fen": 12_000,
            "item_id": item_id,
            "key": f"payday_transfer:{friend.id}",
        }
    ]

    _txn(conn, state, "paid", datetime(2026, 10, 17, 9), -12_000, friend.name)
    conn.commit()
    assert nudges(conn, customer_id, NOW) == []


def test_payday_nudge_can_be_dismissed(env) -> None:
    state, conn = env
    friend = state.contacts[1]
    customer_id = state.customer.id
    _txn(conn, state, "p1", datetime(2026, 8, 15, 9), -10_000, friend.name)
    _txn(conn, state, "p2", datetime(2026, 9, 15, 9), -12_000, friend.name)
    _txn(conn, state, "salary0", datetime(2026, 9, 16, 9), 190_000, "公司", "salary")
    _txn(conn, state, "salary1", datetime(2026, 10, 16, 9), 200_000, "公司", "salary")
    conn.commit()
    _confirm_payday_and_monthly(conn, state, friend.id, 15, 12_000)

    nudge = nudges(conn, customer_id, NOW)[0]
    assert dismiss(conn, customer_id, nudge["key"], NOW)["key"] == nudge["key"]
    assert nudges(conn, customer_id, NOW) == []


def test_bill_due_nudge_and_dismiss(env) -> None:
    state, conn = env
    customer_id = state.customer.id
    seed_bills(conn, customer_id, TODAY)
    with conn:
        conn.execute(
            "UPDATE bills SET due_date = ?, amount_due_fen = 9999 WHERE customer_id = ?",
            ((TODAY + timedelta(days=10)).isoformat(), customer_id),
        )
        conn.execute(
            """
            UPDATE bills SET due_date = ?, amount_due_fen = 12345
            WHERE customer_id = ? AND category = 'water'
            """,
            ((TODAY + timedelta(days=2)).isoformat(), customer_id),
        )

    found = nudges(conn, customer_id, NOW)
    assert found == [
        {
            "kind": "bill_due",
            "category": "water",
            "amount_due_fen": 12_345,
            "due_date": (TODAY + timedelta(days=2)).isoformat(),
            "key": "bill_due:water",
        }
    ]

    dismiss(conn, customer_id, "bill_due:water", NOW)
    assert nudges(conn, customer_id, NOW) == []


def test_payday_requires_two_calendar_months(env) -> None:
    state, conn = env
    customer_id = state.customer.id
    with conn:
        conn.execute("DELETE FROM transactions WHERE customer_id = ?", (customer_id,))
    _txn(conn, state, "s1", datetime(2026, 10, 16, 9), 100_000, "公司", "salary")
    conn.commit()
    assert all(
        item["id"] != "payday" for item in learn(conn, customer_id, NOW)
    )

    _txn(conn, state, "s2", datetime(2026, 9, 30, 9), 90_000, "公司", "salary")
    conn.commit()
    payday = next(
        item for item in learn(conn, customer_id, NOW) if item["id"] == "payday"
    )
    assert payday["amount_fen"] == 100_000
    assert payday["day_of_month"] == 16
