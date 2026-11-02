"""Tests for the one-time-card errand payment gateway."""

from __future__ import annotations

import importlib
import sqlite3
from datetime import date, datetime, timedelta

import pytest

import bank.api as bank_api
import bank.errand_payments as errand_payments
from bank.api import Bank, BankError
from bank.db import connect, create_schema, load_state
from bank.errand_payments import (
    authorize_charge,
    close_errand,
    errand_status,
    open_errand,
)
from bank.seed import generate

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
DELIVER_BY = TODAY + timedelta(days=2)
EXPIRES_AT = NOW + timedelta(hours=1)
SEED = 42

LOCKOUTS_DDL = """
CREATE TABLE security_lockouts (
    customer_id TEXT PRIMARY KEY,
    locked_until TEXT NOT NULL,
    reason TEXT NOT NULL
)
"""

SCAM_FREEZES_DDL = """
CREATE TABLE security_scam_freezes (
    customer_id TEXT PRIMARY KEY,
    frozen_until TEXT NOT NULL,
    flags_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


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


def _card(conn: sqlite3.Connection, customer_id: str, card_id: str) -> sqlite3.Row:
    return conn.execute(
        "SELECT * FROM cards WHERE customer_id = ? AND id = ?",
        (customer_id, card_id),
    ).fetchone()


def _txn_count(conn: sqlite3.Connection, customer_id: str) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()[0]
    )


def _shopping_count(conn: sqlite3.Connection, customer_id: str) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(*) FROM transactions
            WHERE customer_id = ? AND category = 'shopping'
            """,
            (customer_id,),
        ).fetchone()[0]
    )


def _open(
    conn: sqlite3.Connection,
    customer_id: str,
    ticket_id: str = "ticket_1",
    budget_fen: int = 5000,
    category: str = "flowers",
    deliver_by: date = DELIVER_BY,
    expires_at: datetime = EXPIRES_AT,
) -> dict[str, object]:
    return open_errand(
        conn,
        customer_id,
        ticket_id=ticket_id,
        category=category,
        budget_fen=budget_fen,
        deliver_by=deliver_by,
        expires_at=expires_at,
        now=NOW,
    )


def _lockout(conn: sqlite3.Connection, customer_id: str, until: datetime) -> None:
    conn.execute(LOCKOUTS_DDL)
    conn.execute(
        "INSERT INTO security_lockouts (customer_id, locked_until, reason) "
        "VALUES (?, ?, 'test')",
        (customer_id, until.isoformat()),
    )
    conn.commit()


def _scam_freeze(conn: sqlite3.Connection, customer_id: str, until: datetime) -> None:
    conn.execute(SCAM_FREEZES_DDL)
    conn.execute(
        "INSERT INTO security_scam_freezes "
        "(customer_id, frozen_until, flags_json, updated_at) VALUES (?, ?, '[]', ?)",
        (customer_id, until.isoformat(), NOW.isoformat()),
    )
    conn.commit()


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


def test_open_uses_no_money_and_creates_active_virtual_card(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    before = _current_balance(conn, customer_id)

    opened = _open(conn, customer_id, budget_fen=7000)

    assert opened == {
        "ticket_id": "ticket_1",
        "card_id": opened["card_id"],
        "budget_fen": 7000,
        "status": "issued",
    }
    assert _current_balance(conn, customer_id) == before
    card = _card(conn, customer_id, str(opened["card_id"]))
    assert card["kind"] == "virtual"
    assert card["status"] == "active"
    assert int(card["online_limit_fen"]) == 7000
    assert int(_txn_count(conn, customer_id)) == _txn_count(conn, customer_id)
    status = errand_status(conn, customer_id, "ticket_1")
    assert status["status"] == "issued"
    assert status["charged_fen"] is None
    assert status["card_id"] == opened["card_id"]


def test_charge_debits_once_writes_one_shopping_row_and_closes_card(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    opened = _open(conn, customer_id, budget_fen=5000)
    before = _current_balance(conn, customer_id)
    before_txns = _txn_count(conn, customer_id)
    before_shopping = _shopping_count(conn, customer_id)

    receipt = authorize_charge(
        conn,
        customer_id,
        ticket_id="ticket_1",
        card_id=str(opened["card_id"]),
        amount_fen=4200,
        merchant="Lily's Flowers",
        order_ref="ORDER-1",
        now=NOW,
    )

    assert receipt["replayed"] is False
    assert receipt["charged_fen"] == 4200
    assert _current_balance(conn, customer_id) == before - 4200
    assert _txn_count(conn, customer_id) == before_txns + 1
    assert _shopping_count(conn, customer_id) == before_shopping + 1
    txn = conn.execute(
        "SELECT * FROM transactions WHERE id = ?", (receipt["txn_id"],)
    ).fetchone()
    assert txn["category"] == "shopping"
    assert txn["channel"] == "errand"
    assert int(txn["amount_fen"]) == -4200
    assert txn["memo"] == "ORDER-1"
    assert _card(conn, customer_id, str(opened["card_id"]))["status"] == "closed"
    status = errand_status(conn, customer_id, "ticket_1")
    assert status["status"] == "charged"
    assert status["charged_fen"] == 4200
    assert status["order_ref"] == "ORDER-1"
    assert status["txn_id"] == receipt["txn_id"]


def test_same_order_ref_replay_returns_stored_receipt(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    opened = _open(conn, customer_id)
    first = authorize_charge(
        conn,
        customer_id,
        ticket_id="ticket_1",
        card_id=str(opened["card_id"]),
        amount_fen=3000,
        merchant="Lily's Flowers",
        order_ref="ORDER-1",
        now=NOW,
    )
    balance = _current_balance(conn, customer_id)
    txns = _txn_count(conn, customer_id)

    replay = authorize_charge(
        conn,
        customer_id,
        ticket_id="ticket_1",
        card_id=str(opened["card_id"]),
        amount_fen=3000,
        merchant="Lily's Flowers",
        order_ref="ORDER-1",
        now=NOW,
    )

    assert replay["replayed"] is True
    assert replay["txn_id"] == first["txn_id"]
    assert _current_balance(conn, customer_id) == balance
    assert _txn_count(conn, customer_id) == txns


def test_different_order_ref_after_charge_is_ticket_used(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    opened = _open(conn, customer_id)
    authorize_charge(
        conn,
        customer_id,
        ticket_id="ticket_1",
        card_id=str(opened["card_id"]),
        amount_fen=3000,
        merchant="Lily's Flowers",
        order_ref="ORDER-1",
        now=NOW,
    )

    with pytest.raises(BankError) as exc:
        authorize_charge(
            conn,
            customer_id,
            ticket_id="ticket_1",
            card_id=str(opened["card_id"]),
            amount_fen=3000,
            merchant="Lily's Flowers",
            order_ref="ORDER-2",
            now=NOW,
        )
    assert exc.value.code == "ticket_used"


def test_amount_over_budget_is_over_limit_and_moves_no_money(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    opened = _open(conn, customer_id, budget_fen=5000)
    before = _current_balance(conn, customer_id)

    with pytest.raises(BankError) as exc:
        authorize_charge(
            conn,
            customer_id,
            ticket_id="ticket_1",
            card_id=str(opened["card_id"]),
            amount_fen=5001,
            merchant="Lily's Flowers",
            order_ref="ORDER-1",
            now=NOW,
        )
    assert exc.value.code == "over_limit"
    assert _current_balance(conn, customer_id) == before
    assert _card(conn, customer_id, str(opened["card_id"]))["status"] == "active"
    assert errand_status(conn, customer_id, "ticket_1")["status"] == "issued"


def test_wrong_card_is_refused(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    _open(conn, customer_id)

    with pytest.raises(BankError) as exc:
        authorize_charge(
            conn,
            customer_id,
            ticket_id="ticket_1",
            card_id="card_physical_1_young_pro",
            amount_fen=3000,
            merchant="Lily's Flowers",
            order_ref="ORDER-1",
            now=NOW,
        )
    assert exc.value.code == "wrong_card"


def test_charge_after_close_is_ticket_closed(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    opened = _open(conn, customer_id)
    close_errand(conn, customer_id, ticket_id="ticket_1", now=NOW)

    with pytest.raises(BankError) as exc:
        authorize_charge(
            conn,
            customer_id,
            ticket_id="ticket_1",
            card_id=str(opened["card_id"]),
            amount_fen=3000,
            merchant="Lily's Flowers",
            order_ref="ORDER-1",
            now=NOW,
        )
    assert exc.value.code == "ticket_closed"


def test_close_is_idempotent_and_closes_card(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    opened = _open(conn, customer_id)

    first = close_errand(conn, customer_id, ticket_id="ticket_1", now=NOW)
    second = close_errand(conn, customer_id, ticket_id="ticket_1", now=NOW)

    assert first == {"ticket_id": "ticket_1", "status": "closed"}
    assert second == first
    assert _card(conn, customer_id, str(opened["card_id"]))["status"] == "closed"


def test_close_after_charge_is_ticket_used(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    opened = _open(conn, customer_id)
    authorize_charge(
        conn,
        customer_id,
        ticket_id="ticket_1",
        card_id=str(opened["card_id"]),
        amount_fen=3000,
        merchant="Lily's Flowers",
        order_ref="ORDER-1",
        now=NOW,
    )

    with pytest.raises(BankError) as exc:
        close_errand(conn, customer_id, ticket_id="ticket_1", now=NOW)
    assert exc.value.code == "ticket_used"


def test_open_without_enough_money_is_insufficient_funds(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    conn.execute(
        """
        UPDATE accounts SET balance_fen = 500
        WHERE customer_id = ? AND type = 'current'
        """,
        (customer_id,),
    )
    conn.commit()

    with pytest.raises(BankError) as exc:
        _open(conn, customer_id, budget_fen=1000)
    assert exc.value.code == "insufficient_funds"
    assert _current_balance(conn, customer_id) == 500


@pytest.mark.parametrize("budget_fen", [0, 100_001, True])
def test_invalid_budget_is_invalid_amount(env, budget_fen) -> None:
    conn, state = env
    customer_id = state.customer.id

    with pytest.raises(BankError) as exc:
        _open(conn, customer_id, budget_fen=budget_fen)
    assert exc.value.code == "invalid_amount"


def test_mid_transaction_failure_rolls_everything_back(env, monkeypatch) -> None:
    conn, state = env
    customer_id = state.customer.id
    opened = _open(conn, customer_id, budget_fen=5000)
    balance = _current_balance(conn, customer_id)
    before_shopping = _shopping_count(conn, customer_id)
    real_next_id = errand_payments._next_id

    def failing_next_id(conn: sqlite3.Connection, prefix: str) -> str:
        if prefix == "txn":
            raise RuntimeError("simulated transaction id failure")
        return real_next_id(conn, prefix)

    monkeypatch.setattr(errand_payments, "_next_id", failing_next_id)

    with pytest.raises(RuntimeError):
        authorize_charge(
            conn,
            customer_id,
            ticket_id="ticket_1",
            card_id=str(opened["card_id"]),
            amount_fen=4200,
            merchant="Lily's Flowers",
            order_ref="ORDER-1",
            now=NOW,
        )

    assert _current_balance(conn, customer_id) == balance
    assert _card(conn, customer_id, str(opened["card_id"]))["status"] == "active"
    status = errand_status(conn, customer_id, "ticket_1")
    assert status["status"] == "issued"
    assert status["charged_fen"] is None
    assert _shopping_count(conn, customer_id) == before_shopping


def test_another_customer_cannot_charge_the_ticket(multi_env) -> None:
    conn, state_a, state_b = multi_env
    opened = _open(conn, state_a.customer.id)

    with pytest.raises(BankError) as exc:
        authorize_charge(
            conn,
            state_b.customer.id,
            ticket_id="ticket_1",
            card_id=str(opened["card_id"]),
            amount_fen=3000,
            merchant="Lily's Flowers",
            order_ref="ORDER-1",
            now=NOW,
        )
    assert exc.value.code == "not_found"


def test_open_while_locked_is_r_locked(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    _lockout(conn, customer_id, NOW + timedelta(hours=1))

    with pytest.raises(BankError) as exc:
        _open(conn, customer_id)
    assert exc.value.code == "R_LOCKED"


def test_charge_while_locked_is_r_locked(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    opened = _open(conn, customer_id)
    _lockout(conn, customer_id, NOW + timedelta(hours=1))

    with pytest.raises(BankError) as exc:
        authorize_charge(
            conn,
            customer_id,
            ticket_id="ticket_1",
            card_id=str(opened["card_id"]),
            amount_fen=3000,
            merchant="Lily's Flowers",
            order_ref="ORDER-1",
            now=NOW,
        )
    assert exc.value.code == "R_LOCKED"


def test_open_while_scam_frozen_is_r_scam(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    _scam_freeze(conn, customer_id, NOW + timedelta(hours=1))

    with pytest.raises(BankError) as exc:
        _open(conn, customer_id)
    assert exc.value.code == "R_SCAM"


def test_charge_while_scam_frozen_is_r_scam(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    opened = _open(conn, customer_id)
    _scam_freeze(conn, customer_id, NOW + timedelta(hours=1))

    with pytest.raises(BankError) as exc:
        authorize_charge(
            conn,
            customer_id,
            ticket_id="ticket_1",
            card_id=str(opened["card_id"]),
            amount_fen=3000,
            merchant="Lily's Flowers",
            order_ref="ORDER-1",
            now=NOW,
        )
    assert exc.value.code == "R_SCAM"


def test_expired_charge_is_ticket_expired(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    opened = _open(conn, customer_id, expires_at=NOW + timedelta(hours=1))

    with pytest.raises(BankError) as exc:
        authorize_charge(
            conn,
            customer_id,
            ticket_id="ticket_1",
            card_id=str(opened["card_id"]),
            amount_fen=3000,
            merchant="Lily's Flowers",
            order_ref="ORDER-1",
            now=NOW + timedelta(hours=1),
        )
    assert exc.value.code == "ticket_expired"


def test_replay_after_expiry_returns_stored_receipt(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    opened = _open(conn, customer_id, expires_at=NOW + timedelta(hours=1))
    first = authorize_charge(
        conn,
        customer_id,
        ticket_id="ticket_1",
        card_id=str(opened["card_id"]),
        amount_fen=3000,
        merchant="Lily's Flowers",
        order_ref="ORDER-1",
        now=NOW,
    )

    replay = authorize_charge(
        conn,
        customer_id,
        ticket_id="ticket_1",
        card_id=str(opened["card_id"]),
        amount_fen=3000,
        merchant="Lily's Flowers",
        order_ref="ORDER-1",
        now=NOW + timedelta(hours=2),
    )

    assert replay["replayed"] is True
    assert replay["txn_id"] == first["txn_id"]


def test_daily_limit_blocks_second_ticket(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    _open(conn, customer_id, ticket_id="ticket_600", budget_fen=60_000)

    with pytest.raises(BankError) as exc:
        _open(conn, customer_id, ticket_id="ticket_500", budget_fen=50_000)
    assert exc.value.code == "over_limit"


def test_charged_errand_counts_towards_daily_outgoing_total(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    before = Bank(conn, customer_id).daily_outgoing_total(NOW)
    opened = _open(conn, customer_id, budget_fen=60_000)

    authorize_charge(
        conn,
        customer_id,
        ticket_id="ticket_1",
        card_id=str(opened["card_id"]),
        amount_fen=60_000,
        merchant="Lily's Flowers",
        order_ref="ORDER-1",
        now=NOW,
    )

    assert Bank(conn, customer_id).daily_outgoing_total(NOW) == before + 60_000


@pytest.mark.parametrize(
    "overrides",
    [
        {"ticket_id": "a" * 33},
        {"ticket_id": "a b"},
        {"deliver_by": datetime(2026, 10, 17, 12, 0, 0)},
        {"deliver_by": TODAY - timedelta(days=1)},
        {"expires_at": NOW},
        {"expires_at": NOW - timedelta(seconds=1)},
        {"expires_at": NOW + timedelta(hours=4, seconds=1)},
    ],
)
def test_invalid_open_arguments_are_invalid_state(env, overrides) -> None:
    conn, state = env
    with pytest.raises(BankError) as exc:
        _open(conn, state.customer.id, **overrides)
    assert exc.value.code == "invalid_state"


def test_import_does_not_mutate_api_error_codes() -> None:
    before = set(bank_api.ERROR_CODES)
    importlib.reload(errand_payments)
    assert bank_api.ERROR_CODES == before
