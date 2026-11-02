"""Tests for unattended monthly standing orders."""

from __future__ import annotations

import sqlite3
import threading
from datetime import date, datetime, timedelta

import pytest

from bank.api import Bank
from bank.db import connect, create_schema, load_state
from bank.errand_payments import open_errand
from bank.seed import generate
from bank.standing_orders import (
    run_due_standing_orders,
    standing_order_overview,
)
from security import compliance

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
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


def _balance(conn: sqlite3.Connection, customer_id: str) -> int:
    return int(
        conn.execute(
            """
            SELECT balance_fen FROM accounts
            WHERE customer_id = ? AND type = 'current'
            """,
            (customer_id,),
        ).fetchone()[0]
    )


def _count(conn: sqlite3.Connection, table: str, customer_id: str) -> int:
    return int(
        conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()[0]
    )


def _schedule(
    conn: sqlite3.Connection,
    customer_id: str,
    contact_id: str,
    *,
    amount_fen: int = 1_000,
    day_of_month: int = 5,
    note: str = "rent",
) -> str:
    scheduled = Bank(conn, customer_id).schedule_transfer(
        contact_id, amount_fen, day_of_month, note
    )
    return scheduled.id


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
def retiree_env():
    state = generate("retiree", SEED, TODAY)
    conn = connect()
    create_schema(conn)
    load_state(conn, state)
    try:
        yield conn, state
    finally:
        conn.close()


def test_first_call_never_pays_even_when_the_day_passed(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    contact = state.contacts[0].id
    _schedule(conn, customer_id, contact, day_of_month=5)

    runs = run_due_standing_orders(conn, customer_id, datetime(2026, 10, 20, 12))

    assert runs == []
    assert _count(conn, "transfers", customer_id) == 0
    assert _count(conn, "standing_order_seen", customer_id) == 1
    assert _count(conn, "standing_order_runs", customer_id) == 0


def test_due_day_next_month_pays_exactly_once(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    contact = state.contacts[0].id
    scheduled_id = _schedule(conn, customer_id, contact, note="rent")
    run_due_standing_orders(conn, customer_id, NOW)

    first = run_due_standing_orders(conn, customer_id, datetime(2026, 11, 5, 12))
    second = run_due_standing_orders(conn, customer_id, datetime(2026, 11, 5, 18))

    assert first == [
        {
            "scheduled_id": scheduled_id,
            "occurrence": "2026-11",
            "status": "paid",
            "reason": None,
            "amount_fen": 1_000,
            "transfer_id": first[0]["transfer_id"],
        }
    ]
    assert second == []
    assert _count(conn, "transfers", customer_id) == 1
    assert _count(conn, "standing_order_runs", customer_id) == 1
    transfer = conn.execute(
        "SELECT * FROM transfers WHERE customer_id = ?", (customer_id,)
    ).fetchone()
    assert transfer["status"] == "pending"
    assert transfer["note"] == "rent"
    assert int(transfer["amount_fen"]) == 1_000


def test_two_connections_pay_only_once(tmp_path) -> None:
    path = str(tmp_path / "bank.sqlite")
    state = generate("young_pro", SEED, TODAY)
    customer_id = state.customer.id
    contact = state.contacts[0].id

    setup = connect(path)
    create_schema(setup)
    load_state(setup, state)
    _schedule(setup, customer_id, contact, note="rent")
    run_due_standing_orders(setup, customer_id, NOW)
    setup.close()

    results: list[list[dict[str, object]]] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        worker_conn = connect(path)
        try:
            barrier.wait()
            results.append(
                run_due_standing_orders(
                    worker_conn, customer_id, datetime(2026, 11, 5, 12)
                )
            )
        except BaseException as exc:  # noqa: BLE001 - surface thread failures
            errors.append(exc)
        finally:
            worker_conn.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sum(len(result) for result in results) == 1

    check = connect(path)
    try:
        assert _count(check, "standing_order_runs", customer_id) == 1
        paid = check.execute(
            "SELECT * FROM standing_order_runs WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
        assert paid["status"] == "paid"
        assert _count(check, "transfers", customer_id) == 1
    finally:
        check.close()


def test_missed_month_is_recorded_and_current_month_pays(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    contact = state.contacts[0].id
    _schedule(conn, customer_id, contact, day_of_month=10)
    run_due_standing_orders(conn, customer_id, datetime(2026, 10, 20, 12))
    before = _balance(conn, customer_id)

    runs = run_due_standing_orders(conn, customer_id, datetime(2026, 12, 20, 12))

    by_occurrence = {run["occurrence"]: run for run in runs}
    assert by_occurrence["2026-11"]["status"] == "missed"
    assert by_occurrence["2026-11"]["reason"] == "not_run_on_time"
    assert by_occurrence["2026-11"]["transfer_id"] is None
    assert by_occurrence["2026-12"]["status"] == "paid"
    assert _count(conn, "transfers", customer_id) == 1
    assert _balance(conn, customer_id) == before - 1_000


def test_lockout_skips_without_paying(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    contact = state.contacts[0].id
    _schedule(conn, customer_id, contact)
    run_due_standing_orders(conn, customer_id, NOW)
    _lockout(conn, customer_id, datetime(2026, 11, 5, 13))

    runs = run_due_standing_orders(conn, customer_id, datetime(2026, 11, 5, 12))

    assert [(run["status"], run["reason"]) for run in runs] == [
        ("skipped", "payments_locked")
    ]
    assert runs[0]["transfer_id"] is None
    assert _count(conn, "transfers", customer_id) == 0


def test_scam_freeze_skips_without_paying(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    contact = state.contacts[0].id
    _schedule(conn, customer_id, contact)
    run_due_standing_orders(conn, customer_id, NOW)
    _scam_freeze(conn, customer_id, datetime(2026, 11, 5, 13))

    runs = run_due_standing_orders(conn, customer_id, datetime(2026, 11, 5, 12))

    assert [(run["status"], run["reason"]) for run in runs] == [
        ("skipped", "scam_freeze")
    ]
    assert _count(conn, "transfers", customer_id) == 0


def test_missing_payee_skips_without_paying(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    contact = state.contacts[0].id
    _schedule(conn, customer_id, contact)
    run_due_standing_orders(conn, customer_id, NOW)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        "DELETE FROM contacts WHERE customer_id = ? AND id = ?",
        (customer_id, contact),
    )
    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")

    runs = run_due_standing_orders(conn, customer_id, datetime(2026, 11, 5, 12))

    assert [(run["status"], run["reason"]) for run in runs] == [
        ("skipped", "payee_missing")
    ]
    assert _count(conn, "transfers", customer_id) == 0


def test_unverified_payee_skips_without_paying(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    contact = state.contacts[0].id
    _schedule(conn, customer_id, contact)
    run_due_standing_orders(conn, customer_id, NOW)
    conn.execute(
        "UPDATE contacts SET verified = 0 WHERE customer_id = ? AND id = ?",
        (customer_id, contact),
    )
    conn.commit()

    runs = run_due_standing_orders(conn, customer_id, datetime(2026, 11, 5, 12))

    assert [(run["status"], run["reason"]) for run in runs] == [
        ("skipped", "payee_unverified")
    ]
    assert _count(conn, "transfers", customer_id) == 0


def test_daily_limit_skips_without_paying(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    contact = state.contacts[0].id
    _schedule(conn, customer_id, contact, amount_fen=1_000)
    run_due_standing_orders(conn, customer_id, NOW)
    due = datetime(2026, 11, 5, 12)
    Bank(conn, customer_id).transfer(contact, 99_500, "other", "daily-push", due)

    runs = run_due_standing_orders(conn, customer_id, due)

    assert [(run["status"], run["reason"]) for run in runs] == [
        ("skipped", "over_daily_limit")
    ]
    assert _count(conn, "transfers", customer_id) == 1


def test_elder_limit_skips_without_paying(retiree_env) -> None:
    conn, state = retiree_env
    customer_id = state.customer.id
    contact = state.contacts[0].id
    _schedule(conn, customer_id, contact, amount_fen=60_000)
    run_due_standing_orders(conn, customer_id, NOW)

    runs = run_due_standing_orders(conn, customer_id, datetime(2026, 11, 5, 12))

    assert [(run["status"], run["reason"]) for run in runs] == [
        ("skipped", "elder_limit")
    ]
    assert _count(conn, "transfers", customer_id) == 0


def test_elder_daily_limit_is_cumulative(retiree_env) -> None:
    conn, state = retiree_env
    customer_id = state.customer.id
    contact = state.contacts[0].id
    first_id = _schedule(conn, customer_id, contact, amount_fen=40_000, note="first")
    second_id = _schedule(conn, customer_id, contact, amount_fen=40_000, note="second")
    run_due_standing_orders(conn, customer_id, NOW)

    runs = run_due_standing_orders(conn, customer_id, datetime(2026, 11, 5, 12))
    by_order = {run["scheduled_id"]: run for run in runs}

    assert by_order[first_id]["status"] == "paid"
    assert (by_order[second_id]["status"], by_order[second_id]["reason"]) == (
        "skipped",
        "elder_limit",
    )
    assert _count(conn, "transfers", customer_id) == 1


def test_daily_limit_includes_errand_budgets_reserved_today(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    contact = state.contacts[0].id
    due = datetime(2026, 11, 5, 12)
    _schedule(conn, customer_id, contact, amount_fen=60_000, day_of_month=5)
    run_due_standing_orders(conn, customer_id, NOW)
    open_errand(
        conn,
        customer_id,
        ticket_id="errand-today",
        category="flowers",
        budget_fen=90_000,
        deliver_by=due.date(),
        expires_at=due + timedelta(hours=2),
        now=due,
    )

    runs = run_due_standing_orders(conn, customer_id, due)

    assert [(run["status"], run["reason"]) for run in runs] == [
        ("skipped", "over_daily_limit")
    ]
    assert _count(conn, "transfers", customer_id) == 0


def test_transfer_uses_the_customers_arrival_delay(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    contact = state.contacts[0].id
    due = datetime(2026, 11, 5, 12)
    _schedule(conn, customer_id, contact, day_of_month=5)
    run_due_standing_orders(conn, customer_id, NOW)
    compliance.update_settings(
        conn, customer_id, {"arrival_delay": "24h"}, now=NOW
    )

    runs = run_due_standing_orders(conn, customer_id, due)

    assert [(run["status"], run["reason"]) for run in runs] == [("paid", None)]
    transfer = conn.execute(
        "SELECT created_at, release_at FROM transfers WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()
    assert datetime.fromisoformat(transfer["release_at"]) - datetime.fromisoformat(
        transfer["created_at"]
    ) == timedelta(hours=24)


def test_one_failing_order_does_not_stop_the_rest(env, monkeypatch) -> None:
    conn, state = env
    customer_id = state.customer.id
    contact = state.contacts[0].id
    first_id = _schedule(conn, customer_id, contact, amount_fen=1_000, note="first")
    second_id = _schedule(conn, customer_id, contact, amount_fen=2_000, note="second")
    run_due_standing_orders(conn, customer_id, NOW)

    original_transfer = Bank.transfer
    calls: list[str] = []

    def flaky_transfer(self, *args, **kwargs):
        calls.append(str(args[0]))
        if len(calls) == 1:
            raise RuntimeError("simulated failure")
        return original_transfer(self, *args, **kwargs)

    monkeypatch.setattr(Bank, "transfer", flaky_transfer)

    runs = run_due_standing_orders(conn, customer_id, datetime(2026, 11, 5, 12))
    by_order = {run["scheduled_id"]: run for run in runs}

    assert (by_order[first_id]["status"], by_order[first_id]["reason"]) == (
        "skipped",
        "error",
    )
    assert by_order[second_id]["status"] == "paid"
    assert _count(conn, "transfers", customer_id) == 1
    assert _count(conn, "standing_order_runs", customer_id) == 2


def test_insufficient_funds_skips_without_paying(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    contact = state.contacts[0].id
    _schedule(conn, customer_id, contact, amount_fen=5_000)
    run_due_standing_orders(conn, customer_id, NOW)
    conn.execute(
        """
        UPDATE accounts SET balance_fen = ?
        WHERE customer_id = ? AND type = 'current'
        """,
        (4_999, customer_id),
    )
    conn.commit()

    runs = run_due_standing_orders(conn, customer_id, datetime(2026, 11, 5, 12))

    assert [(run["status"], run["reason"]) for run in runs] == [
        ("skipped", "insufficient_funds")
    ]
    assert _count(conn, "transfers", customer_id) == 0


def test_cancelled_order_never_runs(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    contact = state.contacts[0].id
    scheduled_id = _schedule(conn, customer_id, contact)
    run_due_standing_orders(conn, customer_id, NOW)
    Bank(conn, customer_id).cancel_scheduled(scheduled_id)

    runs = run_due_standing_orders(conn, customer_id, datetime(2026, 11, 5, 12))

    assert runs == []
    assert _count(conn, "transfers", customer_id) == 0
    overview = standing_order_overview(conn, customer_id, datetime(2026, 11, 6, 12))
    assert overview[0]["status"] == "cancelled"
    assert overview[0]["next_due"] is None


def test_overview_shows_next_due_and_last_run(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    contact = state.contacts[0].id
    scheduled_id = _schedule(conn, customer_id, contact, note="rent")
    run_due_standing_orders(conn, customer_id, NOW)

    before = standing_order_overview(conn, customer_id, NOW)
    assert before == [
        {
            "scheduled_id": scheduled_id,
            "contact_id": contact,
            "amount_fen": 1_000,
            "day_of_month": 5,
            "status": "scheduled",
            "note": "rent",
            "next_due": "2026-11-05",
            "last_run": None,
        }
    ]

    run_due_standing_orders(conn, customer_id, datetime(2026, 11, 5, 12))
    after = standing_order_overview(conn, customer_id, datetime(2026, 11, 6, 12))

    assert after[0]["next_due"] == "2026-12-05"
    assert after[0]["last_run"] == {
        "occurrence": "2026-11",
        "status": "paid",
        "reason": None,
        "amount_fen": 1_000,
    }
