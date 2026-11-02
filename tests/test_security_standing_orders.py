"""Independent adversarial standing-order regressions; scratch databases only."""
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone

import pytest

from agent.standing_order_hook import run_and_audit
from bank.api import Bank
from bank.db import connect, create_schema, load_state
from bank.seed import generate
from bank import standing_orders
from security import compliance
from security.audit import AuditKind, AuditLog, create_audit_schema

SEEN = datetime(2026, 10, 17, 12)
DUE = datetime(2026, 11, 5, 12)


@pytest.fixture
def scratch(tmp_path):
    conn = connect(str(tmp_path / "standing.sqlite3"))
    state = generate("young_pro", 42, SEEN.date())
    create_schema(conn)
    load_state(conn, state)
    try:
        yield conn, state
    finally:
        conn.close()


def balance(conn, customer):
    return conn.execute("SELECT balance_fen FROM accounts WHERE customer_id=? AND type='current'", (customer,)).fetchone()[0]


def arm(conn, state, amount=1000, day=5):
    order = Bank(conn, state.customer.id).schedule_transfer(state.contacts[0].id, amount, day, "security test")
    standing_orders.run_due_standing_orders(conn, state.customer.id, SEEN)
    return order.id


def test_overview_does_not_commit_unrelated_caller_transaction(scratch):
    conn, state = scratch
    arm(conn, state)
    before = balance(conn, state.customer.id)
    conn.execute("UPDATE accounts SET balance_fen=balance_fen+123 WHERE customer_id=? AND type='current'", (state.customer.id,))
    standing_orders.standing_order_overview(conn, state.customer.id, DUE)
    assert conn.in_transaction, "Read-only overview committed unrelated account mutations"
    conn.rollback()
    assert balance(conn, state.customer.id) == before


@pytest.mark.parametrize("protection", ["elder_mode", "arrival_delay"])
def test_strengthened_settings_committed_before_payment_lock_take_effect(scratch, monkeypatch, protection):
    conn, state = scratch
    arm(conn, state, amount=60000)
    before = balance(conn, state.customer.id)
    original_transaction = standing_orders._transaction
    changed = False

    @contextmanager
    def strengthen_before_lock(connection, savepoint):
        nonlocal changed
        if savepoint == "bank_standing_order_sp" and not changed:
            changed = True
            # A genuine second connection commits an authorized settings
            # change at the deterministic boundary before payment's write lock.
            path = connection.execute("PRAGMA database_list").fetchone()[2]
            other = connect(path)
            try:
                compliance.update_settings(other, state.customer.id,
                                           {protection: True if protection == "elder_mode" else "24h"}, now=DUE)
            finally:
                other.close()
            assert not connection.in_transaction
        with original_transaction(connection, savepoint):
            yield

    monkeypatch.setattr(standing_orders, "_transaction", strengthen_before_lock)
    runs = standing_orders.run_due_standing_orders(conn, state.customer.id, DUE)
    assert changed
    if protection == "elder_mode":
        assert runs[0]["status"] == "skipped", "Committed elder mode ignored: 60000 fen exceeds 50000 daily limit"
        assert runs[0]["reason"] == "elder_limit"
        assert balance(conn, state.customer.id) == before
    else:
        transfer = conn.execute("SELECT created_at, release_at FROM transfers WHERE customer_id=?", (state.customer.id,)).fetchone()
        assert datetime.fromisoformat(transfer["release_at"]) - datetime.fromisoformat(transfer["created_at"]) >= timedelta(hours=24), "Committed 24h arrival protection shortened to stale 2h"


def test_external_anchor_failure_blocks_later_batch_until_repaired(scratch, tmp_path):
    conn, state = scratch
    arm(conn, state)
    create_audit_schema(conn)
    anchor = tmp_path / "head.json"
    audit = AuditLog(conn, b"standing-security test", anchor_path=anchor)
    audit.anchor(SEEN)
    # A directory at the temporary file destination deterministically makes
    # the post-commit external-anchor write fail, without touching real files.
    anchor.with_name(anchor.name + ".tmp").mkdir()
    first = run_and_audit(conn, audit, state.customer.id, DUE)
    assert len(first) == 1 and first[0]["status"] == "paid"
    assert not audit.verify().ok
    before = balance(conn, state.customer.id)
    second = run_and_audit(conn, audit, state.customer.id, datetime(2026, 12, 5, 12))
    assert second == [], "Replay accepted the committed row although external anchor never persisted"
    assert balance(conn, state.customer.id) == before


def test_preexisting_wrong_key_audit_chain_blocks_autonomous_payment(scratch):
    conn, state = scratch
    arm(conn, state)
    create_audit_schema(conn)
    AuditLog(conn, b"old-key").append(customer_id=state.customer.id, session_id="test", plan_id="earlier", kind=AuditKind.CHECK, payload={}, now=SEEN)
    audit = AuditLog(conn, b"different-key")
    assert not audit.verify().ok
    before = balance(conn, state.customer.id)
    assert run_and_audit(conn, audit, state.customer.id, DUE) == [], "Unverified chain was extended and re-anchored under a different key"
    assert balance(conn, state.customer.id) == before


def test_same_order_plan_for_another_customer_does_not_suppress_audit(scratch):
    conn, state = scratch
    order = arm(conn, state)
    create_audit_schema(conn)
    audit = AuditLog(conn, b"standing-security test")
    audit.append(customer_id="different-customer", session_id="standing-orders", plan_id=f"so:{order}:2026-11", kind=AuditKind.RUN, payload={"tool": "standing_order", "status": "paid"}, now=DUE)
    runs = run_and_audit(conn, audit, state.customer.id, DUE)
    assert len(runs) == 1
    own = audit.entries_for(state.customer.id)
    assert len(own) == 1 and own[0].plan_id == f"so:{order}:2026-11"
    assert audit.verify().ok


def test_month_end_due_and_daily_limit_use_shanghai_date(scratch):
    conn, state = scratch
    arm(conn, state, amount=60000, day=31)
    # 2027-02-27 16:30 UTC is Shanghai's February 28; clamp day 31.
    instant = datetime(2027, 2, 27, 16, 30, tzinfo=timezone.utc)
    local = datetime(2027, 2, 28, 0, 20)
    Bank(conn, state.customer.id).transfer(state.contacts[0].id, 45000, "earlier local day", "local-prior", local)
    runs = standing_orders.run_due_standing_orders(conn, state.customer.id, instant)
    current = next(row for row in runs if row["occurrence"] == "2027-02")
    assert current["status"] == "skipped" and current["reason"] == "over_daily_limit"
    assert conn.execute("SELECT count(*) FROM transfers WHERE customer_id=?", (state.customer.id,)).fetchone()[0] == 1
