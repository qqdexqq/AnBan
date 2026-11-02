"""Fault recovery, cancellation races and fail-closed unattended payments."""
from datetime import datetime
import threading

from agent.standing_order_hook import run_and_audit
from bank.api import Bank
from bank.standing_orders import run_due_standing_orders
from security import compliance
from security.audit import AuditLog, create_audit_schema
from tests.test_standing_orders import env, NOW, _schedule, _balance, _count

DUE = datetime(2026, 11, 5, 12)


def test_failed_audit_is_recovered_without_repeating_payment(env, monkeypatch):
    conn, state = env
    create_audit_schema(conn)
    audit = AuditLog(conn, b"standing-order-recovery-test")
    _schedule(conn, state.customer.id, state.contacts[0].id)
    run_and_audit(conn, audit, state.customer.id, NOW)
    original = audit.append
    def fail(**kwargs):
        raise OSError("simulated unavailable audit")
    monkeypatch.setattr(audit, "append", fail)
    paid = run_and_audit(conn, audit, state.customer.id, DUE)
    assert len(paid) == 1 and paid[0]["status"] == "paid"
    assert _count(conn, "transfers", state.customer.id) == 1
    monkeypatch.setattr(audit, "append", original)
    assert run_and_audit(conn, audit, state.customer.id, DUE) == []
    assert len(audit.entries_for(state.customer.id)) == 1
    assert _count(conn, "transfers", state.customer.id) == 1
    run_and_audit(conn, audit, state.customer.id, DUE)
    assert len(audit.entries_for(state.customer.id)) == 1
    assert audit.verify().ok


def test_cancel_committed_after_prefetch_blocks_payment(env, monkeypatch):
    conn, state = env
    order = _schedule(conn, state.customer.id, state.contacts[0].id)
    run_due_standing_orders(conn, state.customer.id, NOW)
    original = compliance.arrival_delay
    def cancel_before_transaction(*args):
        Bank(conn, state.customer.id).cancel_scheduled(order)
        return original(*args)
    monkeypatch.setattr(compliance, "arrival_delay", cancel_before_transaction)
    assert run_due_standing_orders(conn, state.customer.id, DUE) == []
    assert _count(conn, "transfers", state.customer.id) == 0


def test_fraud_list_is_rechecked_before_holding_money(env, monkeypatch):
    conn, state = env
    _schedule(conn, state.customer.id, state.contacts[0].id)
    run_due_standing_orders(conn, state.customer.id, NOW)
    monkeypatch.setattr(compliance, "fraud_accounts", lambda: {
        compliance.normalize_account(state.contacts[0].account_no): "test"
    })
    before = _balance(conn, state.customer.id)
    runs = run_due_standing_orders(conn, state.customer.id, DUE)
    assert len(runs) == 1 and runs[0]["reason"] == "fraud_list"
    assert _balance(conn, state.customer.id) == before
    assert _count(conn, "transfers", state.customer.id) == 0


def test_caller_uncommitted_work_is_never_committed_by_scheduler(env):
    conn, state = env
    _schedule(conn, state.customer.id, state.contacts[0].id)
    run_due_standing_orders(conn, state.customer.id, NOW)
    before = _balance(conn, state.customer.id)
    conn.execute("UPDATE accounts SET balance_fen=balance_fen+123 WHERE customer_id=? AND type='current'", (state.customer.id,))
    assert conn.in_transaction
    assert run_due_standing_orders(conn, state.customer.id, DUE) == []
    assert conn.in_transaction
    conn.rollback()
    assert _balance(conn, state.customer.id) == before
    assert _count(conn, "transfers", state.customer.id) == 0


def test_repair_crash_after_transfer_committed_without_redebit(env):
    conn, state = env
    _schedule(conn, state.customer.id, state.contacts[0].id)
    run_due_standing_orders(conn, state.customer.id, NOW)
    run_due_standing_orders(conn, state.customer.id, DUE)
    transfer_id = conn.execute("SELECT id FROM transfers WHERE customer_id=?", (state.customer.id,)).fetchone()[0]
    # Reproduce a previously persisted run whose final link update was lost.
    conn.execute("UPDATE standing_order_runs SET transfer_id=NULL WHERE customer_id=?", (state.customer.id,))
    conn.commit()
    before = _balance(conn, state.customer.id)
    assert run_due_standing_orders(conn, state.customer.id, DUE) == []
    row = conn.execute("SELECT transfer_id FROM standing_order_runs WHERE customer_id=?", (state.customer.id,)).fetchone()
    assert row[0] == transfer_id
    assert _balance(conn, state.customer.id) == before
    assert _count(conn, "transfers", state.customer.id) == 1


def test_separate_audit_coordinators_cannot_duplicate_a_run(tmp_path):
    from agent.standing_order_hook import _flush_audit
    from bank.db import connect, create_schema, load_state
    from bank.seed import generate
    from datetime import date
    path = str(tmp_path / "audit-race.sqlite3")
    conn = connect(path)
    state = generate("young_pro", 42, date(2026, 10, 17))
    create_schema(conn)
    load_state(conn, state)
    create_audit_schema(conn)
    audit = AuditLog(conn, b"race-test")
    _schedule(conn, state.customer.id, state.contacts[0].id)
    run_and_audit(conn, audit, state.customer.id, NOW)
    run_due_standing_orders(conn, state.customer.id, DUE)
    conn.close()
    barrier = threading.Barrier(2)
    results, errors = [], []
    def worker():
        worker_conn = connect(path)
        worker_audit = AuditLog(worker_conn, b"race-test")
        original = worker_audit.entries_for
        def stale_read(*args):
            entries = original(*args)
            if not entries:
                barrier.wait(timeout=5)
            return entries
        worker_audit.entries_for = stale_read
        try:
            # Bypass the process-local hook lock to simulate two app processes.
            results.append(_flush_audit(worker_conn, worker_audit, state.customer.id, DUE))
        except BaseException as exc:
            errors.append(exc)
        finally:
            worker_conn.close()
    workers = [threading.Thread(target=worker) for _ in range(2)]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join(timeout=10)
    assert not any(worker_thread.is_alive() for worker_thread in workers)
    assert errors == [] and results == [True, True]
    check_conn = connect(path)
    try:
        checked = AuditLog(check_conn, b"race-test")
        assert len(checked.entries_for(state.customer.id)) == 1
        assert checked.verify().ok
    finally:
        check_conn.close()


def test_skipped_audit_never_claims_a_payment(env, monkeypatch):
    conn, state = env
    create_audit_schema(conn)
    audit = AuditLog(conn, b"skip-status-test")
    order = _schedule(conn, state.customer.id, state.contacts[0].id)
    run_and_audit(conn, audit, state.customer.id, NOW)
    monkeypatch.setattr(compliance, "fraud_accounts", lambda: {
        compliance.normalize_account(state.contacts[0].account_no): "test"
    })
    run_and_audit(conn, audit, state.customer.id, DUE)
    for lang, word in (("en", "skipped"), ("zh", "跳过")):
        explanation = "\n".join(audit.explain(state.customer.id, f"so:{order}:2026-11", lang))
        assert word in explanation.lower()
        assert "Done:" not in explanation and "已执行" not in explanation


def test_duplicate_legacy_audit_is_preserved_and_blocks_new_payments(env):
    from security.audit import AuditKind
    conn, state = env
    create_audit_schema(conn)
    audit = AuditLog(conn, b"legacy-audit-test")
    _schedule(conn, state.customer.id, state.contacts[0].id)
    run_due_standing_orders(conn, state.customer.id, NOW)
    for _ in range(2):
        audit.append(customer_id=state.customer.id, session_id="standing-orders",
                     plan_id="so:legacy:2026-10", kind=AuditKind.RUN,
                     payload={"tool": "standing_order", "status": "missed"}, now=NOW)
    before = _balance(conn, state.customer.id)
    assert run_and_audit(conn, audit, state.customer.id, DUE) == []
    assert _balance(conn, state.customer.id) == before
    assert _count(conn, "transfers", state.customer.id) == 0
    assert len(audit.entries_for(state.customer.id)) == 2
    assert audit.verify().ok
