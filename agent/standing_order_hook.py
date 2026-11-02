"""Run due standing orders and audit every result, from any entry point."""

from __future__ import annotations

import sqlite3
import logging
import threading
from datetime import date, datetime

from bank.standing_orders import run_due_standing_orders
from security.audit import AuditKind, AuditLog

_LOCK = threading.Lock()
_LOG = logging.getLogger(__name__)
_AUDIT_ONCE = """
CREATE UNIQUE INDEX IF NOT EXISTS ix_standing_order_audit_once
ON security_audit(customer_id, plan_id)
WHERE kind='RUN' AND session_id='standing-orders'
"""


def _has_audit(audit: AuditLog, customer_id: str, plan_id: str) -> bool:
    return any(entry.kind == AuditKind.RUN and entry.session_id == "standing-orders"
               and entry.payload.get("tool") == "standing_order"
               for entry in audit.entries_for(customer_id, plan_id))


def _flush_audit(conn: sqlite3.Connection, audit: AuditLog, customer_id: str, now: date | datetime) -> bool:
    """Replay the durable execution journal, including runs from a prior crash."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='standing_order_runs'"
    ).fetchone()
    if exists is None:
        return True
    complete = True
    for run in conn.execute(
        "SELECT * FROM standing_order_runs WHERE customer_id=? ORDER BY at, scheduled_id, occurrence",
        (customer_id,),
    ).fetchall():
        plan_id = f"so:{run['scheduled_id']}:{run['occurrence']}"
        try:
            if _has_audit(audit, customer_id, plan_id):
                continue
            audit.append(
                customer_id=customer_id, session_id="standing-orders", plan_id=plan_id,
                kind=AuditKind.RUN,
                payload={"tool": "standing_order", "status": run["status"],
                         "reason": run["reason"], "amount_fen": run["amount_fen"],
                         "occurrence": run["occurrence"], "executed_at": run["at"]},
                now=now,
            )
        except sqlite3.IntegrityError:
            # Another process may have appended since our read. The partial
            # unique index makes the replay exactly once across connections.
            if not _has_audit(audit, customer_id, plan_id):
                complete = False
                _LOG.warning("Standing-order audit pending after integrity failure")
        except Exception:
            # Keep the durable run row pending and continue with other rows.
            complete = False
            _LOG.warning("Standing-order audit pending; will retry before another batch")
    return complete


def run_and_audit(
    conn: sqlite3.Connection,
    audit: AuditLog,
    customer_id: str,
    now: date | datetime,
) -> list[dict[str, object]]:
    """Run the unattended batch and append one RUN audit entry per run.

    Completed payments are never hidden when an audit append fails. Persisted
    run rows are replayed before later batches, without reissuing the payment.
    """
    with _LOCK:
        if conn.in_transaction:
            return []
        try:
            # An existing broken/wrong-key chain or failed external anchor
            # must never be silently re-anchored by an autonomous payment.
            if not audit.verify().ok:
                _LOG.warning("Standing-order batch blocked: audit integrity unavailable")
                return []
            # Existing duplicate history is never deleted to create this index:
            # a migration failure leaves execution closed for manual review.
            conn.execute(_AUDIT_ONCE)
            if not _flush_audit(conn, audit, customer_id, now):
                return []
            runs = run_due_standing_orders(conn, customer_id, now)
        except Exception:
            _LOG.warning("Standing-order batch unavailable; no new batch started")
            return []
        try:
            _flush_audit(conn, audit, customer_id, now)
        except Exception:
            _LOG.warning("Standing-order audit pending after batch")
        return runs
