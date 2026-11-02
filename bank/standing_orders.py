"""Unattended execution of monthly standing orders.

A standing order is approved once when it is created.  Every unattended run
re-checks it from scratch: anything doubtful is skipped with a reason, never
retried and never escalated.  ``now`` is always injected by the caller.
"""

from __future__ import annotations

import calendar
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from typing import Iterator

from security import compliance
from security.cooling import cooling_for

from .api import Bank, BankError
from .errand_payments import _check_not_blocked, reserved_budget
from .redemption import _as_now

DAILY_LIMIT_FEN = 100_000

_SEEN_TABLE = """
CREATE TABLE IF NOT EXISTS standing_order_seen (
    customer_id TEXT NOT NULL,
    scheduled_id TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    PRIMARY KEY (customer_id, scheduled_id)
)
"""

_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS standing_order_runs (
    customer_id TEXT NOT NULL,
    scheduled_id TEXT NOT NULL,
    occurrence TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('paid', 'skipped', 'missed')),
    reason TEXT,
    transfer_id TEXT,
    amount_fen INTEGER,
    at TEXT NOT NULL,
    PRIMARY KEY (customer_id, scheduled_id, occurrence)
)
"""

_ACTIVE_STATUS = "scheduled"


@contextmanager
def _transaction(conn: sqlite3.Connection, savepoint: str) -> Iterator[None]:
    """Run a block in one transaction, mirroring the errand payment module."""
    began_transaction = not conn.in_transaction
    if began_transaction:
        conn.execute("BEGIN IMMEDIATE")
    sp = None if began_transaction else savepoint
    if sp is not None:
        conn.execute(f"SAVEPOINT {sp}")

    try:
        conn.execute(_SEEN_TABLE)
        conn.execute(_RUNS_TABLE)
        yield
    except BaseException:
        if sp is not None:
            try:
                conn.execute(f"ROLLBACK TO {sp}")
                conn.execute(f"RELEASE {sp}")
            except sqlite3.OperationalError:
                # Bank.transfer() opened its own connection context and may
                # already have closed the nested transaction.
                conn.rollback()
        elif began_transaction:
            conn.rollback()
        raise
    else:
        if sp is not None:
            try:
                conn.execute(f"RELEASE {sp}")
            except sqlite3.OperationalError:
                # Same as above: the bank call committed, so finish any work
                # that was written after it.
                conn.commit()
        elif began_transaction:
            conn.commit()


def _days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _due_date(year: int, month: int, day_of_month: int) -> date:
    return date(year, month, min(day_of_month, _days_in_month(year, month)))


def _next_month(year: int, month: int) -> tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def _next_due_date(day_of_month: int, today: date) -> date:
    due = _due_date(today.year, today.month, day_of_month)
    if due <= today:
        year, month = _next_month(today.year, today.month)
        due = _due_date(year, month, day_of_month)
    return due


def _occurrence(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _reserved_today(
    conn: sqlite3.Connection, customer_id: str, now_dt: datetime
) -> int:
    """Outstanding live exposure, including still-valid midnight crossings."""
    return reserved_budget(conn, customer_id, now_dt)


def _run_exists(
    conn: sqlite3.Connection, customer_id: str, scheduled_id: str, occurrence: str
) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM standing_order_runs
        WHERE customer_id = ? AND scheduled_id = ? AND occurrence = ?
        """,
        (customer_id, scheduled_id, occurrence),
    ).fetchone()
    return row is not None


def _run_row(
    customer_id: str,
    scheduled_id: str,
    occurrence: str,
    status: str,
    reason: str | None,
    amount_fen: int,
    at: str,
    transfer_id: str | None = None,
) -> dict[str, object]:
    return {
        "scheduled_id": scheduled_id,
        "occurrence": occurrence,
        "status": status,
        "reason": reason,
        "amount_fen": amount_fen,
        "transfer_id": transfer_id,
    }


def _skip_reason(
    conn: sqlite3.Connection,
    customer_id: str,
    order: sqlite3.Row,
    amount_fen: int,
    now_dt: datetime,
    *,
    elder: bool,
    elder_daily_limit_fen: int,
) -> str | None:
    """The first failing check for this occurrence, or None when it may pay."""
    try:
        _check_not_blocked(conn, customer_id, now_dt)
    except BankError as exc:
        if exc.code == "R_LOCKED":
            return "payments_locked"
        if exc.code == "R_SCAM":
            return "scam_freeze"
        raise

    contact = conn.execute(
        "SELECT * FROM contacts WHERE customer_id = ? AND id = ?",
        (customer_id, order["to_contact_id"]),
    ).fetchone()
    if contact is None:
        return "payee_missing"
    if "verified" in contact.keys() and not bool(contact["verified"]):
        return "payee_unverified"

    # The payee may have been listed since the order was authorised. Refuse
    # before debiting, rather than relying only on the later release check.
    hold = compliance.transfer_release_hold(conn, customer_id, contact["id"], now_dt)
    if hold is not None:
        return {
            "R_FRAUD_LIST": "fraud_list", "R_SCAM": "scam_freeze",
            "R_LOCKED": "payments_locked",
        }.get(hold[0], "release_check_unavailable")

    reserved = _reserved_today(conn, customer_id, now_dt)
    outgoing = Bank(conn, customer_id).daily_outgoing_total(now_dt) + reserved
    if outgoing + amount_fen > DAILY_LIMIT_FEN:
        return "over_daily_limit"

    if elder and outgoing + amount_fen > elder_daily_limit_fen:
        return "elder_limit"

    account = conn.execute(
        """
        SELECT balance_fen FROM accounts
        WHERE customer_id = ? AND type = 'current'
        ORDER BY id LIMIT 1
        """,
        (customer_id,),
    ).fetchone()
    if account is None or int(account["balance_fen"]) < amount_fen:
        return "insufficient_funds"
    return None


def _insert_run(
    conn: sqlite3.Connection,
    customer_id: str,
    scheduled_id: str,
    occurrence: str,
    status: str,
    reason: str | None,
    amount_fen: int,
    at: str,
    transfer_id: str | None = None,
) -> bool:
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO standing_order_runs
            (customer_id, scheduled_id, occurrence, status, reason,
             transfer_id, amount_fen, at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            customer_id,
            scheduled_id,
            occurrence,
            status,
            reason,
            transfer_id,
            amount_fen,
            at,
        ),
    )
    return cursor.rowcount == 1


def _run_one(
    conn: sqlite3.Connection,
    customer_id: str,
    order: sqlite3.Row,
    now_dt: datetime,
) -> list[dict[str, object]]:
    scheduled_id = str(order["id"])
    amount_fen = int(order["amount_fen"])
    day_of_month = int(order["day_of_month"])
    at = now_dt.isoformat()
    created: list[dict[str, object]] = []

    # Resolve policy/compliance values before opening the payment transaction:
    # compliance.get_settings() may create its schema, which commits.
    from security.policy import PolicyConfig, _daily_limit, _is_elder

    policy_config = PolicyConfig()
    policy_bank = Bank(conn, customer_id)
    compliance.arrival_delay(conn, customer_id)  # Initialise schema before the payment lock.

    with _transaction(conn, "bank_standing_order_sp"):
        # Re-read effective protections under the payment lock, so a concurrent
        # strengthening committed before it cannot be ignored. These are pure
        # reads and cannot close the payment transaction through schema DDL.
        elder = _is_elder(policy_bank, policy_config)
        elder_daily_limit_fen = _daily_limit(policy_bank, policy_config)
        arrival_delay = compliance.ARRIVAL_DELAYS[compliance.read_settings(conn, customer_id).arrival_delay]
        # The batch SELECT preceded this write lock. A committed cancellation
        # or edit must win over that stale snapshot.
        current = conn.execute(
            "SELECT * FROM scheduled_transfers WHERE customer_id=? AND id=?",
            (customer_id, scheduled_id),
        ).fetchone()
        if current is None or current["status"] != _ACTIVE_STATUS:
            return created
        order = current
        amount_fen = int(order["amount_fen"])
        day_of_month = int(order["day_of_month"])
        seen = conn.execute(
            """
            SELECT first_seen FROM standing_order_seen
            WHERE customer_id = ? AND scheduled_id = ?
            """,
            (customer_id, scheduled_id),
        ).fetchone()
        if seen is None:
            # Only start counting after the customer first saw this order:
            # never surprise them with a payment on creation day.
            conn.execute(
                """
                INSERT INTO standing_order_seen
                    (customer_id, scheduled_id, first_seen)
                VALUES (?, ?, ?)
                """,
                (customer_id, scheduled_id, at),
            )
            return created

        first_seen = datetime.fromisoformat(str(seen["first_seen"]))
        seen_date = first_seen.date()
        today = now_dt.date()

        # Earlier months are never backfilled: record them as missed only.
        year, month = seen_date.year, seen_date.month
        while (year, month) < (now_dt.year, now_dt.month):
            due = _due_date(year, month, day_of_month)
            occurrence = _occurrence(year, month)
            if due > seen_date and due <= today:
                if _insert_run(
                    conn,
                    customer_id,
                    scheduled_id,
                    occurrence,
                    "missed",
                    "not_run_on_time",
                    amount_fen,
                    at,
                ):
                    created.append(
                        _run_row(
                            customer_id,
                            scheduled_id,
                            occurrence,
                            "missed",
                            "not_run_on_time",
                            amount_fen,
                            at,
                        )
                    )
            year, month = _next_month(year, month)

        due = _due_date(now_dt.year, now_dt.month, day_of_month)
        occurrence = _occurrence(now_dt.year, now_dt.month)
        if due <= today and due > seen_date and not _run_exists(
            conn, customer_id, scheduled_id, occurrence
        ):
            reason = _skip_reason(
                conn,
                customer_id,
                order,
                amount_fen,
                now_dt,
                elder=elder,
                elder_daily_limit_fen=elder_daily_limit_fen,
            )
            if reason is not None:
                if _insert_run(
                    conn,
                    customer_id,
                    scheduled_id,
                    occurrence,
                    "skipped",
                    reason,
                    amount_fen,
                    at,
                ):
                    created.append(
                        _run_row(
                            customer_id,
                            scheduled_id,
                            occurrence,
                            "skipped",
                            reason,
                            amount_fen,
                            at,
                        )
                    )
            else:
                # The run row's primary key is what stops a concurrent second
                # run from paying the same order twice in the same month.
                if _insert_run(
                    conn,
                    customer_id,
                    scheduled_id,
                    occurrence,
                    "paid",
                    None,
                    amount_fen,
                    at,
                ):
                    transfer = Bank(conn, customer_id).transfer(
                        str(order["to_contact_id"]),
                        amount_fen,
                        str(order["note"] or ""),
                        f"standing_order:{scheduled_id}:{occurrence}",
                        now_dt,
                        cooling=cooling_for(
                            "yellow",
                            elder=elder,
                            arrival_delay=arrival_delay,
                        ),
                    )
                    conn.execute(
                        """
                        UPDATE standing_order_runs SET transfer_id = ?
                        WHERE customer_id = ? AND scheduled_id = ?
                          AND occurrence = ?
                        """,
                        (transfer.id, customer_id, scheduled_id, occurrence),
                    )
                    created.append(
                        _run_row(
                            customer_id,
                            scheduled_id,
                            occurrence,
                            "paid",
                            None,
                            amount_fen,
                            at,
                            transfer.id,
                        )
                    )
    return created


def run_due_standing_orders(
    conn: sqlite3.Connection, customer_id: str, now: datetime
) -> list[dict[str, object]]:
    """Pay every due standing order once, skipping anything doubtful."""
    # Settings/schema setup and the bank's transaction context may commit.
    # Never run this autonomous hook inside a caller's unfinished transaction.
    if conn.in_transaction:
        return []
    now_dt = _as_now(now)
    with _transaction(conn, "bank_standing_reconcile_sp"):
        # A crash in the previous implementation could persist the debit and
        # paid marker before saving its transfer ID. Repair from the existing
        # idempotent ledger entry; never create a payment during reconciliation.
        incomplete = conn.execute(
            "SELECT scheduled_id, occurrence, amount_fen FROM standing_order_runs "
            "WHERE customer_id=? AND status='paid' AND transfer_id IS NULL",
            (customer_id,),
        ).fetchall()
        for run in incomplete:
            transfer = conn.execute(
                "SELECT id, amount_fen FROM transfers WHERE customer_id=? AND idempotency_key=?",
                (customer_id, f"standing_order:{run['scheduled_id']}:{run['occurrence']}"),
            ).fetchone()
            if transfer is not None and transfer["amount_fen"] == run["amount_fen"]:
                conn.execute(
                    "UPDATE standing_order_runs SET transfer_id=? WHERE customer_id=? "
                    "AND scheduled_id=? AND occurrence=? AND transfer_id IS NULL",
                    (transfer["id"], customer_id, run["scheduled_id"], run["occurrence"]),
                )
    orders = conn.execute(
        """
        SELECT * FROM scheduled_transfers
        WHERE customer_id = ? AND status = ?
        ORDER BY id
        """,
        (customer_id, _ACTIVE_STATUS),
    ).fetchall()

    created: list[dict[str, object]] = []
    for order in orders:
        try:
            created.extend(_run_one(conn, customer_id, order, now_dt))
        except Exception:
            # One bad order must never stop the rest of the batch.  Record the
            # failure when the run tables are usable, then carry on.
            occurrence = _occurrence(now_dt.year, now_dt.month)
            at = now_dt.isoformat()
            try:
                with conn:
                    conn.execute(_SEEN_TABLE)
                    conn.execute(_RUNS_TABLE)
                    inserted = _insert_run(
                        conn,
                        customer_id,
                        str(order["id"]),
                        occurrence,
                        "skipped",
                        "error",
                        int(order["amount_fen"]),
                        at,
                    )
                if inserted:
                    created.append(
                        _run_row(
                            customer_id,
                            str(order["id"]),
                            occurrence,
                            "skipped",
                            "error",
                            int(order["amount_fen"]),
                            at,
                        )
                    )
            except Exception:
                continue
    return created


def standing_order_overview(
    conn: sqlite3.Connection, customer_id: str, now: datetime
) -> list[dict[str, object]]:
    """Read-only view of every standing order and its latest run."""
    now_dt = _as_now(now)
    has_runs = _table_exists(conn, "standing_order_runs")

    rows = conn.execute(
        """
        SELECT * FROM scheduled_transfers
        WHERE customer_id = ?
        ORDER BY id
        """,
        (customer_id,),
    ).fetchall()

    overview: list[dict[str, object]] = []
    for row in rows:
        scheduled_id = str(row["id"])
        status = str(row["status"])
        day_of_month = int(row["day_of_month"])
        next_due = None
        if status != "cancelled":
            next_due = _next_due_date(day_of_month, now_dt.date()).isoformat()

        last = conn.execute(
            """
            SELECT occurrence, status, reason, amount_fen
            FROM standing_order_runs
            WHERE customer_id = ? AND scheduled_id = ?
            ORDER BY occurrence DESC
            LIMIT 1
            """,
            (customer_id, scheduled_id),
        ).fetchone() if has_runs else None
        last_run = None
        if last is not None:
            last_run = {
                "occurrence": last["occurrence"],
                "status": last["status"],
                "reason": last["reason"],
                "amount_fen": (
                    int(last["amount_fen"]) if last["amount_fen"] is not None else None
                ),
            }

        overview.append(
            {
                "scheduled_id": scheduled_id,
                "contact_id": str(row["to_contact_id"]),
                "amount_fen": int(row["amount_fen"]),
                "day_of_month": day_of_month,
                "status": status,
                "note": str(row["note"] or ""),
                "next_due": next_due,
                "last_run": last_run,
            }
        )
    return overview
