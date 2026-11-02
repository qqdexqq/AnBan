"""In-process mock bank API backed by SQLite.

Every write runs in one SQLite transaction. Money is always integer fen.
All customer-owned rows are scoped by ``customer_id``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone

from security import compliance

from .merchants import price_of
from .models import (
    Card,
    Contact,
    Holding,
    MerchantOrder,
    PaymentRequest,
    Pot,
    Product,
    ScheduledTransfer,
    Subscription,
    Transaction,
    Transfer,
)

ERROR_CODES = {
    "not_found",
    "insufficient_funds",
    "invalid_state",
    "invalid_amount",
    "window_closed",
    "protected_hold",
    # ``transfer_release_hold`` reasons are surfaced unchanged so callers can
    # tell a scam freeze from a lockout or an unverifiable fraud list.
    "R_SCAM",
    "R_LOCKED",
    "R_FRAUD_LIST",
    "R_RELEASE_CHECK_UNAVAILABLE",
    # One-time-card errand gateway error codes.
    "ticket_used",
    "ticket_closed",
    "ticket_expired",
    "wrong_card",
    "card_inactive",
    "over_limit",
}

IdFactory = Callable[[str], str]


class BankError(Exception):
    """Domain error carrying a stable machine-readable ``code``."""

    def __init__(self, code: str, message: str | None = None) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown bank error code: {code!r}")
        self.code = code
        super().__init__(message or code)


def _next_id(conn: sqlite3.Connection, prefix: str) -> str:
    row = conn.execute("SELECT value FROM meta WHERE key = 'id_seq'").fetchone()
    seq = int(row["value"]) + 1 if row is not None else 1
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('id_seq', ?)", (str(seq),)
    )
    return f"{prefix}_{seq:06d}"


def _last4(card_id: str) -> str:
    digits = "".join(character for character in card_id if character.isdigit())
    return (digits or "0000").rjust(4, "0")[-4:]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ¥1 billion: far above any retail operation, and well inside SQLite INTEGER.
MAX_AMOUNT_FEN = 100_000_000_000

# China has no DST, so a fixed +08:00 offset is exact (and needs no tzdata on Windows).
LOCAL_TZ = timezone(timedelta(hours=8))


def _require_amount(amount_fen: int) -> None:
    if not _is_int(amount_fen) or amount_fen <= 0 or amount_fen > MAX_AMOUNT_FEN:
        raise BankError("invalid_amount", "amount must be a positive integer in fen")


def _require_non_negative(amount_fen: int) -> None:
    if not _is_int(amount_fen) or amount_fen < 0 or amount_fen > MAX_AMOUNT_FEN:
        raise BankError("invalid_amount", "amount must be a non-negative integer in fen")


def _require_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BankError("invalid_state", f"{field} must be a non-empty string")
    return value


def _to_local_naive(value: datetime) -> datetime:
    """Bank times are naive Asia/Shanghai local time; convert tz-aware input once here.

    Stored timestamps are compared by their local calendar date, so a 00:30+08:00
    transfer must not land on the previous (UTC) day.
    """
    if value.tzinfo is None:
        return value
    return value.astimezone(LOCAL_TZ).replace(tzinfo=None)


def _as_datetime(value: date | datetime, *, end_of_day: bool = False) -> datetime:
    if isinstance(value, datetime):
        return _to_local_naive(value)
    if isinstance(value, date):
        return datetime.combine(value, time.max if end_of_day else time.min)
    if isinstance(value, str):
        try:
            return _to_local_naive(datetime.fromisoformat(value))
        except ValueError as exc:
            raise BankError("invalid_state", "invalid datetime") from exc
    raise BankError("invalid_state", "invalid datetime")


def _as_now(value: date | datetime) -> datetime:
    return _as_datetime(value)


def _as_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return _to_local_naive(value).date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise BankError("invalid_state", "invalid date") from exc
    raise BankError("invalid_state", "invalid date")


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _contact(row: sqlite3.Row) -> Contact:
    return Contact(
        id=row["id"],
        name=row["name"],
        relation=row["relation"],
        note=row["note"],
        account_no=row["account_no"],
        phone=row["phone"],
        bank_name=row["bank_name"],
        birthday=_parse_date(row["birthday"]),
        verified=bool(row["verified"]),
    )


def _transaction(row: sqlite3.Row) -> Transaction:
    return Transaction(
        id=row["id"],
        account_id=row["account_id"],
        ts=datetime.fromisoformat(row["ts"]),
        amount_fen=row["amount_fen"],
        counterparty=row["counterparty"],
        category=row["category"],
        memo=row["memo"],
        channel=row["channel"],
        country=row["country"],
        card_present=bool(row["card_present"]),
    )


def _transfer(row: sqlite3.Row) -> Transfer:
    return Transfer(
        id=row["id"],
        to_contact_id=row["to_contact_id"],
        amount_fen=row["amount_fen"],
        note=row["note"],
        idempotency_key=row["idempotency_key"],
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        release_at=datetime.fromisoformat(row["release_at"]),
        account_id=row["account_id"],
    )


def _scheduled(row: sqlite3.Row) -> ScheduledTransfer:
    return ScheduledTransfer(
        id=row["id"],
        to_contact_id=row["to_contact_id"],
        amount_fen=row["amount_fen"],
        day_of_month=row["day_of_month"],
        note=row["note"],
        status=row["status"],
    )


def _payment_request(row: sqlite3.Row) -> PaymentRequest:
    return PaymentRequest(
        id=row["id"],
        from_contact_id=row["from_contact_id"],
        amount_fen=row["amount_fen"],
        note=row["note"],
        status=row["status"],
    )


def _order(row: sqlite3.Row) -> MerchantOrder:
    keys = row.keys()
    return MerchantOrder(
        id=row["id"],
        merchant=row["merchant"],
        item=row["item"],
        amount_fen=row["amount_fen"],
        deliver_on=date.fromisoformat(row["deliver_on"]),
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        funding_pot_id=row["funding_pot_id"] if "funding_pot_id" in keys else None,
    )


def _subscription(row: sqlite3.Row) -> Subscription:
    return Subscription(
        id=row["id"],
        merchant=row["merchant"],
        amount_fen=row["amount_fen"],
        day_of_month=row["day_of_month"],
        status=row["status"],
        last_used_at=date.fromisoformat(row["last_used_at"]),
    )


def _card(row: sqlite3.Row) -> Card:
    return Card(
        id=row["id"],
        kind=row["kind"],
        last4=row["last4"],
        status=row["status"],
        online_limit_fen=row["online_limit_fen"],
        account_id=row["account_id"],
    )


def _product(row: sqlite3.Row) -> Product:
    return Product(
        id=row["id"],
        name=row["name"],
        risk_level=row["risk_level"],
        annual_rate_bp=row["annual_rate_bp"],
        min_fen=row["min_fen"],
    )


def _holding(row: sqlite3.Row) -> Holding:
    return Holding(
        product_id=row["product_id"],
        principal_fen=row["principal_fen"],
        value_fen=row["value_fen"],
    )


def _pot(row: sqlite3.Row) -> Pot:
    return Pot(id=row["id"], name=row["name"], balance_fen=row["balance_fen"])


def _ensure_merchant_order_funding_column(conn: sqlite3.Connection) -> None:
    """Add the nullable funding source column to older databases in place.

    The DDL never commits a transaction the caller already had open.  When
    there is no caller transaction, the DDL's own transaction is committed so
    the schema change survives (nothing else is pending in that case).
    """

    columns = {row[1] for row in conn.execute("PRAGMA table_info(merchant_orders)")}
    if not columns or "funding_pot_id" in columns:
        return
    had_transaction = conn.in_transaction
    conn.execute("ALTER TABLE merchant_orders ADD COLUMN funding_pot_id TEXT")
    if not had_transaction:
        conn.commit()


class Bank:
    """Customer-scoped banking operations over a loaded SQLite state."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        customer_id: str,
        id_factory: IdFactory | None = None,
    ) -> None:
        self.conn = conn
        self.customer_id = customer_id
        self._id_factory = id_factory
        row = conn.execute(
            "SELECT id FROM customers WHERE id = ?", (customer_id,)
        ).fetchone()
        if row is None:
            raise BankError("not_found", "customer not found")
        _ensure_merchant_order_funding_column(conn)

    def _make_id(self, prefix: str) -> str:
        if self._id_factory is None:
            return _next_id(self.conn, prefix)
        try:
            return str(self._id_factory(prefix))
        except TypeError:
            return str(self._id_factory())

    # ------------------------------------------------------------------ reads

    def get_balance(self) -> int:
        return self._current_account()["balance_fen"]

    def list_transactions(
        self,
        start: date | datetime | None = None,
        end: date | datetime | None = None,
        category: str | None = None,
        counterparty: str | None = None,
        limit: int | None = 50,
    ) -> list[Transaction]:
        clauses = ["customer_id = ?"]
        params: list[object] = [self.customer_id]
        if start is not None:
            clauses.append("ts >= ?")
            params.append(_as_datetime(start).isoformat())
        if end is not None:
            clauses.append("ts <= ?")
            params.append(_as_datetime(end, end_of_day=True).isoformat())
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if counterparty is not None:
            clauses.append("counterparty = ?")
            params.append(counterparty)

        sql = "SELECT * FROM transactions WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts DESC, id DESC"
        if limit is not None:
            if not _is_int(limit) or limit < 0:
                raise BankError("invalid_state", "limit must be a non-negative integer")
            if limit == 0:
                return []
            sql += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [_transaction(row) for row in rows]

    def find_contact(self, query: str) -> list[Contact]:
        pattern = f"%{query or ''}%"
        rows = self.conn.execute(
            """
            SELECT * FROM contacts
            WHERE customer_id = ?
              AND (
                  LOWER(name) LIKE LOWER(?)
                  OR LOWER(note) LIKE LOWER(?)
                  OR LOWER(relation) LIKE LOWER(?)
              )
            ORDER BY id
            """,
            (self.customer_id, pattern, pattern, pattern),
        ).fetchall()
        return [_contact(row) for row in rows]

    def get_contact_date(self, contact_id: str) -> date | None:
        row = self.conn.execute(
            "SELECT birthday FROM contacts WHERE customer_id = ? AND id = ?",
            (self.customer_id, contact_id),
        ).fetchone()
        if row is None:
            raise BankError("not_found", "contact not found")
        return _parse_date(row["birthday"])

    def can_cover(self, amount_fen: int) -> bool:
        if not _is_int(amount_fen) or amount_fen < 0:
            return False
        return amount_fen <= self.get_balance()

    def list_subscriptions(self) -> list[Subscription]:
        rows = self.conn.execute(
            "SELECT * FROM subscriptions WHERE customer_id = ? ORDER BY id",
            (self.customer_id,),
        ).fetchall()
        return [_subscription(row) for row in rows]

    def list_cards(self) -> list[Card]:
        rows = self.conn.execute(
            "SELECT * FROM cards WHERE customer_id = ? ORDER BY id",
            (self.customer_id,),
        ).fetchall()
        return [_card(row) for row in rows]

    def list_products(self) -> list[Product]:
        rows = self.conn.execute("SELECT * FROM products ORDER BY id").fetchall()
        return [_product(row) for row in rows]

    def get_portfolio(self) -> list[Holding]:
        rows = self.conn.execute(
            "SELECT * FROM holdings WHERE customer_id = ? ORDER BY product_id",
            (self.customer_id,),
        ).fetchall()
        return [_holding(row) for row in rows]

    def spending_report(
        self, start: date | datetime, end: date | datetime,
        category: str | None = None, counterparty: str | None = None,
    ) -> dict[str, object]:
        rows = self.list_transactions(start, end, category, counterparty, limit=None)
        by_category: dict[str, int] = {}
        for row in rows:
            if row.amount_fen >= 0 or (row.category == "transfer" and row.counterparty == "我的储蓄账户"):
                continue
            by_category[row.category] = by_category.get(row.category, 0) - row.amount_fen
        return {"by_category": by_category, "total": sum(by_category.values())}

    def income_report(
        self, start: date | datetime, end: date | datetime,
        category: str | None = None, counterparty: str | None = None,
    ) -> int:
        """Incoming ledger credits, excluding transfers between owned accounts."""
        return sum(row.amount_fen for row in self.list_transactions(
            start, end, category, counterparty, limit=None
        ) if row.amount_fen > 0 and row.counterparty != "我的储蓄账户")

    def detect_anomalies(
        self, now: date | datetime, days: int = 30
    ) -> list[dict[str, object]]:
        now_dt = _as_now(now)
        cutoff = (now_dt - timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            """
            SELECT * FROM transactions
            WHERE customer_id = ? AND ts >= ? AND ts <= ?
            ORDER BY ts, id
            """,
            (self.customer_id, cutoff, now_dt.isoformat()),
        ).fetchall()
        txns = [_transaction(row) for row in rows]
        anomalies: list[dict[str, object]] = []
        used: set[str] = set()

        for index, first in enumerate(txns):
            if first.id in used or first.amount_fen >= 0:
                continue
            for second in txns[index + 1 :]:
                if second.id in used or second.amount_fen != first.amount_fen:
                    continue
                if second.counterparty != first.counterparty:
                    continue
                if first.ts.date() != second.ts.date():
                    continue
                if abs((first.ts - second.ts).total_seconds()) > 5 * 60:
                    continue
                anomalies.append(
                    {"kind": "duplicate", "txn_ids": [first.id, second.id]}
                )
                used.update((first.id, second.id))
                break

        for txn in txns:
            if txn.id in used:
                continue
            if txn.country != "CN" and txn.card_present is False:
                anomalies.append({"kind": "foreign_cnp", "txn_ids": [txn.id]})

        return anomalies

    def daily_outgoing_total(self, day: date | datetime) -> int:
        day_str = _as_date(day).isoformat()
        row = self.conn.execute(
            """
            SELECT
                (
                    SELECT COALESCE(SUM(amount_fen), 0)
                    FROM transfers
                    WHERE customer_id = ?
                      AND substr(created_at, 1, 10) = ?
                      AND status IN ('pending', 'released')
                )
                +
                (
                    SELECT COALESCE(SUM(amount_fen), 0)
                    FROM merchant_orders
                    WHERE customer_id = ?
                      AND substr(created_at, 1, 10) = ?
                      AND status != 'cancelled'
                )
                +
                (
                    -- product purchases count towards the daily limit too
                    SELECT COALESCE(SUM(-amount_fen), 0)
                    FROM transactions
                    WHERE customer_id = ?
                      AND substr(ts, 1, 10) = ?
                      AND (category = 'investment' OR (category = 'utilities' AND channel = 'bill') OR (category = 'shopping' AND channel = 'errand'))
                      AND amount_fen < 0
                ) AS total
            """,
            (
                self.customer_id,
                day_str,
                self.customer_id,
                day_str,
                self.customer_id,
                day_str,
            ),
        ).fetchone()
        return int(row["total"])

    def scheduled_total_for_day(self, day_of_month: int) -> int:
        """Sum of this customer's active scheduled transfers on ``day_of_month``."""
        row = self.conn.execute(
            """
            SELECT COALESCE(SUM(amount_fen), 0) AS total
            FROM scheduled_transfers
            WHERE customer_id = ? AND day_of_month = ? AND status != 'cancelled'
            """,
            (self.customer_id, int(day_of_month)),
        ).fetchone()
        return int(row["total"])

    # --------------------------------------------------------------- transfers

    def transfer(
        self,
        to_contact_id: str,
        amount_fen: int,
        note: str,
        idempotency_key: str,
        now: date | datetime,
        cooling: timedelta = timedelta(hours=2),
    ) -> Transfer:
        _require_amount(amount_fen)
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise BankError("invalid_state", "idempotency_key is required")
        if not isinstance(cooling, timedelta):
            raise BankError("invalid_state", "cooling must be a timedelta")
        now_dt = _as_now(now)
        note = note or ""

        with self.conn:
            existing = self.conn.execute(
                """
                SELECT * FROM transfers
                WHERE customer_id = ? AND idempotency_key = ?
                """,
                (self.customer_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                stored_cooling = datetime.fromisoformat(
                    existing["release_at"]
                ) - datetime.fromisoformat(existing["created_at"])
                if (
                    existing["to_contact_id"] != to_contact_id
                    or existing["amount_fen"] != amount_fen
                    or existing["note"] != note
                    or stored_cooling != cooling
                ):
                    raise BankError(
                        "invalid_state",
                        "idempotency_key already used with different arguments",
                    )
                return _transfer(existing)

            self._contact_row(to_contact_id)
            account = self._current_account()
            if account["balance_fen"] < amount_fen:
                raise BankError("insufficient_funds")

            release_at = now_dt + cooling
            transfer_id = self._make_id("tr")
            self.conn.execute(
                """
                UPDATE accounts SET balance_fen = balance_fen - ?
                WHERE customer_id = ? AND id = ?
                """,
                (amount_fen, self.customer_id, account["id"]),
            )
            self.conn.execute(
                """
                INSERT INTO transfers
                    (customer_id, id, to_contact_id, amount_fen, note,
                     idempotency_key, status, created_at, release_at, account_id)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (
                    self.customer_id,
                    transfer_id,
                    to_contact_id,
                    amount_fen,
                    note,
                    idempotency_key,
                    now_dt.isoformat(),
                    release_at.isoformat(),
                    account["id"],
                ),
            )
            return Transfer(
                id=transfer_id,
                to_contact_id=to_contact_id,
                amount_fen=amount_fen,
                note=note,
                idempotency_key=idempotency_key,
                status="pending",
                created_at=now_dt,
                release_at=release_at,
                account_id=account["id"],
            )

    def cancel_transfer(self, transfer_id: str, now: date | datetime) -> Transfer:
        now_dt = _as_now(now)
        with self.conn:
            if not self.conn.in_transaction:
                self.conn.execute("BEGIN IMMEDIATE")
            row = self.conn.execute(
                "SELECT * FROM transfers WHERE customer_id = ? AND id = ?",
                (self.customer_id, transfer_id),
            ).fetchone()
            if row is None:
                raise BankError("not_found", "transfer not found")
            if row["status"] != "pending":
                raise BankError("window_closed")
            if not now_dt < datetime.fromisoformat(row["release_at"]):
                # Past the normal window, a still-pending transfer may only be
                # refunded while a protective hold keeps it from releasing.
                hold = compliance.transfer_release_hold(
                    self.conn, self.customer_id, row["to_contact_id"], now_dt
                )
                if hold is None:
                    raise BankError("window_closed")
            self.conn.execute(
                """
                UPDATE accounts SET balance_fen = balance_fen + ?
                WHERE customer_id = ? AND id = ?
                """,
                (row["amount_fen"], self.customer_id, row["account_id"]),
            )
            self.conn.execute(
                "UPDATE transfers SET status = 'cancelled' WHERE customer_id = ? AND id = ?",
                (self.customer_id, transfer_id),
            )
            updated = dict(row)
            updated["status"] = "cancelled"
            return _transfer(updated)

    def release_due(self, now: date | datetime) -> list[Transfer]:
        now_dt = _as_now(now)
        released: list[Transfer] = []
        began_transaction = not self.conn.in_transaction
        if began_transaction:
            self.conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self.conn.execute(
                """
                SELECT t.*, c.name AS contact_name
                FROM transfers t
                JOIN contacts c
                  ON c.customer_id = t.customer_id AND c.id = t.to_contact_id
                WHERE t.customer_id = ?
                  AND t.status = 'pending'
                  AND t.release_at <= ?
                ORDER BY t.release_at, t.id
                """,
                (self.customer_id, now_dt.isoformat()),
            ).fetchall()
            for row in rows:
                if (
                    compliance.transfer_release_hold(
                        self.conn, self.customer_id, row["to_contact_id"], now_dt
                    )
                    is not None
                ):
                    # Keep the transfer pending, its clock and the reserved money.
                    continue
                self.conn.execute(
                    """
                    UPDATE transfers SET status = 'released'
                    WHERE customer_id = ? AND id = ?
                    """,
                    (self.customer_id, row["id"]),
                )
                self.conn.execute(
                    """
                    INSERT INTO transactions
                        (customer_id, id, account_id, ts, amount_fen, counterparty,
                         category, memo, channel, country, card_present)
                    VALUES (?, ?, ?, ?, ?, ?, 'transfer', ?, 'transfer', 'CN', 0)
                    """,
                    (
                        self.customer_id,
                        self._make_id("txn"),
                        row["account_id"],
                        now_dt.isoformat(),
                        -row["amount_fen"],
                        row["contact_name"],
                        row["note"],
                    ),
                )
                updated = dict(row)
                updated["status"] = "released"
                released.append(_transfer(updated))
        except BaseException:
            if began_transaction:
                self.conn.rollback()
            raise
        if began_transaction:
            self.conn.commit()
        return released

    def release_early(
        self,
        transfer_id: str,
        now: date | datetime,
        *,
        expected_contact_id: str,
        expected_amount_fen: int,
    ) -> Transfer:
        """Release one pending transfer before its normal clock.

        The money was reserved when the transfer was created, so this only
        flips the row and writes the ledger entry.  The caller must name the
        contact and amount it approved: a changed target is not released.
        """

        now_dt = _as_now(now)
        began_transaction = not self.conn.in_transaction
        if began_transaction:
            self.conn.execute("BEGIN IMMEDIATE")
        result: Transfer | None = None
        try:
            row = self.conn.execute(
                """
                SELECT t.*, c.name AS contact_name
                FROM transfers t
                JOIN contacts c
                  ON c.customer_id = t.customer_id AND c.id = t.to_contact_id
                WHERE t.customer_id = ? AND t.id = ?
                """,
                (self.customer_id, transfer_id),
            ).fetchone()
            if row is None:
                raise BankError("not_found", "transfer not found")
            if row["status"] != "pending":
                raise BankError("window_closed")
            if datetime.fromisoformat(row["release_at"]) - datetime.fromisoformat(row["created_at"]) > timedelta(hours=2):
                raise BankError("protected_hold", "a protected longer hold cannot be shortened")
            if (
                row["to_contact_id"] != expected_contact_id
                or not _is_int(expected_amount_fen)
                or row["amount_fen"] != expected_amount_fen
            ):
                raise BankError(
                    "invalid_state", "transfer does not match the approved release"
                )

            hold = compliance.transfer_release_hold(
                self.conn, self.customer_id, row["to_contact_id"], now_dt
            )
            if hold is not None:
                raise BankError(str(hold[0]), "transfer release is held")

            cursor = self.conn.execute(
                """
                UPDATE transfers SET status = 'released'
                WHERE customer_id = ? AND id = ? AND status = 'pending'
                """,
                (self.customer_id, transfer_id),
            )
            if cursor.rowcount != 1:
                raise BankError("window_closed")
            self.conn.execute(
                """
                INSERT INTO transactions
                    (customer_id, id, account_id, ts, amount_fen, counterparty,
                     category, memo, channel, country, card_present)
                VALUES (?, ?, ?, ?, ?, ?, 'transfer', ?, 'transfer', 'CN', 0)
                """,
                (
                    self.customer_id,
                    self._make_id("txn"),
                    row["account_id"],
                    now_dt.isoformat(),
                    -row["amount_fen"],
                    row["contact_name"],
                    row["note"],
                ),
            )
            updated = dict(row)
            updated["status"] = "released"
            result = _transfer(updated)
        except BaseException:
            if began_transaction:
                self.conn.rollback()
            raise
        if began_transaction:
            self.conn.commit()
        assert result is not None
        return result

    # ------------------------------------------------------- scheduled / requests

    def schedule_transfer(
        self,
        to_contact_id: str,
        amount_fen: int,
        day_of_month: int,
        note: str,
    ) -> ScheduledTransfer:
        _require_amount(amount_fen)
        if not _is_int(day_of_month) or not 1 <= day_of_month <= 31:
            raise BankError("invalid_state", "day_of_month must be between 1 and 31")
        note = note or ""
        with self.conn:
            self._contact_row(to_contact_id)
            scheduled_id = self._make_id("sch")
            self.conn.execute(
                """
                INSERT INTO scheduled_transfers
                    (customer_id, id, to_contact_id, amount_fen, day_of_month,
                     note, status)
                VALUES (?, ?, ?, ?, ?, ?, 'scheduled')
                """,
                (
                    self.customer_id,
                    scheduled_id,
                    to_contact_id,
                    amount_fen,
                    day_of_month,
                    note,
                ),
            )
            return ScheduledTransfer(
                id=scheduled_id,
                to_contact_id=to_contact_id,
                amount_fen=amount_fen,
                day_of_month=day_of_month,
                note=note,
                status="scheduled",
            )

    def cancel_scheduled(self, scheduled_id: str) -> ScheduledTransfer:
        with self.conn:
            row = self.conn.execute(
                """
                SELECT * FROM scheduled_transfers
                WHERE customer_id = ? AND id = ?
                """,
                (self.customer_id, scheduled_id),
            ).fetchone()
            if row is None:
                raise BankError("not_found", "scheduled transfer not found")
            if row["status"] != "scheduled":
                raise BankError("invalid_state")
            self.conn.execute(
                """
                UPDATE scheduled_transfers SET status = 'cancelled'
                WHERE customer_id = ? AND id = ?
                """,
                (self.customer_id, scheduled_id),
            )
            updated = dict(row)
            updated["status"] = "cancelled"
            return _scheduled(updated)

    def request_payment(
        self, from_contact_id: str, amount_fen: int, note: str
    ) -> PaymentRequest:
        _require_amount(amount_fen)
        note = note or ""
        with self.conn:
            self._contact_row(from_contact_id)
            request_id = self._make_id("req")
            self.conn.execute(
                """
                INSERT INTO payment_requests
                    (customer_id, id, from_contact_id, amount_fen, note, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
                """,
                (
                    self.customer_id,
                    request_id,
                    from_contact_id,
                    amount_fen,
                    note,
                ),
            )
            return PaymentRequest(
                id=request_id,
                from_contact_id=from_contact_id,
                amount_fen=amount_fen,
                note=note,
                status="pending",
            )

    def cancel_request(self, request_id: str) -> PaymentRequest:
        with self.conn:
            row = self.conn.execute(
                "SELECT * FROM payment_requests WHERE customer_id = ? AND id = ?",
                (self.customer_id, request_id),
            ).fetchone()
            if row is None:
                raise BankError("not_found", "payment request not found")
            if row["status"] != "pending":
                raise BankError("invalid_state")
            self.conn.execute(
                """
                UPDATE payment_requests SET status = 'cancelled'
                WHERE customer_id = ? AND id = ?
                """,
                (self.customer_id, request_id),
            )
            updated = dict(row)
            updated["status"] = "cancelled"
            return _payment_request(updated)

    # -------------------------------------------------------------- subscriptions

    def cancel_subscription(self, subscription_id: str) -> Subscription:
        with self.conn:
            row = self.conn.execute(
                "SELECT * FROM subscriptions WHERE customer_id = ? AND id = ?",
                (self.customer_id, subscription_id),
            ).fetchone()
            if row is None:
                raise BankError("not_found", "subscription not found")
            if row["status"] != "active":
                raise BankError("invalid_state")
            self.conn.execute(
                """
                UPDATE subscriptions SET status = 'cancelled'
                WHERE customer_id = ? AND id = ?
                """,
                (self.customer_id, subscription_id),
            )
            updated = dict(row)
            updated["status"] = "cancelled"
            return _subscription(updated)

    def restore_subscription(self, subscription_id: str) -> Subscription:
        with self.conn:
            row = self.conn.execute(
                "SELECT * FROM subscriptions WHERE customer_id = ? AND id = ?",
                (self.customer_id, subscription_id),
            ).fetchone()
            if row is None:
                raise BankError("not_found", "subscription not found")
            if row["status"] != "cancelled":
                raise BankError("invalid_state")
            self.conn.execute(
                """
                UPDATE subscriptions SET status = 'active'
                WHERE customer_id = ? AND id = ?
                """,
                (self.customer_id, subscription_id),
            )
            updated = dict(row)
            updated["status"] = "active"
            return _subscription(updated)

    # ---------------------------------------------------------------------- pots

    def move_to_pot(self, name: str, amount_fen: int) -> Pot:
        name = _require_text(name, "name")
        _require_amount(amount_fen)
        with self.conn:
            account = self._current_account()
            if account["balance_fen"] < amount_fen:
                raise BankError("insufficient_funds")
            row = self.conn.execute(
                "SELECT * FROM pots WHERE customer_id = ? AND name = ?",
                (self.customer_id, name),
            ).fetchone()
            if row is None:
                pot_id = self._make_id("pot")
                self.conn.execute(
                    """
                    INSERT INTO pots (customer_id, id, name, balance_fen)
                    VALUES (?, ?, ?, ?)
                    """,
                    (self.customer_id, pot_id, name, amount_fen),
                )
                balance = amount_fen
            else:
                pot_id = row["id"]
                balance = row["balance_fen"] + amount_fen
                self.conn.execute(
                    "UPDATE pots SET balance_fen = ? WHERE customer_id = ? AND id = ?",
                    (balance, self.customer_id, pot_id),
                )
            self.conn.execute(
                """
                UPDATE accounts SET balance_fen = balance_fen - ?
                WHERE customer_id = ? AND id = ?
                """,
                (amount_fen, self.customer_id, account["id"]),
            )
            return Pot(id=pot_id, name=name, balance_fen=balance)

    def move_from_pot(self, name: str, amount_fen: int) -> Pot:
        name = _require_text(name, "name")
        _require_amount(amount_fen)
        with self.conn:
            row = self.conn.execute(
                "SELECT * FROM pots WHERE customer_id = ? AND name = ?",
                (self.customer_id, name),
            ).fetchone()
            if row is None:
                raise BankError("not_found", "pot not found")
            if row["balance_fen"] < amount_fen:
                raise BankError("insufficient_funds")
            account = self._current_account()
            balance = row["balance_fen"] - amount_fen
            self.conn.execute(
                "UPDATE pots SET balance_fen = ? WHERE customer_id = ? AND id = ?",
                (balance, self.customer_id, row["id"]),
            )
            self.conn.execute(
                """
                UPDATE accounts SET balance_fen = balance_fen + ?
                WHERE customer_id = ? AND id = ?
                """,
                (amount_fen, self.customer_id, account["id"]),
            )
            return Pot(id=row["id"], name=row["name"], balance_fen=balance)

    # -------------------------------------------------------------- investments

    def subscribe_product(
        self, product_id: str, amount_fen: int, now: date | datetime
    ) -> Holding:
        _require_amount(amount_fen)
        now_dt = _as_now(now)
        with self.conn:
            product = self.conn.execute(
                "SELECT * FROM products WHERE id = ?", (product_id,)
            ).fetchone()
            if product is None:
                raise BankError("not_found", "product not found")
            if amount_fen < product["min_fen"]:
                raise BankError("invalid_amount", "below the product minimum")
            account = self._current_account()
            if account["balance_fen"] < amount_fen:
                raise BankError("insufficient_funds")
            self.conn.execute(
                """
                UPDATE accounts SET balance_fen = balance_fen - ?
                WHERE customer_id = ? AND id = ?
                """,
                (amount_fen, self.customer_id, account["id"]),
            )
            self.conn.execute(
                """
                INSERT INTO transactions
                    (customer_id, id, account_id, ts, amount_fen, counterparty,
                     category, memo, channel, country, card_present)
                VALUES (?, ?, ?, ?, ?, ?, 'investment', '', 'app', 'CN', 0)
                """,
                (
                    self.customer_id,
                    self._make_id("txn"),
                    account["id"],
                    now_dt.isoformat(),
                    -amount_fen,
                    product["name"],
                ),
            )
            row = self.conn.execute(
                """
                SELECT * FROM holdings
                WHERE customer_id = ? AND product_id = ?
                """,
                (self.customer_id, product_id),
            ).fetchone()
            if row is None:
                principal = value = amount_fen
                self.conn.execute(
                    """
                    INSERT INTO holdings
                        (customer_id, product_id, principal_fen, value_fen)
                    VALUES (?, ?, ?, ?)
                    """,
                    (self.customer_id, product_id, principal, value),
                )
            else:
                principal = row["principal_fen"] + amount_fen
                value = row["value_fen"] + amount_fen
                self.conn.execute(
                    """
                    UPDATE holdings SET principal_fen = ?, value_fen = ?
                    WHERE customer_id = ? AND product_id = ?
                    """,
                    (principal, value, self.customer_id, product_id),
                )
            return Holding(
                product_id=product_id, principal_fen=principal, value_fen=value
            )

    # ------------------------------------------------------------ merchant orders

    def merchant_order(
        self,
        merchant: str,
        item_code: str,
        deliver_on: date | datetime,
        now: date | datetime,
        *,
        funding_pot_id: str | None = None,
    ) -> MerchantOrder:
        merchant = _require_text(merchant, "merchant")
        item_code = _require_text(item_code, "item_code")
        if funding_pot_id is not None:
            funding_pot_id = _require_text(funding_pot_id, "funding_pot_id")
        try:
            amount_fen = price_of(merchant, item_code)
        except KeyError as exc:
            raise BankError("invalid_state", "unknown merchant or item") from exc

        now_dt = _as_now(now)
        deliver_date = _as_date(deliver_on)
        if deliver_date < now_dt.date():
            raise BankError("invalid_state", "deliver_on cannot be in the past")
        _ensure_merchant_order_funding_column(self.conn)
        with self.conn:
            if funding_pot_id is None:
                account = self._current_account()
                if account["balance_fen"] < amount_fen:
                    raise BankError("insufficient_funds")
                self.conn.execute(
                    """
                    UPDATE accounts SET balance_fen = balance_fen - ?
                    WHERE customer_id = ? AND id = ?
                    """,
                    (amount_fen, self.customer_id, account["id"]),
                )
            else:
                pot = self._pot_row(funding_pot_id)
                if pot["balance_fen"] < amount_fen:
                    raise BankError("insufficient_funds")
                self.conn.execute(
                    """
                    UPDATE pots SET balance_fen = balance_fen - ?
                    WHERE customer_id = ? AND id = ?
                    """,
                    (amount_fen, self.customer_id, pot["id"]),
                )
            order_id = self._make_id("ord")
            self.conn.execute(
                """
                INSERT INTO merchant_orders
                    (customer_id, id, merchant, item, amount_fen, deliver_on,
                     status, created_at, funding_pot_id)
                VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?)
                """,
                (
                    self.customer_id,
                    order_id,
                    merchant,
                    item_code,
                    amount_fen,
                    deliver_date.isoformat(),
                    now_dt.isoformat(),
                    funding_pot_id,
                ),
            )
            return MerchantOrder(
                id=order_id,
                merchant=merchant,
                item=item_code,
                amount_fen=amount_fen,
                deliver_on=deliver_date,
                status="open",
                created_at=now_dt,
                funding_pot_id=funding_pot_id,
            )

    def cancel_order(
        self, order_id: str, now: date | datetime
    ) -> MerchantOrder:
        now_dt = _as_now(now)
        with self.conn:
            row = self.conn.execute(
                "SELECT * FROM merchant_orders WHERE customer_id = ? AND id = ?",
                (self.customer_id, order_id),
            ).fetchone()
            if row is None:
                raise BankError("not_found", "order not found")
            if row["status"] != "open":
                raise BankError("invalid_state")
            if not now_dt.date() < date.fromisoformat(row["deliver_on"]):
                raise BankError("window_closed")
            funding_pot_id = (
                row["funding_pot_id"] if "funding_pot_id" in row.keys() else None
            )
            if funding_pot_id is None:
                account = self._current_account()
                self.conn.execute(
                    """
                    UPDATE accounts SET balance_fen = balance_fen + ?
                    WHERE customer_id = ? AND id = ?
                    """,
                    (row["amount_fen"], self.customer_id, account["id"]),
                )
            else:
                pot = self._pot_row(funding_pot_id)
                self.conn.execute(
                    """
                    UPDATE pots SET balance_fen = balance_fen + ?
                    WHERE customer_id = ? AND id = ?
                    """,
                    (row["amount_fen"], self.customer_id, pot["id"]),
                )
            self.conn.execute(
                """
                UPDATE merchant_orders SET status = 'cancelled'
                WHERE customer_id = ? AND id = ?
                """,
                (self.customer_id, order_id),
            )
            updated = dict(row)
            updated["status"] = "cancelled"
            if "funding_pot_id" not in updated:
                updated["funding_pot_id"] = None
            return _order(updated)

    # --------------------------------------------------------------------- cards

    def freeze_card(self, card_id: str) -> Card:
        with self.conn:
            row = self._card_row(card_id)
            if row["status"] in {"frozen", "lost"}:
                return _card(row)
            if row["status"] != "active":
                raise BankError("invalid_state")
            self.conn.execute(
                "UPDATE cards SET status = 'frozen' WHERE customer_id = ? AND id = ?",
                (self.customer_id, card_id),
            )
            updated = dict(row)
            updated["status"] = "frozen"
            return _card(updated)

    def unfreeze_card(self, card_id: str) -> Card:
        with self.conn:
            row = self._card_row(card_id)
            if row["status"] == "lost" or row["status"] != "frozen":
                raise BankError("invalid_state")
            self.conn.execute(
                "UPDATE cards SET status = 'active' WHERE customer_id = ? AND id = ?",
                (self.customer_id, card_id),
            )
            updated = dict(row)
            updated["status"] = "active"
            return _card(updated)

    def report_card_lost(self, card_id: str) -> Card:
        with self.conn:
            row = self._card_row(card_id)
            if row["status"] == "lost":
                raise BankError("invalid_state")
            self.conn.execute(
                "UPDATE cards SET status = 'lost' WHERE customer_id = ? AND id = ?",
                (self.customer_id, card_id),
            )
            updated = dict(row)
            updated["status"] = "lost"
            return _card(updated)

    def previous_card_limit(self, card_id: str) -> int | None:
        self._card_row(card_id)
        if self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='card_limit_history'").fetchone() is None:
            return None
        row = self.conn.execute("SELECT previous_limit_fen FROM card_limit_history WHERE customer_id=? AND card_id=?", (self.customer_id, card_id)).fetchone()
        return int(row[0]) if row is not None else None

    def set_card_limit(self, card_id: str, fen: int, *, restore: bool = False) -> Card:
        _require_non_negative(fen)
        with self.conn:
            row = self._card_row(card_id)
            if row["status"] in {"lost", "closed"}:
                raise BankError("invalid_state", "card is unavailable")
            if restore:
                if self.previous_card_limit(card_id) != fen:
                    raise BankError("invalid_state", "previous limit changed; request it again")
                self.conn.execute("DELETE FROM card_limit_history WHERE customer_id=? AND card_id=?", (self.customer_id, card_id))
            elif fen != row["online_limit_fen"]:
                self.conn.execute("CREATE TABLE IF NOT EXISTS card_limit_history (customer_id TEXT NOT NULL, card_id TEXT NOT NULL, previous_limit_fen INTEGER NOT NULL, PRIMARY KEY(customer_id,card_id))")
                self.conn.execute("INSERT INTO card_limit_history (customer_id,card_id,previous_limit_fen) VALUES (?,?,?) ON CONFLICT(customer_id,card_id) DO UPDATE SET previous_limit_fen=excluded.previous_limit_fen", (self.customer_id,card_id,row["online_limit_fen"]))
            self.conn.execute("UPDATE cards SET online_limit_fen=? WHERE customer_id=? AND id=?", (fen,self.customer_id,card_id))
            updated = dict(row)
            updated["online_limit_fen"] = fen
            return _card(updated)

    def restore_card_limit(self, card_id: str) -> Card | None:
        """Green restoration can tighten only; increases use set_card_limit approval."""
        with self.conn:
            row = self._card_row(card_id)
            previous = self.previous_card_limit(card_id)
            if previous is None:
                return None
            if previous > row["online_limit_fen"]:
                raise BankError("invalid_state", "restoring a higher limit needs fresh approval")
            # Refuse an obsolete high elder limit through this green undo path.
            persona = self.conn.execute("SELECT persona_id FROM customers WHERE id=?", (self.customer_id,)).fetchone()
            elder = (persona is not None and persona[0] == "retiree") or compliance.elder_mode(self.conn, self.customer_id)
            if elder and previous > 50_000:
                raise BankError("invalid_state", "restoration needs a policy-approved limit change")
            if row["status"] in {"lost", "closed"}:
                raise BankError("invalid_state", "card is unavailable")
            self.conn.execute("UPDATE cards SET online_limit_fen=? WHERE customer_id=? AND id=?", (previous,self.customer_id,card_id))
            self.conn.execute("DELETE FROM card_limit_history WHERE customer_id=? AND card_id=?", (self.customer_id,card_id))
            updated = dict(row)
            updated["online_limit_fen"] = previous
            return _card(updated)

    def card_details(self, now: datetime) -> list[dict]:
        """Authenticated customer view: masked cards and effective controls."""
        from bank import controls
        blocked = controls.blocked_categories(self.conn, self.customer_id, now)
        details = []
        for card in self.list_cards():
            settings = controls.get_controls(self.conn, self.customer_id, card.id)
            details.append({"last4": "****" + card.last4, "kind": card.kind,
                "status": card.status, "online_limit_fen": card.online_limit_fen,
                "controls": settings,
                "effective_controls": {key: enabled and card.status == "active" for key, enabled in settings.items()},
                "active_blocks": sorted(set(blocked) | ({"online_shopping"} if not settings["online"] else set()))})
        return details

    def create_virtual_card(self, limit_fen: int) -> Card:
        _require_non_negative(limit_fen)
        with self.conn:
            account = self._current_account()
            card_id = self._make_id("card")
            last4 = _last4(card_id)
            self.conn.execute(
                """
                INSERT INTO cards
                    (customer_id, id, kind, last4, status, online_limit_fen,
                     account_id)
                VALUES (?, ?, 'virtual', ?, 'active', ?, ?)
                """,
                (
                    self.customer_id,
                    card_id,
                    last4,
                    limit_fen,
                    account["id"],
                ),
            )
            return Card(
                id=card_id,
                kind="virtual",
                last4=last4,
                status="active",
                online_limit_fen=limit_fen,
                account_id=account["id"],
            )

    def close_virtual_card(self, card_id: str) -> Card:
        with self.conn:
            row = self._card_row(card_id)
            if row["kind"] != "virtual" or row["status"] in {"closed", "lost"}:
                raise BankError("invalid_state")
            self.conn.execute(
                "UPDATE cards SET status = 'closed' WHERE customer_id = ? AND id = ?",
                (self.customer_id, card_id),
            )
            updated = dict(row)
            updated["status"] = "closed"
            return _card(updated)

    # ------------------------------------------------------------------- helpers

    def _current_account(self) -> sqlite3.Row:
        row = self.conn.execute(
            """
            SELECT * FROM accounts
            WHERE customer_id = ? AND type = 'current'
            ORDER BY id LIMIT 1
            """,
            (self.customer_id,),
        ).fetchone()
        if row is None:
            raise BankError("invalid_state", "current account not found")
        return row

    def _pot_row(self, pot_id: str) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM pots WHERE customer_id = ? AND id = ?",
            (self.customer_id, pot_id),
        ).fetchone()
        if row is None:
            raise BankError("not_found", "pot not found")
        return row

    def _contact_row(self, contact_id: str) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM contacts WHERE customer_id = ? AND id = ?",
            (self.customer_id, contact_id),
        ).fetchone()
        if row is None:
            raise BankError("not_found", "contact not found")
        return row

    def _card_row(self, card_id: str) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM cards WHERE customer_id = ? AND id = ?",
            (self.customer_id, card_id),
        ).fetchone()
        if row is None:
            raise BankError("not_found", "card not found")
        return row
