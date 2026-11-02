"""Signed confirmation mandates, OTP verification and lockout handling.

A mandate binds one write step to the exact arguments the user confirmed.
The executor may consume it exactly once; a second attempt is a replay.
All timestamps are supplied by the caller (``now``), so the module never
reads the wall clock itself.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

PAYMENT_LOCK_PROTECTIVE_ACTIONS = frozenset({"freeze_card", "report_card_lost"})

ERROR_CODES = {
    "not_found",
    "bad_signature",
    "expired",
    "mismatch",
    "not_approved",
    "mfa_required",
    "co_approval_required",
    "not_co_approver",
    "replayed",
    "locked",
    "revoked",
}

OPEN_STATUSES = ("issued", "mfa_pending", "approved")
MFA_FAILURE_WINDOW = timedelta(hours=24)
MFA_LOCKOUT_DURATION = timedelta(hours=24)
MFA_MAX_FAILURES = 3
OTP_TTL = timedelta(minutes=5)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS security_mandates (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    action TEXT NOT NULL,
    payee_ref TEXT NOT NULL,
    amount_fen INTEGER NOT NULL,
    tier TEXT NOT NULL,
    nonce TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('issued', 'mfa_pending', 'approved', 'consumed', 'revoked', 'expired')
    ),
    sig TEXT NOT NULL,
    co_approval_required INTEGER NOT NULL DEFAULT 0,
    co_approved_by TEXT,
    mfa_verified INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS ix_mandates_customer_plan
    ON security_mandates(customer_id, plan_id);

CREATE TABLE IF NOT EXISTS security_otp (
    mandate_id TEXT NOT NULL,
    purpose TEXT NOT NULL DEFAULT 'customer_mfa',
    customer_id TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (mandate_id, purpose)
);

CREATE TABLE IF NOT EXISTS security_mfa_failures (
    customer_id TEXT NOT NULL,
    at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_mfa_failures_customer_at
    ON security_mfa_failures(customer_id, at);

CREATE TABLE IF NOT EXISTS security_lockouts (
    customer_id TEXT PRIMARY KEY,
    locked_until TEXT NOT NULL,
    reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS security_unlocks (
    customer_id TEXT NOT NULL,
    by_staff TEXT NOT NULL,
    at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_security_unlocks_customer
    ON security_unlocks(customer_id, at);

CREATE TABLE IF NOT EXISTS security_scam_freezes (
    customer_id TEXT PRIMARY KEY,
    frozen_until TEXT NOT NULL,
    flags_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

IdFactory = Callable[[str], str]


class MandateError(Exception):
    """Domain error carrying a stable machine-readable ``code``."""

    def __init__(self, code: str, message: str | None = None) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown mandate error code: {code!r}")
        self.code = code
        super().__init__(message or code)


@dataclass
class Mandate:
    id: str
    customer_id: str
    session_id: str
    plan_id: str
    step_id: str
    action: str
    payee_ref: str
    amount_fen: int
    tier: str
    nonce: str
    expires_at: datetime
    status: str
    sig: str
    co_approval_required: bool = False
    co_approved_by: str | None = None
    mfa_verified: bool = False


def create_security_schema(conn: sqlite3.Connection) -> None:
    """Create the mandate, OTP, failure and lockout tables."""
    conn.executescript(SCHEMA)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(security_otp)")}
    if "purpose" not in columns:
        conn.executescript(
            """
            ALTER TABLE security_otp RENAME TO security_otp_legacy;
            CREATE TABLE security_otp (
                mandate_id TEXT NOT NULL,
                purpose TEXT NOT NULL DEFAULT 'customer_mfa',
                customer_id TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (mandate_id, purpose)
            );
            INSERT INTO security_otp
                (mandate_id, purpose, customer_id, code_hash, salt, expires_at,
                 attempts)
            SELECT mandate_id, 'customer_mfa', customer_id, code_hash, salt,
                   expires_at, attempts
            FROM security_otp_legacy;
            DROP TABLE security_otp_legacy;
            """
        )
    conn.commit()


def _as_datetime(value: date | datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError("now must be a date, datetime or ISO string")


def _next_mandate_id(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT value FROM meta WHERE key = 'mandate_id_seq'"
    ).fetchone()
    seq = int(row["value"]) + 1 if row is not None else 1
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('mandate_id_seq', ?)",
        (str(seq),),
    )
    return f"m_{seq:06d}"


def _row_to_mandate(row: sqlite3.Row) -> Mandate:
    return Mandate(
        id=row["id"],
        customer_id=row["customer_id"],
        session_id=row["session_id"],
        plan_id=row["plan_id"],
        step_id=row["step_id"],
        action=row["action"],
        payee_ref=row["payee_ref"],
        amount_fen=row["amount_fen"],
        tier=row["tier"],
        nonce=row["nonce"],
        expires_at=datetime.fromisoformat(row["expires_at"]),
        status=row["status"],
        sig=row["sig"],
        co_approval_required=bool(row["co_approval_required"]),
        co_approved_by=row["co_approved_by"],
        mfa_verified=bool(row["mfa_verified"]),
    )


class MandateService:
    """Issue, approve, verify and consume signed step mandates."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        secret_key: bytes,
        id_factory: IdFactory | None = None,
        otp_rng=None,
    ) -> None:
        if not isinstance(secret_key, (bytes, bytearray)) or not secret_key:
            raise ValueError("secret_key must be a non-empty bytes value")
        self.conn = conn
        self._secret_key = bytes(secret_key)
        self._id_factory = id_factory
        self._otp_rng = otp_rng

    # ------------------------------------------------------------------ signing

    def _make_id(self) -> str:
        if self._id_factory is None:
            return _next_mandate_id(self.conn)
        try:
            return str(self._id_factory("m"))
        except TypeError:
            return str(self._id_factory())

    def _random_hex(self, nbytes: int) -> str:
        rng = self._otp_rng
        if rng is not None:
            token_hex = getattr(rng, "token_hex", None)
            if callable(token_hex):
                return token_hex(nbytes)
            getrandbits = getattr(rng, "getrandbits", None)
            if callable(getrandbits):
                return getrandbits(nbytes * 8).to_bytes(nbytes, "big").hex()
        return secrets.token_hex(nbytes)

    def _otp_code(self) -> str:
        rng = self._otp_rng
        if rng is None:
            return f"{secrets.randbelow(1_000_000):06d}"
        randrange = getattr(rng, "randrange", None)
        if callable(randrange):
            return f"{randrange(1_000_000):06d}"
        randint = getattr(rng, "randint", None)
        if callable(randint):
            return f"{randint(0, 999_999):06d}"
        return f"{secrets.randbelow(1_000_000):06d}"

    def _hash_otp(self, salt: str, otp: str) -> str:
        message = f"{salt}:{otp}".encode("utf-8")
        return hmac.new(self._secret_key, message, hashlib.sha256).hexdigest()

    @staticmethod
    def _binding(payload: dict[str, object]) -> bytes:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def _sign(self, payload: dict[str, object]) -> str:
        return hmac.new(
            self._secret_key, self._binding(payload), hashlib.sha256
        ).hexdigest()

    @staticmethod
    def _row_binding(row: sqlite3.Row) -> dict[str, object]:
        return {
            "customer_id": row["customer_id"],
            "session_id": row["session_id"],
            "plan_id": row["plan_id"],
            "step_id": row["step_id"],
            "action": row["action"],
            "payee_ref": row["payee_ref"],
            "amount_fen": row["amount_fen"],
            "tier": row["tier"],
            "co_approval_required": bool(row["co_approval_required"]),
            "expires_at": row["expires_at"],
            "nonce": row["nonce"],
        }

    def _check_signature(self, row: sqlite3.Row) -> None:
        expected = self._sign(self._row_binding(row))
        if not hmac.compare_digest(expected, row["sig"]):
            raise MandateError("bad_signature")

    # ------------------------------------------------------------------- reads

    def _row(self, mandate_id: str, customer_id: str) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM security_mandates WHERE id = ? AND customer_id = ?",
            (mandate_id, customer_id),
        ).fetchone()
        if row is None:
            raise MandateError("not_found")
        return row

    def get(self, mandate_id: str, customer_id: str) -> Mandate:
        """Return a mandate scoped to its owner."""
        return _row_to_mandate(self._row(mandate_id, customer_id))

    def co_approvers(self, customer_id: str) -> list[str]:
        """Return this customer's contacts marked as co-approvers."""
        rows = self.conn.execute(
            """
            SELECT id FROM contacts
            WHERE customer_id = ? AND lower(note) LIKE '%co-approver%'
            ORDER BY id
            """,
            (customer_id,),
        ).fetchall()
        return [row["id"] for row in rows]

    def is_payments_locked(self, customer_id: str, now: date | datetime | str) -> bool:
        now_dt = _as_datetime(now)
        row = self.conn.execute(
            "SELECT locked_until FROM security_lockouts WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
        if row is None:
            return False
        return now_dt < datetime.fromisoformat(row["locked_until"])

    # ---------------------------------------------------------------- lifecycle

    def _check_payment_lock(self, row: sqlite3.Row, customer_id: str, now_dt: datetime) -> None:
        if not self.is_payments_locked(customer_id, now_dt):
            return
        # Only a signed protective action is exempt, never an action string
        # supplied by a caller of verify_and_consume or a tampered DB row.
        self._check_signature(row)
        if row["action"] not in PAYMENT_LOCK_PROTECTIVE_ACTIONS:
            raise MandateError("locked")

    def issue(
        self,
        *,
        customer_id: str,
        session_id: str,
        plan_id: str,
        step_id: str,
        action: str,
        payee_ref: str,
        amount_fen: int,
        tier: str,
        now: date | datetime | str,
        ttl: timedelta = timedelta(minutes=5),
        co_approval_required: bool = False,
    ) -> Mandate:
        now_dt = _as_datetime(now)
        if tier not in {"yellow", "red"}:
            raise MandateError("not_approved", "a mandate requires a yellow or red tier")
        if self.is_payments_locked(customer_id, now_dt) and action not in PAYMENT_LOCK_PROTECTIVE_ACTIONS:
            raise MandateError("locked")

        expires_at = now_dt + ttl
        nonce = self._random_hex(16)
        status = "mfa_pending" if tier == "red" else "issued"
        payload = {
            "customer_id": customer_id,
            "session_id": session_id,
            "plan_id": plan_id,
            "step_id": step_id,
            "action": action,
            "payee_ref": payee_ref,
            "amount_fen": int(amount_fen),
            "tier": tier,
            "co_approval_required": bool(co_approval_required),
            "expires_at": expires_at.isoformat(),
            "nonce": nonce,
        }
        sig = self._sign(payload)

        with self.conn:
            mandate_id = self._make_id()
            self.conn.execute(
                """
                INSERT INTO security_mandates
                    (id, customer_id, session_id, plan_id, step_id, action,
                     payee_ref, amount_fen, tier, nonce, expires_at, status, sig,
                     co_approval_required, co_approved_by, mfa_verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0)
                """,
                (
                    mandate_id,
                    customer_id,
                    session_id,
                    plan_id,
                    step_id,
                    action,
                    payee_ref,
                    int(amount_fen),
                    tier,
                    nonce,
                    expires_at.isoformat(),
                    status,
                    sig,
                    int(bool(co_approval_required)),
                ),
            )
        return _row_to_mandate(self._row(mandate_id, customer_id))

    def start_mfa(
        self, mandate_id: str, customer_id: str, now: date | datetime | str
    ) -> str:
        """Create a fresh 6-digit OTP for a red-tier mandate and return it."""
        now_dt = _as_datetime(now)
        with self.conn:
            row = self._row(mandate_id, customer_id)
            self._check_usable(row, now_dt)
            self._check_payment_lock(row, customer_id, now_dt)
            if row["status"] != "mfa_pending" or bool(row["mfa_verified"]):
                raise MandateError("not_approved", "mandate is not awaiting MFA")

            code = self._otp_code()
            salt = self._random_hex(16)
            self.conn.execute(
                """
                INSERT OR REPLACE INTO security_otp
                    (mandate_id, purpose, customer_id, code_hash, salt,
                     expires_at, attempts)
                VALUES (?, 'customer_mfa', ?, ?, ?, ?, 0)
                """,
                (
                    mandate_id,
                    customer_id,
                    self._hash_otp(salt, code),
                    salt,
                    (now_dt + OTP_TTL).isoformat(),
                ),
            )
        return code

    def verify_mfa(
        self,
        mandate_id: str,
        customer_id: str,
        *,
        otp: str,
        retyped_amount_fen: int,
        face_ok: bool,
        now: date | datetime | str,
    ) -> Mandate:
        now_dt = _as_datetime(now)
        failure: str | None = None

        with self.conn:
            row = self._row(mandate_id, customer_id)
            self._check_usable(row, now_dt)
            self._check_payment_lock(row, customer_id, now_dt)
            if row["status"] != "mfa_pending" or bool(row["mfa_verified"]):
                raise MandateError("not_approved", "mandate is not awaiting MFA")

            otp_row = self.conn.execute(
                "SELECT * FROM security_otp"
                " WHERE mandate_id = ? AND customer_id = ?"
                " AND purpose = 'customer_mfa'",
                (mandate_id, customer_id),
            ).fetchone()
            otp_ok = False
            if otp_row is not None:
                code = str(otp).strip()
                if code.isdigit() and len(code) < 6:
                    code = code.zfill(6)
                expected = self._hash_otp(otp_row["salt"], code)
                unexpired = now_dt < datetime.fromisoformat(otp_row["expires_at"])
                otp_ok = unexpired and hmac.compare_digest(
                    expected, otp_row["code_hash"]
                )

            amount_ok = retyped_amount_fen == row["amount_fen"]
            face_passed = bool(face_ok)

            if otp_ok and amount_ok and face_passed:
                finished = bool(row["co_approval_required"]) and (
                    row["co_approved_by"] is None
                )
                self.conn.execute(
                    """
                    UPDATE security_mandates
                    SET mfa_verified = 1, status = ?
                    WHERE id = ? AND customer_id = ?
                    """,
                    ("mfa_pending" if finished else "approved", mandate_id, customer_id),
                )
                self.conn.execute(
                    "DELETE FROM security_otp"
                    " WHERE mandate_id = ? AND purpose = 'customer_mfa'",
                    (mandate_id,),
                )
            else:
                if otp_row is not None:
                    self.conn.execute(
                        "UPDATE security_otp SET attempts = attempts + 1"
                        " WHERE mandate_id = ? AND purpose = 'customer_mfa'",
                        (mandate_id,),
                    )
                locked = self._record_failure(customer_id, now_dt)
                failure = "locked" if locked else "mismatch"

        if failure is not None:
            raise MandateError(failure, "MFA verification failed")
        return _row_to_mandate(self._row(mandate_id, customer_id))

    def start_co_approval(
        self,
        mandate_id: str,
        customer_id: str,
        approver_contact_id: str,
        now: date | datetime | str,
    ) -> str:
        """Create a fresh OTP for a registered co-approver and return it."""
        now_dt = _as_datetime(now)
        with self.conn:
            row = self._row(mandate_id, customer_id)
            self._check_usable(row, now_dt)
            self._check_payment_lock(row, customer_id, now_dt)
            if approver_contact_id not in self.co_approvers(customer_id):
                raise MandateError("not_co_approver")
            if not bool(row["co_approval_required"]):
                raise MandateError("not_approved", "mandate does not need co-approval")
            if row["co_approved_by"] is not None:
                raise MandateError("not_approved", "mandate is already co-approved")

            code = self._otp_code()
            salt = self._random_hex(16)
            self.conn.execute(
                """
                INSERT OR REPLACE INTO security_otp
                    (mandate_id, purpose, customer_id, code_hash, salt,
                     expires_at, attempts)
                VALUES (?, 'co_approval', ?, ?, ?, ?, 0)
                """,
                (
                    mandate_id,
                    customer_id,
                    self._hash_otp(salt, code),
                    salt,
                    (now_dt + OTP_TTL).isoformat(),
                ),
            )
        return code

    def co_approve(
        self,
        mandate_id: str,
        customer_id: str,
        approver_contact_id: str,
        otp: str,
        now: date | datetime | str,
    ) -> Mandate:
        now_dt = _as_datetime(now)
        failure: str | None = None

        with self.conn:
            row = self._row(mandate_id, customer_id)
            self._check_usable(row, now_dt)
            self._check_payment_lock(row, customer_id, now_dt)
            if approver_contact_id not in self.co_approvers(customer_id):
                raise MandateError("not_co_approver")
            if not bool(row["mfa_verified"]):
                raise MandateError("mfa_required", "MFA must pass before co-approval")
            if not bool(row["co_approval_required"]):
                raise MandateError("not_approved", "mandate does not need co-approval")

            otp_row = self.conn.execute(
                "SELECT * FROM security_otp"
                " WHERE mandate_id = ? AND customer_id = ?"
                " AND purpose = 'co_approval'",
                (mandate_id, customer_id),
            ).fetchone()
            otp_ok = False
            if otp_row is not None:
                code = str(otp).strip()
                if code.isdigit() and len(code) < 6:
                    code = code.zfill(6)
                expected = self._hash_otp(otp_row["salt"], code)
                unexpired = now_dt < datetime.fromisoformat(otp_row["expires_at"])
                otp_ok = unexpired and hmac.compare_digest(
                    expected, otp_row["code_hash"]
                )

            if otp_ok:
                cursor = self.conn.execute(
                    """
                    UPDATE security_mandates
                    SET co_approved_by = ?, status = 'approved'
                    WHERE id = ? AND customer_id = ? AND co_approved_by IS NULL
                    """,
                    (approver_contact_id, mandate_id, customer_id),
                )
                if cursor.rowcount != 1:
                    raise MandateError("not_approved", "mandate is already co-approved")
                self.conn.execute(
                    "DELETE FROM security_otp"
                    " WHERE mandate_id = ? AND purpose = 'co_approval'",
                    (mandate_id,),
                )
            else:
                if otp_row is not None:
                    self.conn.execute(
                        "UPDATE security_otp SET attempts = attempts + 1"
                        " WHERE mandate_id = ? AND purpose = 'co_approval'",
                        (mandate_id,),
                    )
                locked = self._record_failure(customer_id, now_dt)
                failure = "locked" if locked else "mismatch"

        if failure is not None:
            raise MandateError(failure, "co-approval OTP verification failed")
        return _row_to_mandate(self._row(mandate_id, customer_id))

    def approve_yellow(
        self, mandate_id: str, customer_id: str, now: date | datetime | str
    ) -> Mandate:
        now_dt = _as_datetime(now)
        with self.conn:
            row = self._row(mandate_id, customer_id)
            self._check_usable(row, now_dt)
            self._check_payment_lock(row, customer_id, now_dt)
            if row["status"] == "issued":
                self.conn.execute(
                    """
                    UPDATE security_mandates SET status = 'approved'
                    WHERE id = ? AND customer_id = ?
                    """,
                    (mandate_id, customer_id),
                )
            elif row["status"] != "approved":
                raise MandateError("not_approved", "mandate is not awaiting confirmation")
        return _row_to_mandate(self._row(mandate_id, customer_id))

    def verify_and_consume(
        self,
        mandate_id: str,
        *,
        customer_id: str,
        session_id: str,
        plan_id: str,
        step_id: str,
        action: str,
        payee_ref: str,
        amount_fen: int,
        now: date | datetime | str,
    ) -> Mandate:
        now_dt = _as_datetime(now)
        with self.conn:
            row = self._row(mandate_id, customer_id)
            self._check_signature(row)
            self._check_usable(row, now_dt)

            self._check_payment_lock(row, customer_id, now_dt)

            if row["status"] == "mfa_pending":
                if bool(row["mfa_verified"]):
                    raise MandateError("co_approval_required")
                raise MandateError("mfa_required")
            if row["status"] != "approved":
                raise MandateError("not_approved")

            if bool(row["co_approval_required"]) and row["co_approved_by"] is None:
                raise MandateError("co_approval_required")

            if (
                row["session_id"] != session_id
                or row["plan_id"] != plan_id
                or row["step_id"] != step_id
                or row["action"] != action
                or row["payee_ref"] != payee_ref
                or int(row["amount_fen"]) != int(amount_fen)
            ):
                raise MandateError("mismatch")

            cursor = self.conn.execute(
                """
                UPDATE security_mandates SET status = 'consumed'
                WHERE id = ? AND customer_id = ? AND status = 'approved'
                """,
                (mandate_id, customer_id),
            )
            if cursor.rowcount != 1:
                raise MandateError("replayed")
            self.conn.execute(
                "DELETE FROM security_otp WHERE mandate_id = ?", (mandate_id,)
            )
        return _row_to_mandate(self._row(mandate_id, customer_id))

    # ------------------------------------------------------------------ safety

    def freeze_for_scam(
        self,
        customer_id: str,
        flags: Iterable[str],
        now: date | datetime | str,
        hours: int = 24,
    ) -> None:
        """Freeze payments for a customer-wide scam window.

        The window only ever grows and flags only accumulate, so a later turn
        (or a model that sees no scam) cannot shorten a victim's cooling-off
        period.
        """
        now_dt = _as_datetime(now)
        until = now_dt + timedelta(hours=hours)
        merged: set[str] = {str(flag) for flag in flags}
        row = self.conn.execute(
            "SELECT frozen_until, flags_json FROM security_scam_freezes"
            " WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
        if row is not None:
            existing = datetime.fromisoformat(row["frozen_until"])
            if existing > until:
                until = existing
            try:
                stored = json.loads(row["flags_json"])
            except (TypeError, ValueError):
                stored = ()
            if isinstance(stored, list):
                merged.update(str(flag) for flag in stored)

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO security_scam_freezes
                    (customer_id, frozen_until, flags_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                    frozen_until = excluded.frozen_until,
                    flags_json = excluded.flags_json,
                    updated_at = excluded.updated_at
                """,
                (
                    customer_id,
                    until.isoformat(),
                    json.dumps(sorted(merged)),
                    now_dt.isoformat(),
                ),
            )

    def scam_flags(
        self, customer_id: str, now: date | datetime | str
    ) -> frozenset[str]:
        """Return the customer's active scam flags; empty once the freeze expires."""
        now_dt = _as_datetime(now)
        row = self.conn.execute(
            "SELECT frozen_until, flags_json FROM security_scam_freezes"
            " WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
        if row is None or now_dt >= datetime.fromisoformat(row["frozen_until"]):
            return frozenset()
        try:
            flags = json.loads(row["flags_json"])
        except (TypeError, ValueError):
            return frozenset()
        if not isinstance(flags, list):
            return frozenset()
        return frozenset(str(flag) for flag in flags)

    def unlock(self, customer_id: str, by_staff: str, now: date | datetime | str) -> None:
        """Clear a lockout after a human has taken over the account.

        A staff takeover resets the rolling failure window as well, otherwise
        one more bad OTP would immediately re-lock the customer.  The takeover
        itself is recorded so the reset is auditable.  Staff verification also
        clears any scam freeze.
        """
        if not isinstance(by_staff, str) or not by_staff.strip():
            raise ValueError("by_staff is required")
        now_dt = _as_datetime(now)
        cutoff = (now_dt - MFA_FAILURE_WINDOW).isoformat()
        with self.conn:
            self.conn.execute(
                "DELETE FROM security_lockouts WHERE customer_id = ?", (customer_id,)
            )
            self.conn.execute(
                "DELETE FROM security_mfa_failures WHERE customer_id = ? AND at >= ?",
                (customer_id, cutoff),
            )
            self.conn.execute(
                "DELETE FROM security_scam_freezes WHERE customer_id = ?",
                (customer_id,),
            )
            self.conn.execute(
                "INSERT INTO security_unlocks (customer_id, by_staff, at) VALUES (?, ?, ?)",
                (customer_id, by_staff, now_dt.isoformat()),
            )

    def revoke_plan(
        self, customer_id: str, plan_id: str, now: date | datetime | str
    ) -> int:
        """Revoke every open mandate belonging to a plan (pause/cancel)."""
        _as_datetime(now)
        with self.conn:
            return self._revoke(customer_id, plan_id=plan_id)

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _check_usable(row: sqlite3.Row, now_dt: datetime) -> None:
        if row["status"] == "consumed":
            raise MandateError("replayed")
        if row["status"] == "revoked":
            raise MandateError("revoked")
        if row["status"] == "expired" or now_dt >= datetime.fromisoformat(
            row["expires_at"]
        ):
            raise MandateError("expired")

    def _record_failure(self, customer_id: str, now_dt: datetime) -> bool:
        self.conn.execute(
            "INSERT INTO security_mfa_failures (customer_id, at) VALUES (?, ?)",
            (customer_id, now_dt.isoformat()),
        )
        cutoff = (now_dt - MFA_FAILURE_WINDOW).isoformat()
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS failures FROM security_mfa_failures
            WHERE customer_id = ? AND at >= ?
            """,
            (customer_id, cutoff),
        ).fetchone()
        if int(row["failures"]) < MFA_MAX_FAILURES:
            return False

        # A wrong code while already locked does not push the lock further out.
        if not self.is_payments_locked(customer_id, now_dt):
            self.conn.execute(
                """
                INSERT OR REPLACE INTO security_lockouts
                    (customer_id, locked_until, reason)
                VALUES (?, ?, 'mfa_failures')
                """,
                (customer_id, (now_dt + MFA_LOCKOUT_DURATION).isoformat()),
            )
        # Keep an open card freeze: a typo must not leave a stolen card active
        #. Every other mandate, including a loss report, is revoked.
        self._revoke(customer_id, keep_actions=frozenset({"freeze_card"}))
        return True

    def _revoke(
        self,
        customer_id: str,
        plan_id: str | None = None,
        *,
        keep_actions: frozenset[str] = frozenset(),
    ) -> int:
        placeholders = ", ".join("?" for _ in OPEN_STATUSES)
        clauses = ["customer_id = ?", f"status IN ({placeholders})"]
        params: list[object] = [customer_id, *OPEN_STATUSES]
        if plan_id is not None:
            clauses.append("plan_id = ?")
            params.append(plan_id)
        if keep_actions:
            kept = ", ".join("?" for _ in keep_actions)
            clauses.append(f"action NOT IN ({kept})")
            params.extend(sorted(keep_actions))
        where = " AND ".join(clauses)

        self.conn.execute(
            f"DELETE FROM security_otp WHERE mandate_id IN"
            f" (SELECT id FROM security_mandates WHERE {where})",
            params,
        )
        cursor = self.conn.execute(
            f"UPDATE security_mandates SET status = 'revoked' WHERE {where}", params
        )
        return cursor.rowcount
