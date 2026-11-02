"""Hash-chained, append-only audit log.

Every security-relevant decision (intent, plan, policy, mandate, MFA, run,
undo, grounding, veto, scam, lock, handoff, reply) is written to one global
chain on the bank's SQLite connection.  Each entry commits to the hash of the
previous entry, so editing, deleting or reordering rows breaks verification.
A separate head anchor (HMAC over the current ``seq:hash``) detects truncation
of the newest rows.  Inside the database the anchor can be rolled back together
with the rows by anyone with DB access, so an optional *external* anchor file
(see :class:`AuditLog`) records the same head outside the database.  In
production that anchor belongs on a separate log server or WORM storage; the
file used here is the demo equivalent.

All timestamps come from the caller (``now``); this module never reads the
wall clock itself.
"""

from __future__ import annotations

import hashlib
import threading
import hmac
import json
import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import StrEnum
from pathlib import Path

from bank.money import fmt_fen

GENESIS_PREV_HASH = "0" * 64
ANCHOR_ROW_ID = 1

#: Any run of this many digits in a payload string looks like a phone number
#: or an account number and is rejected before it can reach the log.
_PII_RE = re.compile(r"\d{11,}")

#: Chinese digit characters normalised to ASCII before the PII check.
_CN_DIGITS = str.maketrans("零〇一二三四五六七八九幺", "001234567891")

#: Spaces, dots and dashes sitting between two digits are ignored by the guard.
_DIGIT_SEPARATORS_RE = re.compile(r"(?<=\d)[\s.\-\u2010-\u2015\u2212]+(?=\d)")


_APPEND_LOCK = threading.Lock()

class AuditKind(StrEnum):
    """Stable event kinds stored in :data:`security_audit.kind`."""

    INTENT = "INTENT"
    PLAN = "PLAN"
    CHECK = "CHECK"
    POLICY = "POLICY"
    MANDATE = "MANDATE"
    MFA = "MFA"
    RUN = "RUN"
    UNDO = "UNDO"
    GROUND = "GROUND"
    VETO = "VETO"
    SCAM = "SCAM"
    LOCK = "LOCK"
    HANDOFF = "HANDOFF"
    RECALL = "RECALL"
    SETTINGS = "SETTINGS"
    REPLY = "REPLY"
    SANDBOX = "SANDBOX"


AUDIT_KINDS = frozenset(kind.value for kind in AuditKind)

SCHEMA = """
CREATE TABLE IF NOT EXISTS security_audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_audit_customer_plan
    ON security_audit(customer_id, plan_id, seq);

-- A chain can never fork: each hash is the predecessor of at most one entry.
CREATE UNIQUE INDEX IF NOT EXISTS ux_audit_prev_hash
    ON security_audit(prev_hash);

CREATE TRIGGER IF NOT EXISTS security_audit_no_update
BEFORE UPDATE ON security_audit
BEGIN
    SELECT RAISE(ABORT, 'security_audit is append-only');
END;

CREATE TRIGGER IF NOT EXISTS security_audit_no_delete
BEFORE DELETE ON security_audit
BEGIN
    SELECT RAISE(ABORT, 'security_audit is append-only');
END;

CREATE TABLE IF NOT EXISTS security_audit_anchor (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_seq INTEGER NOT NULL,
    last_hash TEXT NOT NULL,
    anchor TEXT NOT NULL,
    ts TEXT NOT NULL
);
"""


class AuditError(Exception):
    """Domain error carrying a stable machine-readable ``code``."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """One decoded audit row."""

    seq: int
    customer_id: str
    session_id: str
    plan_id: str
    ts: str
    kind: str
    payload: dict
    prev_hash: str
    hash: str


@dataclass(frozen=True, slots=True)
class VerifyResult:
    """Outcome of :func:`verify`."""

    ok: bool
    entries: int
    first_bad_seq: int | None = None
    reason: str | None = None


def create_audit_schema(conn: sqlite3.Connection) -> None:
    """Create the audit chain, its immutability triggers and the head anchor."""
    conn.executescript(SCHEMA)
    conn.commit()


# --------------------------------------------------------------------- hashing


def _as_ts(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return datetime.combine(value, time.min).isoformat()
    if isinstance(value, str) and value:
        return value
    raise TypeError("now must be a date, datetime or ISO string")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _binding(
    *,
    customer_id: str,
    session_id: str,
    plan_id: str,
    ts: str,
    kind: str,
    payload: object,
) -> dict[str, object]:
    # ``seq`` is deliberately excluded: the entry hash commits to the previous
    # hash, which already fixes the position in the chain.
    return {
        "customer_id": customer_id,
        "session_id": session_id,
        "plan_id": plan_id,
        "ts": ts,
        "kind": kind,
        "payload": payload,
    }


def _entry_hash(prev_hash: str, binding: dict[str, object]) -> str:
    material = (prev_hash + _canonical_json(binding)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _anchor_hash(secret_key: bytes, last_seq: int, last_hash: str) -> str:
    message = f"{last_seq}:{last_hash}".encode("utf-8")
    return hmac.new(secret_key, message, hashlib.sha256).hexdigest()


# ------------------------------------------------------------------------ PII


def _normalise_for_pii(value: str) -> str:
    """Fold common obfuscations before looking for a long digit run.

    NFKC turns full-width digits into ASCII, Unicode format characters (``Cf``)
    are removed, Chinese digits become 0-9, and spaces/dashes/dots between
    digits are dropped.  Masked tails such as ``\u00b7\u00b7\u00b70932`` keep their
    leading dots because they are not between digits.
    """
    text = unicodedata.normalize("NFKC", value)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = text.translate(_CN_DIGITS)
    return _DIGIT_SEPARATORS_RE.sub("", text)


def _check_pii(value: object) -> None:
    """Recursively reject payload strings containing 11+ consecutive digits."""
    if type(value) is _CodeFingerprint:
        return
    if isinstance(value, str):
        if _PII_RE.search(_normalise_for_pii(value)):
            raise AuditError(
                "pii",
                "audit payload contains a raw phone/account number; "
                "use a token label such as <PAYEE_1> or a masked tail such as "
                "\u00b7\u00b7\u00b730932",
            )
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_pii(key)
            _check_pii(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_pii(item)


class _CodeFingerprint(str):
    """Trusted computed digest; external strings never acquire this type.

    Long digit runs occur naturally in SHA-256. Accept only hashes computed
    here from code, never arbitrary hex-looking payload values or key names.
    JSON serialises this as an ordinary string without changing the hash chain.
    """

    def __new__(cls, code: str):
        return super().__new__(cls, hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest())


# ---------------------------------------------------------------------- chain


def _last_row(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT seq, hash FROM security_audit ORDER BY seq DESC LIMIT 1"
    ).fetchone()


def _store_anchor(
    conn: sqlite3.Connection, secret_key: bytes, last_seq: int, last_hash: str, ts: str
) -> str:
    anchor = _anchor_hash(secret_key, last_seq, last_hash)
    conn.execute(
        """
        INSERT INTO security_audit_anchor (id, last_seq, last_hash, anchor, ts)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            last_seq = excluded.last_seq,
            last_hash = excluded.last_hash,
            anchor = excluded.anchor,
            ts = excluded.ts
        """,
        (ANCHOR_ROW_ID, last_seq, last_hash, anchor, ts),
    )
    return anchor


def _write_anchor_file(path: Path, seq: int, last_hash: str, anchor: str) -> None:
    """Atomically persist the external head anchor (temp file + ``os.replace``)."""
    payload = {"seq": seq, "hash": last_hash, "hmac": anchor}
    data = _canonical_json(payload)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _read_anchor_file(path: Path) -> tuple[int, str, str] | None:
    """Return ``(seq, hash, hmac)`` from the external anchor, or ``None``."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return int(payload["seq"]), str(payload["hash"]), str(payload["hmac"])
    except (OSError, TypeError, ValueError, KeyError):
        return None


def _row_to_entry(row: sqlite3.Row) -> AuditEntry:
    try:
        payload = json.loads(row["payload_json"])
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise AuditError("payload", "audit payload is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise AuditError("payload", "audit payload must be a JSON object")
    return AuditEntry(
        seq=int(row["seq"]),
        customer_id=row["customer_id"],
        session_id=row["session_id"],
        plan_id=row["plan_id"],
        ts=row["ts"],
        kind=row["kind"],
        payload=payload,
        prev_hash=row["prev_hash"],
        hash=row["hash"],
    )


class AuditLog:
    """Append and query the tamper-evident audit chain.

    When ``anchor_path`` is given, the current head anchor ``{seq, hash, hmac}``
    is additionally written to that file after every append (atomically, via a
    temp file and ``os.replace``).  This keeps a copy of the head outside the
    database, so deleting the newest rows *and* restoring the old in-database
    anchor row is still detected.  In production the external anchor belongs on
    a separate log server or WORM storage; a local file is the demo equivalent.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        secret_key: bytes,
        anchor_path: Path | None = None,
    ) -> None:
        if not isinstance(secret_key, (bytes, bytearray)) or not secret_key:
            raise ValueError("secret_key must be a non-empty bytes value")
        self.conn = conn
        self._secret_key = bytes(secret_key)
        self._anchor_path = Path(anchor_path) if anchor_path is not None else None

    # ------------------------------------------------------------------ writes

    def append(
        self,
        *,
        customer_id: str,
        session_id: str,
        plan_id: str,
        kind: AuditKind | str,
        payload: dict,
        now: date | datetime | str,
    ) -> AuditEntry:
        """Append one entry and advance the head anchor atomically.

        Payloads are part of the hash and of the human-readable audit trail, so
        they must not contain raw personal data.  Any string value (or key)
        containing a run of 11 or more digits is rejected with
        :class:`AuditError` code ``"pii"``.  Phone and account numbers must be
        represented by token labels (``<PAYEE_1>``), masked tails (``\u00b7\u00b7\u00b70932``)
        or, for money, integer amounts in fen (``amount_fen``).
        """
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        _check_pii(payload)
        try:
            kind_value = AuditKind(kind).value
        except (TypeError, ValueError) as exc:
            raise AuditError("kind", f"unknown audit kind: {kind!r}") from exc

        ts = _as_ts(now)
        payload_json = _canonical_json(payload)
        binding = _binding(
            customer_id=customer_id,
            session_id=session_id,
            plan_id=plan_id,
            ts=ts,
            kind=kind_value,
            payload=payload,
        )

        # Read-last-then-insert must be atomic or concurrent writers fork the
        # chain: a process lock for threads sharing a connection,
        # BEGIN IMMEDIATE for other connections to the same file.
        with _APPEND_LOCK, self.conn:
            if not self.conn.in_transaction:
                self.conn.execute("BEGIN IMMEDIATE")
            last = _last_row(self.conn)
            prev_hash = last["hash"] if last is not None else GENESIS_PREV_HASH
            entry_hash = _entry_hash(prev_hash, binding)
            cursor = self.conn.execute(
                """
                INSERT INTO security_audit
                    (customer_id, session_id, plan_id, ts, kind, payload_json,
                     prev_hash, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    customer_id,
                    session_id,
                    plan_id,
                    ts,
                    kind_value,
                    payload_json,
                    prev_hash,
                    entry_hash,
                ),
            )
            seq = int(cursor.lastrowid)
            anchor = _store_anchor(self.conn, self._secret_key, seq, entry_hash, ts)

        if self._anchor_path is not None:
            _write_anchor_file(self._anchor_path, seq, entry_hash, anchor)

        return AuditEntry(
            seq=seq,
            customer_id=customer_id,
            session_id=session_id,
            plan_id=plan_id,
            ts=ts,
            kind=kind_value,
            payload=payload,
            prev_hash=prev_hash,
            hash=entry_hash,
        )

    def anchor(self, now: date | datetime | str) -> str:
        """Recompute and store the HMAC anchor for the current head."""
        ts = _as_ts(now)
        with self.conn:
            last = _last_row(self.conn)
            last_seq = int(last["seq"]) if last is not None else 0
            last_hash = last["hash"] if last is not None else GENESIS_PREV_HASH
            anchor = _store_anchor(
                self.conn, self._secret_key, last_seq, last_hash, ts
            )
        if self._anchor_path is not None:
            _write_anchor_file(
                self._anchor_path, last_seq, last_hash, anchor
            )
        return anchor

    # ------------------------------------------------------------------- reads

    def entries_for(
        self, customer_id: str, plan_id: str | None = None
    ) -> list[AuditEntry]:
        """Return this customer's entries in chain order (optionally by plan)."""
        sql = "SELECT * FROM security_audit WHERE customer_id = ?"
        params: list[object] = [customer_id]
        if plan_id is not None:
            sql += " AND plan_id = ?"
            params.append(plan_id)
        sql += " ORDER BY seq ASC"
        return [_row_to_entry(row) for row in self.conn.execute(sql, params)]

    def explain(
        self, customer_id: str, plan_id: str, lang: str = "en"
    ) -> list[str]:
        """Return customer-readable sentences for one customer/plan history.

        Internal kinds, tool ids, mandate ids and rule codes never reach the
        customer; each meaningful event becomes one plain sentence prefixed
        with its ``HH:MM:SS`` time.
        """
        if lang not in {"en", "zh"}:
            raise ValueError("lang must be 'en' or 'zh'")
        entries = self.entries_for(customer_id, plan_id)
        lines: list[str] = []
        for entry in entries:
            lines.extend(_explain_entry(entry, lang))

        if not lines and any(entry.kind == "INTENT" for entry in entries):
            prefix = _time_of(entries[0].ts) + " "
            lines = [
                prefix
                + ("你发来一条请求" if lang == "zh" else "You sent a request")
            ]

        cleaned: list[str] = []
        for line in lines:
            # Defensive: even if a row was written around ``append`` (raw DB
            # access), never surface an internal id or long digit run.
            line = _INTERNAL_ID_RE.sub("", line)
            line = _PII_RE.sub("[redacted]", line)
            line = re.sub(r" {2,}", " ", line).strip()
            if line:
                cleaned.append(line)
        return cleaned

    def verify(self) -> VerifyResult:
        """Recompute the whole chain and check the head anchors."""
        return verify(self.conn, self._secret_key, anchor_path=self._anchor_path)


# -------------------------------------------------------------------- verify


def verify(
    conn: sqlite3.Connection,
    secret_key: bytes | None = None,
    anchor_path: Path | None = None,
) -> VerifyResult:
    """Recompute the chain and check it against the stored head anchors.

    Without ``secret_key`` the anchor's recorded ``seq``/``hash`` are compared
    to the live head, which still catches a deleted tail.  When the key is
    supplied the HMAC itself is also checked.

    With ``anchor_path`` the external anchor file must match the live head
    exactly and its HMAC must verify.  An external anchor ahead of the database
    head means the newest rows were deleted and the in-database anchor was
    rolled back: the result is ``"truncated (external anchor ahead)"``.
    """
    rows = conn.execute("SELECT * FROM security_audit ORDER BY seq ASC").fetchall()
    entries = len(rows)

    prev_hash = GENESIS_PREV_HASH
    prev_seq = 0
    for row in rows:
        seq = int(row["seq"])
        if seq != prev_seq + 1:
            return VerifyResult(False, entries, seq, "sequence gap")
        if row["prev_hash"] != prev_hash:
            return VerifyResult(False, entries, seq, "prev_hash break")
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError):
            return VerifyResult(False, entries, seq, "payload is not JSON")
        binding = _binding(
            customer_id=row["customer_id"],
            session_id=row["session_id"],
            plan_id=row["plan_id"],
            ts=row["ts"],
            kind=row["kind"],
            payload=payload,
        )
        expected_hash = _entry_hash(prev_hash, binding)
        if not hmac.compare_digest(expected_hash, row["hash"]):
            return VerifyResult(False, entries, seq, "hash mismatch")
        prev_hash = row["hash"]
        prev_seq = seq

    anchor_row = conn.execute(
        "SELECT * FROM security_audit_anchor WHERE id = ?", (ANCHOR_ROW_ID,)
    ).fetchone()
    if anchor_row is None:
        if rows:
            return VerifyResult(False, entries, prev_seq + 1, "anchor missing")
    else:
        if (
            int(anchor_row["last_seq"]) != prev_seq
            or anchor_row["last_hash"] != prev_hash
        ):
            return VerifyResult(False, entries, prev_seq + 1, "anchor mismatch")

        if secret_key is not None:
            expected_anchor = _anchor_hash(secret_key, prev_seq, prev_hash)
            if not hmac.compare_digest(expected_anchor, anchor_row["anchor"]):
                return VerifyResult(
                    False, entries, prev_seq + 1, "anchor mismatch"
                )

    if anchor_path is not None:
        external = _read_anchor_file(Path(anchor_path))
        if external is None:
            return VerifyResult(False, entries, prev_seq + 1, "anchor missing")
        ext_seq, ext_hash, ext_hmac = external
        if ext_seq > prev_seq:
            return VerifyResult(
                False,
                entries,
                prev_seq + 1,
                "truncated (external anchor ahead)",
            )
        if ext_seq != prev_seq or ext_hash != prev_hash:
            return VerifyResult(False, entries, prev_seq + 1, "anchor mismatch")
        if secret_key is not None:
            expected_external = _anchor_hash(secret_key, ext_seq, ext_hash)
            if not hmac.compare_digest(expected_external, ext_hmac):
                return VerifyResult(
                    False, entries, prev_seq + 1, "anchor mismatch"
                )

    return VerifyResult(True, entries, None, None)


# ------------------------------------------------------------------- explain

_TIER_WORDS = {
    "green": ("直接执行", "runs directly"),
    "low": ("直接执行", "runs directly"),
    "yellow": ("需要你确认", "needs your confirmation"),
    "medium": ("需要你确认", "needs your confirmation"),
    "red": ("需要短信验证码和刷脸", "needs an SMS code and a face check"),
    "high": ("需要短信验证码和刷脸", "needs an SMS code and a face check"),
}

#: Tool identifiers are internal; customers only see these plain names.
_TOOL_WORDS = {
    "standing_order": ("定时转账", "scheduled transfer"),
    # reads
    "get_balance": ("查询余额", "check the balance"),
    "list_transactions": ("查询流水", "view transactions"),
    "spending_report": ("查看消费报告", "view the spending report"),
    "run_analysis": ("分析消费规律", "analyse spending patterns"),
    "detect_anomalies": ("检查异常交易", "check for unusual payments"),
    "list_subscriptions": ("查看订阅", "view subscriptions"),
    "list_cards": ("查看卡片", "view cards"),
    "list_products": ("查看理财产品", "view products"),
    "get_portfolio": ("查看持仓", "view holdings"),
    "find_contact": ("查找联系人", "look up a contact"),
    "list_contacts": ("查看联系人", "list contacts"),
    "get_contact_date": ("查询纪念日", "look up a date"),
    "can_cover": ("检查余额是否足够", "check the balance covers it"),
    "sandbox_chart": ("生成图表", "make a chart"),
    "chart": ("生成图表", "make a chart"),
    # writes
    "set_reminder": ("设置提醒", "set a reminder"),
    "transfer": ("转账", "bank transfer"),
    "schedule_transfer": ("定时转账", "schedule a transfer"),
    "request_payment": ("发起收款", "request a payment"),
    "merchant_order": ("下单", "place an order"),
    "cancel_subscription": ("取消订阅", "cancel a subscription"),
    "reactivate_subscription": ("恢复订阅", "reactivate a subscription"),
    "move_to_pot": ("存入储蓄罐", "move money to a savings pot"),
    "freeze_card": ("冻结卡片", "freeze the card"),
    "unfreeze_card": ("解冻卡片", "unfreeze the card"),
    "set_card_limit": ("调整卡片限额", "change the card limit"),
    "create_virtual_card": ("开通虚拟卡", "create a virtual card"),
    "report_card_lost": ("挂失卡片", "report the card lost"),
    "subscribe_product": ("购买理财", "buy a product"),
    "redeem_product": ("模拟赎回理财", "redeem a mock product"),
    "change_password": ("修改密码", "change the password"),
    # controls
    "ask_user": ("确认信息", "confirm the details"),
    "refuse": ("拒绝请求", "decline the request"),
    "handoff_human": ("转人工", "hand off to a person"),
    # undo-style writes
    "cancel_transfer": ("撤销转账", "cancel the transfer"),
    "cancel_scheduled": ("撤销定时转账", "cancel the scheduled transfer"),
    "cancel_request": ("撤销收款请求", "cancel the payment request"),
    "restore_subscription": ("恢复订阅", "restore the subscription"),
    "move_from_pot": ("从储蓄罐转出", "move money from the savings pot"),
    "cancel_order": ("取消订单", "cancel the order"),
    "restore_card_limit": ("恢复卡片限额", "restore the card limit"),
    "close_virtual_card": ("关闭虚拟卡", "close the virtual card"),
    # legacy aliases
    "merchant_pay": ("付款", "pay a merchant"),
    "pay": ("付款", "make a payment"),
    "payee_add": ("添加收款人", "add a payee"),
    "add_payee": ("添加收款人", "add a payee"),
    "card_lock": ("锁卡", "lock the card"),
    "lock_card": ("锁卡", "lock the card"),
    "balance": ("查询余额", "check the balance"),
    "transactions": ("查询流水", "view transactions"),
    "history": ("查询流水", "view transactions"),
    "refund": ("退款", "issue a refund"),
    "bill": ("缴费", "pay a bill"),
    # tools the fine-tuned planner was not trained on yet
    "list_bills": ("查看账单", "view bills"),
    "list_reminders": ("查看提醒", "view reminders"),
    "list_scheduled": ("查看定时转账", "view standing orders"),
    "stop_standing_order": ("停止定时转账", "stop a standing order"),
    "pay_bill": ("缴费", "pay a bill"),
    "installment_quote": ("分期试算", "quote an instalment plan"),
    "deposit_quote": ("定期试算", "quote a deposit"),
    "forecast_cashflow": ("余额预测", "forecast the balance"),
    "help_answer": ("查询银行知识", "look up banking information"),
    "list_transfers": ("查看转账状态", "check transfer status"),
    "verify_payee": ("核验收款人", "check a payee"),
    "set_card_control": ("调整卡片开关", "change a card switch"),
    "block_merchant": ("屏蔽消费类别", "block a spending category"),
    "unblock_merchant": ("申请解除屏蔽", "request lifting a block"),
}

#: Registry tools that only read; a RUN line for these says "Checked", not
#: "Done", so a balance check never reads like money moved.
_READ_TOOLS = frozenset(
    {
        "get_balance",
        "list_transactions",
        "spending_report",
        "run_analysis",
        "detect_anomalies",
        "list_subscriptions",
        "list_cards",
        "list_products",
        "get_portfolio",
        "find_contact",
        "list_contacts",
        "get_contact_date",
        "can_cover",
        "sandbox_chart",
        "list_bills",
        "list_reminders",
        "list_scheduled",
        "installment_quote",
        "deposit_quote",
        "forecast_cashflow",
        "help_answer",
        "list_transfers",
        "verify_payee",
    }
)

_VETO_WORDS = (
    ("scam", "检测到诈骗风险", "fraud risk detected"),
    ("fraud", "检测到诈骗风险", "fraud risk detected"),
    ("lock", "支付已锁定", "payment is locked"),
    ("window", "撤销时间已过", "the undo window has passed"),
    ("undo", "撤销时间已过", "the undo window has passed"),
)

#: Defensive scrub for ids such as ``m_000001`` or ``tr_000009``.
_INTERNAL_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:m|tr|tx|txn|plan|sess|req|step)_[A-Za-z0-9]+"
    r"(?![A-Za-z0-9_])"
)

#: A bare uppercase token (``R_NEW_PAYEE``) is an internal rule code, not prose.
_TAG_WORD_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,}$")


def _time_of(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%H:%M:%S")
    except (TypeError, ValueError):
        return ts[:8]


def _fmt_optional(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return fmt_fen(value)
    return None


def _zh(lang: str) -> bool:
    return lang == "zh"


def _reason_texts(payload: dict, lang: str) -> list[str]:
    """Return only plain-language reason messages, never internal codes."""
    texts: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        if isinstance(value, str):
            text = value.strip()
            if not text or _TAG_WORD_RE.match(text) or text in seen:
                return
            seen.add(text)
            texts.append(text)
        elif isinstance(value, dict):
            for key in (f"message_{lang}", f"reason_{lang}", "message", "reason"):
                text = value.get(key)
                if isinstance(text, str) and text.strip():
                    add(text)
                    return
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)

    for key in ("reasons", "messages"):
        if key in payload:
            add(payload[key])
    for key in (
        f"message_{lang}",
        f"reason_{lang}",
        "message",
        "reason",
        "message_en" if _zh(lang) else "message_zh",
    ):
        if key in payload:
            add(payload[key])
    return texts


def _plain_action(payload: dict, lang: str) -> str:
    for key in ("action_name", "display_name", "label"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raw = ""
    for key in ("tool", "tool_id", "action", "action_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            raw = value.strip()
            break
    if not raw:
        return "操作" if _zh(lang) else "the action"
    words = _TOOL_WORDS.get(raw.lower())
    if words is not None:
        return words[0] if _zh(lang) else words[1]
    if any(ord(ch) > 127 for ch in raw):
        return raw
    # Unknown tool id: show a generic label rather than the internal id.
    return "操作" if _zh(lang) else "the action"


def _plan_count(payload: dict) -> int | None:
    for key in ("step_count", "n_steps", "count"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    for key in ("steps", "plan"):
        value = payload.get(key)
        if isinstance(value, (list, tuple)):
            return len(value)
    return None


def _money_phrase(payload: dict, lang: str) -> str | None:
    amount = _fmt_optional(payload.get("amount_fen"))
    daily = _fmt_optional(payload.get("daily_total_fen"))
    limit = _fmt_optional(payload.get("daily_limit_fen")) or _fmt_optional(
        payload.get("limit_fen")
    )
    if _zh(lang):
        parts: list[str] = []
        if amount is not None:
            parts.append(amount)
        if daily is not None:
            text = f"今日累计 {daily}"
            if limit is not None:
                text += f" / 限额 {limit}"
            parts.append(text)
        return "，".join(parts) if parts else None
    parts = []
    if amount is not None:
        parts.append(amount)
    if daily is not None:
        text = f"{daily} today"
        if limit is not None:
            text += f" / limit {limit}"
        parts.append(text)
    return ", ".join(parts) if parts else None


def _policy_sentence(payload: dict, lang: str) -> str:
    action = _plain_action(payload, lang)
    tier = payload.get("tier")
    words = _TIER_WORDS.get(tier.lower()) if isinstance(tier, str) else None
    if words is not None:
        body = f"{action}{words[0]}" if _zh(lang) else f"{action} {words[1]}"
    else:
        body = f"已检查：{action}" if _zh(lang) else f"Checked: {action}"
    money = _money_phrase(payload, lang)
    if money is not None:
        body += f"：{money}" if _zh(lang) else f": {money}"
    reasons = _reason_texts(payload, lang)
    if reasons:
        joiner = "；" if _zh(lang) else "; "
        body += joiner.join([""] + reasons)
    return body


def _policy_sentences(payload: dict, lang: str) -> list[str]:
    steps = payload.get("steps")
    if isinstance(steps, list):
        merged = [
            {**payload, **step}
            for step in steps
            if isinstance(step, dict)
        ]
        merged = [item for item in merged if isinstance(item.get("tier"), str)]
        if merged:
            return [_policy_sentence(item, lang) for item in merged]
    return [_policy_sentence(payload, lang)]


def _mfa_sentence(payload: dict, lang: str) -> str:
    status = ""
    for key in ("status", "result", "outcome", "state"):
        value = payload.get(key)
        if isinstance(value, str):
            status = value.strip().lower()
            break
    flags = [payload.get(key) for key in ("ok", "success", "verified", "code_ok")]
    failed = status in {
        "wrong",
        "failed",
        "fail",
        "invalid",
        "error",
        "denied",
        "mismatch",
    } or any(flag is False for flag in flags)
    if failed:
        return "验证码错误" if _zh(lang) else "The verification code was incorrect"
    passed = status in {"ok", "passed", "success", "verified", "pass"} or any(
        flag is True for flag in flags
    )
    if passed:
        return "验证通过" if _zh(lang) else "Verification passed"
    return "你确认了这笔操作" if _zh(lang) else "You confirmed this action"


def _veto_reason(payload: dict, lang: str) -> str:
    code = ""
    for key in ("reason", "reason_code", "code", "veto", "decision", "rule"):
        value = payload.get(key)
        if isinstance(value, str):
            code = value.lower()
            break
    for needle, zh_text, en_text in _VETO_WORDS:
        if needle in code:
            return zh_text if _zh(lang) else en_text
    reasons = _reason_texts(payload, lang)
    if reasons:
        return reasons[0]
    return "安全策略已拦截" if _zh(lang) else "blocked by a security rule"


def _explain_entry(entry: AuditEntry, lang: str) -> list[str]:
    """Return zero or more customer-readable sentences for one event."""
    payload = entry.payload
    kind = entry.kind
    prefix = _time_of(entry.ts) + " "
    zh = _zh(lang)
    sentences: list[str] = []

    if kind == "PLAN":
        count = _plan_count(payload)
        if count is None:
            sentences.append(
                "安办已理解你的请求" if zh else "AnBan understood your request"
            )
        elif zh:
            sentences.append(f"安办理解为：{count} 个步骤")
        else:
            sentences.append(f"AnBan understood this as {count} steps")
    elif kind == "POLICY":
        sentences.extend(_policy_sentences(payload, lang))
    elif kind == "MANDATE":
        sentences.append("你确认了这笔操作" if zh else "You confirmed this action")
    elif kind == "MFA":
        sentences.append(_mfa_sentence(payload, lang))
    elif kind == "RUN":
        action = _plain_action(payload, lang)
        amount = _fmt_optional(payload.get("amount_fen"))
        tool = payload.get("tool")
        read_only = isinstance(tool, str) and tool.strip().lower() in _READ_TOOLS
        if tool == "standing_order":
            status = payload.get("status")
            if status == "paid":
                sentences.append(
                    "定时转账已提交，仍需通过到账检查" if zh
                    else "Scheduled transfer submitted, subject to release checks"
                )
            elif status == "skipped":
                sentences.append("定时转账已跳过，未付款" if zh else "Scheduled transfer skipped; no payment submitted")
            elif status == "missed":
                sentences.append("定时转账已错过，未付款" if zh else "Scheduled transfer missed; no payment submitted")
            else:
                sentences.append("定时转账状态待核实" if zh else "Scheduled transfer status needs review")
        elif zh:
            verb = "已查看：" if read_only else "已执行："
            sentences.append(verb + action + (f" {amount}" if amount else ""))
        else:
            verb = "Checked: " if read_only else "Done: "
            sentences.append(verb + action + (f" {amount}" if amount else ""))
    elif kind == "VETO":
        reason = _veto_reason(payload, lang)
        sentences.append(
            f"未执行：{reason}" if zh else f"Not done: {reason}"
        )
    elif kind == "SCAM":
        sentences.append(
            "检测到诈骗风险，转账已暂停 24 小时"
            if zh
            else "Fraud risk detected; transfers are paused for 24 hours"
        )
    elif kind == "UNDO":
        amount = _fmt_optional(payload.get("amount_fen"))
        if zh:
            sentences.append("已撤销" + (f" {amount}" if amount else "这笔操作"))
        else:
            sentences.append("Undone" + (f" {amount}" if amount else " this action"))
    elif kind == "HANDOFF":
        sentences.append("已转人工" if zh else "Handed to a person")
    elif kind == "SANDBOX":
        if payload.get("backend") == "subprocess-dev":
            sentences.append("开发模式分析已完成" if zh else "Development analysis attempted")
        elif payload.get("ok"):
            sentences.append("隔离分析已完成" if zh else "Isolated analysis completed")
        else:
            sentences.append("隔离分析未完成" if zh else "Isolated analysis could not complete")
    elif kind == "SETTINGS":
        if payload.get("queued"):
            sentences.append(
                "已申请降低保护设置，24 小时后生效" if zh
                else "Lowering a protection was requested; it takes effect in 24 hours"
            )
        else:
            sentences.append("设置已更新" if zh else "Settings updated")
    elif kind == "RECALL":
        amount = _fmt_optional(payload.get("amount_fen"))
        if zh:
            sentences.append("已提交追回申请" + (f" {amount}" if amount else ""))
        else:
            sentences.append("Recall request sent" + (f" for {amount}" if amount else ""))
    elif kind == "LOCK":
        sentences.append("支付已锁定" if zh else "Payment is locked")
    else:
        # INTENT, CHECK, GROUND, REPLY and unknown kinds carry no customer value.
        return []

    return [prefix + sentence for sentence in sentences]
