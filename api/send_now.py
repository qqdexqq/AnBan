"""Send-now: release one held transfer early, behind a fresh red mandate.

The existing Undo path *cancels* a pending transfer.  "Send now" is the other
direction: it releases the customer's own held transfer before its cooling
window ends.  Because that moves money sooner it is a new red action, never an
undo: it needs its own one-shot mandate, the current policy verdict, MFA (and
the elder guardian when policy asks for it), and a database transaction that
restores the mandate, ledger and undo record together on any failure.

Only the HTTP wrapper lives here; the policy engine stays the sole gate.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from agent.executor import _undo_sig, undo_record
from bank.api import Bank, BankError
from bank.money import fmt_fen
from security.audit import AuditKind, AuditLog
from security.mandate import MandateError, MandateService

ACTION = "release_transfer_now"
RELEASE_STEP = "release"
MANDATE_TTL = timedelta(minutes=5)
MAX_AMOUNT_FEN = 100_000_000_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS send_now_challenges (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    mandate_id TEXT NOT NULL,
    mandate_plan_id TEXT NOT NULL,
    transfer_id TEXT NOT NULL,
    contact_id TEXT NOT NULL,
    amount_fen INTEGER NOT NULL,
    account_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    needs_co_approval INTEGER NOT NULL DEFAULT 0,
    approver_id TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
"""


class SendNowError(Exception):
    """Domain refusal carrying a stable code and an HTTP status."""

    def __init__(self, code: str, status: int = 403) -> None:
        self.code = code
        self.status = status
        super().__init__(code)


class _TxConnection:
    """Connection proxy that keeps inner ``with conn`` blocks from committing.

    ``Bank``, ``MandateService`` and ``AuditLog`` all use ``with self.conn:``.
    Inside the send-now transaction those blocks must only release the context,
    not commit, or a later failure could leave half the release applied.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __enter__(self) -> "_TxConnection":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise SendNowError(f"invalid_{name}", 400)
    return value


def _fingerprint(
    *,
    customer_id: str,
    session_id: str,
    plan_id: str,
    step_id: str,
    transfer: dict[str, Any],
) -> str:
    """Bind the challenge to the exact signed receipt and bank row."""

    payload = {
        "customer_id": customer_id,
        "session_id": session_id,
        "plan_id": plan_id,
        "step_id": step_id,
        "transfer_id": str(transfer["id"]),
        "contact_id": str(transfer["to_contact_id"]),
        "amount_fen": int(transfer["amount_fen"]),
        "account_id": str(transfer["account_id"]),
        "created_at": str(transfer["created_at"]),
        "release_at": str(transfer["release_at"]),
    }
    material = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class SendNow:
    """Prepare, request codes for and confirm one early release."""

    def __init__(self, rt: Any) -> None:
        self.rt = rt
        self.conn: sqlite3.Connection = rt.conn
        self._ensure_schema()

    # ----------------------------------------------------------------- schema

    def _ensure_schema(self) -> None:
        # CREATE IF NOT EXISTS once; a challenge is never dropped or deleted,
        # so a receipt can be refused as replayed after a restart.
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table'"
            " AND name = 'send_now_challenges'"
        ).fetchone()
        if row is None:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # ------------------------------------------------------------------ reads

    def _require_session(self, customer_id: str, session_id: str, plan_id: str) -> None:
        record = self.rt.services.sessions.get(session_id)
        if record is not None:
            if record.get("customer_id") != customer_id:
                raise SendNowError("not_found", 404)
            return
        # Chat store fallback: only audit rows owned by this customer prove the
        # session belongs to them.
        owned = self.conn.execute(
            "SELECT 1 FROM security_audit"
            " WHERE customer_id = ? AND session_id = ? AND plan_id = ? LIMIT 1",
            (customer_id, session_id, plan_id),
        ).fetchone()
        if owned is None:
            raise SendNowError("not_found", 404)

    def _challenge(self, customer_id: str, challenge_id: str, now: datetime) -> Any:
        row = self.conn.execute(
            "SELECT * FROM send_now_challenges WHERE id = ? AND customer_id = ?",
            (challenge_id, customer_id),
        ).fetchone()
        if row is None:
            raise SendNowError("not_found", 404)
        if now >= datetime.fromisoformat(row["expires_at"]):
            raise SendNowError("expired", 403)
        return row

    def _receipt(
        self,
        conn: Any,
        customer_id: str,
        plan_id: str,
        step_id: str,
        *,
        direct: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        if direct:
            # Inside a transaction: plain SQL, never undo_record/executescript.
            row = conn.execute(
                "SELECT * FROM exec_undo"
                " WHERE customer_id = ? AND plan_id = ? AND step_id = ?",
                (customer_id, plan_id, step_id),
            ).fetchone()
        else:
            row = undo_record(conn, customer_id, plan_id, step_id)
        if row is None:
            raise SendNowError("not_found", 404)
        if str(row["customer_id"]) != customer_id:
            raise SendNowError("not_found", 404)
        if str(row["tool"]) != "cancel_transfer":
            raise SendNowError("invalid_receipt", 400)
        if int(row["used"]) != 0:
            raise SendNowError("replayed", 403)
        expected = _undo_sig(
            self.rt.services.secret_key,
            str(row["customer_id"]),
            str(row["plan_id"]),
            str(row["step_id"]),
            str(row["tool"]),
            str(row["args_json"]),
            str(row["created_at"]),
        )
        if not hmac.compare_digest(expected, str(row["sig"])):
            raise SendNowError("bad_signature", 403)
        try:
            args = json.loads(str(row["args_json"]))
        except (TypeError, ValueError):
            raise SendNowError("bad_signature", 403) from None
        if not isinstance(args, dict):
            raise SendNowError("bad_signature", 403)
        return row, args

    def _transfer_for(
        self, conn: Any, customer_id: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        # The executor records both keys in its dispatchers; accept either.
        transfer_id = args.get("transfer_id") or args.get("id")
        if not isinstance(transfer_id, str) or not transfer_id:
            raise SendNowError("bad_signature", 403)
        row = conn.execute(
            "SELECT id, to_contact_id, amount_fen, status, created_at, release_at,"
            " account_id FROM transfers WHERE customer_id = ? AND id = ?",
            (customer_id, transfer_id),
        ).fetchone()
        if row is None:
            raise SendNowError("not_found", 404)
        amount = row["amount_fen"]
        if not _is_int(amount) or not 0 < amount <= MAX_AMOUNT_FEN:
            raise SendNowError("invalid_amount", 400)
        recorded = args.get("amount_fen")
        if recorded is not None and (
            not _is_int(recorded) or recorded != amount
        ):
            raise SendNowError("bad_signature", 403)
        if str(row["status"]) != "pending":
            raise SendNowError("window_closed", 403)
        transfer = {
            "id": str(row["id"]),
            "to_contact_id": str(row["to_contact_id"]),
            "amount_fen": int(amount),
            "status": str(row["status"]),
            "created_at": str(row["created_at"]),
            "release_at": str(row["release_at"]),
            "account_id": str(row["account_id"]),
        }
        contact = conn.execute(
            "SELECT name FROM contacts WHERE customer_id = ? AND id = ?",
            (customer_id, transfer["to_contact_id"]),
        ).fetchone()
        if contact is None:
            raise SendNowError("not_found", 404)
        transfer["payee_label"] = str(contact["name"])
        return transfer

    def _mandate_for(self, challenge: Any, customer_id: str) -> Any:
        try:
            mandate = self.rt.services.mandates.get(
                str(challenge["mandate_id"]), customer_id
            )
        except MandateError as exc:
            raise _mandate_error(exc) from None
        if (
            mandate.session_id != str(challenge["session_id"])
            or mandate.plan_id != str(challenge["mandate_plan_id"])
            or mandate.step_id != RELEASE_STEP
            or mandate.action != ACTION
            or mandate.payee_ref != str(challenge["fingerprint"])
            or int(mandate.amount_fen) != int(challenge["amount_fen"])
            or mandate.tier != "red"
            or bool(mandate.co_approval_required)
            != bool(challenge["needs_co_approval"])
        ):
            raise SendNowError("mismatch", 403)
        return mandate

    # ---------------------------------------------------------------- policy

    def _audit_ok(self) -> None:
        if not self.rt.services.audit.verify().ok:
            raise SendNowError("audit_invalid", 403)

    def _policy(
        self, bank: Any, customer_id: str, transfer_id: str, now: datetime
    ) -> Any:
        mandates = self.rt.services.mandates
        if mandates.is_payments_locked(customer_id, now):
            raise SendNowError("locked", 403)
        if mandates.scam_flags(customer_id, now):
            raise SendNowError("scam", 403)
        try:
            from security.policy import evaluate_early_release
        except ImportError:  # pragma: no cover - deploy mismatch fails closed
            raise SendNowError("policy_unavailable", 403) from None
        try:
            decision = evaluate_early_release(
                bank=bank,
                transfer_id=transfer_id,
                now=now,
                config=self.rt.services.policy_config,
            )
        except SendNowError:
            raise
        except Exception:
            raise SendNowError("policy_unavailable", 403) from None
        if bool(getattr(decision, "vetoed", False)):
            raise SendNowError("policy_veto", 403)
        return decision

    # --------------------------------------------------------------- prepare

    def prepare(
        self, customer_id: str, session_id: str, plan_id: str, step_id: str
    ) -> dict[str, Any]:
        customer_id = _text(customer_id, "customer")
        session_id = _text(session_id, "session")
        plan_id = _text(plan_id, "plan")
        step_id = _text(step_id, "step")
        prefix = f"{session_id}:p"
        suffix = plan_id[len(prefix):] if plan_id.startswith(prefix) else ""
        if not suffix.isascii() or not suffix.isdigit():
            raise SendNowError("not_found", 404)

        now = self.rt.clock()
        self._require_session(customer_id, session_id, plan_id)
        _, args = self._receipt(self.conn, customer_id, plan_id, step_id)
        transfer = self._transfer_for(self.conn, customer_id, args)
        fingerprint = _fingerprint(
            customer_id=customer_id,
            session_id=session_id,
            plan_id=plan_id,
            step_id=step_id,
            transfer=transfer,
        )
        self._audit_ok()
        decision = self._policy(
            Bank(self.conn, customer_id), customer_id, transfer["id"], now
        )
        needs_co = bool(getattr(decision, "needs_co_approval", False))
        mandate_plan_id = f"{plan_id}:now:{secrets.token_hex(8)}"
        try:
            mandate = self.rt.services.mandates.issue(
                customer_id=customer_id,
                session_id=session_id,
                plan_id=mandate_plan_id,
                step_id=RELEASE_STEP,
                action=ACTION,
                payee_ref=fingerprint,
                amount_fen=transfer["amount_fen"],
                tier="red",
                now=now,
                ttl=MANDATE_TTL,
                co_approval_required=needs_co,
            )
        except MandateError as exc:
            raise _mandate_error(exc) from None

        challenge_id = secrets.token_hex(16)
        self.conn.execute(
            """
            INSERT INTO send_now_challenges
                (id, customer_id, session_id, plan_id, step_id, mandate_id,
                 mandate_plan_id, transfer_id, contact_id, amount_fen, account_id,
                 fingerprint, needs_co_approval, approver_id, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                challenge_id,
                customer_id,
                session_id,
                plan_id,
                step_id,
                mandate.id,
                mandate_plan_id,
                transfer["id"],
                transfer["to_contact_id"],
                transfer["amount_fen"],
                transfer["account_id"],
                fingerprint,
                int(needs_co),
                now.isoformat(),
                mandate.expires_at.isoformat(),
            ),
        )
        self.conn.commit()
        return {
            "challenge_id": challenge_id,
            "amount_fen": transfer["amount_fen"],
            "payee_label": transfer["payee_label"],
            "expires_at": mandate.expires_at.isoformat(),
            "needs_co_approval": needs_co,
        }

    # ---------------------------------------------------------- request-code

    def request_code(self, customer_id: str, challenge_id: str) -> dict[str, Any]:
        customer_id = _text(customer_id, "customer")
        challenge_id = _text(challenge_id, "challenge")
        now = self.rt.clock()
        challenge = self._challenge(customer_id, challenge_id, now)
        self._require_session(
            customer_id, str(challenge["session_id"]), str(challenge["plan_id"])
        )
        _, args = self._receipt(
            self.conn, customer_id, str(challenge["plan_id"]), str(challenge["step_id"])
        )
        transfer = self._transfer_for(self.conn, customer_id, args)
        if _fingerprint(
            customer_id=customer_id,
            session_id=str(challenge["session_id"]),
            plan_id=str(challenge["plan_id"]),
            step_id=str(challenge["step_id"]),
            transfer=transfer,
        ) != str(challenge["fingerprint"]):
            raise SendNowError("mismatch", 403)
        self._audit_ok()
        decision = self._policy(
            Bank(self.conn, customer_id), customer_id, transfer["id"], now
        )
        if bool(getattr(decision, "needs_co_approval", False)) != bool(
            challenge["needs_co_approval"]
        ):
            raise SendNowError("mismatch", 403)
        self._mandate_for(challenge, customer_id)

        approver_id: str | None = None
        co_code = ""
        try:
            code = self.rt.services.mandates.start_mfa(
                str(challenge["mandate_id"]), customer_id, now
            )
            if bool(challenge["needs_co_approval"]):
                approvers = self.rt.services.mandates.co_approvers(customer_id)
                if not approvers:
                    raise SendNowError("no_co_approver", 403)
                approver_id = approvers[0]
                co_code = self.rt.services.mandates.start_co_approval(
                    str(challenge["mandate_id"]), customer_id, approver_id, now
                )
                self.conn.execute(
                    "UPDATE send_now_challenges SET approver_id = ? WHERE id = ?",
                    (approver_id, challenge_id),
                )
                self.conn.commit()
        except MandateError as exc:
            raise _mandate_error(exc) from None
        return {
            "challenge_id": challenge_id,
            "needs_co_approval": bool(challenge["needs_co_approval"]),
            "otp": code,
            "co_otp": co_code,
        }

    # ---------------------------------------------------------------- confirm

    def confirm(
        self,
        customer_id: str,
        challenge_id: str,
        otp: str,
        retyped_amount_fen: int,
        face_ok: bool,
        co_otp: str = "",
    ) -> dict[str, Any]:
        customer_id = _text(customer_id, "customer")
        challenge_id = _text(challenge_id, "challenge")
        now = self.rt.clock()
        challenge = self._challenge(customer_id, challenge_id, now)
        session_id = str(challenge["session_id"])
        plan_id = str(challenge["plan_id"])
        step_id = str(challenge["step_id"])
        self._require_session(customer_id, session_id, plan_id)
        _, args = self._receipt(self.conn, customer_id, plan_id, step_id)
        transfer = self._transfer_for(self.conn, customer_id, args)
        fingerprint = _fingerprint(
            customer_id=customer_id,
            session_id=session_id,
            plan_id=plan_id,
            step_id=step_id,
            transfer=transfer,
        )
        if fingerprint != str(challenge["fingerprint"]):
            raise SendNowError("mismatch", 403)
        self._audit_ok()
        # Current receipt and policy both hold BEFORE any code is checked.
        self._policy(Bank(self.conn, customer_id), customer_id, transfer["id"], now)
        mandate = self._mandate_for(challenge, customer_id)

        try:
            if not mandate.mfa_verified:
                # Failures count in their own committed transactions, even when
                # the retyped amount or face check is wrong.
                self.rt.services.mandates.verify_mfa(
                    str(challenge["mandate_id"]),
                    customer_id,
                    otp=otp,
                    retyped_amount_fen=retyped_amount_fen,
                    face_ok=face_ok,
                    now=now,
                )
                mandate = self._mandate_for(challenge, customer_id)
            if bool(challenge["needs_co_approval"]) and mandate.co_approved_by is None:
                approver_id = challenge["approver_id"]
                if not approver_id:
                    raise SendNowError("no_co_approver", 403)
                if not isinstance(co_otp, str) or not co_otp.strip():
                    raise SendNowError("co_otp_missing", 403)
                self.rt.services.mandates.co_approve(
                    str(challenge["mandate_id"]),
                    customer_id,
                    str(approver_id),
                    co_otp,
                    now,
                )
        except MandateError as exc:
            raise _mandate_error(exc) from None

        return self._execute(customer_id, challenge, transfer, fingerprint, now)

    def _execute(
        self,
        customer_id: str,
        challenge: Any,
        transfer: dict[str, Any],
        fingerprint: str,
        now: datetime,
    ) -> dict[str, Any]:
        conn = self.conn
        if conn.in_transaction:
            raise SendNowError("busy", 400)
        conn.execute("BEGIN IMMEDIATE")
        proxy = _TxConnection(conn)
        try:
            tx_bank = Bank(proxy, customer_id)
            tx_mandates = MandateService(proxy, self.rt.services.secret_key)
            tx_audit = AuditLog(proxy, self.rt.services.secret_key)

            # Re-read the signed receipt and bank row inside the transaction
            # with plain SQL (no DDL) and require the identical fingerprint.
            _, args = self._receipt(
                proxy, customer_id, str(challenge["plan_id"]), str(challenge["step_id"]),
                direct=True,
            )
            current = self._transfer_for(proxy, customer_id, args)
            if (
                current["id"] != transfer["id"]
                or _fingerprint(
                    customer_id=customer_id,
                    session_id=str(challenge["session_id"]),
                    plan_id=str(challenge["plan_id"]),
                    step_id=str(challenge["step_id"]),
                    transfer=current,
                )
                != fingerprint
                or fingerprint != str(challenge["fingerprint"])
            ):
                raise SendNowError("mismatch", 403)

            decision = self._policy(
                tx_bank, customer_id, transfer["id"], now
            )
            if bool(getattr(decision, "needs_co_approval", False)) != bool(
                challenge["needs_co_approval"]
            ):
                raise SendNowError("mismatch", 403)

            try:
                tx_mandates.verify_and_consume(
                    str(challenge["mandate_id"]),
                    customer_id=customer_id,
                    session_id=str(challenge["session_id"]),
                    plan_id=str(challenge["mandate_plan_id"]),
                    step_id=RELEASE_STEP,
                    action=ACTION,
                    payee_ref=fingerprint,
                    amount_fen=transfer["amount_fen"],
                    now=now,
                )
                tx_bank.release_early(
                    transfer["id"],
                    now,
                    expected_contact_id=transfer["to_contact_id"],
                    expected_amount_fen=transfer["amount_fen"],
                )
            except MandateError as exc:
                raise _mandate_error(exc) from None
            except BankError as exc:
                raise _bank_error(exc) from None

            cursor = proxy.execute(
                "UPDATE exec_undo SET used = 1"
                " WHERE customer_id = ? AND plan_id = ? AND step_id = ? AND used = 0",
                (customer_id, str(challenge["plan_id"]), str(challenge["step_id"])),
            )
            if cursor.rowcount != 1:
                raise SendNowError("replayed", 403)

            # Signed, PII-free audit events; no OTP, name or account number.
            common = {"step_id": str(challenge["step_id"])}
            tx_audit.append(
                customer_id=customer_id,
                session_id=str(challenge["session_id"]),
                plan_id=str(challenge["plan_id"]),
                kind=AuditKind.MFA,
                payload={**common, "tool": ACTION, "ok": True},
                now=now,
            )
            tx_audit.append(
                customer_id=customer_id,
                session_id=str(challenge["session_id"]),
                plan_id=str(challenge["plan_id"]),
                kind=AuditKind.POLICY,
                payload={
                    **common,
                    "tool": ACTION,
                    "tier": "red",
                    "ok": True,
                    "needs_co_approval": bool(challenge["needs_co_approval"]),
                },
                now=now,
            )
            tx_audit.append(
                customer_id=customer_id,
                session_id=str(challenge["session_id"]),
                plan_id=str(challenge["plan_id"]),
                kind=AuditKind.RUN,
                payload={
                    **common,
                    "tool": "transfer",
                    "tier": "red",
                    "ok": True,
                    "amount_fen": transfer["amount_fen"],
                    "mandate_id": str(challenge["mandate_id"]),
                },
                now=now,
            )
            conn.commit()
        except BaseException as exc:
            conn.rollback()
            if isinstance(exc, SendNowError):
                raise
            if isinstance(exc, MandateError):
                raise _mandate_error(exc) from None
            if isinstance(exc, BankError):
                raise _bank_error(exc) from None
            if isinstance(exc, (sqlite3.Error, OverflowError, TypeError, ValueError)):
                raise SendNowError("internal", 400) from None
            raise

        result = {
            "status": "done",
            "reply": (
                f"Transfer confirmed: {fmt_fen(transfer['amount_fen'])}"
                f" to {transfer['payee_label']}. Sent now; Undo is no longer available."
            ),
            "ai_generated": True,
            "session_id": str(challenge["session_id"]),
            "plan_id": str(challenge["plan_id"]),
            "plan_view": [
                {
                    "step_id": str(challenge["step_id"]),
                    "tool": "transfer",
                    "status": "released",
                    "tier": "red",
                    "after": [],
                }
            ],
            "approvals": [],
            "actions": ["send_now_released"],
        }
        self._record_history(customer_id, result, str(challenge["mandate_id"]))
        return result

    def _record_history(
        self, customer_id: str, result: dict[str, Any], event_key: str
    ) -> None:
        try:
            from api.chat_history import history
            store = history(self.rt).store
            store.create(customer_id, self.rt.clock(), chat_id=result["session_id"])
            store.append(customer_id, result["session_id"], "assistant",
                         result["reply"], self.rt.clock(), turn=result,
                         event_key="send-now:" + event_key)
            self.conn.commit()
        except Exception:
            # History is a convenience; it must never undo a successful release.
            if self.conn.in_transaction:
                self.conn.rollback()
            return


def _mandate_error(exc: MandateError) -> SendNowError:
    if exc.code == "not_found":
        return SendNowError("not_found", 404)
    return SendNowError(exc.code, 403)


def _bank_error(exc: BankError) -> SendNowError:
    if exc.code == "not_found":
        return SendNowError("not_found", 404)
    if exc.code in {"invalid_amount", "invalid_state"}:
        return SendNowError(exc.code, 400)
    return SendNowError(exc.code, 403)


# --------------------------------------------------------------------- routes


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PrepareBody(_Strict):
    session_id: str = Field(min_length=1, max_length=128)
    plan_id: str = Field(min_length=1, max_length=128)
    step_id: str = Field(min_length=1, max_length=128)


class RequestCodeBody(_Strict):
    challenge_id: str = Field(min_length=16, max_length=64, pattern=r"^[0-9a-fA-F]+$")


class ConfirmBody(_Strict):
    challenge_id: str = Field(min_length=16, max_length=64, pattern=r"^[0-9a-fA-F]+$")
    otp: str = Field(pattern=r"^\d{6}$")
    retyped_amount_fen: StrictInt = Field(ge=1, le=MAX_AMOUNT_FEN)
    face_ok: StrictBool
    co_otp: str = Field(default="", pattern=r"^(?:\d{6})?$")


def register_send_now(app: Any, current_customer: Any, get_runtime: Any, demo_mode: bool) -> None:
    """Register the three send-now routes on the existing authenticated app."""

    def _run(call: Any, *args: Any) -> dict[str, Any]:
        try:
            return call(*args)
        except SendNowError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.code) from None

    @app.post("/send-now/prepare")
    def send_now_prepare(
        body: PrepareBody,
        customer_id: str = Depends(current_customer),
    ) -> dict[str, Any]:
        rt = get_runtime()
        with rt.lock:
            return _run(
                SendNow(rt).prepare,
                customer_id,
                body.session_id,
                body.plan_id,
                body.step_id,
            )

    @app.post("/send-now/request-code")
    def send_now_request_code(
        body: RequestCodeBody,
        customer_id: str = Depends(current_customer),
    ) -> dict[str, Any]:
        rt = get_runtime()
        with rt.lock:
            result = _run(SendNow(rt).request_code, customer_id, body.challenge_id)
        response = {"sent": True}
        if demo_mode:
            response["demo_otp"] = result["otp"]
            if result.get("co_otp"):
                response["demo_co_otp"] = result["co_otp"]
        return response

    @app.post("/send-now/confirm")
    def send_now_confirm(
        body: ConfirmBody,
        customer_id: str = Depends(current_customer),
    ) -> dict[str, Any]:
        rt = get_runtime()
        with rt.lock:
            return _run(
                SendNow(rt).confirm,
                customer_id,
                body.challenge_id,
                body.otp,
                body.retyped_amount_fen,
                body.face_ok,
                body.co_otp,
            )
