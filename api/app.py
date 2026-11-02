"""HTTP edge for AnBan: auth, chat turns, summaries and the demo controls.

This module is deliberately thin.  It owns process-level concerns (the SQLite
connection, one lock, tokens, rate limiting) and forwards every business
decision to :class:`~agent.graph.AnBanAgent` and the policy/mandate layers.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from agent.graph import AnBanAgent, Services, TurnResult
from agent.planner import make_fallback_planner_from_env, make_planner_from_env
from agent.prompts import prompt_for_profile
from agent.standing_order_hook import run_and_audit
from bank import db as bank_db
from bank import seed as bank_seed
from bank.money import fmt_fen
from bank.standing_orders import standing_order_overview
from security import compliance
from security.audit import AuditKind, AuditLog, create_audit_schema
from security.mandate import MandateService, create_security_schema
from security.policy import PolicyConfig
from security.tokens import TokenVault
from api.chat_history import history, register_chat_history
from api.scheduler import BackgroundTicker, tick
from api.send_now import register_send_now
from api.voice import MAX_AUDIO_BYTES, register_voice

MAX_BODY_BYTES = 16 * 1024
RATE_LIMIT = 30
READ_RATE_LIMIT = 180
_DASHBOARD_READS = frozenset({
    "/me/summary", "/me/transactions", "/me/settings", "/me/audit", "/compliance/status",
})
RATE_WINDOW_SECONDS = 60.0
TOKEN_TTL_SECONDS = 8 * 60 * 60
DEFAULT_SCHEDULER_INTERVAL_S = 60.0
_LOG = logging.getLogger(__name__)


def _scheduler_interval_s() -> float:
    """Seconds between background ticks; 0 or less disables the timer."""
    raw = os.environ.get("ANBAN_SCHEDULER_INTERVAL_S")
    if raw is None or not raw.strip():
        return DEFAULT_SCHEDULER_INTERVAL_S
    try:
        return float(raw)
    except ValueError:
        _LOG.warning("ANBAN_SCHEDULER_INTERVAL_S is not a number; using the default")
        return DEFAULT_SCHEDULER_INTERVAL_S


# ---------------------------------------------------------------- requests


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginBody(_Strict):
    persona: str = Field(min_length=1, max_length=80)


class ChatBody(_Strict):
    text: str = Field(min_length=1, max_length=500)
    session_id: str | None = Field(default=None, max_length=128)
    # Reply language from Settings; omitted -> detected from the text.
    lang: str | None = Field(default=None, pattern="^(en|zh)$")


class SettingsBody(_Strict):
    lang: str | None = None
    text_size: str | None = None
    elder_mode: bool | None = None
    arrival_delay: str | None = None
    guardian: str | None = Field(default=None, max_length=40)


class ApproveBody(_Strict):
    approvals: dict[str, Any]


class UndoBody(_Strict):
    session_id: str = Field(min_length=1, max_length=128)
    plan_id: str = Field(min_length=1, max_length=128)
    step_id: str = Field(min_length=1, max_length=128)


class RecallBody(UndoBody):
    lang: str | None = Field(default=None, pattern="^(en|zh)$")


class ClockBody(_Strict):
    # Forward only and bounded: rewinding would revive expired tokens and
    # mandates, and a huge value overflows datetime.
    advance_minutes: int = Field(ge=0, le=7 * 24 * 60)


# ------------------------------------------------------------------- clock


@dataclass
class DemoClock:
    """Wall clock plus a mutable offset used by ``POST /demo/clock``."""

    offset: timedelta = field(default_factory=timedelta)

    def __call__(self) -> datetime:
        # This is the edge of the system; core code always receives ``now``.
        return datetime.now() + self.offset

    def advance(self, minutes: int) -> None:
        self.offset += timedelta(minutes=minutes)


# ----------------------------------------------------------------- runtime


@dataclass
class _Runtime:
    conn: sqlite3.Connection
    services: Services
    agent: AnBanAgent
    states: dict[str, Any]
    customer_ids: frozenset[str]
    clock: Callable[[], datetime]
    lock: threading.Lock = field(default_factory=threading.Lock)


def _open_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _close_quietly(runtime: _Runtime) -> None:
    try:
        runtime.conn.close()
    except sqlite3.Error:
        pass


def _upcoming_reminders(
    conn: sqlite3.Connection, customer_id: str, today: date, limit: int = 10
) -> list[dict[str, Any]]:
    """Read upcoming and recently due reminders, if the optional table exists."""

    try:
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'agent_reminders'"
        ).fetchone()
        if table is None:
            return []
        # Keep reminders that just came due so they still surface once.
        since = (today - timedelta(days=7)).isoformat()
        rows = conn.execute(
            """SELECT due_on, note FROM agent_reminders
               WHERE customer_id = ? AND due_on IS NOT NULL AND due_on >= ?
               ORDER BY due_on LIMIT ?""",
            (customer_id, since, limit),
        ).fetchall()
    except sqlite3.Error:
        return []
    return [{"due_on": row["due_on"], "note": row["note"]} for row in rows]


def _masked_payee(conn: sqlite3.Connection, customer_id: str, contact_id: str) -> str:
    """Display name plus the last four account digits; never the raw id."""
    row = conn.execute(
        "SELECT name, account_no FROM contacts WHERE customer_id = ? AND id = ?",
        (customer_id, contact_id),
    ).fetchone()
    if row is None:
        return "Unknown"
    return f"{row['name']} ···{str(row['account_no'])[-4:]}"


def _conversation_memory(tokenizer_path: str) -> Any:
    """Build bounded conversation memory from an explicit tokenizer.json.

    Only called when ``ANBAN_TOKENIZER_PATH`` is set.  Bad configuration fails
    loudly with a message that names the variable, never its value.
    """

    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise ValueError(
            "ANBAN_TOKENIZER_PATH requires the optional 'tokenizers' package"
        ) from None
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except Exception:
        raise ValueError(
            "ANBAN_TOKENIZER_PATH is not a readable tokenizer.json file"
        ) from None

    from agent.conversation_memory import BoundedConversationMemory

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return BoundedConversationMemory(
        count_tokens,
        context_tokens=8192,
        history_tokens=1536,
        output_reserve=512,
        max_turns=24,
    )


def _build_runtime(
    *,
    db_path: str,
    secret_key: bytes,
    planner: Any,
    clock: Callable[[], datetime],
) -> _Runtime:
    conn = _open_connection(db_path)
    bank_db.create_schema(conn)

    states = bank_seed.generate_all(42, clock().date())
    already_loaded = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    if not already_loaded:
        for state in states.values():
            bank_db.load_state(conn, state)

    create_security_schema(conn)
    create_audit_schema(conn)

    # Conversation replies are opt-in and use the same planner instance as the
    # bank path: no fallback planner, no extra cloud call.
    conversation_gateway = None
    if os.environ.get("ANBAN_CONVERSATION") == "1":
        from agent.conversation import ConversationGateway

        conversation_gateway = ConversationGateway(planner)
    # Bounded history needs a real tokenizer, so it stays off unless a
    # tokenizer.json path is given explicitly.
    conversation_memory = None
    tokenizer_path = os.environ.get("ANBAN_TOKENIZER_PATH")
    if tokenizer_path:
        conversation_memory = _conversation_memory(tokenizer_path)

    services = Services(
        conn=conn,
        vault=TokenVault(secret_key),
        mandates=MandateService(conn, secret_key),
        audit=AuditLog(conn, secret_key),
        planner=planner,
        policy_config=PolicyConfig(),
        secret_key=secret_key,
        now=clock,
        planner_system_prompt=prompt_for_profile(os.environ.get("ANBAN_PLANNER_PROFILE", "r2")),
        fallback_planner=make_fallback_planner_from_env(),
        conversation_gateway=conversation_gateway,
        conversation_memory=conversation_memory,
    )
    return _Runtime(
        conn=conn,
        services=services,
        agent=AnBanAgent(services),
        states=states,
        customer_ids=frozenset(state.customer.id for state in states.values()),
        clock=clock,
    )


# English display names for the demo merchants (UI only).
_MERCHANT_EN = {
    "美团": "Meituan",
    "饿了么": "Ele.me",
    "盒马": "Hema",
    "京东": "JD.com",
    "淘宝": "Taobao",
    "拼多多": "Pinduoduo",
    "滴滴出行": "DiDi",
    "我的储蓄账户": "To my savings",
    "现金收款": "Cash received",
    "工资": "Salary",
    "百度网盘": "Baidu Netdisk",
    "爱奇艺 VIP": "iQIYI VIP",
}


# --------------------------------------------------------------- app factory


def create_app(
    *,
    db_path: str = ":memory:",
    secret_key: bytes | None = None,
    planner: Any = None,
    clock: Callable[[], datetime] | None = None,
    demo_mode: bool | None = None,
    allowed_origins: tuple[str, ...] = ("http://localhost:5173",),
) -> FastAPI:
    """Build the FastAPI application and its single database connection."""

    if demo_mode is None:
        # Demo routes (open login, clock, reset) are opt-in: ANBAN_DEMO=1.
        demo_mode = os.environ.get("ANBAN_DEMO") == "1"
    if secret_key is None:
        env_secret = os.environ.get("ANBAN_SECRET")
        secret_key = (
            bytes.fromhex(env_secret) if env_secret else secrets.token_bytes(32)
        )
    if planner is None:
        planner = make_planner_from_env()
    if clock is None:
        clock = DemoClock()

    runtime = _build_runtime(
        db_path=db_path, secret_key=secret_key, planner=planner, clock=clock
    )

    def get_runtime() -> _Runtime:
        return app.state.runtime

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        interval_s = _scheduler_interval_s()
        scheduler = (
            BackgroundTicker(get_runtime, interval_s) if interval_s > 0 else None
        )
        _app.state.scheduler = scheduler
        if scheduler is not None:
            # Catch up on anything that fell due while the process was down.
            try:
                tick(get_runtime())
            except Exception:
                _LOG.warning("Scheduler catch-up tick failed", exc_info=True)
            scheduler.start()
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.stop()

    app = FastAPI(title="AnBan", version="0.1", lifespan=lifespan)
    app.state.runtime = runtime
    app.state.secret_key = secret_key
    app.state.db_path = db_path
    app.state.scheduler = None

    rate_lock = threading.Lock()
    rate_buckets: dict[str, deque[float]] = {}

    # ------------------------------------------------------------ middleware

    @app.middleware("http")
    async def harden(request: Request, call_next):
        limit = MAX_AUDIO_BYTES if request.url.path == "/voice/transcribe" else MAX_BODY_BYTES
        declared = request.headers.get("content-length")
        if declared:
            try:
                if int(declared) > limit:
                    return JSONResponse(
                        {"error": "payload_too_large"}, status_code=413
                    )
            except ValueError:
                return JSONResponse({"error": "bad_request"}, status_code=400)

        chunks = bytearray()
        async for chunk in request.stream():
            if len(chunks) + len(chunk) > limit:
                return JSONResponse({"error": "payload_too_large"}, status_code=413)
            chunks.extend(chunk)
        request._body = bytes(chunks)

        authorization = request.headers.get("authorization", "")
        token = (
            authorization[7:].strip()
            if authorization.lower().startswith("bearer ")
            else ""
        )
        customer = verify_token(token) if token else None
        request_limit = RATE_LIMIT
        if customer is not None:
            # Dashboard refreshes cannot exhaust the action or undo budget.
            # Verify first: rotating arbitrary bearer strings creates no quota.
            scope = "action"
            if request.method == "GET" and request.url.path in _DASHBOARD_READS:
                scope, request_limit = "read", READ_RATE_LIMIT
            elif request.method == "POST" and request.url.path == "/undo":
                scope = "undo"
            key = f"customer:{customer}:{scope}"
        else:
            host = request.client.host if request.client is not None else "unknown"
            key = f"ip:{host}"

        now = time.monotonic()
        with rate_lock:
            bucket = rate_buckets.setdefault(key, deque())
            while bucket and bucket[0] <= now - RATE_WINDOW_SECONDS:
                bucket.popleft()
            if len(bucket) >= request_limit:
                return JSONResponse(
                    {"error": "rate_limited"}, status_code=429
                )
            bucket.append(now)

        return await call_next(request)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        # Never expose a traceback, a token or user text.
        return JSONResponse({"error": "internal"}, status_code=500)

    # ------------------------------------------------------------------ auth

    def issue_token(customer_id: str) -> str:
        rt = get_runtime()
        expiry = int(rt.clock().timestamp()) + TOKEN_TTL_SECONDS
        message = f"{customer_id}.{expiry}".encode("utf-8")
        signature = hmac.new(secret_key, message, hashlib.sha256).hexdigest()
        return f"{customer_id}.{expiry}.{signature}"

    def verify_token(token: str) -> str | None:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        customer_id, expiry_text, signature = parts
        if len(signature) != 64 or any(char not in "0123456789abcdef" for char in signature):
            return None
        try:
            expiry = int(expiry_text)
        except ValueError:
            return None
        rt = get_runtime()
        if expiry <= int(rt.clock().timestamp()):
            return None
        message = f"{customer_id}.{expiry_text}".encode("utf-8")
        expected = hmac.new(secret_key, message, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            return None
        if customer_id not in rt.customer_ids:
            return None
        return customer_id

    def current_customer(
        authorization: str | None = Header(default=None),
    ) -> str:
        if not authorization:
            raise HTTPException(status_code=401, detail="unauthorized")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(status_code=401, detail="unauthorized")
        customer_id = verify_token(token.strip())
        if customer_id is None:
            raise HTTPException(status_code=401, detail="unauthorized")
        return customer_id

    def as_dict(result: TurnResult) -> dict[str, Any]:
        # 《人工智能生成合成内容标识办法》: replies are marked as AI-generated so the
        # client shows an explicit label on every assistant message.
        body = {**asdict(result), "ai_generated": True}
        if not demo_mode:
            # One-time codes go to the customer's phone / the guardian, never
            # back to whoever holds the session.
            for approval in body.get("approvals") or ():
                approval.pop("demo_otp", None)
                approval.pop("demo_co_otp", None)
        return body

    def restore_turn(customer_id: str, session_id: str) -> None:
        # An owned chat is restored; a cross-customer or unknown owner is 404.
        try:
            history(get_runtime()).before_turn(customer_id, session_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="not_found") from None

    # --------------------------------------------------------------- routes

    register_voice(app, current_customer)
    register_chat_history(app, current_customer, get_runtime, as_dict)
    register_send_now(app, current_customer, get_runtime, demo_mode)

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    if demo_mode:

        @app.post("/auth/demo-login")
        def demo_login(body: LoginBody) -> dict[str, str]:
            rt = get_runtime()
            persona = body.persona
            if persona not in rt.states:
                raise HTTPException(status_code=404, detail="unknown persona")
            state = rt.states[persona]
            customer_id = state.customer.id
            return {
                "token": issue_token(customer_id),
                "customer_id": customer_id,
                "display_name": state.customer.name,
            }

        @app.post("/demo/clock")
        def demo_clock(body: ClockBody) -> dict[str, str]:
            rt = get_runtime()
            advance = getattr(rt.clock, "advance", None)
            if not callable(advance):
                raise HTTPException(status_code=400, detail="clock_not_adjustable")
            advance(body.advance_minutes)
            return {"now": rt.clock().isoformat()}

        @app.get("/compliance/status")
        def compliance_status(customer_id: str = Depends(current_customer)) -> dict[str, Any]:
            """Jury view: regulator-side records (all SIMULATED).

            Demo-only: a real customer must never see STR/LTR records about
            themselves (no tipping off).
            """
            rt = get_runtime()
            with rt.lock:
                return {
                    "reports": compliance.reports(rt.conn, customer_id),
                    "fraud_list_size": len(compliance.fraud_accounts()),
                    "large_transfer_threshold_fen": compliance.LARGE_TRANSFER_FEN,
                    "settings": compliance.get_settings(rt.conn, customer_id).as_dict(),
                }


        @app.post("/demo/reset")
        def demo_reset() -> dict[str, bool]:
            old = get_runtime()
            with old.lock:
                _close_quietly(old)
            if db_path != ":memory:" and os.path.exists(db_path):
                os.remove(db_path)
            reset_clock = getattr(old.clock, "offset", None)
            if reset_clock is not None and hasattr(old.clock, "advance"):
                old.clock.offset = timedelta()
            app.state.runtime = _build_runtime(
                db_path=db_path, secret_key=secret_key, planner=planner, clock=clock
            )
            return {"ok": True}

    @app.post("/chat")
    def chat(
        body: ChatBody,
        customer_id: str = Depends(current_customer),
    ) -> dict[str, Any]:
        rt = get_runtime()
        with rt.lock:
            if body.session_id:
                restore_turn(customer_id, body.session_id)
            controller = history(rt)
            # No language sent: use the one saved in Settings, else auto-detect.
            lang = body.lang or compliance.saved_lang(
                rt.services.bank_for(customer_id).conn, customer_id
            )
            result = rt.agent.start(customer_id, body.text, body.session_id, lang=lang)
            output = as_dict(result)
            controller.record(customer_id, result, output, user_text=body.text)
        return output

    @app.post("/chat/{session_id}/approve")
    def approve(
        session_id: str,
        body: ApproveBody,
        customer_id: str = Depends(current_customer),
    ) -> dict[str, Any]:
        rt = get_runtime()
        with rt.lock:
            restore_turn(customer_id, session_id)
            controller = history(rt)
            result = rt.agent.resume(session_id, customer_id, body.approvals)
            output = as_dict(result)
            controller.record(customer_id, result, output)
        return output

    @app.post("/chat/{session_id}/pause")
    def pause(
        session_id: str,
        customer_id: str = Depends(current_customer),
    ) -> dict[str, Any]:
        rt = get_runtime()
        with rt.lock:
            restore_turn(customer_id, session_id)
            controller = history(rt)
            result = rt.agent.pause(session_id, customer_id)
            output = as_dict(result)
            controller.record(customer_id, result, output)
        return output

    @app.post("/chat/{session_id}/handoff")
    def handoff(
        session_id: str,
        customer_id: str = Depends(current_customer),
    ) -> dict[str, Any]:
        rt = get_runtime()
        with rt.lock:
            restore_turn(customer_id, session_id)
            controller = history(rt)
            result = rt.agent.handoff(session_id, customer_id)
            output = as_dict(result)
            controller.record(customer_id, result, output)
        return output

    @app.post("/undo")
    def undo(
        body: UndoBody,
        customer_id: str = Depends(current_customer),
    ) -> dict[str, Any]:
        rt = get_runtime()
        with rt.lock:
            restore_turn(customer_id, body.session_id)
            controller = history(rt)
            result = rt.agent.undo(
                body.session_id, customer_id, body.plan_id, body.step_id
            )
            output = as_dict(result)
            controller.record(customer_id, result, output)
        return output

    @app.post("/recall")
    def recall(
        body: RecallBody,
        customer_id: str = Depends(current_customer),
    ) -> dict[str, Any]:
        # SIMULATED 退汇 request for a transfer whose undo window has closed.
        rt = get_runtime()
        with rt.lock:
            restore_turn(customer_id, body.session_id)
            controller = history(rt)
            result = rt.agent.recall(
                body.session_id, customer_id, body.plan_id, body.step_id, lang=body.lang
            )
            output = as_dict(result)
            controller.record(customer_id, result, output)
        return output

    @app.get("/me/summary")
    def summary(customer_id: str = Depends(current_customer)) -> dict[str, Any]:
        rt = get_runtime()
        now = rt.clock()
        today = now.date()
        with rt.lock:
            run_and_audit(rt.conn, rt.services.audit, customer_id, now)
            bank = rt.services.bank_for(customer_id)
            balance = bank.get_balance()
            cards = bank.card_details(now)
            subscriptions = bank.list_subscriptions()
            anomalies = bank.detect_anomalies(now)
            reminders = _upcoming_reminders(rt.conn, customer_id, today)
            standing_orders = [
                {
                    **{key: value for key, value in order.items() if key != "contact_id"},
                    "payee": _masked_payee(
                        rt.conn, customer_id, str(order["contact_id"])
                    ),
                }
                for order in standing_order_overview(rt.conn, customer_id, now)
            ]

        return {
            "balance_fen": balance,
            "display": fmt_fen(balance),
            "cards": cards,
            "subscriptions": [
                {
                    "merchant": sub.merchant,
                    "amount_fen": sub.amount_fen,
                    "unused_60d": (today - sub.last_used_at).days >= 60,
                }
                for sub in subscriptions
                if getattr(sub, "status", "active") == "active"
            ],
            "anomalies": len(anomalies),
            "reminders": reminders,
            "standing_orders": standing_orders,
            "now": now.isoformat(),
        }

    @app.get("/me/transactions")
    def transactions(
        limit: int = 10,
        customer_id: str = Depends(current_customer),
    ) -> dict[str, Any]:
        rt = get_runtime()
        with rt.lock:
            bank = rt.services.bank_for(customer_id)
            bank.release_due(rt.clock())  # past their hold: into the ledger
            rows = bank.list_transactions(limit=max(1, min(int(limit), 50)))
            # Transfers still on hold are not in the ledger yet, but the money
            # has left the balance: show them first, with the undo deadline.
            held = bank.conn.execute(
                """
                SELECT t.id, t.to_contact_id, t.amount_fen, t.created_at, t.release_at, c.name
                FROM transfers t
                JOIN contacts c ON c.customer_id = t.customer_id AND c.id = t.to_contact_id
                WHERE t.customer_id = ? AND t.status = 'pending'
                ORDER BY t.created_at DESC
                """,
                (customer_id,),
            ).fetchall()
            holds = {
                row["id"]: compliance.transfer_release_hold(
                    rt.conn, customer_id, row["to_contact_id"], rt.clock()
                )
                for row in held
            }
        return {
            "transactions": [
                {
                    "id": row["id"],
                    "ts": row["created_at"],
                    "amount_fen": -int(row["amount_fen"]),
                    "display": fmt_fen(-int(row["amount_fen"])),
                    "counterparty": row["name"],
                    "counterparty_en": row["name"],
                    "category": "on_hold",
                    "held_until": row["release_at"],
                    "hold_reason": holds[row["id"]][0] if holds[row["id"]] else None,
                }
                for row in held
            ]
            + [
                {
                    "id": row.id,
                    "ts": row.ts.isoformat(),
                    "amount_fen": row.amount_fen,
                    "display": fmt_fen(row.amount_fen),
                    "counterparty": row.counterparty,
                    "counterparty_en": _MERCHANT_EN.get(row.counterparty, row.counterparty),
                    "category": row.category,
                }
                for row in rows
            ]
        }

    @app.get("/me/settings")
    def get_settings(customer_id: str = Depends(current_customer)) -> dict[str, Any]:
        rt = get_runtime()
        with rt.lock:
            compliance.apply_due(rt.conn, customer_id, rt.clock())
            return {
                **compliance.get_settings(rt.conn, customer_id).as_dict(),
                "pending": compliance.pending_changes(rt.conn, customer_id),
            }

    @app.put("/me/settings")
    def put_settings(
        body: SettingsBody, customer_id: str = Depends(current_customer)
    ) -> dict[str, Any]:
        changes = {key: value for key, value in body.model_dump().items() if value is not None}
        rt = get_runtime()
        with rt.lock:
            now = rt.clock()
            try:
                saved = compliance.update_settings(rt.conn, customer_id, changes, now=now)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from None
            pending = compliance.pending_changes(rt.conn, customer_id)
            # Which keys changed, never the guardian's name (audit stays PII-free).
            rt.services.audit.append(
                customer_id=customer_id,
                session_id="settings",
                plan_id="settings",
                kind=AuditKind.SETTINGS,
                payload={
                    "keys": sorted(changes),
                    "queued": sorted(item["key"] for item in pending if item["key"] in changes),
                },
                now=now,
            )
            return {**saved.as_dict(), "pending": pending}

    @app.get("/me/audit")
    def audit(
        plan_id: str | None = None,
        lang: str = "en",
        customer_id: str = Depends(current_customer),
    ) -> dict[str, Any]:
        if lang not in {"en", "zh"}:
            raise HTTPException(status_code=422, detail="lang must be en or zh")
        rt = get_runtime()
        with rt.lock:
            log = rt.services.audit
            if plan_id:
                lines = log.explain(customer_id, plan_id, lang)
            else:
                # Order plans by when they first appear in the chain, not by
                # plan id: ids are strings, so "…:p10" sorts before "…:p9".
                first_seen: dict[str, object] = {}
                for entry in log.entries_for(customer_id):
                    seq = getattr(entry, "seq", None)
                    first_seen.setdefault(
                        entry.plan_id, seq if isinstance(seq, int) else entry.ts
                    )
                lines = [
                    line
                    for current in sorted(first_seen, key=lambda p: first_seen[p])
                    for line in log.explain(customer_id, current, lang)
                ]
            chain_ok = log.verify().ok
        return {"lines": lines, "chain_ok": chain_ok}

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


def main() -> None:
    """Run the demo server; the app is created lazily so imports stay cheap."""

    import uvicorn

    uvicorn.run("api.app:create_app", factory=True, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
