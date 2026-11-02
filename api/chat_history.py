"""Durable, customer-scoped chat history for the HTTP edge.

The controller is deliberately thin: the store owns persistence, the agent owns
sessions.  This module only decides which chat may be restored, what may be
shown again, and how a finished turn is written down.  Old approvals, codes,
tokens and LangGraph interrupts are never brought back; only the masked memory
pairs are replayed into the bounded conversation memory.
"""

from __future__ import annotations

import re
import hmac
import json
from collections.abc import Callable
from datetime import date, datetime, time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from api.chat_store import ChatStore
from agent.executor import _undo_sig
from security.mandate import OPEN_STATUSES

NEW_CHAT_TITLE = "New chat"
RECONFIRM_REPLY = (
    "This approval is no longer active. Please reconfirm by sending the request "
    "again; old approvals and codes cannot be reused."
)

#: A memory-only bridge row: it gives ``ChatStore.memory_pairs`` the masked user
#: half of an approved turn whose user message was already stored earlier.  It
#: is never returned by ``GET /chats/{id}/messages``.
_BRIDGE_TEXT = ""

_FORBIDDEN_TURN_KEYS = frozenset(
    {"demo_otp", "demo_co_otp", "otp", "co_otp", "challenge_code"}
)


def _iso(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return datetime.combine(value, time.min).isoformat()
    if isinstance(value, str) and value:
        return value
    raise TypeError("now must be a date, datetime or ISO string")


def _title(user_text: str | None) -> str:
    """Use the first user text as the new chat's display title, bounded."""
    if user_text is None:
        return NEW_CHAT_TITLE
    cleaned = " ".join(str(user_text).split())[:60].strip()
    return cleaned or NEW_CHAT_TITLE


def _scrub(value: Any) -> Any:
    """Recursively drop verification-code keys from a stored display turn."""
    if isinstance(value, dict):
        return {
            key: _scrub(item)
            for key, item in value.items()
            if str(key).lower() not in _FORBIDDEN_TURN_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_scrub(item) for item in value]
    return value


def _summary(result: Any) -> str:
    """Code-built, fact-free memory note: status plus validated tool names."""
    status = str(getattr(result, "status", "") or "")
    tools = {
        str(item.get("tool"))
        for item in (getattr(result, "plan_view", ()) or ())
        if isinstance(item, dict) and str(item.get("tool") or "").strip()
    }
    summary = "status=" + status
    if tools:
        summary += " tools=" + ",".join(sorted(tools))
    return summary


def _needs_bridge(message: Any) -> bool:
    """True when the newest row is an unresolved approval card."""
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return False
    turn = message.get("turn")
    return isinstance(turn, dict) and turn.get("status") == "needs_approval"


class ChatHistory:
    """Owns the durable store and the safe restore rules for one runtime."""

    def __init__(self, rt: Any) -> None:
        self.rt = rt
        self.conn = rt.conn
        self.store = ChatStore(rt.conn)
        # CREATE IF NOT EXISTS only: history is never dropped or migrated.
        self.store.create_schema()
        self.conn.commit()
        # Set by register_chat_history; a fallback for callers without a body.
        self.as_dict: Callable[[Any], dict[str, Any]] | None = None

    # ------------------------------------------------------------------ reads

    def create(self, customer_id: str, title: str | None = None) -> dict[str, str]:
        chat = self.store.create(
            customer_id,
            self.rt.clock(),
            (title or "").strip() or NEW_CHAT_TITLE,
        )
        self.conn.commit()
        return chat

    def list_chats(
        self, customer_id: str, query: str = "", limit: int = 100
    ) -> list[dict[str, str]]:
        return self.store.list(customer_id, query or "", limit)

    def messages(
        self,
        customer_id: str,
        chat_id: str,
        before: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        page = self.store.messages(
            customer_id, chat_id, before=before or None, limit=limit
        )
        shown: list[dict[str, Any]] = []
        for message in page.get("messages", ()):
            if message.get("role") == "user" and message.get("text") == _BRIDGE_TEXT:
                continue
            message = dict(message)
            message["turn"] = self._display_turn(
                customer_id, chat_id, message.get("turn")
            )
            shown.append(message)
        result = dict(page)
        result["messages"] = shown
        result["oldest_id"] = page.get("oldest_id")
        return result

    def _display_turn(
        self, customer_id: str, chat_id: str, turn: Any
    ) -> Any:
        if not isinstance(turn, dict):
            return turn
        clean = _scrub(turn)
        clean["approvals"] = []
        for step in clean.get("plan_view") or []:
            if step.get("tool") != "transfer" or step.get("status") != "done":
                continue
            row = self.conn.execute(
                "SELECT * FROM exec_undo WHERE customer_id = ? AND plan_id = ? AND step_id = ?",
                (customer_id, clean.get("plan_id"), step.get("step_id")),
            ).fetchone()
            if row is None or row["tool"] != "cancel_transfer":
                step["status"] = "unavailable"
                continue
            expected = _undo_sig(self.rt.services.secret_key, customer_id, row["plan_id"], row["step_id"], row["tool"], row["args_json"], row["created_at"])
            if not hmac.compare_digest(expected.encode(), str(row["sig"]).encode()):
                step["status"] = "unavailable"
                continue
            args = json.loads(row["args_json"])
            transfer = self.conn.execute(
                "SELECT status FROM transfers WHERE customer_id = ? AND id = ?",
                (customer_id, args.get("transfer_id") or args.get("id")),
            ).fetchone()
            if transfer is not None and transfer[0] in {"released", "cancelled"}:
                step["status"] = transfer[0]
        if clean.get("status") == "needs_approval":
            clean["status"] = "clarify"
            clean["reply"] = RECONFIRM_REPLY
        return clean

    def _active_plan(self, customer_id: str, chat_id: str) -> str:
        """The current pending plan, only while its mandates are still open."""
        record = self.rt.services.sessions.get(chat_id)
        if not isinstance(record, dict) or record.get("customer_id") != customer_id:
            return ""
        plan_id = str(record.get("pending_plan_id") or "")
        if not plan_id:
            return ""
        placeholders = ", ".join("?" for _ in OPEN_STATUSES)
        row = self.conn.execute(
            f"SELECT 1 FROM security_mandates"
            f" WHERE customer_id = ? AND plan_id = ?"
            f" AND status IN ({placeholders}) AND expires_at > ? LIMIT 1",
            (customer_id, plan_id, *OPEN_STATUSES, _iso(self.rt.clock())),
        ).fetchone()
        return plan_id if row is not None else ""

    # ---------------------------------------------------------------- restore

    def before_turn(self, customer_id: str, session_id: str) -> dict[str, Any]:
        """Restore one owned chat before a turn, or reject cross-customer use.

        Only the durable masked pairs and the audit-derived plan counter are
        replayed.  No checkpoint, OTP, approval, vault entry or clarify state
        is recreated here.
        """
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("not_found")

        # Database ownership is checked before anything is restored.
        owner = self.conn.execute(
            "SELECT customer_id FROM chat_threads WHERE id = ?", (session_id,)
        ).fetchone()
        if owner is not None:
            if str(owner[0]) != customer_id:
                raise ValueError("not_found")
            self.store.require(customer_id, session_id)

        record = self.rt.services.sessions.get(session_id)
        if record is not None:
            if record.get("customer_id") != customer_id:
                raise ValueError("not_found")
            self._restore_risk_texts(customer_id)
            return record

        if self.conn.execute(
            "SELECT 1 FROM security_audit WHERE session_id = ? AND customer_id != ? LIMIT 1",
            (session_id, customer_id),
        ).fetchone():
            raise ValueError("not_found")

        # A new legacy session id is allowed only when no chat owns it.
        verified = self.rt.services.audit.verify()
        if not getattr(verified, "ok", False):
            raise ValueError("not_found")

        if owner is None:
            self.store.create(customer_id, self.rt.clock(), chat_id=session_id)
            self.conn.commit()

        record = self.rt.agent._new_session(session_id, customer_id)
        plans, plan_seq = self._audited_plans(customer_id, session_id)
        record["plan_seq"] = plan_seq

        now = self.rt.clock()
        for plan_id in plans:
            self.rt.services.mandates.revoke_plan(customer_id, plan_id, now)

        self._restore_memory(customer_id, session_id)
        self._restore_risk_texts(customer_id)
        return record

    def _audited_plans(
        self, customer_id: str, session_id: str
    ) -> tuple[set[str], int]:
        pattern = re.compile(re.escape(session_id) + r":p(\d+)\Z")
        plans: set[str] = set()
        highest = 0
        for entry in self.rt.services.audit.entries_for(customer_id):
            match = pattern.fullmatch(str(entry.plan_id))
            if match is None:
                continue
            plans.add(str(entry.plan_id))
            highest = max(highest, int(match.group(1)))
        return plans, highest

    def _restore_memory(self, customer_id: str, session_id: str) -> None:
        memory = self.rt.services.conversation_memory
        if memory is None:
            return
        pairs = self.store.memory_pairs(customer_id, session_id, limit=24)
        if not pairs:
            return
        clear = getattr(memory, "clear", None)
        if callable(clear):
            clear(customer_id, session_id)
        for pair in pairs:
            user = str(pair.get("user") or "")
            assistant = str(pair.get("assistant") or "")
            if user and assistant:
                memory.remember(customer_id, session_id, user, assistant)

    def _restore_risk_texts(self, customer_id: str) -> None:
        """Code-only scam evidence: raw text is never rendered to the model."""
        texts = [
            text
            for text in self.store.recent_user_texts(customer_id, limit=5)
            if text
        ]
        self.rt.services.customer_risk_texts[customer_id] = texts[-5:]

    # ----------------------------------------------------------------- write

    def record(
        self,
        customer_id: str,
        result: Any,
        bodydict: dict[str, Any] | None = None,
        user_text: str | None = None,
    ) -> None:
        """Append one turn; persist a memory pair only when it is final."""
        session_id = str(getattr(result, "session_id", "") or "")
        now = self.rt.clock()
        title = _title(user_text)

        if session_id:
            owner = self.conn.execute(
                "SELECT customer_id FROM chat_threads WHERE id = ?",
                (session_id,),
            ).fetchone()
            if owner is not None and str(owner[0]) != customer_id:
                return
            if owner is None:
                self.store.create(customer_id, now, title, chat_id=session_id)
        else:
            session_id = self.store.create(customer_id, now, title)["id"]

        status = str(getattr(result, "status", "") or "")
        final = status != "needs_approval"
        tokenized = self._tokenized(session_id)
        last = self._last_message(customer_id, session_id)
        body = self._body(result, bodydict)

        if user_text is not None:
            self.conn.execute(
                "UPDATE chat_threads SET title = ? WHERE id = ? AND customer_id = ? "
                "AND title = ? AND NOT EXISTS (SELECT 1 FROM chat_messages WHERE chat_id = ? AND role = 'user')",
                (title, session_id, customer_id, NEW_CHAT_TITLE, session_id),
            )
            self.store.append(
                customer_id,
                session_id,
                "user",
                str(user_text),
                now,
                masked_user=tokenized if final else "",
            )
        elif final and _needs_bridge(last) and tokenized:
            # The user half of an approved turn was stored before the code was
            # entered; this hidden row completes the pair without a duplicate
            # visible message.
            self.store.append(
                customer_id,
                session_id,
                "user",
                _BRIDGE_TEXT,
                now,
                masked_user=tokenized,
            )

        reply = str(getattr(result, "reply", "") or "")
        if final:
            self.store.append(
                customer_id,
                session_id,
                "assistant",
                reply,
                now,
                turn=body,
                memory_summary=_summary(result),
            )
        else:
            self.store.append(
                customer_id, session_id, "assistant", reply, now, turn=body
            )
        self.conn.commit()

    def _body(
        self, result: Any, bodydict: dict[str, Any] | None
    ) -> dict[str, Any]:
        if isinstance(bodydict, dict):
            return bodydict
        converter = self.as_dict
        if callable(converter):
            try:
                converted = converter(result)
            except Exception:  # noqa: BLE001 - history must not break a turn
                converted = None
            if isinstance(converted, dict):
                return converted
        return {}

    def _last_message(self, customer_id: str, chat_id: str) -> dict[str, Any] | None:
        page = self.store.messages(customer_id, chat_id, limit=1)
        messages = page.get("messages") or ()
        return messages[-1] if messages else None

    def _tokenized(self, session_id: str) -> str:
        """Only the graph's tokenized user text; never ``user_msg`` or raw text."""
        try:
            snapshot = self.rt.agent.graph.get_state(
                {"configurable": {"thread_id": session_id}}
            )
        except Exception:  # noqa: BLE001 - no checkpoint means no masked text
            return ""
        values = getattr(snapshot, "values", None)
        if not isinstance(values, dict):
            return ""
        tokenized = values.get("tokenized_text")
        return tokenized if isinstance(tokenized, str) else ""


def history(rt: Any) -> ChatHistory:
    """Return the lazy singleton controller for this runtime."""
    controller = getattr(rt, "chat_history", None)
    if controller is None:
        controller = ChatHistory(rt)
        rt.chat_history = controller
    return controller


# --------------------------------------------------------------------- routes


class _ChatBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=80)


def register_chat_history(
    app: FastAPI,
    current_customer: Callable[..., str],
    get_runtime: Callable[[], Any],
    as_dict: Callable[[Any], dict[str, Any]],
) -> None:
    """Attach the three authenticated history routes to an existing app."""

    def controller(rt: Any) -> ChatHistory:
        result = history(rt)
        result.as_dict = as_dict
        return result

    @app.post("/chats")
    def create_chat(
        body: _ChatBody, customer_id: str = Depends(current_customer)
    ) -> dict[str, Any]:
        rt = get_runtime()
        with rt.lock:
            return {"chat": controller(rt).create(customer_id, body.title)}

    @app.get("/chats")
    def list_chats(
        query: str = "",
        limit: int = 100,
        customer_id: str = Depends(current_customer),
    ) -> dict[str, Any]:
        rt = get_runtime()
        with rt.lock:
            return {"chats": controller(rt).list_chats(customer_id, query, limit)}

    @app.get("/chats/{chat_id}/messages")
    def chat_messages(
        chat_id: str,
        before: str | None = None,
        limit: int = 100,
        customer_id: str = Depends(current_customer),
    ) -> dict[str, Any]:
        rt = get_runtime()
        with rt.lock:
            try:
                return controller(rt).messages(
                    customer_id, chat_id, before=before, limit=limit
                )
            except ValueError as exc:
                if str(exc) == "not_found":
                    raise HTTPException(
                        status_code=404, detail="not_found"
                    ) from None
                raise HTTPException(status_code=422, detail=str(exc)) from None
