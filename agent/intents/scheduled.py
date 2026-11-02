"""Scheduled-transfer (standing order) intent rules.

See agent/intents/__init__.py for the rule contract.  These rules only ever
emit the two read/write tools ``list_scheduled`` and ``stop_standing_order``;
creating an order stays with the planner.
"""

from __future__ import annotations

import re
from typing import Any

from agent.intents.bills import _has_write
from agent.intents.subscriptions import _has_any
from bank.api import Bank
from security.tokens import Session, _payee_tags

_SCHEDULED = (
    "定时转账",
    "自动转账",
    "每月转",
    "每个月转",
    "固定转账",
    "standing order",
    "standing orders",
    "scheduled transfer",
    "scheduled transfers",
    "recurring transfer",
    "recurring transfers",
    "automatic transfer",
    "automatic transfers",
    "monthly transfer",
    "monthly transfers",
)
_CANCEL = ("cancel", "stop", "取消", "停止", "停掉", "不要了")
# Negation, questions and hypotheticals: never stop from those.  "不要了"
# alone is a real stop, so only "不要取消" (and the other full phrases) count.
_DECLINE = re.compile(
    r"\b(?:do\s+not|don't|don’t|not|never|keep|what\s+if|what\s+happens|will|is\s+it"
    r"|should|would|why|how)\b"
    r"|不要取消|不取消|别取消|不要停|别停|先别|不想|保留|会不会|吗|怎么|如何|为什么|如果",
    re.IGNORECASE,
)
_SCHEDULED_SURFACE_RE = re.compile(
    "|".join(
        sorted(
            (
                rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])"
                if word.isascii()
                else re.escape(word)
                for word in _SCHEDULED
            ),
            key=len,
            reverse=True,
        )
    ),
    re.IGNORECASE,
)
_CLAUSE_SPLIT_RE = re.compile(
    r"\b(?:and|but|then|also|plus)\b|以及|并且|然后|再|还有|[,;，、；。]",
    re.IGNORECASE,
)
_PAYEE_RE = re.compile(r"<PAYEE_[A-Za-z0-9]+>")
_USER_MARKER_RE = re.compile(r"(?:^|\n)User:\s*", re.IGNORECASE)


def _step(tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": "s1", "tool": tool, "args": args or {}, "after": []}


def _plan(tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"kind": "plan", "steps": [_step(tool, args)]}


def _clarify(question: str, options: list[str]) -> dict[str, Any]:
    return {"kind": "clarify", "question": question, "options": options}


def _session(ctx: Any) -> Session:
    return Session(
        session_id=ctx.session_id,
        plan_id=ctx.plan_id,
        customer_id=ctx.customer_id,
    )


def _user_text(tokenized: str) -> str:
    """Only the current turn, never an earlier context line."""

    markers = list(_USER_MARKER_RE.finditer(tokenized))
    return tokenized[markers[-1].end() :] if markers else tokenized


def _lang(ctx: Any) -> str:
    return "zh" if ctx.lang == "zh" else "en"


def _bank(ctx: Any) -> Bank:
    return Bank(ctx.conn, ctx.customer_id)


def _active_contact_ids(ctx: Any) -> list[str]:
    """Contact ids of this customer's active standing orders, in order."""

    try:
        rows = ctx.conn.execute(
            "SELECT to_contact_id FROM scheduled_transfers "
            "WHERE customer_id = ? AND status = 'scheduled' ORDER BY id",
            (ctx.customer_id,),
        ).fetchall()
    except Exception:  # noqa: BLE001 - no readable orders means no rule
        return []
    return [str(row["to_contact_id"]) for row in rows]


def _contact(ctx: Any, contact_id: str) -> Any:
    try:
        contacts = _bank(ctx).find_contact("")
    except Exception:  # noqa: BLE001 - unknown contact: the plan is left out
        return None
    return next((item for item in contacts if str(item.id) == contact_id), None)


def _mint_payee(ctx: Any, contact: Any) -> str:
    # Same tag shape the tokenizer uses when it mints a contact from user text.
    entry = ctx.vault.mint(
        _session(ctx),
        "PAYEE",
        contact.id,
        _payee_tags(_bank(ctx), contact, ctx.now.date()),
        "bank_context",
    )
    return entry.label


def _has_mixed_write(text: str) -> bool:
    """Keep a second write action with the planner instead of dropping it."""

    residual: list[str] = []
    for clause in _CLAUSE_SPLIT_RE.split(text):
        if not _SCHEDULED_SURFACE_RE.search(clause):
            residual.append(clause)
            continue
        clause = _SCHEDULED_SURFACE_RE.sub(" ", clause)
        # Remove a cancellation verb only from the clause whose scheduled
        # object it acts on. A separate "cancel subscription" must remain.
        for word in _CANCEL:
            if word.isascii():
                clause, count = re.subn(
                    rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])",
                    " ",
                    clause,
                    count=1,
                    flags=re.IGNORECASE,
                )
            else:
                count = int(word in clause)
                clause = clause.replace(word, " ", 1)
            if count:
                break
        residual.append(clause)
    return _has_write(" ".join(residual))


def _resolve(ctx: Any) -> dict[str, Any] | None:
    text = ctx.raw_text if isinstance(ctx.raw_text, str) else ""
    tokenized = ctx.tokenized_text if isinstance(ctx.tokenized_text, str) else ""
    if not text.strip():
        return None
    user_text = _user_text(tokenized)

    if not _has_any(text, _SCHEDULED):
        return None

    cancel = _has_any(text, _CANCEL)
    # "transfer 500 to mum every month" creates an order: the planner owns it.
    if (
        "<AMT_" in user_text
        and "<PAYEE_" in user_text
        and _has_any(text, ("every month", "each month", "每月", "每个月"))
    ):
        return None

    # Listing is read-only, but must not swallow a separate write request.
    # Leave the whole mixed turn to the planner just like a mixed stop request.
    if _has_mixed_write(text):
        return None

    if not cancel:
        return _plan("list_scheduled")

    # A negated, hypothetical or questioning turn is not a stop request: only
    # read-only listing, so a misread sentence can never cancel rent.
    if _DECLINE.search(user_text) or user_text.rstrip().endswith(("?", "？")):
        return _plan("list_scheduled")

    labels = _PAYEE_RE.findall(user_text)
    if len(labels) > 1:
        # A single-payee tool cannot safely represent a multi-order request.
        return None
    if labels:
        return _plan("stop_standing_order", {"payee": labels[0]})

    order_contact_ids = _active_contact_ids(ctx)
    lang = _lang(ctx)
    if not order_contact_ids:
        question = (
            "你目前没有生效的定时转账。"
            if lang == "zh"
            else "You have no active standing orders."
        )
        return _clarify(question, [])
    if len(order_contact_ids) == 1:
        contact = _contact(ctx, order_contact_ids[0])
        if contact is None:
            return None
        return _plan("stop_standing_order", {"payee": _mint_payee(ctx, contact)})

    contacts: list[Any] = []
    seen: set[str] = set()
    for contact_id in order_contact_ids:
        contact = _contact(ctx, contact_id)
        if contact is None or contact.id in seen:
            continue
        seen.add(contact.id)
        contacts.append(contact)
    if not contacts:
        question = (
            "你目前没有生效的定时转账。"
            if lang == "zh"
            else "You have no active standing orders."
        )
        return _clarify(question, [])
    question = (
        "你想停止哪一笔定时转账？"
        if lang == "zh"
        else "Which standing order should I stop?"
    )
    return _clarify(question, [_mint_payee(ctx, contact) for contact in contacts])


def resolve(ctx: Any) -> dict[str, Any] | None:
    try:
        return _resolve(ctx)
    except Exception:  # noqa: BLE001 - a rule must never break a turn
        return None
