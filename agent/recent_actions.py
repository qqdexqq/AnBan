"""Trusted recent-action context for explicit repeat requests.

Only server-side state is trusted here: the latest verified bank action for this
customer and session, its signed undo record, and the transfer row it points at. The
helper mints a fresh scoped PAYEE/AMT pair and returns masked context lines.
It never builds or runs a plan; the planner still chooses a fresh normal plan
that goes through the same policy, approval and execution gates.
"""

from __future__ import annotations

import hmac
import json
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from agent.executor import _undo_sig, undo_record
from agent.tools import REGISTRY
from security.audit import AuditLog
from security.tokens import Session, TokenEntry, TokenVault, _find_name_matches, _payee_tags, amount_bucket

_RECENT_STATUSES = frozenset({"pending", "released", "cancelled"})
_TRANSFER_TOOLS = frozenset({"transfer", "schedule_transfer", "request_payment"})

# A narrow "do it again" permission cue.  It is not an instruction to move
# money: it only allows the prior payee/amount to be offered back to the model.
_REPEAT_RE = re.compile(
    r"\b(?:again|retry|repeat|once\s+more|one\s+more\s+time)\b|"
    r"再转|重新转|再汇|重新汇|再发|再给|再来一次|再试一次|重试|重复",
    re.IGNORECASE,
)
_REPEAT_ONLY_RE = re.compile(
    r"^(?:(?:please|pls|can|could|would|you|kindly|请|麻烦|帮我|帮忙)\s*)*"
    r"(?:again|retry|repeat(?: it| that)?|once\s+more|one\s+more\s+time|"
    r"再转(?:一次)?|重新转(?:一次)?|再来一次|再试一次|重试|重复(?:一次)?)"
    r"(?:\s*(?:please|pls|thanks|thank you|谢谢|一下|吧|了))?"
    r"[\s.!?,，。！？~]*$",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b(?:send|transfer|pay|do\s+(?:it|that|this)|make\s+(?:the|a)\s+transfer|"
    r"try\s+(?:it|that)|run\s+(?:it|that))\b|"
    r"转|汇款|打钱|发钱|给[^，。,.!?]{0,20}转",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:don'?t|do\s+not|does\s+not|never|not|stop|needn'?t|no\s+need)\b|"
    r"别|不要|不用|不需要|不再|先别|不想|甭|毋须",
    re.IGNORECASE,
)
# A recipient phrase ("to Alice", "给Alice", "转给Alice") plus no PAYEE token
# means the user named someone the bank does not know.  Infinitives such as
# "want to transfer/send/do" are not recipients.
_RECIPIENT_INTENT_RE = re.compile(
    r"\bto\s+(?!(?:transfer|send|pay|do|make|retry|repeat|try|run|"
    r"it|that|this|them|him|her|me|us|the|a|an)\b)"
    r"[A-Za-z][A-Za-z .'-]{0,40}"
    r"|(?:转给|给)\s*(?!<PAYEE>)[^\s，。,.!?]{1,20}",
    re.IGNORECASE,
)
_DIRECT_RECIPIENT_RE = re.compile(
    r"\b(?:pay|send|transfer)\s+(?!(?:it|that|this|the|a|an|again|please|pls|"
    r"money|funds|payment|transfer|me|us|them|him|her|back|once|one|another|to)\b)"
    r"[A-Za-z][A-Za-z .'-]{0,40}\s+(?:again|once\s+more|one\s+more\s+time)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class RecentTransfer:
    """The bank row a signed cancel_transfer undo record points at."""

    transfer_id: str
    payee_id: str
    amount_fen: int
    status: str


@dataclass(slots=True)
class RecentAction:
    """Masked repeat context; labels are scoped to the current plan."""

    payee_label: str
    amount_label: str
    status: str
    explicit: bool
    explicit_payee: tuple[str, ...]
    explicit_amount: tuple[str, ...]
    cue_tokens: tuple[str, ...]
    context_lines: list[str]
    unknown_recipient: bool = False


def explicit_repeat(text: str) -> bool:
    """True only for an explicit "do that again" with no negation."""
    if not isinstance(text, str) or not text.strip():
        return False
    if _NEGATION_RE.search(text):
        return False
    stripped = text.strip()
    if _REPEAT_ONLY_RE.match(stripped):
        return True
    return bool(_REPEAT_RE.search(text) and _ACTION_RE.search(text))


def read_recent_transfer(
    conn: Any, secret_key: bytes, customer_id: str, session_id: str
) -> RecentTransfer | None:
    """Read the most recent verified bank action, including an older receipt's undo."""
    if conn is None or not AuditLog(conn, secret_key).verify().ok:
        return None
    events = conn.execute(
        "SELECT kind, plan_id, payload_json FROM security_audit "
        "WHERE customer_id = ? AND session_id = ? AND kind IN ('RUN', 'UNDO', 'PLAN', 'VETO') ORDER BY seq DESC",
        (customer_id, session_id),
    ).fetchall()
    row = None
    was_undo = False
    for event in events:
        payload = json.loads(event["payload_json"])
        tool = payload.get("tool")
        spec = REGISTRY.get(tool)
        if event["kind"] == "PLAN":
            output = payload.get("output") or {}
            steps = output.get("steps") or []
            if output.get("kind") != "plan" or any(
                (candidate := REGISTRY.get(step.get("tool"))) is None or candidate.kind == "write"
                for step in steps
            ):
                return None  # A newer unresolved request supersedes the old action.
            continue
        if event["kind"] == "VETO":
            if spec is not None and spec.kind == "write":
                return None
            continue
        if event["kind"] == "RUN" and (spec is None or spec.kind != "write"):
            continue
        if not payload.get("ok"):
            return None
        was_undo = event["kind"] == "UNDO"
        if tool != ("cancel_transfer" if was_undo else "transfer"):
            return None
        row = undo_record(conn, customer_id, event["plan_id"], str(payload.get("step_id") or ""))
        if row is None or (was_undo and not row.get("used")):
            return None
        count = conn.execute("SELECT COUNT(*) FROM exec_undo WHERE customer_id = ? AND plan_id = ? AND tool = 'cancel_transfer'", (customer_id, event["plan_id"])).fetchone()[0]
        if not was_undo and count != 1:
            return None
        break
    if row is None or str(row.get("tool") or "") != "cancel_transfer":
        return None
    expected = _undo_sig(
        bytes(secret_key or b""),
        str(row.get("customer_id") or ""),
        str(row.get("plan_id") or ""),
        str(row.get("step_id") or ""),
        str(row.get("tool") or ""),
        str(row.get("args_json") or ""),
        str(row.get("created_at") or ""),
    )
    if not hmac.compare_digest(expected, str(row.get("sig") or "")):
        return None
    try:
        args = json.loads(str(row.get("args_json") or ""))
    except (TypeError, ValueError):
        return None
    if not isinstance(args, dict):
        return None
    transfer_id = str(args.get("transfer_id") or args.get("id") or "")
    if not transfer_id:
        return None
    transfer = conn.execute(
        "SELECT to_contact_id, amount_fen, status FROM transfers"
        " WHERE customer_id = ? AND id = ?",
        (customer_id, transfer_id),
    ).fetchone()
    if transfer is None:
        return None
    amount = transfer["amount_fen"]
    if not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
        return None
    recorded = args.get("amount_fen")
    if recorded is not None and (
        isinstance(recorded, bool)
        or not isinstance(recorded, int)
        or recorded != amount
    ):
        return None
    payee_id = str(transfer["to_contact_id"] or "")
    status = str(transfer["status"] or "")
    if not payee_id or status not in _RECENT_STATUSES:
        return None
    if was_undo and status != "cancelled":
        return None
    contact = conn.execute(
        "SELECT id FROM contacts WHERE customer_id = ? AND id = ?",
        (customer_id, payee_id),
    ).fetchone()
    if contact is None:
        return None
    return RecentTransfer(transfer_id, payee_id, int(amount), status)


def _now_date(now: Any) -> date:
    if isinstance(now, datetime):
        return now.date()
    if isinstance(now, date):
        return now
    return date.fromisoformat(str(now)[:10])


def _unknown_recipient(text: str, bank: Any) -> bool:
    """A named recipient phrase the tokenizer did not resolve to a PAYEE."""
    # Mask only the exact verified spans. A known name elsewhere must not
    # conceal an unresolved recipient in this same request.
    contacts = bank.find_contact("")
    for start, end, _ in reversed(_find_name_matches(text, contacts)):
        text = text[:start] + "<PAYEE>" + text[end:]
    return bool(_RECIPIENT_INTENT_RE.search(text) or _DIRECT_RECIPIENT_RE.search(text))


def build_action(
    *,
    conn: Any,
    secret_key: bytes,
    customer_id: str,
    session_id: str,
    session: Session,
    vault: TokenVault,
    bank: Any,
    raw_text: str,
    user_entries: Sequence[TokenEntry],
    now: Any,
) -> RecentAction | None:
    """Read the last transfer and mint masked labels for the planner input.

    Ordinary requests get no context at all: a previous payee/amount may only
    be offered back after the user explicitly asks to repeat it.
    """
    if not explicit_repeat(raw_text):
        return None
    record = read_recent_transfer(conn, secret_key, customer_id, session_id)
    if record is None:
        return None
    explicit_payee = tuple(e.label for e in user_entries if e.kind == "PAYEE")
    explicit_amount = tuple(e.label for e in user_entries if e.kind == "AMT")
    if _unknown_recipient(raw_text, bank):
        # "send again to Alice" where Alice is not a contact: never fall back
        # to the old payee.  The reuse guard asks for the full request instead.
        return RecentAction(
            payee_label="",
            amount_label="",
            status=record.status,
            explicit=True,
            explicit_payee=explicit_payee,
            explicit_amount=explicit_amount,
            cue_tokens=(),
            context_lines=[],
            unknown_recipient=True,
        )
    finder = getattr(bank, "find_contact", None)
    contact = None
    if callable(finder):
        contact = next((c for c in finder("") if str(c.id) == record.payee_id), None)
    if contact is None:
        return None
    payee = vault.mint(
        session,
        "PAYEE",
        record.payee_id,
        _payee_tags(bank, contact, _now_date(now)),
        origin="bank_context",
    )
    amount = vault.mint(
        session,
        "AMT",
        record.amount_fen,
        OrderedDict([("bucket", amount_bucket(record.amount_fen))]),
        origin="bank_context",
    )
    cue: list[str] = []
    # Never offer an old value the user already replaced in this message.
    if not explicit_payee:
        cue.append(payee.label)
    if not explicit_amount:
        cue.append(amount.label)
    return RecentAction(
        payee_label=payee.label,
        amount_label=amount.label,
        status=record.status,
        explicit=True,
        explicit_payee=explicit_payee,
        explicit_amount=explicit_amount,
        cue_tokens=tuple(cue),
        context_lines=[
            f"Recent transfer: {payee.label} {amount.label} status={record.status}"
        ],
        unknown_recipient=False,
    )


def append_cues(tokenized: str, action: RecentAction) -> str:
    """Append only the missing authorized labels to the user's tokenized text."""
    present = set(re.findall(r"<[A-Z]+_[A-Za-z0-9]+>", tokenized))
    missing = [label for label in action.cue_tokens if label not in present]
    if not missing:
        return tokenized
    return f"{tokenized} {' '.join(missing)}".strip()


def as_state(action: RecentAction | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return {
        "payee_label": action.payee_label,
        "amount_label": action.amount_label,
        "status": action.status,
        "explicit": action.explicit,
        "explicit_payee": list(action.explicit_payee),
        "explicit_amount": list(action.explicit_amount),
        "unknown_recipient": action.unknown_recipient,
    }


def plan_reuse_question(
    plan: Any, action: Mapping[str, Any] | None, lang: str = "en"
) -> str | None:
    """Ask instead of letting a plan reuse the recorded transfer unsafely."""
    if not action:
        return None
    payee_label = str(action.get("payee_label") or "")
    amount_label = str(action.get("amount_label") or "")
    explicit = bool(action.get("explicit"))
    explicit_payee = {str(item) for item in action.get("explicit_payee") or ()}
    explicit_amount = {str(item) for item in action.get("explicit_amount") or ()}
    all_steps = list(getattr(plan, "steps", ()))
    transfer_steps = [
        step for step in all_steps if step.tool in _TRANSFER_TOOLS
    ]
    write_steps = [
        step
        for step in all_steps
        if (spec := REGISTRY.get(step.tool)) is not None and spec.kind == "write"
    ]
    if not all_steps:
        return None
    question = (
        'Please say the full request again, with who and how much, for example '
        '"send mom 300".'
        if lang == "en"
        else "请把收款人和金额完整说一遍，比如“给妈妈转300”。"
    )
    if action.get("unknown_recipient"):
        # The user named a recipient the bank does not know; any write is wrong.
        return question if write_steps else None
    if not explicit:
        # Ordinary requests get no implicit reuse (kept as a defensive guard).
        for step in transfer_steps:
            payee = step.args.get("payee")
            amount = step.args.get("amount")
            if (
                isinstance(payee, str)
                and payee == payee_label
                and payee not in explicit_payee
            ):
                return question
            if (
                isinstance(amount, str)
                and amount == amount_label
                and amount not in explicit_amount
            ):
                return question
        return None
    # A repeat may contain reads, but exactly one write: a single transfer.
    if len(write_steps) > 1:
        return question
    if write_steps and write_steps[0].tool != "transfer":
        return question
    if not write_steps:
        return None
    step = write_steps[0]
    amount = step.args.get("amount")
    payee = step.args.get("payee")
    if explicit_amount:
        # A new amount the user typed must win over the recorded one.
        if isinstance(amount, str) and amount not in explicit_amount:
            return question
    elif isinstance(amount, str) and amount != amount_label:
        # No new amount: only the recorded amount is authorized.
        return question
    if explicit_payee:
        if isinstance(payee, str) and payee not in explicit_payee:
            return question
    elif isinstance(payee, str) and payee != payee_label:
        # No new payee: only the recorded payee is authorized.
        return question
    return None
