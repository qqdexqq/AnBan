"""Narrow read-only resolver for "who did I transfer to?" questions.

Only the shape of the raw question is classified here; the returned plan is
always the existing ``list_transactions`` read.  Nothing is written, no new
tool or schema is added, and the raw text never leaves this module.  A missing
period may be taken from the most recent verified successful read in the same
customer/session, and only while the audit chain verifies.
"""

from __future__ import annotations

import json
import re
from typing import Any

from security.audit import AuditLog

_READS = frozenset({"list_transactions", "spending_report"})
_PERIODS = frozenset(
    {"this_month", "last_month", "this_year", "last_7_days", "last_30_days"}
)

_PERIOD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "last_7_days",
        re.compile(
            r"\b(?:last|past)\s+(?:7|seven)\s+days?\b|\blast\s+week\b|"
            r"最近7天|近7天|过去7天|最近一周|近一周|过去一周|这周|本周",
            re.IGNORECASE,
        ),
    ),
    (
        "last_30_days",
        re.compile(
            r"\b(?:last|past)\s+(?:30|thirty)\s+days?\b|"
            r"最近30天|近30天|过去30天|最近一个月|近一个月|过去一个月",
            re.IGNORECASE,
        ),
    ),
    ("last_month", re.compile(r"\blast\s+month\b|\bprevious\s+month\b|上个月|上月", re.IGNORECASE)),
    ("this_year", re.compile(r"\bthis\s+year\b|今年|本年", re.IGNORECASE)),
    ("this_month", re.compile(r"\bthis\s+month\b|\bcurrent\s+month\b|这个月|这月|本月", re.IGNORECASE)),
)

_EN_QUESTION = re.compile(
    r"\bto\s+whom\b|\bwhom\s+(?:did|have|had)\s+i\b|"
    r"\bwho\s+(?:did|have|had)\s+i\s+"
    r"(?:transfer|send|sent|pay|paid|give|given|wire|wired)\b|"
    r"\bwhich\s+(?:person|people)\s+did\s+i\s+(?:transfer|send|pay|give)\b|"
    r"\b(?:did|have)\s+i\s+(?:transfer|send|sent|pay|paid|give|given)\b",
    re.IGNORECASE,
)
_EN_PAST = re.compile(
    r"\b(?:did|have|had|already|earlier|previously)\b|"
    r"\bi\s+(?:sent|transferred|paid|gave)\b",
    re.IGNORECASE,
)
_EN_FINANCE = re.compile(
    r"\b(?:transfer(?:red|s)?|pay(?:ment|ments|ing)?|paid|money|funds|cash|"
    r"yuan|rmb)\b|[¥￥]",
    re.IGNORECASE,
)
_ZH_QUESTION = re.compile(
    r"(?:转|汇)(?:账|钱)?给(?:了)?谁|给谁(?:转|汇)(?:了)?|"
    r"谁.{0,6}(?:收到|收了).{0,8}(?:转账|汇款|钱)"
)
_ZH_PAST = re.compile(
    r"了|过|刚才|刚刚|上次|上个月|上月|上回|之前|昨天|前天|上周|去年|这笔|那笔"
)
# Cancellation/undo/refund, negations, contact listings and other commands are
# not a past-recipient lookup: let the normal planner handle them.
_BLOCK = re.compile(
    r"\b(?:cancel|undo|reverse|recall|refund|chargeback|contacts?|"
    r"pay\s+a\s+visit|pay\s+attention|request|schedule|reserve|subscribe|"
    r"purchase|buy|order|withdraw|deposit|recharge|top\s*up|"
    r"don'?t|does?\s+not|did\s+not|didn'?t|never|without|stop|not)\b|"
    r"取消|撤销|撤回|退款|退钱|退汇|退回|联系人|通讯录|收款|定时转|预约转|"
    r"购买|下单|充值|提现|存款|别|不要|不用|不需要|"
    r"不再|先别|不想|甭",
    re.IGNORECASE,
)
# An extra imperative write/money move makes the message mixed: refuse to
# answer it with a read plan and leave it to the existing planner.
_EXTRA_WRITE = re.compile(
    r"(?:^|[,.!?;。！？；]\s*|\b(?:and|then|also)\s+)\s*"
    r"(?:(?:please|pls|kindly|can you|could you|I want to|I am going to|I'm going to)\s+)?"
    r"(?:send|transfer|pay|give)\b|"
    r"(?:^|[,，;；。！？\n]|然后|再)\s*(?:请|麻烦|帮我)?\s*"
    r"(?:(?:给|向)(?!谁)[^\s，。,.!?]{1,20}(?:转|汇|打钱|发钱)|(?:转|汇|打钱|发钱)\s*(?:<AMT_|\d|¥|￥))",
    re.IGNORECASE,
)
_FUTURE = re.compile(
    r"\b(?:should|shall|would|could|will|might|must)\s+(?:i\s+)?"
    r"(?:have\s+)?(?:send|transfer|pay|give)\b|"
    r"(?:该|应该|要不要|想|要|需要|可以|能不能|能否|帮我|帮忙|请|麻烦)\s*"
    r"(?:给|转|汇|打)",
    re.IGNORECASE,
)


def _explicit_period(text: str) -> str | None:
    clauses = re.split(r"[!?;；。！？\n]|(?<!\d)\.|\.(?!\d)", text)
    for clause in clauses:
        if _EN_QUESTION.search(clause) or _ZH_QUESTION.search(clause):
            for period, pattern in _PERIOD_PATTERNS:
                if pattern.search(clause):
                    return period
    for period, pattern in _PERIOD_PATTERNS:
        if pattern.search(text):
            return period
    return None


def _is_past_transfer_recipient(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    if _BLOCK.search(text) or _EXTRA_WRITE.search(text) or _FUTURE.search(text):
        return False
    if _ZH_QUESTION.search(text):
        return bool(_ZH_PAST.search(text))
    return bool(
        _EN_QUESTION.search(text)
        and _EN_PAST.search(text)
        and _EN_FINANCE.search(text)
    )


def _inherited_period(
    conn: Any, secret_key: bytes, customer_id: str, session_id: str
) -> str | None:
    """The last verified successful read's period in this exact chat."""
    try:
        if conn is None or not AuditLog(conn, secret_key).verify().ok:
            return None
        rows = conn.execute(
            "SELECT plan_id, kind, payload_json FROM security_audit "
            "WHERE customer_id = ? AND session_id = ? "
            "AND kind IN ('PLAN', 'RUN') ORDER BY seq DESC",
            (customer_id, session_id),
        ).fetchall()
        plans: dict[str, dict[str, str | None]] = {}
        for row in rows:
            if row["kind"] != "PLAN":
                continue
            output = json.loads(row["payload_json"]).get("output")
            if not isinstance(output, dict):
                return None
            if output.get("kind") != "plan":
                continue
            steps = output.get("steps")
            if not isinstance(steps, list):
                return None
            parsed: dict[str, str | None] = {}
            for step in steps:
                if not isinstance(step, dict) or not isinstance(step.get("id"), str):
                    return None
                args = step.get("args") or {}
                if not isinstance(args, dict):
                    return None
                if step.get("tool") in _READS:
                    period = args.get("period")
                    if period is not None and period not in _PERIODS:
                        return None
                    parsed[step["id"]] = period
            plans[str(row["plan_id"])] = parsed
        for row in rows:  # newest RUN first
            if row["kind"] != "RUN":
                continue
            payload = json.loads(row["payload_json"])
            if not isinstance(payload, dict) or payload.get("ok") is not True:
                continue
            if payload.get("tool") not in _READS:
                continue
            read_steps = plans.get(str(row["plan_id"]), {})
            step_id = str(payload.get("step_id") or "")
            if step_id in read_steps:
                return read_steps[step_id] or "this_month"
        return None
    except Exception:  # noqa: BLE001 - no verified history means no inheritance
        return None


def historical_transfer_plan(
    text: str,
    *,
    conn: Any,
    secret_key: bytes,
    customer_id: str,
    session_id: str,
) -> dict[str, Any] | None:
    """Return a read-only transfer-recipient lookup plan, or ``None``."""
    if not _is_past_transfer_recipient(text):
        return None
    period = _explicit_period(text) or _inherited_period(
        conn, secret_key, customer_id, session_id
    ) or "this_month"
    return {
        "kind": "plan",
        "steps": [
            {
                "id": "s1",
                "tool": "list_transactions",
                "args": {"period": period, "category": "transfer", "direction": "out"},
                "after": [],
            }
        ],
    }
