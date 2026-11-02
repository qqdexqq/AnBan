"""LangGraph turn pipeline: intake, plan, check, policy, approve, execute, reply.

The graph only carries JSON-safe state.  Live objects (bank connection, vault,
mandates, audit log, planner) live on :class:`Services` and are never
checkpointed.

Approval is a LangGraph ``interrupt``: the first invoke returns
``needs_approval`` with the pending steps and demo OTPs, and a later
``Command(resume=...)`` continues the same thread.  Coders never decide a tier;
policy does, and the executor consumes a one-shot mandate for every yellow or
red write.
"""

from __future__ import annotations

import json
import math
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from agent.conversation import CONVERSATION_SYSTEM_PROMPT
from agent.conversation_memory import ContextBudgetExceeded
from agent.executor import (
    ExecError,
    StepContext,
    latest_undo,
    resolve_args,
    run_step,
    undo_record,
    undo_step,
)
from agent import habits, recent_actions
from agent.basket_preflight import budget_question
from agent.prompts import FALLBACK_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT, render_planner_input
from agent.responder import ReplyResult, compose_reply
from agent.standing_order_hook import run_and_audit
from agent.tools import REGISTRY
from bank.api import Bank, BankError
from bank.money import fmt_fen
from bank.recall import open_recall
from security import compliance
from security.audit import AuditKind, AuditLog
from security.dag import PlanError, PlanStep, ValidatedPlan, parse_planner_output, validate_plan
from security.mandate import OPEN_STATUSES, MandateError, MandateService
from security.policy import PolicyConfig, Reason, StepDecision, evaluate, recheck_step
from security.scam import HIGH_RISK_FLAGS, detect
from security.tokens import (
    Session,
    TokenError,
    TokenVault,
    build_context,
    detokenize,
    tokenize_untrusted,
    tokenize_user_text,
)

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_TOKEN_RE = re.compile(r"^<[A-Z]+_[A-Za-z0-9]+>$")

_MAX_APPROVAL_ROUNDS = 3
_MAX_RECENT_TEXTS = 5

_CANCEL_RE = re.compile(
    r"算了|不转了|不要了|取消吧|别转了|cancel|never ?mind|forget it",
    re.IGNORECASE,
)
# What may be left of a cancel message once the cancel phrase is removed.
_CANCEL_FILLER_RE = re.compile(
    r"[\s\W_]|吧|了|啦|呢|啊|吗|好|那|这|笔|个|转账|付款|操作|"
    r"please|thanks|thank you|谢谢|不用|it|that|this|the|transfer|payment|plan|order|"
    r"ok|okay|actually",
    re.IGNORECASE,
)
# A whole message that only says "yes" / "ok" while an approval is waiting.
_BARE_CONFIRM_RE = re.compile(
    r"(?:好的?|好吧|好啊|可以|行|嗯+|确认|确定|是的?|对的?|没问题|转吧|"
    r"ok(?:ay)?|yes|yep|yeah|sure|confirm(?:ed)?|go ahead|do it|"
    r"i confirm(?:[\s,]*go ahead)?)(?:[\s,，]*(?:please|thanks|谢谢))?[\s。.!！,，~]*",
    re.IGNORECASE,
)
# "Undo what I just did" typed in chat; undo itself is the card's button.
_UNDO_RE = re.compile(r"撤销|撤回|撤掉|退回来|undo|take (?:it|that) back|reverse", re.IGNORECASE)
_UNDO_RECENT_RE = re.compile(
    r"刚才|刚刚|上一笔|那笔|这笔|转错|\blast\b|\bjust\b|\bthat\b|\bit\b", re.IGNORECASE
)
# Typed undo runs the same server-side undo as the button, but only when the
# whole message is an undo/recall phrase: anything more (a coached script, a
# new request) goes through the graph and its scam guard.
_RECALL_RE = re.compile(
    r"追回|退汇|要回来|拿回来|recall|get (?:my|the) money back", re.IGNORECASE
)
_SHORTCUT_FILLER_RE = re.compile(
    r"撤销|撤回|撤掉|退回来|追回|退汇|要回来|拿回来|取消|刚才|刚刚|上一笔|那笔|这笔|转账|"
    r"一下|帮我|请|麻烦|吧|了|的|我要|我想|申请|"
    r"\b(?:undo|recall|reverse|take|back|get|my|the|money|that|it|last|just|"
    r"transfer|payment|please|pls|can|you|i|want|to|request|a)\b"
    r"|[\s,，。.!！?？~]",
    re.IGNORECASE,
)
# Undos that only give money back or tighten something; others need a request.
_TYPED_UNDO_TOOLS = frozenset(
    {"cancel_transfer", "cancel_scheduled", "cancel_request", "cancel_order", "close_virtual_card"}
)
# What a typed undo names, by the undo tool it needs.  Naming a different kind
# of action than the newest one falls back to the Undo button.
_UNDO_KIND_WORDS = {
    "cancel_transfer": ("转账", "转的", "那笔", "钱", "transfer", "payment", "money"),
    "cancel_scheduled": ("定期", "每月", "自动转", "standing order", "scheduled", "recurring"),
    "cancel_request": ("收款", "要钱", "request"),
    "restore_subscription": ("订阅", "会员", "subscription"),
    "cancel_order": ("订单", "购买", "买的", "order", "purchase"),
    "unfreeze_card": ("冻结", "卡", "freeze", "frozen", "card"),
    "restore_card_limit": ("限额", "limit"),
    "close_virtual_card": ("虚拟卡", "virtual card"),
}
_USUAL_RE = re.compile(
    r"老样子|照常|照旧|跟上次一样|和上次一样|还是那么多|the usual|as usual|same as (?:last time|usual)",
    re.IGNORECASE,
)
# "Is this a scam?" about a message the customer received: answered with the
# detector's verdict and advice, even while transfers are frozen.
_SCAM_QUESTION_RE = re.compile(
    r"是诈骗吗|是不是诈骗|是不是骗|是骗子吗|是骗局吗|是真的吗|靠谱吗|可信吗|能信吗|"
    r"is (?:this|it|that) (?:a )?(?:scam|fraud|real|legit)|scam\?|legit\?",
    re.IGNORECASE,
)
_HANDOFF_RE = re.compile(
    r"人工|真人|客服|找人|human|real person|talk to (?:a|an) (?:person|agent)",
    re.IGNORECASE,
)
_WHY_RE = re.compile(
    r"为什么.*(?:验证|刷脸|短信|限额|冻结|暂停|锁定)"
    r"|why .*(?:check|verif|face|code|limit|block|frozen|locked)",
    re.IGNORECASE,
)
# Frozen customers may still ask read-only questions.  This is only a cheap
# intake hint: a write-looking request never reaches the planner while frozen.
_READ_ONLY_RE = re.compile(
    r"余额|花了多少|支出|账单|交易|流水|订阅|持仓|理财|产品|异常|"
    r"balance|spend|spent|transactions?|subscriptions?|portfolio|products?|"
    r"anomal|statement",
    re.IGNORECASE,
)

_CN_ORDINALS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5}
_EN_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


# --------------------------------------------------------------------- service


@dataclass(slots=True)
class Services:
    """Live dependencies shared by every node; never stored in graph state."""

    conn: Any
    vault: TokenVault
    mandates: MandateService
    audit: AuditLog
    planner: Any
    policy_config: PolicyConfig
    secret_key: bytes
    now: Callable[[], datetime]
    planner_system_prompt: str = PLANNER_SYSTEM_PROMPT
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Raw text for the in-flight turn only; popped when that turn returns.
    turn_text: dict[str, str] = field(default_factory=dict)
    # Reply language chosen in Settings for this turn ("en"/"zh"); overrides
    # detection from the message text.
    turn_lang: dict[str, str] = field(default_factory=dict)
    # General planner for requests the fine-tuned one cannot handle (unusual
    # wording, intents it was not trained on). Same input, same gates.
    fallback_planner: Any = None
    # Previous raw texts for scam detection; never part of graph state.
    recent_raw: dict[str, list[str]] = field(default_factory=dict)
    # Account-wide scam evidence only; never rendered into planner context.
    customer_risk_texts: dict[str, list[str]] = field(default_factory=dict)
    # Full (raw) step results for the responder; never part of graph state.
    turn_results: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    # A pending clarify turn: previous raw text and option display names.
    clarify: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Investment-bait warning per customer: product purchases are refused
    # until this time, in every session.
    invest_warned: dict[str, datetime] = field(default_factory=dict)
    # Conversation-only replies for safe small talk; None keeps every turn on
    # the banking planner.  Both are optional so the normal runtime and all
    # existing tests run exactly as before.
    conversation_gateway: Any = None
    conversation_memory: Any = None

    def bank_for(self, customer_id: str) -> Bank:
        return Bank(self.conn, customer_id)


@dataclass(slots=True)
class TurnResult:
    """One turn's public result."""

    status: str
    reply: str
    approvals: list[dict[str, Any]] = field(default_factory=list)
    plan_view: list[dict[str, Any]] = field(default_factory=list)
    session_id: str = ""
    plan_id: str = ""
    # Buttons the client may offer next, e.g. "recall" after a closed undo window.
    actions: list[str] = field(default_factory=list)


class AgentState(TypedDict, total=False):
    actions: list[str]
    session_id: str
    customer_id: str
    lang: str
    plan_id: str
    tokenized_text: str
    user_msg: str
    planner_output: str
    plan: dict[str, Any]
    decisions: list[dict[str, Any]]
    plan_view: list[dict[str, Any]]
    approvals_pending: list[dict[str, Any]]
    results: dict[str, Any]
    outcome: dict[str, Any]
    status: str
    reply: str
    grounded: bool
    rejected: list[str]
    recent_action: dict[str, Any] | None


# -------------------------------------------------------------------- helpers


def _detect_lang(text: str) -> str:
    """Return ``zh`` for CJK-dominant text, ``en`` for Latin, ``mixed`` for both."""

    visible = [char for char in text if not char.isspace()]
    if not visible:
        return "en"
    cjk = sum(1 for char in visible if _CJK_RE.match(char))
    latin = sum(1 for char in visible if _LATIN_RE.match(char))
    if cjk and latin:
        return "mixed"
    if cjk / len(visible) > 0.3:
        return "zh"
    return "en"


def _is_en(lang: str) -> bool:
    return lang == "en"


def _short_detail(detail: Any, limit: int = 200) -> str:
    return re.sub(r"\s+", " ", str(detail or "")).strip()[:limit]


def _ordinal_index(text: str) -> int | None:
    """Return the 1-based option number in a short clarify answer, if any."""

    cleaned = text.strip().lower()
    match = re.search(r"第\s*([一二两三四五1-5])", cleaned)
    if match is not None:
        token = match.group(1)
        return _CN_ORDINALS.get(token) or int(token)
    match = re.fullmatch(r"\s*([一二两三四五1-5])\s*(?:个|位|条|张|项)?\s*", cleaned)
    if match is not None:
        token = match.group(1)
        return _CN_ORDINALS.get(token) or int(token)
    match = re.search(
        r"\b(?:the\s+)?(first|second|third|fourth|fifth)(?:\s+one)?\b", cleaned
    )
    if match is not None:
        return _EN_ORDINALS[match.group(1)]
    match = re.fullmatch(
        r"\s*(?:option|number|no\.?)?\s*([1-5])\s*(?:st|nd|rd|th)?"
        r"\s*(?:one|option)?\s*",
        cleaned,
    )
    if match is not None:
        return int(match.group(1))
    return None


def _apply_clarify(raw: str, pending: Mapping[str, Any]) -> str:
    """Merge a clarify answer with the previous raw text (see customer flow)."""

    answer = str(raw).strip()
    previous = str(pending.get("raw") or "").strip()
    options = [str(option) for option in (pending.get("options") or ()) if option]

    choice: str | None = None
    index = _ordinal_index(answer)
    if index is not None and 1 <= index <= len(options):
        choice = options[index - 1]
    if choice is None:
        for option in options:
            if option in answer:
                choice = option
                break
    if choice is not None:
        return f"{previous} {choice}".strip()
    if _is_short_answer(answer):
        return f"{previous} {answer}".strip()
    # A full new request is not an answer: merging it with the old one
    # carried a stale amount (¥200 instead of ¥500) and swallowed a scam
    # message as "the answer".
    return answer


def _is_short_answer(text: str) -> bool:
    """An amount, a name or "the second one": at most 12 CJK characters or
    4 words; anything longer is a new request."""

    cjk = len(_CJK_RE.findall(text))
    words = len(re.findall(r"[A-Za-z0-9.,¥]+", text))
    return cjk <= 12 and words <= 4


_OPTION_RELATION_WORDS = {"friend": "朋友", "colleague": "同事"}


def _option_display(
    services: Services,
    session: Session,
    token: str,
) -> str:
    """Human-readable option name for clarify continuity (no real PII)."""

    try:
        entry = services.vault.resolve(token, session)
    except TokenError:
        return token
    if entry.kind == "PAYEE":
        row = services.conn.execute(
            "SELECT name, relation FROM contacts WHERE customer_id = ? AND id = ?",
            (session.customer_id, entry.real_ref),
        ).fetchone()
        if row is not None:
            # "朋友 张伟" / "同事 张伟": the relation word makes the merged answer
            # name exactly one contact, so a same-name clarify cannot loop.
            prefix = _OPTION_RELATION_WORDS.get(str(row["relation"] or ""))
            return f"{prefix} {row['name']}" if prefix else str(row["name"])
    elif entry.kind == "SUB":
        row = services.conn.execute(
            "SELECT merchant FROM subscriptions WHERE customer_id = ? AND id = ?",
            (session.customer_id, entry.real_ref),
        ).fetchone()
        if row is not None:
            return str(row["merchant"])
    elif entry.kind == "CARD":
        row = services.conn.execute(
            "SELECT last4 FROM cards WHERE customer_id = ? AND id = ?",
            (session.customer_id, entry.real_ref),
        ).fetchone()
        if row is not None:
            return f"尾号{row['last4']}"
    return detokenize(token, services.vault, session, services.bank_for(session.customer_id))


def _freeze_until(
    services: Services,
    customer_id: str,
    record: Mapping[str, Any],
    now: datetime,
) -> datetime | None:
    """When the customer-wide scam freeze ends, if it is still active."""

    until = record.get("frozen_until")
    if isinstance(until, datetime) and now < until:
        return until
    row = services.conn.execute(
        "SELECT frozen_until FROM security_scam_freezes WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        until = datetime.fromisoformat(row["frozen_until"])
    except (TypeError, ValueError):
        return None
    return until if now < until else None


def _frozen_reply(until: datetime, now: datetime, lang: str) -> str:
    hours = max(1, math.ceil((until - now).total_seconds() / 3600))
    # A bare "15:48" read as "in one minute" when the pause ends tomorrow.
    days = (until.date() - now.date()).days
    clock = until.strftime("%H:%M")
    if days == 0:
        moment = clock
    elif days == 1:
        moment = f"tomorrow {clock}" if _is_en(lang) else f"明天 {clock}"
    else:
        moment = until.strftime("%m-%d ") + clock
    if _is_en(lang):
        plural = "hour" if hours == 1 else "hours"
        return (
            f"For your safety, transfers are paused until {moment} "
            f"(about {hours} {plural}). For help call 96110, or reply "
            "\"human\" to reach support."
        )
    return (
        f"为了安全，转账已暂停至 {moment}（约{hours}小时后恢复）。"
        "如需帮助请拨打 96110，或回复“人工”联系客服。"
    )


def _why_reply(config: PolicyConfig, lang: str) -> str:
    daily = fmt_fen(config.daily_limit_fen)
    elder = fmt_fen(config.elder_daily_limit_fen)
    if _is_en(lang):
        co_approval = (
            " (older customers also need family co-approval)"
            if config.red_needs_co_approval_for_elder
            else ""
        )
        return (
            "Everyday small payments only need one tap to confirm.\n"
            f"To send more than {daily} a day ({elder} for older customers), "
            f"you need an SMS code and a face check{co_approval}.\n"
            "A scam warning pauses transfers for 24 hours.\n"
            "Three wrong codes lock transfers for 24 hours."
        )
    co_approval = "（老人还需要家人共同确认）" if config.red_needs_co_approval_for_elder else ""
    return (
        "日常小额操作，你点一下确认就行。\n"
        f"每天超过 {daily}（老人 {elder}）需要短信验证码 + 刷脸{co_approval}。\n"
        "提示有诈骗风险时，转账会暂停 24 小时。\n"
        "验证码连续输错 3 次，会锁定 24 小时。"
    )


def _guard_model_output(services: Any, state: Any, now: datetime, raw: str) -> str:
    """Code checks on every model-written plan (first try, fallback and retry):
    e.g. a model may not pick a subscription the user never named."""
    from agent.intents import IntentContext, guard_model_plan

    ctx = IntentContext(
        raw_text=services.turn_text.get(state["session_id"], ""),
        tokenized_text=str(state.get("tokenized_text") or ""),
        conn=services.conn, secret_key=services.secret_key,
        customer_id=state["customer_id"], session_id=state["session_id"],
        now=now, lang=state["lang"],
        vault=services.vault, plan_id=state["plan_id"],
    )
    return guard_model_plan(ctx, raw) or raw


def _planner_error_reply(lang: str) -> str:
    if _is_en(lang):
        return (
            "I didn't quite understand that. Could you rephrase it? For example, "
            "\"send 300 to mum\" or \"how much did I spend this month?\"."
        )
    return (
        "没太明白你的意思，可以换个说法吗？"
        "比如“给妈妈转300”或“这个月花了多少”。"
    )


_SHORTEN_REQUEST = "Please shorten this request or start a new chat."


def _shorten_request(state: AgentState) -> AgentState:
    """Ask for a shorter request when system prompt plus turn cannot fit.

    ``reply`` is set because the responder only passes known clarify questions
    through; this code-owned question must never be replaced or truncated.
    """

    state["status"] = "clarify"
    state["outcome"] = {
        "kind": "clarify",
        "question": _SHORTEN_REQUEST,
        "reply": _SHORTEN_REQUEST,
        "options": [],
    }
    return state


def _assistant_summary(state: AgentState, outcome: Any, reply_text: str) -> str:
    """A memory note that is safe to show the model later.

    A conversation reply is already validated prose.  Every banking turn is
    reduced to the trusted status and tool names from the current plan: never
    a bank reply, a step result, an amount or a semantic label.
    """

    if isinstance(outcome, Mapping) and outcome.get("kind") == "conversation":
        return reply_text
    tools: set[str] = set()
    plan = state.get("plan")
    if isinstance(plan, Mapping):
        for step in plan.get("steps") or ():
            if not isinstance(step, Mapping):
                continue
            tool = step.get("tool")
            if isinstance(tool, str) and tool in REGISTRY:
                tools.add(tool)
    summary = "status=" + str(state.get("status") or "")
    if tools:
        summary += " tools=" + ",".join(sorted(tools))
    return summary


def _jsonable(value: Any) -> Any:
    """Recursively convert executor results to JSON-safe checkpoint values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    return str(value)


def _public_result(result: Any) -> Any:
    """Drop the full bank row from a step result before checkpointing it."""

    if isinstance(result, Mapping):
        return {str(key): item for key, item in result.items() if key != "raw"}
    return result


def _outcome_with_raw(
    outcome: Mapping[str, Any],
    raw_results: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Reattach full step results for the responder without touching state."""

    merged = dict(outcome)
    steps = outcome.get("steps")
    if not raw_results or not isinstance(steps, list):
        return merged
    merged_steps = []
    for step in steps:
        if isinstance(step, Mapping):
            step = dict(step)
            result = raw_results.get(str(step.get("step_id") or ""))
            if result is not None:
                step["result"] = result
        merged_steps.append(step)
    merged["steps"] = merged_steps
    return merged


def _ancestor_ids(plan: ValidatedPlan, step_ids: set[str]) -> set[str]:
    """Return every transitive dependency of the given plan steps."""

    by_id = {step.id: step for step in plan.steps}
    ancestors: set[str] = set()
    stack = [
        dependency
        for step_id in step_ids
        for dependency in by_id[step_id].after
    ]
    while stack:
        step_id = stack.pop()
        if step_id in ancestors or step_id not in by_id:
            continue
        ancestors.add(step_id)
        stack.extend(by_id[step_id].after)
    return ancestors


def _is_bare_cancel(text: str) -> bool:
    """True for "算了" / "cancel it", not for a new request like "cancel my Spotify"."""
    if not _CANCEL_RE.search(text):
        return False
    rest = _CANCEL_FILLER_RE.sub("", _CANCEL_RE.sub("", text))
    return not rest


_UNDO_TAIL_RE = re.compile(
    r"^(?:now|asap|right now|thanks|thank you|现在|马上|立刻|赶紧|快|"
    r"一下|掉|啊|呀|哦|嘛|哈|呢|吧|了|啦)$",
    re.IGNORECASE,
)


def _whole_undo_intent(text: str) -> bool:
    """True when the message is only a typed undo, not a redo/new request.

    "I accidentally clicked undone ... send it again" contains the substring
    "undo", but it is a request to repeat the transfer, not an undo command.
    A short action noun may remain ("撤销刚才取消的订阅"), which is still the
    same undo-help path; anything longer is a new request for the planner.
    """
    if recent_actions.explicit_repeat(text):
        return False
    if not _UNDO_RE.search(text):
        return False
    rest = _SHORTCUT_FILLER_RE.sub("", text).strip()
    return len(rest) <= 2 or bool(_UNDO_TAIL_RE.fullmatch(rest))


# "Same as last time" words: only then may an amount come from history.
_BACKREF_RE = re.compile(
    # Whole phrases only: "上次的事算了", "不一样的数", "unusual", "against"
    # must not unlock an amount from history.
    r"上次(?!的事)|上回|上一次|上一笔|上个月|老样子|老规矩|照旧|"
    r"(?:跟|和|同|像)上次一样|(?<!不)一样的?(?:金额|数|钱)|(?<!不)一样多|"
    r"按之前|再转一次|再来一次|那么多|这次也一样|跟上一样|"
    r"\bsame (?:as|amount)\b|\blike last\b|\blast time\b|\bagain\b|\bas usual\b|"
    r"\busual\b|\bas before\b|\bprevious (?:amount|payment|transfer|time)\b",
    re.IGNORECASE,
)
# Recurrence words: only then may the plan set up a standing order.  A date on
# its own ("明天") is a one-off, and "3号楼" / "everyone" are not cues.
_RECURRING_RE = re.compile(
    r"每月|每个月|每周|每星期|每天|每日|每年|定期|定时|按月|月初|月底|"
    r"自动(?:转|扣|付)|发工资|固定(?:转|给|打)|以后(?:每|都|照)|"
    r"\d+\s*号(?![楼栋室房门线床院口])|[一二三四五六七八九十]+号(?![楼栋室房门线床院口])|"
    r"\bevery\b|\bmonthly\b|\bweekly\b|\bdaily\b|\beach month\b|\brecurring\b|"
    r"\bstanding order\b|\bschedul|\bpayday\b|\bpay day\b|\bsalary\b|\bpaycheck\b|"
    r"\b\d+(?:st|nd|rd|th)\b",
    re.IGNORECASE,
)


_UNFREEZE_RE = re.compile(r"解冻|解锁|解封|恢复使用|unfreeze|unlock|unblock|reactivate|thaw", re.IGNORECASE)
_LOST_RE = re.compile(r"丢|掉了|不见了|被偷|被盗|遗失|挂失|\blost\b|stolen|missing", re.IGNORECASE)
_SPLIT_RE = re.compile(
    r"AA|平分|均分|分摊|每人|每个人|一人一半|一半|分成|分\s*\d+\s*(?:次|笔|份)|"
    r"\bsplit\b|\beach\b|per person|equally|\bhalf\b|share",
    re.IGNORECASE,
)


def _cue_question(plan: ValidatedPlan, text: str, lang: str) -> str | None:
    """Ask instead of planning when the user's words give no cue for a step.

    Live tests: the planner copied an amount from history the user never
    referred to, and turned plain transfers into monthly standing orders.
    """
    en = _is_en(lang)
    for step in plan.steps:
        if step.tool == "unfreeze_card" and (
            not _UNFREEZE_RE.search(text) or _LOST_RE.search(text)
        ):
            # "i lost my card!!" was planned as unfreeze_card (tester B3).
            return (
                "Do you want to freeze or unfreeze the card?"
                if en
                else "请问是要冻结还是解冻卡片？"
            )
        if step.args.get("divide_by") and not _SPLIT_RE.search(text):
            # The planner split ¥4,999 by 4 unasked.
            return "How much would you like to transfer?" if en else "请问要转多少金额？"
    # Steps whose amount comes from history: list_transactions, and any step
    # that took its own amount from such a step (this covers chains, and money
    # tools other than transfer).
    history_steps = {step.id for step in plan.steps if step.tool == "list_transactions"}
    for step in plan.steps:
        amount = step.args.get("amount")
        if isinstance(amount, str) and any(
            amount.startswith(f"${ref}.") for ref in history_steps
        ):
            history_steps.add(step.id)
    for step in plan.steps:
        payee = step.args.get("payee")
        if (
            step.tool in {"transfer", "schedule_transfer", "request_payment"}
            and isinstance(payee, str)
            and payee.startswith("<")
            and payee not in text
        ):
            # Money only goes to someone the user named: a compromised planner
            # added a transfer to a contact from the bank context (eval A31).
            return (
                "Who should I send it to? Please tell me the payee's name."
                if en
                else "请问要转给谁？请说出收款人的名字。"
            )
    for step in plan.steps:
        amount = step.args.get("amount")
        if (
            step.tool != "list_transactions"
            and isinstance(amount, str)
            and any(amount.startswith(f"${ref}.") for ref in history_steps)
            and not _BACKREF_RE.search(text)
        ):
            return (
                "How much would you like to transfer?" if en else "请问要转多少金额？"
            )
        if step.tool == "schedule_transfer" and not _RECURRING_RE.search(text):
            return (
                "Do you want to send it once now, or every month?"
                if en
                else "你是想现在转一次，还是每月定期转？"
            )
    return None


_DATE_TOKEN_RE = re.compile(r"<DATE_\d+>")
_AMBIGUOUS_PAYEES_RE = re.compile(r"<PAYEE_\d+>(?:/<PAYEE_\d+>)+")


def _with_usual_amount(
    tokenized: str,
    raw_text: str,
    entries: list[Any],
    bank: Any,
    vault: TokenVault,
    session: Session,
) -> str:
    """"老样子给妈妈转" / "send Mom the usual": add the amount the customer
    confirmed in their memory panel.  Only a confirmed item, only one payee,
    only when no amount was given; the approval card still shows it."""

    if not _USUAL_RE.search(raw_text) or any(e.kind == "AMT" for e in entries):
        return tokenized
    payees = {str(e.real_ref) for e in entries if e.kind == "PAYEE"}
    conn = getattr(bank, "conn", None)
    if len(payees) != 1 or conn is None:
        return tokenized
    amount = habits.usual_amount(conn, str(getattr(bank, "customer_id", "")), payees.pop())
    if amount is None:
        return tokenized
    entry = vault.mint(session, "AMT", amount, tags={"source": "memory"})
    return f"{tokenized} {entry.label}"


def _scam_check_reply(scam: Any, lang: str) -> str:
    """Verdict for "is this a scam?"; the detector decides, never the model."""

    en = _is_en(lang)
    if scam.flags:
        head = (
            "This has clear warning signs of a scam. Don't pay, click links or share codes."
            if en
            else "这条信息有明显的诈骗特征。不要付款、不要点链接、不要告诉对方验证码。"
        )
        advice = scam.advice_en if en else scam.advice_zh
        return f"{head}\n{advice}"
    return (
        "I don't see known scam signs, but stay careful: never share SMS codes, don't "
        "tap links in messages, and call the official number back if unsure. "
        "Anti-fraud hotline: 96110."
        if en
        else "没有发现已知的诈骗特征，但仍请小心：不要透露验证码，不要点消息里的链接，"
        "拿不准就回拨官方电话核实。反诈专线：96110。"
    )


def _wants_fallback(raw: str) -> bool:
    """The fine-tuned planner gave up: unparseable output, or "out of scope"
    for something it was never trained on. Refusals for someone else's data
    or policy overrides stay with the primary planner."""

    try:
        output = parse_planner_output(raw)
    except PlanError:
        return True
    return output.kind == "refuse" and getattr(output, "reason", "") == "out_of_scope"


def _ambiguous_payees(plan: ValidatedPlan, tokenized: str) -> list[str]:
    """The same-name group a money step pays from, e.g. ["<PAYEE_1>", "<PAYEE_2>"]."""

    # A payee also named on its own (e.g. after the customer picked "朋友 张伟")
    # is no longer ambiguous.
    outside = _AMBIGUOUS_PAYEES_RE.sub(" ", tokenized)
    for group in _AMBIGUOUS_PAYEES_RE.findall(tokenized):
        labels = group.split("/")
        for step in plan.steps:
            payee = step.args.get("payee")
            if (
                step.tool in {"transfer", "schedule_transfer", "request_payment"}
                and payee in labels
                and payee not in outside
            ):
                return labels
    return []


def _days_from_text(
    plan: ValidatedPlan, tokenized: str, vault: TokenVault, session: Session, lang: str
) -> tuple[ValidatedPlan, str | None]:
    """Set each standing order's day from the dates the user wrote, in order.

    Returns the (possibly updated) plan and a clarify question when the user
    gave no usable day.
    """
    schedules = [step for step in plan.steps if step.tool == "schedule_transfer"]
    if not schedules:
        return plan, None
    question = (
        "Which day of the month should it go out?"
        if _is_en(lang)
        else "请问每个月几号转？"
    )
    days: list[int] = []
    for label in _DATE_TOKEN_RE.findall(tokenized):
        try:
            value = vault.resolve(label, session).real_ref
        except TokenError:
            continue
        if isinstance(value, str):
            try:
                value = date.fromisoformat(value[:10])
            except ValueError:
                continue
        if isinstance(value, (date, datetime)):
            days.append(value.day)
    if not days or any(day > 28 for day in days):
        return plan, question
    data = plan.model_dump(mode="json")
    index = 0
    for step in data["steps"]:
        if step["tool"] == "schedule_transfer":
            step["args"]["day_of_month"] = days[min(index, len(days) - 1)]
            index += 1
    return ValidatedPlan.model_validate(data), None


def _untrusted_name(text: str) -> str | None:
    """A third-party name (subscription/merchant) made safe for an approval
    card: one short line, no masked-account tails or long digit runs."""

    cleaned = "".join(" " if ord(ch) < 32 else ch for ch in str(text))
    cleaned = re.sub(r"[·•∙⋅]+\s*\d*", " ", cleaned)
    cleaned = re.sub(r"\d{3,}", " ", cleaned)
    cleaned = " ".join(cleaned.split())[:24].strip()
    return cleaned or None


def _int_arg(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _decision_from_dict(data: Mapping[str, Any]) -> StepDecision:
    return StepDecision(
        step_id=str(data.get("step_id") or ""),
        tool=str(data.get("tool") or ""),
        tier=str(data.get("tier") or "green"),  # type: ignore[arg-type]
        needs_mandate=bool(data.get("needs_mandate")),
        needs_mfa=bool(data.get("needs_mfa")),
        needs_co_approval=bool(data.get("needs_co_approval")),
        amount_fen=data.get("amount_fen"),
        deferred=bool(data.get("deferred")),
        # Reasons must survive the round trip: recheck_step relies on the
        # persistent ones (new payee, unverified, ...) to keep the tier.
        reasons=[
            Reason(
                code=str(item.get("code") or ""),
                message_en=str(item.get("message_en") or ""),
                message_zh=str(item.get("message_zh") or ""),
            )
            for item in data.get("reasons") or ()
            if isinstance(item, Mapping)
        ],
        requires_reapproval=bool(data.get("requires_reapproval")),
        vetoed=bool(data.get("vetoed")),
    )


def _record_scam_flags(
    services: Services,
    customer_id: str,
    record: Mapping[str, Any],
    now: datetime,
) -> frozenset[str]:
    """Active scam flags for this session and its customer (customer-wide freeze)."""
    flags: set[str] = set()
    until = record.get("frozen_until")
    if isinstance(until, datetime) and now < until:
        flags.update(record.get("scam_flags") or ())
    flags.update(services.mandates.scam_flags(customer_id, now))
    return frozenset(flags)


def _step_payee(step: PlanStep) -> tuple[str | None, str]:
    """Return ``(token label, real contact id)`` for a step's payee token."""

    value = step.args.get("payee")
    if not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None:
        return None, ""
    return value, ""


def _resolve_payee(
    vault: TokenVault, session: Session, step: PlanStep
) -> tuple[str | None, str]:
    label, _ = _step_payee(step)
    if label is None:
        return None, ""
    try:
        entry = vault.resolve(label, session)
    except TokenError:
        return label, ""
    if entry.kind != "PAYEE":
        return label, ""
    return label, str(entry.real_ref)


def _label_payee(result: dict[str, Any], vault: TokenVault, session: Session) -> None:
    """Use the current plan's PAYEE label so replies never leak an old token.

    A session keeps every previous plan's vault entries.  The responder maps a
    contact id to the first matching entry, which after a prior turn can be a
    closed-plan label that cannot be detokenized in this plan.
    """

    payee = result.get("payee")
    if not isinstance(payee, str) or _TOKEN_RE.fullmatch(payee):
        return
    for entry in vault.entries(session):
        if (
            entry.kind == "PAYEE"
            and entry.plan_id == session.plan_id
            and str(entry.real_ref) == payee
        ):
            result["payee"] = entry.label
            return


def _approval_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "step_id": record.get("step_id"),
        "tool": record.get("tool"),
        "tier": record.get("tier"),
        "amount_fen": record.get("amount_fen"),
        "payee_label": record.get("payee_label"),
        "payee_display": record.get("payee_display"),
        "target_display": record.get("target_display"),
        "day_of_month": record.get("day_of_month"),
        "reasons": list(record.get("reasons") or ()),
        "needs_mfa": bool(record.get("needs_mfa")),
        "needs_co_approval": bool(record.get("needs_co_approval")),
        "demo_otp": record.get("demo_otp"),
        "demo_co_otp": record.get("demo_co_otp"),
        "error": record.get("error"),
    }


def _plan_view(
    plan: ValidatedPlan,
    decisions: Mapping[str, StepDecision],
    statuses: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    statuses = statuses or {}
    view: list[dict[str, Any]] = []
    for step in plan.steps:
        decision = decisions.get(step.id)
        view.append(
            {
                "step_id": step.id,
                "tool": step.tool,
                "tier": decision.tier if decision is not None else "green",
                "status": statuses.get(step.id, "planned"),
                "after": list(step.after),
            }
        )
    return view


def _steps_outcome(
    plan: ValidatedPlan,
    decisions: Mapping[str, StepDecision],
    view: list[dict[str, Any]],
    results: Mapping[str, Any],
    errors: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for step, view_step in zip(plan.steps, view):
        decision = decisions.get(step.id)
        entry = dict(view_step)
        entry["amount_fen"] = decision.amount_fen if decision is not None else None
        if decision is not None:
            entry["needs_mfa"] = decision.needs_mfa
            entry["needs_co_approval"] = decision.needs_co_approval
        if step.id in results:
            entry["result"] = results[step.id]
        if step.id in errors:
            entry["error"] = errors[step.id]
        entries.append(entry)
    return entries


# ----------------------------------------------------------------------- agent


class AnBanAgent:
    """One LangGraph state machine per process, one thread per session."""

    def __init__(self, services: Services) -> None:
        self.services = services
        self.checkpointer = MemorySaver()
        self._approvals: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        self._approvals_applied: dict[tuple[str, str], int] = {}
        self.graph = self._build()

    # ------------------------------------------------------------------ graph

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("intake", self._intake)
        graph.add_node("tokenize", self._tokenize)
        graph.add_node("plan", self._plan)
        graph.add_node("check", self._check)
        graph.add_node("policy", self._policy)
        graph.add_node("approve", self._approve)
        graph.add_node("execute", self._execute)
        graph.add_node("respond", self._respond)

        graph.set_entry_point("intake")
        graph.add_conditional_edges(
            "intake",
            lambda state: "respond" if state.get("outcome") else "tokenize",
            {"respond": "respond", "tokenize": "tokenize"},
        )
        graph.add_conditional_edges(
            "tokenize",
            lambda state: "respond" if state.get("outcome") else "plan",
            {"respond": "respond", "plan": "plan"},
        )
        graph.add_edge("plan", "check")
        graph.add_conditional_edges(
            "check",
            lambda state: "policy" if state.get("plan") else "respond",
            {"policy": "policy", "respond": "respond"},
        )
        graph.add_conditional_edges(
            "policy",
            lambda state: "respond" if state.get("status") == "vetoed" else "approve",
            {"respond": "respond", "approve": "approve"},
        )
        graph.add_edge("approve", "execute")
        graph.add_edge("execute", "respond")
        graph.add_edge("respond", END)
        return graph.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------ nodes

    def _intake(self, state: AgentState) -> AgentState:
        services = self.services
        now = services.now()
        session_id = state["session_id"]
        customer_id = state["customer_id"]
        # Raw text lives only on Services for this turn; it never enters state.
        text = services.turn_text.get(session_id, "")
        record = services.sessions[session_id]

        # A short answer to a clarify turn is merged back into the previous
        # raw text, so the planner sees one complete request.
        pending_clarify = services.clarify.pop(session_id, None)
        if pending_clarify:
            text = _apply_clarify(text, pending_clarify)
            services.turn_text[session_id] = text

        # The thread keeps its previous turn in the checkpoint; explicit
        # ``None`` writes reset turn-scoped channels (omitting a key would
        # leave its old checkpoint value untouched).
        state["tokenized_text"] = None  # type: ignore[typeddict-item]
        state["user_msg"] = None  # type: ignore[typeddict-item]
        state["planner_output"] = None  # type: ignore[typeddict-item]
        state["plan"] = None  # type: ignore[typeddict-item]
        state["decisions"] = []  # type: ignore[typeddict-item]
        state["plan_view"] = []
        state["approvals_pending"] = []
        state["results"] = {}
        state["recent_action"] = None
        state["actions"] = []
        state["outcome"] = None  # type: ignore[typeddict-item]
        state["status"] = ""  # type: ignore[typeddict-item]
        state["reply"] = ""  # type: ignore[typeddict-item]
        state["grounded"] = False  # type: ignore[typeddict-item]
        state["rejected"] = []

        lang = services.turn_lang.get(session_id) or _detect_lang(text)
        record["lang"] = lang
        state["lang"] = lang

        pending_plan_id = str(record.get("pending_plan_id") or "")
        state["plan_id"] = pending_plan_id or str(record.get("plan_id") or "")

        # Scam detection runs before every shortcut below: "客服说…安全账户"
        # must not be swallowed by the handoff or why replies.
        recent = services.recent_raw.setdefault(session_id, [])
        shared_risk = services.customer_risk_texts.get(customer_id, [])
        scam = detect(text, recent_texts=list(recent))
        # Share incomplete evidence across chats, not already-handled warnings.
        # Historical evidence alone must never restart an expired freeze.
        fragments = [item for item in shared_risk if not detect(item).flags]
        combined = detect(text, recent_texts=fragments)
        baseline = detect("", recent_texts=fragments)
        if combined.flags - baseline.flags:
            scam = combined
        recent.append(text)
        services.recent_raw[session_id] = recent[-_MAX_RECENT_TEXTS:]
        services.customer_risk_texts[customer_id] = [*shared_risk, text][-_MAX_RECENT_TEXTS:]
        if "investment_bait" in scam.flags:
            # Before the handoff/undo/freeze replies, which return early.
            services.invest_warned[customer_id] = now + timedelta(hours=24)

        # Only high-risk scripts freeze payments; weaker bait merely warns.
        high_risk = scam.flags & HIGH_RISK_FLAGS
        if (
            high_risk
            and _freeze_until(services, customer_id, record, now) is not None
            and not detect(text).flags & HIGH_RISK_FLAGS
        ):
            # Already frozen and this message is only risky in context: keep
            # the 24h pause running from the latest coaching, but
            # answer the message instead of repeating the same warning.
            services.mandates.freeze_for_scam(customer_id, scam.flags, now)
            record["frozen_until"] = now + timedelta(hours=24)
            scam = detect(text)
            high_risk = scam.flags & HIGH_RISK_FLAGS
        if high_risk:
            if pending_plan_id:
                # A plan waiting for approval must not survive a scam turn.
                services.mandates.revoke_plan(customer_id, pending_plan_id, now)
            record["pending_plan_id"] = ""
            services.mandates.freeze_for_scam(customer_id, scam.flags, now)
            compliance.report_scam_freeze(services.conn, customer_id, sorted(scam.flags), now)
            record["frozen_until"] = now + timedelta(hours=24)
            record["scam_flags"] = sorted(
                set(record.get("scam_flags") or ()) | set(scam.flags)
            )
            plan_id = self._next_plan_id(session_id, record)
            state["plan_id"] = plan_id
            services.audit.append(
                customer_id=customer_id,
                session_id=session_id,
                plan_id=plan_id,
                kind=AuditKind.SCAM,
                payload={"flags": sorted(scam.flags), "score": scam.score},
                now=now,
            )
            state["status"] = "scam"
            state["outcome"] = {
                "kind": "scam",
                "scam": {
                    "flags": sorted(scam.flags),
                    "score": scam.score,
                    "advice_zh": scam.advice_zh,
                    "advice_en": scam.advice_en,
                },
            }
            return state

        # Inspect this turn's risk before releasing anything that became due.
        services.bank_for(customer_id).release_due(now)

        # Unattended standing orders run on every turn; a failure here must
        # never break the chat.
        run_and_audit(services.conn, services.audit, customer_id, now)

        if _SCAM_QUESTION_RE.search(text):
            # High-risk scripts already took the scam path above (freeze and
            # all); here the verdict is "warning signs" or "no known signs".
            if scam.flags:
                record["scam_flags"] = sorted(
                    set(record.get("scam_flags") or ()) | set(scam.flags)
                )
            state["plan_id"] = pending_plan_id or str(record.get("plan_id") or "scam_check")
            state["status"] = "done"
            state["outcome"] = {"kind": "done", "reply": _scam_check_reply(scam, lang)}
            return state

        # Cancel an unconfirmed plan without planning anything.
        if pending_plan_id and _is_bare_cancel(text):
            services.mandates.revoke_plan(customer_id, pending_plan_id, now)
            services.audit.append(
                customer_id=customer_id,
                session_id=session_id,
                plan_id=pending_plan_id,
                kind=AuditKind.MANDATE,
                payload={"action": "cancel"},
                now=now,
            )
            record["pending_plan_id"] = ""
            state["status"] = "done"
            state["outcome"] = {
                "kind": "done",
                "reply": (
                    "OK, cancelled — no money moved."
                    if _is_en(lang)
                    else "好的，已取消，没有扣款。"
                ),
            }
            return state

        # Any new turn supersedes an interrupt that was waiting for approval.
        record["pending_plan_id"] = ""

        if _HANDOFF_RE.search(text):
            plan_id = pending_plan_id or str(record.get("plan_id") or "handoff")
            services.audit.append(
                customer_id=customer_id,
                session_id=session_id,
                plan_id=plan_id,
                kind=AuditKind.HANDOFF,
                payload={"reason": "requested"},
                now=now,
            )
            state["plan_id"] = plan_id
            state["status"] = "paused"
            state["outcome"] = {
                "kind": "paused",
                "reply": (
                    "Your human-support request was recorded. This demo has no "
                    "live staff connection; contact your bank directly for help."
                    if _is_en(lang)
                    else "人工客服请求已记录。本演示未接入真人客服，请直接联系银行获取帮助。"
                ),
            }
            return state

        if _whole_undo_intent(text) and (
            _UNDO_RECENT_RE.search(text) or len(text.strip()) <= 6
        ):
            state["plan_id"] = pending_plan_id or str(record.get("plan_id") or "undo")
            state["status"] = "done"
            state["outcome"] = {
                "kind": "done",
                "reply": (
                    "To undo a transfer, tap \"Undo\" under it in the chat. It works "
                    "until the time shown there; after that the money has left."
                    if _is_en(lang)
                    else "要撤回刚才的转账，请点那笔转账下面的「撤销」按钮，"
                    "在显示的截止时间前都可以撤销；过了时间钱就已经转出。"
                ),
            }
            return state

        if _WHY_RE.search(text):
            state["plan_id"] = pending_plan_id or str(record.get("plan_id") or "why")
            state["status"] = "done"
            state["outcome"] = {
                "kind": "done",
                "reply": _why_reply(services.policy_config, lang),
            }
            return state


        freeze_until = _freeze_until(services, customer_id, record, now)
        if freeze_until is not None and not _READ_ONLY_RE.search(text):
            plan_id = self._next_plan_id(session_id, record)
            state["plan_id"] = plan_id
            services.audit.append(
                customer_id=customer_id,
                session_id=session_id,
                plan_id=plan_id,
                kind=AuditKind.SCAM,
                payload={"frozen": True, "until": freeze_until.isoformat()},
                now=now,
            )
            state["status"] = "frozen"
            state["outcome"] = {
                "kind": "frozen",
                "reply": _frozen_reply(freeze_until, now, lang),
            }
            return state

        if scam.flags:
            plan_id = self._next_plan_id(session_id, record)
            state["plan_id"] = plan_id
            services.audit.append(
                customer_id=customer_id,
                session_id=session_id,
                plan_id=plan_id,
                kind=AuditKind.SCAM,
                payload={
                    "flags": sorted(scam.flags),
                    "score": scam.score,
                    "warn_only": True,
                },
                now=now,
            )
            state["status"] = "scam_warning"
            state["outcome"] = {
                "kind": "scam_warning",
                "reply": scam.advice_en if _is_en(lang) else scam.advice_zh,
            }
            return state

        from agent.send_now_intent import send_now_request, resolve_send_now
        if not pending_plan_id and send_now_request(text):
            selected = resolve_send_now(services, customer_id, session_id, lang)
            state.update(selected)
            state["outcome"] = {"kind": selected["status"], "reply": selected["reply"]}
            return state

        plan_id = self._next_plan_id(session_id, record)
        state["plan_id"] = plan_id
        return state

    def _next_plan_id(self, session_id: str, record: dict[str, Any]) -> str:
        plan_seq = int(record.get("plan_seq") or 0) + 1
        record["plan_seq"] = plan_seq
        # Globally unique: undo records and mandate revocation are keyed by
        # plan id, so a per-session counter collided across sessions (G2, G7).
        plan_id = f"{session_id}:p{plan_seq}"
        record["plan_id"] = plan_id
        return plan_id

    def _tokenize(self, state: AgentState) -> AgentState:
        services = self.services
        now = services.now()
        session = Session(
            session_id=state["session_id"],
            plan_id=state["plan_id"],
            customer_id=state["customer_id"],
        )
        bank = services.bank_for(state["customer_id"])
        raw_text = services.turn_text.get(state["session_id"], "")
        tokenized, entries = tokenize_user_text(
            raw_text, bank, services.vault, session, now
        )
        tokenized = _with_usual_amount(
            tokenized, raw_text, entries, bank, services.vault, session
        )
        try:
            action = recent_actions.build_action(
                conn=getattr(bank, "conn", None),
                secret_key=services.secret_key,
                customer_id=state["customer_id"],
                session_id=state["session_id"],
                session=session,
                vault=services.vault,
                bank=bank,
                raw_text=raw_text,
                user_entries=entries,
                now=now,
            )
        except Exception:  # noqa: BLE001 - context is optional; fail closed
            action = None
        if action is not None:
            tokenized = recent_actions.append_cues(tokenized, action)
        context = build_context(bank, services.vault, session, now)
        if action is not None:
            context.extend(action.context_lines)
        services.audit.append(
            customer_id=state["customer_id"],
            session_id=state["session_id"],
            plan_id=state["plan_id"],
            kind=AuditKind.INTENT,
            payload={"text": tokenized},
            now=now,
        )
        state["tokenized_text"] = tokenized
        state["user_msg"] = render_planner_input(tokenized, context)
        state["recent_action"] = recent_actions.as_state(action)
        unknown_repeat_recipient = bool(action is not None and action.unknown_recipient)
        if unknown_repeat_recipient or (
            action is None
            and recent_actions.explicit_repeat(raw_text)
            and not any(entry.kind in {"PAYEE", "AMT"} for entry in entries)
        ):
            # "Do it again" with no signed transfer to repeat: ask, never guess.
            services.clarify[state["session_id"]] = {
                "raw": raw_text,
                "options": [],
            }
            services.audit.append(
                customer_id=state["customer_id"],
                session_id=state["session_id"],
                plan_id=state["plan_id"],
                kind=AuditKind.VETO,
                payload={"code": "unknown_repeat_recipient" if unknown_repeat_recipient else "no_recent_transfer"},
                now=now,
            )
            state["status"] = "clarify"
            question = (
                'Please say the full request again, with who and how much, for example "send mom 300".'
                if _is_en(state.get("lang") or "en") else "请告诉我收款人和金额。"
            )
            if unknown_repeat_recipient:
                question = (
                    "I couldn't match that recipient to a verified contact. Which saved contact should receive it?"
                    if _is_en(state.get("lang") or "en") else "未找到该收款人，请问要转给哪一位已保存的联系人？"
                )
            state["outcome"] = {
                "kind": "clarify",
                "question": question,
                "reply": question,
                "options": [],
            }
            return state

        # Safe small talk is answered only here: after intake's scam, freeze,
        # cancel, undo, handoff and pending-approval handling and after the
        # unknown-repeat clarify, so it can never bypass a banking gate.
        gateway = services.conversation_gateway
        memory = services.conversation_memory
        lang = state.get("lang") or "en"
        history = ""
        memory_failed = False
        if memory is not None and gateway is not None:
            try:
                # Only the historical block: the current message is never part
                # of the conversation model's input.
                history = memory.render(
                    state["customer_id"],
                    state["session_id"],
                    CONVERSATION_SYSTEM_PROMPT,
                    "",
                )
            except ContextBudgetExceeded:
                return _shorten_request(state)
            except Exception:  # noqa: BLE001 - memory is optional context
                memory_failed = True
        if memory_failed:
            # Broken memory never changes the plain banking path.
            return state
        if gateway is not None:
            try:
                answer = gateway.answer(tokenized, history, lang)
            except Exception:  # noqa: BLE001 - a broken gateway must not break turns
                answer = None
            if isinstance(answer, str) and answer.strip():
                state["status"] = "done"
                state["outcome"] = {"kind": "conversation", "reply": answer}
                return state
        if memory is not None:
            try:
                # Inert, sanitized history is prepended to the planner's user
                # message only; the live current message stays last.
                state["user_msg"] = memory.render(
                    state["customer_id"],
                    state["session_id"],
                    services.planner_system_prompt,
                    state["user_msg"] or "",
                )
            except ContextBudgetExceeded:
                return _shorten_request(state)
            except Exception:  # noqa: BLE001 - fall back to the untouched path
                pass
        return state

    def _plan(self, state: AgentState) -> AgentState:
        services = self.services
        now = services.now()
        try:
            from agent.history_queries import historical_transfer_plan
            read_plan = historical_transfer_plan(
                services.turn_text.get(state["session_id"], ""),
                conn=services.conn, secret_key=services.secret_key,
                customer_id=state["customer_id"], session_id=state["session_id"],
            )
            intent_ctx = None
            if read_plan is None:
                from agent.intents import IntentContext, resolve as resolve_intent
                intent_ctx = IntentContext(
                    raw_text=services.turn_text.get(state["session_id"], ""),
                    tokenized_text=str(state.get("tokenized_text") or ""),
                    conn=services.conn, secret_key=services.secret_key,
                    customer_id=state["customer_id"], session_id=state["session_id"],
                    now=now, lang=state["lang"],
                    vault=services.vault, plan_id=state["plan_id"],
                )
                read_plan = resolve_intent(intent_ctx)
            raw = (json.dumps(read_plan, ensure_ascii=False) if read_plan is not None
                   else services.planner.plan(services.planner_system_prompt, state["user_msg"]))
            if read_plan is None:
                raw = _guard_model_output(services, state, now, raw)
        except Exception:  # noqa: BLE001 - planner transports are untrusted
            services.audit.append(
                customer_id=state["customer_id"],
                session_id=state["session_id"],
                plan_id=state["plan_id"],
                kind=AuditKind.VETO,
                payload={"code": "planner_error"},
                now=now,
            )
            state["status"] = "error"
            state["outcome"] = {"kind": "error", "code": "planner_error"}
            return state

        fallback_used = False
        if services.fallback_planner is not None and _wants_fallback(raw):
            try:
                raw = services.fallback_planner.plan(FALLBACK_SYSTEM_PROMPT, state["user_msg"])
                raw = _guard_model_output(services, state, now, raw)
                fallback_used = True
            except Exception:  # noqa: BLE001 - keep the primary answer
                pass

        state["planner_output"] = raw
        try:
            parsed = parse_planner_output(raw)
            audit_payload: dict[str, Any] = {"output": parsed.model_dump(mode="json")}
        except PlanError:
            audit_payload = {"invalid_json": True, "length": len(raw)}
        if fallback_used:
            audit_payload["fallback"] = True
        services.audit.append(
            customer_id=state["customer_id"],
            session_id=state["session_id"],
            plan_id=state["plan_id"],
            kind=AuditKind.PLAN,
            payload=audit_payload,
            now=now,
        )
        return state

    def _check(self, state: AgentState) -> AgentState:
        services = self.services
        now = services.now()
        session = Session(
            session_id=state["session_id"],
            plan_id=state["plan_id"],
            customer_id=state["customer_id"],
        )
        known_tokens = services.vault.known_tokens(session)
        def check_product_domain(candidate: Any) -> None:
            # A named bank product is not a merchant catalogue purchase. Ask
            # the model to repair the plan; never substitute or execute one.
            current = str(state.get("tokenized_text") or "")
            if re.search(r"<PROD_[0-9]+>", current) and re.search(r"<AMT_[0-9]+>", current):
                tools = {step.tool for step in getattr(candidate, "steps", [])}
                raw_request = services.turn_text.get(state["session_id"], "")
                purchase = re.search(r"\b(?:buy|purchase|invest|subscribe)\b|申购|买入|购买|买", raw_request, re.I)
                redemption = re.search(r"\b(?:redeem|redemption|sell)\b|赎回|卖出", raw_request, re.I)
                wrong_merchant = not re.search(r"<MERCHANT_[0-9]+>", current) and "merchant_order" in tools
                wrong_redemption = purchase and not redemption and "redeem_product" in tools
                if wrong_merchant or wrong_redemption:
                    raise PlanError("tool_domain", "The user requested purchase of a named bank investment product, not merchant goods or redemption. Use subscribe_product with the named product and amount tokens; keep all required policy and approval checks.")
        try:
            output = parse_planner_output(state.get("planner_output") or "")
            check_product_domain(output)
            plan = validate_plan(output, known_tokens)
        except PlanError as first:
            services.audit.append(
                customer_id=state["customer_id"],
                session_id=state["session_id"],
                plan_id=state["plan_id"],
                kind=AuditKind.VETO,
                payload={"code": first.code, "retry": True},
                now=now,
            )
            retry_user = (state.get("user_msg") or "") + (
                "\n\n上一次输出被拒绝 / previous output rejected: "
                f"{first.code}: {_short_detail(first.detail)}. "
                "请只用允许的工具和参数重新输出 JSON。"
            )
            retry_raw: str | None = None
            try:
                retry_raw = services.planner.plan(services.planner_system_prompt, retry_user)
                retry_raw = _guard_model_output(services, state, now, retry_raw)
            except Exception:  # noqa: BLE001 - planner transports are untrusted
                retry_raw = None

            code = first.code
            failed = retry_raw is None
            if retry_raw is not None:
                state["planner_output"] = retry_raw
                try:
                    output = parse_planner_output(retry_raw)
                    check_product_domain(output)
                    plan = validate_plan(output, known_tokens)
                except PlanError as second:
                    code = second.code
                    failed = True
                    services.audit.append(
                        customer_id=state["customer_id"],
                        session_id=state["session_id"],
                        plan_id=state["plan_id"],
                        kind=AuditKind.VETO,
                        payload={"code": second.code, "retry": True},
                        now=now,
                    )
                else:
                    services.audit.append(
                        customer_id=state["customer_id"],
                        session_id=state["session_id"],
                        plan_id=state["plan_id"],
                        kind=AuditKind.PLAN,
                        payload={
                            "output": output.model_dump(mode="json"),
                            "retry": True,
                        },
                        now=now,
                    )
            if failed:
                state["status"] = "error"
                state["outcome"] = {
                    "kind": "error",
                    "code": code,
                    "reply": _planner_error_reply(state.get("lang") or "en"),
                }
                return state

        if plan.kind == "clarify":
            target = plan.output
            options = list(getattr(target, "options", ()) or ())
            services.clarify[state["session_id"]] = {
                "raw": services.turn_text.get(state["session_id"], ""),
                "options": [
                    _option_display(services, session, str(option))
                    for option in options
                ],
            }
            state.pop("plan", None)  # type: ignore[misc]
            state["plan"] = None  # type: ignore[typeddict-item]
            state["status"] = "clarify"
            state["outcome"] = {
                "kind": "clarify",
                "question": getattr(target, "question", ""),
                "options": options,
            }
            return state
        if plan.kind == "refuse":
            state.pop("plan", None)  # type: ignore[misc]
            state["plan"] = None  # type: ignore[typeddict-item]
            state["status"] = "refused"
            state["outcome"] = {
                "kind": "refused",
                "refuse_reason": getattr(plan.output, "reason", "not_supported"),
            }
            return state
        if plan.kind == "scam":
            signals = [
                str(item) for item in (getattr(plan.output, "signals", ()) or ())
            ]
            services.mandates.freeze_for_scam(
                state["customer_id"], {"llm_scam"}, now
            )
            compliance.report_scam_freeze(services.conn, state["customer_id"], ["llm_scam"], now)
            record = services.sessions[state["session_id"]]
            record["frozen_until"] = now + timedelta(hours=24)
            record["scam_flags"] = sorted(
                set(record.get("scam_flags") or ()) | {"llm_scam"}
            )
            services.audit.append(
                customer_id=state["customer_id"],
                session_id=state["session_id"],
                plan_id=state["plan_id"],
                kind=AuditKind.SCAM,
                payload={"flags": ["llm_scam"], "signals": signals},
                now=now,
            )
            state.pop("plan", None)  # type: ignore[misc]
            state["plan"] = None  # type: ignore[typeddict-item]
            state["status"] = "scam"
            state["outcome"] = {
                "kind": "scam",
                "scam": {"signals": signals},
            }
            return state

        # The day of a standing order is the customer's data, not the model's:
        # the model never sees the number behind <DATE_n>, and data v1/v2 only
        # had "15号", so it proposed 15 for "每月1号" (eval F10).
        plan, day_question = _days_from_text(
            plan,
            str(state.get("tokenized_text") or ""),
            services.vault,
            session,
            state.get("lang") or "en",
        )
        if day_question is None and any(
            step.tool == "schedule_transfer" for step in plan.steps
        ):
            services.audit.append(
                customer_id=state["customer_id"],
                session_id=state["session_id"],
                plan_id=state["plan_id"],
                kind=AuditKind.CHECK,
                payload={
                    "day_of_month_from_text": [
                        step.args.get("day_of_month")
                        for step in plan.steps
                        if step.tool == "schedule_transfer"
                    ]
                },
                now=now,
            )
        # Raw words plus the tokenized form, so a date the tokenizer turned
        # into <DATE_n> still counts as a recurrence cue.
        cue_text = " ".join(
            (
                # A label the user typed ("<PAYEE_3>") is not a cue.
                re.sub(r"<[^<>]*>", " ", services.turn_text.get(state["session_id"], "")),
                str(state.get("tokenized_text") or ""),
            )
        )
        options: list[str] = []
        question = _cue_question(plan, cue_text, state.get("lang") or "en") or day_question
        if question is None:
            # The plan may not reuse the undone transfer without permission,
            # may not override an amount/payee the user just wrote, and may not
            # turn one repeat into several transfers.
            question = recent_actions.plan_reuse_question(
                plan, state.get("recent_action"), state.get("lang") or "en"
            )
        if question is None:
            try:
                question = budget_question(
                    plan, services.vault, session, state.get("lang") or "en"
                )
            except ValueError:
                question = (
                    "I couldn't safely quote this basket. Please review its items and budget."
                    if _is_en(state.get("lang") or "en")
                    else "暂时无法安全核算购物篮，请确认商品与预算。"
                )
        ambiguous = _ambiguous_payees(plan, str(state.get("tokenized_text") or ""))
        if question is None and ambiguous:
            # The text names two contacts with the same name and the plan pays
            # one (or both) of them: the customer must choose (eval F05).
            options = ambiguous
            question = (
                "There are two contacts with the same name. Which one should I transfer to?"
                if _is_en(state.get("lang") or "en")
                else "有两个同名联系人，请问要转给哪一位？"
            )
        if question is not None:
            # The plan does something the user's words give no cue for; ask
            # instead of showing it for approval.
            services.clarify[state["session_id"]] = {
                "raw": services.turn_text.get(state["session_id"], ""),
                "options": [
                    _option_display(services, session, option) for option in options
                ],
            }
            services.audit.append(
                customer_id=state["customer_id"],
                session_id=state["session_id"],
                plan_id=state["plan_id"],
                kind=AuditKind.VETO,
                payload={"code": "no_user_cue"},
                now=now,
            )
            state.pop("plan", None)  # type: ignore[misc]
            state["plan"] = None  # type: ignore[typeddict-item]
            state["status"] = "clarify"
            state["outcome"] = {
                "kind": "clarify", "question": question, "options": options,
                **({"reply": question} if not options else {}),
            }
            return state

        state["plan"] = plan.model_dump(mode="json")
        return state

    def _policy(self, state: AgentState) -> AgentState:
        services = self.services
        now = services.now()
        session = Session(
            session_id=state["session_id"],
            plan_id=state["plan_id"],
            customer_id=state["customer_id"],
        )
        plan = ValidatedPlan.model_validate(state["plan"])
        bank = services.bank_for(state["customer_id"])
        record = services.sessions[state["session_id"]]
        decision = evaluate(
            plan,
            bank=bank,
            vault=services.vault,
            session=session,
            now=now,
            config=services.policy_config,
            payments_locked=services.mandates.is_payments_locked(
                state["customer_id"], now
            ),
            scam_flags=_record_scam_flags(
                services, state["customer_id"], record, now
            ),
        )

        if any(reason.code == "R_FRAUD_LIST" for reason in decision.veto_reasons):
            for step in plan.steps:
                if step.tool not in {"transfer", "schedule_transfer"}:
                    continue
                _, contact_id = _resolve_payee(services.vault, session, step)
                if contact_id and compliance.contact_on_fraud_list(
                    services.conn, state["customer_id"], contact_id
                ):
                    compliance.report_fraud_list_hit(
                        services.conn, state["customer_id"], contact_id, now
                    )

        warned_until = services.invest_warned.get(state["customer_id"])
        if (
            not decision.vetoed
            and warned_until is not None
            and now < warned_until
            and any(step.tool == "subscribe_product" for step in plan.steps)
        ):
            # After an investment-bait warning a purchase is refused, not just
            # escalated: a coached victim passes MFA without hesitation.
            decision = replace(
                decision,
                vetoed=True,
                veto_reasons=[
                    Reason(
                        code="R_INVEST_WARN",
                        message_en=(
                            "After the investment-scam warning, product purchases "
                            "are paused for 24 hours."
                        ),
                        message_zh="刚才提示过投资诈骗风险，24 小时内暂停购买理财产品。",
                    )
                ],
            )

        decisions = {item.step_id: item for item in decision.steps}
        state["decisions"] = [asdict(item) for item in decision.steps]
        services.audit.append(
            customer_id=state["customer_id"],
            session_id=state["session_id"],
            plan_id=state["plan_id"],
            kind=AuditKind.POLICY,
            payload={
                "vetoed": decision.vetoed,
                "scam": decision.scam,
                "veto_reasons": [reason.code for reason in decision.veto_reasons],
                "steps": [
                    {
                        "step_id": item.step_id,
                        "tool": item.tool,
                        "tier": item.tier,
                        "amount_fen": item.amount_fen,
                        "rule_codes": [reason.code for reason in item.reasons],
                    }
                    for item in decision.steps
                ],
            },
            now=now,
        )

        if decision.vetoed:
            state["status"] = "vetoed"
            state["outcome"] = {
                "kind": "vetoed",
                "veto_reasons": [asdict(reason) for reason in decision.veto_reasons],
                "steps": _steps_outcome(plan, decisions, _plan_view(plan, decisions), {}, {}),
            }
            return state

        statuses = {
            step.id: ("pending" if decisions[step.id].needs_mandate else "planned")
            for step in plan.steps
        }
        state["plan_view"] = _plan_view(plan, decisions, statuses)
        return state

    def _approve(self, state: AgentState) -> AgentState:
        services = self.services
        now = services.now()
        session_id = state["session_id"]
        plan_id = state["plan_id"]
        customer_id = state["customer_id"]
        session = Session(
            session_id=session_id, plan_id=plan_id, customer_id=customer_id
        )
        plan = ValidatedPlan.model_validate(state["plan"])
        decisions = {
            str(item.get("step_id")): _decision_from_dict(item)
            for item in state.get("decisions") or ()
        }
        cache = self._approvals.setdefault((session_id, plan_id), {})

        # A deferred (step-reference) write amount is only known after its
        # ancestor reads have run.  Run them now so the user is shown - and
        # asked to approve - the real amount instead of ¥0.00.
        deferred_ids = {
            step.id
            for step in plan.steps
            if decisions[step.id].deferred
            and REGISTRY.get(step.tool) is not None
            and REGISTRY[step.tool].kind == "write"
        }
        failures: dict[str, str] = {}
        raw_results: dict[str, Any] = dict(
            services.turn_results.get((session_id, plan_id)) or {}
        )
        for step_id, result in (state.get("results") or {}).items():
            raw_results.setdefault(str(step_id), result)

        if deferred_ids:
            ctx = StepContext(
                bank=services.bank_for(customer_id),
                vault=services.vault,
                session=session,
                mandates=services.mandates,
                audit=services.audit,
                policy_config=services.policy_config,
                now=now,
                scam_flags=_record_scam_flags(
                    services, customer_id, services.sessions[session_id], now
                ),
                payments_locked=services.mandates.is_payments_locked(
                    customer_id, now
                ),
                secret_key=services.secret_key,
            )
            needed_reads = _ancestor_ids(plan, deferred_ids)
            for step in plan.steps:
                if step.id not in needed_reads:
                    continue
                spec = REGISTRY.get(step.tool)
                if spec is None or spec.kind != "read" or step.id in raw_results:
                    continue
                try:
                    result = run_step(step, decisions[step.id], ctx, raw_results)
                except ExecError:
                    continue
                _label_payee(result, services.vault, session)
                raw_results[step.id] = _jsonable(result)

            services.turn_results[(session_id, plan_id)] = raw_results
            state["results"] = {
                step_id: _public_result(result)
                for step_id, result in raw_results.items()
            }

            # Same-day schedules in this plan are not in the bank yet, so the
            # per-day limit must count them (non-deferred ones and deferred
            # ones resolved earlier in this loop).
            scheduled_by_day: dict[int, int] = {}
            for step in plan.steps:
                if (
                    step.tool == "schedule_transfer"
                    and step.id not in deferred_ids
                    and decisions[step.id].amount_fen
                ):
                    day = _int_arg(step.args.get("day_of_month"))
                    if day is not None:
                        scheduled_by_day[day] = scheduled_by_day.get(day, 0) + int(
                            decisions[step.id].amount_fen or 0
                        )
            for step in plan.steps:
                if step.id not in deferred_ids:
                    continue
                decision = decisions[step.id]
                day = _int_arg(step.args.get("day_of_month"))
                try:
                    resolved = resolve_args(
                        step, services.vault, session, raw_results
                    )
                except (ExecError, TypeError, ValueError):
                    resolved = {}
                amount = (
                    resolved.get("amount")
                    if isinstance(resolved, Mapping)
                    else None
                )
                if (
                    not isinstance(amount, int)
                    or isinstance(amount, bool)
                    or amount <= 0
                ):
                    # Never ask the user to approve a blank or ¥0 amount.
                    failures[step.id] = "amount_unknown"
                    continue
                try:
                    checked = recheck_step(
                        decision,
                        resolved_amount_fen=amount,
                        bank=ctx.bank,
                        now=now,
                        approved_tier=decision.tier,
                        config=services.policy_config,
                        payments_locked=ctx.payments_locked,
                        scam_flags=ctx.scam_flags,
                        day_of_month=day,
                        scheduled_pending_fen=scheduled_by_day.get(day, 0)
                        if day is not None
                        else 0,
                    )
                except (OverflowError, ValueError):
                    failures[step.id] = "amount_unknown"
                    continue
                decisions[step.id] = checked
                if step.tool == "schedule_transfer" and day is not None:
                    scheduled_by_day[day] = scheduled_by_day.get(day, 0) + amount
            state["decisions"] = [
                asdict(decisions[step.id]) for step in plan.steps
            ]

        for step in plan.steps:
            decision = decisions[step.id]
            if (
                step.id in failures
                or not decision.needs_mandate
                or step.id in cache
            ):
                continue
            cache[step.id] = self._issue_mandate(
                services, step, decision, session, now
            )

        if not cache and not failures:
            return state

        applied = self._approvals_applied.setdefault((session_id, plan_id), 0)
        round_no = 0
        pending = [
            record
            for record in cache.values()
            if record.get("status") == "pending"
        ]
        while pending:
            payload = {
                "approvals": [_approval_payload(record) for record in pending],
                "plan_view": state.get("plan_view") or [],
            }
            resumed = interrupt(payload)

            if round_no < applied:
                # Replaying a resume value that was already applied.  Never
                # consume another OTP attempt for the same answer.
                round_no += 1
                continue

            self._apply_approvals(
                services, cache, plan, decisions, resumed, customer_id, now
            )
            applied += 1
            self._approvals_applied[(session_id, plan_id)] = applied
            round_no += 1

            pending = [
                record
                for record in cache.values()
                if record.get("status") == "pending"
            ]
            if not pending or round_no >= _MAX_APPROVAL_ROUNDS:
                break

        statuses: dict[str, str] = {}
        for step in plan.steps:
            record = cache.get(step.id)
            if step.id in failures:
                statuses[step.id] = "failed"
            elif step.id in raw_results:
                statuses[step.id] = "done"
            elif record is None:
                statuses[step.id] = "planned"
            elif record.get("status") == "approved":
                statuses[step.id] = "ready"
            elif record.get("status") == "rejected":
                statuses[step.id] = "skipped"
            elif record.get("status") == "locked":
                statuses[step.id] = "failed"
            else:
                statuses[step.id] = "pending"
        state["plan_view"] = _plan_view(plan, decisions, statuses)
        state["approvals_pending"] = [
            _approval_payload(record)
            for record in cache.values()
            if record.get("status") == "pending"
        ]
        return state

    def _issue_mandate(
        self,
        services: Services,
        step: PlanStep,
        decision: StepDecision,
        session: Session,
        now: datetime,
    ) -> dict[str, Any]:
        payee_label, payee_ref = _resolve_payee(services.vault, session, step)
        if step.tool == "redeem_product":
            product_entry = services.vault.resolve(step.args["product"], session)
            if product_entry.kind != "PROD":
                raise TokenError("wrong product token")
            payee_ref = "product:" + str(product_entry.real_ref)
        payee_display = None
        if payee_label:
            # Clients get a masked display name ("妈妈 ···0189"), never a token.
            payee_display = detokenize(
                payee_label, services.vault, session, services.bank_for(session.customer_id)
            )
        # Say what the step acts on ("安享稳健 90 天", "爱奇艺 VIP", ···0189),
        # so an approval card never hides a wrong product/subscription/card.
        target_display = None
        for key in ("product", "subscription", "sub", "card", "merchant"):
            value = step.args.get(key)
            if not isinstance(value, str) or not _TOKEN_RE.fullmatch(value):
                continue
            display = detokenize(
                value, services.vault, session, services.bank_for(session.customer_id)
            )
            # An unresolvable token stays as "<KIND_N>"; never show that.
            if not _TOKEN_RE.fullmatch(display):
                target_display = (
                    _untrusted_name(display)
                    if key in {"subscription", "sub", "merchant"}
                    else display
                )
            break
        day_of_month = (
            step.args.get("day_of_month") if step.tool == "schedule_transfer" else None
        )
        record: dict[str, Any] = {
            "step_id": step.id,
            "tool": step.tool,
            "tier": decision.tier,
            "amount_fen": decision.amount_fen or 0,
            "payee_label": payee_label,
            "payee_display": payee_display,
            "target_display": target_display,
            "day_of_month": day_of_month,
            # Why this tier, in plain words (e.g. "today's total is over ¥1,000").
            "reasons": [
                {"message_zh": r.message_zh, "message_en": r.message_en}
                for r in decision.reasons
                if getattr(r, "code", "") != "R_BASE"
            ],
            "needs_mfa": bool(decision.needs_mfa),
            "needs_co_approval": bool(decision.needs_co_approval),
            "mandate_id": None,
            "demo_otp": None,
            "demo_co_otp": None,
            "approver_id": None,
            "mfa_verified": False,
            "status": "pending",
            "error": None,
        }
        try:
            mandate = services.mandates.issue(
                customer_id=session.customer_id,
                session_id=session.session_id,
                plan_id=session.plan_id,
                step_id=step.id,
                action=step.tool,
                payee_ref=payee_ref,
                amount_fen=int(decision.amount_fen or 0),
                tier=decision.tier,
                now=now,
                co_approval_required=decision.needs_co_approval,
            )
        except MandateError as exc:
            record["error"] = exc.code
            return record

        record["mandate_id"] = mandate.id
        record["mfa_verified"] = bool(mandate.mfa_verified)

        if decision.needs_mfa:
            try:
                record["demo_otp"] = services.mandates.start_mfa(
                    mandate.id, session.customer_id, now
                )
            except MandateError as exc:
                record["error"] = exc.code

        if decision.needs_co_approval:
            approvers = services.mandates.co_approvers(session.customer_id)
            if not approvers:
                record["error"] = record["error"] or "no_co_approver"
            else:
                record["approver_id"] = approvers[0]
                try:
                    record["demo_co_otp"] = services.mandates.start_co_approval(
                        mandate.id, session.customer_id, approvers[0], now
                    )
                except MandateError as exc:
                    record["error"] = exc.code

        return record

    def _apply_approvals(
        self,
        services: Services,
        cache: Mapping[str, dict[str, Any]],
        plan: ValidatedPlan,
        decisions: Mapping[str, StepDecision],
        resumed: Any,
        customer_id: str,
        now: datetime,
    ) -> None:
        approvals: Mapping[str, Any] = {}
        if isinstance(resumed, Mapping):
            inner = resumed.get("approvals")
            approvals = inner if isinstance(inner, Mapping) else resumed

        for step in plan.steps:
            record = cache.get(step.id)
            if record is None or record.get("status") != "pending":
                continue
            action = approvals.get(step.id)
            if not isinstance(action, Mapping):
                continue
            # Strict booleans: only an explicit reject rejects and only an
            # explicit confirm approves. {"confirm": false}, {} or any other
            # shape leaves the step pending.
            if action.get("reject") is True:
                record["status"] = "rejected"
                record["error"] = "rejected"
                continue
            if action.get("confirm") is not True:
                continue

            mandate_id = record.get("mandate_id")
            if not mandate_id:
                continue

            try:
                self._apply_one_mandate(
                    services, record, action, customer_id, now
                )
            except MandateError as exc:
                record["error"] = exc.code
                if exc.code == "locked":
                    # Payments are locked and the mandate revoked: the card is
                    # finished, so don't show it again as if it could pass (B6).
                    record["status"] = "locked"
            except (OverflowError, TypeError, ValueError):
                record["error"] = "invalid"

    def _apply_one_mandate(
        self,
        services: Services,
        record: dict[str, Any],
        action: Mapping[str, Any],
        customer_id: str,
        now: datetime,
    ) -> None:
        mandate_id = str(record["mandate_id"])

        if record.get("needs_mfa") and not record.get("mfa_verified"):
            otp = action.get("otp")
            retyped = action.get("retyped_amount_fen")
            if not isinstance(otp, str) or not otp:
                record["error"] = "otp_missing"
                return
            if not isinstance(retyped, int) or isinstance(retyped, bool):
                record["error"] = "amount_missing"
                return
            services.mandates.verify_mfa(
                mandate_id,
                customer_id,
                otp=otp,
                retyped_amount_fen=retyped,
                face_ok=bool(action.get("face_ok")),
                now=now,
            )
            record["mfa_verified"] = True
            record["error"] = None

        if record.get("needs_co_approval"):
            approver_id = record.get("approver_id")
            if not approver_id:
                record["error"] = "no_co_approver"
                return
            co_otp = action.get("co_otp")
            if not isinstance(co_otp, str) or not co_otp:
                record["error"] = "co_otp_missing"
                return
            services.mandates.co_approve(
                mandate_id, customer_id, str(approver_id), co_otp, now
            )
            record["status"] = "approved"
            record["error"] = None
            return

        if record.get("needs_mfa"):
            record["status"] = "approved"
            record["error"] = None
            return

        services.mandates.approve_yellow(mandate_id, customer_id, now)
        record["status"] = "approved"
        record["error"] = None

    def _execute(self, state: AgentState) -> AgentState:
        services = self.services
        now = services.now()
        session = Session(
            session_id=state["session_id"],
            plan_id=state["plan_id"],
            customer_id=state["customer_id"],
        )
        plan = ValidatedPlan.model_validate(state["plan"])
        decisions = {
            str(item.get("step_id")): _decision_from_dict(item)
            for item in state.get("decisions") or ()
        }
        record = services.sessions[state["session_id"]]
        bank = services.bank_for(state["customer_id"])
        ctx = StepContext(
            bank=bank,
            vault=services.vault,
            session=session,
            mandates=services.mandates,
            audit=services.audit,
            policy_config=services.policy_config,
            now=now,
            scam_flags=_record_scam_flags(
                services, state["customer_id"], record, now
            ),
            payments_locked=services.mandates.is_payments_locked(
                state["customer_id"], now
            ),
            secret_key=services.secret_key,
        )

        cache = self._approvals.get((state["session_id"], state["plan_id"]), {})
        previous = {
            item["step_id"]: item
            for item in state.get("plan_view") or ()
            if isinstance(item, Mapping) and item.get("step_id")
        }
        results: dict[str, Any] = dict(
            services.turn_results.get((state["session_id"], state["plan_id"])) or {}
        )
        for step_id, result in (state.get("results") or {}).items():
            results.setdefault(str(step_id), result)
        errors: dict[str, str] = {}
        statuses: dict[str, str] = {}
        blocked: set[str] = set()
        pending_approvals: list[dict[str, Any]] = []

        for step in plan.steps:
            decision = decisions[step.id]
            previous_status = str(previous.get(step.id, {}).get("status") or "")

            if previous_status == "skipped":
                statuses[step.id] = "skipped"
                errors[step.id] = "rejected"
                blocked.add(step.id)
                continue
            if previous_status == "failed":
                approval = (
                    self._approvals.get((state["session_id"], state["plan_id"])) or {}
                ).get(step.id) or {}
                statuses[step.id] = "failed"
                errors[step.id] = (
                    "locked" if approval.get("status") == "locked" else "amount_unknown"
                )
                blocked.add(step.id)
                continue
            if any(dependency in blocked for dependency in step.after):
                statuses[step.id] = "skipped"
                errors[step.id] = "dependency"
                blocked.add(step.id)
                continue
            warned_until = services.invest_warned.get(state["customer_id"])
            if (
                step.tool == "subscribe_product"
                and warned_until is not None
                and services.now() < warned_until
            ):
                # Re-checked at run time: a purchase approved in another
                # session after the warning must not go through.
                statuses[step.id] = "failed"
                errors[step.id] = "invest_warned"
                blocked.add(step.id)
                continue

            spec = REGISTRY.get(step.tool)
            if spec is not None and spec.kind == "read" and step.id in results:
                # Ancestor reads already ran during approval (G8).
                statuses[step.id] = "done"
                continue

            approval = cache.get(step.id)
            if previous_status == "pending" or (
                decision.needs_mandate
                and (approval is None or approval.get("status") != "approved")
            ):
                statuses[step.id] = "pending"
                if approval is not None:
                    pending_approvals.append(_approval_payload(approval))
                else:
                    pending_approvals.append(
                        {
                            "step_id": step.id,
                            "tool": step.tool,
                            "tier": decision.tier,
                            "amount_fen": decision.amount_fen,
                            "payee_label": step.args.get("payee"),
                            "needs_mfa": decision.needs_mfa,
                            "needs_co_approval": decision.needs_co_approval,
                            "demo_otp": None,
                            "demo_co_otp": None,
                            "error": "not_approved",
                        }
                    )
                blocked.add(step.id)
                continue

            mandate_id = (
                str(approval.get("mandate_id"))
                if approval is not None and approval.get("mandate_id")
                else None
            )
            if decision.needs_mandate and not mandate_id:
                statuses[step.id] = "pending"
                pending_approvals.append(
                    {
                        "step_id": step.id,
                        "tool": step.tool,
                        "tier": decision.tier,
                        "amount_fen": decision.amount_fen,
                        "payee_label": step.args.get("payee"),
                        "needs_mfa": decision.needs_mfa,
                        "needs_co_approval": decision.needs_co_approval,
                        "demo_otp": None,
                        "demo_co_otp": None,
                        "error": "mandate_missing",
                    }
                )
                blocked.add(step.id)
                continue

            try:
                result = run_step(
                    step,
                    decision,
                    ctx,
                    results,
                    mandate_id,
                    idempotency_key=(
                        f"{state['session_id']}:{state['plan_id']}:{step.id}"
                    ),
                )
            except ExecError as exc:
                if exc.code == "reapproval_required":
                    statuses[step.id] = "pending"
                    pending_approvals.append(
                        {
                            "step_id": step.id,
                            "tool": step.tool,
                            "tier": decision.tier,
                            "amount_fen": decision.amount_fen,
                            "payee_label": step.args.get("payee"),
                            "needs_mfa": decision.needs_mfa,
                            "needs_co_approval": decision.needs_co_approval,
                            "demo_otp": None,
                            "demo_co_otp": None,
                            "error": exc.code,
                        }
                    )
                    blocked.add(step.id)
                else:
                    statuses[step.id] = "failed"
                    errors[step.id] = exc.code
                    blocked.add(step.id)
                continue

            _label_payee(result, services.vault, session)
            results[step.id] = _jsonable(result)
            statuses[step.id] = "done"
            if step.tool == "can_cover" and result.get("ok") is False:
                # A failed coverage check must stop the transfers that depend
                # on it; the step itself still reports its result.
                blocked.add(step.id)

        # Keep the full bank rows on Services for the responder/executor; the
        # checkpoint only ever sees the public, raw-free projection.
        services.turn_results[(state["session_id"], state["plan_id"])] = results
        public_results = {
            step_id: _public_result(result) for step_id, result in results.items()
        }

        view = _plan_view(plan, decisions, statuses)
        state["results"] = public_results
        state["plan_view"] = view
        steps = _steps_outcome(plan, decisions, view, public_results, errors)

        if pending_approvals:
            state["status"] = "needs_approval"
            state["approvals_pending"] = pending_approvals
            state["outcome"] = {
                "kind": "needs_approval",
                "approvals": pending_approvals,
                "steps": steps,
            }
        else:
            state["status"] = "done"
            state["approvals_pending"] = []
            state["outcome"] = {"kind": "done", "steps": steps}
        if any(
            statuses.get(step.id) == "done" and step.tool in {"transfer", "merchant_order", "subscribe_product"}
            for step in plan.steps
        ):
            # 大额交易报告: filed to the (SIMULATED) regulator queue only; the
            # customer is never told (no tipping off).
            compliance.check_large_transactions(
                services.conn,
                state["customer_id"],
                int(services.bank_for(state["customer_id"]).daily_outgoing_total(now)),
                now,
            )
        return state

    def _respond(self, state: AgentState) -> AgentState:
        services = self.services
        now = services.now()
        session = Session(
            session_id=state["session_id"],
            plan_id=state["plan_id"],
            customer_id=state["customer_id"],
        )
        bank = services.bank_for(state["customer_id"])
        outcome = state.get("outcome") or {"kind": state.get("status") or "done"}
        outcome = _outcome_with_raw(
            outcome,
            services.turn_results.get((state["session_id"], state["plan_id"])),
        )
        direct = outcome.get("reply") if isinstance(outcome, Mapping) else None
        conversation = (
            isinstance(outcome, Mapping) and outcome.get("kind") == "conversation"
        )
        if isinstance(direct, str) and direct:
            # Canned/unrecoverable turns carry their final text verbatim so a
            # generic responder branch cannot replace it.  A conversation reply
            # is not bank evidence, so it is never marked grounded.
            reply: ReplyResult = ReplyResult(
                text=direct, grounded=not conversation, rejected=[]
            )
        else:
            reply = compose_reply(
                outcome,
                lang=state.get("lang") or "en",
                vault=services.vault,
                session=session,
                bank=bank,
                now=now,
            )
        services.audit.append(
            customer_id=state["customer_id"],
            session_id=state["session_id"],
            plan_id=state["plan_id"],
            kind=AuditKind.GROUND,
            payload={
                "grounded": reply.grounded,
                "rejected_count": len(reply.rejected),
            },
            now=now,
        )
        try:
            tokenized_reply, _ = tokenize_untrusted(
                reply.text, bank, services.vault, session, now
            )
        except Exception:  # noqa: BLE001 - audit must never break a reply
            tokenized_reply = ""
        services.audit.append(
            customer_id=state["customer_id"],
            session_id=state["session_id"],
            plan_id=state["plan_id"],
            kind=AuditKind.REPLY,
            payload={"text": tokenized_reply},
            now=now,
        )

        # Remember the masked user text and a code-built, fact-free summary.
        # A pending approval is remembered when it finishes, so one turn is
        # never stored twice.
        memory = services.conversation_memory
        tokenized_text = state.get("tokenized_text")
        if (
            memory is not None
            and isinstance(tokenized_text, str)
            and tokenized_text
            and state.get("status") != "needs_approval"
        ):
            try:
                memory.remember(
                    state["customer_id"],
                    state["session_id"],
                    tokenized_text,
                    _assistant_summary(state, outcome, reply.text),
                )
            except Exception:  # noqa: BLE001 - memory must never break a reply
                pass
        services.vault.close_plan(session)

        # A turn that still waits for approval resumes later; only a final
        # turn releases its full step results.
        if state.get("status") != "needs_approval":
            services.turn_results.pop(
                (state["session_id"], state["plan_id"]), None
            )

        state["reply"] = reply.text
        state["grounded"] = reply.grounded
        state["rejected"] = list(reply.rejected)
        return state

    # --------------------------------------------------------------- public API

    def start(
        self,
        customer_id: str,
        text: str,
        session_id: str | None = None,
        lang: str | None = None,
    ) -> TurnResult:
        session_id = session_id or secrets.token_hex(16)
        record = self.services.sessions.get(session_id)
        if record is not None and record.get("customer_id") != customer_id:
            return TurnResult(status="error", reply="not found", session_id=session_id)
        if record is None:
            record = self._new_session(session_id, customer_id)

        config = {"configurable": {"thread_id": session_id}}
        pending_plan = str(record.get("pending_plan_id") or "")
        if pending_plan and _BARE_CONFIRM_RE.fullmatch(str(text).strip()):
            # A typed "yes" is not an approval (that needs the card's Confirm,
            # plus codes for red steps).  Planning "yes" as a new request lost
            # the pending plan or invented a transfer, so show
            # the pending approvals again without running the graph.
            again = self._pending_again(config, session_id, customer_id, text, pending_plan)
            if again is not None:
                return again
        turn_bank = self.services.bank_for(customer_id)
        bank_conn = getattr(turn_bank, "conn", None)
        if bank_conn is not None:
            # Lowered protections take effect only after their 24 h wait, and
            # transfers past their hold reach the ledger.
            compliance.apply_due(bank_conn, customer_id, self.services.now())
        stripped = str(text).strip()
        if (
            not pending_plan
            and len(stripped) <= 40
            and not _SHORTCUT_FILLER_RE.sub("", stripped)
            and not detect(stripped).flags
        ):
            typed: TurnResult | None = None
            if (
                _UNDO_RE.search(stripped)
                and (_UNDO_RECENT_RE.search(stripped) or len(stripped) <= 6)
                and not recent_actions.explicit_repeat(stripped)
            ):
                typed = self._typed_undo(session_id, customer_id, lang, stripped)
            elif _RECALL_RE.search(stripped):
                typed = self._typed_recall(session_id, customer_id, lang)
            if typed is not None:
                return typed
        self.services.turn_text[session_id] = str(text)
        if lang in {"en", "zh"}:
            self.services.turn_lang[session_id] = lang
        try:
            result = self.graph.invoke(
                {
                    "session_id": session_id,
                    "customer_id": customer_id,
                },
                config,
            )
        except Exception:  # noqa: BLE001 - never leak internals to the caller
            failed_plan = str(record.get("plan_id") or "")
            self.services.turn_results.pop(
                (session_id, failed_plan), None
            )
            # An unexpected failure must not leave live tokens or approval
            # mandates for the next turn. Ledger results remain authoritative.
            self.services.vault.close_plan(Session(
                session_id=session_id, plan_id=failed_plan, customer_id=customer_id
            ))
            try:
                self.services.mandates.revoke_plan(customer_id, failed_plan, self.services.now())
            except Exception:  # noqa: BLE001 - still return a safe error
                pass
            record["pending_plan_id"] = ""
            return TurnResult(
                status="error",
                reply=(
                    "I couldn't confirm the result. Please check transaction history before trying again."
                    if _is_en(record.get("lang") or "en")
                    else "暂时无法确认操作结果，请先查看交易记录，再决定是否重试。"
                ),
                session_id=session_id,
                plan_id=str(record.get("plan_id") or ""),
            )
        finally:
            self.services.turn_text.pop(session_id, None)
            self.services.turn_lang.pop(session_id, None)
        return self._turn_result(result, session_id, customer_id)

    def _pending_again(
        self,
        config: Mapping[str, Any],
        session_id: str,
        customer_id: str,
        text: str,
        plan_id: str,
    ) -> TurnResult | None:
        # Only re-show a card that can still be confirmed; otherwise let the
        # graph give the frozen / locked / expired reply.
        services = self.services
        now = services.now()
        record = services.sessions.get(session_id) or {}
        if _freeze_until(
            services, customer_id, record, now
        ) is not None or services.mandates.is_payments_locked(customer_id, now):
            return None
        for item in (self._approvals.get((session_id, plan_id)) or {}).values():
            mandate_id = item.get("mandate_id")
            if item.get("status") != "pending" or not mandate_id:
                continue
            try:
                mandate = services.mandates.get(str(mandate_id), customer_id)
            except Exception:  # noqa: BLE001 - unknown or tampered mandate
                return None
            if mandate.status not in OPEN_STATUSES or now >= mandate.expires_at:
                return None
        snapshot = self.graph.get_state(config)
        for task in getattr(snapshot, "tasks", ()) or ():
            for item in getattr(task, "interrupts", ()) or ():
                payload = getattr(item, "value", item)
                if not isinstance(payload, Mapping):
                    continue
                approvals = list(payload.get("approvals") or ())
                if not approvals:
                    continue
                hint = (
                    "Please tap Confirm on the card to go ahead - nothing has been done yet."
                    if _is_en(_detect_lang(text))
                    else "请在确认卡片上点「确认」继续，目前还没有执行任何操作。"
                )
                return TurnResult(
                    status="needs_approval",
                    reply=hint
                    + "\n"
                    + self._pending_reply(session_id, customer_id, approvals),
                    approvals=approvals,
                    plan_view=list(payload.get("plan_view") or ()),
                    session_id=session_id,
                    plan_id=plan_id,
                )
        return None

    def resume(
        self,
        session_id: str,
        customer_id: str,
        approvals: Mapping[str, Any],
    ) -> TurnResult:
        record = self.services.sessions.get(session_id)
        if record is None or record.get("customer_id") != customer_id:
            return TurnResult(status="error", reply="not found", session_id=session_id)
        config = {"configurable": {"thread_id": session_id}}
        answers = dict(approvals or {})
        if isinstance(answers.get("approvals"), Mapping):
            answers = dict(answers["approvals"])
        try:
            result = self.graph.invoke(
                Command(resume={"approvals": answers}), config
            )
        except Exception:  # noqa: BLE001
            return TurnResult(
                status="error",
                reply="not found",
                session_id=session_id,
                plan_id=str(record.get("plan_id") or ""),
            )
        return self._turn_result(result, session_id, customer_id)

    def undo(
        self,
        session_id: str,
        customer_id: str,
        plan_id: str,
        step_id: str,
        mandate_id: str | None = None,
        lang: str | None = None,
    ) -> TurnResult:
        record = self.services.sessions.get(session_id)
        if record is None or record.get("customer_id") != customer_id:
            return TurnResult(status="error", reply="not found", session_id=session_id)
        # The plan must belong to this session (plan ids are "<session_id>:pN").
        if str(plan_id).rsplit(":", 1)[0] != session_id:
            return TurnResult(status="error", reply="not found", session_id=session_id)
        session = Session(
            session_id=session_id, plan_id=plan_id, customer_id=customer_id
        )
        bank = self.services.bank_for(customer_id)
        now = self.services.now()
        ctx = StepContext(
            bank=bank,
            vault=self.services.vault,
            session=session,
            mandates=self.services.mandates,
            audit=self.services.audit,
            policy_config=self.services.policy_config,
            now=now,
            scam_flags=_record_scam_flags(
                self.services, customer_id, record, now
            ),
            secret_key=self.services.secret_key,
        )
        try:
            result = undo_step(plan_id, step_id, ctx, mandate_id)
        except ExecError as exc:
            closed = "window_closed" in (exc.code, exc.detail) or "window" in exc.detail
            zh = self._lang(session_id, lang) != "en"
            actions: list[str] = []
            if exc.detail == "replayed":
                text = "这笔已经撤销过了。" if zh else "That was already undone."
                receipt = undo_record(bank.conn, customer_id, plan_id, step_id)
                if receipt and receipt.get("tool") == "cancel_transfer":
                    args = json.loads(receipt["args_json"])
                    transfer = bank.conn.execute(
                        "SELECT status FROM transfers WHERE customer_id = ? AND id = ?",
                        (customer_id, args.get("transfer_id") or args.get("id")),
                    ).fetchone()
                    if transfer and transfer[0] == "released":
                        text = "这笔钱已经转出，无法撤销。" if zh else "This transfer has already been sent. Undo is no longer available."
            elif closed:
                text = "撤销时间已过，钱已经转出，无法撤回。" if zh else (
                    "The undo window has closed; the money has left and can't be pulled back."
                )
                row = undo_record(bank.conn, customer_id, plan_id, step_id)
                if row is not None and row.get("tool") == "cancel_transfer":
                    actions = ["recall"]
                    text += (
                        "我可以帮你向收款银行提交追回申请（退汇），请对方自愿退回，"
                        "但不保证能追回。需要的话请点「申请追回」或回复「追回」。"
                        if zh
                        else " I can send the receiving bank a recall request asking "
                        "the payee to return it voluntarily; it is not guaranteed. "
                        "Tap \"Request recall\" or type \"recall\"."
                    )
            else:
                text = "这一步无法撤销。" if zh else "This step can't be undone."
            return TurnResult(
                status="error",
                reply=text,
                session_id=session_id,
                plan_id=plan_id,
                actions=actions,
            )
        outcome = {
            "kind": "done",
            "steps": [
                {
                    "step_id": step_id,
                    "tool": (result or {}).get("tool", ""),
                    "status": "done",
                    "result": {"raw": _jsonable(result)},
                }
            ],
        }
        reply = compose_reply(
            outcome,
            lang=self._lang(session_id, lang),
            vault=self.services.vault,
            session=session,
            bank=bank,
            now=now,
        )
        return TurnResult(
            status="done",
            reply=reply.text,
            session_id=session_id,
            plan_id=plan_id,
        )

    def pause(self, session_id: str, customer_id: str) -> TurnResult:
        record = self.services.sessions.get(session_id)
        if record is None or record.get("customer_id") != customer_id:
            return TurnResult(status="error", reply="not found", session_id=session_id)
        plan_id = str(record.get("plan_id") or "")
        now = self.services.now()
        if plan_id:
            self.services.mandates.revoke_plan(customer_id, plan_id, now)
        session = Session(
            session_id=session_id, plan_id=plan_id, customer_id=customer_id
        )
        bank = self.services.bank_for(customer_id)
        reply = compose_reply(
            {"kind": "paused"},
            lang=self._lang(session_id),
            vault=self.services.vault,
            session=session,
            bank=bank,
            now=now,
        )
        self.services.vault.close_plan(session)
        return TurnResult(
            status="paused",
            reply=reply.text,
            session_id=session_id,
            plan_id=plan_id,
        )

    def handoff(self, session_id: str, customer_id: str) -> TurnResult:
        record = self.services.sessions.get(session_id)
        if record is None or record.get("customer_id") != customer_id:
            return TurnResult(status="error", reply="not found", session_id=session_id)
        plan_id = str(record.get("plan_id") or "")
        self.services.audit.append(
            customer_id=customer_id,
            session_id=session_id,
            plan_id=plan_id or "handoff",
            kind=AuditKind.HANDOFF,
            payload={"reason": "requested"},
            now=self.services.now(),
        )
        lang = self._lang(session_id)
        reply = "人工客服请求已记录。本演示未接入真人客服，请直接联系银行。" if lang != "en" else "Human-support request recorded. This demo has no live staff connection; contact your bank directly."
        return TurnResult(
            status="handoff",
            reply=reply,
            session_id=session_id,
            plan_id=plan_id,
        )

    # ---------------------------------------------------------------- internals

    def _new_session(self, session_id: str, customer_id: str) -> dict[str, Any]:
        record: dict[str, Any] = {
            "customer_id": customer_id,
            "frozen_until": None,
            "scam_flags": [],
            "plan_seq": 0,
            "plan_id": "",
            "pending_plan_id": "",
        }
        self.services.sessions[session_id] = record
        return record

    def _lang(self, session_id: str, override: str | None = None) -> str:
        if override in {"en", "zh"}:
            return str(override)
        record = self.services.sessions.get(session_id) or {}
        return str(record.get("lang") or "en")

    def _typed_undo(
        self, session_id: str, customer_id: str, lang: str | None, text: str = ""
    ) -> TurnResult | None:
        """Typed "undo that": run the button's undo on the newest step here."""

        bank = self.services.bank_for(customer_id)
        row = latest_undo(getattr(bank, "conn", None), customer_id, session_id)
        if row is None:
            return None  # the graph explains where Undo lives
        lowered = text.lower()
        named = {
            tool
            for tool, words in _UNDO_KIND_WORDS.items()
            if any(word in lowered for word in words)
        }
        if named and str(row.get("tool")) not in named:
            return None  # names something else; the button is unambiguous
        zh = self._lang(session_id, lang) != "en"
        if row.get("tool") not in _TYPED_UNDO_TOOLS:
            # Loosening a protection or restarting a charge needs a fresh,
            # explicit request with its own confirmation.
            return TurnResult(
                status="done",
                reply="这一步不能直接撤销，请重新提出要求，我会请你确认。" if zh else (
                    "That step can't be undone by text - ask for the change again "
                    "and I'll ask you to confirm it."
                ),
                session_id=session_id,
                plan_id=str(row["plan_id"]),
            )
        return self.undo(
            session_id, customer_id, str(row["plan_id"]), str(row["step_id"]), lang=lang
        )

    def _typed_recall(
        self, session_id: str, customer_id: str, lang: str | None
    ) -> TurnResult | None:
        bank = self.services.bank_for(customer_id)
        row = latest_undo(getattr(bank, "conn", None), customer_id, session_id)
        if row is None or row.get("tool") != "cancel_transfer":
            return None
        return self.recall(
            session_id, customer_id, str(row["plan_id"]), str(row["step_id"]), lang=lang
        )

    def recall(
        self,
        session_id: str,
        customer_id: str,
        plan_id: str,
        step_id: str,
        lang: str | None = None,
    ) -> TurnResult:
        """SIMULATED recall request (退汇) for a transfer that already left."""

        record = self.services.sessions.get(session_id)
        if record is None or record.get("customer_id") != customer_id:
            return TurnResult(status="error", reply="not found", session_id=session_id)
        if str(plan_id).rsplit(":", 1)[0] != session_id:
            return TurnResult(status="error", reply="not found", session_id=session_id)
        bank = self.services.bank_for(customer_id)
        zh = self._lang(session_id, lang) != "en"
        row = undo_record(getattr(bank, "conn", None), customer_id, plan_id, step_id)
        transfer_id = ""
        if row is not None and row.get("tool") == "cancel_transfer":
            try:
                args = json.loads(str(row["args_json"]))
            except ValueError:
                args = {}
            if isinstance(args, dict):
                transfer_id = str(args.get("transfer_id") or args.get("id") or "")
        if not transfer_id:
            return TurnResult(
                status="error", reply="not found", session_id=session_id, plan_id=plan_id
            )
        now = self.services.now()
        try:
            ticket = open_recall(bank.conn, customer_id, transfer_id, now)
        except BankError as exc:
            if exc.code == "invalid_state" and "undoable" in str(exc):
                text = "这笔还在可撤销时间内，直接撤销就能退回。" if zh else (
                    "This transfer can still be undone - type \"undo\" or tap Undo to get it back."
                )
            elif exc.code == "invalid_state":
                text = "这笔已经撤销过了。" if zh else "That was already undone."
            else:
                text = "找不到这笔转账。" if zh else "I couldn't find that transfer."
            return TurnResult(
                status="error", reply=text, session_id=session_id, plan_id=plan_id
            )
        self.services.audit.append(
            customer_id=customer_id,
            session_id=session_id,
            plan_id=plan_id,
            kind=AuditKind.RECALL,
            payload={
                "ticket": ticket["id"],
                "transfer_id": transfer_id,
                "amount_fen": ticket["amount_fen"],
                "simulated": True,
            },
            now=now,
        )
        amount = fmt_fen(int(ticket["amount_fen"]))
        if zh:
            text = (
                f"已提交追回申请 {ticket['id']}（模拟），金额 {amount}。收款银行会联系收款人请其自愿退回，"
                "可能需要几天，不保证能追回。请保留转账凭证；如果是被骗，请立即拨打 96110 或 110。"
                "对方拒绝退回的话，可以依法起诉要求返还（不当得利）。"
            )
        else:
            text = (
                f"Recall request {ticket['id']} sent (simulated) for {amount}. The receiving bank "
                "will ask the payee to return it voluntarily; this can take days and is not "
                "guaranteed. Keep the transfer receipt. If you were scammed, call 96110 or 110 "
                "now. If the payee refuses, you can claim it back in court (unjust enrichment)."
            )
        return TurnResult(status="done", reply=text, session_id=session_id, plan_id=plan_id)

    def _turn_result(
        self, result: Any, session_id: str, customer_id: str
    ) -> TurnResult:
        record = self.services.sessions.get(session_id) or {}
        plan_id = str(record.get("plan_id") or "")
        if isinstance(result, Mapping):
            plan_id = str(result.get("plan_id") or plan_id)
            interrupts = result.get("__interrupt__")
            if interrupts:
                first = interrupts[0]
                payload = getattr(first, "value", first)
                if isinstance(payload, Mapping):
                    approvals = list(payload.get("approvals") or ())
                    plan_view = list(payload.get("plan_view") or ())
                    if record:
                        record["pending_plan_id"] = plan_id
                    return TurnResult(
                        status="needs_approval",
                        reply=self._pending_reply(session_id, customer_id, approvals),
                        approvals=approvals,
                        plan_view=plan_view,
                        session_id=session_id,
                        plan_id=plan_id,
                    )
            if record:
                record["pending_plan_id"] = ""
            return TurnResult(
                status=str(result.get("status") or "done"),
                reply=str(result.get("reply") or ""),
                approvals=list(result.get("approvals_pending") or ()),
                actions=list(result.get("actions") or ()),
                plan_view=list(result.get("plan_view") or ()),
                session_id=session_id,
                plan_id=plan_id,
            )
        return TurnResult(
            status="error",
            reply="not found",
            session_id=session_id,
            plan_id=plan_id,
        )

    def _pending_reply(
        self,
        session_id: str,
        customer_id: str,
        approvals: list[dict[str, Any]],
    ) -> str:
        record = self.services.sessions.get(session_id) or {}
        plan_id = str(record.get("plan_id") or "")
        session = Session(
            session_id=session_id, plan_id=plan_id, customer_id=customer_id
        )
        try:
            bank = self.services.bank_for(customer_id)
            reply = compose_reply(
                {"kind": "needs_approval", "approvals": approvals},
                lang=self._lang(session_id),
                vault=self.services.vault,
                session=session,
                bank=bank,
                now=self.services.now(),
            )
            return reply.text
        except Exception:  # noqa: BLE001
            return "需要你的确认。" if self._lang(session_id) != "en" else "Confirmation needed."
