"""Template-based reply writer for one finished agent turn.

There is no language model here.  The templates phrase values that already
exist in the executor results (or in the policy/scam verdict), the draft is
written with semantic-token labels for people, and :mod:`security.tokens`
detokenizes it only at the end.  :mod:`security.ground` is the final gate: any
figure the template cannot trace back to a tool result is replaced by a plain
status-only fallback.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from bank.money import fmt_fen
from security.cooling import describe_window
from security.ground import Facts, check, collect_facts, extract_claims
from security.tokens import detokenize

__all__ = ["ReplyResult", "compose_reply"]

_TOKEN_FULL_RE = re.compile(r"^<[A-Z]+_[A-Za-z0-9]+>$")
_LONG_DIGITS_RE = re.compile(r"\d{11,}")
_INTERNAL_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+$")
_INTERNAL_SNAKE_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+(?![A-Za-z0-9_])"
)
#: Any leftover token-like label after detokenizing is never shown raw.
_PLACEHOLDER_RE = re.compile(r"<[^<>\n]{0,120}>")
_MONEY_IN_TEXT_RE = re.compile(
    r"¥\s*(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
)

_INVESTMENT_RISK = (
    "理财都有风险，没有产品能保证收益。",
    "All investments carry risk; no product can guarantee returns.",
)

_ANOMALY_KINDS: dict[str, tuple[str, str]] = {
    "duplicate": ("疑似重复扣款", "possible double charge"),
    "foreign_cnp": ("境外无卡交易", "overseas online charge"),
    "late_night_large": ("深夜大额", "large late-night charge"),
}
_UNKNOWN_ANOMALY = ("异常交易", "unusual charge")

_MISMATCH_CODES = {"mismatch", "otp_mismatch", "mfa_mismatch", "verification_mismatch"}
_GUARANTEE_CODES = {"W4", "w4", "investment_guarantee", "guaranteed_return"}
_UNDO_TOOLS = {
    "cancel_transfer",
    "cancel_scheduled",
    "cancel_request",
    "restore_subscription",
    "move_from_pot",
    "cancel_order",
    "restore_card_limit",
    "close_virtual_card",
}
_MONEY_UNDO_TOOLS = {
    "cancel_transfer",
    "cancel_scheduled",
    "cancel_request",
    "move_from_pot",
    "cancel_order",
}

_CATEGORY_ZH = {
    "food": "餐饮",
    "groceries": "食品杂货",
    "transport": "交通",
    "shopping": "购物",
    "bills_utilities": "账单",
    "housing": "住房",
    "entertainment": "娱乐",
    "subscriptions": "订阅",
    "health": "健康",
    "education": "教育",
    "transfer": "转账",
    "income": "收入",
    "other": "其他",
}

_CATEGORY_EN = {
    "food": "Food",
    "groceries": "Groceries",
    "transport": "Transport",
    "shopping": "Shopping",
    "bills_utilities": "Bills & utilities",
    "housing": "Housing",
    "entertainment": "Entertainment",
    "subscriptions": "Subscriptions",
    "health": "Health",
    "education": "Education",
    "transfer": "Transfers",
    "income": "Income",
    "other": "Other",
}

_MONTHS_EN = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

_CARD_CONTROLS = {
    "online": ("网上支付", "online payments"),
    "abroad": ("境外支付", "payments abroad"),
    "atm": ("ATM 取现", "ATM cash"),
    "contactless": ("闪付", "contactless"),
}

_BLOCK_CATEGORIES_ZH = {
    "gambling": "赌博",
    "crypto": "加密货币",
    "lottery": "彩票",
    "adult": "成人内容",
    "cash_like": "类现金",
}
_BLOCK_CATEGORIES_EN = {
    "gambling": "Gambling",
    "crypto": "Crypto",
    "lottery": "Lottery",
    "adult": "Adult content",
    "cash_like": "Cash-like",
}

_REFUSE_MESSAGES = {
    "not_account_owner": (
        "为了保护隐私，我只能查看你自己的账户。",
        "For privacy, I can only look at your own account.",
    ),
    "out_of_scope": (
        "这个我帮不上忙。我可以帮你查余额、转账、看账单、管理卡片和会员。",
        "I can't help with that. I can check balances, send transfers, "
        "show transactions, and manage cards and memberships.",
    ),
    "not_supported": (
        "这个操作不支持，也不会更改任何安全设置。",
        "That action isn't supported, and it won't change any security settings.",
    ),
    "W4": _INVESTMENT_RISK,
    "w4": _INVESTMENT_RISK,
    "investment_guarantee": _INVESTMENT_RISK,
    "guaranteed_return": _INVESTMENT_RISK,
    "unsafe": (
        "这个请求不安全，我不能执行。",
        "This request is unsafe, so I cannot carry it out.",
    ),
    "missing_details": (
        "还需要一些信息才能办理。",
        "Some details are still missing.",
    ),
}

# Machine codes never reach the user.  Every failure is phrased as a plain
# explanation in the reply language.
_ERROR_MESSAGES: dict[str, tuple[str, str]] = {
    "policy_veto": ("根据安全规则未执行", "Not done: blocked by a safety rule"),
    "reapproval_required": (
        "金额有变化，需要你重新确认",
        "The amount changed - please confirm again",
    ),
    "mandate": (
        "确认信息不完整，请重新操作",
        "The confirmation details are incomplete - please try again",
    ),
    "mandate_missing": ("需要先确认这一步", "This step needs confirmation first"),
    "mandate_required": ("需要先确认这一步", "This step needs confirmation first"),
    "not_approved": (
        "还没有确认，暂时无法执行",
        "Not confirmed yet, so it can't run",
    ),
    "tier_too_low": (
        "安全等级不够，不能执行",
        "The safety level is too low for this action",
    ),
    "mfa_missing": (
        "还需要短信验证码或刷脸确认",
        "An SMS code or face check is still needed",
    ),
    "mfa_required": (
        "需要先完成短信验证码或刷脸确认",
        "Complete the SMS or face check first",
    ),
    "co_approval_missing": (
        "还需要家人共同确认",
        "Family co-approval is still needed",
    ),
    "co_approval_required": ("需要家人共同确认", "Family co-approval is needed"),
    "missing": ("确认信息已失效，请重新操作", "The confirmation is missing"),
    "expired": (
        "确认已过期，请重新操作",
        "The confirmation expired - please try again",
    ),
    "replayed": (
        "这一步已经处理过，请勿重复操作",
        "This step was already processed - no need to repeat it",
    ),
    "mismatch": (
        "验证信息不匹配，请重试",
        "The verification details didn't match - please try again",
    ),
    "not_account_owner": (
        "为了保护隐私，我只能查看你自己的账户。",
        "For privacy, I can only look at your own account.",
    ),
    "out_of_scope": _REFUSE_MESSAGES["out_of_scope"],
    "not_supported": (
        "这个操作不支持，也不会更改任何安全设置。",
        "That action isn't supported, and it won't change any security settings.",
    ),
    "W4": _INVESTMENT_RISK,
    "w4": _INVESTMENT_RISK,
    "investment_guarantee": _INVESTMENT_RISK,
    "guaranteed_return": _INVESTMENT_RISK,
    "not_frozen": (
        "这张卡目前没有冻结，不需要解冻。",
        "This card isn't frozen, so there's nothing to unfreeze.",
    ),
    "card_not_frozen": (
        "这张卡目前没有冻结，不需要解冻。",
        "This card isn't frozen, so there's nothing to unfreeze.",
    ),
    "revoked": (
        "确认已撤销，请重新操作",
        "The confirmation was revoked - please try again",
    ),
    "locked": (
        "因为验证码连续错误，支付已暂停 24 小时。如需帮助请回复“人工”。",
        "Payments are paused for 24 hours after too many wrong codes. "
        "Reply 'human' if you need help.",
    ),
    "otp_missing": ("还需要输入短信验证码", "The SMS code is still needed"),
    "co_otp_missing": (
        "还需要家人提供验证码确认",
        "A family member still needs to enter their code",
    ),
    "rejected": ("你已取消这一步", "You cancelled this step"),
    "invalid_amount": ("金额不正确", "The amount isn't valid"),
    "amount_missing": ("还缺少金额", "The amount is missing"),
    "amount_unknown": (
        "没有找到上一笔可参考的金额，请直接告诉我金额",
        "I couldn't find an earlier amount to copy - please tell me the amount",
    ),
    "token": (
        "信息不完整，请把要求再说一遍",
        "Some details were missing - please say it again",
    ),
    "invest_warned": (
        "刚才提示过投资诈骗风险，24 小时内暂停购买理财产品",
        "Product purchases are paused for 24 hours after the investment-scam warning",
    ),
    "date_unknown": (
        "没有找到这个日期，请直接告诉我具体日期",
        "I couldn't find that date - please tell me the exact date",
    ),
    "payee_ref_not_allowed": (
        "不能向这个收款人转账",
        "You can't send money to this payee",
    ),
    "invalid_date": ("日期不正确", "The date isn't valid"),
    "unknown_ref": (
        "缺少之前步骤的信息，请重新操作",
        "Earlier step details are missing - please start again",
    ),
    "missing_field": (
        "缺少必要信息，请重新操作",
        "Some required details are missing - please start again",
    ),
    "unsupported": ("这个操作暂时不支持", "This action isn't supported yet"),
    "bank": (
        "银行暂时无法处理这笔操作",
        "The bank couldn't process this action",
    ),
    "insufficient_funds": ("余额不足", "Insufficient balance"),
    "not_found": ("没有找到相关信息", "The information couldn't be found"),
    "invalid_state": (
        "当前状态不能执行这个操作",
        "This action can't run in the current state",
    ),
    "invalid": ("信息不正确，请重新操作", "Some details aren't valid"),
    "window_closed": (
        "操作已超过可办理时间",
        "The time window for this action has closed",
    ),
    "dependency": (
        "因上一步未完成而跳过",
        "Skipped because an earlier step didn't finish",
    ),
    "undo": ("无法撤销这一步", "This step can't be undone"),
    "bad_signature": (
        "撤销信息无效，请重新操作",
        "The undo details weren't valid - please try again",
    ),
    "not_co_approver": (
        "需要由指定的家人确认",
        "A designated family member must confirm this",
    ),
    "no_co_approver": (
        "还没有设置家人共同确认",
        "Family co-approval isn't set up yet",
    ),
    "unknown": ("没有找到这项信息", "This information couldn't be found"),
    "tampered": (
        "信息已被改动，请重新操作",
        "The details were changed - please start again",
    ),
}

_UNKNOWN_ERROR = (
    "这个操作未能完成，请稍后再试。",
    "Couldn't be completed - please try again shortly.",
)


@dataclass(slots=True)
class ReplyResult:
    """A reply ready for the user, plus the grounding verdict."""

    text: str
    grounded: bool
    rejected: list[str]


# --------------------------------------------------------------------- helpers


def _is_en(lang: str) -> bool:
    # "mixed" is Chinese-dominant, as is any unknown value.
    return lang == "en"


def _t(lang: str, zh: str, en: str) -> str:
    return en if _is_en(lang) else zh


def _error_code(value: Any) -> str:
    """Extract a stable code from any executor/graph error shape."""

    if isinstance(value, Mapping):
        value = value.get("code") or value.get("error") or value.get("reason")
    elif isinstance(value, (list, tuple, set, frozenset)):
        value = next(iter(value), None)
    if value is None:
        return ""
    return str(value).split(":", 1)[0].strip()


def _context_value(
    outcome: Any, step: Any, error: Any, *names: str
) -> Any:
    """Find a value on the outcome, the step, its result or the error mapping."""

    sources = [outcome, step, _field(step, "result") if step is not None else None]
    if isinstance(error, Mapping):
        sources.append(error)
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for name in names:
            value = source.get(name)
            if value is not None:
                return value
    return None


def _time_text(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%H:%M")
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).strftime("%H:%M")
        except ValueError:
            match = re.search(r"(?<!\d)(\d{1,2}:\d{2})(?!\d)", value)
            if match:
                return match.group(1)
    return ""


def _mismatch_message(attempts: int, lang: str) -> str:
    if _is_en(lang):
        more = "1 more time" if attempts == 1 else f"{attempts} more times"
        return (
            "The verification details don't match - one of the code, amount "
            f"or face check is wrong. You can try {more}."
        )
    return (
        "验证信息不匹配（验证码、金额或刷脸有一项不对），"
        f"还可以再试 {attempts} 次。"
    )


def _locked_message(until: str, lang: str) -> str:
    if _is_en(lang):
        when = f" (until {until})" if until else ""
        return (
            "Payments are paused for 24 hours after too many wrong codes"
            f"{when}. Reply 'human' if you need help."
        )
    when = f"（到 {until}）" if until else ""
    return (
        f"因为验证码连续错误，支付已暂停 24 小时{when}。"
        "如需帮助请回复“人工”。"
    )


def _plain_error(
    value: Any, lang: str, *, outcome: Any = None, step: Any = None
) -> str:
    """Translate any machine code; unknown codes get a generic message."""

    code = _error_code(value)
    if code in _MISMATCH_CODES:
        attempts = _int_or_none(
            _context_value(outcome, step, value, "attempts_left", "attempts_remaining")
        )
        if attempts is not None:
            return _mismatch_message(attempts, lang)
    if code == "locked":
        until = _time_text(
            _context_value(outcome, step, value, "locked_until", "until")
        )
        return _locked_message(until, lang)
    pair = _ERROR_MESSAGES.get(code, _UNKNOWN_ERROR)
    return pair[1] if _is_en(lang) else pair[0]


def _tool_name(tool: Any, lang: str) -> str:
    zh, en = _ACTION_NAMES.get(str(tool or ""), ("操作", "Action"))
    return en if _is_en(lang) else zh


def _scrub_internal_tokens(text: str, lang: str) -> str:
    """Final safety net: no snake_case id or ``<label>`` reaches the user."""

    replacement = _t(lang, "相关操作", "the relevant action")
    text = _PLACEHOLDER_RE.sub(replacement, text)
    generic = _t(lang, "这项操作", "this action")
    return _INTERNAL_SNAKE_RE.sub(generic, text)


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise TypeError("expected a date")


def _date_text(value: Any) -> str | None:
    """Return an ISO date string, or None when the field is absent/unusable."""

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10]).isoformat()
        except ValueError:
            return None
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _first_int(sources: list[Any], names: tuple[str, ...]) -> int | None:
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for name in names:
            value = _int_or_none(source.get(name))
            if value is not None:
                return value
    return None


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _money(fen: int) -> str:
    return fmt_fen(int(fen))


def _join(lines: list[str]) -> str:
    return "\n".join(line for line in lines if line)


def _redact_long_digits(text: str) -> str:
    """Final safety net: no reply may carry a full account number or phone."""

    return _LONG_DIGITS_RE.sub(lambda match: "···" + match.group(0)[-4:], text)


def _payee_label(vault: Any, session: Any, ref: Any) -> str | None:
    """Recover the PAYEE token label for a contact id (or pass a label through)."""

    if isinstance(ref, str) and _TOKEN_FULL_RE.fullmatch(ref):
        return ref
    if ref is None:
        return None
    for entry in vault.entries(session):
        if entry.kind == "PAYEE" and str(entry.real_ref) == str(ref):
            return entry.label
    return None


def _resolve_entry(vault: Any, session: Any, label: str) -> Any:
    if not _TOKEN_FULL_RE.fullmatch(label):
        return None
    try:
        return vault.resolve(label, session)
    except Exception:  # noqa: BLE001 - unknown/expired/tampered tokens
        return None


def _contact_row(bank: Any, contact_id: Any) -> Any:
    conn = getattr(bank, "conn", None)
    if conn is None or contact_id is None:
        return None
    return conn.execute(
        "SELECT name, relation, note, account_no FROM contacts "
        "WHERE customer_id = ? AND id = ?",
        (getattr(bank, "customer_id", ""), str(contact_id)),
    ).fetchone()


def _guarantee_line(lang: str) -> str:
    return _INVESTMENT_RISK[1] if _is_en(lang) else _INVESTMENT_RISK[0]


def _is_guarantee_code(code: Any) -> bool:
    text = str(code or "").strip().casefold()
    return (
        text in _GUARANTEE_CODES
        or text in {"w4", "investment", "investment_risk"}
        or "guarantee" in text
    )


def _with_guarantee(text: str, code: Any, lang: str) -> str:
    if not _is_guarantee_code(code):
        return text
    line = _guarantee_line(lang)
    if not text or line in text:
        return line
    return f"{text} {line}"


def _reason_text(reason: Any, lang: str) -> str:
    if reason is None:
        return ""
    if isinstance(reason, str):
        pair = _REFUSE_MESSAGES.get(reason)
        if pair is not None:
            return pair[1] if _is_en(lang) else pair[0]
        if _is_guarantee_code(reason):
            return _guarantee_line(lang)
        if _INTERNAL_CODE_RE.fullmatch(reason):
            return _with_guarantee(_plain_error(reason, lang), reason, lang)
        return reason
    if isinstance(reason, Mapping):
        code = reason.get("code") or reason.get("reason")
        key = "message_en" if _is_en(lang) else "message_zh"
        text = reason.get(key)
        if _is_guarantee_code(code):
            return _with_guarantee(str(text) if text else "", code, lang)
        if code in _REFUSE_MESSAGES:
            pair = _REFUSE_MESSAGES[code]
            return pair[1] if _is_en(lang) else pair[0]
        if text:
            return str(text)
        return _plain_error(code, lang) if code else ""
    code = getattr(reason, "code", None)
    text = getattr(reason, "message_en" if _is_en(lang) else "message_zh", None)
    if _is_guarantee_code(code):
        return _with_guarantee(str(text) if text else "", code, lang)
    if code in _REFUSE_MESSAGES:
        pair = _REFUSE_MESSAGES[code]
        return pair[1] if _is_en(lang) else pair[0]
    if text:
        return str(text)
    return _plain_error(code, lang) if code else ""


def _rate_text(annual_rate_bp: Any) -> str:
    bp = _int_or_none(annual_rate_bp)
    if bp is None:
        return ""
    rate = Decimal(bp) / Decimal(100)
    if rate == rate.to_integral():
        return f"{int(rate)}"
    return f"{rate:f}".rstrip("0").rstrip(".")


def _format_moment(value: date | datetime, lang: str, *, with_time: bool) -> str:
    if _is_en(lang):
        day = f"{value.strftime('%b')} {value.day}"
        return f"{day}, {value.strftime('%H:%M')}" if with_time else day
    return value.strftime("%m-%d %H:%M") if with_time else value.strftime("%m-%d")


def _fmt_ts(value: Any, facts: Facts, lang: str) -> str:
    """Format a transaction timestamp; parse strings instead of echoing them."""

    if isinstance(value, datetime):
        facts.add_date(value)
        return _format_moment(value, lang, with_time=True)
    if isinstance(value, date):
        facts.add_date(value)
        return _format_moment(value, lang, with_time=False)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return ""
        facts.add_date(parsed)
        return _format_moment(parsed, lang, with_time=True)
    return ""


def _add_money_mentions(text: str, facts: Facts) -> None:
    """Allow figures quoted verbatim by a policy reason (they are not guessed)."""

    for match in _MONEY_IN_TEXT_RE.finditer(text):
        try:
            amount = Decimal(match.group("num").replace(",", ""))
        except ArithmeticError:
            continue
        facts.add_amount(int((amount * 100).to_integral_value()))


# --------------------------------------------------------------- tool renders


def _balance_value(result: Mapping[str, Any]) -> int | None:
    value = _int_or_none(result.get("balance"))
    if value is not None:
        return value
    value = _int_or_none(result.get("balance_fen"))
    if value is not None:
        return value
    return _int_or_none(result.get("raw"))


def _render_balance(result: Mapping[str, Any], lang: str) -> list[str]:
    balance = _balance_value(result)
    if balance is None:
        return [_t(lang, "已查询余额。", "Balance checked.")]
    return [
        _t(
            lang,
            f"当前账户可用余额 {_money(balance)}。",
            f"Available balance: {_money(balance)}.",
        )
    ]


def _category_name(category: str, lang: str) -> str:
    if _is_en(lang):
        # Never show a machine category id: ``transfer`` -> ``Transfers``.
        return _CATEGORY_EN.get(category, category.replace("_", " ").title())
    return _CATEGORY_ZH.get(category, category)


def _add_when_fact(value: Any, facts: Facts) -> None:
    """Allow a transfer row's timestamp as a fact without printing it."""

    if isinstance(value, datetime):
        facts.add_date(value)
    elif isinstance(value, date):
        facts.add_date(value)
    elif isinstance(value, str):
        try:
            facts.add_date(datetime.fromisoformat(value))
        except ValueError:
            pass


def _spending_transfer_rows(result: Mapping[str, Any]) -> list[Any]:
    rows = result.get("transfer_transactions")
    if rows is None:
        raw = result.get("raw")
        if isinstance(raw, Mapping):
            rows = raw.get("transfer_transactions")
    return list(rows) if isinstance(rows, (list, tuple)) else []


def _render_transfer_recipients(
    result: Mapping[str, Any], lang: str, facts: Facts
) -> list[str]:
    """Group actual outgoing transfer rows by counterparty (never contacts)."""

    totals: dict[str, int] = {}
    for row in _spending_transfer_rows(result):
        if row is None:
            continue
        amount = _int_or_none(_field(row, "amount_fen"))
        if amount is None:
            amount = _int_or_none(_field(row, "amount"))
        if amount is None or amount >= 0:
            continue
        if str(_field(row, "category") or "").strip().casefold() != "transfer":
            continue
        status = str(_field(row, "status") or "").strip().casefold()
        if status in {"cancelled", "canceled", "pending", "failed", "reversed", "undone"}:
            continue
        who = str(_field(row, "counterparty") or "").strip()
        if not who or _is_own_savings(who):
            continue
        # Each row is a trusted bank fact; the grouped figure is derived below.
        facts.add_amount(amount)
        _add_when_fact(_field(row, "ts"), facts)
        totals[who] = totals.get(who, 0) + -amount

    if not totals:
        return []

    lines = [_t(lang, "转账收款人：", "Transfer recipients:")]
    for who, amount in sorted(totals.items(), key=lambda item: (-item[1], item[0])):
        facts.add_derived(amount, "transfer recipients total")
        lines.append(_t(lang, f"- {who}：{_money(amount)}", f"- {who}: {_money(amount)}"))
    return lines


def _render_spending(result: Mapping[str, Any], lang: str, facts: Facts) -> list[str]:
    by_category = result.get("by_category")
    parts: list[tuple[str, int]] = []
    if isinstance(by_category, Mapping):
        for key, value in by_category.items():
            amount = _int_or_none(value)
            if amount is not None:
                parts.append((str(key), amount))

    total = _int_or_none(result.get("total"))
    if total is None:
        total = sum(amount for _key, amount in parts)
        # The template computed this itself; declare it before grounding.
        facts.add_derived(total, "spending total")

    lines: list[str] = []
    if total is not None:
        lines.append(
            _t(lang, f"支出共 {_money(total)}。", f"Total spending: {_money(total)}.")
        )

    top = sorted(parts, key=lambda item: (-item[1], item[0]))[:3]
    if top:
        lines.append(_t(lang, "主要类别：", "Top categories:"))
        for key, amount in top:
            lines.append(
                _t(
                    lang,
                    f"- {_category_name(key, lang)}：{_money(amount)}",
                    f"- {_category_name(key, lang)}: {_money(amount)}",
                )
            )
    # Rich reporting is additive; old tool-result fixtures keep their established format.
    period_lines = _report_period_lines(result, lang, facts)
    prefix = list(period_lines)
    if result.get("available") is False:
        return prefix + [_t(lang, "该期间没有可用的交易数据。", "No transaction data is available for this period."),
                         _t(lang, "转入本人储蓄账户不计入消费。", "Transfers to your own savings are excluded from spending.")]
    category = result.get("filter_category")
    merchant = result.get("filter_merchant")
    if category:
        prefix.append(_t(lang, f"类别：{_category_name(str(category), lang)}", f"Category: {_category_name(str(category), lang)}"))
    if merchant:
        prefix.append(_t(lang, f"商户/收款人：{merchant}", f"Merchant/payee: {merchant}"))
    mode = result.get("mode")
    def money(value: Any, reason: str) -> str:
        amount = _int_or_none(value) or 0
        facts.add_derived(amount, reason)
        return _money(amount)
    if mode == "income":
        lines = [_t(lang, f"进账合计：{money(result.get('income'), 'income total')}。",
                    f"Total income: {money(result.get('income'), 'income total')}.")]
    elif mode == "report":
        lines.insert(0, _t(lang, f"进账：{money(result.get('income'), 'income total')}；净额：{money(result.get('net'), 'income minus spending')}。",
                              f"Income: {money(result.get('income'), 'income total')}; net: {money(result.get('net'), 'income minus spending')}.") )
    if mode != "income":
        lines.extend(_render_transfer_recipients(result, lang, facts))
    months = result.get("months")
    if isinstance(months, list):
        lines.append(_t(lang, "每月支出：", "Monthly spending:"))
        for month in months:
            _add_when_fact(month.get("period_start"), facts)
            value = money(month.get("total"), "monthly spending") if month.get("available") else _t(lang, "无数据", "no data")
            lines.append(f"- {month['period_start']}: {value}")
        observed = [month for month in months if month.get("available")]
        if observed:
            highest = max(observed, key=lambda month: month["total"])
            lines.append(_t(lang, f"支出最多月份（起始日）：{highest['period_start']}，{money(highest['total'], 'highest month')}。",
                                  f"Highest spending month: {highest['period_start']}, {money(highest['total'], 'highest month')}.") )
    comparison = result.get("comparison")
    if isinstance(comparison, Mapping) and not months:
        _add_when_fact(comparison.get("period_start"), facts)
        _add_when_fact(comparison.get("period_end"), facts)
        if not comparison.get("available"):
            lines.append(_t(lang, f"{comparison['period_start']} 至 {comparison['period_end']} 无可用数据，无法比较。", f"No data for {comparison['period_start']} to {comparison['period_end']}; comparison unavailable."))
        else:
            delta = int(comparison["delta"])
            amount = money(abs(delta), "spending change")
            categories = comparison.get("by_category", {})
            # Attribute the change only to categories moving in the same direction.
            contributors = [(key, value) for key, value in categories.items() if value * delta > 0]
            main = _category_name(max(contributors, key=lambda item: abs(item[1]))[0], lang) if contributors else ""
            compared_period = f"{comparison['period_start']} to {comparison['period_end']}"
            baseline = _int_or_none(comparison.get("total"))
            if baseline is not None:
                baseline_money = money(baseline, "comparison period spending")
                if mode == "summary" and result.get("period_start") and result.get("period_end"):
                    current_period = f"{result['period_start']} to {result['period_end']}"
                    # Explicit comparison follow-ups show both sides together,
                    # rather than repeating the earlier category report.
                    lines = [_t(lang, "支出对比：", "Spending comparison:"),
                             f"- {current_period}: {money(total, 'current period spending')}",
                             f"- {compared_period}: {baseline_money}"]
                    prefix = prefix[len(period_lines):]
                else:
                    lines.append(_t(lang, f"对比期间（{compared_period}）支出：{baseline_money}。",
                                     f"Comparison period ({compared_period}) spending: {baseline_money}."))
            verdict = _t(lang, f"比 {compared_period} {'多' if delta > 0 else '少'}支出 {amount}",
                               f"Spent {amount} {'more' if delta > 0 else 'less'} than {compared_period}") if delta else _t(lang, f"与 {compared_period} 支出持平", f"Spending unchanged from {compared_period}")
            lines.append(verdict + (_t(lang, f"，主要是{main}。", f", mainly {main}.") if main else "."))
    if result.get("period_label"):
        lines.append(_t(lang, "转入本人储蓄账户不计入消费。", "Transfers to your own savings are excluded from spending."))
    return prefix + lines


def _render_analysis(result: Mapping[str, Any], lang: str, facts: Facts) -> list[str]:
    """Describe validated sandbox output; never imply dev mode is isolated."""
    sandbox = result.get("sandbox")
    raw = result.get("raw")
    failure = _t(lang, "暂时无法完成隔离分析，请稍后重试。", "I could not complete the isolated analysis. Please try again later.")
    if not isinstance(sandbox, Mapping) or sandbox.get("ok") is not True or not isinstance(raw, Mapping):
        return [failure]
    backend = sandbox.get("backend")
    if backend not in {"subprocess-dev", "wsl-unshare", "unshare"}:
        return [failure]
    notice = (
        _t(lang, "本次使用开发测试进程，未启用隔离沙箱。", "Calculated in a development test process; OS sandbox isolation was not enabled.")
        if backend == "subprocess-dev"
        else _t(lang, "计算在隔离沙箱中完成，无网络访问，并限制运行时间和内存。", "Calculated in an isolated sandbox with no network access and limited time and memory.")
    )
    if raw.get("row_count") == 0:
        return [_t(lang, "该期间没有可供分析的消费记录。", "There are no spending transactions to analyse for this period."), notice]

    def integer(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    def money(value: int) -> str:
        facts.add_amount(value)
        return _money(value)

    analysis = result.get("analysis")
    kind = analysis.get("kind") if isinstance(analysis, Mapping) else None
    lines: list[str] = []
    if kind == "weekend_vs_weekday":
        fields = ("weekend_total_fen", "weekday_total_fen", "weekend_days", "weekday_days", "weekend_average_fen", "weekday_average_fen", "difference_fen")
        if not all(integer(raw.get(key)) for key in fields):
            return [failure]
        lines.append(_t(lang,
            f"周末消费 {money(raw['weekend_total_fen'])}，涵盖 {raw['weekend_days']} 个有消费记录的日子；工作日消费 {money(raw['weekday_total_fen'])}，涵盖 {raw['weekday_days']} 个有消费记录的日子。",
            f"Weekend spending: {money(raw['weekend_total_fen'])} across {raw['weekend_days']} spending days. Weekday spending: {money(raw['weekday_total_fen'])} across {raw['weekday_days']} spending days."))
        lines.append(_t(lang,
            f"按有消费记录的天数计算，日均分别为周末 {money(raw['weekend_average_fen'])}、工作日 {money(raw['weekday_average_fen'])}。未消费的日子不计入日均。",
            f"Average per observed spending day: weekends {money(raw['weekend_average_fen'])}; weekdays {money(raw['weekday_average_fen'])}. Days without spending are excluded from these averages."))
        if raw["weekend_days"] and raw["weekday_days"]:
            difference = raw["difference_fen"]
            if difference:
                category = raw.get("largest_category")
                categories = raw.get("categories")
                deltas = raw.get("category_differences_fens")
                # Only attribute the change to a category moving in its direction.
                contributor = ""
                if isinstance(categories, list) and isinstance(deltas, list):
                    contributors = [(key, value) for key, value in zip(categories, deltas) if key in _CATEGORY_EN and integer(value) and value * difference > 0]
                    if contributors:
                        category = max(contributors, key=lambda item: abs(item[1]))[0]
                        contributor = _t(lang, f"，主要差异来自{_category_name(category, lang)}", f", mainly from {_category_name(category, lang)}")
                lines.append(_t(lang,
                    f"周末每个有消费记录的日子平均{'多' if difference > 0 else '少'}花 {money(abs(difference))}{contributor}。",
                    f"You spend {money(abs(difference))} {'more' if difference > 0 else 'less'} per spending day on weekends{contributor}."))
            else:
                lines.append(_t(lang, "周末和工作日的日均消费相同。", "Average spending per spending day is the same on weekends and weekdays."))
        else:
            lines.append(_t(lang, "其中一组没有消费记录，无法比较日均消费。", "One group has no spending transactions, so its daily average cannot be compared."))
    elif kind == "by_hour":
        totals = raw.get("bucket_totals_fens")
        peak = raw.get("peak_bucket")
        if not isinstance(totals, list) or len(totals) != 4 or not all(integer(value) for value in totals) or not integer(peak) or not 0 <= peak < 4:
            return [failure]
        buckets = ("夜间", "上午", "下午", "晚间") if lang == "zh" else ("Night", "Morning", "Afternoon", "Evening")
        lines = [f"{label}: {money(value)}." for label, value in zip(buckets, totals)]
        lines.append(_t(lang, f"消费最多的时段：{buckets[peak]}。", f"Highest spending period: {buckets[peak].lower()}."))
    elif kind in {"by_weekday", "custom"}:
        totals, averages, days = raw.get("weekday_totals_fens"), raw.get("weekday_averages_fens"), raw.get("weekday_days")
        if not all(isinstance(values, list) and len(values) == 7 and all(integer(value) for value in values) for values in (totals, averages, days)):
            return [failure]
        labels = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日") if lang == "zh" else ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
        lines.append(_t(lang, "以下日均只计算有消费记录的日子。", "Daily averages below include only days with spending transactions."))
        for label, total, average, count in zip(labels, totals, averages, days):
            lines.append(_t(lang, f"{label}：合计 {money(total)}，日均 {money(average)}，涵盖 {count} 个消费日。", f"{label}: total {money(total)}, daily average {money(average)}, across {count} spending days."))
    else:
        return [failure]
    return lines + [notice]


def _report_period_lines(result: Mapping[str, Any], lang: str, facts: Facts) -> list[str]:
    label = result.get("period_label")
    if not label:
        return []
    for key in ("period_start", "period_end"):
        _add_when_fact(result.get(key), facts)
    start, end = result.get("period_start"), result.get("period_end")
    # Full ISO dates are recognized by the grounding gate. A bare YYYY-MM is
    # otherwise misread as a currency amount; never whitelist it as money.
    return [_t(lang, f"期间：{start} 至 {end}", f"Period: {start} to {end}")]


def _lookup_transaction(
    bank: Any, txn_id: Any, facts: Facts, lang: str
) -> tuple[str, int, str] | None:
    conn = getattr(bank, "conn", None)
    if conn is None or txn_id is None:
        return None
    row = conn.execute(
        "SELECT ts, amount_fen, counterparty FROM transactions "
        "WHERE customer_id = ? AND id = ?",
        (getattr(bank, "customer_id", ""), str(txn_id)),
    ).fetchone()
    if row is None:
        return None
    amount = _int_or_none(row["amount_fen"])
    if amount is None:
        return None
    try:
        ts = datetime.fromisoformat(row["ts"])
    except (TypeError, ValueError):
        return None
    facts.add_amount(amount)
    facts.add_date(ts)
    return (
        str(row["counterparty"] or ""),
        abs(amount),
        _format_moment(ts, lang, with_time=True),
    )


def _anomaly_details(
    bank: Any, anomaly: Mapping[str, Any], facts: Facts, lang: str
) -> list[tuple[str, int | None, str]]:
    details: list[tuple[str, int | None, str]] = []
    ids = anomaly.get("txn_ids")
    if not isinstance(ids, (list, tuple, set)):
        ids = [anomaly.get("txn_id")] if anomaly.get("txn_id") else []
    for txn_id in ids:
        found = _lookup_transaction(bank, txn_id, facts, lang)
        if found is not None:
            details.append(found)

    if not details:
        merchant = (
            anomaly.get("counterparty")
            or anomaly.get("merchant")
            or anomaly.get("payee")
            or ""
        )
        amount = _int_or_none(anomaly.get("amount_fen"))
        if amount is None:
            amount = _int_or_none(anomaly.get("amount"))
        ts = anomaly.get("ts") or anomaly.get("time") or anomaly.get("date")
        formatted = _fmt_ts(ts, facts, lang) if ts is not None else ""
        if amount is not None:
            facts.add_amount(amount)
        details.append(
            (str(merchant), None if amount is None else abs(amount), formatted)
        )
    return details


def _render_anomalies(
    result: Mapping[str, Any], lang: str, bank: Any, facts: Facts
) -> list[str]:
    anomalies = result.get("anomalies")
    if anomalies is None:
        anomalies = result.get("raw")
    if not isinstance(anomalies, list):
        anomalies = []

    if not anomalies:
        return _report_period_lines(result, lang, facts) + [_t(lang, "没有发现异常交易。", "No anomalies found.")]

    rows: list[str] = []
    for anomaly in anomalies:
        if not isinstance(anomaly, Mapping):
            continue
        zh_kind, en_kind = _ANOMALY_KINDS.get(str(anomaly.get("kind") or ""), _UNKNOWN_ANOMALY)
        kind = en_kind if _is_en(lang) else zh_kind
        reason = {"duplicate": _t(lang, "同一商户、同一金额在五分钟内重复扣款", "same merchant and amount charged twice within five minutes"),
                  "foreign_cnp": _t(lang, "境外无实体卡交易", "foreign transaction without the physical card present")}.get(str(anomaly.get("kind")), "")
        for merchant, amount, when in _anomaly_details(bank, anomaly, facts, lang):
            bits = [merchant] if merchant else []
            if amount is not None:
                bits.append(_money(amount))
            line = f"- {kind}" + (_t(lang, "：", ": ") + " ".join(bits) if bits else "")
            if when:
                line += _t(lang, f"，{when}", f" on {when}")
            if reason:
                line += _t(lang, f"（{reason}）", f" - {reason}")
            rows.append(line)
    lines = _report_period_lines(result, lang, facts) + [_t(lang, f"发现 {len(rows)} 笔异常交易：", f"Found {len(rows)} anomalies:")] + rows
    lines.append(
        _t(
            lang,
            "如不是你本人操作，可以回复“冻结卡片”。",
            "If this wasn't you, reply 'freeze my card'.",
        )
    )
    return lines


def _render_subscriptions(
    result: Mapping[str, Any], lang: str, now: Any, facts: Facts
) -> list[str]:
    items = result.get("raw")
    if items is None:
        items = result.get("sub")
    if not isinstance(items, list):
        items = []

    recurring = result.get("recurring")
    if not isinstance(recurring, list):
        recurring = []
    additional_only = result.get("scope") == "additional"
    if additional_only:
        items = []
    by_id: dict[str, Mapping[str, Any]] = {}
    by_merchant: dict[str, Mapping[str, Any]] = {}
    for entry in recurring:
        if not isinstance(entry, Mapping):
            continue
        subscription_id = entry.get("subscription_id")
        if subscription_id is not None:
            by_id[str(subscription_id)] = entry
        merchant_key = str(entry.get("merchant") or "").casefold()
        if merchant_key:
            by_merchant.setdefault(merchant_key, entry)

    now_date = _as_date(now)
    lines: list[str] = []
    total = 0
    has_total = False
    for item in items:
        status = _field(item, "status")
        if status is not None and status != "active":
            continue
        merchant = str(_field(item, "merchant") or "")
        amount = _int_or_none(_field(item, "amount_fen"))
        if amount is None:
            amount = _int_or_none(_field(item, "amount"))
        last_used = _field(item, "last_used_at")
        unused = False
        if isinstance(last_used, (date, datetime)):
            unused = (now_date - _as_date(last_used)).days >= 60

        text = f"- {merchant}" if merchant else "-"
        if amount is not None:
            text += _t(lang, f" {_money(amount)}/月", f" {_money(amount)}/month")
            total += amount
            has_total = True
        if unused:
            text += _t(lang, "（60 天未使用）", " (unused 60d+)")

        info = None
        item_id = _field(item, "id")
        if item_id is not None:
            info = by_id.get(str(item_id))
        if info is None and merchant:
            info = by_merchant.get(merchant.casefold())
        if info is not None:
            charged = _date_text(info.get("last_charged"))
            source = str(info.get("source") or "")
            if charged:
                facts.add_date(date.fromisoformat(charged))
                text += _t(
                    lang,
                    f"（上次扣费 {charged}）",
                    f" (last charged {charged})",
                )
            elif source in {"", "registered", "both"}:
                text += _t(
                    lang,
                    "（最近180天无扣费记录）",
                    " (no charge in the last 180 days)",
                )
        lines.append(text)

    if not lines and not additional_only:
        lines.append(_t(lang, "没有订阅。", "No subscriptions."))
    if has_total:
        facts.add_derived(total, "monthly subscription total")
        lines.append(
            _t(
                lang,
                f"每月订阅共 {_money(total)}。",
                f"Monthly subscriptions total {_money(total)}.",
            )
        )

    detected = [
        entry
        for entry in recurring
        if isinstance(entry, Mapping) and entry.get("source") == "detected"
    ]
    if detected:
        lines.append(
            _t(
                lang,
                "另外从你的交易中发现的定期扣费：",
                "Also found in your transactions (recurring charges):",
            )
        )
        for entry in detected:
            merchant = str(entry.get("merchant") or "")
            amount = _int_or_none(entry.get("amount_fen"))
            charged = _date_text(entry.get("last_charged"))
            text = f"- {merchant}" if merchant else "-"
            if amount is not None:
                facts.add_amount(amount)
                text += _t(lang, f" {_money(amount)}/月", f" {_money(amount)}/month")
            if charged:
                facts.add_date(date.fromisoformat(charged))
                text += _t(
                    lang,
                    f"，上次扣费 {charged}",
                    f", last charged {charged}",
                )
            lines.append(text)
    elif recurring or additional_only:
        # The detector ran and found nothing extra: say so, so "any charges
        # not in my list?" gets a real answer. No digits (grounding check).
        lines.append(
            _t(
                lang,
                "从你最近半年的交易中没有发现其他定期扣费。",
                "No other recurring charges found in your transactions from the last six months.",
            )
        )
    return lines


_STANDING_REASONS: dict[str, tuple[str, str]] = {
    "payments_locked": ("付款已锁定", "payments are locked"),
    "scam_freeze": ("防诈骗冻结", "scam protection freeze"),
    "payee_missing": ("收款人不存在", "payee no longer exists"),
    "payee_unverified": ("收款人未验证", "payee is not verified"),
    "over_daily_limit": ("超过每日限额", "over the daily limit"),
    "elder_limit": ("超过长辈模式限额", "over the elder-mode limit"),
    "insufficient_funds": ("余额不足", "insufficient balance"),
    "not_run_on_time": ("未能按时执行", "not run on time"),
}
_UNKNOWN_STANDING_REASON = ("未执行", "not run")


def _ground_occurrence(value: Any, facts: Facts) -> str:
    """Text of a ``YYYY-MM`` run label, with the year grounded as a figure.

    The grounding checker reads a bare four-digit year as an amount, so the
    template registers it the same way before showing the label.
    """

    text = str(value or "")
    match = re.fullmatch(r"([0-9]{4})-([0-9]{2})", text)
    if match is not None:
        facts.add_amount(int(match.group(1)) * 100)
    return text


def _render_list_scheduled(
    result: Mapping[str, Any], lang: str, facts: Facts
) -> list[str]:
    items = result.get("scheduled")
    if not isinstance(items, list):
        items = []

    lines: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("status") or "") != "scheduled":
            continue
        payee = str(item.get("payee_display") or "")
        amount = _int_or_none(item.get("amount_fen"))
        if amount is not None:
            facts.add_amount(amount)
        day = _int_or_none(item.get("day_of_month"))
        next_due = _date_text(item.get("next_due"))
        if next_due:
            facts.add_date(date.fromisoformat(next_due))
        money = _money(amount) if amount is not None else ""

        if _is_en(lang):
            line = f"· On day {day} each month: {money} to {payee}"
            if next_due:
                line += f" (next {next_due})"
        else:
            line = f"· 每月{day}号 给 {payee} 转 {money}"
            if next_due:
                line += f"（下次 {next_due}）"

        last = item.get("last_run")
        if isinstance(last, Mapping):
            occurrence = _ground_occurrence(last.get("occurrence"), facts)
            if str(last.get("status") or "") == "paid":
                run_text = _t(lang, "已转出", "paid")
            else:
                reason = _STANDING_REASONS.get(
                    str(last.get("reason") or ""), _UNKNOWN_STANDING_REASON
                )
                reason_text = reason[1] if _is_en(lang) else reason[0]
                run_text = _t(lang, f"已跳过：{reason_text}", f"skipped: {reason_text}")
            line += _t(
                lang,
                f" 上次（{occurrence}）：{run_text}",
                f" last run ({occurrence}): {run_text}",
            )
        lines.append(line)

    if not lines:
        return [_t(lang, "你目前没有定时转账。", "You have no standing orders.")]
    return lines


def _render_stop_standing_order(
    result: Mapping[str, Any], lang: str, facts: Facts
) -> list[str]:
    def item_line(item: Mapping[str, Any]) -> str:
        payee = str(item.get("payee_display") or "")
        amount = _int_or_none(item.get("amount_fen"))
        if amount is not None:
            facts.add_amount(amount)
        money = _money(amount) if amount is not None else ""

        if _is_en(lang):
            if payee and amount is not None:
                return f"Stopped the standing order to {payee} ({money}/month)."
            if payee:
                return f"Stopped the standing order to {payee}."
            if amount is not None:
                return f"Stopped the standing order ({money}/month)."
            return "Standing order stopped."

        if payee and amount is not None:
            return f"已停止给 {payee} 的定时转账（每月 {money}）。"
        if payee:
            return f"已停止给 {payee} 的定时转账。"
        if amount is not None:
            return f"已停止定时转账（每月 {money}）。"
        return "已停止定时转账。"

    items = result.get("stopped")
    if not isinstance(items, list):
        raw = result.get("raw")
        items = raw.get("stopped") if isinstance(raw, Mapping) else None
    if isinstance(items, list):
        lines = [item_line(item) for item in items if isinstance(item, Mapping)]
        if lines:
            return lines

    # Legacy single-result shape (payee/amount at the top level or in ``raw``).
    raw = result.get("raw")
    source: Mapping[str, Any] = raw if isinstance(raw, Mapping) else result
    amount = _int_or_none(source.get("amount_fen"))
    if amount is None:
        amount = _int_or_none(result.get("amount_fen"))
    return [
        item_line(
            {
                "payee_display": source.get("payee_display")
                or result.get("payee_display"),
                "amount_fen": amount,
            }
        )
    ]


def _render_reminders(
    result: Mapping[str, Any], lang: str, facts: Facts, now: Any = None
) -> list[str]:
    items = result.get("reminder")
    if not isinstance(items, list):
        items = result.get("raw")
    if not isinstance(items, list):
        items = []
    due = result.get("due")
    if not isinstance(due, list):
        due = []

    def line_for(item: Mapping[str, Any]) -> str:
        day = _date_text(item.get("due_on"))
        note = str(item.get("note") or "").strip()
        if not note:
            note = _t(lang, "续费提醒", "Renewal reminder")
        if day:
            facts.add_date(date.fromisoformat(day))
            return _t(
                lang,
                f"· {day} 提醒：{note}",
                f"· {day} Reminder: {note}",
            )
        return _t(lang, f"· 提醒：{note}", f"· Reminder: {note}")

    def is_due(item: Mapping[str, Any]) -> bool:
        for entry in due:
            if not isinstance(entry, Mapping):
                continue
            if (entry.get("due_on"), entry.get("note")) != (
                item.get("due_on"),
                item.get("note"),
            ):
                continue
            left = entry.get("created_at")
            right = item.get("created_at")
            if not left or not right or left == right:
                return True
        return False

    lines: list[str] = []
    if not items and not due:
        lines.append(_t(lang, "你还没有设置提醒。", "You have no reminders."))
    else:
        if due:
            today = _date_text(now)
            if today:
                facts.add_date(date.fromisoformat(today))
            lines.append(_t(lang, "今天到期的提醒：", "Due now:"))
            for item in due:
                if isinstance(item, Mapping):
                    lines.append(line_for(item))
        for item in items:
            if not isinstance(item, Mapping) or is_due(item):
                continue
            lines.append(line_for(item))
    lines.append(
        _t(
            lang,
            "本演示不会推送通知。",
            "This demo does not send notifications.",
        )
    )
    return lines


def _active_subscription_lines(
    items: list[Any], lang: str, now: Any, facts: Facts
) -> list[str]:
    """Render only active subscriptions, with a freshly computed monthly total."""

    now_date = _as_date(now)
    body: list[str] = []
    total = 0
    has_total = False
    for item in items:
        status = _field(item, "status")
        if status is not None and status != "active":
            continue
        merchant = str(_field(item, "merchant") or "")
        amount = _int_or_none(_field(item, "amount_fen"))
        if amount is None:
            amount = _int_or_none(_field(item, "amount"))
        last_used = _field(item, "last_used_at")
        unused = False
        if isinstance(last_used, (date, datetime)):
            unused = (now_date - _as_date(last_used)).days >= 60

        text = f"- {merchant}" if merchant else "-"
        if amount is not None:
            # The fresh snapshot is not part of the step results, so declare
            # its amounts here instead of relying on collect_facts.
            facts.add_amount(amount)
            text += _t(lang, f" {_money(amount)}/月", f" {_money(amount)}/month")
            total += amount
            has_total = True
        if unused:
            text += _t(lang, "（60 天未使用）", " (unused 60d+)")
        body.append(text)

    if not body:
        body = [_t(lang, "没有生效中的订阅。", "No active subscriptions.")]
    if has_total:
        facts.add_derived(total, "active subscription monthly total")
        body.append(
            _t(
                lang,
                f"当前每月订阅共 {_money(total)}。",
                f"Current monthly subscriptions total {_money(total)}.",
            )
        )
    return [_t(lang, "生效中的订阅：", "Active subscriptions:"), *body]


def _fresh_active_subscriptions(
    bank: Any, lang: str, now: Any, facts: Facts
) -> list[str]:
    """Read the bank again after a cancellation; never reuse the stale snapshot."""

    try:
        items = bank.list_subscriptions()
    except Exception:  # noqa: BLE001 - a read failure must not show a stale list
        return [
            _t(
                lang,
                "订阅已取消，但暂时无法加载当前订阅列表。",
                "Subscription cancelled, but the current subscription list "
                "can't be loaded right now.",
            )
        ]
    if not isinstance(items, list):
        return [
            _t(
                lang,
                "订阅已取消，但暂时无法加载当前订阅列表。",
                "Subscription cancelled, but the current subscription list "
                "can't be loaded right now.",
            )
        ]
    return _active_subscription_lines(items, lang, now, facts)


def _is_own_savings(counterparty: str) -> bool:
    lowered = counterparty.strip().casefold()
    return "储蓄账户" in counterparty or "savings" in lowered


def _render_transactions(
    result: Mapping[str, Any], lang: str, facts: Facts
) -> list[str]:
    rows = result.get("transactions")
    if rows is None:
        rows = result.get("raw")
    if not isinstance(rows, list):
        rows = []
    if not rows:
        return [_t(lang, "没有找到交易。", "No matching transactions.")]

    lines: list[str] = []
    for row in rows[:5]:
        when = _fmt_ts(_field(row, "ts"), facts, lang)
        amount = _int_or_none(_field(row, "amount_fen"))
        if amount is None:
            amount = _int_or_none(_field(row, "amount"))
        who = str(_field(row, "counterparty") or _field(row, "merchant") or "")
        if who and _is_own_savings(who):
            who = _t(lang, "转入储蓄", "moved to savings")
        parts = []
        if when:
            parts.append(when)
        if who:
            parts.append(who)
        if amount is not None:
            parts.append(_money(amount))
        if parts:
            lines.append("- " + " ".join(parts))
    return lines or [_t(lang, "没有找到交易。", "No matching transactions.")]


def _products_by_id(bank: Any) -> dict[str, Any]:
    try:
        return {str(product.id): product for product in bank.list_products()}
    except Exception:  # noqa: BLE001 - a stub bank may not support products
        return {}


def _render_products(
    tool: str, result: Mapping[str, Any], lang: str, bank: Any, facts: Facts
) -> list[str]:
    items = result.get("raw")
    if items is None:
        items = result.get("portfolio")
    if not isinstance(items, list):
        items = []

    products = _products_by_id(bank) if tool == "get_portfolio" else {}
    lines: list[str] = []
    for item in items:
        product = item
        if tool == "get_portfolio":
            product = products.get(str(_field(item, "product_id")))
        name = str(_field(product, "name") or _field(item, "product_id") or "")
        risk = _int_or_none(_field(product, "risk_level"))
        rate = _rate_text(_field(product, "annual_rate_bp"))
        if rate:
            facts.add_percent(Decimal(rate))
        bits = [name] if name else []
        if risk is not None:
            bits.append(_t(lang, f"风险等级 {risk}", f"risk {risk}"))
        if rate:
            bits.append(f"{rate}%")
        if bits:
            lines.append("- " + _t(lang, "，".join(bits), ", ".join(bits)))
    if not lines:
        return [_t(lang, "没有可显示的产品。", "No products to show.")]
    return lines


def _pct_text(value: Any) -> str:
    """Render a percentage value without trailing zeros."""

    if value is None or isinstance(value, bool):
        return ""
    try:
        number = Decimal(str(value))
    except ArithmeticError:
        return ""
    text = f"{number:f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _add_pct_fact(value: Any, facts: Facts) -> str:
    text = _pct_text(value)
    if text:
        facts.add_percent(Decimal(text))
    return text


def _ground_text(text: str, facts: Facts, now: Any) -> None:
    """Allow the figures carried verbatim by a tool-provided text."""

    for claim in extract_claims(text, now):
        if claim.kind == "money" and isinstance(claim.value, int):
            facts.add_amount(claim.value)
        elif claim.kind == "percent" and claim.value is not None:
            facts.add_percent(claim.value)
        elif claim.kind == "account" and claim.value:
            facts.add_tail(str(claim.value))
        elif claim.kind == "date" and claim.value is not None:
            if isinstance(claim.value, tuple):
                month, day = claim.value
                try:
                    facts.add_date(date(2000, month, day))
                except ValueError:
                    pass
            else:
                facts.add_date(claim.value)


def _bill_due_text(raw: Any, lang: str, facts: Facts) -> str:
    value = _field(raw, "due_date")
    if value is None:
        return ""
    try:
        due = _as_date(value)
    except (TypeError, ValueError):
        return ""
    facts.add_date(due)
    if _is_en(lang):
        return f", due {due.day} {_MONTHS_EN[due.month - 1]}"
    return f"，{due.month}月{due.day}日前"


def _render_bills(result: Mapping[str, Any], lang: str, facts: Facts) -> list[str]:
    items = result.get("raw")
    if not isinstance(items, list):
        items = []
    if not items:
        return [_t(lang, "目前没有待缴账单。", "No bills are due.")]

    lines = [_t(lang, "待缴账单：", "Bills due:")]
    for item in items:
        if not isinstance(item, Mapping):
            continue
        biller = str(
            (item.get("biller_en") if _is_en(lang) else item.get("biller"))
            or item.get("biller")
            or ""
        )
        amount = _int_or_none(item.get("amount_due_fen"))
        if amount is not None:
            facts.add_amount(amount)
        line = "- " + biller
        if amount is not None:
            line += (" " if biller else "") + _money(amount)
        line += _bill_due_text(item, lang, facts)
        lines.append(line)
    return lines


def _render_pay_bill(result: Mapping[str, Any], lang: str) -> list[str]:
    raw = result.get("raw")
    amount = _int_or_none(_field(raw, "amount_due_fen"))
    if amount is None:
        amount = _result_amount(result)
    biller = str(
        (_field(raw, "biller_en") if _is_en(lang) else _field(raw, "biller"))
        or _field(raw, "biller")
        or ""
    )
    if amount is None or not biller:
        return [_t(lang, "已缴纳账单。", "Bill paid.")]
    return [
        _t(
            lang,
            f"已缴纳 {biller} {_money(amount)}。",
            f"Paid {biller} {_money(amount)}.",
        )
    ]


def _render_installment(result: Mapping[str, Any], lang: str, facts: Facts) -> list[str]:
    raw = result.get("raw")
    if not isinstance(raw, Mapping):
        raw = result
    periods = _int_or_none(_field(raw, "periods"))
    payment = _int_or_none(_field(raw, "payment_fen"))
    total_fee = _int_or_none(_field(raw, "total_fee_fen"))
    nominal = _field(raw, "nominal_annual_pct")
    apr = _field(raw, "apr_pct")
    fee_rate = _field(raw, "fee_rate_per_period_pct")
    if None in (periods, payment, total_fee, nominal, apr, fee_rate):
        return [_t(lang, "已生成分期试算。", "Instalment quote ready.")]

    facts.add_amount(payment)
    facts.add_amount(total_fee)
    fee_text = _add_pct_fact(fee_rate, facts)
    nominal_text = _add_pct_fact(nominal, facts)
    apr_text = _add_pct_fact(apr, facts)
    return [
        _t(
            lang,
            f"分 {periods} 期：每期 {_money(payment)}，手续费共 "
            f"{_money(total_fee)}。手续费率每期 {fee_text}%，看起来年化 "
            f"{nominal_text}%，但实际年化利率约 {apr_text}%。",
            f"{periods} instalments: {_money(payment)} each, "
            f"{_money(total_fee)} in fees. The fee is {fee_text}% per month, "
            f"which looks like {nominal_text}% a year, but the real annual "
            f"rate is about {apr_text}%.",
        )
    ]


def _render_deposit(result: Mapping[str, Any], lang: str, facts: Facts) -> list[str]:
    raw = result.get("raw")
    if not isinstance(raw, Mapping):
        raw = result
    principal = _int_or_none(_field(raw, "principal_fen"))
    term = _int_or_none(_field(raw, "term_months"))
    rate = _field(raw, "rate_pct")
    interest = _int_or_none(_field(raw, "interest_fen"))
    demand = _field(raw, "demand_rate_pct")
    early = _int_or_none(_field(raw, "early_half_term_fen"))
    if None in (principal, term, rate, interest, demand, early):
        return [_t(lang, "已生成定期试算。", "Deposit quote ready.")]

    facts.add_amount(principal)
    facts.add_amount(interest)
    facts.add_amount(early)
    rate_text = _add_pct_fact(rate, facts)
    demand_text = _add_pct_fact(demand, facts)
    return [
        _t(
            lang,
            f"存 {_money(principal)} 定期 {term} 个月，年利率 {rate_text}%，"
            f"到期利息约 {_money(interest)}。如果存到一半提前支取，只按活期 "
            f"{demand_text}% 计息，约 {_money(early)}。（演示利率）",
            f"Put {_money(principal)} in a {term}-month deposit at {rate_text}% "
            f"a year; interest at maturity is about {_money(interest)}. If you "
            f"take it out halfway, it earns the demand rate of {demand_text}%, "
            f"about {_money(early)}. (demo rates)",
        )
    ]


def _render_forecast(result: Mapping[str, Any], lang: str, facts: Facts) -> list[str]:
    raw = result.get("raw")
    if not isinstance(raw, Mapping):
        raw = result
    lowest = _int_or_none(_field(raw, "lowest_balance_fen"))
    end = _int_or_none(_field(raw, "end_balance_fen"))
    offset = _int_or_none(_field(raw, "lowest_on_day_offset"))
    days = _int_or_none(_field(raw, "days"))
    goes_negative = bool(_field(raw, "goes_negative"))
    if lowest is None or end is None:
        return [_t(lang, "已生成余额预测。", "Balance forecast ready.")]

    facts.add_amount(lowest)
    facts.add_amount(end)
    when = ""
    if offset:
        when = _t(lang, f"（第 {offset} 天）", f" (day {offset})")
    if _is_en(lang):
        window = f"Over the next {days} days" if days is not None else "Ahead"
        text = (
            f"{window}: the lowest balance is about {_money(lowest)}{when}, "
            f"and about {_money(end)} at the end."
        )
        if goes_negative:
            text += (
                " ⚠ The balance may not be enough - set money aside in advance."
            )
        else:
            text += " The balance is enough."
        return [text]
    window = f"未来 {days} 天" if days is not None else "接下来"
    text = (
        f"{window}：预计最低余额 {_money(lowest)}{when}，"
        f"期末约 {_money(end)}。"
    )
    if goes_negative:
        text += "⚠ 余额可能不够，建议提前留出资金。"
    else:
        text += "余额够用。"
    return [text]


def _render_help(result: Mapping[str, Any], lang: str, facts: Facts, now: Any) -> list[str]:
    raw = result.get("raw")
    if not isinstance(raw, Mapping):
        raw = result
    text = str((raw.get("en") if _is_en(lang) else raw.get("zh")) or "")
    if not text:
        return []
    source = str(raw.get("source") or "")
    line = text + (_t(lang, f"（来源：{source}）", f" (Source: {source})") if source else "")
    _ground_text(line, facts, now)
    return [line]


def _render_transfers(
    result: Mapping[str, Any], lang: str, vault: Any, session: Any, facts: Facts
) -> list[str]:
    items = result.get("raw")
    if not isinstance(items, list):
        items = []
    if not items:
        return [_t(lang, "最近没有转账。", "No recent transfers.")]

    lines: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        amount = _int_or_none(item.get("amount_fen"))
        if amount is None:
            continue
        # The line starts with "-", so the grounding parser reads the figure
        # as negative; allow both signs.
        facts.add_amount(-amount)
        label = _payee_label(vault, session, item.get("to_contact_id"))
        prefix = f"{label} " if label else ""
        money = _money(amount)
        status = str(item.get("status") or "")
        if status == "held":
            moment = _time_text(item.get("release_at"))
            if moment:
                lines.append(
                    _t(
                        lang,
                        f"- {prefix}{money}：暂缓中，{moment} 前可撤销",
                        f"- {prefix}{money}: on hold, undo until {moment}",
                    )
                )
            else:
                lines.append(
                    _t(
                        lang,
                        f"- {prefix}{money}：暂缓中，可撤销",
                        f"- {prefix}{money}: on hold, can be undone",
                    )
                )
        elif status == "undone":
            lines.append(_t(lang, f"- {prefix}{money}：已撤销", f"- {prefix}{money}: undone"))
        elif status == "sent":
            lines.append(_t(lang, f"- {prefix}{money}：已转出", f"- {prefix}{money}: sent"))
        else:
            lines.append(_t(lang, f"- {prefix}{money}：处理中", f"- {prefix}{money}: processing"))
    return lines or [_t(lang, "最近没有转账。", "No recent transfers.")]


def _render_verify_payee(result: Mapping[str, Any], lang: str, facts: Facts) -> list[str]:
    raw = result.get("raw")
    if not isinstance(raw, Mapping) or "on_fraud_list" not in raw:
        return [_t(lang, "已核验收款人。", "Payee checked.")]
    if raw.get("on_fraud_list"):
        # The fixed public hotline is not money, but a bare five-digit run
        # would otherwise be read as a figure by the grounding check.
        facts.add_amount(9611000)
        return [
            _t(
                lang,
                "⚠ 这个收款账户在反诈名单上（模拟）。请不要转账，"
                "如有疑问拨打 96110。",
                "⚠ This account is on the anti-fraud list (simulated). "
                "Don't send money; call 96110 if unsure.",
            )
        ]
    text = _t(
        lang,
        "这个收款账户不在反诈名单上。",
        "This account isn't on the anti-fraud list.",
    )
    if raw.get("first_time"):
        text += _t(
            lang,
            " 这是你第一次给这个人转账，请先通过电话等方式确认对方身份。",
            " This is the first time you've sent to them - confirm who they are "
            "by phone first.",
        )
    return [text]


def _render_card_control(result: Mapping[str, Any], lang: str, facts: Facts | None = None) -> list[str]:
    raw = result.get("raw")
    if not isinstance(raw, Mapping):
        raw = result
    control = str(_field(raw, "control") or "")
    if control not in _CARD_CONTROLS:
        return [_t(lang, "已调整卡片开关。", "Card control updated.")]
    enabled = bool(_field(raw, "enabled"))
    zh_name, en_name = _CARD_CONTROLS[control]
    tail = _card_tail(result, facts if facts is not None else Facts())
    if enabled and _field(raw, "effective_enabled") is False:
        status = str(_field(raw, "card_status") or "unavailable")
        next_step = " Say 'unfreeze my card' to request separate confirmation." if status == "frozen" else " Contact the bank about this card's status."
        return [_t(lang, f"卡片{tail}的{zh_name}设置已开启，但卡片状态仍为 {status}，目前不能付款。",
                   f"Card{tail}: {en_name} setting is on, but the card is still {status}; payments remain unavailable.{next_step}")]
    return [
        _t(
            lang,
            f"{('卡片' + tail) if tail else ''}{'已开启' if enabled else '已关闭'} {zh_name}。",
            f"{en_name[0].upper() + en_name[1:]} "
            f"{'turned on' if enabled else 'turned off'}{(' for card' + tail) if tail else ''}.",
        )
    ]


def _card_tail(result: Mapping[str, Any], facts: Facts) -> str:
    raw = result.get("raw")
    last4 = _field(raw, "last4") or result.get("last4")
    if not last4:
        return ""
    tail = str(last4)[-4:]
    facts.add_tail(tail)
    return f" ···{tail}"


def _render_cards(result: Mapping[str, Any], lang: str, facts: Facts) -> list[str]:
    if "cards" not in result and "raw" not in result:
        return []
    cards = result.get("cards")
    if not isinstance(cards, list):
        cards = result.get("raw") or []
    lines = []
    for card in cards:
        tail = str(_field(card, "last4") or "")[-4:]
        facts.add_tail(tail)
        kind = str(_field(card, "kind") or "physical")
        status = str(_field(card, "status") or "unknown")
        limit = _int_or_none(_field(card, "online_limit_fen"))
        money = _money(limit) if limit is not None else ""
        if limit is not None:
            facts.add_amount(limit)
        blocks = _field(card, "active_blocks") or []
        block_text = ", ".join(str(value) for value in blocks) or _t(lang, "无", "none")
        lines.append(_t(lang, f"- {'虚拟卡' if kind == 'virtual' else '实体卡'} ···{tail}：{status}；网购限额 {money}；生效限制：{block_text}",
                         f"- {kind.capitalize()} card ···{tail}: {status}; online limit {money}; active blocks: {block_text}"))
    return lines or [_t(lang, "没有找到银行卡。", "No cards found.")]


def _block_name(category: str, lang: str) -> str:
    if _is_en(lang):
        return _BLOCK_CATEGORIES_EN.get(category, category)
    return _BLOCK_CATEGORIES_ZH.get(category, category)


def _render_block_merchant(result: Mapping[str, Any], lang: str) -> list[str]:
    raw = result.get("raw")
    if not isinstance(raw, Mapping):
        raw = result
    category = str(_field(raw, "category") or "")
    if category not in _BLOCK_CATEGORIES_ZH:
        return [_t(lang, "已更新消费屏蔽。", "Spending block updated.")]
    if _is_en(lang):
        return [
            f"{_block_name(category, lang)} spending is blocked now. "
            "Lifting the block takes 48 hours."
        ]
    return [
        f"已屏蔽 {_block_name(category, lang)} 类消费，立即生效。"
        "解除需要等待 48 小时。"
    ]


def _render_unblock_merchant(result: Mapping[str, Any], lang: str, facts: Facts) -> list[str]:
    raw = result.get("raw")
    if not isinstance(raw, Mapping):
        raw = result
    category = str(_field(raw, "category") or "")
    value = _field(raw, "unblock_at")
    if category not in _BLOCK_CATEGORIES_ZH or value is None:
        return [_t(lang, "已申请解除屏蔽。", "Block removal requested.")]
    try:
        moment = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return [_t(lang, "已申请解除屏蔽。", "Block removal requested.")]
    facts.add_date(moment)
    zh_name = _block_name(category, lang)
    if _is_en(lang):
        return [
            f"The {zh_name.lower()} block will lift on {moment.day} "
            f"{_MONTHS_EN[moment.month - 1]} {moment.strftime('%H:%M')}; "
            "it stays on until then."
        ]
    return [
        f"已申请解除 {zh_name} 屏蔽，将在 {moment.month}月{moment.day}日 "
        f"{moment.strftime('%H:%M')} 生效，在此之前仍然屏蔽。"
    ]


def _undo_phrase(result: Mapping[str, Any], lang: str, now: Any) -> str:
    raw = result.get("raw")
    created = _field(raw, "created_at")
    release = _field(raw, "release_at")
    if release is None:
        release = result.get("release_at")
    if created is None:
        created = result.get("created_at")

    status = _field(raw, "status")
    if status == "released":
        return _t(lang, "钱已转出。", "The money has left.")

    if release is None:
        return _t(lang, "资金变动已提交。", "The transfer was submitted.")
    try:
        window = describe_window(created or now, release, now)
    except (TypeError, ValueError):
        return _t(lang, "资金变动已提交。", "The transfer was submitted.")

    if window.get("can_undo"):
        try:
            moment = datetime.fromisoformat(str(window["release_at"])).strftime("%H:%M")
        except (TypeError, ValueError):
            return _t(lang, "资金变动已提交。", "The transfer was submitted.")
        return _t(lang, f"可在 {moment} 前撤销。", f"You can undo until {moment}.")
    return _t(lang, "钱已转出。", "The money has left.")


def _result_amount(result: Mapping[str, Any]) -> int | None:
    for key in ("amount", "amount_fen"):
        value = _int_or_none(result.get(key))
        if value is not None:
            return value
    raw = result.get("raw")
    return _int_or_none(_field(raw, "amount_fen"))


def _result_payee(result: Mapping[str, Any]) -> Any:
    value = result.get("payee")
    if value is not None:
        return value
    raw = result.get("raw")
    for name in ("to_contact_id", "from_contact_id"):
        value = _field(raw, name)
        if value is not None:
            return value
    return None


def _is_already_undone(result: Mapping[str, Any]) -> bool:
    raw = result.get("raw")
    for source in (result, raw):
        if not isinstance(source, Mapping):
            continue
        if source.get("already_undone") is True:
            return True
        status = str(source.get("status") or "").strip().lower()
        if status in {"already_undone", "already", "undone"}:
            return True
        if _error_code(source) in {"already_undone", "replayed"}:
            return True
    return False


def _undo_lines(result: Mapping[str, Any], lang: str, facts: Facts) -> list[str]:
    if _is_already_undone(result):
        return [_t(lang, "这笔已经撤销过了。", "That was already undone.")]
    amount = _result_amount(result)
    if amount is None or amount == 0:
        return [_t(lang, "已撤销。", "Undone.")]
    facts.add_amount(amount)
    refund = _money(abs(amount))
    return [
        _t(
            lang,
            f"已撤销，{refund} 已退回你的账户。",
            f"Undone — {refund} is back in your account.",
        )
    ]


def _is_not_frozen(result: Mapping[str, Any]) -> bool:
    raw = result.get("raw")
    for source in (result, raw):
        if not isinstance(source, Mapping):
            continue
        if source.get("not_frozen") is True or source.get("already_unfrozen") is True:
            return True
        status = str(source.get("status") or "").strip().lower()
        if status == "not_frozen":
            return True
        if _error_code(source) in {"not_frozen", "card_not_frozen"}:
            return True
    return False


def _render_transfer(
    result: Mapping[str, Any], lang: str, vault: Any, session: Any, now: Any
) -> list[str]:
    amount = _result_amount(result)
    label = _payee_label(vault, session, _result_payee(result))
    if amount is None:
        text = _t(lang, "已提交转账。", "Transfer submitted.")
    elif label is not None:
        text = _t(
            lang,
            f"已提交转账 {_money(amount)} 给 {label}。",
            f"Transfer submitted: {_money(amount)} to {label}.",
        )
    else:
        text = _t(
            lang,
            f"已提交转账 {_money(amount)}。",
            f"Transfer submitted: {_money(amount)}.",
        )
    separator = " " if _is_en(lang) else ""
    return [text + separator + _undo_phrase(result, lang, now)]


def _render_scheduled(
    result: Mapping[str, Any], lang: str, vault: Any, session: Any
) -> list[str]:
    amount = _result_amount(result)
    label = _payee_label(vault, session, _result_payee(result))
    day = _int_or_none(_field(result.get("raw"), "day_of_month"))
    if day is None:
        day = _int_or_none(result.get("day_of_month"))
    if amount is None:
        return [_t(lang, "已设置定期转账。", "Scheduled transfer created.")]
    target = label if label is not None else ""
    if _is_en(lang):
        where = f" to {target}" if target else ""
        when = f" on day {day} of each month" if day is not None else ""
        return [f"Scheduled {_money(amount)}{where}{when}."]
    where = f"给 {target}" if target else ""
    when = f"每月 {day} 日" if day is not None else "定期"
    return [f"已设置{when}{where}转账 {_money(amount)}。"]


def _render_write(
    tool: str,
    result: Mapping[str, Any],
    lang: str,
    vault: Any,
    session: Any,
    now: Any,
    facts: Facts,
) -> list[str]:
    raw = result.get("raw")
    amount = _result_amount(result)

    if tool == "transfer":
        return _render_transfer(result, lang, vault, session, now)

    if tool == "schedule_transfer":
        return _render_scheduled(result, lang, vault, session)

    if tool == "stop_standing_order":
        return _render_stop_standing_order(result, lang, facts)

    if tool == "request_payment":
        label = _payee_label(vault, session, _result_payee(result))
        if amount is None:
            return [_t(lang, "已发起收款请求。", "Payment request sent.")]
        if label is not None:
            return [
                _t(
                    lang,
                    f"已向 {label} 发起 {_money(amount)} 的收款请求。",
                    f"Payment request for {_money(amount)} sent to {label}.",
                )
            ]
        return [
            _t(
                lang,
                f"已发起 {_money(amount)} 的收款请求。",
                f"Payment request for {_money(amount)} sent.",
            )
        ]

    if tool == "merchant_order":
        merchant = str(_field(raw, "merchant") or "")
        delivered = result.get("date")
        if isinstance(delivered, (date, datetime)):
            day = (
                f"{delivered.month}月{delivered.day}日"
                if not _is_en(lang)
                else delivered.strftime("%Y-%m-%d")
            )
        else:
            day = ""
        if amount is None:
            return [_t(lang, "已下单。", "Order placed.")]
        text = (
            f"已在 {merchant} 下单 {_money(amount)}"
            if merchant
            else f"已下单 {_money(amount)}"
        )
        if day:
            text += _t(lang, f"，预计 {day} 送达。", f", expected on {day}.")
        else:
            text += "。"
        return [text]

    if tool == "cancel_subscription":
        merchant = str(_field(raw, "merchant") or "")
        text = _t(
            lang,
            f"已取消 {merchant} 订阅。" if merchant else "已取消订阅。",
            f"Cancelled the {merchant} subscription."
            if merchant
            else "Subscription cancelled.",
        )
        saved = _int_or_none(result.get("saved_monthly_fen"))
        if saved is not None and saved > 0:
            facts.add_amount(saved)
            text += _t(
                lang,
                f"每月节省 {_money(saved)}。",
                f" You save {_money(saved)} per month.",
            )
        return [text]

    if tool == "reactivate_subscription":
        merchant = str(_field(raw, "merchant") or "")
        if merchant and amount is not None:
            facts.add_amount(amount)
            return [
                _t(
                    lang,
                    f"已重新开通 {merchant} 订阅（每月 {_money(amount)}）。",
                    f"Re-activated {merchant} ({_money(amount)}/month).",
                )
            ]
        if merchant:
            return [
                _t(
                    lang,
                    f"已重新开通 {merchant} 订阅。",
                    f"Re-activated {merchant}.",
                )
            ]
        return [_t(lang, "已重新开通订阅。", "Subscription re-activated.")]

    if tool == "move_to_pot":
        pot = str(_field(raw, "name") or "")
        if amount is None:
            return [_t(lang, "已转入储蓄罐。", "Moved money to the pot.")]
        return [
            _t(
                lang,
                f"已转入 {pot} {_money(amount)}。" if pot else f"已转入 {_money(amount)}。",
                f"Moved {_money(amount)} to {pot}."
                if pot
                else f"Moved {_money(amount)}.",
            )
        ]

    if tool == "set_card_limit":
        if amount is None:
            return [_t(lang, "已修改卡片限额。", "Card limit updated.")]
        return [
            _t(
                lang,
                f"已将卡片{_card_tail(result, facts)}限额设为 {_money(amount)}。",
                f"Card{_card_tail(result, facts)} limit set to {_money(amount)}.",
            )
        ]

    if tool == "create_virtual_card":
        if amount is None:
            return [_t(lang, "已创建虚拟卡。", "Virtual card created.")]
        return [
            _t(
                lang,
                f"已创建虚拟卡{_card_tail(result, facts)}，限额 {_money(amount)}。",
                f"Virtual card{_card_tail(result, facts)} created with a {_money(amount)} limit.",
            )
        ]

    if tool == "freeze_card":
        if _field(raw, "status") == "lost":
            return [_t(lang, f"卡片{_card_tail(result, facts)}已挂失，付款仍被阻止。", f"Card{_card_tail(result, facts)} is already reported lost; payments remain blocked.")]
        return [_t(lang, f"已冻结卡片{_card_tail(result, facts)}。", f"Card frozen{_card_tail(result, facts)}.")]
    if tool == "unfreeze_card":
        if _is_not_frozen(result):
            return [
                _t(
                    lang,
                    "这张卡目前没有冻结，不需要解冻。",
                    "This card isn't frozen, so there's nothing to unfreeze.",
                )
            ]
        return [_t(lang, f"已解冻卡片{_card_tail(result, facts)}。", f"Card unfrozen{_card_tail(result, facts)}.")]
    if tool == "restore_card_limit":
        if result.get("restored") is False:
            return [_t(lang, "没有可恢复的旧限额，卡片限额没有改变。", "No previous card limit to restore; the limit is unchanged.")]
        return [_t(lang, f"卡片{_card_tail(result, facts)}限额已恢复为 {_money(amount)}。", f"Card{_card_tail(result, facts)} limit restored to {_money(amount)}.")] if amount is not None else [_t(lang, "卡片限额已恢复。", "Card limit restored.")]
    if tool == "report_card_lost":
        return [
            _t(
                lang,
                f"卡片{_card_tail(result, facts)}已挂失。新卡可以在网点或 App 里申请补办。",
                f"Reported lost{_card_tail(result, facts)}. You can apply for a replacement card "
                "at a branch or in the app.",
            )
        ]
    if tool == "set_reminder":
        due_on = _date_text(_field(raw, "due_on"))
        if due_on:
            facts.add_date(date.fromisoformat(due_on))
            return [
                _t(
                    lang,
                    f"提醒已保存：{due_on}。本演示不会推送通知。",
                    f"Reminder saved for {due_on}. This demo does not send notifications.",
                )
            ]
        return [
            _t(
                lang,
                "提醒已保存。本演示不会推送通知。",
                "Reminder saved. This demo does not send notifications.",
            )
        ]
    if tool == "subscribe_product":
        if amount is None:
            return [_t(lang, "已申购产品。", "Product purchased.")]
        return [
            _t(lang, f"已申购 {_money(amount)}。", f"Purchased {_money(amount)}.")
        ]
    if tool == "redeem_product":
        if amount is None:
            return [_t(lang, "模拟赎回已完成，请查看回执。", "Mock redemption completed; see the receipt.")]
        return [_t(lang, f"模拟赎回 {_money(amount)} 已入账。", f"Mock redemption credited {_money(amount)}.")]
    if tool == "change_password":
        return [_t(lang, "请在银行安全页面本人修改密码。本演示没有修改密码。", "Change your password yourself in the bank's secure settings. This demo has not changed it.")]
    if tool in _UNDO_TOOLS:
        if tool == "restore_subscription":
            merchant = str(_field(raw, "merchant") or "")
            if merchant:
                if amount is not None:
                    facts.add_amount(amount)
                    return [
                        _t(
                            lang,
                            f"已恢复 {merchant} 订阅（每月 {_money(amount)}）。",
                            f"Restored {merchant} ({_money(amount)}/month).",
                        )
                    ]
                return [
                    _t(
                        lang,
                        f"已恢复 {merchant} 订阅。",
                        f"Restored {merchant}.",
                    )
                ]
        if tool in _MONEY_UNDO_TOOLS:
            return _undo_lines(result, lang, facts)
        if _is_already_undone(result):
            return [_t(lang, "这笔已经撤销过了。", "That was already undone.")]
        return [_t(lang, "已撤销。", "Undone.")]

    return [_t(lang, f"已完成 {tool}。", f"Done: {tool}.")]


def _done_lines(
    step: Mapping[str, Any],
    *,
    lang: str,
    vault: Any,
    session: Any,
    bank: Any,
    now: Any,
    facts: Facts,
) -> list[str]:
    tool = str(step.get("tool") or "")
    result = step.get("result")
    if not isinstance(result, Mapping):
        result = {}

    if tool == "get_balance":
        return _render_balance(result, lang)
    if tool == "spending_report":
        return _render_spending(result, lang, facts)
    if tool == "run_analysis":
        return _render_analysis(result, lang, facts)
    if tool == "detect_anomalies":
        return _render_anomalies(result, lang, bank, facts)
    if tool == "list_cards":
        return _render_cards(result, lang, facts)
    if tool == "list_subscriptions":
        return _render_subscriptions(result, lang, now, facts)
    if tool == "list_scheduled":
        return _render_list_scheduled(result, lang, facts)
    if tool == "list_reminders":
        return _render_reminders(result, lang, facts, now)
    if tool == "list_bills":
        return _render_bills(result, lang, facts)
    if tool == "pay_bill":
        return _render_pay_bill(result, lang)
    if tool == "installment_quote":
        return _render_installment(result, lang, facts)
    if tool == "deposit_quote":
        return _render_deposit(result, lang, facts)
    if tool == "forecast_cashflow":
        return _render_forecast(result, lang, facts)
    if tool == "help_answer":
        return _render_help(result, lang, facts, now)
    if tool == "list_transfers":
        return _render_transfers(result, lang, vault, session, facts)
    if tool == "verify_payee":
        return _render_verify_payee(result, lang, facts)
    if tool == "set_card_control":
        return _render_card_control(result, lang, facts)
    if tool == "block_merchant":
        return _render_block_merchant(result, lang)
    if tool == "unblock_merchant":
        return _render_unblock_merchant(result, lang, facts)
    if tool in {"list_contacts", "find_contact"}:
        lines = _render_contacts(result, lang, facts)
        if lines:
            return lines
    if tool == "list_transactions":
        if "anomalies" in result:
            return _render_anomalies(result, lang, bank, facts)
        return _report_period_lines(result, lang, facts) + _render_transactions(result, lang, facts)
    if tool in {"get_portfolio", "list_products"}:
        return _render_products(tool, result, lang, bank, facts)
    if tool == "can_cover":
        covered = result.get("ok")
        if covered is None:
            covered = result.get("raw")
        if isinstance(covered, Mapping):
            covered = next(
                (
                    covered.get(key)
                    for key in ("ok", "covered", "can_cover")
                    if covered.get(key) is not None
                ),
                None,
            )
        if covered is not False and covered != 0:
            return []
        sources = [result]
        raw = result.get("raw")
        if isinstance(raw, Mapping):
            sources.append(raw)
        balance = _first_int(
            sources,
            (
                "balance",
                "balance_fen",
                "available",
                "available_fen",
                "available_balance",
                "available_balance_fen",
            ),
        )
        required = _first_int(
            sources,
            (
                "amount",
                "amount_fen",
                "required",
                "required_fen",
                "required_amount",
                "required_amount_fen",
                "need",
                "need_fen",
            ),
        )
        if balance is not None:
            facts.add_amount(balance)
        if required is not None:
            facts.add_amount(required)
        if balance is not None and required is not None:
            return [
                _t(
                    lang,
                    f"余额不足：当前可用 {_money(balance)}，"
                    f"这笔需要 {_money(required)}，没有转账。",
                    f"Insufficient balance: available {_money(balance)}, "
                    f"this needs {_money(required)} - no transfer was made.",
                )
            ]
        if balance is not None:
            return [
                _t(
                    lang,
                    f"余额不足：当前可用 {_money(balance)}，没有转账。",
                    f"Insufficient balance: available {_money(balance)} - "
                    "no transfer was made.",
                )
            ]
        if required is not None:
            return [
                _t(
                    lang,
                    f"余额不足：这笔需要 {_money(required)}，没有转账。",
                    f"Insufficient balance: this needs {_money(required)} - "
                    "no transfer was made.",
                )
            ]
        return [
            _t(
                lang,
                "余额不足，没有执行后续转账。",
                "Insufficient balance, so the following transfer was not sent.",
            )
        ]
    if tool in {"list_contacts", "find_contact", "get_contact_date", "list_cards", "sandbox_chart"}:
        # Successful read-only steps stay quiet unless they add visible content.
        return []
    if tool in _UNDO_TOOLS and _error_code(step.get("error")) in {
        "already_undone",
        "replayed",
    }:
        return [_t(lang, "这笔已经撤销过了。", "That was already undone.")]

    return _render_write(tool, result, lang, vault, session, now, facts)


def _status_line(
    step: Mapping[str, Any], lang: str, outcome: Any = None
) -> str:
    tool = _tool_name(step.get("tool"), lang)
    status = str(step.get("status") or "")
    error = step.get("error")
    if status == "failed":
        code = _error_code(error)
        message = _plain_error(error, lang, outcome=outcome, step=step)
        attempts = _context_value(
            outcome, step, error, "attempts_left", "attempts_remaining"
        )
        if code in _MISMATCH_CODES and attempts is not None:
            return message
        if code == "locked" or code in {"not_frozen", "card_not_frozen"}:
            return message
        if code in {"already_undone", "replayed"} and step.get("tool") in _UNDO_TOOLS:
            return _t(lang, "这笔已经撤销过了。", "That was already undone.")
        return _t(
            lang,
            f"- {tool}：失败（{message}）",
            f"- {tool}: failed ({message})",
        )
    if status == "skipped":
        if _error_code(error) == "rejected":
            return _t(
                lang,
                f"- {tool}：已跳过（你已取消这一步）",
                f"- {tool}: skipped (you cancelled this step)",
            )
        return _t(
            lang,
            f"- {tool}：因上一步未完成而跳过",
            f"- {tool}: skipped because an earlier step didn't finish",
        )
    if status == "pending":
        return _t(lang, f"- {tool}：等待确认", f"- {tool}: waiting for confirmation")
    return _t(lang, f"- {tool}：已完成", f"- {tool}: done")


def _steps_text(
    steps: list[Any],
    *,
    lang: str,
    vault: Any,
    session: Any,
    bank: Any,
    now: Any,
    facts: Facts,
    outcome: Any = None,
) -> list[str]:
    cancels = [
        index
        for index, step in enumerate(steps)
        if isinstance(step, Mapping)
        and step.get("status") == "done"
        and step.get("tool") == "cancel_subscription"
    ]
    last_cancel = max(cancels) if cancels else None

    lines: list[str] = []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            continue
        if last_cancel is not None and step.get("status") == "done":
            if step.get("tool") == "list_subscriptions":
                # A snapshot read before the cancellation is stale, and any
                # later one would duplicate the single fresh read below.
                continue
        if step.get("status") == "done":
            lines.extend(
                _done_lines(
                    step,
                    lang=lang,
                    vault=vault,
                    session=session,
                    bank=bank,
                    now=now,
                    facts=facts,
                )
            )
        else:
            lines.append(_status_line(step, lang, outcome))
        if last_cancel is not None and index == last_cancel:
            lines.extend(_fresh_active_subscriptions(bank, lang, now, facts))
    return lines


# -------------------------------------------------------------- verdict renders


def _approval_word(
    tier: Any, needs_mfa: bool, needs_co_approval: bool, lang: str
) -> str:
    tier = str(tier or "")
    if tier == "red":
        parts: list[str] = []
        if needs_mfa:
            parts.append(
                _t(lang, "需要短信验证码和刷脸", "an SMS code and face check")
            )
        if needs_co_approval:
            parts.append(_t(lang, "需要家人共同确认", "family co-approval"))
        if parts:
            joiner = _t(lang, "，并且", " and ")
            return joiner.join(parts)
        return _t(lang, "需要确认", "needs confirmation")
    if tier == "yellow":
        return _t(lang, "需要确认", "needs confirmation")
    return _t(lang, "无需额外确认", "no extra confirmation")


def _approval_lines(
    approvals: list[Any],
    *,
    lang: str,
    vault: Any,
    session: Any,
    facts: Facts,
    bank: Any = None,
) -> list[str]:
    lines: list[str] = []
    for approval in approvals:
        if not isinstance(approval, Mapping):
            continue
        amount = _int_or_none(approval.get("amount_fen"))
        if amount is None:
            amount = _int_or_none(approval.get("amount"))
        if amount is None:
            amount = _int_or_none(approval.get("limit_fen"))
        if amount is None:
            amount = _int_or_none(approval.get("limit"))
        money = ""
        if amount is not None and amount != 0:
            facts.add_amount(amount)
            money = f" {_money(amount)}"
        tool = str(approval.get("tool") or "")
        target = _approval_target(approval, lang, vault, session, facts)
        if tool == "set_card_control":
            target += _card_control_approval_detail(approval, bank, session, lang)
        if tool == "create_virtual_card":
            try:
                defaulted = any(entry.kind == "AMT" and entry.plan_id == session.plan_id
                    and entry.real_ref == amount and entry.tags.get("default_virtual_limit")
                    for entry in vault.entries(session))
            except (AttributeError, TypeError):
                defaulted = False
            if defaulted:
                target += _t(lang, "（未指定限额，建议默认限额）", " (no limit supplied; proposed default)")
        if tool == "schedule_transfer":
            day = approval.get("day_of_month")
            if isinstance(day, int) and not isinstance(day, bool):
                target += _t(lang, f"（每月{day}日）", f" (every month on day {day})")
        word = _approval_word(
            approval.get("tier"),
            bool(approval.get("needs_mfa")),
            bool(approval.get("needs_co_approval")),
            lang,
        )
        zh_name, en_name = _ACTION_NAMES.get(tool, ("操作", "Action"))
        suffix = ""
        reason = _approval_reason(approval, lang, facts)
        if reason:
            suffix += _t(lang, f"（{reason}）", f" ({reason})")
        error = approval.get("error")
        note = _plain_error(error, lang) if _error_code(error) else ""
        if note and note != reason:
            suffix += _t(lang, f"（{note}）", f" ({note})")
        lines.append(
            _t(
                lang,
                f"- {zh_name}{money}{target}：{word}{suffix}",
                f"- {en_name}{money}{target}: {word}{suffix}",
            )
        )
    return lines


def _card_control_approval_detail(approval: Mapping[str, Any], bank: Any, session: Any, lang: str) -> str:
    """Describe only this pending card step from its own stored plan."""
    import json
    try:
        rows = bank.conn.execute("SELECT payload_json FROM security_audit WHERE customer_id=? AND session_id=? AND plan_id=? AND kind='PLAN' ORDER BY seq DESC",
                                (session.customer_id, session.session_id, session.plan_id)).fetchall()
        for row in rows:
            plan = json.loads(row[0]).get("output", {})
            for step in plan.get("steps", []):
                if step.get("id") != approval.get("step_id") or step.get("tool") != "set_card_control":
                    continue
                args = step.get("args", {})
                names = _CARD_CONTROLS.get(args.get("control"))
                if names is None:
                    return ""
                name = names[1] if _is_en(lang) else names[0]
                action = _t(lang, "开启", "enable") if args.get("enabled") is True else _t(lang, "关闭", "disable")
                scope = _t(lang, "（包括网购）", " (includes online shopping)") if args.get("control") == "online" else ""
                return f" — {action} {name}{scope}"
    except Exception:
        pass
    return ""


def _approval_target(
    approval: Mapping[str, Any],
    lang: str,
    vault: Any,
    session: Any,
    facts: Facts,
) -> str:
    label = _payee_label(vault, session, approval.get("payee_label"))
    if label:
        return _t(lang, f" 给 {label}", f" to {label}")

    target_display = approval.get("target_display")
    if isinstance(target_display, str) and target_display:
        # A masked tail like "···1234" is an account claim; ground it so the
        # whole approval card is not replaced by the fallback.
        for match in re.finditer(r"[·•∙⋅]{2,}\s*(\d{4})(?!\d)", target_display):
            facts.add_tail(match.group(1))
        return f" {target_display}"

    sources = [approval]
    raw = approval.get("raw")
    if isinstance(raw, Mapping):
        sources.append(raw)
    card = approval.get("card")
    if isinstance(card, Mapping):
        sources.append(card)

    merchant = next(
        (
            source.get("merchant")
            for source in sources
            if isinstance(source, Mapping) and source.get("merchant")
        ),
        None,
    )
    if merchant:
        return f" {merchant}"

    last4 = next(
        (
            source.get(key)
            for source in sources
            if isinstance(source, Mapping)
            for key in ("card_last4", "last4", "account_last4")
            if source.get(key)
        ),
        None,
    )
    if last4:
        digits = "".join(character for character in str(last4) if character.isdigit())
        if digits:
            facts.add_tail(digits)
            return f" {_t(lang, f'尾号{digits[-4:]}', f'card ending {digits[-4:]}')}"
    return ""


def _approval_reason(
    approval: Mapping[str, Any], lang: str, facts: Facts
) -> str:
    reasons = approval.get("reasons")
    if not isinstance(reasons, list) or not reasons:
        return ""
    text = _reason_text(reasons[0], lang)
    if text:
        _add_money_mentions(text, facts)
    return text


# Plain names for every registry tool; never show raw tool ids to users.
_ACTION_NAMES: dict[str, tuple[str, str]] = {
    # reads
    "get_balance": ("查询余额", "Check balance"),
    "list_transactions": ("查询交易", "Look up transactions"),
    "spending_report": ("支出报告", "Spending report"),
    "detect_anomalies": ("检查异常交易", "Check unusual transactions"),
    "list_subscriptions": ("查询订阅", "Look up subscriptions"),
    "list_scheduled": ("查看定时转账", "View standing orders"),
    "stop_standing_order": ("停止定时转账", "Stop standing order"),
    "list_cards": ("查询卡片", "Look up cards"),
    "list_products": ("查询产品", "Look up products"),
    "get_portfolio": ("查询持仓", "Look up holdings"),
    "find_contact": ("查找联系人", "Look up contact"),
    "list_contacts": ("查看联系人", "List contacts"),
    "get_contact_date": ("查询联系人日期", "Look up contact date"),
    "can_cover": ("检查余额", "Check balance"),
    "sandbox_chart": ("生成图表", "Create chart"),
    "run_analysis": ("隔离消费分析", "Isolated spending analysis"),
    # writes
    "set_reminder": ("设置提醒", "Set reminder"),
    "transfer": ("转账", "Transfer"),
    "schedule_transfer": ("定时转账", "Scheduled transfer"),
    "request_payment": ("发起收款", "Payment request"),
    "move_to_pot": ("存入储蓄罐", "Move to savings pot"),
    "merchant_order": ("下单", "Order"),
    "cancel_subscription": ("取消订阅", "Cancel subscription"),
    "reactivate_subscription": ("恢复订阅", "Restore subscription"),
    "list_reminders": ("查看提醒", "View reminders"),
    "freeze_card": ("冻结卡片", "Freeze card"),
    "unfreeze_card": ("解冻卡片", "Unfreeze card"),
    "set_card_limit": ("调整卡片限额", "Change card limit"),
    "create_virtual_card": ("开通虚拟卡", "Create virtual card"),
    "report_card_lost": ("挂失卡片", "Report card lost"),
    "subscribe_product": ("购买理财", "Buy product"),
    "change_password": ("修改密码", "Change password"),
    # bills, quotes, knowledge and newer controls
    "list_bills": ("查看账单", "Bills"),
    "pay_bill": ("缴费", "Bill payment"),
    "installment_quote": ("分期试算", "Instalment quote"),
    "deposit_quote": ("定期试算", "Deposit quote"),
    "forecast_cashflow": ("余额预测", "Balance forecast"),
    "help_answer": ("银行知识", "Banking info"),
    "list_transfers": ("转账状态", "Transfer status"),
    "verify_payee": ("收款人核验", "Payee check"),
    "set_card_control": ("卡片开关", "Card control"),
    "block_merchant": ("消费屏蔽", "Spending block"),
    "unblock_merchant": ("解除屏蔽", "Lift block"),
    # controls
    "ask_user": ("确认信息", "Confirm details"),
    "refuse": ("拒绝请求", "Decline request"),
    "handoff_human": ("转人工", "Hand off to a person"),
    # undo-style writes
    "cancel_transfer": ("撤销转账", "Cancel transfer"),
    "cancel_scheduled": ("撤销定时转账", "Cancel scheduled transfer"),
    "cancel_request": ("撤销收款请求", "Cancel payment request"),
    "restore_subscription": ("恢复订阅", "Restore subscription"),
    "move_from_pot": ("从储蓄罐转出", "Move money from savings pot"),
    "cancel_order": ("取消订单", "Cancel order"),
    "restore_card_limit": ("恢复卡片限额", "Restore card limit"),
    "close_virtual_card": ("关闭虚拟卡", "Close virtual card"),
}


_RELATION_ZH = {
    "parent": "父母",
    "spouse": "配偶",
    "child": "子女",
    "friend": "朋友",
    "colleague": "同事",
    "landlord": "房东",
    "merchant": "商户",
    "other": "其他",
}


def _render_contacts(result: Any, lang: str, facts: Facts) -> list[str]:
    """Payees the customer can send money to: name, ···last4 and relation."""

    contacts = result.get("raw") if isinstance(result, Mapping) else None
    if not isinstance(contacts, list) or not contacts:
        return []
    def field(contact: Any, key: str, default: Any = "") -> Any:
        # Results arrive as dicts once checkpointed, as dataclasses otherwise.
        if isinstance(contact, Mapping):
            return contact.get(key, default)
        return getattr(contact, key, default)

    lines = [_t(lang, "你可以转账给：", "You can send money to:")]
    for contact in contacts:
        name = str(field(contact, "name") or "")
        tail = str(field(contact, "account_no") or "")[-4:]
        if tail.isdigit():
            facts.add_tail(tail)
        relation = str(field(contact, "relation") or "")
        label = _RELATION_ZH.get(relation, relation) if not _is_en(lang) else relation
        new = "" if field(contact, "verified", True) else _t(lang, "，新收款人", ", new payee")
        lines.append(f"- {name} ···{tail}（{label}{new}）" if not _is_en(lang) else f"- {name} ···{tail} ({label}{new})")
    return lines


def _pending_approvals(outcome: Mapping[str, Any]) -> list[Any]:
    approvals = outcome.get("approvals")
    if isinstance(approvals, list) and approvals:
        return approvals
    derived: list[Any] = []
    steps = outcome.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, Mapping) and step.get("status") == "pending":
                derived.append(
                    {
                        "tool": step.get("tool"),
                        "amount_fen": step.get("amount_fen"),
                        "payee_label": step.get("payee_label"),
                        "merchant": step.get("merchant"),
                        "card_last4": step.get("card_last4"),
                        "last4": step.get("last4"),
                        "reasons": step.get("reasons"),
                        "tier": step.get("tier"),
                        "needs_mfa": step.get("needs_mfa"),
                        "needs_co_approval": step.get("needs_co_approval"),
                        "error": step.get("error"),
                    }
                )
    return derived


_CLARIFY_EMPTY_LABELS = frozenset({"相关操作", "the relevant action"})
# The planner's clarify text is shown as the bank's own words, so only the
# questions the planner was trained on (training/gold.py) pass through.  A
# deny-list kept missing coached-scam asks such as "tell me the 6 digits"
#; anything else gets a fixed question.
_CLARIFY_KNOWN_QUESTIONS = frozenset(
    {
        "请问要转多少金额？",
        "How much would you like to transfer?",
        "有两个同名联系人，请问要转给哪一位？",
        "There are two contacts with the same name. Which one should I transfer to?",
        "请先添加并验证这个收款人，然后再帮你转账。",
        "Please add and verify this payee first, then I can send the transfer.",
        # T9: an amount change that names no payee (training data v2).
        "请把收款人和金额完整说一遍，比如“给妈妈转300”。",
        'Please say the full request again, with who and how much, for example "send mom 300".',
        # Asked by code (agent/graph.py) when unfreeze has no unfreeze word.
        "请问是要冻结还是解冻卡片？",
        "Do you want to freeze or unfreeze the card?",
        # Asked by code (agent/graph.py) when a standing order has no usable day.
        "请问每个月几号转？",
        "Which day of the month should it go out?",
        # Asked by code (agent/graph.py) when a payee was never named.
        "请问要转给谁？请说出收款人的名字。",
        "Who should I send it to? Please tell me the payee's name.",
        # Asked by code (agent/graph.py) when a plan schedules without a cue.
        "你是想现在转一次，还是每月定期转？",
        "Do you want to send it once now, or every month?",
    }
)


def _safe_clarify_question(text: str, lang: str, *, has_options: bool) -> str:
    """Return the planner's question only when it is one of the known ones."""

    if text.strip() in _CLARIFY_KNOWN_QUESTIONS:
        return text.strip()
    if has_options:
        return _t(lang, "你指的是哪一个？", "Which one do you mean?")
    return _t(
        lang,
        "请再说具体一点：你想做什么、对象是谁、金额多少？",
        "Could you say a bit more: what you'd like to do, for whom, and how much?",
    )


def _clarify_display_label(
    label: str, *, lang: str, vault: Any, session: Any
) -> str:
    """Leave resolvable tokens intact; scrub what the user could never see."""

    if _resolve_entry(vault, session, label) is not None:
        return label
    return _scrub_internal_tokens(label, lang)


def _clarify_text(
    outcome: Mapping[str, Any], *, lang: str, vault: Any, session: Any, bank: Any
) -> str:
    question = outcome.get("question")
    if isinstance(question, Mapping):
        question = question.get(
            "message_en" if _is_en(lang) else "message_zh"
        ) or question.get("message_zh")
    options = outcome.get("options")
    question_text = str(question) if question else ""
    from agent.intents import is_trusted_question

    session_id = getattr(session, "session_id", None)
    customer_id = getattr(session, "customer_id", None)
    if question_text and not (
        isinstance(session_id, str)
        and isinstance(customer_id, str)
        and is_trusted_question(customer_id, session_id, question_text)
    ):
        question_text = _safe_clarify_question(
            question_text, lang, has_options=isinstance(options, list) and bool(options)
        )
    lines = [question_text] if question_text else []

    if not isinstance(options, list):
        options = []
    for index, option in enumerate(options, start=1):
        notes: list[str] = []
        if isinstance(option, Mapping):
            label = (
                option.get("label")
                or option.get("token")
                or option.get("value")
                or ""
            )
            notes = [
                str(option[key])
                for key in ("relation", "note", "detail")
                if option.get(key)
            ]
            label = str(label)
        else:
            label = str(option)

        label = _clarify_display_label(
            label.strip(), lang=lang, vault=vault, session=session
        )
        if not label.strip() or label.strip() in _CLARIFY_EMPTY_LABELS:
            continue

        extras: list[str] = []
        entry = _resolve_entry(vault, session, label)
        if entry is not None and getattr(entry, "kind", "") == "PAYEE":
            row = _contact_row(bank, getattr(entry, "real_ref", None))
            if row is not None:
                for value in (row["relation"], row["note"]):
                    if value and str(value) not in extras:
                        extras.append(str(value))
        for note in notes:
            if note not in extras:
                extras.append(note)

        text = f"{index}. {label}"
        if extras:
            text += "（" + " · ".join(extras) + "）"
        lines.append(text)
    return _join(lines)


def _status_fallback(outcome: Mapping[str, Any], lang: str) -> str:
    """Minimal safe reply: statuses only, no money, names or dates."""

    steps = outcome.get("steps")
    lines: list[str] = []
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, Mapping):
                lines.append(_status_line(step, lang, outcome))
    if not lines:
        return _t(lang, "未执行操作。", "No action was taken.")
    return _join(lines)


# -------------------------------------------------------------------- compose


def compose_reply(
    outcome: dict,
    *,
    lang: str,
    vault,
    session,
    bank,
    now,
) -> ReplyResult:
    """Turn one finished turn into a short, grounded reply."""

    if not isinstance(outcome, Mapping):
        outcome = {}
    kind = str(outcome.get("kind") or "done")
    steps = outcome.get("steps")
    if not isinstance(steps, list):
        steps = []

    step_results = [
        step.get("result")
        for step in steps
        if isinstance(step, Mapping) and step.get("result") is not None
    ]
    facts = collect_facts(step_results)

    if kind == "needs_approval":
        approvals = _pending_approvals(outcome)
        lines = [_t(lang, "需要你的确认：", "Confirmation needed:")]
        lines.extend(
            _approval_lines(
                approvals, lang=lang, vault=vault, session=session, facts=facts, bank=bank
            )
        )
        if len(lines) == 1:
            lines.append(
                _t(lang, "有操作等待确认。", "An action is waiting for confirmation.")
            )
        draft = _join(lines)
    elif kind == "clarify":
        draft = _clarify_text(
            outcome, lang=lang, vault=vault, session=session, bank=bank
        )
        if not draft:
            draft = _t(lang, "请补充一下信息。", "Please add a little more detail.")
    elif kind == "refused":
        draft = _reason_text(outcome.get("refuse_reason"), lang) or _t(
            lang, "这个请求我不能办理。", "I cannot carry out this request."
        )
    elif kind == "vetoed":
        reasons = outcome.get("veto_reasons")
        if not isinstance(reasons, list):
            reasons = []
        messages = [_reason_text(reason, lang) for reason in reasons]
        messages = [message for message in messages if message]
        if messages:
            header = _t(lang, "没有执行：", "Not executed:")
            draft = _join([header, *[f"- {message}" for message in messages]])
        else:
            draft = _t(lang, "没有执行操作。", "No action was taken.")
    elif kind == "scam":
        scam = outcome.get("scam")
        key = "advice_en" if _is_en(lang) else "advice_zh"
        if isinstance(scam, Mapping):
            advice = scam.get(key)
        else:
            advice = getattr(scam, key, None)
        if not advice:
            advice = _t(
                lang,
                "请立即停止转账和付款，不要按对方要求操作。",
                "Stop any transfer or payment now and do not follow their instructions.",
            )
        pause = _t(lang, "转账已暂停 24 小时。", "Transfers are paused for 24 hours.")
        draft = f"{advice}\n{pause}"
    elif kind == "paused":
        draft = _t(
            lang,
            "转账已暂停，等待冷静期结束。",
            "The transfer is paused for the cooling-off period.",
        )
    elif kind == "error":
        draft = _t(
            lang,
            "出错了，请稍后再试。",
            "Something went wrong. Please try again.",
        )
    elif kind == "done":
        lines = _steps_text(
            steps,
            lang=lang,
            vault=vault,
            session=session,
            bank=bank,
            now=now,
            facts=facts,
            outcome=outcome,
        )
        draft = _join(lines) or _t(lang, "已完成。", "Done.")
    else:
        draft = _t(lang, "已完成。", "Done.")

    fallback = _status_fallback(outcome, lang)
    # Resolve relative dates ("今天") against the injected clock so a reply
    # may mention the current day when a tool result is due today.
    ground_result = check(draft, facts, now)
    text_token = draft if ground_result.ok else fallback
    text = _redact_long_digits(detokenize(text_token, vault, session, bank))
    text = _scrub_internal_tokens(text, lang)
    return ReplyResult(
        text=text,
        grounded=ground_result.ok,
        rejected=[claim.raw for claim in ground_result.unknown],
    )
