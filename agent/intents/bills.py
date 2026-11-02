"""Scenario 2 (bill analysis) intent rules. See agent/intents/__init__.py.

The rules here only ever produce the three existing read tools
(``spending_report``, ``list_transactions`` and ``detect_anomalies``).  A
message that also contains a write is left to the normal planner so a read
rule can never silently drop the write clause.  Follow-ups may inherit a
period/subject only from a cryptographically verified successful read in the
same customer and chat.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import bank.categories as _bank_categories
from bank.categories import CATEGORIES

_READ_TOOLS = frozenset({"spending_report", "list_transactions", "detect_anomalies"})
_ALLOWED_YEAR = range(1900, 2101)
_GENERIC_COUNTERPARTIES = frozenset(
    {
        "我的储蓄账户",
        "储蓄账户",
        "本人",
        "现金",
        "工资",
        "转账",
        "我的账户",
    }
)

# --------------------------------------------------------------- user message

_USER_MARKER_RE = re.compile(r"(?:^|\n)User:\s*", re.IGNORECASE)
_PAYEE_TOKEN_RE = re.compile(r"<PAYEE_[0-9]+>")
_MERCHANT_TOKEN_RE = re.compile(r"<MERCHANT_[0-9]+>")


def _user_text(ctx: Any) -> str:
    """The current turn only, not the bank context rendered above ``User:``."""

    text = str(getattr(ctx, "tokenized_text", "") or "")
    markers = list(_USER_MARKER_RE.finditer(text))
    return text[markers[-1].end() :] if markers else text


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _is_zh(ctx: Any, text: str) -> bool:
    lang = str(getattr(ctx, "lang", "") or "")
    if lang == "zh":
        return True
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _clarify(ctx: Any, reason: str, options: list[str] | None = None) -> dict[str, Any]:
    if reason == "period":
        question = (
            "请问你想看哪个月或哪一年？"
            if _is_zh(ctx, "")
            else "Which month or year would you like to see?"
        )
    elif reason == "subject":
        question = (
            "请问你指的是哪个商家？"
            if _is_zh(ctx, "")
            else "Which merchant do you mean?"
        )
    else:
        question = (
            "请问你想接着看哪段时间的消费？"
            if _is_zh(ctx, "")
            else "Which period would you like to see next?"
        )
    return {"kind": "clarify", "question": question, "options": list(options or [])}


def _plan(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "plan",
        "steps": [{"id": "s1", "tool": tool, "args": args, "after": []}],
    }


# -------------------------------------------------------------------- periods

_MONTH_BEFORE_LAST_RE = re.compile(
    r"\b(?:the\s+)?month\s+before\s+last\b|上上个月|前个月", re.IGNORECASE
)
# "and the month before?" means one month before the previous read, not the
# month two months before now.
_MONTH_BEFORE_FOLLOWUP_RE = re.compile(
    r"\b(?:the\s+)?month\s+before\b(?!\s+last)|前一个月|再上个月", re.IGNORECASE
)
_LAST_7_RE = re.compile(
    r"\b(?:last|past)\s+(?:7|seven)\s+days?\b|\blast\s+week\b|"
    r"最近7天|近7天|过去7天|最近一周|近一周|过去一周|上周",
    re.IGNORECASE,
)
_LAST_30_RE = re.compile(
    r"\b(?:last|past)\s+(?:30|thirty)\s+days?\b|"
    r"最近30天|近30天|过去30天|最近一个月|近一个月|过去一个月",
    re.IGNORECASE,
)
_LAST_MONTH_RE = re.compile(
    r"\blast\s*month\b|\bprevious\s+month\b|上个月|上月", re.IGNORECASE
)
_THIS_MONTH_RE = re.compile(
    r"\bthis\s*month\b|\bcurrent\s*month\b|这个月|这月|本月", re.IGNORECASE
)
_LAST_YEAR_RE = re.compile(r"\blast\s*year\b|去年", re.IGNORECASE)
_THIS_YEAR_RE = re.compile(r"\bthis\s*year\b|今年|本年", re.IGNORECASE)
_ISO_MONTH_RE = re.compile(r"(?<![0-9])([0-9]{4})-([0-9]{1,2})(?![0-9])")
_YEAR_RE = re.compile(r"(?<![0-9])([0-9]{4})(?![0-9])")
_CN_MONTH_RE = re.compile(
    r"(?<![0-9])([0-9]{1,2}|[一二三四五六七八九十两]{1,3})\s*月(?:份)?"
)
_EN_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b",
    re.IGNORECASE,
)
_EN_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_CN_DIGITS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def _month_number(raw: str) -> int | None:
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    if raw == "十":
        return 10
    if "十" in raw:
        left, _, right = raw.partition("十")
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    return _CN_DIGITS.get(raw)


def _month_period(year: int, month: int) -> str:
    return f"month:{year:04d}-{month:02d}"


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    value = year * 12 + (month - 1) + delta
    return value // 12, value % 12 + 1


def _period_month(period: str | None, now: datetime) -> tuple[int, int] | None:
    if not isinstance(period, str):
        return None
    if period.startswith("month:"):
        try:
            year, month = period[6:].split("-", 1)
            return int(year), int(month)
        except (TypeError, ValueError):
            return None
    if period == "this_month":
        return now.year, now.month
    if period == "last_month":
        return _shift_month(now.year, now.month, -1)
    return None


def _period_year(period: str | None, now: datetime) -> int | None:
    if isinstance(period, str) and period.startswith("year:"):
        try:
            return int(period[5:])
        except ValueError:
            return None
    month = _period_month(period, now)
    if month is not None:
        return month[0]
    if period == "this_year":
        return now.year
    if period == "last_year":
        return now.year - 1
    return None


def _parse_period(text: str, now: datetime) -> tuple[str | None, bool]:
    """Return ``(period, invalid)`` for an explicit period in the message."""

    month_before_last = _MONTH_BEFORE_LAST_RE.search(text)
    if month_before_last is not None:
        year, month = _shift_month(now.year, now.month, -2)
        return _month_period(year, month), False

    # Pick the period the customer mentioned first; "compare this month with
    # last month" must not silently become last month's report.
    relative: list[tuple[int, str]] = []
    for pattern, value in (
        (_LAST_7_RE, "last_7_days"),
        (_LAST_30_RE, "last_30_days"),
        (_LAST_MONTH_RE, "last_month"),
        (_THIS_MONTH_RE, "this_month"),
        (_LAST_YEAR_RE, "last_year"),
        (_THIS_YEAR_RE, "this_year"),
    ):
        match = pattern.search(text)
        if match is not None:
            relative.append((match.start(), value))
    if relative:
        return min(relative, key=lambda item: item[0])[1], False

    iso = _ISO_MONTH_RE.search(text)
    if iso is not None:
        year, month = int(iso.group(1)), int(iso.group(2))
        if year not in _ALLOWED_YEAR or not 1 <= month <= 12:
            return None, True
        return _month_period(year, month), False

    year_match = _YEAR_RE.search(text)
    explicit_year = int(year_match.group(1)) if year_match is not None else None
    if explicit_year is not None and explicit_year not in _ALLOWED_YEAR:
        return None, True

    month: int | None = None
    cn_month = _CN_MONTH_RE.search(text)
    if cn_month is not None:
        month = _month_number(cn_month.group(1))
    else:
        en_month = _EN_MONTH_RE.search(text)
        if en_month is not None:
            name = en_month.group(1)
            # "May I see my spending?" is the modal verb, not the month.
            if name.casefold() != "may" or not re.match(r"may\s+i\b", text[en_month.start():], re.IGNORECASE):
                month = _EN_MONTHS[name.casefold()]

    if month is not None:
        if not 1 <= month <= 12:
            return None, True
        year = explicit_year if explicit_year is not None else now.year
        # A named month with no year always means that month in the current
        # year; never silently roll October (now November) into last year.
        return _month_period(year, month), False

    if explicit_year is not None:
        return f"year:{explicit_year}", False
    return None, False


def _default_period(text: str, mode: str) -> str:
    if mode == "highest_month":
        return "this_year"
    if re.search(r"年报|年度|annual|yearly|\byear\b", text, re.IGNORECASE):
        return "this_year"
    return "this_month"


def _previous_month_period(period: str | None, now: datetime) -> str | None:
    month = _period_month(period, now)
    if month is None:
        return None
    year, month_number = _shift_month(month[0], month[1], -1)
    return _month_period(year, month_number)


def _last_year_from_history(period: str | None, now: datetime) -> str | None:
    """Last year is relative to the injected clock, not a remembered month."""
    return f"year:{now.year - 1}"


# ------------------------------------------------------------------- subjects

_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "food": (
        "food",
        "restaurant",
        "restaurants",
        "dining",
        "cafe",
        "coffee",
        "餐饮",
        "吃饭",
        "外卖",
        "餐厅",
        "食品",
        "咖啡",
    ),
    "groceries": (
        "groceries",
        "grocery",
        "supermarket",
        "杂货",
        "超市",
        "菜市场",
        "买菜",
        "生鲜",
        "菜",
    ),
    "transport": (
        "transport",
        "transportation",
        "taxi",
        "taxis",
        "uber",
        "ride",
        "rides",
        "交通",
        "出行",
        "打车",
        "出租车",
        "网约车",
        "公交",
        "地铁",
        "加油",
    ),
    "shopping": ("shopping", "retail", "购物", "买东西", "网购", "商场"),
    "bills_utilities": (
        "bills_utilities",
        "utilities",
        "utility",
        "缴费",
        "水电",
        "燃气",
        "话费",
        "宽带",
        "物业",
    ),
    "housing": ("housing", "rent", "mortgage", "住房", "房租", "租金", "房贷"),
    "entertainment": (
        "entertainment",
        "movies",
        "games",
        "娱乐",
        "电影",
        "游戏",
        "演出",
    ),
    "subscriptions": ("subscriptions", "subscription", "订阅", "会员"),
    "health": (
        "health",
        "medical",
        "pharmacy",
        "hospital",
        "医疗",
        "健康",
        "药店",
        "看病",
        "医院",
    ),
    "education": ("education", "school", "tuition", "教育", "学费", "培训", "幼儿园"),
    "transfer": ("transfer", "转账", "汇款"),
    "income": ("income", "收入", "工资"),
    "other": ("other", "其他"),
}


def _all_category_aliases() -> dict[str, tuple[str, ...]]:
    """Our aliases plus any bilingual aliases added to ``bank.categories``."""

    merged: dict[str, list[str]] = {
        category: list(aliases) for category, aliases in _CATEGORY_ALIASES.items()
    }
    root_aliases = getattr(_bank_categories, "CATEGORY_ALIASES", None)
    if isinstance(root_aliases, dict):
        for key, value in root_aliases.items():
            values = value if isinstance(value, (list, tuple, set)) else (value,)
            if key in CATEGORIES:
                merged.setdefault(key, []).extend(str(item) for item in values)
            elif value in CATEGORIES:
                merged.setdefault(str(value), []).append(str(key))
    return {key: tuple(values) for key, values in merged.items()}


def _category(text: str) -> str | None:
    lowered = text.casefold()
    best: str | None = None
    best_length = 0
    for category, aliases in _all_category_aliases().items():
        if category not in CATEGORIES:
            continue
        for alias in aliases:
            if alias.isascii():
                found = re.search(
                    rf"(?<![a-z]){re.escape(alias)}(?![a-z])", lowered
                )
            else:
                found = alias in text
            if found and len(alias) > best_length:
                best, best_length = category, len(alias)
    return best


def _counterparty_rows(ctx: Any) -> list[Any]:
    conn = getattr(ctx, "conn", None)
    if conn is None:
        return []
    try:
        return conn.execute(
            "SELECT id, counterparty FROM transactions WHERE customer_id = ? "
            "ORDER BY ts DESC, id DESC",
            (getattr(ctx, "customer_id", ""),),
        ).fetchall()
    except Exception:  # noqa: BLE001 - no local data means no merchant match
        return []


def _match_counterparty(ctx: Any, raw_text: str) -> tuple[str | None, bool]:
    """Find one own-customer counterparty named in the text.

    Returns ``("txn:<id>", False)`` on a unique match and ``(None, True)`` when
    several different counterparties match.  Only the opaque transaction id is
    ever emitted; the raw merchant name never enters plan args.
    """

    lowered = raw_text.casefold()
    matches: dict[str, str] = {}
    for row in _counterparty_rows(ctx):
        try:
            name = str(row["counterparty"]).strip()
            txn_id = str(row["id"])
        except (KeyError, IndexError, TypeError):
            continue
        if len(name) < 2 or name in _GENERIC_COUNTERPARTIES:
            continue
        surfaces = {name.casefold()}
        # A country suffix is not normally spoken ("Apple Store" vs "Apple
        # Store US"). Candidates still come exclusively from owned records.
        short = re.sub(r"\s+(?:US|CN|UK|China|中国|美国)$", "", name, flags=re.IGNORECASE)
        if len(short) >= 4:
            surfaces.add(short.casefold())
        if any(surface in lowered for surface in surfaces):
            matches.setdefault(name, txn_id)
    if not matches:
        return None, False

    # "Starbucks" and "Starbucks Reserve" both contain the same text; keep the
    # longest surface form instead of calling that ambiguous.
    kept: list[str] = []
    for name in sorted(matches, key=lambda item: (-len(item), item)):
        folded = name.casefold()
        if any(folded in longer.casefold() for longer in kept):
            continue
        kept.append(name)
    if len(kept) > 1:
        return None, True
    return f"txn:{matches[kept[0]]}", False


def _subject_args(ctx: Any, text: str) -> tuple[dict[str, str], dict[str, Any] | None]:
    user = _user_text(ctx)
    payees = _unique(_PAYEE_TOKEN_RE.findall(user))
    merchants = _unique(_MERCHANT_TOKEN_RE.findall(user))
    if len(payees) > 1:
        # Two named contacts are a real choice; show their known tokens.
        return {}, _clarify(ctx, "subject", payees)
    if len(merchants) > 1:
        return {}, _clarify(ctx, "subject")

    merchant_ref, ambiguous = _match_counterparty(ctx, text)
    if ambiguous:
        return {}, _clarify(ctx, "subject")
    if merchant_ref is not None:
        return {"merchant_ref": merchant_ref}, None

    args: dict[str, str] = {}
    if payees:
        args["payee"] = payees[0]
    if merchants:
        args["merchant"] = merchants[0]
    if payees or merchants:
        return args, None

    if re.search(r"\bat\s+\S+|花在\S+", text, re.IGNORECASE) and _category(text) is None:
        return {}, _clarify(ctx, "subject")
    return args, None


# -------------------------------------------------------------- request kinds

_SPEND_RE = re.compile(
    r"\b(?:spend|spent|spending|expenses?|outgoings?|costs?)\b|"
    r"\bwhere\s+did\s+my\s+money\s+go\b|"
    r"\bhow\s+much\s+(?:on|at|for)\b|\bhow\s+much\s+did\s+i\s+pay\b|"
    r"花了|花销|支出|消费|开销|花费|花掉|钱(?:都)?(?:花|用)(?:在)?哪|花哪了",
    re.IGNORECASE,
)
_INCOME_RE = re.compile(
    r"\b(?:income|earnings?|salary|wages?|paycheck)\b|"
    r"\bhow\s+much\s+(?:did|have)\s+i\s+(?:receive|received|earn|earned)\b|"
    r"\bmoney\s+(?:received|came\s+in|in)\b|"
    r"收入|工资|进账|入账|到账|回款|收到多少钱|赚了多少|挣了多少",
    re.IGNORECASE,
)
_TX_RE = re.compile(
    r"\b(?:transactions?|purchases?|payments?|charges?|statement|history)\b|"
    r"流水|交易|记录|明细|消费记录|付款记录|买了什么|哪些消费|消费情况|扣款|对账|进货",
    re.IGNORECASE,
)
_ANOMALY_RE = re.compile(
    r"\b(?:anomal\w*|unusual|strange|weird|suspicious|duplicate|foreign|"
    r"fraud(?:ulent)?)\b|异常|奇怪|可疑|不明|重复扣款|境外|盗刷",
    re.IGNORECASE,
)
_ANOMALY_EXPLAIN_RE = re.compile(
    r"\bwhy\b[^?.!]*\b(?:strange|weird|unusual|suspicious|anomal\w*)\b|"
    r"\bexplain\b[^?.!]*\b(?:strange|unusual|suspicious|anomal\w*)\b|"
    r"\b(?:why|explain)\b[^?.!]*\b(?:this|that|the)\s+(?:one|transaction|charge)\b|"
    r"为什么[^？?。!！]*(?:奇怪|异常|可疑)|"
    r"(?:奇怪|异常|可疑)[^？?。!！]*为什么|解释[^？?。!！]*(?:异常|奇怪)",
    re.IGNORECASE,
)
_HIGHEST_RE = re.compile(
    r"\b(?:which|what)\s+month\b[^?.!]*\b(?:most|highest|biggest|max(?:imum)?)\b|"
    r"\b(?:most|highest|biggest)\b[^?.!]*\bmonth\b|"
    r"哪(?:个|一)?月(?:份)?[^？?。!！]*(?:最多|最高|最大)|"
    r"(?:最多|最高|最大)[^？?。!！]*月",
    re.IGNORECASE,
)
_REPORT_RE = re.compile(
    r"\b(?:bill\s+report|spending\s+report|expense\s+report|annual\s+report|"
    r"yearly\s+report|report|summary|annual|yearly)\b|"
    r"账单|报告|汇总|总结|小结|消费报告|支出报告|年报|年度|月报",
    re.IGNORECASE,
)
_COMPARE_RE = re.compile(r"\bcompare(?:d|s)?\b|\bvs\.?\b|比较|对比|相比", re.IGNORECASE)
_COMPARE_OTHER_RE = re.compile(
    r"\b(?:products?|cards?|subscriptions?|deposits?|loans?|holdings?|"
    r"portfolios?|rates?)\b|产品|卡片|订阅|存款|贷款|持仓|利率",
    re.IGNORECASE,
)
_SAVINGS_EXCLUSION_RE = re.compile(
    r"\bsavings?\b[^?.!]*\b(?:spend|spending|expense|count|counted|include|exclude)\b|"
    r"(?:储蓄|存款)[^？?。!！]*(?:算|包括|不算|支出|消费)",
    re.IGNORECASE,
)
_FOLLOWUP_CUE_RE = re.compile(
    r"^\s*(?:and|what\s+about|how\s+about|then|also|那|还有|那么|换成|至于)|"
    r"呢\s*[？?]?\s*$",
    re.IGNORECASE,
)

_WRITE_EN_RE = re.compile(
    r"\b(?:transfer(?:red)?s?|wire|remit|move|withdraw|deposit|top\s*up|"
    r"recharge|buy|purchase|order|subscribe|cancel|freeze|unfreeze|block|"
    r"unblock|schedule|split|redeem|invest|repay|save)\b"
    r"(?!\s+(?:records?|history|statement|report|fees?|counted|included|"
    r"excluded|part\s+of))"
    r"|\bsend\s+(?:money|funds|cash|¥|￥|\$|[0-9]|<AMT_|"
    r"(?!me\b)[A-Za-z\u4e00-\u9fff][\w\u4e00-\u9fff ]{0,24}"
    r"\s+(?:<AMT_|[¥￥$]?\d))"
    r"|\bpay\s+(?:my|the|a|an|this|that|bill|rent|tuition|utilities|"
    r"electricity|water|gas|property|mobile|broadband|school)\b"
    r"|\bpay\s+(?:<[A-Z]+_[A-Za-z0-9]+>|[¥￥$]?\s*[0-9]+(?:\.[0-9]+)?)\b"
    r"|\bpay\s+(?!at\b|for\b|in\b|with\b|by\b)[A-Za-z\u4e00-\u9fff]"
    r"[\w\u4e00-\u9fff ]{0,24}\s+(?:<AMT_|[¥￥$]?\d)"
    r"|\bgive\s+(?!me\b)[A-Za-z\u4e00-\u9fff][\w\u4e00-\u9fff ]{0,24}"
    r"\s+(?:<AMT_|[¥￥$]?\d)"
    r"|\b(?:please|kindly|can\s+you|could\s+you|would\s+you|i\s+want\s+to|"
    r"i'd\s+like\s+to|i\s+will|i'm\s+going\s+to)\s+pay\b",
    re.IGNORECASE,
)
_WRITE_ZH_RE = re.compile(
    r"(?:给|向|把|帮我把)[^，。！？!?；;\n]{0,24}"
    r"(?:转|汇|打钱|打款|付款|支付|存|取|买|缴|交)"
    r"|(?:转|汇)\s*(?:<AMT_|[¥￥]|[0-9]|[一二三四五六七八九十百千万两])"
    r"|(?:我要|我想|请|帮我|麻烦|能不能|可以|要不要|先别|别|不要)"
    r"[^，。！？!?；;\n]{0,16}(?:转|汇|付款|支付|买|存|取|缴|充值|提现)"
    r"|(?:转给|汇给|打钱给|打款给|付款给)"
    r"|(?:付款|支付|缴费|交费|付[^，。！？!?；;\n]{0,6}费|"
    r"交[^，。！？!?；;\n]{0,6}费|还款|还钱|提现|充值|下单|购买|取消|"
    r"冻结|解冻|挂失|补卡|屏蔽|赎回)"
    r"(?![^，。！？!?；;\n]{0,8}(?:记录|流水|明细|报告|多少|了几|过几|情况|历史))"
)


_CARD_ACTION_RE = re.compile(
    r"虚拟卡|开卡|开张|开一张|冻结|解冻|挂失|限额|"
    r"\bvirtual\s+card\b|\bfreeze\b|\bunfreeze\b|\bcard\s+limit\b",
    re.IGNORECASE,
)


def _has_write(text: str) -> bool:
    # A card action is a write for the planner, even when the turn also asks a
    # spending question; never answer the whole turn from the bill rules.
    if _CARD_ACTION_RE.search(text):
        return True
    # These action clauses can contain report/category words, but must remain
    # whole requests for the planner and existing approval flow.
    if re.search(r"\b(?:send|give)\s+(?!me\b)|\b(?:enable|disable|restore|raise|lower|set)\b|\bturn\b.{0,45}\b(?:on|off)\b", text, re.IGNORECASE):
        return True
    if re.search(r"预留|订花|订蛋糕|订一|帮我交|帮我付|帮我缴|开启|关闭|恢复|调整|设置", text):
        return True
    return bool(_WRITE_EN_RE.search(text) or _WRITE_ZH_RE.search(text))


def _kind(text: str) -> str | None:
    if _HIGHEST_RE.search(text):
        return "highest_month"
    if _INCOME_RE.search(text):
        return "income"
    if _ANOMALY_RE.search(text):
        return "anomaly"
    if _REPORT_RE.search(text):
        return "report"
    if _TX_RE.search(text):
        return "transactions"
    if _SPEND_RE.search(text) or _SAVINGS_EXCLUSION_RE.search(text):
        return "spending"
    return None


def _is_followup(text: str, kind: str | None, period: str | None) -> bool:
    if _ANOMALY_EXPLAIN_RE.search(text):
        return True
    if _COMPARE_RE.search(text) and kind is None:
        return not bool(_COMPARE_OTHER_RE.search(text))
    if kind is not None:
        return False
    if period is not None and (
        _FOLLOWUP_CUE_RE.search(text) or len(text.strip()) <= 32
    ):
        return True
    return bool(_FOLLOWUP_CUE_RE.search(text)) and len(text.strip()) <= 48


# ------------------------------------------------------------------- history


@dataclass(frozen=True)
class _Read:
    tool: str
    args: dict[str, Any]

    @property
    def period(self) -> str | None:
        value = self.args.get("period")
        return value if isinstance(value, str) else None


def _history_reads(ctx: Any) -> list[_Read]:
    """Recent verified successful reads for this exact customer and chat."""

    conn = getattr(ctx, "conn", None)
    customer_id = str(getattr(ctx, "customer_id", "") or "")
    session_id = str(getattr(ctx, "session_id", "") or "")
    secret_key = getattr(ctx, "secret_key", None)
    if conn is None or not customer_id or not session_id or not secret_key:
        return []
    try:
        from security.audit import AuditLog

        if not AuditLog(conn, secret_key).verify().ok:
            return []
        rows = conn.execute(
            "SELECT plan_id, kind, payload_json FROM security_audit "
            "WHERE customer_id = ? AND session_id = ? "
            "AND kind IN ('PLAN', 'RUN') ORDER BY seq ASC",
            (customer_id, session_id),
        ).fetchall()
    except Exception:  # noqa: BLE001 - broken history never yields inheritance
        return []

    plans: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["kind"] != "PLAN":
            continue
        try:
            output = json.loads(row["payload_json"]).get("output")
        except (TypeError, ValueError, AttributeError):
            continue
        if not isinstance(output, dict) or output.get("kind") != "plan":
            continue
        steps = output.get("steps")
        if not isinstance(steps, list):
            continue
        parsed: dict[str, dict[str, Any]] = {}
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_id = step.get("id")
            tool = step.get("tool")
            args = step.get("args") or {}
            if isinstance(step_id, str) and isinstance(tool, str) and isinstance(args, dict):
                parsed[step_id] = {"tool": tool, "args": args}
        plans[str(row["plan_id"])] = parsed

    reads: list[_Read] = []
    for row in reversed(rows):
        if row["kind"] != "RUN":
            continue
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            continue
        tool = payload.get("tool")
        if tool not in _READ_TOOLS:
            continue
        step = plans.get(str(row["plan_id"]), {}).get(str(payload.get("step_id") or ""))
        if step is None or step.get("tool") != tool:
            continue
        reads.append(_Read(tool=str(tool), args=dict(step.get("args") or {})))
    return reads


def _last_anomaly_read(ctx: Any) -> _Read | None:
    for read in _history_reads(ctx):
        if read.tool == "detect_anomalies":
            return read
        if read.tool == "list_transactions" and read.args.get("explain_anomalies") is True:
            return read
    return None


# --------------------------------------------------------------- plan builder


def _spending_plan(
    period: str,
    mode: str,
    *,
    subject: dict[str, str] | None = None,
    compare: bool = False,
) -> dict[str, Any]:
    args: dict[str, Any] = {"period": period}
    args.update(subject or {})
    if mode != "summary":
        args["mode"] = mode
    if compare:
        args["compare"] = True
    return _plan("spending_report", args)


def _transactions_plan(
    period: str,
    *,
    subject: dict[str, str] | None = None,
    explain_anomalies: bool = False,
) -> dict[str, Any]:
    args: dict[str, Any] = {"period": period}
    args.update(subject or {})
    if explain_anomalies:
        args["explain_anomalies"] = True
    return _plan("list_transactions", args)


def _subject_from_read(read: _Read) -> dict[str, str]:
    subject: dict[str, str] = {}
    for key in ("category", "payee", "merchant", "merchant_ref"):
        value = read.args.get(key)
        if not isinstance(value, str) or not value:
            continue
        if key == "category" and value not in CATEGORIES:
            continue
        if key == "payee" and _PAYEE_TOKEN_RE.fullmatch(value) is None:
            continue
        if key == "merchant" and _MERCHANT_TOKEN_RE.fullmatch(value) is None:
            continue
        if key == "merchant_ref" and not value.startswith("txn:"):
            continue
        subject[key] = value
    return subject


def _followup_plan(
    ctx: Any,
    previous: _Read,
    text: str,
    period: str | None,
    *,
    compare: bool = False,
    month_before: bool = False,
) -> dict[str, Any] | None:
    subject = _subject_from_read(previous)
    # Token labels are scoped to a plan and can be reassigned on the next
    # turn. Only stable ledger references are safe to inherit automatically.
    if any(key in subject for key in ("payee", "merchant")):
        return _clarify(ctx, "subject")
    category = _category(text)
    if category is not None:
        subject["category"] = category
    if _ANOMALY_EXPLAIN_RE.search(text):
        inherited = previous.period or "last_30_days"
        return _transactions_plan(inherited, subject=subject, explain_anomalies=True)

    if compare:
        # A comparison follow-up keeps the subject and period of the read it
        # follows; without verified history the current month is the default.
        # After a yearly read, "compare with last month" means month vs month.
        inherited = previous.period or "this_month"
        if _period_month(inherited, ctx.now) is None and re.search(
            r"\bmonth\b|上个?月|本月|这个?月", text, re.IGNORECASE
        ):
            inherited = "this_month"
        return _spending_plan(
            inherited,
            "summary",
            subject=subject,
            compare=True,
        )

    if re.search(r"全年|整年|\b(?:whole|full)\s+year\b", text, re.IGNORECASE):
        year = _period_year(previous.period, ctx.now) or ctx.now.year
        period = f"year:{year}"
    elif month_before:
        chosen = _previous_month_period(previous.period, ctx.now)
        if chosen is None:
            return None
        period = chosen
    elif _LAST_YEAR_RE.search(text):
        chosen = _last_year_from_history(previous.period, ctx.now)
        if chosen is None:
            return None
        period = chosen
    elif period is None:
        period = previous.period or "this_month"

    if previous.tool == "spending_report":
        mode = str(previous.args.get("mode") or "summary")
        return _spending_plan(period, mode, subject=subject)
    if previous.tool == "list_transactions":
        explain = previous.args.get("explain_anomalies") is True
        return _transactions_plan(period, subject=subject, explain_anomalies=explain)
    # A follow-up to a general anomaly check is the specific form.
    return _transactions_plan(period, subject=subject, explain_anomalies=True)


# --------------------------------------------------------------------- entry


def resolve(ctx: Any) -> dict[str, Any] | None:
    raw = str(getattr(ctx, "raw_text", "") or "").strip()
    if not raw:
        return None
    if re.search(r"\b(?:forecast|predict|projection)\b|预测|预计", raw, re.IGNORECASE):
        return None
    # A mixed read/write message must reach the normal planner with its write
    # clause intact.  Never answer it with a read plan from this module.
    if _has_write(raw):
        return None

    # The specialised sandbox rule runs after bills. Do not turn its weekday
    # or hourly comparison into an ordinary monthly total.
    from agent.intents.analysis import match_kind

    if match_kind(raw) is not None:
        return None

    month_before = bool(
        _MONTH_BEFORE_FOLLOWUP_RE.search(raw)
        and not _MONTH_BEFORE_LAST_RE.search(raw)
    )
    period, invalid = _parse_period(raw, ctx.now)
    if invalid:
        return _clarify(ctx, "period")

    if _SAVINGS_EXCLUSION_RE.search(raw) and period is None:
        question = (
            "转入你本人储蓄账户的钱不算消费，不计入支出统计。"
            if _is_zh(ctx, raw)
            else "Money moved to your own savings account is not counted as spending."
        )
        return {"kind": "clarify", "question": question, "options": []}

    kind = _kind(raw)
    category = _category(raw)
    if kind is None and category is not None and period is not None:
        kind = "spending"
    compare = bool(_COMPARE_RE.search(raw))

    if kind == "anomaly" and period is not None:
        # A complete dated request is self-contained even when it also says
        # "explain"; it must not require an earlier anomaly report.
        subject, question = _subject_args(ctx, raw)
        if question is not None:
            return question
        return _transactions_plan(period, subject=subject, explain_anomalies=True)

    if _ANOMALY_EXPLAIN_RE.search(raw):
        previous = _last_anomaly_read(ctx)
        if previous is None:
            return _clarify(ctx, "history")
        subject, question = _subject_args(ctx, raw)
        if question is not None:
            return question
        inherited = dict(_subject_from_read(previous))
        if any(key in inherited for key in ("payee", "merchant")):
            return _clarify(ctx, "subject")
        inherited.update(subject)
        return _transactions_plan(
            period or previous.period or "last_30_days",
            subject=inherited,
            explain_anomalies=True,
        )

    if _is_followup(raw, kind, period) or (
        kind is None and category is not None
        and re.search(r"^(?:只看|只要|仅看|only\b|just\b)", raw, re.IGNORECASE)
    ):
        history = _history_reads(ctx)
        if not history:
            if compare and kind is None:
                # "compare with last month" works without history by comparing
                # the default current period; nothing is inherited.
                subject, question = _subject_args(ctx, raw)
                if question is not None:
                    return question
                category = _category(raw)
                if category is not None:
                    subject.setdefault("category", category)
                return _spending_plan(
                    "this_month", "summary", subject=subject, compare=True
                )
            return _clarify(ctx, "history")
        plan = _followup_plan(
            ctx,
            history[0],
            raw,
            period,
            compare=compare,
            month_before=month_before,
        )
        if plan is not None:
            return plan
        return _clarify(ctx, "period")

    if kind is None:
        return None

    subject, question = _subject_args(ctx, raw)
    if question is not None:
        return question
    if kind != "income":
        category = _category(raw)
        if category is not None:
            subject.setdefault("category", category)

    if kind == "anomaly":
        specific = bool(subject) or bool(
            re.search(
                r"\b(?:this|that|the)\s+(?:one|transaction|charge)\b|这笔|那笔|这一笔",
                raw,
                re.IGNORECASE,
            )
        )
        if specific:
            return _transactions_plan(
                period or "last_30_days",
                subject=subject,
                explain_anomalies=True,
            )
        if period is not None:
            # "unusual transactions last month" means that month, not the
            # rolling 30 days the plain detector covers.
            return _transactions_plan(period, explain_anomalies=True)
        return _plan("detect_anomalies", {})

    if kind == "transactions":
        return _transactions_plan(period or "this_month", subject=subject)

    if kind == "highest_month":
        return _spending_plan(period or "this_year", "highest_month", subject=subject)
    if kind == "income":
        return _spending_plan(period or "this_month", "income", subject=subject)
    if kind == "report":
        return _spending_plan(period or _default_period(raw, "report"), "report", subject=subject)
    return _spending_plan(
        period or "this_month", "summary", subject=subject, compare=compare
    )
