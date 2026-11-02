"""Read-only routing for schema-constrained, isolated spending analysis.

Only a bounded analysis kind and period cross into the plan. User wording is
never put into tool arguments or forwarded to the code generator by this rule.
"""

from __future__ import annotations

import re
from typing import Any

from agent.intents import bills

_SPENDING = re.compile(r"\b(?:spend(?:ing)?|expenses?|expenditure|transactions?)\b|花钱|花费|花[得的]?最多|消费|支出", re.I)
_DECLINE = re.compile(r"\b(?:do\s+not|don't|never)\s+(?:analyse|analyze|compare)|不要(?:分析|比较)|别(?:分析|比较)", re.I)
_FORECAST = re.compile(r"\b(?:forecast|predict|projection)\b|预测|预计", re.I)


def match_kind(text: str) -> str | None:
    """Recognise supported read analyses without consuming ordinary bill queries."""
    if not text or _DECLINE.search(text) or _FORECAST.search(text):
        return None
    spending = bool(_SPENDING.search(text))
    weekend = bool(re.search(r"\bweekends?\b|周末", text, re.I))
    weekday = bool(re.search(r"\bweekdays?\b|工作日|平日", text, re.I))
    if weekend and weekday and (spending or re.search(r"比较|不同|区别|\b(?:compare|vs|versus|difference)\b", text, re.I)):
        return "weekend_vs_weekday"
    if spending and re.search(r"\b(?:time\s+of\s+day|what\s+time|which\s+hour|by\s+hour|hourly)\b|几点|哪个时段|什么时段|按小时|分时段", text, re.I):
        return "by_hour"
    if spending and re.search(r"\b(?:which\s+weekday|what\s+day\s+of\s+the\s+week|by\s+weekday)\b|星期几|周几|按星期|按周几", text, re.I):
        return "by_weekday"
    if spending and re.search(r"\b(?:analyse|analyze|analysis)\b|分析", text, re.I):
        return "custom"
    return None


def resolve(ctx: Any) -> dict[str, Any] | None:
    raw = str(getattr(ctx, "raw_text", "") or "").strip()
    if bills._has_write(raw):
        return None
    kind = match_kind(raw)
    if kind is None:
        return None
    period, invalid = bills._parse_period(raw, ctx.now)
    if invalid:
        return {
            "kind": "clarify",
            "question": "请问你想分析哪个月或哪一年？" if getattr(ctx, "lang", "en") == "zh" else "Which month or year would you like to analyse?",
            "options": [],
        }
    if period is None:
        period = "this_year" if re.search(r"\byear\b|今年", raw, re.I) else "last_30_days"
    return {
        "kind": "plan",
        "steps": [{"id": "s1", "tool": "run_analysis", "args": {"period": period, "kind": kind}, "after": []}],
    }
