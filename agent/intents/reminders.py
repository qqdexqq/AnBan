"""General dated-reminder intent rules.

A reminder about a named subscription belongs to the subscription module (it
knows the renewal date); this module only handles a plain "remind me ..." whose
date is in the turn.  See agent/intents/__init__.py for the rule contract.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from datetime import timedelta
from typing import Any

from agent.intents.subscriptions import _BRANDS, _SUBW, _has_any
from security.tokens import Session, relative_date_text

_REMIND = ("remind", "reminder", "reminders", "提醒", "提醒我")
_LIST = ("what", "which", "show", "list", "哪些", "有什么", "查看")
_DATE_RE = re.compile(r"<DATE_[A-Za-z0-9]+>")
_TOKEN_RE = re.compile(r"<[^<>\n]{0,120}>")
_USER_MARKER_RE = re.compile(r"(?:^|\n)User:\s*", re.IGNORECASE)

# Bare weekdays the tokenizer does not mint (it only handles "next friday").
_EN_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_EN_WEEKDAY_RE = re.compile(
    r"(?<![A-Za-z])(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)
_CN_WEEKDAY_RE = re.compile(r"(?<![上下])(?:周|星期)([一二三四五六日天])")
_CN_WEEKDAYS = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}

_EN_FILLER_RE = re.compile(
    r"(?<![A-Za-z])(?:reminders|reminder|remind|me|to|on|please)(?![A-Za-z])",
    re.IGNORECASE,
)
_CN_FILLER = ("提醒我", "提醒", "帮我", "我")


def _step(tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": "s1", "tool": tool, "args": args or {}, "after": []}


def _plan(tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"kind": "plan", "steps": [_step(tool, args)]}


def _clarify(question: str) -> dict[str, Any]:
    return {"kind": "clarify", "question": question, "options": []}


def _session(ctx: Any) -> Session:
    return Session(
        session_id=ctx.session_id,
        plan_id=ctx.plan_id,
        customer_id=ctx.customer_id,
    )


def _user_text(tokenized: str) -> str:
    markers = list(_USER_MARKER_RE.finditer(tokenized))
    return tokenized[markers[-1].end() :] if markers else tokenized


def _lang(ctx: Any) -> str:
    return "zh" if ctx.lang == "zh" else "en"


def _subscription_named(text: str, tokenized: str) -> bool:
    return (
        "<SUB_" in tokenized
        or _has_any(text, _BRANDS)
        or _has_any(text, _SUBW)
    )


def _bare_weekday(text: str, now: Any) -> tuple[Any, str] | None:
    """A plain weekday ("friday", "周五") and the surface to strip from the note."""

    match = _EN_WEEKDAY_RE.search(text)
    if match is not None:
        weekday = _EN_WEEKDAYS[match.group(1).casefold()]
        surface = match.group(0)
    else:
        match = _CN_WEEKDAY_RE.search(text)
        if match is None:
            return None
        weekday = _CN_WEEKDAYS[match.group(1)]
        surface = match.group(0)
    delta = (weekday - now.date().weekday()) % 7
    return now.date() + timedelta(days=delta), surface


def _mint_date(ctx: Any, target: Any) -> str:
    entry = ctx.vault.mint(
        _session(ctx),
        "DATE",
        target,
        OrderedDict([("relative", relative_date_text(target, ctx.now.date()))]),
        "user_text",
    )
    return entry.label


def _note(text: str, surface: str | None) -> str:
    if surface:
        text = re.sub(re.escape(surface), " ", text, flags=re.IGNORECASE)
    text = _TOKEN_RE.sub(" ", text)
    text = _EN_FILLER_RE.sub(" ", text)
    for word in _CN_FILLER:
        text = text.replace(word, " ")
    text = re.sub(r"[^A-Za-z\u4e00-\u9fff ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:20].strip()


def _resolve(ctx: Any) -> dict[str, Any] | None:
    text = ctx.raw_text if isinstance(ctx.raw_text, str) else ""
    tokenized = ctx.tokenized_text if isinstance(ctx.tokenized_text, str) else ""
    if not text.strip():
        return None
    user_text = _user_text(tokenized)

    if not _has_any(text, _REMIND):
        return None
    if _subscription_named(text, user_text):
        return None
    if _has_any(text, _LIST):
        return None

    labels = _DATE_RE.findall(user_text)
    surface: str | None = None
    if labels:
        when = labels[0]
    else:
        weekday = _bare_weekday(user_text, ctx.now)
        if weekday is None:
            question = (
                "你想让我哪天提醒你？"
                if _lang(ctx) == "zh"
                else "Which day should I remind you?"
            )
            return _clarify(question)
        target, surface = weekday
        when = _mint_date(ctx, target)

    args: dict[str, Any] = {"when": when}
    note = _note(user_text, surface)
    if note:
        args["note"] = note
    return _plan("set_reminder", args)


def resolve(ctx: Any) -> dict[str, Any] | None:
    try:
        return _resolve(ctx)
    except Exception:  # noqa: BLE001 - a rule must never break a turn
        return None
