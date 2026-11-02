"""Scenario 5 (subscriptions) intent rules. See agent/intents/__init__.py."""

from __future__ import annotations

import calendar
import json
import re
from collections import OrderedDict
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

from bank.api import Bank
from bank.recurring import subscription_category
from security.tokens import (
    SUB_ALIASES,
    Session,
    _brand_surface_forms,
    _subscription_tags,
    relative_date_text,
)

_SUB_TOKEN_RE = re.compile(r"<SUB_[A-Za-z0-9]+>")
_INTEGER_RE = re.compile(r"(?<![A-Za-z0-9])([0-9]{1,2})(?![A-Za-z0-9])")
_CN_DAY_RE = re.compile(r"([一二两三四五六七八九十])\s*天")
_CN_DAY_VALUES = {
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
    "十": 10,
}

_CANCEL = (
    "cancel",
    "cancle",
    "cancl",
    "cancell",
    "canel",
    "unsubscribe",
    "unsub",
    "stop",
    "取消",
    "取消掉",
    "退订",
    "退了",
    "退掉",
    "停掉",
    "关掉",
    "不要了",
)
_REMIND = ("remind", "reminder", "reminders", "rmind", "提醒")
_LISTQ = ("what", "which", "show", "list", "哪些", "有什么", "列出", "看看", "查看", "有没有")
# Other banking objects: a turn about these is never a subscription rule's business.
_OTHER_DOMAIN = (
    "transfer", "transfers", "payment", "payments", "pay", "order", "orders", "card",
    "cards", "pot", "pots", "request", "requests", "scheduled", "transaction",
    "transactions", "loan", "loans", "fund", "funds", "转账", "汇款", "付款", "订单", "卡", "储蓄", "基金", "理财",
)
# Well-known subscription brands, so "cancel netflix" is recognised without the word "subscription".
_BRANDS = (
    "netflix", "spotify", "youtube", "disney", "hbo", "hulu", "prime", "icloud",
    "apple music", "apple tv", "chatgpt", "notion", "dropbox", "onedrive", "iqiyi", "youku",
    # Chinese names, so "取消奈飞会员" is checked against the customer's own list too.
    "奈飞", "网飞", "爱奇艺", "优酷", "芒果tv", "网易云音乐", "qq音乐", "酷狗", "百度网盘", "迅雷",
    "喜马拉雅", "知乎", "美团会员", "饿了么会员", "京东plus", "88vip", "哔哩哔哩", "腾讯视频",
    "b站", "夸克", "王者荣耀", "爱奇艺会员",
)
_RESTORE = ("undo", "restore", "resubscribe", "reactivate", "恢复", "撤销", "重新订阅", "重新开通")
_SUBW = (
    "subscription",
    "subscriptions",
    "sub",
    "subs",
    "membership",
    "订阅",
    "会员",
    "自动扣费",
    "续费",
    "扣费",
)
_ALL = ("all", "所有", "全部", "全都")
# 都 only counts right before a cancel verb ("都退了", "都取消").
_ALL_ZH_RE = re.compile(r"都(?:给我)?(?:退|取消|停|关)")
# Negation, exceptions and questions: never turn these into a bulk action.
_HEDGE_RE = re.compile(
    r"(?<![A-Za-z])(?:not|don'?t|do\s+not|never|except|keep|but|how|why|what\s+if|should|would|will)(?![A-Za-z])"
    r"|不想|不要取消|别取消|先别|除了|保留|会不会|吗|怎么|如何|为什么|[?？]",
    re.IGNORECASE,
)
_NEVER = (
    "never use",
    "don't use",
    "dont use",
    "not using",
    "unused",
    # Not bare 不用/没用: "不用了" just means "never mind".
    "不用的",
    "没用的",
    "没用过",
    "没在用",
    "很久没用",
    "不怎么用",
    "从来不用",
    "从不用",
)
_CHARGED = (
    "charged for",
    "did i get charged",
    "扣过",
    "扣了没",
    "扣费了吗",
    "有没有扣",
)

_CATEGORY_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("video", ("video", "videos", "vid", "vids", "streaming", "视频")),
    ("music", ("music", "音乐")),
    ("cloud", ("cloud", "storage", "网盘", "云盘")),
    ("game", ("game", "games", "游戏")),
)

_CATEGORY_LABELS = {
    "video": ("视频类", "video"),
    "music": ("音乐类", "music"),
    "cloud": ("云存储类", "cloud storage"),
    "game": ("游戏类", "game"),
}

_STOPWORDS = frozenset(
    {
        "cancel",
        "unsubscribe",
        "stop",
        "remind",
        "reminder",
        "me",
        "my",
        "the",
        "please",
        "pls",
        "plz",
        "subscription",
        "subscriptions",
        "sub",
        "subs",
        "before",
        "renews",
        "renew",
        "renewal",
        "it",
        "that",
        "this",
        "one",
        "for",
        "to",
        "of",
        "and",
        "all",
        "every",
        "when",
        "about",
        "can",
        "you",
        "could",
        "would",
        "want",
        "need",
        "help",
        "account",
        "monthly",
        "membership",
        "app",
        "cancle", "cancl", "canel", "cancell", "unsub", "rmind", "from", "thing",
        "stuff", "lol", "the", "just", "now", "get", "rid", "undo", "restore",
        "resubscribe", "reactivate", "bring", "back", "again", "turn", "off",
        "day", "days", "week", "weeks", "month", "months", "year",
        "after", "expires", "ends", "charge", "charged", "charges", "date",
    }
)

_DETECT_EXTRA = re.compile(
    r"(?:aren'?t|isn'?t|not)\s+in\s+my\b[^.!?。！？]{0,40}?\blist\b",
    re.IGNORECASE,
)


def _is_cjk(word: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in word)


@lru_cache(maxsize=None)
def _english_pattern(word: str) -> re.Pattern[str]:
    parts: list[str] = []
    for char in word.casefold():
        if char == "'":
            parts.append(r"'?")
        elif char.isspace():
            parts.append(r"\s+")
        else:
            parts.append(re.escape(char))
    body = "".join(parts)
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    folded = text.casefold()
    for word in words:
        if _is_cjk(word):
            if word.casefold() in folded:
                return True
        elif _english_pattern(word).search(text):
            return True
    return False


def _sub_surfaces(sub: Any) -> set[str]:
    """Every surface the customer may use to name this subscription."""

    forms = set(_brand_surface_forms(sub.merchant))
    forms.add(sub.merchant)
    for alias, canonical in SUB_ALIASES.items():
        if canonical == sub.merchant:
            forms.add(alias)
    for brand in _BRANDS:
        if brand.casefold() in sub.merchant.casefold():
            forms.add(brand)
    return {form for form in forms if form.strip()}


def _name_surfaces(subs: list[Any]) -> tuple[str, ...]:
    surfaces = set(_BRANDS)
    for sub in subs:
        surfaces.update(_sub_surfaces(sub))
    # Longest first, so the full merchant name is removed before its parts.
    return tuple(sorted(surfaces, key=len, reverse=True))


def _category(text: str, subs: list[Any]) -> str | None:
    """Category word left after every merchant/alias name is removed."""

    stripped = text
    for surface in _name_surfaces(subs):
        stripped = re.sub(re.escape(surface), "", stripped, flags=re.IGNORECASE)
    for category, words in _CATEGORY_WORDS:
        if _has_any(stripped, words):
            return category
    return None


def _is_hedged(text: str) -> bool:
    """Negation/exception/question wording, ignoring never-used phrases."""

    unhedged = text
    for phrase in _NEVER:
        unhedged = re.sub(re.escape(phrase), " ", unhedged, flags=re.IGNORECASE)
    return bool(_HEDGE_RE.search(unhedged))


def _described_ids(
    ctx: Any,
    text: str,
    pool: list[Any],
    named_ids: set[str],
    category: str | None,
) -> set[str]:
    """Subscriptions the customer described, so the model may target them.

    Only subscriptions in ``pool`` qualify: an unused phrase may target active
    subscriptions, while a restore targets cancelled ones.
    """

    ids = {sub.id for sub in pool if sub.id in named_ids}
    if _has_any(text, _NEVER):
        ids.update(
            sub.id
            for sub in pool
            if (ctx.now.date() - sub.last_used_at).days >= 60
        )
    if category is not None:
        ids.update(
            sub.id
            for sub in pool
            if subscription_category(sub.merchant) == category
        )
    if _has_any(text, _ALL) and not _is_hedged(text):
        ids.update(sub.id for sub in pool)
    return ids


def _session(ctx: Any) -> Session:
    return Session(
        session_id=ctx.session_id,
        plan_id=ctx.plan_id,
        customer_id=ctx.customer_id,
    )


def _mint(ctx: Any, sub: Any) -> str:
    entry = ctx.vault.mint(
        _session(ctx),
        "SUB",
        sub.id,
        _subscription_tags(sub, ctx.now.date()),
        "bank_context",
    )
    return entry.label


def _step(step_id: str, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": step_id, "tool": tool, "args": args or {}, "after": []}


def _plan(steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": "plan", "steps": steps}


def _clarify(question: str, options: list[str]) -> dict[str, Any]:
    return {"kind": "clarify", "question": question, "options": options}


def _subs_by_id(subs: list[Any]) -> dict[str, Any]:
    return {sub.id: sub for sub in subs}


def _token_targets(
    ctx: Any, tokenized: str, session: Session, by_id: dict[str, Any]
) -> list[Any]:
    targets: list[Any] = []
    seen: set[str] = set()
    for label in _SUB_TOKEN_RE.findall(tokenized):
        try:
            entry = ctx.vault.resolve(label, session)
        except Exception:  # noqa: BLE001 - an unknown label is simply skipped
            continue
        sub = by_id.get(entry.real_ref)
        if sub is None or sub.id in seen:
            continue
        seen.add(sub.id)
        targets.append(sub)
    return targets


def _named_subs(
    ctx: Any,
    text: str,
    tokenized: str,
    session: Session,
    subs: list[Any],
    by_id: dict[str, Any],
) -> list[Any]:
    """Subscriptions named by a SUB token or by merchant/alias text."""

    targets = _token_targets(ctx, tokenized, session, by_id)
    seen = {sub.id for sub in targets}
    for sub in subs:
        if sub.id in seen:
            continue
        if _has_any(text, tuple(_sub_surfaces(sub))):
            targets.append(sub)
            seen.add(sub.id)
    return targets


def _remind_days(text: str) -> int:
    """Days before renewal: first integer 1..30, a Chinese numeral, else 3."""

    for match in _INTEGER_RE.finditer(text):
        days = int(match.group(1))
        if 1 <= days <= 30:
            return days
    match = _CN_DAY_RE.search(text)
    if match is not None:
        return _CN_DAY_VALUES[match.group(1)]
    return 3


def _month_renewal(year: int, month: int, day_of_month: int) -> date:
    last = calendar.monthrange(year, month)[1]
    return date(year, month, min(day_of_month, last))


def _next_month(year: int, month: int) -> tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def _renewal_date(day_of_month: int, today: date, days: int) -> date:
    renewal = _month_renewal(today.year, today.month, day_of_month)
    if renewal < today:
        renewal = _month_renewal(*_next_month(today.year, today.month), day_of_month)
    if renewal - timedelta(days=days) < today:
        renewal = _month_renewal(
            *_next_month(renewal.year, renewal.month), day_of_month
        )
    return renewal


def _restore_targets(
    ctx: Any,
    text: str,
    tokenized: str,
    session: Session,
    cancelled: list[Any],
    by_id: dict[str, Any],
) -> list[Any]:
    targets = [
        sub
        for sub in _token_targets(ctx, tokenized, session, by_id)
        if sub.status == "cancelled"
    ]
    seen = {sub.id for sub in targets}
    lowered = text.casefold()
    for sub in cancelled:
        if sub.id in seen:
            continue
        if sub.merchant.casefold() in lowered:
            targets.append(sub)
            seen.add(sub.id)
    return targets or cancelled


def _detect(text: str) -> bool:
    first = _has_any(text, ("recurring", "charges", "扣费", "自动扣"))
    second = _has_any(text, ("not in my list", "other", "hidden", "其他", "别的", "还有"))
    return first and (second or bool(_DETECT_EXTRA.search(text)))


def _unknown_word(text: str, subs: list[Any], tokenized: str = "") -> str | None:
    merchants = [sub.merchant.casefold() for sub in subs]
    lowered_text = text.casefold()
    for brand in _BRANDS:
        if _is_cjk(brand):
            continue
        if _english_pattern(brand).search(text) is None:
            continue
        if any(brand.casefold() in merchant for merchant in merchants):
            continue
        return brand
    for brand in _BRANDS:
        if not _is_cjk(brand) or brand not in lowered_text:
            continue
        if any(brand in merchant or merchant in brand for merchant in merchants):
            continue
        return brand
    tokenized_folded = tokenized.casefold() if tokenized else lowered_text
    for word in re.findall(r"[A-Za-z]{3,}", text):
        lowered = word.casefold()
        if lowered in _STOPWORDS:
            continue
        if lowered not in tokenized_folded:
            # The tokenizer turned it into a contact/merchant token: not a
            # subscription name, and never echo a real name back.
            continue
        if any(lowered in merchant for merchant in merchants):
            continue
        return word
    return None


def _resolve(ctx: Any) -> dict[str, Any] | None:
    text = ctx.raw_text if isinstance(ctx.raw_text, str) else ""
    tokenized = ctx.tokenized_text if isinstance(ctx.tokenized_text, str) else ""
    if not text.strip():
        return None

    bank = Bank(ctx.conn, ctx.customer_id)
    subs = bank.list_subscriptions()
    by_id = _subs_by_id(subs)
    active = [sub for sub in subs if sub.status == "active"]
    cancelled = [sub for sub in subs if sub.status == "cancelled"]
    session = _session(ctx)
    lang = "zh" if ctx.lang == "zh" else "en"

    cancel = _has_any(text, _CANCEL)
    remind = _has_any(text, _REMIND)
    listq = _has_any(text, _LISTQ)
    restore = _has_any(text, _RESTORE)
    subw = _has_any(text, _SUBW)
    all_words = _has_any(text, _ALL) or bool(_ALL_ZH_RE.search(text))
    # "the one I don't use" is the never-used rule, not a negation.
    unhedged = text
    for phrase in _NEVER:
        unhedged = re.sub(re.escape(phrase), " ", unhedged, flags=re.IGNORECASE)
    hedged = bool(_HEDGE_RE.search(unhedged))
    never = _has_any(text, _NEVER)
    category = _category(text, subs)
    named = _named_subs(ctx, text, tokenized, session, subs, by_id)
    list_args = {"scope": "additional"} if _detect(text) else {}

    # R1: reminder listing.
    if remind and listq and not cancel:
        return _plan([_step("s1", "list_reminders")])

    # R1b: an "automatic charges" listing. Subscriptions runs before bills, so
    # this wins over the bill-analysis period clarify.
    if not cancel and not remind and not restore and _has_any(
        text,
        (
            "自动扣费",
            "自动续费",
            "扣费的",
            "recurring charges",
            "direct debits",
            "what am I paying every month",
            "每个月扣",
        ),
    ):
        return _plan([_step("s1", "list_subscriptions", list_args)])

    # Transfers, orders, cards and the like belong to other rules or the model.
    if _has_any(text, _OTHER_DOMAIN):
        return _plan([_step("s1", "list_subscriptions", list_args)]) if _detect(text) else None
    brand = next((b for b in _BRANDS if _has_any(text, (b,))), None)

    # R-REMIND: one named subscription and a reminder before its renewal.
    if remind and not listq and not cancel and named:
        if len(named) > 1:
            if lang == "zh":
                question = "你想让我提醒哪一个订阅？"
            else:
                question = "Which subscription do you want me to remind you about?"
            return _clarify(question, [_mint(ctx, sub) for sub in named])
        sub = named[0]
        days = _remind_days(text)
        renewal = _renewal_date(sub.day_of_month, ctx.now.date(), days)
        entry = ctx.vault.mint(
            session,
            "DATE",
            renewal,
            OrderedDict([("relative", relative_date_text(renewal, ctx.now.date()))]),
            "bank_context",
        )
        return _plan(
            [
                _step(
                    "s1",
                    "set_reminder",
                    {
                        "when": {"date": entry.label, "offset_days": -days},
                        "note": sub.merchant[:40],
                    },
                )
            ]
        )

    # R2: restore / resubscribe a cancelled subscription.
    if not hedged and restore and (
        subw
        or "<SUB_" in tokenized
        or cancel
        or any(sub.merchant.casefold() in text.casefold() for sub in cancelled)
    ):
        if named:
            targets = [sub for sub in named if sub.status == "cancelled"]
            if not targets:
                if lang == "zh":
                    question = "这个订阅没有被取消，不需要恢复。"
                else:
                    question = "That subscription isn't cancelled, so there is nothing to restore."
                return _clarify(question, [])
        elif brand or _unknown_word(text, subs, tokenized):
            if lang == "zh":
                question = "我没有找到你说的已取消订阅。你是指下面哪一个？"
            else:
                question = "I couldn't find that cancelled subscription. Did you mean one of these?"
            return _clarify(question, [_mint(ctx, sub) for sub in cancelled])
        else:
            targets = cancelled
        if not targets:
            if lang == "zh":
                question = "没有可以恢复的已取消订阅。"
            else:
                question = "There is no cancelled subscription to restore."
            return _clarify(question, [])
        if len(targets) == 1:
            return _plan(
                [_step("s1", "reactivate_subscription", {"sub": _mint(ctx, targets[0])})]
            )
        if lang == "zh":
            question = "你想恢复哪一个已取消的订阅？"
        else:
            question = "Which cancelled subscription do you want to restore?"
        return _clarify(question, [_mint(ctx, sub) for sub in targets])

    # R3: bulk cancel (all subscriptions or one category). Never bulk-cancel
    # when the turn names one subscription and does not say "all".
    if (
        cancel
        and not hedged
        and not (named and all_words)
        and (all_words or category)
        and (subw or category)
        and (all_words or not named)
    ):
        targets = active
        if category is not None:
            targets = [
                sub
                for sub in active
                if subscription_category(sub.merchant) == category
            ]
        if not targets:
            zh_label, en_label = _CATEGORY_LABELS.get(category or "", ("", ""))
            if lang == "zh":
                question = f"你没有正在生效的{zh_label}订阅。"
            else:
                question = f"You have no active {en_label} subscriptions."
            return _clarify(question, [])
        if not all_words and len(targets) > 1:
            # "cancel tencnt video" names a kind, not "all of them": ask.
            zh_label, en_label = _CATEGORY_LABELS.get(category or "", ("", ""))
            if lang == "zh":
                question = f"你有好几个{zh_label}订阅，要取消哪一个？如果都要取消，请说“全部取消”。"
            else:
                question = (
                    f"You have several {en_label} subscriptions. Which one should I cancel? "
                    'To cancel them all, say "cancel all".'
                )
            return _clarify(question, [_mint(ctx, sub) for sub in targets])
        steps = [
            _step(f"s{index}", "cancel_subscription", {"sub": _mint(ctx, sub)})
            for index, sub in enumerate(targets, start=1)
        ]
        return _plan(steps)

    # R4: cancel the one the customer never uses.
    if cancel and never and not hedged and "<SUB_" not in tokenized:
        candidates = [
            sub
            for sub in active
            if (ctx.now.date() - sub.last_used_at).days >= 60
        ]
        # No numbers in rule questions: the reply grounding check rejects
        # amounts and counts that no tool result backs.
        if not candidates:
            if lang == "zh":
                question = "你所有正在生效的订阅最近两个月都用过。"
            else:
                question = "All your active subscriptions were used in the last two months."
            return _clarify(question, [])
        if len(candidates) == 1:
            # The approval card is the confirmation for this yellow-tier action.
            return _plan(
                [
                    _step(
                        "s1",
                        "cancel_subscription",
                        {"sub": _mint(ctx, candidates[0])},
                    )
                ]
            )
        if lang == "zh":
            question = "这些订阅已经两个多月没用了，要取消哪一个？"
        else:
            question = "These subscriptions haven't been used for over two months. Which one should I cancel?"
        return _clarify(question, [_mint(ctx, sub) for sub in candidates])

    # R5: the named subscription is already cancelled.
    if cancel and named and all(sub.status != "active" for sub in named):
        names: list[str] = []
        for sub in named:
            if sub.merchant not in names:
                names.append(sub.merchant)
        joined = "、".join(names) if lang == "zh" else ", ".join(names)
        if lang == "zh":
            question = f"{joined} 已经取消了。"
        else:
            question = f"{joined} is already cancelled."
        return _clarify(question, [])

    # R6: a name that is not one of the customer's subscriptions. Only when the
    # turn is clearly about subscriptions (the word, or a known brand).
    if (
        (cancel or remind)
        and (subw or brand)
        and not named
        and not (all_words or category or never or restore)
    ):
        word = _unknown_word(text, subs, tokenized)
        if word is not None:
            if active:
                if lang == "zh":
                    question = f"你没有名为 {word} 的订阅。你是指下面哪一个？"
                else:
                    question = (
                        f"You don't have a subscription called {word}. "
                        "Did you mean one of these?"
                    )
            elif lang == "zh":
                question = f"你没有名为 {word} 的订阅。你目前没有正在生效的订阅。"
            else:
                question = (
                    f"You don't have a subscription called {word}. "
                    "You have no active subscriptions."
                )
            return _clarify(question, [_mint(ctx, sub) for sub in active])

    # R7: recurring-charge / charged-this detection is a read-only lookup.
    if _detect(text) or _has_any(text, _CHARGED):
        return _plan([_step("s1", "list_subscriptions", list_args)])

    return None


def resolve(ctx: Any) -> dict[str, Any] | None:
    try:
        return _resolve(ctx)
    except Exception:  # noqa: BLE001 - a rule must never break a turn
        return None


_TARGETED_TOOLS = frozenset({"cancel_subscription", "reactivate_subscription"})


def guard_model_plan(ctx: Any, raw: str) -> str | None:
    """Stop the model from choosing a subscription the user never described.

    A model plan that cancels or restores a subscription must target a SUB token
    minted from the user's own words, or a bank-context token for a subscription
    the customer described (never used, a category, or all of them). If it picked
    an unrelated one from the bank context (a guess, e.g. "cancle netfliks" ->
    some other merchant), return a clarify output instead; otherwise ``None``
    keeps the model's plan.
    """
    try:
        output = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(output, dict) or output.get("kind") != "plan":
        return None
    steps = output.get("steps")
    if not isinstance(steps, list):
        return None
    session = _session(ctx)
    text = ctx.raw_text if isinstance(ctx.raw_text, str) else ""
    tokenized = ctx.tokenized_text if isinstance(ctx.tokenized_text, str) else ""
    try:
        subs = Bank(ctx.conn, ctx.customer_id).list_subscriptions()
    except Exception:  # noqa: BLE001 - no readable subs means no described set
        subs = []
    by_id = _subs_by_id(subs)
    named_ids = {
        sub.id for sub in _named_subs(ctx, text, tokenized, session, subs, by_id)
    }
    active = [sub for sub in subs if sub.status == "active"]
    cancelled = [sub for sub in subs if sub.status == "cancelled"]
    category = _category(text, subs)
    cancel_allowed = _described_ids(ctx, text, active, named_ids, category)
    restore_allowed = _described_ids(ctx, text, cancelled, named_ids, category)
    guessed = False
    for step in steps:
        if not isinstance(step, dict) or step.get("tool") not in _TARGETED_TOOLS:
            continue
        tool = step.get("tool")
        label = (step.get("args") or {}).get("sub")
        try:
            entry = ctx.vault.resolve(label, session)
        except Exception:  # noqa: BLE001 - unknown token: the validator rejects it
            continue
        if entry.origin == "user_text":
            continue
        sub = by_id.get(entry.real_ref)
        allowed = restore_allowed if tool == "reactivate_subscription" else cancel_allowed
        if sub is None or sub.id not in allowed:
            guessed = True
            break
    if not guessed:
        return None
    try:
        wants_restore = any(
            isinstance(step, dict) and step.get("tool") == "reactivate_subscription"
            for step in steps
        )
        status = "cancelled" if wants_restore else "active"
        options = [_mint(ctx, sub) for sub in subs if sub.status == status]
    except Exception:  # noqa: BLE001 - fall back to a plain question
        options = []
    if ctx.lang == "zh":
        question = "我不确定你指的是哪个订阅，请从下面选一个。"
    else:
        question = "I'm not sure which subscription you mean. Please pick one below."
    from agent.intents import remember_question

    remember_question(ctx.customer_id, ctx.session_id, question)
    return json.dumps(_clarify(question, options), ensure_ascii=False)
