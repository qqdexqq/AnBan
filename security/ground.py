"""Grounding checks that every number, ID and name in a reply came from tool output.

The responder is allowed to phrase a tool result, but it must never invent a
money figure, a percentage, a masked account or a date.  :func:`collect_facts`
harvests those facts from the raw tool results, :func:`extract_claims` finds the
corresponding figures in the draft reply, :func:`check` compares them, and
:func:`grounded_reply` replaces an ungrounded draft with a safe template.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from bank.money import yuan

__all__ = [
    "Claim",
    "Facts",
    "GroundResult",
    "check",
    "collect_facts",
    "extract_claims",
    "grounded_reply",
]

_PERCENT_TOLERANCE = Decimal("0.01")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")

#: Field names (or suffixes) that carry integer fen amounts.
_MONEY_NAMES = frozenset({"amount", "balance", "total", "value", "price"})
_MONEY_SUFFIXES = ("_fen", "_amount", "_balance", "_total", "_value", "_price")


# --------------------------------------------------------------------- models


@dataclass(frozen=True, slots=True)
class Claim:
    """One figure found in a reply.

    ``kind`` is ``"money"``, ``"percent"``, ``"account"`` or ``"date"``.
    ``value`` is integer fen for money, ``Decimal`` percent for percentages,
    a four-digit string for account tails, and a :class:`datetime.date` (or a
    ``(month, day)`` tuple for year-less Chinese dates) for dates.
    """

    kind: str
    text: str
    value: Any
    start: int = 0
    end: int = 0

    @property
    def raw(self) -> str:
        """Alias for the matched text."""
        return self.text


@dataclass
class Facts:
    """Everything a reply is allowed to say in this turn."""

    amounts_fen: set[int] = field(default_factory=set)
    percents: set[Decimal] = field(default_factory=set)
    tails: set[str] = field(default_factory=set)
    dates: set[date] = field(default_factory=set)
    derived: list[tuple[int, str]] = field(default_factory=list)

    @property
    def amounts(self) -> set[int]:
        """Alias for :attr:`amounts_fen`."""
        return self.amounts_fen

    def add_amount(self, fen: int) -> None:
        """Record an integer fen fact."""
        if isinstance(fen, bool) or not isinstance(fen, int):
            raise TypeError("fen must be an int")
        self.amounts_fen.add(fen)
        if fen < 0:
            # Replies normally describe a debit as a positive figure ("spent
            # ¥12.00"), so the magnitude of a signed transaction is a fact too.
            self.amounts_fen.add(-fen)

    def add_percent(self, percent: Decimal | int | float | str) -> None:
        """Record a percentage, rounded to two decimal places."""
        value = Decimal(str(percent)).quantize(_ONE / _HUNDRED, rounding=ROUND_HALF_UP)
        self.percents.add(value)

    def add_tail(self, tail: str) -> None:
        """Record a masked account tail (last four digits)."""
        digits = "".join(character for character in str(tail) if character.isdigit())
        if digits:
            self.tails.add(digits[-4:].rjust(4, "0"))

    def add_date(self, value: date | datetime) -> None:
        """Record a calendar date."""
        self.dates.add(value.date() if isinstance(value, datetime) else value)

    def add_derived(self, fen: int, reason: str) -> None:
        """Allow an explicitly computed sum produced by the reply template.

        Only sums the caller has deliberately computed (for example "total
        spending = dining + transport") may be added this way; every other
        figure must come from a tool result.
        """
        if isinstance(fen, bool) or not isinstance(fen, int):
            raise TypeError("fen must be an int")
        self.add_amount(fen)
        self.derived.append((fen, str(reason)))


@dataclass(slots=True)
class GroundResult:
    """Outcome of :func:`check`."""

    ok: bool
    unknown: list[Claim]

    @property
    def rejected(self) -> list[Claim]:
        """Alias used when writing the audit ``GROUND`` entry."""
        return self.unknown


# ----------------------------------------------------------------- collection


def _is_money_field(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return lowered in _MONEY_NAMES or lowered.endswith(_MONEY_SUFFIXES)


def _is_tail_field(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return lowered == "account_no" or lowered == "last4" or lowered.endswith(
        ("_account_no", "_last4")
    )


def _is_date_field(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return (
        lowered in {"date", "birthday"}
        or lowered.endswith("_date")
        or lowered.endswith("_birthday")
    )


def _percent_from_number(value: int | float | Decimal) -> Decimal:
    number = Decimal(str(value))
    # Fractional rates (0.035) and whole percentages (3.5) are both common;
    # anything in [-1, 1] is treated as a fraction.
    if abs(number) <= _ONE:
        number *= _HUNDRED
    return number.quantize(_ONE / _HUNDRED, rounding=ROUND_HALF_UP)


def _collect_all_ints(value: object, facts: Facts) -> None:
    """Add every integer below ``value`` as a fen amount."""
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        facts.add_amount(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            _collect_all_ints(item, facts)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _collect_all_ints(item, facts)


def _is_percent_field(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return (
        lowered.endswith("_bp")
        or "rate" in lowered
        or "pct" in lowered
        or "percent" in lowered
    )


def _percent_field_value(value: int | float | Decimal, name: str) -> Decimal:
    if name.lower().endswith("_bp"):
        return Decimal(str(value)) / _HUNDRED
    return _percent_from_number(value)


def _collect(value: object, facts: Facts, field_name: str | None = None) -> None:
    if value is None or isinstance(value, (str, bytes, bool)):
        if isinstance(value, str):
            if _is_tail_field(field_name):
                facts.add_tail(value)
            if _is_date_field(field_name):
                try:
                    facts.add_date(date.fromisoformat(value[:10]))
                except ValueError:
                    pass
        return

    if isinstance(value, datetime):
        facts.add_date(value)
        return
    if isinstance(value, date):
        facts.add_date(value)
        return

    if isinstance(value, int):
        if _is_percent_field(field_name):
            facts.add_percent(_percent_field_value(value, field_name or ""))
        elif field_name is None or _is_money_field(field_name):
            facts.add_amount(value)
        return

    if isinstance(value, (float, Decimal)):
        if _is_percent_field(field_name):
            facts.add_percent(_percent_field_value(value, field_name or ""))
        elif field_name and _is_money_field(field_name):
            # Float/Decimal money fields are expressed in yuan.
            facts.add_amount(yuan(Decimal(str(value))))
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            name = key if isinstance(key, str) else str(key)
            if name == "by_category":
                # spending_report.by_category is a mapping of category -> fen;
                # its keys are category names, not money field names.
                _collect_all_ints(item, facts)
            _collect(item, facts, name)
        return

    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _collect(getattr(value, field.name), facts, field.name)
        return

    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _collect(item, facts, field_name)
        return


def collect_facts(tool_results: list[Any]) -> Facts:
    """Harvest grounded figures from this turn's tool results."""
    facts = Facts()
    for result in tool_results:
        _collect(result, facts)
    return facts


# ------------------------------------------------------------------ extraction

_CN_DIGITS = {
    "零": 0,
    "〇": 0,
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
_CN_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_LARGE_UNITS = {"万": 10000, "亿": 100000000}
_CN_NUMBER_CHARS = "零〇一二两三四五六七八九十百千万亿"


def _parse_cn_integer(text: str) -> int:
    total = 0
    section = 0
    number: int | None = None
    for character in text:
        if character in _CN_DIGITS:
            number = _CN_DIGITS[character]
        elif character in _CN_SMALL_UNITS:
            section += (1 if number is None else number) * _CN_SMALL_UNITS[character]
            number = None
        elif character in _CN_LARGE_UNITS:
            section += 0 if number is None else number
            if section == 0:
                section = 1
            total += section * _CN_LARGE_UNITS[character]
            section = 0
            number = None
        else:
            raise ValueError(f"not a Chinese numeral: {text!r}")
    return total + section + (0 if number is None else number)


def _parse_cn_number(text: str) -> Decimal:
    if "点" not in text:
        return Decimal(_parse_cn_integer(text))
    whole, fraction = text.split("点", 1)
    value = Decimal(_parse_cn_integer(whole)) if whole else Decimal(0)
    scale = Decimal("0.1")
    for character in fraction:
        if character not in _CN_DIGITS:
            raise ValueError(f"not a Chinese numeral: {text!r}")
        value += Decimal(_CN_DIGITS[character]) * scale
        scale /= 10
    return value


_MONEY_UNIT_MULTIPLIERS = {
    "万": 10000,
    "万元": 10000,
    "w": 10000,
    "k": 1000,
}


def _amount_to_fen(amount: Decimal, unit: str | None) -> int:
    multiplier = _HUNDRED * _MONEY_UNIT_MULTIPLIERS.get((unit or "").lower(), 1)
    return int((amount * multiplier).quantize(_ONE, rounding=ROUND_HALF_UP))


def _decimal(text: str) -> Decimal:
    return Decimal(text.replace(",", ""))


_CURRENCY_RE = re.compile(
    r"(?P<sign>[+-]?)\s*(?P<currency>¥|￥|RMB|CNY|yuan)\s*"
    r"(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
    re.IGNORECASE,
)
_FOREIGN_CURRENCY_RE = re.compile(
    r"(?P<sign>[+-]?)\s*(?P<currency>US\$|USD|\$)\s*"
    r"(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
    re.IGNORECASE,
)
_SUFFIX_RE = re.compile(
    r"(?P<sign>[+-]?)\s*(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*"
    r"(?P<unit>万元|万|元|块|yuan|RMB|CNY|k|w)(?![A-Za-z])",
    re.IGNORECASE,
)
_FOREIGN_SUFFIX_RE = re.compile(
    r"(?P<sign>[+-]?)\s*(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*"
    r"(?P<currency>USD|US\$|\$)(?![A-Za-z])",
    re.IGNORECASE,
)
_CN_MONEY_RE = re.compile(
    rf"(?<![\d.])(?P<sign>[+-]?)\s*(?P<num>[{_CN_NUMBER_CHARS}]+"
    r"(?:点[零〇一二两三四五六七八九]+)?)\s*(?P<unit>万元|万|元|块)"
)
_PERCENT_RE = re.compile(r"(?P<num>[+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*[%％]")
_CN_PERCENT_RE = re.compile(
    rf"百分之\s*(?:(?P<num>[+-]?\d+(?:\.\d+)?)|(?P<cn>[{_CN_NUMBER_CHARS}]+"
    r"(?:点[零〇一二两三四五六七八九]+)?))"
)
_ACCOUNT_RE = re.compile(
    r"(?:[·•∙⋅]{2,}|\.{2,}|[*＊×✕xX#]{2,})\s*(?P<tail>\d{4})(?!\d)"
)
_TAIL_RE = re.compile(
    r"(?:尾号|尾號|末四位|last\s*4(?:\s*digits)?|ending(?:\s+in)?)\s*[:：]?\s*"
    r"(?:为|是)?\s*(?:(?:[·•∙⋅]{2,}|\.{2,}|[*＊×✕xX#]{2,})\s*)?"
    r"(?P<tail>\d{4})(?!\d)",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(
    r"\b(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})\b"
)
_CN_DATE_RE = re.compile(
    r"(?P<year>\d{4})\s*年\s*(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})\s*[日号]?"
)
_CN_SHORT_DATE_RE = re.compile(
    r"(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})\s*[日号]"
)
_SLASH_DATE_RE = re.compile(
    r"(?<!\d)(?:(?P<year>\d{4})\s*[/／]\s*)?"
    r"(?P<month>\d{1,2})\s*[/／]\s*(?P<day>\d{1,2})(?!\d)"
)

_CN_RELATIVE_RE = re.compile(r"今天|明天|后天")
_CN_NEXT_WEEK_RE = re.compile(r"下周(?P<weekday>[一二三四五六日天])")
_EN_TOMORROW_RE = re.compile(r"(?<![A-Za-z])tomorrow(?![A-Za-z])", re.IGNORECASE)
_EN_NEXT_RE = re.compile(
    r"(?<![A-Za-z])next\s+"
    r"(?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)
_EN_IN_DAYS_RE = re.compile(
    r"(?<![A-Za-z])in\s+(?P<days>\d+)\s+days?(?![A-Za-z])",
    re.IGNORECASE,
)

_CN_WEEKDAYS = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
}
_EN_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

#: A numeric run that still needs to be classified as money or a non-figure.
_NUMBER_RUN_RE = re.compile(
    r"(?<![\d.,])(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?!\d)"
)
_TIME_RE = re.compile(r"(?<!\d)\d{1,3}:\d{2}(?::\d{2})?(?!\d)")
_ORDINAL_LINE_RE = re.compile(r"(?m)^[ \t]*\d{1,3}[.)](?=\s|$)")
_DURATION_AFTER_RE = re.compile(
    r"\s*(?:小时|分钟|天|hours?|hrs?|h|minutes?|mins?|min|days?)(?![A-Za-z])",
    re.IGNORECASE,
)
_COUNT_AFTER_RE = re.compile(
    r"\s*(?:笔|次|个|项|transactions?|times?|items?)(?![A-Za-z])",
    re.IGNORECASE,
)
_YEAR_AFTER_RE = re.compile(r"\s*年")
_PERCENT_AFTER_RE = re.compile(r"\s*[%％]")
#: The public anti-fraud hotline is a fixed, non-financial number.
_HOTLINE_RE = re.compile(
    r"(?<!\d)96110(?!\d)(?=\s*(?:反诈|anti-fraud|hotline))",
    re.IGNORECASE,
)


def _date_claim(match: re.Match[str], *, short: bool) -> Claim | None:
    try:
        if short:
            value = (int(match.group("month")), int(match.group("day")))
            date(2000, value[0], value[1])  # validate month/day
        else:
            value = date(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            )
    except ValueError:
        return None
    return Claim("date", match.group(0), value, match.start(), match.end())


def _slash_date_claim(match: re.Match[str]) -> Claim | None:
    try:
        if match.group("year") is not None:
            value: date | tuple[int, int] = date(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            )
        else:
            value = (int(match.group("month")), int(match.group("day")))
            date(2000, value[0], value[1])  # validate month/day
    except ValueError:
        return None
    return Claim("date", match.group(0), value, match.start(), match.end())


def _next_week_date(today: date, weekday: int) -> date:
    days_to_monday = (7 - today.weekday()) % 7 or 7
    return today + timedelta(days=days_to_monday + weekday)


def _relative_date_claims(text: str, today: date | None) -> list[Claim]:
    claims: list[Claim] = []

    def add(match: re.Match[str], value: date | None) -> None:
        claims.append(
            Claim("date", match.group(0), value, match.start(), match.end())
        )

    for match in _CN_RELATIVE_RE.finditer(text):
        phrase = match.group(0)
        offset = {"今天": 0, "明天": 1, "后天": 2}[phrase]
        add(match, None if today is None else today + timedelta(days=offset))

    for match in _CN_NEXT_WEEK_RE.finditer(text):
        weekday = _CN_WEEKDAYS[match.group("weekday")]
        add(match, None if today is None else _next_week_date(today, weekday))

    for match in _EN_TOMORROW_RE.finditer(text):
        add(match, None if today is None else today + timedelta(days=1))

    for match in _EN_NEXT_RE.finditer(text):
        weekday = _EN_WEEKDAYS[match.group("weekday").lower()]
        add(match, None if today is None else _next_week_date(today, weekday))

    for match in _EN_IN_DAYS_RE.finditer(text):
        days = int(match.group("days"))
        add(match, None if today is None else today + timedelta(days=days))

    return claims


def _dedupe(candidates: list[Claim]) -> list[Claim]:
    """Keep the first (preferably longest) claim for every overlapping span."""
    candidates.sort(key=lambda claim: (claim.start, -(claim.end - claim.start)))
    selected: list[Claim] = []
    for claim in candidates:
        if any(
            claim.start < existing.end and existing.start < claim.end
            for existing in selected
        ):
            continue
        selected.append(claim)
    return selected


def extract_claims(
    text: str,
    today: date | datetime | None = None,
) -> list[Claim]:
    """Find money, percentage, masked-account and date claims in a reply.

    ``today`` is only needed to resolve relative dates such as ``下周三`` or
    ``tomorrow``; without it those claims stay unresolved and will be rejected.
    """
    candidates: list[Claim] = []

    def add(kind: str, match: re.Match[str], value: Any) -> None:
        candidates.append(
            Claim(kind, match.group(0), value, match.start(), match.end())
        )

    for match in _CURRENCY_RE.finditer(text):
        amount = _decimal(match.group("num"))
        if match.group("sign") == "-":
            amount = -amount
        add("money", match, _amount_to_fen(amount, None))

    for match in _FOREIGN_CURRENCY_RE.finditer(text):
        # Foreign-currency figures are always ungrounded: the tool results are
        # in fen and no exchange rate is available to verify them.
        add("money", match, None)

    for match in _SUFFIX_RE.finditer(text):
        amount = _decimal(match.group("num"))
        if match.group("sign") == "-":
            amount = -amount
        add("money", match, _amount_to_fen(amount, match.group("unit")))

    for match in _FOREIGN_SUFFIX_RE.finditer(text):
        add("money", match, None)

    for match in _CN_MONEY_RE.finditer(text):
        amount = _parse_cn_number(match.group("num"))
        if match.group("sign") == "-":
            amount = -amount
        add("money", match, _amount_to_fen(amount, match.group("unit")))

    for match in _PERCENT_RE.finditer(text):
        add("percent", match, _decimal(match.group("num")))

    for match in _CN_PERCENT_RE.finditer(text):
        if match.group("cn") is not None:
            value = _parse_cn_number(match.group("cn"))
        else:
            value = _decimal(match.group("num"))
        add("percent", match, value)

    for pattern in (_ACCOUNT_RE, _TAIL_RE):
        for match in pattern.finditer(text):
            add("account", match, match.group("tail"))

    for match in _ISO_DATE_RE.finditer(text):
        claim = _date_claim(match, short=False)
        if claim is not None:
            candidates.append(claim)

    for match in _CN_DATE_RE.finditer(text):
        claim = _date_claim(match, short=False)
        if claim is not None:
            candidates.append(claim)

    for match in _CN_SHORT_DATE_RE.finditer(text):
        claim = _date_claim(match, short=True)
        if claim is not None:
            candidates.append(claim)

    for match in _SLASH_DATE_RE.finditer(text):
        claim = _slash_date_claim(match)
        if claim is not None:
            candidates.append(claim)

    if isinstance(today, datetime):
        today = today.date()
    candidates.extend(_relative_date_claims(text, today))

    # Everything already recognised is off limits to the bare-number fallback:
    # currency figures, percentages, account tails, dates and times.
    protected = [(claim.start, claim.end) for claim in candidates]
    protected.extend(match.span() for match in _TIME_RE.finditer(text))
    protected.extend(match.span() for match in _HOTLINE_RE.finditer(text))
    ordinals = [match.span() for match in _ORDINAL_LINE_RE.finditer(text)]

    def overlaps(start: int, end: int) -> bool:
        return any(
            start < span_end and span_start < end
            for span_start, span_end in protected
        )

    for match in _NUMBER_RUN_RE.finditer(text):
        start, end = match.span()
        if overlaps(start, end):
            continue
        if any(
            start >= ordinal_start and end <= ordinal_end
            for ordinal_start, ordinal_end in ordinals
        ):
            continue

        raw = match.group("num")
        digits = "".join(character for character in raw if character.isdigit())
        has_thousands = "," in raw
        has_decimal = "." in raw
        if len(digits) < 3 and not has_thousands and not has_decimal:
            continue

        tail = text[end:]
        if (
            len(digits) == 4
            and raw.isdigit()
            and 2000 <= int(raw) <= 2099
            and _YEAR_AFTER_RE.match(tail)
        ):
            continue
        if _DURATION_AFTER_RE.match(tail):
            continue
        if _COUNT_AFTER_RE.match(tail):
            continue
        if _PERCENT_AFTER_RE.match(tail):
            continue

        candidates.append(
            Claim(
                "money",
                match.group(0),
                _amount_to_fen(_decimal(raw), None),
                start,
                end,
            )
        )

    return _dedupe(candidates)


# ------------------------------------------------------------------- checking


def _matches(claim: Claim, facts: Facts) -> bool:
    if claim.kind == "money":
        return claim.value in facts.amounts_fen
    if claim.kind == "percent":
        return any(
            abs(claim.value - Decimal(str(fact))) <= _PERCENT_TOLERANCE
            for fact in facts.percents
        )
    if claim.kind == "account":
        return claim.value in facts.tails
    if claim.kind == "date":
        if claim.value is None:
            return False
        if isinstance(claim.value, tuple):
            month, day = claim.value
            return any((fact.month, fact.day) == (month, day) for fact in facts.dates)
        return claim.value in facts.dates
    return False


def check(
    text: str,
    facts: Facts,
    today: date | datetime | None = None,
) -> GroundResult:
    """Return which claims in ``text`` are not supported by ``facts``.

    ``today`` is used to resolve relative dates; without it a relative date
    claim is unknown and therefore rejected.
    """
    unknown = [
        claim
        for claim in extract_claims(text, today)
        if not _matches(claim, facts)
    ]
    return GroundResult(not unknown, unknown)


def grounded_reply(
    draft: str,
    facts: Facts,
    fallback: Callable[[], str],
) -> tuple[str, GroundResult]:
    """Return the draft only when every claim is grounded, else the fallback."""
    result = check(draft, facts)
    if result.ok:
        return draft, result
    return fallback(), result
