"""Strict spoken-English RMB amount extraction.

Only amounts written with an explicit yuan/RMB/renminbi/元 suffix are returned.
The parser is deliberately not a general language model: it understands the
closed English cardinal grammar, rejects malformed or ambiguous runs, and never
falls back to a shorter positive substring of a bad expression.
"""

from __future__ import annotations

import re

_MAX_FEN = 10_000_000_000

_ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_BIG_SCALES = {"thousand": 1_000, "million": 1_000_000}
_NUMBER_WORDS = (
    set(_ONES) | set(_TEENS) | set(_TENS) | set(_BIG_SCALES) | {"hundred", "and", "point"}
)
_TOKEN_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:"
    + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
    + r")(?![A-Za-z0-9])"
)
_SUFFIX_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:renminbi|yuan|rmb)(?![A-Za-z0-9])|元"
)
_GAP_RE = re.compile(r"[\s\-\u2010\u2011]*")
_HYPHEN_RE = re.compile(r"[-\u2010\u2011]")
_NEGATIVE_RE = re.compile(r"(?i)(?:^|[^a-z])(?:minus|negative)\s*$")
_SIGN_RE = re.compile(r"[-\u2212\uff0d]\s*$")
_FOREIGN_PREFIX_RE = re.compile(
    r"(?i)(?:us\$|\$|€|£|usd|hkd|eur|dollars?|euros?|pounds?)\s*$"
)
_FOREIGN_SUFFIX_RE = re.compile(
    r"(?i)^\s*(?:usd|hkd|eur|us\s+dollars?|hk\s+dollars?|dollars?|euros?|pounds?"
    r"|us\$|\$|€|£)(?![A-Za-z0-9])"
)
_BETWEEN_RE = re.compile(r"(?i)(?:^|[^a-z])between\s*$")
_CONNECTOR_RE = re.compile(r"(?i)(?:^|[^a-z])(to|or|through)\s*$")
_BETWEEN_WORD_RE = re.compile(r"(?i)(?<![a-z])between(?![a-z])")
_CONNECTOR_ONLY_RE = re.compile(r"(?i)(?:to|or|through)")


def _parse_under_100(
    tokens: list[tuple[int, int, str]], pos: int
) -> tuple[int | None, int]:
    if pos >= len(tokens):
        return None, pos
    word = tokens[pos][2]
    if word in _TEENS:
        return _TEENS[word], pos + 1
    if word in _ONES and word != "zero":
        return _ONES[word], pos + 1
    if word in _TENS:
        value = _TENS[word]
        pos += 1
        if pos < len(tokens) and tokens[pos][2] in _ONES and tokens[pos][2] != "zero":
            value += _ONES[tokens[pos][2]]
            pos += 1
        return value, pos
    return None, pos


def _starts_under_100(word: str) -> bool:
    return word in _TEENS or word in _TENS or (word in _ONES and word != "zero")


def _parse_group(
    tokens: list[tuple[int, int, str]], pos: int
) -> tuple[int | None, int]:
    if pos >= len(tokens):
        return None, pos
    word = tokens[pos][2]
    if (
        word in _ONES
        and word != "zero"
        and pos + 1 < len(tokens)
        and tokens[pos + 1][2] == "hundred"
    ):
        value = _ONES[word] * 100
        pos += 2
        if pos < len(tokens) and tokens[pos][2] == "and":
            pos += 1
            tail, pos = _parse_under_100(tokens, pos)
            if tail is None:
                return None, pos
            return value + tail, pos
        if pos < len(tokens) and _starts_under_100(tokens[pos][2]):
            tail, pos = _parse_under_100(tokens, pos)
            if tail is None:
                return None, pos
            return value + tail, pos
        return value, pos
    return _parse_under_100(tokens, pos)


def _parse_cardinal(tokens: list[tuple[int, int, str]]) -> int | None:
    if not tokens:
        return None
    if len(tokens) == 1 and tokens[0][2] == "zero":
        return 0

    total = 0
    pos = 0
    last_scale: int | None = None
    while pos < len(tokens):
        if tokens[pos][2] == "and":
            return None
        group, pos = _parse_group(tokens, pos)
        if group is None:
            return None
        if pos < len(tokens) and tokens[pos][2] in _BIG_SCALES:
            scale = _BIG_SCALES[tokens[pos][2]]
            if last_scale is not None and scale >= last_scale:
                return None
            total += group * scale
            last_scale = scale
            pos += 1
            if pos < len(tokens) and tokens[pos][2] == "and":
                pos += 1
                if pos >= len(tokens):
                    return None
        else:
            total += group
            return total if pos == len(tokens) else None
    return total


def _parse_fen(tokens: list[tuple[int, int, str]]) -> int | None:
    points = [index for index, token in enumerate(tokens) if token[2] == "point"]
    if not points:
        yuan = _parse_cardinal(tokens)
        return None if yuan is None else yuan * 100
    if len(points) != 1:
        return None

    point = points[0]
    yuan = _parse_cardinal(tokens[:point])
    fraction = tokens[point + 1 :]
    if yuan is None or not 1 <= len(fraction) <= 2:
        return None

    cents = 0
    for _, _, word in fraction:
        if word not in _ONES:
            return None
        cents = cents * 10 + _ONES[word]
    if len(fraction) == 1:
        cents *= 10
    return yuan * 100 + cents


def _tokens_before(text: str, end: int) -> list[tuple[int, int, str]]:
    """The maximal contiguous run of number words ending at ``end``."""

    tokens: list[tuple[int, int, str]] = []
    cursor = end
    right_word: str | None = None
    for match in reversed(list(_TOKEN_RE.finditer(text, 0, end))):
        gap = text[match.end() : cursor]
        if _GAP_RE.fullmatch(gap) is None:
            break
        word = match.group(0).lower()
        if _HYPHEN_RE.search(gap) is not None:
            if not (
                word in _TENS
                and right_word in _ONES
                and right_word != "zero"
            ):
                return []
        tokens.append((match.start(), match.end(), word))
        cursor = match.start()
        right_word = word
    tokens.reverse()
    return tokens


def _has_negative_prefix(text: str, start: int) -> bool:
    prefix = text[:start]
    return _NEGATIVE_RE.search(prefix) is not None or _SIGN_RE.search(prefix) is not None


def _has_foreign_context(text: str, start: int, end: int) -> bool:
    return (
        _FOREIGN_PREFIX_RE.search(text[:start]) is not None
        or _FOREIGN_SUFFIX_RE.match(text[end:]) is not None
    )


def _has_number_word(text: str) -> bool:
    return re.search(r"\d", text) is not None or any(
        match.group(0).lower() not in {"and", "point"}
        for match in _TOKEN_RE.finditer(text)
    )


def _has_prior_number(prefix: str) -> bool:
    return _has_number_word(re.split(r"[.!?;:\n]", prefix)[-1])


def _inside_between_phrase(prefix: str) -> bool:
    matches = list(_BETWEEN_WORD_RE.finditer(prefix))
    if not matches:
        return False
    tail = re.split(r"[.!?;:,\n]", prefix[matches[-1].end() :])[-1]
    return _has_number_word(tail)


def _ambiguous_prefix(text: str, start: int) -> bool:
    prefix = text[:start]
    if _BETWEEN_RE.search(prefix):
        return True
    connector = _CONNECTOR_RE.search(prefix)
    if connector is not None and _has_prior_number(prefix[: connector.start(1)]):
        return True
    return _inside_between_phrase(prefix)


def _connector_only_between(text: str, left: int, right: int) -> bool:
    middle = text[left:right].strip(" \t\r\n,，")
    return _CONNECTOR_ONLY_RE.fullmatch(middle) is not None


def find_english_rmb(text: str) -> list[tuple[int, int, int]]:
    """Return ``(start, end, fen)`` for unambiguous spoken English RMB amounts.

    ``end`` includes the currency suffix; the input text is never changed.  A
    malformed, negative, out-of-range or range/choice expression contributes no
    amount at all, so no positive substring of it can be minted.
    """

    if not isinstance(text, str):
        raise TypeError("text must be str")

    candidates: list[tuple[int, int, int]] = []
    for suffix in _SUFFIX_RE.finditer(text):
        tokens = _tokens_before(text, suffix.start())
        if not tokens:
            continue
        if tokens[0][2] == "and":
            previous = text[:tokens[0][0]].rstrip()
            if any(match.end() == len(previous) for match in _SUFFIX_RE.finditer(previous)):
                tokens = tokens[1:]
        fen = _parse_fen(tokens)
        if fen is None or fen <= 0 or fen > _MAX_FEN:
            continue
        start = tokens[0][0]
        end = suffix.end()
        if _has_negative_prefix(text, start):
            continue
        if _has_foreign_context(text, start, end):
            continue
        candidates.append((start, end, fen))

    candidates.sort(key=lambda item: item[0])
    ambiguous: set[int] = set()
    for index in range(len(candidates) - 1):
        if _connector_only_between(
            text, candidates[index][1], candidates[index + 1][0]
        ):
            ambiguous.update((index, index + 1))

    found: list[tuple[int, int, int]] = []
    for index, (start, end, fen) in enumerate(candidates):
        if index in ambiguous:
            continue
        if _ambiguous_prefix(text, start):
            continue
        found.append((start, end, fen))
    return found
