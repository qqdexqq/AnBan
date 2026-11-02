"""Vetted banking-knowledge answers.

The model only chooses a topic; every answer is copied verbatim from
articles.json, which carries a source and a checked date.  If no article fits,
the caller gets nothing and must fall back to a refusal, not invented text.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_ARTICLES_PATH = Path(__file__).with_name("articles.json")

# [0-9][0-9,.]* with any trailing punctuation removed and commas dropped.
_NUMBER = re.compile(r"[0-9][0-9,.]*")

TOPICS: tuple[str, ...] = (
    "delayed_arrival",
    "recall",
    "deposit_insurance",
    "fx_quota",
    "complaints",
    "anti_fraud",
    "large_deposit_cd",
    "suitability",
    "personal_pension",
    "account_tiers",
    "installment_rate",
    "account_restriction",
)


@lru_cache(maxsize=1)
def load_articles() -> list[dict[str, Any]]:
    """Load the article list that sits next to this module."""

    with _ARTICLES_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def article(topic: str) -> dict[str, Any] | None:
    """Return the article for a topic id, or None when it is unknown."""

    for item in load_articles():
        if item["id"] == topic:
            return item
    return None


def search(text: str) -> dict[str, Any] | None:
    """Return the article with the most keyword hits, or None for no hit."""

    haystack = text.casefold()
    best: dict[str, Any] | None = None
    best_score = 0
    for item in load_articles():
        keywords = item["keywords_zh"] + item["keywords_en"]
        score = sum(1 for kw in keywords if kw.casefold() in haystack)
        if score > best_score:
            best = item
            best_score = score
    return best


def answer(topic: str, lang: str) -> str:
    """Return the vetted text for a topic plus its source line."""

    item = article(topic)
    if item is None:
        raise ValueError(f"unknown topic: {topic}")
    if lang == "en":
        return f"{item['en']} (Source: {item['source']})"
    return f"{item['zh']}（来源：{item['source']}）"


def _numbers(text: str) -> list[str]:
    return [
        match.group(0).rstrip(".,").replace(",", "")
        for match in _NUMBER.finditer(text)
    ]


def numbers_supported(reply: str, topic: str) -> bool:
    """True when every number in reply occurs in the article text or source."""

    item = article(topic)
    if item is None:
        raise ValueError(f"unknown topic: {topic}")
    haystack = (item["zh"] + item["en"] + item["source"]).replace(",", "")
    return all(number in haystack for number in _numbers(reply))
