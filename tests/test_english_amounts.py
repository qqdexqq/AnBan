"""Strict spoken-English RMB amount extraction tests."""

from __future__ import annotations

import pytest

from security.english_amounts import find_english_rmb


def _one(text: str) -> tuple[str, int]:
    found = find_english_rmb(text)
    assert len(found) == 1
    start, end, fen = found[0]
    return text[start:end], fen


def test_original_voice_phrase_keeps_both_amounts_and_positions() -> None:
    text = (
        "around four hundred fifty yuan? No, no, no, "
        "five hundred fifty yuan to Lena"
    )

    found = find_english_rmb(text)

    assert [(text[start:end], fen) for start, end, fen in found] == [
        ("four hundred fifty yuan", 45_000),
        ("five hundred fifty yuan", 55_000),
    ]
    assert found == [
        (text.index("four hundred fifty yuan"), text.index("four hundred fifty yuan") + 23, 45_000),
        (text.index("five hundred fifty yuan"), text.index("five hundred fifty yuan") + 23, 55_000),
    ]


@pytest.mark.parametrize(
    ("text", "fen"),
    [
        ("one yuan", 100),
        ("nineteen yuan", 1_900),
        ("twenty yuan", 2_000),
        ("twenty-one yuan", 2_100),
        ("fifty-five yuan", 5_500),
        ("one hundred five yuan", 10_500),
        ("one hundred and five yuan", 10_500),
        ("four hundred fifty yuan", 45_000),
        ("one thousand and five yuan", 100_500),
        ("one thousand two hundred thirty-four yuan", 123_400),
        ("one hundred thousand yuan", 10_000_000),
        ("one million two hundred thousand yuan", 120_000_000),
        (
            "two million three hundred thousand four hundred fifty-six yuan",
            230_045_600,
        ),
        ("five hundred RMB", 50_000),
        ("one thousand renminbi", 100_000),
        ("five hundred 元", 50_000),
    ],
)
def test_standard_cardinals_and_suffixes(text: str, fen: int) -> None:
    assert _one(text) == (text, fen)


@pytest.mark.parametrize(
    ("text", "fen"),
    [
        ("five point five yuan", 550),
        ("five point five zero yuan", 550),
        ("five point zero five yuan", 505),
        ("zero point zero five yuan", 5),
        ("one hundred point zero one yuan", 10_001),
    ],
)
def test_decimals_are_exact_fen(text: str, fen: int) -> None:
    assert _one(text) == (text, fen)


@pytest.mark.parametrize(
    "text",
    [
        "five months",
        "two people",
        "send five hundred fifty to Lena",
        "five hundred dollars",
        "five hundred euros",
        "five hundred USD",
        "$five hundred yuan",
        "USD five hundred yuan",
        "five hundred yuan USD",
        "five hundred yuan US dollars",
        "five hundred dollars yuan",
    ],
)
def test_counts_and_foreign_currency_are_not_rmb(text: str) -> None:
    assert find_english_rmb(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "minus five hundred yuan",
        "minusfive hundred yuan",
        "minusfivehundredyuan",
        "negative five hundred yuan",
        "negativefive hundred yuan",
        "negativefivehundredyuan",
        "-five hundred yuan",
        "- five hundred yuan",
        "-fivehundredyuan",
        "− five hundred yuan",
        "－five hundred yuan",
        "minus fifty-five yuan",
    ],
)
def test_negative_amounts_mint_nothing(text: str) -> None:
    assert find_english_rmb(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "four five hundred yuan",
        "four-hundred-five yuan",
        "one-hundred yuan",
        "hundred yuan",
        "one hundred hundred yuan",
        "one thousand thousand yuan",
        "one million one million yuan",
        "ten hundred yuan",
        "one hundred and yuan",
        "one hundred and and five yuan",
        "and five hundred yuan",
        "zero yuan",
        "one million zero yuan",
        "five point five zero zero yuan",
        "five point five five five yuan",
        "five point twenty yuan",
        "five point yuan",
        "point five yuan",
        "one point five hundred yuan",
        "four hundred and five hundred yuan",
        "five hundred five six yuan",
    ],
)
def test_malformed_or_overprecise_runs_mint_no_positive_substring(text: str) -> None:
    assert find_english_rmb(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "between four hundred and five hundred yuan",
        "between 400 and five hundred yuan",
        "from four hundred to five hundred yuan",
        "four hundred to five hundred yuan",
        "four hundred, to five hundred yuan",
        "four hundred or five hundred yuan",
        "four hundred through five hundred yuan",
        "between four hundred yuan and five hundred yuan",
        "four hundred yuan to five hundred yuan",
        "four hundred yuan or five hundred yuan",
    ],
)
def test_ranges_and_choices_are_declined(text: str) -> None:
    assert find_english_rmb(text) == []


def test_target_word_to_is_not_a_range() -> None:
    assert _one("send five hundred yuan to Lena") == ("five hundred yuan", 50_000)


def test_cap_is_inclusive_and_out_of_range_mints_nothing() -> None:
    assert _one("one hundred million yuan") == ("one hundred million yuan", 10_000_000_000)
    assert _one("one hundred million point zero zero yuan") == (
        "one hundred million point zero zero yuan",
        10_000_000_000,
    )

    for text in [
        "one hundred million one yuan",
        "one hundred million point zero one yuan",
        "nine hundred ninety-nine million yuan",
        "one thousand million yuan",
        "one billion yuan",
    ]:
        assert find_english_rmb(text) == []


def test_multiple_amounts_keep_text_order_and_exact_spans() -> None:
    text = "pay five hundred yuan and six hundred yuan"

    found = find_english_rmb(text)

    assert [(text[start:end], fen) for start, end, fen in found] == [
        ("five hundred yuan", 50_000),
        ("six hundred yuan", 60_000),
    ]
    assert found[0][0] < found[1][0]
