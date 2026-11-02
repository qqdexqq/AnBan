"""Gold-label generation tests.

Every scenario id is rendered in every language bucket with a suitable
persona.  The generated assistant JSON must parse, every plan must pass the
real DAG validator, and no prompt sent to the planner may contain PII.
"""

from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path

import pytest
import yaml

from security.dag import parse_planner_output, validate_plan
from security.policy import evaluate
from training.gold import (
    GoldError,
    TeacherMismatch,
    build_gold,
    make_sample,
)
from training.templates import render_sample, supported_variations

TODAY = date(2026, 10, 17)
SEED = 42
LANGS = ("zh", "en", "mixed")
SPEC_PATH = Path(__file__).resolve().parents[1] / "data_spec" / "scenarios.yaml"


def _scenario_ids() -> list[str]:
    with SPEC_PATH.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return [scenario["id"] for scenario in data["scenarios"]]


SCENARIO_IDS = _scenario_ids()
_PERSONA_OVERRIDES = {
    "T5_split_bill": "student",
    "T7_insufficient": "student",
    "X1_memo_injection": "micro_business",
    "X4_safe_account_scam": "retiree",
}


def _persona_for(scenario_id: str) -> str:
    return _PERSONA_OVERRIDES.get(scenario_id, "young_pro")


_SAMPLES: dict[tuple[str, str], dict] = {}


def _sample(scenario_id: str, lang: str) -> dict:
    key = (scenario_id, lang)
    if key not in _SAMPLES:
        _SAMPLES[key] = make_sample(
            scenario_id,
            _persona_for(scenario_id),
            lang,
            "explicit",
            SEED,
            TODAY,
            with_runtime=True,
        )
    return _SAMPLES[key]


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
@pytest.mark.parametrize("lang", LANGS)
def test_every_scenario_and_language_builds(scenario_id: str, lang: str) -> None:
    sample = _sample(scenario_id, lang)

    assert sample["system"]
    assert sample["user"]
    assert isinstance(sample["assistant"], str)

    parsed = parse_planner_output(sample["assistant"])
    validated = validate_plan(parsed, sample["meta"]["known_tokens"])

    assert validated.kind == parsed.kind
    if parsed.kind == "plan":
        assert validated.steps
    else:
        assert validated.steps == []


def test_assistant_is_compact_sorted_json() -> None:
    sample = _sample("O1_birthday", "zh")
    assistant = sample["assistant"]

    assert json.loads(assistant) == json.loads(
        json.dumps(
            json.loads(assistant),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    assert "\n" not in assistant


def test_x3_policy_marks_the_fourth_and_fifth_transfers_red() -> None:
    sample = _sample("X3_limit_splitting", "zh")
    parsed = parse_planner_output(sample["assistant"])
    validated = validate_plan(parsed, sample["meta"]["known_tokens"])
    runtime = sample["meta"]["_runtime"]

    decision = evaluate(
        validated,
        bank=runtime["bank"],
        vault=runtime["vault"],
        session=runtime["session"],
        now=TODAY,
    )

    # Expected tiers follow the running total against the ¥1,000 day limit.
    running = runtime["bank"].daily_outgoing_total(TODAY)
    expected = []
    for step in decision.steps:
        running += step.amount_fen
        expected.append("red" if running > 100_000 else "yellow")
    tiers = [step.tier for step in decision.steps]
    assert tiers == expected
    assert tiers[-1] == "red"


@pytest.mark.parametrize("lang", LANGS)
def test_x1_memo_injection_never_transfers(lang: str) -> None:
    sample = _sample("X1_memo_injection", lang)
    parsed = parse_planner_output(sample["assistant"])

    assert parsed.kind == "plan"
    tools = [step.tool for step in parsed.steps]
    assert tools == ["list_transactions"]
    assert "transfer" not in tools
    assert "Memo (untrusted):" in sample["user"]


def test_user_prompts_contain_no_persona_pii() -> None:
    for scenario_id in SCENARIO_IDS:
        for lang in LANGS:
            sample = _sample(scenario_id, lang)
            text = sample["user"]
            state = sample["meta"]["_runtime"]["state"]

            for contact in state.contacts:
                assert contact.name not in text, (scenario_id, lang, contact.name)
                assert contact.phone not in text, (scenario_id, lang, contact.phone)
                assert contact.account_no not in text, (
                    scenario_id,
                    lang,
                    contact.account_no,
                )


def test_unknown_scenario_id_raises() -> None:
    with pytest.raises(GoldError):
        build_gold({"id": "NOPE"}, {}, {})
    with pytest.raises(GoldError):
        make_sample("NOPE", "young_pro", "zh", "explicit", SEED, TODAY)


def _t8_transfers(sample: dict) -> list:
    parsed = parse_planner_output(sample["assistant"])
    assert parsed.kind == "plan"
    return [step for step in parsed.steps if step.tool == "transfer"]


def test_t8_two_payees_two_amounts_use_distinct_labels() -> None:
    sample = make_sample(
        "T8_two_payees_two_amounts",
        "young_pro",
        "zh",
        "explicit",
        SEED,
        TODAY,
        with_runtime=True,
    )
    parsed = parse_planner_output(sample["assistant"])
    assert [step.tool for step in parsed.steps] == [
        "can_cover",
        "transfer",
        "can_cover",
        "transfer",
    ]

    transfers = _t8_transfers(sample)
    assert len(transfers) == 2
    assert transfers[0].args["amount"] != transfers[1].args["amount"]
    assert transfers[0].args["payee"] != transfers[1].args["payee"]

    known = sample["meta"]["known_tokens"]
    for transfer in transfers:
        assert known[transfer.args["amount"]] == "AMT"
        assert known[transfer.args["payee"]] == "PAYEE"


_CN_NUMERALS = "零〇一二三四五六七八九十百千万亿两"


def _has_chinese_numeral(text: str) -> bool:
    return any(char in _CN_NUMERALS for char in text)


def test_t8_mixed_numerals_keep_each_amount_with_its_payee() -> None:
    rendered = None
    sample = None
    for sample_seed in range(200):
        candidate = make_sample(
            "T8_two_payees_two_amounts",
            "young_pro",
            "zh",
            "numbers_in_chinese",
            SEED,
            TODAY,
            with_runtime=True,
            sample_seed=sample_seed,
        )
        state = candidate["meta"]["_runtime"]["state"]
        candidate_rendered = render_sample(
            "T8_two_payees_two_amounts",
            state,
            TODAY,
            random.Random(sample_seed),
            "zh",
            "numbers_in_chinese",
        )
        amount_has_cn = _has_chinese_numeral(candidate_rendered.slots["amount"])
        amount2_has_cn = _has_chinese_numeral(candidate_rendered.slots["amount2"])
        if amount_has_cn and not amount2_has_cn:
            sample = candidate
            rendered = candidate_rendered
            break
    assert sample is not None and rendered is not None

    transfers = _t8_transfers(sample)
    assert len(transfers) == 2
    vault = sample["meta"]["_runtime"]["vault"]
    session = sample["meta"]["_runtime"]["session"]
    entries = {entry.label: entry for entry in vault.entries(session)}

    first, second = transfers
    assert int(entries[first.args["amount"]].real_ref) == rendered.meta["amount_fen"]
    assert int(entries[second.args["amount"]].real_ref) == rendered.meta["amount2_fen"]
    assert rendered.meta["amount_fen"] != rendered.meta["amount2_fen"]
    assert entries[first.args["payee"]].real_ref == rendered.meta["payee_id"]
    assert entries[second.args["payee"]].real_ref == rendered.meta["payee2_id"]


_REVERSED_T8_ZH = "{payee2}那边{amount2}，{payee}那边{amount}，都转一下"


def test_t8_reversed_payee_order_still_maps_roles() -> None:
    sample = None
    rendered = None
    for sample_seed in range(100):
        candidate = make_sample(
            "T8_two_payees_two_amounts",
            "young_pro",
            "zh",
            "explicit",
            SEED,
            TODAY,
            with_runtime=True,
            sample_seed=sample_seed,
        )
        if candidate["meta"]["template"] != _REVERSED_T8_ZH:
            continue
        state = candidate["meta"]["_runtime"]["state"]
        rendered = render_sample(
            "T8_two_payees_two_amounts",
            state,
            TODAY,
            random.Random(sample_seed),
            "zh",
            "explicit",
        )
        sample = candidate
        break
    assert sample is not None and rendered is not None

    roles = sample["meta"]["token_roles"]
    # The reversed template puts payee2/amount2 first, so AMT_1 belongs to payee2.
    assert roles["payee2"] == "<PAYEE_1>"
    assert roles["amount2"] == "<AMT_1>"

    transfers = _t8_transfers(sample)
    assert transfers[0].args["payee"] == roles["payee"]
    assert transfers[0].args["amount"] == roles["amount"]
    assert transfers[1].args["payee"] == roles["payee2"]
    assert transfers[1].args["amount"] == roles["amount2"]

    vault = sample["meta"]["_runtime"]["vault"]
    session = sample["meta"]["_runtime"]["session"]
    entries = {entry.label: entry for entry in vault.entries(session)}
    assert int(entries["<AMT_1>"].real_ref) == rendered.meta["amount2_fen"]
    assert entries["<PAYEE_1>"].real_ref == rendered.meta["payee2_id"]


def test_b6_bill_payment_is_out_of_scope() -> None:
    for lang in LANGS:
        sample = make_sample(
            "B6_bill_pay_unsupported",
            "young_pro",
            lang,
            "explicit",
            SEED,
            TODAY,
        )
        parsed = parse_planner_output(sample["assistant"])
        assert parsed.kind == "refuse", lang
        assert parsed.reason == "out_of_scope", lang


def test_s5_cancel_advice_only_lists_subscriptions() -> None:
    for lang in LANGS:
        sample = make_sample(
            "S5_cancel_advice",
            "young_pro",
            lang,
            "explicit",
            SEED,
            TODAY,
        )
        parsed = parse_planner_output(sample["assistant"])
        assert parsed.kind == "plan", lang
        assert [step.tool for step in parsed.steps] == ["list_subscriptions"], lang


def test_t9_amount_change_without_payee_asks_for_full_request() -> None:
    zh_question = "请把收款人和金额完整说一遍，比如“给妈妈转300”。"
    en_question = (
        "Please say the full request again, with who and how much, "
        'for example "send mom 300".'
    )
    for lang in LANGS:
        sample = make_sample(
            "T9_amount_change_no_payee",
            "young_pro",
            lang,
            "explicit",
            SEED,
            TODAY,
        )
        parsed = parse_planner_output(sample["assistant"])
        assert parsed.kind == "clarify", lang
        assert parsed.options == [], lang
        assert parsed.question == (en_question if lang == "en" else zh_question), lang


def test_x10_bare_confirm_is_out_of_scope() -> None:
    for lang in LANGS:
        sample = make_sample(
            "X10_bare_confirm",
            "young_pro",
            lang,
            "explicit",
            SEED,
            TODAY,
        )
        parsed = parse_planner_output(sample["assistant"])
        assert parsed.kind == "refuse", lang
        assert parsed.reason == "out_of_scope", lang
    assert "multi_intent" not in supported_variations("X10_bare_confirm")


def test_t6_rejects_back_reference_paraphrase() -> None:
    sample = make_sample(
        "T6_missing_amount",
        "young_pro",
        "zh",
        "explicit",
        SEED,
        TODAY,
        with_runtime=True,
    )
    state = sample["meta"]["_runtime"]["state"]
    rendered = render_sample(
        "T6_missing_amount",
        state,
        TODAY,
        random.Random(0),
        "zh",
        "explicit",
    )
    with pytest.raises(TeacherMismatch):
        make_sample(
            "T6_missing_amount",
            "young_pro",
            "zh",
            "explicit",
            SEED,
            TODAY,
            text_override=rendered.text + "，跟上次一样",
        )


# ---------------------------------------------------------------------------
# Held-out polite_long style
# ---------------------------------------------------------------------------

# The wrapper must be applied without touching templates, slots or gold.
_POLITE_WRAPPERS = {
    "zh": ("我想问一下，", "，可以帮我办一下吗？"),
    "en": ("Could you please help me with this: ", " Thanks a lot."),
    "mixed": ("我想问一下，can you help me ", " please?"),
}

# Punctuation may be normalised during tokenization, so probe word fragments.
_POLITE_MARKERS = {
    "zh": ("我想问一下", "可以帮我办一下吗"),
    "en": ("Could you please help me with this", "Thanks a lot"),
    "mixed": ("can you help me", "please?"),
}

_POLITE_SCENARIOS = {
    "T1_transfer_by_name": "plan",
    "T6_missing_amount": "clarify",
    "X4_safe_account_scam": "scam",
    "X8_out_of_scope": "refuse",
}


@pytest.mark.parametrize("scenario_id", sorted(_POLITE_SCENARIOS))
@pytest.mark.parametrize("lang", LANGS)
def test_polite_long_wraps_explicit_render_only(
    scenario_id: str, lang: str
) -> None:
    sample_seed = 0
    sample = make_sample(
        scenario_id,
        _persona_for(scenario_id),
        lang,
        "explicit",
        SEED,
        TODAY,
        with_runtime=True,
        sample_seed=sample_seed,
    )
    state = sample["meta"]["_runtime"]["state"]

    explicit = render_sample(
        scenario_id, state, TODAY, random.Random(sample_seed), lang, "explicit"
    )
    polite = render_sample(
        scenario_id, state, TODAY, random.Random(sample_seed), lang, "polite_long"
    )

    prefix, suffix = _POLITE_WRAPPERS[lang]
    assert polite.text == prefix + explicit.text + suffix
    assert polite.template == explicit.template
    assert polite.slots == explicit.slots
    assert polite.meta == explicit.meta
    assert polite.lang == explicit.lang

    # Held out of the normal training pool even though it can be rendered.
    assert "polite_long" not in supported_variations(scenario_id)


@pytest.mark.parametrize("scenario_id", sorted(_POLITE_SCENARIOS))
@pytest.mark.parametrize("lang", LANGS)
def test_polite_long_keeps_gold_and_tokenized_wording(
    scenario_id: str, lang: str
) -> None:
    persona = _persona_for(scenario_id)
    explicit = make_sample(
        scenario_id, persona, lang, "explicit", SEED, TODAY, with_runtime=True
    )
    polite = make_sample(
        scenario_id, persona, lang, "polite_long", SEED, TODAY, with_runtime=True
    )

    assert polite["assistant"] == explicit["assistant"]
    assert polite["meta"]["variation"] == "polite_long"
    assert polite["meta"]["token_roles"] == explicit["meta"]["token_roles"]

    parsed = parse_planner_output(polite["assistant"])
    assert parsed.kind == _POLITE_SCENARIOS[scenario_id]
    validated = validate_plan(parsed, polite["meta"]["known_tokens"])
    assert validated.kind == parsed.kind

    start_marker, end_marker = _POLITE_MARKERS[lang]
    tokenized = polite["meta"]["tokenized"]
    assert tokenized != explicit["meta"]["tokenized"]
    # The wrapper wording survives tokenization instead of being reset away.
    assert start_marker in tokenized
    assert end_marker in tokenized
