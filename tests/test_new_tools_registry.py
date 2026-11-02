"""Guard rails for the untrained tool batch and the prompt split."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from agent.prompts import (
    FALLBACK_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    R3_RENDERED_PROMPT,
    R3_SYSTEM_PROMPT,
    build_system_prompt,
    prompt_for_profile,
)
from agent.tools import (
    REGISTRY,
    CardControlArgs,
    ForecastArgs,
    HelpAnswerArgs,
    PayBillArgs,
)
from knowledge.kb import TOPICS

POST_R3_TOOLS = frozenset(
    {
        "run_analysis",
        "list_reminders",
        "list_scheduled",
        "stop_standing_order",
        "reactivate_subscription",
    }
)

NEW_TOOLS = frozenset(
    {
        "run_analysis",
        "list_bills",
        "list_reminders",
        "list_scheduled",
        "stop_standing_order",
        "reactivate_subscription",
        "pay_bill",
        "installment_quote",
        "deposit_quote",
        "forecast_cashflow",
        "help_answer",
        "list_transfers",
        "verify_payee",
        "set_card_control",
        "block_merchant",
        "unblock_merchant",
    }
)


def test_planner_prompt_is_byte_identical() -> None:
    digest = hashlib.sha256(PLANNER_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert digest == "8815d22cae24c5ea25ef37ec61cb3770bec3fca488ba726e0511036283b1aeda"


def test_new_tools_absent_from_planner_prompt() -> None:
    for name in NEW_TOOLS:
        assert name not in PLANNER_SYSTEM_PROMPT


def test_new_tools_present_in_fallback_prompt() -> None:
    for name in NEW_TOOLS:
        assert name in FALLBACK_SYSTEM_PROMPT


def test_post_r3_tools_absent_from_trained_prompt() -> None:
    # r3 is frozen: tools added after training are reached by code-side intent
    # rules, not by adding them to the model's prompt.
    for name in POST_R3_TOOLS:
        assert name not in R3_SYSTEM_PROMPT
    # The registry-rendered prompt is for building a future planner. run_analysis
    # is code-routed, so it is deliberately omitted there as well.
    for name in POST_R3_TOOLS - {"run_analysis"}:
        assert name in R3_RENDERED_PROMPT
    assert "run_analysis" not in R3_RENDERED_PROMPT


def test_r3_era_new_tools_present_in_r3_prompt() -> None:
    for name in NEW_TOOLS - POST_R3_TOOLS:
        assert name in R3_SYSTEM_PROMPT


def test_r3_lists_every_r3_era_registry_tool() -> None:
    for name in REGISTRY:
        if name in POST_R3_TOOLS:
            assert name not in R3_SYSTEM_PROMPT
        else:
            assert name in R3_SYSTEM_PROMPT


def test_r3_prompt_matches_training_hash() -> None:
    # This is the exact prompt r3 was trained with: it equals the r3 training
    # data's prompt_sha256. It is frozen and must never change.
    assert hashlib.sha256(R3_SYSTEM_PROMPT.encode("utf-8")).hexdigest() == (
        "170e21fb07579332f1ead36a6ebc8e6273bde75815be2baf49fcf05dbd7288a2"
    )


def test_r3_profile_returns_the_frozen_prompt_file() -> None:
    path = Path(__file__).resolve().parents[1] / "agent" / "prompts_r3.txt"
    assert prompt_for_profile("r3").encode("utf-8") == path.read_bytes()


def test_build_system_prompt_keeps_compatibility() -> None:
    assert build_system_prompt(False) == PLANNER_SYSTEM_PROMPT
    assert FALLBACK_SYSTEM_PROMPT.startswith(build_system_prompt(True))


def test_prompt_for_profile_dispatch() -> None:
    assert prompt_for_profile("r2") == PLANNER_SYSTEM_PROMPT
    assert prompt_for_profile("r3") == R3_SYSTEM_PROMPT
    for unknown in ("", "r1", "r4", "R2", "fallback"):
        with pytest.raises(ValueError):
            prompt_for_profile(unknown)


def test_r3_contains_help_bill_and_control_enums() -> None:
    for topic in TOPICS:
        assert topic in R3_SYSTEM_PROMPT

    bill_categories = get_args(PayBillArgs.model_fields["category"].annotation)
    assert tuple(bill_categories) == (
        "water",
        "electricity",
        "gas",
        "mobile",
        "broadband",
        "property",
    )
    for category in bill_categories:
        assert category in R3_SYSTEM_PROMPT

    controls = get_args(CardControlArgs.model_fields["control"].annotation)
    for control in controls:
        assert control in R3_SYSTEM_PROMPT


def test_r3_schema_constraints_and_outputs() -> None:
    assert "divide_by?=2..20" in R3_SYSTEM_PROMPT
    assert "day_of_month=1..28" in R3_SYSTEM_PROMPT
    assert "days?=7..60" in R3_SYSTEM_PROMPT
    assert "periods={3,6,12,24}" in R3_SYSTEM_PROMPT
    assert "term_months={3,6,12,24,36,60}" in R3_SYSTEM_PROMPT
    for output in ("balance", "transfer", "forecast", "answer", "bill"):
        assert f"-> {output}" in R3_SYSTEM_PROMPT


def test_trained_flag_matches_batch() -> None:
    for name, spec in REGISTRY.items():
        if name in NEW_TOOLS or name == "redeem_product":
            assert spec.trained is False, name
        else:
            assert spec.trained is True, name


def test_help_answer_topics_match_kb() -> None:
    literal = get_args(HelpAnswerArgs.model_fields["topic"].annotation)
    assert tuple(literal) == TOPICS


def test_pay_bill_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        PayBillArgs(category="rent")


def test_forecast_rejects_short_window() -> None:
    with pytest.raises(ValidationError):
        ForecastArgs(days=5)
