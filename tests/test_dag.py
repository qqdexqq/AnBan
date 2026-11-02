"""DAG validator and tool registry tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from agent.tools import REGISTRY
from security.dag import PlanError, parse_planner_output, validate_plan

SPEC_PATH = Path(__file__).resolve().parents[1] / "data_spec" / "scenarios.yaml"

KNOWN_TOKENS = {
    "<PAYEE_1>": "PAYEE",
    "<AMT_1>": "AMT",
    "<DATE_1>": "DATE",
    "<SUB_1>": "SUB",
    "<CARD_1>": "CARD",
    "<PROD_1>": "PROD",
    "<MERCHANT_1>": "MERCHANT",
    "<MERCHANT_2>": "MERCHANT",
    "<ACCT_A>": "ACCT",
    "<RAW_1>": "RAW",
}

BIRTHDAY_PLAN = {
    "kind": "plan",
    "steps": [
        {
            "id": "s1",
            "tool": "get_contact_date",
            "args": {"payee": "<PAYEE_1>", "what": "birthday"},
            "after": [],
        },
        {
            "id": "s2",
            "tool": "can_cover",
            "args": {"amount": "<AMT_1>"},
            "after": [],
        },
        {
            "id": "s3",
            "tool": "move_to_pot",
            "args": {"pot_name": "birthday", "amount": "<AMT_1>"},
            "after": ["s2"],
        },
        {
            "id": "s4",
            "tool": "merchant_order",
            "args": {
                "merchant": "<MERCHANT_1>",
                "item": "bouquet_m",
                "deliver": {"date": "$s1.date", "offset_days": -2},
            },
            "after": ["s1", "s3"],
        },
        {
            "id": "s5",
            "tool": "merchant_order",
            "args": {
                "merchant": "<MERCHANT_2>",
                "item": "cake_8in",
                "deliver": {"date": "$s1.date", "offset_days": -2},
            },
            "after": ["s1", "s3"],
        },
    ],
}


def _load_scenarios() -> list[dict[str, Any]]:
    with SPEC_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["scenarios"]


def _flatten_gold_steps(items: list[Any]) -> list[str]:
    tools: list[str] = []
    for item in items:
        if isinstance(item, str):
            tools.append(item)
        elif isinstance(item, dict) and "parallel" in item:
            tools.extend(_flatten_gold_steps(item["parallel"]))
        else:
            raise AssertionError(f"unexpected gold step {item!r}")
    return tools


def _value_for(name: str) -> Any:
    return {
        "payee": "<PAYEE_1>",
        "query": "<PAYEE_1>",
        "amount": "<AMT_1>",
        "date": "<DATE_1>",
        "deliver": "<DATE_1>",
        "when": "<DATE_1>",
        "sub": "<SUB_1>",
        "card": "<CARD_1>",
        "product": "<PROD_1>",
        "merchant": "<MERCHANT_1>",
        "account": "<ACCT_A>",
        "pot_name": "birthday",
        "item": "bouquet_m",
        "period": "this_month",
        "category": "food",
        "day_of_month": 15,
        "what": "birthday",
        "question": "Which one?",
        "options": ["<PAYEE_1>"],
        "reason": "out_of_scope",
    }[name]


def _fill_args(tool: str) -> dict[str, Any]:
    spec = REGISTRY[tool]
    return {
        name: _value_for(name)
        for name, field in spec.args_model.model_fields.items()
        if field.is_required()
    }


def _plan_from_gold(gold: dict[str, Any]) -> dict[str, Any]:
    steps = []
    for index, tool in enumerate(_flatten_gold_steps(gold["steps"]), start=1):
        steps.append(
            {
                "id": f"s{index}",
                "tool": tool,
                "args": _fill_args(tool),
                "after": [],
            }
        )
    return {"kind": "plan", "steps": steps}


def _plan(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "plan", "steps": list(steps)}


def _step(
    step_id: str,
    tool: str,
    args: dict[str, Any] | None = None,
    after: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": step_id,
        "tool": tool,
        "args": _fill_args(tool) if args is None else args,
        "after": [] if after is None else after,
    }


def _assert_code(plan: Any, code: str, known: dict[str, str] | None = None) -> None:
    with pytest.raises(PlanError) as excinfo:
        validate_plan(plan, KNOWN_TOKENS if known is None else known)
    assert excinfo.value.code == code


def test_registry_invariants() -> None:
    control_tools = {"ask_user", "refuse", "handoff_human"}

    for spec in REGISTRY.values():
        if spec.kind == "write" and spec.money_args:
            assert spec.base_tier in {"yellow", "red"}, (
                f"{spec.name} moves money but has base_tier {spec.base_tier}"
            )
        if spec.undo is not None:
            assert spec.undo in REGISTRY, f"{spec.name} undo tool missing"

    control_names = {name for name, spec in REGISTRY.items() if spec.kind == "control"}
    assert control_names == control_tools


def test_plan_scenarios_validate() -> None:
    scenarios = [
        scenario
        for scenario in _load_scenarios()
        if scenario["gold"].get("kind") == "plan"
    ]
    assert scenarios
    for scenario in scenarios:
        plan = _plan_from_gold(scenario["gold"])
        validated = validate_plan(plan, KNOWN_TOKENS)
        assert validated.kind == "plan"
        assert [step.tool for step in validated.steps] == _flatten_gold_steps(
            scenario["gold"]["steps"]
        ), scenario["id"]


def test_birthday_contract_example_validates() -> None:
    validated = validate_plan(BIRTHDAY_PLAN, KNOWN_TOKENS)
    assert [step.id for step in validated.steps] == ["s1", "s2", "s3", "s4", "s5"]

    groups = [[step.id for step in group] for group in validated.parallel_groups()]
    assert {"s4", "s5"} in [set(group) for group in groups]
    assert groups == [["s1", "s2"], ["s3"], ["s4", "s5"]]


def test_fenced_json_parses() -> None:
    text = (
        "```json\n"
        '{"kind": "refuse", "reason": "out_of_scope"}\n'
        "```"
    )
    output = parse_planner_output(text)
    assert output.kind == "refuse"


def test_prose_plus_json_is_rejected() -> None:
    with pytest.raises(PlanError):
        parse_planner_output(
            'Sure: {"kind": "refuse", "reason": "out_of_scope"}'
        )


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(PlanError):
        parse_planner_output('{"kind": "refuse", "reason":')


def test_clarify_options_must_be_known_tokens() -> None:
    output = {
        "kind": "clarify",
        "question": "Which one?",
        "options": ["<PAYEE_1>"],
    }
    validated = validate_plan(output, KNOWN_TOKENS)
    assert validated.kind == "clarify"

    _assert_code(
        {"kind": "clarify", "question": "Which one?", "options": ["<PAYEE_9>"]},
        "unknown_token",
    )
    _assert_code(
        {"kind": "clarify", "question": "Which one?", "options": ["<RAW_1>"]},
        "wrong_token_kind",
    )


@pytest.mark.parametrize(
    ("plan", "code"),
    [
        (_plan(), "empty_plan"),
        (
            _plan(
                *[
                    _step(f"s{index}", "get_balance")
                    for index in range(1, 14)
                ]
            ),
            "too_many_steps",
        ),
        (
            _plan(_step("s1", "get_balance"), _step("s1", "get_balance")),
            "duplicate_step_id",
        ),
        (_plan(_step("s1", "exec_shell", {})), "unknown_tool"),
        (
            _plan(_step("s1", "ask_user")),
            "control_tool_in_plan",
        ),
        (_plan(_step("s1", "refuse")), "control_tool_in_plan"),
        (
            _plan(
                _step("s1", "get_balance", after=["s2"]),
                _step("s2", "get_balance", after=["s1"]),
            ),
            "cycle",
        ),
        (
            _plan(
                _step(
                    "s1",
                    "transfer",
                    {"payee": "<PAYEE_1>", "amount": "$s9.amount"},
                )
            ),
            "bad_step_ref",
        ),
        (
            _plan(
                _step(
                    "s1",
                    "transfer",
                    {"payee": "$s1.payee", "amount": "<AMT_1>"},
                )
            ),
            "bad_args",
        ),
        (
            _plan(
                _step(
                    "s1",
                    "transfer",
                    {"payee": "<PAYEE_1>", "amount": "<AMT_1>", "bonus": 1},
                )
            ),
            "bad_args",
        ),
        (
            _plan(
                _step(
                    "s1",
                    "merchant_order",
                    {
                        "merchant": "<MERCHANT_1>",
                        "item": "bouquet_m",
                        "deliver": "$s2.date",
                    },
                    after=["s2"],
                ),
                _step("s2", "can_cover"),
            ),
            "bad_step_ref",
        ),
        (
            _plan(
                _step(
                    "s1",
                    "transfer",
                    {"payee": "<PAYEE_9>", "amount": "<AMT_1>"},
                )
            ),
            "unknown_token",
        ),
        (
            _plan(
                _step(
                    "s1",
                    "transfer",
                    {"payee": "<RAW_1>", "amount": "<AMT_1>"},
                )
            ),
            "wrong_token_kind",
        ),
    ],
)
def test_rejection_codes(plan: Any, code: str) -> None:
    _assert_code(plan, code)


@pytest.mark.parametrize(
    "args",
    [
        {"payee": "<PAYEE_1>", "amount": "5000"},
        {"payee": "<PAYEE_1>", "amount": 5000},
        {"payee": "6222 0000 1234 5678", "amount": "<AMT_1>"},
        {"payee": "<RAW_1>", "amount": "<AMT_1>"},
        {"payee": "<PAYEE_1>", "amount": "<AMT_1>", "unknown": "x"},
        # Non-ASCII digits must not satisfy a token pattern.
        {"payee": "<PAYEE_\uff11>", "amount": "<AMT_1>"},
        {"payee": "<PAYEE_1>", "amount": "<AMT_\u0661>"},
    ],
)
def test_bad_transfer_args_rejected(args: dict[str, Any]) -> None:
    expected = "wrong_token_kind" if args["payee"] == "<RAW_1>" else "bad_args"
    _assert_code(_plan(_step("s1", "transfer", args)), expected)


def test_parser_rejects_deep_nesting_and_huge_output() -> None:
    """Regression: hostile planner output must fail as PlanError, never crash."""
    deep = '{"kind":"plan","steps":' + "[" * 100_000 + "]" * 100_000 + "}"
    with pytest.raises(PlanError):
        parse_planner_output(deep)
    with pytest.raises(PlanError):
        parse_planner_output('{"kind":"refuse","reason":"' + "x" * 50_000 + '"}')
