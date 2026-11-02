"""Preflight budget checks for merchant baskets funded by a same-plan pot."""

from __future__ import annotations

from datetime import date

import pytest

from agent.basket_preflight import budget_question
from security.dag import (
    PlanError,
    PlanOutput,
    PlanStep,
    ValidatedPlan,
    validate_plan,
)
from security.tokens import Session, TokenVault

SECRET = b"anban-test-secret"


def _env() -> tuple[TokenVault, Session]:
    return TokenVault(SECRET), Session(customer_id="cust_test")


def _amount(vault: TokenVault, session: Session, fen: int) -> str:
    return vault.mint(
        session, "AMT", fen, {"bucket": "1000-5000"}, "user_text"
    ).label


def _merchant(vault: TokenVault, session: Session, name: str) -> str:
    return vault.mint(session, "MERCHANT", name, {}, "bank_context").label


def _date(vault: TokenVault, session: Session) -> str:
    return vault.mint(
        session, "DATE", date(2026, 12, 1), {"relative": "in 1 day"}, "bank_context"
    ).label


def _move(step_id: str, pot: str, amount: str) -> dict:
    return {
        "id": step_id,
        "tool": "move_to_pot",
        "args": {"pot_name": pot, "amount": amount},
        "after": [],
    }


def _order(
    step_id: str,
    merchant: str,
    item: str,
    deliver: str,
    funding: str,
    after: list[str],
) -> dict:
    args = {"merchant": merchant, "item": item, "deliver": deliver}
    if funding:
        args["funding_pot"] = funding
    return {
        "id": step_id,
        "tool": "merchant_order",
        "args": args,
        "after": after,
    }


def _validated(
    vault: TokenVault, session: Session, steps: list[dict], known: dict | None = None
) -> ValidatedPlan:
    plan = validate_plan(
        {"kind": "plan", "steps": steps},
        vault.known_tokens(session) if known is None else known,
    )
    assert plan.kind == "plan"
    return plan


def _birthday_plan(
    vault: TokenVault, session: Session, budget_fen: int
) -> ValidatedPlan:
    amount = _amount(vault, session, budget_fen)
    florist = _merchant(vault, session, "花店")
    cake = _merchant(vault, session, "蛋糕店")
    deliver = _date(vault, session)
    return _validated(
        vault,
        session,
        [
            _move("s1", "birthday", amount),
            _order("s2", florist, "bouquet_m", deliver, "$s1.pot", ["s1"]),
            _order("s3", cake, "cake_8in", deliver, "$s1.pot", ["s1"]),
        ],
    )


def test_500_budget_against_567_basket_asks_to_increase_or_change() -> None:
    vault, session = _env()
    plan = _birthday_plan(vault, session, 50_000)

    question = budget_question(plan, vault, session, "en")

    assert question is not None
    assert "¥567.00" in question
    assert "¥500.00" in question
    assert "¥67.00" in question
    assert "increase the budget" in question
    assert "change the items" in question


def test_chinese_question_is_localised_with_the_same_exact_amounts() -> None:
    vault, session = _env()
    plan = _birthday_plan(vault, session, 50_000)

    question = budget_question(plan, vault, session, "zh")

    assert question is not None
    assert "¥567.00" in question
    assert "¥500.00" in question
    assert "¥67.00" in question
    assert "增加预算" in question
    assert "更换商品" in question


@pytest.mark.parametrize("budget_fen", [56_700, 100_000])
def test_exact_or_larger_budget_passes(budget_fen: int) -> None:
    vault, session = _env()
    plan = _birthday_plan(vault, session, budget_fen)

    assert budget_question(plan, vault, session) is None


def test_independent_pot_budgets_are_not_merged() -> None:
    vault, session = _env()
    amount = _amount(vault, session, 30_000)
    florist = _merchant(vault, session, "花店")
    deliver = _date(vault, session)
    plan = _validated(
        vault,
        session,
        [
            _move("s1", "gift", amount),
            _order("s2", florist, "bouquet_m", deliver, "$s1.pot", ["s1"]),
            _move("s3", "party", amount),
            _order("s4", florist, "bouquet_m", deliver, "$s3.pot", ["s3"]),
        ],
    )

    # Two separate ¥299 baskets each fit their own ¥300 reserve; the combined
    # ¥598 must not be charged against either single reserve.
    assert budget_question(plan, vault, session) is None


def test_duplicate_orders_on_one_reserve_are_aggregated() -> None:
    vault, session = _env()
    amount = _amount(vault, session, 50_000)
    florist = _merchant(vault, session, "花店")
    deliver = _date(vault, session)
    plan = _validated(
        vault,
        session,
        [
            _move("s1", "birthday", amount),
            _order("s2", florist, "bouquet_m", deliver, "$s1.pot", ["s1"]),
            _order("s3", florist, "bouquet_m", deliver, "$s1.pot", ["s1"]),
        ],
    )

    question = budget_question(plan, vault, session)

    assert question is not None
    assert "¥598.00" in question
    assert "¥500.00" in question
    assert "¥98.00" in question


def test_non_basket_plan_returns_none() -> None:
    vault, session = _env()
    amount = _amount(vault, session, 50_000)
    florist = _merchant(vault, session, "花店")
    deliver = _date(vault, session)
    plan = _validated(
        vault,
        session,
        [_order("s1", florist, "bouquet_m", deliver, "", [])],
    )

    assert budget_question(plan, vault, session) is None


def test_malformed_funding_reference_is_rejected_by_schema_first() -> None:
    vault, session = _env()
    amount = _amount(vault, session, 50_000)
    florist = _merchant(vault, session, "花店")
    deliver = _date(vault, session)
    steps = [
        _move("s1", "birthday", amount),
        _order("s2", florist, "bouquet_m", deliver, "$s1.amount", ["s1"]),
    ]

    with pytest.raises(PlanError):
        validate_plan(
            {"kind": "plan", "steps": steps}, vault.known_tokens(session)
        )


def test_malformed_funding_reference_fails_closed_even_if_unvalidated() -> None:
    vault, session = _env()
    amount = _amount(vault, session, 50_000)
    florist = _merchant(vault, session, "花店")
    deliver = _date(vault, session)
    plan = ValidatedPlan(
        kind="plan",
        steps=[
            PlanStep(
                id="s1",
                tool="move_to_pot",
                args={"pot_name": "birthday", "amount": amount},
            ),
            PlanStep(
                id="s2",
                tool="merchant_order",
                args={
                    "merchant": florist,
                    "item": "bouquet_m",
                    "deliver": deliver,
                    "funding_pot": "$s1.amount",
                },
                after=["s1"],
            ),
        ],
        output=PlanOutput(kind="plan"),
    )

    with pytest.raises(ValueError, match="basket_quote_unavailable"):
        budget_question(plan, vault, session)


def test_direct_ref_to_non_pot_step_fails_closed() -> None:
    vault, session = _env()
    amount = _amount(vault, session, 50_000)
    florist = _merchant(vault, session, "花店")
    deliver = _date(vault, session)
    plan = ValidatedPlan(
        kind="plan",
        steps=[
            PlanStep(
                id="s1",
                tool="can_cover",
                args={"amount": amount},
            ),
            PlanStep(
                id="s2",
                tool="merchant_order",
                args={
                    "merchant": florist,
                    "item": "bouquet_m",
                    "deliver": deliver,
                    "funding_pot": "$s1.pot",
                },
                after=["s1"],
            ),
        ],
        output=PlanOutput(kind="plan"),
    )

    with pytest.raises(ValueError, match="basket_quote_unavailable"):
        budget_question(plan, vault, session)


def test_token_from_another_session_fails_closed_after_schema_validation() -> None:
    vault, session = _env()
    other = Session(customer_id=session.customer_id)
    leaked = _amount(vault, other, 50_000)
    florist = _merchant(vault, session, "花店")
    deliver = _date(vault, session)
    known = vault.known_tokens(session) | {leaked: "AMT"}
    plan = _validated(
        vault,
        session,
        [
            _move("s1", "birthday", leaked),
            _order("s2", florist, "bouquet_m", deliver, "$s1.pot", ["s1"]),
        ],
        known=known,
    )

    with pytest.raises(ValueError, match="basket_quote_unavailable"):
        budget_question(plan, vault, session)


def test_unknown_merchant_token_fails_closed() -> None:
    vault, session = _env()
    amount = _amount(vault, session, 50_000)
    deliver = _date(vault, session)
    plan = ValidatedPlan(
        kind="plan",
        steps=[
            PlanStep(
                id="s1",
                tool="move_to_pot",
                args={"pot_name": "birthday", "amount": amount},
            ),
            PlanStep(
                id="s2",
                tool="merchant_order",
                args={
                    "merchant": "<MERCHANT_99>",
                    "item": "bouquet_m",
                    "deliver": deliver,
                    "funding_pot": "$s1.pot",
                },
                after=["s1"],
            ),
        ],
        output=PlanOutput(kind="plan"),
    )

    with pytest.raises(ValueError, match="basket_quote_unavailable"):
        budget_question(plan, vault, session)


def test_inputs_and_vault_are_not_mutated() -> None:
    vault, session = _env()
    plan = _birthday_plan(vault, session, 50_000)
    before_plan = plan.model_dump()
    before_tokens = {
        label: (entry.kind, entry.real_ref, entry.origin, entry.sig)
        for label, entry in vault.store[session.session_id].items()
    }

    first = budget_question(plan, vault, session)
    second = budget_question(plan, vault, session)

    assert first is not None and first == second
    assert plan.model_dump() == before_plan
    assert {
        label: (entry.kind, entry.real_ref, entry.origin, entry.sig)
        for label, entry in vault.store[session.session_id].items()
    } == before_tokens
