"""Executor dispatch tests for the untrained tool batch."""

from __future__ import annotations

import random
from dataclasses import replace
from datetime import date, datetime, timedelta

import pytest

from bank import bills

from agent.executor import ExecError, StepContext, run_step
from bank import db
from bank.api import Bank
from bank.seed import generate
from security.audit import AuditLog, create_audit_schema
from security.dag import PlanStep
from security.mandate import MandateService, create_security_schema
from security.policy import PolicyConfig, StepDecision
from security.tokens import Session, TokenVault

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42
SECRET = b"anban-executor-secret"


def _make_env(persona: str = "young_pro"):
    state = generate(persona, SEED, TODAY)
    conn = db.connect(":memory:")
    db.create_schema(conn)
    db.load_state(conn, state)
    create_security_schema(conn)
    create_audit_schema(conn)

    bank = Bank(conn, state.customer.id)
    ctx = StepContext(
        bank=bank,
        vault=TokenVault(SECRET),
        session=Session(customer_id=state.customer.id),
        mandates=MandateService(conn, SECRET, otp_rng=random.Random(7)),
        audit=AuditLog(conn, SECRET),
        policy_config=PolicyConfig(),
        now=NOW,
        secret_key=SECRET,
    )
    return ctx, state, conn


@pytest.fixture
def env():
    ctx, state, conn = _make_env()
    try:
        yield ctx, state, conn
    finally:
        conn.close()


def _run(ctx, tool, args, *, tier="green", mandate_id=None, step_id="s1"):
    step = PlanStep(id=step_id, tool=tool, args=args)
    decision = StepDecision(step_id=step_id, tool=tool, tier=tier)
    return run_step(step, decision, ctx, {}, mandate_id=mandate_id)


def _yellow_mandate(ctx, step, amount_fen=0):
    mandate = ctx.mandates.issue(
        customer_id=ctx.bank.customer_id,
        session_id=ctx.session.session_id,
        plan_id=ctx.session.plan_id,
        step_id=step.id,
        action=step.tool,
        payee_ref="",
        amount_fen=amount_fen,
        tier="yellow",
        now=ctx.now,
    )
    return ctx.mandates.approve_yellow(mandate.id, ctx.bank.customer_id, ctx.now)


# ---------------------------------------------------------------- happy paths


def test_list_bills(env) -> None:
    ctx, _, _ = env

    result = _run(ctx, "list_bills", {})

    assert result["bill"]
    assert set(result["bill"]) == {
        "water",
        "electricity",
        "gas",
        "mobile",
        "broadband",
        "property",
    }
    assert set(result["raw"][0]) == {
        "category",
        "biller",
        "biller_en",
        "amount_due_fen",
        "due_date",
    }


def _water_due(ctx) -> int:
    # The mandate is bound to the bill's own amount.
    return bills.due_amount(ctx.bank.conn, ctx.bank.customer_id, "water", ctx.now.date())


def test_pay_bill_with_a_mandate_for_another_amount_is_refused(env) -> None:
    ctx, _, _ = env
    step = PlanStep(id="s1", tool="pay_bill", args={"category": "water"})
    mandate = _yellow_mandate(ctx, step, 1)

    with pytest.raises(ExecError):
        run_step(step, StepDecision(step_id="s1", tool="pay_bill", tier="yellow"), ctx, {}, mandate_id=mandate.id)


def test_pay_bill(env) -> None:
    ctx, _, conn = env
    step = PlanStep(id="s1", tool="pay_bill", args={"category": "water"})
    mandate = _yellow_mandate(ctx, step, _water_due(ctx))

    result = run_step(
        step,
        StepDecision(step_id="s1", tool="pay_bill", tier="yellow"),
        ctx,
        {},
        mandate_id=mandate.id,
    )

    assert result["bill"] == "water"
    assert result["amount"] > 0
    assert result["raw"]["status"] == "paid"
    txn = conn.execute(
        "SELECT amount_fen FROM transactions WHERE category = 'utilities'"
    ).fetchone()
    assert txn["amount_fen"] == -result["amount"]


def test_pay_bill_twice_raises(env) -> None:
    ctx, _, _ = env
    step = PlanStep(id="s1", tool="pay_bill", args={"category": "water"})
    decision = StepDecision(step_id="s1", tool="pay_bill", tier="yellow")

    first = _yellow_mandate(ctx, step, _water_due(ctx))
    run_step(step, decision, ctx, {}, mandate_id=first.id)

    second = _yellow_mandate(ctx, step)
    with pytest.raises(ExecError) as exc:
        run_step(step, decision, ctx, {}, mandate_id=second.id)

    assert exc.value.code == "bank"
    assert exc.value.detail == "not_found"


def test_installment_quote(env) -> None:
    ctx, _, _ = env

    result = _run(
        ctx, "installment_quote", {"amount": 120_000, "periods": 12}
    )

    assert result["quote"] is result["raw"]
    assert result["raw"]["principal_fen"] == 120_000
    assert result["raw"]["periods"] == 12
    assert result["raw"]["fee_rate_per_period_pct"] == 0.60
    assert result["raw"]["payment_fen"] > 0


def test_deposit_quote(env) -> None:
    ctx, _, _ = env

    result = _run(ctx, "deposit_quote", {"amount": 500_000, "term_months": 12})

    assert result["quote"] is result["raw"]
    assert result["raw"]["principal_fen"] == 500_000
    assert result["raw"]["term_months"] == 12
    assert result["raw"]["rate_pct"] == 1.15
    assert result["raw"]["interest_fen"] > result["raw"]["early_half_term_fen"]


def test_forecast_cashflow(env) -> None:
    ctx, _, _ = env

    result = _run(ctx, "forecast_cashflow", {"days": 30})

    assert result["forecast"] is result["raw"]
    assert result["raw"]["days"] == 30
    assert result["raw"]["start_balance_fen"] == ctx.bank.get_balance()
    for item in result["raw"]["items"]:
        assert 1 <= item["day"] <= 31
        assert {"day", "amount_fen", "label"} <= set(item)
        assert item["recurrence"] in {"monthly", "one_off"}
        if item["recurrence"] == "one_off":
            assert item["date"]


def test_forecast_detects_the_seeded_income_row_as_salary(env) -> None:
    ctx, _, _ = env

    result = _run(ctx, "forecast_cashflow", {"days": 30})

    salary_items = [
        item for item in result["raw"]["items"] if item["label"] == "salary"
    ]
    assert salary_items
    assert salary_items[0]["amount_fen"] > 0
    assert salary_items[0]["assumption"] is True


def test_forecast_bills_use_their_actual_dates_once(env) -> None:
    ctx, _, _ = env

    result = _run(ctx, "forecast_cashflow", {"days": 60})
    bill_items = [
        item for item in result["raw"]["items"] if item["recurrence"] == "one_off"
    ]
    assert bill_items
    labels = [item["label"] for item in bill_items]
    assert len(labels) == len(set(labels))

    due_dates = {
        row["biller_en"]: row["due_date"]
        for row in bills.list_bills(ctx.bank.conn, ctx.bank.customer_id)
    }
    for item in bill_items:
        assert item["date"] == due_dates[item["label"]]


def test_forecast_does_not_double_debit_a_pending_transfer(env) -> None:
    ctx, state, _ = env
    contact = state.contacts[0]

    before = ctx.bank.get_balance()
    ctx.bank.transfer(
        to_contact_id=contact.id,
        amount_fen=30_000,
        note="pending",
        idempotency_key="forecast-pending-1",
        now=NOW,
    )
    spendable = ctx.bank.get_balance()
    assert spendable == before - 30_000

    result = _run(ctx, "forecast_cashflow", {"days": 30})

    # The pending transfer already left the spendable balance; subtracting it
    # again would double-count the hold.
    assert result["raw"]["start_balance_fen"] == spendable


def test_forecast_does_not_go_negative_for_rich_persona() -> None:
    ctx, _, conn = _make_env("retiree")
    try:
        result = _run(ctx, "forecast_cashflow", {"days": 30})
        assert result["raw"]["goes_negative"] is False
    finally:
        conn.close()


def test_help_answer(env) -> None:
    ctx, _, _ = env

    result = _run(ctx, "help_answer", {"topic": "deposit_insurance"})

    assert result["answer"] == "deposit_insurance"
    assert result["raw"]["topic"] == "deposit_insurance"
    assert result["raw"]["zh"] and result["raw"]["en"] and result["raw"]["source"]


def test_list_transfers_held_then_sent(env) -> None:
    ctx, state, _ = env
    contact = state.contacts[0]
    ctx.bank.transfer(
        to_contact_id=contact.id,
        amount_fen=30_000,
        note="test",
        idempotency_key="lt-1",
        now=NOW,
    )

    held = _run(ctx, "list_transfers", {})
    assert held["transfer"]
    assert held["raw"][0]["status"] == "held"

    later = replace(ctx, now=NOW + timedelta(hours=3))
    sent = _run(later, "list_transfers", {})
    assert sent["raw"][0]["status"] == "sent"


def test_verify_payee_first_time(env) -> None:
    ctx, state, _ = env
    contact = state.contacts[0]

    result = _run(ctx, "verify_payee", {"payee": contact.id})

    assert result["raw"]["payee"] == contact.id
    assert result["raw"]["first_time"] is True
    assert result["raw"]["on_fraud_list"] is False
    assert result["risk"] is False


def test_set_card_control(env) -> None:
    ctx, _, _ = env

    result = _run(
        ctx, "set_card_control", {"control": "online", "enabled": False}
    )

    assert result["card"]
    assert result["raw"]["card_id"] == result["card"]
    assert result["raw"]["enabled"] is False


def test_block_merchant(env) -> None:
    ctx, _, _ = env

    result = _run(ctx, "block_merchant", {"category": "gambling"})

    assert result["block"] == "gambling"
    assert result["raw"]["blocked"] is True


def test_unblock_merchant(env) -> None:
    ctx, _, _ = env
    _run(ctx, "block_merchant", {"category": "gambling"})

    step = PlanStep(
        id="s2", tool="unblock_merchant", args={"category": "gambling"}
    )
    mandate = _yellow_mandate(ctx, step)

    result = run_step(
        step,
        StepDecision(step_id="s2", tool="unblock_merchant", tier="yellow"),
        ctx,
        {},
        mandate_id=mandate.id,
    )

    assert result["block"] == "gambling"
    assert result["raw"]["unblock_at"]
