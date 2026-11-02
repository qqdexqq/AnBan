"""Registry and dispatch tests for reminders and subscription re-activation."""

from __future__ import annotations

import random
from datetime import date, datetime

import pytest

from agent.executor import ExecError, StepContext, run_step
from agent.tools import REGISTRY, CancelSubscriptionArgs, EmptyArgs
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
SECRET = b"anban-subscription-secret"


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


def _yellow_mandate(ctx, tool, *, step_id="s1", amount_fen=0):
    step = PlanStep(id=step_id, tool=tool, args={})
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


def _sub_token(ctx, sub_id: str) -> str:
    return ctx.vault.mint(ctx.session, "SUB", sub_id).label


# ------------------------------------------------------------------- registry


def test_registry_entries() -> None:
    reminders = REGISTRY["list_reminders"]
    assert reminders.kind == "read"
    assert reminders.base_tier == "green"
    assert reminders.args_model is EmptyArgs
    assert reminders.trained is False

    reactivate = REGISTRY["reactivate_subscription"]
    assert reactivate.kind == "write"
    assert reactivate.base_tier == "yellow"
    assert reactivate.args_model is CancelSubscriptionArgs
    assert reactivate.trained is False


# ------------------------------------------------------------- subscriptions


def test_list_subscriptions_includes_recurring_overview(env) -> None:
    ctx, _, _ = env

    result = _run(ctx, "list_subscriptions", {})

    assert result["sub"] == [item.id for item in ctx.bank.list_subscriptions()]
    assert isinstance(result["recurring"], list)


def test_cancel_reports_saved_monthly_amount(env) -> None:
    ctx, _, _ = env
    sub = ctx.bank.list_subscriptions()[0]
    mandate = _yellow_mandate(ctx, "cancel_subscription")

    result = _run(
        ctx,
        "cancel_subscription",
        {"sub": _sub_token(ctx, sub.id)},
        tier="yellow",
        mandate_id=mandate.id,
    )

    assert result["raw"].status == "cancelled"
    assert result["saved_monthly_fen"] == sub.amount_fen


def test_reactivate_restores_a_cancelled_subscription(env) -> None:
    ctx, _, _ = env
    sub = ctx.bank.list_subscriptions()[0]
    token = _sub_token(ctx, sub.id)

    cancel_mandate = _yellow_mandate(ctx, "cancel_subscription", step_id="s1")
    _run(
        ctx,
        "cancel_subscription",
        {"sub": token},
        tier="yellow",
        mandate_id=cancel_mandate.id,
        step_id="s1",
    )

    reactivate_mandate = _yellow_mandate(
        ctx, "reactivate_subscription", step_id="s2"
    )
    result = _run(
        ctx,
        "reactivate_subscription",
        {"sub": token},
        tier="yellow",
        mandate_id=reactivate_mandate.id,
        step_id="s2",
    )

    assert result["subscription"] == sub.id
    assert result["raw"].status == "active"
    assert result["raw"].amount_fen == sub.amount_fen
    refreshed = next(
        item for item in ctx.bank.list_subscriptions() if item.id == sub.id
    )
    assert refreshed.status == "active"


def test_reactivate_active_subscription_raises_invalid_state(env) -> None:
    ctx, _, _ = env
    sub = ctx.bank.list_subscriptions()[0]
    token = _sub_token(ctx, sub.id)
    mandate = _yellow_mandate(ctx, "reactivate_subscription")

    with pytest.raises(ExecError) as exc:
        _run(
            ctx,
            "reactivate_subscription",
            {"sub": token},
            tier="yellow",
            mandate_id=mandate.id,
        )

    assert exc.value.detail == "invalid_state"
    refreshed = next(
        item for item in ctx.bank.list_subscriptions() if item.id == sub.id
    )
    assert refreshed.status == "active"


# ----------------------------------------------------------------- reminders


def test_list_reminders_empty_without_the_table(env) -> None:
    ctx, _, _ = env

    result = _run(ctx, "list_reminders", {})

    assert result["raw"] == []
    assert result["reminder"] == []
    assert result["due"] == []


def test_list_reminders_returns_saved_items_and_due(env) -> None:
    ctx, _, _ = env

    saved = _run(
        ctx,
        "set_reminder",
        {"when": date(2026, 10, 16), "note": "pay rent"},
        step_id="s1",
    )
    assert saved["reminder"]["due_on"] == "2026-10-16"

    later = _run(
        ctx,
        "set_reminder",
        {"when": date(2026, 10, 20), "note": "buy milk"},
        step_id="s2",
    )
    assert later["reminder"]["due_on"] == "2026-10-20"

    result = _run(ctx, "list_reminders", {})

    assert len(result["reminder"]) == 2
    assert set(result["reminder"][0]) == {"due_on", "note", "created_at"}
    assert [item["note"] for item in result["reminder"]] == ["pay rent", "buy milk"]
    assert [item["note"] for item in result["due"]] == ["pay rent"]
