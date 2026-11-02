"""Executor tests for the standing-order listing and stopping tools."""

from __future__ import annotations

import random
from datetime import date, datetime

import pytest

from agent.executor import ExecError, StepContext, run_step
from agent.tools import REGISTRY
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
SECRET = b"anban-standing-order-secret"


@pytest.fixture
def env():
    state = generate("young_pro", SEED, TODAY)
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
    try:
        yield ctx, state, conn
    finally:
        conn.close()


def _run(ctx, tool, args, *, tier="green", step_id="s1"):
    step = PlanStep(id=step_id, tool=tool, args=args)
    decision = StepDecision(step_id=step_id, tool=tool, tier=tier)
    return run_step(step, decision, ctx, {})


def _payee_token(ctx, contact) -> str:
    return ctx.vault.mint(ctx.session, "PAYEE", contact.id, {}, "user_text").label


def _schedule(bank, contact, amount_fen=100_000, day=5, note="rent"):
    return bank.schedule_transfer(contact.id, amount_fen, day, note)


def _display(contact) -> str:
    return f"{contact.name} ···{str(contact.account_no)[-4:]}"


def test_registry_tiers() -> None:
    listing = REGISTRY["list_scheduled"]
    assert listing.kind == "read"
    assert listing.base_tier == "green"

    stopping = REGISTRY["stop_standing_order"]
    assert stopping.kind == "write"
    assert stopping.base_tier == "yellow"


def test_list_scheduled_returns_the_order_with_a_masked_payee(env) -> None:
    ctx, state, _ = env
    contact = state.contacts[0]
    _schedule(ctx.bank, contact, 100_000, 5)

    result = _run(ctx, "list_scheduled", {})

    assert result["scheduled"] is result["raw"]
    item = result["scheduled"][0]
    assert item["scheduled_id"]
    assert item["amount_fen"] == 100_000
    assert item["day_of_month"] == 5
    assert item["status"] == "scheduled"
    assert "contact_id" not in item
    assert item["payee_display"] == _display(contact)


def test_stop_cancels_the_order_and_a_second_stop_is_not_found(env) -> None:
    ctx, state, conn = env
    contact = state.contacts[0]
    order = _schedule(ctx.bank, contact, 20_000, 9)
    token = _payee_token(ctx, contact)

    result = _run(ctx, "stop_standing_order", {"payee": token})

    assert result["raw"]["stopped"] == result["stopped"]
    assert len(result["stopped"]) == 1
    stopped = result["stopped"][0]
    assert stopped["payee_display"] == _display(contact)
    assert stopped["amount_fen"] == 20_000
    assert stopped["day_of_month"] == 9
    status = conn.execute(
        "SELECT status FROM scheduled_transfers WHERE id = ?", (order.id,)
    ).fetchone()["status"]
    assert status == "cancelled"

    with pytest.raises(ExecError) as exc:
        _run(ctx, "stop_standing_order", {"payee": token})

    assert exc.value.code == "bank"
    assert exc.value.detail == "not_found"


def test_stop_cancels_every_order_for_the_payee(env) -> None:
    ctx, state, conn = env
    contact = state.contacts[0]
    first = _schedule(ctx.bank, contact, 10_000, 3)
    second = _schedule(ctx.bank, contact, 15_000, 21)
    token = _payee_token(ctx, contact)

    result = _run(ctx, "stop_standing_order", {"payee": token})

    assert len(result["stopped"]) == 2
    assert {item["amount_fen"] for item in result["stopped"]} == {10_000, 15_000}
    rows = conn.execute(
        "SELECT id, status FROM scheduled_transfers WHERE to_contact_id = ?",
        (contact.id,),
    ).fetchall()
    assert {row["id"] for row in rows} == {first.id, second.id}
    assert all(row["status"] == "cancelled" for row in rows)
