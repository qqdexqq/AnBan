"""Policy and executor tests for the mock-only fund redemption step.

Focused on the security-critical path: the amount must come from the bank's
quote (red / MFA), the approved mandate must bind the exact product and payout,
and a failed mandate gate must never credit the account.
"""

from __future__ import annotations

import random
from datetime import date, datetime

import pytest

from agent.executor import ExecError, StepContext, run_step
from agent.tools import REGISTRY
from bank import db
from bank.api import Bank
from bank.redemption import quote
from bank.seed import generate
from security.audit import AuditLog, create_audit_schema
from security.dag import PlanStep, validate_plan
from security.mandate import MandateService, create_security_schema
from security.policy import (
    MONEY_TOOLS,
    OUTGOING_TOOLS,
    PolicyConfig,
    evaluate,
)
from security.tokens import Session, TokenVault, build_context

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42
SECRET = b"anban-redemption-execution-secret"

P3_PAYOUT_FEN = 1_120_000


# ----------------------------------------------------------------------- setup


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


def _product_token(ctx: StepContext, product_id: str):
    build_context(ctx.bank, ctx.vault, ctx.session, ctx.now)
    matches = [
        entry
        for entry in ctx.vault.entries(ctx.session)
        if entry.kind == "PROD" and entry.real_ref == product_id
    ]
    assert len(matches) == 1, f"expected one PROD token for {product_id}"
    return matches[0]


def _redeem_step(
    ctx: StepContext, product_id: str, step_id: str = "s1"
) -> PlanStep:
    return PlanStep(
        id=step_id,
        tool="redeem_product",
        args={"product": _product_token(ctx, product_id).label},
    )


def _decide(ctx: StepContext, *steps: PlanStep):
    validated = validate_plan(
        {"kind": "plan", "steps": [step.model_dump() for step in steps]},
        ctx.vault.known_tokens(ctx.session),
    )
    return validated, evaluate(
        validated,
        bank=ctx.bank,
        vault=ctx.vault,
        session=ctx.session,
        now=ctx.now,
        config=ctx.policy_config,
    )


def _approve_red(
    ctx: StepContext,
    step: PlanStep,
    amount_fen: int,
    payee_ref: str,
):
    mandate = ctx.mandates.issue(
        customer_id=ctx.bank.customer_id,
        session_id=ctx.session.session_id,
        plan_id=ctx.session.plan_id,
        step_id=step.id,
        action=step.tool,
        payee_ref=payee_ref,
        amount_fen=amount_fen,
        tier="red",
        now=ctx.now,
    )
    code = ctx.mandates.start_mfa(mandate.id, ctx.bank.customer_id, ctx.now)
    return ctx.mandates.verify_mfa(
        mandate.id,
        ctx.bank.customer_id,
        otp=code,
        retyped_amount_fen=amount_fen,
        face_ok=True,
        now=ctx.now,
    )


def _holding(conn, customer_id: str, product_id: str):
    return conn.execute(
        """
        SELECT principal_fen, value_fen FROM holdings
        WHERE customer_id = ? AND product_id = ?
        """,
        (customer_id, product_id),
    ).fetchone()


def _redemption_count(conn, customer_id: str) -> int:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'bank_redemptions'"
    ).fetchone()
    if exists is None:
        return 0
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM bank_redemptions WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()[0]
    )


# ---------------------------------------------------------------------- policy


def test_redeem_quote_is_red_with_mfa_from_the_bank_quote(env) -> None:
    ctx, _, conn = env
    step = _redeem_step(ctx, "p3")
    _, decision = _decide(ctx, step)
    step_decision = decision.steps[0]

    assert REGISTRY["redeem_product"].base_tier == "red"
    assert "redeem_product" in MONEY_TOOLS
    assert "redeem_product" not in OUTGOING_TOOLS

    assert decision.vetoed is False
    assert step_decision.tool == "redeem_product"
    assert step_decision.tier == "red"
    assert step_decision.needs_mandate is True
    assert step_decision.needs_mfa is True
    assert step_decision.deferred is False
    # The planner never supplies this amount: it is the bank's payout.
    assert step_decision.amount_fen == quote(
        conn, ctx.bank.customer_id, "p3"
    )["payout_fen"]
    assert step_decision.amount_fen == P3_PAYOUT_FEN


def test_unquoted_or_unowned_product_is_vetoed(env) -> None:
    ctx, _, _ = env

    # p1 is a term product whose maturity is not modelled.
    _, term = _decide(ctx, _redeem_step(ctx, "p1"))
    term_step = term.steps[0]
    assert term.vetoed is True
    assert term_step.vetoed is True
    assert term_step.amount_fen is None
    assert any(reason.code == "R_BAD_ITEM" for reason in term_step.reasons)

    # p2 is not owned by this young_pro customer.
    _, unowned = _decide(ctx, _redeem_step(ctx, "p2"))
    unowned_step = unowned.steps[0]
    assert unowned.vetoed is True
    assert unowned_step.amount_fen is None
    assert any(reason.code == "R_BAD_ITEM" for reason in unowned_step.reasons)


# -------------------------------------------------------------------- executor


def test_redeem_through_an_approved_red_mandate_credits_once(env) -> None:
    ctx, _, conn = env
    step = _redeem_step(ctx, "p3")
    _, decision = _decide(ctx, step)
    payout = decision.steps[0].amount_fen
    assert payout == P3_PAYOUT_FEN

    mandate = _approve_red(ctx, step, payout, "product:p3")
    before = ctx.bank.get_balance()

    result = run_step(
        step,
        decision.steps[0],
        ctx,
        {},
        mandate_id=mandate.id,
        idempotency_key="redeem-p3-1",
    )

    assert result["amount"] == payout
    assert result["product"] == "p3"
    assert result["simulated"] is True
    assert result["raw"]["status"] == "redeemed"
    assert result["raw"]["simulated"] is True
    assert result["raw"]["receipt_id"]
    # Redemption is inbound: there is no undo to invent.
    assert "undo" not in result
    assert "undo_ref" not in result

    assert ctx.bank.get_balance() == before + payout
    holding = _holding(conn, ctx.bank.customer_id, "p3")
    assert holding["principal_fen"] == 0
    assert holding["value_fen"] == 0
    assert _redemption_count(conn, ctx.bank.customer_id) == 1
    row = conn.execute(
        "SELECT idempotency_key, quoted_amount_fen FROM bank_redemptions"
    ).fetchone()
    assert row["idempotency_key"] == "redeem-p3-1"
    assert row["quoted_amount_fen"] == payout


def test_changed_valuation_is_not_silently_requoted(env) -> None:
    ctx, _, conn = env
    step = _redeem_step(ctx, "p3")
    _, decision = _decide(ctx, step)
    payout = decision.steps[0].amount_fen
    mandate = _approve_red(ctx, step, payout, "product:p3")

    # The bank's valuation moves after consent but before execution.
    conn.execute(
        """
        UPDATE holdings SET value_fen = value_fen + 1
        WHERE customer_id = ? AND product_id = 'p3'
        """,
        (ctx.bank.customer_id,),
    )
    conn.commit()
    before = ctx.bank.get_balance()

    with pytest.raises(ExecError) as exc:
        run_step(
            step,
            decision.steps[0],
            ctx,
            {},
            mandate_id=mandate.id,
            idempotency_key="changed-valuation",
        )

    assert exc.value.code == "mandate"
    assert exc.value.detail == "mismatch"
    assert ctx.bank.get_balance() == before
    assert _redemption_count(conn, ctx.bank.customer_id) == 0


def test_redeem_without_a_mandate_does_not_credit(env) -> None:
    ctx, _, conn = env
    step = _redeem_step(ctx, "p3")
    _, decision = _decide(ctx, step)
    before = ctx.bank.get_balance()
    before_holding = tuple(_holding(conn, ctx.bank.customer_id, "p3"))

    with pytest.raises(ExecError) as exc:
        run_step(step, decision.steps[0], ctx, {}, idempotency_key="no-mandate")

    assert exc.value.code == "mandate"
    assert exc.value.detail == "missing"
    assert ctx.bank.get_balance() == before
    assert tuple(_holding(conn, ctx.bank.customer_id, "p3")) == before_holding
    assert _redemption_count(conn, ctx.bank.customer_id) == 0


def test_redeem_with_a_red_mandate_but_no_mfa_does_not_credit(env) -> None:
    ctx, _, conn = env
    step = _redeem_step(ctx, "p3")
    _, decision = _decide(ctx, step)
    payout = decision.steps[0].amount_fen

    mandate = ctx.mandates.issue(
        customer_id=ctx.bank.customer_id,
        session_id=ctx.session.session_id,
        plan_id=ctx.session.plan_id,
        step_id=step.id,
        action=step.tool,
        payee_ref="product:p3",
        amount_fen=payout,
        tier="red",
        now=ctx.now,
    )
    assert mandate.mfa_verified is False
    before = ctx.bank.get_balance()
    before_holding = tuple(_holding(conn, ctx.bank.customer_id, "p3"))

    with pytest.raises(ExecError) as exc:
        run_step(
            step,
            decision.steps[0],
            ctx,
            {},
            mandate_id=mandate.id,
            idempotency_key="no-mfa",
        )

    assert exc.value.code == "mandate"
    assert exc.value.detail == "mfa_missing"
    assert ctx.bank.get_balance() == before
    assert tuple(_holding(conn, ctx.bank.customer_id, "p3")) == before_holding
    assert _redemption_count(conn, ctx.bank.customer_id) == 0


def test_product_substitution_is_rejected_and_does_not_credit(env) -> None:
    ctx, _, conn = env
    # Give the customer a second redeemable holding so the substituted step
    # would be executable if the mandate did not bind the product.
    ctx.bank.subscribe_product("p2", 20_000, ctx.now)

    approved_step = _redeem_step(ctx, "p3", step_id="s1")
    _, decision = _decide(ctx, approved_step)
    payout = decision.steps[0].amount_fen
    mandate = _approve_red(ctx, approved_step, payout, "product:p3")

    substituted = PlanStep(
        id="s1",
        tool="redeem_product",
        args={"product": _product_token(ctx, "p2").label},
    )
    before = ctx.bank.get_balance()
    before_p2 = tuple(_holding(conn, ctx.bank.customer_id, "p2"))
    before_p3 = tuple(_holding(conn, ctx.bank.customer_id, "p3"))

    with pytest.raises(ExecError) as exc:
        run_step(
            substituted,
            decision.steps[0],
            ctx,
            {},
            mandate_id=mandate.id,
            idempotency_key="substituted",
        )

    assert exc.value.code == "mandate"
    assert exc.value.detail == "mismatch"
    assert ctx.bank.get_balance() == before
    assert tuple(_holding(conn, ctx.bank.customer_id, "p2")) == before_p2
    assert tuple(_holding(conn, ctx.bank.customer_id, "p3")) == before_p3
    assert _redemption_count(conn, ctx.bank.customer_id) == 0
