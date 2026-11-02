"""Executor tests against the real seeded bank, vault, DAG, policy and mandates."""

from __future__ import annotations

import json
import random
import re
from dataclasses import replace
from datetime import date, datetime, timedelta

import pytest

from agent.executor import (
    ExecError,
    StepContext,
    resolve_args,
    run_step,
    undo_step,
)
from agent.tools import REGISTRY
from bank import db
from bank.api import Bank
from bank.merchants import price_of
from bank.seed import generate
from security.audit import AuditKind, AuditLog, create_audit_schema
from security.dag import PlanStep, validate_plan
from security.mandate import MandateService, create_security_schema
from security.policy import PolicyConfig, StepDecision, evaluate
from security.tokens import (
    Session,
    TokenVault,
    build_context,
    tokenize_user_text,
)

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42
SECRET = b"anban-executor-secret"
PII_RE = re.compile(r"\d{11,}")


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


def _tokenize(ctx: StepContext, text: str):
    return tokenize_user_text(text, ctx.bank, ctx.vault, ctx.session, ctx.now)


def _only(entries, kind: str):
    matches = [entry for entry in entries if entry.kind == kind]
    assert len(matches) == 1, f"expected one {kind}, got {len(matches)}"
    return matches[0]


def _validated(ctx: StepContext, *steps: PlanStep):
    return validate_plan(
        {"kind": "plan", "steps": [step.model_dump() for step in steps]},
        ctx.vault.known_tokens(ctx.session),
    )


def _decide(ctx: StepContext, *steps: PlanStep):
    validated = _validated(ctx, *steps)
    return validated, evaluate(
        validated,
        bank=ctx.bank,
        vault=ctx.vault,
        session=ctx.session,
        now=ctx.now,
        config=ctx.policy_config,
    )


def test_reminder_is_persisted_once_without_notification_claim():
    ctx, _, conn = _make_env()
    try:
        step = PlanStep(id="s1", tool="set_reminder", args={"note": "check bill"})
        _, policy = _decide(ctx, step)
        for _ in range(2):
            result = run_step(step, policy.steps[0], ctx, {}, idempotency_key="reminder-once")
            assert result["reminder"]["saved"] is True
            assert result["reminder"]["notifications_enabled"] is False
        rows = conn.execute("SELECT * FROM agent_reminders WHERE customer_id = ?", (ctx.bank.customer_id,)).fetchall()
        assert len(rows) == 1
        assert rows[0]["note"] == "check bill"
    finally:
        conn.close()


def _issue_yellow(
    ctx: StepContext,
    step: PlanStep,
    amount_fen: int,
    payee_ref: str = "",
):
    mandate = ctx.mandates.issue(
        customer_id=ctx.bank.customer_id,
        session_id=ctx.session.session_id,
        plan_id=ctx.session.plan_id,
        step_id=step.id,
        action=step.tool,
        payee_ref=payee_ref,
        amount_fen=amount_fen,
        tier="yellow",
        now=ctx.now,
    )
    return ctx.mandates.approve_yellow(mandate.id, ctx.bank.customer_id, ctx.now)


def _entries(ctx: StepContext):
    return ctx.audit.entries_for(ctx.bank.customer_id, ctx.session.plan_id)


def _assert_no_pii(payloads) -> None:
    for payload in payloads:
        assert not PII_RE.search(json.dumps(payload, ensure_ascii=False))


# -------------------------------------------------------------------- transfer


def test_yellow_transfer_runs_inside_cooling_window_and_undo_refunds(env) -> None:
    ctx, state, conn = env
    _, entries = _tokenize(ctx, "transfer 300 to 王建国")
    payee = _only(entries, "PAYEE")
    amount = _only(entries, "AMT")

    step = PlanStep(
        id="s1",
        tool="transfer",
        args={"payee": payee.label, "amount": amount.label},
    )
    _, decision = _decide(ctx, step)
    assert decision.steps[0].needs_mandate is True

    mandate = _issue_yellow(ctx, step, 30_000, payee.real_ref)
    before = ctx.bank.get_balance()

    result = run_step(
        step,
        decision.steps[0],
        ctx,
        {},
        mandate_id=mandate.id,
        idempotency_key="idem-300",
    )

    assert result["amount"] == 30_000
    assert "undo" not in result
    assert result["undo_ref"] == {
        "plan_id": ctx.session.plan_id,
        "step_id": "s1",
    }
    assert ctx.bank.get_balance() == before - 30_000
    row = conn.execute("SELECT * FROM transfers").fetchone()
    assert row["status"] == "pending"
    assert row["amount_fen"] == 30_000

    runs = [entry for entry in _entries(ctx) if entry.kind == AuditKind.RUN]
    assert runs
    assert runs[-1].payload["amount_fen"] == 30_000
    _assert_no_pii(entry.payload for entry in _entries(ctx))

    later = replace(ctx, now=NOW + timedelta(hours=1))
    undone = undo_step(result["undo_ref"]["plan_id"], "s1", later)

    assert undone is not None and undone["ok"] is True
    assert ctx.bank.get_balance() == before
    row = conn.execute("SELECT status FROM transfers").fetchone()
    assert row["status"] == "cancelled"
    assert any(entry.kind == AuditKind.UNDO for entry in _entries(later))


def test_missing_mandate_is_rejected_and_audited_as_veto(env) -> None:
    ctx, _, conn = env
    _, entries = _tokenize(ctx, "transfer 300 to 王建国")
    payee = _only(entries, "PAYEE")
    amount = _only(entries, "AMT")
    step = PlanStep(
        id="s1",
        tool="transfer",
        args={"payee": payee.label, "amount": amount.label},
    )
    _, decision = _decide(ctx, step)
    before = ctx.bank.get_balance()

    with pytest.raises(ExecError) as exc:
        run_step(step, decision.steps[0], ctx, {}, idempotency_key="no-mandate")

    assert exc.value.code == "mandate"
    assert exc.value.detail == "missing"
    assert ctx.bank.get_balance() == before
    assert conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0] == 0

    vetoes = [entry for entry in _entries(ctx) if entry.kind == AuditKind.VETO]
    assert vetoes and vetoes[-1].payload["error"] == "mandate"
    _assert_no_pii(entry.payload for entry in _entries(ctx))


def test_mandate_amount_mismatch_and_replay(env) -> None:
    ctx, _, conn = env
    _, three = _tokenize(ctx, "transfer 300 to 王建国")
    payee = _only(three, "PAYEE")
    amount_300 = _only(three, "AMT")

    _, four = _tokenize(ctx, "transfer 400 to 王建国")
    amount_400 = _only(four, "AMT")

    step_300 = PlanStep(
        id="s1",
        tool="transfer",
        args={"payee": payee.label, "amount": amount_300.label},
    )
    step_400 = PlanStep(
        id="s1",
        tool="transfer",
        args={"payee": payee.label, "amount": amount_400.label},
    )
    _, decision = _decide(ctx, step_300)
    mandate = _issue_yellow(ctx, step_300, 30_000, payee.real_ref)
    before = ctx.bank.get_balance()

    with pytest.raises(ExecError) as exc:
        run_step(
            step_400,
            decision.steps[0],
            ctx,
            {},
            mandate_id=mandate.id,
            idempotency_key="wrong-amount",
        )
    assert exc.value.code == "mandate"
    assert exc.value.detail == "mismatch"
    assert ctx.bank.get_balance() == before

    run_step(
        step_300,
        decision.steps[0],
        ctx,
        {},
        mandate_id=mandate.id,
        idempotency_key="right-amount",
    )
    assert ctx.bank.get_balance() == before - 30_000
    assert conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0] == 1

    with pytest.raises(ExecError) as replay:
        run_step(
            step_300,
            decision.steps[0],
            ctx,
            {},
            mandate_id=mandate.id,
            idempotency_key="right-amount",
        )
    assert replay.value.code == "mandate"
    assert replay.value.detail == "replayed"
    assert ctx.bank.get_balance() == before - 30_000


def test_recheck_escalation_requires_reapproval(env) -> None:
    ctx, state, _ = env
    _, entries = _tokenize(ctx, "transfer 300 to 王建国")
    payee = _only(entries, "PAYEE")
    amount = _only(entries, "AMT")
    step = PlanStep(
        id="s1",
        tool="transfer",
        args={"payee": payee.label, "amount": amount.label},
    )
    _, decision = _decide(ctx, step)
    assert decision.steps[0].tier == "yellow"

    # Same day, after planning: ¥900 leaves the account.
    contact = next(item for item in state.contacts if item.id == payee.real_ref)
    ctx.bank.transfer(contact.id, 90_000, "meanwhile", "between-plan-and-run", now=NOW)

    with pytest.raises(ExecError) as exc:
        run_step(step, decision.steps[0], ctx, {}, idempotency_key="escalated")

    assert exc.value.code == "reapproval_required"
    assert exc.value.detail == "red"


# ------------------------------------------------------- refs and divide_by


def test_step_ref_amount_from_list_transactions(env) -> None:
    ctx, _, _ = env
    _, entries = _tokenize(ctx, "transfer 420 to 王建国")
    payee = _only(entries, "PAYEE")
    amount = _only(entries, "AMT")

    list_step = PlanStep(
        id="s1",
        tool="list_transactions",
        args={"payee": payee.label},
    )
    list_decision = StepDecision(
        step_id="s1", tool="list_transactions", tier="green"
    )
    listed = run_step(list_step, list_decision, ctx, {})

    assert isinstance(listed["raw"], list)
    assert listed["raw"]
    assert listed["amount"] == abs(listed["raw"][0].amount_fen)

    transfer_step = PlanStep(
        id="s2",
        tool="transfer",
        args={"payee": payee.label, "amount": "$s1.amount"},
    )
    resolved = resolve_args(
        transfer_step, ctx.vault, ctx.session, {"s1": listed}
    )
    assert resolved["amount"] == listed["amount"]
    assert isinstance(resolved["amount"], int)


@pytest.mark.parametrize("birthday,expected", [
    (date(1996, 10, 29), date(2026, 10, 29)),
    (date(1996, 10, 1), date(2027, 10, 1)),
    (date(1996, 2, 29), date(2027, 2, 28)),
])
def test_contact_date_is_next_birthday_not_birth_year(env, monkeypatch, birthday, expected):
    ctx, _, _ = env
    _, entries = _tokenize(ctx, "李娜")
    payee = _only(entries, "PAYEE")
    monkeypatch.setattr(ctx.bank, "get_contact_date", lambda contact_id: birthday)
    step = PlanStep(id="s1", tool="get_contact_date", args={"payee": payee.label})
    decision = StepDecision(step_id="s1", tool="get_contact_date", tier="green")
    assert run_step(step, decision, ctx, {})["date"] == expected


def test_contact_without_a_date_fails_with_a_clear_reason(env) -> None:
    # Regression: an order for a friend with no birthday on file failed later
    # as an unresolved "$s1.date" ("The details have expired").
    ctx, _, conn = env
    _, entries = _tokenize(ctx, "transfer 420 to 王建国")
    payee = _only(entries, "PAYEE")
    conn.execute(
        "UPDATE contacts SET birthday = NULL WHERE customer_id = ? AND id = ?",
        (ctx.bank.customer_id, payee.real_ref),
    )
    conn.commit()
    step = PlanStep(id="s1", tool="get_contact_date", args={"payee": payee.label})
    decision = StepDecision(step_id="s1", tool="get_contact_date", tier="green")

    with pytest.raises(ExecError) as exc:
        run_step(step, decision, ctx, {})

    assert exc.value.code == "date_unknown"


def test_step_ref_amount_ignores_incoming_rows(env) -> None:
    # Regression: an incoming payment was abs()-ed into a transfer amount.
    ctx, _, conn = env
    account_id = conn.execute(
        "SELECT id FROM accounts WHERE customer_id = ? LIMIT 1",
        (ctx.bank.customer_id,),
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO transactions (customer_id, id, account_id, ts, amount_fen,"
        " counterparty, category, memo, channel, country, card_present)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (ctx.bank.customer_id, "txn_in", account_id, "2026-10-17T11:00:00",
         88_800, "工资", "income", "", "app", "CN", 0),
    )
    conn.commit()
    step = PlanStep(id="s1", tool="list_transactions", args={})
    decision = StepDecision(step_id="s1", tool="list_transactions", tier="green")
    listed = run_step(step, decision, ctx, {})

    outgoing = [row for row in listed["raw"] if row.amount_fen < 0]
    assert listed["raw"][0].amount_fen == 88_800  # newest row is incoming
    assert listed["amount"] == abs(outgoing[0].amount_fen)
    assert listed["payee"] == outgoing[0].counterparty


def test_divide_by_uses_integer_fen_and_rounds_half_up(env) -> None:
    ctx, _, _ = env
    _, entries = _tokenize(ctx, "transfer 420 to 王建国")
    payee = _only(entries, "PAYEE")
    amount = _only(entries, "AMT")
    assert amount.real_ref == 42_000

    step = PlanStep(
        id="s1",
        tool="transfer",
        args={"payee": payee.label, "amount": amount.label, "divide_by": 3},
    )
    resolved = resolve_args(step, ctx.vault, ctx.session, {})

    assert resolved["amount"] == 14_000


# ---------------------------------------------------------------- merchant


def test_merchant_order_price_delivery_and_cancel_undo(env) -> None:
    ctx, _, conn = env
    build_context(ctx.bank, ctx.vault, ctx.session, ctx.now)
    entries = ctx.vault.entries(ctx.session)

    merchant = next(
        entry
        for entry in entries
        if entry.kind == "MERCHANT" and entry.real_ref == "花店"
    )
    birthday = next(
        entry
        for entry in entries
        if entry.kind == "DATE" and entry.real_ref == TODAY + timedelta(days=12)
    )

    step = PlanStep(
        id="s1",
        tool="merchant_order",
        args={
            "merchant": merchant.label,
            "item": "bouquet_m",
            "deliver": {"date": birthday.label, "offset_days": -2},
        },
    )
    _, decision = _decide(ctx, step)
    assert decision.steps[0].amount_fen == price_of("花店", "bouquet_m")
    assert decision.steps[0].needs_mandate is True

    mandate = _issue_yellow(ctx, step, 29_900)
    before = ctx.bank.get_balance()

    result = run_step(
        step,
        decision.steps[0],
        ctx,
        {},
        mandate_id=mandate.id,
        idempotency_key="order-1",
    )

    row = conn.execute("SELECT * FROM merchant_orders").fetchone()
    assert row["merchant"] == "花店"
    assert row["item"] == "bouquet_m"
    assert row["amount_fen"] == 29_900
    assert date.fromisoformat(row["deliver_on"]) == TODAY + timedelta(days=10)
    assert result["date"] == TODAY + timedelta(days=10)
    assert "undo" not in result
    assert result["undo_ref"] == {
        "plan_id": ctx.session.plan_id,
        "step_id": "s1",
    }
    assert ctx.bank.get_balance() == before - 29_900

    undone = undo_step(ctx.session.plan_id, "s1", ctx)

    assert undone is not None and undone["ok"] is True
    row = conn.execute("SELECT status FROM merchant_orders").fetchone()
    assert row["status"] == "cancelled"
    assert ctx.bank.get_balance() == before


# --------------------------------------------------------------- investments


def test_subscribe_product_step_debits_the_bank(env) -> None:
    ctx, _, conn = env
    build_context(ctx.bank, ctx.vault, ctx.session, ctx.now)
    product = next(
        entry
        for entry in ctx.vault.entries(ctx.session)
        if entry.kind == "PROD" and entry.real_ref == "p2"
    )
    _, entries = _tokenize(ctx, "transfer 200")
    amount = _only(entries, "AMT")
    step = PlanStep(
        id="s1",
        tool="subscribe_product",
        args={"product": product.label, "amount": amount.label},
    )
    _, decision = _decide(ctx, step)
    assert decision.steps[0].tier == "red"
    assert decision.steps[0].needs_mfa is True

    mandate = ctx.mandates.issue(
        customer_id=ctx.bank.customer_id,
        session_id=ctx.session.session_id,
        plan_id=ctx.session.plan_id,
        step_id="s1",
        action="subscribe_product",
        payee_ref="",
        amount_fen=20_000,
        tier="red",
        now=ctx.now,
    )
    code = ctx.mandates.start_mfa(mandate.id, ctx.bank.customer_id, ctx.now)
    ctx.mandates.verify_mfa(
        mandate.id,
        ctx.bank.customer_id,
        otp=code,
        retyped_amount_fen=20_000,
        face_ok=True,
        now=ctx.now,
    )

    before = ctx.bank.get_balance()
    result = run_step(
        step,
        decision.steps[0],
        ctx,
        {},
        mandate_id=mandate.id,
        idempotency_key="subscribe-1",
    )

    assert result["amount"] == 20_000
    assert result["holding"] == {
        "product_id": "p2",
        "amount_fen": 20_000,
        "status": "subscribed",
    }
    assert ctx.bank.get_balance() == before - 20_000
    portfolio = {item.product_id: item for item in ctx.bank.get_portfolio()}
    assert portfolio["p2"].principal_fen == 20_000
    assert portfolio["p2"].value_fen == 20_000
    assert conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE category = 'investment'"
    ).fetchone()[0] == 1


# ------------------------------------------------------------------- unfreeze


def test_unfreeze_undo_requires_a_fresh_mandate(env) -> None:
    ctx, _, _ = env
    build_context(ctx.bank, ctx.vault, ctx.session, ctx.now)
    card = next(
        entry for entry in ctx.vault.entries(ctx.session) if entry.kind == "CARD"
    )

    step = PlanStep(id="s1", tool="freeze_card", args={"card": card.label})
    _, decision = _decide(ctx, step)
    assert decision.steps[0].needs_mandate is True

    mandate = _issue_yellow(ctx, step, 0)
    result = run_step(
        step,
        decision.steps[0],
        ctx,
        {},
        mandate_id=mandate.id,
        idempotency_key="freeze-1",
    )
    assert result["undo_ref"] == {
        "plan_id": ctx.session.plan_id,
        "step_id": "s1",
    }

    with pytest.raises(ExecError) as exc:
        undo_step(ctx.session.plan_id, "s1", ctx)
    assert exc.value.code == "mandate_required"

    fresh = ctx.mandates.issue(
        customer_id=ctx.bank.customer_id,
        session_id=ctx.session.session_id,
        plan_id=ctx.session.plan_id,
        step_id="s1",
        action="unfreeze_card",
        payee_ref="",
        amount_fen=0,
        tier="yellow",
        now=ctx.now,
    )
    ctx.mandates.approve_yellow(fresh.id, ctx.bank.customer_id, ctx.now)

    undone = undo_step(
        ctx.session.plan_id, "s1", ctx, mandate_id=fresh.id
    )

    assert undone is not None and undone["ok"] is True
    card_id = card.real_ref
    row = ctx.bank.conn.execute(
        "SELECT status FROM cards WHERE customer_id = ? AND id = ?",
        (ctx.bank.customer_id, card_id),
    ).fetchone()
    assert row["status"] == "active"


# -------------------------------------------------------------- dispatchability


def test_every_non_control_tool_is_dispatchable(env) -> None:
    ctx, _, _ = env

    for name, spec in REGISTRY.items():
        if spec.kind == "control":
            continue
        decision = StepDecision(
            step_id="s1",
            tool=name,
            tier="green" if spec.base_tier == "none" else spec.base_tier,
        )
        step = PlanStep(id="s1", tool=name, args={})
        try:
            run_step(step, decision, ctx, {})
        except ExecError as exc:
            assert exc.code != "unsupported", name


# --------------------------------------------------------- mandate run-time gates


def test_red_step_with_yellow_mandate_is_tier_too_low(env) -> None:
    ctx, _, conn = env
    _, entries = _tokenize(ctx, "transfer 300 to 王建国")
    payee = _only(entries, "PAYEE")
    amount = _only(entries, "AMT")
    step = PlanStep(
        id="s1",
        tool="transfer",
        args={"payee": payee.label, "amount": amount.label},
    )
    decision = StepDecision(
        step_id="s1",
        tool="transfer",
        tier="red",
        needs_mandate=True,
        needs_mfa=True,
    )
    mandate = _issue_yellow(ctx, step, 30_000, payee.real_ref)
    before = ctx.bank.get_balance()

    with pytest.raises(ExecError) as exc:
        run_step(
            step,
            decision,
            ctx,
            {},
            mandate_id=mandate.id,
            idempotency_key="red-with-yellow",
        )

    assert exc.value.code == "mandate"
    assert exc.value.detail == "tier_too_low"
    assert ctx.bank.get_balance() == before
    assert conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0] == 0
    assert conn.execute(
        "SELECT status FROM security_mandates WHERE id = ?", (mandate.id,)
    ).fetchone()["status"] == "approved"


def test_elder_red_with_mfa_but_no_co_approval_is_rejected() -> None:
    ctx, state, conn = _make_env("retiree")
    try:
        contact = state.contacts[0]
        _, entries = _tokenize(ctx, f"transfer 300 to {contact.name}")
        payee = _only(entries, "PAYEE")
        amount = _only(entries, "AMT")
        step = PlanStep(
            id="s1",
            tool="transfer",
            args={"payee": payee.label, "amount": amount.label},
        )
        decision = StepDecision(
            step_id="s1",
            tool="transfer",
            tier="red",
            needs_mandate=True,
            needs_mfa=True,
            needs_co_approval=True,
        )
        mandate = ctx.mandates.issue(
            customer_id=ctx.bank.customer_id,
            session_id=ctx.session.session_id,
            plan_id=ctx.session.plan_id,
            step_id="s1",
            action="transfer",
            payee_ref=payee.real_ref,
            amount_fen=30_000,
            tier="red",
            now=ctx.now,
            co_approval_required=True,
        )
        code = ctx.mandates.start_mfa(
            mandate.id, ctx.bank.customer_id, ctx.now
        )
        verified = ctx.mandates.verify_mfa(
            mandate.id,
            ctx.bank.customer_id,
            otp=code,
            retyped_amount_fen=30_000,
            face_ok=True,
            now=ctx.now,
        )
        assert verified.mfa_verified is True
        assert verified.co_approved_by is None
        before = ctx.bank.get_balance()

        with pytest.raises(ExecError) as exc:
            run_step(
                step,
                decision,
                ctx,
                {},
                mandate_id=mandate.id,
                idempotency_key="elder-no-co-approval",
            )

        assert exc.value.code == "mandate"
        assert exc.value.detail == "co_approval_missing"
        assert ctx.bank.get_balance() == before
        assert conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0] == 0
    finally:
        conn.close()


def test_scam_flags_veto_before_mandate_and_money_does_not_move(env) -> None:
    ctx, _, conn = env
    _, entries = _tokenize(ctx, "transfer 300 to 王建国")
    payee = _only(entries, "PAYEE")
    amount = _only(entries, "AMT")
    step = PlanStep(
        id="s1",
        tool="transfer",
        args={"payee": payee.label, "amount": amount.label},
    )
    _, decision = _decide(ctx, step)
    mandate = _issue_yellow(ctx, step, 30_000, payee.real_ref)
    before = ctx.bank.get_balance()
    flagged = replace(ctx, scam_flags=frozenset({"police_claim"}))

    with pytest.raises(ExecError) as exc:
        run_step(
            step,
            decision.steps[0],
            flagged,
            {},
            mandate_id=mandate.id,
            idempotency_key="scam-veto",
        )

    assert exc.value.code == "policy_veto"
    assert "R_SCAM" in exc.value.detail
    assert ctx.bank.get_balance() == before
    assert conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0] == 0
    vetoes = [entry for entry in _entries(flagged) if entry.kind == AuditKind.VETO]
    assert vetoes and vetoes[-1].payload["error"] == "policy_veto"


# ---------------------------------------------------------------- server undo


def test_tampered_exec_undo_row_is_rejected(env) -> None:
    ctx, _, conn = env
    _, entries = _tokenize(ctx, "transfer 300 to 王建国")
    payee = _only(entries, "PAYEE")
    amount = _only(entries, "AMT")
    step = PlanStep(
        id="s1",
        tool="transfer",
        args={"payee": payee.label, "amount": amount.label},
    )
    _, decision = _decide(ctx, step)
    mandate = _issue_yellow(ctx, step, 30_000, payee.real_ref)
    after_plan = ctx.bank.get_balance()
    run_step(
        step,
        decision.steps[0],
        ctx,
        {},
        mandate_id=mandate.id,
        idempotency_key="tamper-undo",
    )

    conn.execute(
        "UPDATE exec_undo SET args_json = ?"
        " WHERE customer_id = ? AND plan_id = ? AND step_id = ?",
        ('{"transfer_id":"tr_999999"}', ctx.bank.customer_id, ctx.session.plan_id, "s1"),
    )
    conn.commit()

    with pytest.raises(ExecError) as exc:
        undo_step(ctx.session.plan_id, "s1", ctx)

    assert exc.value.code == "undo"
    assert exc.value.detail == "bad_signature"
    assert ctx.bank.get_balance() == after_plan - 30_000
    assert conn.execute("SELECT status FROM transfers").fetchone()["status"] == "pending"


def test_undo_twice_is_rejected(env) -> None:
    ctx, _, conn = env
    _, entries = _tokenize(ctx, "transfer 300 to 王建国")
    payee = _only(entries, "PAYEE")
    amount = _only(entries, "AMT")
    step = PlanStep(
        id="s1",
        tool="transfer",
        args={"payee": payee.label, "amount": amount.label},
    )
    _, decision = _decide(ctx, step)
    mandate = _issue_yellow(ctx, step, 30_000, payee.real_ref)
    before = ctx.bank.get_balance()
    run_step(
        step,
        decision.steps[0],
        ctx,
        {},
        mandate_id=mandate.id,
        idempotency_key="undo-twice",
    )

    first = undo_step(ctx.session.plan_id, "s1", ctx)
    assert first is not None and first["ok"] is True
    assert ctx.bank.get_balance() == before

    with pytest.raises(ExecError) as exc:
        undo_step(ctx.session.plan_id, "s1", ctx)

    assert exc.value.code == "undo"
    assert exc.value.detail == "replayed"
    assert ctx.bank.get_balance() == before


# ----------------------------------------------------------- token edge cases


def test_find_contact_with_payee_token_returns_exact_contact(env) -> None:
    ctx, _, _ = env
    build_context(ctx.bank, ctx.vault, ctx.session, ctx.now)
    payees = [
        entry for entry in ctx.vault.entries(ctx.session) if entry.kind == "PAYEE"
    ]
    assert len(payees) >= 2
    target = payees[1]

    step = PlanStep(
        id="s1",
        tool="find_contact",
        args={"query": target.label},
    )
    decision = StepDecision(step_id="s1", tool="find_contact", tier="green")
    result = run_step(step, decision, ctx, {})

    assert result["payee"] == target.real_ref
    assert [contact.id for contact in result["raw"]] == [target.real_ref]


def test_write_payee_step_ref_is_rejected(env) -> None:
    ctx, _, conn = env
    _, entries = _tokenize(ctx, "transfer 300 to 王建国")
    payee = _only(entries, "PAYEE")
    amount = _only(entries, "AMT")
    step = PlanStep(
        id="s2",
        tool="transfer",
        args={"payee": "$s1.payee", "amount": amount.label},
    )
    decision = StepDecision(
        step_id="s2", tool="transfer", tier="yellow", needs_mandate=True
    )

    with pytest.raises(ExecError) as exc:
        run_step(step, decision, ctx, {"s1": {"payee": payee.real_ref}})

    assert exc.value.code == "token"
    assert exc.value.detail == "payee_ref_not_allowed"
    assert conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0] == 0


def test_oversized_amount_is_invalid_amount_not_a_crash(env) -> None:
    ctx, _, conn = env
    build_context(ctx.bank, ctx.vault, ctx.session, ctx.now)
    card = next(
        entry for entry in ctx.vault.entries(ctx.session) if entry.kind == "CARD"
    )
    before = ctx.bank.get_balance()
    step = PlanStep(
        id="s1",
        tool="restore_card_limit",
        args={
            "card_id": card.real_ref,
            "previous_limit_fen": 9 * 10**21,
        },
    )
    decision = StepDecision(step_id="s1", tool="restore_card_limit", tier="green")

    with pytest.raises(ExecError) as exc:
        run_step(step, decision, ctx, {}, idempotency_key="huge-limit")

    assert exc.value.code == "bank"
    assert exc.value.detail == "invalid_amount"
    assert ctx.bank.get_balance() == before
    vetoes = [entry for entry in _entries(ctx) if entry.kind == AuditKind.VETO]
    assert vetoes and vetoes[-1].payload["error"] == "bank"


def test_card_step_without_a_card_uses_the_only_card(env) -> None:
    # Regression: "freeze my card" had no card token and failed after approval.
    from agent.executor import _card_id

    ctx, _, _ = env
    cards = [card for card in ctx.bank.list_cards() if card.kind != "virtual"]
    assert len(cards) == 1

    assert _card_id(ctx.bank, {}) == cards[0].id
    assert _card_id(ctx.bank, {"card": "explicit"}) == "explicit"


def test_list_transactions_direction_filter(env) -> None:
    ctx, _, _ = env
    decision = StepDecision(step_id="s1", tool="list_transactions", tier="green")
    incoming = run_step(PlanStep(id="s1", tool="list_transactions", args={"direction": "in"}), decision, ctx, {})
    outgoing = run_step(PlanStep(id="s1", tool="list_transactions", args={"direction": "out"}), decision, ctx, {})

    assert incoming["raw"] and all(row.amount_fen > 0 for row in incoming["raw"])
    assert outgoing["raw"] and all(row.amount_fen < 0 for row in outgoing["raw"])
