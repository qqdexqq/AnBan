"""Unit tests for the scenario 4 card intent resolver."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from agent.intents import IntentContext
from agent.intents.cards import resolve
from bank import db
from bank.api import Bank
from bank.seed import generate
from security.tokens import Session, TokenVault, tokenize_user_text

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42
SECRET = b"anban-test-secret"


@pytest.fixture
def env():
    state = generate("young_pro", SEED, TODAY)
    conn = db.connect(":memory:")
    db.create_schema(conn)
    db.load_state(conn, state)
    bank = Bank(conn, state.customer.id)
    vault = TokenVault(SECRET)
    session = Session(
        session_id="session", plan_id="plan", customer_id=state.customer.id
    )
    try:
        yield bank, conn, vault, session, state
    finally:
        conn.close()


def _context(
    bank: Bank,
    conn,
    vault: TokenVault,
    session: Session,
    raw: str,
    lang: str = "en",
    tokenized: str | None = None,
):
    if tokenized is None:
        tokenized, _ = tokenize_user_text(raw, bank, vault, session, NOW)
    return IntentContext(
        raw_text=raw,
        tokenized_text=f"User: {tokenized}",
        conn=conn,
        secret_key=SECRET,
        customer_id=session.customer_id,
        session_id=session.session_id,
        now=NOW,
        lang=lang,
        vault=vault,
        plan_id=session.plan_id,
    )


def _tools(plan) -> list[str]:
    return [step["tool"] for step in plan["steps"]]


@pytest.mark.parametrize(
    ("raw", "lang"),
    [
        ("我的卡丢了，帮我挂失", "zh"),
        ("I lost my card, report it lost", "en"),
    ],
)
def test_loss_freezes_then_reports(env, raw, lang) -> None:
    bank, conn, vault, session, _ = env
    plan = resolve(_context(bank, conn, vault, session, raw, lang))

    assert plan["kind"] == "plan"
    assert _tools(plan) == ["freeze_card", "report_card_lost"]
    assert plan["steps"][1]["after"] == ["s1"]
    assert plan["steps"][0]["args"]["card"].startswith("<CARD_")


def test_loss_on_frozen_card_only_reports(env) -> None:
    bank, conn, vault, session, _ = env
    bank.freeze_card(bank.list_cards()[0].id)

    plan = resolve(_context(bank, conn, vault, session, "卡丢了，挂失", "zh"))

    assert _tools(plan) == ["report_card_lost"]


@pytest.mark.parametrize(
    ("raw", "lang"),
    [
        ("我的卡丢了，先冻结", "zh"),
        ("我的卡丢了，只冻结", "zh"),
        ("我的卡丢了，先冻上", "zh"),
        ("我的卡丢了，冻结就行", "zh"),
        ("我的卡丢了，冻一下", "zh"),
        ("I lost my card, freeze it first", "en"),
        ("I lost my card, just freeze", "en"),
        ("I lost my card, only freeze", "en"),
        ("I lost my card, freeze for now", "en"),
    ],
)
def test_loss_freeze_only_plans_only_freeze(env, raw, lang) -> None:
    bank, conn, vault, session, _ = env

    plan = resolve(_context(bank, conn, vault, session, raw, lang))

    assert plan["kind"] == "plan"
    assert _tools(plan) == ["freeze_card"]
    assert plan["steps"][0]["args"]["card"].startswith("<CARD_")


def test_loss_freeze_first_then_report_keeps_both(env) -> None:
    bank, conn, vault, session, _ = env

    plan = resolve(
        _context(bank, conn, vault, session, "我的卡丢了，先冻结再挂失", "zh")
    )

    assert _tools(plan) == ["freeze_card", "report_card_lost"]


@pytest.mark.parametrize(
    "raw",
    [
        "冻结我的卡，还能用吗",
        "冻结我的卡，还可以吗",
        "冻结我的卡，还剩几天",
    ],
)
def test_hai_words_are_not_read_as_a_payment(env, raw) -> None:
    bank, conn, vault, session, _ = env

    plan = resolve(_context(bank, conn, vault, session, raw, "zh"))

    assert _tools(plan) == ["freeze_card"]


def test_virtual_card_and_spending_question_reaches_the_model(env) -> None:
    from agent.intents import RULES

    bank, conn, vault, session, _ = env
    raw = "下周去成都，开张2000额度的虚拟卡，再看看这个月还能花多少"
    ctx = _context(bank, conn, vault, session, raw, "zh")

    for rule in RULES:
        assert rule(ctx) is None, rule.__module__


def test_virtual_card_without_amount_uses_safe_default(env) -> None:
    bank, conn, vault, session, _ = env
    plan = resolve(_context(bank, conn, vault, session, "开一张虚拟卡", "zh"))

    assert _tools(plan) == ["create_virtual_card"]
    amount = vault.resolve(plan["steps"][0]["args"]["amount"], session)
    assert amount.real_ref == 20_000


def test_virtual_card_amount_is_not_a_limit_change(env) -> None:
    bank, conn, vault, session, _ = env
    plan = resolve(_context(bank, conn, vault, session, "虚拟卡限额2000", "zh"))

    assert _tools(plan) == ["create_virtual_card"]
    amount = vault.resolve(plan["steps"][0]["args"]["amount"], session)
    assert amount.real_ref == 200_000


@pytest.mark.parametrize(
    ("raw", "lang", "fen"),
    [
        ("set my card limit to 300", "en", 30_000),
        ("把限额改成300", "zh", 30_000),
    ],
)
def test_set_limit_uses_user_amount(env, raw, lang, fen) -> None:
    bank, conn, vault, session, _ = env
    plan = resolve(_context(bank, conn, vault, session, raw, lang))

    assert _tools(plan) == ["set_card_limit"]
    args = plan["steps"][0]["args"]
    assert args["card"].startswith("<CARD_")
    assert vault.resolve(args["amount"], session).real_ref == fen


def test_set_limit_without_amount_clarifies(env) -> None:
    bank, conn, vault, session, _ = env
    plan = resolve(_context(bank, conn, vault, session, "change my card limit"))

    assert plan["kind"] == "clarify"
    assert plan["options"] == []


def test_context_amount_is_not_used_for_a_limit(env) -> None:
    bank, conn, vault, session, _ = env
    context_amount = vault.mint(session, "AMT", 50_000, {}, "bank_context")
    tokenized = f"Context amount {context_amount.label}\n\nUser: change my card limit"

    plan = resolve(
        _context(
            bank, conn, vault, session, "change my card limit",
            tokenized=tokenized,
        )
    )

    assert plan["kind"] == "clarify"


def test_multiple_physical_cards_are_not_guessed(env) -> None:
    bank, conn, vault, session, _ = env
    account_id = bank.list_cards()[0].account_id
    conn.execute(
        "INSERT INTO cards "
        "(customer_id, id, kind, last4, status, online_limit_fen, account_id) "
        "VALUES (?, 'card_physical_2', 'physical', '2222', 'active', 100000, ?)",
        (session.customer_id, account_id),
    )
    conn.commit()

    assert resolve(_context(bank, conn, vault, session, "freeze my card")) is None


def test_mixed_transfer_is_left_to_the_planner(env) -> None:
    bank, conn, vault, session, _ = env

    plan = resolve(
        _context(
            bank, conn, vault, session,
            "freeze my card and transfer 500 to 张伟",
        )
    )

    assert plan is None


def _create_limit_history(conn) -> None:
    conn.execute(
        "CREATE TABLE card_limit_history ("
        "customer_id TEXT NOT NULL, card_id TEXT NOT NULL, "
        "previous_limit_fen INTEGER NOT NULL)"
    )
    conn.commit()


def test_restore_higher_limit_uses_set_card_limit(env) -> None:
    bank, conn, vault, session, _ = env
    card = bank.list_cards()[0]
    original = card.online_limit_fen
    bank.set_card_limit(card.id, 100_000)

    plan = resolve(_context(bank, conn, vault, session, "restore my card limit"))

    assert _tools(plan) == ["set_card_limit"]
    args = plan["steps"][0]["args"]
    assert vault.resolve(args["amount"], session).real_ref == original


def test_restore_lower_limit_uses_green_restore(env) -> None:
    bank, conn, vault, session, _ = env
    card = bank.list_cards()[0]
    bank.set_card_limit(card.id, 300_000)
    bank.set_card_limit(card.id, 500_000)

    plan = resolve(_context(bank, conn, vault, session, "restore my card limit"))

    assert plan["steps"] == [
        {"id": "s1", "tool": "restore_card_limit", "args": {}, "after": []}
    ]


def test_restore_without_history_uses_green_restore(env) -> None:
    bank, conn, vault, session, _ = env

    plan = resolve(_context(bank, conn, vault, session, "restore my card limit"))

    assert _tools(plan) == ["restore_card_limit"]


def test_work_online_again_changes_setting_without_implicit_unfreeze(env) -> None:
    bank, conn, vault, session, _ = env
    bank.freeze_card(bank.list_cards()[0].id)

    plan = resolve(
        _context(bank, conn, vault, session, "make my card work online again")
    )

    assert _tools(plan) == ["set_card_control"]
    assert plan["steps"][0]["args"]["control"] == "online"
    assert plan["steps"][0]["args"]["enabled"] is True


@pytest.mark.parametrize(
    ("raw", "enabled"),
    [
        ("禁止网购", False),
        ("unblock online shopping category", True),
    ],
)
def test_online_category_maps_to_card_control(env, raw, enabled) -> None:
    bank, conn, vault, session, _ = env
    plan = resolve(_context(bank, conn, vault, session, raw))

    assert _tools(plan) == ["set_card_control"]
    assert plan["steps"][0]["args"]["control"] == "online"
    assert plan["steps"][0]["args"]["enabled"] is enabled


@pytest.mark.parametrize("raw", ["show my cards", "我的卡", "卡现在什么状态"])
def test_list_cards(env, raw) -> None:
    bank, conn, vault, session, _ = env
    plan = resolve(_context(bank, conn, vault, session, raw))

    assert plan["steps"] == [
        {"id": "s1", "tool": "list_cards", "args": {}, "after": []}
    ]


# Acceptance tests exercise the real policy, mandates, bank, graph and replies.
from evals.runner import make_env as _flow_env, _answers as _flow_answers
from security.mandate import MandateError as _MandateError
from security.policy import PAYMENT_LOCK_PROTECTIVE_ACTIONS as _LOCK_PROTECTION


class _NoCardPlanner:
    calls = 0
    def plan(self, *args):
        raise AssertionError("Card acceptance case unexpectedly reached the planner")


@pytest.fixture
def cardflow():
    value = _flow_env("young_pro", _NoCardPlanner())
    yield value
    value.conn.close()


def _lock_after_three_wrong_codes(value):
    now = value.clock["now"]
    service = value.services.mandates
    mandate = service.issue(customer_id=value.customer_id, session_id="lock-test", plan_id="lock-test:p1",
        step_id="s1", action="transfer", payee_ref="test-payee", amount_fen=150_000, tier="red", now=now)
    code = service.start_mfa(mandate.id, value.customer_id, now)
    wrong = "000000" if code != "000000" else "111111"
    for _ in range(3):
        with pytest.raises(_MandateError):
            service.verify_mfa(mandate.id, value.customer_id, otp=wrong, retyped_amount_fen=150_000, face_ok=True, now=now)
    assert service.is_payments_locked(value.customer_id, now)


def _card_turn(value, text, sid=None, lang="en"):
    return value.agent.start(value.customer_id, text, session_id=sid, lang=lang)


def _card_approve(value, result):
    for _ in range(4):
        if result.status != "needs_approval":
            return result
        result = value.agent.resume(result.session_id, value.customer_id, _flow_answers(result.approvals, "all"))
    raise AssertionError("Approval did not finish")


@pytest.mark.parametrize("text", ["冻结我的卡", "freeze my card"])
def test_locked_customer_can_freeze_without_unlocking_payments(cardflow, text):
    _lock_after_three_wrong_codes(cardflow)
    result = _card_turn(cardflow, text)
    assert result.status == "needs_approval", result.reply
    assert result.approvals[0]["tier"] == "yellow"
    result = _card_approve(cardflow, result)
    assert result.status == "done", result.reply
    card = cardflow.services.bank_for(cardflow.customer_id).list_cards()[0]
    assert card.status == "frozen" and card.last4 in result.reply
    assert cardflow.services.mandates.is_payments_locked(cardflow.customer_id, cardflow.clock["now"])
    assert cardflow.services.audit.verify().ok


@pytest.mark.parametrize("text", ["我的卡丢了，帮我挂失", "我的卡不见了", "卡被偷了", "I lost my card"])
def test_loss_during_lock_still_requires_fresh_red_verification(cardflow, text):
    _lock_after_three_wrong_codes(cardflow)
    result = _card_turn(cardflow, text)
    assert result.status == "needs_approval", result.reply
    loss = next(item for item in result.approvals if item["tool"] == "report_card_lost")
    assert loss["tier"] == "red" and loss["needs_mfa"]
    bank = cardflow.services.bank_for(cardflow.customer_id)
    assert bank.list_cards()[0].status == "active"
    result = _card_approve(cardflow, result)
    assert result.status == "done", result.reply
    assert bank.list_cards()[0].status == "lost"
    assert cardflow.services.mandates.is_payments_locked(cardflow.customer_id, cardflow.clock["now"])


def test_wrong_code_on_loss_during_lock_still_freezes_the_card(cardflow):
    # Regression: one typo on the red loss step must not leave a stolen card active.
    _lock_after_three_wrong_codes(cardflow)
    service = cardflow.services.mandates
    locked_until = cardflow.conn.execute(
        "SELECT locked_until FROM security_lockouts WHERE customer_id = ?", (cardflow.customer_id,)
    ).fetchone()[0]
    result = _card_turn(cardflow, "我的卡丢了，帮我挂失")
    assert result.status == "needs_approval", result.reply
    answers = _flow_answers(result.approvals, "all")
    for approval in result.approvals:
        if approval["tool"] == "report_card_lost":
            good = answers[approval["step_id"]].get("otp") or ""
            answers[approval["step_id"]]["otp"] = "000000" if good != "000000" else "111111"
    for _ in range(3):
        result = cardflow.agent.resume(result.session_id, cardflow.customer_id, answers)
        if result.status != "needs_approval":
            break
        answers = {a["step_id"]: v for a in result.approvals for k, v in answers.items() if k == a["step_id"]}
    card = cardflow.services.bank_for(cardflow.customer_id).list_cards()[0]
    assert card.status == "frozen", (card.status, result.reply)
    assert service.is_payments_locked(cardflow.customer_id, cardflow.clock["now"])
    # The typo did not push the lock further out.
    assert cardflow.conn.execute(
        "SELECT locked_until FROM security_lockouts WHERE customer_id = ?", (cardflow.customer_id,)
    ).fetchone()[0] == locked_until


@pytest.mark.parametrize("text", ["unfreeze my card", "set my card limit to 6000", "apply for a virtual card", "unblock online shopping category", "restore my original card limit"])
def test_lock_still_blocks_new_cards_and_looser_controls(cardflow, text):
    bank = cardflow.services.bank_for(cardflow.customer_id)
    bank.freeze_card(bank.list_cards()[0].id)
    before = [(c.id, c.status, c.online_limit_fen) for c in bank.list_cards()]
    _lock_after_three_wrong_codes(cardflow)
    result = _card_turn(cardflow, text)
    assert result.status == "vetoed", result.reply
    assert "24" in result.reply and "human" in result.reply
    assert [(c.id, c.status, c.online_limit_fen) for c in bank.list_cards()] == before


def test_protective_mandate_cannot_be_used_as_payment(cardflow):
    _lock_after_three_wrong_codes(cardflow)
    service = cardflow.services.mandates
    now = cardflow.clock["now"]
    mandate = service.issue(customer_id=cardflow.customer_id, session_id="protected", plan_id="protected:p1", step_id="s1",
        action="report_card_lost", payee_ref="card", amount_fen=0, tier="red", now=now)
    code = service.start_mfa(mandate.id, cardflow.customer_id, now)
    with pytest.raises(_MandateError):
        service.verify_and_consume(mandate.id, customer_id=cardflow.customer_id, session_id="protected", plan_id="protected:p1", step_id="s1", action="report_card_lost", payee_ref="card", amount_fen=0, now=now)
    service.verify_mfa(mandate.id, cardflow.customer_id, otp=code, retyped_amount_fen=0, face_ok=True, now=now)
    with pytest.raises(_MandateError):
        service.verify_and_consume(mandate.id, customer_id=cardflow.customer_id, session_id="protected", plan_id="protected:p1", step_id="s1", action="transfer", payee_ref="card", amount_fen=0, now=now)
    service.verify_and_consume(mandate.id, customer_id=cardflow.customer_id, session_id="protected", plan_id="protected:p1", step_id="s1", action="report_card_lost", payee_ref="card", amount_fen=0, now=now)
    with pytest.raises(_MandateError):
        service.verify_and_consume(mandate.id, customer_id=cardflow.customer_id, session_id="protected", plan_id="protected:p1", step_id="s1", action="report_card_lost", payee_ref="card", amount_fen=0, now=now)


def test_lock_bypass_does_not_trust_tampered_mandate_action(cardflow):
    service = cardflow.services.mandates
    now = cardflow.clock["now"]
    mandate = service.issue(customer_id=cardflow.customer_id, session_id="tamper", plan_id="tamper:p1", step_id="s1", action="transfer", payee_ref="person", amount_fen=10_000, tier="yellow", now=now)
    _lock_after_three_wrong_codes(cardflow)
    cardflow.conn.execute("UPDATE security_mandates SET action='report_card_lost', status='issued' WHERE id=?", (mandate.id,))
    cardflow.conn.commit()
    with pytest.raises(_MandateError) as error:
        service.approve_yellow(mandate.id, cardflow.customer_id, now)
    assert error.value.code == "bad_signature"
    assert _LOCK_PROTECTION == {"freeze_card", "report_card_lost"}


@pytest.mark.parametrize("text", ["帮我开一张虚拟卡", "我想申请一张虚拟银行卡", "apply for a virtual card"])
def test_default_virtual_card_is_yellow_and_confirmed(cardflow, text):
    result = _card_turn(cardflow, text)
    assert result.status == "needs_approval", result.reply
    approval = result.approvals[0]
    assert approval["tool"] == "create_virtual_card" and approval["tier"] == "yellow"
    assert approval["amount_fen"] == 20_000
    assert "default" in result.reply and "200.00" in result.reply
    result = _card_approve(cardflow, result)
    assert result.status == "done", result.reply
    virtual = next(c for c in cardflow.services.bank_for(cardflow.customer_id).list_cards() if c.kind == "virtual")
    assert virtual.online_limit_fen == 20_000 and virtual.last4 in result.reply


def test_new_online_shopping_card_does_not_change_existing_limit(cardflow):
    bank = cardflow.services.bank_for(cardflow.customer_id)
    physical = next(c for c in bank.list_cards() if c.kind == "physical")
    original = physical.online_limit_fen
    result = _card_turn(cardflow, "新开一张网购用的卡，每月限额2000")
    assert result.status == "needs_approval", result.reply
    assert result.approvals[0]["tool"] == "create_virtual_card"
    result = _card_approve(cardflow, result)
    assert result.status == "done"
    assert next(c for c in bank.list_cards() if c.id == physical.id).online_limit_fen == original
    assert any(c.kind == "virtual" and c.online_limit_fen == 200_000 for c in bank.list_cards())


def test_elder_card_cap_and_loss_guardian_are_preserved():
    value = _flow_env("retiree", _NoCardPlanner())
    try:
        result = _card_turn(value, "开一张虚拟卡，限额600")
        assert result.approvals[0]["tier"] == "red" and result.approvals[0]["needs_co_approval"]
        result = _card_turn(value, "开一张虚拟卡，限额500")
        assert result.approvals[0]["tier"] == "yellow"
        _lock_after_three_wrong_codes(value)
        result = _card_turn(value, "我的卡不见了")
        loss = next(item for item in result.approvals if item["tool"] == "report_card_lost")
        assert loss["tier"] == "red" and loss["needs_co_approval"]
        result = _card_approve(value, result)
        assert result.status == "done", result.reply
        assert value.services.bank_for(value.customer_id).list_cards()[0].status == "lost"
    finally:
        value.conn.close()


def test_restoring_higher_limit_is_not_a_green_undo(cardflow):
    bank = cardflow.services.bank_for(cardflow.customer_id)
    card = bank.list_cards()[0]
    original = card.online_limit_fen
    bank.set_card_limit(card.id, 30_000)
    assert Bank(cardflow.conn, cardflow.customer_id).previous_card_limit(card.id) == original
    with pytest.raises(Exception):
        bank.restore_card_limit(card.id)
    result = _card_turn(cardflow, "restore my original card limit")
    assert result.status == "needs_approval" and result.approvals[0]["tool"] == "set_card_limit"
    result = _card_approve(cardflow, result)
    assert result.status == "done" and bank.list_cards()[0].online_limit_fen == original
    assert bank.previous_card_limit(card.id) is None
    again = _card_turn(cardflow, "restore my original card limit")
    assert again.status == "done" and "No previous" in again.reply


def test_lower_restoration_and_no_history_are_clear(cardflow):
    bank = cardflow.services.bank_for(cardflow.customer_id)
    card = bank.list_cards()[0]
    result = _card_turn(cardflow, "改回原来的额度")
    assert result.status == "done" and "No previous" in result.reply
    bank.set_card_limit(card.id, 20_000)
    bank.set_card_limit(card.id, 30_000)
    result = _card_turn(cardflow, "restore my original card limit")
    assert result.status == "done" and "restored" in result.reply
    assert bank.list_cards()[0].online_limit_fen == 20_000
    assert bank.previous_card_limit(card.id) is None


def test_online_again_stays_frozen_until_explicit_confirmed_unfreeze(cardflow):
    bank = cardflow.services.bank_for(cardflow.customer_id)
    bank.freeze_card(bank.list_cards()[0].id)
    result = _card_turn(cardflow, "make my card work online again")
    assert result.status == "needs_approval", result.reply
    assert all(a["tier"] == "yellow" for a in result.approvals)
    result = _card_approve(cardflow, result)
    assert result.status == "done", result.reply
    assert bank.list_cards()[0].status == "frozen" and "still frozen" in result.reply
    explicit = _card_turn(cardflow, "unfreeze my card", sid=result.session_id)
    assert explicit.status == "needs_approval"
    explicit = _card_approve(cardflow, explicit)
    assert bank.list_cards()[0].status == "active" and "Card unfrozen" in explicit.reply
    assert "Undone" not in explicit.reply


def test_online_setting_on_frozen_card_does_not_claim_payments_work(cardflow):
    from bank import controls
    from agent.responder import compose_reply
    bank = cardflow.services.bank_for(cardflow.customer_id)
    card = bank.freeze_card(bank.list_cards()[0].id)
    raw = controls.set_control(cardflow.conn, cardflow.customer_id, card.id, "online", True, cardflow.clock["now"])
    raw = dict(raw, card_status=card.status, last4=card.last4, effective_enabled=False)
    result = compose_reply({"kind":"done", "steps":[{"step_id":"s1", "tool":"set_card_control", "status":"done", "result":{"raw":raw}}]},
        lang="en", vault=cardflow.services.vault, session=Session(customer_id=cardflow.customer_id), bank=bank, now=cardflow.clock["now"])
    assert result.grounded and "still frozen" in result.text and "remain unavailable" in result.text


def test_online_shopping_approval_names_whole_control(cardflow):
    result = _card_turn(cardflow, "解除对网购的限制")
    assert result.status == "needs_approval" and "online shopping" in result.reply
    assert "enable" in result.reply


@pytest.mark.parametrize("text", ["don't freeze my card", "不要挂失我的卡", "what if I lost my card?", "show my cards and transfer 200 to Lina"])
def test_negated_hypothetical_and_mixed_card_requests_are_not_executed(env, text):
    bank, conn, vault, session, _ = env
    assert resolve(_context(bank, conn, vault, session, text)) is None


def test_summary_endpoint_shows_masked_state_and_blocks():
    from fastapi.testclient import TestClient
    from api.app import create_app
    from bank import controls
    app = create_app(secret_key=b"card-summary-test-key", planner=_NoCardPlanner(), clock=lambda: NOW, demo_mode=True)
    try:
        with TestClient(app) as client:
            auth = client.post("/auth/demo-login", json={"persona":"young_pro"}).json()["token"]
            headers = {"Authorization":"Bearer " + auth}
            cid = app.state.runtime.conn.execute("SELECT id FROM customers WHERE persona_id='young_pro'").fetchone()[0]
            bank = app.state.runtime.services.bank_for(cid)
            card = bank.list_cards()[0]
            controls.set_control(bank.conn, cid, card.id, "online", False, NOW)
            bank.freeze_card(card.id)
            data = client.get("/me/summary", headers=headers).json()
            item = next(c for c in data["cards"] if c["last4"] == "****" + card.last4)
            assert item["kind"] == "physical" and item["status"] == "frozen"
            assert item["online_limit_fen"] == card.online_limit_fen
            assert "online_shopping" in item["active_blocks"]
            assert item["controls"]["online"] is False
            assert item["effective_controls"]["online"] is False
            assert "account_id" not in item and "id" not in item
    finally:
        app.state.runtime.conn.close()


@pytest.mark.parametrize("tool,tier", [("freeze_card","yellow"), ("report_card_lost","red"),
    ("unfreeze_card","yellow"), ("set_card_limit","yellow"), ("create_virtual_card","yellow"),
    ("transfer","yellow"), ("unblock_merchant","yellow"), ("restore_card_limit","green")])
def test_runtime_lock_exemption_is_exactly_two_actions(cardflow, tool, tier):
    from security.policy import StepDecision, recheck_step
    bank = cardflow.services.bank_for(cardflow.customer_id)
    decision = recheck_step(StepDecision(step_id="s1", tool=tool, tier=tier), resolved_amount_fen=20_000,
        bank=bank, now=cardflow.clock["now"], approved_tier=tier, payments_locked=True)
    assert decision.vetoed is (tool not in _LOCK_PROTECTION)
    assert decision.tier == tier
    if tool == "report_card_lost":
        assert decision.needs_mfa


def test_virtual_card_and_foreign_owner_are_not_confused(cardflow):
    bank = cardflow.services.bank_for(cardflow.customer_id)
    virtual = bank.create_virtual_card(20_000)
    physical = next(c for c in bank.list_cards() if c.kind == "physical")
    result = _card_turn(cardflow, "freeze my virtual card")
    assert result.status == "needs_approval", result.reply
    result = _card_approve(cardflow, result)
    assert result.status == "done"
    assert next(c for c in bank.list_cards() if c.id == virtual.id).status == "frozen"
    assert next(c for c in bank.list_cards() if c.id == physical.id).status == "active"


def test_list_cards_has_visible_kind_state_limit_and_mask(cardflow):
    result = _card_turn(cardflow, "show my cards")
    assert result.status == "done", result.reply
    assert "Physical card" in result.reply and "online limit" in result.reply and "active blocks" in result.reply
    assert "···" in result.reply and "acc_current" not in result.reply


def test_loss_mandate_does_not_allow_wrong_face_or_retyped_amount(cardflow):
    _lock_after_three_wrong_codes(cardflow)
    service = cardflow.services.mandates
    now = cardflow.clock["now"]
    mandate = service.issue(customer_id=cardflow.customer_id, session_id="face-test", plan_id="face-test:p1", step_id="s1",
        action="report_card_lost", payee_ref="card", amount_fen=0, tier="red", now=now)
    code = service.start_mfa(mandate.id, cardflow.customer_id, now)
    with pytest.raises(_MandateError):
        service.verify_mfa(mandate.id, cardflow.customer_id, otp=code, retyped_amount_fen=0, face_ok=False, now=now)
    with pytest.raises(_MandateError):
        service.verify_mfa(mandate.id, cardflow.customer_id, otp=code, retyped_amount_fen=1, face_ok=True, now=now)


def test_restoration_rejects_stale_approved_previous_limit(cardflow):
    bank = cardflow.services.bank_for(cardflow.customer_id)
    card = bank.list_cards()[0]
    bank.set_card_limit(card.id, 30_000)
    pending = _card_turn(cardflow, "restore my original card limit")
    assert pending.status == "needs_approval"
    bank.set_card_limit(card.id, 40_000)
    result = _card_approve(cardflow, pending)
    assert next(c for c in bank.list_cards() if c.id == card.id).online_limit_fen == 40_000
    assert "limit set" not in result.reply.lower()


@pytest.mark.parametrize("text", ["How much have I spent on my card last month?", "show my card subscriptions", "我的卡没丢，不要挂失", "I don't want to freeze my card"])
def test_card_rule_does_not_swallow_financial_reads_or_denials(env, text):
    bank, conn, vault, session, _ = env
    assert resolve(_context(bank, conn, vault, session, text)) is None


def test_undo_that_restarts_debits_waits_while_payments_are_locked(cardflow):
    # Regression: undo must not restart a cancelled subscription during a lock.
    bank = cardflow.services.bank_for(cardflow.customer_id)
    sub = next(s for s in bank.list_subscriptions() if s.status == "active")
    result = _card_turn(cardflow, "cancel all my subscriptions")
    assert result.status == "needs_approval", result.reply
    result = _card_approve(cardflow, result)
    assert next(s for s in bank.list_subscriptions() if s.id == sub.id).status == "cancelled"
    _lock_after_three_wrong_codes(cardflow)
    step_id = next(step["step_id"] for step in result.plan_view if step["tool"] == "cancel_subscription")
    undone = cardflow.agent.undo(result.session_id, cardflow.customer_id, result.plan_id, step_id)
    assert next(s for s in bank.list_subscriptions() if s.id == sub.id).status == "cancelled", undone.reply


@pytest.mark.parametrize("text", [
    "I lost my card, also wire 500 to my landlord",
    "freeze my card then move 500 to my landlord",
    "freeze my card and give my landlord 500",
    "lock my card and tip the driver 50",
    "卡丢了，再给妈妈打300",
    "冻结我的卡，还房东300",
])
def test_card_rule_leaves_card_plus_payment_to_the_planner(text):
    from types import SimpleNamespace

    from agent.intents import cards

    ctx = SimpleNamespace(raw_text=text, tokenized_text=text, conn=None, customer_id="c",
                          session_id="s", plan_id="p", vault=None, lang="en")
    assert cards.resolve(ctx) is None
