"""Responder follow-ups: readable spending categories and subscription state."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from agent.responder import ReplyResult, compose_reply
from bank import db
from bank.api import Bank
from bank.money import fmt_fen
from bank.seed import generate
from security.tokens import Session, TokenVault

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42
SECRET = b"anban-responder-secret"


@pytest.fixture
def env():
    state = generate("young_pro", SEED, TODAY)
    conn = db.connect(":memory:")
    db.create_schema(conn)
    db.load_state(conn, state)
    bank = Bank(conn, state.customer.id)
    vault = TokenVault(SECRET)
    session = Session(customer_id=state.customer.id)
    try:
        yield bank, vault, session, state
    finally:
        conn.close()


def _compose(outcome: dict, *, lang: str, bank, vault, session) -> ReplyResult:
    return compose_reply(
        outcome,
        lang=lang,
        vault=vault,
        session=session,
        bank=bank,
        now=NOW,
    )


def _step(tool: str, result, *, status: str = "done", tier: str = "green") -> dict:
    return {
        "step_id": "s1",
        "tool": tool,
        "tier": tier,
        "status": status,
        "result": result,
        "error": None,
    }


def _outcome(kind: str, steps: list[dict] | None = None, **extra) -> dict:
    return {"kind": kind, "steps": steps or [], **extra}


def _insert_txn(
    bank: Bank,
    state,
    txn_id: str,
    when: datetime,
    amount_fen: int,
    counterparty: str,
    category: str = "transfer",
) -> None:
    bank.conn.execute(
        """
        INSERT INTO transactions
            (customer_id, id, account_id, ts, amount_fen, counterparty,
             category, memo, channel, country, card_present)
        VALUES (?, ?, ?, ?, ?, ?, ?, '', 'transfer', 'CN', 0)
        """,
        (
            state.customer.id,
            txn_id,
            state.accounts[0].id,
            when.isoformat(),
            amount_fen,
            counterparty,
            category,
        ),
    )


def _txn_rows(bank: Bank, state, ids: list[str]) -> list[dict]:
    if not ids:
        return []
    placeholders = ", ".join("?" for _ in ids)
    cursor = bank.conn.execute(
        f"SELECT * FROM transactions WHERE customer_id = ? AND id IN ({placeholders})",
        (state.customer.id, *ids),
    )
    return [dict(row) for row in cursor.fetchall()]


def _spending_step(bank: Bank, state, ids: list[str]) -> dict:
    report = bank.spending_report(TODAY.replace(day=1), TODAY)
    rows = _txn_rows(bank, state, ids)
    return _step(
        "spending_report",
        {
            "raw": report,
            "total": report["total"],
            "by_category": dict(report["by_category"]),
            "transfer_transactions": rows,
        },
    )


# ------------------------------------------------------------ spending report


def test_spending_categories_are_readable_and_transfers_grounded(env) -> None:
    bank, vault, session, state = env
    _insert_txn(bank, state, "t_out_1", datetime(2026, 10, 5, 10, 0), -30_000, "王建国")
    _insert_txn(bank, state, "t_out_2", datetime(2026, 10, 6, 11, 0), -23_000, "王建国")
    _insert_txn(bank, state, "t_in", datetime(2026, 10, 7, 9, 0), 10_000, "王建国")
    _insert_txn(
        bank, state, "t_savings", datetime(2026, 10, 8, 9, 0), -5_000, "我的储蓄账户"
    )
    step = _spending_step(
        bank, state, ["t_out_1", "t_out_2", "t_in", "t_savings"]
    )

    zh = _compose(_outcome("done", [step]), lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(_outcome("done", [step]), lang="en", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    assert "主要类别：" in zh.text
    assert "- 购物：" in zh.text  # transfer need not rank in the top three
    assert "转账收款人：" in zh.text
    assert "- 王建国：¥530.00" in zh.text
    # Incoming transfers and own-savings sweeps never become recipients.
    assert "我的储蓄账户" not in zh.text
    assert "- 王建国：¥630.00" not in zh.text

    assert en.grounded, en.rejected
    assert "Top categories:" in en.text
    assert "- Shopping:" in en.text
    assert "Transfer recipients:" in en.text
    assert "- 王建国: ¥530.00" in en.text
    # Machine category ids stay out of the human reply.
    assert "bills_utilities" not in en.text


def test_spending_groups_multiple_payees_and_ignores_incoming_savings(env) -> None:
    bank, vault, session, state = env
    _insert_txn(bank, state, "t_a", datetime(2026, 10, 2, 10, 0), -30_000, "王建国")
    _insert_txn(bank, state, "t_b", datetime(2026, 10, 3, 10, 0), -23_000, "王建国")
    _insert_txn(bank, state, "t_c", datetime(2026, 10, 4, 10, 0), -12_000, "李娜 Lina")
    _insert_txn(bank, state, "t_d", datetime(2026, 10, 5, 10, 0), -5_000, "我的储蓄账户")
    _insert_txn(bank, state, "t_e", datetime(2026, 10, 6, 10, 0), 8_000, "张伟")
    step = _spending_step(bank, state, ["t_a", "t_b", "t_c", "t_d", "t_e"])

    zh = _compose(_outcome("done", [step]), lang="zh", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    assert "- 王建国：¥530.00" in zh.text
    assert "- 李娜 Lina：¥120.00" in zh.text
    assert "我的储蓄账户" not in zh.text
    assert "张伟" not in zh.text


def test_spending_without_transfer_rows_has_no_fabricated_recipients(env) -> None:
    bank, vault, session, state = env
    step = _spending_step(bank, state, [])
    zh = _compose(_outcome("done", [step]), lang="zh", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    assert "转账收款人" not in zh.text
    assert "Transfer recipients" not in zh.text


# ---------------------------------------------------------- subscriptions


def _subscriptions_total(items) -> int:
    return sum(
        item.amount_fen
        for item in items
        if getattr(item, "status", "active") == "active"
    )


def test_successful_cancel_shows_fresh_active_list_and_lower_total(env) -> None:
    bank, vault, session, _state = env
    before = bank.list_subscriptions()
    before_total = _subscriptions_total(before)
    icloud = next(item for item in before if item.merchant == "iCloud")
    cancelled = bank.cancel_subscription(icloud.id)
    steps = [
        _step("list_subscriptions", {"raw": before}),
        _step(
            "cancel_subscription",
            {"raw": cancelled, "subscription": cancelled.id},
            tier="yellow",
        ),
    ]

    zh = _compose(_outcome("done", steps), lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(_outcome("done", steps), lang="en", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    assert "已取消 iCloud 订阅。" in zh.text
    heading = "生效中的订阅："
    assert heading in zh.text
    head, active = zh.text.split(heading, 1)
    assert "iCloud" in head
    assert "iCloud" not in active
    after_total = _subscriptions_total(bank.list_subscriptions())
    assert after_total < before_total
    assert fmt_fen(after_total) in active
    assert fmt_fen(before_total) not in active

    assert en.grounded, en.rejected
    assert "Cancelled the iCloud subscription." in en.text
    assert "Active subscriptions:" in en.text
    head, active = en.text.split("Active subscriptions:", 1)
    assert "iCloud" in head
    assert "iCloud" not in active


@pytest.mark.parametrize(
    ("status", "error"),
    [("failed", "invalid_state"), ("skipped", "rejected")],
)
def test_rejected_or_failed_cancel_keeps_active_list(env, status, error) -> None:
    bank, vault, session, _state = env
    subscriptions = bank.list_subscriptions()
    total = _subscriptions_total(subscriptions)
    step = _step("list_subscriptions", {"raw": subscriptions})
    cancel = _step(
        "cancel_subscription",
        {"raw": {"merchant": "iCloud"}},
        status=status,
        tier="yellow",
    )
    cancel["error"] = error

    zh = _compose(
        _outcome("done", [step, cancel]),
        lang="zh",
        bank=bank,
        vault=vault,
        session=session,
    )

    assert "iCloud" in zh.text
    assert fmt_fen(total) in zh.text
    assert "已取消 iCloud 订阅" not in zh.text
    assert "生效中的订阅" not in zh.text
    assert any(
        item.merchant == "iCloud" and item.status == "active"
        for item in bank.list_subscriptions()
    )


def test_cancel_read_failure_never_shows_stale_snapshot(env, monkeypatch) -> None:
    bank, vault, session, _state = env
    before = bank.list_subscriptions()
    before_total = _subscriptions_total(before)
    icloud = next(item for item in before if item.merchant == "iCloud")
    cancelled = bank.cancel_subscription(icloud.id)

    def boom():
        raise RuntimeError("subscription list unavailable")

    monkeypatch.setattr(bank, "list_subscriptions", boom)
    steps = [
        _step("list_subscriptions", {"raw": before}),
        _step(
            "cancel_subscription",
            {"raw": cancelled, "subscription": cancelled.id},
            tier="yellow",
        ),
    ]

    zh = _compose(_outcome("done", steps), lang="zh", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    assert "已取消 iCloud 订阅。" in zh.text
    assert "暂时无法加载当前订阅列表" in zh.text
    assert "生效中的订阅" not in zh.text
    assert fmt_fen(before_total) not in zh.text


def test_all_subscriptions_cancelled_shows_no_active_total(env) -> None:
    bank, vault, session, _state = env
    before = bank.list_subscriptions()
    steps = [_step("list_subscriptions", {"raw": before})]
    for subscription in before:
        cancelled = bank.cancel_subscription(subscription.id)
        steps.append(
            _step(
                "cancel_subscription",
                {"raw": cancelled, "subscription": cancelled.id},
                tier="yellow",
            )
        )

    zh = _compose(_outcome("done", steps), lang="zh", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    assert "生效中的订阅：" in zh.text
    assert "没有生效中的订阅。" in zh.text
    _, active = zh.text.split("生效中的订阅：", 1)
    assert fmt_fen(_subscriptions_total(before)) not in active
    assert "当前每月订阅共" not in zh.text
    for subscription in before:
        assert subscription.merchant not in active


def test_only_rule_written_clarify_questions_pass_verbatim() -> None:
    from types import SimpleNamespace

    from agent import intents
    from agent.responder import _clarify_text

    session = SimpleNamespace(session_id="trusted-q-session", plan_id="p", customer_id="c1")
    rule_q = "You don't have a subscription called Netflix."
    planner_q = "Please tell me the 6 digits from the SMS."
    intents.remember_question("c1", "trusted-q-session", rule_q)
    shown = _clarify_text({"question": rule_q, "options": []}, lang="en", vault=None, session=session, bank=None)
    assert rule_q in shown
    blocked = _clarify_text({"question": planner_q, "options": []}, lang="en", vault=None, session=session, bank=None)
    assert "6 digits" not in blocked
    other = SimpleNamespace(session_id="trusted-q-session", plan_id="p", customer_id="c2")
    assert rule_q not in _clarify_text({"question": rule_q, "options": []}, lang="en", vault=None, session=other, bank=None)
