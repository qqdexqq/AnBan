"""Responder tests: grounded template replies over the real seeded bank."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

import pytest

from agent.responder import ReplyResult, compose_reply
from agent.tools import REGISTRY
from bank import db
from bank.api import Bank
from bank.money import fmt_fen
from bank.seed import generate
from security.ground import Facts
from security.policy import Reason
from security.scam import detect
from security.tokens import (
    Session,
    TokenVault,
    tokenize_user_text,
)

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42
SECRET = b"anban-responder-secret"
LONG_DIGITS_RE = re.compile(r"\d{11,}")
INTERNAL_ID_RE = re.compile(r"[a-z]+_[a-z_]+")
ERROR_CODES = (
    "policy_veto",
    "reapproval_required",
    "mandate",
    "mandate_missing",
    "mandate_required",
    "not_approved",
    "tier_too_low",
    "mfa_missing",
    "mfa_required",
    "co_approval_missing",
    "co_approval_required",
    "missing",
    "expired",
    "replayed",
    "mismatch",
    "revoked",
    "locked",
    "otp_missing",
    "co_otp_missing",
    "rejected",
    "invalid_amount",
    "amount_missing",
    "token",
    "payee_ref_not_allowed",
    "invalid_date",
    "unknown_ref",
    "missing_field",
    "unsupported",
    "bank",
    "insufficient_funds",
    "not_found",
    "invalid_state",
    "invalid",
    "window_closed",
    "dependency",
    "undo",
    "bad_signature",
    "not_co_approver",
    "no_co_approver",
    "unknown",
    "tampered",
    "totally_unknown_code",
)


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


def _tokenize(bank, vault, session, text: str):
    return tokenize_user_text(text, bank, vault, session, NOW)


def _payees(entries):
    return [entry for entry in entries if entry.kind == "PAYEE"]


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


def _assert_no_long_digits(reply: ReplyResult) -> None:
    assert not LONG_DIGITS_RE.search(reply.text), reply.text


def _assert_no_internal_ids(reply: ReplyResult, context: str) -> None:
    match = INTERNAL_ID_RE.search(reply.text)
    assert match is None, f"{context}: {match.group(0)!r} in {reply.text!r}"


# ------------------------------------------------------------------- balance


def test_balance_zh_and_en(env) -> None:
    bank, vault, session, _state = env
    balance = bank.get_balance()
    outcome = _outcome(
        "done",
        [_step("get_balance", {"raw": balance, "balance": balance})],
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(outcome, lang="en", bank=bank, vault=vault, session=session)

    assert zh.grounded and not zh.rejected
    assert "当前账户可用余额" in zh.text
    assert fmt_fen(balance) in zh.text
    assert en.grounded
    assert "Available balance" in en.text
    assert fmt_fen(balance) in en.text
    _assert_no_long_digits(zh)
    _assert_no_long_digits(en)


# ------------------------------------------------------------ spending report


def test_spending_report_shows_total_and_top_three(env) -> None:
    bank, vault, session, _state = env
    report = bank.spending_report(TODAY.replace(day=1), TODAY)
    outcome = _outcome(
        "done",
        [
            _step(
                "spending_report",
                {
                    "raw": report,
                    "total": report["total"],
                    "by_category": dict(report["by_category"]),
                },
            )
        ],
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    assert fmt_fen(report["total"]) in zh.text
    top = sorted(report["by_category"].items(), key=lambda item: -item[1])[:3]
    for _category, amount in top:
        assert fmt_fen(amount) in zh.text
    _assert_no_long_digits(zh)


def test_spending_report_derived_total_is_grounded(env) -> None:
    bank, vault, session, _state = env
    outcome = _outcome(
        "done",
        [
            _step(
                "spending_report",
                {"by_category": {"food": 2000, "transport": 1000}},
            )
        ],
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    assert fmt_fen(3000) in zh.text


# --------------------------------------------------------------- anomalies


def test_anomalies_show_count_merchant_amount_and_time(env) -> None:
    bank, vault, session, _state = env
    outcome = _outcome(
        "done",
        [
            _step(
                "detect_anomalies",
                {
                    "anomalies": [
                        {
                            "kind": "duplicate",
                            "counterparty": "瑞幸咖啡",
                            "amount_fen": 3600,
                            "ts": datetime(2026, 10, 9, 23, 4),
                        }
                    ]
                },
            )
        ],
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    assert "发现 1 笔异常" in zh.text
    assert "瑞幸咖啡" in zh.text
    assert fmt_fen(3600) in zh.text
    assert "10-09 23:04" in zh.text


# ----------------------------------------------------------- subscriptions


def test_subscriptions_show_price_and_unused_flag(env) -> None:
    bank, vault, session, _state = env
    subscriptions = bank.list_subscriptions()
    assert subscriptions
    outcome = _outcome(
        "done", [_step("list_subscriptions", {"raw": subscriptions})]
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    first = subscriptions[0]
    assert first.merchant in zh.text
    assert fmt_fen(first.amount_fen) in zh.text


# ------------------------------------------------------------ transactions


def test_transactions_list_at_most_five(env) -> None:
    bank, vault, session, _state = env
    transactions = bank.list_transactions(limit=8)
    outcome = _outcome(
        "done", [_step("list_transactions", {"transactions": transactions})]
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    lines = [line for line in zh.text.splitlines() if line.startswith("- ")]
    assert len(lines) == min(5, len(transactions))
    _assert_no_long_digits(zh)


# ------------------------------------------------------------ portfolio


def test_portfolio_and_products_show_name_risk_rate(env) -> None:
    bank, vault, session, _state = env
    holdings = bank.get_portfolio()
    products = bank.list_products()

    portfolio = _compose(
        _outcome("done", [_step("get_portfolio", {"raw": holdings})]),
        lang="zh",
        bank=bank,
        vault=vault,
        session=session,
    )
    catalogue = _compose(
        _outcome("done", [_step("list_products", {"raw": products})]),
        lang="en",
        bank=bank,
        vault=vault,
        session=session,
    )

    assert portfolio.grounded, portfolio.rejected
    assert portfolio.text
    assert catalogue.grounded, catalogue.rejected
    for product in products:
        assert product.name in catalogue.text
    assert "%" in catalogue.text
    _assert_no_long_digits(portfolio)


# -------------------------------------------------------- pending transfer


def test_pending_transfer_shows_undo_time_zh_and_en(env) -> None:
    bank, vault, session, _state = env
    _tokenized, entries = _tokenize(bank, vault, session, "王建国")
    payee = _payees(entries)[0]
    raw = bank.transfer(
        to_contact_id=payee.real_ref,
        amount_fen=30_000,
        note="",
        idempotency_key="responder-undo",
        now=NOW,
        cooling=timedelta(hours=2),
    )
    result = {
        "raw": raw,
        "amount": raw.amount_fen,
        "payee": raw.to_contact_id,
        "transfer": raw.id,
        "undo": {
            "tool": "cancel_transfer",
            "args": {"transfer_id": raw.id, "amount_fen": raw.amount_fen},
        },
    }
    outcome = _outcome("done", [_step("transfer", result, tier="yellow")])

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(outcome, lang="en", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    assert "可在 14:00 前撤销" in zh.text
    assert fmt_fen(30_000) in zh.text
    assert "王建国" in zh.text
    assert "···" in zh.text
    assert en.grounded, en.rejected
    assert "You can undo until 14:00" in en.text
    assert "王建国" in en.text
    _assert_no_long_digits(zh)
    _assert_no_long_digits(en)
    for contact in _state.contacts:
        assert contact.account_no not in zh.text


def test_pending_transfer_after_window_says_money_left(env) -> None:
    bank, vault, session, _state = env
    _tokenized, entries = _tokenize(bank, vault, session, "王建国")
    payee = _payees(entries)[0]
    raw = bank.transfer(
        to_contact_id=payee.real_ref,
        amount_fen=12_000,
        note="",
        idempotency_key="responder-closed",
        now=NOW - timedelta(hours=4),
        cooling=timedelta(hours=2),
    )
    outcome = _outcome(
        "done", [_step("transfer", {"raw": raw, "amount": raw.amount_fen})]
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(outcome, lang="en", bank=bank, vault=vault, session=session)

    assert "钱已转出" in zh.text
    assert "The money has left" in en.text


# ----------------------------------------------------------- needs approval


def test_needs_approval_yellow_zh(env) -> None:
    bank, vault, session, _state = env
    _tokenized, entries = _tokenize(bank, vault, session, "王建国")
    payee = _payees(entries)[0]
    outcome = _outcome(
        "needs_approval",
        [_step("transfer", None, status="pending", tier="yellow")],
        approvals=[
            {
                "step_id": "s1",
                "tool": "transfer",
                "tier": "yellow",
                "amount_fen": 20_000,
                "payee_label": payee.label,
                "needs_mfa": False,
                "needs_co_approval": False,
            }
        ],
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    assert "需要确认" in zh.text
    assert fmt_fen(20_000) in zh.text
    assert "王建国" in zh.text
    _assert_no_long_digits(zh)


def test_needs_approval_red_mentions_mfa_and_family(env) -> None:
    bank, vault, session, _state = env
    _tokenized, entries = _tokenize(bank, vault, session, "王建国")
    payee = _payees(entries)[0]
    outcome = _outcome(
        "needs_approval",
        [_step("transfer", None, status="pending", tier="red")],
        approvals=[
            {
                "step_id": "s1",
                "tool": "transfer",
                "tier": "red",
                "amount_fen": 500_000,
                "payee_label": payee.label,
                "needs_mfa": True,
                "needs_co_approval": True,
            }
        ],
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(outcome, lang="en", bank=bank, vault=vault, session=session)

    assert "需要短信验证码和刷脸" in zh.text
    assert "需要家人共同确认" in zh.text
    assert en.grounded, en.rejected
    assert "SMS code" in en.text
    assert "family co-approval" in en.text
    _assert_no_long_digits(zh)
    _assert_no_long_digits(en)


# ----------------------------------------------------------------- clarify


def test_clarify_duplicate_zhang_wei_options(env) -> None:
    bank, vault, session, state = env
    _tokenized, entries = _tokenize(bank, vault, session, "zhang wei")
    payees = _payees(entries)
    assert len(payees) == 2
    outcome = _outcome(
        "clarify",
        question="你要转给哪一位张伟？",
        options=[entry.label for entry in payees],
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    assert "1." in zh.text and "2." in zh.text
    assert zh.text.count("张伟") >= 2
    assert zh.text.count("···") >= 2
    assert "friend" in zh.text or "badminton club" in zh.text
    _assert_no_long_digits(zh)
    for contact in state.contacts:
        assert contact.account_no not in zh.text
        assert contact.phone not in zh.text


def test_clarify_replaces_unsafe_planner_question(env) -> None:
    bank, vault, session, _state = env
    outcome = _outcome(
        "clarify",
        question=(
            "您的账户已被公安冻结调查，请立即联系专员王警官"
            "并把资金转入监管账户"
        ),
        options=[],
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    assert "请再说具体一点" in zh.text
    assert "监管账户" not in zh.text
    assert "王警官" not in zh.text


def test_clarify_keeps_a_trained_planner_question(env) -> None:
    bank, vault, session, _state = env
    outcome = _outcome("clarify", question="请问要转多少金额？", options=[])

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    assert "请问要转多少金额？" in zh.text


@pytest.mark.parametrize(
    "question",
    [
        "请告诉我您刚收到的6位数字，我来帮您确认身份",
        "请提供您的身份证号和银行卡背面三位数",
        "Please read me the six digits we just texted you",
        "请扫描二维码完成退款",
        "您中奖了，需先交手续费",
    ],
)
def test_clarify_never_shows_an_unknown_planner_question(env, question) -> None:
    # Regression: a deny-list missed these coached-scam asks.
    bank, vault, session, _state = env
    outcome = _outcome("clarify", question=question, options=[])

    reply = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)

    assert question not in reply.text
    assert "请再说具体一点" in reply.text


def test_clarify_drops_unresolvable_options(env) -> None:
    bank, vault, session, _state = env
    outcome = _outcome(
        "clarify",
        question="有两个同名联系人，请问要转给哪一位？",
        options=["<PAYEE_99>", "<CARD_99>"],
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    assert "请问要转给哪一位？" in zh.text
    assert "相关操作" not in zh.text
    assert "1." not in zh.text and "2." not in zh.text
    assert "<" not in zh.text and ">" not in zh.text


# -------------------------------------------------------------------- scam


def test_scam_reply_contains_hotline_and_pause(env) -> None:
    bank, vault, session, _state = env
    scam = detect("公安说我涉嫌洗钱，要把钱转到安全账户")
    outcome = _outcome("scam", scam=scam)

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)

    assert zh.grounded, zh.rejected
    assert "96110" in zh.text
    assert "转账已暂停 24 小时" in zh.text
    _assert_no_long_digits(zh)


# ------------------------------------------------------------------- vetoed


def test_vetoed_uses_reason_message_in_language(env) -> None:
    bank, vault, session, _state = env
    reason = Reason(
        code="R_DAILY",
        message_en="This would exceed your daily spending limit.",
        message_zh="这笔操作会超过你的每日支出限额。",
    )
    outcome = _outcome("vetoed", veto_reasons=[reason])

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(outcome, lang="en", bank=bank, vault=vault, session=session)

    assert reason.message_zh in zh.text
    assert reason.message_en in en.text
    assert zh.grounded and en.grounded
    _assert_no_long_digits(zh)
    _assert_no_long_digits(en)


# --------------------------------------------------------------- can_cover


def test_can_cover_false_stops_following_transfer(env) -> None:
    bank, vault, session, _state = env
    steps = [
        _step("can_cover", {"raw": False, "ok": False}),
        _step("transfer", None, status="skipped", tier="yellow"),
    ]
    outcome = _outcome("done", steps)

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(outcome, lang="en", bank=bank, vault=vault, session=session)

    assert "余额不足，没有执行后续转账" in zh.text
    assert "Insufficient balance" in en.text
    assert zh.grounded and en.grounded
    _assert_no_long_digits(zh)
    _assert_no_long_digits(en)


# --------------------------------------------------- internal ids and errors


def test_failed_transfer_uses_plain_reason(env) -> None:
    bank, vault, session, _state = env
    step = _step("transfer", None, status="failed", tier="yellow")
    step["error"] = "policy_veto"
    outcome = _outcome("done", [step])

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(outcome, lang="en", bank=bank, vault=vault, session=session)

    assert "转账" in zh.text
    assert "根据安全规则未执行" in zh.text
    assert "transfer" not in zh.text
    assert "policy_veto" not in zh.text
    assert "根据安全规则未执行" not in en.text
    assert "Not done: blocked by a safety rule" in en.text
    _assert_no_internal_ids(zh, "zh policy_veto")
    _assert_no_internal_ids(en, "en policy_veto")


def test_pending_reapproval_explains_amount_change(env) -> None:
    bank, vault, session, _state = env
    outcome = _outcome(
        "needs_approval",
        [_step("transfer", None, status="pending", tier="yellow")],
        approvals=[
            {
                "step_id": "s1",
                "tool": "transfer",
                "tier": "yellow",
                "amount_fen": 20_000,
                "needs_mfa": False,
                "needs_co_approval": False,
                "error": "reapproval_required",
            }
        ],
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(outcome, lang="en", bank=bank, vault=vault, session=session)

    assert "金额有变化，需要你重新确认" in zh.text
    assert "The amount changed - please confirm again" in en.text
    _assert_no_internal_ids(zh, "zh reapproval_required")
    _assert_no_internal_ids(en, "en reapproval_required")


def test_read_only_lookup_has_no_done_line(env) -> None:
    bank, vault, session, _state = env
    outcome = _outcome(
        "done",
        [
            _step("find_contact", {}),
            _step("list_cards", {}),
            _step("sandbox_chart", {}),
        ],
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)

    assert zh.text.strip() == "已完成。"
    assert "查找联系人" not in zh.text
    _assert_no_internal_ids(zh, "read-only lookup")


def test_every_tool_and_error_code_stays_plain(env) -> None:
    bank, vault, session, _state = env

    for tool in REGISTRY:
        for status, error in (
            ("done", None),
            ("failed", "policy_veto"),
            ("skipped", "dependency"),
            ("pending", None),
        ):
            step = _step(tool, {}, status=status)
            step["error"] = error
            for lang in ("zh", "en"):
                reply = _compose(
                    _outcome("done", [step]),
                    lang=lang,
                    bank=bank,
                    vault=vault,
                    session=session,
                )
                _assert_no_internal_ids(reply, f"{tool} {status} {lang}")

    for code in ERROR_CODES:
        step = _step("transfer", {}, status="failed", tier="yellow")
        step["error"] = code
        for lang in ("zh", "en"):
            reply = _compose(
                _outcome("done", [step]),
                lang=lang,
                bank=bank,
                vault=vault,
                session=session,
            )
            _assert_no_internal_ids(reply, f"error {code} {lang}")
            if code == "totally_unknown_code":
                expected = "未能完成" if lang == "zh" else "Couldn't be completed"
                assert expected in reply.text
            else:
                assert "未能完成" not in reply.text
                assert "Couldn't be completed" not in reply.text


# --------------------------------------------------------------- grounding fallback


def test_wrong_derived_sum_falls_back_to_status_only(env, monkeypatch) -> None:
    bank, vault, session, _state = env
    original = Facts.add_derived

    def wrong_derived(self, fen, reason):  # noqa: ANN001 - test shim
        # The template says ¥30.00, but the fact declared for it is ¥30.01.
        original(self, fen + 1, reason)

    monkeypatch.setattr(Facts, "add_derived", wrong_derived)
    outcome = _outcome(
        "done",
        [_step("spending_report", {"by_category": {"food": 2000, "transport": 1000}})],
    )

    reply = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)

    assert reply.grounded is False
    assert reply.rejected
    assert "¥" not in reply.text
    assert "支出报告" in reply.text
    assert "spending_report" not in reply.text
    _assert_no_long_digits(reply)


def test_ungrounded_figure_falls_back_to_status_only(env) -> None:
    bank, vault, session, _state = env
    # This hand-built result only exposes a bare integer under "raw"; the
    # template would phrase it as a balance, but grounding has no such fact.
    outcome = _outcome("done", [_step("get_balance", {"raw": 123_456})])

    reply = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)

    assert reply.grounded is False
    assert reply.rejected
    assert "¥" not in reply.text
    assert "查询余额" in reply.text
    assert "get_balance" not in reply.text
    assert "已完成" in reply.text
    _assert_no_long_digits(reply)


# --------------------------------------------------- customer wording fixes


def test_can_cover_success_is_silent_and_failure_shows_amounts(env) -> None:
    bank, vault, session, _state = env

    quiet = _outcome(
        "done", [_step("can_cover", {"ok": True, "balance": 50_000})]
    )
    zh = _compose(quiet, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(quiet, lang="en", bank=bank, vault=vault, session=session)
    assert "余额足够" not in zh.text
    assert "Balance is sufficient" not in en.text

    short = _outcome(
        "done",
        [
            _step(
                "can_cover",
                {"ok": False, "balance": 5_000, "amount": 30_000},
            )
        ],
    )
    zh = _compose(short, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(short, lang="en", bank=bank, vault=vault, session=session)
    assert f"余额不足：当前可用 {fmt_fen(5_000)}" in zh.text
    assert f"这笔需要 {fmt_fen(30_000)}，没有转账。" in zh.text
    assert zh.grounded, zh.rejected
    assert "Insufficient balance" in en.text
    assert fmt_fen(30_000) in en.text


def test_approval_omits_zero_amount_and_shows_targets(env) -> None:
    bank, vault, session, _state = env
    approvals = [
        {
            "step_id": "s1",
            "tool": "cancel_subscription",
            "tier": "red",
            "amount_fen": 0,
            "merchant": "Netflix",
            "needs_mfa": True,
            "needs_co_approval": False,
        },
        {
            "step_id": "s2",
            "tool": "freeze_card",
            "tier": "yellow",
            "amount_fen": 0,
            "card_last4": "1234",
            "needs_mfa": False,
            "needs_co_approval": False,
        },
    ]
    outcome = _outcome("needs_approval", approvals=approvals)

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(outcome, lang="en", bank=bank, vault=vault, session=session)

    assert "¥0.00" not in zh.text and "¥0.00" not in en.text
    assert "取消订阅 Netflix" in zh.text
    assert "冻结卡片 尾号1234" in zh.text
    assert "card ending 1234" in en.text
    assert zh.grounded, zh.rejected
    assert en.grounded, en.rejected


def test_approval_shows_the_target_and_the_monthly_day(env) -> None:
    bank, vault, session, _state = env
    approvals = [
        {
            "step_id": "s1",
            "tool": "cancel_subscription",
            "tier": "yellow",
            "amount_fen": 0,
            "target_display": "爱奇艺 VIP",
            "needs_mfa": False,
            "needs_co_approval": False,
        },
        {
            "step_id": "s2",
            "tool": "freeze_card",
            "tier": "yellow",
            "amount_fen": 0,
            "target_display": "···1234",
            "needs_mfa": False,
            "needs_co_approval": False,
        },
        {
            "step_id": "s3",
            "tool": "schedule_transfer",
            "tier": "yellow",
            "amount_fen": 30_000,
            "target_display": "妈妈",
            "day_of_month": 5,
            "needs_mfa": False,
            "needs_co_approval": False,
        },
    ]
    outcome = _outcome("needs_approval", approvals=approvals)

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(outcome, lang="en", bank=bank, vault=vault, session=session)

    assert "取消订阅 爱奇艺 VIP" in zh.text
    assert "冻结卡片 ···1234" in zh.text
    assert "定时转账 ¥300.00 妈妈（每月5日）" in zh.text
    assert "Cancel subscription 爱奇艺 VIP" in en.text
    assert "Scheduled transfer ¥300.00 妈妈 (every month on day 5)" in en.text
    assert zh.grounded, zh.rejected
    assert en.grounded, en.rejected


def test_approval_appends_first_reason(env) -> None:
    bank, vault, session, _state = env
    outcome = _outcome(
        "needs_approval",
        approvals=[
            {
                "step_id": "s1",
                "tool": "transfer",
                "tier": "red",
                "amount_fen": 500_000,
                "needs_mfa": True,
                "needs_co_approval": False,
                "reasons": [
                    {
                        "message_zh": "今日累计超过 ¥1,000，需要短信验证码和刷脸",
                        "message_en": "Today's total is over ¥1,000; "
                        "an SMS code and face check are needed",
                    }
                ],
            }
        ],
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(outcome, lang="en", bank=bank, vault=vault, session=session)

    assert "（今日累计超过 ¥1,000，需要短信验证码和刷脸）" in zh.text
    assert "Today's total is over ¥1,000" in en.text
    assert zh.grounded, zh.rejected
    assert en.grounded, en.rejected


def test_anomaly_kinds_are_translated_with_advisory(env) -> None:
    bank, vault, session, _state = env
    outcome = _outcome(
        "done",
        [
            _step(
                "detect_anomalies",
                {
                    "anomalies": [
                        {
                            "kind": "duplicate",
                            "counterparty": "瑞幸咖啡",
                            "amount_fen": 3600,
                            "ts": datetime(2026, 10, 9, 23, 4),
                        },
                        {
                            "kind": "foreign_cnp",
                            "counterparty": "Amazon",
                            "amount_fen": 9900,
                            "ts": datetime(2026, 10, 10, 8, 15),
                        },
                        {
                            "kind": "late_night_large",
                            "counterparty": "某商户",
                            "amount_fen": 20000,
                            "ts": datetime(2026, 10, 11, 2, 0),
                        },
                    ]
                },
            )
        ],
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(outcome, lang="en", bank=bank, vault=vault, session=session)

    assert "疑似重复扣款" in zh.text
    assert "境外无卡交易" in zh.text
    assert "深夜大额" in zh.text
    assert "如不是你本人操作，可以回复“冻结卡片”。" in zh.text
    assert "duplicate" not in zh.text
    assert "这项操作" not in zh.text
    assert "possible double charge" in en.text
    assert "overseas online charge" in en.text
    assert "large late-night charge" in en.text
    assert "If this wasn't you, reply 'freeze my card'." in en.text
    assert "Oct 9, 23:04" in en.text
    assert zh.grounded, zh.rejected


def test_english_dates_and_savings_sweeps(env) -> None:
    bank, vault, session, _state = env
    outcome = _outcome(
        "done",
        [
            _step(
                "list_transactions",
                {
                    "transactions": [
                        {
                            "ts": datetime(2026, 10, 18, 23, 59),
                            "amount_fen": -5000,
                            "counterparty": "我的储蓄账户",
                        }
                    ]
                },
            )
        ],
    )

    zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(outcome, lang="en", bank=bank, vault=vault, session=session)

    assert "转入储蓄" in zh.text
    assert "moved to savings" in en.text
    assert "Oct 18, 23:59" in en.text
    assert "10-18 23:59" not in en.text
    assert zh.grounded, zh.rejected
    assert en.grounded, en.rejected


def test_undo_success_and_already_undone(env) -> None:
    bank, vault, session, _state = env
    done = _outcome(
        "done", [_step("cancel_transfer", {"amount_fen": 30_000})]
    )
    again = _outcome(
        "done", [_step("cancel_transfer", {"already_undone": True})]
    )

    zh = _compose(done, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(done, lang="en", bank=bank, vault=vault, session=session)
    assert f"已撤销，{fmt_fen(30_000)} 已退回你的账户。" in zh.text
    assert f"Undone — {fmt_fen(30_000)} is back in your account." in en.text
    assert zh.grounded, zh.rejected

    zh = _compose(again, lang="zh", bank=bank, vault=vault, session=session)
    en = _compose(again, lang="en", bank=bank, vault=vault, session=session)
    assert "这笔已经撤销过了。" in zh.text
    assert "That was already undone." in en.text


def test_refusal_messages_are_plain(env) -> None:
    bank, vault, session, _state = env
    cases = {
        "not_account_owner": (
            "为了保护隐私，我只能查看你自己的账户。",
            "For privacy, I can only look at your own account.",
        ),
        "out_of_scope": (
            "这个我帮不上忙。我可以帮你查余额、转账、看账单、管理卡片和会员。",
            "I can't help with that.",
        ),
        "not_supported": (
            "这个操作不支持，也不会更改任何安全设置。",
            "That action isn't supported",
        ),
        "W4": (
            "理财都有风险，没有产品能保证收益。",
            "no product can guarantee returns",
        ),
    }
    for code, (zh_expected, en_expected) in cases.items():
        outcome = _outcome("refused", refuse_reason=code)
        zh = _compose(outcome, lang="zh", bank=bank, vault=vault, session=session)
        en = _compose(outcome, lang="en", bank=bank, vault=vault, session=session)
        assert zh_expected in zh.text, (code, zh.text)
        assert en_expected in en.text, (code, en.text)
        assert "未能完成" not in zh.text
        assert "Couldn't be completed" not in en.text


def test_lockout_and_mfa_attempts(env) -> None:
    bank, vault, session, _state = env
    mismatch = _step("transfer", None, status="failed", tier="red")
    mismatch["error"] = "mismatch"
    outcome = _outcome


def test_card_error_messages(env) -> None:
    bank, vault, session, _state = env

    not_frozen = _compose(
        _outcome("done", [_step("unfreeze_card", {"not_frozen": True})]),
        lang="zh",
        bank=bank,
        vault=vault,
        session=session,
    )
    assert "这张卡目前没有冻结，不需要解冻。" in not_frozen.text

    lost = _compose(
        _outcome("done", [_step("report_card_lost", {})]),
        lang="zh",
        bank=bank,
        vault=vault,
        session=session,
    )
    assert "已挂失。新卡可以在网点或 App 里申请补办。" in lost.text


def test_leftover_placeholders_are_scrubbed(env) -> None:
    bank, vault, session, _state = env
    zh = _compose(
        _outcome("clarify", question="请确认 <this action>", options=[]),
        lang="zh",
        bank=bank,
        vault=vault,
        session=session,
    )
    en = _compose(
        _outcome("clarify", question="Please confirm <this action>", options=[]),
        lang="en",
        bank=bank,
        vault=vault,
        session=session,
    )

    # Unknown clarify questions are replaced outright, placeholders and all.
    assert "<" not in zh.text and ">" not in zh.text
    assert "请再说具体一点" in zh.text
    assert "<" not in en.text and ">" not in en.text
    assert "say a bit more" in en.text


def test_every_trained_clarify_question_is_allowed() -> None:
    # The responder replaces unknown clarify questions; a question the planner
    # was trained on must never be replaced.
    import json
    from datetime import date

    from agent.responder import _CLARIFY_KNOWN_QUESTIONS
    from training.gold import load_scenarios, make_sample

    for scenario_id, scenario in load_scenarios().items():
        if (scenario.get("gold") or {}).get("kind") != "clarify":
            continue
        for lang in ("zh", "en"):
            try:
                sample = make_sample(
                    scenario_id, "young_pro", lang, "explicit", 42, date(2026, 10, 17)
                )
            except Exception:  # noqa: BLE001 - persona without the needed contacts
                continue
            gold = json.loads(sample["assistant"])
            if gold.get("kind") == "clarify":
                assert gold["question"] in _CLARIFY_KNOWN_QUESTIONS, (scenario_id, lang)
