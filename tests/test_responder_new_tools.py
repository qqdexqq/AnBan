"""Responder renderers for the untrained tool batch."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from agent.responder import compose_reply
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
        yield bank, vault, session
    finally:
        conn.close()


def _compose(outcome: dict, *, lang: str, bank, vault, session):
    return compose_reply(
        outcome, lang=lang, vault=vault, session=session, bank=bank, now=NOW
    )


def _step(tool: str, result) -> dict:
    return {
        "step_id": "s1",
        "tool": tool,
        "tier": "green",
        "status": "done",
        "result": result,
        "error": None,
    }


def _outcome(tool: str, result) -> dict:
    return {"kind": "done", "steps": [_step(tool, result)]}


def _reply(env, tool: str, result, lang: str):
    bank, vault, session = env
    return _compose(_outcome(tool, result), lang=lang, bank=bank, vault=vault, session=session)


def _both(env, tool: str, result) -> tuple:
    return _reply(env, tool, result, "zh"), _reply(env, tool, result, "en")


# ------------------------------------------------------------------- bills


def test_list_bills_zh_and_en(env) -> None:
    result = {
        "raw": [
            {
                "category": "electricity",
                "biller": "国家电网",
                "biller_en": "State Grid",
                "amount_due_fen": 12345,
                "due_date": "2026-03-05",
            }
        ]
    }
    zh, en = _both(env, "list_bills", result)

    assert zh.grounded, zh.rejected
    assert "待缴账单" in zh.text
    assert "国家电网" in zh.text
    assert f"{fmt_fen(12345)}" in zh.text
    assert "3月5日前" in zh.text

    assert en.grounded, en.rejected
    assert "Bills due" in en.text
    assert "State Grid" in en.text
    assert f"{fmt_fen(12345)}" in en.text
    assert "5 Mar" in en.text


def test_list_bills_empty_zh_and_en(env) -> None:
    zh, en = _both(env, "list_bills", {"raw": []})

    assert zh.text == "目前没有待缴账单。"
    assert en.text == "No bills are due."


def test_pay_bill_zh_and_en(env) -> None:
    result = {
        "raw": {
            "category": "electricity",
            "biller": "国家电网",
            "biller_en": "State Grid",
            "amount_due_fen": 12345,
            "paid_at": "2026-10-17T12:00:00",
            "status": "paid",
        }
    }
    zh, en = _both(env, "pay_bill", result)

    assert zh.grounded, zh.rejected
    assert zh.text == f"已缴纳 国家电网 {fmt_fen(12345)}。"
    assert en.grounded, en.rejected
    assert en.text == f"Paid State Grid {fmt_fen(12345)}."


# ------------------------------------------------------------------ quotes


def test_installment_quote_zh_and_en(env) -> None:
    result = {
        "raw": {
            "periods": 12,
            "payment_fen": 107200,
            "total_fee_fen": 86400,
            "nominal_annual_pct": 7.2,
            "apr_pct": 13.03,
            "principal_fen": 1000000,
            "fee_rate_per_period_pct": 0.6,
        }
    }
    zh, en = _both(env, "installment_quote", result)

    assert zh.grounded, zh.rejected
    assert f"分 12 期" in zh.text
    assert fmt_fen(107200) in zh.text
    assert fmt_fen(86400) in zh.text
    assert "0.6%" in zh.text
    assert "7.2%" in zh.text
    assert "13.03%" in zh.text

    assert en.grounded, en.rejected
    assert "12 instalments" in en.text
    assert fmt_fen(107200) in en.text
    assert fmt_fen(86400) in en.text
    assert "0.6%" in en.text
    assert "7.2%" in en.text
    assert "13.03%" in en.text


def test_deposit_quote_zh_and_en(env) -> None:
    result = {
        "raw": {
            "principal_fen": 1000000,
            "term_months": 12,
            "rate_pct": 1.15,
            "interest_fen": 11500,
            "demand_rate_pct": 0.05,
            "early_half_term_fen": 250,
        }
    }
    zh, en = _both(env, "deposit_quote", result)

    assert zh.grounded, zh.rejected
    assert f"存 {fmt_fen(1000000)} 定期 12 个月" in zh.text
    assert "年利率 1.15%" in zh.text
    assert fmt_fen(11500) in zh.text
    assert "活期 0.05%" in zh.text
    assert fmt_fen(250) in zh.text
    assert "演示利率" in zh.text

    assert en.grounded, en.rejected
    assert fmt_fen(1000000) in en.text
    assert "12-month" in en.text
    assert "1.15%" in en.text
    assert fmt_fen(11500) in en.text
    assert "0.05%" in en.text
    assert fmt_fen(250) in en.text
    assert "(demo rates)" in en.text


# --------------------------------------------------------------- forecast


def test_forecast_cashflow_zh_and_en(env) -> None:
    result = {
        "raw": {
            "end_balance_fen": 500000,
            "lowest_balance_fen": 120000,
            "lowest_on_day_offset": 5,
            "goes_negative": False,
            "days": 30,
            "start_balance_fen": 900000,
            "items": [{"day": 18, "amount_fen": -30000, "label": "rent"}],
        }
    }
    zh, en = _both(env, "forecast_cashflow", result)

    assert zh.grounded, zh.rejected
    assert "未来 30 天" in zh.text
    assert f"预计最低余额 {fmt_fen(120000)}（第 5 天）" in zh.text
    assert f"期末约 {fmt_fen(500000)}" in zh.text
    assert "余额够用" in zh.text

    assert en.grounded, en.rejected
    assert "Over the next 30 days" in en.text
    assert fmt_fen(120000) in en.text
    assert "(day 5)" in en.text
    assert fmt_fen(500000) in en.text
    assert "balance is enough" in en.text


def test_forecast_negative_branch(env) -> None:
    result = {
        "raw": {
            "end_balance_fen": -5000,
            "lowest_balance_fen": -12345,
            "lowest_on_day_offset": 0,
            "goes_negative": True,
            "days": 30,
            "start_balance_fen": 100000,
            "items": [],
        }
    }
    zh, en = _both(env, "forecast_cashflow", result)

    assert zh.grounded, zh.rejected
    assert fmt_fen(-12345) in zh.text
    assert "（第 0 天）" not in zh.text
    assert "余额可能不够" in zh.text

    assert en.grounded, en.rejected
    assert fmt_fen(-12345) in en.text
    assert "(day 0)" not in en.text
    assert "may not be enough" in en.text


# ------------------------------------------------------------- help answer


def test_help_answer_is_verbatim_zh_and_en(env) -> None:
    zh_text = "存款保险最高保障人民币50万元。"
    en_text = "Deposits are protected up to ¥500,000 per depositor."
    result = {
        "raw": {
            "topic": "deposit_insurance",
            "zh": zh_text,
            "en": en_text,
            "source": "《存款保险条例》(2015) 第5条",
        }
    }
    zh, en = _both(env, "help_answer", result)

    assert zh.grounded, zh.rejected
    assert zh.text.startswith(zh_text)
    assert "来源" in zh.text
    assert "《存款保险条例》(2015) 第5条" in zh.text

    assert en.grounded, en.rejected
    assert en.text.startswith(en_text)
    assert "(Source:" in en.text


# ---------------------------------------------------------- transfer status


def test_list_transfers_zh_and_en(env) -> None:
    result = {
        "raw": [
            {
                "id": "t1",
                "to_contact_id": "c1",
                "amount_fen": 30000,
                "status": "held",
                "created_at": "2026-10-17T11:00:00",
                "release_at": "2026-10-17T17:30:00",
            },
            {
                "id": "t2",
                "to_contact_id": "c2",
                "amount_fen": 30000,
                "status": "sent",
                "created_at": "2026-10-16T11:00:00",
                "release_at": "2026-10-16T13:00:00",
            },
            {
                "id": "t3",
                "to_contact_id": "c3",
                "amount_fen": 30000,
                "status": "undone",
                "created_at": "2026-10-15T11:00:00",
                "release_at": "2026-10-15T13:00:00",
            },
        ]
    }
    zh, en = _both(env, "list_transfers", result)

    assert zh.grounded, zh.rejected
    assert f"- {fmt_fen(30000)}：暂缓中，17:30 前可撤销" in zh.text
    assert f"- {fmt_fen(30000)}：已转出" in zh.text
    assert f"- {fmt_fen(30000)}：已撤销" in zh.text

    assert en.grounded, en.rejected
    assert f"- {fmt_fen(30000)}: on hold, undo until 17:30" in en.text
    assert f"- {fmt_fen(30000)}: sent" in en.text
    assert f"- {fmt_fen(30000)}: undone" in en.text


def test_list_transfers_empty_zh_and_en(env) -> None:
    zh, en = _both(env, "list_transfers", {"raw": []})

    assert zh.text == "最近没有转账。"
    assert en.text == "No recent transfers."


# ------------------------------------------------------------ payee check


def test_verify_payee_fraud_list_zh_and_en(env) -> None:
    result = {"raw": {"payee": "c1", "on_fraud_list": True, "first_time": True}}
    zh, en = _both(env, "verify_payee", result)

    assert zh.grounded, zh.rejected
    assert "反诈名单" in zh.text
    assert "96110" in zh.text
    assert "请不要转账" in zh.text

    assert en.grounded, en.rejected
    assert "anti-fraud list" in en.text
    assert "96110" in en.text
    assert "Don't send money" in en.text


def test_verify_payee_clean_and_first_time(env) -> None:
    result = {"raw": {"payee": "c1", "on_fraud_list": False, "first_time": True}}
    zh, en = _both(env, "verify_payee", result)

    assert zh.grounded, zh.rejected
    assert "不在反诈名单上" in zh.text
    assert "第一次" in zh.text

    assert en.grounded, en.rejected
    assert "isn't on the anti-fraud list" in en.text
    assert "first time" in en.text

    seen = {"raw": {"payee": "c1", "on_fraud_list": False, "first_time": False}}
    zh2, en2 = _both(env, "verify_payee", seen)
    assert "第一次" not in zh2.text
    assert "first time" not in en2.text


# ---------------------------------------------------------- card controls


def test_set_card_control_zh_and_en(env) -> None:
    on = {"raw": {"card_id": "card_1", "control": "online", "enabled": True, "previous": False}}
    zh, en = _both(env, "set_card_control", on)

    assert zh.text == "已开启 网上支付。"
    assert en.text == "Online payments turned on."

    off = {"raw": {"card_id": "card_1", "control": "online", "enabled": False, "previous": True}}
    zh2, en2 = _both(env, "set_card_control", off)

    assert zh2.text == "已关闭 网上支付。"
    assert en2.text == "Online payments turned off."


def test_block_merchant_zh_and_en(env) -> None:
    result = {"raw": {"category": "gambling", "blocked": True}}
    zh, en = _both(env, "block_merchant", result)

    assert zh.text == "已屏蔽 赌博 类消费，立即生效。解除需要等待 48 小时。"
    assert en.text == (
        "Gambling spending is blocked now. Lifting the block takes 48 hours."
    )


def test_unblock_merchant_zh_and_en(env) -> None:
    result = {"raw": {"category": "gambling", "unblock_at": "2026-10-19T14:00:00"}}
    zh, en = _both(env, "unblock_merchant", result)

    assert zh.grounded, zh.rejected
    assert zh.text == "已申请解除 赌博 屏蔽，将在 10月19日 14:00 生效，在此之前仍然屏蔽。"

    assert en.grounded, en.rejected
    assert en.text == (
        "The gambling block will lift on 19 Oct 14:00; it stays on until then."
    )
