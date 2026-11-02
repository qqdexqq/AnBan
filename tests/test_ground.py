"""Grounding tests: reply figures must exist in this turn's tool results."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from bank.models import Card, Contact, Holding, Product, Transaction
from security.ground import (
    Claim,
    Facts,
    check,
    collect_facts,
    extract_claims,
    grounded_reply,
)


# ----------------------------------------------------------------- collection


def test_balance_reply_is_grounded() -> None:
    facts = collect_facts([2348050])  # get_balance() returns int fen

    assert check("您的余额是 ¥23,480.50。", facts).ok
    assert check("Your balance is ¥23,480.50.", facts).ok


def test_hallucinated_balance_is_rejected() -> None:
    facts = collect_facts([2348050])

    result = check("您的余额是 ¥23,480.00。", facts)

    assert not result.ok
    assert [claim.kind for claim in result.unknown] == ["money"]
    assert result.unknown[0].value == 2348000


def test_dataclass_money_fields_are_collected() -> None:
    facts = collect_facts(
        [
            Holding(product_id="p1", principal_fen=500000, value_fen=512345),
            Product(
                id="p1",
                name="稳健理财",
                risk_level=2,
                annual_rate_bp=350,
                min_fen=100000,
            ),
        ]
    )

    assert {500000, 512345, 100000} <= facts.amounts_fen
    assert check("投入 ¥5,000.00，现值 ¥5,123.45。", facts).ok


# --------------------------------------------------------------------- money


def test_wan_yuan_is_equivalent_to_12000_yuan() -> None:
    facts = collect_facts([{"balance_fen": 1200000}])  # ¥12,000.00

    assert check("到账 1.2万元。", facts).ok
    assert check("到账 RMB 12,000。", facts).ok


def test_chinese_numeral_amounts() -> None:
    facts = collect_facts([30000])

    assert check("一共三百块。", facts).ok
    assert check("需要付两千元。", collect_facts([200000])).ok


def test_signed_transaction_can_be_described_as_a_positive_spend() -> None:
    txn = Transaction(
        id="txn_1",
        account_id="acc_1",
        ts=datetime(2026, 10, 29, 13, 38),
        amount_fen=-12345,
        counterparty="咖啡店",
        category="dining",
        memo="",
        channel="card",
        country="CN",
        card_present=True,
    )
    facts = collect_facts([txn])

    assert check("消费 ¥123.45。", facts).ok


def test_short_bare_numbers_are_not_money() -> None:
    # One- and two-digit runs carry no money meaning on their own.
    assert extract_claims("交易金额 12") == []


def test_bare_three_digit_runs_are_money_claims() -> None:
    claims = extract_claims("交易金额 1234")

    assert [claim.kind for claim in claims] == ["money"]
    assert claims[0].value == 123400
    assert not check("余额是 9999", Facts()).ok


def test_money_currency_forms_and_units() -> None:
    assert extract_claims("到账 9,999 yuan。")[0].value == 999900
    assert extract_claims("到账 9999 RMB。")[0].value == 999900
    assert extract_claims("到账 9.9k。")[0].value == 990000
    assert extract_claims("到账 9.9w。")[0].value == 9900000

    facts = collect_facts([{"balance_fen": 523100}])  # ¥5,231.00
    assert check("余额 ¥5,231.00。", facts).ok
    assert check("余额 5,231 yuan。", facts).ok
    assert check("余额 5231 RMB。", facts).ok
    assert check("余额 5.231k。", facts).ok


def test_foreign_currency_is_never_grounded() -> None:
    facts = collect_facts([{"balance_fen": 999900}])

    for text in (
        "余额 $9,999。",
        "余额 USD 9999。",
        "余额 US$9,999。",
        "余额 9999 USD。",
    ):
        result = check(text, facts)
        assert not result.ok, text
        assert [claim.kind for claim in result.unknown] == ["money"]


# ---------------------------------------------------------------- percentages


def test_percentage_from_annual_rate_bp() -> None:
    product = Product(
        id="p1",
        name="稳健理财",
        risk_level=2,
        annual_rate_bp=350,
        min_fen=100000,
    )
    facts = collect_facts([product])

    assert facts.percents == {Decimal("3.50")}
    assert check("年化利率 3.5%。", facts).ok
    assert not check("年化利率 4%。", facts).ok


def test_chinese_percentage() -> None:
    facts = collect_facts([{"rate": 3.5}])

    assert check("年化百分之三点五。", facts).ok
    assert check("年化 3.5%", facts).ok


# -------------------------------------------------------------- masked accounts


def test_masked_account_tails() -> None:
    contact = Contact(
        id="c1",
        name="张三",
        relation="friend",
        note="",
        account_no="6222021234560932",
        phone="13800000000",
        bank_name="安办银行",
        birthday=date(1990, 1, 1),
        verified=True,
    )
    facts = collect_facts([contact])

    assert facts.tails == {"0932"}
    assert check("尾号0932的账户。", facts).ok
    assert check("已转到 ···0932。", facts).ok
    assert check("已转到 ****0932。", facts).ok
    assert not check("已转到 ···1111。", facts).ok


def test_card_last4_field() -> None:
    card = Card(
        id="card_1",
        kind="virtual",
        last4="0932",
        status="active",
        online_limit_fen=50000,
        account_id="acc_1",
    )
    facts = collect_facts([card])

    assert check("虚拟卡尾号0932。", facts).ok


# ------------------------------------------------------------------ non-figures


def test_counts_times_and_durations_are_ignored() -> None:
    text = (
        "1. 13:38，共 3 笔交易；"
        "3 transactions; 2 小时; 2h; 5 次"
    )

    assert extract_claims(text) == []
    for snippet in ("3 笔交易", "13:38", "2 小时后到账", "2026年", "1. 余额"):
        assert extract_claims(snippet) == [], snippet


# ---------------------------------------------------------------- spending report


def test_spending_report_totals_and_categories() -> None:
    report = {
        "by_category": {"dining": 12345, "transport": 6789},
        "total": 19134,
    }
    facts = collect_facts([report])

    assert check(
        "餐饮 ¥123.45，交通 ¥67.89，合计 ¥191.34。",
        facts,
    ).ok
    assert not check("餐饮 ¥123.45，合计 ¥200.00。", facts).ok


# -------------------------------------------------------------------- derived


def test_derived_sum_requires_add_derived() -> None:
    facts = collect_facts([{"balance_fen": 10000}])

    assert not check("转出 ¥300.00。", facts).ok

    facts.add_derived(30000, "sum of two transfers")
    result = check("转出 ¥300.00。", facts)

    assert result.ok
    assert facts.derived == [(30000, "sum of two transfers")]


# ---------------------------------------------------------------------- dates


def test_dates_are_grounded() -> None:
    txn = Transaction(
        id="txn_1",
        account_id="acc_1",
        ts=datetime(2026, 10, 29, 13, 38),
        amount_fen=-12345,
        counterparty="咖啡店",
        category="dining",
        memo="",
        channel="card",
        country="CN",
        card_present=True,
    )
    facts = collect_facts([txn])

    assert check("10月29日有一笔消费。", facts).ok
    assert check("2026-10-29 有一笔消费。", facts).ok
    assert not check("10月30日有一笔消费。", facts).ok
    assert not check("2026-10-30 有一笔消费。", facts).ok


def test_slash_dates_are_claims() -> None:
    facts = collect_facts([{"date": date(2026, 10, 25)}])

    assert check("10/25 到账。", facts).ok
    assert not check("10/26 到账。", Facts()).ok


def test_relative_dates_require_a_fact_and_today() -> None:
    today = date(2026, 10, 19)  # Monday

    # Without today, a relative date cannot be resolved.
    assert not check("下周三到账。", Facts()).ok
    assert not check("tomorrow 到账。", Facts()).ok

    next_wednesday = collect_facts([{"date": date(2026, 10, 28)}])
    assert check("下周三到账。", next_wednesday, today=today).ok
    assert check("next Wednesday 到账。", next_wednesday, today=today).ok

    tomorrow = collect_facts([{"date": date(2026, 10, 20)}])
    assert check("明天到账。", tomorrow, today=today).ok
    assert check("tomorrow 到账。", tomorrow, today=today).ok
    assert check("in 1 day 到账。", tomorrow, today=today).ok

    assert not check("后天到账。", tomorrow, today=today).ok


# --------------------------------------------------------------- grounded_reply


def test_grounded_reply_returns_draft_when_ok() -> None:
    facts = collect_facts([2348050])

    text, result = grounded_reply(
        "您的余额是 ¥23,480.50。",
        facts,
        lambda: "您的余额是 ¥23,480.50。",
    )

    assert text == "您的余额是 ¥23,480.50。"
    assert result.ok


def test_grounded_reply_falls_back_on_unknown_figure() -> None:
    facts = collect_facts([2348050])
    calls: list[bool] = []

    def fallback() -> str:
        calls.append(True)
        return "具体金额请以账户页面为准。"

    text, result = grounded_reply("您的余额是 ¥23,480.00。", facts, fallback)

    assert calls == [True]
    assert text == "具体金额请以账户页面为准。"
    assert not result.ok
    assert [claim.value for claim in result.unknown] == [2348000]


# ------------------------------------------------------------------ primitives


def test_claim_exposes_raw_text_and_value() -> None:
    claims = extract_claims("合计 ¥1,234.50")

    assert claims == [Claim("money", " ¥1,234.50", 123450, 2, 12)]
    assert claims[0].raw == claims[0].text


def test_empty_facts_reject_every_figure() -> None:
    facts = Facts()

    result = check("¥1.00 / 1% / 尾号0001 / 2026-10-29", facts)

    assert not result.ok
    assert {claim.kind for claim in result.unknown} == {
        "money",
        "percent",
        "account",
        "date",
    }
