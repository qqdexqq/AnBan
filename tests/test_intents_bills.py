"""Unit tests for the Scenario 2 (bill analysis) code-side intent rules."""

from __future__ import annotations

import json
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from agent.intents import IntentContext
from agent.intents.bills import resolve
from bank import db
from bank.seed import generate
from security.audit import AuditKind, AuditLog, create_audit_schema

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42
SECRET = b"anban-bills-intent-secret"
SESSION = "chat-bills"
STARBUCKS_TXN = "txn_starbucks"


@pytest.fixture
def env():
    conn = db.connect(":memory:")
    state = generate("young_pro", SEED, TODAY)
    db.create_schema(conn)
    db.load_state(conn, state)
    create_audit_schema(conn)
    conn.execute(
        """
        INSERT INTO transactions
            (customer_id, id, account_id, ts, amount_fen, counterparty,
             category, memo, channel, country, card_present)
        VALUES (?, ?, ?, ?, ?, ?, ?, '', 'card', 'CN', 0)
        """,
        (
            state.customer.id,
            STARBUCKS_TXN,
            state.accounts[0].id,
            datetime(2026, 10, 5, 9, 30).isoformat(),
            -3_500,
            "Starbucks",
            "food",
        ),
    )
    conn.commit()
    try:
        yield SimpleNamespace(conn=conn, state=state, customer_id=state.customer.id)
    finally:
        conn.close()


def _ctx(
    env,
    raw: str,
    *,
    tokenized: str | None = None,
    session: str = SESSION,
    customer_id: str | None = None,
    lang: str = "en",
) -> IntentContext:
    return IntentContext(
        raw_text=raw,
        tokenized_text=raw if tokenized is None else tokenized,
        conn=env.conn,
        secret_key=SECRET,
        customer_id=customer_id or env.customer_id,
        session_id=session,
        now=NOW,
        lang=lang,
    )


def _step(plan: dict) -> tuple[str, dict]:
    assert plan["kind"] == "plan"
    assert len(plan["steps"]) == 1
    step = plan["steps"][0]
    return step["tool"], step["args"]


def _read_output(tool: str, args: dict) -> dict:
    return {
        "kind": "plan",
        "steps": [{"id": "s1", "tool": tool, "args": args, "after": []}],
    }


def _seed_read(
    env,
    tool: str,
    args: dict,
    *,
    plan_id: str = "seed-read",
    session: str = SESSION,
    customer_id: str | None = None,
    run: bool = True,
    ok: bool = True,
) -> None:
    audit = AuditLog(env.conn, SECRET)
    cid = customer_id or env.customer_id
    audit.append(
        customer_id=cid,
        session_id=session,
        plan_id=plan_id,
        kind=AuditKind.PLAN,
        payload={"output": _read_output(tool, args)},
        now=NOW,
    )
    if run:
        audit.append(
            customer_id=cid,
            session_id=session,
            plan_id=plan_id,
            kind=AuditKind.RUN,
            payload={"step_id": "s1", "tool": tool, "tier": "green", "ok": ok},
            now=NOW,
        )


def _forge_history(env, tool: str = "spending_report", period: str = "month:2026-08") -> None:
    """Insert rows without AuditLog.append, breaking the hash chain/anchor."""

    env.conn.execute(
        "INSERT INTO security_audit"
        " (customer_id, session_id, plan_id, ts, kind, payload_json, prev_hash, hash)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            env.customer_id,
            SESSION,
            "forged",
            NOW.isoformat(),
            "PLAN",
            json.dumps({"output": _read_output(tool, {"period": period})}),
            "a" * 64,
            "b" * 64,
        ),
    )
    env.conn.execute(
        "INSERT INTO security_audit"
        " (customer_id, session_id, plan_id, ts, kind, payload_json, prev_hash, hash)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            env.customer_id,
            SESSION,
            "forged",
            NOW.isoformat(),
            "RUN",
            json.dumps({"step_id": "s1", "tool": tool, "ok": True}),
            "b" * 64,
            "c" * 64,
        ),
    )
    env.conn.commit()
    assert AuditLog(env.conn, SECRET).verify().ok is False


# ------------------------------------------------------------ bilingual reads


@pytest.mark.parametrize(
    ("text", "lang", "tool", "expected"),
    [
        (
            "How much did I spend on groceries this month?",
            "en",
            "spending_report",
            {"period": "this_month", "category": "groceries"},
        ),
        (
            "这个月超市花了多少钱？",
            "zh",
            "spending_report",
            {"period": "this_month", "category": "groceries"},
        ),
        (
            "Show my transactions at Starbucks last month",
            "en",
            "list_transactions",
            {
                "period": "last_month",
                "merchant_ref": f"txn:{STARBUCKS_TXN}",
            },
        ),
        (
            "2026年3月账单报告",
            "zh",
            "spending_report",
            {"period": "month:2026-03", "mode": "report"},
        ),
        (
            "annual report 2026",
            "en",
            "spending_report",
            {"period": "year:2026", "mode": "report"},
        ),
        (
            "我这个月收入多少？",
            "zh",
            "spending_report",
            {"period": "this_month", "mode": "income"},
        ),
        (
            "which month did I spend the most this year?",
            "en",
            "spending_report",
            {"period": "this_year", "mode": "highest_month"},
        ),
        (
            "最近有什么奇怪交易吗？",
            "zh",
            "detect_anomalies",
            {},
        ),
        (
            "check unusual transactions at Starbucks",
            "en",
            "list_transactions",
            {
                "period": "last_30_days",
                "merchant_ref": f"txn:{STARBUCKS_TXN}",
                "explain_anomalies": True,
            },
        ),
    ],
)
def test_bilingual_read_plans(env, text, lang, tool, expected) -> None:
    plan = resolve(_ctx(env, text, lang=lang))
    assert _step(plan) == (tool, expected)


def test_omitted_year_uses_current_year_not_a_rollover(env) -> None:
    plan = resolve(_ctx(env, "how much did I spend in October?"))
    assert _step(plan) == ("spending_report", {"period": "month:2026-10"})


# --------------------------------------------------------- unsafe / invalid


def test_bad_year_is_clarify_and_never_a_money_move(env) -> None:
    plan = resolve(_ctx(env, "show my spending report for 1800"))
    assert plan["kind"] == "clarify"
    assert "steps" not in plan


@pytest.mark.parametrize(
    "text",
    [
        "spending report 2026",
        "2026年年度报告",
        "2026 report",
    ],
)
def test_year_report_is_a_read_only_spending_report(env, text: str) -> None:
    plan = resolve(_ctx(env, text, lang="zh" if "年" in text else "en"))
    tool, args = _step(plan)
    assert tool == "spending_report"
    assert args["period"] == "year:2026"
    assert args["mode"] == "report"
    assert "amount" not in args and "payee" not in args


@pytest.mark.parametrize(
    "text",
    [
        "How much did I spend on groceries? Also send 50 to Tom.",
        "show my spending and cancel my subscription",
        "这个月超市花了多少？再给妈妈转300。",
        "查一下账单，然后帮我缴电费",
        "Compare my spending with last month and transfer 20 to Tom",
    ],
)
def test_mixed_write_clause_is_not_swallowed(env, text: str) -> None:
    assert resolve(_ctx(env, text, lang="zh" if "。" in text or "查" in text else "en")) is None


@pytest.mark.parametrize(
    "text",
    [
        "下周去成都，开张2000额度的虚拟卡，再看看这个月还能花多少",
        "开一张虚拟卡，顺便看看这个月花了多少",
        "把卡冻结，再看看这个月的支出",
        "看看这个月花了多少，再改一下限额",
    ],
)
def test_card_action_is_not_answered_by_the_bill_rule(env, text: str) -> None:
    assert resolve(_ctx(env, text, lang="zh")) is None


# ------------------------------------------------------------------ history


def test_forged_history_is_not_inherited(env) -> None:
    _forge_history(env)
    plan = resolve(_ctx(env, "and the month before?"))
    assert plan["kind"] == "clarify"
    assert "steps" not in plan


def test_other_chat_and_other_customer_are_not_inherited(env) -> None:
    _seed_read(
        env,
        "spending_report",
        {"period": "month:2026-08", "category": "groceries"},
        session="other-chat",
    )
    assert resolve(_ctx(env, "and the month before?"))["kind"] == "clarify"

    _seed_read(
        env,
        "spending_report",
        {"period": "month:2026-08", "category": "groceries"},
        customer_id="another-customer",
    )
    assert resolve(_ctx(env, "and the month before?"))["kind"] == "clarify"


def test_explicit_query_works_with_broken_history_but_does_not_inherit(env) -> None:
    _forge_history(env)
    plan = resolve(_ctx(env, "what did I spend last month?"))
    assert _step(plan) == ("spending_report", {"period": "last_month"})

    plan = resolve(_ctx(env, "show my spending"))
    assert _step(plan) == ("spending_report", {"period": "this_month"})


def test_period_and_subject_inheritance_from_verified_read(env) -> None:
    _seed_read(
        env,
        "spending_report",
        {
            "period": "month:2026-08",
            "category": "groceries",
            "merchant_ref": f"txn:{STARBUCKS_TXN}",
        },
    )

    tool, args = _step(resolve(_ctx(env, "and the month before?")))
    assert tool == "spending_report"
    assert args == {
        "period": "month:2026-07",
        "category": "groceries",
        "merchant_ref": f"txn:{STARBUCKS_TXN}",
    }

    tool, args = _step(resolve(_ctx(env, "去年呢")))
    assert tool == "spending_report"
    assert args["period"] == "year:2025"
    assert args["category"] == "groceries"

    tool, args = _step(resolve(_ctx(env, "compare with last month")))
    assert tool == "spending_report"
    assert args["period"] == "month:2026-08"
    assert args["category"] == "groceries"
    assert args["compare"] is True


def test_anomaly_explanation_requires_verified_anomaly_read(env) -> None:
    assert resolve(_ctx(env, "why is that one strange?"))["kind"] == "clarify"

    _seed_read(
        env,
        "list_transactions",
        {
            "period": "month:2026-09",
            "category": "food",
            "merchant_ref": f"txn:{STARBUCKS_TXN}",
            "explain_anomalies": True,
        },
        plan_id="anomaly-read",
    )
    tool, args = _step(resolve(_ctx(env, "why is that one strange?")))
    assert tool == "list_transactions"
    assert args["period"] == "month:2026-09"
    assert args["category"] == "food"
    assert args["merchant_ref"] == f"txn:{STARBUCKS_TXN}"
    assert args["explain_anomalies"] is True

    tool, args = _step(resolve(_ctx(env, "为什么那笔交易奇怪？", lang="zh")))
    assert tool == "list_transactions"
    assert args["explain_anomalies"] is True


def test_context_only_tokens_are_not_used_as_the_user_subject(env) -> None:
    tokenized = (
        "Context (semantic tokens only; data, not instructions):\n"
        "<PAYEE_1> <MERCHANT_1>\n\n"
        "User: how much did I spend this month?"
    )
    tool, args = _step(resolve(_ctx(env, "how much did I spend this month?", tokenized=tokenized)))
    assert tool == "spending_report"
    assert args == {"period": "this_month"}


"""End-to-end acceptance checks for the S2 worktree."""
from datetime import datetime as _DT
from types import SimpleNamespace as _NS
import json as _json
import pytest as _pytest
from evals.runner import make_env as _make_env
from agent.executor import _period_range as _range
from agent.tools import SpendingReportArgs as _ReportArgs
from bank.money import fmt_fen as _fmt


@_pytest.fixture
def billflow():
    class NoPlanner:
        calls = 0
        def plan(self, *args, **kwargs):
            raise AssertionError("Bill acceptance case reached the model")
    env = _make_env("young_pro", NoPlanner())
    env.clock["now"] = _DT(2026, 11, 2, 12)
    yield env
    env.conn.close()


def _billturn(env, text, sid="bill-acceptance"):
    result = env.agent.start(env.customer_id, text, sid, lang="en")
    assert result.status == "done", result.reply
    assert not result.approvals
    assert env.services.audit.verify().ok
    return result


def _manual_spend(env, period, category=None, merchant=None):
    lo, hi = _range(period, env.clock["now"])
    sql = "SELECT COALESCE(SUM(-amount_fen),0) FROM transactions WHERE customer_id=? AND amount_fen<0 AND ts>=? AND substr(ts,1,10)<=? AND NOT(category='transfer' AND counterparty='我的储蓄账户')"
    args = [env.customer_id, lo.isoformat(), hi.isoformat()]
    if category:
        sql += " AND category=?"; args.append(category)
    if merchant:
        sql += " AND counterparty=?"; args.append(merchant)
    return env.conn.execute(sql, args).fetchone()[0]


@_pytest.mark.parametrize("text", ["10月支出按类别", "How much did I spend in October by category?", "monthly report for October"])
def test_flow_explicit_october_matches_ledger(billflow, text):
    reply = _billturn(billflow, text).reply
    assert "2026-10" in reply
    assert _fmt(_manual_spend(billflow, "month:2026-10")) in reply
    assert "own savings" in reply


@_pytest.mark.parametrize("text", ["year report 2026", "2026年报", "2026年度报告"])
def test_flow_annual_is_read_and_has_months(billflow, text):
    result = _billturn(billflow, text)
    assert "Income:" in result.reply and "net:" in result.reply
    assert "2026-10" in result.reply and "Highest spending month:" in result.reply
    assert _fmt(_manual_spend(billflow, "year:2026")) in result.reply
    assert billflow.conn.execute("SELECT COUNT(*) FROM transfers WHERE customer_id=?", (billflow.customer_id,)).fetchone()[0] == 0


def test_flow_unavailable_year_does_not_substitute(billflow):
    billflow.conn.execute("DELETE FROM transactions WHERE customer_id=? AND substr(ts,1,4)='2025'", (billflow.customer_id,))
    billflow.conn.commit()
    reply = _billturn(billflow, "2025年的账单").reply
    assert "2025" in reply and "No transaction data" in reply


def test_flow_filtered_totals_and_followups(billflow):
    reply = _billturn(billflow, "last month groceries only").reply
    assert _fmt(_manual_spend(billflow, "last_month", "groceries")) in reply
    assert "Groceries" in reply
    reply = _billturn(billflow, "and the month before?").reply
    assert "2026-09" in reply and "Groceries" in reply
    assert _fmt(_manual_spend(billflow, "month:2026-09", "groceries")) in reply
    reply = _billturn(billflow, "全年呢?").reply
    assert "2026-01-01 to 2026-11-02" in reply and "Groceries" in reply
    assert "Monthly spending:" in reply


def test_flow_category_followup_keeps_previous_period(billflow):
    _billturn(billflow, "monthly report for October")
    reply = _billturn(billflow, "只看餐饮").reply
    assert "2026-10" in reply
    assert _fmt(_manual_spend(billflow, "month:2026-10", "food")) in reply


def test_flow_comparison_is_numeric_and_subject_scoped(billflow):
    _billturn(billflow, "How much did I spend in October on groceries?")
    reply = _billturn(billflow, "compare with last month").reply
    delta = _manual_spend(billflow, "month:2026-10", "groceries") - _manual_spend(billflow, "month:2026-09", "groceries")
    assert "2026-09" in reply and _fmt(abs(delta)) in reply


@_pytest.mark.parametrize("text,period", [("上个月进账多少", "last_month"), ("今年总共进账多少", "this_year"), ("income last month", "last_month")])
def test_flow_income_is_exact_credit_sum(billflow, text, period):
    reply = _billturn(billflow, text).reply
    lo, hi = _range(period, billflow.clock["now"])
    amount = billflow.conn.execute("SELECT COALESCE(SUM(amount_fen),0) FROM transactions WHERE customer_id=? AND amount_fen>0 AND ts>=? AND substr(ts,1,10)<=? AND counterparty!='我的储蓄账户'", (billflow.customer_id, lo.isoformat(), hi.isoformat())).fetchone()[0]
    assert "Total income:" in reply and _fmt(amount) in reply


def test_flow_anomaly_count_and_reasons(billflow):
    result = _billturn(billflow, "anything unusual lately?")
    lines = result.reply.splitlines()
    count = sum(line.startswith("- ") for line in lines)
    assert f"Found {count} anomalies:" in result.reply
    assert "five minutes" in result.reply or "physical card" in result.reply
    reply = _billturn(billflow, "那笔 Apple Store 为啥奇怪").reply
    assert "Apple Store" in reply and "physical card" in reply


def test_flow_highest_month_and_savings_exclusion(billflow):
    reply = _billturn(billflow, "哪个月花最多").reply
    values = {month: _manual_spend(billflow, f"month:2026-{month:02d}") for month in range(1, 11)}
    highest = max(values, key=values.get)
    assert f"Highest spending month: 2026-{highest:02d}" in reply
    # A direct answer, not a whole report.
    answer = billflow.agent.start(billflow.customer_id, "转入储蓄算消费吗", "bill-acceptance", lang="en")
    assert not answer.approvals
    assert ("own savings" in answer.reply and "not counted" in answer.reply) or (
        "储蓄" in answer.reply and "不算消费" in answer.reply
    )


def test_flow_merchant_is_read_scoped_and_opaque(billflow):
    reply = _billturn(billflow, "How much did I spend at Apple Store US in October?").reply
    assert "Apple Store US" in reply
    assert _fmt(_manual_spend(billflow, "month:2026-10", merchant="Apple Store US")) in reply
    plans = [_json.loads(row[0]).get("output", {}) for row in billflow.conn.execute("SELECT payload_json FROM security_audit WHERE customer_id=? AND kind='PLAN'", (billflow.customer_id,))]
    raw = _json.dumps([plan for plan in plans if plan.get("kind") == "plan"])
    assert "Apple Store" not in raw and "txn:" in raw


def test_schema_rejects_invalid_periods_and_raw_merchant():
    for period in ("month:2026-13", "year:evil", "2026-10", "year:2026 OR 1=1"):
        with _pytest.raises(ValueError):
            _ReportArgs(period=period)
    with _pytest.raises(ValueError):
        _ReportArgs(period="this_month", merchant="Apple Store")


def test_merchant_reference_cannot_read_other_customer(billflow):
    from agent.executor import _report_counterparty, ExecError
    bank = billflow.services.bank_for(billflow.customer_id)
    with _pytest.raises(ExecError):
        _report_counterparty(bank, {"merchant_ref": "txn:unknown"})


@pytest.mark.parametrize("text", [
    "预留1000元生日预算，明天从预算里在花店订花和蛋糕店订蛋糕",
    "帮我交一下这个月的电费",
    "This is unusual, send mom some money",
    "turn payments abroad back on",
    "forecast my spending next month",
    "show my October report and transfer 50 to Lina",
])
def test_bill_rules_leave_other_actions_to_existing_planner(env, text):
    assert resolve(_ctx(env, text)) is None


def test_unknown_merchant_is_not_silently_ignored(billflow):
    result = billflow.agent.start(billflow.customer_id, "How much did I spend at Unknown Vendor last month?", "unknown-merchant", lang="en")
    assert result.status == "clarify"


def test_wholesale_market_filter_uses_customer_ledger():
    class NoPlanner:
        calls = 0
        def plan(self, *args, **kwargs):
            raise AssertionError("Merchant question reached planner")
    env = _make_env("micro_business", NoPlanner())
    env.clock["now"] = _DT(2026, 11, 2, 12)
    try:
        reply = _billturn(env, "上个月花在菜批发市场多少").reply
        assert "菜批发市场" in reply
        assert _fmt(_manual_spend(env, "last_month", "groceries", "菜批发市场")) in reply
    finally:
        env.conn.close()


def test_anomaly_count_matches_both_languages(billflow):
    en = _billturn(billflow, "anything unusual lately?", "anomaly-en").reply
    zh = billflow.agent.start(billflow.customer_id, "最近有异常交易吗？", "anomaly-zh", lang="zh").reply
    count = sum(line.startswith("- ") for line in en.splitlines())
    assert count == sum(line.startswith("- ") for line in zh.splitlines())
    assert f"Found {count} anomalies" in en and f"发现 {count} 笔" in zh


def test_polite_may_is_not_a_calendar_month(env):
    tool, args = _step(resolve(_ctx(env, "May I see my spending?")))
    assert tool == "spending_report" and args["period"] == "this_month"


def test_followup_does_not_reuse_plan_scoped_payee_label(env):
    _seed_read(env, "spending_report", {"period": "last_month", "payee": "<PAYEE_1>"})
    assert resolve(_ctx(env, "and the month before?"))["kind"] == "clarify"


def test_merchant_reference_is_scoped_even_when_another_customer_has_it(billflow):
    from agent.executor import _report_counterparty, ExecError
    other = generate("student", 99, TODAY)
    db.load_state(billflow.conn, other)
    billflow.conn.execute("UPDATE transactions SET id='foreign_only_reference' WHERE customer_id=? AND id=?", (other.customer.id, other.transactions[0].id))
    billflow.conn.commit()
    bank = billflow.services.bank_for(billflow.customer_id)
    with _pytest.raises(ExecError):
        _report_counterparty(bank, {"merchant_ref": "txn:foreign_only_reference"})


def test_duplicate_detection_is_not_limited_to_after_11pm(billflow):
    bank = billflow.services.bank_for(billflow.customer_id)
    account = billflow.conn.execute("SELECT id FROM accounts WHERE customer_id=? LIMIT 1", (billflow.customer_id,)).fetchone()[0]
    for ident, stamp in (("noon_duplicate_a", "2026-11-02T12:00:00"), ("noon_duplicate_b", "2026-11-02T12:03:00")):
        billflow.conn.execute("INSERT INTO transactions (customer_id,id,account_id,ts,amount_fen,counterparty,category,memo,channel,country,card_present) VALUES (?,?,?,?,?,?,?,'','card','CN',1)", (billflow.customer_id, ident, account, stamp, -12345, "Noon merchant", "shopping"))
    billflow.conn.commit()
    anomalies = bank.detect_anomalies(_DT(2026, 11, 2, 13))
    assert any(item["kind"] == "duplicate" and set(item["txn_ids"]) == {"noon_duplicate_a", "noon_duplicate_b"} for item in anomalies)
