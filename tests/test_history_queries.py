"""Read-only past transfer-recipient resolver tests.

The resolver must only ever produce a ``list_transactions`` read, must not be
confused by write/future/negated/refund wording, and may inherit a period only
from a verified successful read in the same customer and session.
"""

from __future__ import annotations

import json
import random
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from agent.graph import AnBanAgent, Services
from agent.history_queries import historical_transfer_plan
from bank import db
from bank.seed import generate
from security.audit import AuditKind, AuditLog, create_audit_schema
from security.mandate import MandateService, create_security_schema
from security.policy import PolicyConfig
from security.tokens import TokenVault

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42
SECRET = b"anban-history-queries-secret"
SESSION = "chat-1"


@pytest.fixture
def env():
    conn = db.connect(":memory:")
    state = generate("young_pro", SEED, TODAY)
    db.create_schema(conn)
    db.load_state(conn, state)
    create_security_schema(conn)
    create_audit_schema(conn)
    try:
        yield SimpleNamespace(conn=conn, state=state, customer_id=state.customer.id)
    finally:
        conn.close()


def _read_output(tool: str, period: str | None, step_id: str = "s1") -> dict:
    args = {} if period is None else {"period": period}
    return {
        "kind": "plan",
        "steps": [{"id": step_id, "tool": tool, "args": args, "after": []}],
    }


def _seed_read(
    env,
    *,
    tool: str = "spending_report",
    period: str | None = "last_month",
    plan_id: str = "seed-read",
    session_id: str = SESSION,
    customer_id: str | None = None,
    run: bool = True,
    ok: bool = True,
) -> None:
    audit = AuditLog(env.conn, SECRET)
    cid = customer_id or env.customer_id
    audit.append(
        customer_id=cid,
        session_id=session_id,
        plan_id=plan_id,
        kind=AuditKind.PLAN,
        payload={"output": _read_output(tool, period)},
        now=NOW,
    )
    if run:
        audit.append(
            customer_id=cid,
            session_id=session_id,
            plan_id=plan_id,
            kind=AuditKind.RUN,
            payload={"step_id": "s1", "tool": tool, "tier": "green", "ok": ok},
            now=NOW,
        )


def _plan(env, text: str, *, session_id: str = SESSION, customer_id: str | None = None):
    return historical_transfer_plan(
        text,
        conn=env.conn,
        secret_key=SECRET,
        customer_id=customer_id or env.customer_id,
        session_id=session_id,
    )


def _period(plan: dict) -> str:
    assert plan is not None
    assert plan["kind"] == "plan"
    assert len(plan["steps"]) == 1
    step = plan["steps"][0]
    assert step["tool"] == "list_transactions"
    assert step["args"]["category"] == "transfer"
    assert step["args"]["direction"] == "out"
    return step["args"]["period"]


@pytest.mark.parametrize("text", ["Who did I transfer 530 to?", "Whom did I transfer 530?", "我给谁转了530", "给谁转了530？"])
def test_past_question_amount_is_not_a_new_transfer(env, text):
    assert _plan(env, text)["steps"][0]["tool"] == "list_transactions"


@pytest.mark.parametrize("text", ["Who did I transfer to? Send Tom 50 yuan.", "Who did I transfer to last month? Transfer Tom 50.", "Who did I transfer to? I'm going to send Tom 50 next week."])
def test_separate_write_clause_is_not_silently_dropped(env, text):
    assert _plan(env, text) is None


def test_question_period_wins_over_background_month(env):
    assert _period(_plan(env, "Who did I transfer to this month? Last month I sent money to Lina.")) == "this_month"


@pytest.mark.parametrize(
    ("text", "period"),
    [
        ("Can you check to whom did I transfer?", "this_month"),
        ("To whom did I transfer about 530 yuan?", "this_month"),
        ("Who did I send money to last month?", "last_month"),
        (
            "You know there is a transfer of about 530 yuan to whom I did transfer",
            "this_month",
        ),
        ("我上个月转给谁了", "last_month"),
        ("刚才的钱转给谁了", "this_month"),
        ("这530元转给谁了", "this_month"),
    ],
)
def test_past_recipient_questions_return_a_read_plan(env, text: str, period: str) -> None:
    plan = _plan(env, text)
    assert _period(plan) == period
    assert "530" not in json.dumps(plan)
    assert text not in json.dumps(plan)


@pytest.mark.parametrize(
    "text",
    [
        "Who should I send money to?",
        "Can you send 530 to Lina?",
        "Send 530 to Tom",
        "Don't show my transfers",
        "Cancel my transfer, who did I send money to?",
        "Undo the transfer and tell me who I paid",
        "Who did I get a refund from?",
        "Show me my contacts",
        "Who did this?",
        "I don't want to see my transfers",
        "帮我把上个月的转账取消，是谁？",
        "给李娜转300",
        "我的联系人有哪些",
        "退款给谁了",
    ],
)
def test_writes_futures_negations_and_refunds_are_not_hijacked(env, text: str) -> None:
    assert _plan(env, text) is None


@pytest.mark.parametrize(
    ("text", "period"),
    [
        ("Who did I transfer to this year?", "this_year"),
        ("Who did I send money to in the last 7 days?", "last_7_days"),
        ("Who did I transfer to in the past 30 days?", "last_30_days"),
        ("谁在上个月收到我的转账？", "last_month"),
    ],
)
def test_explicit_period_in_the_current_text_wins(env, text: str, period: str) -> None:
    _seed_read(env, period="this_month")
    assert _period(_plan(env, text)) == period


def test_inherits_the_last_successful_read_in_the_same_chat(env) -> None:
    _seed_read(env, tool="spending_report", period="last_month")
    assert _period(_plan(env, "Who did I transfer to?")) == "last_month"

    _seed_read(env, tool="list_transactions", period="last_30_days", plan_id="seed-2")
    assert _period(_plan(env, "Who did I transfer to?")) == "last_30_days"


def test_unexecuted_failed_or_malformed_reads_are_not_inherited(env) -> None:
    _seed_read(env, period="last_month", run=False)
    assert _period(_plan(env, "Who did I transfer to?")) == "this_month"

    _seed_read(env, period="last_month", plan_id="failed", ok=False)
    assert _period(_plan(env, "Who did I transfer to?")) == "this_month"

    audit = AuditLog(env.conn, SECRET)
    audit.append(
        customer_id=env.customer_id,
        session_id=SESSION,
        plan_id="malformed",
        kind=AuditKind.PLAN,
        payload={"output": {"kind": "plan", "steps": "not-a-list"}},
        now=NOW,
    )
    assert _period(_plan(env, "Who did I transfer to?")) == "this_month"


def test_another_chat_or_customer_never_inherits(env) -> None:
    _seed_read(env, period="last_month", session_id="other-chat")
    assert _period(_plan(env, "Who did I transfer to?")) == "this_month"

    _seed_read(env, period="last_month", customer_id="another-customer")
    assert _period(_plan(env, "Who did I transfer to?")) == "this_month"


def test_forged_audit_chain_is_not_inherited(env) -> None:
    # Bypass AuditLog.append so the hash chain and its anchor are invalid.
    output = json.dumps(_read_output("spending_report", "last_month"))
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
            json.dumps({"output": json.loads(output)}),
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
            json.dumps({"step_id": "s1", "tool": "spending_report", "ok": True}),
            "b" * 64,
            "c" * 64,
        ),
    )
    env.conn.commit()
    assert AuditLog(env.conn, SECRET).verify().ok is False
    assert _period(_plan(env, "Who did I transfer to?")) == "this_month"


def test_graph_runs_the_read_plan_without_the_planner(env) -> None:
    _seed_read(env, period="last_month")

    def explode(*_args):
        raise AssertionError("the planner must not see a resolved history question")

    services = Services(
        conn=env.conn,
        vault=TokenVault(SECRET),
        mandates=MandateService(env.conn, SECRET, otp_rng=random.Random(7)),
        audit=AuditLog(env.conn, SECRET),
        planner=SimpleNamespace(plan=explode, calls=0),
        policy_config=PolicyConfig(),
        secret_key=SECRET,
        now=lambda: NOW,
    )
    agent = AnBanAgent(services)
    result = agent.start(env.customer_id, "Who did I transfer to?", SESSION, lang="en")
    assert result.status == "done", result.reply
    assert not result.approvals

    outputs = [
        json.loads(row["payload_json"]).get("output") or {}
        for row in env.conn.execute(
            "SELECT payload_json FROM security_audit"
            " WHERE customer_id = ? AND plan_id = ? AND kind = 'PLAN'",
            (env.customer_id, result.plan_id),
        )
    ]
    plan_outputs = [out for out in outputs if out.get("kind") == "plan"]
    assert plan_outputs
    assert plan_outputs[-1]["steps"][0]["tool"] == "list_transactions"
    assert plan_outputs[-1]["steps"][0]["args"] == {
        "period": "last_month",
        "category": "transfer",
        "direction": "out",
    }
