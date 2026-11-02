"""End-to-end tests for the LangGraph turn pipeline.

Every test drives the real graph with the real seeded bank, vault, mandates,
policy and audit log.  Only the planner is scripted: it returns whatever the
test tells it to, so the tests assert what *code* does with a model proposal.
"""

from __future__ import annotations

import json
import random
import re
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from agent import habits
from agent.graph import AnBanAgent, Services, _freeze_until
from agent.planner import DEFAULT_REFUSAL
from bank import bills, db
from bank.money import fmt_fen
from bank.seed import generate
from security.audit import AuditLog, create_audit_schema
from security.mandate import MandateService, create_security_schema
from security.policy import PolicyConfig
from security.tokens import TokenVault

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42
SECRET = b"anban-graph-test-secret"
PII_RE = re.compile(r"\d{11,}")


# ----------------------------------------------------------------------- fakes


class RecordingPlanner:
    """A planner whose answer is supplied by the test; counts every call."""

    def __init__(self) -> None:
        self.calls = 0
        self.user_messages: list[str] = []
        self.respond = lambda user: DEFAULT_REFUSAL

    def plan(self, system: str, user: str) -> str:
        self.calls += 1
        self.user_messages.append(user)
        return self.respond(user)


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def make_env():
    """Build one seeded customer plus a fresh agent for a single test."""

    connections = []

    def _make(persona: str = "young_pro"):
        state = generate(persona, SEED, TODAY)
        conn = db.connect(":memory:")
        connections.append(conn)
        db.create_schema(conn)
        db.load_state(conn, state)
        create_security_schema(conn)
        create_audit_schema(conn)

        clock = {"now": NOW}
        planner = RecordingPlanner()
        services = Services(
            conn=conn,
            vault=TokenVault(SECRET),
            mandates=MandateService(conn, SECRET, otp_rng=random.Random(7)),
            audit=AuditLog(conn, SECRET),
            planner=planner,
            policy_config=PolicyConfig(),
            secret_key=SECRET,
            now=lambda: clock["now"],
        )
        agent = AnBanAgent(services)
        return SimpleNamespace(
            agent=agent,
            services=services,
            planner=planner,
            clock=clock,
            conn=conn,
            state=state,
            customer_id=state.customer.id,
        )

    try:
        yield _make
    finally:
        for conn in connections:
            conn.close()


@pytest.fixture
def env(make_env):
    return make_env("young_pro")


# ---------------------------------------------------------------------- helpers


def _plan(*steps: dict) -> str:
    return json.dumps({"kind": "plan", "steps": list(steps)}, ensure_ascii=False)


def _step(step_id: str, tool: str, args: dict | None = None, after=None) -> dict:
    return {
        "id": step_id,
        "tool": tool,
        "args": args or {},
        "after": list(after or ()),
    }


def _user_tokens(user: str, kind: str) -> list[str]:
    """Token labels from the ``User:`` line (never from bank context)."""

    line = user.rsplit("User:", 1)[-1]
    return re.findall(rf"<{kind}_[A-Za-z0-9]+>", line)


def _only_token(user: str, kind: str) -> str:
    labels = _user_tokens(user, kind)
    assert len(labels) == 1, f"expected one {kind} token in {user!r}, got {labels}"
    return labels[0]


def _transfer_output(user: str) -> str:
    payee = _only_token(user, "PAYEE")
    amount = _only_token(user, "AMT")
    return _plan(
        _step("s1", "transfer", {"payee": payee, "amount": amount})
    )


def _transfer_rows(env) -> int:
    return int(env.conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0])


def _approval(result):
    assert len(result.approvals) == 1, result.approvals
    return result.approvals[0]


def _start_transfer(env, text: str = "给妈妈转300"):
    env.planner.respond = _transfer_output
    result = env.agent.start(env.customer_id, text)
    assert result.status == "needs_approval", (result.status, result.reply)
    return result


# ------------------------------------------------------------------------ tests


def test_balance_question_returns_formatted_balance(env) -> None:
    env.planner.respond = lambda user: _plan(_step("s1", "get_balance"))

    result = env.agent.start(env.customer_id, "我的余额还有多少？")

    balance = env.services.bank_for(env.customer_id).get_balance()
    assert result.status == "done"
    assert fmt_fen(balance) in result.reply


def test_yellow_transfer_needs_one_approval_then_executes(env) -> None:
    result = _start_transfer(env)

    approval = _approval(result)
    assert approval["tier"] == "yellow"
    assert approval["amount_fen"] == 30_000
    assert approval["step_id"] == "s1"

    resumed = env.agent.resume(
        result.session_id, env.customer_id, {"s1": {"confirm": True}}
    )

    assert resumed.status == "done"
    assert _transfer_rows(env) == 1
    assert "撤销" in resumed.reply


def test_approval_card_names_the_product(env) -> None:
    product = env.services.bank_for(env.customer_id).list_products()[0]

    def output(user: str) -> str:
        # Products only appear in the bank context, rendered with tags after
        # the label ("<PROD_1 | risk 1 | ...>"), so take the bare label.
        token = re.search(r"<PROD_[A-Za-z0-9]+", user)
        assert token is not None, user
        return _plan(
            _step(
                "s1",
                "subscribe_product",
                {"product": token.group(0) + ">", "amount": _only_token(user, "AMT")},
            )
        )

    env.planner.respond = output
    result = env.agent.start(env.customer_id, "买安享稳健 20000")

    approval = _approval(result)
    assert approval["target_display"] == product.name
    assert product.name in result.reply


def test_mock_redemption_uses_graph_product_binding_and_mfa(env):
    from bank.redemption import quote

    bank = env.services.bank_for(env.customer_id)
    expected = quote(env.conn, env.customer_id, "p3")["payout_fen"]
    index = [p.id for p in bank.list_products()].index("p3")
    def output(user):
        products = re.findall(r"<PROD_[A-Za-z0-9]+", user)
        return _plan(_step("s1", "redeem_product", {"product": products[index] + ">"}))
    env.planner.respond = output
    before = bank.get_balance()
    result = env.agent.start(env.customer_id, "赎回我持有的基金")
    approval = _approval(result)
    assert approval["tier"] == "red"
    assert approval["amount_fen"] == expected
    assert bank.get_balance() == before
    resumed = env.agent.resume(result.session_id, env.customer_id, {"s1": {
        "confirm": True, "otp": approval["demo_otp"],
        "retyped_amount_fen": expected, "face_ok": True,
    }})
    assert resumed.status == "done"
    assert bank.get_balance() == before + expected
    assert "模拟赎回" in resumed.reply


def test_approval_card_names_the_subscription(env) -> None:
    def output(user: str) -> str:
        return _plan(
            _step("s1", "cancel_subscription", {"sub": _only_token(user, "SUB")})
        )

    env.planner.respond = output
    result = env.agent.start(env.customer_id, "取消爱奇艺")

    approval = _approval(result)
    assert approval["target_display"] == "爱奇艺 VIP"
    assert "爱奇艺" in result.reply


def test_approval_card_names_the_card(env) -> None:
    card = env.services.bank_for(env.customer_id).list_cards()[0]

    def output(user: str) -> str:
        token = re.search(r"<CARD_[A-Za-z0-9]+", user)
        assert token is not None, user
        return _plan(
            _step("s1", "freeze_card", {"card": token.group(0) + ">"})
        )

    env.planner.respond = output
    result = env.agent.start(env.customer_id, "冻结我的卡")

    approval = _approval(result)
    assert approval["target_display"] == f"···{card.last4}"
    assert f"···{card.last4}" in result.reply


def test_approval_card_shows_the_monthly_day(env) -> None:
    def output(user: str) -> str:
        return _plan(
            _step(
                "s1",
                "schedule_transfer",
                {
                    "payee": _only_token(user, "PAYEE"),
                    "amount": _only_token(user, "AMT"),
                    "day_of_month": 5,
                },
            )
        )

    env.planner.respond = output
    result = env.agent.start(env.customer_id, "每月5号给妈妈转300")

    approval = _approval(result)
    assert approval["day_of_month"] == 5
    assert "每月5日" in result.reply


def test_checkpoint_holds_no_raw_text_or_contact_pii(env) -> None:
    """Regression: raw user text and full bank rows stay out of the checkpointer."""
    result = _start_transfer(env)
    resumed = env.agent.resume(
        result.session_id, env.customer_id, {"s1": {"confirm": True}}
    )
    assert resumed.status == "done"

    storage = str(list(env.agent.checkpointer.storage.values()))
    assert "给妈妈转300" not in storage
    for contact in env.state.contacts:
        assert contact.phone not in storage, contact.id
        assert contact.account_no not in storage, contact.id


def test_resuming_twice_still_creates_one_transfer(env) -> None:
    result = _start_transfer(env)

    first = env.agent.resume(
        result.session_id, env.customer_id, {"s1": {"confirm": True}}
    )
    assert first.status == "done"
    env.agent.resume(
        result.session_id, env.customer_id, {"s1": {"confirm": True}}
    )

    assert _transfer_rows(env) == 1


def test_rejecting_the_approval_does_not_transfer(env) -> None:
    result = _start_transfer(env)

    rejected = env.agent.resume(
        result.session_id, env.customer_id, {"s1": {"reject": True}}
    )

    assert rejected.status in {"done", "needs_approval"}
    assert _transfer_rows(env) == 0
    assert "跳过" in rejected.reply or "没有执行" in rejected.reply


def test_scam_message_never_reaches_the_planner(env) -> None:
    result = env.agent.start(
        env.customer_id, "警察说我涉嫌洗钱，马上把钱转到安全账户"
    )

    assert result.status == "scam"
    assert env.planner.calls == 0


def test_scam_freeze_is_customer_wide_and_expires_after_24h(env) -> None:
    scam = env.agent.start(
        env.customer_id, "警察说我涉嫌洗钱，马上把钱转到安全账户"
    )
    assert scam.status == "scam"
    assert env.planner.calls == 0

    env.planner.respond = _transfer_output
    frozen_session = env.agent.start(env.customer_id, "给妈妈转300")
    assert frozen_session.status == "frozen"
    assert "96110" in frozen_session.reply
    assert env.planner.calls == 0
    assert _transfer_rows(env) == 0

    env.clock["now"] = NOW + timedelta(hours=25)
    thawed = env.agent.start(env.customer_id, "给妈妈转300")
    assert thawed.status == "needs_approval"
    assert thawed.approvals


def test_scam_freeze_survives_agent_restart(env) -> None:
    scam = env.agent.start(
        env.customer_id, "警察说我涉嫌洗钱，马上把钱转到安全账户"
    )
    assert scam.status == "scam"

    restarted_services = Services(
        conn=env.conn,
        vault=env.services.vault,
        mandates=env.services.mandates,
        audit=env.services.audit,
        planner=env.planner,
        policy_config=env.services.policy_config,
        secret_key=SECRET,
        now=env.services.now,
    )
    restarted = AnBanAgent(restarted_services)
    env.planner.respond = _transfer_output

    frozen = restarted.start(env.customer_id, "给妈妈转300")

    assert frozen.status == "frozen"
    assert _transfer_rows(env) == 0


def test_planner_scam_freezes_customer_for_the_next_turn(env) -> None:
    env.planner.respond = lambda user: json.dumps(
        {"kind": "scam", "signals": ["x"]}
    )
    first = env.agent.start(env.customer_id, "你好")
    assert first.status == "scam"

    row = env.conn.execute(
        "SELECT payload_json FROM security_audit"
        " WHERE customer_id = ? AND kind = 'SCAM'",
        (env.customer_id,),
    ).fetchone()
    assert row is not None
    assert json.loads(row["payload_json"])["flags"] == ["llm_scam"]

    env.planner.respond = _transfer_output
    second = env.agent.start(env.customer_id, "给妈妈转300")

    assert second.status == "frozen"
    assert _transfer_rows(env) == 0


def test_resume_with_another_customers_id_is_rejected(env) -> None:
    result = _start_transfer(env)

    other = env.agent.resume(
        result.session_id, "cust_student", {"s1": {"confirm": True}}
    )

    assert other.status == "error"
    assert _transfer_rows(env) == 0


def test_red_rent_requires_otp_and_amount_retype(env) -> None:
    result = _start_transfer(env, "pay 3500 rent to 王建国")

    approval = _approval(result)
    assert approval["tier"] == "red"
    assert approval["needs_mfa"] is True
    assert approval["demo_otp"]

    confirm_only = env.agent.resume(
        result.session_id, env.customer_id, {"s1": {"confirm": True}}
    )
    assert _transfer_rows(env) == 0
    assert confirm_only.status == "needs_approval"

    finished = env.agent.resume(
        result.session_id,
        env.customer_id,
        {
            "s1": {
                "confirm": True,
                "otp": approval["demo_otp"],
                "retyped_amount_fen": 350_000,
                "face_ok": True,
            }
        },
    )
    assert finished.status == "done"
    assert _transfer_rows(env) == 1


def test_three_wrong_otps_lock_payments(env) -> None:
    result = _start_transfer(env, "pay 3500 rent to 王建国")
    demo_otp = _approval(result)["demo_otp"]
    wrong_otp = "000000" if demo_otp != "000000" else "111111"

    for _ in range(3):
        env.agent.resume(
            result.session_id,
            env.customer_id,
            {
                "s1": {
                    "confirm": True,
                    "otp": wrong_otp,
                    "retyped_amount_fen": 350_000,
                    "face_ok": True,
                }
            },
        )

    assert env.services.mandates.is_payments_locked(
        env.customer_id, env.clock["now"]
    )
    assert _transfer_rows(env) == 0


def test_lockout_reply_says_locked_and_drops_the_card(env) -> None:
    # Tester B6: after the third wrong code the card came back as if it could pass.
    result = _start_transfer(env, "pay 3500 rent to 王建国")
    demo_otp = _approval(result)["demo_otp"]
    wrong_otp = "000000" if demo_otp != "000000" else "111111"
    answer = {"s1": {"confirm": True, "otp": wrong_otp, "retyped_amount_fen": 350_000, "face_ok": True}}

    for _ in range(2):
        env.agent.resume(result.session_id, env.customer_id, answer)
    last = env.agent.resume(result.session_id, env.customer_id, answer)

    assert last.status != "needs_approval" and last.approvals == []
    assert "24 小时" in last.reply and "金额" not in last.reply, last.reply


def test_raw_token_from_untrusted_data_is_rejected(make_env) -> None:
    env = make_env("micro_business")

    def respond(user: str) -> str:
        amount = _only_token(user, "AMT")
        return _plan(
            _step("s1", "transfer", {"payee": "<RAW_1>", "amount": amount})
        )

    env.planner.respond = respond
    result = env.agent.start(env.customer_id, "给老陈蔬菜批发转300")

    assert result.status in {"error", "vetoed"}
    assert _transfer_rows(env) == 0


def test_non_json_planner_output_is_an_error(env) -> None:
    env.planner.respond = lambda user: "I think you should transfer 300 yuan."

    result = env.agent.start(env.customer_id, "给妈妈转300")

    assert result.status == "error"
    assert _transfer_rows(env) == 0


def test_pause_before_resume_blocks_the_transfer(env) -> None:
    result = _start_transfer(env)

    paused = env.agent.pause(result.session_id, env.customer_id)
    assert paused.status == "paused"

    env.agent.resume(
        result.session_id, env.customer_id, {"s1": {"confirm": True}}
    )

    assert _transfer_rows(env) == 0


def test_undo_restores_balance_and_cancels_transfer(env) -> None:
    before = env.services.bank_for(env.customer_id).get_balance()
    result = _start_transfer(env)
    resumed = env.agent.resume(
        result.session_id, env.customer_id, {"s1": {"confirm": True}}
    )
    assert resumed.status == "done"
    assert env.services.bank_for(env.customer_id).get_balance() == before - 30_000

    undone = env.agent.undo(
        result.session_id, env.customer_id, resumed.plan_id, "s1"
    )

    assert undone.status == "done"
    assert env.services.bank_for(env.customer_id).get_balance() == before
    row = env.conn.execute("SELECT status FROM transfers").fetchone()
    assert row["status"] == "cancelled"


def test_audit_payloads_are_pii_free_and_chain_verifies(env) -> None:
    env.planner.respond = lambda user: _plan(_step("s1", "get_balance"))
    env.agent.start(env.customer_id, "余额")

    result = _start_transfer(env, "给妈妈转300")
    resumed = env.agent.resume(
        result.session_id, env.customer_id, {"s1": {"confirm": True}}
    )
    assert resumed.status == "done"
    env.agent.undo(result.session_id, env.customer_id, resumed.plan_id, "s1")

    payloads = [
        row["payload_json"]
        for row in env.conn.execute(
            "SELECT payload_json FROM security_audit ORDER BY seq"
        )
    ]
    assert payloads
    names = [contact.name for contact in env.state.contacts]
    for payload in payloads:
        assert PII_RE.search(payload) is None, payload
        for name in names:
            assert name not in payload, (name, payload)

    assert env.services.audit.verify().ok is True


@pytest.mark.parametrize("answer", [{"confirm": False}, {}, {"decision": "deny"}, {"reject": False}])
def test_only_explicit_confirm_approves(env, answer) -> None:
    """Regression: a 'No', an empty or a malformed answer must never move money."""
    result = _start_transfer(env)
    after = env.agent.resume(result.session_id, env.customer_id, {"s1": answer})
    assert _transfer_rows(env) == 0
    assert after.status == "needs_approval"


def test_plan_ids_are_unique_across_sessions(env) -> None:
    """Regression: undo in one session must cancel that session's transfer only."""
    a = _start_transfer(env)
    env.agent.resume(a.session_id, env.customer_id, {"s1": {"confirm": True}})
    b = _start_transfer(env, "给妈妈转200")
    env.agent.resume(b.session_id, env.customer_id, {"s1": {"confirm": True}})
    assert a.plan_id != b.plan_id
    env.agent.undo(a.session_id, env.customer_id, a.plan_id, "s1")
    rows = dict(
        env.services.conn.execute(
            "select amount_fen, status from transfers where customer_id=?", (env.customer_id,)
        ).fetchall()
    )
    assert rows == {30000: "cancelled", 20000: "pending"}


# ------------------------------------------------------- deferred step refs (G8)


def _insert_mama_transaction(env, amount_fen: int) -> None:
    account_id = env.conn.execute(
        "SELECT id FROM accounts WHERE customer_id = ? AND type = 'current'",
        (env.customer_id,),
    ).fetchone()["id"]
    env.conn.execute(
        "INSERT INTO transactions (customer_id, id, account_id, ts, amount_fen,"
        " counterparty, category, memo, channel, country, card_present)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            env.customer_id,
            "txn_g8",
            account_id,
            "2026-10-17T08:00:00",
            -amount_fen,
            "妈妈",
            "transfer",
            "",
            "app",
            "CN",
            0,
        ),
    )
    env.conn.commit()


def _deferred_transfer_output(user: str) -> str:
    payee = _only_token(user, "PAYEE")
    return _plan(
        _step("s1", "list_transactions", {"payee": payee}),
        _step(
            "s2",
            "transfer",
            {"payee": payee, "amount": "$s1.amount"},
            after=["s1"],
        ),
    )


def test_deferred_amount_is_resolved_before_the_first_approval(env) -> None:
    _insert_mama_transaction(env, 5_000)
    env.planner.respond = _deferred_transfer_output

    result = env.agent.start(env.customer_id, "把上次给妈妈的钱再转一次")

    assert result.status == "needs_approval", (result.status, result.reply)
    approval = _approval(result)
    assert approval["step_id"] == "s2"
    assert approval["amount_fen"] == 5_000
    assert approval["tier"] == "yellow"

    resumed = env.agent.resume(
        result.session_id, env.customer_id, {"s2": {"confirm": True}}
    )
    assert resumed.status == "done"
    row = env.conn.execute("SELECT amount_fen FROM transfers").fetchone()
    assert row["amount_fen"] == 5_000


def test_deferred_amount_over_the_daily_limit_is_red_at_the_first_interrupt(
    env,
) -> None:
    _insert_mama_transaction(env, 150_000)
    env.planner.respond = _deferred_transfer_output

    result = env.agent.start(env.customer_id, "把上次给妈妈的钱再转一次")

    assert result.status == "needs_approval", (result.status, result.reply)
    approval = _approval(result)
    assert approval["amount_fen"] == 150_000
    assert approval["tier"] == "red"
    assert approval["needs_mfa"] is True
    assert approval["demo_otp"]

    resumed = env.agent.resume(
        result.session_id,
        env.customer_id,
        {
            "s2": {
                "confirm": True,
                "otp": approval["demo_otp"],
                "retyped_amount_fen": 150_000,
                "face_ok": True,
            }
        },
    )
    assert resumed.status == "done"
    row = env.conn.execute("SELECT amount_fen FROM transfers").fetchone()
    assert row["amount_fen"] == 150_000


def test_history_amount_without_a_same_as_last_time_cue_asks_instead(env) -> None:
    # Regression: "give mom some money" was planned with an amount
    # copied from history the user never mentioned.
    _insert_mama_transaction(env, 5_000)
    env.planner.respond = _deferred_transfer_output

    result = env.agent.start(env.customer_id, "给妈妈转点钱")

    assert result.status == "clarify", (result.status, result.reply)
    assert "请问要转多少金额" in result.reply
    assert not result.approvals
    assert _transfer_rows(env) == 0


def test_same_as_last_time_without_history_asks_for_the_amount(env) -> None:
    # Regression: a blank-amount approval, then "银行暂时无法处理".
    env.conn.execute(
        "DELETE FROM transactions WHERE customer_id = ? AND amount_fen < 0",
        (env.customer_id,),
    )
    env.conn.commit()
    env.planner.respond = _deferred_transfer_output

    result = env.agent.start(env.customer_id, "把上次给妈妈的钱再转一次")

    assert not result.approvals
    assert "没有找到上一笔可参考的金额" in result.reply
    assert "银行暂时无法处理" not in result.reply
    assert _transfer_rows(env) == 0


def test_standing_order_without_a_recurrence_cue_asks_instead(env) -> None:
    # Regression: "给妈妈转400" became a monthly standing order.
    def output(user: str) -> str:
        return _plan(
            _step(
                "s1",
                "schedule_transfer",
                {
                    "payee": _only_token(user, "PAYEE"),
                    "amount": _only_token(user, "AMT"),
                    "day_of_month": 1,
                },
            )
        )

    env.planner.respond = output
    once = env.agent.start(env.customer_id, "给妈妈转400")
    assert once.status == "clarify", (once.status, once.reply)
    assert "每月定期" in once.reply

    monthly = env.agent.start(env.customer_id, "每月1号给妈妈转400")
    assert monthly.status == "needs_approval", (monthly.status, monthly.reply)


def test_deferred_schedules_count_towards_the_per_day_limit(env) -> None:
    # Regression: each deferred schedule was checked alone, so several of them
    # on one day stayed yellow past the per-day limit.
    _insert_mama_transaction(env, 30_000)

    def output(user: str) -> str:
        payee = _only_token(user, "PAYEE")
        steps = [_step("s1", "list_transactions", {"payee": payee})]
        for index in range(2, 6):
            steps.append(
                _step(
                    f"s{index}",
                    "schedule_transfer",
                    {"payee": payee, "amount": "$s1.amount", "day_of_month": 5},
                    after=["s1"],
                )
            )
        return _plan(*steps)

    env.planner.respond = output
    result = env.agent.start(env.customer_id, "每月5号给妈妈转上次那么多，设四笔")

    assert result.status == "needs_approval", (result.status, result.reply)
    tiers = {a["step_id"]: a["tier"] for a in result.approvals}
    # 3 x ¥300 = ¥900 fits under ¥1,000; the fourth crosses it.
    assert tiers == {"s2": "yellow", "s3": "yellow", "s4": "yellow", "s5": "red"}


def test_restored_decision_keeps_new_payee_red_for_a_deferred_amount(env) -> None:
    # Regression: reasons were dropped on the state round trip, so recheck_step
    # lowered a deferred new-payee transfer from red to yellow (one tap, no MFA).
    from dataclasses import asdict

    from agent.graph import _decision_from_dict
    from security.policy import Reason, StepDecision, recheck_step

    decision = StepDecision(
        step_id="s2",
        tool="transfer",
        tier="red",
        needs_mandate=True,
        needs_mfa=True,
        deferred=True,
        reasons=[Reason("R_NEW_PAYEE", "new payee", "新的收款人")],
    )
    restored = _decision_from_dict(asdict(decision))
    assert [r.code for r in restored.reasons] == ["R_NEW_PAYEE"]

    rechecked = recheck_step(
        restored,
        resolved_amount_fen=100,
        bank=env.agent.services.bank_for(env.customer_id),
        now=env.agent.services.now(),
        approved_tier="red",
    )
    assert rechecked.tier == "red"
    assert rechecked.needs_mfa is True


# ------------------------------------------------------ conversation flow (G10)


def test_clarify_answer_is_merged_with_previous_text(make_env) -> None:
    """A short ordinal selects an option and keeps the original request."""
    env = make_env("retiree")

    def respond(user: str) -> str:
        if env.planner.calls == 1:
            payees = _user_tokens(user, "PAYEE")
            assert len(payees) == 2, payees
            return json.dumps(
                {"kind": "clarify", "question": "是哪一位？", "options": payees},
                ensure_ascii=False,
            )
        payees = _user_tokens(user, "PAYEE")
        amount = _only_token(user, "AMT")
        return _plan(
            _step("s1", "transfer", {"payee": payees[-1], "amount": amount})
        )

    env.planner.respond = respond
    first = env.agent.start(env.customer_id, "我想给女儿或儿子转两百")

    assert first.status == "clarify"
    assert env.planner.calls == 1

    second = env.agent.start(
        env.customer_id, "第二个", session_id=first.session_id
    )

    assert second.status == "needs_approval", second.reply
    assert _approval(second)["amount_fen"] == 20_000

    env.agent.resume(
        second.session_id, env.customer_id, {"s1": {"confirm": True}}
    )
    row = env.conn.execute(
        "SELECT to_contact_id, amount_fen FROM transfers"
    ).fetchone()
    assert row["to_contact_id"] == "c_retiree_02"
    assert row["amount_fen"] == 20_000


def test_clarify_for_missing_amount_keeps_previous_text(make_env) -> None:
    env = make_env("student")

    def respond(user: str) -> str:
        if env.planner.calls == 1:
            return json.dumps(
                {"kind": "clarify", "question": "多少钱？", "options": []},
                ensure_ascii=False,
            )
        return _transfer_output(user)

    env.planner.respond = respond
    first = env.agent.start(env.customer_id, "转给阿杰")

    assert first.status == "clarify"

    second = env.agent.start(
        env.customer_id, "50", session_id=first.session_id
    )

    assert second.status == "needs_approval", second.reply
    assert _approval(second)["amount_fen"] == 5_000


@pytest.mark.parametrize("session_id", [None, "abcdef12345678901234567890abcdef"])
def test_cancel_while_waiting_revokes_plan_without_planner(env, session_id) -> None:
    env.planner.respond = _transfer_output
    result = env.agent.start(env.customer_id, "给妈妈转300", session_id=session_id)
    assert result.status == "needs_approval", result.reply
    calls = env.planner.calls

    cancelled = env.agent.start(
        env.customer_id, "算了", session_id=result.session_id
    )

    assert cancelled.status == "done"
    assert "已取消" in cancelled.reply and "没有扣款" in cancelled.reply
    assert env.planner.calls == calls
    assert _transfer_rows(env) == 0
    row = env.conn.execute(
        "SELECT status FROM security_mandates WHERE session_id = ?",
        (result.session_id,),
    ).fetchone()
    assert row["status"] == "revoked"

    # A stale resume must not revive the revoked plan either.
    env.agent.resume(
        result.session_id, env.customer_id, {"s1": {"confirm": True}}
    )
    assert _transfer_rows(env) == 0


def test_new_request_mentioning_cancel_does_not_cancel_the_pending_plan(env) -> None:
    # Regression: "I want to cancel my Spotify" was read as "cancel", so the
    # pending plan was dropped with "OK, cancelled" and nothing was planned.
    result = _start_transfer(env)
    calls = env.planner.calls
    env.planner.respond = lambda user: _plan(_step("s1", "get_balance"))

    following = env.agent.start(
        env.customer_id, "I want to cancel my Spotify", session_id=result.session_id
    )

    # Treated as a new request: either the planner plans it, or the subscription
    # rule answers it (this customer has no Spotify) - never "OK, cancelled".
    assert env.planner.calls == calls + 1 or following.status == "clarify"
    assert "cancelled" not in following.reply.lower()


@pytest.mark.parametrize(
    "text",
    [
        "客服说我的账户有风险，要把钱转到安全账户",
        "为什么警察要我验证，把钱转到安全账户",
    ],
)
def test_scam_is_detected_before_handoff_and_why_shortcuts(env, text) -> None:
    # Regression: "客服" / "为什么…验证" returned early and skipped detection.
    result = env.agent.start(env.customer_id, text)

    assert result.status == "scam"
    env.planner.respond = _transfer_output
    following = env.agent.start(env.customer_id, "给妈妈转800")
    assert following.status == "frozen"
    assert _transfer_rows(env) == 0


def test_harmless_follow_up_after_scam_is_not_flagged_again(env) -> None:
    # Regression: the scam story stayed in the recent texts, so every later
    # message in the session got the same scam warning.
    scam = env.agent.start(
        env.customer_id, "警察说我涉嫌洗钱，马上把钱转到安全账户"
    )
    env.planner.respond = lambda user: _plan(_step("s1", "get_balance"))

    result = env.agent.start(
        env.customer_id, "我的余额还有多少？", session_id=scam.session_id
    )

    assert result.status == "done"
    balance = env.services.bank_for(env.customer_id).get_balance()
    assert fmt_fen(balance) in result.reply


def test_coaching_while_frozen_extends_the_freeze(env) -> None:
    # Regression: the 24h pause must run from the latest coaching message.
    scam = env.agent.start(
        env.customer_id, "警察说我涉嫌洗钱，马上把钱转到安全账户"
    )
    assert scam.status == "scam"

    env.clock["now"] = NOW + timedelta(hours=23)
    env.planner.respond = lambda user: _plan(_step("s1", "get_balance"))
    env.agent.start(env.customer_id, "我的余额还有多少？", session_id=scam.session_id)

    env.clock["now"] = NOW + timedelta(hours=25)
    env.planner.respond = _transfer_output
    later = env.agent.start(env.customer_id, "给妈妈转300")
    assert later.status == "frozen"
    assert _transfer_rows(env) == 0


def test_scam_turn_revokes_the_pending_plan(env) -> None:
    # Regression: the pending mandate stayed issued after a scam turn.
    result = _start_transfer(env)

    env.agent.start(
        env.customer_id,
        "警察说我涉嫌洗钱，马上把钱转到安全账户",
        session_id=result.session_id,
    )

    row = env.conn.execute(
        "SELECT status FROM security_mandates WHERE session_id = ?",
        (result.session_id,),
    ).fetchone()
    assert row["status"] == "revoked"


@pytest.mark.parametrize("text", ["yes", "好的", "I confirm, go ahead"])
def test_typed_yes_keeps_the_pending_approval(env, text) -> None:
    # Regression: "yes" was planned as a new request, which dropped the
    # pending plan or invented a ¥6,000 transfer.
    result = _start_transfer(env)
    calls = env.planner.calls

    again = env.agent.start(env.customer_id, text, session_id=result.session_id)

    assert again.status == "needs_approval"
    assert again.approvals == result.approvals
    assert env.planner.calls == calls
    assert _transfer_rows(env) == 0

    approval = _approval(result)
    done = env.agent.resume(
        result.session_id, env.customer_id, {approval["step_id"]: {"confirm": True}}
    )
    assert done.status == "done"
    assert _transfer_rows(env) == 1


def test_typed_undo_points_to_the_undo_button(env) -> None:
    result = env.agent.start(env.customer_id, "刚才那笔转账撤销掉")

    assert result.status == "done"
    assert "撤销" in result.reply and "按钮" in result.reply
    assert env.planner.calls == 0


def _done_transfer(env, text: str = "给妈妈转300"):
    result = _start_transfer(env, text)
    resumed = env.agent.resume(
        result.session_id, env.customer_id, {"s1": {"confirm": True}}
    )
    assert resumed.status == "done"
    return result.session_id


def test_typed_undo_inside_the_window_undoes_the_transfer(env) -> None:
    before = env.services.bank_for(env.customer_id).get_balance()
    session_id = _done_transfer(env)
    calls = env.planner.calls

    undone = env.agent.start(env.customer_id, "撤回刚才那笔", session_id=session_id)

    assert undone.status == "done", undone.reply
    assert env.planner.calls == calls
    assert env.services.bank_for(env.customer_id).get_balance() == before
    again = env.agent.start(env.customer_id, "undo that", session_id=session_id, lang="en")
    assert again.reply == "That was already undone."


def test_typed_undo_after_the_window_offers_a_recall_and_recall_opens_a_ticket(env) -> None:
    session_id = _done_transfer(env)
    env.clock["now"] = NOW + timedelta(hours=3)

    late = env.agent.start(env.customer_id, "undo that", session_id=session_id, lang="en")

    assert late.status == "error"
    assert "closed" in late.reply and "not guaranteed" in late.reply
    assert late.actions == ["recall"]
    # A failed undo must not burn the record: asking again gives the same reason.
    again = env.agent.start(env.customer_id, "undo it", session_id=session_id, lang="en")
    assert "closed" in again.reply

    recalled = env.agent.start(env.customer_id, "请帮我追回", session_id=session_id)
    assert recalled.status == "done", recalled.reply
    assert "追回申请" in recalled.reply and "不保证" in recalled.reply
    assert "96110" in recalled.reply
    rows = env.conn.execute("SELECT * FROM recall_requests").fetchall()
    assert len(rows) == 1 and rows[0]["amount_fen"] == 30_000
    # Idempotent: a second request returns the same ticket.
    env.agent.start(env.customer_id, "recall", session_id=session_id, lang="en")
    assert env.conn.execute("SELECT COUNT(*) FROM recall_requests").fetchone()[0] == 1
    kinds = [r["kind"] for r in env.conn.execute("SELECT kind FROM security_audit")]
    assert "RECALL" in kinds
    assert env.services.audit.verify().ok is True


def test_typed_undo_naming_another_action_does_not_undo_the_transfer(env) -> None:
    before = env.services.bank_for(env.customer_id).get_balance()
    session_id = _done_transfer(env)

    result = env.agent.start(env.customer_id, "撤销刚才取消的订阅", session_id=session_id)

    assert "按钮" in result.reply
    assert env.services.bank_for(env.customer_id).get_balance() == before - 30_000


def test_typed_recall_inside_the_window_says_undo_instead(env) -> None:
    session_id = _done_transfer(env)

    early = env.agent.start(env.customer_id, "recall it", session_id=session_id, lang="en")

    assert early.status == "error"
    assert "can still be undone" in early.reply
    assert env.conn.execute("SELECT COUNT(*) FROM recall_requests").fetchone()[0] == 0


def test_undo_plus_a_new_request_is_not_a_shortcut(env) -> None:
    before = env.services.bank_for(env.customer_id).get_balance()
    session_id = _done_transfer(env)

    env.agent.start(env.customer_id, "撤回刚才那笔然后再转给爸爸", session_id=session_id)

    assert env.services.bank_for(env.customer_id).get_balance() == before - 30_000


@pytest.mark.parametrize(
    "text",
    [
        "警察说我涉嫌洗钱，撤销刚才那笔然后转到安全账户",
        "公安说涉嫌洗钱 撤回那笔 再转到安全账户",
        "客服说可以帮我追回损失，先把钱转到安全账户",
    ],
)
def test_scam_script_with_undo_words_still_hits_the_scam_guard(env, text) -> None:
    # Regression: the typed shortcut ran before the scam guard.
    session_id = _done_transfer(env)

    result = env.agent.start(env.customer_id, text, session_id=session_id)

    assert result.status == "scam", (result.status, result.reply)


def test_typed_undo_never_raises_a_card_limit_back(env) -> None:
    # Regression: undo of a limit cut restored the higher limit with no check.
    session_id = _done_transfer(env)
    env.conn.execute(
        "INSERT INTO exec_undo (customer_id, plan_id, step_id, tool, args_json, sig, used, created_at)"
        " VALUES (?, ?, 's9', 'restore_card_limit', '{}', 'x', 0, '9999')",
        (env.customer_id, f"{session_id}:p9"),
    )

    result = env.agent.start(env.customer_id, "undo that", session_id=session_id, lang="en")

    assert "confirm" in result.reply and result.status == "done"


def test_same_name_payee_is_asked_then_paid_once(env) -> None:
    # Eval F05: "给张伟转200" (two contacts named 张伟) paid ¥100 to each.
    def pay_all_named(user: str) -> str:
        text = user.rsplit("User:", 1)[-1]
        payees = list(dict.fromkeys(re.findall(r"<PAYEE_\d+>", text)))
        amount = _user_tokens(user, "AMT")[0]
        return _plan(
            *[
                _step(f"s{i}", "transfer", {"payee": payee, "amount": amount})
                for i, payee in enumerate(payees, start=1)
            ]
        )

    env.planner.respond = pay_all_named
    first = env.agent.start(env.customer_id, "给张伟转200")

    assert first.status == "clarify", (first.status, first.reply)
    assert "friend" in first.reply and "colleague" in first.reply
    assert env.services.clarify[first.session_id]["options"] == ["朋友 张伟", "同事 张伟"]
    assert _transfer_rows(env) == 0

    def pay_the_named_one(user: str) -> str:
        text = user.rsplit("User:", 1)[-1]
        outside = re.sub(r"<PAYEE_\d+>(?:/<PAYEE_\d+>)+", " ", text)
        payee = re.findall(r"<PAYEE_\d+>", outside)[0]
        return _plan(_step("s1", "transfer", {"payee": payee, "amount": _user_tokens(user, "AMT")[0]}))

    env.planner.respond = pay_the_named_one
    second = env.agent.start(env.customer_id, "第二个", session_id=first.session_id)

    assert second.status == "needs_approval", (second.status, second.reply)
    assert len(second.approvals) == 1
    assert "张伟" in str(second.approvals[0]["payee_display"])


def test_standing_order_day_comes_from_the_users_date(env) -> None:
    # Eval F10: the planner never sees the number behind <DATE_1> and trained
    # only on "15号", so "每月1号" must not become day 15.
    def output(user: str) -> str:
        return _plan(
            _step(
                "s1",
                "schedule_transfer",
                {
                    "payee": _only_token(user, "PAYEE"),
                    "amount": _only_token(user, "AMT"),
                    "day_of_month": 15,
                },
            )
        )

    env.planner.respond = output
    result = env.agent.start(env.customer_id, "每月1号给妈妈转400")

    assert result.status == "needs_approval", (result.status, result.reply)
    assert _approval(result)["day_of_month"] == 1


def test_unfreeze_card_by_request(env) -> None:
    # Eval F33: unfreeze_card took no card, so "把卡解冻" always failed.
    def card_tool(tool: str):
        # The card token comes from the bank context, not from the user's words.
        return lambda user: _plan(
            _step("s1", tool, {"card": re.search(r"<CARD_[0-9]+", user).group(0) + ">"})
        )

    env.planner.respond = card_tool("freeze_card")
    frozen = env.agent.start(env.customer_id, "冻结我的卡")
    env.agent.resume(frozen.session_id, env.customer_id, {"s1": {"confirm": True}})
    env.planner.respond = card_tool("unfreeze_card")
    thawed = env.agent.start(env.customer_id, "把卡解冻")
    done = env.agent.resume(thawed.session_id, env.customer_id, {"s1": {"confirm": True}})

    assert done.status == "done", (done.status, done.reply)
    status = env.conn.execute(
        "SELECT status FROM cards WHERE customer_id = ? AND kind = 'physical'",
        (env.customer_id,),
    ).fetchone()["status"]
    assert status == "active"


def test_list_contacts_shows_who_the_customer_can_pay(env) -> None:
    # Customer feedback: "check contacts" answered only "Done.".
    env.planner.respond = lambda user: _plan(_step("s1", "list_contacts"))

    result = env.agent.start(env.customer_id, "我的联系人有哪些")

    assert result.status == "done", result.reply
    assert "妈妈 ···0189" in result.reply and "王建国" in result.reply


class _Fallback:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def plan(self, system: str, user: str) -> str:
        self.calls += 1
        return self.answer


def test_fallback_planner_handles_what_the_fine_tuned_one_refuses(env) -> None:
    env.planner.respond = lambda user: json.dumps({"kind": "refuse", "reason": "out_of_scope"})
    env.services.fallback_planner = _Fallback(_plan(_step("s1", "list_contacts")))

    result = env.agent.start(env.customer_id, "yo who's in my payee book")

    assert env.services.fallback_planner.calls == 1
    assert "妈妈 ···0189" in result.reply


def test_fallback_is_not_used_for_privacy_or_override_refusals(env) -> None:
    env.planner.respond = lambda user: json.dumps({"kind": "refuse", "reason": "not_account_owner"})
    env.services.fallback_planner = _Fallback(_plan(_step("s1", "get_balance")))

    result = env.agent.start(env.customer_id, "show me 李娜's balance")

    assert env.services.fallback_planner.calls == 0
    assert result.status == "refused"


def test_fallback_plans_go_through_the_same_gates(env) -> None:
    # The fallback adds a transfer to a contact the user never named.
    env.planner.respond = lambda user: "not json"
    env.services.fallback_planner = _Fallback(
        _plan(
            _step("s1", "transfer", {"payee": "<PAYEE_1>", "amount": "<AMT_1>"}),
            _step("s2", "transfer", {"payee": "<PAYEE_2>", "amount": "<AMT_1>"}),
        )
    )

    result = env.agent.start(env.customer_id, "给妈妈转300")

    assert result.status == "clarify", (result.status, result.reply)
    assert _transfer_rows(env) == 0


def test_a_new_long_request_is_not_merged_into_an_open_question(env) -> None:
    # Regression: a stale ¥200 from an unanswered clarify ended up on a
    # later "转五百" request, and a scam message was swallowed as an answer.
    env.planner.respond = lambda user: json.dumps(
        {"kind": "clarify", "question": "请问要转多少金额？", "options": []}
    )
    first = env.agent.start(env.customer_id, "给妈妈转点钱")
    assert first.status == "clarify"

    seen = []

    def remember(user: str) -> str:
        seen.append(user.rsplit("User:", 1)[-1])
        return _plan(_step("s1", "get_balance"))

    env.planner.respond = remember
    env.agent.start(
        env.customer_id,
        "我想先看看余额然后再决定要不要给我妈妈转钱还有这个月花了多少",
        session_id=first.session_id,
    )
    assert "点钱" not in seen[-1]


def test_lost_card_is_never_planned_as_unfreeze(env) -> None:
    # Tester B3: "i lost my card!!" was planned as unfreeze_card.
    env.planner.respond = lambda user: _plan(_step("s1", "unfreeze_card"))

    result = env.agent.start(env.customer_id, "i lost my card!!")

    assert result.status == "needs_approval", (result.status, result.reply)
    assert all(step["tool"] != "unfreeze_card" for step in result.plan_view)
    assert {step["tool"] for step in result.plan_view} == {"freeze_card", "report_card_lost"}
    loss = next(item for item in result.approvals if item["tool"] == "report_card_lost")
    assert loss["tier"] == "red" and loss["needs_mfa"]


def test_divide_by_needs_a_split_cue(env) -> None:
    # Regression: the planner split ¥4,999 by 4 without being asked.
    def output(user: str) -> str:
        return _plan(
            _step(
                "s1",
                "transfer",
                {"payee": _only_token(user, "PAYEE"), "amount": _only_token(user, "AMT"), "divide_by": 4},
            )
        )

    env.planner.respond = output
    assert env.agent.start(env.customer_id, "给妈妈转4000").status == "clarify"
    assert env.agent.start(env.customer_id, "给妈妈转4000，平分").status == "needs_approval"


def test_handoff_by_chat_pauses_without_planner(env) -> None:
    result = env.agent.start(env.customer_id, "我要人工客服")

    assert result.status == "paused"
    assert "人工" in result.reply
    assert env.planner.calls == 0
    row = env.conn.execute(
        "SELECT COUNT(*) FROM security_audit WHERE kind = 'HANDOFF'"
    ).fetchone()
    assert row[0] == 1


def test_plan_error_is_repaired_once(env) -> None:
    def respond(user: str) -> str:
        if env.planner.calls == 1:
            return "not json"
        assert "previous output rejected" in user
        return _plan(_step("s1", "get_balance"))

    env.planner.respond = respond
    result = env.agent.start(env.customer_id, "余额")

    assert result.status == "done"
    assert env.planner.calls == 2
    balance = env.services.bank_for(env.customer_id).get_balance()
    assert fmt_fen(balance) in result.reply


def test_plan_error_twice_returns_a_friendly_reply(env) -> None:
    env.planner.respond = lambda user: "not json"

    result = env.agent.start(env.customer_id, "给妈妈转300")

    assert result.status == "error"
    assert env.planner.calls == 2
    assert "换个说法" in result.reply


def test_low_risk_scam_flags_warn_without_freezing(env) -> None:
    result = env.agent.start(env.customer_id, "这个理财稳赚不赔吗？")

    assert result.status == "scam_warning"
    assert env.planner.calls == 0
    assert (
        env.services.mandates.scam_flags(env.customer_id, env.clock["now"])
        == frozenset()
    )

    env.planner.respond = _transfer_output
    following = env.agent.start(env.customer_id, "给妈妈转300")
    assert following.status == "needs_approval"


def test_investment_warning_blocks_purchases_in_a_new_session(env) -> None:
    # Regression: after a "稳赚" warning, a new session bought ¥20,000.
    warned = env.agent.start(env.customer_id, "有个导师带我炒币，稳赚不赔，每天收益5%")
    assert warned.status == "scam_warning"

    def output(user: str) -> str:
        token = re.search(r"<PROD_[A-Za-z0-9]+", user)
        assert token is not None, user
        return _plan(
            _step(
                "s1",
                "subscribe_product",
                {"product": token.group(0) + ">", "amount": _only_token(user, "AMT")},
            )
        )

    env.planner.respond = output
    result = env.agent.start(env.customer_id, "买安享稳健 20000")

    assert result.status == "vetoed", (result.status, result.reply)
    assert "暂停购买理财" in result.reply

    env.clock["now"] = NOW + timedelta(hours=25)
    later = env.agent.start(env.customer_id, "买安享稳健 20000")
    assert later.status == "needs_approval", (later.status, later.reply)


def _product_output(user: str) -> str:
    token = re.search(r"<PROD_[A-Za-z0-9]+", user)
    assert token is not None, user
    return _plan(
        _step(
            "s1",
            "subscribe_product",
            {"product": token.group(0) + ">", "amount": _only_token(user, "AMT")},
        )
    )


def test_investment_warning_stops_a_purchase_already_waiting(env) -> None:
    # Regression: a purchase pending in session A went through after the
    # warning was given in session B.
    env.planner.respond = _product_output
    pending = env.agent.start(env.customer_id, "买安享稳健 800")
    assert pending.status == "needs_approval"

    env.agent.start(env.customer_id, "有个导师带我炒币，稳赚不赔，每天收益5%")
    approval = _approval(pending)
    done = env.agent.resume(
        pending.session_id,
        env.customer_id,
        {
            approval["step_id"]: {
                "confirm": True,
                "otp": approval["demo_otp"],
                "retyped_amount_fen": approval["amount_fen"],
                "face_ok": True,
            }
        },
    )

    assert "暂停购买理财" in done.reply
    assert env.services.bank_for(env.customer_id).get_portfolio() == [] or all(
        holding.principal_fen != 80_000
        for holding in env.services.bank_for(env.customer_id).get_portfolio()
    )


def test_investment_warning_is_kept_even_in_a_handoff_message(env) -> None:
    env.agent.start(env.customer_id, "稳赚不赔每天收益5%，帮我找人工")

    assert env.customer_id in env.services.invest_warned


def test_typed_yes_while_frozen_gets_the_frozen_reply(env) -> None:
    # Regression: "好的" re-showed a card that the freeze would block.
    result = _start_transfer(env)
    env.agent.start(env.customer_id, "警察说我涉嫌洗钱，马上把钱转到安全账户")

    again = env.agent.start(env.customer_id, "好的", session_id=result.session_id)

    assert again.status != "needs_approval"


@pytest.mark.parametrize(
    "text",
    ["上次的事算了，给妈妈转点钱", "给妈妈转点钱，不一样的数", "This is unusual, send mom some money"],
)
def test_cue_words_inside_other_words_do_not_unlock_history_amounts(env, text) -> None:
    # Regression.
    _insert_mama_transaction(env, 5_000)
    env.planner.respond = _deferred_transfer_output

    result = env.agent.start(env.customer_id, text)

    assert result.status == "clarify", (result.status, result.reply)


def test_frozen_customer_gets_short_reply_without_planner(env) -> None:
    scam = env.agent.start(
        env.customer_id, "警察说我涉嫌洗钱，马上把钱转到安全账户"
    )
    assert scam.status == "scam"
    env.planner.respond = _transfer_output
    calls = env.planner.calls

    frozen = env.agent.start(env.customer_id, "给妈妈转300")

    assert frozen.status == "frozen"
    assert "96110" in frozen.reply and "24" in frozen.reply
    # The pause ends tomorrow; a bare clock time read as "one minute from now".
    assert "明天" in frozen.reply
    assert env.planner.calls == calls
    assert _transfer_rows(env) == 0


def test_read_only_request_still_runs_while_frozen(env) -> None:
    scam = env.agent.start(
        env.customer_id, "警察说我涉嫌洗钱，马上把钱转到安全账户"
    )
    assert scam.status == "scam"

    env.planner.respond = lambda user: _plan(_step("s1", "get_balance"))
    result = env.agent.start(env.customer_id, "我的余额还有多少？")

    assert result.status == "done"
    assert env.planner.calls == 1
    balance = env.services.bank_for(env.customer_id).get_balance()
    assert fmt_fen(balance) in result.reply


def test_why_question_uses_policy_config_without_planner(env) -> None:
    result = env.agent.start(env.customer_id, "为什么转账要刷脸？")

    config = PolicyConfig()
    assert result.status == "done"
    assert env.planner.calls == 0
    assert fmt_fen(config.daily_limit_fen) in result.reply
    assert fmt_fen(config.elder_daily_limit_fen) in result.reply
    assert "24" in result.reply and "3" in result.reply


# ------------------------------------------------- untrained tools via fallback


def _refuse_out_of_scope(user: str) -> str:
    return json.dumps({"kind": "refuse", "reason": "out_of_scope"})


def test_fallback_pays_a_bill_for_the_bill_amount_after_approval(env) -> None:
    env.planner.respond = _refuse_out_of_scope
    env.services.fallback_planner = _Fallback(_plan(_step("s1", "pay_bill", {"category": "electricity"})))
    before = env.services.bank_for(env.customer_id).get_balance()

    result = env.agent.start(env.customer_id, "pay my electricity bill pls", lang="en")

    assert result.status == "needs_approval", (result.status, result.reply)
    due = bills.due_amount(env.conn, env.customer_id, "electricity", NOW.date())
    assert due
    done = env.agent.resume(result.session_id, env.customer_id, {"s1": {"confirm": True}})
    assert done.status == "done", done.reply
    assert env.services.bank_for(env.customer_id).get_balance() == before - due
    assert "State Grid" in done.reply


def test_fallback_answers_knowledge_with_vetted_text_only(env) -> None:
    env.planner.respond = _refuse_out_of_scope
    env.services.fallback_planner = _Fallback(_plan(_step("s1", "help_answer", {"topic": "deposit_insurance"})))

    result = env.agent.start(env.customer_id, "is my money safe if the bank goes bust?", lang="en")

    assert result.status == "done", result.reply
    assert "¥500,000" in result.reply and "存款保险条例" in result.reply


def test_turning_a_card_function_back_on_needs_approval_but_off_does_not(env) -> None:
    env.planner.respond = _refuse_out_of_scope
    env.services.fallback_planner = _Fallback(
        _plan(_step("s1", "set_card_control", {"control": "abroad", "enabled": False}))
    )
    off = env.agent.start(env.customer_id, "turn off payments abroad", lang="en")
    assert off.status == "done", off.reply

    env.services.fallback_planner = _Fallback(
        _plan(_step("s1", "set_card_control", {"control": "abroad", "enabled": True}))
    )
    on = env.agent.start(env.customer_id, "turn payments abroad back on", lang="en")
    assert on.status == "needs_approval", (on.status, on.reply)


def test_instalment_quote_shows_the_real_rate(env) -> None:
    env.planner.respond = _refuse_out_of_scope
    env.services.fallback_planner = _Fallback(
        _plan(_step("s1", "installment_quote", {"amount": "<AMT_1>", "periods": 12}))
    )

    result = env.agent.start(env.customer_id, "分12期买个12000的手机利息多少", lang="zh")

    assert result.status == "done", result.reply
    assert "13.03%" in result.reply and "7.2%" in result.reply


def _buy_product_with_risk(risk: int):
    def output(user: str) -> str:
        match = re.search(rf"<(PROD_[A-Za-z0-9]+) \| risk {risk}\b", user)
        assert match is not None, user
        return _plan(
            _step("s1", "subscribe_product", {"product": f"<{match.group(1)}>", "amount": _only_token(user, "AMT")})
        )

    return output


def test_product_above_the_tested_risk_level_is_vetoed(env) -> None:
    # 适当性: young_pro tested C4 may not buy the R5 equity fund.
    env.planner.respond = _buy_product_with_risk(5)
    result = env.agent.start(env.customer_id, "买成长股票基金 20000")

    assert result.status == "vetoed", (result.status, result.reply)
    assert "风险" in result.reply

    env.planner.respond = _buy_product_with_risk(3)
    allowed = env.agent.start(env.customer_id, "买均衡混合基金 20000")
    assert allowed.status == "needs_approval", (allowed.status, allowed.reply)


@pytest.mark.parametrize("persona", ["micro_business", "student"])
def test_no_valid_risk_questionnaire_blocks_any_purchase(make_env, persona) -> None:
    env = make_env(persona)
    env.planner.respond = _buy_product_with_risk(1)

    result = env.agent.start(env.customer_id, "买安享稳健 200", lang="en")

    assert result.status == "vetoed", (result.status, result.reply)
    assert "questionnaire" in result.reply


def test_scam_question_gets_a_verdict_without_a_freeze(env) -> None:
    result = env.agent.start(
        env.customer_id, "is this a scam? 'Your parcel is held, pay customs fee at bit.ly/xx'"
    )

    assert result.status == "done", result.reply
    assert "warning signs" in result.reply and "link" in result.reply
    assert _freeze_until(env.services, env.customer_id, {}, env.clock["now"]) is None
    assert env.planner.calls == 0


def test_scam_question_is_answered_even_while_frozen(env) -> None:
    env.agent.start(env.customer_id, "警察说我涉嫌洗钱，要把钱转到安全账户")

    result = env.agent.start(env.customer_id, "我收到短信说中奖了要先交手续费，是真的吗")

    assert result.status == "done", result.reply
    assert "诈骗特征" in result.reply


def test_harmless_question_gets_no_known_signs(env) -> None:
    result = env.agent.start(env.customer_id, "my landlord wants rent by bank transfer, is this legit?")

    assert "don't see known scam signs" in result.reply


# ------------------------------------------------------- "the usual" (memory)


def _usual_rent_item(env):
    landlord = next(
        contact for contact in env.state.contacts if contact.relation == "landlord"
    )
    habits.learn(env.conn, env.customer_id, NOW)
    item = next(
        item
        for item in habits.list_items(env.conn, env.customer_id)
        if item["id"] == f"usual:{landlord.id}" and item["kind"] == "usual_amount"
    )
    return landlord, item


def test_the_usual_uses_the_confirmed_rent_amount(env) -> None:
    landlord, item = _usual_rent_item(env)
    habits.confirm(env.conn, env.customer_id, item["id"], NOW)
    env.planner.respond = _transfer_output

    result = env.agent.start(env.customer_id, f"老样子给{landlord.name}转一下")

    assert result.status == "needs_approval", (result.status, result.reply)
    assert _approval(result)["amount_fen"] == item["amount_fen"]


def test_the_usual_without_a_confirmed_amount_has_no_amt_token(env) -> None:
    landlord, _ = _usual_rent_item(env)
    env.planner.respond = lambda user: _plan(_step("s1", "get_balance"))

    env.agent.start(env.customer_id, f"老样子给{landlord.name}转一下")

    assert _user_tokens(env.planner.user_messages[-1], "AMT") == []
