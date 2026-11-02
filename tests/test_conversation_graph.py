"""Integration tests for the optional conversation gateway and memory.

Every test drives the real graph, seeded bank, vault, mandates, policy and
audit log.  Only the planner is fake: it answers the conversation prompt with
small talk and the banking prompt with plans, so the assertions describe what
code does with a model proposal.
"""

from __future__ import annotations

import json
import random
import re
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from agent.conversation import CONVERSATION_SYSTEM_PROMPT, ConversationGateway
from agent.conversation_memory import BoundedConversationMemory
from agent.graph import AnBanAgent, Services
from agent.planner import DEFAULT_REFUSAL
from bank import db
from bank.money import fmt_fen
from bank.seed import generate
from security.audit import AuditLog, create_audit_schema
from security.mandate import MandateService, create_security_schema
from security.policy import PolicyConfig
from security.tokens import TokenVault

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42
SECRET = b"anban-conversation-test-secret"


# ----------------------------------------------------------------------- fakes


class DualPlanner:
    """Conversation answers for the gateway, plans for the banking graph."""

    def __init__(self) -> None:
        self.conversation_calls: list[str] = []
        self.bank_calls: list[str] = []
        self.conversation_reply = lambda user: '{"kind":"banking"}'
        self.bank_reply = lambda user: DEFAULT_REFUSAL

    def plan(self, system: str, user: str) -> str:
        if system == CONVERSATION_SYSTEM_PROMPT:
            self.conversation_calls.append(user)
            return self.conversation_reply(user)
        self.bank_calls.append(user)
        return self.bank_reply(user)


def conversation(category: str, reply: str) -> str:
    return json.dumps(
        {"kind": "conversation", "category": category, "reply": reply},
        ensure_ascii=False,
    )


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def make_env():
    """Build one seeded customer plus a fresh agent for a single test."""

    connections = []

    def _make(
        persona: str = "young_pro",
        *,
        context_tokens: int = 100_000,
        gateway: bool = True,
        memory: bool = True,
    ):
        state = generate(persona, SEED, TODAY)
        conn = db.connect(":memory:")
        connections.append(conn)
        db.create_schema(conn)
        db.load_state(conn, state)
        create_security_schema(conn)
        create_audit_schema(conn)

        clock = {"now": NOW}
        planner = DualPlanner()
        conversation_memory = (
            BoundedConversationMemory(
                len,
                context_tokens=context_tokens,
                history_tokens=100_000,
                output_reserve=512,
                max_turns=24,
            )
            if memory
            else None
        )
        services = Services(
            conn=conn,
            vault=TokenVault(SECRET),
            mandates=MandateService(conn, SECRET, otp_rng=random.Random(7)),
            audit=AuditLog(conn, SECRET),
            planner=planner,
            policy_config=PolicyConfig(),
            secret_key=SECRET,
            now=lambda: clock["now"],
            conversation_gateway=ConversationGateway(planner) if gateway else None,
            conversation_memory=conversation_memory,
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


def _step(step_id: str, tool: str, args: dict | None = None, after=None) -> dict:
    return {
        "id": step_id,
        "tool": tool,
        "args": args or {},
        "after": list(after or ()),
    }


def _plan(*steps: dict) -> str:
    return json.dumps({"kind": "plan", "steps": list(steps)}, ensure_ascii=False)


def _user_tokens(user: str, kind: str) -> list[str]:
    line = user.rsplit("User:", 1)[-1]
    return re.findall(rf"<{kind}_[A-Za-z0-9]+>", line)


def _transfer_output(user: str) -> str:
    payees = _user_tokens(user, "PAYEE")
    amounts = _user_tokens(user, "AMT")
    assert payees and amounts, user
    return _plan(_step("s1", "transfer", {"payee": payees[0], "amount": amounts[0]}))


def _transfer_rows(env) -> int:
    return int(env.conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0])


def _history_entries(message: str) -> list[dict]:
    entries = []
    for line in message.splitlines():
        if not line.startswith("- "):
            continue
        try:
            data = json.loads(line[2:])
        except ValueError:
            continue
        if isinstance(data, dict) and set(data) == {"user", "assistant"}:
            entries.append(data)
    return entries


# ------------------------------------------------------------------------ tests


def test_greeting_reply_moves_no_money(env) -> None:
    env.planner.conversation_reply = lambda user: conversation(
        "greeting", "Hello! How can I help today?"
    )

    result = env.agent.start(env.customer_id, "hello")

    assert result.status == "done"
    assert result.reply == "Hello! How can I help today?"
    assert not result.approvals
    assert env.planner.conversation_calls
    assert env.planner.bank_calls == []
    assert _transfer_rows(env) == 0
    mandates = env.conn.execute("SELECT COUNT(*) FROM security_mandates").fetchone()[0]
    assert int(mandates) == 0


def test_greeting_with_a_transfer_still_needs_a_fresh_approval(env) -> None:
    env.planner.conversation_reply = lambda user: conversation(
        "greeting", "Hello!"
    )
    env.planner.bank_reply = _transfer_output

    result = env.agent.start(env.customer_id, "hi 给妈妈转300")

    assert result.status == "needs_approval", (result.status, result.reply)
    approval = result.approvals[0]
    assert approval["step_id"] == "s1"
    assert approval["amount_fen"] == 30_000
    # The tokenized message never reached the conversation model.
    assert env.planner.conversation_calls == []
    assert env.planner.bank_calls
    # Nothing moved before the customer confirms.
    assert _transfer_rows(env) == 0


def test_next_turn_history_has_the_greeting_but_no_old_amounts(env) -> None:
    env.planner.conversation_reply = lambda user: conversation(
        "greeting", "Hello! Nice to meet you."
    )
    first = env.agent.start(env.customer_id, "hello")
    assert first.status == "done"

    env.planner.bank_reply = _transfer_output
    second = env.agent.start(
        env.customer_id, "给妈妈转300", session_id=first.session_id
    )
    assert second.status == "needs_approval", (second.status, second.reply)
    env.agent.resume(first.session_id, env.customer_id, {"s1": {"confirm": True}})
    assert _transfer_rows(env) == 1

    # The next turn must go back to the banking planner, not to the gateway.
    env.planner.conversation_reply = lambda user: '{"kind":"banking"}'
    env.planner.bank_reply = lambda user: _plan(_step("s1", "get_balance"))
    env.planner.bank_calls.clear()
    third = env.agent.start(
        env.customer_id, "我的余额还有多少？", session_id=first.session_id
    )
    assert third.status == "done"

    message = env.planner.bank_calls[-1]
    assert message.splitlines()[-1].startswith("User:")
    history = _history_entries(message)
    assert any(entry["user"] == "hello" for entry in history)
    assert any(entry["assistant"] == "Hello! Nice to meet you." for entry in history)

    balance = env.services.bank_for(env.customer_id).get_balance()
    stale = [
        "<PAYEE_",
        "<AMT_",
        "<ACCT_",
        "300",
        fmt_fen(30_000),
        fmt_fen(balance),
    ]
    for entry in history:
        blob = json.dumps(entry, ensure_ascii=False)
        assert not re.search(r"\d", blob), blob
        for piece in stale:
            assert piece not in blob, (piece, blob)


def test_history_is_isolated_between_sessions(env) -> None:
    env.planner.conversation_reply = lambda user: conversation(
        "greeting", "Hello!"
    )
    first = env.agent.start(env.customer_id, "hello", session_id="session-a")
    assert first.status == "done"

    env.planner.conversation_reply = lambda user: '{"kind":"banking"}'
    env.planner.bank_reply = lambda user: _plan(_step("s1", "get_balance"))
    env.planner.bank_calls.clear()
    other = env.agent.start(
        env.customer_id, "我的余额还有多少？", session_id="session-b"
    )
    assert other.status == "done"

    message = env.planner.bank_calls[-1]
    assert "hello" not in message
    assert "Historical conversation" not in message


def test_invalid_gateway_reply_falls_back_to_the_bank_path(env) -> None:
    env.planner.conversation_reply = lambda user: "not json"
    env.planner.bank_reply = lambda user: _plan(_step("s1", "get_balance"))

    result = env.agent.start(env.customer_id, "hello")

    assert result.status == "done"
    balance = env.services.bank_for(env.customer_id).get_balance()
    assert fmt_fen(balance) in result.reply
    assert env.planner.bank_calls


def test_context_overflow_asks_to_shorten_and_never_plans(make_env) -> None:
    env = make_env(context_tokens=10)
    env.planner.conversation_reply = lambda user: conversation(
        "greeting", "Hello!"
    )
    env.planner.bank_reply = lambda user: _plan(_step("s1", "get_balance"))

    result = env.agent.start(env.customer_id, "hello")

    assert result.status == "clarify"
    assert result.reply == "Please shorten this request or start a new chat."
    assert env.planner.conversation_calls == []
    assert env.planner.bank_calls == []
    assert _transfer_rows(env) == 0


def test_scam_intake_still_wins_over_the_conversation_gateway(env) -> None:
    env.planner.conversation_reply = lambda user: conversation(
        "greeting", "Hello!"
    )

    result = env.agent.start(
        env.customer_id, "警察说我涉嫌洗钱，马上把钱转到安全账户"
    )

    assert result.status == "scam"
    assert env.planner.conversation_calls == []
    assert env.planner.bank_calls == []
    assert _transfer_rows(env) == 0


def test_unknown_repeat_still_asks_instead_of_answering_small_talk(env) -> None:
    env.planner.conversation_reply = lambda user: conversation(
        "support", "Take your time and review everything."
    )
    env.planner.bank_reply = _transfer_output

    result = env.agent.start(env.customer_id, "please transfer again")

    assert result.status == "clarify"
    assert "full request" in result.reply
    assert env.planner.conversation_calls == []
    assert env.planner.bank_calls == []
    assert _transfer_rows(env) == 0


@pytest.mark.parametrize("budget", [500, 567, 1000])
def test_whole_basket_is_quoted_before_any_approval_or_write(env, budget):
    def basket(user):
        amount = _user_tokens(user, "AMT")[0]
        merchants = list(dict.fromkeys(_user_tokens(user, "MERCHANT")))
        deliver = _user_tokens(user, "DATE")[0]
        assert len(merchants) == 2
        return _plan(
            _step("s1", "move_to_pot", {"pot_name": "birthday", "amount": amount}),
            _step("s2", "merchant_order", {
                "merchant": merchants[0], "item": "bouquet_m", "deliver": deliver,
                "funding_pot": "$s1.pot",
            }, ["s1"]),
            _step("s3", "merchant_order", {
                "merchant": merchants[1], "item": "cake_8in", "deliver": deliver,
                "funding_pot": "$s1.pot",
            }, ["s1"]),
        )
    env.planner.bank_reply = basket
    before = env.services.bank_for(env.customer_id).get_balance()
    pots_before = env.conn.execute("SELECT COUNT(*) FROM pots").fetchone()[0]
    result = env.agent.start(
        env.customer_id, f"预留{budget}元生日预算，明天从预算里在花店订花和蛋糕店订蛋糕",
        lang="en",
    )
    assert env.services.bank_for(env.customer_id).get_balance() == before
    assert env.conn.execute("SELECT COUNT(*) FROM pots").fetchone()[0] == pots_before
    assert env.conn.execute("SELECT COUNT(*) FROM merchant_orders").fetchone()[0] == 0
    if budget < 567:
        assert result.status == "clarify", result
        assert "567.00" in result.reply
        assert "67.00" in result.reply
        assert not result.approvals
    else:
        assert result.status == "needs_approval", result


def test_unexpected_failure_closes_tokens_before_the_next_turn(env, monkeypatch):
    import agent.graph as graph_module
    original = graph_module.evaluate
    calls = 0
    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated policy resource failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(graph_module, "evaluate", fail_once)
    env.planner.bank_reply = _transfer_output
    first = env.agent.start(env.customer_id, "给妈妈转300", lang="en")
    assert first.status == "error"
    assert "check transaction history" in first.reply
    again = env.agent.start(env.customer_id, "给妈妈转300", session_id=first.session_id, lang="en")
    assert again.status == "needs_approval", again
    assert again.plan_id != first.plan_id
    assert _transfer_rows(env) == 0
