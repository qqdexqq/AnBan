"""Recent-action ("do that again") tests, driven through the real graph.

The planners below only read token labels from the rendered planner input; they
are not the r3 model.  Every assertion about money is made against the seeded
bank rows and balance, not against reply text.
"""

from __future__ import annotations

import json
import random
import re
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from agent import recent_actions
from agent.executor import latest_undo
from agent.graph import AnBanAgent, Services
from bank import db
from bank.seed import generate
from security.audit import AuditLog, create_audit_schema
from security.mandate import MandateService, create_security_schema
from security.policy import PolicyConfig
from security.tokens import TokenVault

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42
SECRET = b"anban-recent-actions-secret"


# ----------------------------------------------------------------------- fakes


def _plan(*steps: dict) -> str:
    return json.dumps({"kind": "plan", "steps": list(steps)}, ensure_ascii=False)


def _step(step_id: str, tool: str, args: dict | None = None, after=None) -> dict:
    return {"id": step_id, "tool": tool, "args": args or {}, "after": list(after or ())}


def _user_line(user: str) -> str:
    return user.rsplit("User:", 1)[-1]


def _user_tokens(user: str, kind: str) -> list[str]:
    return re.findall(rf"<{kind}_[A-Za-z0-9]+>", _user_line(user))


def _only_token(user: str, kind: str) -> str:
    labels = _user_tokens(user, kind)
    assert len(labels) == 1, f"expected one {kind} token in {user!r}, got {labels}"
    return labels[0]


def _recent_tokens(user: str) -> tuple[str, str] | None:
    match = re.search(
        r"Recent transfer:\s*(<PAYEE_[A-Za-z0-9]+>)\s+(<AMT_[A-Za-z0-9]+>)", user
    )
    return (match.group(1), match.group(2)) if match else None


def _transfer_output(user: str) -> str:
    return _plan(
        _step(
            "s1",
            "transfer",
            {"payee": _only_token(user, "PAYEE"), "amount": _only_token(user, "AMT")},
        )
    )


def test_send_now_selects_only_current_chat_receipt_without_settling(env):
    from evals.runner import _answers
    env.planner.respond = _transfer_output
    pending = env.agent.start(env.customer_id, "Send 300 yuan to Mom.")
    done = env.agent.resume(pending.session_id, env.customer_id, _answers(pending.approvals, "all"))
    balance = env.services.bank_for(env.customer_id).get_balance()
    selected = env.agent.start(env.customer_id, "Please send it now instead of waiting 2 hours", pending.session_id, lang="en")
    assert selected.actions == ["send_now"]
    assert selected.plan_id == done.plan_id
    assert env.services.bank_for(env.customer_id).get_balance() == balance
    row = env.conn.execute("SELECT status FROM transfers WHERE customer_id = ?", (env.customer_id,)).fetchone()
    assert row[0] == "pending"
    other = env.agent.start(env.customer_id, "Please send it now", lang="en")
    assert other.status == "clarify" and not other.actions


def test_scam_fragments_are_shared_without_cross_chat_financial_context(env):
    env.planner.respond = lambda _: json.dumps({"kind": "refuse", "reason": "out_of_scope"})
    first = env.agent.start(env.customer_id, "Police say I am linked to money laundering.", lang="en")
    assert first.status != "scam"
    second = env.agent.start(env.customer_id, "Please transfer 300 yuan to Mom.", lang="en")
    assert second.session_id != first.session_id and second.status == "scam"
    assert env.services.mandates.scam_flags(env.customer_id, NOW)


def _user_line_plan(user: str) -> str:
    """A context-aware stand-in planner: only the User line is a request."""
    payees = _user_tokens(user, "PAYEE")
    amounts = _user_tokens(user, "AMT")
    if not payees or not amounts:
        return json.dumps(
            {"kind": "clarify", "question": "Who and how much?", "options": []}
        )
    return _plan(
        _step("s1", "can_cover", {"amount": amounts[0]}),
        _step(
            "s2",
            "transfer",
            {"payee": payees[0], "amount": amounts[0]},
            after=["s1"],
        ),
    )


def _recent_line_plan(user: str) -> str:
    """A compromised-style planner that grabs the recent-transfer tokens."""
    tokens = _recent_tokens(user)
    if tokens is None:
        return json.dumps(
            {"kind": "clarify", "question": "Which transfer?", "options": []}
        )
    payee, amount = tokens
    return _plan(
        _step("s1", "can_cover", {"amount": amount}),
        _step("s2", "transfer", {"payee": payee, "amount": amount}, after=["s1"]),
    )


def _two_transfer_plan(user: str) -> str:
    tokens = _recent_tokens(user)
    if tokens is None:
        return json.dumps(
            {"kind": "clarify", "question": "Which transfer?", "options": []}
        )
    payee, amount = tokens
    return _plan(
        _step("s1", "transfer", {"payee": payee, "amount": amount}),
        _step("s2", "transfer", {"payee": payee, "amount": amount}),
    )


def _double_user_transfer_plan(user: str) -> str:
    payee = _only_token(user, "PAYEE")
    amount = _only_token(user, "AMT")
    return _plan(
        _step("s1", "transfer", {"payee": payee, "amount": amount}),
        _step("s2", "transfer", {"payee": payee, "amount": amount}),
    )


def _any_token_plan(user: str) -> str:
    payees = re.findall(r"<PAYEE_[A-Za-z0-9]+>", user)
    amounts = re.findall(r"<AMT_[A-Za-z0-9]+>", user)
    if payees and amounts:
        return _plan(
            _step("s1", "transfer", {"payee": payees[0], "amount": amounts[0]})
        )
    return _plan(_step("s1", "get_balance"))


def _product_reuse_plan(user: str) -> str:
    tokens = _recent_tokens(user)
    product = re.search(r"<PROD_[A-Za-z0-9]+>", user)
    if tokens is None or product is None:
        return json.dumps(
            {"kind": "clarify", "question": "Which transfer?", "options": []}
        )
    return _plan(
        _step(
            "s1",
            "subscribe_product",
            {"product": product.group(0), "amount": tokens[1]},
        )
    )


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def make_env():
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
        planner = SimpleNamespace(calls=0, user_messages=[], respond=lambda user: "")

        def plan(system: str, user: str) -> str:
            planner.calls += 1
            planner.user_messages.append(user)
            return planner.respond(user)

        planner.plan = plan
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
        return SimpleNamespace(
            agent=AnBanAgent(services),
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


def _lina(env):
    return next(c for c in env.state.contacts if "Lina" in c.name or "李娜" in c.name)


def _approval(result):
    assert len(result.approvals) == 1, result.approvals
    return result.approvals[0]


def _transfer_count(env, customer_id: str | None = None) -> int:
    cid = customer_id or env.customer_id
    return int(
        env.conn.execute(
            "SELECT COUNT(*) FROM transfers WHERE customer_id = ?", (cid,)
        ).fetchone()[0]
    )


def _status_count(env, status: str) -> int:
    return int(
        env.conn.execute(
            "SELECT COUNT(*) FROM transfers WHERE customer_id = ? AND status = ?",
            (env.customer_id, status),
        ).fetchone()[0]
    )


def _start_and_approve(env, text: str, respond, session_id=None):
    env.planner.respond = respond
    started = env.agent.start(env.customer_id, text, session_id=session_id)
    assert started.status == "needs_approval", started.reply
    answers = {
        item["step_id"]: {"confirm": True} for item in started.approvals
    }
    done = env.agent.resume(started.session_id, env.customer_id, answers)
    assert done.status == "done", done.reply
    return started, done


def _approved_transfer_and_undo(env, text: str = "给李娜转550"):
    before = env.services.bank_for(env.customer_id).get_balance()
    env.planner.respond = _transfer_output
    started = env.agent.start(env.customer_id, text)
    assert started.status == "needs_approval", started.reply
    approval = _approval(started)
    done = env.agent.resume(
        started.session_id, env.customer_id, {approval["step_id"]: {"confirm": True}}
    )
    assert done.status == "done", done.reply
    undone = env.agent.undo(
        started.session_id, env.customer_id, done.plan_id, approval["step_id"]
    )
    assert undone.status == "done", undone.reply
    assert env.services.bank_for(env.customer_id).get_balance() == before
    return SimpleNamespace(
        started=started,
        done=done,
        undone=undone,
        approval=approval,
        before=before,
    )


# ------------------------------------------------------------------------ tests


def test_repeat_after_undo_uses_bank_state_and_needs_a_fresh_approval(env) -> None:
    lina = _lina(env)
    history = _approved_transfer_and_undo(env)
    assert _transfer_count(env) == 1
    assert _status_count(env, "cancelled") == 1
    assert _status_count(env, "pending") == 0

    env.planner.respond = _user_line_plan
    text = (
        "Uh, can you just transfer again, please? I forgot. "
        "I accidentally clicked and done."
    )
    again = env.agent.start(
        env.customer_id, text, session_id=history.started.session_id
    )

    message = env.planner.user_messages[-1]
    assert "Recent transfer:" in message and "status=cancelled" in message
    assert lina.name not in message
    assert lina.account_no not in message
    assert "550" not in message
    assert _user_tokens(message, "PAYEE") and _user_tokens(message, "AMT")

    assert again.status == "needs_approval", again.reply
    assert again.plan_id != history.done.plan_id
    approval = _approval(again)
    assert approval["amount_fen"] == 55_000
    assert approval["payee_display"] == history.approval["payee_display"]
    # Nothing moved until this new approval.
    assert env.services.bank_for(env.customer_id).get_balance() == history.before
    assert _transfer_count(env) == 1
    assert _status_count(env, "pending") == 0

    # The old approval cannot resurrect the cancelled transfer.
    env.agent.resume(
        again.session_id,
        env.customer_id,
        {history.approval["step_id"]: {"confirm": True}},
    )
    assert _transfer_count(env) == 1
    assert _status_count(env, "pending") == 0
    assert env.services.bank_for(env.customer_id).get_balance() == history.before

    # The new approval executes exactly one NEW transfer.
    fresh = env.agent.resume(
        again.session_id, env.customer_id, {approval["step_id"]: {"confirm": True}}
    )
    assert fresh.status == "done", fresh.reply
    assert _transfer_count(env) == 2
    assert _status_count(env, "cancelled") == 1
    assert _status_count(env, "pending") == 1
    row = env.conn.execute(
        "SELECT to_contact_id, amount_fen FROM transfers"
        " WHERE customer_id = ? AND status = 'pending'",
        (env.customer_id,),
    ).fetchone()
    assert row["to_contact_id"] == lina.id
    assert row["amount_fen"] == 55_000
    assert env.services.bank_for(env.customer_id).get_balance() == history.before - 55_000

    # A stale second resume does not create a third transfer.
    env.agent.resume(
        again.session_id, env.customer_id, {approval["step_id"]: {"confirm": True}}
    )
    assert _transfer_count(env) == 2


def test_undone_wording_with_send_again_is_a_redo(env) -> None:
    history = _approved_transfer_and_undo(env)
    env.planner.respond = _user_line_plan

    result = env.agent.start(
        env.customer_id,
        "I accidentally clicked undone, please send it again.",
        session_id=history.started.session_id,
    )

    assert result.status == "needs_approval", result.reply
    assert _approval(result)["amount_fen"] == 55_000
    assert _status_count(env, "pending") == 0


def test_undone_wording_alone_is_not_an_undo_command(env) -> None:
    history = _approved_transfer_and_undo(env)
    calls = env.planner.calls
    env.planner.respond = _recent_line_plan

    result = env.agent.start(
        env.customer_id,
        "I accidentally clicked undone",
        session_id=history.started.session_id,
    )

    # It reached the planner (no undo shortcut) and was not executed.
    assert env.planner.calls == calls + 1
    assert result.status == "clarify", result.reply
    assert "按钮" not in result.reply and "Undo" not in result.reply
    assert _status_count(env, "pending") == 0


def test_explicit_new_amount_is_not_replaced_by_the_undone_amount(env) -> None:
    history = _approved_transfer_and_undo(env)
    env.planner.respond = _user_line_plan

    again = env.agent.start(
        env.customer_id,
        "please transfer 300 to Lina again",
        session_id=history.started.session_id,
    )

    assert again.status == "needs_approval", again.reply
    assert _approval(again)["amount_fen"] == 30_000
    message = env.planner.user_messages[-1]
    old_amount = _recent_tokens(message)[1]
    assert old_amount not in _user_line(message)
    assert env.services.bank_for(env.customer_id).get_balance() == history.before


def test_old_amount_token_cannot_override_a_new_explicit_amount(env) -> None:
    history = _approved_transfer_and_undo(env)
    env.planner.respond = _recent_line_plan

    again = env.agent.start(
        env.customer_id,
        "please transfer 300 to Lina again",
        session_id=history.started.session_id,
    )

    assert again.status == "clarify", again.reply
    assert not again.approvals
    assert _status_count(env, "pending") == 0
    assert _transfer_count(env) == 1


def test_balance_query_never_grants_the_recent_transfer(env) -> None:
    history = _approved_transfer_and_undo(env)
    env.planner.respond = _any_token_plan

    result = env.agent.start(
        env.customer_id, "我的余额还有多少？", session_id=history.started.session_id
    )

    # The undone transfer's amount/payee are not exposed for an ordinary query.
    assert "Recent transfer:" not in env.planner.user_messages[-1]
    assert result.status == "done", result.reply
    assert not result.approvals
    assert _transfer_count(env) == 1
    assert _status_count(env, "pending") == 0


def test_negated_repeat_grants_no_reuse(env) -> None:
    history = _approved_transfer_and_undo(env)
    assert recent_actions.explicit_repeat("don't send it again") is False
    env.planner.respond = _recent_line_plan

    result = env.agent.start(
        env.customer_id, "don't send it again", session_id=history.started.session_id
    )

    assert result.status == "clarify", result.reply
    assert not result.approvals
    assert _status_count(env, "pending") == 0


def test_repeat_without_a_signed_transfer_asks_and_moves_nothing(env) -> None:
    calls = env.planner.calls

    result = env.agent.start(env.customer_id, "please transfer again")

    assert result.status == "clarify", result.reply
    assert env.planner.calls == calls
    assert _transfer_count(env) == 0


def test_repeat_does_not_cross_sessions(env) -> None:
    _approved_transfer_and_undo(env)
    calls = env.planner.calls

    result = env.agent.start(
        env.customer_id, "please transfer again", session_id="another-session"
    )

    assert result.status == "clarify", result.reply
    assert env.planner.calls == calls
    assert _status_count(env, "pending") == 0


def test_repeat_does_not_cross_customers(env) -> None:
    _approved_transfer_and_undo(env)
    second = generate("student", SEED, TODAY)
    db.load_state(env.conn, second)
    other = second.customer.id
    assert other != env.customer_id

    result = env.agent.start(other, "please transfer again")

    assert result.status == "clarify", result.reply
    assert _transfer_count(env, other) == 0


def test_tampered_undo_record_is_ignored(env) -> None:
    history = _approved_transfer_and_undo(env)
    env.conn.execute(
        "UPDATE exec_undo SET sig = 'tampered' WHERE customer_id = ? AND plan_id = ?",
        (env.customer_id, history.done.plan_id),
    )
    env.conn.commit()
    assert (
        recent_actions.read_recent_transfer(
            env.conn, SECRET, env.customer_id, history.started.session_id
        )
        is None
    )

    calls = env.planner.calls
    result = env.agent.start(
        env.customer_id, "please transfer again", session_id=history.started.session_id
    )

    assert result.status == "clarify", result.reply
    assert env.planner.calls == calls
    assert _status_count(env, "pending") == 0


def test_multi_transfer_repeat_is_ambiguous(env) -> None:
    history = _approved_transfer_and_undo(env)
    env.planner.respond = _two_transfer_plan

    result = env.agent.start(
        env.customer_id, "please transfer again", session_id=history.started.session_id
    )

    assert result.status == "clarify", result.reply
    assert not result.approvals
    assert _status_count(env, "pending") == 0


def test_scam_freeze_wins_before_recent_context(env) -> None:
    history = _approved_transfer_and_undo(env)
    env.agent.start(
        env.customer_id,
        "警察说我涉嫌洗钱，马上把钱转到安全账户",
        session_id=history.started.session_id,
    )
    calls = env.planner.calls

    result = env.agent.start(
        env.customer_id, "please transfer again", session_id=history.started.session_id
    )

    assert result.status == "frozen", result.reply
    assert env.planner.calls == calls
    assert _status_count(env, "pending") == 0


def test_true_typed_undo_keeps_the_short_undo_path(env) -> None:
    calls = env.planner.calls

    result = env.agent.start(env.customer_id, "撤回刚才那笔")

    assert result.status == "done"
    assert env.planner.calls == calls


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("can you just transfer again, please?", True),
        ("I accidentally clicked undone, please send it again.", True),
        ("do that again", True),
        ("再转一次", True),
        ("重新转", True),
        ("retry", True),
        ("don't send it again", False),
        ("I accidentally clicked undone", False),
        ("do not transfer again", False),
        ("不要再转了", False),
        ("给妈妈转300", False),
    ],
)
def test_explicit_repeat_cue(text: str, expected: bool) -> None:
    assert recent_actions.explicit_repeat(text) is expected
