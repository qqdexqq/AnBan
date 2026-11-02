"""Adversarial recipient and action-ordering regressions for repeat requests."""
import pytest
import re

from agent import recent_actions
from tests.test_recent_actions import (
    make_env, env, _approved_transfer_and_undo, _approval, _plan, _step,
    _transfer_output, _user_line_plan, _only_token,
)


@pytest.mark.parametrize("text", [
    "send again to Alice", "pay Alice again", "send Alice again",
    "transfer Alice again", "send to Alice again rather than Lina",
    "重新转给Alice", "再给Alice转一次",
])
def test_unknown_recipient_never_inherits_prior_payee(env, text):
    history = _approved_transfer_and_undo(env)
    env.planner.respond = _user_line_plan
    result = env.agent.start(env.customer_id, text, session_id=history.started.session_id)
    assert result.status != "needs_approval", result.reply
    assert not result.approvals
    assert env.services.bank_for(env.customer_id).get_balance() == history.before


def _submit(env, text, session_id=None):
    env.planner.respond = _transfer_output
    started = env.agent.start(env.customer_id, text, session_id=session_id)
    assert started.status == "needs_approval", started.reply
    approval = _approval(started)
    done = env.agent.resume(started.session_id, env.customer_id,
                            {approval["step_id"]: {"confirm": True}})
    assert done.status == "done", done.reply
    return started, done, approval


def test_undo_of_older_receipt_becomes_the_current_reference(env):
    first, done, approval = _submit(env, "给李娜转550")
    _submit(env, "给妈妈转300", first.session_id)
    undone = env.agent.undo(first.session_id, env.customer_id, done.plan_id, approval["step_id"])
    assert undone.status == "done"
    before = env.services.bank_for(env.customer_id).get_balance()
    env.planner.respond = _user_line_plan
    result = env.agent.start(env.customer_id, "I accidentally clicked undone, please send it again.", session_id=first.session_id)
    assert result.status == "needs_approval", result.reply
    assert _approval(result)["payee_display"] == approval["payee_display"]
    assert _approval(result)["amount_fen"] == 55_000
    assert env.services.bank_for(env.customer_id).get_balance() == before


def test_multi_transfer_source_requires_clarification(env):
    def two(user):
        args = {"payee": _only_token(user, "PAYEE"), "amount": _only_token(user, "AMT")}
        return _plan(_step("s1", "transfer", args), _step("s2", "transfer", args))
    env.planner.respond = two
    started = env.agent.start(env.customer_id, "给李娜转两笔，每笔100")
    done = env.agent.resume(started.session_id, env.customer_id,
                           {a["step_id"]: {"confirm": True} for a in started.approvals})
    assert done.status == "done", done.reply
    assert recent_actions.read_recent_transfer(env.conn, env.services.secret_key, env.customer_id, started.session_id) is None
    env.planner.respond = _user_line_plan
    result = env.agent.start(env.customer_id, "do it again", session_id=started.session_id)
    assert result.status == "clarify" and not result.approvals


def test_recent_transfer_cannot_authorize_an_investment(env):
    history = _approved_transfer_and_undo(env)
    env.planner.respond = lambda user: _plan(_step("s9", "subscribe_product", {
        "product": f"<PROD_{re.search(r'<PROD_(\d+)\s*\|', user).group(1)}>", "amount": _only_token(user, "AMT"),
    }))
    result = env.agent.start(env.customer_id, "send it again", session_id=history.started.session_id)
    assert result.status == "clarify" and not result.approvals
    assert env.services.bank_for(env.customer_id).get_balance() == history.before


def test_new_unconfirmed_request_does_not_repeat_an_older_transfer(env):
    history = _approved_transfer_and_undo(env)
    env.planner.respond = _transfer_output
    newer = env.agent.start(env.customer_id, "给妈妈转300", session_id=history.started.session_id)
    assert newer.status == "needs_approval"
    env.planner.respond = _user_line_plan
    result = env.agent.start(env.customer_id, "try it again", session_id=history.started.session_id)
    assert result.status == "clarify" and not result.approvals
    assert env.services.bank_for(env.customer_id).get_balance() == history.before
