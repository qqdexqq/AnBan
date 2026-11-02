"""Recording requests use actual report context and receipt-backed controls."""
import pytest
import json
import re

from agent.send_now_intent import send_now_request
from tests.test_intents_bills import billflow, _billturn
from tests.test_recent_actions import env as recent_env
from tests.test_intents_subscriptions import env as subscriptions_env, _turn
from tests.test_graph import make_env


def test_dated_anomaly_request_with_explanation_needs_no_previous_report(billflow):
    reply = _billturn(billflow, "Check for unusual transactions last month and explain what looks unusual.")
    assert "2026-10" in reply.reply
    assert "Which period" not in reply.reply
    next_reply = _billturn(billflow, "September 2026")
    assert "2026-09" in next_reply.reply


@pytest.mark.parametrize("text", [
    "Please send that transfer now instead of waiting.",
    "Send it now.", "Please send the transfer now instead of waiting two hours.",
])
def test_natural_send_now_wording_selects_verification_flow(text):
    assert send_now_request(text)


@pytest.mark.parametrize("text", ["Don't send that transfer now.", "Send 100 yuan to Lina now."])
def test_send_now_shortcut_does_not_take_new_or_negated_transfer(text):
    assert not send_now_request(text)


def test_other_recurring_lookup_has_additional_scope(subscriptions_env):
    from agent.intents.subscriptions import resolve
    ctx, _ = _turn(subscriptions_env, "Are there any other recurring charges not in my subscription list?")
    plan = resolve(ctx)
    assert plan["steps"][0]["args"] == {"scope": "additional"}


def test_exact_send_now_request_uses_only_own_pending_receipt(recent_env):
    from tests.test_recent_actions import _transfer_output
    from evals.runner import _answers
    recent_env.planner.respond = _transfer_output
    pending = recent_env.agent.start(recent_env.customer_id, "Send 300 yuan to Mom.")
    recent_env.agent.resume(pending.session_id, recent_env.customer_id, _answers(pending.approvals, "all"))
    before = recent_env.services.bank_for(recent_env.customer_id).get_balance()
    request = "Please send that transfer now instead of waiting."
    selected = recent_env.agent.start(recent_env.customer_id, request, pending.session_id, lang="en")
    assert selected.actions == ["send_now"]
    assert recent_env.services.bank_for(recent_env.customer_id).get_balance() == before
    other = recent_env.agent.start(recent_env.customer_id, request, lang="en")
    assert other.status == "clarify" and not other.actions


@pytest.mark.parametrize("wrong_tool", ["merchant_order", "redeem_product"])
def test_wrong_merchant_domain_gets_model_retry_and_still_requires_red_approval(make_env, wrong_tool):
    env = make_env()
    def propose(user):
        if "previous output rejected" in user:
            assert "tool_domain" in user
            current = user.split("User:", 1)[-1]
            product = re.search(r"<PROD_[0-9]+>", current).group()
            amount = re.search(r"<AMT_[0-9]+>", current).group()
            step = {"id": "s1", "tool": "subscribe_product", "args": {"product": product, "amount": amount}, "after": []}
        else:
            merchant = re.search(r"<MERCHANT_[0-9]+", user).group() + ">"
            step = {"id": "s1", "tool": "merchant_order", "args": {"merchant": merchant, "item": "roses_11", "deliver": "<DATE_1>"}, "after": []}
            if wrong_tool == "redeem_product":
                product = re.search(r"<PROD_[0-9]+>", user.split("User:", 1)[-1]).group()
                step = {"id": "s1", "tool": "redeem_product", "args": {"product": product}, "after": []}
        return json.dumps({"kind": "plan", "steps": [step]})
    env.planner.respond = propose
    result = env.agent.start(env.customer_id, "帮我申购安心债券基金，金额200元。")
    assert result.status == "needs_approval", result.reply
    assert result.approvals[0]["tier"] == "red"
    assert result.approvals[0]["tool"] == "subscribe_product"
    assert env.planner.calls == 2
    assert env.conn.execute("SELECT COUNT(*) FROM merchant_orders").fetchone()[0] == 0
