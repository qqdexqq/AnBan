"""Cross-layer regressions for actual bill debits and post-approval freezes."""

import json
from datetime import timedelta

import pytest

from bank import bills
from evals.runner import FixedPlanner, _answers, make_env
from security import compliance


def _plan(tool, args):
    return json.dumps({"kind": "plan", "steps": [
        {"id": "s1", "tool": tool, "args": args, "after": []}
    ]})


def test_separate_bills_crossing_daily_threshold_require_mfa():
    env = make_env("young_pro", None)
    try:
        bank = env.services.bank_for(env.customer_id)
        before = bank.get_balance()
        red_seen = False
        for category in bills.CATEGORIES:
            env.services.planner = FixedPlanner(_plan("pay_bill", {"category": category}))
            result = env.agent.start(env.customer_id, f"Please pay my {category} bill.")
            assert result.status == "needs_approval"
            red_seen |= any(a["tier"] == "red" and a["needs_mfa"] for a in result.approvals)
            result = env.agent.resume(result.session_id, env.customer_id, _answers(result.approvals, "all"))
            assert result.status == "done"
        assert before - bank.get_balance() > 100_000
        assert bank.daily_outgoing_total(env.clock["now"]) == before - bank.get_balance()
        assert red_seen
    finally:
        env.conn.close()


def test_first_scam_disclosure_holds_overdue_transfer_before_release():
    env = make_env("young_pro", FixedPlanner(_plan("transfer", {
        "payee": "<PAYEE_1>", "amount": "<AMT_1>"
    })))
    try:
        result = env.agent.start(env.customer_id, "Transfer 300 to Mom.")
        approved = env.agent.resume(result.session_id, env.customer_id, _answers(result.approvals, "all"))
        assert approved.status == "done"
        env.clock["now"] += timedelta(hours=3)
        scam = env.agent.start(env.customer_id, "The police told me to send money to a safe account.")
        assert scam.status == "scam"
        row = env.conn.execute("SELECT status FROM transfers WHERE customer_id = ?", (env.customer_id,)).fetchone()
        assert row["status"] == "pending"
        undone = env.agent.undo(result.session_id, env.customer_id, result.plan_id, "s1")
        assert undone.status == "done"
    finally:
        env.conn.close()


@pytest.mark.parametrize("payload", [
    {"accounts": {}}, {"accounts": [None]},
    {"accounts": [{"account_no": "", "reason": "bad"}]},
    {"accounts": [{"account_no": 123, "reason": "bad"}]},
    {"accounts": [{"account_no": "123456789012"}]},
])
def test_malformed_fraud_list_shape_holds_release(tmp_path, monkeypatch, payload):
    env = make_env("young_pro", None)
    try:
        bank = env.services.bank_for(env.customer_id)
        contact = bank.find_contact("")[0]
        bank.transfer(contact.id, 1000, "", "shape-test", env.clock["now"])
        path = tmp_path / "fraud.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(compliance, "FRAUD_LIST_PATH", path)
        assert bank.release_due(env.clock["now"] + timedelta(hours=3)) == []
    finally:
        env.conn.close()
