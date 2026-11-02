"""Chinese-regulation rules: fraud list, delayed arrival, regulator reports,
elder mode, reply language, AI-content label."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from evals.runner import FixedPlanner, make_env
from security import compliance

_TRANSFER = json.dumps(
    {
        "kind": "plan",
        "steps": [
            {
                "id": "s1",
                "tool": "transfer",
                "args": {"payee": "<PAYEE_1>", "amount": "<AMT_1>"},
                "after": [],
            }
        ],
    }
)


@pytest.fixture
def env_for():
    envs = []

    def _make(persona: str = "young_pro"):
        env = make_env(persona, FixedPlanner(_TRANSFER))
        envs.append(env)
        return env

    yield _make
    for env in envs:
        env.conn.close()


def _approve_all(env, result):
    answers = {}
    for approval in result.approvals:
        answers[approval["step_id"]] = {
            key: value
            for key, value in {
                "confirm": True,
                "otp": approval.get("demo_otp"),
                "retyped_amount_fen": approval.get("amount_fen"),
                "face_ok": True,
                "co_otp": approval.get("demo_co_otp"),
            }.items()
            if value is not None
        }
    return env.agent.resume(result.session_id, env.customer_id, answers)


def _transfers(env):
    return env.conn.execute(
        "SELECT amount_fen, release_at, status FROM transfers WHERE customer_id = ?",
        (env.customer_id,),
    ).fetchall()


def test_payment_to_a_fraud_listed_account_is_blocked_and_reported(env_for) -> None:
    env = env_for("retiree")

    result = env.agent.start(env.customer_id, "给社区王阿姨转300")

    assert result.status == "vetoed", (result.status, result.reply)
    assert "反诈" in result.reply
    assert _transfers(env) == []
    kinds = [row["kind"] for row in compliance.reports(env.conn, env.customer_id)]
    assert kinds == ["fraud_list_hit"]


def test_chosen_24h_arrival_delays_the_transfer_and_it_can_be_undone(env_for) -> None:
    env = env_for()
    compliance.update_settings(env.conn, env.customer_id, {"arrival_delay": "24h"})

    done = _approve_all(env, env.agent.start(env.customer_id, "给妈妈转300"))
    assert done.status == "done", done.reply

    (row,) = _transfers(env)
    created = env.clock["now"]
    assert row["release_at"].startswith((created + timedelta(hours=24)).date().isoformat())

    env.clock["now"] = created + timedelta(hours=23)
    step = next(item for item in done.plan_view if item["tool"] == "transfer")
    undone = env.agent.undo(done.session_id, env.customer_id, done.plan_id, step["step_id"])
    assert "撤销" in undone.reply or "Undone" in undone.reply
    assert _transfers(env)[0]["status"] == "cancelled"


def test_arrival_setting_never_shortens_the_default_window() -> None:
    from security.cooling import cooling_for

    assert cooling_for("yellow", arrival_delay=timedelta(0)) == timedelta(hours=2)
    assert cooling_for("yellow", arrival_delay=timedelta(hours=24)) == timedelta(hours=24)


def test_invalid_settings_are_rejected(env_for) -> None:
    env = env_for()
    for bad in ({"arrival_delay": "instant"}, {"lang": "fr"}, {"elder_mode": "yes"}, {"pin": "1"}):
        with pytest.raises(ValueError):
            compliance.update_settings(env.conn, env.customer_id, bad)


def test_large_transaction_report_is_filed_but_never_shown(env_for, monkeypatch) -> None:
    monkeypatch.setattr(compliance, "LARGE_TRANSFER_FEN", 20_000)  # ¥200 for the test
    env = env_for()

    done = _approve_all(env, env.agent.start(env.customer_id, "给妈妈转300"))

    reports = compliance.reports(env.conn, env.customer_id)
    assert [row["kind"] for row in reports] == ["large_transaction_report"]
    assert reports[0]["simulated"] is True
    for word in ("大额", "report", "报告", "regulator"):
        assert word not in done.reply

    # One report per day, not one per transfer.
    _approve_all(env, env.agent.start(env.customer_id, "再给妈妈转300"))
    assert len(compliance.reports(env.conn, env.customer_id)) == 1


def test_large_transaction_threshold_is_the_regulation_value() -> None:
    # 《金融机构大额交易和可疑交易报告管理办法》: natural person, domestic, ¥500,000.
    assert compliance.LARGE_TRANSFER_FEN == 50_000_000


def test_scam_freeze_is_reported_to_the_simulated_anti_fraud_centre(env_for) -> None:
    env = env_for("retiree")

    result = env.agent.start(env.customer_id, "警察说我涉嫌洗钱，要把钱转到安全账户")

    assert result.status == "scam"
    kinds = sorted(row["kind"] for row in compliance.reports(env.conn, env.customer_id))
    assert kinds == ["anti_fraud_center_report", "suspicious_transaction_report"]


def test_elder_mode_turns_on_elder_protections(env_for) -> None:
    env = env_for()
    compliance.update_settings(env.conn, env.customer_id, {"elder_mode": True})

    result = env.agent.start(env.customer_id, "给妈妈转800")

    # Elder daily limit is ¥500, so ¥800 needs the red checks and family co-approval.
    assert result.status == "needs_approval"
    assert result.approvals[0]["tier"] == "red"
    assert result.approvals[0]["needs_co_approval"] is True


def test_reply_language_follows_settings(env_for) -> None:
    env = env_for()

    english = env.agent.start(env.customer_id, "给妈妈转300", lang="en")
    chinese = env.agent.start(env.customer_id, "给妈妈转300", lang="zh")

    assert "Mom" in english.reply or "confirm" in english.reply.lower()
    assert "确认" in chinese.reply


def test_fraud_list_matches_spaced_dashed_and_full_width_numbers() -> None:
    # Regression.
    for variant in ("2659 6849 7193 1393", "2659-6849-7193-1393", "２６５９６８４９７１９３１３９３"):
        assert compliance.normalize_account(variant) in compliance.fraud_accounts()


def test_self_enabled_elder_mode_gets_24h_red_cooling(env_for) -> None:
    # Regression: the executor used its own persona-only elder check.
    from security.policy import PolicyConfig
    from agent.executor import _is_elder

    env = env_for()
    bank = env.services.bank_for(env.customer_id)
    assert _is_elder(bank, PolicyConfig()) is False
    compliance.update_settings(env.conn, env.customer_id, {"elder_mode": True})
    assert _is_elder(bank, PolicyConfig()) is True


def test_lowering_a_protection_waits_24h_and_raising_is_instant(env_for) -> None:
    env = env_for("young_pro")
    now = datetime(2026, 10, 1, 12, 0)
    compliance.update_settings(env.conn, env.customer_id, {"arrival_delay": "24h", "elder_mode": True}, now=now)
    assert compliance.get_settings(env.conn, env.customer_id).arrival_delay == "24h"

    lowered = compliance.update_settings(
        env.conn, env.customer_id, {"arrival_delay": "2h", "elder_mode": False, "lang": "zh"}, now=now
    )
    assert lowered.arrival_delay == "24h" and lowered.elder_mode is True and lowered.lang == "zh"
    assert {p["key"] for p in compliance.pending_changes(env.conn, env.customer_id)} == {
        "arrival_delay",
        "elder_mode",
    }
    # Asking again later does not restart the clock.
    compliance.update_settings(env.conn, env.customer_id, {"arrival_delay": "2h"}, now=now + timedelta(hours=20))
    compliance.apply_due(env.conn, env.customer_id, now + timedelta(hours=23))
    assert compliance.get_settings(env.conn, env.customer_id).arrival_delay == "24h"

    compliance.apply_due(env.conn, env.customer_id, now + timedelta(hours=24))
    settled = compliance.get_settings(env.conn, env.customer_id)
    assert settled.arrival_delay == "2h" and settled.elder_mode is False
    assert compliance.pending_changes(env.conn, env.customer_id) == []


def test_raising_again_cancels_a_queued_lowering(env_for) -> None:
    env = env_for("young_pro")
    now = datetime(2026, 10, 1, 12, 0)
    compliance.update_settings(env.conn, env.customer_id, {"elder_mode": True}, now=now)
    compliance.update_settings(env.conn, env.customer_id, {"elder_mode": False}, now=now)
    compliance.update_settings(env.conn, env.customer_id, {"elder_mode": True}, now=now)

    compliance.apply_due(env.conn, env.customer_id, now + timedelta(days=2))

    assert compliance.get_settings(env.conn, env.customer_id).elder_mode is True


def test_lowering_without_a_clock_is_refused(env_for) -> None:
    env = env_for("young_pro")
    compliance.update_settings(env.conn, env.customer_id, {"elder_mode": True})
    with pytest.raises(ValueError):
        compliance.update_settings(env.conn, env.customer_id, {"elder_mode": False})
