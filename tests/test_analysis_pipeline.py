"""S6: policy, privacy, grounding and failure paths through the real graph."""
from __future__ import annotations

import json
import re

import pytest

from agent import analysis_runner
from agent.analysis_data import analysis_rows
from agent.codegen import TEMPLATES
from bank.api import Bank
from bank.money import fmt_fen
from security.sandbox import SandboxResult
from tests.test_graph import make_env  # noqa: F401 - shared seeded graph fixture


@pytest.fixture(autouse=True)
def offline_codegen(monkeypatch):
    monkeypatch.setenv("ANBAN_SANDBOX_DEV", "1")
    monkeypatch.delenv("ANBAN_CODEGEN_URL", raising=False)
    monkeypatch.delenv("ANBAN_LOCAL_URL", raising=False)


def _events(env):
    return [json.loads(row[0]) for row in env.conn.execute(
        "SELECT payload_json FROM security_audit WHERE kind='SANDBOX' ORDER BY seq"
    )]


def test_weekend_graph_reply_is_grounded_audited_and_read_only(make_env):
    from datetime import timedelta
    env = make_env()
    bank = Bank(env.conn, env.customer_id)
    balance = bank.get_balance()
    transactions = bank.list_transactions(limit=None)
    reply = env.agent.start(env.customer_id,
        "Compare my weekend spending with weekdays and explain the difference")
    assert reply.status == "done", reply.reply
    assert env.planner.calls == 0
    assert "weekend" in reply.reply.lower(), reply.reply
    assert "development" in reply.reply.lower(), reply.reply
    assert "isolated sandbox" not in reply.reply.lower()
    rows = analysis_rows(bank, env.clock["now"].date() - timedelta(days=29), env.clock["now"].date())
    for subset in ([r for r in rows if r["weekday"] >= 5], [r for r in rows if r["weekday"] < 5]):
        total = sum(r["amount_fen"] for r in subset)
        assert fmt_fen(total) in reply.reply
    assert bank.get_balance() == balance
    assert bank.list_transactions(limit=None) == transactions
    events = _events(env)
    assert len(events) == 1
    assert events[0]["source"] == "template"
    assert events[0]["backend"] == "subprocess-dev"
    assert events[0]["row_count"] == len(rows)
    assert re.fullmatch(r"[a-f0-9]{64}", events[0]["code_sha256"])
    assert "def analyze(rows):" in events[0]["code"]
    assert "rows" not in events[0] and "result" not in events[0]
    assert env.services.audit.verify().ok


@pytest.mark.parametrize("question", [
    "What time of day do I spend the most?",
    "Which weekday do I spend the most?",
    "周末和工作日花钱有什么不同？",
])
def test_all_supported_analyses_reach_grounded_reply(make_env, question):
    env = make_env()
    result = env.agent.start(env.customer_id, question)
    assert result.status == "done", result.reply
    assert _events(env), result.reply
    assert "No action was taken" not in result.reply
    assert "未执行" not in result.reply


def test_unavailable_is_honest_and_has_no_numeric_claims(make_env, monkeypatch):
    monkeypatch.setattr(analysis_runner, "run_analysis_code", lambda *args, **kwargs:
        SandboxResult(False, None, "unavailable", 0, "a" * 64, (), "unavailable"))
    env = make_env()
    result = env.agent.start(env.customer_id, "Compare weekend spending with weekdays")
    assert not re.search(r"\d", result.reply), result.reply
    assert "unavailable" in result.reply.lower() or "could not" in result.reply.lower(), result.reply
    assert "Calculated in an isolated sandbox" not in result.reply
    assert _events(env)[0]["error"] == "unavailable"


def test_codegen_exception_runs_audited_template(make_env, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("do not echo this")
    monkeypatch.setattr(analysis_runner, "generate_analysis_code", fail)
    env = make_env()
    result = env.agent.start(env.customer_id, "Compare weekend spending with weekdays")
    assert "do not echo" not in result.reply
    assert _events(env)[0]["source"] == "template"
    assert _events(env)[0]["ok"]


def test_model_invented_totals_never_reach_reply(make_env, monkeypatch):
    monkeypatch.setattr(analysis_runner, "generate_analysis_code", lambda *a, **kw:
        ('def analyze(rows):\n    return {"weekend_total_fen": 987654321}', "model"))
    env = make_env()
    result = env.agent.start(env.customer_id, "Compare weekend spending with weekdays")
    assert "9,876,543.21" not in result.reply
    assert [e["source"] for e in _events(env)] == ["model", "template"]
    assert all(e["ok"] for e in _events(env))


def test_verified_model_proposal_runs_in_sandbox(make_env, monkeypatch):
    monkeypatch.setattr(analysis_runner, "generate_analysis_code", lambda *a, **kw:
        (TEMPLATES["weekend_vs_weekday"], "model"))
    env = make_env()
    result = env.agent.start(env.customer_id, "Compare weekend spending with weekdays")
    assert result.status == "done"
    assert [e["source"] for e in _events(env)] == ["model", "template"]
    assert all(e["ok"] for e in _events(env))


def test_model_code_with_pii_literal_is_not_executed_or_logged(make_env, monkeypatch):
    monkeypatch.setattr(analysis_runner, "generate_analysis_code", lambda *a, **kw:
        ('def analyze(rows):\n    return {"number": 13800138000}', "model"))
    env = make_env()
    env.agent.start(env.customer_id, "Compare weekend spending with weekdays")
    events = _events(env)
    assert events[0]["error"] == "violation:pii_code"
    assert events[0]["code_redacted"] and events[0]["code"] == ""
    assert "13800138000" not in json.dumps(events)
    assert events[-1]["source"] == "template"


def test_codegen_sees_only_canonical_question(make_env, monkeypatch):
    seen = []
    def propose(question, **kwargs):
        seen.append(question)
        return TEMPLATES["weekend_vs_weekday"], "template"
    monkeypatch.setattr(analysis_runner, "generate_analysis_code", propose)
    env = make_env()
    env.agent.start(env.customer_id, "Compare weekend spending with weekdays")
    assert seen == ["Compare weekend spending with weekday spending"]


def test_audit_digest_digits_are_not_mistaken_for_phone_numbers():
    from security.audit import AuditError, _CodeFingerprint, _check_pii
    _check_pii({"code_sha256": _CodeFingerprint(TEMPLATES["by_hour"])})
    with pytest.raises(AuditError):
        _check_pii({"code_sha256": "13800138000" + "a" * 53})
    with pytest.raises(AuditError):
        _check_pii({"code_sha256": "13800138000"})
    with pytest.raises(AuditError):
        _check_pii({"note": "13800138000"})
