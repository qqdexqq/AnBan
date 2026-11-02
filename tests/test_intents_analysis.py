"""Analysis routing and honest, grounded sandbox result rendering."""

from datetime import datetime
from types import SimpleNamespace

import pytest

from agent.intents import RULES, resolve as resolve_intent
from agent.intents.analysis import resolve
from agent.responder import _render_analysis
from security.ground import Facts, check


def context(text, lang="en"):
    return SimpleNamespace(raw_text=text, tokenized_text=text, now=datetime(2026, 11, 2, 12), lang=lang,
                           conn=None, secret_key=b"test", customer_id="customer", session_id="chat")


@pytest.mark.parametrize("text,kind,period", [
    ("Compare my weekend spending with weekdays and explain the difference", "weekend_vs_weekday", "last_30_days"),
    ("周末和工作日花钱有什么不同？", "weekend_vs_weekday", "last_30_days"),
    ("比较今年周末和平日的消费", "weekend_vs_weekday", "this_year"),
    ("What time of day do I spend most last month?", "by_hour", "last_month"),
    ("我几点花钱最多？", "by_hour", "last_30_days"),
    ("Which weekday do I spend the most?", "by_weekday", "last_30_days"),
    ("星期几花得最多？消费分析", "by_weekday", "last_30_days"),
    ("星期几花得最多？", "by_weekday", "last_30_days"),
    ("Analyse my spending in October 2026", "custom", "month:2026-10"),
    ("分析今年消费", "custom", "this_year"),
])
def test_analysis_kind_and_period(text, kind, period):
    plan = resolve(context(text))
    assert plan == {"kind": "plan", "steps": [{"id": "s1", "tool": "run_analysis", "args": {"period": period, "kind": kind}, "after": []}]}


@pytest.mark.parametrize("text", [
    "Compare weekends and send 100 to mum",
    "Compare my weekend spending with weekdays and transfer money to mum",
    "分析消费，帮我付款",
    "Compare my weekend spending with weekdays and cancel iCloud",
    "Don't analyse my spending",
    "不要分析我的消费",
    "Forecast spending by weekday",
    "What is my balance?",
    "Compare my spending this month with last month",
])
def test_other_requests_and_mixed_writes_not_consumed(text):
    assert resolve(context(text)) is None


def test_analysis_runs_after_bills_without_bill_rule_consuming_it():
    names = [rule.__module__ for rule in RULES]
    assert names.index("agent.intents.bills") < names.index("agent.intents.analysis")
    result = resolve_intent(context("Compare my weekend spending with weekdays and explain the difference"))
    assert result["steps"][0]["tool"] == "run_analysis"


def weekend_result(backend="unshare"):
    return {"raw": {
        "row_count": 5, "weekend_total_fen": 18000, "weekday_total_fen": 9000,
        "weekend_days": 2, "weekday_days": 3, "weekend_average_fen": 9000,
        "weekday_average_fen": 3000, "difference_fen": 6000,
        "categories": ["food"], "category_differences_fens": [6000], "largest_category": "food",
    }, "analysis": {"kind": "weekend_vs_weekday", "ok": True},
        "sandbox": {"ok": True, "backend": backend}}


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_weekend_response_only_uses_grounded_numbers(lang):
    facts = Facts()
    text = "\n".join(_render_analysis(weekend_result(), lang, facts))
    assert "180.00" in text and "90.00" in text and "30.00" in text and "60.00" in text
    assert check(text, facts).ok
    assert "隔离沙箱" in text if lang == "zh" else "isolated sandbox" in text
    assert "有消费记录" in text if lang == "zh" else "Days without spending are excluded" in text


def test_development_response_does_not_claim_isolation():
    text = "\n".join(_render_analysis(weekend_result("subprocess-dev"), "en", Facts()))
    assert "development test process" in text
    assert "isolation was not enabled" in text
    assert "Calculated in an isolated sandbox" not in text


@pytest.mark.parametrize("result", [
    {"raw": None, "sandbox": {"ok": False, "error": "unavailable"}},
    {"raw": {"total_fen": 50000}, "sandbox": {"ok": True, "backend": "unavailable"}},
    {"raw": {}, "analysis": {"kind": "weekend_vs_weekday"}, "sandbox": {"ok": True, "backend": "unshare"}},
])
def test_failure_reply_is_honest_without_numbers(result):
    text = "\n".join(_render_analysis(result, "en", Facts()))
    assert "could not complete" in text
    assert not any(character.isdigit() for character in text)
    assert "Calculated in an isolated sandbox" not in text


def test_missing_weekend_days_does_not_claim_meaningful_difference():
    result = weekend_result()
    result["raw"].update(weekend_days=0, weekend_total_fen=0, weekend_average_fen=0, difference_fen=-3000)
    text = "\n".join(_render_analysis(result, "en", Facts()))
    assert "cannot be compared" in text
    assert "less per spending day" not in text


def test_empty_data_is_not_reported_as_no_spending_in_the_real_world():
    result = weekend_result()
    result["raw"]["row_count"] = 0
    text = "\n".join(_render_analysis(result, "en", Facts()))
    assert "no spending transactions to analyse" in text
    assert "¥" not in text


def test_hour_and_weekday_responses_ground_arrays():
    hour = {"raw": {"row_count": 4, "bucket_totals_fens": [100, 200, 300, 400], "peak_bucket": 3},
            "analysis": {"kind": "by_hour"}, "sandbox": {"ok": True, "backend": "unshare"}}
    days = {"raw": {"row_count": 7, "weekday_totals_fens": [100] * 7, "weekday_averages_fens": [50] * 7,
                    "weekday_days": [2] * 7}, "analysis": {"kind": "by_weekday"},
            "sandbox": {"ok": True, "backend": "unshare"}}
    for result in (hour, days):
        facts = Facts()
        text = "\n".join(_render_analysis(result, "en", facts))
        assert check(text, facts).ok
    assert "Highest spending period: evening" in "\n".join(_render_analysis(hour, "en", Facts()))
