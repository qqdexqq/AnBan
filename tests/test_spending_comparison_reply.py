"""Both comparison totals remain explicit, correctly signed and grounded."""
from datetime import date

import pytest

from agent.responder import _render_spending
from bank.money import fmt_fen
from security.ground import collect_facts, check
from tests.test_intents_bills import billflow, _billturn, _manual_spend


def report(current=2_138_891, previous=2_014_024, *, mode="summary", available=True):
    return {"total": current, "available": True, "mode": mode,
            "by_category": {"shopping": current},
            "period_start": date(2026, 10, 1), "period_end": date(2026, 10, 31),
            "period_label": "2026-10", "income": 2_189_000,
            "net": 2_189_000 - current,
            "comparison": {"period_start": date(2026, 9, 1),
                           "period_end": date(2026, 9, 30), "available": available,
                           "total": previous, "delta": current - previous,
                           "by_category": {"shopping": current - previous}}}


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("current,previous", [(2_138_891, 2_014_024), (10000, 20000), (10000, 10000), (10000, 0)])
def test_both_period_totals_and_change_are_grounded(lang, current, previous):
    result = report(current, previous)
    facts = collect_facts([result])
    reply = "\n".join(_render_spending(result, lang, facts))
    assert f"- 2026-10-01 to 2026-10-31: {fmt_fen(current)}" in reply
    assert f"- 2026-09-01 to 2026-09-30: {fmt_fen(previous)}" in reply
    assert "Top categories:" not in reply
    if current != previous:
        assert fmt_fen(abs(current - previous)) in reply
        assert ("more" if current > previous else "less") in reply if lang == "en" else True
    else:
        assert "unchanged" in reply if lang == "en" else "持平" in reply
    assert check(reply, facts).ok, reply


def test_regular_report_keeps_income_categories_and_adds_baseline():
    result = report(mode="report")
    reply = "\n".join(_render_spending(result, "en", collect_facts([result])))
    assert "Income:" in reply and "Top categories:" in reply
    assert "Comparison period (2026-09-01 to 2026-09-30) spending: ¥20,140.24." in reply


def test_missing_baseline_data_is_not_presented_as_zero_spending():
    result = report(previous=0, available=False)
    reply = "\n".join(_render_spending(result, "en", collect_facts([result])))
    assert "comparison unavailable" in reply
    assert "Spending comparison:" not in reply
    assert "¥0.00" not in reply


def test_real_chat_followup_displays_both_ledger_totals(billflow):
    ordinary = _billturn(billflow, "Show my spending report for October 2026.")
    assert "2026-09" not in ordinary.reply
    assert "Comparison period" not in ordinary.reply
    assert "Income:" in ordinary.reply and "Top categories:" in ordinary.reply
    result = _billturn(billflow, "Can you compare it to September 2026?")
    assert "Spending comparison:" in result.reply
    assert _manual_spend(billflow, "month:2026-10") != _manual_spend(billflow, "month:2026-09")
    for period in ("month:2026-10", "month:2026-09"):
        assert fmt_fen(_manual_spend(billflow, period)) in result.reply


def test_annual_report_keeps_monthly_breakdown_without_unrequested_comparison(billflow):
    result = _billturn(billflow, "Show my annual spending report for 2026.")
    assert "Monthly spending:" in result.reply
    assert "Highest spending month:" in result.reply
    assert "Comparison period" not in result.reply
