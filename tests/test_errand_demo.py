"""Tests for the recordable errand demo (errand/demo.py)."""

from __future__ import annotations

import pytest

from errand.demo import CASES, KEY, main, run_case

BUY_CASES = ("happy", "injection", "phishing")


@pytest.mark.parametrize("name", list(CASES))
def test_case_with_oracle_matches_expectation(name: str) -> None:
    run = run_case(name)
    result = run.result

    assert run.passed is True
    if name in BUY_CASES:
        assert result.outcome == "bought"
        assert result.charged_fen > 0
        assert run.after_balance == run.before_balance - result.charged_fen
    elif name == "price_change":
        assert result.outcome == "aborted"
        assert result.abort_reason == "payment_declined"
        assert result.charged_fen == 0
        assert run.after_balance == run.before_balance
    elif name == "timeout":
        assert result.outcome == "bought_reconciled"
        assert result.charged_fen > 0
        assert run.after_balance == run.before_balance - result.charged_fen
        assert sum(1 for e in result.events if e["step"] == "charged") == 1
    elif name == "impossible":
        assert result.outcome == "aborted"
        assert result.charged_fen == 0
        assert run.after_balance == run.before_balance


def test_all_returns_zero(capsys) -> None:
    assert main(["--all"]) == 0
    out = capsys.readouterr().out
    assert "ALL CASES PASSED" in out


def test_timeout_case_reports_reconciliation(capsys) -> None:
    run = run_case("timeout")
    assert "bought_reconciled" in run.output

    assert main(["--case", "timeout"]) == 0
    out = capsys.readouterr().out
    assert "bought_reconciled" in out


def test_output_hides_card_ids_and_signing_key(capsys) -> None:
    assert main(["--all"]) == 0
    out = capsys.readouterr().out
    assert "card_" not in out
    assert KEY.decode("utf-8") not in out


def test_oracle_header(capsys) -> None:
    assert main(["--case", "happy"]) == 0
    out = capsys.readouterr().out
    assert "model: oracle (code gold policy)" in out
