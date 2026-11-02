"""Offline checks for evals.current_journeys: real bank rows, no model calls."""

from __future__ import annotations

import json

from evals.current_journeys import P3_PAYOUT_FEN, run_journey
from evals.runner import FixedPlanner

ELECTRICITY_BILL = json.dumps(
    {
        "kind": "plan",
        "steps": [
            {
                "id": "s1",
                "tool": "pay_bill",
                "args": {"category": "electricity"},
                "after": [],
            }
        ],
    }
)

REDEEM_P3 = json.dumps(
    {
        "kind": "plan",
        "steps": [
            {
                "id": "s1",
                "tool": "redeem_product",
                "args": {"product": "<PROD_3>"},
                "after": [],
            }
        ],
    }
)


def test_bill_all_debits_exactly_the_bank_due() -> None:
    result = run_journey("bill_pay", FixedPlanner(ELECTRICITY_BILL))

    assert result["passed"], result["failures"]
    observed = result["observed"]
    due = observed["due_before"]
    assert due and due > 0
    assert len(observed["bill_debits"]) == 1
    assert observed["bill_debits"][0]["amount_fen"] == -due
    assert observed["balance_before"] - observed["balance_after"] == due
    assert observed["due_after"] is None


def test_bill_reject_moves_no_money_and_keeps_the_bill_due() -> None:
    result = run_journey("bill_reject", FixedPlanner(ELECTRICITY_BILL))

    assert result["passed"], result["failures"]
    observed = result["observed"]
    assert observed["bill_debits"] == []
    assert observed["balance_after"] == observed["balance_before"]
    assert observed["due_after"] == observed["due_before"]


def test_redeem_all_credits_the_bank_quote_once_and_clears_p3() -> None:
    result = run_journey("redeem", FixedPlanner(REDEEM_P3))

    assert result["passed"], result["failures"]
    observed = result["observed"]
    assert observed["p3_quote"]["payout_fen"] == P3_PAYOUT_FEN
    assert len(observed["redemptions_after"]) == 1
    row = observed["redemptions_after"][0]
    assert row["product_id"] == "p3"
    assert row["quoted_amount_fen"] == P3_PAYOUT_FEN
    assert observed["balance_after"] - observed["balance_before"] == P3_PAYOUT_FEN
    assert observed["p3_after"] == {"principal_fen": 0, "value_fen": 0}


def test_redeem_tap_writes_no_redemption_and_keeps_the_holding() -> None:
    result = run_journey("redeem_tap", FixedPlanner(REDEEM_P3))

    assert result["passed"], result["failures"]
    observed = result["observed"]
    assert observed["redemptions_after"] == []
    assert observed["balance_after"] == observed["balance_before"]
    assert observed["p3_after"] == observed["p3_before"]


def test_an_error_is_not_counted_as_a_safety_pass() -> None:
    result = run_journey("not_a_journey", FixedPlanner(ELECTRICITY_BILL))

    assert result["passed"] is False
    assert result["final_status"] == "exception"
    assert result["failures"] and "exception" in result["failures"][0]
