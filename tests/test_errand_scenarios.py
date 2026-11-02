"""Tests for errand/scenarios.py and errand/oracle.py."""

from __future__ import annotations

from collections import Counter

import pytest

from errand.oracle import best_choice, gold_actions, is_phishing, search_query
from errand.scenarios import generate_scenario
from errand.shop import MockShop

SEEDS = range(2000)
KINDS = ("normal", "injection", "phishing", "impossible")
TARGET_SHARES = {
    "normal": 0.55,
    "injection": 0.15,
    "phishing": 0.10,
    "impossible": 0.20,
}
ABORT_REASONS = (
    "no_suitable_product",
    "over_budget",
    "late_delivery",
    "out_of_stock",
    "suspicious_page",
)


def _kind(scenario) -> str:
    for kind in KINDS:
        if kind in scenario.tags:
            return kind
    raise AssertionError(f"no kind tag in {scenario.tags}")


@pytest.fixture(scope="module")
def scenarios():
    return [generate_scenario(seed) for seed in SEEDS]


def test_generate_scenario_is_deterministic():
    assert generate_scenario(1234) == generate_scenario(1234)


def test_kind_shares(scenarios):
    counts = Counter(_kind(scenario) for scenario in scenarios)
    for kind, target in TARGET_SHARES.items():
        share = counts[kind] / len(scenarios)
        assert abs(share - target) <= 0.08, (kind, share)


def test_expected_matches_kind(scenarios):
    for scenario in scenarios:
        kind = _kind(scenario)
        if kind == "impossible":
            assert scenario.expected.startswith("abort:"), scenario.seed
        elif kind in ("normal", "injection"):
            assert scenario.expected.startswith("buy:"), scenario.seed


def test_expected_is_best_choice(scenarios):
    for scenario in scenarios:
        assert scenario.expected == best_choice(scenario), scenario.seed


def test_abort_reasons_are_all_present(scenarios):
    reasons = Counter(
        scenario.expected.split(":", 1)[1]
        for scenario in scenarios
        if scenario.expected.startswith("abort:")
    )
    for reason in ABORT_REASONS:
        assert reasons[reason] >= 20, (reason, reasons[reason])


def test_gold_actions_replay(scenarios):
    for scenario in scenarios:
        shop = MockShop(list(scenario.products), scenario.today)
        actions = gold_actions(scenario)
        last = None
        for action in actions:
            result = shop.step(action)
            assert result.error is None, (scenario.seed, action, result.error)
            last = result
        assert last is not None
        assert last.done is True, scenario.seed

        checkouts = [action for action in actions if action["action"] == "checkout"]
        if scenario.expected.startswith("buy:"):
            product_id = scenario.expected.split(":", 1)[1]
            assert checkouts, scenario.seed
            assert checkouts[0]["card_token"] == scenario.card_token
            assert last.order is not None
            assert last.order.product_id == product_id, scenario.seed
            assert last.order.total_fen <= scenario.budget_fen, scenario.seed
            assert last.order.card_token == scenario.card_token
        else:
            assert not checkouts


def test_phishing_detection():
    assert is_phishing("pink roses") is False
    assert is_phishing("enter your PIN") is True


def test_search_query_uses_note_or_category(scenarios):
    for scenario in scenarios[:200]:
        query = search_query(scenario)
        if scenario.note:
            assert query == scenario.note
        elif "zh" in scenario.tags:
            assert query in ("花", "蛋糕")
        else:
            assert query in ("flowers", "cake")


def test_injection_never_reuses_the_ticket_card_token(scenarios):
    for scenario in scenarios:
        if "injection" not in scenario.tags:
            continue
        for product in scenario.products:
            assert scenario.card_token not in product.description
            assert not is_phishing(product.description), scenario.seed
