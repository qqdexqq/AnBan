"""Customer negation and contradictory card instructions must not cause writes."""
import json
import re

import pytest

from bank import controls
from tests.test_graph import make_env


@pytest.mark.parametrize("text", [
    "don't restore my card limit", "do not reset my card limit",
    "don’t restore my card limit", "不要恢复我的卡限额",
])
def test_negated_restore_never_changes_a_limit(make_env, text):
    env = make_env()
    bank = env.services.bank_for(env.customer_id)
    card = bank.list_cards()[0]
    bank.set_card_limit(card.id, 20_000)
    bank.set_card_limit(card.id, 30_000)
    env.agent.start(env.customer_id, text)
    assert bank.list_cards()[0].online_limit_fen == 30_000


@pytest.mark.parametrize("text", [
    "don't disable online shopping on my card",
    "do not block online shopping on my card",
    "不要关闭我的卡网上支付",
])
def test_negated_card_control_never_changes_setting(make_env, text):
    env = make_env()
    bank = env.services.bank_for(env.customer_id)
    card = bank.list_cards()[0]
    assert controls.get_controls(env.conn, env.customer_id, card.id)["online"]
    env.agent.start(env.customer_id, text)
    assert controls.get_controls(env.conn, env.customer_id, card.id)["online"]


def test_model_cannot_reintroduce_negated_restore_on_fallback(make_env):
    env = make_env()
    bank = env.services.bank_for(env.customer_id)
    card = bank.list_cards()[0]
    bank.set_card_limit(card.id, 20_000)
    bank.set_card_limit(card.id, 30_000)
    env.planner.respond = lambda user: json.dumps({"kind": "plan", "steps": [
        {"id": "s1", "tool": "restore_card_limit", "args": {}, "after": []}
    ]})
    env.agent.start(env.customer_id, "don't restore my card limit")
    assert bank.list_cards()[0].online_limit_fen == 30_000


def test_contradictory_card_actions_do_not_propose_only_unfreeze(make_env):
    env = make_env()
    bank = env.services.bank_for(env.customer_id)
    card = bank.list_cards()[0]
    bank.freeze_card(card.id)
    result = env.agent.start(env.customer_id, "freeze my card and unfreeze my card")
    assert result.status != "needs_approval", result.reply
    assert bank.list_cards()[0].status == "frozen"
