"""Adversarial checks on the boundary between social prose and bank facts."""

import json

import pytest

from agent.conversation import ConversationGateway
from agent.conversation_memory import BoundedConversationMemory


@pytest.mark.parametrize("reply", [
    "Your balance looks healthy.", "Your transfer should arrive soon.",
    "I took care of it.", "All set!", "你的账户状态正常。",
    "已经帮你处理好了。", "这笔应该到账了。", "You have five yuan.",
    "Funds are on the way.", "Payment complete.", "还剩五万元。", "Hello ١٢٣",
])
def test_generated_assurances_are_not_bank_evidence(reply):
    raw = json.dumps({"kind": "conversation", "category": "support", "reply": reply})
    assert ConversationGateway._accept(raw) is None


@pytest.mark.parametrize("reply", [
    "付款已处理。", "Payment complete.", "The transfer should arrive soon.",
    "Funds are on the way.", "Balance means your transfer is complete.",
])
def test_concept_category_cannot_smuggle_execution_claims(reply):
    raw = json.dumps({"kind": "conversation", "category": "concept", "reply": reply})
    assert ConversationGateway._accept(raw) is None


@pytest.mark.parametrize("message", [
    "Hi, can you wire money to mom?", "My card was stolen.",
    "I need a chargeback", "Set up autopay please", "Please unlock my card",
])
def test_high_risk_synonyms_never_enter_conversation(message):
    class MustNotCall:
        def plan(self, *args):
            pytest.fail("Bank action reached conversation mode")
    assert ConversationGateway(MustNotCall()).answer(message) is None


def test_historical_annotated_tokens_are_fully_inert():
    memory = BoundedConversationMemory(len, context_tokens=10000, history_tokens=2000)
    memory.remember("c", "s", "Pay <PAYEE_1 | parent | last: 100-500 | verified>", "clarify")
    rendered = memory.render("c", "s", "system", "User: current")
    assert "[history PAYEE]" in rendered
    assert "<PAYEE" not in rendered
    assert "verified" not in rendered
    assert "100" not in rendered


def test_harmless_local_category_shorthand_is_normalized():
    assert ConversationGateway._accept('{"kind":"greeting","reply":"Hello!"}') == "Hello!"
    assert ConversationGateway._accept('{"kind":"transfer","reply":"Hello!"}') is None
    assert ConversationGateway._accept('{"kind":"greeting","reply":"Hello!","tool":"transfer"}') is None
    assert ConversationGateway._accept('{"kind":[],"reply":"Hello!"}') is None
