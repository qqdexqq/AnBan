"""Tests for errand/agent.py."""

from __future__ import annotations

import json

import httpx

from errand.agent import (
    OpenAICompatibleErrandModel,
    OracleModel,
    parse_action,
    run_errand,
    scenario_ticket,
)
from errand.oracle import gold_actions
from errand.scenarios import generate_scenario

SEEDS = range(300)


class _HelloModel:
    def next_action(self, messages: list[dict]) -> str:
        return "hello"


def _assert_alternating_transcript(transcript: list[dict]) -> None:
    roles = [message["role"] for message in transcript]
    assert roles[:2] == ["system", "user"]
    assert roles[-1] == "assistant"
    for index, role in enumerate(roles[2:], start=2):
        expected = "assistant" if (index - 2) % 2 == 0 else "user"
        assert role == expected, (index, role)


def test_oracle_replays_gold_actions_for_seeds():
    for seed in SEEDS:
        scenario = generate_scenario(seed)
        ticket, shop = scenario_ticket(scenario)
        result = run_errand(
            ticket, shop, OracleModel(gold_actions(scenario)), scenario.today
        )

        assert result.refusals == [], (seed, result.refusals)
        assert result.steps <= 12, seed
        _assert_alternating_transcript(result.transcript)
        assert len(result.actions) == len(
            [m for m in result.transcript if m["role"] == "assistant"]
        )

        if scenario.expected.startswith("buy:"):
            product_id = scenario.expected.split(":", 1)[1]
            assert result.status == "bought", seed
            assert result.order is not None, seed
            assert result.order.product_id == product_id, seed
            assert result.abort_reason is None, seed
        else:
            reason = scenario.expected.split(":", 1)[1]
            assert result.status == "aborted", seed
            assert result.abort_reason == reason, seed
            assert result.order is None, seed


def test_unparseable_actions_become_bad_action_refusals():
    scenario = generate_scenario(0)
    ticket, shop = scenario_ticket(scenario)
    result = run_errand(ticket, shop, _HelloModel(), scenario.today)

    assert result.status == "failed"
    assert result.steps == 12
    assert result.refusals == ["bad_action"] * 12
    assert result.actions == [None] * 12
    assert result.order is None
    assert result.abort_reason is None


def test_wrong_card_is_refused_before_checkout():
    scenario = next(
        candidate
        for candidate in (generate_scenario(seed) for seed in SEEDS)
        if candidate.card_token != "<VCARD_999>"
    )
    ticket, shop = scenario_ticket(scenario)
    actions = [
        {
            "action": "checkout",
            "card_token": "<VCARD_999>",
            "deliver_on": scenario.today.isoformat(),
        },
        {"action": "abort", "reason": "suspicious_page"},
    ]
    result = run_errand(ticket, shop, OracleModel(actions), scenario.today)

    assert "GUARD REFUSED: wrong_card" in [
        message["content"] for message in result.transcript
    ]
    assert result.refusals == ["wrong_card"]
    assert result.status == "aborted"
    assert shop.orders() == []


def test_parse_action_handles_think_blocks_and_fences():
    think = '<think>\n\n</think>\n\n{"action": "search", "query": "roses"}'
    assert parse_action(think) == {"action": "search", "query": "roses"}

    fenced = '```json\n{"action": "abort", "reason": "over_budget"}\n```'
    assert parse_action(fenced) == {"action": "abort", "reason": "over_budget"}

    plain_fence = '```\n{"action": "view", "product_id": "P001"}\n```'
    assert parse_action(plain_fence) == {"action": "view", "product_id": "P001"}

    assert parse_action('{"action": "finish", "order_id": "O000001"}') == {
        "action": "finish",
        "order_id": "O000001",
    }
    assert parse_action("[1,2]") is None
    assert parse_action("hello") is None


def test_openai_compatible_model_sends_expected_request():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        content = '  {"action": "abort", "reason": "over_budget"}  '
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "ticket"},
    ]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleErrandModel(
            "https://api.example.test/v1/",
            "errand-model",
            api_key="secret-key",
            client=client,
        )
        text = model.next_action(messages)

    assert text == '{"action": "abort", "reason": "over_budget"}'
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.example.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret-key"
    payload = json.loads(request.content)
    assert payload["model"] == "errand-model"
    assert payload["messages"] == messages
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 200
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_openai_compatible_model_omits_authorization_without_key():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"action": "abort"}'}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleErrandModel(
            "https://api.example.test/v1", "errand-model", client=client
        )
        assert model.next_action([]) == '{"action": "abort"}'
