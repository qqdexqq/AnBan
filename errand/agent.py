"""Runner for the online errand agent (docs/ERRAND_AGENT.md).

The model only proposes one JSON action per turn. Code parses it, the guard
checks it against the ticket and the shop, and only then does the shop see it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Protocol

import httpx

from errand.guard import GuardState, check_action, verify_receipt
from errand.prompts import ERRAND_SYSTEM_PROMPT
from errand.scenarios import Scenario
from errand.shop import MockShop, Order
from errand.ticket import ErrandTicket, issue_ticket, ticket_brief


class ErrandModel(Protocol):
    def next_action(self, messages: list[dict]) -> str:  # pragma: no cover - protocol
        ...


class OpenAICompatibleErrandModel:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint.

    The key is sent only when supplied and is never logged or put into an
    exception message.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.client = client

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, messages: list[dict]) -> dict[str, object]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def next_action(self, messages: list[dict]) -> str:
        payload = self._payload(messages)
        headers = self._headers()
        try:
            if self.client is None:
                response = httpx.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
            else:
                response = self.client.post(
                    self.endpoint, json=payload, headers=headers
                )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            # Never include headers (which may hold the key) in the error.
            raise RuntimeError(
                f"errand model HTTP error: {type(exc).__name__}"
            ) from None
        except ValueError:
            raise RuntimeError("errand model returned invalid JSON") from None

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("errand model returned an invalid response") from None
        if not isinstance(content, str):
            raise RuntimeError("errand model returned non-text content") from None
        return content.strip()


class OracleModel:
    """Replay a fixed list of actions, then abort with no_suitable_product."""

    def __init__(self, actions: list[dict]) -> None:
        self.actions = list(actions)
        self.calls = 0

    def next_action(self, messages: list[dict]) -> str:
        index = self.calls
        self.calls += 1
        if index < len(self.actions):
            return json.dumps(self.actions[index], ensure_ascii=False)
        return '{"action":"abort","reason":"no_suitable_product"}'


@dataclass
class RunResult:
    status: str
    order: Order | None
    abort_reason: str | None
    steps: int
    transcript: list[dict]
    refusals: list[str]
    actions: list[dict | None]


def parse_action(text: str) -> dict | None:
    """Extract one JSON action object from raw model text."""
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if cleaned.startswith("<think>"):
        end = cleaned.find("</think>")
        if end == -1:
            return None
        cleaned = cleaned[end + len("</think>") :].strip()
    if cleaned.startswith("```"):
        newline = cleaned.find("\n")
        if newline == -1:
            cleaned = cleaned[3:]
        else:
            cleaned = cleaned[newline + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()
    try:
        action = json.loads(cleaned)
    except (ValueError, TypeError):
        return None
    return action if isinstance(action, dict) else None


def run_errand(
    ticket: ErrandTicket,
    shop: MockShop,
    model: ErrandModel,
    today: date,
    max_steps: int = 12,
    state: GuardState | None = None,
) -> RunResult:
    messages: list[dict] = [
        {"role": "system", "content": ERRAND_SYSTEM_PROMPT},
        {"role": "user", "content": ticket_brief(ticket, today)},
    ]
    state = GuardState() if state is None else state
    refusals: list[str] = []
    actions: list[dict | None] = []
    status = "failed"
    order: Order | None = None
    abort_reason: str | None = None

    while state.steps < max_steps:
        text = model.next_action(messages)
        messages.append({"role": "assistant", "content": text})
        action = parse_action(text)
        actions.append(action)

        if action is None:
            state.steps += 1
            refusal = "bad_action"
        else:
            refusal = check_action(action, ticket, shop, today, state)

        if refusal == "too_many_steps":
            refusals.append(refusal)
            status = "failed"
            break

        if refusal is not None:
            refusals.append(refusal)
            messages.append(
                {"role": "user", "content": f"GUARD REFUSED: {refusal}"}
            )
            continue

        result = shop.step(action)
        if result.done:
            if action["action"] == "finish":
                code = verify_receipt(result.order, ticket, shop, today)
                if code is None:
                    status = "bought"
                    order = result.order
                else:
                    refusals.append(code)
                    status = "failed"
            elif action["action"] == "abort":
                status = "aborted"
                abort_reason = action["reason"]
            break
        if result.error == "payment_declined":
            # The bank refused the charge and the guard allows no second
            # checkout, so code ends the errand instead of the model.
            status = "aborted"
            abort_reason = "payment_declined"
            break

        messages.append({"role": "user", "content": result.observation})

    return RunResult(
        status=status,
        order=order,
        abort_reason=abort_reason,
        steps=state.steps,
        transcript=messages,
        refusals=refusals,
        actions=actions,
    )


def scenario_ticket(
    scenario: Scenario, key: bytes = b"errand-dataset"
) -> tuple[ErrandTicket, MockShop]:
    ticket, _signature = issue_ticket(
        category=scenario.category,
        budget_fen=scenario.budget_fen,
        deliver_by=scenario.deliver_by,
        now=datetime.combine(scenario.today, time(9, 0)),
        key=key,
        card_token=scenario.card_token,
        note=scenario.note,
        ticket_id=f"T{scenario.seed:010x}",
    )
    return ticket, MockShop(list(scenario.products), scenario.today)
