"""Tests for the trusted bank-side errand saga (errand/workflow.py)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import date, datetime, time, timedelta

import pytest

from bank.api import BankError
from bank.db import connect, create_schema, load_state
from bank.errand_payments import errand_status, open_errand
from bank.seed import generate
from errand.agent import OracleModel
from errand.oracle import gold_actions
from errand.scenarios import generate_scenario
from errand.workflow import run_bank_errand

TODAY = date(2026, 10, 17)
SEED = 42
KEY = b"workflow-test-key"


@pytest.fixture
def env():
    state = generate("young_pro", SEED, TODAY)
    conn = connect()
    create_schema(conn)
    load_state(conn, state)
    try:
        yield conn, state
    finally:
        conn.close()


def _balance(conn: sqlite3.Connection, customer_id: str) -> int:
    return int(
        conn.execute(
            """
            SELECT balance_fen FROM accounts
            WHERE customer_id = ? AND type = 'current'
            """,
            (customer_id,),
        ).fetchone()[0]
    )


def _txn_count(
    conn: sqlite3.Connection, customer_id: str, category: str | None = None
) -> int:
    if category is None:
        row = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM transactions
            WHERE customer_id = ? AND category = ?
            """,
            (customer_id, category),
        ).fetchone()
    return int(row[0])


def _card_status(conn: sqlite3.Connection, customer_id: str, card_id: str) -> str:
    return str(
        conn.execute(
            "SELECT status FROM cards WHERE customer_id = ? AND id = ?",
            (customer_id, card_id),
        ).fetchone()["status"]
    )


def _scenario(kind: str):
    return next(
        scenario
        for scenario in (generate_scenario(seed) for seed in range(300))
        if scenario.tags[0] == kind
    )


def _run(conn, state, scenario, **overrides):
    now = datetime.combine(scenario.today, time(9, 0))
    model = overrides.pop("model", None)
    if model is None:
        model = OracleModel(
            gold_actions(replace(scenario, card_token="<VCARD_1>"))
        )
    return run_bank_errand(
        conn,
        state.customer.id,
        category=scenario.category,
        budget_fen=scenario.budget_fen,
        deliver_by=scenario.deliver_by,
        note=scenario.note,
        now=now,
        key=KEY,
        model=model,
        products=list(scenario.products),
        **overrides,
    )


def _ticket_state(conn, customer_id, ticket_id):
    return errand_status(conn, customer_id, ticket_id)


def _ticket_row_count(conn, customer_id, ticket_id) -> int:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'errand_tickets'"
    ).fetchone()
    if exists is None:
        return 0
    row = conn.execute(
        """
        SELECT COUNT(*) FROM errand_tickets
        WHERE customer_id = ? AND ticket_id = ?
        """,
        (customer_id, ticket_id),
    ).fetchone()
    return int(row[0])


class _KeyboardInterruptModel:
    def next_action(self, messages: list[dict]) -> str:
        raise KeyboardInterrupt


class _BoomAfterCheckoutModel:
    """Replay a fixed list of actions, then blow up once checkout was sent."""

    def __init__(self, actions: list[dict]) -> None:
        self.actions = list(actions)
        self.calls = 0
        self._returned_checkout = False

    def next_action(self, messages: list[dict]) -> str:
        if self._returned_checkout:
            raise RuntimeError("model exploded after checkout")
        index = self.calls
        self.calls += 1
        action = (
            self.actions[index]
            if index < len(self.actions)
            else {"action": "abort", "reason": "no_suitable_product"}
        )
        if action.get("action") == "checkout":
            self._returned_checkout = True
        return json.dumps(action, ensure_ascii=False)


def test_normal_buy_charges_once_and_closes_the_card(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    scenario = _scenario("normal")
    before_balance = _balance(conn, customer_id)
    before_shopping = _txn_count(conn, customer_id, "shopping")

    result = _run(conn, state, scenario)

    assert result.outcome == "bought"
    assert result.order is not None
    assert result.charged_fen == result.order.total_fen
    assert _balance(conn, customer_id) == before_balance - result.order.total_fen
    assert _txn_count(conn, customer_id, "shopping") == before_shopping + 1

    st = _ticket_state(conn, customer_id, result.ticket_id)
    assert st["status"] == "charged"
    assert _card_status(conn, customer_id, str(st["card_id"])) == "closed"
    assert [event["step"] for event in result.events] == [
        "opened",
        "charged",
        "agent_done",
        "reconciled",
    ]


def test_impossible_scenario_aborts_and_closes_everything(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    scenario = _scenario("impossible")
    before_balance = _balance(conn, customer_id)
    before_txns = _txn_count(conn, customer_id)

    result = _run(conn, state, scenario)

    assert result.outcome == "aborted"
    assert result.charged_fen == 0
    assert result.order is None
    assert _balance(conn, customer_id) == before_balance
    assert _txn_count(conn, customer_id) == before_txns

    st = _ticket_state(conn, customer_id, result.ticket_id)
    assert st["status"] == "closed"
    assert _card_status(conn, customer_id, str(st["card_id"])) == "closed"
    assert result.events[-1]["step"] == "closed"


def test_checkout_surcharge_is_terms_changed_and_moves_no_money(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    scenario = _scenario("normal")
    before_balance = _balance(conn, customer_id)

    result = _run(conn, state, scenario, checkout_surcharge_fen=500)

    assert "terms_changed" in result.declines
    assert result.charged_fen == 0
    assert result.outcome == "aborted"
    assert result.abort_reason == "payment_declined"
    # Code ends the run at the decline; the model gets no further turns.
    assert result.transcript[-1]["role"] == "assistant"
    assert '"checkout"' in result.transcript[-1]["content"]
    assert _balance(conn, customer_id) == before_balance

    st = _ticket_state(conn, customer_id, result.ticket_id)
    assert st["status"] == "closed"
    assert _card_status(conn, customer_id, str(st["card_id"])) == "closed"


def test_lost_confirmation_after_charge_reconciles_from_the_ledger(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    scenario = _scenario("normal")
    actions = gold_actions(replace(scenario, card_token="<VCARD_1>"))
    checkout_index = next(
        index
        for index, action in enumerate(actions)
        if action["action"] == "checkout"
    )
    model = OracleModel(
        actions[: checkout_index + 1]
        + [{"action": "abort", "reason": "no_suitable_product"}]
    )
    before_balance = _balance(conn, customer_id)
    before_shopping = _txn_count(conn, customer_id, "shopping")

    result = _run(
        conn, state, scenario, model=model, timeout_after_charge=True
    )

    assert result.outcome == "bought_reconciled"
    assert result.order is not None
    assert result.charged_fen == result.order.total_fen
    assert _balance(conn, customer_id) == before_balance - result.order.total_fen
    assert _txn_count(conn, customer_id, "shopping") == before_shopping + 1

    st = _ticket_state(conn, customer_id, result.ticket_id)
    assert st["status"] == "charged"
    assert _card_status(conn, customer_id, str(st["card_id"])) == "closed"


def test_guard_refuses_a_checkout_with_another_card_token(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    scenario = _scenario("normal")
    actions = []
    for action in gold_actions(replace(scenario, card_token="<VCARD_1>")):
        if action["action"] == "checkout":
            action = {**action, "card_token": "<VCARD_9>"}
        actions.append(action)
    before_balance = _balance(conn, customer_id)

    result = _run(conn, state, scenario, model=OracleModel(actions))

    assert "wrong_card" in result.refusals
    assert result.charged_fen == 0
    assert _balance(conn, customer_id) == before_balance


def test_injection_scenario_still_buys_the_expected_product(env) -> None:
    conn, state = env
    scenario = _scenario("injection")
    assert scenario.expected.startswith("buy:")
    expected_id = scenario.expected.split(":", 1)[1]

    result = _run(conn, state, scenario)

    assert result.outcome == "bought"
    assert result.order is not None
    assert result.order.product_id == expected_id


def test_budget_above_balance_propagates_and_leaves_no_ticket(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    scenario = _scenario("normal")
    now = datetime.combine(scenario.today, time(9, 0))
    open_errand(
        conn,
        customer_id,
        ticket_id="T_seed",
        category=scenario.category,
        budget_fen=100,
        deliver_by=scenario.deliver_by,
        expires_at=now + timedelta(hours=1),
        now=now,
    )
    conn.execute(
        """
        UPDATE accounts SET balance_fen = 500
        WHERE customer_id = ? AND type = 'current'
        """,
        (customer_id,),
    )
    conn.commit()

    with pytest.raises(BankError) as exc:
        _run(conn, state, scenario, ticket_id="T_attempt")

    assert exc.value.code == "insufficient_funds"
    rows = conn.execute(
        """
        SELECT COUNT(*) FROM errand_tickets
        WHERE customer_id = ? AND ticket_id = ?
        """,
        (customer_id, "T_attempt"),
    ).fetchone()
    assert int(rows[0]) == 0


def test_events_and_transcript_do_not_leak_bank_secrets(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    scenario = _scenario("normal")

    result = _run(conn, state, scenario)

    st = _ticket_state(conn, customer_id, result.ticket_id)
    card_id = str(st["card_id"])
    events_blob = json.dumps(result.events)

    assert KEY.decode("utf-8") not in events_blob
    assert card_id not in events_blob
    assert "card_" not in events_blob

    transcript_blob = json.dumps(result.transcript, ensure_ascii=False)
    assert card_id not in transcript_blob
    assert result.delivery_ref not in transcript_blob


def test_keyboard_interrupt_is_reraised_after_closing_ticket_and_card(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    scenario = _scenario("normal")

    with pytest.raises(KeyboardInterrupt):
        _run(
            conn,
            state,
            scenario,
            model=_KeyboardInterruptModel(),
            ticket_id="T_kb",
        )

    st = _ticket_state(conn, customer_id, "T_kb")
    assert st["status"] == "closed"
    assert _card_status(conn, customer_id, str(st["card_id"])) == "closed"


def test_model_error_after_charge_reconciles_instead_of_needs_review(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    scenario = _scenario("normal")
    model = _BoomAfterCheckoutModel(
        gold_actions(replace(scenario, card_token="<VCARD_1>"))
    )
    before_balance = _balance(conn, customer_id)

    result = _run(conn, state, scenario, model=model)

    assert result.outcome == "bought_reconciled"
    assert result.order is not None
    assert result.charged_fen == result.order.total_fen
    assert result.abort_reason is None
    assert _balance(conn, customer_id) == before_balance - result.charged_fen
    assert sum(1 for event in result.events if event["step"] == "charged") == 1

    st = _ticket_state(conn, customer_id, result.ticket_id)
    assert st["status"] == "charged"
    assert _card_status(conn, customer_id, str(st["card_id"])) == "closed"


def test_charge_after_ticket_expiry_declines_and_closes(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    scenario = _scenario("normal")
    now = datetime.combine(scenario.today, time(9, 0))
    before_balance = _balance(conn, customer_id)

    result = _run(
        conn,
        state,
        scenario,
        clock=lambda: now + timedelta(hours=2),
    )

    assert "ticket_expired" in result.declines
    assert any(
        event.get("step") == "declined" and event.get("code") == "ticket_expired"
        for event in result.events
    )
    assert result.charged_fen == 0
    assert _balance(conn, customer_id) == before_balance

    st = _ticket_state(conn, customer_id, result.ticket_id)
    assert st["status"] == "closed"
    assert _card_status(conn, customer_id, str(st["card_id"])) == "closed"


def test_bad_deliver_by_or_key_leaves_no_errand_ticket_row(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    scenario = _scenario("normal")
    now = datetime.combine(scenario.today, time(9, 0))
    model = OracleModel(gold_actions(replace(scenario, card_token="<VCARD_1>")))
    base = dict(
        category=scenario.category,
        budget_fen=scenario.budget_fen,
        deliver_by=scenario.deliver_by,
        note=scenario.note,
        now=now,
        key=KEY,
        model=model,
        products=list(scenario.products),
        ticket_id="T_bad",
    )

    for override in ({"deliver_by": now}, {"key": b""}):
        with pytest.raises(ValueError):
            run_bank_errand(conn, customer_id, **{**base, **override})
        assert _ticket_row_count(conn, customer_id, "T_bad") == 0
