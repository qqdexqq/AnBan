"""End-to-end wiring of unattended standing orders through the HTTP edge."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from agent.planner import ScriptedPlanner
from api.app import create_app
from bank.api import Bank

NOW = datetime(2026, 10, 17, 14, 0, 0)
SECRET = b"k" * 32
DAY_OF_MONTH = 20


class FixedClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now += timedelta(minutes=minutes)


def _refuse(user: str) -> str:
    return json.dumps({"kind": "refuse", "reason": "out_of_scope"})


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(NOW)


@pytest.fixture
def client(clock: FixedClock):
    app = create_app(
        secret_key=SECRET,
        planner=ScriptedPlanner(_refuse),
        clock=clock,
        demo_mode=True,
    )
    with TestClient(app) as test_client:
        yield test_client
    app.state.runtime.conn.close()


def _login(client: TestClient, persona: str = "young_pro") -> tuple[str, str]:
    response = client.post("/auth/demo-login", json={"persona": persona})
    assert response.status_code == 200, response.text
    return response.json()["token"], response.json()["customer_id"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _contact(conn, customer_id: str):
    return conn.execute(
        "SELECT id, name, account_no FROM contacts WHERE customer_id = ? ORDER BY id LIMIT 1",
        (customer_id,),
    ).fetchone()


def _order(summary: dict, scheduled_id: str) -> dict:
    return next(
        order
        for order in summary["standing_orders"]
        if order["scheduled_id"] == scheduled_id
    )


def _transfer_count(conn, customer_id: str, scheduled_id: str) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(*) FROM transfers
            WHERE customer_id = ? AND idempotency_key LIKE ?
            """,
            (customer_id, f"standing_order:{scheduled_id}:%"),
        ).fetchone()[0]
    )


def test_summary_runs_a_due_standing_order_exactly_once(client: TestClient) -> None:
    token, customer_id = _login(client)
    rt = client.app.state.runtime
    contact = _contact(rt.conn, customer_id)
    scheduled = Bank(rt.conn, customer_id).schedule_transfer(
        contact["id"], 1_000, DAY_OF_MONTH, "rent"
    )

    first = client.get("/me/summary", headers=_auth(token))
    assert first.status_code == 200, first.text
    order = _order(first.json(), scheduled.id)
    assert order["next_due"] is not None
    assert order["last_run"] is None
    assert "contact_id" not in order
    assert contact["id"] not in first.text
    assert contact["account_no"] not in first.text
    assert order["payee"].endswith(contact["account_no"][-4:])

    advanced = client.post("/demo/clock", json={"advance_minutes": 4 * 24 * 60})
    assert advanced.status_code == 200, advanced.text
    # Session tokens live 8 hours, so re-login after the jump; the customer
    # and the standing order stay the same.
    token, relogged_customer = _login(client)
    assert relogged_customer == customer_id

    paid = client.get("/me/summary", headers=_auth(token)).json()
    order = _order(paid, scheduled.id)
    assert order["last_run"]["status"] == "paid"
    assert _transfer_count(rt.conn, customer_id, scheduled.id) == 1

    again = client.get("/me/summary", headers=_auth(token)).json()
    assert _order(again, scheduled.id)["last_run"]["status"] == "paid"
    assert _transfer_count(rt.conn, customer_id, scheduled.id) == 1

    runs = [
        entry
        for entry in rt.services.audit.entries_for(customer_id)
        if entry.kind == "RUN"
    ]
    assert len(runs) == 1
    assert runs[0].session_id == "standing-orders"
    assert runs[0].plan_id.startswith(f"so:{scheduled.id}:")
    assert runs[0].payload["tool"] == "standing_order"
    assert runs[0].payload["status"] == "paid"
    assert rt.services.audit.verify().ok


def test_chat_turn_runs_and_audits_a_due_standing_order(client: TestClient) -> None:
    token, customer_id = _login(client)
    rt = client.app.state.runtime
    contact = _contact(rt.conn, customer_id)
    scheduled = Bank(rt.conn, customer_id).schedule_transfer(
        contact["id"], 1_000, DAY_OF_MONTH, "rent"
    )

    # The first pass only marks the order as seen.
    first = client.get("/me/summary", headers=_auth(token)).json()
    assert _order(first, scheduled.id)["last_run"] is None

    assert client.post(
        "/demo/clock", json={"advance_minutes": 4 * 24 * 60}
    ).status_code == 200
    # Session tokens live 8 hours, so re-login after the jump.
    token, relogged_customer = _login(client)
    assert relogged_customer == customer_id

    chat = client.post("/chat", json={"text": "hello"}, headers=_auth(token))
    assert chat.status_code == 200, chat.text

    summary = client.get("/me/summary", headers=_auth(token)).json()
    assert _order(summary, scheduled.id)["last_run"]["status"] == "paid"
    assert _transfer_count(rt.conn, customer_id, scheduled.id) == 1

    runs = [
        entry
        for entry in rt.services.audit.entries_for(customer_id)
        if entry.kind == "RUN"
    ]
    assert len(runs) == 1
    assert runs[0].plan_id.startswith(f"so:{scheduled.id}:")
    assert rt.services.audit.verify().ok


def test_summary_returns_recently_due_reminders_and_injected_now(
    client: TestClient, clock: FixedClock
) -> None:
    token, customer_id = _login(client)
    rt = client.app.state.runtime
    due_on = (clock().date() - timedelta(days=1)).isoformat()
    rt.conn.execute(
        """CREATE TABLE IF NOT EXISTS agent_reminders (
            customer_id TEXT NOT NULL, idempotency_key TEXT NOT NULL,
            due_on TEXT, note TEXT, created_at TEXT NOT NULL,
            PRIMARY KEY(customer_id, idempotency_key))"""
    )
    rt.conn.execute(
        """INSERT INTO agent_reminders
           (customer_id, idempotency_key, due_on, note, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (customer_id, "test:due-yesterday", due_on, "renew passport", clock().isoformat()),
    )
    rt.conn.commit()

    response = client.get("/me/summary", headers=_auth(token))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["now"] == clock().isoformat()
    assert any(
        reminder["due_on"] == due_on and reminder["note"] == "renew passport"
        for reminder in body["reminders"]
    )
