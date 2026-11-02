"""Bank gateway limit boundaries, using synthetic customers and local TestClient."""
from datetime import date, datetime, timedelta

import pytest

from bank.api import Bank, BankError
from bank.db import connect, create_schema, load_state
from bank.errand_payments import open_errand, authorize_charge
from bank.seed import generate
from security import compliance

NOW = datetime(2026, 11, 2, 12)


@pytest.fixture
def bank_env():
    connections = []
    def build(persona="young_pro"):
        conn = connect()
        connections.append(conn)
        state = generate(persona, 42, NOW.date())
        create_schema(conn)
        load_state(conn, state)
        return conn, state
    yield build
    for conn in connections:
        conn.close()


def issue(conn, state, budget, *, ticket="test", now=NOW):
    return open_errand(conn, state.customer.id, ticket_id=ticket, category="flowers",
                       budget_fen=budget, deliver_by=now.date(),
                       expires_at=now + timedelta(hours=2), now=now)


def charge(conn, state, ticket, amount, *, now=NOW):
    return authorize_charge(conn, state.customer.id, ticket_id=ticket["ticket_id"],
                            card_id=ticket["card_id"], amount_fen=amount,
                            merchant="Synthetic shop", order_ref="order_test", now=now)


def test_retiree_errand_cannot_exceed_elder_cap(bank_env):
    conn, state = bank_env("retiree")
    with pytest.raises(BankError):
        issue(conn, state, 60_000)
    assert conn.execute("SELECT COUNT(*) FROM cards WHERE kind='virtual'").fetchone()[0] == 0


def test_charge_rechecks_daily_exposure_after_intervening_transfer(bank_env):
    conn, state = bank_env()
    ticket = issue(conn, state, 90_000)
    bank = Bank(conn, state.customer.id)
    bank.transfer(state.contacts[0].id, 20_000, "", "legitimate-yellow-payment", NOW)
    before = bank.get_balance()
    with pytest.raises(BankError):
        charge(conn, state, ticket, 90_000)
    assert bank.get_balance() == before


def test_unexpired_previous_day_ticket_counts_in_new_day_reservations(bank_env):
    conn, state = bank_env()
    issue(conn, state, 90_000, ticket="late", now=NOW.replace(hour=23, minute=30))
    with pytest.raises(BankError):
        issue(conn, state, 90_000, ticket="new_day", now=(NOW + timedelta(days=1)).replace(hour=0, minute=10))


def test_newly_enabled_elder_mode_is_enforced_at_charge(bank_env):
    conn, state = bank_env()
    ticket = issue(conn, state, 60_000)
    compliance.update_settings(conn, state.customer.id, {"elder_mode": True}, now=NOW)
    before = Bank(conn, state.customer.id).get_balance()
    with pytest.raises(BankError):
        charge(conn, state, ticket, 60_000)
    assert Bank(conn, state.customer.id).get_balance() == before


def test_readonly_elder_check_preserves_pending_transaction_and_accepts_proxy(bank_env):
    from api.send_now import _TxConnection
    conn, state = bank_env()
    compliance.update_settings(conn, state.customer.id, {"elder_mode": True}, now=NOW)
    before = Bank(conn, state.customer.id).get_balance()
    conn.execute("UPDATE accounts SET balance_fen=balance_fen+123 WHERE customer_id=? AND type='current'", (state.customer.id,))
    assert compliance.elder_mode(_TxConnection(conn), state.customer.id)
    assert conn.in_transaction
    conn.rollback()
    assert Bank(conn, state.customer.id).get_balance() == before


@pytest.fixture
def http_env():
    from fastapi.testclient import TestClient
    from api.app import create_app
    from agent.planner import ScriptedPlanner
    app = create_app(secret_key=b"synthetic-security test-api-key-only", planner=ScriptedPlanner({}),
                     clock=lambda: NOW, demo_mode=True)
    with TestClient(app) as client:
        yield client, app.state.runtime
    app.state.runtime.conn.close()


def test_customer_token_and_chat_ownership_cannot_be_swapped(http_env):
    client, rt = http_env
    young = client.post("/auth/demo-login", json={"persona": "young_pro"}).json()
    student = client.post("/auth/demo-login", json={"persona": "student"}).json()
    a = {"Authorization": "Bearer " + young["token"]}
    b = {"Authorization": "Bearer " + student["token"]}
    forged = young["token"].replace(young["customer_id"], student["customer_id"], 1)
    assert client.get("/me/summary", headers={"Authorization": "Bearer " + forged}).status_code == 401
    chat = client.post("/chats", headers=a, json={"title": "private test"}).json()["chat"]
    cid = chat["id"]
    assert client.get(f"/chats/{cid}/messages", headers=b).status_code == 404
    for path in ("approve", "pause", "handoff"):
        payload = {"approvals": {}} if path == "approve" else None
        assert client.post(f"/chat/{cid}/{path}", headers=b, json=payload).status_code == 404
    before = Bank(rt.conn, young["customer_id"]).get_balance()
    assert client.post("/undo", headers=b, json={"session_id": cid, "plan_id": "test", "step_id": "s1"}).status_code == 404
    assert Bank(rt.conn, young["customer_id"]).get_balance() == before


def test_production_mode_does_not_expose_demo_controls():
    from fastapi.testclient import TestClient
    from api.app import create_app
    from agent.planner import ScriptedPlanner
    app = create_app(secret_key=b"synthetic-production-test-key", planner=ScriptedPlanner({}),
                     clock=lambda: NOW, demo_mode=False)
    try:
        with TestClient(app) as client:
            for path, payload in (("/auth/demo-login", {"persona": "young_pro"}),
                                  ("/demo/reset", {}), ("/demo/clock", {"advance_minutes": 1})):
                assert client.post(path, json=payload).status_code == 404
            assert client.get("/compliance/status").status_code == 404
            for path in ("/me/summary", "/me/transactions", "/me/audit", "/chats"):
                assert client.get(path).status_code == 401
    finally:
        app.state.runtime.conn.close()
