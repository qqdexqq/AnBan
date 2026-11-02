"""HTTP tests for durable, customer-scoped chat history."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from agent.conversation_memory import BoundedConversationMemory
from agent.planner import ScriptedPlanner
from api.app import create_app

NOW = datetime(2026, 10, 17, 14, 0, 0)
SECRET = b"k" * 32


class FixedClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now += timedelta(minutes=minutes)


def _user_line(user: str) -> str:
    return user.rsplit("User:", 1)[-1]


def _scripted(user: str) -> str:
    line = _user_line(user)
    if "余额" in line:
        return json.dumps(
            {
                "kind": "plan",
                "steps": [
                    {"id": "s1", "tool": "get_balance", "args": {}, "after": []}
                ],
            },
            ensure_ascii=False,
        )

    payees = re.findall(r"<PAYEE_[A-Za-z0-9]+>", line)
    amounts = re.findall(r"<AMT_[A-Za-z0-9]+>", line)
    if not payees or not amounts:
        return json.dumps({"kind": "refuse", "reason": "out_of_scope"})
    return json.dumps(
        {
            "kind": "plan",
            "steps": [
                {
                    "id": "s1",
                    "tool": "can_cover",
                    "args": {"amount": amounts[0]},
                    "after": [],
                },
                {
                    "id": "s2",
                    "tool": "transfer",
                    "args": {"payee": payees[0], "amount": amounts[0]},
                    "after": ["s1"],
                },
            ],
        },
        ensure_ascii=False,
    )


def _new_app(db_path: str = ":memory:", clock: FixedClock | None = None):
    return create_app(
        db_path=db_path,
        secret_key=SECRET,
        planner=ScriptedPlanner(_scripted),
        clock=clock or FixedClock(NOW),
        demo_mode=True,
    )


@pytest.fixture
def client():
    app = _new_app()
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.state.runtime.conn.close()


def _login(client: TestClient, persona: str = "young_pro") -> str:
    response = client.post("/auth/demo-login", json={"persona": persona})
    assert response.status_code == 200, response.text
    return response.json()["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_reopened_history_uses_current_cancelled_transfer_status(client):
    token = _login(client)
    headers = _auth(token)
    started = client.post("/chat", json={"text": "给妈妈转300"}, headers=headers).json()
    sid = started["session_id"]
    step = started["approvals"][0]["step_id"]
    done = client.post(f"/chat/{sid}/approve", json={"approvals": {step: {"confirm": True}}}, headers=headers).json()
    assert done["status"] == "done"
    receipt = {"session_id": sid, "plan_id": done["plan_id"], "step_id": step}
    assert client.post("/undo", json=receipt, headers=headers).json()["status"] == "done"
    page = client.get(f"/chats/{sid}/messages", headers=headers).json()
    original = next(m["turn"] for m in page["messages"] if (m.get("turn") or {}).get("status") == "done" and m["turn"].get("plan_view"))
    assert next(s for s in original["plan_view"] if s["tool"] == "transfer")["status"] == "cancelled"


def test_history_requires_auth_and_hides_other_customers(client: TestClient) -> None:
    assert client.post("/chats", json={}).status_code == 401
    assert client.get("/chats").status_code == 401

    token = _login(client, "young_pro")
    other = _login(client, "student")
    chat = client.post(
        "/chats", json={"title": "Private"}, headers=_auth(token)
    ).json()["chat"]

    assert client.get(f"/chats/{chat['id']}/messages", headers=_auth(other)).status_code == 404
    assert client.get("/chats", headers=_auth(other)).json()["chats"] == []
    cross = client.post(
        "/chat",
        json={"text": "我的余额", "session_id": chat["id"]},
        headers=_auth(other),
    )
    assert cross.status_code == 404


def test_create_list_search_and_reopen(client: TestClient) -> None:
    token = _login(client)
    chat = client.post(
        "/chats", json={"title": "Budget plan"}, headers=_auth(token)
    ).json()["chat"]
    assert set(chat) == {"id", "title", "created_at", "updated_at"}
    assert chat["title"] == "Budget plan"

    first = client.post(
        "/chat",
        json={"text": "我的余额", "session_id": chat["id"]},
        headers=_auth(token),
    )
    assert first.status_code == 200, first.text
    assert first.json()["status"] == "done"
    assert first.json()["plan_id"] == f"{chat['id']}:p1"

    chats = client.get("/chats", headers=_auth(token)).json()["chats"]
    assert [item["id"] for item in chats] == [chat["id"]]
    assert [
        item["id"]
        for item in client.get(
            "/chats", params={"query": "Budget"}, headers=_auth(token)
        ).json()["chats"]
    ] == [chat["id"]]
    assert [
        item["id"]
        for item in client.get(
            "/chats", params={"query": "余额"}, headers=_auth(token)
        ).json()["chats"]
    ] == [chat["id"]]

    page = client.get(
        f"/chats/{chat['id']}/messages", headers=_auth(token)
    ).json()
    assert [item["role"] for item in page["messages"]] == ["user", "assistant"]
    assert page["messages"][0]["text"] == "我的余额"
    assert page["messages"][1]["turn"]["status"] == "done"

    second = client.post(
        "/chat",
        json={"text": "我的余额", "session_id": chat["id"]},
        headers=_auth(token),
    ).json()
    assert second["status"] == "done"
    assert second["plan_id"] == f"{chat['id']}:p2"
    reopened = client.get(f"/chats/{chat['id']}/messages", headers=_auth(token)).json()
    assert len(reopened["messages"]) == 4


def test_durable_restart_restores_masked_memory_and_plan_sequence(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("ANBAN_TOKENIZER_PATH", "unit-test-tokenizer.json")
    monkeypatch.setattr(
        "api.app._conversation_memory",
        lambda path: BoundedConversationMemory(
            lambda text: max(1, len(text) // 4)
        ),
    )
    db_path = str(tmp_path / "bank.db")
    clock = FixedClock(NOW)

    app = _new_app(db_path, clock)
    try:
        with TestClient(app) as first_client:
            token = _login(first_client)
            chat = first_client.post(
                "/chats", json={"title": "Balance"}, headers=_auth(token)
            ).json()["chat"]
            first = first_client.post(
                "/chat",
                json={"text": "我的余额", "session_id": chat["id"]},
                headers=_auth(token),
            )
            assert first.status_code == 200, first.text
            assert first.json()["plan_id"] == f"{chat['id']}:p1"

            rows = app.state.runtime.conn.execute(
                "SELECT role, masked_user, memory_summary FROM chat_messages"
                " WHERE chat_id = ? ORDER BY rowid",
                (chat["id"],),
            ).fetchall()
            assert [row["role"] for row in rows] == ["user", "assistant"]
            assert "余额" in rows[0]["masked_user"]
            assert "User:" not in rows[0]["masked_user"]
            assert "status=done" in rows[1]["memory_summary"]
            assert "get_balance" in rows[1]["memory_summary"]
    finally:
        app.state.runtime.conn.close()

    reopened = _new_app(db_path, clock)
    try:
        with TestClient(reopened) as second_client:
            page = second_client.get(
                f"/chats/{chat['id']}/messages", headers=_auth(token)
            )
            assert page.status_code == 200, page.text
            assert len(page.json()["messages"]) == 2

            second = second_client.post(
                "/chat",
                json={"text": "我的余额", "session_id": chat["id"]},
                headers=_auth(token),
            )
            assert second.status_code == 200, second.text
            assert second.json()["plan_id"] == f"{chat['id']}:p2"

            memory = reopened.state.runtime.services.conversation_memory
            assert memory is not None
            rendered = memory.render(token.split(".")[0], chat["id"], "system", "")
            assert "Historical conversation" in rendered
            assert "我的余额" in rendered
    finally:
        reopened.state.runtime.conn.close()


def test_otp_codes_are_stripped_from_history(client: TestClient) -> None:
    token = _login(client)
    started = client.post(
        "/chat", json={"text": "给妈妈转2000"}, headers=_auth(token)
    ).json()
    assert started["status"] == "needs_approval", started
    codes = [
        item.get("demo_otp")
        for item in started["approvals"]
        if item.get("demo_otp")
    ]
    assert codes, started

    chat_id = started["session_id"]
    response = client.get(f"/chats/{chat_id}/messages", headers=_auth(token))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["messages"]
    for message in body["messages"]:
        turn = message["turn"]
        if isinstance(turn, dict):
            assert turn["approvals"] == []

    rows = client.app.state.runtime.conn.execute(
        "SELECT COALESCE(turn_json, ''), text FROM chat_messages WHERE chat_id = ?",
        (chat_id,),
    ).fetchall()
    persisted = "\n".join(str(row[0]) + "\n" + str(row[1]) for row in rows)
    for code in codes:
        assert code not in response.text
        assert code not in persisted
    for key in ("demo_otp", "demo_co_otp", "challenge_code"):
        assert key not in persisted


def test_expired_approval_is_not_revived_after_restart(tmp_path) -> None:
    db_path = str(tmp_path / "bank.db")
    clock = FixedClock(NOW)

    app = _new_app(db_path, clock)
    try:
        with TestClient(app) as first_client:
            token = _login(first_client)
            chat = first_client.post(
                "/chats", json={"title": "Approval"}, headers=_auth(token)
            ).json()["chat"]
            started = first_client.post(
                "/chat",
                json={"text": "给妈妈转300", "session_id": chat["id"]},
                headers=_auth(token),
            ).json()
            assert started["status"] == "needs_approval", started
            step_id = next(
                item["step_id"]
                for item in started["approvals"]
                if item["tool"] == "transfer"
            )
            advanced = first_client.post(
                "/demo/clock", json={"advance_minutes": 10}
            )
            assert advanced.status_code == 200, advanced.text
    finally:
        app.state.runtime.conn.close()

    reopened = _new_app(db_path, clock)
    try:
        with TestClient(reopened) as second_client:
            page = second_client.get(
                f"/chats/{chat['id']}/messages", headers=_auth(token)
            ).json()
            turns = [
                message["turn"]
                for message in page["messages"]
                if isinstance(message.get("turn"), dict)
            ]
            pending = [
                turn
                for turn in turns
                if turn.get("plan_id") == f"{chat['id']}:p1"
            ]
            assert pending
            assert pending[-1]["status"] == "clarify"
            assert "reconfirm" in pending[-1]["reply"].lower()
            assert pending[-1]["approvals"] == []

            response = second_client.post(
                f"/chat/{chat['id']}/approve",
                json={"approvals": {step_id: {"confirm": True}}},
                headers=_auth(token),
            )
            assert response.status_code == 200, response.text
            assert response.json()["status"] != "done"

            customer = token.split(".")[0]
            count = reopened.state.runtime.conn.execute(
                "SELECT COUNT(*) FROM transfers WHERE customer_id = ?",
                (customer,),
            ).fetchone()[0]
            assert count == 0
    finally:
        reopened.state.runtime.conn.close()
