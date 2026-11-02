"""HTTP-edge tests for the AnBan FastAPI app.

These drive the public API exactly as a client would: real middleware, real
tokens, the real seeded bank and the real LangGraph agent.  Only the planner
is scripted, so the assertions describe what *code* does with a proposal.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from agent.planner import ScriptedPlanner
from api.app import create_app

NOW = datetime(2026, 10, 17, 14, 0, 0)
SECRET = b"k" * 32
PII_RE = re.compile(r"\d{11,}")
MAX_BODY_BYTES = 16 * 1024


# ------------------------------------------------------------------- helpers


class FixedClock:
    """Deterministic clock with the hook ``POST /demo/clock`` expects."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now += timedelta(minutes=minutes)


def _user_line(user: str) -> str:
    return user.rsplit("User:", 1)[-1]


def _scripted(user: str) -> str:
    """Balance for balance questions, can_cover + transfer otherwise."""

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


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(NOW)


@pytest.fixture
def client(clock: FixedClock):
    app = create_app(
        secret_key=SECRET,
        planner=ScriptedPlanner(_scripted),
        clock=clock,
        demo_mode=True,
    )
    with TestClient(app) as test_client:
        yield test_client
    app.state.runtime.conn.close()


def _login(client: TestClient, persona: str = "young_pro") -> str:
    response = client.post("/auth/demo-login", json={"persona": persona})
    assert response.status_code == 200, response.text
    return response.json()["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _chat(client: TestClient, token: str, text: str) -> dict:
    response = client.post("/chat", json={"text": text}, headers=_auth(token))
    assert response.status_code == 200, response.text
    return response.json()


def _approve(
    client: TestClient, token: str, session_id: str, step_id: str, decision: dict
):
    return client.post(
        f"/chat/{session_id}/approve",
        json={"approvals": {step_id: decision}},
        headers=_auth(token),
    )


def _approve_transfer(client: TestClient, token: str, text: str = "给妈妈转300"):
    started = _chat(client, token, text)
    assert started["status"] == "needs_approval", started
    step_id = next(
        item["step_id"]
        for item in started["approvals"]
        if item["tool"] == "transfer"
    )
    response = _approve(client, token, started["session_id"], step_id, {"confirm": True})
    assert response.status_code == 200, response.text
    resumed = response.json()
    assert resumed["status"] == "done", resumed
    return started["session_id"], resumed["plan_id"], step_id


def _undo(client: TestClient, token: str, session_id: str, plan_id: str, step_id: str):
    return client.post(
        "/undo",
        json={"session_id": session_id, "plan_id": plan_id, "step_id": step_id},
        headers=_auth(token),
    )


# --------------------------------------------------------------------- tests


def test_health_is_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_demo_login_and_summary_expose_no_long_digits(client: TestClient) -> None:
    token = _login(client, "young_pro")

    response = client.get("/me/summary", headers=_auth(token))

    assert response.status_code == 200, response.text
    data = response.json()
    assert isinstance(data["balance_fen"], int)
    assert data["display"].startswith("¥")
    assert PII_RE.search(response.text) is None, response.text


def test_missing_tampered_and_expired_tokens_are_401(client: TestClient) -> None:
    assert client.get("/me/summary").status_code == 401

    token = _login(client)
    replacement = "0" if token[-1] != "0" else "1"
    tampered = token[:-1] + replacement
    assert client.get("/me/summary", headers=_auth(tampered)).status_code == 401

    advanced = client.post("/demo/clock", json={"advance_minutes": 8 * 60 + 1})
    assert advanced.status_code == 200, advanced.text
    assert client.get("/me/summary", headers=_auth(token)).status_code == 401


def test_balance_chat_returns_done(client: TestClient) -> None:
    token = _login(client)

    result = _chat(client, token, "我的余额")

    assert result["status"] == "done"
    assert result["session_id"]


def test_transfer_needs_approval_then_completes(client: TestClient) -> None:
    token = _login(client)
    started = _chat(client, token, "给妈妈转300")

    assert started["status"] == "needs_approval"
    assert started["approvals"]
    approval = next(
        item for item in started["approvals"] if item["tool"] == "transfer"
    )
    assert approval["amount_fen"] == 30_000

    response = _approve(
        client, token, started["session_id"], approval["step_id"], {"confirm": True}
    )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "done"


def test_another_customer_cannot_resume_an_approved_session(client: TestClient) -> None:
    token = _login(client)
    session_id, _plan_id, step_id = _approve_transfer(client, token)
    before = client.get("/me/summary", headers=_auth(token)).json()["balance_fen"]

    student = _login(client, "student")
    response = _approve(client, student, session_id, step_id, {"confirm": True})

    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "not_found"
    after = client.get("/me/summary", headers=_auth(token)).json()["balance_fen"]
    assert after == before


def test_non_confirming_answer_keeps_the_transfer_pending(client: TestClient) -> None:
    token = _login(client)
    started = _chat(client, token, "给妈妈转300")
    step_id = next(
        item["step_id"]
        for item in started["approvals"]
        if item["tool"] == "transfer"
    )

    response = _approve(
        client, token, started["session_id"], step_id, {"confirm": False}
    )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "needs_approval"


def test_body_with_extra_customer_id_is_rejected(client: TestClient) -> None:
    token = _login(client)

    response = client.post(
        "/chat",
        json={"text": "我的余额", "customer_id": "cust_young_pro"},
        headers=_auth(token),
    )

    assert response.status_code == 422, response.text


def test_undo_window_closes_but_fresh_transfer_can_be_undone(
    client: TestClient,
) -> None:
    token = _login(client)
    session_id, plan_id, step_id = _approve_transfer(client, token, "给妈妈转300")

    advanced = client.post("/demo/clock", json={"advance_minutes": 180})
    assert advanced.status_code == 200, advanced.text

    before = client.get("/me/summary", headers=_auth(token)).json()["balance_fen"]
    stale = _undo(client, token, session_id, plan_id, step_id)
    assert stale.status_code == 200, stale.text
    stale_body = stale.json()
    assert stale_body["status"] == "error", stale_body
    after = client.get("/me/summary", headers=_auth(token)).json()["balance_fen"]
    assert after == before

    fresh_session, fresh_plan, fresh_step = _approve_transfer(
        client, token, "给妈妈转200"
    )
    fresh = _undo(client, token, fresh_session, fresh_plan, fresh_step)
    assert fresh.status_code == 200, fresh.text
    assert fresh.json()["status"] == "done"


def test_recall_after_the_window_opens_a_simulated_ticket(client: TestClient) -> None:
    token = _login(client)
    session_id, plan_id, step_id = _approve_transfer(client, token, "给妈妈转300")
    early = client.post(
        "/recall",
        json={"session_id": session_id, "plan_id": plan_id, "step_id": step_id},
        headers=_auth(token),
    )
    assert early.json()["status"] == "error"

    assert client.post("/demo/clock", json={"advance_minutes": 180}).status_code == 200
    closed = _undo(client, token, session_id, plan_id, step_id).json()
    assert closed["actions"] == ["recall"]

    response = client.post(
        "/recall",
        json={"session_id": session_id, "plan_id": plan_id, "step_id": step_id, "lang": "en"},
        headers=_auth(token),
    )
    body = response.json()
    assert body["status"] == "done", body
    assert "simulated" in body["reply"] and "not guaranteed" in body["reply"]


def test_held_transfer_shows_first_in_recent_activity(client: TestClient) -> None:
    token = _login(client)
    _approve_transfer(client, token, "给妈妈转300")

    rows = client.get("/me/transactions?limit=3", headers=_auth(token)).json()["transactions"]

    assert rows[0]["category"] == "on_hold" and rows[0]["amount_fen"] == -30_000
    assert rows[0]["held_until"]
    assert client.post("/demo/clock", json={"advance_minutes": 180}).status_code == 200
    later = client.get("/me/transactions?limit=3", headers=_auth(token)).json()["transactions"]
    assert all(row["category"] != "on_hold" for row in later)
    # Regression: once released it is in the ledger, not gone.
    assert any(row["amount_fen"] == -30_000 for row in later), later


def test_oversized_body_is_413(client: TestClient) -> None:
    token = _login(client)
    response = client.post(
        "/chat",
        content=json.dumps({"text": "x" * (MAX_BODY_BYTES + 100)}),
        headers={**_auth(token), "Content-Type": "application/json"},
    )

    assert response.status_code == 413, response.text


def test_thirty_one_requests_from_one_token_hit_the_rate_limit(
    client: TestClient,
) -> None:
    token = _login(client)

    statuses = [
        client.get("/health", headers=_auth(token)).status_code
        for _ in range(31)
    ]

    assert statuses[0] == 200
    assert 429 in statuses, statuses


def test_dashboard_refreshes_do_not_consume_action_quota(client: TestClient) -> None:
    token = _login(client)
    for _ in range(45):
        assert client.get("/me/settings", headers=_auth(token)).status_code == 200
    assert client.post("/chat", json={"text": "余额多少"}, headers=_auth(token)).status_code == 200


def test_action_limit_does_not_block_undo_or_reads(client: TestClient) -> None:
    token = _login(client)
    for _ in range(30):
        assert client.get("/health", headers=_auth(token)).status_code == 200
    assert client.get("/health", headers=_auth(token)).status_code == 429
    assert client.get("/me/settings", headers=_auth(token)).status_code == 200
    undo = client.post("/undo", headers=_auth(token), json={
        "session_id": "missing", "plan_id": "missing", "step_id": "s1",
    })
    assert undo.status_code == 200
    assert undo.json()["status"] == "error"  # Quota separation grants no undo authority.


def test_rotating_invalid_bearers_does_not_reset_rate_limit(client: TestClient) -> None:
    statuses = [client.get("/health", headers=_auth(f"fake-{i}")).status_code for i in range(31)]
    assert statuses[-1] == 429


def test_dashboard_reads_still_have_a_bound(client: TestClient) -> None:
    from api.app import READ_RATE_LIMIT
    token = _login(client)
    statuses = [client.get("/me/settings", headers=_auth(token)).status_code
                for _ in range(READ_RATE_LIMIT + 1)]
    assert statuses[0] == 200 and statuses[-1] == 429


def test_reissued_valid_token_keeps_the_same_customer_quota(client: TestClient) -> None:
    import hashlib
    import hmac
    token = _login(client)
    for _ in range(30):
        assert client.get("/health", headers=_auth(token)).status_code == 200
    customer, expiry, _ = token.split(".")
    message = f"{customer}.{int(expiry) + 1}".encode()
    new_token = message.decode() + "." + hmac.new(SECRET, message, hashlib.sha256).hexdigest()
    assert client.get("/health", headers=_auth(new_token)).status_code == 429


def test_non_ascii_token_signature_is_rejected_without_server_error(client: TestClient) -> None:
    token = _login(client)
    customer, expiry, _ = token.split(".")
    malformed = f"Bearer {customer}.{expiry}." + "é" * 64
    headers = {"authorization": malformed.encode("latin-1")}
    assert client.get("/health", headers=headers).status_code == 200
    assert client.get("/me/settings", headers=headers).status_code == 401


def test_demo_routes_are_absent_when_demo_mode_is_off() -> None:
    app = create_app(
        secret_key=SECRET,
        planner=ScriptedPlanner(_scripted),
        clock=FixedClock(NOW),
        demo_mode=False,
    )
    try:
        with TestClient(app) as client:
            login = client.post(
                "/auth/demo-login", json={"persona": "young_pro"}
            )
            clock = client.post("/demo/clock", json={"advance_minutes": 1})
            assert login.status_code == 404, login.text
            assert clock.status_code == 404, clock.text
            assert client.get("/health").status_code == 200
    finally:
        app.state.runtime.conn.close()


def test_demo_mode_is_off_unless_anban_demo_is_set(monkeypatch) -> None:
    # Regression: the demo routes used to be on by default.
    monkeypatch.delenv("ANBAN_DEMO", raising=False)
    app = create_app(
        secret_key=SECRET, planner=ScriptedPlanner(_scripted), clock=FixedClock(NOW)
    )
    try:
        with TestClient(app) as client:
            response = client.post("/demo/clock", json={"advance_minutes": 1})
            assert response.status_code == 404, response.text
    finally:
        app.state.runtime.conn.close()


@pytest.mark.parametrize("minutes", [-10, -(10**6), 10**15])
def test_demo_clock_only_moves_forward_within_bounds(
    client: TestClient, minutes: int
) -> None:
    # Regression: rewinding revived expired tokens; huge values were a 500.
    response = client.post("/demo/clock", json={"advance_minutes": minutes})
    assert response.status_code == 422, response.text


def test_unknown_persona_is_404(client: TestClient) -> None:
    response = client.post("/auth/demo-login", json={"persona": "does_not_exist"})

    assert response.status_code == 404, response.text


def test_chat_replies_are_labelled_ai_generated(client: TestClient) -> None:
    # 《人工智能生成合成内容标识办法》: the client shows an explicit AI label.
    token = _login(client)
    assert _chat(client, token, "余额多少")["ai_generated"] is True


def test_settings_round_trip_and_validation(client: TestClient) -> None:
    token = _login(client)
    assert client.get("/me/settings", headers=_auth(token)).json()["arrival_delay"] == "2h"

    saved = client.put(
        "/me/settings", json={"lang": "zh", "elder_mode": True}, headers=_auth(token)
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["lang"] == "zh" and saved.json()["elder_mode"] is True

    bad = client.put("/me/settings", json={"arrival_delay": "instant"}, headers=_auth(token))
    assert bad.status_code == 422


def test_chat_without_lang_uses_the_saved_language(client: TestClient) -> None:
    token = _login(client)
    # Nothing saved: the language is detected from the message.
    assert re.search(r"[一-鿿]", _chat(client, token, "余额多少")["reply"])
    # Regression: saving another setting is not choosing a language.
    client.put("/me/settings", json={"elder_mode": True}, headers=_auth(token))
    assert re.search(r"[一-鿿]", _chat(client, token, "余额多少")["reply"])
    client.put("/me/settings", json={"lang": "en"}, headers=_auth(token))
    reply = _chat(client, token, "余额多少")["reply"]
    assert re.search(r"[A-Za-z]{3,}", reply), reply
    assert not re.search(r"[一-鿿]", reply), reply


def test_recent_transactions_and_compliance_status(client: TestClient) -> None:
    token = _login(client)
    rows = client.get("/me/transactions?limit=3", headers=_auth(token)).json()["transactions"]
    assert len(rows) == 3 and "counterparty_en" in rows[0]

    status = client.get("/compliance/status", headers=_auth(token)).json()
    assert status["large_transfer_threshold_fen"] == 50_000_000
    assert status["reports"] == []

    # Another customer's settings and reports are never visible.
    other = _login(client, "retiree")
    client.put("/me/settings", json={"lang": "zh"}, headers=_auth(other))
    assert client.get("/me/settings", headers=_auth(token)).json()["lang"] == "en"


def test_regulator_records_are_not_a_customer_route(monkeypatch) -> None:
    # Regression: no tipping off; the records view exists only in demo mode.
    monkeypatch.delenv("ANBAN_DEMO", raising=False)
    app = create_app(secret_key=SECRET, planner=ScriptedPlanner(_scripted), clock=FixedClock(NOW))
    try:
        with TestClient(app) as client:
            assert client.get("/compliance/status").status_code == 404
    finally:
        app.state.runtime.conn.close()


def test_one_time_codes_are_not_returned_outside_demo_mode(monkeypatch) -> None:
    # Regression: demo_otp / demo_co_otp came back even with demo mode off.
    import hashlib
    import hmac as hmac_mod

    monkeypatch.delenv("ANBAN_DEMO", raising=False)
    clock = FixedClock(NOW)
    app = create_app(secret_key=SECRET, planner=ScriptedPlanner(_scripted), clock=clock)
    try:
        with TestClient(app) as client:
            customer = app.state.runtime.states["young_pro"].customer.id
            expiry = int(clock().timestamp()) + 3600
            message = f"{customer}.{expiry}".encode("utf-8")
            signature = hmac_mod.new(SECRET, message, hashlib.sha256).hexdigest()
            token = f"{customer}.{expiry}.{signature}"
            body = client.post("/chat", json={"text": "给妈妈转300"}, headers=_auth(token)).json()
            assert body["status"] == "needs_approval"
            assert all("demo_otp" not in a and "demo_co_otp" not in a for a in body["approvals"])
    finally:
        app.state.runtime.conn.close()
