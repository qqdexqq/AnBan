"""Send-now tests: one early release, no second debit, atomic rollback.

The real seeded bank, mandate service, audit chain and FastAPI edge are used;
only the planner is scripted.  Most cases build the pending transfer and its
signed cancel_transfer receipt directly so the assertions stay about send-now.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from agent.executor import UNDO_SCHEMA, _undo_sig
from agent.planner import ScriptedPlanner
from api.app import create_app
from api.send_now import SendNow, SendNowError
from bank.api import Bank, BankError

NOW = datetime(2026, 10, 17, 14, 0, 0)
SECRET = b"k" * 32


class FixedClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now += timedelta(minutes=minutes)


def _refuse(user: str) -> str:
    return json.dumps({"kind": "refuse", "reason": "out_of_scope"})


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@pytest.fixture
def env():
    clock = FixedClock(NOW)
    app = create_app(
        secret_key=SECRET,
        planner=ScriptedPlanner(_refuse),
        clock=clock,
        demo_mode=True,
    )
    with TestClient(app) as client:
        yield client, app.state.runtime, clock
    app.state.runtime.conn.close()


def _customer(rt, persona: str) -> str:
    return rt.states[persona].customer.id


def _receipt(rt, customer: str, session: str, plan: str, step: str = "s1",
             contact_id: str | None = None, amount_fen: int = 10_000) -> str:
    bank = Bank(rt.conn, customer)
    contact = contact_id or bank.find_contact("")[0].id
    transfer = bank.transfer(
        to_contact_id=contact,
        amount_fen=amount_fen,
        note="",
        idempotency_key=f"{plan}:{step}",
        now=rt.clock(),
        cooling=timedelta(hours=2),
    )
    args = {"transfer_id": transfer.id, "amount_fen": amount_fen, "step_id": step}
    args_json = _canonical(args)
    created_at = rt.clock().isoformat()
    sig = _undo_sig(
        rt.services.secret_key, customer, plan, step, "cancel_transfer",
        args_json, created_at,
    )
    rt.conn.executescript(UNDO_SCHEMA)
    rt.conn.execute(
        "INSERT INTO exec_undo (customer_id, plan_id, step_id, tool, args_json,"
        " sig, used, created_at) VALUES (?, ?, ?, 'cancel_transfer', ?, ?, 0, ?)",
        (customer, plan, step, args_json, sig, created_at),
    )
    rt.conn.commit()
    rt.services.sessions[session] = {
        "customer_id": customer,
        "plan_id": plan,
        "plan_seq": 1,
        "pending_plan_id": "",
        "scam_flags": [],
        "frozen_until": None,
    }
    return transfer.id


def _login(client: TestClient, persona: str = "young_pro") -> str:
    response = client.post("/auth/demo-login", json={"persona": persona})
    assert response.status_code == 200, response.text
    return response.json()["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _rows(rt, sql: str, params: tuple = ()) -> list:
    return rt.conn.execute(sql, params).fetchall()


# --------------------------------------------------------------------- success


def test_success_releases_once_without_second_debit_and_valid_audit(env) -> None:
    _, rt, _ = env
    customer = _customer(rt, "young_pro")
    session, plan, step = "sess-ok", "sess-ok:p1", "s1"
    amount = 10_000
    transfer_id = _receipt(rt, customer, session, plan, step, amount_fen=amount)
    bank = Bank(rt.conn, customer)
    balance_after_transfer = bank.get_balance()
    txns_before = len(_rows(rt, "SELECT id FROM transactions WHERE customer_id = ?", (customer,)))

    send = SendNow(rt)
    prepared = send.prepare(customer, session, plan, step)
    codes = send.request_code(customer, prepared["challenge_id"])
    result = send.confirm(
        customer, prepared["challenge_id"], codes["otp"], amount, True
    )

    assert result["status"] == "done" and result["ai_generated"] is True
    assert result["session_id"] == session and result["plan_id"] == plan
    assert result["plan_view"] == [
        {"step_id": step, "tool": "transfer", "status": "released", "tier": "red", "after": []}
    ]
    assert result["approvals"] == [] and result["actions"] == ["send_now_released"]
    assert Bank(rt.conn, customer).get_balance() == balance_after_transfer
    status = rt.conn.execute(
        "SELECT status FROM transfers WHERE customer_id = ? AND id = ?",
        (customer, transfer_id),
    ).fetchone()["status"]
    assert status == "released"
    txns_after = len(_rows(rt, "SELECT id FROM transactions WHERE customer_id = ?", (customer,)))
    assert txns_after == txns_before + 1  # exactly one ledger entry
    assert rt.services.audit.verify().ok
    kinds = [e.kind for e in rt.services.audit.entries_for(customer, plan)]
    assert {"MFA", "POLICY", "RUN"} <= set(kinds)
    assert rt.conn.execute(
        "SELECT used FROM exec_undo WHERE customer_id = ? AND plan_id = ? AND step_id = ?",
        (customer, plan, step),
    ).fetchone()["used"] == 1

    # The selected transfer's undo is refused; nothing else changes.
    undo = rt.agent.undo(session, customer, plan, step)
    assert undo.status == "error"
    assert Bank(rt.conn, customer).get_balance() == balance_after_transfer


# ------------------------------------------------------------------- bad codes


def test_wrong_otp_persists_attempts_and_locks(env) -> None:
    _, rt, _ = env
    customer = _customer(rt, "young_pro")
    session, plan, step = "sess-otp", "sess-otp:p1", "s1"
    transfer_id = _receipt(rt, customer, session, plan, step)
    send = SendNow(rt)
    prepared = send.prepare(customer, session, plan, step)
    codes = send.request_code(customer, prepared["challenge_id"])
    wrong = "000000" if codes["otp"] != "000000" else "123456"
    mandate_id = _mandate_id(rt, prepared["challenge_id"])

    for _ in range(2):
        with pytest.raises(SendNowError) as raised:
            send.confirm(customer, prepared["challenge_id"], wrong, 10_000, True)
        assert raised.value.code == "mismatch"
    # Failures are committed before the third attempt can lock the account.
    assert rt.conn.execute(
        "SELECT attempts FROM security_otp WHERE mandate_id = ? AND purpose = 'customer_mfa'",
        (mandate_id,),
    ).fetchone()["attempts"] == 2

    with pytest.raises(SendNowError) as locked:
        send.confirm(customer, prepared["challenge_id"], wrong, 10_000, True)
    assert locked.value.code == "locked"
    assert rt.conn.execute(
        "SELECT COUNT(*) AS n FROM security_mfa_failures WHERE customer_id = ?", (customer,)
    ).fetchone()["n"] == 3
    assert rt.conn.execute(
        "SELECT 1 FROM security_lockouts WHERE customer_id = ?", (customer,)
    ).fetchone() is not None
    assert rt.conn.execute(
        "SELECT status FROM security_mandates WHERE id = ?", (mandate_id,)
    ).fetchone()["status"] == "revoked"
    assert rt.conn.execute(
        "SELECT status FROM transfers WHERE customer_id = ? AND id = ?",
        (customer, transfer_id),
    ).fetchone()["status"] == "pending"


def _mandate_id(rt, challenge_id: str) -> str:
    return rt.conn.execute(
        "SELECT mandate_id FROM send_now_challenges WHERE id = ?", (challenge_id,)
    ).fetchone()["mandate_id"]


# ---------------------------------------------------- replay / stale / tamper


def test_replay_stale_and_tampered_receipts_are_rejected(env) -> None:
    _, rt, clock = env
    customer = _customer(rt, "young_pro")
    send = SendNow(rt)

    # Replay: a consumed receipt cannot run twice.
    session, plan, step = "sess-replay", "sess-replay:p1", "s1"
    _receipt(rt, customer, session, plan, step)
    prepared = send.prepare(customer, session, plan, step)
    codes = send.request_code(customer, prepared["challenge_id"])
    send.confirm(customer, prepared["challenge_id"], codes["otp"], 10_000, True)
    with pytest.raises(SendNowError) as replay:
        send.confirm(customer, prepared["challenge_id"], codes["otp"], 10_000, True)
    assert replay.value.code == "replayed"

    # Stale: the five-minute challenge window has closed.
    session, plan, step = "sess-old", "sess-old:p1", "s1"
    _receipt(rt, customer, session, plan, step)
    prepared = send.prepare(customer, session, plan, step)
    codes = send.request_code(customer, prepared["challenge_id"])
    clock.advance(6)
    with pytest.raises(SendNowError) as stale:
        send.confirm(customer, prepared["challenge_id"], codes["otp"], 10_000, True)
    assert stale.value.code == "expired"

    # Tamper: changing the signed undo arguments breaks the HMAC.
    session, plan, step = "sess-tamper", "sess-tamper:p1", "s1"
    _receipt(rt, customer, session, plan, step)
    prepared = send.prepare(customer, session, plan, step)
    codes = send.request_code(customer, prepared["challenge_id"])
    row = rt.conn.execute(
        "SELECT args_json FROM exec_undo WHERE customer_id = ? AND plan_id = ?",
        (customer, plan),
    ).fetchone()
    args = json.loads(row["args_json"])
    args["amount_fen"] = 999_999
    rt.conn.execute(
        "UPDATE exec_undo SET args_json = ? WHERE customer_id = ? AND plan_id = ?",
        (_canonical(args), customer, plan),
    )
    rt.conn.commit()
    with pytest.raises(SendNowError) as tampered:
        send.confirm(customer, prepared["challenge_id"], codes["otp"], 10_000, True)
    assert tampered.value.code == "bad_signature"


def test_cross_chat_and_cross_customer_challenges_are_rejected(env) -> None:
    _, rt, _ = env
    customer = _customer(rt, "young_pro")
    other = _customer(rt, "student")
    session, plan, step = "sess-cross", "sess-cross:p1", "s1"
    _receipt(rt, customer, session, plan, step)
    send = SendNow(rt)
    prepared = send.prepare(customer, session, plan, step)

    with pytest.raises(SendNowError) as cross_chat:
        send.prepare(customer, "other-session", plan, step)
    assert cross_chat.value.code == "not_found"

    with pytest.raises(SendNowError) as cross_customer:
        send.request_code(other, prepared["challenge_id"])
    assert cross_customer.value.code == "not_found"


# ---------------------------------------------------------------- freeze / rollback


def test_freeze_after_request_code_denies_confirmation(env) -> None:
    _, rt, _ = env
    customer = _customer(rt, "young_pro")
    session, plan, step = "sess-freeze", "sess-freeze:p1", "s1"
    transfer_id = _receipt(rt, customer, session, plan, step)
    send = SendNow(rt)
    prepared = send.prepare(customer, session, plan, step)
    codes = send.request_code(customer, prepared["challenge_id"])
    balance = Bank(rt.conn, customer).get_balance()

    rt.services.mandates.freeze_for_scam(customer, ["scam"], rt.clock())
    with pytest.raises(SendNowError) as denied:
        send.confirm(customer, prepared["challenge_id"], codes["otp"], 10_000, True)
    assert denied.value.code == "scam"
    assert rt.conn.execute(
        "SELECT status FROM transfers WHERE customer_id = ? AND id = ?",
        (customer, transfer_id),
    ).fetchone()["status"] == "pending"
    assert Bank(rt.conn, customer).get_balance() == balance


def test_bank_failure_rolls_back_mandate_ledger_and_undo(env, monkeypatch) -> None:
    _, rt, _ = env
    customer = _customer(rt, "young_pro")
    session, plan, step = "sess-fail", "sess-fail:p1", "s1"
    transfer_id = _receipt(rt, customer, session, plan, step)
    send = SendNow(rt)
    prepared = send.prepare(customer, session, plan, step)
    codes = send.request_code(customer, prepared["challenge_id"])
    bank = Bank(rt.conn, customer)
    balance = bank.get_balance()
    txns = len(_rows(rt, "SELECT id FROM transactions WHERE customer_id = ?", (customer,)))
    mandate_id = _mandate_id(rt, prepared["challenge_id"])

    def boom(self, transfer_id, now, *, expected_contact_id, expected_amount_fen):
        raise BankError("invalid_state")

    monkeypatch.setattr(Bank, "release_early", boom)
    with pytest.raises(SendNowError):
        send.confirm(customer, prepared["challenge_id"], codes["otp"], 10_000, True)

    assert rt.conn.execute(
        "SELECT status FROM transfers WHERE customer_id = ? AND id = ?",
        (customer, transfer_id),
    ).fetchone()["status"] == "pending"
    assert Bank(rt.conn, customer).get_balance() == balance
    assert len(_rows(rt, "SELECT id FROM transactions WHERE customer_id = ?", (customer,))) == txns
    assert rt.conn.execute(
        "SELECT status FROM security_mandates WHERE id = ?", (mandate_id,)
    ).fetchone()["status"] == "approved"  # consumption rolled back
    assert rt.conn.execute(
        "SELECT used FROM exec_undo WHERE customer_id = ? AND plan_id = ? AND step_id = ?",
        (customer, plan, step),
    ).fetchone()["used"] == 0


# ----------------------------------------------------------------------- elder


def test_elder_requires_the_registered_guardian_code(env) -> None:
    _, rt, _ = env
    customer = _customer(rt, "retiree")
    session, plan, step = "sess-elder", "sess-elder:p1", "s1"
    _receipt(rt, customer, session, plan, step)
    send = SendNow(rt)
    prepared = send.prepare(customer, session, plan, step)
    assert prepared["needs_co_approval"] is True
    codes = send.request_code(customer, prepared["challenge_id"])
    assert codes["otp"] and codes["co_otp"]

    with pytest.raises(SendNowError) as missing:
        send.confirm(customer, prepared["challenge_id"], codes["otp"], 10_000, True)
    assert missing.value.code == "co_otp_missing"

    result = send.confirm(
        customer, prepared["challenge_id"], codes["otp"], 10_000, True,
        co_otp=codes["co_otp"],
    )
    assert result["status"] == "done"


# ------------------------------------------------------------------------- API

def test_simultaneous_confirm_requests_settle_exactly_once(env):
    from concurrent.futures import ThreadPoolExecutor
    client, rt, _ = env
    customer = _customer(rt, "young_pro")
    tid = _receipt(rt, customer, "concurrent", "concurrent:p1")
    send = SendNow(rt)
    challenge = send.prepare(customer, "concurrent", "concurrent:p1", "s1")["challenge_id"]
    code = send.request_code(customer, challenge)["otp"]
    headers = _auth(_login(client))
    body = {"challenge_id": challenge, "otp": code, "retyped_amount_fen": 10000, "face_ok": True}
    ledger_before = rt.conn.execute("SELECT COUNT(*) FROM transactions WHERE customer_id = ?", (customer,)).fetchone()[0]
    balance = Bank(rt.conn, customer).get_balance()
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(lambda _: client.post("/send-now/confirm", json=body, headers=headers), range(2)))
    assert sorted(r.status_code for r in replies) == [200, 403]
    assert rt.conn.execute("SELECT status FROM transfers WHERE id = ?", (tid,)).fetchone()[0] == "released"
    assert rt.conn.execute("SELECT COUNT(*) FROM transactions WHERE customer_id = ?", (customer,)).fetchone()[0] == ledger_before + 1
    assert Bank(rt.conn, customer).get_balance() == balance
    assert rt.services.audit.verify().ok

def test_self_enabled_elder_keeps_guardian_gate_inside_transaction(env):
    from api.send_now import _TxConnection
    from security import compliance
    from security.policy import evaluate_early_release
    _, rt, _ = env
    customer = _customer(rt, "young_pro")
    tid = _receipt(rt, customer, "elder-mode", "elder-mode:p1")
    compliance.ensure_schema(rt.conn)
    rt.conn.execute("INSERT OR REPLACE INTO customer_settings(customer_id,elder_mode) VALUES (?,1)", (customer,))
    rt.conn.commit()
    rt.conn.execute("BEGIN IMMEDIATE")
    verdict = evaluate_early_release(bank=Bank(_TxConnection(rt.conn), customer), transfer_id=tid, now=rt.clock())
    assert verdict.needs_co_approval and not verdict.vetoed
    assert rt.conn.in_transaction
    rt.conn.rollback()


def test_send_now_cannot_shorten_a_protected_longer_hold(env):
    _, rt, _ = env
    customer = _customer(rt, "young_pro")
    tid = _receipt(rt, customer, "protected", "protected:p1")
    rt.conn.execute("UPDATE transfers SET release_at = ? WHERE id = ?", ((rt.clock() + timedelta(hours=24)).isoformat(), tid))
    rt.conn.commit()
    with pytest.raises(SendNowError, match="policy_veto"):
        SendNow(rt).prepare(customer, "protected", "protected:p1", "s1")
    contact = rt.conn.execute("SELECT to_contact_id FROM transfers WHERE id = ?", (tid,)).fetchone()[0]
    with pytest.raises(BankError) as blocked:
        Bank(rt.conn, customer).release_early(tid, rt.clock(), expected_contact_id=contact, expected_amount_fen=10000)
    assert blocked.value.code == "protected_hold"
    assert rt.conn.execute("SELECT status FROM transfers WHERE id = ?", (tid,)).fetchone()[0] == "pending"


def test_resend_invalidates_old_code_and_expired_challenge_cannot_settle(env):
    _, rt, clock = env
    customer = _customer(rt, "young_pro")
    tid = _receipt(rt, customer, "expiry", "expiry:p1")
    send = SendNow(rt)
    challenge = send.prepare(customer, "expiry", "expiry:p1", "s1")["challenge_id"]
    codes = iter(["123456", "654321"])
    rt.services.mandates._otp_code = lambda: next(codes)
    old = send.request_code(customer, challenge)
    new = send.request_code(customer, challenge)
    assert old["otp"] != new["otp"]
    with pytest.raises(SendNowError):
        send.confirm(customer, challenge, old["otp"], 10000, True)
    clock.advance(6)
    with pytest.raises(SendNowError, match="expired"):
        send.confirm(customer, challenge, new["otp"], 10000, True)
    assert rt.conn.execute("SELECT status FROM transfers WHERE id = ?", (tid,)).fetchone()[0] == "pending"


def test_demo_route_returns_codes_but_production_strips_them(env) -> None:
    client, rt, _ = env
    customer = _customer(rt, "young_pro")
    session, plan, step = "sess-api", "sess-api:p1", "s1"
    _receipt(rt, customer, session, plan, step)
    token = _login(client)
    prepared = client.post(
        "/send-now/prepare",
        json={"session_id": session, "plan_id": plan, "step_id": step},
        headers=_auth(token),
    )
    assert prepared.status_code == 200, prepared.text
    codes = client.post(
        "/send-now/request-code",
        json={"challenge_id": prepared.json()["challenge_id"]},
        headers=_auth(token),
    )
    assert codes.status_code == 200 and "demo_otp" in codes.json()

    clock = FixedClock(NOW)
    app = create_app(
        secret_key=SECRET, planner=ScriptedPlanner(_refuse), clock=clock,
        demo_mode=False,
    )
    try:
        with TestClient(app) as prod:
            prod_rt = app.state.runtime
            prod_customer = _customer(prod_rt, "young_pro")
            prod_session, prod_plan, prod_step = "sess-prod", "sess-prod:p1", "s1"
            _receipt(prod_rt, prod_customer, prod_session, prod_plan, prod_step)
            expiry = int(clock().timestamp()) + 3600
            message = f"{prod_customer}.{expiry}".encode("utf-8")
            signature = hmac.new(SECRET, message, hashlib.sha256).hexdigest()
            prod_token = f"{prod_customer}.{expiry}.{signature}"
            body = prod.post(
                "/send-now/prepare",
                json={
                    "session_id": prod_session,
                    "plan_id": prod_plan,
                    "step_id": prod_step,
                },
                headers=_auth(prod_token),
            )
            assert body.status_code == 200, body.text
            hidden = prod.post(
                "/send-now/request-code",
                json={"challenge_id": body.json()["challenge_id"]},
                headers=_auth(prod_token),
            )
            assert hidden.status_code == 200
            assert "otp" not in hidden.json() and "co_otp" not in hidden.json()
            assert "demo_otp" not in hidden.json() and "demo_co_otp" not in hidden.json()
    finally:
        app.state.runtime.conn.close()
