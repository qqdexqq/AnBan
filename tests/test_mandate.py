"""Signed mandate, simulated MFA and lockout tests."""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta

import pytest

from bank.db import connect, create_schema, load_state
from bank.seed import generate
from security.mandate import MandateError, MandateService, create_security_schema

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42
SECRET = b"anban-mandate-secret"


@pytest.fixture
def env():
    conn = connect()
    create_schema(conn)
    young = generate("young_pro", SEED, TODAY)
    student = generate("student", SEED, TODAY)
    load_state(conn, young)
    load_state(conn, student)
    create_security_schema(conn)
    service = MandateService(conn, SECRET, otp_rng=random.Random(7))
    try:
        yield service, conn, young, student
    finally:
        conn.close()


@pytest.fixture
def elder_env():
    conn = connect()
    create_schema(conn)
    retiree = generate("retiree", SEED, TODAY)
    load_state(conn, retiree)
    create_security_schema(conn)
    service = MandateService(conn, SECRET, otp_rng=random.Random(7))
    try:
        yield service, conn, retiree
    finally:
        conn.close()


def _co_approver_id(retiree) -> str:
    for contact in retiree.contacts:
        if "co-approver" in contact.note.lower():
            return contact.id
    raise AssertionError("seed is missing a co-approver contact")


def _issue(service, customer_id, **overrides):
    fields = {
        "customer_id": customer_id,
        "session_id": "sess_1",
        "plan_id": "plan_1",
        "step_id": "s1",
        "action": "transfer",
        "payee_ref": "<PAYEE_1>",
        "amount_fen": 12_345,
        "tier": "yellow",
        "now": NOW,
    }
    fields.update(overrides)
    return service.issue(**fields)


def _consume(service, mandate, **overrides):
    fields = {
        "customer_id": mandate.customer_id,
        "session_id": mandate.session_id,
        "plan_id": mandate.plan_id,
        "step_id": mandate.step_id,
        "action": mandate.action,
        "payee_ref": mandate.payee_ref,
        "amount_fen": mandate.amount_fen,
        "now": NOW,
    }
    fields.update(overrides)
    return service.verify_and_consume(mandate.id, **fields)


def _status(conn, mandate_id):
    return conn.execute(
        "SELECT status FROM security_mandates WHERE id = ?", (mandate_id,)
    ).fetchone()["status"]


# --------------------------------------------------------------------- yellow


def test_yellow_approve_and_consume_is_single_use(env) -> None:
    service, conn, young, _ = env
    mandate = _issue(service, young.customer.id)

    assert mandate.id.startswith("m_")
    assert mandate.status == "issued"

    approved = service.approve_yellow(mandate.id, young.customer.id, NOW)
    assert approved.status == "approved"

    consumed = _consume(service, mandate)
    assert consumed.status == "consumed"
    assert _status(conn, mandate.id) == "consumed"

    with pytest.raises(MandateError) as exc:
        _consume(service, mandate)
    assert exc.value.code == "replayed"


def test_consume_requires_approval_and_rejects_expiry(env) -> None:
    service, _, young, _ = env
    pending = _issue(service, young.customer.id)

    with pytest.raises(MandateError) as exc:
        _consume(service, pending)
    assert exc.value.code == "not_approved"

    approved = _issue(service, young.customer.id)
    service.approve_yellow(approved.id, young.customer.id, NOW)
    with pytest.raises(MandateError) as exc:
        _consume(service, approved, now=NOW + timedelta(minutes=5, seconds=1))
    assert exc.value.code == "expired"


def test_any_binding_field_mismatch_is_rejected(env) -> None:
    service, _, young, _ = env
    mandate = _issue(service, young.customer.id)
    service.approve_yellow(mandate.id, young.customer.id, NOW)

    changes = [
        {"amount_fen": mandate.amount_fen + 1},
        {"payee_ref": "<PAYEE_2>"},
        {"step_id": "s2"},
        {"plan_id": "plan_2"},
        {"session_id": "sess_2"},
        {"action": "merchant_order"},
    ]
    for change in changes:
        with pytest.raises(MandateError) as exc:
            _consume(service, mandate, **change)
        assert exc.value.code == "mismatch", change

    # A failed comparison does not burn the mandate.
    assert _status(service.conn, mandate.id) == "approved"
    assert _consume(service, mandate).status == "consumed"


def test_tampered_amount_breaks_signature(env) -> None:
    service, conn, young, _ = env
    mandate = _issue(service, young.customer.id)
    service.approve_yellow(mandate.id, young.customer.id, NOW)
    conn.execute(
        "UPDATE security_mandates SET amount_fen = amount_fen + 1 WHERE id = ?",
        (mandate.id,),
    )
    conn.commit()

    with pytest.raises(MandateError) as exc:
        _consume(service, mandate)
    assert exc.value.code == "bad_signature"


# ------------------------------------------------------------------- red MFA


def test_red_requires_mfa_then_approves(env) -> None:
    service, conn, young, _ = env
    mandate = _issue(service, young.customer.id, tier="red")
    assert mandate.status == "mfa_pending"

    with pytest.raises(MandateError) as exc:
        _consume(service, mandate)
    assert exc.value.code == "mfa_required"

    code = service.start_mfa(mandate.id, young.customer.id, NOW)
    assert len(code) == 6 and code.isdigit()

    approved = service.verify_mfa(
        mandate.id,
        young.customer.id,
        otp=code,
        retyped_amount_fen=mandate.amount_fen,
        face_ok=True,
        now=NOW,
    )
    assert approved.status == "approved"
    assert _status(conn, mandate.id) == "approved"
    assert _consume(service, mandate).status == "consumed"

    assert (
        conn.execute(
            "SELECT COUNT(*) FROM security_otp WHERE mandate_id = ?", (mandate.id,)
        ).fetchone()[0]
        == 0
    )


def test_wrong_retyped_amount_counts_as_failure(env) -> None:
    service, conn, young, _ = env
    mandate = _issue(service, young.customer.id, tier="red")
    code = service.start_mfa(mandate.id, young.customer.id, NOW)

    with pytest.raises(MandateError) as exc:
        service.verify_mfa(
            mandate.id,
            young.customer.id,
            otp=code,
            retyped_amount_fen=mandate.amount_fen + 1,
            face_ok=True,
            now=NOW,
        )
    assert exc.value.code == "mismatch"
    assert _status(conn, mandate.id) == "mfa_pending"
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM security_mfa_failures WHERE customer_id = ?",
            (young.customer.id,),
        ).fetchone()[0]
        == 1
    )


def test_face_failure_counts_as_failure(env) -> None:
    service, conn, young, _ = env
    mandate = _issue(service, young.customer.id, tier="red")
    code = service.start_mfa(mandate.id, young.customer.id, NOW)

    with pytest.raises(MandateError):
        service.verify_mfa(
            mandate.id,
            young.customer.id,
            otp=code,
            retyped_amount_fen=mandate.amount_fen,
            face_ok=False,
            now=NOW,
        )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM security_mfa_failures WHERE customer_id = ?",
            (young.customer.id,),
        ).fetchone()[0]
        == 1
    )


def _fail_three_times(service, mandate, now=NOW) -> None:
    for _ in range(3):
        with pytest.raises(MandateError):
            service.verify_mfa(
                mandate.id,
                mandate.customer_id,
                otp="000000",
                retyped_amount_fen=mandate.amount_fen,
                face_ok=True,
                now=now,
            )


def test_three_failures_lock_for_24h_and_revoke_open_mandates(env) -> None:
    service, conn, young, _ = env
    customer_id = young.customer.id

    red = _issue(service, customer_id, tier="red")
    service.start_mfa(red.id, customer_id, NOW)
    yellow = _issue(service, customer_id)
    service.approve_yellow(yellow.id, customer_id, NOW)

    for attempt in range(3):
        with pytest.raises(MandateError) as exc:
            service.verify_mfa(
                red.id,
                customer_id,
                otp="000000",
                retyped_amount_fen=red.amount_fen,
                face_ok=True,
                now=NOW,
            )
        if attempt == 2:
            assert exc.value.code == "locked"

    assert service.is_payments_locked(customer_id, NOW)
    assert service.is_payments_locked(customer_id, NOW + timedelta(hours=23))
    assert not service.is_payments_locked(customer_id, NOW + timedelta(hours=24))
    row = conn.execute(
        "SELECT locked_until, reason FROM security_lockouts WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()
    assert row["reason"] == "mfa_failures"
    assert datetime.fromisoformat(row["locked_until"]) == NOW + timedelta(hours=24)

    assert _status(conn, red.id) == "revoked"
    assert _status(conn, yellow.id) == "revoked"

    with pytest.raises(MandateError) as exc:
        _issue(service, customer_id)
    assert exc.value.code == "locked"

    later = NOW + timedelta(hours=24, seconds=1)
    assert not service.is_payments_locked(customer_id, later)
    assert _issue(service, customer_id, now=later).status == "issued"


def test_failures_outside_rolling_window_do_not_count(env) -> None:
    service, conn, young, _ = env
    customer_id = young.customer.id
    old = (NOW - timedelta(hours=25)).isoformat()
    conn.executemany(
        "INSERT INTO security_mfa_failures (customer_id, at) VALUES (?, ?)",
        [(customer_id, old), (customer_id, old)],
    )
    conn.commit()

    mandate = _issue(service, customer_id, tier="red")
    service.start_mfa(mandate.id, customer_id, NOW)

    with pytest.raises(MandateError) as exc:
        service.verify_mfa(
            mandate.id,
            customer_id,
            otp="000000",
            retyped_amount_fen=mandate.amount_fen,
            face_ok=True,
            now=NOW,
        )
    assert exc.value.code == "mismatch"
    assert not service.is_payments_locked(customer_id, NOW)


def test_staff_unlock_clears_lockout(env) -> None:
    service, conn, young, _ = env
    customer_id = young.customer.id
    mandate = _issue(service, customer_id, tier="red")
    service.start_mfa(mandate.id, customer_id, NOW)
    _fail_three_times(service, mandate)
    assert service.is_payments_locked(customer_id, NOW)

    service.unlock(customer_id, by_staff="ops_alice", now=NOW)

    assert not service.is_payments_locked(customer_id, NOW)
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM security_lockouts WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()[0]
        == 0
    )
    assert _issue(service, customer_id).status == "issued"

    with pytest.raises(ValueError):
        service.unlock(customer_id, by_staff="", now=NOW)


# --------------------------------------------------------------- co-approval


def _verify_mfa(service, mandate, customer_id):
    code = service.start_mfa(mandate.id, customer_id, NOW)
    return service.verify_mfa(
        mandate.id,
        customer_id,
        otp=code,
        retyped_amount_fen=mandate.amount_fen,
        face_ok=True,
        now=NOW,
    )


def test_elder_co_approval_gates_consumption(elder_env) -> None:
    service, conn, retiree = elder_env
    customer_id = retiree.customer.id
    approver = _co_approver_id(retiree)
    mandate = _issue(
        service, customer_id, tier="red", co_approval_required=True
    )

    verified = _verify_mfa(service, mandate, customer_id)
    assert verified.mfa_verified is True
    assert verified.co_approved_by is None

    with pytest.raises(MandateError) as exc:
        _consume(service, mandate)
    assert exc.value.code == "co_approval_required"

    otp = service.start_co_approval(mandate.id, customer_id, approver, NOW)
    approved = service.co_approve(mandate.id, customer_id, approver, otp, NOW)
    assert approved.status == "approved"
    assert approved.co_approved_by == approver

    assert _consume(service, mandate).status == "consumed"


def test_co_approval_before_mfa_is_rejected(elder_env) -> None:
    service, _, retiree = elder_env
    customer_id = retiree.customer.id
    approver = _co_approver_id(retiree)
    mandate = _issue(
        service, customer_id, tier="red", co_approval_required=True
    )

    with pytest.raises(MandateError) as exc:
        service.co_approve(mandate.id, customer_id, approver, "000000", NOW)
    assert exc.value.code == "mfa_required"

    _verify_mfa(service, mandate, customer_id)
    otp = service.start_co_approval(mandate.id, customer_id, approver, NOW)
    assert (
        service.co_approve(mandate.id, customer_id, approver, otp, NOW).status
        == "approved"
    )
    assert _consume(service, mandate).status == "consumed"


def test_only_registered_co_approvers_are_accepted(elder_env) -> None:
    service, conn, retiree = elder_env
    customer_id = retiree.customer.id
    approver = _co_approver_id(retiree)
    assert service.co_approvers(customer_id) == [approver]

    mandate = _issue(
        service, customer_id, tier="red", co_approval_required=True
    )
    _verify_mfa(service, mandate, customer_id)

    for bad_approver in ("anyone", customer_id, retiree.contacts[1].id):
        with pytest.raises(MandateError) as exc:
            service.co_approve(mandate.id, customer_id, bad_approver, "000000", NOW)
        assert exc.value.code == "not_co_approver"

    assert (
        conn.execute(
            "SELECT COUNT(*) FROM security_mfa_failures WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()[0]
        == 0
    )


def test_wrong_co_approval_otp_three_times_locks_customer(elder_env) -> None:
    service, conn, retiree = elder_env
    customer_id = retiree.customer.id
    approver = _co_approver_id(retiree)
    mandate = _issue(
        service, customer_id, tier="red", co_approval_required=True
    )
    _verify_mfa(service, mandate, customer_id)
    otp = service.start_co_approval(mandate.id, customer_id, approver, NOW)
    wrong = "000000" if otp != "000000" else "999999"

    for attempt in range(3):
        with pytest.raises(MandateError) as exc:
            service.co_approve(mandate.id, customer_id, approver, wrong, NOW)
        if attempt == 2:
            assert exc.value.code == "locked"

    assert service.is_payments_locked(customer_id, NOW)
    assert _status(conn, mandate.id) == "revoked"
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM security_mfa_failures WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()[0]
        == 3
    )


def test_co_approval_otp_is_single_use(elder_env) -> None:
    service, conn, retiree = elder_env
    customer_id = retiree.customer.id
    approver = _co_approver_id(retiree)
    mandate = _issue(
        service, customer_id, tier="red", co_approval_required=True
    )
    _verify_mfa(service, mandate, customer_id)
    otp = service.start_co_approval(mandate.id, customer_id, approver, NOW)
    service.co_approve(mandate.id, customer_id, approver, otp, NOW)

    with pytest.raises(MandateError):
        service.co_approve(mandate.id, customer_id, approver, otp, NOW)

    assert _status(conn, mandate.id) == "approved"
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM security_otp WHERE mandate_id = ? AND"
            " purpose = 'co_approval'",
            (mandate.id,),
        ).fetchone()[0]
        == 0
    )


def test_start_co_approval_requires_flag_and_contact(elder_env) -> None:
    service, _, retiree = elder_env
    customer_id = retiree.customer.id
    approver = _co_approver_id(retiree)

    no_flag = _issue(service, customer_id, tier="red")
    _verify_mfa(service, no_flag, customer_id)
    with pytest.raises(MandateError) as exc:
        service.start_co_approval(no_flag.id, customer_id, approver, NOW)
    assert exc.value.code == "not_approved"

    flagged = _issue(service, customer_id, tier="red", co_approval_required=True)
    with pytest.raises(MandateError) as exc:
        service.start_co_approval(flagged.id, customer_id, "anyone", NOW)
    assert exc.value.code == "not_co_approver"


# --------------------------------------------------------------------- scope


def test_cross_customer_mandate_is_not_found(env) -> None:
    service, _, young, student = env
    mandate = _issue(service, young.customer.id)
    service.approve_yellow(mandate.id, young.customer.id, NOW)

    with pytest.raises(MandateError) as exc:
        service.approve_yellow(mandate.id, student.customer.id, NOW)
    assert exc.value.code == "not_found"

    with pytest.raises(MandateError) as exc:
        _consume(service, mandate, customer_id=student.customer.id)
    assert exc.value.code == "not_found"

    with pytest.raises(MandateError) as exc:
        service.start_mfa(mandate.id, student.customer.id, NOW)
    assert exc.value.code == "not_found"

    with pytest.raises(MandateError) as exc:
        service.co_approve(
            mandate.id, student.customer.id, student.contacts[0].id, "000000", NOW
        )
    assert exc.value.code == "not_found"


def test_staff_unlock_resets_failure_strikes(env) -> None:
    service, conn, young, _ = env
    customer_id = young.customer.id
    mandate = _issue(service, customer_id, tier="red")
    service.start_mfa(mandate.id, customer_id, NOW)
    _fail_three_times(service, mandate)
    assert service.is_payments_locked(customer_id, NOW)

    service.unlock(customer_id, by_staff="ops_alice", now=NOW)

    assert (
        conn.execute(
            "SELECT COUNT(*) FROM security_mfa_failures WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()[0]
        == 0
    )
    audit = conn.execute(
        "SELECT by_staff, at FROM security_unlocks WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()
    assert audit["by_staff"] == "ops_alice"
    assert datetime.fromisoformat(audit["at"]) == NOW

    # A single new failure must not immediately re-lock the customer.
    fresh = _issue(service, customer_id, tier="red", now=NOW)
    code = service.start_mfa(fresh.id, customer_id, NOW)
    wrong_otp = "000000" if code != "000000" else "999999"
    with pytest.raises(MandateError) as exc:
        service.verify_mfa(
            fresh.id,
            customer_id,
            otp=wrong_otp,
            retyped_amount_fen=fresh.amount_fen,
            face_ok=True,
            now=NOW + timedelta(minutes=1),
        )
    assert exc.value.code == "mismatch"
    assert not service.is_payments_locked(customer_id, NOW + timedelta(minutes=1))
    assert _status(conn, fresh.id) == "mfa_pending"


def test_otp_is_stored_only_as_a_salted_hash(env) -> None:
    service, conn, young, _ = env
    mandate = _issue(service, young.customer.id, tier="red")
    code = service.start_mfa(mandate.id, young.customer.id, NOW)

    row = conn.execute(
        "SELECT * FROM security_otp WHERE mandate_id = ?", (mandate.id,)
    ).fetchone()
    assert row is not None
    assert row["code_hash"] != code
    assert code not in row["code_hash"]
    assert code not in row["salt"]
    assert row["salt"]


def test_revoke_plan_revokes_only_its_open_mandates(env) -> None:
    service, conn, young, _ = env
    customer_id = young.customer.id
    first = _issue(service, customer_id, plan_id="plan_1")
    second = _issue(service, customer_id, plan_id="plan_1", step_id="s2")
    other = _issue(service, customer_id, plan_id="plan_2")
    service.approve_yellow(first.id, customer_id, NOW)

    revoked = service.revoke_plan(customer_id, "plan_1", NOW)
    assert revoked == 2
    assert _status(conn, first.id) == "revoked"
    assert _status(conn, second.id) == "revoked"
    assert _status(conn, other.id) == "issued"

    with pytest.raises(MandateError) as exc:
        _consume(service, first)
    assert exc.value.code == "revoked"


# ---------------------------------------------------------------- scam freeze


def test_scam_freeze_persists_unions_and_expires(env) -> None:
    service, conn, young, _ = env
    customer_id = young.customer.id

    service.freeze_for_scam(customer_id, {"romance"}, NOW)
    assert service.scam_flags(customer_id, NOW) == frozenset({"romance"})
    assert service.scam_flags(
        customer_id, NOW + timedelta(hours=23, minutes=59)
    ) == frozenset({"romance"})
    assert service.scam_flags(customer_id, NOW + timedelta(hours=24)) == frozenset()

    restarted = MandateService(conn, SECRET, otp_rng=random.Random(7))
    assert restarted.scam_flags(customer_id, NOW + timedelta(hours=1)) == frozenset(
        {"romance"}
    )

    later = NOW + timedelta(hours=12)
    restarted.freeze_for_scam(customer_id, {"police"}, later)
    assert restarted.scam_flags(customer_id, later) == frozenset(
        {"romance", "police"}
    )
    # The first window would have ended at NOW+24h; the second extends it.
    assert restarted.scam_flags(
        customer_id, NOW + timedelta(hours=25)
    ) == frozenset({"romance", "police"})
    assert restarted.scam_flags(
        customer_id, later + timedelta(hours=24)
    ) == frozenset()


def test_staff_unlock_clears_scam_freeze(env) -> None:
    service, conn, young, _ = env
    customer_id = young.customer.id
    service.freeze_for_scam(customer_id, {"police"}, NOW)
    assert service.scam_flags(customer_id, NOW)

    service.unlock(customer_id, by_staff="ops_alice", now=NOW)

    assert service.scam_flags(customer_id, NOW) == frozenset()
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM security_scam_freezes WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()[0]
        == 0
    )
