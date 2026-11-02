"""Tests for releasing one pending transfer early (bank + policy)."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta

import pytest

from bank.api import Bank, BankError
from bank.db import connect, create_schema, load_state
from bank.seed import generate
from security import compliance
from security.mandate import create_security_schema
from security.policy import Rule, evaluate_early_release

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42


def _open(persona: str = "young_pro"):
    state = generate(persona, SEED, TODAY)
    conn = connect()
    create_schema(conn)
    load_state(conn, state)
    return Bank(conn, state.customer.id), state, conn


@pytest.fixture
def env():
    bank, state, conn = _open()
    try:
        yield bank, state, conn
    finally:
        conn.close()


@pytest.fixture
def elder_env():
    bank, state, conn = _open("retiree")
    try:
        yield bank, state, conn
    finally:
        conn.close()


@pytest.fixture
def multi_env():
    state_a = generate("young_pro", SEED, TODAY)
    state_b = generate("student", SEED, TODAY)
    conn = connect()
    create_schema(conn)
    load_state(conn, state_a)
    load_state(conn, state_b)
    bank_a = Bank(conn, state_a.customer.id)
    bank_b = Bank(conn, state_b.customer.id)
    try:
        yield bank_a, state_a, bank_b, state_b, conn
    finally:
        conn.close()


def _contact_id(bank: Bank, name: str = "张伟") -> str:
    return bank.find_contact(name)[0].id


def _any_contact(bank: Bank) -> str:
    return bank.find_contact("")[0].id


def _txn_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]


def _status(conn: sqlite3.Connection, customer_id: str, transfer_id: str) -> str:
    return conn.execute(
        "SELECT status FROM transfers WHERE customer_id = ? AND id = ?",
        (customer_id, transfer_id),
    ).fetchone()[0]


def _add_scam_freeze(
    conn: sqlite3.Connection, customer_id: str, until: datetime
) -> None:
    conn.execute(
        """
        INSERT INTO security_scam_freezes
            (customer_id, frozen_until, flags_json, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (customer_id, until.isoformat(), '["test"]', NOW.isoformat()),
    )
    conn.commit()


def _release(bank: Bank, transfer, now: datetime = NOW + timedelta(minutes=5)):
    return bank.release_early(
        transfer.id,
        now,
        expected_contact_id=transfer.to_contact_id,
        expected_amount_fen=transfer.amount_fen,
    )


# ---------------------------------------------------------------------- bank


def test_release_early_releases_only_selected_and_keeps_reserved_balance(env) -> None:
    bank, _, conn = env
    contact = _contact_id(bank)
    before = bank.get_balance()
    first = bank.transfer(contact, 10_000, "rent", "early-1", NOW)
    second = bank.transfer(contact, 20_000, "food", "early-2", NOW)
    txn_before = _txn_count(conn)

    released = _release(bank, first)

    assert released.id == first.id
    assert released.status == "released"
    # Early release does not move money again: it was reserved at transfer time.
    assert bank.get_balance() == before - 30_000
    assert _status(conn, bank.customer_id, first.id) == "released"
    assert _status(conn, bank.customer_id, second.id) == "pending"
    # The normal delay is untouched by the early path.
    assert released.created_at == first.created_at
    assert released.release_at == first.release_at
    assert _txn_count(conn) == txn_before + 1

    created = bank.list_transactions(limit=1)[0]
    assert created.amount_fen == -10_000
    assert created.counterparty == "张伟"
    assert created.category == "transfer"
    assert created.memo == "rent"
    assert created.channel == "transfer"
    assert created.country == "CN"
    assert created.account_id == first.account_id


def test_release_early_replay_is_window_closed(env) -> None:
    bank, _, conn = env
    transfer = bank.transfer(_contact_id(bank), 10_000, "", "early-replay", NOW)
    _release(bank, transfer)
    txn_after_release = _txn_count(conn)

    with pytest.raises(BankError) as exc:
        _release(bank, transfer)

    assert exc.value.code == "window_closed"
    assert _status(conn, bank.customer_id, transfer.id) == "released"
    assert _txn_count(conn) == txn_after_release


def test_release_early_requires_the_approved_contact_and_amount(env) -> None:
    bank, _, conn = env
    transfer = bank.transfer(_contact_id(bank), 10_000, "", "early-match", NOW)
    txn_before = _txn_count(conn)

    with pytest.raises(BankError) as exc:
        bank.release_early(
            transfer.id,
            NOW + timedelta(minutes=1),
            expected_contact_id="someone-else",
            expected_amount_fen=transfer.amount_fen,
        )
    assert exc.value.code == "invalid_state"

    with pytest.raises(BankError) as exc:
        bank.release_early(
            transfer.id,
            NOW + timedelta(minutes=1),
            expected_contact_id=transfer.to_contact_id,
            expected_amount_fen=transfer.amount_fen + 1,
        )
    assert exc.value.code == "invalid_state"

    assert _status(conn, bank.customer_id, transfer.id) == "pending"
    assert _txn_count(conn) == txn_before


def test_release_early_is_customer_scoped(multi_env) -> None:
    bank_a, state_a, bank_b, state_b, conn = multi_env
    transfer_a = bank_a.transfer(
        state_a.contacts[0].id, 5_000, "", "a-early", NOW
    )
    txn_before = _txn_count(conn)

    with pytest.raises(BankError) as exc:
        bank_b.release_early(
            transfer_a.id,
            NOW + timedelta(minutes=1),
            expected_contact_id=transfer_a.to_contact_id,
            expected_amount_fen=transfer_a.amount_fen,
        )

    assert exc.value.code == "not_found"
    assert _status(conn, state_a.customer.id, transfer_a.id) == "pending"
    assert _txn_count(conn) == txn_before


def test_release_early_respects_a_scam_freeze(env) -> None:
    bank, _, conn = env
    create_security_schema(conn)
    transfer = bank.transfer(_contact_id(bank), 10_000, "", "early-scam", NOW)
    _add_scam_freeze(conn, bank.customer_id, NOW + timedelta(hours=24))
    txn_before = _txn_count(conn)

    with pytest.raises(BankError) as exc:
        _release(bank, transfer)

    assert exc.value.code == "R_SCAM"
    assert _status(conn, bank.customer_id, transfer.id) == "pending"
    assert _txn_count(conn) == txn_before


def test_release_early_respects_the_fraud_list(env, tmp_path, monkeypatch) -> None:
    fraud_file = tmp_path / "fraud_accounts.json"
    fraud_file.write_text(json.dumps({"accounts": []}), encoding="utf-8")
    monkeypatch.setattr(compliance, "FRAUD_LIST_PATH", fraud_file)

    bank, _, conn = env
    contact = _contact_id(bank)
    transfer = bank.transfer(contact, 10_000, "", "early-fraud", NOW)
    account_no = conn.execute(
        "SELECT account_no FROM contacts WHERE customer_id = ? AND id = ?",
        (bank.customer_id, contact),
    ).fetchone()[0]
    fraud_file.write_text(
        json.dumps(
            {"accounts": [{"account_no": account_no, "reason": "test listing"}]}
        ),
        encoding="utf-8",
    )
    txn_before = _txn_count(conn)

    with pytest.raises(BankError) as exc:
        _release(bank, transfer)

    assert exc.value.code == "R_FRAUD_LIST"
    assert _status(conn, bank.customer_id, transfer.id) == "pending"
    assert _txn_count(conn) == txn_before


@pytest.mark.parametrize("state", ["missing", "malformed"])
def test_release_early_fails_closed_when_hold_check_is_unavailable(
    env, tmp_path, monkeypatch, state
) -> None:
    fraud_file = tmp_path / "fraud_accounts.json"
    if state == "malformed":
        fraud_file.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(compliance, "FRAUD_LIST_PATH", fraud_file)

    bank, _, conn = env
    transfer = bank.transfer(_contact_id(bank), 10_000, "", f"early-{state}", NOW)
    txn_before = _txn_count(conn)

    with pytest.raises(BankError) as exc:
        _release(bank, transfer)

    assert exc.value.code == "R_RELEASE_CHECK_UNAVAILABLE"
    assert _status(conn, bank.customer_id, transfer.id) == "pending"
    assert _txn_count(conn) == txn_before


def test_release_early_defers_to_the_outer_transaction(env) -> None:
    bank, _, conn = env
    transfer = bank.transfer(_contact_id(bank), 10_000, "", "early-outer", NOW)
    txn_before = _txn_count(conn)

    conn.execute("BEGIN IMMEDIATE")
    try:
        released = _release(bank, transfer)
        assert released.status == "released"
        # Visible inside the caller's transaction, but not committed by the bank.
        assert _status(conn, bank.customer_id, transfer.id) == "released"
        assert _txn_count(conn) == txn_before + 1
    finally:
        conn.rollback()

    assert _status(conn, bank.customer_id, transfer.id) == "pending"
    assert _txn_count(conn) == txn_before


def test_release_early_rolls_back_cleanly_inside_an_outer_transaction(env) -> None:
    bank, _, conn = env
    transfer = bank.transfer(_contact_id(bank), 10_000, "", "early-failure", NOW)
    txn_before = _txn_count(conn)
    conn.execute(
        """
        CREATE TRIGGER fail_early_txn
        BEFORE INSERT ON transactions
        BEGIN
            SELECT RAISE(ABORT, 'forced failure');
        END;
        """
    )
    conn.commit()

    conn.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            _release(bank, transfer)
    finally:
        conn.rollback()

    assert _status(conn, bank.customer_id, transfer.id) == "pending"
    assert _txn_count(conn) == txn_before


# -------------------------------------------------------------------- policy


def test_evaluate_early_release_is_red_with_mfa(env) -> None:
    bank, _, _ = env
    transfer = bank.transfer(_contact_id(bank), 10_000, "", "policy-red", NOW)

    decision = evaluate_early_release(
        bank=bank, transfer_id=transfer.id, now=NOW + timedelta(minutes=1)
    )

    assert decision.step_id == "release"
    assert decision.tool == "release_transfer_now"
    assert decision.tier == "red"
    assert decision.needs_mandate is True
    assert decision.needs_mfa is True
    assert decision.needs_co_approval is False
    assert decision.vetoed is False
    assert decision.amount_fen == transfer.amount_fen
    assert any(reason.code == Rule.EARLY_RELEASE for reason in decision.reasons)


def test_evaluate_early_release_requires_elder_guardian(elder_env) -> None:
    bank, _, _ = elder_env
    transfer = bank.transfer(_any_contact(bank), 5_000, "", "policy-elder", NOW)

    decision = evaluate_early_release(
        bank=bank, transfer_id=transfer.id, now=NOW + timedelta(minutes=1)
    )

    assert decision.tier == "red"
    assert decision.needs_mfa is True
    assert decision.needs_co_approval is True
    assert any(reason.code == Rule.ELDER for reason in decision.reasons)
    assert decision.vetoed is False


def test_evaluate_early_release_vetoes_on_a_scam_freeze(env) -> None:
    bank, _, conn = env
    create_security_schema(conn)
    transfer = bank.transfer(_contact_id(bank), 5_000, "", "policy-freeze", NOW)
    _add_scam_freeze(conn, bank.customer_id, NOW + timedelta(hours=24))

    decision = evaluate_early_release(
        bank=bank, transfer_id=transfer.id, now=NOW + timedelta(minutes=1)
    )

    assert decision.tier == "red"
    assert decision.vetoed is True
    assert any(reason.code == Rule.SCAM for reason in decision.reasons)


def test_evaluate_early_release_vetoes_missing_or_stale_transfers(env) -> None:
    bank, _, _ = env
    transfer = bank.transfer(_contact_id(bank), 5_000, "", "policy-stale", NOW)
    _release(bank, transfer)

    missing = evaluate_early_release(bank=bank, transfer_id="tr_missing", now=NOW)
    stale = evaluate_early_release(bank=bank, transfer_id=transfer.id, now=NOW)

    assert missing.vetoed is True
    assert stale.vetoed is True


def test_evaluate_early_release_fails_closed_without_a_fraud_list(
    env, tmp_path, monkeypatch
) -> None:
    fraud_file = tmp_path / "fraud_accounts.json"
    fraud_file.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(compliance, "FRAUD_LIST_PATH", fraud_file)

    bank, _, _ = env
    transfer = bank.transfer(_contact_id(bank), 5_000, "", "policy-unavailable", NOW)

    decision = evaluate_early_release(
        bank=bank, transfer_id=transfer.id, now=NOW + timedelta(minutes=1)
    )

    assert decision.vetoed is True
    assert any(
        reason.code == Rule.RELEASE_CHECK_UNAVAILABLE
        for reason in decision.reasons
    )
