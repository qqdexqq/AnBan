"""Tests for card controls and self-imposed merchant blocks."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from bank.api import BankError
from bank.controls import (
    BLOCK_CATEGORIES,
    CONTROLS,
    UNBLOCK_DELAY,
    apply_due_unblocks,
    block_category,
    blocked_categories,
    get_controls,
    is_blocked,
    request_unblock,
    set_control,
)
from bank.db import connect, create_schema, load_state
from bank.seed import generate

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42


@pytest.fixture
def env():
    state_a = generate("young_pro", SEED, TODAY)
    state_b = generate("student", SEED, TODAY)
    conn = connect()
    create_schema(conn)
    load_state(conn, state_a)
    load_state(conn, state_b)
    try:
        yield conn, state_a, state_b
    finally:
        conn.close()


def test_defaults_all_enabled(env) -> None:
    conn, state_a, _ = env
    card_id = state_a.cards[0].id

    controls = get_controls(conn, state_a.customer.id, card_id)

    assert controls == {control: True for control in CONTROLS}


def test_set_and_read_back(env) -> None:
    conn, state_a, _ = env
    card_id = state_a.cards[0].id
    customer_id = state_a.customer.id

    result = set_control(conn, customer_id, card_id, "online", False, NOW)

    assert result == {
        "card_id": card_id,
        "control": "online",
        "enabled": False,
        "previous": True,
    }
    controls = get_controls(conn, customer_id, card_id)
    assert controls["online"] is False
    assert controls["abroad"] is True
    assert controls["atm"] is True
    assert controls["contactless"] is True

    again = set_control(
        conn, customer_id, card_id, "online", True, NOW + timedelta(minutes=1)
    )
    assert again["previous"] is False
    assert get_controls(conn, customer_id, card_id)["online"] is True


def test_unknown_card_is_not_found(env) -> None:
    conn, state_a, state_b = env
    other_card = state_b.cards[0].id

    with pytest.raises(BankError) as exc:
        get_controls(conn, state_a.customer.id, "missing-card")
    assert exc.value.code == "not_found"

    with pytest.raises(BankError) as exc:
        set_control(conn, state_a.customer.id, other_card, "online", False, NOW)
    assert exc.value.code == "not_found"

    with pytest.raises(BankError) as exc:
        get_controls(conn, state_a.customer.id, other_card)
    assert exc.value.code == "not_found"


def test_invalid_control(env) -> None:
    conn, state_a, _ = env

    with pytest.raises(BankError) as exc:
        set_control(conn, state_a.customer.id, state_a.cards[0].id, "nope", False, NOW)
    assert exc.value.code == "invalid_state"


def test_block_and_is_blocked(env) -> None:
    conn, state_a, _ = env
    customer_id = state_a.customer.id

    assert block_category(conn, customer_id, "gambling", NOW) == {
        "category": "gambling",
        "blocked": True,
    }

    assert is_blocked(conn, customer_id, "gambling", NOW) is True
    assert is_blocked(conn, customer_id, "crypto", NOW) is False
    assert blocked_categories(conn, customer_id, NOW) == ["gambling"]


def test_unblock_waits_48_hours(env) -> None:
    conn, state_a, _ = env
    customer_id = state_a.customer.id

    block_category(conn, customer_id, "crypto", NOW)
    requested = request_unblock(conn, customer_id, "crypto", NOW)

    assert requested["category"] == "crypto"
    assert datetime.fromisoformat(requested["unblock_at"]) == NOW + UNBLOCK_DELAY

    almost = NOW + UNBLOCK_DELAY - timedelta(minutes=1)
    assert blocked_categories(conn, customer_id, almost) == ["crypto"]
    assert is_blocked(conn, customer_id, "crypto", almost) is True

    due = NOW + UNBLOCK_DELAY
    assert blocked_categories(conn, customer_id, due) == []
    assert is_blocked(conn, customer_id, "crypto", due) is False


def test_second_unblock_request_does_not_restart_clock(env) -> None:
    conn, state_a, _ = env
    customer_id = state_a.customer.id

    block_category(conn, customer_id, "lottery", NOW)
    first = request_unblock(conn, customer_id, "lottery", NOW)
    second = request_unblock(
        conn, customer_id, "lottery", NOW + timedelta(hours=10)
    )

    assert second["unblock_at"] == first["unblock_at"]
    assert datetime.fromisoformat(second["unblock_at"]) == NOW + UNBLOCK_DELAY
    assert blocked_categories(conn, customer_id, NOW + UNBLOCK_DELAY) == []


def test_reblocking_cancels_pending_unblock(env) -> None:
    conn, state_a, _ = env
    customer_id = state_a.customer.id

    block_category(conn, customer_id, "adult", NOW)
    request_unblock(conn, customer_id, "adult", NOW)
    block_category(conn, customer_id, "adult", NOW + timedelta(hours=1))

    assert blocked_categories(conn, customer_id, NOW + UNBLOCK_DELAY) == ["adult"]

    restarted = request_unblock(
        conn, customer_id, "adult", NOW + UNBLOCK_DELAY
    )
    assert datetime.fromisoformat(restarted["unblock_at"]) == (
        NOW + UNBLOCK_DELAY + UNBLOCK_DELAY
    )


def test_apply_due_unblocks_returns_categories(env) -> None:
    conn, state_a, _ = env
    customer_id = state_a.customer.id

    block_category(conn, customer_id, "gambling", NOW)
    block_category(conn, customer_id, "cash_like", NOW)
    block_category(conn, customer_id, "crypto", NOW)
    request_unblock(conn, customer_id, "gambling", NOW)
    request_unblock(conn, customer_id, "cash_like", NOW + timedelta(hours=1))

    assert apply_due_unblocks(conn, customer_id, NOW) == []
    assert apply_due_unblocks(conn, customer_id, NOW + UNBLOCK_DELAY) == ["gambling"]
    assert apply_due_unblocks(
        conn, customer_id, NOW + UNBLOCK_DELAY + timedelta(hours=1)
    ) == ["cash_like"]
    assert blocked_categories(conn, customer_id, NOW) == ["crypto"]


def test_invalid_category(env) -> None:
    conn, state_a, _ = env
    customer_id = state_a.customer.id

    with pytest.raises(BankError) as exc:
        block_category(conn, customer_id, "food", NOW)
    assert exc.value.code == "invalid_state"

    with pytest.raises(BankError) as exc:
        request_unblock(conn, customer_id, "food", NOW)
    assert exc.value.code == "not_found"


def test_require_block_before_unblock(env) -> None:
    conn, state_a, _ = env

    with pytest.raises(BankError) as exc:
        request_unblock(conn, state_a.customer.id, "gambling", NOW)
    assert exc.value.code == "not_found"


def test_categories_are_isolated_between_customers(env) -> None:
    conn, state_a, state_b = env

    block_category(conn, state_a.customer.id, "gambling", NOW)

    assert blocked_categories(conn, state_b.customer.id, NOW) == []
    assert is_blocked(conn, state_b.customer.id, "gambling", NOW) is False


def test_catalogue_constants() -> None:
    assert CONTROLS == ("online", "abroad", "atm", "contactless")
    assert BLOCK_CATEGORIES == ("gambling", "crypto", "lottery", "adult", "cash_like")
    assert UNBLOCK_DELAY == timedelta(hours=48)
