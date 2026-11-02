"""Tests for recurring-charge detection and subscription overview."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from bank.db import connect, create_schema, load_state
from bank.recurring import (
    detect_recurring,
    last_charge,
    recurring_overview,
    subscription_category,
)
from bank.seed import generate

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42


def _insert_debit(
    conn,
    customer_id: str,
    account_id: str,
    txn_id: str,
    when: date,
    amount_fen: int,
    counterparty: str,
    category: str = "subscriptions",
    channel: str = "online",
) -> None:
    conn.execute(
        """
        INSERT INTO transactions
            (customer_id, id, account_id, ts, amount_fen, counterparty,
             category, memo, channel, country, card_present)
        VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, 'CN', 0)
        """,
        (
            customer_id,
            txn_id,
            account_id,
            datetime(when.year, when.month, when.day, 12, 0, 0).isoformat(),
            amount_fen,
            counterparty,
            category,
            channel,
        ),
    )
    conn.commit()


def _monthly(conn, customer_id, account_id, prefix, merchant, amount_fen, category="subscriptions"):
    for index, offset in enumerate((60, 30, 0), start=1):
        _insert_debit(
            conn,
            customer_id,
            account_id,
            f"{prefix}_{index}",
            TODAY - timedelta(days=offset),
            amount_fen,
            merchant,
            category=category,
        )


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


def _find(items: list[dict], merchant: str) -> dict | None:
    for item in items:
        if item["merchant"] == merchant:
            return item
    return None


def test_detect_monthly_spotify(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    _monthly(
        conn,
        customer_id,
        state.accounts[0].id,
        "spotify",
        "Spotify",
        -1500,
    )

    items = detect_recurring(conn, customer_id, NOW)
    spotify = _find(items, "Spotify")
    assert spotify is not None
    assert spotify["amount_fen"] == 1500
    assert spotify["occurrences"] == 3
    assert spotify["category"] == "music"
    assert spotify["last_charged"] == TODAY.isoformat()
    assert spotify["next_expected"] == (TODAY + timedelta(days=30)).isoformat()


def test_irregular_payee_not_detected(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    account_id = state.accounts[0].id
    for index, offset in enumerate((70, 67, 17), start=1):
        _insert_debit(
            conn,
            customer_id,
            account_id,
            f"irregular_{index}",
            TODAY - timedelta(days=offset),
            -2000,
            "Irregular Payee",
        )

    assert _find(detect_recurring(conn, customer_id, NOW), "Irregular Payee") is None


def test_transfer_series_not_detected(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    _monthly(
        conn,
        customer_id,
        state.accounts[0].id,
        "transfer",
        "Transfer Series",
        -500,
        category="transfer",
    )

    assert _find(detect_recurring(conn, customer_id, NOW), "Transfer Series") is None


def test_amounts_outside_tolerance_not_detected(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    account_id = state.accounts[0].id
    for index, (offset, amount) in enumerate(
        ((60, -1000), (30, -1300), (0, -1000)), start=1
    ):
        _insert_debit(
            conn,
            customer_id,
            account_id,
            f"variable_{index}",
            TODAY - timedelta(days=offset),
            amount,
            "Variable Payee",
        )

    assert _find(detect_recurring(conn, customer_id, NOW), "Variable Payee") is None


def test_overview_marks_seeded_subscription_both(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    subscription = state.subscriptions[0]

    items = recurring_overview(conn, customer_id, NOW)
    matched = next(
        item for item in items if item["subscription_id"] == subscription.id
    )
    assert matched["source"] == "both"
    assert matched["status"] == subscription.status
    assert matched["amount_fen"] == subscription.amount_fen
    assert matched["last_charged"] is not None


def test_overview_includes_detected_spotify(env) -> None:
    conn, state = env
    customer_id = state.customer.id
    _monthly(
        conn,
        customer_id,
        state.accounts[0].id,
        "spotify",
        "Spotify",
        -1500,
    )

    items = recurring_overview(conn, customer_id, NOW)
    spotify = _find(items, "Spotify")
    assert spotify is not None
    assert spotify["subscription_id"] is None
    assert spotify["source"] == "detected"
    assert spotify["status"] == "detected"
    assert spotify["category"] == "music"
    assert spotify["amount_fen"] == 1500


def test_last_charge_none_without_debits(env) -> None:
    conn, state = env
    assert last_charge(conn, state.customer.id, "No Debits Merchant", NOW) is None


@pytest.mark.parametrize(
    ("merchant", "expected"),
    [
        ("腾讯视频", "video"),
        ("B站大会员", "video"),
        ("夸克网盘", "cloud"),
        ("王者荣耀 月卡", "game"),
        ("QQ音乐", "music"),
        ("Some Gym", "other"),
    ],
)
def test_subscription_category(merchant: str, expected: str) -> None:
    assert subscription_category(merchant) == expected
