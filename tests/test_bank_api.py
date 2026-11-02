"""Tests for the SQLite-backed mock bank API."""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import date, datetime, timedelta
from inspect import signature

import pytest

from bank.api import Bank, BankError
from bank.db import connect, create_schema, load_state
from bank.merchants import price_of
from bank.seed import generate
from security import compliance
from security.mandate import create_security_schema

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42


def _assets(conn: sqlite3.Connection) -> int:
    """Total value held by the customer (cash, pots, pending holds, orders)."""
    current = conn.execute(
        "SELECT balance_fen FROM accounts WHERE type = 'current'"
    ).fetchone()[0]
    savings = conn.execute(
        "SELECT balance_fen FROM accounts WHERE type = 'savings'"
    ).fetchone()[0]
    pots = conn.execute("SELECT COALESCE(SUM(balance_fen), 0) FROM pots").fetchone()[0]
    holds = conn.execute(
        "SELECT COALESCE(SUM(amount_fen), 0) FROM transfers WHERE status = 'pending'"
    ).fetchone()[0]
    orders = conn.execute(
        """
        SELECT COALESCE(SUM(amount_fen), 0)
        FROM merchant_orders WHERE status = 'open'
        """
    ).fetchone()[0]
    return int(current + savings + pots + holds + orders)


@pytest.fixture
def env():
    state = generate("young_pro", SEED, TODAY)
    conn = connect()
    create_schema(conn)
    load_state(conn, state)
    bank = Bank(conn, state.customer.id)
    try:
        yield bank, state, conn
    finally:
        conn.close()


def _contact_id(bank: Bank, name: str = "张伟") -> str:
    return bank.find_contact(name)[0].id


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


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }


def test_idempotent_transfer(env) -> None:
    bank, _, conn = env
    contact_id = _contact_id(bank)
    before = bank.get_balance()

    first = bank.transfer(contact_id, 10_000, "lunch", "idem-1", NOW)
    second = bank.transfer(contact_id, 10_000, "lunch", "idem-1", NOW + timedelta(hours=1))

    assert second.id == first.id
    assert bank.get_balance() == before - 10_000
    assert conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0] == 1

    with pytest.raises(BankError) as exc:
        bank.transfer(contact_id, 20_000, "lunch", "idem-1", NOW)
    assert exc.value.code == "invalid_state"
    assert bank.get_balance() == before - 10_000


def test_insufficient_funds_changes_nothing(env) -> None:
    bank, _, conn = env
    contact_id = _contact_id(bank)
    before = bank.get_balance()

    with pytest.raises(BankError) as exc:
        bank.transfer(contact_id, before + 1, "", "too-much", NOW)
    assert exc.value.code == "insufficient_funds"
    assert bank.get_balance() == before
    assert conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0] == 0


def test_unknown_contact_raises_before_debit(env) -> None:
    bank, _, conn = env
    before = bank.get_balance()

    with pytest.raises(BankError) as exc:
        bank.transfer("missing-contact", 1000, "", "missing", NOW)
    assert exc.value.code == "not_found"
    assert bank.get_balance() == before
    assert conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0] == 0


def test_cancel_inside_window_refunds(env) -> None:
    bank, _, _ = env
    contact_id = _contact_id(bank)
    before = bank.get_balance()
    transfer = bank.transfer(contact_id, 10_000, "", "cancel-now", NOW)

    cancelled = bank.cancel_transfer(transfer.id, NOW + timedelta(hours=1))
    assert cancelled.status == "cancelled"
    assert bank.get_balance() == before


def test_cancel_after_release_at_is_window_closed(env) -> None:
    bank, _, _ = env
    contact_id = _contact_id(bank)
    before = bank.get_balance()
    transfer = bank.transfer(contact_id, 10_000, "", "late-cancel", NOW)

    with pytest.raises(BankError) as exc:
        bank.cancel_transfer(transfer.id, transfer.release_at)
    assert exc.value.code == "window_closed"
    assert bank.get_balance() == before - 10_000


def test_release_due_creates_transaction(env) -> None:
    bank, _, conn = env
    contact_id = _contact_id(bank)
    transfer = bank.transfer(contact_id, 10_000, "rent share", "release-1", NOW)
    txn_count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    released = bank.release_due(transfer.release_at)
    assert len(released) == 1
    assert released[0].status == "released"
    assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == txn_count + 1

    created = bank.list_transactions(limit=1)[0]
    assert created.category == "transfer"
    assert created.amount_fen == -10_000
    assert created.counterparty == "张伟"

    assert bank.release_due(transfer.release_at + timedelta(days=1)) == []
    assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == txn_count + 1


def test_active_scam_freeze_holds_overdue_transfer(env) -> None:
    bank, _, conn = env
    create_security_schema(conn)
    contact_id = _contact_id(bank)
    before = bank.get_balance()
    transfer = bank.transfer(contact_id, 10_000, "held", "scam-hold", NOW)
    _add_scam_freeze(conn, bank.customer_id, NOW + timedelta(hours=24))
    txn_count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    assert bank.release_due(transfer.release_at + timedelta(hours=3)) == []

    row = conn.execute(
        "SELECT status, release_at FROM transfers WHERE customer_id = ? AND id = ?",
        (bank.customer_id, transfer.id),
    ).fetchone()
    assert row["status"] == "pending"
    assert datetime.fromisoformat(row["release_at"]) == transfer.release_at
    assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == txn_count
    assert bank.get_balance() == before - 10_000


def test_overdue_held_transfer_refunds_exactly_once(env) -> None:
    bank, _, conn = env
    create_security_schema(conn)
    contact_id = _contact_id(bank)
    before = bank.get_balance()
    transfer = bank.transfer(contact_id, 10_000, "", "held-refund", NOW)
    _add_scam_freeze(conn, bank.customer_id, NOW + timedelta(hours=24))
    overdue = transfer.release_at + timedelta(hours=3)

    cancelled = bank.cancel_transfer(transfer.id, overdue)
    assert cancelled.status == "cancelled"
    assert bank.get_balance() == before

    with pytest.raises(BankError) as exc:
        bank.cancel_transfer(transfer.id, overdue + timedelta(minutes=1))
    assert exc.value.code == "window_closed"
    assert bank.get_balance() == before


def test_freeze_expiry_allows_release_once(env) -> None:
    bank, _, conn = env
    create_security_schema(conn)
    contact_id = _contact_id(bank)
    transfer = bank.transfer(contact_id, 10_000, "", "freeze-expiry", NOW)
    until = NOW + timedelta(hours=3)
    _add_scam_freeze(conn, bank.customer_id, until)

    # now == frozen_until is no longer active (now < until is strict).
    assert [item.id for item in bank.release_due(until)] == [transfer.id]
    assert bank.release_due(until) == []


def test_scam_freeze_is_customer_scoped(multi_env) -> None:
    bank_a, state_a, bank_b, state_b, conn = multi_env
    create_security_schema(conn)
    contact_a = state_a.contacts[0].id
    contact_b = state_b.contacts[0].id
    before_b = bank_b.get_balance()
    transfer_a = bank_a.transfer(contact_a, 5_000, "", "a-freeze", NOW)
    transfer_b = bank_b.transfer(contact_b, 7_000, "", "b-freeze", NOW)
    _add_scam_freeze(conn, state_a.customer.id, NOW + timedelta(hours=24))

    assert bank_a.release_due(transfer_a.release_at + timedelta(hours=1)) == []
    released_b = bank_b.release_due(transfer_b.release_at + timedelta(hours=1))
    assert [item.id for item in released_b] == [transfer_b.id]
    assert bank_b.get_balance() == before_b - 7_000
    assert (
        conn.execute(
            "SELECT status FROM transfers WHERE customer_id = ? AND id = ?",
            (state_a.customer.id, transfer_a.id),
        ).fetchone()[0]
        == "pending"
    )


def test_fraud_list_change_after_approval_holds_release(
    env, tmp_path, monkeypatch
) -> None:
    fraud_file = tmp_path / "fraud_accounts.json"
    fraud_file.write_text(json.dumps({"accounts": []}), encoding="utf-8")
    monkeypatch.setattr(compliance, "FRAUD_LIST_PATH", fraud_file)

    bank, _, conn = env
    contact_id = _contact_id(bank)
    account_no = conn.execute(
        "SELECT account_no FROM contacts WHERE customer_id = ? AND id = ?",
        (bank.customer_id, contact_id),
    ).fetchone()[0]
    digits = compliance.normalize_account(account_no)

    # An earlier read must not be cached for the rest of the process.
    assert compliance.fraud_accounts() == {}

    transfer = bank.transfer(contact_id, 10_000, "", "fresh-list", NOW)
    spaced = " ".join(digits[index : index + 4] for index in range(0, len(digits), 4))
    fraud_file.write_text(
        json.dumps({"accounts": [{"account_no": spaced, "reason": "test listing"}]}),
        encoding="utf-8",
    )

    assert bank.release_due(transfer.release_at + timedelta(hours=1)) == []
    assert (
        conn.execute(
            "SELECT status FROM transfers WHERE customer_id = ? AND id = ?",
            (bank.customer_id, transfer.id),
        ).fetchone()[0]
        == "pending"
    )


@pytest.mark.parametrize("state", ["missing", "malformed"])
def test_unavailable_fraud_list_blocks_release_but_allows_refund(
    env, tmp_path, monkeypatch, state
) -> None:
    fraud_file = tmp_path / "fraud_accounts.json"
    if state == "malformed":
        fraud_file.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(compliance, "FRAUD_LIST_PATH", fraud_file)

    bank, _, conn = env
    contact_id = _contact_id(bank)
    before = bank.get_balance()
    transfer = bank.transfer(contact_id, 10_000, "", f"bad-list-{state}", NOW)
    overdue = transfer.release_at + timedelta(hours=3)

    assert bank.release_due(overdue) == []
    assert bank.get_balance() == before - 10_000

    assert bank.cancel_transfer(transfer.id, overdue).status == "cancelled"
    assert bank.get_balance() == before


def test_missing_security_tables_do_not_change_bank_behaviour(env) -> None:
    bank, _, conn = env
    tables_before = _table_names(conn)
    assert "security_scam_freezes" not in tables_before
    assert "security_lockouts" not in tables_before

    contact_id = _contact_id(bank)
    transfer = bank.transfer(contact_id, 10_000, "", "no-security-schema", NOW)

    released = bank.release_due(transfer.release_at)
    assert [item.id for item in released] == [transfer.id]
    assert _table_names(conn) == tables_before

    # The same idempotency key still resolves to the original row (same cooling).
    assert (
        bank.transfer(
            contact_id, 10_000, "", "no-security-schema", NOW + timedelta(hours=1)
        ).id
        == transfer.id
    )

    second = bank.transfer(contact_id, 10_000, "", "no-security-window", NOW)
    with pytest.raises(BankError) as exc:
        bank.cancel_transfer(second.id, second.release_at)
    assert exc.value.code == "window_closed"


def test_daily_outgoing_total(env) -> None:
    bank, _, _ = env
    contact_id = _contact_id(bank)
    for index in range(5):
        bank.transfer(contact_id, 30_000, "", f"daily-{index}", NOW)
    assert bank.daily_outgoing_total(NOW.date()) == 150_000
    assert bank.daily_outgoing_total(TODAY - timedelta(days=1)) == 0


def test_find_contact_duplicate_names_and_filters(env) -> None:
    bank, state, _ = env
    zhang_weis = bank.find_contact("张伟")
    assert len(zhang_weis) == 2
    assert zhang_weis[0].account_no != zhang_weis[1].account_no
    assert len(bank.find_contact("BADMINTON")) == 1
    assert bank.find_contact("landlord")[0].name == "王建国"

    lina = bank.find_contact("李娜 Lina")[0]
    birthday = bank.get_contact_date(lina.id)
    assert birthday is not None
    expected = TODAY + timedelta(days=12)
    assert (birthday.month, birthday.day) == (expected.month, expected.day)


def test_detect_anomalies_finds_seeded_anomalies(env) -> None:
    bank, _, _ = env
    anomalies = bank.detect_anomalies(NOW)
    kinds = {anomaly["kind"] for anomaly in anomalies}
    assert kinds == {"duplicate", "foreign_cnp"}

    duplicate = next(a for a in anomalies if a["kind"] == "duplicate")
    assert len(duplicate["txn_ids"]) == 2
    foreign = next(a for a in anomalies if a["kind"] == "foreign_cnp")
    assert len(foreign["txn_ids"]) == 1


def test_spending_report_matches_manual_sum(env) -> None:
    bank, _, _ = env
    start = TODAY - timedelta(days=365)
    end = TODAY - timedelta(days=1)
    report = bank.spending_report(start, end)

    txns = bank.list_transactions(start, end, limit=100_000)
    manual = sum(
        -txn.amount_fen
        for txn in txns
        if txn.amount_fen < 0
        and not (
            txn.category == "transfer" and txn.counterparty == "我的储蓄账户"
        )
    )
    assert report["total"] == manual
    assert sum(report["by_category"].values()) == manual


def test_pot_and_subscription_undo_paths(env) -> None:
    bank, _, _ = env
    before = bank.get_balance()

    pot = bank.move_to_pot("新基金", 10_000)
    assert pot.balance_fen == 10_000
    assert bank.get_balance() == before - 10_000

    pot = bank.move_from_pot("新基金", 10_000)
    assert pot.balance_fen == 0
    assert bank.get_balance() == before

    with pytest.raises(BankError) as exc:
        bank.move_from_pot("新基金", 1)
    assert exc.value.code == "insufficient_funds"

    subscription = bank.list_subscriptions()[0]
    assert bank.cancel_subscription(subscription.id).status == "cancelled"
    assert bank.restore_subscription(subscription.id).status == "active"
    with pytest.raises(BankError) as exc:
        bank.restore_subscription(subscription.id)
    assert exc.value.code == "invalid_state"


def test_subscribe_product_debits_records_and_tops_up(env) -> None:
    bank, _, conn = env
    product = next(item for item in bank.list_products() if item.id == "p2")
    before = bank.get_balance()
    txn_count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    holding = bank.subscribe_product(product.id, 20_000, NOW)
    assert holding.product_id == product.id
    assert holding.principal_fen == 20_000
    assert holding.value_fen == 20_000
    assert bank.get_balance() == before - 20_000

    portfolio = {item.product_id: item for item in bank.get_portfolio()}
    assert portfolio[product.id].principal_fen == 20_000
    assert portfolio[product.id].value_fen == 20_000
    assert (
        conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        == txn_count + 1
    )

    created = bank.list_transactions(limit=1)[0]
    assert created.amount_fen == -20_000
    assert created.counterparty == product.name
    assert created.category == "investment"
    assert created.channel == "app"

    topped = bank.subscribe_product(product.id, 10_000, NOW)
    assert topped.principal_fen == 30_000
    assert topped.value_fen == 30_000
    assert bank.get_balance() == before - 30_000
    assert (
        conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        == txn_count + 2
    )


def test_subscribe_product_counts_towards_daily_outgoing_total(env) -> None:
    bank, _, _ = env
    before = bank.daily_outgoing_total(NOW)

    bank.subscribe_product("p2", 20_000, NOW)

    assert bank.daily_outgoing_total(NOW) == before + 20_000


def test_subscribe_product_below_minimum_is_rejected(env) -> None:
    # Regression: ¥0.01 purchases went through despite min_fen.
    bank, _, _ = env
    product = next(item for item in bank.list_products() if item.id == "p2")
    before = bank.get_balance()

    with pytest.raises(BankError) as exc:
        bank.subscribe_product(product.id, product.min_fen - 1, NOW)

    assert exc.value.code == "invalid_amount"
    assert bank.get_balance() == before


def test_subscribe_product_insufficient_funds_changes_nothing(env) -> None:
    bank, _, conn = env
    before = bank.get_balance()
    holdings_before = bank.get_portfolio()
    txn_count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    with pytest.raises(BankError) as exc:
        bank.subscribe_product("p2", before + 1, NOW)
    assert exc.value.code == "insufficient_funds"
    assert bank.get_balance() == before
    assert bank.get_portfolio() == holdings_before
    assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == txn_count


def test_subscribe_product_unknown_product_raises(env) -> None:
    bank, _, conn = env
    before = bank.get_balance()
    holdings_before = bank.get_portfolio()
    txn_count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    with pytest.raises(BankError) as exc:
        bank.subscribe_product("missing-product", 20_000, NOW)
    assert exc.value.code == "not_found"
    assert bank.get_balance() == before
    assert bank.get_portfolio() == holdings_before
    assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == txn_count

    with pytest.raises(BankError) as exc:
        bank.subscribe_product("p2", 0, NOW)
    assert exc.value.code == "invalid_amount"
    assert bank.get_balance() == before


def test_card_undo_paths_and_lost_card(env) -> None:
    bank, _, _ = env
    card = bank.list_cards()[0]

    assert bank.freeze_card(card.id).status == "frozen"
    assert bank.unfreeze_card(card.id).status == "active"
    assert bank.set_card_limit(card.id, 12_345).online_limit_fen == 12_345

    virtual = bank.create_virtual_card(50_000)
    assert virtual.kind == "virtual"
    assert bank.close_virtual_card(virtual.id).status == "closed"

    with pytest.raises(BankError) as exc:
        bank.close_virtual_card(card.id)
    assert exc.value.code == "invalid_state"

    assert bank.report_card_lost(card.id).status == "lost"
    with pytest.raises(BankError) as exc:
        bank.unfreeze_card(card.id)
    assert exc.value.code == "invalid_state"


def test_order_undo_paths(env) -> None:
    bank, _, _ = env
    before = bank.get_balance()
    bouquet_price = price_of("花店", "bouquet_m")

    order = bank.merchant_order(
        "花店", "bouquet_m", TODAY + timedelta(days=1), NOW
    )
    assert order.status == "open"
    assert order.item == "bouquet_m"
    assert order.amount_fen == bouquet_price
    assert bank.get_balance() == before - bouquet_price
    assert bank.cancel_order(order.id, NOW).status == "cancelled"
    assert bank.get_balance() == before

    cake_price = price_of("蛋糕店", "cake_6in")
    late = bank.merchant_order("蛋糕店", "cake_6in", TODAY, NOW)
    assert late.amount_fen == cake_price
    with pytest.raises(BankError) as exc:
        bank.cancel_order(late.id, NOW)
    assert exc.value.code == "window_closed"
    assert bank.get_balance() == before - cake_price


def test_merchant_order_price_comes_from_catalogue(env) -> None:
    bank, _, conn = env
    before = bank.get_balance()

    order = bank.merchant_order(
        "花店", "roses_11", TODAY + timedelta(days=1), NOW
    )

    assert order.amount_fen == price_of("花店", "roses_11")
    assert bank.get_balance() == before - order.amount_fen
    row = conn.execute(
        "SELECT item, amount_fen FROM merchant_orders WHERE id = ?",
        (order.id,),
    ).fetchone()
    assert row["item"] == "roses_11"
    assert row["amount_fen"] == order.amount_fen

    # No amount parameter exists, so a caller cannot override the price.
    assert "amount" not in signature(Bank.merchant_order).parameters
    assert "amount_fen" not in signature(Bank.merchant_order).parameters
    with pytest.raises(TypeError):
        bank.merchant_order(
            "花店",
            "roses_11",
            TODAY + timedelta(days=1),
            NOW,
            amount_fen=1,
        )


def test_merchant_order_rejects_unknown_merchant_or_item(env) -> None:
    bank, _, conn = env
    before = bank.get_balance()

    with pytest.raises(BankError) as exc:
        bank.merchant_order(
            "花店", "cake_8in", TODAY + timedelta(days=1), NOW
        )
    assert exc.value.code == "invalid_state"

    with pytest.raises(BankError) as exc:
        bank.merchant_order(
            "unknown shop", "bouquet_m", TODAY + timedelta(days=1), NOW
        )
    assert exc.value.code == "invalid_state"

    assert bank.get_balance() == before
    assert conn.execute("SELECT COUNT(*) FROM merchant_orders").fetchone()[0] == 0


def test_daily_outgoing_total_includes_merchant_orders(env) -> None:
    bank, _, _ = env

    open_order = bank.merchant_order(
        "花店", "bouquet_m", TODAY + timedelta(days=1), NOW
    )
    assert bank.daily_outgoing_total(TODAY) == price_of("花店", "bouquet_m")

    bank.cancel_order(open_order.id, NOW)
    assert bank.daily_outgoing_total(TODAY) == 0

    bank.merchant_order(
        "蛋糕店",
        "cake_8in",
        TODAY,
        NOW - timedelta(days=1),
    )
    assert bank.daily_outgoing_total(TODAY) == 0
    assert bank.daily_outgoing_total(TODAY - timedelta(days=1)) == price_of(
        "蛋糕店", "cake_8in"
    )


def test_random_operations_conserve_value(env) -> None:
    bank, state, conn = env
    rng = random.Random(1234)
    contacts = [contact.id for contact in state.contacts]
    operations = [
        "transfer",
        "cancel_transfer",
        "release",
        "pot_in",
        "pot_out",
        "order",
        "cancel_order",
        "schedule",
        "cancel_schedule",
        "request",
        "cancel_request",
        "sub_cancel",
        "sub_restore",
        "freeze",
        "unfreeze",
        "limit",
        "virtual",
        "close_virtual",
        "lost",
    ]

    for step in range(50):
        before = _assets(conn)
        op_now = NOW + timedelta(minutes=step)
        released: list = []
        try:
            operation = rng.choice(operations)
            if operation == "transfer":
                bank.transfer(
                    rng.choice(contacts),
                    rng.randint(100, 2_000),
                    "random",
                    f"rand-{step}",
                    op_now,
                )
            elif operation == "cancel_transfer":
                pending = [
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM transfers WHERE status = 'pending'"
                    ).fetchall()
                ]
                if pending:
                    bank.cancel_transfer(rng.choice(pending), op_now)
            elif operation == "release":
                released = bank.release_due(NOW + timedelta(days=2))
            elif operation == "pot_in":
                bank.move_to_pot(rng.choice(("A", "B")), rng.randint(100, 2_000))
            elif operation == "pot_out":
                bank.move_from_pot(rng.choice(("A", "B")), rng.randint(100, 2_000))
            elif operation == "order":
                merchant, item_code = rng.choice(
                    (
                        ("花店", "bouquet_s"),
                        ("花店", "roses_11"),
                        ("蛋糕店", "cake_6in"),
                        ("蛋糕店", "cake_8in"),
                    )
                )
                bank.merchant_order(
                    merchant,
                    item_code,
                    TODAY + timedelta(days=1),
                    op_now,
                )
            elif operation == "cancel_order":
                open_orders = [
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM merchant_orders WHERE status = 'open'"
                    ).fetchall()
                ]
                if open_orders:
                    bank.cancel_order(rng.choice(open_orders), op_now)
            elif operation == "schedule":
                bank.schedule_transfer(
                    rng.choice(contacts), 1_000, 15, "scheduled"
                )
            elif operation == "cancel_schedule":
                scheduled = [
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM scheduled_transfers WHERE status = 'scheduled'"
                    ).fetchall()
                ]
                if scheduled:
                    bank.cancel_scheduled(rng.choice(scheduled))
            elif operation == "request":
                bank.request_payment(rng.choice(contacts), 1_000, "split")
            elif operation == "cancel_request":
                requests = [
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM payment_requests WHERE status = 'pending'"
                    ).fetchall()
                ]
                if requests:
                    bank.cancel_request(rng.choice(requests))
            elif operation == "sub_cancel":
                active = [
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM subscriptions WHERE status = 'active'"
                    ).fetchall()
                ]
                if active:
                    bank.cancel_subscription(rng.choice(active))
            elif operation == "sub_restore":
                cancelled = [
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM subscriptions WHERE status = 'cancelled'"
                    ).fetchall()
                ]
                if cancelled:
                    bank.restore_subscription(rng.choice(cancelled))
            elif operation == "freeze":
                cards = [
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM cards WHERE status = 'active'"
                    ).fetchall()
                ]
                if cards:
                    bank.freeze_card(rng.choice(cards))
            elif operation == "unfreeze":
                cards = [
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM cards WHERE status = 'frozen'"
                    ).fetchall()
                ]
                if cards:
                    bank.unfreeze_card(rng.choice(cards))
            elif operation == "limit":
                cards = [row[0] for row in conn.execute("SELECT id FROM cards").fetchall()]
                if cards:
                    bank.set_card_limit(rng.choice(cards), rng.randint(0, 100_000))
            elif operation == "virtual":
                bank.create_virtual_card(rng.randint(0, 100_000))
            elif operation == "close_virtual":
                virtual = [
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM cards WHERE kind = 'virtual' AND status = 'active'"
                    ).fetchall()
                ]
                if virtual:
                    bank.close_virtual_card(rng.choice(virtual))
            elif operation == "lost":
                cards = [
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM cards WHERE status != 'lost'"
                    ).fetchall()
                ]
                if cards:
                    bank.report_card_lost(rng.choice(cards))
        except BankError:
            assert _assets(conn) == before
            continue

        after = _assets(conn)
        external_out = sum(transfer.amount_fen for transfer in released)
        assert after == before - external_out


def test_failed_write_rolls_back(env) -> None:
    bank, _, conn = env
    contact_id = _contact_id(bank)
    before = bank.get_balance()

    conn.execute(
        """
        CREATE TRIGGER fail_transfer_insert
        BEFORE INSERT ON transfers
        BEGIN
            SELECT RAISE(ABORT, 'forced failure');
        END;
        """
    )
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError):
        bank.transfer(contact_id, 10_000, "", "forced", NOW)

    assert bank.get_balance() == before
    assert conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0] == 0


# --------------------------------------------------------------- ownership


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


def _customer_snapshot(conn: sqlite3.Connection, customer_id: str) -> dict:
    tables = (
        "contacts",
        "accounts",
        "pots",
        "cards",
        "subscriptions",
        "transactions",
        "holdings",
        "transfers",
        "scheduled_transfers",
        "payment_requests",
        "merchant_orders",
    )
    snapshot = {}
    for table in tables:
        rows = conn.execute(
            f"SELECT * FROM {table} WHERE customer_id = ?", (customer_id,)
        ).fetchall()
        snapshot[table] = sorted(tuple(row) for row in rows)
    return snapshot


def test_customer_data_is_isolated(multi_env) -> None:
    bank_a, state_a, bank_b, state_b, conn = multi_env
    contact_a = state_a.contacts[0].id
    contact_b = state_b.contacts[0].id

    transfer_a = bank_a.transfer(contact_a, 5_000, "", "a-transfer", NOW)
    order_a = bank_a.merchant_order(
        "花店", "bouquet_s", TODAY + timedelta(days=1), NOW
    )
    transfer_b = bank_b.transfer(contact_b, 7_000, "", "b-transfer", NOW)
    order_b = bank_b.merchant_order(
        "蛋糕店", "cake_6in", TODAY + timedelta(days=1), NOW
    )

    card_a = bank_a.list_cards()[0]
    card_b = bank_b.list_cards()[0]
    sub_a = bank_a.list_subscriptions()[0]
    sub_b = bank_b.list_subscriptions()[0]

    snapshot_a = _customer_snapshot(conn, state_a.customer.id)
    snapshot_b = _customer_snapshot(conn, state_b.customer.id)

    # A cannot touch B's rows, and vice versa.
    for bank, contact, transfer, order, card, sub in (
        (bank_a, contact_b, transfer_b, order_b, card_b, sub_b),
        (bank_b, contact_a, transfer_a, order_a, card_a, sub_a),
    ):
        with pytest.raises(BankError) as exc:
            bank.get_contact_date(contact)
        assert exc.value.code == "not_found"
        with pytest.raises(BankError) as exc:
            bank.transfer(contact, 1_000, "", f"cross-{contact}", NOW)
        assert exc.value.code == "not_found"
        with pytest.raises(BankError) as exc:
            bank.cancel_transfer(transfer.id, NOW)
        assert exc.value.code == "not_found"
        with pytest.raises(BankError) as exc:
            bank.freeze_card(card.id)
        assert exc.value.code == "not_found"
        with pytest.raises(BankError) as exc:
            bank.cancel_order(order.id, NOW)
        assert exc.value.code == "not_found"
        with pytest.raises(BankError) as exc:
            bank.cancel_subscription(sub.id)
        assert exc.value.code == "not_found"

    assert _customer_snapshot(conn, state_a.customer.id) == snapshot_a
    assert _customer_snapshot(conn, state_b.customer.id) == snapshot_b

    assert (
        conn.execute(
            "SELECT status FROM transfers WHERE customer_id = ? AND id = ?",
            (state_b.customer.id, transfer_b.id),
        ).fetchone()[0]
        == "pending"
    )


def test_idempotency_key_is_scoped_per_customer(multi_env) -> None:
    bank_a, state_a, bank_b, state_b, conn = multi_env
    contact_a = state_a.contacts[0].id
    contact_b = state_b.contacts[0].id

    transfer_a = bank_a.transfer(contact_a, 1_000, "", "shared-key", NOW)
    transfer_b = bank_b.transfer(contact_b, 1_000, "", "shared-key", NOW)

    assert transfer_a.id != transfer_b.id
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM transfers WHERE idempotency_key = ?",
            ("shared-key",),
        ).fetchone()[0]
        == 2
    )

    assert (
        bank_a.transfer(contact_a, 1_000, "", "shared-key", NOW).id
        == transfer_a.id
    )
    assert (
        bank_b.transfer(contact_b, 1_000, "", "shared-key", NOW).id
        == transfer_b.id
    )


def test_reads_are_customer_scoped(multi_env) -> None:
    bank_a, state_a, bank_b, state_b, _ = multi_env
    contact_ids_a = {contact.id for contact in state_a.contacts}
    txn_ids_a = {txn.id for txn in state_a.transactions}

    contacts = bank_a.find_contact("")
    assert {contact.id for contact in contacts} == contact_ids_a

    transactions = bank_a.list_transactions(limit=100_000)
    assert {txn.id for txn in transactions} == txn_ids_a

    report = bank_a.spending_report(
        TODAY - timedelta(days=365), TODAY - timedelta(days=1)
    )
    manual = sum(
        -txn.amount_fen
        for txn in transactions
        if txn.amount_fen < 0
        and not (
            txn.category == "transfer" and txn.counterparty == "我的储蓄账户"
        )
    )
    assert report["total"] == manual
    assert report["total"] != bank_b.spending_report(
        TODAY - timedelta(days=365), TODAY - timedelta(days=1)
    )["total"]


def _run_deterministic_sequence(bank: Bank) -> list[str]:
    contact = bank.find_contact("张伟")[0].id
    ids: list[str] = []

    ids.append(bank.transfer(contact, 1_000, "a", "det-1", NOW).id)
    second = bank.transfer(contact, 2_000, "b", "det-2", NOW)
    ids.append(second.id)
    bank.cancel_transfer(second.id, NOW + timedelta(minutes=1))

    order = bank.merchant_order(
        "花店", "bouquet_m", TODAY + timedelta(days=1), NOW
    )
    ids.append(order.id)
    bank.cancel_order(order.id, NOW)

    card = bank.create_virtual_card(20_000)
    ids.append(card.id)
    bank.set_card_limit(card.id, 10_000)
    bank.freeze_card(card.id)
    bank.unfreeze_card(card.id)

    pot = bank.move_to_pot("deterministic pot", 1_000)
    ids.append(pot.id)
    bank.move_from_pot("deterministic pot", 1_000)

    scheduled = bank.schedule_transfer(contact, 1_000, 15, "scheduled")
    ids.append(scheduled.id)
    bank.cancel_scheduled(scheduled.id)

    request = bank.request_payment(contact, 1_000, "split")
    ids.append(request.id)
    bank.cancel_request(request.id)

    subscription = bank.list_subscriptions()[0]
    bank.cancel_subscription(subscription.id)
    bank.restore_subscription(subscription.id)

    third = bank.transfer(contact, 3_000, "c", "det-3", NOW)
    ids.append(third.id)
    ids.extend(transfer.id for transfer in bank.release_due(NOW + timedelta(days=1)))

    fourth = bank.transfer(contact, 4_000, "d", "det-4", NOW)
    ids.append(fourth.id)
    return ids


def test_deterministic_ids_and_final_rows() -> None:
    state = generate("young_pro", SEED, TODAY)
    results = []
    for _ in range(2):
        conn = connect()
        create_schema(conn)
        load_state(conn, state)
        bank = Bank(conn, state.customer.id)
        ids = _run_deterministic_sequence(bank)
        results.append((ids, _customer_snapshot(conn, state.customer.id)))
        conn.close()

    assert results[0][0] == results[1][0]
    assert results[0][1] == results[1][1]


def test_bank_rejects_unknown_customer(env) -> None:
    _, _, conn = env
    with pytest.raises(BankError) as exc:
        Bank(conn, "missing-customer")
    assert exc.value.code == "not_found"


def test_tz_aware_times_count_on_local_day(env) -> None:
    bank, _state, _conn = env
    """A 00:30+08:00 transfer belongs to the local day, not the previous UTC day."""
    from datetime import timezone

    cst = timezone(timedelta(hours=8))
    contact = bank.find_contact("张伟")[0].id
    bank.transfer(contact, 99_900, "a", "tz-1", datetime(2026, 10, 17, 0, 30, tzinfo=cst))
    bank.transfer(contact, 99_900, "b", "tz-2", datetime(2026, 10, 17, 9, 0, tzinfo=cst))
    assert bank.daily_outgoing_total(date(2026, 10, 17)) == 199_800


def test_scheduled_total_for_day(env) -> None:
    bank, _state, _conn = env
    contact = bank.find_contact("张伟")[0].id
    bank.schedule_transfer(contact, 99_900, 5, "x")
    bank.schedule_transfer(contact, 50_000, 5, "y")
    bank.schedule_transfer(contact, 10_000, 6, "z")
    assert bank.scheduled_total_for_day(5) == 149_900


def test_huge_amount_rejected_without_overflow(env) -> None:
    bank, _state, _conn = env
    contact = bank.find_contact("张伟")[0].id
    with pytest.raises(BankError) as err:
        bank.transfer(contact, 9 * 10**21, "x", "huge", datetime(2026, 10, 17, 12))
    assert err.value.code == "invalid_amount"
