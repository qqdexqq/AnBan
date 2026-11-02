"""Tests for the deterministic mock-bank seed generator."""

from datetime import date, time, timedelta
from pathlib import Path

import yaml

from bank.categories import CATEGORIES
from bank.money import fmt_fen, yuan
from bank.seed import INJECTION_MEMO, generate, generate_all

TODAY = date(2026, 10, 17)
SEED = 42

SPEC_PATH = Path(__file__).resolve().parents[1] / "data_spec" / "customer_segments.yaml"


def _segments() -> dict[str, dict]:
    with SPEC_PATH.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return {segment["id"]: segment for segment in data["segments"]}


def _current(state):
    return next(account for account in state.accounts if account.type == "current")


def test_money_helpers() -> None:
    assert yuan("12.5") == 1250
    assert yuan(12) == 1200
    assert yuan("0.01") == 1
    assert fmt_fen(123450) == "¥1,234.50"
    assert fmt_fen(-8600) == "-¥86.00"


def test_same_args_are_identical() -> None:
    assert generate("young_pro", SEED, TODAY) == generate("young_pro", SEED, TODAY)


def test_different_seed_differs() -> None:
    assert generate("young_pro", SEED, TODAY) != generate("young_pro", SEED + 1, TODAY)


def test_generate_all_covers_every_persona() -> None:
    states = generate_all(SEED, TODAY)
    assert set(states) == set(_segments())


def test_ledgers_and_categories_for_every_persona() -> None:
    segments = _segments()
    for persona_id, state in generate_all(SEED, TODAY).items():
        current = _current(state)
        assert current.balance_fen == sum(
            txn.amount_fen
            for txn in state.transactions
            if txn.account_id == current.id
        )

        low, high = segments[persona_id]["finance"]["balance_range"]
        assert yuan(low) <= current.balance_fen <= yuan(high)

        assert state.transactions
        assert all(txn.category in CATEGORIES for txn in state.transactions)
        assert all(type(txn.amount_fen) is int for txn in state.transactions)
        assert all(type(account.balance_fen) is int for account in state.accounts)
        assert all(type(sub.amount_fen) is int for sub in state.subscriptions)


def test_opening_and_running_balances_for_every_persona() -> None:
    segments = _segments()
    for persona_id, state in generate_all(SEED, TODAY).items():
        low, high = segments[persona_id]["finance"]["balance_range"]
        opening = next(
            txn for txn in state.transactions if txn.memo == "Opening balance"
        )
        assert opening.category == "income"
        assert 0 < opening.amount_fen <= yuan(high)
        assert yuan(low) <= opening.amount_fen <= yuan(high)

        running = 0
        for txn in sorted(state.transactions, key=lambda item: item.ts):
            running += txn.amount_fen
            assert running >= 0, (persona_id, txn.id, running)

        savings = next(
            account for account in state.accounts if account.type == "savings"
        )
        assert savings.balance_fen >= 0


def test_student_opening_is_below_five_thousand() -> None:
    state = generate("student", SEED, TODAY)
    opening = next(txn for txn in state.transactions if txn.memo == "Opening balance")
    assert opening.amount_fen < yuan(5000)


def test_final_balance_is_inside_middle_sixty_percent() -> None:
    segments = _segments()
    for persona_id, state in generate_all(SEED, TODAY).items():
        low, high = segments[persona_id]["finance"]["balance_range"]
        low_fen, high_fen = yuan(low), yuan(high)
        margin = (high_fen - low_fen) * 20 // 100
        current = _current(state)
        assert low_fen + margin < current.balance_fen < high_fen - margin


def test_subscriptions_have_exactly_one_forgotten_sub() -> None:
    for state in generate_all(SEED, TODAY).values():
        assert state.subscriptions
        forgotten = [
            sub
            for sub in state.subscriptions
            if (TODAY - sub.last_used_at).days >= 60
        ]
        assert len(forgotten) == 1
        assert all(
            (TODAY - sub.last_used_at).days < 60
            for sub in state.subscriptions
            if sub is not forgotten[0]
        )


def test_anomalies_present_for_every_persona() -> None:
    for state in generate_all(SEED, TODAY).values():
        txns = state.transactions

        duplicate = any(
            first.counterparty == second.counterparty
            and first.amount_fen == second.amount_fen
            and first.amount_fen < 0
            and first.ts.date() == second.ts.date()
            and first.ts.time() > time(23, 0)
            and second.ts.time() > time(23, 0)
            and abs((first.ts - second.ts).total_seconds()) <= 5 * 60
            for first in txns
            for second in txns
            if first is not second
        )
        assert duplicate, state.customer.persona_id

        foreign_cnp = any(
            txn.country != "CN"
            and txn.card_present is False
            and txn.amount_fen == -128800
            for txn in txns
        )
        assert foreign_cnp, state.customer.persona_id


def test_young_pro_keeps_duplicate_contact_names() -> None:
    state = generate("young_pro", SEED, TODAY)
    zhang_weis = [contact for contact in state.contacts if contact.name == "张伟"]
    assert len(zhang_weis) == 2
    assert zhang_weis[0].account_no != zhang_weis[1].account_no
    assert any(contact.note == "badminton club" for contact in zhang_weis)


def test_lina_birthday_is_twelve_days_away() -> None:
    state = generate("young_pro", SEED, TODAY)
    lina = next(contact for contact in state.contacts if contact.name == "李娜 Lina")
    expected = TODAY + timedelta(days=12)
    assert lina.birthday is not None
    assert (lina.birthday.month, lina.birthday.day) == (expected.month, expected.day)
    assert lina.birthday.year < TODAY.year


def test_micro_business_injection_memo() -> None:
    state = generate("micro_business", SEED, TODAY)
    injected = [
        txn
        for txn in state.transactions
        if txn.counterparty == "鑫达贸易" and txn.memo == INJECTION_MEMO
    ]
    assert len(injected) == 1
    assert injected[0].amount_fen == 1
