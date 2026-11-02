"""Policy engine and cooling-window tests.

Everything here uses the real seeded bank, the real token vault/tokenizer and
the real DAG validator, so the policy is exercised on the same objects the
executor will see.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from agent.tools import REGISTRY
from bank import db
from bank.api import Bank
from bank.seed import generate
from security.cooling import cooling_for, describe_window
from security.dag import PlanError, PlanStep, validate_plan
from security.policy import (
    PlanDecision,
    PolicyConfig,
    Rule,
    StepDecision,
    evaluate,
    recheck_step,
)
from security.tokens import (
    Session,
    TokenVault,
    tokenize_untrusted,
    tokenize_user_text,
)

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42
SECRET = b"anban-policy-secret"

CONFIG = PolicyConfig()


# ----------------------------------------------------------------------- setup


def _make_bank(persona: str = "young_pro"):
    state = generate(persona, SEED, TODAY)
    conn = db.connect(":memory:")
    db.create_schema(conn)
    db.load_state(conn, state)
    return Bank(conn, state.customer.id), state


def _make_session(state) -> Session:
    return Session(customer_id=state.customer.id)


def _context(persona: str = "young_pro"):
    bank, state = _make_bank(persona)
    vault = TokenVault(SECRET)
    session = _make_session(state)
    return bank, state, vault, session


def _entries_of(entries, kind: str):
    return [entry for entry in entries if entry.kind == kind]


def _only(entries, kind: str):
    matches = _entries_of(entries, kind)
    assert len(matches) == 1, f"expected one {kind}, got {len(matches)}"
    return matches[0]


def _contact(state, name: str):
    return next(contact for contact in state.contacts if contact.name == name)


def _validate(raw_plan, vault: TokenVault, session: Session):
    return validate_plan(raw_plan, vault.known_tokens(session))


def _plan(*steps):
    return {"kind": "plan", "steps": list(steps)}


def _step(step_id, tool, args=None, after=None):
    return {
        "id": step_id,
        "tool": tool,
        "args": {} if args is None else args,
        "after": [] if after is None else after,
    }


def _decide(plan, bank, vault, session, **kwargs) -> PlanDecision:
    validated = _validate(plan, vault, session)
    return evaluate(
        validated,
        bank=bank,
        vault=vault,
        session=session,
        now=NOW,
        config=kwargs.pop("config", CONFIG),
        **kwargs,
    )


def _codes(decision: StepDecision) -> set[str]:
    return {reason.code for reason in decision.reasons}


def _assert_bilingual(decision: PlanDecision) -> None:
    reasons = list(decision.veto_reasons)
    for step in decision.steps:
        reasons.extend(step.reasons)
    assert reasons
    for reason in reasons:
        assert reason.code
        assert reason.message_en.strip()
        assert reason.message_zh.strip()


def _transfer_tokens(bank, vault, session, text: str):
    _, entries = tokenize_user_text(text, bank, vault, session, NOW)
    return _only(entries, "PAYEE"), _only(entries, "AMT")


# ------------------------------------------------------------------------ rules


def test_read_only_plan_is_green_without_mandates() -> None:
    bank, state, vault, session = _context()

    decision = _decide(
        _plan(
            _step("s1", "get_balance"),
            _step("s2", "list_transactions", {"period": "this_month"}),
            _step("s3", "list_cards"),
        ),
        bank,
        vault,
        session,
    )

    assert decision.vetoed is False
    assert decision.veto_reasons == []
    assert [step.tier for step in decision.steps] == ["green", "green", "green"]
    for step in decision.steps:
        assert step.needs_mandate is False
        assert step.needs_mfa is False
        assert step.needs_co_approval is False
        assert Rule.BASE in _codes(step)
    _assert_bilingual(decision)


def test_single_300_transfer_is_yellow_with_mandate() -> None:
    bank, state, vault, session = _context()
    payee, amount = _transfer_tokens(bank, vault, session, "transfer 300 to 王建国")

    decision = _decide(
        _plan(
            _step(
                "s1",
                "transfer",
                {"payee": payee.label, "amount": amount.label},
            )
        ),
        bank,
        vault,
        session,
    )

    step = decision.steps[0]
    assert decision.vetoed is False
    assert step.tier == "yellow"
    assert step.amount_fen == 30_000
    assert step.needs_mandate is True
    assert step.needs_mfa is False
    assert step.needs_co_approval is False
    assert step.deferred is False
    _assert_bilingual(decision)


def test_3500_rent_is_red_with_mfa() -> None:
    bank, state, vault, session = _context()
    payee, amount = _transfer_tokens(
        bank, vault, session, "pay 3500 rent to 王建国"
    )

    decision = _decide(
        _plan(
            _step(
                "s1",
                "transfer",
                {"payee": payee.label, "amount": amount.label},
            )
        ),
        bank,
        vault,
        session,
    )

    step = decision.steps[0]
    assert step.tier == "red"
    assert Rule.DAILY in _codes(step)
    assert step.needs_mandate is True
    assert step.needs_mfa is True
    assert decision.vetoed is False
    _assert_bilingual(decision)


def test_five_300_transfers_go_red_on_the_fourth_and_fifth() -> None:
    bank, state, vault, session = _context()
    payee, amount = _transfer_tokens(bank, vault, session, "transfer 300 to 王建国")

    steps = [
        _step(
            f"s{index}",
            "transfer",
            {"payee": payee.label, "amount": amount.label},
        )
        for index in range(1, 6)
    ]
    decision = _decide(_plan(*steps), bank, vault, session)

    assert [step.tier for step in decision.steps] == [
        "yellow",
        "yellow",
        "yellow",
        "red",
        "red",
    ]
    assert Rule.DAILY not in _codes(decision.steps[0])
    assert Rule.DAILY in _codes(decision.steps[3])
    assert Rule.DAILY in _codes(decision.steps[4])
    _assert_bilingual(decision)


def test_divide_by_is_applied_before_the_daily_total() -> None:
    bank, state, vault, session = _context()
    payee, amount = _transfer_tokens(
        bank, vault, session, "transfer 600 to 王建国"
    )

    decision = _decide(
        _plan(
            _step(
                "s1",
                "transfer",
                {"payee": payee.label, "amount": amount.label, "divide_by": 2},
            )
        ),
        bank,
        vault,
        session,
    )

    step = decision.steps[0]
    assert step.amount_fen == 30_000
    assert step.tier == "yellow"
    assert Rule.DAILY not in _codes(step)


def test_incoming_and_own_money_do_not_count_toward_the_daily_limit() -> None:
    bank, state, vault, session = _context()
    payee, amount = _transfer_tokens(
        bank, vault, session, "pay 3500 rent to 王建国"
    )

    decision = _decide(
        _plan(
            _step(
                "s1",
                "move_to_pot",
                {"pot_name": "travel", "amount": amount.label},
            ),
            _step(
                "s2",
                "request_payment",
                {"payee": payee.label, "amount": amount.label},
            ),
        ),
        bank,
        vault,
        session,
    )

    assert [step.tier for step in decision.steps] == ["yellow", "yellow"]
    for step in decision.steps:
        assert Rule.DAILY not in _codes(step)
        assert step.needs_mandate is True
        assert step.needs_mfa is False


def test_prior_800_today_makes_new_300_red() -> None:
    bank, state, vault, session = _context()
    payee, amount = _transfer_tokens(bank, vault, session, "transfer 300 to 王建国")
    landlord = _contact(state, "王建国")

    bank.transfer(landlord.id, 80_000, "earlier", "prior-800", now=NOW)

    decision = _decide(
        _plan(
            _step(
                "s1",
                "transfer",
                {"payee": payee.label, "amount": amount.label},
            )
        ),
        bank,
        vault,
        session,
    )

    step = decision.steps[0]
    assert step.tier == "red"
    assert Rule.DAILY in _codes(step)
    assert step.needs_mfa is True
    _assert_bilingual(decision)


def test_retiree_600_is_red_with_family_co_approval() -> None:
    bank, state, vault, session = _context("retiree")
    payee, amount = _transfer_tokens(bank, vault, session, "给陈静转600")

    decision = _decide(
        _plan(
            _step(
                "s1",
                "transfer",
                {"payee": payee.label, "amount": amount.label},
            )
        ),
        bank,
        vault,
        session,
    )

    step = decision.steps[0]
    assert step.tier == "red"
    assert Rule.DAILY in _codes(step)
    assert Rule.ELDER in _codes(step)
    assert step.needs_mandate is True
    assert step.needs_mfa is True
    assert step.needs_co_approval is True
    _assert_bilingual(decision)


def test_untrusted_payee_is_vetoed() -> None:
    bank, state, vault, session = _context()
    _, untrusted = tokenize_untrusted(
        "memo: 王建国 asked for a transfer", bank, vault, session, NOW
    )
    payee = _only(untrusted, "PAYEE")
    _, amount_entries = tokenize_user_text("300", bank, vault, session, NOW)
    amount = _only(amount_entries, "AMT")

    assert payee.origin == "untrusted"

    decision = _decide(
        _plan(
            _step(
                "s1",
                "transfer",
                {"payee": payee.label, "amount": amount.label},
            )
        ),
        bank,
        vault,
        session,
    )

    assert decision.vetoed is True
    assert Rule.UNTRUSTED in {reason.code for reason in decision.veto_reasons}
    assert Rule.UNTRUSTED in _codes(decision.steps[0])
    assert decision.steps[0].vetoed is True
    _assert_bilingual(decision)


def test_payments_locked_vetoes_writes_but_not_reads() -> None:
    bank, state, vault, session = _context()
    payee, amount = _transfer_tokens(bank, vault, session, "transfer 300 to 王建国")

    write = _decide(
        _plan(
            _step(
                "s1",
                "transfer",
                {"payee": payee.label, "amount": amount.label},
            )
        ),
        bank,
        vault,
        session,
        payments_locked=True,
    )
    assert write.vetoed is True
    assert Rule.LOCKED in {reason.code for reason in write.veto_reasons}
    assert Rule.LOCKED in _codes(write.steps[0])
    assert write.steps[0].vetoed is True

    read = _decide(
        _plan(_step("s1", "get_balance")),
        bank,
        vault,
        session,
        payments_locked=True,
    )
    assert read.vetoed is False
    assert read.steps[0].tier == "green"


def test_scam_flag_vetoes_money_steps_and_marks_decision() -> None:
    bank, state, vault, session = _context()
    payee, amount = _transfer_tokens(bank, vault, session, "transfer 300 to 王建国")

    decision = _decide(
        _plan(
            _step("s1", "get_balance"),
            _step(
                "s2",
                "transfer",
                {"payee": payee.label, "amount": amount.label},
            ),
        ),
        bank,
        vault,
        session,
        scam_flags=frozenset({"safe_account"}),
    )

    assert decision.scam is True
    assert decision.vetoed is True
    assert decision.steps[0].vetoed is False
    assert decision.steps[1].vetoed is True
    assert Rule.SCAM in _codes(decision.steps[1])
    _assert_bilingual(decision)


def test_new_payee_tag_is_red() -> None:
    bank, state, vault, session = _context("micro_business")
    _, entries = tokenize_user_text(
        "给鑫达贸易转300", bank, vault, session, NOW
    )
    payee = _only(entries, "PAYEE")
    amount = _only(entries, "AMT")

    assert payee.tags["trust"] == "new_payee"

    decision = _decide(
        _plan(
            _step(
                "s1",
                "transfer",
                {"payee": payee.label, "amount": amount.label},
            )
        ),
        bank,
        vault,
        session,
    )

    step = decision.steps[0]
    assert decision.vetoed is False
    assert step.tier == "red"
    assert Rule.NEW_PAYEE in _codes(step)
    assert step.needs_mandate is True
    assert step.needs_mfa is True
    _assert_bilingual(decision)


def test_flagged_payee_is_vetoed() -> None:
    bank, state, vault, session = _context()
    contact = _contact(state, "王建国")
    payee = vault.mint(
        session, "PAYEE", contact.id, {"trust": "flagged"}, "user_text"
    )
    amount = vault.mint(
        session, "AMT", 30_000, {"bucket": "100-500"}, "user_text"
    )

    decision = _decide(
        _plan(
            _step(
                "s1",
                "transfer",
                {"payee": payee.label, "amount": amount.label},
            )
        ),
        bank,
        vault,
        session,
    )

    assert decision.vetoed is True
    assert Rule.NEW_PAYEE in {reason.code for reason in decision.veto_reasons}
    assert Rule.NEW_PAYEE in _codes(decision.steps[0])
    assert decision.steps[0].vetoed is True
    _assert_bilingual(decision)


def test_deferred_amount_is_rechecked_before_execution() -> None:
    bank, state, vault, session = _context()
    payee = vault.mint(
        session,
        "PAYEE",
        _contact(state, "王建国").id,
        {"trust": "verified"},
        "user_text",
    )

    decision = _decide(
        _plan(
            _step("s1", "list_transactions", {"period": "this_month"}),
            _step(
                "s2",
                "transfer",
                {"payee": payee.label, "amount": "$s1.amount"},
                after=["s1"],
            ),
        ),
        bank,
        vault,
        session,
    )

    planned = decision.steps[1]
    assert planned.deferred is True
    assert planned.amount_fen is None
    assert planned.tier == "yellow"
    assert Rule.DEFERRED in _codes(planned)
    assert planned.needs_mandate is True

    rechecked = recheck_step(
        planned,
        resolved_amount_fen=200_000,
        bank=bank,
        now=NOW,
        config=CONFIG,
        approved_tier="yellow",
    )

    assert rechecked.deferred is False
    assert rechecked.amount_fen == 200_000
    assert rechecked.tier == "red"
    assert Rule.DAILY in _codes(rechecked)
    assert rechecked.requires_reapproval is True
    assert rechecked.needs_mfa is True


def test_recheck_escalates_after_another_transfer() -> None:
    bank, state, vault, session = _context()
    payee, amount = _transfer_tokens(bank, vault, session, "transfer 300 to 王建国")
    landlord = _contact(state, "王建国")

    decision = _decide(
        _plan(
            _step(
                "s1",
                "transfer",
                {"payee": payee.label, "amount": amount.label},
            )
        ),
        bank,
        vault,
        session,
    )
    planned = decision.steps[0]
    assert planned.tier == "yellow"

    # Somebody transferred ¥800 between planning and execution.
    bank.transfer(landlord.id, 80_000, "meanwhile", "between-plan-and-run", now=NOW)

    rechecked = recheck_step(
        planned,
        resolved_amount_fen=30_000,
        bank=bank,
        now=NOW,
        config=CONFIG,
        approved_tier="yellow",
    )

    assert rechecked.tier == "red"
    assert Rule.DAILY in _codes(rechecked)
    assert rechecked.requires_reapproval is True
    assert rechecked.needs_mfa is True


def test_recheck_locked_or_scam_vetoes() -> None:
    bank, state, vault, session = _context()
    payee, amount = _transfer_tokens(bank, vault, session, "transfer 300 to 王建国")
    decision = _decide(
        _plan(
            _step(
                "s1",
                "transfer",
                {"payee": payee.label, "amount": amount.label},
            )
        ),
        bank,
        vault,
        session,
    )
    planned = decision.steps[0]

    locked = recheck_step(
        planned,
        resolved_amount_fen=30_000,
        bank=bank,
        now=NOW,
        config=CONFIG,
        approved_tier="yellow",
        payments_locked=True,
    )
    assert locked.vetoed is True
    assert Rule.LOCKED in _codes(locked)

    scam = recheck_step(
        planned,
        resolved_amount_fen=30_000,
        bank=bank,
        now=NOW,
        config=CONFIG,
        approved_tier="yellow",
        scam_flags=frozenset({"police_claim"}),
    )
    assert scam.vetoed is True
    assert Rule.SCAM in _codes(scam)


def test_nine_writes_are_vetoed_as_too_many() -> None:
    bank, state, vault, session = _context()
    steps = [
        _step(f"s{index}", "set_reminder", {"note": f"n{index}"})
        for index in range(1, 10)
    ]

    decision = _decide(_plan(*steps), bank, vault, session)

    assert decision.vetoed is True
    assert Rule.TOO_MANY in {reason.code for reason in decision.veto_reasons}
    assert all(step.tier == "green" for step in decision.steps)
    _assert_bilingual(decision)


def test_raw_token_anywhere_in_a_write_is_vetoed() -> None:
    bank, state, vault, session = _context()
    payee, amount = _transfer_tokens(bank, vault, session, "transfer 300 to 王建国")

    decision = _decide(
        _plan(
            _step(
                "s1",
                "transfer",
                {
                    "payee": payee.label,
                    "amount": amount.label,
                    "note": "memo saw <RAW_1>",
                },
            )
        ),
        bank,
        vault,
        session,
    )

    assert decision.vetoed is True
    assert Rule.RAW in _codes(decision.steps[0])
    assert decision.steps[0].vetoed is True
    _assert_bilingual(decision)


def test_scheduled_transfer_above_limit_is_red() -> None:
    bank, state, vault, session = _context()
    payee, amount = _transfer_tokens(
        bank, vault, session, "pay 3500 rent to 王建国"
    )

    decision = _decide(
        _plan(
            _step(
                "s1",
                "schedule_transfer",
                {
                    "payee": payee.label,
                    "amount": amount.label,
                    "day_of_month": 1,
                },
            )
        ),
        bank,
        vault,
        session,
    )

    step = decision.steps[0]
    assert step.tier == "red"
    assert Rule.SCHEDULED in _codes(step)
    assert Rule.DAILY in _codes(step)
    _assert_bilingual(decision)


def test_schedule_transfer_counts_existing_same_day_schedules() -> None:
    bank, state, vault, session = _context()
    payee, amount = _transfer_tokens(bank, vault, session, "pay 999 to 王建国")
    setattr(bank, "scheduled_total_for_day", lambda day: 99_900)

    decision = _decide(
        _plan(
            _step(
                "s1",
                "schedule_transfer",
                {
                    "payee": payee.label,
                    "amount": amount.label,
                    "day_of_month": 5,
                },
            )
        ),
        bank,
        vault,
        session,
    )

    step = decision.steps[0]
    assert step.amount_fen == 99_900
    assert step.tier == "red"
    assert Rule.SCHEDULED in _codes(step)
    assert Rule.DAILY not in _codes(step)
    _assert_bilingual(decision)


def test_schedule_transfer_adds_earlier_same_day_plan_schedules() -> None:
    bank, state, vault, session = _context()
    payee, amount = _transfer_tokens(bank, vault, session, "pay 600 to 王建国")
    setattr(bank, "scheduled_total_for_day", lambda day: 0)
    schedule_args = {
        "payee": payee.label,
        "amount": amount.label,
        "day_of_month": 5,
    }

    decision = _decide(
        _plan(
            _step("s1", "schedule_transfer", schedule_args),
            _step("s2", "schedule_transfer", schedule_args, after=["s1"]),
        ),
        bank,
        vault,
        session,
    )

    assert decision.steps[0].tier == "yellow"
    assert Rule.SCHEDULED not in _codes(decision.steps[0])
    assert decision.steps[1].tier == "red"
    assert Rule.SCHEDULED in _codes(decision.steps[1])
    _assert_bilingual(decision)


def test_retiree_set_card_limit_is_red() -> None:
    bank, state, vault, session = _context("retiree")
    card = bank.list_cards()[0]
    card_token = vault.mint(
        session, "CARD", card.id, {"kind": card.kind}, "bank_context"
    )

    def decide(amount_fen: int) -> PlanDecision:
        amount = vault.mint(
            session, "AMT", amount_fen, {"bucket": "5000+"}, "user_text"
        )
        return _decide(
            _plan(
                _step(
                    "s1",
                    "set_card_limit",
                    {"card": card_token.label, "amount": amount.label},
                )
            ),
            bank,
            vault,
            session,
        )

    huge = decide(50_000_000)  # ¥500,000
    assert huge.steps[0].tier == "red"
    assert Rule.CARD_LIMIT in _codes(huge.steps[0])
    _assert_bilingual(huge)

    # ¥1,000 is under the general card cap but over the retiree daily limit.
    elder_level = decide(100_000)
    assert elder_level.steps[0].tier == "red"
    assert Rule.CARD_LIMIT in _codes(elder_level.steps[0])


def test_virtual_card_limit_thresholds() -> None:
    bank, state, vault, session = _context()

    def decide(amount_fen: int) -> PlanDecision:
        amount = vault.mint(
            session, "AMT", amount_fen, {"bucket": "1000-5000"}, "user_text"
        )
        return _decide(
            _plan(
                _step("s1", "create_virtual_card", {"amount": amount.label})
            ),
            bank,
            vault,
            session,
        )

    low = decide(100_000)  # ¥1,000
    assert low.steps[0].tier == "yellow"
    assert Rule.CARD_LIMIT not in _codes(low.steps[0])

    high = decide(500_000)  # ¥5,000
    assert high.steps[0].tier == "red"
    assert Rule.CARD_LIMIT in _codes(high.steps[0])
    _assert_bilingual(high)


def test_birthday_plan_tiers() -> None:
    bank, state, vault, session = _context()
    _, entries = tokenize_user_text(
        "lina's birthday is the 29th, put 1000 aside and get flowers and a cake",
        bank,
        vault,
        session,
        NOW,
    )
    payee = _only(entries, "PAYEE")
    amount = _only(entries, "AMT")
    merchants = {entry.real_ref: entry for entry in _entries_of(entries, "MERCHANT")}
    florist = merchants["花店"]
    cake_shop = merchants["蛋糕店"]

    decision = _decide(
        _plan(
            _step(
                "s1",
                "get_contact_date",
                {"payee": payee.label, "what": "birthday"},
            ),
            _step("s2", "can_cover", {"amount": amount.label}),
            _step(
                "s3",
                "move_to_pot",
                {"pot_name": "birthday", "amount": amount.label},
                after=["s2"],
            ),
            _step(
                "s4",
                "merchant_order",
                {
                    "merchant": florist.label,
                    "item": "bouquet_m",
                    "deliver": {"date": "$s1.date", "offset_days": -2},
                },
                after=["s1", "s3"],
            ),
            _step(
                "s5",
                "merchant_order",
                {
                    "merchant": cake_shop.label,
                    "item": "cake_8in",
                    "deliver": {"date": "$s1.date", "offset_days": -2},
                },
                after=["s1", "s3"],
            ),
        ),
        bank,
        vault,
        session,
    )

    assert [step.step_id for step in decision.steps] == [
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
    ]
    assert [step.tier for step in decision.steps] == [
        "green",
        "green",
        "yellow",
        "yellow",
        "yellow",
    ]
    assert decision.steps[3].amount_fen == 29_900
    assert decision.steps[4].amount_fen == 26_800
    assert decision.steps[2].needs_mandate is True
    assert decision.steps[3].needs_mandate is True
    assert decision.steps[4].needs_mandate is True
    assert decision.vetoed is False
    _assert_bilingual(decision)


def _birthday_order_plan(florist, cake_shop):
    return _plan(
        _step(
            "s1",
            "merchant_order",
            {
                "merchant": florist.label,
                "item": "bouquet_m",
                "deliver": {"date": "<DATE_1>", "offset_days": 0},
            },
        ),
        _step(
            "s2",
            "merchant_order",
            {
                "merchant": cake_shop.label,
                "item": "cake_8in",
                "deliver": {"date": "<DATE_1>", "offset_days": 0},
            },
        ),
    )


def test_birthday_orders_are_counted_in_the_daily_limit() -> None:
    bank, state, vault, session = _context()
    _, entries = tokenize_user_text(
        "get flowers and a cake", bank, vault, session, NOW
    )
    merchants = {entry.real_ref: entry for entry in _entries_of(entries, "MERCHANT")}
    florist = merchants["花店"]
    cake_shop = merchants["蛋糕店"]
    vault.mint(
        session,
        "DATE",
        date(2026, 10, 29),
        {"relative": "in 12 days"},
        "user_text",
    )

    decision = _decide(
        _birthday_order_plan(florist, cake_shop), bank, vault, session
    )

    assert [step.amount_fen for step in decision.steps] == [29_900, 26_800]
    assert [step.tier for step in decision.steps] == ["yellow", "yellow"]
    assert decision.vetoed is False


def test_birthday_orders_go_red_after_600_already_sent() -> None:
    bank, state, vault, session = _context()
    _, entries = tokenize_user_text(
        "get flowers and a cake", bank, vault, session, NOW
    )
    merchants = {entry.real_ref: entry for entry in _entries_of(entries, "MERCHANT")}
    florist = merchants["花店"]
    cake_shop = merchants["蛋糕店"]
    vault.mint(
        session,
        "DATE",
        date(2026, 10, 29),
        {"relative": "in 12 days"},
        "user_text",
    )
    landlord = _contact(state, "王建国")
    bank.transfer(landlord.id, 60_000, "earlier", "prior-600", now=NOW)

    decision = _decide(
        _birthday_order_plan(florist, cake_shop), bank, vault, session
    )

    assert [step.tier for step in decision.steps] == ["yellow", "red"]
    assert Rule.DAILY not in _codes(decision.steps[0])
    assert Rule.DAILY in _codes(decision.steps[1])
    assert decision.steps[1].needs_mfa is True


def test_item_not_sold_by_merchant_is_vetoed() -> None:
    bank, state, vault, session = _context()
    _, entries = tokenize_user_text(
        "get flowers and a cake", bank, vault, session, NOW
    )
    merchants = {entry.real_ref: entry for entry in _entries_of(entries, "MERCHANT")}
    florist = merchants["花店"]
    vault.mint(
        session,
        "DATE",
        date(2026, 10, 29),
        {"relative": "in 12 days"},
        "user_text",
    )

    decision = _decide(
        _plan(
            _step(
                "s1",
                "merchant_order",
                {
                    "merchant": florist.label,
                    "item": "cake_8in",
                    "deliver": {"date": "<DATE_1>", "offset_days": 0},
                },
            )
        ),
        bank,
        vault,
        session,
    )

    assert decision.vetoed is True
    assert Rule.BAD_ITEM in {reason.code for reason in decision.veto_reasons}
    assert Rule.BAD_ITEM in _codes(decision.steps[0])
    assert decision.steps[0].vetoed is True
    _assert_bilingual(decision)


def test_planner_has_no_tier_field_and_cannot_lower_a_tier() -> None:
    assert "tier" not in PlanStep.model_fields

    raw = _plan(
        _step("s1", "get_balance", {"tier": "green"}),
    )
    with pytest.raises(PlanError):
        validate_plan(raw, {})


def test_every_reason_has_bilingual_messages() -> None:
    # The registry-driven reason catalogue is what every generated Reason uses.
    from security.policy import _FLAGGED_PAYEE_MESSAGE, _MESSAGES

    for en, zh in _MESSAGES.values():
        assert en.strip()
        assert zh.strip()
        assert en != zh
    assert _FLAGGED_PAYEE_MESSAGE[0].strip()
    assert _FLAGGED_PAYEE_MESSAGE[1].strip()


# ----------------------------------------------------------------------- cooling


def test_cooling_periods() -> None:
    assert cooling_for("green") == timedelta(0)
    assert cooling_for("yellow") == timedelta(hours=2)
    assert cooling_for("red") == timedelta(hours=2)
    assert cooling_for("red", elder=True) == timedelta(hours=24)
    assert cooling_for("green", scam_flags=frozenset({"safe_account"})) == timedelta(
        hours=24
    )
    assert cooling_for("yellow", scam_flags=frozenset({"police_claim"})) == timedelta(
        hours=24
    )


def test_describe_window() -> None:
    created = datetime(2026, 10, 17, 10, 0, 0)
    release = datetime(2026, 10, 17, 12, 0, 0)

    pending = describe_window(created, release, datetime(2026, 10, 17, 11, 30, 0))
    assert pending == {
        "seconds_left": 1800,
        "can_undo": True,
        "release_at": "2026-10-17T12:00:00",
    }

    closed = describe_window(created, release, datetime(2026, 10, 17, 12, 30, 0))
    assert closed["seconds_left"] == 0
    assert closed["can_undo"] is False


def test_every_step_decision_has_a_tier_from_the_registry() -> None:
    bank, state, vault, session = _context()
    decision = _decide(
        _plan(
            _step("s1", "get_balance"),
            _step("s2", "set_reminder", {"note": "water plants"}),
        ),
        bank,
        vault,
        session,
    )

    for step in decision.steps:
        assert step.tier == REGISTRY[step.tool].base_tier


def test_split_rounding_matches_executor() -> None:
    """Policy and executor share bank.money.split_fen, so mandates bind the same fen."""
    from bank.money import split_fen

    assert split_fen(10001, 2) == 5001
    assert split_fen(42000, 3) == 14000
    assert split_fen(100, 3) == 33
