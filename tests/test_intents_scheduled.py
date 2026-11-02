"""Tests for the scheduled-transfer and general reminder intent rules."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from agent import intents
from agent.intents import IntentContext
from agent.intents import reminders as reminders_rule
from agent.intents import scheduled as scheduled_rule
from agent.responder import _render_list_scheduled, _render_stop_standing_order
from bank.api import Bank
from bank.db import connect, create_schema, load_state
from bank.seed import generate
from security.ground import Facts, check
from security.tokens import Session, TokenVault, tokenize_user_text

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42
SECRET = b"anban-intent-scheduled-secret"


@pytest.fixture
def env():
    state = generate("young_pro", SEED, TODAY)
    conn = connect(":memory:")
    create_schema(conn)
    load_state(conn, state)
    bank = Bank(conn, state.customer.id)
    vault = TokenVault(SECRET)
    try:
        yield conn, state, bank, vault
    finally:
        conn.close()


@pytest.fixture
def student_env():
    state = generate("student", SEED, TODAY)
    conn = connect(":memory:")
    create_schema(conn)
    load_state(conn, state)
    bank = Bank(conn, state.customer.id)
    vault = TokenVault(SECRET)
    try:
        yield conn, state, bank, vault
    finally:
        conn.close()


def _session(state, session_id="session-1", plan_id="plan-1") -> Session:
    return Session(
        session_id=session_id,
        plan_id=plan_id,
        customer_id=state.customer.id,
    )


def _turn(
    env,
    raw: str,
    *,
    lang: str = "en",
    session: Session | None = None,
) -> tuple[IntentContext, Session]:
    conn, state, bank, vault = env
    if session is None:
        session = _session(state)
    tokenized, _ = tokenize_user_text(raw, bank, vault, session, NOW)
    ctx = IntentContext(
        raw_text=raw,
        tokenized_text=tokenized,
        conn=conn,
        secret_key=SECRET,
        customer_id=state.customer.id,
        session_id=session.session_id,
        now=NOW,
        lang=lang,
        vault=vault,
        plan_id=session.plan_id,
    )
    return ctx, session


def _contacts(bank: Bank) -> list:
    return bank.find_contact("")


def _schedule(bank: Bank, contact, *, amount_fen: int = 100_000, day: int = 5):
    return bank.schedule_transfer(contact.id, amount_fen, day, "rent")


# ------------------------------------------------------------------ listing


def test_chinese_scheduled_list_plans_list_scheduled(env) -> None:
    ctx, _ = _turn(env, "我的定时转账有哪些", lang="zh")

    output = scheduled_rule.resolve(ctx)

    assert output["kind"] == "plan"
    assert [step["tool"] for step in output["steps"]] == ["list_scheduled"]
    assert output["steps"][0]["args"] == {}


def test_english_standing_orders_list_plans_list_scheduled(env) -> None:
    ctx, _ = _turn(env, "show my standing orders")

    output = scheduled_rule.resolve(ctx)

    assert output["kind"] == "plan"
    assert [step["tool"] for step in output["steps"]] == ["list_scheduled"]


def test_full_rule_chain_lists_scheduled(env) -> None:
    ctx, _ = _turn(env, "我的定时转账有哪些", lang="zh")

    output = intents.resolve(ctx)

    assert output is not None and output["kind"] == "plan"
    assert [step["tool"] for step in output["steps"]] == ["list_scheduled"]


# ------------------------------------------------------------------- stopping


def test_cancel_with_one_order_mints_its_payee(env) -> None:
    _, _, bank, _ = env
    contact = _contacts(bank)[0]
    _schedule(bank, contact)
    ctx, session = _turn(env, "取消我的定时转账", lang="zh")

    output = scheduled_rule.resolve(ctx)

    assert output["kind"] == "plan"
    step = output["steps"][0]
    assert step["tool"] == "stop_standing_order"
    entry = ctx.vault.resolve(step["args"]["payee"], session)
    assert entry.kind == "PAYEE"
    assert entry.real_ref == contact.id


def test_cancel_with_two_orders_asks_which_one(env) -> None:
    _, _, bank, _ = env
    contacts = _contacts(bank)
    first, second = contacts[0], contacts[1]
    assert first.id != second.id
    _schedule(bank, first)
    _schedule(bank, second)
    ctx, session = _turn(env, "cancel my standing order")

    output = scheduled_rule.resolve(ctx)

    assert output["kind"] == "clarify"
    assert not any(char.isdigit() for char in output["question"])
    assert len(output["options"]) == 2
    resolved = {
        ctx.vault.resolve(label, session).real_ref for label in output["options"]
    }
    assert resolved == {first.id, second.id}


def test_cancel_with_no_order_clarifies_without_digits(env) -> None:
    ctx, _ = _turn(env, "cancel my standing order")

    output = scheduled_rule.resolve(ctx)

    assert output["kind"] == "clarify"
    assert output["options"] == []
    assert not any(char.isdigit() for char in output["question"])


def test_creating_an_order_is_left_to_the_planner(env) -> None:
    ctx, _ = _turn(env, "每个月转500给妈妈", lang="zh")

    assert scheduled_rule.resolve(ctx) is None


@pytest.mark.parametrize(
    "raw",
    [
        "Don't cancel my standing order to {payee}, I still need it",
        "Don't stop my standing order to {payee}, I still need it",
        "Don’t cancel my standing order to {payee}",
        "Don’t stop my standing order to {payee}",
        "Is my standing order to {payee} going to end next year?",
        "What happens if I stop my monthly transfer to {payee}?",
        "不要取消给{payee}的定时转账",
        "不要停掉给{payee}的定时转账",
        "不要停止给{payee}的定时转账",
        "别停给{payee}的定时转账",
        "不取消给{payee}的定时转账",
        "my standing order runs at the end of the month",
    ],
)
def test_negated_or_hypothetical_turns_only_list(env, raw: str) -> None:
    _, state, bank, vault = env
    contact = _contacts(bank)[0]
    _schedule(bank, contact)
    token = vault.mint(_session(state), "PAYEE", contact.id, {}, "user_text").label
    lang = "zh" if any("\u4e00" <= char <= "\u9fff" for char in raw) else "en"
    ctx, _ = _turn(env, raw.format(payee=token), lang=lang)

    output = scheduled_rule.resolve(ctx)

    assert output["kind"] == "plan"
    assert [step["tool"] for step in output["steps"]] == ["list_scheduled"]


@pytest.mark.parametrize(
    "raw",
    [
        "Stop my standing order to {payee} and send 200 to Bob",
        "Cancel my standing order to {payee} and buy a phone",
        "Stop my standing order to {payee} and freeze my card",
        "Cancel my standing order to {payee} and cancel my subscription",
        "停止给{payee}的定时转账，再给Bob转200",
    ],
)
def test_mixed_stop_and_write_is_left_to_planner(env, raw: str) -> None:
    _, state, bank, vault = env
    contact = _contacts(bank)[0]
    _schedule(bank, contact)
    token = vault.mint(_session(state), "PAYEE", contact.id, {}, "user_text").label
    lang = "zh" if any("\u4e00" <= char <= "\u9fff" for char in raw) else "en"
    ctx, _ = _turn(env, raw.format(payee=token), lang=lang)

    assert scheduled_rule.resolve(ctx) is None


@pytest.mark.parametrize(
    "raw, lang",
    [
        ("List my standing orders and send 200 to mum", "en"),
        ("列出我的定时转账，再给妈妈转200", "zh"),
        ("Show my standing orders and cancel my subscription", "en"),
        ("看看我的定时转账，再取消我的订阅", "zh"),
    ],
)
def test_mixed_list_and_write_is_left_to_planner(env, raw: str, lang: str) -> None:
    ctx, _ = _turn(env, raw, lang=lang)

    assert scheduled_rule.resolve(ctx) is None


def test_stop_with_multiple_explicit_payees_never_picks_first(env) -> None:
    _, state, bank, _ = env
    contacts = _contacts(bank)
    first, second = contacts[0], contacts[1]
    _schedule(bank, first)
    _schedule(bank, second)
    ctx, _ = _turn(env, f"stop my standing orders to {first.name} and {second.name}")

    assert scheduled_rule.resolve(ctx) is None


@pytest.mark.parametrize(
    "raw, lang",
    [
        ("取消我的定时转账", "zh"),
        ("stop my standing order to {payee}", "en"),
        ("给{payee}的定时转账不要了", "zh"),
    ],
)
def test_plain_stop_still_stops(env, raw: str, lang: str) -> None:
    _, state, bank, vault = env
    contact = _contacts(bank)[0]
    _schedule(bank, contact)
    token = vault.mint(_session(state), "PAYEE", contact.id, {}, "user_text").label
    ctx, _ = _turn(env, raw.format(payee=token), lang=lang)

    output = scheduled_rule.resolve(ctx)

    assert output["kind"] == "plan"
    assert [step["tool"] for step in output["steps"]] == ["stop_standing_order"]


# ------------------------------------------------------------------- reminders


def test_remind_on_a_weekday_plans_a_dated_reminder(env) -> None:
    ctx, session = _turn(env, "remind me to pay rent on friday")

    output = reminders_rule.resolve(ctx)

    assert output["kind"] == "plan"
    step = output["steps"][0]
    assert step["tool"] == "set_reminder"
    entry = ctx.vault.resolve(step["args"]["when"], session)
    assert entry.kind == "DATE"
    assert step["args"]["note"] == "pay rent"
    assert not any(char.isdigit() for char in step["args"]["note"])
    assert "<" not in step["args"]["note"] and ">" not in step["args"]["note"]


def test_bare_remind_asks_which_day(env) -> None:
    ctx, _ = _turn(env, "提醒我", lang="zh")

    output = reminders_rule.resolve(ctx)

    assert output["kind"] == "clarify"
    assert output["options"] == []
    assert not any(char.isdigit() for char in output["question"])


def test_subscription_reminder_stays_with_the_subscription_rule(student_env) -> None:
    ctx, _ = _turn(student_env, "remind me 2 days before 腾讯视频 renews")

    assert reminders_rule.resolve(ctx) is None
    output = intents.resolve(ctx)
    assert output is not None and output["kind"] == "plan"
    step = output["steps"][0]
    assert step["tool"] == "set_reminder"
    # The subscription rule owns this turn and uses its renewal-date shape.
    assert step["args"]["when"]["offset_days"] == -2
    assert step["args"]["note"] == "腾讯视频"


# ------------------------------------------------------------------- rendering


def _list_result() -> dict:
    return {
        "scheduled": [
            {
                "payee_display": "张伟",
                "amount_fen": 100_000,
                "day_of_month": 5,
                "status": "scheduled",
                "next_due": "2026-11-05",
                "last_run": {
                    "occurrence": "2026-10",
                    "status": "skipped",
                    "reason": "insufficient_funds",
                    "amount_fen": 100_000,
                },
            },
            {
                "payee_display": "李娜 Lina",
                "amount_fen": 5_000,
                "day_of_month": 1,
                "status": "cancelled",
                "next_due": None,
                "last_run": None,
            },
        ]
    }


def test_render_scheduled_list_chinese_grounds_its_figures() -> None:
    facts = Facts()

    text = "\n".join(_render_list_scheduled(_list_result(), "zh", facts))

    assert "· 每月5号 给 张伟 转 ¥1,000.00（下次 2026-11-05）" in text
    assert "上次（2026-10）：已跳过：余额不足" in text
    assert "李娜" not in text
    assert check(text, facts, NOW).ok


def test_render_scheduled_list_english_grounds_its_figures() -> None:
    facts = Facts()

    text = "\n".join(_render_list_scheduled(_list_result(), "en", facts))

    assert (
        "· On day 5 each month: ¥1,000.00 to 张伟 (next 2026-11-05)" in text
    )
    assert "last run (2026-10): skipped: insufficient balance" in text
    assert check(text, facts, NOW).ok


def test_render_empty_scheduled_list() -> None:
    assert _render_list_scheduled({"scheduled": []}, "zh", Facts()) == [
        "你目前没有定时转账。"
    ]
    assert _render_list_scheduled({"scheduled": []}, "en", Facts()) == [
        "You have no standing orders."
    ]


def test_render_stop_standing_order_in_both_languages() -> None:
    result = {"payee_display": "张伟", "amount_fen": 100_000}
    zh_facts = Facts()
    en_facts = Facts()

    zh = _render_stop_standing_order(result, "zh", zh_facts)
    en = _render_stop_standing_order(result, "en", en_facts)

    assert zh == ["已停止给 张伟 的定时转账（每月 ¥1,000.00）。"]
    assert en == ["Stopped the standing order to 张伟 (¥1,000.00/month)."]
    assert check(zh[0], zh_facts, NOW).ok
    assert check(en[0], en_facts, NOW).ok


def test_render_stop_standing_order_lists_every_stopped_item() -> None:
    items = [
        {"payee_display": "张伟", "amount_fen": 10_000, "day_of_month": 3},
        {"payee_display": "李娜", "amount_fen": 15_000, "day_of_month": 21},
    ]
    result = {"raw": {"stopped": items}, "stopped": items}
    zh_facts = Facts()
    en_facts = Facts()

    zh = _render_stop_standing_order(result, "zh", zh_facts)
    en = _render_stop_standing_order(result, "en", en_facts)

    assert zh == [
        "已停止给 张伟 的定时转账（每月 ¥100.00）。",
        "已停止给 李娜 的定时转账（每月 ¥150.00）。",
    ]
    assert en == [
        "Stopped the standing order to 张伟 (¥100.00/month).",
        "Stopped the standing order to 李娜 (¥150.00/month).",
    ]
    assert check("\n".join(zh), zh_facts, NOW).ok
    assert check("\n".join(en), en_facts, NOW).ok

    raw_only = {"raw": {"stopped": items}}
    assert _render_stop_standing_order(raw_only, "zh", Facts()) == zh
