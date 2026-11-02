"""Tests for the code-side subscription intent rules."""

from __future__ import annotations

import json
from calendar import monthrange
from datetime import date, datetime, timedelta

import pytest

from agent.intents import IntentContext
from agent.intents.subscriptions import resolve
from bank.api import Bank
from bank.db import connect, create_schema, load_state
from bank.recurring import subscription_category
from bank.seed import generate
from security.dag import parse_planner_output, validate_plan
from security.tokens import Session, TokenVault, tokenize_user_text

TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, 0)
SEED = 42
SECRET = b"anban-intent-subscriptions-secret"


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
    tokenize: bool = True,
) -> tuple[IntentContext, Session]:
    conn, state, bank, vault = env
    if session is None:
        session = _session(state)
    tokenized = raw
    if tokenize:
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


def _active(bank: Bank):
    return [sub for sub in bank.list_subscriptions() if sub.status == "active"]


def _resolved_ids(ctx: IntentContext, session: Session, steps) -> set[str]:
    return {
        ctx.vault.resolve(step["args"]["sub"], session).real_ref for step in steps
    }


def _assert_plan(output) -> list[dict]:
    assert output["kind"] == "plan"
    parse_planner_output(json.dumps(output))
    return output["steps"]


# ------------------------------------------------------------------ R6 / R1


def test_cancel_netflix_clarifies_and_never_plans(env) -> None:
    ctx, _ = _turn(env, "cancel netflix")

    output = resolve(ctx)

    assert output is not None and output["kind"] == "clarify"
    assert "netflix" in output["question"].casefold()
    _, _, bank, vault = env
    # Options are SUB tokens (the validator only accepts tokens) naming the active subs.
    session = Session(session_id=ctx.session_id, plan_id=ctx.plan_id, customer_id=ctx.customer_id)
    assert {vault.resolve(label, session).real_ref for label in output["options"]} == {
        sub.id for sub in _active(bank)
    }
    from security.dag import validate_plan

    validate_plan(parse_planner_output(json.dumps(output)), vault.known_tokens(session))


def test_remind_before_netflix_renews_clarifies(env) -> None:
    ctx, _ = _turn(env, "remind me before netflix renews")

    output = resolve(ctx)

    assert output is not None and output["kind"] == "clarify"
    assert "netflix" in output["question"].casefold()


def test_remind_before_netflix_renews_ignores_day_words(env) -> None:
    ctx, _ = _turn(env, "remind me 2 days before netflix renews")

    output = resolve(ctx)

    assert output is not None and output["kind"] == "clarify"
    assert "netflix" in output["question"].casefold()
    assert "days" not in output["question"].casefold()


def test_list_reminders_english_and_chinese(env) -> None:
    ctx, _ = _turn(env, "what reminders do I have?")
    steps = _assert_plan(resolve(ctx))
    assert [step["tool"] for step in steps] == ["list_reminders"]

    ctx_zh, _ = _turn(env, "我设置了哪些提醒", lang="zh")
    steps_zh = _assert_plan(resolve(ctx_zh))
    assert [step["tool"] for step in steps_zh] == ["list_reminders"]


def test_chinese_automatic_charges_lists_subscriptions(env) -> None:
    ctx, _ = _turn(env, "那些自动扣费的", lang="zh")

    steps = _assert_plan(resolve(ctx))

    assert [step["tool"] for step in steps] == ["list_subscriptions"]


# ------------------------------------------------------------------ R3 bulk


def test_cancel_all_video_cancels_only_active_video_subs(env) -> None:
    ctx, session = _turn(env, "cancel all video ones")

    steps = _assert_plan(resolve(ctx))

    _, _, bank, _ = env
    expected = {
        sub.id
        for sub in _active(bank)
        if subscription_category(sub.merchant) == "video"
    }
    assert expected
    assert [step["tool"] for step in steps] == ["cancel_subscription"] * len(expected)
    assert _resolved_ids(ctx, session, steps) == expected


def test_chinese_cancel_all_has_one_step_per_active_sub(env) -> None:
    ctx, session = _turn(env, "退订所有订阅", lang="zh")

    steps = _assert_plan(resolve(ctx))

    _, _, bank, _ = env
    expected = {sub.id for sub in _active(bank)}
    assert [step["tool"] for step in steps] == ["cancel_subscription"] * len(expected)
    assert _resolved_ids(ctx, session, steps) == expected
    assert [step["id"] for step in steps] == [
        f"s{index}" for index in range(1, len(expected) + 1)
    ]


# --------------------------------------- R3 merchant names are not categories


def test_cancel_merchant_with_category_word_in_name_is_not_bulk(student_env) -> None:
    # "视频" inside "腾讯视频" must not be read as the video category.
    ctx, _ = _turn(student_env, "cancel 腾讯视频")

    assert resolve(ctx) is None


def test_cancel_merchant_already_cancelled_says_so(student_env) -> None:
    _, _, bank, _ = student_env
    target = next(
        sub for sub in bank.list_subscriptions() if sub.merchant == "腾讯视频"
    )
    bank.cancel_subscription(target.id)

    ctx, _ = _turn(student_env, "cancel 腾讯视频")
    output = resolve(ctx)

    assert output is not None and output["kind"] == "clarify"
    assert target.merchant in output["question"]
    assert output["options"] == []


def test_cancel_all_vid_subscriptions_cancels_only_video(student_env) -> None:
    ctx, session = _turn(student_env, "cancel all the vid subs lol")

    steps = _assert_plan(resolve(ctx))

    _, _, bank, _ = student_env
    expected = {
        sub.id
        for sub in _active(bank)
        if subscription_category(sub.merchant) == "video"
    }
    assert len(expected) == 2
    assert _resolved_ids(ctx, session, steps) == expected


def test_chinese_cancel_all_video_variant(student_env) -> None:
    ctx, session = _turn(student_env, "把视频会员都退了", lang="zh")

    steps = _assert_plan(resolve(ctx))

    _, _, bank, _ = student_env
    expected = {
        sub.id
        for sub in _active(bank)
        if subscription_category(sub.merchant) == "video"
    }
    assert len(expected) == 2
    assert _resolved_ids(ctx, session, steps) == expected


def test_chinese_never_used_variant_plans_cancel(student_env) -> None:
    ctx, session = _turn(student_env, "把我不用的退了", lang="zh")

    steps = _assert_plan(resolve(ctx))

    _, _, bank, _ = student_env
    forgotten = [
        sub for sub in _active(bank) if (NOW.date() - sub.last_used_at).days >= 60
    ]
    assert len(forgotten) == 1
    assert [step["tool"] for step in steps] == ["cancel_subscription"]
    assert _resolved_ids(ctx, session, steps) == {forgotten[0].id}


def test_typo_cancel_unknown_brand_never_cancels_another_merchant(student_env) -> None:
    ctx, _ = _turn(student_env, "cancle netfliks plz")

    output = resolve(ctx)

    assert output is None or output["kind"] == "clarify"


# --------------------------------------------------------------- R-REMIND


def _expected_renewal(day_of_month: int, days: int) -> date:
    def month_date(year: int, month: int) -> date:
        last = monthrange(year, month)[1]
        return date(year, month, min(day_of_month, last))

    def next_month(year: int, month: int) -> tuple[int, int]:
        return (year + 1, 1) if month == 12 else (year, month + 1)

    renewal = month_date(TODAY.year, TODAY.month)
    if renewal < TODAY:
        renewal = month_date(*next_month(TODAY.year, TODAY.month))
    if renewal - timedelta(days=days) < TODAY:
        renewal = month_date(*next_month(renewal.year, renewal.month))
    return renewal


@pytest.mark.parametrize(
    "raw,lang,merchant,days",
    [
        ("remind me 3 days before 腾讯视频 renews", "en", "腾讯视频", 3),
        ("提前3天提醒我腾讯视频续费", "zh", "腾讯视频", 3),
        ("提前三天提醒我夸克网盘续费", "zh", "夸克网盘", 3),
        ("rmind me 2 days b4 夸克网盘 renews pls", "en", "夸克网盘", 2),
    ],
)
def test_remind_before_a_named_subscription_renews(
    student_env, raw, lang, merchant, days
) -> None:
    ctx, session = _turn(student_env, raw, lang=lang)

    output = resolve(ctx)
    steps = _assert_plan(output)
    validate_plan(
        parse_planner_output(json.dumps(output)),
        ctx.vault.known_tokens(session),
    )

    assert len(steps) == 1
    step = steps[0]
    assert step["tool"] == "set_reminder"
    _, _, bank, _ = student_env
    sub = next(item for item in bank.list_subscriptions() if item.merchant == merchant)
    assert step["args"]["note"] == sub.merchant[:40]
    assert step["args"]["when"]["offset_days"] == -days
    entry = ctx.vault.resolve(step["args"]["when"]["date"], session)
    assert entry.kind == "DATE"
    assert entry.real_ref == _expected_renewal(sub.day_of_month, days)


# ------------------------------------------------------------- R4 never used


def test_cancel_the_one_never_used_plans_that_subscription(env) -> None:
    ctx, session = _turn(env, "cancel the one I never use")

    steps = _assert_plan(resolve(ctx))

    _, _, bank, _ = env
    forgotten = [
        sub
        for sub in _active(bank)
        if (NOW.date() - sub.last_used_at).days >= 60
    ]
    assert len(forgotten) == 1
    assert [step["tool"] for step in steps] == ["cancel_subscription"]
    assert _resolved_ids(ctx, session, steps) == {forgotten[0].id}


# --------------------------------------------------------------- R7 detection


def test_detect_recurring_charges_is_a_read_only_plan(env) -> None:
    ctx, _ = _turn(env, "any recurring charges that aren't in my subscription list?")

    steps = _assert_plan(resolve(ctx))

    assert [step["tool"] for step in steps] == ["list_subscriptions"]
    assert all(step["tool"] != "cancel_subscription" for step in steps)


def test_charged_for_a_merchant_lists_subscriptions(env) -> None:
    ctx, _ = _turn(env, "did I get charged for 夸克网盘 recently?")

    steps = _assert_plan(resolve(ctx))

    assert [step["tool"] for step in steps] == ["list_subscriptions"]


# ------------------------------------------------------------------- R2 / R5


def test_undo_that_cancel_reactivates_the_cancelled_sub(env) -> None:
    conn, _, bank, _ = env
    target = _active(bank)[0]
    bank.cancel_subscription(target.id)

    ctx, session = _turn(env, "undo that cancel")

    steps = _assert_plan(resolve(ctx))

    assert [step["tool"] for step in steps] == ["reactivate_subscription"]
    assert _resolved_ids(ctx, session, steps) == {target.id}


def test_cancel_with_a_cancelled_sub_token_says_already_cancelled(env) -> None:
    _, state, bank, vault = env
    target = bank.list_subscriptions()[0]
    bank.cancel_subscription(target.id)

    session = _session(state)
    label = vault.mint(
        session,
        "SUB",
        target.id,
        {"status": "cancelled"},
        "bank_context",
    ).label
    ctx, _ = _turn(env, f"cancel {label}", session=session, tokenize=False)

    output = resolve(ctx)

    assert output is not None and output["kind"] == "clarify"
    assert target.merchant in output["question"]
    assert output["options"] == []


# ------------------------------------------------------------------ unrelated


@pytest.mark.parametrize(
    "text",
    [
        "how much did I spend last month",
        "send 100 to mum",
        "hello",
    ],
)
def test_unrelated_texts_return_none(env, text) -> None:
    ctx, _ = _turn(env, text)

    assert resolve(ctx) is None


@pytest.mark.parametrize(
    "raw,lang",
    [
        ("cancel my transfer to Lina", "en"),
        ("取消转账，不用了", "zh"),
        ("remind me to pay rent on Friday", "en"),
        ("stop the payment please", "en"),
        ("undo that, cancel the transfer", "en"),
        ("撤销刚才的转账", "zh"),
        ("把我的卡冻结，不用了谢谢", "zh"),
        ("cancel my order", "en"),
    ],
)
def test_everyday_banking_text_is_left_to_other_rules(env, raw, lang) -> None:
    ctx, _ = _turn(env, raw, lang=lang)
    assert resolve(ctx) is None


def test_setting_a_reminder_with_any_is_not_a_listing(env) -> None:
    ctx, _ = _turn(env, "remind me before any renewal")
    output = resolve(ctx)
    assert output is None or all(
        step["tool"] != "list_reminders" for step in output.get("steps", [])
    )


def test_chinese_brand_the_customer_lacks_clarifies(env) -> None:
    ctx, _ = _turn(env, "取消奈飞会员", lang="zh")
    output = resolve(ctx)
    assert output is not None and output["kind"] == "clarify"
    assert "奈飞" in output["question"]


def test_model_guess_of_an_unnamed_subscription_becomes_a_question(env) -> None:
    from agent.intents.subscriptions import guard_model_plan

    conn, state, bank, vault = env
    ctx, session = _turn(env, "cancle netfliks plz")
    sub = next(s for s in bank.list_subscriptions() if s.status == "active")
    # The model picked a subscription from the bank context, not from the user's words.
    label = vault.mint(session, "SUB", sub.id, {}, "bank_context").label
    model = json.dumps({"kind": "plan", "steps": [
        {"id": "s1", "tool": "cancel_subscription", "args": {"sub": label}, "after": []}]})
    replaced = guard_model_plan(ctx, model)
    assert replaced is not None and json.loads(replaced)["kind"] == "clarify"


def test_model_plan_on_a_user_named_subscription_is_kept(env) -> None:
    from agent.intents.subscriptions import guard_model_plan

    conn, state, bank, vault = env
    sub = next(s for s in bank.list_subscriptions() if s.status == "active")
    ctx, session = _turn(env, f"cancel {sub.merchant}")
    label = next(e.label for e in vault.entries(session) if e.kind == "SUB" and e.real_ref == sub.id)
    model = json.dumps({"kind": "plan", "steps": [
        {"id": "s1", "tool": "cancel_subscription", "args": {"sub": label}, "after": []}]})
    assert guard_model_plan(ctx, model) is None


def test_guard_keeps_model_cancel_of_unused_subs(env) -> None:
    from agent.intents.subscriptions import guard_model_plan

    conn, state, bank, vault = env
    ctx, session = _turn(env, "把不用的会员都退了，再存300到储蓄", lang="zh")
    unused = [
        sub
        for sub in _active(bank)
        if (NOW.date() - sub.last_used_at).days >= 60
    ]
    assert unused
    steps: list[dict] = []
    for index, sub in enumerate(unused, start=1):
        label = vault.mint(
            session, "SUB", sub.id, {"status": sub.status}, "bank_context"
        ).label
        steps.append(
            {
                "id": f"s{index}",
                "tool": "cancel_subscription",
                "args": {"sub": label},
                "after": [],
            }
        )
    steps.append(
        {
            "id": f"s{len(steps) + 1}",
            "tool": "move_to_pot",
            "args": {"amount": "<AMT_1>", "pot": "<POT_1>"},
            "after": [],
        }
    )
    model = json.dumps({"kind": "plan", "steps": steps})

    assert guard_model_plan(ctx, model) is None


def test_guard_still_replaces_a_recently_used_sub_for_unused_request(env) -> None:
    from agent.intents.subscriptions import guard_model_plan

    conn, state, bank, vault = env
    ctx, session = _turn(env, "把不用的会员都退了", lang="zh")
    recent = [
        sub
        for sub in _active(bank)
        if (NOW.date() - sub.last_used_at).days < 60
    ]
    assert recent
    label = vault.mint(
        session, "SUB", recent[0].id, {"status": recent[0].status}, "bank_context"
    ).label
    model = json.dumps({"kind": "plan", "steps": [
        {"id": "s1", "tool": "cancel_subscription", "args": {"sub": label}, "after": []}]})

    replaced = guard_model_plan(ctx, model)

    assert replaced is not None and json.loads(replaced)["kind"] == "clarify"


def test_category_without_all_asks_which_one(env) -> None:
    conn, state, bank, vault = env
    video = [s for s in bank.list_subscriptions() if s.status == "active" and subscription_category(s.merchant) == "video"]
    ctx, _ = _turn(env, "cancle tencnt video")
    output = resolve(ctx)
    if len(video) > 1:
        assert output is not None and output["kind"] == "clarify" and len(output["options"]) == len(video)


# --- subscription rule regressions ---


@pytest.mark.parametrize(
    "raw,lang",
    [
        ("cancel all my subscriptions except iCloud", "en"),
        ("I don't want to cancel all my subscriptions, just QQ音乐", "en"),
        ("stop! do not cancel all my subscriptions", "en"),
        ("这些订阅我都不想取消", "zh"),
        ("取消订阅会不会都扣费？", "zh"),
        ("how do I cancel a subscription? do you charge every month?", "en"),
        ("cancel my transfers and all my subscriptions", "en"),
        ("cancel all my orders and subscriptions", "en"),
    ],
)
def test_no_bulk_cancel_on_negation_question_or_mixed_request(env, raw, lang) -> None:
    ctx, _ = _turn(env, raw, lang=lang)
    output = resolve(ctx)
    steps = (output or {}).get("steps") or []
    assert sum(1 for step in steps if step["tool"] == "cancel_subscription") <= 1


def test_unused_phrase_is_not_treated_as_negation(env) -> None:
    ctx, _ = _turn(env, "cancel the ones I don't use")
    output = resolve(ctx)
    assert output is not None and output["kind"] == "plan"
    assert [step["tool"] for step in output["steps"]] == ["cancel_subscription"]


def test_restore_never_targets_an_unnamed_subscription(env) -> None:
    conn, state, bank, vault = env
    subs = [s for s in bank.list_subscriptions() if s.status == "active"]
    bank.cancel_subscription(subs[0].id)
    other = subs[1].merchant
    for raw in (f"restore my {other} subscription", "restore my spotify membership"):
        ctx, session = _turn(env, raw, lang="zh" if "恢复" in raw else "en")
        output = resolve(ctx)
        if output is None:
            continue
        for step in output.get("steps") or []:
            assert step["tool"] != "reactivate_subscription", raw


def test_remind_to_transfer_is_left_to_the_planner(env) -> None:
    ctx, _ = _turn(env, "remind me to transfer 5000 to mom before qq音乐 renews")
    assert resolve(ctx) is None
