"""Bilingual replies for subscriptions, reminders and re-activation."""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from agent.planner import ScriptedPlanner
from agent.responder import (
    Facts,
    _render_reminders,
    _render_subscriptions,
    _render_write,
    compose_reply,
)
from api.app import create_app
from bank import db
from bank.api import Bank
from bank.seed import generate
from security.tokens import Session, TokenVault
from tests.test_api import NOW, SECRET, FixedClock, _auth, _login, _scripted


def _subscriptions_result() -> dict:
    return {
        "raw": [
            {
                "id": "sub_1",
                "merchant": "Netflix",
                "amount_fen": 6_800,
                "status": "active",
                "last_used_at": datetime(2026, 10, 1),
            },
            {
                "id": "sub_2",
                "merchant": "iCloud",
                "amount_fen": 600,
                "status": "active",
                "last_used_at": datetime(2026, 10, 2),
            },
        ],
        "recurring": [
            {
                "subscription_id": "sub_1",
                "merchant": "Netflix",
                "amount_fen": 6_800,
                "status": "active",
                "category": "video",
                "source": "both",
                "last_charged": "2026-10-14",
                "unused_days": 3,
            },
            {
                "subscription_id": "sub_2",
                "merchant": "iCloud",
                "amount_fen": 600,
                "status": "active",
                "category": "cloud",
                "source": "registered",
                "last_charged": None,
                "unused_days": 15,
            },
            {
                "subscription_id": None,
                "merchant": "Spotify",
                "amount_fen": 1_500,
                "status": "detected",
                "category": "music",
                "source": "detected",
                "last_charged": "2026-10-10",
                "unused_days": None,
            },
        ],
    }


def test_subscriptions_show_charge_state_and_detected_charges() -> None:
    result = _subscriptions_result()

    zh = "\n".join(_render_subscriptions(result, "zh", NOW, Facts()))
    en = "\n".join(_render_subscriptions(result, "en", NOW, Facts()))

    assert "（上次扣费 2026-10-14）" in zh
    assert "（最近180天无扣费记录）" in zh
    assert "另外从你的交易中发现的定期扣费：" in zh
    assert "Spotify" in zh and "¥15.00/月" in zh and "上次扣费 2026-10-10" in zh

    assert "(last charged 2026-10-14)" in en
    assert "(no charge in the last 180 days)" in en
    assert "Also found in your transactions (recurring charges):" in en
    assert "Spotify" in en and "¥15.00/month" in en
    assert "last charged 2026-10-10" in en

    for text in (zh, en):
        assert "Amazon" not in text
        assert "Netflix" in text and "iCloud" in text and "Spotify" in text


def test_subscriptions_without_recurring_data_stay_generic() -> None:
    result = {
        "raw": [
            {
                "id": "sub_9",
                "merchant": "Netflix",
                "amount_fen": 6_800,
                "status": "active",
                "last_used_at": datetime(2026, 10, 1),
            }
        ]
    }

    zh = "\n".join(_render_subscriptions(result, "zh", NOW, Facts()))
    en = "\n".join(_render_subscriptions(result, "en", NOW, Facts()))

    assert "上次扣费" not in zh and "无扣费记录" not in zh
    assert "last charged" not in en and "no charge" not in en
    assert "另外从你的交易中发现" not in zh
    assert "Also found in your transactions" not in en
    assert "iCloud" not in zh and "iCloud" not in en


def test_list_reminders_empty_and_with_due_items() -> None:
    empty_zh = _render_reminders(
        {"raw": [], "reminder": [], "due": []}, "zh", Facts(), NOW
    )
    empty_en = _render_reminders(
        {"raw": [], "reminder": [], "due": []}, "en", Facts(), NOW
    )
    assert "你还没有设置提醒。" in empty_zh
    assert "本演示不会推送通知。" in empty_zh
    assert "You have no reminders." in empty_en
    assert "This demo does not send notifications." in empty_en

    due = [
        {
            "due_on": "2026-10-17",
            "note": None,
            "created_at": "2026-10-01T09:00:00",
        }
    ]
    items = [
        *due,
        {
            "due_on": "2026-11-12",
            "note": "续费 Spotify",
            "created_at": "2026-10-02T09:00:00",
        },
    ]
    result = {"raw": items, "reminder": items, "due": due}

    zh = _render_reminders(result, "zh", Facts(), NOW)
    en = _render_reminders(result, "en", Facts(), NOW)

    assert zh[0] == "今天到期的提醒："
    assert "· 2026-10-17 提醒：续费提醒" in zh
    assert "· 2026-11-12 提醒：续费 Spotify" in zh
    assert zh[-1] == "本演示不会推送通知。"

    assert en[0] == "Due now:"
    assert "· 2026-10-17 Reminder: Renewal reminder" in en
    assert "· 2026-11-12 Reminder: 续费 Spotify" in en
    assert en[-1] == "This demo does not send notifications."


def test_list_reminders_reply_stays_grounded_with_a_due_item() -> None:
    state = generate("young_pro", 42, NOW.date())
    conn = db.connect(":memory:")
    db.create_schema(conn)
    db.load_state(conn, state)
    try:
        bank = Bank(conn, state.customer.id)
        vault = TokenVault(b"k" * 32)
        session = Session(customer_id=state.customer.id)
        items = [
            {"due_on": "2026-10-17", "note": None, "created_at": "2026-10-01"},
            {"due_on": "2026-11-12", "note": "renew", "created_at": "2026-10-02"},
        ]
        outcome = {
            "kind": "done",
            "steps": [
                {
                    "step_id": "s1",
                    "tool": "list_reminders",
                    "tier": "green",
                    "status": "done",
                    "result": {"raw": items, "reminder": items, "due": [items[0]]},
                    "error": None,
                }
            ],
        }

        zh = compose_reply(
            outcome, lang="zh", vault=vault, session=session, bank=bank, now=NOW
        )
        en = compose_reply(
            outcome, lang="en", vault=vault, session=session, bank=bank, now=NOW
        )

        assert zh.grounded, zh.rejected
        assert "今天到期的提醒：" in zh.text
        assert "本演示不会推送通知。" in zh.text
        assert en.grounded, en.rejected
        assert "Due now:" in en.text
        assert "This demo does not send notifications." in en.text
    finally:
        conn.close()


def test_set_reminder_reply_uses_the_date_when_present() -> None:
    result = {
        "raw": {
            "when": "2026-11-12",
            "note": None,
            "due_on": "2026-11-12",
            "saved": True,
            "notifications_enabled": False,
        }
    }

    zh = _render_write("set_reminder", result, "zh", None, None, NOW, Facts())[0]
    en = _render_write("set_reminder", result, "en", None, None, NOW, Facts())[0]

    assert zh == "提醒已保存：2026-11-12。本演示不会推送通知。"
    assert en == "Reminder saved for 2026-11-12. This demo does not send notifications."

    undated = {"raw": {"when": None, "note": "check bill", "due_on": None}}
    zh_plain = _render_write("set_reminder", undated, "zh", None, None, NOW, Facts())[0]
    en_plain = _render_write("set_reminder", undated, "en", None, None, NOW, Facts())[0]
    assert zh_plain == "提醒已保存。本演示不会推送通知。"
    assert en_plain == "Reminder saved. This demo does not send notifications."


def test_cancel_reply_reports_monthly_savings() -> None:
    result = {
        "raw": {"merchant": "Netflix", "amount_fen": 6_800, "status": "cancelled"},
        "saved_monthly_fen": 6_800,
    }

    zh = _render_write("cancel_subscription", result, "zh", None, None, NOW, Facts())[0]
    en = _render_write("cancel_subscription", result, "en", None, None, NOW, Facts())[0]

    assert zh == "已取消 Netflix 订阅。每月节省 ¥68.00。"
    assert en == "Cancelled the Netflix subscription. You save ¥68.00 per month."
    assert "iCloud" not in zh and "iCloud" not in en


def test_restore_and_reactivate_replies_name_the_merchant() -> None:
    raw = {"merchant": "Netflix", "amount_fen": 6_800, "status": "active"}
    result = {"raw": raw, "subscription": "sub_1"}

    restored_zh = _render_write(
        "restore_subscription", result, "zh", None, None, NOW, Facts()
    )[0]
    restored_en = _render_write(
        "restore_subscription", result, "en", None, None, NOW, Facts()
    )[0]
    assert restored_zh == "已恢复 Netflix 订阅（每月 ¥68.00）。"
    assert restored_en == "Restored Netflix (¥68.00/month)."

    reactivated_zh = _render_write(
        "reactivate_subscription", result, "zh", None, None, NOW, Facts()
    )[0]
    reactivated_en = _render_write(
        "reactivate_subscription", result, "en", None, None, NOW, Facts()
    )[0]
    assert reactivated_zh == "已重新开通 Netflix 订阅（每月 ¥68.00）。"
    assert reactivated_en == "Re-activated Netflix (¥68.00/month)."

    generic = {"raw": {}}
    assert _render_write(
        "restore_subscription", generic, "zh", None, None, NOW, Facts()
    )[0] == "已撤销。"
    assert _render_write(
        "restore_subscription", generic, "en", None, None, NOW, Facts()
    )[0] == "Undone."
    for text in (restored_zh, restored_en, reactivated_zh, reactivated_en):
        assert "iCloud" not in text


# --------------------------------------------------------------- API summary


@pytest.fixture
def api_client():
    app = create_app(
        secret_key=SECRET,
        planner=ScriptedPlanner(_scripted),
        clock=FixedClock(NOW),
        demo_mode=True,
    )
    with TestClient(app) as client:
        yield client, app
    app.state.runtime.conn.close()


def test_summary_reminders_missing_table_is_empty(api_client) -> None:
    client, _app = api_client
    token = _login(client)

    data = client.get("/me/summary", headers=_auth(token)).json()

    assert data["reminders"] == []


def test_summary_reminders_are_scoped_sorted_and_limited(api_client) -> None:
    client, app = api_client
    token = _login(client)
    customer_id = token.split(".")[0]
    conn = app.state.runtime.conn
    conn.execute(
        """CREATE TABLE agent_reminders (
            customer_id TEXT NOT NULL, idempotency_key TEXT NOT NULL,
            due_on TEXT, note TEXT, created_at TEXT NOT NULL,
            PRIMARY KEY(customer_id, idempotency_key))"""
    )
    rows = [
        (customer_id, f"k{index}", f"2026-11-{index:02d}", f"note {index}", NOW.isoformat())
        for index in range(12, 0, -1)
    ]
    rows.append((customer_id, "past", "2026-10-01", "too late", NOW.isoformat()))
    rows.append(("cust_someone_else", "other", "2026-11-01", "not mine", NOW.isoformat()))
    conn.executemany(
        """INSERT INTO agent_reminders
           (customer_id, idempotency_key, due_on, note, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()

    data = client.get("/me/summary", headers=_auth(token)).json()
    reminders = data["reminders"]

    assert len(reminders) == 10
    assert [item["due_on"] for item in reminders] == [
        f"2026-11-{index:02d}" for index in range(1, 11)
    ]
    assert reminders[0] == {"due_on": "2026-11-01", "note": "note 1"}
    assert all(set(item) == {"due_on", "note"} for item in reminders)
    assert all(item["note"] != "too late" for item in reminders)
    assert all(item["note"] != "not mine" for item in reminders)
