"""Tests for the durable customer-scoped chat store."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

import pytest

from api.chat_store import MAX_TEXT_CHARS, ChatStore

NOW = datetime(2026, 10, 17, 14, 0, 0)


def _now(minutes: int = 0) -> datetime:
    return NOW + timedelta(minutes=minutes)


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    ChatStore(connection).create_schema()
    yield connection
    connection.close()


@pytest.fixture
def store(conn: sqlite3.Connection) -> ChatStore:
    return ChatStore(conn)


def _count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_customer_isolation_listing_and_search(store: ChatStore) -> None:
    alice = store.create("alice", _now(), "Alice budget")
    bob = store.create("bob", _now(), "Bob budget")
    store.append("alice", alice["id"], "user", "alice private note", _now(1))
    store.append("bob", bob["id"], "user", "bob private note", _now(2))

    assert set(alice) == {"id", "title", "created_at", "updated_at"}
    assert [item["id"] for item in store.list("alice")] == [alice["id"]]
    assert [item["id"] for item in store.list("bob")] == [bob["id"]]
    assert [item["id"] for item in store.list("alice", query="private")] == [
        alice["id"]
    ]
    assert store.list("bob", query="alice") == []
    assert store.list("alice", query="budget")  # title search
    assert [item["id"] for item in store.list("alice")] == [alice["id"]]

    with pytest.raises(ValueError, match="not_found"):
        store.require("bob", alice["id"])
    with pytest.raises(ValueError, match="not_found"):
        store.messages("bob", alice["id"])
    with pytest.raises(ValueError, match="not_found"):
        store.append("bob", alice["id"], "user", "intrusion", _now(3))
    with pytest.raises(ValueError, match="not_found"):
        store.memory_pairs("bob", alice["id"])

    store.append("alice", alice["id"], "assistant", "ok", _now(4))
    assert store.list("alice")[0]["id"] == alice["id"]


def test_raw_text_and_display_turn_persist_across_store_instances(
    store: ChatStore, conn: sqlite3.Connection
) -> None:
    chat = store.create("alice", _now(), "Legacy <chat>")
    text = "I paid 50% at <CAFE_1>\nThanks"
    turn = {"kind": "reply", "display": {"amount": "¥12.34", "memo": "line1\nline2 50%"}}
    store.append("alice", chat["id"], "user", text, _now(), turn=turn)

    reopened = ChatStore(conn)
    assert reopened.list("alice")[0]["title"] == "Legacy <chat>"
    page = reopened.messages("alice", chat["id"])
    assert page["messages"][0]["text"] == text
    assert page["messages"][0]["turn"] == turn

    reopened.append("alice", chat["id"], "assistant", "x" * MAX_TEXT_CHARS, _now(1))
    with pytest.raises(ValueError, match="text_too_long"):
        reopened.append(
            "alice", chat["id"], "assistant", "x" * (MAX_TEXT_CHARS + 1), _now(2)
        )
    assert len(reopened.messages("alice", chat["id"])["messages"]) == 2


def test_pagination_is_ordered_and_lossless(store: ChatStore) -> None:
    chat = store.create("alice", _now())
    ids = [
        store.append("alice", chat["id"], "user", f"message {index}", _now())["id"]
        for index in range(25)
    ]

    newest = store.messages("alice", chat["id"], limit=10)
    assert [item["id"] for item in newest["messages"]] == ids[15:]
    assert newest["has_more"] is True
    assert newest["oldest_id"] == ids[15]

    middle = store.messages(
        "alice", chat["id"], before=newest["oldest_id"], limit=10
    )
    assert [item["id"] for item in middle["messages"]] == ids[5:15]
    assert middle["has_more"] is True

    oldest = store.messages("alice", chat["id"], before=middle["oldest_id"], limit=10)
    assert [item["id"] for item in oldest["messages"]] == ids[:5]
    assert oldest["has_more"] is False
    assert oldest["oldest_id"] == ids[0]

    collected = (
        oldest["messages"] + middle["messages"] + newest["messages"]
    )
    assert [item["id"] for item in collected] == ids
    assert [item["id"] for item in store.messages("alice", chat["id"])["messages"]] == ids

    with pytest.raises(ValueError, match="not_found"):
        store.messages("alice", chat["id"], before="missing")


def test_memory_pairs_use_only_explicit_server_fields(store: ChatStore) -> None:
    first = store.create("alice", _now())
    second = store.create("alice", _now())
    store.append(
        "alice",
        first["id"],
        "user",
        "raw secret A",
        _now(1),
        turn={"otp": "111111"},
        masked_user="<PAYEE_1>",
    )
    store.append(
        "alice",
        first["id"],
        "assistant",
        "raw answer A",
        _now(2),
        turn={"display": {"otp": "222222"}},
        memory_summary="Moved <AMT_1>",
    )
    # Missing masked_user: this user/assistant pair must be skipped entirely.
    store.append("alice", first["id"], "user", "raw only", _now(3))
    store.append(
        "alice",
        first["id"],
        "assistant",
        "raw reply",
        _now(4),
        memory_summary="must not appear",
    )
    # A raw-text fallback must never produce a pair.
    store.append("alice", second["id"], "user", "raw B", _now(5))
    store.append(
        "alice", second["id"], "assistant", "answer B", _now(6),
        memory_summary="summary B",
    )

    assert store.memory_pairs("alice", first["id"]) == [
        {"user": "<PAYEE_1>", "assistant": "Moved <AMT_1>"}
    ]
    assert store.memory_pairs("alice", second["id"]) == []


def test_recent_user_texts_span_own_chats_but_not_other_customers(
    store: ChatStore,
) -> None:
    alice_first = store.create("alice", _now())
    alice_second = store.create("alice", _now())
    bob = store.create("bob", _now())
    store.append("alice", alice_first["id"], "user", "first raw", _now(1))
    store.append("alice", alice_second["id"], "user", "second raw", _now(2))
    store.append("alice", alice_first["id"], "assistant", "assistant only", _now(3))
    store.append("bob", bob["id"], "user", "bob raw", _now(4))

    assert store.recent_user_texts("alice") == ["first raw", "second raw"]
    assert store.recent_user_texts("alice", limit=1) == ["second raw"]
    assert store.recent_user_texts("bob") == ["bob raw"]


def test_otp_keys_are_stripped_recursively_from_stored_turn(
    store: ChatStore, conn: sqlite3.Connection
) -> None:
    chat = store.create("alice", _now())
    turn = {
        "display": {"amount": "¥1.00", "memo": "safe"},
        "otp": "111111",
        "demo_otp": "222222",
        "demo_co_otp": "333333",
        "co_otp": "444444",
        "challenge_code": "555555",
        "nested": [
            {"otp": "666666", "keep": "yes"},
            {"deep": {"challenge_code": "777777", "x": 1}},
        ],
    }
    message = store.append("alice", chat["id"], "user", "raw text", _now(), turn=turn)
    expected = {
        "display": {"amount": "¥1.00", "memo": "safe"},
        "nested": [{"keep": "yes"}, {"deep": {"x": 1}}],
    }
    assert message["turn"] == expected
    assert store.messages("alice", chat["id"])["messages"][0]["turn"] == expected

    raw = conn.execute(
        "SELECT turn_json FROM chat_messages WHERE id = ?", (message["id"],)
    ).fetchone()[0]
    for code in ("111111", "222222", "333333", "444444", "555555", "666666", "777777"):
        assert code not in raw
    assert json.loads(raw) == expected

    as_string = store.append(
        "alice", chat["id"], "assistant", "reply", _now(1),
        turn='{"display": {"otp": "888888"}, "keep": true}',
    )
    assert as_string["turn"] == {"display": {}, "keep": True}


def test_event_key_dedupes_retries_per_chat(store: ChatStore, conn: sqlite3.Connection) -> None:
    chat = store.create("alice", _now())
    other = store.create("alice", _now())
    first = store.append("alice", chat["id"], "user", "first", _now(), event_key="evt-1")
    retry = store.append("alice", chat["id"], "user", "changed", _now(1), event_key="evt-1")
    assert retry["id"] == first["id"]
    assert retry["text"] == "first"
    assert _count(conn, "chat_messages") == 1

    assert store.append(
        "alice", chat["id"], "assistant", "second", _now(2), event_key="evt-2"
    )["id"] != first["id"]
    assert store.append(
        "alice", other["id"], "user", "other chat", _now(3), event_key="evt-1"
    )["id"] != first["id"]

    one = store.append("alice", chat["id"], "user", "no key", _now(4))
    two = store.append("alice", chat["id"], "user", "no key", _now(5))
    assert one["id"] != two["id"]


def test_like_wildcards_and_sql_injection_are_literal(store: ChatStore) -> None:
    percent = store.create("alice", _now(), "100% done")
    underscore = store.create("alice", _now(), "100_done")
    plain = store.create("alice", _now(), "plain")
    store.append("alice", plain["id"], "user", "rate is 50% now", _now(1))
    slash = store.create("alice", _now(), r"a\b")
    injection = store.create("alice", _now(), "Robert'); DROP TABLE chatthreads;--")

    assert {item["id"] for item in store.list("alice", query="%")} == {
        plain["id"],
        percent["id"],
    }
    assert {item["id"] for item in store.list("alice", query="_")} == {underscore["id"]}
    assert [item["id"] for item in store.list("alice", query="\\")] == [slash["id"]]
    assert [item["id"] for item in store.list("alice", query="DROP TABLE")] == [
        injection["id"]
    ]
    assert store.list("alice", query="' OR '1'='1") == []
    assert store.require("alice", injection["id"])["id"] == injection["id"]


def test_supplied_chat_id_is_globally_unique_and_owner_checked(
    store: ChatStore,
) -> None:
    root = store.create("alice", _now(), "Root", chat_id="root-chat")
    assert root["id"] == "root-chat"
    assert store.create("alice", _now(1), "Ignored", chat_id="root-chat") == root

    with pytest.raises(ValueError, match="not_found"):
        store.create("bob", _now(), chat_id="root-chat")
    with pytest.raises(ValueError, match="not_found"):
        store.append("bob", "root-chat", "user", "x", _now())
    assert store.list("bob") == []

    generated = store.create("carol", _now())
    assert len(generated["id"]) == 32
    assert store.list("carol")[0]["id"] == generated["id"]


def test_append_does_not_commit_the_outer_transaction(
    store: ChatStore, conn: sqlite3.Connection
) -> None:
    chat = store.create("alice", _now())
    conn.commit()
    conn.execute("BEGIN")
    store.append("alice", chat["id"], "user", "pending", _now(1))
    assert conn.in_transaction is True
    conn.rollback()
    assert store.messages("alice", chat["id"])["messages"] == []
