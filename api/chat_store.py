"""Durable, customer-scoped chat threads and append-only messages.

Display data only: every read binds ``customer_id`` and ``chat_id``, raw user
text exists solely for the code-only risk detector, and no model or balance is
consulted here.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from collections import deque
from datetime import date, datetime, time

MAX_TEXT_CHARS = 20_000
_FORBIDDEN_TURN_KEYS = frozenset(
    {"demo_otp", "demo_co_otp", "otp", "co_otp", "challenge_code"}
)

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS chat_threads ("
    " id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, title TEXT NOT NULL,"
    " created_at TEXT NOT NULL, updated_at TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS ix_chat_threads_customer"
    " ON chat_threads(customer_id, updated_at DESC)",
    "CREATE TABLE IF NOT EXISTS chat_messages ("
    " id TEXT PRIMARY KEY, chat_id TEXT NOT NULL REFERENCES chat_threads(id),"
    " customer_id TEXT NOT NULL,"
    " role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),"
    " text TEXT NOT NULL, turn_json TEXT,"
    " masked_user TEXT NOT NULL DEFAULT '',"
    " memory_summary TEXT NOT NULL DEFAULT '',"
    " event_key TEXT, created_at TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS ix_chat_messages_chat"
    " ON chat_messages(chat_id, created_at)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_chat_messages_event"
    " ON chat_messages(chat_id, event_key)",
)


def _as_ts(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return datetime.combine(value, time.min).isoformat()
    if isinstance(value, str) and value:
        return value
    raise TypeError("now must be a date, datetime or ISO string")


def _limit(value: int, default: int = 100) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return min(200, max(0, number))


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _scrub_turn(value: object) -> object:
    """Recursively drop verification-code keys from a server display payload."""
    if isinstance(value, dict):
        return {
            key: _scrub_turn(item)
            for key, item in value.items()
            if str(key).lower() not in _FORBIDDEN_TURN_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_scrub_turn(item) for item in value]
    return value


def _turn_payload(turn: object) -> tuple[object | None, str | None]:
    if turn is None:
        return None, None
    if isinstance(turn, str):
        try:
            turn = json.loads(turn)
        except ValueError:
            raise ValueError("turn") from None
    cleaned = _scrub_turn(turn)
    return cleaned, json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))


def _load_turn(raw: str | None) -> object | None:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _metadata(row: sqlite3.Row | tuple) -> dict[str, str]:
    return {
        "id": row[0],
        "title": row[1],
        "created_at": row[2],
        "updated_at": row[3],
    }


def _message(row: sqlite3.Row | tuple) -> dict[str, object]:
    return {
        "id": row[0],
        "role": row[1],
        "text": row[2],
        "turn": _load_turn(row[3]),
        "timestamp": row[4],
    }


def create_schema(conn: sqlite3.Connection) -> None:
    """Module-level entry point, matching the other store modules."""
    ChatStore(conn).create_schema()


class ChatStore:
    """SQLite-backed chats and messages, always scoped by customer and chat."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create_schema(self) -> None:
        """Create the append-only chat tables without dropping anything."""
        for statement in _SCHEMA:
            self.conn.execute(statement)

    def _lookup(self, customer_id: str, chat_id: str) -> dict[str, str] | None:
        row = self.conn.execute(
            "SELECT id, title, created_at, updated_at FROM chat_threads"
            " WHERE customer_id = ? AND id = ?",
            (customer_id, chat_id),
        ).fetchone()
        return _metadata(row) if row is not None else None

    def require(self, customer_id: str, chat_id: str) -> dict[str, str]:
        """Return metadata for an owned chat, or hide it as ``not_found``."""
        metadata = self._lookup(customer_id, chat_id)
        if metadata is None:
            raise ValueError("not_found")
        return metadata

    def create(
        self,
        customer_id: str,
        now: date | datetime | str,
        title: str = "New chat",
        chat_id: str | None = None,
    ) -> dict[str, str]:
        """Create a chat; a supplied id is the globally unique legacy binding.

        An id owned by another customer is reported as ``not_found``; an id
        already owned by this customer is returned unchanged (idempotent).
        """
        timestamp = _as_ts(now)
        if chat_id is not None:
            chat_id = str(chat_id)
            if not chat_id:
                raise ValueError("chat_id")
            row = self.conn.execute(
                "SELECT customer_id FROM chat_threads WHERE id = ?", (chat_id,)
            ).fetchone()
            if row is not None:
                if row[0] != customer_id:
                    raise ValueError("not_found")
                return self.require(customer_id, chat_id)
        else:
            chat_id = secrets.token_hex(16)

        title = str(title)
        self.conn.execute(
            "INSERT INTO chat_threads"
            " (id, customer_id, title, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (chat_id, customer_id, title, timestamp, timestamp),
        )
        return {
            "id": chat_id,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
        }

    def list(
        self, customer_id: str, query: str = "", limit: int = 100
    ) -> list[dict[str, str]]:
        """Return this customer's chats, most recently updated first."""
        limit = _limit(limit)
        if not limit:
            return []
        if query:
            pattern = "%" + _like_escape(str(query)) + "%"
            rows = self.conn.execute(
                "SELECT DISTINCT t.id, t.title, t.created_at, t.updated_at"
                " FROM chat_threads AS t"
                " LEFT JOIN chat_messages AS m"
                " ON m.chat_id = t.id AND m.customer_id = t.customer_id"
                " WHERE t.customer_id = ? AND (t.title LIKE ? ESCAPE '\\'"
                " OR m.text LIKE ? ESCAPE '\\')"
                " ORDER BY t.updated_at DESC, t.id DESC LIMIT ?",
                (customer_id, pattern, pattern, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id, title, created_at, updated_at FROM chat_threads"
                " WHERE customer_id = ?"
                " ORDER BY updated_at DESC, id DESC LIMIT ?",
                (customer_id, limit),
            ).fetchall()
        return [_metadata(row) for row in rows]

    def append(
        self,
        customer_id: str,
        chat_id: str,
        role: str,
        text: str,
        now: date | datetime | str,
        turn: object | None = None,
        masked_user: str = "",
        memory_summary: str = "",
        event_key: str | None = None,
    ) -> dict[str, object]:
        """Append one message without committing the caller's transaction."""
        self.require(customer_id, chat_id)
        if role not in ("user", "assistant"):
            raise ValueError("role")
        if not isinstance(text, str):
            raise TypeError("text must be str")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError("text_too_long")

        timestamp = _as_ts(now)
        if event_key is not None:
            event_key = str(event_key) or None
        if event_key is not None:
            row = self.conn.execute(
                "SELECT id, role, text, turn_json, created_at"
                " FROM chat_messages"
                " WHERE customer_id = ? AND chat_id = ? AND event_key = ?",
                (customer_id, chat_id, event_key),
            ).fetchone()
            if row is not None:
                return _message(row)

        display_turn, turn_json = _turn_payload(turn)
        message_id = secrets.token_hex(16)
        self.conn.execute(
            "INSERT INTO chat_messages"
            " (id, chat_id, customer_id, role, text, turn_json, masked_user,"
            " memory_summary, event_key, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                message_id,
                chat_id,
                customer_id,
                role,
                text,
                turn_json,
                str(masked_user or ""),
                str(memory_summary or ""),
                event_key,
                timestamp,
            ),
        )
        self.conn.execute(
            "UPDATE chat_threads SET updated_at = ?"
            " WHERE customer_id = ? AND id = ?",
            (timestamp, customer_id, chat_id),
        )
        return {
            "id": message_id,
            "role": role,
            "text": text,
            "turn": display_turn,
            "timestamp": timestamp,
        }

    def messages(
        self,
        customer_id: str,
        chat_id: str,
        before: str | None = None,
        limit: int = 100,
    ) -> dict[str, object]:
        """Return one chronological page of messages, oldest first."""
        self.require(customer_id, chat_id)
        limit = _limit(limit)
        if not limit:
            return {"messages": [], "has_more": False, "oldest_id": None}

        sql = (
            "SELECT id, role, text, turn_json, created_at FROM chat_messages"
            " WHERE customer_id = ? AND chat_id = ?"
        )
        params: list[object] = [customer_id, chat_id]
        if before is not None:
            reference = self.conn.execute(
                "SELECT rowid FROM chat_messages"
                " WHERE customer_id = ? AND chat_id = ? AND id = ?",
                (customer_id, chat_id, before),
            ).fetchone()
            if reference is None:
                raise ValueError("not_found")
            sql += " AND rowid < ?"
            params.append(reference[0])
        sql += " ORDER BY rowid DESC LIMIT ?"
        params.append(limit + 1)

        rows = self.conn.execute(sql, params).fetchall()
        has_more = len(rows) > limit
        page = [_message(row) for row in reversed(rows[:limit])]
        return {
            "messages": page,
            "has_more": has_more,
            "oldest_id": page[0]["id"] if page else None,
        }

    def memory_pairs(
        self, customer_id: str, chat_id: str, limit: int = 24
    ) -> list[dict[str, str]]:
        """Return paired server records for the bounded memory renderer.

        Only explicit ``masked_user`` and ``memory_summary`` fields are used;
        raw text, display ``turn`` and codes are never fallbacks, so a pair
        missing either side is skipped.
        """
        self.require(customer_id, chat_id)
        limit = _limit(limit)
        if not limit:
            return []

        rows = self.conn.execute(
            "SELECT role, masked_user, memory_summary FROM chat_messages"
            " WHERE customer_id = ? AND chat_id = ? ORDER BY rowid ASC",
            (customer_id, chat_id),
        ).fetchall()
        pairs: deque[dict[str, str]] = deque(maxlen=limit)
        pending: str | None = None
        for role, masked_user, summary in rows:
            if role == "user":
                pending = masked_user or None
            elif role == "assistant":
                if pending and summary:
                    pairs.append({"user": pending, "assistant": summary})
                pending = None
        return list(pairs)

    def recent_user_texts(self, customer_id: str, limit: int = 5) -> list[str]:
        """Return this customer's latest raw user texts, oldest first.

        This exists only for the code-only risk detector; never send it to a
        model.
        """
        limit = _limit(limit)
        if not limit:
            return []
        rows = self.conn.execute(
            "SELECT text FROM (SELECT text, rowid AS rid FROM chat_messages"
            " WHERE customer_id = ? AND role = 'user'"
            " ORDER BY rowid DESC LIMIT ?) ORDER BY rid ASC",
            (customer_id, limit),
        ).fetchall()
        return [row[0] for row in rows]
