"""Chinese-regulation hooks: customer settings, fraud list, regulatory reports.

Real rules implemented here (see docs/COMPLIANCE.md):

* PBOC 银发〔2016〕261号: the customer may choose delayed arrival for transfers
  (``arrival_delay``), which is also the window in which a transfer can be undone.
* 反电信网络诈骗法 (2022): a scam freeze is reported to the national anti-fraud
  centre (SIMULATED here) and transfers to accounts on the national fraud list
  are refused (the list itself is SIMULATED: ``compliance/fraud_accounts.json``).
* 《金融机构大额交易和可疑交易报告管理办法》: a natural person's domestic
  transfers of ¥500,000 or more in one day produce a large-transaction report;
  scam-flagged attempts produce a suspicious-transaction report.  Both go to a
  SIMULATED regulator queue and are never mentioned to the customer (no
  tipping off).

Every report row carries ``simulated = 1``; nothing here talks to a real
government system.
"""

from __future__ import annotations

import json
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

FRAUD_LIST_PATH = Path(__file__).resolve().parents[1] / "compliance" / "fraud_accounts.json"

#: 大额交易: natural person, domestic transfers, one day, inclusive.
LARGE_TRANSFER_FEN = 50_000_000

#: The 2 h default is the shortest undo window AnBan allows; 24 h is the
#: delayed arrival a customer may choose (PBOC 261).
ARRIVAL_DELAYS: dict[str, timedelta] = {
    "2h": timedelta(hours=2),
    "24h": timedelta(hours=24),
}
LANGS = frozenset({"en", "zh"})
TEXT_SIZES = frozenset({"standard", "large", "xlarge"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS customer_settings (
    customer_id TEXT PRIMARY KEY,
    lang TEXT NOT NULL DEFAULT 'en',
    text_size TEXT NOT NULL DEFAULT 'standard',
    elder_mode INTEGER NOT NULL DEFAULT 0,
    arrival_delay TEXT NOT NULL DEFAULT '2h',
    guardian TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS compliance_reports (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    day TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    simulated INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS risk_profiles (
    customer_id TEXT PRIMARY KEY,
    level INTEGER NOT NULL,
    assessed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings_lang (
    customer_id TEXT PRIMARY KEY,
    lang TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings_pending (
    customer_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    PRIMARY KEY (customer_id, key)
);
"""

# Turning a protection down (24 h arrival -> 2 h, elder mode off, removing or
# changing the guardian) waits this long, so a scammer coaching the customer
# can't switch it off and get paid in the same call.  Turning a
# protection up applies at once.
LOOSEN_DELAY = timedelta(hours=24)


@dataclass(frozen=True, slots=True)
class CustomerSettings:
    lang: str = "en"
    text_size: str = "standard"
    elder_mode: bool = False
    arrival_delay: str = "2h"
    guardian: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "lang": self.lang,
            "text_size": self.text_size,
            "elder_mode": self.elder_mode,
            "arrival_delay": self.arrival_delay,
            "guardian": self.guardian,
        }


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


# ------------------------------------------------------------------ settings


def saved_lang(conn: sqlite3.Connection, customer_id: str) -> str | None:
    """The language the customer chose in Settings, or None if never saved."""

    ensure_schema(conn)
    row = conn.execute(
        "SELECT lang FROM settings_lang WHERE customer_id = ?", (customer_id,)
    ).fetchone()
    return str(row["lang"]) if row is not None and row["lang"] in {"en", "zh"} else None


def get_settings(conn: sqlite3.Connection, customer_id: str) -> CustomerSettings:
    ensure_schema(conn)
    return read_settings(conn, customer_id)


def read_settings(conn: sqlite3.Connection, customer_id: str) -> CustomerSettings:
    """Read existing protection settings without DDL or transaction commits.

    Payment gates call this after acquiring their write lock. Schema setup
    belongs outside that lock; a query must never commit the caller's work.
    Connection proxies used by atomic payment flows are supported as well.
    """
    if conn is None or not callable(getattr(conn, "execute", None)) or not _has_table(conn, "customer_settings"):
        return CustomerSettings()
    row = conn.execute(
        "SELECT * FROM customer_settings WHERE customer_id = ?", (customer_id,)
    ).fetchone()
    if row is None:
        return CustomerSettings()
    return CustomerSettings(
        lang=row["lang"],
        text_size=row["text_size"],
        elder_mode=bool(row["elder_mode"]),
        arrival_delay=row["arrival_delay"],
        guardian=row["guardian"],
    )


def _loosens(key: str, old: Any, new: Any) -> bool:
    if key == "arrival_delay":
        return ARRIVAL_DELAYS[new] < ARRIVAL_DELAYS[old]
    if key == "elder_mode":
        return bool(old) and not new
    if key == "guardian":
        return bool(old) and new != old
    return False


def apply_due(conn: sqlite3.Connection, customer_id: str, now: datetime) -> None:
    """Apply loosening changes whose waiting time is over."""

    ensure_schema(conn)
    rows = conn.execute(
        "SELECT key, value FROM settings_pending WHERE customer_id = ? AND effective_at <= ?",
        (customer_id, now.isoformat()),
    ).fetchall()
    if not rows:
        return
    changes = {row["key"]: json.loads(row["value"]) for row in rows}
    with conn:
        conn.execute(
            "DELETE FROM settings_pending WHERE customer_id = ? AND effective_at <= ?",
            (customer_id, now.isoformat()),
        )
    _store(conn, customer_id, {**get_settings(conn, customer_id).as_dict(), **changes})


def pending_changes(conn: sqlite3.Connection, customer_id: str) -> list[dict[str, Any]]:
    ensure_schema(conn)
    rows = conn.execute(
        "SELECT key, value, effective_at FROM settings_pending WHERE customer_id = ?"
        " ORDER BY key",
        (customer_id,),
    ).fetchall()
    return [
        {"key": row["key"], "value": json.loads(row["value"]), "effective_at": row["effective_at"]}
        for row in rows
    ]


def update_settings(
    conn: sqlite3.Connection,
    customer_id: str,
    changes: dict[str, Any],
    now: datetime | None = None,
) -> CustomerSettings:
    """Validate and store a partial settings update.

    Loosening a protection is queued for :data:`LOOSEN_DELAY` and needs ``now``;
    everything else applies at once and cancels a queued loosening of that key.
    """

    if now is not None:
        apply_due(conn, customer_id, now)
    stored = get_settings(conn, customer_id).as_dict()
    current = dict(stored)
    for key, value in changes.items():
        if key == "lang" and value in LANGS:
            current["lang"] = value
        elif key == "text_size" and value in TEXT_SIZES:
            current["text_size"] = value
        elif key == "elder_mode" and isinstance(value, bool):
            current["elder_mode"] = value
        elif key == "arrival_delay" and value in ARRIVAL_DELAYS:
            current["arrival_delay"] = value
        elif key == "guardian" and isinstance(value, str) and len(value) <= 40:
            current["guardian"] = value
        else:
            raise ValueError(f"invalid setting {key!r}")
    queued: dict[str, Any] = {}
    for key in ("arrival_delay", "elder_mode", "guardian"):
        if key in changes and _loosens(key, stored[key], current[key]):
            queued[key] = current[key]
            current[key] = stored[key]
    if queued and now is None:
        raise ValueError("now is required to lower a protection")
    waiting = {item["key"]: item["value"] for item in pending_changes(conn, customer_id)}
    with conn:
        for key in changes:
            if key in queued and key in waiting and waiting[key] == queued[key]:
                # Asking again for the same change keeps the original clock.
                del queued[key]
                continue
            conn.execute(
                "DELETE FROM settings_pending WHERE customer_id = ? AND key = ?",
                (customer_id, key),
            )
        for key, value in queued.items():
            conn.execute(
                """
                INSERT INTO settings_pending
                    (customer_id, key, value, requested_at, effective_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    customer_id,
                    key,
                    json.dumps(value),
                    now.isoformat(),  # type: ignore[union-attr]
                    (now + LOOSEN_DELAY).isoformat(),  # type: ignore[operator]
                ),
            )
    _store(conn, customer_id, current)
    if "lang" in changes:
        # Remember that the customer chose it; the stored default doesn't count.
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO settings_lang (customer_id, lang) VALUES (?, ?)",
                (customer_id, current["lang"]),
            )
    return get_settings(conn, customer_id)


def _store(conn: sqlite3.Connection, customer_id: str, current: dict[str, Any]) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO customer_settings
                (customer_id, lang, text_size, elder_mode, arrival_delay, guardian)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(customer_id) DO UPDATE SET
                lang = excluded.lang, text_size = excluded.text_size,
                elder_mode = excluded.elder_mode,
                arrival_delay = excluded.arrival_delay, guardian = excluded.guardian
            """,
            (
                customer_id,
                current["lang"],
                current["text_size"],
                int(current["elder_mode"]),
                current["arrival_delay"],
                current["guardian"],
            ),
        )


def arrival_delay(conn: sqlite3.Connection, customer_id: str) -> timedelta:
    return ARRIVAL_DELAYS[get_settings(conn, customer_id).arrival_delay]


def elder_mode(conn: Any, customer_id: str) -> bool:
    return read_settings(conn, customer_id).elder_mode


# ---------------------------------------------------------------- fraud list


def normalize_account(value: str) -> str:
    """Digits only after NFKC: spaced, dashed or full-width numbers match (C5)."""

    return "".join(ch for ch in unicodedata.normalize("NFKC", str(value)) if ch.isdigit())


def fraud_accounts() -> dict[str, str]:
    """SIMULATED national fraud-account list: account number -> reason.

    Read from disk on every call: a listing added after a transfer was approved
    must still hold the release (no permanent cache).
    """

    data = json.loads(FRAUD_LIST_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("accounts"), list):
        raise ValueError("invalid fraud list")
    accounts: dict[str, str] = {}
    for item in data["accounts"]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("account_no"), str)
            or not normalize_account(item["account_no"])
            or not isinstance(item.get("reason"), str)
        ):
            raise ValueError("invalid fraud list entry")
        accounts[normalize_account(item["account_no"])] = item["reason"]
    return accounts


def contact_on_fraud_list(conn: Any, customer_id: str, contact_id: str) -> bool:
    if not isinstance(conn, sqlite3.Connection):
        return False
    row = conn.execute(
        "SELECT account_no FROM contacts WHERE customer_id = ? AND id = ?",
        (customer_id, contact_id),
    ).fetchone()
    return row is not None and normalize_account(row["account_no"]) in fraud_accounts()


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def transfer_release_hold(
    conn: sqlite3.Connection,
    customer_id: str,
    contact_id: str,
    now: datetime,
) -> tuple[str, datetime | None] | None:
    """Whether a pending transfer release must be held, with the hold expiry.

    Read-only: it never creates or writes security tables, so bank-only fixtures
    without a security schema simply report no freeze/lock.  ``now`` is the
    caller's clock; this helper never reads the wall clock.
    """

    timed_reason: str | None = None
    timed_until: datetime | None = None

    if _has_table(conn, "security_scam_freezes"):
        row = conn.execute(
            "SELECT frozen_until FROM security_scam_freezes WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
        if row is not None:
            until = datetime.fromisoformat(row["frozen_until"])
            if now < until:
                timed_reason, timed_until = "R_SCAM", until

    if _has_table(conn, "security_lockouts"):
        row = conn.execute(
            "SELECT locked_until FROM security_lockouts WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
        if row is not None:
            until = datetime.fromisoformat(row["locked_until"])
            if now < until:
                if timed_until is None or until > timed_until:
                    timed_until = until
                if timed_reason is None:
                    timed_reason = "R_LOCKED"

    row = conn.execute(
        "SELECT account_no FROM contacts WHERE customer_id = ? AND id = ?",
        (customer_id, contact_id),
    ).fetchone()
    if row is None:
        return ("R_RELEASE_CHECK_UNAVAILABLE", None)
    account_no = normalize_account(row["account_no"])
    if not account_no:
        return ("R_RELEASE_CHECK_UNAVAILABLE", None)

    try:
        listed = fraud_accounts()
        if not isinstance(listed, dict):
            raise TypeError("fraud list must be a mapping")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        # Missing/unreadable/malformed list: refuse to release, reveal nothing.
        return ("R_RELEASE_CHECK_UNAVAILABLE", None)

    if account_no in listed:
        return ("R_FRAUD_LIST", None)
    if timed_reason is not None:
        return (timed_reason, timed_until)
    return None


# ------------------------------------------------------------------- reports


def _record(
    conn: sqlite3.Connection,
    customer_id: str,
    kind: str,
    payload: dict[str, Any],
    now: datetime,
) -> None:
    ensure_schema(conn)
    with conn:
        conn.execute(
            """
            INSERT INTO compliance_reports (customer_id, kind, created_at, day, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (customer_id, kind, now.isoformat(), now.date().isoformat(), json.dumps(payload)),
        )


def report_scam_freeze(
    conn: sqlite3.Connection, customer_id: str, flags: list[str], now: datetime
) -> None:
    """反诈: report the freeze (SIMULATED 国家反诈中心) and file an STR."""

    _record(conn, customer_id, "anti_fraud_center_report", {"flags": sorted(flags)}, now)
    _record(conn, customer_id, "suspicious_transaction_report", {"reason": "scam_script", "flags": sorted(flags)}, now)


def report_fraud_list_hit(
    conn: sqlite3.Connection, customer_id: str, contact_id: str, now: datetime
) -> None:
    _record(conn, customer_id, "fraud_list_hit", {"contact_id": contact_id}, now)


def check_large_transactions(
    conn: sqlite3.Connection, customer_id: str, outgoing_today_fen: int, now: datetime
) -> bool:
    """File one large-transaction report per day once ¥500,000 is reached."""

    if outgoing_today_fen < LARGE_TRANSFER_FEN:
        return False
    ensure_schema(conn)
    already = conn.execute(
        "SELECT 1 FROM compliance_reports WHERE customer_id = ? AND kind = ? AND day = ?",
        (customer_id, "large_transaction_report", now.date().isoformat()),
    ).fetchone()
    if already:
        return False
    _record(
        conn,
        customer_id,
        "large_transaction_report",
        {"outgoing_today_fen": int(outgoing_today_fen), "threshold_fen": LARGE_TRANSFER_FEN},
        now,
    )
    return True


def reports(conn: sqlite3.Connection, customer_id: str | None = None) -> list[dict[str, Any]]:
    ensure_schema(conn)
    sql = "SELECT * FROM compliance_reports"
    params: tuple[Any, ...] = ()
    if customer_id is not None:
        sql += " WHERE customer_id = ?"
        params = (customer_id,)
    rows = conn.execute(sql + " ORDER BY seq DESC LIMIT 100", params).fetchall()
    return [
        {
            "kind": row["kind"],
            "created_at": row["created_at"],
            "payload": json.loads(row["payload_json"]),
            "simulated": bool(row["simulated"]),
        }
        for row in rows
    ]



# ------------------------------------------------------------- suitability

# 适当性管理: products R1-R5, customers C1-C5 from a questionnaire the customer
# answers themselves; a Cn customer may buy R1..Rn only, and the result is
# valid for one year (see knowledge/articles.json "suitability").
RISK_TEST_VALID = timedelta(days=365)

# Demo questionnaire results: (level, days since the customer took it).  The
# student never took one; micro_business's result has expired.
_DEMO_RISK_PROFILES: dict[str, tuple[int, int]] = {
    "young_pro": (4, 120),
    "gig_worker": (2, 60),
    "retiree": (2, 200),
    "young_parent": (3, 90),
    "micro_business": (3, 400),
    "expat": (4, 45),
}


def risk_profile(
    conn: Any, customer_id: str, persona_id: str | None, now: datetime
) -> tuple[int, datetime] | None:
    """The customer's (level, assessed_at), or None if never assessed."""

    if not isinstance(conn, sqlite3.Connection):
        return None
    ensure_schema(conn)
    row = conn.execute(
        "SELECT level, assessed_at FROM risk_profiles WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()
    if row is None and persona_id in _DEMO_RISK_PROFILES:
        level, days_ago = _DEMO_RISK_PROFILES[persona_id]
        assessed = now - timedelta(days=days_ago)
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO risk_profiles (customer_id, level, assessed_at)"
                " VALUES (?, ?, ?)",
                (customer_id, level, assessed.isoformat()),
            )
        return level, assessed
    if row is None:
        return None
    return int(row["level"]), datetime.fromisoformat(row["assessed_at"])
