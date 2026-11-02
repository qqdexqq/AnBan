"""Hash-chained audit log tests: tamper evidence, PII guard and explanations."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timedelta

import pytest

from bank.db import connect
from security.audit import (
    AuditError,
    AuditKind,
    AuditLog,
    _READ_TOOLS,
    _TOOL_WORDS,
    create_audit_schema,
    verify,
)

SECRET = b"anban-audit-secret"
BASE = datetime(2026, 10, 17, 13, 38, 2)


@pytest.fixture
def env():
    conn = connect()
    create_audit_schema(conn)
    log = AuditLog(conn, SECRET)
    try:
        yield log, conn
    finally:
        conn.close()


def _append(log, i=0, customer_id="cust_a", plan_id="plan_1", kind="RUN", payload=None):
    return log.append(
        customer_id=customer_id,
        session_id="sess_1",
        plan_id=plan_id,
        kind=kind,
        payload=payload if payload is not None else {"step_id": f"s{i}"},
        now=BASE + timedelta(seconds=i),
    )


def _drop_triggers(conn):
    conn.execute("DROP TRIGGER security_audit_no_update")
    conn.execute("DROP TRIGGER security_audit_no_delete")
    conn.commit()


# ---------------------------------------------------------------------- chain


def test_chain_of_twenty_verifies(env) -> None:
    log, conn = env
    entries = [_append(log, i) for i in range(20)]

    assert [e.seq for e in entries] == list(range(1, 21))
    assert entries[0].prev_hash == "0" * 64
    assert entries[1].prev_hash == entries[0].hash
    assert all(
        e.hash
        == hashlib.sha256(
            (
                e.prev_hash
                + json.dumps(
                    {
                        "customer_id": e.customer_id,
                        "session_id": e.session_id,
                        "plan_id": e.plan_id,
                        "ts": e.ts,
                        "kind": e.kind,
                        "payload": e.payload,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            ).encode("utf-8")
        ).hexdigest()
        for e in entries
    )

    result = verify(conn)
    assert result.ok is True
    assert result.entries == 20
    assert result.first_bad_seq is None
    assert result.reason is None
    assert log.verify().ok is True


def test_empty_log_verifies(env) -> None:
    log, conn = env
    assert verify(conn).ok is True
    assert verify(conn).entries == 0
    assert log.verify().ok is True


def test_anchor_changes_as_head_advances(env) -> None:
    log, _ = env
    first = _append(log, 0)
    second = _append(log, 1)
    anchor = log.anchor(BASE + timedelta(seconds=2))

    assert len(anchor) == 64
    assert first.hash != second.hash
    assert log.verify().ok is True


def test_external_anchor_verifies_clean_chain(env, tmp_path) -> None:
    _, conn = env
    anchor_path = tmp_path / "audit.anchor"
    log = AuditLog(conn, SECRET, anchor_path=anchor_path)
    entries = [_append(log, i) for i in range(3)]

    stored = json.loads(anchor_path.read_text(encoding="utf-8"))
    assert stored["seq"] == entries[-1].seq
    assert stored["hash"] == entries[-1].hash
    assert len(stored["hmac"]) == 64

    assert log.verify().ok is True
    assert verify(conn, SECRET, anchor_path=anchor_path).ok is True


# --------------------------------------------------------------------- tamper


def test_tampered_payload_reports_first_bad_seq(env) -> None:
    log, conn = env
    for i in range(5):
        _append(log, i)

    _drop_triggers(conn)
    conn.execute(
        "UPDATE security_audit SET payload_json = ? WHERE seq = 3",
        ('{"step_id":"forged"}',),
    )
    conn.commit()

    result = verify(conn)
    assert result.ok is False
    assert result.entries == 5
    assert result.first_bad_seq == 3
    assert result.reason == "hash mismatch"

    result = log.verify()
    assert result.ok is False
    assert result.first_bad_seq == 3


def test_edited_hash_is_detected(env) -> None:
    log, conn = env
    for i in range(4):
        _append(log, i)

    _drop_triggers(conn)
    conn.execute(
        "UPDATE security_audit SET hash = ? WHERE seq = 2", ("f" * 64,)
    )
    conn.commit()

    result = verify(conn)
    assert result.ok is False
    assert result.first_bad_seq == 2
    assert result.reason == "hash mismatch"


def test_deleted_middle_row_breaks_chain(env) -> None:
    log, conn = env
    for i in range(20):
        _append(log, i)

    _drop_triggers(conn)
    conn.execute("DELETE FROM security_audit WHERE seq = 10")
    conn.commit()

    result = verify(conn)
    assert result.ok is False
    assert result.entries == 19
    assert result.first_bad_seq == 11
    assert result.reason in {"sequence gap", "prev_hash break"}


def test_deleted_tail_is_caught_by_anchor(env) -> None:
    log, conn = env
    for i in range(20):
        _append(log, i)

    _drop_triggers(conn)
    conn.execute("DELETE FROM security_audit WHERE seq > 18")
    conn.commit()

    result = verify(conn)
    assert result.ok is False
    assert result.entries == 18
    assert result.first_bad_seq == 19
    assert result.reason == "anchor mismatch"

    # The HMAC check (used through the service) catches it as well.
    assert log.verify().ok is False


def test_rollback_attack_is_caught_by_external_anchor(env, tmp_path) -> None:
    _, conn = env
    anchor_path = tmp_path / "audit.anchor"
    log = AuditLog(conn, SECRET, anchor_path=anchor_path)
    for i in range(2):
        _append(log, i)

    assert json.loads(anchor_path.read_text(encoding="utf-8"))["seq"] == 2
    db_anchor = conn.execute(
        "SELECT last_seq, last_hash, anchor, ts"
        " FROM security_audit_anchor WHERE id = 1"
    ).fetchone()
    db_anchor_values = tuple(db_anchor)

    for i in range(2, 4):
        _append(log, i)
    assert log.verify().ok is True

    # Attacker with DB access deletes the two newest rows and restores the old
    # in-database anchor row.  The external file still points at seq 4.
    _drop_triggers(conn)
    conn.execute("DELETE FROM security_audit WHERE seq > 2")
    conn.execute(
        "UPDATE security_audit_anchor"
        " SET last_seq = ?, last_hash = ?, anchor = ?, ts = ?"
        " WHERE id = 1",
        db_anchor_values,
    )
    conn.commit()

    result = log.verify()
    assert result.ok is False
    assert result.entries == 2
    assert result.reason == "truncated (external anchor ahead)"

    direct = verify(conn, SECRET, anchor_path=anchor_path)
    assert direct.ok is False
    assert direct.reason == "truncated (external anchor ahead)"


def test_reordered_rows_are_detected(env) -> None:
    log, conn = env
    for i in range(4):
        _append(log, i)

    _drop_triggers(conn)
    # Swap the seq of rows 2 and 3; the prev_hash links no longer line up.
    conn.execute("UPDATE security_audit SET seq = 99 WHERE seq = 2")
    conn.execute("UPDATE security_audit SET seq = 2 WHERE seq = 3")
    conn.execute("UPDATE security_audit SET seq = 3 WHERE seq = 99")
    conn.commit()

    result = verify(conn)
    assert result.ok is False
    assert result.first_bad_seq is not None


# ----------------------------------------------------------------- triggers


def test_update_and_delete_are_blocked_by_triggers(env) -> None:
    log, conn = env
    entry = _append(log, 0)

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(
            "UPDATE security_audit SET payload_json = '{}' WHERE seq = ?",
            (entry.seq,),
        )

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM security_audit WHERE seq = ?", (entry.seq,))

    assert verify(conn).ok is True


# ---------------------------------------------------------------------- PII


def test_pii_phone_number_is_rejected(env) -> None:
    log, conn = env
    with pytest.raises(AuditError) as exc:
        log.append(
            customer_id="cust_a",
            session_id="sess_1",
            plan_id="plan_1",
            kind=AuditKind.INTENT,
            payload={"phone": "13800138000"},
            now=BASE,
        )
    assert exc.value.code == "pii"
    assert verify(conn).entries == 0


def test_pii_sixteen_digit_account_is_rejected(env) -> None:
    log, _ = env
    with pytest.raises(AuditError) as exc:
        log.append(
            customer_id="cust_a",
            session_id="sess_1",
            plan_id="plan_1",
            kind=AuditKind.PLAN,
            payload={"nested": [{"account": "6222020200001234"}]},
            now=BASE,
        )
    assert exc.value.code == "pii"


@pytest.mark.parametrize(
    "phone",
    [
        "138 0013 8000",
        "138-0013-8000",
        "１３８００１３８０００",
        "一三八零零一三八零零零",
    ],
)
def test_pii_obfuscated_phone_is_rejected(env, phone) -> None:
    log, conn = env
    with pytest.raises(AuditError) as exc:
        _append(log, 0, payload={"phone": phone})
    assert exc.value.code == "pii"
    assert verify(conn).entries == 0


def test_masked_tail_without_separators_is_allowed(env) -> None:
    log, conn = env
    entry = log.append(
        customer_id="cust_a",
        session_id="sess_1",
        plan_id="plan_1",
        kind=AuditKind.MANDATE,
        payload={"account": "···0932"},
        now=BASE,
    )
    assert entry.seq == 1
    assert verify(conn).ok is True


def test_masked_tail_and_token_labels_are_allowed(env) -> None:
    log, conn = env
    entry = log.append(
        customer_id="cust_a",
        session_id="sess_1",
        plan_id="plan_1",
        kind=AuditKind.MANDATE,
        payload={
            "payee": "<PAYEE_1>",
            "account": "\u00b7\u00b7\u00b70932",
            "amount_fen": 30000,
        },
        now=BASE,
    )
    assert entry.seq == 1
    assert verify(conn).ok is True


def test_pii_guard_does_not_break_the_next_append(env) -> None:
    log, conn = env
    _append(log, 0)
    with pytest.raises(AuditError):
        _append(log, 1, payload={"account": "1234567890123456"})
    _append(log, 2)
    assert verify(conn).ok is True
    assert verify(conn).entries == 2


def test_unknown_kind_is_rejected(env) -> None:
    log, _ = env
    with pytest.raises(AuditError) as exc:
        _append(log, 0, kind="NOT_A_KIND")
    assert exc.value.code == "kind"


# -------------------------------------------------------------------- explain


def test_explain_builds_short_lines_from_payload(env) -> None:
    log, _ = env
    _append(
        log,
        0,
        kind="POLICY",
        payload={
            "tier": "yellow",
            "amount_fen": 30000,
            "daily_total_fen": 100000,
            "rules": ["R_NEW_PAYEE"],
            "mandate_id": "m_000001",
            "reasons": [
                {
                    "code": "R_NEW_PAYEE",
                    "message_en": "This is a new payee.",
                    "message_zh": "这是新的收款人。",
                }
            ],
        },
    )
    _append(
        log,
        1,
        kind="MANDATE",
        payload={
            "mandate_id": "m_000001",
            "amount_fen": 30000,
            "payee": "<PAYEE_1>",
        },
    )

    lines = log.explain("cust_a", "plan_1")
    assert len(lines) == 2
    assert lines[0].startswith("13:38:02 ")
    assert "needs your confirmation" in lines[0]
    assert "\u00a5300.00, \u00a51,000.00 today" in lines[0]
    assert "This is a new payee." in lines[0]
    assert "You confirmed this action" in lines[1]

    zh = log.explain("cust_a", "plan_1", lang="zh")
    assert "需要你确认" in zh[0]
    assert "今日累计 \u00a51,000.00" in zh[0]
    assert "这是新的收款人。" in zh[0]
    assert "你确认了这笔操作" in zh[1]

    for line in lines + zh:
        assert re.search(r"\d{11,}", line) is None
        assert "m_000001" not in line
        assert "R_NEW_PAYEE" not in line


def _customer_chain(log) -> None:
    _append(
        log,
        0,
        kind="PLAN",
        payload={"steps": [{"tool": "transfer"}, {"tool": "merchant_order"}]},
    )
    _append(
        log,
        1,
        kind="POLICY",
        payload={
            "tier": "red",
            "action": "transfer",
            "amount_fen": 600000,
            "daily_total_fen": 800000,
            "daily_limit_fen": 1000000,
            "rules": ["R_RED"],
            "reasons": [
                {
                    "code": "R_RED",
                    "message_zh": "收款人是新账户。",
                    "message_en": "The payee is a new account.",
                }
            ],
        },
    )
    _append(
        log,
        2,
        kind="MANDATE",
        payload={"mandate_id": "m_000001", "amount_fen": 600000},
    )
    _append(
        log,
        3,
        kind="RUN",
        payload={
            "tool": "transfer",
            "amount_fen": 600000,
            "mandate_id": "m_000001",
            "tx_id": "tr_000009",
        },
    )
    _append(log, 4, kind="REPLY", payload={"text": "done"})


def test_explain_chain_is_customer_readable_in_chinese(env) -> None:
    log, _ = env
    _customer_chain(log)

    zh = log.explain("cust_a", "plan_1", lang="zh")
    joined = "\n".join(zh)

    assert len(zh) == 4  # REPLY is not shown.
    assert "2 个步骤" in zh[0]
    assert "短信验证码" in joined
    assert "转账" in joined
    assert "\u00a56,000.00" in joined
    assert "今日累计 \u00a58,000.00" in joined
    assert "限额 \u00a510,000.00" in joined
    assert "收款人是新账户。" in joined
    assert "你确认了这笔操作" in joined
    assert "已执行：转账 \u00a56,000.00" in joined
    assert "m_" not in joined
    assert "tr_" not in joined
    assert "transfer" not in joined
    assert "merchant_order" not in joined
    assert re.search(r"\b(?:PLAN|POLICY|MANDATE|RUN|REPLY)\b", joined) is None
    assert re.search(r"\b[A-Z]{2,}\b", joined) is None
    assert re.search(r"\d{11,}", joined) is None
    for line in zh:
        assert re.match(r"^\d{2}:\d{2}:\d{2} ", line)


def test_explain_chain_is_customer_readable_in_english(env) -> None:
    log, _ = env
    _customer_chain(log)

    en = log.explain("cust_a", "plan_1", lang="en")
    joined = "\n".join(en)

    assert len(en) == 4
    assert "AnBan understood this as 2 steps" in joined
    assert "SMS code" in joined
    assert "The payee is a new account." in joined
    assert "You confirmed this action" in joined
    assert "Done: bank transfer \u00a56,000.00" in joined
    assert "today / limit" in joined
    assert re.search(r"[\u4e00-\u9fff]", joined) is None
    assert "m_000001" not in joined
    assert "tr_000009" not in joined


def test_explain_veto_scam_is_plain(env) -> None:
    log, _ = env
    _append(
        log,
        0,
        kind="VETO",
        payload={"reason": "scam_risk", "rule": "R_SCAM", "amount_fen": 600000},
    )

    zh = log.explain("cust_a", "plan_1", lang="zh")
    assert len(zh) == 1
    assert "未执行：检测到诈骗风险" in zh[0]
    assert "scam" not in zh[0].lower()
    assert "R_SCAM" not in zh[0]

    en = log.explain("cust_a", "plan_1", lang="en")
    assert "Not done: fraud risk detected" in en[0]
    assert "scam" not in en[0].lower()


def test_explain_skips_intent_unless_it_is_the_only_event(env) -> None:
    log, _ = env
    _append(log, 0, kind="INTENT", payload={"text": "send money"})
    assert log.explain("cust_a", "plan_1", lang="zh") == ["13:38:02 你发来一条请求"]

    _append(log, 1, kind="PLAN", payload={"steps": [{}, {}]})
    zh = log.explain("cust_a", "plan_1", lang="zh")
    assert zh == ["13:38:03 安办理解为：2 个步骤"]


def test_explain_never_leaks_long_digit_runs(env) -> None:
    log, conn = env
    # Write around the API to prove explain itself is defensive too.
    _drop_triggers(conn)
    conn.execute(
        "INSERT INTO security_audit"
        " (customer_id, session_id, plan_id, ts, kind, payload_json,"
        "  prev_hash, hash)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "cust_a",
            "sess_1",
            "plan_1",
            BASE.isoformat(),
            "SCAM",
            '{"reason":"call 13800138000 now"}',
            "0" * 64,
            "0" * 64,
        ),
    )
    conn.commit()

    for line in log.explain("cust_a", "plan_1"):
        assert re.search(r"\d{11,}", line) is None


def test_explain_run_read_tool_says_checked_not_done(env) -> None:
    log, _ = env
    _append(log, 0, kind="RUN", payload={"tool": "can_cover", "amount_fen": 20000})

    zh = log.explain("cust_a", "plan_1", lang="zh")
    assert len(zh) == 1
    assert "已查看" in zh[0]
    assert "\u00a5200.00" in zh[0]
    assert "已执行" not in zh[0]

    en = log.explain("cust_a", "plan_1", lang="en")
    assert "Checked:" in en[0]
    assert "Done:" not in en[0]


def test_explain_run_write_tool_still_says_done(env) -> None:
    log, _ = env
    _append(log, 0, kind="RUN", payload={"tool": "transfer", "amount_fen": 20000})

    zh = log.explain("cust_a", "plan_1", lang="zh")
    assert len(zh) == 1
    assert "已执行：转账" in zh[0]
    assert "已查看" not in zh[0]


def test_every_registry_tool_has_plain_words() -> None:
    from agent.tools import REGISTRY

    missing = set(REGISTRY) - set(_TOOL_WORDS)
    assert missing == set()

    read_tools = {name for name, spec in REGISTRY.items() if spec.kind == "read"}
    assert _READ_TOOLS == read_tools


def test_explain_rejects_unknown_language(env) -> None:
    log, _ = env
    with pytest.raises(ValueError):
        log.explain("cust_a", "plan_1", lang="fr")


# --------------------------------------------------------------------- scope


def test_customer_isolation(env) -> None:
    log, _ = env
    _append(log, 0, customer_id="cust_a")
    _append(log, 1, customer_id="cust_b")
    _append(log, 2, customer_id="cust_a", plan_id="plan_2")
    _append(log, 3, customer_id="cust_b")

    a_entries = log.entries_for("cust_a")
    assert [e.seq for e in a_entries] == [1, 3]
    assert all(e.customer_id == "cust_a" for e in a_entries)

    a_plan_1 = log.entries_for("cust_a", plan_id="plan_1")
    assert [e.seq for e in a_plan_1] == [1]

    b_lines = log.explain("cust_b", "plan_1")
    assert len(b_lines) == 2
    assert all("cust_a" not in line for line in b_lines)

    assert log.entries_for("nobody") == []
