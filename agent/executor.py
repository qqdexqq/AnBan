"""Execute one policy-approved plan step against the mock bank.

Semantic tokens are resolved to real values only here, immediately before the
bank call, and only after the run-time policy re-check and the one-shot mandate
consumption.  Audit payloads contain token labels and integer fen amounts, never
real names or account numbers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from agent import habits
from agent.tools import REGISTRY, STEP_REF_RE, TOKEN_RE, ToolSpec
from bank import bills, calc, controls, redemption
from bank.api import BankError
from bank.merchants import price_of
from bank.money import split_fen
from bank.recurring import recurring_overview
from bank.standing_orders import standing_order_overview
from knowledge import kb
from security import compliance
from security.audit import AuditKind, AuditLog
from security.cooling import cooling_for
from security.mandate import MandateError, MandateService
from security.policy import TIER_RANK, PolicyConfig, StepDecision, recheck_step
from security.tokens import Session, TokenError, TokenVault, _last4, _next_birthday

_TOKEN_FULL_RE = re.compile(rf"^{TOKEN_RE}$")
_TOKEN_ANY_RE = re.compile(TOKEN_RE)
_STEP_REF_FULL_RE = re.compile(rf"^{STEP_REF_RE}$")
_STEP_REF_PARTS_RE = re.compile(r"^\$(s\d+)\.([a-z_]+)$")

# Demo-only rates for the untrained quote tools.
_INSTALLMENT_FEE_PCT = {3: 0.75, 6: 0.70, 12: 0.60, 24: 0.60}
_DEPOSIT_RATES_PCT = {3: 0.85, 6: 1.05, 12: 1.15, 24: 1.20, 36: 1.25, 60: 1.30}
_DEMAND_RATE_PCT = 0.05


class ExecError(Exception):
    """Execution refused or failed, carrying a stable machine-readable code."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(slots=True)
class StepContext:
    """Everything one step needs, with ``now`` always injected by the caller."""

    bank: Any
    vault: TokenVault
    session: Session
    mandates: MandateService
    audit: AuditLog
    policy_config: PolicyConfig
    now: date | datetime | str
    scam_flags: frozenset[str] = frozenset()
    payments_locked: bool | None = None
    secret_key: bytes = b""


# ---------------------------------------------------------------------- helpers


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError("expected a date, datetime or ISO date string")


def _step_id(step: object) -> str:
    if isinstance(step, Mapping):
        return str(step.get("id", ""))
    return str(getattr(step, "id", ""))


def _tool_name(step: object) -> str:
    if isinstance(step, Mapping):
        return str(step.get("tool", ""))
    return str(getattr(step, "tool", ""))


def _step_args(step: object) -> dict[str, Any]:
    if isinstance(step, Mapping):
        args = step.get("args") or {}
    else:
        args = getattr(step, "args", {}) or {}
    return dict(args)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _now_iso(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time()).isoformat()
    return str(value)


def _token_labels(value: object) -> list[str]:
    labels: list[str] = []

    def walk(item: object) -> None:
        if isinstance(item, str):
            labels.extend(match.group(0) for match in _TOKEN_ANY_RE.finditer(item))
        elif isinstance(item, Mapping):
            for nested in item.values():
                walk(nested)
        elif isinstance(item, (list, tuple, set, frozenset)):
            for nested in item:
                walk(nested)

    walk(value)
    return sorted(set(labels))


# -------------------------------------------------------------------- argument IO


def _resolve_value(
    value: Any,
    vault: TokenVault,
    session: Session,
    results: Mapping[str, dict],
) -> Any:
    if isinstance(value, str):
        if _TOKEN_FULL_RE.fullmatch(value):
            try:
                entry = vault.resolve(value, session)
            except TokenError as exc:
                raise ExecError("token", exc.code) from exc
            if entry.kind == "DATE":
                try:
                    return _as_date(entry.real_ref)
                except (TypeError, ValueError) as exc:
                    raise ExecError("token", "invalid_date") from exc
            return entry.real_ref

        if _STEP_REF_FULL_RE.fullmatch(value):
            match = _STEP_REF_PARTS_RE.fullmatch(value)
            assert match is not None
            source_id, field_name = match.groups()
            source = results.get(source_id)
            if source is None:
                raise ExecError("token", "unknown_ref")
            if field_name not in source:
                raise ExecError("token", "missing_field")
            return source[field_name]
        return value

    if isinstance(value, Mapping):
        # Relative date object: {"date": token|ref, "offset_days": int}.
        if "date" in value:
            try:
                target = _as_date(
                    _resolve_value(value["date"], vault, session, results)
                )
            except (TypeError, ValueError) as exc:
                raise ExecError("token", "invalid_date") from exc
            offset = value.get("offset_days", 0)
            if isinstance(offset, int) and not isinstance(offset, bool):
                target = target + timedelta(days=offset)
            return target
        return {
            key: _resolve_value(item, vault, session, results)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_resolve_value(item, vault, session, results) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_value(item, vault, session, results) for item in value)
    return value


def resolve_args(
    step: object,
    vault: TokenVault,
    session: Session,
    results: Mapping[str, dict],
) -> dict[str, Any]:
    """Resolve tokens, step references and relative dates into real values."""

    resolved = {
        key: _resolve_value(value, vault, session, results)
        for key, value in _step_args(step).items()
    }

    amount = resolved.get("amount")
    divide_by = resolved.get("divide_by")
    if (
        isinstance(amount, int)
        and not isinstance(amount, bool)
        and isinstance(divide_by, int)
        and not isinstance(divide_by, bool)
        and divide_by >= 2
    ):
        resolved["amount"] = split_fen(amount, divide_by)
    return resolved


def _day_of_month(resolved: Mapping[str, Any]) -> int | None:
    day = resolved.get("day_of_month")
    if isinstance(day, int) and not isinstance(day, bool):
        return day
    return None


def _amount_for(tool: str, resolved: Mapping[str, Any]) -> int:
    if tool == "merchant_order":
        try:
            return int(
                price_of(str(resolved.get("merchant")), str(resolved.get("item")))
            )
        except (KeyError, TypeError):
            return 0
    amount = resolved.get("amount")
    if isinstance(amount, int) and not isinstance(amount, bool):
        return amount
    return 0


# --------------------------------------------------------------------- dispatch


def _contact_name(bank: Any, contact_id: object) -> str | None:
    if contact_id is None:
        return None
    conn = getattr(bank, "conn", None)
    if conn is None:
        return str(contact_id)
    row = conn.execute(
        "SELECT name FROM contacts WHERE customer_id = ? AND id = ?",
        (bank.customer_id, str(contact_id)),
    ).fetchone()
    if row is None:
        return str(contact_id)
    return row["name"]


def _contact_display(bank: Any, contact_id: object) -> str | None:
    """Masked contact display ("李娜 Lina ···1660"), never the account number."""
    if contact_id is None:
        return None
    conn = getattr(bank, "conn", None)
    if conn is None:
        return str(contact_id)
    row = conn.execute(
        "SELECT name, account_no FROM contacts WHERE customer_id = ? AND id = ?",
        (bank.customer_id, str(contact_id)),
    ).fetchone()
    if row is None:
        return str(contact_id)
    return f"{row['name']} ···{_last4(row['account_no'])}"


def _persona_id(bank: Any) -> str | None:
    persona = getattr(bank, "persona_id", None)
    if isinstance(persona, str) and persona:
        return persona
    conn = getattr(bank, "conn", None)
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT persona_id FROM customers WHERE id = ?",
            (getattr(bank, "customer_id", ""),),
        ).fetchone()
    except Exception:  # pragma: no cover - defensive, bad bank stub
        return None
    if row is None:
        return None
    try:
        return row["persona_id"]
    except (KeyError, TypeError, IndexError):
        return row[0]


def _is_elder(bank: Any, config: PolicyConfig) -> bool:
    persona = _persona_id(bank)
    if persona is not None and persona in config.elder_personas:
        return True
    # Self-enabled elder mode gets the same 24 h red cooling.
    return compliance.elder_mode(getattr(bank, "conn", None), str(getattr(bank, "customer_id", "")))


def _period_range(period: str | None, now: date | datetime | str) -> tuple[date, date] | None:
    if not period:
        return None
    current = _as_date(now)
    if period == "this_month":
        return current.replace(day=1), current
    if period == "last_month":
        first_this = current.replace(day=1)
        last_month_end = first_this - timedelta(days=1)
        return last_month_end.replace(day=1), last_month_end
    if period == "this_year":
        return date(current.year, 1, 1), current
    if period == "last_year":
        return date(current.year - 1, 1, 1), date(current.year - 1, 12, 31)
    if re.fullmatch(r"month:[0-9]{4}-(?:0[1-9]|1[0-2])", period):
        try:
            start = date.fromisoformat(period[6:] + "-01")
            next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
            end = next_month - timedelta(days=1)
            return start, min(end, current) if start <= current <= end else end
        except ValueError as exc:
            raise ExecError("bank", "invalid_period") from exc
    if re.fullmatch(r"year:[0-9]{4}", period):
        try:
            year = int(period[5:])
            return date(year, 1, 1), current if year == current.year else date(year, 12, 31)
        except ValueError as exc:
            raise ExecError("bank", "invalid_period") from exc
    if period == "last_7_days":
        return current - timedelta(days=6), current
    if period == "last_30_days":
        return current - timedelta(days=29), current
    raise ExecError("bank", "invalid_period")


def _report_label(start: date, end: date) -> str:
    if start.year == end.year and start.month == 1 and start.day == 1 and (end.month > 1):
        return str(start.year)
    return start.strftime("%Y-%m") if start.year == end.year and start.month == end.month else f"{start.isoformat()} to {end.isoformat()}"


def _report_counterparty(bank: Any, resolved: Mapping[str, Any]) -> str | None:
    """Resolve opaque read references only within the authenticated customer's ledger."""
    selected = []
    if resolved.get("payee") is not None:
        selected.append(_contact_name(bank, resolved["payee"]))
    if resolved.get("merchant") is not None:
        selected.append(str(resolved["merchant"]))
    if resolved.get("merchant_ref") is not None:
        row = bank.conn.execute("SELECT counterparty FROM transactions WHERE customer_id = ? AND id = ?",
                                (bank.customer_id, str(resolved["merchant_ref"])[4:])).fetchone()
        if row is None:
            raise ExecError("bank", "not_found")
        selected.append(str(row["counterparty"]))
    if len(set(selected)) > 1:
        raise ExecError("bank", "ambiguous_subject")
    return selected[0] if selected else None


def _required(resolved: Mapping[str, Any], name: str) -> Any:
    value = resolved.get(name)
    if value is None:
        raise ExecError("bank", "invalid_state")
    return value


def _card_id(
    bank: Any,
    resolved: Mapping[str, Any],
    keys: tuple[str, ...] = ("card",),
) -> str:
    """Pick the named card, or the customer's only non-virtual open card."""

    for key in keys:
        value = resolved.get(key)
        if value is not None and str(value):
            return str(value)
    cards = [
        card
        for card in bank.list_cards()
        if card.kind != "virtual" and card.status != "closed"
    ]
    if len(cards) == 1:
        return cards[0].id
    raise ExecError("bank", "invalid_state")


def _raw(value: Any) -> Any:
    """Keep the bank object for the responder; tests may inspect it as-is."""

    return value


def _transaction_rows(transactions: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for txn in transactions:
        rows.append(
            {
                "id": txn.id,
                "ts": txn.ts,
                "amount_fen": txn.amount_fen,
                "counterparty": txn.counterparty,
                "category": txn.category,
                "memo": txn.memo,
                "channel": txn.channel,
            }
        )
    return rows


def _dispatch(
    tool: str,
    spec: ToolSpec,
    resolved: Mapping[str, Any],
    step: object,
    decision: StepDecision,
    ctx: StepContext,
    results: Mapping[str, dict],
    idempotency_key: str,
) -> dict[str, Any]:
    bank = ctx.bank
    step_id = _step_id(step)

    # ------------------------------------------------------------------ reads
    if tool == "run_analysis":
        from agent.analysis_runner import run_spending_analysis

        start, end = _period_range(resolved.get("period") or "last_30_days", ctx.now)
        return run_spending_analysis(
            bank, start, end, str(resolved.get("kind", "by_weekday")),
            lambda payload: _append_audit(ctx, AuditKind.SANDBOX, payload),
        )

    if tool == "get_balance":
        raw = bank.get_balance()
        return {"raw": raw, "balance": raw}

    if tool == "list_transactions":
        start = end = None
        period_range = _period_range(resolved.get("period"), ctx.now)
        if period_range is not None:
            start, end = period_range
        counterparty = _report_counterparty(bank, resolved)
        raw = bank.list_transactions(start=start, end=end, category=resolved.get("category"),
                                     counterparty=counterparty, limit=None)
        direction = resolved.get("direction")
        if direction == "in":
            raw = [row for row in raw if row.amount_fen > 0]
        elif direction == "out":
            raw = [row for row in raw if row.amount_fen < 0]
        outgoing = [row for row in raw if row.amount_fen < 0]
        result = {"raw": raw, "transactions": _transaction_rows(raw),
                  "amount": abs(outgoing[0].amount_fen) if outgoing else None,
                  "payee": outgoing[0].counterparty if outgoing else None}
        if period_range:
            result.update(period_start=start, period_end=end, period_label=_report_label(start, end))
        if resolved.get("explain_anomalies"):
            # Detection uses all relevant history; only matched customer-owned rows are shown.
            days = max(30, (_as_date(ctx.now) - start).days + 1) if start else 30
            ids = {row.id for row in raw}
            result["anomalies"] = [dict(item, txn_ids=[tid for tid in item["txn_ids"] if tid in ids])
                                   for item in bank.detect_anomalies(ctx.now, days=days)
                                   if any(tid in ids for tid in item["txn_ids"])]
        return result

    if tool == "spending_report":
        start, end = _period_range(resolved.get("period") or "this_month", ctx.now)
        annual = str(resolved.get("period", "")).startswith("year:") or resolved.get("period") in {"this_year", "last_year"}
        mode = resolved.get("mode", "summary")
        if annual and mode == "summary":
            mode = "report"
        category = resolved.get("category")
        counterparty = _report_counterparty(bank, resolved)
        raw = bank.spending_report(start, end, category=category, counterparty=counterparty)
        income = bank.income_report(start, end, category=category, counterparty=counterparty)
        records = bank.list_transactions(start, end, category, counterparty, limit=None)
        result = {"raw": raw, "total": raw.get("total", 0), "by_category": dict(raw.get("by_category", {})),
                  "income": income, "net": income - raw.get("total", 0), "mode": mode,
                  "period_start": start, "period_end": end, "period_label": _report_label(start, end),
                  # Data exists for the period even if this category/merchant had none.
                  "available": bool(records) or bool(bank.list_transactions(start, end, None, None, limit=1)),
                  "filter_category": category, "filter_merchant": counterparty,
                  "transfer_transactions": _transaction_rows([row for row in records if row.amount_fen < 0
                    and row.category == "transfer" and row.counterparty != "我的储蓄账户"])}
        if resolved.get("compare"):
            first = start.replace(day=1)
            prev_end = first - timedelta(days=1)
            prev_start = prev_end.replace(day=1)
            previous = bank.spending_report(prev_start, prev_end, category=category, counterparty=counterparty)
            result["comparison"] = {"period_label": _report_label(prev_start, prev_end),
                "period_start": prev_start, "period_end": prev_end, "total": previous["total"],
                "available": bool(bank.list_transactions(prev_start, prev_end, category, counterparty, limit=None)),
                "delta": result["total"] - previous["total"],
                "by_category": {key: result["by_category"].get(key, 0) - previous["by_category"].get(key, 0)
                                for key in result["by_category"].keys() | previous["by_category"].keys()}}
        if annual or mode == "highest_month":
            months = []
            for month in range(1, 13):
                lo = date(start.year, month, 1)
                hi = (lo.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
                hi = min(hi, end)
                month_rows = bank.list_transactions(lo, hi, category, counterparty, limit=None) if lo <= hi else []
                report = bank.spending_report(lo, hi, category=category, counterparty=counterparty) if month_rows else {"total": 0}
                months.append({"period_label": lo.strftime("%Y-%m"), "period_start": lo,
                               "total": report["total"], "available": bool(month_rows)})
            result["months"] = months
        return result

    if tool == "detect_anomalies":
        raw = bank.detect_anomalies(ctx.now, days=30)
        end = _as_date(ctx.now)
        start = end - timedelta(days=30)
        return {"raw": _raw(raw), "anomalies": _raw(raw), "period_label": _report_label(start, end),
                "period_start": start, "period_end": end}

    if tool == "list_subscriptions":
        raw = bank.list_subscriptions()
        try:
            recurring = recurring_overview(bank.conn, bank.customer_id, ctx.now)
        except Exception:
            recurring = []
        return {"raw": _raw(raw), "sub": [item.id for item in raw], "recurring": recurring,
                "scope": resolved.get("scope", "all")}

    if tool == "list_reminders":
        exists = bank.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'agent_reminders'"
        ).fetchone()
        rows = (
            bank.conn.execute(
                """SELECT due_on, note, created_at FROM agent_reminders
                WHERE customer_id = ?""",
                (bank.customer_id,),
            ).fetchall()
            if exists is not None
            else []
        )
        items = [
            {
                "due_on": row["due_on"],
                "note": row["note"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        items.sort(
            key=lambda item: (
                item["due_on"] is None,
                item["due_on"] or "",
                item["created_at"],
            )
        )
        today = _as_date(ctx.now).isoformat()
        due = [
            item
            for item in items
            if item["due_on"] is not None and item["due_on"] <= today
        ]
        return {"raw": items, "reminder": items, "due": due}

    if tool == "list_scheduled":
        items = standing_order_overview(bank.conn, bank.customer_id, ctx.now)
        rows: list[dict[str, Any]] = []
        for item in items:
            row = dict(item)
            contact_id = row.pop("contact_id", None)
            row["payee_display"] = _contact_display(bank, contact_id)
            rows.append(row)
        return {"raw": rows, "scheduled": rows}

    if tool == "list_cards":
        raw = bank.list_cards()
        return {"raw": _raw(raw), "card": [item.id for item in raw], "cards": bank.card_details(ctx.now)}

    if tool == "list_products":
        raw = bank.list_products()
        return {"raw": _raw(raw), "product": [item.id for item in raw]}

    if tool == "get_portfolio":
        raw = bank.get_portfolio()
        return {"raw": _raw(raw), "portfolio": _raw(raw)}

    if tool == "list_contacts":
        raw = bank.find_contact("")
        return {"raw": _raw(raw), "payee": [item.id for item in raw]}

    if tool == "find_contact":
        query = resolved.get("query")
        raw_query = _step_args(step).get("query")
        if isinstance(raw_query, str) and _TOKEN_FULL_RE.fullmatch(raw_query):
            try:
                entry = ctx.vault.resolve(raw_query, ctx.session)
            except TokenError as exc:
                raise ExecError("token", exc.code) from exc
            if entry.kind == "PAYEE":
                contact_id = str(entry.real_ref)
                raw = [
                    contact
                    for contact in bank.find_contact("")
                    if contact.id == contact_id
                ]
                return {
                    "raw": _raw(raw),
                    "payee": contact_id if raw else None,
                }
        name = _contact_name(bank, query)
        raw = bank.find_contact(name or "")
        return {
            "raw": _raw(raw),
            "payee": raw[0].id if raw else None,
        }

    if tool == "get_contact_date":
        target = bank.get_contact_date(str(_required(resolved, "payee")))
        if target is None:
            # No date on file: fail here with a clear reason instead of letting
            # a later "$sN.date" step fail as an unresolved reference.
            raise ExecError("date_unknown", "contact_date")
        # Context exposes the next birthday, not the customer's birth year.
        target = _next_birthday(target, _as_date(ctx.now))
        return {"raw": target, "date": target}

    if tool == "can_cover":
        raw = bool(bank.can_cover(_amount_for(tool, resolved)))
        return {"raw": raw, "ok": raw}

    if tool == "sandbox_chart":
        series: dict[str, Any] = {}
        for earlier in reversed(list(results.values())):
            if isinstance(earlier, Mapping) and "by_category" in earlier:
                series = dict(earlier.get("by_category") or {})
                break
        chart = {"type": "bar", "by_category": series}
        return {"raw": chart, "chart": chart}

    # ----------------------------------------------------------- local writes
    if tool == "set_reminder":
        # The planner writes the note, and it is later shown in the bank's own
        # voice: keep plain words only (no digits, tokens, links or symbols).
        note = resolved.get("note")
        if not (isinstance(note, str) and re.fullmatch(r"[A-Za-z一-鿿 ]{1,20}", note.strip())):
            note = None
        reminder = {"when": resolved.get("when"), "note": note.strip() if note else None}
        when = reminder["when"]
        due_on = when.isoformat() if isinstance(when, (date, datetime)) else None
        owned = not bank.conn.in_transaction
        if owned:
            bank.conn.execute("BEGIN")
        try:
            bank.conn.execute("""CREATE TABLE IF NOT EXISTS agent_reminders (
                customer_id TEXT NOT NULL, idempotency_key TEXT NOT NULL,
                due_on TEXT, note TEXT, created_at TEXT NOT NULL,
                PRIMARY KEY(customer_id, idempotency_key))""")
            bank.conn.execute("""INSERT OR IGNORE INTO agent_reminders
                (customer_id, idempotency_key, due_on, note, created_at)
                VALUES (?, ?, ?, ?, ?)""", (
                bank.customer_id, idempotency_key, due_on, reminder["note"],
                ctx.now.isoformat(),
            ))
            if owned:
                bank.conn.commit()
        except BaseException:
            if owned:
                bank.conn.rollback()
            raise
        reminder.update(saved=True, notifications_enabled=False)
        reminder["due_on"] = due_on
        return {"raw": reminder, "reminder": reminder}

    if tool == "subscribe_product":
        amount = _amount_for(tool, resolved)
        raw = bank.subscribe_product(
            str(_required(resolved, "product")), amount, ctx.now
        )
        holding = {
            "product_id": raw.product_id,
            "amount_fen": amount,
            "status": "subscribed",
        }
        return {"raw": raw, "holding": holding, "amount": amount}

    if tool == "change_password":
        raw = {"changed": False, "status": "requires_secure_channel", "simulated": True}
        return {"raw": raw, "changed": False}

    # ------------------------------------------------------ untrained batch
    if tool == "list_bills":
        cid = bank.customer_id
        bills.seed_bills(bank.conn, cid, _as_date(ctx.now))
        rows = bills.list_bills(bank.conn, cid)
        return {
            "raw": [
                {
                    key: row[key]
                    for key in (
                        "category",
                        "biller",
                        "biller_en",
                        "amount_due_fen",
                        "due_date",
                    )
                }
                for row in rows
            ],
            "bill": [row["category"] for row in rows],
        }

    if tool == "pay_bill":
        cid = bank.customer_id
        bills.seed_bills(bank.conn, cid, _as_date(ctx.now))
        r = bills.pay_bill(bank.conn, cid, resolved["category"], ctx.now)
        return {
            "raw": {
                key: r[key]
                for key in (
                    "category",
                    "biller",
                    "biller_en",
                    "amount_due_fen",
                    "paid_at",
                    "status",
                )
            },
            "bill": r["category"],
            "amount": r["amount_due_fen"],
        }

    if tool == "installment_quote":
        amount = _amount_for(tool, resolved)
        periods = resolved["periods"]
        fee = _INSTALLMENT_FEE_PCT[periods]
        q = calc.installment_quote(amount, periods, fee)
        raw = {**q, "principal_fen": amount, "fee_rate_per_period_pct": fee}
        return {"raw": raw, "quote": raw}

    if tool == "deposit_quote":
        amount = _amount_for(tool, resolved)
        term_months = resolved["term_months"]
        rate = _DEPOSIT_RATES_PCT[term_months]
        days = term_months * 30
        interest = calc.deposit_interest(amount, rate, days)
        early = calc.deposit_interest(amount, _DEMAND_RATE_PCT, days // 2)
        raw = {
            "principal_fen": amount,
            "term_months": term_months,
            "rate_pct": rate,
            "interest_fen": interest,
            "demand_rate_pct": _DEMAND_RATE_PCT,
            "early_half_term_fen": early,
        }
        return {"raw": raw, "quote": raw}

    if tool == "forecast_cashflow":
        cid = bank.customer_id
        today = _as_date(ctx.now)
        days = resolved.get("days") or 30

        recurring: list[tuple[int, int]] = []
        one_off: list[tuple[date, int]] = []
        items: list[dict[str, Any]] = []

        for sub in bank.list_subscriptions():
            if sub.status == "active":
                amount = -sub.amount_fen
                recurring.append((sub.day_of_month, amount))
                items.append(
                    {
                        "day": sub.day_of_month,
                        "amount_fen": amount,
                        "label": sub.merchant,
                        "recurrence": "monthly",
                    }
                )
        scheduled = bank.conn.execute(
            """
            SELECT day_of_month, amount_fen FROM scheduled_transfers
            WHERE customer_id = ? AND status = 'scheduled'
            """,
            (cid,),
        ).fetchall()
        for row in scheduled:
            amount = -row["amount_fen"]
            recurring.append((row["day_of_month"], amount))
            items.append(
                {
                    "day": row["day_of_month"],
                    "amount_fen": amount,
                    "label": "scheduled",
                    "recurrence": "monthly",
                }
            )
        salary_rows = bank.conn.execute(
            """
            SELECT ts, amount_fen, category, counterparty
            FROM transactions
            WHERE customer_id = ? AND amount_fen > 0 AND ts >= ?
            ORDER BY ts DESC, id DESC
            """,
            (cid, (today - timedelta(days=45)).isoformat()),
        ).fetchall()
        # Seeded salary rows use category "income"; habits knows how to tell
        # them apart from the opening-balance row.
        salary = next(
            (row for row in salary_rows if habits._is_salary_row(row)), None
        )
        if salary is not None:
            salary_day = date.fromisoformat(str(salary["ts"])[:10])
            amount = salary["amount_fen"]
            recurring.append((salary_day.day, amount))
            # Inferred from one past payment, so it is an assumption only.
            items.append(
                {
                    "day": salary_day.day,
                    "amount_fen": amount,
                    "label": "salary",
                    "recurrence": "monthly",
                    "assumption": True,
                }
            )
        bills.seed_bills(bank.conn, cid, today)
        cutoff = today + timedelta(days=days)
        for bill in bills.list_bills(bank.conn, cid):
            due = date.fromisoformat(bill["due_date"])
            if today <= due <= cutoff:
                amount = -bill["amount_due_fen"]
                one_off.append((due, amount))
                items.append(
                    {
                        "day": due.day,
                        "amount_fen": amount,
                        "label": bill["biller_en"],
                        "date": due.isoformat(),
                        "recurrence": "one_off",
                    }
                )

        balance = bank.get_balance()
        f = calc.calendar_cashflow_forecast(
            balance, recurring, one_off, today, days
        )
        raw = {
            **f,
            "days": days,
            "start_balance_fen": balance,
            "items": items,
        }
        return {"raw": raw, "forecast": raw}

    if tool == "help_answer":
        topic = resolved["topic"]
        a = kb.article(topic)
        return {
            "raw": {
                "topic": topic,
                "zh": a["zh"],
                "en": a["en"],
                "source": a["source"],
            },
            "answer": topic,
        }

    if tool == "list_transfers":
        now = ctx.now
        now_dt = (
            now
            if isinstance(now, datetime)
            else datetime.combine(_as_date(now), datetime.min.time())
        )
        rows = bank.conn.execute(
            """
            SELECT id, to_contact_id, amount_fen, status, created_at, release_at
            FROM transfers
            WHERE customer_id = ?
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (bank.customer_id,),
        ).fetchall()

        def _status(row: Any) -> str:
            stored = row["status"]
            if stored == "cancelled":
                return "undone"
            if stored == "released":
                return "sent"
            if stored == "pending":
                if compliance.transfer_release_hold(
                    bank.conn, bank.customer_id, row["to_contact_id"], now_dt
                ) is not None:
                    return "held"
                return (
                    "held"
                    if now_dt < datetime.fromisoformat(row["release_at"])
                    else "sent"
                )
            return stored

        raw = [
            {
                "id": row["id"],
                "to_contact_id": row["to_contact_id"],
                "amount_fen": row["amount_fen"],
                "status": _status(row),
                "created_at": row["created_at"],
                "release_at": row["release_at"],
            }
            for row in rows
        ]
        return {"raw": raw, "transfer": [row["id"] for row in rows]}

    if tool == "verify_payee":
        cid = bank.customer_id
        contact_id = str(resolved["payee"])
        on_list = compliance.contact_on_fraud_list(bank.conn, cid, contact_id)
        seen = bank.conn.execute(
            """
            SELECT 1 FROM transfers
            WHERE customer_id = ? AND to_contact_id = ?
            LIMIT 1
            """,
            (cid, contact_id),
        ).fetchone()
        return {
            "raw": {
                "payee": contact_id,
                "on_fraud_list": on_list,
                "first_time": seen is None,
            },
            "risk": on_list,
        }

    if tool == "set_card_control":
        card_id = _card_id(bank, resolved)
        r = controls.set_control(
            bank.conn,
            bank.customer_id,
            card_id,
            resolved["control"],
            bool(resolved["enabled"]),
            ctx.now,
        )
        card = next(card for card in bank.list_cards() if card.id == card_id)
        r = dict(r, card_status=card.status, last4=card.last4,
                 effective_enabled=bool(resolved["enabled"]) and card.status == "active")
        return {"raw": r, "card": card_id}

    if tool == "block_merchant":
        r = controls.block_category(
            bank.conn, bank.customer_id, resolved["category"], ctx.now
        )
        return {"raw": r, "block": r["category"]}

    if tool == "unblock_merchant":
        r = controls.request_unblock(
            bank.conn, bank.customer_id, resolved["category"], ctx.now
        )
        return {"raw": r, "block": r["category"]}

    # ------------------------------------------------------------- bank writes
    if tool == "transfer":
        payee = _required(resolved, "payee")
        amount = _amount_for(tool, resolved)
        cooling = cooling_for(
            decision.tier,
            scam_flags=ctx.scam_flags,
            elder=_is_elder(bank, ctx.policy_config),
            arrival_delay=compliance.arrival_delay(bank.conn, bank.customer_id),
        )
        raw = bank.transfer(
            to_contact_id=str(payee),
            amount_fen=amount,
            note=str(resolved.get("note") or ""),
            idempotency_key=idempotency_key,
            now=ctx.now,
            cooling=cooling,
        )
        return {
            "raw": raw,
            "amount": raw.amount_fen,
            "payee": raw.to_contact_id,
            "transfer": raw.id,
            "undo": {
                "tool": "cancel_transfer",
                "args": {
                    "transfer_id": raw.id,
                    "amount_fen": raw.amount_fen,
                    "step_id": step_id,
                },
            },
        }

    if tool == "schedule_transfer":
        raw = bank.schedule_transfer(
            to_contact_id=str(_required(resolved, "payee")),
            amount_fen=_amount_for(tool, resolved),
            day_of_month=int(_required(resolved, "day_of_month")),
            note=str(resolved.get("note") or ""),
        )
        return {
            "raw": raw,
            "amount": raw.amount_fen,
            "payee": raw.to_contact_id,
            "scheduled": raw.id,
            "undo": {
                "tool": "cancel_scheduled",
                "args": {
                    "scheduled_id": raw.id,
                    "amount_fen": raw.amount_fen,
                    "step_id": step_id,
                },
            },
        }

    if tool == "stop_standing_order":
        contact_id = str(_required(resolved, "payee"))
        rows = bank.conn.execute(
            """
            SELECT id, to_contact_id, amount_fen, day_of_month
            FROM scheduled_transfers
            WHERE customer_id = ? AND status = 'scheduled' AND to_contact_id = ?
            ORDER BY id
            """,
            (bank.customer_id, contact_id),
        ).fetchall()
        if not rows:
            raise ExecError("bank", "not_found")
        stopped: list[dict[str, Any]] = []
        for row in rows:
            bank.cancel_scheduled(str(row["id"]))
            stopped.append(
                {
                    "payee_display": _contact_display(bank, row["to_contact_id"]),
                    "amount_fen": int(row["amount_fen"]),
                    "day_of_month": int(row["day_of_month"]),
                }
            )
        return {"raw": {"stopped": stopped}, "stopped": stopped}

    if tool == "request_payment":
        raw = bank.request_payment(
            from_contact_id=str(_required(resolved, "payee")),
            amount_fen=_amount_for(tool, resolved),
            note=str(resolved.get("note") or ""),
        )
        return {
            "raw": raw,
            "amount": raw.amount_fen,
            "payee": raw.from_contact_id,
            "request": raw.id,
            "undo": {
                "tool": "cancel_request",
                "args": {
                    "request_id": raw.id,
                    "amount_fen": raw.amount_fen,
                    "step_id": step_id,
                },
            },
        }

    if tool == "cancel_subscription":
        raw = bank.cancel_subscription(str(_required(resolved, "sub")))
        return {
            "raw": raw,
            "subscription": raw.id,
            "saved_monthly_fen": raw.amount_fen,
            "undo": {
                "tool": "restore_subscription",
                "args": {"subscription_id": raw.id, "step_id": step_id},
            },
        }

    if tool == "reactivate_subscription":
        sub_id = str(_required(resolved, "sub"))
        current = next(
            (item for item in bank.list_subscriptions() if item.id == sub_id), None
        )
        if current is not None and current.status == "active":
            raise ExecError("bank", "invalid_state")
        raw = bank.restore_subscription(sub_id)
        return {"raw": raw, "subscription": raw.id}

    if tool == "move_to_pot":
        raw = bank.move_to_pot(
            str(_required(resolved, "pot_name")),
            _amount_for(tool, resolved),
        )
        return {
            "raw": raw,
            "amount": _amount_for(tool, resolved),
            "pot": raw.id,
            "undo": {
                "tool": "move_from_pot",
                "args": {
                    "pot_name": raw.name,
                    "amount_fen": _amount_for(tool, resolved),
                    "step_id": step_id,
                },
            },
        }

    if tool == "redeem_product":
        # Redeem at the payout the policy re-check just saw.  Never re-quote
        # here: a changed valuation must fail the mandate binding, not silently
        # redeem at a different amount than the customer approved.
        if decision.amount_fen is None:
            raise ExecError("bank", "invalid_amount")
        raw = redemption.redeem(
            ctx.bank.conn,
            ctx.bank.customer_id,
            str(_required(resolved, "product")),
            int(decision.amount_fen),
            idempotency_key,
            ctx.now,
        )
        return {
            "raw": raw,
            "amount": raw["amount_fen"],
            "product": raw["product_id"],
            "simulated": True,
        }

    if tool == "merchant_order":
        raw = bank.merchant_order(
            merchant=str(_required(resolved, "merchant")),
            item_code=str(_required(resolved, "item")),
            deliver_on=_as_date(_required(resolved, "deliver")),
            now=ctx.now,
            funding_pot_id=resolved.get("funding_pot"),
        )
        return {
            "raw": raw,
            "amount": raw.amount_fen,
            "date": raw.deliver_on,
            "order": raw.id,
            "undo": {
                "tool": "cancel_order",
                "args": {
                    "order_id": raw.id,
                    "amount_fen": raw.amount_fen,
                    "deliver_on": raw.deliver_on.isoformat(),
                    "step_id": step_id,
                },
            },
        }

    if tool == "freeze_card":
        raw = bank.freeze_card(_card_id(bank, resolved))
        return {
            "raw": raw,
            "card": raw.id,
            "undo": {
                "tool": "unfreeze_card",
                "args": {"card_id": raw.id, "step_id": step_id},
            },
        }

    if tool == "set_card_limit":
        card_id = _card_id(bank, resolved)
        previous = next(
            (card.online_limit_fen for card in bank.list_cards() if card.id == card_id),
            None,
        )
        amount_token = _step_args(step).get("amount")
        restoring = False
        if isinstance(amount_token, str) and _TOKEN_FULL_RE.fullmatch(amount_token):
            entry = ctx.vault.resolve(amount_token, ctx.session)
            restoring = entry.origin == "bank_context" and entry.tags.get("restored_limit") is True
        raw = bank.set_card_limit(card_id, _amount_for(tool, resolved), restore=True) if restoring else bank.set_card_limit(card_id, _amount_for(tool, resolved))
        return {
            "raw": raw,
            "amount": raw.online_limit_fen,
            "card": raw.id,
            "undo": {
                "tool": "restore_card_limit",
                "args": {
                    "card_id": raw.id,
                    "previous_limit_fen": previous,
                    "step_id": step_id,
                },
            },
        }

    if tool == "create_virtual_card":
        raw = bank.create_virtual_card(_amount_for(tool, resolved))
        return {
            "raw": raw,
            "amount": raw.online_limit_fen,
            "card": raw.id,
            "undo": {
                "tool": "close_virtual_card",
                "args": {"card_id": raw.id, "step_id": step_id},
            },
        }

    if tool == "report_card_lost":
        raw = bank.report_card_lost(_card_id(bank, resolved))
        return {"raw": raw, "card": raw.id}

    # --------------------------------------------------------- undo-style tools
    if tool == "cancel_transfer":
        raw = bank.cancel_transfer(
            str(resolved.get("transfer_id") or resolved.get("id") or ""), ctx.now
        )
        return {"raw": raw, "transfer": raw.id}

    if tool == "cancel_scheduled":
        raw = bank.cancel_scheduled(
            str(resolved.get("scheduled_id") or resolved.get("id") or "")
        )
        return {"raw": raw, "scheduled": raw.id}

    if tool == "cancel_request":
        raw = bank.cancel_request(
            str(resolved.get("request_id") or resolved.get("id") or "")
        )
        return {"raw": raw, "request": raw.id}

    if tool == "restore_subscription":
        raw = bank.restore_subscription(
            str(resolved.get("subscription_id") or resolved.get("id") or "")
        )
        return {"raw": raw, "subscription": raw.id}

    if tool == "move_from_pot":
        raw = bank.move_from_pot(
            str(resolved.get("pot_name") or resolved.get("name") or ""),
            int(resolved.get("amount_fen") or resolved.get("amount") or 0),
        )
        return {"raw": raw, "pot": raw.id}

    if tool == "cancel_order":
        raw = bank.cancel_order(
            str(resolved.get("order_id") or resolved.get("id") or ""), ctx.now
        )
        return {"raw": raw, "order": raw.id}

    if tool == "unfreeze_card":
        raw = bank.unfreeze_card(_card_id(bank, resolved, ("card", "card_id", "id")))
        return {"raw": raw, "card": raw.id}

    if tool == "restore_card_limit":
        card_id = _card_id(bank, resolved, ("card", "card_id", "id"))
        if resolved.get("previous_limit_fen") is not None:
            target = int(resolved["previous_limit_fen"])
            if target < 0 or target > 2**63 - 1:
                raise ExecError("bank", "invalid_amount")
            current = next(card.online_limit_fen for card in bank.list_cards() if card.id == card_id)
            if target > current:
                raise ExecError("bank", "invalid_state")
            raw = bank.set_card_limit(card_id, target)
        else:
            raw = bank.restore_card_limit(card_id)
        return {"raw": raw, "card": card_id, "restored": raw is not None,
                "amount": raw.online_limit_fen if raw is not None else None}

    if tool == "close_virtual_card":
        raw = bank.close_virtual_card(
            str(resolved.get("card_id") or resolved.get("id") or "")
        )
        return {"raw": raw, "card": raw.id}

    # -------------------------------------------------------------- control
    if spec.kind == "control":
        return {"raw": dict(resolved), "control": tool}

    raise ExecError("unsupported", tool)


# ----------------------------------------------------------------------- audit


def _audit_payload(
    step_id: str,
    tool: str,
    decision: StepDecision,
    amount_fen: int,
    args: Mapping[str, Any],
    *,
    mandate_id: str | None = None,
    error: ExecError | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "step_id": step_id,
        "tool": tool,
        "tier": decision.tier,
        "tokens": _token_labels(args),
    }
    if amount_fen:
        payload["amount_fen"] = int(amount_fen)
    if mandate_id:
        payload["mandate_id"] = mandate_id
    if error is not None:
        payload["ok"] = False
        payload["error"] = error.code
        if error.detail:
            payload["error_detail"] = error.detail
    else:
        payload["ok"] = True
    return payload


def _append_audit(
    ctx: StepContext, kind: AuditKind, payload: dict[str, Any]
) -> None:
    ctx.audit.append(
        customer_id=ctx.bank.customer_id,
        session_id=ctx.session.session_id,
        plan_id=ctx.session.plan_id,
        kind=kind,
        payload=payload,
        now=ctx.now,
    )


# ------------------------------------------------------------------ undo store

UNDO_SCHEMA = """
CREATE TABLE IF NOT EXISTS exec_undo (
    customer_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    args_json TEXT NOT NULL,
    sig TEXT NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (customer_id, plan_id, step_id)
);
"""


def _undo_conn(ctx: StepContext) -> Any:
    return getattr(ctx.bank, "conn", None)


def _ensure_undo_table(conn: Any) -> None:
    conn.executescript(UNDO_SCHEMA)
    conn.commit()


def _undo_binding(
    customer_id: str,
    plan_id: str,
    step_id: str,
    tool: str,
    args_json: str,
    created_at: str,
) -> dict[str, str]:
    return {
        "customer_id": customer_id,
        "plan_id": plan_id,
        "step_id": step_id,
        "tool": tool,
        "args_json": args_json,
        "created_at": created_at,
    }


def _undo_sig(
    secret_key: bytes,
    customer_id: str,
    plan_id: str,
    step_id: str,
    tool: str,
    args_json: str,
    created_at: str,
) -> str:
    payload = _undo_binding(
        customer_id, plan_id, step_id, tool, args_json, created_at
    )
    return hmac.new(
        bytes(secret_key), _canonical_json(payload).encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _store_undo(
    ctx: StepContext,
    step_id: str,
    tool: str,
    args: Mapping[str, Any],
) -> bool:
    conn = _undo_conn(ctx)
    if conn is None:
        return False
    _ensure_undo_table(conn)
    args_json = _canonical_json(dict(args))
    created_at = _now_iso(ctx.now)
    customer_id = ctx.bank.customer_id
    plan_id = ctx.session.plan_id
    sig = _undo_sig(
        ctx.secret_key, customer_id, plan_id, step_id, tool, args_json, created_at
    )
    conn.execute(
        """
        INSERT INTO exec_undo
            (customer_id, plan_id, step_id, tool, args_json, sig, used, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 0, ?)
        ON CONFLICT(customer_id, plan_id, step_id) DO UPDATE SET
            tool = excluded.tool,
            args_json = excluded.args_json,
            sig = excluded.sig,
            created_at = excluded.created_at
        WHERE exec_undo.used = 0
        """,
        (customer_id, plan_id, step_id, tool, args_json, sig, created_at),
    )
    conn.commit()
    return True


# ------------------------------------------------------------------------- run


def run_step(
    step: object,
    decision: StepDecision,
    ctx: StepContext,
    results: Mapping[str, dict] | None = None,
    mandate_id: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Run one step after re-checking policy and consuming its mandate."""

    results = {} if results is None else results
    step_id = _step_id(step)
    tool = _tool_name(step)
    raw_args = _step_args(step)

    spec = REGISTRY.get(tool)
    if spec is None:
        error = ExecError("unsupported", tool)
        _append_audit(
            ctx,
            AuditKind.VETO,
            _audit_payload(
                step_id, tool, decision, 0, raw_args, error=error
            ),
        )
        raise error

    try:
        if spec.kind == "write":
            raw_payee = raw_args.get("payee")
            if isinstance(raw_payee, str) and raw_payee.startswith("$s"):
                raise ExecError("token", "payee_ref_not_allowed")

        resolved = resolve_args(step, ctx.vault, ctx.session, results)
        amount_fen = _amount_for(tool, resolved)
        if tool == "pay_bill":
            # The bill's amount, the same one the mandate was issued for.
            amount_fen = bills.due_amount(
                ctx.bank.conn,
                ctx.bank.customer_id,
                str(resolved.get("category")),
                _as_date(ctx.now),
            ) or 0
        if tool == "redeem_product":
            # The payout is the bank's current quote, never the planner's.
            try:
                quote = redemption.quote(
                    ctx.bank.conn,
                    ctx.bank.customer_id,
                    str(_required(resolved, "product")),
                )
                amount_fen = int(quote["payout_fen"])
            except BankError as exc:
                raise ExecError("bank", exc.code) from exc
            except (KeyError, TypeError, ValueError) as exc:
                raise ExecError("bank", "invalid_state") from exc

        # 1. Run-time policy re-check with the real amount.
        checked = decision
        if spec.kind == "write":
            payments_locked = ctx.payments_locked
            if payments_locked is None:
                try:
                    payments_locked = ctx.mandates.is_payments_locked(
                        ctx.bank.customer_id, ctx.now
                    )
                except (OverflowError, ValueError) as exc:
                    raise ExecError("bank", "invalid_amount") from exc
            try:
                checked = recheck_step(
                    decision,
                    resolved_amount_fen=amount_fen,
                    bank=ctx.bank,
                    now=ctx.now,
                    approved_tier=decision.tier,
                    config=ctx.policy_config,
                    payments_locked=payments_locked,
                    scam_flags=ctx.scam_flags,
                    day_of_month=_day_of_month(resolved),
                )
            except (OverflowError, ValueError) as exc:
                raise ExecError("bank", "invalid_amount") from exc
            if checked.vetoed:
                codes = ",".join(reason.code for reason in checked.reasons)
                raise ExecError("policy_veto", codes)
            if checked.requires_reapproval:
                raise ExecError("reapproval_required", checked.tier)

        # 2. One-shot mandate binding for yellow/red writes.
        if spec.kind == "write" and checked.needs_mandate:
            if not mandate_id:
                raise ExecError("mandate", "missing")
            try:
                mandate = ctx.mandates.get(mandate_id, ctx.bank.customer_id)
            except MandateError as exc:
                raise ExecError("mandate", exc.code) from exc
            except (OverflowError, ValueError) as exc:
                raise ExecError("bank", "invalid_amount") from exc

            if TIER_RANK.get(mandate.tier, -1) < TIER_RANK[checked.tier]:
                raise ExecError("mandate", "tier_too_low")
            if checked.needs_mfa and not mandate.mfa_verified:
                raise ExecError("mandate", "mfa_missing")
            if checked.needs_co_approval and (
                not mandate.co_approval_required
                or mandate.co_approved_by is None
            ):
                raise ExecError("mandate", "co_approval_missing")

            payee = resolved.get("payee")
            payee_ref = "" if payee is None else str(payee)
            if tool == "redeem_product":
                # Same binding graph._issue_mandate uses for a redemption.
                payee_ref = "product:" + str(_required(resolved, "product"))
            try:
                ctx.mandates.verify_and_consume(
                    mandate_id,
                    customer_id=ctx.bank.customer_id,
                    session_id=ctx.session.session_id,
                    plan_id=ctx.session.plan_id,
                    step_id=step_id,
                    action=tool,
                    payee_ref=payee_ref,
                    amount_fen=amount_fen,
                    now=ctx.now,
                )
            except MandateError as exc:
                raise ExecError("mandate", exc.code) from exc
            except (OverflowError, ValueError) as exc:
                raise ExecError("bank", "invalid_amount") from exc

        # 3. Dispatch to the bank.
        if idempotency_key is None:
            idempotency_key = f"{ctx.session.plan_id}:{step_id}"

        try:
            result = _dispatch(
                tool,
                spec,
                resolved,
                step,
                checked,
                ctx,
                results,
                idempotency_key,
            )
        except ExecError:
            raise
        except BankError as exc:
            raise ExecError("bank", exc.code) from exc
        except OverflowError as exc:
            raise ExecError("bank", "invalid_amount") from exc
        except ValueError as exc:
            raise ExecError("bank", "invalid_amount") from exc
        except (KeyError, TypeError) as exc:
            raise ExecError("bank", "invalid_state") from exc

        result.setdefault("step_id", step_id)
        result.setdefault("tool", tool)

        undo = result.pop("undo", None)
        if isinstance(undo, Mapping) and spec.kind == "write":
            undo_tool = str(undo.get("tool") or "")
            if undo_tool and _store_undo(
                ctx, step_id, undo_tool, dict(undo.get("args") or {})
            ):
                result["undo_ref"] = {
                    "plan_id": ctx.session.plan_id,
                    "step_id": step_id,
                }

        # 4. Audit the run with token labels and fen only.
        _append_audit(
            ctx,
            AuditKind.RUN,
            _audit_payload(
                step_id,
                tool,
                decision,
                amount_fen,
                raw_args,
                mandate_id=mandate_id if spec.kind == "write" else None,
            ),
        )
        return result

    except ExecError as exc:
        _append_audit(
            ctx,
            AuditKind.VETO,
            _audit_payload(
                step_id, tool, decision, 0, raw_args, error=exc
            ),
        )
        raise


# ------------------------------------------------------------------------ undo


def _dispatch_undo(tool: str, args: Mapping[str, Any], ctx: StepContext) -> Any:
    bank = ctx.bank
    if tool == "cancel_transfer":
        return bank.cancel_transfer(
            str(args.get("transfer_id") or args.get("id") or ""), ctx.now
        )
    if tool == "cancel_scheduled":
        return bank.cancel_scheduled(
            str(args.get("scheduled_id") or args.get("id") or "")
        )
    if tool == "cancel_request":
        return bank.cancel_request(str(args.get("request_id") or args.get("id") or ""))
    if tool == "restore_subscription":
        return bank.restore_subscription(
            str(args.get("subscription_id") or args.get("id") or "")
        )
    if tool == "move_from_pot":
        return bank.move_from_pot(
            str(args.get("pot_name") or args.get("name") or ""),
            int(args.get("amount_fen") or args.get("amount") or 0),
        )
    if tool == "cancel_order":
        return bank.cancel_order(str(args.get("order_id") or args.get("id") or ""), ctx.now)
    if tool == "unfreeze_card":
        return bank.unfreeze_card(str(args.get("card_id") or args.get("id") or ""))
    if tool == "restore_card_limit":
        card_id = str(args.get("card_id") or args.get("id") or "")
        value = args.get("previous_limit_fen")
        if value is None:
            raise ExecError("bank", "invalid_state")
        return bank.set_card_limit(card_id, int(value))
    if tool == "close_virtual_card":
        return bank.close_virtual_card(str(args.get("card_id") or args.get("id") or ""))
    raise ExecError("unsupported", tool)


def _consume_undo_mandate(
    mandate_id: str,
    tool: str,
    args: Mapping[str, Any],
    ctx: StepContext,
) -> None:
    """Require and consume a fresh mandate for a yellow undo.

    The mandate must belong to this session/plan and be for the same action and
    a zero amount.  A card id may be bound as ``payee_ref`` (or left empty).
    """

    try:
        mandate = ctx.mandates.get(mandate_id, ctx.bank.customer_id)
        card_id = str(args.get("card_id") or args.get("id") or "")
        if (
            mandate.action != tool
            or mandate.amount_fen != 0
            or mandate.session_id != ctx.session.session_id
            or mandate.plan_id != ctx.session.plan_id
            or mandate.payee_ref not in {"", card_id}
        ):
            raise ExecError("mandate", "mismatch")
        ctx.mandates.verify_and_consume(
            mandate_id,
            customer_id=ctx.bank.customer_id,
            session_id=mandate.session_id,
            plan_id=mandate.plan_id,
            step_id=mandate.step_id,
            action=mandate.action,
            payee_ref=mandate.payee_ref,
            amount_fen=0,
            now=ctx.now,
        )
    except MandateError as exc:
        raise ExecError("mandate", exc.code) from exc
    except (OverflowError, ValueError) as exc:
        raise ExecError("bank", "invalid_amount") from exc


def latest_undo(
    conn: Any, customer_id: str, session_id: str
) -> dict[str, Any] | None:
    """The newest undo record of this session (used or not), for typed undo."""

    if conn is None:
        return None
    _ensure_undo_table(conn)
    rows = conn.execute(
        """
        SELECT * FROM exec_undo
        WHERE customer_id = ? AND substr(plan_id, 1, ?) = ?
        ORDER BY created_at DESC, rowid DESC
        """,
        (customer_id, len(session_id) + 1, f"{session_id}:"),
    ).fetchall()
    for row in rows:
        # Exact session: "abc" must not match plans of session "abc:x".
        if str(row["plan_id"]).rsplit(":", 1)[0] == session_id:
            return dict(row)
    return None


def undo_record(
    conn: Any, customer_id: str, plan_id: str, step_id: str
) -> dict[str, Any] | None:
    if conn is None:
        return None
    _ensure_undo_table(conn)
    row = conn.execute(
        "SELECT * FROM exec_undo WHERE customer_id = ? AND plan_id = ? AND step_id = ?",
        (customer_id, plan_id, step_id),
    ).fetchone()
    return dict(row) if row is not None else None


_SAFE_UNDOS_WHILE_PROTECTED = frozenset(
    {"cancel_transfer", "cancel_scheduled", "cancel_order", "cancel_request",
     "close_virtual_card", "restore_card_limit"}
)


def _payments_locked(conn: Any, customer_id: str, now: Any) -> bool:
    try:
        row = conn.execute(
            "SELECT locked_until FROM security_lockouts WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
    except Exception:  # noqa: BLE001 - no lockout table means no lock
        return False
    if row is None:
        return False
    try:
        return datetime.fromisoformat(_now_iso(now)) < datetime.fromisoformat(str(row[0]))
    except ValueError:
        return True  # an unreadable lock is treated as locked


def undo_step(
    plan_id: str,
    step_id: str,
    ctx: StepContext,
    mandate_id: str | None = None,
) -> dict[str, Any] | None:
    """Run the server-side undo registered for one executed step."""

    conn = _undo_conn(ctx)
    customer_id = ctx.bank.customer_id
    if conn is None:
        error = ExecError("undo", "not_found")
        _append_audit(
            ctx,
            AuditKind.VETO,
            {"step_id": step_id, "tool": "", "ok": False, "error": error.code},
        )
        raise error
    _ensure_undo_table(conn)
    row = conn.execute(
        "SELECT * FROM exec_undo WHERE customer_id = ? AND plan_id = ? AND step_id = ?",
        (customer_id, plan_id, step_id),
    ).fetchone()
    if row is None:
        error = ExecError("undo", "not_found")
        _append_audit(
            ctx,
            AuditKind.VETO,
            {"step_id": step_id, "tool": "", "ok": False, "error": error.code},
        )
        raise error

    tool = str(row["tool"])
    payload: dict[str, Any] = {
        "step_id": step_id,
        "tool": tool,
        "ok": True,
    }

    try:
        expected = _undo_sig(
            ctx.secret_key,
            str(row["customer_id"]),
            str(row["plan_id"]),
            str(row["step_id"]),
            tool,
            str(row["args_json"]),
            str(row["created_at"]),
        )
        if not hmac.compare_digest(expected, str(row["sig"])):
            raise ExecError("undo", "bad_signature")
        if int(row["used"]):
            raise ExecError("undo", "replayed")
        # While payments are locked or a scam warning is active, only undos that
        # remove risk may run; restarting debits or moving money waits.
        if tool not in _SAFE_UNDOS_WHILE_PROTECTED and (
            ctx.scam_flags or _payments_locked(conn, customer_id, ctx.now)
        ):
            raise ExecError("undo", "locked")

        try:
            args = json.loads(row["args_json"])
        except (TypeError, ValueError) as exc:
            raise ExecError("undo", "bad_signature") from exc
        if not isinstance(args, dict):
            raise ExecError("undo", "bad_signature")

        if args.get("amount_fen") is not None:
            payload["amount_fen"] = int(args["amount_fen"])
        if mandate_id:
            payload["mandate_id"] = mandate_id

        if tool == "unfreeze_card":
            if not mandate_id:
                raise ExecError("mandate_required", "unfreeze_card")
            _consume_undo_mandate(mandate_id, tool, args, ctx)
        if tool == "restore_card_limit":
            # Putting a higher limit back loosens a protection: that is a new
            # request with its own policy check, never an undo.
            card_id = str(args.get("card_id") or args.get("id") or "")
            previous = args.get("previous_limit_fen")
            current = next(
                (c.online_limit_fen for c in ctx.bank.list_cards() if c.id == card_id), None
            )
            if previous is None or current is None or int(previous) > int(current):
                raise ExecError("undo", "loosens_protection")

        with conn:
            cursor = conn.execute(
                """
                UPDATE exec_undo SET used = 1
                WHERE customer_id = ? AND plan_id = ? AND step_id = ? AND used = 0
                """,
                (customer_id, plan_id, step_id),
            )
            if cursor.rowcount != 1:
                raise ExecError("undo", "replayed")

        try:
            raw = _dispatch_undo(tool, args, ctx)
        except BankError:
            # Nothing changed (e.g. the window closed): free the record so a
            # retry gets the real reason, not "already undone".
            with conn:
                conn.execute(
                    """
                    UPDATE exec_undo SET used = 0
                    WHERE customer_id = ? AND plan_id = ? AND step_id = ?
                    """,
                    (customer_id, plan_id, step_id),
                )
            raise
    except ExecError as exc:
        payload["ok"] = False
        payload["error"] = exc.code
        if exc.detail:
            payload["error_detail"] = exc.detail
        _append_audit(ctx, AuditKind.VETO, payload)
        raise
    except BankError as exc:
        error = ExecError("bank", exc.code)
        payload["ok"] = False
        payload["error"] = error.code
        _append_audit(ctx, AuditKind.VETO, payload)
        raise error from exc
    except OverflowError as exc:
        error = ExecError("bank", "invalid_amount")
        payload["ok"] = False
        payload["error"] = error.code
        _append_audit(ctx, AuditKind.VETO, payload)
        raise error from exc
    except ValueError as exc:
        error = ExecError("bank", "invalid_amount")
        payload["ok"] = False
        payload["error"] = error.code
        _append_audit(ctx, AuditKind.VETO, payload)
        raise error from exc
    except (KeyError, TypeError) as exc:
        error = ExecError("bank", "invalid_state")
        payload["ok"] = False
        payload["error"] = error.code
        _append_audit(ctx, AuditKind.VETO, payload)
        raise error from exc

    _append_audit(ctx, AuditKind.UNDO, payload)
    return {"ok": True, "raw": raw, "tool": tool}
