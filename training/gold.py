"""Code-generated gold planner outputs for the training pipeline.

The scenario YAML describes the *intent* of each template; this module turns
that intent into a planner output that only uses semantic-token labels minted
by the real tokenizer/vault for the sample.  Nothing here asks an LLM for a
label, so the target format can never drift from ``docs/CONTRACTS.md``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import random
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from agent.prompts import PLANNER_SYSTEM_PROMPT, render_planner_input
from bank import db
from bank.api import Bank
from bank.seed import INJECTION_MEMO, generate
from security.dag import validate_plan
from security.scam import detect as detect_scam
from security.tokens import (
    Session,
    TokenVault,
    build_context,
    tokenize_untrusted,
    tokenize_user_text,
)
from training.templates import LANGS, render_sample

_SCENARIOS_PATH = Path(__file__).resolve().parents[1] / "data_spec" / "scenarios.yaml"

# The merchant identities are fixed in bank/merchants.py and bank/seed.py.
_FLORIST = "花店"
_CAKE_SHOP = "蛋糕店"


# Intent classes a paraphrase may not ADD: entity signatures can't see
# "don't send 300 to Mom". Classes, not words, so 取消 -> "cancel" is fine and
# casual negation ("别漏了", "不用再问") is not a false alarm.
_CANCEL_RE = re.compile(r"取消|撤销|退订|退款|cancel|undo|refund|\bstop\b", re.IGNORECASE)
_NEG_MONEY_RE = re.compile(
    r"(不要|别|不用|不想|don'?t|do not|never)\s*(再|要)?\s*(给|转|汇|打钱|付|send|transfer|pay|move)",
    re.IGNORECASE,
)
_BACK_REFERENCE_RE = re.compile(
    r"上次|上回|上一次|老样子|照旧|一样|same as|last time|again|as usual|usual",
    re.IGNORECASE,
)


def _intent_delta(
    template: str, paraphrase: str, scenario_id: str | None = None
) -> str | None:
    """Why a paraphrase changes intent, or None. Code check, not an LLM."""

    for name, pattern in (("cancel", _CANCEL_RE), ("negated payment", _NEG_MONEY_RE)):
        if pattern.search(paraphrase) and not pattern.search(template):
            return f"paraphrase adds {name} intent"
    if (
        scenario_id == "T6_missing_amount"
        and _BACK_REFERENCE_RE.search(paraphrase)
        and not _BACK_REFERENCE_RE.search(template)
    ):
        return "paraphrase adds a back-reference to a previous amount"
    if detect_scam(template).flags != detect_scam(paraphrase).flags:
        return "paraphrase changes scam signals"
    return None


class GoldError(ValueError):
    """A scenario cannot be represented with the tokens actually minted."""


class TeacherMismatch(GoldError):
    """A teacher paraphrase does not tokenize to the template's entities."""


# ---------------------------------------------------------------------------
# Small output builders
# ---------------------------------------------------------------------------


def _step(
    step_id: str,
    tool: str,
    args: dict[str, Any] | None = None,
    after: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": step_id,
        "tool": tool,
        "args": args or {},
        "after": list(after or []),
    }


def _plan(steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": "plan", "steps": steps}


def _clarify(question: str, options: list[str] | None = None) -> dict[str, Any]:
    return {"kind": "clarify", "question": question, "options": list(options or [])}


def _refuse(reason: str) -> dict[str, Any]:
    return {"kind": "refuse", "reason": reason}


def _scam(signals: list[str]) -> dict[str, Any]:
    return {"kind": "scam", "signals": signals}


# ---------------------------------------------------------------------------
# Token-map helpers
# ---------------------------------------------------------------------------


def _value_label(value: Any) -> str:
    if isinstance(value, str):
        return value
    label = getattr(value, "label", None)
    if isinstance(label, str):
        return label
    if isinstance(value, dict):
        for key in ("label", "token"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    raise GoldError(f"value {value!r} is not a token label or token entry")


def _value_kind(value: Any) -> str | None:
    kind = getattr(value, "kind", None)
    if isinstance(kind, str):
        return kind
    if isinstance(value, dict):
        candidate = value.get("kind")
        if isinstance(candidate, str):
            return candidate
    return None


def _value_ref(value: Any) -> Any:
    ref = getattr(value, "real_ref", None)
    if ref is not None:
        return ref
    if isinstance(value, dict):
        return value.get("real_ref")
    return None


def _flatten(tokens: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    seen: set[str] = set()
    for value in tokens.values():
        group = value if isinstance(value, (list, tuple, set)) else [value]
        for item in group:
            if item is None:
                continue
            try:
                label = _value_label(item)
            except GoldError:
                continue
            if label in seen:
                continue
            seen.add(label)
            values.append(item)
    return values


def _role_values(tokens: dict[str, Any], role: str) -> list[Any]:
    """Return the token entries for a semantic role.

    ``make_sample`` passes role keys (``payee``, ``amount``, ...).  The
    fallbacks keep ``build_gold`` usable with a raw ``label -> TokenEntry``
    mapping produced by the tokenizer.
    """

    if role in tokens:
        value = tokens[role]
        if value is None:
            return []
        return list(value) if isinstance(value, (list, tuple)) else [value]

    values = _flatten(tokens)
    kinds = {
        "payee": "PAYEE",
        "payee2": "PAYEE",
        "duplicates": "PAYEE",
        "amount": "AMT",
        "amount2": "AMT",
        "date": "DATE",
        "sub": "SUB",
        "unused_subs": "SUB",
        "card": "CARD",
        "product": "PROD",
        "florist": "MERCHANT",
        "cake": "MERCHANT",
    }

    if role in {"duplicates", "unused_subs"}:
        kind = kinds[role]
        return [item for item in values if _value_kind(item) == kind]

    if role == "florist":
        merchants = [item for item in values if _value_kind(item) == "MERCHANT"]
        return [item for item in merchants if _value_ref(item) == _FLORIST] or merchants[:1]

    if role == "cake":
        merchants = [item for item in values if _value_kind(item) == "MERCHANT"]
        return [item for item in merchants if _value_ref(item) == _CAKE_SHOP] or merchants[-1:]

    kind = kinds.get(role)
    if kind is None:
        return []
    matches = [item for item in values if _value_kind(item) == kind]
    if role in {"payee2", "amount2"}:
        return matches[1:2]
    return matches[:1]


def _label(tokens: dict[str, Any], role: str) -> str:
    values = _role_values(tokens, role)
    if not values:
        raise GoldError(f"no token minted for role {role!r}")
    return _value_label(values[0])


def _labels(tokens: dict[str, Any], role: str) -> list[str]:
    return [_value_label(value) for value in _role_values(tokens, role)]


def _slot_map(slots: Any) -> dict[str, Any]:
    if isinstance(slots, dict):
        return slots
    data: dict[str, Any] = {}
    if hasattr(slots, "slots"):
        data.update(getattr(slots, "slots") or {})
    if hasattr(slots, "meta"):
        data.update(getattr(slots, "meta") or {})
    if hasattr(slots, "lang"):
        data["lang"] = getattr(slots, "lang")
    return data


def _slot(slots: dict[str, Any], key: str, default: Any = None) -> Any:
    value = slots.get(key)
    return default if value is None else value


def _scenario_id(scenario: dict[str, Any] | str) -> str:
    if isinstance(scenario, str):
        return scenario
    if isinstance(scenario, dict) and isinstance(scenario.get("id"), str):
        return scenario["id"]
    raise GoldError(f"scenario must be an id or a mapping with an id: {scenario!r}")


def _is_en(lang: str) -> bool:
    return lang == "en"


# ---------------------------------------------------------------------------
# Gold builder
# ---------------------------------------------------------------------------


def build_gold(
    scenario: dict[str, Any] | str,
    tokens: dict[str, Any],
    slots: Any,
) -> dict[str, Any]:
    """Build the planner output for one scenario from minted token labels."""

    scenario_id = _scenario_id(scenario)
    slot = _slot_map(slots)
    lang = str(slot.get("lang") or "zh")

    if scenario_id == "T1_transfer_by_name":
        # T1: the payee is already a PAYEE token, so only can_cover + transfer remain.
        amount = _label(tokens, "amount")
        payee = _label(tokens, "payee")
        return _plan(
            [
                _step("s1", "can_cover", {"amount": amount}),
                _step(
                    "s2",
                    "transfer",
                    {"payee": payee, "amount": amount},
                    ["s1"],
                ),
            ]
        )

    if scenario_id == "T2_duplicate_name":
        # T2: two contacts share a name -> ask which of the two PAYEE tokens.
        options = _labels(tokens, "duplicates")
        if len(options) < 2:
            options = _labels(tokens, "payee") + [
                label for label in _labels(tokens, "payee2") if label not in options
            ]
        if _is_en(lang):
            question = "There are two contacts with the same name. Which one should I transfer to?"
        else:
            question = "有两个同名联系人，请问要转给哪一位？"
        return _clarify(question, options)

    if scenario_id == "T3_repeat_last":
        # T3: list the payee's history, then transfer the amount it returns.
        payee = _label(tokens, "payee")
        return _plan(
            [
                _step("s1", "list_transactions", {"payee": payee}),
                _step(
                    "s2",
                    "transfer",
                    {"payee": payee, "amount": "$s1.amount"},
                    ["s1"],
                ),
            ]
        )

    if scenario_id == "T4_scheduled":
        # T4: the day comes from the template, so a single schedule_transfer is enough.
        return _plan(
            [
                _step(
                    "s1",
                    "schedule_transfer",
                    {
                        "payee": _label(tokens, "payee"),
                        "amount": _label(tokens, "amount"),
                        "day_of_month": int(_slot(slot, "day_of_month", 15)),
                    },
                )
            ]
        )

    if scenario_id == "T5_split_bill":
        # T5: request the total from each friend, split across both friends + the user.
        payees = _labels(tokens, "payee")
        payees += [label for label in _labels(tokens, "payee2") if label not in payees]
        amount = _label(tokens, "amount")
        divide_by = len(payees) + 1
        steps = [
            _step(
                f"s{index}",
                "request_payment",
                {"payee": payee, "amount": amount, "divide_by": divide_by},
            )
            for index, payee in enumerate(payees, start=1)
        ]
        return _plan(steps)

    if scenario_id == "T6_missing_amount":
        # T6: no amount token exists, so the only safe output is a clarification.
        if _is_en(lang):
            question = "How much would you like to transfer?"
        else:
            question = "请问要转多少金额？"
        return _clarify(question)

    if scenario_id == "T7_insufficient":
        # T7: can_cover is checked first; code will stop the plan when it returns no.
        amount = _label(tokens, "amount")
        payee = _label(tokens, "payee")
        return _plan(
            [
                _step("s1", "can_cover", {"amount": amount}),
                _step(
                    "s2",
                    "transfer",
                    {"payee": payee, "amount": amount},
                    ["s1"],
                ),
            ]
        )

    if scenario_id == "T8_two_payees_two_amounts":
        # T8: two payees with different amounts -> check and transfer each separately.
        amount = _label(tokens, "amount")
        amount2 = _label(tokens, "amount2")
        return _plan(
            [
                _step("s1", "can_cover", {"amount": amount}),
                _step(
                    "s2",
                    "transfer",
                    {"payee": _label(tokens, "payee"), "amount": amount},
                    ["s1"],
                ),
                _step("s3", "can_cover", {"amount": amount2}),
                _step(
                    "s4",
                    "transfer",
                    {"payee": _label(tokens, "payee2"), "amount": amount2},
                    ["s3"],
                ),
            ]
        )

    if scenario_id == "T9_amount_change_no_payee":
        # T9: amount-only follow-up without a payee -> ask for the full request.
        if _is_en(lang):
            question = (
                'Please say the full request again, with who and how much, '
                'for example "send mom 300".'
            )
        else:
            question = "请把收款人和金额完整说一遍，比如“给妈妈转300”。"
        return _clarify(question, [])

    if scenario_id == "B1_where_money_went":
        # B1: a single monthly spending report.
        return _plan([_step("s1", "spending_report", {"period": "this_month"})])

    if scenario_id == "B2_category_compare":
        # B2: compare this month with last month -> two independent reports.
        return _plan(
            [
                _step("s1", "spending_report", {"period": "this_month"}),
                _step("s2", "spending_report", {"period": "last_month"}),
            ]
        )

    if scenario_id == "B3_anomaly":
        # B3: the anomaly detector already scans the card ledger.
        return _plan([_step("s1", "detect_anomalies")])

    if scenario_id == "B4_yearly_report":
        # B4: yearly report, then chart it.
        return _plan(
            [
                _step("s1", "spending_report", {"period": "this_year"}),
                _step("s2", "sandbox_chart", after=["s1"]),
            ]
        )

    if scenario_id == "B5_reconcile_business":
        # B5: the merchant is already a PAYEE token; list its transactions.
        return _plan(
            [_step("s1", "list_transactions", {"payee": _label(tokens, "payee")})]
        )

    if scenario_id == "B6_bill_pay_unsupported":
        # B6: there is no bill-payment tool -> refuse instead of moving money.
        return _refuse("out_of_scope")

    if scenario_id == "W1_returns":
        # W1: portfolio holdings already carry the return figures.
        return _plan([_step("s1", "get_portfolio")])

    if scenario_id == "W2_compare_products":
        # W2: read the product list; comparison is done by code from the rows.
        return _plan([_step("s1", "list_products")])

    if scenario_id == "W3_subscribe":
        # W3: list products, check funds, then subscribe to the stable PROD token.
        amount = _label(tokens, "amount")
        return _plan(
            [
                _step("s1", "list_products"),
                _step("s2", "can_cover", {"amount": amount}),
                _step(
                    "s3",
                    "subscribe_product",
                    {"product": _label(tokens, "product"), "amount": amount},
                    ["s2"],
                ),
            ]
        )

    if scenario_id == "W4_advice_request":
        # W4: show products only; the no-guarantee wording is a reply constraint.
        return _plan([_step("s1", "list_products")])

    if scenario_id == "C1_freeze":
        # C1: the CARD token is already in context; list then freeze it.
        card = _label(tokens, "card")
        return _plan(
            [
                _step("s1", "list_cards"),
                _step("s2", "freeze_card", {"card": card}, ["s1"]),
            ]
        )

    if scenario_id == "C2_report_lost":
        # C2: freeze first, then report the same CARD lost.
        card = _label(tokens, "card")
        return _plan(
            [
                _step("s1", "list_cards"),
                _step("s2", "freeze_card", {"card": card}, ["s1"]),
                _step("s3", "report_card_lost", {"card": card}, ["s2"]),
            ]
        )

    if scenario_id == "C3_limit":
        # C3: set the online limit on the existing CARD token.
        return _plan(
            [
                _step("s1", "list_cards"),
                _step(
                    "s2",
                    "set_card_limit",
                    {
                        "card": _label(tokens, "card"),
                        "amount": _label(tokens, "amount"),
                    },
                    ["s1"],
                ),
            ]
        )

    if scenario_id == "C4_virtual_card":
        # C4: no lookup needed; create a virtual card with the stated limit.
        return _plan(
            [
                _step(
                    "s1",
                    "create_virtual_card",
                    {"amount": _label(tokens, "amount")},
                )
            ]
        )

    if scenario_id == "S1_list":
        # S1: the user explicitly asks for the subscription list.
        return _plan([_step("s1", "list_subscriptions")])

    if scenario_id == "S2_cancel_unused":
        # S2: list, then one cancel node per unused SUB token.
        steps = [_step("s1", "list_subscriptions")]
        for index, sub in enumerate(_labels(tokens, "unused_subs"), start=2):
            steps.append(
                _step(
                    f"s{index}",
                    "cancel_subscription",
                    {"sub": sub},
                    ["s1"],
                )
            )
        return _plan(steps)

    if scenario_id == "S3_cancel_named":
        # S3: list then cancel the named SUB token.
        return _plan(
            [
                _step("s1", "list_subscriptions"),
                _step(
                    "s2",
                    "cancel_subscription",
                    {"sub": _label(tokens, "sub")},
                    ["s1"],
                ),
            ]
        )

    if scenario_id == "S4_renewal_reminder":
        # S4: list subscriptions, then set a renewal reminder.
        return _plan(
            [
                _step("s1", "list_subscriptions"),
                _step(
                    "s2",
                    "set_reminder",
                    {"note": "subscription renewal"},
                    ["s1"],
                ),
            ]
        )

    if scenario_id == "S5_cancel_advice":
        # S5: advice only -> list subscriptions, never cancel anything.
        return _plan([_step("s1", "list_subscriptions")])

    if scenario_id == "O1_birthday":
        # O1: CONTRACTS.md shape: date, can_cover, pot, then two parallel orders.
        payee = _label(tokens, "payee")
        amount = _label(tokens, "amount")
        return _plan(
            [
                _step(
                    "s1",
                    "get_contact_date",
                    {"payee": payee, "what": "birthday"},
                ),
                _step("s2", "can_cover", {"amount": amount}),
                _step(
                    "s3",
                    "move_to_pot",
                    {"pot_name": "birthday", "amount": amount},
                    ["s2"],
                ),
                _step(
                    "s4",
                    "merchant_order",
                    {
                        "merchant": _label(tokens, "florist"),
                        "item": "bouquet_m",
                        "deliver": {"date": "$s1.date", "offset_days": -2},
                    },
                    ["s1", "s3"],
                ),
                _step(
                    "s5",
                    "merchant_order",
                    {
                        "merchant": _label(tokens, "cake"),
                        "item": "cake_8in",
                        "deliver": {"date": "$s1.date", "offset_days": -2},
                    },
                    ["s1", "s3"],
                ),
            ]
        )

    if scenario_id == "O2_payday_routine":
        # O2: two schedules with the stated amount; "the rest" has no token, so it is omitted.
        amount = _label(tokens, "amount")
        days = list(_slot(slot, "day_of_months", []) or [])
        day_one = int(days[0]) if days else 1
        day_two = int(days[1]) if len(days) > 1 else 5
        return _plan(
            [
                _step(
                    "s1",
                    "schedule_transfer",
                    {
                        "payee": _label(tokens, "payee"),
                        "amount": amount,
                        "day_of_month": day_one,
                    },
                ),
                _step(
                    "s2",
                    "schedule_transfer",
                    {
                        "payee": _label(tokens, "payee2"),
                        "amount": amount,
                        "day_of_month": day_two,
                    },
                ),
            ]
        )

    if scenario_id == "O3_trip_prep":
        # O3: virtual card and monthly spending report are independent.
        return _plan(
            [
                _step(
                    "s1",
                    "create_virtual_card",
                    {"amount": _label(tokens, "amount")},
                ),
                _step("s2", "spending_report", {"period": "this_month"}),
            ]
        )

    if scenario_id == "O4_clean_and_save":
        # O4: list, cancel each unused SUB, then move the stated amount to savings.
        steps = [_step("s1", "list_subscriptions")]
        dependencies = ["s1"]
        for index, sub in enumerate(_labels(tokens, "unused_subs"), start=2):
            step_id = f"s{index}"
            steps.append(
                _step(
                    step_id,
                    "cancel_subscription",
                    {"sub": sub},
                    ["s1"],
                )
            )
            dependencies.append(step_id)
        steps.append(
            _step(
                f"s{len(steps) + 1}",
                "move_to_pot",
                {"pot_name": "savings", "amount": _label(tokens, "amount")},
                dependencies,
            )
        )
        return _plan(steps)

    if scenario_id == "X1_memo_injection":
        # X1: the memo is untrusted data -> summarise transactions, never transfer.
        return _plan(
            [_step("s1", "list_transactions", {"period": "last_7_days"})]
        )

    if scenario_id == "X2_raw_account_in_text":
        # X2: a RAW number cannot be a payee -> add and verify the payee first.
        if _is_en(lang):
            question = "Please add and verify this payee first, then I can send the transfer."
        else:
            question = "请先添加并验证这个收款人，然后再帮你转账。"
        return _clarify(question)

    if scenario_id == "X3_limit_splitting":
        # X3: the planner emits all five transfers; policy catches the split at steps 4-5.
        payee = _label(tokens, "payee")
        amount = _label(tokens, "amount")
        return _plan(
            [
                _step(
                    f"s{index}",
                    "transfer",
                    {"payee": payee, "amount": amount},
                )
                for index in range(1, 6)
            ]
        )

    if scenario_id == "X4_safe_account_scam":
        # X4: police/safe-account story -> scam flow, no money tools.
        return _scam(["safe_account", "police_claim"])

    if scenario_id == "X5_impersonation":
        # X5: family/supplier changed account -> impersonation scam, no money tools.
        return _scam(["impersonation", "changed_account", "urgency"])

    if scenario_id == "X6_other_user_data":
        # X6: asking for another person's bank data -> refuse.
        return _refuse("not_account_owner")

    if scenario_id == "X7_prompt_override":
        # X7: admin-mode/policy override -> refuse.
        return _refuse("not_supported")

    if scenario_id == "X8_out_of_scope":
        # X8: non-banking request -> refuse.
        return _refuse("out_of_scope")

    if scenario_id == "X10_bare_confirm":
        # X10: a bare confirmation has nothing to act on -> refuse.
        return _refuse("out_of_scope")

    if scenario_id == "X9_fake_approval":
        # X9: "already confirmed" is ignored; produce the normal checked transfer.
        amount = _label(tokens, "amount")
        payee = _label(tokens, "payee")
        return _plan(
            [
                _step("s1", "can_cover", {"amount": amount}),
                _step(
                    "s2",
                    "transfer",
                    {"payee": payee, "amount": amount},
                    ["s1"],
                ),
            ]
        )

    raise GoldError(f"unknown scenario id: {scenario_id!r}")


# ---------------------------------------------------------------------------
# Scenario metadata and sample construction
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_scenarios() -> dict[str, dict[str, Any]]:
    """Load ``data_spec/scenarios.yaml`` keyed by scenario id."""

    with _SCENARIOS_PATH.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return {scenario["id"]: scenario for scenario in data["scenarios"]}


def scenario_ids() -> tuple[str, ...]:
    return tuple(load_scenarios())


def _secret(seed: int) -> bytes:
    return hashlib.sha256(f"anban-gold-v1:{seed}".encode("utf-8")).digest()


def _session(seed: int, scenario_id: str, persona_id: str, lang: str, variation: str) -> Session:
    digest = hashlib.sha256(
        f"{seed}|{scenario_id}|{persona_id}|{lang}|{variation}".encode("utf-8")
    ).hexdigest()
    return Session(
        session_id=digest[:32],
        plan_id=digest[32:48],
        customer_id=f"cust_{persona_id}",
    )


def _ref_key(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


@functools.lru_cache(maxsize=64)
def _template_bank(persona_id: str, seed: int, today: date) -> tuple[Any, Any]:
    """Generate and load a persona once; samples get a private copy of the DB."""

    state = generate(persona_id, seed, today)
    conn = db.connect(":memory:")
    db.create_schema(conn)
    db.load_state(conn, state)
    return conn, state


def _load_bank(persona_id: str, seed: int, today: date) -> tuple[Bank, Any]:
    template, state = _template_bank(persona_id, seed, today)
    conn = db.connect(":memory:")
    template.backup(conn)
    return Bank(conn, state.customer.id), state


def _token_map(
    vault: TokenVault,
    session: Session,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Index minted entries by the semantic role the gold builder needs."""

    entries = vault.entries(session)
    by_kind_ref = {(entry.kind, _ref_key(entry.real_ref)): entry for entry in entries}

    def find(kind: str, ref: Any) -> Any:
        if ref is None:
            return None
        return by_kind_ref.get((kind, _ref_key(ref)))

    tokens: dict[str, Any] = {}
    roles = {
        "payee": ("PAYEE", meta.get("payee_id")),
        "payee2": ("PAYEE", meta.get("payee2_id")),
        "amount": ("AMT", meta.get("amount_fen")),
        "amount2": ("AMT", meta.get("amount2_fen")),
        "date": ("DATE", meta.get("date_iso")),
        "sub": ("SUB", meta.get("sub_id")),
        "card": ("CARD", meta.get("card_id")),
        "product": ("PROD", meta.get("product_id")),
    }
    for role, (kind, ref) in roles.items():
        entry = find(kind, ref)
        if entry is not None:
            tokens[role] = entry

    for role, kind, ref in (
        ("florist", "MERCHANT", _FLORIST),
        ("cake", "MERCHANT", _CAKE_SHOP),
    ):
        entry = find(kind, ref)
        if entry is not None:
            tokens[role] = entry

    duplicates = [
        entry
        for entry in (
            find("PAYEE", contact_id)
            for contact_id in meta.get("duplicate_payee_ids", [])
        )
        if entry is not None
    ]
    tokens["duplicates"] = duplicates

    unused = [
        entry
        for entry in (
            find("SUB", sub_id) for sub_id in meta.get("unused_sub_ids", [])
        )
        if entry is not None
    ]
    tokens["unused_subs"] = unused

    # Also expose every label, so a caller can pass the map around without
    # losing the ability to look up a token by its label.
    for entry in entries:
        tokens.setdefault(entry.label, entry)
    return tokens


_TEACHER_ENTITY_KINDS = frozenset(
    {"PAYEE", "AMT", "DATE", "SUB", "MERCHANT", "RAW"}
)


def _entity_signature(
    text: str,
    bank: Any,
    bank_seed: int,
    scenario_id: str,
    persona_id: str,
    lang: str,
    tag: str,
    today: date,
) -> list[tuple[str, str]]:
    """Sorted multiset of ``(kind, real_ref)`` for the teacher-guarded kinds."""

    vault = TokenVault(_secret(bank_seed))
    session = _session(bank_seed, scenario_id, persona_id, lang, tag)
    _, entries = tokenize_user_text(text, bank, vault, session, today)
    return sorted(
        (entry.kind, str(_ref_key(entry.real_ref)))
        for entry in entries
        if entry.kind in _TEACHER_ENTITY_KINDS
    )


def render_seed(
    scenario_id: str,
    persona_id: str,
    lang: str,
    variation: str,
    seed: int,
    today: date,
    *,
    sample_seed: int,
) -> tuple[str, list[str]]:
    """Render the template utterance plus the slot surfaces it inserts.

    The same ``sample_seed`` makes this exactly the text ``make_sample`` would
    render, so the teacher can paraphrase it without changing the slot values.
    """

    scenarios = load_scenarios()
    if scenario_id not in scenarios:
        raise GoldError(f"unknown scenario id: {scenario_id!r}")
    if lang not in LANGS:
        raise GoldError(f"unknown language bucket: {lang!r}")
    if isinstance(today, datetime):
        today = today.date()

    bank_seed = seed + (sample_seed % 5)
    _, state = _template_bank(persona_id, bank_seed, today)
    rng = random.Random(sample_seed)
    rendered = render_sample(scenario_id, state, today, rng, lang, variation)

    keep: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\{([a-z0-9_]+)\}", rendered.template):
        value = rendered.slots.get(match.group(1))
        if value and value not in seen:
            seen.add(value)
            keep.append(value)
    return rendered.text, keep


def make_sample(
    scenario_id: str,
    persona_id: str,
    lang: str,
    variation: str,
    seed: int,
    today: date,
    *,
    with_runtime: bool = False,
    sample_seed: int = 0,
    text_override: str | None = None,
) -> dict[str, Any]:
    """Build one prompt + gold sample for a scenario/persona/language.

    ``sample_seed`` drives phrasing/slot variety; the bank state cycles through
    five seeds derived from it so balances and histories also differ.  A
    ``text_override`` (a teacher paraphrase) is only accepted when it tokenizes
    to exactly the same entities as the template text; gold always comes from
    the template, never from the paraphrase.
    """

    scenarios = load_scenarios()
    if scenario_id not in scenarios:
        raise GoldError(f"unknown scenario id: {scenario_id!r}")
    if lang not in LANGS:
        raise GoldError(f"unknown language bucket: {lang!r}")
    if isinstance(today, datetime):
        today = today.date()

    scenario = scenarios[scenario_id]
    bank_seed = seed + (sample_seed % 5)
    bank, state = _load_bank(persona_id, bank_seed, today)
    vault = TokenVault(_secret(bank_seed))
    session = _session(bank_seed, scenario_id, persona_id, lang, variation)

    rng = random.Random(sample_seed)
    rendered = render_sample(scenario_id, state, today, rng, lang, variation)

    if text_override is not None:
        template_signature = _entity_signature(
            rendered.text,
            bank,
            bank_seed,
            scenario_id,
            persona_id,
            lang,
            "teacher-template",
            today,
        )
        override_signature = _entity_signature(
            text_override,
            bank,
            bank_seed,
            scenario_id,
            persona_id,
            lang,
            "teacher-override",
            today,
        )
        if template_signature != override_signature:
            raise TeacherMismatch(
                f"{scenario_id}/{persona_id}/{lang}: template entities "
                f"{template_signature} != override entities {override_signature}"
            )
        intent_delta = _intent_delta(rendered.text, text_override, scenario_id)
        if intent_delta:
            raise TeacherMismatch(f"{scenario_id}/{persona_id}/{lang}: {intent_delta}")
        source_text = text_override
    else:
        source_text = rendered.text

    tokenized, entries = tokenize_user_text(
        source_text, bank, vault, session, today
    )
    context_lines = build_context(bank, vault, session, today)

    if scenario_id == "X1_memo_injection":
        # The seeded inbound memo is untrusted data; tokenize it separately and
        # show it to the planner as a memo, never as a user instruction.
        memo = next(
            (txn.memo for txn in state.transactions if txn.memo == INJECTION_MEMO),
            INJECTION_MEMO,
        )
        tokenized_memo, memo_entries = tokenize_untrusted(
            memo, bank, vault, session, today
        )
        entries.extend(memo_entries)
        context_lines.append("Memo (untrusted): " + tokenized_memo)

    tokens = _token_map(vault, session, rendered.meta)
    slots = dict(rendered.slots)
    slots.update(rendered.meta)
    slots["lang"] = lang

    gold = build_gold(scenario, tokens, slots)
    known_tokens = vault.known_tokens(session)
    validate_plan(gold, known_tokens)

    role_names = (
        "payee",
        "payee2",
        "amount",
        "amount2",
        "date",
        "sub",
        "card",
        "product",
        "florist",
        "cake",
    )
    meta: dict[str, Any] = {
        "scenario_id": scenario_id,
        "persona_id": persona_id,
        "lang": lang,
        "variation": variation,
        "seed": seed,
        "sample_seed": sample_seed,
        "bank_seed": bank_seed,
        "today": today.isoformat(),
        "template": rendered.template,
        "tokenized": tokenized,
        "kind": gold["kind"],
        "token_roles": {
            role: _value_label(tokens[role])
            for role in role_names
            if role in tokens
        },
        "known_tokens": known_tokens,
    }
    if text_override is not None:
        meta["variation"] = "teacher"
        meta["teacher_seed"] = rendered.text
    if with_runtime:
        meta["_runtime"] = {
            "bank": bank,
            "vault": vault,
            "session": session,
            "state": state,
        }

    return {
        "system": PLANNER_SYSTEM_PROMPT,
        "user": render_planner_input(tokenized, context_lines),
        "assistant": json.dumps(
            gold, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "meta": meta,
    }
