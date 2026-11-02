"""Standalone generator for the R3 planner training data (v3).

This module owns only the *new* v3 examples.  It does not touch the legacy
dataset under ``training/data`` and it never calls an LLM: every assistant target
is built by code from the semantic tokens the real tokenizer mints for the real
seeded bank state.

``generate_samples`` returns the same ShareGPT shape the existing builder writes:
``{"messages": [...], "meta": "<json string>"}``.  ``known_tokens`` is stripped
from the emitted meta; it is only used internally to validate every gold output
with :func:`security.dag.validate_plan`.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

from agent.prompts import R3_SYSTEM_PROMPT, render_planner_input
from bank import db
from bank.api import Bank
from bank.redemption import REDEEMABLE_PRODUCTS
from bank.seed import generate
from security.dag import parse_planner_output, validate_plan
from security.tokens import (
    SUB_ALIASES,
    Session,
    TokenEntry,
    TokenVault,
    build_context,
    tokenize_user_text,
)

_ROOT = Path(__file__).resolve().parents[1]
_SEGMENTS_PATH = _ROOT / "data_spec" / "customer_segments.yaml"

LANGS: tuple[str, ...] = ("zh", "en", "mixed")
BILL_CATEGORIES: tuple[str, ...] = (
    "water",
    "electricity",
    "gas",
    "mobile",
    "broadband",
    "property",
)
MERCHANT_BLOCK_CATEGORIES: tuple[str, ...] = (
    "gambling",
    "crypto",
    "lottery",
    "adult",
    "cash_like",
)
CARD_CONTROLS: tuple[str, ...] = ("online", "abroad", "atm", "contactless")
HELP_TOPICS: tuple[str, ...] = (
    "delayed_arrival",
    "recall",
    "deposit_insurance",
    "fx_quota",
    "complaints",
    "anti_fraud",
    "large_deposit_cd",
    "suitability",
    "personal_pension",
    "account_tiers",
    "installment_rate",
    "account_restriction",
)

# The eleven registry tools that were still ``trained=False`` when v3 started,
# plus the basic balance/contact reads the model must not forget.
REQUIRED_POSITIVE_TOOLS: frozenset[str] = frozenset(
    {
        "list_bills",
        "pay_bill",
        "installment_quote",
        "deposit_quote",
        "forecast_cashflow",
        "help_answer",
        "list_transfers",
        "verify_payee",
        "set_card_control",
        "block_merchant",
        "unblock_merchant",
        "get_balance",
        "list_contacts",
        "redeem_product",
    }
)

# A six-variant spread for tools that only have one real semantic variant.  The
# variants mainly drive the deterministic slot seed, so each tool contributes
# enough source families for an 80/10/10 split with per-tool test coverage.
_WORDING_VARIANTS: tuple[str, ...] = ("a", "b", "c", "d", "e", "f")


# ---------------------------------------------------------------------------
# Intent specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntentSpec:
    """One v3 intent family and the variants that must be covered."""

    name: str
    scenario: str
    variants: tuple[str, ...]
    negative: bool = False
    weight: int = 1


POSITIVE_INTENTS: tuple[IntentSpec, ...] = (
    IntentSpec("list_bills", "V3_list_bills", _WORDING_VARIANTS),
    IntentSpec("pay_bill", "V3_pay_bill", BILL_CATEGORIES),
    IntentSpec(
        "installment_quote",
        "V3_installment_quote",
        ("3", "6", "12", "24", "3b", "6b"),
    ),
    IntentSpec(
        "deposit_quote",
        "V3_deposit_quote",
        ("3", "6", "12", "24", "36", "60"),
    ),
    IntentSpec(
        "forecast_cashflow",
        "V3_forecast_cashflow",
        ("7", "14", "30", "45", "60", "30b"),
    ),
    IntentSpec("help_answer", "V3_help_answer", HELP_TOPICS),
    IntentSpec("list_transfers", "V3_list_transfers", _WORDING_VARIANTS),
    IntentSpec("verify_payee_known", "V3_verify_payee", _WORDING_VARIANTS),
    IntentSpec(
        "set_card_control",
        "V3_set_card_control",
        tuple(f"{control}:{state}" for control in CARD_CONTROLS for state in ("on", "off")),
    ),
    IntentSpec(
        "block_merchant",
        "V3_block_merchant",
        MERCHANT_BLOCK_CATEGORIES + ("gambling_b",),
    ),
    IntentSpec(
        "unblock_merchant",
        "V3_unblock_merchant",
        MERCHANT_BLOCK_CATEGORIES + ("gambling_b",),
    ),
    IntentSpec("get_balance", "V3_get_balance", _WORDING_VARIANTS),
    IntentSpec("list_contacts", "V3_list_contacts", _WORDING_VARIANTS),
    IntentSpec("unfreeze_card", "V3_unfreeze_card", _WORDING_VARIANTS),
    IntentSpec(
        "subscriptions_suggestions",
        "V3_subscription_advice",
        _WORDING_VARIANTS,
    ),
    IntentSpec("cancel_subscription", "V3_cancel_subscription", _WORDING_VARIANTS),
    IntentSpec("redeem_product", "V3_redeem_product", _WORDING_VARIANTS),
)

# About 28% of the generated examples are negatives.  Each variant is repeated
# five times so the source-family count, not just the sample count, stays in the
# requested 25-30% band after the split.
NEGATIVE_INTENTS: tuple[IntentSpec, ...] = (
    IntentSpec(
        "clarify_missing_amount",
        "V3_clarify_missing_amount",
        ("x",),
        negative=True,
        weight=5,
    ),
    IntentSpec(
        "clarify_missing_payee",
        "V3_clarify_missing_payee",
        ("x",),
        negative=True,
        weight=5,
    ),
    IntentSpec(
        "clarify_missing_term",
        "V3_clarify_missing_term",
        ("x",),
        negative=True,
        weight=5,
    ),
    IntentSpec(
        "clarify_unsupported_literal",
        "V3_clarify_unsupported",
        ("term", "horizon"),
        negative=True,
        weight=5,
    ),
    IntentSpec(
        "negated_transfer",
        "V3_negated_transfer",
        ("x",),
        negative=True,
        weight=5,
    ),
    IntentSpec(
        "raw_unverified_target",
        "V3_raw_unverified_target",
        ("x",),
        negative=True,
        weight=5,
    ),
    IntentSpec(
        "verify_payee_unknown",
        "V3_verify_payee_unknown",
        ("x",),
        negative=True,
        weight=5,
    ),
)

ALL_INTENTS: tuple[IntentSpec, ...] = POSITIVE_INTENTS + NEGATIVE_INTENTS


# ---------------------------------------------------------------------------
# Small gold builders
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
        "args": dict(args or {}),
        "after": list(after or []),
    }


def _plan(steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": "plan", "steps": steps}


def _clarify(question: str, options: list[str] | None = None) -> dict[str, Any]:
    return {"kind": "clarify", "question": question, "options": list(options or [])}


def _refuse(reason: str) -> dict[str, Any]:
    return {"kind": "refuse", "reason": reason}


def _first_label(entries: list[TokenEntry], kind: str) -> str:
    for entry in entries:
        if entry.kind == kind:
            return entry.label
    raise ValueError(f"no {kind} token was minted for this sample")


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _pick(rng: random.Random, values: tuple[Any, ...] | list[Any]) -> Any:
    return values[rng.randrange(len(values))]


@lru_cache(maxsize=1)
def _personas() -> tuple[dict[str, Any], ...]:
    with _SEGMENTS_PATH.open(encoding="utf-8") as fh:
        segments = yaml.safe_load(fh)["segments"]
    # Expat is the held-out persona; it must never enter the training families.
    return tuple(
        segment for segment in segments if not segment.get("held_out")
    )


@lru_cache(maxsize=64)
def _template_bank(persona_id: str, seed: int, today: date) -> tuple[Any, Any]:
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


def _secret(bank_seed: int) -> bytes:
    return hashlib.sha256(f"anban-v3-gold:{bank_seed}".encode("utf-8")).digest()


def _session(
    bank_seed: int,
    scenario: str,
    persona_id: str,
    lang: str,
    slot_seed: int,
) -> Session:
    digest = hashlib.sha256(
        f"{bank_seed}|{scenario}|{persona_id}|{lang}|{slot_seed}".encode("utf-8")
    ).hexdigest()
    return Session(
        session_id=digest[:32],
        plan_id=digest[32:48],
        customer_id=f"cust_{persona_id}",
    )


# ---------------------------------------------------------------------------
# Slot surfaces
# ---------------------------------------------------------------------------

_AMOUNT_YUAN: tuple[int, ...] = (100, 200, 300, 500, 800, 1000, 2000, 3000, 5000, 8000)


def _amount_surface(rng: random.Random, lang: str) -> str:
    yuan = _pick(rng, _AMOUNT_YUAN)
    return str(yuan) if lang == "en" else f"{yuan}块"


def _unique_contacts(state: Any) -> list[Any]:
    contacts = list(state.contacts)
    counts = Counter(contact.name for contact in contacts)
    unique = [contact for contact in contacts if counts[contact.name] == 1]
    return unique or contacts


def _payee_surface(state: Any, rng: random.Random) -> str:
    return _pick(rng, _unique_contacts(state)).name


def _subscription_surface(subscription: Any, lang: str) -> str:
    aliases = [
        alias
        for alias, canonical in SUB_ALIASES.items()
        if canonical == subscription.merchant
    ]
    if lang == "en":
        ascii_aliases = [alias for alias in aliases if alias.isascii()]
        if ascii_aliases:
            return sorted(ascii_aliases, key=len)[0]
    else:
        cjk_aliases = [alias for alias in aliases if not alias.isascii()]
        if cjk_aliases:
            return sorted(cjk_aliases, key=len)[0]
    if aliases:
        return sorted(aliases, key=len)[0]
    return subscription.merchant


def _pick_subscription(state: Any, rng: random.Random) -> Any:
    subscriptions = list(state.subscriptions)
    if not subscriptions:
        raise ValueError("seeded persona has no subscriptions")
    return _pick(rng, subscriptions)


def _redeemable_product_ids(state: Any) -> list[str]:
    """Product ids with a positive bank holding that the mock bank may redeem."""

    return sorted(
        holding.product_id
        for holding in state.holdings
        if int(holding.value_fen) > 0
        and holding.product_id in REDEEMABLE_PRODUCTS
    )


_BILL_WORDS: dict[str, dict[str, str]] = {
    "water": {"zh": "水费", "en": "water bill", "mixed": "water 费"},
    "electricity": {
        "zh": "电费",
        "en": "electricity bill",
        "mixed": "electricity 费",
    },
    "gas": {"zh": "燃气费", "en": "gas bill", "mixed": "gas 费"},
    "mobile": {"zh": "话费", "en": "mobile bill", "mixed": "话费 phone bill"},
    "broadband": {"zh": "宽带费", "en": "broadband bill", "mixed": "broadband 费"},
    "property": {"zh": "物业费", "en": "property fee", "mixed": "property 管理费"},
}

_BLOCK_WORDS: dict[str, dict[str, str]] = {
    "gambling": {"zh": "赌博", "en": "gambling", "mixed": "gambling 赌博"},
    "crypto": {"zh": "加密货币", "en": "crypto", "mixed": "crypto 加密"},
    "lottery": {"zh": "彩票", "en": "lottery", "mixed": "lottery 彩票"},
    "adult": {"zh": "成人内容", "en": "adult", "mixed": "adult 成人"},
    "cash_like": {"zh": "现金类", "en": "cash-like", "mixed": "cash-like 现金"},
}

_CONTROL_WORDS: dict[str, dict[str, str]] = {
    "online": {"zh": "网上支付", "en": "online payments", "mixed": "online 支付"},
    "abroad": {"zh": "境外支付", "en": "overseas payments", "mixed": "abroad 支付"},
    "atm": {"zh": "ATM取现", "en": "ATM withdrawals", "mixed": "ATM 取现"},
    "contactless": {"zh": "闪付", "en": "contactless payments", "mixed": "contactless 闪付"},
}

_HELP_QUESTIONS: dict[str, dict[str, str]] = {
    "delayed_arrival": {
        "zh": "转账一般多久到账",
        "en": "how long do transfers take to arrive",
        "mixed": "transfer 一般多久到账",
    },
    "recall": {
        "zh": "转出去的钱还能撤回吗",
        "en": "can I recall a transfer after sending it",
        "mixed": "转出去的钱 can I recall",
    },
    "deposit_insurance": {
        "zh": "存款保险保多少",
        "en": "how much does deposit insurance cover",
        "mixed": "存款保险 cover 多少",
    },
    "fx_quota": {
        "zh": "个人换汇额度是多少",
        "en": "what is the personal foreign exchange quota",
        "mixed": "换汇 quota 是多少",
    },
    "complaints": {
        "zh": "怎么投诉银行服务",
        "en": "how do I complain about the bank",
        "mixed": "怎么 complaint 银行服务",
    },
    "anti_fraud": {
        "zh": "怎么防诈骗",
        "en": "how can I avoid scams",
        "mixed": "怎么防 scam",
    },
    "large_deposit_cd": {
        "zh": "大额存单是什么",
        "en": "what is a large deposit certificate",
        "mixed": "大额存单 what is it",
    },
    "suitability": {
        "zh": "风险测评怎么做",
        "en": "how does the suitability assessment work",
        "mixed": "风险测评 how to do",
    },
    "personal_pension": {
        "zh": "个人养老金怎么开",
        "en": "how do I open a personal pension account",
        "mixed": "个人养老金 how to open",
    },
    "account_tiers": {
        "zh": "账户分类是什么",
        "en": "what are the bank account tiers",
        "mixed": "账户 tiers 是什么",
    },
    "installment_rate": {
        "zh": "分期利率怎么算",
        "en": "how is the installment rate calculated",
        "mixed": "分期 rate 怎么算",
    },
    "account_restriction": {
        "zh": "账户被限制怎么办",
        "en": "why is my account restricted",
        "mixed": "账户 restricted 怎么办",
    },
}

_RAW_TARGETS: tuple[str, ...] = (
    "6222 0000 1234 5678",
    "6217 0000 9876 5432",
)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_TEXTS: dict[str, dict[str, tuple[str, ...]]] = {
    "list_bills": {
        "zh": (
            "我这个月有哪些账单要交",
            "帮我看看这期的账单",
            "这个月还有什么费用没交",
        ),
        "en": (
            "what bills do I have this month",
            "show me my upcoming bills",
            "what payments are due this month",
        ),
        "mixed": (
            "这个月有哪些 bills 要交",
            "帮我 check 一下这期账单",
            "what bills 还没交",
        ),
    },
    "pay_bill": {
        "zh": (
            "帮我交一下这个月的{bill}",
            "这个月的{bill}帮我交了",
            "把{bill}交一下，谢谢",
        ),
        "en": (
            "pay my {bill} this month",
            "can you pay the {bill} please",
            "settle this month's {bill}",
        ),
        "mixed": (
            "帮我 pay 一下这个月的{bill}",
            "{bill} this month help me pay",
            "把 {bill} 交一下，thanks",
        ),
    },
    "installment_quote": {
        "zh": (
            "分期{amount}，分{periods}期，每期多少钱",
            "{amount}分{periods}期手续费多少",
            "帮我算一下{amount}分{periods}期的分期成本",
        ),
        "en": (
            "what is the instalment quote for {amount} over {periods} periods",
            "{amount} in {periods} instalments, what is the fee",
            "quote me {amount} over {periods} instalments",
        ),
        "mixed": (
            "{amount} 分 {periods} 期，quote 一下",
            "instalment {amount} over {periods} periods 多少钱",
            "帮我算 {amount} 分 {periods} 期的 cost",
        ),
    },
    "deposit_quote": {
        "zh": (
            "存{amount}定期{term}个月，利息多少",
            "{amount}存{term}个月定期，利率多少",
            "帮我看看{amount}存{term}个月的利息",
        ),
        "en": (
            "deposit quote for {amount} for {term} months",
            "what interest would {amount} earn over {term} months",
            "how much interest on {amount} for {term} months",
        ),
        "mixed": (
            "存 {amount} 定期 {term} months，rate 多少",
            "{amount} deposit {term} 个月，interest 多少",
            "帮我 quote 一下 {amount} 存 {term} 个月",
        ),
    },
    "forecast_cashflow": {
        "zh": (
            "未来{days}天我的钱够花吗",
            "帮我预测{days}天后的现金流",
            "接下来{days}天现金够不够",
        ),
        "en": (
            "will I have enough money for the next {days} days",
            "cashflow forecast for the next {days} days",
            "do I have enough for {days} days",
        ),
        "mixed": (
            "未来 {days} 天 cashflow 够不够",
            "forecast 一下接下来 {days} 天的现金",
            "帮我看看 {days} 天钱够不够",
        ),
    },
    "list_transfers": {
        "zh": (
            "我最近的转账到账了吗",
            "查一下最近的转账状态",
            "帮我看看最近的转账到哪了",
        ),
        "en": (
            "did my recent transfers arrive",
            "show me my recent transfer status",
            "where are my recent transfers",
        ),
        "mixed": (
            "帮我 check 最近的 transfer 状态",
            "最近的 transfer 到账了吗",
            "show me recent transfers 到哪了",
        ),
    },
    "verify_payee_known": {
        "zh": (
            "查一下{payee}这个收款人安全吗",
            "帮我验证{payee}的账户",
            "{payee}这个收款人有没有风险",
        ),
        "en": (
            "is {payee} a safe payee",
            "verify {payee} for me",
            "does {payee} look risky as a payee",
        ),
        "mixed": (
            "check 一下 {payee} 这个 payee 安全吗",
            "帮我 verify {payee} 的账户",
            "{payee} 这个 payee 有没有 risk",
        ),
    },
    "set_card_control": {
        "zh": (
            "把卡的{control}功能{action}",
            "帮我{action}卡的{control}",
            "这张卡的{control}帮我{action}",
        ),
        "en": (
            "turn {action} {control} on my card",
            "can you turn {action} {control} for my card",
            "please switch {control} {action}",
        ),
        "mixed": (
            "帮我把卡 {control} {action}",
            "{control} on my card help me {action}",
            "把这张卡的 {control} {action}，thanks",
        ),
    },
    "block_merchant": {
        "zh": (
            "禁止{category}类消费",
            "帮我屏蔽{category}商户",
            "把{category}类的消费挡掉",
        ),
        "en": (
            "block {category} spending",
            "please block my card for {category}",
            "stop {category} payments on my card",
        ),
        "mixed": (
            "block 一下 {category} 消费",
            "帮我 block {category} 商户",
            "把 {category} spending 关掉",
        ),
    },
    "unblock_merchant": {
        "zh": (
            "解除{category}类消费限制",
            "帮我恢复{category}商户",
            "把{category}类的限制取消",
        ),
        "en": (
            "unblock {category} spending",
            "remove the block on {category} payments",
            "let {category} payments through again",
        ),
        "mixed": (
            "unblock 一下 {category} 消费",
            "帮我解除 {category} 限制",
            "把 {category} spending 恢复",
        ),
    },
    "get_balance": {
        "zh": ("查一下我的余额", "我卡里还有多少钱", "看看账户余额"),
        "en": (
            "what is my balance",
            "show me my account balance",
            "how much money do I have",
        ),
        "mixed": (
            "check 一下我的 balance",
            "我账户里还有多少钱",
            "show me 账户余额",
        ),
    },
    "list_contacts": {
        "zh": ("我的联系人有哪些", "帮我列出收款人", "看看我的通讯录"),
        "en": (
            "list my contacts",
            "who can I send money to",
            "show me my payees",
        ),
        "mixed": (
            "看看我的 contacts",
            "list 一下我的 payees",
            "我的收款人有哪些",
        ),
    },
    "unfreeze_card": {
        "zh": ("把我的卡解冻", "帮我把卡恢复使用", "卡不冻了，解冻一下"),
        "en": (
            "unfreeze my card",
            "unlock my card please",
            "my card should work again, unfreeze it",
        ),
        "mixed": (
            "帮我把卡 unfreeze 一下",
            "my card 解冻，please",
            "把这张卡 unlock 回来",
        ),
    },
    "subscriptions_suggestions": {
        "zh": (
            "我该取消哪些订阅",
            "哪些会员我很久没用了",
            "帮我看看有没有不用的订阅",
        ),
        "en": (
            "which subscriptions should I cancel",
            "are any of my memberships unused",
            "show me subscriptions I could drop",
        ),
        "mixed": (
            "哪些 subscription 该 cancel",
            "帮我看看有没有 unused 的会员",
            "which memberships 可以取消",
        ),
    },
    "cancel_subscription": {
        "zh": (
            "帮我取消{sub}",
            "{sub}不要了，退订",
            "把{sub}的自动续费停掉",
        ),
        "en": (
            "cancel {sub}",
            "unsubscribe from {sub}",
            "stop the {sub} renewal",
        ),
        "mixed": (
            "cancel 一下 {sub}",
            "帮我 stop {sub}",
            "把 {sub} 的续费取消",
        ),
    },
    "redeem_product": {
        "zh": (
            "把我唯一那只可以赎回的基金全部赎回",
            "我那只开放的基金帮我全部赎回了",
            "把能赎回的那只基金全赎出来",
        ),
        "en": (
            "redeem my only redeemable fund in full",
            "cash out all of my single redeemable fund",
            "please fully redeem the one fund I can redeem",
        ),
        "mixed": (
            "帮我 full redeem 我唯一那只 fund",
            "把我那 only 可以 redeem 的基金全部赎回",
            "redeem 那只 fund 全部，full redemption",
        ),
    },
    "clarify_missing_amount": {
        "zh": ("给{payee}转点钱", "帮我给{payee}转些钱", "{payee}那边要转点钱"),
        "en": (
            "send {payee} some money",
            "transfer something to {payee}",
            "pay {payee} some amount",
        ),
        "mixed": (
            "给 {payee} send some money",
            "transfer 点钱给 {payee}",
            "{payee} 那边 some money",
        ),
    },
    "clarify_missing_payee": {
        "zh": ("转{amount}", "帮我转{amount}", "这笔{amount}转出去"),
        "en": (
            "send {amount}",
            "transfer {amount} please",
            "move {amount} out",
        ),
        "mixed": (
            "transfer {amount}",
            "帮我 send {amount}",
            "转 {amount} 出去",
        ),
    },
    "clarify_missing_term": {
        "zh": (
            "存{amount}定期，利息多少",
            "{amount}存定期，利率多少",
            "帮我看看{amount}定期的利息",
        ),
        "en": (
            "deposit {amount}, what is the rate",
            "what interest does a {amount} deposit pay",
            "put {amount} on deposit, what is the rate",
        ),
        "mixed": (
            "存 {amount} 定期，rate 多少",
            "{amount} deposit，interest 多少",
            "帮我 quote 一下 {amount} 定期",
        ),
    },
    "clarify_unsupported_literal": {
        "zh": (
            "存{amount}定期{term}个月，利息多少",
            "帮我看看{amount}存{term}个月定期的利息",
            "未来{days}天我的钱够花吗",
        ),
        "en": (
            "deposit {amount} for {term} months",
            "what interest on {amount} for {term} months",
            "will I have enough for the next {days} days",
        ),
        "mixed": (
            "存 {amount} 定期 {term} months，rate 多少",
            "{amount} deposit {term} 个月，interest 多少",
            "未来 {days} 天 cashflow 够不够",
        ),
    },
    "negated_transfer": {
        "zh": (
            "不要给{payee}转{amount}",
            "别给{payee}打钱，{amount}也别转",
            "{payee}那笔{amount}先别转",
        ),
        "en": (
            "don't send {amount} to {payee}",
            "do not transfer {amount} to {payee}",
            "cancel that {amount} payment to {payee}",
        ),
        "mixed": (
            "别给 {payee} transfer {amount}",
            "don't send {amount} to {payee}，先别转",
            "{payee} 那笔 {amount} 不要转",
        ),
    },
    "raw_unverified_target": {
        "zh": (
            "给{raw}转{amount}",
            "帮我转{amount}到这个账号{raw}",
            "往{raw}打{amount}",
        ),
        "en": (
            "send {amount} to {raw}",
            "transfer {amount} to account {raw}",
            "pay {amount} into {raw}",
        ),
        "mixed": (
            "transfer {amount} 到 {raw}",
            "给 {raw} 转 {amount}",
            "帮我 send {amount} to {raw}",
        ),
    },
    "verify_payee_unknown": {
        "zh": (
            "帮我查一下这个新收款人安全吗",
            "这个陌生账户有风险吗",
            "新收款人能不能直接转",
        ),
        "en": (
            "is this new payee safe",
            "does this unknown account look risky",
            "can I pay this new payee directly",
        ),
        "mixed": (
            "check 一下这个 new payee 安全吗",
            "这个 unknown account 有 risk 吗",
            "新 payee 能不能直接转",
        ),
    },
}


# ---------------------------------------------------------------------------
# Clarify / refuse questions
# ---------------------------------------------------------------------------

_CLARIFY_QUESTIONS: dict[str, dict[str, str]] = {
    "missing_amount": {
        "zh": "请问要转多少金额？",
        "en": "How much would you like to transfer?",
        "mixed": "请问要转多少 amount？",
    },
    "missing_payee": {
        "zh": "请问要转给谁？",
        "en": "Who would you like to send it to?",
        "mixed": "请问要转给谁？",
    },
    "missing_term": {
        "zh": "请问要存多久？可选3、6、12、24、36或60个月。",
        "en": "What term would you like? Choose 3, 6, 12, 24, 36 or 60 months.",
        "mixed": "请问存多久？可选 3/6/12/24/36/60 months。",
    },
    "unsupported_term": {
        "zh": "这个期限不支持，可选3、6、12、24、36或60个月。请问选哪个？",
        "en": "That term is not supported. Choose 3, 6, 12, 24, 36 or 60 months.",
        "mixed": "这个 term 不支持，可选 3/6/12/24/36/60 months。",
    },
    "unsupported_horizon": {
        "zh": "这个天数不支持，可选7、14、30、45或60天。请问要看多少天？",
        "en": "That horizon is not supported. Choose 7, 14, 30, 45 or 60 days.",
        "mixed": "这个 horizon 不支持，可选 7/14/30/45/60 days。",
    },
    "negated": {
        "zh": "好的，不会发起这笔转账。需要我改成查询或提醒吗？",
        "en": "Understood, I will not send that transfer. Would you like a reminder or a lookup instead?",
        "mixed": "好的，不会发起这笔 transfer。需要我改成查询或提醒吗？",
    },
    "unknown_payee": {
        "zh": "请先添加并验证这个收款人，然后再帮你转账。",
        "en": "Please add and verify this payee first, then I can help with the transfer.",
        "mixed": "请先 add 并 verify 这个 payee，然后再帮你转账。",
    },
    "redeem_none": {
        "zh": "你目前没有可赎回的基金持仓，需要先看看持仓吗？",
        "en": "You have no redeemable fund holding right now. Would you like to see your portfolio?",
        "mixed": "你现在没有 redeemable 的基金持仓，要先看 portfolio 吗？",
    },
    "redeem_multiple": {
        "zh": "你有不止一只可赎回的基金，请问要赎回哪一只？",
        "en": "You have more than one redeemable fund. Which one would you like to redeem?",
        "mixed": "你有不止一只 redeemable fund，请问要 redeem 哪一只？",
    },
}


# ---------------------------------------------------------------------------
# Render one intent
# ---------------------------------------------------------------------------


def _bill_category(variant: str) -> str:
    return variant.removesuffix("_b")


def _merchant_category(variant: str) -> str:
    return variant.removesuffix("_b")


def _numeric_variant(variant: str) -> int:
    digits = ""
    for char in variant:
        if char.isdigit():
            digits += char
        elif digits:
            break
    if not digits:
        raise ValueError(f"variant {variant!r} has no number")
    return int(digits)


def _pick_text(intent: str, lang: str, rng: random.Random) -> str:
    return _pick(rng, _TEXTS[intent][lang])


def _render(
    intent: str,
    variant: str,
    lang: str,
    state: Any,
    rng: random.Random,
) -> tuple[str, Callable[[list[TokenEntry]], dict[str, Any]], dict[str, Any]]:
    """Return the user text, a token-aware gold builder and extra metadata."""

    def no_tokens(gold: dict[str, Any]) -> Callable[[list[TokenEntry]], dict[str, Any]]:
        return lambda _entries: gold

    if intent == "list_bills":
        return _pick_text(intent, lang, rng), no_tokens(
            _plan([_step("s1", "list_bills")])
        ), {}

    if intent == "pay_bill":
        category = _bill_category(variant)
        text = _pick_text(intent, lang, rng).format(
            bill=_BILL_WORDS[category][lang]
        )
        return text, no_tokens(
            _plan([_step("s1", "pay_bill", {"category": category})])
        ), {"category": category}

    if intent == "installment_quote":
        periods = _numeric_variant(variant)
        amount = _amount_surface(rng, lang)
        text = _pick_text(intent, lang, rng).format(amount=amount, periods=periods)
        return text, lambda entries: _plan(
            [
                _step(
                    "s1",
                    "installment_quote",
                    {"amount": _first_label(entries, "AMT"), "periods": periods},
                )
            ]
        ), {"periods": periods}

    if intent == "deposit_quote":
        term = _numeric_variant(variant)
        amount = _amount_surface(rng, lang)
        text = _pick_text(intent, lang, rng).format(amount=amount, term=term)
        return text, lambda entries: _plan(
            [
                _step(
                    "s1",
                    "deposit_quote",
                    {"amount": _first_label(entries, "AMT"), "term_months": term},
                )
            ]
        ), {"term_months": term}

    if intent == "forecast_cashflow":
        days = _numeric_variant(variant)
        text = _pick_text(intent, lang, rng).format(days=days)
        return text, no_tokens(
            _plan([_step("s1", "forecast_cashflow", {"days": days})])
        ), {"days": days}

    if intent == "help_answer":
        text = _HELP_QUESTIONS[variant][lang]
        return text, no_tokens(
            _plan([_step("s1", "help_answer", {"topic": variant})])
        ), {"topic": variant}

    if intent == "list_transfers":
        return _pick_text(intent, lang, rng), no_tokens(
            _plan([_step("s1", "list_transfers")])
        ), {}

    if intent == "verify_payee_known":
        payee = _payee_surface(state, rng)
        text = _pick_text(intent, lang, rng).format(payee=payee)
        return text, lambda entries: _plan(
            [
                _step(
                    "s1",
                    "verify_payee",
                    {"payee": _first_label(entries, "PAYEE")},
                )
            ]
        ), {}

    if intent == "set_card_control":
        control, state_flag = variant.split(":", 1)
        enabled = state_flag == "on"
        if lang == "en":
            action = "on" if enabled else "off"
        else:
            action = "打开" if enabled else "关掉"
        text = _pick_text(intent, lang, rng).format(
            control=_CONTROL_WORDS[control][lang],
            action=action,
        )
        return text, lambda entries: _plan(
            [
                _step(
                    "s1",
                    "set_card_control",
                    {
                        "control": control,
                        "enabled": enabled,
                        "card": _first_label(entries, "CARD"),
                    },
                )
            ]
        ), {"control": control, "enabled": enabled}

    if intent == "block_merchant":
        category = _merchant_category(variant)
        text = _pick_text(intent, lang, rng).format(
            category=_BLOCK_WORDS[category][lang]
        )
        return text, no_tokens(
            _plan([_step("s1", "block_merchant", {"category": category})])
        ), {"category": category}

    if intent == "unblock_merchant":
        category = _merchant_category(variant)
        text = _pick_text(intent, lang, rng).format(
            category=_BLOCK_WORDS[category][lang]
        )
        return text, no_tokens(
            _plan([_step("s1", "unblock_merchant", {"category": category})])
        ), {"category": category}

    if intent == "get_balance":
        return _pick_text(intent, lang, rng), no_tokens(
            _plan([_step("s1", "get_balance")])
        ), {}

    if intent == "list_contacts":
        return _pick_text(intent, lang, rng), no_tokens(
            _plan([_step("s1", "list_contacts")])
        ), {}

    if intent == "unfreeze_card":
        return _pick_text(intent, lang, rng), lambda entries: _plan(
            [
                _step("s1", "list_cards"),
                _step(
                    "s2",
                    "unfreeze_card",
                    {"card": _first_label(entries, "CARD")},
                    ["s1"],
                ),
            ]
        ), {}

    if intent == "subscriptions_suggestions":
        return _pick_text(intent, lang, rng), no_tokens(
            _plan([_step("s1", "list_subscriptions")])
        ), {}

    if intent == "cancel_subscription":
        subscription = _pick_subscription(state, rng)
        text = _pick_text(intent, lang, rng).format(
            sub=_subscription_surface(subscription, lang)
        )
        return text, lambda entries: _plan(
            [
                _step("s1", "list_subscriptions"),
                _step(
                    "s2",
                    "cancel_subscription",
                    {"sub": _first_label(entries, "SUB")},
                    ["s1"],
                ),
            ]
        ), {}

    if intent == "redeem_product":
        text = _pick_text(intent, lang, rng)
        eligible = _redeemable_product_ids(state)
        if len(eligible) != 1:
            question = _CLARIFY_QUESTIONS[
                "redeem_none" if not eligible else "redeem_multiple"
            ][lang]
            return text, no_tokens(_clarify(question)), {"redeemable": eligible}
        product_id = eligible[0]

        def redeem_gold(entries: list[TokenEntry]) -> dict[str, Any]:
            matches = [
                entry
                for entry in entries
                if entry.kind == "PROD"
                and entry.real_ref == product_id
                and entry.tags.get("held") is True
                and entry.tags.get("redeemable") is True
            ]
            if not matches:
                # The bank context did not expose a safe product token.
                return _clarify(_CLARIFY_QUESTIONS["redeem_none"][lang])
            return _plan(
                [
                    _step("s1", "get_portfolio"),
                    _step(
                        "s2",
                        "redeem_product",
                        {"product": matches[0].label},
                        ["s1"],
                    ),
                ]
            )

        return text, redeem_gold, {"redeemable": eligible}

    if intent == "clarify_missing_amount":
        payee = _payee_surface(state, rng)
        text = _pick_text(intent, lang, rng).format(payee=payee)
        return text, no_tokens(
            _clarify(_CLARIFY_QUESTIONS["missing_amount"][lang])
        ), {}

    if intent == "clarify_missing_payee":
        amount = _amount_surface(rng, lang)
        text = _pick_text(intent, lang, rng).format(amount=amount)
        return text, no_tokens(
            _clarify(_CLARIFY_QUESTIONS["missing_payee"][lang])
        ), {}

    if intent == "clarify_missing_term":
        amount = _amount_surface(rng, lang)
        text = _pick_text(intent, lang, rng).format(amount=amount)
        return text, no_tokens(
            _clarify(_CLARIFY_QUESTIONS["missing_term"][lang])
        ), {}

    if intent == "clarify_unsupported_literal":
        amount = _amount_surface(rng, lang)
        templates = _TEXTS[intent][lang]
        if variant == "term":
            text = _pick(rng, templates[:2]).format(
                amount=amount, term=48, days=30
            )
            question = _CLARIFY_QUESTIONS["unsupported_term"][lang]
        else:
            text = templates[2].format(amount=amount, term=12, days=90)
            question = _CLARIFY_QUESTIONS["unsupported_horizon"][lang]
        return text, no_tokens(_clarify(question)), {"unsupported": variant}

    if intent == "negated_transfer":
        payee = _payee_surface(state, rng)
        amount = _amount_surface(rng, lang)
        text = _pick_text(intent, lang, rng).format(payee=payee, amount=amount)
        return text, no_tokens(_clarify(_CLARIFY_QUESTIONS["negated"][lang])), {}

    if intent == "raw_unverified_target":
        amount = _amount_surface(rng, lang)
        raw = _pick(rng, _RAW_TARGETS)
        text = _pick_text(intent, lang, rng).format(amount=amount, raw=raw)
        # A raw account/phone is not a verified PAYEE; never build a transfer.
        return text, no_tokens(_refuse("not_supported")), {}

    if intent == "verify_payee_unknown":
        return _pick_text(intent, lang, rng), no_tokens(
            _clarify(_CLARIFY_QUESTIONS["unknown_payee"][lang])
        ), {}

    raise ValueError(f"unknown v3 intent: {intent!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _slot_seed(seed: int, intent: str, variant: str, rep: int) -> int:
    # Deliberately independent of language: the zh/en/mixed wordings of one
    # source family stay together.
    return _stable_int(f"v3|{seed}|{intent}|{variant}|{rep}")


def _persona_for(
    seed: int, intent: str, variant: str, rep: int
) -> str:
    personas = _personas()
    index = _stable_int(f"persona|{seed}|{intent}|{variant}|{rep}") % len(personas)
    return str(personas[index]["id"])


def _build_record(
    seed: int,
    today: date,
    spec: IntentSpec,
    variant: str,
    rep: int,
    lang: str,
) -> dict[str, Any]:
    slot_seed = _slot_seed(seed, spec.name, variant, rep)
    bank_seed = seed + (slot_seed % 5)
    persona_id = _persona_for(seed, spec.name, variant, rep)
    bank, state = _load_bank(persona_id, bank_seed, today)
    if spec.name == "unfreeze_card":
        # The independent copied bank must match the operation being taught.
        bank.freeze_card(bank.list_cards()[0].id)
    vault = TokenVault(_secret(bank_seed))
    session = _session(bank_seed, spec.scenario, persona_id, lang, slot_seed)

    rng = random.Random(slot_seed ^ _stable_int(lang))
    text, gold_fn, extra = _render(spec.name, variant, lang, state, rng)

    tokenized, _entries = tokenize_user_text(text, bank, vault, session, today)
    context_lines = build_context(bank, vault, session, today)
    gold = gold_fn(vault.entries(session))

    assistant = json.dumps(
        gold, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    parsed = parse_planner_output(assistant)
    validate_plan(parsed, vault.known_tokens(session))

    tools = sorted(
        {step["tool"] for step in gold.get("steps", []) if isinstance(step, dict)}
    )
    meta: dict[str, Any] = {
        "source": "v3",
        "scenario": spec.scenario,
        "scenario_id": spec.scenario,
        "intent": spec.name,
        "variant": variant,
        "persona": persona_id,
        "persona_id": persona_id,
        "bankseed": bank_seed,
        "bank_seed": bank_seed,
        "slotseed": slot_seed,
        "slot_seed": slot_seed,
        "family": f"{spec.scenario}|{persona_id}|{bank_seed}|{slot_seed}",
        "seed": seed,
        "today": today.isoformat(),
        "lang": lang,
        "kind": gold["kind"],
        "negative": bool(spec.negative),
        "tools": tools,
    }
    meta.update(extra)
    bank.conn.close()

    return {
        "messages": [
            {"role": "system", "content": R3_SYSTEM_PROMPT},
            {"role": "user", "content": render_planner_input(tokenized, context_lines)},
            {"role": "assistant", "content": assistant},
        ],
        # The same JSON-string meta contract as training.build_dataset: pyarrow
        # infers one schema from early rows and crashes on heterogeneous objects.
        "meta": json.dumps(meta, ensure_ascii=False, sort_keys=True, default=str),
    }


def generate_samples(
    seed: int,
    today: date,
    per_intent: int = 1,
) -> list[dict[str, Any]]:
    """Generate the standalone v3 ShareGPT samples.

    ``per_intent`` repeats every variant; negative intents carry a fixed weight
    so roughly 28% of samples are clarify/refuse negatives.  The result is
    deterministic for a given ``(seed, today, per_intent)``.
    """

    if isinstance(today, datetime):
        today = today.date()
    if per_intent < 1:
        raise ValueError("per_intent must be at least 1")

    records: list[dict[str, Any]] = []
    for spec in ALL_INTENTS:
        repeats = per_intent * spec.weight
        for variant in spec.variants:
            for rep in range(repeats):
                for lang in LANGS:
                    records.append(
                        _build_record(seed, today, spec, variant, rep, lang)
                    )
    return records
