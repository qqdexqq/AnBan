"""Tier computation, cumulative limits and run-time re-checks over the ledger.

The planner output is untrusted input.  This module is the security boundary
that turns a :class:`~security.dag.ValidatedPlan` into per-step decisions:
risk tiers, cumulative daily limits, vetoes and the mandate/MFA flags the
executor needs.  Nothing the planner writes can lower a tier.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Literal

from agent.tools import REGISTRY, STEP_REF_RE, TOKEN_RE, ToolSpec
from bank import bills, redemption
from bank.api import BankError
from bank.merchants import price_of
from bank.money import split_fen
from security import compliance
from security.dag import ValidatedPlan
from security.mandate import PAYMENT_LOCK_PROTECTIVE_ACTIONS
from security.tokens import Session, TokenEntry, TokenError, TokenVault

Tier = Literal["green", "yellow", "red"]

TIER_RANK: dict[str, int] = {"green": 0, "yellow": 1, "red": 2}

# Tools whose amount leaves the customer's spendable money today.
OUTGOING_TOOLS = frozenset(
    {"transfer", "schedule_transfer", "merchant_order", "subscribe_product", "pay_bill"}
)

# Tools that move the customer's money at all (used by the scam veto).
# During a scam freeze only protective writes may run. Coached steps are not just
# transfers: "open a virtual card and read me the number", "change your password",
# "unfreeze the card".
PROTECTIVE_TOOLS = frozenset(
    {
        "freeze_card",
        "report_card_lost",
        "cancel_transfer",
        "cancel_scheduled",
        "cancel_request",
        "cancel_order",
        "cancel_subscription",
        "close_virtual_card",
        "set_reminder",
        "block_merchant",
    }
)


def _blocked_under_scam(tool: str, kind: str) -> bool:
    return kind == "write" and tool not in PROTECTIVE_TOOLS


MONEY_TOOLS = frozenset(
    {
        "transfer",
        "schedule_transfer",
        "request_payment",
        "merchant_order",
        "move_to_pot",
        "subscribe_product",
        "pay_bill",
        # Redemption pays money back in, so it is money but not outgoing.
        "redeem_product",
    }
)

DEFAULT_ELDER_PERSONAS = frozenset({"retiree"})

_TOKEN_FULL_RE = re.compile(rf"^{TOKEN_RE}$")
_TOKEN_ANY_RE = re.compile(TOKEN_RE)
_STEP_REF_FULL_RE = re.compile(rf"^{STEP_REF_RE}$")

# Reasons that were produced by plan-time checks which a run-time recheck
# cannot repeat (it has no vault or session).  They stay in force.
_PERSISTENT_REASON_CODES = frozenset(
    {"R_RAW", "R_UNTRUSTED", "R_UNVERIFIED", "R_NEW_PAYEE"}
)


class Rule:
    """Stable rule codes used by the audit log and the UI."""

    BASE = "R_BASE"
    LOCKED = "R_LOCKED"
    SCAM = "R_SCAM"
    UNTRUSTED = "R_UNTRUSTED"
    RAW = "R_RAW"
    DAILY = "R_DAILY"
    SCHEDULED = "R_SCHEDULED"
    NEW_PAYEE = "R_NEW_PAYEE"
    DEFERRED = "R_DEFERRED"
    TOO_MANY = "R_TOO_MANY"
    ELDER = "R_ELDER"
    UNVERIFIED = "R_UNVERIFIED"
    BAD_ITEM = "R_BAD_ITEM"
    CARD_LIMIT = "R_CARD_LIMIT"
    FRAUD_LIST = "R_FRAUD_LIST"
    SUITABILITY = "R_SUITABILITY"
    RISK_TEST = "R_RISK_TEST"
    EARLY_RELEASE = "R_EARLY_RELEASE"
    RELEASE_CHECK_UNAVAILABLE = "R_RELEASE_CHECK_UNAVAILABLE"


# (English, Chinese) plain-language messages for each rule.  A few rules
# deliberately use a more specific message at the call site.
_MESSAGES: dict[str, tuple[str, str]] = {
    Rule.BASE: (
        "This action is allowed at its base risk tier.",
        "此操作按其基础风险等级允许。",
    ),
    Rule.LOCKED: (
        "Payments are locked for up to 24 hours. Reply 'human' for help. Freezing or reporting a lost card remains available.",
        "支付已锁定，最长持续24小时。请回复“人工”寻求帮助；冻结卡片和挂失仍可办理。",
    ),
    Rule.SCAM: (
        "A scam warning is active, so money cannot be moved.",
        "检测到诈骗风险，不能转移资金。",
    ),
    Rule.UNTRUSTED: (
        "This target came from a message, not from you.",
        "这个收款对象来自消息内容，而不是你本人。",
    ),
    Rule.RAW: (
        "A raw account or phone number cannot be used in a write.",
        "写操作不能使用原始账号或电话号码。",
    ),
    Rule.DAILY: (
        "This would exceed your daily spending limit.",
        "这笔操作会超过你的每日支出限额。",
    ),
    Rule.SCHEDULED: (
        "This scheduled payment is above your daily limit.",
        "这笔定期转账超过了你的每日限额。",
    ),
    Rule.NEW_PAYEE: (
        "This is a new payee, so extra confirmation is required.",
        "这是新的收款人，需要额外确认。",
    ),
    Rule.FRAUD_LIST: (
        "This account is on the national anti-fraud list, so the payment is blocked.",
        "该账户在国家反诈名单中，已拦截这笔付款。",
    ),
    Rule.SUITABILITY: (
        "This product's risk rating is above your risk level, so it can't be bought.",
        "这个产品的风险等级高于你的风险承受等级，不能购买。",
    ),
    Rule.RISK_TEST: (
        "Your risk questionnaire is missing or more than a year old. Please retake "
        "it yourself in the app before buying.",
        "你的风险测评没有做或已超过一年，请先在 App 里本人重新测评再购买。",
    ),
    Rule.DEFERRED: (
        "The amount is only known at run time and will be re-checked.",
        "金额要到执行时才能确定，届时会重新检查。",
    ),
    Rule.TOO_MANY: (
        "The plan changes too many things at once.",
        "这个计划一次修改的项目太多。",
    ),
    Rule.ELDER: (
        "Family co-approval is required for this action.",
        "此操作需要家人共同确认。",
    ),
    Rule.UNVERIFIED: (
        "This target could not be verified.",
        "无法验证此收款对象。",
    ),
    Rule.BAD_ITEM: (
        "This item is not sold by that merchant.",
        "该商品不是这家商户出售的。",
    ),
    Rule.CARD_LIMIT: (
        "This card limit is above the allowed maximum.",
        "该卡片限额超过允许的上限。",
    ),
    Rule.EARLY_RELEASE: (
        "Releasing this pending transfer right now needs your own high-risk "
        "confirmation.",
        "立即放出这笔待处理转账属于高风险操作，需要你本人确认。",
    ),
    Rule.RELEASE_CHECK_UNAVAILABLE: (
        "The release could not be safety-checked, so it was blocked.",
        "无法完成放款安全校验，已拦截。",
    ),
}

_FLAGGED_PAYEE_MESSAGE = (
    "This payee is flagged as risky and cannot be paid.",
    "此收款人已被标记为高风险，不能付款。",
)


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Limits and switches for one policy evaluation."""

    daily_limit_fen: int = 100_000
    max_write_steps: int = 8
    elder_personas: frozenset[str] = DEFAULT_ELDER_PERSONAS
    elder_daily_limit_fen: int = 50_000
    card_limit_red_fen: int = 200_000
    red_needs_co_approval_for_elder: bool = True
    new_payee_red: bool = True


@dataclass(frozen=True, slots=True)
class Reason:
    """One plain-language explanation attached to a step or a plan."""

    code: str
    message_en: str
    message_zh: str


@dataclass(slots=True)
class StepDecision:
    """The policy verdict for one plan step."""

    step_id: str
    tool: str
    tier: Tier
    needs_mandate: bool = False
    needs_mfa: bool = False
    needs_co_approval: bool = False
    amount_fen: int | None = None
    deferred: bool = False
    reasons: list[Reason] = field(default_factory=list)
    # Re-check fields.  ``vetoed`` is true when execution must not happen;
    # ``requires_reapproval`` is true when the new tier is above the tier the
    # user already approved.
    requires_reapproval: bool = False
    vetoed: bool = False


@dataclass(slots=True)
class PlanDecision:
    """The policy verdict for a whole plan."""

    vetoed: bool
    veto_reasons: list[Reason]
    steps: list[StepDecision]
    scam: bool = False


# --------------------------------------------------------------------- helpers


def _reason(
    code: str,
    *,
    message_en: str | None = None,
    message_zh: str | None = None,
) -> Reason:
    default_en, default_zh = _MESSAGES[code]
    return Reason(
        code=code,
        message_en=message_en or default_en,
        message_zh=message_zh or default_zh,
    )


def _add_reason(reasons: list[Reason], reason: Reason) -> None:
    for existing in reasons:
        if (
            existing.code == reason.code
            and existing.message_en == reason.message_en
            and existing.message_zh == reason.message_zh
        ):
            return
    reasons.append(reason)


def _escalate(decision: StepDecision, tier: Tier) -> None:
    if TIER_RANK[tier] > TIER_RANK[decision.tier]:
        decision.tier = tier


def _veto(
    decision: StepDecision, plan_reasons: list[Reason], reason: Reason
) -> None:
    _add_reason(decision.reasons, reason)
    _add_reason(plan_reasons, reason)
    decision.vetoed = True


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError("now must be a date, datetime or ISO date string")


def _as_datetime(value: date | datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and "T" in value:
        return datetime.fromisoformat(value)
    return datetime.combine(_as_date(value), datetime.min.time())


def _iter_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_strings(item)


def _is_token(value: object) -> bool:
    return isinstance(value, str) and _TOKEN_FULL_RE.fullmatch(value) is not None


def _is_step_ref(value: object) -> bool:
    return isinstance(value, str) and _STEP_REF_FULL_RE.fullmatch(value) is not None


def _resolve(
    vault: TokenVault, session: Session, label: str
) -> TokenEntry | None:
    try:
        return vault.resolve(label, session)
    except (TokenError, KeyError):
        return None


def _persona_id(bank: object) -> str | None:
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


def _daily_limit(bank: object, config: PolicyConfig) -> int:
    if _is_elder(bank, config):
        return config.elder_daily_limit_fen
    return config.daily_limit_fen


def _is_elder(bank: object, config: PolicyConfig) -> bool:
    persona = _persona_id(bank)
    if persona is not None and persona in config.elder_personas:
        return True
    # A customer who switches on elder mode (适老模式) gets the same protections.
    return compliance.elder_mode(
        getattr(bank, "conn", None), str(getattr(bank, "customer_id", ""))
    )


def _merchant_order_amount(
    step: object,
    vault: TokenVault,
    session: Session,
) -> tuple[int | None, bool, bool]:
    """Return ``(price_fen, deferred, bad_item)`` for a merchant order."""

    args = getattr(step, "args", {})
    merchant_label = args.get("merchant")
    if not _is_token(merchant_label):
        return None, False, False
    entry = _resolve(vault, session, merchant_label)
    if entry is None:
        return None, False, False

    item_code = args.get("item")
    try:
        price = price_of(str(entry.real_ref), str(item_code))
    except (KeyError, TypeError):
        return None, False, True
    return int(price), False, False


def _suitability(
    step: object,
    decision: StepDecision,
    plan_reasons: list[Reason],
    bank: object,
    vault: TokenVault,
    session: Session,
    now: date | datetime | str,
) -> None:
    """适当性: veto a product above the customer's tested level, or any purchase
    without a valid (under one year old) questionnaire."""

    conn = getattr(bank, "conn", None)
    now_dt = _as_datetime(now)
    profile = compliance.risk_profile(
        conn, str(getattr(bank, "customer_id", "")), _persona_id(bank), now_dt
    )
    if profile is None or now_dt - profile[1] > compliance.RISK_TEST_VALID:
        _veto(decision, plan_reasons, _reason(Rule.RISK_TEST))
        return
    label = getattr(step, "args", {}).get("product")
    entry = _resolve(vault, session, label) if _is_token(label) else None
    row = (
        conn.execute("SELECT risk_level FROM products WHERE id = ?", (str(entry.real_ref),)).fetchone()
        if entry is not None and conn is not None
        else None
    )
    if row is None or int(row["risk_level"]) > profile[0]:
        _veto(decision, plan_reasons, _reason(Rule.SUITABILITY))


def _bill_amount(step: object, bank: object, today: date) -> int | None:
    conn = getattr(bank, "conn", None)
    if conn is None:
        return None
    return bills.due_amount(
        conn,
        str(getattr(bank, "customer_id", "")),
        str(getattr(step, "args", {}).get("category")),
        today,
    )


def _redemption_amount(
    step: object,
    vault: TokenVault,
    session: Session,
    bank: object,
) -> tuple[int | None, bool, bool]:
    """Return ``(payout_fen, deferred, bad_item)`` from the bank's own quote.

    The planner never supplies a price for a redemption: the payout is whatever
    the bank calculates for the customer's holding.  Any quote failure is a
    bad item, so the step is vetoed instead of running with a guessed amount.
    """

    label = getattr(step, "args", {}).get("product")
    if not _is_token(label):
        return None, False, True
    entry = _resolve(vault, session, label)
    if entry is None or entry.kind != "PROD":
        return None, False, True

    conn = getattr(bank, "conn", None)
    customer_id = str(getattr(bank, "customer_id", "") or "")
    if conn is None or not customer_id:
        return None, False, True
    try:
        quote = redemption.quote(conn, customer_id, str(entry.real_ref))
        payout = int(quote["payout_fen"])
    except (BankError, KeyError, TypeError, ValueError):
        return None, False, True
    if payout <= 0:
        return None, False, True
    return payout, False, False


def _amount_info(
    step: object,
    spec: ToolSpec,
    vault: TokenVault,
    session: Session,
    bank: object = None,
    today: date = date.min,
) -> tuple[int | None, bool, bool]:
    """Return ``(amount_fen, deferred, bad_item)`` for a step."""

    if getattr(step, "tool", "") == "merchant_order":
        return _merchant_order_amount(step, vault, session)
    if getattr(step, "tool", "") == "redeem_product":
        # The payout is the bank's, never the planner's.
        return _redemption_amount(step, vault, session, bank)
    if getattr(step, "tool", "") == "pay_bill":
        # The amount is the bill's, never the planner's.
        return _bill_amount(step, bank, today), False, False

    if "amount" not in spec.money_args:
        return None, False, False
    value = getattr(step, "args", {}).get("amount")
    if _is_step_ref(value):
        return None, True, False
    if not _is_token(value):
        return None, False, False

    entry = _resolve(vault, session, value)
    if entry is None or not isinstance(entry.real_ref, int):
        return None, False, False

    amount = int(entry.real_ref)
    divide_by = getattr(step, "args", {}).get("divide_by")
    if (
        isinstance(divide_by, int)
        and not isinstance(divide_by, bool)
        and divide_by >= 2
    ):
        amount = split_fen(amount, divide_by)
    return amount, False, False


def _payee_label(step: object) -> str | None:
    """Return the payee token; payee step references are never followed."""

    value = getattr(step, "args", {}).get("payee")
    return value if _is_token(value) else None


def _has_raw_token(step: object) -> bool:
    for value in _iter_strings(getattr(step, "args", {})):
        for match in _TOKEN_ANY_RE.finditer(value):
            label = match.group(0)
            kind = label[1:-1].split("_", 1)[0]
            if kind == "RAW":
                return True
    return False


def _money_args(step: object, spec: ToolSpec) -> Iterable[str]:
    args = getattr(step, "args", {})
    for name in spec.money_args:
        value = args.get(name)
        if isinstance(value, str):
            yield value


def _apply_mandate_flags(
    decision: StepDecision,
    *,
    elder: bool,
    config: PolicyConfig,
) -> None:
    tier = decision.tier
    decision.needs_mandate = tier in {"yellow", "red"}
    decision.needs_mfa = tier == "red"
    decision.needs_co_approval = False
    if (
        tier == "red"
        and elder
        and config.red_needs_co_approval_for_elder
    ):
        decision.needs_co_approval = True
        _add_reason(decision.reasons, _reason(Rule.ELDER))


# -------------------------------------------------------------------- evaluate


def evaluate(
    plan: ValidatedPlan,
    *,
    bank: object,
    vault: TokenVault,
    session: Session,
    now: date | datetime | str,
    config: PolicyConfig = PolicyConfig(),
    payments_locked: bool = False,
    scam_flags: frozenset[str] = frozenset(),
) -> PlanDecision:
    """Compute the tier, mandates and vetoes for a validated plan.

    Tiers start from the tool registry and only ever move up.  The plan cannot
    supply a tier: :class:`~security.dag.PlanStep` has no such field.
    """

    now_date = _as_date(now)
    elder = _is_elder(bank, config)
    limit = _daily_limit(bank, config)
    steps = list(plan.steps)
    decisions: list[StepDecision] = []
    plan_reasons: list[Reason] = []
    scam = bool(scam_flags)

    write_count = sum(
        1 for step in steps if REGISTRY[step.tool].kind == "write"
    )
    if write_count > config.max_write_steps:
        _add_reason(plan_reasons, _reason(Rule.TOO_MANY))

    for step in steps:
        spec = REGISTRY[step.tool]
        base_tier: Tier = (
            "green" if spec.base_tier == "none" else spec.base_tier  # type: ignore[assignment]
        )
        decision = StepDecision(
            step_id=step.id,
            tool=step.tool,
            tier=base_tier,
            reasons=[_reason(Rule.BASE)],
        )

        amount, deferred, bad_item = _amount_info(step, spec, vault, session, bank, now_date)
        decision.amount_fen = amount
        decision.deferred = deferred
        if bad_item:
            _veto(decision, plan_reasons, _reason(Rule.BAD_ITEM))
        if deferred:
            _escalate(decision, "yellow")
            _add_reason(decision.reasons, _reason(Rule.DEFERRED))

        if payments_locked and spec.kind == "write" and step.tool not in PAYMENT_LOCK_PROTECTIVE_ACTIONS:
            _veto(decision, plan_reasons, _reason(Rule.LOCKED))

        if scam and _blocked_under_scam(step.tool, spec.kind):
            _veto(decision, plan_reasons, _reason(Rule.SCAM))

        if spec.kind == "write":
            if _has_raw_token(step):
                _veto(decision, plan_reasons, _reason(Rule.RAW))

            for value in _money_args(step, spec):
                if not _is_token(value):
                    continue
                entry = _resolve(vault, session, value)
                if entry is None:
                    _veto(decision, plan_reasons, _reason(Rule.UNVERIFIED))
                elif entry.origin == "untrusted":
                    _veto(decision, plan_reasons, _reason(Rule.UNTRUSTED))

            if step.tool in {"transfer", "schedule_transfer"}:
                label = _payee_label(step)
                entry = (
                    _resolve(vault, session, label) if label is not None else None
                )
                if entry is not None and entry.origin == "untrusted":
                    _veto(decision, plan_reasons, _reason(Rule.UNTRUSTED))
                if entry is not None and compliance.contact_on_fraud_list(
                    getattr(bank, "conn", None),
                    str(getattr(bank, "customer_id", "")),
                    str(entry.real_ref),
                ):
                    # 反电信网络诈骗法: no payment to a listed fraud account.
                    _veto(decision, plan_reasons, _reason(Rule.FRAUD_LIST))
                trust = entry.tags.get("trust") if entry is not None else None
                if trust == "flagged":
                    _veto(
                        decision,
                        plan_reasons,
                        _reason(
                            Rule.NEW_PAYEE,
                            message_en=_FLAGGED_PAYEE_MESSAGE[0],
                            message_zh=_FLAGGED_PAYEE_MESSAGE[1],
                        ),
                    )
                elif trust == "new_payee" and config.new_payee_red:
                    _escalate(decision, "red")
                    _add_reason(decision.reasons, _reason(Rule.NEW_PAYEE))

            if step.tool == "subscribe_product":
                _suitability(step, decision, plan_reasons, bank, vault, session, now)

            if step.tool == "set_card_control" and step.args.get("enabled") is True:
                # Switching a card function back on loosens a protection.
                _escalate(decision, "yellow")
            if step.tool == "pay_bill" and amount is None:
                _veto(decision, plan_reasons, _reason(Rule.BAD_ITEM))

            if step.tool in {"set_card_limit", "create_virtual_card"}:
                if amount is not None and (
                    amount > config.card_limit_red_fen
                    or (elder and amount > config.elder_daily_limit_fen)
                ):
                    _escalate(decision, "red")
                    _add_reason(decision.reasons, _reason(Rule.CARD_LIMIT))

        decisions.append(decision)

    # Cumulative limit: walk the topological order, adding each outgoing
    # amount to what the ledger already shows for today.
    running: int | None = None
    # Scheduled payments are limited per day-of-month: the bank's existing total
    # for that day plus earlier same-day schedules in this plan.
    scheduled_existing: dict[int, int] = {}
    scheduled_planned: dict[int, int] = {}
    for step, decision in zip(steps, decisions):
        if step.tool == "schedule_transfer" and decision.amount_fen is not None:
            day = int(step.args["day_of_month"])
            if day not in scheduled_existing:
                lookup = getattr(bank, "scheduled_total_for_day", None)
                scheduled_existing[day] = (
                    int(lookup(day)) if callable(lookup) else 0
                )
            scheduled_planned[day] = (
                scheduled_planned.get(day, 0) + decision.amount_fen
            )
            if scheduled_existing[day] + scheduled_planned[day] > limit:
                _escalate(decision, "red")
                _add_reason(decision.reasons, _reason(Rule.SCHEDULED))

        if step.tool not in OUTGOING_TOOLS or decision.amount_fen is None:
            continue
        if running is None:
            running = int(bank.daily_outgoing_total(now_date))  # type: ignore[attr-defined]
        running += decision.amount_fen
        if running > limit:
            _escalate(decision, "red")
            _add_reason(decision.reasons, _reason(Rule.DAILY))

    for decision in decisions:
        _apply_mandate_flags(decision, elder=elder, config=config)

    return PlanDecision(
        vetoed=bool(plan_reasons),
        veto_reasons=plan_reasons,
        steps=decisions,
        scam=scam,
    )


# ------------------------------------------------------------- run-time recheck


def recheck_step(
    step_decision: StepDecision,
    *,
    resolved_amount_fen: int,
    bank: object,
    now: date | datetime | str,
    approved_tier: str,
    config: PolicyConfig = PolicyConfig(),
    payments_locked: bool = False,
    scam_flags: frozenset[str] = frozenset(),
    day_of_month: int | None = None,
    scheduled_pending_fen: int = 0,
) -> StepDecision:
    """Re-check one write immediately before execution.

    The daily total is read from the ledger *now*, because other transfers may
    have happened since planning.  A deferred amount is resolved here, so this
    is also where its final tier is decided.
    """

    now_date = _as_date(now)
    spec = REGISTRY[step_decision.tool]
    tier: Tier = step_decision.tier
    if step_decision.deferred and not any(
        reason.code in _PERSISTENT_REASON_CODES
        for reason in step_decision.reasons
    ):
        tier = "green" if spec.base_tier == "none" else spec.base_tier  # type: ignore[assignment]

    reasons = list(step_decision.reasons)
    vetoed = step_decision.vetoed
    amount = int(resolved_amount_fen)

    if payments_locked and spec.kind == "write" and step_decision.tool not in PAYMENT_LOCK_PROTECTIVE_ACTIONS:
        _add_reason(reasons, _reason(Rule.LOCKED))
        vetoed = True

    if scam_flags and _blocked_under_scam(step_decision.tool, spec.kind):
        _add_reason(reasons, _reason(Rule.SCAM))
        vetoed = True

    limit = _daily_limit(bank, config)
    if step_decision.tool in {"set_card_limit", "create_virtual_card"} and (
        amount > config.card_limit_red_fen
        or (_is_elder(bank, config) and amount > config.elder_daily_limit_fen)
    ):
        if TIER_RANK["red"] > TIER_RANK[tier]:
            tier = "red"
        _add_reason(reasons, _reason(Rule.CARD_LIMIT))
    if step_decision.tool in OUTGOING_TOOLS:
        running = int(bank.daily_outgoing_total(now_date)) + amount  # type: ignore[attr-defined]
        if running > limit:
            if TIER_RANK["red"] > TIER_RANK[tier]:
                tier = "red"
            _add_reason(reasons, _reason(Rule.DAILY))
        if step_decision.tool == "schedule_transfer":
            # Per day-of-month limit: what the bank already has scheduled for
            # that day, plus same-day schedules approved alongside this one.
            existing = 0
            lookup = getattr(bank, "scheduled_total_for_day", None)
            if day_of_month is not None and callable(lookup):
                existing = int(lookup(int(day_of_month)))
            scheduled = existing + int(scheduled_pending_fen) + amount
        if step_decision.tool == "schedule_transfer" and scheduled > limit:
            if TIER_RANK["red"] > TIER_RANK[tier]:
                tier = "red"
            _add_reason(reasons, _reason(Rule.SCHEDULED))

    decision = StepDecision(
        step_id=step_decision.step_id,
        tool=step_decision.tool,
        tier=tier,
        amount_fen=amount,
        deferred=False,
        reasons=reasons,
        requires_reapproval=TIER_RANK[tier] > TIER_RANK.get(approved_tier, 0),
        vetoed=vetoed,
    )
    _apply_mandate_flags(decision, elder=_is_elder(bank, config), config=config)
    return decision


def evaluate_early_release(
    *,
    bank: object,
    transfer_id: str,
    now: date | datetime | str,
    config: PolicyConfig = PolicyConfig(),
) -> StepDecision:
    """Policy verdict for releasing one pending transfer before its clock.

    This is the executor's pre-flight for the customer's own "release now"
    control, not a planner tool.  It is always red (MFA, and co-approval for an
    elder) and fails closed: a missing/stale transfer or any release-hold check
    that cannot be completed vetoes the step.
    """

    now_dt = _as_datetime(now)
    decision = StepDecision(
        step_id="release",
        tool="release_transfer_now",
        tier="red",
        reasons=[_reason(Rule.EARLY_RELEASE)],
    )
    plan_reasons: list[Reason] = []
    conn = getattr(bank, "conn", None)
    customer_id = str(getattr(bank, "customer_id", "") or "")

    try:
        if conn is None or not customer_id:
            raise BankError("not_found", "bank connection unavailable")
        row = conn.execute(
            "SELECT * FROM transfers WHERE customer_id = ? AND id = ?",
            (customer_id, transfer_id),
        ).fetchone()
        if row is None or row["status"] != "pending":
            # The customer approved a different state: never release a stale row.
            _veto(decision, plan_reasons, _reason(Rule.EARLY_RELEASE))
        else:
            decision.amount_fen = int(row["amount_fen"])
            if datetime.fromisoformat(row["release_at"]) - datetime.fromisoformat(row["created_at"]) > timedelta(hours=2):
                _veto(decision, plan_reasons, _reason(Rule.EARLY_RELEASE,
                    message_en="A protected longer hold cannot be shortened with Send now.",
                    message_zh="此笔转账有更长的保护等待期，不能立即转出。"))
            hold = compliance.transfer_release_hold(
                conn, customer_id, row["to_contact_id"], now_dt
            )
            if hold is not None:
                code = str(hold[0])
                if code not in _MESSAGES:
                    code = Rule.RELEASE_CHECK_UNAVAILABLE
                _veto(decision, plan_reasons, _reason(code))
    except Exception:  # noqa: BLE001 - any lookup/check failure must fail closed
        _veto(
            decision,
            plan_reasons,
            _reason(Rule.RELEASE_CHECK_UNAVAILABLE),
        )

    # Read settings without schema creation or commits: this gate also runs
    # inside the settlement transaction through a connection proxy.
    elder = _persona_id(bank) in config.elder_personas
    try:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='customer_settings'").fetchone():
            settings = conn.execute(
                "SELECT elder_mode FROM customer_settings WHERE customer_id = ?",
                (customer_id,),
            ).fetchone()
            elder = elder or bool(settings and settings[0])
    except Exception:
        _veto(decision, plan_reasons, _reason(Rule.RELEASE_CHECK_UNAVAILABLE))
    _apply_mandate_flags(decision, elder=elder, config=config)
    return decision
