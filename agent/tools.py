"""Closed allowlist tool registry shared by planner and executor.

Every planner tool name and its argument schema lives here.  Argument values are
semantic tokens or step references, never real names, amounts or account
numbers; the few literal arguments are small enums or short strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from bank.categories import CATEGORIES
from bank.merchants import CATALOGUE

MERCHANT_ITEM_CODES: tuple[str, ...] = tuple(
    dict.fromkeys(code for items in CATALOGUE.values() for code in items)
)

ToolKind = Literal["read", "write", "control"]
BaseTier = Literal["green", "yellow", "red", "none"]

# A token label is ``<KIND_n>``; ACCT uses letters instead of digits.
# Patterns are ASCII-only: pydantic's regex treats ``\d`` as any Unicode digit,
# so ``[0-9]`` is required to reject things like ``<PAYEE_１>``.
TOKEN_RE = r"<[A-Z]+_[A-Za-z0-9]+>"
STEP_REF_RE = r"\$s[0-9]+\.[a-z_]+"


class TokenKind:
    """Metadata marker: which semantic-token kind an argument accepts."""

    __slots__ = ("kind",)

    def __init__(self, kind: str) -> None:
        self.kind = kind


def _token_pattern(kind: str, *, field: str | None = None) -> str:
    token = rf"<{kind}_[0-9]+>"
    if field is None:
        return rf"^{token}$"
    return rf"^(?:{token}|\$s[0-9]+\.{field})$"


PayeeArg = Annotated[
    str,
    StringConstraints(pattern=_token_pattern("PAYEE")),
    TokenKind("PAYEE"),
]

AmountArg = Annotated[
    str,
    StringConstraints(pattern=_token_pattern("AMT", field="amount")),
    TokenKind("AMT"),
]

_DateToken = Annotated[
    str,
    StringConstraints(pattern=_token_pattern("DATE", field="date")),
    TokenKind("DATE"),
]

SubArg = Annotated[
    str,
    StringConstraints(pattern=r"^<SUB_[0-9]+>$"),
    TokenKind("SUB"),
]
CardArg = Annotated[
    str,
    StringConstraints(pattern=r"^<CARD_[0-9]+>$"),
    TokenKind("CARD"),
]
ProdArg = Annotated[
    str,
    StringConstraints(pattern=r"^<PROD_[0-9]+>$"),
    TokenKind("PROD"),
]
MerchantArg = Annotated[
    str,
    StringConstraints(pattern=r"^<MERCHANT_[0-9]+>$"),
    TokenKind("MERCHANT"),
]
AcctArg = Annotated[
    str,
    StringConstraints(pattern=r"^<ACCT_[A-Z]>$"),
    TokenKind("ACCT"),
]

NoteArg = Annotated[str, StringConstraints(max_length=40)]
ItemArg = Annotated[str, StringConstraints(min_length=1, max_length=40)]
PotNameArg = Annotated[
    str, StringConstraints(min_length=1, max_length=20, pattern=r"^[\w一-鿿 ]+$")
]
Period = Literal[
    "this_month", "last_month", "this_year", "last_year", "last_7_days", "last_30_days"
] | Annotated[str, StringConstraints(pattern=r"^(?:month:[0-9]{4}-(?:0[1-9]|1[0-2])|year:[0-9]{4})$")]


class RunAnalysisArgs(BaseModel):
    """Read-only sandbox request; code never accepts a free-form program here."""

    model_config = ConfigDict(extra="forbid", strict=True)
    period: Period
    kind: Literal["weekend_vs_weekday", "by_hour", "by_weekday", "custom"]


Category = Literal[*CATEGORIES]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RelativeDate(_Args):
    date: _DateToken
    offset_days: int = Field(ge=-60, le=60)


DateArg = _DateToken | RelativeDate


class EmptyArgs(_Args):
    pass


class ListTransactionsArgs(_Args):
    period: Period | None = None
    category: Category | None = None
    payee: PayeeArg | None = None
    direction: Literal["in", "out"] | None = None
    merchant: MerchantArg | None = None
    merchant_ref: Annotated[str, StringConstraints(pattern=r"^txn:[A-Za-z0-9_-]{1,80}$")] | None = None
    explain_anomalies: bool = False


class SpendingReportArgs(_Args):
    period: Period
    category: Category | None = None
    payee: PayeeArg | None = None
    merchant: MerchantArg | None = None
    merchant_ref: Annotated[str, StringConstraints(pattern=r"^txn:[A-Za-z0-9_-]{1,80}$")] | None = None
    mode: Literal["summary", "report", "income", "highest_month"] = "summary"
    compare: bool = False


class ListSubscriptionsArgs(_Args):
    scope: Literal["all", "additional"] = "all"


class FindContactArgs(_Args):
    query: PayeeArg


class GetContactDateArgs(_Args):
    payee: PayeeArg
    what: Literal["birthday"]


class CanCoverArgs(_Args):
    amount: AmountArg


class SetReminderArgs(_Args):
    when: DateArg | None = None
    note: NoteArg | None = None


class TransferArgs(_Args):
    payee: PayeeArg
    amount: AmountArg
    note: NoteArg | None = None
    divide_by: int | None = Field(default=None, ge=2, le=20)


class ScheduleTransferArgs(_Args):
    payee: PayeeArg
    amount: AmountArg
    # The planner may pass the user's <DATE_n>; agent/graph.py replaces it with
    # the day of that date before policy and execution see the plan.
    day_of_month: Annotated[int, Field(ge=1, le=28)] | _DateToken
    note: NoteArg | None = None


class StopStandingOrderArgs(_Args):
    payee: PayeeArg


class RequestPaymentArgs(_Args):
    payee: PayeeArg
    amount: AmountArg
    divide_by: int | None = Field(default=None, ge=2, le=20)
    note: NoteArg | None = None


class CancelSubscriptionArgs(_Args):
    sub: SubArg


class MoveToPotArgs(_Args):
    pot_name: PotNameArg
    amount: AmountArg


class MerchantOrderArgs(_Args):
    merchant: MerchantArg
    item: Literal[*MERCHANT_ITEM_CODES]
    deliver: DateArg
    funding_pot: Annotated[
        str,
        StringConstraints(pattern=_token_pattern("POT", field="pot")),
        TokenKind("POT"),
    ] | None = None


class FreezeCardArgs(_Args):
    card: CardArg | None = None


class SetCardLimitArgs(_Args):
    card: CardArg | None = None
    amount: AmountArg


class CreateVirtualCardArgs(_Args):
    amount: AmountArg


class ReportCardLostArgs(_Args):
    card: CardArg | None = None


class SubscribeProductArgs(_Args):
    product: ProdArg
    amount: AmountArg


class RedeemProductArgs(_Args):
    product: ProdArg


class AskUserArgs(_Args):
    question: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    options: list[PayeeArg | SubArg | CardArg] = Field(max_length=5)


class RefuseArgs(_Args):
    reason: Literal["out_of_scope", "not_account_owner", "not_supported"]


class HandoffHumanArgs(_Args):
    reason: NoteArg | None = None


BillCategory = Literal[
    "water", "electricity", "gas", "mobile", "broadband", "property"
]


class PayBillArgs(_Args):
    category: BillCategory


class InstallmentQuoteArgs(_Args):
    amount: AmountArg
    periods: Literal[3, 6, 12, 24]


class DepositQuoteArgs(_Args):
    amount: AmountArg
    term_months: Literal[3, 6, 12, 24, 36, 60]


class ForecastArgs(_Args):
    days: Annotated[int, Field(ge=7, le=60)] = 30


class HelpAnswerArgs(_Args):
    topic: Literal[
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
    ]


class CardControlArgs(_Args):
    control: Literal["online", "abroad", "atm", "contactless"]
    enabled: bool
    card: CardArg | None = None


class MerchantBlockArgs(_Args):
    category: Literal["gambling", "crypto", "lottery", "adult", "cash_like"]


class VerifyPayeeArgs(_Args):
    payee: PayeeArg


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One allowlisted planner tool."""

    name: str
    kind: ToolKind
    base_tier: BaseTier
    args_model: type[BaseModel]
    outputs: frozenset[str] = field(default_factory=frozenset)
    undo: str | None = None
    money_args: tuple[str, ...] = ()
    trained: bool = True


def _spec(
    name: str,
    kind: ToolKind,
    base_tier: BaseTier,
    args_model: type[BaseModel],
    outputs: str | frozenset[str] = frozenset(),
    undo: str | None = None,
    money_args: tuple[str, ...] = (),
    *,
    trained: bool = True,
) -> ToolSpec:
    declared = frozenset(outputs.split()) if isinstance(outputs, str) else outputs
    return ToolSpec(
        name=name,
        kind=kind,
        base_tier=base_tier,
        args_model=args_model,
        outputs=declared,
        undo=undo,
        money_args=money_args,
        trained=trained,
    )


REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        # ------------------------------------------------------------- reads
        _spec("get_balance", "read", "green", EmptyArgs, "balance"),
        _spec("run_analysis", "read", "green", RunAnalysisArgs, "analysis", trained=False),
        _spec(
            "list_transactions",
            "read",
            "green",
            ListTransactionsArgs,
            "amount",
        ),
        _spec(
            "spending_report",
            "read",
            "green",
            SpendingReportArgs,
            "total by_category",
        ),
        _spec("detect_anomalies", "read", "green", EmptyArgs, "anomalies"),
        _spec("list_subscriptions", "read", "green", ListSubscriptionsArgs, "sub"),
        _spec("list_cards", "read", "green", EmptyArgs, "card"),
        _spec("list_products", "read", "green", EmptyArgs, "product"),
        _spec("get_portfolio", "read", "green", EmptyArgs, "portfolio"),
        _spec("find_contact", "read", "green", FindContactArgs),
        # "Who can I send money to?" / "我的联系人有哪些" (customer feedback Oct 26).
        _spec("list_contacts", "read", "green", EmptyArgs, "payee"),
        _spec("get_contact_date", "read", "green", GetContactDateArgs, "date"),
        _spec("can_cover", "read", "green", CanCoverArgs, "ok"),
        _spec("sandbox_chart", "read", "green", EmptyArgs, "chart"),
        _spec("set_reminder", "write", "green", SetReminderArgs, "reminder"),
        # ------------------------------------------------------------ writes
        _spec(
            "transfer",
            "write",
            "yellow",
            TransferArgs,
            "transfer",
            undo="cancel_transfer",
            money_args=("payee", "amount"),
        ),
        _spec(
            "schedule_transfer",
            "write",
            "yellow",
            ScheduleTransferArgs,
            "scheduled",
            undo="cancel_scheduled",
            money_args=("payee", "amount"),
        ),
        _spec(
            "request_payment",
            "write",
            "yellow",
            RequestPaymentArgs,
            "request",
            undo="cancel_request",
            money_args=("payee", "amount"),
        ),
        _spec(
            "cancel_subscription",
            "write",
            "yellow",
            CancelSubscriptionArgs,
            "subscription",
            undo="restore_subscription",
        ),
        _spec(
            "move_to_pot",
            "write",
            "yellow",
            MoveToPotArgs,
            "pot",
            undo="move_from_pot",
            money_args=("amount",),
        ),
        _spec(
            "merchant_order",
            "write",
            "yellow",
            MerchantOrderArgs,
            "order",
            undo="cancel_order",
            money_args=("merchant",),
        ),
        _spec(
            "freeze_card",
            "write",
            "yellow",
            FreezeCardArgs,
            "card",
            undo="unfreeze_card",
        ),
        _spec(
            "set_card_limit",
            "write",
            "yellow",
            SetCardLimitArgs,
            "card",
            undo="restore_card_limit",
            money_args=("amount",),
        ),
        _spec(
            "create_virtual_card",
            "write",
            "yellow",
            CreateVirtualCardArgs,
            "card",
            undo="close_virtual_card",
            money_args=("amount",),
        ),
        _spec("report_card_lost", "write", "red", ReportCardLostArgs, "card"),
        _spec(
            "subscribe_product",
            "write",
            "red",
            SubscribeProductArgs,
            "holding",
            money_args=("amount",),
        ),
        _spec("change_password", "write", "red", EmptyArgs),
        # ----------------------------------------------------------- control
        _spec("ask_user", "control", "none", AskUserArgs),
        _spec("refuse", "control", "none", RefuseArgs),
        _spec("handoff_human", "control", "none", HandoffHumanArgs),
        # -------------------------------------------------------------- undo
        _spec("cancel_transfer", "write", "green", EmptyArgs),
        _spec("cancel_scheduled", "write", "green", EmptyArgs),
        _spec("cancel_request", "write", "green", EmptyArgs),
        _spec("restore_subscription", "write", "green", EmptyArgs),
        _spec("move_from_pot", "write", "green", EmptyArgs),
        _spec("cancel_order", "write", "green", EmptyArgs),
        # Also a plain request ("把卡解冻"), so it names the card like freeze_card.
        _spec("unfreeze_card", "write", "yellow", FreezeCardArgs),
        _spec("restore_card_limit", "write", "green", EmptyArgs),
        _spec("close_virtual_card", "write", "green", EmptyArgs),
        # --------------------------------------------------------- untrained
        _spec("list_bills", "read", "green", EmptyArgs, "bill", trained=False),
        _spec("list_reminders", "read", "green", EmptyArgs, "reminder", trained=False),
        _spec("list_scheduled", "read", "green", EmptyArgs, "scheduled", trained=False),
        # A wrong stop silently misses rent, so the customer confirms first.
        _spec(
            "stop_standing_order",
            "write",
            "yellow",
            StopStandingOrderArgs,
            "scheduled",
            trained=False,
        ),
        _spec(
            "reactivate_subscription",
            "write",
            "yellow",
            CancelSubscriptionArgs,
            "subscription",
            trained=False,
        ),
        _spec("pay_bill", "write", "yellow", PayBillArgs, "bill", trained=False),
        _spec(
            "installment_quote",
            "read",
            "green",
            InstallmentQuoteArgs,
            "quote",
            trained=False,
        ),
        _spec(
            "deposit_quote",
            "read",
            "green",
            DepositQuoteArgs,
            "quote",
            trained=False,
        ),
        _spec(
            "forecast_cashflow",
            "read",
            "green",
            ForecastArgs,
            "forecast",
            trained=False,
        ),
        _spec(
            "help_answer",
            "read",
            "green",
            HelpAnswerArgs,
            "answer",
            trained=False,
        ),
        _spec(
            "list_transfers",
            "read",
            "green",
            EmptyArgs,
            "transfer",
            trained=False,
        ),
        _spec(
            "verify_payee",
            "read",
            "green",
            VerifyPayeeArgs,
            "risk",
            trained=False,
        ),
        _spec(
            "set_card_control",
            "write",
            "green",
            CardControlArgs,
            "card",
            trained=False,
        ),
        _spec(
            "block_merchant",
            "write",
            "green",
            MerchantBlockArgs,
            "block",
            trained=False,
        ),
        _spec(
            "unblock_merchant",
            "write",
            "yellow",
            MerchantBlockArgs,
            "block",
            trained=False,
        ),
        _spec(
            "redeem_product",
            "write",
            "red",
            RedeemProductArgs,
            "amount product",
            trained=False,
        ),
    )
}

# Tools that only make sense as the ``kind`` of a planner output, never as a
# step inside a plan.
CONTROL_PLAN_TOOLS = frozenset({"ask_user", "refuse"})
