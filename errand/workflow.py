"""Trusted bank-side orchestration of one delegated purchase (a saga).

The errand agent only ever sees the signed ticket and a ``<VCARD_n>`` token.
Card ids, the signing key and the delivery address stay on the bank side.  The
workflow opens a one-time card, lets the agent shop, and then reconciles against
the bank ledger and the shop's own order record.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

from bank.api import BankError
from bank.errand_payments import (
    authorize_charge,
    close_errand,
    errand_status,
    open_errand,
)
from errand.agent import ErrandModel, run_errand
from errand.guard import GuardState
from errand.shop import MockShop, Order, Product
from errand.ticket import issue_ticket, verify_ticket

CARD_TOKEN = "<VCARD_1>"
_DELIVERY_REF_RE = re.compile(r"^<ADDR_[1-9][0-9]{0,2}>$")


@dataclass
class WorkflowResult:
    outcome: str  # "bought" | "bought_reconciled" | "aborted" | "failed" | "needs_review"
    ticket_id: str
    charged_fen: int
    order: Order | None
    abort_reason: str | None
    refusals: list[str]
    declines: list[str]
    delivery_ref: str
    events: list[dict]
    transcript: list[dict]


def _reconcile(
    conn,
    customer_id,
    ticket_id: str,
    shop: MockShop,
    agent_status: str | None,
) -> tuple[str, int, Order | None] | None:
    """Match the bank ledger with the merchant's own order record.

    Returns None when nothing was charged.  Money moved but no matching
    merchant order means a human must decide the refund: never auto-retry.
    """
    st = errand_status(conn, customer_id, ticket_id)
    if st["status"] != "charged":
        return None
    charged_fen = int(st["charged_fen"] or 0)
    order_ref = st["order_ref"]
    merchant_order = next(
        (o for o in shop.orders() if o.order_id == order_ref), None
    )
    if merchant_order is not None and merchant_order.total_fen == st["charged_fen"]:
        outcome = "bought" if agent_status == "bought" else "bought_reconciled"
        return outcome, charged_fen, merchant_order
    return "needs_review", charged_fen, None


def run_bank_errand(
    conn,
    customer_id,
    *,
    category,
    budget_fen,
    deliver_by,
    note,
    now: datetime,
    key: bytes,
    model: ErrandModel,
    products: list[Product],
    delivery_ref: str = "<ADDR_1>",
    ticket_id: str | None = None,
    checkout_surcharge_fen: int = 0,
    timeout_after_charge: bool = False,
    merchant: str = "AnBan demo shop",
    clock: Callable[[], datetime] | None = None,
    state: GuardState | None = None,
) -> WorkflowResult:
    if ticket_id is None:
        ticket_id = "T" + secrets.token_hex(5)
    if (
        not isinstance(delivery_ref, str)
        or _DELIVERY_REF_RE.fullmatch(delivery_ref) is None
    ):
        raise ValueError("delivery_ref must look like <ADDR_n>")
    if not isinstance(deliver_by, date) or isinstance(deliver_by, datetime):
        raise ValueError("deliver_by must be a date")
    if not isinstance(key, bytes) or not key:
        raise ValueError("key must be non-empty bytes")

    # Validate and sign the ticket before any card exists: a bad ticket must
    # never leave an open virtual card behind.
    ticket, signature = issue_ticket(
        category=category,
        budget_fen=budget_fen,
        deliver_by=deliver_by,
        now=now,
        key=key,
        card_token=CARD_TOKEN,
        note=note,
        ticket_id=ticket_id,
    )
    verify_ticket(ticket, signature, key, now)

    events: list[dict] = []
    refusals: list[str] = []
    declines: list[str] = []
    res = None
    outcome = "failed"
    order: Order | None = None
    abort_reason: str | None = None
    charged_fen = 0
    guard_state = state if state is not None else GuardState()
    shop: MockShop | None = None

    opened = open_errand(
        conn,
        customer_id,
        ticket_id=ticket_id,
        category=category,
        budget_fen=budget_fen,
        deliver_by=deliver_by,
        expires_at=ticket.expires_at,
        now=now,
    )

    try:
        events.append(
            {"step": "opened", "ticket_id": ticket_id, "budget_fen": budget_fen}
        )

        # Bank-side only: the model sees the token, never the card id.
        tokens = {CARD_TOKEN: opened["card_id"]}

        def charge(token: str, amount_fen: int, order_id: str) -> str | None:
            card_id = tokens.get(token)
            if card_id is None:
                declines.append("wrong_card")
                events.append({"step": "declined", "code": "wrong_card"})
                return "wrong_card"
            if (
                guard_state.approved_total_fen is None
                or amount_fen != guard_state.approved_total_fen
            ):
                # The merchant tried to charge more than the guard-approved total.
                declines.append("terms_changed")
                events.append({"step": "declined", "code": "terms_changed"})
                return "terms_changed"
            try:
                receipt = authorize_charge(
                    conn,
                    customer_id,
                    ticket_id=ticket_id,
                    card_id=card_id,
                    amount_fen=amount_fen,
                    merchant=merchant,
                    order_ref=order_id,
                    now=clock() if clock is not None else now,
                )
            except BankError as exc:
                declines.append(exc.code)
                events.append({"step": "declined", "code": exc.code})
                return exc.code
            events.append(
                {
                    "step": "charged",
                    "order_ref": receipt["order_ref"],
                    "charged_fen": receipt["charged_fen"],
                    "replayed": receipt["replayed"],
                }
            )
            return None

        shop = MockShop(
            products,
            now.date(),
            charge=charge,
            checkout_surcharge_fen=checkout_surcharge_fen,
            timeout_after_charge=timeout_after_charge,
        )

        res = run_errand(ticket, shop, model, now.date(), state=guard_state)
        refusals = list(res.refusals)
        abort_reason = res.abort_reason
        events.append(
            {
                "step": "agent_done",
                "status": res.status,
                "steps": res.steps,
            }
        )

        reconciled = _reconcile(conn, customer_id, ticket_id, shop, res.status)
        if reconciled is None:
            # Compensation: close the unused card, nothing was charged.
            close_errand(conn, customer_id, ticket_id=ticket_id, now=now)
            events.append({"step": "closed"})
            outcome = "aborted" if res.status == "aborted" else "failed"
        else:
            outcome, charged_fen, order = reconciled
            abort_reason = None
            if outcome == "needs_review":
                events.append({"step": "escalated"})
            else:
                events.append(
                    {"step": "reconciled", "order_ref": order.order_id}
                )
    except BaseException as exc:
        events.append({"step": "error", "type": type(exc).__name__})
        try:
            st = errand_status(conn, customer_id, ticket_id)
            if st["status"] == "issued":
                close_errand(conn, customer_id, ticket_id=ticket_id, now=now)
            reconciled = (
                _reconcile(
                    conn,
                    customer_id,
                    ticket_id,
                    shop,
                    res.status if res is not None else None,
                )
                if shop is not None
                else None
            )
        except Exception:
            reconciled = None
            charged_fen = 0
        else:
            if reconciled is None:
                charged_fen = 0
            else:
                outcome, charged_fen, order = reconciled
                abort_reason = None
                if outcome == "needs_review":
                    events.append({"step": "escalated"})
                else:
                    events.append(
                        {"step": "reconciled", "order_ref": order.order_id}
                    )
        if res is not None:
            refusals = list(res.refusals)
            if charged_fen == 0:
                abort_reason = res.abort_reason
        if not isinstance(exc, Exception):
            raise

    return WorkflowResult(
        outcome=outcome,
        ticket_id=ticket_id,
        charged_fen=charged_fen,
        order=order,
        abort_reason=abort_reason,
        refusals=refusals,
        declines=declines,
        delivery_ref=delivery_ref,
        events=events,
        transcript=list(res.transcript) if res is not None else [],
    )
