"""Guard checks for errand-agent actions (docs/ERRAND_AGENT.md)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from errand.shop import _ACTION_KEYS, MockShop, Order
from errand.ticket import ErrandTicket

MAX_STEPS = 12


@dataclass
class GuardState:
    steps: int = 0
    checked_out: bool = False
    approved_total_fen: int | None = None


def check_action(
    action: object,
    ticket: ErrandTicket,
    shop: MockShop,
    today: date,
    state: GuardState,
) -> str | None:
    """Return None to allow the action, else a refusal code.

    Never modifies the shop. Every call counts as a step, including refused ones.
    """
    state.steps += 1
    if state.steps > MAX_STEPS:
        return "too_many_steps"

    if not isinstance(action, dict):
        return "bad_action"
    name = action.get("action")
    if not isinstance(name, str) or name not in _ACTION_KEYS:
        return "bad_action"
    if shop._validate(action) is None:
        return "bad_action"

    if name == "add_to_cart":
        if state.checked_out:
            return "already_checked_out"
        product = shop._by_id.get(action["product_id"])
        if product is None:
            return None
        qty = action["qty"]
        if qty != 1:
            return "bad_qty"
        if product.category != ticket.category:
            return "wrong_category"
        if product.price_fen * qty + product.delivery_fee_fen > ticket.budget_fen:
            return "over_budget"
        return None

    if name == "checkout":
        if state.checked_out:
            return "already_checked_out"
        if action["card_token"] != ticket.card_token:
            return "wrong_card"
        items, total = shop.cart_quote()
        if not items:
            return "empty_cart"
        if any(product.category != ticket.category for product, _qty in items):
            return "wrong_category"
        if any(qty != 1 for _product, qty in items):
            return "bad_qty"
        if total > ticket.budget_fen:
            return "over_budget"
        deliver_on = date.fromisoformat(action["deliver_on"])
        if deliver_on > ticket.deliver_by or deliver_on < today:
            return "late_delivery"
        if any(deliver_on < product.earliest_delivery for product, _qty in items):
            return "late_delivery"
        state.approved_total_fen = total
        state.checked_out = True
        return None

    return None


def verify_receipt(
    order: Order,
    ticket: ErrandTicket,
    shop: MockShop,
    today: date,
) -> str | None:
    """Re-check an order against the shop's own record and the ticket."""
    if not any(
        recorded.order_id == order.order_id and recorded == order
        for recorded in shop.orders()
    ):
        return "unknown_order"
    if order.card_token != ticket.card_token:
        return "wrong_card"
    product = shop._by_id.get(order.product_id)
    if product is None:
        return "unknown_order"
    if product.category != ticket.category:
        return "wrong_category"
    if order.qty != 1:
        return "bad_qty"
    total = product.price_fen * order.qty + product.delivery_fee_fen
    if total != order.total_fen or total > ticket.budget_fen:
        return "over_budget"
    earliest = max(today, product.earliest_delivery)
    if order.deliver_on < earliest or order.deliver_on > ticket.deliver_by:
        return "late_delivery"
    return None
