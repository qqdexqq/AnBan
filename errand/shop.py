"""Mock online shop used by the errand agent (docs/ERRAND_AGENT.md)."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Literal

Category = Literal["flowers", "cake"]

_PRODUCT_ID_RE = re.compile(r"P\d{3}")
_ORDER_ID_RE = re.compile(r"O\d{6}")

_CATEGORY_WORDS: dict[Category, tuple[str, ...]] = {
    "flowers": ("flowers", "花"),
    "cake": ("cake", "蛋糕"),
}

_ABORT_REASONS = (
    "no_suitable_product",
    "over_budget",
    "late_delivery",
    "out_of_stock",
    "suspicious_page",
)

_ACTION_KEYS: dict[str, frozenset[str]] = {
    "search": frozenset({"query"}),
    "view": frozenset({"product_id"}),
    "add_to_cart": frozenset({"product_id", "qty"}),
    "checkout": frozenset({"card_token", "deliver_on"}),
    "finish": frozenset({"order_id"}),
    "abort": frozenset({"reason"}),
}

_PAGE_OPEN = "<<UNTRUSTED_PAGE>>"
_PAGE_CLOSE = "<</UNTRUSTED_PAGE>>"


@dataclass(frozen=True)
class Product:
    id: str
    category: Category
    name: str
    tags: tuple[str, ...]
    price_fen: int
    delivery_fee_fen: int
    earliest_delivery: date
    in_stock: bool
    seller: str
    description: str


@dataclass(frozen=True)
class Order:
    order_id: str
    product_id: str
    qty: int
    total_fen: int
    deliver_on: date
    card_token: str


@dataclass
class StepResult:
    observation: str
    done: bool
    order: Order | None = None
    error: str | None = None


def yuan(fen: int) -> str:
    return f"{fen // 100}.{fen % 100:02d}"


def _bad_action() -> StepResult:
    return StepResult(observation="ERROR: bad_action", done=False, error="bad_action")


def _valid_product_id(value: object) -> bool:
    return isinstance(value, str) and _PRODUCT_ID_RE.fullmatch(value) is not None


def _wrap(body: str) -> str:
    safe = body.replace("<<", "«").replace(">>", "»")
    return f"{_PAGE_OPEN}\n{safe}\n{_PAGE_CLOSE}"


def _product_line(p: Product) -> str:
    stock = "in stock" if p.in_stock else "out of stock"
    return (
        f"{p.id} | {p.name} | price ¥{yuan(p.price_fen)}"
        f" | delivery ¥{yuan(p.delivery_fee_fen)}"
        f" | earliest {p.earliest_delivery.isoformat()}"
        f" | {stock} | {p.seller}"
    )


def _product_page(p: Product) -> str:
    return "\n".join(
        (
            f"ID: {p.id}",
            f"Category: {p.category}",
            f"Name: {p.name}",
            f"Tags: {', '.join(p.tags)}",
            f"Price: ¥{yuan(p.price_fen)}",
            f"Delivery: ¥{yuan(p.delivery_fee_fen)}",
            f"Earliest delivery: {p.earliest_delivery.isoformat()}",
            f"Stock: {'in stock' if p.in_stock else 'out of stock'}",
            f"Seller: {p.seller}",
            f"Description: {p.description}",
        )
    )


# charge(card_token, amount_fen, order_id) -> None if approved, else a decline code.
ChargeFn = Callable[[str, int, str], "str | None"]


class MockShop:
    def __init__(
        self,
        products: list[Product],
        today: date,
        *,
        charge: ChargeFn | None = None,
        checkout_surcharge_fen: int = 0,
        timeout_after_charge: bool = False,
    ):
        self._products = list(products)
        ids = [p.id for p in self._products]
        if any(
            not isinstance(product_id, str)
            or _PRODUCT_ID_RE.fullmatch(product_id) is None
            for product_id in ids
        ):
            raise ValueError("product id must look like P000")
        if len(set(ids)) != len(ids):
            raise ValueError("product ids must be unique")
        self._by_id = {p.id: p for p in self._products}
        self._today = today
        self._cart: dict[str, int] = {}
        self._orders: list[Order] = []
        self._next_order = 1
        # The card network: the shop must get the bank's approval to be paid.
        self._charge = charge
        # Fault injection for demos: a fee that appears only at checkout, and a
        # lost response after the payment went through.
        self._surcharge = checkout_surcharge_fen
        self._timeout_after_charge = timeout_after_charge

    # -- validation -----------------------------------------------------
    def _validate(self, action: object) -> str | None:
        if not isinstance(action, dict):
            return None
        name = action.get("action")
        if not isinstance(name, str) or name not in _ACTION_KEYS:
            return None
        if set(action) != _ACTION_KEYS[name] | {"action"}:
            return None
        if name == "search":
            query = action["query"]
            if not isinstance(query, str):
                return None
            query = query.strip()
            if not 1 <= len(query) <= 40:
                return None
        elif name == "view":
            if not _valid_product_id(action["product_id"]):
                return None
        elif name == "add_to_cart":
            if not _valid_product_id(action["product_id"]):
                return None
            qty = action["qty"]
            if not isinstance(qty, int) or isinstance(qty, bool):
                return None
            if not 1 <= qty <= 3:
                return None
        elif name == "checkout":
            if not isinstance(action["card_token"], str):
                return None
            raw = action["deliver_on"]
            if not isinstance(raw, str):
                return None
            try:
                date.fromisoformat(raw)
            except ValueError:
                return None
        elif name == "finish":
            order_id = action["order_id"]
            if (
                not isinstance(order_id, str)
                or _ORDER_ID_RE.fullmatch(order_id) is None
            ):
                return None
        elif name == "abort":
            if action["reason"] not in _ABORT_REASONS:
                return None
        return name

    # -- search ---------------------------------------------------------
    def _word_matches(self, p: Product, word: str) -> bool:
        for category, words in _CATEGORY_WORDS.items():
            if word in words:
                return p.category == category
        return word in p.tags or word in p.name.lower()

    def _search(self, query: str) -> StepResult:
        words = query.strip().lower().split()
        matches = [
            p
            for p in self._products
            if any(self._word_matches(p, word) for word in words)
        ]
        matches.sort(key=lambda p: p.id)
        matches = matches[:8]
        body = "\n".join(_product_line(p) for p in matches) if matches else "No results."
        return StepResult(observation=_wrap(body), done=False)

    def _view(self, product_id: str) -> StepResult:
        product = self._by_id.get(product_id)
        if product is None:
            return StepResult(
                observation="ERROR: unknown_product",
                done=False,
                error="unknown_product",
            )
        return StepResult(observation=_wrap(_product_page(product)), done=False)

    # -- cart -----------------------------------------------------------
    def _add_to_cart(self, product_id: str, qty: int) -> StepResult:
        product = self._by_id.get(product_id)
        if product is None:
            return StepResult(
                observation="ERROR: unknown_product",
                done=False,
                error="unknown_product",
            )
        if not product.in_stock:
            return StepResult(
                observation="ERROR: out_of_stock",
                done=False,
                error="out_of_stock",
            )
        self._cart = {product_id: qty}
        total = product.price_fen * qty + product.delivery_fee_fen
        return StepResult(
            observation=f"Cart: {product_id} x{qty}, total ¥{yuan(total)}",
            done=False,
        )

    def _checkout(self, card_token: str, deliver_on: date) -> StepResult:
        if not self._cart:
            return StepResult(
                observation="ERROR: empty_cart", done=False, error="empty_cart"
            )
        product_id, qty = next(iter(self._cart.items()))
        product = self._by_id[product_id]
        if deliver_on < product.earliest_delivery:
            return StepResult(
                observation="ERROR: too_early", done=False, error="too_early"
            )
        total = product.price_fen * qty + product.delivery_fee_fen + self._surcharge
        order_id = f"O{self._next_order:06d}"
        if self._charge is not None:
            declined = self._charge(card_token, total, order_id)
            if declined is not None:
                return StepResult(
                    observation=f"ERROR: payment_declined ({declined})",
                    done=False,
                    error="payment_declined",
                )
        order = Order(
            order_id=order_id,
            product_id=product_id,
            qty=qty,
            total_fen=total,
            deliver_on=deliver_on,
            card_token=card_token,
        )
        self._next_order += 1
        self._orders.append(order)
        self._cart = {}
        if self._timeout_after_charge:
            return StepResult(
                observation="ERROR: timeout (no confirmation received)",
                done=False,
                error="timeout",
            )
        return StepResult(
            observation=(
                f"Order {order.order_id} placed: {product_id} x{qty}, "
                f"total ¥{yuan(total)}, deliver on {deliver_on.isoformat()}"
            ),
            done=False,
        )

    # -- public API -----------------------------------------------------
    def step(self, action: dict) -> StepResult:
        name = self._validate(action)
        if name is None:
            return _bad_action()
        if name == "search":
            return self._search(action["query"])
        if name == "view":
            return self._view(action["product_id"])
        if name == "add_to_cart":
            return self._add_to_cart(action["product_id"], action["qty"])
        if name == "checkout":
            return self._checkout(
                action["card_token"], date.fromisoformat(action["deliver_on"])
            )
        if name == "finish":
            for order in self._orders:
                if order.order_id == action["order_id"]:
                    return StepResult(
                        observation=f"Finished order {order.order_id}.",
                        done=True,
                        order=order,
                    )
            return StepResult(
                observation="ERROR: unknown_order", done=False, error="unknown_order"
            )
        # abort
        return StepResult(observation=f"Aborted: {action['reason']}", done=True)

    def cart_quote(self) -> tuple[list[tuple[Product, int]], int]:
        items = [
            (self._by_id[product_id], qty) for product_id, qty in self._cart.items()
        ]
        total = sum(
            product.price_fen * qty + product.delivery_fee_fen
            for product, qty in items
        )
        return items, total

    def orders(self) -> list[Order]:
        return list(self._orders)
