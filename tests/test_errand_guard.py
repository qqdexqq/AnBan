"""Tests for errand/guard.py."""

from __future__ import annotations

from datetime import date, datetime

from errand.guard import GuardState, check_action, verify_receipt
from errand.shop import MockShop, Order, Product
from errand.ticket import issue_ticket

NOW = datetime(2026, 11, 1, 9, 0, 0)
TODAY = date(2026, 11, 1)
DELIVER_BY = date(2026, 11, 4)
KEY = b"errand-guard-test-key-0123456789"
CARD = "<VCARD_7>"


def _product(**overrides) -> Product:
    base = dict(
        id="P001",
        category="flowers",
        name="Red roses",
        tags=("roses", "red"),
        price_fen=8800,
        delivery_fee_fen=500,
        earliest_delivery=TODAY,
        in_stock=True,
        seller="Bloom Shop",
        description="Fresh flowers.",
    )
    base.update(overrides)
    return Product(**base)


def _shop() -> MockShop:
    return MockShop(
        [
            _product(),
            _product(
                id="P002",
                category="cake",
                name="Chocolate cake",
                tags=("cake",),
                price_fen=5000,
            ),
            _product(
                id="P003",
                name="Luxury bouquet",
                price_fen=35000,
            ),
        ],
        TODAY,
    )


def _ticket():
    ticket, _signature = issue_ticket(
        category="flowers",
        budget_fen=30000,
        deliver_by=DELIVER_BY,
        now=NOW,
        key=KEY,
        card_token=CARD,
    )
    return ticket


def _stock_cart(shop: MockShop, product_id: str = "P001") -> None:
    result = shop.step(
        {"action": "add_to_cart", "product_id": product_id, "qty": 1}
    )
    assert result.error is None


def _checkout(card: str = CARD, deliver_on: date = DELIVER_BY) -> dict:
    return {
        "action": "checkout",
        "card_token": card,
        "deliver_on": deliver_on.isoformat(),
    }


def _state() -> GuardState:
    return GuardState()


def test_allowed_search():
    shop, ticket, state = _shop(), _ticket(), _state()
    assert check_action({"action": "search", "query": "roses"}, ticket, shop, TODAY, state) is None
    assert state.steps == 1


def test_extra_key_bad_action():
    shop, ticket, state = _shop(), _ticket(), _state()
    action = {"action": "search", "query": "roses", "extra": 1}
    assert check_action(action, ticket, shop, TODAY, state) == "bad_action"


def test_non_dict_bad_action():
    shop, ticket, state = _shop(), _ticket(), _state()
    assert check_action("search roses", ticket, shop, TODAY, state) == "bad_action"


def test_thirteenth_call_too_many_steps():
    shop, ticket, state = _shop(), _ticket(), _state()
    action = {"action": "search", "query": "roses"}
    for _ in range(12):
        assert check_action(action, ticket, shop, TODAY, state) is None
    assert check_action(action, ticket, shop, TODAY, state) == "too_many_steps"


def test_add_to_cart_qty_two_bad_qty():
    shop, ticket, state = _shop(), _ticket(), _state()
    action = {"action": "add_to_cart", "product_id": "P001", "qty": 2}
    assert check_action(action, ticket, shop, TODAY, state) == "bad_qty"


def test_add_to_cart_wrong_category():
    shop, ticket, state = _shop(), _ticket(), _state()
    action = {"action": "add_to_cart", "product_id": "P002", "qty": 1}
    assert check_action(action, ticket, shop, TODAY, state) == "wrong_category"


def test_add_to_cart_over_budget():
    shop, ticket, state = _shop(), _ticket(), _state()
    action = {"action": "add_to_cart", "product_id": "P003", "qty": 1}
    assert check_action(action, ticket, shop, TODAY, state) == "over_budget"


def test_checkout_wrong_card():
    shop, ticket, state = _shop(), _ticket(), _state()
    assert check_action(_checkout(card="<VCARD_2>"), ticket, shop, TODAY, state) == "wrong_card"


def test_checkout_empty_cart():
    shop, ticket, state = _shop(), _ticket(), _state()
    assert check_action(_checkout(), ticket, shop, TODAY, state) == "empty_cart"


def test_checkout_after_deliver_by_late_delivery():
    shop, ticket, state = _shop(), _ticket(), _state()
    _stock_cart(shop)
    later = date(2026, 11, 5)
    assert check_action(_checkout(deliver_on=later), ticket, shop, TODAY, state) == "late_delivery"


def test_checkout_before_today_late_delivery():
    shop, ticket, state = _shop(), _ticket(), _state()
    _stock_cart(shop)
    yesterday = date(2026, 10, 31)
    assert (
        check_action(_checkout(deliver_on=yesterday), ticket, shop, TODAY, state)
        == "late_delivery"
    )


def test_valid_checkout_then_already_checked_out():
    shop, ticket, state = _shop(), _ticket(), _state()
    _stock_cart(shop)
    action = _checkout(deliver_on=TODAY)
    assert check_action(action, ticket, shop, TODAY, state) is None
    assert state.checked_out is True
    assert check_action(action, ticket, shop, TODAY, state) == "already_checked_out"


def test_valid_checkout_pins_the_approved_total():
    shop, ticket, state = _shop(), _ticket(), _state()
    _stock_cart(shop)
    assert state.approved_total_fen is None
    assert check_action(_checkout(deliver_on=TODAY), ticket, shop, TODAY, state) is None
    assert state.approved_total_fen == 8800 + 500


def test_verify_receipt_valid_order():
    shop, ticket = _shop(), _ticket()
    _stock_cart(shop)
    result = shop.step(_checkout(deliver_on=TODAY))
    assert result.error is None
    order = shop.orders()[-1]
    assert verify_receipt(order, ticket, shop, TODAY) is None


def test_verify_receipt_order_not_in_shop():
    shop, ticket = _shop(), _ticket()
    order = Order(
        order_id="O999999",
        product_id="P001",
        qty=1,
        total_fen=9300,
        deliver_on=TODAY,
        card_token=CARD,
    )
    assert verify_receipt(order, ticket, shop, TODAY) == "unknown_order"


def test_verify_receipt_order_with_different_card():
    shop, ticket = _shop(), _ticket()
    _stock_cart(shop)
    result = shop.step(_checkout(card="<VCARD_2>", deliver_on=TODAY))
    assert result.error is None
    order = shop.orders()[-1]
    assert verify_receipt(order, ticket, shop, TODAY) == "wrong_card"
