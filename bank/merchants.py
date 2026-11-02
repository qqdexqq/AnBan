"""Fixed merchant catalogue: item codes and prices live in code, not the plan.

The planner may only choose an item code from a closed literal set.  The bank
looks the price up here, so a model or caller can never supply or edit it.
"""

from __future__ import annotations

CatalogueItem = tuple[str, str, int]

CATALOGUE: dict[str, dict[str, CatalogueItem]] = {
    "花店": {
        "bouquet_s": ("小花束", "small bouquet", 19900),
        "bouquet_m": ("花束", "bouquet", 29900),
        "roses_11": ("11支玫瑰", "11 roses", 39900),
    },
    "蛋糕店": {
        "cake_6in": ("6寸蛋糕", "6-inch cake", 16800),
        "cake_8in": ("8寸蛋糕", "8-inch cake", 26800),
        "cake_10in": ("10寸蛋糕", "10-inch cake", 36800),
    },
}


def price_of(merchant: str, item_code: str) -> int:
    """Return the item price in fen, or raise ``KeyError`` if not sold."""

    return CATALOGUE[merchant][item_code][2]
