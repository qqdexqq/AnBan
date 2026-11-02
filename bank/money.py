"""Money helpers: everything is stored as integer fen (1/100 of a yuan)."""

from decimal import Decimal, ROUND_HALF_UP

_ONE_FEN = Decimal("1")
_HUNDRED = Decimal("100")


def yuan(x: int | str | Decimal) -> int:
    """Convert a yuan amount to integer fen.

    ``int`` inputs are treated as whole yuan (``yuan(12) == 1200``); strings and
    ``Decimal`` values may contain decimals (``yuan("12.5") == 1250``).
    """
    if isinstance(x, bool):
        raise TypeError("bool is not a money amount")
    if isinstance(x, int):
        return x * 100
    if isinstance(x, Decimal):
        amount = x
    elif isinstance(x, str):
        amount = Decimal(x.strip())
    else:
        # Floats are deliberately not part of the public signature, but
        # accepting them via str() keeps accidental literals predictable.
        amount = Decimal(str(x))
    return int((amount * _HUNDRED).quantize(_ONE_FEN, rounding=ROUND_HALF_UP))


def fmt_fen(fen: int) -> str:
    """Format integer fen as a signed yuan string, e.g. ``¥1,234.50``."""
    if not isinstance(fen, int) or isinstance(fen, bool):
        raise TypeError("fen must be an int")
    sign = "-" if fen < 0 else ""
    fen = abs(fen)
    return f"{sign}¥{fen // 100:,}.{fen % 100:02d}"


def split_fen(amount_fen: int, parts: int) -> int:
    """One share of ``amount_fen`` split ``parts`` ways, rounded half up.

    The single rounding rule for split bills: policy (tiering, mandates) and
    the executor must agree to the fen, or the mandate binding never matches.
    """
    if isinstance(amount_fen, bool) or isinstance(parts, bool):
        raise TypeError("bool is not a money amount")
    if parts < 1:
        raise ValueError("parts must be >= 1")
    value = Decimal(int(amount_fen)) / Decimal(int(parts))
    return int(value.quantize(_ONE_FEN, rounding=ROUND_HALF_UP))
