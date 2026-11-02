"""Preflight check for merchant baskets funded by a new pot in the same plan.

The planner may reserve a birthday/event budget with ``move_to_pot`` and then
fund several ``merchant_order`` steps from that step's ``$sN.pot``.  This module
quotes the whole basket at the bank's own catalogue prices before policy,
approval or execution.  It never changes the plan or moves money; it only
returns the question the user must answer.
"""

from __future__ import annotations

import re
from typing import NoReturn

from agent.tools import TOKEN_RE
from bank.merchants import price_of
from bank.money import fmt_fen
from security.dag import PlanStep, ValidatedPlan
from security.tokens import Session, TokenError, TokenVault

_DIRECT_POT_REF_RE = re.compile(r"^\$(s[0-9]+)\.pot$")
_FULL_TOKEN_RE = re.compile(rf"^{TOKEN_RE}$")


def _fail() -> NoReturn:
    raise ValueError("basket_quote_unavailable")


def _token_value(value: object, kind: str, vault: TokenVault, session: Session) -> object:
    if not isinstance(value, str) or _FULL_TOKEN_RE.fullmatch(value) is None:
        _fail()
    try:
        entry = vault.resolve(value, session)
    except TokenError:
        _fail()
    if entry.kind != kind:
        _fail()
    return entry.real_ref


def _amount_fen(value: object, vault: TokenVault, session: Session) -> int:
    real = _token_value(value, "AMT", vault, session)
    if isinstance(real, bool) or not isinstance(real, int) or real < 0:
        _fail()
    return real


def _merchant_price(step: PlanStep, vault: TokenVault, session: Session) -> int:
    args = step.args
    merchant = _token_value(args.get("merchant"), "MERCHANT", vault, session)
    item = args.get("item")
    if not isinstance(merchant, str) or not isinstance(item, str):
        _fail()
    try:
        price = price_of(merchant, item)
    except (KeyError, TypeError):
        _fail()
    if isinstance(price, bool) or not isinstance(price, int) or price < 0:
        _fail()
    return price


def _question(pot: str, total: int, budget: int, shortfall: int, lang: str) -> str:
    if lang == "en":
        return (
            f'Your “{pot}” basket totals {fmt_fen(total)} at bank prices, but '
            f"its reserved budget is {fmt_fen(budget)} - short of "
            f"{fmt_fen(shortfall)}. Would you like to increase the budget or "
            "change the items?"
        )
    return (
        f"「{pot}」购物篮按银行报价合计 {fmt_fen(total)}，预留预算为 "
        f"{fmt_fen(budget)}，还差 {fmt_fen(shortfall)}。"
        "您想增加预算，还是更换商品？"
    )


def budget_question(
    plan: ValidatedPlan,
    vault: TokenVault,
    session: Session,
    lang: str = "en",
) -> str | None:
    """Return a bilingual over-budget question, or ``None`` when it fits.

    Only orders funded by a direct ``$sN.pot`` reference to a ``move_to_pot``
    step in this same plan form a basket, and only that step's newly reserved
    amount is the budget.  An existing pot balance is never added to it.  If a
    related basket's price, token or source cannot be quoted safely, fail closed
    with ``ValueError("basket_quote_unavailable")`` instead of letting it run.
    """

    if plan.kind != "plan":
        return None

    steps = {step.id: step for step in plan.steps}
    baskets: dict[str, dict[str, object]] = {}

    for step in plan.steps:
        if step.tool != "merchant_order":
            continue
        funding = step.args.get("funding_pot")
        if funding is None:
            continue
        if not isinstance(funding, str):
            _fail()
        if _FULL_TOKEN_RE.fullmatch(funding) is not None:
            # A POT token names a pre-existing pot, not a budget reserved here,
            # but the token itself must still resolve for this session.
            _token_value(funding, "POT", vault, session)
            continue
        match = _DIRECT_POT_REF_RE.fullmatch(funding)
        if match is None:
            _fail()
        source = steps.get(match.group(1))
        if source is None or source.tool != "move_to_pot":
            _fail()

        basket = baskets.get(source.id)
        if basket is None:
            pot_name = source.args.get("pot_name")
            if not isinstance(pot_name, str) or not pot_name:
                _fail()
            basket = {
                "pot": pot_name,
                "budget": _amount_fen(source.args.get("amount"), vault, session),
                "total": 0,
            }
            baskets[source.id] = basket
        basket["total"] = int(basket["total"]) + _merchant_price(step, vault, session)

    lines: list[str] = []
    for basket in baskets.values():
        total = int(basket["total"])
        budget = int(basket["budget"])
        if total > budget:
            lines.append(
                _question(str(basket["pot"]), total, budget, total - budget, lang)
            )
    return "\n".join(lines) if lines else None
