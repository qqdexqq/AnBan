"""Tests for the redeem_product v3 samples and the bank-derived PROD flags."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest
import yaml

from bank import db
from bank.api import Bank
from bank.redemption import REDEEMABLE_PRODUCTS
from bank.seed import generate
from security.dag import parse_planner_output, validate_plan
from security.tokens import Session, TokenVault, build_context, render
from training.v3_samples import REQUIRED_POSITIVE_TOOLS, generate_samples

SEED = 7
TODAY = date(2026, 10, 17)
SECRET = b"anban-redemption-training-secret"
SEGMENTS_PATH = (
    Path(__file__).resolve().parents[1] / "data_spec" / "customer_segments.yaml"
)

_LABEL_RE = re.compile(r"<([A-Z]+_[A-Za-z0-9]+)")
_PROD_LINE_RE = re.compile(r"<(PROD_[A-Za-z0-9]+) \| ([^>]*)>")


def _all_persona_ids() -> list[str]:
    with SEGMENTS_PATH.open(encoding="utf-8") as fh:
        segments = yaml.safe_load(fh)["segments"]
    return [str(segment["id"]) for segment in segments]


def _make_bank(persona_id: str) -> tuple[Bank, object]:
    state = generate(persona_id, SEED, TODAY)
    conn = db.connect(":memory:")
    db.create_schema(conn)
    db.load_state(conn, state)
    return Bank(conn, state.customer.id), state


def _meta(record: dict) -> dict:
    value = record.get("meta")
    return json.loads(value) if isinstance(value, str) else dict(value or {})


def _user(record: dict) -> str:
    return next(
        message["content"]
        for message in record["messages"]
        if message["role"] == "user"
    )


def _assistant(record: dict) -> str:
    return next(
        message["content"]
        for message in record["messages"]
        if message["role"] == "assistant"
    )


def _known_tokens(user: str) -> dict[str, str]:
    return {
        f"<{label}>": label.split("_", 1)[0]
        for label in _LABEL_RE.findall(user)
    }


def _prod_flags(user: str) -> dict[str, dict[str, bool]]:
    flags: dict[str, dict[str, bool]] = {}
    for label, body in _PROD_LINE_RE.findall(user):
        flags[f"<{label}>"] = {
            "held": "held=true" in body,
            "redeemable": "redeemable=true" in body,
        }
    return flags


@pytest.fixture(scope="module")
def samples() -> list[dict]:
    return generate_samples(SEED, TODAY, per_intent=1)


def test_redeem_product_is_a_required_positive_tool() -> None:
    assert "redeem_product" in REQUIRED_POSITIVE_TOOLS


def test_prod_held_and_redeemable_flags_are_bank_derived() -> None:
    for persona_id in _all_persona_ids():
        bank, state = _make_bank(persona_id)
        try:
            vault = TokenVault(SECRET)
            session = Session(customer_id=state.customer.id)
            build_context(bank, vault, session, TODAY)

            held = {
                holding.product_id: holding
                for holding in bank.get_portfolio()
                if int(holding.value_fen) > 0
            }
            products = {product.id: product for product in bank.list_products()}
            entries = {
                entry.real_ref: entry
                for entry in vault.entries(session)
                if entry.kind == "PROD"
            }

            assert set(entries) == set(products)
            for product_id, entry in entries.items():
                expected_held = product_id in held
                expected_redeemable = (
                    expected_held and product_id in REDEEMABLE_PRODUCTS
                )

                assert entry.tags["held"] is expected_held
                assert entry.tags["redeemable"] is expected_redeemable

                rendered = render(entry)
                assert (
                    f"held={'true' if expected_held else 'false'}" in rendered
                )
                assert (
                    f"redeemable={'true' if expected_redeemable else 'false'}"
                    in rendered
                )

                # Tags carry only semantic booleans: no product name or value.
                product = products[product_id]
                assert product.name not in rendered
                if product_id in held:
                    assert str(held[product_id].value_fen) not in rendered
                    assert str(held[product_id].principal_fen) not in rendered
        finally:
            bank.conn.close()


def test_redeem_gold_uses_only_the_sole_eligible_holding(
    samples: list[dict],
) -> None:
    records = [
        record for record in samples if _meta(record)["intent"] == "redeem_product"
    ]
    assert records
    assert {"zh", "en", "mixed"} <= {_meta(record)["lang"] for record in records}

    plan_count = 0
    for record in records:
        user = _user(record)
        known = _known_tokens(user)
        flags = _prod_flags(user)
        eligible = {
            label
            for label, product_flags in flags.items()
            if product_flags["held"] and product_flags["redeemable"]
        }

        parsed = parse_planner_output(_assistant(record))
        validated = validate_plan(parsed, known)
        assert validated.kind == parsed.kind

        if parsed.kind != "plan":
            assert parsed.kind == "clarify"
            assert len(eligible) != 1
            continue

        plan_count += 1
        assert len(eligible) == 1
        assert [step.tool for step in validated.steps] == [
            "get_portfolio",
            "redeem_product",
        ]

        redeem_step = validated.steps[1]
        assert redeem_step.after == ["s1"]
        product_label = redeem_step.args["product"]
        assert product_label in eligible
        assert known[product_label] == "PROD"

    assert plan_count
