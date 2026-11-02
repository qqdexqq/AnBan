"""Tests for the guarded v3 dataset builder and its code-built gold labels."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pytest

from agent.prompts import R3_SYSTEM_PROMPT
from security.dag import parse_planner_output, validate_plan
from training.build_v3_dataset import build
from training.v3_samples import (
    BILL_CATEGORIES,
    CARD_CONTROLS,
    HELP_TOPICS,
    MERCHANT_BLOCK_CATEGORIES,
    REQUIRED_POSITIVE_TOOLS,
    generate_samples,
)

SEED = 7
TODAY = date(2026, 10, 17)
REQUIRED_TOOLS = set(REQUIRED_POSITIVE_TOOLS)
LABEL_RE = re.compile(r"<([A-Z]+_[A-Za-z0-9]+)")
ASSISTANT_LABEL_RE = re.compile(r"<[A-Z]+_[A-Za-z0-9]+>")


def _known_tokens(prompt: str) -> dict[str, str]:
    known: dict[str, str] = {}
    for match in LABEL_RE.finditer(prompt):
        full = match.group(1)
        kind = full.split("_", 1)[0]
        known[f"<{full}>"] = kind
    return known


def _meta(record: dict) -> dict:
    value = record.get("meta")
    return json.loads(value) if isinstance(value, str) else dict(value or {})


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


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


def _plan_steps(record: dict) -> list:
    parsed = parse_planner_output(_assistant(record))
    return list(parsed.steps) if parsed.kind == "plan" else []


@pytest.fixture(scope="module")
def samples() -> list[dict]:
    return generate_samples(SEED, TODAY, per_intent=1)


def test_roles_and_every_gold_validates(samples: list[dict]) -> None:
    assert samples
    for record in samples:
        assert [message["role"] for message in record["messages"]] == [
            "system",
            "user",
            "assistant",
        ]
        assert record["messages"][0]["content"] == R3_SYSTEM_PROMPT
        assert _user(record)
        assert _assistant(record)

        meta = _meta(record)
        assert "known_tokens" not in meta
        assert meta["scenario"]
        assert meta["persona"]
        assert meta["bankseed"] is not None
        assert meta["slotseed"] is not None

        parsed = parse_planner_output(_assistant(record))
        validated = validate_plan(parsed, _known_tokens(_user(record)))
        assert validated.kind == parsed.kind
        if parsed.kind == "plan":
            assert validated.steps

        known = _known_tokens(_user(record))
        for label in ASSISTANT_LABEL_RE.findall(_assistant(record)):
            assert label in known, (meta["scenario"], label)


def test_negative_ratio_is_about_quarter(samples: list[dict]) -> None:
    negatives = sum(1 for record in samples if _meta(record)["negative"])
    ratio = negatives / len(samples)
    assert 0.25 <= ratio <= 0.30, ratio


def test_required_tools_and_v3_coverage(samples: list[dict]) -> None:
    tools: set[str] = set()
    bill_categories: set[str] = set()
    installment_periods: set[int] = set()
    deposit_terms: set[int] = set()
    forecast_days: set[int] = set()
    help_topics: set[str] = set()
    controls: set[tuple[str, bool]] = set()
    block_categories: set[str] = set()
    unblock_categories: set[str] = set()
    langs: set[str] = set()

    for record in samples:
        langs.add(_meta(record)["lang"])
        for step in _plan_steps(record):
            tools.add(step.tool)
            if step.tool == "pay_bill":
                bill_categories.add(step.args["category"])
            elif step.tool == "installment_quote":
                installment_periods.add(int(step.args["periods"]))
            elif step.tool == "deposit_quote":
                deposit_terms.add(int(step.args["term_months"]))
            elif step.tool == "forecast_cashflow":
                forecast_days.add(int(step.args["days"]))
            elif step.tool == "help_answer":
                help_topics.add(step.args["topic"])
            elif step.tool == "set_card_control":
                controls.add((step.args["control"], bool(step.args["enabled"])))
            elif step.tool == "block_merchant":
                block_categories.add(step.args["category"])
            elif step.tool == "unblock_merchant":
                unblock_categories.add(step.args["category"])

    assert REQUIRED_TOOLS <= tools
    assert langs == {"zh", "en", "mixed"}
    assert bill_categories == set(BILL_CATEGORIES)
    assert installment_periods == {3, 6, 12, 24}
    assert deposit_terms == {3, 6, 12, 24, 36, 60}
    assert forecast_days == {7, 14, 30, 45, 60}
    assert help_topics == set(HELP_TOPICS)
    assert controls == {
        (control, enabled)
        for control in CARD_CONTROLS
        for enabled in (True, False)
    }
    assert block_categories == set(MERCHANT_BLOCK_CATEGORIES)
    assert unblock_categories == set(MERCHANT_BLOCK_CATEGORIES)


def test_clarify_and_refuse_negatives_never_write(samples: list[dict]) -> None:
    kinds = {_meta(record)["scenario"]: _meta(record)["kind"] for record in samples}
    assert kinds["V3_clarify_missing_amount"] == "clarify"
    assert kinds["V3_clarify_missing_payee"] == "clarify"
    assert kinds["V3_clarify_missing_term"] == "clarify"
    assert kinds["V3_clarify_unsupported"] == "clarify"
    assert kinds["V3_negated_transfer"] == "clarify"
    assert kinds["V3_verify_payee_unknown"] == "clarify"
    assert kinds["V3_raw_unverified_target"] == "refuse"

    for record in samples:
        meta = _meta(record)
        if meta["scenario"] == "V3_negated_transfer":
            assert not _plan_steps(record)
        if meta["scenario"] == "V3_raw_unverified_target":
            parsed = parse_planner_output(_assistant(record))
            assert parsed.kind == "refuse"


# ---------------------------------------------------------------------------
# Builder fixture
# ---------------------------------------------------------------------------


def _legacy_record(
    user: str,
    *,
    scenario: str,
    persona: str,
    bank_seed: int,
    sample_seed: int,
    variation: str = "explicit",
    held_out: bool = False,
    kind: str = "clarify",
) -> dict:
    if kind == "clarify":
        assistant = {
            "kind": "clarify",
            "question": "legacy question",
            "options": [],
        }
    else:
        assistant = {"kind": "refuse", "reason": "out_of_scope"}
    meta = {
        "scenario_id": scenario,
        "persona_id": persona,
        "bank_seed": bank_seed,
        "sample_seed": sample_seed,
        "variation": variation,
        "held_out": held_out,
        "kind": assistant["kind"],
    }
    return {
        "messages": [
            {"role": "system", "content": "OLD LEGACY SYSTEM PROMPT"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(assistant)},
        ],
        "meta": json.dumps(meta, ensure_ascii=False),
    }


def _write_legacy(root: Path) -> Path:
    legacy = root / "legacy"
    legacy.mkdir()
    train = [
        _legacy_record(
            "legacy train one",
            scenario="V3_legacy_train",
            persona="young_pro",
            bank_seed=1,
            sample_seed=11,
        ),
        # Teacher sibling: same family key, different wording.
        _legacy_record(
            "legacy train teacher sibling",
            scenario="V3_legacy_train",
            persona="young_pro",
            bank_seed=1,
            sample_seed=11,
            variation="teacher",
        ),
        _legacy_record(
            "legacy b6 train",
            scenario="B6_bill_pay_unsupported",
            persona="young_pro",
            bank_seed=1,
            sample_seed=12,
            kind="refuse",
        ),
    ]
    dev = [
        _legacy_record(
            "legacy dev one",
            scenario="V3_legacy_dev",
            persona="student",
            bank_seed=2,
            sample_seed=21,
        ),
        _legacy_record(
            "legacy b6 dev",
            scenario="B6_bill_pay_unsupported",
            persona="student",
            bank_seed=2,
            sample_seed=22,
            kind="refuse",
        ),
    ]
    test = [
        # The train sibling below shares this family, so test isolation must
        # move both rows to test.
        _legacy_record(
            "legacy held out test",
            scenario="V3_legacy_test",
            persona="retiree",
            bank_seed=3,
            sample_seed=31,
            held_out=True,
        ),
        _legacy_record(
            "legacy held out train sibling",
            scenario="V3_legacy_test",
            persona="retiree",
            bank_seed=3,
            sample_seed=31,
        ),
    ]
    # The sibling belongs to train physically, but family isolation must force
    # it into test with its held-out sibling.
    train.append(test.pop())
    test.append(
        _legacy_record(
            "legacy pure test",
            scenario="V3_legacy_pure_test",
            persona="young_parent",
            bank_seed=4,
            sample_seed=41,
        )
    )

    for name, rows in (("train", train), ("dev", dev), ("test", test)):
        with (legacy / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return legacy


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    root = tmp_path_factory.mktemp("v3_build")
    legacy = _write_legacy(root)
    out = root / "out"
    stats = build(SEED, TODAY, out, per_intent=1, legacy=legacy)
    return out, stats


def test_builder_artifacts_info_and_stats(built: tuple[Path, dict]) -> None:
    out, stats = built
    for name in (
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "dataset_info.json",
        "stats.json",
    ):
        assert (out / name).is_file(), name

    info = json.loads((out / "dataset_info.json").read_text(encoding="utf-8"))
    for key in ("anban_train", "anban_dev"):
        assert info[key]["formatting"] == "sharegpt"
        assert info[key]["columns"] == {"messages": "messages"}

    assert stats["draft"] is True
    assert stats["release_ready"] is False
    assert "six-scenario backend closure" in stats["draft_note"]
    assert stats["prompt_sha256"] == hashlib.sha256(
        R3_SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest()
    assert stats["source_hashes"]["legacy_train"]
    assert stats["drop_counts"]["legacy_b6_bill_pay_unsupported"] == {
        "dev": 1,
        "train": 1,
    }
    assert stats["legacy"]["b6_bill_pay_unsupported_dropped"] == 2
    assert stats["negative_ratio"] > 0
    assert 0.25 <= stats["v3_negative_ratio"] <= 0.30
    assert stats["dropcounts"] == stats["drop_counts"]
    assert stats["sourcehashes"] == stats["source_hashes"]
    assert stats["promptSHA256"] == stats["prompt_sha256"]

    for split in ("train", "dev", "test"):
        assert stats["splits"][split]["count"] == len(_rows(out / f"{split}.jsonl"))


def test_builder_replaces_system_and_drops_b6(built: tuple[Path, dict]) -> None:
    out, _stats = built
    for name in ("train", "dev", "test"):
        for record in _rows(out / f"{name}.jsonl"):
            assert record["messages"][0]["content"] == R3_SYSTEM_PROMPT
            meta = _meta(record)
            assert meta.get("scenario_id") != "B6_bill_pay_unsupported"


def test_builder_has_required_tool_coverage_in_test(built: tuple[Path, dict]) -> None:
    out, _stats = built
    test_tools: set[str] = set()
    all_tools: set[str] = set()
    for name in ("train", "dev", "test"):
        for record in _rows(out / f"{name}.jsonl"):
            tools = {step.tool for step in _plan_steps(record)}
            all_tools |= tools
            if name == "test":
                test_tools |= tools
    assert REQUIRED_TOOLS <= all_tools
    assert REQUIRED_TOOLS <= test_tools


def test_no_source_family_crosses_splits(built: tuple[Path, dict]) -> None:
    out, _stats = built
    families: dict[str, set[str]] = {}
    for name in ("train", "dev", "test"):
        for record in _rows(out / f"{name}.jsonl"):
            family = _meta(record)["family"]
            families.setdefault(family, set()).add(name)
    assert families
    for family, splits in families.items():
        assert len(splits) == 1, (family, splits)


def test_test_isolation_for_legacy_families(built: tuple[Path, dict]) -> None:
    out, _stats = built
    test_users = {
        _user(record) for record in _rows(out / "test.jsonl")
    }
    assert "legacy held out test" in test_users
    assert "legacy held out train sibling" in test_users
    assert "legacy pure test" in test_users


def test_deterministic_build(
    built: tuple[Path, dict], tmp_path: Path
) -> None:
    out, _stats = built
    legacy = _write_legacy(tmp_path)
    rebuild = tmp_path / "rebuild"
    build(SEED, TODAY, rebuild, per_intent=1, legacy=legacy)
    for name in (
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "dataset_info.json",
        "stats.json",
    ):
        assert (out / name).read_bytes() == (rebuild / name).read_bytes(), name


def test_refuses_existing_output_and_preserves_sentinel(
    tmp_path: Path,
) -> None:
    legacy = _write_legacy(tmp_path)
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "sentinel.txt"
    sentinel.write_text("do not touch", encoding="utf-8")

    with pytest.raises(SystemExit):
        build(SEED, TODAY, out, per_intent=1, legacy=legacy)

    assert sentinel.read_text(encoding="utf-8") == "do not touch"
    assert sorted(path.name for path in out.iterdir()) == ["sentinel.txt"]


def test_refuses_old_training_data_path(tmp_path: Path) -> None:
    legacy = _write_legacy(tmp_path)
    repo = Path(__file__).resolve().parents[1]
    forbidden = repo / "training" / "data" / "should_not_exist_v3"
    with pytest.raises(SystemExit):
        build(SEED, TODAY, forbidden, per_intent=1, legacy=legacy)
    assert not forbidden.exists()


def test_rerun_requires_new_output_path(built: tuple[Path, dict]) -> None:
    out, _stats = built
    with pytest.raises(SystemExit):
        build(SEED, TODAY, out, per_intent=1, legacy=out.parent / "legacy")
