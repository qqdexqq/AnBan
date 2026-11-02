"""End-to-end checks for the generated training dataset."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest
import yaml

from bank.seed import generate
from security.dag import parse_planner_output
from training.build_dataset import build, check_leakage

SEED = 42
PER_SCENARIO = 6
TODAY = date(2026, 10, 17)
MIN_TEST = 2
SPLITS = ("train", "dev", "test")
SPEC_PATH = Path(__file__).resolve().parents[1] / "data_spec" / "scenarios.yaml"
LONG_DIGITS = re.compile(r"\d{11,}")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("training_data")
    build(SEED, PER_SCENARIO, out, TODAY, MIN_TEST)
    return out


def _rows(out: Path, split: str) -> list[dict]:
    with (out / f"{split}.jsonl").open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    for row in rows:
        if isinstance(row.get("meta"), str):
            row["meta"] = json.loads(row["meta"])
    return rows


def _user_text(row: dict) -> str:
    return row["messages"][1]["content"]


def test_artifacts_and_dataset_info(dataset: Path) -> None:
    for name in (
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "dataset_info.json",
        "stats.json",
    ):
        assert (dataset / name).is_file(), name

    info = json.loads((dataset / "dataset_info.json").read_text(encoding="utf-8"))
    for key in ("anban_train", "anban_dev"):
        assert info[key]["formatting"] == "sharegpt"


def test_message_roles_and_assistant_json(dataset: Path) -> None:
    for split in SPLITS:
        rows = _rows(dataset, split)
        assert rows, split
        for row in rows:
            messages = row["messages"]
            assert [message["role"] for message in messages] == [
                "system",
                "user",
                "assistant",
            ]
            assert all(message["content"] for message in messages)
            parse_planner_output(messages[2]["content"])


def test_every_scenario_appears_in_some_split(dataset: Path) -> None:
    with SPEC_PATH.open(encoding="utf-8") as fh:
        expected = {scenario["id"] for scenario in yaml.safe_load(fh)["scenarios"]}
    seen = {
        row["meta"]["scenario_id"]
        for split in SPLITS
        for row in _rows(dataset, split)
    }
    assert expected <= seen


def test_test_split_is_held_out(dataset: Path) -> None:
    for split in ("train", "dev"):
        for row in _rows(dataset, split):
            assert row["meta"]["persona_id"] != "expat", (split, row["meta"])

    for row in _rows(dataset, "test"):
        meta = row["meta"]
        # T2 needs a duplicate contact name, which only young_pro has.  When
        # the expat sample fails, build._extra_sample falls back to those
        # working personas with an arbitrary variation.
        if meta["scenario_id"] == "T2_duplicate_name" and meta["persona_id"] == "young_pro":
            continue
        assert (
            meta["persona_id"] == "expat" or meta["variation"] == "polite_long"
        ), meta


def test_no_user_text_overlap_between_train_and_test(dataset: Path) -> None:
    train_users = {_user_text(row) for row in _rows(dataset, "train")}
    test_users = {_user_text(row) for row in _rows(dataset, "test")}
    assert not train_users & test_users


def test_build_is_deterministic(
    dataset: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    rebuild = tmp_path_factory.mktemp("training_data_rebuild")
    build(SEED, PER_SCENARIO, rebuild, TODAY, MIN_TEST)
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        assert (dataset / name).read_bytes() == (rebuild / name).read_bytes(), name


def test_check_leakage_flags_contact_name_and_phone() -> None:
    state = generate("young_pro", SEED, TODAY)
    contact = state.contacts[0]
    text = f"Please transfer it to {contact.name} on {contact.phone} today."

    leaks = check_leakage(text, state)

    assert contact.name in leaks
    assert contact.phone in leaks


def test_no_long_digit_runs_in_prompts(dataset: Path) -> None:
    for split in SPLITS:
        for row in _rows(dataset, split):
            for message in row["messages"][:2]:
                assert not LONG_DIGITS.search(message["content"]), (
                    split,
                    message["role"],
                )


def test_paraphrases_of_one_source_share_train_dev_split() -> None:
    from training.build_dataset import _split_for

    meta = {
        "scenario_id": "T1_transfer_by_name", "persona_id": "young_pro",
        "seed": 42, "bank_seed": 44, "sample_seed": 42001234,
        "variation": "teacher",
    }
    splits = {
        _split_for({"meta": dict(meta), "user": f"distinct paraphrase {i}"}, set(), set())
        for i in range(50)
    }
    assert len(splits) == 1
