"""Tests for training/build_errand_dataset.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from errand.prompts import ERRAND_SYSTEM_PROMPT
from training import build_errand_dataset as builder

N_TRAIN = 60
N_DEV = 10
N_TEST = 10


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("errand") / "data"
    stats = builder.build(out, N_TRAIN, N_DEV, N_TEST)
    return out, stats


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_build_writes_expected_files(built):
    out, _stats = built
    for name in (
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "dataset_info.json",
        "stats.json",
    ):
        assert (out / name).is_file(), name


def test_refuses_existing_dir(tmp_path):
    existing = tmp_path / "already"
    existing.mkdir()
    with pytest.raises(SystemExit):
        builder.build(existing, 2, 1, 1)
    assert list(existing.iterdir()) == []


def test_every_line_has_messages_and_string_meta(built):
    out, _stats = built
    for split in ("train", "dev", "test"):
        rows = _rows(out / f"{split}.jsonl")
        assert rows
        for row in rows:
            assert isinstance(row["messages"], list) and row["messages"]
            assert isinstance(row["meta"], str)
            meta = json.loads(row["meta"])
            assert meta["split"] == split
            assert set(meta) == {"seed", "split", "expected", "tags", "steps"}


def test_artifact_hashes_match_files(built):
    out, stats = built
    assert set(stats["artifact_sha256"]) == {
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "dataset_info.json",
    }
    for name, digest in stats["artifact_sha256"].items():
        expected = hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert digest == expected


def test_counts_and_prompt_hash(built):
    out, stats = built
    assert stats["counts"] == {
        "train": N_TRAIN,
        "dev": N_DEV,
        "test": N_TEST,
    }
    for split, count in stats["counts"].items():
        assert len(_rows(out / f"{split}.jsonl")) == count
    assert stats["prompt_sha256"] == hashlib.sha256(
        ERRAND_SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest()
    assert isinstance(stats["max_chars"], int)
    assert isinstance(stats["mean_assistant_turns"], float)


def test_assistant_turns_are_json_and_transcript_ends_with_assistant(built):
    out, _stats = built
    for split in ("train", "dev", "test"):
        for row in _rows(out / f"{split}.jsonl"):
            messages = row["messages"]
            assert messages[0]["role"] == "system"
            assert messages[-1]["role"] == "assistant"
            roles = [message["role"] for message in messages[1:]]
            assert roles == [
                "user" if index % 2 == 0 else "assistant"
                for index in range(len(roles))
            ]
            for message in messages:
                if message["role"] != "assistant":
                    continue
                parsed = json.loads(message["content"])
                assert isinstance(parsed, dict)


def test_train_and_test_seeds_are_disjoint(built):
    out, _stats = built
    train_seeds = {
        json.loads(row["meta"])["seed"] for row in _rows(out / "train.jsonl")
    }
    test_seeds = {
        json.loads(row["meta"])["seed"] for row in _rows(out / "test.jsonl")
    }
    assert train_seeds.isdisjoint(test_seeds)


def test_meta_expected_matches_transcript(built):
    out, _stats = built
    for split in ("train", "dev", "test"):
        for row in _rows(out / f"{split}.jsonl"):
            meta = json.loads(row["meta"])
            last = json.loads(row["messages"][-1]["content"])
            if meta["expected"].startswith("buy:"):
                assert last["action"] == "finish"
            else:
                assert last["action"] == "abort"
                assert last["reason"] == meta["expected"].split(":", 1)[1]
