"""Tests for training/eval_errand.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from errand.scenarios import generate_scenario
from training import eval_errand as ev


class _AbortModel:
    def next_action(self, messages: list[dict]) -> str:
        return '{"action": "abort", "reason": "no_suitable_product"}'


class _BoomModel:
    def next_action(self, messages: list[dict]) -> str:
        raise RuntimeError("model exploded")


def _write_dataset(path: Path, seeds: list[int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for seed in seeds:
            meta = json.dumps(
                {
                    "seed": seed,
                    "split": "test",
                    "expected": generate_scenario(seed).expected,
                    "tags": list(generate_scenario(seed).tags),
                },
                ensure_ascii=False,
            )
            handle.write(json.dumps({"messages": [], "meta": meta}) + "\n")


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_oracle_scores_one_and_has_no_violations(tmp_path):
    seeds = list(range(2_000_000, 2_000_050))
    data = tmp_path / "test.jsonl"
    _write_dataset(data, seeds)
    out = tmp_path / "out"

    assert ev.main(["--data", str(data), "--oracle", "--workers", "4", "--out", str(out)]) == 0

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["n"] == len(seeds)
    assert summary["success_rate"] == 1.0
    for key in (
        "overspend",
        "wrong_card",
        "bad_qty",
        "injection_followed",
        "phishing_bought",
        "failed",
    ):
        assert summary[key] == 0, key
    for rate in summary["by_tag"].values():
        assert rate == 1.0
    for rate in summary["by_expected"].values():
        assert rate == 1.0

    cases = _rows(out / "cases.jsonl")
    assert len(cases) == len(seeds)
    assert all(case["success"] for case in cases)
    assert all("transcript" not in case for case in cases)


def test_always_abort_model_fails_buy_cases():
    buy_seeds = [
        seed
        for seed in range(2_000_000, 2_000_050)
        if generate_scenario(seed).expected.startswith("buy:")
    ][:10]
    assert buy_seeds

    cases = ev.evaluate(buy_seeds, lambda scenario: _AbortModel(), workers=4)

    assert len(cases) == len(buy_seeds)
    assert all(case["expected"].startswith("buy:") for case in cases)
    assert all(not case["success"] for case in cases)
    assert all(case["got"].startswith("abort:") for case in cases)
    assert all("transcript" in case for case in cases)
    summary = ev.summarize(cases)
    assert summary["success_rate"] < 1.0
    assert summary["success_rate"] == 0.0


def test_model_exception_becomes_failed_case_without_crash():
    seeds = [2_000_000, 2_000_001, 2_000_002]
    cases = ev.evaluate(seeds, lambda scenario: _BoomModel(), workers=3)

    assert len(cases) == len(seeds)
    for case in cases:
        assert case["status"] == "failed"
        assert case["got"] == "failed"
        assert case["success"] is False
        assert case["error"].startswith("RuntimeError:")
        assert "transcript" in case
    assert ev.summarize(cases)["failed"] == len(seeds)


def test_refuses_existing_output_dir(tmp_path):
    data = tmp_path / "test.jsonl"
    _write_dataset(data, [2_000_000])
    existing = tmp_path / "already"
    existing.mkdir()

    with pytest.raises(SystemExit):
        ev.main(["--data", str(data), "--oracle", "--out", str(existing)])
    assert list(existing.iterdir()) == []
