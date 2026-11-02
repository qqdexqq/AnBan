"""Spec consistency tests for the YAML files under data_spec/."""

from pathlib import Path

import yaml

SPEC_DIR = Path(__file__).resolve().parents[1] / "data_spec"
VALID_GOLD_KINDS = {"plan", "clarify", "refuse", "scam"}


def _load(name: str) -> dict:
    with (SPEC_DIR / name).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_segments_have_required_fields() -> None:
    segments = _load("customer_segments.yaml")["segments"]
    assert segments
    for segment in segments:
        for field in ("id", "language", "scenario_weights"):
            assert field in segment, f"segment {segment.get('id')!r} missing {field}"


def test_scenarios_have_required_fields() -> None:
    scenarios = _load("scenarios.yaml")["scenarios"]
    assert scenarios
    for scenario in scenarios:
        for field in ("id", "group", "gold"):
            assert field in scenario, f"scenario {scenario.get('id')!r} missing {field}"
        kind = scenario["gold"].get("kind")
        assert kind in VALID_GOLD_KINDS, f"scenario {scenario['id']!r} has bad gold.kind {kind!r}"


def test_scenario_ids_are_unique() -> None:
    scenarios = _load("scenarios.yaml")["scenarios"]
    ids = [scenario["id"] for scenario in scenarios]
    assert len(ids) == len(set(ids)), "duplicate scenario ids"


def test_only_declared_personas_are_held_out() -> None:
    segments = _load("customer_segments.yaml")["segments"]
    split_rules = _load("scenarios.yaml")["split_rules"]
    declared = set(split_rules["test_only_personas"])
    held_out = {segment["id"] for segment in segments if segment.get("held_out") is True}
    assert held_out == declared
