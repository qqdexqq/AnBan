"""Model/prompt changes cannot silently reuse old predictions."""

import json

import pytest

from training.eval_planner import evaluate, score


def _sample():
    return {"id": "one", "system": "prompt", "user": "User: no tools",
            "gold": '{"kind":"refuse","reason":"out_of_scope"}',
            "scenario_id": "X_test", "persona_id": "test", "lang": "en"}


def test_invalid_attack_response_is_a_failed_safety_proposal():
    row = _sample()
    metrics = score(row["gold"], "invalid JSON", row["user"], row["scenario_id"])
    assert not metrics["valid"] and not metrics["safety"]


def test_prompt_change_rejects_resume_and_preserves_predictions(tmp_path):
    row = _sample()
    assert len(evaluate(None, [row], tmp_path, 1)) == 1
    predictions = (tmp_path / "predictions.jsonl").read_bytes()
    assert len(evaluate(None, [row], tmp_path, 1)) == 1
    with pytest.raises(ValueError, match="cache differs"):
        evaluate(None, [{**row, "system": "changed prompt"}], tmp_path, 1)
    assert (tmp_path / "predictions.jsonl").read_bytes() == predictions


def test_legacy_cache_is_preserved_without_implicit_adoption(tmp_path):
    path = tmp_path / "predictions.jsonl"
    original = json.dumps({"id": "one", "prediction": "old"})
    path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="no provenance"):
        evaluate(None, [_sample()], tmp_path, 1)
    assert path.read_text(encoding="utf-8") == original
