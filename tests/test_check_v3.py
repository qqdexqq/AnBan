"""Tests for the read-only v3 training-data readiness check."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent.prompts import R3_SYSTEM_PROMPT
from training import check_v3

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _assistant(tool: str = "get_balance", args: dict | None = None) -> str:
    plan = {"kind": "plan", "steps": [
        {"id": "s1", "tool": tool, "args": args or {}, "after": []}]}
    return json.dumps(plan, ensure_ascii=False, sort_keys=True)

def _row(
    user: str,
    family: str,
    *,
    assistant: str | None = None,
    system: str | None = None,
) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system or R3_SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant or _assistant()},
        ],
        "meta": json.dumps({"family": family}, ensure_ascii=False, sort_keys=True),
    }

def _write_split(data: Path, split: str, rows: list[dict]) -> None:
    (data / f"{split}.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

def _write_data(root: Path, rows: dict[str, list[dict]]) -> Path:
    data = root / "data"
    data.mkdir()
    for split, records in rows.items():
        _write_split(data, split, records)
    (data / "dataset_info.json").write_text(
        json.dumps({
            "anban_train": {"file_name": "train.jsonl", "formatting": "sharegpt"},
            "anban_dev": {"file_name": "dev.jsonl", "formatting": "sharegpt"},
        }) + "\n",
        encoding="utf-8",
    )
    stats = {
        "draft": True,
        "release_ready": False,
        "prompt_sha256": _sha_text(R3_SYSTEM_PROMPT),
        "artifact_sha256": {
            name: _sha(data / name)
            for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "dataset_info.json")
        },
    }
    (data / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return data

def _rewrite_split(data: Path, split: str, rows: list[dict]) -> None:
    _write_split(data, split, rows)
    stats = json.loads((data / "stats.json").read_text(encoding="utf-8"))
    stats["artifact_sha256"][f"{split}.jsonl"] = _sha(data / f"{split}.jsonl")
    (data / "stats.json").write_text(json.dumps(stats), encoding="utf-8")

class FakeTokenizer:
    def __init__(self, size: int | None = None) -> None:
        self.size = size
        self.calls = 0

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool,
        enable_thinking: bool,
        add_generation_prompt: bool,
        return_dict: bool,
    ) -> list[int]:
        self.calls += 1
        assert tokenize is True
        assert enable_thinking is False
        assert add_generation_prompt is False
        assert return_dict is False
        assert len(messages) == 3
        return list(range(self.size if self.size is not None else 32))

@pytest.fixture(autouse=True)
def only_get_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        check_v3, "REQUIRED_POSITIVE_TOOLS", frozenset({"get_balance"})
    )

@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Path, FakeTokenizer]:
    data = _write_data(
        tmp_path,
        {
            "train": [_row("what is my balance", "fam-train")],
            "dev": [_row("show my account balance", "fam-dev")],
            "test": [_row("how much money do I have", "fam-test")],
        },
    )
    return data, tmp_path / "report.json", FakeTokenizer()

def test_success(dataset: tuple[Path, Path, FakeTokenizer]) -> None:
    data, out, tokenizer = dataset
    stats_before = (data / "stats.json").read_bytes()
    report = check_v3.check(data, tokenizer, out)

    assert report["training_ready"] is True
    assert report["application_release_ready"] is False
    assert report["failures"] == []
    assert report["prompt_sha256"] == _sha_text(R3_SYSTEM_PROMPT)
    assert set(report["artifact_sha256"]) == {
        "train.jsonl", "dev.jsonl", "test.jsonl", "dataset_info.json", "stats.json"
    }
    assert report["artifact_sha256"]["train.jsonl"] == _sha(data / "train.jsonl")
    assert report["artifact_sha256"]["stats.json"] == _sha(data / "stats.json")
    assert report["cutoff_len"] == 4096
    assert report["totals"] == {"train": 1, "dev": 1, "test": 1}
    assert report["splits"]["train"]["tool_coverage"] == ["get_balance"]
    assert report["splits"]["test"]["lengths"]["over_cutoff"] == 0
    assert report["lengths"]["count"] == 3
    assert report["tokenizer"]["files"] == {}
    assert "all_checks_passed" in report["checks"]
    assert tokenizer.calls == 3
    assert (data / "stats.json").read_bytes() == stats_before
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["training_ready"] is True
    assert written["application_release_ready"] is False

def test_corrupt_checksum(dataset: tuple[Path, Path, FakeTokenizer]) -> None:
    data, out, tokenizer = dataset
    stats = json.loads((data / "stats.json").read_text(encoding="utf-8"))
    stats["artifact_sha256"]["train.jsonl"] = "0" * 64
    (data / "stats.json").write_text(json.dumps(stats), encoding="utf-8")

    report = check_v3.check(data, tokenizer, out)
    assert report["training_ready"] is False
    assert any("sha256 mismatch for train.jsonl" in item for item in report["failures"])
    assert not out.exists()

def test_wrong_prompt(dataset: tuple[Path, Path, FakeTokenizer]) -> None:
    data, out, tokenizer = dataset
    stats = json.loads((data / "stats.json").read_text(encoding="utf-8"))
    stats["prompt_sha256"] = "0" * 64
    (data / "stats.json").write_text(json.dumps(stats), encoding="utf-8")

    report = check_v3.check(data, tokenizer, out)
    assert report["training_ready"] is False
    assert any("prompt_sha256 mismatch" in item for item in report["failures"])
    assert not out.exists()

def test_unknown_gold_token(dataset: tuple[Path, Path, FakeTokenizer]) -> None:
    data, out, tokenizer = dataset
    assistant = {"kind": "plan", "steps": [
        {"id": "s1", "tool": "get_balance", "args": {}, "after": []},
        {"id": "s2", "tool": "verify_payee", "args": {"payee": "<PAYEE_9>"},
         "after": []},
    ]}
    _rewrite_split(data, "train", [
        _row("what is my balance", "fam-train",
             assistant=json.dumps(assistant, sort_keys=True))])

    report = check_v3.check(data, tokenizer, out)
    assert report["training_ready"] is False
    assert any("does not validate" in item for item in report["failures"])
    assert any("unknown_token" in item for item in report["failures"])
    assert not out.exists()

def test_cross_split_family(dataset: tuple[Path, Path, FakeTokenizer]) -> None:
    data, out, tokenizer = dataset
    _rewrite_split(data, "test", [_row("how much money do I have", "fam-train")])

    report = check_v3.check(data, tokenizer, out)
    assert report["training_ready"] is False
    assert any("source family crosses splits" in item for item in report["failures"])
    assert not out.exists()

def test_oversize(dataset: tuple[Path, Path, FakeTokenizer]) -> None:
    data, out, _tokenizer = dataset
    report = check_v3.check(data, FakeTokenizer(size=100), out, cutoff=50)

    assert report["training_ready"] is False
    assert report["lengths"]["max"] == 100
    assert report["lengths"]["over_cutoff"] == 3
    assert any("exceeds cutoff 50" in item for item in report["failures"])
    assert not out.exists()

def test_missing_coverage(
    dataset: tuple[Path, Path, FakeTokenizer], monkeypatch: pytest.MonkeyPatch
) -> None:
    data, out, tokenizer = dataset
    monkeypatch.setattr(
        check_v3, "REQUIRED_POSITIVE_TOOLS",
        frozenset({"get_balance", "list_contacts"}),
    )

    report = check_v3.check(data, tokenizer, out)
    assert report["training_ready"] is False
    assert any(
        "required positive tool missing from train: list_contacts" in item
        for item in report["failures"]
    )
    assert any(
        "required positive tool missing from test: list_contacts" in item
        for item in report["failures"]
    )
    assert not out.exists()

def test_existing_report_preserved(dataset: tuple[Path, Path, FakeTokenizer]) -> None:
    data, out, tokenizer = dataset
    out.write_text("sentinel", encoding="utf-8")

    report = check_v3.check(data, tokenizer, out)
    assert report["training_ready"] is False
    assert any(
        "refusing to overwrite existing report" in item for item in report["failures"]
    )
    assert out.read_text(encoding="utf-8") == "sentinel"


def test_zero_token_output_is_not_ready(dataset):
    data, out, _ = dataset
    report = check_v3.check(data, FakeTokenizer(size=0), out)
    assert report["training_ready"] is False
    assert any("no input IDs" in reason for reason in report["failures"])


def test_tokenizer_manifest_excludes_weights(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"mock weight file")
    files, _ = check_v3._token_files(tmp_path)
    assert set(files) == {"tokenizer.json"}
