"""Offline tests for the release evaluator: routing, provenance and data selection."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.prompts import prompt_for_profile
from training import release_eval
from training.release_eval import (
    EndpointPool,
    load_common_legacy,
    load_r3_heldout,
    with_prompt,
)

REFUSAL = '{"kind":"refuse","reason":"out_of_scope"}'


class _FakePlanner:
    """Stands in for OpenAICompatiblePlanner; records per-endpoint concurrency."""

    def __init__(self, base_url, model, timeout=120.0, temperature=0.0, extra_body=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.extra_body = dict(extra_body or {})
        self.supports_response_format = True
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def plan(self, system, user):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self._lock:
            self.active -= 1
        return REFUSAL


def _row(user: str, scenario: str, gold: str = REFUSAL, system: str = "old") -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": gold},
        ],
        "meta": {"scenario_id": scenario},
    }


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_endpoint_pool_is_exclusive_and_releases(monkeypatch):
    monkeypatch.setattr(release_eval, "OpenAICompatiblePlanner", _FakePlanner)
    pool = EndpointPool(["http://a/v1", "http://b/v1"], "anban-r3", "sha-r3")

    with ThreadPoolExecutor(max_workers=8) as executor:
        outputs = list(executor.map(lambda _: pool.plan("system", "user"), range(16)))

    assert outputs == [REFUSAL] * 16
    assert pool.clients.qsize() == 2
    assert all(client.inner.max_active == 1 for client in pool._clients)
    assert sum(client.inner.calls for client in pool._clients) == 16
    assert all(client.inner.calls > 0 for client in pool._clients)


def test_endpoint_is_released_when_a_request_raises(monkeypatch):
    class _Boom(_FakePlanner):
        def plan(self, system, user):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return "ok"

    monkeypatch.setattr(release_eval, "OpenAICompatiblePlanner", _Boom)
    pool = EndpointPool(["http://a/v1"], "anban-r3", "sha-r3")

    with pytest.raises(RuntimeError, match="boom"):
        pool.plan("system", "user")
    assert pool.plan("system", "user") == "ok"
    assert pool.clients.qsize() == 1


def test_adapter_hash_is_provenance_only(monkeypatch):
    monkeypatch.setattr(release_eval, "OpenAICompatiblePlanner", _FakePlanner)
    pool = EndpointPool(["http://a/v1"], "anban-r3", "sha-r3")
    wrapper = pool._clients[0]

    assert wrapper.extra_body["adapter_sha256"] == "sha-r3"
    assert "adapter_sha256" not in wrapper.inner.extra_body
    assert wrapper.inner.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert pool.extra_body[0]["adapter_sha256"] == "sha-r3"
    assert pool.model == "anban-r3" and pool.temperature == 0.0


def test_common_legacy_keeps_users_and_gold_but_uses_profile_prompts(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    rows = [
        _row("User: transfer", "T_transfer", gold='{"kind":"plan","steps":[]}'),
        _row("User: birthday", "O1_birthday"),
        _row("User: bill", "B6_bill_pay_unsupported"),
        _row("User: not heldout", "T_other"),
    ]
    _write_jsonl(legacy / "test.jsonl", rows)

    common = load_common_legacy(legacy, {"User: transfer", "User: birthday", "User: bill"})
    assert [sample["scenario_id"] for sample in common] == ["T_transfer"]

    r2 = with_prompt(common, "r2")
    r3 = with_prompt(common, "r3")
    assert [sample["user"] for sample in r2] == [sample["user"] for sample in r3]
    assert [sample["gold"] for sample in r2] == [sample["gold"] for sample in r3]
    assert [sample["id"] for sample in r2] == [sample["id"] for sample in r3]
    assert r2[0]["system"] == prompt_for_profile("r2")
    assert r3[0]["system"] == prompt_for_profile("r3")
    assert r2[0]["system"] != r3[0]["system"]


def test_r3_heldout_rejects_a_stale_system_prompt(tmp_path):
    path = tmp_path / "test.jsonl"
    _write_jsonl(path, [_row("User: hello", "T_transfer", system="old prompt")])

    with pytest.raises(ValueError, match="R3_SYSTEM_PROMPT"):
        load_r3_heldout(path)


def test_existing_output_is_refused_without_inference(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "sentinel.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    args = [
        "--r3-url", "http://unused:1/v1",
        "--r2-url", "http://unused:2/v1",
        "--r3-data", str(tmp_path / "missing-r3"),
        "--legacy-data", str(tmp_path / "missing-legacy"),
        "--r3-adapter", str(tmp_path / "missing-r3-adapter"),
        "--r2-adapter", str(tmp_path / "missing-r2-adapter"),
        "--out", str(out),
    ]
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        release_eval.main(args)

    assert sentinel.read_text(encoding="utf-8") == "keep me"
    assert sorted(item.name for item in out.iterdir()) == ["sentinel.txt"]
