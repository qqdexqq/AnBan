"""Tests for the teacher-LLM utterance generator (no network)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

import training.gen_utterances as gen
from training.gen_utterances import (
    TeacherClient,
    TeacherError,
    TeacherUnavailable,
)

SECRET = "sk-test-secret-do-not-log"
PERSONA = {
    "style": "short, lowercase, assumes context",
    "channel": ["app_text"],
}
SEED = "transfer 300 to Li Na"
MUST_KEEP = ["Li Na", "300"]


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gen, "RETRY_BACKOFF_SECONDS", 0.0)


def _response(variants: list[str]) -> httpx.Response:
    content = json.dumps({"variants": variants}, ensure_ascii=False)
    return httpx.Response(
        200, json={"choices": [{"message": {"content": content}}]}
    )


def _client(
    tmp_path: Path,
    handler: object,
    *,
    model: str = "deepseek-chat",
) -> TeacherClient:
    return TeacherClient(
        api_key=SECRET,
        model=model,
        cache_path=tmp_path / "teacher_cache.jsonl",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )


def test_request_shape(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(["can you transfer 300 to Li Na"])

    client = _client(tmp_path, handler)
    result = client.paraphrase(SEED, PERSONA, "en", 1, MUST_KEEP)
    client.close()

    assert result == ["can you transfer 300 to Li Na"]
    assert len(requests) == 1

    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.deepseek.com/chat/completions"
    assert request.headers["authorization"] == f"Bearer {SECRET}"

    payload = json.loads(request.content)
    assert payload["model"] == "deepseek-chat"
    assert payload["temperature"] == 1.0
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0]["role"] == "user"
    prompt = payload["messages"][0]["content"]
    assert "short, lowercase, assumes context" in prompt
    assert SEED in prompt
    assert json.dumps(MUST_KEEP, ensure_ascii=False) in prompt
    assert "中英夹杂" in prompt


def test_cache_hit_skips_http_and_survives_new_client(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(["can you transfer 300 to Li Na"])

    cache_path = tmp_path / "cache.jsonl"
    first = TeacherClient(
        api_key=SECRET,
        cache_path=cache_path,
        transport=httpx.MockTransport(handler),
    )
    result1 = first.paraphrase(SEED, PERSONA, "en", 1, MUST_KEEP)
    result2 = first.paraphrase(SEED, PERSONA, "en", 1, MUST_KEEP)
    first.close()

    assert calls == 1
    assert result2 == result1

    second = TeacherClient(
        api_key=SECRET,
        cache_path=cache_path,
        transport=httpx.MockTransport(handler),
    )
    result3 = second.paraphrase(SEED, PERSONA, "en", 1, MUST_KEEP)
    second.close()

    assert calls == 1
    assert result3 == result1

    rows = [
        json.loads(line)
        for line in cache_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["model"] == "deepseek-chat"
    assert rows[0]["variants"] == result1
    assert set(rows[0]) == {"key", "model", "variants"}


def test_retry_on_429_then_success(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return _response(["can you transfer 300 to Li Na"])

    client = _client(tmp_path, handler)
    result = client.paraphrase(SEED, PERSONA, "en", 1, MUST_KEEP)
    client.close()

    assert calls == 2
    assert result == ["can you transfer 300 to Li Na"]


def test_gives_up_after_max_attempts_of_500s(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, text="internal error")

    client = _client(tmp_path, handler)
    with pytest.raises(TeacherError) as excinfo:
        client.paraphrase(SEED, PERSONA, "en", 1, MUST_KEEP)
    client.close()

    assert calls == gen.MAX_ATTEMPTS
    assert f"HTTP 500 after {gen.MAX_ATTEMPTS} attempts" in str(excinfo.value)
    assert SECRET not in str(excinfo.value)
    assert SECRET not in caplog.text


def test_connection_reset_is_retried(tmp_path: Path) -> None:
    # The data v2 build aborted on one reset connection.
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("connection reset", request=request)
        return _response(["can you transfer 300 to Li Na"])

    client = _client(tmp_path, handler)
    result = client.paraphrase(SEED, PERSONA, "en", 1, MUST_KEEP)
    client.close()

    assert calls == 2
    assert result == ["can you transfer 300 to Li Na"]


def test_filters_bad_variants(tmp_path: Path) -> None:
    variants = [
        "can you transfer 300 to Li Na",
        "300 to Li Na please",
        "Can   you transfer 300 to Li Na!!!",
        SEED,
        "transfer 300 to someone",
        "please transfer 300 to Li Na " + "x" * 50,
        "transfer 300 to Li Na " + "y" * 250,
        "transfer 300 to Li Na 13800138000",
        "",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _response(variants)

    client = _client(tmp_path, handler)
    result = client.paraphrase(SEED, PERSONA, "en", len(variants), MUST_KEEP)
    client.close()

    assert result == [
        "can you transfer 300 to Li Na",
        "300 to Li Na please",
    ]


def test_no_key_raises_teacher_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(TeacherUnavailable, match="DEEPSEEK_API_KEY"):
        TeacherClient(cache_path=tmp_path / "cache.jsonl")


def test_paraphrase_many_preserves_order(tmp_path: Path) -> None:
    seeds = ("seed one", "seed two", "seed three")

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][0]["content"]
        for seed in seeds:
            if seed in prompt:
                return _response([f"say {seed} now"])
        return _response([])

    jobs = [
        {
            "seed_text": seed,
            "persona": PERSONA,
            "lang": "en",
            "n": 1,
            "must_keep": [seed],
        }
        for seed in seeds
    ]

    client = _client(tmp_path, handler)
    result = client.paraphrase_many(jobs, workers=3)
    client.close()

    assert result == [[f"say {seed} now"] for seed in seeds]


def test_cli_check_reports_missing_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert gen.main(["--check"]) == 1
    output = capsys.readouterr().out
    assert "not configured" in output
