"""Independent sandbox security test: safe probes and boundary regressions.

No potentially host-capable proposal executes here. Resource probes require
the real production backend and use 64 MB, short limits, and disposable rows.
"""
from __future__ import annotations

from datetime import date, datetime
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from agent import analysis_runner, codegen
from security import sandbox


@pytest.fixture
def offline(monkeypatch):
    for name in ("ANBAN_CODEGEN_URL", "ANBAN_CODEGEN_API_KEY", "ANBAN_LOCAL_URL"):
        monkeypatch.delenv(name, raising=False)


def _bank():
    return SimpleNamespace(list_transactions=lambda *args, **kwargs: [
        SimpleNamespace(amount_fen=-125, category="food", counterparty="SYNTHETIC_PERSON",
                        ts=datetime(2026, 11, 2, 9), memo="SYNTHETIC_MEMO"),
    ])


@pytest.mark.parametrize("code", [
    "import json as safe\ndef analyze(rows): return safe.loads.__globals__",
    "import json\ndef analyze(rows): return (lambda f: f.__globals__)(json.loads)",
    "from collections import Counter as C\ndef analyze(rows): return C.__mro__",
    "def analyze(rows): return [x for x in ().__class__.__base__.__subclasses__()]",
    "def analyze(rows): return str.format_map('{x.__class__}', {'x': rows})",
    "import json\ndef analyze(rows):\n json.dumps, x = (str, 1)\n return {}",
    "import math as m\ndef analyze(rows):\n m.pi += 1\n return {}",
    "def analyze(rows): return (lambda __import__: __import__('os'))(str)",
    "def analyze(rows): return rows.＿＿class＿＿",
])
def test_security_static_escape_attempts_never_launch(monkeypatch, code):
    monkeypatch.setattr(sandbox, "_capture", lambda *a, **kw: pytest.fail("unsafe code launched"))
    assert sandbox.check_code(code)
    outcome = sandbox.run_analysis_code(code, [])
    assert not outcome.ok and outcome.error.startswith("violation:")


def test_security_lone_surrogate_rejected_without_worker(monkeypatch):
    monkeypatch.setattr(sandbox, "_capture", lambda *a, **kw: pytest.fail("malformed code launched"))
    outcome = sandbox.run_analysis_code("def analyze(rows): return {'value': '\ud800'}", [])
    assert outcome.error == "violation:encoding"
    assert len(outcome.code_sha256) == 64


@pytest.mark.parametrize("replacement", ["True", "1.0", "2"])
def test_security_exact_reference_comparison_rejects_count_type_or_value_changes(monkeypatch, offline, replacement):
    monkeypatch.setenv("ANBAN_SANDBOX_DEV", "1")
    proposal = codegen.TEMPLATES["by_hour"].replace('"row_count": len(rows)', '"row_count": ' + replacement)
    monkeypatch.setattr(analysis_runner, "generate_analysis_code", lambda *a, **kw: (proposal, "model"))
    audits = []
    result = analysis_runner.run_spending_analysis(_bank(), date(2026, 11, 1), date(2026, 11, 2), "by_hour", audits.append)
    assert all(audit["ok"] for audit in audits)
    assert result["sandbox"]["source"] == "template"
    assert result["sandbox"]["verified"]
    assert type(result["raw"]["row_count"]) is int
    assert result["raw"]["row_count"] == 1
    assert "SYNTHETIC_PERSON" not in json.dumps(audits)
    assert "SYNTHETIC_MEMO" not in json.dumps(audits)


def test_security_float_money_list_is_not_verified(monkeypatch, offline):
    monkeypatch.setenv("ANBAN_SANDBOX_DEV", "1")
    proposal = codegen.TEMPLATES["by_hour"].replace('"bucket_totals_fens": totals', '"bucket_totals_fens": [float(v) for v in totals]')
    monkeypatch.setattr(analysis_runner, "generate_analysis_code", lambda *a, **kw: (proposal, "model"))
    audits = []
    result = analysis_runner.run_spending_analysis(_bank(), date(2026, 11, 1), date(2026, 11, 2), "by_hour", audits.append)
    assert result["sandbox"]["source"] == "template"
    assert all(type(value) is int for value in result["raw"]["bucket_totals_fens"])


def test_security_model_row_mutation_does_not_poison_reference(monkeypatch, offline):
    monkeypatch.setenv("ANBAN_SANDBOX_DEV", "1")
    proposal = codegen.TEMPLATES["by_hour"].replace("def analyze(rows):", "def analyze(rows):\n    rows[0]['amount_fen'] = 9999")
    monkeypatch.setattr(analysis_runner, "generate_analysis_code", lambda *a, **kw: (proposal, "model"))
    audits = []
    result = analysis_runner.run_spending_analysis(_bank(), date(2026, 11, 1), date(2026, 11, 2), "by_hour", audits.append)
    assert result["sandbox"]["source"] == "template"
    assert result["raw"]["total_fen"] == 125
    assert [event["source"] for event in audits] == ["model", "template"]


@pytest.mark.parametrize("value", [[1.5], [True], ["food"], [2**63], 125])
def test_security_money_lists_require_integer_fen(value):
    # Pure validation only: the production process is never started.
    assert not sandbox._validate_output({"bucket_totals_fens": value})


@pytest.mark.parametrize("literal", ["138_0013_8000", "0x3368c49f0", "'\\x31\\x33\\x38\\x30\\x30\\x31\\x33\\x38\\x30\\x30\\x30'"])
def test_security_obfuscated_pii_literals_are_redacted_before_execution(monkeypatch, offline, literal):
    proposal = "def analyze(rows):\n    number = " + literal + "\n    return {}"
    assert "pii_code" in sandbox.check_code(proposal)
    monkeypatch.setattr(analysis_runner, "generate_analysis_code", lambda *a, **kw: (proposal, "model"))
    attempted = []

    def fake_sandbox(code, rows):
        attempted.append(code)
        return sandbox.SandboxResult(True, {}, None, 0, "a" * 64, (), "subprocess-dev")

    monkeypatch.setattr(analysis_runner, "run_analysis_code", fake_sandbox)
    audits = []
    analysis_runner.run_spending_analysis(_bank(), date(2026, 11, 1), date(2026, 11, 2), "by_hour", audits.append)
    assert proposal not in attempted
    assert audits[0]["error"] == "violation:pii_code"
    assert audits[0]["code_redacted"] and audits[0]["code"] == ""


def test_security_production_command_limits_and_environment_are_explicit(monkeypatch):
    monkeypatch.delenv("ANBAN_SANDBOX_DEV", raising=False)
    monkeypatch.setattr(sandbox, "_production_backend", lambda: ("wsl-unshare", ("wsl.exe", "--exec", "/usr/bin/env", "-i", "/usr/bin/unshare", "-rn", "/usr/bin/prlimit", "python3")))
    observed = {}

    def capture(command, payload, timeout, limit):
        observed.update(command=command, payload=json.loads(payload), timeout=timeout, limit=limit)
        return 0, b"{}", None

    monkeypatch.setattr(sandbox, "_capture", capture)
    assert sandbox.run_analysis_code("def analyze(rows): return {}", [], timeout_s=0.25, mem_mb=64).ok
    assert "--as=67108864" in observed["command"]
    assert "--cpu=2" in observed["command"]
    assert observed["command"][-4:-2] == ["-I", "-S"]
    assert observed["payload"]["memory"] == 64 * 1024 * 1024
    assert observed["timeout"] == 0.25


@pytest.mark.parametrize("proposal", [
    "def analyze(rows):\n flag = True\n while flag: pass\n return {}",
    "def analyze(rows): return {'value': len([0] * (64 * 1024 * 1024))}",
    "def analyze(rows): return {'value': 'x' * (64 * 1024 * 1024)}",
])
def test_security_production_resource_limits_only(monkeypatch, proposal):
    monkeypatch.delenv("ANBAN_SANDBOX_DEV", raising=False)
    if sandbox._production_backend()[0] == "unavailable":
        pytest.skip("No real production backend; resource attack not run in dev mode")
    result = sandbox.run_analysis_code(proposal, [], timeout_s=0.3, mem_mb=64, max_output_bytes=1024)
    assert not result.ok
    assert result.error in {"timeout", "memory", "bad_output", "crash"}
    assert result.result is None


def test_security_codegen_bounds_untrusted_http_response_before_buffering(monkeypatch):
    consumed = []
    closed = []
    # A provider need not obey requested max_tokens. Use a small synthetic body
    # to demonstrate eager buffering safely, with no network or credentials.
    body = json.dumps({"choices": [{"message": {"content": " " * 131072}}]}).encode()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for offset in range(0, len(body), 1024):
                chunk = body[offset:offset + 1024]
                consumed.append(len(chunk))
                yield chunk

        async def aclose(self):
            closed.append(True)

    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream()))
    real_client = httpx.AsyncClient
    monkeypatch.setattr(codegen.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
    monkeypatch.setenv("ANBAN_CODEGEN_URL", "https://synthetic.invalid/v1")
    monkeypatch.delenv("ANBAN_CODEGEN_API_KEY", raising=False)
    assert codegen.generate_analysis_code("by hour", lang="en")[1] == "template"
    # 4,000 source characters plus JSON overhead fit comfortably in 64 KiB.
    # At most one bounded transport chunk is read to discover overflow; it is
    # rejected before appending to the application buffer.
    assert sum(consumed) <= 65536 + 1024
    assert sum(consumed) < len(body)
    assert closed == [True]


def test_security_codegen_elapsed_body_deadline_closes_response(monkeypatch):
    consumed, closed = [], []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            consumed.append(True)
            await asyncio.sleep(0.05)
            yield b" "

        async def aclose(self):
            closed.append(True)

    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream()))
    real_client = httpx.AsyncClient
    monkeypatch.setattr(codegen.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
    monkeypatch.setattr(codegen, "_REQUEST_SECONDS", 0.01)
    monkeypatch.setenv("ANBAN_CODEGEN_URL", "https://synthetic.invalid/v1")
    monkeypatch.delenv("ANBAN_CODEGEN_API_KEY", raising=False)
    assert codegen.generate_analysis_code("by hour", lang="en")[1] == "template"
    assert consumed == [True]
    assert closed == [True]


def test_security_codegen_rejects_compression_without_reading_body(monkeypatch):
    class Unreadable(httpx.AsyncByteStream):
        async def __aiter__(self):
            pytest.fail("encoded provider body must be rejected before decoding")
            yield b""

    transport = httpx.MockTransport(lambda request: httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=Unreadable()))
    real_client = httpx.AsyncClient
    monkeypatch.setattr(codegen.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
    monkeypatch.setenv("ANBAN_CODEGEN_URL", "https://synthetic.invalid/v1")
    monkeypatch.delenv("ANBAN_CODEGEN_API_KEY", raising=False)
    assert codegen.generate_analysis_code("by hour", lang="en")[1] == "template"
