"""Adversarial and resource-boundary checks for the restricted analysis runner."""
import hashlib
import os

import pytest

from security import sandbox


@pytest.mark.parametrize("code", [
    "import os\ndef analyze(rows): return {}",
    "def analyze(rows): return __import__('os')",
    "def analyze(rows): return open('secret')",
    "def analyze(rows): return eval('1')",
    "def analyze(rows): return exec('1')",
    "def analyze(rows): return getattr(rows, 'append')",
    "def analyze(rows): return rows.__class__",
    "def analyze(rows):\n while True: pass",
    "class C: pass\ndef analyze(rows): return {}",
    "x = 2\ndef analyze(rows): return {}",
    "def analyze(rows): return {}\n" + " " * 4000,
    "def analyze(rows):\n global x\n return {}",
    "def analyze(rows):\n with rows: pass\n return {}",
    "def analyze(rows):\n try: return {}\n except: return {}",
    "def analyze(rows): return '{0.__class__}'.format(rows)",
    "def analyze(rows): return [x async for x in rows]",
    "def analyze(rows, secret=1): return {}",
    "@print\ndef analyze(rows): return {}",
    "from math import __dict__\ndef analyze(rows): return {}",
    "def analyze(rows):\n def nested(): return {}\n return nested()",
    "def analyze(rows): return math.not_allowed()",
    "def analyze(rows): return print('extra output')",
])
def test_static_rejects_forbidden_constructs(code):
    assert sandbox.check_code(code)


@pytest.mark.parametrize("code", [
    "def analyze(rows): return {'total_fen': sum(r['amount_fen'] for r in rows)}",
    "import math\ndef analyze(rows): return {'value': math.floor(1.2)}",
    "from statistics import mean\ndef analyze(rows): return {'value': mean([1, 2])}",
    "def analyze(rows):\n totals = {}\n for row in rows:\n  totals[row['category']] = totals.get(row['category'], 0) + row['amount_fen']\n return {'categories': list(totals.keys())}",
    "def analyze(rows): return {'value': sum(map(lambda r: r['amount_fen'], rows))}",
])
def test_static_accepts_small_analysis_language(code):
    assert sandbox.check_code(code) == []


@pytest.fixture
def dev(monkeypatch):
    monkeypatch.setenv("ANBAN_SANDBOX_DEV", "1")


def test_exact_analysis_result_and_metadata(dev):
    rows = [{"day": "2026-11-01", "weekday": 6, "hour": 12, "amount_fen": 125, "category": "food"}, {"day": "2026-11-02", "weekday": 0, "hour": 9, "amount_fen": 270, "category": "transport"}]
    code = "def analyze(rows): return {'total_fen': sum(r['amount_fen'] for r in rows), 'count': len(rows)}"
    result = sandbox.run_analysis_code(code, rows)
    assert result.ok, result
    assert result.result == {"total_fen": sum(r["amount_fen"] for r in rows), "count": len(rows)}
    assert result.code_sha256 == hashlib.sha256(code.encode()).hexdigest()
    assert result.backend == "subprocess-dev"
    assert result.runtime_ms >= 0
    assert not result.violations


def test_timeout_kills_nonconstant_infinite_loop(dev):
    result = sandbox.run_analysis_code("def analyze(rows):\n flag = True\n while flag: pass\n return {}", [], timeout_s=0.15)
    assert result.error == "timeout"
    assert not result.ok and result.result is None
    assert result.runtime_ms < 5000


def test_impossible_allocation_fails_bounded(dev):
    result = sandbox.run_analysis_code("def analyze(rows): return {'value': len(list(range(10 ** 30)))}", [], timeout_s=1)
    assert result.error in {"memory", "timeout"}, result


@pytest.mark.parametrize("literal", [
    "{'nested': {'value': 1}}", "{'values': list(range(1000))}",
    "{'value': 'secret customer text'}", "{'total_fen': 1.5}",
    "{'total_fen': True}", "{'bad-key': 1}", "[1, 2]",
    "{'value': float('nan')}", "{'value': float('inf')}",
    "{'value': None}", "{str(i): i for i in range(31)}",
])
def test_bad_outputs_never_return_data(dev, literal):
    result = sandbox.run_analysis_code("def analyze(rows): return " + literal, [])
    assert result.error == "bad_output", result
    assert result.result is None


def test_output_limit_is_enforced_in_child(dev):
    result = sandbox.run_analysis_code("def analyze(rows): return {'value': 123456789}", [], max_output_bytes=5)
    assert result.error == "bad_output"


def test_imports_have_restricted_bridge(dev):
    result = sandbox.run_analysis_code("import math\nfrom statistics import mean\ndef analyze(rows): return {'value': math.floor(mean([1, 2, 3]))}", [])
    assert result.ok and result.result == {"value": 2}, result


def test_unavailable_backend_never_falls_back(monkeypatch):
    monkeypatch.delenv("ANBAN_SANDBOX_DEV", raising=False)
    monkeypatch.setattr(sandbox, "_production_backend", lambda: ("unavailable", ()))
    result = sandbox.run_analysis_code("def analyze(rows): return {}", [])
    assert result.error == "unavailable"
    assert result.backend == "unavailable"


def test_nonexplicit_dev_flag_is_ignored(monkeypatch):
    monkeypatch.setenv("ANBAN_SANDBOX_DEV", "true")
    monkeypatch.setattr(sandbox, "_production_backend", lambda: ("unavailable", ()))
    assert sandbox.run_analysis_code("def analyze(rows): return {}", []).error == "unavailable"


def test_static_failure_does_not_launch(monkeypatch, dev):
    def unexpected(*args, **kwargs):
        pytest.fail("unsafe proposal launched")
    monkeypatch.setattr(sandbox, "_capture", unexpected)
    result = sandbox.run_analysis_code("import os\ndef analyze(rows): return {}", [])
    assert result.error.startswith("violation:")


def test_wrapper_contains_only_parent_builtins_not_inherited_environment(dev, monkeypatch):
    # Even if the parent has credentials, the worker starts with env={}.
    monkeypatch.setenv("ANBAN_TEST_SECRET", "must-not-inherit")
    observed = {}
    original = sandbox.subprocess.Popen
    def spy(*args, **kwargs):
        observed.update(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(sandbox.subprocess, "Popen", spy)
    assert sandbox.run_analysis_code("def analyze(rows): return {}", []).ok
    assert observed["env"] == {}


def test_capture_caps_arbitrary_stdout_and_stderr():
    # Trusted helper test; generated proposals cannot import sys or call print.
    for stream in ("stdout", "stderr"):
        command = [sandbox.sys.executable, "-I", "-S", "-c", f"import sys; sys.{stream}.write('x' * 100000)"]
        status, output, error = sandbox._capture(command, b"", 2, 1024)
        assert error == "bad_output"
        assert len(output) <= 1024


def test_network_namespace_has_no_external_route():
    backend, prefix = sandbox._production_backend()
    if backend == "unavailable":
        pytest.skip("production namespace backend unavailable on this host")
    # Trusted test-only command bypasses AST checking to test the actual boundary.
    snippet = "import socket; s=socket.socket(); s.settimeout(0.2); print(s.connect_ex(('1.1.1.1', 443)))"
    command = [*prefix[:-1], "--as=268435456", "--cpu=2", prefix[-1], "-I", "-S", "-c", snippet]
    status, output, error = sandbox._capture(command, b"", 10, 2048)
    assert error is None and status == 0
    assert output.strip() != b"0"


def test_proposal_cannot_replace_trusted_json_encoder():
    code = '''import json
def analyze(rows):
    json.dumps = lambda value: '{"total_fen":1}'
    return {}
'''
    assert "attribute_write" in sandbox.check_code(code)


def test_production_backend_runs_checked_code(monkeypatch):
    monkeypatch.delenv("ANBAN_SANDBOX_DEV", raising=False)
    backend, prefix = sandbox._production_backend()
    if backend == "unavailable":
        pytest.skip("production namespace backend unavailable on this host")
    result = sandbox.run_analysis_code("def analyze(rows): return {'total_fen': sum(r['amount_fen'] for r in rows)}", [])
    assert result.ok, result
    assert result.result == {"total_fen": 0}
    assert result.backend in {"unshare", "wsl-unshare"}


@pytest.mark.parametrize("change", [
    {"counterparty": "secret"}, {"day": "not a date"}, {"day": "20261101"},
    {"weekday": 0}, {"weekday": True}, {"hour": 24}, {"hour": False},
    {"amount_fen": True}, {"amount_fen": 0}, {"amount_fen": 2**63},
    {"category": "secret person"},
])
def test_input_schema_rejects_sensitive_or_invalid_rows(dev, change):
    row = {"day": "2026-11-01", "weekday": 6, "hour": 12, "amount_fen": 125, "category": "food"}
    row.update(change)
    result = sandbox.run_analysis_code("def analyze(rows): return {}", [row])
    assert result.error == "bad_output"
    assert result.result is None


def test_integer_output_is_bounded(dev):
    assert sandbox.run_analysis_code("def analyze(rows): return {'value': 2 ** 100}", []).error == "bad_output"
