"""Fail-closed, bounded execution of read-only Python analysis proposals.

The production backend uses a user/network namespace, not an OS filesystem jail.
The AST language and reduced builtins prevent file access. This is a defence in
depth boundary, not a facility for executing arbitrary Python packages.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tempfile
from typing import Any

from bank.categories import CATEGORIES
from security.audit import AuditError, _check_pii

ALLOWED_IMPORTS = frozenset({"math", "statistics", "datetime", "collections", "itertools", "json"})
ALLOWED_BUILTINS = frozenset({"len", "sum", "min", "max", "sorted", "round", "abs", "range", "enumerate", "zip", "dict", "list", "set", "tuple", "str", "int", "float", "bool", "isinstance", "any", "all", "map", "filter", "reversed"})
_FORBIDDEN = frozenset({"__import__", "eval", "exec", "compile", "open", "input", "globals", "locals", "vars", "getattr", "setattr", "delattr", "breakpoint", "help", "exit", "quit", "print", "type", "object", "super", "memoryview"})
_METHODS = frozenset({"append", "extend", "get", "items", "keys", "values", "setdefault", "update", "add", "count", "index", "copy", "pop", "sort", "reverse", "union", "intersection", "difference", "split", "strip", "lower", "upper", "startswith", "endswith", "replace", "join", "isoformat", "weekday", "isoweekday", "date", "total_seconds", "year", "month", "day", "hour", "minute", "second", "days", "seconds", "microseconds"})
_MODULE_MEMBERS = {
    "math": frozenset({"ceil", "floor", "sqrt", "fsum", "isfinite", "fabs", "prod", "trunc", "pow", "log", "log10", "exp", "pi", "e"}),
    "statistics": frozenset({"mean", "median", "median_low", "median_high", "mode", "multimode", "pstdev", "stdev", "pvariance", "variance", "fmean"}),
    "datetime": frozenset({"date", "datetime", "timedelta", "fromisoformat", "strptime"}),
    "collections": frozenset({"Counter", "defaultdict", "deque"}),
    "itertools": frozenset({"chain", "groupby", "islice", "accumulate", "product", "combinations", "permutations", "repeat", "count", "cycle", "zip_longest"}),
    "json": frozenset({"loads", "dumps"}),
}
_NODES = frozenset({ast.Module, ast.Import, ast.ImportFrom, ast.alias, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign, ast.Expr, ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass, ast.Name, ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Subscript, ast.Slice, ast.Attribute, ast.Call, ast.keyword, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot})


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    result: dict | None
    error: str | None
    runtime_ms: int
    code_sha256: str
    violations: tuple[str, ...]
    backend: str


def check_code(code: str) -> list[str]:
    """Accept only a deliberately small AST language; never execute a proposal."""
    if not isinstance(code, str) or len(code) > 4000:
        return ["code_size"]
    try:
        code.encode("utf-8")
    except UnicodeEncodeError:
        return ["encoding"]
    try:
        tree = ast.parse(code)
        nodes = list(ast.walk(tree))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return ["syntax"]
    errors: set[str] = set()
    try:
        _check_pii(code)
    except AuditError:
        errors.add("pii_code")
    if len(nodes) > 2000:
        errors.add("complexity")
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1 or any(not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef)) for n in tree.body):
        errors.add("top_level")
    aliases: dict[str, str] = {}
    for node in nodes:
        if type(node) not in _NODES:
            errors.add("node:" + type(node).__name__)
        if isinstance(node, ast.Constant):
            # Source spelling can hide a phone/account literal behind Python
            # escapes, numeric separators or a non-decimal integer notation.
            value = node.value
            if type(value) is int:
                value = str(value)
            elif type(value) is bytes:
                value = value.decode("utf-8", errors="replace")
            if type(value) is str:
                try:
                    _check_pii(value)
                except AuditError:
                    errors.add("pii_code")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if node not in tree.body:
                errors.add("nested_import")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in ALLOWED_IMPORTS:
                        errors.add("import")
                    aliases[alias.asname or alias.name] = alias.name
            else:
                if node.level or node.module not in ALLOWED_IMPORTS:
                    errors.add("import")
                for alias in node.names:
                    if alias.name not in _MODULE_MEMBERS.get(node.module, ()):
                        errors.add("import_member")
                    aliases[alias.asname or alias.name] = node.module or ""
        if isinstance(node, ast.alias) and ((node.asname or node.name).startswith("_") or node.name.startswith("_")):
            errors.add("private_name")
        if isinstance(node, ast.FunctionDef):
            if node not in functions or node.name != "analyze" or node.decorator_list or node.returns or getattr(node, "type_params", []):
                errors.add("function")
            args = node.args
            if len(args.args) != 1 or args.args[0].arg != "rows" or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults or args.kw_defaults or any(a.annotation for a in args.args):
                errors.add("signature")
        if isinstance(node, (ast.Name, ast.arg)):
            name = node.id if isinstance(node, ast.Name) else node.arg
            if name.startswith("_"):
                errors.add("private_name")
            if name in _FORBIDDEN:
                errors.add("forbidden_name")
        if isinstance(node, ast.Attribute):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                errors.add("attribute_write")
            if node.attr.startswith("_"):
                errors.add("private_attribute")
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            module = aliases.get(root.id) if isinstance(root, ast.Name) else None
            if module:
                if node.attr not in _MODULE_MEMBERS.get(module, ()) and node.attr not in _METHODS:
                    errors.add("module_attribute")
            elif node.attr not in _METHODS:
                errors.add("attribute")
        if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value is True:
            if not any(isinstance(child, ast.Break) for child in ast.walk(node)):
                errors.add("unbounded_loop")
        if isinstance(node, ast.comprehension) and node.is_async:
            errors.add("async")
    return sorted(errors)


def _validate_output(value: Any) -> bool:
    def scalar(v: Any) -> bool:
        return (type(v) is int and -(2**63) <= v < 2**63) or type(v) is bool or (type(v) is float and math.isfinite(v)) or (type(v) is str and v in (*CATEGORIES, "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
    if type(value) is not dict or len(value) > 30:
        return False
    for key, item in value.items():
        if not isinstance(key, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,40}", key) is None:
            return False
        if key.endswith("_fen") and type(item) is not int:
            return False
        if key.endswith("_fens") and not (type(item) is list and len(item) <= 12 and all(type(v) is int and -(2**63) <= v < 2**63 for v in item)):
            return False
        if not scalar(item) and not (type(item) is list and len(item) <= 12 and all(scalar(v) for v in item)):
            return False
    return True


def _validate_rows(rows: Any) -> bool:
    if type(rows) is not list or len(rows) > 100000:
        return False
    for row in rows:
        if type(row) is not dict or set(row) != {"day", "weekday", "hour", "amount_fen", "category"}:
            return False
        try:
            day = date.fromisoformat(row["day"])
        except (ValueError, TypeError):
            return False
        if day.isoformat() != row["day"] or type(row["weekday"]) is not int or row["weekday"] != day.weekday():
            return False
        if type(row["hour"]) is not int or not 0 <= row["hour"] <= 23:
            return False
        if type(row["amount_fen"]) is not int or not 0 < row["amount_fen"] < 2**63:
            return False
        if type(row["category"]) is not str or row["category"] not in CATEGORIES:
            return False
    return True


# This wrapper is fixed trusted code; proposals cannot obtain sys or real builtins.
_WRAPPER = r'''
import sys, json, builtins, importlib
payload = json.loads(sys.stdin.buffer.read())
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (payload["memory"], payload["memory"]))
    resource.setrlimit(resource.RLIMIT_CPU, (payload["cpu"], payload["cpu"]))
    for limit, value in ((resource.RLIMIT_CORE, 0), (resource.RLIMIT_FSIZE, 0), (resource.RLIMIT_NOFILE, 32), (resource.RLIMIT_NPROC, 0)):
        resource.setrlimit(limit, (value, value))
    import ctypes
    if ctypes.CDLL(None).prctl(38, 1, 0, 0, 0) != 0:
        sys.exit(44)
except ImportError:
    pass
allowed = {name: importlib.import_module(name) for name in payload["imports"]}
def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in allowed:
        raise ImportError("blocked")
    return allowed[name]
safe = {name: getattr(builtins, name) for name in payload["builtins"]}
safe["__import__"] = safe_import
scope = {"__builtins__": safe}
try:
    exec(compile(payload["code"], "<analysis>", "exec"), scope)
    result = scope["analyze"](payload["rows"])
    encoded = json.dumps(result, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > payload["output_limit"]:
        raise ValueError("bad_output")
    sys.stdout.write(encoded)
except (MemoryError, OverflowError):
    sys.exit(42)
except (ValueError, TypeError):
    sys.exit(43)
except BaseException:
    sys.exit(44)
'''


def _kill(process: subprocess.Popen) -> None:
    if os.name == "nt":
        # Kill the WSL launcher and any Windows descendants without a shell.
        killer = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "taskkill.exe")
        try:
            subprocess.run([killer, "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            pass
        process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _capture(command: list[str], payload: bytes, timeout: float, limit: int) -> tuple[int | None, bytes, str | None]:
    """Bound both streams while running, rather than buffering unlimited output."""
    directory = tempfile.TemporaryDirectory(prefix="anban-analysis-", ignore_cleanup_errors=True)
    try:
        # WSL's Windows RPC client requires SystemRoot. It is an OS path, not
        # inherited application state; /usr/bin/env -i below clears the Linux
        # worker's entire environment before unshare and Python start.
        environment = {"SystemRoot": os.environ.get("SystemRoot", r"C:\Windows")} if os.name == "nt" and os.path.basename(command[0]).lower() in {"wsl", "wsl.exe"} else {}
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment, cwd=directory.name, start_new_session=os.name != "nt", creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except OSError:
        directory.cleanup()
        return None, b"", "unavailable"
    output = bytearray()
    excessive = threading.Event()
    def read(stream: Any, target: bytearray | None) -> None:
        count = 0
        try:
            while chunk := stream.read1(1024):
                count += len(chunk)
                if count > limit:
                    excessive.set()
                    break
                if target is not None:
                    target.extend(chunk)
        except (OSError, ValueError):
            pass
    def write() -> None:
        try:
            process.stdin.write(payload)
            process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
    threads = [threading.Thread(target=read, args=(process.stdout, output), daemon=True), threading.Thread(target=read, args=(process.stderr, None), daemon=True), threading.Thread(target=write, daemon=True)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + timeout
    error = None
    while process.poll() is None:
        if excessive.is_set():
            error = "bad_output"
            break
        if time.monotonic() >= deadline:
            error = "timeout"
            break
        time.sleep(0.01)
    if error:
        _kill(process)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        _kill(process)
    for thread in threads:
        thread.join(timeout=0.2)
    for stream in (process.stdin, process.stdout, process.stderr):
        try:
            stream.close()
        except OSError:
            pass
    directory.cleanup()
    return process.returncode, bytes(output), error or ("bad_output" if excessive.is_set() else None)


@lru_cache(maxsize=1)
def _production_backend() -> tuple[str, tuple[str, ...]]:
    if os.name == "nt":
        wsl = shutil.which("wsl")
        if not wsl:
            return "unavailable", ()
        # --exec avoids WSL's default shell reinterpreting Python parentheses.
        prefix = (wsl, "-d", "Ubuntu-22.04", "--exec", "/usr/bin/env", "-i", "/usr/bin/unshare", "-rn", "/usr/bin/prlimit")
        python = "python3"
    else:
        unshare, prlimit, python = shutil.which("unshare"), shutil.which("prlimit"), shutil.which("python3")
        if not all((unshare, prlimit, python)):
            return "unavailable", ()
        prefix = (unshare, "-rn", prlimit)
    command = [*prefix, "--as=268435456", "--cpu=2", python, "-I", "-S", "-c", "print('sandbox-probe')"]
    # WSL cold-start can take several seconds before the bounded Python starts.
    status, output, error = _capture(command, b"", 10.0, 2048)
    if error or status != 0 or output.strip() != b"sandbox-probe":
        return "unavailable", ()
    return ("wsl-unshare" if os.name == "nt" else "unshare"), (*prefix, python)


def run_analysis_code(code: str, rows: list[dict], *, timeout_s: float = 5.0, mem_mb: int = 256, max_output_bytes: int = 16_000) -> SandboxResult:
    started = time.monotonic()
    digest = hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest() if isinstance(code, str) else hashlib.sha256(b"").hexdigest()
    violations = tuple(check_code(code))
    backend = "subprocess-dev" if os.environ.get("ANBAN_SANDBOX_DEV") == "1" else "unavailable"
    def finish(error: str | None, result: dict | None = None) -> SandboxResult:
        return SandboxResult(error is None, result, error, int((time.monotonic() - started) * 1000), digest, violations, backend)
    if violations:
        return finish("violation:" + violations[0])
    if type(timeout_s) not in (float, int) or not 0 < timeout_s <= 30 or type(mem_mb) is not int or not 32 <= mem_mb <= 1024 or type(max_output_bytes) is not int or not 1 <= max_output_bytes <= 65536:
        return finish("bad_output")
    if not _validate_rows(rows):
        return finish("bad_output")
    if backend == "subprocess-dev":
        command = [sys.executable, "-I", "-S", "-c", _WRAPPER]
    else:
        backend, prefix = _production_backend()
        if backend == "unavailable":
            return finish("unavailable")
        command = [*prefix[:-1], f"--as={mem_mb * 1024 * 1024}", f"--cpu={math.ceil(timeout_s) + 1}", prefix[-1], "-I", "-S", "-c", _WRAPPER]
    try:
        payload = json.dumps({"code": code, "rows": rows, "imports": sorted(ALLOWED_IMPORTS), "builtins": sorted(ALLOWED_BUILTINS), "memory": mem_mb * 1024 * 1024, "cpu": math.ceil(timeout_s) + 1, "output_limit": max_output_bytes}, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError, OverflowError):
        return finish("bad_output")
    if len(payload) > 8_000_000:
        return finish("bad_output")
    status, output, error = _capture(command, payload, timeout_s, max_output_bytes)
    if error:
        return finish(error)
    if status != 0:
        return finish("memory" if status == 42 else "bad_output" if status == 43 else "timeout" if status == -getattr(signal, "SIGXCPU", 24) else "crash")
    try:
        value = json.loads(output)
    except (ValueError, RecursionError):
        return finish("bad_output")
    return finish(None, value) if _validate_output(value) else finish("bad_output")
