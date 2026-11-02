"""Read-only analysis orchestration; no banking mutation or model-visible rows.

Only a canonical analysis question goes to code generation. A generated program
must agree with the vetted calculation before its output can reach a reply.
Every execution attempt, including fallback, uses the same sandbox and audit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import json
from typing import Any

from agent.analysis_data import analysis_rows
from agent.codegen import analysis_template, generate_analysis_code
from security.audit import AuditError, _CodeFingerprint, _check_pii
from security.sandbox import SandboxResult, check_code, run_analysis_code

QUESTIONS = {
    "weekend_vs_weekday": "Compare weekend spending with weekday spending",
    "by_hour": "Analyze spending by time of day",
    "by_weekday": "Analyze spending by weekday",
}


def run_spending_analysis(
    bank: Any, start: Any, end: Any, kind: str,
    audit_attempt: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    effective_kind = kind if kind in QUESTIONS else "by_weekday"
    rows = analysis_rows(bank, start, end)
    template = analysis_template(effective_kind)

    def attempt(code: str, source: str) -> SandboxResult:
        # Model text is untrusted even when syntactically valid. Do not put
        # phone/account-looking literals in the audit or execute such code.
        redacted = False
        try:
            if "pii_code" in check_code(code):
                raise AuditError("pii")
            _check_pii(code)
        except AuditError:
            import hashlib
            redacted = True
            outcome = SandboxResult(
                ok=False, result=None, error="violation:pii_code", runtime_ms=0,
                code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                violations=("pii_code",), backend="unavailable",
            )
        else:
            outcome = run_analysis_code(code, rows)
        audit_attempt({
            "code_sha256": _CodeFingerprint(code),
            "code": "" if redacted else code[:4000],
            "code_redacted": redacted, "source": source,
            "backend": outcome.backend, "ok": outcome.ok,
            "error": outcome.error, "runtime_ms": outcome.runtime_ms,
            "violations": list(outcome.violations), "row_count": len(rows),
        })
        return outcome

    try:
        code, source = generate_analysis_code(QUESTIONS[effective_kind], lang="en")
    except Exception:
        code, source = template, "template"
    if source not in {"template", "model"}:
        code, source = template, "template"
    outcome = attempt(code, source)
    verified = source == "template"
    if source == "model":
        # Schema validity is insufficient: an LLM could invent plausible totals
        # or return dates as numbers. Compare against an independently vetted
        # program, executed under the identical restrictions.
        reference = attempt(template, "template")
        verified = bool(outcome.ok and reference.ok and
            json.dumps(outcome.result, sort_keys=True, allow_nan=False) ==
            json.dumps(reference.result, sort_keys=True, allow_nan=False))
        if not verified:
            outcome, source = reference, "template"
    metadata = asdict(outcome)
    metadata.pop("result")
    metadata.update(source=source, verified=bool(outcome.ok and (verified or source == "template")))
    return {
        "raw": outcome.result if outcome.ok else None,
        "analysis": {"kind": effective_kind, "requested_kind": kind, "ok": outcome.ok},
        "sandbox": metadata,
    }
