"""Schema-only code proposals plus reviewed analysis templates.

The code generator never receives ledger values or the original conversation.
Even unknown personal data in a nominally tokenized question is excluded: only
the corresponding canonical analysis question is sent to the endpoint.
"""

from __future__ import annotations

import asyncio
import json
import os
import re

import httpx

from bank.categories import CATEGORIES
from security.sandbox import ALLOWED_IMPORTS, check_code

_RESPONSE_BYTES = 65536
_REQUEST_SECONDS = 15.0


TEMPLATES: dict[str, str] = {
    "weekend_vs_weekday": '''def analyze(rows):
    weekend = [r for r in rows if r["weekday"] >= 5]
    weekday = [r for r in rows if r["weekday"] < 5]
    weekend_total = sum(r["amount_fen"] for r in weekend)
    weekday_total = sum(r["amount_fen"] for r in weekday)
    weekend_days = len(set(r["day"] for r in weekend))
    weekday_days = len(set(r["day"] for r in weekday))
    weekend_average = (weekend_total + weekend_days // 2) // max(weekend_days, 1)
    weekday_average = (weekday_total + weekday_days // 2) // max(weekday_days, 1)
    categories = sorted(set(r["category"] for r in rows))
    differences = {}
    for category in categories:
        a = sum(r["amount_fen"] for r in weekend if r["category"] == category)
        b = sum(r["amount_fen"] for r in weekday if r["category"] == category)
        differences[category] = (a + weekend_days // 2) // max(weekend_days, 1) - (b + weekday_days // 2) // max(weekday_days, 1)
    categories = sorted(categories, key=lambda c: -abs(differences[c]))[:12]
    return {"weekend_total_fen": weekend_total, "weekday_total_fen": weekday_total,
            "weekend_days": weekend_days, "weekday_days": weekday_days,
            "weekend_average_fen": weekend_average, "weekday_average_fen": weekday_average,
            "difference_fen": weekend_average - weekday_average,
            "categories": categories, "category_differences_fens": [differences[c] for c in categories],
            "largest_category": categories[0] if categories else "other", "row_count": len(rows)}
''',
    "by_hour": '''def analyze(rows):
    totals = [sum(r["amount_fen"] for r in rows if r["hour"] // 6 == b) for b in range(4)]
    counts = [sum(1 for r in rows if r["hour"] // 6 == b) for b in range(4)]
    return {"bucket_totals_fens": totals, "bucket_counts": counts,
            "peak_bucket": max(range(4), key=lambda b: totals[b]),
            "total_fen": sum(totals), "row_count": len(rows)}
''',
    "by_weekday": '''def analyze(rows):
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    totals = [sum(r["amount_fen"] for r in rows if r["weekday"] == d) for d in range(7)]
    days = [len(set(r["day"] for r in rows if r["weekday"] == d)) for d in range(7)]
    averages = [(totals[d] + days[d] // 2) // max(days[d], 1) for d in range(7)]
    return {"weekday_totals_fens": totals, "weekday_averages_fens": averages,
            "weekday_days": days, "busiest_weekday": names[max(range(7), key=lambda d: totals[d])],
            "total_fen": sum(totals), "row_count": len(rows)}
''',
}

CANONICAL_QUESTIONS = {
    "weekend_vs_weekday": "Compare weekend spending with weekday spending, including totals, distinct spending days, average per spending day, and per-category average differences.",
    "by_hour": "Group spending into four six-hour buckets starting at midnight and identify the largest total.",
    "by_weekday": "Compare spending totals and averages per distinct spending day for each weekday.",
}


def infer_analysis_kind(question_tokenized: str) -> str:
    """Recognize a supported analysis; generic/custom defaults to weekdays."""
    text = question_tokenized.casefold()
    if re.search(r"weekend|周末|平日|工作日", text):
        return "weekend_vs_weekday"
    if re.search(r"time of day|by.hour|hour|几点|时段|时间段", text):
        return "by_hour"
    return "by_weekday"


def analysis_template(kind: str) -> str:
    return TEMPLATES.get(kind, TEMPLATES["by_weekday"])


def _system_prompt(kind: str) -> str:
    # Output keys are fixed for deterministic verification by trusted code.
    # The template is code describing the computation, not customer data.
    return (
        "Write only Python code defining analyze(rows). No markdown or commentary. "
        "No top-level code except imports and this function. No IO, filesystem, "
        "network, private attributes, dynamic code, or recursion. "
        "Rows have only: day (ISO date string), weekday (integer Monday=0 to Sunday=6), "
        "hour (integer 0..23), amount_fen (positive integer), category (category string). "
        "No data values are supplied. Money must use integer fen. Count distinct "
        "observed spending days, not all calendar days. Round averages half up to fen. "
        "Allowed imports: " + ", ".join(sorted(ALLOWED_IMPORTS)) + ". "
        "Allowed builtins: len, sum, min, max, sorted, round, abs, range, enumerate, zip, "
        "dict, list, set, tuple, str, int, float, bool, isinstance, any, all, map, filter, reversed. "
        "Return a flat dict of at most thirty keys, lowercase letters/digits/underscores; "
        "values may be finite numbers, booleans, categories, weekday names, or lists of "
        "at most twelve such scalars. Money scalars/lists must be integers. Categories: "
        + ", ".join(CATEGORIES) + ". Use exactly the result shape and semantics of "
        "this reviewed reference computation:\n" + analysis_template(kind)
    )


def _strip_fence(code: str) -> str:
    match = re.fullmatch(r"\s*```(?:python|py)?\s*\n(.*?)\n```\s*", code, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else code).strip()


async def _fetch_content(endpoint: str, request_options: dict[str, object]) -> object:
    async with httpx.AsyncClient(timeout=httpx.Timeout(2.0, connect=3.0), trust_env=False,
                                 follow_redirects=False, headers={"Accept-Encoding": "identity"}) as client:
        async with client.stream("POST", endpoint, **request_options) as response:
            response.raise_for_status()
            if response.headers.get("Content-Encoding", "").strip().lower() not in {"", "identity"}:
                raise ValueError("encoded_response")
            encoded = bytearray()
            # Raw identity bytes avoid an automatic decompression allocation
            # before the application byte budget can be checked.
            async for chunk in response.aiter_raw():
                if len(encoded) + len(chunk) > _RESPONSE_BYTES:
                    raise ValueError("response_size")
                encoded.extend(chunk)
            return json.loads(encoded)["choices"][0]["message"]["content"]


async def _bounded_content(endpoint: str, request_options: dict[str, object]) -> object:
    # Cancellation covers connection, response headers, and every body read;
    # async context managers close the response/client on every exit path.
    return await asyncio.wait_for(_fetch_content(endpoint, request_options), timeout=_REQUEST_SECONDS)


def generate_analysis_code(question_tokenized: str, *, lang: str) -> tuple[str, str]:
    """Propose checked code, falling back to the same-sandbox vetted template.

    Execution errors are handled by the caller, which must run
    ``analysis_template(kind)`` through the identical sandbox if a model proposal
    fails. This function neither executes code nor receives ledger rows.
    """
    kind = infer_analysis_kind(question_tokenized)
    template = analysis_template(kind)
    configured_codegen_url = os.environ.get("ANBAN_CODEGEN_URL")
    base_url = configured_codegen_url or os.environ.get("ANBAN_LOCAL_URL")
    model = os.environ.get("ANBAN_CODEGEN_MODEL") or os.environ.get("ANBAN_LOCAL_MODEL") or "anban-planner"
    if not base_url:
        return template, "template"
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        # This entry point runs in a synchronous worker. A caller already in an
        # event loop receives the safe template before any coroutine is created.
        return template, "template"
    base_url = base_url.rstrip("/")
    endpoint = base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt(kind)},
            {"role": "user", "content": CANONICAL_QUESTIONS[kind]},
        ],
        "temperature": 0,
        "max_tokens": 600,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request_options: dict[str, object] = {"json": payload}
    # A provider credential belongs only to the explicitly configured codegen
    # endpoint. Never forward it to the planner URL selected as a fallback.
    api_key = os.environ.get("ANBAN_CODEGEN_API_KEY") if configured_codegen_url else None
    if api_key:
        request_options["headers"] = {"Authorization": "Bearer " + api_key}
    try:
        content = asyncio.run(_bounded_content(endpoint, request_options))
        if not isinstance(content, str):
            return template, "template"
        code = _strip_fence(content)
        if not code or check_code(code):
            return template, "template"
        return code, "model"
    except (httpx.HTTPError, TimeoutError, ValueError, KeyError, IndexError, TypeError):
        return template, "template"
