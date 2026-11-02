"""Release evaluation for the served r3 candidate against the preserved r2.

Every score comes from the existing graders in :mod:`training.eval_planner` and
:mod:`evals.runner`; this module only selects data, routes real HTTP calls over
an endpoint pool and writes the artifacts.  It never invents predictions, gold
labels or metrics, and it has no cloud fallback.

    python -m training.release_eval --r3-url http://gpu-a/v1 \
        --r2-url http://gpu-b/v1 --r3-data data/r3 --legacy-data data/r2 \
        --r3-adapter r3.safetensors --r2-adapter r2.safetensors \
        --out outputs/release_eval_01

``--out`` must not exist.  Inference starts only after that check and after the
runtime prompts and dataset systems have been validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.planner import NO_THINKING, OpenAICompatiblePlanner
from agent.prompts import R3_SYSTEM_PROMPT, prompt_for_profile
from evals.runner import DEFAULT_CASES, load_cases, run_case, scorecard
from training.eval_planner import build_report, evaluate, load_samples

# r3 intentionally changed these behaviours; comparing them with r2 would
# measure the change, not a regression.
EXCLUDED_LEGACY_SCENARIOS = frozenset({"O1_birthday", "B6_bill_pay_unsupported"})
TIMEOUT = 120.0


# --------------------------------------------------------------------- helpers


def sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _data_file(path: Any) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / "test.jsonl"
    if not resolved.is_file():
        raise FileNotFoundError(f"dataset file not found: {resolved}")
    return resolved


def _adapter_sha(path: Any) -> str:
    if not Path(path).is_file():
        raise SystemExit(f"adapter not found: {path}")
    return sha256_file(path)


def prepare_out(out: Any) -> Path:
    """Refuse an existing output directory before any inference can run."""

    out = Path(out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f"refusing to overwrite existing output: {out}") from None
    return out


# -------------------------------------------------------------------- endpoint


class AdapterPlanner:
    """One endpoint with the adapter hash kept in provenance, not in the payload."""

    def __init__(
        self, base_url: str, model: str, adapter_sha256: str, timeout: float = TIMEOUT
    ) -> None:
        self.inner = OpenAICompatiblePlanner(
            base_url,
            model=model,
            timeout=timeout,
            temperature=0.0,
            extra_body=dict(NO_THINKING),
        )
        self.adapter_sha256 = adapter_sha256
        # These attributes are read by evaluate() for its provenance fingerprint.
        self.model = self.inner.model
        self.base_url = self.inner.base_url
        self.temperature = self.inner.temperature
        self.supports_response_format = self.inner.supports_response_format
        # Provenance only.  plan() delegates to the inner client, whose
        # extra_body still holds just NO_THINKING, so the hash is never sent.
        self.extra_body = {**self.inner.extra_body, "adapter_sha256": adapter_sha256}

    def plan(self, system: str, user: str) -> str:
        return self.inner.plan(system, user)


class EndpointPool:
    """Serialise requests per endpoint; ``workers`` equals the endpoint count."""

    def __init__(
        self, urls: list[str], model: str, adapter_sha256: str, timeout: float = TIMEOUT
    ) -> None:
        if not urls:
            raise ValueError("at least one endpoint URL is required")
        self._clients = [
            AdapterPlanner(url, model, adapter_sha256, timeout) for url in urls
        ]
        self.clients: queue.Queue[AdapterPlanner] = queue.Queue()
        for client in self._clients:
            self.clients.put(client)
        self.size = self.workers = len(self._clients)
        self.model = model
        self.base_url = [client.base_url for client in self._clients]
        self.temperature = 0.0
        self.supports_response_format = True
        self.extra_body = [client.extra_body for client in self._clients]

    def acquire(self) -> AdapterPlanner:
        return self.clients.get()

    def release(self, client: AdapterPlanner) -> None:
        self.clients.put(client)

    def plan(self, system: str, user: str) -> str:
        client = self.acquire()
        try:
            return client.plan(system, user)
        finally:
            self.release(client)


# ------------------------------------------------------------------------ data


def load_r3_heldout(r3_data: Any) -> list[dict[str, Any]]:
    samples = load_samples(_data_file(r3_data))
    for sample in samples:
        if sample["system"] != R3_SYSTEM_PROMPT:
            raise ValueError(
                f"r3 heldout sample {sample['id']} does not use the current "
                "R3_SYSTEM_PROMPT"
            )
    return samples


def _scenario(sample: dict[str, Any]) -> str:
    return sample["scenario_id"] or str((sample.get("meta") or {}).get("scenario") or "")


def load_common_legacy(legacy_data: Any, r3_users: set[str]) -> list[dict[str, Any]]:
    """Original legacy users/gold that the r3 heldout also contains, minus exclusions."""

    samples = load_samples(_data_file(legacy_data))
    common: list[dict[str, Any]] = []
    for sample in samples:
        scenario = _scenario(sample)
        if scenario in EXCLUDED_LEGACY_SCENARIOS or sample["user"] not in r3_users:
            continue
        common.append({**sample, "scenario_id": scenario})
    return common


def with_prompt(samples: list[dict[str, Any]], profile: str) -> list[dict[str, Any]]:
    prompt = prompt_for_profile(profile)
    return [{**sample, "system": prompt} for sample in samples]


# ----------------------------------------------------------------------- evals


def _failed(row: dict[str, Any]) -> bool:
    return bool(row.get("error")) or not all(row["metrics"].values())


def _eval_set(
    name: str, planner: Any, samples: list[dict[str, Any]], root: Path, data_path: Path
) -> dict[str, Any]:
    out = root / name
    records = evaluate(planner, samples, out, planner.size)
    args = argparse.Namespace(planner=name, model=planner.model, data=str(data_path))
    card, markdown = build_report(records, args)
    failures = [
        {
            "id": row["id"],
            "scenario_id": row["scenario_id"],
            "error": row.get("error", ""),
            "failed_metrics": [key for key, ok in row["metrics"].items() if not ok],
            "prediction": row["prediction"],
        }
        for row in records
        if _failed(row)
    ]
    with (out / "failures.jsonl").open("w", encoding="utf-8") as sink:
        for failure in failures:
            sink.write(json.dumps(failure, ensure_ascii=False) + "\n")
    card["failures"] = len(failures)
    card["error_rows"] = sum(1 for row in records if row.get("error"))
    card["failures_path"] = str(out / "failures.jsonl")
    _write_json(out / "scorecard.json", card)
    (out / "scorecard.md").write_text(markdown, encoding="utf-8")
    return card


def _run_model_sets(
    r3_pool: EndpointPool,
    r2_pool: EndpointPool,
    r3_samples: list[dict[str, Any]],
    common: list[dict[str, Any]],
    root: Path,
    r3_file: Path,
    legacy_file: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    jobs = {
        "r3_heldout": (r3_pool, r3_samples, r3_file),
        "common_r3": (r3_pool, with_prompt(common, "r3"), legacy_file),
        "common_r2": (r2_pool, with_prompt(common, "r2"), legacy_file),
    }
    cards: dict[str, Any] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(_eval_set, name, spec[0], spec[1], root, spec[2]): name
            for name, spec in jobs.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                cards[name] = future.result()
            except Exception as exc:  # noqa: BLE001 - reported, never hidden
                errors[name] = f"{type(exc).__name__}: {exc}"
    return cards, errors


def _run_cases(
    name: str,
    cases: list[dict[str, Any]],
    planner: Any,
    root: Path,
    k: int,
    label: str,
) -> dict[str, Any]:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    total = len(cases) * k
    with (out / "results.jsonl").open("w", encoding="utf-8") as sink, ThreadPoolExecutor(max_workers=planner.size) as executor:
        futures = {
            executor.submit(run_case, case, planner): rep
            for case in cases for rep in range(k)
        }
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            result["rep"] = futures[future]
            results.append(result)
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            if done % 10 == 0 or done == total:
                passed = sum(row["passed"] for row in results)
                print(f"[{name}] {done}/{total} ({passed} passed)", flush=True)
    card, markdown = scorecard(results, k, label)
    _write_json(out / "scorecard.json", card)
    (out / "scorecard.md").write_text(markdown, encoding="utf-8")
    return card


def _profile_env():
    previous = os.environ.get("ANBAN_PLANNER_PROFILE")
    os.environ["ANBAN_PLANNER_PROFILE"] = "r3"
    return previous


def _restore_profile(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("ANBAN_PLANNER_PROFILE", None)
    else:
        os.environ["ANBAN_PLANNER_PROFILE"] = previous


# ------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--r3-url", action="append", required=True)
    parser.add_argument("--r2-url", action="append", required=True)
    parser.add_argument("--r3-data", type=Path, required=True)
    parser.add_argument("--legacy-data", type=Path, required=True)
    parser.add_argument("--r3-adapter", type=Path, required=True)
    parser.add_argument("--r2-adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.k < 1:
        raise SystemExit("--k must be at least 1")
    # Refuse before loading data, hashing adapters or making any HTTP call.
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")

    r3_file = _data_file(args.r3_data)
    legacy_file = _data_file(args.legacy_data)
    r3_samples = load_r3_heldout(r3_file)
    r3_users = {sample["user"] for sample in r3_samples}
    common = load_common_legacy(legacy_file, r3_users)
    if not common:
        raise SystemExit("no common legacy samples; refusing an unlike comparison")

    r3_sha = _adapter_sha(args.r3_adapter)
    r2_sha = _adapter_sha(args.r2_adapter)
    out = prepare_out(args.out)

    r3_pool = EndpointPool(args.r3_url, "anban-r3", r3_sha)
    r2_pool = EndpointPool(args.r2_url, "anban-r2", r2_sha)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {
        "generated_at": generated_at,
        "out": str(out.resolve()),
        "endpoints": {"r3": list(args.r3_url), "r2": list(args.r2_url)},
        "adapter_hashes": {
            "r3": {"path": str(args.r3_adapter), "sha256": r3_sha},
            "r2": {"path": str(args.r2_adapter), "sha256": r2_sha},
        },
        "prompt_hashes": {
            "r3_system": sha256_text(R3_SYSTEM_PROMPT),
            "common_r3": sha256_text(prompt_for_profile("r3")),
            "common_r2": sha256_text(prompt_for_profile("r2")),
        },
        "dataset_hashes": {
            "r3_heldout": {"path": str(r3_file), "sha256": sha256_file(r3_file)},
            "legacy_common": {"path": str(legacy_file), "sha256": sha256_file(legacy_file)},
        },
        "samples": {
            "r3_heldout": len(r3_samples),
            "common_legacy": len(common),
            "excluded_legacy_scenarios": sorted(EXCLUDED_LEGACY_SCENARIOS),
        },
        "planner_provenance": {
            "r3": {
                "model": r3_pool.model,
                "base_url": r3_pool.base_url,
                "temperature": r3_pool.temperature,
                "extra_body": r3_pool.extra_body,
                "supports_response_format": r3_pool.supports_response_format,
                "timeout": TIMEOUT,
                "workers": r3_pool.size,
            },
            "r2": {
                "model": r2_pool.model,
                "base_url": r2_pool.base_url,
                "temperature": r2_pool.temperature,
                "extra_body": r2_pool.extra_body,
                "supports_response_format": r2_pool.supports_response_format,
                "timeout": TIMEOUT,
                "workers": r2_pool.size,
            },
        },
    }
    _write_json(out / "RELEASE_MANIFEST.json", manifest)

    errors: dict[str, str] = {}
    cards: dict[str, Any] = {}
    try:
        cards, run_errors = _run_model_sets(
            r3_pool, r2_pool, r3_samples, common, out, r3_file, legacy_file
        )
        errors.update(run_errors)
    except Exception as exc:  # noqa: BLE001
        errors["model_sets"] = f"{type(exc).__name__}: {exc}"

    previous = _profile_env()
    state_card: dict[str, Any] | None = None
    redteam_card: dict[str, Any] | None = None
    try:
        try:
            cases = load_cases(args.cases)
            state_card = _run_cases(
                "state_suite", cases, r3_pool, out, args.k, "r3 EndpointPool"
            )
        except Exception as exc:  # noqa: BLE001
            errors["state_suite"] = f"{type(exc).__name__}: {exc}"
            cases = []
        try:
            redteam = [case for case in cases if case["kind"] != "functional"]
            redteam_card = _run_cases(
                "redteam", redteam, r3_pool, out, 1, "r3 EndpointPool"
            )
        except Exception as exc:  # noqa: BLE001
            errors["redteam"] = f"{type(exc).__name__}: {exc}"
    finally:
        _restore_profile(previous)

    summary = {
        "generated_at": generated_at,
        "manifest": "RELEASE_MANIFEST.json",
        "endpoints": manifest["endpoints"],
        "adapter_hashes": manifest["adapter_hashes"],
        "prompt_hashes": manifest["prompt_hashes"],
        "dataset_hashes": manifest["dataset_hashes"],
        "runs": cards,
        "state_suite": state_card,
        "redteam": redteam_card,
        "errors": errors,
        "paths": {
            "r3_heldout": str(out / "r3_heldout"),
            "common_r3": str(out / "common_r3"),
            "common_r2": str(out / "common_r2"),
            "state_suite": str(out / "state_suite"),
            "redteam": str(out / "redteam"),
        },
        "scorecards": {
            name: str(out / name / "scorecard.json")
            for name in ("r3_heldout", "common_r3", "common_r2", "state_suite", "redteam")
        },
        "results": {
            "state_suite": str(out / "state_suite" / "results.jsonl"),
            "redteam": str(out / "redteam" / "results.jsonl"),
        },
    }
    _write_json(out / "RELEASE_SUMMARY.json", summary)
    if errors:
        print("incomplete; preserved outputs and errors in RELEASE_SUMMARY.json")
        for name, message in errors.items():
            print(f"  {name}: {message}")
        return 1
    print(f"release summary written to {out / 'RELEASE_SUMMARY.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
