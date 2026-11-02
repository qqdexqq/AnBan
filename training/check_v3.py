"""Read-only v3 training-data readiness check (never edits the dataset)."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any
from agent.prompts import R3_SYSTEM_PROMPT
from security.dag import PlanError, parse_planner_output, validate_plan
from training.eval_planner import known_tokens
from training.v3_samples import REQUIRED_POSITIVE_TOOLS
_SPLITS = ("train", "dev", "test")
_ARTIFACTS = ("train.jsonl", "dev.jsonl", "test.jsonl", "dataset_info.json")
_STATS = "stats.json"
def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _lengths(values: list[int], cutoff: int) -> dict[str, Any]:
    ordered = sorted(values)
    def pct(percent: float) -> int | None:
        return ordered[max(0, math.ceil(percent / 100.0 * len(ordered)) - 1)] if ordered else None
    return {"count": len(values), "min": ordered[0] if ordered else None,
            "max": ordered[-1] if ordered else None, "p50": pct(50), "p95": pct(95),
            "p99": pct(99), "over_cutoff": sum(v > cutoff for v in values)}
def _meta(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("meta")
    value = json.loads(value) if isinstance(value, str) else value
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("meta must decode to an object")
    return value
def _ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict) or hasattr(encoded, "get"):
        encoded = encoded.get("input_ids", [])
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)
def _token_files(tokenizer: Any) -> tuple[dict[str, str], str | None]:
    if not isinstance(tokenizer, (str, Path)):
        return {}, None
    path = Path(tokenizer)
    if path.is_file():
        return {path.name: _sha(path)}, str(path)
    if path.is_dir():
        return {str(p.relative_to(path)): _sha(p)
                for p in sorted(path.iterdir()) if p.is_file() and p.name in {
                    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                    "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model"}}, str(path)
    return {}, str(path)
def _write_report(out: Path, report: dict[str, Any]) -> str:
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return ""
    except OSError as exc:
        return f"could not write report {out}: {exc}"
def _hash_checks(stats: dict[str, Any], actual: dict[str, str], prompt_sha: str,
                 failures: list[str], checks: list[str]) -> None:
    before = len(failures)
    recorded = stats.get("artifact_sha256")
    if not isinstance(recorded, dict):
        failures.append("stats.json has no artifact_sha256 map")
    else:
        for name in _ARTIFACTS:
            expected, observed = recorded.get(name), actual[name]
            if not isinstance(expected, str) or not expected:
                failures.append(f"stats artifact_sha256 missing {name}")
            elif expected != observed:
                failures.append(f"sha256 mismatch for {name}: {expected} != {observed}")
    if len(failures) == before:
        checks.append("artifact_sha256_match")
    before = len(failures)
    want = stats.get("prompt_sha256") or stats.get("promptSHA256")
    if not isinstance(want, str) or not want:
        failures.append("stats.json has no prompt_sha256")
    elif want != prompt_sha:
        failures.append(f"prompt_sha256 mismatch: {want} != {prompt_sha}")
    if len(failures) == before:
        checks.append("prompt_sha256_match")
def check(data: Path, tokenizer: Any, out: Path, cutoff: int = 4096) -> dict[str, Any]:
    """Validate ``data``; write the report to a brand-new ``out`` only on success."""
    data, out, cutoff = Path(data), Path(out), int(cutoff)
    prompt_sha = hashlib.sha256(R3_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    failures: list[str] = []
    checks: list[str] = []
    report: dict[str, Any] = {
        "training_ready": False, "application_release_ready": False,
        "prompt_sha256": prompt_sha, "artifact_sha256": {}, "cutoff_len": cutoff,
        "totals": {}, "splits": {}, "tokenizer": {}, "checks": checks,
        "failures": failures}
    if out.exists():
        failures.append(f"refusing to overwrite existing report: {out}")
        return report
    if cutoff <= 0:
        failures.append(f"cutoff must be positive, got {cutoff}")
    missing = [name for name in _ARTIFACTS + (_STATS,) if not (data / name).is_file()]
    if missing:
        failures.append("missing required artifacts: " + ", ".join(missing))
        return report
    checks.append("required_artifacts_present")
    report["artifact_sha256"] = {name: _sha(data / name) for name in _ARTIFACTS + (_STATS,)}
    try:
        stats = json.loads((data / _STATS).read_text(encoding="utf-8"))
        info = json.loads((data / "dataset_info.json").read_text(encoding="utf-8"))
        if not isinstance(stats, dict) or not isinstance(info, dict):
            raise ValueError("stats.json and dataset_info.json must be objects")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        failures.append(f"malformed required artifact: {exc}")
        return report
    _hash_checks(stats, report["artifact_sha256"], prompt_sha, failures, checks)
    for alias, filename in (("anban_train", "train.jsonl"), ("anban_dev", "dev.jsonl")):
        entry = info.get(alias, {})
        if not isinstance(entry, dict) or entry.get("file_name") != filename or entry.get("formatting") != "sharegpt":
            failures.append(f"dataset_info.json: invalid {alias} mapping")
    checks.append("dataset_info_valid")
    split_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for split in _SPLITS:
        rows: list[tuple[int, dict[str, Any]]] = []
        try:
            with (data / f"{split}.jsonl").open(encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    if line.strip():
                        try:
                            rows.append((line_no, json.loads(line)))
                        except json.JSONDecodeError as exc:
                            failures.append(f"{split}.jsonl:{line_no}: malformed JSON: {exc.msg}")
        except OSError as exc:
            failures.append(f"cannot read {split}.jsonl: {exc}")
        split_rows[split] = rows
    report["totals"] = {split: len(rows) for split, rows in split_rows.items()}
    families: dict[str, set[str]] = {}
    user_split: dict[str, str] = {}
    tools = {split: set() for split in _SPLITS}
    before = len(failures)
    for split, rows in split_rows.items():
        for line_no, row in rows:
            tag = f"{split}.jsonl:{line_no}"
            messages = row.get("messages") if isinstance(row, dict) else None
            if not (isinstance(messages, list) and len(messages) == 3
                    and all(isinstance(m, dict) for m in messages)):
                failures.append(f"{tag}: expected exactly 3 message objects"); continue
            if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
                failures.append(f"{tag}: roles must be system/user/assistant"); continue
            system, user, assistant = (m.get("content") for m in messages)
            if not all(isinstance(text, str) for text in (system, user, assistant)):
                failures.append(f"{tag}: message content must be text"); continue
            if system != R3_SYSTEM_PROMPT:
                failures.append(f"{tag}: system prompt is not the current R3 prompt")
            if not user.strip():
                failures.append(f"{tag}: empty user prompt")
            if not assistant.strip():
                failures.append(f"{tag}: no assistant content"); continue
            try:
                family = _meta(row).get("family")
            except (json.JSONDecodeError, ValueError) as exc:
                failures.append(f"{tag}: malformed meta: {exc}"); family = None
            if not isinstance(family, str) or not family:
                failures.append(f"{tag}: meta.family is missing")
            else:
                families.setdefault(family, set()).add(split)
            prior = user_split.setdefault(user, split)
            if prior != split:
                failures.append(f"{tag}: user prompt also appears in {prior}")
            try:
                parsed = parse_planner_output(assistant)
                validate_plan(parsed, known_tokens(user))
            except PlanError as exc:
                failures.append(f"{tag}: assistant does not validate: {exc}"); continue
            if parsed.kind == "plan":
                tools[split].update(step.tool for step in parsed.steps)
    for family, splits in families.items():
        if len(splits) > 1:
            failures.append(f"source family crosses splits: {family} -> {sorted(splits)}")
    if len(failures) == before:
        checks.append("messages_gold_and_family_isolation")
    before = len(failures)
    for tool in sorted(REQUIRED_POSITIVE_TOOLS):
        for split in ("train", "test"):
            if tool not in tools[split]:
                failures.append(f"required positive tool missing from {split}: {tool}")
    if len(failures) == before:
        checks.append("required_tool_coverage")
    report["splits"] = {split: {"count": len(rows), "tool_coverage": sorted(tools[split])}
                        for split, rows in split_rows.items()}
    if failures:
        return report
    if hasattr(tokenizer, "apply_chat_template"):
        loaded, (files, tokenizer_path) = tokenizer, _token_files(tokenizer)
    else:
        tokenizer_path = str(tokenizer)
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            failures.append(f"transformers is required to load the tokenizer: {exc}")
            return report
        try:
            loaded = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        except Exception as exc:  # noqa: BLE001 - surfaced in the report
            failures.append(f"could not load tokenizer {tokenizer_path}: "
                            f"{type(exc).__name__}: {exc}")
            return report
        files, tokenizer_path = _token_files(tokenizer)
    report["tokenizer"] = {"path": tokenizer_path, "files": files}
    before = len(failures)
    lengths_by_split = {split: [] for split in _SPLITS}
    for split, rows in split_rows.items():
        for line_no, row in rows:
            try:
                encoded = loaded.apply_chat_template(
                    row["messages"], tokenize=True, enable_thinking=False,
                    add_generation_prompt=False, return_dict=False)
                length = len(_ids(encoded))
                if length == 0:
                    raise ValueError("tokenizer returned no input IDs")
            except Exception as exc:  # noqa: BLE001 - surfaced in the report
                failures.append(f"{split}.jsonl:{line_no}: tokenization failed: "
                                f"{type(exc).__name__}: {exc}")
                continue
            lengths_by_split[split].append(length)
            if length > cutoff:
                failures.append(f"{split}.jsonl:{line_no}: {length} tokens "
                                f"exceeds cutoff {cutoff}")
    all_lengths = [value for values in lengths_by_split.values() for value in values]
    report["lengths"] = _lengths(all_lengths, cutoff)
    for split, values in lengths_by_split.items():
        report["splits"][split]["lengths"] = _lengths(values, cutoff)
    if len(failures) == before:
        checks.append("tokenization_within_cutoff")
    if failures:
        return report
    report["training_ready"] = True
    checks.append("all_checks_passed")
    error = _write_report(out, report)
    if error:
        failures.append(error)
        report["training_ready"] = False
        checks.remove("all_checks_passed")
    return report
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=4096)
    return parser.parse_args(argv)
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check(args.data, args.tokenizer, args.out, cutoff=args.cutoff)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("training_ready") else 1
if __name__ == "__main__":
    raise SystemExit(main())
