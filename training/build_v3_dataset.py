"""Guarded v3 dataset builder.

The builder is deliberately *standalone* from ``training/build_dataset.py``: it
writes to a brand-new directory only, replays the legacy dataset read-only with
the R3 system prompt, and adds the code-built v3 samples from
:mod:`training.v3_samples`.

Safety rules enforced here:

* ``--out`` is required and must not exist at all (even with identical bytes).
* Known legacy / model / output artifact roots are rejected.
* The dataset is staged next to ``--out`` and moved into place in one rename;
  nothing is ever deleted from an existing path.
* The legacy tree is only read, never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from agent.prompts import R3_SYSTEM_PROMPT
from security.dag import parse_planner_output, validate_plan
from training.v3_samples import REQUIRED_POSITIVE_TOOLS, generate_samples

_ROOT = Path(__file__).resolve().parents[1]
_SCENARIOS_PATH = _ROOT / "data_spec" / "scenarios.yaml"
_SEGMENTS_PATH = _ROOT / "data_spec" / "customer_segments.yaml"
_DEFAULT_LEGACY = _ROOT / "training" / "data"
_SPLITS = ("train", "dev", "test")

# Never write a dataset into an old dataset tree or a model/output artifact
# directory.  The v3 builder only accepts a fresh path.
_FORBIDDEN_OUTPUTS: tuple[Path, ...] = (
    _ROOT / "training" / "data",
    _ROOT / "training" / "artifacts",
    _ROOT / "model",
    _ROOT / "models",
    _ROOT / "output",
    _ROOT / "outputs",
    _ROOT / "artifacts",
)

_DRAFT_NOTE = (
    "Draft v3 dataset. The six-scenario backend closure is still required before "
    "any release-readiness claim."
)


@dataclass
class Row:
    """One dataset row with its decoded metadata for grouping."""

    messages: list[dict[str, str]]
    meta: dict[str, Any]
    source: str


# ---------------------------------------------------------------------------
# Reading / hashing
# ---------------------------------------------------------------------------


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _load_split_rules() -> dict[str, Any]:
    with _SCENARIOS_PATH.open(encoding="utf-8") as fh:
        return dict(yaml.safe_load(fh).get("split_rules") or {})


def _meta_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _assistant_json(messages: list[dict[str, str]]) -> dict[str, Any]:
    for message in messages:
        if message.get("role") == "assistant":
            try:
                parsed = json.loads(message.get("content") or "{}")
            except (json.JSONDecodeError, TypeError, ValueError):
                return {}
            return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _tools(messages: list[dict[str, str]]) -> set[str]:
    output = _assistant_json(messages)
    if output.get("kind") != "plan":
        return set()
    return {
        str(step.get("tool"))
        for step in output.get("steps", [])
        if isinstance(step, dict) and step.get("tool")
    }


def _present(meta: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key not in meta:
            continue
        value = meta[key]
        if value is None or value == "":
            continue
        return str(value)
    return ""


def _scenario(meta: dict[str, Any]) -> str:
    return _present(meta, "scenario", "scenario_id")


def _family_key(meta: dict[str, Any]) -> str:
    explicit = meta.get("family")
    if isinstance(explicit, str) and explicit:
        return explicit
    parts = [
        _scenario(meta),
        _present(meta, "persona", "persona_id"),
        _present(meta, "bankseed", "bank_seed"),
        _present(meta, "slotseed", "sample_seed"),
    ]
    if all(parts):
        return "|".join(parts)
    fallback = _present(meta, "legacy_prompt_hash")
    if fallback:
        return fallback
    serialised = json.dumps(meta, ensure_ascii=False, sort_keys=True, default=str)
    return "meta|" + _sha256_bytes(serialised.encode("utf-8"))


# ---------------------------------------------------------------------------
# Legacy replay
# ---------------------------------------------------------------------------


def _replay_legacy(
    legacy: Path,
    drops: dict[str, Counter[str]],
    source_hashes: dict[str, str],
) -> list[Row]:
    rules = _load_split_rules()
    test_personas = set(rules.get("test_only_personas") or [])
    test_variations = set(rules.get("test_only_variations") or [])
    rows: list[Row] = []

    for split in _SPLITS:
        path = legacy / f"{split}.jsonl"
        if not path.is_file():
            continue
        source_hashes[f"legacy_{split}"] = _sha256_file(path)

        for record in _read_jsonl(path):
            messages = record.get("messages") or []
            if not isinstance(messages, list):
                drops["legacy_unusable"][split] += 1
                continue
            by_role = {
                message.get("role"): str(message.get("content") or "")
                for message in messages
                if isinstance(message, dict)
            }
            user = by_role.get("user")
            assistant = by_role.get("assistant")
            if not user or not assistant:
                drops["legacy_unusable"][split] += 1
                continue

            meta = _meta_dict(record.get("meta"))
            scenario = _scenario(meta)
            if scenario == "B6_bill_pay_unsupported":
                # Obsolete: v3 has real pay_bill coverage, so the old "refuse
                # all bill payments" rows must not teach the model to refuse.
                drops["legacy_b6_bill_pay_unsupported"][split] += 1
                continue

            if scenario == "O1_birthday":
                output = json.loads(assistant)
                pots = [s for s in output.get("steps", []) if s.get("tool") == "move_to_pot"]
                if len(pots) != 1:
                    raise ValueError("Birthday gold must reserve one funding pot")
                for step in output["steps"]:
                    if step.get("tool") == "merchant_order":
                        step["args"]["funding_pot"] = f"${pots[0]['id']}.pot"
                assistant = json.dumps(output, ensure_ascii=False, sort_keys=True)

            persona = str(meta.get("persona") or meta.get("persona_id") or "")
            variation = str(meta.get("variation") or "")
            held_out = bool(meta.get("held_out")) or split == "test"
            held_out = held_out or persona in test_personas
            held_out = held_out or variation in test_variations

            meta.pop("known_tokens", None)
            meta.pop("_runtime", None)
            meta["source"] = "legacy"
            meta["legacy_split"] = split
            meta["held_out"] = held_out
            meta.setdefault("scenario", scenario)
            meta.setdefault("persona", persona)
            meta.setdefault("bankseed", meta.get("bank_seed"))
            meta.setdefault("slotseed", meta.get("sample_seed"))
            meta["legacy_prompt_hash"] = "legacy-prompt|" + _sha256_bytes(
                user.encode("utf-8")
            )
            meta["family"] = _family_key(meta)
            meta.setdefault("lang", "unknown")
            if not meta.get("kind"):
                meta["kind"] = str(_assistant_json(messages).get("kind") or "")

            rows.append(
                Row(
                    messages=[
                        {"role": "system", "content": R3_SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ],
                    meta=meta,
                    source="legacy",
                )
            )
    return rows


# ---------------------------------------------------------------------------
# New v3 rows
# ---------------------------------------------------------------------------


def _new_rows(seed: int, today: date, per_intent: int) -> list[Row]:
    rows: list[Row] = []
    for record in generate_samples(seed, today, per_intent):
        meta = _meta_dict(record.get("meta"))
        meta.setdefault("held_out", False)
        rows.append(
            Row(
                messages=list(record["messages"]),
                meta=meta,
                source="v3",
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Deterministic grouped split
# ---------------------------------------------------------------------------


def _assign_splits(
    rows: list[Row],
    seed: int,
    forced_test: set[str],
    preferred: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[_family_key(row.meta)].append(row)

    by_tool: dict[str, list[str]] = defaultdict(list)
    for family, family_rows in groups.items():
        tools: set[str] = set()
        for row in family_rows:
            tools.update(_tools(row.messages))
            tools.update(str(tool) for tool in row.meta.get("tools", []) or [])
        for tool in tools:
            by_tool[tool].append(family)

    reserved_test: set[str] = set()
    for tool in sorted(REQUIRED_POSITIVE_TOOLS):
        candidates = sorted(by_tool.get(tool, []))
        if not candidates:
            raise SystemExit(f"v3 generation has no family for required tool {tool}")
        # One deterministic family per tool goes to test, so every new tool is
        # represented in the held-out split without letting siblings cross.
        fresh = [family for family in candidates if family not in reserved_test]
        pool = fresh or candidates
        index = int(
            hashlib.sha256(f"split-test|{seed}|{tool}".encode("utf-8")).hexdigest()[:8],
            16,
        ) % len(pool)
        reserved_test.add(pool[index])

    reserved_train: set[str] = set()
    for tool in sorted(REQUIRED_POSITIVE_TOOLS):
        candidates = sorted(by_tool.get(tool, []))
        fresh = [
            family
            for family in candidates
            if family not in reserved_test and family not in reserved_train
        ]
        if not fresh:
            fresh = [family for family in candidates if family not in reserved_test]
        if not fresh:
            continue
        index = int(
            hashlib.sha256(f"split-train|{seed}|{tool}".encode("utf-8")).hexdigest()[:8],
            16,
        ) % len(fresh)
        reserved_train.add(fresh[index])

    split_of: dict[str, str] = {}
    for family in groups:
        if family in forced_test or family in reserved_test:
            split_of[family] = "test"
        elif family in reserved_train:
            split_of[family] = "train"
        elif preferred is not None and family in preferred:
            split_of[family] = preferred[family]
        else:
            # No extra test bucket here: test coverage already comes from the
            # one reserved family per required tool, which keeps the new-data
            # split close to 80/10/10 instead of drifting to ~20% test.
            bucket = int(
                hashlib.sha256(f"split|{seed}|{family}".encode("utf-8")).hexdigest()[:8],
                16,
            ) % 10
            split_of[family] = "dev" if bucket == 0 else "train"

    # Safety net: every required tool must have at least one training family.
    for tool in sorted(REQUIRED_POSITIVE_TOOLS):
        families = sorted(by_tool.get(tool, []))
        if any(split_of.get(family) == "train" for family in families):
            continue
        candidates = [
            family
            for family in families
            if family not in forced_test and family not in reserved_test
        ]
        if not candidates:
            candidates = [family for family in families if family not in forced_test]
        if candidates:
            split_of[candidates[0]] = "train"

    details = {
        "reserved_test_families": len(reserved_test),
        "reserved_train_families": len(reserved_train),
        "forced_test_families": len(forced_test),
        "strategy": (
            "family hash 90/10 train-dev plus one reserved test family per "
            "required tool; legacy test/held-out families are forced test"
        ),
    }
    return split_of, details


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------


def _dedupe(rows_by_split: dict[str, list[Row]], drops: dict[str, Counter[str]]) -> None:
    seen: dict[str, str] = {}
    # Test is kept first so a prompt that somehow appears in more than one
    # split never leaks from test into train/dev.
    for split in ("test", "dev", "train"):
        kept: list[Row] = []
        for row in rows_by_split[split]:
            user = next(
                (
                    message["content"]
                    for message in row.messages
                    if message.get("role") == "user"
                ),
                "",
            )
            if user in seen:
                drops["duplicate_prompt"][split] += 1
                continue
            seen[user] = split
            kept.append(row)
        rows_by_split[split] = kept


# ---------------------------------------------------------------------------
# Stats / output
# ---------------------------------------------------------------------------


def _kind_of(row: Row) -> str:
    return str(row.meta.get("kind") or _assistant_json(row.messages).get("kind") or "")


def _negative(row: Row) -> bool:
    if row.meta.get("negative") is not None:
        return bool(row.meta.get("negative"))
    return _kind_of(row) in {"clarify", "refuse", "scam"}


def _split_stats(rows: list[Row]) -> dict[str, Any]:
    tools: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    langs: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for row in rows:
        tools.update(_tools(row.messages))
        kinds[_kind_of(row) or "unknown"] += 1
        langs[str(row.meta.get("lang") or "unknown")] += 1
        sources[row.source] += 1
    return {
        "count": len(rows),
        "tools": dict(sorted(tools.items())),
        "kind": dict(sorted(kinds.items())),
        "lang": dict(sorted(langs.items())),
        "source": dict(sorted(sources.items())),
        "negative": sum(1 for row in rows if _negative(row)),
    }


def _dataset_info() -> dict[str, Any]:
    def entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }

    return {"anban_train": entry("train.jsonl"), "anban_dev": entry("dev.jsonl")}


def _write_jsonl(path: Path, rows: list[Row]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            record = {
                "messages": row.messages,
                "meta": json.dumps(
                    row.meta, ensure_ascii=False, sort_keys=True, default=str
                ),
            }
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def _validate_output_path(out: Path, legacy: Path | None) -> Path:
    out = Path(out).expanduser()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output path: {out}")
    resolved = out.resolve()
    for forbidden in _FORBIDDEN_OUTPUTS:
        forbidden_resolved = forbidden.resolve()
        if resolved == forbidden_resolved or forbidden_resolved in resolved.parents:
            raise SystemExit(
                f"refusing to write into legacy/artifact path: {resolved}"
            )
    if legacy is not None:
        legacy_resolved = Path(legacy).expanduser().resolve()
        if resolved == legacy_resolved or legacy_resolved in resolved.parents:
            raise SystemExit(f"refusing to write into the legacy source: {resolved}")
    return out


def _stage_and_commit(
    out: Path,
    rows_by_split: dict[str, list[Row]],
    dataset_info: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=str(out.parent))
    )
    try:
        for split in _SPLITS:
            _write_jsonl(staging / f"{split}.jsonl", rows_by_split[split])
        _write_json(staging / "dataset_info.json", dataset_info)
        stats["artifact_sha256"] = {
            name: _sha256_file(staging / name)
            for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "dataset_info.json")
        }
        _write_json(staging / "stats.json", stats)

        # Re-check immediately before the rename: another process may have
        # created the path while this build was running.
        if out.exists():
            raise SystemExit(f"refusing to overwrite existing output path: {out}")
        os.replace(staging, out)
    except BaseException:
        # Preserve even incomplete staging output for inspection and recovery.
        raise


# ---------------------------------------------------------------------------
# Public build
# ---------------------------------------------------------------------------


def build(
    seed: int,
    today: date,
    out: Path,
    per_intent: int = 1,
    legacy: Path | None = _DEFAULT_LEGACY,
) -> dict[str, Any]:
    """Build the v3 draft dataset into a brand-new ``out`` directory."""

    if isinstance(today, datetime):
        today = today.date()

    # The output contract is checked before any generation or legacy read.
    out = _validate_output_path(Path(out), legacy)
    if per_intent < 1:
        raise ValueError("per_intent must be at least 1")

    drops: dict[str, Counter[str]] = {
        "legacy_b6_bill_pay_unsupported": Counter(),
        "legacy_unusable": Counter(),
        "duplicate_prompt": Counter(),
    }
    source_hashes: dict[str, str] = {}
    for relative in ("training/v3_samples.py", "training/build_v3_dataset.py",
                     "agent/prompts.py", "agent/tools.py", "security/tokens.py"):
        source_hashes[relative] = _sha256_file(_ROOT / relative)
    for name in ("scenarios.yaml", "customer_segments.yaml"):
        path = _ROOT / "data_spec" / name
        if path.is_file():
            source_hashes[name] = _sha256_file(path)

    legacy_rows: list[Row] = []
    legacy_path = Path(legacy).expanduser() if legacy is not None else None
    if legacy_path is not None and legacy_path.is_dir():
        legacy_rows = _replay_legacy(legacy_path, drops, source_hashes)
    elif legacy_path is not None:
        source_hashes["legacy_missing"] = _sha256_bytes(
            f"missing:{legacy_path}".encode("utf-8")
        )

    new_rows = _new_rows(seed, today, per_intent)
    rows = legacy_rows + new_rows

    forced_test = {
        _family_key(row.meta)
        for row in legacy_rows
        if row.meta.get("held_out")
        or row.meta.get("legacy_split") == "test"
    }
    # Replay keeps each legacy train/dev *family* in its original side; only
    # held-out/test families are forced to test.
    legacy_preferred: dict[str, str] = {}
    for row in legacy_rows:
        split = str(row.meta.get("legacy_split") or "")
        if split not in {"train", "dev"}:
            continue
        family = _family_key(row.meta)
        if split == "dev":
            legacy_preferred[family] = "dev"
        else:
            legacy_preferred.setdefault(family, "train")
    split_of, split_details = _assign_splits(
        rows, seed, forced_test, preferred=legacy_preferred
    )

    rows_by_split: dict[str, list[Row]] = {split: [] for split in _SPLITS}
    for row in rows:
        rows_by_split[split_of[_family_key(row.meta)]].append(row)

    _dedupe(rows_by_split, drops)

    # Deterministic shuffle so any prefix of the file still spans intents.
    for index, split in enumerate(_SPLITS):
        random.Random(seed + 10_000 + index).shuffle(rows_by_split[split])

    totals = {split: len(rows_by_split[split]) for split in _SPLITS}
    all_rows = [row for split in _SPLITS for row in rows_by_split[split]]

    # Re-validate every gold target against the labels visible in its own
    # prompt.  The regex intentionally matches a label inside a tagged context
    # line too (``<CARD_1 | physical | active>``), not just a bare label.
    for row in all_rows:
        user = next(m["content"] for m in row.messages if m["role"] == "user")
        assistant = next(
            m["content"] for m in row.messages if m["role"] == "assistant"
        )
        known = {
            f"<{label}>": label.split("_", 1)[0]
            for label in re.findall(r"<([A-Z]+_[A-Za-z0-9]+)", user)
        }
        validate_plan(parse_planner_output(assistant), known)
    negatives = sum(1 for row in all_rows if _negative(row))
    v3_negatives = sum(1 for row in new_rows if _negative(row))

    by_tool = dict(
        sorted(
            Counter(
                tool for row in all_rows for tool in _tools(row.messages)
            ).items()
        )
    )
    by_kind = dict(
        sorted(Counter(_kind_of(row) or "unknown" for row in all_rows).items())
    )
    by_lang = dict(
        sorted(
            Counter(str(row.meta.get("lang") or "unknown") for row in all_rows).items()
        )
    )
    by_source = dict(sorted(Counter(row.source for row in all_rows).items()))
    prompt_hash = _sha256_bytes(R3_SYSTEM_PROMPT.encode("utf-8"))
    drop_counts = {
        name: dict(sorted(counter.items())) for name, counter in drops.items()
    }

    stats: dict[str, Any] = {
        "draft": True,
        "release_ready": False,
        "draft_note": _DRAFT_NOTE,
        "seed": seed,
        "today": today.isoformat(),
        "per_intent": per_intent,
        "totals": totals,
        # ``negative_ratio`` describes the newly generated v3 rows; the
        # combined legacy+v3 figure is kept separately below.
        "negative": v3_negatives,
        "negative_ratio": round(v3_negatives / len(new_rows), 4) if new_rows else 0.0,
        "v3_negative_ratio": round(v3_negatives / len(new_rows), 4)
        if new_rows
        else 0.0,
        "combined_negative": negatives,
        "combined_negative_ratio": round(negatives / len(all_rows), 4)
        if all_rows
        else 0.0,
        "splits": {
            split: _split_stats(rows_by_split[split]) for split in _SPLITS
        },
        "coverage": {
            "by_tool": by_tool,
            "by_kind": by_kind,
            "by_lang": by_lang,
            "by_source": by_source,
            # Compact aliases for tooling that keys on the shorter names.
            "tool": by_tool,
            "kind": by_kind,
            "lang": by_lang,
            "source": by_source,
        },
        "prompt_sha256": prompt_hash,
        "promptSHA256": prompt_hash,
        "source_hashes": dict(sorted(source_hashes.items())),
        "sourcehashes": dict(sorted(source_hashes.items())),
        "drop_counts": drop_counts,
        "dropcounts": drop_counts,
        "legacy": {
            "b6_bill_pay_unsupported_dropped": sum(
                drops["legacy_b6_bill_pay_unsupported"].values()
            ),
            "b6_by_split": dict(
                sorted(drops["legacy_b6_bill_pay_unsupported"].items())
            ),
        },
        "split_strategy": split_details,
    }

    _stage_and_commit(out, rows_by_split, _dataset_info(), stats)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--today", type=date.fromisoformat, required=True)
    parser.add_argument("--per-intent", type=int, default=1)
    parser.add_argument("--legacy", type=Path, default=_DEFAULT_LEGACY)
    parser.add_argument("--no-legacy", action="store_true")
    args = parser.parse_args(argv)

    stats = build(
        args.seed,
        args.today,
        args.out,
        per_intent=args.per_intent,
        legacy=None if args.no_legacy else args.legacy,
    )
    print(json.dumps(stats["totals"], ensure_ascii=False, sort_keys=True))
    print(json.dumps(stats["coverage"], ensure_ascii=False, sort_keys=True))
    print("draft: " + json.dumps(
        {"draft": stats["draft"], "draft_note": stats["draft_note"]},
        ensure_ascii=False,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
