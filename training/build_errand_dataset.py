"""Build the errand-agent training data (docs/ERRAND_AGENT.md).

Every record is a full gold trajectory replayed through ``run_errand`` with
``OracleModel``. A scenario that does not end exactly as its oracle predicted
aborts the whole build, so a bad record can never reach a dataset file.

The output directory must be new: this builder never overwrites or deletes an
existing path. Files are written only after every split has been replayed,
validated and checked for cross-split trajectory leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from errand.agent import OracleModel, RunResult, run_errand, scenario_ticket
from errand.oracle import gold_actions
from errand.prompts import ERRAND_SYSTEM_PROMPT
from errand.scenarios import Scenario, generate_scenario

_SPLITS = ("train", "dev", "test")

_DATASET_INFO = {
    "errand_train": {
        "file_name": "train.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    },
    "errand_dev": {
        "file_name": "dev.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    },
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seeds(split: str, count: int) -> range:
    if split == "train":
        return range(0, count)
    if split == "dev":
        return range(1_000_000, 1_000_000 + count)
    if split == "test":
        return range(2_000_000, 2_000_000 + count)
    raise ValueError(f"unknown split: {split}")


def _check_result(scenario: Scenario, result: RunResult) -> None:
    if result.refusals:
        raise RuntimeError(
            f"seed {scenario.seed}: unexpected guard refusals {result.refusals}"
        )
    kind, _, value = scenario.expected.partition(":")
    if kind == "buy":
        if (
            result.status != "bought"
            or result.order is None
            or result.order.product_id != value
        ):
            raise RuntimeError(
                f"seed {scenario.seed}: expected {scenario.expected}, "
                f"got {result.status} {result.order!r}"
            )
    elif kind == "abort":
        if result.status != "aborted" or result.abort_reason != value:
            raise RuntimeError(
                f"seed {scenario.seed}: expected {scenario.expected}, "
                f"got {result.status} {result.abort_reason!r}"
            )
    else:
        raise RuntimeError(f"seed {scenario.seed}: bad expected {scenario.expected!r}")


def _check_messages(messages: list[dict], seed: int) -> None:
    if not messages or messages[0].get("role") != "system":
        raise RuntimeError(f"seed {seed}: transcript does not start with system")
    expected_roles = ("user", "assistant")
    for index, message in enumerate(messages[1:]):
        if message.get("role") != expected_roles[index % 2]:
            raise RuntimeError(
                f"seed {seed}: role {message.get('role')!r} at {index + 1}, "
                f"expected {expected_roles[index % 2]!r}"
            )
    if messages[-1].get("role") != "assistant":
        raise RuntimeError(f"seed {seed}: transcript does not end with assistant")
    for message in messages:
        if message.get("role") != "assistant":
            continue
        try:
            content = json.loads(message["content"])
        except (ValueError, TypeError):
            raise RuntimeError(f"seed {seed}: assistant turn is not JSON") from None
        if not isinstance(content, dict):
            raise RuntimeError(f"seed {seed}: assistant turn is not a JSON object")


def _record(seed: int, split: str) -> tuple[dict, Scenario, RunResult]:
    scenario = generate_scenario(seed)
    ticket, shop = scenario_ticket(scenario)
    result = run_errand(
        ticket,
        shop,
        OracleModel(gold_actions(scenario)),
        scenario.today,
    )
    _check_result(scenario, result)
    _check_messages(result.transcript, seed)
    meta = {
        "seed": seed,
        "split": split,
        "expected": scenario.expected,
        "tags": list(scenario.tags),
        "steps": result.steps,
    }
    record = {
        "messages": result.transcript,
        "meta": json.dumps(meta, ensure_ascii=False, sort_keys=True),
    }
    return record, scenario, result


def _trajectory_sha256(messages: list[dict]) -> str:
    payload = json.dumps(messages[1:], ensure_ascii=False, sort_keys=True)
    return _sha256_bytes(payload.encode("utf-8"))


def build(out: Path, n_train: int, n_dev: int, n_test: int) -> dict:
    out = Path(out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing path: {out}")

    counts = {"train": n_train, "dev": n_dev, "test": n_test}
    records: dict[str, list[dict]] = {}
    trajectory_hashes: dict[str, set[str]] = {}
    expected_kinds: dict[str, Counter[str]] = {}
    tag_counts: dict[str, Counter[str]] = {}
    max_chars = 0
    assistant_turns = 0
    total_records = 0

    for split in _SPLITS:
        split_records: list[dict] = []
        split_hashes: set[str] = set()
        kinds: Counter[str] = Counter()
        tags: Counter[str] = Counter()
        for seed in _seeds(split, counts[split]):
            record, scenario, result = _record(seed, split)
            split_records.append(record)
            split_hashes.add(_trajectory_sha256(result.transcript))

            kind = "buy" if scenario.expected.startswith("buy:") else scenario.expected
            kinds[kind] += 1
            tags.update(scenario.tags)

            chars = sum(len(message["content"]) for message in result.transcript)
            max_chars = max(max_chars, chars)
            assistant_turns += sum(
                1
                for message in result.transcript
                if message["role"] == "assistant"
            )
            total_records += 1
        records[split] = split_records
        trajectory_hashes[split] = split_hashes
        expected_kinds[split] = kinds
        tag_counts[split] = tags

    for index, first in enumerate(_SPLITS):
        for second in _SPLITS[index + 1 :]:
            shared = trajectory_hashes[first] & trajectory_hashes[second]
            if shared:
                raise RuntimeError(
                    f"trajectory leakage between {first} and {second}: "
                    f"{len(shared)} shared record(s)"
                )

    os.makedirs(out, exist_ok=False)

    artifact_sha256: dict[str, str] = {}
    for split in _SPLITS:
        path = out / f"{split}.jsonl"
        payload = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records[split]
        ).encode("utf-8")
        path.write_bytes(payload)
        artifact_sha256[path.name] = _sha256_bytes(payload)

    info_payload = (
        json.dumps(_DATASET_INFO, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (out / "dataset_info.json").write_bytes(info_payload)
    artifact_sha256["dataset_info.json"] = _sha256_bytes(info_payload)

    stats = {
        "counts": {split: len(records[split]) for split in _SPLITS},
        "expected_kinds": {
            split: dict(sorted(expected_kinds[split].items())) for split in _SPLITS
        },
        "tags": {
            split: dict(sorted(tag_counts[split].items())) for split in _SPLITS
        },
        "max_chars": max_chars,
        "mean_assistant_turns": (
            assistant_turns / total_records if total_records else 0.0
        ),
        "prompt_sha256": _sha256_bytes(ERRAND_SYSTEM_PROMPT.encode("utf-8")),
        "artifact_sha256": artifact_sha256,
    }
    (out / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the errand-agent gold training dataset."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--dev", type=int, default=300)
    parser.add_argument("--test", type=int, default=600)
    args = parser.parse_args(argv)

    stats = build(args.out, args.train, args.dev, args.test)
    total = sum(stats["counts"].values())
    print(
        f"Wrote {total} errand records to {args.out} "
        f"(max_chars={stats['max_chars']}, "
        f"mean_assistant_turns={stats['mean_assistant_turns']:.2f})"
    )
    for split in _SPLITS:
        print(
            f"  {split}: {stats['counts'][split]} records, "
            f"kinds={stats['expected_kinds'][split]}"
        )


if __name__ == "__main__":
    main()
