"""Builds ChatML/sharegpt JSONL training data with code-generated gold labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from bank.money import fmt_fen
from bank.seed import generate
from training.gold import TeacherMismatch, make_sample, render_seed
from training.templates import supported_variations

_ROOT = Path(__file__).resolve().parents[1]
_SCENARIOS_PATH = _ROOT / "data_spec" / "scenarios.yaml"
_SEGMENTS_PATH = _ROOT / "data_spec" / "customer_segments.yaml"

_SCAM_WEIGHTS = {"retiree": 0.5, "young_parent": 0.25, "student": 0.25}
_MAX_TEST_PER_SCENARIO = 15


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _personas() -> list[dict[str, Any]]:
    return list(_load_yaml(_SEGMENTS_PATH)["segments"])


def _scenarios() -> list[dict[str, Any]]:
    return list(_load_yaml(_SCENARIOS_PATH)["scenarios"])


def _split_rules() -> dict[str, Any]:
    return dict(_load_yaml(_SCENARIOS_PATH).get("split_rules") or {})


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _persona_pool(
    personas: list[dict[str, Any]], group: str
) -> tuple[list[dict[str, Any]], list[float]]:
    """Personas and weights for one scenario group."""

    if group == "scam":
        pool = [
            (persona, _SCAM_WEIGHTS[persona["id"]])
            for persona in personas
            if persona["id"] in _SCAM_WEIGHTS
        ]
    elif group in {"attack", "refuse"}:
        pool = [(persona, 1.0) for persona in personas]
    else:
        pool = []
        for persona in personas:
            weights = persona.get("scenario_weights") or {}
            weight = float(weights.get(group, 0.02))
            if weight > 0:
                pool.append((persona, weight))
    if not pool:
        pool = [(persona, 1.0) for persona in personas]
    return [persona for persona, _ in pool], [weight for _, weight in pool]


def _pick_persona(
    rng: random.Random, personas: list[dict[str, Any]], group: str
) -> dict[str, Any]:
    population, weights = _persona_pool(personas, group)
    return rng.choices(population, weights=weights, k=1)[0]


def _pick_lang(rng: random.Random, persona: dict[str, Any]) -> str:
    language = persona.get("language") or {"zh": 1.0}
    pool = [(lang, float(weight)) for lang, weight in language.items() if weight > 0]
    if not pool:
        return "zh"
    return rng.choices(
        [lang for lang, _ in pool], weights=[weight for _, weight in pool], k=1
    )[0]


def _state_for(
    persona_id: str,
    bank_seed: int,
    today: date,
    cache: dict[tuple[str, int], Any],
) -> Any:
    key = (persona_id, bank_seed)
    if key not in cache:
        cache[key] = generate(persona_id, bank_seed, today)
    return cache[key]


def _try_sample(
    scenario_id: str,
    persona: dict[str, Any],
    variation: str,
    rng: random.Random,
    seed: int,
    today: date,
    sample_seed: int,
    state_cache: dict[tuple[str, int], Any] | None = None,
    leak_log: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    lang = _pick_lang(rng, persona)
    try:
        sample = make_sample(
            scenario_id,
            persona["id"],
            lang,
            variation,
            seed,
            today,
            sample_seed=sample_seed,
        )
    except Exception as exc:  # noqa: BLE001 - reasons are reported to the caller
        return None, f"{type(exc).__name__}: {str(exc)[:80]}"

    if state_cache is not None:
        meta = sample["meta"]
        state = _state_for(meta["persona_id"], meta["bank_seed"], today, state_cache)
        leaks = check_leakage(sample["user"], state)
        if leaks:
            if leak_log is not None:
                text = sample["user"]
                leak_log.append(
                    {
                        "scenario_id": scenario_id,
                        "persona_id": persona["id"],
                        "lang": lang,
                        "variation": variation,
                        "sample_seed": sample_seed,
                        "leaked": leaks,
                        "text": text,
                        "user_text": text.rsplit("User: ", 1)[-1],
                    }
                )
            return None, f"Leakage: {', '.join(leaks)}"
    return sample, None


def _extra_sample(
    scenario_id: str,
    personas: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    working: list[dict[str, Any]],
    rng: random.Random,
    seed: int,
    today: date,
    sample_seed: int,
    state_cache: dict[tuple[str, int], Any],
    leak_log: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """One extra test sample: expat first, then a persona that supports the scenario."""

    expat = by_id.get("expat")
    variations = supported_variations(scenario_id)
    reason: str | None = None

    # Half of the held-out set is the unseen polite_long style from the
    # zh-first personas, so the test split reflects the domestic track and
    # not only the English-speaking expat.
    zh_pool = [p for p in (working or personas) if p.get("id") != "expat"]
    if zh_pool and rng.random() < 0.5:
        sample, reason = _try_sample(
            scenario_id,
            rng.choice(zh_pool),
            "polite_long",
            rng,
            seed,
            today,
            sample_seed,
            state_cache,
            leak_log,
        )
        if sample is not None:
            return sample, None

    if expat is not None:
        sample, reason = _try_sample(
            scenario_id,
            expat,
            rng.choice(variations),
            rng,
            seed,
            today,
            sample_seed,
            state_cache,
            leak_log,
        )
        if sample is not None:
            return sample, None
        sample, reason = _try_sample(
            scenario_id,
            expat,
            "polite_long",
            rng,
            seed,
            today,
            sample_seed,
            state_cache,
            leak_log,
        )
        if sample is not None:
            return sample, None
    else:
        reason = "KeyError: expat"

    # Some scenarios are impossible for expat (T2 needs a duplicate contact
    # name that only young_pro has).  Reuse the personas that already produced
    # valid samples; extras are added to test directly.
    pool = working or personas
    for _ in range(10):
        persona = rng.choice(pool)
        sample, fallback_reason = _try_sample(
            scenario_id,
            persona,
            rng.choice(variations),
            rng,
            seed,
            today,
            sample_seed,
            state_cache,
            leak_log,
        )
        if sample is not None:
            return sample, None
        reason = fallback_reason or reason
    return None, reason


def _teacher_samples(
    scenario_id: str,
    group: str,
    train_personas: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    rng: random.Random,
    seed: int,
    today: date,
    teacher: Any,
    teacher_seeds: int,
    teacher_n: int,
    next_sample_seed: Any,
    state_cache: dict[tuple[str, int], Any],
    leak_log: list[dict[str, Any]],
    drops: dict[str, Counter[str]],
) -> list[dict[str, Any]]:
    """Paraphrase seed utterances and keep the ones that survive checks.

    Gold is always rebuilt from the template, so a paraphrase only contributes
    wording.  Extra expat seeds reach the test split through the normal
    ``_split_for`` rule.
    """

    expat = by_id.get("expat")
    # (persona, held_out): held-out seeds go to test only. Half are the
    # expat persona, half zh-first personas, so the test split keeps the
    # domestic language mix while its phrasing stays unseen in train.
    targets: list[tuple[dict[str, Any], bool]] = [
        (_pick_persona(rng, train_personas, group), False) for _ in range(teacher_seeds)
    ]
    held = math.ceil(teacher_seeds / 6)
    if expat is not None:
        targets.extend([(expat, True)] * held)
    targets.extend((_pick_persona(rng, train_personas, group), True) for _ in range(held))

    planned: list[tuple[dict[str, Any], str, int, str, list[str], bool]] = []
    for persona, held_out in targets:
        sample_seed = next_sample_seed()
        lang = _pick_lang(rng, persona)
        try:
            seed_text, must_keep = render_seed(
                scenario_id,
                persona["id"],
                lang,
                "explicit",
                seed,
                today,
                sample_seed=sample_seed,
            )
        except Exception:  # noqa: BLE001 - teacher seeds are best-effort
            drops["teacher"]["other"] += 1
            continue
        planned.append((persona, lang, sample_seed, seed_text, must_keep, held_out))

    if not planned:
        return []

    jobs = [
        {
            "seed_text": seed_text,
            "persona": persona,
            "lang": lang,
            "n": teacher_n,
            "must_keep": must_keep,
        }
        for persona, lang, _sample_seed, seed_text, must_keep, _held in planned
    ]
    variants_by_job = teacher.paraphrase_many(jobs)

    samples: list[dict[str, Any]] = []
    for (persona, lang, sample_seed, _seed_text, _must_keep, held_out), variants in zip(
        planned, variants_by_job
    ):
        for variant in variants:
            try:
                sample = make_sample(
                    scenario_id,
                    persona["id"],
                    lang,
                    "explicit",
                    seed,
                    today,
                    sample_seed=sample_seed,
                    text_override=variant,
                )
            except TeacherMismatch:
                drops["teacher"]["TeacherMismatch"] += 1
                continue
            except Exception:  # noqa: BLE001 - teacher variants are best-effort
                drops["teacher"]["other"] += 1
                continue

            meta = sample["meta"]
            if held_out:
                meta["held_out"] = True
            state = _state_for(
                meta["persona_id"], meta["bank_seed"], today, state_cache
            )
            leaks = check_leakage(sample["user"], state)
            if leaks:
                drops["teacher"]["leakage"] += 1
                text = sample["user"]
                leak_log.append(
                    {
                        "scenario_id": scenario_id,
                        "persona_id": persona["id"],
                        "lang": lang,
                        "variation": "teacher",
                        "sample_seed": sample_seed,
                        "leaked": leaks,
                        "text": text,
                        "user_text": text.rsplit("User: ", 1)[-1],
                    }
                )
                continue
            samples.append(sample)
    return samples


# ---------------------------------------------------------------------------
# Splitting / dedupe / leakage
# ---------------------------------------------------------------------------


def _split_for(
    sample: dict[str, Any],
    test_personas: set[str],
    test_variations: set[str],
) -> str:
    meta = sample["meta"]
    if (
        meta.get("held_out")
        or meta["persona_id"] in test_personas
        or meta["variation"] in test_variations
    ):
        return "test"
    # Paraphrases of one source must not leak across train and development.
    family = (
        json.dumps([meta.get(key) for key in (
            "scenario_id", "persona_id", "seed", "bank_seed", "sample_seed"
        )], ensure_ascii=False)
        if meta.get("sample_seed") is not None else sample["user"]
    )
    digest = hashlib.sha256(family.encode("utf-8")).digest()
    return "dev" if digest[0] < 26 else "train"


def _dedupe(
    samples: list[dict[str, Any]], split: str, drops: dict[str, Counter]
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    kept: list[dict[str, Any]] = []
    for sample in samples:
        key = (sample["user"], sample["assistant"])
        if key in seen:
            drops["duplicate"][split] += 1
            continue
        seen.add(key)
        kept.append(sample)
    return kept


def check_leakage(text: str, state: Any) -> list[str]:
    """Return raw values from the seeded bank state that leaked into ``text``."""

    leaks: list[str] = []

    def add(value: Any) -> None:
        value = str(value)
        if value and value not in leaks:
            leaks.append(value)

    digits = re.sub(r"\D", "", text)
    contacts = list(getattr(state, "contacts", []) or [])
    for contact in contacts:
        name = str(contact.name)
        if len(name) >= 2 and name in text:
            add(name)
        for part in re.split(r"[\s,，、/()（）]+", name):
            if len(part) >= 2 and part in text:
                add(part)
        if contact.phone and contact.phone in text:
            add(contact.phone)
        if contact.account_no and contact.account_no in digits:
            add(contact.account_no)

    customer = getattr(state, "customer", None)
    if customer is not None and customer.phone and customer.phone in text:
        add(customer.phone)

    current = next(
        (account for account in state.accounts if account.type == "current"), None
    )
    if current is not None:
        balance_text = fmt_fen(int(current.balance_fen))
        if balance_text in text:
            add(balance_text)
    return leaks


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _record(sample: dict[str, Any]) -> dict[str, Any]:
    meta = dict(sample["meta"])
    meta.pop("known_tokens", None)
    meta.pop("_runtime", None)
    return {
        "messages": [
            {"role": "system", "content": sample["system"]},
            {"role": "user", "content": sample["user"]},
            {"role": "assistant", "content": sample["assistant"]},
        ],
        # A JSON string, not an object: pyarrow (LLaMA-Factory's loader) infers one
        # schema from the first rows and crashes when later rows add meta keys.
        "meta": json.dumps(meta, ensure_ascii=False, sort_keys=True, default=str),
    }


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a split file, decoding the JSON-string ``meta`` back into a dict."""

    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if isinstance(row.get("meta"), str):
                    row["meta"] = json.loads(row["meta"])
                rows.append(row)
    return rows


def _write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            line = json.dumps(_record(sample), ensure_ascii=False, sort_keys=True)
            fh.write(line + "\n")


def _split_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def tally(key: str) -> dict[str, int]:
        counts = Counter(str(sample["meta"].get(key)) for sample in samples)
        return dict(sorted(counts.items()))

    variation = tally("variation")
    return {
        "count": len(samples),
        "unique_user_texts": len({sample["user"] for sample in samples}),
        "scenario": tally("scenario_id"),
        "persona": tally("persona_id"),
        "lang": tally("lang"),
        "variation": variation,
        "teacher": variation.get("teacher", 0),
        "kind": tally("kind"),
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


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(
    seed: int,
    per_scenario: int,
    out: Path,
    today: date,
    min_test: int = 8,
    teacher: Any = None,
    teacher_seeds: int = 0,
    teacher_n: int = 6,
) -> dict[str, Any]:
    if isinstance(today, datetime):
        today = today.date()
    out = Path(out)

    scenarios = _scenarios()
    personas = _personas()
    by_id = {persona["id"]: persona for persona in personas}
    rules = _split_rules()
    test_personas = set(rules.get("test_only_personas") or [])
    test_variations = set(rules.get("test_only_variations") or [])
    train_personas = [p for p in personas if p["id"] not in test_personas]
    state_cache: dict[tuple[str, int], Any] = {}
    leak_log: list[dict[str, Any]] = []

    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    test_keys: dict[str, set[tuple[str, str]]] = {}
    expat_in_test: dict[str, int] = {}
    drops: dict[str, Counter[str]] = {
        "generation": Counter(),
        "extras": Counter(),
        "duplicate": Counter(),
        "test_overlap": Counter(),
        "test_cap": Counter(),
        "teacher": Counter(),
    }
    reasons_by_scenario: dict[str, Counter[str]] = {}

    for index, scenario in enumerate(scenarios):
        scenario_id = scenario["id"]
        group = str(scenario.get("group") or "")
        rng = random.Random(seed * 1000 + index)
        scenario_reasons = reasons_by_scenario.setdefault(scenario_id, Counter())
        generated: list[dict[str, Any]] = []
        working: dict[str, dict[str, Any]] = {}
        attempts = 0
        draw_index = 0
        limit = max(per_scenario * 20, per_scenario + 100)

        def next_sample_seed() -> int:
            nonlocal draw_index
            draw_index += 1
            return seed * 1_000_000 + index * 10_000 + draw_index

        # Some personas cannot express a scenario (e.g. only young_pro has the
        # duplicate contact needed by T2).  Retry failures, but keep recording
        # them and still enforce the 80% threshold below.
        while len(generated) < per_scenario and attempts < limit:
            attempts += 1
            sample_seed = next_sample_seed()
            # Held-out personas never come from the main loop; the test split is
            # filled by _extra_test_sample with a balanced expat / polite_long mix.
            persona = _pick_persona(rng, train_personas, group)
            variation = rng.choice(supported_variations(scenario_id))
            sample, reason = _try_sample(
                scenario_id,
                persona,
                variation,
                rng,
                seed,
                today,
                sample_seed,
                state_cache,
                leak_log,
            )
            if sample is None:
                drops["generation"][reason] += 1
                scenario_reasons[reason] += 1
                continue
            generated.append(sample)
            working[persona["id"]] = persona

        if len(generated) < per_scenario * 0.8:
            detail = (
                ", ".join(f"{reason} x{count}" for reason, count in sorted(scenario_reasons.items()))
                or "none"
            )
            raise SystemExit(
                f"scenario {scenario_id}: generated {len(generated)}/{per_scenario} "
                f"samples (below 80%); drop reasons: {detail}"
            )

        if teacher is not None and teacher_seeds > 0:
            generated.extend(
                _teacher_samples(
                    scenario_id,
                    group,
                    train_personas,
                    by_id,
                    rng,
                    seed,
                    today,
                    teacher,
                    teacher_seeds,
                    teacher_n,
                    next_sample_seed,
                    state_cache,
                    leak_log,
                    drops,
                )
            )

        for sample in generated:
            split = _split_for(sample, test_personas, test_variations)
            if split == "test":
                keys = test_keys.setdefault(scenario_id, set())
                key = (sample["user"], sample["assistant"])
                if key in keys:
                    drops["duplicate"]["test"] += 1
                    continue
                if len(keys) >= _MAX_TEST_PER_SCENARIO:
                    drops["test_cap"]["test"] += 1
                    continue
                # At most half of a scenario's test quota is expat, so the
                # held-out set stays zh-first like the domestic track.
                is_expat = sample["meta"]["persona_id"] == "expat"
                if is_expat and expat_in_test.get(scenario_id, 0) >= _MAX_TEST_PER_SCENARIO // 2:
                    drops["test_cap"]["expat"] += 1
                    continue
                if is_expat:
                    expat_in_test[scenario_id] = expat_in_test.get(scenario_id, 0) + 1
                keys.add(key)
                test.append(sample)
            elif split == "dev":
                dev.append(sample)
            else:
                train.append(sample)

        keys = test_keys.setdefault(scenario_id, set())
        attempts = 0
        limit = max(200, min_test * 40)
        while len(keys) < min_test and attempts < limit:
            attempts += 1
            sample_seed = next_sample_seed()
            sample, reason = _extra_sample(
                scenario_id,
                personas,
                by_id,
                list(working.values()),
                rng,
                seed,
                today,
                sample_seed,
                state_cache,
                leak_log,
            )
            if sample is None:
                drops["extras"][reason] += 1
                scenario_reasons[reason] += 1
                continue
            key = (sample["user"], sample["assistant"])
            if key in keys:
                drops["duplicate"]["test"] += 1
                continue
            keys.add(key)
            test.append(sample)

        if len(keys) < min_test:
            detail = (
                ", ".join(f"{reason} x{count}" for reason, count in sorted(scenario_reasons.items()))
                or "none"
            )
            raise SystemExit(
                f"scenario {scenario_id}: only {len(keys)}/{min_test} unique test "
                f"samples after {attempts} extra attempts; reasons: {detail}"
            )

    test = _dedupe(test, "test", drops)
    train = _dedupe(train, "train", drops)
    dev = _dedupe(dev, "dev", drops)

    # Compare the tokenized request, not the whole user message: the same
    # request with a different context block would still leak into test.
    def _utterance(sample: dict[str, Any]) -> str:
        return sample["meta"].get("tokenized") or sample["user"]

    test_users = {_utterance(sample) for sample in test}
    for name, samples in (("train", train), ("dev", dev)):
        kept = []
        for sample in samples:
            if _utterance(sample) in test_users:
                drops["test_overlap"][name] += 1
                continue
            kept.append(sample)
        if name == "train":
            train = kept
        else:
            dev = kept

    # Leakage safety net.  States were already cached while generating.
    leaks: list[str] = []
    for split, samples in (("train", train), ("dev", dev), ("test", test)):
        for sample in samples:
            meta = sample["meta"]
            state = _state_for(
                meta["persona_id"], meta["bank_seed"], today, state_cache
            )
            found = check_leakage(sample["user"], state)
            if found:
                leaks.append(
                    f"{split}/{meta['persona_id']}/{meta['scenario_id']}: "
                    + ", ".join(found)
                )
    if leaks:
        raise SystemExit(
            "leakage detected in generated user text:\n" + "\n".join(leaks[:20])
        )

    out.mkdir(parents=True, exist_ok=True)
    # Rows are generated scenario by scenario; shuffle (deterministically) so any
    # prefix of the file (e.g. LLaMA-Factory max_samples) covers all scenarios.
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(dev)
    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "dev.jsonl", dev)
    _write_jsonl(out / "test.jsonl", test)
    with (out / "dataset_info.json").open("w", encoding="utf-8") as fh:
        json.dump(_dataset_info(), fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    stats = {
        "seed": seed,
        "per_scenario": per_scenario,
        "today": today.isoformat(),
        "min_test_per_scenario": min_test,
        "totals": {"train": len(train), "dev": len(dev), "test": len(test)},
        "splits": {
            "train": _split_stats(train),
            "dev": _split_stats(dev),
            "test": _split_stats(test),
        },
        "drops": {name: dict(sorted(counter.items())) for name, counter in drops.items()},
        "leakage": leak_log,
    }
    with (out / "stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-scenario", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("training/data"))
    # Required: the dataset must not depend on the day it was built.
    parser.add_argument("--today", type=date.fromisoformat, required=True)
    parser.add_argument("--min-test-per-scenario", type=int, default=8)
    parser.add_argument("--teacher-seeds", type=int, default=0)
    parser.add_argument("--teacher-n", type=int, default=6)
    args = parser.parse_args(argv)

    teacher = None
    if args.teacher_seeds > 0:
        from training.gen_utterances import TeacherClient, TeacherUnavailable

        try:
            teacher = TeacherClient()
        except TeacherUnavailable as exc:
            print(f"teacher unavailable: {exc}")
            return 1

    try:
        stats = build(
            args.seed,
            args.per_scenario,
            args.out,
            args.today,
            args.min_test_per_scenario,
            teacher=teacher,
            teacher_seeds=args.teacher_seeds,
            teacher_n=args.teacher_n,
        )
    finally:
        if teacher is not None:
            teacher.close()

    print(json.dumps(stats["totals"], ensure_ascii=False, sort_keys=True))
    print(json.dumps(stats["drops"], ensure_ascii=False, sort_keys=True))
    if args.teacher_seeds > 0:
        summary = {
            "splits": {
                split: stats["splits"][split]["teacher"]
                for split in ("train", "dev", "test")
            },
            "drops": stats["drops"].get("teacher", {}),
        }
        print("teacher: " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
