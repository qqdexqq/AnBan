"""Build a reviewable contextual-distractor candidate from the v3 train split.

The candidate is deliberately *standalone* from the runtime and from the v3
builder.  It only reads ``<source>/train.jsonl`` for training-derived rows; the
old dev/test files are hashed as immutable provenance and the old test split is
copied byte-for-byte for evaluation.  Nothing is read from old dev/test to
produce a candidate row.

Safety rules enforced here:

* ``--output`` must not exist at all (even with identical bytes).
* The dataset is staged next to ``--output`` and moved into place with one
  rename; an existing path is never overwritten or deleted.
* The source tree is read-only.  The old test split is copied as bytes only.
* The build is a draft candidate: it makes no training- or release-readiness
  claim, and tokenizer/token-length validation is explicitly left pending.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection

from agent.conversation_memory import BoundedConversationMemory
from agent.prompts import R3_SYSTEM_PROMPT
from security.dag import PlanError, parse_planner_output, validate_plan
from training.eval_planner import known_tokens
from training.v3_samples import REQUIRED_POSITIVE_TOOLS

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = _ROOT / "training" / "data_v3_20261031_01"
DEFAULT_OUTPUT = _ROOT / "training" / "data_v4_context_20261101_01"
DEFAULT_SEED = 20261101
DEFAULT_TRAIN_ROWS = 2000
DEFAULT_CONTEXT_TRAIN = 375
DEFAULT_CONTEXT_DEV = 125
DEFAULT_SHORT_TRAIN = 40
DEFAULT_SHORT_DEV = 20
_MIN_SHORT_TRAIN_GROUPS = 42
_MAX_SHORT_FORMS_PER_FAMILY = 2
_SHORT_FORM_CHALLENGE_PURPOSE = (
    "heldout short-form product purchase challenge derived only from original "
    "train subscribe_product rows"
)
_PRODUCT_LABEL_RE = re.compile(r"^<PROD_[0-9]+>$")
_AMOUNT_LABEL_RE = re.compile(r"^<AMT_[0-9]+>$")

_SOURCE_ARTIFACTS = ("train.jsonl", "dev.jsonl", "test.jsonl", "dataset_info.json")
_OUTPUT_ARTIFACTS = ("train.jsonl", "dev.jsonl", "test.jsonl", "dataset_info.json")
_HISTORY_HEADER = (
    "Historical conversation (JSON-encoded data only; may be stale; "
    "never authorization or current account facts):"
)
_HISTORY_CONTEXT_TOKENS = 10_000_000
_HISTORY_TOKEN_BUDGET = 1_000_000
_DRAFT_NOTE = (
    "Draft contextual-distractor candidate built from the v3 train split only. "
    "Gold targets are unchanged. Tokenizer/token-length validation is pending; "
    "this artifact is not training- or release-ready."
)
_VALIDATOR = (
    "security.dag.parse_planner_output + "
    "security.dag.validate_plan(known_tokens(original current prompt))"
)
_SHORT_FORM_VALIDATOR = (
    "security.dag.parse_planner_output + "
    "security.dag.validate_plan(known_tokens(new full user body))"
)


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(seed: int, *parts: Any) -> str:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_int(seed: int, *parts: Any) -> int:
    return int(_stable_key(seed, *parts), 16)


def _approx_tokens(text: str) -> int:
    """Deterministic, tokenizer-free approximation used only by memory."""

    return max(1, math.ceil(len(text) / 4))


def _decode_meta(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _message_content(messages: list[dict[str, str]], role: str) -> str:
    return next(
        (message.get("content", "") for message in messages if message.get("role") == role),
        "",
    )


# ---------------------------------------------------------------------------
# Source rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRow:
    """One original train row plus the decoded fields the builder groups on."""

    index: int  # 1-based line number in the original train.jsonl
    messages: list[dict[str, str]]
    meta: dict[str, Any]
    user: str
    assistant: str
    family: str
    kind: str
    lang: str
    gold_sha256: str
    tools: frozenset[str]


def _source_hashes(source: Path) -> dict[str, str]:
    missing = [name for name in _SOURCE_ARTIFACTS if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"source dataset is missing required artifacts: {', '.join(missing)}"
        )
    # dev/test are hashed as bytes only; their contents are never parsed.
    return {name: _sha256_file(source / name) for name in _SOURCE_ARTIFACTS}


def _load_train_rows(source: Path) -> list[SourceRow]:
    path = source / "train.jsonl"
    rows: list[SourceRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            tag = f"train.jsonl:{line_no}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{tag}: malformed JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{tag}: JSON line must be an object")
            messages = record.get("messages")
            if not (
                isinstance(messages, list)
                and len(messages) == 3
                and all(isinstance(message, dict) for message in messages)
            ):
                raise ValueError(f"{tag}: expected exactly three message objects")
            if [message.get("role") for message in messages] != [
                "system",
                "user",
                "assistant",
            ]:
                raise ValueError(f"{tag}: roles must be system/user/assistant")
            system = messages[0].get("content")
            user = messages[1].get("content")
            assistant = messages[2].get("content")
            if system != R3_SYSTEM_PROMPT:
                raise ValueError(f"{tag}: system prompt is not the current R3 prompt")
            if not isinstance(user, str) or not user.strip():
                raise ValueError(f"{tag}: empty user prompt")
            if not isinstance(assistant, str) or not assistant.strip():
                raise ValueError(f"{tag}: empty assistant output")

            meta = _decode_meta(record.get("meta"))
            family = meta.get("family")
            if not isinstance(family, str) or not family:
                raise ValueError(f"{tag}: meta.family is missing")

            try:
                parsed = parse_planner_output(assistant)
                validated = validate_plan(parsed, known_tokens(user))
            except PlanError as exc:
                raise ValueError(f"{tag}: assistant does not validate: {exc}") from exc

            tools = frozenset(step.tool for step in validated.steps)
            rows.append(
                SourceRow(
                    index=line_no,
                    messages=list(messages),
                    meta=meta,
                    user=user,
                    assistant=assistant,
                    family=family,
                    kind=str(meta.get("kind") or validated.kind),
                    lang=str(meta.get("lang") or "unknown"),
                    gold_sha256=_sha256_text(assistant),
                    tools=tools,
                )
            )
    if not rows:
        raise ValueError("source train.jsonl contains no rows")
    return rows


# ---------------------------------------------------------------------------
# Deterministic balanced selection
# ---------------------------------------------------------------------------


def _family_rows(rows: list[SourceRow]) -> dict[str, list[SourceRow]]:
    grouped: dict[str, list[SourceRow]] = defaultdict(list)
    for row in rows:
        grouped[row.family].append(row)
    return dict(grouped)


def _default_representative(
    seed: int, family: str, family_rows: list[SourceRow], required: frozenset[str]
) -> SourceRow:
    # Prefer a row that carries the most required tools, then a stable hash.
    return min(
        family_rows,
        key=lambda row: (
            -len(row.tools & required),
            _stable_key(seed, "representative", family, row.index),
        ),
    )


def _unique_prompt_families(
    seed: int,
    representatives: dict[str, SourceRow],
    required: frozenset[str],
) -> dict[str, SourceRow]:
    """Keep one deterministic family per current user prompt."""

    kept: dict[str, SourceRow] = {}
    used: set[str] = set()
    ordered = sorted(
        representatives,
        key=lambda family: (
            -len(representatives[family].tools & required),
            _stable_key(seed, "family", family),
        ),
    )
    for family in ordered:
        row = representatives[family]
        if row.user in used:
            continue
        used.add(row.user)
        kept[family] = row
    return kept


def _stratified_pick(
    candidates: dict[str, SourceRow], count: int, seed: int
) -> list[tuple[str, SourceRow]]:
    """Pick ``count`` rows with a deterministic kind/lang proportional spread."""

    if count <= 0:
        return []
    if count > len(candidates):
        raise ValueError(
            f"only {len(candidates)} source families are available; requested {count}"
        )
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for family, row in candidates.items():
        strata[(row.kind, row.lang)].append(family)
    for key in strata:
        strata[key].sort(key=lambda family: _stable_key(seed, "fill", *key, family))

    total = len(candidates)
    raw = {key: count * len(families) / total for key, families in strata.items()}
    quotas = {key: int(math.floor(value)) for key, value in raw.items()}
    left = count - sum(quotas.values())
    order = sorted(strata, key=lambda key: (-(raw[key] - quotas[key]), key))
    guard = 0
    while left > 0:
        progressed = False
        for key in order:
            if left <= 0:
                break
            if quotas[key] < len(strata[key]):
                quotas[key] += 1
                left -= 1
                progressed = True
        guard += 1
        if not progressed or guard > count + 1:
            break

    picked: list[str] = []
    for key in sorted(strata):
        picked.extend(strata[key][: quotas[key]])
    if len(picked) < count:
        chosen = set(picked)
        for family in sorted(
            candidates, key=lambda item: _stable_key(seed, "fill-fallback", item)
        ):
            if family in chosen:
                continue
            picked.append(family)
            chosen.add(family)
            if len(picked) >= count:
                break
    return [(family, candidates[family]) for family in picked[:count]]


def _select_base_rows(
    rows: list[SourceRow],
    seed: int,
    count: int,
    required: frozenset[str],
    *,
    must_include: Collection[str] = (),
    must_exclude: Collection[str] = (),
) -> list[SourceRow]:
    if count < 1:
        raise ValueError("train_rows must be at least 1")

    include = set(must_include)
    exclude = set(must_exclude)
    if len(include) > count:
        raise ValueError(
            f"train_rows={count} is too small to include every short-form train "
            f"source family ({len(include)} families)"
        )

    grouped = _family_rows(rows)
    missing_include = sorted(include - set(grouped))
    if missing_include:
        raise ValueError(
            "short-form train source families are missing from the source: "
            + ", ".join(missing_include)
        )
    all_tools = set().union(*(row.tools for row in rows))
    missing_tools = sorted(required - all_tools)
    if missing_tools:
        raise ValueError(
            "source train split has no row for required tool(s): "
            + ", ".join(missing_tools)
        )

    representatives = {
        family: _default_representative(seed, family, family_rows, required)
        for family, family_rows in grouped.items()
        if family not in exclude
    }
    available = _unique_prompt_families(seed, representatives, required)
    if len(available) < count:
        raise ValueError(
            f"only {len(available)} unique source families with unique prompts are "
            f"available; requested {count} base train rows"
        )
    unavailable_include = sorted(include - set(available))
    if unavailable_include:
        raise ValueError(
            "short-form train source families have no unique base prompt: "
            + ", ".join(unavailable_include)
        )

    selected: dict[str, SourceRow] = {}
    used_prompts: set[str] = set()

    # Reserve the short-form train families before any other base row so the
    # short-form group requirement is always satisfied.
    for family in sorted(include, key=lambda item: _stable_key(seed, "include", item)):
        row = available[family]
        selected[family] = row
        used_prompts.add(row.user)

    # Greedily reserve families for required tools before filling the remainder.
    uncovered = set(required)
    while uncovered:
        if len(selected) >= count:
            raise ValueError(
                f"train_rows={count} is too small to cover required tool(s): "
                + ", ".join(sorted(uncovered))
            )
        options: list[tuple[int, str, str, SourceRow, set[str]]] = []
        for family, family_rows in grouped.items():
            if family in selected or family in exclude:
                continue
            for row in family_rows:
                cover = row.tools & uncovered
                if not cover or row.user in used_prompts:
                    continue
                options.append(
                    (
                        -len(cover),
                        _stable_key(seed, "cover", family, row.index),
                        family,
                        row,
                        set(cover),
                    )
                )
        if not options:
            raise ValueError(
                "no available source family can cover required tool(s): "
                + ", ".join(sorted(uncovered))
            )
        options.sort(key=lambda item: (item[0], item[1]))
        _negative, _key, family, row, cover = options[0]
        selected[family] = row
        used_prompts.add(row.user)
        uncovered -= cover

    remaining = {
        family: row
        for family, row in available.items()
        if family not in selected and row.user not in used_prompts
    }
    need = count - len(selected)
    for family, row in _stratified_pick(remaining, need, seed):
        if family not in selected and row.user not in used_prompts:
            selected[family] = row
            used_prompts.add(row.user)
    if len(selected) < count:
        for family in sorted(
            remaining, key=lambda item: _stable_key(seed, "base-fallback", item)
        ):
            if len(selected) >= count:
                break
            row = remaining[family]
            if family in selected or row.user in used_prompts:
                continue
            selected[family] = row
            used_prompts.add(row.user)

    covered = set().union(*(row.tools for row in selected.values()))
    missing = sorted(required - covered)
    if missing:
        raise ValueError(
            "selected base rows still do not cover required tool(s): "
            + ", ".join(missing)
        )
    if len(selected) != count:
        raise ValueError(
            f"could not select {count} unique source families; selected {len(selected)}"
        )
    return sorted(
        selected.values(), key=lambda row: _stable_key(seed, "base-emit", row.family)
    )


def _dev_candidates(
    rows: list[SourceRow],
    seed: int,
    train_families: set[str],
    train_prompts: set[str],
    required: frozenset[str],
    *,
    exclude_families: Collection[str] = (),
) -> dict[str, SourceRow]:
    grouped = _family_rows(rows)
    excluded = set(exclude_families)
    representatives = {
        family: _default_representative(seed, family, family_rows, required)
        for family, family_rows in grouped.items()
    }
    available = _unique_prompt_families(seed, representatives, required)
    return {
        family: row
        for family, row in available.items()
        if family not in train_families
        and family not in excluded
        and row.user not in train_prompts
    }


# ---------------------------------------------------------------------------
# Context variants
# ---------------------------------------------------------------------------


def _unused_label(prompt: str, kind: str) -> str:
    for number in range(900, 1000):
        label = f"<{kind}_{number}>"
        if label not in prompt:
            return label
    raise ValueError(f"could not find an unused {kind} label for the history turn")


def _unused_digits(prompt: str) -> str:
    for candidate in ("9876", "4321", "2468", "1357", "8642"):
        if candidate not in prompt:
            return candidate
    return "97531"


def _prior_turn(current: str, lang: str) -> tuple[str, str]:
    """Build a clearly stale, hypothetical prior turn with generic slots."""

    payee = _unused_label(current, "PAYEE")
    amount = _unused_digits(current)
    if lang == "zh":
        user = f"早前的示例（不是本次请求）：给{payee}转{amount}。"
        assistant = (
            f"这只是过时的示例：之前示例的收款人是{payee}，金额是{amount}，"
            "没有执行任何真实操作。"
        )
    elif lang == "mixed":
        user = f"早前的 example（not this request）：send {payee} {amount}。"
        assistant = (
            f"Only a stale example：之前的 recipient 是 {payee}，amount 是 {amount}；"
            "nothing was executed。"
        )
    else:
        user = (
            "Earlier hypothetical example (not this request): "
            f"sending {payee} {amount}."
        )
        assistant = (
            f"Stale example only: the prior example recipient was {payee} and "
            f"the amount was {amount}; nothing was executed."
        )
    return user, assistant


_FILLERS: dict[str, tuple[tuple[str, str], ...]] = {
    "en": (("", ""), ("Quick question: ", ""), ("Just checking: ", "")),
    "zh": (("", ""), ("顺便问一下，", ""), ("麻烦你，", "")),
    "mixed": (("", ""), ("顺便问一下，", ""), ("quick check，", "")),
}
_NO_FILLER = ("", "")


def _pick_filler(seed: int, row: SourceRow) -> tuple[str, str]:
    options = _FILLERS.get(row.lang)
    if not options:
        return _NO_FILLER
    index = _stable_int(seed, "filler", row.index) % len(options)
    return options[index]


def _insert_filler(body: str, filler: tuple[str, str]) -> str:
    prefix, suffix = filler
    if not prefix and not suffix:
        return body
    marker = "\nUser: "
    head, separator, request = body.rpartition(marker)
    if not separator:
        marker = "User: "
        if not body.startswith(marker):
            raise ValueError("current prompt has no final 'User:' line")
        head, separator, request = "", marker, body[len(marker) :]
    return head + separator + prefix + request + suffix


def _render_context_variant(
    row: SourceRow, seed: int, variant_index: int
) -> tuple[str, tuple[str, str]]:
    filler = _pick_filler(seed, row)
    current = _insert_filler(row.user, filler)
    prior_user, prior_assistant = _prior_turn(row.user, row.lang)

    memory = BoundedConversationMemory(
        _approx_tokens,
        context_tokens=_HISTORY_CONTEXT_TOKENS,
        history_tokens=_HISTORY_TOKEN_BUDGET,
        output_reserve=0,
        max_turns=2,
    )
    customer_id = f"candidate-{seed}"
    session_id = f"context-{variant_index}-{row.index}"
    memory.remember(customer_id, session_id, prior_user, prior_assistant)
    rendered = memory.render(customer_id, session_id, R3_SYSTEM_PROMPT, current)

    if not rendered.endswith(current):
        raise ValueError("conversation memory did not keep the current prompt last")
    history = rendered[: len(rendered) - len(current)].removesuffix("\n\n")
    if not history.startswith(_HISTORY_HEADER):
        raise ValueError("conversation memory did not emit its exact history header")
    if "[history PAYEE]" not in history or "[number]" not in history:
        raise ValueError("sanitized history is missing the generic prior slots")
    if any(character.isdigit() for character in history):
        raise ValueError("sanitized history still contains a digit")
    leaked = sorted(label for label in known_tokens(row.user) if label in history)
    if leaked:
        raise ValueError(
            "current action token leaked into the history block: "
            + ", ".join(leaked)
        )
    return rendered, filler


# ---------------------------------------------------------------------------
# Short-form product purchase variants
# ---------------------------------------------------------------------------

# Concise, language-appropriate equivalents.  Each form only embeds the exact
# product and amount labels taken from the source subscribe_product args.
_SHORT_FORM_TEMPLATES: dict[str, tuple[str, ...]] = {
    "en": (
        "Please buy {product} for {amount}.",
        "Buy {product} for {amount}.",
        "I want to buy {product} for {amount}.",
        "Please purchase {product} with {amount}.",
        "Could you buy {product} for {amount}?",
        "Purchase {product}, amount {amount}.",
        "I'd like to buy {product} for {amount}.",
        "Can I buy {product} with {amount}?",
        "Buy {product} using {amount}.",
        "Please order {product} for {amount}.",
        "Order {product}, amount {amount}.",
        "I want to purchase {product} for {amount}.",
    ),
    "zh": (
        "请帮我买{product}，金额{amount}。",
        "帮我购买{product}，金额{amount}。",
        "我要买{product}，付款{amount}。",
        "能帮我买{product}，金额{amount}吗？",
        "请用{amount}购买{product}。",
        "购买{product}，使用金额{amount}。",
        "我想用{amount}买{product}。",
        "请购买{product}，支付{amount}。",
        "用{amount}购买{product}。",
        "帮我订购{product}，金额{amount}。",
        "我要购买{product}，金额为{amount}。",
        "请下单购买{product}，金额{amount}。",
    ),
    "mixed": (
        "Please 帮我 buy {product}，金额 {amount}。",
        "帮我 buy {product}，amount {amount}。",
        "buy 一下 {product}，{amount}。",
        "麻烦 purchase {product}，预算 {amount}。",
        "Could you buy {product}, 金额 {amount}?",
        "请 buy {product} for {amount}。",
        "Please purchase {product}, amount {amount}。",
        "帮我 purchase {product} with {amount}。",
        "Can you 买 {product}，金额 {amount}?",
        "用 {amount} 来 buy {product}。",
        "请 order {product}，金额 {amount}。",
        "I'd like to buy {product}，{amount}。",
    ),
}


@dataclass(frozen=True)
class ShortFormSource:
    row: SourceRow
    product: str
    amount: str


@dataclass(frozen=True)
class ShortFormOption:
    row: SourceRow
    product: str
    amount: str
    template_index: int
    request: str
    user: str


def _planner_context_prefix(user: str) -> str:
    """Return everything up to and including the final ``User: `` marker."""

    marker = "\nUser: "
    head, separator, _request = user.rpartition(marker)
    if separator:
        return head + separator
    marker = "User: "
    if user.startswith(marker):
        return marker
    raise ValueError("source prompt has no final 'User:' line")


def _short_form_templates(lang: str) -> tuple[str, ...]:
    return _SHORT_FORM_TEMPLATES.get(lang, _SHORT_FORM_TEMPLATES["en"])


def _subscribe_target(row: SourceRow) -> tuple[str, str] | None:
    """Read the exact product/amount labels from one subscribe_product step."""

    parsed = parse_planner_output(row.assistant)
    if parsed.kind != "plan":
        return None
    steps = [step for step in parsed.steps if step.tool == "subscribe_product"]
    if len(steps) != 1:
        return None
    args = steps[0].args or {}
    product = args.get("product")
    amount = args.get("amount")
    if not isinstance(product, str) or not _PRODUCT_LABEL_RE.fullmatch(product):
        return None
    if not isinstance(amount, str) or not _AMOUNT_LABEL_RE.fullmatch(amount):
        return None
    return product, amount


def _short_form_options(
    family: str, rows: list[SourceRow], seed: int
) -> list[ShortFormOption]:
    """Enumerate valid short forms for one source family, deterministically."""

    sources: list[ShortFormSource] = []
    for row in sorted(
        rows, key=lambda item: _stable_key(seed, "short-row", family, item.index)
    ):
        target = _subscribe_target(row)
        if target is not None:
            sources.append(ShortFormSource(row, *target))

    options: list[ShortFormOption] = []
    seen: set[str] = set()
    for source in sources:
        templates = _short_form_templates(source.row.lang)
        if not templates:
            continue
        offset = (
            _stable_int(seed, "short-template", family, source.row.index)
            % len(templates)
        )
        parsed = parse_planner_output(source.row.assistant)
        for step in range(len(templates)):
            template_index = (offset + step) % len(templates)
            request = templates[template_index].format(
                product=source.product, amount=source.amount
            )
            user = _planner_context_prefix(source.row.user) + request
            if user in seen or user == source.row.user:
                continue
            try:
                validate_plan(parsed, known_tokens(user))
            except PlanError:
                continue
            seen.add(user)
            options.append(
                ShortFormOption(
                    row=source.row,
                    product=source.product,
                    amount=source.amount,
                    template_index=template_index,
                    request=request,
                    user=user,
                )
            )
    return options


def _short_form_options_by_family(
    rows: list[SourceRow], seed: int
) -> dict[str, list[ShortFormOption]]:
    grouped: dict[str, list[SourceRow]] = defaultdict(list)
    for row in rows:
        grouped[row.family].append(row)
    options: dict[str, list[ShortFormOption]] = {}
    for family in sorted(grouped):
        family_options = _short_form_options(family, grouped[family], seed)
        if family_options:
            options[family] = family_options
    return options


def _short_train_group_target(count: int) -> int:
    if count <= 0:
        return 0
    return max(min(count, _MIN_SHORT_TRAIN_GROUPS), math.ceil(count / 2))


def _reserve_short_families(
    options: dict[str, list[ShortFormOption]],
    seed: int,
    short_train: int,
    short_dev: int,
) -> tuple[list[str], list[str]]:
    """Reserve dev groups first, then train groups from the remaining families."""

    if short_train < 0 or short_dev < 0:
        raise ValueError("short-form counts must be non-negative")
    if short_train == 0 and short_dev == 0:
        return [], []

    target = _short_train_group_target(short_train)

    def capacity_of(family: str) -> int:
        return min(
            _MAX_SHORT_FORMS_PER_FAMILY,
            len({option.template_index for option in options[family]}),
        )

    # Hold out the narrowest families first so the train groups keep room for
    # the required second wording form.
    families = sorted(
        options,
        key=lambda item: (
            capacity_of(item),
            _stable_key(seed, "short-dev-family", item),
        ),
    )
    if short_dev > 0 and len(families) < short_dev + target:
        raise ValueError(
            f"only {len(families)} subscribe_product source families provide valid "
            f"short-form targets; need {short_dev} heldout dev groups and {target} "
            f"train groups ({short_dev + target} total)"
        )
    if short_dev == 0 and len(families) < target:
        raise ValueError(
            f"only {len(families)} subscribe_product source families provide valid "
            f"short-form targets; need {target} train groups"
        )

    dev_families = families[:short_dev]
    dev_set = set(dev_families)
    remaining = [family for family in families if family not in dev_set]
    train_families = remaining[:target]
    capacity = sum(capacity_of(family) for family in train_families)
    if capacity < short_train:
        for family in remaining[target:]:
            train_families.append(family)
            capacity += capacity_of(family)
            if capacity >= short_train:
                break
    if capacity < short_train:
        raise ValueError(
            f"subscribe_product source families can supply only {capacity} "
            f"short-form train prompts; requested {short_train}"
        )
    return train_families, dev_families


def _select_short_options(
    families: Collection[str],
    options: dict[str, list[ShortFormOption]],
    seed: int,
    count: int,
    *,
    forbidden_prompts: Collection[str],
    tag: str,
) -> list[ShortFormOption]:
    if count <= 0:
        return []
    ordered = sorted(
        families, key=lambda item: _stable_key(seed, f"short-{tag}-family", item)
    )
    used = set(forbidden_prompts)
    used_templates: dict[str, set[int]] = {family: set() for family in ordered}
    used_rows: dict[str, set[int]] = {family: set() for family in ordered}
    template_count = max(len(value) for value in _SHORT_FORM_TEMPLATES.values())
    chosen: list[ShortFormOption] = []
    for round_index in range(_MAX_SHORT_FORMS_PER_FAMILY):
        for position, family in enumerate(ordered):
            if len(chosen) >= count:
                break
            target = (position + round_index) % template_count
            candidates = sorted(
                options[family],
                key=lambda candidate: (
                    candidate.template_index != target,
                    candidate.row.index in used_rows[family],
                    candidate.template_index,
                    _stable_key(seed, "short-candidate", family, candidate.row.index),
                ),
            )
            for candidate in candidates:
                if candidate.template_index in used_templates[family]:
                    continue
                if candidate.user in used:
                    continue
                chosen.append(candidate)
                used.add(candidate.user)
                used_templates[family].add(candidate.template_index)
                used_rows[family].add(candidate.row.index)
                break
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        raise ValueError(
            f"could not build {count} unique short-form {tag} prompts from the "
            "reserved source families"
        )
    return chosen


def _assert_short_form_context(source: SourceRow, user: str, request: str) -> None:
    prefix = _planner_context_prefix(source.user)
    if user != prefix + request:
        raise ValueError(
            f"row {source.index}: short form did not retain the source context block"
        )


# ---------------------------------------------------------------------------
# Output rows / provenance
# ---------------------------------------------------------------------------


def _output_meta(
    row: SourceRow,
    *,
    candidate_split: str,
    context_variant: bool,
    filler: tuple[str, str] | None = None,
    short_form_variant: bool = False,
    short_form_challenge: bool = False,
) -> dict[str, Any]:
    meta = dict(row.meta)
    meta.pop("known_tokens", None)
    meta.pop("_runtime", None)
    meta.setdefault("kind", row.kind)
    meta.setdefault("lang", row.lang)
    meta.update(
        {
            "candidate": True,
            "candidate_split": candidate_split,
            "source_split": "train",
            "source_row": row.index,
            "source_family": row.family,
            "gold_sha256": row.gold_sha256,
            "validator_passed": True,
            "context_variant": context_variant,
            "short_form_variant": short_form_variant,
            "short_form_challenge": short_form_challenge,
        }
    )
    if filler is not None:
        meta["filler_prefix"] = filler[0]
        meta["filler_suffix"] = filler[1]
    if short_form_variant:
        meta["short_form_validator"] = _SHORT_FORM_VALIDATOR
    return meta


def _record(
    row: SourceRow,
    *,
    candidate_split: str,
    user: str,
    context_variant: bool,
    filler: tuple[str, str] | None = None,
    short_form_variant: bool = False,
    short_form_challenge: bool = False,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": R3_SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": row.assistant},
    ]
    return {
        "messages": messages,
        "meta": json.dumps(
            _output_meta(
                row,
                candidate_split=candidate_split,
                context_variant=context_variant,
                filler=filler,
                short_form_variant=short_form_variant,
                short_form_challenge=short_form_challenge,
            ),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ),
    }


def _provenance(record: dict[str, Any]) -> dict[str, Any]:
    meta = _decode_meta(record.get("meta"))
    return {
        "candidate_split": meta["candidate_split"],
        "source_split": meta["source_split"],
        "source_row": meta["source_row"],
        "source_family": meta["source_family"],
        "gold_sha256": meta["gold_sha256"],
        "validator_passed": meta["validator_passed"],
        "context_variant": meta["context_variant"],
        "short_form_variant": bool(meta.get("short_form_variant", False)),
        "short_form_challenge": bool(meta.get("short_form_challenge", False)),
        "short_form_validator": meta.get("short_form_validator"),
        "kind": meta.get("kind"),
        "lang": meta.get("lang"),
    }


def _validate_emitted(
    record: dict[str, Any],
    source: SourceRow,
    *,
    validate_user: str | None = None,
) -> None:
    assistant = _message_content(record["messages"], "assistant")
    if assistant != source.assistant:
        raise ValueError(f"row {source.index}: assistant gold changed during build")
    if _sha256_text(assistant) != source.gold_sha256:
        raise ValueError(f"row {source.index}: gold hash changed during build")
    user = _message_content(record["messages"], "user")
    if not user.strip():
        raise ValueError(f"row {source.index}: emitted user prompt is empty")
    parsed = parse_planner_output(assistant)
    validate_plan(parsed, known_tokens(source.user))
    if validate_user is not None:
        validate_plan(parsed, known_tokens(validate_user))


# ---------------------------------------------------------------------------
# Stats / artifacts
# ---------------------------------------------------------------------------


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


def _distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    kinds: Counter[str] = Counter()
    langs: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    for record in records:
        meta = _decode_meta(record.get("meta"))
        kinds[str(meta.get("kind") or "unknown")] += 1
        langs[str(meta.get("lang") or "unknown")] += 1
        parsed = parse_planner_output(_message_content(record["messages"], "assistant"))
        if parsed.kind == "plan":
            tools.update(step.tool for step in parsed.steps)
    return {
        "kind": dict(sorted(kinds.items())),
        "lang": dict(sorted(langs.items())),
        "tools": dict(sorted(tools.items())),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: Any, *, indent: int | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            default=str,
        )
        handle.write("\n")


# ---------------------------------------------------------------------------
# Public build
# ---------------------------------------------------------------------------


def _validate_output_path(source: Path, output: Path) -> Path:
    output = Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output path: {output}")
    resolved_output = output.resolve()
    resolved_source = source.resolve()
    if resolved_output == resolved_source or resolved_source in resolved_output.parents:
        raise ValueError(
            f"refusing to write the candidate inside the source dataset: {output}"
        )
    return output


def build(
    source: Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT,
    *,
    seed: int = DEFAULT_SEED,
    train_rows: int = DEFAULT_TRAIN_ROWS,
    context_train: int = DEFAULT_CONTEXT_TRAIN,
    context_dev: int = DEFAULT_CONTEXT_DEV,
    short_train: int = DEFAULT_SHORT_TRAIN,
    short_dev: int = DEFAULT_SHORT_DEV,
    required_tools: Collection[str] | None = None,
) -> dict[str, Any]:
    """Build the draft contextual candidate into a brand-new ``output`` directory."""

    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source dataset directory does not exist: {source}")
    output = _validate_output_path(source, Path(output))

    if train_rows < 1:
        raise ValueError("train_rows must be at least 1")
    if context_train < 0 or context_dev < 0:
        raise ValueError("context counts must be non-negative")
    if short_train < 0 or short_dev < 0:
        raise ValueError("short-form counts must be non-negative")
    if context_train > train_rows:
        raise ValueError("context_train cannot exceed train_rows")

    required = frozenset(
        REQUIRED_POSITIVE_TOOLS if required_tools is None else required_tools
    )
    source_hashes = _source_hashes(source)
    rows = _load_train_rows(source)

    short_options = (
        _short_form_options_by_family(rows, seed)
        if short_train or short_dev
        else {}
    )
    short_train_families, short_dev_families = _reserve_short_families(
        short_options, seed, short_train, short_dev
    )

    base_rows = _select_base_rows(
        rows,
        seed,
        train_rows,
        required,
        must_include=short_train_families,
        must_exclude=short_dev_families,
    )
    train_families = {row.family for row in base_rows}
    train_prompts = {row.user for row in base_rows}

    variant_candidates = {row.family: row for row in base_rows}
    train_variant_sources = _stratified_pick(
        variant_candidates, context_train, seed
    )

    dev_candidates = _dev_candidates(
        rows,
        seed,
        train_families,
        train_prompts,
        required,
        exclude_families=short_dev_families,
    )
    if len(dev_candidates) < context_dev:
        raise ValueError(
            f"only {len(dev_candidates)} source families remain outside the "
            f"{train_rows} sampled train families; requested {context_dev} dev "
            "context variants"
        )
    dev_variant_sources = _stratified_pick(dev_candidates, context_dev, seed)

    train_records: list[dict[str, Any]] = []
    for row in base_rows:
        record = _record(
            row, candidate_split="train", user=row.user, context_variant=False
        )
        _validate_emitted(record, row)
        train_records.append(record)

    for variant_index, (_family, row) in enumerate(train_variant_sources):
        user, filler = _render_context_variant(row, seed, variant_index)
        record = _record(
            row,
            candidate_split="train",
            user=user,
            context_variant=True,
            filler=filler,
        )
        _validate_emitted(record, row)
        train_records.append(record)

    train_base_users = [
        _message_content(record["messages"], "user") for record in train_records
    ]
    train_short_options = _select_short_options(
        short_train_families,
        short_options,
        seed,
        short_train,
        forbidden_prompts=train_base_users,
        tag="train",
    )
    for option in train_short_options:
        _assert_short_form_context(option.row, option.user, option.request)
        record = _record(
            option.row,
            candidate_split="train",
            user=option.user,
            context_variant=False,
            short_form_variant=True,
        )
        _validate_emitted(record, option.row, validate_user=option.user)
        train_records.append(record)

    dev_records: list[dict[str, Any]] = []
    for variant_index, (_family, row) in enumerate(dev_variant_sources):
        user, filler = _render_context_variant(
            row, seed, train_rows + variant_index
        )
        record = _record(
            row,
            candidate_split="dev",
            user=user,
            context_variant=True,
            filler=filler,
        )
        _validate_emitted(record, row)
        dev_records.append(record)

    train_users = [
        _message_content(record["messages"], "user") for record in train_records
    ]
    dev_context_users = [
        _message_content(record["messages"], "user") for record in dev_records
    ]
    dev_short_options = _select_short_options(
        short_dev_families,
        short_options,
        seed,
        short_dev,
        forbidden_prompts=set(train_users) | set(dev_context_users),
        tag="dev",
    )
    for option in dev_short_options:
        _assert_short_form_context(option.row, option.user, option.request)
        record = _record(
            option.row,
            candidate_split="dev",
            user=option.user,
            context_variant=False,
            short_form_variant=True,
            short_form_challenge=True,
        )
        _validate_emitted(record, option.row, validate_user=option.user)
        dev_records.append(record)

    random.Random(seed ^ 0xC0FFEE).shuffle(train_records)

    dev_families = {row.family for _family, row in dev_variant_sources}
    short_train_used = {option.row.family for option in train_short_options}
    short_dev_used = {option.row.family for option in dev_short_options}
    if train_families & dev_families:
        raise ValueError("source family crosses candidate train/dev splits")
    if not {row.family for _family, row in train_variant_sources} <= train_families:
        raise ValueError("train variant family is outside the sampled train families")
    if short_train_used != set(short_train_families):
        raise ValueError("not every reserved short-form train family was emitted")
    if short_dev_used != set(short_dev_families):
        raise ValueError("not every reserved short-form dev family was emitted")
    if not short_train_used <= train_families:
        raise ValueError("short-form train family is outside the sampled train families")
    if short_dev_used & train_families:
        raise ValueError("short-form dev challenge family crosses into candidate train")
    if short_dev_used & dev_families:
        raise ValueError("short-form dev challenge family collides with context dev")

    train_users = [
        _message_content(record["messages"], "user") for record in train_records
    ]
    dev_users = [
        _message_content(record["messages"], "user") for record in dev_records
    ]
    if len(set(train_users)) != len(train_users):
        raise ValueError("duplicate current prompt in candidate train")
    if len(set(dev_users)) != len(dev_users):
        raise ValueError("duplicate current prompt in candidate dev")
    if set(train_users) & set(dev_users):
        raise ValueError("a current prompt crosses candidate train/dev splits")

    prompt_sha = _sha256_text(R3_SYSTEM_PROMPT)
    requested = {
        "train_rows": train_rows,
        "context_train": context_train,
        "context_dev": context_dev,
        "context_total": context_train + context_dev,
        "short_train": short_train,
        "short_dev": short_dev,
        "short_total": short_train + short_dev,
    }
    actual = {
        "train": len(train_records),
        "dev": len(dev_records),
        "train_base_rows": len(base_rows),
        "train_context_rows": len(train_variant_sources),
        "train_short_form_rows": len(train_short_options),
        "train_total": len(train_records),
        "dev_context_rows": len(dev_variant_sources),
        "dev_short_form_challenge_rows": len(dev_short_options),
        "dev_total": len(dev_records),
        "short_form_total": len(train_short_options) + len(dev_short_options),
    }
    stats: dict[str, Any] = {
        "draft": True,
        "candidate": True,
        "training_ready": False,
        "release_ready": False,
        "application_release_ready": False,
        "readiness_claim": "none",
        "draft_note": _DRAFT_NOTE,
        "seed": seed,
        "requested": requested,
        "actual": actual,
        "counts": {**requested, **actual},
        "totals": {
            "train": len(train_records),
            "dev": len(dev_records),
            "test": "not_inspected",
        },
        "source": {
            "source_dir": str(source),
            "artifacts_sha256": source_hashes,
            "train_rows_read": len(rows),
            "training_derivation": "train.jsonl only",
            "dev_test_content_inspected": False,
        },
        "source_artifacts_sha256": source_hashes,
        "source_hashes": source_hashes,
        "sourcehashes": source_hashes,
        "prompt_sha256": prompt_sha,
        "promptSHA256": prompt_sha,
        "rows": {
            "train": [_provenance(record) for record in train_records],
            "dev": [_provenance(record) for record in dev_records],
        },
        "splits": {
            "train": {
                "count": len(train_records),
                "base_rows": len(base_rows),
                "context_variants": len(train_variant_sources),
                "short_form_variants": len(train_short_options),
                "short_form_groups": len(short_train_used),
                "families": len(train_families),
                **_distribution(train_records),
            },
            "dev": {
                "count": len(dev_records),
                "base_rows": 0,
                "context_variants": len(dev_variant_sources),
                "families": len(dev_families),
                "short_form_challenge": {
                    "purpose": _SHORT_FORM_CHALLENGE_PURPOSE,
                    "count": len(dev_short_options),
                    "groups": len(short_dev_used),
                    "heldout": True,
                },
                **_distribution(dev_records),
            },
            "test": {
                "count": "not_inspected",
                "copied_byte_for_byte": True,
                "used_for_training": False,
            },
        },
        "families": {
            "train": sorted(train_families),
            "train_base": sorted(row.family for row in base_rows),
            "train_variants": sorted(
                row.family for _family, row in train_variant_sources
            ),
            "dev": sorted(dev_families),
            "dev_variants": sorted(dev_families),
            "train_short_form": sorted(short_train_used),
            "dev_short_form_challenge": sorted(short_dev_used),
        },
        "family_isolation": {
            "train_dev_disjoint": True,
            "train_variant_families_subset_of_train": True,
            "short_train_families_subset_of_train": True,
            "short_dev_challenge_disjoint_from_train": True,
            "short_dev_challenge_disjoint_from_context_dev": True,
            "dev_source_split": "train",
        },
        "short_form": {
            "source": "original v3 train subscribe_product rows only",
            "actual_args_derived": True,
            "invented_labels": False,
            "gold_unchanged": True,
            "validated_with_new_user": True,
            "train": {
                "count": len(train_short_options),
                "groups": len(short_train_used),
                "max_forms_per_family": _MAX_SHORT_FORMS_PER_FAMILY,
                "families": sorted(short_train_used),
            },
            "dev_challenge": {
                "purpose": _SHORT_FORM_CHALLENGE_PURPOSE,
                "count": len(dev_short_options),
                "groups": len(short_dev_used),
                "families": sorted(short_dev_used),
            },
        },
        "validation": {
            "performed": True,
            "validator": _VALIDATOR,
            "all_passed": True,
            "gold_unchanged": True,
            "short_form_gold_validated_with_new_user": True,
            "short_form_validator": _SHORT_FORM_VALIDATOR,
            "token_validation": "pending",
            "tokenizer_length_check": "pending",
        },
        "history": {
            "builder": "agent.conversation_memory.BoundedConversationMemory.render",
            "count_tokens": "ceil(len(text)/4) deterministic approximation",
            "context_tokens": _HISTORY_CONTEXT_TOKENS,
            "history_tokens": _HISTORY_TOKEN_BUDGET,
            "output_reserve": 0,
        },
        "test_eval_only": {
            "copied_byte_for_byte": True,
            "sha256": source_hashes["test.jsonl"],
            "used_for_training": False,
        },
        "dev_eval_only": {
            "copied": False,
            "note": (
                "candidate dev contains the new context variants plus the heldout "
                "short-form challenge rows"
            ),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        _write_jsonl(staging / "train.jsonl", train_records)
        _write_jsonl(staging / "dev.jsonl", dev_records)
        # Evaluation-only split: copy bytes, never parse or derive from it.
        shutil.copyfile(source / "test.jsonl", staging / "test.jsonl")
        _write_json(staging / "dataset_info.json", _dataset_info(), indent=2)

        artifact_hashes = {
            name: _sha256_file(staging / name) for name in _OUTPUT_ARTIFACTS
        }
        if artifact_hashes["test.jsonl"] != source_hashes["test.jsonl"]:
            raise ValueError("test split copy is not byte-for-byte identical")
        stats["artifact_sha256"] = artifact_hashes
        _write_json(staging / "stats.json", stats, indent=2)

        manifest = {
            "artifact_sha256": {
                **artifact_hashes,
                "stats.json": _sha256_file(staging / "stats.json"),
            },
            "prompt_sha256": prompt_sha,
            "draft": True,
            "release_ready": False,
        }
        _write_json(staging / "manifest.json", manifest, indent=2)

        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite existing output path: {output}"
            )
        os.replace(staging, output)
    except BaseException:
        # Preserve the staging directory for inspection; never delete source or
        # an existing output path.
        raise
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS)
    parser.add_argument("--context-train", type=int, default=DEFAULT_CONTEXT_TRAIN)
    parser.add_argument("--context-dev", type=int, default=DEFAULT_CONTEXT_DEV)
    parser.add_argument("--short-train", type=int, default=DEFAULT_SHORT_TRAIN)
    parser.add_argument("--short-dev", type=int, default=DEFAULT_SHORT_DEV)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stats = build(
        args.source,
        args.output,
        seed=args.seed,
        train_rows=args.train_rows,
        context_train=args.context_train,
        context_dev=args.context_dev,
        short_train=args.short_train,
        short_dev=args.short_dev,
    )
    print(json.dumps(stats["actual"], ensure_ascii=False, sort_keys=True))
    print(json.dumps(stats["splits"], ensure_ascii=False, sort_keys=True))
    print("draft: " + json.dumps(
        {"draft": stats["draft"], "draft_note": stats["draft_note"]},
        ensure_ascii=False,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
