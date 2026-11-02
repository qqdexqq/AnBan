"""Pydantic schemas and validation for planner task DAGs and tool arguments.

The planner only proposes one of four output kinds.  Everything it puts in a
plan is re-checked here against the closed tool registry: tool names, argument
types, semantic tokens and step references.
"""

from __future__ import annotations

import heapq
import json
import re
import types
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from agent.tools import (
    CONTROL_PLAN_TOOLS,
    REGISTRY,
    STEP_REF_RE,
    TOKEN_RE,
    TokenKind,
)

MAX_STEPS = 12

_FENCE_RE = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n?(.*?)\r?\n?```\s*\Z",
    re.DOTALL | re.IGNORECASE,
)
_TOKEN_FULL_RE = re.compile(rf"^{TOKEN_RE}$")
_STEP_REF_FULL_RE = re.compile(rf"^{STEP_REF_RE}$")
_STEP_REF_PARTS_RE = re.compile(r"^\$(s\d+)\.([a-z_]+)$")


class PlanError(Exception):
    """Planner output rejected by the DAG check."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


# --------------------------------------------------------------------- outputs


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    after: list[str] = Field(default_factory=list)


class PlanOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["plan"]
    steps: list[PlanStep] = Field(default_factory=list)


class ClarifyOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["clarify"]
    question: str
    options: list[str] = Field(default_factory=list)


class RefuseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["refuse"]
    reason: Literal["out_of_scope", "not_account_owner", "not_supported"]


class ScamOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["scam"]
    signals: list[str] = Field(default_factory=list)


PlannerOutput = Annotated[
    PlanOutput | ClarifyOutput | RefuseOutput | ScamOutput,
    Field(discriminator="kind"),
]

_OUTPUT_ADAPTER = TypeAdapter(PlannerOutput)


def _validate_output_data(data: Any) -> PlannerOutput:
    try:
        return _OUTPUT_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise PlanError("bad_output", str(exc)) from exc


MAX_PLANNER_OUTPUT_CHARS = 20_000


def parse_planner_output(text: str) -> PlannerOutput:
    """Parse a planner JSON object, tolerating a ```` ```json ```` fence."""

    if not isinstance(text, str):
        raise PlanError("invalid_json", "planner output must be a string")

    # A real plan is < 4 KB; refuse anything far larger before parsing it.
    if len(text) > MAX_PLANNER_OUTPUT_CHARS:
        raise PlanError("invalid_json", "planner output too large")

    body = text
    fenced = _FENCE_RE.match(text)
    if fenced is not None:
        body = fenced.group(1)

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError) as exc:
        raise PlanError("invalid_json", type(exc).__name__) from exc
    if not isinstance(data, dict):
        raise PlanError("invalid_json", "planner output must be a JSON object")
    return _validate_output_data(data)


# ---------------------------------------------------------------- validated plan


def _id_key(step_id: str) -> tuple[str, int, str]:
    prefix = step_id.rstrip("0123456789")
    digits = step_id[len(prefix) :]
    return (prefix, int(digits) if digits else -1, step_id)


class ValidatedPlan(BaseModel):
    """A planner output that passed the DAG check.

    ``steps`` are in a deterministic topological order for plans and empty for
    the other output kinds.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["plan", "clarify", "refuse", "scam"]
    steps: list[PlanStep] = Field(default_factory=list)
    output: PlannerOutput

    def parallel_groups(self) -> list[list[PlanStep]]:
        """Return steps grouped into levels that may run concurrently."""

        if not self.steps:
            return []
        levels: dict[str, int] = {}
        for step in self.steps:
            level = 0
            for dependency in step.after:
                level = max(level, levels.get(dependency, -1) + 1)
            levels[step.id] = level

        grouped: dict[int, list[PlanStep]] = {}
        for step in self.steps:
            grouped.setdefault(levels[step.id], []).append(step)
        return [
            sorted(grouped[level], key=lambda item: _id_key(item.id))
            for level in sorted(grouped)
        ]


# ------------------------------------------------------------- argument helpers


def _annotation_kinds(annotation: Any) -> frozenset[str]:
    """Collect the token kinds an annotation accepts."""

    origin = get_origin(annotation)
    if origin is Annotated:
        for metadata in getattr(annotation, "__metadata__", ()):
            if isinstance(metadata, TokenKind):
                return frozenset({metadata.kind})
        return _annotation_kinds(annotation.__origin__)
    if origin in (Union, types.UnionType):
        kinds: set[str] = set()
        for argument in get_args(annotation):
            kinds.update(_annotation_kinds(argument))
        return frozenset(kinds)
    if origin in (list, tuple, set, frozenset):
        arguments = get_args(annotation)
        return _annotation_kinds(arguments[0]) if arguments else frozenset()
    return frozenset()


def _field_kinds(name: str, field: Any) -> frozenset[str]:
    for metadata in field.metadata:
        if isinstance(metadata, TokenKind):
            return frozenset({metadata.kind})
    kinds = _annotation_kinds(field.annotation)
    if kinds:
        return kinds
    return {
        "payee": frozenset({"PAYEE"}),
        "query": frozenset({"PAYEE"}),
        "amount": frozenset({"AMT"}),
        "date": frozenset({"DATE"}),
        "deliver": frozenset({"DATE"}),
        "when": frozenset({"DATE"}),
        "sub": frozenset({"SUB"}),
        "card": frozenset({"CARD"}),
        "product": frozenset({"PROD"}),
        "merchant": frozenset({"MERCHANT"}),
        "account": frozenset({"ACCT"}),
    }.get(name, frozenset())


def _model_type(annotation: Any) -> type[BaseModel] | None:
    origin = get_origin(annotation)
    if origin is Annotated:
        return _model_type(annotation.__origin__)
    if origin in (Union, types.UnionType):
        for argument in get_args(annotation):
            model = _model_type(argument)
            if model is not None:
                return model
        return None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _scan_tokens(
    value: Any,
    annotation: Any,
    known_tokens: Mapping[str, str],
    path: str,
    *,
    kinds: frozenset[str] | None = None,
) -> None:
    if isinstance(value, str):
        if _TOKEN_FULL_RE.fullmatch(value):
            actual = known_tokens.get(value)
            if actual is None:
                raise PlanError("unknown_token", f"{path}: token {value} is not known")
            expected = _annotation_kinds(annotation) if kinds is None else kinds
            if expected and actual not in expected:
                raise PlanError(
                    "wrong_token_kind",
                    f"{path}: token {value} is {actual}, expected "
                    f"{'/'.join(sorted(expected))}",
                )
        return

    if isinstance(value, list):
        arguments = get_args(annotation)
        item_annotation = arguments[0] if get_origin(annotation) is list else Any
        for index, item in enumerate(value):
            _scan_tokens(item, item_annotation, known_tokens, f"{path}[{index}]")
        return

    if isinstance(value, dict):
        model = _model_type(annotation)
        if model is not None:
            for key, item in value.items():
                field = model.model_fields.get(key)
                if field is None:
                    continue
                _scan_tokens(
                    item,
                    field.annotation,
                    known_tokens,
                    f"{path}.{key}",
                    kinds=_field_kinds(key, field),
                )
        else:
            for key, item in value.items():
                _scan_tokens(item, Any, known_tokens, f"{path}.{key}")


def _scan_refs(
    value: Any,
    step_id: str,
    step_map: Mapping[str, PlanStep],
    ancestors: Mapping[str, frozenset[str]],
    path: str,
) -> None:
    if isinstance(value, str):
        if not _STEP_REF_FULL_RE.fullmatch(value):
            return
        match = _STEP_REF_PARTS_RE.fullmatch(value)
        assert match is not None
        source_id, field_name = match.groups()
        source = step_map.get(source_id)
        if source is None:
            raise PlanError("bad_step_ref", f"{path}: step {source_id} does not exist")
        if source_id not in ancestors[step_id]:
            raise PlanError(
                "bad_step_ref",
                f"{path}: step {source_id} is not an ancestor of {step_id}",
            )
        expected_outputs = REGISTRY[source.tool].outputs
        if field_name not in expected_outputs:
            raise PlanError(
                "bad_step_ref",
                f"{path}: {source.tool} does not output {field_name!r}",
            )
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            _scan_refs(item, step_id, step_map, ancestors, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _scan_refs(item, step_id, step_map, ancestors, f"{path}.{key}")


# -------------------------------------------------------------------- validation


def _as_output(output: Any) -> PlannerOutput:
    if isinstance(output, (PlanOutput, ClarifyOutput, RefuseOutput, ScamOutput)):
        return output
    if isinstance(output, str):
        return parse_planner_output(output)
    if isinstance(output, Mapping):
        return _validate_output_data(dict(output))
    raise PlanError("bad_output", f"unsupported planner output type {type(output)!r}")


def _validate_clarify(
    output: ClarifyOutput, known_tokens: Mapping[str, str]
) -> ValidatedPlan:
    allowed = {"PAYEE", "SUB", "CARD"}
    for option in output.options:
        kind = known_tokens.get(option)
        if kind is None:
            raise PlanError(
                "unknown_token", f"clarify option {option} is not a known token"
            )
        if kind not in allowed:
            raise PlanError(
                "wrong_token_kind",
                f"clarify option {option} is {kind}, expected PAYEE/SUB/CARD",
            )
    return ValidatedPlan(kind="clarify", steps=[], output=output)


def _dependency_graph(
    steps: list[PlanStep],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    step_ids = {step.id for step in steps}
    dependencies: dict[str, set[str]] = {}
    dependents: dict[str, set[str]] = {step_id: set() for step_id in step_ids}

    for step in steps:
        deps = set(step.after)
        for dependency in deps:
            if dependency not in step_ids:
                raise PlanError(
                    "bad_step_ref",
                    f"{step.id}: after references unknown step {dependency}",
                )
            dependents[dependency].add(step.id)
        dependencies[step.id] = deps
    return dependencies, dependents


def _topological_order(
    dependencies: Mapping[str, set[str]], dependents: Mapping[str, set[str]]
) -> list[str]:
    remaining = {step_id: len(deps) for step_id, deps in dependencies.items()}
    ready = [_id_key(step_id) for step_id, count in remaining.items() if count == 0]
    heapq.heapify(ready)
    order: list[str] = []

    while ready:
        _, _, step_id = heapq.heappop(ready)
        order.append(step_id)
        for dependent in sorted(dependents[step_id], key=_id_key):
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                heapq.heappush(ready, _id_key(dependent))

    if len(order) != len(remaining):
        cyclic = sorted(
            (step_id for step_id, count in remaining.items() if count > 0), key=_id_key
        )
        raise PlanError("cycle", f"cycle among steps: {', '.join(cyclic)}")
    return order


def _ancestor_sets(
    steps: Mapping[str, PlanStep], order: list[str]
) -> dict[str, frozenset[str]]:
    ancestors: dict[str, frozenset[str]] = {}
    for step_id in order:
        reached: set[str] = set()
        for dependency in steps[step_id].after:
            reached.add(dependency)
            reached.update(ancestors[dependency])
        ancestors[step_id] = frozenset(reached)
    return ancestors


def _validate_plan(output: PlanOutput, known_tokens: Mapping[str, str]) -> ValidatedPlan:
    steps = output.steps
    if not steps:
        raise PlanError("empty_plan", "plan has no steps")
    if len(steps) > MAX_STEPS:
        raise PlanError(
            "too_many_steps", f"{len(steps)} steps exceeds the limit of {MAX_STEPS}"
        )

    seen_ids: set[str] = set()
    for step in steps:
        if step.id in seen_ids:
            raise PlanError("duplicate_step_id", f"duplicate step id {step.id}")
        seen_ids.add(step.id)

    step_map = {step.id: step for step in steps}
    for step in steps:
        if step.tool not in REGISTRY:
            raise PlanError("unknown_tool", f"{step.id}: unknown tool {step.tool!r}")
        if step.tool in CONTROL_PLAN_TOOLS:
            raise PlanError(
                "control_tool_in_plan",
                f"{step.id}: {step.tool} must be a kind, not a plan step",
            )

    dependencies, dependents = _dependency_graph(steps)
    order = _topological_order(dependencies, dependents)
    ancestors = _ancestor_sets(step_map, order)

    typed_args: dict[str, dict[str, Any]] = {}
    for step in steps:
        spec = REGISTRY[step.tool]
        model = spec.args_model

        # Semantic tokens are checked against the schema before pydantic turns
        # a wrong-kind token into a generic bad_args error.
        for name, field in model.model_fields.items():
            if name not in step.args:
                continue
            _scan_tokens(
                step.args[name],
                field.annotation,
                known_tokens,
                f"{step.id}.{name}",
                kinds=_field_kinds(name, field),
            )

        try:
            typed = model.model_validate(step.args)
        except ValidationError as exc:
            raise PlanError("bad_args", f"{step.id}.{step.tool}: {exc}") from exc
        typed_args[step.id] = typed.model_dump(exclude_none=True)

        _scan_refs(step.args, step.id, step_map, ancestors, f"{step.id}.args")

    ordered_steps = [
        PlanStep(
            id=step_id,
            tool=step_map[step_id].tool,
            args=typed_args[step_id],
            after=list(step_map[step_id].after),
        )
        for step_id in order
    ]
    return ValidatedPlan(kind="plan", steps=ordered_steps, output=output)


def validate_plan(
    output: PlannerOutput | Mapping[str, Any] | str,
    known_tokens: Mapping[str, str],
) -> ValidatedPlan:
    """Validate a planner output against the tool registry and session tokens."""

    parsed = _as_output(output)
    if isinstance(parsed, PlanOutput):
        return _validate_plan(parsed, known_tokens)
    if isinstance(parsed, ClarifyOutput):
        return _validate_clarify(parsed, known_tokens)
    return ValidatedPlan(kind=parsed.kind, steps=[], output=parsed)
