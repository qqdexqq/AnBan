"""Code-side intent rules that run before the planner.

A rule may return a planner-shaped output (``{"kind": "plan", ...}`` or
``{"kind": "clarify", ...}``) or ``None`` to let the next rule, and finally the
model, decide. Its output goes through the same parser, DAG validator, policy
engine and approval flow as a model plan, so a rule can never lower a tier.
Rules see the raw turn text because they are code, not the model; plan args must
still use the session's semantic tokens.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agent.intents import analysis, bills, cards, reminders, scheduled, subscriptions


@dataclass(frozen=True)
class IntentContext:
    raw_text: str
    tokenized_text: str
    conn: Any
    secret_key: bytes
    customer_id: str
    session_id: str
    now: datetime
    lang: str
    # To refer to a bank object, mint its token with
    # ``vault.mint(Session(session_id=..., plan_id=..., customer_id=...), kind, real_id, tags, "bank_context")``
    # (security.tokens) and put the returned ``entry.label`` in the plan args.
    vault: Any = None
    plan_id: str = ""


Rule = Callable[[IntentContext], "dict[str, Any] | None"]

# Each scenario owns its own module; order only matters when two rules match.
RULES: tuple[Rule, ...] = (
    cards.resolve,
    scheduled.resolve,
    subscriptions.resolve,
    bills.resolve,
    analysis.resolve,
    reminders.resolve,
)


# Clarify questions written by these rules are the bank's own words, so the
# responder may show them verbatim (planner questions stay allowlisted). Keyed by
# session; bounded so a long-running demo cannot grow it without limit.
_TRUSTED_QUESTIONS: dict[str, set[str]] = {}
_MAX_TRUSTED_SESSIONS = 1000


def _key(customer_id: str, session_id: str) -> str:
    # Clients choose session ids, so a question is trusted only for its owner.
    return f"{customer_id}{session_id}"


def is_trusted_question(customer_id: str, session_id: str, question: str) -> bool:
    return question in _TRUSTED_QUESTIONS.get(_key(customer_id, session_id), set())


def remember_question(customer_id: str, session_id: str, question: str) -> None:
    """Mark a code-written clarify question as safe to show verbatim."""
    key = _key(customer_id, session_id)
    if len(_TRUSTED_QUESTIONS) >= _MAX_TRUSTED_SESSIONS and key not in _TRUSTED_QUESTIONS:
        _TRUSTED_QUESTIONS.pop(next(iter(_TRUSTED_QUESTIONS)))
    _TRUSTED_QUESTIONS.setdefault(key, set()).add(question)


def resolve(ctx: IntentContext) -> dict[str, Any] | None:
    for rule in RULES:
        output = rule(ctx)
        if output is not None:
            question = output.get("question") if output.get("kind") == "clarify" else None
            if isinstance(question, str) and question:
                remember_question(ctx.customer_id, ctx.session_id, question)
            return output
    return None


def guard_model_plan(ctx: IntentContext, raw: str) -> str | None:
    """Checks on a model-written plan; a replacement output, or None to keep it."""
    return cards.guard_model_plan(ctx, raw) or subscriptions.guard_model_plan(ctx, raw)
