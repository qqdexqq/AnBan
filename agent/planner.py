"""Planner clients: local vLLM, cloud APIs and an offline scripted demo mode.

Every planner returns *raw* model text.  Callers parse it with
:func:`security.dag.parse_planner_output`; nothing here trusts or executes the
output.  API keys are read from the environment at call time and are never
logged or included in error messages.
"""

from __future__ import annotations

import inspect
import json
import os
import re
from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

import httpx

from security.dag import parse_planner_output

DEFAULT_REFUSAL = '{"kind":"refuse","reason":"out_of_scope"}'

_WHITESPACE_RE = re.compile(r"\s+")


class PlannerError(Exception):
    """A planner could not produce a response."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


@runtime_checkable
class Planner(Protocol):
    """Anything that turns a system/user prompt into raw model text."""

    def plan(self, system: str, user: str) -> str:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------- HTTP API


# Qwen3 served by vLLM thinks by default; the planner is trained to answer JSON directly.
NO_THINKING: dict[str, object] = {"chat_template_kwargs": {"enable_thinking": False}}


class OpenAICompatiblePlanner:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint.

    Works with local vLLM (a fine-tuned Qwen), DeepSeek and DashScope.  The key
    is read from ``api_key_env`` on every call; if that variable is unset no
    Authorization header is sent.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        timeout: float = 30.0,
        temperature: float = 0.0,
        supports_response_format: bool = True,
        extra_body: dict[str, object] | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.temperature = temperature
        self.supports_response_format = bool(supports_response_format)
        # Server-specific fields, e.g. vLLM + Qwen3:
        # {"chat_template_kwargs": {"enable_thinking": False}}
        self.extra_body = dict(extra_body or {})

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    def _payload(self, system: str, user: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
        }
        if self.supports_response_format:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.extra_body)
        return payload

    @staticmethod
    def _content(data: object) -> str:
        try:
            choices = data["choices"]  # type: ignore[index]
            message = choices[0]["message"]  # type: ignore[index]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PlannerError("bad_response", "missing choices[0].message.content") from exc

        if isinstance(content, str):
            return content
        # Some providers return a list of content parts.
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            if parts:
                return "".join(parts)
        raise PlannerError("bad_response", "message content is not text")

    def plan(self, system: str, user: str) -> str:
        payload = self._payload(system, user)
        try:
            response = httpx.post(
                self.endpoint,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            # Never include headers (which may hold the key) in the error.
            raise PlannerError("http", type(exc).__name__) from exc
        except ValueError as exc:
            raise PlannerError("bad_response", "response is not JSON") from exc
        return self._content(data)


# ---------------------------------------------------------------- fallback chain


class FallbackPlanner:
    """Try ``primary`` once, then ``backup`` if it failed or returned bad JSON."""

    def __init__(self, primary: Planner, backup: Planner) -> None:
        self.primary = primary
        self.backup = backup
        self.last_source: str | None = None
        self.last_planner: Planner | None = None
        self.last_error: Exception | None = None

    def _try(self, planner: Planner, system: str, user: str) -> str:
        text = planner.plan(system, user)
        # A syntactically valid but unknown output must still fail over.
        parse_planner_output(text)
        return text

    def plan(self, system: str, user: str) -> str:
        try:
            text = self._try(self.primary, system, user)
        except Exception as primary_error:
            self.last_error = primary_error
            text = self._try(self.backup, system, user)
            self.last_source = "backup"
            self.last_planner = self.backup
            return text

        self.last_source = "primary"
        self.last_planner = self.primary
        self.last_error = None
        return text


# --------------------------------------------------------------------- offline


def _normalise(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(text)).strip().casefold()


def _user_line(text: str) -> str:
    """Return the ``User:`` line of a rendered planner input, if present."""

    match = re.search(r"(?:^|\n)User:\s*", str(text))
    return str(text)[match.end() :] if match is not None else str(text)


def _call_mapping(mapping: Callable, system: str, user: str) -> str:
    """Call a user supplied mapping as ``(user)`` or ``(system, user)``."""

    try:
        signature = inspect.signature(mapping)
    except (TypeError, ValueError):
        signature = None

    positional = 0
    if signature is not None:
        positional = sum(
            1
            for parameter in signature.parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        if any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        ):
            positional = max(positional, 2)

    if positional >= 2:
        return mapping(system, user)
    return mapping(user)


class ScriptedPlanner:
    """Offline/demo planner backed by canned responses.

    A mapping is matched against the ``User:`` line of the rendered input (or
    the whole string when there is no such line).  The longest matching key
    wins; without a match the planner refuses, so a demo can never invent an
    action.
    """

    def __init__(self, mapping: Mapping[str, str] | Callable) -> None:
        self._mapping = mapping
        self.last_source = "scripted"

    def _lookup(self, system: str, user: str) -> str:
        if callable(self._mapping):
            return str(_call_mapping(self._mapping, system, user))

        haystack = _normalise(_user_line(user))
        best_key: str | None = None
        best_value: str | None = None
        for key, value in self._mapping.items():
            needle = _normalise(key)
            if not needle or needle not in haystack:
                continue
            if best_key is None or len(needle) > len(_normalise(best_key)):
                best_key = key
                best_value = value
        if best_value is None:
            return DEFAULT_REFUSAL
        if isinstance(best_value, str):
            return best_value
        return json.dumps(best_value, ensure_ascii=False)

    def plan(self, system: str, user: str) -> str:
        return self._lookup(system, user)


# ------------------------------------------------------------------ environment


def make_planner_from_env(
    environ: Mapping[str, str] | None = None,
) -> Planner:
    """Build a planner from ``ANBAN_PLANNER`` and related environment values.

    ``scripted`` (the default) never touches the network.  Errors mention the
    variable names only, never their values.
    """

    env = os.environ if environ is None else environ
    choice = (env.get("ANBAN_PLANNER") or "scripted").strip().lower()

    if choice in {"scripted", "offline", "demo"}:
        return ScriptedPlanner({})

    if choice == "local":
        base_url = env.get("ANBAN_LOCAL_URL") or "http://localhost:8000/v1"
        model = env.get("ANBAN_LOCAL_MODEL") or "anban-planner"
        key_env = env.get("ANBAN_LOCAL_API_KEY_ENV") or None
        return OpenAICompatiblePlanner(base_url, model, key_env, extra_body=NO_THINKING)

    if choice in {"deepseek", "dashscope"}:
        key_env = "DEEPSEEK_API_KEY" if choice == "deepseek" else "DASHSCOPE_API_KEY"
        if not env.get(key_env):
            raise PlannerError(
                "missing_api_key",
                f"{key_env} is not set",
            )
        if choice == "deepseek":
            base_url = env.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
            model = env.get("DEEPSEEK_MODEL") or "deepseek-chat"
        else:
            base_url = (
                env.get("DASHSCOPE_BASE_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            model = env.get("DASHSCOPE_MODEL") or "qwen-plus"
        return OpenAICompatiblePlanner(base_url, model, key_env)

    raise ValueError(
        "ANBAN_PLANNER must be one of scripted, local, deepseek (got "
        f"{choice!r})"
    )


def make_fallback_planner_from_env(
    environ: Mapping[str, str] | None = None,
) -> Planner | None:
    """General-purpose planner used when the fine-tuned one cannot handle a
    request (``ANBAN_FALLBACK_PLANNER=deepseek``); ``None`` when unset.

    It sees exactly the same tokenized input as the primary planner, and its
    plans go through the same validator, policy and approvals.
    """

    env = os.environ if environ is None else environ
    choice = (env.get("ANBAN_FALLBACK_PLANNER") or "").strip().lower()
    if not choice or choice == "none":
        return None
    return make_planner_from_env({**env, "ANBAN_PLANNER": choice})
