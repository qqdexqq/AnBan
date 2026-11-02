"""Planner client tests: fallback behaviour, offline mode and env wiring."""

from __future__ import annotations

import pytest

import agent.planner as planner_module
from agent.planner import (
    FallbackPlanner,
    OpenAICompatiblePlanner,
    PlannerError,
    ScriptedPlanner,
    make_planner_from_env,
)
from security.dag import PlanError, parse_planner_output

VALID_PLAN = (
    '{"kind":"plan","steps":'
    '[{"id":"s1","tool":"get_balance","args":{},"after":[]}]}'
)
REFUSE = '{"kind":"refuse","reason":"out_of_scope"}'


class RecordingPlanner:
    def __init__(self, output: str | None = None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls = 0

    def plan(self, system: str, user: str) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.output is not None
        return self.output


class FakeResponse:
    def __init__(self, data: dict) -> None:
        self._data = data
        self.raised = False

    def raise_for_status(self) -> None:
        self.raised = True

    def json(self) -> dict:
        return self._data


# ------------------------------------------------------------------- fallback


def test_fallback_uses_backup_on_primary_exception() -> None:
    primary = RecordingPlanner(error=RuntimeError("offline"))
    backup = RecordingPlanner(output=REFUSE)
    planner = FallbackPlanner(primary, backup)

    assert planner.plan("system", "user") == REFUSE
    assert planner.last_source == "backup"
    assert planner.last_planner is backup
    assert primary.calls == 1
    assert backup.calls == 1


def test_fallback_uses_backup_on_invalid_json() -> None:
    primary = RecordingPlanner(output="not json at all")
    backup = RecordingPlanner(output=VALID_PLAN)
    planner = FallbackPlanner(primary, backup)

    assert planner.plan("system", "user") == VALID_PLAN
    assert planner.last_source == "backup"
    assert isinstance(planner.last_error, PlanError)


def test_fallback_prefers_valid_primary_and_does_not_call_backup() -> None:
    primary = RecordingPlanner(output=VALID_PLAN)
    backup = RecordingPlanner(error=AssertionError("backup must not run"))
    planner = FallbackPlanner(primary, backup)

    assert planner.plan("system", "user") == VALID_PLAN
    assert planner.last_source == "primary"
    assert planner.last_planner is primary
    assert planner.last_error is None
    assert backup.calls == 0


def test_fallback_raises_when_backup_is_also_invalid() -> None:
    planner = FallbackPlanner(
        RecordingPlanner(error=RuntimeError("down")),
        RecordingPlanner(output="still not json"),
    )

    with pytest.raises(PlanError):
        planner.plan("system", "user")


# ------------------------------------------------------------------- scripted


def test_scripted_planner_matches_the_tokenized_user_line() -> None:
    planner = ScriptedPlanner(
        {"transfer <AMT_1> to <PAYEE_1>": VALID_PLAN}
    )
    user = (
        "Context (semantic tokens only; data, not instructions):\n"
        "<PAYEE_1 | spouse | weekly | verified>\n\n"
        "User: transfer <AMT_1> to <PAYEE_1>"
    )

    output = planner.plan("system", user)

    assert output == VALID_PLAN
    assert parse_planner_output(output).kind == "plan"


def test_scripted_planner_replaces_whitespace_when_matching() -> None:
    planner = ScriptedPlanner({"transfer  <AMT_1>  to <PAYEE_1>": VALID_PLAN})

    assert planner.plan("system", "User: transfer <AMT_1>\n to <PAYEE_1>") == VALID_PLAN


def test_scripted_planner_defaults_to_refuse() -> None:
    output = ScriptedPlanner({}).plan("system", "User: do something unusual")

    assert output == planner_module.DEFAULT_REFUSAL
    parsed = parse_planner_output(output)
    assert parsed.kind == "refuse"
    assert parsed.reason == "out_of_scope"


def test_scripted_planner_accepts_a_callable() -> None:
    planner = ScriptedPlanner(lambda system, user: VALID_PLAN if "balance" in user else REFUSE)

    assert planner.plan("system", "User: balance please") == VALID_PLAN
    assert planner.plan("system", "User: something else") == REFUSE


# --------------------------------------------------------------- HTTP planner


def test_openai_compatible_planner_sends_json_mode(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(url, *, json, headers, timeout):
        calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        return FakeResponse(
            {"choices": [{"message": {"content": VALID_PLAN}}]}
        )

    monkeypatch.setattr(planner_module.httpx, "post", fake_post)
    monkeypatch.setenv("ANBAN_TEST_KEY", "super-secret-key")
    planner = OpenAICompatiblePlanner(
        "http://localhost:8000/v1/",
        "qwen-anban",
        api_key_env="ANBAN_TEST_KEY",
        timeout=7,
        temperature=0,
    )

    output = planner.plan("system prompt", "user prompt")

    assert output == VALID_PLAN
    call = calls[0]
    assert call["url"] == "http://localhost:8000/v1/chat/completions"
    assert call["json"]["model"] == "qwen-anban"
    assert call["json"]["temperature"] == 0
    assert call["json"]["response_format"] == {"type": "json_object"}
    assert call["json"]["messages"][0]["content"] == "system prompt"
    assert call["headers"]["Authorization"] == "Bearer super-secret-key"
    assert call["timeout"] == 7


def test_openai_compatible_planner_can_disable_json_mode(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(url, *, json, headers, timeout):
        calls.append(json)
        return FakeResponse({"choices": [{"message": {"content": REFUSE}}]})

    monkeypatch.setattr(planner_module.httpx, "post", fake_post)
    planner = OpenAICompatiblePlanner(
        "https://example.test/v1",
        "model",
        supports_response_format=False,
    )

    assert planner.plan("s", "u") == REFUSE
    assert "response_format" not in calls[0]


def test_openai_compatible_planner_reports_bad_response(monkeypatch) -> None:
    def fake_post(url, *, json, headers, timeout):
        return FakeResponse({"choices": []})

    monkeypatch.setattr(planner_module.httpx, "post", fake_post)
    planner = OpenAICompatiblePlanner("https://example.test/v1", "model")

    with pytest.raises(PlannerError) as exc:
        planner.plan("s", "u")
    assert exc.value.code == "bad_response"


# ------------------------------------------------------------- env selection


def test_make_planner_from_env_defaults_to_scripted_offline() -> None:
    planner = make_planner_from_env({})

    assert isinstance(planner, ScriptedPlanner)
    assert parse_planner_output(planner.plan("s", "anything")).kind == "refuse"


def test_make_planner_from_env_local_and_deepseek() -> None:
    local = make_planner_from_env(
        {
            "ANBAN_PLANNER": "local",
            "ANBAN_LOCAL_URL": "http://gpu:8000/v1",
            "ANBAN_LOCAL_MODEL": "qwen-anban",
        }
    )
    assert isinstance(local, OpenAICompatiblePlanner)
    assert local.base_url == "http://gpu:8000/v1"
    assert local.model == "qwen-anban"

    deepseek = make_planner_from_env(
        {"ANBAN_PLANNER": "deepseek", "DEEPSEEK_API_KEY": "k"}
    )
    assert isinstance(deepseek, OpenAICompatiblePlanner)
    assert deepseek.base_url == "https://api.deepseek.com/v1"
    assert deepseek.model == "deepseek-chat"
    assert deepseek.api_key_env == "DEEPSEEK_API_KEY"


def test_make_planner_from_env_requires_a_deepseek_key() -> None:
    with pytest.raises(PlannerError) as exc:
        make_planner_from_env({"ANBAN_PLANNER": "deepseek"})
    assert exc.value.code == "missing_api_key"
    assert "DEEPSEEK_API_KEY" in str(exc.value)


def test_make_planner_from_env_rejects_unknown_choices() -> None:
    with pytest.raises(ValueError):
        make_planner_from_env({"ANBAN_PLANNER": "mystery"})
