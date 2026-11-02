"""Privacy, deterministic computations, and code-proposal fallback tests."""

from datetime import date, datetime
from contextlib import asynccontextmanager
import asyncio
from types import SimpleNamespace

import httpx
import pytest

from agent.analysis_data import analysis_rows
from agent import codegen
from bank.api import Bank
from bank.db import connect, create_schema, load_state
from bank.seed import generate
from security.sandbox import check_code


@pytest.fixture(autouse=True)
def clean_codegen_environment(monkeypatch):
    for name in ("ANBAN_CODEGEN_URL", "ANBAN_CODEGEN_MODEL", "ANBAN_CODEGEN_API_KEY", "ANBAN_LOCAL_URL", "ANBAN_LOCAL_MODEL"):
        monkeypatch.delenv(name, raising=False)


def _compute(kind, rows):
    # Reference templates only: model proposals are never executed in-process.
    namespace = {}
    exec(codegen.analysis_template(kind), namespace)
    return namespace["analyze"](rows)


def _row(day, weekday, amount, category="food", hour=12):
    return {"day": day, "weekday": weekday, "hour": hour, "amount_fen": amount, "category": category}


@pytest.mark.parametrize("kind", codegen.TEMPLATES)
def test_templates_pass_static_check_and_handle_empty_rows(kind):
    assert not check_code(codegen.analysis_template(kind))
    output = _compute(kind, [])
    assert output["row_count"] == 0
    assert all(not isinstance(value, dict) for value in output.values())


def test_weekend_totals_spending_day_denominators_and_category_difference():
    rows = [_row("2026-10-03", 5, 1001), _row("2026-10-03", 5, 499, "shopping"),
            _row("2026-10-04", 6, 1000), _row("2026-10-05", 0, 501)]
    output = _compute("weekend_vs_weekday", rows)
    assert output["weekend_total_fen"] == 2500
    assert output["weekday_total_fen"] == 501
    assert output["weekend_days"] == 2
    assert output["weekday_days"] == 1
    assert output["weekend_average_fen"] == 1250
    assert output["weekday_average_fen"] == 501
    assert output["difference_fen"] == 749
    assert output["categories"] == ["food", "shopping"]
    assert output["category_differences_fens"] == [500, 250]
    assert output["largest_category"] == "food"


def test_hour_buckets_exact_boundaries():
    rows = [_row("2026-10-05", 0, hour + 1, hour=hour) for hour in (0, 5, 6, 11, 12, 17, 18, 23)]
    output = _compute("by_hour", rows)
    assert output["bucket_totals_fens"] == [7, 19, 31, 43]
    assert output["bucket_counts"] == [2, 2, 2, 2]
    assert output["peak_bucket"] == 3
    assert output["total_fen"] == 100


def test_weekday_averages_round_half_up_and_use_distinct_dates():
    output = _compute("by_weekday", [_row("2026-10-05", 0, 100),
                                    _row("2026-10-12", 0, 101),
                                    _row("2026-10-12", 0, 100),
                                    _row("2026-10-06", 1, 400)])
    assert output["weekday_totals_fens"] == [301, 400, 0, 0, 0, 0, 0]
    assert output["weekday_days"] == [2, 1, 0, 0, 0, 0, 0]
    assert output["weekday_averages_fens"] == [151, 400, 0, 0, 0, 0, 0]
    assert output["busiest_weekday"] == "Tuesday"


def test_categories_are_capped_without_discarding_total_spending():
    from bank.categories import CATEGORIES
    rows = [_row("2026-10-03", 5, index + 1, category) for index, category in enumerate(CATEGORIES)]
    output = _compute("weekend_vs_weekday", rows)
    assert len(output["categories"]) == 12
    assert output["weekend_total_fen"] == sum(range(1, len(CATEGORIES) + 1))


def test_minimal_projection_excludes_income_and_own_savings():
    observed = []
    def transactions(start, end, *, limit):
        observed.append((start, end, limit))
        return [SimpleNamespace(amount_fen=amount, category=category, counterparty=party,
                                ts=datetime(2026, 10, 3, 17, 35), memo="secret", account_id="private")
                for amount, category, party in [(-1200, "food", "private person"),
                                                (9000, "income", "private employer"),
                                                (-5000, "transfer", "我的储蓄账户"),
                                                (-600, "unrecognized", "unknown private name")]]
    rows = analysis_rows(SimpleNamespace(list_transactions=transactions), date(2026, 10, 1), date(2026, 10, 31))
    assert observed == [(date(2026, 10, 1), date(2026, 10, 31), None)]
    assert rows == [_row("2026-10-03", 5, 1200, hour=17), _row("2026-10-03", 5, 600, "other", 17)]
    assert set(rows[0]) == {"day", "weekday", "hour", "amount_fen", "category"}


def test_seeded_projection_matches_bank_total_without_writes():
    today = date(2026, 11, 2)
    state = generate("young_pro", 42, today)
    conn = connect()
    try:
        create_schema(conn)
        load_state(conn, state)
        bank = Bank(conn, state.customer.id)
        before = conn.total_changes
        rows = analysis_rows(bank, date(2026, 10, 1), date(2026, 10, 31))
        assert sum(row["amount_fen"] for row in rows) == bank.spending_report(date(2026, 10, 1), date(2026, 10, 31))["total"]
        assert conn.total_changes == before
        assert rows
    finally:
        conn.close()


@pytest.mark.parametrize("text,kind", [("周末和工作日花钱有什么不同", "weekend_vs_weekday"),
                                      ("Compare weekend spending with weekdays", "weekend_vs_weekday"),
                                      ("What time of day do I spend", "by_hour"),
                                      ("几点花钱最多", "by_hour"),
                                      ("Which weekday has the most spending", "by_weekday"),
                                      ("custom analysis", "by_weekday")])
def test_offline_codegen_uses_labelled_template(text, kind):
    assert codegen.generate_analysis_code(text, lang="en") == (codegen.TEMPLATES[kind], "template")


def _install_client(monkeypatch, content=None, error=None):
    captured = {}
    class Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        def post(self, url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            if error:
                raise error
            return httpx.Response(200, request=httpx.Request("POST", url), json={"choices": [{"message": {"content": content}}]})
        @asynccontextmanager
        async def stream(self, method, url, **kwargs):
            response = self.post(url, **kwargs)
            # Real streaming responses have not already consumed their body.
            streamed = httpx.Response(response.status_code, request=response.request,
                                      stream=httpx.ByteStream(response.content))
            try:
                yield streamed
            finally:
                await streamed.aclose()
    monkeypatch.setattr(codegen.httpx, "AsyncClient", Client)
    monkeypatch.setenv("ANBAN_LOCAL_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("ANBAN_LOCAL_MODEL", "existing-model")
    return captured


def test_model_request_schema_only_no_original_question_or_proxy_environment(monkeypatch):
    code = 'def analyze(rows):\n    return {"row_count": len(rows)}'
    captured = _install_client(monkeypatch, "```python\n" + code + "\n```")
    result = codegen.generate_analysis_code("Compare weekend spending for Alice 13800138000, balance 123456.78, 2026-11-02", lang="en")
    assert result == (code, "model")
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["client"] == {"timeout": httpx.Timeout(2.0, connect=3.0), "trust_env": False,
                                   "follow_redirects": False, "headers": {"Accept-Encoding": "identity"}}
    payload = captured["json"]
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 600
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["model"] == "existing-model"
    transmitted = str(payload["messages"])
    for value in ("Alice", "13800138000", "123456.78", "2026-11-02"):
        assert value not in transmitted


@pytest.mark.parametrize("content", ["import os\ndef analyze(rows):\n    return {}", "not python", None, "", {"code": "x"}])
def test_invalid_model_response_falls_back(monkeypatch, content):
    _install_client(monkeypatch, content)
    assert codegen.generate_analysis_code("weekend", lang="zh") == (codegen.TEMPLATES["weekend_vs_weekday"], "template")


def test_endpoint_failure_falls_back_without_error_echo(monkeypatch):
    _install_client(monkeypatch, error=httpx.ConnectError("secret endpoint error"))
    assert codegen.generate_analysis_code("weekday", lang="en") == (codegen.TEMPLATES["by_weekday"], "template")


def test_specific_codegen_configuration_overrides_planner(monkeypatch):
    captured = _install_client(monkeypatch, 'def analyze(rows):\n    return {}')
    monkeypatch.setenv("ANBAN_CODEGEN_URL", "http://127.0.0.1:9000/v1/chat/completions")
    monkeypatch.setenv("ANBAN_CODEGEN_MODEL", "code-model")
    codegen.generate_analysis_code("weekend", lang="en")
    assert captured["url"] == "http://127.0.0.1:9000/v1/chat/completions"
    assert captured["json"]["model"] == "code-model"


def test_codegen_credential_only_sent_to_explicit_codegen_endpoint(monkeypatch):
    captured = _install_client(monkeypatch, 'def analyze(rows):\n    return {}')
    monkeypatch.setenv("ANBAN_CODEGEN_URL", "https://codegen.example.invalid/v1")
    monkeypatch.setenv("ANBAN_CODEGEN_API_KEY", "unit-test-secret")
    code, source = codegen.generate_analysis_code("weekend", lang="en")
    assert source == "model"
    assert captured["headers"] == {"Authorization": "Bearer unit-test-secret"}
    assert "unit-test-secret" not in str(captured["json"])
    assert "unit-test-secret" not in code


def test_codegen_credential_never_forwarded_to_planner_fallback(monkeypatch):
    captured = _install_client(monkeypatch, 'def analyze(rows):\n    return {}')
    monkeypatch.setenv("ANBAN_CODEGEN_API_KEY", "unit-test-secret")
    codegen.generate_analysis_code("weekend", lang="en")
    assert "headers" not in captured


def test_provider_failure_does_not_return_credentials(monkeypatch):
    captured = _install_client(monkeypatch, error=httpx.ConnectError("unit-test-secret"))
    monkeypatch.setenv("ANBAN_CODEGEN_URL", "https://codegen.example.invalid/v1")
    monkeypatch.setenv("ANBAN_CODEGEN_API_KEY", "unit-test-secret")
    code, source = codegen.generate_analysis_code("weekend", lang="en")
    assert source == "template"
    assert "unit-test-secret" not in code
    assert captured["headers"] == {"Authorization": "Bearer unit-test-secret"}


def test_provider_header_deadline_cancels_before_response(monkeypatch):
    events = []

    async def headers(request):
        try:
            events.append("started")
            await asyncio.sleep(0.05)
            events.append("completed")
            return httpx.Response(200, json={"choices": []})
        finally:
            events.append("finished")

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(headers)
    monkeypatch.setattr(codegen.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
    monkeypatch.setattr(codegen, "_REQUEST_SECONDS", 0.01)
    monkeypatch.setenv("ANBAN_CODEGEN_URL", "https://synthetic.invalid/v1")
    assert codegen.generate_analysis_code("weekend", lang="en") == (codegen.TEMPLATES["weekend_vs_weekday"], "template")
    assert events == ["started", "finished"]


def test_running_event_loop_returns_template_without_creating_coroutine(monkeypatch):
    monkeypatch.setenv("ANBAN_CODEGEN_URL", "https://synthetic.invalid/v1")
    monkeypatch.setattr(codegen, "_bounded_content", lambda *a, **kw: pytest.fail("coroutine created inside running loop"))

    async def caller():
        return codegen.generate_analysis_code("weekend", lang="en")

    assert asyncio.run(caller()) == (codegen.TEMPLATES["weekend_vs_weekday"], "template")
