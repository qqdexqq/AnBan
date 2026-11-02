"""The independent timer: due orders and held transfers run without a request."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from agent.planner import ScriptedPlanner
from api import scheduler as scheduler_module
from api.app import create_app
from api.scheduler import BackgroundTicker, tick
from bank.api import Bank

NOW = datetime(2026, 10, 17, 14, 0, 0)
SECRET = b"k" * 32
DAY_OF_MONTH = 20


class FixedClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now += timedelta(minutes=minutes)


def _refuse(user: str) -> str:
    return json.dumps({"kind": "refuse", "reason": "out_of_scope"})


@pytest.fixture
def app_and_clock(monkeypatch):
    # No background thread: the tests drive tick() themselves unless stated.
    monkeypatch.setenv("ANBAN_SCHEDULER_INTERVAL_S", "0")
    clock = FixedClock(NOW)
    app = create_app(
        secret_key=SECRET,
        planner=ScriptedPlanner(_refuse),
        clock=clock,
        demo_mode=True,
    )
    try:
        yield app, clock
    finally:
        app.state.runtime.conn.close()


def _customer(runtime) -> str:
    return runtime.states["young_pro"].customer.id


def _contact(conn, customer_id: str):
    return conn.execute(
        "SELECT id FROM contacts WHERE customer_id = ? ORDER BY id LIMIT 1",
        (customer_id,),
    ).fetchone()


def _transfer_count(conn, customer_id: str, scheduled_id: str) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM transfers "
            "WHERE customer_id = ? AND idempotency_key LIKE ?",
            (customer_id, f"standing_order:{scheduled_id}:%"),
        ).fetchone()[0]
    )


def test_tick_pays_a_due_order_without_any_request(app_and_clock) -> None:
    app, clock = app_and_clock
    runtime = app.state.runtime
    customer_id = _customer(runtime)
    contact = _contact(runtime.conn, customer_id)
    scheduled = Bank(runtime.conn, customer_id).schedule_transfer(
        contact["id"], 1_000, DAY_OF_MONTH, "rent"
    )

    # The first pass only records first_seen, exactly like the request path.
    assert tick(runtime) == 0
    assert _transfer_count(runtime.conn, customer_id, scheduled.id) == 0

    clock.advance(4 * 24 * 60)  # past the 20th
    assert tick(runtime) == 1
    assert _transfer_count(runtime.conn, customer_id, scheduled.id) == 1

    # A later tick is a no-op: the occurrence is paid only once.
    assert tick(runtime) == 0
    assert _transfer_count(runtime.conn, customer_id, scheduled.id) == 1

    runs = [
        entry
        for entry in runtime.services.audit.entries_for(customer_id)
        if entry.kind == "RUN"
    ]
    assert len(runs) == 1
    assert runs[0].session_id == "standing-orders"
    assert runs[0].plan_id.startswith(f"so:{scheduled.id}:")
    assert runs[0].payload["tool"] == "standing_order"
    assert runs[0].payload["status"] == "paid"
    assert runtime.services.audit.verify().ok


def test_tick_releases_a_held_transfer_past_its_hold(app_and_clock) -> None:
    app, clock = app_and_clock
    runtime = app.state.runtime
    customer_id = _customer(runtime)
    contact = _contact(runtime.conn, customer_id)
    transfer = Bank(runtime.conn, customer_id).transfer(
        contact["id"], 1_000, "test", "timer:hold:1", clock()
    )

    assert transfer.status == "pending"
    clock.advance(3 * 60)  # default cooling is two hours

    assert tick(runtime) == 0

    row = runtime.conn.execute(
        "SELECT status FROM transfers WHERE customer_id = ? AND id = ?",
        (customer_id, transfer.id),
    ).fetchone()
    assert row["status"] == "released"
    assert runtime.services.audit.verify().ok


def test_one_failing_customer_does_not_stop_another_order(
    app_and_clock, monkeypatch
) -> None:
    app, clock = app_and_clock
    runtime = app.state.runtime
    good = _customer(runtime)
    bad = next(customer for customer in runtime.customer_ids if customer != good)
    contact = _contact(runtime.conn, good)
    scheduled = Bank(runtime.conn, good).schedule_transfer(
        contact["id"], 1_000, DAY_OF_MONTH, "rent"
    )

    assert tick(runtime) == 0
    clock.advance(4 * 24 * 60)

    original = scheduler_module.run_and_audit

    def flaky(conn, audit, customer_id, now):
        if customer_id == bad:
            raise RuntimeError("boom")
        return original(conn, audit, customer_id, now)

    monkeypatch.setattr(scheduler_module, "run_and_audit", flaky)

    assert tick(runtime) == 1
    assert _transfer_count(runtime.conn, good, scheduled.id) == 1
    assert runtime.services.audit.verify().ok


def test_zero_interval_starts_no_thread(monkeypatch) -> None:
    monkeypatch.setenv("ANBAN_SCHEDULER_INTERVAL_S", "0")
    app = create_app(
        secret_key=SECRET,
        planner=ScriptedPlanner(_refuse),
        clock=FixedClock(NOW),
        demo_mode=True,
    )
    try:
        with TestClient(app) as client:
            assert app.state.scheduler is None
            assert client.get("/health").status_code == 200
        assert not any(
            thread.name == "anban-scheduler" and thread.is_alive()
            for thread in threading.enumerate()
        )
    finally:
        app.state.runtime.conn.close()


def test_small_interval_starts_on_startup_and_stops_on_shutdown(monkeypatch) -> None:
    monkeypatch.setenv("ANBAN_SCHEDULER_INTERVAL_S", "0.05")
    app = create_app(
        secret_key=SECRET,
        planner=ScriptedPlanner(_refuse),
        clock=FixedClock(NOW),
        demo_mode=True,
    )
    try:
        with TestClient(app):
            scheduler = app.state.scheduler
            assert isinstance(scheduler, BackgroundTicker)
            thread = scheduler._thread
            assert thread is not None and thread.is_alive()
            time.sleep(0.02)
        thread.join(timeout=1)
        assert not thread.is_alive()
    finally:
        app.state.runtime.conn.close()
