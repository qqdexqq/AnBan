"""Independent timer for unattended work.

Standing orders and held transfers must not wait for a customer to open the
app.  This module owns a small daemon thread that ticks the runtime it is
given, using the same lock and clock as the request handlers.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from agent.standing_order_hook import run_and_audit

_LOG = logging.getLogger(__name__)


def tick(runtime: Any) -> int:
    """One pass: for every customer, release held transfers and run due standing orders.

    Returns the number of standing-order runs created.  A failure for one
    customer is logged and never stops the rest of the pass.
    """

    created = 0
    with runtime.lock:
        now = runtime.clock()
        for customer_id in runtime.customer_ids:
            try:
                runtime.services.bank_for(customer_id).release_due(now)
                created += len(
                    run_and_audit(
                        runtime.conn,
                        runtime.services.audit,
                        customer_id,
                        now,
                    )
                )
            except Exception:
                _LOG.warning(
                    "Scheduled tick failed for one customer; continuing",
                    exc_info=True,
                )
    return created


class BackgroundTicker:
    """Daemon thread that calls :func:`tick` every ``interval_s`` seconds."""

    def __init__(self, get_runtime: Callable[[], Any], interval_s: float) -> None:
        self._get_runtime = get_runtime
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    def start(self) -> None:
        """Start the thread; calling it again while running is a no-op."""
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="anban-scheduler",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Ask the thread to finish and wait for it (bounded)."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        # The startup catch-up already ran once; wait for the first interval so
        # a freshly started app never ticks concurrently with its first request.
        while not self._stop.wait(self._interval_s):
            try:
                tick(self._get_runtime())
            except Exception:
                _LOG.warning("Scheduled tick failed; continuing", exc_info=True)
