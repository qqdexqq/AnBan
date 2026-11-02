"""Standing-order stop really waits for approval through the complete graph."""
import pytest

from bank.api import Bank
from tests.test_graph import make_env


@pytest.mark.parametrize("confirm", [False, True])
def test_stop_waits_for_fresh_confirmation(make_env, confirm):
    env = make_env()
    bank = Bank(env.conn, env.customer_id)
    order = bank.schedule_transfer(bank.find_contact("")[0].id, 10_000, 5, "rent")
    result = env.agent.start(env.customer_id, "Stop my standing order")
    assert result.status == "needs_approval", result.reply
    assert len(result.approvals) == 1
    assert result.approvals[0]["tier"] == "yellow"
    assert env.conn.execute("SELECT status FROM scheduled_transfers WHERE id=?", (order.id,)).fetchone()[0] == "scheduled"
    answer = {"confirm": True} if confirm else {"reject": True}
    finished = env.agent.resume(result.session_id, env.customer_id, {"s1": answer})
    assert finished.status == "done", finished.reply
    status = env.conn.execute("SELECT status FROM scheduled_transfers WHERE id=?", (order.id,)).fetchone()[0]
    assert status == ("cancelled" if confirm else "scheduled")
