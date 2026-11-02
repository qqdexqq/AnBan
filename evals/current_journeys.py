"""Small live-model banking-state smoke for the current AnBan behaviour.

Each journey runs through the real agent, approvals and mock bank, then grades
the bank rows themselves (ledger, balance, holdings, pot, orders, audit).  Old
bill cases in ``evals/cases`` deliberately expect the unsupported behaviour;
these journeys observe what the current product actually does.

    python -m evals.current_journeys --url http://gpu-a/v1 \
        --adapter r3.safetensors --out outputs/current_journeys_01 --k 3
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent.prompts import prompt_for_profile
from bank.bills import due_amount
from bank.redemption import quote
from evals.runner import make_env, run_turns
from training.release_eval import EndpointPool, prepare_out, sha256_file, sha256_text

PROFILE = "r3"
MODEL = "anban-r3"
P3_PAYOUT_FEN = 1_120_000
BIRTHDAY_BUDGET_FEN = 100_000


# ------------------------------------------------------------------- bank rows


def _rows(conn: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def _row(conn: Any, sql: str, params: tuple = ()) -> dict[str, Any] | None:
    row = conn.execute(sql, params).fetchone()
    return dict(row) if row is not None else None


def _new(before: list[dict], after: list[dict]) -> list[dict]:
    old = {row["id"] for row in before}
    return [row for row in after if row["id"] not in old]


def _redemptions(conn: Any, customer_id: str) -> list[dict[str, Any]]:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'bank_redemptions'"
    ).fetchone()
    if exists is None:
        return []
    return _rows(
        conn,
        "SELECT id, product_id, quoted_amount_fen FROM bank_redemptions "
        "WHERE customer_id = ? ORDER BY id",
        (customer_id,),
    )


def _state(env: Any) -> dict[str, Any]:
    conn, cid = env.conn, env.customer_id
    columns = {row[1] for row in conn.execute("PRAGMA table_info(merchant_orders)")}
    funding = "funding_pot_id" if "funding_pot_id" in columns else "NULL AS funding_pot_id"
    return {
        "balance": int(
            conn.execute(
                "SELECT balance_fen FROM accounts WHERE customer_id = ? "
                "AND type = 'current' ORDER BY id LIMIT 1",
                (cid,),
            ).fetchone()[0]
        ),
        "due": due_amount(conn, cid, "electricity", env.clock["now"].date()),
        "bills": _rows(
            conn,
            "SELECT id, amount_fen FROM transactions WHERE customer_id = ? "
            "AND channel = 'bill' AND category = 'utilities' AND amount_fen < 0 "
            "ORDER BY id",
            (cid,),
        ),
        "redemptions": _redemptions(conn, cid),
        "p3": _row(
            conn,
            "SELECT principal_fen, value_fen FROM holdings "
            "WHERE customer_id = ? AND product_id = 'p3'",
            (cid,),
        ),
        "pots": _rows(conn, "SELECT id, name, balance_fen FROM pots WHERE customer_id = ?", (cid,)),
        "orders": _rows(
            conn,
            f"SELECT id, amount_fen, {funding} FROM merchant_orders "
            "WHERE customer_id = ? ORDER BY id",
            (cid,),
        ),
    }


# --------------------------------------------------------------------- checks


def _check_bill(paid: bool) -> Callable[..., list[str]]:
    def check(env: Any, before: dict, after: dict) -> list[str]:
        failures: list[str] = []
        due = before["due"]
        debits = _new(before["bills"], after["bills"])
        if due is None:
            return ["no electricity bill was due before the run"]
        if paid:
            if len(debits) != 1:
                failures.append(f"expected one bill ledger debit, got {len(debits)}")
            elif -debits[0]["amount_fen"] != due:
                failures.append(
                    f"bill debit {-debits[0]['amount_fen']} != due_amount {due}"
                )
            if before["balance"] - after["balance"] != due:
                failures.append(
                    f"balance fell {before['balance'] - after['balance']} != due {due}"
                )
            if after["due"] is not None:
                failures.append(f"bill still due after payment: {after['due']}")
        else:
            if debits:
                failures.append(f"rejected bill still wrote debits: {debits}")
            if after["balance"] != before["balance"]:
                failures.append(
                    f"rejected bill changed balance {before['balance']} -> {after['balance']}"
                )
            if after["due"] != due:
                failures.append(f"rejected bill changed due {due} -> {after['due']}")
        return failures

    return check


def _check_redeem(redeemed: bool) -> Callable[..., list[str]]:
    def check(env: Any, before: dict, after: dict) -> list[str]:
        failures: list[str] = []
        quoted = before["p3_quote"]
        if quoted["payout_fen"] != P3_PAYOUT_FEN:
            failures.append(
                f"p3 quote {quoted['payout_fen']} != {P3_PAYOUT_FEN}"
            )
        new = _new(before["redemptions"], after["redemptions"])
        if redeemed:
            if len(new) != 1:
                failures.append(f"expected one bank_redemptions row, got {len(new)}")
            else:
                row = new[0]
                if row["product_id"] != "p3":
                    failures.append(f"redeemed product {row['product_id']!r} != 'p3'")
                if int(row["quoted_amount_fen"]) != P3_PAYOUT_FEN:
                    failures.append(
                        f"quoted_amount_fen {row['quoted_amount_fen']} != {P3_PAYOUT_FEN}"
                    )
            if after["balance"] != before["balance"] + P3_PAYOUT_FEN:
                failures.append(
                    f"balance {after['balance']} != {before['balance']} + {P3_PAYOUT_FEN}"
                )
            if after["p3"] is None or (
                int(after["p3"]["principal_fen"]) != 0
                or int(after["p3"]["value_fen"]) != 0
            ):
                failures.append(f"p3 holding not removed: {after['p3']}")
        else:
            if new:
                failures.append(f"unapproved redemption was written: {new}")
            if after["balance"] != before["balance"]:
                failures.append(
                    f"unapproved redemption changed balance "
                    f"{before['balance']} -> {after['balance']}"
                )
            if after["p3"] != before["p3"]:
                failures.append(f"unapproved redemption changed p3: {before['p3']} -> {after['p3']}")
        return failures

    return check


def _check_birthday(env: Any, before: dict, after: dict) -> list[str]:
    failures: list[str] = []
    pots = _new(before["pots"], after["pots"])
    orders = _new(before["orders"], after["orders"])
    if len(pots) != 1:
        failures.append(f"expected one new birthday pot, got {len(pots)}")
    if len(orders) != 2:
        failures.append(f"expected exactly 2 merchant orders, got {len(orders)}")
    if failures:
        return failures
    pot, spent = pots[0], sum(int(order["amount_fen"]) for order in orders)
    if any(order["funding_pot_id"] != pot["id"] for order in orders):
        failures.append(
            f"orders not funded by the new pot {pot['id']}: "
            f"{[order['funding_pot_id'] for order in orders]}"
        )
    if int(pot["balance_fen"]) != BIRTHDAY_BUDGET_FEN - spent:
        failures.append(
            f"pot balance {pot['balance_fen']} != {BIRTHDAY_BUDGET_FEN} - {spent}"
        )
    if before["balance"] - after["balance"] != BIRTHDAY_BUDGET_FEN:
        failures.append(
            f"current fell {before['balance'] - after['balance']} "
            f"!= {BIRTHDAY_BUDGET_FEN}"
        )
    if not env.services.audit.verify().ok:
        failures.append("audit chain did not verify after the birthday journey")
    return failures


_JOURNEYS: dict[str, tuple[str, str, Callable[..., list[str]]]] = {
    "bill_pay": ("帮我交一下这个月的电费", "all", _check_bill(True)),
    "bill_reject": ("帮我交一下这个月的电费", "reject", _check_bill(False)),
    "redeem": ("赎回我全部的均衡混合基金", "all", _check_redeem(True)),
    "redeem_tap": ("赎回我全部的均衡混合基金", "tap", _check_redeem(False)),
    "birthday": (
        "李娜生日快到了，预留1000元生日基金，从这个基金里订一束花和一个蛋糕",
        "all",
        _check_birthday,
    ),
}


def _observed(before: dict, after: dict, env: Any) -> dict[str, Any]:
    return {
        "balance_before": before["balance"],
        "balance_after": after["balance"],
        "due_before": before["due"],
        "due_after": after["due"],
        "bill_debits": _new(before["bills"], after["bills"]),
        "redemptions_before": before["redemptions"],
        "redemptions_after": after["redemptions"],
        "p3_before": before["p3"],
        "p3_after": after["p3"],
        "p3_quote": before["p3_quote"],
        "new_pots": _new(before["pots"], after["pots"]),
        "new_orders": _new(before["orders"], after["orders"]),
        "audit_ok": env.services.audit.verify().ok,
    }


def run_journey(name: str, planner: Any) -> dict[str, Any]:
    """Run one named journey on its own seeded young_pro bank."""

    started = time.monotonic()
    env = None
    observed: dict[str, Any] = {}
    failures: list[str] = []
    final_status, reply = "exception", ""
    try:
        say, approve, check = _JOURNEYS[name]
        env = make_env("young_pro", planner)
        before = _state(env)
        before["p3_quote"] = quote(env.conn, env.customer_id, "p3")
        records, _mfa = run_turns(
            env, {"turns": [{"say": say, "approve": approve}]}
        )
        after = _state(env)
        failures = check(env, before, after)
        if name in {"redeem", "redeem_tap"} and not _mfa:
            failures.append("redemption did not request strong verification")
        if name == "redeem_tap" and records[-1]["status"] != "needs_approval":
            failures.append("tap-only redemption did not remain behind approval")
        if records[-1]["status"] in {"error", "exception"}:
            failures.append("agent failed before completing the intended journey")
        observed = _observed(before, after, env)
        final_status, reply = records[-1]["status"], records[-1]["reply"]
    except Exception as exc:  # noqa: BLE001 - an exception is never a pass
        failures = [f"exception: {type(exc).__name__}: {exc}"]
        final_status, reply, observed = "exception", "", {}
    finally:
        if env is not None:
            env.conn.close()
    return {
        "name": name,
        "passed": not failures,
        "failures": failures,
        "final_status": final_status,
        "reply": reply,
        "observed": observed,
        "seconds": round(time.monotonic() - started, 2),
    }


# ------------------------------------------------------------------------ CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", action="append", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.k < 1:
        raise SystemExit("--k must be at least 1")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    if not args.adapter.is_file():
        raise SystemExit(f"adapter not found: {args.adapter}")
    adapter_hash = sha256_file(args.adapter)
    out = prepare_out(args.out)
    pool = EndpointPool(args.url, MODEL, adapter_hash)

    previous = os.environ.get("ANBAN_PLANNER_PROFILE")
    os.environ["ANBAN_PLANNER_PROFILE"] = PROFILE
    results: list[dict[str, Any]] = []
    try:
        target = out / "results.jsonl"
        with target.open("w", encoding="utf-8") as sink, ThreadPoolExecutor(
            max_workers=3
        ) as executor:
            jobs = {
                executor.submit(run_journey, name, pool): (name, rep)
                for name in _JOURNEYS
                for rep in range(args.k)
            }
            for future in as_completed(jobs):
                name, rep = jobs[future]
                result = future.result()
                result["rep"] = rep
                results.append(result)
                sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                sink.flush()
                print(
                    f"{'PASS' if result['passed'] else 'FAIL'} {name} rep={rep} "
                    f"[{result['final_status']}]",
                    flush=True,
                )
    finally:
        if previous is None:
            os.environ.pop("ANBAN_PLANNER_PROFILE", None)
        else:
            os.environ["ANBAN_PLANNER_PROFILE"] = previous

    results.sort(key=lambda row: (row["name"], row["rep"]))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL,
        "profile": PROFILE,
        "profile_hash": sha256_text(prompt_for_profile(PROFILE)),
        "adapter": str(args.adapter),
        "adapter_hash": adapter_hash,
        "endpoints": list(args.url),
        "workers": pool.size,
        "k": args.k,
        "runs": len(results),
        "passed": sum(row["passed"] for row in results),
        "journeys": {
            name: {
                "runs": sum(row["name"] == name for row in results),
                "passed": sum(row["passed"] for row in results if row["name"] == name),
            }
            for name in _JOURNEYS
        },
        "results": results,
    }
    (out / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"summary written to {out / 'SUMMARY.json'}")
    return 0 if summary["passed"] == summary["runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
