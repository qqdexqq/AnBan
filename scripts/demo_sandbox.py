"""Demonstrate S6 against a fresh in-memory bank; never open the live database."""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.graph import AnBanAgent, Services
from agent.planner import ScriptedPlanner
from bank import db
from bank.seed import generate
from security.audit import AuditLog, create_audit_schema
from security.mandate import MandateService, create_security_schema
from security.policy import PolicyConfig
from security.tokens import TokenVault


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="Compare my weekend spending with weekdays and explain the difference")
    parser.add_argument("--dev", action="store_true", help="Explicitly use an unisolated development worker")
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    args = parser.parse_args()
    if args.dev:
        os.environ["ANBAN_SANDBOX_DEV"] = "1"
    elif os.environ.get("ANBAN_SANDBOX_DEV") == "1":
        parser.error("Unset ANBAN_SANDBOX_DEV for production, or explicitly pass --dev")
    # A deterministic injected clock and synthetic bank make the demo reproducible.
    today = date(2026, 10, 17)
    now = datetime(2026, 10, 17, 12)
    secret = b"s6-isolated-demo-only-not-a-production-key"
    state = generate("young_pro", 42, today)
    conn = db.connect(":memory:")
    try:
        db.create_schema(conn)
        db.load_state(conn, state)
        create_security_schema(conn)
        create_audit_schema(conn)
        audit = AuditLog(conn, secret)
        services = Services(
            conn=conn, vault=TokenVault(secret),
            mandates=MandateService(conn, secret, otp_rng=random.Random(7)),
            audit=audit, planner=ScriptedPlanner({}), policy_config=PolicyConfig(),
            secret_key=secret, now=lambda: now,
        )
        result = AnBanAgent(services).start(state.customer.id, args.question, lang=args.lang)
        print(result.reply)
        successful = False
        for row in conn.execute("SELECT payload_json FROM security_audit WHERE kind='SANDBOX' ORDER BY seq"):
            event = json.loads(row[0])
            successful = successful or event["ok"]
            print(json.dumps({key: event[key] for key in (
                "source", "backend", "ok", "error", "runtime_ms", "code_sha256", "row_count"
            )}))
        verified = audit.verify().ok
        print(json.dumps({"status": result.status, "audit_chain_ok": verified}))
        return 0 if result.status == "done" and verified and successful else 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
