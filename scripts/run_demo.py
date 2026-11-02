"""Start the AnBan demo API.

Settings come from environment variables (see the README, "Run the demo").
With nothing set, the API runs on rules only: no model server is needed, and
requests the rules do not cover are answered with a clarifying question.

    python scripts/run_demo.py                    # in-memory data, port 8000
    python scripts/run_demo.py --db demo.sqlite3  # data and chats survive restarts

The demo routes (open persona login, clock control, reset) are switched on
here. Do not expose this server to a network you do not control.
"""
from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_secret(db: str) -> bytes | None:
    """Keep one signing key next to a file database so audit records still verify after a restart."""
    if db == ":memory:":
        return None
    path = Path(db + ".secret")
    if not path.exists():
        path.write_bytes(secrets.token_bytes(32))
    return path.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", default=":memory:", help="SQLite file, or :memory: (default)")
    parser.add_argument(
        "--origin", action="append", default=[],
        help="allowed browser origin; repeat for several (default: the Vite dev server)",
    )
    args = parser.parse_args()
    origins = tuple(args.origin) or ("http://localhost:5173", "http://127.0.0.1:5173")

    os.chdir(ROOT)
    import uvicorn

    from api.app import create_app

    planner = os.environ.get("ANBAN_PLANNER", "scripted")
    print(f"planner: {planner}")
    print(f"voice:   {os.environ.get('ANBAN_ASR_URL', 'not configured')}")
    print(f"memory:  {'on' if os.environ.get('ANBAN_TOKENIZER_PATH') else 'off (set ANBAN_TOKENIZER_PATH)'}")
    print(f"data:    {args.db}")
    app = create_app(
        db_path=args.db,
        secret_key=load_secret(args.db),
        demo_mode=True,
        allowed_origins=origins,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
