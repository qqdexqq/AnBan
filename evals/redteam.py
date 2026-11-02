"""Red-team table: fire every attack case and show which rule stopped it.

    python -m evals.redteam                  # compromised-planner attacks, offline
    ANBAN_PLANNER=deepseek python -m evals.redteam --planner env   # all attacks
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.planner import make_planner_from_env
from evals.runner import DEFAULT_CASES, load_cases, run_case, select_cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--planner", choices=["env", "none"], default="none")
    args = parser.parse_args(argv)

    attacks = [case for case in load_cases(args.cases) if case["kind"] != "functional"]
    cases, skipped = select_cases(attacks, args.planner, None, None)
    planner = make_planner_from_env() if args.planner == "env" else None

    rows = ["| case | attack | expected layer | stopped by | money moved | result |", "|---|---|---|---|---|---|"]
    passed = 0
    for case in cases:
        result = run_case(case, planner)
        passed += result["passed"]
        rows.append(
            f"| {case['id']} | {case['title']} | {case.get('layer', '—')} | "
            f"{result['blocked_by']} | ¥{result['money_out_yuan']:g} | "
            f"{'PASS' if result['passed'] else 'FAIL: ' + '; '.join(result['failures'])} |"
        )
    print("\n".join(rows))
    print(f"\n{passed}/{len(cases)} attacks stopped as expected.")
    if skipped:
        print(f"{len(skipped)} text attacks need a real planner: --planner env")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
