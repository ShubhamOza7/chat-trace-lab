#!/usr/bin/env python3
"""Fire the scripted conversations at the model and print their trace ids.

    python scripts/run_scenarios.py                 # all scenarios
    python scripts/run_scenarios.py single_tool     # just one
    python scripts/run_scenarios.py --list
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.scenarios import SCENARIOS, SCENARIOS_BY_NAME, run_all, run_scenario  # noqa: E402
from app.tracing import init_tracing, shutdown_tracing  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", nargs="?", help="scenario name; omit to run all")
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument("--user", default="scenario-runner")
    args = parser.parse_args()

    if args.list:
        for s in SCENARIOS:
            print(f"{s.name:22} {len(s.turns)} turn(s)  {s.description}")
        return 0

    init_tracing()

    # Say out loud whether PRISMtrace is on. It no-ops when disabled, so without
    # this a misconfiguration looks identical to a successful run.
    from app.prismtrace_setup import status as prismtrace_status  # noqa: E402

    pt = prismtrace_status()
    print(
        f"PRISMtrace: {'ON  -> ' + str(pt['host']) if pt['enabled'] else 'OFF (PRISMTRACE_API_KEY unset)'}"
    )

    try:
        if args.scenario:
            scenario = SCENARIOS_BY_NAME.get(args.scenario)
            if scenario is None:
                print(f"No scenario named {args.scenario!r}. Use --list.", file=sys.stderr)
                return 2
            runs = [run_scenario(scenario, user_id=args.user)]
        else:
            runs = run_all(user_id=args.user)

        print()
        for run in runs:
            status = f"errored ({run.error})" if run.error else "ok"
            print(f"{run.scenario:22} {status}")
            print(f"  session {run.session_id}")
            for tid in run.trace_ids:
                print(f"  trace   {tid}")
    finally:
        shutdown_tracing()

    return 0


if __name__ == "__main__":
    sys.exit(main())
