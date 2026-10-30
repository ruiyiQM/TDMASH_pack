#!/usr/bin/env python3
"""Check retrieved structure_*/mash_output.npz files and count hop outcomes.

Example: python examples/check_mash_runs.py runs/ensemble --steps 10000 --expected-runs 100
Exit nonzero for missing, unreadable, incomplete, or legacy outputs without status.
"""
import argparse
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--expected-runs", type=int, default=100)
    args = parser.parse_args()
    runs = sorted(path for path in args.root.glob("structure_*") if path.is_dir())
    failed = len(runs) != args.expected_runs
    if failed:
        print(f"Directory count mismatch: found {len(runs)}, expected {args.expected_runs}")
    print("trajectory status completed_steps accepted frustrated zero_nac")
    for run in runs:
        filename = run / "mash_output.npz"
        try:
            with np.load(filename, allow_pickle=False) as data:
                required = {"time", "hop_events", "completed", "completed_steps", "requested_steps", "failure_reason"}
                if not required.issubset(data.files):
                    raise ValueError("missing completion/hop metadata; legacy output cannot be verified")
                events = data["hop_events"]
                outcomes = events["outcome"]
                if not np.isin(outcomes, ["accepted", "frustrated", "zero_nac"]).all():
                    raise ValueError("unknown hop outcome")
                counts = [np.count_nonzero(outcomes == status)
                          for status in ("accepted", "frustrated", "zero_nac")]
                steps = len(data["time"])
                reason = str(data["failure_reason"].item())
                complete = (bool(data["completed"].item()) and not reason
                            and steps == int(data["completed_steps"].item()) == args.steps
                            and int(data["requested_steps"].item()) == args.steps
                            and counts[2] == 0)
                status = "ok" if complete else reason or "incomplete"
                failed |= not complete
                print(run.name, status, steps, *counts)
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
            failed = True
            print(run.name, "invalid/missing", str(exc))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
