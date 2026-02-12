#!/usr/bin/env python3
"""Print the maximum number of intentions needed to solve any instance.

For bootstrap-only solved: 1 + attempt_idx (1–3).
For solved in Tabu loop: total_evals from the loop.
"""
import json
import sys
from pathlib import Path
from typing import Optional


def num_intentions_to_solve(obj: dict) -> Optional[int]:
    """Return how many intentions were needed to solve, or None if not solved."""
    if not obj.get("solved"):
        return None
    boot = obj.get("bootstrap") or {}
    if boot.get("solved_in_bootstrap"):
        # Solved in bootstrap: 1-based attempt number (1, 2, or 3)
        initial = boot.get("initial_solution") or {}
        meta = initial.get("meta") or {}
        attempt_idx = meta.get("attempt_idx", 0)
        return int(attempt_idx) + 1
    loop = obj.get("loop")
    if loop and loop.get("solved"):
        return int(loop.get("total_evals", 0))
    return None


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path("dataset_runs/bash_only_dev_tabu/bash_only_dev/results.jsonl")
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)
    max_intentions = 0
    count_solved = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                n = num_intentions_to_solve(obj)
                if n is not None:
                    count_solved += 1
                    if n > max_intentions:
                        max_intentions = n
            except json.JSONDecodeError:
                continue
    print(f"Solved count: {count_solved}")
    print(f"Max number of intentions needed to solve (any instance): {max_intentions}")


if __name__ == "__main__":
    main()
