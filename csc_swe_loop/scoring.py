from __future__ import annotations

from typing import Any, Dict, Tuple


def score_from_instance_result(inst: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Extract soft score for CMA-ES from per-instance harness output."""
    resolved = bool(inst.get("resolved", False))

    f2p = inst.get("fail_to_pass_ratio", inst.get("f2p"))
    p2p = inst.get("pass_to_pass_ratio", inst.get("p2p"))

    # compute from counts if available
    if f2p is None:
        passed = inst.get("fail_to_pass_passed")
        total = inst.get("fail_to_pass_total")
        if isinstance(passed, int) and isinstance(total, int) and total > 0:
            f2p = passed / total

    if p2p is None:
        passed = inst.get("pass_to_pass_passed")
        total = inst.get("pass_to_pass_total")
        if isinstance(passed, int) and isinstance(total, int) and total > 0:
            p2p = passed / total

    # fallback
    if f2p is None:
        f2p = 1.0 if resolved else 0.0
    if p2p is None:
        p2p = 1.0 if resolved else 0.0

    f2p = float(f2p)
    p2p = float(p2p)
    # Changed from f2p * p2p to f2p + (1/60) * p2p to avoid zeroing score
    # This provides better gradient for CMA-ES even when f2p=0.0
    # f2p is weighted much more (primary goal: fix failing tests)
    # p2p provides small bonus/penalty for regressions (secondary goal: don't break existing tests)
    score = f2p + (1.0 / 60.0) * p2p

    resolved = bool(resolved or (f2p == 1.0 and p2p == 1.0))
    info = {"resolved": resolved, "f2p": f2p, "p2p": p2p}
    return score, info
