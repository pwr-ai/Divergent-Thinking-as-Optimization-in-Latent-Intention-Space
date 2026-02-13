from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx


@dataclass
class Candidate:
    candidate_id: str
    intention: str
    intention_raw: str
    meta: Dict[str, Any]
    z: List[float]


class CSCClient:
    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def set_task(
        self,
        task_id: str,
        task_context: str,
        anchors: List[Dict[str, Any]],
        config: Dict[str, Any],
        use_refinement: bool = False,
        intention_prompt: Optional[str] = None,
        # Legacy CMA params (ignored for GA/Tabu)
        novelty_lambda: float = 0.0,
        novelty_mode: str = "batch_distance",
        # Algorithm type: "cma", "ga", or "tabu"
        algorithm: str = "ga",
        # Initial solution for Tabu Search (from bootstrap)
        initial_solution: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        task_payload = {
            "task_context": task_context,
            "anchors": anchors,
            "use_refinement": use_refinement,
            "intention_prompt": intention_prompt,
        }
        
        if algorithm == "tabu":
            # Tabu Search mode
            if initial_solution:
                task_payload["initial_solution"] = initial_solution
            payload = {"task": task_payload, "tabu": config}
        elif algorithm == "ga":
            payload = {"task": task_payload, "ga": config}
        else:
            # Legacy CMA mode
            task_payload["novelty_lambda"] = novelty_lambda
            task_payload["novelty_mode"] = novelty_mode
            payload = {"task": task_payload, "cma": config}
        r = httpx.put(f"{self.base_url}/tasks/{task_id}", json=payload, timeout=self.timeout)
        if r.status_code == 404:
            try:
                error_detail = r.text
            except Exception:
                error_detail = "No error details available"
            raise httpx.HTTPStatusError(
                f"404 Not Found for {self.base_url}/tasks/{task_id}. "
                f"Server may not be running or endpoint not available. "
                f"Error: {error_detail}",
                request=r.request,
                response=r,
            )
        if r.status_code == 400:
            try:
                err_body = r.json()
                detail = err_body.get("detail", r.text)
            except Exception:
                detail = r.text
            raise httpx.HTTPStatusError(
                f"400 Bad Request for {self.base_url}/tasks/{task_id}: {detail}",
                request=r.request,
                response=r,
            )
        r.raise_for_status()
        return r.json()

    def suggest(
        self,
        task_id: str,
        n: int = 8,
        max_new_tokens: int = 120,
        do_sample: bool = True,
    ) -> Tuple[str, List[Candidate]]:
        payload = {"n": n, "max_new_tokens": max_new_tokens, "do_sample": do_sample}
        r = httpx.post(f"{self.base_url}/tasks/{task_id}/suggest", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        batch_id = data["batch_id"]
        cands: List[Candidate] = []
        for c in data["candidates"]:
            cands.append(
                Candidate(
                    candidate_id=c["candidate_id"],
                    intention=c["intention"],
                    intention_raw=c["intention_raw"],
                    meta=c.get("meta", {}),
                    z=c.get("z", []),
                )
            )
        return batch_id, cands

    def feedback(
        self,
        task_id: str,
        batch_id: str,
        evaluations: List[Dict[str, Any]],
        maximize: bool = True,
    ) -> Dict[str, Any]:
        payload = {"batch_id": batch_id, "evaluations": evaluations, "maximize": maximize}
        r = httpx.post(f"{self.base_url}/tasks/{task_id}/feedback", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
