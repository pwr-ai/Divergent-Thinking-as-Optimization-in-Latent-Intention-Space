#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset runner: CSC (CMA-ES) × mini-swe-agent × SWE-bench harness

Pipeline per instance:
  1) Load SWE-bench instance (problem_statement)
  2) Run N bootstrap attempts WITHOUT intention (direct problem solving)
  3) Evaluate each attempt: patch -> swebench harness -> (f2p, p2p, score)
  4) Extract intentions from patches/traces using LLM
  5) PUT task into CSC Intention Server with extracted intention anchors
  6) Run CMA loop: suggest -> eval -> feedback for (rounds × k) budget
  7) Append per-instance summary to results.jsonl

This file is meant to run across a whole dataset split, with resume support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from datasets import load_dataset

from csc_swe_loop.intent_client import CSCClient
from csc_swe_loop.mini_runner import run_mini_on_instance
from csc_swe_loop.swebench_eval import evaluate_patch, extract_instance_result
from csc_swe_loop.scoring import score_from_instance_result
from csc_swe_loop.swebench_utils import get_dataset_path, get_effective_split, get_harness_dataset_name


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def extract_intention_from_solution(
    problem_statement: str,
    patch: str,
    trace: str,
    model_name: str,
    attempt_idx: int,
) -> str:
    """Use LLM to extract a detailed intention from a patch and agent trace."""
    import litellm

    api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8080/v1")

    system_prompt = """You are an expert software engineer. Analyze the given patch and agent trace and extract the FIX INTENTION in 2-3 short sentences.

Output ONLY the intention: what the fix aims to do and how (strategy), in plain language. No code, no pros/cons, no file paths or function names. Just the intention."""

    # Truncate for context limits
    patch_truncated = patch[:3000] if patch else "(empty patch)"
    trace_truncated = trace[:2000] if trace else "(no trace)"
    problem_truncated = problem_statement[:1500]

    user_prompt = f"""## Problem Statement
{problem_truncated}

## Patch Applied
```diff
{patch_truncated}
```

## Agent Reasoning Trace
{trace_truncated}

## Task
Extract the fix intention in 2-3 sentences. Plain language only, no code, no pros/cons."""

    try:
        response = litellm.completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=700,
            temperature=0.3,
            api_base=api_base,
        )
        intention = response.choices[0].message.content.strip()
        print(f"[EXTRACT {attempt_idx}] Extracted intention: {len(intention)} chars")
        return intention
    except Exception as e:
        print(f"[EXTRACT {attempt_idx}] Extraction failed: {e}")
        # Fallback: use the patch itself as a crude intention
        return f"Apply the following patch to fix the issue:\n\n```diff\n{patch[:1000]}\n```"


# Hardcoded diversity anchors to increase exploration
HARDCODED_DIVERSITY_ANCHORS = [
    {
        "intention": "Add input validation and error handling to prevent edge cases.",
        "score": 0.5,
        "info": {"resolved": False, "f2p": 0.0, "p2p": 0.0, "note": "hardcoded_diversity_anchor"},
    },
    {
        "intention": "Refactor code structure to improve maintainability and fix logic flow.",
        "score": 0.5,
        "info": {"resolved": False, "f2p": 0.0, "p2p": 0.0, "note": "hardcoded_diversity_anchor"},
    },
]


def safe_eval_patch(
    *,
    instance_id: str,
    patch: str,
    dataset_name: str,
    eval_dir: Path,
    run_id: str,
    split: str = "test",
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate patch with SWE-bench harness; return (score, info).
    Fail-closed: if anything goes wrong, return score=0 with reason.
    split: effective split for harness (e.g. "test" for Verified, "dev" for Lite).
    """
    if not patch or not patch.strip():
        return 0.0, {"resolved": False, "f2p": 0.0, "p2p": 0.0, "note": "empty_patch"}
    try:
        results = evaluate_patch(
            instance_id=instance_id,
            patch=patch,
            dataset_name=dataset_name,
            out_dir=eval_dir,
            run_id=run_id,
            split=split,
        )
        inst_res = extract_instance_result(results, instance_id)
        if inst_res is None:
            return 0.0, {"resolved": False, "f2p": 0.0, "p2p": 0.0, "note": "no_instance_result"}
        score, info = score_from_instance_result(inst_res)
        return float(score), dict(info)
    except Exception as e:
        return 0.0, {"resolved": False, "f2p": 0.0, "p2p": 0.0, "note": f"eval_exception:{type(e).__name__}"}


BOOTSTRAP_CACHE_FILENAME = "bootstrap_cache.json"


def _load_bootstrap_cache(out_dir: Path, num_attempts: int = 3) -> Optional[Dict[str, Any]]:
    """Load bootstrap cache if valid. Returns None if missing or invalid.
    
    New format stores 'attempts' (raw agent runs) and 'anchors' (extracted intentions).
    Old format only stored 'anchors' with hardcoded intentions - not compatible.
    """
    cache_path = out_dir / BOOTSTRAP_CACHE_FILENAME
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        # Check for new format: must have 'attempts' list
        attempts = data.get("attempts")
        if not isinstance(attempts, list) or len(attempts) < num_attempts:
            return None
        for a in attempts:
            if not isinstance(a, dict) or "patch" not in a or "trace" not in a or "score" not in a:
                return None
        # Also check anchors
        anchors = data.get("anchors")
        if not isinstance(anchors, list) or len(anchors) < 1:
            return None
        for r in anchors:
            if not isinstance(r, dict) or "text" not in r or "score" not in r or "info" not in r:
                return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def bootstrap_instance(
    *,
    instance: Dict[str, Any],
    instance_id: str,
    csc: CSCClient,
    csc_base: str,
    mini_model: str,
    mini_config: Path,
    environment_class: str,
    dataset_name: str,
    harness_split: str,
    out_dir: Path,
    k_after: int,
    use_bootstrap_cache: bool = True,
    num_attempts: int = 3,
    max_steps: int = 30,
    # Algorithm selection and Tabu-specific params
    algorithm: str = "ga",
    tabu_tenure: int = 50,
    sigma_local: float = 0.5,
    sigma_kick: float = 3.0,
    stagnation_threshold: int = 10,
    kick_probability: float = 0.15,
) -> Dict[str, Any]:
    """
    1) Run N attempts WITHOUT intention (direct problem solving with varying temps)
    2) Evaluate each attempt's patch
    3) Extract intentions from patches/traces using LLM
    4) PUT task into CSC with extracted intention anchors + diversity anchors

    Returns a dict summary, including anchors and whether solved during bootstrap.
    """
    task_context = instance["problem_statement"]
    cache_path = out_dir / BOOTSTRAP_CACHE_FILENAME
    anchor_rows: List[Dict[str, Any]] = []
    attempts_data: List[Dict[str, Any]] = []
    solved = False

    # Try loading from cache
    if use_bootstrap_cache:
        cached = _load_bootstrap_cache(out_dir, num_attempts=num_attempts)
        if cached is not None:
            anchor_rows = [dict(r) for r in cached["anchors"]]
            attempts_data = [dict(a) for a in cached["attempts"]]
            solved = bool(cached.get("solved_in_bootstrap", False))
            print(f"[INFO] Bootstrap cache hit for {instance_id}: {len(attempts_data)} attempts, {len(anchor_rows)} anchors (solved_in_bootstrap={solved})")

    if not anchor_rows:
        eval_dir = out_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Run N attempts WITHOUT intention (varying temperatures for diversity)
        print(f"[BOOTSTRAP] Phase 1: Running {num_attempts} agent attempts (no intention)")
        for j in range(num_attempts):
            temperature = 0.3 + (j * 0.3)  # 0.3, 0.6, 0.9 for diversity
            run_id = f"boot_attempt{j}_{int(time.time())}"
            print(f"[ATTEMPT {j}] Running agent (temp={temperature:.1f}, max_steps={max_steps})...")

            try:
                result = run_mini_on_instance(
                    instance=instance,
                    intention=None,  # No intention - direct problem solving
                    model_name=mini_model,
                    config_path=mini_config,
                    environment_class=environment_class,
                    exit_immediately=True,
                    capture_trace=True,
                    max_steps=max_steps if max_steps > 0 else None,
                )
                if isinstance(result, tuple):
                    patch, trace = result
                else:
                    patch, trace = result, ""
            except Exception as e:
                print(f"[ATTEMPT {j}] Failed: {e}")
                patch, trace = "", f"Error: {e}"

            # Phase 2: Evaluate patch
            if patch:
                score, info = safe_eval_patch(
                    instance_id=instance_id,
                    patch=patch,
                    dataset_name=dataset_name,
                    eval_dir=eval_dir,
                    run_id=run_id,
                    split=harness_split,
                )
            else:
                score, info = 0.0, {"resolved": False, "f2p": 0.0, "p2p": 0.0, "note": "empty_patch"}

            attempt = {
                "idx": j,
                "patch": patch,
                "trace": trace,
                "temperature": temperature,
                "score": float(score),
                "info": info,
            }
            attempts_data.append(attempt)
            print(f"[ATTEMPT {j}] patch={len(patch)} chars, score={score:.3f}, resolved={info.get('resolved')}")

            if info.get("resolved"):
                solved = True

        # Phase 3: Extract intentions from patches/traces using LLM
        print(f"[BOOTSTRAP] Phase 2: Extracting intentions from {len(attempts_data)} attempts")
        for attempt in attempts_data:
            print(f"[EXTRACT {attempt['idx']}] Extracting intention (patch={len(attempt['patch'])} chars)...")
            intention = extract_intention_from_solution(
                problem_statement=task_context,
                patch=attempt["patch"],
                trace=attempt["trace"],
                model_name=mini_model,
                attempt_idx=attempt["idx"],
            )
            anchor_rows.append({
                "text": intention,
                "score": float(attempt["score"]),
                "info": attempt["info"],
                "from_attempt": attempt["idx"],
            })

        # Phase 4: Add hardcoded diversity anchors
        for i, div_anchor in enumerate(HARDCODED_DIVERSITY_ANCHORS):
            anchor_rows.append({
                "text": div_anchor["intention"],
                "score": div_anchor["score"],
                "info": div_anchor["info"],
                "from_attempt": f"hardcoded_{i}",
            })
        print(f"[BOOTSTRAP] Added {len(HARDCODED_DIVERSITY_ANCHORS)} hardcoded diversity anchors")

        # Save bootstrap cache
        try:
            cache_path.write_text(
                json.dumps({
                    "attempts": attempts_data,
                    "anchors": anchor_rows,
                    "solved_in_bootstrap": solved,
                }, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    # If solved already, we still PUT task (useful for CSC debugging), but CMA loop can be skipped.
    eps = 0.05
    raw_w = [max(eps, float(r["score"])) for r in anchor_rows]
    Z = sum(raw_w) if sum(raw_w) > 0 else 1.0
    weights = [w / Z for w in raw_w]

    anchors_payload = []
    for idx, (r, w) in enumerate(zip(anchor_rows, weights)):
        anchors_payload.append({"id": f"boot{idx}", "text": r["text"], "weight": float(w)})

    # For Tabu Search: select the best attempt as initial solution
    initial_solution = None
    if algorithm == "tabu" and attempts_data:
        # Find the attempt with highest score
        best_attempt = max(attempts_data, key=lambda a: a["score"])
        best_anchor = next((r for r in anchor_rows if r.get("from_attempt") == best_attempt["idx"]), None)
        if best_anchor:
            initial_solution = {
                "intention": best_anchor["text"],
                "intention_raw": best_anchor["text"],
                "score": float(best_attempt["score"]),
                "aux": best_attempt.get("info", {}),
                "meta": {"from_bootstrap": True, "attempt_idx": best_attempt["idx"]},
            }
            print(f"[TABU] Selected best attempt {best_attempt['idx']} as initial solution (score={best_attempt['score']:.3f})")

    if algorithm == "tabu":
        # Tabu Search configuration
        tabu_cfg = {
            "dim": None,
            "seed": 123,
            "tabu_tenure": tabu_tenure,
            "max_tabu_size": 1000,
            "use_strict_fingerprint": False,
            "sigma_local": sigma_local,
            "sigma_kick": sigma_kick,
            "anchor_attraction": 0.3,
            "pca_bias": 0.2,
            "stagnation_threshold": stagnation_threshold,
            "kick_probability": kick_probability,
            "kick_intensity": 2.0,
            "use_aspiration": True,
            "accept_equal": True,  # Accept moves with score >= current (crucial for plateau navigation)
            "max_banned": 10000,
            "max_seen_intentions": 10000,
            "ban_threshold": 0.001,
        }
        
        put_resp = csc.set_task(
            task_id=instance_id,
            task_context=task_context,
            anchors=anchors_payload,
            config=tabu_cfg,
            use_refinement=False,
            intention_prompt=None,
            algorithm="tabu",
            initial_solution=initial_solution,
        )
    else:
        # GA configuration (default)
        ga_cfg = {
            "dim": None,
            "seed": 123,
            "pop_size": 10,
            "elite_frac": 0.6,
            "min_pop_to_crossover": 3,
            "sigma": 0.25,  # Mutation noise in embedding space
            
            # Lambda sampling - three modes:
            "interp_lo": 0.4,      # Interpolation: local refinement
            "interp_hi": 0.6,
            "extrap_lo": 3,      # Extrapolation: aggressive exploration
            "extrap_hi": 6,
            "extrap2_lo": 6,     # Super-extrapolation: paper peak ~10
            "extrap2_hi": 10,
            "mod_extrap_lo": 1.5,  # Moderate extrapolation: buffer zone
            "mod_extrap_hi": 3,
            
            # Schedule: start with high extrapolation, decay to more interpolation
            "extrap_prob_start": 0.95,  # 90% extrapolation at start
            "extrap_prob_end": 0.2,    # 20% extrapolation at end
            "extrap_decay_iters": 50, # decay over 200 iterations
            "super_extrap_frac": 0.3,  # 30% of extrap uses [6,10] instead of [3,6]
            
            "max_banned": 10000,
            "max_seen_intentions": 10000,
            "ban_threshold": 0.001,
        }

        put_resp = csc.set_task(
            task_id=instance_id,
            task_context=task_context,
            anchors=anchors_payload,
            config=ga_cfg,
            use_refinement=False,
            intention_prompt=None,
            algorithm="ga",
        )

    return {
        "instance_id": instance_id,
        "solved_in_bootstrap": bool(solved),
        "anchors": [
            {**r, "weight": float(w)} for r, w in zip(anchor_rows, weights)
        ],
        "csc_put_resp": put_resp,
        "algorithm": algorithm,
        "initial_solution": initial_solution if algorithm == "tabu" else None,
    }


def cma_loop_instance(
    *,
    instance: Dict[str, Any],
    instance_id: str,
    csc: CSCClient,
    mini_model: str,
    mini_config: Path,
    environment_class: str,
    dataset_name: str,
    harness_split: str,
    out_dir: Path,
    rounds: int,
    k: int,
    use_cache: bool,
    max_steps: Optional[int] = 100,
) -> Dict[str, Any]:
    """
    Run CMA iterations for one instance. Returns best metrics and whether solved.
    """
    eval_dir = out_dir / "eval"
    intentions_log_path = out_dir / "intentions_log.jsonl"
    eval_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "cache.json"
    cache: Dict[str, Any] = {}
    if use_cache and cache_path.exists():
        try:
            content = cache_path.read_text(encoding="utf-8").strip()
            if content:  # Only parse if file has content
                cache = json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARNING] Failed to load cache from {cache_path}: {e}")
            print(f"[WARNING] Starting with empty cache")
            cache = {}

    best = {"score": -1.0}
    solved = False
    total_evals = 0

    for r in range(rounds):
        batch_id, candidates = csc.suggest(task_id=instance_id, n=k)
        evaluations: List[Dict[str, Any]] = []

        for i, cand in enumerate(candidates):
            intent = cand.intention
            key = sha1(intent)
            try:
                with open(intentions_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"round": r, "idx": i, "intention": intent, "key": key}, ensure_ascii=False) + "\n")
            except Exception:
                pass

            if use_cache and key in cache:
                score = float(cache[key]["score"])
                info = dict(cache[key]["info"])
                evaluations.append(
                    {
                        "candidate_id": cand.candidate_id,
                        "score": score,
                        "aux": {"round": r, "idx": i, "cached": True, "intention": intent, **info},
                    }
                )
                if score > best["score"]:
                    best = {"score": score, "round": r, "idx": i, "intention": intent, "info": info}
                if info.get("resolved"):
                    solved = True
                continue

            run_id = f"r{r}_i{i}_{int(time.time())}"
            try:
                patch = run_mini_on_instance(
                    instance=instance,
                    intention=intent,
                    model_name=mini_model,
                    config_path=mini_config,
                    environment_class=environment_class,
                    exit_immediately=True,
                    max_steps=max_steps if max_steps and max_steps > 0 else None,
                )
            except Exception as e:
                patch = ""
                score, info = 0.0, {"resolved": False, "f2p": 0.0, "p2p": 0.0, "note": f"mini_exception:{type(e).__name__}"}
            else:
                score, info = safe_eval_patch(
                    instance_id=instance_id,
                    patch=patch,
                    dataset_name=dataset_name,
                    eval_dir=eval_dir,
                    run_id=run_id,
                    split=harness_split,
                )

            total_evals += 1

            if use_cache:
                cache[key] = {"score": float(score), "info": info}
                try:
                    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
                except Exception:
                    pass

            evaluations.append(
                {
                    "candidate_id": cand.candidate_id,
                    "score": float(score),
                    "aux": {"round": r, "idx": i, "cached": False, "intention": intent, **info},
                }
            )

            if float(score) > best.get("score", -1.0):
                best = {"score": float(score), "round": r, "idx": i, "intention": intent, "info": info}

            if info.get("resolved"):
                solved = True
                # Tell CSC about partial batch before exit
                csc.feedback(task_id=instance_id, batch_id=batch_id, evaluations=evaluations, maximize=True)
                return {"solved": True, "best": best, "rounds_run": r + 1, "total_evals": total_evals}

        # end-of-round tell
        csc.feedback(task_id=instance_id, batch_id=batch_id, evaluations=evaluations, maximize=True)

    return {"solved": bool(solved), "best": best, "rounds_run": rounds, "total_evals": total_evals}


def load_split(subset: str, split: str) -> Iterable[Dict[str, Any]]:
    dataset_path = get_dataset_path(subset)
    effective_split = get_effective_split(dataset_path, split)
    return load_dataset(dataset_path, split=effective_split)


def read_done_instance_ids(results_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not results_path.exists():
        return done
    for line in results_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            iid = obj.get("instance_id")
            if isinstance(iid, str) and iid:
                done.add(iid)
        except Exception:
            continue
    return done


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csc", required=True, help="CSC server base URL, e.g. http://127.0.0.1:8000")
    ap.add_argument("--subset", default="bash_only", help="subset: bash_only (Verified), lite, verified, or dataset path")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--dataset_name", default=None, help="SWE-bench harness dataset (default: from subset, e.g. Verified for bash_only)")
    ap.add_argument("--mini_model", required=True)
    ap.add_argument("--mini_config", default=str(Path(__file__).parent / "csc_swe_loop" / "swebench_minimal.yaml"))
    ap.add_argument("--environment_class", default="docker")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--k_after", type=int, default=8, help="popsize in CSC after bootstrap")
    ap.add_argument("--out", default="dataset_runs")
    ap.add_argument("--max_instances", type=int, default=0, help="0 = no limit")
    ap.add_argument("--start_at", type=int, default=0, help="skip first N instances in split")
    ap.add_argument("--resume", action="store_true", help="resume from results.jsonl")
    ap.add_argument("--cache", action="store_true", help="cache intention->score per instance")
    ap.add_argument("--no-bootstrap-cache", action="store_true", help="disable bootstrap cache: always re-run bootstrap attempts (default: use cache when bootstrap_cache.json exists)")
    ap.add_argument("--only_instance_ids", default="", help="comma-separated allowlist of instance_ids")
    ap.add_argument("--num_attempts", type=int, default=3, help="number of bootstrap attempts without intention (default: 3)")
    ap.add_argument("--max_steps", type=int, default=100, help="max steps per mini-agent run in bootstrap and CMA loop (0 = use config default, default: 100)")
    # Algorithm selection
    ap.add_argument("--algorithm", default="ga", choices=["ga", "tabu"], help="search algorithm: ga (Genetic Algorithm) or tabu (Tabu Search)")
    # Tabu Search specific parameters
    ap.add_argument("--tabu_tenure", type=int, default=50, help="how long solutions stay tabu (default: 50)")
    ap.add_argument("--sigma_local", type=float, default=0.5, help="std for local neighborhood moves (default: 0.5)")
    ap.add_argument("--sigma_kick", type=float, default=3.0, help="std for kick/diversification moves (default: 3.0)")
    ap.add_argument("--stagnation_threshold", type=int, default=10, help="iterations without improvement before kick (default: 10)")
    ap.add_argument("--kick_probability", type=float, default=0.15, help="base probability of kick move (default: 0.15)")
    args = ap.parse_args()
    dataset_name = args.dataset_name or get_harness_dataset_name(args.subset)
    # Harness needs effective split (Verified has only "test", not "dev")
    harness_split = get_effective_split(get_dataset_path(args.subset), args.split)

    out_root = Path(args.out) / f"{args.subset}_{args.split}"
    results_path = out_root / "results.jsonl"

    allow: Optional[Set[str]] = None
    if args.only_instance_ids.strip():
        allow = {x.strip() for x in args.only_instance_ids.split(",") if x.strip()}

    done: Set[str] = set()
    if args.resume:
        done = read_done_instance_ids(results_path)

    csc = CSCClient(args.csc)
    mini_config = Path(args.mini_config)

    ds = load_split(args.subset, args.split)

    if args.resume and done:
        print(f"[INFO] Resume: skipping {len(done)} already-done instance(s). Will run first instance(s) with dataset index >= {args.start_at} that are not in results.")
    else:
        print(f"[INFO] Will run instance(s) starting at dataset index {args.start_at} (first instance = index {args.start_at}).")

    count_total = 0
    count_run = 0
    for idx, inst in enumerate(ds):
        if idx < args.start_at:
            continue
        instance_id = inst.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            continue
        if allow is not None and instance_id not in allow:
            continue
        if instance_id in done:
            continue
        if args.max_instances and count_run >= args.max_instances:
            break

        count_total += 1
        print(f"[INFO] Running instance at dataset index {idx}: {instance_id}")
        inst_out = out_root / instance_id
        inst_out.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        status: Dict[str, Any] = {"instance_id": instance_id, "subset": args.subset, "split": args.split}
        try:
            boot = bootstrap_instance(
                instance=dict(inst),
                instance_id=instance_id,
                csc=csc,
                csc_base=args.csc,
                mini_model=args.mini_model,
                mini_config=mini_config,
                environment_class=args.environment_class,
                dataset_name=dataset_name,
                harness_split=harness_split,
                out_dir=inst_out,
                k_after=args.k_after,
                use_bootstrap_cache=not args.no_bootstrap_cache,
                num_attempts=args.num_attempts,
                max_steps=args.max_steps,
                # Algorithm selection and Tabu-specific params
                algorithm=args.algorithm,
                tabu_tenure=args.tabu_tenure,
                sigma_local=args.sigma_local,
                sigma_kick=args.sigma_kick,
                stagnation_threshold=args.stagnation_threshold,
                kick_probability=args.kick_probability,
            )
            status["bootstrap"] = boot

            if boot.get("solved_in_bootstrap"):
                status["solved"] = True
                status["best"] = {
                    "score": max(a["score"] for a in boot["anchors"]) if boot.get("anchors") else 1.0,
                    "info": next((a["info"] for a in boot["anchors"] if a["info"].get("resolved")), {}),
                    "intention": next((a["text"] for a in boot["anchors"] if a["info"].get("resolved")), None),
                }
            else:
                loop = cma_loop_instance(
                    instance=dict(inst),
                    instance_id=instance_id,
                    csc=csc,
                    mini_model=args.mini_model,
                    mini_config=mini_config,
                    environment_class=args.environment_class,
                    dataset_name=dataset_name,
                    harness_split=harness_split,
                    out_dir=inst_out,
                    rounds=args.rounds,
                    k=args.k,
                    use_cache=args.cache,
                    max_steps=args.max_steps,
                )
                status["loop"] = loop
                status["solved"] = bool(loop.get("solved"))
                status["best"] = loop.get("best")

        except Exception as e:
            status["error"] = f"{type(e).__name__}: {e}"
            status["solved"] = False

        status["wall_time_sec"] = round(time.time() - t0, 3)

        append_jsonl(results_path, status)
        count_run += 1
        print(f"[{count_run}] (idx={idx}) {instance_id} solved={status.get('solved')} best={status.get('best', {}).get('score') if isinstance(status.get('best'), dict) else None} time={status['wall_time_sec']}s")

    print(f"Done. ran={count_run} (start_at={args.start_at}) resume={args.resume} out={out_root}")


if __name__ == "__main__":
    main()
