#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap phase for CSC-SWE pipeline.

New architecture:
1. Run coding agent 3 times WITHOUT intention (direct problem solving)
2. Evaluate each patch with SWE-bench harness
3. Extract intentions from patches/traces using LLM
4. Use extracted intentions as anchors for CMA-ES
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from csc_swe_loop.intent_client import CSCClient
from csc_swe_loop.mini_runner import run_mini_on_instance
from csc_swe_loop.swebench_utils import get_dataset_path, get_effective_split, get_harness_dataset_name, load_swebench_instance
from csc_swe_loop.swebench_eval import evaluate_patch, extract_instance_result
from csc_swe_loop.scoring import score_from_instance_result


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


def run_bootstrap_attempts(
    instance: Dict[str, Any],
    model_name: str,
    config_path: Path,
    environment_class: str,
    num_attempts: int = 3,
    max_steps: Optional[int] = None,
) -> List[Tuple[str, str, float]]:
    """Run N attempts without intention, return list of (patch, trace, temperature)."""
    attempts = []
    
    for i in range(num_attempts):
        temperature = 0.3 + (i * 0.3)  # 0.3, 0.6, 0.9 for diversity
        print(f"\n[ATTEMPT {i}] Running agent (temp={temperature}, max_steps={max_steps})...")
        
        try:
            result = run_mini_on_instance(
                instance=instance,
                intention=None,  # No intention - direct problem solving
                model_name=model_name,
                config_path=config_path,
                environment_class=environment_class,
                exit_immediately=True,
                capture_trace=True,
                max_steps=max_steps,
            )
            
            if isinstance(result, tuple):
                patch, trace = result
            else:
                patch, trace = result, ""
                
            attempts.append((patch, trace, temperature))
            print(f"[ATTEMPT {i}] Got patch: {len(patch)} chars")
            
        except Exception as e:
            print(f"[ATTEMPT {i}] Failed: {e}")
            attempts.append(("", f"Error: {e}", temperature))
    
    return attempts


def main():
    ap = argparse.ArgumentParser(description="Bootstrap: 3 attempts → extract intentions → init CMA")
    ap.add_argument("--csc", required=True, help="CSC server URL, e.g. http://127.0.0.1:8000")
    ap.add_argument("--subset", default="bash_only", help="bash_only (Verified), lite, verified, or dataset path")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--instance_id", required=True)
    ap.add_argument("--mini_model", required=True)
    ap.add_argument("--mini_config", default=str(Path(__file__).parent / "csc_swe_loop" / "swebench_minimal.yaml"))
    ap.add_argument("--environment_class", default="docker", help="docker, singularity, or apptainer (set MSWEA_SINGULARITY_EXECUTABLE=apptainer if needed)")
    ap.add_argument("--dataset_name", default=None, help="harness dataset (default: from subset)")
    ap.add_argument("--out", default="bootstrap_runs")
    ap.add_argument("--k_after", type=int, default=8, help="popsize after bootstrap")
    ap.add_argument("--num_attempts", type=int, default=3, help="number of bootstrap attempts")
    ap.add_argument("--max_steps", type=int, default=30, help="max steps per agent attempt (0 = no limit)")
    args = ap.parse_args()
    dataset_name = args.dataset_name or get_harness_dataset_name(args.subset)
    # Harness needs effective split (Verified has only "test", not "dev")
    harness_split = get_effective_split(get_dataset_path(args.subset), args.split)

    out_dir = Path(args.out) / args.instance_id
    eval_dir = out_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load SWE-bench instance
    print(f"\n{'='*60}")
    print(f"BOOTSTRAP: {args.instance_id}")
    print(f"{'='*60}")
    
    instance = load_swebench_instance(args.subset, args.split, args.instance_id)
    task_context = instance["problem_statement"]

    # 2) Run attempts one-by-one; evaluate each; stop as soon as one resolves
    print(f"\n--- Phase 1 & 2: Run attempts (no intention), evaluate, stop if solved ---")
    max_steps = args.max_steps if args.max_steps > 0 else None
    evaluated_attempts: List[Dict[str, Any]] = []

    for i in range(args.num_attempts):
        temperature = 0.3 + (i * 0.3)
        print(f"\n[ATTEMPT {i}] Running agent (temp={temperature}, max_steps={max_steps})...")
        try:
            result = run_mini_on_instance(
                instance=instance,
                intention=None,
                model_name=args.mini_model,
                config_path=Path(args.mini_config),
                environment_class=args.environment_class,
                exit_immediately=True,
                capture_trace=True,
                max_steps=max_steps,
            )
            patch, trace = (result[0], result[1]) if isinstance(result, tuple) else (result, "")
        except Exception as e:
            print(f"[ATTEMPT {i}] Failed: {e}")
            patch, trace = "", f"Error: {e}"

        run_id = f"attempt{i}_{int(time.time())}"
        if not patch:
            score, info = 0.0, {"resolved": False, "f2p": 0.0, "p2p": 0.0, "note": "empty patch"}
        else:
            try:
                results = evaluate_patch(
                    instance_id=args.instance_id,
                    patch=patch,
                    dataset_name=dataset_name,
                    out_dir=eval_dir,
                    run_id=run_id,
                    split=harness_split,
                )
                inst_res = extract_instance_result(results, args.instance_id)
                if inst_res is None:
                    score, info = 0.0, {"resolved": False, "f2p": 0.0, "p2p": 0.0, "note": "no instance result"}
                else:
                    score, info = score_from_instance_result(inst_res)
            except Exception as e:
                score, info = 0.0, {"resolved": False, "f2p": 0.0, "p2p": 0.0, "note": f"eval error: {e}"}

        evaluated_attempts.append({
            "idx": i,
            "patch": patch,
            "trace": trace,
            "temperature": temperature,
            "score": float(score),
            "info": info,
        })
        print(f"[ATTEMPT {i}] score={score:.3f} f2p={info.get('f2p')} p2p={info.get('p2p')} resolved={info.get('resolved')}")

        if info.get("resolved"):
            print("✅ Solved during bootstrap attempt! Stopping further attempts.")
            summary = {
                "instance_id": args.instance_id,
                "solved_in_bootstrap": True,
                "solving_attempt": i,
                "attempts": evaluated_attempts,
            }
            (out_dir / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return

    # 3) Extract intentions from patches/traces using LLM
    print(f"\n--- Phase 3: Extracting intentions from solutions ---")
    anchor_data: List[Dict[str, Any]] = []
    
    for attempt in evaluated_attempts:
        print(f"\n[EXTRACT {attempt['idx']}] Extracting intention from attempt {attempt['idx']}...")
        print(f"  Patch length: {len(attempt['patch'])} chars")
        print(f"  Score: {attempt['score']:.3f} (f2p={attempt['info'].get('f2p', 0.0):.3f}, p2p={attempt['info'].get('p2p', 0.0):.3f})")
        print(f"  Resolved: {attempt['info'].get('resolved', False)}")
        
        intention = extract_intention_from_solution(
            problem_statement=task_context,
            patch=attempt["patch"],
            trace=attempt["trace"],
            model_name=args.mini_model,
            attempt_idx=attempt["idx"],
        )
        
        print(f"[EXTRACT {attempt['idx']}] Extracted intention ({len(intention)} chars):")
        print(f"  {intention[:200]}..." if len(intention) > 200 else f"  {intention}")
        
        anchor_data.append({
            "intention": intention,
            "score": attempt["score"],
            "info": attempt["info"],
            "from_attempt": attempt["idx"],
        })

    # 5) Add hardcoded diversity anchors with score=0.5 to increase exploration
    # These provide additional points in embedding space with moderate weight
    hardcoded_anchors = [
        {
            "intention": """Add input validation and error handling to prevent edge cases.
""",
            "score": 0.5,
            "info": {"resolved": False, "f2p": 0.0, "p2p": 0.0, "note": "hardcoded_diversity_anchor"},
            "from_attempt": "hardcoded_1",
        },
        {
            "intention": """Refactor code structure to improve maintainability and fix logic flow.
""",
            "score": 0.5,
            "info": {"resolved": False, "f2p": 0.0, "p2p": 0.0, "note": "hardcoded_diversity_anchor"},
            "from_attempt": "hardcoded_2",
        },
    ]
    
    # Add hardcoded anchors to anchor_data
    anchor_data.extend(hardcoded_anchors)
    print(f"[INFO] Added {len(hardcoded_anchors)} hardcoded diversity anchors with score=0.5")

    # 6) Compute weights from scores
    eps = 0.05
    raw_w = [max(eps, a["score"]) for a in anchor_data]
    Z = sum(raw_w) if sum(raw_w) > 0 else 1.0
    weights = [w / Z for w in raw_w]

    anchors_payload = []
    print(f"\n--- Summary: {len(anchor_data)} anchors prepared ---")
    for idx, (anchor, w) in enumerate(zip(anchor_data, weights)):
        anchors_payload.append({
            "id": f"boot{idx}",
            "text": anchor["intention"],
            "weight": float(w),
        })
        from_attempt = anchor.get("from_attempt", "")
        anchor_type = "hardcoded" if isinstance(from_attempt, str) and from_attempt.startswith("hardcoded") else "extracted"
        print(f"\n[ANCHOR {idx}] {anchor_type} weight={w:.3f} score={anchor['score']:.3f}")
        print(f"  From attempt: {from_attempt}")
        print(f"  Intention ({len(anchor['intention'])} chars):")
        # Show first 300 chars of intention
        intention_preview = anchor['intention'][:300] + "..." if len(anchor['intention']) > 300 else anchor['intention']
        intention_lines = intention_preview.split('\n')
        for line in intention_lines[:10]:  # Show first 10 lines
            print(f"    {line}")
        anchor_lines = anchor['intention'].split('\n')
        if len(anchor_lines) > 10:
            remaining = len(anchor_lines) - 10
            print(f"    ... ({remaining} more lines)")

    # 7) PUT task to CSC
    print(f"\n--- Phase 4: Initializing GA ---")
    ga_cfg = {
        "dim": None,
        "seed": 123,
        "pop_size": 15,
        "elite_frac": 0.4,
        "min_pop_to_crossover": 4,
        "sigma": 1.0,  # Mutation noise in embedding space
        
        # Lambda sampling - three modes:
        "interp_lo": 0.4,      # Interpolation: local refinement
        "interp_hi": 0.6,
        "extrap_lo": 3.0,      # Extrapolation: aggressive exploration
        "extrap_hi": 6.0,
        "extrap2_lo": 6.0,     # Super-extrapolation: paper peak ~10
        "extrap2_hi": 10.0,
        "mod_extrap_lo": 1.2,  # Moderate extrapolation: buffer zone
        "mod_extrap_hi": 2.0,
        
        # Schedule: start with high extrapolation, decay to more interpolation
        "extrap_prob_start": 0.9,  # 90% extrapolation at start
        "extrap_prob_end": 0.2,    # 20% extrapolation at end
        "extrap_decay_iters": 200, # decay over 200 iterations
        "super_extrap_frac": 0.3,  # 30% of extrap uses [6,10] instead of [3,6]
        
        "max_banned": 10000,
        "max_seen_intentions": 10000,
        "ban_threshold": 0.001,
    }

    csc = CSCClient(args.csc)
    resp = csc.set_task(
        task_id=args.instance_id,
        task_context=task_context,
        anchors=anchors_payload,
        config=ga_cfg,
        use_refinement=False,  # Disabled for speed
        intention_prompt=None,
        algorithm="ga",
    )

    # 8) Save summary
    summary = {
        "instance_id": args.instance_id,
        "solved_in_bootstrap": False,
        "attempts": [
            {
                "idx": a["idx"],
                "score": a["score"],
                "info": a["info"],
                "patch_len": len(a["patch"]),
            }
            for a in evaluated_attempts
        ],
        "anchors": [
            {
                "text": a["intention"][:500] + "..." if len(a["intention"]) > 500 else a["intention"],
                "score": a["score"],
                "weight": w,
                "from_attempt": a["from_attempt"],
            }
            for a, w in zip(anchor_data, weights)
        ],
        "csc_put_resp": resp,
    }
    (out_dir / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    print(f"\n{'='*60}")
    print(f"✅ Bootstrap complete. Anchors extracted from {len(attempts)} attempts.")
    print(f"   Best attempt score: {max(a['score'] for a in evaluated_attempts):.3f}")
    print(f"   Summary: {out_dir / 'bootstrap_summary.json'}")
    print(f"   Ready for CMA loop: python cma_loop.py --instance_id {args.instance_id} ...")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
