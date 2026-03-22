#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from csc_swe_loop.intent_client import CSCClient
from csc_swe_loop.mini_runner import run_mini_on_instance
from csc_swe_loop.swebench_utils import get_dataset_path, get_effective_split, get_harness_dataset_name, load_swebench_instance
from csc_swe_loop.swebench_eval import evaluate_patch, extract_instance_result
from csc_swe_loop.scoring import score_from_instance_result


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csc", required=True, help="http://127.0.0.1:8000")
    ap.add_argument("--subset", default="bash_only", help="bash_only (Verified), lite, verified, or dataset path")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--instance_id", required=True)
    ap.add_argument("--mini_model", required=True)
    ap.add_argument("--mini_config", default=str(Path(__file__).parent / "csc_swe_loop" / "swebench_minimal.yaml"))
    ap.add_argument("--environment_class", default="docker", help="docker, singularity, or apptainer (set MSWEA_SINGULARITY_EXECUTABLE=apptainer if needed)")
    ap.add_argument("--dataset_name", default=None, help="harness dataset (default: from subset)")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="csc_runs")
    ap.add_argument("--cache", action="store_true", help="cache intention->score to avoid re-eval")
    ap.add_argument("--max_steps", type=int, default=25, help="max steps per agent attempt (0 = no limit)")
    args = ap.parse_args()
    dataset_name = args.dataset_name or get_harness_dataset_name(args.subset)
    # Harness needs effective split (Verified has only "test", not "dev")
    harness_split = get_effective_split(get_dataset_path(args.subset), args.split)

    out_dir = Path(args.out) / args.instance_id
    eval_dir = out_dir / "eval"
    intentions_log_path = out_dir / "intentions_log.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    instance = load_swebench_instance(args.subset, args.split, args.instance_id)
    csc = CSCClient(args.csc)

    cache_path = out_dir / "cache.json"
    cache: Dict[str, Any] = {}
    if args.cache and cache_path.exists():
        try:
            content = cache_path.read_text(encoding="utf-8").strip()
            if content:  # Only parse if file has content
                cache = json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARNING] Failed to load cache from {cache_path}: {e}")
            print(f"[WARNING] Starting with empty cache")
            cache = {}

    best = {"score": -1.0}

    for r in range(args.rounds):
        # GA generates diverse points in embedding space via crossover/mutation
        batch_id, candidates = csc.suggest(task_id=args.instance_id, n=args.k, do_sample=False)
        print(f"\n[r={r}] Got {len(candidates)} candidates from GA")
        evaluations: List[Dict[str, Any]] = []

        for i, c in enumerate(candidates):
            intent = c.intention
            key = sha1(intent)
            with open(intentions_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"round": r, "idx": i, "intention": intent, "key": key}, ensure_ascii=False) + "\n")
            print(f"[r={r} i={i}] Intention ({len(intent)} chars): {intent[:100]}..." if len(intent) > 100 else f"[r={r} i={i}] Intention: {intent}")

            if args.cache and key in cache:
                score = float(cache[key]["score"])
                info = dict(cache[key]["info"])
                evaluations.append(
                    {
                        "candidate_id": c.candidate_id,
                        "score": score,
                        "aux": {"round": r, "idx": i, "cached": True, "intention": intent, **info},
                    }
                )
                print(f"[r={r} i={i}] CACHED score={score:.3f} f2p={info.get('f2p')} p2p={info.get('p2p')} resolved={info.get('resolved')}")
                continue

            run_id = f"r{r}_i{i}_{int(time.time())}"

            max_steps = args.max_steps if args.max_steps > 0 else None
            patch = run_mini_on_instance(
                instance=instance,
                intention=intent,
                model_name=args.mini_model,
                config_path=Path(args.mini_config),
                environment_class=args.environment_class,
                exit_immediately=True,
                max_steps=max_steps,
            )

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
                        print(f"[EVAL r={r} i={i}] extract_instance_result returned None")
                        print(f"[EVAL r={r} i={i}] Results keys: {list(results.keys())}")
                        score, info = 0.0, {"resolved": False, "f2p": 0.0, "p2p": 0.0, "note": "no instance result"}
                    else:
                        score, info = score_from_instance_result(inst_res)
                        print(f"[EVAL r={r} i={i}] Extracted result: resolved={inst_res.get('resolved')}, f2p={inst_res.get('f2p')}, p2p={inst_res.get('p2p')}, keys={list(inst_res.keys())}")
                except Exception as e:
                    print(f"[EVAL r={r} i={i}] Exception during evaluation: {e}")
                    score, info = 0.0, {"resolved": False, "f2p": 0.0, "p2p": 0.0, "note": f"eval_exception:{type(e).__name__}"}

            if args.cache:
                cache[key] = {"score": float(score), "info": info}
                cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

            evaluations.append(
                {
                    "candidate_id": c.candidate_id,
                    "score": float(score),
                    "aux": {"round": r, "idx": i, "cached": False, "intention": intent, **info},
                }
            )

            if float(score) > best.get("score", -1.0):
                best = {"score": float(score), "round": r, "idx": i, "intention": intent, "info": info}

            print(f"[r={r} i={i}] score={float(score):.3f} f2p={info.get('f2p')} p2p={info.get('p2p')} resolved={info.get('resolved')}")

            if info.get("resolved"):
                csc.feedback(task_id=args.instance_id, batch_id=batch_id, evaluations=evaluations, maximize=True)
                print("✅ SOLVED. Best:", best)
                return

        resp = csc.feedback(task_id=args.instance_id, batch_id=batch_id, evaluations=evaluations, maximize=True)
        if resp:
            server_best = (resp.get("best") or {}).get("score")
            print(f"[tell] updated={resp.get('updated')} server_best={server_best} running_best={best}")
        else:
            print(f"[tell] resp=None (CSC server issue?) running_best={best}")

    print("Stopped after max rounds. Best:", best)


if __name__ == "__main__":
    main()
