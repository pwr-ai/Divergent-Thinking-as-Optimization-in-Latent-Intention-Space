#!/usr/bin/env python3
import json
from pathlib import Path
import random

base_dir = Path("./dataset_runs/bash_only_dev")

results = {
    "total_runs": 0,
    "resolved_in_bootstrap": 0,
    "resolved_in_intentions": 0,
    "not_resolved": 0,
    "runs_details": []
}

# Find all run directories
run_dirs = sorted([
    d for d in base_dir.iterdir()
    if d.is_dir()
])
results["total_runs"] = len(run_dirs)

for run_dir in run_dirs:
    run_name = run_dir.name
    run_info = {
        "run": run_name,
        "resolved": False,
        "resolved_in_bootstrap": False,
        "resolved_in_intentions": False
    }

    # Check bootstrap_cache.json
    bootstrap_cache = run_dir / "bootstrap_cache.json"
    if bootstrap_cache.exists():
        try:
            with open(bootstrap_cache, "r", encoding="utf-8") as f:
                bootstrap_data = json.load(f)
            if bootstrap_data.get("solved_in_bootstrap", False):
                run_info["resolved"] = True
                run_info["resolved_in_bootstrap"] = True
                results["resolved_in_bootstrap"] += 1
        except Exception as e:
            print(f"Error reading bootstrap_cache.json for {run_name}: {e}")

    # Check cache.json (intentions)
    cache_file = run_dir / "cache.json"
    if cache_file.exists() and not run_info["resolved"]:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # Check if any entry has info.resolved == true
            for _, value in cache_data.items():
                if isinstance(value, dict) and value.get("info", {}).get("resolved", False):
                    run_info["resolved"] = True
                    run_info["resolved_in_intentions"] = True
                    results["resolved_in_intentions"] += 1
                    break
        except Exception as e:
            print(f"Error reading cache.json for {run_name}: {e}")

    if not run_info["resolved"]:
        results["not_resolved"] += 1

    results["runs_details"].append(run_info)

# --- Sample 50 combined: resolved by intentions + unresolved (seed=42) ---
intent_solved_runs = [
    rd["run"] for rd in results["runs_details"]
    if rd.get("resolved_in_intentions", False)
]
unresolved_runs = [
    rd["run"] for rd in results["runs_details"]
    if not rd.get("resolved", False)
]

combined_runs = intent_solved_runs + unresolved_runs
rng = random.Random(42)
k = min(50, len(combined_runs))
sampled = rng.sample(combined_runs, k) if k > 0 else []

out_sample_file = base_dir.parent / "sample_50.json"
with open(out_sample_file, "w", encoding="utf-8") as f:
    json.dump(sampled, f, indent=2, ensure_ascii=False)

print(f"\nSampled {k} / {len(combined_runs)} runs (intentions resolved: {len(intent_solved_runs)}, unresolved: {len(unresolved_runs)}, seed=42).")
print(f"List saved to: {out_sample_file}")

# Print summary
print("=" * 80)
print("RUN ANALYSIS - bash_only_dev")
print("=" * 80)
print(f"\nTotal runs:                {results['total_runs']}")
print(f"\nResolved in bootstrap:     {results['resolved_in_bootstrap']}")
print(f"Resolved by intentions:    {results['resolved_in_intentions']}")
print(f"Not resolved:              {results['not_resolved']}")
print(f"\nTOTAL RESOLVED:            {results['resolved_in_bootstrap'] + results['resolved_in_intentions']}")
if results["total_runs"] > 0:
    print(f"Resolved percentage:       {(results['resolved_in_bootstrap'] + results['resolved_in_intentions']) / results['total_runs'] * 100:.2f}%")

# Save detailed results
output_file = base_dir.parent / "analysis_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nDetailed results saved to: {output_file}")
