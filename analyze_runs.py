#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import defaultdict

base_dir = Path("/mnt/bystry/research/swe/dataset_runs/bash_only_dev_tabu/bash_only_dev")

results = {
    "total_runs": 0,
    "resolved_in_bootstrap": 0,
    "resolved_in_intentions": 0,
    "not_resolved": 0,
    "runs_details": []
}

# Find all run directories
run_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("django__")])
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
            with open(bootstrap_cache, 'r') as f:
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
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                # Check if any entry has resolved: true
                for key, value in cache_data.items():
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

# Print summary
print("=" * 80)
print("ANALIZA RUNÓW - bash_only_dev_tabu")
print("=" * 80)
print(f"\nCałkowita liczba runów: {results['total_runs']}")
print(f"\nRozwiązane w bootstrap: {results['resolved_in_bootstrap']}")
print(f"Rozwiązane przez intencje: {results['resolved_in_intentions']}")
print(f"Nierozwiązane: {results['not_resolved']}")
print(f"\nRAZEM ROZWIĄZANE: {results['resolved_in_bootstrap'] + results['resolved_in_intentions']}")
print(f"Procent rozwiązanych: {(results['resolved_in_bootstrap'] + results['resolved_in_intentions']) / results['total_runs'] * 100:.2f}%")

print("\n" + "=" * 80)
print("SZCZEGÓŁY RUNÓW:")
print("=" * 80)
for run_detail in results["runs_details"]:
    status = []
    if run_detail["resolved_in_bootstrap"]:
        status.append("BOOTSTRAP")
    if run_detail["resolved_in_intentions"]:
        status.append("INTENCJE")
    if not run_detail["resolved"]:
        status.append("NIE ROZWIĄZANE")
    
    print(f"{run_detail['run']}: {', '.join(status)}")

# Save detailed results
output_file = base_dir.parent / "analysis_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSzczegółowe wyniki zapisane do: {output_file}")
