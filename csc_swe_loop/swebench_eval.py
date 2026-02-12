from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from swebench.harness.run_evaluation import (
    run_instances,
    get_dataset_from_preds,
    make_run_report,
    LOG_REPORT,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.modal_eval import run_instances_modal, validate_modal_credentials
from swebench.harness.utils import load_swebench_dataset

# In-memory dataset cache: avoids calling load_dataset() on every evaluate_patch()
# (each call does ~50 stat/open ops on HF cache; with 80+ evals per instance this adds up)
_dataset_cache: Dict[str, list] = {}


def _get_cached_dataset(dataset_name: str, split: str) -> list:
    """Load dataset once, reuse from memory on subsequent calls."""
    key = f"{dataset_name}::{split}"
    if key not in _dataset_cache:
        _dataset_cache[key] = load_swebench_dataset(dataset_name, split)
    return _dataset_cache[key]


def _filter_new_files_from_patch(patch: str) -> str:
    """Remove new file diffs from patch, keeping only modifications to existing files.
    
    New files are often:
    - Test scripts (test_*.py, reproduce_*.py)
    - Django projects (manage.py, settings.py, models.py in new dirs)
    - Backup files (*.bak, *.orig)
    - Binary files (*.sqlite3)
    
    These are usually agent mistakes and often truncated, causing 'malformed patch' errors.
    """
    if not patch or not patch.strip():
        return patch
    
    lines = patch.split("\n")
    result_lines = []
    skip_until_next_diff = False
    current_file = None
    removed_files = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Start of a new diff section
        if line.startswith("diff --git "):
            skip_until_next_diff = False
            current_file = line
            
            # Look ahead for "new file mode" within next few lines
            is_new_file = False
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].startswith("diff --git "):
                    break
                if lines[j].startswith("new file mode"):
                    is_new_file = True
                    break
            
            if is_new_file:
                # Extract filename for logging
                parts = line.split()
                if len(parts) >= 4:
                    filename = parts[3]  # b/path/to/file
                    if filename.startswith("b/"):
                        filename = filename[2:]
                    removed_files.append(filename)
                skip_until_next_diff = True
                i += 1
                continue
        
        if not skip_until_next_diff:
            result_lines.append(line)
        
        i += 1
    
    if removed_files:
        print(f"[HARNESS] Filtered {len(removed_files)} new file(s) from patch: {removed_files}")
    
    return "\n".join(result_lines)


def _sanitize_patch_for_apply(patch: str) -> str:
    """Ensure patch can be applied by git apply (no 'ends in middle of line').
    If patch is truncated (no trailing newline), truncate to last complete line.
    """
    if not patch or not patch.strip():
        return patch
    patch = patch.rstrip()
    if not patch:
        return patch
    # git apply requires each line to end with newline; truncated output often lacks it
    if not patch.endswith("\n"):
        last_nl = patch.rfind("\n")
        if last_nl >= 0:
            patch = patch[: last_nl + 1]
            print(f"[HARNESS] Patch looked truncated (no trailing newline); truncated to last complete line ({last_nl + 1} chars)")
        else:
            patch = patch + "\n"
    else:
        patch = patch.rstrip("\n") + "\n"
    return patch


def _prepare_patch_and_predictions(
    *,
    instance_id: str,
    patch: str,
    dataset_name: str,
    out_dir: Path,
    run_id: str,
    split: str,
) -> tuple[Dict[str, Any], list] | Dict[str, Any]:
    """Common preparation: filter/sanitize patch, build predictions, get dataset instances.
    
    Returns (predictions, instances) on success, or error dict on failure.
    """
    patch = _filter_new_files_from_patch(patch or "")
    patch = _sanitize_patch_for_apply(patch or "")
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"preds_{run_id}.jsonl"
    preds_path.write_text(
        json.dumps({
            "instance_id": instance_id,
            "model_name_or_path": "csc+mini",
            "model_patch": patch,
        }) + "\n",
        encoding="utf-8",
    )

    predictions = {
        instance_id: {
            "instance_id": instance_id,
            "model_name_or_path": "csc+mini",
            "model_patch": patch,
        }
    }

    # Use in-memory cached dataset instead of get_dataset_from_preds()
    # (avoids load_dataset() on every call — saves ~50 file ops × 80+ evals per instance)
    print(f"[HARNESS] Getting dataset instance for {instance_id} (split={split}, dataset={dataset_name})")
    full_dataset = _get_cached_dataset(dataset_name, split)
    instances = [i for i in full_dataset if i["instance_id"] == instance_id and instance_id in predictions]

    if not instances:
        print(f"[HARNESS] Instance {instance_id} not found in dataset")
        return {"error": "No instances to evaluate", "instance_id": instance_id}

    print(f"[HARNESS] Found {len(instances)} instance(s) to evaluate")
    return predictions, instances


def _read_report(run_id: str, instance_id: str) -> Dict[str, Any]:
    """Read evaluation report from disk. Tries multiple paths for swebench compat."""
    model_name = "csc+mini"
    possible_paths = [
        RUN_EVALUATION_LOG_DIR / run_id / LOG_REPORT,
        RUN_EVALUATION_LOG_DIR / run_id / model_name / instance_id / LOG_REPORT,
    ]

    for path in possible_paths:
        if path.exists():
            report = json.loads(path.read_text())
            print(f"[HARNESS] Report found at {path}, keys: {list(report.keys())}")
            if "resolved" in report:
                print(f"[HARNESS] resolved: {report.get('resolved', [])}")
            if "applied" in report:
                print(f"[HARNESS] applied: {report.get('applied', [])}")
            if instance_id in report and isinstance(report[instance_id], dict):
                print(f"[HARNESS] Found instance data at report['{instance_id}']")
            return report

    print(f"[HARNESS] Report not found. Tried paths:")
    for path in possible_paths:
        print(f"[HARNESS]   - {path}")
    return {"error": "Report not found", "instance_id": instance_id}


# ---------------------------------------------------------------------------
# Docker evaluation (original path)
# ---------------------------------------------------------------------------
def _evaluate_patch_docker(
    predictions: Dict[str, Any],
    instances: list,
    run_id: str,
    instance_id: str,
) -> Dict[str, Any]:
    """Evaluate patch using local Docker/Podman containers."""
    import docker as docker_lib

    docker_host = os.environ.get("DOCKER_HOST")
    print(f"[HARNESS] DOCKER_HOST env: {docker_host}")
    if not docker_host:
        import subprocess
        podman_sock = None
        try:
            result = subprocess.run(
                ["podman", "info", "--format", "{{.Host.RemoteSocket.Path}}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                podman_sock = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        if not podman_sock or not os.path.exists(podman_sock):
            podman_sockets = [
                f"/run/user/{os.getuid()}/podman/podman.sock",
                "/run/podman/podman.sock",
            ]
            for sock_path in podman_sockets:
                if os.path.exists(sock_path):
                    podman_sock = sock_path
                    break

        if podman_sock and not os.path.exists(podman_sock):
            try:
                subprocess.Popen(
                    ["podman", "system", "service", "--time=0", f"unix://{podman_sock}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                import time
                for _ in range(5):
                    if os.path.exists(podman_sock):
                        break
                    time.sleep(0.5)
            except (FileNotFoundError, Exception):
                pass

        if podman_sock and os.path.exists(podman_sock):
            os.environ["DOCKER_HOST"] = f"unix://{podman_sock}"
            print(f"[HARNESS] Using podman socket: {podman_sock}")

    try:
        client = docker_lib.from_env()
        client.ping()
        print(f"[HARNESS] Docker/Podman connection verified")
    except Exception as e:
        return {
            "error": "Docker not available",
            "instance_id": instance_id,
            "error_type": type(e).__name__,
            "error_msg": str(e),
        }

    try:
        run_instances(
            predictions=predictions,
            instances=instances,
            cache_level="instance",
            clean=False,
            force_rebuild=False,
            max_workers=1,
            run_id=run_id,
            timeout=1800,
        )
    except Exception as e:
        error_msg = str(e)
        if any(kw in error_msg for kw in ("docker", "Connection", "No such file", "API version")):
            return {
                "error": "Docker connection failed",
                "instance_id": instance_id,
                "error_type": type(e).__name__,
                "error_msg": error_msg,
            }
        raise

    return _read_report(run_id, instance_id)


# ---------------------------------------------------------------------------
# Modal evaluation (cloud, no local Docker needed)
# ---------------------------------------------------------------------------
def _evaluate_patch_modal(
    predictions: Dict[str, Any],
    instances: list,
    full_dataset: list,
    run_id: str,
    instance_id: str,
) -> Dict[str, Any]:
    """Evaluate patch using Modal cloud sandboxes (no local Docker required)."""
    validate_modal_credentials()
    print(f"[HARNESS-MODAL] Running evaluation on Modal for {instance_id}")
    try:
        run_instances_modal(
            predictions=predictions,
            instances=instances,
            full_dataset=full_dataset,
            run_id=run_id,
            timeout=1800,
        )
    except Exception as e:
        error_msg = str(e)
        print(f"[HARNESS-MODAL] Exception during run_instances_modal: {type(e).__name__}: {error_msg}")
        return {
            "error": "Modal evaluation failed",
            "instance_id": instance_id,
            "error_type": type(e).__name__,
            "error_msg": error_msg,
        }

    return _read_report(run_id, instance_id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def evaluate_patch(
    *,
    instance_id: str,
    patch: str,
    dataset_name: str,
    out_dir: Path,
    run_id: str,
    split: str = "test",
    eval_backend: str = "docker",
) -> Dict[str, Any]:
    """Run SWE-bench harness for a single patch.
    
    eval_backend: "docker" (local Docker/Podman) or "modal" (Modal cloud).
    """
    result = _prepare_patch_and_predictions(
        instance_id=instance_id,
        patch=patch,
        dataset_name=dataset_name,
        out_dir=out_dir,
        run_id=run_id,
        split=split,
    )
    if isinstance(result, dict):
        # Error during preparation
        return result
    predictions, instances = result

    if eval_backend == "modal":
        full_dataset = _get_cached_dataset(dataset_name, split)
        return _evaluate_patch_modal(predictions, instances, full_dataset, run_id, instance_id)
    else:
        return _evaluate_patch_docker(predictions, instances, run_id, instance_id)


def extract_instance_result(results: Dict[str, Any], instance_id: str) -> Optional[Dict[str, Any]]:
    """Try common keys for per-instance results across swebench versions."""
    # Harness returned an error (e.g. Docker connection failed)
    if results.get("error") and "error_msg" in results:
        print(f"[EXTRACT] Harness error for {instance_id}: {results.get('error_type', 'Unknown')}: {results.get('error_msg', '')}")
        return None
    # Check if instance is in resolved list
    resolved = instance_id in results.get("resolved", [])
    
    # swebench v3 format: instance_id is a top-level key directly in the report
    if instance_id in results and isinstance(results[instance_id], dict):
        result = results[instance_id]
        print(f"[EXTRACT] Found result at top-level key '{instance_id}': {list(result.keys())}")
        # Parse tests_status to compute f2p/p2p
        tests_status = result.get("tests_status", {})
        f2p_data = tests_status.get("FAIL_TO_PASS", {})
        p2p_data = tests_status.get("PASS_TO_PASS", {})
        
        f2p_success = len(f2p_data.get("success", []))
        f2p_failure = len(f2p_data.get("failure", []))
        f2p_total = f2p_success + f2p_failure
        
        p2p_success = len(p2p_data.get("success", []))
        p2p_failure = len(p2p_data.get("failure", []))
        p2p_total = p2p_success + p2p_failure
        
        # Compute ratios
        f2p_ratio = f2p_success / f2p_total if f2p_total > 0 else (1.0 if result.get("resolved") else 0.0)
        p2p_ratio = p2p_success / p2p_total if p2p_total > 0 else 1.0
        
        print(f"[EXTRACT] f2p={f2p_success}/{f2p_total}={f2p_ratio:.3f}, p2p={p2p_success}/{p2p_total}={p2p_ratio:.3f}")
        
        return {
            "resolved": result.get("resolved", False),
            "patch_applied": result.get("patch_successfully_applied", False),
            "f2p": f2p_ratio,
            "p2p": p2p_ratio,
            "fail_to_pass_passed": f2p_success,
            "fail_to_pass_total": f2p_total,
            "pass_to_pass_passed": p2p_success,
            "pass_to_pass_total": p2p_total,
        }
    
    # Try common keys for per-instance results (older swebench versions)
    for key in ("instance_results", "results", "instances"):
        if key in results and isinstance(results[key], dict) and instance_id in results[key]:
            result = results[key][instance_id]
            print(f"[EXTRACT] Found result in '{key}': {list(result.keys()) if isinstance(result, dict) else type(result)}")
            return result
    
    # Build a minimal result from the report structure
    if "resolved" in results or "applied" in results:
        print(f"[EXTRACT] Building minimal result: resolved={resolved}, applied={instance_id in results.get('applied', [])}")
        return {
            "resolved": resolved,
            "applied": instance_id in results.get("applied", []),
            "f2p": 1.0 if resolved else 0.0,
            "p2p": 1.0 if resolved else 0.0,
        }
    
    print(f"[EXTRACT] No result found for {instance_id}. Results keys: {list(results.keys())}")
    return None
