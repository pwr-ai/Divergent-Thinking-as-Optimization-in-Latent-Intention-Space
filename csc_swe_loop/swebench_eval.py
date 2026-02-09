from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from swebench.harness.run_evaluation import (
    run_instances,
    get_dataset_from_preds,
    make_run_report,
    LOG_REPORT,
    RUN_EVALUATION_LOG_DIR,
)


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


def evaluate_patch(
    *,
    instance_id: str,
    patch: str,
    dataset_name: str,
    out_dir: Path,
    run_id: str,
    split: str = "test",
) -> Dict[str, Any]:
    """Run SWE-bench harness for a single patch by writing a JSONL predictions file."""
    # First filter out new files (test scripts, Django projects, etc) - these are usually agent mistakes
    patch = _filter_new_files_from_patch(patch or "")
    # Then sanitize for git apply (handle truncation)
    patch = _sanitize_patch_for_apply(patch or "")
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"preds_{run_id}.jsonl"
    preds_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "model_name_or_path": "csc+mini",
                "model_patch": patch,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Build predictions dict
    predictions = {
        instance_id: {
            "instance_id": instance_id,
            "model_name_or_path": "csc+mini",
            "model_patch": patch,
        }
    }

    # Get dataset instances
    # Note: rewrite_reports=True only returns instances with existing test_output.txt
    # We use rewrite_reports=False to evaluate fresh instances
    print(f"[HARNESS] Getting dataset instances for {instance_id} (split={split}, dataset={dataset_name})")
    instances = get_dataset_from_preds(
        dataset_name=dataset_name,
        split=split,
        instance_ids=[instance_id],
        predictions=predictions,
        run_id=run_id,
        rewrite_reports=False,
        exclude_completed=False,
    )

    if not instances:
        print(f"[HARNESS] No instances returned from get_dataset_from_preds for {instance_id}")
        print(f"[HARNESS] Predictions keys: {list(predictions.keys())}")
        print(f"[HARNESS] Prediction structure: {list(predictions[instance_id].keys()) if instance_id in predictions else 'missing'}")
        return {"error": "No instances to evaluate", "instance_id": instance_id}
    
    print(f"[HARNESS] Found {len(instances)} instance(s) to evaluate")

    # Run evaluation
    # Note: run_instances() creates docker client internally with docker.from_env()
    # We verify Docker/Podman is available first to provide better error messages
    import docker
    import os
    
    # Try to find podman socket if docker socket is not available
    docker_host = os.environ.get("DOCKER_HOST")
    print(f"[HARNESS] DOCKER_HOST env: {docker_host}")
    if not docker_host:
        # Try to get podman socket location from podman info
        import subprocess
        podman_sock = None
        try:
            result = subprocess.run(
                ["podman", "info", "--format", "{{.Host.RemoteSocket.Path}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                podman_sock = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        # If socket path from podman info doesn't exist, try common locations
        if not podman_sock or not os.path.exists(podman_sock):
            podman_sockets = [
                f"/run/user/{os.getuid()}/podman/podman.sock",
                "/run/podman/podman.sock",
            ]
            for sock_path in podman_sockets:
                if os.path.exists(sock_path):
                    podman_sock = sock_path
                    break
        
        # If we found a socket path but it doesn't exist, try to start podman service
        if podman_sock and not os.path.exists(podman_sock):
            try:
                # Start podman system service in background (time=0 means run until stopped)
                # Note: This creates the socket if podman is available
                subprocess.Popen(
                    ["podman", "system", "service", "--time=0", f"unix://{podman_sock}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                # Wait a bit for socket to be created
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
        client = docker.from_env()
        client.ping()
        print(f"[HARNESS] Docker/Podman connection verified")
    except Exception as docker_check_error:
        error_type = type(docker_check_error).__name__
        error_msg = str(docker_check_error)
        print(f"[HARNESS] Docker/Podman check failed: {error_type}: {error_msg}")
        print(f"[HARNESS] Docker/Podman daemon may not be running or socket not accessible")
        print(f"[HARNESS] Try: sudo systemctl start docker (Linux) or check Docker Desktop (macOS/Windows)")
        print(f"[HARNESS] Or: sudo usermod -aG docker $USER (add user to docker group)")
        print(f"[HARNESS] Or set DOCKER_HOST environment variable to podman socket path")
        return {"error": "Docker not available", "instance_id": instance_id, "error_type": error_type, "error_msg": error_msg}
    
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
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"[HARNESS] Exception during run_instances: {error_type}: {error_msg}")
        # Check if it's a Docker connection error
        if "docker" in error_msg.lower() or "Connection" in error_msg or "No such file" in error_msg or "API version" in error_msg:
            print(f"[HARNESS] Docker connection error - Docker daemon may not be running or socket not accessible")
            print(f"[HARNESS] Try: sudo systemctl start docker (Linux) or check Docker Desktop (macOS/Windows)")
            return {"error": "Docker connection failed", "instance_id": instance_id, "error_type": error_type, "error_msg": error_msg}
        # Re-raise other exceptions
        raise

    # Read report - try multiple paths as swebench v3 changed the structure
    # Old path: logs/run_evaluation/{run_id}/report.json
    # New path: logs/run_evaluation/{run_id}/{model_name}/{instance_id}/report.json
    model_name = "csc+mini"
    possible_paths = [
        RUN_EVALUATION_LOG_DIR / run_id / LOG_REPORT,  # Old swebench path
        RUN_EVALUATION_LOG_DIR / run_id / model_name / instance_id / LOG_REPORT,  # New swebench v3 path
    ]
    
    report_path = None
    for path in possible_paths:
        if path.exists():
            report_path = path
            break
    
    if report_path:
        report = json.loads(report_path.read_text())
        print(f"[HARNESS] Report found at {report_path}, keys: {list(report.keys())}")
        # Log key fields for debugging
        if "resolved" in report:
            print(f"[HARNESS] resolved: {report.get('resolved', [])}")
        if "applied" in report:
            print(f"[HARNESS] applied: {report.get('applied', [])}")
        if "instance_results" in report:
            print(f"[HARNESS] instance_results keys: {list(report['instance_results'].keys()) if isinstance(report.get('instance_results'), dict) else 'not a dict'}")
        # swebench v3 uses instance_id as top-level key directly
        if instance_id in report and isinstance(report[instance_id], dict):
            print(f"[HARNESS] Found instance data at report['{instance_id}']")
        return report

    print(f"[HARNESS] Report not found. Tried paths:")
    for path in possible_paths:
        print(f"[HARNESS]   - {path}")
    print(f"[HARNESS] RUN_EVALUATION_LOG_DIR: {RUN_EVALUATION_LOG_DIR}")
    print(f"[HARNESS] run_id: {run_id}")
    return {"error": "Report not found", "instance_id": instance_id}


def extract_instance_result(results: Dict[str, Any], instance_id: str) -> Optional[Dict[str, Any]]:
    """Try common keys for per-instance results across swebench versions."""
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
