from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import litellm
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_path
from minisweagent.models import get_model
from minisweagent.run.extra.swebench import get_sb_environment


def run_mini_on_instance(
    *,
    instance: Dict[str, Any],
    intention: Optional[str],
    model_name: str,
    config_path: Path,
    environment_class: Optional[str] = None,
    exit_immediately: bool = True,
    capture_trace: bool = False,
    max_steps: Optional[int] = None,
) -> str | tuple[str, str]:
    """Run mini-swe-agent on one SWE-bench instance.
    
    If intention is provided, it's injected as HIGH-PRIORITY FIX INTENTION.
    If intention is None, agent runs directly on problem statement.
    
    If capture_trace=True, returns (patch, trace) tuple.
    
    If max_steps is provided, sets step_limit in agent config to prevent infinite loops.
    """
    config_path = get_config_path(config_path)
    config = yaml.safe_load(config_path.read_text())

    if environment_class is not None:
        # mini-swe-agent uses "singularity" for Singularity/Apptainer; accept "apptainer" as alias
        ec = "singularity" if environment_class.strip().lower() == "apptainer" else environment_class
        config.setdefault("environment", {})["environment_class"] = ec
    if max_steps is not None and max_steps > 0:
        config.setdefault("agent", {})["step_limit"] = max_steps

    # Note: max_tokens is calculated internally by mini-swe-agent based on context_length
    # We don't set it here as LitellmModelConfig doesn't accept it as a parameter
    # The max_tokens=-100 error should be handled by mini-swe-agent's internal logic

    env = get_sb_environment(config, instance)
    # Use DefaultAgent instead of InteractiveAgent to avoid interactive prompts
    # when step limit is exceeded
    # Note: DefaultAgent doesn't support 'confirm_exit' - it's InteractiveAgent-specific
    agent_config = config.get("agent", {}).copy()
    agent_config.pop("confirm_exit", None)  # Remove if present (InteractiveAgent-only)
    
    agent = DefaultAgent(
        get_model(model_name, config.get("model", {})),
        env,
        **agent_config,
    )

    # Truncate context to prevent max_tokens errors
    # vLLM has max_model_len=32000, we need to leave space for:
    # - System prompt: ~2000 tokens
    # - Agent messages/history: ~5000-10000 tokens (grows with steps)
    # - Response: ~2000 tokens
    # - Safety margin: ~2000 tokens
    # Total available for initial task: ~15000-20000 tokens
    # Using ~3 chars per token: 15000 * 3 = 45000 chars max for task
    # However, we need to be more conservative to account for message history growth
    # Reduce to ~10000 tokens = 30000 chars to leave room for growing history
    MAX_TASK_CHARS = 30000
    
    if intention and intention.strip():
        intention_text = intention.strip()
        problem_text = instance['problem_statement']
        
        # Calculate available space (leave some for headers)
        header_len = len("### HIGH-PRIORITY FIX INTENTION (must follow)\n\n### ORIGINAL SWE-BENCH PROBLEM STATEMENT\n")
        available_chars = MAX_TASK_CHARS - header_len
        
        # Prioritize intention (keep at least 20% of space for it, but cap at 5000 chars)
        # Intention should be concise, problem statement can be longer
        intention_max = min(max(int(available_chars * 0.2), 500), 5000)  # 500-5000 chars for intention
        problem_max = available_chars - intention_max
        
        if len(intention_text) > intention_max:
            print(f"[WARNING] Truncating intention from {len(intention_text)} to {intention_max} chars")
            intention_text = intention_text[:intention_max] + "\n\n[... intention truncated ...]"
        
        if len(problem_text) > problem_max:
            print(f"[WARNING] Truncating problem_statement from {len(problem_text)} to {problem_max} chars")
            problem_text = problem_text[:problem_max] + "\n\n[... problem statement truncated ...]"
        
        task = (
            "### HIGH-PRIORITY FIX INTENTION (must follow)\n\n"
            "RULES (must follow):\n"
            "- DO NOT create new files (no test projects, no debug scripts, no reproduction scripts).\n"
            "- DO NOT create new modules or directories.\n"
            "- ONLY edit existing files in the repository.\n"
            "- If you think a new file is needed, STOP and instead modify the existing implementation.\n"
            "- Provide a MINIMAL patch — change only what is necessary to fix the issue.\n"
            "- Do NOT add print statements, logging, or debug code.\n"
            "- Do NOT create Django projects, test apps, or standalone scripts.\n\n"
            f"Intention: {intention_text}\n\n"
            "### ORIGINAL SWE-BENCH PROBLEM STATEMENT\n"
            f"{problem_text}"
        )
    else:
        # No intention - run directly on problem statement
        problem_text = instance['problem_statement']
        # Add instruction to edit directly
        direct_edit_instruction = (
            "RULES (must follow):\n"
            "- DO NOT create new files (no test projects, no debug scripts, no reproduction scripts).\n"
            "- DO NOT create new modules or directories.\n"
            "- ONLY edit existing files in the repository.\n"
            "- If you think a new file is needed, STOP and instead modify the existing implementation.\n"
            "- Provide a MINIMAL patch — change only what is necessary to fix the issue.\n"
            "- Do NOT add print statements, logging, or debug code.\n"
            "- Do NOT create Django projects, test apps, or standalone scripts.\n\n"
        )
        available = MAX_TASK_CHARS - len(direct_edit_instruction)
        if len(problem_text) > available:
            print(f"[WARNING] Truncating problem_statement from {len(problem_text)} to {available} chars")
            task = direct_edit_instruction + problem_text[:available] + "\n\n[... problem statement truncated ...]"
        else:
            task = direct_edit_instruction + problem_text

    try:
        exit_status, patch = agent.run(task)
    except litellm.exceptions.ContextWindowExceededError as e:
        # Context window exceeded - return empty patch gracefully
        print(f"[ERROR] Context window exceeded: {e}")
        print(f"[ERROR] This candidate's context grew too large. Returning empty patch.")
        exit_status, patch = "ContextWindowExceeded", ""

    # Check if step/cost limit was exceeded (run() returns status, doesn't raise)
    # LimitsExceeded is caught internally by run() and returned as ("LimitsExceeded", "")
    if exit_status == "LimitsExceeded" and not patch.strip():
        print(f"[WARNING] Step/cost limit exceeded. Attempting to extract partial patch from git diff...")
        try:
            # Try to get partial patch by executing git diff in the environment
            # Environment should still be running at this point
            action = {"action": "git add -A && git diff --cached"}
            result = agent.execute_action(action)
            if result and isinstance(result, dict):
                output = result.get("output", "")
                if output and output.strip():
                    patch = output.strip()
                    print(f"[INFO] Extracted partial patch ({len(patch)} chars) from git diff")
                else:
                    print(f"[INFO] No staged changes found in git diff --cached")
        except Exception as e:
            print(f"[WARNING] Could not extract partial patch: {e}")
            patch = ""

    if hasattr(env, "stop"):
        env.stop()

    patch = (patch or "").strip()
    
    # Debug: Check patch format and content
    if patch:
        is_unified_diff = (
            patch.startswith("diff --git") or
            (patch.startswith("---") and "+++" in patch[:200]) or
            ("@@ -" in patch[:200] and "@@ +" in patch[:200])
        )
        if not is_unified_diff:
            print(f"[WARNING] Patch doesn't look like unified diff format. First 500 chars:")
            print(patch[:500])
        else:
            # Count files changed
            num_files = patch.count("diff --git")
            # Check for potential issues
            has_test_files = "test/" in patch.lower() or "/test" in patch.lower()
            print(f"[INFO] Patch appears to be unified diff format ({len(patch)} chars, {num_files} files changed)")
            if has_test_files:
                print(f"[WARNING] Patch contains test files - this may cause issues")
            if len(patch) > 50000:
                print(f"[WARNING] Patch is very large ({len(patch)} chars) - may contain unnecessary changes")
    
    if capture_trace:
        # Capture agent's conversation as trace
        trace_parts = []
        for msg in agent.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                trace_parts.append(f"[{role}]\n{content[:2000]}")
        trace = "\n\n---\n\n".join(trace_parts[-10:])  # Last 10 messages
        return patch, trace
    
    return patch
