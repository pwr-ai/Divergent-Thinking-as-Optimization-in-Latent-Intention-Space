from __future__ import annotations

from typing import Any, Dict

from datasets import load_dataset
from minisweagent.run.extra.swebench import DATASET_MAPPING

# SWE-bench Bash Only uses the Verified dataset (mini-SWE-agent, minimal bash env).
# https://www.swebench.com/bash-only.html
BASH_ONLY_DATASET = "princeton-nlp/SWE-Bench_Verified"


def get_dataset_path(subset: str) -> str:
    """Resolve subset to HuggingFace dataset path. Supports 'bash_only' -> Verified."""
    if subset == "bash_only":
        return BASH_ONLY_DATASET
    return DATASET_MAPPING.get(subset, subset)


def get_harness_dataset_name(subset: str) -> str:
    """Dataset name for SWE-bench harness (e.g. for run_evaluation). Matches get_dataset_path."""
    return get_dataset_path(subset)


def get_effective_split(dataset_path: str, split: str) -> str:
    """Return split to use for load_dataset. Verified has only 'test', so dev -> test."""
    if dataset_path == BASH_ONLY_DATASET or "SWE-Bench_Verified" in dataset_path:
        return "test" if split == "dev" else split
    return split


def load_swebench_instance(subset: str, split: str, instance_id: str) -> Dict[str, Any]:
    """Load a SWE-bench instance by instance_id, using mini-swe-agent's DATASET_MAPPING."""
    dataset_path = get_dataset_path(subset)
    effective_split = get_effective_split(dataset_path, split)
    ds = load_dataset(dataset_path, split=effective_split)
    for inst in ds:
        if inst["instance_id"] == instance_id:
            return dict(inst)
    hint = ""
    if "SWE-Bench_Verified" in dataset_path or subset in ("bash_only", "verified"):
        hint = (
            " For bash_only/Verified use an instance from the Verified set (500 instances, NO sqlfluff). "
            "Example Verified instances: django__django-11265, matplotlib__matplotlib-24177. "
            "Or use --subset lite for Lite instances (e.g. sqlfluff__sqlfluff-1625)."
        )
    raise KeyError(f"instance_id not found: {instance_id}.{hint}")
