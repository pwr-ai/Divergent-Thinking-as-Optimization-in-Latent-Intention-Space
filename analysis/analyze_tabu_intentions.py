#!/usr/bin/env python3
"""
Analyze all tabu instances to determine how many resolved patches are consistent with their intentions.
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

def sha1(text: str) -> str:
    """Compute SHA1 hash of text."""
    return hashlib.sha1(text.encode('utf-8')).hexdigest()

def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file."""
    if not file_path.exists():
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def load_json(file_path: Path) -> Dict:
    """Load JSON file."""
    if not file_path.exists():
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_patches_by_round(instance_dir: Path) -> Dict[Tuple[int, int], str]:
    """Find patches organized by (round, idx)."""
    eval_dir = instance_dir / "eval"
    patches = {}
    
    if not eval_dir.exists():
        return patches
    
    # Check all preds files - they are named like preds_r{round}_i{idx}_{timestamp}.jsonl
    for pred_file in sorted(eval_dir.glob("preds_*.jsonl")):
        preds = load_jsonl(pred_file)
        for pred in preds:
            if "model_patch" in pred and pred["model_patch"]:
                # Try to extract round/idx from filename
                match = re.search(r'preds_r(\d+)_i(\d+)_', pred_file.name)
                if match:
                    round_num = int(match.group(1))
                    idx_num = int(match.group(2))
                    patches[(round_num, idx_num)] = pred["model_patch"]
                else:
                    # Check for bootstrap attempts
                    match = re.search(r'preds_boot_attempt(\d+)_', pred_file.name)
                    if match:
                        # Bootstrap attempts are before round 0
                        attempt = int(match.group(1))
                        patches[(-1, attempt)] = pred["model_patch"]
    
    return patches

def extract_key_from_patch(patch: str) -> Optional[str]:
    """Extract a hash/key from patch content (if possible)."""
    # This is a placeholder - we'll need to match patches differently
    return None

def analyze_intention_patch_alignment(intention: str, patch: str) -> Dict[str, any]:
    """
    Analyze if the patch aligns with the intention.
    Returns a dict with alignment score and details.
    """
    if not patch or not patch.strip():
        return {
            "aligned": False,
            "score": 0.0,
            "reason": "empty_patch",
            "details": {}
        }
    
    intention_lower = intention.lower()
    patch_lower = patch.lower()
    
    # Extract key concepts from intention
    intention_keywords = set()
    intention_files = set()
    intention_actions = set()
    
    # Common patterns in intentions - keywords (more comprehensive)
    if "validationerror" in intention_lower or "validation error" in intention_lower:
        intention_keywords.add("validationerror")
    if "validator" in intention_lower:
        intention_keywords.add("validator")
    if "value" in intention_lower and ("include" in intention_lower or "pass" in intention_lower or "provide" in intention_lower):
        intention_keywords.add("value_in_error")
    if "placeholder" in intention_lower or "%(value)s" in intention_lower:
        intention_keywords.add("placeholder")
    if "error" in intention_lower and "message" in intention_lower:
        intention_keywords.add("error_message")
    
    # Technical concepts
    if "__spec__" in intention_lower or "__spec" in intention_lower:
        intention_keywords.add("__spec__")
    if "__file__" in intention_lower or "__file" in intention_lower:
        intention_keywords.add("__file__")
    if "legacy" in intention_lower or "encoding" in intention_lower:
        intention_keywords.add("legacy_encoding")
    if "hashing" in intention_lower or "hash" in intention_lower or "algorithm" in intention_lower:
        intention_keywords.add("hashing")
    if "session" in intention_lower and ("store" in intention_lower or "encode" in intention_lower or "decode" in intention_lower):
        intention_keywords.add("session_encoding")
    if "duration" in intention_lower or "subtraction" in intention_lower or "distinct" in intention_lower:
        intention_keywords.add("duration_subtraction")
    if "query" in intention_lower and ("or" in intention_lower or "and" in intention_lower or "combined" in intention_lower):
        intention_keywords.add("query_logic")
    if "migration" in intention_lower or "__file__" in intention_lower or "__init__" in intention_lower:
        intention_keywords.add("migration_loader")
    if "middleware" in intention_lower or "asgi" in intention_lower or "handler" in intention_lower:
        intention_keywords.add("middleware_handler")
    if "sqlite" in intention_lower or "quote" in intention_lower or "keyword" in intention_lower:
        intention_keywords.add("sqlite_quotes")
    if "primary key" in intention_lower or "pk" in intention_lower or "auto" in intention_lower:
        intention_keywords.add("primary_key")
    
    # File/component mentions
    if "emailvalidator" in intention_lower or ("email" in intention_lower and "validator" in intention_lower):
        intention_files.add("email")
    if "session" in intention_lower:
        intention_files.add("session")
    if "django" in intention_lower and "contrib" in intention_lower:
        intention_files.add("django_contrib")
    if "autoreload" in intention_lower or "reload" in intention_lower:
        intention_files.add("autoreload")
    if "expressions" in intention_lower or "orm" in intention_lower:
        intention_files.add("expressions")
    if "query" in intention_lower:
        intention_files.add("query")
    if "migration" in intention_lower:
        intention_files.add("migration")
    if "sqlite" in intention_lower:
        intention_files.add("sqlite")
    if "handler" in intention_lower or "base" in intention_lower:
        intention_files.add("handler")
    if "options" in intention_lower or "model" in intention_lower:
        intention_files.add("options")
    
    # Actions mentioned
    if "include" in intention_lower or "add" in intention_lower:
        intention_actions.add("include")
    if "modify" in intention_lower or "change" in intention_lower or "update" in intention_lower:
        intention_actions.add("modify")
    if "implement" in intention_lower or "create" in intention_lower:
        intention_actions.add("implement")
    
    # Check if patch contains relevant code
    patch_keywords = set()
    patch_files = set()
    patch_actions = set()
    
    # Keywords in patch (more comprehensive)
    if "validationerror" in patch_lower:
        patch_keywords.add("validationerror")
    if "validator" in patch_lower:
        patch_keywords.add("validator")
    if "value" in patch_lower and ("error" in patch_lower or "message" in patch_lower or "validationerror" in patch_lower):
        patch_keywords.add("value_in_error")
    if "%(value)s" in patch_lower or "placeholder" in patch_lower or "%" in patch_lower:
        patch_keywords.add("placeholder")
    if "error" in patch_lower and "message" in patch_lower:
        patch_keywords.add("error_message")
    
    # Technical concepts in patch
    if "__spec__" in patch_lower or "__spec" in patch_lower or "spec.parent" in patch_lower:
        patch_keywords.add("__spec__")
    if "__file__" in patch_lower or "__file" in patch_lower:
        patch_keywords.add("__file__")
    if "legacy" in patch_lower or ("encoding" in patch_lower and ("session" in patch_lower or "encode" in patch_lower)):
        patch_keywords.add("legacy_encoding")
    if "hashing" in patch_lower or "hash" in patch_lower or "algorithm" in patch_lower or "DEFAULT_HASHING_ALGORITHM" in patch:
        patch_keywords.add("hashing")
    if "session" in patch_lower and ("encode" in patch_lower or "decode" in patch_lower or "serialize" in patch_lower):
        patch_keywords.add("session_encoding")
    if "duration" in patch_lower or "DurationField" in patch or "subtraction" in patch_lower or "subtract" in patch_lower or "SUB" in patch or "temporal" in patch_lower:
        patch_keywords.add("duration_subtraction")
    if ("query" in patch_lower and ("combined" in patch_lower or "union" in patch_lower or "intersection" in patch_lower)) or "combined_queries" in patch_lower:
        patch_keywords.add("query_logic")
    if "migration" in patch_lower or "__file__" in patch_lower or "__init__" in patch_lower:
        patch_keywords.add("migration_loader")
    if "middleware" in patch_lower or "asgi" in patch_lower or "handler" in patch_lower:
        patch_keywords.add("middleware_handler")
    if "sqlite" in patch_lower or "quote_name" in patch_lower or ("quote" in patch_lower and "table" in patch_lower) or "double-quote" in patch_lower or "PRAGMA" in patch:
        patch_keywords.add("sqlite_quotes")
    if ("primary" in patch_lower and "key" in patch_lower) or ("primary_key" in patch_lower) or ("pk" in patch_lower and "field" in patch_lower) or ("auto_created" in patch_lower and "primary" in patch_lower):
        patch_keywords.add("primary_key")
    
    # Files changed in patch
    if "email" in patch_lower or "emailvalidator" in patch_lower:
        patch_files.add("email")
    if "session" in patch_lower:
        patch_files.add("session")
    if "django/contrib" in patch_lower or "django\\contrib" in patch_lower:
        patch_files.add("django_contrib")
    if "autoreload" in patch_lower:
        patch_files.add("autoreload")
    if "expressions" in patch_lower or "models/expressions" in patch_lower:
        patch_files.add("expressions")
    if ("query" in patch_lower and ".py" in patch_lower) or "models/query" in patch_lower:
        patch_files.add("query")
    if "migration" in patch_lower and ".py" in patch_lower:
        patch_files.add("migration")
    if "sqlite" in patch_lower:
        patch_files.add("sqlite")
    if "handler" in patch_lower and "base" in patch_lower:
        patch_files.add("handler")
    if "options" in patch_lower:
        patch_files.add("options")
    
    # Actions in patch (code changes)
    if "+" in patch and ("value" in patch_lower or "error" in patch_lower):
        patch_actions.add("include")
    if "def " in patch_lower or "class " in patch_lower:
        patch_actions.add("implement")
    if "+" in patch or "-" in patch:
        patch_actions.add("modify")
    
    # Calculate alignment scores
    keyword_overlap = len(intention_keywords & patch_keywords)
    keyword_total = len(intention_keywords) if intention_keywords else 1
    keyword_score = keyword_overlap / keyword_total if keyword_total > 0 else 0.0
    
    file_overlap = len(intention_files & patch_files)
    file_total = len(intention_files) if intention_files else 1
    file_score = file_overlap / file_total if file_total > 0 else 0.0
    
    action_overlap = len(intention_actions & patch_actions)
    action_total = len(intention_actions) if intention_actions else 1
    action_score = action_overlap / action_total if action_total > 0 else 0.0
    
    # Weighted alignment score
    # Keywords are most important (50%), files (30%), actions (20%)
    # But if actions weren't mentioned in intention, don't penalize
    if action_total == 1 and len(intention_actions) == 0:
        # Actions weren't mentioned, so weight keywords and files more
        alignment_score = (keyword_score * 0.6 + file_score * 0.4)
    else:
        alignment_score = (keyword_score * 0.5 + file_score * 0.3 + action_score * 0.2)
    
    # If no keywords/files mentioned at all, check for general semantic similarity
    if keyword_total == 1 and len(intention_keywords) == 0 and file_total == 1 and len(intention_files) == 0:
        # Fallback: check if patch seems to address the general problem
        # This is a weaker signal
        if len(patch) > 50:  # Non-trivial patch
            alignment_score = 0.3  # Neutral score
    
    # Determine if aligned (threshold: 0.4 for more lenient matching)
    aligned = alignment_score >= 0.4
    
    return {
        "aligned": aligned,
        "score": alignment_score,
        "keyword_score": keyword_score,
        "file_score": file_score,
        "action_score": action_score,
        "intention_keywords": list(intention_keywords),
        "patch_keywords": list(patch_keywords),
        "intention_files": list(intention_files),
        "patch_files": list(patch_files),
        "intention_actions": list(intention_actions),
        "patch_actions": list(patch_actions),
        "keyword_overlap": keyword_overlap,
        "file_overlap": file_overlap,
        "action_overlap": action_overlap,
        "details": {
            "keyword_match": keyword_overlap > 0,
            "file_match": file_overlap > 0,
            "action_match": action_overlap > 0
        }
    }

def get_bootstrap_resolved_patch(instance_dir: Path) -> Optional[Tuple[str, str]]:
    """Get (intention, patch) from bootstrap if instance was solved in bootstrap. Patch from bootstrap_cache or eval."""
    bootstrap_file = instance_dir / "bootstrap_cache.json"
    if not bootstrap_file.exists():
        return None
    bootstrap = load_json(bootstrap_file)
    attempts = bootstrap.get("attempts") or []
    for att in attempts:
        if att.get("info", {}).get("resolved", False) and att.get("patch"):
            # Intention from anchors (first resolved anchor) or we'll get it from results
            intention = None
            for anc in bootstrap.get("anchors") or []:
                if anc.get("info", {}).get("resolved", False):
                    intention = anc.get("text", "")
                    break
            return (intention or "", att["patch"])
    # Fallback: patch from eval preds_boot_attempt*
    eval_dir = instance_dir / "eval"
    if eval_dir.exists():
        for pred_file in sorted(eval_dir.glob("preds_boot_attempt*.jsonl")):
            preds = load_jsonl(pred_file)
            for pred in preds:
                if pred.get("model_patch"):
                    return (None, pred["model_patch"])
    return None


def analyze_instance(instance_dir: Path, results_row: Optional[Dict] = None) -> Dict:
    """Analyze a single instance. If results_row is provided and solved, use it (bootstrap or loop best)."""
    instance_id = instance_dir.name
    cache_file = instance_dir / "cache.json"
    intentions_file = instance_dir / "intentions_log.jsonl"
    cache = load_json(cache_file)
    intentions = load_jsonl(intentions_file)
    intention_by_key = {intent["key"]: intent for intent in intentions}
    patches_by_round_idx = find_patches_by_round(instance_dir)

    resolved_patches = []  # list of dicts: intention, round, idx, key, patch (filled later)

    # 1) Resolved from results.jsonl (source of truth: bootstrap or loop best)
    if results_row and results_row.get("solved"):
        best = results_row.get("best") or {}
        intention = best.get("intention", "")
        if not intention and results_row.get("bootstrap", {}).get("initial_solution"):
            intention = results_row["bootstrap"]["initial_solution"].get("intention", "")
        if results_row.get("bootstrap", {}).get("solved_in_bootstrap"):
            boot = get_bootstrap_resolved_patch(instance_dir)
            if boot:
                boot_intention, boot_patch = boot
                if boot_intention is not None:
                    intention = boot_intention
                resolved_patches.append({
                    "key": "from_bootstrap",
                    "score": best.get("score", 1.0),
                    "info": best.get("info", {}),
                    "intention": intention,
                    "round": -1,
                    "idx": 0,
                    "patch_pre": boot_patch,
                    "source": "bootstrap",
                })
            else:
                resolved_patches.append({
                    "key": "from_results",
                    "score": best.get("score", 1.0),
                    "info": best.get("info", {}),
                    "intention": intention,
                    "round": -1,
                    "idx": 0,
                    "patch_pre": None,
                    "source": "bootstrap_results_only",
                })
        else:
            r, i = best.get("round", 0), best.get("idx", 0)
            patch = patches_by_round_idx.get((r, i))
            if not patch and r >= 0:
                for (rr, ii), p in patches_by_round_idx.items():
                    if rr == r:
                        patch = p
                        break
            if not patch and patches_by_round_idx:
                patch = list(patches_by_round_idx.values())[0]
            resolved_patches.append({
                "key": "from_loop",
                "score": best.get("score", 1.0),
                "info": best.get("info", {}),
                "intention": intention,
                "round": r,
                "idx": i,
                "patch_pre": patch,
                "source": "loop",
            })

    # 2) Resolved from cache.json (CMA loop) – add only if not already added from results
    for key, data in cache.items():
        if not isinstance(data, dict) or not data.get("info", {}).get("resolved", False):
            continue
        intention = intention_by_key.get(key, {})
        resolved_patches.append({
            "key": key,
            "score": data.get("score", 0.0),
            "info": data.get("info", {}),
            "intention": intention.get("intention", ""),
            "round": intention.get("round", -1),
            "idx": intention.get("idx", -1),
            "patch_pre": None,
            "source": "cache",
        })

    # Deduplicate: if we already have one from results (bootstrap or loop), keep only that one
    from_results = [p for p in resolved_patches if p.get("source") in ("bootstrap", "bootstrap_results_only", "loop")]
    from_cache = [p for p in resolved_patches if p.get("source") == "cache"]
    if from_results:
        resolved_patches = from_results
    else:
        resolved_patches = from_cache

    # Resolve patch for each where patch_pre is None
    for rp in resolved_patches:
        if rp.get("patch_pre") is not None:
            continue
        patch = patches_by_round_idx.get((rp["round"], rp["idx"]))
        if not patch and rp["round"] >= 0:
            for (r, i), p in patches_by_round_idx.items():
                if r == rp["round"]:
                    patch = p
                    break
        if not patch and patches_by_round_idx:
            patch = list(patches_by_round_idx.values())[0]
        rp["patch_pre"] = patch

    analyses = []
    for rp in resolved_patches:
        patch = rp.get("patch_pre") or ""
        alignment = analyze_intention_patch_alignment(rp["intention"], patch)
        analyses.append({
            "key": rp["key"],
            "round": rp["round"],
            "intention": rp["intention"],
            "patch": patch[:500] + "..." if patch and len(patch) > 500 else patch,
            "alignment": alignment,
            "resolved": True,
            "source": rp.get("source", "cache"),
        })

    return {
        "instance_id": instance_id,
        "resolved_count": len(resolved_patches),
        "analyses": analyses,
    }

def main():
    base_dir = Path("/mnt/bystry/research/swe/dataset_runs/bash_only_dev_tabu/bash_only_dev")
    results_file = base_dir / "results.jsonl"
    results_by_id: Dict[str, Dict] = {}
    if results_file.exists():
        for line in results_file.read_text(encoding="utf-8").strip().split("\n"):
            if not line:
                continue
            row = json.loads(line)
            results_by_id[row["instance_id"]] = row

    # Find all tabu instances (only django__* directories)
    instances = sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("django__")
    )

    all_results = []
    total_resolved = 0
    total_aligned = 0

    for instance_dir in instances:
        print(f"Analyzing {instance_dir.name}...")
        results_row = results_by_id.get(instance_dir.name)
        result = analyze_instance(instance_dir, results_row=results_row)
        all_results.append(result)
        
        for analysis in result["analyses"]:
            total_resolved += 1
            if analysis["alignment"]["aligned"]:
                total_aligned += 1
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY (wszystkie instancje tabu)")
    print("="*80)
    print(f"Wszystkie instancje w tabu: {len(instances)}")
    print(f"Instancje z co najmniej jednym resolved: {sum(1 for r in all_results if r['resolved_count'] > 0)}")
    print(f"Łącznie resolved patches: {total_resolved}")
    if total_resolved > 0:
        print(f"Zgodne z intencjami: {total_aligned} ({total_aligned/total_resolved*100:.1f}%)")
        print(f"Niezgodne: {total_resolved - total_aligned} ({(total_resolved-total_aligned)/total_resolved*100:.1f}%)")
    
    # Lista wszystkich instancji
    print("\n" + "-"*80)
    print("WSZYSTKIE INSTANCJE TABU (status resolved)")
    print("-"*80)
    for result in all_results:
        status = f"resolved: {result['resolved_count']}" if result["resolved_count"] > 0 else "brak resolved"
        print(f"  {result['instance_id']}: {status}")
    
    # Print detailed results (tylko te z resolved)
    print("\n" + "="*80)
    print("SZCZEGÓŁY: intencja vs patch (tylko instancje z resolved)")
    print("="*80)
    
    for result in all_results:
        if result["resolved_count"] > 0:
            print(f"\n{result['instance_id']}: {result['resolved_count']} resolved patch(es)")
            for analysis in result["analyses"]:
                aligned_str = "✓ ALIGNED" if analysis["alignment"]["aligned"] else "✗ NOT ALIGNED"
                print(f"  Round {analysis['round']}: {aligned_str} (score: {analysis['alignment']['score']:.2f})")
                print(f"    Intention: {analysis['intention'][:100]}...")
                if analysis["patch"]:
                    print(f"    Patch preview: {analysis['patch'][:100]}...")
                print()
    
    # Save results
    output_file = Path("/mnt/bystry/research/swe/tabu_intention_analysis.json")
    instances_with_resolved = [r["instance_id"] for r in all_results if r["resolved_count"] > 0]
    instances_without_resolved = [r["instance_id"] for r in all_results if r["resolved_count"] == 0]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_instances": len(instances),
                "instances_with_resolved": len(instances_with_resolved),
                "instances_without_resolved": len(instances_without_resolved),
                "total_resolved_patches": total_resolved,
                "total_aligned": total_aligned,
                "alignment_rate": total_aligned / total_resolved if total_resolved > 0 else 0
            },
            "all_instance_ids": [r["instance_id"] for r in all_results],
            "instances_with_resolved_ids": instances_with_resolved,
            "instances_without_resolved_ids": instances_without_resolved,
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
