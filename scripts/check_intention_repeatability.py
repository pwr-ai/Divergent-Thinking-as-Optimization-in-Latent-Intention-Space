#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sprawdza powtarzalność intencji na podstawie intentions_log.jsonl.
Użycie: python scripts/check_intention_repeatability.py [ścieżka do intentions_log.jsonl lub katalog csc_runs/instance_id]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


def _normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def sha256_normalized(s: str) -> str:
    return hashlib.sha256(_normalize(s).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Analiza powtarzalności intencji z intentions_log.jsonl")
    ap.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Ścieżka do intentions_log.jsonl lub katalog csc_runs/instance_id (szuka intentions_log.jsonl)",
    )
    ap.add_argument("--out", "-o", help="Zapisz raport do pliku (JSON)")
    args = ap.parse_args()

    if args.path is None:
        # Szukaj wszystkich intentions_log.jsonl w csc_runs
        base = Path("csc_runs")
        if not base.exists():
            print("Podaj ścieżkę lub uruchom z katalogu zawierającego csc_runs/", file=sys.stderr)
            sys.exit(1)
        log_files = list(base.glob("*/intentions_log.jsonl"))
    else:
        p = Path(args.path)
        if p.is_file():
            log_files = [p]
        else:
            log_files = [p / "intentions_log.jsonl"] if (p / "intentions_log.jsonl").exists() else []

    if not log_files:
        print("Nie znaleziono intentions_log.jsonl. Uruchom najpierw cma_loop z zapisem (intentions_log jest tworzony przy każdym suggest).", file=sys.stderr)
        sys.exit(1)

    for log_path in log_files:
        if not log_path.exists():
            continue
        instance_id = log_path.parent.name if "csc_runs" in str(log_path) else log_path.stem
        rows: list[dict] = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        total = len(rows)
        if total == 0:
            print(f"[{instance_id}] Brak wpisów w {log_path}")
            continue

        by_key: dict[str, list[dict]] = defaultdict(list)  # key (sha1 raw) -> list of rows
        by_norm: dict[str, list[dict]] = defaultdict(list)  # sha256(normalized) -> list of rows
        for r in rows:
            intent = r.get("intention", "")
            key = r.get("key") or sha1(intent)
            by_key[key].append(r)
            by_norm[sha256_normalized(intent)].append(r)

        unique_by_key = len(by_key)
        unique_by_norm = len(by_norm)
        duplicates_key = [k for k, v in by_key.items() if len(v) > 1]
        duplicates_norm = [k for k, v in by_norm.items() if len(v) > 1]

        report = {
            "instance_id": instance_id,
            "log_path": str(log_path),
            "total_proposals": total,
            "unique_by_sha1_raw": unique_by_key,
            "unique_by_normalized": unique_by_norm,
            "repeat_rate_sha1": 1.0 - (unique_by_key / total) if total else 0.0,
            "repeat_rate_normalized": 1.0 - (unique_by_norm / total) if total else 0.0,
            "duplicate_keys_count": len(duplicates_key),
            "duplicate_normalized_count": len(duplicates_norm),
            "examples_repeated_by_key": [
                {"key": k, "count": len(by_key[k]), "first_intention_preview": (by_key[k][0].get("intention", ""))[:120]}
                for k in duplicates_key[:5]
            ],
            "examples_repeated_by_norm": [
                {"count": len(by_norm[k]), "first_intention_preview": (by_norm[k][0].get("intention", ""))[:120]}
                for k in duplicates_norm[:5]
            ],
        }

        print(f"\n=== Powtarzalność intencji: {instance_id} ===\n")
        print(f"  Łącznie propozycji:     {total}")
        print(f"  Unikalne (SHA1 raw):    {unique_by_key}  ({100 * unique_by_key / total:.1f}%)" if total else "")
        print(f"  Unikalne (znormaliz.):  {unique_by_norm}  ({100 * unique_by_norm / total:.1f}%)" if total else "")
        print(f"  Powtórzenia (raw):      {total - unique_by_key}  (wsp. powtarzalności: {report['repeat_rate_sha1']:.2%})")
        print(f"  Powtórzenia (norm.):    {total - unique_by_norm}  (wsp. powtarzalności: {report['repeat_rate_normalized']:.2%})")
        if duplicates_key:
            print(f"\n  Przykłady intencji proponowanych wielokrotnie (raw):")
            for ex in report["examples_repeated_by_key"]:
                print(f"    ×{ex['count']}: {ex['first_intention_preview']}...")
        if duplicates_norm and len(duplicates_norm) != len(duplicates_key):
            print(f"\n  Przykłady intencji powtórzonych po normalizacji:")
            for ex in report["examples_repeated_by_norm"]:
                print(f"    ×{ex['count']}: {ex['first_intention_preview']}...")

        if args.out:
            out_path = Path(args.out)
            if len(log_files) > 1:
                out_path = out_path.parent / f"{out_path.stem}_{instance_id}{out_path.suffix}"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n  Raport zapisany: {out_path}")

    print()


if __name__ == "__main__":
    main()
