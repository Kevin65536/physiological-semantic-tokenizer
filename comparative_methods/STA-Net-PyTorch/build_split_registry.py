#!/usr/bin/env python3
"""Build immutable public/protected split registries for both evaluation protocols."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.data.clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT
from sta_net_pytorch.data import STANetUnifiedTaskDataset, get_sta_net_task_spec
from sta_net_pytorch.splits import (
    build_cross_subject_registry,
    build_single_subject_registry,
    write_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(METHOD_ROOT / "split_registry"))
    parser.add_argument("--cache-root", default=DEFAULT_CLEAN_CACHE_ROOT)
    parser.add_argument("--tasks", nargs="+", default=[
        "motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual", "refed_regression"
    ])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    summary = []
    for task in args.tasks:
        dataset = STANetUnifiedTaskDataset(get_sta_net_task_spec(task), cache_root=args.cache_root)
        cross_public, cross_protected = build_cross_subject_registry(
            dataset, seed=args.seed, outer_folds=args.outer_folds, inner_folds=args.inner_folds
        )
        single_public, single_protected = build_single_subject_registry(dataset, seed=args.seed)
        write_registry(cross_public, cross_protected, root / task / "cross_subject")
        write_registry(single_public, single_protected, root / task / "single_subject")
        summary.append({
            "task": task,
            "cross_subject_public_folds": len(cross_public),
            "cross_subject_outer_folds": len(cross_protected),
            "single_subject_public_folds": len(single_public),
            "single_subject_test_folds": len(single_protected),
        })
    root.mkdir(parents=True, exist_ok=True)
    (root / "registry_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "output_root": str(root), "tasks": summary}, indent=2))


if __name__ == "__main__":
    main()
