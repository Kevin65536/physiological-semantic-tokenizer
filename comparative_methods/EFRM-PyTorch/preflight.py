#!/usr/bin/env python3
"""Build an auditable EFRM data/split preflight without touching protected files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.data.clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT
from efrm_pytorch.data import EFRMSyncPretrainDataset
from efrm_pytorch.protocol import PretrainingBoundary, role_counts, sha256_file


DEVELOPMENT_MANIFESTS = {
    task: REPO_ROOT / "comparative_methods/STA-Net-PyTorch/runs/training" /
    "20260719_sta_net_all_tasks_v4_optimized_frozen" / task / "split_manifest.json"
    for task in (
        "motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual",
        "refed_regression",
    )
}


def _git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default=DEFAULT_CLEAN_CACHE_ROOT)
    parser.add_argument("--split-manifest", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--mode", default="development_public_only")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if bool(args.split_manifest) != bool(args.task):
        raise SystemExit("--split-manifest and --task must be supplied together")
    if args.split_manifest and len(args.split_manifest) != len(args.task):
        raise SystemExit("each --split-manifest requires one --task")

    if args.split_manifest:
        paths = [Path(value) for value in args.split_manifest]
        tasks = args.task
    else:
        tasks = list(DEVELOPMENT_MANIFESTS)
        paths = [DEVELOPMENT_MANIFESTS[task] for task in tasks]

    boundary = PretrainingBoundary.from_manifests(
        paths, tasks=tasks, mode=args.mode, cache_root=args.cache_root
    )
    dataset = EFRMSyncPretrainDataset(cache_root=args.cache_root)
    train_indices = boundary.indices_for(dataset, "train")
    validation_indices = boundary.indices_for(dataset, "validation")
    if set(train_indices).intersection(validation_indices):
        raise RuntimeError("pretraining train/validation windows overlap")

    vendor = METHOD_ROOT.parent / "EFRM-A-Multimodal-EEG-fNIRS-Representation-learning-Model"
    config = METHOD_ROOT / "configs/pretrain_sync.yaml"
    report = {
        "schema": "efrm_sync_preflight_v1",
        "status": "preflight_passed",
        "protocol_id": "efrm_sync_200_10_variable_channel_v1",
        "protected_test_opened": False,
        "boundary": boundary.manifest(),
        "all_synchronized_data": dataset.contract_summary(),
        "train": role_counts(dataset, train_indices),
        "validation": role_counts(dataset, validation_indices),
        "adapter": dataset.adapter.manifest(),
        "provenance": {
            "upstream_revision": _git_revision(vendor),
            "upstream_path": str(vendor.resolve()),
            "config_path": str(config.resolve()),
            "config_sha256": sha256_file(config),
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "preflight.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "boundary_sha256": report["boundary"]["boundary_sha256"],
        "train": report["train"],
        "validation": report["validation"],
        "output": str(destination),
    }, indent=2))


if __name__ == "__main__":
    main()
