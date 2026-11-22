#!/usr/bin/env python3
"""Audit shared public comparison inputs without opening protected manifests.

The audit uses one strict-cross-subject public fold as the development surface.
It verifies the method-neutral registry and public split fingerprints, then
summarizes every selected public sample's measured channels, geometry, masks,
and targets.  Paths containing a ``protected`` component are rejected before
they can be read.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EFRM_ROOT = REPO_ROOT / "comparative_methods/EFRM-PyTorch"
for import_path in (REPO_ROOT, EFRM_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from src.data.clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT
from efrm_pytorch.tasks import EFRMUnifiedTaskDataset, TASK_SPECS


TASK_ORDER = (
    "motor_imagery",
    "mental_arithmetic",
    "wg",
    "nback",
    "dsr",
    "visual",
    "refed_regression",
)
EXPECTED_REGISTRY_SCHEMA = "method_neutral_full_target_fold_registry_v2"
EXPECTED_REGISTRY_SHA256 = "2a10b36db85dba6ec5543edc7810ff85d978ea5af8c79fda3d38a1e5cfd11106"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing to read protected path: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {resolved}")
    if value.get("protected_test_opened", value.get("reserved_test_opened", False)):
        raise PermissionError(f"public manifest reports opened protected data: {resolved}")
    forbidden = {"test_indices", "protected_indices", "reserved_test_indices"}.intersection(value)
    if forbidden:
        raise PermissionError(f"public manifest exposes protected indices: {sorted(forbidden)}")
    return value


def registry_manifest(path: Path) -> dict[str, Any]:
    value = public_json(path)
    if value.get("schema") != EXPECTED_REGISTRY_SCHEMA:
        raise ValueError(f"unexpected registry schema: {value.get('schema')!r}")
    if value.get("registry_sha256") != EXPECTED_REGISTRY_SHA256:
        raise ValueError("method-neutral registry fingerprint drifted")
    if value.get("protected_test_default") != "locked":
        raise PermissionError("method-neutral registry does not default to a locked protected test")
    return value


def strict_public_entry(
    registry: Mapping[str, Any], *, task: str, outer_fold: int
) -> Mapping[str, Any]:
    matches = [
        row
        for row in registry["folds"]
        if row.get("task") == task
        and row.get("protocol") == "strict_cross_subject"
        and int(row.get("outer_fold", -1)) == outer_fold
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one strict public registry entry for {task}/outer{outer_fold}, "
            f"found {len(matches)}"
        )
    return matches[0]


def crop_mask(mask: Sequence[bool], *, rate_hz: float, duration_s: float) -> np.ndarray:
    required = int(round(float(rate_hz) * float(duration_s)))
    source = np.asarray(mask, dtype=bool).reshape(-1)
    output = np.zeros(required, dtype=bool)
    usable = min(required, source.size)
    output[:usable] = source[:usable]
    return output


def panel_key(names: Iterable[str]) -> str:
    return "|".join(str(value) for value in names)


def summarize_sample(sample: Mapping[str, Any], *, duration_s: float) -> dict[str, Any]:
    eeg_names = tuple(str(value) for value in sample["channel_names"]["eeg"])
    fnirs_names = tuple(str(value) for value in sample["channel_names"]["fnirs"])
    if len(eeg_names) != len(set(eeg_names)) or len(fnirs_names) != len(set(fnirs_names)):
        raise ValueError(f"non-unique measured channel identity for {sample['join_key']}")
    eeg = np.asarray(sample["eeg"])
    fnirs = np.asarray(sample["fnirs"])
    if eeg.shape[0] != len(eeg_names) or fnirs.shape[0] != len(fnirs_names):
        raise ValueError(f"channel metadata does not match arrays for {sample['join_key']}")
    if not np.isfinite(eeg).all() or not np.isfinite(fnirs).all():
        raise ValueError(f"non-finite public signal for {sample['join_key']}")

    eeg_mask = crop_mask(
        sample["analysis_valid_mask"]["eeg"],
        rate_hz=float(sample["sample_rate_hz"]["eeg"]),
        duration_s=duration_s,
    )
    fnirs_mask = crop_mask(
        sample["analysis_valid_mask"]["fnirs"],
        rate_hz=float(sample["sample_rate_hz"]["fnirs"]),
        duration_s=duration_s,
    )
    eeg_bad = np.asarray(sample["bad_channel_mask"]["eeg"], dtype=bool)
    fnirs_bad = np.asarray(sample["bad_channel_mask"]["fnirs"], dtype=bool)
    eeg_geometry = list(sample["channel_geometry"]["eeg"])
    fnirs_geometry = list(sample["channel_geometry"]["fnirs"])
    if len(eeg_bad) != len(eeg_names) or len(eeg_geometry) != len(eeg_names):
        raise ValueError(f"EEG mask/geometry width mismatch for {sample['join_key']}")
    if len(fnirs_bad) != len(fnirs_names) or len(fnirs_geometry) != len(fnirs_names):
        raise ValueError(f"fNIRS mask/geometry width mismatch for {sample['join_key']}")

    roles = tuple(str(value) for value in sample["component_roles"]["fnirs"])
    role_counts = Counter(roles)
    target_valid = sample.get("target_valid_mask")
    return {
        "join_key": str(sample["join_key"]),
        "eeg_panel": eeg_names,
        "fnirs_panel": fnirs_names,
        "eeg_good_channels": tuple(
            name for name, is_bad in zip(eeg_names, eeg_bad, strict=True) if not is_bad
        ),
        "eeg_bad_channel_count": int(eeg_bad.sum()),
        "fnirs_bad_channel_count": int(fnirs_bad.sum()),
        "eeg_geometry_position_count": sum(bool(row.get("position_available")) for row in eeg_geometry),
        "fnirs_geometry_position_count": sum(
            bool(row.get("position_available")) for row in fnirs_geometry
        ),
        "eeg_full_task_support": bool(eeg_mask.all()),
        "fnirs_full_task_support": bool(fnirs_mask.all()),
        "eeg_valid_fraction": float(eeg_mask.mean()),
        "fnirs_valid_fraction": float(fnirs_mask.mean()),
        "fnirs_hbo_count": int(role_counts["HbO"]),
        "fnirs_hbr_count": int(role_counts["HbR"]),
        "target_valid_fraction": (
            float(np.asarray(target_valid, dtype=bool).mean()) if target_valid is not None else None
        ),
    }


def clear_record_cache(dataset: EFRMUnifiedTaskDataset) -> None:
    cache = getattr(dataset.base, "_record_cache", None)
    if isinstance(cache, dict):
        cache.clear()
    gc.collect()


def audited_indices_by_record(
    dataset: EFRMUnifiedTaskDataset,
    public_indices: Sequence[int],
    *,
    max_records: int,
    max_samples_per_record: int,
) -> list[list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in public_indices:
        grouped[str(dataset.lightweight_metadata(index)["join_key"])].append(int(index))
    output = []
    for join_key in sorted(grouped):
        selected = grouped[join_key]
        if max_samples_per_record > 0:
            selected = selected[:max_samples_per_record]
        output.append(selected)
        if max_records > 0 and len(output) >= max_records:
            break
    return output


def audit_task(
    *,
    task: str,
    entry: Mapping[str, Any],
    cache_root: Path,
    max_records: int,
    max_samples_per_record: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    public_path = Path(str(entry["public_path"]))
    manifest = public_json(public_path)
    actual_public_sha256 = sha256_file(public_path)
    if actual_public_sha256 != entry["public_sha256"]:
        raise RuntimeError(f"public split hash drifted for {task}: {actual_public_sha256}")

    spec = TASK_SPECS[task]
    dataset = EFRMUnifiedTaskDataset(spec, cache_root=str(cache_root))
    train, validation = dataset.validate_shared_public_split(public_path)
    public_indices = train + validation
    if set(train).intersection(validation):
        raise RuntimeError(f"public train/validation overlap for {task}")
    rows = [dataset.lightweight_metadata(index) for index in public_indices]
    groups = audited_indices_by_record(
        dataset,
        public_indices,
        max_records=max_records,
        max_samples_per_record=max_samples_per_record,
    )
    sample_summaries: list[dict[str, Any]] = []
    for indices in groups:
        for index in indices:
            sample = dataset.base[dataset.indices[int(index)]]
            sample_summaries.append(summarize_sample(sample, duration_s=spec.input_duration_s))
        clear_record_cache(dataset)

    conditions = Counter(str(row["condition"]) for row in rows)
    public_record_count = len({str(row["join_key"]) for row in rows})
    task_report = {
        "dataset_id": spec.dataset_id,
        "task_type": spec.task_type,
        "input_duration_s": spec.input_duration_s,
        "public_manifest": str(public_path.resolve()),
        "public_manifest_sha256": actual_public_sha256,
        "metadata_sha256": manifest.get("metadata_sha256"),
        "split_sha256": manifest.get("split_sha256"),
        "train_sample_count": len(train),
        "validation_sample_count": len(validation),
        "public_subject_count": len({str(row["subject"]) for row in rows}),
        "public_record_count": public_record_count,
        "condition_counts": dict(sorted(conditions.items())),
        "audited_record_count": len(groups),
        "audited_sample_count": len(sample_summaries),
        "all_public_records_audited": len(groups) == public_record_count,
        "all_public_samples_audited": len(sample_summaries) == len(public_indices),
        "eeg_panel_counts": dict(
            sorted(Counter(panel_key(row["eeg_panel"]) for row in sample_summaries).items())
        ),
        "fnirs_panel_counts": dict(
            sorted(Counter(panel_key(row["fnirs_panel"]) for row in sample_summaries).items())
        ),
        "eeg_full_task_support_count": sum(row["eeg_full_task_support"] for row in sample_summaries),
        "fnirs_full_task_support_count": sum(
            row["fnirs_full_task_support"] for row in sample_summaries
        ),
        "minimum_eeg_valid_fraction": min(
            (row["eeg_valid_fraction"] for row in sample_summaries), default=None
        ),
        "minimum_fnirs_valid_fraction": min(
            (row["fnirs_valid_fraction"] for row in sample_summaries), default=None
        ),
        "maximum_eeg_bad_channel_count": max(
            (row["eeg_bad_channel_count"] for row in sample_summaries), default=None
        ),
        "minimum_eeg_geometry_fraction": min(
            (
                row["eeg_geometry_position_count"] / len(row["eeg_panel"])
                for row in sample_summaries
            ),
            default=None,
        ),
        "minimum_fnirs_geometry_fraction": min(
            (
                row["fnirs_geometry_position_count"] / len(row["fnirs_panel"])
                for row in sample_summaries
            ),
            default=None,
        ),
        "minimum_target_valid_fraction": min(
            (
                row["target_valid_fraction"]
                for row in sample_summaries
                if row["target_valid_fraction"] is not None
            ),
            default=None,
        ),
        "protected_test_opened": False,
    }
    del dataset
    gc.collect()
    return task_report, sample_summaries


def ordered_intersection(panels: Sequence[Sequence[str]]) -> list[str]:
    if not panels:
        return []
    common = set(panels[0])
    for panel in panels[1:]:
        common.intersection_update(panel)
    return [name for name in panels[0] if name in common]


def global_support(
    summaries_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reve_position_names: set[str],
) -> dict[str, Any]:
    all_summaries = [row for task_rows in summaries_by_task.values() for row in task_rows]
    all_panels = [tuple(row["eeg_good_channels"]) for row in all_summaries]
    cross_dataset_common = ordered_intersection(all_panels)
    task_support: dict[str, Any] = {}
    for task in TASK_ORDER:
        rows = list(summaries_by_task.get(task, ()))
        panels = [tuple(row["eeg_good_channels"]) for row in rows]
        common = ordered_intersection(panels)
        common_reve = [name for name in common if name in reve_position_names]
        all_eeg_full = bool(rows) and all(bool(row["eeg_full_task_support"]) for row in rows)
        all_geometry = bool(rows) and all(
            int(row["eeg_geometry_position_count"]) == len(row["eeg_panel"]) for row in rows
        )
        task_support[task] = {
            "audited_sample_count": len(rows),
            "all_eeg_has_full_task_support": all_eeg_full,
            "all_eeg_geometry_available": all_geometry,
            "common_good_measured_channel_count": len(common),
            "common_good_measured_channels": common,
            "common_reve_position_bank_channel_count": len(common_reve),
            "common_reve_position_bank_channels": common_reve,
            "biot_18_panel": common[:18] if all_eeg_full and len(common) >= 18 else [],
            "biot_16_panel": common[:16] if all_eeg_full and len(common) >= 16 else [],
            "cbramod_supported": all_eeg_full and bool(common),
            "reve_supported": all_eeg_full and all_geometry and bool(common_reve),
        }

    def candidate(count: int) -> dict[str, Any]:
        field = f"biot_{count}_panel"
        task_panels = {
            task: task_support[task][field]
            for task in TASK_ORDER
            if task_support[task][field]
        }
        unsupported_tasks = [task for task in TASK_ORDER if task not in task_panels]
        all_tasks_supported = not unsupported_tasks
        return {
            "all_tasks_supported_on_audited_public_scope": all_tasks_supported,
            "channel_count": count,
            "supported_tasks": list(task_panels),
            "unsupported_tasks": unsupported_tasks,
            "task_panels": task_panels,
            "selection_policy": "frozen_task_panel_real_channels_reordered_no_copy_or_padding",
            "reason": (
                "every_task_has_an_authentic_fixed_panel_with_full_time_support"
                if all_tasks_supported
                else "at_least_one_task_lacks_enough_common_measured_channels_or_time_support"
            ),
        }

    biot18 = candidate(18)
    biot16 = candidate(16)
    if biot18["all_tasks_supported_on_audited_public_scope"]:
        biot_decision = "eeg_six_datasets_18"
    elif biot16["supported_tasks"]:
        biot_decision = "eeg_prest_16"
    else:
        biot_decision = "unsupported"
    full_scope = bool(all_summaries)
    return {
        "audited_sample_count": len(all_summaries),
        "task_support": task_support,
        "cross_dataset_same_name_channel_count": len(cross_dataset_common),
        "cross_dataset_same_name_channels": cross_dataset_common,
        "cross_dataset_same_name_panel_required": False,
        "biot_18": biot18,
        "biot_16": biot16,
        "biot_preliminary_decision": biot_decision,
        "biot_primary_unsupported_tasks": (
            biot18["unsupported_tasks"]
            if biot_decision == "eeg_six_datasets_18"
            else biot16["unsupported_tasks"]
        ),
        "cbramod_all_tasks_supported": all(
            task_support[task]["cbramod_supported"] for task in TASK_ORDER
        ),
        "reve_all_tasks_supported": all(
            task_support[task]["reve_supported"] for task in TASK_ORDER
        ),
        "decision_scope": "audited_public_samples_only" if full_scope else "empty",
    }


def parse_args() -> argparse.Namespace:
    default_registry = (
        EFRM_ROOT
        / "runs/formal/efrm_lodo_full_target_fivefold_v2/protocol"
        / "shared_full_target_fold_registry/registry_manifest.json"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-manifest", type=Path, default=default_registry)
    parser.add_argument(
        "--cache-root", type=Path, default=REPO_ROOT / DEFAULT_CLEAN_CACHE_ROOT
    )
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--max-records-per-task", type=int, default=0)
    parser.add_argument("--max-samples-per-record", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.outer_fold < 5:
        raise SystemExit("--outer-fold must be in [0, 4]")
    if args.max_records_per_task < 0 or args.max_samples_per_record < 0:
        raise SystemExit("audit limits must be non-negative")

    registry = registry_manifest(args.registry_manifest)
    task_reports: dict[str, Any] = {}
    summaries_by_task: dict[str, list[dict[str, Any]]] = {}
    for task in TASK_ORDER:
        print(f"[public-preflight] {task}: loading public contract", flush=True)
        report, summaries = audit_task(
            task=task,
            entry=strict_public_entry(registry, task=task, outer_fold=args.outer_fold),
            cache_root=args.cache_root,
            max_records=args.max_records_per_task,
            max_samples_per_record=args.max_samples_per_record,
        )
        task_reports[task] = report
        summaries_by_task[task] = summaries
        print(
            f"[public-preflight] {task}: {report['audited_sample_count']} public samples, "
            f"{report['audited_record_count']} records audited",
            flush=True,
        )

    position_config = json.loads(
        (
            REPO_ROOT / "comparative_methods/REVE/checkpoints/reve-positions/config.json"
        ).read_text(encoding="utf-8")
    )
    position_names = {str(value) for value in position_config["position_names"]}
    complete = all(report["all_public_samples_audited"] for report in task_reports.values())
    report = {
        "schema": "comparison_public_preflight_v1",
        "status": "complete" if complete else "partial_started",
        "scope": {
            "protocol": "strict_cross_subject",
            "outer_fold": args.outer_fold,
            "partitions": ["train", "validation"],
            "protected_test_opened": False,
            "max_records_per_task": args.max_records_per_task,
            "max_samples_per_record": args.max_samples_per_record,
        },
        "registry": {
            "path": str(args.registry_manifest.resolve()),
            "file_sha256": sha256_file(args.registry_manifest),
            "registry_sha256": registry["registry_sha256"],
            "outer_seed": registry["outer_seed"],
            "inner_seed": registry["inner_seed"],
            "protected_test_default": registry["protected_test_default"],
        },
        "tasks": task_reports,
        "global_eeg_support": global_support(
            summaries_by_task, reve_position_names=position_names
        ),
        "protected_test_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "registry_sha256": report["registry"]["registry_sha256"],
                "audited_sample_count": report["global_eeg_support"]["audited_sample_count"],
                "biot_preliminary_decision": report["global_eeg_support"][
                    "biot_preliminary_decision"
                ],
                "output": str(args.output.resolve()),
                "protected_test_opened": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
