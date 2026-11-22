#!/usr/bin/env python3
"""Materialize the frozen EFRM v2 LODO manifests and shared target folds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from sklearn.model_selection import KFold
import numpy as np
import yaml


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.data.clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT
from efrm_pytorch.data import EFRMPairedWindowAdapter, EFRMSyncPretrainDataset
from efrm_pytorch.tasks import EFRMUnifiedTaskDataset, TASK_SPECS


PROTOCOL_ID = "efrm_lodo_full_target_fivefold_v2"
DATASET_IDS = (
    "eeg_fnirs_single_trial",
    "simultaneous_eeg_nirs",
    "refed",
    "visual_cognitive_motivation",
)
DEFAULT_OUTPUT_ROOT = (
    METHOD_ROOT / "runs/formal" / PROTOCOL_ID
)
DEFAULT_STRICT_REGISTRY = (
    METHOD_ROOT.parent / "STA-Net-PyTorch" / "split_registry"
)
DEFAULT_SAMPLE_REGISTRY = (
    METHOD_ROOT.parent
    / "STA-Net-PyTorch/runs/fivefold"
    / "20260727_sta_net_no_artifact_mask_converged_5fold_v1"
    / "split_registry"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def first_kfold(values: list[str], *, seed: int) -> tuple[list[str], list[str]]:
    array = np.asarray(sorted(values), dtype=object)
    train, validation = next(
        KFold(n_splits=5, shuffle=True, random_state=seed).split(array)
    )
    return sorted(array[train].tolist()), sorted(array[validation].tolist())


def _sample_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pretrain_index": index,
        "dataset_id": str(row["dataset_id"]),
        "subject": str(row["subject"]),
        "record_id": str(row["record_id"]),
        "join_key": str(row["join_key"]),
        "event_index": int(row["event_index"]),
        "window_offset_s": float(row["window_offset_s"]),
        "task_namespace": str(row["task_namespace"]),
        "condition": str(row["condition"]),
    }


def _validate_shared_manifest(
    *,
    public_path: Path,
    protected_path: Path,
    dataset: EFRMUnifiedTaskDataset,
    task: str,
) -> dict[str, Any]:
    public = json.loads(public_path.read_text(encoding="utf-8"))
    protected = json.loads(protected_path.read_text(encoding="utf-8"))
    if public.get("schema") != "sta_net_split_registry_v2":
        raise ValueError(f"unsupported shared public schema: {public_path}")
    if public.get("task") != task or protected.get("task") != task:
        raise RuntimeError(f"task mismatch in shared registry for {task}")
    if public.get("protected_test_opened") is not False:
        raise PermissionError(f"shared public fold is already open: {public_path}")
    if "test_indices" in public:
        raise PermissionError(f"public fold exposes protected indices: {public_path}")
    if str(public.get("metadata_sha256")) != dataset.metadata_fingerprint():
        raise RuntimeError(f"metadata fingerprint mismatch for {task}")

    train = {int(value) for value in public["train_indices"]}
    validation = {int(value) for value in public["validation_indices"]}
    test = {int(value) for value in protected["test_indices"]}
    expected = set(range(len(dataset)))
    if not train or not validation or not test:
        raise RuntimeError(f"empty shared partition for {task}")
    if train & validation or train & test or validation & test:
        raise RuntimeError(f"overlapping shared partitions for {task}")
    if train | validation | test != expected:
        raise RuntimeError(f"shared fold does not cover full target task {task}")
    if public["protected_test"]["indices_sha256"] != protected["indices_sha256"]:
        raise RuntimeError(f"public/protected hash mismatch for {task}")
    return {
        "public_sha256": file_hash(public_path),
        "protected_sha256": file_hash(protected_path),
        "train_sample_count": len(train),
        "validation_sample_count": len(validation),
        "protected_sample_count": len(test),
        "protected_indices_sha256": protected["indices_sha256"],
    }


def _copy_shared_registry(
    *,
    output_root: Path,
    cache_root: str,
    strict_root: Path,
    sample_root: Path,
) -> dict[str, Any]:
    destination = output_root / "protocol/shared_full_target_fold_registry"
    rows: list[dict[str, Any]] = []
    for task, spec in TASK_SPECS.items():
        dataset = EFRMUnifiedTaskDataset(spec, cache_root=cache_root)
        for protocol in ("strict_cross_subject", "sample_random"):
            for outer in range(5):
                if protocol == "strict_cross_subject":
                    source_public = (
                        strict_root
                        / task
                        / "cross_subject/public"
                        / f"outer{outer}_inner0.json"
                    )
                    source_protected = (
                        strict_root
                        / task
                        / "cross_subject/protected"
                        / f"outer{outer}.json"
                    )
                else:
                    source_public = (
                        sample_root
                        / task
                        / "sample_random/public"
                        / f"outer{outer}.json"
                    )
                    source_protected = (
                        sample_root
                        / task
                        / "sample_random/protected"
                        / f"outer{outer}.json"
                    )
                if not source_public.is_file() or not source_protected.is_file():
                    raise FileNotFoundError(
                        f"missing shared comparison fold for {task}/{protocol}/"
                        f"outer{outer}"
                    )
                audit = _validate_shared_manifest(
                    public_path=source_public,
                    protected_path=source_protected,
                    dataset=dataset,
                    task=task,
                )
                public_destination = (
                    destination / task / protocol / "public" / f"outer{outer}.json"
                )
                protected_destination = (
                    destination
                    / task
                    / protocol
                    / "protected"
                    / f"outer{outer}.json"
                )
                public_destination.parent.mkdir(parents=True, exist_ok=True)
                protected_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_public, public_destination)
                shutil.copy2(source_protected, protected_destination)
                if (
                    file_hash(public_destination) != audit["public_sha256"]
                    or file_hash(protected_destination) != audit["protected_sha256"]
                ):
                    raise RuntimeError("shared registry copy hash drifted")
                rows.append(
                    {
                        "task": task,
                        "dataset_id": spec.dataset_id,
                        "protocol": protocol,
                        "outer_fold": outer,
                        "public_path": str(public_destination.resolve()),
                        "protected_path": str(protected_destination.resolve()),
                        "source_public_path": str(source_public.resolve()),
                        "source_protected_path": str(source_protected.resolve()),
                        **audit,
                    }
                )
    manifest: dict[str, Any] = {
        "schema": "method_neutral_full_target_fold_registry_v2",
        "protocol_id": PROTOCOL_ID,
        "created_at": utc_now(),
        "authority": "shared_comparison_registry_exact_copy",
        "outer_folds": 5,
        "inner_folds": 3,
        "outer_seed": 42,
        "inner_seed": "43_plus_outer_index",
        "protected_test_default": "locked",
        "task_fold_count": len(rows),
        "folds": rows,
    }
    manifest["registry_sha256"] = stable_hash(manifest)
    write_json(destination / "registry_manifest.json", manifest)
    return manifest


def build(
    *,
    output_root: Path,
    cache_root: str,
    strict_root: Path,
    sample_root: Path,
) -> dict[str, Any]:
    protocol_root = output_root / "protocol"
    if protocol_root.exists():
        raise FileExistsError(
            f"v2 protocol artifacts already exist and are immutable: {protocol_root}"
        )
    protocol_root.mkdir(parents=True)
    shutil.copy2(
        METHOD_ROOT / "sources/lodo_full_target_fivefold_v2.yaml",
        protocol_root / "lodo_full_target_fivefold_v2.yaml",
    )
    shutil.copy2(
        METHOD_ROOT
        / "sources/20260727_LODO_FULL_TARGET_FIVEFOLD_PROTOCOL_FREEZE.md",
        protocol_root / "20260727_LODO_FULL_TARGET_FIVEFOLD_PROTOCOL_FREEZE.md",
    )

    dataset = EFRMSyncPretrainDataset(
        cache_root=cache_root,
        dataset_ids=DATASET_IDS,
        seed=42,
        adapter=EFRMPairedWindowAdapter(duration_s=8.0),
    )
    metadata = [
        _sample_row(index, dataset.lightweight_metadata(index))
        for index in range(len(dataset))
    ]
    observed = {str(row["dataset_id"]) for row in metadata}
    if observed != set(DATASET_IDS):
        raise RuntimeError(
            f"pretraining datasets differ from frozen v2: {sorted(observed)}"
        )

    base_config = yaml.safe_load(
        (METHOD_ROOT / "configs/pretrain_sync.yaml").read_text(encoding="utf-8")
    )
    lodo_rows: list[dict[str, Any]] = []
    for target_dataset in DATASET_IDS:
        included = [value for value in DATASET_IDS if value != target_dataset]
        dataset_rows: dict[str, Any] = {}
        complete_samples: list[dict[str, Any]] = []
        for dataset_id in included:
            samples = [
                row for row in metadata if row["dataset_id"] == dataset_id
            ]
            subjects = sorted({str(row["subject"]) for row in samples})
            selection_train, selection_validation = first_kfold(subjects, seed=43)
            dataset_rows[dataset_id] = {
                "all_subjects": subjects,
                "all_subject_count": len(subjects),
                "selection_train_subjects": selection_train,
                "selection_validation_subjects": selection_validation,
                "complete_sample_indices": [
                    int(row["pretrain_index"]) for row in samples
                ],
                "complete_samples_sha256": stable_hash(samples),
                "event_index_fingerprint": stable_hash(
                    [
                        {
                            key: row[key]
                            for key in (
                                "subject",
                                "record_id",
                                "join_key",
                                "event_index",
                                "window_offset_s",
                                "task_namespace",
                                "condition",
                            )
                        }
                        for row in samples
                    ]
                ),
            }
            complete_samples.extend(samples)
        manifest: dict[str, Any] = {
            "schema": "efrm_lodo_pretraining_manifest_v2",
            "protocol_id": PROTOCOL_ID,
            "created_at": utc_now(),
            "excluded_target_dataset": target_dataset,
            "included_datasets": included,
            "target_dataset_exposure": False,
            "selection_splitter": {
                "unit": "canonical_subject_id",
                "scope": "within_each_included_dataset",
                "type": "KFold",
                "n_splits": 5,
                "shuffle": True,
                "random_state": 43,
                "validation_partition": "first_test_partition",
            },
            "pretrain_metadata_sha256": stable_hash(metadata),
            "complete_included_sample_count": len(complete_samples),
            "complete_included_samples_sha256": stable_hash(complete_samples),
            "datasets": dataset_rows,
            "implementation_hashes": {
                "build_lodo_protocol.py": file_hash(Path(__file__)),
                "efrm_pytorch/protocol.py": file_hash(
                    METHOD_ROOT / "efrm_pytorch/protocol.py"
                ),
                "train_pretrain.py": file_hash(METHOD_ROOT / "train_pretrain.py"),
            },
        }
        manifest["manifest_sha256"] = stable_hash(manifest)
        manifest_path = (
            protocol_root
            / "lodo_manifests"
            / f"exclude_{target_dataset}.json"
        )
        write_json(manifest_path, manifest)

        config_dir = protocol_root / "configs" / target_dataset
        selection_config = yaml.safe_load(yaml.safe_dump(base_config))
        selection_config["data"]["dataset_ids"] = included
        selection_config["data"]["lodo_manifest"] = str(manifest_path.resolve())
        selection_config["data"]["split_strategy"] = "lodo_selection_v2"
        selection_config["formal_protocol"] = {
            "protocol_id": PROTOCOL_ID,
            "lodo_stage": "selection",
            "excluded_target_dataset": target_dataset,
            "lodo_manifest_sha256": file_hash(manifest_path),
        }
        final_config = yaml.safe_load(yaml.safe_dump(selection_config))
        final_config["data"]["split_strategy"] = "lodo_final_refit_v2"
        final_config["formal_protocol"]["lodo_stage"] = "final_refit"
        final_config["formal_protocol"]["requires_explicit_selected_epoch"] = True
        config_dir.mkdir(parents=True, exist_ok=True)
        selection_path = config_dir / "stage_a_selection.yaml"
        final_path = config_dir / "stage_b_final_refit.yaml"
        selection_path.write_text(
            yaml.safe_dump(selection_config, sort_keys=False), encoding="utf-8"
        )
        final_path.write_text(
            yaml.safe_dump(final_config, sort_keys=False), encoding="utf-8"
        )
        lodo_rows.append(
            {
                "excluded_target_dataset": target_dataset,
                "included_datasets": included,
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": file_hash(manifest_path),
                "selection_config": str(selection_path.resolve()),
                "selection_config_sha256": file_hash(selection_path),
                "final_refit_config": str(final_path.resolve()),
                "final_refit_config_sha256": file_hash(final_path),
                "selection_run_id": (
                    f"{PROTOCOL_ID}__exclude_{target_dataset}__stage_a_seed42"
                ),
                "final_refit_run_id": (
                    f"{PROTOCOL_ID}__exclude_{target_dataset}__stage_b_seed42"
                ),
            }
        )

    registry = _copy_shared_registry(
        output_root=output_root,
        cache_root=cache_root,
        strict_root=strict_root,
        sample_root=sample_root,
    )
    matrix: dict[str, Any] = {
        "schema": "efrm_lodo_pretraining_job_matrix_v2",
        "protocol_id": PROTOCOL_ID,
        "created_at": utc_now(),
        "pretraining_seed": 42,
        "selection_maximum_epochs": 100,
        "selection_minimum_epochs": 20,
        "selection_patience": 15,
        "lodo_jobs": lodo_rows,
        "shared_fold_registry": str(
            (
                protocol_root
                / "shared_full_target_fold_registry/registry_manifest.json"
            ).resolve()
        ),
        "shared_fold_registry_sha256": registry["registry_sha256"],
    }
    matrix["job_matrix_sha256"] = stable_hash(matrix)
    write_json(protocol_root / "pretraining_job_matrix.json", matrix)
    status = {
        "schema": "efrm_lodo_protocol_status_v2",
        "protocol_id": PROTOCOL_ID,
        "status": "manifests_and_shared_folds_materialized",
        "created_at": utc_now(),
        "target_dataset_exposure": False,
        "lodo_selection_jobs": 4,
        "lodo_final_refit_jobs": 4,
        "selection_completed": 0,
        "final_refit_completed": 0,
        "protected_test_opened": False,
        "job_matrix_sha256": matrix["job_matrix_sha256"],
        "shared_fold_registry_sha256": registry["registry_sha256"],
    }
    write_json(output_root / "status.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--cache-root", default=DEFAULT_CLEAN_CACHE_ROOT
    )
    parser.add_argument(
        "--strict-registry-root", default=str(DEFAULT_STRICT_REGISTRY)
    )
    parser.add_argument(
        "--sample-registry-root", default=str(DEFAULT_SAMPLE_REGISTRY)
    )
    args = parser.parse_args()
    result = build(
        output_root=Path(args.output_root).resolve(),
        cache_root=args.cache_root,
        strict_root=Path(args.strict_registry_root).resolve(),
        sample_root=Path(args.sample_registry_root).resolve(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
