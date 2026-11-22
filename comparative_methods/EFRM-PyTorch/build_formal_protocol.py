#!/usr/bin/env python3
"""Materialize the frozen EFRM source/target cohorts and target fold registries."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import yaml


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.data.clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT
from efrm_pytorch.data import EFRMPairedWindowAdapter, EFRMSyncPretrainDataset
from efrm_pytorch.tasks import EFRMUnifiedTaskDataset, TASK_SPECS


PROTOCOL_ID = "efrm_resource_bounded_dual_protocol_v1"
TASKS = tuple(TASK_SPECS)
PROTOCOLS = {
    "strict_cross_subject": "efrm_source_target_strict_cross_subject_5fold_v1",
    "sample_random": "efrm_source_target_sample_random_5fold_v1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
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


def first_kfold(
    values: list[str], *, n_splits: int, seed: int
) -> tuple[list[str], list[str]]:
    array = np.asarray(values, dtype=object)
    train_index, test_index = next(
        KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(array)
    )
    return sorted(array[train_index].tolist()), sorted(array[test_index].tolist())


def index_hash(indices: Iterable[int]) -> str:
    return stable_hash([int(value) for value in sorted(indices)])


def subject_hash(subjects: Iterable[str]) -> str:
    return stable_hash([str(value) for value in sorted(subjects)])


def class_labels(
    dataset: EFRMUnifiedTaskDataset, indices: Iterable[int]
) -> np.ndarray:
    return np.asarray(
        [
            dataset.class_to_index[
                str(dataset.lightweight_metadata(int(index))["condition"])
            ]
            for index in indices
        ],
        dtype=np.int64,
    )


def partition_indices(
    dataset: EFRMUnifiedTaskDataset, subjects: set[str]
) -> list[int]:
    return [
        index
        for index in range(len(dataset))
        if str(dataset.lightweight_metadata(index)["subject"]) in subjects
    ]


def fold_manifest(
    *,
    task: str,
    protocol: str,
    outer_index: int,
    dataset: EFRMUnifiedTaskDataset,
    train: list[int],
    validation: list[int],
    test: list[int],
    cohort_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_subjects = sorted(
        {str(dataset.lightweight_metadata(index)["subject"]) for index in train}
    )
    validation_subjects = sorted(
        {str(dataset.lightweight_metadata(index)["subject"]) for index in validation}
    )
    test_subjects = sorted(
        {str(dataset.lightweight_metadata(index)["subject"]) for index in test}
    )
    public = {
        "schema": "efrm_target_public_fold_v1",
        "protocol_id": PROTOCOL_ID,
        "reporting_name": PROTOCOLS[protocol],
        "protocol": protocol,
        "task": task,
        "outer_fold": outer_index,
        "inner_fold": 0,
        "outer_seed": 42,
        "inner_seed": 43 + outer_index,
        "metadata_sha256": dataset.metadata_fingerprint(),
        "cohort_manifest_sha256": cohort_sha256,
        "protected_test_opened": False,
        "train_indices": sorted(train),
        "validation_indices": sorted(validation),
        "train_indices_sha256": index_hash(train),
        "validation_indices_sha256": index_hash(validation),
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "protected_test": {
            "indices_sha256": index_hash(test),
            "subject_ids_sha256": subject_hash(test_subjects),
            "sample_count": len(test),
            "subject_count": len(test_subjects),
        },
    }
    public["split_sha256"] = stable_hash(public)
    protected = {
        "schema": "efrm_target_protected_fold_v1",
        "protocol_id": PROTOCOL_ID,
        "reporting_name": PROTOCOLS[protocol],
        "protocol": protocol,
        "task": task,
        "outer_fold": outer_index,
        "metadata_sha256": dataset.metadata_fingerprint(),
        "cohort_manifest_sha256": cohort_sha256,
        "public_split_sha256": public["split_sha256"],
        "protected_test_opened": False,
        "test_indices": sorted(test),
        "test_indices_sha256": index_hash(test),
        "test_subjects": test_subjects,
        "test_subject_ids_sha256": subject_hash(test_subjects),
    }
    protected["protected_split_sha256"] = stable_hash(protected)
    return public, protected


def strict_folds(
    dataset: EFRMUnifiedTaskDataset, target_subjects: list[str]
) -> list[tuple[list[int], list[int], list[int]]]:
    subjects = np.asarray(sorted(target_subjects), dtype=object)
    output = []
    for outer_index, (development_position, test_position) in enumerate(
        KFold(n_splits=5, shuffle=True, random_state=42).split(subjects)
    ):
        development_subjects = subjects[development_position]
        test_subjects = set(subjects[test_position].tolist())
        inner_train_position, inner_validation_position = next(
            KFold(
                n_splits=3, shuffle=True, random_state=43 + outer_index
            ).split(development_subjects)
        )
        train_subjects = set(development_subjects[inner_train_position].tolist())
        validation_subjects = set(
            development_subjects[inner_validation_position].tolist()
        )
        output.append(
            (
                partition_indices(dataset, train_subjects),
                partition_indices(dataset, validation_subjects),
                partition_indices(dataset, test_subjects),
            )
        )
    return output


def sample_random_folds(
    dataset: EFRMUnifiedTaskDataset, target_indices: list[int]
) -> list[tuple[list[int], list[int], list[int]]]:
    target = np.asarray(sorted(target_indices), dtype=np.int64)
    if dataset.spec.task_type == "classification":
        outer_splitter = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=42
        )
        outer_iterator = outer_splitter.split(target, class_labels(dataset, target))
    else:
        outer_iterator = KFold(
            n_splits=5, shuffle=True, random_state=42
        ).split(target)
    output = []
    for outer_index, (development_position, test_position) in enumerate(outer_iterator):
        development = target[development_position]
        test = target[test_position]
        if dataset.spec.task_type == "classification":
            inner_iterator = StratifiedKFold(
                n_splits=3, shuffle=True, random_state=43 + outer_index
            ).split(development, class_labels(dataset, development))
        else:
            inner_iterator = KFold(
                n_splits=3, shuffle=True, random_state=43 + outer_index
            ).split(development)
        inner_train_position, inner_validation_position = next(inner_iterator)
        output.append(
            (
                sorted(development[inner_train_position].tolist()),
                sorted(development[inner_validation_position].tolist()),
                sorted(test.tolist()),
            )
        )
    return output


def validate_folds(
    *,
    dataset: EFRMUnifiedTaskDataset,
    target_indices: list[int],
    folds: list[tuple[list[int], list[int], list[int]]],
    strict: bool,
) -> None:
    expected = set(target_indices)
    tests: list[int] = []
    for train, validation, test in folds:
        sets = [set(train), set(validation), set(test)]
        if not all(sets) or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise RuntimeError("formal fold contains an empty or overlapping partition")
        if set.union(*sets) != expected:
            raise RuntimeError("formal fold does not exactly cover the target task indices")
        if dataset.spec.task_type == "classification":
            expected_classes = set(range(len(dataset.spec.class_names)))
            for role, values in zip(
                ("train", "validation", "test"), (train, validation, test)
            ):
                observed = set(class_labels(dataset, values).tolist())
                if observed != expected_classes:
                    raise RuntimeError(
                        f"formal {role} fold omits classes: "
                        f"{sorted(expected_classes - observed)}"
                    )
        if strict:
            subject_sets = [
                {
                    str(dataset.lightweight_metadata(index)["subject"])
                    for index in values
                }
                for values in (train, validation, test)
            ]
            if any(
                subject_sets[i] & subject_sets[j]
                for i in range(3)
                for j in range(i)
            ):
                raise RuntimeError("strict fold has cross-partition subject overlap")
        tests.extend(test)
    if len(tests) != len(expected) or set(tests) != expected:
        raise RuntimeError("outer tests do not partition every target sample exactly once")


def build_cohort(output_root: Path, cache_root: str) -> dict[str, Any]:
    protocol_root = output_root / "protocol"
    split_root = protocol_root / "split_registry"
    if protocol_root.exists():
        raise FileExistsError(
            f"formal protocol artifacts already exist and are immutable: {protocol_root}"
        )
    pretrain = EFRMSyncPretrainDataset(
        cache_root=cache_root,
        dataset_ids=(
            "eeg_fnirs_single_trial",
            "refed",
            "visual_cognitive_motivation",
            "simultaneous_eeg_nirs",
        ),
        seed=42,
        adapter=EFRMPairedWindowAdapter(duration_s=8.0),
    )
    pretrain_rows = [pretrain.lightweight_metadata(index) for index in range(len(pretrain))]
    dataset_subjects: dict[str, set[str]] = {}
    for row in pretrain_rows:
        dataset_subjects.setdefault(str(row["dataset_id"]), set()).add(str(row["subject"]))

    datasets: dict[str, Any] = {}
    for dataset_id, values in sorted(dataset_subjects.items()):
        subjects = sorted(values)
        target, source = first_kfold(subjects, n_splits=3, seed=42)
        source_train, source_validation = first_kfold(
            source, n_splits=min(5, len(source)), seed=43
        )
        if len(source) < 2 or set(source) & set(target):
            raise RuntimeError(f"invalid frozen cohort for {dataset_id}")
        dataset_rows = [
            row for row in pretrain_rows if str(row["dataset_id"]) == dataset_id
        ]
        datasets[dataset_id] = {
            "all_subjects": subjects,
            "all_subject_ids_sha256": subject_hash(subjects),
            "source_subjects": source,
            "source_train_subjects": source_train,
            "source_validation_subjects": source_validation,
            "target_subjects": target,
            "source_sample_indices_sha256": index_hash(
                index
                for index, row in enumerate(pretrain_rows)
                if str(row["dataset_id"]) == dataset_id
                and str(row["subject"]) in set(source)
            ),
            "target_sample_indices_sha256": index_hash(
                index
                for index, row in enumerate(pretrain_rows)
                if str(row["dataset_id"]) == dataset_id
                and str(row["subject"]) in set(target)
            ),
            "event_index_fingerprint": stable_hash(
                [
                    {
                        "subject": row["subject"],
                        "record_id": row["record_id"],
                        "event_index": row["event_index"],
                        "window_offset_s": row["window_offset_s"],
                    }
                    for row in dataset_rows
                ]
            ),
        }

    implementation_paths = [
        Path(__file__).resolve(),
        METHOD_ROOT / "efrm_pytorch/tasks.py",
        METHOD_ROOT / "efrm_pytorch/protocol.py",
        METHOD_ROOT / "train_pretrain.py",
        METHOD_ROOT / "train_downstream.py",
    ]
    cohort: dict[str, Any] = {
        "schema": "efrm_source_target_cohort_v1",
        "protocol_id": PROTOCOL_ID,
        "created_at": utc_now(),
        "construction": {
            "unit": "canonical_subject_id",
            "splitter": "KFold",
            "n_splits": 3,
            "shuffle": True,
            "random_state": 42,
            "source_partition": "first_test_partition",
            "source_validation_random_state": 43,
        },
        "cache_root": str(Path(cache_root).resolve()),
        "pretrain_metadata_sha256": stable_hash(pretrain_rows),
        "pretrain_contract_sha256": stable_hash(pretrain.contract_summary()),
        "builder_implementation_hashes": {
            str(path.relative_to(REPO_ROOT)): file_hash(path)
            for path in implementation_paths
        },
        "target_opened_during_pretraining": False,
        "datasets": datasets,
        "tasks": {},
    }

    task_datasets: dict[str, EFRMUnifiedTaskDataset] = {}
    for task, spec in TASK_SPECS.items():
        dataset = EFRMUnifiedTaskDataset(spec, cache_root=cache_root)
        task_datasets[task] = dataset
        target_subjects = set(datasets[spec.dataset_id]["target_subjects"])
        eligible = sorted(
            {
                str(dataset.lightweight_metadata(index)["subject"])
                for index in range(len(dataset))
                if str(dataset.lightweight_metadata(index)["subject"])
                in target_subjects
            }
        )
        target_indices = partition_indices(dataset, set(eligible))
        if len(eligible) < 5:
            raise RuntimeError(f"{task} has fewer than five eligible target subjects")
        cohort["tasks"][task] = {
            "dataset_id": spec.dataset_id,
            "eligible_target_subjects": eligible,
            "eligible_target_subject_count": len(eligible),
            "eligible_target_sample_count": len(target_indices),
            "eligible_target_sample_indices_sha256": index_hash(target_indices),
            "metadata_sha256": dataset.metadata_fingerprint(),
        }
    cohort["cohort_sha256"] = stable_hash(cohort)
    cohort_path = protocol_root / "cohort_manifest.json"
    write_json(cohort_path, cohort)
    cohort_file_sha256 = file_hash(cohort_path)
    pretrain_config = yaml.safe_load(
        (METHOD_ROOT / "configs/pretrain_sync.yaml").read_text(encoding="utf-8")
    )
    pretrain_config["data"]["cohort_manifest"] = str(cohort_path.resolve())
    pretrain_config["data"]["split_strategy"] = "source_target_source_only_v1"
    pretrain_config["formal_protocol"] = {
        "protocol_id": PROTOCOL_ID,
        "cohort_manifest_sha256": cohort_file_sha256,
    }
    config_root = protocol_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "pretrain_source_only.yaml").write_text(
        yaml.safe_dump(pretrain_config, sort_keys=False), encoding="utf-8"
    )
    shutil.copy2(
        METHOD_ROOT / "sources/resource_bounded_dual_protocol_v1.yaml",
        protocol_root / "resource_bounded_dual_protocol_v1.yaml",
    )
    status = {
        "schema": "efrm_formal_protocol_status_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "source_cohort_materialized",
        "created_at": utc_now(),
        "cohort_manifest_sha256": cohort_file_sha256,
        "required_job_count": 70,
        "completed_public_jobs": 0,
        "completed_protected_jobs": 0,
        "protected_test_opened": False,
    }
    write_json(output_root / "status.json", status)
    return status


def build_folds(
    output_root: Path, cache_root: str, source_checkpoint: Path
) -> dict[str, Any]:
    protocol_root = output_root / "protocol"
    split_root = protocol_root / "split_registry"
    cohort_path = protocol_root / "cohort_manifest.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort_file_sha256 = file_hash(cohort_path)
    status_path = output_root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "source_pretraining_completed":
        raise RuntimeError(
            "folds may be materialized only after source_pretraining_completed"
        )
    if source_checkpoint.resolve() != Path(status["source_checkpoint"]).resolve():
        raise RuntimeError("requested source checkpoint differs from frozen status")
    if file_hash(source_checkpoint) != status["source_checkpoint_sha256"]:
        raise RuntimeError("frozen source checkpoint hash drifted")
    source_run = source_checkpoint.parent.parent
    source_manifest = json.loads(
        (source_run / "manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("status") != "completed":
        raise RuntimeError("source-only pretraining run is not complete")
    if source_manifest.get("protected_test_opened") is not False:
        raise PermissionError("source-only pretraining reports target/protected access")
    if (protocol_root / "job_matrix.json").exists() or split_root.exists():
        raise FileExistsError("formal fold registry is already materialized and immutable")

    task_datasets = {
        task: EFRMUnifiedTaskDataset(spec, cache_root=cache_root)
        for task, spec in TASK_SPECS.items()
    }
    job_rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for task, dataset in task_datasets.items():
        target_subjects = cohort["tasks"][task]["eligible_target_subjects"]
        target_indices = partition_indices(dataset, set(target_subjects))
        task_coverage: dict[str, Any] = {}
        for protocol, folds in (
            ("strict_cross_subject", strict_folds(dataset, target_subjects)),
            ("sample_random", sample_random_folds(dataset, target_indices)),
        ):
            validate_folds(
                dataset=dataset,
                target_indices=target_indices,
                folds=folds,
                strict=protocol == "strict_cross_subject",
            )
            test_union: list[int] = []
            for outer_index, (train, validation, test) in enumerate(folds):
                public, protected = fold_manifest(
                    task=task,
                    protocol=protocol,
                    outer_index=outer_index,
                    dataset=dataset,
                    train=train,
                    validation=validation,
                    test=test,
                    cohort_sha256=cohort_file_sha256,
                )
                public_path = (
                    split_root / task / protocol / "public" / f"outer{outer_index}.json"
                )
                protected_path = (
                    split_root
                    / task
                    / protocol
                    / "protected"
                    / f"outer{outer_index}.json"
                )
                write_json(public_path, public)
                write_json(protected_path, protected)
                job_rows.append(
                    {
                        "job_id": f"{task}__{protocol}__outer{outer_index}",
                        "task": task,
                        "protocol": protocol,
                        "reporting_name": PROTOCOLS[protocol],
                        "outer_fold": outer_index,
                        "seed": 42,
                        "transfer_mode": "linear_probe",
                        "modality": "paired",
                        "initialization": "pretrained",
                        "public_manifest": str(public_path.resolve()),
                        "public_manifest_sha256": file_hash(public_path),
                        "protected_manifest": str(protected_path.resolve()),
                        "protected_manifest_sha256": file_hash(protected_path),
                        "status": "pending",
                    }
                )
                test_union.extend(test)
            task_coverage[protocol] = {
                "fold_count": 5,
                "test_partition_exact": (
                    len(test_union) == len(target_indices)
                    and set(test_union) == set(target_indices)
                ),
                "test_union_indices_sha256": index_hash(test_union),
            }
        coverage[task] = task_coverage
    if len(job_rows) != 70:
        raise RuntimeError(f"frozen grid must contain 70 jobs, got {len(job_rows)}")

    downstream_config = yaml.safe_load(
        (METHOD_ROOT / "configs/downstream_public_pilot.yaml").read_text(
            encoding="utf-8"
        )
    )
    downstream_config["formal_protocol"] = {
        "protocol_id": PROTOCOL_ID,
        "cohort_manifest_sha256": cohort_file_sha256,
        "source_checkpoint_sha256": status["source_checkpoint_sha256"],
        "hyperparameters_frozen_before_target_outcomes": True,
    }
    downstream_path = protocol_root / "configs/downstream_linear_probe.yaml"
    downstream_path.write_text(
        yaml.safe_dump(downstream_config, sort_keys=False), encoding="utf-8"
    )
    metric_registry = {
        "schema": "efrm_formal_metric_registry_v1",
        "selection": {
            "classification": "validation_macro_f1_max",
            "regression": "validation_masked_scaled_rmse_min",
        },
        "classification": {
            "primary": "macro_f1",
            "companion": "accuracy",
            "required": [
                "balanced_accuracy",
                "cohen_kappa",
                "per_class_precision",
                "per_class_recall",
                "per_class_f1",
                "confusion_matrix",
                "expected_calibration_error",
            ],
        },
        "regression": {
            "primary": "native_ccc",
            "required": [
                "native_mae",
                "native_rmse",
                "native_r2",
                "native_pearson",
                "native_spearman",
                "native_coverage",
                "native_valid_count",
            ],
        },
        "aggregation": {
            "unit": "target_outer_fold",
            "fold_count": 5,
            "sample_sd_ddof": 1,
            "confidence_interval": "two_sided_t_95_df_4",
        },
    }
    metric_path = protocol_root / "metric_registry.json"
    write_json(metric_path, metric_registry)
    implementation_paths = [
        Path(__file__).resolve(),
        METHOD_ROOT / "train_downstream.py",
        METHOD_ROOT / "evaluate_formal_fold.py",
        METHOD_ROOT / "aggregate_formal_results.py",
        METHOD_ROOT / "efrm_pytorch/tasks.py",
        METHOD_ROOT / "efrm_pytorch/metrics.py",
    ]
    job_matrix = {
        "schema": "efrm_formal_job_matrix_v1",
        "protocol_id": PROTOCOL_ID,
        "created_at": utc_now(),
        "cohort_manifest": str(cohort_path.resolve()),
        "cohort_manifest_sha256": cohort_file_sha256,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": status["source_checkpoint_sha256"],
        "downstream_config": str(downstream_path.resolve()),
        "downstream_config_sha256": file_hash(downstream_path),
        "metric_registry": str(metric_path.resolve()),
        "metric_registry_sha256": file_hash(metric_path),
        "implementation_hashes": {
            str(path.relative_to(REPO_ROOT)): file_hash(path)
            for path in implementation_paths
        },
        "required_job_count": 70,
        "protected_test_opened": False,
        "coverage": coverage,
        "jobs": job_rows,
    }
    job_matrix["job_matrix_sha256"] = stable_hash(job_matrix)
    matrix_path = protocol_root / "job_matrix.json"
    write_json(matrix_path, job_matrix)
    status.update(
        {
            "status": "folds_jobs_metrics_frozen",
            "folds_frozen_at": utc_now(),
            "job_matrix_sha256": file_hash(matrix_path),
            "downstream_config_sha256": file_hash(downstream_path),
            "metric_registry_sha256": file_hash(metric_path),
        }
    )
    write_json(status_path, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=str(METHOD_ROOT / f"runs/formal/{PROTOCOL_ID}"),
    )
    parser.add_argument(
        "--cache-root", default=DEFAULT_CLEAN_CACHE_ROOT
    )
    parser.add_argument("--stage", choices=("cohort", "folds"), required=True)
    parser.add_argument("--source-checkpoint")
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    if args.stage == "cohort":
        result = build_cohort(output_root, args.cache_root)
    else:
        if not args.source_checkpoint:
            raise SystemExit("--source-checkpoint is required for --stage folds")
        result = build_folds(
            output_root, args.cache_root, Path(args.source_checkpoint).resolve()
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
