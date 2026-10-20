#!/usr/bin/env python3
"""Aggregate only the explicitly scheduled public rebenchmark evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.metrics.downstream_benchmark import regression_train_mean_baseline, score_predictions

SCHEMA = "downstream_rebench_aggregate_v1"
BOOTSTRAP_SEED = 20261016
BOOTSTRAP_REPLICATES = 2000
KEYS = ("method", "task", "track", "modality_profile", "baseline_modality")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return value


def _identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _finite_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {name: _finite_json(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return value


def _path(value: str, parent: Path) -> Path:
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else parent / candidate).resolve()


def _job_path(entry: dict[str, Any], parent: Path) -> Path | None:
    command = entry.get("command", [])
    found = []
    for index, item in enumerate(command):
        if item == "--job" and index + 1 < len(command):
            found.append(command[index + 1])
        elif item.startswith("--job="):
            found.append(item.split("=", 1)[1])
    if len(found) > 1:
        raise ValueError("schedule command has multiple --job arguments")
    return _path(found[0], _path(entry.get("cwd", str(parent)), parent)) if found else None


def _failure(row: dict[str, Any], reason: str, status: str | None = None) -> dict[str, Any]:
    return {"schedule_job_id": row["schedule_job_id"], "job_id": row.get("job_id"),
            "status": status or row.get("status", "invalid"), "reason": reason,
            "job_path": row.get("job_path"), "result_path": row.get("result_path")}


def _expected(job: dict[str, Any], parent: Path, cache: dict[Path, dict[str, Any]],
              sources: set[str]) -> tuple[dict[str, tuple[str, str]], int]:
    paths = [_path(job[name], parent) for name in ("inventory_path", "splits_path")]
    for path in paths:
        sources.add(str(path))
        if path not in cache:
            cache[path] = _json(path)
    inventory, splits = [cache[path] for path in paths]
    fold = splits["tasks"][job["task"]]["folds"][int(job["outer_fold"])]
    expected = {}
    for index in fold["eval_indices"]:
        sample = inventory["samples"][index]
        if sample["task_id"] != job["task"]:
            raise ValueError("planned evaluation indices include another task")
        sample_id = str(sample["sample_id"])
        if sample_id in expected:
            raise ValueError("duplicate sample in planned evaluation support")
        expected[sample_id] = (str(sample["subject_id"]), str(sample["record_id"]))
    if not expected:
        raise ValueError("empty planned evaluation support")
    classes = len(inventory["tasks"][job["task"]].get("classes", []))
    return expected, classes


def _prediction_arrays(path: Path, task: str, classes: int,
                       expected: dict[str, tuple[str, str]]) -> tuple[dict[str, np.ndarray], str]:
    names = ("target", "prediction", "mask", "subject_id", "sample_id", "record_id")
    with np.load(path, allow_pickle=False) as retained:
        arrays = {name: np.asarray(retained[name]) for name in names}
    n = len(expected)
    for name in ("subject_id", "sample_id", "record_id"):
        if arrays[name].shape != (n,):
            raise ValueError(f"{name} does not preserve the complete planned evaluation denominator")
        arrays[name] = arrays[name].astype(str)
    ids = arrays["sample_id"].tolist()
    if len(set(ids)) != n or set(ids) != set(expected):
        raise ValueError("predicted sample IDs differ from complete planned evaluation support")
    for sample_id, subject, record in zip(ids, arrays["subject_id"], arrays["record_id"]):
        if expected[sample_id] != (subject, record):
            raise ValueError("prediction subject/record identity differs from frozen inventory")
    shape = (n, 2, 20) if task == "refed_regression" else (n,)
    if any(arrays[name].shape != shape for name in ("target", "prediction", "mask")):
        raise ValueError("prediction, target or mask has the wrong endpoint shape")
    if arrays["mask"].dtype != np.bool_:
        raise ValueError("target mask must be a retained boolean array")
    valid = arrays["mask"]
    if not np.isfinite(arrays["target"][valid]).all() or not np.isfinite(arrays["prediction"][valid]).all():
        raise ValueError("non-finite target or prediction on scored support")
    if task != "refed_regression":
        if classes < 2:
            raise ValueError("classification requires its frozen complete class list")
        for name in ("target", "prediction"):
            values = arrays[name][valid]
            if np.any(values != values.astype(int)) or np.any((values < 0) | (values >= classes)):
                raise ValueError("classification labels outside the frozen class list")
        if any(not valid[arrays["subject_id"] == subject].any() for subject in set(arrays["subject_id"])):
            raise ValueError("subject has no scored classification labels")
    order = np.argsort(arrays["sample_id"])
    support = {name: arrays[name][order].tolist() for name in
               ("sample_id", "subject_id", "record_id", "mask")}
    support["target"] = np.where(valid, arrays["target"], 0)[order].tolist()
    signature = _identity(support)
    return arrays, signature


def _regression_errors(arrays: dict[str, np.ndarray], subjects: list[dict[str, Any]]) -> None:
    for subject in subjects:
        selected_subject = arrays["subject_id"] == subject["subject_id"]
        errors = []
        for record in sorted(set(arrays["record_id"][selected_subject])):
            selected = selected_subject & (arrays["record_id"] == record)
            dimensions = []
            for dimension in range(2):
                mask = arrays["mask"][selected, dimension]
                delta = (arrays["prediction"][selected, dimension] - arrays["target"][selected, dimension])[mask]
                dimensions.append((float(np.mean(np.abs(delta))), float(np.sqrt(np.mean(delta**2)))) if delta.size else (None, None))
            errors.append(dimensions)
        for index, name in enumerate(("mae", "rmse")):
            values = [pair[index] for record in errors for pair in record]
            subject[name] = float(np.mean(values)) if values and all(value is not None for value in values) else None


def _check_refed_baseline(baseline: dict[str, Any], arrays: dict[str, np.ndarray],
                          subjects: list[dict[str, Any]]) -> None:
    if (baseline.get("available") is not True
            or baseline.get("type") != "outer_training_coordinate_mean"
            or baseline.get("coverage_complete") is not True
            or baseline.get("b0") != 0.0):
        raise ValueError("REFED train-mean baseline is unavailable or has incomplete coverage")
    mean = np.asarray(baseline.get("train_mean"), dtype=float)
    counts = np.asarray(baseline.get("train_valid_point_counts"))
    if mean.shape != (2,) or not np.isfinite(mean).all() or counts.shape != (2,) or not np.all(counts > 0):
        raise ValueError("REFED baseline lacks valid outer-training coordinate means/counts")
    saved = baseline.get("subjects", [])
    by_subject = {str(item["subject_id"]): item for item in saved}
    if len(saved) != len(by_subject) or set(by_subject) != {subject["subject_id"] for subject in subjects}:
        raise ValueError("REFED baseline subjects differ from complete scored evaluation support")
    expected = regression_train_mean_baseline(mean, counts, arrays)
    def check(actual, expected):
        if (not isinstance(actual, (int, float)) or not math.isfinite(actual) or actual < 0
                or not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9)):
            raise ValueError("REFED baseline error disagrees with train mean on retained scoring support")
    for subject in expected["subjects"]:
        retained = by_subject[subject["subject_id"]]
        for error in ("mae", "rmse"):
            check(retained.get(error), subject[error])
        records = retained.get("records", [])
        by_record = {str(item["record_id"]): item for item in records}
        expected_records = {str(item["record_id"]): item for item in subject["records"]}
        if len(records) != len(by_record) or set(by_record) != set(expected_records):
            raise ValueError("REFED baseline records differ from complete scored evaluation support")
        for name, record in by_record.items():
            expected_record = expected_records[name]
            if record.get("valid_point_counts") != expected_record["valid_point_counts"]:
                raise ValueError("REFED baseline uses a different native coordinate mask")
            for error in ("mae", "rmse"):
                check(record.get(error), expected_record[error])
                values = record.get(error + "_by_dimension", [])
                if len(values) != 2:
                    raise ValueError("REFED baseline lacks its coordinate-level error evidence")
                for dimension in range(2):
                    check(values[dimension], expected_record[error + "_by_dimension"][dimension])


def _load_result(row: dict[str, Any], job: dict[str, Any], expected: dict[str, tuple[str, str]],
                 classes: int) -> None:
    result = _json(Path(row["result_path"]))
    row["status"] = str(result.get("status", "missing_status"))
    if row["status"] != "completed":
        raise ValueError(f"{result.get('error_type', row['status'])}: {result.get('error', 'no completed result')}")
    if result.get("schema") != "downstream_rebench_result_v1" or result.get("job_identity") != _identity(job):
        raise ValueError("completed result does not match its frozen rebenchmark job contract")
    for name in ("method", "task", "track"):
        if result.get(name) != job.get(name, "lp" if name == "track" else None):
            raise ValueError(f"completed result has a different {name}")
    arrays, signature = _prediction_arrays(Path(row["predictions_path"]), job["task"], classes, expected)
    metrics = score_predictions(arrays["target"], arrays["prediction"], arrays["mask"],
                                arrays["subject_id"], job["task"], classes, arrays["record_id"])
    for subject in metrics["subjects"]:
        for endpoint in [subject, *subject.get("records", [])]:
            if endpoint["score"] is not None and not math.isfinite(endpoint["score"]):
                endpoint["score"] = None
                metrics["coverage_complete"] = False
    if job["task"] == "refed_regression":
        _regression_errors(arrays, metrics["subjects"])
    row.update(metrics=metrics, support_signature=signature, subjects=metrics["subjects"], baselines=result.get("baselines"))
    retained = result.get("metrics", {})
    if not metrics["coverage_complete"] or not retained.get("coverage_complete", False):
        raise ValueError("metric coverage is undefined for at least one planned subject/record/dimension")
    if retained.get("primary") is None or not math.isclose(float(retained["primary"]), float(metrics["primary"]), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("retained primary metric disagrees with retained predictions")
    if job["task"] == "refed_regression" and row["baselines"] is not None:
        _check_refed_baseline(row["baselines"], arrays, metrics["subjects"])
        row["refed_baseline_valid"] = True
    if job["task"] != "refed_regression":
        baselines = row["baselines"]
        baseline_rows = {} if not isinstance(baselines, dict) else {str(item["subject_id"]): item for item in baselines.get("subjects", [])}
        if (len(baseline_rows) != len(metrics["subjects"])
                or len(baselines.get("subjects", [])) != len(baseline_rows)
                or set(baseline_rows) != {item["subject_id"] for item in metrics["subjects"]}):
            raise ValueError("retained per-subject classification B0 coverage is missing or inconsistent")
        for subject in metrics["subjects"]:
            baseline = baseline_rows[subject["subject_id"]]
            if baseline.get("sample_count") != subject["sample_count"]:
                raise ValueError("B0 uses a different scored evaluation denominator")
            for field in ("majority_macro_f1", "prior_random_expected_macro_f1"):
                value = baseline.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"invalid retained classification B0 field: {field}")
    row["valid"] = True


def _bootstrap(values: list[float]) -> list[float]:
    vector = np.asarray(values, dtype=float)
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(vector), size=(BOOTSTRAP_REPLICATES, len(vector)))
    return np.quantile(vector[indices].mean(1), [0.025, 0.975]).tolist()


def _sd(values: list[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _cell(key: tuple[str, ...], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    result = dict(zip(KEYS, key))
    seeds = sorted({row["seed"] for row in jobs})
    folds = sorted({row["outer_fold"] for row in jobs})
    failures = [failure for row in jobs for failure in row["failures"]]
    pairs = [(row["outer_fold"], row["seed"]) for row in jobs]
    if len(set(pairs)) != len(pairs) or set(pairs) != {(fold, seed) for fold in folds for seed in seeds}:
        failures.append(_failure(jobs[0], "planned fold/seed grid is duplicated or incomplete", "invalid_plan"))
    expected_by_fold: dict[int, set[str]] = {}
    by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    baseline_by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    support_by_fold: dict[int, str] = {}
    for row in jobs:
        expected_subjects = set(row.get("expected_subjects", []))
        fold, seed = row["outer_fold"], row["seed"]
        if fold in expected_by_fold and expected_by_fold[fold] != expected_subjects:
            failures.append(_failure(row, "seeds have different planned evaluation subjects", "invalid_plan"))
        expected_by_fold[fold] = expected_subjects
        if row.get("support_signature"):
            if fold in support_by_fold and support_by_fold[fold] != row["support_signature"]:
                failures.append(_failure(row, "seeds have different evaluation targets or scoring support", "invalid_support"))
            support_by_fold[fold] = row["support_signature"]
        for subject in row.get("subjects", []):
            name = subject["subject_id"]
            if name in by_seed[seed]:
                failures.append(_failure(row, f"subject {name} appears in multiple outer folds for one seed", "invalid_support"))
            else:
                by_seed[seed][name] = subject
        baselines = row.get("baselines")
        if isinstance(baselines, dict) and (key[1] != "refed_regression" or row.get("refed_baseline_valid")):
            baseline_by_seed[seed].update({str(item["subject_id"]): item for item in baselines.get("subjects", [])})
    expected = set().union(*expected_by_fold.values())
    if sum(map(len, expected_by_fold.values())) != len(expected):
        failures.append(_failure(jobs[0], "planned outer evaluation subjects overlap between folds", "invalid_plan"))
    subjects = []
    for name in sorted(expected):
        scores = [by_seed[seed][name]["score"] for seed in seeds if name in by_seed[seed] and by_seed[seed][name]["score"] is not None]
        subject = dict(subject_id=name, seed_scores={str(seed): by_seed[seed].get(name, {}).get("score") for seed in seeds},
                       defined_seed_count=len(scores), score=float(np.mean(scores)) if scores else None)
        for error in ("mae", "rmse"):
            values = [by_seed[seed].get(name, {}).get(error) for seed in seeds]
            subject[error] = float(np.mean(values)) if values and all(value is not None for value in values) else None
        subjects.append(subject)
    if any(subject["defined_seed_count"] != len(seeds) for subject in subjects) or not subjects:
        failures.append(_failure(jobs[0], "incomplete planned subject-by-seed score coverage", "undefined_metric"))
    defined = [subject["score"] for subject in subjects if subject["score"] is not None]
    diagnostic = float(np.mean(defined)) if defined else None
    complete = not failures
    value = diagnostic if complete else None
    baseline: dict[str, Any] = {"value": 0.0, "reference": "no_positive_association",
                                "train_mean_errors": {"status": "missing", "mae": None, "rmse": None,
                                                      "reason": "not retained by evaluation worker"}}
    if key[1] == "refed_regression":
        baseline_subjects = []
        for name in sorted(expected):
            subject = {"subject_id": name}
            for error in ("mae", "rmse"):
                values = [baseline_by_seed[seed].get(name, {}).get(error) for seed in seeds]
                subject[error] = float(np.mean(values)) if values and all(value is not None for value in values) else None
            baseline_subjects.append(subject)
        available = bool(baseline_subjects) and all(subject[error] is not None for subject in baseline_subjects for error in ("mae", "rmse"))
        if available:
            baseline["train_mean_errors"] = {
                "status": "available", "mae": float(np.mean([subject["mae"] for subject in baseline_subjects])),
                "rmse": float(np.mean([subject["rmse"] for subject in baseline_subjects])),
                "subjects": baseline_subjects,
                "aggregation": "equal native coordinates and records within subject; then seeds within subject; then equal subjects",
                "fitting": "separate coordinate means from all valid outer-training target points only",
            }
    if key[1] != "refed_regression":
        components = {}
        for field in ("majority_macro_f1", "prior_random_expected_macro_f1"):
            values = [np.mean([baseline_by_seed[seed][name][field] for seed in seeds]) for name in sorted(expected)
                      if all(name in baseline_by_seed[seed] and field in baseline_by_seed[seed][name] for seed in seeds)]
            components[field] = float(np.mean(values)) if len(values) == len(expected) and values else None
        baseline = {"value": max(components.values()) if all(item is not None for item in components.values()) else None,
                    "components": components, "aggregation": "subject_mean_after_seed_mean; max_of_component_means"}
    b0 = baseline["value"]
    minimum = min(1.0, b0 + 0.02) if b0 is not None else None
    if value is None or b0 is None:
        decision = "INVALID_VALUE"
    else:
        decision = "TABLE_READY_WITH_NOTE" if value > b0 and value >= minimum else "REJECTED_VALUE"
    fold_values = []
    for fold in folds:
        names = expected_by_fold[fold]
        values = [subject["score"] for subject in subjects if subject["subject_id"] in names and subject["defined_seed_count"] == len(seeds)]
        if values and len(values) == len(names):
            fold_values.append(float(np.mean(values)))
    seed_values = [float(np.mean([by_seed[seed][name]["score"] for name in expected])) for seed in seeds
                   if expected and set(by_seed[seed]) == expected and all(by_seed[seed][name]["score"] is not None for name in expected)]
    result.update(value=value, decision=decision, ranking_eligible=decision == "TABLE_READY_WITH_NOTE",
                  table_value=value if decision == "TABLE_READY_WITH_NOTE" else None,
                  primary_name="subject_equal_mean_ccc" if key[1] == "refed_regression" else "subject_equal_macro_f1",
                  direction="higher_is_better", coverage_complete=complete,
                  planned_job_count=len(jobs), completed_job_count=sum(row.get("valid", False) for row in jobs),
                  planned_folds=folds, planned_seeds=seeds, subject_count=len(expected), subjects=subjects,
                  diagnostic_defined_subset_mean=diagnostic, diagnostic_defined_subject_count=len(defined),
                  diagnostic_note="defined subset only; unequal or missing seed support may remain; never substitutes for full value",
                  uncertainty={"type": "subject_block_percentile_bootstrap", "replicates": BOOTSTRAP_REPLICATES,
                               "seed": BOOTSTRAP_SEED, "confidence": 0.95,
                               "interval": _bootstrap(defined) if complete else None,
                               "conditioning": "fixed trained models, folds, selection and seeds; all records and seed scores stay inside subject block"},
                  companion={"fold_sd": _sd(fold_values) if complete else None, "fold_sd_count": len(fold_values),
                             "seed_sd": _sd(seed_values) if complete else None, "seed_sd_count": len(seed_values),
                             "definition": "sample SD of fold subject means after seed averaging; separate sample SD of seed subject means"},
                  B0=baseline, target={"minimum_admissible": minimum, "preferred_target": minimum,
                                      "rule": "min(legal_max=1,B0+0.02)", "H": None, "source_target": "not_applicable"},
                  relation_status="not_applicable; descriptive comparisons only; no project primary method specified",
                  note="No compatible H or source target: near-upper-bound performance has not been established.",
                  reporting_track="open_world_pretrained_with_target_corpus_overlap" if key[0].lower() == "reve" and key[1] in {"motor_imagery", "mental_arithmetic"} else "declared_pretraining_track",
                  failures=failures, _support_by_fold=support_by_fold)
    if key[1] == "refed_regression":
        result["companion"].update({error: float(np.mean([subject[error] for subject in subjects]))
                                   if complete and all(subject[error] is not None for subject in subjects) else None
                                   for error in ("mae", "rmse")})
        result["companion"]["error_aggregation"] = "native label coordinates: per-coordinate within-record MAE/RMSE, equal coordinates and records, then seeds within subject, then equal subjects"
    return result


def summarize(schedule_path: str | Path) -> dict[str, Any]:
    schedule_path = Path(schedule_path).resolve()
    schedule = _json(schedule_path)
    if schedule.get("schema") != "downstream_rebench_schedule_v1":
        raise ValueError("expected downstream_rebench_schedule_v1")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    failures, sources, inputs, unresolved_plan = [], {str(schedule_path)}, [], []
    cache: dict[Path, dict[str, Any]] = {}
    for entry in schedule["jobs"]:
        row = {"schedule_job_id": entry["id"], "status": "missing", "valid": False, "failures": []}
        try:
            job_path = _job_path(entry, schedule_path.parent)
            if job_path is None:
                if entry.get("kind") == "evaluate":
                    raise ValueError("planned evaluate job has no explicit --job contract")
                continue
            row["job_path"] = str(job_path)
            job = _json(job_path)
            if job.get("kind") != "evaluate":
                continue
            if job.get("schema") != "downstream_rebench_job_v1":
                raise ValueError("evaluate contract is not a versioned downstream rebenchmark job")
            sources.add(str(job_path))
            baseline_modality = str(job.get("method_profile", {}).get("baseline_modality", "eeg")) if job["method"] == "signal_linear" else "not_applicable"
            modality = str(job.get("modality_profile", "eeg_only" if job["method"].lower() in {"biot", "cbramod", "labram", "reve"} else "multimodal"))
            if job["method"] == "signal_linear" and baseline_modality == "eeg":
                modality = "eeg_only"
            key = (job["method"], job["task"], job.get("track", "lp"), modality, baseline_modality)
            output = _path(job["output_dir"], job_path.parent)
            row.update(job_id=job["job_id"], job_path=str(job_path), job_identity=_identity(job),
                       result_path=str(output / "result.json"), predictions_path=str(output / "predictions.npz"),
                       outer_fold=int(job["outer_fold"]), seed=int(job.get("seed", 42)))
            groups[key].append(row)
            inputs.append(row)
            if _path(entry["output_dir"], schedule_path.parent) != output:
                raise ValueError("scheduler and frozen worker contract disagree on output_dir")
            expected, classes = _expected(job, job_path.parent, cache, sources)
            row["expected_subjects"] = sorted({item[0] for item in expected.values()})
            row["expected_sample_count"] = len(expected)
            _load_result(row, job, expected, classes)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            issue = _failure(row, str(error))
            row["failures"].append(issue)
            if row not in inputs:
                failures.append(issue)
                if entry.get("kind") in {None, "evaluate"}:
                    unresolved_plan.append(issue)
                    inputs.append(row)
    cells = [_cell(key, jobs) for key, jobs in sorted(groups.items())]
    # Same-task/profile cells are paired on exact retained targets and masks.
    paired = defaultdict(list)
    for cell in cells:
        if cell["coverage_complete"]:
            paired[(cell["task"], cell["modality_profile"])].append(cell)
    for comparable in paired.values():
        if len({_identity(cell["_support_by_fold"]) for cell in comparable}) > 1:
            for affected in comparable:
                affected.update(value=None, table_value=None, ranking_eligible=False, coverage_complete=False, decision="INVALID_VALUE")
                affected["uncertainty"]["interval"] = None
                affected["failures"].append({"schedule_job_id": None, "job_id": None, "status": "invalid_support",
                                              "reason": "same-task/profile cells have different retained evaluation targets or masks"})
    for cell in cells:
        if unresolved_plan:
            cell.update(value=None, table_value=None, ranking_eligible=False, coverage_complete=False, decision="INVALID_VALUE")
            cell["uncertainty"]["interval"] = None
            cell["failures"].append({"schedule_job_id": None, "job_id": None, "status": "invalid_plan",
                                     "reason": "unreadable or unassigned planned evaluate contract prevents a complete denominator audit"})
        cell.pop("_support_by_fold")
        failures.extend(cell["failures"])
    source_inputs = [{name: row.get(name) for name in ("schedule_job_id", "job_id", "job_path", "job_identity", "result_path", "predictions_path", "status", "valid", "expected_sample_count")} for row in inputs]
    return _finite_json({"schema": SCHEMA, "run_id": schedule.get("run_id"), "schedule_path": str(schedule_path),
            "schedule_identity": _identity(schedule), "scope": "public_development_not_fresh_confirmation",
            "aggregation": "within-record scores -> subject per seed -> seed mean within subject -> equal subject mean; classification aggregates all subject events before macro-F1",
            "comparison_policy": "descriptive only; no automatic model-pair significance tests",
            "planned_evaluate_job_count": len(inputs), "valid_evaluate_job_count": sum(row["valid"] for row in inputs),
            "cells": cells, "unsupported_cells": schedule.get("unsupported_cells", []),
            "failures": failures, "source_inputs": source_inputs, "metadata_inputs": sorted(sources),
            "status": "complete" if not failures else "complete_with_failures"})


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_summary(summary: dict[str, Any], output: str | Path) -> None:
    output = Path(output).resolve()
    if (output / "summary.json").is_file():
        previous = _json(output / "summary.json")
        if previous.get("schema") != SCHEMA or previous.get("schedule_identity") != summary["schedule_identity"]:
            raise ValueError("output belongs to a different aggregate; use a fresh versioned directory")
    output.mkdir(parents=True, exist_ok=True)
    fields = [*KEYS, "reporting_track", "planned_job_count", "completed_job_count", "subject_count", "value", "ci_lower", "ci_upper", "B0", "minimum_admissible", "decision", "fold_sd", "seed_sd", "failure_count", "diagnostic_defined_subset_mean", "mae", "rmse", "train_mean_mae", "train_mean_rmse", "train_mean_error_status"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    lines = ["# DOWNSTREAM-REBENCH-v1 公开开发集汇总", "", "范围：已公开开发历史的数据；不是新的未查看确认集。", "",
             "主结果先按被试和种子计算，再对种子和被试等权平均。95% 区间为固定训练结果下的 2,000 次被试块 percentile bootstrap；fold SD 与 seed SD 分列。", "",
             "各模型比较仅作描述，不自动进行显著性检验。无相容 H/source target，达到 fallback 的最高状态为 TABLE_READY_WITH_NOTE，不能证明接近能力上界。", "",
             "| 方法 | 任务 | track / 输入 | 主值 | 95% CI | B0 | 状态 | 完成/计划 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    def number(value):
        return "—" if value is None else f"{value:.4f}"
    for cell in summary["cells"]:
        interval = cell["uncertainty"]["interval"]
        row = {name: cell.get(name) for name in fields}
        row.update(ci_lower=interval[0] if interval else None, ci_upper=interval[1] if interval else None,
                   B0=cell["B0"]["value"], minimum_admissible=cell["target"]["minimum_admissible"],
                   fold_sd=cell["companion"]["fold_sd"], seed_sd=cell["companion"]["seed_sd"], failure_count=len(cell["failures"]))
        mean_errors = cell["B0"].get("train_mean_errors", {})
        row.update(mae=cell["companion"].get("mae"), rmse=cell["companion"].get("rmse"),
                   train_mean_mae=mean_errors.get("mae"), train_mean_rmse=mean_errors.get("rmse"),
                   train_mean_error_status=mean_errors.get("status"))
        writer.writerow(row)
        method = cell["method"] + (f" ({cell['baseline_modality']})" if cell["method"] == "signal_linear" else "")
        if cell["reporting_track"] == "open_world_pretrained_with_target_corpus_overlap":
            method += " [已知目标语料重叠]"
        ci = "—" if interval is None else f"[{number(interval[0])}, {number(interval[1])}]"
        lines.append(f"| {method} | {cell['task']} | {cell['track']} / {cell['modality_profile']} | {number(cell['value'])} | {ci} | {number(cell['B0']['value'])} | {cell['decision']} | {cell['completed_job_count']}/{cell['planned_job_count']} |")
    refed_cells = [cell for cell in summary["cells"] if cell["task"] == "refed_regression"]
    if refed_cells:
        lines.extend(["", "## REFED 原生标签误差", "", "MAE/RMSE 越低越好。每条记录内按有效目标点计算两个坐标的误差，再依次对坐标、记录、种子和被试等权汇总；训练均值仅由对应外层训练目标拟合。", "",
                      "| 方法 / track / 输入 | MAE | 训练均值 MAE | RMSE | 训练均值 RMSE | 基线状态 |", "| --- | --- | --- | --- | --- | --- |"])
        for cell in refed_cells:
            errors = cell["B0"]["train_mean_errors"]
            label = f"{cell['method']} / {cell['track']} / {cell['modality_profile']} / {cell['baseline_modality']}"
            lines.append(f"| {label} | {number(cell['companion']['mae'])} | {number(errors['mae'])} | {number(cell['companion']['rmse'])} | {number(errors['rmse'])} | {errors['status']} |")
    lines.extend(["", "缺失/失败或未定义覆盖使整格主值为空；成功子集只保留为 JSON/CSV diagnostic，不参与主排名。REFED 的训练均值预测误差若未保留，明确标为 missing；不填零。", "",
                  f"计划：`{summary['schedule_path']}`；逐 job 输入路径、合同身份和完整失败原因见同目录 `summary.json`。", "", "## 失败记录", "", "| Job | 状态 | 原因 |", "| --- | --- | --- |"])
    for failure in summary["failures"]:
        reason = str(failure["reason"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {failure.get('schedule_job_id') or failure.get('job_id') or 'cell support'} | {failure['status']} | {reason} |")
    if not summary["failures"]:
        lines.append("| — | — | 无 |")
    lines.extend(["", "## 预先不支持的单元", "", "```json", json.dumps(summary["unsupported_cells"], ensure_ascii=False, indent=2), "```", ""])
    _atomic_text(output / "cells.csv", stream.getvalue())
    _atomic_text(output / "summary.md", "\n".join(lines))
    _atomic_text(output / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = summarize(args.schedule)
    write_summary(summary, args.output)
    print(json.dumps({"status": summary["status"], "cells": len(summary["cells"]), "failures": len(summary["failures"])}))
    return 0  # terminal failures are evidence to report, not a reason to skip delivery


if __name__ == "__main__":
    raise SystemExit(main())
