#!/usr/bin/env python3
"""Public-development EEG band-power baselines.

This module is intentionally independent of the foundation-model adapters.  It
loads the same method-neutral task inventory, reads only the public half of
the strict cross-subject folds, and fits a fixed shrinkage-LDA classifier on
Welch log-band-power features.  The result is a diagnostic baseline, not a
claim to reproduce CSP or Riemannian methods.  In particular, no protected
split or protected prediction is ever opened by this file.

The default invocation writes JSON/NPZ artifacts to
``comparative_methods/runs/performance_analysis/20260816_p0/classical_baselines``.
Use ``--task`` to run one task while developing; the intended P0 invocation is
``--task motor_imagery --task nback --task visual``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
EFRM_ROOT = REPO_ROOT / "comparative_methods" / "EFRM-PyTorch"
for _import_root in (REPO_ROOT, EFRM_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from src.data.clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT
from efrm_pytorch.tasks import EFRMUnifiedTaskDataset, TASK_SPECS  # noqa: E402


TASKS = ("motor_imagery", "nback", "visual")
TASK_ALIASES = {"mi": "motor_imagery", "motor_imagery": "motor_imagery"}
PANEL_BY_TASK: dict[str, tuple[str, ...]] = {
    "motor_imagery": (
        "F7", "AFF5h", "F3", "AFp1", "AFp2", "AFF6h", "F4", "F8",
        "AFF1h", "AFF2h", "Cz", "Pz", "FCC5h", "FCC3h", "CCP5h", "CCP3h",
    ),
    "nback": (
        "Fp1", "AFF5h", "AFz", "F1", "FC5", "FC1", "T7", "C3",
        "Cz", "CP5", "CP1", "P7", "P3", "Pz", "POz", "O1",
    ),
    "visual": (
        "Fp1", "Fp2", "AF3", "AF4", "F3", "F4", "FC3", "FC4",
        "C3", "C4", "PO3", "PO4", "O1", "O2", "F5", "F6",
    ),
}
BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)
DEFAULT_CACHE_ROOT = REPO_ROOT / DEFAULT_CLEAN_CACHE_ROOT
DEFAULT_SPLIT_ROOT = (
    EFRM_ROOT
    / "runs"
    / "formal"
    / "efrm_resource_bounded_dual_protocol_v1"
    / "protocol"
    / "split_registry"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "comparative_methods"
    / "runs"
    / "performance_analysis"
    / "20260816_p0"
    / "classical_baselines"
)
PUBLIC_SCHEMAS = {
    "efrm_target_public_fold_v1",
    "sta_net_split_registry_v2",
    "sta_net_subject_split_v1",
}


class PublicBoundaryError(RuntimeError):
    """Raised when a requested input crosses the protected-data boundary."""


def _safe_public_path(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PublicBoundaryError(f"{label} resolves inside a protected path: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def _json_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_manifest(path: Path, *, task: str, outer_fold: int) -> dict[str, Any]:
    path = _safe_public_path(path, label="public split manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"public split manifest is not an object: {path}")
    if manifest.get("schema") not in PUBLIC_SCHEMAS:
        raise ValueError(f"unsupported public split schema: {manifest.get('schema')!r}")
    if bool(manifest.get("protected_test_opened", manifest.get("reserved_test_opened", False))):
        raise PublicBoundaryError(f"public manifest reports an opened protected test: {path}")
    forbidden = {"test_indices", "protected_indices", "reserved_test_indices"}.intersection(manifest)
    if forbidden:
        raise PublicBoundaryError(f"public manifest exposes protected indices: {sorted(forbidden)}")
    if manifest.get("task") not in {None, task}:
        raise ValueError(f"split task mismatch for {task}: {manifest.get('task')!r}")
    if int(manifest.get("outer_fold", outer_fold)) != int(outer_fold):
        raise ValueError(f"split outer-fold mismatch for {task}: {manifest.get('outer_fold')!r}")
    if manifest.get("protocol") not in {None, "strict_cross_subject"}:
        raise ValueError(f"classical baseline requires strict cross-subject public folds: {path}")
    train = manifest.get("train_indices")
    validation = manifest.get("validation_indices")
    if not isinstance(train, list) or not isinstance(validation, list):
        raise ValueError(f"public split must provide train_indices and validation_indices: {path}")
    train = [int(value) for value in train]
    validation = [int(value) for value in validation]
    if not train or not validation or set(train).intersection(validation):
        raise ValueError(f"public split is empty or overlapping: {path}")
    manifest["train_indices"] = train
    manifest["validation_indices"] = validation
    manifest["_public_manifest_path"] = str(path)
    manifest["_public_manifest_sha256"] = _json_sha256(path)
    return manifest


def public_split_path(split_root: Path, task: str, outer_fold: int) -> Path:
    """Resolve the public strict-cross-subject manifest without globbing protected files."""
    if task not in TASKS:
        raise KeyError(f"unsupported diagnostic task: {task}")
    root = split_root.expanduser().resolve()
    if "protected" in {part.lower() for part in root.parts}:
        raise PublicBoundaryError(f"split root resolves inside a protected path: {root}")
    return root / task / "strict_cross_subject" / "public" / f"outer{int(outer_fold)}.json"


@dataclass(frozen=True)
class FeatureBatch:
    """Features and metadata extracted from one public split partition."""

    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    join_keys: np.ndarray
    sample_ids: np.ndarray
    dataset_indices: np.ndarray
    skipped: tuple[dict[str, Any], ...]


def _bandpower_features(eeg: np.ndarray, *, sample_rate_hz: float) -> np.ndarray:
    """Return channel-major log Welch band power for one EEG window."""
    array = np.asarray(eeg, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 8:
        raise ValueError(f"EEG window must be [channels,time], got {array.shape}")
    array = array - np.median(array, axis=1, keepdims=True)
    nperseg = min(int(round(2.0 * sample_rate_hz)), array.shape[1])
    frequencies, density = welch(
        array,
        fs=float(sample_rate_hz),
        nperseg=nperseg,
        noverlap=nperseg // 2,
        axis=-1,
        detrend="constant",
        scaling="density",
    )
    values: list[np.ndarray] = []
    for _, low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        if int(mask.sum()) < 2:
            raise ValueError(f"band {low:g}-{high:g} Hz is unresolved by Welch grid")
        power = trapezoid(density[..., mask], frequencies[mask], axis=-1)
        values.append(np.log(np.maximum(power, np.finfo(np.float64).tiny)))
    # [channel, band] -> [channel * band], preserving panel order.
    return np.stack(values, axis=-1).reshape(-1).astype(np.float32)


def extract_features(
    dataset: EFRMUnifiedTaskDataset,
    indices: Sequence[int],
    *,
    task: str,
    require_full_support: bool = True,
) -> FeatureBatch:
    """Extract fixed-panel EEG band power using only the requested inventory."""
    panel = PANEL_BY_TASK[task]
    duration_s = float(TASK_SPECS[task].input_duration_s)
    expected_samples = int(round(duration_s * 200.0))
    features: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[str] = []
    join_keys: list[str] = []
    sample_ids: list[str] = []
    kept_indices: list[int] = []
    skipped: list[dict[str, Any]] = []
    for public_index in indices:
        index = int(public_index)
        if index < 0 or index >= len(dataset):
            raise IndexError(f"public index out of range for {task}: {index}")
        row = dataset.lightweight_metadata(index)
        try:
            source = dataset.base[dataset.indices[index]]
            names = tuple(str(name) for name in source["channel_names"]["eeg"])
            lookup = {name: position for position, name in enumerate(names)}
            missing = [name for name in panel if name not in lookup]
            if missing:
                raise ValueError(f"missing panel channels: {missing}")
            rate = float(source["sample_rate_hz"]["eeg"])
            if not np.isclose(rate, 200.0):
                raise ValueError(f"expected EEG@200 Hz, received {rate:g} Hz")
            eeg = np.asarray(source["eeg"], dtype=np.float32)
            if eeg.ndim != 2 or eeg.shape[1] < expected_samples:
                raise ValueError(f"EEG support shorter than {expected_samples} samples: {eeg.shape}")
            valid = np.asarray(source["analysis_valid_mask"]["eeg"], dtype=bool).reshape(-1)
            if valid.size < expected_samples or (require_full_support and not valid[:expected_samples].all()):
                raise ValueError("incomplete recorded EEG support")
            selected = np.asarray([lookup[name] for name in panel], dtype=np.int64)
            bad = np.asarray(source["bad_channel_mask"]["eeg"], dtype=bool).reshape(-1)
            if bad.size != len(names) or bool(bad[selected].any()):
                raise ValueError("selected panel contains a bad measured channel")
            window = np.asarray(eeg[selected, :expected_samples], dtype=np.float32)
            if not np.isfinite(window).all():
                raise ValueError("non-finite EEG values")
            features.append(_bandpower_features(window, sample_rate_hz=rate))
            labels.append(int(row["class_index"]))
            subjects.append(str(row["subject"]))
            join_keys.append(str(row["join_key"]))
            sample_ids.append(
                f"{task}|{row['join_key']}|event={int(row['event_index'])}"
                f"|offset_ms={int(round(float(row['window_offset_s']) * 1000.0))}"
            )
            kept_indices.append(index)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            skipped.append({"dataset_index": index, "subject": str(row.get("subject", "")), "reason": str(exc)})
    if not features:
        raise RuntimeError(f"no valid public EEG windows survived feature extraction for {task}")
    return FeatureBatch(
        x=np.stack(features).astype(np.float32),
        y=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray(subjects, dtype=str),
        join_keys=np.asarray(join_keys, dtype=str),
        sample_ids=np.asarray(sample_ids, dtype=str),
        dataset_indices=np.asarray(kept_indices, dtype=np.int64),
        skipped=tuple(skipped),
    )


def _classifier(kind: str) -> Pipeline:
    if kind == "lda":
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif kind == "logistic":
        estimator = LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=2000, class_weight=None,
            random_state=42,
        )
    else:
        raise ValueError(f"unknown classifier {kind!r}; expected lda or logistic")
    return Pipeline([("standardize", StandardScaler()), ("classifier", estimator)])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for subject in sorted(set(subjects.tolist())):
        mask = subjects == subject
        values = _metrics(y_true[mask], y_pred[mask])
        output[str(subject)] = {"sample_count": int(mask.sum()), **values}
    return output


def _package_versions() -> dict[str, str]:
    names = ("numpy", "scipy", "scikit-learn", "torch")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def run_task(
    task: str,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    split_root: Path = DEFAULT_SPLIT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    classifier: str = "lda",
    outer_folds: Iterable[int] = range(5),
    require_full_support: bool = True,
) -> dict[str, Any]:
    """Run public strict-cross-subject folds for one task and persist artifacts."""
    if task not in TASKS:
        raise KeyError(f"task must be one of {TASKS}, got {task!r}")
    cache_root = cache_root.expanduser().resolve()
    split_root = split_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if "protected" in {part.lower() for part in split_root.parts}:
        raise PublicBoundaryError(f"split root resolves inside protected data: {split_root}")
    dataset = EFRMUnifiedTaskDataset(TASK_SPECS[task], cache_root=str(cache_root))
    task_root = output_root / task / classifier
    task_root.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []
    requested_folds = tuple(int(value) for value in outer_folds)
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_subject: list[np.ndarray] = []
    skipped_total = 0
    for outer_fold in requested_folds:
        manifest_path = public_split_path(split_root, task, outer_fold)
        manifest = _public_manifest(manifest_path, task=task, outer_fold=outer_fold)
        train = extract_features(dataset, manifest["train_indices"], task=task, require_full_support=require_full_support)
        validation = extract_features(dataset, manifest["validation_indices"], task=task, require_full_support=require_full_support)
        model = _classifier(classifier)
        model.fit(train.x, train.y)
        pred = np.asarray(model.predict(validation.x), dtype=np.int64)
        metric = _metrics(validation.y, pred)
        group = _group_metrics(validation.y, pred, validation.subjects)
        fold_row = {
            "task": task,
            "outer_fold": outer_fold,
            "protocol": "strict_cross_subject_public_development",
            "claim_scope": "public_development_only_not_table_admissible",
            "classifier": classifier,
            "feature_schema": "welch_log_bandpower_v1",
            "bands_hz": [{"name": name, "low": low, "high": high} for name, low, high in BANDS],
            "panel": list(PANEL_BY_TASK[task]),
            "sample_rate_hz": 200.0,
            "window_duration_s": float(TASK_SPECS[task].input_duration_s),
            "full_support_required": bool(require_full_support),
            "split_manifest": str(manifest_path.resolve()),
            "split_manifest_sha256": str(manifest["_public_manifest_sha256"]),
            "split_sha256": manifest.get("split_sha256"),
            "train_sample_count": int(train.x.shape[0]),
            "validation_sample_count": int(validation.x.shape[0]),
            "train_subject_count": int(len(set(train.subjects.tolist()))),
            "validation_subject_count": int(len(set(validation.subjects.tolist()))),
            "train_subjects": sorted(set(train.subjects.tolist())),
            "validation_subjects": sorted(set(validation.subjects.tolist())),
            "train_skipped_count": len(train.skipped),
            "validation_skipped_count": len(validation.skipped),
            "metrics": metric,
            "validation_group_metrics": group,
        }
        fold_json = task_root / f"outer{outer_fold}.json"
        fold_json.write_text(json.dumps(fold_row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        np.savez_compressed(
            task_root / f"outer{outer_fold}_predictions.npz",
            y_true=validation.y,
            y_pred=pred,
            subject=validation.subjects,
            join_key=validation.join_keys,
            sample_id=validation.sample_ids,
            dataset_index=validation.dataset_indices,
        )
        fold_rows.append(fold_row)
        all_true.append(validation.y)
        all_pred.append(pred)
        all_subject.append(validation.subjects)
        skipped_total += len(train.skipped) + len(validation.skipped)

    # A single-fold continuation must not erase already-completed public
    # folds.  Keep prior fold metadata only when that fold was not rerun in
    # this invocation; rerun folds always replace their old artifact.
    existing_summary_path = task_root / "summary.json"
    previous_rows: dict[int, dict[str, Any]] = {}
    if existing_summary_path.is_file():
        try:
            previous = json.loads(existing_summary_path.read_text(encoding="utf-8"))
            if previous.get("task") == task and previous.get("protected_test_opened") is False:
                previous_rows = {
                    int(row["outer_fold"]): row
                    for row in previous.get("folds", ())
                    if int(row["outer_fold"]) not in set(requested_folds)
                }
        except (OSError, ValueError, TypeError, KeyError):
            previous_rows = {}
    if previous_rows:
        fold_rows = [*previous_rows.values(), *fold_rows]
        fold_rows.sort(key=lambda row: int(row["outer_fold"]))

    metric_names = ("accuracy", "balanced_accuracy", "macro_f1")
    summary_metrics = {
        name: {
            "mean": float(np.mean([row["metrics"][name] for row in fold_rows])),
            "std": float(np.std([row["metrics"][name] for row in fold_rows], ddof=1)) if len(fold_rows) > 1 else 0.0,
            "fold_values": [float(row["metrics"][name]) for row in fold_rows],
        }
        for name in metric_names
    }
    # Re-read every retained fold so a continuation invocation produces a
    # complete pooled artifact instead of silently reporting only its latest
    # fold.  This pooled value is secondary/descriptive; the report builder
    # uses fold-level subject metrics as the primary endpoint.
    pooled_true_parts: list[np.ndarray] = []
    pooled_pred_parts: list[np.ndarray] = []
    pooled_subject_parts: list[np.ndarray] = []
    for row in fold_rows:
        fold_path = task_root / f"outer{int(row['outer_fold'])}_predictions.npz"
        if not fold_path.is_file():
            continue
        with np.load(fold_path, allow_pickle=False) as payload:
            pooled_true_parts.append(np.asarray(payload["y_true"], dtype=np.int64))
            pooled_pred_parts.append(np.asarray(payload["y_pred"], dtype=np.int64))
            pooled_subject_parts.append(np.asarray(payload["subject"], dtype=str))
    pooled_true = np.concatenate(pooled_true_parts) if pooled_true_parts else np.empty(0, dtype=np.int64)
    pooled_pred = np.concatenate(pooled_pred_parts) if pooled_pred_parts else np.empty(0, dtype=np.int64)
    pooled_subject = np.concatenate(pooled_subject_parts) if pooled_subject_parts else np.empty(0, dtype=str)
    summary = {
        "schema": "classical_eeg_bandpower_public_analysis_v1",
        "status": "completed",
        "task": task,
        "protocol": "strict_cross_subject_public_development",
        "claim_scope": "public_development_only_not_table_admissible",
        "protected_test_opened": False,
        "exploratory": True,
        "classifier": classifier,
        "feature_schema": "welch_log_bandpower_v1",
        "bands_hz": [{"name": name, "low": low, "high": high} for name, low, high in BANDS],
        "panel": list(PANEL_BY_TASK[task]),
        "sample_rate_hz": 200.0,
        "window_duration_s": float(TASK_SPECS[task].input_duration_s),
        "full_support_required": bool(require_full_support),
        "cache_root": str(cache_root),
        "split_root": str(split_root),
        "dataset_metadata_sha256": dataset.metadata_fingerprint(),
        "fold_count": len(fold_rows),
        "folds": fold_rows,
        "metrics": summary_metrics,
        "pooled_public_validation": {
            "sample_count": int(pooled_true.size),
            "subject_count": int(len(set(pooled_subject.tolist()))),
            "metrics": _metrics(pooled_true, pooled_pred),
            "group_metrics": _group_metrics(pooled_true, pooled_pred, pooled_subject),
        },
        "skipped_window_count": int(skipped_total),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
    }
    (task_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        task_root / "pooled_public_validation_predictions.npz",
        y_true=pooled_true,
        y_pred=pooled_pred,
        subject=pooled_subject,
    )
    return summary


def _parse_tasks(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return TASKS
    output: list[str] = []
    for value in values:
        canonical = TASK_ALIASES.get(value.lower(), value.lower())
        if canonical not in TASKS:
            raise ValueError(f"unsupported task {value!r}; choose from {TASKS}")
        if canonical not in output:
            output.append(canonical)
    return tuple(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", help="task to run (repeatable; default: all P0 tasks)")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--classifier", choices=("lda", "logistic"), default="lda")
    parser.add_argument("--outer-fold", action="append", type=int, help="public outer fold (repeatable; default: 0..4)")
    parser.add_argument("--allow-incomplete-support", action="store_true", help="diagnostic override; default rejects incomplete windows")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = _parse_tasks(args.task)
    folds = tuple(args.outer_fold) if args.outer_fold else tuple(range(5))
    for task in tasks:
        summary = run_task(
            task,
            cache_root=args.cache_root,
            split_root=args.split_root,
            output_root=args.output_root,
            classifier=args.classifier,
            outer_folds=folds,
            require_full_support=not bool(args.allow_incomplete_support),
        )
        metrics = summary["metrics"]
        print(
            f"[{task}] public folds={summary['fold_count']} samples="
            f"{summary['pooled_public_validation']['sample_count']} "
            f"macro_f1={metrics['macro_f1']['mean']:.4f}±{metrics['macro_f1']['std']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
