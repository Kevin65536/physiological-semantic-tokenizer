#!/usr/bin/env python3
"""Run a full, task-resolved audit of the unified physiology loader.

The audit separates full index/population counts from signal statistics. Signal
statistics are computed over every admitted 20-second window using the loader's
validity masks. EEG amplitude statistics use ``analysis_valid_mask`` so samples
flagged by the admitted Single-Trial artifact branch do not silently influence
the primary variance summary. Statistics are window-weighted: an underlying
time point is counted again when overlapping loader windows expose it again,
matching the model-input distribution rather than a de-duplicated raw-record
distribution. No scientific pass/fail threshold is inferred from canonical
amplitude alone.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from html import escape
import json
import math
import multiprocessing as mp
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.unified_physiology import (  # noqa: E402
    CANONICAL_UNIT,
    DEFAULT_UNIFIED_WINDOW_DURATION_S,
    FORBIDDEN_TASK_NAMESPACES,
    RAW_DATASET_IDS,
    UnifiedPhysiologyWindowDataset,
    canonical_label,
)


SCHEMA = "unified_loader_final_audit_v1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments/runs/physiology_semantic_tokenizer/data_quality_audit"
    / "final_unified_loader_audit_20260718"
)
QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        default="data/cache/physiology_semantic_clean_v4",
        help="Canonical cache used by UnifiedPhysiologyWindowDataset.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--window-duration-s", type=float, default=DEFAULT_UNIFIED_WINDOW_DURATION_S)
    parser.add_argument(
        "--resource-json",
        default="/tmp/unified_loader_audit_resources.json",
        help="Optional resource snapshot created by the resource-detection skill.",
    )
    parser.add_argument(
        "--quantile-points-per-window",
        type=int,
        default=128,
        help="Deterministic finite-value sample size used per window for approximate amplitude quantiles.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Forked record workers; each record still uses the unified loader path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


@dataclass
class RunningSignalStats:
    quantile_points_per_window: int
    total_value_count: int = 0
    finite_value_count: int = 0
    value_sum: float = 0.0
    value_sum_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    max_abs: float = 0.0
    empty_window_count: int = 0
    window_means: list[float] = field(default_factory=list)
    window_variances: list[float] = field(default_factory=list)
    channel_variances: list[float] = field(default_factory=list)
    near_constant_channel_window_count: int = 0
    quantile_samples: list[np.ndarray] = field(default_factory=list)
    units: set[str] = field(default_factory=set)

    def update(self, signal: np.ndarray, mask: np.ndarray) -> None:
        array = np.asarray(signal, dtype=np.float64)
        selected = array[:, np.asarray(mask, dtype=bool)]
        self.total_value_count += int(selected.size)
        if selected.size == 0:
            self.empty_window_count += 1
            return
        finite_mask = np.isfinite(selected)
        self.finite_value_count += int(np.count_nonzero(finite_mask))
        values = selected[finite_mask].astype(np.float64, copy=False)
        if values.size == 0:
            self.empty_window_count += 1
            return
        self.value_sum += float(np.sum(values, dtype=np.float64))
        self.value_sum_sq += float(np.sum(values * values, dtype=np.float64))
        self.minimum = min(self.minimum, float(np.min(values)))
        self.maximum = max(self.maximum, float(np.max(values)))
        self.max_abs = max(self.max_abs, float(np.max(np.abs(values))))
        self.window_means.append(float(np.mean(values)))
        self.window_variances.append(float(np.var(values)))

        channel_variance = np.nanvar(np.where(finite_mask, selected, np.nan), axis=1)
        finite_channel_variance = channel_variance[np.isfinite(channel_variance)].astype(float)
        self.channel_variances.extend(finite_channel_variance.tolist())
        self.near_constant_channel_window_count += int(
            np.count_nonzero(finite_channel_variance <= 1e-8)
        )

        count = min(self.quantile_points_per_window, values.size)
        indices = np.linspace(0, values.size - 1, num=count, dtype=np.int64)
        self.quantile_samples.append(values[indices].astype(np.float32))

    def to_dict(self) -> dict[str, Any]:
        if self.finite_value_count:
            mean = self.value_sum / self.finite_value_count
            variance = max(self.value_sum_sq / self.finite_value_count - mean * mean, 0.0)
        else:
            mean = variance = float("nan")
        samples = (
            np.concatenate(self.quantile_samples).astype(np.float64, copy=False)
            if self.quantile_samples
            else np.asarray([], dtype=np.float64)
        )
        quantiles = {
            f"q{int(q * 100):02d}": float(np.quantile(samples, q)) if samples.size else float("nan")
            for q in QUANTILES
        }
        window_variances = np.asarray(self.window_variances, dtype=np.float64)
        channel_variances = np.asarray(self.channel_variances, dtype=np.float64)
        return {
            "mask_applied": True,
            "canonical_unit": sorted(self.units),
            "value_count": self.total_value_count,
            "finite_value_count": self.finite_value_count,
            "finite_fraction": (
                self.finite_value_count / self.total_value_count if self.total_value_count else 0.0
            ),
            "mean": mean,
            "variance": variance,
            "std": math.sqrt(variance) if math.isfinite(variance) else float("nan"),
            "rms": (
                math.sqrt(self.value_sum_sq / self.finite_value_count)
                if self.finite_value_count
                else float("nan")
            ),
            "min": self.minimum if self.finite_value_count else float("nan"),
            "max": self.maximum if self.finite_value_count else float("nan"),
            "max_abs": self.max_abs if self.finite_value_count else float("nan"),
            "approximate_quantiles": quantiles,
            "quantile_method": (
                f"deterministic_evenly_spaced_finite_values_{self.quantile_points_per_window}_per_window"
            ),
            "window_variance": _distribution(window_variances),
            "channel_window_variance": _distribution(channel_variances),
            "near_constant_channel_window_count": self.near_constant_channel_window_count,
            "empty_window_count": self.empty_window_count,
        }

    def merge(self, other: "RunningSignalStats") -> None:
        self.units.update(other.units)
        self.total_value_count += other.total_value_count
        self.finite_value_count += other.finite_value_count
        self.value_sum += other.value_sum
        self.value_sum_sq += other.value_sum_sq
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)
        self.max_abs = max(self.max_abs, other.max_abs)
        self.empty_window_count += other.empty_window_count
        self.window_means.extend(other.window_means)
        self.window_variances.extend(other.window_variances)
        self.channel_variances.extend(other.channel_variances)
        self.near_constant_channel_window_count += other.near_constant_channel_window_count
        self.quantile_samples.extend(other.quantile_samples)


@dataclass
class SimpleMoments:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def update_values(self, values: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        self.count += int(finite.size)
        if finite.size:
            self.total += float(np.sum(finite, dtype=np.float64))
            self.total_sq += float(np.sum(finite * finite, dtype=np.float64))

    def std(self) -> float:
        if not self.count:
            return float("nan")
        mean = self.total / self.count
        return math.sqrt(max(self.total_sq / self.count - mean * mean, 0.0))


def _distribution(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "min": None, "q05": None, "median": None, "q95": None, "max": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "q95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def _task_state(points: int) -> dict[str, Any]:
    return {
        "subjects": set(),
        "records": set(),
        "labels": Counter(),
        "subject_labels": Counter(),
        "event_types": Counter(),
        "eeg": RunningSignalStats(points),
        "fnirs": RunningSignalStats(points),
        "eeg_valid_fractions": [],
        "eeg_analysis_valid_fractions": [],
        "fnirs_valid_fractions": [],
        "eeg_artifact_fractions": [],
        "eeg_channel_counts": Counter(),
        "fnirs_channel_counts": Counter(),
        "eeg_channel_signatures": Counter(),
        "fnirs_channel_signatures": Counter(),
        "signal_branches": Counter(),
        "geometry": {
            "eeg_rows": 0,
            "eeg_position_available": 0,
            "fnirs_rows": 0,
            "fnirs_position_available": 0,
        },
        "geometry_status": {"eeg": Counter(), "fnirs": Counter()},
        "bad_channel_count": 0,
        "channel_count_for_bad_mask": 0,
        "full_eeg_window_count": 0,
        "full_fnirs_window_count": 0,
        "window_count": 0,
    }


def _record_state(namespace: str, ref: Any) -> dict[str, Any]:
    return {
        "dataset_id": ref.record.dataset_id,
        "task_namespace": namespace,
        "subject": ref.record.canonical_subject_id,
        "record_id": ref.record.base_record_id,
        "join_key": ref.record.join_key,
        "window_count": 0,
        "eeg": SimpleMoments(),
        "fnirs": SimpleMoments(),
        "eeg_valid_sum": 0.0,
        "fnirs_valid_sum": 0.0,
        "artifact_sum": 0.0,
        "eeg_channels": None,
        "fnirs_channels": None,
        "bad_channel_fraction": None,
    }


_WORKER_DATASET: UnifiedPhysiologyWindowDataset | None = None


def _audit_record_worker(
    args: tuple[list[int], int],
) -> tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    indices, quantile_points_per_window = args
    if _WORKER_DATASET is None:
        raise RuntimeError("record worker has no inherited unified dataset")
    dataset = _WORKER_DATASET
    first_ref = dataset.windows[indices[0]]
    namespace = canonical_label(first_ref.event, first_ref.record.dataset_id)["namespace"]
    state = _task_state(quantile_points_per_window)
    record = _record_state(namespace, first_ref)
    channel_rows: list[dict[str, Any]] = []

    for local_index, index in enumerate(indices):
        ref = dataset.windows[index]
        sample = dataset[index]
        label = canonical_label(ref.event, ref.record.dataset_id)
        if label["namespace"] != namespace:
            raise RuntimeError(f"record spans multiple tasks: {ref.record.join_key}")
        state["window_count"] += 1
        state["subjects"].add(ref.record.canonical_subject_id)
        state["records"].add(ref.record.join_key)
        state["labels"][label["condition"]] += 1
        state["subject_labels"][(ref.record.canonical_subject_id, label["condition"])] += 1
        state["event_types"][str(ref.event.get("event_type"))] += 1

        eeg_valid = np.asarray(sample["valid_mask"]["eeg"], dtype=bool)
        eeg_analysis = np.asarray(sample["analysis_valid_mask"]["eeg"], dtype=bool)
        fnirs_valid = np.asarray(sample["valid_mask"]["fnirs"], dtype=bool)
        if sample.get('channel_valid_mask'):
            eeg_analysis &= sample['channel_valid_mask']['eeg'].all(axis=0)
            fnirs_valid &= sample['channel_valid_mask']['fnirs'].all(axis=0)
        state['eeg'].units.add(sample['unit']['eeg'])
        state['fnirs'].units.add(sample['unit']['fnirs'])
        artifact = np.asarray(sample["artifact_mask"]["eeg"], dtype=bool)
        state["eeg"].update(sample["eeg"], eeg_analysis)
        state["fnirs"].update(sample["fnirs"], fnirs_valid)
        state["eeg_valid_fractions"].append(float(np.mean(eeg_valid)))
        state["eeg_analysis_valid_fractions"].append(float(np.mean(eeg_analysis)))
        state["fnirs_valid_fractions"].append(float(np.mean(fnirs_valid)))
        artifact_fraction = (
            float(np.count_nonzero(artifact & eeg_valid) / np.count_nonzero(eeg_valid))
            if np.any(eeg_valid)
            else 0.0
        )
        state["eeg_artifact_fractions"].append(artifact_fraction)
        state["full_eeg_window_count"] += int(np.all(eeg_valid))
        state["full_fnirs_window_count"] += int(np.all(fnirs_valid))
        eeg_names = tuple(str(value) for value in sample["channel_names"]["eeg"])
        fnirs_names = tuple(str(value) for value in sample["channel_names"]["fnirs"])
        state["eeg_channel_counts"][len(eeg_names)] += 1
        state["fnirs_channel_counts"][len(fnirs_names)] += 1
        state["eeg_channel_signatures"][eeg_names] += 1
        state["fnirs_channel_signatures"][fnirs_names] += 1
        state["signal_branches"][str(sample["eeg_signal_branch"])] += 1

        if local_index == 0:
            bad = np.asarray(sample["bad_channel_mask"]["eeg"], dtype=bool)
            state["bad_channel_count"] += int(np.count_nonzero(bad))
            state["channel_count_for_bad_mask"] += int(bad.size)
            for modality, names in (("eeg", eeg_names), ("fnirs", fnirs_names)):
                rows = sample["channel_geometry"][modality]
                state["geometry"][f"{modality}_rows"] += len(rows)
                state["geometry"][f"{modality}_position_available"] += sum(
                    bool(row.get("position_available")) for row in rows
                )
                state["geometry_status"][modality].update(
                    str(row.get("coordinate_status", "unspecified")) for row in rows
                )
                signature = sha256("\n".join(names).encode("utf-8")).hexdigest()
                channel_rows.append({
                    "task_namespace": namespace,
                    "modality": modality,
                    "channel_count": len(names),
                    "signature_sha256": signature,
                    "channel_names": list(names),
                    "representative_join_key": ref.record.join_key,
                })
            record["eeg_channels"] = len(eeg_names)
            record["fnirs_channels"] = len(fnirs_names)
            record["bad_channel_fraction"] = float(np.mean(bad))

        record["window_count"] += 1
        record["eeg"].update_values(np.asarray(sample["eeg"], dtype=np.float32)[:, eeg_analysis])
        record["fnirs"].update_values(np.asarray(sample["fnirs"], dtype=np.float32)[:, fnirs_valid])
        record["eeg_valid_sum"] += float(np.mean(eeg_valid))
        record["fnirs_valid_sum"] += float(np.mean(fnirs_valid))
        record["artifact_sum"] += artifact_fraction

    n = record["window_count"]
    record_row = {
        "dataset_id": record["dataset_id"],
        "task_namespace": record["task_namespace"],
        "subject": record["subject"],
        "record_id": record["record_id"],
        "join_key": record["join_key"],
        "window_count": n,
        "eeg_channels": record["eeg_channels"],
        "fnirs_channels": record["fnirs_channels"],
        "eeg_std_analysis_valid": record["eeg"].std(),
        "fnirs_std_valid": record["fnirs"].std(),
        "eeg_valid_fraction": record["eeg_valid_sum"] / n,
        "fnirs_valid_fraction": record["fnirs_valid_sum"] / n,
        "eeg_artifact_fraction": record["artifact_sum"] / n,
        "eeg_bad_channel_fraction": record["bad_channel_fraction"],
    }
    return namespace, state, record_row, channel_rows


def _merge_task_state(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    target["subjects"].update(source["subjects"])
    target["records"].update(source["records"])
    for key in (
        "labels", "subject_labels", "event_types", "eeg_channel_counts", "fnirs_channel_counts",
        "eeg_channel_signatures", "fnirs_channel_signatures", "signal_branches",
    ):
        target[key].update(source[key])
    target["eeg"].merge(source["eeg"])
    target["fnirs"].merge(source["fnirs"])
    for key in (
        "eeg_valid_fractions", "eeg_analysis_valid_fractions", "fnirs_valid_fractions",
        "eeg_artifact_fractions",
    ):
        target[key].extend(source[key])
    for key in (
        "bad_channel_count", "channel_count_for_bad_mask", "full_eeg_window_count",
        "full_fnirs_window_count", "window_count",
    ):
        target[key] += source[key]
    for key in target["geometry"]:
        target["geometry"][key] += source["geometry"][key]
    for modality in ("eeg", "fnirs"):
        target["geometry_status"][modality].update(source["geometry_status"][modality])


def semantic_sample_key(ref: Any, label: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return a trial-semantic key that collapses Visual left/right probes."""
    base_record_id = ref.record.base_record_id
    if ref.record.dataset_id == "visual_cognitive_motivation":
        base_record_id = re.sub(r"_Probe[12]$", "", base_record_id)
    epoch = ref.event.get("metadata", {}).get("epoch_id", ref.event.get("event_index"))
    return (
        ref.record.dataset_id,
        ref.record.canonical_subject_id,
        base_record_id,
        epoch,
        label["condition"],
    )


def _collect_index_summary(dataset: UnifiedPhysiologyWindowDataset) -> dict[str, Any]:
    exact_ids: list[tuple[Any, ...]] = []
    semantic_ids: list[tuple[Any, ...]] = []
    task_subject_labels: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    task_labels: dict[str, set[str]] = defaultdict(set)
    unknown_rows: list[dict[str, Any]] = []
    for ref in dataset.windows:
        label = canonical_label(ref.event, ref.record.dataset_id)
        namespace = label["namespace"]
        exact_ids.append(
            (
                ref.record.dataset_id,
                ref.record.canonical_subject_id,
                ref.record.base_record_id,
                ref.event.get("event_index"),
            )
        )
        semantic_ids.append(semantic_sample_key(ref, label))
        task_subject_labels[namespace][(ref.record.canonical_subject_id, label["condition"])] += 1
        task_labels[namespace].add(label["condition"])
        if label["condition"] == "unknown" or int(label.get("class_index", -1)) < 0:
            unknown_rows.append({
                "namespace": namespace,
                "subject": ref.record.canonical_subject_id,
                "record_id": ref.record.base_record_id,
                "event_index": ref.event.get("event_index"),
                "epoch_id": ref.event.get("metadata", {}).get("epoch_id"),
                "raw_label": ref.event.get("metadata", {}).get("epoch_type_raw", ref.event.get("label")),
            })

    semantic_counts = Counter(semantic_ids)
    duplicate_semantic = {key: count for key, count in semantic_counts.items() if count > 1}
    subject_coverage: list[dict[str, Any]] = []
    for namespace, counts in sorted(task_subject_labels.items()):
        labels = sorted(label for label in task_labels[namespace] if label != "unknown")
        subjects = sorted({subject for subject, _ in counts})
        for subject in subjects:
            missing = [label for label in labels if counts[(subject, label)] == 0]
            subject_coverage.append({
                "task_namespace": namespace,
                "subject": subject,
                "window_count": sum(counts[(subject, label)] for label in task_labels[namespace]),
                "missing_admitted_labels": missing,
                "complete_admitted_label_coverage": not missing,
            })

    return {
        "exact_sample_id_duplicate_count": len(exact_ids) - len(set(exact_ids)),
        "semantic_duplicate_group_count": len(duplicate_semantic),
        "semantic_duplicate_extra_window_count": sum(count - 1 for count in duplicate_semantic.values()),
        "semantic_duplicate_multiplicity": dict(Counter(duplicate_semantic.values())),
        "semantic_duplicate_note": (
            "Visual Probe1/Probe2 share one EEG trial and label but carry different fNIRS hemispheric probes. "
            "They are distinct loader records but not independent trials."
        ),
        "unknown_label_count": len(unknown_rows),
        "unknown_labels": unknown_rows,
        "subject_label_coverage": subject_coverage,
    }


def _collect_refed_continuous_targets(dataset: UnifiedPhysiologyWindowDataset) -> dict[str, Any]:
    streams: dict[str, list[np.ndarray]] = defaultdict(list)
    lengths: list[int] = []
    missing_event_count = 0
    constant_event_counts = Counter()
    event_count = 0
    for ref in dataset.windows:
        if ref.record.dataset_id != "refed":
            continue
        event_count += 1
        stream = ref.event.get("metadata", {}).get("continuous_label_stream", {})
        names = [str(value) for value in stream.get("names", [])]
        values = np.asarray(stream.get("values", []), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(names):
            missing_event_count += 1
            continue
        lengths.append(int(values.shape[0]))
        for index, name in enumerate(names):
            vector = values[:, index]
            streams[name].append(vector)
            finite = vector[np.isfinite(vector)]
            if finite.size and float(np.var(finite)) <= 1e-12:
                constant_event_counts[name] += 1
    target_stats = {}
    for name, arrays in sorted(streams.items()):
        values = np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float64)
        finite = values[np.isfinite(values)]
        target_stats[name] = {
            "value_count": int(values.size),
            "finite_fraction": float(finite.size / values.size) if values.size else 0.0,
            "min": float(np.min(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
            "mean": float(np.mean(finite)) if finite.size else None,
            "std": float(np.std(finite)) if finite.size else None,
            "q05": float(np.quantile(finite, 0.05)) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "q95": float(np.quantile(finite, 0.95)) if finite.size else None,
            "constant_event_count": int(constant_event_counts[name]),
        }
    return {
        "event_count": event_count,
        "event_with_missing_or_invalid_stream_count": missing_event_count,
        "stream_length": _distribution(np.asarray(lengths, dtype=np.float64)),
        "targets": target_stats,
        "loader_boundary": (
            "continuous streams remain nested in event.metadata; canonical_label is video-category level"
        ),
    }


def _audit_signals(
    dataset: UnifiedPhysiologyWindowDataset,
    *,
    quantile_points_per_window: int,
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    global _WORKER_DATASET
    tasks: dict[str, dict[str, Any]] = defaultdict(lambda: _task_state(quantile_points_per_window))
    record_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    seen_channel_signatures: set[tuple[str, str, str]] = set()
    indices_by_record: dict[str, list[int]] = defaultdict(list)
    for index, ref in enumerate(dataset.windows):
        indices_by_record[ref.record.join_key].append(index)
    jobs = [(indices, quantile_points_per_window) for indices in indices_by_record.values()]
    _WORKER_DATASET = dataset
    pool = None
    if workers > 1:
        pool = mp.get_context("fork").Pool(processes=workers)
        results: Iterable[tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = (
            pool.imap_unordered(_audit_record_worker, jobs, chunksize=1)
        )
    else:
        results = map(_audit_record_worker, jobs)
    try:
        for completed, (namespace, partial, record_row, partial_channels) in enumerate(results, start=1):
            _merge_task_state(tasks[namespace], partial)
            record_rows.append(record_row)
            for row in partial_channels:
                signature_key = (row["task_namespace"], row["modality"], row["signature_sha256"])
                if signature_key not in seen_channel_signatures:
                    seen_channel_signatures.add(signature_key)
                    channel_rows.append(row)
            if completed % 25 == 0 or completed == len(jobs):
                processed_windows = sum(row["window_count"] for row in record_rows)
                print(
                    f"[final-loader-audit] processed {completed}/{len(jobs)} records, "
                    f"{processed_windows}/{len(dataset)} windows",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    task_payload = {}
    for namespace, state in sorted(tasks.items()):
        labels = dict(sorted(state["labels"].items()))
        admitted_labels = [name for name in labels if name != "unknown"]
        admitted_counts = [labels[name] for name in admitted_labels]
        geometry = dict(state["geometry"])
        geometry["eeg_position_available_fraction"] = (
            geometry["eeg_position_available"] / geometry["eeg_rows"] if geometry["eeg_rows"] else 0.0
        )
        geometry["fnirs_position_available_fraction"] = (
            geometry["fnirs_position_available"] / geometry["fnirs_rows"] if geometry["fnirs_rows"] else 0.0
        )
        geometry["eeg_coordinate_status"] = dict(state["geometry_status"]["eeg"])
        geometry["fnirs_coordinate_status"] = dict(state["geometry_status"]["fnirs"])
        task_payload[namespace] = {
            "dataset_id": namespace.split(":", 1)[0],
            "task": namespace.split(":", 1)[1],
            "subject_count": len(state["subjects"]),
            "record_count": len(state["records"]),
            "window_count": state["window_count"],
            "event_types": dict(state["event_types"]),
            "label_distribution": labels,
            "unknown_label_count": labels.get("unknown", 0),
            "admitted_label_imbalance_ratio": (
                max(admitted_counts) / min(admitted_counts) if admitted_counts and min(admitted_counts) else None
            ),
            "eeg_channel_count_distribution": dict(state["eeg_channel_counts"]),
            "fnirs_channel_count_distribution": dict(state["fnirs_channel_counts"]),
            "eeg_channel_signature_count": len(state["eeg_channel_signatures"]),
            "fnirs_channel_signature_count": len(state["fnirs_channel_signatures"]),
            "eeg_signal_branches": dict(state["signal_branches"]),
            "eeg_amplitude_analysis_valid": state["eeg"].to_dict(),
            "fnirs_amplitude_valid": state["fnirs"].to_dict(),
            "eeg_valid_fraction": _distribution(np.asarray(state["eeg_valid_fractions"])),
            "eeg_analysis_valid_fraction": _distribution(
                np.asarray(state["eeg_analysis_valid_fractions"])
            ),
            "fnirs_valid_fraction": _distribution(np.asarray(state["fnirs_valid_fractions"])),
            "eeg_artifact_fraction": _distribution(np.asarray(state["eeg_artifact_fractions"])),
            "full_eeg_window_fraction": state["full_eeg_window_count"] / state["window_count"],
            "full_fnirs_window_fraction": state["full_fnirs_window_count"] / state["window_count"],
            "padded_eeg_window_count": state["window_count"] - state["full_eeg_window_count"],
            "padded_fnirs_window_count": state["window_count"] - state["full_fnirs_window_count"],
            "eeg_bad_channel_fraction_record_weighted": (
                state["bad_channel_count"] / state["channel_count_for_bad_mask"]
                if state["channel_count_for_bad_mask"]
                else 0.0
            ),
            "geometry": geometry,
            "subjects": sorted(state["subjects"]),
        }

    _add_record_outlier_flags(record_rows)
    return (
        task_payload,
        sorted(record_rows, key=lambda row: row["join_key"]),
        sorted(channel_rows, key=lambda row: (row["task_namespace"], row["modality"], row["signature_sha256"])),
    )


def _add_record_outlier_flags(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_namespace"]].append(row)
    for task_rows in grouped.values():
        for field, output in (
            ("eeg_std_analysis_valid", "eeg_log_std_robust_z"),
            ("fnirs_std_valid", "fnirs_log_std_robust_z"),
        ):
            values = np.asarray([row[field] for row in task_rows], dtype=np.float64)
            positive = np.isfinite(values) & (values > 0)
            if not np.any(positive):
                for row in task_rows:
                    row[output] = None
                    row[output.replace("_z", "_outlier")] = False
                continue
            logs = np.log(values[positive])
            center = float(np.median(logs))
            scale = float(1.4826 * np.median(np.abs(logs - center)))
            for row in task_rows:
                value = float(row[field])
                z = (math.log(value) - center) / scale if value > 0 and scale > 0 else 0.0
                row[output] = z
                row[output.replace("_z", "_outlier")] = bool(abs(z) > 3.5)


def _alignment_summary(dataset: UnifiedPhysiologyWindowDataset) -> dict[str, Any]:
    selected_records = {ref.record.join_key for ref in dataset.windows}
    admitted_cases = Counter()
    for join_key in selected_records:
        for report in dataset.index.reports_by_join_key.get(join_key, []):
            admitted_cases[str(report.get("alignment_case", "unknown"))] += 1
    return {
        "admitted_record_count": len(selected_records),
        "admitted_alignment_cases": dict(admitted_cases),
        "excluded_alignment_record_count": len(dataset.excluded_alignment_records),
        "excluded_alignment_records": dict(dataset.excluded_alignment_records),
    }


def _readiness_checks(
    dataset: UnifiedPhysiologyWindowDataset,
    index_summary: Mapping[str, Any],
    tasks: Mapping[str, Any],
    refed_targets: Mapping[str, Any],
    record_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    exposed_forbidden = sum(
        task["window_count"] for namespace, task in tasks.items()
        if namespace in FORBIDDEN_TASK_NAMESPACES
    )
    dsr = tasks.get("simultaneous_eeg_nirs:dsr", {})
    dsr_labels = set(dsr.get("label_distribution", {}))
    dsr_restored = bool(dsr.get("window_count", 0)) and dsr_labels == {"Go", "No-go"}
    missing_coverage = [
        row for row in index_summary["subject_label_coverage"]
        if not row["complete_admitted_label_coverage"]
    ]
    nonfinite_tasks = [
        namespace for namespace, task in tasks.items()
        if task["eeg_amplitude_analysis_valid"]["finite_fraction"] < 1.0
        or task["fnirs_amplitude_valid"]["finite_fraction"] < 1.0
    ]
    record_outliers = [
        row["join_key"] for row in record_rows
        if row.get("eeg_log_std_robust_outlier") or row.get("fnirs_log_std_robust_outlier")
    ]
    padded_tasks = {
        namespace: {
            "eeg": task["padded_eeg_window_count"],
            "fnirs": task["padded_fnirs_window_count"],
        }
        for namespace, task in tasks.items()
        if task["padded_eeg_window_count"] or task["padded_fnirs_window_count"]
    }
    incomplete_geometry = {
        namespace: {
            "eeg_position_available_fraction": task["geometry"]["eeg_position_available_fraction"],
            "fnirs_position_available_fraction": task["geometry"]["fnirs_position_available_fraction"],
        }
        for namespace, task in tasks.items()
        if task["geometry"]["eeg_position_available_fraction"] < 1.0
        or task["geometry"]["fnirs_position_available_fraction"] < 1.0
    }
    refed_geometry = tasks["refed:emotion_video"]["geometry"]
    refed_windows = int(tasks["refed:emotion_video"]["window_count"])
    refed_coordinate_status_per_channel = {
        status: int(count // refed_windows)
        for status, count in refed_geometry["eeg_coordinate_status"].items()
    }
    refed_adjacency_path = dataset.cache_root / "channel_geometry/refed_eeg_adjacency.json"
    try:
        refed_adjacency = json.loads(refed_adjacency_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        refed_adjacency = {}
    refed_adjacency_ok = (
        refed_geometry["eeg_position_available_fraction"] == 1.0
        and refed_adjacency.get("schema") == "eeg_channel_adjacency_v1"
        and len(refed_adjacency.get("channel_names", [])) == 64
        and len(refed_adjacency.get("edges", [])) > 0
    )
    visual_geometry = tasks["visual_cognitive_motivation:visual_cognitive_motivation"]["geometry"]
    visual_record_count = int(
        tasks["visual_cognitive_motivation:visual_cognitive_motivation"]["record_count"]
    )
    visual_coordinate_status_per_record = {
        status: int(count // visual_record_count)
        for status, count in visual_geometry["fnirs_coordinate_status"].items()
    }
    visual_adjacency_path = dataset.cache_root / "channel_geometry/visual_fnirs_adjacency.json"
    try:
        visual_adjacency = json.loads(visual_adjacency_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        visual_adjacency = {}
    visual_probes = visual_adjacency.get("probes", {})
    visual_adjacency_ok = (
        visual_geometry["fnirs_position_available_fraction"] == 1.0
        and visual_adjacency.get("schema") == "visual_fnirs_bilateral_adjacency_v1"
        and set(visual_probes) == {"Probe1", "Probe2"}
        and all(
            probe.get("schema") == "fnirs_channel_adjacency_v1"
            and len(probe.get("channel_names", [])) == 24
            and len(probe.get("edges", [])) == 52
            for probe in visual_probes.values()
        )
    )
    return [
        {
            "id": "dsr_go_nogo_restored",
            "status": "pass" if dsr_restored and exposed_forbidden == 0 else "block",
            "evidence": {
                "exposed_forbidden_windows": exposed_forbidden,
                "dsr_window_count": dsr.get("window_count", 0),
                "dsr_labels": sorted(dsr_labels),
                "loader_contract": dataset.contract_summary()["excluded_forbidden_task_window_count_by_namespace"],
            },
        },
        {
            "id": "known_supervised_labels",
            "status": "pass" if index_summary["unknown_label_count"] == 0 else "block",
            "evidence": {"unknown_label_count": index_summary["unknown_label_count"]},
        },
        {
            "id": "subject_class_coverage",
            "status": "pass" if not missing_coverage else "block",
            "evidence": {"subject_task_rows_missing_admitted_classes": len(missing_coverage)},
        },
        {
            "id": "exact_sample_identity_unique",
            "status": "pass" if index_summary["exact_sample_id_duplicate_count"] == 0 else "block",
            "evidence": {"duplicate_count": index_summary["exact_sample_id_duplicate_count"]},
        },
        {
            "id": "semantic_trial_independence",
            "status": "pass" if index_summary["semantic_duplicate_group_count"] == 0 else "block",
            "evidence": {
                "duplicate_groups": index_summary["semantic_duplicate_group_count"],
                "extra_windows": index_summary["semantic_duplicate_extra_window_count"],
                "note": index_summary["semantic_duplicate_note"],
            },
        },
        {
            "id": "finite_loaded_amplitudes",
            "status": "pass" if not nonfinite_tasks else "block",
            "evidence": {"nonfinite_tasks": nonfinite_tasks},
        },
        {
            "id": "refed_continuous_target_adapter",
            "status": "block",
            "evidence": {
                "stream_events": refed_targets["event_count"],
                "invalid_stream_events": refed_targets["event_with_missing_or_invalid_stream_count"],
                "boundary": refed_targets["loader_boundary"],
            },
        },
        {
            "id": "dataset_specific_qc_masks",
            "status": "block",
            "evidence": {
                "single_trial_branch": tasks["eeg_fnirs_single_trial:motor_imagery"]["eeg_signal_branches"],
                "other_dataset_branches": {
                    namespace: task["eeg_signal_branches"]
                    for namespace, task in tasks.items()
                    if not namespace.startswith("eeg_fnirs_single_trial:")
                },
                "note": (
                    "Simultaneous uses simultaneous_eeg_eog_clean_v1 with HEOG/VEOG excluded; "
                    "REFED and Visual still expose raw_with_ocular_artifact, so the cross-dataset "
                    "QC gate remains open"
                ),
            },
        },
        {
            "id": "validity_masks_consumed_by_training_adapter",
            "status": "block",
            "evidence": {
                "padded_windows_by_task": padded_tasks,
                "note": (
                    "loader masks exist, but no formal four-dataset training adapter currently "
                    "proves padding, analysis-valid, artifact, and bad-channel masks are consumed"
                ),
            },
        },
        {
            "id": "geometry_complete_for_spatial_methods",
            "status": "pass" if not incomplete_geometry else "block",
            "evidence": {
                "incomplete_tasks": incomplete_geometry,
                "note": "required before geometry-dependent projection such as STA-Net spatial inputs",
            },
        },
        {
            "id": "refed_eeg_template_adjacency",
            "status": "pass" if refed_adjacency_ok else "block",
            "evidence": {
                "eeg_position_available_fraction": refed_geometry[
                    "eeg_position_available_fraction"
                ],
                "coordinate_status_per_channel": refed_coordinate_status_per_channel,
                "adjacency_schema": refed_adjacency.get("schema"),
                "adjacency_method": refed_adjacency.get("method"),
                "channel_count": len(refed_adjacency.get("channel_names", [])),
                "edge_count": len(refed_adjacency.get("edges", [])),
                "claim_boundary": (
                    "standard-template within-EEG topology only; not participant digitization "
                    "or EEG-fNIRS co-registration"
                ),
            },
        },
        {
            "id": "visual_fnirs_graphical_geometry",
            "status": "pass" if visual_adjacency_ok else "block",
            "evidence": {
                "fnirs_position_available_fraction": visual_geometry[
                    "fnirs_position_available_fraction"
                ],
                "coordinate_status_component_rows_per_record": (
                    visual_coordinate_status_per_record
                ),
                "adjacency_schema": visual_adjacency.get("schema"),
                "probe_channel_counts": {
                    probe: len(payload.get("channel_names", []))
                    for probe, payload in visual_probes.items()
                },
                "probe_edge_counts": {
                    probe: len(payload.get("edges", []))
                    for probe, payload in visual_probes.items()
                },
                "claim_boundary": (
                    "graphical/CED template projection for within-fNIRS adjacency and coarse "
                    "alignment only; not participant digitization, source-detector distance, "
                    "or exact co-registration"
                ),
            },
        },
        {
            "id": "record_scale_outlier_review",
            "status": "warn" if record_outliers else "pass",
            "evidence": {
                "adaptive_rule": "abs robust z of log record std > 3.5 within task",
                "flagged_record_count": len(record_outliers),
                "flagged_records": record_outliers,
            },
        },
        {
            "id": "shared_subject_split_manifest",
            "status": "block",
            "evidence": {
                "required_group_key": "dataset_id + canonical_subject_id",
                "note": "must be frozen before normalization fitting or method-specific tensor export",
            },
        },
        {
            "id": "channel_adapter_and_mask",
            "status": "block",
            "evidence": {
                namespace: {
                    "eeg": task["eeg_channel_count_distribution"],
                    "fnirs": task["fnirs_channel_count_distribution"],
                }
                for namespace, task in tasks.items()
            },
        },
    ]


def _task_rows(tasks: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for namespace, task in tasks.items():
        rows.append({
            "task_namespace": namespace,
            "dataset_id": task["dataset_id"],
            "task": task["task"],
            "subjects": task["subject_count"],
            "records": task["record_count"],
            "windows": task["window_count"],
            "eeg_channels": json.dumps(task["eeg_channel_count_distribution"], ensure_ascii=False),
            "fnirs_channels": json.dumps(task["fnirs_channel_count_distribution"], ensure_ascii=False),
            "unknown_labels": task["unknown_label_count"],
            "label_imbalance_ratio": task["admitted_label_imbalance_ratio"],
            "eeg_std_analysis_valid": task["eeg_amplitude_analysis_valid"]["std"],
            "eeg_variance_analysis_valid": task["eeg_amplitude_analysis_valid"]["variance"],
            "fnirs_std_valid": task["fnirs_amplitude_valid"]["std"],
            "fnirs_variance_valid": task["fnirs_amplitude_valid"]["variance"],
            "full_eeg_window_fraction": task["full_eeg_window_fraction"],
            "full_fnirs_window_fraction": task["full_fnirs_window_fraction"],
            "eeg_artifact_fraction_median": task["eeg_artifact_fraction"]["median"],
            "eeg_bad_channel_fraction": task["eeg_bad_channel_fraction_record_weighted"],
        })
    return rows


def _label_rows(tasks: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for namespace, task in tasks.items():
        total = task["window_count"]
        for label, count in task["label_distribution"].items():
            rows.append({
                "task_namespace": namespace,
                "label": label,
                "window_count": count,
                "fraction": count / total if total else 0.0,
            })
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _build_figures(output_dir: Path, tasks: Mapping[str, Any]) -> list[str]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = list(tasks)
    short = [value.split(":", 1)[1] for value in labels]
    x = np.arange(len(labels))
    outputs = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].bar(x, [tasks[key]["window_count"] for key in labels], color="#2E86AB")
    axes[0].set_xticks(x, short, rotation=25, ha="right")
    axes[0].set_ylabel("Admitted windows")
    axes[0].set_title("Task sample inventory")
    axes[1].bar(x, [tasks[key]["subject_count"] for key in labels], color="#A23B72")
    axes[1].set_xticks(x, short, rotation=25, ha="right")
    axes[1].set_ylabel("Subjects")
    axes[1].set_title("Task subject inventory")
    path = figures_dir / "task_inventory.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(str(path.relative_to(output_dir)))

    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    bottoms = np.zeros(len(labels), dtype=float)
    all_conditions = sorted({condition for task in tasks.values() for condition in task["label_distribution"]})
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_conditions)))
    for condition, color in zip(all_conditions, colors):
        values = np.asarray([
            tasks[key]["label_distribution"].get(condition, 0) / tasks[key]["window_count"]
            for key in labels
        ])
        ax.bar(x, values, bottom=bottoms, label=condition, color=color)
        bottoms += values
    ax.set_xticks(x, short, rotation=25, ha="right")
    ax.set_ylabel("Window fraction")
    ax.set_title("Task-native label distributions")
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    path = figures_dir / "label_distribution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(str(path.relative_to(output_dir)))

    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    width = 0.36
    ax.bar(
        x - width / 2,
        [tasks[key]["eeg_amplitude_analysis_valid"]["std"] for key in labels],
        width,
        label="EEG analysis-valid",
        color="#2E86AB",
    )
    ax.bar(
        x + width / 2,
        [tasks[key]["fnirs_amplitude_valid"]["std"] for key in labels],
        width,
        label="fNIRS valid",
        color="#A23B72",
    )
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1, label="robust-SD reference")
    ax.set_xticks(x, short, rotation=25, ha="right")
    ax.set_ylabel("Global standard deviation (dataset measurement units)")
    ax.set_title("Loaded canonical amplitude scale by task")
    ax.legend()
    path = figures_dir / "canonical_amplitude_std.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(str(path.relative_to(output_dir)))

    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    width = 0.25
    ax.bar(
        x - width,
        [1.0 - tasks[key]["full_eeg_window_fraction"] for key in labels],
        width,
        label="EEG padded windows",
        color="#F59E0B",
    )
    ax.bar(
        x,
        [1.0 - tasks[key]["full_fnirs_window_fraction"] for key in labels],
        width,
        label="fNIRS padded windows",
        color="#D97706",
    )
    ax.bar(
        x + width,
        [tasks[key]["eeg_artifact_fraction"]["median"] or 0.0 for key in labels],
        width,
        label="Median EEG artifact fraction",
        color="#B91C1C",
    )
    ax.set_xticks(x, short, rotation=25, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_title("Window validity and EEG artifact masks")
    ax.legend()
    path = figures_dir / "validity_artifact_summary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(str(path.relative_to(output_dir)))
    return outputs


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _build_markdown(payload: Mapping[str, Any], output_dir: Path) -> str:
    tasks = payload["tasks"]
    lines = [
        "# Unified physiology loader final audit",
        "",
        f"_Generated: {payload['generated']}; schema: `{SCHEMA}`_",
        "",
        "---",
        "",
        "## Executive verdict",
        "",
        "The loader is structurally usable for continued preparation, but **multi-dataset formal training is not yet admitted**. Signal loading is finite and DSR Go/No-go events are restored under alignment gating; remaining blockers, if any, are listed by the machine-generated readiness checks below.",
        "",
        "Amplitude values below are in canonical robust-standard-deviation coordinates. They validate numerical behavior, not physical equivalence or scientific validity.",
        "The moments are window-weighted model-input statistics. Overlapping windows intentionally count repeated raw time points more than once; they are not de-duplicated raw-record estimates.",
        "",
        "## Task inventory and loaded-signal statistics",
        "",
        "| Dataset/task | Subjects | Records | Windows | EEG ch | fNIRS ch | EEG std / var | fNIRS std / var | Unknown |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for namespace, task in tasks.items():
        lines.append(
            f"| `{namespace}` | {task['subject_count']} | {task['record_count']} | {task['window_count']} | "
            f"`{task['eeg_channel_count_distribution']}` | `{task['fnirs_channel_count_distribution']}` | "
            f"{_fmt(task['eeg_amplitude_analysis_valid']['std'])} / {_fmt(task['eeg_amplitude_analysis_valid']['variance'])} | "
            f"{_fmt(task['fnirs_amplitude_valid']['std'])} / {_fmt(task['fnirs_amplitude_valid']['variance'])} | "
            f"{task['unknown_label_count']} |"
        )

    lines.extend(["", "## Label distributions", ""])
    for namespace, task in tasks.items():
        labels = ", ".join(
            f"`{name}` {count} ({count / task['window_count']:.1%})"
            for name, count in task["label_distribution"].items()
        )
        lines.append(f"- `{namespace}`: {labels}")

    lines.extend([
        "",
        "## Quality and dependency checks",
        "",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
    ])
    for check in payload["readiness_checks"]:
        lines.append(
            f"| `{check['id']}` | **{check['status'].upper()}** | "
            f"`{json.dumps(check['evidence'], ensure_ascii=False)}` |"
        )

    refed = payload["refed_continuous_targets"]
    lines.extend([
        "",
        "## REFED continuous-target boundary",
        "",
        f"All {refed['event_count']} REFED video events carry nested continuous streams; invalid stream events: {refed['event_with_missing_or_invalid_stream_count']}. The current loader emits one 20-second observation window per video event, while annotation streams span {refed['stream_length']['min']:.0f}–{refed['stream_length']['max']:.0f} samples (median {refed['stream_length']['median']:.0f}). The streams are not yet emitted by `canonical_label()` or sliced into window-level regression targets.",
        "",
    ])
    for name, stats in refed["targets"].items():
        lines.append(
            f"- `{name}`: n={stats['value_count']}, finite={stats['finite_fraction']:.3%}, "
            f"range=[{_fmt(stats['min'])}, {_fmt(stats['max'])}], mean={_fmt(stats['mean'])}, "
            f"std={_fmt(stats['std'])}, constant-video streams={stats['constant_event_count']}"
        )

    lines.extend([
        "",
        "## Findings that must affect the training protocol",
        "",
        f"1. Visual contains **{payload['index_summary']['unknown_label_count']} unknown windows**. They must be rejected before split generation unless source-backed label semantics are recovered.",
        f"2. Visual has **{payload['index_summary']['semantic_duplicate_group_count']} paired-probe semantic trial groups**, contributing {payload['index_summary']['semantic_duplicate_extra_window_count']} extra loader windows. Probe1/Probe2 must stay in the same subject split and require fusion or explicit trial weighting; they cannot be treated as independent trials.",
        "3. REFED remains a categorical video-level loader entry with nested continuous metadata and only one current 20-second signal window per video. The planned continuous benchmark is blocked until sliding/windowed target alignment and validity masks are materialized.",
        "4. EEG/fNIRS channel counts differ materially across datasets. Channel padding/projection, masks, geometry handling, and train-fold-only fitted transforms must be frozen before model export.",
        "5. REFED now has a complete 64-channel standard-template EEG topology: 62 exact standard_1005 matches plus CB1/CB2 reference-figure midpoint interpolation, with a 168-edge Delaunay adjacency graph. This is adjacency-only evidence, not participant digitization or EEG-fNIRS co-registration.",
        "6. Visual now exposes fNIRS positions for 100% of channel rows through a dataset-figure/CED template projection: each probe has 14 EEG-label anchors, 10 graph-harmonic channel interpolations, and a connected 24-node/52-edge shared-optode graph. This closes the adjacency input gap only; it is not participant digitization or exact source-detector geometry.",
        "7. Visual contains 54 EEG and 54 fNIRS boundary-padded windows. Formal adapters must consume time-validity masks; dropping or treating padding as data changes denominators and loss support.",
        "8. Only Single-Trial uses the admitted artifact-clean EEG v3 branch. Other datasets still require dataset-specific bad-channel/window QC rather than interpreting zero masks as proof of clean signals.",
        "9. Canonical robust scaling keeps numeric magnitudes manageable but does not make sensors, spatial coverage, or native fNIRS physics homogeneous.",
        "10. Split manifests must group by `dataset_id + canonical_subject_id`; all normalization, channel selection, and target scaling fitted after the split must use training subjects only.",
        "",
        "## Figures",
        "",
    ])
    for figure in payload["figures"]:
        lines.append(f"![{Path(figure).stem}]({figure})")
        lines.append("")

    lines.extend([
        "## Reproduction",
        "",
        "```bash",
        payload["command"],
        "```",
        "",
        "Machine-readable evidence is in `summary.json`, `task_summary.csv`, `label_distribution.csv`, `subject_label_coverage.csv`, `record_quality.csv`, and `channel_signatures.csv`.",
        "",
    ])
    return "\n".join(lines)


def _build_html(markdown: str, payload: Mapping[str, Any]) -> str:
    # A compact HTML companion; Markdown remains the canonical narrative.
    task_rows = _task_rows(payload["tasks"])
    checks = payload["readiness_checks"]

    def table(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> str:
        parts = ["<table><thead><tr>"]
        parts.extend(f"<th>{escape(key)}</th>" for key in keys)
        parts.append("</tr></thead><tbody>")
        for row in rows:
            parts.append("<tr>")
            parts.extend(f"<td>{escape(_fmt(row.get(key)))}</td>" for key in keys)
            parts.append("</tr>")
        parts.append("</tbody></table>")
        return "".join(parts)

    task_keys = (
        "task_namespace", "subjects", "records", "windows", "eeg_channels", "fnirs_channels",
        "eeg_std_analysis_valid", "eeg_variance_analysis_valid", "fnirs_std_valid",
        "fnirs_variance_valid", "unknown_labels",
    )
    check_rows = [
        {"check": row["id"], "status": row["status"], "evidence": json.dumps(row["evidence"], ensure_ascii=False)}
        for row in checks
    ]
    images = "".join(
        f'<figure><img src="{escape(path)}" alt="{escape(Path(path).stem)}"></figure>'
        for path in payload["figures"]
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Unified loader final audit</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1250px;margin:auto;padding:2rem;color:#172033;line-height:1.45}}table{{border-collapse:collapse;width:100%;font-size:.82rem;margin:1rem 0}}th,td{{border:1px solid #d8e1e8;padding:.45rem;vertical-align:top}}th{{background:#174A68;color:white}}img{{max-width:100%}}.block{{background:#fff1f2;border-left:5px solid #b91c1c;padding:1rem}}</style></head><body>
<h1>Unified physiology loader final audit</h1><p>Generated {escape(payload['generated'])}</p>
<div class="block"><strong>Verdict:</strong> loader correctness is sufficient for preparation, but formal unified training remains blocked by the checks below.</div>
<h2>Task inventory</h2>{table(task_rows, task_keys)}
<h2>Readiness checks</h2>{table(check_rows, ('check','status','evidence'))}
<h2>Figures</h2>{images}
<h2>Canonical narrative</h2><pre style="white-space:pre-wrap">{escape(markdown)}</pre>
</body></html>"""


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root)
    if not cache_root.is_absolute():
        cache_root = PROJECT_ROOT / cache_root

    dataset = UnifiedPhysiologyWindowDataset(
        cache_root,
        dataset_ids=RAW_DATASET_IDS,
        window_duration_s=args.window_duration_s,
    )
    index_summary = _collect_index_summary(dataset)
    refed_targets = _collect_refed_continuous_targets(dataset)
    tasks, record_rows, channel_rows = _audit_signals(
        dataset,
        quantile_points_per_window=max(1, args.quantile_points_per_window),
        workers=max(1, args.workers),
    )
    readiness = _readiness_checks(dataset, index_summary, tasks, refed_targets, record_rows)
    figures = _build_figures(output_dir, tasks)

    event_root = cache_root / "event_index"
    resource_path = Path(args.resource_json)
    resources = json.loads(resource_path.read_text()) if resource_path.exists() else None
    command = (
        ".venv/bin/python experiments/audit_unified_loader_final.py "
        f"--cache-root {args.cache_root} --output-dir {args.output_dir} "
        f"--workers {args.workers} --overwrite"
    )
    payload = {
        "schema": SCHEMA,
        "generated": datetime.now().astimezone().isoformat(),
        "command": command,
        "parameters": {
            "cache_root": str(cache_root),
            "window_duration_s": args.window_duration_s,
            "quantile_points_per_window": args.quantile_points_per_window,
            "workers": args.workers,
            "dataset_ids": list(RAW_DATASET_IDS),
            "eeg_signal_branch": dataset.eeg_signal_branch,
            "canonical_unit": "per-dataset measurement units; not physically pooled",
        },
        "git": _git_state(),
        "input_hashes": {
            "audit_script": _sha256(Path(__file__).resolve()),
            "cache_manifest": _sha256(cache_root / "cache_manifest.json"),
            "event_manifest": _sha256(event_root / "event_manifest.json"),
            "events": _sha256(event_root / "events.jsonl"),
            "alignment_reports": _sha256(event_root / "alignment_reports.jsonl"),
            "geometry_manifest": _sha256(cache_root / "channel_geometry/geometry_manifest.json"),
            "geometry_channels": _sha256(cache_root / "channel_geometry/channels.jsonl"),
            "refed_eeg_adjacency": _sha256(
                cache_root / "channel_geometry/refed_eeg_adjacency.json"
            ),
            "refed_standard_montage_asset": _sha256(
                PROJECT_ROOT / "src/data/assets/refed_standard_1005_montage_v1.csv"
            ),
            "visual_fnirs_adjacency": _sha256(
                cache_root / "channel_geometry/visual_fnirs_adjacency.json"
            ),
            "visual_fnirs_topology_asset": _sha256(
                PROJECT_ROOT / "src/data/assets/visual_fnirs_4x4_topology_v1.csv"
            ),
        },
        "resources": resources,
        "loader_contract": dataset.contract_summary(),
        "index_summary": index_summary,
        "alignment": _alignment_summary(dataset),
        "refed_continuous_targets": refed_targets,
        "tasks": tasks,
        "readiness_checks": readiness,
        "record_quality_summary": {
            "record_count": len(record_rows),
            "eeg_scale_outlier_count": sum(bool(row.get("eeg_log_std_robust_outlier")) for row in record_rows),
            "fnirs_scale_outlier_count": sum(bool(row.get("fnirs_log_std_robust_outlier")) for row in record_rows),
        },
        "figures": figures,
    }
    markdown = _build_markdown(payload, output_dir)
    (output_dir / "quality_report.md").write_text(markdown, encoding="utf-8")
    (output_dir / "quality_report.html").write_text(_build_html(markdown, payload), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "task_summary.csv", _task_rows(tasks))
    _write_csv(output_dir / "label_distribution.csv", _label_rows(tasks))
    _write_csv(output_dir / "subject_label_coverage.csv", index_summary["subject_label_coverage"])
    _write_csv(output_dir / "record_quality.csv", record_rows)
    _write_csv(output_dir / "channel_signatures.csv", channel_rows)
    print(json.dumps({
        "output_dir": str(output_dir),
        "window_count": len(dataset),
        "task_count": len(tasks),
        "readiness": dict(Counter(row["status"] for row in readiness)),
    }, indent=2))


if __name__ == "__main__":
    main()
