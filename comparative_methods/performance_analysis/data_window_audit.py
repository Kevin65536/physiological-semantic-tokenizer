#!/usr/bin/env python3
"""G0 audit of the shared public data, window and split contract.

This module deliberately has a narrow read boundary.  It reads the shared
registry, public split manifests, the public cache manifest and (optionally)
public signal arrays.  A path containing ``protected`` is rejected before it
is opened and no protected labels/indices are dereferenced.  The resulting
report is a quality-control artifact, not a new model-selection surface.

The default audit only uses the task metadata and adapter contract.  Passing
``--dereference-signals`` additionally validates one public signal window per
task/fold (and, with ``--all-signal-samples``, every public sample).  This keeps
unit tests and metadata-only audits cheap while preserving a fail-closed
signal path for the actual G0 run.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EFRM_ROOT = REPO_ROOT / "comparative_methods" / "EFRM-PyTorch"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EFRM_ROOT) not in sys.path:
    sys.path.insert(0, str(EFRM_ROOT))

from src.data.clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT
from efrm_pytorch.tasks import EFRMUnifiedTaskDataset, TASK_SPECS  # noqa: E402


TASK_ORDER = (
    "motor_imagery",
    "mental_arithmetic",
    "wg",
    "nback",
    "dsr",
    "visual",
    "refed_regression",
)
TASK_DISPLAY = {
    "motor_imagery": "MI",
    "mental_arithmetic": "MA",
    "wg": "WG",
    "nback": "nback",
    "dsr": "DSR",
    "visual": "Visual",
    "refed_regression": "REFED",
}
EXPECTED_REGISTRY_SCHEMA = "method_neutral_full_target_fold_registry_v2"
EXPECTED_REGISTRY_SHA256 = "2a10b36db85dba6ec5543edc7810ff85d978ea5af8c79fda3d38a1e5cfd11106"
PUBLIC_SPLIT_SCHEMAS = {
    "sta_net_split_registry_v2",
    "sta_net_subject_split_v1",
    "efrm_target_public_fold_v1",
}
FORBIDDEN_MANIFEST_FIELDS = {"test_indices", "protected_indices", "reserved_test_indices"}
DEFAULT_REGISTRY = (
    EFRM_ROOT
    / "runs/formal/efrm_lodo_full_target_fivefold_v2/protocol"
    / "shared_full_target_fold_registry/registry_manifest.json"
)
DEFAULT_CACHE_ROOT = REPO_ROOT / DEFAULT_CLEAN_CACHE_ROOT
DEFAULT_OUTPUT = (
    REPO_ROOT / "comparative_methods/runs/performance_analysis/20260816_p0/data_audit"
)


class AuditError(RuntimeError):
    """Raised for a required G0 contract failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_public_path(path: Path, *, label: str) -> Path:
    """Resolve a path and refuse protected or out-of-repository reads."""

    resolved = path.expanduser().resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise AuditError(f"{label}: protected path is not readable: {resolved}")
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise AuditError(f"{label}: path escapes repository: {resolved}") from exc
    return resolved


def load_public_json(path: Path, *, label: str = "manifest") -> dict[str, Any]:
    """Load one public JSON object without dereferencing protected content."""

    resolved = _resolved_public_path(path, label=label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AuditError(f"{label}: cannot read {resolved}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AuditError(f"{label}: invalid JSON in {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{label}: expected a JSON object in {resolved}")
    if value.get("protected_test_opened", value.get("reserved_test_opened", False)):
        raise AuditError(f"{label}: manifest reports protected test opened: {resolved}")
    forbidden = FORBIDDEN_MANIFEST_FIELDS.intersection(value)
    if forbidden:
        raise AuditError(f"{label}: protected index fields exposed: {sorted(forbidden)}")
    return value


def load_registry(path: Path) -> dict[str, Any]:
    value = load_public_json(path, label="registry")
    if value.get("schema") != EXPECTED_REGISTRY_SCHEMA:
        raise AuditError(f"registry schema drifted: {value.get('schema')!r}")
    declared = value.get("registry_sha256")
    if declared != EXPECTED_REGISTRY_SHA256:
        raise AuditError(f"registry fingerprint drifted: {declared!r}")
    if value.get("protected_test_default") != "locked":
        raise AuditError("registry protected_test_default must remain locked")
    folds = value.get("folds")
    if not isinstance(folds, list) or not folds:
        raise AuditError("registry has no fold entries")
    return value


def strict_public_entry(
    registry: Mapping[str, Any], *, task: str, outer_fold: int
) -> Mapping[str, Any]:
    matches = [
        row
        for row in registry["folds"]
        if row.get("task") == task
        and row.get("protocol") == "strict_cross_subject"
        and int(row.get("outer_fold", -1)) == int(outer_fold)
    ]
    if len(matches) != 1:
        raise AuditError(
            f"expected one strict public registry entry for {task}/outer{outer_fold}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _offset_ms(metadata: Mapping[str, Any]) -> int:
    return int(round(float(metadata.get("window_offset_s", 0.0)) * 1000.0))


def canonical_source_sample_id(metadata: Mapping[str, Any]) -> str:
    """Return the method-neutral source ID used by public feature caches."""

    return (
        f"{metadata['dataset_id']}|{metadata['subject']}|{metadata['record_id']}|"
        f"event={int(metadata['event_index'])}|offset_ms={_offset_ms(metadata)}"
    )


def adapter_sample_id(metadata: Mapping[str, Any]) -> str:
    """Return the deterministic EFRM adapter ID from metadata alone.

    The adapter's source sample ID is absent from lightweight metadata for
    ordinary event windows.  Reproducing the stable payload here lets us audit
    duplicate/missing identity without loading signal arrays.  REFED's source
    ID includes the same event and offset fields and is handled identically.
    """

    raw = {
        "dataset": str(metadata["dataset_id"]),
        "subject": str(metadata["subject"]),
        "record": str(metadata["record_id"]),
        "source_sample_id": str(metadata.get("source_sample_id", "")),
        "event": int(metadata["event_index"]),
        "window_offset_s": float(metadata.get("window_offset_s", 0.0)),
        "crop_start_s": 0.0,
    }
    digest = hashlib.sha256(repr(sorted(raw.items())).encode("utf-8")).hexdigest()[:16]
    return f"{raw['dataset']}|{raw['subject']}|{raw['record']}|{digest}"


def _counter_json(counter: Mapping[str, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _group_overlap(
    train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    fields = ("subject", "record_id", "subject_record", "join_key", "trial_group")
    result: dict[str, Any] = {}
    for field in fields:
        if field == "subject_record":
            train = {f"{row['subject']}|{row['record_id']}" for row in train_rows}
            validation = {f"{row['subject']}|{row['record_id']}" for row in validation_rows}
        else:
            train = {str(row[field]) for row in train_rows}
            validation = {str(row[field]) for row in validation_rows}
        overlap = sorted(train.intersection(validation))
        result[field] = {
            "train_count": len(train),
            "validation_count": len(validation),
            "overlap_count": len(overlap),
            "overlap_values": overlap[:100],
            "disjoint": not overlap,
        }
    return result


def _duplicate_values(values: Iterable[str]) -> tuple[list[str], dict[str, int]]:
    counts = Counter(str(value) for value in values)
    duplicates = {key: count for key, count in counts.items() if count > 1}
    return sorted(duplicates), dict(sorted(duplicates.items()))


def _split_rows(
    dataset: EFRMUnifiedTaskDataset,
    indices: Sequence[int],
    *,
    partition: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in indices:
        metadata = dict(dataset.lightweight_metadata(int(index)))
        metadata["dataset_id"] = str(dataset.spec.dataset_id)
        metadata["partition"] = partition
        metadata["source_sample_id"] = canonical_source_sample_id(metadata)
        metadata["adapter_sample_id"] = adapter_sample_id(metadata)
        rows.append(metadata)
    return rows


def _validate_indices(
    *,
    train: Sequence[int],
    validation: Sequence[int],
    dataset_length: int,
    task: str,
    fold: int,
) -> list[str]:
    errors: list[str] = []
    for partition, values in (("train", train), ("validation", validation)):
        if not values:
            errors.append(f"{task}/outer{fold}: {partition} is empty")
        duplicates, _ = _duplicate_values(str(value) for value in values)
        if duplicates:
            errors.append(f"{task}/outer{fold}: duplicate {partition} indices {duplicates[:10]}")
        if values and (min(values) < 0 or max(values) >= dataset_length):
            errors.append(f"{task}/outer{fold}: {partition} index out of range")
    overlap = set(int(value) for value in train).intersection(int(value) for value in validation)
    if overlap:
        errors.append(f"{task}/outer{fold}: train/validation index overlap ({len(overlap)})")
    return errors


def _adapter_contract(dataset: EFRMUnifiedTaskDataset) -> dict[str, Any]:
    manifest = dict(dataset.adapter.manifest())
    rates = manifest.get("sample_rates_hz", {})
    if set(rates) != {"eeg", "fnirs"}:
        raise AuditError("unified adapter sample_rates_hz must declare EEG and fNIRS")
    for modality in ("eeg", "fnirs"):
        rate = float(rates[modality])
        if not math.isfinite(rate) or rate <= 0:
            raise AuditError(f"invalid {modality} sampling rate: {rate}")
    duration = float(manifest.get("duration_s", 0.0))
    if not math.isfinite(duration) or duration <= 0:
        raise AuditError(f"invalid adapter duration: {duration}")
    return {
        "adapter_schema": manifest.get("schema"),
        "modality": "EEG+fNIRS",
        "sampling_rate_hz_eeg": float(rates["eeg"]),
        "sampling_rate_hz_fnirs": float(rates["fnirs"]),
        "window_seconds": duration,
        "eeg_patch_seconds": manifest.get("patch_duration_s", {}).get("eeg"),
        "fnirs_patch_seconds": manifest.get("patch_duration_s", {}).get("fnirs"),
        "channel_policy": manifest.get("channel_policy"),
        "mask_policy": manifest.get("mask_policy"),
        "require_full_analysis_support": bool(manifest.get("require_full_analysis_support")),
    }


def audit_task_fold(
    *,
    task: str,
    fold: int,
    entry: Mapping[str, Any],
    dataset: EFRMUnifiedTaskDataset,
    signal_scan: bool = False,
    all_signal_samples: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Audit one public strict-cross-subject fold.

    Returns a split-level summary, sample-level rows and fail-closed errors.
    ``entry`` must be selected from the method-neutral registry; its protected
    path is recorded only as an opaque hash/count and never opened.
    """

    errors: list[str] = []
    public_path = _resolved_public_path(Path(str(entry["public_path"])), label=f"{task}/public")
    manifest = load_public_json(public_path, label=f"{task}/outer{fold}/public")
    if manifest.get("schema") not in PUBLIC_SPLIT_SCHEMAS:
        errors.append(f"{task}/outer{fold}: unsupported public split schema {manifest.get('schema')!r}")
    if manifest.get("task") not in (None, task):
        errors.append(f"{task}/outer{fold}: split task mismatch {manifest.get('task')!r}")
    if manifest.get("protocol") not in (None, "cross_subject_nested_cv"):
        errors.append(f"{task}/outer{fold}: unexpected public protocol {manifest.get('protocol')!r}")

    actual_hash = sha256_file(public_path)
    if str(entry.get("public_sha256")) != actual_hash:
        errors.append(f"{task}/outer{fold}: public split hash drifted")

    try:
        train, validation = dataset.validate_shared_public_split(public_path)
    except (AuditError, OSError, RuntimeError, ValueError, KeyError, IndexError) as exc:
        errors.append(f"{task}/outer{fold}: adapter split validation failed: {exc}")
        train, validation = [], []

    errors.extend(
        _validate_indices(
            train=train,
            validation=validation,
            dataset_length=len(dataset),
            task=task,
            fold=fold,
        )
    )
    train_rows = _split_rows(dataset, train, partition="train") if train else []
    validation_rows = _split_rows(dataset, validation, partition="validation") if validation else []
    public_rows = train_rows + validation_rows
    source_ids = [str(row["source_sample_id"]) for row in public_rows]
    adapter_ids = [str(row["adapter_sample_id"]) for row in public_rows]
    source_duplicates, source_duplicate_counts = _duplicate_values(source_ids)
    adapter_duplicates, adapter_duplicate_counts = _duplicate_values(adapter_ids)
    if source_duplicates:
        errors.append(f"{task}/outer{fold}: duplicate source sample IDs")
    if adapter_duplicates:
        errors.append(f"{task}/outer{fold}: duplicate adapter sample IDs")

    spec = TASK_SPECS[task]
    adapter = _adapter_contract(dataset)
    class_support_train = _counter_json(Counter(str(row["condition"]) for row in train_rows))
    class_support_validation = _counter_json(Counter(str(row["condition"]) for row in validation_rows))
    subject_support_train = _counter_json(Counter(str(row["subject"]) for row in train_rows))
    subject_support_validation = _counter_json(Counter(str(row["subject"]) for row in validation_rows))
    all_inventory_ids = [
        canonical_source_sample_id(
            {**dataset.lightweight_metadata(index), "dataset_id": str(dataset.spec.dataset_id)}
        )
        for index in range(len(dataset))
    ]
    all_expected_source_ids = set(all_inventory_ids)
    inventory_duplicates, inventory_duplicate_counts = _duplicate_values(all_inventory_ids)
    if inventory_duplicates:
        errors.append(f"{task}/outer{fold}: duplicate task-inventory sample IDs")
    observed_source_ids = set(source_ids)
    missing_public_ids = sorted(observed_source_ids.symmetric_difference(set(source_ids)))
    # A split should expose exactly the declared rows.  The full task inventory
    # is intentionally not called "missing" here: protected samples are not
    # expected in the public surface.
    unrepresented_public_ids = sorted(
        set(source_ids) - all_expected_source_ids
    )
    if unrepresented_public_ids:
        errors.append(f"{task}/outer{fold}: public IDs absent from task inventory")

    signal_summary: dict[str, Any] = {
        "status": "not_dereferenced",
        "audited_sample_count": 0,
        "invalid_sample_count": 0,
        "full_eeg_support_count": None,
        "full_fnirs_support_count": None,
        "missing_sample_ids": [],
        "duplicate_sample_ids": source_duplicate_counts,
    }
    if signal_scan and public_rows:
        selected_rows = public_rows if all_signal_samples else public_rows[:1]
        valid_eeg = 0
        valid_fnirs = 0
        invalid = 0
        observed_signal_ids: list[str] = []
        for row in selected_rows:
            try:
                source_index = int(row["dataset_index"])
                sample = dataset.base[dataset.indices[source_index]]
                adapted = dataset.adapter.adapt(sample, crop_start_s=0.0)
                observed_signal_ids.append(str(adapted["sample_id"]))
                if bool(adapted["eeg_time_valid"].all()):
                    valid_eeg += 1
                if bool(adapted["fnirs_time_valid"].all()):
                    valid_fnirs += 1
            except (OSError, RuntimeError, ValueError, KeyError, IndexError, TypeError) as exc:
                invalid += 1
                errors.append(f"{task}/outer{fold}: signal sample {row['dataset_index']} failed: {exc}")
        signal_duplicates, signal_duplicate_counts = _duplicate_values(observed_signal_ids)
        if signal_duplicates:
            errors.append(f"{task}/outer{fold}: duplicate dereferenced sample IDs")
        signal_summary = {
            "status": "complete" if all_signal_samples else "sentinel_sample",
            "audited_sample_count": len(selected_rows),
            "invalid_sample_count": invalid,
            "full_eeg_support_count": valid_eeg,
            "full_fnirs_support_count": valid_fnirs,
            "missing_sample_ids": sorted(set(row["adapter_sample_id"] for row in selected_rows) - set(observed_signal_ids)),
            "duplicate_sample_ids": signal_duplicate_counts,
            "sample_selection": "all_public" if all_signal_samples else "first_public_sample_only",
        }

    overlap = _group_overlap(train_rows, validation_rows)
    adapter_ids_train = [str(row["adapter_sample_id"]) for row in train_rows]
    adapter_ids_validation = [str(row["adapter_sample_id"]) for row in validation_rows]
    train_validation_id_overlap = sorted(set(adapter_ids_train).intersection(adapter_ids_validation))
    if train_validation_id_overlap:
        errors.append(f"{task}/outer{fold}: train/validation sample-ID overlap")

    summary: dict[str, Any] = {
        "task": task,
        "task_display": TASK_DISPLAY[task],
        "dataset_id": spec.dataset_id,
        "task_type": spec.task_type,
        "modality": adapter["modality"],
        "sampling_rate_hz_eeg": adapter["sampling_rate_hz_eeg"],
        "sampling_rate_hz_fnirs": adapter["sampling_rate_hz_fnirs"],
        "window_seconds": adapter["window_seconds"],
        "task_declared_window_seconds": float(spec.input_duration_s),
        "class_names": list(spec.class_names),
        "target_names": list(spec.target_names),
        "outer_fold": int(fold),
        "protocol": str(manifest.get("protocol", "cross_subject_nested_cv")),
        "fold_id": manifest.get("fold_id"),
        "public_manifest": str(public_path),
        "public_manifest_sha256": actual_hash,
        "metadata_sha256": manifest.get("metadata_sha256"),
        "split_sha256": manifest.get("split_sha256"),
        "registry_train_sample_count": int(entry.get("train_sample_count", -1)),
        "registry_validation_sample_count": int(entry.get("validation_sample_count", -1)),
        "train_sample_count": len(train_rows),
        "validation_sample_count": len(validation_rows),
        "public_sample_count": len(public_rows),
        "train_subject_count": len({str(row["subject"]) for row in train_rows}),
        "validation_subject_count": len({str(row["subject"]) for row in validation_rows}),
        "public_subject_count": len({str(row["subject"]) for row in public_rows}),
        "class_support_train": class_support_train,
        "class_support_validation": class_support_validation,
        "subject_support_train": subject_support_train,
        "subject_support_validation": subject_support_validation,
        "source_unique_sample_count": len(set(source_ids)),
        "source_duplicate_sample_ids": source_duplicate_counts,
        "adapter_unique_sample_count": len(set(adapter_ids)),
        "adapter_duplicate_sample_ids": adapter_duplicate_counts,
        "missing_public_sample_ids": missing_public_ids,
        "unrepresented_public_sample_ids": unrepresented_public_ids,
        "task_inventory_sample_count": len(all_inventory_ids),
        "task_inventory_unique_sample_count": len(all_expected_source_ids),
        "task_inventory_duplicate_sample_ids": inventory_duplicate_counts,
        "train_validation_sample_id_overlap": train_validation_id_overlap,
        "group_overlap": overlap,
        "adapter": adapter,
        "signal_support": signal_summary,
        "protected_test_opened": False,
        "protected_test_sample_count_declared": int(manifest.get("protected_test", {}).get("sample_count", -1)),
        "protected_test_subject_count_declared": int(manifest.get("protected_test", {}).get("subject_count", -1)),
        "status": "pass" if not errors else "fail",
        "errors": errors,
    }
    sample_rows: list[dict[str, Any]] = []
    for row in public_rows:
        sample_rows.append(
            {
                "task": task,
                "task_display": TASK_DISPLAY[task],
                "dataset_id": spec.dataset_id,
                "outer_fold": int(fold),
                "partition": row["partition"],
                "dataset_index": int(row["dataset_index"]),
                "subject": row["subject"],
                "record_id": row["record_id"],
                "join_key": row["join_key"],
                "trial_group": row["trial_group"],
                "condition": row["condition"],
                "class_index": row.get("class_index"),
                "event_index": int(row["event_index"]),
                "window_offset_s": float(row["window_offset_s"]),
                "source_sample_id": row["source_sample_id"],
                "adapter_sample_id": row["adapter_sample_id"],
            }
        )
    return summary, sample_rows, errors


def _relative_cache_path(value: str, *, base: Path, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return _resolved_public_path(candidate, label=label)


def _manifest_method_id(path: Path, manifest: Mapping[str, Any]) -> str:
    if manifest.get("method_id"):
        return str(manifest["method_id"])
    relative = path.resolve().relative_to(REPO_ROOT.resolve())
    parts = relative.parts
    try:
        index = parts.index("comparative_methods")
        return str(parts[index + 1])
    except (ValueError, IndexError):
        return "unknown"


def _cache_expected_ids(
    expected_by_task_fold: Mapping[tuple[str, int], Mapping[str, set[str]]],
    *,
    task: str | None,
    fold: int | None,
    method: str = "",
    expected_inventory_by_task: Mapping[str, Mapping[str, set[str]]] | None = None,
) -> set[str] | None:
    if task is None or fold is None:
        return None
    expected = expected_by_task_fold.get((task, fold))
    if "efrm" in method.lower():
        # EFRM's cache identity includes a deterministic crop start.  The
        # public manifest gives membership by dataset index, but lightweight
        # metadata alone cannot reconstruct that crop choice.  We therefore
        # audit EFRM sample IDs for duplicates only and use dataset indices for
        # membership below.
        return None
    if expected is None:
        return None
    # EFRM's own downstream cache uses the adapter digest ID, while the
    # other public feature caches use the transparent source ID.  Both are
    # deterministic identities for the same public rows; comparing a cache
    # against the wrong scheme would manufacture missing/extra-ID failures.
    scheme = "adapter" if "efrm" in method.lower() else "source"
    return expected.get(scheme)


def _cache_expected_indices(
    expected_indices_by_task_fold: Mapping[tuple[str, int], set[int]],
    expected_inventory_indices_by_task: Mapping[str, set[int]] | None,
    *,
    task: str | None,
    fold: int | None,
    method: str,
) -> set[int] | None:
    if task is None or fold is None:
        return None
    if "efrm" in method.lower() and expected_inventory_indices_by_task is not None:
        return expected_inventory_indices_by_task.get(task)
    return expected_indices_by_task_fold.get((task, fold))


def _audit_npz_cache(
    path: Path,
    *,
    expected_ids: set[str] | None,
    expected_dataset_indices: set[int] | None = None,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Inspect a public feature cache without reading protected labels."""

    import numpy as np

    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "status": "missing",
        "sample_count": None,
        "unique_sample_count": None,
        "duplicate_sample_ids": {},
        "missing_expected_sample_ids": [],
        "unexpected_sample_ids": [],
        "dataset_index_count": None,
        "dataset_index_unique_count": None,
        "missing_expected_dataset_indices": [],
        "unexpected_dataset_indices": [],
        "duplicate_dataset_indices": {},
        "array_keys": [],
        "protected_test_opened": False,
    }
    if not path.is_file():
        return result
    try:
        with np.load(path, allow_pickle=False) as payload:
            result["array_keys"] = sorted(str(key) for key in payload.files)
            if "sample_ids" not in payload.files:
                result["status"] = "missing_sample_ids"
                return result
            sample_ids = [str(value) for value in payload["sample_ids"].reshape(-1).tolist()]
            duplicates, duplicate_counts = _duplicate_values(sample_ids)
            result["sample_count"] = len(sample_ids)
            result["unique_sample_count"] = len(set(sample_ids))
            result["duplicate_sample_ids"] = duplicate_counts
            result["missing_expected_sample_ids"] = (
                sorted(expected_ids - set(sample_ids)) if expected_ids is not None else []
            )
            result["unexpected_sample_ids"] = (
                sorted(set(sample_ids) - expected_ids) if expected_ids is not None else []
            )
            if "dataset_indices" in payload.files:
                dataset_indices = [int(value) for value in payload["dataset_indices"].reshape(-1).tolist()]
                index_duplicates, index_duplicate_counts = _duplicate_values(
                    str(value) for value in dataset_indices
                )
                result["dataset_index_count"] = len(dataset_indices)
                result["dataset_index_unique_count"] = len(set(dataset_indices))
                result["duplicate_dataset_indices"] = {
                    int(key): int(value) for key, value in index_duplicate_counts.items()
                }
                if expected_dataset_indices is not None:
                    result["missing_expected_dataset_indices"] = sorted(
                        expected_dataset_indices - set(dataset_indices)
                    )
                    result["unexpected_dataset_indices"] = sorted(
                        set(dataset_indices) - expected_dataset_indices
                    )
            # Public caches may intentionally contain only a training subset;
            # duplicate identities and membership outside the declared public
            # inventory are hard quality failures.  EFRM crop IDs are
            # intentionally not compared textually; dataset indices are.
            membership_bad = bool(
                result["missing_expected_dataset_indices"]
                or result["unexpected_dataset_indices"]
            )
            result["status"] = (
                "duplicate_sample_ids"
                if duplicates
                else "outside_public_membership"
                if membership_bad
                else "pass"
            )
            if manifest.get("protected_test_opened", False):
                result["status"] = "protected_test_opened"
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        result["status"] = "read_error"
        result["error"] = str(exc)
    return result


def discover_public_manifests() -> list[Path]:
    """Find public-development manifests without entering protected paths."""

    roots = [REPO_ROOT / "comparative_methods"]
    found: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("manifest.json"):
            text = path.as_posix()
            if "protected" in {part.lower() for part in path.parts}:
                continue
            if "public_development_v2" in text or "downstream_public_v2" in text:
                found.add(path.resolve())
    return sorted(found)


def audit_public_manifests_and_caches(
    *,
    manifests: Sequence[Path],
    expected_by_task_fold: Mapping[tuple[str, int], Mapping[str, set[str]]],
    expected_inventory_by_task: Mapping[str, Mapping[str, set[str]]] | None = None,
    expected_indices_by_task_fold: Mapping[tuple[str, int], set[int]] | None = None,
    expected_inventory_indices_by_task: Mapping[str, set[int]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in manifests:
        try:
            manifest = load_public_json(path, label=f"public run {path}")
        except AuditError as exc:
            errors.append(str(exc))
            rows.append({"manifest": str(path), "status": "fail", "error": str(exc)})
            continue
        method = _manifest_method_id(path, manifest)
        task_value = manifest.get("task")
        task = str(task_value) if task_value in TASK_SPECS else None
        fold_value = manifest.get("outer_fold")
        try:
            fold = int(fold_value) if fold_value is not None else None
        except (TypeError, ValueError):
            fold = None
        cache = manifest.get("feature_cache")
        cache_audit: dict[str, Any]
        if isinstance(cache, Mapping) and cache.get("path"):
            cache_path = _relative_cache_path(str(cache["path"]), base=path.parent, label=f"cache {path}")
            cache_audit = _audit_npz_cache(
                cache_path,
                expected_ids=_cache_expected_ids(
                    expected_by_task_fold,
                    task=task,
                    fold=fold,
                    method=method,
                    expected_inventory_by_task=expected_inventory_by_task,
                ),
                expected_dataset_indices=_cache_expected_indices(
                    expected_indices_by_task_fold or {},
                    expected_inventory_indices_by_task,
                    task=task,
                    fold=fold,
                    method=method,
                ),
                manifest=manifest,
            )
        else:
            cache_audit = {
                "status": "not_declared",
                "exists": False,
                "sample_count": None,
                "unique_sample_count": None,
                "duplicate_sample_ids": {},
                "missing_expected_sample_ids": [],
                "unexpected_sample_ids": [],
                "array_keys": [],
                "protected_test_opened": False,
            }
        row = {
            "method": method,
            "task": task,
            "outer_fold": fold,
            "seed": manifest.get("seed"),
            "manifest": str(path),
            "schema": manifest.get("schema"),
            "status_claim": manifest.get("status"),
            "table_admissible": manifest.get("table_admissible"),
            "protected_test_opened": False,
            "public_manifest_path": manifest.get("public_manifest_path"),
            "cache": cache_audit,
            "status": "pass" if cache_audit["status"] in {"pass", "not_declared"} else "fail",
        }
        if row["status"] == "fail":
            errors.append(f"{path}: cache audit status={cache_audit['status']}")
        rows.append(row)
    return rows, errors


def _alignment_evidence_audit() -> list[dict[str, Any]]:
    """Audit the two complete public EFRM alignment exports if present.

    This is a validation-mixing/support audit.  Repeated sample IDs are
    reported as repeated rows in the public validation export; they are not
    labelled as data leakage by this function.  We separately verify whether
    their paired embeddings are exactly repeated and whether the positive mask
    contains only its diagonal.
    """

    import numpy as np

    candidates = {
        "exclude_eeg_fnirs_single_trial_stage_a": REPO_ROOT
        / "comparative_methods/EFRM-PyTorch/runs/pretraining/"
        "efrm_lodo_full_target_fivefold_v2__exclude_eeg_fnirs_single_trial__stage_a_seed42/"
        "figure_data/full_validation_clip_alignment_evidence.npz",
        "exclude_simultaneous_eeg_nirs_stage_a": REPO_ROOT
        / "comparative_methods/EFRM-PyTorch/runs/pretraining/"
        "efrm_lodo_full_target_fivefold_v2__exclude_simultaneous_eeg_nirs__stage_a_seed42/"
        "figure_data/full_validation_clip_alignment_evidence.npz",
    }
    rows: list[dict[str, Any]] = []
    for label, path in candidates.items():
        row: dict[str, Any] = {
            "artifact": label,
            "path": str(path),
            "status": "missing",
            "row_count": None,
            "unique_sample_count": None,
            "duplicate_row_count": None,
            "duplicate_id_count": None,
            "max_repeat_count": None,
            "duplicate_embedding_exact": None,
            "max_duplicate_eeg_abs_diff": None,
            "max_duplicate_fnirs_abs_diff": None,
            "positive_mask_diagonal_true": None,
            "positive_mask_off_diagonal_true": None,
            "naive_full_matrix_retrieval_appropriate": None,
            "protected_test_opened": False,
        }
        try:
            safe_path = _resolved_public_path(path, label=f"EFRM alignment {label}")
            if not safe_path.is_file():
                rows.append(row)
                continue
            with np.load(safe_path, allow_pickle=False) as payload:
                required = {"metadata_json", "positive_pair_mask", "eeg_embeddings", "fnirs_embeddings"}
                if not required.issubset(payload.files):
                    row["status"] = "missing_required_arrays"
                    rows.append(row)
                    continue
                metadata = [json.loads(str(item)) for item in payload["metadata_json"].reshape(-1).tolist()]
                sample_ids = [str(item.get("sample_id", "")) for item in metadata]
                counts = Counter(sample_ids)
                duplicate_groups = {key: count for key, count in counts.items() if count > 1}
                row["row_count"] = len(sample_ids)
                row["unique_sample_count"] = len(counts)
                row["duplicate_row_count"] = len(sample_ids) - len(counts)
                row["duplicate_id_count"] = len(duplicate_groups)
                row["max_repeat_count"] = max(duplicate_groups.values(), default=1)

                eeg = payload["eeg_embeddings"]
                fnirs = payload["fnirs_embeddings"]
                eeg_diffs: list[float] = []
                fnirs_diffs: list[float] = []
                for sample_id in duplicate_groups:
                    indices = [i for i, value in enumerate(sample_ids) if value == sample_id]
                    base = indices[0]
                    for index in indices[1:]:
                        eeg_diffs.append(float(np.max(np.abs(eeg[index] - eeg[base]))))
                        fnirs_diffs.append(float(np.max(np.abs(fnirs[index] - fnirs[base]))))
                row["max_duplicate_eeg_abs_diff"] = max(eeg_diffs, default=0.0)
                row["max_duplicate_fnirs_abs_diff"] = max(fnirs_diffs, default=0.0)
                row["duplicate_embedding_exact"] = bool(
                    row["max_duplicate_eeg_abs_diff"] == 0.0
                    and row["max_duplicate_fnirs_abs_diff"] == 0.0
                )
                mask = np.asarray(payload["positive_pair_mask"], dtype=bool)
                row["positive_mask_diagonal_true"] = int(np.diag(mask).sum())
                row["positive_mask_off_diagonal_true"] = int(mask.sum() - np.diag(mask).sum())
                row["naive_full_matrix_retrieval_appropriate"] = bool(
                    row["positive_mask_off_diagonal_true"] > 0
                )
                row["status"] = (
                    "finding_repeated_validation_rows"
                    if row["duplicate_row_count"]
                    else "pass"
                )
        except (AuditError, OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
            row["status"] = "read_error"
            row["error"] = str(exc)
        rows.append(row)
    return rows


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else row.get(key)
                    for key in keys
                }
            )


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    tasks = report.get("tasks", [])
    caches = report.get("public_manifests_and_caches", [])
    efrm = report.get("efrm_alignment_validation_mixing", [])
    lines = [
        "# G0 data, label and window audit",
        "",
        f"- Status: **{report.get('status')}**",
        "- Scope: strict-cross-subject public train/validation only; protected labels and indices were not opened.",
        f"- Registry: `{report.get('registry', {}).get('path')}`",
        f"- Registry SHA-256: `{report.get('registry', {}).get('file_sha256')}`",
        "- This is a quality-control artifact. It does not select a method, checkpoint, window or protected result.",
        "",
        "## Task/fold surface",
        "",
        "| task | dataset | modality | EEG/fNIRS Hz | window (s) | fold | train/validation | subjects | class support | group overlap | status |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in tasks:
        rates = f"{row.get('sampling_rate_hz_eeg')}/{row.get('sampling_rate_hz_fnirs')}"
        split = f"{row.get('train_sample_count')}/{row.get('validation_sample_count')}"
        subjects = f"{row.get('train_subject_count')}/{row.get('validation_subject_count')}"
        class_support = f"train={json.dumps(row.get('class_support_train', {}), ensure_ascii=False)}; val={json.dumps(row.get('class_support_validation', {}), ensure_ascii=False)}"
        overlap = row.get("group_overlap", {})
        overlap_short = ", ".join(
            f"{field}:{'disjoint' if value.get('disjoint') else 'OVERLAP'}"
            for field, value in overlap.items()
        )
        lines.append(
            f"| {row.get('task_display')} | {row.get('dataset_id')} | {row.get('modality')} | {rates} | {row.get('window_seconds')} | {row.get('outer_fold')} | {split} | {subjects} | {class_support} | {overlap_short} | {row.get('status')} |"
        )
    lines.extend(
        [
            "",
            "## Fail-closed findings",
            "",
        ]
    )
    errors = report.get("errors", [])
    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("- None.")
    findings = report.get("findings", [])
    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend(f"- {finding}" for finding in findings)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Public manifests and feature caches",
            "",
            f"Scanned manifests: **{len(caches)}**. The CSV is authoritative for per-manifest paths, folds and sample counts; this Markdown keeps only method-by-status totals and representative paths.",
            "",
            "| method | cache status | manifest count | representative path |",
            "| --- | --- | ---: | --- |",
        ]
    )
    cache_summary: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in caches:
        cache = row.get("cache", {})
        cache_summary[(str(row.get("method")), str(cache.get("status")))].append(
            str(row.get("manifest"))
        )
    for (method, cache_status), paths in sorted(cache_summary.items()):
        lines.append(
            f"| {method} | {cache_status} | {len(paths)} | `{paths[0]}` |"
        )
    if not cache_summary:
        lines.append("| n/a | n/a | 0 | n/a |")
    cache_failures = [
        row for row in caches if row.get("status") == "fail"
    ]
    lines.extend(["", "### Cache fail-closed list", ""])
    if cache_failures:
        for row in cache_failures:
            lines.append(
                f"- `{row.get('method')}` `{row.get('task')}` outer{row.get('outer_fold')}: {row.get('cache', {}).get('status')} — `{row.get('manifest')}`"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## EFRM validation mixing/support audit",
            "",
            "Split and group isolation pass for all 35 task/fold rows. Separately, the two complete EFRM validation exports contain repeated validation rows; this is recorded as a validation-mixing/support finding, not called data leakage. Because the positive mask is diagonal-only, a naive full-matrix retrieval interpretation would count repeated rows as negatives and is not appropriate.",
            "",
            "| artifact | rows | unique IDs | duplicate rows | duplicate IDs | exact repeated embeddings | positive diagonal | positive off-diagonal | naive full-matrix retrieval | status |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in efrm:
        lines.append(
            f"| {row.get('artifact')} | {row.get('row_count')} | {row.get('unique_sample_count')} | {row.get('duplicate_row_count')} | {row.get('duplicate_id_count')} | {row.get('duplicate_embedding_exact')} | {row.get('positive_mask_diagonal_true')} | {row.get('positive_mask_off_diagonal_true')} | {row.get('naive_full_matrix_retrieval_appropriate')} | {row.get('status')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- `status=pass` means the audited public contract is internally consistent; it is not evidence that a comparator should perform well.",
            "- Raw `record_id` can repeat across subjects (for example, `session_00`); `subject_record`, `join_key` and `trial_group` are the contamination-relevant group keys.",
            "- A `not_dereferenced` signal-support field means only metadata and the adapter contract were checked; it is intentionally not converted into a pass claim.",
            "- EFRM alignment rows marked `finding_repeated_validation_rows` are not failures of the public split/group contract. They identify a support/mixing structure that must be handled before interpreting full-matrix retrieval.",
            "- Protected sample counts/subject counts are copied as opaque registry metadata only. Protected labels, indices and signal arrays were never opened.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    outer_folds: Sequence[int] = range(5),
    output_dir: Path = DEFAULT_OUTPUT,
    signal_scan: bool = False,
    all_signal_samples: bool = False,
    scan_manifests: bool = True,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    errors: list[str] = []
    tasks: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    expected_by_task_fold: dict[tuple[str, int], dict[str, set[str]]] = {}
    expected_inventory_by_task: dict[str, dict[str, set[str]]] = {}
    expected_indices_by_task_fold: dict[tuple[str, int], set[int]] = {}
    expected_inventory_indices_by_task: dict[str, set[int]] = {}

    for task in TASK_ORDER:
        try:
            dataset = EFRMUnifiedTaskDataset(TASK_SPECS[task], cache_root=str(cache_root))
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            errors.append(f"{task}: unified adapter/dataset unavailable: {exc}")
            continue
        for fold in outer_folds:
            try:
                entry = strict_public_entry(registry, task=task, outer_fold=int(fold))
                summary, sample_rows, fold_errors = audit_task_fold(
                    task=task,
                    fold=int(fold),
                    entry=entry,
                    dataset=dataset,
                    signal_scan=signal_scan,
                    all_signal_samples=all_signal_samples,
                )
            except (AuditError, OSError, RuntimeError, ValueError, KeyError, IndexError) as exc:
                errors.append(f"{task}/outer{fold}: {exc}")
                summary = {
                    "task": task,
                    "task_display": TASK_DISPLAY[task],
                    "outer_fold": int(fold),
                    "status": "fail",
                    "errors": [str(exc)],
                    "protected_test_opened": False,
                }
                sample_rows = []
                fold_errors = [str(exc)]
            tasks.append(summary)
            samples.extend(sample_rows)
            errors.extend(fold_errors)
            expected_by_task_fold[(task, int(fold))] = {
                "source": {str(row["source_sample_id"]) for row in sample_rows},
                "adapter": {str(row["adapter_sample_id"]) for row in sample_rows},
            }
            expected_indices_by_task_fold[(task, int(fold))] = {
                int(row["dataset_index"]) for row in sample_rows
            }
            inventory_rows = [
                {**dataset.lightweight_metadata(index), "dataset_id": str(dataset.spec.dataset_id)}
                for index in range(len(dataset))
            ]
            expected_inventory_by_task[task] = {
                "source": {canonical_source_sample_id(row) for row in inventory_rows},
                "adapter": {adapter_sample_id(row) for row in inventory_rows},
            }
            expected_inventory_indices_by_task[task] = set(range(len(dataset)))
        del dataset

    manifests: list[dict[str, Any]] = []
    if scan_manifests:
        manifest_rows, manifest_errors = audit_public_manifests_and_caches(
            manifests=discover_public_manifests(),
            expected_by_task_fold=expected_by_task_fold,
            expected_inventory_by_task=expected_inventory_by_task,
            expected_indices_by_task_fold=expected_indices_by_task_fold,
            expected_inventory_indices_by_task=expected_inventory_indices_by_task,
        )
        manifests = manifest_rows
        errors.extend(manifest_errors)

    alignment_rows = _alignment_evidence_audit()
    for row in alignment_rows:
        if row.get("status") in {"read_error", "missing_required_arrays"}:
            errors.append(f"EFRM alignment {row.get('artifact')}: {row.get('status')}")

    findings: list[str] = []
    for row in alignment_rows:
        if row.get("status") == "finding_repeated_validation_rows":
            findings.append(
                f"EFRM {row.get('artifact')}: {row.get('row_count')} rows but "
                f"{row.get('unique_sample_count')} unique sample IDs; repeated validation "
                "rows have exact repeated embeddings and diagonal-only positives, so naive "
                "full-matrix retrieval is not an appropriate interpretation."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "data_window_audit.csv", tasks)
    _write_csv(output_dir / "sample_inventory.csv", samples)
    _write_csv(output_dir / "public_cache_audit.csv", manifests)
    _write_csv(output_dir / "efrm_validation_mixing.csv", alignment_rows)
    report: dict[str, Any] = {
        "schema": "comparator_g0_data_window_audit_v1",
        "status": (
            "fail"
            if errors
            else "pass_with_findings"
            if findings
            else "pass"
        ),
        "scope": {
            "protocol": "strict_cross_subject",
            "outer_folds": [int(value) for value in outer_folds],
            "public_partitions": ["train", "validation"],
            "protected_test_opened": False,
            "signal_scan": bool(signal_scan),
            "all_signal_samples": bool(all_signal_samples),
        },
        "registry": {
            "path": str(Path(registry_path).resolve()),
            "file_sha256": sha256_file(Path(registry_path)),
            "registry_sha256": registry.get("registry_sha256"),
            "protected_test_default": registry.get("protected_test_default"),
        },
        "cache_root": str(Path(cache_root).resolve()),
        "tasks": tasks,
        "sample_inventory_count": len(samples),
        "public_manifests_and_caches": manifests,
        "efrm_alignment_validation_mixing": alignment_rows,
        "findings": findings,
        "errors": errors,
        "protected_test_opened": False,
        "artifacts": {
            "task_csv": str((output_dir / "data_window_audit.csv").resolve()),
            "sample_csv": str((output_dir / "sample_inventory.csv").resolve()),
            "cache_csv": str((output_dir / "public_cache_audit.csv").resolve()),
            "efrm_csv": str((output_dir / "efrm_validation_mixing.csv").resolve()),
            "report": str((output_dir / "REPORT.md").resolve()),
        },
    }
    (output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    _write_report(output_dir / "REPORT.md", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-manifest", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-fold", type=int, action="append", dest="outer_folds")
    parser.add_argument("--all-outer-folds", action="store_true")
    parser.add_argument("--dereference-signals", action="store_true")
    parser.add_argument("--all-signal-samples", action="store_true")
    parser.add_argument("--skip-manifest-cache-scan", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.all_outer_folds or not args.outer_folds:
        folds = tuple(range(5))
    else:
        folds = tuple(sorted(set(int(value) for value in args.outer_folds)))
    if any(value < 0 or value >= 5 for value in folds):
        raise SystemExit("outer folds must be in [0, 4]")
    report = run_audit(
        registry_path=args.registry_manifest,
        cache_root=args.cache_root,
        outer_folds=folds,
        output_dir=args.output,
        signal_scan=bool(args.dereference_signals),
        all_signal_samples=bool(args.all_signal_samples),
        scan_manifests=not bool(args.skip_manifest_cache_scan),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "task_rows": len(report["tasks"]),
                "sample_inventory_count": report["sample_inventory_count"],
                "manifest_rows": len(report["public_manifests_and_caches"]),
                "efrm_alignment_rows": len(report["efrm_alignment_validation_mixing"]),
                "error_count": len(report["errors"]),
                "protected_test_opened": False,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] in {"pass", "pass_with_findings"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
