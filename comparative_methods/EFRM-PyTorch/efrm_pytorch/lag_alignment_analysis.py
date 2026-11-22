"""Public-only EFRM relative-crop alignment audit.

This module deliberately does *not* claim to measure a physiological lag.  The
``crop_start_s`` field in the saved EFRM evidence is an offset inside an
event/source window; it is not a record-absolute acquisition timestamp.  The
analysis therefore exposes a reproducible, embedding-space
``relative_crop_offset_proxy`` only.  A physical EEG→fNIRS lag is marked
unidentifiable until an export contains an event identity and absolute
modality-clock window starts.

The existing full-validation evidence contains the EEG×fNIRS cosine matrix, so
the proxy can be computed without reopening raw signals or a checkpoint.  All
inputs are required to be public validation artifacts.  Paths containing
``protected`` or manifests reporting protected access are rejected.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:  # package import in tests; direct-file import for the documented CLI
    from .visualization import EVIDENCE_SCHEMA
except ImportError:  # pragma: no cover - exercised only by direct execution
    _method_root = Path(__file__).resolve().parents[1]
    _repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_repo_root))
    sys.path.insert(0, str(_method_root))
    from efrm_pytorch.visualization import EVIDENCE_SCHEMA

from src.data.clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT


ANALYSIS_SCHEMA = "efrm_relative_crop_alignment_proxy_v1"
CAPABILITY_SCHEMA = "efrm_lag_alignment_capability_v1"
DEFAULT_LAG_GRID_S = (-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0)
DEFAULT_CROP_GRID_S = 0.1
DEFAULT_MATCH_TOLERANCE_S = 0.051
DEFAULT_NULL_PERMUTATIONS = 200
DEFAULT_SEED = 20260816
MIN_SUPPORTED_COVERAGE = 0.30
REQUIRED_METADATA_FIELDS = (
    "sample_id",
    "dataset_id",
    "subject",
    "record_id",
    "join_key",
    "task_namespace",
    "condition",
    "crop_start_s",
    "duration_s",
)
REEXPORT_METADATA_FIELDS = (
    *REQUIRED_METADATA_FIELDS,
    "event_index",
    "event_id",
    "eeg_time_ms",
    "fnirs_time_ms",
    "absolute_eeg_window_start_s",
    "absolute_fnirs_window_start_s",
    "absolute_eeg_window_end_s",
    "absolute_fnirs_window_end_s",
    "record_duration_s",
    "context_before_s",
    "context_after_s",
)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_public_path(path: Path) -> None:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"EFRM lag analysis refuses protected path: {resolved}")


def _find_run_manifest(evidence_path: Path) -> tuple[Path | None, dict[str, Any] | None]:
    """Find the nearest EFRM pretraining manifest and enforce public status."""

    for parent in (evidence_path.parent, *evidence_path.parents):
        candidate = parent / "manifest.json"
        if not candidate.is_file():
            continue
        try:
            manifest = _read_json(candidate)
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("schema") != "efrm_sync_pretraining_run_v1":
            continue
        _assert_public_path(candidate)
        if bool(manifest.get("protected_test_opened", False)):
            raise PermissionError(f"EFRM run manifest reports protected access: {candidate}")
        status_path = candidate.parent / "status.json"
        if status_path.is_file():
            status = _read_json(status_path)
            if bool(status.get("protected_test_opened", False)):
                raise PermissionError(f"EFRM status reports protected access: {status_path}")
        return candidate, manifest
    return None, None


def _normalise_metadata_value(value: Any) -> Any:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return str(value)


@dataclass(frozen=True)
class LoadedEvidence:
    path: Path
    cosine: np.ndarray
    eeg_embeddings: np.ndarray
    fnirs_embeddings: np.ndarray
    metadata: tuple[dict[str, Any], ...]
    source_indices: tuple[int, ...]
    duplicate_report: dict[str, Any]
    run_manifest_path: Path | None
    run_manifest: dict[str, Any] | None


def _deduplicate_evidence(
    path: Path,
    cosine: np.ndarray,
    eeg_embeddings: np.ndarray,
    fnirs_embeddings: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, Any], ...], tuple[int, ...], dict[str, Any]]:
    """Collapse sampler-repeat rows by stable sample_id.

    Full-validation exports are produced by an inventory sampler.  The
    sampler may cycle through the same validation tensor several times, so row
    count is not the number of unique windows.  Duplicate rows are accepted
    only when required metadata and embeddings agree; contradictory duplicate
    identities fail closed rather than silently averaging them.
    """

    if len(metadata) != cosine.shape[0]:
        raise ValueError(f"metadata count {len(metadata)} != cosine size {cosine.shape[0]} in {path}")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"metadata row {index} has no stable sample_id in {path}")
        groups[sample_id].append(index)

    keep: list[int] = []
    contradictory: list[str] = []
    max_embedding_delta = 0.0
    max_cosine_delta = 0.0
    for sample_id, indices in groups.items():
        first = indices[0]
        for index in indices[1:]:
            for field in REQUIRED_METADATA_FIELDS:
                left = _normalise_metadata_value(metadata[first].get(field))
                right = _normalise_metadata_value(metadata[index].get(field))
                if field == "crop_start_s":
                    if not np.isclose(float(left), float(right), atol=1e-7, rtol=0.0):
                        contradictory.append(f"{sample_id}:field={field}")
                elif left != right:
                    contradictory.append(f"{sample_id}:field={field}")
            max_embedding_delta = max(
                max_embedding_delta,
                float(np.max(np.abs(eeg_embeddings[index] - eeg_embeddings[first]))),
                float(np.max(np.abs(fnirs_embeddings[index] - fnirs_embeddings[first]))),
            )
            max_cosine_delta = max(
                max_cosine_delta,
                float(np.max(np.abs(cosine[index] - cosine[first]))),
                float(np.max(np.abs(cosine[:, index] - cosine[:, first]))),
            )
        keep.append(first)
    if contradictory:
        examples = ", ".join(contradictory[:5])
        raise ValueError(f"contradictory duplicate sample identities in {path}: {examples}")
    keep_array = np.asarray(keep, dtype=np.int64)
    dedup_metadata = tuple(dict(metadata[index]) for index in keep)
    report = {
        "rule": "keep_first_row_per_stable_sample_id_after_metadata_and_embedding_consistency_check_v1",
        "input_row_count": int(len(metadata)),
        "unique_sample_count": int(len(keep)),
        "duplicate_row_count_removed": int(len(metadata) - len(keep)),
        "duplicate_group_count": int(sum(len(values) > 1 for values in groups.values())),
        "maximum_duplicate_group_size": int(max(len(values) for values in groups.values())),
        "max_duplicate_embedding_abs_delta": max_embedding_delta,
        "max_duplicate_cosine_abs_delta": max_cosine_delta,
        "stable_key": "sample_id",
        "contradictory_duplicate_count": int(len(contradictory)),
    }
    return (
        cosine[np.ix_(keep_array, keep_array)],
        eeg_embeddings[keep_array],
        fnirs_embeddings[keep_array],
        dedup_metadata,
        tuple(int(value) for value in keep),
        report,
    )


def load_evidence(path: str | Path) -> LoadedEvidence:
    evidence_path = Path(path).resolve()
    _assert_public_path(evidence_path)
    if not evidence_path.is_file():
        raise FileNotFoundError(evidence_path)
    run_manifest_path, run_manifest = _find_run_manifest(evidence_path)
    with np.load(evidence_path, allow_pickle=False) as payload:
        keys = set(payload.files)
        required = {"schema", "cosine_similarity", "eeg_embeddings", "fnirs_embeddings", "metadata_json"}
        missing = sorted(required - keys)
        if missing:
            raise ValueError(f"EFRM evidence missing keys {missing}: {evidence_path}")
        schema = str(np.asarray(payload["schema"]).item())
        if schema != EVIDENCE_SCHEMA:
            raise ValueError(f"unsupported evidence schema {schema!r}: {evidence_path}")
        cosine = np.asarray(payload["cosine_similarity"], dtype=np.float64)
        eeg = np.asarray(payload["eeg_embeddings"], dtype=np.float64)
        fnirs = np.asarray(payload["fnirs_embeddings"], dtype=np.float64)
        metadata = tuple(json.loads(str(value)) for value in payload["metadata_json"].tolist())
    if cosine.ndim != 2 or cosine.shape[0] != cosine.shape[1]:
        raise ValueError(f"cosine_similarity must be square, got {cosine.shape}: {evidence_path}")
    if eeg.ndim != 2 or fnirs.ndim != 2 or eeg.shape[0] != cosine.shape[0] or fnirs.shape[0] != cosine.shape[0]:
        raise ValueError(f"embedding/cosine row counts disagree: {evidence_path}")
    missing_fields = sorted(set(REQUIRED_METADATA_FIELDS) - set().union(*(row.keys() for row in metadata)))
    if missing_fields:
        raise ValueError(f"EFRM evidence metadata lacks required fields {missing_fields}: {evidence_path}")
    dedup = _deduplicate_evidence(evidence_path, cosine, eeg, fnirs, metadata)
    return LoadedEvidence(
        path=evidence_path,
        cosine=dedup[0],
        eeg_embeddings=dedup[1],
        fnirs_embeddings=dedup[2],
        metadata=dedup[3],
        source_indices=dedup[4],
        duplicate_report=dedup[5],
        run_manifest_path=run_manifest_path,
        run_manifest=run_manifest,
    )


def audit_raw_metadata(
    metadata: Sequence[Mapping[str, Any]],
    *,
    cache_root: str | Path,
) -> dict[str, Any]:
    """Audit raw/event availability without using raw values for the proxy."""

    root = Path(cache_root).resolve()
    result: dict[str, Any] = {
        "cache_root": str(root),
        "cache_manifest_present": (root / "cache_manifest.json").is_file(),
        "event_manifest_present": (root / "event_index/event_manifest.json").is_file(),
        "events_jsonl_present": (root / "event_index/events.jsonl").is_file(),
        "requested_record_count": len({str(row["join_key"]) for row in metadata}),
        "requested_record_keys_found": 0,
        "event_rows_for_requested_records": 0,
        "event_timestamp_field_counts": {field: 0 for field in ("eeg_time_ms", "fnirs_time_ms", "onset_ms")},
        "event_identity_field_counts": {field: 0 for field in ("event_index", "event_id")},
        "record_absolute_context_proven": False,
        "evidence_contains_absolute_modality_timestamps": any(
            field in row for row in metadata for field in ("eeg_time_ms", "fnirs_time_ms", "absolute_eeg_window_start_s", "absolute_fnirs_window_start_s")
        ),
    }
    if not result["cache_manifest_present"]:
        result["status"] = "raw_cache_missing"
        return result
    try:
        cache_manifest = _read_json(root / "cache_manifest.json")
    except (OSError, json.JSONDecodeError) as error:
        result["status"] = "raw_cache_manifest_unreadable"
        result["error"] = str(error)
        return result
    wanted = {str(row["join_key"]) for row in metadata}
    found = set()
    for row in cache_manifest.get("records", []):
        if str(row.get("join_key", "")) in wanted:
            found.add(str(row["join_key"]))
    result["requested_record_keys_found"] = len(found)
    events_path = root / "event_index/events.jsonl"
    if events_path.is_file():
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("join_key", "")) not in wanted:
                    continue
                result["event_rows_for_requested_records"] += 1
                for field in result["event_timestamp_field_counts"]:
                    if row.get(field) is not None:
                        result["event_timestamp_field_counts"][field] += 1
                for field in result["event_identity_field_counts"]:
                    if row.get(field) is not None:
                        result["event_identity_field_counts"][field] += 1
    result["status"] = "raw_metadata_available_but_evidence_event_mapping_incomplete"
    result["physical_lag_identifiable_from_current_evidence"] = False
    result["reason"] = (
        "crop_start_s is event/source-window-relative; current evidence has no event identity or absolute "
        "modality-clock window starts, so a same-record crop difference is not a physical lag."
    )
    return result


def _group_indices(metadata: Sequence[Mapping[str, Any]], crop_grid_s: float) -> tuple[dict[str, dict[int, list[int]]], np.ndarray]:
    groups: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    ticks = np.empty(len(metadata), dtype=np.int64)
    for index, row in enumerate(metadata):
        start = float(row["crop_start_s"])
        if not np.isfinite(start) or start < -1e-7:
            raise ValueError(f"crop_start_s must be finite and non-negative at row {index}: {start}")
        tick = int(round(start / crop_grid_s))
        ticks[index] = tick
        groups[str(row["join_key"])][tick].append(index)
    return groups, ticks


def _pairs_for_lag(
    metadata: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Mapping[int, Sequence[int]]],
    *,
    lag_s: float,
    crop_grid_s: float,
    tolerance_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return EEG query rows and same-record fNIRS target rows for one grid lag."""

    if abs(lag_s / crop_grid_s - round(lag_s / crop_grid_s)) > 1e-6:
        raise ValueError(f"lag {lag_s} is not on crop grid {crop_grid_s}")
    lag_tick = int(round(lag_s / crop_grid_s))
    radius = max(1, int(np.ceil(tolerance_s / crop_grid_s)))
    rows: list[int] = []
    columns: list[int] = []
    errors: list[float] = []
    for row_index, row in enumerate(metadata):
        record = str(row["join_key"])
        start = float(row["crop_start_s"])
        target = start + lag_s
        tick = int(round(target / crop_grid_s))
        for candidate_tick in range(tick - radius, tick + radius + 1):
            for candidate in groups.get(record, {}).get(candidate_tick, ()):
                error = abs(float(metadata[candidate]["crop_start_s"]) - target)
                if error <= tolerance_s + 1e-9:
                    rows.append(row_index)
                    columns.append(int(candidate))
                    errors.append(error)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(columns, dtype=np.int64),
        np.asarray(errors, dtype=np.float64),
    )


def _score_pairs(cosine: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> dict[str, Any]:
    if rows.size == 0:
        return {
            "pair_count": 0,
            "unique_query_count": 0,
            "mean_cosine": None,
            "median_cosine": None,
            "std_cosine": None,
            "mean_query_cosine": None,
        }
    values = np.asarray(cosine[rows, columns], dtype=np.float64)
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return {
        "pair_count": int(values.size),
        "unique_query_count": int(unique_rows.size),
        "mean_cosine": float(values.mean()),
        "median_cosine": float(np.median(values)),
        "std_cosine": float(values.std()),
        "mean_query_cosine": float(np.mean(sums / np.maximum(counts, 1))),
    }


def _block_permutation_null(
    cosine: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    if rows.size == 0 or permutations <= 0:
        return {"permutations": int(permutations), "mean": None, "q025": None, "q975": None, "p_one_sided": None}
    grouped_lists: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        grouped_lists[str(row["join_key"])].append(index)
    grouped = {key: np.asarray(values, dtype=np.int64) for key, values in grouped_lists.items()}
    position: dict[int, int] = {}
    for values in grouped.values():
        for pos, index in enumerate(values.tolist()):
            position[int(index)] = pos
    pair_group = np.asarray([str(metadata[int(index)]["join_key"]) for index in columns], dtype=object)
    null_means = np.empty(permutations, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for permutation in range(permutations):
        targets = np.empty(columns.size, dtype=np.int64)
        for key, values in grouped.items():
            mask = pair_group == key
            if not np.any(mask):
                continue
            shuffled = values[rng.permutation(len(values))]
            target_positions = np.asarray([position[int(index)] for index in columns[mask]], dtype=np.int64)
            targets[mask] = shuffled[target_positions]
        null_means[permutation] = float(np.mean(cosine[rows, targets]))
    observed = float(np.mean(cosine[rows, columns]))
    return {
        "permutations": int(permutations),
        "seed": int(seed),
        "mean": float(null_means.mean()),
        "q025": float(np.quantile(null_means, 0.025)),
        "q975": float(np.quantile(null_means, 0.975)),
        "p_one_sided": float((1 + np.count_nonzero(null_means >= observed)) / (permutations + 1)),
        "observed_minus_null_mean": float(observed - null_means.mean()),
    }


def _coverage(metadata_count: int, rows: np.ndarray) -> dict[str, Any]:
    unique = int(np.unique(rows).size) if rows.size else 0
    return {
        "query_count": int(metadata_count),
        "matched_query_count": unique,
        "coverage_fraction": float(unique / metadata_count) if metadata_count else 0.0,
    }


def _profile_one_lag(
    evidence: LoadedEvidence,
    groups: Mapping[str, Mapping[int, Sequence[int]]],
    *,
    lag_s: float,
    crop_grid_s: float,
    tolerance_s: float,
    null_permutations: int,
    seed: int,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rows, columns, errors = _pairs_for_lag(
        evidence.metadata,
        groups,
        lag_s=lag_s,
        crop_grid_s=crop_grid_s,
        tolerance_s=tolerance_s,
    )
    all_score = _score_pairs(evidence.cosine, rows, columns)
    all_score.update(_coverage(len(evidence.metadata), rows))
    all_score["mean_abs_match_error_s"] = float(errors.mean()) if errors.size else None
    all_score["max_abs_match_error_s"] = float(errors.max()) if errors.size else None
    identity_mask = np.asarray(
        [str(evidence.metadata[int(row)]["sample_id"]) == str(evidence.metadata[int(column)]["sample_id"])
         for row, column in zip(rows.tolist(), columns.tolist(), strict=True)],
        dtype=bool,
    ) if rows.size else np.zeros(0, dtype=bool)
    non_identity = ~identity_mask
    identity_score = _score_pairs(evidence.cosine, rows[identity_mask], columns[identity_mask])
    non_identity_score = _score_pairs(evidence.cosine, rows[non_identity], columns[non_identity])
    all_score["identity_pair_count"] = identity_score["pair_count"]
    all_score["identity_mean_cosine"] = identity_score["mean_cosine"]
    all_score["non_identity_pair_count"] = non_identity_score["pair_count"]
    all_score["non_identity_mean_cosine"] = non_identity_score["mean_cosine"]
    null = _block_permutation_null(
        evidence.cosine,
        evidence.metadata,
        rows,
        columns,
        permutations=null_permutations,
        seed=seed + int(round((lag_s + 1000.0) * 100)),
    )
    same_condition = np.asarray(
        [str(evidence.metadata[int(row)]["condition"]) == str(evidence.metadata[int(column)]["condition"])
         for row, column in zip(rows.tolist(), columns.tolist(), strict=True)],
        dtype=bool,
    ) if rows.size else np.zeros(0, dtype=bool)
    same_score = _score_pairs(evidence.cosine, rows[same_condition], columns[same_condition])
    same_score.update(_coverage(len(evidence.metadata), rows[same_condition]))
    per_dataset: dict[str, Any] = {}
    datasets = sorted({str(row["dataset_id"]) for row in evidence.metadata})
    for dataset in datasets:
        dataset_mask = np.asarray(
            [str(evidence.metadata[int(row)]["dataset_id"]) == dataset for row in rows.tolist()],
            dtype=bool,
        ) if rows.size else np.zeros(0, dtype=bool)
        score = _score_pairs(evidence.cosine, rows[dataset_mask], columns[dataset_mask])
        score.update(_coverage(sum(str(row["dataset_id"]) == dataset for row in evidence.metadata), rows[dataset_mask]))
        score["mean_abs_match_error_s"] = float(errors[dataset_mask].mean()) if np.any(dataset_mask) else None
        per_dataset[dataset] = score
    profile = {
        "lag_s": float(lag_s),
        "direction": "fNIRS_relative_crop_start_minus_EEG_relative_crop_start",
        "interpretation_scope": "same_record_relative_crop_offset_proxy",
        "physical_lag_identifiable": False,
        "all_same_record": all_score,
        "same_record_same_condition": same_score,
        "same_record_block_permutation_null": null,
        "per_dataset": per_dataset,
        "supported_for_descriptive_profile": bool(all_score["coverage_fraction"] >= MIN_SUPPORTED_COVERAGE and all_score["pair_count"] > 0),
    }
    return profile, (rows, columns, errors)


def _save_csv(path: Path, profiles: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "lag_s",
        "pair_count",
        "matched_query_count",
        "coverage_fraction",
        "mean_abs_match_error_s",
        "max_abs_match_error_s",
        "mean_cosine",
        "median_cosine",
        "std_cosine",
        "mean_query_cosine",
        "identity_pair_count",
        "identity_mean_cosine",
        "non_identity_pair_count",
        "non_identity_mean_cosine",
        "same_condition_pair_count",
        "same_condition_mean_cosine",
        "null_mean",
        "null_q025",
        "null_q975",
        "null_p_one_sided",
        "supported_for_descriptive_profile",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for profile in profiles:
            all_score = profile["all_same_record"]
            same_score = profile["same_record_same_condition"]
            null = profile["same_record_block_permutation_null"]
            writer.writerow({
                "lag_s": profile["lag_s"],
                "pair_count": all_score["pair_count"],
                "matched_query_count": all_score["matched_query_count"],
                "coverage_fraction": all_score["coverage_fraction"],
                "mean_abs_match_error_s": all_score["mean_abs_match_error_s"],
                "max_abs_match_error_s": all_score["max_abs_match_error_s"],
                "mean_cosine": all_score["mean_cosine"],
                "median_cosine": all_score["median_cosine"],
                "std_cosine": all_score["std_cosine"],
                "mean_query_cosine": all_score["mean_query_cosine"],
                "identity_pair_count": all_score["identity_pair_count"],
                "identity_mean_cosine": all_score["identity_mean_cosine"],
                "non_identity_pair_count": all_score["non_identity_pair_count"],
                "non_identity_mean_cosine": all_score["non_identity_mean_cosine"],
                "same_condition_pair_count": same_score["pair_count"],
                "same_condition_mean_cosine": same_score["mean_cosine"],
                "null_mean": null["mean"],
                "null_q025": null["q025"],
                "null_q975": null["q975"],
                "null_p_one_sided": null["p_one_sided"],
                "supported_for_descriptive_profile": profile["supported_for_descriptive_profile"],
            })


def _render_plot(path_base: Path, profiles: Sequence[Mapping[str, Any]], *, title: str) -> None:
    if not profiles:
        return
    lags = np.asarray([float(profile["lag_s"]) for profile in profiles], dtype=np.float64)
    means = np.asarray([
        np.nan if profile["all_same_record"]["mean_cosine"] is None else profile["all_same_record"]["mean_cosine"]
        for profile in profiles
    ])
    null_low = np.asarray([
        np.nan if profile["same_record_block_permutation_null"]["q025"] is None else profile["same_record_block_permutation_null"]["q025"]
        for profile in profiles
    ])
    null_high = np.asarray([
        np.nan if profile["same_record_block_permutation_null"]["q975"] is None else profile["same_record_block_permutation_null"]["q975"]
        for profile in profiles
    ])
    same = np.asarray([
        np.nan if profile["same_record_same_condition"]["mean_cosine"] is None else profile["same_record_same_condition"]["mean_cosine"]
        for profile in profiles
    ])
    non_identity = np.asarray([
        np.nan if profile["all_same_record"]["non_identity_mean_cosine"] is None else profile["all_same_record"]["non_identity_mean_cosine"]
        for profile in profiles
    ])
    coverage = np.asarray([float(profile["all_same_record"]["coverage_fraction"]) for profile in profiles])
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True, constrained_layout=True)
    axis = axes[0]
    axis.fill_between(lags, null_low, null_high, color="#BDBDBD", alpha=0.35, label="same-record block-permutation 95% null")
    axis.plot(lags, means, marker="o", lw=1.8, color="#0072B2", label="all same-record matches")
    axis.plot(lags, same, marker="s", lw=1.3, color="#D55E00", label="same-record + same-condition")
    axis.plot(lags, non_identity, marker="^", lw=1.3, color="#CC79A7", label="same-record non-identity pairs")
    axis.axvline(0.0, color="#666666", ls="--", lw=0.9)
    axis.set_ylabel("embedding cosine")
    axis.set_title(title)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    axis = axes[1]
    axis.plot(lags, coverage, marker="o", color="#009E73")
    axis.axhline(MIN_SUPPORTED_COVERAGE, color="#888888", ls=":", lw=0.9, label=f"descriptive-support threshold={MIN_SUPPORTED_COVERAGE:.2f}")
    axis.set_xlabel("relative crop-offset difference, fNIRS minus EEG (s)")
    axis.set_ylabel("matched EEG-query coverage")
    axis.set_ylim(0.0, 1.05)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _short_run_label(run_id: str | None, fallback: str) -> str:
    aliases = {
        "eeg_fnirs_single_trial": "exclude-ST",
        "simultaneous_eeg_nirs": "exclude-Sim",
        "visual_cognitive_motivation": "exclude-Visual",
        "refed": "exclude-REFED",
    }
    value = str(run_id or fallback)
    if "__exclude_" in value:
        excluded = value.split("__exclude_", 1)[1].split("__stage_", 1)[0]
        return aliases.get(excluded, f"exclude-{excluded}")
    return value[:32]


def _raw_reexport_contract() -> dict[str, Any]:
    return {
        "status": "required_for_physical_lag_claims",
        "required_metadata_fields": list(REEXPORT_METADATA_FIELDS),
        "required_array_contract": {
            "schema": "efrm_clip_alignment_lag_export_v1",
            "eeg_embeddings": "[unique_sample, embedding_dim] at fixed EEG event/window identity",
            "fnirs_embeddings": "[unique_sample, lag_grid, embedding_dim] or one row per (sample,lag)",
            "lag_grid_s": list(DEFAULT_LAG_GRID_S),
            "sample_identity": "event_id plus absolute EEG/fNIRS window starts; sample_id alone is insufficient",
        },
        "physical_definition": "lag_s = absolute_fnirs_window_start_s - absolute_eeg_window_start_s",
        "current_producer_status": "evaluate_pretrain_checkpoint.py exports synchronized windows only; it does not export differential modality-clock windows",
        "command_interface_after_reexport": (
            ".venv/bin/python comparative_methods/EFRM-PyTorch/efrm_pytorch/lag_alignment_analysis.py "
            "--evidence <public_lag_export.npz> --output-root "
            "comparative_methods/runs/performance_analysis/20260816_p0/efrm_lag "
            "--lag-grid=-4,-2,0,2,4,6,8 --tolerance 0.051"
        ),
        "no_current_raw_reexport_command": True,
    }


def analyze_evidence(
    evidence: LoadedEvidence,
    output_dir: str | Path,
    *,
    lag_grid_s: Sequence[float] = DEFAULT_LAG_GRID_S,
    crop_grid_s: float = DEFAULT_CROP_GRID_S,
    tolerance_s: float = DEFAULT_MATCH_TOLERANCE_S,
    null_permutations: int = DEFAULT_NULL_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    raw_cache_root: str | Path = DEFAULT_CLEAN_CACHE_ROOT,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _assert_public_path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups, ticks = _group_indices(evidence.metadata, crop_grid_s)
    profiles: list[dict[str, Any]] = []
    pair_arrays: dict[str, np.ndarray] = {}
    for lag in lag_grid_s:
        profile, pairs = _profile_one_lag(
            evidence,
            groups,
            lag_s=float(lag),
            crop_grid_s=crop_grid_s,
            tolerance_s=tolerance_s,
            null_permutations=null_permutations,
            seed=seed,
        )
        profiles.append(profile)
        key = str(lag).replace("-", "neg").replace(".", "p")
        pair_arrays[f"rows_{key}"] = pairs[0]
        pair_arrays[f"columns_{key}"] = pairs[1]
        pair_arrays[f"errors_{key}"] = pairs[2]
    np.savez_compressed(output / "lag_pair_indices.npz", **pair_arrays)
    _save_csv(output / "lag_profile.csv", profiles)
    label = _short_run_label(evidence.run_manifest.get("run_id") if evidence.run_manifest else None, output.name)
    _render_plot(
        output / "relative_crop_offset_profile",
        profiles,
        title=f"{label}\nrelative crop-offset proxy; physical lag not identifiable",
    )
    raw_report = audit_raw_metadata(evidence.metadata, cache_root=raw_cache_root)
    unique_starts = sorted({float(row["crop_start_s"]) for row in evidence.metadata})
    start_diffs = np.diff(unique_starts)
    result = {
        "schema": ANALYSIS_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "surface": "public_validation_only",
        "protected_accessed": False,
        "physical_lag_identifiable": False,
        "interpretation": (
            "The profile compares precomputed embeddings for windows from the same record whose event-relative "
            "crop offsets differ by the requested grid. It is a relative_crop_offset_proxy, not a physiological "
            "EEG-to-fNIRS lag curve."
        ),
        "evidence": {
            "path": str(evidence.path),
            "sha256": _sha256(evidence.path),
            "run_manifest_path": str(evidence.run_manifest_path) if evidence.run_manifest_path else None,
            "run_id": evidence.run_manifest.get("run_id") if evidence.run_manifest else None,
            "input_row_count": evidence.duplicate_report["input_row_count"],
            "unique_sample_count": evidence.duplicate_report["unique_sample_count"],
            "unique_record_count": len({str(row["join_key"]) for row in evidence.metadata}),
            "unique_dataset_count": len({str(row["dataset_id"]) for row in evidence.metadata}),
            "sample_id_duplicate_audit": evidence.duplicate_report,
            "unique_crop_start_count": len(unique_starts),
            "crop_start_min_s": float(min(unique_starts)) if unique_starts else None,
            "crop_start_max_s": float(max(unique_starts)) if unique_starts else None,
            "crop_start_positive_step_median_s": float(np.median(start_diffs[start_diffs > 1e-7])) if np.any(start_diffs > 1e-7) else None,
            "crop_start_semantics": "event/source-window-relative; not record-absolute",
        },
        "matching_contract": {
            "lag_grid_s": [float(value) for value in lag_grid_s],
            "crop_grid_s": float(crop_grid_s),
            "tolerance_s": float(tolerance_s),
            "direction": "positive means fNIRS_relative_crop_start > EEG_relative_crop_start",
            "record_constraint": "same join_key",
            "duplicate_policy": evidence.duplicate_report["rule"],
            "minimum_descriptive_coverage": MIN_SUPPORTED_COVERAGE,
        },
        "raw_metadata_audit": raw_report,
        "profiles": profiles,
        "reexport_contract": _raw_reexport_contract(),
        "artifacts": {
            "lag_profile_csv": str(output / "lag_profile.csv"),
            "lag_pair_indices_npz": str(output / "lag_pair_indices.npz"),
            "relative_crop_offset_profile_svg": str(output / "relative_crop_offset_profile.svg"),
            "relative_crop_offset_profile_png": str(output / "relative_crop_offset_profile.png"),
            "relative_crop_offset_profile_pdf": str(output / "relative_crop_offset_profile.pdf"),
        },
    }
    (output / "analysis_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_jsonable), encoding="utf-8"
    )
    report_lines = [
        "# EFRM relative-crop alignment proxy",
        "",
        "**Public validation only. Protected artifacts were not opened.**",
        "",
        "This is not a physical EEG-to-fNIRS lag analysis. `crop_start_s` is an offset inside an event/source window; the current evidence has no event identity or absolute modality-clock window start. The values below are a same-record relative-crop-offset proxy.",
        "",
        f"- Input rows: {evidence.duplicate_report['input_row_count']}",
        f"- Unique stable samples after deduplication: {evidence.duplicate_report['unique_sample_count']}",
        f"- Duplicate rows removed: {evidence.duplicate_report['duplicate_row_count_removed']}",
        f"- Matching tolerance: {tolerance_s:.3f} s",
        f"- Predefined grid: {', '.join(f'{float(value):g}' for value in lag_grid_s)} s",
        "",
        "| Relative crop offset (fNIRS − EEG) | Pairs | Query coverage | Mean cosine | Non-identity mean | Same-condition mean | Null 95% interval |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in profiles:
        score = profile["all_same_record"]
        same = profile["same_record_same_condition"]
        null = profile["same_record_block_permutation_null"]
        interval = "n/a" if null["q025"] is None else f"[{null['q025']:.4f}, {null['q975']:.4f}]"
        report_lines.append(
            f"| {profile['lag_s']:g} | {score['pair_count']} | {score['coverage_fraction']:.3f} | "
            f"{score['mean_cosine'] if score['mean_cosine'] is not None else float('nan'):.4f} | "
            f"{score['non_identity_mean_cosine'] if score['non_identity_mean_cosine'] is not None else float('nan'):.4f} | "
            f"{same['mean_cosine'] if same['mean_cosine'] is not None else float('nan'):.4f} | {interval} |"
        )
    report_lines += [
        "",
        "## Physical-lag status",
        "",
        "`physical_lag_identifiable=false`. A physical analysis requires event identity plus absolute EEG/fNIRS modality-clock starts and enough pre/post context. The exact re-export contract is recorded in `analysis_metrics.json`.",
        "",
        "## Artifacts",
        "",
        "- `lag_profile.csv`",
        "- `lag_pair_indices.npz`",
        "- `relative_crop_offset_profile.svg` / `.png` / `.pdf`",
        "- `analysis_metrics.json`",
    ]
    (output / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (output / "capability_report.json").write_text(
        json.dumps({
            "schema": CAPABILITY_SCHEMA,
            "status": "proxy_ready_physical_lag_not_identifiable",
            "surface": "public_validation_only",
            "protected_accessed": False,
            "physical_lag_identifiable": False,
            "relative_crop_offset_proxy_available": True,
            "evidence_path": str(evidence.path),
            "raw_metadata_audit": raw_report,
            "reexport_contract": _raw_reexport_contract(),
            "analysis_metrics": str(output / "analysis_metrics.json"),
        }, indent=2, sort_keys=True, default=_jsonable),
        encoding="utf-8",
    )
    return result


def _parse_lag_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid lag grid: {value!r}") from error
    if not values:
        raise argparse.ArgumentTypeError("lag grid cannot be empty")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("lag grid contains duplicates")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", action="append", required=True, help="public full-validation evidence .npz; repeat for multiple runs")
    parser.add_argument("--output-root", default="comparative_methods/runs/performance_analysis/20260816_p0/efrm_lag")
    parser.add_argument("--raw-cache-root", default=DEFAULT_CLEAN_CACHE_ROOT)
    parser.add_argument("--lag-grid", type=_parse_lag_grid, default=DEFAULT_LAG_GRID_S)
    parser.add_argument("--crop-grid-s", type=float, default=DEFAULT_CROP_GRID_S)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_MATCH_TOLERANCE_S)
    parser.add_argument("--null-permutations", type=int, default=DEFAULT_NULL_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    output_root = Path(args.output_root).resolve()
    _assert_public_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for raw_path in args.evidence:
        evidence_path = Path(raw_path).resolve()
        label = evidence_path.parent.parent.name
        destination = output_root / label
        try:
            loaded = load_evidence(evidence_path)
            reports.append(analyze_evidence(
                loaded,
                destination,
                lag_grid_s=args.lag_grid,
                crop_grid_s=float(args.crop_grid_s),
                tolerance_s=float(args.tolerance),
                null_permutations=int(args.null_permutations),
                seed=int(args.seed),
                raw_cache_root=args.raw_cache_root,
            ))
        except Exception as error:  # noqa: BLE001 - fail-closed report includes exact error
            failure = {
                "schema": ANALYSIS_SCHEMA,
                "status": "fail_closed",
                "surface": "public_validation_only",
                "protected_accessed": False,
                "evidence_path": str(evidence_path),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "capability_report.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
            failures.append(failure)
    capability = {
        "schema": CAPABILITY_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "proxy_ready_physical_lag_not_identifiable" if reports and not failures else "fail_closed",
        "surface": "public_validation_only",
        "protected_accessed": False,
        "analysis_count": len(reports),
        "failure_count": len(failures),
        "physical_lag_identifiable": False,
        "relative_crop_offset_proxy_available": bool(reports),
        "reports": [
            {
                "run_id": row.get("evidence", {}).get("run_id"),
                "evidence_path": row.get("evidence", {}).get("path"),
                "analysis_metrics": row.get("artifacts", {}).get("lag_profile_csv"),
            }
            for row in reports
        ],
        "failures": failures,
        "reexport_contract": _raw_reexport_contract(),
    }
    (output_root / "capability_report.json").write_text(
        json.dumps(capability, indent=2, sort_keys=True, default=_jsonable), encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        "# EFRM lag-alignment capability\n\n"
        "This directory contains public-only relative-crop-offset proxy analyses. `crop_start_s` is not an absolute acquisition timestamp; physical EEG→fNIRS lag remains unidentifiable until the re-export contract in `capability_report.json` is satisfied.\n",
        encoding="utf-8",
    )
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
