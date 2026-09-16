#!/usr/bin/env python3
"""Build a unified event, task-label, and timing-alignment index."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.event_alignment import (  # noqa: E402
    EVENT_ALIGNMENT_SCHEMA,
    CanonicalEvent,
    EventAlignmentReport,
    align_paired_marker_streams,
    normalize_marker_struct,
    read_xlsx_rows,
    read_visual_fnirs_csv,
    visual_stimulus_onsets_from_dc9,
)
from src.data.clean_physiology_cache import with_canonical_fields  # noqa: E402
from src.utils.io import write_json  # noqa: E402


EVENT_INDEX_SCHEMA = "clean_eeg_fnirs_event_index_v2"

DATA_ROOTS = {
    "eeg_fnirs_single_trial": PROJECT_ROOT / "data/EEG+NIRS Single-Trial",
    "refed": PROJECT_ROOT / "data/REFED-dataset",
    "visual_cognitive_motivation": PROJECT_ROOT / "data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults",
    "simultaneous_eeg_nirs": PROJECT_ROOT / "data/Simultaneous EEG&NIRS",
}

SINGLE_TRIAL_TASK_BY_SESSION = {
    0: "motor_imagery",
    1: "mental_arithmetic",
    2: "motor_imagery",
    3: "mental_arithmetic",
    4: "motor_imagery",
    5: "mental_arithmetic",
}

SINGLE_TRIAL_LABELS_BY_TASK = {
    "motor_imagery": ("LMI", "RMI"),
    "mental_arithmetic": ("MA", "BL"),
}

VISUAL_MARK_LABELS = {
    1: "stimulus_onset",
    2: "stimulus_offset",
    3: "participant_response",
}
VISUAL_VALID_EPOCH_TYPES = {"RR", "RF", "FF", "FR"}
VISUAL_EPOCH_TYPE_INDICES = {"RR": 0, "RF": 1, "FF": 2, "FR": 3, "unknown": -1}
VISUAL_TIMING_CONTRACT = {
    "schema": "visual_dc9_stimulus_timing_v2",
    "eeg_trigger": "DC9",
    "eeg_stimulus_onset_rule": "dc9_followed_by_stimulus_offset_at_3000ms",
    "stimulus_duration_ms": 3_000.0,
    "tolerance_ms": 10.0,
    "fnirs_stimulus_onset_mark": 1,
    "duplicate_eeg_timestamps": "deduplicate_before_semantic_pair_detection",
    "source": "dataset_readme_and_data_in_brief_2024_110260",
}
REFED_CONTINUOUS_TIMING_CONTRACT = {
    "schema": "refed_continuous_annotation_timing_v2",
    "targets": ["valence", "arousal"],
    "released_layout": "time_by_target",
    "native_grid": "approximately_1_hz",
    "time_basis": "event_relative_published_1_hz",
    "signal_support": "intersection_of_eeg_fnirs_and_annotation_support",
    "value_coordinate": "refed_joystick_native",
    "scaling_policy": "fit_on_train_subjects_only",
    "source": "data/REFED-dataset/README.md_and_annotations/*_label.mat",
}

SIMULTANEOUS_SESSION_CODEBOOKS = {
    "nback": {
        "eeg": {112: "0-back session", 128: "2-back session", 144: "3-back session"},
        "fnirs": {7: "0-back session", 8: "2-back session", 9: "3-back session"},
    },
    "dsr": {
        "eeg": {48: "session"},
        "fnirs": {3: "session"},
    },
}
SIMULTANEOUS_DSR_STIMULUS_CODEBOOK = {
    16: ("Go", 0),
    32: ("No-go", 1),
}
SIMULTANEOUS_DSR_TIMING_CONTRACT = {
    "schema": "simultaneous_dsr_go_nogo_timing_v2",
    "eeg_stimulus_codes": {"16": "Go", "32": "No-go"},
    "eeg_block_code": 48,
    "fnirs_block_code": 3,
    "stimulus_display_duration_ms": 500.0,
    "stimulus_cycle_duration_ms": 2_000.0,
    "recommended_eeg_epoch_duration_ms": 2_000.0,
    "fnirs_role": "synchronized_hemodynamic_context_not_symbol_native_marker",
    "source": "Shin_et_al_Scientific_Data_2018_and_released_marker_streams",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATA_ROOTS), choices=list(DATA_ROOTS))
    parser.add_argument("--subjects-per-dataset", type=int, default=1000)
    parser.add_argument("--records-per-subject", type=int, default=1000)
    from src.data.clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT
    parser.add_argument("--output-dir", default=f"{DEFAULT_CLEAN_CACHE_ROOT}/event_index")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _mat_payload(path: Path, key: str | None = None) -> Any:
    payload = loadmat(path, struct_as_record=False, squeeze_me=True)
    if key is None:
        key = next(name for name in payload if not name.startswith("__"))
    value = payload[key]
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        value = value.item()
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path)


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def _class_names(value: Any) -> list[str]:
    return [str(item) for item in np.asarray(value, dtype=object).ravel().tolist()]


def _single_trial_subjects(root: Path, limit: int) -> list[int]:
    subjects = [p for p in sorted((root / "EEG_01-29").glob("subject *"))
                if int(p.name.split()[-1]) < 24][:limit]
    return [int(subject.name.split()[-1]) for subject in subjects]


def iter_single_trial(root: Path, subject_limit: int, record_limit: int, *,
                      subject_ids: Iterable[int] | None = None, session_ids: Iterable[int] | None = None) -> tuple[list[CanonicalEvent], list[EventAlignmentReport]]:
    events: list[CanonicalEvent] = []
    reports: list[EventAlignmentReport] = []
    subjects = list(subject_ids) if subject_ids is not None else _single_trial_subjects(root, subject_limit)
    if any(s not in range(1,24) for s in subjects):
        raise ValueError('Single-Trial protected subject boundary before file access')
    for subject_id in subjects:
        eeg_dir = root / "EEG_01-29" / f"subject {subject_id:02d}" / "with occular artifact"
        if not eeg_dir.exists():
            eeg_dir = root / "EEG_01-29" / f"subject {subject_id:02d}"
        nirs_dir = root / "NIRS_01-29" / f"subject {subject_id:02d}"
        eeg_mrk = np.atleast_1d(_mat_payload(eeg_dir / "mrk.mat", "mrk"))
        nirs_mrk = np.atleast_1d(_mat_payload(nirs_dir / "mrk.mat", "mrk"))
        for session_idx in range(min(len(eeg_mrk), len(nirs_mrk), record_limit)):
            if session_ids is not None and session_idx not in session_ids:
                continue
            record_id = f"session_{session_idx:02d}"
            task = SINGLE_TRIAL_TASK_BY_SESSION.get(session_idx, "unknown")
            eeg_marker = normalize_marker_struct(eeg_mrk[session_idx])
            nirs_marker = normalize_marker_struct(nirs_mrk[session_idx])
            if task in SINGLE_TRIAL_LABELS_BY_TASK:
                labels = list(SINGLE_TRIAL_LABELS_BY_TASK[task])
                eeg_marker["className"] = labels
                nirs_marker["className"] = labels
            session_events, report = align_paired_marker_streams(
                dataset_id="eeg_fnirs_single_trial",
                subject=f"subject {subject_id:02d}",
                record_id=record_id,
                eeg_marker=eeg_marker,
                fnirs_marker=nirs_marker,
                event_type="trial",
            )
            events.extend(
                CanonicalEvent(
                    **{
                        **event.to_dict(),
                        "metadata": {
                            **dict(event.metadata),
                            "task": task,
                            "session_idx": session_idx,
                            "source_files": [_rel(eeg_dir / "mrk.mat"), _rel(nirs_dir / "mrk.mat")],
                        },
                    }
                )
                for event in session_events
            )
            reports.append(report)
    return events, reports


def _sim_subjects(root: Path, limit: int) -> list[int]:
    subjects = sorted(root.glob("VP*-EEG"))[:limit]
    return [int(subject.name.split("-")[0].replace("VP", "")) for subject in subjects]


def _simultaneous_marker_for_alignment(marker_struct: Any, task: str, modality: str) -> dict[str, Any]:
    marker = normalize_marker_struct(marker_struct)
    codebook = SIMULTANEOUS_SESSION_CODEBOOKS.get(task, {}).get(modality)
    if not codebook:
        return marker
    event_desc = marker.get("event_desc")
    if event_desc is None:
        return marker
    desc = np.asarray(event_desc, dtype=np.int64).reshape(-1)
    mask = np.isin(desc, list(codebook))
    times = np.asarray(marker["time"], dtype=np.float64)[mask]
    labels = [codebook[int(value)] for value in desc[mask].tolist()]
    class_names = list(dict.fromkeys(labels))
    y = np.zeros((len(class_names), len(labels)), dtype=np.float32)
    lookup = {label: index for index, label in enumerate(class_names)}
    for event_index, label in enumerate(labels):
        y[lookup[label], event_index] = 1.0
    return {
        "time": times,
        "y": y,
        "className": class_names,
        "event_desc": desc[mask],
    }


def _simultaneous_dsr_events(
    *,
    subject: str,
    record_id: str,
    eeg_marker_struct: Any,
    fnirs_marker_struct: Any,
    source_files: list[str],
) -> tuple[list[CanonicalEvent], EventAlignmentReport]:
    """Project EEG-native Go/No-go markers onto the synchronized fNIRS clock.

    The released fNIRS marker stream is block-level only.  Therefore each
    symbol uses the offset of its own aligned block anchor; a block whose
    anchor was skipped during alignment contributes no symbol events.
    """
    eeg_raw = normalize_marker_struct(eeg_marker_struct)
    eeg_blocks = _simultaneous_marker_for_alignment(eeg_marker_struct, "dsr", "eeg")
    fnirs_blocks = _simultaneous_marker_for_alignment(fnirs_marker_struct, "dsr", "fnirs")
    anchors, base_report = align_paired_marker_streams(
        dataset_id="simultaneous_eeg_nirs",
        subject=subject,
        record_id=record_id,
        eeg_marker=eeg_blocks,
        fnirs_marker=fnirs_blocks,
        event_type="session_block",
    )
    all_times = np.asarray(eeg_raw.get("time", []), dtype=np.float64).reshape(-1)
    all_desc = np.asarray(eeg_raw.get("event_desc", []), dtype=np.int64).reshape(-1)
    block_times = np.asarray(eeg_blocks.get("time", []), dtype=np.float64).reshape(-1)
    output: list[CanonicalEvent] = []
    per_block_counts: dict[str, int] = {}
    for anchor in anchors:
        block_index = int(np.argmin(np.abs(block_times - float(anchor.eeg_time_ms))))
        block_start = float(block_times[block_index])
        block_stop = (
            float(block_times[block_index + 1])
            if block_index + 1 < len(block_times)
            else float("inf")
        )
        selected = (
            (all_times >= block_start)
            & (all_times < block_stop)
            & np.isin(all_desc, list(SIMULTANEOUS_DSR_STIMULUS_CODEBOOK))
        )
        indices = np.flatnonzero(selected)
        offset_ms = float(anchor.fnirs_time_ms) - float(anchor.eeg_time_ms)
        per_block_counts[str(block_index)] = int(len(indices))
        for source_index in indices:
            code = int(all_desc[source_index])
            label, label_index = SIMULTANEOUS_DSR_STIMULUS_CODEBOOK[code]
            eeg_time_ms = float(all_times[source_index])
            fnirs_time_ms = eeg_time_ms + offset_ms
            output.append(CanonicalEvent(
                dataset_id="simultaneous_eeg_nirs",
                subject=subject,
                record_id=record_id,
                event_index=len(output),
                event_type="stimulus",
                label=label,
                label_index=label_index,
                eeg_time_ms=eeg_time_ms,
                fnirs_time_ms=fnirs_time_ms,
                onset_ms=fnirs_time_ms,
                duration_ms=SIMULTANEOUS_DSR_TIMING_CONTRACT["stimulus_display_duration_ms"],
                alignment_role="eeg_stimulus_projected_from_aligned_block_anchor",
                metadata={
                    "task": "dsr",
                    "event_role": "stimulus_onset",
                    "condition_label": label,
                    "eeg_marker_code": code,
                    "source_event_index": int(source_index),
                    "block_index": block_index,
                    "block_anchor_offset_ms": offset_ms,
                    "alignment_support_ms": anchor.metadata["alignment_support_ms"],
                    "timing_contract": SIMULTANEOUS_DSR_TIMING_CONTRACT,
                    "source_files": source_files,
                },
            ))
    report = EventAlignmentReport(**{
        **base_report.to_dict(),
        "metadata": {
            **dict(base_report.metadata),
            "dsr_contract": SIMULTANEOUS_DSR_TIMING_CONTRACT,
            "released_eeg_stimulus_count": int(np.count_nonzero(
                np.isin(all_desc, list(SIMULTANEOUS_DSR_STIMULUS_CODEBOOK))
            )),
            "projected_stimulus_count": len(output),
            "projected_stimulus_count_by_eeg_block": per_block_counts,
            "source_files": source_files,
        },
    })
    return output, report


def iter_simultaneous(root: Path, subject_limit: int, record_limit: int) -> tuple[list[CanonicalEvent], list[EventAlignmentReport]]:
    events: list[CanonicalEvent] = []
    reports: list[EventAlignmentReport] = []
    # Match the signal builder's lexicographic cnt_*.mat selection, including
    # bounded smoke runs; a different prefix otherwise creates orphan joins.
    tasks = tuple(sorted(("nback", "dsr", "wg")))[:record_limit]
    for subject_id in _sim_subjects(root, subject_limit):
        for task in tasks:
            eeg_path = root / f"VP{subject_id:03d}-EEG" / f"mrk_{task}.mat"
            fnirs_path = root / f"VP{subject_id:03d}-NIRS" / f"mrk_{task}.mat"
            if not eeg_path.exists() or not fnirs_path.exists():
                continue
            source_files = [_rel(eeg_path), _rel(fnirs_path)]
            eeg_marker_struct = _mat_payload(eeg_path)
            fnirs_marker_struct = _mat_payload(fnirs_path)
            if task == "dsr":
                session_events, report = _simultaneous_dsr_events(
                    subject=f"VP{subject_id:03d}",
                    record_id=f"cnt_{task}",
                    eeg_marker_struct=eeg_marker_struct,
                    fnirs_marker_struct=fnirs_marker_struct,
                    source_files=source_files,
                )
            else:
                session_events, report = align_paired_marker_streams(
                    dataset_id="simultaneous_eeg_nirs",
                    subject=f"VP{subject_id:03d}",
                    record_id=f"cnt_{task}",
                    eeg_marker=_simultaneous_marker_for_alignment(eeg_marker_struct, task, "eeg"),
                    fnirs_marker=_simultaneous_marker_for_alignment(fnirs_marker_struct, task, "fnirs"),
                    event_type="trial" if task == "wg" else "session_block",
                )
            events.extend(
                CanonicalEvent(
                    **{
                        **event.to_dict(),
                        "metadata": {
                            **dict(event.metadata),
                            "task": task,
                            "source_files": source_files,
                        },
                    }
                )
                for event in session_events
            )
            reports.append(report)
    return events, reports


def _read_csv_dict(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _refed_video_info(root: Path) -> dict[int, dict[str, str]]:
    rows = _read_csv_dict(root / "Video_info.csv")
    return {int(row["video_id"]): {str(k).strip(): str(v).strip() for k, v in row.items()} for row in rows}


def _refed_sam(root: Path) -> dict[int, dict[int, dict[str, int]]]:
    output: dict[int, dict[int, dict[str, int]]] = {}
    for row in _read_csv_dict(root / "SAM_score.csv"):
        subject = int(row["sub_id"])
        output[subject] = {}
        for video in range(1, 16):
            output[subject][video] = {
                "valence": int(row.get(f"Video_{video}_Valence", 0) or 0),
                "arousal": int(row.get(f"Video_{video}_Arousal", 0) or 0),
                "dominance": int(row.get(f"Video_{video}_Dominance", 0) or 0),
                "familiarity": int(row.get(f"Video_{video}_Familiarity", 0) or 0),
            }
    return output


def iter_refed(root: Path, subject_limit: int, record_limit: int) -> tuple[list[CanonicalEvent], list[EventAlignmentReport]]:
    events: list[CanonicalEvent] = []
    reports: list[EventAlignmentReport] = []
    video_info = _refed_video_info(root)
    sam = _refed_sam(root)
    subjects = sorted((root / "data").glob("[0-9]*"), key=lambda item: int(item.name))[:subject_limit]
    for subject_dir in subjects:
        subject_id = int(subject_dir.name)
        eeg_payload = loadmat(subject_dir / "EEG_videos.mat", variable_names=[f"video_{i}" for i in range(1, record_limit + 1)])
        fnirs_payload = loadmat(subject_dir / "fNIRS_videos.mat", variable_names=[f"video_{i}" for i in range(1, record_limit + 1)])
        label_payload = loadmat(root / "annotations" / f"{subject_id}_label.mat", variable_names=[f"video_{i}" for i in range(1, record_limit + 1)])
        for video in range(1, record_limit + 1):
            key = f"video_{video}"
            if key not in fnirs_payload:
                continue
            eeg_len = int(np.asarray(eeg_payload[key]).shape[-1]) if key in eeg_payload else None
            fnirs_len = int(np.asarray(fnirs_payload[key]).shape[-1])
            labels = np.asarray(label_payload.get(key, np.empty((0, 2))), dtype=np.float32)
            if labels.ndim != 2 or 2 not in labels.shape:
                raise ValueError(f"Invalid REFED label shape for {subject_id}/{key}: {labels.shape}")
            if labels.shape[1] != 2:
                labels = labels.T
            duration_ms = float(fnirs_len / 47.62 * 1000.0)
            label_sample_count = int(labels.shape[0]) if labels.ndim == 2 else 0
            metadata = {
                "task": "emotion_video",
                "video_id": video,
                "video_info": video_info.get(video, {}),
                "sam_score": sam.get(subject_id, {}).get(video, {}),
                "continuous_label_stream": {
                    "names": ["valence", "arousal"],
                    "values": labels.tolist(),
                    "sample_count": label_sample_count,
                    "layout": "time_by_target",
                    "native_sample_rate_hz": 1.0,
                    "time_basis": "event_relative_published_1_hz",
                    "value_coordinate": "refed_joystick_native",
                    "sampling_note": (
                        "released REFED annotation grid is approximately 1 Hz; "
                        "preserve the published rate; first-sample phase is assumed at video zero"
                    ),
                },
                "eeg_samples": eeg_len,
                "fnirs_samples": fnirs_len,
                "clock_evidence": "publisher_segment_origin_assumed_not_measured",
                "annotation_first_sample_time_s_assumed": 0.0,
                "source_files": [
                    _rel(subject_dir / "EEG_videos.mat"),
                    _rel(subject_dir / "fNIRS_videos.mat"),
                    _rel(root / "annotations" / f"{subject_id}_label.mat"),
                ],
            }
            events.append(
                CanonicalEvent(
                    dataset_id="refed",
                    subject=str(subject_id),
                    record_id=key,
                    event_index=video - 1,
                    event_type="video_segment_with_continuous_labels",
                    label=video_info.get(video, {}).get("TargetedEmotion", ""),
                    label_index=video - 1,
                    eeg_time_ms=0.0,
                    fnirs_time_ms=0.0,
                    onset_ms=0.0,
                    duration_ms=duration_ms,
                    alignment_role="shared_video_segment_index",
                    metadata=metadata,
                )
            )
            reports.append(
                EventAlignmentReport(
                    dataset_id="refed",
                    subject=str(subject_id),
                    record_id=key,
                    num_eeg_events=1 if eeg_len else 0,
                    num_fnirs_events=1,
                    num_aligned_events=1 if eeg_len else 0,
                    alignment_case=("shared_segment_index_no_marker_stream"
                                    if eeg_len and abs(eeg_len / 1000.0 - duration_ms / 1000.0) <= 1.0 / 47.62
                                    else "segment_duration_mismatch"),
                    label_sequence_match=True,
                    offset_mean_ms=None,
                    offset_std_ms=None,
                    drift_slope_ms_per_min=None,
                    metadata={"eeg_samples": eeg_len, "fnirs_samples": fnirs_len, "duration_ms": duration_ms,
                              "clock_evidence": "publisher_segment_origin_assumed_not_measured",
                              "assumed_offset_ms": 0.0},
                )
            )
    return events, reports


def _read_visual_marks(path: Path) -> tuple[list[dict[str, Any]], float]:
    recording = read_visual_fnirs_csv(path)
    return recording["marks"], recording["sample_rate_hz"]


def _visual_type_map(subject_dir: Path) -> dict[int, str]:
    path = subject_dir / f"{subject_dir.name}_type.xlsx"
    if not path.exists():
        return {}
    mapping = {}
    for row in read_xlsx_rows(str(path)):
        try:
            epoch_id = int(float(row.get("Epoch_ID", "")))
        except ValueError:
            continue
        mapping[epoch_id] = str(row.get("Type", ""))
    return mapping


def _read_visual_eeg_onsets(path: Path) -> tuple[list[float], dict[str, Any]]:
    """Read Visual DC9 annotations and identify documented stimulus onsets."""
    onsets_s: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            onsets_s.append(float(parts[1]))
        except ValueError:
            continue
    onsets_ms, diagnostics = visual_stimulus_onsets_from_dc9(
        [float(value * 1000.0) for value in onsets_s],
        stimulus_duration_ms=VISUAL_TIMING_CONTRACT["stimulus_duration_ms"],
        tolerance_ms=VISUAL_TIMING_CONTRACT["tolerance_ms"],
    )
    return onsets_ms.astype(float).tolist(), diagnostics


def _visual_annotation_path(subject_dir: Path, record_id: str) -> Path | None:
    raw_dir = subject_dir / "EEG" / "raw"
    part = re.search(r"Part(\d+)", record_id, flags=re.IGNORECASE)
    if part:
        candidates = sorted(raw_dir.glob(f"*part{part.group(1)}_annotations.txt"))
    else:
        candidates = sorted(raw_dir.glob(f"{subject_dir.name}_annotations.txt"))
    if not candidates:
        candidates = sorted(raw_dir.glob("*_annotations.txt"))
    return candidates[0] if candidates else None


def iter_visual(root: Path, subject_limit: int, record_limit: int) -> tuple[list[CanonicalEvent], list[EventAlignmentReport]]:
    events: list[CanonicalEvent] = []
    reports: list[EventAlignmentReport] = []
    subjects = sorted(path for path in root.glob("S[0-9][0-9]") if (path / "fNIRS").exists())[:subject_limit]
    for subject_dir in subjects:
        type_map = _visual_type_map(subject_dir)
        for oxy_path in sorted((subject_dir / "fNIRS").glob("*Oxy.csv"))[:record_limit]:
            deoxy_path = Path(str(oxy_path).replace("_Oxy.csv", "_Deoxy.csv"))
            if not deoxy_path.exists():
                continue
            record_id = oxy_path.stem.replace("_Oxy", "")
            marks, sample_rate = _read_visual_marks(oxy_path)
            stimulus_marks = [item for item in marks if item["mark"] == 1]
            annotation_path = _visual_annotation_path(subject_dir, record_id)
            if annotation_path is None:
                reports.append(
                    EventAlignmentReport(
                        dataset_id="visual_cognitive_motivation",
                        subject=subject_dir.name,
                        record_id=record_id,
                        num_eeg_events=0,
                        num_fnirs_events=len(stimulus_marks),
                        num_aligned_events=0,
                        alignment_case="missing_eeg_annotation_sidecar",
                        label_sequence_match=None,
                        offset_mean_ms=None,
                        offset_std_ms=None,
                        drift_slope_ms_per_min=None,
                        metadata={"source_file": _rel(oxy_path)},
                    )
                )
                continue

            eeg_onsets_ms, eeg_trigger_diagnostics = _read_visual_eeg_onsets(annotation_path)
            eeg_marker = {
                "time": np.asarray(eeg_onsets_ms, dtype=np.float64),
                "y": np.ones((1, len(eeg_onsets_ms)), dtype=np.float32),
                "className": ["stimulus_onset"],
            }
            fnirs_marker = {
                "time": np.asarray([item["onset_ms"] for item in stimulus_marks], dtype=np.float64),
                "y": np.ones((1, len(stimulus_marks)), dtype=np.float32),
                "className": ["stimulus_onset"],
            }
            aligned_events, report = align_paired_marker_streams(
                dataset_id="visual_cognitive_motivation",
                subject=subject_dir.name,
                record_id=record_id,
                eeg_marker=eeg_marker,
                fnirs_marker=fnirs_marker,
                event_type="trial",
            )
            part_match = re.search(r"Part(\d+)", record_id, flags=re.IGNORECASE)
            epoch_offset = 125 * (int(part_match.group(1)) - 1) if part_match else 0
            source_files = [
                _rel(annotation_path),
                _rel(oxy_path),
                _rel(deoxy_path),
                _rel(subject_dir / f"{subject_dir.name}_type.xlsx"),
            ]
            for index, event in enumerate(aligned_events):
                source_index = int(event.metadata["fnirs_source_index"])
                epoch_id = epoch_offset + source_index + 1
                epoch_type_raw = type_map.get(epoch_id, "")
                epoch_type = epoch_type_raw if epoch_type_raw in VISUAL_VALID_EPOCH_TYPES else "unknown"
                mark = stimulus_marks[source_index]
                events.append(
                    CanonicalEvent(
                        **{
                            **event.to_dict(),
                            "event_index": index,
                            "label": epoch_type,
                            "label_index": VISUAL_EPOCH_TYPE_INDICES[epoch_type],
                            "metadata": {
                                **dict(event.metadata),
                                **mark,
                                "task": "visual_cognitive_motivation",
                                "event_role": "stimulus_onset",
                                "condition_label": epoch_type,
                                "epoch_id": epoch_id,
                                "epoch_type": epoch_type,
                                "epoch_type_raw": epoch_type_raw,
                                "sample_rate_hz": sample_rate,
                                "source_files": source_files,
                            },
                        }
                    )
                )
            reports.append(
                EventAlignmentReport(
                    **{
                        **report.to_dict(),
                        "metadata": {
                            **dict(report.metadata),
                            **eeg_trigger_diagnostics,
                            "eeg_annotation_trigger_count": eeg_trigger_diagnostics["raw_dc9_count"],
                            "eeg_stimulus_count": len(eeg_onsets_ms),
                            "fnirs_stimulus_count": len(stimulus_marks),
                            "source_files": source_files,
                        },
                    }
                )
            )
    return events, reports


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    previous = output_dir / "event_manifest.json"
    if previous.exists() and json.loads(previous.read_text()).get("event_alignment_schema") != EVENT_ALIGNMENT_SCHEMA:
        raise ValueError("Refusing to overwrite a deprecated event index; use a new versioned directory")
    parent_manifest = output_dir.parent / "cache_manifest.json"
    if parent_manifest.exists():
        from src.data.clean_physiology_cache import require_current_cache_manifest
        require_current_cache_manifest(json.loads(parent_manifest.read_text()))
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output directory exists; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    reports_path = output_dir / "alignment_reports.jsonl"
    events_path.write_text("", encoding="utf-8")
    reports_path.write_text("", encoding="utf-8")

    counts: dict[str, dict[str, int]] = {}
    for dataset in args.datasets:
        if dataset == "eeg_fnirs_single_trial":
            events, reports = iter_single_trial(DATA_ROOTS[dataset], args.subjects_per_dataset, args.records_per_subject)
        elif dataset == "simultaneous_eeg_nirs":
            events, reports = iter_simultaneous(DATA_ROOTS[dataset], args.subjects_per_dataset, args.records_per_subject)
        elif dataset == "refed":
            events, reports = iter_refed(DATA_ROOTS[dataset], args.subjects_per_dataset, args.records_per_subject)
        elif dataset == "visual_cognitive_motivation":
            events, reports = iter_visual(DATA_ROOTS[dataset], args.subjects_per_dataset, args.records_per_subject)
        else:
            continue
        event_count = _append_jsonl(events_path, (with_canonical_fields(event.to_dict()) for event in events))
        report_count = _append_jsonl(reports_path, (with_canonical_fields(report.to_dict()) for report in reports))
        counts[dataset] = {"events": event_count, "alignment_reports": report_count}

    manifest = {
        "schema": EVENT_INDEX_SCHEMA,
        "event_alignment_schema": EVENT_ALIGNMENT_SCHEMA,
        "canonical_join_contract": {
            "schema": "clean_physiology_cache_index_v1",
            "key_fields": ["dataset_id", "canonical_subject_id", "base_record_id"],
            "join_key": "dataset_id|canonical_subject_id|base_record_id",
            "signal_branch": "separates multiple signal exports for the same canonical record",
        },
        "dataset_timing_contracts": {
            "refed": REFED_CONTINUOUS_TIMING_CONTRACT,
            "visual_cognitive_motivation": VISUAL_TIMING_CONTRACT,
        },
        "parameters": {
            "datasets": args.datasets,
            "subjects_per_dataset": args.subjects_per_dataset,
            "records_per_subject": args.records_per_subject,
        },
        "files": {
            "events_jsonl": _rel(events_path),
            "alignment_reports_jsonl": _rel(reports_path),
        },
        "counts": counts,
    }
    write_json(output_dir / "event_manifest.json", manifest, ensure_ascii=False)
    print(json.dumps({"output_dir": str(output_dir), "counts": counts}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
