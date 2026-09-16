"""Unified event, label, and cross-modal timing contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np


EVENT_ALIGNMENT_SCHEMA = "physiology_event_alignment_v2"
ADMISSIBLE_ALIGNMENT_CASES = frozenset({
    "stable_fixed_offset", "piecewise_constant_offset",
    "skip_aligned_piecewise_constant_offset", "shared_segment_index_no_marker_stream",
})


@dataclass(frozen=True)
class CanonicalEvent:
    dataset_id: str
    subject: str
    record_id: str
    event_index: int
    event_type: str
    label: str
    label_index: int | None = None
    eeg_time_ms: float | None = None
    fnirs_time_ms: float | None = None
    onset_ms: float | None = None
    duration_ms: float | None = None
    alignment_role: str = "native"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = EVENT_ALIGNMENT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = _jsonable(payload["metadata"])
        return payload


@dataclass(frozen=True)
class EventAlignmentReport:
    dataset_id: str
    subject: str
    record_id: str
    num_eeg_events: int
    num_fnirs_events: int
    num_aligned_events: int
    alignment_case: str
    label_sequence_match: bool | None
    offset_mean_ms: float | None
    offset_std_ms: float | None
    drift_slope_ms_per_min: float | None
    offset_blocks: tuple[Mapping[str, Any], ...] = ()
    skipped_marker_indices: Mapping[str, Sequence[int]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = EVENT_ALIGNMENT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["offset_blocks"] = [_jsonable(block) for block in payload["offset_blocks"]]
        payload["skipped_marker_indices"] = _jsonable(payload["skipped_marker_indices"])
        payload["metadata"] = _jsonable(payload["metadata"])
        return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_class_names(value: Any) -> list[str]:
    if value is None:
        return []
    array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        return [str(array.item())]
    return [str(item) for item in array.ravel().tolist()]


def normalize_marker_targets(marker_y: Any, n_events: int) -> np.ndarray:
    y = np.asarray(marker_y)
    if y.ndim == 2 and y.shape[1] == n_events:
        return y.astype(np.float32, copy=False)
    if y.ndim == 2 and y.shape[0] == n_events:
        return y.T.astype(np.float32, copy=False)
    if y.ndim == 1 and y.shape[0] == n_events:
        unique = list(dict.fromkeys(int(item) for item in y.tolist()))
        matrix = np.zeros((len(unique), n_events), dtype=np.float32)
        lookup = {value: index for index, value in enumerate(unique)}
        for event_index, value in enumerate(y.tolist()):
            matrix[lookup[int(value)], event_index] = 1.0
        return matrix
    return np.ones((1, n_events), dtype=np.float32)


def normalize_marker_struct(marker_struct: Any) -> dict[str, Any]:
    time = np.asarray(getattr(marker_struct, "time"), dtype=np.float64).reshape(-1)
    event = getattr(marker_struct, "event", None)
    event_desc = getattr(event, "desc", None)
    if event_desc is not None:
        event_desc = np.asarray(event_desc).reshape(-1)
    y = normalize_marker_targets(getattr(marker_struct, "y", None), len(time))
    return {
        "time": time,
        "y": y,
        "className": normalize_class_names(getattr(marker_struct, "className", None)),
        "event_desc": event_desc,
    }


def marker_label_indices(marker_info: Mapping[str, Any]) -> np.ndarray:
    y = np.asarray(marker_info.get("y"))
    if y.ndim == 2 and y.shape[1] == len(marker_info.get("time", [])):
        return np.argmax(y, axis=0).astype(np.int64)
    return np.zeros(len(marker_info.get("time", [])), dtype=np.int64)


def marker_label_names(marker_info: Mapping[str, Any]) -> list[str]:
    indices = marker_label_indices(marker_info)
    class_names = [str(item) for item in marker_info.get("className", [])]
    output = []
    for index in indices.tolist():
        output.append(class_names[index] if 0 <= index < len(class_names) else str(index))
    return output


def detect_offset_blocks(residual_ms: np.ndarray, jump_threshold_ms: float = 20_000.0) -> list[dict[str, Any]]:
    residual = np.asarray(residual_ms, dtype=np.float64).reshape(-1)
    if residual.size == 0:
        return []
    start = 0
    blocks: list[dict[str, Any]] = []
    for index in range(1, len(residual)):
        if abs(float(residual[index] - residual[index - 1])) > jump_threshold_ms:
            blocks.append(_offset_block(start, index - 1, residual[start:index]))
            start = index
    blocks.append(_offset_block(start, len(residual) - 1, residual[start:]))
    return blocks


def _offset_block(start: int, end: int, residual: np.ndarray) -> dict[str, Any]:
    return {
        "start_index": int(start),
        "end_index": int(end),
        "count": int(len(residual)),
        "offset_mean_ms": float(np.mean(residual)),
        "offset_std_ms": float(np.std(residual)),
    }


def drift_slope_ms_per_min(eeg_time_ms: np.ndarray, residual_ms: np.ndarray) -> float | None:
    if len(eeg_time_ms) < 2 or len(residual_ms) < 2:
        return None
    x_min = (np.asarray(eeg_time_ms, dtype=np.float64) - float(eeg_time_ms[0])) / 60_000.0
    x0 = x_min - float(np.mean(x_min))
    denom = float(np.dot(x0, x0))
    if denom <= 0:
        return None
    y = np.asarray(residual_ms, dtype=np.float64)
    return float(np.dot(x0, y - float(np.mean(y))) / denom)


def visual_stimulus_onsets_from_dc9(
    trigger_onsets_ms: Sequence[float],
    *,
    stimulus_duration_ms: float = 3_000.0,
    tolerance_ms: float = 10.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover Visual stimulus onsets from the untyped EEG DC9 stream.

    The source dataset emits the same DC9 annotation for stimulus appearance,
    stimulus disappearance, and participant response.  Appearance is the DC9
    event followed by the fixed three-second disappearance event.  Detecting
    that semantic pair is robust to a missing response annotation and to the
    duplicated annotation rows present in S15 Part1; taking every third row is
    not.
    """
    raw = np.asarray(trigger_onsets_ms, dtype=np.float64).reshape(-1)
    finite = raw[np.isfinite(raw)]
    distinct = np.unique(finite)
    if distinct.size < 2:
        onsets = np.asarray([], dtype=np.float64)
    else:
        intervals = np.diff(distinct)
        onset_mask = np.abs(intervals - float(stimulus_duration_ms)) <= float(tolerance_ms)
        onsets = distinct[:-1][onset_mask]
    return onsets, {
        "extraction_rule": "dc9_followed_by_stimulus_offset_at_3000ms",
        "stimulus_duration_ms": float(stimulus_duration_ms),
        "tolerance_ms": float(tolerance_ms),
        "raw_dc9_count": int(raw.size),
        "finite_dc9_count": int(finite.size),
        "distinct_dc9_count": int(distinct.size),
        "duplicate_dc9_count": int(finite.size - distinct.size),
        "stimulus_onset_candidate_count": int(onsets.size),
    }


def classify_alignment(
    residual_ms: np.ndarray,
    blocks: Sequence[Mapping[str, Any]],
    *,
    eeg_time_ms: np.ndarray | None = None,
    skipped_marker_indices: Mapping[str, Sequence[int]] | None = None,
    stable_block_std_threshold_ms: float = 100.0,
    continuous_drift_slope_threshold_ms_per_min: float = 10.0,
) -> str:
    residual = np.asarray(residual_ms, dtype=np.float64).reshape(-1)
    if residual.size == 0:
        return "no_common_events"
    # Evaluate physical drift within segments before the dispersion shortcut.
    for block in blocks:
        start, stop = int(block["start_index"]), int(block["end_index"]) + 1
        slope = (drift_slope_ms_per_min(eeg_time_ms[start:stop], residual[start:stop])
                 if eeg_time_ms is not None else block.get("drift_slope_ms_per_min"))
        if slope is not None and abs(slope) >= continuous_drift_slope_threshold_ms_per_min:
            return "continuous_drift"
    stable_blocks = all(float(block["offset_std_ms"]) <= stable_block_std_threshold_ms for block in blocks)
    skipped = bool(skipped_marker_indices and any(skipped_marker_indices.values()))
    if len(blocks) == 1 and stable_blocks:
        return "stable_fixed_offset"
    if len(blocks) > 1 and stable_blocks:
        return "skip_aligned_piecewise_constant_offset" if skipped else "piecewise_constant_offset"
    return "mixed_or_unstable_offset"


def align_paired_marker_streams(
    *, dataset_id: str, subject: str, record_id: str,
    eeg_marker: Mapping[str, Any], fnirs_marker: Mapping[str, Any],
    event_type: str = "trial", jump_threshold_ms: float = 20_000.0,
) -> tuple[list[CanonicalEvent], EventAlignmentReport]:
    eeg_times = np.asarray(eeg_marker.get("time", []), dtype=np.float64).reshape(-1)
    fnirs_times = np.asarray(fnirs_marker.get("time", []), dtype=np.float64).reshape(-1)
    eeg_labels, fnirs_labels = marker_label_names(eeg_marker), marker_label_names(fnirs_marker)
    labels = marker_label_indices(eeg_marker)
    ei, fi = np.arange(len(eeg_times)), np.arange(len(fnirs_times))
    skipped: dict[str, list[int]] = {"eeg_indices": [], "fnirs_indices": []}

    def reject(reason: str) -> tuple[list[CanonicalEvent], EventAlignmentReport]:
        return [], EventAlignmentReport(
            dataset_id, subject, record_id, len(eeg_times), len(fnirs_times), 0,
            reason, False, None, None, None, metadata={"rejection_reason": reason},
        )

    if any(not np.isfinite(t).all() or np.any(np.diff(t) <= 0) for t in (eeg_times, fnirs_times)):
        return reject("invalid_marker_timestamps")
    if not len(ei) or not len(fi):
        return reject("no_common_events")
    if abs(len(ei) - len(fi)) > 1:
        return reject("unresolved_marker_count_mismatch")
    if len(ei) != len(fi):
        candidates = []
        eeg_longer = len(ei) > len(fi)
        for skip in range(max(len(ei), len(fi))):
            ce = np.delete(ei, skip) if eeg_longer else ei
            cf = fi if eeg_longer else np.delete(fi, skip)
            if [eeg_labels[i] for i in ce] != [fnirs_labels[i] for i in cf]:
                continue
            residual = fnirs_times[cf] - eeg_times[ce]
            blocks = detect_offset_blocks(residual, jump_threshold_ms)
            # A wrong pairing must not benefit from inventing extra segments.
            score = (len(blocks), sum(b["count"] * b["offset_std_ms"] ** 2 for b in blocks))
            candidates.append((score, skip, ce, cf))
        candidates.sort(key=lambda item: item[0])
        if not candidates:
            return reject("label_sequence_mismatch")
        if (len(candidates) > 1 and candidates[0][0][0] == candidates[1][0][0]
                and np.isclose(candidates[0][0][1], candidates[1][0][1])):
            return reject("ambiguous_marker_pairing")
        _, skip, ei, fi = candidates[0]
        skipped["eeg_indices" if eeg_longer else "fnirs_indices"] = [int(skip)]
    if [eeg_labels[i] for i in ei] != [fnirs_labels[i] for i in fi]:
        return reject("label_sequence_mismatch")

    et, ft = eeg_times[ei], fnirs_times[fi]
    residual = ft - et
    blocks = detect_offset_blocks(residual, jump_threshold_ms)
    for block in blocks:
        start, stop = block["start_index"], block["end_index"] + 1
        block["drift_slope_ms_per_min"] = drift_slope_ms_per_min(et[start:stop], residual[start:stop])
    case = classify_alignment(residual, blocks, eeg_time_ms=et, skipped_marker_indices=skipped)
    report = EventAlignmentReport(
        dataset_id, subject, record_id, len(eeg_times), len(fnirs_times), len(residual),
        case, True, float(residual.mean()), float(residual.std()),
        drift_slope_ms_per_min(et, residual), tuple(blocks), skipped,
        metadata={"residual_series_ms": residual.tolist(),
                  "global_slope_includes_segment_jumps": len(blocks) > 1},
    )
    events = []
    for block_index, block in enumerate(blocks):
        start, stop = block["start_index"], block["end_index"]
        # Exact append boundaries are unavailable: the inter-anchor gap around
        # an offset jump is unverified support, not a guessed midpoint boundary.
        support = {
            "policy": "exclude_unverified_concatenation_gaps",
            "eeg": [float(et[start]) if block_index else None,
                    float(et[stop]) if block_index < len(blocks) - 1 else None],
            "fnirs": [float(ft[start]) if block_index else None,
                      float(ft[stop]) if block_index < len(blocks) - 1 else None],
        }
        for index in range(start, stop + 1):
            events.append(CanonicalEvent(
                dataset_id, subject, record_id, index, event_type,
                eeg_labels[ei[index]], int(labels[ei[index]]),
                float(et[index]), float(ft[index]), float(ft[index]),
                alignment_role="paired_eeg_fnirs_marker",
                metadata={"eeg_label": eeg_labels[ei[index]], "fnirs_label": fnirs_labels[fi[index]],
                          "eeg_source_index": int(ei[index]), "fnirs_source_index": int(fi[index]),
                          "offset_ms": float(residual[index]), "alignment_support_ms": support},
            ))
    return events, report


def window_within_alignment_support(
    event: Mapping[str, Any], offset_s: float, duration_s: float,
    eeg_rate: float = 200.0, fnirs_rate: float = 10.0,
) -> bool:
    """Check the entire sampled window against verified segment support."""
    support = event.get("metadata", {}).get("alignment_support_ms", {})
    for modality, rate in (("eeg", eeg_rate), ("fnirs", fnirs_rate)):
        lower, upper = support.get(modality, (None, None))
        anchor = event.get(f"{modality}_time_ms")
        if anchor is None:
            anchor = event.get("onset_ms")
        if anchor is None or not np.isfinite(anchor):
            return False
        onset = float(anchor) + offset_s * 1000.0
        start = round(onset * rate / 1000.0) * 1000.0 / rate
        stop = start + round(duration_s * rate) * 1000.0 / rate
        if lower is not None and start < lower - 1e-8:
            return False
        if upper is not None and stop > upper + 1e-8:
            return False
    return True



def read_visual_fnirs_csv(path: str | Path, *, load_signals: bool = False) -> dict[str, Any]:
    """Read one ETG export without compressing dropped rows or its native clock."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    data_line = lines.index("Data")
    period = float(next(csv.reader([next(
        line for line in lines[:data_line] if line.startswith("Sampling Period[s]")
    )]))[1])
    if not np.isfinite(period) or period <= 0:
        raise ValueError(f"Invalid sampling period: {path}")
    header, *rows = list(csv.reader(lines[data_line + 1:]))
    time_index, mark_index = header.index("Time"), header.index("Mark")
    channel_indices = [i for i, name in enumerate(header) if re.fullmatch(r"CH\d+", name.strip())]
    if not rows or not channel_indices:
        raise ValueError(f"Empty ETG recording: {path}")
    times, marks, values = [], [], []
    day = 0.0
    previous = None
    for index, row in enumerate(rows):
        if len(row) != len(header):
            raise ValueError(f"Malformed ETG row {index}: {path}")
        try:
            parts = row[time_index].strip().split(":")
            if len(parts) not in (2, 3):
                raise ValueError("expected mm:ss or hh:mm:ss")
            seconds = sum(float(v) * scale for v, scale in zip(parts, (3600, 60, 1)[-len(parts):]))
            if previous is not None and seconds - previous < -43200:
                day += 86400.0
            previous = seconds
            times.append(seconds + day)
            mark = int(float(row[mark_index]))
            if load_signals:
                sample = [float(row[i]) for i in channel_indices]
                if not np.isfinite(sample).all():
                    raise ValueError("nonfinite signal")
                values.append(sample)
        except (ValueError, IndexError) as error:
            raise ValueError(f"Invalid ETG row {index}: {path}: {error}") from error
        if mark > 0:
            marks.append({"sample_index": index, "mark": mark, "clock_time": row[time_index],
                          "body_movement": row[header.index("BodyMovement")] if "BodyMovement" in header else "",
                          "removal_mark": row[header.index("RemovalMark")] if "RemovalMark" in header else ""})
    clock = np.asarray(times, dtype=np.float64)
    if not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0):
        raise ValueError(f"Non-monotonic ETG clock: {path}")
    if np.any(np.diff(clock) > 1.5 * period):
        raise ValueError(f"Missing ETG time support; refusing to interpolate a gap: {path}")
    relative = clock - clock[0]
    for mark in marks:
        mark["onset_ms"] = float(relative[mark["sample_index"]] * 1000.0)
    return {"time_s": relative, "clock_s": clock, "sample_rate_hz": 1.0 / period,
            "marks": marks, "values": np.asarray(values, dtype=np.float64) if load_signals else None}

def read_xlsx_rows(path: str) -> list[dict[str, str]]:
    """Read the first worksheet of a small xlsx file using only stdlib."""
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        shared: list[str] = []
        ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared.append("".join(text.text or "" for text in item.findall(".//a:t", ns)))
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in root.findall(".//a:row", ns):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", ns):
                ref = cell.get("r", "")
                column = _excel_column_index(ref)
                value_node = cell.find("a:v", ns)
                value = "" if value_node is None else str(value_node.text or "")
                if cell.get("t") == "s" and value:
                    value = shared[int(value)]
                values[column] = value
            if values:
                max_col = max(values)
                rows.append([values.get(index, "") for index in range(max_col + 1)])
    if not rows:
        return []
    header = [str(item).strip() for item in rows[0]]
    return [
        {header[index]: row[index] if index < len(row) else "" for index in range(len(header)) if header[index]}
        for row in rows[1:]
    ]


def _excel_column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1
