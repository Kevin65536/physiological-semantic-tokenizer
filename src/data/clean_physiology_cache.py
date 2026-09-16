"""Canonical index and aligned-window loader for the clean physiology cache."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np

from .event_alignment import EVENT_ALIGNMENT_SCHEMA


CLEAN_PHYSIOLOGY_CACHE_INDEX_SCHEMA = "clean_physiology_cache_index_v1"
CLEAN_CACHE_SCHEMA = "clean_eeg_fnirs_cache_v2"
# Immutable v1 artifacts remain historical evidence, not inputs to the v2 reader.
DEPRECATED_CACHE_SCHEMAS = frozenset({"clean_eeg_fnirs_cache_v1"})


def require_current_cache_manifest(manifest: Mapping[str, Any]) -> None:
    schema = manifest.get("schema")
    if schema != CLEAN_CACHE_SCHEMA:
        status = "Deprecated" if schema in DEPRECATED_CACHE_SCHEMAS else "Unsupported"
        raise ValueError(
            f"{status} physiology cache schema {schema!r}; expected {CLEAN_CACHE_SCHEMA}. "
            "Old caches are retained for historical evidence only. A separate versioned "
            "rebuild is required; this reader never rebuilds automatically."
        )


def canonical_subject_id(dataset_id: str, subject: str) -> str:
    """Return the subject key shared by signal manifests and event sidecars."""
    value = str(subject)
    if dataset_id == "simultaneous_eeg_nirs":
        return value.split("-")[0]
    if dataset_id == "eeg_fnirs_single_trial":
        match = re.search(r"(\d+)$", value)
        return f"subject_{int(match.group(1)):02d}" if match else value.replace(" ", "_")
    return value


def base_record_id(dataset_id: str, record_id: str) -> str:
    """Return the task/segment id before branch-specific signal suffixes."""
    value = str(record_id)
    if dataset_id == "refed":
        for suffix in ("_hbo_hbr", "_absorbance_780_805_830"):
            if value.endswith(suffix):
                return value[: -len(suffix)]
    return value


def signal_branch(dataset_id: str, record_id: str, metadata: Mapping[str, Any] | None = None) -> str:
    """Return the signal branch within a canonical record."""
    metadata = metadata or {}
    if dataset_id == "refed":
        value = str(record_id)
        if value.endswith("_absorbance_780_805_830"):
            return "absorbance_780_805_830"
        if value.endswith("_hbo_hbr"):
            return "hbo_hbr"
    if dataset_id == "eeg_fnirs_single_trial":
        return "homer2_wavelength_pair"
    if dataset_id in {"simultaneous_eeg_nirs", "visual_cognitive_motivation"}:
        return "oxy_deoxy"
    return str(metadata.get("metadata_signal") or "default")


def join_key(dataset_id: str, subject: str, record_id: str) -> str:
    return "|".join(
        (
            str(dataset_id),
            canonical_subject_id(dataset_id, subject),
            base_record_id(dataset_id, record_id),
        )
    )


def with_canonical_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a manifest/event row and add normalized join fields."""
    dataset_id = str(row.get("dataset_id", ""))
    subject = str(row.get("subject", ""))
    record_id = str(row.get("record_id", ""))
    output = dict(row)
    output["canonical_subject_id"] = canonical_subject_id(dataset_id, subject)
    output["base_record_id"] = base_record_id(dataset_id, record_id)
    output["signal_branch"] = signal_branch(dataset_id, record_id, row.get("metadata", {}))
    output["join_key"] = join_key(dataset_id, subject, record_id)
    return output


@dataclass(frozen=True)
class CleanCacheRecord:
    dataset_id: str
    subject: str
    record_id: str
    canonical_subject_id: str
    base_record_id: str
    signal_branch: str
    join_key: str
    sample_rate_hz: float
    npz_path: Path
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class AlignedWindow:
    record: CleanCacheRecord
    event: Mapping[str, Any]
    start_index: int
    stop_index: int


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class CleanPhysiologyCacheIndex:
    """Index signal records, events, and alignment reports by canonical keys."""

    def __init__(self, cache_root: str | Path = "data/cache/physiology_semantic_clean_v1") -> None:
        self.cache_root = Path(cache_root)
        self.project_root = Path.cwd()
        self.cache_manifest = _read_json(self.cache_root / "cache_manifest.json")
        require_current_cache_manifest(self.cache_manifest)
        self.event_manifest_path = self.cache_root / "event_index" / "event_manifest.json"
        self.event_manifest = _read_json(self.event_manifest_path) if self.event_manifest_path.exists() else {}
        if self.event_manifest.get("event_alignment_schema") != EVENT_ALIGNMENT_SCHEMA:
            raise ValueError(f"Deprecated or missing event index; expected {EVENT_ALIGNMENT_SCHEMA}; rebuild separately.")
        self.records = [self._record_from_manifest(row) for row in self.cache_manifest.get("records", [])]
        self.events = [with_canonical_fields(row) for row in _read_jsonl(self.cache_root / "event_index" / "events.jsonl")]
        self.alignment_reports = [
            with_canonical_fields(row) for row in _read_jsonl(self.cache_root / "event_index" / "alignment_reports.jsonl")
        ]
        self.records_by_join_key = self._group_records(self.records)
        self.events_by_join_key = self._group_rows(self.events)
        self.reports_by_join_key = self._group_rows(self.alignment_reports)

    def _resolve_cache_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        candidate = self.project_root / path
        return candidate if candidate.exists() else path

    def _record_from_manifest(self, row: Mapping[str, Any]) -> CleanCacheRecord:
        require_current_cache_manifest(row)
        normalized = with_canonical_fields(row)
        return CleanCacheRecord(
            dataset_id=str(normalized["dataset_id"]),
            subject=str(normalized["subject"]),
            record_id=str(normalized["record_id"]),
            canonical_subject_id=str(normalized["canonical_subject_id"]),
            base_record_id=str(normalized["base_record_id"]),
            signal_branch=str(normalized["signal_branch"]),
            join_key=str(normalized["join_key"]),
            sample_rate_hz=float(normalized["sample_rate_hz"]),
            npz_path=self._resolve_cache_path(str(normalized["record_npz"])),
            manifest=normalized,
        )

    @staticmethod
    def _group_records(records: Iterable[CleanCacheRecord]) -> dict[str, list[CleanCacheRecord]]:
        grouped: dict[str, list[CleanCacheRecord]] = {}
        for record in records:
            grouped.setdefault(record.join_key, []).append(record)
        return grouped

    @staticmethod
    def _group_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["join_key"]), []).append(row)
        return grouped

    def records_with_events(self, *, branch_preference: str | None = None) -> list[CleanCacheRecord]:
        records: list[CleanCacheRecord] = []
        for record in self.records:
            if branch_preference is not None and record.signal_branch != branch_preference:
                continue
            if record.join_key in self.events_by_join_key:
                records.append(record)
        return records

    def coverage_summary(self) -> dict[str, Any]:
        record_keys = set(self.records_by_join_key)
        event_keys = set(self.events_by_join_key)
        report_keys = set(self.reports_by_join_key)
        return {
            "schema": CLEAN_PHYSIOLOGY_CACHE_INDEX_SCHEMA,
            "record_count": len(self.records),
            "event_count": len(self.events),
            "alignment_report_count": len(self.alignment_reports),
            "record_join_key_count": len(record_keys),
            "event_join_key_count": len(event_keys),
            "alignment_report_join_key_count": len(report_keys),
            "record_keys_without_events": sorted(record_keys - event_keys),
            "event_keys_without_records": sorted(event_keys - record_keys),
            "record_keys_without_alignment_reports": sorted(record_keys - report_keys),
        }

    def load_record_arrays(self, record: CleanCacheRecord) -> dict[str, np.ndarray]:
        with np.load(record.npz_path, allow_pickle=False) as npz:
            return {key: np.asarray(npz[key]) for key in npz.files}


class CleanPhysiologyAlignedWindowDataset:
    """Event-aligned fNIRS windows from the normalized clean cache."""

    def __init__(
        self,
        cache_root: str | Path = "data/cache/physiology_semantic_clean_v1",
        *,
        array_key: str = "homer2_aligned_fnirs",
        branch_preference: str | None = "hbo_hbr",
        window_duration_s: float = 8.0,
        window_offset_s: float = 0.0,
        include_event_types: set[str] | None = None,
    ) -> None:
        self.index = CleanPhysiologyCacheIndex(cache_root)
        self.array_key = str(array_key)
        self.branch_preference = branch_preference
        self.window_duration_s = float(window_duration_s)
        self.window_offset_s = float(window_offset_s)
        self.include_event_types = include_event_types
        self.windows = self._build_windows()

    def _select_records(self) -> Iterable[CleanCacheRecord]:
        for record in self.index.records:
            if record.join_key not in self.index.events_by_join_key:
                continue
            if record.dataset_id == "refed" and self.branch_preference and record.signal_branch != self.branch_preference:
                continue
            yield record

    def _build_windows(self) -> list[AlignedWindow]:
        windows: list[AlignedWindow] = []
        for record in self._select_records():
            num_samples = self._record_num_samples(record)
            if num_samples is None:
                continue
            sample_rate = float(record.sample_rate_hz)
            length = max(int(round(self.window_duration_s * sample_rate)), 1)
            offset = int(round(self.window_offset_s * sample_rate))
            for event in self.index.events_by_join_key.get(record.join_key, []):
                if self.include_event_types is not None and str(event.get("event_type")) not in self.include_event_types:
                    continue
                onset_ms = event.get("onset_ms")
                if onset_ms is None:
                    continue
                start = int(round(float(onset_ms) / 1000.0 * sample_rate)) + offset
                stop = start + length
                lower, upper = event.get("metadata", {}).get("alignment_support_ms", {}).get("fnirs", (None, None))
                if ((lower is not None and start / sample_rate * 1000 < lower)
                        or (upper is not None and stop / sample_rate * 1000 > upper)):
                    continue
                if start < 0 or stop > num_samples:
                    continue
                windows.append(AlignedWindow(record=record, event=event, start_index=start, stop_index=stop))
        return windows

    def _record_num_samples(self, record: CleanCacheRecord) -> int | None:
        for contract_name in ("homer2_aligned_contract", "raw_native_contract"):
            contract = record.manifest.get(contract_name, {})
            if contract.get("array_key") != self.array_key:
                continue
            shape = contract.get("summary", {}).get("shape")
            if shape:
                return int(shape[0])
        arrays = self.index.load_record_arrays(record)
        if self.array_key not in arrays:
            return None
        return int(arrays[self.array_key].shape[0])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        arrays = self.index.load_record_arrays(window.record)
        fnirs = np.asarray(arrays[self.array_key][window.start_index : window.stop_index], dtype=np.float32).T
        return {
            "fnirs": fnirs,
            "eeg": None,
            "modality_available": {"fnirs": True, "eeg": False},
            "label": str(window.event.get("label", "")),
            "label_index": window.event.get("label_index"),
            "dataset_id": window.record.dataset_id,
            "subject": window.record.subject,
            "canonical_subject_id": window.record.canonical_subject_id,
            "record_id": window.record.record_id,
            "base_record_id": window.record.base_record_id,
            "signal_branch": window.record.signal_branch,
            "join_key": window.record.join_key,
            "event": dict(window.event),
            "sample_rate_hz": window.record.sample_rate_hz,
            "sample_slice": (window.start_index, window.stop_index),
        }
