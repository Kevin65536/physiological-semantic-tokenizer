#!/usr/bin/env python3
"""Build a versioned clean fNIRS cache with raw-native and HOMER2-aligned branches."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Sequence

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fnirs_standardization import DATASET_FNIRS_CONTRACTS, FNIRSMeasurementContract, standardize_fnirs_record  # noqa: E402
from src.data.homer2_preprocessing import (  # noqa: E402
    HOMER2_ALIGNMENT_SCHEMA,
    apply_homer2_aligned_contract,
    homer2_compatibility_manifest,
)
from src.data.clean_physiology_cache import CLEAN_CACHE_SCHEMA, require_current_cache_manifest, with_canonical_fields  # noqa: E402
from src.utils.io import save_npz, write_json  # noqa: E402


from src.data.event_alignment import read_visual_fnirs_csv

DATA_ROOTS = {
    "eeg_fnirs_single_trial": PROJECT_ROOT / "data/EEG+NIRS Single-Trial",
    "refed": PROJECT_ROOT / "data/REFED-dataset",
    "visual_cognitive_motivation": PROJECT_ROOT / "data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults",
    "simultaneous_eeg_nirs": PROJECT_ROOT / "data/Simultaneous EEG&NIRS",
}

_SOURCE_HASH_CACHE: dict[Path, str] = {}


@dataclass(frozen=True)
class CleanInputRecord:
    dataset_id: str
    subject: str
    record_id: str
    source_paths: tuple[Path, ...]
    values: np.ndarray
    homer2_input: np.ndarray
    sample_rate_hz: float
    contract: FNIRSMeasurementContract
    entry_stage: str
    wavelengths_nm: tuple[float, ...]
    channel_names: tuple[str, ...]
    homer2_channel_names: tuple[str, ...]
    metadata: dict[str, Any]
    native_time_s: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATA_ROOTS),
        choices=list(DATA_ROOTS),
        help="Datasets to include.",
    )
    parser.add_argument("--subjects-per-dataset", type=int, default=1)
    parser.add_argument("--records-per-subject", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional leading sample cap for smoke runs.")
    parser.add_argument("--include-refed-absorbance", action="store_true")
    parser.add_argument("--output-dir", default="data/cache/physiology_semantic_clean_v2")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    resolved = path.resolve()
    if resolved in _SOURCE_HASH_CACHE:
        return _SOURCE_HASH_CACHE[resolved]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    value = digest.hexdigest()
    _SOURCE_HASH_CACHE[resolved] = value
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


def _mat_payload(path: Path, key: str | None = None) -> Any:
    payload = loadmat(path, struct_as_record=False, squeeze_me=True)
    if key is None:
        key = next(name for name in payload if not name.startswith("__"))
    value = payload[key]
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        value = value.item()
    return value


def _labels(value: Any) -> list[str]:
    return [str(item) for item in np.asarray(value, dtype=object).ravel()]


def _cap(values: np.ndarray, max_samples: int) -> np.ndarray:
    if max_samples <= 0 or values.shape[0] <= max_samples:
        return values
    return values[:max_samples]


def _pair_single_trial_wavelengths(values: np.ndarray, labels: Sequence[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    low = [index for index, label in enumerate(labels) if "lowWL" in label]
    high = [index for index, label in enumerate(labels) if "highWL" in label]
    if not low or len(low) != len(high):
        raise ValueError(f"cannot pair lowWL/highWL channels: low={len(low)}, high={len(high)}")
    paired = np.stack((values[:, low], values[:, high]), axis=2)
    pair_labels = tuple(re.sub(r"(lowWL|highWL).*$", "", labels[index]).strip() or f"pair_{i:02d}" for i, index in enumerate(low))
    return paired, pair_labels


def iter_single_trial(root: Path, subject_limit: int, record_limit: int, max_samples: int) -> Iterator[CleanInputRecord]:
    contract = DATASET_FNIRS_CONTRACTS["eeg_fnirs_single_trial"]["wavelength_pair"]
    subjects = sorted((root / "NIRS_01-29").glob("subject *"))[:subject_limit]
    for subject in subjects:
        path = subject / "cnt.mat"
        sessions = np.atleast_1d(_mat_payload(path, "cnt"))
        for index, session in enumerate(sessions[:record_limit]):
            values = _cap(np.asarray(session.x, dtype=np.float64), max_samples)
            labels = _labels(session.clab)
            paired, pair_labels = _pair_single_trial_wavelengths(values, labels)
            yield CleanInputRecord(
                dataset_id=contract.dataset_id,
                subject=subject.name,
                record_id=f"session_{index:02d}",
                source_paths=(path,),
                values=values,
                homer2_input=paired,
                sample_rate_hz=float(session.fs),
                contract=contract,
                entry_stage="raw_intensity",
                wavelengths_nm=(760.0, 850.0),
                channel_names=tuple(labels),
                homer2_channel_names=tuple(f"{label}_{role}" for label in pair_labels for role in ("HbO", "HbR")),
                metadata={
                    "metadata_unit": str(getattr(session, "yUnit", "")),
                    "metadata_signal": str(getattr(session, "signal", "")),
                    "homer2_pair_labels": list(pair_labels),
                },
            )


def iter_simultaneous(root: Path, subject_limit: int, record_limit: int, max_samples: int) -> Iterator[CleanInputRecord]:
    contract = DATASET_FNIRS_CONTRACTS["simultaneous_eeg_nirs"]["oxy_deoxy"]
    for subject in sorted(root.glob("VP*-NIRS"))[:subject_limit]:
        for path in sorted(subject.glob("cnt_*.mat"))[:record_limit]:
            payload = _mat_payload(path)
            oxy = payload.oxy
            deoxy = payload.deoxy
            length = min(len(oxy.x), len(deoxy.x))
            stacked = np.stack(
                (np.asarray(oxy.x, dtype=np.float64)[:length], np.asarray(deoxy.x, dtype=np.float64)[:length]),
                axis=2,
            )
            stacked = _cap(stacked, max_samples)
            values = stacked.reshape(stacked.shape[0], -1)
            channel_names = [f"{label}_{role}" for label in _labels(oxy.clab) for role in ("Oxy", "Deoxy")]
            yield CleanInputRecord(
                dataset_id=contract.dataset_id,
                subject=subject.name,
                record_id=path.stem,
                source_paths=(path,),
                values=values,
                homer2_input=values,
                sample_rate_hz=float(oxy.fs),
                contract=contract,
                entry_stage="chromophore",
                wavelengths_nm=(760.0, 850.0),
                channel_names=tuple(channel_names),
                homer2_channel_names=tuple(channel_names),
                metadata={
                    "metadata_unit": str(getattr(oxy, "yUnit", "")),
                    "metadata_signal": str(getattr(oxy, "signal", "")),
                },
            )


def iter_refed(
    root: Path,
    subject_limit: int,
    record_limit: int,
    max_samples: int,
    include_absorbance: bool,
) -> Iterator[CleanInputRecord]:
    contracts = DATASET_FNIRS_CONTRACTS["refed"]
    signal_specs = [("hbo_hbr", (0, 1), "chromophore", ())]
    if include_absorbance:
        signal_specs.append(("absorbance_780_805_830", (3, 4, 5), "absorbance", (780.0, 805.0, 830.0)))
    subjects = sorted((root / "data").glob("[0-9]*"), key=lambda item: int(item.name))[:subject_limit]
    for subject in subjects:
        path = subject / "fNIRS_videos.mat"
        payload = loadmat(path, variable_names=[f"video_{i}" for i in range(1, record_limit + 1)])
        keys = sorted((name for name in payload if name.startswith("video_")), key=lambda name: int(name.split("_")[1]))
        for key in keys:
            tensor = np.asarray(payload[key], dtype=np.float64)
            for signal_key, indices, entry_stage, wavelengths in signal_specs:
                selected = tensor[list(indices)].transpose(2, 1, 0)
                selected = _cap(selected, max_samples)
                values = selected.reshape(selected.shape[0], -1)
                channel_names = tuple(f"CH{channel + 1}_{role}" for channel in range(selected.shape[1]) for role in contracts[signal_key].channel_roles)
                yield CleanInputRecord(
                    dataset_id="refed",
                    subject=subject.name,
                    record_id=f"{key}_{signal_key}",
                    source_paths=(path,),
                    values=values,
                    homer2_input=values,
                    sample_rate_hz=47.62,
                    contract=contracts[signal_key],
                    entry_stage=entry_stage,
                    wavelengths_nm=wavelengths,
                    channel_names=channel_names,
                    homer2_channel_names=channel_names,
                    metadata={"metadata_unit": contracts[signal_key].native_unit, "metadata_signal": signal_key},
                )


def _read_etg_csv(path: Path) -> tuple[np.ndarray, float]:
    recording = read_visual_fnirs_csv(path, load_signals=True)
    return recording["values"], recording["sample_rate_hz"]


def iter_visual(root: Path, subject_limit: int, record_limit: int, max_samples: int) -> Iterator[CleanInputRecord]:
    contract = DATASET_FNIRS_CONTRACTS["visual_cognitive_motivation"]["oxy_deoxy"]
    subjects = sorted(path for path in root.glob("S[0-9][0-9]") if (path / "fNIRS").exists())[:subject_limit]
    for subject in subjects:
        for oxy_path in sorted((subject / "fNIRS").glob("*Oxy.csv"))[:record_limit]:
            deoxy_path = Path(str(oxy_path).replace("_Oxy.csv", "_Deoxy.csv"))
            if not deoxy_path.exists():
                continue
            oxy_record = read_visual_fnirs_csv(oxy_path, load_signals=True)
            deoxy_record = read_visual_fnirs_csv(deoxy_path, load_signals=True)
            if (oxy_record["sample_rate_hz"] != deoxy_record["sample_rate_hz"]
                    or not np.array_equal(oxy_record["clock_s"], deoxy_record["clock_s"])
                    or oxy_record["marks"] != deoxy_record["marks"]):
                raise ValueError(f"Oxy/Deoxy clock or marker mismatch: {oxy_path}")
            stacked = np.stack((oxy_record["values"], deoxy_record["values"]), axis=2)
            stacked = _cap(stacked, max_samples)
            values = stacked.reshape(stacked.shape[0], -1)
            channel_names = tuple(f"CH{channel + 1}_{role}" for channel in range(stacked.shape[1]) for role in ("Oxy", "Deoxy"))
            yield CleanInputRecord(
                dataset_id=contract.dataset_id,
                subject=subject.name,
                record_id=oxy_path.stem.replace("_Oxy", ""),
                source_paths=(oxy_path, deoxy_path),
                values=values,
                homer2_input=values,
                sample_rate_hz=oxy_record["sample_rate_hz"],
                contract=contract,
                entry_stage="chromophore",
                wavelengths_nm=(695.0, 830.0),
                channel_names=channel_names,
                homer2_channel_names=channel_names,
                metadata={"metadata_unit": contract.native_unit, "metadata_signal": "ETG-7100 Oxy/Deoxy export",
                          "clock_policy": "native_Time_column_resampled_before_filtering",
                          "native_clock_origin_s": float(oxy_record["clock_s"][0])},
                native_time_s=oxy_record["time_s"][:len(values)],
            )


def iter_records(args: argparse.Namespace) -> Iterator[CleanInputRecord]:
    if "eeg_fnirs_single_trial" in args.datasets:
        yield from iter_single_trial(DATA_ROOTS["eeg_fnirs_single_trial"], args.subjects_per_dataset, args.records_per_subject, args.max_samples)
    if "refed" in args.datasets:
        yield from iter_refed(DATA_ROOTS["refed"], args.subjects_per_dataset, args.records_per_subject, args.max_samples, args.include_refed_absorbance)
    if "visual_cognitive_motivation" in args.datasets:
        yield from iter_visual(DATA_ROOTS["visual_cognitive_motivation"], args.subjects_per_dataset, args.records_per_subject, args.max_samples)
    if "simultaneous_eeg_nirs" in args.datasets:
        yield from iter_simultaneous(DATA_ROOTS["simultaneous_eeg_nirs"], args.subjects_per_dataset, args.records_per_subject, args.max_samples)


def _summarize_array(values: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(values.shape[0], -1)
    return {
        "shape": list(values.shape),
        "finite_fraction": float(np.isfinite(flat).mean()),
        "channel_std_median": float(np.median(np.nanstd(flat, axis=0))),
        "absolute_p99": float(np.nanquantile(np.abs(flat), 0.99)),
    }


def build_record(record: CleanInputRecord, output_dir: Path, overwrite: bool) -> dict[str, Any]:
    subject_dir = output_dir / record.dataset_id / _safe_name(record.subject)
    record_name = _safe_name(record.record_id)
    npz_path = subject_dir / f"{record_name}.npz"
    manifest_path = subject_dir / f"{record_name}.manifest.json"
    if npz_path.exists() and manifest_path.exists() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require_current_cache_manifest(manifest)
        return manifest
    if manifest_path.exists():
        require_current_cache_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))

    values, homer2_input = record.values, record.homer2_input
    native_time_s = (np.asarray(record.native_time_s, dtype=np.float64) if record.native_time_s is not None
                     else np.arange(len(values), dtype=np.float64) / record.sample_rate_hz)
    time_s = np.arange(len(values), dtype=np.float64) / record.sample_rate_hz
    if record.native_time_s is not None:
        time_s = np.arange(int(np.floor(native_time_s[-1] * record.sample_rate_hz)) + 1) / record.sample_rate_hz
        values = np.column_stack([np.interp(time_s, native_time_s, c) for c in values.T])
        homer2_input = np.column_stack([np.interp(time_s, native_time_s, c) for c in homer2_input.T])

    raw_native = standardize_fnirs_record(
        values,
        sample_rate_hz=record.sample_rate_hz,
        contract=record.contract,
    )
    homer2 = apply_homer2_aligned_contract(
        homer2_input,
        dataset_id=record.dataset_id,
        sample_rate_hz=record.sample_rate_hz,
        entry_stage=record.entry_stage,
        wavelengths_nm=record.wavelengths_nm,
    )
    save_npz(
        npz_path,
        raw_native_fnirs=raw_native.values,
        homer2_aligned_fnirs=homer2.values,
        native_input_fnirs=np.asarray(record.values, dtype=np.float32),
        time_s=time_s,
        native_input_time_s=native_time_s,
        native_channel_names=np.asarray(record.channel_names, dtype=str),
        homer2_channel_names=np.asarray(record.homer2_channel_names, dtype=str),
    )
    source_files = [
        {
            "path": str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
            "sha256": _file_hash(path),
        }
        for path in record.source_paths
    ]
    manifest = with_canonical_fields({
        "schema": CLEAN_CACHE_SCHEMA,
        "record_npz": str(npz_path.relative_to(PROJECT_ROOT) if npz_path.is_relative_to(PROJECT_ROOT) else npz_path),
        "dataset_id": record.dataset_id,
        "subject": record.subject,
        "record_id": record.record_id,
        "sample_rate_hz": record.sample_rate_hz,
        "source_files": source_files,
        "native_contract": record.contract.to_dict(),
        "metadata": record.metadata,
        "native_channel_names": list(record.channel_names),
        "homer2_channel_names": list(record.homer2_channel_names),
        "raw_native_contract": {
            "array_key": "raw_native_fnirs",
            "summary": _summarize_array(raw_native.values),
            "standardization_state": raw_native.state.to_dict(),
            "quality": dict(raw_native.quality),
        },
        "homer2_aligned_contract": {
            "array_key": "homer2_aligned_fnirs",
            "summary": _summarize_array(homer2.values),
            "alignment_state": homer2.state.to_dict(),
            "quality": dict(homer2.quality),
        },
    })
    write_json(manifest_path, _jsonable(manifest), ensure_ascii=False)
    return manifest


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    old_manifest = output_dir / "cache_manifest.json"
    if old_manifest.exists():
        require_current_cache_manifest(json.loads(old_manifest.read_text()))
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [build_record(record, output_dir, args.overwrite) for record in iter_records(args)]
    cache_manifest = {
        "schema": CLEAN_CACHE_SCHEMA,
        "homer2_alignment_schema": HOMER2_ALIGNMENT_SCHEMA,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT) if output_dir.is_relative_to(PROJECT_ROOT) else output_dir),
        "canonical_join_contract": {
            "schema": "clean_physiology_cache_index_v1",
            "key_fields": ["dataset_id", "canonical_subject_id", "base_record_id"],
            "join_key": "dataset_id|canonical_subject_id|base_record_id",
            "signal_branch": "separates multiple signal exports for the same canonical record",
        },
        "parameters": {
            "datasets": args.datasets,
            "subjects_per_dataset": args.subjects_per_dataset,
            "records_per_subject": args.records_per_subject,
            "max_samples": args.max_samples,
            "include_refed_absorbance": bool(args.include_refed_absorbance),
        },
        "homer2_compatibility": homer2_compatibility_manifest(),
        "records": records,
        "record_count": len(records),
    }
    write_json(output_dir / "cache_manifest.json", _jsonable(cache_manifest), ensure_ascii=False)
    print(json.dumps({"output_dir": str(output_dir), "records": len(records)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
