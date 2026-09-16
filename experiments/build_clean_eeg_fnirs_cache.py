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
import os
import time
from typing import Any, Iterator, Sequence

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fnirs_standardization import DATASET_FNIRS_CONTRACTS, FNIRSMeasurementContract, standardize_fnirs_record  # noqa: E402
from src.data.homer2_preprocessing import (  # noqa: E402
    HOMER2_ALIGNMENT_SCHEMA,
    MEASUREMENT_ALIGNMENT_SCHEMA,
    apply_homer2_aligned_contract,
    homer2_compatibility_manifest,
)
from src.data.clean_physiology_cache import (CLEAN_CACHE_SCHEMA, require_current_cache_manifest,
    with_canonical_fields, DEFAULT_CLEAN_CACHE_ROOT, MEASUREMENT_CACHE_STORAGE, CleanCacheRecord)  # noqa: E402
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
    parser.add_argument("--subjects-per-dataset", type=int, default=1000)
    parser.add_argument("--records-per-subject", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional leading sample cap for smoke runs.")
    parser.add_argument("--include-refed-absorbance", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_CLEAN_CACHE_ROOT)
    parser.add_argument("--storage", choices=[MEASUREMENT_CACHE_STORAGE, "legacy_npz"], default=MEASUREMENT_CACHE_STORAGE)
    parser.add_argument("--processing-schema", default=MEASUREMENT_ALIGNMENT_SCHEMA,
                        choices=[MEASUREMENT_ALIGNMENT_SCHEMA, HOMER2_ALIGNMENT_SCHEMA])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument('--ssm-training-config',type=Path,
                        help='Build only fresh float64 native inputs and timing for the configured 72-trial SSM revision.')
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


def iter_single_trial(root: Path, subject_limit: int, record_limit: int, max_samples: int,
                      *, subject_ids: Sequence[int] | None = None, session_ids: Sequence[int] | None = None) -> Iterator[CleanInputRecord]:
    contract = DATASET_FNIRS_CONTRACTS["eeg_fnirs_single_trial"]["wavelength_pair"]
    if subject_ids is not None and any(s not in range(1,24) for s in subject_ids):
        raise ValueError('Single-Trial protected subject boundary before file access')
    subjects = ([root/'NIRS_01-29'/f'subject {s:02d}' for s in subject_ids] if subject_ids is not None
                else [p for p in sorted((root / "NIRS_01-29").glob("subject *"))
                      if int(p.name.split()[-1]) < 24][:subject_limit])
    if any(int(s.name.split()[-1]) >= 24 for s in subjects):
        raise ValueError('Single-Trial protected subject boundary before file access')
    for subject in subjects:
        path = subject / "cnt.mat"
        sessions = np.atleast_1d(_mat_payload(path, "cnt"))
        for index, session in enumerate(sessions[:record_limit]):
            if session_ids is not None and index not in session_ids:
                continue
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
            if (np.shape(oxy.x) != np.shape(deoxy.x) or float(oxy.fs) != float(deoxy.fs)
                    or _labels(oxy.clab) != _labels(deoxy.clab)
                    or str(getattr(oxy, 'yUnit', '')) != str(getattr(deoxy, 'yUnit', ''))):
                raise ValueError(f"published Hb pair shape/clock/channel/unit disagreement: {path}")
            length = len(oxy.x)
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
                          "device_metadata":oxy_record['device_metadata'],
                          "quality_annotations":oxy_record['quality_annotations'],
                          "quality_policy":oxy_record['quality_policy'],
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


def build_record(record: CleanInputRecord, output_dir: Path, overwrite: bool,
                 processing_schema: str = MEASUREMENT_ALIGNMENT_SCHEMA, *, native_only: bool = False,
                 storage: str = "legacy_npz") -> dict[str, Any]:
    subject_dir = output_dir / record.dataset_id / _safe_name(record.subject)
    record_name = _safe_name(record.record_id)
    npz_path = subject_dir / f"{record_name}.npz"
    manifest_path = subject_dir / f"{record_name}.manifest.json"
    if storage == MEASUREMENT_CACHE_STORAGE:
        if native_only or processing_schema != MEASUREMENT_ALIGNMENT_SCHEMA:
            raise ValueError("Complete measurement storage requires the current measurement producer")
        npz_path = subject_dir / record_name
    if npz_path.exists() and manifest_path.exists() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require_current_cache_manifest(manifest)
        if manifest.get('processing_schema') != processing_schema:
            raise ValueError('processing version differs; use a new cache namespace')
        if bool(manifest.get('native_only', False)) != native_only:
            raise ValueError('native-only and processed caches require separate namespaces')
        if manifest.get('storage', 'legacy_npz') != storage:
            raise ValueError('storage migration requires a new namespace')
        return manifest
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        require_current_cache_manifest(existing)
        if existing.get('processing_schema') != processing_schema:
            raise ValueError('cannot upgrade a retained processing version in place')
        if bool(existing.get('native_only', False)) != native_only:
            raise ValueError('cannot replace native-only/processed cache identity in place')
        if existing.get('storage', 'legacy_npz') != storage:
            raise ValueError('cannot replace retained storage identity in place')

    if native_only:
        if record.dataset_id != 'eeg_fnirs_single_trial':
            raise ValueError('native-only cache is scoped to original Single-Trial SSM input')
        save_npz(npz_path,native_input_fnirs=np.asarray(record.values,dtype=np.float64),
                 native_channel_names=np.asarray(record.channel_names,dtype=str))
        manifest = with_canonical_fields(dict(schema=CLEAN_CACHE_SCHEMA,processing_schema=processing_schema,
            record_npz=str(npz_path.resolve()),dataset_id=record.dataset_id,subject=record.subject,
            record_id=record.record_id,sample_rate_hz=record.sample_rate_hz,
            native_contract=record.contract.to_dict(),metadata=record.metadata,
            preprocessing_scope='none; original training trials sliced before any nonlinear processing',
            native_only=True,source_files=[dict(path=str(p),sha256=_file_hash(p)) for p in record.source_paths]))
        write_json(manifest_path,_jsonable(manifest),ensure_ascii=False)
        return manifest

    values, homer2_input = record.values, record.homer2_input
    native_time_s = (np.asarray(record.native_time_s, dtype=np.float64) if record.native_time_s is not None
                     else np.arange(len(values), dtype=np.float64) / record.sample_rate_hz)
    time_s = np.arange(len(values), dtype=np.float64) / record.sample_rate_hz
    if record.native_time_s is not None:
        time_s = np.arange(int(np.floor(native_time_s[-1] * record.sample_rate_hz)) + 1) / record.sample_rate_hz
        values = np.column_stack([np.interp(time_s, native_time_s, c) for c in values.T])
        homer2_input = np.column_stack([np.interp(time_s, native_time_s, c) for c in homer2_input.T])

    raw_native = None if storage == MEASUREMENT_CACHE_STORAGE else standardize_fnirs_record(
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
        processing_schema=processing_schema,
        native_unit=str(record.metadata.get('metadata_unit', 'unknown')),
        unit_evidence=('MAT cnt.yUnit' if record.dataset_id in ('eeg_fnirs_single_trial', 'simultaneous_eeg_nirs')
                       and record.metadata.get('metadata_unit') else ''),
    )
    if storage == MEASUREMENT_CACHE_STORAGE:
        return build_measurement_record(record, homer2, npz_path, manifest_path, processing_schema)
    save_npz(
        npz_path,
        raw_native_fnirs=raw_native.values,
        homer2_aligned_fnirs=homer2.values,
        native_input_fnirs=np.asarray(record.values, dtype=np.float64),
        native_valid_mask=np.isfinite(record.values) & (record.values > 0 if record.entry_stage == 'raw_intensity' else True),
        processed_valid_mask=homer2.processed_valid_mask,
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
        "processing_schema": processing_schema,
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


def build_measurement_record(record, homer2, directory, manifest_path, processing_schema):
    """Persist complete measured coordinates once; readers mmap and slice only."""
    from src.data.unified_physiology import (
        load_native_eeg_record, preprocess_eeg_record_with_quality, preprocess_fnirs_record,
        canonical_fnirs_channel_names, fnirs_component_roles,
        SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4, SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1,
    )
    row = with_canonical_fields(dict(dataset_id=record.dataset_id, subject=record.subject,
                                    record_id=record.record_id))
    ref = CleanCacheRecord(**{k: row[k] for k in (
        'dataset_id', 'subject', 'record_id', 'canonical_subject_id', 'base_record_id', 'signal_branch', 'join_key')},
        sample_rate_hz=record.sample_rate_hz, npz_path=directory, manifest={})
    native = load_native_eeg_record(PROJECT_ROOT, ref)
    branch = (SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4 if record.dataset_id == 'eeg_fnirs_single_trial'
              else SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1 if record.dataset_id == 'simultaneous_eeg_nirs'
              else 'raw_with_ocular_artifact')
    eeg, eeg_state, quality = preprocess_eeg_record_with_quality(native, signal_branch=branch,
                                                               output_coordinate='measurement')
    native_contract = homer2.quality.get('unit_conversion', homer2.quality.get('modified_beer_lambert', {}))
    fnirs, fnirs_state = preprocess_fnirs_record(homer2.values, sample_rate_hz=record.sample_rate_hz,
                                               native_contract=native_contract, output_coordinate='measurement')
    names = canonical_fnirs_channel_names(record.homer2_channel_names)
    if set(fnirs_component_roles(names)) != {'HbO', 'HbR'}:
        raise ValueError('Complete measurement cache accepts paired HbO/HbR only')
    arrays = dict(eeg=eeg, fnirs=fnirs,
        eeg_supported_channels=quality['processed_valid_mask'].all(axis=0),
        fnirs_supported_channels=homer2.processed_valid_mask.all(axis=0),
        eeg_bad_channel_mask=quality['bad_channel_mask'])
    directory.mkdir(parents=True, exist_ok=True)
    for key, value in arrays.items():
        temporary = directory / f'{key}.partial.npy'
        np.save(temporary, value, allow_pickle=False)
        os.replace(temporary, directory / f'{key}.npy')
    manifest = dict(row, schema=CLEAN_CACHE_SCHEMA, storage=MEASUREMENT_CACHE_STORAGE,
        processing_schema=processing_schema, record_npz=str(directory.resolve()),
        arrays={key:f'{key}.npy' for key in arrays}, array_shapes={k:list(v.shape) for k,v in arrays.items()},
        sample_rate_hz=10., native_sample_rate_hz=record.sample_rate_hz,
        native_contract=record.contract.to_dict(), metadata=record.metadata,
        source_files=[dict(path=str(p),sha256=_file_hash(p)) for p in dict.fromkeys((*record.source_paths,native.source_path))],
        measurement=dict(eeg_channel_names=native.channel_names, fnirs_channel_names=names,
            fnirs_component_roles=fnirs_component_roles(names), eeg_preprocessing_state=eeg_state,
            fnirs_preprocessing_state=fnirs_state),
        homer2_aligned_contract=dict(alignment_state=homer2.state.to_dict(),quality=homer2.quality),
        preprocessing_scope='offline continuous record; no learned population scale',
        execution='completed', created_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    temporary = manifest_path.with_suffix('.partial.json')
    write_json(temporary, _jsonable(manifest), ensure_ascii=False)
    os.replace(temporary, manifest_path)
    return _jsonable(manifest)


def build_ssm_training_inputs(config_path: Path) -> dict[str, Any]:
    """Use the existing scope controller and readers; never preprocess held-out trials."""
    from experiments import evaluate_ssm_overnight_diagnostics as suite
    from experiments import build_clean_event_index as event_builder
    cfg,_,_,_,_ = suite.load_config(config_path)
    revision = cfg.get('measurement_revision')
    if revision is None:
        raise ValueError('fresh native rebuild requires a versioned measurement revision')
    destination = PROJECT_ROOT/revision['cache_root']
    if destination.exists():
        raise FileExistsError('native revision namespace already exists; retained inputs are immutable')
    subjects = [int(s.rsplit('_',1)[1]) for s in cfg['subjects']]
    sessions = [int(s.rsplit('_',1)[1]) for s in cfg['sessions']]
    root = DATA_ROOTS['eeg_fnirs_single_trial']
    events,reports = event_builder.iter_single_trial(root,3,6,subject_ids=subjects,session_ids=sessions)
    # The signal-free admission inventory is fixed before source arrays are opened.
    destination.mkdir(parents=True)
    event_dir = destination/'event_index';event_dir.mkdir()
    for filename,rows in [('events.jsonl',events),('alignment_reports.jsonl',reports)]:
        (event_dir/filename).write_text(''.join(json.dumps(_jsonable(with_canonical_fields(r.to_dict())))+'\n' for r in rows))
    write_json(event_dir/'event_manifest.json',dict(schema=event_builder.EVENT_INDEX_SCHEMA,
        event_alignment_schema=event_builder.EVENT_ALIGNMENT_SCHEMA,source_config=str(config_path)))
    records = [build_record(r,destination,False,revision['processing_schema'],native_only=True)
               for r in iter_single_trial(root,3,6,0,subject_ids=subjects,session_ids=sessions)]
    manifest = dict(schema=CLEAN_CACHE_SCHEMA,processing_schema=revision['processing_schema'],
        records=records,source_config=str(config_path),native_only=True,
        preprocessing_scope='none; only scoped original training windows processed by owning SSM loader')
    write_json(destination/'cache_manifest.json',_jsonable(manifest))
    return dict(records=len(records),events=len(events),output_dir=str(destination))


def main() -> None:
    args = parse_args()
    if getattr(args, 'ssm_training_config', None):
        print(json.dumps(build_ssm_training_inputs(args.ssm_training_config)))
        return
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    old_manifest = output_dir / "cache_manifest.json"
    if old_manifest.exists():
        require_current_cache_manifest(json.loads(old_manifest.read_text()))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.storage == MEASUREMENT_CACHE_STORAGE and args.include_refed_absorbance:
        raise ValueError('Measurement inputs select HbO/HbR; absorbance is a separate source quantity')
    old_records = json.loads(old_manifest.read_text()).get('records', []) if old_manifest.exists() else []
    if old_manifest.exists() and json.loads(old_manifest.read_text()).get('storage', 'legacy_npz') != args.storage:
        raise ValueError('Cache storage changed; use a new root')
    records = [r for r in old_records if r['dataset_id'] not in args.datasets]
    write_json(old_manifest, dict(schema=CLEAN_CACHE_SCHEMA, storage=args.storage, execution='building', records=records))
    for record in iter_records(args):
        started = time.monotonic()
        manifest = build_record(record, output_dir, args.overwrite, args.processing_schema, storage=args.storage)
        records.append(manifest)
        print(json.dumps(dict(record=manifest['join_key'],elapsed_seconds=time.monotonic()-started)), flush=True)
    cache_manifest = {
        "schema": CLEAN_CACHE_SCHEMA,
        "storage": args.storage,
        "execution": "completed",
        "processing_schema": args.processing_schema,
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
    temporary = output_dir / 'cache_manifest.partial.json'
    write_json(temporary, _jsonable(cache_manifest), ensure_ascii=False)
    os.replace(temporary, old_manifest)
    print(json.dumps({"output_dir": str(output_dir), "records": len(records)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
