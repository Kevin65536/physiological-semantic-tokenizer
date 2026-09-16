"""Unified, provenance-preserving access to the four raw physiology datasets.

The Croce source/observation caches are deliberately absent from this module:
they are derived supervision targets, not an additional dataset.  This loader
joins the clean fNIRS cache to the original EEG recordings, applies one
canonical preprocessing contract, and returns event-aligned multimodal
windows with a common schema.

Canonical numerical coordinates are dimensionless robust standard deviations.
Native units and measurement families remain in metadata; the transform does
not claim that volts and chromophore concentration are physically identical.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import csv
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt

from .clean_physiology_cache import CleanCacheRecord, CleanPhysiologyCacheIndex
from .event_alignment import ADMISSIBLE_ALIGNMENT_CASES, window_within_alignment_support
from .eeg_artifact_preprocessing import (
    EEGArtifactCleaningConfig,
    SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA,
    SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V2,
    SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V3,
    SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4,
    clean_single_trial_eeg,
)


RAW_DATASET_IDS: tuple[str, ...] = (
    "eeg_fnirs_single_trial",
    "refed",
    "visual_cognitive_motivation",
    "simultaneous_eeg_nirs",
)

UNIFIED_PHYSIOLOGY_SCHEMA = "unified_physiology_window_v2"
EEG_ARTIFACT_MASK_POLICY = "disabled_all_false_no_invalid_authority_v1"
REFED_CONTINUOUS_SEQUENCE_SCHEMA = "refed_continuous_va_sequence_v2"
REFED_CONTINUOUS_TARGET_NAMES = ("valence", "arousal")
REFED_DEFAULT_TARGET_SAMPLE_RATE_HZ = 1.0
CANONICAL_UNIT = "robust_standard_deviation"
CANONICAL_EEG_SAMPLE_RATE_HZ = 200.0
CANONICAL_FNIRS_SAMPLE_RATE_HZ = 10.0
CANONICAL_EEG_BAND_HZ = (1.0, 45.0)
CANONICAL_FNIRS_BAND_HZ = (0.01, 0.2)
CANONICAL_FNIRS_COMPONENTS = ("HbO", "HbR")
DEFAULT_UNIFIED_WINDOW_DURATION_S = 20.0
SINGLE_TRIAL_EEG_SIGNAL_BRANCHES = (
    "raw_with_ocular_artifact",
    SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V2,
    SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V3,
    SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4,
)
SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1 = "simultaneous_eeg_eog_clean_v1"
SUPPORTED_EEG_SIGNAL_BRANCHES = SINGLE_TRIAL_EEG_SIGNAL_BRANCHES + (
    SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1,
)
DEFAULT_ADMISSIBLE_ALIGNMENT_CASES = ADMISSIBLE_ALIGNMENT_CASES
FORBIDDEN_TASK_NAMESPACES: frozenset[str] = frozenset()
FORBIDDEN_TASK_POLICY = "no_hard_exclusions_dsr_restored_v2"

VISUAL_CONDITION_INDICES = {"RR": 0, "RF": 1, "FF": 2, "FR": 3, "unknown": -1}
_REQUIRE_SINGLE_TRIAL_EEG_ARTIFACT_CACHE: ContextVar[bool] = ContextVar(
    "require_single_trial_eeg_artifact_cache",
    default=False,
)


@contextmanager
def require_single_trial_eeg_artifact_cache() -> Iterable[None]:
    """Make nested unified datasets fail closed on artifact-cache fallback."""

    token = _REQUIRE_SINGLE_TRIAL_EEG_ARTIFACT_CACHE.set(True)
    try:
        yield
    finally:
        _REQUIRE_SINGLE_TRIAL_EEG_ARTIFACT_CACHE.reset(token)


def single_trial_eeg_artifact_cache_required() -> bool:
    """Report the task-local strict-cache requirement for audit and tests."""

    return bool(_REQUIRE_SINGLE_TRIAL_EEG_ARTIFACT_CACHE.get())


def simultaneous_eeg_eog_cleaning_config(
    base: EEGArtifactCleaningConfig | None = None,
) -> EEGArtifactCleaningConfig:
    """Return the conservative, EOG-only Simultaneous dataset contract."""
    return replace(
        base or EEGArtifactCleaningConfig(),
        schema=SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1,
        bad_channel_robust_z=1.0e12,
        bad_channel_extreme_robust_z=1.0e12,
        max_bad_channel_fraction=0.0,
        line_noise_frequency_hz=None,
        bad_channel_action="disabled_for_cross_dataset_uniformity",
        high_frequency_window_robust_z=1.0e12,
        muscle_action="mask_only",
        reference_strategy="native_reference_preserved_eog_auxiliary_excluded",
        calibration_scope="record_robust_distribution_simultaneous_eog_v1",
    )


@dataclass(frozen=True)
class CanonicalPreprocessingContract:
    canonical_unit: str = CANONICAL_UNIT
    eeg_sample_rate_hz: float = CANONICAL_EEG_SAMPLE_RATE_HZ
    fnirs_sample_rate_hz: float = CANONICAL_FNIRS_SAMPLE_RATE_HZ
    eeg_band_hz: tuple[float, float] = CANONICAL_EEG_BAND_HZ
    fnirs_band_hz: tuple[float, float] = CANONICAL_FNIRS_BAND_HZ
    eeg_steps: tuple[str, ...] = (
        "finite_interpolation",
        "bandpass_1_45_hz",
        "resample_200_hz",
        "full_record_channel_median_mad",
    )
    fnirs_steps: tuple[str, ...] = (
        "dataset_available_homer2_aligned_hbo_hbr",
        "resample_10_hz",
        "full_record_channel_median_mad",
    )
    schema: str = UNIFIED_PHYSIOLOGY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("eeg_band_hz", "fnirs_band_hz", "eeg_steps", "fnirs_steps"):
            payload[key] = list(payload[key])
        return payload


CANONICAL_PREPROCESSING = CanonicalPreprocessingContract()


@dataclass(frozen=True)
class NativeEEGRecord:
    values: np.ndarray
    sample_rate_hz: float
    channel_names: tuple[str, ...]
    native_unit: str
    source_path: Path
    auxiliary_values: np.ndarray | None = None
    auxiliary_channel_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnifiedWindowRef:
    record: CleanCacheRecord
    event: Mapping[str, Any]
    window_offset_s: float = 0.0


def _first_mat_value(path: Path, key: str | None = None) -> Any:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    if key is None:
        key = next(name for name in payload if not name.startswith("__"))
    value = payload[key]
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        value = value.item()
    return value


def _labels(value: Any) -> tuple[str, ...]:
    return tuple(str(item).strip() for item in np.asarray(value, dtype=object).reshape(-1).tolist())


def canonical_channel_name(value: str) -> str:
    """Return a stable channel label without destroying 10-20 case semantics."""
    return re.sub(r"\s+", "", str(value).strip())


def canonical_fnirs_channel_names(names: Sequence[str]) -> tuple[str, ...]:
    output = []
    for name in names:
        value = canonical_channel_name(name)
        value = re.sub(r"_(Oxy|Hbo)$", "_HbO", value, flags=re.IGNORECASE)
        value = re.sub(r"_(Deoxy|Hbr|Hb)$", "_HbR", value, flags=re.IGNORECASE)
        output.append(value)
    return tuple(output)


def fnirs_component_roles(names: Sequence[str]) -> tuple[str, ...]:
    roles = []
    for name in canonical_fnirs_channel_names(names):
        if name.endswith("_HbO"):
            roles.append("HbO")
        elif name.endswith("_HbR"):
            roles.append("HbR")
        else:
            roles.append("unknown")
    return tuple(roles)


def _as_time_channels(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"expected [time, channels], got {array.shape}")
    return array


def _interpolate_nonfinite(values: np.ndarray) -> tuple[np.ndarray, int]:
    output = _as_time_channels(values).copy()
    repaired = 0
    time = np.arange(output.shape[0], dtype=np.float64)
    for channel in range(output.shape[1]):
        finite = np.isfinite(output[:, channel])
        repaired += int(np.count_nonzero(~finite))
        if not np.any(finite):
            output[:, channel] = 0.0
        elif not np.all(finite):
            output[:, channel] = np.interp(time, time[finite], output[finite, channel])
    return output, repaired


def _bandpass(values: np.ndarray, sample_rate_hz: float, band_hz: tuple[float, float]) -> np.ndarray:
    low, high = band_hz
    nyquist = float(sample_rate_hz) * 0.5
    high = min(float(high), nyquist * 0.95)
    if low <= 0 or high <= low or values.shape[0] < 32:
        return values
    sos = butter(4, [float(low) / nyquist, high / nyquist], btype="bandpass", output="sos")
    try:
        return sosfiltfilt(sos, values, axis=0)
    except ValueError:
        return values


def _resample(values: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    if abs(float(source_hz) - float(target_hz)) < 1e-9:
        return values
    ratio = Fraction(float(target_hz) / float(source_hz)).limit_denominator(10_000)
    return resample_poly(values, ratio.numerator, ratio.denominator, axis=0)


def _robust_standardize(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    array, repaired = _interpolate_nonfinite(values)
    median = np.median(array, axis=0)
    mad = 1.482602218505602 * np.median(np.abs(array - median[None, :]), axis=0)
    q25, q75 = np.quantile(array, [0.25, 0.75], axis=0)
    iqr = (q75 - q25) / 1.3489795003921634
    std = np.std(array, axis=0)
    scale = np.where(np.isfinite(mad) & (mad > 1e-8), mad, iqr)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, std)
    flat = ~(np.isfinite(scale) & (scale > 1e-8))
    scale = np.where(flat, 1.0, scale)
    canonical = (array - median[None, :]) / scale[None, :]
    return canonical.astype(np.float32), {
        "repaired_nonfinite_samples": int(repaired),
        "flat_channel_count": int(np.count_nonzero(flat)),
        "channel_location": median.astype(float).tolist(),
        "channel_scale": scale.astype(float).tolist(),
        "canonical_unit": CANONICAL_UNIT,
    }


def _resample_boolean_mask(mask: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    array = np.asarray(mask, dtype=bool).reshape(-1)
    target_length = max(1, int(round(len(array) * float(target_hz) / float(source_hz))))
    source_indices = np.minimum(
        np.floor(np.arange(target_length) * float(source_hz) / float(target_hz)).astype(int),
        len(array) - 1,
    )
    return array[source_indices]


def preprocess_eeg_record_with_quality(
    record: NativeEEGRecord,
    *,
    signal_branch: str = "raw_with_ocular_artifact",
    artifact_config: EEGArtifactCleaningConfig | None = None,
    channel_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    if signal_branch not in SUPPORTED_EEG_SIGNAL_BRANCHES:
        raise ValueError(f"unsupported EEG signal branch: {signal_branch!r}")
    finite, repaired = _interpolate_nonfinite(record.values)
    artifact_mask = np.zeros(len(finite), dtype=bool)
    bad_channel_mask = np.zeros(finite.shape[1], dtype=bool)
    cleaning_state: dict[str, Any] = {
        "schema": "raw_with_ocular_artifact",
        "action": "no_artifact_removal",
    }
    if signal_branch in {
        SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V2,
        SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V3,
        SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4,
        SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1,
    }:
        if record.auxiliary_values is None or not record.auxiliary_channel_names:
            raise ValueError("EEG artifact-clean branches require retained EOG auxiliary channels")
        resolved_config = artifact_config or EEGArtifactCleaningConfig()
        if signal_branch == SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1:
            resolved_config = simultaneous_eeg_eog_cleaning_config(resolved_config)
        elif signal_branch == SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V2:
            resolved_config = replace(
                resolved_config,
                schema=SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V2,
                line_noise_frequency_hz=None,
                bad_channel_action="detect_and_interpolate",
                muscle_action="mask_only",
            )
        elif signal_branch == SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V3:
            resolved_config = replace(
                resolved_config,
                schema=SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V3,
                line_noise_frequency_hz=None,
                bad_channel_action="detect_and_interpolate",
                muscle_action="mask_gated_high_frequency_attenuation_v1",
            )
        else:
            resolved_config = replace(
                resolved_config,
                schema=SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4,
                line_noise_frequency_hz=50.0,
                bad_channel_action="disabled_for_cross_dataset_uniformity",
                muscle_action="mask_gated_high_frequency_attenuation_v1",
            )
        cleaned = clean_single_trial_eeg(
            finite,
            record.auxiliary_values,
            sample_rate_hz=record.sample_rate_hz,
            channel_names=record.channel_names,
            eog_channel_names=record.auxiliary_channel_names,
            channel_positions=channel_positions,
            config=resolved_config,
        )
        filtered = np.asarray(cleaned.cleaned_values, dtype=np.float64)
        bad_channel_mask = cleaned.bad_channel_mask
        cleaning_state = cleaned.state
        stride = max(1, len(filtered) // 20_000)
        preserved = ~np.asarray(cleaned.ocular_mask, dtype=bool)
        preserved_indices = np.flatnonzero(preserved)[::stride]
        if len(preserved_indices) < 2:
            preserved_indices = np.arange(0, len(filtered), stride)
        before_preserved = np.asarray(cleaned.filtered_raw_values, dtype=np.float64)[preserved_indices]
        after_preserved = filtered[preserved_indices]
        waveform_correlation = []
        for channel in range(filtered.shape[1]):
            correlation = np.corrcoef(
                before_preserved[:, channel], after_preserved[:, channel]
            )[0, 1]
            waveform_correlation.append(float(correlation) if np.isfinite(correlation) else 0.0)
        before_high = _bandpass(
            np.asarray(cleaned.filtered_raw_values, dtype=np.float64),
            record.sample_rate_hz,
            (15.0, 45.0),
        )[::stride]
        after_high = _bandpass(filtered, record.sample_rate_hz, (15.0, 45.0))[::stride]
        high_frequency_variance_ratio = np.var(after_high, axis=0) / np.maximum(
            np.var(before_high, axis=0), np.finfo(np.float64).eps
        )
        cleaning_state["information_preservation"] = {
            "scope": "samples_outside_detected_ocular_mask",
            "evaluated_sample_count": int(len(preserved_indices)),
            "waveform_correlation_by_channel": waveform_correlation,
            "median_waveform_correlation": float(np.median(waveform_correlation)),
            "minimum_waveform_correlation": float(np.min(waveform_correlation)),
            "high_frequency_band_hz": [15.0, 45.0],
            "high_frequency_variance_ratio_by_channel": high_frequency_variance_ratio.tolist(),
            "median_high_frequency_variance_ratio": float(np.median(high_frequency_variance_ratio)),
        }
    else:
        filtered = _bandpass(finite, record.sample_rate_hz, CANONICAL_EEG_BAND_HZ)
    resampled = _resample(filtered, record.sample_rate_hz, CANONICAL_EEG_SAMPLE_RATE_HZ)
    canonical, state = _robust_standardize(resampled)
    canonical_artifact_mask = _resample_boolean_mask(
        artifact_mask, record.sample_rate_hz, CANONICAL_EEG_SAMPLE_RATE_HZ
    )
    state.update({
        "native_unit": record.native_unit,
        "native_sample_rate_hz": float(record.sample_rate_hz),
        "canonical_sample_rate_hz": CANONICAL_EEG_SAMPLE_RATE_HZ,
        "filter_band_hz": list(CANONICAL_EEG_BAND_HZ),
        "filter_input_repaired_nonfinite_samples": int(repaired),
        "source_path": str(record.source_path),
        "signal_branch": signal_branch,
        "artifact_cleaning": cleaning_state,
        "artifact_mask_policy": EEG_ARTIFACT_MASK_POLICY,
    })
    return canonical, state, {
        "artifact_mask": canonical_artifact_mask,
        "bad_channel_mask": bad_channel_mask.astype(bool),
    }


def preprocess_eeg_record(record: NativeEEGRecord) -> tuple[np.ndarray, dict[str, Any]]:
    """Backward-compatible raw branch used by existing callers and tests."""
    canonical, state, _ = preprocess_eeg_record_with_quality(record)
    return canonical, state


def preprocess_fnirs_record(
    values: np.ndarray,
    *,
    sample_rate_hz: float,
    native_contract: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    # The clean-cache branch has already applied the best available common
    # 0.01-0.2 Hz/motion/HbO-HbR path.  Re-filtering would double-filter it.
    resampled = _resample(_as_time_channels(values), sample_rate_hz, CANONICAL_FNIRS_SAMPLE_RATE_HZ)
    canonical, state = _robust_standardize(resampled)
    state.update({
        "native_contract": dict(native_contract),
        "native_sample_rate_hz": float(sample_rate_hz),
        "canonical_sample_rate_hz": CANONICAL_FNIRS_SAMPLE_RATE_HZ,
        "filter_band_hz": list(CANONICAL_FNIRS_BAND_HZ),
        "input_branch": "homer2_aligned_fnirs",
    })
    return canonical, state


def _single_trial_eeg(project_root: Path, record: CleanCacheRecord) -> NativeEEGRecord:
    subject = record.canonical_subject_id.replace("subject_", "subject ")
    directory = project_root / "data/EEG+NIRS Single-Trial/EEG_01-29" / subject / "with occular artifact"
    if not directory.exists():
        directory = directory.parent
    path = directory / "cnt.mat"
    sessions = np.atleast_1d(_first_mat_value(path, "cnt"))
    session_index = int(record.base_record_id.rsplit("_", 1)[-1])
    session = sessions[session_index]
    values = _as_time_channels(np.asarray(session.x, dtype=np.float64))
    names = _labels(session.clab)
    eeg_keep = np.asarray(["EOG" not in name.upper() for name in names], dtype=bool)
    eog_keep = ~eeg_keep
    return NativeEEGRecord(
        values=values[:, eeg_keep],
        sample_rate_hz=float(session.fs),
        channel_names=tuple(canonical_channel_name(name) for name, selected in zip(names, eeg_keep) if selected),
        native_unit=str(getattr(session, "yUnit", "uV") or "uV"),
        source_path=path,
        auxiliary_values=values[:, eog_keep],
        auxiliary_channel_names=tuple(canonical_channel_name(name) for name, selected in zip(names, eog_keep) if selected),
    )


def _simultaneous_eeg(project_root: Path, record: CleanCacheRecord) -> NativeEEGRecord:
    path = project_root / "data/Simultaneous EEG&NIRS" / f"{record.canonical_subject_id}-EEG" / f"{record.base_record_id}.mat"
    payload = _first_mat_value(path)
    values = _as_time_channels(np.asarray(payload.x, dtype=np.float64))
    names = _labels(payload.clab)
    auxiliary = np.asarray(["EOG" in name.upper() for name in names], dtype=bool)
    if not np.any(auxiliary):
        raise ValueError(f"Simultaneous EEG record has no HEOG/VEOG reference channels: {path}")
    return NativeEEGRecord(
        values=values[:, ~auxiliary],
        sample_rate_hz=float(payload.fs),
        channel_names=tuple(
            canonical_channel_name(name) for name, selected in zip(names, ~auxiliary) if selected
        ),
        native_unit=str(getattr(payload, "yUnit", "uV") or "uV"),
        source_path=path,
        auxiliary_values=values[:, auxiliary],
        auxiliary_channel_names=tuple(
            canonical_channel_name(name) for name, selected in zip(names, auxiliary) if selected
        ),
    )


def _refed_eeg_names(project_root: Path, count: int) -> tuple[str, ...]:
    path = project_root / "data/REFED-dataset/EEG_channels.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        names = [canonical_channel_name(row.get("ch_name", "")) for row in csv.DictReader(handle)]
    return tuple(names[:count]) if len(names) >= count else tuple(f"EEG{index + 1}" for index in range(count))


def _refed_eeg(project_root: Path, record: CleanCacheRecord) -> NativeEEGRecord:
    path = project_root / "data/REFED-dataset/data" / record.canonical_subject_id / "EEG_videos.mat"
    values = np.asarray(_first_mat_value(path, record.base_record_id), dtype=np.float64)
    if values.shape[0] <= values.shape[1]:
        values = values.T
    return NativeEEGRecord(
        values=_as_time_channels(values),
        sample_rate_hz=1000.0,
        channel_names=_refed_eeg_names(project_root, values.shape[1]),
        native_unit="V",
        source_path=path,
    )


def _read_edf(path: Path) -> tuple[np.ndarray, float, list[str], list[str]]:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        header_bytes = int(fixed[184:192].decode().strip())
        records = int(fixed[236:244].decode().strip())
        duration = float(fixed[244:252].decode().strip())
        signal_count = int(fixed[252:256].decode().strip())
        header = handle.read(header_bytes - 256)
    cursor = 0
    fields: list[list[str]] = []
    for width in (16, 80, 8, 8, 8, 8, 8, 80, 8, 32):
        fields.append([
            header[cursor + index * width : cursor + (index + 1) * width].decode(errors="replace").strip()
            for index in range(signal_count)
        ])
        cursor += width * signal_count
    labels, units = fields[0], fields[2]
    physical_min = np.asarray([float(value) for value in fields[3]])
    physical_max = np.asarray([float(value) for value in fields[4]])
    digital_min = np.asarray([float(value) for value in fields[5]])
    digital_max = np.asarray([float(value) for value in fields[6]])
    samples_per_record = np.asarray([int(value) for value in fields[8]], dtype=int)
    candidates = [
        index for index, label in enumerate(labels)
        if "ANNOTATION" not in label.upper() and label.upper() not in {"A64", "DC9", "DC09"}
    ]
    sample_rate = float(max(samples_per_record[index] / duration for index in candidates))
    offsets = np.concatenate(([0], np.cumsum(samples_per_record)))
    raw = np.memmap(path, dtype="<i2", mode="r", offset=header_bytes, shape=(records, int(samples_per_record.sum())))
    channels = []
    for index in candidates:
        digital = np.asarray(raw[:, offsets[index] : offsets[index + 1]], dtype=np.float64).reshape(-1)
        denominator = max(digital_max[index] - digital_min[index], 1.0)
        physical = (digital - digital_min[index]) * (physical_max[index] - physical_min[index]) / denominator + physical_min[index]
        channel_rate = samples_per_record[index] / duration
        physical = _resample(physical[:, None], channel_rate, sample_rate)[:, 0]
        channels.append(physical)
    length = min(map(len, channels))
    return (
        np.stack([channel[:length] for channel in channels], axis=1),
        sample_rate,
        [labels[index] for index in candidates],
        [units[index] or "unknown" for index in candidates],
    )


def _visual_eeg_name_map(project_root: Path) -> dict[str, str]:
    path = project_root / "data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults/Location.ced"
    rows = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    header = [value.strip() for value in rows[0].split("\t")]
    mapping: dict[str, str] = {}
    for line in rows[1:]:
        values = line.split("\t")
        row = {key: values[index].strip() if index < len(values) else "" for index, key in enumerate(header)}
        if row.get("Number") and row.get("labels"):
            mapping[f"A{int(float(row['Number']))}"] = canonical_channel_name(row["labels"])
    return mapping


def _visual_eeg(project_root: Path, record: CleanCacheRecord) -> NativeEEGRecord:
    root = project_root / "data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults"
    raw_dir = root / record.canonical_subject_id / "EEG/raw"
    part = re.search(r"Part(\d+)", record.base_record_id, flags=re.IGNORECASE)
    if part:
        candidates = sorted(raw_dir.glob(f"*part{part.group(1)}.edf"))
    else:
        candidates = sorted(raw_dir.glob(f"{record.canonical_subject_id}.edf"))
    if not candidates:
        candidates = sorted(raw_dir.glob("*.edf"))
    if not candidates:
        raise FileNotFoundError(f"no EDF file for {record.join_key}")
    values, sample_rate, names, units = _read_edf(candidates[0])
    name_map = _visual_eeg_name_map(project_root)
    canonical_names = tuple(name_map.get(name, canonical_channel_name(name)) for name in names)
    unique_units = sorted(set(units))
    return NativeEEGRecord(
        values=values,
        sample_rate_hz=sample_rate,
        channel_names=canonical_names,
        native_unit=unique_units[0] if len(unique_units) == 1 else "/".join(unique_units),
        source_path=candidates[0],
    )


def load_native_eeg_record(project_root: Path, record: CleanCacheRecord) -> NativeEEGRecord:
    if record.dataset_id == "eeg_fnirs_single_trial":
        return _single_trial_eeg(project_root, record)
    if record.dataset_id == "simultaneous_eeg_nirs":
        return _simultaneous_eeg(project_root, record)
    if record.dataset_id == "refed":
        return _refed_eeg(project_root, record)
    if record.dataset_id == "visual_cognitive_motivation":
        return _visual_eeg(project_root, record)
    raise KeyError(f"not one of the four raw datasets: {record.dataset_id}")


def canonical_label(event: Mapping[str, Any], dataset_id: str) -> dict[str, Any]:
    metadata = dict(event.get("metadata", {}))
    task = str(metadata.get("task") or event.get("event_type") or "unknown")
    event_role = str(metadata.get("event_role") or event.get("event_type") or "unknown")
    condition = str(metadata.get("condition_label") or event.get("label") or "unknown")
    if dataset_id == "visual_cognitive_motivation":
        condition = str(metadata.get("epoch_type") or condition or "unknown")
        class_index = VISUAL_CONDITION_INDICES.get(condition, -1)
    else:
        raw_index = event.get("label_index")
        class_index = int(raw_index) if raw_index is not None else -1
    return {
        "schema": "canonical_task_label_v1",
        "namespace": f"{dataset_id}:{task}",
        "dataset_id": dataset_id,
        "task": task,
        "condition": condition,
        "class_index": class_index,
        "event_role": event_role,
    }


def _slice_window(values: np.ndarray, onset_ms: float, duration_s: float, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    length = max(1, int(round(float(duration_s) * float(sample_rate_hz))))
    start = int(round(float(onset_ms) / 1000.0 * float(sample_rate_hz)))
    stop = start + length
    output = np.zeros((length, values.shape[1]), dtype=np.float32)
    mask = np.zeros(length, dtype=bool)
    src_start = max(start, 0)
    src_stop = min(stop, values.shape[0])
    if src_stop > src_start:
        dst_start = src_start - start
        dst_stop = dst_start + (src_stop - src_start)
        output[dst_start:dst_stop] = values[src_start:src_stop]
        mask[dst_start:dst_stop] = True
    return output.T, mask


class ChannelGeometryIndex:
    """Read the common channel-geometry sidecar and emit one row per channel."""

    def __init__(self, cache_root: Path):
        path = cache_root / "channel_geometry/channels.jsonl"
        self.rows = []
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                self.rows = [json.loads(line) for line in handle if line.strip()]

    def for_channels(
        self,
        *,
        record: CleanCacheRecord,
        modality: str,
        channel_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        candidates = [row for row in self.rows if row.get("dataset_id") == record.dataset_id and row.get("modality") == modality]
        subject_rows = [
            row for row in candidates
            if row.get("canonical_subject_id") in {record.canonical_subject_id, "all"}
        ]
        if subject_rows:
            candidates = subject_rows
        exact_record = [row for row in candidates if row.get("base_record_id") == record.base_record_id]
        if exact_record:
            candidates = exact_record
        elif record.dataset_id == "visual_cognitive_motivation" and modality == "fnirs":
            probe_match = re.search(r"Probe[12]", record.base_record_id, flags=re.IGNORECASE)
            if probe_match:
                probe = f"Probe{probe_match.group(0)[-1]}"
                probe_rows = [row for row in candidates if row.get("base_record_id") == probe]
                if probe_rows:
                    candidates = probe_rows
        lookup = {canonical_channel_name(str(row.get("channel_name", ""))): row for row in candidates}

        # Legacy Visual sidecars may contain label references without explicit
        # coordinates.  New graphical/CED projections already carry x/y/z.
        visual_eeg = {
            canonical_channel_name(str(row.get("channel_name", ""))): row
            for row in self.rows
            if row.get("dataset_id") == "visual_cognitive_motivation" and row.get("modality") == "eeg"
        }
        output = []
        for channel_name in channel_names:
            canonical = canonical_channel_name(channel_name)
            base = re.sub(r"_(HbO|HbR)$", "", canonical)
            row = dict(lookup.get(canonical) or lookup.get(base) or {})
            if record.dataset_id == "visual_cognitive_motivation" and modality == "fnirs" and row:
                reference = canonical_channel_name(str(row.get("metadata", {}).get("nearest_eeg_label", "")))
                reference_row = visual_eeg.get(reference, {})
                for axis in ("x", "y", "z"):
                    if row.get(axis) is None and reference_row.get(axis) is not None:
                        row[axis] = reference_row[axis]
                row["coordinate_system"] = "referenced_visual_eeg_head_coordinates"
            metadata = dict(row.get("metadata", {}))
            output.append({
                "schema": "canonical_channel_geometry_v1",
                "channel_name": canonical,
                "base_channel_name": base,
                "component": canonical.rsplit("_", 1)[-1] if canonical.endswith(("_HbO", "_HbR")) else modality,
                "modality": modality,
                "x": row.get("x"),
                "y": row.get("y"),
                "z": row.get("z"),
                "coordinate_system": row.get("coordinate_system", "unavailable"),
                "coordinate_units": row.get("coordinate_units", "unavailable"),
                "source_index": row.get("source_index"),
                "detector_index": row.get("detector_index"),
                "position_available": any(row.get(axis) is not None for axis in ("x", "y", "z")),
                "coordinate_status": metadata.get(
                    "coordinate_status",
                    "source_geometry" if row else "unavailable",
                ),
                "measured_subject_coordinate": metadata.get("measured_subject_coordinate"),
                "intended_use": metadata.get("intended_use"),
                "source_file": row.get("source_file"),
            })
        return output


def _refed_continuous_stream(event: Mapping[str, Any]) -> tuple[np.ndarray, float, float]:
    """Return REFED targets as ``[time, target]`` plus duration and native rate.

    REFED stores one joystick sample per video second in the current release.
    The event index intentionally carries the values so that target construction
    remains tied to the same event and modality-clock provenance as the signal
    window.  Orientation is accepted from either the released ``[time, 2]``
    arrays or the transposed layout described in the README.
    """

    metadata = event.get("metadata", {})
    stream = metadata.get("continuous_label_stream", {}) if isinstance(metadata, Mapping) else {}
    values = np.asarray(stream.get("values", []), dtype=np.float64)
    names = tuple(str(value).strip().lower() for value in stream.get("names", ()))
    if values.ndim != 2:
        raise ValueError(f"REFED continuous targets must be two-dimensional, got shape={values.shape}")
    if values.shape[1] == len(REFED_CONTINUOUS_TARGET_NAMES):
        pass
    elif values.shape[0] == len(REFED_CONTINUOUS_TARGET_NAMES):
        values = values.T
    else:
        raise ValueError(f"REFED continuous targets must contain two coordinates, got shape={values.shape}")
    if names:
        if set(names) != set(REFED_CONTINUOUS_TARGET_NAMES):
            raise ValueError(f"unexpected REFED continuous target names: {names}")
        values = values[:, [names.index(name) for name in REFED_CONTINUOUS_TARGET_NAMES]]
    if values.shape[0] == 0:
        raise ValueError("REFED continuous target stream is empty")
    declared_count = stream.get("sample_count")
    if declared_count is not None and int(declared_count) != values.shape[0]:
        raise ValueError(
            "REFED continuous target sample_count does not match values: "
            f"declared={declared_count}, actual={values.shape[0]}"
        )
    duration_s = float(event.get("duration_ms", 0.0)) / 1000.0
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError(f"REFED event duration must be positive, got {duration_s}")
    native_rate_hz = float(stream.get("native_sample_rate_hz", 1.0))
    if not np.isfinite(native_rate_hz) or native_rate_hz <= 0:
        raise ValueError("Invalid REFED annotation sampling rate")
    return values, duration_s, native_rate_hz


def refed_continuous_target_window(
    event: Mapping[str, Any],
    *,
    window_start_s: float,
    window_duration_s: float,
    target_sample_rate_hz: float = REFED_DEFAULT_TARGET_SAMPLE_RATE_HZ,
) -> dict[str, Any]:
    """Build a fixed-shape valence/arousal target sequence for one REFED window.

    Target timestamps are expressed on the event-relative clock.  The released
    annotation grid retains its published 1 Hz rate. Nominal fNIRS duration
    rounding must not stretch the annotation clock. Invalid time support and
    non-finite source values are
    zero-filled and identified by a per-coordinate mask; callers must consume
    that mask in the regression loss.
    """

    if not np.isfinite(window_start_s) or window_start_s < 0.0:
        raise ValueError(f"window_start_s must be finite and non-negative, got {window_start_s}")
    if not np.isfinite(window_duration_s) or window_duration_s <= 0.0:
        raise ValueError(f"window_duration_s must be positive, got {window_duration_s}")
    if not np.isfinite(target_sample_rate_hz) or target_sample_rate_hz <= 0.0:
        raise ValueError(f"target_sample_rate_hz must be positive, got {target_sample_rate_hz}")
    exact_count = window_duration_s * target_sample_rate_hz
    target_count = int(round(exact_count))
    if target_count <= 0 or not np.isclose(exact_count, target_count, rtol=0.0, atol=1e-6):
        raise ValueError(
            "window_duration_s * target_sample_rate_hz must be an integer for fixed-shape batching, "
            f"got {exact_count}"
        )

    source, event_duration_s, native_rate_hz = _refed_continuous_stream(event)
    target_time_s = window_start_s + np.arange(target_count, dtype=np.float64) / target_sample_rate_hz
    metadata = event.get("metadata", {})
    paired_signal_duration_s = min(event_duration_s, len(source) / native_rate_hz)
    if isinstance(metadata, Mapping):
        eeg_samples = metadata.get("eeg_samples")
        fnirs_samples = metadata.get("fnirs_samples")
        if eeg_samples is not None:
            paired_signal_duration_s = min(paired_signal_duration_s, float(eeg_samples) / 1000.0)
        if fnirs_samples is not None:
            paired_signal_duration_s = min(paired_signal_duration_s, float(fnirs_samples) / 47.62)
    time_valid = (target_time_s >= 0.0) & (target_time_s < paired_signal_duration_s)
    source_position = np.clip(
        target_time_s * native_rate_hz,
        0.0,
        float(source.shape[0] - 1),
    )
    left = np.floor(source_position).astype(np.int64)
    right = np.minimum(left + 1, source.shape[0] - 1)
    fraction = source_position - left

    target = np.zeros((len(REFED_CONTINUOUS_TARGET_NAMES), target_count), dtype=np.float32)
    valid_mask = np.zeros_like(target, dtype=bool)
    for coordinate in range(len(REFED_CONTINUOUS_TARGET_NAMES)):
        left_value = source[left, coordinate]
        right_value = source[right, coordinate]
        needs_right = fraction > 1e-7
        coordinate_valid = time_valid & np.isfinite(left_value) & (~needs_right | np.isfinite(right_value))
        interpolated = left_value.copy()
        interpolation_mask = coordinate_valid & needs_right
        interpolated[interpolation_mask] = (
            left_value[interpolation_mask] * (1.0 - fraction[interpolation_mask])
            + right_value[interpolation_mask] * fraction[interpolation_mask]
        )
        target[coordinate, coordinate_valid] = interpolated[coordinate_valid].astype(np.float32)
        valid_mask[coordinate] = coordinate_valid

    return {
        "schema": REFED_CONTINUOUS_SEQUENCE_SCHEMA,
        "values": target,
        "valid_mask": valid_mask,
        "time_s": target_time_s.astype(np.float32),
        "target_names": list(REFED_CONTINUOUS_TARGET_NAMES),
        "target_sample_rate_hz": float(target_sample_rate_hz),
        "source_sample_rate_hz": native_rate_hz,
        "source_sample_count": int(source.shape[0]),
        "event_duration_s": event_duration_s,
        "paired_signal_duration_s": paired_signal_duration_s,
        "value_coordinate": "refed_joystick_native",
        "scaling_policy": "preserve_native_in_loader_fit_scaling_on_train_subjects_only",
        "alignment_policy": "published_annotation_rate_linear_interpolation_v2",
    }


class UnifiedPhysiologyWindowDataset:
    """Event-aligned EEG/fNIRS windows for the four original datasets.

    The 20-second default is an observation-context contract, not a claim that
    every event lasts 20 seconds.  Event labels remain anchored at the event
    timestamp and ``valid_mask`` identifies record-boundary padding.  Models
    may subdivide the returned context into shorter patches without changing
    the loader contract.
    """

    def __init__(
        self,
        cache_root: str | Path = "data/cache/physiology_semantic_clean_v1",
        *,
        dataset_ids: Sequence[str] = RAW_DATASET_IDS,
        window_duration_s: float = DEFAULT_UNIFIED_WINDOW_DURATION_S,
        window_offset_s: float = 0.0,
        eeg_signal_branch: str = SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA,
        eeg_artifact_config: EEGArtifactCleaningConfig | None = None,
        eeg_artifact_cache_root: str | Path | None = None,
        require_eeg_artifact_cache: bool = False,
        simultaneous_eeg_cache_root: str | Path | None = None,
        require_paired_timestamps: bool = True,
        include_event_types: set[str] | None = None,
        admissible_alignment_cases: set[str] | frozenset[str] | None = DEFAULT_ADMISSIBLE_ALIGNMENT_CASES,
    ) -> None:
        requested = tuple(str(value) for value in dataset_ids)
        invalid = sorted(set(requested) - set(RAW_DATASET_IDS))
        if invalid:
            raise ValueError(f"only the four original datasets are supported; invalid={invalid}")
        self.cache_root = Path(cache_root)
        self.project_root = Path(__file__).resolve().parents[2]
        self.index = CleanPhysiologyCacheIndex(self.cache_root)
        self.geometry_index = ChannelGeometryIndex(self.cache_root)
        self.dataset_ids = requested
        self.window_duration_s = float(window_duration_s)
        self.window_offset_s = float(window_offset_s)
        if eeg_signal_branch not in SINGLE_TRIAL_EEG_SIGNAL_BRANCHES:
            raise ValueError(
                f"eeg_signal_branch must be one of {SINGLE_TRIAL_EEG_SIGNAL_BRANCHES}, got {eeg_signal_branch!r}"
            )
        self.eeg_signal_branch = eeg_signal_branch
        self.eeg_artifact_config = eeg_artifact_config
        self.require_eeg_artifact_cache = bool(
            require_eeg_artifact_cache
            or _REQUIRE_SINGLE_TRIAL_EEG_ARTIFACT_CACHE.get()
        )
        if self.require_eeg_artifact_cache and eeg_signal_branch not in {
            SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V3,
            SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4,
        }:
            raise ValueError(
                "require_eeg_artifact_cache requires a single-trial "
                "artifact-clean EEG branch"
            )
        self.eeg_artifact_cache_root = (
            Path(eeg_artifact_cache_root)
            if eeg_artifact_cache_root is not None
            else self.cache_root / (
                "eeg_artifact_clean_v4"
                if eeg_signal_branch == SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4
                else "eeg_artifact_clean_v3"
            )
        )
        self._artifact_cache_manifest: dict[str, Any] | None = None
        self._eeg_artifact_cache_record_keys: set[str] = set()
        self._eeg_native_fallback_record_keys: set[str] = set()
        self.simultaneous_eeg_cache_root = (
            Path(simultaneous_eeg_cache_root)
            if simultaneous_eeg_cache_root is not None
            else self.cache_root / SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1
        )
        self._simultaneous_eeg_cache_manifest: dict[str, Any] | None = None
        self.require_paired_timestamps = bool(require_paired_timestamps)
        self.include_event_types = include_event_types
        self.admissible_alignment_cases = None if admissible_alignment_cases is None else frozenset(admissible_alignment_cases)
        self.excluded_alignment_records: dict[str, str] = {}
        self.excluded_alignment_support_windows = 0
        self.excluded_forbidden_task_counts = {namespace: 0 for namespace in FORBIDDEN_TASK_NAMESPACES}
        self.excluded_forbidden_task_records: set[str] = set()
        self.windows = self._build_windows()
        self._record_cache: dict[str, dict[str, Any]] = {}

    def _selected_records(self) -> Iterable[CleanCacheRecord]:
        for record in self.index.records:
            if record.dataset_id not in self.dataset_ids:
                continue
            if record.dataset_id == "refed" and record.signal_branch != "hbo_hbr":
                continue
            yield record

    def _build_windows(self) -> list[UnifiedWindowRef]:
        windows = []
        for record in self._selected_records():
            admitted_events = []
            for event in self.index.events_by_join_key.get(record.join_key, []):
                namespace = canonical_label(event, record.dataset_id)["namespace"]
                if namespace in FORBIDDEN_TASK_NAMESPACES:
                    self.excluded_forbidden_task_counts[namespace] += 1
                    self.excluded_forbidden_task_records.add(record.join_key)
                    continue
                admitted_events.append(event)
            reports = self.index.reports_by_join_key.get(record.join_key, [])
            if self.admissible_alignment_cases is not None:
                cases = {str(report.get("alignment_case", "")) for report in reports}
                label_match = all(report.get("label_sequence_match") is not False for report in reports)
                if not reports or not cases.issubset(self.admissible_alignment_cases) or not label_match:
                    self.excluded_alignment_records[record.join_key] = ",".join(sorted(cases)) or "missing_alignment_report"
                    continue
            for event in admitted_events:
                if self.include_event_types is not None and str(event.get("event_type")) not in self.include_event_types:
                    continue
                eeg_time = event.get("eeg_time_ms")
                fnirs_time = event.get("fnirs_time_ms", event.get("onset_ms"))
                if eeg_time is None and not self.require_paired_timestamps:
                    eeg_time = event.get("onset_ms")
                if fnirs_time is None or (self.require_paired_timestamps and eeg_time is None):
                    continue
                if not window_within_alignment_support(event, self.window_offset_s, self.window_duration_s):
                    self.excluded_alignment_support_windows += 1
                    continue
                windows.append(UnifiedWindowRef(record=record, event=event))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def _load_canonical_record(self, record: CleanCacheRecord) -> dict[str, Any]:
        cached = self._record_cache.get(record.join_key)
        if cached is not None:
            return cached
        self._assert_required_single_trial_artifact_cache(record)
        arrays = self.index.load_record_arrays(record)
        if "homer2_aligned_fnirs" not in arrays:
            raise KeyError(f"missing homer2_aligned_fnirs: {record.npz_path}")
        fnirs_names = canonical_fnirs_channel_names(
            [str(value) for value in arrays.get("homer2_channel_names", record.manifest.get("homer2_channel_names", []))]
        )
        roles = fnirs_component_roles(fnirs_names)
        if set(roles) != set(CANONICAL_FNIRS_COMPONENTS):
            raise ValueError(f"fNIRS component contract failed for {record.join_key}: {sorted(set(roles))}")
        fnirs, fnirs_state = preprocess_fnirs_record(
            arrays["homer2_aligned_fnirs"],
            sample_rate_hz=record.sample_rate_hz,
            native_contract=record.manifest.get("native_contract", {}),
        )
        if record.dataset_id == "eeg_fnirs_single_trial":
            eeg_branch = self.eeg_signal_branch
        elif record.dataset_id == "simultaneous_eeg_nirs":
            eeg_branch = SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1
        else:
            eeg_branch = "raw_with_ocular_artifact"
        cached_eeg = self._load_cached_single_trial_eeg(record, eeg_branch)
        if cached_eeg is None:
            cached_eeg = self._load_cached_simultaneous_eeg(record, eeg_branch)
        if cached_eeg is not None:
            eeg, eeg_names, eeg_state, eeg_quality = cached_eeg
            if record.dataset_id == "eeg_fnirs_single_trial":
                self._eeg_artifact_cache_record_keys.add(record.join_key)
        else:
            if (
                self.require_eeg_artifact_cache
                and record.dataset_id == "eeg_fnirs_single_trial"
                and eeg_branch
                in {
                    SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V3,
                    SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4,
                }
            ):
                raise RuntimeError(
                    "Required EEG artifact cache was not used for "
                    f"{record.join_key}; native EEG fallback is forbidden"
                )
            self._eeg_native_fallback_record_keys.add(record.join_key)
            eeg_native = load_native_eeg_record(self.project_root, record)
            eeg_names = eeg_native.channel_names
            eeg_geometry_for_cleaning = self.geometry_index.for_channels(
                record=record, modality="eeg", channel_names=eeg_names
            )
            eeg_positions = np.asarray(
                [[row.get(axis) for axis in ("x", "y", "z")] for row in eeg_geometry_for_cleaning],
                dtype=np.float64,
            )
            eeg, eeg_state, eeg_quality = preprocess_eeg_record_with_quality(
                eeg_native,
                signal_branch=eeg_branch,
                artifact_config=self.eeg_artifact_config,
                channel_positions=eeg_positions,
            )
        eeg_geometry = self.geometry_index.for_channels(
            record=record, modality="eeg", channel_names=eeg_names
        )
        payload = {
            "eeg": eeg,
            "fnirs": fnirs,
            "eeg_channel_names": eeg_names,
            "fnirs_channel_names": fnirs_names,
            "fnirs_component_roles": roles,
            "eeg_preprocessing_state": eeg_state,
            "fnirs_preprocessing_state": fnirs_state,
            "eeg_quality": eeg_quality,
            "eeg_geometry": eeg_geometry,
        }
        # Keep memory bounded while allowing repeated events from one record.
        if len(self._record_cache) >= 2:
            self._record_cache.pop(next(iter(self._record_cache)))
        self._record_cache[record.join_key] = payload
        return payload

    def _load_cached_single_trial_eeg(
        self,
        record: CleanCacheRecord,
        eeg_branch: str,
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any], dict[str, np.ndarray]] | None:
        if (
            record.dataset_id != "eeg_fnirs_single_trial"
            or eeg_branch not in {
                SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V3,
                SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4,
            }
        ):
            return None
        cache_schema = (
            "single_trial_eeg_artifact_cache_v4"
            if eeg_branch == SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4
            else "single_trial_eeg_artifact_cache_v3"
        )
        path = (
            self.eeg_artifact_cache_root
            / record.canonical_subject_id
            / f"{record.base_record_id}.npz"
        )
        if not path.exists():
            return None
        manifest = self._validated_artifact_cache_manifest()
        if (
            manifest.get("schema") != cache_schema
            or manifest.get("signal_branch") != eeg_branch
        ):
            raise RuntimeError(
                "EEG artifact cache manifest schema/branch pair differs "
                f"from requested branch {eeg_branch!r}"
            )
        expected_config = replace(
            self.eeg_artifact_config or EEGArtifactCleaningConfig(),
            schema=eeg_branch,
            line_noise_frequency_hz=(
                50.0 if eeg_branch == SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4 else None
            ),
            bad_channel_action=(
                "disabled_for_cross_dataset_uniformity"
                if eeg_branch == SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4
                else "detect_and_interpolate"
            ),
            muscle_action="mask_gated_high_frequency_attenuation_v1",
        ).to_dict()
        if manifest.get("cleaning_config") != expected_config:
            return None
        manifest_record = next(
            (item for item in manifest.get("records", []) if item.get("join_key") == record.join_key),
            None,
        )
        if manifest_record is None:
            raise RuntimeError(f"EEG artifact cache manifest has no record for {record.join_key}")
        manifest_cache_path = Path(str(manifest_record.get("cache_path", "")))
        if not manifest_cache_path.is_absolute():
            manifest_cache_path = self.project_root / manifest_cache_path
        if manifest_cache_path.resolve() != path.resolve():
            raise RuntimeError(
                "EEG artifact cache manifest path differs from loader path "
                f"for {record.join_key}"
            )
        with np.load(path, allow_pickle=False) as payload:
            schema = str(np.asarray(payload["schema"]).item())
            signal_branch = str(
                np.asarray(payload["signal_branch"]).item()
            )
            join_key = str(np.asarray(payload["join_key"]).item())
            if (
                schema != cache_schema
                or signal_branch != eeg_branch
                or join_key != record.join_key
            ):
                raise RuntimeError(
                    "stale/incompatible EEG artifact cache "
                    f"{path}: schema={schema!r}, "
                    f"signal_branch={signal_branch!r}, "
                    f"join_key={join_key!r}"
                )
            source_path = self.project_root / str(np.asarray(payload["source_path"]).item())
            source_stat = source_path.stat()
            expected_size = int(np.asarray(payload["source_size_bytes"]).item())
            expected_mtime = int(np.asarray(payload["source_mtime_ns"]).item())
            if source_stat.st_size != expected_size or source_stat.st_mtime_ns != expected_mtime:
                raise RuntimeError(f"source EEG changed after artifact cache build: {source_path}")
            state = json.loads(str(np.asarray(payload["preprocessing_state_json"]).item()))
            state["artifact_mask_policy"] = EEG_ARTIFACT_MASK_POLICY
            state["artifact_cache"] = {
                "used": True,
                "path": str(path),
                "schema": schema,
                "stored_detected_artifact_fraction": float(
                    np.mean(np.asarray(payload["artifact_mask"], dtype=bool))
                ),
                "stored_detected_artifact_mask_exposed": False,
            }
            stored_artifact_mask = np.asarray(payload["artifact_mask"], dtype=bool)
            return (
                np.asarray(payload["eeg"], dtype=np.float32),
                tuple(str(value) for value in np.asarray(payload["channel_names"]).tolist()),
                state,
                {
                    "artifact_mask": np.zeros_like(stored_artifact_mask),
                    "bad_channel_mask": np.asarray(payload["bad_channel_mask"], dtype=bool),
                },
            )

    def _assert_required_single_trial_artifact_cache(
        self,
        record: CleanCacheRecord,
    ) -> None:
        """Fail before fNIRS dereference if strict artifact-cache use cannot hold."""

        if (
            not self.require_eeg_artifact_cache
            or record.dataset_id != "eeg_fnirs_single_trial"
        ):
            return
        eeg_branch = self.eeg_signal_branch
        cache_schema = (
            "single_trial_eeg_artifact_cache_v4"
            if eeg_branch == SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4
            else "single_trial_eeg_artifact_cache_v3"
        )
        path = (
            self.eeg_artifact_cache_root
            / record.canonical_subject_id
            / f"{record.base_record_id}.npz"
        )
        if not path.exists():
            raise RuntimeError(
                f"Required EEG artifact cache file is missing: {path}"
            )
        manifest = self._validated_artifact_cache_manifest()
        if (
            manifest.get("schema") != cache_schema
            or manifest.get("signal_branch") != eeg_branch
        ):
            raise RuntimeError(
                "Required EEG artifact cache manifest does not match the "
                "exact requested schema/branch pair"
            )
        expected_config = replace(
            self.eeg_artifact_config or EEGArtifactCleaningConfig(),
            schema=eeg_branch,
            line_noise_frequency_hz=(
                50.0
                if eeg_branch == SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4
                else None
            ),
            bad_channel_action=(
                "disabled_for_cross_dataset_uniformity"
                if eeg_branch == SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4
                else "detect_and_interpolate"
            ),
            muscle_action="mask_gated_high_frequency_attenuation_v1",
        ).to_dict()
        if manifest.get("cleaning_config") != expected_config:
            raise RuntimeError(
                "Required EEG artifact cache cleaning config mismatch"
            )
        manifest_record = next(
            (
                item
                for item in manifest.get("records", [])
                if item.get("join_key") == record.join_key
            ),
            None,
        )
        if manifest_record is None:
            raise RuntimeError(
                "Required EEG artifact cache manifest has no record for "
                f"{record.join_key}"
            )
        manifest_cache_path = Path(str(manifest_record.get("cache_path", "")))
        if not manifest_cache_path.is_absolute():
            manifest_cache_path = self.project_root / manifest_cache_path
        if manifest_cache_path.resolve() != path.resolve():
            raise RuntimeError(
                "Required EEG artifact cache manifest path differs from "
                f"loader path for {record.join_key}"
            )

    def _validated_artifact_cache_manifest(self) -> dict[str, Any]:
        cached = getattr(self, "_artifact_cache_manifest", None)
        if cached is not None:
            return cached
        path = self.eeg_artifact_cache_root / "cache_manifest.json"
        if not path.exists():
            raise RuntimeError(f"EEG artifact cache record exists without manifest: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema")
            not in {
                "single_trial_eeg_artifact_cache_v3",
                "single_trial_eeg_artifact_cache_v4",
            }
            or manifest.get("signal_branch")
            not in {
                SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V3,
                SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4,
            }
        ):
            raise RuntimeError(f"stale/incompatible EEG artifact cache manifest: {path}")
        code_root = Path(__file__).resolve().parents[2]
        code_paths = {
            "audit": code_root / "experiments/audit_single_trial_eeg_artifact_v2.py",
            "cleaner": Path(clean_single_trial_eeg.__code__.co_filename).resolve(),
        }
        recorded_hashes = manifest.get("code_sha256", {})
        for name, code_path in code_paths.items():
            digest = hashlib.sha256(code_path.read_bytes()).hexdigest()
            if recorded_hashes.get(name) != digest:
                raise RuntimeError(
                    f"EEG artifact cache code hash mismatch for {name}: rebuild {self.eeg_artifact_cache_root}"
                )
        self._artifact_cache_manifest = manifest
        return manifest

    def _load_cached_simultaneous_eeg(
        self,
        record: CleanCacheRecord,
        eeg_branch: str,
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any], dict[str, np.ndarray]] | None:
        if (
            record.dataset_id != "simultaneous_eeg_nirs"
            or eeg_branch != SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1
        ):
            return None
        path = self.simultaneous_eeg_cache_root / record.canonical_subject_id / f"{record.base_record_id}.npz"
        if not path.exists():
            return None
        manifest = self._validated_simultaneous_eeg_cache_manifest()
        with np.load(path, allow_pickle=False) as payload:
            schema = str(np.asarray(payload["schema"]).item())
            join_key = str(np.asarray(payload["join_key"]).item())
            if schema != "simultaneous_eeg_eog_cache_v1" or join_key != record.join_key:
                raise RuntimeError(
                    f"stale/incompatible Simultaneous EEG cache {path}: "
                    f"schema={schema!r}, join_key={join_key!r}"
                )
            source_path = self.project_root / str(np.asarray(payload["source_path"]).item())
            source_stat = source_path.stat()
            if (
                source_stat.st_size != int(np.asarray(payload["source_size_bytes"]).item())
                or source_stat.st_mtime_ns != int(np.asarray(payload["source_mtime_ns"]).item())
            ):
                raise RuntimeError(f"source EEG changed after Simultaneous EOG cache build: {source_path}")
            state = json.loads(str(np.asarray(payload["preprocessing_state_json"]).item()))
            state["artifact_mask_policy"] = EEG_ARTIFACT_MASK_POLICY
            state["artifact_cache"] = {
                "used": True,
                "path": str(path),
                "schema": schema,
                "stored_detected_artifact_fraction": float(
                    np.mean(np.asarray(payload["artifact_mask"], dtype=bool))
                ),
                "stored_detected_artifact_mask_exposed": False,
            }
            stored_artifact_mask = np.asarray(payload["artifact_mask"], dtype=bool)
            return (
                np.asarray(payload["eeg"], dtype=np.float32),
                tuple(str(value) for value in np.asarray(payload["channel_names"]).tolist()),
                state,
                {
                    "artifact_mask": np.zeros_like(stored_artifact_mask),
                    "bad_channel_mask": np.asarray(payload["bad_channel_mask"], dtype=bool),
                },
            )

    def _validated_simultaneous_eeg_cache_manifest(self) -> dict[str, Any]:
        if self._simultaneous_eeg_cache_manifest is not None:
            return self._simultaneous_eeg_cache_manifest
        path = self.simultaneous_eeg_cache_root / "cache_manifest.json"
        if not path.exists():
            raise RuntimeError(f"Simultaneous EEG cache record exists without manifest: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected_config = simultaneous_eeg_eog_cleaning_config(
            self.eeg_artifact_config
        ).to_dict()
        if (
            manifest.get("schema") != "simultaneous_eeg_eog_cache_v1"
            or manifest.get("signal_branch") != SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1
            or manifest.get("cleaning_config") != expected_config
        ):
            raise RuntimeError(f"stale/incompatible Simultaneous EEG cache manifest: {path}")
        code_root = Path(__file__).resolve().parents[2]
        code_paths = {
            "builder": code_root / "experiments/build_simultaneous_eeg_eog_clean_cache.py",
            "cleaner": Path(clean_single_trial_eeg.__code__.co_filename).resolve(),
        }
        for name, code_path in code_paths.items():
            digest = hashlib.sha256(code_path.read_bytes()).hexdigest()
            if manifest.get("code_sha256", {}).get(name) != digest:
                raise RuntimeError(
                    f"Simultaneous EEG cache code hash mismatch for {name}: "
                    f"rebuild {self.simultaneous_eeg_cache_root}"
                )
        self._simultaneous_eeg_cache_manifest = manifest
        return manifest

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.windows[index]
        if not window_within_alignment_support(
            ref.event, self.window_offset_s + ref.window_offset_s, self.window_duration_s,
        ):
            raise ValueError("Window enters an unverified concatenation interval")
        record_data = self._load_canonical_record(ref.record)
        offset_ms = (self.window_offset_s + ref.window_offset_s) * 1000.0
        eeg_time_ms = float(ref.event.get("eeg_time_ms", ref.event.get("onset_ms"))) + offset_ms
        fnirs_time_ms = float(ref.event.get("fnirs_time_ms", ref.event.get("onset_ms"))) + offset_ms
        eeg, eeg_mask = _slice_window(
            record_data["eeg"], eeg_time_ms, self.window_duration_s, CANONICAL_EEG_SAMPLE_RATE_HZ
        )
        fnirs, fnirs_mask = _slice_window(
            record_data["fnirs"], fnirs_time_ms, self.window_duration_s, CANONICAL_FNIRS_SAMPLE_RATE_HZ
        )
        # Artifact detections retained in historical cleaning caches are audit
        # provenance only. They are deliberately not exposed as sample labels
        # and have no authority over measurement validity.
        eeg_artifact_mask = np.zeros_like(eeg_mask)
        label = canonical_label(ref.event, ref.record.dataset_id)
        return {
            "schema": UNIFIED_PHYSIOLOGY_SCHEMA,
            "eeg": eeg,
            "fnirs": fnirs,
            "valid_mask": {"eeg": eeg_mask, "fnirs": fnirs_mask},
            "analysis_valid_mask": {"eeg": eeg_mask.copy(), "fnirs": fnirs_mask.copy()},
            "artifact_mask": {"eeg": eeg_artifact_mask, "fnirs": np.zeros_like(fnirs_mask)},
            "artifact_mask_policy": EEG_ARTIFACT_MASK_POLICY,
            "bad_channel_mask": {
                "eeg": record_data["eeg_quality"]["bad_channel_mask"].copy(),
                "fnirs": np.zeros(fnirs.shape[0], dtype=bool),
            },
            "modality_available": {"eeg": True, "fnirs": True},
            "sample_rate_hz": {
                "eeg": CANONICAL_EEG_SAMPLE_RATE_HZ,
                "fnirs": CANONICAL_FNIRS_SAMPLE_RATE_HZ,
            },
            "unit": {"eeg": CANONICAL_UNIT, "fnirs": CANONICAL_UNIT},
            "channel_names": {
                "eeg": list(record_data["eeg_channel_names"]),
                "fnirs": list(record_data["fnirs_channel_names"]),
            },
            "component_roles": {
                "eeg": ["electrical_potential"] * eeg.shape[0],
                "fnirs": list(record_data["fnirs_component_roles"]),
            },
            "channel_geometry": {
                "eeg": record_data["eeg_geometry"],
                "fnirs": self.geometry_index.for_channels(
                    record=ref.record, modality="fnirs", channel_names=record_data["fnirs_channel_names"]
                ),
            },
            "label": label,
            "dataset_id": ref.record.dataset_id,
            "subject": ref.record.canonical_subject_id,
            "record_id": ref.record.base_record_id,
            "signal_branch": ref.record.signal_branch,
            "eeg_signal_branch": record_data["eeg_preprocessing_state"]["signal_branch"],
            "join_key": ref.record.join_key,
            "event": dict(ref.event),
            "alignment": {
                "eeg_time_ms": eeg_time_ms,
                "fnirs_time_ms": fnirs_time_ms,
                "offset_ms": fnirs_time_ms - eeg_time_ms,
                "event_relative_window_start_s": self.window_offset_s + ref.window_offset_s,
                "separate_modality_clocks_used": True,
                "sample_grid": {
                    modality: {
                        "start_index": int(round(time_ms * rate / 1000.0)),
                        "actual_start_ms": round(time_ms * rate / 1000.0) * 1000.0 / rate,
                        "rounding_error_ms": round(time_ms * rate / 1000.0) * 1000.0 / rate - time_ms,
                        "sample_period_ms": 1000.0 / rate,
                    }
                    for modality, time_ms, rate in (
                        ("eeg", eeg_time_ms, CANONICAL_EEG_SAMPLE_RATE_HZ),
                        ("fnirs", fnirs_time_ms, CANONICAL_FNIRS_SAMPLE_RATE_HZ),
                    )
                },
            },
            "preprocessing_contract": CANONICAL_PREPROCESSING.to_dict(),
            "preprocessing_state": {
                "eeg": record_data["eeg_preprocessing_state"],
                "fnirs": record_data["fnirs_preprocessing_state"],
            },
        }

    def contract_summary(self) -> dict[str, Any]:
        counts = {dataset_id: 0 for dataset_id in self.dataset_ids}
        for window in self.windows:
            counts[window.record.dataset_id] += 1
        return {
            "schema": UNIFIED_PHYSIOLOGY_SCHEMA,
            "dataset_ids": list(self.dataset_ids),
            "derived_targets_excluded": ["croce_local_cache"],
            "forbidden_task_policy": FORBIDDEN_TASK_POLICY,
            "forbidden_task_namespaces": sorted(FORBIDDEN_TASK_NAMESPACES),
            "excluded_forbidden_task_window_count": sum(self.excluded_forbidden_task_counts.values()),
            "excluded_forbidden_task_window_count_by_namespace": dict(self.excluded_forbidden_task_counts),
            "excluded_forbidden_task_record_count": len(self.excluded_forbidden_task_records),
            "excluded_forbidden_task_records": sorted(self.excluded_forbidden_task_records),
            "window_count_by_dataset": counts,
            "admissible_alignment_cases": sorted(self.admissible_alignment_cases or []),
            "excluded_alignment_record_count": len(self.excluded_alignment_records),
            "excluded_alignment_support_windows": self.excluded_alignment_support_windows,
            "excluded_alignment_records": dict(self.excluded_alignment_records),
            "eeg_signal_branch": self.eeg_signal_branch,
            "require_eeg_artifact_cache": self.require_eeg_artifact_cache,
            "eeg_artifact_cache_record_count": len(
                self._eeg_artifact_cache_record_keys
            ),
            "eeg_artifact_cache_record_keys": sorted(
                self._eeg_artifact_cache_record_keys
            ),
            "eeg_native_fallback_record_count": len(
                self._eeg_native_fallback_record_keys
            ),
            "eeg_native_fallback_record_keys": sorted(
                self._eeg_native_fallback_record_keys
            ),
            "eeg_artifact_mask_policy": EEG_ARTIFACT_MASK_POLICY,
            "preprocessing": CANONICAL_PREPROCESSING.to_dict(),
            "fnirs_components": list(CANONICAL_FNIRS_COMPONENTS),
            "label_schema": "canonical_task_label_v1",
            "geometry_schema": "canonical_channel_geometry_v1",
        }


class REFEDContinuousSequenceDataset(UnifiedPhysiologyWindowDataset):
    """Sliding multimodal REFED windows with valence/arousal sequence targets.

    The loader expands each video event into deterministic, event-relative
    windows.  By default the stride equals the observation duration, avoiding
    duplicate signal support while exposing all of the video rather than only
    its first window.  The final partial window is retained with signal and
    target masks; it can be disabled explicitly without changing alignment.
    Subject/video grouping must be applied before split generation.
    """

    def __init__(
        self,
        cache_root: str | Path = "data/cache/physiology_semantic_clean_v1",
        *,
        window_duration_s: float = DEFAULT_UNIFIED_WINDOW_DURATION_S,
        window_stride_s: float | None = None,
        target_sample_rate_hz: float = REFED_DEFAULT_TARGET_SAMPLE_RATE_HZ,
        include_partial_windows: bool = True,
        eeg_signal_branch: str = SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA,
        eeg_artifact_config: EEGArtifactCleaningConfig | None = None,
        eeg_artifact_cache_root: str | Path | None = None,
        require_paired_timestamps: bool = True,
        admissible_alignment_cases: set[str] | frozenset[str] | None = DEFAULT_ADMISSIBLE_ALIGNMENT_CASES,
    ) -> None:
        self.window_stride_s = float(window_duration_s if window_stride_s is None else window_stride_s)
        self.target_sample_rate_hz = float(target_sample_rate_hz)
        self.include_partial_windows = bool(include_partial_windows)
        if not np.isfinite(self.window_stride_s) or self.window_stride_s <= 0.0:
            raise ValueError(f"window_stride_s must be positive, got {self.window_stride_s}")
        exact_target_count = float(window_duration_s) * self.target_sample_rate_hz
        if (
            not np.isfinite(self.target_sample_rate_hz)
            or self.target_sample_rate_hz <= 0.0
            or not np.isclose(exact_target_count, round(exact_target_count), rtol=0.0, atol=1e-6)
        ):
            raise ValueError(
                "target_sample_rate_hz must be positive and produce an integer target length per window"
            )
        super().__init__(
            cache_root,
            dataset_ids=("refed",),
            window_duration_s=window_duration_s,
            window_offset_s=0.0,
            eeg_signal_branch=eeg_signal_branch,
            eeg_artifact_config=eeg_artifact_config,
            eeg_artifact_cache_root=eeg_artifact_cache_root,
            require_paired_timestamps=require_paired_timestamps,
            include_event_types={"video_segment_with_continuous_labels"},
            admissible_alignment_cases=admissible_alignment_cases,
        )
        source_events = tuple(self.windows)
        self.source_event_count = len(source_events)
        expanded: list[UnifiedWindowRef] = []
        for ref in source_events:
            _source, duration_s, _source_rate_hz = _refed_continuous_stream(ref.event)
            starts = np.arange(0.0, duration_s, self.window_stride_s, dtype=np.float64)
            for start_s in starts.tolist():
                if not self.include_partial_windows and start_s + self.window_duration_s > duration_s + 1e-6:
                    continue
                expanded.append(replace(ref, window_offset_s=float(start_s)))
        self.windows = expanded

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        ref = self.windows[index]
        target = refed_continuous_target_window(
            ref.event,
            window_start_s=ref.window_offset_s,
            window_duration_s=self.window_duration_s,
            target_sample_rate_hz=self.target_sample_rate_hz,
        )
        relative_target_time_s = np.arange(target["values"].shape[1], dtype=np.float64) / self.target_sample_rate_hz
        paired_window_mask = np.ones(relative_target_time_s.shape, dtype=bool)
        for modality in ("eeg", "fnirs"):
            modality_mask = np.asarray(sample["valid_mask"][modality], dtype=bool)
            modality_indices = np.floor(
                relative_target_time_s * float(sample["sample_rate_hz"][modality]) + 1e-9
            ).astype(np.int64)
            within = modality_indices < modality_mask.size
            modality_valid = np.zeros_like(paired_window_mask)
            modality_valid[within] = modality_mask[modality_indices[within]]
            paired_window_mask &= modality_valid
        target["valid_mask"] &= paired_window_mask[None, :]
        target["values"][~target["valid_mask"]] = 0.0
        target["paired_window_signal_valid_time_count"] = int(paired_window_mask.sum())
        context_label = sample["label"]
        event_payload = dict(sample["event"])
        event_metadata = dict(event_payload.get("metadata", {}))
        stream_metadata = dict(event_metadata.get("continuous_label_stream", {}))
        stream_metadata.pop("values", None)
        event_metadata["continuous_label_stream"] = stream_metadata
        event_payload["metadata"] = event_metadata
        sample["event"] = event_payload
        event_index = int(ref.event.get("event_index", context_label.get("class_index", -1)))
        start_ms = int(round(ref.window_offset_s * 1000.0))
        sample.update(
            {
                "schema": REFED_CONTINUOUS_SEQUENCE_SCHEMA,
                "source_window_schema": UNIFIED_PHYSIOLOGY_SCHEMA,
                "sample_id": f"{ref.record.join_key}|event={event_index}|start_ms={start_ms}",
                "label": {
                    "schema": REFED_CONTINUOUS_SEQUENCE_SCHEMA,
                    "namespace": "refed:emotion_video",
                    "task": "emotion_video",
                    "target_type": "continuous_sequence_regression",
                    "target_names": list(REFED_CONTINUOUS_TARGET_NAMES),
                },
                "video_context_label": context_label,
                "target": target["values"],
                "target_valid_mask": target["valid_mask"],
                "target_time_s": target["time_s"],
                "target_names": target["target_names"],
                "target_sample_rate_hz": target["target_sample_rate_hz"],
                "target_metadata": {
                    key: value
                    for key, value in target.items()
                    if key not in {"values", "valid_mask", "time_s", "target_names", "target_sample_rate_hz"}
                },
            }
        )
        return sample

    def contract_summary(self) -> dict[str, Any]:
        base = super().contract_summary()
        event_path = self.cache_root / "event_index" / "events.jsonl"
        event_index_sha256 = hashlib.sha256(event_path.read_bytes()).hexdigest() if event_path.exists() else None
        partial_windows = 0
        valid_target_values = 0
        total_target_values = 0
        source_rates = []
        for ref in self.windows:
            target = refed_continuous_target_window(
                ref.event,
                window_start_s=ref.window_offset_s,
                window_duration_s=self.window_duration_s,
                target_sample_rate_hz=self.target_sample_rate_hz,
            )
            source_rates.append(float(target["source_sample_rate_hz"]))
            valid_target_values += int(target["valid_mask"].sum())
            total_target_values += int(target["valid_mask"].size)
            if not bool(target["valid_mask"].all()):
                partial_windows += 1
        base.update(
            {
                "schema": REFED_CONTINUOUS_SEQUENCE_SCHEMA,
                "source_window_schema": UNIFIED_PHYSIOLOGY_SCHEMA,
                "source_event_count": self.source_event_count,
                "window_count": len(self.windows),
                "window_duration_s": self.window_duration_s,
                "window_stride_s": self.window_stride_s,
                "include_partial_windows": self.include_partial_windows,
                "partial_window_count": partial_windows,
                "target_type": "continuous_sequence_regression",
                "label_schema": REFED_CONTINUOUS_SEQUENCE_SCHEMA,
                "target_names": list(REFED_CONTINUOUS_TARGET_NAMES),
                "target_shape": [len(REFED_CONTINUOUS_TARGET_NAMES), int(round(self.window_duration_s * self.target_sample_rate_hz))],
                "target_sample_rate_hz": self.target_sample_rate_hz,
                "source_target_sample_rate_hz_range": (
                    [min(source_rates), max(source_rates)] if source_rates else []
                ),
                "valid_target_value_fraction": (
                    valid_target_values / total_target_values if total_target_values else 0.0
                ),
                "value_coordinate": "refed_joystick_native",
                "target_scaling_policy": "fit_on_train_subjects_only",
                "split_group_keys": ["subject"],
                "window_dependency_group_keys": ["subject", "record_id"],
                "event_index_sha256": event_index_sha256,
            }
        )
        return base


def collate_refed_continuous_sequences(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate REFED regression samples without batching nullable provenance.

    PyTorch's default collator cannot process the nullable coordinate/provenance
    fields carried by the unified loader.  This adapter stacks only fixed-shape
    model inputs and masks, keeps the shared channel contract once, and retains
    sample-specific provenance as a list.
    """

    if not samples:
        raise ValueError("cannot collate an empty REFED batch")
    required = {
        "eeg",
        "fnirs",
        "valid_mask",
        "analysis_valid_mask",
        "artifact_mask",
        "bad_channel_mask",
        "target",
        "target_valid_mask",
        "target_time_s",
    }
    missing = sorted(required - set(samples[0]))
    if missing:
        raise KeyError(f"REFED sample is missing batch fields: {missing}")
    from torch.utils.data import default_collate

    stacked = default_collate([{key: sample[key] for key in sorted(required)} for sample in samples])
    first = samples[0]
    stacked.update(
        {
            "schema": REFED_CONTINUOUS_SEQUENCE_SCHEMA,
            "label": dict(first["label"]),
            "target_names": list(first["target_names"]),
            "target_sample_rate_hz": float(first["target_sample_rate_hz"]),
            "sample_rate_hz": dict(first["sample_rate_hz"]),
            "channel_names": first["channel_names"],
            "component_roles": first["component_roles"],
            "channel_geometry": first["channel_geometry"],
            "sample_id": [str(sample["sample_id"]) for sample in samples],
            "dataset_id": [str(sample["dataset_id"]) for sample in samples],
            "subject": [str(sample["subject"]) for sample in samples],
            "record_id": [str(sample["record_id"]) for sample in samples],
            "join_key": [str(sample["join_key"]) for sample in samples],
            "provenance": [
                {
                    "event": sample["event"],
                    "alignment": sample["alignment"],
                    "target_metadata": sample["target_metadata"],
                    "video_context_label": sample["video_context_label"],
                    "preprocessing_state": sample["preprocessing_state"],
                    "eeg_signal_branch": sample["eeg_signal_branch"],
                }
                for sample in samples
            ],
        }
    )
    return stacked
