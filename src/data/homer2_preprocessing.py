"""HOMER2-aligned fNIRS preprocessing contracts.

This module does not claim to be a full HOMER2 reimplementation.  It provides
the repository contract needed to keep raw-native fNIRS coordinates separate
from a best-effort HOMER2-aligned branch, while recording which canonical
HOMER2 inputs are missing for each dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt


HOMER2_ALIGNMENT_SCHEMA = "homer2_alignment_contract_v1"


@dataclass(frozen=True)
class Homer2DatasetCompatibility:
    dataset_id: str
    entry_stage: str
    completeness: str
    available_inputs: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    possible_steps: tuple[str, ...]
    blocked_steps: tuple[str, ...]
    wavelengths_nm: tuple[float, ...]
    native_unit: str
    notes: tuple[str, ...]
    schema: str = HOMER2_ALIGNMENT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "available_inputs",
            "missing_inputs",
            "possible_steps",
            "blocked_steps",
            "wavelengths_nm",
            "notes",
        ):
            payload[key] = list(payload[key])
        return payload


DATASET_HOMER2_COMPATIBILITY: dict[str, Homer2DatasetCompatibility] = {
    "eeg_fnirs_single_trial": Homer2DatasetCompatibility(
        dataset_id="eeg_fnirs_single_trial",
        entry_stage="raw_intensity",
        completeness="near_full_without_short_channels",
        available_inputs=("paired_760_850_intensity", "source_detector_pairs", "sample_rate"),
        missing_inputs=("short_separation_channels", "subject_specific_dpf", "homer2_quality_marks"),
        possible_steps=(
            "intensity_to_optical_density",
            "motion_detection",
            "motion_correction",
            "bandpass",
            "modified_beer_lambert",
        ),
        blocked_steps=("short_channel_regression", "exact_homer2_channel_pruning_policy"),
        wavelengths_nm=(760.0, 850.0),
        native_unit="V",
        notes=("Raw optical voltage is present, so this is the only dataset that can enter the branch before OD conversion.",),
    ),
    "simultaneous_eeg_nirs": Homer2DatasetCompatibility(
        dataset_id="simultaneous_eeg_nirs",
        entry_stage="chromophore",
        completeness="partial_post_conversion",
        available_inputs=("oxy_deoxy_matlab_export", "sample_rate", "channel_labels"),
        missing_inputs=("raw_wl1_wl2_intensity", "source_detector_geometry_in_cache", "short_separation_channels"),
        possible_steps=("motion_detection", "motion_correction", "bandpass"),
        blocked_steps=("intensity_to_optical_density", "modified_beer_lambert_from_raw", "short_channel_regression"),
        wavelengths_nm=(760.0, 850.0),
        native_unit="mmol/L",
        notes=("MATLAB files are already oxy/deoxy, so raw optical-domain HOMER2 conversion cannot be replayed from this cache.",),
    ),
    "refed": Homer2DatasetCompatibility(
        dataset_id="refed",
        entry_stage="chromophore_or_absorbance_export",
        completeness="partial_post_conversion",
        available_inputs=("hbo_hbr_hbt_export", "absorbance_780_805_830_export", "channel_coordinates", "bad_channel_reservations"),
        missing_inputs=("raw_light_intensity", "declared_physical_units", "short_separation_channels"),
        possible_steps=("motion_detection", "motion_correction", "bandpass", "reservation_based_channel_masking"),
        blocked_steps=("intensity_to_optical_density", "modified_beer_lambert_from_raw", "short_channel_regression"),
        wavelengths_nm=(780.0, 805.0, 830.0),
        native_unit="unreported_LABNIRS_export",
        notes=("Absorbance exports help audit optical-domain behavior but do not restore raw intensity or a full HOMER2 chain.",),
    ),
    "visual_cognitive_motivation": Homer2DatasetCompatibility(
        dataset_id="visual_cognitive_motivation",
        entry_stage="chromophore",
        completeness="partial_post_conversion",
        available_inputs=("oxy_deoxy_csv_export", "sample_rate", "wavelength_metadata_695_830"),
        missing_inputs=("raw_695_830_intensity", "source_detector_geometry", "declared_physical_units", "short_separation_channels"),
        possible_steps=("motion_detection", "motion_correction", "bandpass"),
        blocked_steps=("intensity_to_optical_density", "modified_beer_lambert_from_raw", "short_channel_regression"),
        wavelengths_nm=(695.0, 830.0),
        native_unit="unreported_ETG7100_export",
        notes=("CSV Oxy/Deoxy exports can be cleaned as post-conversion traces but cannot be converted from raw intensity.",),
    ),
}


@dataclass(frozen=True)
class Homer2AlignmentState:
    dataset_id: str
    entry_stage: str
    sample_rate_hz: float
    applied_steps: tuple[str, ...]
    skipped_steps: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    parameters: Mapping[str, Any]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    schema: str = HOMER2_ALIGNMENT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("applied_steps", "skipped_steps", "missing_inputs", "input_shape", "output_shape"):
            payload[key] = list(payload[key])
        payload["parameters"] = dict(payload["parameters"])
        return payload


@dataclass(frozen=True)
class Homer2PreprocessResult:
    values: np.ndarray
    state: Homer2AlignmentState
    quality: Mapping[str, Any]
    pre_linear_values: np.ndarray | None = None
    pre_linear_optical_density: np.ndarray | None = None


def get_homer2_dataset_compatibility(dataset_id: str) -> Homer2DatasetCompatibility:
    try:
        return DATASET_HOMER2_COMPATIBILITY[str(dataset_id)]
    except KeyError as exc:
        raise KeyError(f"unknown HOMER2 compatibility dataset_id={dataset_id!r}") from exc


def homer2_compatibility_manifest() -> dict[str, Any]:
    return {
        "schema": HOMER2_ALIGNMENT_SCHEMA,
        "datasets": {
            dataset_id: compatibility.to_dict()
            for dataset_id, compatibility in sorted(DATASET_HOMER2_COMPATIBILITY.items())
        },
    }


def _as_float_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError(f"fNIRS values must have at least [time, channel] axes, got {array.shape}")
    if array.shape[0] < 4:
        raise ValueError(f"fNIRS record is too short for preprocessing: {array.shape}")
    return array


def _finite_interp(values: np.ndarray) -> tuple[np.ndarray, int]:
    flat = values.reshape(values.shape[0], -1).copy()
    repaired = 0
    x = np.arange(flat.shape[0], dtype=np.float64)
    for channel in range(flat.shape[1]):
        finite = np.isfinite(flat[:, channel])
        if not np.any(finite):
            flat[:, channel] = 0.0
            repaired += flat.shape[0]
            continue
        repaired += int(np.count_nonzero(~finite))
        if not np.all(finite):
            flat[:, channel] = np.interp(x, x[finite], flat[finite, channel])
    return flat.reshape(values.shape), repaired


def intensity_to_optical_density(
    intensity: np.ndarray,
    *,
    baseline: str = "median",
    epsilon: float = 1e-9,
) -> tuple[np.ndarray, dict[str, float]]:
    """Convert positive light intensity to optical density with provenance."""
    raw = _as_float_array(intensity)
    finite, repaired = _finite_interp(raw)
    positive = np.where(finite > epsilon, finite, epsilon)
    if baseline == "median":
        reference = np.median(positive, axis=0, keepdims=True)
    elif baseline == "mean":
        reference = np.mean(positive, axis=0, keepdims=True)
    else:
        raise ValueError(f"unsupported OD baseline rule: {baseline!r}")
    reference = np.where(reference > epsilon, reference, epsilon)
    optical_density = -np.log(positive / reference)
    quality = {
        "nonfinite_repaired": float(repaired),
        "clamped_nonpositive_fraction": float(np.mean(finite <= epsilon)),
        "od_abs_p99": float(np.quantile(np.abs(optical_density), 0.99)),
    }
    return optical_density, quality


def robust_derivative_motion_suppression(
    values: np.ndarray,
    *,
    tune: float = 4.685,
    epsilon: float = 1e-9,
) -> tuple[np.ndarray, dict[str, float]]:
    """TDDR-like robust derivative suppression for spike-heavy fNIRS traces."""
    array = _as_float_array(values)
    finite, repaired = _finite_interp(array)
    flat = finite.reshape(finite.shape[0], -1)
    derivative = np.diff(flat, axis=0, prepend=flat[:1])
    median = np.median(derivative, axis=0, keepdims=True)
    mad = 1.482602218505602 * np.median(np.abs(derivative - median), axis=0, keepdims=True)
    scale = np.where(mad > epsilon, mad, np.std(derivative, axis=0, keepdims=True))
    scale = np.where(scale > epsilon, scale, 1.0)
    z = (derivative - median) / scale
    weights = np.square(np.clip(1.0 - np.square(z / tune), 0.0, 1.0))
    corrected_derivative = median + weights * (derivative - median)
    corrected = np.cumsum(corrected_derivative, axis=0)
    corrected += flat[:1] - corrected[:1]
    outlier_fraction = float(np.mean(np.abs(z) > tune))
    return corrected.reshape(finite.shape), {
        "nonfinite_repaired": float(repaired),
        "motion_derivative_outlier_fraction": outlier_fraction,
        "median_derivative_weight": float(np.median(weights)),
    }


def bandpass_fnirs(
    values: np.ndarray,
    *,
    sample_rate_hz: float,
    low_hz: float = 0.01,
    high_hz: float = 0.2,
    order: int = 3,
) -> tuple[np.ndarray, dict[str, float | str]]:
    array = _as_float_array(values)
    finite, repaired = _finite_interp(array)
    nyquist = 0.5 * float(sample_rate_hz)
    if nyquist <= 0 or high_hz >= nyquist:
        return finite, {"status": "skipped_invalid_cutoff", "nonfinite_repaired": float(repaired)}
    sos = butter(int(order), [float(low_hz) / nyquist, float(high_hz) / nyquist], btype="bandpass", output="sos")
    flat = finite.reshape(finite.shape[0], -1)
    try:
        filtered = sosfiltfilt(sos, flat, axis=0)
    except ValueError:
        return finite, {"status": "skipped_record_too_short", "nonfinite_repaired": float(repaired)}
    return filtered.reshape(finite.shape), {
        "status": "applied",
        "low_hz": float(low_hz),
        "high_hz": float(high_hz),
        "order": float(order),
        "nonfinite_repaired": float(repaired),
    }


DEFAULT_EXTINCTION_COEFFICIENTS = {
    760.0: (0.148, 0.384),
    780.0: (0.180, 0.276),
    805.0: (0.223, 0.223),
    830.0: (0.244, 0.179),
    850.0: (0.252, 0.179),
}


def modified_beer_lambert(
    optical_density: np.ndarray,
    *,
    wavelengths_nm: Sequence[float],
    source_detector_distance_cm: float = 3.0,
    partial_pathlength_factor: float = 6.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert OD pairs to relative HbO/HbR estimates via an explicit MBLL assumption."""
    od = _as_float_array(optical_density)
    if od.ndim != 3:
        raise ValueError(f"MBLL expects [time, spatial_channel, wavelength], got {od.shape}")
    wavelengths = tuple(float(item) for item in wavelengths_nm)
    if len(wavelengths) != od.shape[2] or len(wavelengths) < 2:
        raise ValueError("wavelength count must match the OD wavelength axis and include at least two wavelengths")
    coefficients = []
    for wavelength in wavelengths[:2]:
        nearest = min(DEFAULT_EXTINCTION_COEFFICIENTS, key=lambda key: abs(key - wavelength))
        coefficients.append(DEFAULT_EXTINCTION_COEFFICIENTS[nearest])
    extinction = np.asarray(coefficients, dtype=np.float64)
    pathlength = float(source_detector_distance_cm) * float(partial_pathlength_factor)
    transform = np.linalg.pinv(extinction * pathlength)
    concentration = od[:, :, :2] @ transform.T
    quality = {
        "wavelengths_nm": list(wavelengths[:2]),
        "source_detector_distance_cm": float(source_detector_distance_cm),
        "partial_pathlength_factor": float(partial_pathlength_factor),
        "extinction_coefficients_source": "repo_approximate_table_for_alignment_audit_not_subject_calibrated",
        "condition_number": float(np.linalg.cond(extinction * pathlength)),
    }
    return concentration, quality


def apply_homer2_aligned_contract(
    values: np.ndarray,
    *,
    dataset_id: str,
    sample_rate_hz: float,
    entry_stage: str,
    wavelengths_nm: Sequence[float] = (),
    low_hz: float = 0.01,
    high_hz: float = 0.2,
    motion_correction: bool = True,
    source_detector_distance_cm: float = 3.0,
    partial_pathlength_factor: float = 6.0,
    retain_feature_boundary: bool = False,
) -> Homer2PreprocessResult:
    """Apply the best available HOMER2-aligned branch for one fNIRS record."""
    compatibility = get_homer2_dataset_compatibility(dataset_id)
    array = _as_float_array(values)
    applied: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []
    quality: dict[str, Any] = {
        "input_finite_fraction": float(np.isfinite(array).mean()),
        "compatibility": compatibility.to_dict(),
    }
    working = array

    if entry_stage == "raw_intensity":
        od, od_quality = intensity_to_optical_density(working)
        working = od
        applied.append("intensity_to_optical_density")
        quality["intensity_to_optical_density"] = od_quality
    else:
        skipped.append("intensity_to_optical_density")
        missing.append("raw_light_intensity")

    if motion_correction:
        working, motion_quality = robust_derivative_motion_suppression(working)
        applied.append("robust_derivative_motion_suppression")
        quality["motion_correction"] = motion_quality
    else:
        skipped.append("robust_derivative_motion_suppression")

    # Opt-in diagnostic view after all data-dependent nonlinear processing.
    # The established output path/order and its float32 boundary stay intact.
    pre_linear = np.array(working, dtype=float, copy=True) if retain_feature_boundary else None
    pre_od = pre_linear.copy() if retain_feature_boundary and entry_stage == "raw_intensity" else None
    working, filter_quality = bandpass_fnirs(
        working,
        sample_rate_hz=sample_rate_hz,
        low_hz=low_hz,
        high_hz=high_hz,
    )
    if filter_quality.get("status") == "applied":
        applied.append("bandpass")
    else:
        skipped.append("bandpass")
    quality["bandpass"] = filter_quality

    if entry_stage == "raw_intensity" and working.ndim == 3:
        concentration, mbll_quality = modified_beer_lambert(
            working,
            wavelengths_nm=wavelengths_nm,
            source_detector_distance_cm=source_detector_distance_cm,
            partial_pathlength_factor=partial_pathlength_factor,
        )
        working = concentration.reshape(concentration.shape[0], -1)
        if retain_feature_boundary:
            pre_linear, _ = modified_beer_lambert(
                pre_linear, wavelengths_nm=wavelengths_nm,
                source_detector_distance_cm=source_detector_distance_cm,
                partial_pathlength_factor=partial_pathlength_factor,
            )
        applied.append("modified_beer_lambert")
        quality["modified_beer_lambert"] = mbll_quality
    else:
        skipped.append("modified_beer_lambert")
        if entry_stage != "raw_intensity":
            missing.append("pre_conversion_optical_density")
        elif working.ndim != 3:
            missing.append("wavelength_axis")

    output = np.asarray(working, dtype=np.float32)
    quality["output_finite_fraction"] = float(np.isfinite(output).mean())
    quality["output_channel_std_median"] = float(np.median(np.nanstd(output.reshape(output.shape[0], -1), axis=0)))
    state = Homer2AlignmentState(
        dataset_id=str(dataset_id),
        entry_stage=str(entry_stage),
        sample_rate_hz=float(sample_rate_hz),
        applied_steps=tuple(applied),
        skipped_steps=tuple(skipped),
        missing_inputs=tuple(dict.fromkeys((*compatibility.missing_inputs, *missing))),
        parameters={
            "low_hz": float(low_hz),
            "high_hz": float(high_hz),
            "motion_correction": bool(motion_correction),
            "wavelengths_nm": [float(item) for item in wavelengths_nm],
            "source_detector_distance_cm": float(source_detector_distance_cm),
            "partial_pathlength_factor": float(partial_pathlength_factor),
        },
        input_shape=tuple(int(item) for item in array.shape),
        output_shape=tuple(int(item) for item in output.shape),
    )
    return Homer2PreprocessResult(values=output, state=state, quality=quality,
                                 pre_linear_values=pre_linear,
                                 pre_linear_optical_density=pre_od)
