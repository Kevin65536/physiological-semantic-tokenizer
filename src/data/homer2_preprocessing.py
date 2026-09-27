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
MEASUREMENT_ALIGNMENT_SCHEMA = "physiology_measurement_alignment_v3"
MEASUREMENT_ALIGNMENT_V4_SCHEMA = "physiology_measurement_alignment_v4"
MEASUREMENT_ALIGNMENT_V5_SCHEMA = "physiology_measurement_alignment_v5"
MEASUREMENT_ALIGNMENT_SCHEMAS = frozenset({
    MEASUREMENT_ALIGNMENT_SCHEMA, MEASUREMENT_ALIGNMENT_V4_SCHEMA, MEASUREMENT_ALIGNMENT_V5_SCHEMA,
})


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
    recorded_mask: np.ndarray | None = None
    processed_valid_mask: np.ndarray | None = None


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
    """Legacy derivative suppression retained for exact V1/V3 replay.

    Asymmetric derivative weights can introduce a cumulative drift even for
    a zero-net-change input. This is not canonical TDDR; new optical work
    should declare a V4 motion method explicitly.
    """
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


def _mne_tddr_optical_density(values, sample_rate_hz, wavelengths_nm):
    """Official public TDDR API on [time, pair, wavelength] natural OD.

    Synthetic channel geometry serves only MNE's channel contract, not an
    anatomical measurement. Correction is independently applied per channel.
    """
    import mne
    from mne.preprocessing.nirs import temporal_derivative_distribution_repair

    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    labels = [f'S{pair+1}_D{pair+1} {wavelength:g}'
              for pair in range(values.shape[1]) for wavelength in wavelengths]
    info = mne.create_info(labels, sample_rate_hz, ch_types=['fnirs_od']*len(labels))
    for index, ch in enumerate(info['chs']):
        ch['loc'][:3] = [.015, 0., 0.]
        ch['loc'][3:6] = [0., 0., 0.]
        ch['loc'][6:9] = [.03, 0., 0.]
        ch['loc'][9] = wavelengths[index % 2]
    raw = mne.io.RawArray(values.reshape(len(values), -1).T, info, verbose=False)
    corrected = temporal_derivative_distribution_repair(raw, verbose=False).get_data().T
    return corrected.reshape(values.shape), dict(
        implementation='mne.preprocessing.nirs.temporal_derivative_distribution_repair',
        mne_version=mne.__version__, channel_labels=labels,
        geometry='synthetic_API_metadata_not_anatomical_calibration',
        interpretation='motion_algorithm_not_ground_truth; may_remove_true_slow_trends')


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
    extinction_coefficients: np.ndarray | None = None,
    extinction_unit: str | None = None,
    od_log_base: str = "e",
    coefficient_log_base: str = "e",
    coefficient_source: str = "",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert OD pairs to relative HbO/HbR estimates via an explicit MBLL assumption."""
    od = _as_float_array(optical_density)
    if od.ndim != 3:
        raise ValueError(f"MBLL expects [time, spatial_channel, wavelength], got {od.shape}")
    wavelengths = tuple(float(item) for item in wavelengths_nm)
    if len(wavelengths) != od.shape[2] or len(wavelengths) < 2:
        raise ValueError("wavelength count must match the OD wavelength axis and include at least two wavelengths")
    if len(wavelengths) != 2 or len(set(wavelengths)) != 2:
        raise ValueError("this MBLL contract requires exactly two distinct wavelengths")
    if not (np.isfinite(source_detector_distance_cm) and source_detector_distance_cm > 0
            and np.isfinite(partial_pathlength_factor) and partial_pathlength_factor > 0):
        raise ValueError("MBLL distance and pathlength factor must be positive and finite")
    if od_log_base not in ("e", "10") or coefficient_log_base not in ("e", "10"):
        raise ValueError("explicit natural or base-10 OD/coefficient convention required")
    if extinction_coefficients is None:
        if any(w not in DEFAULT_EXTINCTION_COEFFICIENTS for w in wavelengths):
            raise ValueError("unknown wavelength: no silent nearest-wavelength substitution")
        extinction = np.array([DEFAULT_EXTINCTION_COEFFICIENTS[w] for w in wavelengths])
        output_unit = "relative_Hb_repo_approximation"
        source = "repo_approximate_table_for_alignment_audit_not_subject_calibrated"
    else:
        extinction = np.asarray(extinction_coefficients, dtype=np.float64)
        if extinction_unit != "1/(M cm)" or not coefficient_source:
            raise ValueError("physical MBLL requires coefficient units 1/(M cm) and a source")
        if extinction.shape != (2, 2) or not np.isfinite(extinction).all():
            raise ValueError("extinction coefficients must be finite [wavelength, HbO/HbR]")
        output_unit, source = "uM", coefficient_source
    if np.linalg.matrix_rank(extinction) != 2:
        raise ValueError("rank deficient extinction matrix")
    log_factor = (np.log(10.) if od_log_base == "10" else 1.) / (
        np.log(10.) if coefficient_log_base == "10" else 1.)
    pathlength = float(source_detector_distance_cm) * float(partial_pathlength_factor)
    transform = np.linalg.pinv(extinction * pathlength) * log_factor
    if output_unit == "uM":
        transform *= 1e6
    concentration = od[:, :, :2] @ transform.T
    quality = {
        "wavelengths_nm": list(wavelengths[:2]),
        "source_detector_distance_cm": float(source_detector_distance_cm),
        "partial_pathlength_factor": float(partial_pathlength_factor),
        "extinction_coefficients_source": source,
        "extinction_coefficients": extinction.tolist(),
        "extinction_unit": extinction_unit or "unverified_relative_table",
        "od_log_base": od_log_base,
        "coefficient_log_base": coefficient_log_base,
        "output_unit": output_unit,
        "transform": transform.tolist(),
        "absolute_baseline_concentration_known": False,
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
    motion_method: str | None = None,
    source_detector_distance_cm: float = 3.0,
    partial_pathlength_factor: float = 6.0,
    retain_feature_boundary: bool = False,
    processing_schema: str = HOMER2_ALIGNMENT_SCHEMA,
    native_unit: str = "unknown",
    unit_evidence: str = "",
    valid_mask: np.ndarray | None = None,
) -> Homer2PreprocessResult:
    """Apply the declared branch over exactly the supplied time support.

    V4 requires motion_method='none' or 'mne_tddr' on intensity/OD input.
    V5 additionally admits released chromophores with explicit 'none', without
    repeating optical conversion or motion correction on an upstream Hb export.
    That method exclusively controls motion correction; the legacy
    motion_correction boolean is unused in V4. V1/V3 retain their original
    boolean behavior and reject a non-None motion_method. V4 retains V3's
    relative Hb mapping, not an independently calibrated concentration scale.
    """
    compatibility = get_homer2_dataset_compatibility(dataset_id)
    v4 = processing_schema == MEASUREMENT_ALIGNMENT_V4_SCHEMA
    v5 = processing_schema == MEASUREMENT_ALIGNMENT_V5_SCHEMA
    explicit_motion = v4 or v5
    new_measurement = processing_schema in MEASUREMENT_ALIGNMENT_SCHEMAS
    optical = entry_stage in ('raw_intensity', 'optical_density')
    if processing_schema not in MEASUREMENT_ALIGNMENT_SCHEMAS | {HOMER2_ALIGNMENT_SCHEMA}:
        raise ValueError("unsupported measurement alignment schema")
    if explicit_motion:
        if motion_method not in ('none', 'mne_tddr'):
            raise ValueError("V4/V5 requires explicit motion_method 'none' or 'mne_tddr'")
        if v4 and not optical:
            raise ValueError('V4 supports only raw_intensity or optical_density')
        if v5 and not optical and (entry_stage != 'chromophore' or motion_method != 'none'):
            raise ValueError("V5 released chromophores require motion_method='none'; MNE is an optical-only comparison")
        if optical:
            wavelengths = np.asarray(wavelengths_nm, dtype=float)
            if (wavelengths.shape != (2,) or not np.isfinite(wavelengths).all()
                    or np.any(wavelengths <= 0) or wavelengths[0] >= wavelengths[1]):
                raise ValueError('V4/V5 requires two finite positive increasing wavelengths')
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError('V4/V5 requires finite positive sample rate')
    elif motion_method is not None:
        raise ValueError('motion_method is only supported by V4/V5')
    if entry_stage not in ("raw_intensity", "optical_density", "chromophore", "absorbance"):
        raise ValueError("explicit intensity, OD, chromophore or absorbance entry required")
    if entry_stage == "raw_intensity" and dataset_id != "eeg_fnirs_single_trial":
        raise ValueError("released chromophores cannot re-enter intensity/MBLL processing")
    array = _as_float_array(values)
    if explicit_motion and optical and (array.ndim != 3 or array.shape[-1] != 2 or array.shape[0] < 4 or array.shape[1] < 1):
        raise ValueError('V4/V5 requires [time>=4, pair>=1, two wavelengths]')
    if v5 and not optical and (array.ndim != 2 or array.shape[0] < 4 or array.shape[1] < 2 or array.shape[1] % 2):
        raise ValueError('V5 chromophores require [time>=4, paired HbO/HbR channels]')
    applied: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []
    quality: dict[str, Any] = {
        "input_finite_fraction": float(np.isfinite(array).mean()),
        "compatibility": compatibility.to_dict(),
    }
    working = array
    recorded = np.isfinite(array)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask,dtype=bool)
        if supplied.ndim == 1:
            supplied = supplied.reshape((-1,)+(1,)*(array.ndim-1))
        recorded &= np.broadcast_to(supplied,array.shape)
    if entry_stage == 'raw_intensity':
        recorded &= array > 0
    if new_measurement:
        working = np.where(recorded,array,np.nan)
        quality['recorded_fraction'] = float(recorded.mean())
        quality['missing_policy'] = 'mask_before_nonlinear_processing; visible_only_interpolation; no_new_measurements'
    if new_measurement and entry_stage == "chromophore":
        from .physiology_measurement_adapter import measurement_unit_conversion
        conversion = measurement_unit_conversion(native_unit, quantity="concentration_change",
                                                  evidence=unit_evidence, group=dataset_id)
        working = working * conversion['factor']
        quality['unit_conversion'] = conversion

    if entry_stage == "raw_intensity":
        od, od_quality = intensity_to_optical_density(
            working,epsilon=np.finfo(float).tiny if new_measurement else 1e-9)
        working = od
        applied.append("intensity_to_optical_density")
        quality["intensity_to_optical_density"] = od_quality
    else:
        skipped.append("intensity_to_optical_density")
        missing.append("raw_light_intensity")

    if explicit_motion:
        working, repaired = _finite_interp(working)
        quality['motion_input_nonfinite_repaired'] = float(repaired)
        quality['motion_method'] = motion_method
        if motion_method == 'mne_tddr':
            working, motion_quality = _mne_tddr_optical_density(working, sample_rate_hz, wavelengths_nm)
            quality['motion_correction'] = motion_quality
            applied.append('mne_tddr')
        else:
            skipped.append('motion_correction')
    elif motion_correction:
        working, motion_quality = robust_derivative_motion_suppression(working)
        applied.append("robust_derivative_motion_suppression")
        quality["motion_correction"] = motion_quality
    else:
        skipped.append("robust_derivative_motion_suppression")

    # Keep the old branch's rounding/order for retained replay. New measurement
    # coordinates apply MBLL before linear filtering and remain float64.
    pre_linear = np.array(working, dtype=float, copy=True) if retain_feature_boundary else None
    pre_od = pre_linear.copy() if retain_feature_boundary and entry_stage in ("raw_intensity", "optical_density") else None
    if optical and (array.ndim != 3 or array.shape[-1] != 2):
        raise ValueError("intensity/OD requires [time, pair, two wavelengths]")
    if optical and new_measurement:
        working, mbll_quality = modified_beer_lambert(
            working, wavelengths_nm=wavelengths_nm,
            source_detector_distance_cm=source_detector_distance_cm,
            partial_pathlength_factor=partial_pathlength_factor)
        if retain_feature_boundary:
            pre_linear = working.copy()
        working = working.reshape(len(working), -1)
        quality['modified_beer_lambert'] = mbll_quality
        applied.append('modified_beer_lambert')
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

    if optical and not new_measurement:
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
    elif not (optical and new_measurement):
        skipped.append("modified_beer_lambert")
        if entry_stage != "raw_intensity":
            missing.append("pre_conversion_optical_density")
        elif working.ndim != 3:
            missing.append("wavelength_axis")

    output = np.asarray(working, dtype=np.float64 if new_measurement else np.float32)
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
            "motion_correction": motion_method != 'none' if explicit_motion else bool(motion_correction),
            **({"motion_method": motion_method, "legacy_motion_correction_argument": "unused_in_v5" if v5 else "unused_in_v4"} if explicit_motion else {}),
            "wavelengths_nm": [float(item) for item in wavelengths_nm],
            "source_detector_distance_cm": float(source_detector_distance_cm),
            "partial_pathlength_factor": float(partial_pathlength_factor),
            "input_dtype": str(np.asarray(values).dtype),
            "feature_dtype": "float64",
            "output_dtype": str(output.dtype),
            "native_unit": native_unit,
            "unit_evidence": unit_evidence,
            "linear_order": "MBLL_then_filter" if new_measurement and optical else "filter_then_MBLL_or_published_Hb",
            "time_support_seconds": len(array)/float(sample_rate_hz),
            "low_frequency_cycles_in_support": len(array)/float(sample_rate_hz)*float(low_hz),
            "causal": False,
            "filter_edge_policy": "scipy_sosfiltfilt_odd_padding_within_declared_input_support",
        },
        input_shape=tuple(int(item) for item in array.shape),
        output_shape=tuple(int(item) for item in output.shape),
        schema=processing_schema,
    )
    # A two-sided IIR output depends on the whole admitted input interval. Any
    # imputed value therefore invalidates strict processed support for that
    # channel, while the original sample support remains available separately.
    supported = recorded.all(axis=0)
    if optical:
        supported = np.repeat(supported.all(axis=-1),2)
    supported = np.broadcast_to(supported.reshape(-1),output.shape).copy()
    return Homer2PreprocessResult(values=output, state=state, quality=quality,
                                 pre_linear_values=pre_linear,
                                 pre_linear_optical_density=pre_od,
                                 recorded_mask=recorded,processed_valid_mask=supported)
