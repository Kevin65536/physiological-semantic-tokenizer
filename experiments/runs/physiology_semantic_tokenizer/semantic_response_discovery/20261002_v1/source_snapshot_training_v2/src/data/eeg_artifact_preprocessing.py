"""Versioned, record-level EEG artifact cleaning for Single-Trial EEG.

The implementation is deliberately conservative: ocular activity is removed
with robust EOG regression, while only the 30–45 Hz component inside detected
transient bursts is tapered out.  The detected intervals remain explicitly
masked after correction.  All thresholds are configurable and interpreted
relative to record-level robust reference distributions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch


SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V2 = "single_trial_eeg_artifact_clean_v2"
SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V3 = "single_trial_eeg_artifact_clean_v3"
SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4 = "single_trial_eeg_artifact_clean_v4"
SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA = SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4


@dataclass(frozen=True)
class EEGArtifactCleaningConfig:
    schema: str = SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA
    band_hz: tuple[float, float] = (1.0, 45.0)
    line_noise_frequency_hz: float | None = 50.0
    line_noise_quality_factor: float = 30.0
    bad_channel_action: str = "disabled_for_cross_dataset_uniformity"
    bad_channel_robust_z: float = 4.5
    bad_channel_extreme_robust_z: float = 9.0
    bad_channel_metric_count: int = 2
    max_bad_channel_fraction: float = 0.2
    ocular_envelope_robust_z: float = 7.0
    ocular_velocity_robust_z: float = 25.0
    high_frequency_window_robust_z: float = 4.0
    high_frequency_band_hz: tuple[float, float] = (30.0, 45.0)
    burst_window_s: float = 1.0
    mask_dilation_s: float = 0.15
    muscle_action: str = "mask_gated_high_frequency_attenuation_v1"
    muscle_attenuation_strength: float = 1.0
    muscle_taper_s: float = 0.2
    eog_lag_s: tuple[float, ...] = (-0.05, 0.0, 0.05)
    eog_ridge: float = 1e-3
    huber_delta: float = 1.5
    regression_iterations: int = 2
    max_regression_samples: int = 30_000
    max_removed_variance_fraction: float = 0.5
    interpolation_neighbors: int = 4
    reference_strategy: str = "native_linked_mastoids_preserved"
    calibration_scope: str = "record_robust_distribution_v2"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("band_hz", "high_frequency_band_hz", "eog_lag_s"):
            payload[key] = list(payload[key])
        return payload


@dataclass
class EEGArtifactCleaningResult:
    cleaned_values: np.ndarray
    filtered_raw_values: np.ndarray
    artifact_mask: np.ndarray
    ocular_mask: np.ndarray
    high_frequency_mask: np.ndarray
    bad_channel_mask: np.ndarray
    state: dict[str, Any]


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
    low, high = map(float, band_hz)
    nyquist = 0.5 * float(sample_rate_hz)
    high = min(high, nyquist * 0.95)
    if values.shape[0] < 32 or low <= 0 or high <= low:
        return values.copy()
    sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, values, axis=0)


def _remove_line_noise(
    values: np.ndarray,
    sample_rate_hz: float,
    config: EEGArtifactCleaningConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    frequency_hz = config.line_noise_frequency_hz
    if frequency_hz is None:
        return values.copy(), {
            "method": "disabled",
            "frequency_hz": None,
            "quality_factor": None,
        }
    frequency_hz = float(frequency_hz)
    quality_factor = float(config.line_noise_quality_factor)
    nyquist = 0.5 * float(sample_rate_hz)
    if frequency_hz <= 0.0 or frequency_hz >= nyquist:
        raise ValueError(
            f"line-noise frequency must be inside (0, Nyquist): "
            f"{frequency_hz} Hz for fs={sample_rate_hz} Hz"
        )
    if quality_factor <= 0.0:
        raise ValueError(f"line-noise quality factor must be positive: {quality_factor}")
    if values.shape[0] < 32:
        return values.copy(), {
            "method": "skipped_short_record",
            "frequency_hz": frequency_hz,
            "quality_factor": quality_factor,
        }
    numerator, denominator = iirnotch(
        frequency_hz,
        quality_factor,
        fs=float(sample_rate_hz),
    )
    return filtfilt(numerator, denominator, values, axis=0), {
        "method": "zero_phase_iirnotch",
        "frequency_hz": frequency_hz,
        "quality_factor": quality_factor,
    }


def _robust_location_scale(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    location = np.median(array, axis=axis)
    mad = 1.482602218505602 * np.median(np.abs(array - np.expand_dims(location, axis=axis)), axis=axis)
    std = np.std(array, axis=axis)
    scale = np.where(np.isfinite(mad) & (mad > 1e-12), mad, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return location, scale


def _robust_positive_z(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    location = float(np.median(array))
    scale = float(1.482602218505602 * np.median(np.abs(array - location)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(array))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(array)
    return np.maximum((array - location) / scale, 0.0)


def _band_power(
    values: np.ndarray,
    sample_rate_hz: float,
    band_hz: tuple[float, float],
    *,
    reference_band_hz: tuple[float, float] | None = None,
) -> np.ndarray:
    nperseg = min(values.shape[0], max(int(round(sample_rate_hz * 4.0)), 64))
    frequencies, density = welch(values, fs=sample_rate_hz, nperseg=nperseg, axis=0)
    selected = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    power = np.trapezoid(density[selected], frequencies[selected], axis=0) if np.any(selected) else np.zeros(values.shape[1])
    if reference_band_hz is None:
        return power
    reference = (frequencies >= reference_band_hz[0]) & (frequencies <= reference_band_hz[1])
    total = np.trapezoid(density[reference], frequencies[reference], axis=0) if np.any(reference) else np.ones(values.shape[1])
    return power / np.maximum(total, np.finfo(np.float64).eps)


def compute_channel_quality_metrics(values: np.ndarray, sample_rate_hz: float) -> dict[str, np.ndarray]:
    """Return channel-wise, record-level metrics without applying rejection."""
    array, _ = _interpolate_nonfinite(values)
    filtered = _bandpass(array, sample_rate_hz, (1.0, 45.0))
    _, robust_scale = _robust_location_scale(filtered, axis=0)
    differences = np.diff(array, axis=0)
    flat_fraction = np.mean(np.abs(differences) <= np.finfo(np.float64).eps, axis=0)
    stride = max(1, filtered.shape[0] // 20_000)
    reduced = filtered[::stride]
    _, reduced_scale = _robust_location_scale(reduced, axis=0)
    normalized = (reduced - np.median(reduced, axis=0)) / reduced_scale
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(normalized, rowvar=False)
    np.fill_diagonal(correlation, np.nan)
    absolute_correlation = np.abs(correlation)
    usable_rows = np.any(np.isfinite(absolute_correlation), axis=1)
    median_abs_correlation = np.zeros(values.shape[1], dtype=np.float64)
    median_abs_correlation[usable_rows] = np.nanmedian(absolute_correlation[usable_rows], axis=1)
    return {
        "robust_scale": robust_scale,
        "flat_fraction": flat_fraction,
        "median_abs_correlation": median_abs_correlation,
        "low_frequency_ratio": _band_power(filtered, sample_rate_hz, (1.0, 4.0), reference_band_hz=(1.0, 45.0)),
        "high_frequency_ratio": _band_power(filtered, sample_rate_hz, (30.0, 45.0), reference_band_hz=(1.0, 45.0)),
        "line_noise_ratio": _band_power(array, sample_rate_hz, (48.0, 52.0), reference_band_hz=(1.0, 80.0)),
    }


def detect_bad_channels(
    metrics: dict[str, np.ndarray],
    config: EEGArtifactCleaningConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    scale_z = _robust_positive_z(np.maximum(metrics["robust_scale"], 1e-12))
    inverse_scale_z = _robust_positive_z(1.0 / np.maximum(metrics["robust_scale"], 1e-12))
    low_z = _robust_positive_z(metrics["low_frequency_ratio"])
    high_z = _robust_positive_z(metrics["high_frequency_ratio"])
    line_z = _robust_positive_z(metrics["line_noise_ratio"])
    correlation_deficit_z = _robust_positive_z(1.0 - metrics["median_abs_correlation"])
    metric_z = np.vstack((scale_z, inverse_scale_z, low_z, high_z, line_z, correlation_deficit_z))
    score = np.max(metric_z, axis=0)
    consensus = np.count_nonzero(metric_z >= config.bad_channel_robust_z, axis=0)
    bad = (
        (consensus >= config.bad_channel_metric_count)
        | (score >= config.bad_channel_extreme_robust_z)
        | (metrics["flat_fraction"] > 0.01)
    )
    maximum = max(1, int(np.floor(len(bad) * config.max_bad_channel_fraction)))
    if np.count_nonzero(bad) > maximum:
        selected = np.argsort(score)[-maximum:]
        limited = np.zeros_like(bad)
        limited[selected] = True
        bad = limited
    return bad, {
        "quality_score": score,
        "metric_consensus_count": consensus,
        "scale_z": scale_z,
        "inverse_scale_z": inverse_scale_z,
        "low_frequency_z": low_z,
        "high_frequency_z": high_z,
        "line_noise_z": line_z,
        "correlation_deficit_z": correlation_deficit_z,
    }


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(mask):
        return mask.astype(bool, copy=True)
    kernel = np.ones(2 * radius + 1, dtype=np.int8)
    return np.convolve(mask.astype(np.int8), kernel, mode="same") > 0


def detect_ocular_mask(
    eog_values: np.ndarray,
    sample_rate_hz: float,
    config: EEGArtifactCleaningConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    eog, repaired = _interpolate_nonfinite(eog_values)
    eog = _bandpass(eog, sample_rate_hz, (0.5, 15.0))
    location, scale = _robust_location_scale(eog, axis=0)
    standardized = (eog - location) / scale
    derivative = np.diff(standardized, axis=0, prepend=standardized[:1]) * sample_rate_hz
    _, derivative_scale = _robust_location_scale(derivative, axis=0)
    derivative_z = np.abs(derivative) / derivative_scale
    smoothing = max(1, int(round(0.1 * sample_rate_hz)))
    kernel = np.ones(smoothing, dtype=np.float64) / smoothing
    amplitude_envelope = np.column_stack(
        [np.convolve(np.abs(standardized[:, channel]), kernel, mode="same") for channel in range(standardized.shape[1])]
    )
    velocity_envelope = np.column_stack(
        [np.convolve(derivative_z[:, channel], kernel, mode="same") for channel in range(derivative_z.shape[1])]
    )
    amplitude_z = np.max(amplitude_envelope, axis=1)
    velocity_z = np.max(velocity_envelope, axis=1)
    raw_mask = (amplitude_z >= config.ocular_envelope_robust_z) | (
        velocity_z >= config.ocular_velocity_robust_z
    )
    mask = _dilate_mask(raw_mask, int(round(config.mask_dilation_s * sample_rate_hz)))
    return mask, {
        "repaired_nonfinite_samples": repaired,
        "amplitude_z_p99": float(np.quantile(amplitude_z, 0.99)),
        "velocity_z_p99": float(np.quantile(velocity_z, 0.99)),
        "raw_fraction": float(np.mean(raw_mask)),
        "dilated_fraction": float(np.mean(mask)),
    }


def detect_high_frequency_mask(
    eeg_values: np.ndarray,
    sample_rate_hz: float,
    config: EEGArtifactCleaningConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    high = _bandpass(eeg_values, sample_rate_hz, config.high_frequency_band_hz)
    window = max(1, int(round(config.burst_window_s * sample_rate_hz)))
    count = int(np.ceil(len(high) / window))
    powers = np.zeros(count, dtype=np.float64)
    for index in range(count):
        segment = high[index * window : min((index + 1) * window, len(high))]
        powers[index] = float(np.median(np.sqrt(np.mean(segment * segment, axis=0))))
    robust_z = _robust_positive_z(powers)
    flagged = robust_z >= config.high_frequency_window_robust_z
    mask = np.repeat(flagged, window)[: len(high)]
    mask = _dilate_mask(mask, int(round(config.mask_dilation_s * sample_rate_hz)))
    return mask, {
        "window_s": config.burst_window_s,
        "window_count": count,
        "flagged_window_count": int(np.count_nonzero(flagged)),
        "dilated_fraction": float(np.mean(mask)),
        "power_median": float(np.median(powers)),
        "power_p99": float(np.quantile(powers, 0.99)),
    }


def correct_high_frequency_bursts(
    eeg_values: np.ndarray,
    high_frequency_mask: np.ndarray,
    sample_rate_hz: float,
    config: EEGArtifactCleaningConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Attenuate only 30–45 Hz content inside adaptively detected burst intervals."""
    values = _as_time_channels(eeg_values)
    mask = np.asarray(high_frequency_mask, dtype=bool).reshape(-1)
    if len(mask) != len(values):
        raise ValueError(f"EEG/muscle-mask length mismatch: {len(values)} != {len(mask)}")
    if config.muscle_action == "mask_only":
        return values.copy(), {
            "method": "mask_only",
            "corrected_sample_fraction": 0.0,
            "high_frequency_energy_reduction_in_mask": 0.0,
        }
    if config.muscle_action != "mask_gated_high_frequency_attenuation_v1":
        raise ValueError(f"unsupported muscle_action: {config.muscle_action!r}")
    high_frequency = _bandpass(values, sample_rate_hz, config.high_frequency_band_hz)
    taper_samples = max(1, int(round(config.muscle_taper_s * sample_rate_hz)))
    kernel = np.hanning(2 * taper_samples + 1)
    if not np.any(kernel):
        kernel = np.ones(2 * taper_samples + 1, dtype=np.float64)
    envelope = np.convolve(mask.astype(np.float64), kernel / np.sum(kernel), mode="same")
    envelope = np.clip(envelope, 0.0, 1.0)
    strength = float(np.clip(config.muscle_attenuation_strength, 0.0, 1.0))
    corrected = values - strength * envelope[:, None] * high_frequency
    evaluation_mask = mask if np.any(mask) else np.ones(len(mask), dtype=bool)
    before = float(np.mean(high_frequency[evaluation_mask] ** 2))
    residual_high = _bandpass(corrected, sample_rate_hz, config.high_frequency_band_hz)
    after = float(np.mean(residual_high[evaluation_mask] ** 2))
    reduction = 1.0 - after / max(before, np.finfo(np.float64).eps)
    return corrected, {
        "method": config.muscle_action,
        "attenuation_strength": strength,
        "taper_s": float(config.muscle_taper_s),
        "corrected_sample_fraction": float(np.mean(envelope > 0.0)),
        "mean_attenuation_envelope": float(np.mean(envelope)),
        "high_frequency_energy_before_in_mask": before,
        "high_frequency_energy_after_in_mask": after,
        "high_frequency_energy_reduction_in_mask": float(reduction),
    }


def _lagged_eog_design(
    eog_values: np.ndarray,
    sample_rate_hz: float,
    lags_s: Sequence[float],
) -> tuple[np.ndarray, list[str]]:
    eog, _ = _interpolate_nonfinite(eog_values)
    eog = _bandpass(eog, sample_rate_hz, (0.5, 15.0))
    location, scale = _robust_location_scale(eog, axis=0)
    eog = (eog - location) / scale
    columns = []
    names = []
    for lag_s in lags_s:
        shift = int(round(float(lag_s) * sample_rate_hz))
        shifted = np.zeros_like(eog)
        if shift < 0:
            shifted[:shift] = eog[-shift:]
        elif shift > 0:
            shifted[shift:] = eog[:-shift]
        else:
            shifted = eog.copy()
        columns.append(shifted)
        names.extend([f"eog{index}_lag_{lag_s:+.3f}s" for index in range(eog.shape[1])])
    return np.concatenate(columns, axis=1), names


def _robust_eog_regression(
    eeg_values: np.ndarray,
    eog_values: np.ndarray,
    ocular_mask: np.ndarray,
    sample_rate_hz: float,
    config: EEGArtifactCleaningConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    design, predictor_names = _lagged_eog_design(eog_values, sample_rate_hz, config.eog_lag_s)
    design = np.column_stack((np.ones(len(design)), design))
    stride = max(1, int(np.ceil(len(design) / config.max_regression_samples)))
    indices = np.arange(0, len(design), stride)
    x_fit = design[indices]
    y_fit = eeg_values[indices]
    artifact_weights = np.where(ocular_mask[indices], 4.0, 1.0)
    coefficients = np.zeros((design.shape[1], eeg_values.shape[1]), dtype=np.float64)
    removed_fraction = np.zeros(eeg_values.shape[1], dtype=np.float64)
    eye_design = np.eye(design.shape[1], dtype=np.float64)
    eye_design[0, 0] = 0.0
    for channel in range(eeg_values.shape[1]):
        weights = artifact_weights.copy()
        beta = np.zeros(design.shape[1], dtype=np.float64)
        for _ in range(max(1, config.regression_iterations)):
            xtw = x_fit.T * weights[None, :]
            beta = np.linalg.solve(xtw @ x_fit + config.eog_ridge * eye_design, xtw @ y_fit[:, channel])
            residual = y_fit[:, channel] - x_fit @ beta
            _, residual_scale = _robust_location_scale(residual, axis=0)
            standardized = np.abs(residual) / float(residual_scale)
            huber = np.minimum(1.0, config.huber_delta / np.maximum(standardized, 1e-12))
            weights = artifact_weights * huber
        coefficients[:, channel] = beta
        predicted = design[:, 1:] @ beta[1:]
        raw_variance = float(np.var(eeg_values[:, channel]))
        fraction = float(np.var(predicted) / max(raw_variance, np.finfo(np.float64).eps))
        if fraction > config.max_removed_variance_fraction:
            predicted *= np.sqrt(config.max_removed_variance_fraction / fraction)
            fraction = config.max_removed_variance_fraction
        eeg_values[:, channel] -= predicted
        removed_fraction[channel] = fraction
    return eeg_values, {
        "predictor_names": predictor_names,
        "fit_sample_count": int(len(indices)),
        "fit_stride": int(stride),
        "coefficients": coefficients.tolist(),
        "removed_variance_fraction": removed_fraction.tolist(),
    }


def _max_abs_eog_correlation(eeg_values: np.ndarray, eog_values: np.ndarray) -> np.ndarray:
    stride = max(1, len(eeg_values) // 20_000)
    eeg = eeg_values[::stride]
    eog = eog_values[::stride]
    eeg_location, eeg_scale = _robust_location_scale(eeg, axis=0)
    eog_location, eog_scale = _robust_location_scale(eog, axis=0)
    eeg = (eeg - eeg_location) / eeg_scale
    eog = (eog - eog_location) / eog_scale
    eeg = eeg - np.mean(eeg, axis=0)
    eog = eog - np.mean(eog, axis=0)
    denominator = np.sqrt(np.sum(eeg * eeg, axis=0))[:, None] * np.sqrt(
        np.sum(eog * eog, axis=0)
    )[None, :]
    block = np.divide(
        eeg.T @ eog,
        denominator,
        out=np.zeros((eeg.shape[1], eog.shape[1]), dtype=np.float64),
        where=denominator > np.finfo(np.float64).eps,
    )
    return np.max(np.abs(block), axis=1)


def _interpolate_bad_channels(
    values: np.ndarray,
    bad_mask: np.ndarray,
    neighbors: int,
    channel_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not np.any(bad_mask):
        return values, {"method": "none", "interpolated_channel_count": 0, "neighbors": {}}
    good_indices = np.flatnonzero(~bad_mask)
    bad_indices = np.flatnonzero(bad_mask)
    if len(good_indices) == 0:
        return values, {"method": "skipped_no_good_channels", "interpolated_channel_count": 0, "neighbors": {}}
    positions = None if channel_positions is None else np.asarray(channel_positions, dtype=np.float64)
    geometry_available = positions is not None and positions.shape == (values.shape[1], 3)
    stride = max(1, len(values) // 20_000)
    reduced = values[::stride]
    location, scale = _robust_location_scale(reduced, axis=0)
    normalized = (reduced - location) / scale
    output = values.copy()
    neighbor_map: dict[str, list[int]] = {}
    methods: set[str] = set()
    for bad in bad_indices:
        use_geometry = bool(
            geometry_available
            and np.all(np.isfinite(positions[bad]))
            and np.any(np.all(np.isfinite(positions[good_indices]), axis=1))
        )
        if use_geometry:
            positioned_good = good_indices[np.all(np.isfinite(positions[good_indices]), axis=1)]
            distances = np.linalg.norm(positions[positioned_good] - positions[bad], axis=1)
            order = np.argsort(distances)[: max(1, min(neighbors, len(positioned_good)))]
            selected = positioned_good[order]
            weights = 1.0 / np.maximum(distances[order], 1e-6)
            methods.add("geometry_inverse_distance")
        else:
            correlations = []
            for good in good_indices:
                correlation = float(np.corrcoef(normalized[:, bad], normalized[:, good])[0, 1])
                correlations.append(abs(correlation) if np.isfinite(correlation) else 0.0)
            order = np.argsort(correlations)[::-1][: max(1, min(neighbors, len(good_indices)))]
            selected = good_indices[order]
            weights = np.asarray(correlations, dtype=np.float64)[order]
            methods.add("correlation_weighted_fallback")
        weights = weights / max(float(np.sum(weights)), np.finfo(np.float64).eps)
        standardized_neighbors = (values[:, selected] - location[selected]) / scale[selected]
        replacement = standardized_neighbors @ weights
        target_location = float(np.median(location[selected]))
        target_scale = float(np.median(scale[selected]))
        output[:, bad] = replacement * target_scale + target_location
        neighbor_map[str(int(bad))] = [int(value) for value in selected]
    return output, {
        "method": "+".join(sorted(methods)),
        "geometry_available": bool(geometry_available),
        "interpolated_channel_count": int(len(bad_indices)),
        "neighbors": neighbor_map,
    }


def clean_single_trial_eeg(
    eeg_values: np.ndarray,
    eog_values: np.ndarray,
    *,
    sample_rate_hz: float,
    channel_names: Sequence[str] | None = None,
    eog_channel_names: Sequence[str] | None = None,
    channel_positions: np.ndarray | None = None,
    config: EEGArtifactCleaningConfig | None = None,
    output_dtype: str = "float32",
) -> EEGArtifactCleaningResult:
    """Clean one complete task recording and return masks plus provenance."""
    if output_dtype not in ('float32','float64'):
        raise ValueError('EEG cleaning output must be float32 or float64')
    cfg = config or EEGArtifactCleaningConfig()
    eeg, eeg_repaired = _interpolate_nonfinite(eeg_values)
    eog, eog_repaired = _interpolate_nonfinite(eog_values)
    resolved_channel_names = list(channel_names) if channel_names is not None else [
        f"EEG{index + 1}" for index in range(eeg.shape[1])
    ]
    resolved_eog_names = list(eog_channel_names) if eog_channel_names is not None else [
        f"EOG{index + 1}" for index in range(eog.shape[1])
    ]
    if len(eeg) != len(eog):
        raise ValueError(f"EEG/EOG length mismatch: {len(eeg)} != {len(eog)}")
    line_band = (
        max(0.0, float(cfg.line_noise_frequency_hz or 50.0) - 2.0),
        min(0.5 * float(sample_rate_hz), float(cfg.line_noise_frequency_hz or 50.0) + 2.0),
    )
    line_ratio_before = _band_power(
        eeg,
        sample_rate_hz,
        line_band,
        reference_band_hz=(1.0, min(80.0, 0.5 * float(sample_rate_hz))),
    )
    line_cleaned, line_noise_state = _remove_line_noise(eeg, sample_rate_hz, cfg)
    filtered = _bandpass(line_cleaned, sample_rate_hz, cfg.band_hz)
    line_ratio_after = _band_power(
        filtered,
        sample_rate_hz,
        line_band,
        reference_band_hz=(1.0, min(80.0, 0.5 * float(sample_rate_hz))),
    )
    eog_for_metrics = _bandpass(eog, sample_rate_hz, (0.5, 15.0))
    metrics_before = compute_channel_quality_metrics(filtered, sample_rate_hz)
    if cfg.bad_channel_action == "detect_and_interpolate":
        bad_mask, channel_quality_scores = detect_bad_channels(metrics_before, cfg)
    elif cfg.bad_channel_action == "disabled_for_cross_dataset_uniformity":
        bad_mask = np.zeros(eeg.shape[1], dtype=bool)
        _, channel_quality_scores = detect_bad_channels(metrics_before, cfg)
    else:
        raise ValueError(f"unsupported bad_channel_action: {cfg.bad_channel_action!r}")
    ocular_mask, ocular_state = detect_ocular_mask(eog, sample_rate_hz, cfg)
    high_frequency_mask, high_frequency_state = detect_high_frequency_mask(filtered, sample_rate_hz, cfg)
    correlation_before = _max_abs_eog_correlation(filtered, eog_for_metrics)
    regressed, regression_state = _robust_eog_regression(
        filtered.copy(), eog, ocular_mask, sample_rate_hz, cfg
    )
    if cfg.bad_channel_action == "detect_and_interpolate":
        interpolated, interpolation_state = _interpolate_bad_channels(
            regressed, bad_mask, cfg.interpolation_neighbors, channel_positions
        )
    else:
        interpolated = regressed
        interpolation_state = {
            "method": "disabled_for_cross_dataset_uniformity",
            "geometry_available": False,
            "interpolated_channel_count": 0,
            "neighbors": {},
        }
    corrected, muscle_correction_state = correct_high_frequency_bursts(
        interpolated, high_frequency_mask, sample_rate_hz, cfg
    )
    correlation_after = _max_abs_eog_correlation(corrected, eog_for_metrics)
    metrics_after = compute_channel_quality_metrics(corrected, sample_rate_hz)
    artifact_mask = ocular_mask | high_frequency_mask
    state = {
        "schema": cfg.schema,
        "config": cfg.to_dict(),
        "channel_names": resolved_channel_names,
        "eog_channel_names": resolved_eog_names,
        "sample_rate_hz": float(sample_rate_hz),
        "sample_count": int(len(eeg)),
        "input_repaired_nonfinite_samples": {"eeg": eeg_repaired, "eog": eog_repaired},
        "bad_channel_indices": np.flatnonzero(bad_mask).astype(int).tolist(),
        "bad_channel_names": [
            str(resolved_channel_names[index])
            for index in np.flatnonzero(bad_mask)
        ],
        "bad_channel_policy": {
            "action": cfg.bad_channel_action,
            "mask_emitted": bool(np.any(bad_mask)),
            "interpolation_applied": bool(
                interpolation_state["interpolated_channel_count"]
            ),
        },
        "channel_quality_scores": {
            key: np.asarray(value, dtype=float).tolist()
            for key, value in channel_quality_scores.items()
        },
        "metrics_before": {key: np.asarray(value, dtype=float).tolist() for key, value in metrics_before.items()},
        "metrics_after": {key: np.asarray(value, dtype=float).tolist() for key, value in metrics_after.items()},
        "ocular": ocular_state,
        "high_frequency": high_frequency_state,
        "muscle_correction": muscle_correction_state,
        "eog_regression": regression_state,
        "interpolation": interpolation_state,
        "eog_correlation_before": correlation_before.tolist(),
        "eog_correlation_after": correlation_after.tolist(),
        "median_eog_correlation_before": float(np.median(correlation_before)),
        "median_eog_correlation_after": float(np.median(correlation_after)),
        "artifact_fraction": float(np.mean(artifact_mask)),
        "reference_strategy": cfg.reference_strategy,
        "line_noise_action": line_noise_state["method"],
        "line_noise": {
            **line_noise_state,
            "audit_band_hz": list(line_band),
            "ratio_before_by_channel": line_ratio_before.tolist(),
            "ratio_after_by_channel": line_ratio_after.tolist(),
            "median_ratio_before": float(np.median(line_ratio_before)),
            "median_ratio_after": float(np.median(line_ratio_after)),
            "maximum_ratio_after": float(np.max(line_ratio_after)),
        },
        "muscle_action": cfg.muscle_action,
    }
    return EEGArtifactCleaningResult(
        cleaned_values=corrected.astype(output_dtype),
        filtered_raw_values=filtered.astype(output_dtype),
        artifact_mask=artifact_mask,
        ocular_mask=ocular_mask,
        high_frequency_mask=high_frequency_mask,
        bad_channel_mask=bad_mask,
        state=state,
    )


__all__ = [
    "SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA",
    "SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V2",
    "SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V3",
    "SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4",
    "EEGArtifactCleaningConfig",
    "EEGArtifactCleaningResult",
    "clean_single_trial_eeg",
    "compute_channel_quality_metrics",
    "correct_high_frequency_bursts",
    "detect_bad_channels",
    "detect_high_frequency_mask",
    "detect_ocular_mask",
]
