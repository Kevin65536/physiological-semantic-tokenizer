"""Dataset-aware, semantics-preserving fNIRS measurement standardization.

The four datasets in this repository do not share a physical measurement
unit.  This module therefore does *not* pretend that voltage, absorbance, and
chromophore concentration are interchangeable.  It preserves the native
measurement family in metadata and maps each full continuous record to a
dimensionless deviation coordinate for numerical comparison and Croce-style
inference.

Standardization must be fitted on a full record before downstream cropping.
This keeps a sample invariant to crop position and prevents every crop from
silently defining a different baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


FNIRS_STANDARDIZATION_SCHEMA = "fnirs_measurement_standardization_v1"
MAD_TO_STD = 1.482602218505602


@dataclass(frozen=True)
class FNIRSMeasurementContract:
    dataset_id: str
    signal_key: str
    measurement_family: str
    native_unit: str
    channel_roles: tuple[str, ...]
    canonical_semantics: str = "dimensionless_native_semantics_deviation"
    schema: str = FNIRS_STANDARDIZATION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["channel_roles"] = list(self.channel_roles)
        return payload


DATASET_FNIRS_CONTRACTS: dict[str, dict[str, FNIRSMeasurementContract]] = {
    "eeg_fnirs_single_trial": {
        "wavelength_pair": FNIRSMeasurementContract(
            dataset_id="eeg_fnirs_single_trial",
            signal_key="wavelength_pair",
            measurement_family="optical_intensity",
            native_unit="V",
            channel_roles=("lowWL_760nm", "highWL_850nm"),
        ),
    },
    "refed": {
        "hbo_hbr": FNIRSMeasurementContract(
            dataset_id="refed",
            signal_key="hbo_hbr",
            measurement_family="chromophore_export",
            native_unit="unreported_LABNIRS_export",
            channel_roles=("HbO", "HbR"),
        ),
        "absorbance_780_805_830": FNIRSMeasurementContract(
            dataset_id="refed",
            signal_key="absorbance_780_805_830",
            measurement_family="absorbance",
            native_unit="unreported_absorbance",
            channel_roles=("Abs780", "Abs805", "Abs830"),
        ),
    },
    "visual_cognitive_motivation": {
        "oxy_deoxy": FNIRSMeasurementContract(
            dataset_id="visual_cognitive_motivation",
            signal_key="oxy_deoxy",
            measurement_family="chromophore_export",
            native_unit="unreported_ETG7100_export",
            channel_roles=("Oxy", "Deoxy"),
        ),
    },
    "simultaneous_eeg_nirs": {
        "oxy_deoxy": FNIRSMeasurementContract(
            dataset_id="simultaneous_eeg_nirs",
            signal_key="oxy_deoxy",
            measurement_family="chromophore_concentration",
            native_unit="mmol/L",
            channel_roles=("Oxy", "Deoxy"),
        ),
    },
}

DEFAULT_FNIRS_SIGNAL_KEYS = {
    "eeg_fnirs_single_trial": "wavelength_pair",
    "refed": "hbo_hbr",
    "visual_cognitive_motivation": "oxy_deoxy",
    "simultaneous_eeg_nirs": "oxy_deoxy",
}


def get_fnirs_measurement_contract(dataset_id: str, signal_key: str) -> FNIRSMeasurementContract:
    try:
        return DATASET_FNIRS_CONTRACTS[str(dataset_id)][str(signal_key)]
    except KeyError as exc:
        available = sorted(DATASET_FNIRS_CONTRACTS.get(str(dataset_id), {}))
        raise KeyError(
            f"No fNIRS measurement contract for dataset={dataset_id!r}, "
            f"signal_key={signal_key!r}; available={available}"
        ) from exc


def _as_time_channels(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"fNIRS values must have shape [time, channels], got {array.shape}")
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"fNIRS record is too small to standardize: {array.shape}")
    return array


def _interpolate_nonfinite(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    output = values.copy()
    repaired = np.zeros(values.shape[1], dtype=np.int64)
    index = np.arange(values.shape[0], dtype=np.float64)
    for channel in range(values.shape[1]):
        finite = np.isfinite(values[:, channel])
        if not np.any(finite):
            raise ValueError(f"fNIRS channel {channel} contains no finite samples")
        repaired[channel] = int(np.count_nonzero(~finite))
        if not np.all(finite):
            output[:, channel] = np.interp(index, index[finite], values[finite, channel])
    return output, repaired


def _robust_scale(values: np.ndarray, epsilon: float) -> np.ndarray:
    median = np.median(values, axis=0)
    scale = MAD_TO_STD * np.median(np.abs(values - median[None, :]), axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    iqr_scale = (q75 - q25) / 1.3489795003921634
    std_scale = np.std(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale >= epsilon), scale, iqr_scale)
    scale = np.where(np.isfinite(scale) & (scale >= epsilon), scale, std_scale)
    return np.where(np.isfinite(scale) & (scale >= epsilon), scale, 1.0)


def _block_median_linear_trend(values: np.ndarray, sample_rate_hz: float, blocks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = values.shape[0]
    n_blocks = max(2, min(int(blocks), n_samples))
    edges = np.linspace(0, n_samples, n_blocks + 1, dtype=np.int64)
    block_centres = []
    block_values = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        block_centres.append((start + end - 1) * 0.5 / sample_rate_hz)
        block_values.append(np.median(values[start:end], axis=0))
    x = np.asarray(block_centres, dtype=np.float64)
    y = np.asarray(block_values, dtype=np.float64)
    x_centre = float(np.mean(x))
    x0 = x - x_centre
    denominator = float(np.dot(x0, x0))
    slope = np.zeros(values.shape[1], dtype=np.float64) if denominator <= 0 else (x0[:, None] * y).sum(axis=0) / denominator
    intercept = np.median(y - x[:, None] * slope[None, :], axis=0)
    time_s = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    trend = intercept[None, :] + time_s[:, None] * slope[None, :]
    return trend, intercept, slope


@dataclass(frozen=True)
class FNIRSStandardizationState:
    contract: FNIRSMeasurementContract
    sample_rate_hz: float
    baseline_rule: str
    baseline_intercept: tuple[float, ...]
    baseline_slope_per_s: tuple[float, ...]
    channel_scale: tuple[float, ...]
    repaired_nonfinite: tuple[int, ...]
    schema: str = FNIRS_STANDARDIZATION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = self.contract.to_dict()
        for key in ("baseline_intercept", "baseline_slope_per_s", "channel_scale", "repaired_nonfinite"):
            payload[key] = list(payload[key])
        return payload


@dataclass(frozen=True)
class FNIRSStandardizationResult:
    values: np.ndarray
    state: FNIRSStandardizationState
    quality: Mapping[str, Any]


def standardize_fnirs_record(
    values: np.ndarray,
    *,
    sample_rate_hz: float,
    contract: FNIRSMeasurementContract,
    baseline_rule: str = "robust_linear",
    trend_blocks: int = 20,
    epsilon: float = 1e-8,
) -> FNIRSStandardizationResult:
    """Fit and apply record-level baseline/drift removal and robust scaling."""
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    raw = _as_time_channels(values)
    finite, repaired = _interpolate_nonfinite(raw)

    if baseline_rule == "median":
        intercept = np.median(finite, axis=0)
        slope = np.zeros(finite.shape[1], dtype=np.float64)
        trend = np.broadcast_to(intercept[None, :], finite.shape)
    elif baseline_rule == "robust_linear":
        trend, intercept, slope = _block_median_linear_trend(finite, sample_rate_hz, trend_blocks)
    else:
        raise ValueError(f"unsupported fNIRS baseline rule: {baseline_rule!r}")

    residual = finite - trend
    # The blockwise robust regression estimates slope well but its intercept can
    # retain a channel-specific offset when a record contains long asymmetric
    # task blocks.  Absorb that residual median into the reversible baseline.
    residual_offset = np.median(residual, axis=0)
    intercept = intercept + residual_offset
    trend = trend + residual_offset[None, :]
    residual = residual - residual_offset[None, :]
    scale = _robust_scale(residual, epsilon)
    canonical = residual / scale[None, :]
    _, _, residual_slope = _block_median_linear_trend(canonical, sample_rate_hz, trend_blocks)
    edges = np.linspace(0, canonical.shape[0], max(2, min(int(trend_blocks), canonical.shape[0])) + 1, dtype=np.int64)
    block_medians = np.asarray([
        np.median(canonical[start:end], axis=0)
        for start, end in zip(edges[:-1], edges[1:])
        if end > start
    ])
    block_range = np.ptp(block_medians, axis=0)
    state = FNIRSStandardizationState(
        contract=contract,
        sample_rate_hz=float(sample_rate_hz),
        baseline_rule=baseline_rule,
        baseline_intercept=tuple(float(item) for item in intercept),
        baseline_slope_per_s=tuple(float(item) for item in slope),
        channel_scale=tuple(float(item) for item in scale),
        repaired_nonfinite=tuple(int(item) for item in repaired),
    )
    quality = {
        "raw_finite_fraction": float(np.isfinite(raw).mean()),
        "raw_channel_std_median": float(np.median(np.nanstd(raw, axis=0))),
        "native_drift_abs_per_min_median": float(np.median(np.abs(slope)) * 60.0),
        "canonical_channel_median_abs_max": float(np.max(np.abs(np.median(canonical, axis=0)))),
        "canonical_robust_scale_median": float(np.median(_robust_scale(canonical, epsilon))),
        "residual_drift_sd_per_min_median": float(np.median(np.abs(residual_slope)) * 60.0),
        "canonical_block_median_range_median": float(np.median(block_range)),
        "canonical_block_median_range_max": float(np.max(block_range)),
    }
    return FNIRSStandardizationResult(
        values=canonical.astype(np.float32),
        state=state,
        quality=quality,
    )


def restore_fnirs_record(canonical: np.ndarray, state: FNIRSStandardizationState, *, start_sample: int = 0) -> np.ndarray:
    """Invert a standardized full record or crop using its full-record state."""
    values = _as_time_channels(canonical)
    scale = np.asarray(state.channel_scale, dtype=np.float64)
    intercept = np.asarray(state.baseline_intercept, dtype=np.float64)
    slope = np.asarray(state.baseline_slope_per_s, dtype=np.float64)
    if values.shape[1] != scale.size:
        raise ValueError("canonical channel count does not match standardization state")
    time_s = (int(start_sample) + np.arange(values.shape[0], dtype=np.float64)) / state.sample_rate_hz
    trend = intercept[None, :] + time_s[:, None] * slope[None, :]
    return values * scale[None, :] + trend


def standardization_config_from_mapping(config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate the optional loader preprocessing block."""
    if not isinstance(config, Mapping):
        return None
    block = config.get("measurement_standardization")
    if not isinstance(block, Mapping) or not bool(block.get("enabled", False)):
        return None
    return dict(block)


def default_standardization_config(dataset_id: str) -> dict[str, Any] | None:
    signal_key = DEFAULT_FNIRS_SIGNAL_KEYS.get(str(dataset_id))
    if signal_key is None:
        return None
    return {
        "enabled": True,
        "schema": FNIRS_STANDARDIZATION_SCHEMA,
        "dataset_id": str(dataset_id),
        "signal_key": signal_key,
        "baseline_rule": "robust_linear",
        "trend_blocks": 20,
    }
