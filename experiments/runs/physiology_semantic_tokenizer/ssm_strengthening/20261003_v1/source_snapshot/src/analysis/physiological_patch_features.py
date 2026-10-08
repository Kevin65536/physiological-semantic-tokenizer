"""Versioned, channel-resolved physiological patch feature extraction.

The extractors in this module operate on the canonical robust-SD normalized
signals used by the physiology tokenizer.  Amplitude-derived features therefore
do *not* have physical voltage or concentration units.

Both patch-major ``[batch, patch, channel, sample]`` arrays and legacy
continuous ``[batch, channel, sample]`` arrays are accepted.  The latter require
``patch_size`` so that the token-aligned patches can be reconstructed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import warnings
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np


FEATURE_SPEC_SCHEMA_VERSION = "physiological_patch_features.v1"
CANONICAL_SIGNAL_UNIT = "canonical_robust_sd"


@dataclass(frozen=True)
class FrequencyBand:
    """Named half-open EEG frequency band ``[low_hz, high_hz)``."""

    name: str
    low_hz: float
    high_hz: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Frequency-band names must be non-empty")
        if not (0.0 <= float(self.low_hz) < float(self.high_hz)):
            raise ValueError(
                f"Invalid frequency band {self.name!r}: "
                f"{self.low_hz}--{self.high_hz} Hz"
            )


def _default_eeg_bands() -> Tuple[FrequencyBand, ...]:
    return (
        FrequencyBand("delta", 1.0, 4.0),
        FrequencyBand("theta", 4.0, 8.0),
        FrequencyBand("alpha", 8.0, 13.0),
        FrequencyBand("beta", 13.0, 30.0),
        FrequencyBand("low_gamma", 30.0, 45.0),
    )


@dataclass(frozen=True)
class PhysiologicalPatchFeatureSpec:
    """Immutable definition of the physiological patch feature contract."""

    schema_version: str = FEATURE_SPEC_SCHEMA_VERSION
    input_unit: str = CANONICAL_SIGNAL_UNIT
    eeg_bands: Tuple[FrequencyBand, ...] = field(default_factory=_default_eeg_bands)
    eeg_reference_band_hz: Tuple[float, float] = (1.0, 45.0)
    psd_window: str = "hann_symmetric"
    psd_detrend: str = "constant"
    log_epsilon: float = 1e-12
    minimum_valid_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.input_unit != CANONICAL_SIGNAL_UNIT:
            raise ValueError(
                "Physiological patch features currently require canonical "
                "robust-SD normalized inputs"
            )
        if self.psd_window != "hann_symmetric":
            raise ValueError("Only the audited symmetric Hann PSD is supported")
        if self.psd_detrend != "constant":
            raise ValueError("Only constant detrending is supported")
        if float(self.log_epsilon) <= 0.0:
            raise ValueError("log_epsilon must be positive")
        if not (0.0 < float(self.minimum_valid_fraction) <= 1.0):
            raise ValueError("minimum_valid_fraction must be in (0, 1]")
        reference_low, reference_high = self.eeg_reference_band_hz
        if not (0.0 <= float(reference_low) < float(reference_high)):
            raise ValueError("eeg_reference_band_hz must be increasing")
        band_names = [band.name for band in self.eeg_bands]
        if len(band_names) != len(set(band_names)):
            raise ValueError("EEG frequency-band names must be unique")

    def to_dict(self) -> Dict[str, Any]:
        """Return a canonical JSON-safe specification dictionary."""
        payload = asdict(self)
        payload["eeg_bands"] = [asdict(band) for band in self.eeg_bands]
        payload["eeg_reference_band_hz"] = list(self.eeg_reference_band_hz)
        return payload

    @property
    def spec_hash(self) -> str:
        """SHA-256 hash of the canonical feature specification."""
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


DEFAULT_FEATURE_SPEC = PhysiologicalPatchFeatureSpec()


@dataclass(frozen=True)
class FeatureDefinition:
    """Name, unit and audit description for one feature coordinate."""

    name: str
    unit: str
    family: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureExtractionManifest:
    """Serializable provenance for one feature-extraction batch."""

    schema_version: str
    feature_spec_hash: str
    feature_spec: Mapping[str, Any]
    modality: str
    input_layout: str
    input_unit: str
    sample_rate_hz: float
    patch_size_samples: int
    patch_duration_seconds: float
    batch_size: int
    patch_count: int
    channel_count: int
    channel_names: Tuple[str, ...]
    feature_definitions: Tuple[FeatureDefinition, ...]
    band_availability: Mapping[str, bool]
    notes: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe manifest suitable for export sidecars."""
        return {
            "schema_version": self.schema_version,
            "feature_spec_hash": self.feature_spec_hash,
            "feature_spec": dict(self.feature_spec),
            "modality": self.modality,
            "input_layout": self.input_layout,
            "input_unit": self.input_unit,
            "sample_rate_hz": float(self.sample_rate_hz),
            "patch_size_samples": int(self.patch_size_samples),
            "patch_duration_seconds": float(self.patch_duration_seconds),
            "batch_size": int(self.batch_size),
            "patch_count": int(self.patch_count),
            "channel_count": int(self.channel_count),
            "channel_names": list(self.channel_names),
            "feature_definitions": [
                definition.to_dict() for definition in self.feature_definitions
            ],
            "band_availability": {
                str(name): bool(available)
                for name, available in self.band_availability.items()
            },
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class PatchFeatureBatch:
    """Channel-level patch features and their validity/provenance metadata.

    ``values`` and ``feature_valid_mask`` have shape
    ``[batch, patch, channel, feature]``. ``channel_valid_mask`` and
    ``valid_sample_fraction`` have shape ``[batch, patch, channel]``.
    Invalid feature coordinates are represented by NaN and by a false entry in
    ``feature_valid_mask``.
    """

    values: np.ndarray
    feature_valid_mask: np.ndarray
    channel_valid_mask: np.ndarray
    valid_sample_fraction: np.ndarray
    feature_definitions: Tuple[FeatureDefinition, ...]
    channel_names: Tuple[str, ...]
    manifest: FeatureExtractionManifest

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return tuple(definition.name for definition in self.feature_definitions)

    @property
    def feature_units(self) -> Tuple[str, ...]:
        return tuple(definition.unit for definition in self.feature_definitions)

    @property
    def flattened_feature_names(self) -> Tuple[str, ...]:
        """Channel-qualified names matching :meth:`flatten_channels`."""
        return tuple(
            f"{channel}/{definition.name}"
            for channel in self.channel_names
            for definition in self.feature_definitions
        )

    @property
    def flattened_feature_units(self) -> Tuple[str, ...]:
        return self.feature_units * len(self.channel_names)

    def flatten_channels(self) -> np.ndarray:
        """Return ``[batch, patch, channel * feature]`` values."""
        return self.values.reshape(
            self.values.shape[0],
            self.values.shape[1],
            self.values.shape[2] * self.values.shape[3],
        )


def _eeg_feature_definitions(
    spec: PhysiologicalPatchFeatureSpec,
) -> Tuple[FeatureDefinition, ...]:
    definitions = [
        FeatureDefinition(
            "mean",
            CANONICAL_SIGNAL_UNIT,
            "time_domain",
            "Arithmetic mean of finite, valid samples.",
        ),
        FeatureDefinition(
            "std",
            CANONICAL_SIGNAL_UNIT,
            "time_domain",
            "Population standard deviation of finite, valid samples.",
        ),
        FeatureDefinition(
            "rms",
            CANONICAL_SIGNAL_UNIT,
            "time_domain",
            "Root mean square of finite, valid samples.",
        ),
        FeatureDefinition(
            "slope",
            f"{CANONICAL_SIGNAL_UNIT}_per_second",
            "time_domain",
            "Least-squares linear slope against time in seconds.",
        ),
        FeatureDefinition(
            "endpoint_delta",
            CANONICAL_SIGNAL_UNIT,
            "time_domain",
            "Last sample minus first sample.",
        ),
        FeatureDefinition(
            "line_length",
            CANONICAL_SIGNAL_UNIT,
            "time_domain",
            "Sum of absolute adjacent-sample differences.",
        ),
        FeatureDefinition(
            "hjorth_activity",
            f"{CANONICAL_SIGNAL_UNIT}_squared",
            "hjorth",
            "Population variance of the patch.",
        ),
        FeatureDefinition(
            "hjorth_mobility",
            "per_second",
            "hjorth",
            "Square root of first-derivative variance divided by signal variance.",
        ),
        FeatureDefinition(
            "hjorth_complexity",
            "dimensionless",
            "hjorth",
            "Derivative mobility divided by signal mobility.",
        ),
    ]
    for band in spec.eeg_bands:
        definitions.append(
            FeatureDefinition(
                f"log_absolute_power_{band.name}",
                f"log_{CANONICAL_SIGNAL_UNIT}_squared",
                "frequency_domain",
                (
                    "Natural log of one-sided Hann periodogram power integrated "
                    f"over [{band.low_hz:g}, {band.high_hz:g}) Hz."
                ),
            )
        )
    for band in spec.eeg_bands:
        definitions.append(
            FeatureDefinition(
                f"log_relative_power_{band.name}",
                "log_fraction",
                "frequency_domain",
                (
                    "Natural log of band power divided by power in the "
                    f"[{spec.eeg_reference_band_hz[0]:g}, "
                    f"{spec.eeg_reference_band_hz[1]:g}) Hz reference band."
                ),
            )
        )
    definitions.extend(
        (
            FeatureDefinition(
                "spectral_entropy",
                "dimensionless",
                "frequency_domain",
                "Normalized Shannon entropy of reference-band periodogram power.",
            ),
            FeatureDefinition(
                "peak_frequency",
                "hz",
                "frequency_domain",
                "Frequency of maximum reference-band periodogram power.",
            ),
        )
    )
    return tuple(definitions)


def _fnirs_feature_definitions() -> Tuple[FeatureDefinition, ...]:
    return (
        FeatureDefinition(
            "mean",
            CANONICAL_SIGNAL_UNIT,
            "time_domain",
            "Arithmetic mean of finite, valid samples.",
        ),
        FeatureDefinition(
            "median",
            CANONICAL_SIGNAL_UNIT,
            "time_domain",
            "Median of finite, valid samples.",
        ),
        FeatureDefinition(
            "std",
            CANONICAL_SIGNAL_UNIT,
            "time_domain",
            "Population standard deviation of finite, valid samples.",
        ),
        FeatureDefinition(
            "rms",
            CANONICAL_SIGNAL_UNIT,
            "time_domain",
            "Root mean square of finite, valid samples.",
        ),
        FeatureDefinition(
            "slope",
            f"{CANONICAL_SIGNAL_UNIT}_per_second",
            "time_domain",
            "Least-squares linear slope against time in seconds.",
        ),
        FeatureDefinition(
            "endpoint_delta",
            CANONICAL_SIGNAL_UNIT,
            "time_domain",
            "Last sample minus first sample.",
        ),
        FeatureDefinition(
            "auc",
            f"{CANONICAL_SIGNAL_UNIT}_second",
            "time_domain",
            "Trapezoidal integral over the patch duration.",
        ),
        FeatureDefinition(
            "derivative_spike",
            f"{CANONICAL_SIGNAL_UNIT}_per_second",
            "artifact",
            "Maximum absolute first derivative in the patch.",
        ),
    )


def _normalise_signal(
    signal: np.ndarray,
    *,
    patch_size: Optional[int],
) -> Tuple[np.ndarray, str]:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim == 4:
        if any(size <= 0 for size in values.shape):
            raise ValueError("signal dimensions must be non-empty")
        if patch_size is not None and int(patch_size) != values.shape[-1]:
            raise ValueError(
                f"patch_size={patch_size} does not match patch axis "
                f"length {values.shape[-1]}"
            )
        return values, "batch_patch_channel_sample"
    if values.ndim == 3:
        if patch_size is None:
            raise ValueError(
                "patch_size is required for [batch, channel, sample] signals"
            )
        patch_size = int(patch_size)
        if patch_size <= 0 or values.shape[-1] % patch_size:
            raise ValueError(
                "Continuous signal length must be divisible by positive patch_size"
            )
        if values.shape[0] <= 0 or values.shape[1] <= 0:
            raise ValueError("signal dimensions must be non-empty")
        patch_count = values.shape[-1] // patch_size
        patches = values.reshape(
            values.shape[0], values.shape[1], patch_count, patch_size
        ).transpose(0, 2, 1, 3)
        return patches, "batch_channel_sample"
    raise ValueError(
        "signal must have shape [batch, patch, channel, sample] or "
        "[batch, channel, sample]"
    )


def _coerce_bool_mask(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.bool_):
        return array
    return np.isfinite(array) & (array != 0)


def _normalise_valid_mask(
    valid_mask: Optional[np.ndarray],
    *,
    original_signal_shape: Tuple[int, ...],
    patches_shape: Tuple[int, int, int, int],
    input_layout: str,
) -> np.ndarray:
    if valid_mask is None:
        return np.ones(patches_shape, dtype=bool)

    mask = _coerce_bool_mask(np.asarray(valid_mask))
    batch_size, patch_count, channel_count, patch_size = patches_shape

    if input_layout == "batch_channel_sample" and mask.shape == original_signal_shape:
        mask = mask.reshape(
            batch_size, channel_count, patch_count, patch_size
        ).transpose(0, 2, 1, 3)
    elif mask.shape == patches_shape:
        pass
    elif mask.shape == (batch_size, patch_count, channel_count):
        mask = mask[..., None]
    elif mask.shape == (batch_size, patch_count):
        mask = mask[:, :, None, None]
    elif mask.shape == (batch_size, channel_count):
        mask = mask[:, None, :, None]
    else:
        raise ValueError(
            "valid_mask must match the input signal or have shape [B,T], "
            "[B,C], [B,T,C], or [B,T,C,P]"
        )
    return np.broadcast_to(mask, patches_shape).copy()


def _resolve_channel_names(
    channel_names: Optional[Sequence[str]],
    *,
    channel_count: int,
    modality: str,
) -> Tuple[str, ...]:
    if channel_names is None:
        return tuple(f"{modality}_{index:02d}" for index in range(channel_count))
    resolved = tuple(str(name) for name in channel_names)
    if len(resolved) != channel_count:
        raise ValueError(
            f"Expected {channel_count} channel names, received {len(resolved)}"
        )
    if any(not name for name in resolved):
        raise ValueError("Channel names must be non-empty")
    return resolved


def _masked_mean_and_variance(
    values: np.ndarray,
    valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = valid.sum(axis=-1)
    total = np.where(valid, values, 0.0).sum(axis=-1)
    mean = np.divide(
        total,
        count,
        out=np.full(total.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )
    centered = np.where(valid, values - mean[..., None], 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=-1),
        count,
        out=np.full(total.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )
    return mean, variance, count


def _common_time_features(
    patches: np.ndarray,
    sample_valid: np.ndarray,
    channel_valid: np.ndarray,
    *,
    sample_rate_hz: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    mean, variance, count = _masked_mean_and_variance(patches, sample_valid)
    squared_total = np.where(sample_valid, np.square(patches), 0.0).sum(axis=-1)
    rms = np.sqrt(
        np.divide(
            squared_total,
            count,
            out=np.full(mean.shape, np.nan, dtype=np.float64),
            where=count > 0,
        )
    )

    time = np.arange(patches.shape[-1], dtype=np.float64) / sample_rate_hz
    time_sum = np.where(sample_valid, time, 0.0).sum(axis=-1)
    time_mean = np.divide(
        time_sum,
        count,
        out=np.full(mean.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )
    centered_time = time - time_mean[..., None]
    slope_denominator = np.where(
        sample_valid, np.square(centered_time), 0.0
    ).sum(axis=-1)
    slope_numerator = np.where(
        sample_valid,
        centered_time * (patches - mean[..., None]),
        0.0,
    ).sum(axis=-1)
    slope = np.divide(
        slope_numerator,
        slope_denominator,
        out=np.full(mean.shape, np.nan, dtype=np.float64),
        where=slope_denominator > 0.0,
    )
    slope_valid = channel_valid & np.isfinite(slope)

    endpoint_valid = (
        channel_valid & sample_valid[..., 0] & sample_valid[..., -1]
    )
    endpoint = patches[..., -1] - patches[..., 0]

    adjacent_valid = sample_valid[..., :-1] & sample_valid[..., 1:]
    absolute_difference = np.abs(np.diff(patches, axis=-1))
    line_length = np.where(adjacent_valid, absolute_difference, 0.0).sum(axis=-1)
    line_valid = channel_valid & (adjacent_valid.sum(axis=-1) > 0)

    return {
        "mean": (mean, channel_valid),
        "std": (np.sqrt(np.maximum(variance, 0.0)), channel_valid),
        "rms": (rms, channel_valid),
        "slope": (slope, slope_valid),
        "endpoint_delta": (endpoint, endpoint_valid),
        "line_length": (line_length, line_valid),
    }


def _stack_features(
    features: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    definitions: Tuple[FeatureDefinition, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    values = []
    valid_masks = []
    for definition in definitions:
        if definition.name not in features:
            raise RuntimeError(f"Missing implementation for {definition.name}")
        feature_values, feature_valid = features[definition.name]
        feature_values = np.asarray(feature_values, dtype=np.float64)
        feature_valid = np.asarray(feature_valid, dtype=bool)
        feature_valid = feature_valid & np.isfinite(feature_values)
        values.append(np.where(feature_valid, feature_values, np.nan))
        valid_masks.append(feature_valid)
    return (
        np.stack(values, axis=-1).astype(np.float32),
        np.stack(valid_masks, axis=-1),
    )


def _extract_eeg(
    patches: np.ndarray,
    sample_valid: np.ndarray,
    channel_valid: np.ndarray,
    *,
    sample_rate_hz: float,
    spec: PhysiologicalPatchFeatureSpec,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Tuple[FeatureDefinition, ...],
    Dict[str, bool],
]:
    features = _common_time_features(
        patches,
        sample_valid,
        channel_valid,
        sample_rate_hz=sample_rate_hz,
    )
    activity = features["std"][0] ** 2

    first_difference = np.diff(patches, axis=-1) * sample_rate_hz
    first_valid = sample_valid[..., :-1] & sample_valid[..., 1:]
    _, first_variance, first_count = _masked_mean_and_variance(
        first_difference, first_valid
    )
    second_difference = np.diff(first_difference, axis=-1) * sample_rate_hz
    second_valid = (
        sample_valid[..., :-2]
        & sample_valid[..., 1:-1]
        & sample_valid[..., 2:]
    )
    _, second_variance, second_count = _masked_mean_and_variance(
        second_difference, second_valid
    )

    epsilon = float(spec.log_epsilon)
    mobility = np.sqrt(
        np.divide(
            first_variance,
            activity,
            out=np.full(activity.shape, np.nan, dtype=np.float64),
            where=activity > epsilon,
        )
    )
    derivative_mobility = np.sqrt(
        np.divide(
            second_variance,
            first_variance,
            out=np.full(activity.shape, np.nan, dtype=np.float64),
            where=first_variance > epsilon,
        )
    )
    complexity = np.divide(
        derivative_mobility,
        mobility,
        out=np.full(activity.shape, np.nan, dtype=np.float64),
        where=mobility > epsilon,
    )
    activity_valid = channel_valid & np.isfinite(activity)
    mobility_valid = (
        channel_valid
        & (first_count >= 2)
        & (activity > epsilon)
        & np.isfinite(mobility)
    )
    complexity_valid = (
        mobility_valid
        & (second_count >= 2)
        & (first_variance > epsilon)
        & np.isfinite(complexity)
    )
    features.update(
        {
            "hjorth_activity": (activity, activity_valid),
            "hjorth_mobility": (mobility, mobility_valid),
            "hjorth_complexity": (complexity, complexity_valid),
        }
    )

    patch_size = patches.shape[-1]
    frequencies = np.fft.rfftfreq(patch_size, d=1.0 / sample_rate_hz)
    window = np.hanning(patch_size).astype(np.float64)
    patch_mean = features["mean"][0]
    centered = np.where(
        sample_valid, patches - patch_mean[..., None], 0.0
    )
    windowed = centered * window
    spectrum = np.square(np.abs(np.fft.rfft(windowed, axis=-1)))
    window_energy = np.square(window) * sample_valid
    psd_scale = sample_rate_hz * window_energy.sum(axis=-1)
    psd = np.divide(
        spectrum,
        psd_scale[..., None],
        out=np.zeros_like(spectrum),
        where=psd_scale[..., None] > 0.0,
    )
    if patch_size % 2 == 0:
        psd[..., 1:-1] *= 2.0
    else:
        psd[..., 1:] *= 2.0
    frequency_step = sample_rate_hz / patch_size
    nyquist_hz = sample_rate_hz / 2.0

    band_availability: Dict[str, bool] = {}
    band_power: Dict[str, np.ndarray] = {}
    band_power_valid: Dict[str, np.ndarray] = {}
    for band in spec.eeg_bands:
        frequency_mask = (
            (frequencies >= band.low_hz) & (frequencies < band.high_hz)
        )
        available = bool(
            band.high_hz <= nyquist_hz + 1e-12 and frequency_mask.any()
        )
        band_availability[band.name] = available
        if available:
            power = psd[..., frequency_mask].sum(axis=-1) * frequency_step
            valid = channel_valid & (psd_scale > 0.0) & np.isfinite(power)
        else:
            power = np.full(channel_valid.shape, np.nan, dtype=np.float64)
            valid = np.zeros(channel_valid.shape, dtype=bool)
        band_power[band.name] = power
        band_power_valid[band.name] = valid
        features[f"log_absolute_power_{band.name}"] = (
            np.log(np.maximum(power, epsilon)),
            valid,
        )

    reference_low, reference_high = spec.eeg_reference_band_hz
    reference_mask = (
        (frequencies >= reference_low) & (frequencies < reference_high)
    )
    reference_available = bool(
        reference_high <= nyquist_hz + 1e-12
        and np.count_nonzero(reference_mask) >= 2
    )
    band_availability["reference_band"] = reference_available
    if reference_available:
        reference_psd = psd[..., reference_mask]
        reference_power = reference_psd.sum(axis=-1) * frequency_step
        reference_valid = (
            channel_valid
            & (psd_scale > 0.0)
            & (reference_power > epsilon)
            & np.isfinite(reference_power)
        )
    else:
        reference_psd = np.empty((*channel_valid.shape, 0), dtype=np.float64)
        reference_power = np.full(channel_valid.shape, np.nan, dtype=np.float64)
        reference_valid = np.zeros(channel_valid.shape, dtype=bool)

    for band in spec.eeg_bands:
        ratio = np.divide(
            band_power[band.name],
            reference_power,
            out=np.full(channel_valid.shape, np.nan, dtype=np.float64),
            where=reference_valid,
        )
        valid = (
            band_power_valid[band.name]
            & reference_valid
            & (ratio > 0.0)
            & np.isfinite(ratio)
        )
        features[f"log_relative_power_{band.name}"] = (
            np.log(np.maximum(ratio, epsilon)),
            valid,
        )

    if reference_available:
        probability = np.divide(
            reference_psd,
            reference_psd.sum(axis=-1, keepdims=True),
            out=np.zeros_like(reference_psd),
            where=reference_psd.sum(axis=-1, keepdims=True) > epsilon,
        )
        entropy_denominator = np.log(reference_psd.shape[-1])
        spectral_entropy = -np.sum(
            np.where(
                probability > 0.0,
                probability * np.log(np.maximum(probability, epsilon)),
                0.0,
            ),
            axis=-1,
        ) / entropy_denominator
        reference_frequencies = frequencies[reference_mask]
        peak_frequency = reference_frequencies[
            np.argmax(reference_psd, axis=-1)
        ]
        spectral_valid = reference_valid & np.isfinite(spectral_entropy)
        peak_valid = reference_valid & np.isfinite(peak_frequency)
    else:
        spectral_entropy = np.full(
            channel_valid.shape, np.nan, dtype=np.float64
        )
        peak_frequency = np.full(
            channel_valid.shape, np.nan, dtype=np.float64
        )
        spectral_valid = np.zeros(channel_valid.shape, dtype=bool)
        peak_valid = np.zeros(channel_valid.shape, dtype=bool)
    features["spectral_entropy"] = (spectral_entropy, spectral_valid)
    features["peak_frequency"] = (peak_frequency, peak_valid)

    definitions = _eeg_feature_definitions(spec)
    values, valid_mask = _stack_features(features, definitions)
    return values, valid_mask, definitions, band_availability


def _extract_fnirs(
    patches: np.ndarray,
    sample_valid: np.ndarray,
    channel_valid: np.ndarray,
    *,
    sample_rate_hz: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Tuple[FeatureDefinition, ...],
    Dict[str, bool],
]:
    features = _common_time_features(
        patches,
        sample_valid,
        channel_valid,
        sample_rate_hz=sample_rate_hz,
    )
    masked_values = np.where(sample_valid, patches, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(masked_values, axis=-1)
    features["median"] = (median, channel_valid)

    adjacent_valid = sample_valid[..., :-1] & sample_valid[..., 1:]
    complete_interval = channel_valid & np.all(adjacent_valid, axis=-1)
    time_step = 1.0 / sample_rate_hz
    trapezoids = 0.5 * (
        patches[..., :-1] + patches[..., 1:]
    ) * time_step
    auc = np.where(adjacent_valid, trapezoids, 0.0).sum(axis=-1)
    features["auc"] = (auc, complete_interval)

    absolute_derivative = np.abs(np.diff(patches, axis=-1)) * sample_rate_hz
    derivative_candidate = np.where(adjacent_valid, absolute_derivative, -np.inf)
    derivative_spike = np.max(derivative_candidate, axis=-1)
    derivative_valid = channel_valid & np.any(adjacent_valid, axis=-1)
    derivative_spike = np.where(derivative_valid, derivative_spike, np.nan)
    features["derivative_spike"] = (derivative_spike, derivative_valid)

    definitions = _fnirs_feature_definitions()
    values, valid_mask = _stack_features(features, definitions)
    # Short fNIRS token patches deliberately have no spectral-band features.
    return values, valid_mask, definitions, {}


def extract_physiological_patch_features(
    signal: np.ndarray,
    *,
    modality: Literal["eeg", "fnirs"],
    sample_rate_hz: float,
    patch_size: Optional[int] = None,
    valid_mask: Optional[np.ndarray] = None,
    channel_names: Optional[Sequence[str]] = None,
    spec: PhysiologicalPatchFeatureSpec = DEFAULT_FEATURE_SPEC,
) -> PatchFeatureBatch:
    """Extract versioned channel-level features aligned to tokenizer patches.

    Parameters
    ----------
    signal:
        Canonical robust-SD normalized signal with shape ``[B,T,C,P]`` or
        continuous legacy shape ``[B,C,total]``.
    modality:
        ``"eeg"`` or ``"fnirs"``. fNIRS intentionally receives only local
        time-domain morphology features; a 2-second patch does not support
        meaningful slow-band power estimates.
    sample_rate_hz:
        Sampling rate after tokenizer preprocessing.
    patch_size:
        Required only for continuous ``[B,C,total]`` input. If supplied for
        patch-major input, it must equal ``P``.
    valid_mask:
        Optional sample, channel-patch or token validity mask. Accepted shapes
        are the input signal shape, ``[B,T,C,P]``, ``[B,T,C]``, ``[B,C]`` or
        ``[B,T]``.
    channel_names:
        Optional channel identities. Generated stable names are used otherwise.
    spec:
        Immutable, hashable feature specification.
    """
    if modality not in {"eeg", "fnirs"}:
        raise ValueError("modality must be 'eeg' or 'fnirs'")
    if not np.isfinite(sample_rate_hz) or float(sample_rate_hz) <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive")
    sample_rate_hz = float(sample_rate_hz)
    original_signal = np.asarray(signal)
    patches, input_layout = _normalise_signal(signal, patch_size=patch_size)
    if patches.shape[-1] < 2:
        raise ValueError("Each patch must contain at least two samples")
    user_valid = _normalise_valid_mask(
        valid_mask,
        original_signal_shape=tuple(original_signal.shape),
        patches_shape=tuple(patches.shape),
        input_layout=input_layout,
    )
    sample_valid = user_valid & np.isfinite(patches)
    valid_sample_fraction = sample_valid.mean(axis=-1)
    minimum_samples = max(
        2,
        int(np.ceil(float(spec.minimum_valid_fraction) * patches.shape[-1])),
    )
    channel_valid = sample_valid.sum(axis=-1) >= minimum_samples
    resolved_channel_names = _resolve_channel_names(
        channel_names,
        channel_count=patches.shape[2],
        modality=modality,
    )

    if modality == "eeg":
        values, feature_valid, definitions, availability = _extract_eeg(
            patches,
            sample_valid,
            channel_valid,
            sample_rate_hz=sample_rate_hz,
            spec=spec,
        )
        notes = (
            "Amplitude-derived units refer to canonical robust-SD normalized "
            "signals, not physical voltage.",
            "PSD uses a constant-detrended, one-sided symmetric-Hann periodogram.",
            "Unavailable or invalid feature coordinates are NaN and have a "
            "false feature_valid_mask entry.",
        )
    else:
        values, feature_valid, definitions, availability = _extract_fnirs(
            patches,
            sample_valid,
            channel_valid,
            sample_rate_hz=sample_rate_hz,
        )
        notes = (
            "Amplitude-derived units refer to canonical robust-SD normalized "
            "signals, not physical concentration.",
            "Frequency-band power is intentionally excluded from local fNIRS "
            "patch features because short token patches do not resolve slow "
            "hemodynamic bands.",
            "Unavailable or invalid feature coordinates are NaN and have a "
            "false feature_valid_mask entry.",
        )

    manifest = FeatureExtractionManifest(
        schema_version=FEATURE_SPEC_SCHEMA_VERSION,
        feature_spec_hash=spec.spec_hash,
        feature_spec=spec.to_dict(),
        modality=modality,
        input_layout=input_layout,
        input_unit=spec.input_unit,
        sample_rate_hz=sample_rate_hz,
        patch_size_samples=patches.shape[-1],
        patch_duration_seconds=patches.shape[-1] / sample_rate_hz,
        batch_size=patches.shape[0],
        patch_count=patches.shape[1],
        channel_count=patches.shape[2],
        channel_names=resolved_channel_names,
        feature_definitions=definitions,
        band_availability=availability,
        notes=notes,
    )
    return PatchFeatureBatch(
        values=values,
        feature_valid_mask=feature_valid,
        channel_valid_mask=channel_valid,
        valid_sample_fraction=valid_sample_fraction.astype(np.float32),
        feature_definitions=definitions,
        channel_names=resolved_channel_names,
        manifest=manifest,
    )


def extract_eeg_patch_features(
    signal: np.ndarray,
    *,
    sample_rate_hz: float,
    patch_size: Optional[int] = None,
    valid_mask: Optional[np.ndarray] = None,
    channel_names: Optional[Sequence[str]] = None,
    spec: PhysiologicalPatchFeatureSpec = DEFAULT_FEATURE_SPEC,
) -> PatchFeatureBatch:
    """Convenience wrapper for :func:`extract_physiological_patch_features`."""
    return extract_physiological_patch_features(
        signal,
        modality="eeg",
        sample_rate_hz=sample_rate_hz,
        patch_size=patch_size,
        valid_mask=valid_mask,
        channel_names=channel_names,
        spec=spec,
    )


def extract_fnirs_patch_features(
    signal: np.ndarray,
    *,
    sample_rate_hz: float,
    patch_size: Optional[int] = None,
    valid_mask: Optional[np.ndarray] = None,
    channel_names: Optional[Sequence[str]] = None,
    spec: PhysiologicalPatchFeatureSpec = DEFAULT_FEATURE_SPEC,
) -> PatchFeatureBatch:
    """Convenience wrapper for :func:`extract_physiological_patch_features`."""
    return extract_physiological_patch_features(
        signal,
        modality="fnirs",
        sample_rate_hz=sample_rate_hz,
        patch_size=patch_size,
        valid_mask=valid_mask,
        channel_names=channel_names,
        spec=spec,
    )


__all__ = [
    "CANONICAL_SIGNAL_UNIT",
    "DEFAULT_FEATURE_SPEC",
    "FEATURE_SPEC_SCHEMA_VERSION",
    "FeatureDefinition",
    "FeatureExtractionManifest",
    "FrequencyBand",
    "PatchFeatureBatch",
    "PhysiologicalPatchFeatureSpec",
    "extract_eeg_patch_features",
    "extract_fnirs_patch_features",
    "extract_physiological_patch_features",
]
