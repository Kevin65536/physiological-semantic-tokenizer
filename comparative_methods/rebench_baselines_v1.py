"""Fixed masked signal features for DOWNSTREAM-REBENCH-v1 linear readouts.

No targets, fitted statistics, interpolation, or measured-data access belong in
this module. The worker fits input scaling and the linear feature scaler inside
the appropriate training partition. These are simple references, not a task's
performance ceiling.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
CLASSIFICATION_TASKS = frozenset({"motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual"})
OUTPUT_KIND = "features"
DETERMINISTIC = True
SEARCH_SOURCE = "DOWNSTREAM-REBENCH-v1 fixed signal features and six-value linear penalty grid"


def _masked_segments(values: torch.Tensor, mask: torch.Tensor | None, *, channels: int,
                     samples: int, segments: int, label: str) -> tuple[torch.Tensor, torch.Tensor]:
    if (not torch.is_tensor(values) or values.ndim != 3 or values.shape[0] < 1
            or values.shape[1:] != (channels, samples) or not values.is_floating_point()):
        raise ValueError(f"{label} must be floating-point [B,{channels},{samples}] with B >= 1")
    if mask is None:
        mask = torch.ones_like(values, dtype=torch.bool)
    elif not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.shape != values.shape:
        raise ValueError(f"{label}_mask must be bool with the same [B,C,T] shape as {label}")
    if mask.device != values.device:
        raise ValueError(f"{label} and its input mask must be on the same device")
    if not bool(torch.isfinite(values[mask]).all()):
        raise ValueError(f"valid {label} samples must be finite")
    # Invalid NaNs or arbitrary padding values cannot enter any moment or FFT.
    cleaned = torch.where(mask, values, 0.0)
    length = samples // segments
    return (cleaned.reshape(values.shape[0], channels, segments, length).transpose(1, 2),
            mask.reshape(values.shape[0], channels, segments, length).transpose(1, 2))


def _eeg_features(values: torch.Tensor, mask: torch.Tensor, rate: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked rectangular-window periodogram, integrated over disjoint bands."""
    count = mask.sum(-1)
    mean = values.sum(-1) / count.clamp_min(1)
    centered = torch.where(mask, values - mean.unsqueeze(-1), 0.0)
    length = values.shape[-1]
    spectrum = torch.fft.rfft(centered, dim=-1).abs().square()
    one_sided = torch.full((spectrum.shape[-1],), 2.0, device=values.device, dtype=values.dtype)
    one_sided[0] = 1.0
    if length % 2 == 0:
        one_sided[-1] = 1.0
    # PSD = w*|FFT|^2/(fs*n_valid); summing PSD*df gives this
    # denominator N*n_valid. Complete-input total power equals population var.
    power = spectrum * one_sided / (length * count.clamp_min(1).unsqueeze(-1))
    frequencies = torch.fft.rfftfreq(length, d=1.0 / rate, device=values.device)
    valid = count >= 2
    bands = []
    for index, (low, high) in enumerate(BANDS):
        selected = (frequencies >= low) & ((frequencies <= high) if index == len(BANDS) - 1 else (frequencies < high))
        band = torch.log1p(power[..., selected].sum(-1))
        bands.append(torch.where(valid, band, 0.0))
    features = torch.stack([*bands, count.to(values.dtype) / length], dim=-1)
    support = torch.stack([*[valid] * len(BANDS), torch.ones_like(valid)], dim=-1)
    return features.flatten(-2), support.flatten(-2)


def _fnirs_features(values: torch.Tensor, mask: torch.Tensor, rate: float) -> tuple[torch.Tensor, torch.Tensor]:
    count = mask.sum(-1)
    mean = values.sum(-1) / count.clamp_min(1)
    centered = torch.where(mask, values - mean.unsqueeze(-1), 0.0)
    sd = (centered.square().sum(-1) / count.clamp_min(1)).sqrt()
    times = torch.arange(values.shape[-1], device=values.device, dtype=values.dtype) / rate
    mean_time = (mask * times).sum(-1) / count.clamp_min(1)
    time_delta = torch.where(mask, times - mean_time.unsqueeze(-1), 0.0)
    denominator = time_delta.square().sum(-1)
    slope_valid = (count >= 2) & (denominator > 0)
    slope = (centered * time_delta).sum(-1) / denominator.clamp_min(torch.finfo(values.dtype).tiny)
    observed = count >= 1
    features = torch.stack((torch.where(observed, mean, 0.0),
                            torch.where(observed, sd, 0.0),
                            torch.where(slope_valid, slope, 0.0),
                            count.to(values.dtype) / values.shape[-1]), dim=-1)
    support = torch.stack((observed, observed, slope_valid, torch.ones_like(observed)), dim=-1)
    return features.flatten(-2), support.flatten(-2)


class SignalFeatures(nn.Module):
    supports_mask = True
    supports_temporal_output = True
    deterministic = True

    def __init__(self, task: str, profile: dict[str, Any]) -> None:
        super().__init__()
        if task not in CLASSIFICATION_TASKS | {"refed_regression"}:
            raise ValueError(f"unknown signal-baseline task: {task!r}")
        self.task = task
        self.baseline_modality = str(profile.get("baseline_modality", "eeg"))
        if self.baseline_modality not in {"eeg", "fnirs", "multimodal"}:
            raise ValueError("baseline_modality must be eeg, fnirs, or multimodal")
        self.use_eeg = self.baseline_modality in {"eeg", "multimodal"}
        self.use_fnirs = self.baseline_modality in {"fnirs", "multimodal"}
        self.supports_fnirs = self.use_fnirs
        self.sequence_features = task == "refed_regression"
        self.segments = 20 if self.sequence_features else 1
        duration = 20 if self.sequence_features else (2 if task == "dsr" else 8)
        self.input_samples = duration * 200
        self.fnirs_samples = duration * 10
        self.eeg_channels = len(profile.get("channel_names", ()))
        self.fnirs_channels = len(profile.get("fnirs_channel_names", ()))
        for used, label, channels, rate_key, rate, sample_key, expected in (
            (self.use_eeg, "eeg", self.eeg_channels, "sampling_rate_hz", 200, "input_samples", self.input_samples),
            (self.use_fnirs, "fnirs", self.fnirs_channels, "fnirs_sampling_rate_hz", 10, "fnirs_samples", self.fnirs_samples),
        ):
            if used:
                if channels < 1:
                    raise ValueError(f"{label} baseline requires a nonempty measured channel panel")
                names = profile["channel_names" if label == "eeg" else "fnirs_channel_names"]
                if len(set(names)) != channels:
                    raise ValueError(f"{label} channel identities must be unique")
                if float(profile.get(rate_key, 0)) != rate or profile.get(sample_key, expected) != expected:
                    raise ValueError(f"{label} baseline requires {rate} Hz and {expected} samples")
        self.feature_dim = (6 * self.eeg_channels if self.use_eeg else 0) + (4 * self.fnirs_channels if self.use_fnirs else 0)
        self.method_identity = f"fixed_masked_signal_features_{self.baseline_modality}_linear_rebench_v1"
        feature_names = []
        if self.use_eeg:
            feature_names.extend(f"eeg:{name}:{stat}" for name in profile["channel_names"]
                                 for stat in ("log1p_power_1_4Hz", "log1p_power_4_8Hz", "log1p_power_8_13Hz", "log1p_power_13_30Hz", "log1p_power_30_45Hz", "input_coverage"))
        if self.use_fnirs:
            feature_names.extend(f"fnirs:{name}:{stat}" for name in profile["fnirs_channel_names"]
                                 for stat in ("mean", "population_sd", "slope_per_s", "input_coverage"))
        self.metadata = {
            "schema": "downstream_rebench_signal_features_v1",
            "method_identity": self.method_identity,
            "task": task,
            "baseline_modality": self.baseline_modality,
            "feature_dim": self.feature_dim,
            "feature_names": feature_names,
            "sequence_features": self.sequence_features,
            "supports_mask": True,
            "deterministic": True,
            "trainable_parameter_count": 0,
            "input_preprocessing": "caller_train_partition_scaling; feature_standardization_fit_training_only",
            "eeg": {
                "source": "rectangular_window_one_sided_rfft_periodogram",
                "bands_hz": [list(band) for band in BANDS],
                "band_boundaries": "low_inclusive_high_exclusive_except_45Hz_inclusive",
                "formula": "log1p(sum_band(w_k*abs(rfft(m*(x-mean_valid)))^2)/(N*n_valid))",
                "one_sided_weight": "1 at DC/Nyquist; 2 at other positive frequencies",
                "segment_seconds": 1 if self.sequence_features else duration,
                "frequency_spacing_hz": 1.0 if self.sequence_features else 1.0 / duration,
                "partial_support": "zero_weight_invalid_samples; normalize_by_observed_count; no_interpolation; fewer_than_2_valid_samples_fixed_zero_and_feature_mask_false",
                "limitation": "1s_REFED_has_1Hz_nominal_bins; shorter_or_gapped_support_has_mask_window_bias_and_poorer_effective_resolution",
            },
            "fnirs": {
                "source": "valid_native_sample_moments_and_OLS_time_slope",
                "formula": "mean=sum_valid(x)/n; sd=sqrt(sum_valid((x-mean)^2)/n); slope=sum_valid((t-mean_t)*(x-mean))/sum_valid((t-mean_t)^2)",
                "time_units": "seconds",
                "segment_seconds": 1 if self.sequence_features else duration,
                "partial_support": "mean_sd_require_1_sample; slope_requires_2_distinct_times; unavailable_features_fixed_zero_and_feature_mask_false",
            },
            "coverage": "per_channel_n_valid/N; observed_even_when_zero; derived_only_from_selected_input_masks",
            "sequence_feature_mask": "any_recorded_selected_modality_input_in_bin; independent_of_targets",
            "interpretation": "simple_signal_reference_not_performance_ceiling",
        }

    def _extract(self, eeg: torch.Tensor, fnirs: torch.Tensor | None = None, *,
                 eeg_mask: torch.Tensor | None = None, fnirs_mask: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, supports, recorded = [], [], []
        for used, label, values, mask, channels, samples, rate, transform in (
            (self.use_eeg, "eeg", eeg, eeg_mask, self.eeg_channels, self.input_samples, 200.0, _eeg_features),
            (self.use_fnirs, "fnirs", fnirs, fnirs_mask, self.fnirs_channels, self.fnirs_samples, 10.0, _fnirs_features),
        ):
            if not used:
                continue
            values, mask = _masked_segments(values, mask, channels=channels, samples=samples, segments=self.segments, label=label)
            transformed, supported = transform(values, mask, rate)
            features.append(transformed)
            supports.append(supported)
            recorded.append(mask.any(-1).any(-1))
        if len(features) == 2 and (features[0].shape[:2] != features[1].shape[:2] or features[0].device != features[1].device):
            raise ValueError("selected EEG and fNIRS must share batch size and device")
        output, supported = torch.cat(features, -1), torch.cat(supports, -1)
        present = torch.stack(recorded).any(0)
        if not self.sequence_features:
            output, supported, present = output[:, 0], supported[:, 0], present[:, 0]
        return output, supported, present

    def forward(self, eeg: torch.Tensor, fnirs: torch.Tensor | None = None, *,
                eeg_mask: torch.Tensor | None = None, fnirs_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self._extract(eeg, fnirs, eeg_mask=eeg_mask, fnirs_mask=fnirs_mask)[0]

    def feature_mask(self, eeg: torch.Tensor, fnirs: torch.Tensor | None = None, *,
                     eeg_mask: torch.Tensor | None = None, fnirs_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self._extract(eeg, fnirs, eeg_mask=eeg_mask, fnirs_mask=fnirs_mask)[1]

    def sequence_feature_mask(self, eeg: torch.Tensor, fnirs: torch.Tensor | None = None, *,
                              eeg_mask: torch.Tensor | None = None, fnirs_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self._extract(eeg, fnirs, eeg_mask=eeg_mask, fnirs_mask=fnirs_mask)[2]


def create_model(task: str, profile: dict, device: str, trainable: bool = False) -> nn.Module:
    # A smoke worker may request trainable=True to exercise its downstream head;
    # these fixed features themselves always have zero trainable parameters.
    return SignalFeatures(task, profile).to(device)
