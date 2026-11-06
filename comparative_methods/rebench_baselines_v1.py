"""Fixed masked signal features for DOWNSTREAM-REBENCH-v1 linear readouts.

No targets, fitted statistics, interpolation, or measured-data access belong in
this module. The worker fits input scaling and the linear feature scaler inside
the appropriate training partition. These are simple references, not a task's
performance ceiling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from comparative_methods.rebench_eeg_common_v1 import encoding_duration


BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
CLASSIFICATION_TASKS = frozenset({"motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual"})
OUTPUT_KIND = "features"
DETERMINISTIC = True
SEARCH_SOURCE = "DOWNSTREAM-REBENCH-v1 fixed signal features and six-value linear penalty grid"


def fnirs_block_covariances(values, mask, components):
    """HbO/HbR Ledoit-Wolf covariance blocks with sample-local scalar scaling.

    Real channels keep their identities. Missing values have zero weight; the
    shrinkage and a small diagonal floor make missing-channel blocks positive
    definite without inventing measurements. No targets or population fit enter
    this transform. The Riemannian reference must subsequently fit training only.
    """
    from sklearn.covariance import ledoit_wolf
    x, support = np.asarray(values, np.float64), np.asarray(mask)
    if x.ndim != 3 or support.shape != x.shape or support.dtype != bool:
        raise ValueError("Hb covariance requires [N,C,T] and a matching bool mask")
    if len(components) != x.shape[1] or not np.isfinite(x[support]).all():
        raise ValueError("Hb covariance channel identity or finite support mismatch")
    groups = [np.flatnonzero(np.asarray(components) == kind) for kind in ("hbo", "hbr")]
    if not len(groups[0]) or len(groups[0]) != len(groups[1]) or sum(map(len, groups)) != x.shape[1]:
        raise ValueError("paired HbO/HbR blocks must cover the same number of real locations")
    output = np.empty((len(x), 2, len(groups[0]), len(groups[0])), np.float64)
    coverage = np.empty((len(x), x.shape[1]), np.float64)
    for row in range(len(x)):
        for block, channels in enumerate(groups):
            valid = support[row, channels]
            count = valid.sum(-1, keepdims=True)
            clean = np.where(valid, x[row, channels], 0.)
            mean = clean.sum(-1, keepdims=True) / count.clip(1)
            centered = np.where(valid, clean - mean, 0.)
            rms = np.sqrt(np.square(centered).sum() / max(int(valid.sum()), 1))
            normalized = centered / rms if rms > 0 else centered
            covariance, _ = ledoit_wolf(normalized.T, assume_centered=True)
            floor = 1e-6 * max(float(np.trace(covariance) / len(channels)), 1.)
            output[row, block] = covariance + np.eye(len(channels)) * floor
            coverage[row, channels] = valid.mean(-1)
    return output, coverage


def fnirs_tangent_features(covariances, train):
    """Fit two Riemannian references only on the training covariance blocks."""
    from pyriemann.tangentspace import TangentSpace
    covariances = np.asarray(covariances, np.float64)
    if covariances.ndim != 4 or covariances.shape[1] != 2:
        raise ValueError("expected two Hb covariance blocks")
    columns, references = [], []
    for block in range(2):
        mapper = TangentSpace(metric="riemann", tsupdate=False).fit(covariances[train, block])
        columns.append(mapper.transform(covariances[:, block]))
        references.append(mapper.reference_)
    return np.concatenate(columns, axis=1), np.stack(references)


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
        duration = int(encoding_duration(task, profile))
        self.segments = duration if self.sequence_features else 1
        if not self.sequence_features and profile.get("classification_segments", 1) != 1:
            if profile.get("input_window_schema") != "downstream_input_v1":
                raise ValueError("segmented classification features require the frozen downstream input")
            self.segments = int(profile["classification_segments"])
            if self.segments < 1 or duration % self.segments:
                raise ValueError("classification segments must divide the physical window")
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
        if not self.sequence_features and self.segments > 1:
            self.feature_dim *= self.segments
            feature_names = [f"segment{segment}:{name}" for segment in range(self.segments) for name in feature_names]
        self.metadata = {
            "schema": "downstream_rebench_signal_features_v1",
            "method_identity": self.method_identity,
            "task": task,
            "baseline_modality": self.baseline_modality,
            "feature_dim": self.feature_dim,
            "feature_names": feature_names,
            "sequence_features": self.sequence_features,
            "segment_count": self.segments,
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
                "segment_seconds": duration / self.segments,
                "frequency_spacing_hz": self.segments / duration,
                "partial_support": "zero_weight_invalid_samples; normalize_by_observed_count; no_interpolation; fewer_than_2_valid_samples_fixed_zero_and_feature_mask_false",
                "limitation": "1s_REFED_has_1Hz_nominal_bins; shorter_or_gapped_support_has_mask_window_bias_and_poorer_effective_resolution",
            },
            "fnirs": {
                "source": "valid_native_sample_moments_and_OLS_time_slope",
                "formula": "mean=sum_valid(x)/n; sd=sqrt(sum_valid((x-mean)^2)/n); slope=sum_valid((t-mean_t)*(x-mean))/sum_valid((t-mean_t)^2)",
                "time_units": "seconds",
                "segment_seconds": duration / self.segments,
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
            output, supported, present = output.flatten(1), supported.flatten(1), present.any(1)
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


def repair_probe_features(eeg, fnirs, eeg_mask, fnirs_mask, *, task, version=1):
    """Label-free, fixed physical-time features for the versioned repair probe.

    The caller chooses the actual channel panel. Arrays are [batch,channel,time]
    at 200/10 Hz over 30 seconds. Unsupported values never enter an FFT, moment,
    covariance or baseline. Population scaling belongs to each training fold.
    """
    from scipy.signal import butter, sosfiltfilt

    if task not in {"visual", "refed_regression"} or version not in (1, 2):
        raise ValueError("repair features require the declared Visual/REFED task")
    eeg, fnirs = np.asarray(eeg, float), np.asarray(fnirs, float)
    eeg_mask, fnirs_mask = np.asarray(eeg_mask, bool), np.asarray(fnirs_mask, bool)
    for x, mask, count in ((eeg, eeg_mask, 6000), (fnirs, fnirs_mask, 300)):
        if x.ndim != 3 or x.shape != mask.shape or x.shape[-1] != count or not np.isfinite(x[mask]).all():
            raise ValueError("finite recorded 30-second input and identical masks required")
    eeg, fnirs = np.where(eeg_mask, eeg, 0.), np.where(fnirs_mask, fnirs, 0.)
    batch, channels, _ = eeg.shape
    if version == 2:
        # A declared reference change on the same measured electrode panel.
        # Partial tails cannot introduce padded channels into the reference.
        mean = eeg.sum(1, keepdims=True) / np.maximum(eeg_mask.sum(1, keepdims=True), 1)
        eeg = np.where(eeg_mask, eeg - mean, 0.)

    def moments(x, mask, bins, rate):
        length = x.shape[-1] // bins
        x = x.reshape(len(x), x.shape[1], bins, length).transpose(0, 2, 1, 3)
        mask = mask.reshape(x.shape[0], x.shape[2], bins, length).transpose(0, 2, 1, 3)
        count = mask.sum(-1); mean = x.sum(-1) / np.maximum(count, 1)
        time = np.arange(length) / rate
        center = np.where(mask, time - (mask * time).sum(-1)[..., None] / np.maximum(count, 1)[..., None], 0.)
        denominator = (center * center).sum(-1)
        slope = (center * (x - mean[..., None])).sum(-1) / np.maximum(denominator, np.finfo(float).tiny)
        return mean, slope, count, denominator

    segments = eeg.reshape(batch, channels, 30, 200).transpose(0, 2, 1, 3)
    valid = eeg_mask.reshape(batch, channels, 30, 200).transpose(0, 2, 1, 3)
    count = valid.sum(-1)
    centered = np.where(valid, segments - (segments.sum(-1) / np.maximum(count, 1))[..., None], 0.)
    taper = np.hanning(200) * valid
    spectrum = np.abs(np.fft.rfft(centered * taper, axis=-1)) ** 2
    weights = np.full(101, 2.); weights[[0, -1]] = 1.
    power = spectrum * weights / np.maximum(200 * (taper * taper).sum(-1)[..., None], np.finfo(float).tiny)
    total = power.sum(-1)
    floor = np.maximum(total * 1e-12, np.finfo(float).tiny)
    frequencies = np.fft.rfftfreq(200, 1 / 200)
    bands = []
    for index, (lo, hi) in enumerate(BANDS):
        chosen = (frequencies >= lo) & ((frequencies <= hi) if index == len(BANDS) - 1 else (frequencies < hi))
        value = np.log(np.maximum(power[..., chosen].sum(-1), floor))
        bands.append(np.where((count >= 2) & (total > 0), value, 0.))
    spectral = np.stack(bands, -1).reshape(batch, 30, -1)
    spectral_mask = np.repeat(count >= 2, len(BANDS), axis=-1)
    hb_mean, hb_slope, hb_count, hb_denominator = moments(fnirs, fnirs_mask, 30, 10.)
    hb = np.concatenate((hb_mean, hb_slope), -1)
    hb_mask = np.concatenate((hb_count >= 1, (hb_count >= 2) & (hb_denominator > 0)), -1)
    output = {}

    def add(name, blocks):
        values = np.concatenate([np.where(mask, value, 0.) for value, mask in blocks], -1)
        masks = np.concatenate([mask for _, mask in blocks], -1)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite repair feature")
        output[name] = (values.astype(np.float32), masks, [value.shape[-1] for value, _ in blocks])

    if task == "refed_regression":
        add("eeg", [(spectral, spectral_mask)])
        add("fnirs", [(hb, hb_mask)])
        add("multimodal", [(spectral, spectral_mask), (hb, hb_mask)])
        context = []
        for values, masks, offsets in ((spectral, spectral_mask, (-2, 0, 2)), (hb, hb_mask, (0, 2, 4, 6))):
            for offset in offsets:
                x = np.zeros_like(values); m = np.zeros_like(masks)
                destination = slice(max(0, -offset), min(30, 30 - offset))
                source = slice(max(0, offset), min(30, 30 + offset))
                x[:, destination], m[:, destination] = values[:, source], masks[:, source]
                context.append((x, m))
        add("context", context)
    else:
        if not eeg_mask.all():
            raise ValueError("Visual covariance/ERP requires the common complete EEG support")
        baseline = eeg[..., :1000].mean(-1, keepdims=True)
        early = (eeg[..., 1000:1600] - baseline).reshape(batch, channels, 30, 20).mean(-1)
        decision = (eeg[..., 1600:3400] - baseline).reshape(batch, channels, 30, 60).mean(-1)
        erp = np.concatenate((early, decision), -1).reshape(batch, -1)
        add("event", [(erp, np.ones_like(erp, bool))])
        absolute = spectral[:, 5:17].reshape(batch, 4, 3, -1).mean(2)
        relative = absolute - spectral[:, :5].mean(1)[:, None]
        band = np.concatenate((absolute.reshape(batch, -1), relative.reshape(batch, -1)), -1)
        add("band", [(band, np.ones_like(band, bool))])
        mean, slope, observed, denominator = moments(fnirs, fnirs_mask, 6, 10.)
        base_support = observed[:, :1] > 0
        relative_hb = mean - mean[:, :1]
        hf = np.concatenate((relative_hb.reshape(batch, -1), slope.reshape(batch, -1)), -1)
        hm = np.concatenate((((observed > 0) & base_support).reshape(batch, -1),
                             ((observed >= 2) & (denominator > 0)).reshape(batch, -1)), -1)
        add("fnirs", [(hf, hm)])
        add("multimodal", [(band, np.ones_like(band, bool)), (hf, hm)])
        covariance = []
        upper = np.triu_indices(channels)
        for low, high in ((4, 8), (8, 13), (13, 30)):
            filtered = sosfiltfilt(butter(4, (low, high), btype="bandpass", fs=200, output="sos"), eeg, axis=-1)
            for start, stop in ((1000, 1600), (1600, 3400)):
                segment = filtered[..., start:stop]
                segment = segment - segment.mean(-1, keepdims=True)
                cov = segment @ segment.swapaxes(-1, -2) / segment.shape[-1]
                scale = np.trace(cov, axis1=-2, axis2=-1) / channels
                normalized = cov / np.maximum(scale, np.finfo(float).tiny)[:, None, None]
                normalized = .9 * normalized + .1 * np.eye(channels)
                normalized[scale == 0] = np.eye(channels)
                eigenvalues, vectors = np.linalg.eigh(normalized)
                logarithm = (vectors * np.log(eigenvalues)[:, None, :]) @ vectors.swapaxes(-1, -2)
                features = logarithm[:, upper[0], upper[1]]
                features[:, upper[0] != upper[1]] *= np.sqrt(2.)
                covariance.append(features)
        cf = np.concatenate(covariance, -1)
        add("covariance", [(cf, np.ones_like(cf, bool))])
    if version == 2:
        count = spectral_mask.sum(1, keepdims=True)
        center = (spectral * spectral_mask).sum(1, keepdims=True) / np.maximum(count, 1)
        relative = np.where(spectral_mask, spectral - center, 0.)
        # Hb level and slope share a per-window channel amplitude scale. This
        # uses no target or other subject, and preserves chromophore identities.
        hc = fnirs_mask.sum(-1, keepdims=True)
        hm = fnirs.sum(-1, keepdims=True) / np.maximum(hc, 1)
        hs = np.sqrt(np.where(fnirs_mask, (fnirs - hm) ** 2, 0.).sum(-1, keepdims=True) / np.maximum(hc, 1))
        hs = np.where(hs > 0, hs, 1.)
        local_hb = np.concatenate(((hb_mean - hm.transpose(0, 2, 1)) / hs.transpose(0, 2, 1),
                                   hb_slope / hs.transpose(0, 2, 1)), -1)
        if task == "refed_regression":
            add("relative", [(relative, spectral_mask), (local_hb, hb_mask)])
            for name in ("multimodal", "relative"):
                values, masks, lengths = output[name]
                # Ten ordered three-second bins retain the full input window.
                values = values.reshape(batch, 10, 3, -1)
                masks = masks.reshape(batch, 10, 3, -1)
                counts = masks.sum(2)
                pooled = (values * masks).sum(2) / np.maximum(counts, 1)
                pieces = []; offset = 0
                for length in lengths:
                    pieces.append((pooled[..., offset:offset + length].reshape(batch, -1),
                                   (counts[..., offset:offset + length] > 0).reshape(batch, -1)))
                    offset += length
                add(name + "_trajectory", pieces)
        else:
            channel_sd = np.sqrt(np.mean(eeg ** 2, axis=-1, keepdims=True))
            normalized = eeg / np.maximum(channel_sd, np.finfo(float).tiny)
            baseline = normalized[..., :1000].mean(-1, keepdims=True)
            early = (normalized[..., 1000:1600] - baseline).reshape(batch, channels, 30, 20).mean(-1)
            decision = (normalized[..., 1600:3400] - baseline).reshape(batch, channels, 30, 60).mean(-1)
            erp = np.concatenate((early, decision), -1).reshape(batch, -1)
            add("normalized_event", [(erp, np.ones_like(erp, bool))])
            encoding = early.reshape(batch, -1)
            add("normalized_encoding", [(encoding, np.ones_like(encoding, bool))])
            relative = relative[:, 5:17].reshape(batch, 4, 3, -1).mean(2).reshape(batch, -1)
            add("relative_band", [(relative, np.ones_like(relative, bool))])
            whitened = []
            for low, high in ((4, 8), (8, 13), (13, 30)):
                filtered = sosfiltfilt(butter(4, (low, high), btype="bandpass", fs=200, output="sos"), eeg, axis=-1)
                def covariance_at(left, right):
                    segment = filtered[..., left:right]
                    segment = segment - segment.mean(-1, keepdims=True)
                    covariance = segment @ segment.swapaxes(-1, -2) / segment.shape[-1]
                    scale = np.trace(covariance, axis1=-2, axis2=-1) / channels
                    return .9 * covariance + .1 * np.maximum(scale, np.finfo(float).tiny)[:, None, None] * np.eye(channels)
                baseline_covariance = covariance_at(0, 1000)
                eigenvalues, vectors = np.linalg.eigh(baseline_covariance)
                inverse_root = (vectors * (1 / np.sqrt(eigenvalues))[:, None, :]) @ vectors.swapaxes(-1, -2)
                for left, right in ((1000, 1600), (1600, 3400)):
                    covariance = inverse_root @ covariance_at(left, right) @ inverse_root
                    eigenvalues, vectors = np.linalg.eigh(covariance)
                    logarithm = (vectors * np.log(np.maximum(eigenvalues, np.finfo(float).tiny))[:, None, :]) @ vectors.swapaxes(-1, -2)
                    values = logarithm[:, upper[0], upper[1]]
                    values[:, upper[0] != upper[1]] *= np.sqrt(2.)
                    whitened.append(values)
            values = np.concatenate(whitened, -1)
            add("baseline_covariance", [(values, np.ones_like(values, bool))])
    return output


def repair_hb_probe_features(fnirs, mask, *, task):
    """Fixed HbT features for the versioned native/processed input audit.

    The caller validates real HbO-then-HbR pairs, clock and source support.
    Native and retained-cache views use this identical feature calculation.
    """
    fnirs, mask = np.asarray(fnirs, float), np.asarray(mask, bool)
    if (fnirs.ndim != 3 or fnirs.shape != mask.shape or fnirs.shape[-1] != 300
            or fnirs.shape[1] % 2 or not np.isfinite(fnirs[mask]).all()
            or task not in {"visual", "refed_regression"}):
        raise ValueError("paired finite 30-second Hb observations and masks required")
    channels = fnirs.shape[1] // 2
    support = mask[:, :channels] & mask[:, channels:]
    values = np.where(support, fnirs[:, :channels] + fnirs[:, channels:], 0.)
    bins = 30 if task == "refed_regression" else 6
    x = values.reshape(len(values), channels, bins, 300 // bins).transpose(0, 2, 1, 3)
    m = support.reshape(len(values), channels, bins, 300 // bins).transpose(0, 2, 1, 3)
    count = m.sum(-1); mean = x.sum(-1) / np.maximum(count, 1)
    sd = np.sqrt(np.where(m, (x - mean[..., None]) ** 2, 0.).sum(-1) / np.maximum(count, 1))
    time = np.arange(x.shape[-1]) / 10.
    centered_time = np.where(m, time - (time * m).sum(-1)[..., None] / np.maximum(count, 1)[..., None], 0.)
    denominator = (centered_time ** 2).sum(-1)
    slope = (centered_time * x).sum(-1) / np.maximum(denominator, np.finfo(float).tiny)
    maximum = np.where(m, x, -np.inf).max(-1); minimum = np.where(m, x, np.inf).min(-1)
    entropy = .5 * np.log(2 * np.pi * np.e * np.maximum(sd ** 2, np.finfo(float).tiny))
    features = np.stack((mean, maximum, minimum, sd, slope, entropy), -1)
    masks = np.stack((count > 0, count > 0, count > 0, count > 1, denominator > 0, (count > 1) & (sd > 0)), -1)
    if task == "visual":
        # The first five seconds are the declared input reference, not rest.
        features[..., :3] -= mean[:, :1, :, None]
        masks[..., :3] &= (count[:, :1, :, None] > 0)
        features, masks = features.reshape(len(values), -1), masks.reshape(len(values), -1)
    else:
        features, masks = features.reshape(len(values), 30, -1), masks.reshape(len(values), 30, -1)
    return np.where(masks, features, 0.).astype(np.float32), masks, [features.shape[-1]]


def repair_voltage_probe_features(eeg, mask, *, task):
    """Same voltage features for cached EEG and two native high-pass controls."""
    eeg, mask = np.asarray(eeg, float), np.asarray(mask, bool)
    if eeg.ndim != 3 or eeg.shape != mask.shape or eeg.shape[-1] != 6000 or not np.isfinite(eeg[mask]).all():
        raise ValueError("finite masked 30-second EEG at200Hz required")
    if task == "visual":
        if not mask.all():
            raise ValueError("Visual ERP requires the retained complete EEG support")
        centered = eeg - eeg[..., :1000].mean(-1,keepdims=True)
        first = centered[...,1000:1600].reshape(len(eeg),eeg.shape[1],30,20).mean(-1)
        second = centered[...,1600:3400].reshape(len(eeg),eeg.shape[1],30,60).mean(-1)
        features = np.concatenate((first,second),-1).reshape(len(eeg),-1); valid = np.ones(features.shape,bool)
    elif task == "refed_regression":
        values = np.where(mask,eeg,0.).reshape(len(eeg),eeg.shape[1],30,200).transpose(0,2,1,3)
        mask = mask.reshape(len(eeg),eeg.shape[1],30,200).transpose(0,2,1,3)
        count = mask.sum(-1);mean = values.sum(-1)/np.maximum(count,1);time=np.arange(200)/200.
        centered = np.where(mask,time-(time*mask).sum(-1)[...,None]/np.maximum(count,1)[...,None],0.)
        denominator=(centered**2).sum(-1);slope=(centered*values).sum(-1)/np.maximum(denominator,np.finfo(float).tiny)
        features=np.concatenate((mean,slope),-1);valid=np.concatenate((count>0,denominator>0),-1)
    else:
        raise ValueError("unknown native voltage probe task")
    return np.where(valid,features,0.).astype(np.float32),valid,[features.shape[-1]]
