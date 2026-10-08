"""Independent continuous semantic/observation encoders for synthetic discovery.

The source coordinates are a reference-centred shape and a record log amplitude.
They are generator-conditional coordinates, not recovered absolute physiology.
The convolutional encoders use both past and future within each offline window.
Gaussian heads describe predictive marginal uncertainty; they are not a full
Bayesian posterior and do not specify temporal covariance.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn


def visible_values(values: torch.Tensor, mask: torch.Tensor | None = None):
    """Remove hidden values *before* arithmetic, including NaN/Inf poison."""
    if values.ndim != 3:
        raise ValueError("values must have shape [batch,time,channel]")
    if mask is None:
        mask = torch.ones_like(values, dtype=torch.bool)
    else:
        mask = torch.as_tensor(mask, device=values.device, dtype=torch.bool)
        if mask.ndim == 2 and mask.shape == values.shape[:2]:
            mask = mask.unsqueeze(-1)
        try:
            mask = torch.broadcast_to(mask, values.shape)
        except RuntimeError as exc:
            raise ValueError("mask must broadcast to [batch,time,channel]") from exc
    clean = torch.where(mask, values, torch.zeros_like(values))
    if not torch.isfinite(clean).all():
        raise ValueError("visible observations must be finite")
    return clean, mask


class TrainingNormalizer(nn.Module):
    """One train-fitted channel coordinate, reused unchanged for every split."""

    def __init__(self, channels: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(channels))
        self.register_buffer("scale", torch.ones(channels))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, values, mask=None, *, partition="train"):
        if partition != "train":
            raise ValueError("normalization may only fit the training partition")
        clean, valid = visible_values(values, mask)
        count = valid.sum(dim=(0, 1))
        if (count == 0).any():
            raise ValueError("every channel needs training support")
        mean = clean.sum(dim=(0, 1)) / count
        residual = torch.where(valid, clean - mean, torch.zeros_like(clean))
        scale = (residual.square().sum(dim=(0, 1)) / count).sqrt().clamp_min(1e-5)
        self.mean.copy_(mean)
        self.scale.copy_(scale)
        self.fitted.fill_(True)
        return self

    def forward(self, values, mask=None):
        clean, valid = visible_values(values, mask)
        normalized = torch.where(valid, (clean - self.mean) / self.scale,
                                 torch.zeros_like(clean))
        return normalized, valid


class _TemporalEncoder(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.stem = nn.Linear(2 * channels, width)
        self.layers = nn.ModuleList([
            nn.Conv1d(width, width, 9, padding=4 * dilation, dilation=dilation)
            for dilation in (1, 3, 9)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in self.layers])

    def forward(self, clean, valid):
        hidden = torch.nn.functional.gelu(self.stem(torch.cat((clean, valid.to(clean.dtype)), -1)))
        for layer, norm in zip(self.layers, self.norms):
            update = layer(hidden.transpose(1, 2)).transpose(1, 2)
            hidden = norm(hidden + torch.nn.functional.gelu(update))
        return hidden


class _ModalityBranch(nn.Module):
    def __init__(self, channels, width, observation_dimensions, reference_steps,
                 observation_only):
        super().__init__()
        self.channels = channels
        self.reference_steps = reference_steps
        self.observation_only = observation_only
        self.normalizer = TrainingNormalizer(channels)
        self.source_encoder = _TemporalEncoder(channels, width)
        self.observation_encoder = _TemporalEncoder(channels, width)
        # All arms instantiate identical heads and parameters. Point arms leave
        # the scale heads unsupervised and do not claim their values as intervals.
        self.shape_head = nn.Linear(width, 2)
        self.amplitude_head = nn.Linear(width, 2)
        self.observation_head = nn.Linear(width, observation_dimensions)
        self.decoder = nn.Sequential(
            nn.Linear(observation_dimensions + 2, width), nn.GELU(),
            nn.Linear(width, channels),
        )
        nn.init.constant_(self.shape_head.bias[1:], -1.0)
        nn.init.constant_(self.amplitude_head.bias[1:], -1.0)

    def forward(self, values, mask=None):
        clean, valid = self.normalizer(values, mask)
        source_hidden = self.source_encoder(clean, valid)
        shape_parameters = self.shape_head(source_hidden)
        shape_mean = shape_parameters[..., 0]
        shape_mean = shape_mean - shape_mean[:, :self.reference_steps].mean(1, keepdim=True)
        amplitude_parameters = self.amplitude_head(source_hidden.mean(1))
        observation = self.observation_head(self.observation_encoder(clean, valid))
        source_condition = torch.stack((shape_mean,
            amplitude_parameters[:, 0, None].expand_as(shape_mean)), dim=-1)
        if not self.observation_only:
            source_condition = source_condition.detach()
        reconstruction = self.decoder(torch.cat((source_condition, observation), -1))
        return {
            "shape_mean": shape_mean,
            "shape_log_sd": shape_parameters[..., 1].clamp(-8.0, 4.0),
            "log_amplitude_mean": amplitude_parameters[:, 0],
            "log_amplitude_log_sd": amplitude_parameters[:, 1].clamp(-8.0, 4.0),
            "observation_code": observation,
            "reconstruction": reconstruction,
            "time_support": valid.any(-1),
            "channel_support": valid,
        }


class SemanticResponseTokenizer(nn.Module):
    """Two self-contained modality branches with equal per-arm capacity."""

    def __init__(self, steps=120, eeg_channels=30, width=64,
                 observation_dimensions=8, reference_steps=20, patches=12,
                 distribution=False, observation_only=False):
        super().__init__()
        if min(steps, eeg_channels, width, observation_dimensions, reference_steps, patches) < 1:
            raise ValueError("dimensions must be positive")
        if steps % patches or reference_steps > steps:
            raise ValueError("patches must divide steps and reference must fit")
        self.steps, self.eeg_channels = steps, eeg_channels
        self.patches, self.distribution = patches, distribution
        self.observation_only = observation_only
        self.eeg_branch = _ModalityBranch(eeg_channels, width, observation_dimensions,
                                         reference_steps, observation_only)
        self.hb_branch = _ModalityBranch(2, width, observation_dimensions,
                                        reference_steps, observation_only)

    def _check(self, values, channels):
        if values.ndim != 3 or values.shape[1:] != (self.steps, channels):
            raise ValueError(f"expected [batch,{self.steps},{channels}]")

    def encode_eeg(self, eeg, mask=None):
        self._check(eeg, self.eeg_channels)
        return self.eeg_branch(eeg, mask)

    def encode_hb(self, hb, mask=None):
        self._check(hb, 2)
        return self.hb_branch(hb, mask)

    def forward(self, eeg, hb, eeg_mask=None, hb_mask=None):
        return {"eeg": self.encode_eeg(eeg, eeg_mask),
                "hb": self.encode_hb(hb, hb_mask)}

    def export_tokens(self, output: Mapping[str, torch.Tensor]):
        """Aggregate *after* continuous inference, preserving support counts.

        Marginal SDs have no temporal covariance. The mean SD in each patch is
        the upper bound on aggregate SD under perfectly positive correlation;
        it is explicitly not a calibrated patch posterior or confidence bound.
        """
        mean = output["shape_mean"]
        support = output["time_support"].reshape(-1, self.patches, self.steps // self.patches)
        values = mean.reshape_as(support)
        count = support.sum(-1)
        denominator = count.clamp_min(1)
        token_mean = torch.where(support, values, torch.zeros_like(values)).sum(-1) / denominator
        sd = output["shape_log_sd"].exp().reshape_as(values)
        token_sd = torch.where(support, sd, torch.zeros_like(sd)).sum(-1) / denominator
        return {
            "semantic_mean": token_mean,
            "semantic_sd_upper_bound": token_sd if self.distribution else torch.full_like(token_sd, float("nan")),
            "support": count > 0, "support_count": count,
            "log_amplitude_mean": output["log_amplitude_mean"],
            "log_amplitude_sd": output["log_amplitude_log_sd"].exp() if self.distribution else None,
            "observation_code": output["observation_code"],
            "continuous_shape_mean": mean,
            "continuous_shape_sd": output["shape_log_sd"].exp() if self.distribution else None,
        }

    def token_metadata(self):
        return {
            "schema": "semantic_response_tokenizer_v1", "continuous": True,
            "source_coordinate": "synthetic_reference_centered_shape_and_train_standardized_log_amplitude",
            "source_status": "generator_conditional_measured_physiology_unqualified",
            "encoder_context": "offline_bidirectional_window",
            "modality_inputs": {"eeg": "own_30_features_and_mask", "hb": "own_HbO_HbR_and_mask"},
            "decoder_condition": "own_source_mean_and_own_observation_code",
            "reconstruction_gradient_to_source": self.observation_only,
            "patches": self.patches, "aggregation": "mean_of_directly_supported_continuous_predictions",
            "uncertainty": "Gaussian_predictive_marginals_not_full_Bayesian_posterior" if self.distribution else "none",
            "network_log_SD_bounds": [-8.0, 4.0],
            "patch_uncertainty": "perfect_positive_correlation_SD_upper_bound_not_calibrated_interval",
            "qualified_for_measured_physiology": False,
        }


def gaussian_crps(mean, log_sd, target):
    """Proper univariate Gaussian CRPS, in the coordinate's own units."""
    sd = log_sd.exp()
    z = (target - mean) / sd
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return sd * (z * (2 * cdf - 1) + 2 * phi - 1 / math.sqrt(math.pi))


def gaussian_nll(mean, log_sd, target):
    return 0.5 * ((target - mean) * torch.exp(-log_sd)).square() + log_sd + 0.5 * math.log(2 * math.pi)


def masked_mean(values, mask):
    mask = torch.broadcast_to(mask.to(torch.bool), values.shape)
    return torch.where(mask, values, torch.zeros_like(values)).sum() / mask.sum().clamp_min(1)


def response_loss(model, outputs, eeg, hb, shape, log_amplitude,
                  eeg_mask=None, hb_mask=None, source_mask=None,
                  reconstruction_weight=0.1, amplitude_weight=1.0, source_weight=1.0):
    """Loss sees truth; the encoders never receive it or the other modality."""
    if source_mask is None:
        source_mask = torch.ones_like(shape, dtype=torch.bool)
    source_mask = source_mask.to(torch.bool)
    losses = {}
    total = shape.new_zeros(())
    for modality, values, mask, branch in (("eeg", eeg, eeg_mask, model.eeg_branch),
                                         ("hb", hb, hb_mask, model.hb_branch)):
        output = outputs[modality]
        target, valid = branch.normalizer(values, mask)
        reconstruction = masked_mean((output["reconstruction"] - target).square(), valid)
        if model.observation_only:
            semantic = shape.new_zeros(())
        else:
            if model.distribution:
                shape_loss = gaussian_nll(output["shape_mean"], output["shape_log_sd"], shape)
                amplitude_loss = gaussian_nll(output["log_amplitude_mean"],
                                              output["log_amplitude_log_sd"], log_amplitude)
            else:
                shape_loss = (output["shape_mean"] - shape).square()
                amplitude_loss = (output["log_amplitude_mean"] - log_amplitude).square()
            semantic = masked_mean(shape_loss, source_mask) + amplitude_weight * amplitude_loss.mean()
        losses[f"{modality}_semantic"] = semantic
        losses[f"{modality}_reconstruction"] = reconstruction
        total = total + source_weight * semantic + reconstruction_weight * reconstruction
    losses["total"] = total
    return losses
