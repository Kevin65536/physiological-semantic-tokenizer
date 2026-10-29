"""DOWNSTREAM-REBENCH-v1 CBraMod features with genuine optional FT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from comparative_methods.CBraMod.adapters.cbramod import METHOD_ROOT, load_verified_cbramod_encoder
from comparative_methods.rebench_eeg_common_v1 import (
    EEGRebenchFeatures,
    checkpoint_identity,
    synthetic_smoke as _synthetic_smoke,
    validate_profile,
)


class CBraModRebenchFeatures(EEGRebenchFeatures):
    def _encode(self, eeg: torch.Tensor) -> torch.Tensor:
        batch, channels, samples = eeg.shape
        tokens = self.encoder(eeg.reshape(batch, channels, samples // 200, 200))
        if tokens.shape != (batch, channels, samples // 200, self.feature_dim):
            raise RuntimeError(f"invalid CBraMod latent token shape: {tuple(tokens.shape)}")
        return tokens.mean(dim=(1, 2))


def create_model(task: str, profile: dict, device: str, trainable: bool = False) -> nn.Module:
    validate_profile(task, profile)
    encoder, identity = load_verified_cbramod_encoder(
        str(profile.get("artifact_id", "cbramod_pretrained")),
        device=device,
        method_root=Path(profile.get("method_root", METHOD_ROOT)),
    )
    if not isinstance(encoder.proj_out, nn.Identity):
        raise ValueError("CBraMod downstream features require proj_out=Identity")
    return CBraModRebenchFeatures(
        encoder,
        task=task,
        profile=profile,
        trainable=trainable,
        feature_dim=200,
        fixed_parameter_names=("patch_embedding.mask_encoding",),
        method_identity="official_pretrained_cbramod_mean_token_transfer_rebench_v1",
        metadata={
            "checkpoint": checkpoint_identity(identity),
            "pooling": "mean_channel_and_patch_latent_tokens",
            "source_deviation": "project_mean_token_readout_replaces_upstream_nonlinear_downstream_head",
            "target_corpus_overlap": "none_by_declared_dataset_identity",
            "fixed_backbone_parameters": ["patch_embedding.mask_encoding"],
        },
    )


def synthetic_smoke(task: str, profile: dict, device: str, trainable: bool = False, **kwargs: Any) -> dict:
    return _synthetic_smoke(create_model, task, profile, device, trainable, **kwargs)


class CBraModAuditModel(nn.Module):
    """Explicit MA diagnostic; same observations, controlled scale and readout.

    The caller supplies training-standardized EEG. The physical-scale control
    restores its measured microvolt coordinate using training statistics only.
    The source head follows pinned upstream quick_example.py, with dimensions
    resolved from the shared panel/window. It is not a new pretraining model.
    """

    def __init__(self, task, profile, device, trainable=False):
        super().__init__()
        self.base = create_model(task, profile, device, trainable=trainable)
        self.readout = profile.get("audit_readout", "mean")
        self.scale_mode = profile.get("audit_scale", "train_sd")
        self.head_kind = profile.get("audit_head", "none")
        if self.readout not in {"mean", "flatten"} or self.scale_mode not in {"train_sd", "physical_100uv"}:
            raise ValueError("unknown CBraMod diagnostic readout or scale")
        if self.head_kind not in {"none", "linear", "source_mlp"}:
            raise ValueError("unknown CBraMod diagnostic head")
        self.feature_dim = 200 if self.readout == "mean" else 16 * (self.base.input_samples // 200) * 200
        self.supports_fnirs = False
        scaler = profile["audit_training_eeg_scaler"]
        self.register_buffer("input_mean", torch.tensor(scaler["mean"], dtype=torch.float32))
        self.register_buffer("input_scale", torch.tensor(scaler["scale"], dtype=torch.float32))
        if self.scale_mode == "physical_100uv" and profile.get("eeg_unit") != "uV":
            raise ValueError("physical CBraMod scale requires verified microvolt inputs")
        if self.head_kind == "source_mlp":
            if self.readout != "flatten":
                raise ValueError("source MLP requires ordered, flattened tokens")
            # Preserve the upstream quick-example hidden widths (800, 200).
            # Only the input dimension changes with the shared 30 s window.
            hidden = 800
            self.head = nn.Sequential(nn.Linear(self.feature_dim, hidden), nn.ELU(), nn.Dropout(.1),
                                      nn.Linear(hidden, 200), nn.ELU(), nn.Dropout(.1), nn.Linear(200, 2))
        elif self.head_kind == "linear":
            self.head = nn.Linear(self.feature_dim, 2)
        else:
            self.head = nn.Identity()
        self.head.requires_grad_(trainable)
        self.to(device)

    def forward(self, eeg, *, eeg_mask=None):
        if eeg.shape[1:] != (16, self.base.input_samples):
            raise ValueError("CBraMod audit input differs from its shared panel/window")
        if eeg_mask is not None and not bool(eeg_mask.all()):
            raise ValueError("CBraMod audit requires complete measured EEG support")
        if self.scale_mode == "physical_100uv":
            eeg = (eeg * self.input_scale + self.input_mean) / 100.0
        tokens = self.base.encoder(eeg.reshape(len(eeg), 16, -1, 200))
        features = tokens.mean((1, 2)) if self.readout == "mean" else tokens.flatten(1)
        return self.head(features)


def create_audit_model(task: str, profile: dict, device: str, trainable: bool = False):
    return CBraModAuditModel(task, profile, device, trainable)
