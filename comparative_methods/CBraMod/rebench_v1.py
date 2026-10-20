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
