"""DOWNSTREAM-REBENCH-v1 BIOT features; retained v2 adapters stay unchanged."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from comparative_methods.BIOT.adapters.biot import METHOD_ROOT, load_verified_biot_encoder
from comparative_methods.rebench_eeg_common_v1 import (
    EEGRebenchFeatures,
    checkpoint_identity,
    synthetic_smoke as _synthetic_smoke,
    validate_profile,
)


class BIOTRebenchFeatures(EEGRebenchFeatures):
    def _encode(self, eeg: torch.Tensor) -> torch.Tensor:
        # Keep the exact official STFT, sequential channel tokens and mean output.
        return self.encoder(eeg)


def create_model(task: str, profile: dict, device: str, trainable: bool = False) -> nn.Module:
    validate_profile(task, profile)
    encoder, identity = load_verified_biot_encoder(
        str(profile.get("artifact_id", "eeg_prest_16")),
        device=device,
        method_root=Path(profile.get("method_root", METHOD_ROOT)),
    )
    if identity.n_channels < 16:
        raise ValueError("BIOT checkpoint cannot represent the 16-channel panel")
    return BIOTRebenchFeatures(
        encoder,
        task=task,
        profile=profile,
        trainable=trainable,
        feature_dim=256,
        method_identity="official_pretrained_biot_native_electrode_positional_transfer_rebench_v1",
        metadata={
            "checkpoint": checkpoint_identity(identity),
            "pooling": "official_mean_transformer_tokens",
            "source_deviation": "measured_native_electrodes_in_fixed_order_not_original_bipolar_montage",
            "target_corpus_overlap": "none_by_declared_dataset_identity",
        },
    )


def synthetic_smoke(task: str, profile: dict, device: str, trainable: bool = False, **kwargs: Any) -> dict:
    return _synthetic_smoke(create_model, task, profile, device, trainable, **kwargs)
