"""DOWNSTREAM-REBENCH-v1 REVE-base features and coordinate-preserving FT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from comparative_methods.REVE.adapters.reve import METHOD_ROOT, load_verified_reve_base
from comparative_methods.rebench_eeg_common_v1 import (
    EEGRebenchFeatures,
    checkpoint_identity,
    synthetic_smoke as _synthetic_smoke,
    validate_profile,
)


class REVERebenchFeatures(EEGRebenchFeatures):
    def __init__(self, *args: Any, position_bank: nn.Module, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.position_bank = position_bank
        self.position_bank.requires_grad_(False).eval()
        unknown = [name for name in self.channel_names if name not in position_bank.mapping]
        if unknown:
            raise ValueError(f"REVE official position bank has no coordinates for: {unknown}")
        indices = torch.tensor(
            [position_bank.mapping[name] for name in self.channel_names],
            dtype=torch.long,
            device=position_bank.embedding.device,
        )
        positions = position_bank.embedding[indices].detach().clone()
        if positions.shape != (16, 3) or not bool(torch.isfinite(positions).all()):
            raise ValueError("invalid REVE official electrode coordinates")
        self.register_buffer("positions", positions)

    def train(self, mode: bool = True) -> REVERebenchFeatures:
        super().train(mode)
        if hasattr(self, "position_bank"):
            self.position_bank.eval()
        return self

    def _encode(self, eeg: torch.Tensor) -> torch.Tensor:
        positions = self.positions.unsqueeze(0).expand(eeg.shape[0], -1, -1).to(eeg)
        tokens = self.encoder(eeg=eeg, pos=positions)
        patches = 1 + (eeg.shape[-1] - 200) // 180
        if tokens.shape != (eeg.shape[0], 16, patches, self.feature_dim):
            raise RuntimeError(f"invalid REVE latent token shape: {tuple(tokens.shape)}")
        return self.encoder.attention_pooling(tokens)


def create_model(task: str, profile: dict, device: str, trainable: bool = False) -> nn.Module:
    validate_profile(task, profile)
    if profile.get("artifact_id", "reve_base") != "reve_base":
        raise ValueError("this rebenchmark adapter fixes the verified REVE-base encoder")
    encoder, position_bank, identity = load_verified_reve_base(
        device=device,
        method_root=Path(profile.get("method_root", METHOD_ROOT)),
    )
    overlap = task in {"motor_imagery", "mental_arithmetic"}
    return REVERebenchFeatures(
        encoder,
        position_bank=position_bank,
        task=task,
        profile=profile,
        trainable=trainable,
        feature_dim=512,
        method_identity="official_reve_base_position_bank_attention_pooling_rebench_v1",
        metadata={
            "checkpoint": checkpoint_identity(identity),
            "pooling": "pretrained_cls_query_attention_pooling",
            "pooling_query_trainable": bool(trainable),
            "coordinate_source": "official_reve_positions_bank_by_fixed_profile_channel_names",
            "position_bank_trainable": False,
            "source_deviation": "shared_16_measured_channel_panel_and_common_linear_readout",
            "target_corpus_overlap": "confirmed_Shin2017A_overlap" if overlap else "none_in_published_exhaustive_list",
            "reporting_track": "open_world_pretrained_with_target_corpus_overlap" if overlap else "official_pretrained",
        },
    )


def synthetic_smoke(task: str, profile: dict, device: str, trainable: bool = False, **kwargs: Any) -> dict:
    return _synthetic_smoke(create_model, task, profile, device, trainable, **kwargs)
