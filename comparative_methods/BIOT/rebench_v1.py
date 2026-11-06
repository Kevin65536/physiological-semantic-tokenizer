"""BIOT features with retained v1 defaults and an explicit PREST subset view."""

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


PREST_PARASAGITTAL_PAIRS = (
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
)


def prest_bipolar_profile(profile: dict) -> dict:
    """The exact PREST slots 8–15 from electrodes already in this panel."""
    lookup = {name.casefold(): name for name in profile["channel_names"]}
    if any(name.casefold() not in lookup for pair in PREST_PARASAGITTAL_PAIRS for name in pair):
        raise ValueError("PREST parasagittal montage requires exact physical electrodes; no nearest-channel substitution")
    pairs = [[lookup[name.casefold()] for name in pair] for pair in PREST_PARASAGITTAL_PAIRS]
    names = ["-".join(pair) for pair in pairs]
    return {**profile, "biot_montage": "prest_parasagittal8", "eeg_derivations": pairs,
            "channel_names": names, "eeg_panel": "derived_subset_of_same_matched16",
            "eeg_geometry": [dict(name=name, kind="bipolar_voltage_difference", electrodes=pair)
                             for name, pair in zip(names, pairs)]}


class BIOTRebenchFeatures(EEGRebenchFeatures):
    def __init__(self, *args, channel_offset: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_offset = channel_offset

    def _encode(self, eeg: torch.Tensor) -> torch.Tensor:
        # Keep the exact official STFT, sequential channel tokens and mean output.
        return self.encoder(eeg, n_channel_offset=self.channel_offset) if self.channel_offset else self.encoder(eeg)


def create_model(task: str, profile: dict, device: str, trainable: bool = False) -> nn.Module:
    validate_profile(task, profile)
    montage = profile.get("biot_montage", "native_positional")
    if montage not in {"native_positional", "prest_parasagittal8"}:
        raise ValueError("unknown BIOT input montage")
    if montage == "prest_parasagittal8":
        expected = ["-".join(pair).casefold() for pair in PREST_PARASAGITTAL_PAIRS]
        if [str(name).casefold() for name in profile["channel_names"]] != expected:
            raise ValueError("BIOT PREST slots 8–15 require the exact ordered bipolar derivations")
    offset = 8 if montage == "prest_parasagittal8" else 0
    encoder, identity = load_verified_biot_encoder(
        str(profile.get("artifact_id", "eeg_prest_16")),
        device=device,
        method_root=Path(profile.get("method_root", METHOD_ROOT)),
    )
    if identity.n_channels < len(profile["channel_names"]) + offset:
        raise ValueError("BIOT checkpoint cannot represent the requested channel panel")
    return BIOTRebenchFeatures(
        encoder,
        task=task,
        profile=profile,
        trainable=trainable,
        feature_dim=256,
        channel_offset=offset,
        method_identity=("official_pretrained_biot_prest_parasagittal8_rebench_v2" if offset else
                         "official_pretrained_biot_native_electrode_positional_transfer_rebench_v1"),
        metadata={
            "checkpoint": checkpoint_identity(identity),
            "pooling": "official_mean_transformer_tokens",
            "source_deviation": ("exact_PREST_slots8to15_subset_from_same_measured_panel;not_full16montage_reproduction" if offset else
                                 "measured_native_electrodes_in_fixed_order_not_original_bipolar_montage"),
            "channel_token_offset": offset,
            "target_corpus_overlap": "none_by_declared_dataset_identity",
        },
    )


def synthetic_smoke(task: str, profile: dict, device: str, trainable: bool = False, **kwargs: Any) -> dict:
    return _synthetic_smoke(create_model, task, profile, device, trainable, **kwargs)
