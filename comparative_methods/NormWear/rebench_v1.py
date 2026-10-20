"""NormWear EEG–fNIRS features for DOWNSTREAM-REBENCH-v1.

The common trainer owns train-fitted input and feature standardization. This
adapter retains only the declared stateless 65 Hz polyphase resampling and the
pinned upstream CWT. The upstream corpus did not contain fNIRS. Inputs contain
all profile-listed real channels, with HbO followed by HbR in ``fnirs``.
Missing input support is retained through the common trainer's training-mean
imputation: after training-only standardization it is exactly zero. Input masks
validate that imputation; the pinned upstream encoder has no native mask input.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from comparative_methods.NormWear.adapters.normwear import (
    NormWearFrozenEncoder,
    load_verified_normwear_encoder,
)


METHOD_ID = "normwear_eeg_fnirs_adapted"
SUPPORTED_TASKS = frozenset(
    {"motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual"}
)
REFED_UNSUPPORTED_REASON = (
    "NORMWEAR_NO_PARTIAL_TIME_MASK_OR_MASKED_SEQUENCE_REGRESSION_CONTRACT"
)


def _names(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"NormWear profile {field} must list real channel identities")
    result = tuple(str(name) for name in value)
    if not result or any(not name.strip() for name in result) or len(set(result)) != len(result):
        raise ValueError(f"NormWear profile {field} must contain unique nonempty identities")
    return result


def _profile(task: str, profile: Mapping[str, Any]) -> dict[str, Any]:
    # Reject unsupported targets before opening source files or weights.
    if task in {"refed", "refed_regression"}:
        raise NotImplementedError(REFED_UNSUPPORTED_REASON)
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported NormWear rebenchmark task: {task}")
    eeg_names = _names(
        profile.get("channel_names", profile.get("eeg_channel_names")), "channel_names"
    )
    locations = _names(profile.get("fnirs_location_names"), "fnirs_location_names")
    if "eeg_channel_names" in profile and _names(
        profile["eeg_channel_names"], "eeg_channel_names"
    ) != eeg_names:
        raise ValueError("NormWear profile EEG channel aliases disagree")
    eeg_rate = float(profile.get("sampling_rate_hz", profile.get("eeg_sample_rate_hz", 200)))
    hb_rate = float(profile.get("fnirs_sampling_rate_hz", 10))
    if eeg_rate != 200 or hb_rate != 10:
        raise ValueError("NormWear rebenchmark requires EEG 200 Hz and fNIRS 10 Hz")
    if "eeg_sample_rate_hz" in profile and float(profile["eeg_sample_rate_hz"]) != eeg_rate:
        raise ValueError("NormWear profile EEG sampling-rate aliases disagree")
    duration = 2.0 if task == "dsr" else 8.0
    if float(profile.get("duration_s", duration)) != duration:
        raise ValueError(f"NormWear {task} requires {duration:g} seconds")
    input_samples, fnirs_samples = int(duration * eeg_rate), int(duration * hb_rate)
    if profile.get("input_samples", input_samples) != input_samples:
        raise ValueError("NormWear profile input_samples disagrees with task duration")
    if profile.get("fnirs_samples", fnirs_samples) != fnirs_samples:
        raise ValueError("NormWear profile fnirs_samples disagrees with task duration")
    hb_names = tuple(
        profile.get(
            "fnirs_channel_names",
            [*(f"hbo:{name}" for name in locations), *(f"hbr:{name}" for name in locations)],
        )
    )
    hb_names = _names(hb_names, "fnirs_channel_names")
    if len(hb_names) != 2 * len(locations):
        raise ValueError("NormWear profile requires one HbO and one HbR channel per location")
    chunk_size = int(profile.get("channel_chunk_size", 16))
    if chunk_size < 1:
        raise ValueError("NormWear channel_chunk_size must be positive")
    return {
        "channel_names": eeg_names,
        "fnirs_location_names": locations,
        "fnirs_channel_names": hb_names,
        "sampling_rate_hz": eeg_rate,
        "fnirs_sampling_rate_hz": hb_rate,
        "duration_s": duration,
        "input_samples": input_samples,
        "fnirs_samples": fnirs_samples,
        "channel_chunk_size": chunk_size,
        "gradient_checkpointing": bool(profile.get("gradient_checkpointing", True)),
    }


class NormWearRebenchEncoder(NormWearFrozenEncoder):
    """Same channel-fusion computation with frozen-LP or trainable-FT execution."""

    supports_mask = True  # Adapter imputation contract, not native attention masking.

    def __init__(
        self,
        backbone: nn.Module,
        upstream_module: Any,
        *,
        task: str,
        profile: Mapping[str, Any],
        trainable: bool,
        checkpoint_metadata: Mapping[str, Any],
    ) -> None:
        resolved = _profile(task, profile)
        super().__init__(
            backbone, upstream_module, channel_chunk_size=resolved["channel_chunk_size"]
        )
        self.task = task
        self.profile = resolved
        self.trainable = bool(trainable)
        self.gradient_checkpointing = resolved["gradient_checkpointing"]
        self.feature_dim = 768 * (
            len(resolved["channel_names"]) + 2 * len(resolved["fnirs_location_names"])
        )
        self.backbone.requires_grad_(self.trainable)
        # Sinusoidal pretrained positions remain a fixed interpolated coordinate.
        self.backbone.pos_embed.requires_grad_(False)
        self.method_identity = {
            "method_id": METHOD_ID,
            "display_name": "NormWear EEG-fNIRS adapted",
            "adapter_version": "downstream_rebench_v1",
            "track": "full_finetune" if self.trainable else "frozen_linear_probe",
            "pretraining_included_fnirs": False,
            "original_paper_fnirs_reproduction": False,
            "checkpoint": dict(checkpoint_metadata),
            "input_profile": resolved,
            "feature_dim": self.feature_dim,
            "fnirs_order": "all_HbO_then_all_HbR_in_profile_location_order",
            "normalization_owner": "common_trainer_training_partition_only",
            "stateless_transforms": {
                "resampling": "scipy.signal.resample_poly_kaiser_beta5_constant_padding",
                "eeg_ratio_up_down": [13, 40],
                "fnirs_ratio_up_down": [13, 2],
                "model_sample_rate_hz": 65,
                "cwt": "pinned_upstream_optimized_pytorch_ricker",
                "cwt_planes": ["signal_aligned_to_second_difference", "first_difference", "second_difference"],
                "cwt_scales_start_stop_step": [0.1, 64.0, 1.0],
                "cwt_scale_count": 65,
                "cwt_maximum_wavelet_samples": 640,
                "input_transform_gradients": False,
                "upstream_basic_preproc_applied": False,
            },
            "pooling": "mean_all_final_norm_tokens_including_cls_then_concatenate_real_channels",
            "patch_kernel_stride": [9, 5],
            "position_interpolation": "bicubic_align_corners_false_from_43x13",
            "mask_handling": "train_mean_standardized_zero_imputation_preserve_input_masks",
            "input_imputation": {
                "fitted_statistics_owner": "common_trainer_training_partition_only",
                "missing_value_in_standardized_coordinate": 0.0,
                "input_mask_interface": "eeg_mask_and_fnirs_mask_bool_B_C_T",
                "input_mask_owner": "common_trainer_batch_and_input_coverage_audit",
                "native_mask_support": False,
                "sample_and_channel_policy": "retain_all_windows_and_profile_channel_positions",
                "limitation": "imputed_values_enter_CWT_and_complete_cross_channel_CLS_fusion",
            },
            "channel_chunk_size": self.channel_chunk_size,
            "gradient_checkpointing": self.gradient_checkpointing and self.trainable,
        }
        self.train(self.trainable)

    def train(self, mode: bool = True) -> "NormWearRebenchEncoder":
        nn.Module.train(self, mode)
        self.backbone.train(bool(mode and self.trainable))
        return self

    def _encode_layer(
        self, block: nn.Module, tokens: torch.Tensor, batch: int, channels: int
    ) -> torch.Tensor:
        independent = torch.cat(
            [
                block.variate_encoder(tokens[start : start + self.channel_chunk_size])
                for start in range(0, tokens.shape[0], self.channel_chunk_size)
            ],
            dim=0,
        )
        if block.curr_layer % block.fuse_frequency != 0 or block.no_fusion:
            return independent
        token_count, embedding_dim = independent.shape[1:]
        structured = independent.reshape(batch, channels, token_count, embedding_dim)
        cls = structured.mean(dim=2) if block.mean_fuse else structured[:, :, 0, :]
        fused = block.cls_fusion(cls).unsqueeze(2)
        return torch.cat((fused, structured[:, :, 1:, :]), dim=2).reshape(
            batch * channels, token_count, embedding_dim
        )

    def encode_cwt(self, cwt: torch.Tensor) -> torch.Tensor:
        if not (self.trainable and self.gradient_checkpointing and torch.is_grad_enabled()):
            return super().encode_cwt(cwt)
        if cwt.ndim != 5 or cwt.shape[2] != 3 or cwt.shape[-1] != 65:
            raise ValueError(f"NormWear CWT input shape is invalid: {tuple(cwt.shape)}")
        batch, channels, planes, times, scales = cwt.shape
        tokens = self.backbone.patch_embed(cwt.reshape(batch * channels, planes, times, scales))
        with torch.no_grad():
            position = self.backbone.pos_adjust((times, scales), device=cwt.device)
        tokens = tokens + position[:, 1:, :]
        cls = (self.backbone.cls_token + position[:, :1, :]).expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        for block in self.backbone.encoder_blocks:
            tokens = checkpoint(
                self._encode_layer, block, tokens, batch, channels, use_reentrant=False
            )
        tokens = self.backbone.norm(tokens)
        return tokens.reshape(batch, channels, tokens.shape[1], tokens.shape[2])

    def forward(
        self, eeg: torch.Tensor, fnirs: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        if fnirs is None:
            raise ValueError("NormWear EEG-fNIRS adapted requires paired HbO/HbR input")
        expected_eeg = (len(self.profile["channel_names"]), self.profile["input_samples"])
        expected_hb = (2 * len(self.profile["fnirs_location_names"]), self.profile["fnirs_samples"])
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != expected_eeg:
            raise ValueError(f"NormWear EEG must have shape [B,{expected_eeg[0]},{expected_eeg[1]}]")
        if fnirs.ndim != 3 or tuple(fnirs.shape[1:]) != expected_hb:
            raise ValueError(f"NormWear fNIRS must have shape [B,{expected_hb[0]},{expected_hb[1]}]")
        allowed = {"channel_valid", "eeg_sample_valid", "fnirs_sample_valid", "eeg_mask", "fnirs_mask"}
        if set(kwargs) - allowed:
            raise TypeError(f"unsupported NormWear forward fields: {sorted(set(kwargs) - allowed)}")
        if eeg.device != fnirs.device:
            raise ValueError("NormWear modalities must be on the same device")
        for label, values in (("eeg", eeg), ("fnirs", fnirs)):
            mask = kwargs.pop(f"{label}_mask", None)
            if mask is None:
                continue
            if (not torch.is_tensor(mask) or mask.dtype != torch.bool
                    or tuple(mask.shape) != tuple(values.shape) or mask.device != values.device):
                raise ValueError(f"NormWear {label}_mask must be bool [B,C,T] matching input shape and device")
            # The trainer has the training-fitted coordinate. A nonzero value
            # behind false support must never reach the fixed upstream model.
            if not bool((values[~mask] == 0).all()):
                raise ValueError(
                    f"NormWear missing {label} support must already have training-mean standardized zero fill"
                )
        hbo, hbr = fnirs.chunk(2, dim=1)
        # Input preprocessing is fixed. Only the backbone participates in FT.
        with torch.no_grad():
            model_input, _ = self.prepare_model_input(
                eeg,
                hbo,
                hbr,
                eeg_sampling_rate_hz=self.profile["sampling_rate_hz"],
                fnirs_sampling_rate_hz=self.profile["fnirs_sampling_rate_hz"],
                eeg_channel_names=self.profile["channel_names"],
                fnirs_location_names=self.profile["fnirs_location_names"],
                **kwargs,
            )
            cwt = self.calculate_cwt(model_input)
        with nullcontext() if self.trainable else torch.no_grad():
            features = self.encode_cwt(cwt).mean(dim=2).flatten(start_dim=1)
        if features.shape != (eeg.shape[0], self.feature_dim):
            raise RuntimeError("NormWear feature dimension drifted")
        if not bool(torch.isfinite(features).all()):
            raise RuntimeError("NormWear encoder produced non-finite features")
        return features


def create_model(
    task: str, profile: dict[str, Any], device: str, trainable: bool = False
) -> nn.Module:
    """Load only the manifest-fixed official encoder, never a prior run/cache."""
    _profile(task, profile)
    backbone, upstream, metadata = load_verified_normwear_encoder(device=device)
    checkpoint_metadata = asdict(metadata)
    checkpoint_metadata["path"] = str(metadata.path)
    return NormWearRebenchEncoder(
        backbone,
        upstream,
        task=task,
        profile=profile,
        trainable=trainable,
        checkpoint_metadata=checkpoint_metadata,
    ).to(device)
