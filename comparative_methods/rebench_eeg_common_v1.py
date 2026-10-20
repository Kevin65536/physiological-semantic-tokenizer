"""Shared input invariants for the three EEG-only rebenchmark adapters.

This module reads no measurement data and owns no experiment lifecycle. Scaling
is fitted by the caller on its permitted training partition, before this API.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn


CLASSIFICATION_TASKS = frozenset(
    {"motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual"}
)


def validate_profile(task: str, profile: Mapping[str, Any]) -> tuple[tuple[str, ...], int]:
    if task == "refed_regression":
        raise NotImplementedError("REFED requires a validated temporal output and target-mask interface")
    if task not in CLASSIFICATION_TASKS:
        raise ValueError(f"unknown rebenchmark classification task: {task!r}")
    if float(profile.get("sampling_rate_hz", 0)) != 200.0:
        raise ValueError("EEG rebenchmark adapters require 200 Hz")
    raw_names = profile.get("channel_names", ())
    if isinstance(raw_names, str):
        raise ValueError("channel_names must identify 16 distinct measured EEG channels")
    names = tuple(str(name) for name in raw_names)
    if len(names) != 16 or len(set(names)) != 16 or any(not name for name in names):
        raise ValueError("channel_names must identify 16 distinct measured EEG channels")
    expected_samples = 400 if task == "dsr" else 1600
    if profile.get("input_samples", expected_samples) != expected_samples:
        raise ValueError(f"{task} requires {expected_samples} EEG samples")
    return names, expected_samples


def checkpoint_identity(metadata: Any) -> dict[str, Any]:
    values = asdict(metadata) if is_dataclass(metadata) else dict(metadata)
    return {name: str(value) if isinstance(value, Path) else value for name, value in values.items()}


class EEGRebenchFeatures(nn.Module):
    """One fixed measured panel, optional backbone gradients, no missing support."""

    supports_mask = False
    supports_temporal_output = False
    supports_fnirs = False

    def __init__(
        self,
        encoder: nn.Module,
        *,
        task: str,
        profile: Mapping[str, Any],
        trainable: bool,
        feature_dim: int,
        method_identity: str,
        fixed_parameter_names: tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.channel_names, self.input_samples = validate_profile(task, profile)
        self.task = task
        self.encoder = encoder
        self.trainable = bool(trainable)
        self.feature_dim = int(feature_dim)
        self.method_identity = method_identity
        self.sampling_rate_hz = 200.0
        for name, parameter in self.encoder.named_parameters():
            # BIOT has an integer index Parameter; CBraMod has a deliberately
            # fixed mask encoding. Neither becomes a learned quantity during FT.
            parameter.requires_grad_(
                self.trainable
                and parameter.is_floating_point()
                and name not in fixed_parameter_names
            )
        self.metadata = {
            "schema": "downstream_rebench_eeg_adapter_v1",
            "method_identity": method_identity,
            "task": task,
            "feature_dim": self.feature_dim,
            "channel_names": list(self.channel_names),
            "sampling_rate_hz": self.sampling_rate_hz,
            "input_samples": self.input_samples,
            "trainable": self.trainable,
            "supports_mask": False,
            "supports_temporal_output": False,
            "supports_fnirs": False,
            "input_preprocessing": "caller_train_partition_scaling_no_adapter_normalization",
            **dict(metadata or {}),
        }
        self.train(False)

    def train(self, mode: bool = True) -> EEGRebenchFeatures:
        super().train(mode)
        self.encoder.train(bool(mode) and self.trainable)
        return self

    def forward(
        self, eeg: torch.Tensor, fnirs: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        if fnirs is not None:
            raise ValueError("this adapter accepts EEG only")
        if eeg.ndim != 3 or eeg.shape[0] < 1 or eeg.shape[1:] != (16, self.input_samples):
            raise ValueError(f"EEG must have shape [B,16,{self.input_samples}] with B >= 1")
        if float(kwargs.pop("sampling_rate_hz", self.sampling_rate_hz)) != self.sampling_rate_hz:
            raise ValueError("EEG rebenchmark adapters require 200 Hz")
        if tuple(kwargs.pop("channel_names", self.channel_names)) != self.channel_names:
            raise ValueError("EEG channel order differs from the fixed profile")
        for key, shape in (
            ("channel_valid", (eeg.shape[0], 16)),
            ("sample_valid", (eeg.shape[0], self.input_samples)),
        ):
            valid = kwargs.pop(key, None)
            if valid is not None and (
                not torch.is_tensor(valid)
                or valid.dtype != torch.bool
                or tuple(valid.shape) != shape
                or not bool(valid.all())
            ):
                raise ValueError(f"{key}: missing or padded EEG support is unsupported")
        eeg_mask = kwargs.pop("eeg_mask", None)
        if eeg_mask is not None and (
            not torch.is_tensor(eeg_mask)
            or eeg_mask.dtype != torch.bool
            or tuple(eeg_mask.shape) not in {tuple(eeg.shape), (eeg.shape[0], self.input_samples)}
            or not bool(eeg_mask.all())
        ):
            raise ValueError("eeg_mask: missing or padded EEG support is unsupported")
        if kwargs:
            raise TypeError(f"unsupported adapter arguments: {sorted(kwargs)}")
        if not eeg.is_floating_point() or not bool(torch.isfinite(eeg).all()):
            raise ValueError("EEG must contain finite floating-point values")
        with nullcontext() if self.trainable else torch.no_grad():
            features = self._encode(eeg)
        if features.shape != (eeg.shape[0], self.feature_dim) or not bool(torch.isfinite(features).all()):
            raise RuntimeError(f"invalid EEG feature shape or values: {tuple(features.shape)}")
        return features

    def _encode(self, eeg: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def synthetic_smoke(
    factory: Callable[..., nn.Module],
    task: str,
    profile: dict[str, Any],
    device: str,
    trainable: bool = False,
    *,
    batch_size: int = 2,
    seed: int = 17,
) -> dict[str, Any]:
    """Exercise official weights with synthetic inputs; caller owns scheduling."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    validate_profile(task, profile)
    target_device = torch.device(device)
    cuda_devices = []
    if target_device.type == "cuda":
        target_device = torch.device("cuda", target_device.index if target_device.index is not None else torch.cuda.current_device())
        cuda_devices = [target_device.index]
        torch.cuda.reset_peak_memory_stats(target_device)
        torch.cuda.synchronize(target_device)
    started = time.monotonic()
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        model = factory(task, profile, str(target_device), trainable=trainable)
        model.train()
        head = nn.Linear(model.feature_dim, 2, device=target_device)
        eeg = torch.randn(batch_size, 16, model.input_samples, device=target_device)
        features = model(eeg)
        loss = nn.functional.cross_entropy(head(features), torch.arange(batch_size, device=target_device) % 2)
        loss.backward()
        gradients = [parameter.grad for parameter in model.encoder.parameters() if parameter.grad is not None]
        if not bool(torch.isfinite(loss)) or any(not bool(torch.isfinite(grad).all()) for grad in gradients):
            raise RuntimeError("non-finite synthetic smoke loss or gradients")
        gradient_norm = sum(float(grad.detach().double().square().sum()) for grad in gradients) ** 0.5
        if trainable and (not gradients or gradient_norm == 0):
            raise RuntimeError("FT smoke did not propagate a nonzero gradient to the backbone")
        if not trainable and gradients:
            raise RuntimeError("LP smoke unexpectedly propagated gradients to the backbone")
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        return {
            "method_identity": model.method_identity,
            "task": task,
            "trainable": trainable,
            "batch_size": batch_size,
            "feature_shape": list(features.shape),
            "loss": float(loss.detach()),
            "backbone_grad_norm": gradient_norm,
            "backbone_grad_parameter_count": len(gradients),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(target_device) if target_device.type == "cuda" else 0,
            "elapsed_seconds": time.monotonic() - started,
            "metadata": model.metadata,
        }
