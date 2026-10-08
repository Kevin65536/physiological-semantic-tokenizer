"""Patch pooling adapter for the cached Croce physical-state posterior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn as nn


@dataclass
class PhysicalTeacherOutput:
    full_summary: torch.Tensor
    full_uncertainty: torch.Tensor
    eeg_target: torch.Tensor
    eeg_uncertainty: torch.Tensor
    fnirs_target: torch.Tensor
    fnirs_uncertainty: torch.Tensor
    valid_mask: torch.Tensor
    context_valid_mask: torch.Tensor
    entry_masks: Dict[str, Dict[str, torch.Tensor]]
    target_family: str
    target_version: str


class PhysicalStateTeacher(nn.Module):
    """Convert sample-rate posterior tensors into detached two-second targets."""

    state_names = ("s", "delta_f", "delta_hbo", "delta_hb", "r")

    def __init__(
        self,
        patch_duration_s: float = 2.0,
        fnirs_sample_rate_hz: float = 10.0,
        eeg_sample_rate_hz: float = 200.0,
        target_family: str = "croce_physical_state",
        target_version: str = "physiology_semantic_v2",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.fnirs_patch_samples = int(round(patch_duration_s * fnirs_sample_rate_hz))
        self.eeg_patch_samples = int(round(patch_duration_s * eeg_sample_rate_hz))
        self.fnirs_sample_rate_hz = float(fnirs_sample_rate_hz)
        self.eeg_sample_rate_hz = float(eeg_sample_rate_hz)
        self.target_family = str(target_family)
        self.target_version = str(target_version)
        self.eps = float(eps)
        if self.fnirs_patch_samples <= 1 or self.eeg_patch_samples <= 1:
            raise ValueError("Teacher patches require at least two samples")

    @staticmethod
    def _reshape_patches(value: torch.Tensor, patch_samples: int) -> torch.Tensor:
        if value.dim() != 3:
            raise ValueError(f"Expected [B,T,D], got {tuple(value.shape)}")
        if value.shape[1] % patch_samples != 0:
            raise ValueError(f"Length {value.shape[1]} is not divisible by patch size {patch_samples}")
        return value.reshape(value.shape[0], value.shape[1] // patch_samples, patch_samples, value.shape[2])

    def _summarize(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor,
        sample_rate_hz: float,
        patch_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean_patch = self._reshape_patches(mean, patch_samples)
        var_patch = self._reshape_patches(variance, patch_samples).clamp_min(0.0)
        time = torch.arange(patch_samples, device=mean.device, dtype=mean.dtype) / sample_rate_hz
        centered_time = time - time.mean()
        slope = (mean_patch * centered_time.view(1, 1, -1, 1)).sum(dim=2)
        slope = slope / centered_time.square().sum().clamp_min(self.eps)
        patch_mean = mean_patch.mean(dim=2)
        total_variance = var_patch.mean(dim=2) + mean_patch.var(dim=2, unbiased=False)
        log_variance = total_variance.clamp_min(self.eps).log()
        summary = torch.cat((patch_mean, slope, log_variance), dim=-1)

        mean_uncertainty = var_patch.mean(dim=2) / patch_samples
        slope_weights = centered_time / centered_time.square().sum().clamp_min(self.eps)
        slope_uncertainty = (var_patch * slope_weights.square().view(1, 1, -1, 1)).sum(dim=2)
        logvar_uncertainty = 2.0 / max(patch_samples - 1, 1) * torch.ones_like(total_variance)
        uncertainty = torch.cat((mean_uncertainty, slope_uncertainty, logvar_uncertainty), dim=-1)
        return summary, uncertainty.clamp_min(self.eps)

    def forward(self, teacher: Mapping[str, torch.Tensor]) -> PhysicalTeacherOutput:
        direct_fields = {
            "eeg_target",
            "eeg_uncertainty",
            "fnirs_target",
            "fnirs_uncertainty",
        }
        if direct_fields.issubset(teacher):
            return self._direct_patch_targets(teacher)
        required = {
            "state_mean",
            "state_var",
            "neural_driver_eeg_rate",
            "neural_driver_var_eeg_rate",
            "teacher_valid_mask",
        }
        missing = required.difference(teacher)
        if missing:
            raise KeyError(f"Missing teacher fields: {sorted(missing)}")

        state_mean = teacher["state_mean"].detach()
        state_var = teacher["state_var"].detach()
        driver = teacher["neural_driver_eeg_rate"].detach()
        driver_var = teacher["neural_driver_var_eeg_rate"].detach()
        mask = teacher["teacher_valid_mask"].detach().bool()
        cache_mask = teacher.get("cache_valid_mask", mask).detach().bool()
        causal_mask = teacher.get("causal_valid_mask", torch.ones_like(mask)).detach().bool()
        if state_mean.shape != state_var.shape or state_mean.shape[-1] != len(self.state_names):
            raise ValueError("state_mean/state_var must share shape [B,T,5]")
        if driver.shape != driver_var.shape or driver.shape[-1] != 1:
            raise ValueError("neural driver mean/variance must share shape [B,T,1]")
        if mask.shape != state_mean.shape[:2] or cache_mask.shape != mask.shape or causal_mask.shape != mask.shape:
            raise ValueError("teacher/cache/causal validity masks must have shape [B,T]")

        state_summary, state_uncertainty = self._summarize(
            state_mean, state_var, self.fnirs_sample_rate_hz, self.fnirs_patch_samples
        )
        driver_summary, driver_uncertainty = self._summarize(
            driver, driver_var, self.eeg_sample_rate_hz, self.eeg_patch_samples
        )
        def patch_mask(name: str, fallback: torch.Tensor) -> torch.Tensor:
            value = teacher.get(name, fallback).detach().bool()
            if value.shape != mask.shape:
                raise ValueError(f"{name} must have shape [B,T]")
            return value.reshape(value.shape[0], -1, self.fnirs_patch_samples).all(dim=-1)

        mask_patch = patch_mask("teacher_valid_mask", mask)
        cache_mask_patch = patch_mask("cache_valid_mask", cache_mask)
        causal_mask_patch = patch_mask("causal_valid_mask", causal_mask)
        finite_patch = torch.isfinite(state_summary).all(dim=-1) & torch.isfinite(state_uncertainty).all(dim=-1)
        finite_patch &= torch.isfinite(driver_summary).all(dim=-1) & torch.isfinite(driver_uncertainty).all(dim=-1)
        # Local state/prototype supervision needs only a valid physical
        # posterior.  The fixed-history context objective additionally needs
        # the crop-boundary causal history.  `mask_patch` remains part of the
        # context mask for compatibility with caches that already combine the
        # two conditions.
        valid_mask = cache_mask_patch & finite_patch
        context_valid_mask = mask_patch & cache_mask_patch & causal_mask_patch & finite_patch

        entry_masks: Dict[str, Dict[str, torch.Tensor]] = {}
        for modality in ("eeg", "fnirs"):
            local = patch_mask(
                f"{modality}_local_valid_mask",
                teacher.get("local_valid_mask", cache_mask),
            ) & finite_patch
            prototype = patch_mask(
                f"{modality}_prototype_valid_mask",
                teacher.get("prototype_valid_mask", cache_mask),
            ) & finite_patch
            context = patch_mask(
                f"{modality}_context_valid_mask",
                teacher.get("context_valid_mask", mask & cache_mask & causal_mask),
            ) & finite_patch
            coupling = patch_mask(
                f"{modality}_coupling_valid_mask",
                teacher.get("coupling_valid_mask", mask & cache_mask & causal_mask),
            ) & finite_patch
            entry_masks[modality] = {
                "local": local.detach(),
                "prototype": prototype.detach(),
                "context": context.detach(),
                "coupling": coupling.detach(),
            }

        # Summary layout is statistic-major: [all means, all slopes, all log variances].
        s_indices = torch.tensor([0, 5, 10], device=state_mean.device)
        fnirs_indices = torch.tensor([1, 2, 3, 6, 7, 8, 11, 12, 13], device=state_mean.device)
        eeg_target = torch.cat((driver_summary, state_summary.index_select(-1, s_indices)), dim=-1)
        eeg_uncertainty = torch.cat((driver_uncertainty, state_uncertainty.index_select(-1, s_indices)), dim=-1)
        fnirs_target = state_summary.index_select(-1, fnirs_indices)
        fnirs_uncertainty = state_uncertainty.index_select(-1, fnirs_indices)

        return PhysicalTeacherOutput(
            full_summary=state_summary.detach(),
            full_uncertainty=state_uncertainty.detach(),
            eeg_target=eeg_target.detach(),
            eeg_uncertainty=eeg_uncertainty.detach(),
            fnirs_target=fnirs_target.detach(),
            fnirs_uncertainty=fnirs_uncertainty.detach(),
            valid_mask=valid_mask.detach(),
            context_valid_mask=context_valid_mask.detach(),
            entry_masks=entry_masks,
            target_family=self.target_family,
            target_version=self.target_version,
        )

    def _direct_patch_targets(
        self,
        teacher: Mapping[str, torch.Tensor],
    ) -> PhysicalTeacherOutput:
        """Adapt a family-versioned sidecar that already uses the token grid."""

        eeg_target = teacher["eeg_target"].detach()
        eeg_uncertainty = teacher["eeg_uncertainty"].detach()
        fnirs_target = teacher["fnirs_target"].detach()
        fnirs_uncertainty = teacher["fnirs_uncertainty"].detach()
        if eeg_target.ndim != 3 or eeg_target.shape[-1] != 6:
            raise ValueError("Direct EEG targets must have shape [B,N,6]")
        if fnirs_target.ndim != 3 or fnirs_target.shape[-1] != 9:
            raise ValueError("Direct fNIRS targets must have shape [B,N,9]")
        if eeg_target.shape[:2] != fnirs_target.shape[:2]:
            raise ValueError("Direct EEG/fNIRS targets must share the token grid")
        if eeg_uncertainty.shape != eeg_target.shape:
            raise ValueError("Direct EEG target uncertainty shape mismatch")
        if fnirs_uncertainty.shape != fnirs_target.shape:
            raise ValueError("Direct fNIRS target uncertainty shape mismatch")

        finite_eeg = torch.isfinite(eeg_target).all(dim=-1)
        finite_eeg &= torch.isfinite(eeg_uncertainty).all(dim=-1)
        finite_fnirs = torch.isfinite(fnirs_target).all(dim=-1)
        finite_fnirs &= torch.isfinite(fnirs_uncertainty).all(dim=-1)
        entry_masks: Dict[str, Dict[str, torch.Tensor]] = {"eeg": {}, "fnirs": {}}
        for modality, finite in (("eeg", finite_eeg), ("fnirs", finite_fnirs)):
            for entry in ("local", "prototype", "context", "coupling"):
                name = f"{modality}_{entry}_valid_mask"
                if name not in teacher:
                    raise KeyError(f"Direct target sidecar is missing {name}")
                mask = teacher[name].detach().bool()
                if mask.shape != finite.shape:
                    raise ValueError(f"{name} must have shape [B,N]")
                entry_masks[modality][entry] = (mask & finite).detach()

        valid = entry_masks["eeg"]["local"] & entry_masks["fnirs"]["local"]
        context_valid = (
            entry_masks["eeg"]["context"] & entry_masks["fnirs"]["context"]
        )
        return PhysicalTeacherOutput(
            full_summary=torch.cat((eeg_target, fnirs_target), dim=-1).detach(),
            full_uncertainty=torch.cat((eeg_uncertainty, fnirs_uncertainty), dim=-1).detach(),
            eeg_target=eeg_target.detach(),
            eeg_uncertainty=eeg_uncertainty.clamp_min(self.eps).detach(),
            fnirs_target=fnirs_target.detach(),
            fnirs_uncertainty=fnirs_uncertainty.clamp_min(self.eps).detach(),
            valid_mask=valid.detach(),
            context_valid_mask=context_valid.detach(),
            entry_masks=entry_masks,
            target_family=self.target_family,
            target_version=self.target_version,
        )


__all__ = ["PhysicalStateTeacher", "PhysicalTeacherOutput"]
