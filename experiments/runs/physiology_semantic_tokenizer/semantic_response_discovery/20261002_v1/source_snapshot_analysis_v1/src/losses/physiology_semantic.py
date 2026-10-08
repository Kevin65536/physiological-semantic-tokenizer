"""Mask-aware objectives for physiology-semantic tokenization."""

from __future__ import annotations

import math
from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.teachers.physical_state_teacher import PhysicalTeacherOutput
from src.tokenizers.physiology_semantic_tokenizer import ModalityTokenizerOutput


def straight_through_codebook_balance_loss(
    logits: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize low hard-ID marginal entropy with soft-assignment gradients."""

    if logits.ndim < 2:
        raise ValueError("Quantizer logits must include token and codebook dimensions")
    if temperature <= 0.0:
        raise ValueError("Balance temperature must be positive")
    codebook_size = int(logits.shape[-1])
    flat = logits.reshape(-1, codebook_size)
    if valid_mask is not None:
        if tuple(valid_mask.shape) != tuple(logits.shape[:-1]):
            raise ValueError("Balance validity mask must match quantizer token dimensions")
        flat = flat[valid_mask.reshape(-1).to(device=logits.device, dtype=torch.bool)]
    if flat.shape[0] == 0 or codebook_size <= 1:
        return logits.sum() * 0.0

    soft = F.softmax(flat.float() / float(temperature), dim=-1)
    hard = F.one_hot(soft.argmax(dim=-1), num_classes=codebook_size).to(soft.dtype)
    assignments = hard + soft - soft.detach()
    # Additive smoothing keeps the hard-ID forward statistic while preserving
    # gradients for currently unused codes. clamp_min would give every dead
    # entry a zero local derivative, making a near-one collapse penalty unable
    # to repopulate the codebook.
    marginal = assignments.mean(dim=0) + float(eps)
    marginal = marginal / marginal.sum().clamp_min(float(eps))
    entropy = -(marginal * marginal.log()).sum()
    return 1.0 - entropy / math.log(float(codebook_size))


class PhysiologySemanticLoss(nn.Module):
    def __init__(
        self,
        state_weight: float = 1.0,
        prototype_weight: float = 1.0,
        masked_state_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        vq_weight: float = 1.0,
        private_weight: float = 0.0,
        balance_weight: float = 0.0,
        reconstruction_mode: str = "combined",
        reconstruction_semantic_input: str = "expected",
        balance_temperature: float = 1.0,
        eeg_balance_temperature: float | None = None,
        fnirs_balance_temperature: float | None = None,
        eeg_balance_scale: float = 1.0,
        fnirs_balance_scale: float = 1.0,
        eeg_coordinate_mask: torch.Tensor | None = None,
        fnirs_coordinate_mask: torch.Tensor | None = None,
        eeg_entry_coordinate_masks: Mapping[str, torch.Tensor] | None = None,
        fnirs_entry_coordinate_masks: Mapping[str, torch.Tensor] | None = None,
        uncertainty_weighting: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weights = {
            "state": float(state_weight),
            "prototype": float(prototype_weight),
            "masked_state": float(masked_state_weight),
            "reconstruction": float(reconstruction_weight),
            "vq": float(vq_weight),
            "private": float(private_weight),
            "balance": float(balance_weight),
        }
        self.eps = float(eps)
        if reconstruction_mode not in {"combined", "semantic_only", "residual_only"}:
            raise ValueError(
                "reconstruction_mode must be combined, semantic_only, or residual_only"
            )
        self.reconstruction_mode = str(reconstruction_mode)
        if reconstruction_semantic_input not in {"expected", "hard", "annealed_hard"}:
            raise ValueError(
                "reconstruction_semantic_input must be expected, hard, or annealed_hard"
            )
        if balance_temperature <= 0.0:
            raise ValueError("balance_temperature must be positive")
        if eeg_balance_scale < 0.0 or fnirs_balance_scale < 0.0:
            raise ValueError("Modality balance scales must be non-negative")
        self.reconstruction_semantic_input = str(reconstruction_semantic_input)
        self.balance_temperature = float(balance_temperature)
        self.eeg_balance_temperature = float(
            balance_temperature
            if eeg_balance_temperature is None
            else eeg_balance_temperature
        )
        self.fnirs_balance_temperature = float(
            balance_temperature
            if fnirs_balance_temperature is None
            else fnirs_balance_temperature
        )
        if self.eeg_balance_temperature <= 0.0 or self.fnirs_balance_temperature <= 0.0:
            raise ValueError("Modality balance temperatures must be positive")
        self.eeg_balance_scale = float(eeg_balance_scale)
        self.fnirs_balance_scale = float(fnirs_balance_scale)
        self.uncertainty_weighting = bool(uncertainty_weighting)
        self._register_entry_masks(
            "eeg", 6, eeg_coordinate_mask, eeg_entry_coordinate_masks
        )
        self._register_entry_masks(
            "fnirs", 9, fnirs_coordinate_mask, fnirs_entry_coordinate_masks
        )

    def _register_entry_masks(
        self,
        modality: str,
        coordinate_count: int,
        fallback: torch.Tensor | None,
        per_entry: Mapping[str, torch.Tensor] | None,
    ) -> None:
        fallback_mask = (
            torch.ones(coordinate_count, dtype=torch.bool)
            if fallback is None
            else fallback.bool()
        )
        if fallback_mask.shape != (coordinate_count,):
            raise ValueError(
                f"{modality} coordinate mask must have shape [{coordinate_count}]"
            )
        entry_masks = dict(per_entry or {})
        for entry in ("local", "prototype", "context", "coupling"):
            mask = entry_masks.get(entry, fallback_mask).bool()
            if mask.shape != (coordinate_count,):
                raise ValueError(
                    f"{modality} {entry} coordinate mask must have shape [{coordinate_count}]"
                )
            self.register_buffer(f"{modality}_{entry}_coordinate_mask", mask)

    @property
    def requires_teacher(self) -> bool:
        return any(self.weights[name] > 0.0 for name in ("state", "prototype", "masked_state"))

    def _masked_target_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        uncertainty: torch.Tensor,
        mask: torch.Tensor,
        coordinate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape or target.shape != uncertainty.shape:
            raise ValueError("Prediction, target, and uncertainty shapes must match")
        if mask.shape != prediction.shape[:2]:
            raise ValueError("Mask must have shape [B,N]")
        if not coordinate_mask.any():
            return prediction.sum() * 0.0
        error = (prediction - target).square()
        if self.uncertainty_weighting:
            error = error / uncertainty.clamp_min(self.eps)
        per_patch = error[..., coordinate_mask].mean(dim=-1)
        mask_float = mask.to(per_patch.dtype)
        return (per_patch * mask_float).sum() / mask_float.sum().clamp_min(1.0)

    @staticmethod
    def _masked_reconstruction_loss(
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if reconstruction.shape != target.shape:
            raise ValueError("Reconstruction and patch target shapes must match")
        error = (reconstruction - target).square()
        if token_mask is None:
            return error.mean()
        if token_mask.shape == reconstruction.shape[:2]:
            mask = token_mask[:, :, None, None]
        elif token_mask.shape == (reconstruction.shape[0], reconstruction.shape[1], reconstruction.shape[3]):
            mask = token_mask[:, :, None, :]
        else:
            raise ValueError("Token validity mask must have shape [B,N] or [B,N,P]")
        mask_float = mask.to(error.dtype).expand_as(error)
        return (error * mask_float).sum() / mask_float.sum().clamp_min(1.0)

    def _selected_reconstruction(self, output: ModalityTokenizerOutput) -> torch.Tensor:
        if self.reconstruction_mode == "residual_only":
            return output.residual_reconstruction
        if self.reconstruction_semantic_input == "hard":
            return {
                "combined": output.hard_reconstruction,
                "semantic_only": output.hard_semantic_reconstruction,
            }[self.reconstruction_mode]
        if self.reconstruction_semantic_input == "annealed_hard":
            return {
                "combined": output.annealed_hard_reconstruction,
                "semantic_only": output.annealed_hard_semantic_reconstruction,
            }[self.reconstruction_mode]
        return {
            "combined": output.reconstruction,
            "semantic_only": output.semantic_reconstruction,
        }[self.reconstruction_mode]

    def _balance_loss(
        self,
        output: ModalityTokenizerOutput,
        token_mask: torch.Tensor | None,
        temperature: float,
    ) -> torch.Tensor:
        return straight_through_codebook_balance_loss(
            output.quantizer.logits,
            token_mask,
            temperature=temperature,
        )

    def _modality_losses(
        self,
        output: ModalityTokenizerOutput,
        target: torch.Tensor,
        uncertainty: torch.Tensor,
        entry_masks: Mapping[str, torch.Tensor],
        coordinate_masks: Mapping[str, torch.Tensor],
        token_mask: torch.Tensor | None,
        balance_temperature: float,
    ) -> Dict[str, torch.Tensor]:
        context_mask = entry_masks["context"] & output.context_valid_mask
        return {
            "state": self._masked_target_loss(
                output.state_prediction, target, uncertainty, entry_masks["local"], coordinate_masks["local"]
            ),
            "prototype": self._masked_target_loss(
                output.prototype_state_prediction, target, uncertainty, entry_masks["prototype"],
                coordinate_masks["prototype"]
            ),
            "masked_state": self._masked_target_loss(
                output.context_state_prediction, target, uncertainty, context_mask, coordinate_masks["context"]
            ),
            "reconstruction": self._masked_reconstruction_loss(
                self._selected_reconstruction(output), output.patches, token_mask
            ),
            "vq": output.quantizer.commitment_loss,
            "private": output.residual.square().mean() * 0.0,
            "balance": self._balance_loss(
                output,
                token_mask,
                balance_temperature,
            ),
        }

    def forward(
        self,
        outputs: Mapping[str, ModalityTokenizerOutput],
        teacher: PhysicalTeacherOutput | None,
        token_valid_masks: Mapping[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        token_valid_masks = dict(token_valid_masks or {})
        zero = outputs["eeg"].semantic_latent.sum() * 0.0
        if teacher is None:
            if self.requires_teacher:
                raise ValueError("Teacher targets are required by enabled semantic losses")
            eeg = {
                "state": zero,
                "prototype": zero,
                "masked_state": zero,
                "reconstruction": self._masked_reconstruction_loss(
                    self._selected_reconstruction(outputs["eeg"]), outputs["eeg"].patches,
                    token_valid_masks.get("eeg"),
                ),
                "vq": outputs["eeg"].quantizer.commitment_loss,
                "private": zero,
                "balance": self._balance_loss(
                    outputs["eeg"], token_valid_masks.get("eeg"), self.eeg_balance_temperature
                ),
            }
            fnirs_zero = outputs["fnirs"].semantic_latent.sum() * 0.0
            fnirs = {
                "state": fnirs_zero,
                "prototype": fnirs_zero,
                "masked_state": fnirs_zero,
                "reconstruction": self._masked_reconstruction_loss(
                    self._selected_reconstruction(outputs["fnirs"]), outputs["fnirs"].patches,
                    token_valid_masks.get("fnirs"),
                ),
                "vq": outputs["fnirs"].quantizer.commitment_loss,
                "private": fnirs_zero,
                "balance": self._balance_loss(
                    outputs["fnirs"], token_valid_masks.get("fnirs"), self.fnirs_balance_temperature
                ),
            }
        else:
            eeg = self._modality_losses(
                outputs["eeg"], teacher.eeg_target, teacher.eeg_uncertainty,
                teacher.entry_masks["eeg"],
                {
                    entry: getattr(self, f"eeg_{entry}_coordinate_mask")
                    for entry in ("local", "prototype", "context", "coupling")
                },
                token_valid_masks.get("eeg"),
                self.eeg_balance_temperature,
            )
            fnirs = self._modality_losses(
                outputs["fnirs"], teacher.fnirs_target, teacher.fnirs_uncertainty,
                teacher.entry_masks["fnirs"],
                {
                    entry: getattr(self, f"fnirs_{entry}_coordinate_mask")
                    for entry in ("local", "prototype", "context", "coupling")
                },
                token_valid_masks.get("fnirs"),
                self.fnirs_balance_temperature,
            )
        components: Dict[str, torch.Tensor] = {}
        total = zero
        for name, weight in self.weights.items():
            if name == "balance":
                component = 0.5 * (
                    self.eeg_balance_scale * eeg[name]
                    + self.fnirs_balance_scale * fnirs[name]
                )
            else:
                component = eeg[name] + fnirs[name]
            components[name] = component
            total = total + weight * component
            components[f"eeg_{name}"] = eeg[name]
            components[f"fnirs_{name}"] = fnirs[name]
        components["eeg_balance"] = eeg["balance"]
        components["fnirs_balance"] = fnirs["balance"]
        components["total"] = total
        return components


__all__ = ["PhysiologySemanticLoss", "straight_through_codebook_balance_loss"]
