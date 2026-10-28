"""Local, independently decodable SSM-component and full-residual vocabularies.

All tensors end in a flattened *local* patch dimension. Leading dimensions
identify independent patches; neither encoder nor decoder mixes those axes.
Teacher components are training targets, never inputs to a student forward.
The component-compression reference has a separate explicit input method.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from .ema_vector_quantizer import EMAVectorQuantizer


VARIANTS = (
    "continuous_student",
    "hard_student",
    "component_compression",
    "smooth_hard",
    "ordinary_hard",
)
COMPONENTS = ("S", "O")


def _mask_for(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Accept an element mask or a patch mask, without implicit axis broadcast."""
    if mask is None:
        return torch.ones_like(values, dtype=torch.bool)
    if mask.dtype != torch.bool:
        raise ValueError("masks must have boolean dtype")
    if tuple(mask.shape) == tuple(values.shape[:-1]):
        mask = mask.unsqueeze(-1).expand_as(values)
    elif tuple(mask.shape) != tuple(values.shape):
        raise ValueError("mask must match all elements or the leading patch dimensions")
    return mask.to(device=values.device)


class LocalComponentBranch(nn.Module):
    """One component's patch encoder, EMA vocabulary, and private decoder."""

    def __init__(self, input_dim: int, target_dim: int, cfg: Mapping, hard: bool):
        super().__init__()
        width = int(cfg.get("width", 128))
        latent = int(cfg.get("latent_dim", 32))
        if width <= 0 or latent <= 0:
            raise ValueError("width and latent_dim must be positive")
        self.input_dim, self.target_dim = input_dim, target_dim
        self.hard = hard
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, width), nn.GELU(), nn.Linear(width, latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, width), nn.GELU(), nn.Linear(width, target_dim)
        )
        self.quantizer = EMAVectorQuantizer(
            codebook_size=int(cfg.get("codebook_size", 128)),
            embedding_dim=latent,
            decay=float(cfg.get("ema_decay", 0.99)),
            commitment_cost=float(cfg.get("commitment_cost", 0.25)),
            kmeans_init=bool(cfg.get("kmeans_init", True)),
            kmeans_iters=int(cfg.get("kmeans_iters", 10)),
            revive_dead_codes=False,
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        if values.ndim < 2 or values.shape[-1] != self.input_dim:
            raise ValueError(f"expected local patches ending in {self.input_dim} inputs")
        if not values.is_floating_point():
            raise ValueError("patch values must use a floating dtype")
        mask = _mask_for(values, mask)
        clean = torch.where(mask, values, torch.zeros_like(values))
        if not torch.isfinite(clean).all():
            raise ValueError("nonfinite visible measurement")
        valid = mask.any(dim=-1)
        latent = self.encoder(torch.cat((clean, mask.to(values.dtype)), dim=-1))
        commitment = latent.sum() * 0.0
        ids, health = None, {}
        if self.hard and bool(valid.any()):
            if not self.training and not bool(self.quantizer.initialized.item()):
                raise RuntimeError("EMA vocabulary is uninitialized; train before evaluation")
            quantized = self.quantizer(latent, valid)
            decoded_latent = quantized.quantized
            ids = quantized.hard_ids.masked_fill(~valid, -1)
            commitment, health = quantized.commitment_loss, quantized.health
        elif self.hard:
            # An empty support must neither initialize nor age an EMA vocabulary.
            decoded_latent = latent * 0.0
            ids = torch.full(valid.shape, -1, dtype=torch.long, device=values.device)
        else:
            decoded_latent = latent
        waveform = self.decoder(decoded_latent)
        return dict(
            waveform=waveform,
            latent=latent,
            quantized=decoded_latent,
            ids=ids,
            valid=valid,
            commitment=commitment,
            health=health,
        )


class DualComponentTokenizer(nn.Module):
    """Two local branches per modality, with independent hard IDs and decoders.

    ``cfg`` declares ``input_dims`` and ``target_dims`` as modality-to-integer
    mappings. For example, 2-second native EEG inputs can have 400*6 inputs
    while the aligned 10 Hz power-proxy target has 20 outputs. Hb inputs and
    targets can each have 20*2 outputs. The model never subtracts native EEG
    voltage from a power proxy.

    The main objective is S error + lambda_O * O error + beta * commitment.
    Each error is a masked, training-scale-normalized element MSE per modality,
    then an equal mean over modalities with valid targets. ``sum`` is diagnostic
    except for ``ordinary_hard``, whose two code streams are supervised only by
    their additive observation reconstruction and have no component semantics.
    """

    def __init__(self, cfg: Mapping, variant: str = "hard_student"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.variant = variant
        self.input_dims = self._dimensions(cfg.get("input_dims"), "input_dims")
        self.target_dims = self._dimensions(cfg.get("target_dims"), "target_dims")
        if self.input_dims.keys() != self.target_dims.keys():
            raise ValueError("input_dims and target_dims must declare the same modalities")
        self.modalities = tuple(self.input_dims)
        self.hard = variant != "continuous_student"
        self.branches = nn.ModuleDict()
        for modality in self.modalities:
            inputs = (self.target_dims[modality] if variant == "component_compression"
                      else self.input_dims[modality])
            self.branches[modality] = nn.ModuleDict({
                component: LocalComponentBranch(inputs, self.target_dims[modality], cfg, self.hard)
                for component in COMPONENTS
            })

    @staticmethod
    def _dimensions(values, field: str) -> dict[str, int]:
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"{field} must be a nonempty modality-to-dimension mapping")
        if any(not isinstance(k, str) or not k or isinstance(v, bool)
               or not isinstance(v, int) or v <= 0 for k, v in values.items()):
            raise ValueError(f"{field} requires named positive integer dimensions")
        return dict(values)

    def _check_modalities(self, values: Mapping, field: str) -> None:
        if not isinstance(values, Mapping) or set(values) != set(self.modalities):
            raise ValueError(f"{field} must contain exactly {self.modalities}")

    def _forward(self, sources: Mapping, masks: Mapping | None) -> dict:
        self._check_modalities(sources, "inputs")
        if masks is not None:
            self._check_modalities(masks, "input_masks")
        result = {}
        for modality in self.modalities:
            source = sources[modality]
            component_results = {}
            for component in COMPONENTS:
                values = source[component]
                mask = None if masks is None else masks[modality][component]
                component_results[component] = self.branches[modality][component](values, mask)
            source_result, residual_result = (component_results[c] for c in COMPONENTS)
            if source_result["waveform"].shape != residual_result["waveform"].shape:
                raise ValueError("S and O must have identical patch support")
            result[modality] = dict(
                S=source_result["waveform"], O=residual_result["waveform"],
                sum=source_result["waveform"] + residual_result["waveform"],
                ids_S=source_result["ids"], ids_O=residual_result["ids"],
                latent_S=source_result["latent"], latent_O=residual_result["latent"],
                quantized_S=source_result["quantized"], quantized_O=residual_result["quantized"],
                valid_S=source_result["valid"], valid_O=residual_result["valid"],
                commitment=source_result["commitment"] + residual_result["commitment"],
                health={c: component_results[c]["health"] for c in COMPONENTS},
            )
        return result

    def forward(self, inputs: Mapping, input_masks: Mapping | None = None) -> dict:
        """Encode deployable observations only; there is no teacher argument."""
        if self.variant == "component_compression":
            raise ValueError("component_compression requires forward_components")
        self._check_modalities(inputs, "inputs")
        if input_masks is not None:
            self._check_modalities(input_masks, "input_masks")
        return self._forward(
            {m: {c: inputs[m] for c in COMPONENTS} for m in self.modalities},
            None if input_masks is None else {
                m: {c: input_masks[m] for c in COMPONENTS} for m in self.modalities
            },
        )

    def forward_components(self, physical: Mapping, remaining: Mapping,
                           target_masks: Mapping | None = None) -> dict:
        """Compression reference: S reads physical; O reads fixed full residual."""
        if self.variant != "component_compression":
            raise ValueError("teacher components are inputs only in component_compression")
        self._check_modalities(physical, "physical")
        self._check_modalities(remaining, "remaining")
        if target_masks is not None:
            self._check_modalities(target_masks, "target_masks")
        sources, masks = {}, {}
        for modality in self.modalities:
            if physical[modality].shape != remaining[modality].shape:
                raise ValueError("physical and remaining must have identical support")
            common = (_mask_for(physical[modality], None if target_masks is None
                               else target_masks[modality])
                      & torch.isfinite(physical[modality]) & torch.isfinite(remaining[modality]))
            sources[modality] = {"S": physical[modality], "O": remaining[modality]}
            masks[modality] = {c: common for c in COMPONENTS}
        return self._forward(sources, masks)

    def objective(self, output: Mapping, observed: Mapping, physical: Mapping | None = None,
                  target_masks: Mapping | None = None, target_scales: Mapping | None = None,
                  lambda_O: float = 1.0, beta: float = 1.0) -> dict:
        """Score aligned target-space observations; O is always observed-physical.

        Scales must be fixed from training data by the caller. A scalar, one
        value per coordinate, or an explicitly broadcastable per-record scale
        is accepted; scales are never fitted here.
        Nonfinite labels and unsupported predictions are excluded before error
        arithmetic. Targets are detached so O cannot track student S errors.
        """
        self._check_modalities(output, "output")
        self._check_modalities(observed, "observed")
        ordinary = self.variant == "ordinary_hard"
        if not ordinary:
            self._check_modalities(physical, "physical")
        if target_masks is not None:
            self._check_modalities(target_masks, "target_masks")
        if target_scales is not None:
            self._check_modalities(target_scales, "target_scales")
        if not math.isfinite(lambda_O) or not math.isfinite(beta) or min(lambda_O, beta) < 0:
            raise ValueError("objective weights must be finite and nonnegative")
        per_modality, active = {}, []
        reference = output[self.modalities[0]]["sum"]
        zero = reference.reshape(-1)[:0].sum()
        for modality in self.modalities:
            prediction, target = output[modality], observed[modality].detach()
            if target.shape != prediction["sum"].shape or target.shape[-1] != self.target_dims[modality]:
                raise ValueError("observed targets must match decoded target-space patches")
            mask = _mask_for(target, None if target_masks is None else target_masks[modality])
            mask = (mask & torch.isfinite(target)
                    & prediction["valid_S"].unsqueeze(-1)
                    & prediction["valid_O"].unsqueeze(-1))
            physical_target = None
            if not ordinary:
                physical_target = physical[modality].detach()
                if physical_target.shape != target.shape:
                    raise ValueError("physical must match observations in units and tensor support")
                mask = mask & torch.isfinite(physical_target)
            scale = torch.as_tensor(1.0 if target_scales is None else target_scales[modality],
                                    device=target.device, dtype=target.dtype).detach()
            try:
                scale = torch.broadcast_to(scale, target.shape)
            except RuntimeError as exc:
                raise ValueError("training scale must broadcast to one value per target element") from exc
            if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
                raise ValueError("training scale must be positive and finite")
            clean_target = torch.where(mask, target, torch.zeros_like(target))

            def mse(estimate, truth):
                clean_estimate = torch.where(mask, estimate, torch.zeros_like(estimate))
                if not bool(torch.isfinite(clean_estimate).all()):
                    raise ValueError("nonfinite prediction on valid support")
                error = ((clean_estimate - truth) / scale).square()
                return error.sum() / mask.sum().clamp_min(1)

            terms = {"sum": mse(prediction["sum"], clean_target),
                     "vq": prediction["commitment"], "valid_elements": mask.sum()}
            if ordinary:
                terms.update(S=zero, O=zero)
            else:
                clean_physical = torch.where(mask, physical_target, torch.zeros_like(physical_target))
                terms.update(S=mse(prediction["S"], clean_physical),
                             O=mse(prediction["O"], clean_target - clean_physical))
            per_modality[modality] = terms
            if bool(mask.any()):
                active.append(terms)
        terms = {name: (torch.stack([m[name] for m in active]).mean() if active else zero)
                 for name in ("S", "O", "sum", "vq")}
        terms["total"] = ((terms["sum"] if ordinary else terms["S"] + lambda_O * terms["O"])
                          + beta * terms["vq"])
        terms["per_modality"] = per_modality
        return terms

    def _require_hard(self) -> None:
        if not self.hard:
            raise ValueError("continuous_student has no deployed hard vocabulary")
        for branches in self.branches.values():
            for branch in branches.values():
                if not bool(branch.quantizer.initialized.item()):
                    raise RuntimeError("EMA vocabulary is uninitialized")

    def decode_ids(self, ids: Mapping) -> dict:
        """Integer-only consumer: reject sidecars, invalid IDs, and soft values."""
        self._require_hard()
        self._check_modalities(ids, "ids")
        result = {}
        for modality in self.modalities:
            if not isinstance(ids[modality], Mapping) or set(ids[modality]) != set(COMPONENTS):
                raise ValueError("each modality must contain only S and O integer IDs")
            decoded, shape = {}, None
            for component in COMPONENTS:
                value = ids[modality][component]
                if not isinstance(value, torch.Tensor) or value.dtype not in (
                    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
                ):
                    raise ValueError("integer tensor IDs required")
                branch = self.branches[modality][component]
                if bool((value < 0).any()) or bool((value >= branch.quantizer.codebook_size).any()):
                    raise ValueError("IDs must lie inside the fixed vocabulary")
                if shape is not None and shape != value.shape:
                    raise ValueError("S and O ID support must match")
                shape = value.shape
                indices = value.to(device=branch.quantizer.codebook.device, dtype=torch.long)
                decoded[component] = branch.decoder(branch.quantizer.get_embedding(indices))
            result[modality] = {**decoded, "sum": decoded["S"] + decoded["O"]}
        return result

    def decoded_codewords(self) -> dict:
        """Each fixed code's decoded waveform, without any sample-specific input."""
        self._require_hard()
        return {m: {c: self.branches[m][c].decoder(self.branches[m][c].quantizer.codebook)
                    for c in COMPONENTS} for m in self.modalities}

    def expected_waveforms(self, probabilities: Mapping) -> dict:
        """Compute p @ D(codewords), never D(p @ codewords)."""
        self._check_modalities(probabilities, "probabilities")
        dictionaries = self.decoded_codewords()
        result = {}
        for modality in self.modalities:
            if (not isinstance(probabilities[modality], Mapping)
                    or set(probabilities[modality]) != set(COMPONENTS)):
                raise ValueError("each modality needs S and O distributions")
            decoded, shape = {}, None
            for component in COMPONENTS:
                p = probabilities[modality][component]
                words = dictionaries[modality][component]
                if not isinstance(p, torch.Tensor) or not p.is_floating_point():
                    raise ValueError("floating probability tensors required")
                if p.ndim < 1 or p.shape[-1] != words.shape[0]:
                    raise ValueError("probability axis must match vocabulary size")
                if (not bool(torch.isfinite(p).all()) or bool((p < 0).any())
                        or not torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-5, rtol=1e-5)):
                    raise ValueError("finite normalized nonnegative probabilities required")
                if shape is not None and shape != p.shape[:-1]:
                    raise ValueError("S and O probability support must match")
                shape = p.shape[:-1]
                decoded[component] = p.to(device=words.device, dtype=words.dtype) @ words
            result[modality] = {**decoded, "sum": decoded["S"] + decoded["O"]}
        return result

    def nominal_bits_per_patch(self) -> dict[str, float]:
        """Total S+O code capacity per modality, with no continuous sidecars."""
        if not self.hard:
            raise ValueError("continuous_student has no deployed hard vocabulary")
        return {m: sum(math.log2(self.branches[m][c].quantizer.codebook_size)
                       for c in COMPONENTS) for m in self.modalities}


__all__ = ["DualComponentTokenizer", "LocalComponentBranch", "VARIANTS"]
