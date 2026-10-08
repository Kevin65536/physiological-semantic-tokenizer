"""Correct count-and-sum EMA vector quantization for physiology-semantic tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantizerOutput:
    """Complete quantizer output used by training, export, and diagnostics."""

    logits: torch.Tensor
    posterior: torch.Tensor
    hard_ids: torch.Tensor
    quantized: torch.Tensor
    annealed_quantized: torch.Tensor
    expected_embedding: torch.Tensor
    codebook: torch.Tensor
    commitment_loss: torch.Tensor
    health: Dict[str, torch.Tensor]


class EMAVectorQuantizer(nn.Module):
    """Euclidean VQ with independently maintained count and vector-sum EMA."""

    def __init__(
        self,
        codebook_size: int = 128,
        embedding_dim: int = 64,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_cost: float = 0.25,
        temperature: float = 1.0,
        assignment: str = "euclidean",
        normalize_latents: bool = False,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        revive_dead_codes: bool = False,
        dead_code_threshold: float = 0.1,
        revival_warmup_steps: int = 100,
        revival_interval: int = 100,
        revival_stop_after_steps: int | None = None,
        max_revivals_per_event: int = 8,
        revival_noise_std: float = 0.0,
        revival_strategy: str = "top_error",
        revival_count_prior: str = "threshold",
    ) -> None:
        super().__init__()
        if codebook_size <= 0 or embedding_dim <= 0:
            raise ValueError("codebook_size and embedding_dim must be positive")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if assignment not in {"euclidean", "cosine"}:
            raise ValueError("assignment must be 'euclidean' or 'cosine'")

        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim)
        self.decay = float(decay)
        self.eps = float(eps)
        self.commitment_cost = float(commitment_cost)
        self.temperature = float(temperature)
        self.assignment = assignment
        self.normalize_latents = bool(normalize_latents)
        self.kmeans_init = bool(kmeans_init)
        self.kmeans_iters = int(kmeans_iters)
        if self.kmeans_iters <= 0:
            raise ValueError("kmeans_iters must be positive")
        self.revive_dead_codes = bool(revive_dead_codes)
        self.dead_code_threshold = float(dead_code_threshold)
        self.revival_warmup_steps = int(revival_warmup_steps)
        self.revival_interval = max(int(revival_interval), 1)
        self.revival_stop_after_steps = (
            None
            if revival_stop_after_steps is None
            else int(revival_stop_after_steps)
        )
        if (
            self.revival_stop_after_steps is not None
            and self.revival_stop_after_steps < self.revival_warmup_steps
        ):
            raise ValueError(
                "revival_stop_after_steps must be at least revival_warmup_steps"
            )
        self.max_revivals_per_event = int(max_revivals_per_event)
        if self.max_revivals_per_event <= 0:
            raise ValueError("max_revivals_per_event must be positive")
        self.revival_noise_std = float(revival_noise_std)
        if self.revival_noise_std < 0.0:
            raise ValueError("revival_noise_std must be non-negative")
        if revival_strategy not in {"top_error", "diverse_farthest"}:
            raise ValueError(
                "revival_strategy must be 'top_error' or 'diverse_farthest'"
            )
        self.revival_strategy = revival_strategy
        if revival_count_prior not in {"threshold", "uniform_batch"}:
            raise ValueError(
                "revival_count_prior must be 'threshold' or 'uniform_batch'"
            )
        self.revival_count_prior = revival_count_prior

        scale = embedding_dim ** -0.5
        codebook = torch.empty(codebook_size, embedding_dim).uniform_(-scale, scale)
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_count", torch.zeros(codebook_size))
        # Count and sum must describe the same prior mass.  A random sum paired
        # with a zero count makes the first assigned update scale the random
        # initialization by roughly decay / ((1 - decay) * batch_count).
        self.register_buffer("ema_sum", torch.zeros_like(codebook))
        self.register_buffer("initialized", torch.tensor(not self.kmeans_init, dtype=torch.bool))
        self.register_buffer("quantization_strength", torch.ones(()))
        self.register_buffer("update_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("revival_count", torch.zeros((), dtype=torch.long))

    def get_codebook_weight(self) -> torch.Tensor:
        return self.codebook

    def get_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.codebook)

    def set_quantization_strength(self, strength: float) -> None:
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("quantization_strength must be in [0, 1]")
        self.quantization_strength.fill_(float(strength))

    def get_quantization_strength(self) -> float:
        return float(self.quantization_strength.item())

    def _assignment_logits(
        self,
        flat: torch.Tensor,
        codebook: torch.Tensor | None = None,
    ) -> torch.Tensor:
        codebook = self.codebook if codebook is None else codebook
        if self.assignment == "cosine":
            flat_norm = F.normalize(flat, dim=-1)
            codebook_norm = F.normalize(codebook, dim=-1)
            return flat_norm @ codebook_norm.t() / self.temperature
        distances = (
            flat.square().sum(dim=-1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.square().sum(dim=-1).unsqueeze(0)
        )
        return -distances / self.temperature

    @staticmethod
    def _distributed_sum(value: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            value = value.clone()
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    @torch.no_grad()
    def _kmeans_initialize(self, flat: torch.Tensor) -> None:
        """Initialize count, sum, and centroids from the first valid latent batch."""

        if bool(self.initialized.item()):
            return
        if dist.is_available() and dist.is_initialized():
            raise RuntimeError(
                "kmeans_init currently requires a single-process E1 calibration run"
            )
        if flat.shape[0] == 0:
            raise ValueError("kmeans_init requires at least one valid latent")
        samples = flat.to(device=self.codebook.device, dtype=self.codebook.dtype)
        if self.normalize_latents:
            samples = F.normalize(samples, dim=-1)
        sample_count = int(samples.shape[0])
        if sample_count >= self.codebook_size:
            indices = torch.randperm(sample_count, device=samples.device)[: self.codebook_size]
        else:
            indices = torch.randint(
                sample_count, (self.codebook_size,), device=samples.device
            )
        centers = samples[indices].clone()

        for _ in range(self.kmeans_iters):
            assignments = self._assignment_logits(samples, centers).argmax(dim=-1)
            counts = torch.bincount(assignments, minlength=self.codebook_size).to(samples.dtype)
            sums = samples.new_zeros(self.codebook_size, self.embedding_dim)
            sums.index_add_(0, assignments, samples)
            nonempty = counts > 0
            centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(-1)
            if self.assignment == "cosine":
                centers[nonempty] = F.normalize(centers[nonempty], dim=-1)

        assignments = self._assignment_logits(samples, centers).argmax(dim=-1)
        counts = torch.bincount(assignments, minlength=self.codebook_size).to(samples.dtype)
        sums = samples.new_zeros(self.codebook_size, self.embedding_dim)
        sums.index_add_(0, assignments, samples)
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(-1)
        if self.assignment == "cosine":
            centers[nonempty] = F.normalize(centers[nonempty], dim=-1)

        if self.normalize_latents:
            centers = F.normalize(centers, dim=-1)
        self.codebook.copy_(centers)
        self.ema_count.copy_(counts)
        self.ema_sum.copy_(centers * counts.unsqueeze(-1))
        self.initialized.fill_(True)

    @torch.no_grad()
    def _select_revival_samples(
        self,
        flat: torch.Tensor,
        hard_ids: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        """Select high-error replacements, optionally spreading them over the batch."""

        assigned_codebook = self.codebook[hard_ids]
        quantization_error = (flat - assigned_codebook).square().sum(dim=-1)
        if self.revival_strategy == "top_error" or count <= 1:
            return torch.topk(
                quantization_error, k=count, largest=True, sorted=True
            ).indices

        # Greedy farthest-point sampling starts at the highest-error latent and
        # then maximizes distance to both its assigned live prototype and every
        # already selected replacement. This keeps the useful high-error bias
        # while avoiding a dense stack of revived codes in one local cluster.
        min_distance = quantization_error.clone()
        selected = []
        unavailable = torch.zeros(flat.shape[0], device=flat.device, dtype=torch.bool)
        for _ in range(count):
            index = min_distance.masked_fill(unavailable, -torch.inf).argmax()
            selected.append(index)
            unavailable[index] = True
            distance = (flat - flat[index]).square().sum(dim=-1)
            min_distance = torch.minimum(min_distance, distance)
        return torch.stack(selected)

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, hard_ids: torch.Tensor) -> torch.Tensor:
        assignments = F.one_hot(hard_ids, self.codebook_size).to(flat.dtype)
        batch_count = self._distributed_sum(assignments.sum(dim=0))
        batch_sum = self._distributed_sum(assignments.t() @ flat)
        assigned = batch_count > 0

        # Decay every statistic, including zero-assignment codes. Count and sum
        # decay by the same factor, so an inactive centroid does not move, while
        # its occupancy evidence ages and can eventually trigger revival.
        self.ema_count.mul_(self.decay).add_(batch_count, alpha=1.0 - self.decay)
        self.ema_sum.mul_(self.decay).add_(batch_sum, alpha=1.0 - self.decay)
        updated = self.ema_sum[assigned] / self.ema_count[assigned].unsqueeze(-1).clamp_min(self.eps)
        if self.normalize_latents:
            updated = F.normalize(updated, dim=-1)
        self.codebook[assigned] = updated
        self.update_count.add_(1)

        revived = torch.zeros((), device=flat.device, dtype=torch.long)
        if (
            self.revive_dead_codes
            and int(self.update_count.item()) >= self.revival_warmup_steps
            and int(self.update_count.item()) % self.revival_interval == 0
            and (
                self.revival_stop_after_steps is None
                or int(self.update_count.item()) <= self.revival_stop_after_steps
            )
            and flat.shape[0] > 0
        ):
            dead = torch.where(self.ema_count < self.dead_code_threshold)[0]
            if dead.numel() > 0:
                revival_count = min(
                    int(dead.numel()),
                    int(flat.shape[0]),
                    self.max_revivals_per_event,
                )
                dead = dead[:revival_count]
                sample_idx = self._select_revival_samples(
                    flat, hard_ids, revival_count
                )
                replacements = flat[sample_idx].to(
                    device=self.codebook.device, dtype=self.codebook.dtype
                )
                if self.revival_noise_std > 0.0:
                    replacements = replacements + torch.randn_like(replacements) * self.revival_noise_std
                if self.normalize_latents:
                    replacements = F.normalize(replacements, dim=-1)
                self.codebook[dead] = replacements
                revival_prior = max(self.dead_code_threshold, self.eps)
                if self.revival_count_prior == "uniform_batch":
                    revival_prior = max(
                        revival_prior,
                        float(flat.shape[0]) / float(self.codebook_size),
                    )
                self.ema_count[dead] = revival_prior
                self.ema_sum[dead] = replacements * self.ema_count[dead].unsqueeze(-1)
                revived = torch.tensor(revival_count, device=flat.device, dtype=torch.long)
                self.revival_count.add_(revived.to(self.revival_count.device))
        return revived

    @torch.no_grad()
    def _health(self, hard_ids: torch.Tensor, drift: torch.Tensor, revived: torch.Tensor) -> Dict[str, torch.Tensor]:
        counts = torch.bincount(hard_ids, minlength=self.codebook_size).float()
        probabilities = counts / counts.sum().clamp_min(1.0)
        nonzero = probabilities > 0
        entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
        effective_codes = entropy.exp()

        normalized = F.normalize(self.codebook, dim=-1)
        cosine = normalized @ normalized.t()
        cosine.fill_diagonal_(-1.0)
        nearest_cosine = cosine.max(dim=1).values.mean()
        effective_rank = torch.linalg.matrix_rank(self.codebook.float()).to(self.codebook.dtype)
        return {
            "assignment_entropy": entropy,
            "effective_codes": effective_codes,
            "batch_active_codes": (counts > 0).sum().to(self.codebook.dtype),
            "ema_active_fraction": (self.ema_count >= self.dead_code_threshold).float().mean(),
            "effective_rank": effective_rank,
            "nearest_neighbor_cosine": nearest_cosine,
            "prototype_drift": drift,
            "initialized": self.initialized.to(self.codebook.dtype),
            "quantization_strength": self.quantization_strength.to(self.codebook.dtype),
            "revived_codes": revived.to(self.codebook.dtype),
            "total_revivals": self.revival_count.to(self.codebook.dtype),
        }

    def forward(
        self,
        latent: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> QuantizerOutput:
        if latent.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected latent dimension {self.embedding_dim}, got {latent.shape[-1]}"
            )
        original_shape = latent.shape[:-1]
        flat = latent.reshape(-1, self.embedding_dim)
        quantization_input = (
            F.normalize(flat, dim=-1) if self.normalize_latents else flat
        )
        if valid_mask is None:
            valid_flat = torch.ones(flat.shape[0], device=flat.device, dtype=torch.bool)
        else:
            if tuple(valid_mask.shape) != tuple(original_shape):
                raise ValueError(
                    f"valid_mask shape {tuple(valid_mask.shape)} does not match latent tokens {tuple(original_shape)}"
                )
            valid_flat = valid_mask.reshape(-1).to(device=flat.device, dtype=torch.bool)
        if self.training and self.kmeans_init and not bool(self.initialized.item()):
            self._kmeans_initialize(quantization_input.detach()[valid_flat])
        # EMA updates prepare the codebook for the next batch.  This batch's
        # differentiable posterior must retain an immutable pre-update view;
        # otherwise the in-place buffer update invalidates autograd's version.
        forward_codebook = self.codebook.detach().clone()
        logits_flat = self._assignment_logits(quantization_input, forward_codebook)
        posterior_flat = logits_flat.softmax(dim=-1)
        hard_flat = posterior_flat.argmax(dim=-1)
        lookup_flat = F.embedding(hard_flat, forward_codebook)
        expected_flat = posterior_flat @ forward_codebook
        if valid_flat.any():
            commitment_loss = (
                self.quantization_strength
                * self.commitment_cost
                * F.mse_loss(
                quantization_input[valid_flat], lookup_flat.detach()[valid_flat]
                )
            )
        else:
            commitment_loss = flat.sum() * 0.0

        previous = self.codebook.detach().clone()
        revived = torch.zeros((), device=flat.device, dtype=torch.long)
        if self.training:
            revived = self._ema_update(
                quantization_input.detach()[valid_flat], hard_flat.detach()[valid_flat]
            )
        drift = (self.codebook - previous).norm(dim=-1).mean()

        quantized_flat = quantization_input + (
            lookup_flat - quantization_input
        ).detach()
        annealed_quantized_flat = quantization_input + self.quantization_strength * (
            lookup_flat - quantization_input
        ).detach()
        health = self._health(
            hard_flat.detach()[valid_flat], drift.detach(), revived.detach()
        )
        return QuantizerOutput(
            logits=logits_flat.reshape(*original_shape, self.codebook_size),
            posterior=posterior_flat.reshape(*original_shape, self.codebook_size),
            hard_ids=hard_flat.reshape(*original_shape),
            quantized=quantized_flat.reshape(*original_shape, self.embedding_dim),
            annealed_quantized=annealed_quantized_flat.reshape(
                *original_shape, self.embedding_dim
            ),
            expected_embedding=expected_flat.reshape(*original_shape, self.embedding_dim),
            codebook=self.codebook,
            commitment_loss=commitment_loss,
            health=health,
        )


__all__ = ["EMAVectorQuantizer", "QuantizerOutput"]
