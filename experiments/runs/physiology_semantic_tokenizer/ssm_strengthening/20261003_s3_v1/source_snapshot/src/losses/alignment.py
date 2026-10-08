"""Alignment-specific loss functions shared by multimodal tokenizers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """Force aligned latent representations from two modalities to stay close."""

    def __init__(self):
        super().__init__()

    def forward(self, latent_a: torch.Tensor, latent_b: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(latent_a, latent_b)


__all__ = ['AlignmentLoss']