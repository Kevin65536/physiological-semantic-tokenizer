"""Shared-driver semantic tokenizer components for the R-series architecture.

This module starts with the R2 continuous observability model.  It deliberately
contains no vector quantizer, raw-signal decoder, cross-modal module, or
coupling objective.  EEG and fNIRS are encoded by separate full-window
encoders, while a single modality-agnostic decoder maps either latent sequence
to the registered shared-driver trajectory coordinate.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class TemporalPatchStem(nn.Module):
    """Project non-overlapping multichannel temporal patches into token space."""

    def __init__(
        self,
        input_channels: int,
        patch_samples: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if patch_samples <= 0:
            raise ValueError("patch_samples must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.input_channels = int(input_channels)
        self.patch_samples = int(patch_samples)
        self.embedding_dim = int(embedding_dim)
        self.projection = nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=self.embedding_dim,
            kernel_size=self.patch_samples,
            stride=self.patch_samples,
        )
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected [B,{self.input_channels},T] signal, got {tuple(signal.shape)}"
            )
        if signal.shape[-1] % self.patch_samples != 0:
            raise ValueError(
                f"Signal length {signal.shape[-1]} is not divisible by "
                f"patch_samples={self.patch_samples}"
            )

        tokens = self.projection(signal).transpose(1, 2)
        return self.norm(self.activation(tokens))


class FullWindowModalityEncoder(nn.Module):
    """Encode one measured modality with bidirectional full-window context.

    ``src_key_padding_mask`` excludes invalid patches from attention keys and
    values.  Rows without any valid patch bypass the Transformer entirely, so
    the implementation never has to expose an invalid sentinel key merely to
    avoid an all-masked softmax.
    """

    token_temporal_scope = "bidirectional_full_window"

    def __init__(
        self,
        input_channels: int,
        patch_samples: int,
        num_tokens: int = 10,
        latent_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if num_heads <= 0 or latent_dim % num_heads != 0:
            raise ValueError("num_heads must be positive and divide latent_dim")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_channels = int(input_channels)
        self.patch_samples = int(patch_samples)
        self.num_tokens = int(num_tokens)
        self.latent_dim = int(latent_dim)
        self.expected_samples = self.num_tokens * self.patch_samples

        self.patch_stem = TemporalPatchStem(
            input_channels=self.input_channels,
            patch_samples=self.patch_samples,
            embedding_dim=self.latent_dim,
        )
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.num_tokens, self.latent_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=int(num_heads),
            dim_feedforward=int(feedforward_dim or self.latent_dim * 4),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.full_window_encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(depth),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)

    def resolve_valid_mask(
        self,
        signal: torch.Tensor,
        token_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = signal.shape[0]
        if token_valid_mask is None:
            return torch.ones(
                batch_size,
                self.num_tokens,
                dtype=torch.bool,
                device=signal.device,
            )
        if tuple(token_valid_mask.shape) != (batch_size, self.num_tokens):
            raise ValueError(
                "token_valid_mask must have shape "
                f"[B,{self.num_tokens}], got {tuple(token_valid_mask.shape)}"
            )
        return token_valid_mask.to(device=signal.device, dtype=torch.bool)

    def forward(
        self,
        signal: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected [B,{self.input_channels},{self.expected_samples}] signal, "
                f"got {tuple(signal.shape)}"
            )
        if signal.shape[-1] != self.expected_samples:
            raise ValueError(
                f"Expected exactly {self.expected_samples} samples "
                f"({self.num_tokens} patches), got {signal.shape[-1]}"
            )

        valid_mask = self.resolve_valid_mask(signal, token_valid_mask)
        tokens = self.patch_stem(signal)
        if tokens.shape[1] != self.num_tokens:
            raise RuntimeError(
                f"Patch stem produced {tokens.shape[1]} tokens; expected {self.num_tokens}"
            )
        tokens = tokens + self.position_embedding.to(dtype=tokens.dtype)
        tokens = tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        encoded = torch.zeros_like(tokens)
        nonempty_rows = valid_mask.any(dim=1)
        if bool(nonempty_rows.any()):
            active_tokens = tokens[nonempty_rows]
            active_mask = valid_mask[nonempty_rows]
            active_encoded = self.full_window_encoder(
                active_tokens,
                src_key_padding_mask=~active_mask,
            )
            active_encoded = self.output_norm(active_encoded)
            active_encoded = active_encoded.masked_fill(
                ~active_mask.unsqueeze(-1), 0.0
            )
            row_indices = nonempty_rows.nonzero(as_tuple=False).squeeze(-1)
            encoded = encoded.index_copy(0, row_indices, active_encoded)
        return encoded


class SharedDriverTrajectoryDecoder(nn.Module):
    """Map a D-dimensional semantic token to one full trajectory patch."""

    def __init__(
        self,
        latent_dim: int = 64,
        target_points: int = 20,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or target_points <= 0:
            raise ValueError("latent_dim and target_points must be positive")
        hidden_dim = int(hidden_dim or latent_dim * 2)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.latent_dim = int(latent_dim)
        self.target_points = int(target_points)
        self.network = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.target_points),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected [B,N,{self.latent_dim}] latent, got {tuple(latent.shape)}"
            )
        return self.network(latent)


class SharedDriverContinuousModel(nn.Module):
    """R2-D/R2-P modality-independent continuous shared-driver students.

    The formal R-series contract uses ``num_tokens=10`` and ``latent_dim=64``.
    Configurability is retained for fast unit and synthetic tests.  Teacher
    trajectories and all task/subject/phase metadata intentionally remain
    outside this model's public forward signature.
    """

    architecture_name = "shared_driver_semantic_vq_v1_continuous"
    token_temporal_scope = "bidirectional_full_window"

    def __init__(
        self,
        eeg_channels: int = 6,
        fnirs_channels: int = 2,
        eeg_patch_samples: int = 400,
        fnirs_patch_samples: int = 20,
        num_tokens: int = 10,
        latent_dim: int = 64,
        target_points: int = 20,
        encoder_depth: int = 2,
        encoder_num_heads: int = 4,
        encoder_feedforward_dim: int | None = None,
        decoder_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.latent_dim = int(latent_dim)
        self.target_points = int(target_points)

        encoder_kwargs = {
            "num_tokens": self.num_tokens,
            "latent_dim": self.latent_dim,
            "depth": int(encoder_depth),
            "num_heads": int(encoder_num_heads),
            "feedforward_dim": encoder_feedforward_dim,
            "dropout": float(dropout),
        }
        self.eeg_encoder = FullWindowModalityEncoder(
            input_channels=int(eeg_channels),
            patch_samples=int(eeg_patch_samples),
            **encoder_kwargs,
        )
        self.fnirs_encoder = FullWindowModalityEncoder(
            input_channels=int(fnirs_channels),
            patch_samples=int(fnirs_patch_samples),
            **encoder_kwargs,
        )
        self.driver_decoder = SharedDriverTrajectoryDecoder(
            latent_dim=self.latent_dim,
            target_points=self.target_points,
            hidden_dim=decoder_hidden_dim,
        )

    def encode_eeg(
        self,
        eeg: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.eeg_encoder(eeg, token_valid_mask)

    def encode_fnirs(
        self,
        fnirs: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.fnirs_encoder(fnirs, token_valid_mask)

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        eeg_token_valid_mask: torch.Tensor | None = None,
        fnirs_token_valid_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        eeg_valid_mask = self.eeg_encoder.resolve_valid_mask(
            eeg, eeg_token_valid_mask
        )
        fnirs_valid_mask = self.fnirs_encoder.resolve_valid_mask(
            fnirs, fnirs_token_valid_mask
        )
        eeg_latent = self.encode_eeg(eeg, eeg_valid_mask)
        fnirs_latent = self.encode_fnirs(fnirs, fnirs_valid_mask)

        eeg_decoded = self.driver_decoder(eeg_latent).masked_fill(
            ~eeg_valid_mask.unsqueeze(-1), 0.0
        )
        fnirs_decoded = self.driver_decoder(fnirs_latent).masked_fill(
            ~fnirs_valid_mask.unsqueeze(-1), 0.0
        )
        return {
            "eeg_latent": eeg_latent,
            "fnirs_latent": fnirs_latent,
            "eeg_decoded": eeg_decoded,
            "fnirs_decoded": fnirs_decoded,
            "eeg_token_valid_mask": eeg_valid_mask,
            "fnirs_token_valid_mask": fnirs_valid_mask,
        }


__all__ = [
    "FullWindowModalityEncoder",
    "SharedDriverContinuousModel",
    "SharedDriverTrajectoryDecoder",
    "TemporalPatchStem",
]
