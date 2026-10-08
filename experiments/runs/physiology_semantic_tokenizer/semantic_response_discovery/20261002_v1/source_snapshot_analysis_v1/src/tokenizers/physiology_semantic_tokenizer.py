"""Independent patch-local physiology-semantic EEG and fNIRS tokenizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

from .ema_vector_quantizer import EMAVectorQuantizer, QuantizerOutput
from .labram_vqnsp import MultiChannelPatchEmbedding
from .registry import register_tokenizer


@dataclass
class ModalityTokenizerOutput:
    patches: torch.Tensor
    encoder_features: torch.Tensor
    semantic_latent: torch.Tensor
    residual: torch.Tensor
    quantizer: QuantizerOutput
    state_prediction: torch.Tensor
    prototype_state_prediction: torch.Tensor
    context_state_prediction: torch.Tensor
    context_valid_mask: torch.Tensor
    reconstruction: torch.Tensor
    semantic_reconstruction: torch.Tensor
    hard_reconstruction: torch.Tensor
    hard_semantic_reconstruction: torch.Tensor
    annealed_hard_reconstruction: torch.Tensor
    annealed_hard_semantic_reconstruction: torch.Tensor
    residual_reconstruction: torch.Tensor


class FixedHistoryContext(nn.Module):
    """Predict current-patch state from exactly the preceding fixed-size history."""

    def __init__(
        self,
        embedding_dim: int,
        state_dim: int,
        history_tokens: int = 5,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.history_tokens = int(history_tokens)
        if self.history_tokens <= 0:
            raise ValueError("history_tokens must be positive")
        self.relative_lag_embedding = nn.Parameter(torch.zeros(self.history_tokens, embedding_dim))
        nn.init.trunc_normal_(self.relative_lag_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, state_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 3:
            raise ValueError(f"Expected [B,N,D] tokens, got {tuple(tokens.shape)}")
        batch_size, token_count, _ = tokens.shape
        if token_valid_mask is None:
            token_valid_mask = torch.ones(
                batch_size, token_count, device=tokens.device, dtype=torch.bool
            )
        elif token_valid_mask.shape != (batch_size, token_count):
            raise ValueError("token_valid_mask must have shape [B,N]")
        else:
            token_valid_mask = token_valid_mask.to(device=tokens.device, dtype=torch.bool)
        tokens = tokens.masked_fill(~token_valid_mask.unsqueeze(-1), 0.0)
        predictions = tokens.new_zeros(batch_size, token_count, self.head.out_features)
        valid = torch.zeros(batch_size, token_count, device=tokens.device, dtype=torch.bool)
        if token_count <= self.history_tokens:
            return predictions, valid

        windows = tokens.unfold(dimension=1, size=self.history_tokens, step=1)
        # unfold emits [B, windows, D, history]; exclude the final window, which
        # would predict a non-existent token after the sequence.
        windows = windows[:, : token_count - self.history_tokens].permute(0, 1, 3, 2).contiguous()
        window_count = windows.shape[1]
        encoded = windows.reshape(batch_size * window_count, self.history_tokens, -1)
        encoded = encoded + self.relative_lag_embedding.unsqueeze(0)
        encoded = self.encoder(encoded)
        state = self.head(self.norm(encoded[:, -1]))
        predictions[:, self.history_tokens :] = state.reshape(batch_size, window_count, -1)
        history_valid = token_valid_mask.unfold(
            dimension=1, size=self.history_tokens, step=1
        )[:, :window_count].all(dim=-1)
        valid[:, self.history_tokens :] = history_valid
        return predictions, valid


class _ModalityBranch(nn.Module):
    def __init__(
        self,
        input_channels: int,
        patch_size: int,
        encoder_dim: int,
        semantic_dim: int,
        residual_dim: int,
        state_dim: int,
        codebook_size: int,
        history_tokens: int,
        quantizer_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.patch_size = int(patch_size)
        self.patch_embedding = MultiChannelPatchEmbedding(
            input_channels=input_channels,
            patch_size=patch_size,
            embed_dim=encoder_dim,
            use_frequency=True,
        )
        self.local_encoder = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim),
            nn.GELU(),
            nn.Linear(encoder_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
        )
        self.semantic_head = nn.Linear(encoder_dim, semantic_dim)
        self.residual_head = nn.Linear(encoder_dim, residual_dim)
        self.quantizer = EMAVectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=semantic_dim,
            **quantizer_kwargs,
        )
        self.state_head = nn.Linear(semantic_dim, state_dim)
        self.prototype_state_head = nn.Linear(semantic_dim, state_dim)
        self.context = FixedHistoryContext(
            embedding_dim=semantic_dim,
            state_dim=state_dim,
            history_tokens=history_tokens,
        )
        decoder_input_dim = semantic_dim + residual_dim
        decoder_output_dim = input_channels * patch_size
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, encoder_dim),
            nn.GELU(),
            nn.Linear(encoder_dim, decoder_output_dim),
        )

    def _patchify(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.dim() != 3 or signal.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected [B,{self.input_channels},T], got {tuple(signal.shape)}"
            )
        if signal.shape[-1] % self.patch_size != 0:
            raise ValueError(f"Signal length {signal.shape[-1]} is not divisible by {self.patch_size}")
        return signal.unfold(dimension=-1, size=self.patch_size, step=self.patch_size).permute(0, 2, 1, 3)

    def _decode(self, semantic: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(torch.cat((semantic, residual), dim=-1))
        return decoded.reshape(*decoded.shape[:2], self.input_channels, self.patch_size)

    def forward(
        self,
        signal: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> ModalityTokenizerOutput:
        patches = self._patchify(signal)
        if token_valid_mask is not None and token_valid_mask.shape != patches.shape[:2]:
            raise ValueError("token_valid_mask must match the [B,N] patch grid")
        features = self.local_encoder(self.patch_embedding(patches))
        semantic = self.semantic_head(features)
        residual = self.residual_head(features)
        quantized = self.quantizer(semantic, valid_mask=token_valid_mask)
        state_prediction = self.state_head(semantic)
        prototype_state_prediction = self.prototype_state_head(quantized.quantized)
        context_prediction, context_valid = self.context(
            quantized.expected_embedding, token_valid_mask=token_valid_mask
        )
        zeros_semantic = torch.zeros_like(quantized.expected_embedding)
        zeros_residual = torch.zeros_like(residual)
        return ModalityTokenizerOutput(
            patches=patches,
            encoder_features=features,
            semantic_latent=semantic,
            residual=residual,
            quantizer=quantized,
            state_prediction=state_prediction,
            prototype_state_prediction=prototype_state_prediction,
            context_state_prediction=context_prediction,
            context_valid_mask=context_valid,
            reconstruction=self._decode(quantized.expected_embedding, residual),
            semantic_reconstruction=self._decode(quantized.expected_embedding, zeros_residual),
            hard_reconstruction=self._decode(quantized.quantized, residual),
            hard_semantic_reconstruction=self._decode(quantized.quantized, zeros_residual),
            annealed_hard_reconstruction=self._decode(
                quantized.annealed_quantized, residual
            ),
            annealed_hard_semantic_reconstruction=self._decode(
                quantized.annealed_quantized, zeros_residual
            ),
            residual_reconstruction=self._decode(zeros_semantic, residual),
        )


@register_tokenizer("physiology_semantic")
class PhysiologySemanticTokenizer(nn.Module):
    """Two independent inference branches with shared output semantics."""

    architecture_name = "physiology_semantic"

    def __init__(
        self,
        eeg_encoder_dim: int = 256,
        fnirs_encoder_dim: int = 160,
        semantic_dim: int = 64,
        eeg_residual_dim: int = 64,
        fnirs_residual_dim: int = 32,
        codebook_size: int = 128,
        history_tokens: int = 5,
        quantizer_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        quantizer_kwargs = dict(quantizer_kwargs or {})
        self.eeg_branch = _ModalityBranch(
            input_channels=6,
            patch_size=400,
            encoder_dim=eeg_encoder_dim,
            semantic_dim=semantic_dim,
            residual_dim=eeg_residual_dim,
            state_dim=6,
            codebook_size=codebook_size,
            history_tokens=history_tokens,
            quantizer_kwargs=quantizer_kwargs,
        )
        self.fnirs_branch = _ModalityBranch(
            input_channels=2,
            patch_size=20,
            encoder_dim=fnirs_encoder_dim,
            semantic_dim=semantic_dim,
            residual_dim=fnirs_residual_dim,
            state_dim=9,
            codebook_size=codebook_size,
            history_tokens=history_tokens,
            quantizer_kwargs=quantizer_kwargs,
        )

    def set_quantization_strength(self, strength: float) -> None:
        self.eeg_branch.quantizer.set_quantization_strength(strength)
        self.fnirs_branch.quantizer.set_quantization_strength(strength)

    def get_quantization_strength(self) -> float:
        eeg_strength = self.eeg_branch.quantizer.get_quantization_strength()
        fnirs_strength = self.fnirs_branch.quantizer.get_quantization_strength()
        if eeg_strength != fnirs_strength:
            raise RuntimeError("EEG and fNIRS quantization strengths diverged")
        return eeg_strength

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PhysiologySemanticTokenizer":
        model = config.get("model", {})
        encoder = model.get("encoder", {})
        residual = model.get("residual", {})
        quantizer = model.get("quantizer", {})
        context = model.get("context", {})
        if int(quantizer.get("codebook_size", 128)) != 128:
            raise ValueError("The first formal physiology-semantic contract requires codebook_size=128")
        if int(quantizer.get("embedding_dim", 64)) != 64:
            raise ValueError("The first formal physiology-semantic contract requires embedding_dim=64")
        return cls(
            eeg_encoder_dim=int(encoder.get("eeg_dim", 256)),
            fnirs_encoder_dim=int(encoder.get("fnirs_dim", 160)),
            semantic_dim=int(quantizer.get("embedding_dim", 64)),
            eeg_residual_dim=int(residual.get("eeg_dim", 64)),
            fnirs_residual_dim=int(residual.get("fnirs_dim", 32)),
            codebook_size=int(quantizer.get("codebook_size", 128)),
            history_tokens=int(context.get("history_tokens", 5)),
            quantizer_kwargs={
                "decay": float(quantizer.get("decay", 0.99)),
                "eps": float(quantizer.get("eps", 1e-5)),
                "commitment_cost": float(quantizer.get("commitment_cost", 0.25)),
                "temperature": float(quantizer.get("temperature", 1.0)),
                "assignment": str(quantizer.get("assignment", "euclidean")),
                "normalize_latents": bool(quantizer.get("normalize_latents", False)),
                "kmeans_init": bool(quantizer.get("kmeans_init", False)),
                "kmeans_iters": int(quantizer.get("kmeans_iters", 10)),
                "revive_dead_codes": bool(quantizer.get("revive_dead_codes", False)),
                "dead_code_threshold": float(quantizer.get("dead_code_threshold", 0.1)),
                "revival_warmup_steps": int(quantizer.get("revival_warmup_steps", 100)),
                "revival_interval": int(quantizer.get("revival_interval", 100)),
                "revival_stop_after_steps": quantizer.get("revival_stop_after_steps"),
                "max_revivals_per_event": int(quantizer.get("max_revivals_per_event", 8)),
                "revival_noise_std": float(quantizer.get("revival_noise_std", 0.0)),
                "revival_strategy": str(quantizer.get("revival_strategy", "top_error")),
                "revival_count_prior": str(quantizer.get("revival_count_prior", "threshold")),
            },
        )

    def encode_eeg(
        self,
        eeg: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> ModalityTokenizerOutput:
        return self.eeg_branch(eeg, token_valid_mask=token_valid_mask)

    def encode_fnirs(
        self,
        fnirs: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> ModalityTokenizerOutput:
        return self.fnirs_branch(fnirs, token_valid_mask=token_valid_mask)

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        token_valid_masks: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, ModalityTokenizerOutput]:
        token_valid_masks = dict(token_valid_masks or {})
        return {
            "eeg": self.encode_eeg(eeg, token_valid_masks.get("eeg")),
            "fnirs": self.encode_fnirs(fnirs, token_valid_masks.get("fnirs")),
        }


__all__ = [
    "FixedHistoryContext",
    "ModalityTokenizerOutput",
    "PhysiologySemanticTokenizer",
]
