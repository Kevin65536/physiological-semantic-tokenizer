"""Foundation-style unified interface for multimodal token pretraining and downstream tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import FoundationOutput, MultimodalTokenBatch, PretrainingTargets


@dataclass
class FoundationModelConfig:
    """Hyperparameters for the multimodal token foundation model."""

    eeg_vocab_size: int
    fnirs_vocab_size: int
    max_seq_len: int = 2048
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_classes: int = 2
    pad_token_id: int = 0
    contrastive_temperature: float = 0.07


class UnifiedMultimodalFoundationModel(nn.Module):
    """Single backbone for self-supervised pretraining and downstream prediction."""

    def __init__(self, cfg: FoundationModelConfig):
        super().__init__()
        self.cfg = cfg

        self.eeg_embedding = nn.Embedding(cfg.eeg_vocab_size, cfg.hidden_dim, padding_idx=cfg.pad_token_id)
        self.fnirs_embedding = nn.Embedding(cfg.fnirs_vocab_size, cfg.hidden_dim, padding_idx=cfg.pad_token_id)
        self.eeg_input_projection = nn.LazyLinear(cfg.hidden_dim)
        self.fnirs_input_projection = nn.LazyLinear(cfg.hidden_dim)
        self.modality_embedding = nn.Embedding(2, cfg.hidden_dim)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=int(cfg.hidden_dim * cfg.mlp_ratio),
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.final_norm = nn.LayerNorm(cfg.hidden_dim)

        self.eeg_mlm_head = nn.Linear(cfg.hidden_dim, cfg.eeg_vocab_size)
        self.fnirs_mlm_head = nn.Linear(cfg.hidden_dim, cfg.fnirs_vocab_size)

        self.eeg_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.fnirs_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        self.downstream_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def encode(self, batch: MultimodalTokenBatch) -> dict[str, torch.Tensor]:
        eeg_states = self._encode_modality(
            input_ids=batch.eeg.input_ids,
            attention_mask=batch.eeg.attention_mask,
            token_embedding=self.eeg_embedding,
            inputs_embeds=batch.eeg.inputs_embeds,
            input_projection=self.eeg_input_projection,
            modality_id=0,
        )
        fnirs_states = self._encode_modality(
            input_ids=batch.fnirs.input_ids,
            attention_mask=batch.fnirs.attention_mask,
            token_embedding=self.fnirs_embedding,
            inputs_embeds=batch.fnirs.inputs_embeds,
            input_projection=self.fnirs_input_projection,
            modality_id=1,
        )

        eeg_pooled = self._masked_pool(eeg_states, batch.eeg.attention_mask)
        fnirs_pooled = self._masked_pool(fnirs_states, batch.fnirs.attention_mask)
        return {
            "eeg_states": eeg_states,
            "fnirs_states": fnirs_states,
            "eeg_pooled": eeg_pooled,
            "fnirs_pooled": fnirs_pooled,
        }

    def forward_pretrain(
        self,
        batch: MultimodalTokenBatch,
        targets: Optional[PretrainingTargets] = None,
        lambda_mlm: float = 1.0,
        lambda_contrastive: float = 1.0,
        ignore_index: int = -100,
    ) -> FoundationOutput:
        encoded = self.encode(batch)
        eeg_states = encoded["eeg_states"]
        fnirs_states = encoded["fnirs_states"]

        eeg_logits = self.eeg_mlm_head(eeg_states)
        fnirs_logits = self.fnirs_mlm_head(fnirs_states)

        metrics = {
            "mlm_loss": torch.tensor(0.0, device=eeg_logits.device),
            "contrastive_loss": torch.tensor(0.0, device=eeg_logits.device),
        }
        loss = None

        if targets is not None:
            eeg_mlm_loss = F.cross_entropy(
                eeg_logits.view(-1, eeg_logits.shape[-1]),
                targets.eeg_mlm_labels.view(-1),
                ignore_index=ignore_index,
            )
            fnirs_mlm_loss = F.cross_entropy(
                fnirs_logits.view(-1, fnirs_logits.shape[-1]),
                targets.fnirs_mlm_labels.view(-1),
                ignore_index=ignore_index,
            )
            mlm_loss = 0.5 * (eeg_mlm_loss + fnirs_mlm_loss)

            eeg_repr = F.normalize(self.eeg_proj(encoded["eeg_pooled"]), dim=-1)
            fnirs_repr = F.normalize(self.fnirs_proj(encoded["fnirs_pooled"]), dim=-1)
            contrastive_loss = self._cross_modal_info_nce(
                eeg_repr,
                fnirs_repr,
                temperature=self.cfg.contrastive_temperature,
            )

            loss = lambda_mlm * mlm_loss + lambda_contrastive * contrastive_loss
            metrics["mlm_loss"] = mlm_loss
            metrics["contrastive_loss"] = contrastive_loss

        return FoundationOutput(
            loss=loss,
            metrics=metrics,
            extra={
                "eeg_mlm_logits": eeg_logits,
                "fnirs_mlm_logits": fnirs_logits,
                "eeg_pooled": encoded["eeg_pooled"],
                "fnirs_pooled": encoded["fnirs_pooled"],
            },
        )

    def forward_downstream(
        self,
        batch: MultimodalTokenBatch,
        labels: Optional[torch.Tensor] = None,
    ) -> FoundationOutput:
        encoded = self.encode(batch)
        fused = torch.cat([encoded["eeg_pooled"], encoded["fnirs_pooled"]], dim=-1)
        logits = self.downstream_head(fused)

        loss = None
        metrics: dict[str, torch.Tensor] = {}
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())
            preds = logits.argmax(dim=-1)
            metrics["accuracy"] = (preds == labels).float().mean()

        return FoundationOutput(
            loss=loss,
            logits=logits,
            metrics=metrics,
            extra={
                "eeg_pooled": encoded["eeg_pooled"],
                "fnirs_pooled": encoded["fnirs_pooled"],
            },
        )

    def forward(
        self,
        batch: MultimodalTokenBatch,
        task: str,
        pretrain_targets: Optional[PretrainingTargets] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> FoundationOutput:
        if task == "pretrain":
            return self.forward_pretrain(batch=batch, targets=pretrain_targets)
        if task == "downstream":
            return self.forward_downstream(batch=batch, labels=labels if labels is not None else batch.labels)
        raise ValueError(f"Unknown task: {task}. Supported: ['pretrain', 'downstream']")

    def _encode_modality(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        token_embedding: nn.Embedding,
        inputs_embeds: Optional[torch.Tensor],
        input_projection: nn.Module,
        modality_id: int,
    ) -> torch.Tensor:
        input_ids = self._truncate_if_needed(input_ids)
        if attention_mask is not None:
            attention_mask = attention_mask[:, : input_ids.shape[1]]

        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        modality_ids = torch.full_like(input_ids, fill_value=modality_id)

        if inputs_embeds is None:
            x = token_embedding(input_ids)
        else:
            inputs_embeds = inputs_embeds[:, :seq_len]
            x = input_projection(inputs_embeds)
        x = x + self.modality_embedding(modality_ids) + self.position_embedding(positions)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        x = self.backbone(x, src_key_padding_mask=key_padding_mask)
        return self.final_norm(x)

    def _truncate_if_needed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] <= self.cfg.max_seq_len:
            return input_ids
        return input_ids[:, : self.cfg.max_seq_len]

    @staticmethod
    def _masked_pool(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = attention_mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return (hidden_states * mask).sum(dim=1) / denom

    @staticmethod
    def _cross_modal_info_nce(eeg_repr: torch.Tensor, fnirs_repr: torch.Tensor, temperature: float) -> torch.Tensor:
        logits = eeg_repr @ fnirs_repr.t()
        logits = logits / max(float(temperature), 1e-6)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_e2f = F.cross_entropy(logits, labels)
        loss_f2e = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_e2f + loss_f2e)
