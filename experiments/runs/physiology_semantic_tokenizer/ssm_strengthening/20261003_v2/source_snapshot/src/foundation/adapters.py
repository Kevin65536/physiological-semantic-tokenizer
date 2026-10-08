"""Adapters that normalize tokenizer outputs into unified foundation inputs."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch

from src.tokenizers.physiology_semantic_tokenizer import ModalityTokenizerOutput

from .types import MultimodalTokenBatch, PretrainingTargets, TokenSequence


class TokenBatchAdapter:
    """Convert tokenizer outputs or raw ids into a standard multimodal token batch."""

    def __init__(self, pad_token_id: int = 0, mask_token_id: int = 1):
        self.pad_token_id = int(pad_token_id)
        self.mask_token_id = int(mask_token_id)

    def from_tokenizer_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> MultimodalTokenBatch:
        eeg_ids = self._resolve_key(outputs, ["eeg_indices", "eeg_tokens"])
        fnirs_ids = self._resolve_key(outputs, ["fnirs_indices", "fnirs_tokens"])
        return self.from_token_ids(eeg_ids=eeg_ids, fnirs_ids=fnirs_ids, labels=labels)

    def from_token_ids(
        self,
        eeg_ids: torch.Tensor,
        fnirs_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        eeg_attention_mask: Optional[torch.Tensor] = None,
        fnirs_attention_mask: Optional[torch.Tensor] = None,
    ) -> MultimodalTokenBatch:
        eeg_ids = self._ensure_2d_tokens(eeg_ids)
        fnirs_ids = self._ensure_2d_tokens(fnirs_ids)

        eeg_mask = eeg_attention_mask
        fnirs_mask = fnirs_attention_mask
        if eeg_mask is None:
            eeg_mask = (eeg_ids != self.pad_token_id).long()
        if fnirs_mask is None:
            fnirs_mask = (fnirs_ids != self.pad_token_id).long()

        return MultimodalTokenBatch(
            eeg=TokenSequence(input_ids=eeg_ids.long(), attention_mask=eeg_mask.long()),
            fnirs=TokenSequence(input_ids=fnirs_ids.long(), attention_mask=fnirs_mask.long()),
            labels=labels,
        )

    def from_physiology_semantic_outputs(
        self,
        outputs: Mapping[str, ModalityTokenizerOutput],
        mode: str,
        labels: Optional[torch.Tensor] = None,
    ) -> MultimodalTokenBatch:
        """Build a whole-brain input without creating replacement codebook weights."""
        if mode not in {"hard", "codebook", "soft", "semantic_residual"}:
            raise ValueError("mode must be one of: hard, codebook, soft, semantic_residual")

        def convert(output: ModalityTokenizerOutput) -> TokenSequence:
            raw_ids = output.quantizer.hard_ids.long()
            input_ids = raw_ids + 2  # reserve 0/1 for padding and masking
            mask = torch.ones_like(input_ids)
            embeddings = None
            if mode == "codebook":
                embeddings = output.quantizer.codebook[raw_ids]
            elif mode == "soft":
                embeddings = output.quantizer.expected_embedding
            elif mode == "semantic_residual":
                embeddings = torch.cat((output.quantizer.expected_embedding, output.residual), dim=-1)
            return TokenSequence(input_ids=input_ids, attention_mask=mask, inputs_embeds=embeddings)

        return MultimodalTokenBatch(
            eeg=convert(outputs["eeg"]),
            fnirs=convert(outputs["fnirs"]),
            labels=labels,
            metadata={"representation_mode": mode},
        )

    def create_mlm_inputs(
        self,
        batch: MultimodalTokenBatch,
        mask_ratio: float = 0.15,
        ignore_index: int = -100,
    ) -> Tuple[MultimodalTokenBatch, PretrainingTargets]:
        eeg_input_ids, eeg_labels = self._mask_one_modality(
            batch.eeg.input_ids,
            batch.eeg.attention_mask,
            mask_ratio=mask_ratio,
            ignore_index=ignore_index,
        )
        fnirs_input_ids, fnirs_labels = self._mask_one_modality(
            batch.fnirs.input_ids,
            batch.fnirs.attention_mask,
            mask_ratio=mask_ratio,
            ignore_index=ignore_index,
        )

        masked_batch = MultimodalTokenBatch(
            eeg=TokenSequence(eeg_input_ids, batch.eeg.attention_mask),
            fnirs=TokenSequence(fnirs_input_ids, batch.fnirs.attention_mask),
            labels=batch.labels,
            metadata=dict(batch.metadata),
        )
        targets = PretrainingTargets(eeg_mlm_labels=eeg_labels, fnirs_mlm_labels=fnirs_labels)
        return masked_batch, targets

    def _mask_one_modality(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        mask_ratio: float,
        ignore_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        valid_tokens = attention_mask.bool() & (input_ids != self.pad_token_id)
        random_tensor = torch.rand_like(input_ids.float())
        mask_positions = (random_tensor < mask_ratio) & valid_tokens

        # Ensure at least one prediction target per sample to stabilize loss.
        no_mask_rows = mask_positions.sum(dim=1) == 0
        if no_mask_rows.any():
            for row_idx in torch.where(no_mask_rows)[0].tolist():
                valid_idx = torch.where(valid_tokens[row_idx])[0]
                if valid_idx.numel() > 0:
                    mask_positions[row_idx, valid_idx[0]] = True

        labels = torch.full_like(input_ids, fill_value=ignore_index)
        labels[mask_positions] = input_ids[mask_positions]

        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_positions] = self.mask_token_id
        return masked_input_ids, labels

    @staticmethod
    def _ensure_2d_tokens(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 3:
            # Flatten lead dimension to a longer sequence [B, C, T] -> [B, C*T]
            return tokens.reshape(tokens.shape[0], -1)
        if tokens.dim() == 2:
            return tokens
        raise ValueError(f"Expected token tensor with rank 2 or 3, got shape {tuple(tokens.shape)}")

    @staticmethod
    def _resolve_key(outputs: Dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
        for key in keys:
            if key in outputs:
                return outputs[key]
        raise KeyError(f"Could not find any key in {keys}. Available keys: {list(outputs.keys())}")
