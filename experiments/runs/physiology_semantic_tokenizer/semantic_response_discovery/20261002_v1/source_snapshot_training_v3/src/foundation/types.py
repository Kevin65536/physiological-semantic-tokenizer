"""Typed containers for multimodal token foundation training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class TokenSequence:
    """Discrete token sequence for one modality."""

    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    inputs_embeds: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "TokenSequence":
        return TokenSequence(
            input_ids=self.input_ids.to(device),
            attention_mask=None if self.attention_mask is None else self.attention_mask.to(device),
            inputs_embeds=None if self.inputs_embeds is None else self.inputs_embeds.to(device),
        )


@dataclass
class MultimodalTokenBatch:
    """Unified input container consumed by foundation tasks."""

    eeg: TokenSequence
    fnirs: TokenSequence
    labels: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device) -> "MultimodalTokenBatch":
        return MultimodalTokenBatch(
            eeg=self.eeg.to(device),
            fnirs=self.fnirs.to(device),
            labels=None if self.labels is None else self.labels.to(device),
            metadata=self.metadata,
        )


@dataclass
class PretrainingTargets:
    """Targets for self-supervised multimodal pretraining."""

    eeg_mlm_labels: torch.Tensor
    fnirs_mlm_labels: torch.Tensor

    def to(self, device: torch.device) -> "PretrainingTargets":
        return PretrainingTargets(
            eeg_mlm_labels=self.eeg_mlm_labels.to(device),
            fnirs_mlm_labels=self.fnirs_mlm_labels.to(device),
        )


@dataclass
class FoundationOutput:
    """Common output format for pretraining and downstream tasks."""

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    metrics: Dict[str, torch.Tensor] = field(default_factory=dict)
    extra: Dict[str, torch.Tensor] = field(default_factory=dict)
