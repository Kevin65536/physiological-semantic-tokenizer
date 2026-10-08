"""Unified multimodal foundation interfaces for token-based pretraining and downstream tasks."""

from .adapters import TokenBatchAdapter
from .model import FoundationModelConfig, UnifiedMultimodalFoundationModel
from .types import FoundationOutput, MultimodalTokenBatch, PretrainingTargets, TokenSequence

__all__ = [
    "TokenBatchAdapter",
    "FoundationModelConfig",
    "UnifiedMultimodalFoundationModel",
    "FoundationOutput",
    "MultimodalTokenBatch",
    "PretrainingTargets",
    "TokenSequence",
]
