"""Small frozen-vocabulary, same-ROI offline masked-completion reader.

The only sample-specific input is integer token IDs. Teacher components, decoder
waveforms and measurement scales are deliberately absent from ``forward``.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import torch
from torch import nn


class DualComponentMaskedModel(nn.Module):
    """Independent modality/branch vocabularies with a shared contextual reader."""

    def __init__(self, vocab_sizes: Mapping[str, int], *, patches=10, width=128,
                 layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.modalities = tuple(vocab_sizes)
        self.vocab_sizes = tuple(int(vocab_sizes[m]) for m in self.modalities)
        if not self.modalities or len(set(self.vocab_sizes)) != 1:
            raise ValueError("pilot requires equally sized modality vocabularies")
        self.vocab_size = self.vocab_sizes[0]
        if self.vocab_size < 2 or patches < 2 or width % heads:
            raise ValueError("invalid vocabulary, context length or attention width")
        self.patches = int(patches)
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size + 1, width)
            for _ in range(len(self.modalities) * 2)
        ])
        self.position = nn.Embedding(self.patches, width)
        self.stream = nn.Embedding(len(self.modalities) * 2, width)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.heads = nn.ModuleList([
            nn.Linear(width, self.vocab_size) for _ in self.embeddings
        ])

    def forward(self, ids: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        """Return logits [batch,time,modality,branch,code]; sanitize before lookup."""
        if ids.dtype not in (torch.int32, torch.int64) or visible.dtype != torch.bool:
            raise ValueError("integer IDs and boolean visibility required")
        expected = (self.patches, len(self.modalities), 2)
        if ids.ndim != 4 or tuple(ids.shape[1:]) != expected or visible.shape != ids.shape:
            raise ValueError("expected [batch,patch,modality,2] IDs and identical visibility")
        if torch.any(visible & ((ids < 0) | (ids >= self.vocab_size))):
            raise ValueError("visible IDs outside the frozen vocabulary")
        clean = torch.where(visible, ids, self.vocab_size)
        streams = []
        for m in range(len(self.modalities)):
            for c in range(2):
                j = m * 2 + c
                streams.append(self.embeddings[j](clean[:, :, m, c]))
        # Time-major order preserves explicit positions for every modality/branch.
        x = torch.stack(streams, dim=2)
        positions = self.position(torch.arange(self.patches, device=ids.device))
        stream = self.stream(torch.arange(len(streams), device=ids.device))
        x = x + positions[None, :, None] + stream[None, None]
        h = self.norm(self.transformer(x.flatten(1, 2))).reshape_as(x)
        logits = torch.stack([head(h[:, :, j]) for j, head in enumerate(self.heads)], dim=2)
        return logits.reshape(ids.shape + (self.vocab_size,))


def block_masks(valid, *, rng, target_modality=None, block_patches=3,
                task="time", single_branch=None):
    """Original-support joint masks for one ROI per sample.

    Time tasks hide both modalities in a block; modality tasks hide one modality.
    In either case both S/O tokens are hidden. A single-branch mask is explicitly
    diagnostic. No cross-ROI tokens enter this reader, avoiding reused EEG support.
    """
    valid = np.asarray(valid, bool)
    if valid.ndim != 4 or valid.shape[-1] != 2:
        raise ValueError("valid must be [sample,time,modality,2]")
    if task not in ("time", "modality"):
        raise ValueError("task must be time or modality")
    n, times, modalities, _ = valid.shape
    if not 1 <= block_patches <= times:
        raise ValueError("block support outside context")
    hidden = np.zeros_like(valid)
    score = np.zeros_like(valid)
    for i in range(n):
        start = int(rng.integers(times - block_patches + 1))
        m = int(rng.integers(modalities)) if target_modality is None else int(target_modality)
        if not 0 <= m < modalities:
            raise ValueError("target modality outside context")
        branches = slice(None) if single_branch is None else int(single_branch)
        if task == "time":
            hidden[i, start:start + block_patches, :, branches] = True
        else:
            hidden[i, start:start + block_patches, m, branches] = True
        score[i, start:start + block_patches, m, branches] = True
    return valid & ~hidden, valid & score


def own_context(visible, target_modality):
    """Remove the other modality, retaining the original target-block mask."""
    result = np.asarray(visible, bool).copy()
    result[:, :, np.arange(result.shape[2]) != int(target_modality), :] = False
    return result


def expected_waveforms(probabilities, decoded_codewords):
    """Decode every codeword before probability averaging (nonlinear-safe)."""
    p = np.asarray(probabilities, float)
    w = np.asarray(decoded_codewords, float)
    if p.ndim < 1 or w.ndim != 2 or p.shape[-1] != len(w):
        raise ValueError("probability/vocabulary shape mismatch")
    if (not np.isfinite(p).all() or not np.isfinite(w).all() or np.any(p < 0)
            or not np.allclose(p.sum(-1), 1.0, atol=1e-5)):
        raise ValueError("finite normalized code probabilities required")
    return p @ w


def fit_token_baselines(ids, rows, vocab_size, *, alpha=0.5):
    """Fit train marginal and dataset/ROI/time baselines; never use event labels."""
    ids = np.asarray(ids)
    if ids.ndim != 4 or len(ids) != len(rows) or ids.dtype.kind not in "iu":
        raise ValueError("IDs/metadata mismatch")
    if alpha <= 0:
        raise ValueError("positive pseudocount required")
    train = np.array([r["split"] == "train" for r in rows])
    if not train.any():
        raise ValueError("training support required for token baselines")
    _, times, modalities, branches = ids.shape
    marginal = np.full((modalities, branches, vocab_size), float(alpha))
    metadata = {}
    for i in np.flatnonzero(train):
        row = rows[i]
        for t in range(times):
            key = (str(row.get("dataset", "")), str(row.get("roi", row.get("region", row.get("anchor", {}).get("region", "")))), t)
            counts = metadata.setdefault(key, np.full_like(marginal, float(alpha)))
            for m in range(modalities):
                for c in range(branches):
                    k = int(ids[i, t, m, c])
                    if 0 <= k < vocab_size:
                        marginal[m, c, k] += 1
                        counts[m, c, k] += 1
    marginal /= marginal.sum(-1, keepdims=True)
    metadata = {key: value / value.sum(-1, keepdims=True) for key, value in metadata.items()}
    return dict(marginal=marginal, metadata=metadata, times=times,
                fit_rows=int(train.sum()), alpha=float(alpha))


def baseline_probabilities(baseline, rows, *, metadata=False):
    marginal = baseline["marginal"]
    result = np.broadcast_to(marginal, (len(rows), baseline["times"]) + marginal.shape).copy()
    if metadata:
        for i, row in enumerate(rows):
            for t in range(baseline["times"]):
                key = (str(row.get("dataset", "")), str(row.get("roi", row.get("region", row.get("anchor", {}).get("region", "")))), t)
                result[i, t] = baseline["metadata"].get(key, marginal)
    return result


def nonwrapping_source_shift(ids, visible, source_modality, shift):
    """Shift only originally visible source information; hide vacated positions."""
    ids, visible = np.asarray(ids), np.asarray(visible, bool)
    if ids.shape != visible.shape or not 1 <= shift < ids.shape[1]:
        raise ValueError('shift must preserve a nonempty nonwrapping context')
    shifted, keep = ids.copy(), visible.copy()
    m = int(source_modality)
    shifted[:, shift:, m] = ids[:, :-shift, m]
    keep[:, :shift, m] = False
    keep[:, shift:, m] &= visible[:, :-shift, m]
    return shifted, keep
