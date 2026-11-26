"""Masked completion over the four independent frozen aligned S/O vocabulary spaces."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F


def condition_permutation(frame, min_seconds, seed):
    """Maximum number of moved trials in a bijection, preserving both modality marginals."""
    rng = np.random.default_rng(seed)
    donor = np.full(len(frame), -1, dtype=int)
    for _, g in frame.groupby(["split", "dataset", "subject", "record", "condition"], sort=True):
        n = len(g)
        eeg, hb = g.eeg_start.to_numpy(), g.hb_start.to_numpy()
        event = g.event.to_numpy()
        legal = (
            (np.abs(eeg[:, None] - eeg) >= min_seconds)
            & (np.abs(hb[:, None] - hb) >= min_seconds)
            & (event[:, None] != event)
        )
        # Diagonal leaves an unmatched trial out of every arm.
        cost = np.where(legal, -2 * (n + 1) + rng.random((n, n)), np.where(np.eye(n), 0, 4 * (n + 1) ** 2))
        left, right = linear_sum_assignment(cost)
        moved = left != right
        if not legal[left[moved], right[moved]].all():
            raise ValueError("inadmissible trial assigned")
        idx = g.index.to_numpy()
        donor[idx[left[moved]]] = idx[right[moved]]
    kept = np.flatnonzero(donor >= 0)
    if not np.array_equal(np.sort(donor[kept]), kept):
        raise ValueError("donor assignment changed marginal support")
    return donor


def hidden_sites(batch, regions, patches, task, rng):
    """Return [batch, modality, native position, time], shared by both branches."""
    if task not in ("span", "eeg_hidden", "hb_hidden") or patches != 15:
        raise ValueError("expected registered mask and 15-patch support")
    shape = (batch, 2, regions, patches)
    if task == "span":
        length = rng.integers(7, 9, size=shape[:-1] + (1,))
        start = rng.integers(0, patches + 1 - length)
        t = np.arange(patches)
        return (t >= start) & (t < start + length)
    mask = np.zeros(shape, bool)
    mask[:, int(task == "hb_hidden")] = True
    return mask


class FrozenCodeReader(nn.Module):
    """Stream-specific frozen books and heads; the reader sees IDs and sensor geometry only."""

    def __init__(self, books, streams, config):
        super().__init__()
        self.streams = tuple(streams)
        width = config["width"]
        self.books = nn.ModuleList(
            [nn.Embedding.from_pretrained(books[i].detach().clone(), freeze=True) for i in streams]
        )
        self.project = nn.ModuleList([nn.Linear(128, width) for _ in streams])
        self.mask = nn.Parameter(torch.randn(len(streams), width) * 0.02)
        self.stream_id = nn.Embedding(4, width)
        self.time = nn.Embedding(15, width)
        self.position = nn.Linear(2, width)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    width,
                    config["heads"],
                    config["feedforward"],
                    config["dropout"],
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config["layers"])
            ]
        )
        self.norm = nn.LayerNorm(width)
        self.output = nn.ModuleList([nn.Linear(width, len(books[i])) for i in streams])

    def encode(self, ids, xy, hidden, *, isolated=False):
        # Each stream contains its own actual montage coordinates, including donor coordinates for M4.
        tokens = []
        for j, stream in enumerate(self.streams):
            mask = hidden[:, stream // 2]
            safe = ids[:, stream].masked_fill(mask, 0)
            value = self.project[j](self.books[j](safe))
            value = torch.where(mask[..., None], self.mask[j], value)
            value = value + self.stream_id.weight[stream] + self.time.weight[None, None]
            value = value + self.position(xy[:, stream // 2])[:, :, None]
            tokens.append(value)
        x = torch.stack(tokens, 1)
        shape = x.shape
        # Fold modality into the batch dimension to isolate attention without an attention mask.
        if isolated and self.streams == (0, 1, 2, 3):
            x = x.reshape(shape[0] * 2, -1, shape[-1])
        else:
            x = x.flatten(1, 3)
        for layer in self.blocks:
            x = layer(x)
        return self.norm(x).reshape(shape)

    def objective(self, ids, xy, hidden, *, isolated=False):
        z = self.encode(ids, xy, hidden, isolated=isolated)
        losses = []
        for j, stream in enumerate(self.streams):
            score = hidden[:, stream // 2]
            if score.any():
                losses.append(
                    F.cross_entropy(self.output[j](z[:, j][score]).float(), ids[:, stream][score])
                )
        if not losses:
            raise ValueError("empty masked target")
        # Every scored modality has both S and O on identical support.
        return torch.stack(losses).mean()


def compose(ids, xy, indices, donors=None):
    v = ids[indices]
    loc = xy[indices]
    if donors is not None:
        v = torch.cat((v[:, :2], ids[donors, 2:]), dim=1)
        loc = torch.cat((loc[:, :1], xy[donors, 1:]), dim=1)
    return v, loc


def descriptors(z, seed=1126):
    # Preserve ordering and separate sensors/streams; use the same fixed-width projection for each comparator.
    shape = z.shape
    bins = z.reshape(*shape[:3], 3, 5, shape[-1]).mean(4).flatten(1)
    gen = torch.Generator(device=z.device).manual_seed(seed)
    proj = torch.randn(bins.shape[1], 128, generator=gen, device=z.device) / bins.shape[1] ** 0.5
    return (bins @ proj).float().cpu().numpy()
