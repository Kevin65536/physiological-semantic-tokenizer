"""Frozen-code readers that form cross-stream relations before pooling the 15 slots."""

import math
import torch
from torch import nn


class HbSlotContext(nn.Module):
    """Four target sites × three horizons attend to 120 distinct source entries."""

    def __init__(self, width):
        super().__init__()
        self.project = nn.ModuleList(nn.Linear(128, width) for _ in range(2))
        self.site = nn.Linear(2, width)
        self.horizon = nn.Parameter(torch.randn(3, width) * 0.05)
        self.slot = nn.Parameter(torch.randn(15, width) * 0.05)
        # Middle of future scoring bin minus midpoint of the historical output slot (seconds).
        dt = torch.tensor([3.0, 5.0, 7.0])[:, None] - torch.arange(-29.0, 0.0, 2.0)[None]
        self.register_buffer("distance", dt / 36.0)
        self.lag = nn.Parameter(torch.zeros(2))

    def forward(self, query, target_xy, source, source_xy):
        keys = torch.stack([p(source[:, :, j]) for j, p in enumerate(self.project)], 2)
        keys = torch.tanh(keys + self.site(source_xy)[:, :, None, None] + self.slot)
        q = query + self.site(target_xy)[:, :, None] + self.horizon
        score = torch.einsum("blhd,bsjtd->blhsjt", q, keys) / math.sqrt(keys.shape[-1])
        offset = self.lag[0] * self.distance + self.lag[1] * self.distance.square()
        w = (score + offset[None, None, :, None, None]).flatten(3).softmax(-1)
        return torch.tanh(q + torch.einsum("blhk,bkd->blhd", w, keys.flatten(1, 3)))


class AllHb(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.context = HbSlotContext(width)
        self.output = nn.Linear(width, 2)

    def features(self, h, xy):
        return self.context(0, xy, h, xy)

    def forward(self, h, xy):
        return self.output(self.features(h, xy))


class HbIncrement(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.hb = HbSlotContext(width)
        self.source = HbSlotContext(width)
        self.output = nn.Linear(width, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, h, xy, source, source_xy):
        # Full Hb route in both arms, including every position at every horizon.
        query = self.hb(0, xy, h, xy)
        return self.output(self.source(query, xy, source, source_xy))


class AdditiveSlots(nn.Module):
    def __init__(self, width, colors):
        super().__init__()
        self.project = nn.ModuleList(nn.Linear(128, width) for _ in range(2))
        self.loc = nn.ModuleList(nn.Linear(2, width) for _ in range(2))
        self.head = nn.ModuleList(
            nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, colors)) for _ in range(2)
        )
        self.time = nn.Parameter(torch.zeros(2, 3, 15))

    def forward(self, x, xy):
        parts = []
        for j in range(2):
            a = torch.tanh(self.project[j](x[:, :, j]) + self.loc[j](xy)[:, :, None])
            parts.append(self.head[j](torch.einsum("ht,bstk->bshk", self.time[j].softmax(-1), a)))
        return parts[0] + parts[1]


class LocalInteraction(nn.Module):
    def __init__(self, rank, colors, *, additive=False):
        super().__init__()
        self.additive = additive
        self.left = nn.Linear(128, rank)
        self.right = nn.Linear(128, rank)
        self.time = nn.Parameter(torch.zeros(3, 3, 15))
        self.output = nn.Linear(rank, colors)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, xy):
        left = self.left(x[:, :, 0])
        right = self.right(x[:, :, 1])
        # Crop invalid boundary pairs, never wrap from the last slot to the first.
        if self.additive:
            left, right = left.tanh(), right.tanh()
            combine = torch.add
        else:
            combine = torch.mul
        pairs = [
            combine(left[:, :, 1:], right[:, :, :-1]),
            combine(left, right),
            combine(left[:, :, :-1], right[:, :, 1:]),
        ]
        weight = [self.time[0, :, 1:], self.time[1], self.time[2, :, :-1]]
        z = torch.einsum("ht,bstr->bshr", torch.cat(weight, -1).softmax(-1), torch.cat(pairs, 2))
        return self.output(z)
