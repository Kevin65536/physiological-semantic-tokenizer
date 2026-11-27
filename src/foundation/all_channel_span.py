"""Variable-size native sensor sets for masked EEG/Hb code prediction."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .typed_span import TypedSpanReader


def neighbors(target_xyz, source_xyz, count, radius):
    """Unit-head chord neighborhoods. NaN positions have no spatial edges."""
    distance = torch.cdist(target_xyz.nan_to_num(), source_xyz.nan_to_num())
    known = target_xyz.isfinite().all(-1)[:, :, None] & source_xyz.isfinite().all(-1)[:, None, :]
    distance = distance.masked_fill(~known, torch.inf)
    d, idx = distance.topk(min(count, distance.shape[-1]), largest=False)
    return idx, d <= radius


def gather_sites(values, indices):
    batch = torch.arange(len(values), device=values.device)[:, None, None]
    return values[batch, indices]


def masked_wave(raw, shown, modality):
    wave = raw.masked_fill(~shown[..., None], 0)
    if modality == 0:
        return wave
    colors = wave.unflatten(-1, (20, 2)).flatten(2, 3)
    valid = shown.repeat_interleave(20, dim=-1)
    ref_count = valid[..., :50].sum(-1, keepdim=True)
    early = colors[:, :, :50].sum(-2) / ref_count.clamp_min(1)
    fallback = colors.sum(-2) / valid.sum(-1, keepdim=True).clamp_min(1)
    ref = torch.where(ref_count > 0, early, fallback)
    return (colors - ref.unsqueeze(-2)).reshape_as(raw).masked_fill(~shown[..., None], 0)


def hidden_sites(valid, xyz, family, modality, generator):
    batch, sites, slots = valid.shape
    if family == "whole":
        return valid.clone()
    if family == "span":
        length = (3, 6)[modality]
        starts = torch.randint(slots - length + 1, (batch, 1, 1), generator=generator, device=valid.device)
        t = torch.arange(slots, device=valid.device)
        return valid & (t >= starts) & (t < starts + length)
    if family != "region":
        raise ValueError("unknown mask family")
    live = valid.any(-1)
    ranks = torch.rand(live.shape, generator=generator, device=valid.device).masked_fill(~live, 2)
    anchor = ranks.argmin(-1)
    centers = xyz[torch.arange(batch, device=valid.device), anchor]
    d = (xyz - centers[:, None]).square().sum(-1).nan_to_num(nan=torch.inf)
    d.scatter_(1, anchor[:, None], -1)
    # Each event's region size depends on its own support, never on batch padding.
    order = d.masked_fill(~live, torch.inf).argsort(-1).argsort(-1)
    region = (order < ((live.sum(-1, keepdim=True) + 3) // 4)) & d.isfinite()
    return valid & region[..., None]


class NativeSpanReader(TypedSpanReader):
    def __init__(self, vocab, width, supports, count, radius):
        super().__init__(vocab, 128, width, supports)
        self.count, self.radius = count, radius
        self.spatial = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in vocab)

    def encode_target(self, codes, xyz, xy, valid, hidden, mod, cross=True):
        source = 1 - mod
        def project(stream, values, positions, mask=None):
            z = self.project[stream](values)
            if mask is not None:
                z = torch.where(mask[..., None], self.mask[stream], z)
            return z + self.stream.weight[stream] + self.time.weight + self.position(positions.nan_to_num())[:, :, None]

        local_idx, local_edge = neighbors(xyz[mod], xyz[mod], self.count, self.radius)
        other_idx, other_edge = neighbors(xyz[mod], xyz[source], self.count, self.radius)
        source_keys = [project(2 * source + a, v, xy[source]) for a, v in enumerate(codes[source])]
        outputs = []
        for b, value in enumerate(codes[mod]):
            stream = 2 * mod + b
            z = project(stream, value, xy[mod], hidden)
            # Transformer requires one unmasked key even for a fully padded site.
            pad = ~valid[mod].flatten(0, 1)
            safe_pad = pad & ~pad.all(-1, keepdim=True)
            query = self.context(z.flatten(0, 1), src_key_padding_mask=safe_pad).reshape_as(z)
            local = gather_sites(z, local_idx)
            available = gather_sites(valid[mod] & ~hidden, local_idx) & local_edge[..., None]
            scores = (local * query[:, :, None]).sum(-1) / math.sqrt(z.shape[-1])
            scores = scores.masked_fill(~available, -torch.inf)
            weights = F.pad(scores.transpose(-1, -2), (0, 1)).softmax(-1)[..., :-1].transpose(-1, -2)
            query = query + self.spatial[stream]((weights[..., None] * local).sum(2))
            message = torch.zeros_like(query)
            if cross:
                for a, keys in enumerate(source_keys):
                    path = self.paths[2 * a + b if mod else 2 * b + a]
                    key = gather_sites(path.key(keys), other_idx).flatten(2, 3)
                    val = gather_sites(path.value(keys), other_idx).flatten(2, 3)
                    allowed = gather_sites(valid[source], other_idx) & other_edge[..., None]
                    lag = (path.prior + path.basis @ path.kernel).masked_fill(~path.allowed, -torch.inf)
                    bias = (lag if mod else lag.T).repeat(1, other_idx.shape[-1])
                    logits = path.query(query) @ key.transpose(-2, -1) / math.sqrt(z.shape[-1])
                    logits = (logits + bias).masked_fill(~allowed.flatten(2, 3)[:, :, None], -torch.inf)
                    w = F.pad(logits, (0, 1)).softmax(-1)[..., :-1]
                    message = message + path.gate * (w @ val)
            outputs.append(self.norm(query + message))
        return outputs
