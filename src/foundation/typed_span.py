"""Single exchange of branch-pure hard codes on paired native sensor sites."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def observed_view(raw, hidden, modality, gain):
    """Hide before any window statistic. Hb uses the visible part of its 5 s reference."""
    if raw.ndim != 4 or raw.shape[:-1] != hidden.shape or hidden.dtype != torch.bool:
        raise ValueError("expected [batch, site, slot, samples] and shared S/O mask")
    shown = raw.masked_fill(hidden[..., None], 0)
    if modality == 0:
        return shown
    if modality != 1 or raw.shape[-2:] != (15, 40):
        raise ValueError("expected 15 two-color Hb patches")
    colors = shown.reshape(*raw.shape[:2], 300, 2)
    valid = (~hidden).repeat_interleave(20, dim=-1)
    prefix = valid[..., :50]
    mean = colors[..., :50, :].sum(-2) / prefix.sum(-1, keepdim=True).clamp_min(1)
    fallback = colors.sum(-2) / valid.sum(-1, keepdim=True).clamp_min(1)
    ref = torch.where(prefix.any(-1, keepdim=True), mean, fallback)
    result = (colors - ref.unsqueeze(-2)) * gain.reshape(1, -1, 1, 1)
    return result.reshape_as(raw).masked_fill(hidden[..., None], 0)


def span_mask(start, length, sites):
    slots = torch.arange(15, device=start.device)
    if not 1 <= length <= 15 or (start < 0).any() or (start + length > 15).any():
        raise ValueError("span outside the recorded window")
    hide = (slots >= start[:, None]) & (slots < start[:, None] + length)
    return hide[:, None].expand(-1, sites, -1)


def view_codes(model, wave):
    """Use current hard assignments with STE, without updating EMA on corrupted windows."""
    flat = wave.flatten(0, 1)
    latents = model.encode_continuous(flat)
    values, ids = [], []
    for branch, latent in latents.items():
        book = model.branches[branch]["quantizer"]
        if book.training or not bool(book.initialized):
            raise ValueError("visible views require an initialized read-only dictionary")
        output = book(latent)
        ids.append(output.hard_ids.reshape(*wave.shape[:3]))
        values.append(output.quantized.reshape(*wave.shape[:3], -1))
    return values, ids


class LagPath(nn.Module):
    def __init__(self, width, support, physical):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.kernel = nn.Parameter(torch.zeros(5))
        # Oriented in seconds from EEG to Hb in both conditional directions.
        grid = torch.arange(15) * 2
        lag = grid[:, None] - grid
        self.register_buffer("allowed", (lag >= support[0]) & (lag <= support[1]))
        centers = torch.linspace(*support, 5)
        self.register_buffer("basis", torch.exp(-0.5 * ((lag[..., None] - centers) / 3) ** 2))
        self.register_buffer("prior", -0.5 * ((lag - 4) / 6) ** 2 if physical else torch.zeros_like(lag))

    def attention(self, target, source, to_hb, timed):
        q, k = self.query(target), self.key(source)
        score = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        if timed:
            bias = (self.prior + self.basis @ self.kernel).masked_fill(~self.allowed, -torch.inf)
            score = score + (bias if to_hb else bias.T)
        # One extra, zero-valued key allows abstention even with keys in range.
        return F.pad(score, (0, 1)).softmax(-1)

    def forward(self, target, source, to_hb, timed):
        weight = self.attention(target, source, to_hb, timed)[..., :-1]
        return self.gate * (weight @ self.value(source))


class TypedSpanReader(nn.Module):
    def __init__(self, vocabs, dim, width, supports):
        super().__init__()
        self.project = nn.ModuleList(nn.Linear(dim, width) for _ in vocabs)
        self.mask = nn.Parameter(torch.randn(4, width) * 0.02)
        self.stream = nn.Embedding(4, width)
        self.time = nn.Embedding(15, width)
        self.position = nn.Linear(2, width)
        self.context = nn.TransformerEncoderLayer(
            width, 4, 2 * width, dropout=0.0, batch_first=True, norm_first=True, activation="gelu"
        )
        self.paths = nn.ModuleList(LagPath(width, support, j == 0) for j, support in enumerate(supports))
        self.norm = nn.LayerNorm(width)
        self.output = nn.ModuleList(nn.Linear(width, k) for k in vocabs)

    def forward(self, source, target, xy, hidden, modality, visible, arm, *, inspect=False, source_xy=None):
        """Native-site pairs; source keys have no reader-side mixing across time or branches."""
        other = 1 - modality

        def token(value, stream, mask=None):
            z = self.project[stream](value)
            if mask is not None:
                z = torch.where(mask[..., None], self.mask[stream], z)
            return (
                z
                + self.stream.weight[stream]
                + self.time.weight
                + self.position(
                    source_xy if source_xy is not None and stream // 2 == other else xy[:, stream // 2]
                )[:, :, None]
            )

        keys = [token(value, 2 * other + j) for j, value in enumerate(source)]
        logits = []
        weights = {}
        for b, value in enumerate(target):
            stream = 2 * modality + b
            z = token(value, stream, hidden)
            query = self.context(z.flatten(0, 1)).reshape_as(z)
            message = torch.zeros_like(query)
            for a in visible:
                path = 2 * a + b if modality else 2 * b + a
                if arm != "ss_only" or path == 0:
                    link = self.paths[path]
                    if inspect:
                        w = link.attention(query, keys[a], bool(modality), arm != "no_lag")
                        weights[path] = w
                        message = message + link.gate * (w[..., :-1] @ link.value(keys[a]))
                    else:
                        message = message + link(query, keys[a], bool(modality), arm != "no_lag")
            logits.append(self.output[stream](self.norm(query + message)))
        return (logits, weights) if inspect else logits
