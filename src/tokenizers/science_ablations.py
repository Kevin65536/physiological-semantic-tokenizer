"""Matched four-site EEG tokenizer ablations with frozen external Hb output."""

import torch
from torch import nn
from torch.nn import functional as F

from src.analysis.dual_component_tokenizer import eeg_spectral_attributes
from src.tokenizers.ema_vector_quantizer import EMAVectorQuantizer


def normalize_hb(h):
    h = h.reshape(*h.shape[:2], 300, 2)
    return ((h - h.mean(2, keepdim=True)) / h.std(2, keepdim=True, unbiased=False).clamp_min(1e-7)).reshape(
        *h.shape[:2], 15, 40
    )


def new_book(source, size):
    return EMAVectorQuantizer(
        size,
        source.embedding_dim,
        kmeans_init=True,
        kmeans_iters=20,
        revive_dead_codes=True,
        dead_code_threshold=0.1 * 64 * 15 / size,
        revival_warmup_steps=100,
        revival_interval=50,
        revival_stop_after_steps=1500,
        max_revivals_per_event=8,
        revival_strategy="diverse_farthest",
        revival_count_prior="uniform_batch",
        geometry_each_forward=False,
    )


@torch.no_grad()
def initialize(book, z):
    flat = z.reshape(-1, z.shape[-1])
    if len(flat) < book.codebook_size or book.initialized:
        raise ValueError("invalid train-only initialization")
    centers = flat[torch.randperm(len(flat), device=flat.device)[: book.codebook_size]].clone()
    for step in range(book.kmeans_iters + 1):
        labels = torch.cat([book._assignment_logits(chunk, centers).argmax(1) for chunk in flat.split(256)])
        counts = torch.bincount(labels, minlength=len(centers)).to(flat.dtype)
        sums = torch.zeros_like(centers).index_add_(0, labels, flat)
        if step < book.kmeans_iters:
            centers = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], centers)
    book.codebook.copy_(centers)
    book.ema_count.copy_(counts)
    book.ema_sum.copy_(centers * counts[:, None])
    book.initialized.fill_(True)


class ScienceTokenizer(nn.Module):
    def __init__(self, base, arm, sizes):
        super().__init__()
        self.base = base
        self.arm = arm
        self.extra = nn.ModuleDict()
        for b in "SO":
            branch = base.branches[b]
            q = branch["quantizer"]
            if sizes[0] != q.codebook_size:
                branch["quantizer"] = new_book(q, sizes[0])
            if len(sizes) == 2:
                self.extra[b] = new_book(q, sizes[1])
        if arm == "shared":
            base.branches["O"]["encoder"].layers[0] = base.branches["S"]["encoder"].layers[0]
        elif arm == "independent_size":
            # Reduce all four independent FFNs equally; approximately matches sharing one full block.
            for b in "SO":
                for layer in base.branches[b]["encoder"].layers:
                    width = layer.linear1.in_features
                    layer.linear1 = nn.Linear(width, width)
                    layer.linear2 = nn.Linear(width, width)
        if arm in ("neighbor", "shuffled_neighbor"):
            self.mix = nn.Parameter(torch.zeros(()))
            self.register_buffer(
                "adjacent", torch.tensor([1, 0, 3, 2] if arm == "neighbor" else [3, 2, 1, 0])
            )
        if arm.startswith("cross_"):
            self.cross = nn.ModuleDict({b: nn.Linear(40, 128, bias=False) for b in "SO"})
            for layer in self.cross.values():
                nn.init.zeros_(layer.weight)

    def encode(self, x, h):
        if hasattr(self, "mix"):
            x = x + self.mix * x[:, self.adjacent]
        if self.arm == "cross_hb_only":
            x = torch.zeros_like(x)
        if self.arm == "cross_eeg_only":
            h = torch.zeros_like(h)
        latent = self.base.encode_continuous(x.flatten(0, 1))
        if hasattr(self, "cross"):
            hb = normalize_hb(h).flatten(0, 1)
            latent = {b: z + self.cross[b](hb) for b, z in latent.items()}
        return latent

    def books(self, b):
        return [self.base.branches[b]["quantizer"]] + ([self.extra[b]] if b in self.extra else [])

    def decode(self, vectors):
        return {b: self.base._decode_component(b, z) for b, z in vectors.items()}

    def lookup(self, ids):
        return {
            b: sum(q.get_embedding(i) for q, i in zip(self.books(b), values, strict=True))
            for b, values in ids.items()
        }

    def forward(self, x, h, hard=True):
        latent = self.encode(x, h)
        vectors = {}
        ids = {}
        parts = {}
        commitment = x.new_zeros(())
        for b, z in latent.items():
            pieces = []
            indices = []
            if hard:
                diff = z
                for q in self.books(b):
                    if not bool(q.initialized):
                        raise ValueError("uninitialized codebook")
                    with torch.no_grad():
                        flat = diff.detach().reshape(-1, z.shape[-1])
                        idx = q._assignment_logits(flat).argmax(1)
                        value = q.get_embedding(idx).reshape_as(z)
                        if self.training:
                            q._ema_update(flat, idx)
                    pieces.append(value)
                    indices.append(idx.reshape(z.shape[:2]))
                    commitment = (
                        commitment + q.commitment_cost * F.mse_loss(diff, value) / len(self.books(b)) / 2
                    )
                    diff = diff - value
                v = sum(pieces)
                vectors[b] = z + (v - z).detach() if self.training else v
                parts[b] = pieces
            else:
                vectors[b] = z
            ids[b] = indices
        waves = self.decode(vectors)
        spec = vectors["S"]
        if self.arm == "dual13_first" and hard:
            z = latent["S"]
            spec = z + (parts["S"][0] - z).detach() if self.training else parts["S"][0]
        return dict(
            **waves,
            sum=waves["S"] + waves["O"],
            ids=ids,
            parts=parts,
            vectors=vectors,
            spectrum=self.base.spectral_head(spec),
            commitment=commitment
        )


def objective(out, x, s, variance, dvariance, scale, calibration, weights, attr):
    targets = (x, s, x - s)
    preds = (out["sum"], out["S"], out["O"])
    wave = []
    deriv = []
    for i, (pred, tar) in enumerate(zip(preds, targets)):
        err = pred.reshape_as(tar) - tar
        wave.append((err.square() / variance[i]).mean())
        deriv.append((err.flatten(2).diff(dim=-1).square() / dvariance[i].reshape(1, -1, 1)).mean())
    cost = (
        0.5 * wave[0]
        + 0.5 * (weights[0] * wave[1] + weights[1] * wave[2])
        + 0.1 * (deriv[0] + weights[0] * deriv[1] + weights[1] * deriv[2]) / 3
        + out["commitment"]
    )
    if attr:
        y = (eeg_spectral_attributes(s.flatten(0, 1), scale.repeat(len(x))) - calibration[0]) / calibration[1]
        cost = cost + attr * F.mse_loss(out["spectrum"], y)
    return cost
