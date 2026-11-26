"""Future-attribute reader prototypes for complete, cutoff-isolated frozen code windows."""

import torch
from torch import nn


def check_history_support(encoder_start, raw_stop, cutoff, target_start, target_stop):
    """Half-open intervals on one event-relative clock; raw_stop includes all preprocessing.

    All times use seconds. Each row has a separate
    EEG and Hb encoder start/raw stop, a shared decision time, and target horizons.
    """
    if (
        encoder_start.ndim != 2
        or encoder_start.shape[1] != 2
        or raw_stop.shape != encoder_start.shape
        or cutoff.shape != encoder_start[:, 0].shape
        or target_start.ndim != 2
        or target_start.shape != target_stop.shape
        or target_start.shape[0] != len(cutoff)
        or target_start.shape[1] == 0
    ):
        raise ValueError("expected paired native-clock histories and explicit target horizons")
    if not all(x.isfinite().all() for x in (encoder_start, raw_stop, cutoff, target_start, target_stop)):
        raise ValueError("unknown time support")
    if (
        (encoder_start + 30 > raw_stop + 1e-9).any()
        or (raw_stop > cutoff[:, None] + 1e-9).any()
        or (target_start < cutoff[:, None]).any()
        or (target_stop <= target_start).any()
    ):
        raise ValueError("encoder context, preprocessing or future horizon violates cutoff")


class StreamHistory(nn.Module):
    """Encode one stream per site with a learned low-order kernel on 15 past patch slots."""

    def __init__(self, book, width):
        super().__init__()
        if book.ndim != 2 or book.shape[1] != 128:
            raise ValueError("expected 128-dimensional frozen words")
        self.words = nn.Embedding.from_pretrained(book.detach().clone(), freeze=True)
        self.project = nn.Linear(128, width)
        self.geometry = nn.Linear(2, width)
        self.lags = nn.Parameter(torch.zeros(15, width))

    def forward(self, ids, xy):
        if ids.ndim != 3 or ids.shape[-1] != 15 or xy.shape != (*ids.shape[:2], 2):
            raise ValueError("expected complete site histories and matching sensor positions")
        if not xy.isfinite().all():
            raise ValueError("missing site geometry")
        if ids.is_floating_point() or ids.is_complex() or ids.dtype == torch.bool:
            raise ValueError("integer token IDs required")
        patches = torch.tanh(self.project(self.words(ids.long())))
        return (self.lags.softmax(0) * patches).sum(-2) + self.geometry(xy)


def split_pair(ids):
    if ids.ndim != 4 or ids.shape[1] != 2:
        raise ValueError("expected exactly one modality with S and O IDs")
    return ids.unbind(1)


class HbHistoryBaseline(nn.Module):
    """The shared b_H fit predicts both chromophores at each future horizon."""

    def __init__(self, hb_books, width, horizons):
        super().__init__()
        self.s, self.o = [StreamHistory(book, width) for book in hb_books]
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 2 * horizons))

    def forward(self, ids, xy):
        s, o = split_pair(ids)
        z = torch.cat([self.s(s, xy), self.o(o, xy)], dim=-1)
        return self.head(z).unflatten(-1, (-1, 2))


class ResponseResidual(nn.Module):
    """Same g architecture with either Hb or EEG as the source, Hb as the query.

    Supply independent instances with different source books for the two fits.
    Add each residual to the same frozen, previously fitted HbHistoryBaseline.
    """

    def __init__(self, hb_books, source_books, width, horizons):
        super().__init__()
        if len(hb_books) != 2 or len(source_books) != 2:
            raise ValueError("expected separate S and O vocabularies for each modality")
        self.hb = nn.ModuleList(StreamHistory(book, width) for book in hb_books)
        self.source = nn.ModuleList(StreamHistory(book, width) for book in source_books)
        self.output = nn.Linear(width, 2 * horizons)

    def forward(self, hb_ids, hb_xy, source_ids, source_xy, *, uniform_pool=False):
        # Query both target branches against both source branches at every source site.
        if hb_ids.shape[0] != source_ids.shape[0]:
            raise ValueError("target and source histories must have paired rows")
        query = torch.stack([fn(ids, hb_xy) for fn, ids in zip(self.hb, split_pair(hb_ids))], dim=2)
        keys = torch.cat([fn(ids, source_xy) for fn, ids in zip(self.source, split_pair(source_ids))], dim=1)
        if uniform_pool:
            context = keys.mean(1)[:, None, None]
        else:
            scores = torch.einsum("bpjd,bkd->bpjk", query, keys) / keys.shape[-1] ** 0.5
            context = torch.einsum("bpjk,bkd->bpjd", scores.softmax(-1), keys)
        relation = torch.tanh(query + context).mean(2)
        return self.output(relation).unflatten(-1, (-1, 2)), relation


class ComponentMainEffect(nn.Module):
    """Standalone S-only or O-only comparator, also used in the additive fit."""

    def __init__(self, book, width, outputs):
        super().__init__()
        self.encoder = StreamHistory(book, width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, outputs))

    def forward(self, ids, xy):
        z = self.encoder(ids, xy)
        return self.head(z), z


class ComponentReadout(nn.Module):
    """Independent nonlinear branch main effects with an optional low-rank interaction."""

    def __init__(self, books, width, outputs, *, rank=0, generic=False):
        super().__init__()
        self.s, self.o = [ComponentMainEffect(book, width, outputs) for book in books]
        if rank < 0:
            raise ValueError("interaction rank must be nonnegative")
        self.generic = generic
        self.rank = rank
        if rank:
            self.left = nn.Linear(width, rank, bias=False)
            self.right = nn.Linear(width, rank, bias=False)
            self.interaction = nn.Linear(rank, outputs, bias=False)

    def forward(self, ids, xy):
        s_ids, o_ids = split_pair(ids)
        main_s, s = self.s(s_ids, xy)
        main_o, o = self.o(o_ids, xy)
        prediction = main_s + main_o
        features = torch.cat([s, o], -1)
        if self.rank:
            left, right = self.left(s), self.right(o)
            joint = torch.tanh(left + right) if self.generic else left * right
            prediction = prediction + self.interaction(joint)
            features = torch.cat([features, joint], -1)
        return prediction, features
