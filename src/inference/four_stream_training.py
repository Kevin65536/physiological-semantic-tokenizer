"""Paired stochastic fits of the future-attribute readers on a prevalidated event panel."""

import numpy as np
import torch

from src.foundation.four_stream_relations import (
    ComponentMainEffect,
    ComponentReadout,
    HbHistoryBaseline,
    ResponseResidual,
)


def training_tensors(data, meta):
    train = meta.split.eq("train").to_numpy()
    tensors = {}
    normalizers = {}
    for mod in ("eeg", "hb"):
        y = data[mod + "_target"]
        mean = y[train].mean(axis=0, dtype=float)
        sd = y[train].std(axis=0, dtype=float)
        if np.any(sd <= 1e-9) or not np.isfinite(y).all():
            raise ValueError("nonfinite or degenerate training endpoints")
        tensors[mod + "_target"] = torch.tensor((y - mean) / sd, dtype=torch.float32).flatten(2)
        normalizers[mod] = dict(mean=mean.tolist(), sd=sd.tolist())
        tensors[mod + "_books"] = [torch.tensor(data[mod + b + "_book"]) for b in "SO"]
        for key in ["_ids", "_alt"]:
            tensors[mod + key] = torch.tensor(data[mod + key].astype("int64"))
    tensors["xy"] = torch.tensor(data["xy"], dtype=torch.float32)
    tensors["donor"] = torch.tensor(meta.donor.to_numpy(), dtype=torch.long)
    tensors["train"] = np.flatnonzero(train)
    tensors["dev"] = np.flatnonzero(~train)
    # Each training person gets equal expected exposure, regardless of eligible event count.
    people = meta.loc[train, "subject"]
    weight = 1 / people.map(people.value_counts()).to_numpy()
    tensors["weights"] = weight / weight.sum()
    return tensors, normalizers


def make_model(arm, tensors, width, rank):
    hb = tensors["hb_books"]
    if arm == "A_base":
        return HbHistoryBaseline(hb, width, 3)
    if arm.startswith("A_"):
        return ResponseResidual(hb, hb if arm == "A_H" else tensors["eeg_books"], width, 3)
    mod, kind = arm.split("_", 1)
    books = tensors[mod + "_books"]
    outputs = tensors[mod + "_target"].shape[-1]
    if kind in ("S", "O"):
        return ComponentMainEffect(books["SO".index(kind)], width, outputs)
    return ComponentReadout(
        books, width, outputs, rank=0 if kind in ("add", "alt_add") else rank, generic=kind == "generic"
    )


def predictions(model, arm, t, idx, base=None, swap=False):
    donor = t["donor"][idx] if swap else idx
    hx = t["xy"][idx, 1]
    if arm.startswith("A_"):
        h = t["hb_ids"][idx]
        if arm == "A_base":
            return model(h, hx).flatten(2)
        own = arm == "A_H"
        j = idx if own else donor
        src = t["hb_ids" if own else "eeg_ids"][j]
        delta, _ = model(h, hx, src, t["xy"][j, int(own)], uniform_pool=arm == "A_generic")
        return base[idx] + delta.flatten(2)
    mod, kind = arm.split("_", 1)
    ids = t[mod + ("_alt" if kind.startswith("alt_") else "_ids")]
    xy = t["xy"][idx, int(mod == "hb")]
    if kind in ("S", "O"):
        return model(ids[idx, "SO".index(kind)], xy)[0]
    pair = torch.stack([ids[idx, 0], ids[donor, 1]], 1)
    return model(pair, xy)[0]


def fit_reader(arm, seed, t, cfg, base=None, *, steps=None):
    torch.manual_seed(seed)
    model = make_model(arm, t, cfg["width"], cfg["rank"])
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg["lr"], weight_decay=0.01
    )
    rng = np.random.default_rng(seed)
    y = t[("hb" if arm.startswith("A_") else arm.split("_")[0]) + "_target"]
    curves = []
    for i in range(steps if steps is not None else cfg["steps"]):
        idx = rng.choice(t["train"], cfg["batch"], p=t["weights"])
        pred = predictions(model, arm, t, idx, base, swap=arm.endswith("_wrong"))
        loss = (pred - y[idx]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        curves.append(loss.item())
    model.eval()
    return model, curves
