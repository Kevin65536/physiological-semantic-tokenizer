"""Infer a frozen G1 parent from unchanged S129/O257 integers."""

from pathlib import Path
import hashlib

import numpy as np
from scipy.special import logsumexp
import torch

from .contextual_latent_dictionary import load_model as load_current
from .hb_parent_leaf import load_parent


def validate_ids(ids):
    if set(ids) != {"S", "O"}:
        raise ValueError("exactly the emitted S129/O257 streams required")
    arrays = {b: np.asarray(ids[b]) for b in "SO"}
    shape = arrays["S"].shape
    if len(shape) != 2 or shape[1] != 15:
        raise ValueError("one integer per two-second patch in a 30-second context")
    for b, size in [("S", 129), ("O", 257)]:
        if (
            arrays[b].shape != shape
            or arrays[b].dtype.kind not in "iu"
            or (arrays[b] < 0).any()
            or (arrays[b] >= size).any()
        ):
            raise ValueError("matched in-range integer streams required")
    quiet = (arrays["S"] == 128).all(1) & (arrays["O"] == 256).all(1)
    reserved = (arrays["S"] == 128).any(1) | (arrays["O"] == 256).any(1)
    if not np.array_equal(quiet, reserved):
        raise ValueError("reserved IDs require the complete paired quiet sequence")
    return arrays, quiet


def categorical_features(ids):
    ids, quiet = validate_ids(ids)
    s = np.where(quiet[:, None], 0, ids["S"])
    o = np.where(quiet[:, None], 0, ids["O"])
    previous = np.pad(o, ((0, 0), (1, 0)), constant_values=256)[:, :15]
    following = np.pad(o, ((0, 0), (0, 1)), constant_values=256)[:, 1:]
    return (o, s, previous, following), quiet


def fit_tables(ids, parent_O, token_weights, shrinkage):
    """Weighted categorical counts, with a global Dirichlet backoff."""
    fields, quiet = categorical_features(ids)
    target = np.asarray(parent_O)
    weights = np.asarray(token_weights, float)
    if (
        target.shape != fields[0].shape
        or target.dtype.kind not in "iu"
        or (target < 0).any()
        or (target >= 128).any()
        or weights.shape != target.shape
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or shrinkage <= 0
    ):
        raise ValueError(
            "train-only hard parent labels and nonnegative token weights required"
        )
    keep = (~quiet[:, None] & (weights > 0)).ravel()
    w = weights.ravel()[keep]
    if not len(w):
        raise ValueError("ordinary supported training tokens required")
    w = w / w.sum() * len(w)
    y = target.ravel()[keep]
    marginal = np.bincount(y, weights=w, minlength=128) + 1e-3
    prior = marginal / marginal.sum()
    tables = []
    supports = []
    for x, size in zip(fields, [256, 128, 257, 257]):
        counts = np.zeros((size, 128))
        np.add.at(counts, (x.ravel()[keep], y), w)
        support = counts.sum(1)
        posterior = (counts + shrinkage * prior) / (support[:, None] + shrinkage)
        tables.append(np.log(posterior.clip(1e-300)))
        supports.append(support)
    # Context uses P(parent|context)/P(parent), then adds a fixed small strength.
    return dict(
        prior=prior,
        log_tables=tables,
        support=supports,
        shrinkage=float(shrinkage),
        weighted_training_tokens=float(w.sum()),
    )


def parent_probabilities(ids, fitted, context_strength):
    fields, _ = categorical_features(ids)
    if context_strength not in (0, 0.25, 0.5, 1.0):
        raise ValueError("frozen declared categorical context strength required")
    table = fitted["log_tables"]
    log_prior = np.log(np.asarray(fitted["prior"]).clip(1e-300))
    logits = table[0][fields[0]].copy()
    if context_strength:
        s = (table[1][fields[1]] - log_prior).clip(-4, 4)
        before = (table[2][fields[2]] - log_prior).clip(-4, 4)
        after = (table[3][fields[3]] - log_prior).clip(-4, 4)
        logits += context_strength * (s + (before + after) * 0.5)
    return np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))


def decode_parent(parent, ids, probability, mode):
    """All continuous decoder inputs are deterministic functions of emitted IDs."""
    arrays, quiet = validate_ids(ids)
    probability = np.asarray(probability)
    if (
        probability.shape != (*arrays["S"].shape, 128)
        or not np.isfinite(probability).all()
        or (probability < 0).any()
        or not np.allclose(probability.sum(-1), 1, atol=1e-10)
    ):
        raise ValueError("finite normalized frozen parent probabilities required")
    with torch.inference_mode():
        s = torch.as_tensor(np.where(quiet[:, None], 0, arrays["S"]), dtype=torch.long)
        quantizer = parent.model.branches["O"]["quantizer"]
        if mode == "hard":
            o = quantizer.get_embedding(torch.as_tensor(probability.argmax(-1)))
        elif mode == "expected_embedding":
            o = torch.as_tensor(probability, dtype=torch.float32) @ quantizer.codebook
        else:
            raise ValueError("hard parent or expected quantized embedding required")
        s_latent = parent.model.branches["S"]["quantizer"].get_embedding(s)
        result = dict(
            S=parent.model._decode_component("S", s_latent).numpy(),
            O=parent.model._decode_component("O", o).numpy(),
        )
    result["sum"] = result["S"] + result["O"]
    return result


def preserve_O_mix_HbO(current, predicted_parent, quiet, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("convex frozen HbO sum mixture required")
    result = {b: np.asarray(current[b]).copy() for b in ("S", "O", "sum")}
    total = np.asarray(current["sum"], float).reshape(-1, 300, 2).copy()
    before = total.copy()
    parent = np.asarray(predicted_parent["sum"], float).reshape(total.shape)
    total[:, :, 0] += alpha * (parent[:, :, 0] - total[:, :, 0])
    total[np.asarray(quiet)] = before[np.asarray(quiet)]
    physical = np.asarray(current["S"], float).reshape(total.shape).copy()
    physical[:, :, 0] = (
        total[:, :, 0] - np.asarray(current["O"]).reshape(total.shape)[:, :, 0]
    )
    physical[np.asarray(quiet)] = np.asarray(current["S"]).reshape(total.shape)[
        np.asarray(quiet)
    ]
    result["sum"] = total.reshape(result["sum"].shape)
    result["S"] = physical.reshape(result["S"].shape)
    np.testing.assert_array_equal(result["O"], current["O"])
    for b in ("S", "O", "sum"):
        np.testing.assert_array_equal(result[b][quiet], current[b][quiet])
        np.testing.assert_array_equal(
            result[b].reshape(total.shape)[:, :, 1],
            np.asarray(current[b]).reshape(total.shape)[:, :, 1],
        )
    np.testing.assert_allclose(
        result["S"] + result["O"], result["sum"], rtol=1e-12, atol=1e-12
    )
    return result


class HbParentDistillation:
    def __init__(self, current, parent, metadata, fitted):
        self.source, self.parent, self.metadata, self.fitted = (
            current,
            parent,
            metadata,
            fitted,
        )

    def encode(self, values):
        return self.source.encode(values)

    def decode_ids(self, ids):
        _, quiet = validate_ids(ids)
        current = self.source.decode_ids(ids)
        p = parent_probabilities(ids, self.fitted, self.metadata["context_strength"])
        predicted = decode_parent(self.parent, ids, p, self.metadata["parent_mode"])
        return preserve_O_mix_HbO(current, predicted, quiet, self.metadata["alpha"])


def load_model(path):
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    meta = saved["metadata"]
    if meta["schema"] != "hb_parent_distillation_v1":
        raise ValueError("versioned frozen Hb parent distillation required")
    files = {}
    for name in ("current", "parent"):
        file = (path.parent / meta[name + "_file"]).resolve()
        if (
            file.parent != path.parent.resolve()
            or hashlib.sha256(file.read_bytes()).hexdigest() != meta[name + "_sha256"]
        ):
            raise ValueError("accompanying frozen source changed")
        files[name] = file
    parent = load_parent(
        files["parent"], saved["parent_setting"], saved["parent_metadata"]
    )
    return HbParentDistillation(
        load_current(files["current"]), parent, meta, saved["fitted"]
    )
