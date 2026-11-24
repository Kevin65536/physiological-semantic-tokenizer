"""Strongly shrunk additive integer tables for a frozen HbO sum residual."""

from pathlib import Path
import hashlib

import numpy as np
from scipy import sparse
from scipy.signal import butter, sosfiltfilt
import torch

from .contextual_latent_dictionary import load_model as load_source
from .hb_parent_distillation import validate_ids, preserve_O_mix_HbO


def smooth_basis():
    t = np.arange(20) + 0.5
    basis = np.cos(np.pi * t[:, None] * np.arange(3)[None] / 20)
    return basis / np.sqrt(np.mean(basis * basis, axis=0))


def supported_pairs(
    ids, complete, subjects, minimum=128, minimum_subjects=4, maximum=256
):
    arrays, quiet = validate_ids(ids)
    key = arrays["S"] * 256 + arrays["O"]
    valid = np.asarray(complete, bool) & ~quiet[:, None]
    counts = np.bincount(key[valid], minlength=32768)
    covered = np.zeros(32768, int)
    for subject in np.unique(subjects):
        covered[np.unique(key[valid & (np.asarray(subjects) == subject)[:, None]])] += 1
    eligible = np.flatnonzero((counts >= minimum) & (covered >= minimum_subjects))
    order = np.lexsort((eligible, -counts[eligible]))[:maximum]
    return np.sort(eligible[order])


def id_design(ids, contextual, pairs):
    arrays, quiet = validate_ids(ids)
    s = np.where(quiet[:, None], 0, arrays["S"])
    o = np.where(quiet[:, None], 0, arrays["O"])
    fields = [(s, 129), (o, 257)]
    if contextual:
        for offset in [-1, 1]:
            for values, size in [(s, 129), (o, 257)]:
                pad = (1, 0) if offset == -1 else (0, 1)
                padded = np.pad(values, ((0, 0), pad), constant_values=size - 1)
                fields.append((padded[:, :15] if offset == -1 else padded[:, 1:], size))
    n = s.size
    row = []
    column = []
    width = 0
    for values, size in fields:
        row.append(np.arange(n))
        column.append(values.ravel() + width)
        width += size
    pairs = np.asarray(pairs, np.int64)
    if len(pairs):
        key = (s * 256 + o).ravel()
        index = np.searchsorted(pairs, key)
        match = (index < len(pairs)) & (pairs[index.clip(max=len(pairs) - 1)] == key)
        row.append(np.flatnonzero(match))
        column.append(width + index[match])
    matrix = sparse.csr_matrix(
        (
            np.ones(sum(len(v) for v in row)),
            (np.concatenate(row), np.concatenate(column)),
        ),
        shape=(n, width + len(pairs)),
    )
    penalty = np.ones(matrix.shape[1])
    penalty[width:] = 4
    return matrix, penalty


def waveform_design(prediction):
    basis = smooth_basis()
    values = np.stack(
        [np.asarray(prediction[b]).reshape(-1, 15, 20, 2) for b in "SO"], axis=2
    )
    features = (
        np.einsum("npbtc,tk->npbck", values, basis).reshape(len(values), 15, -1) / 20
    )
    previous = np.pad(features, ((0, 0), (1, 0), (0, 0)), mode="edge")[:, :15]
    following = np.pad(features, ((0, 0), (0, 1), (0, 0)), mode="edge")[:, 1:]
    return np.concatenate(
        [features, previous, following, np.ones((*features.shape[:2], 1))], axis=-1
    ).reshape(-1, 37)


def fit_head(
    ids,
    prediction,
    residual_coefficients,
    weights,
    complete,
    subjects,
    family,
    ridge,
    settings,
):
    if family not in ("decoded_waveform", "current_IDs", "context_IDs"):
        raise ValueError("one frozen native-ID readout family required")
    _, quiet = validate_ids(ids)
    use = np.asarray(complete, bool) & ~quiet[:, None]
    w = np.asarray(weights, float).ravel()[use.ravel()]
    if not len(w) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("positive train-only G0 hierarchy/variance weights required")
    pairs = np.array([], np.int64)
    if family == "decoded_waveform":
        full = waveform_design(prediction)
        matrix = full[use.ravel()]
        scale = np.sqrt(np.einsum("n,nf->f", w / w.sum(), matrix * matrix)).clip(1e-10)
        matrix = matrix / scale
        penalty = np.ones(matrix.shape[1])
    else:
        pairs = supported_pairs(
            ids,
            use,
            subjects,
            settings["minimum_pair_patches"],
            settings["minimum_pair_subjects"],
            settings["maximum_pairs"],
        )
        full, penalty = id_design(ids, family == "context_IDs", pairs)
        matrix = full[use.ravel()]
        scale = np.ones(matrix.shape[1])
    y = np.asarray(residual_coefficients).reshape(-1, 3)[use.ravel()]
    if sparse.issparse(matrix):
        gram = (matrix.T @ matrix.multiply(w[:, None])).toarray()
        right = matrix.T @ (w[:, None] * y)
    else:
        gram = matrix.T @ (w[:, None] * matrix)
        right = matrix.T @ (w[:, None] * y)
    coefficient = (
        np.linalg.solve(gram + ridge * np.diag(penalty), right) / scale[:, None]
    )
    return dict(
        family=family,
        ridge=float(ridge),
        pairs=pairs,
        coefficient=coefficient,
        scalar_coefficients=int(coefficient.size),
        pair_count=len(pairs),
        training_patches=int(use.sum()),
    )


def residual_from_ids(ids, prediction, fitted):
    arrays, quiet = validate_ids(ids)
    if fitted["family"] == "decoded_waveform":
        design = waveform_design(prediction)
    else:
        design, _ = id_design(ids, fitted["family"] == "context_IDs", fitted["pairs"])
    coefficient = np.asarray(design @ fitted["coefficient"]).reshape(
        len(arrays["S"]), 15, 3
    )
    residual = np.einsum("npk,tk->npt", coefficient, smooth_basis()).reshape(-1, 300)
    residual = sosfiltfilt(butter(4, 0.4, fs=10, output="sos"), residual, axis=1)
    residual[quiet] = 0
    return residual


def apply_head(ids, prediction, fitted):
    _, quiet = validate_ids(ids)
    total = np.asarray(prediction["sum"], float).reshape(-1, 300, 2).copy()
    total[:, :, 0] += residual_from_ids(ids, prediction, fitted)
    desired = dict(sum=total.reshape(prediction["sum"].shape))
    return preserve_O_mix_HbO(prediction, desired, quiet, 1)


class HbNativeIDReadout:
    def __init__(self, source, metadata, fitted):
        self.source, self.metadata, self.fitted = source, metadata, fitted

    def encode(self, values):
        return self.source.encode(values)

    def decode_ids(self, ids):
        prediction = self.source.decode_ids(ids)
        return apply_head(ids, prediction, self.fitted)


def load_model(path):
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    meta = saved["metadata"]
    if meta["schema"] != "hb_native_id_readout_v1":
        raise ValueError("frozen native-ID readout schema required")
    file = (path.parent / meta["source_file"]).resolve()
    if (
        file.parent != path.parent.resolve()
        or hashlib.sha256(file.read_bytes()).hexdigest() != meta["source_sha256"]
    ):
        raise ValueError("accompanying frozen semantic source changed")
    return HbNativeIDReadout(load_source(file), meta, saved["fitted"])
