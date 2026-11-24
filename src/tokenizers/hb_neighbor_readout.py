"""Frozen low-rank cross-position Hb residual readout from complete integer arrays."""

from pathlib import Path
import json
import hashlib

import numpy as np


def decode_native(source, ids, missing_ids):
    if set(ids) != {"S", "O"}:
        raise ValueError("complete two-branch native integer arrays required")
    a = {b: np.asarray(ids[b]) for b in ("S", "O")}
    shape = a["S"].shape
    if (
        len(shape) != 2
        or shape[1] != 15
        or any(x.shape != shape or x.dtype.kind not in "iu" for x in a.values())
    ):
        raise ValueError("native-position by fifteen integer arrays required")
    missing = np.ones(shape[0], bool)
    any_missing = np.zeros(shape[0], bool)
    for b in a:
        if (a[b] < 0).any() or (a[b] > missing_ids[b]).any():
            raise ValueError("code outside frozen missing-inclusive vocabulary")
        missing &= (a[b] == missing_ids[b]).all(1)
        any_missing |= (a[b] == missing_ids[b]).any(1)
    if not np.array_equal(missing, any_missing):
        raise ValueError("missing position requires a complete repeated S/O pair")
    safe = {b: np.where(missing[:, None], 0, x) for b, x in a.items()}
    decoded = source.decode_ids(safe)
    values = np.stack(
        [decoded[b].reshape(shape[0], 300, 2) for b in ("S", "O")], axis=2
    )
    values[missing] = 0.0
    quiet = np.ones(shape[0], bool)
    for b in a:
        quiet &= (a[b] == source.silent_ids[b]).all(1)
    return values, missing, quiet


def apply_coefficients(values, missing, quiet, coefficient):
    """The residual depends on both branches and every native site in the same array."""
    values = np.asarray(values, float)
    n = values.shape[0]
    if values.shape != (n, 300, 2, 2) or np.shape(coefficient) != (n, 4 * n, 4):
        raise ValueError("frozen full-layout coefficient shape required")
    features = values.transpose(1, 0, 2, 3).reshape(300, 4 * n)
    correction = np.einsum("tf,pfo->pto", features, coefficient, optimize=True)
    correction[quiet] = 0.0
    result = values + correction.reshape(n, 300, 2, 2)
    # Missing sites have no target-admitted measurements. Their integer identity
    # stays explicit and their decoder output remains zero.
    result[missing] = 0.0
    return {
        "S": result[:, :, 0].reshape(n, 15, 40),
        "O": result[:, :, 1].reshape(n, 15, 40),
        "sum": (result[:, :, 0] + result[:, :, 1]).reshape(n, 15, 40),
    }


def fit_site(gram, cross, variances, site, rank, ridge, sum_weight=2.0):
    """Fit train-only normalized local features and an orthogonal other-site subspace."""
    width = len(gram)
    own = np.arange(4 * site, 4 * site + 4)
    other = np.setdiff1d(np.arange(width), own)
    rms = np.sqrt(np.diag(gram).clip(1e-16))
    standardized = gram / rms[:, None] / rms[None, :]
    matrix = np.eye(width)[:, own] / rms[:, None]
    if rank:
        spatial = standardized[np.ix_(other, other)]
        eigenvalue, eigenvector = np.linalg.eigh((spatial + spatial.T) * 0.5)
        eigenvalue = eigenvalue[-rank:].clip(1e-12)
        virtual = np.zeros((width, rank))
        virtual[other] = eigenvector[:, -rank:] / np.sqrt(eigenvalue)[None, :]
        matrix = np.concatenate([matrix, virtual / rms[:, None]], axis=1)
    g = matrix.T @ gram @ matrix
    c = matrix.T @ cross
    coefficients = np.zeros((width, 4))
    for color in range(2):
        # S/O target variances and X variance belong to the frozen G0 coordinate.
        vs, vo, vx = variances[1, color], variances[2, color], variances[0, color]
        h = np.array(
            [
                [1 / vs + sum_weight / vx, sum_weight / vx],
                [sum_weight / vx, 1 / vo + sum_weight / vx],
            ]
        )
        coupling = np.kron(h, g)
        penalty = ridge * max(float(np.trace(coupling) / len(coupling)), 1e-12)
        coupling += penalty * np.eye(len(coupling))
        cross_pair = np.stack([c[:, color], c[:, 2 + color]])
        target = (h @ cross_pair).ravel()
        weights = np.linalg.solve(coupling, target).reshape(2, -1)
        coefficients[:, color] = matrix @ weights[0]
        coefficients[:, 2 + color] = matrix @ weights[1]
    return coefficients


class HbNeighborReadout:
    def __init__(self, source, metadata, coefficients):
        self.source, self.metadata, self.coefficients = source, metadata, coefficients
        self.missing_ids = metadata["missing_ids"]

    def encode(self, values, dataset, observed):
        n = len(self.metadata["positions"][dataset])
        if np.shape(values) != (n, 15, 40) or np.shape(observed) != (n,):
            raise ValueError("complete declared native Hb layout required")
        ids = self.source.encode(np.asarray(values, np.float32))
        for b in ids:
            ids[b][~np.asarray(observed, bool)] = self.missing_ids[b]
        return ids

    def decode_ids(self, ids, dataset):
        n = len(self.metadata["positions"][dataset])
        if len(ids["S"]) != n:
            raise ValueError("complete fixed Hb native layout required")
        values, missing, quiet = decode_native(self.source, ids, self.missing_ids)
        return apply_coefficients(values, missing, quiet, self.coefficients[dataset])

    def save(self, path):
        payload = {"metadata": np.asarray(json.dumps(self.metadata))}
        payload.update({"coefficient_" + d: v for d, v in self.coefficients.items()})
        np.savez_compressed(path, **payload)


def load_model(path):
    from .contextual_latent_dictionary import load_model as load_source

    path = Path(path)
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        if metadata["schema"] != "hb_neighbor_readout_v1":
            raise ValueError("frozen neighbor Hb readout required")
        source = Path(metadata["source_checkpoint"])
        with source.open("rb") as stream:
            if (
                hashlib.file_digest(stream, "sha256").hexdigest()
                != metadata["source_sha256"]
            ):
                raise ValueError("frozen Hb source changed")
        coefficients = {d: saved["coefficient_" + d] for d in metadata["positions"]}
    return HbNeighborReadout(load_source(source), metadata, coefficients)
