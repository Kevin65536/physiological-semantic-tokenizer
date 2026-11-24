"""Paired S/O spatial reconstruction from one paid band waveform per branch and site."""

import numpy as np
from scipy.linalg import solve_triangular

from .band_exchange_words import ATOMS, encode_slots, paid_coordinates


def layouts(observed):
    sites = np.flatnonzero(observed)
    native = np.full(len(observed), -1, dtype=np.int8)
    native[sites] = sites % 5
    rotated = native.copy()
    rotated[sites] = np.roll(native[sites], 2)
    return {"aligned": {"S": native, "O": native}, "staggered": {"S": native, "O": rotated}}


def encode_role(fields, role, properties, books):
    sent = np.zeros((len(role), 15), dtype=np.uint16)
    selected = np.zeros((*sent.shape, 400), dtype=np.float64)
    for band, book in enumerate(books):
        take = role == band
        if take.any():
            actual = fields[band, take]
            selected[take] = actual
            sent[take] = encode_slots(
                actual.reshape(-1, 400).astype(np.float32),
                properties[take].reshape(-1, 3),
                band,
                book,
                "balanced",
            ).reshape(-1, 15)
    return sent, selected


def covariance_blocks(s, o, cross):
    """Keep the fitted marginal covariances fixed for an identifiable matched comparison."""
    n = s.shape[1]
    separate = np.zeros((len(s), 2 * n, 2 * n))
    separate[:, :n, :n] = s
    separate[:, n:, n:] = o
    joint = separate.copy()
    canonical = []
    for band, (left, right, xy) in enumerate(zip(s, o, cross)):
        ls, lo = np.linalg.cholesky(left), np.linalg.cholesky(right)
        whitened = solve_triangular(ls, xy, lower=True)
        whitened = solve_triangular(lo, whitened.T, lower=True).T
        u, d, v = np.linalg.svd(whitened, full_matrices=False)
        cross_safe = ls @ ((u * np.minimum(d, 0.995)) @ v) @ lo.T
        joint[band, :n, n:] = cross_safe
        joint[band, n:, :n] = cross_safe.T
        canonical.append(d)
    return separate, joint, np.stack(canonical)


def calibration(moment):
    """Estimate q=g*x+e from training squared source, cross, squared word and mass."""
    xx, xq, qq, mass = moment
    valid = (xx > 1e-16) & (mass > 0)
    g = np.divide(xq, xx, out=np.ones_like(xq), where=valid)
    noise = np.divide(np.maximum(qq - g * xq, 0), mass, out=np.zeros_like(qq), where=valid)
    return g, noise


def reconstruct(coordinates, role, covariance, gain, noise):
    """Predict both full native arrays using only the anchors designated by the paid words."""
    n = len(role) // 2
    if (
        coordinates.shape != (2 * n, 15, 400)
        or covariance.shape != (5, 2 * n, 2 * n)
        or gain.shape != noise.shape
        or gain.shape != (5, 2 * n)
    ):
        raise ValueError("paired native frequency anchors and aligned fitted state required")
    result = np.zeros((2 * n, 6000), dtype=np.float64)
    for band, cov in enumerate(covariance):
        take = np.flatnonzero(role == band)
        if not len(take):
            continue
        g = gain[band, take]
        measured = g[:, None] * cov[np.ix_(take, take)] * g[None, :]
        diagonal = np.diag_indices(len(take))
        measured[diagonal] += noise[band, take]
        # Fit can contain a measured zero-energy coordinate. Its contribution is zero.
        informative = np.diag(measured) > 1e-20
        if informative.any():
            selected = take[informative]
            mapping = np.linalg.solve(
                measured[np.ix_(informative, informative)], g[informative, None] * cov[selected]
            )
            result += mapping.T @ coordinates[selected].reshape(len(selected), 6000)
    spectrum = np.fft.rfft(result)
    freq = np.fft.rfftfreq(6000, 1 / 200)
    spectrum[:, (freq < 1) | (freq > 45)] = 0
    s, o = np.fft.irfft(spectrum, n=6000).reshape(2, n, 15, 400)
    return {"S": s, "O": o, "sum": s + o}


def decode_words(sent, books, covariance, gain, noise):
    blocks, roles = [], []
    for b in "SO":
        block, tags = paid_coordinates(sent[b], books[b], "balanced", True)
        if not np.all(tags == tags[:, :1]):
            raise ValueError("fixed layout over this 30-second native array required")
        roles.append(tags[:, 0])
        blocks.append(block)
    if not np.array_equal(roles[0] < 0, roles[1] < 0):
        raise ValueError("paired native sites have inconsistent missing observations")
    return reconstruct(np.concatenate(blocks), np.concatenate(roles), covariance, gain, noise)
