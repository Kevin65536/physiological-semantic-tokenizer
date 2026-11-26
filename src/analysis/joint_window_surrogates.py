"""Conditional transition probabilities on complete disjoint whole-window rotations."""

import numpy as np


def joint_counts(s, o):
    s = np.asarray(s)
    o = np.asarray(o)
    if s.shape != o.shape or s.ndim != 2 or s.shape[1] != 15 or np.any((s < 0) | (s > 3) | (o < 0) | (o > 3)):
        raise ValueError("expected paired 15-slot quartile sequences")
    j = s * 4 + o
    return np.r_[
        np.bincount(j.ravel(), minlength=16), np.bincount((j[:, :-1] * 16 + j[:, 1:]).ravel(), minlength=256)
    ]


def conditional(counts):
    rows = np.asarray(counts)[..., 16:].reshape(*counts.shape[:-1], 16, 16)
    total = rows.sum(-1, keepdims=True)
    return np.divide(rows, total, out=np.full(rows.shape, np.nan), where=total > 0)


def rotations(starts, seconds=30):
    t = np.asarray(starts)
    return [k for k in range(1, len(t)) if np.all(np.abs(t - np.roll(t, k)) >= seconds - 1e-6)]
