"""Person-grouped, train-only selection for same-support representation comparisons."""

import numpy as np
from scipy.linalg import eigh


def grouped_folds(people, train, n=3):
    train = np.asarray(train)
    names = np.unique(people[train])
    if len(names) < n:
        raise ValueError("insufficient training people")
    # Deterministic assignment independent of labels or fitted representations.
    return [
        (train[~np.isin(people[train], names[k::n])], train[np.isin(people[train], names[k::n])])
        for k in range(n)
    ]


def standardized_ridge(x, y, fit, score, alphas):
    """Dual ridge; ridge constant is relative to average training kernel diagonal."""
    x = x.reshape(len(x), -1).astype(float)
    ref = x[fit]
    mu = ref.mean(0)
    scale = ref.std(0)
    scale[scale < 1e-8] = 1.0
    ref = (ref - mu) / scale
    cross = ((x[score] - mu) / scale) @ ref.T / ref.shape[1]
    kval, u = eigh(ref @ ref.T / ref.shape[1])
    center = y[fit].mean(0)
    proj = u.T @ (y[fit] - center)
    strength = max(kval.clip(0).mean(), 1e-12)
    return [center + (cross @ u) @ (proj / (kval.clip(0)[:, None] + strength * a)) for a in alphas]


def select_ridge(x, y, people, train, dev, grid):
    y = y.reshape(len(y), -1)
    losses = []
    for fit, val in grouped_folds(people, train):
        preds = standardized_ridge(x, y, fit, val, grid)
        sd = y[fit].std(0)
        if np.any(sd == 0):
            raise ValueError("degenerate target")
        loss = np.stack([(((p - y[val]) / sd) ** 2).mean(1) for p in preds], axis=1)
        losses.extend(loss[people[val] == s].mean(0) for s in np.unique(people[val]))
    cv = np.mean(losses, 0)
    j = int(np.argmin(cv))
    return standardized_ridge(x, y, train, dev, [grid[j]])[0], grid[j], cv
