"""Grouped linear probes and sequence controls for task-aligned sensor panels."""

import numpy as np
from scipy.linalg import solve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def path_features(families):
    """Per site count distribution, run-length distribution and directed adjacent pairs."""
    eye = np.eye(4)[families]
    occupancy = eye.mean(-2)
    length = families.shape[-1]
    runs = np.zeros((*families.shape[:-1], length))
    for k in np.ndindex(families.shape[:-1]):
        seq = families[k]
        lengths = np.diff(np.r_[0, 1 + np.flatnonzero(np.diff(seq)), length])
        runs[k] = np.bincount(lengths, minlength=length + 1)[1:] / length
    order = np.einsum("...ti,...tj->...ij", eye[..., :-1, :], eye[..., 1:, :]) / (length - 1)
    return np.concatenate([occupancy, runs], -1), order


def flat(x):
    return x.reshape(len(x), -1)


def join(*xs):
    return np.column_stack([flat(x) for x in xs])


def phase(x):
    # event,site,slot,attributes: three successive 10-second epochs.
    return x.reshape(len(x), x.shape[1], 3, 5, -1).mean(3)


def probe(x, y, people, tr, ev, *, classification, dirty=None, ridge=(0.01, 0.1, 1, 10), folds=3):
    """Training-only component standardization and trace-scaled linear kernel ridge."""
    x = flat(x).astype(float)

    def one(a, b, alphas, other=None):
        scaler = StandardScaler().fit(x[a])
        z = scaler.transform(x[a])
        q = scaler.transform(x[b] if other is None else flat(other))
        k = z @ z.T
        den = max(np.diag(k).mean(), 1e-12)
        k /= den
        cross = q @ z.T / den
        mu = y[a].mean(0)
        for alpha in alphas:
            weights = solve(k + alpha * np.eye(len(a)), y[a] - mu, assume_a="pos")
            yield cross @ weights + mu

    cv = np.zeros(len(ridge))
    for a, b in GroupKFold(folds).split(tr, groups=people[tr]):
        a, b = tr[a], tr[b]
        for j, pr in enumerate(one(a, b, ridge)):
            vals = []
            for person in np.unique(people[b]):
                sel = people[b] == person
                if classification:
                    labels = y[b][sel].argmax(1)
                    choices = pr[sel].argmax(1)
                    vals.append(
                        1 - np.mean([(choices[labels == c] == c).mean() for c in np.unique(labels)])
                    )
                else:
                    vals.append(np.mean(((y[b][sel] - pr[sel]) / np.maximum(y[a].std(0), 1e-8)) ** 2))
            cv[j] += np.sum(vals)
    reg = ridge[int(np.argmin(cv))]
    out = next(one(tr, ev, [reg]))
    changed = None if dirty is None else next(one(tr, ev, [reg], dirty))
    return out, changed, reg
