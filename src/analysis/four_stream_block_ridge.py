"""Train-fold normalized kernel blocks and nested person-level selection."""

import numpy as np
from scipy.linalg import eigh
from src.analysis.four_stream_bridge import grouped_folds


def kernel_block(x, train, evaluate):
    x = np.asarray(x).reshape(len(x), -1).astype(float)
    a = x[train]
    mu = a.mean(0)
    scale = a.std(0)
    scale[scale < 1e-8] = 1.0
    a = (a - mu) / scale
    den = np.sum(a * a) / len(a)
    if den < 1e-12:
        return np.zeros((len(a), len(a))), np.zeros((len(evaluate), len(a)))
    return a @ a.T / den, ((x[evaluate] - mu) / scale) @ a.T / den


def block_predictions(h, e, c, y, tr, ev, etas, alphas):
    a, b = kernel_block(h, tr, ev)
    ac, bc = kernel_block(c, tr, ev)
    ae, be = kernel_block(e, tr, ev)
    mu = y[tr].mean(0)
    for eta in etas:
        vals, vec = eigh(a + ac + eta * ae)
        rhs = vec.T @ (y[tr] - mu)
        cross = (b + bc + eta * be) @ vec
        for alpha in alphas:
            yield (eta, alpha), mu + cross @ (rhs / (vals.clip(0)[:, None] + alpha))


def select_blocks(h, e, c, y, people, tr, ev, etas, alphas, folds):
    if set(people[tr]) & set(people[ev]) or len(set(people[tr])) < folds:
        raise ValueError("subject overlap or insufficient training folds")
    if not etas or min(etas) != 0 or min(alphas) <= 0:
        raise ValueError("expected nested EEG weights and positive ridge penalties")
    y = y.reshape(len(y), -1)
    risks = {}
    for a, v in grouped_folds(people, tr, folds):
        sd = y[a].std(0)
        if np.any(sd <= 1e-9):
            raise ValueError("degenerate training targets")
        for key, pred in block_predictions(h, e, c, y, a, v, etas, alphas):
            loss = (((pred - y[v]) / sd) ** 2).mean(1)
            risks.setdefault(key, []).extend(loss[people[v] == q].mean() for q in np.unique(people[v]))
    mean = {key: float(np.mean(vals)) for key, vals in risks.items()}
    # Iteration order breaks exact ties in favor of smaller EEG weight.
    key = min(mean, key=mean.get)
    return key, mean


def band_histories(wave, bands):
    if wave.ndim != 4 or wave.shape[1:] != (4, 15, 400) or not np.isfinite(wave).all():
        raise ValueError("expected complete 4-site 200 Hz two-second voltage patches")
    z = (wave - wave.mean(-1, keepdims=True)) * np.hanning(400)
    power = np.abs(np.fft.rfft(z, axis=-1)) ** 2
    hz = np.fft.rfftfreq(400, 1 / 200)
    return np.stack(
        [np.log(np.maximum(power[..., (hz >= lo) & (hz < hi)].sum(-1), 1e-12)) for lo, hi in bands], axis=-1
    )
