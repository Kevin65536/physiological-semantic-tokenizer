"""Readout and sequence controls for the public frozen-tokenizer screen."""

import numpy as np
from scipy.linalg import solve
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer

from src.analysis.four_stream_bridge import grouped_folds, select_ridge


def anchor(history, target, people, train, evaluate, grid):
    """Physical-unit predictions, with held-person anchors for every residual target."""
    y = target.reshape(len(target), -1)
    pred, alpha, _ = select_ridge(history, y, people, train, evaluate, grid)
    oof = np.full_like(y, np.nan, dtype=float)
    for fit, held in grouped_folds(people, train):
        oof[held] = select_ridge(history, y, people, fit, held, grid)[0]
    return oof[train], pred, alpha


def slot_vectors(x, train):
    """Keep all source coordinates, preserve slot order and equalize nominal readout widths."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 5 or x.shape[1:4] != (4, 2, 15) or x.shape[4] > 128:
        raise ValueError("expected four sites, two branches, fifteen slots, at most 128 features")
    if not np.isfinite(x).all():
        raise ValueError("nonfinite history")
    mu = x[train].mean(axis=(0, 3), keepdims=True)
    sd = x[train].std(axis=(0, 3), keepdims=True)
    x = (x - mu) / np.where(sd > 1e-7, sd, 1)
    return np.pad(x, ((0, 0), (0, 0), (0, 0), (0, 0), (0, 128 - x.shape[-1])))


def gram(x):
    x = x.reshape(len(x), -1) if isinstance(x, np.ndarray) else x
    product = x @ x.T
    return (
        product.toarray().astype(float)
        if isinstance(product, csr_matrix)
        else np.asarray(product, dtype=float)
    )


def centered(kernel, fit, score):
    """Train-centered, unit mean diagonal. Can be applied to a donor-permuted block."""
    ref = kernel[np.ix_(fit, fit)]
    cross = kernel[np.ix_(score, fit)]
    grand = ref.mean()
    scale = max(ref.diagonal().mean() - grand, 1e-10)
    return (cross - cross.mean(1, keepdims=True) - ref.mean(0) + grand) / scale


def residual(hk, sk, residuals, train, evaluate, donor, wrong_train=False, wrong_eval=False, ridge=1.0):
    src = donor[train] if wrong_train else train
    ev = donor[evaluate] if wrong_eval else evaluate
    k = centered(hk, train, train) + centered(sk, src, src)
    cross = centered(hk, train, evaluate) + centered(sk, src, ev)
    mu = residuals.mean(0)
    return mu + cross @ solve(k + ridge * np.eye(len(train)), residuals - mu, assume_a="pos")


def family_map(ids, alpha, train, size, book_size=8192):
    """Unknown IDs get their own category. Random control exactly preserves train occupancy."""
    rng = np.random.default_rng(1126)
    maps, random = [], []
    for branch in range(2):
        words = ids[train, :, branch].ravel()
        counts = np.bincount(words, minlength=book_size)
        sums = np.bincount(words, weights=alpha[train].ravel(), minlength=book_size)
        means = sums[counts > 0] / counts[counts > 0]
        thresholds = np.quantile(np.repeat(means, counts[counts > 0]), np.arange(1, size) / size)
        mapping = np.full(book_size, size)
        mapping[counts > 0] = np.searchsorted(thresholds, means)
        shuffled = mapping.copy()
        for count in np.unique(counts[counts > 0]):
            idx = np.flatnonzero(counts == count)
            shuffled[idx] = rng.permutation(mapping[idx])
        maps.append(mapping)
        random.append(shuffled)
    return np.stack(maps), np.stack(random)


def assigned(ids, maps):
    return np.stack([maps[b, ids[:, :, b]] for b in range(2)], axis=2)


def shuffle_trails(ids, seed=1126):
    """Random Euler trail of the observed transition multigraph, retaining both endpoints."""
    rng = np.random.default_rng(seed)
    shuffled = ids.copy()
    for index in np.ndindex(ids.shape[:-1]):
        seq = ids[index].tolist()
        edges = {}
        for a, b in zip(seq, seq[1:]):
            edges.setdefault(a, []).append(b)
        for choices in edges.values():
            rng.shuffle(choices)
        stack, reverse = [seq[0]], []
        while stack:
            choices = edges.get(stack[-1])
            if choices:
                stack.append(choices.pop())
            else:
                reverse.append(stack.pop())
        shuffled[index] = reverse[::-1]
    return shuffled


def count_kernel(ids, train, ordered):
    """Fit sequence vocabulary on training events, retaining sites and S/O membership."""
    rows = []
    for event in ids:
        counts = {}
        for (site, branch), seq in zip(np.ndindex(event.shape[:-1]), event.reshape(-1, event.shape[-1])):
            for n in range(1, 4 if ordered else 2):
                for gram_ids in zip(*(seq[i:] for i in range(n))):
                    key = f"{site}/{branch}/" + ",".join(map(str, gram_ids))
                    counts[key] = counts.get(key, 0) + 1
        rows.append(counts)
    vocab = DictVectorizer().fit([rows[i] for i in train])
    return gram(vocab.transform(rows))


def interaction_blocks(ks, ko, train, evaluate, donor):
    """Additive controls and centered S×O product blocks with independently swappable O."""

    def center(ref, cross):
        avg = ref.mean(0)
        scale = max(ref.diagonal().mean() - avg.mean(), 1e-10)
        return (
            (ref - avg - avg[:, None] + avg.mean()) / scale,
            (cross - avg - cross.mean(1, keepdims=True) + avg.mean()) / scale,
        )

    def centered_all(k):
        avg = k[:, train].mean(1)
        grand = avg[train].mean()
        return (k - avg - avg[:, None] + grand) / max(k.diagonal()[train].mean() - grand, 1e-10)

    s, o = centered_all(ks), centered_all(ko)
    tr = np.ix_(train, train)
    ev = np.ix_(evaluate, train)
    base, cross = s[tr] + o[tr], s[ev] + o[ev]
    blocks = {"S": (s[tr], s[ev]), "O": (o[tr], o[ev]), "additive": (base, cross)}
    ar, ac = center(s[tr] ** 2 + o[tr] ** 2, s[ev] ** 2 + o[ev] ** 2)
    blocks["additive_expansion"] = (base + ar, cross + ac)
    for fit_label, src in [("true", train), ("wrong", donor[train])]:
        for input_label, idx in [("aligned", evaluate), ("mismatched", donor[evaluate])]:
            ref = s[tr] * o[np.ix_(src, src)]
            rr, rc = center(ref, s[ev] * o[np.ix_(idx, src)])
            blocks[f"interaction_{fit_label}_{input_label}"] = (base + rr, cross + rc)
    return blocks
