"""Train-selected consecutive native-code and family trajectories with aligned lag controls."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def family_ids(ids, books, train, seed):
    families = np.empty_like(ids, dtype=np.uint8)
    mappings = []
    for j, book in enumerate(books):
        counts = np.bincount(ids[train, j].ravel(), minlength=len(book))
        used = counts > 0
        fit = KMeans(n_clusters=8, n_init=5, random_state=seed).fit(book[used], sample_weight=counts[used])
        mapping = fit.predict(book).astype(np.uint8)
        families[:, j] = mapping[ids[:, j]]
        mappings.append(mapping)
    return families, mappings


def exact_orders(ids, frame):
    """Per-person coverage of actual ordered ID pairs, using a train-only dictionary per task and site."""
    rng = np.random.default_rng(1227)
    rows = []
    for task, ix in frame.groupby("task").indices.items():
        train = ix[frame.split.to_numpy()[ix] == "train"]
        dev = ix[frame.split.to_numpy()[ix] == "validation"]
        for stream in range(4):
            for site in range(4):
                x = ids[:, stream, site].astype(np.int64)
                pair = x[:, :-1] * 65536 + x[:, 1:]
                known = np.unique(pair[train])
                seq = x[dev]
                shuffled = np.take_along_axis(seq, rng.random(seq.shape).argsort(1), axis=1)
                for name, v in (("observed", seq), ("shuffled", shuffled)):
                    encoded = v[:, :-1] * 65536 + v[:, 1:]
                    # Focus on changes; long constant runs are separately counted.
                    diff = v[:, :-1] != v[:, 1:]
                    match = np.isin(encoded, known) & diff
                    for person, loc in frame.iloc[dev].groupby("subject").indices.items():
                        rows.append(
                            dict(
                                task=task,
                                stream=stream,
                                site=site,
                                subject=person,
                                control=name,
                                matched=int(match[loc].sum()),
                                changes=int(diff[loc].sum()),
                                all_pairs=int(diff[loc].size),
                            )
                        )
    return pd.DataFrame(rows)


def pattern_codes(e, h, slot):
    """e consecutive two words at u,u+1; h words at u+slot,u+slot+1."""
    n = e.shape[-1] - 1 - slot
    return ((e[:, :n].astype("int32") * 8 + e[:, 1 : n + 1]) * 8 + h[:, slot:-1]) * 8 + h[:, slot + 1 :]


def pattern_fields(word):
    return [(word // (8**k)) % 8 for k in (3, 2, 1, 0)]


def comparisons(fam, m):
    """Pick one family trajectory per task/path on training-only person-balanced contrasts."""
    valid = np.array([a != b and c != d for a, b, c, d in map(pattern_fields, range(4096))])
    selections = []
    scores = []
    members = []
    for task, idx in m.groupby("task").indices.items():
        idx = idx[m.donor.to_numpy()[idx] >= 0]
        train = idx[m.split.to_numpy()[idx] == "train"]
        dev = idx[m.split.to_numpy()[idx] == "validation"]
        for path in range(4):
            a, b = path // 2, 2 + path % 2
            best = None
            for site in range(4):
                for lag in range(5):
                    e = fam[train, a, site]
                    h = fam[train, b, site]
                    matrices = [
                        pattern_codes(f, h, lag)
                        for f in (e, fam[m.donor.to_numpy()[train], a, site], e[:, ::-1])
                    ]
                    hist = []
                    for _, where in m.iloc[train].groupby("subject").indices.items():
                        hist.append(
                            [
                                np.bincount(matrix[where].ravel(), minlength=4096) / matrix[where].size
                                for matrix in matrices
                            ]
                        )
                    freqs = np.array(hist)
                    means = freqs.mean(0)
                    excess = np.minimum(means[0] - means[1], means[0] - means[2])
                    excess[~(valid & ((freqs[:, 0] > 0).sum(0) >= 3))] = -np.inf
                    word = int(excess.argmax())
                    rank = excess[word]
                    if np.isfinite(rank) and (best is None or rank > best[0]):
                        best = (rank, site, lag, word, means[:, word])
            if best is None:
                selections.append(dict(task=task, path=path, status="no eligible training pattern"))
                continue
            rank, site, lag, word, train_rates = best
            row = dict(
                task=task,
                path=path,
                status="selected",
                site=site,
                lag_seconds=lag * 2,
                word=word,
                train_true=train_rates[0],
                train_wrong=train_rates[1],
                train_reverse=train_rates[2],
            )
            selections.append(row)
            e = fam[dev, a, site]
            h = fam[dev, b, site]
            hits = [
                pattern_codes(f, h, lag) == word
                for f in (e, fam[m.donor.to_numpy()[dev], a, site], e[:, ::-1])
            ]
            for person, where in m.iloc[dev].groupby("subject").indices.items():
                counts = [int(z[where].sum()) for z in hits]
                opportunities = hits[0][where].size
                scores.append(
                    dict(
                        **row,
                        subject=person,
                        events=len(where),
                        opportunities=opportunities,
                        true_count=counts[0],
                        wrong_count=counts[1],
                        reverse_count=counts[2]
                    )
                )
                for ev, slot in np.argwhere(hits[0][where]):
                    members.append(
                        dict(
                            task=task,
                            path=path,
                            site=site,
                            event_index=int(dev[where[ev]]),
                            subject=person,
                            eeg_slot=int(slot),
                            hb_slot=int(slot + lag),
                            word=word,
                        )
                    )
    return pd.DataFrame(selections), pd.DataFrame(scores), pd.DataFrame(members)


def lag_profile(weights, mask, to_hb):
    """Sum real-key weights by physical lag, each eligible directed pair once."""
    import torch

    slots = torch.arange(15, device=weights.device)
    dt = slots[:, None] - slots if to_hb else slots[None, :] - slots[:, None]
    w = weights[..., :-1] * mask[..., None]
    obs = mask[..., None].expand_as(w)
    bins = torch.nn.functional.one_hot(dt.flatten() + 14, num_classes=29).to(w.dtype)
    mass = w.sum(1).flatten(1) @ bins
    opportunities = obs.sum(1).to(w.dtype).flatten(1) @ bins
    return mass, opportunities


def transfer_selected(ref, alt, selected, m):
    """Fit label permutation on corresponding training slots, then replay the same motifs in seed two."""
    from scipy.optimize import linear_sum_assignment

    translated = np.empty_like(alt)
    rates = []
    train = m.split.to_numpy() == "train"
    for j in range(4):
        counts = np.bincount((alt[train, j].astype(int) * 8 + ref[train, j]).ravel(), minlength=64).reshape(
            8, 8
        )
        a, b = linear_sum_assignment(-counts)
        perm = np.empty(8, int)
        perm[a] = b
        translated[:, j] = perm[alt[:, j]]
        rates.append(dict(stream=j, train_agreement=counts[a, b].sum() / counts.sum(), mapping=perm.tolist()))
    scores = []
    for row in selected.query("status=='selected'").itertuples():
        dev = np.flatnonzero((m.task == row.task) & (m.split == "validation") & (m.donor >= 0))
        lag = int(row.lag_seconds // 2)
        site = int(row.site)
        a = int(row.path) // 2
        b = 2 + int(row.path) % 2
        e = translated[dev, a, site]
        h = translated[dev, b, site]
        hits = [
            pattern_codes(f, h, lag) == row.word
            for f in (e, translated[m.donor.to_numpy()[dev], a, site], e[:, ::-1])
        ]
        for person, where in m.iloc[dev].groupby("subject").indices.items():
            scores.append(
                dict(
                    task=row.task,
                    path=row.path,
                    word=row.word,
                    subject=person,
                    site=site,
                    lag_seconds=row.lag_seconds,
                    opportunities=hits[0][where].size,
                    **{
                        k: int(z[where].sum())
                        for k, z in zip(("true_count", "wrong_count", "reverse_count"), hits)
                    }
                )
            )
    return pd.DataFrame(scores), rates
