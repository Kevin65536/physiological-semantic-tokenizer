"""Name Hb words in training subjects and count matched, replicated trend types."""

import numpy as np
import pandas as pd


def quadrants(slopes, threshold):
    z = np.asarray(slopes)
    return np.where(
        (np.abs(z) <= threshold).any(1), -1, 2 * (z[:, 0] > 0).astype(int) + (z[:, 1] > 0).astype(int)
    )


def word_census(person, word, role, n, labels):
    """Uncertain people retain their votes in the majority denominator."""

    def count(mask):
        return np.bincount(word[mask], minlength=1024)

    ok = n >= 3
    tr = ok & (role == 0)
    dev = ok & (role == 1)
    votes = np.stack([count(tr & (labels == q)) for q in range(4)], axis=1)
    nt, nd = count(tr), count(dev)
    q = np.where(votes.max(1) > nt / 2, votes.argmax(1), -1)
    matches = count(dev & (q[word] >= 0) & (labels == q[word]))
    supported = (nt >= 3) & (nd >= 3)
    tab = pd.DataFrame(
        dict(
            word=np.arange(1024),
            train_people=nt,
            dev_people=nd,
            supported=supported,
            quadrant=q,
            named=supported & (q >= 0),
            majority=supported & (q >= 0) & (matches > nd / 2),
            unanimous=supported & (q >= 0) & (matches == nd),
            same_people=matches,
            train_emissions=np.bincount(word[role == 0], weights=n[role == 0], minlength=1024).astype(int),
            dev_emissions=np.bincount(word[role == 1], weights=n[role == 1], minlength=1024).astype(int),
        )
    )
    return tab


def composition(words):
    rows = []
    for (branch, zone), g in words.groupby(["branch", "zone"]):
        for tier in ["named", "majority", "unanimous"]:
            sub = g[g[tier]]
            for weight, w in [
                ("word_type", np.ones(len(sub))),
                ("dev_emissions", sub.dev_emissions.to_numpy()),
            ]:
                counts = np.bincount(sub.quadrant, weights=w, minlength=4)
                rows.append(
                    dict(
                        branch=branch,
                        zone=zone,
                        tier=tier,
                        weight=weight,
                        word_types=len(sub),
                        denominator=int(counts.sum()),
                        **{f"q{j}": int(v) for j, v in enumerate(counts)},
                        anti=(counts[1] + counts[2]) / counts.sum() if counts.sum() else np.nan,
                    )
                )
    return pd.DataFrame(rows)


def frequency_match(words):
    rows = []
    for zone, g in words.groupby("zone"):
        for tier in ["named", "majority", "unanimous"]:
            sub = g[g[tier]].copy()
            keys = []
            for col in ["train_people", "dev_people", "train_emissions", "dev_emissions"]:
                key = col + "_stratum"
                sub[key] = np.floor(np.log2(sub[col])).astype(int)
                keys.append(key)
            sub["anti"] = sub.quadrant.isin([1, 2])
            tab = sub.groupby(keys + ["branch"]).anti.agg(["size", "mean"]).unstack()
            overlap = tab.dropna()
            w = overlap["size"].min(1)
            rows.append(
                dict(
                    zone=zone,
                    tier=tier,
                    matched_per_branch=int(w.sum()),
                    common_strata=len(w),
                    S_total=int((sub.branch == "S").sum()),
                    O_total=int((sub.branch == "O").sum()),
                    S_anti=np.average(overlap["mean"]["S"], weights=w) if len(w) else np.nan,
                    O_anti=np.average(overlap["mean"]["O"], weights=w) if len(w) else np.nan,
                )
            )
    return pd.DataFrame(rows)
