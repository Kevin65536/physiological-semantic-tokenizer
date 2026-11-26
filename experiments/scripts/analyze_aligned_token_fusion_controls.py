"""Matched feature budgets and held-pair ablations on existing frozen-token panels."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from src.analysis.frozen_token_readout import Means, ResidualRidge
from src.analysis.single_vq_relations import fit_ridge
import audit_aligned_tokenizer as audit
from analyze_aligned_token_controls import save, words
from single_vq_member_sequence import crop_mean, hierarchy, sequence_rows


def matched_main(mod):
    train, test = [audit.collect(mod, 17, role) for role in ("train", "validation")]
    model = audit.network(mod, audit.PLAN["models"][mod]["K"], 17)
    params = {}
    rng = np.random.default_rng(1126)
    fam = {}
    for j, b in enumerate("SO"):
        book = model.branches[b]["quantizer"].codebook.detach().numpy()
        fam[b] = words(mod, b).family.to_numpy()
        f = fam[b][train["ids"][:, j]]
        proj = book @ rng.normal(size=(128, 16)) / np.sqrt(128)
        params[b] = (book, np.eye(4)[fam[b]] - np.bincount(f, minlength=4) / len(f), proj)

    def design(ids, arm):
        n = []
        pro = []
        one = []
        for j, b in enumerate("SO"):
            book, hot, proj = params[b]
            v = ids[:, j]
            n.extend([book[v], hot[v]])
            pro.append(proj[v])
            one.append(hot[v])
        nonlinear = np.column_stack([x[:, :8] ** 2 for x in pro])
        n.append(nonlinear)
        extra = (
            np.column_stack([x[:, 8:] ** 2 for x in pro])
            if arm == "main"
            else (one[0][:, :, None] * one[1][:, None, :]).reshape(len(ids), 16)
        )
        return np.column_stack(n + [extra]).astype("float32")

    means = Means(train["bg"], train["y"])
    loss = {}
    for arm in ["main", "cross"]:
        rd = ResidualRidge(train["bg"], design(train["ids"], arm), train["y"], means)
        coef = rd.coefficients(0.01)
        pred = np.concatenate(
            [
                rd.predict(test["bg"][i : i + 8192], design(test["ids"][i : i + 8192], arm), coef)
                for i in range(0, len(test["ids"]), 8192)
            ]
        )
        loss[arm] = crop_mean(test, (pred - test["y"]) ** 2)
    rows = []
    for j, target in enumerate("XSO"):
        h = hierarchy(test["examples"], {k: v[:, j] for k, v in loss.items()})
        h["target"] = target
        rows.append(h)
    save(pd.concat(rows), mod + "_interaction_budget")
    # Equal-capacity bag vs ordered descriptors through 4 fixed orthonormal projections of the richer input.
    from src.analysis.single_vq_relations import descriptors

    seq = []
    for j, b in enumerate("SO"):
        for adj in [False, True]:
            tr, y, _ = sequence_rows(train, fam[b], j, adj)
            te, t, ex = sequence_rows(test, fam[b], j, adj)
            trf = descriptors(tr)
            tef = descriptors(te)
            scores = []
            for name, tx, dev in zip(["bag", "bigram_dwell", "motif"], trf, tef):
                dim = trf[0].shape[1]
                if tx.shape[1] > dim:
                    # Unsupervised fixed-width SVD learned only on training histories.
                    norm = StandardScaler().fit(tx)
                    xx = norm.transform(tx)
                    vv = norm.transform(dev)
                    basis = np.linalg.svd(xx, full_matrices=False)[2][:dim].T
                    tx = xx @ basis
                    dev = vv @ basis
                pred = fit_ridge(tx, y)(dev)
                scores.append((name, (pred - t) ** 2))
            h = hierarchy(ex, dict(scores))
            h["branch"] = b
            h["adjacent"] = adj
            h["dimension"] = dim
            seq.append(h)
    save(pd.concat(seq), mod + "_order_budget")


def fusion_null():
    out = audit.OUT
    pop = pd.read_csv(out / "paired_panel.csv").query("status=='complete'").reset_index(drop=True)
    a = np.load(out / "paired_features.npz")
    ix = np.flatnonzero(pop.condition.isin(["BL", "MA"]))
    pop = pop.iloc[ix].reset_index(drop=True)
    y = pop.condition.eq("MA").to_numpy()
    E, H = [
        np.concatenate([a[m + b][ix].reshape(len(ix), -1) for b in "SO"], axis=1) for m in ("eeg", "hb")
    ]
    rng = np.random.default_rng(1126)
    keep = []
    don = []
    for i, r in pop.iterrows():
        q = (
            (pop.subject == r.subject)
            & (pop.record == r.record)
            & (pop.condition == r.condition)
            & (abs(pop.eeg_start - r.eeg_start) >= 60)
            & (abs(pop.hb_start - r.hb_start) >= 60)
        )
        if q.any():
            keep.append(i)
            don.append(rng.choice(np.flatnonzero(q)))
    k = np.array(keep)
    tr = pop.split.iloc[k].eq("train").to_numpy()
    sub = pop.subject.iloc[k].to_numpy()
    rows = []

    def model(X):
        norm = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.1, max_iter=1000).fit(norm.transform(X[tr]), y[k][tr])
        return lambda x: clf.predict_proba(norm.transform(x))[
            :, int(np.flatnonzero(clf.classes_ == True)[0])
        ]

    def score(method, p):
        for person in np.unique(sub[~tr]):
            v = (~tr) & (sub == person)
            rows.append(
                dict(
                    subject=person,
                    method=method,
                    train=int(tr.sum()),
                    n=v.sum(),
                    n_ma=y[k][v].sum(),
                    ba=(
                        balanced_accuracy_score(y[k][v], p[v] >= 0.5)
                        if len(np.unique(y[k][v])) == 2
                        else np.nan
                    ),
                    brier=brier_score_loss(y[k][v], p[v]),
                )
            )

    good = np.column_stack([E[k], H[k]])
    bad = np.column_stack([E[k], H[don]])
    fit = model(good)
    for name, X in [("aligned", good), ("frozen_mismatch", bad)]:
        score(name, fit(X))
    score("refit_mismatch", model(bad)(bad))
    score("eeg", model(E[k])(E[k]))
    score("hb", model(H[k])(H[k]))
    cont = np.column_stack(
        [a[m + "_y"][ix[k]].reshape(len(k), 4, 3, 5).mean(-1).reshape(len(k), 12) for m in ("eeg", "hb")]
    )
    score("continuous", model(cont)(cont))
    for lag in [1, -1]:
        # Shift Hb bin labels by one ten-second bin, nearest-edge fill. Tests timing sensitivity after summarization.
        altered = (
            H[k]
            .reshape(len(k), 2, 4, 3, 4)[:, :, :, np.clip(np.arange(3) + lag, 0, 2), :]
            .reshape(len(k), -1)
        )
        score("offset_" + str(lag * 10) + "s", fit(np.column_stack((E[k], altered))))
    save(rows, "fusion_pair_controls")
    save(
        pop.iloc[k].assign(
            donor_event=pop.event.iloc[don].values,
            donor_eeg=pop.eeg_start.iloc[don].values,
            donor_hb=pop.hb_start.iloc[don].values,
        ),
        "fusion_pair_support",
    )
