"""Matched transfer, scalp synchrony, and both-modality window-boundary controls."""

from collections import OrderedDict, defaultdict
import numpy as np
import pandas as pd
import torch
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score

import audit_aligned_tokenizer as audit
from analyze_aligned_token_controls import save, words
from src.analysis.single_vq_relations import fit_ridge
from src.data.dual_component_tokenizer import base_tokenizer_batch

OUT = audit.OUT


@torch.inference_mode()
def run(mod):
    rows, scales = audit.relations.maps()
    model = audit.network(mod, audit.PLAN["models"][mod]["K"], 17, "cuda:0")
    maps = {b: words(mod, b).family.to_numpy() for b in "SO"}
    cache = OrderedDict()
    rng = np.random.default_rng(1126)
    dv = audit.collect(mod, 17, "validation")
    ex = dv["examples"]
    look = {tuple(e): i for i, e in enumerate(ex)}
    per = defaultdict(list)
    for i, (r, t, p) in enumerate(ex):
        j = look.get((r, t + 30, p))
        if j is not None and dv["valid"][i].all() and dv["valid"][j].all():
            per[(rows[r]["dataset"], rows[r]["subject"])].append((i, j))
    data = []
    for (ds, sub), candidates in per.items():
        pairs = np.array(candidates)[rng.choice(len(candidates), min(32, len(candidates)), replace=False)]
        x, y = [base_tokenizer_batch(rows, ex[idx], scales, array_cache=cache)["x"] for idx in pairs.T]
        shift = np.concatenate([x[:, 1:], y[:, :1]], 1)
        assert np.array_equal(x[:, 1:], shift[:, :-1])
        orig, alt = [model(torch.as_tensor(v, device="cuda:0"))["ids"] for v in (x, shift)]
        for b in "SO":
            a = orig[b].cpu().numpy()[:, 1:]
            z = alt[b].cpu().numpy()[:, :-1]
            for label, idx in [("edge", [0, 13]), ("internal", list(range(1, 13)))]:
                m, n = a[:, idx], z[:, idx]
                data.append(
                    dict(
                        dataset=ds,
                        subject=sub,
                        branch=b,
                        region=label,
                        eligible=len(candidates),
                        crops=len(pairs),
                        n=m.size,
                        id_match=np.mean(m == n),
                        family_match=np.mean(maps[b][m] == maps[b][n]),
                    )
                )
    save(data, mod + "_window_shift")
    old = (
        audit.PARENT.parent.parent
        / ("single_vq_final_c_audit" if mod == "eeg" else "single_vq_relations")
        / "20261125_v1"
    )
    result = []
    for j, b in enumerate("SO"):
        current = maps[b][dv["ids"][:, j]]
        for seed in (17, 43):
            fam = pd.read_csv(old / f"{mod}_{seed}/{b}_words.csv").family.to_numpy()
            li = []
            for t in audit.tasks("validation"):
                if t[0] == mod:
                    with np.load(old / "inputs" / audit.readout.destination(t).name) as f:
                        li.append(fam[f[f"{b}_{seed}_id"]])
            other = np.concatenate(li)
            for (d, p), i in audit.readout.PEOPLE.items():
                k = dv["person"] == i
                if k.any():
                    result.append(
                        dict(
                            branch=b,
                            dataset=d,
                            subject=p,
                            historical_seed=seed,
                            n=k.sum(),
                            agreement=np.mean(current[k] == other[k]),
                            ARI=adjusted_rand_score(current[k], other[k]),
                        )
                    )
    save(result, mod + "_family_stability")
    pop = (
        pd.read_csv(OUT / "paired_panel.csv", dtype={"subject": str})
        .query("status=='complete'")
        .reset_index(drop=True)
    )
    a = np.load(OUT / "paired_features.npz")
    tr = pop.split.eq("train").to_numpy()
    vec = np.concatenate([a[mod + b] for b in "SO"], axis=-1)
    labels = pop.dataset
    classifier = LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced")

    def predict(X, train, Y, evaluate):
        return classifier.fit(X[train], Y[train]).predict(X[evaluate])

    XX = vec.reshape(len(pop), -1)
    pred = predict(XX, tr, labels, ~tr)
    scores = pop.loc[~tr, ["dataset", "subject"]].copy()
    scores["predicted"] = pred
    scores["correct"] = scores.dataset == pred
    scores.to_csv(OUT / f"{mod}_dataset_identity.csv", index=False)
    position = []
    synchrony = []
    for i in range(4):
        ix = [j for j in range(4) if j != i]
        z = vec[:, ix].reshape(len(pop) * 3, -1)
        y = a[mod + "_y"][:, ix].mean(2).reshape(-1)
        fit = fit_ridge(z[np.repeat(tr, 3)], y[np.repeat(tr, 3)])
        target = a[mod + "_y"][:, i].mean(1)
        pred = fit(vec[:, i].reshape(len(pop), -1))
        same = fit_ridge(vec[tr, i].reshape(tr.sum(), -1), target[tr])(vec[:, i].reshape(len(pop), -1))
        for j in np.flatnonzero(~tr):
            m = pop.iloc[j]
            position.append(
                dict(
                    dataset=m.dataset,
                    subject=m.subject,
                    region=i,
                    held_error=(pred[j] - target[j]) ** 2,
                    same_error=(same[j] - target[j]) ** 2,
                    mean_error=(target[j] - np.mean(target[tr])) ** 2,
                )
            )
    save(position, mod + "_position_transfer")
    positions = pd.read_csv(OUT / "positions.csv", dtype={"subject": str})
    index = {(r["parent"], r["modality"]): i for i, r in enumerate(rows)}
    for (ds, sub, par, t), group in positions[
        (positions.split == "validation") & (positions["mod"] == mod)
    ].groupby(["dataset", "subject", "parent", "start"]):
        r = index[par, mod]
        e = [(r, t, rows[r]["positions"].index(site)) for site in group.position]
        raw = base_tokenizer_batch(rows, e, scales, array_cache=cache)
        x = raw["x"]
        pred = model(torch.as_tensor(x, device="cuda:0"))["sum"].cpu().numpy()

        def metrics(v):
            if mod == "eeg":
                alpha = signal.sosfiltfilt(
                    signal.butter(3, [8, 13], fs=200, btype="bandpass", output="sos"),
                    v.reshape(4, -1),
                    axis=-1,
                )
                h = signal.hilbert(alpha, axis=1)[:, 200:-200]
                u = h / np.maximum(abs(h), 1e-12)
                return {"envelope_r": np.corrcoef(abs(h)), "phase_lock": abs(u @ u.conj().T / u.shape[1])}
            z = v.reshape(4, 300, 2)
            return {"HbO_r": np.corrcoef(z[:, :, 0]), "HbR_r": np.corrcoef(z[:, :, 1])}

        shifted = np.stack([np.roll(row, 2 * k, axis=0) for k, row in enumerate(pred)])
        raw_v = metrics(x)
        for name, v in [("decoded", pred), ("shift", shifted)]:
            for attr, val in metrics(v).items():
                ij = np.triu_indices(4, 1)
                synchrony.append(
                    dict(
                        dataset=ds,
                        subject=sub,
                        parent=par,
                        start=t,
                        method=name,
                        attribute=attr,
                        mse=np.mean((val[ij] - raw_v[attr][ij]) ** 2),
                        observed=raw_v[attr][ij].mean(),
                        estimated=val[ij].mean(),
                    )
                )
    save(synchrony, mod + "_synchrony")
