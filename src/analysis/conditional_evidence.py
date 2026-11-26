"""Physical-unit and train-scale paired scores for conditional reader experiments."""

import numpy as np
import pandas as pd
from src.analysis.single_vq_relations import interval


def cell_scores(meta, y, pred, sd, mod, arm, seed, **labels):
    """All coordinates are physical until the paired error is normalized."""
    yn = y.reshape(len(y), 4, 3, -1)
    pv = pred.reshape(yn.shape)
    scale = sd.reshape(4, 3, -1)
    sites = ["AFF3h", "AFF4h", "P3h", "P4h"] if mod == "hb" else ["F1", "F2", "P3", "P4"]
    out = []
    for who, idx in meta.reset_index(drop=True).groupby("subject").indices.items():
        err = pv[idx] - yn[idx]
        p = pv[idx] - pv[idx].mean(0)
        obs = yn[idx] - yn[idx].mean(0)
        var = (p * p).sum(0)
        slope = np.divide((p * obs).sum(0), var, out=np.full_like(var, np.nan), where=var > 1e-12)
        rmse = np.sqrt((err**2).mean(0))
        bias = err.mean(0)
        for (s, l, c), v in np.ndenumerate(rmse):
            out.append(
                dict(
                    subject=who,
                    modality=mod,
                    arm=arm,
                    seed=seed,
                    site=sites[s],
                    lag_s=[2, 4, 6][l],
                    target=(["HbO", "HbR"] if mod == "hb" else ["alpha"])[c],
                    events=len(idx),
                    nmse=(v / scale[s, l, c]) ** 2,
                    rmse=v,
                    bias=bias[s, l, c],
                    calibration_slope=slope[s, l, c],
                    **labels,
                )
            )
    return out


def effect_tables(people, value, pairs, group):
    out = []
    for keys, g in people.groupby(group):
        table = g.pivot(index="subject", columns="arm", values=value)
        for label, ref, arm in pairs:
            if ref not in table or arm not in table:
                continue
            d = table[ref] - table[arm]
            if d.isna().any():
                raise ValueError("unpaired scores")
            point, lo, hi = interval(d, np.repeat("simultaneous_eeg_nirs", len(d)))
            out.append(
                dict(
                    zip(group, keys if isinstance(keys, tuple) else [keys]),
                    contrast=label,
                    reference=ref,
                    candidate=arm,
                    reference_nmse=table[ref].mean(),
                    candidate_nmse=table[arm].mean(),
                    delta=point,
                    ci_low=lo,
                    ci_high=hi,
                    people=len(d),
                    improved=int((d > 0).sum()),
                )
            )
    return pd.DataFrame(out)


def p1_tables(folder, cells):
    by = cells.groupby(
        ["coordinate", "task", "input", "calibrated", "arm", "subject", "seed"], as_index=False
    ).nmse.mean()
    by.to_csv(folder / "p1_subject_seed.csv", index=False)
    group = ["coordinate", "task", "input", "calibrated"]
    mean = by.groupby(group + ["arm", "subject"], as_index=False).nmse.mean()
    comp = effect_tables(
        mean,
        "nmse",
        [
            ("over_frozen_anchor", "anchor", "true"),
            ("over_Hb_capacity", "capacity", "true"),
            ("over_wrong_training", "wrong", "true"),
        ],
        group,
    )
    comp.to_csv(folder / "p1_contrasts.csv", index=False)
    return comp


def verify_reader_scores(out, seeds, load_parent, read_json):
    results = []
    for coord in ["native", "cached", "strict_common"]:
        _, meta, data = load_parent(coord)
        train = meta.split.eq("train").to_numpy()
        dev = ~train
        for task in ["A", "B_hb", "B_eeg"] if coord == "native" else ["A", "B_hb"]:
            mod = "eeg" if task == "B_eeg" else "hb"
            y = data[mod + "_target"].astype(float).reshape(len(meta), 4, 3, -1)
            sd = y[train].std(0)
            for seed in seeds:
                p = out / "readers" / f"{coord}_{task}_{seed}"
                tab = pd.read_csv(p / "cells.csv.gz")
                want = (
                    tab.query("not calibrated and input=='aligned'").groupby(["arm", "subject"]).nmse.mean()
                )
                fold = read_json(p / "selection.json")
                tr_people = set(meta.loc[train, "subject"])
                outer = [s for i in range(3) for s in fold[f"outer{i}"]["held"]]
                assert len(outer) == len(tr_people) and set(outer) == tr_people
                for k in range(3):
                    item = fold[f"outer{k}"]
                    assert {s for f in item["inner"]["folds"] for s in f} == tr_people - set(item["held"])
                with np.load(p / "predictions.npz") as z:
                    for arm in want.index.get_level_values("arm").unique():
                        pred = z[arm + "_aligned"]
                        mse = (((pred - y[dev]) / sd) ** 2).reshape(dev.sum(), -1).mean(1)
                        scored = pd.Series(mse).groupby(meta.loc[dev, "subject"].to_numpy()).mean()
                        np.testing.assert_allclose(scored, want[arm], rtol=2e-6)
                        assert len(z[arm + "_oof"]) == train.sum() and np.isfinite(z[arm + "_oof"]).all()
                results.append(p.name)
    p2 = read_json(out / "p2.json")
    assert p2["status"] == "completed"
    for c in ["native", "cached"]:
        df = pd.read_csv(out / c / "training_curves.csv.gz")
        assert df.groupby(["arm", "seed", "origin"]).size().eq(300).all()
        assert len(df.query("origin=='parent'")) == 42 * 300
    return results
