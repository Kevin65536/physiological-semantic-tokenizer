"""Audit the full-window masked search from terminal models and held-person scores."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.foundation.aligned_masked_search import search_mask, nominees, check_disjoint_windows


def read(path):
    return json.loads(path.read_text())


def grouped(frame, keys, cols):
    return (
        frame.groupby([*keys, "dataset", "subject", "record"])[cols]
        .mean()
        .groupby([*keys, "dataset", "subject"])
        .mean()
        .reset_index()
    )


def aggregate(people, keys, cols):
    return (
        people.groupby([*keys, "dataset"])[cols].mean().groupby(keys).mean()
        if keys
        else people.groupby("dataset")[cols].mean().mean()
    )


def count_lookup(train, query):
    keys, counts = np.unique(train, return_counts=True)
    ix = np.searchsorted(keys, query).clip(max=len(keys) - 1)
    return counts[ix] * (keys[ix] == query)


def priors(panel, ids, fold, vocab):
    """Fold-fit hierarchical native site/time and visible-neighbour priors, identical masks to training scorer."""
    fit = panel.fold.eq(fold).to_numpy()
    n = (~fit).sum()
    mask = {
        (kind, rep): search_mask(n, kind, np.random.default_rng(1126 + rep))
        for kind in ("span", "scatter", "site")
        for rep in range(4)
    }
    rows = []
    for stream, size in enumerate(vocab):
        mod = ["eeg", "hb"][stream // 2]
        x = ids[:, stream].astype("int64")
        train = x[fit]
        target = x[~fit]
        freq = np.bincount(train.flatten(), minlength=size)
        marginal = (freq[target] + 0.5) / (freq.sum() + 0.5 * size)
        labels = [
            [f"{d}/{s}" for s in names.split("|")] for d, names in zip(panel.dataset, panel[mod + "_sites"])
        ]
        site = pd.factorize(np.array(labels).ravel())[0].reshape(len(panel), 4)
        cell = site[:, :, None] * 15 + np.arange(15)
        den = count_lookup(cell[fit], cell[~fit])
        cond = (count_lookup(cell[fit] * size + train, cell[~fit] * size + target) + 16 * marginal) / (
            den + 16
        )
        prev = np.roll(target, 1, axis=-1)
        nxt = np.roll(target, -1, axis=-1)
        left = (
            count_lookup(train[:, :, :-1] * size + train[:, :, 1:], prev * size + target) + 16 * cond
        ) / (count_lookup(train[:, :, :-1], prev) + 16)
        right = (
            count_lookup(train[:, :, 1:] * size + train[:, :, :-1], nxt * size + target) + 16 * cond
        ) / (count_lookup(train[:, :, 1:], nxt) + 16)
        for (kind, rep), m in mask.items():
            hidden = m[:, stream // 2]
            vl = ~np.roll(hidden, 1, axis=-1)
            vl[:, :, 0] = False
            vr = ~np.roll(hidden, -1, axis=-1)
            vr[:, :, -1] = False
            edges = vl.astype(int) + vr
            markov = np.where(edges, (left * vl + right * vr) / edges.clip(min=1), cond)
            table = pd.DataFrame(dict(panel_row=np.flatnonzero(~fit), mask=kind, repeat=rep, stream=stream))
            for name, v in [("prior", marginal), ("conditional", cond), ("markov", markov)]:
                table[name] = (-np.log2(v) * hidden).sum((1, 2)) / hidden.sum((1, 2))
            rows.append(table)
    return (
        pd.concat(rows, ignore_index=True)
        .groupby(["panel_row", "mask", "stream"])[["prior", "conditional", "markov"]]
        .mean()
    )


def audit(run, out):
    cfg = yaml.safe_load((run / "resolved.yaml").read_text())
    summary = read(run / "training_summary.json")
    assert (
        summary == read(run / "status.json") and summary["fits"] == 48 and summary["status"] == "completed"
    )
    assert not any((run / k).exists() for k in ("B", "C", "validation"))
    panel = pd.read_csv(run / "train/panel.csv")
    check_disjoint_windows(panel)
    meta = ["dataset", "subject", "record"]
    tensor = cfg["tensor"]
    with np.load(run / "train/tokens.npz") as z:
        ids = z["ids"]
        books = [torch.from_numpy(z[f"book{i}"]) for i in range(4)]
    source = pd.read_csv(run / "A_results.csv").set_index(["candidate", "fold"], verify_integrity=True)
    grid = {
        "-".join(map(str, v)): dict(zip(cfg["grid"], v)) for v in itertools.product(*cfg["grid"].values())
    }
    assert set(source.index) == set(itertools.product(grid, [0, 1]))
    strict = panel.donor.ge(0)
    mask_tables = [priors(panel, ids, f, tensor["vocab"]) for f in (0, 1)]
    person_rows = []
    folds = []
    mask_rows = []
    oper = []
    scores = {}
    for (c, f), summary_row in source.iterrows():
        loc = run / "A" / c / f"fold{f}_seed17_true_pair"
        r = read(loc / "result.json")
        assert (r["stage"], r["status"], r["steps"], r["seed"], r["arm"], r["candidate"], r["fold"]) == (
            "A",
            "completed",
            1024,
            17,
            "true_pair",
            c,
            f,
        )
        assert r["signature"] == summary_row.signature
        ck = torch.load(loc / "checkpoint.pt", map_location="cpu", weights_only=False)
        assert ck["step"] == 1024 and ck["signature"] == r["signature"]
        assert all(torch.isfinite(v).all() for v in ck["model"].values())
        assert all(torch.equal(ck["model"][f"books.{i}.weight"], b) for i, b in enumerate(books))
        fit = panel.fold.eq(f)
        val = ~fit
        donor = panel.donor.to_numpy()
        admitted = np.flatnonzero(strict)
        assert np.array_equal(np.sort(donor[admitted]), admitted)
        e = pd.read_csv(loc / "exposure.csv")
        draws = e[["full_draws", "strict_draws"]].to_numpy().T
        np.testing.assert_array_equal(draws, ck["exposure"])
        np.testing.assert_array_equal(ck["hidden"], r["hidden_counts"])
        assert (
            (draws.sum(1) == 4096).all()
            and (draws[0, fit] > 0).all()
            and not draws[:, val].any()
            and not draws[1, ~strict].any()
        )
        for k in ("condition", "record", "dataset", "subject", "fold"):
            np.testing.assert_array_equal(
                panel[k].to_numpy()[admitted], panel[k].to_numpy()[donor[admitted]]
            )
        del ck
        w = pd.read_csv(loc / "held_windows.csv").set_index("panel_row", verify_integrity=True)
        assert w.index.tolist() == list(panel.index[val])
        for k in meta:
            np.testing.assert_array_equal(w[k], panel.loc[val, k])
        metrics = [k for k in w if k.endswith(("bits", "nmse"))]
        for col in metrics:
            keep = strict[val] if col.endswith("bits") else np.ones(val.sum(), bool)
            assert (
                np.isfinite(w[col].to_numpy()[keep]).all()
                and w[col].iloc[np.flatnonzero(~np.asarray(keep))].isna().all()
            )
        people = grouped(w, [], metrics)
        person_rows.append(people.assign(candidate=c, fold=f))
        values = aggregate(people, [], metrics)
        derived = {"local": {}, "cross": {}}
        for mod in ("eeg", "hb"):
            l = {k: values[f"{mod}_{k}_nmse"] for k in ("trained", "random", "fixed")}
            l["gain"] = min(l["fixed"], l["random"]) - l["trained"]
            b = {
                k: (values[f"{mod}S_{k}_bits"] + values[f"{mod}O_{k}_bits"]) / 2
                for k in ("prior", "true", "wrong", "reverse")
            }
            derived["local"][mod] = l
            derived["cross"][mod] = b
            for family, score in [("local", l), ("cross", b)]:
                for key, v in score.items():
                    np.testing.assert_allclose(
                        [r["scores"][family][mod][key], summary_row[f"scores.{family}.{mod}.{key}"]],
                        v,
                        atol=3e-6,
                        rtol=0,
                    )
            folds.append(
                dict(
                    candidate=c,
                    fold=f,
                    target=mod,
                    **grid[c],
                    **l,
                    **b,
                    prior_gain=b["prior"] - b["true"],
                    pair_gain=b["wrong"] - b["true"],
                    order_gain=b["reverse"] - b["true"],
                )
            )
        scores[c, f, "true_pair"] = derived
        mw = pd.read_csv(loc / "common_masks.csv")
        assert (
            len(mw) == val.sum() * 48
            and np.isfinite(mw.bits).all()
            and not mw.duplicated(["panel_row", "mask", "stream", "repeat"]).any()
        )
        assert set(mw.repeat) == set(range(4))
        cache = mask_tables[f]
        mask_bits = mw.groupby(["panel_row", "mask", "stream"]).bits.mean()
        assert mask_bits.index.equals(cache.index)
        combined = (
            cache.join(mask_bits).reset_index().join(panel[meta], on="panel_row", validate="many_to_one")
        )
        fields = ["prior", "conditional", "markov", "bits"]
        macro = aggregate(
            grouped(combined, ["mask", "stream"], fields), ["mask", "stream"], fields
        ).reset_index()
        mask_rows.append(macro.assign(candidate=c, fold=f, train_mask=grid[c]["mask"]))
        progress = read(loc / "progress.json")
        assert progress["step"] == 1024 and np.isfinite(progress["loss"])
        oper.append(
            dict(
                candidate=c,
                fold=f,
                fit_windows=int(fit.sum()),
                covered=int((draws[0] > 0).sum()),
                held_exposures=int(draws[:, val].sum()),
                strict_windows=int((fit & strict).sum()),
                strict_covered=int((draws[1] > 0).sum()),
                hidden_ids=sum(r["hidden_counts"]),
                last32_train_loss=progress["loss"],
                train_seconds=r["train_seconds"],
                scoring_seconds=r["scoring_seconds"],
                peak_reserved_mib=r["peak_gpu_mib"],
            )
        )
    ranked = nominees(scores, grid, "A", {})
    assert ranked == summary["locks"] == read(run / "A_selection.json")
    tables = dict(
        fold_scores=pd.DataFrame(folds),
        person_scores=pd.concat(person_rows, ignore_index=True),
        common_mask_scores=pd.concat(mask_rows, ignore_index=True),
        fit_audit=pd.DataFrame(oper),
    )
    for name, df in tables.items():
        df.to_csv(out / (name + ".csv"), index=False)
    print("Verified 48 fits, both gates, all checkpoints, exposures, and common masks.", flush=True)


def replay_probes(run, out):
    """Re-fit the fixed ridge readout and recover signed predictions for the nearest local candidates."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import run_aligned_masked_search as search
    import os

    assert (
        search.OUT.resolve() == run.resolve()
        and os.environ["CUDA_VISIBLE_DEVICES"] in search.device_uuids()
    )
    torch.set_float32_matmul_precision("high")
    t = pd.read_csv(out / "fold_scores.csv")
    data = []
    for m, mod in enumerate(("eeg", "hb")):
        g = t[t.target.eq(mod)]
        candidate = g.groupby("candidate").gain.min().idxmax()
        for f in (0, 1):
            dest = run / "A" / candidate / f"fold{f}_seed17_true_pair"
            weights = torch.load(dest / "checkpoint.pt", map_location="cpu", weights_only=False)["model"]
            panel, ids, xy, y, books, tr, va = search.arrays(f)
            prior = search.log_priors(ids, tr, search.CFG["tensor"]["vocab"])
            model = search.reader(search.GRID[candidate], books, prior, "true_pair", 17).eval()
            model.load_state_dict(weights)
            measured = pd.read_csv(dest / "held_windows.csv").set_index("panel_row").loc[va]
            local = panel.loc[va, ["dataset", "subject", "record", "parent", "start"]].copy()
            local["observed"] = y[va, m]
            features = search.vectors(model, ids, xy)[m]
            readout = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(features[tr], y[tr, m])
            local["predicted"] = readout.predict(features[va])
            nmse = np.square(local.observed - local.predicted) / y[tr, m].var()
            np.testing.assert_allclose(nmse, measured[mod + "_trained_nmse"], rtol=3e-4, atol=2e-5)
            data.append(local.assign(candidate=candidate, fit_fold=f, target=mod, panel_row=va))
    pd.concat(data, ignore_index=True).to_csv(out / "held_predictions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--replay-probes", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.replay_probes:
        replay_probes(args.run, args.out)
    else:
        audit(args.run, args.out)
