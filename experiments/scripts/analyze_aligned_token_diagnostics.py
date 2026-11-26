"""Paired signal, assignment, and geometry diagnostics for the frozen aligned audit."""

from collections import OrderedDict, defaultdict
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import audit_aligned_tokenizer as audit
from single_vq_final_c_probes import context_pairs
from src.data.dual_component_tokenizer import base_tokenizer_batch

OUT = audit.OUT
SEED = 1126


def attrs(x, mod):
    """Inputs are batches of complete 2s patches in frozen G0 coordinates."""
    x = np.asarray(x, float)
    if mod == "hb":
        y = x.reshape(-1, 20, 2)
        t = (np.arange(20) - 9.5) / 10
        cols = {}
        for i, c in enumerate(("HbO", "HbR")):
            v = y[:, :, i]
            for k, values in dict(
                level=v.mean(1),
                slope=v @ t / (t @ t),
                peak=v.max(1),
                peak_time=v.argmax(1) / 10,
                release=v[:, -1] - v.max(1),
                curvature=np.abs(np.diff(v, n=2)).mean(1),
            ).items():
                cols[c + "_" + k] = values
        cols["peak_lag"] = cols["HbR_peak_time"] - cols["HbO_peak_time"]
        cols["color_cov"] = ((y - y.mean(1, keepdims=True)).prod(2)).mean(1)
        return list(cols), np.column_stack(list(cols.values()))
    v = x - x.mean(1, keepdims=True)
    f = np.fft.rfftfreq(400, 1 / 200)
    w = np.hanning(400)
    power = abs(np.fft.rfft(v * w)) ** 2 * 2 / (400 * (w @ w))
    total = power[:, (f >= 1) & (f <= 45)].sum(1)
    cols = {}
    for name, lo, hi in [
        ("delta", 1, 4),
        ("theta", 4, 8),
        ("alpha", 8, 13),
        ("beta", 13, 30),
        ("low_gamma", 30, 45),
    ]:
        e = power[:, (f >= lo) & (f < hi)].sum(1)
        cols[name + "_power"] = e
        cols[name + "_frac"] = e / np.maximum(total, 1e-15)
    # Local periodic contrast to a log-log straight line fitted on flanks, outside 6–14 Hz.
    flank = ((f >= 2) & (f <= 5)) | ((f >= 15) & (f <= 30))
    band = (f >= 6) & (f <= 14)
    coef = np.linalg.lstsq(
        np.stack((np.ones(flank.sum()), np.log10(f[flank])), 1),
        np.log10(np.maximum(power[:, flank], 1e-15)).T,
        rcond=None,
    )[0]
    contrast = 10 * (
        np.log10(np.maximum(power[:, band], 1e-15))
        - coef[0, :, None]
        - coef[1, :, None] * np.log10(f[band])
    )
    pk = contrast.argmax(1)
    peak = contrast[np.arange(len(x)), pk]
    cols.update(
        peak_hz=f[band][pk],
        peak_db=peak,
        aperiodic_slope=coef[1],
        halfpower_width=np.array(
            [
                np.sum(np.cumprod(z[i:] >= p - 3)) + np.sum(np.cumprod(z[: i + 1][::-1] >= p - 3)) - 1
                for z, i, p in zip(contrast, pk, peak)
            ]
        )
        * 0.5,
    )
    env = np.sqrt(np.mean(v.reshape(-1, 8, 50) ** 2, 2))
    cols.update(
        rms=np.sqrt(np.mean(x * x, 1)),
        burst_peak=env.max(1),
        burst_seconds=0.25 * (env > 1.5 * np.maximum(np.median(env, axis=1, keepdims=True), 1e-12)).sum(1),
        peak_to_peak=np.ptp(x, axis=1),
        slope=np.abs(np.diff(x)).mean(1) * 200,
        curvature=np.abs(np.diff(x, n=2)).mean(1) * 40000,
        zero_cross=np.mean((v[:, 1:] * v[:, :-1]) < 0, 1),
    )
    return list(cols), np.column_stack(list(cols.values()))


def write_frame(rows, name):
    pd.DataFrame(rows).to_csv(OUT / name, index=False)


@torch.inference_mode()
def local(mod):
    rows, scales = audit.relations.maps()
    model = audit.network(mod, audit.PLAN["models"][mod]["K"], 17, "cuda:0")
    rng = np.random.default_rng(SEED)
    data = {}
    ledger = []
    edits = []
    mechanism = []
    cache = OrderedDict()
    for role in ("train", "validation"):
        a = audit.collect(mod, 17, role)
        # Same archive panel and admissible/disjoint same-record donors as the C audit.
        if role == "validation":
            pairs, den = context_pairs(a["examples"], a["valid"], rows)
            write_frame(den, mod + "_context_support.csv")
            pick, donor = pairs.T
        else:
            ix = np.flatnonzero(a["valid"].all(1))
            pick = np.concatenate(
                [
                    rng.choice(
                        ix[a["persons"][ix] == p], min(32, (a["persons"][ix] == p).sum()), replace=False
                    )
                    for p in np.unique(a["persons"][ix])
                ]
            )
        store = defaultdict(list)
        for off in range(0, len(pick), 48):
            ind = pick[off : off + 48]
            ex = a["examples"][ind]
            meta = [
                dict(
                    dataset=rows[r]["dataset"],
                    subject=rows[r]["subject"],
                    record=rows[r]["record"],
                    position=rows[r]["positions"][p],
                    row=int(r),
                    start=int(t),
                )
                for r, t, p in ex
            ]
            ledger.extend(dict(role=role, **m) for m in meta)
            raw = base_tokenizer_batch(rows, ex, scales, array_cache=cache)
            x = raw["x"]
            v = torch.from_numpy(x).to("cuda:0")
            res = model(v)
            center = x[:, 7]
            names, feat = attrs(center, mod)
            store["observed"].append(feat)
            for b in "SO":
                z = res["latent"][b][:, 7].cpu().numpy()
                ids = res["ids"][b][:, 7].cpu().numpy()
                store[b + "_id"].append(ids)
                store[b + "_latent"].append(z)
                store[b + "_target"].append(attrs(raw[b.lower()][:, 7], mod)[1])
                store[b + "_decoded"].append(attrs(res[b][:, 7].cpu().numpy(), mod)[1])
            store["bg"].append(
                np.array(
                    [
                        audit.readout.SITES[mod].get(
                            (m["dataset"], m["position"]), len(audit.readout.SITES[mod])
                        )
                        * 4
                        for m in meta
                    ]
                )
                + np.digitize(np.sqrt((center**2).mean(1)), [0.5, 1, 2])
            )
            store["dataset"].extend(m["dataset"] for m in meta)
            store["subject"].extend(m["subject"] for m in meta)
            if role == "train":
                continue
            other = base_tokenizer_batch(
                rows, a["examples"][donor[off : off + len(ind)]], scales, array_cache=cache
            )["x"]
            swapped = other.copy()
            swapped[:, 7] = center

            def assess(tag, altered, level):
                cur = model(torch.from_numpy(altered.astype("float32")).to("cuda:0"))
                for b in "SO":
                    original = res["ids"][b][:, 7].cpu().numpy()
                    new = cur["ids"][b][:, 7].cpu().numpy()
                    for k, m in enumerate(meta):
                        edits.append(
                            dict(
                                **m,
                                branch=b,
                                kind=tag,
                                dose=level,
                                word=int(original[k]),
                                new=int(new[k]),
                                same=int(new[k] == original[k]),
                            )
                        )

            assess("identity", x, 0)
            assess("context", swapped, 1)
            channels = 1 if mod == "eeg" else 2
            wave = center.reshape(len(center), -1, channels)
            base = wave.mean(1, keepdims=True)
            rms = np.sqrt(((wave - base) ** 2).mean(1, keepdims=True))
            noise = rng.standard_normal(wave.shape)
            for level in (0.05, 0.20, 0.50):
                y = x.copy()
                y[:, 7] = (wave + level * rms * noise).reshape(len(x), -1)
                assess("noise", y, level)
                y[:, 7] = (base + (1 + level) * (wave - base)).reshape(len(x), -1)
                assess("amplitude", y, level)
            p = np.fft.rfft(wave.astype(np.float64), axis=1)
            phase = rng.uniform(-np.pi, np.pi, size=(len(x), p.shape[1], 1))
            phase[:, [0, -1]] = 0
            sur = np.fft.irfft(p * np.exp(1j * phase), n=wave.shape[1], axis=1)
            np.testing.assert_allclose(abs(np.fft.rfft(sur, axis=1)), abs(p), rtol=1e-7, atol=1e-7)
            y = x.copy()
            y[:, 7] = sur.reshape(len(x), -1)
            assess("phase", y, 1)
            if mod == "eeg":
                t = np.arange(400) / 200
                # Equal RMS increment to the center, either a localized 10 Hz burst or broadband noise.
                for tag, s in [
                    (
                        "burst10",
                        np.tile(
                            np.sin(2 * np.pi * 10 * t) * np.exp(-0.5 * ((t - 1) / 0.15) ** 2), (len(x), 1)
                        ),
                    ),
                    ("equal_noise", noise[:, :, 0]),
                ]:
                    s = s - s.mean(1, keepdims=True)
                    y[:, 7] = center + 0.25 * rms[:, :, 0] * s / np.sqrt((s * s).mean(1, keepdims=True))
                    assess(tag, y, 0.25)
            # Change one integer in a real sequence. Donor from same record/site, disjoint time.
            d = model(torch.from_numpy(other).to("cuda:0"))["ids"]
            for b in "SO":
                stream = {k: v.clone() for k, v in res["ids"].items()}
                stream[b][:, 7] = d[b][:, 7]
                cur = model.decode_ids(stream)
                orig = res[b].cpu().numpy()
                delta = cur[b].cpu().numpy() - orig
                target = raw[b.lower()]
                for k, m in enumerate(meta):
                    mechanism.append(
                        dict(
                            **m,
                            branch=b,
                            changed=int(stream[b][k, 7] != res["ids"][b][k, 7]),
                            center_change_rms=np.sqrt((delta[k, 7] ** 2).mean()),
                            offcenter_change_rms=np.sqrt((np.delete(delta[k], 7, axis=0) ** 2).mean()),
                            target_mse_delta=np.mean((cur[b][k, 7].cpu().numpy() - target[k, 7]) ** 2)
                            - np.mean((orig[k, 7] - target[k, 7]) ** 2),
                        )
                    )
        data[role] = {
            k: np.array(v) if k in ("dataset", "subject") else np.concatenate(v) for k, v in store.items()
        }
    write_frame(ledger, mod + "_local_panel.csv")
    write_frame(edits, mod + "_perturb.csv")
    write_frame(mechanism, mod + "_edit.csv")
    from src.analysis.frozen_token_readout import Means, ResidualRidge

    tr, te = data["train"], data["validation"]
    rng = np.random.default_rng(SEED)
    result = []
    retrieval = []
    for b in "SO":
        book = model.branches[b]["quantizer"].codebook.cpu().numpy()
        train, target = tr[b + "_target"], te[b + "_target"]
        mean = Means(tr["bg"], train)
        ref = mean.lookup(te["bg"])[0]
        pred = {"decoded": te[b + "_decoded"]}
        for rep, tf, vf in [
            ("latent", tr[b + "_latent"], te[b + "_latent"]),
            ("embedding", book[tr[b + "_id"]], book[te[b + "_id"]]),
            ("observed", tr["observed"], te["observed"]),
        ]:
            rd = ResidualRidge(tr["bg"], tf, train, mean)
            pred[rep] = rd.predict(te["bg"], vf, rd.coefficients(0.01))
        for person in np.unique(te["subject"] + "@" + te["dataset"]):
            sub, ds = person.split("@")
            hit = (te["subject"] == sub) & (te["dataset"] == ds)
            for rep, vals in pred.items():
                for name, mse, baseline in zip(
                    names, ((vals[hit] - target[hit]) ** 2).mean(0), ((ref[hit] - target[hit]) ** 2).mean(0)
                ):
                    result.append(
                        dict(
                            branch=b,
                            subject=sub,
                            dataset=ds,
                            method=rep,
                            attr=name,
                            mse=mse,
                            baseline=baseline,
                            n=int(hit.sum()),
                        )
                    )
        # Labels and classes defined from training branch alpha fraction or HbO slope within dataset.
        j = names.index("alpha_frac" if mod == "eeg" else "HbO_slope")
        for ds in sorted(set(te["dataset"])):
            t = tr["dataset"] == ds
            v = te["dataset"] == ds
            train_y = train[t, j]
            test_y = target[v, j]
            bounds = np.quantile(train_y, [0.25, 0.5, 0.75])
            lab = np.searchsorted(bounds, train_y)
            truth = np.searchsorted(bounds, test_y)
            for method, x, y in [
                ("embedding", book[tr[b + "_id"][t]], book[te[b + "_id"][v]]),
                ("signal", tr["observed"][t], te["observed"][v]),
                (
                    "amplitude",
                    tr["observed"][t][:, [names.index("rms" if mod == "eeg" else "HbO_level")]],
                    te["observed"][v][:, [names.index("rms" if mod == "eeg" else "HbO_level")]],
                ),
            ]:
                ss = StandardScaler().fit(x)
                dist = cdist(ss.transform(y), ss.transform(x))
                rank = lab[np.argsort(dist, axis=1, kind="stable")] == truth[:, None]
                ap = (np.cumsum(rank, axis=1) / np.arange(1, len(x) + 1) * rank).sum(1) / np.maximum(
                    rank.sum(1), 1
                )
                for person in np.unique(te["subject"][v]):
                    select = te["subject"][v] == person
                    retrieval.append(
                        dict(
                            branch=b,
                            dataset=ds,
                            subject=person,
                            method=method,
                            n=int(select.sum()),
                            gallery=len(x),
                            p5=rank[select, :5].mean(),
                            mAP=ap[select].mean(),
                        )
                    )
    write_frame(result, mod + "_attribute_probes.csv")
    write_frame(retrieval, mod + "_retrieval.csv")


def extended(mod):
    # Reuse exact matched members, never compare arbitrary integer labels across fits.
    from scipy.spatial.distance import pdist

    a = audit.collect(mod, 17, "validation")
    results = []
    timing = []
    stable = []
    size = audit.PLAN["models"][mod]["K"]
    model = audit.network(mod, size, 17)
    rng = np.random.default_rng(SEED)
    old = (
        audit.PARENT.parent.parent
        / ("single_vq_final_c_audit" if mod == "eeg" else "single_vq_relations")
        / "20261125_v1"
    )
    for j, b in enumerate("SO"):
        words = pd.read_csv(OUT / f"{mod}_17/{b}_words.csv")
        book = model.branches[b]["quantizer"].codebook.detach().numpy()
        keys = words.index[(words.eligible) & (words.dev_people >= 3)].to_numpy()
        sub = keys if len(keys) < 1500 else rng.choice(keys, 1500, replace=False)
        target = words.dev_mean.to_numpy()[sub]
        dx = pdist(book[sub], "cosine")
        dy = pdist(target[:, None])
        corr = spearmanr(dx, dy).statistic
        ix = (
            NearestNeighbors(n_neighbors=2, metric="cosine")
            .fit(book[sub])
            .kneighbors(book[sub], return_distance=False)[:, 1]
        )
        results.append(
            dict(
                branch=b,
                words=len(sub),
                eligible=len(keys),
                rho=corr,
                nearest_abs_error=np.mean(abs(target - target[ix])),
                shuffled_abs_error=np.mean(abs(target - rng.permutation(target))),
            )
        )
        hist = {
            seed: pd.read_csv(old / f"{mod}_{seed}/{b}_words.csv").family_score.values for seed in (17, 43)
        }
        # Historical integer assignments are paired on exactly the same patch identities.
        all_ids = {17: [], 43: []}
        for t in audit.tasks("validation"):
            if t[0] != mod:
                continue
            with np.load(old / "inputs" / audit.readout.destination(t).name) as z, np.load(
                audit.inputs(t)
            ) as new:
                for k in ("examples", "valid"):
                    np.testing.assert_array_equal(z[k], new[k])
                for seed in hist:
                    all_ids[seed].append(z[f"{b}_{seed}_id"])
        current = words.family_score.to_numpy()[a["ids"][:, j]]
        for seed in hist:
            other = hist[seed][np.concatenate(all_ids[seed])]
            for (ds, subj), pid in audit.readout.PEOPLE.items():
                sel = a["person"] == pid
                if sel.any():
                    stable.append(
                        dict(
                            branch=b,
                            historical_seed=seed,
                            dataset=ds,
                            subject=subj,
                            n=sel.sum(),
                            rho=spearmanr(current[sel], other[sel]).statistic,
                        )
                    )
        f = words.family.to_numpy()
        fam = np.full(a["valid"].shape, -1)
        fam[a["valid"]] = f[a["ids"][:, j]]
        obs = np.full(a["valid"].shape, np.nan)
        obs[a["valid"]] = a["y"][:, 0]
        train = audit.collect(mod, 17, "train")
        q = np.quantile(train["y"][:, 0], [0.25, 0.5, 0.75])
        observed = np.searchsorted(q, obs)
        rows, _ = audit.relations.maps()
        for i in np.flatnonzero(a["valid"].all(1)):
            r, t, p = a["examples"][i]
            row = rows[r]
            for name, seq in [
                ("token", fam[i]),
                ("observed", observed[i]),
                ("bag_null", rng.permutation(fam[i])),
            ]:
                cuts = np.flatnonzero(np.r_[True, seq[1:] != seq[:-1], True])
                dwell = np.diff(cuts)
                returns = [v for label in range(4) for v in np.diff(np.flatnonzero(seq == label))]
                timing.append(
                    dict(
                        branch=b,
                        dataset=row["dataset"],
                        subject=row["subject"],
                        method=name,
                        dwell_s=2 * dwell.mean(),
                        return_s=2 * np.mean(returns) if returns else np.nan,
                        transition_prob=np.mean(seq[1:] != seq[:-1]),
                    )
                )
    write_frame(results, mod + "_geometry.csv")
    write_frame(stable, mod + "_historical.csv")
    pd.DataFrame(timing).groupby(["branch", "dataset", "subject", "method"]).agg(
        n=("dwell_s", "size"),
        dwell_s=("dwell_s", "mean"),
        return_s=("return_s", "mean"),
        transition_prob=("transition_prob", "mean"),
    ).reset_index().to_csv(OUT / (mod + "_dynamics.csv"), index=False)


def analyze(stage, mod):
    if stage == "local":
        local(mod)
    elif stage == "extended":
        extended(mod)
    else:
        from analyze_aligned_token_pairing import cross

        cross()
