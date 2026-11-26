"""Signal, decoding, clock-placebo and systemic controls for frozen G0 tokenizers."""

from collections import OrderedDict, defaultdict
import json

import numpy as np
import pandas as pd
import torch
from scipy import signal

import audit_aligned_tokenizer as audit
from analyze_aligned_token_diagnostics import attrs
from src.data.dual_component_tokenizer import base_tokenizer_batch, eeg_spectrum_coordinates
from src.analysis.single_vq_relations import fit_ridge
from src.analysis.frozen_token_readout import Means, ResidualRidge

OUT = audit.OUT


def save(rows, name):
    pd.DataFrame(rows).to_csv(OUT / (name + ".csv"), index=False)


def words(mod, b):
    return pd.read_csv(OUT / f"{mod}_17/{b}_words.csv")


def spectral():
    evaluation = audit.PARENT / "eeg/models/eeg__waveform__vq8192/evaluation"
    cal = json.loads(
        (
            audit.PARENT.parent.parent / "single_vq_spectral_location/20261125_v1/calibration.json"
        ).read_text()
    )
    rows, scales = audit.relations.maps()
    records = []
    for ds, ref in cal.items():
        with np.load(evaluation / f"{ds}_train.npz") as tr, np.load(
            evaluation / f"{ds}_validation.npz"
        ) as te:
            cuts = np.quantile(tr["target"], [0.1, 0.5, 0.9], axis=0)
            y, p = te["target"], te["decoded"]
            ex, valid = te["examples"], te["valid"]
            subjects = np.repeat([rows[r]["subject"] for r, _, _ in ex], valid.sum(1))
            scale = np.repeat(eeg_spectrum_coordinates(rows, ex, scales, cal)["input_scale"], valid.sum(1))
            floor = np.log(1e-12) + 2 * np.log(scale)
            for j, name in enumerate(["log_theta", "log_alpha", "frac_theta", "frac_alpha"]):
                group = np.searchsorted(cuts[:, j], y[:, j])
                for person in np.unique(subjects):
                    for k in range(4):
                        sel = (subjects == person) & (group == k)
                        if not sel.any():
                            continue

                        def hits(v):
                            return (
                                int(np.sum(v[sel, j] * ref["sd"][j] + ref["mean"][j] <= floor[sel] + 1e-4))
                                if j < 2
                                else None
                            )

                        records.append(
                            dict(
                                dataset=ds,
                                subject=person,
                                attribute=name,
                                stratum=k,
                                n=sel.sum(),
                                mse=np.mean((p[sel, j] - y[sel, j]) ** 2),
                                target_floor=hits(y),
                                prediction_floor=hits(p),
                            )
                        )
    save(records, "spectral_strata")


@torch.inference_mode()
def waveform(mod):
    rows, scales = audit.relations.maps()
    model = audit.network(mod, audit.PLAN["models"][mod]["K"], 17, "cuda:0")
    panel = pd.read_csv(OUT / f"{mod}_local_panel.csv", dtype={"subject": str})
    cache = OrderedDict()
    results = []
    interventions = []
    feature = defaultdict(list)
    rng = np.random.default_rng(1126)
    c = 1 if mod == "eeg" else 2
    rate = 200 if c == 1 else 10
    for start in range(0, len(panel), 48):
        g = panel.iloc[start : start + 48]
        ex = [(r.row, r.start, rows[r.row]["positions"].index(r.position)) for r in g.itertuples()]
        raw = base_tokenizer_batch(rows, ex, scales, array_cache=cache)
        x = raw["x"]
        out = model(torch.as_tensor(x, device="cuda:0"))
        assert raw["mask"].all()
        obs = x[:, 7].reshape(len(x), -1, c)
        rms = np.sqrt((obs**2).mean((1, 2)))
        feature["rms"].extend(rms)
        for b in "SO":
            ids = out["ids"][b][:, 7].cpu().numpy()
            feature[b + "id"].extend(ids)
            feature[b + "y"].extend(attrs(raw[b.lower()][:, 7], mod)[1])
            feature[b + "z"].extend(out["latent"][b][:, 7].cpu().numpy())
        for b, key in [("X", "sum"), ("S", "S"), ("O", "O")]:
            v = raw[b.lower()].reshape(len(x), -1, c).astype(float)
            p = out[key].cpu().numpy().reshape(v.shape).astype(float)
            diff = p - v
            delta = np.diff(diff, axis=1) * rate
            full = np.fft.rfftfreq(v.shape[1], 1 / rate)
            taper = np.hanning(v.shape[1])[None, :, None]

            def ps(t):
                return abs(np.fft.rfft((t - t.mean(1, keepdims=True)) * taper, axis=1)) ** 2

            err, truth = ps(diff), ps(v)
            bands = (
                [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
                if c == 1
                else [(0, 0.1), (0.1, 0.5), (0.5, 2)]
            )
            scores = {
                "wave": (np.mean(diff**2, 1), np.mean(v**2, 1)),
                "derivative": (np.mean(delta**2, 1), np.mean((np.diff(v, axis=1) * rate) ** 2, 1)),
            }
            for lo, hi in bands:
                ii = (full >= lo) & (full < hi)
                scores[f"{lo}-{hi}Hz"] = (err[:, ii].sum(1), truth[:, ii].sum(1))
            for i, r in enumerate(g.itertuples()):
                if r.role != "validation":
                    continue
                for k, (e, y) in scores.items():
                    for ch in range(c):
                        results.append(
                            dict(
                                dataset=r.dataset,
                                subject=r.subject,
                                record=r.record,
                                row=r.row,
                                start=r.start,
                                position=r.position,
                                component=b,
                                metric=k,
                                color=ch,
                                error=e[i, ch],
                                target=y[i, ch],
                            )
                        )
        # Keep the center integer unchanged; replace only its neighboring IDs within this batch.
        # Use actual contexts disjoint in the same record and site, from the registered context-pair panel.
    dev = audit.collect(mod, 17, "validation")
    from single_vq_final_c_probes import context_pairs

    pair, _ = context_pairs(dev["examples"], dev["valid"], rows)
    for chunk in np.array_split(pair, max(1, len(pair) // 48)):
        a, donor = [
            base_tokenizer_batch(rows, dev["examples"][ii], scales, array_cache=cache)["x"]
            for ii in chunk.T
        ]
        original, other = [model(torch.as_tensor(v, device="cuda:0")) for v in (a, donor)]
        for b in "SO":
            for kind in ("neighbor", "swap_far", "reverse", "shuffle", "family_near"):
                altered = {k: v.clone() for k, v in original["ids"].items()}
                if kind == "neighbor":
                    altered[b] = other["ids"][b].clone()
                    altered[b][:, 7] = original["ids"][b][:, 7]
                elif kind == "swap_far":
                    altered[b][:, 7] = other["ids"][b][:, 7]
                elif kind == "family_near":
                    fam = words(mod, b).family.to_numpy()
                    for i in range(len(a)):
                        old = int(altered[b][i, 7])
                        candidates = other["ids"][b][i].cpu().numpy()
                        ok = candidates[(fam[candidates] == fam[old]) & (candidates != old)]
                        if len(ok):
                            altered[b][i, 7] = int(ok[0])
                elif kind == "reverse":
                    altered[b] = altered[b].flip([1])
                else:
                    altered[b] = altered[b][:, rng.permutation(15)]
                pred = model.decode_ids(altered)
                assert torch.equal(pred["S" if b == "O" else "O"], original["S" if b == "O" else "O"])
                diff = (pred[b] - original[b]).cpu().numpy()
                for i, (r, t, p) in enumerate(dev["examples"][chunk[:, 0]]):
                    interventions.append(
                        dict(
                            dataset=rows[r]["dataset"],
                            subject=rows[r]["subject"],
                            branch=b,
                            kind=kind,
                            any_change=bool((altered[b][i] != original["ids"][b][i]).any()),
                            central_rms=np.sqrt(np.mean(diff[i, 7] ** 2)),
                            outside_rms=np.sqrt(np.mean(np.delete(diff[i], 7, axis=0) ** 2)),
                        )
                    )
    save(results, mod + "_wave_errors")
    save(interventions, mod + "_decode_controls")
    tr = panel.role.eq("train").to_numpy()
    ds = panel.dataset.to_numpy()
    sub = panel.subject.to_numpy()
    pos = panel.position.to_numpy()
    rms = np.array(feature["rms"])
    bins = np.digitize(rms, [0.5, 1, 2])
    bg = pd.MultiIndex.from_arrays([ds, pos, bins]).factorize()[0]
    names = attrs(np.ones((1, 400 if c == 1 else 40)), mod)[0]
    out = []
    retr = []
    signs = []
    variance = []
    for b in "SO":
        ids = np.array(feature[b + "id"])
        true = np.array(feature[b + "y"])
        book = model.branches[b]["quantizer"].codebook.cpu().numpy()
        mean = Means(bg[tr], true[tr])
        ref = mean.lookup(bg[~tr])[0]
        emb = book[ids]
        perm = book[rng.permutation(len(book))][ids]
        for method, X in [
            ("embedding", emb),
            ("random_geometry", perm),
            ("latent", np.array(feature[b + "z"])),
        ]:
            rd = ResidualRidge(bg[tr], X[tr], true[tr], mean)
            pred = rd.predict(bg[~tr], X[~tr], rd.coefficients(0.01))
            for dataset, person in sorted(set(zip(ds[~tr], sub[~tr]))):
                sel = (ds[~tr] == dataset) & (sub[~tr] == person)
                for name, err, base in zip(
                    names,
                    ((pred[sel] - true[~tr][sel]) ** 2).mean(0),
                    ((ref[sel] - true[~tr][sel]) ** 2).mean(0),
                ):
                    out.append(
                        dict(
                            branch=b,
                            dataset=dataset,
                            subject=person,
                            method=method,
                            attribute=name,
                            mse=err,
                            baseline=base,
                        )
                    )
                if c == 2:
                    ix = [names.index("HbO_slope"), names.index("HbR_slope")]
                    for m, arr in [("model", pred), ("background", ref)]:
                        signs.append(
                            dict(
                                branch=b,
                                dataset=dataset,
                                subject=person,
                                method=method,
                                readout=m,
                                n=sel.sum(),
                                correct_both=np.mean(
                                    (np.sign(arr[sel][:, ix]) == np.sign(true[~tr][sel][:, ix])).all(1)
                                ),
                            )
                        )
        w = words(mod, b)
        j = names.index("alpha_frac" if c == 1 else "HbO_slope")
        for dataset in np.unique(ds):
            cut = np.quantile(true[tr & (ds == dataset), j], [0.25, 0.5, 0.75])
            lbl = np.searchsorted(cut, true[:, j])
            for i in np.flatnonzero((~tr) & (ds == dataset)):
                gallery = np.flatnonzero(
                    tr
                    & (ds == dataset)
                    & (pos == pos[i])
                    & (abs(np.log(np.maximum(rms, 1e-12) / max(rms[i], 1e-12))) <= np.log(1.4))
                )
                if len(gallery) < 5:
                    continue
                for method, dist in [
                    ("embedding", np.linalg.norm(emb[gallery] - emb[i], axis=1)),
                    ("amplitude", abs(rms[gallery] - rms[i])),
                ]:
                    hit = lbl[gallery[np.argsort(dist, kind="stable")]] == lbl[i]
                    retr.append(
                        dict(
                            branch=b,
                            dataset=dataset,
                            subject=sub[i],
                            query=i,
                            method=method,
                            gallery=len(hit),
                            p5=hit[:5].mean(),
                            chance=hit.mean(),
                            AP=(np.cumsum(hit) / np.arange(1, len(hit) + 1) * hit).sum()
                            / max(hit.sum(), 1),
                        )
                    )
        # Conditional development residual from the training-only background, within and between words.
        ref_all = mean.lookup(bg[~tr])[0][:, j]
        dev_id = ids[~tr]
        resid = true[~tr, j] - ref_all
        for threshold, mask in [
            ("rare", w.train_patches < 20),
            ("medium", w.train_patches.between(20, 99)),
            ("frequent", w.train_patches >= 100),
        ]:
            sel = mask.to_numpy()[dev_id]
            for d, p in sorted(set(zip(ds[~tr], sub[~tr]))):
                m = sel & (ds[~tr] == d) & (sub[~tr] == p)
                if m.sum() < 2:
                    continue
                f = pd.DataFrame({"id": dev_id[m], "v": resid[m]})
                cg = f.groupby("id").v.transform("mean")
                variance.append(
                    dict(
                        dataset=d,
                        subject=p,
                        branch=b,
                        support=threshold,
                        n=len(f),
                        n_ids=f.id.nunique(),
                        within=((f.v - cg) ** 2).mean(),
                        between=((cg - f.v.mean()) ** 2).mean(),
                    )
                )
    save(out, mod + "_random_readout")
    save(retr, mod + "_matched_retrieval")
    save(variance, mod + "_variance")
    if signs:
        save(signs, "hb_signs")
    if mod == "eeg":
        spectral()


@torch.inference_mode()
def eye_and_shift():
    import audit_projected_so_members as source

    model = audit.network("eeg", 8192, 17, "cuda:0")
    m, unit = source.prepare("cuda:0", network=model)
    v = m.role.eq("validation").to_numpy()
    old = pd.read_csv(source.OUT / "validation_members.csv.gz", dtype={"subject": str})
    keys = ["example", "block", "parent", "site"]
    assert old[keys].reset_index(drop=True).equals(m.loc[v, keys].reset_index(drop=True))
    eog = np.full(len(m), np.nan)
    eog[v] = old.high_eog
    metrics = []
    person_rows = []
    matches = []
    for b in "SO":
        for name, token, train, template in source.select(m, unit, b):
            if name != "slow":
                continue
            support = v & np.isfinite(eog) & m.site.str.upper().str.startswith(("FP", "AF")).to_numpy()
            l = support & m[b].eq(token).to_numpy()
            r = support & ~m[b].eq(token).to_numpy()
            source.summarize(
                m,
                eog,
                l,
                r,
                source.match(m, l, r),
                dict(branch=b, endpoint="eye", token=token),
                metrics,
                person_rows,
                matches,
            )
    save(metrics, "eye_contrasts")
    save(person_rows, "eye_people")
    x = np.load(source.SOURCE / "x.npy", mmap_mode="r").reshape(-1, 15, 400)

    # Frozen signal detector: 8-13 Hz analytic envelope, >=0.25 s above train-specific 90th percentile.
    # Fit threshold on normalized envelopes, hence invariant to patch gain.
    def burst(w):
        filtered = signal.sosfiltfilt(
            signal.butter(3, [8, 13], fs=200, btype="bandpass", output="sos"), w, axis=-1
        )
        env = np.abs(signal.hilbert(filtered, axis=-1)) / np.maximum(
            np.sqrt((w * w).mean(1, keepdims=True)), 1e-10
        )
        return env[:, 40:-40]

    envelope = np.concatenate(
        [
            burst(np.array(x[m.example.values[idx], m.block.values[idx]]))
            for idx in np.array_split(np.arange(len(m)), 180)
        ]
    )
    det = np.zeros(len(m), bool)
    for d, g in m.groupby("dataset"):
        cut = np.quantile(envelope[g.index[g.role.eq("train")]], 0.90)
        high = envelope[g.index] > cut
        s = np.pad(np.cumsum(high, 1), ((0, 0), (1, 0)))
        det[g.index] = (s[:, 50:] - s[:, :-50]).max(1) >= 50
    rr = []
    for b in "SO":
        fam = words("eeg", b).family.to_numpy()[m[b]]
        for p, g in m[v].groupby("person"):
            ii = g.index
            hit = fam[ii] == 3
            y = det[ii]
            l = v & (m.person == p).to_numpy() & (fam == 3)
            r = v & (m.person == p).to_numpy() & (fam != 3)
            pairs = source.match(m, l, r)
            rr.append(
                dict(
                    branch=b,
                    person=p,
                    n=len(ii),
                    positives=y.sum(),
                    selected=hit.sum(),
                    intersection=(hit & y).sum(),
                    pairs=len(pairs),
                    matched_member=det[pairs[:, 0]].mean() if len(pairs) else np.nan,
                    matched_control=det[pairs[:, 1]].mean() if len(pairs) else np.nan,
                )
            )
    save(rr, "detector")
    meta = pd.read_csv(source.SOURCE / "examples.csv", dtype={"subject": str})
    valid = np.load(source.SOURCE / "mask.npy")
    res = []
    for (d, p, site, parent), g in meta[meta.role.ne("train")].groupby(
        ["dataset", "subject", "site", "parent"]
    ):
        a = int(g[g.start == 0].index[0])
        z = int(g[g.start == 2].index[0])
        ok = valid[a, 1:] & valid[z, :-1]
        np.testing.assert_array_equal(x[a, 1:][ok], x[z, :-1][ok])
        ids = model(torch.tensor(x[[a, z]], device="cuda:0"))["ids"]
        for b in "SO":
            ia, ib = ids[b].cpu().numpy()
            wa, wb = ia[1:][ok], ib[:-1][ok]
            f = words("eeg", b).family.to_numpy()
            for k, edge in [
                ("edge", np.isin(np.flatnonzero(ok), [0, 13])),
                ("internal", ~np.isin(np.flatnonzero(ok), [0, 13])),
            ]:
                if edge.any():
                    res.append(
                        dict(
                            dataset=d,
                            subject=p,
                            position=site,
                            branch=b,
                            region=k,
                            n=edge.sum(),
                            id_matches=(wa[edge] == wb[edge]).sum(),
                            family_matches=(f[wa[edge]] == f[wb[edge]]).sum(),
                        )
                    )
    save(res, "boundary_shift")


@torch.inference_mode()
def event_controls():
    import single_vq_event_coupling as ev
    from scipy.stats import gamma

    folder = OUT / "event_controls"
    folder.mkdir(exist_ok=True)
    seed = 17
    net = audit.network("eeg", 8192, seed, "cuda:0")
    fam = words("eeg", "S").family.to_numpy()
    pairs = pd.read_csv(OUT / "events_17/coupling_pairs.csv").query("donor_event>=0")
    ledger = []
    derived = []
    features = defaultdict(list)
    for key, events, record, markers in ev.record_tasks(folder):
        track = ev.eeg_track(ev.SUPPORTED[key])
        role, sub, rec = key
        with np.load(OUT / "events_17/events_by_record" / f"{role}_{sub}_{rec}.npz") as z:
            meta = [json.loads(v) for v in z["D"]]
            wanted = set(
                pairs[(pairs.split == role) & (pairs.subject == sub) & (pairs.record == rec)].event
            )
            raw = ev.CleanPhysiologyCacheIndex.load_record_arrays(
                record, ("fnirs", "fnirs_supported_channels")
            )
            names = record.manifest["measurement"]["fnirs_channel_names"]
            columns = [
                [
                    i
                    for i, n in enumerate(names)
                    if n.endswith("_" + clr)
                    and n.rsplit("_", 1)[0] not in ev.HB
                    and raw["fnirs_supported_channels"][i]
                ]
                for clr in ("HbO", "HbR")
            ]
            # Median outside the two target positions; only samples before the prediction cutoff enter the model.
            for i, r in enumerate(meta):
                if r["event"] not in wanted:
                    continue
                h = round(r["hb_marker"] * 10)
                vals = []
                for t in (-2, 0, 2):
                    v = raw["fnirs"][h + t * 10 : h + (t + 2) * 10]
                    vals.extend([np.median(v[:, ch].mean(0)) for ch in columns])
                hist = ev.context(track, r["eeg_marker"], -26, 2)
                continuous = ev.conventional(hist)
                # HRF-like fixed double gamma sampled at each 2 s bin midpoint, relative to the middle of [10,12].
                dt = 11 - (-26 + np.arange(15) * 2 + 1)
                weights = gamma.pdf(dt, 6) - gamma.pdf(dt, 16) / 6
                hemo = np.einsum("cpf,p->cf", continuous, weights).ravel()
                if not np.isfinite(vals).all():
                    continue
                derived.append(r)
                features["global"].append(vals)
                features["hrf"].append(hemo)
                for k in ("base", "aligned", "continuous", "y"):
                    features[k].append(z[k][i])
            if role != "validation":
                continue
            available = {json.loads(v)["event"] for v in z["B"]}
            for event, native in events:
                number = event["event_index"]
                if number not in available:
                    continue
                t = event["marker_ms"]["eeg"] / 1000
                for off in (0, -13, 13):
                    crop = ev.context(track, t + off, -6, 4)
                    usable = (
                        crop is not None
                        and ev.window_within_alignment_support(native, off - 6, 30)
                        and (off == 0 or np.min(abs(markers - (t + off))) >= 8)
                    )
                    val = np.nan
                    if usable:
                        ids = net(torch.as_tensor(crop.copy(), device="cuda:0"))["ids"]["S"].cpu().numpy()
                        q = fam[ids[2:]] == 3
                        val = q[:, 4:8].mean() - q[:, 1:3].mean()
                    ledger.append(
                        dict(
                            subject=sub,
                            record=rec,
                            event=number,
                            condition=event["condition"],
                            offset=off,
                            used=usable,
                            delta=val,
                        )
                    )
    save(ledger, "pseudo_cues")
    d = pd.DataFrame(derived)
    a = {k: np.asarray(v) for k, v in features.items()}
    tr = d.split.eq("train").to_numpy()
    ix = ev.LAGS.index(
        json.loads((OUT / "events_17/events_summary.json").read_text())["D"]["selected_lag_s"]
    )
    y = a["y"][:, ix]
    var = y[tr].var(0)
    output = []
    for name, pieces in {
        "base": ["base"],
        "aligned": ["base", "aligned"],
        "continuous": ["base", "continuous"],
        "HRF": ["base", "hrf"],
        "global": ["base", "global"],
        "global_HRF": ["base", "global", "hrf"],
        "global_EEG": ["base", "global", "aligned"],
    }.items():
        X = np.column_stack([a[k] for k in pieces])
        pred = fit_ridge(X[tr], y[tr])(X[~tr])
        s = d[~tr].copy()
        for j, clr in enumerate(["HbO", "HbR"]):
            s["loss"] = (pred[:, j] - y[~tr, j]) ** 2 / var[j]
            q = (
                s.groupby(["subject", "record", "condition"])
                .loss.mean()
                .unstack()
                .dropna()
                .mean(1)
                .groupby("subject")
                .mean()
            )
            for p, loss in q.items():
                output.append(
                    dict(method=name, color=clr, subject=p, nmse=loss, train=tr.sum(), dev=(~tr).sum())
                )
    save(output, "systemic_controls")


def run(mod, stage):
    if stage == "waveform":
        waveform(mod)
    elif stage == "eye":
        eye_and_shift()
    elif stage == "event_controls":
        event_controls()
