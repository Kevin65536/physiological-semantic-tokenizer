"""Native-position and dual-clock frozen-token analyses on common v3 crops."""

from collections import defaultdict, OrderedDict
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import audit_aligned_tokenizer as audit
from src.analysis.single_vq_relations import fit_ridge, wave_attribute
from src.data.dual_component_tokenizer import base_tokenizer_batch
from src.data.channel_geometry import project_head_topview
from src.data.unified_physiology import ChannelGeometryIndex
from src.analysis.observation_task_probe import DOWNSTREAM_ROOT
from run_dual_component_tokenizer import read_json

OUT = audit.OUT


def panel(*, split_filter=None, background_limit=12):
    rows, scales = audit.relations.maps()
    groups = defaultdict(dict)
    rng = np.random.default_rng(1126)
    for i, row in enumerate(rows):
        if split_filter is not None and row["split"] != split_filter:
            continue
        for crop in row["inputs"]["score"]:
            if len(crop["positions"]) >= 4:
                groups[row["parent"], crop["start_s"]][row["modality"]] = (i, crop["positions"])
    cfg, _ = audit.aligned.settings("eeg")
    clocks = {
        p["row"]["key"]: p["metadata"]["source"]["members"][0]["window"]
        for p in read_json(audit.aligned.ROOT / cfg["data"]["run"] / "plan.json")["windows"]
        if len(p["metadata"]["source"]["members"]) == 1
    }
    samples = read_json(DOWNSTREAM_ROOT / "input_plan.json")["samples"]
    trials = defaultdict(list)
    for s in samples:
        if (
            s["split"] in ("train", "validation")
            and s["task"] in ("mental_arithmetic", "nback")
            and not s["partial"]
        ):
            trials[s["dataset"], s["split"], s["subject"], s["record"]].append(s)
    chosen = []
    by_person = defaultdict(list)
    event_keys = set()
    for (parent, offset), pair in groups.items():
        if set(pair) != {"eeg", "hb"} or parent not in clocks:
            continue
        r = rows[pair["eeg"][0]]
        ds, split, subj, rec = (r[k] for k in ("dataset", "split", "subject", "record"))
        clock = clocks[parent]
        times = np.array([r["native_start_sample"] / r["native_rate_hz"], clock["hb_start_s"]]) + offset
        sample = dict(
            dataset=ds,
            split=split,
            subject=subj,
            record=rec,
            parent=parent,
            start=offset,
            eeg_start=times[0],
            hb_start=times[1],
            condition="",
            event=-1,
        )
        matches = []
        for e in trials[ds, split, subj, rec]:
            ref = np.array([e["event"]["eeg_time_ms"], e["event"]["fnirs_time_ms"]]) / 1000
            rel = ref - times
            # Native clocks must each place the unique cue inside the first 12 seconds.
            if np.all((rel >= 0) & (rel <= 12)) and np.ptp(rel) <= 0.15:
                matches.append(e)
        if len(matches) == 1:
            e = matches[0]
            key = e["id"]
            if key in event_keys:
                raise ValueError("event matched twice")
            event_keys.add(key)
            sample.update(
                condition=e["condition"],
                event=e["event_index"],
                cue=e["event"]["eeg_time_ms"] / 1000,
                hb_cue=e["event"]["fnirs_time_ms"] / 1000,
            )
            chosen.append((sample, pair))
        else:
            by_person[ds, split, subj].append((sample, pair))
    for values in by_person.values():
        if background_limit is None:
            chosen.extend(values)
        else:
            chosen.extend(
                values[k]
                for k in rng.choice(len(values), min(len(values), background_limit), replace=False)
            )
    return chosen


def native_coordinates():
    geo = ChannelGeometryIndex(DOWNSTREAM_ROOT)

    @lru_cache(maxsize=4096)
    def coords(ds, subj, rec, mod, names):
        geometry = geo.for_channels(
            record=SimpleNamespace(dataset_id=ds, canonical_subject_id=subj, base_record_id=rec),
            modality="eeg" if mod == "eeg" else "fnirs",
            channel_names=names,
        )
        return np.array(
            [project_head_topview(r) if r["position_available"] else [np.nan, np.nan] for r in geometry]
        )

    return coords


def four_landmarks(xy, positions):
    pos = np.array([p for p in positions if np.isfinite(xy[p]).all()], dtype=int)
    if len(pos) < 4:
        return pos[:0]
    from scipy.optimize import linear_sum_assignment

    targets = [[-0.42, 0.60], [0.42, 0.60], [-0.42, -0.60], [0.42, -0.60]]
    _, index = linear_sum_assignment(cdist(targets, xy[pos]))
    return pos[index]


def cross():
    rng = np.random.default_rng(1126)
    cache = OrderedDict()
    rows, scales = audit.relations.maps()
    coords = native_coordinates()

    books = {}
    networks = {}
    for mod in ("eeg", "hb"):
        networks[mod] = audit.network(mod, audit.PLAN["models"][mod]["K"], 17, "cuda:0")
        for b in "SO":
            w = pd.read_csv(OUT / f"{mod}_17/{b}_words.csv")
            books[mod, b] = (w.family.to_numpy(), w.family_score.to_numpy())
    ledger = []
    arrays = defaultdict(list)
    spatial = []
    topo = []
    response = []
    with torch.inference_mode():
        for n, (m, pair) in enumerate(panel()):
            status = "complete"
            fs = {}
            sm = []
            tp = []
            rr = []
            for mod in ("eeg", "hb"):
                row, positions = pair[mod]
                r = rows[row]
                xy = coords(m["dataset"], m["subject"], m["record"], mod, tuple(r["positions"]))
                pos = four_landmarks(xy, positions)
                if len(pos) < 4:
                    status = mod + "_geometry"
                    break
                loc = xy[pos]
                raw = base_tokenizer_batch(
                    rows, [(row, m["start"], p) for p in pos], scales, array_cache=cache
                )
                if not raw["mask"].all():
                    status = mod + "_incomplete"
                    break
                x = raw["x"]
                net = networks[mod]

                def stream(x):
                    res = net(torch.from_numpy(x.astype("float32")).to("cuda:0"))
                    return {b: res["ids"][b].cpu().numpy() for b in "SO"}

                ids = stream(x)
                observed = wave_attribute(x, mod)
                fs[mod + "_y"] = observed
                for b in "SO":
                    family, score = books[mod, b]
                    f = family[ids[b]]
                    one = np.eye(4)[f].reshape(4, 3, 5, 4).mean(2)
                    fs[mod + b] = one
                    if b == "S":
                        v = score[ids[b]]

                        # Remove the four-site mean independently at every patch before comparing time courses.
                        def corrvec(signal):
                            z = signal - signal.mean(0)
                            return np.corrcoef(z)[np.triu_indices(4, 1)]

                        dx, dy = corrvec(observed), corrvec(v)
                        shifted = corrvec(np.stack([np.roll(t, (3 * i) % 15) for i, t in enumerate(v)]))
                        dist = pdist(loc)

                        def rs(a, b):
                            a, b, d = [pd.Series(z).rank().to_numpy() for z in (a, b, dist)]
                            design = np.stack([np.ones(len(d)), d], 1)
                            q = np.linalg.qr(design)[0]
                            a = a - q @ (q.T @ a)
                            b = b - q @ (q.T @ b)
                            return np.corrcoef(a, b)[0, 1] if min(a.std(), b.std()) > 1e-7 else np.nan

                        sm.append(dict(**m, mod=mod, rho=rs(dx, dy), shifted=rs(dx, shifted)))
                        for i, p in enumerate(pos):
                            tp.append(
                                dict(
                                    **m,
                                    mod=mod,
                                    position=r["positions"][p],
                                    x=loc[i, 0],
                                    y=loc[i, 1],
                                    Q4=(f[i] == 3).mean(),
                                    observed=observed[i].mean(),
                                )
                            )
                if m["condition"] in ("MA", "BL"):
                    bad = x.copy()
                    bad += rng.normal(size=x.shape).astype(np.float32) * 0.25 * x.std(2, keepdims=True)
                    noi = stream(bad)
                    for b in "SO":
                        fs[mod + b + "_noise"] = (
                            np.eye(4)[books[mod, b][0][noi[b]]].reshape(4, 3, 5, 4).mean(2)
                        )
                    if mod == "hb":
                        y = x.reshape(4, 300, 2)
                        pred = net(torch.from_numpy(x).to("cuda:0"))["sum"].cpu().numpy().reshape(y.shape)
                        for color in range(2):
                            for kind, arr in [("observed", y), ("decoded", pred)]:
                                vec = arr[:2, :, color].mean(0)
                                z = vec - vec[:20].mean()
                                rr.append(
                                    dict(
                                        **m,
                                        color=["HbO", "HbR"][color],
                                        kind=kind,
                                        peak_s=z.argmax() / 10,
                                        peak=z.max(),
                                        early_slope=(z[100:120].mean() - z[:20].mean()) / 10,
                                        recovery=z[-20:].mean() - z.max(),
                                    )
                                )
                else:
                    for b in "SO":
                        fs[mod + b + "_noise"] = fs[mod + b]
            ledger.append(dict(**m, status=status))
            if status == "complete":
                for k, v in fs.items():
                    arrays[k].append(v)
                spatial.extend(sm)
                topo.extend(tp)
                response.extend(rr)
            if n % 200 == 0:
                print("paired crops", n, flush=True)
    pd.DataFrame(ledger).to_csv(OUT / "paired_panel.csv", index=False)
    pd.DataFrame(spatial).to_csv(OUT / "spatial.csv", index=False)
    pd.DataFrame(topo).to_csv(OUT / "positions.csv", index=False)
    pd.DataFrame(response).to_csv(OUT / "hb_response.csv", index=False)
    a = {k: np.array(v) for k, v in arrays.items()}
    pop = pd.DataFrame(ledger).query("status=='complete'").reset_index(drop=True)
    np.savez(OUT / "paired_features.npz", **a)
    comparisons(pop, a)


def comparisons(pop, a):
    from sklearn.metrics import brier_score_loss

    outcomes = []
    ranking = []
    cov = []
    stability = []
    t = pop.split.eq("train").to_numpy()
    for mod in ("eeg", "hb"):
        other = "hb" if mod == "eeg" else "eeg"
        # Regional analogues, shared global trend, and region-shuffled control on matched windows.
        orig = a[mod + "_y"]
        rev = a[other + "_y"]
        for i, r in pop[~t].iterrows():
            for off in (0, 1, 2, 3):
                v = orig[i, :, : 15 - off]
                h = rev[i, :, off:]
                for label, src in [("native", h), ("region_swap", h[[2, 3, 0, 1]])]:
                    ve = v - v.mean(0)
                    he = src - src.mean(0)
                    cov.append(
                        dict(
                            dataset=r.dataset,
                            subject=r.subject,
                            mod=mod,
                            lag_s=2 * off,
                            alignment=label,
                            r=np.corrcoef(ve.ravel(), he.ravel())[0, 1],
                        )
                    )
        features = np.column_stack([a[other + b].mean(1).reshape(len(t), -1) for b in "SO"])
        own = orig.mean(1)[:, :5].mean(1)[:, None]
        # Same-window offline incremental explanation of the late 10 s, plus a fixed cross-dataset readout.
        target = orig.mean(1)[:, -5:].mean(1)
        for ds in sorted(pop.dataset.unique()):
            v = (~t) & pop.dataset.eq(ds).to_numpy()
            for name, train in [
                ("within", t & pop.dataset.eq(ds).to_numpy()),
                ("leave_dataset", t & ~pop.dataset.eq(ds).to_numpy()),
            ]:
                if train.sum() < 10 or not v.any():
                    continue
                for method, X in [
                    ("own_early", own),
                    ("other", features),
                    ("both", np.column_stack((own, features))),
                ]:
                    pred = fit_ridge(X[train], target[train])(X[v])
                    err = (pred - target[v]) ** 2
                    for p in pop[v].subject.unique():
                        ix = pop[v].subject.eq(p).to_numpy()
                        stability.append(
                            dict(
                                dataset=ds,
                                subject=p,
                                target=mod,
                                fit=name,
                                method=method,
                                n=ix.sum(),
                                mse=err[ix].mean(),
                            )
                        )
    pd.DataFrame(cov).to_csv(OUT / "regional_lags.csv", index=False)
    pd.DataFrame(stability).to_csv(OUT / "direction_transfer.csv", index=False)
    idx = np.flatnonzero(pop.condition.isin(["MA", "BL"]))
    meta = pop.iloc[idx].reset_index(drop=True)
    tr = meta.split.eq("train").to_numpy()
    y = meta.condition.eq("MA").to_numpy().astype(int)
    vec = {}
    for mod in ("eeg", "hb"):
        for b in "SO":
            v = a[mod + b][idx]
            vec[mod + b] = v.reshape(len(idx), -1)
        for mode in ("clean", "noise", "drop"):
            q = [a[mod + b + ("_noise" if mode == "noise" else "")][idx].copy() for b in "SO"]
            if mode == "drop":
                for v in q:
                    v[:, 2:] = v[tr, 2:].mean(0)
            vec[mod + mode] = np.column_stack([v.reshape(len(idx), -1) for v in q])

    def fit(x):
        ss = StandardScaler().fit(x[tr])
        clf = LogisticRegression(C=0.1, max_iter=1000).fit(ss.transform(x[tr]), y[tr])
        return lambda v: clf.predict_proba(ss.transform(v))[:, 1]

    E, H = vec["eegclean"], vec["hbclean"]
    e = fit(E)
    h = fit(H)

    def design(E, H, form):
        if form == "linear":
            return np.column_stack((E, H))
        # Fixed random projections matched across arms; extra slots hold self-products or cross-products.
        rng = np.random.default_rng(1126)
        ep = E @ rng.normal(size=(E.shape[1], 24)) / np.sqrt(E.shape[1])
        hp = H @ rng.normal(size=(H.shape[1], 24)) / np.sqrt(H.shape[1])
        return np.column_stack(
            (
                E,
                H,
                ep**2,
                hp**2,
                ep * hp if form == "interaction" else np.column_stack((ep[:, :12] ** 3, hp[:, 12:] ** 3)),
            )
        )

    heads = {k: fit(design(E, H, k)) for k in ("linear", "nonlinear_main", "interaction")}

    def record(mode, model, prob):
        for sub in meta[~tr].subject.unique():
            sel = (~tr) & meta.subject.eq(sub).to_numpy()
            outcomes.append(
                dict(
                    mode=mode,
                    model=model,
                    subject=sub,
                    n=sel.sum(),
                    n_ma=y[sel].sum(),
                    ba=(
                        balanced_accuracy_score(y[sel], prob[sel] >= 0.5)
                        if np.unique(y[sel]).size == 2
                        else np.nan
                    ),
                    brier=brier_score_loss(y[sel], prob[sel]),
                )
            )

    for mode, ec, hc in [
        ("clean", E, H),
        ("eeg_missing", np.tile(E[tr].mean(0), (len(E), 1)), H),
        ("hb_missing", E, np.tile(H[tr].mean(0), (len(H), 1))),
        ("eeg_noise", vec["eegnoise"], H),
        ("hb_noise", E, vec["hbnoise"]),
        ("eeg_drop", vec["eegdrop"], H),
        ("hb_drop", E, vec["hbdrop"]),
    ]:
        for k, p in {
            "eeg": e(ec),
            "hb": h(hc),
            "late": (e(ec) + h(hc)) / 2,
            **{k: f(design(ec, hc, k)) for k, f in heads.items()},
        }.items():
            record(mode, k, p)
    for key in ("eegS", "eegO", "hbS", "hbO"):
        record("clean", key, fit(vec[key])(vec[key]))
    # Fit cross-modal matching only on training people; candidates share condition, person and record.
    for src, dst in ((E, H), (H, E)):
        est = fit_ridge(src[tr], dst[tr])(src[~tr])
        truth = dst[~tr]
        m = meta[~tr].reset_index(drop=True)
        for i, r in m.iterrows():
            neg = (
                (m.subject == r.subject)
                & (m.record == r.record)
                & (m.condition == r.condition)
                & (abs(m.eeg_start - r.eeg_start) >= 60)
                & (abs(m.hb_start - r.hb_start) >= 60)
            )
            ids = np.flatnonzero(neg)
            if len(ids) < 2:
                continue
            other = np.mean((est[i] - truth[ids]) ** 2, axis=1)
            pos = np.mean((est[i] - truth[i]) ** 2)
            rank = 1 + (other < pos).sum() + 0.5 * (other == pos).sum()
            N = len(ids) + 1
            ranking.append(
                dict(
                    source="eeg" if src is E else "hb",
                    subject=r.subject,
                    record=r.record,
                    event=r.event,
                    candidates=N,
                    rank=rank,
                    R1=float(rank == 1),
                    MRR=1 / rank,
                    chance=1 / N,
                )
            )
    pd.DataFrame(outcomes).to_csv(OUT / "fusion.csv", index=False)
    pd.DataFrame(ranking).to_csv(OUT / "paired_retrieval.csv", index=False)
    tasks = pop[pop.condition.str.endswith("back session")].copy()
    for mod in ("eeg", "hb"):
        tasks[mod + "_Q4"] = a[mod + "S"][tasks.index, :, :, 3].mean((1, 2))
        tasks[mod + "_obs"] = a[mod + "_y"][tasks.index].mean((1, 2))
    tasks.to_csv(OUT / "nback_dose.csv", index=False)
