"""Whole-montage occupancy, signal injections and controlled cross-modal dependence."""

from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from scipy import signal
from sklearn.metrics import mutual_info_score
import audit_aligned_tokenizer as audit
from analyze_aligned_token_controls import save, words
from src.data.unified_physiology import ChannelGeometryIndex
from src.data.channel_geometry import project_head_topview
from src.analysis.observation_task_probe import DOWNSTREAM_ROOT
from src.data.dual_component_tokenizer import base_tokenizer_batch


def run(mod):
    out = audit.OUT
    rows, scales = audit.relations.maps()
    geo = ChannelGeometryIndex(DOWNSTREAM_ROOT)
    maps = {b: words(mod, b).family.to_numpy() for b in "SO"}
    a = audit.collect(mod, 17, "validation")
    win = np.repeat(np.arange(len(a["examples"])), a["valid"].sum(1))
    ex = a["examples"][win]
    frame = pd.DataFrame(
        {
            "row": ex[:, 0],
            "pos": ex[:, 2],
            "obs": a["y"][:, 0],
            "S_Q4": maps["S"][a["ids"][:, 0]] == 3,
            "O_Q4": maps["O"][a["ids"][:, 1]] == 3,
        }
    )
    tab = (
        frame.groupby(["row", "pos"])
        .agg(n=("obs", "size"), observed=("obs", "mean"), S_Q4=("S_Q4", "mean"), O_Q4=("O_Q4", "mean"))
        .reset_index()
    )
    cand = pd.DataFrame(a["examples"][:, [0, 2]], columns=["row", "pos"]).value_counts()
    coords = {}
    result = []
    for r in tab.itertuples():
        row = rows[r.row]
        names = row["positions"]
        key = (row["dataset"], row["subject"], row["record"], tuple(names))
        if key not in coords:
            coords[key] = geo.for_channels(
                record=SimpleNamespace(
                    dataset_id=key[0], canonical_subject_id=key[1], base_record_id=key[2]
                ),
                modality="eeg" if mod == "eeg" else "fnirs",
                channel_names=names,
            )
        g = coords[key][r.pos]
        x, y = project_head_topview(g) if g["position_available"] else (np.nan, np.nan)
        result.append(
            dict(
                dataset=key[0],
                subject=key[1],
                record=key[2],
                row=r.row,
                position=names[r.pos],
                x=x,
                y=y,
                n=r.n,
                possible=15 * cand[r.row, r.pos],
                observed=r.observed,
                S_Q4=r.S_Q4,
                O_Q4=r.O_Q4,
            )
        )
    save(result, mod + "_montage")
    # Positive, near-identity, and equal-energy technical perturbations on matched held-out centers.
    model = audit.network(mod, audit.PLAN["models"][mod]["K"], 17, "cuda:0")
    local = pd.read_csv(out / f"{mod}_local_panel.csv").query("role=='validation'")
    rng = np.random.default_rng(1126)
    cache = OrderedDict()
    res = []
    features = []
    for chunk in np.array_split(local.to_numpy(), int(np.ceil(len(local) / 32))):
        g = pd.DataFrame(chunk, columns=local.columns)
        ex = [(r.row, r.start, rows[r.row]["positions"].index(r.position)) for r in g.itertuples()]
        batch = base_tokenizer_batch(rows, ex, scales, array_cache=cache)
        x = batch["x"]
        w = x[:, 7].reshape(len(x), -1, 1 if mod == "eeg" else 2)
        rate = 200 if mod == "eeg" else 10
        t = np.arange(w.shape[1]) / rate
        with torch.inference_mode():
            orig = model(torch.as_tensor(x, device="cuda:0"))["ids"]
        noise = rng.normal(size=w.shape)
        shapes = {"white": noise, "step": np.broadcast_to(np.sign(t - 1)[None, :, None], w.shape)}
        if mod == "eeg":
            shapes["line50"] = np.broadcast_to(np.sin(2 * np.pi * 50 * t)[None, :, None], w.shape)
            shapes["muscle_band"] = signal.sosfiltfilt(
                signal.butter(3, [30, 80], fs=200, btype="bandpass", output="sos"), noise, axis=1
            )
            f = np.fft.rfftfreq(400, 1 / 200)
            flank = ((f >= 2) & (f <= 5)) | ((f >= 15) & (f <= 30))
            for comp in "XSO":
                target = batch[comp.lower()][:, 7]
                spec = (
                    abs(np.fft.rfft((target - target.mean(1, keepdims=True)) * np.hanning(400))) ** 2
                    * 2
                    / (400 * (np.hanning(400) ** 2).sum())
                )
                axis = np.stack([np.ones(flank.sum()), np.log10(f[flank])], 1)
                yy = np.log10(np.maximum(spec[:, flank], 1e-15))
                coef = np.linalg.lstsq(axis, yy.T, rcond=None)[0]
                rr = 1 - ((axis @ coef - yy.T) ** 2).sum(0) / np.maximum(
                    ((yy - yy.mean(1, keepdims=True)) ** 2).sum(1), 1e-15
                )
                for i, r in enumerate(g.itertuples()):
                    features.append(
                        dict(
                            dataset=r.dataset,
                            subject=r.subject,
                            position=r.position,
                            row=r.row,
                            start=r.start,
                            component=comp,
                            offset=coef[0, i],
                            slope=coef[1, i],
                            fit_R2=rr[i],
                        )
                    )
        for k, s in shapes.items():
            s = s - s.mean(1, keepdims=True)
            s = (
                s
                / np.maximum(np.sqrt((s**2).mean(1, keepdims=True)), 1e-8)
                * np.std(w, 1, keepdims=True)
                * 0.25
            )
            alt = x.copy()
            alt[:, 7] = (w + s).reshape(len(x), -1)
            with torch.inference_mode():
                curr = model(torch.as_tensor(alt, device="cuda:0"))["ids"]
            for b in "SO":
                u = orig[b][:, 7].cpu().numpy()
                v = curr[b][:, 7].cpu().numpy()
                for i, r in enumerate(g.itertuples()):
                    res.append(
                        dict(
                            dataset=r.dataset,
                            subject=r.subject,
                            position=r.position,
                            row=r.row,
                            start=r.start,
                            branch=b,
                            kind=k,
                            same=int(u[i] == v[i]),
                            same_family=int(maps[b][u[i]] == maps[b][v[i]]),
                        )
                    )
    save(res, mod + "_artifact_controls")
    if features:
        save(features, "spectral_fit")
    if mod != "eeg":
        return
    # Conditional MI of paired high-family occupancy on the registered event/condition/session panel.
    meta = pd.read_csv(out / "paired_panel.csv").query("status=='complete'").reset_index(drop=True)
    paired = np.load(out / "paired_features.npz")
    v = meta.split.eq("validation") & meta.condition.isin(["MA", "BL"])
    tr = meta.split.eq("train") & meta.condition.isin(["MA", "BL"])
    d = meta.loc[v].copy()
    for m in ("eeg", "hb"):
        occupancy = paired[m + "S"][:, :, :, 3].mean(1)
        threshold = np.median(occupancy[tr])
        d[m] = list((occupancy[v] > threshold).astype(int))
    output = []
    for (person, session, condition), grp in d.groupby(["subject", "record", "condition"]):
        arr = [np.stack(grp[m]) for m in ("eeg", "hb")]
        starts = grp.eeg_start.to_numpy()
        hb = grp.hb_start.to_numpy()
        for lag in [0, 1]:
            aa = []
            bb = []
            null = []
            for i in range(len(grp)):
                donors = np.flatnonzero((abs(starts - starts[i]) >= 60) & (abs(hb - hb[i]) >= 60))
                if not len(donors):
                    continue
                z = rng.choice(donors)
                aa.extend(arr[0][i, : 3 - lag])
                bb.extend(arr[1][i, lag:])
                null.extend(arr[1][z, lag:])
            if aa:
                output.append(
                    dict(
                        subject=person,
                        record=session,
                        condition=condition,
                        lag_s=lag * 10,
                        n=len(aa),
                        MI_bits=mutual_info_score(aa, bb) / np.log(2),
                        mismatch_bits=mutual_info_score(aa, null) / np.log(2),
                    )
                )
    save(output, "conditional_MI")
