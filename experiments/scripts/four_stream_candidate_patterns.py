"""Fixed double-Q1 test, spatial attribute ladder and frozen WG residual control."""

from collections import OrderedDict
import os, subprocess
import numpy as np
import pandas as pd
import torch

import audit_aligned_tokenizer as audit
import analyze_four_stream_transfer as parent
import run_four_stream_evidence as stage
from src.data.dual_component_tokenizer import base_tokenizer_batch
from src.analysis.single_vq_relations import wave_attribute, interval
from src.analysis.joint_window_surrogates import joint_counts, conditional, rotations
from src.inference.conditional_slot_training import setup, forward, new_model
from src.analysis.conditional_evidence import cell_scores
from src.utils.io import save_npz_atomic

FOLDER = stage.OUT / "patterns"


@torch.inference_mode()
def discovery():
    rows, scales = parent.atlas_inputs()
    raw = (
        pd.read_csv(audit.OUT / "positions.csv", dtype={"subject": str})
        .query("split=='validation' and mod=='eeg'")
        .reset_index(drop=True)
    )
    index = {(r["parent"], r["modality"]): i for i, r in enumerate(rows) if r["split"] == "validation"}
    inv = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=pci.bus_id,uuid", "--format=csv,noheader"], text=True
    )
    card = [
        s.split(",")[1].strip() for s in inv.splitlines() if s.split(",")[0].strip() == "00000000:4B:00.0"
    ]
    if len(card) != 1 or torch.cuda.is_initialized():
        raise ValueError("cold permitted GPU required")
    os.environ["CUDA_VISIBLE_DEVICES"] = card[0]
    torch.set_float32_matmul_precision("highest")
    net = audit.network("eeg", 8192, 17, "cuda:0")
    maps = [parent.family("eeg", b) for b in "SO"]
    cache = OrderedDict()
    seq = []
    points = []
    for i in range(0, len(raw), 4):
        chunk = raw.iloc[i : i + 4]
        ex = []
        for r in chunk.itertuples():
            k = index[r.parent, "eeg"]
            src = rows[k]
            if (src["dataset"], src["subject"], src["record"]) != (r.dataset, r.subject, r.record):
                raise ValueError("identity mismatch")
            ex.append((k, r.start, src["positions"].index(r.position)))
        v = base_tokenizer_batch(rows, ex, scales, array_cache=cache)
        if not v["mask"].all():
            raise ValueError("incomplete discovery panel")
        x = v["x"]
        result = net(torch.from_numpy(x.astype("float32")).to("cuda:0"))
        s, o = [m[result["ids"][b].cpu().numpy()] for m, b in zip(maps, "SO")]
        q4 = s == 3
        np.testing.assert_allclose(q4.mean(1), chunk.Q4)
        orig = wave_attribute(x, "eeg")
        teacher = wave_attribute(v["s"], "eeg")
        decoded = wave_attribute(result["S"].cpu().numpy(), "eeg")
        np.testing.assert_allclose(orig.mean(1), chunk.observed, atol=1e-6)
        for j, r in enumerate(chunk.itertuples()):
            info = dict(
                dataset=r.dataset,
                subject=r.subject,
                record=r.record,
                condition=r.condition if pd.notna(r.condition) else "unlabelled",
                position=r.position,
                start_time=r.eeg_start,
            )
            seq.append(dict(info, s=s[j], o=o[j]))
            f = np.abs(np.fft.rfft(x[j], axis=-1)) ** 2
            line = f[:, 98:103].sum(1) / np.maximum(f[:, 1:].sum(1), 1e-20)
            amp = np.sqrt((x[j] ** 2).mean(1)) * scales["positions"]["eeg"][r.dataset][r.position]["scale"]
            points.append(
                pd.DataFrame(
                    dict(
                        info,
                        back=r.y < 0,
                        amp=amp,
                        line=line,
                        alpha=orig[j],
                        teacher_alpha=teacher[j],
                        decoded_alpha=decoded[j],
                        q4=q4[j].astype(float),
                    )
                )
            )
    pts = pd.concat(points, ignore_index=True)
    effects = []
    values = ["alpha", "teacher_alpha", "decoded_alpha", "q4"]
    for (ds, sub), g in pts.groupby(["dataset", "subject"]):
        g = g.assign(
            a=pd.qcut(g.amp, 3, labels=False, duplicates="drop"),
            q=pd.qcut(g.line, 2, labels=False, duplicates="drop"),
        )
        keys = ["record", "condition", "a", "q"]
        n = g.groupby(keys + ["back"]).size().unstack().dropna()
        matched = g.merge(n.reset_index()[keys], on=keys, validate="many_to_one")

        def raw_diff(z):
            return z.groupby(["back", "position"])[values].mean().groupby("back").mean().diff().iloc[-1]

        means = matched.groupby(keys + ["back"])[values].mean().unstack().loc[n.index]
        adj = {k: np.average(means[k][True] - means[k][False], weights=n.min(1)) for k in values}
        for label, d, total in [
            ("all", raw_diff(g), len(g)),
            ("matched_unadjusted", raw_diff(matched), len(matched)),
            ("matched_adjusted", adj, len(matched)),
        ]:
            effects.append(dict(dataset=ds, subject=sub, analysis=label, patches=total, strata=len(n), **d))
    frame = pd.DataFrame(effects)
    frame.to_csv(FOLDER / "spatial_people.csv", index=False)
    old = (
        pd.read_csv(stage.prev.OUT / "transfer/spatial_people.csv")
        .set_index(["dataset", "subject"])
        .sort_index()
    )
    for label, pre in [("all", ""), ("matched_adjusted", "adjusted_")]:
        g = frame[frame.analysis == label].set_index(["dataset", "subject"]).sort_index()
        for k in ["alpha", "q4"]:
            np.testing.assert_allclose(g[k], old[pre + k], atol=1e-6)
    return pd.DataFrame(seq)


def score_null(panel, frame):
    draws = stage.CFG["semantics"]["q1_repeats"]
    rng = np.random.default_rng(stage.CFG["semantics"]["permutation_seed"])
    rows = []
    trans = []
    ledger = []
    dist = {}
    pmf_rows = []
    for (ds, sub), g in frame.groupby(["dataset", "subject"]):
        observed = np.zeros(272, int)
        sims = np.zeros((draws, 272), np.int32)
        permutations = {}
        joint_choices = {}
        exact = np.zeros(272)
        for (rec, cond, site), b in g.groupby(["record", "condition", "position"]):
            b = b.sort_values("start_time")
            t = b.start_time.to_numpy()
            shifts = rotations(t)
            ledger.append(
                dict(
                    panel=panel,
                    dataset=ds,
                    subject=sub,
                    record=rec,
                    condition=cond,
                    site=site,
                    windows=len(t),
                    legal=len(shifts),
                )
            )
            if not shifts:
                continue
            s, o = np.stack(b.s), np.stack(b.o)
            obs = joint_counts(s, o)
            choices = np.stack([joint_counts(s, np.roll(o, k, axis=0)) for k in shifts])
            observed += obs
            exact += choices.mean(0)
            # Share each draw across sites with the same physical window list.
            key = rec, cond, tuple(t)
            if key not in permutations:
                permutations[key] = rng.integers(len(shifts), size=draws)
            sims += choices[permutations[key]]
            joint_choices[key] = joint_choices.get(key, 0) + choices[:, 0]
        opportunities = int(observed[:16].sum())
        if not opportunities:
            rows.append(dict(panel=panel, dataset=ds, subject=sub, opportunities=0))
            continue
        if not np.all(sims[:, :16].sum(1) == opportunities):
            raise ValueError("opportunities changed")
        real = observed[0] / opportunities
        null = sims[:, 0] / opportunities
        mass = np.ones(1)
        for counts in joint_choices.values():
            mass = np.convolve(mass, np.bincount(counts) / len(counts))
        np.testing.assert_allclose(mass.sum(), 1)
        np.testing.assert_allclose(np.arange(len(mass)) @ mass, exact[0])
        cdf = np.cumsum(mass)
        for count in np.flatnonzero(mass):
            pmf_rows.append(
                dict(
                    dataset=ds, subject=sub, opportunities=opportunities, count=count, probability=mass[count]
                )
            )
        rows.append(
            dict(
                panel=panel,
                dataset=ds,
                subject=sub,
                opportunities=opportunities,
                observed=real,
                exact_null_p025=np.searchsorted(cdf, 0.025) / opportunities,
                exact_null_p975=np.searchsorted(cdf, 0.975) / opportunities,
                exact_upper_tail=mass[observed[0] :].sum(),
                null_mean=exact[0] / opportunities,
                excess=real - exact[0] / opportunities,
                null_p025=np.quantile(null, 0.025),
                null_p975=np.quantile(null, 0.975),
                upper_tail=(1 + sum(null >= real)) / (draws + 1),
            )
        )
        prob = conditional(observed)
        sur = conditional(sims)
        for a in range(16):
            for j in range(16):
                vals = sur[:, a, j]
                vals = vals[np.isfinite(vals)]
                trans.append(
                    dict(
                        panel=panel,
                        dataset=ds,
                        subject=sub,
                        origin=a,
                        destination=j,
                        outgoing=int(observed[16:].reshape(16, 16)[a].sum()),
                        observed=prob[a, j],
                        null_draws=len(vals),
                        null_mean=vals.mean() if len(vals) else np.nan,
                        null_p025=np.quantile(vals, 0.025) if len(vals) else np.nan,
                        null_p975=np.quantile(vals, 0.975) if len(vals) else np.nan,
                    )
                )
        dist[ds + "__" + sub] = sims
    pd.DataFrame(ledger).to_csv(FOLDER / f"{panel}_coverage.csv", index=False)
    pd.DataFrame(trans).to_csv(FOLDER / f"{panel}_conditional_transitions.csv.gz", index=False)
    save_npz_atomic(FOLDER / f"{panel}_null_counts.npz", **dist)
    pd.DataFrame(pmf_rows).to_csv(FOLDER / f"{panel}_q1_exact_null.csv.gz", index=False)
    return pd.DataFrame(rows)


@torch.inference_mode()
def run():
    FOLDER.mkdir(exist_ok=True)
    d = discovery()
    old = score_null("discovery", d)
    check = pd.read_csv(stage.prev.OUT / "transfer/joint_family_null.csv.gz").query(
        "modality=='eeg' and kind=='occupancy' and cell==0"
    )
    cols = ["dataset", "subject"]
    np.testing.assert_allclose(old.sort_values(cols).excess, check.sort_values(cols).excess, atol=1e-9)
    # Reuse the exact public WG panel and frozen coordinates. Keep its ledger in this run.
    meta, data = parent.wg_inputs(FOLDER)
    ledger = pd.read_csv(stage.prev.OUT / "transfer/wg_panel.csv")
    pd.testing.assert_frame_equal(meta[["id", "donor"]], ledger[["id", "donor"]])
    wg = []
    orig = set(zip(d.dataset, d.subject, d.record))
    maps = [parent.family("eeg", b) for b in "SO"]
    for i, r in enumerate(meta.itertuples()):
        if (r.dataset, r.subject, r.record) in orig:
            continue
        for j, pos in enumerate(parent.SITES["eeg"]):
            wg.append(
                dict(
                    dataset=r.dataset,
                    subject=r.subject,
                    record=r.record,
                    condition=r.condition,
                    position=pos,
                    start_time=r.eeg_start,
                    s=maps[0][data["eeg_ids"][i, 0, j]],
                    o=maps[1][data["eeg_ids"][i, 1, j]],
                )
            )
    new = score_null("new_record", pd.DataFrame(wg))
    new["new_person"] = ~new.subject.isin(d.loc[d.dataset == stage.prev.DS, "subject"])
    pd.concat([old, new], ignore_index=True).to_csv(FOLDER / "double_q1_people.csv", index=False)
    # Frozen transfer of the equal-budget B comparison on the full parent WG support.
    idx = np.arange(len(meta))
    results = []
    for mod, coord in [("eeg", "native"), ("hb", "cached")]:
        t = setup(data, meta, "B_" + mod)
        for seed in stage.CFG["B"]["seeds"]:
            state = torch.load(stage.OUT / "b" / f"{coord}_{mod}_{seed}" / "models.pt", weights_only=True)
            for arm, weights in state["models"].items():
                model = new_model("B_" + mod, arm, stage.prev.TR, seed, t["y"].shape[-1])
                model.load_state_dict(weights)
                if arm == "anchor":
                    base = forward(model, t, idx, arm) * state["sd"] + state["mu"]
                for swap in [False] if arm == "anchor" else [False, True]:
                    pred = base if arm == "anchor" else base + forward(model, t, idx, arm, swap) * state["sd"]
                    results.extend(
                        cell_scores(
                            meta,
                            t["y"].numpy(),
                            pred.numpy(),
                            state["sd"].numpy(),
                            mod,
                            arm,
                            seed,
                            coordinate="WG_cached",
                            input="mismatched" if swap else "aligned",
                        )
                    )
    folder = FOLDER / "wg_b"
    folder.mkdir(exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(folder / "cells.csv.gz", index=False)
    b = stage.b_summary(df, folder)
    return dict(
        new_records=new.to_dict("records"),
        wg_events=len(meta),
        wg_people=meta.subject.nunique(),
        transfer=b.to_dict("records"),
    )
