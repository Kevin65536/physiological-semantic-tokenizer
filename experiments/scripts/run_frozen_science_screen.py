"""Run the first frozen scientific-endpoint screen on the registered n-back panel."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import resource
import time

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression

import audit_aligned_tokenizer as audit
import run_four_stream_conditional as parent
import run_single_vq_spectral_location as spectral
from run_dual_component_tokenizer import _base_training_model, identity, owner_lock
from src.analysis.conditional_evidence import cell_scores, effect_tables
from src.analysis.four_stream_block_ridge import band_histories
from src.analysis.frozen_science_screen import (
    anchor,
    assigned,
    centered,
    count_kernel,
    family_map,
    gram,
    residual,
    shuffle_trails,
    slot_vectors,
)
from src.analysis.four_stream_bridge import select_ridge
from src.analysis.single_vq_relations import interval
from src.inference.conditional_slot_training import fit, forward
from src.utils.io import save_npz_atomic

ROOT = parent.ROOT
CONFIG = parent.CONTRACT.with_name("frozen_science_screen_v1.yaml")
CFG = yaml.safe_load(CONFIG.read_text())
OUT = ROOT / CFG["run"]
BASE = ROOT / CFG["parent"]
GRID = CFG["ridge_grid"]
SEEDS = CFG["seeds"]


def check_contract():
    if (
        CFG["scope"]
        != dict(
            coordinate="strict_common",
            train_people=15,
            development_people=4,
            train_events=309,
            development_events=91,
            tokenizer="frozen",
            test="forbidden",
            protected="forbidden",
        )
        or CFG["shape"]
        != dict(
            sites=4,
            slots=15,
            eeg_samples_per_slot=400,
            hb_samples_per_slot=20,
            chromophores=2,
            horizons_seconds=[2, 4, 6],
        )
        or BASE != parent.OUT
        or SEEDS != parent.TR["seeds"]
        or CFG["E0"] != ["raw_bands", "input_bands", "pre_vq", "embedding", "decoded_bands", "family64"]
        or CFG["E1"]["family_sizes"] != [64, 32]
        or CFG["E7"]["candidates"] != ["A", "B", "C"]
    ):
        raise ValueError("unexpected screen contract")


def paired_bands(x):
    return np.repeat(band_histories(x, CFG["bands_hz"])[:, :, None], 2, axis=2)


@torch.inference_mode()
def encode(net, wave, *, details=False):
    ids, pre, decoded = [], [], []
    # Replay the same 32 flattened site histories per batch as the parent.
    for start in range(0, len(wave), 8):
        x = np.ascontiguousarray(wave[start : start + 8].reshape(-1, 15, 400))
        out = net(torch.from_numpy(x))
        ids.append(torch.stack([out["ids"][b] for b in "SO"], 1).reshape(-1, 4, 2, 15).numpy())
        if details:
            replay = net.decode_ids(out["ids"])["sum"]
            torch.testing.assert_close(replay, out["sum"], atol=0, rtol=0)
            pre.append(
                torch.stack([out["latent"][b] for b in "SO"], 1)
                .reshape(-1, 4, 2, 15, out["latent"]["S"].shape[-1])
                .numpy()
            )
            decoded.append(replay.reshape(-1, 4, 15, 400).numpy())
    return np.concatenate(ids), (np.concatenate(pre), np.concatenate(decoded)) if details else None


def prepare():
    path, m, data = parent.load_parent("strict_common")
    if parent.read(BASE / "summary.json")["status"] != "completed":
        raise ValueError("parent incomplete")
    waves = {}
    for sub in m.subject.unique():
        folder = path / "records"
        keys = [r["id"] for r in parent.read(folder / (sub + ".json"))["ledger"] if r["status"] == "complete"]
        with np.load(folder / (sub + ".npz")) as z:
            if (z["raw_stop"] > 4 + 1e-8).any() or (z["hb_target_start"] < z["raw_stop"][:, 1, None]).any():
                raise ValueError("invalid history support")
            waves.update(zip(keys, z["eeg_wave"], strict=True))
    raw = np.stack([waves[key] for key in m.id])
    sites = parent.read(path / "input_coordinates.json")["sites"]["eeg"]
    scales = parent.read(BASE.parents[1] / "dual_component_tokenizer/20261122_unified_base_g0_v1/scales.json")
    if scales["fitting_split"] != "train" or sites != ["F1", "F2", "P3", "P4"]:
        raise ValueError("input-scale identity")
    scale = np.array([scales["positions"]["eeg"][parent.DS][s]["scale"] for s in sites])
    wave = (raw / scale[None, :, None, None]).astype("float32")
    net = audit.network("eeg", 8192, 17)
    ids, (pre, decoded) = encode(net, wave, details=True)
    np.testing.assert_array_equal(ids, data["eeg_ids"].transpose(0, 2, 1, 3))
    embedding = np.stack([data["eeg" + b + "_book"][ids[:, :, j]] for j, b in enumerate("SO")], axis=2)
    tr = np.flatnonzero(m.split == "train")
    bands = paired_bands(raw)
    arrays = dict(
        wave=wave,
        ids=ids.astype("uint16"),
        raw_bands=bands,
        input_bands=paired_bands(wave),
        pre_vq=pre,
        embedding=embedding,
        decoded_bands=paired_bands(decoded),
        hb=data["hb_history"],
        target=data["hb_target"],
        xy=data["xy"],
    )
    for k in CFG["E1"]["family_sizes"]:
        true, random = family_map(ids, bands[:, :, 0, :, 2], tr, k)
        arrays[f"map{k}"] = true
        arrays[f"random{k}"] = random
    cache = OUT / "inputs"
    cache.mkdir(exist_ok=True)
    for key, value in arrays.items():
        np.save(cache / (key + ".npy"), value, allow_pickle=False)
    m.to_csv(OUT / "panel.csv", index=False)
    return dict(
        events=len(m),
        raw_sites=sites,
        source=str(path.relative_to(ROOT)),
        parent=str(BASE.relative_to(ROOT)),
        tokenizer=str(audit.PARENT.relative_to(ROOT)),
        baseline_ids_replayed=True,
        bytes=sum(v.nbytes for v in arrays.values()),
    )


def array(name):
    return np.load(OUT / "inputs" / (name + ".npy"), mmap_mode="r")


def panel():
    m = pd.read_csv(OUT / "panel.csv")
    return m, np.flatnonzero(m.split == "train"), np.flatnonzero(m.split == "validation")


def hb_slots(history, train):
    return slot_vectors(history.reshape(-1, 4, 15, 20, 2).transpose(0, 1, 4, 2, 3), train)


def source(name, train):
    if name.startswith("family"):
        size = int(name.removeprefix("family"))
        x = assigned(array("ids"), array(f"map{size}"))
        x = np.eye(size + 1, dtype=np.float32)[x]
    else:
        x = array(name)
    return slot_vectors(x, train)


def make_anchor():
    m, tr, ev = panel()
    h, y = array("hb"), array("target")
    oof, base, alpha = anchor(h, y, m.subject.to_numpy(), tr, ev, GRID)
    ref = h - h[:, :, :50].mean(2, keepdims=True)
    centered_pred, centered_alpha, _ = select_ridge(ref, y, m.subject.to_numpy(), tr, ev, GRID)
    save_npz_atomic(OUT / "anchor.npz", oof=oof, base=base)
    rows = []
    for arm, p in [("full", base), ("minus_reference", centered_pred)]:
        rows.extend(cell_scores(m.iloc[ev], y[ev], p, y[tr].std(0), "hb", arm, 0))
    pd.DataFrame(rows).to_csv(OUT / "hb_reference.csv", index=False)
    return dict(ridge=alpha, referenced_ridge=centered_alpha)


def score_sources(name, sk, x=None):
    m, tr, ev = panel()
    h = hb_slots(array("hb"), tr)
    hk = gram(h)
    donor = m.donor.to_numpy()
    y = array("target").astype(float)
    sd = y[tr].std(0)
    with np.load(OUT / "anchor.npz") as a:
        base = a["base"].reshape(y[ev].shape)
        errors = y[tr].reshape(len(tr), -1) - a["oof"]
    rows = []

    def score(p, reader, arm, seed, swap=False):
        rows.extend(
            cell_scores(
                m.iloc[ev],
                y[ev],
                p,
                sd,
                "hb",
                arm,
                seed,
                representation=name,
                reader=reader,
                input="mismatched" if swap else "aligned",
            )
        )

    score(base, "linear", "anchor", 0)
    square = gram(h * h)
    for arm in ("capacity", "true", "wrong"):
        for swap in [False] if arm == "capacity" else [False, True]:
            delta = residual(
                hk,
                square if arm == "capacity" else sk,
                errors,
                tr,
                ev,
                donor,
                arm == "wrong",
                swap,
                CFG["residual_ridge"],
            )
            score(base + delta.reshape(base.shape), "linear", arm, 0, swap)
    if x is not None:
        t = dict(
            kind="A",
            people=m.subject.to_numpy(),
            donor=donor,
            y=torch.tensor(y, dtype=torch.float32),
            xy=torch.tensor(array("xy"), dtype=torch.float32),
            v=dict(hb=torch.from_numpy(h), eeg=torch.from_numpy(x)),
            curve=[],
        )
        target = torch.tensor(errors.reshape(y[tr].shape) / sd, dtype=torch.float32)
        for seed in SEEDS:
            score(base, "slot", "anchor", seed)
            for arm in ("capacity", "true", "wrong"):
                net, _ = fit(t, arm, seed, tr, target, parent.TR["residual_steps"], parent.TR)
                for swap in [False] if arm == "capacity" else [False, True]:
                    with torch.no_grad():
                        delta = forward(net, t, ev, arm, swap).numpy() * sd
                    score(base + delta, "slot", arm, seed, swap)
    return pd.DataFrame(rows)


def task_scores(name, hk, sk):
    m, tr, ev = panel()
    y = m.condition.to_numpy()
    ks = [centered(k, tr, tr) for k in (hk, sk)]
    es = [centered(k, tr, ev) for k in (hk, sk)]
    rows = []
    for label, which in [("H", [0]), ("E", [1]), ("HE", [0, 1])]:
        # Factor the training kernel to fit logistic regression in the original linear feature metric.
        k = sum(ks[i] for i in which)
        vals, u = np.linalg.eigh(k)
        ok = vals > 1e-7
        a = u[:, ok] * np.sqrt(vals[ok])
        b = sum(es[i] for i in which) @ (u[:, ok] / np.sqrt(vals[ok]))
        model = LogisticRegression(C=1.0, max_iter=500).fit(a, y[tr])
        loss = ((model.predict_proba(b) - (y[ev, None] == model.classes_)) ** 2).sum(1)
        for person, idx in m.iloc[ev].reset_index(drop=True).groupby("subject").indices.items():
            rows.append(dict(representation=name, arm=label, subject=person, brier=loss[idx].mean()))
    return rows


def e0(name, folder):
    _, tr, _ = panel()
    x = source(name, tr)
    sk = gram(x)
    score_sources(name, sk, x).to_csv(folder / "cells.csv.gz", index=False)
    pd.DataFrame(task_scores(name, gram(hb_slots(array("hb"), tr)), sk)).to_csv(
        folder / "task.csv", index=False
    )


def e1(name, folder):
    _, tr, _ = panel()
    level, order = name.split("-")
    ids = array("ids")
    if level != "exact":
        ids = assigned(ids, array(level))
    changed = 0.0
    if order == "trail":
        null = shuffle_trails(ids)
        changed = float((null != ids).any(-1).mean())
        ids = null
    kernel = count_kernel(ids, tr, order != "bag")
    score_sources(name, kernel).to_csv(folder / "cells.csv.gz", index=False)
    return dict(changed_sequence_fraction=changed)


def e4(slot, folder):
    m, _, ev = panel()
    donor = m.donor.to_numpy()[ev]
    wave = array("wave")[ev].copy()
    wave[:, :, slot] = array("wave")[donor, :, slot]
    ids, (_, decoded) = encode(audit.network("eeg", 8192, 17), wave, details=True)
    ref = array("ids")[ev]
    maps = array("map64")
    changes = dict(
        id_change=(ids != ref).mean((1, 2)),
        family_change=(assigned(ids, maps) != assigned(ref, maps)).mean((1, 2)),
        decoded_abs_log_alpha=np.abs(
            band_histories(decoded, CFG["bands_hz"])[..., 2] - array("decoded_bands")[ev, :, 0, :, 2]
        ).mean(1),
    )
    old = pd.read_csv(BASE / "transfer/real_interval_reencode.csv").query("slot==@slot")
    rows = []
    for who, idx in m.iloc[ev].reset_index(drop=True).groupby("subject").indices.items():
        off = changes["id_change"][idx][:, np.arange(15) != slot].mean()
        np.testing.assert_allclose(old.loc[old.subject == who, "other_slot_id_change"], off, atol=1e-12)
        for j in range(15):
            rows.append(
                dict(
                    subject=who,
                    source_slot=slot,
                    output_slot=j,
                    events=len(idx),
                    **{k: v[idx, j].mean() for k, v in changes.items()},
                )
            )
    pd.DataFrame(rows).to_csv(folder / "influence.csv", index=False)


def historical_model(arm):
    cfg = spectral.training_cfg(17)
    folder = spectral.folder(17, arm)
    if folder.relative_to(BASE.parents[1]).as_posix() != CFG["E7"]["parent"] + "/" + arm:
        raise ValueError("historical pool identity")
    state = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
    summary = parent.read(folder / "summary.json")
    sig = state["signature"]
    objective = (
        None
        if arm == "A"
        else dict(
            location=spectral.PLAN["arms"][arm],
            weight=spectral.PLAN["weight"],
            calibration=parent.read(spectral.OUT / "calibration.json"),
        )
    )
    if (
        summary["status"] != "completed"
        or summary["signature"] != sig
        or sig["config"] != identity(cfg)
        or state["step"] != 10000
        or sig["task"] != spectral.TASK
        or (objective and sig["spectral_objective"] != identity(objective))
    ):
        raise ValueError("historical checkpoint identity")
    model = _base_training_model(cfg, spectral.TASK, single_id_spectrum=arm == "C").eval()
    model.load_state_dict(state["model_state"])
    return model


def candidate(arm, folder, final=False):
    m, tr, dev = panel()
    names = np.unique(m.subject.to_numpy()[tr])
    fit_rows = tr[~np.isin(m.subject.to_numpy()[tr], names[-5:])]
    ev = dev if final else tr[np.isin(m.subject.to_numpy()[tr], names[-5:])]
    if final:
        fit_rows = tr
    used = np.r_[fit_rows, ev]
    model = historical_model(arm)
    wave = array("wave")[used]
    ids, (_, decoded) = encode(model, wave, details=True)
    vec = np.stack(
        [model.branches[b]["quantizer"].codebook.detach().numpy()[ids[:, :, j]] for j, b in enumerate("SO")],
        2,
    )
    n = len(fit_rows)
    a, b = np.arange(n), np.arange(n, len(used))
    m = m.iloc[used].reset_index(drop=True)
    h, y = array("hb")[used], array("target")[used]
    oof, base, _ = anchor(h, y, m.subject.to_numpy(), a, b, GRID)
    h = hb_slots(h, a)
    hk = gram(h)
    r = y[a].reshape(n, -1) - oof
    base = base.reshape(y[b].shape)
    cap = base + residual(hk, gram(h * h), r, a, b, np.arange(len(m))).reshape(base.shape)
    pred = base + residual(hk, gram(slot_vectors(vec, a)), r, a, b, np.arange(len(m))).reshape(base.shape)
    science = (
        (((cap - y[b]) / y[a].std(0)) ** 2 - ((pred - y[b]) / y[a].std(0)) ** 2).reshape(len(b), -1).mean(1)
    )
    # Held-person observed alpha readout from IDs using training-only empirical means.
    alpha = array("raw_bands")[used, :, 0, :, 2]
    readings = []
    for j in range(2):
        words = ids[a, :, j].ravel()
        count = np.bincount(words, minlength=8192)
        sums = np.bincount(words, weights=alpha[a].ravel(), minlength=8192)
        readings.append(((sums + 4 * alpha[a].mean()) / (count + 4))[ids[b, :, j]])
    alpha_error = ((np.mean(readings, axis=0) - alpha[b]) ** 2).mean((1, 2)) / alpha[a].var()
    nmse = ((decoded[b] - wave[b]) ** 2).mean((1, 2, 3)) / (wave[b] ** 2).mean((1, 2, 3))
    rows = []
    for who, idx in m.iloc[b].reset_index(drop=True).groupby("subject").indices.items():
        coverage = np.mean([len(np.unique(ids[b[idx], :, j])) / 8192 for j in range(2)])
        rows.append(
            dict(
                arm=arm,
                subject=who,
                x_nmse=nmse[idx].mean(),
                alpha_nmse=alpha_error[idx].mean(),
                coverage=coverage,
                delta=science[idx].mean(),
            )
        )
    pd.DataFrame(rows).to_csv(folder / "people.csv", index=False)


def job(task):
    stage, name = task
    folder = OUT / "cells" / stage / name
    with owner_lock(folder):
        done = folder / "done.json"
        if done.exists():
            return parent.read(done)
        start = time.monotonic()
        if stage == "E0":
            result = e0(name, folder)
        elif stage == "E1":
            result = e1(name, folder)
        elif stage == "E4":
            result = e4(int(name), folder)
        else:
            result = candidate(name, folder, stage == "E7_evaluate")
        item = dict(
            stage=stage,
            name=name,
            seconds=time.monotonic() - start,
            peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            result=result,
        )
        parent.save(done, item)
        return item


def tasks():
    return (
        [("E0", name) for name in CFG["E0"]]
        + [("E7_select", arm) for arm in CFG["E7"]["candidates"]]
        + [
            ("E1", f"{level}-{order}")
            for level in ["exact", "map64", "random64", "map32", "random32"]
            for order in ["bag", "ordered", "trail"]
        ]
        + [("E4", str(i)) for i in range(15)]
    )


def select_policies():
    table = pd.concat(
        [pd.read_csv(OUT / "cells/E7_select" / arm / "people.csv") for arm in CFG["E7"]["candidates"]]
    )
    g = table.groupby("arm").mean(numeric_only=True)
    r = g.sort_values(["x_nmse", "coverage", "alpha_nmse"], ascending=[True, False, True]).index[0]
    eligible = g[g.x_nmse <= 1.05 * g.x_nmse.min()]
    winner = eligible.sort_values(["delta", "alpha_nmse"], ascending=[False, True]).index[0]
    picked = dict(
        R=r,
        M=winner,
        eligible=eligible.index.tolist(),
        selection_subjects=sorted(table.subject.unique()),
        scores=g.reset_index().to_dict("records"),
    )
    parent.save(OUT / "e7_choices.json", picked)
    return picked


def summarize(work):
    for stage in ("E0", "E1"):
        frames = [pd.read_csv(OUT / "cells" / s / n / "cells.csv.gz") for s, n in work if s == stage]
        df = pd.concat(frames)
        by = ["representation", "reader", "input", "arm", "subject"]
        means = df.groupby(by, as_index=False).nmse.mean()
        means.to_csv(OUT / (stage + "_people.csv"), index=False)
        contrasts = effect_tables(
            means.query("input=='aligned'"),
            "nmse",
            [
                ("over_full_Hb", "anchor", "true"),
                ("over_Hb_capacity", "capacity", "true"),
                ("over_wrong", "wrong", "true"),
            ],
            by[:2],
        )
        contrasts.to_csv(OUT / (stage + "_contrasts.csv"), index=False)
        expected = {n for s, n in work if s == stage}
        if set(means.representation) != expected or not np.isfinite(means.nmse).all():
            raise ValueError("incomplete scientific endpoint")
        if (df.groupby(["representation", "reader", "input", "arm", "seed", "subject"]).size() != 24).any():
            raise ValueError("incomplete site/horizon/color support")
        if not (contrasts.people == 4).all():
            raise ValueError("incomplete development pairing")
        if stage == "E1":
            scores = (
                means.query("arm=='true' and input=='aligned'")
                .drop(columns="arm")
                .rename(columns={"representation": "arm"})
            )
            pairs = [
                ("family_vs_exact_" + order, "exact-" + order, "map64-" + order)
                for order in ("bag", "ordered")
            ]
            pairs.extend(
                (level + "_order", level + "-bag", level + "-ordered")
                for level in ("exact", "map64", "map32")
            )
            for size in (64, 32):
                level = f"map{size}"
                pairs.append((level + "_over_trail", level + "-trail", level + "-ordered"))
                for order in ("bag", "ordered"):
                    pairs.append(
                        (level + "_over_random_" + order, f"random{size}-" + order, level + "-" + order)
                    )
            effect_tables(scores, "nmse", pairs, ["reader"]).to_csv(
                OUT / "E1_paired_controls.csv", index=False
            )
    task = pd.concat([pd.read_csv(OUT / "cells/E0" / n / "task.csv") for n in CFG["E0"]])
    mat = task.pivot(index=["representation", "subject"], columns="arm", values="brier")
    mat.assign(EEG_given_Hb=mat.H - mat.HE, Hb_given_EEG=mat.E - mat.HE).to_csv(OUT / "task_people.csv")
    influence = pd.concat([pd.read_csv(OUT / "cells/E4" / str(i) / "influence.csv") for i in range(15)])
    influence.to_csv(OUT / "E4_influence.csv", index=False)
    ref = pd.read_csv(OUT / "hb_reference.csv").groupby(["subject", "arm"]).nmse.mean().unstack()
    ref.assign(cost_of_reference=ref.minus_reference - ref.full).to_csv(OUT / "hb_reference_people.csv")
    choices = parent.read(OUT / "e7_choices.json")
    e7 = pd.concat(
        [
            pd.read_csv(OUT / "cells/E7_evaluate" / arm / "people.csv").assign(policy=policy)
            for policy, arm in ((p, choices[p]) for p in ("R", "M"))
        ]
    )
    e7.to_csv(OUT / "E7_replay.csv", index=False)
    results = pd.read_csv(OUT / "E0_contrasts.csv")
    primary = results.query("contrast=='over_Hb_capacity'")
    off = (
        influence.query("source_slot!=output_slot").groupby("subject")[["id_change", "family_change"]].mean()
    )
    e7_scores = e7.pivot(index="subject", columns="policy", values="delta")

    def paired_range(values):
        mean, lo, hi = interval(values, np.repeat(parent.DS, len(values)))
        return dict(mean=mean, ci_low=lo, ci_high=hi, people=len(values), positive=int((values > 0).sum()))

    return dict(
        status="completed",
        experiment_id=CFG["experiment_id"],
        coordinate="strict_common",
        endpoint=CFG["primary"],
        train_people=15,
        development_people=4,
        development_events=91,
        E0=primary.to_dict("records"),
        E4_mean_offslot=off.mean().to_dict(),
        E7=dict(R=choices["R"], M=choices["M"], M_minus_R=paired_range(e7_scores.M - e7_scores.R)),
        E1_controls="E1_paired_controls.csv",
        task="task_people.csv",
        Hb_reference_cost=paired_range(ref.minus_reference - ref.full),
        inference="Descriptive 95% percentile bootstrap of four paired people; seeds averaged first. No multiplicity adjustment.",
        interpretation="No EEG representation consistently beat both full Hb and pairing controls. Linear Hb square expansion underperformed the full Hb anchor. No localized quantization loss or higher-order family advantage established.",
        temporal_scope="Off-slot changes bound temporal attribution. A physiological delay and a causal effect cannot be identified from this replacement.",
        selection_scope="Public development replay of three seed-17 historical candidates. No independent confirmatory cohort.",
        versions=dict(numpy=np.__version__, pandas=pd.__version__, torch=str(torch.__version__)),
        scientific_scope="public exploratory screen",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["check", "prepare", "pilot", "run", "summarize"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    check_contract()
    if args.stage == "check":
        return
    torch.set_num_threads(1)
    with owner_lock(OUT):
        resolved = OUT / "resolved.yaml"
        if resolved.exists() and yaml.safe_load(resolved.read_text()) != CFG:
            raise ValueError("output contract drift")
        resolved.write_text(CONFIG.read_text())
        if args.stage == "prepare":
            if not (OUT / "prepared.json").exists():
                parent.save(OUT / "prepared.json", prepare())
            if not (OUT / "anchor.json").exists():
                parent.save(OUT / "anchor.json", make_anchor())
            return
        if not (OUT / "anchor.json").is_file():
            raise ValueError("prepare first")
        if args.stage == "summarize":
            for stage, name in tasks():
                if not (OUT / "cells" / stage / name / "done.json").is_file():
                    raise ValueError(f"incomplete cell {stage}/{name}")
            parent.save(OUT / "summary.json", summarize(tasks()))
            return
        if args.stage == "pilot":
            result = job(("E0", "raw_bands"))
            parent.save(OUT / "pilot.json", result)
            return
        work = tasks()
        if not 1 <= args.workers <= len(work):
            raise ValueError("invalid worker count")
        # Worker processes only read frozen shared inputs; each owns one output directory.
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
            futures = [pool.submit(job, t) for t in work]
            for num, future in enumerate(as_completed(futures), 1):
                item = future.result()
                parent.save(OUT / "progress.json", dict(complete=num, planned=len(work), latest=item))
            choices = select_policies()
            list(pool.map(job, [("E7_evaluate", k) for k in sorted({choices["R"], choices["M"]})]))
        parent.save(OUT / "summary.json", summarize(work))


if __name__ == "__main__":
    main()
