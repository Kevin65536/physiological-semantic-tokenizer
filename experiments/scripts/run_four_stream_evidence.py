"""Frozen v3 conditional kernels, residual controls and phenotype candidates."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import time, resource

import numpy as np
import pandas as pd
import torch
import yaml

import run_four_stream_conditional as prev
from run_dual_component_tokenizer import owner_lock
from src.analysis.four_stream_block_ridge import band_histories, select_blocks, block_predictions
from src.analysis.conditional_evidence import cell_scores, effect_tables
from src.inference.conditional_slot_training import setup, forward, fit, cross_anchor
from src.utils.io import save_npz_atomic

ROOT = prev.ROOT
CONFIG = prev.CONTRACT.with_name("four_stream_evidence_v3.yaml")
CFG = yaml.safe_load(CONFIG.read_text())
OUT = ROOT / CFG["run"]
S = CFG["A"]


def voltage(path, meta):
    parts = {}
    for sub in meta.subject.unique():
        folder = path / "records"
        ledger = [r["id"] for r in prev.read(folder / (sub + ".json"))["ledger"] if r["status"] == "complete"]
        with np.load(folder / (sub + ".npz")) as z:
            if (z["raw_stop"] > 4 + 1e-8).any():
                raise ValueError("lookahead")
            parts.update(zip(ledger, z["eeg_wave"], strict=True))
    return np.stack([parts[k] for k in meta.id])


def run_a():
    folder = OUT / "a"
    folder.mkdir(exist_ok=True)
    path, m, d = prev.load_parent("strict_common")
    if (m.eeg_stop - m.eeg_marker > 4 + 1e-8).any() or (m.hb_stop - m.hb_marker > 4 + 1e-8).any():
        raise ValueError("support")
    gain = prev.read(prev.OUT / "strict_common/coordinates.json")
    tr = np.flatnonzero(m.split == "train")
    ev = np.flatnonzero(m.split == "validation")
    h = d["hb_history"].astype(float)
    ref = h[:, :, :50].mean(2, keepdims=True)
    wave = ((h - ref) * gain["gain"] / np.array(gain["site_scales"])[None, :, None, None]).astype("float32")
    eeg = band_histories(voltage(path, m), S["bands_hz"])
    if set(m.condition) != set(S["conditions"]):
        raise ValueError("unknown task condition")
    cond = np.stack([m.condition == q for q in S["conditions"]], axis=1)
    y = d["hb_target"].astype(float)
    sd = y[tr].std(0)
    vectors = {
        mod: np.stack([d[mod + b + "_book"][d[mod + "_ids"][:, j]] for j, b in enumerate("SO")], axis=2)
        for mod in ["eeg", "hb"]
    }
    donor = m.donor.to_numpy()
    rows = []
    search = []
    preds = {}
    for label, hb, source in [
        ("continuous", wave, eeg),
        ("discrete", vectors["hb"], vectors["eeg"]),
        ("uncentered", h, eeg),
        ("reference_only", ref, eeg),
    ]:
        for arm, src, etas in [
            ("H", source, [0.0]),
            ("HE", source, S["eta"]),
            ("wrong", source[donor], S["eta"]),
        ]:
            if label in ("uncentered", "reference_only") and arm != "H":
                continue
            key, cv = select_blocks(
                hb, src, cond, y, m.subject.to_numpy(), tr, ev, etas, S["ridge"], S["folds"]
            )
            eta, alpha = key
            search.extend(
                dict(representation=label, arm=arm, eta=e, ridge=a, mean_cv=v, selected=(e, a) == key)
                for (e, a), v in cv.items()
            )
            for variant in ["aligned"] if arm == "H" else ["aligned", "mismatched"]:
                # Training source and test source can be permuted independently.
                e_in = src.copy()
                e_in[ev] = source[ev if variant == "aligned" else donor[ev]]
                _, pred = next(
                    block_predictions(hb, e_in, cond, y.reshape(len(y), -1), tr, ev, [eta], [alpha])
                )
                pred = pred.reshape(y[ev].shape)
                preds[f"{label}_{arm}_{variant}"] = pred
                rows.extend(
                    cell_scores(
                        m.iloc[ev], y[ev], pred, sd, "hb", arm, 0, representation=label, input=variant
                    )
                )
    scores = pd.DataFrame(rows)
    scores.to_csv(folder / "cells.csv.gz", index=False)
    person = scores.groupby(["representation", "input", "arm", "subject"], as_index=False).nmse.mean()
    person.to_csv(folder / "people.csv", index=False)
    table = effect_tables(
        person.query("input=='aligned'"),
        "nmse",
        [("over_H", "H", "HE"), ("over_wrong", "wrong", "HE")],
        ["representation"],
    )
    table.to_csv(folder / "contrasts.csv", index=False)
    pd.DataFrame(search).to_csv(folder / "selection.csv", index=False)
    save_npz_atomic(folder / "dev_predictions.npz", **preds)
    prev.bridge("native", output=folder / "expanded_native_bridge", ridge_grid=S["ridge"])
    return dict(contrasts=table.to_dict("records"))


def b_job(args):
    coord, mod, seed = args
    name = f"{coord}_{mod}_{seed}"
    folder = OUT / "b" / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "done.json").exists():
        return prev.read(folder / "done.json")
    _, m, d = prev.load_parent(coord)
    t = setup(d, m, "B_" + mod)
    tr = np.flatnonzero(m.split == "train")
    dev = np.flatnonzero(m.split == "validation")
    anchor, oof, base, mu, sd, sel = cross_anchor(t, seed, tr, dev, prev.TR, 3, "full")
    old = prev.OUT / "readers" / f"{coord}_B_{mod}_{seed}"
    target = (t["y"][tr] - oof[0]) / sd
    rows = []
    models = {}
    values = {}
    with np.load(old / "predictions.npz") as z:
        ref = z["anchor_aligned"]
        product = z["true_aligned"]
    np.testing.assert_allclose(base[0].numpy(), ref, atol=1e-6)
    for arm in ("anchor", "additive_residual", "true", "wrong"):
        if arm == "anchor":
            model = anchor
        else:
            model, _ = fit(t, arm, seed, tr, target, prev.TR["residual_steps"], prev.TR, name="same_oof")
        models[arm] = model
        for swap in [False] if arm == "anchor" else [False, True]:
            with torch.no_grad():
                val = base[0] if arm == "anchor" else base[0] + forward(model, t, dev, arm, swap) * sd
            variant = "mismatched" if swap else "aligned"
            if arm == "true" and not swap:
                np.testing.assert_allclose(val.numpy(), product, atol=2e-6)
            values[arm + "_" + variant] = val.numpy()
            rows.extend(
                cell_scores(
                    m.iloc[dev],
                    t["y"][dev].numpy(),
                    val.numpy(),
                    sd.numpy(),
                    mod,
                    arm,
                    seed,
                    coordinate=coord,
                    input=variant,
                )
            )
    counts = [sum(p.numel() for p in models[a].parameters()) for a in ("additive_residual", "true", "wrong")]
    if len(set(counts)) != 1:
        raise ValueError("residual budget mismatch")
    pd.DataFrame(rows).to_csv(folder / "cells.csv.gz", index=False)
    pd.DataFrame(t["curve"], columns=["fold", "arm", "step", "loss"]).to_csv(
        folder / "curves.csv.gz", index=False
    )
    torch.save(
        dict(models={a: v.state_dict() for a, v in models.items()}, sd=sd, mu=mu), folder / "models.pt"
    )
    save_npz_atomic(folder / "predictions.npz", **values)
    result = dict(
        name=name,
        coordinate=coord,
        modality=mod,
        seed=seed,
        residual_parameters=counts[0],
        anchor_steps=sel["chosen"],
    )
    prev.save(folder / "done.json", result)
    return result


def b_summary(df, folder):
    person = df.groupby(
        ["coordinate", "modality", "arm", "input", "subject", "seed"], as_index=False
    ).nmse.mean()
    person.to_csv(folder / "person_seed.csv", index=False)
    means = (
        person.query("input=='aligned'")
        .groupby(["coordinate", "modality", "arm", "subject"], as_index=False)
        .nmse.mean()
    )
    table = effect_tables(
        means,
        "nmse",
        [
            ("over_additive", "additive_residual", "true"),
            ("over_wrong", "wrong", "true"),
            ("over_anchor", "anchor", "true"),
        ],
        ["coordinate", "modality"],
    )
    table.to_csv(folder / "contrasts.csv", index=False)
    return table


def run_b():
    jobs = [
        (c, mod, s)
        for c in CFG["B"]["coordinates"]
        for mod in ("hb", "eeg")
        for s in CFG["B"]["seeds"]
        if mod == "hb" or c == "native"
    ]
    with ProcessPoolExecutor(4, mp_context=mp.get_context("fork")) as pool:
        results = list(pool.map(b_job, jobs))
    df = pd.concat([pd.read_csv(OUT / "b" / r["name"] / "cells.csv.gz") for r in results])
    return dict(jobs=results, contrasts=b_summary(df, OUT / "b").to_dict("records"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["a", "b", "words", "patterns", "finalize"])
    stage = parser.parse_args().stage
    with owner_lock(OUT, stage):
        torch.set_num_threads(1)
        if (
            CFG["scope"]
            != {
                "nback_train": 15,
                "nback_dev": 4,
                "test": "forbidden",
                "protected": "forbidden",
                "tokenizer": "frozen",
            }
            or prev.read(prev.OUT / "summary.json")["status"] != "completed"
        ):
            raise ValueError("unverified public parent")
        resolved = OUT / "resolved.yaml"
        if resolved.exists() and yaml.safe_load(resolved.read_text()) != CFG:
            raise ValueError("run identity drift")
        if not resolved.exists():
            resolved.write_text(CONFIG.read_text())
        done = OUT / (stage + ".json")
        if done.exists() and (
            stage != "patterns" or (OUT / "patterns/new_record_q1_exact_null.csv.gz").exists()
        ):
            return
        start = time.monotonic()
        if stage == "a":
            result = run_a()
        elif stage == "b":
            result = run_b()
        elif stage == "words":
            from four_stream_vocabulary_controls import run

            result = run()
        elif stage == "patterns":
            from four_stream_candidate_patterns import run

            result = run()
        else:
            from report_four_stream_evidence import finalize

            result = finalize()
        prev.save(
            done,
            dict(
                status="completed",
                stage=stage,
                seconds=time.monotonic() - start,
                peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                result=result,
            ),
        )


if __name__ == "__main__":
    main()
