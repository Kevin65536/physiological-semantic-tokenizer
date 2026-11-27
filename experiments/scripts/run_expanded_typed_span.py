"""Four-task expanded training, preserving the original aligned integer vocabularies."""

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F
import yaml

import run_typed_span as base
from src.data.channel_geometry import project_head_topview
from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_batch
from src.data.event_alignment import window_within_alignment_support
from src.utils.io import save_npz_atomic
from run_dual_component_tokenizer import owner_lock

ROOT = base.ROOT
CFGFILE = base.CONFIG.with_name("typed_span_expanded_v2.yaml")
CFG = yaml.safe_load(CFGFILE.read_text())
OUT = ROOT / CFG["run"]
CACHE = ROOT / CFG["scope"]["cache"]
TR = CFG["training"]
PARENTS = [OUT.parents[1] / p for p in CFG["panels"]]
SINGLE = "eeg_fnirs_single_trial"
SIM = "simultaneous_eeg_nirs"
GROUPS = ["mental_arithmetic", "wg", "nback"]
STATE = {}


def check():
    if (
        CFG["scope"]
        != dict(
            splits=["train", "validation"],
            test="forbidden",
            protected="forbidden",
            cache="data/cache/physiology_downstream_v2",
        )
        or CFG["tensor"]
        != dict(
            sites=4,
            slots=15,
            seconds_per_slot=2,
            eeg_samples=400,
            hb_samples_colors=[20, 2],
            vocabulary=base.SIZES,
        )
        or TR["steps"] % 24
        or TR["seeds"] != [17, 43]
        or TR["arms"] != ["joint", "no_lag"]
        or CFG["tasks"] != ["mental_arithmetic", "motor_imagery", "wg", "nback"]
    ):
        raise ValueError("expanded protocol drift")


def task_dataset(task):
    return SINGLE if task in CFG["tasks"][:2] else SIM


def pair_donors(meta):
    """Maximum disjoint, within-person/record/condition permutation. Unmatched rows stay explicit."""
    out = np.full(len(meta), -1)
    for _, idx in meta.groupby(
        ["dataset", "subject", "split", "task", "record", "condition"]
    ).indices.items():
        a = meta.eeg_start.to_numpy()[idx]
        h = meta.hb_start.to_numpy()[idx]
        valid = (abs(a[:, None] - a) >= 30) & (abs(h[:, None] - h) >= 30)
        rng = np.random.default_rng(1130)
        cost = np.where(valid, -len(idx) * 2 + rng.random(valid.shape), np.where(np.eye(len(idx)), 0, 1e7))
        i, j = linear_sum_assignment(cost)
        moved = i != j
        if not valid[i[moved], j[moved]].all():
            raise ValueError("illegal donor")
        out[idx[i[moved]]] = idx[j[moved]]
    if sorted(out[out >= 0]) != list(np.flatnonzero(out >= 0)):
        raise ValueError("donors change source marginals")
    return out


def extract(item):
    from run_task_pattern_expansion import reader

    key, g = item
    split, task, person, record = key
    ds = task_dataset(task)
    r = reader(split)
    if person not in r.plan["subject_partitions"][ds][split]:
        raise ValueError("person outside split")
    ident = "|".join([ds, person, record])
    rec = r.records[ident]
    path = CACHE / "records" / ds / person / record
    if rec.npz_path not in (path, Path("/SSD_2/pid-mcm-implementation") / path.relative_to(ROOT)):
        raise ValueError("cache record path")
    r.records[ident] = replace(rec, npz_path=path)
    lookup = {s["id"]: i for i, s in enumerate(r.samples) if s["join_key"] == ident}
    an, bn = CFG["sites"][task]
    hb_names = [n + "_" + c for n in bn for c in ("HbO", "HbR")]
    stored = {"e": [], "h": [], "xy": [], "ids": []}
    for row in g.itertuples():
        i = lookup[row.id]
        source = r.samples[i]
        if (
            source["subject"] != person
            or source["task"] != task
            or source["condition"] != row.condition
            or source["event"]["alignment_role"] != "paired_eeg_fnirs_marker"
            or source["duration_s"] != 30
            or source["event_relative_start_s"] != -5
            or not window_within_alignment_support(source["event"], -5, 30)
            or not np.allclose(
                [source["eeg_start_s"], source["hb_start_s"]],
                [row.eeg_start, row.hb_start],
                atol=1e-6,
                rtol=0,
            )
        ):
            raise ValueError("event clock/identity drift")
        s = r[i]
        ei = [s["channel_names"]["eeg"].index(n) for n in an]
        hi = [s["channel_names"]["fnirs"].index(n) for n in hb_names]
        if (
            s["unit"] != {"eeg": "uV", "fnirs": ("uM" if ds == SIM else "relative_Hb_repo_approximation")}
            or not s["valid_mask"]["eeg"][ei].all()
            or not s["valid_mask"]["fnirs"][hi].all()
        ):
            raise ValueError("eligible sensor support changed")
        stored["e"].append(s["eeg"][ei].reshape(4, 15, 400).astype("f"))
        stored["h"].append(
            s["fnirs"][hi].reshape(4, 2, 300).transpose(0, 2, 1).reshape(4, 15, 40).astype("f")
        )
        stored["xy"].append(
            [
                [project_head_topview(s["channel_geometry"][mod][j]) for j in inds]
                for mod, inds in [("eeg", ei), ("fnirs", hi[::2])]
            ]
        )
        stored["ids"].append(row.id)
    dest = OUT / "records" / ("_".join(key) + ".npz")
    save_npz_atomic(dest, **{k: np.asarray(v) for k, v in stored.items()})
    return len(g)


def anchors(task):
    rows, scales = STATE["anchors"]
    ds = task_dataset(task)
    people, records = STATE["groups"][task]
    candidates = {}
    for i, row in enumerate(rows):
        if (
            row["dataset"] != ds
            or (row["subject"], row["split"]) not in people
            or row["record"] not in records
        ):
            continue
        valid_role = "train" if row["split"] == "train" else "score"
        mod = row["modality"]
        names = CFG["sites"][task][int(mod == "hb")]
        if not set(names) <= set(row["positions"]):
            continue
        pos = [row["positions"].index(n) for n in names]
        allowed = {v["start_s"] for v in row["inputs"][valid_role] if set(pos) <= set(v["positions"])}
        for v in row["inputs"]["score"]:
            if v["start_s"] in allowed and set(pos) <= set(v["positions"]):
                key = (row["parent"], v["start_s"], row["subject"], row["split"])
                candidates.setdefault(key, {})[mod] = (i, pos)
    result = {k: [] for k in ("ex", "es", "hx", "hs")}
    meta = []
    cache = OrderedDict()
    for key, pair in sorted(candidates.items()):
        if len(pair) != 2:
            continue
        batch = []
        for mod in ("eeg", "hb"):
            i, pos = pair[mod]
            batch.append(
                base_tokenizer_batch(
                    rows, [(i, key[1], k) for k in pos], scales, array_cache=cache, cache_limit=8
                )
            )
        if not all(b["mask"].all() for b in batch):
            continue
        for mod, b in zip("eh", batch):
            for k in "xs":
                result[mod + k].append(b[k].astype("f"))
        meta.append(key)
    if not meta:
        return dict(task=task, eligible=0)
    frame = pd.DataFrame(meta, columns=["parent", "offset", "subject", "split"])
    frame.to_csv(OUT / (task + "_anchors.csv"), index=False)
    save_npz_atomic(OUT / (task + "_anchors.npz"), **{k: np.stack(v) for k, v in result.items()})
    return dict(task=task, counts=frame.groupby("split").size().to_dict())


def support():
    aa, bb = PARENTS
    for p in PARENTS:
        if base.read(p / "verification.json")["status"] != "passed":
            raise ValueError("unverified upstream")
    a = pd.read_csv(aa / "support.csv")
    b = pd.read_csv(bb / "support.csv").assign(task="nback", dataset=SIM, record="cnt_nback")
    b["eeg_start"] = b.eeg_marker - 5
    b["hb_start"] = b.hb_marker - 5
    columns = [
        "task",
        "dataset",
        "split",
        "subject",
        "record",
        "event",
        "id",
        "condition",
        "eeg_start",
        "hb_start",
        "status",
    ]
    ledger = pd.concat([a[columns], b[columns]], ignore_index=True)
    if (
        set(ledger.split) != {"train", "validation"}
        or ledger.id.duplicated().any()
        or ledger.groupby(["dataset", "subject"]).split.nunique().max() != 1
    ):
        raise ValueError("parent panel overlap")
    meta = (
        ledger[ledger.status == "complete"]
        .sort_values(["split", "task", "subject", "record", "event"])
        .reset_index(drop=True)
    )
    meta["donor"] = pair_donors(meta)
    return ledger, meta


def prepare(stage, workers):
    import retrain_single_vq_aligned as aligned
    import run_single_vq_spectral_location as spec
    from run_task_pattern_expansion import reader

    with owner_lock(OUT, "prepare"):
        if (OUT / "prepared.json").is_file():
            return
        ledger, m = support()
        ledger.to_csv(OUT / "support.csv", index=False)
        m.to_csv(OUT / "events.csv", index=False)
        (OUT / "records").mkdir(exist_ok=True)
        items = list(m.groupby(["split", "task", "subject", "record"], sort=False))
        for split in ("train", "validation"):
            reader(split)
        if stage == "prep_pilot":
            items = [
                next(item for item in items if item[0][1] == t and item[0][0] == "train")
                for t in CFG["tasks"]
            ]
        start = time.monotonic()
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
            events = sum(pool.map(extract, items))
        if stage == "prep_pilot":
            base.save(
                OUT / "prep_pilot.json",
                dict(events=events, workers=workers, seconds=time.monotonic() - start),
            )
            return
        cfg, _ = aligned.settings("eeg")
        manifest, scales, _, _ = base_tokenizer_training_data(
            cfg, ROOT, recorded_root=spec.PLAN["recorded_root"]
        )
        STATE["anchors"] = manifest["rows"], scales
        STATE["groups"] = {}
        for t in GROUPS:
            g = m[m.task.isin(CFG["tasks"][:2] if t == GROUPS[0] else [t])]
            STATE["groups"][t] = set(zip(g.subject, g.split)), set(g.record)
        with ProcessPoolExecutor(min(workers, 3), mp_context=mp.get_context("fork")) as pool:
            counts = list(pool.map(anchors, GROUPS))
        if events != len(m) or not any("counts" in v for v in counts):
            raise ValueError("incomplete preparation")
        base.save(
            OUT / "prepared.json",
            dict(
                status="completed",
                events=events,
                anchors=counts,
                workers=workers,
                seconds=time.monotonic() - start,
                source=cfg["data"]["run"],
                schema="expanded_events_v2",
            ),
        )


def load(device):
    from retrain_single_vq_aligned import objective

    prepared = base.read(OUT / "prepared.json")
    m = pd.read_csv(OUT / "events.csv")
    if prepared["status"] != "completed" or prepared["events"] != len(m):
        raise ValueError("incomplete panel")
    refs = base.read(ROOT / prepared["source"] / "scales.json")
    if refs["fitting_split"] != "train":
        raise ValueError("scale provenance")
    scale = refs["positions"]
    parts = {k: [] for k in ("e", "h", "xy")}
    todev = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    for key, g in m.groupby(["split", "task", "subject", "record"], sort=False):
        with np.load(OUT / "records" / ("_".join(key) + ".npz")) as z:
            np.testing.assert_array_equal(z["ids"], g.id)
            for k in parts:
                parts[k].append(z[k])
    e, h, xy = (np.concatenate(v).astype("f") for v in parts.values())
    tr = m.split.eq("train").to_numpy()
    gains = {}
    for task, indices in m.groupby("task").indices.items():
        ds = task_dataset(task)
        es, hs = (
            [scale[mod][ds][n]["scale"] for n in names]
            for mod, names in zip(("eeg", "hb"), CFG["sites"][task])
        )
        a = h[indices[tr[indices]]].astype(float).reshape(-1, 4, 300, 2)
        gain = 0.025 / np.sqrt(np.mean((a - a[:, :, :50].mean(2, keepdims=True)) ** 2))
        e[indices] /= np.array(es)[None, :, None, None]
        h[indices] *= (gain / np.array(hs))[None, :, None, None]
        gains[task] = gain
    anc = []
    calibration = objective()["calibration"]
    for info in prepared["anchors"]:
        if "counts" not in info:
            continue
        task = info["task"]
        ds = task_dataset(task)
        am = pd.read_csv(OUT / (task + "_anchors.csv"))
        ix = np.flatnonzero(am.split == "train")
        with np.load(OUT / (task + "_anchors.npz")) as z:
            pairs = [(todev(z[c + "x"]), todev(z[c + "s"])) for c in "eh"]
        ref = [[scale[mod][ds][n] for n in names] for mod, names in zip(("eeg", "hb"), CFG["sites"][task])]
        ad = dict(
            anc=pairs,
            scale=todev([v["scale"] for v in ref[0]]),
            var=[
                [
                    todev([[v[field][b] for v in group] for b in "xso"])
                    for field in ("target_variance", "difference_variance")
                ]
                for group in ref
            ],
        )
        stats = []
        for mod, (x, s) in enumerate(pairs):
            for j, y in enumerate((s[ix], x[ix] - s[ix])):
                ft = base.features(y, mod, ad["scale"])
                mean, sd = ft.mean((0, 1, 2)), ft.std((0, 1, 2), unbiased=False).clamp_min(1e-6)
                if mod == j == 0:
                    mean, sd = [todev(calibration[ds][k]) for k in ("mean", "sd")]
                stats.append((mean, sd))
        anc.append((task, ad, stats, am, ix))
    if not np.isfinite(e).all() or not np.isfinite(h).all() or not np.isfinite(xy).all():
        raise ValueError("nonfinite inputs")
    return (
        dict(
            m=m,
            raw=[todev(e), todev(h)],
            xy=todev(xy),
            gain=todev(np.ones(4)),
            train=np.flatnonzero(tr),
            dev=np.flatnonzero(~tr),
            gains=gains,
        ),
        anc,
    )


def training(arm, seed, pilot):
    device = "cuda"
    folder = OUT / ("pilot" if pilot else "fits") / f"{arm}_{seed}"
    with owner_lock(folder):
        if (folder / "done.json").is_file():
            return
        data, anchors = load(device)
        rng = np.random.default_rng(seed)
        model = base.load_model(arm, seed, device)
        opt = torch.optim.AdamW(
            [
                dict(
                    params=list(model.reader.parameters()) + list(model.local.parameters()),
                    lr=base.TR["reader_lr"],
                ),
                dict(params=model.nets.parameters(), lr=base.TR["tokenizer_lr"]),
            ]
        )
        steps = 6 if pilot else TR["steps"]
        order = np.concatenate(
            [rng.permutation(data["train"]) for _ in range(1 + steps * TR["batch"] // len(data["train"]))]
        )
        start = time.monotonic()
        for i in range(steps):
            ix = order[i * TR["batch"] : (i + 1) * TR["batch"]]
            mod, length, visible = base.task(i)
            update = i >= (2 if pilot else TR["calibration"])
            model.nets.requires_grad_(update)
            hidden = base.span_mask(
                torch.as_tensor(rng.integers(16 - length, size=len(ix)), device=device), length, 4
            )
            empty = torch.zeros_like(hidden)
            src = base.codes(model.nets[1 - mod], data, 1 - mod, ix, empty)
            tgt = base.codes(model.nets[mod], data, mod, ix, hidden)
            with torch.no_grad():
                labels = base.codes(model.nets[mod], data, mod, ix, empty)[1]
            logits = model.predict(src, tgt, data["xy"][ix], hidden, mod, visible)
            span = torch.stack([F.cross_entropy(z[hidden], y[hidden]) for z, y in zip(logits, labels)]).mean()
            _, ad, stats, _, eligible = anchors[i % len(anchors)]
            loss = span + base.anchor_loss(model, ad, rng.choice(eligible, len(ix)), stats, update)[0]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            if (i + 1) % 192 == 0 or i + 1 == steps:
                print(arm, seed, i + 1, round(loss.item(), 4), flush=True)
        if not pilot:
            tmp = folder / "weights.partial"
            torch.save(model.state_dict(), tmp)
            tmp.replace(folder / "weights.pt")
        base.save(
            folder / "done.json",
            dict(
                status="completed",
                steps=steps,
                seed=seed,
                arm=arm,
                seconds=time.monotonic() - start,
                train_windows=len(data["train"]),
                unique_exposed=len(np.unique(order[: steps * TR["batch"]])),
                peak_MiB=torch.cuda.max_memory_reserved() / 2**20,
                gains=data["gains"],
            ),
        )


def gpu():
    cards = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"], text=True
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in [
        line.split(", ")[0] for line in cards.splitlines() if line.split(", ")[1] != "00000000:17:00.0"
    ]:
        raise ValueError("one permitted GPU UUID required")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["prep_pilot", "prepare", "pilot", "train", "score", "pair_attention", "summarize"]
    )
    parser.add_argument("--arm", choices=TR["arms"], default="joint")
    parser.add_argument("--seed", type=int, choices=TR["seeds"], default=17)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    check()
    with owner_lock(OUT, "contract", blocking=True):
        resolved = OUT / "resolved.yaml"
        if resolved.exists() and yaml.safe_load(resolved.read_text()) != CFG:
            raise ValueError("contract drift")
        if not resolved.exists():
            resolved.write_text(CFGFILE.read_text())
    if args.stage in ("prep_pilot", "prepare"):
        prepare(args.stage, args.workers)
    elif args.stage in ("train", "pilot"):
        gpu()
        training(args.arm, args.seed, args.stage == "pilot")
    else:
        from score_expanded_typed_span import score, paired_attention, summarize

        if args.stage == "score":
            gpu()
            score(args.arm, args.seed)
        elif args.stage == "pair_attention":
            gpu()
            paired_attention(args.arm, args.seed)
        else:
            summarize()
