"""Run the public three-task pattern screen with resumable record encoding."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
import json
from multiprocessing import get_context
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import torch
import yaml

from src.analysis.task_pattern_expansion import spectral_db
from src.data.task_pattern_panel import record_key
from src.data.downstream_input_cache import DownstreamInputDataset
from src.data.event_alignment import window_within_alignment_support
from src.utils.io import save_npz_atomic, write_json

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "experiments/configs/physiology_semantic_tokenizer/task_pattern_expansion_v1.yaml"
CFG = yaml.safe_load(CONTRACT.read_text())
OUT = ROOT / CFG["run"]
CACHE = ROOT / CFG["cache"]


def read(path):
    return json.loads(path.read_text())


@lru_cache(maxsize=2)
def reader(split):
    if split not in ("train", "validation") or read(CACHE / "verification.json")["status"] != "passed":
        raise ValueError("partition or cache verification")
    return DownstreamInputDataset(CACHE, split=split)


def plan():
    tables = []
    prior = []
    for split in CFG["scope"]["splits"]:
        r = reader(split)
        admitted = {row["id"]: i for i, row in enumerate(r.samples) if row["task"] in CFG["tasks"]}
        for s in r.plan["exclusions"]:
            if s["split"] == split and s["task"] in CFG["tasks"]:
                prior.append(s)
        for s in r.plan["samples"]:
            task = s["task"]
            if s["split"] != split or task not in CFG["tasks"]:
                continue
            t = CFG["tasks"][task]
            ds = t["dataset"]
            if (
                s["dataset"] != ds
                or s["subject"] not in r.plan["subject_partitions"][ds][split]
                or s["record"] not in t["records"]
                or s["condition"] not in t["classes"]
                or s["event_relative_start_s"] != -5
                or s["duration_s"] != 30
                or s["partial"]
                or s["event"]["alignment_role"] != "paired_eeg_fnirs_marker"
            ):
                raise ValueError("scope/shape drift")
            idx = admitted.get(s["id"], -1)
            flag = "ready" if idx >= 0 else "upstream quality exclusion"
            if flag == "ready" and not window_within_alignment_support(s["event"], -5, 30):
                flag = "clock support"
            tables.append(
                dict(
                    task=task,
                    dataset=ds,
                    split=split,
                    subject=s["subject"],
                    record=s["record"],
                    event=s["event_index"],
                    id=s["id"],
                    condition=s["condition"],
                    label=t["classes"].index(s["condition"]),
                    eeg_start=s["eeg_start_s"],
                    hb_start=s["hb_start_s"],
                    idx=idx,
                    status=flag,
                )
            )
    data = pd.DataFrame(tables)
    if data.id.duplicated().any() or data.groupby(["dataset", "subject"]).split.nunique().max() != 1:
        raise ValueError("overlap")
    data.to_csv(OUT / "plan.csv", index=False)
    pd.DataFrame(
        prior,
        columns=["dataset", "subject", "split", "task", "record", "event_index", "reason", "join_key"],
    ).to_csv(OUT / "upstream_exclusions.csv", index=False)
    return data


@lru_cache(maxsize=4)
def family(mod):
    z = CFG["encoder"][mod]
    result = []
    for b in "SO":
        a = np.asarray(read(ROOT / CFG["families"] / f"{mod}_17" / (b + "_family.json"))["mapping"])
        if a.shape != (z,) or set(a) != set(range(4)):
            raise ValueError("training family map")
        result.append(a)
    return np.array(result)


@torch.inference_mode()
def encode(x, ds, mod, gain=1.0):
    from audit_aligned_tokenizer import network

    net = network(mod, CFG["encoder"][mod], 17, "cpu")
    src = read(ROOT / CFG["scales"])
    if src["fitting_split"] != "train":
        raise ValueError("invalid frozen scale")
    scales = [src["positions"][mod][ds][n]["scale"] for n in CFG["sites"][ds][mod]]
    inp = (x * (gain / np.array(scales))[None, :, None, None]).astype("f").reshape(-1, 15, x.shape[-1])
    maps = family(mod)
    out = []
    for i in range(0, len(inp), 24):
        pred = net(torch.from_numpy(np.ascontiguousarray(inp[i : i + 24])))
        out.append(np.stack([maps[j][pred["ids"][b].numpy()] for j, b in enumerate("SO")], -1))
    return np.concatenate(out).reshape(len(x), x.shape[1], 15, 2).astype("uint8")


def extract(g, pilot=False):
    torch.set_num_threads(1)
    started = time.monotonic()
    name = record_key(g)
    folder = OUT / ("pilot" if pilot else "records")
    folder.mkdir(exist_ok=True)
    dest = folder / (name + ".npz")
    if dest.exists():
        return name
    g = g.copy()
    row = g.iloc[0]
    ds = row.dataset
    r = reader(row.split)
    if row.subject not in r.plan["subject_partitions"][ds][row.split]:
        raise ValueError("invalid subject")
    join = "|".join((ds, row.subject, row.record))
    rec = r.records[join]
    path = CACHE / "records" / ds / row.subject / row.record
    if rec.npz_path not in (path, Path("/SSD_2/pid-mcm-implementation") / path.relative_to(ROOT)):
        raise ValueError("record path changed")
    r.records[join] = replace(rec, npz_path=path)
    eeg = []
    hb = []
    ids = []
    for ix, row in g.iterrows():
        if row.status != "ready":
            continue
        s = r[int(row.idx)]
        if (
            s["id"] != row.id
            or s["unit"] != {"eeg": "uV", "fnirs": CFG["sites"][ds]["hb_unit"]}
            or s["split"] != row.split
        ):
            raise ValueError("event coordinates")
        names = CFG["sites"][ds]
        ei = [s["channel_names"]["eeg"].index(n) for n in names["eeg"]]
        hi = [s["channel_names"]["fnirs"].index(n + "_" + c) for n in names["hb"] for c in ("HbO", "HbR")]
        e = s["eeg"][ei]
        h = s["fnirs"][hi].reshape(4, 2, 300).transpose(0, 2, 1)
        if not (
            s["valid_mask"]["eeg"][ei].all()
            and s["valid_mask"]["fnirs"][hi].all()
            and np.isfinite(e).all()
            and np.isfinite(h).all()
        ):
            g.loc[ix, "status"] = "unavailable paired sensors"
            continue
        g.loc[ix, "status"] = "complete"
        ids.append(row.id)
        eeg.append(e.reshape(6, 15, 400))
        hb.append(h - h[:, :50].mean(1, keepdims=True))
    g.to_csv(folder / (name + ".csv"), index=False)
    if ids:
        e = np.array(eeg, dtype="f")
        h = np.array(hb, dtype="f")
        payload = dict(e=spectral_db(e).astype("f"), ef=encode(e, ds, "eeg"), h=h)
    else:
        payload = {}
    save_npz_atomic(dest, ids=np.array(ids, dtype=str), **payload)
    perf = dict(
        record=name,
        events=len(ids),
        seconds=time.monotonic() - started,
        peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    write_json(folder / (name + ".json"), perf)
    return perf


def hbjob(args):
    path, ds, gain = args
    torch.set_num_threads(1)
    dest = OUT / "hb" / path.name
    if dest.exists():
        return
    with np.load(path) as z:
        if len(z["ids"]):
            x = z["h"].reshape(-1, 4, 15, 40)
            save_npz_atomic(dest, hf=encode(x, ds, "hb", gain), ids=z["ids"])


def hbplan(meta):
    (OUT / "hb").mkdir(exist_ok=True)
    gains = {}
    jobs = []
    for task, g in meta.groupby("task"):
        ss = 0.0
        n = 0
        pathlist = []
        for _, group in g.groupby(["subject", "record"]):
            path = OUT / "records" / (record_key(group) + ".npz")
            pathlist.append(path)
            if group.iloc[0].split != "train":
                continue
            with np.load(path) as z:
                if len(z["ids"]):
                    h = z["h"].astype(float)
                    ss += (h * h).sum()
                    n += h.size
        gain = 0.025 / np.sqrt(ss / n)
        gains[task] = dict(fitting_split="train", observations=n, gain=gain)
        jobs.extend((p, CFG["tasks"][task]["dataset"], gain) for p in pathlist)
    write_json(OUT / "hb_scales.json", gains)
    return jobs


def parallel(fn, things, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("fork")) as pool:
        for f in as_completed([pool.submit(fn, arg) for arg in things]):
            res = f.result()
            if res is not None:
                print(res, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["plan", "pilot", "run"])
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "plan":
        if CFG["seed"] != 1127 or CFG["scope"]["splits"] != ["train", "validation"]:
            raise ValueError("contract")
        (OUT / "resolved.yaml").write_text(CONTRACT.read_text())
        plan()
        return
    if not (OUT / "synthetic.log").is_file() or (OUT / "resolved.yaml").read_text() != CONTRACT.read_text():
        raise ValueError("preflight")
    meta = pd.read_csv(OUT / "plan.csv")
    if args.stage == "pilot":
        for task, g in meta.query("split=='train' and status=='ready'").groupby("task"):
            rec = g.iloc[0]
            print(
                extract(g[(g.subject == rec.subject) & (g.record == rec.record)].iloc[:3], True), flush=True
            )
        return
    groups = [g for _, g in meta.groupby(["task", "subject", "record"])]
    # Open the verified metadata once before forking; workers map signal arrays on demand.
    for split in CFG["scope"]["splits"]:
        reader(split)
    parallel(extract, groups, args.workers)
    parallel(hbjob, hbplan(meta), args.workers)


if __name__ == "__main__":
    main()
