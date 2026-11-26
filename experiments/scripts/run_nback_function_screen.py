"""Offline n-back function, event-order and perturbation screen on frozen public inputs."""

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
from scipy.signal import butter, sosfiltfilt
import torch
import yaml

from src.analysis.four_stream_block_ridge import band_histories
from src.data.downstream_input_cache import DownstreamInputDataset
from src.data.event_alignment import window_within_alignment_support
from src.data.window_exclusions import paired_window_excluded
from src.utils.io import save_npz_atomic, write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/configs/physiology_semantic_tokenizer/nback_function_screen_v1.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
OUT = ROOT / CFG["run"]
PARENT = ROOT / CFG["panel"]
CACHE = ROOT / CFG["cache"]
DATASET = CFG["dataset"]


def read(path):
    return json.loads(path.read_text())


def check_panel(panel):
    if (
        panel.groupby("split").size().to_dict()
        != dict(zip(CFG["scope"]["splits"], CFG["scope"]["prior_events"]))
        or panel.groupby("split").subject.nunique().tolist() != CFG["scope"]["people"]
        or panel.groupby("subject").split.nunique().max() != 1
        or not panel.id.is_unique
        or set(panel.dataset) != {DATASET}
        or set(panel.record) != {CFG["record"]}
        or set(panel.condition) != set(CFG["classes"])
        or set(panel.status) != {"complete"}
    ):
        raise ValueError("unverified roster, labels or partition")


def pulse():
    t = np.arange(6000) / 200 - 5
    return (
        (np.array([1, 1, 0.2, 0.2])[:, None] * 60 * np.exp(-0.5 * ((t - 6) / 0.15) ** 2))
        .astype("f")
        .reshape(4, 15, 400)
    )


def filter2(x):
    return (
        sosfiltfilt(butter(4, 2, fs=200, btype="highpass", output="sos"), x.reshape(-1, 6000), axis=-1)
        .reshape(x.shape)
        .astype("f")
    )


def alpha_power(x):
    v = (x - x.mean(-1, keepdims=True)) * np.hanning(400)
    p = abs(np.fft.rfft(v)) ** 2 * (2 / (400 * np.sum(np.hanning(400) ** 2)))
    return p[..., 16:26].sum(-1)


@lru_cache(maxsize=2)
def reader(split):
    if split not in CFG["scope"]["splits"] or read(CACHE / "verification.json")["status"] != "passed":
        raise ValueError("unverified split or source")
    return DownstreamInputDataset(CACHE, split=split, task="nback")


def samples(meta):
    sub, split = meta.subject.iloc[0], meta.split.iloc[0]
    r = reader(split)
    key = f"{DATASET}|{sub}|{CFG['record']}"
    if sub not in r.plan["subject_partitions"][DATASET][split]:
        raise ValueError("cross-split input")
    rec = r.records[key]
    local = CACHE / "records" / DATASET / sub / CFG["record"]
    if rec.npz_path not in (local, Path("/SSD_2/pid-mcm-implementation") / local.relative_to(ROOT)):
        raise ValueError("record location drift")
    r.records[key] = replace(rec, npz_path=local)
    indices = {s["id"]: i for i, s in enumerate(r.samples) if s["join_key"] == key}
    records, voltage, hb = [], [], []
    for row in meta.itertuples():
        entry = dict(
            id=row.id,
            subject=sub,
            split=split,
            event=row.event,
            condition=row.condition,
            eeg_marker=row.eeg_marker,
            hb_marker=row.hb_marker,
            status="complete",
        )
        records.append(entry)
        if row.id not in indices:
            entry["status"] = "upstream synchrony exclusion"
            continue
        idx = indices[row.id]
        plan = r.samples[idx]
        if (
            plan["condition"] != row.condition
            or plan["event_index"] != row.event
            or plan["event_relative_start_s"] != -5
            or plan["duration_s"] != 30
            or plan["event"]["alignment_role"] != "paired_eeg_fnirs_marker"
            or not np.allclose(
                [plan["eeg_start_s"], plan["hb_start_s"]],
                [row.eeg_marker - 5, row.hb_marker - 5],
                atol=1e-7,
                rtol=0,
            )
        ):
            raise ValueError("event identity or marker drift")
        if not window_within_alignment_support(plan["event"], -5, 30) or paired_window_excluded(
            DATASET, split, sub, CFG["record"], plan["eeg_start_s"], plan["hb_start_s"], 30
        ):
            entry["status"] = "time support or registered quality exclusion"
            continue
        s = r[idx]
        if s["unit"] != {"eeg": "uV", "fnirs": "uM"} or s["partial"]:
            raise ValueError("unexpected coordinates")
        names = CFG["sites"]
        ei = [s["channel_names"]["eeg"].index(n) for n in names["eeg"]]
        hi = [s["channel_names"]["fnirs"].index(n + "_" + c) for n in names["hb"] for c in ("HbO", "HbR")]
        e = s["eeg"][ei].reshape(4, 15, 400)
        h = s["fnirs"][hi].reshape(4, 2, 300).transpose(0, 2, 1)
        if not (
            s["valid_mask"]["eeg"][ei].all()
            and s["valid_mask"]["fnirs"][hi].all()
            and np.isfinite(e).all()
            and np.isfinite(h).all()
        ):
            entry["status"] = "incomplete measurements"
            continue
        voltage.append(e)
        hb.append(h - h[:, :50].mean(1, keepdims=True))
    return pd.DataFrame(records), np.asarray(voltage, dtype="f"), np.asarray(hb, dtype="f")


@torch.inference_mode()
def encode(x, mod, scale):
    from audit_aligned_tokenizer import network

    net = network(mod, CFG["model"][mod], CFG["model"]["seed"], "cpu")
    z = (x / scale[None, :, None, None]).astype("f").reshape(-1, 15, x.shape[-1])
    ids, dec = [], []
    for lo in range(0, len(z), 16):
        pred = net(torch.from_numpy(z[lo : lo + 16].copy()))
        ids.append(np.stack([pred["ids"][b].numpy() for b in "SO"], axis=1))
        if mod == "eeg":
            dec.append(np.stack([pred[b].numpy() for b in "SO"], axis=1))
    codes = np.concatenate(ids).reshape(len(x), 4, 2, 15).astype("uint16")
    if mod == "eeg":
        decoded = np.concatenate(dec).reshape(len(x), 4, 2, 15, 400) * scale[None, :, None, None, None]
        return codes, decoded.astype("f")
    return codes, None


def extract(meta, pilot=False):
    started = time.monotonic()
    torch.set_num_threads(1)
    sub = meta.subject.iloc[0]
    folder = OUT / ("pilot" if pilot else "subjects")
    folder.mkdir(exist_ok=True)
    dest = folder / (sub + ".npz")
    if dest.exists():
        return read(folder / (sub + ".json"))
    ledger, e, h = samples(meta)
    ledger.to_csv(folder / (sub + ".csv"), index=False)
    if not len(e):
        raise ValueError("empty subject: " + sub)
    scales = read(ROOT / CFG["scales"])
    coord = read(ROOT / CFG["hb_coordinates"])
    if scales["fitting_split"] != "train" or coord["sites"] != CFG["sites"]:
        raise ValueError("invalid frozen scales")
    sc = {
        m: np.array([scales["positions"][m][DATASET][n]["scale"] for n in names], dtype="f")
        for m, names in CFG["sites"].items()
    }
    hids, _ = encode(h.reshape(len(h), 4, 15, 40), "hb", sc["hb"] / coord["training_hb_gain"])
    payload = dict(id=ledger.query("status=='complete'").id.to_numpy(dtype=str), h=h, hids=hids)

    def methods(voltage):
        ids, d = encode(voltage, "eeg", sc["eeg"])
        return ids, dict(X=voltage, HP=filter2(voltage), S=d[:, :, 0], O=d[:, :, 1], SO=d.sum(2))

    clean, waves = methods(e)
    dev = meta.split.iloc[0] == "validation"
    payload["ids"] = clean
    for k, w in waves.items():
        payload["bands_" + k] = band_histories(w, CFG["bands"]).astype("f")
    if dev:
        dirty, disturbed = methods(e + pulse())
        payload.update(dirty_ids=dirty, e=e, wave_S=waves["S"], dirty_S=disturbed["S"])
        for k, v in disturbed.items():
            payload["dirty_bands_" + k] = band_histories(v, CFG["bands"]).astype("f")
            payload["transfer_" + k] = np.sqrt(
                np.mean((v - waves[k]) ** 2, axis=(1, 2, 3)) / np.mean(pulse() ** 2)
            )
            payload["power_" + k] = alpha_power(waves[k])[:, 2:].mean((1, 2)) / alpha_power(e)[:, 2:].mean(
                (1, 2)
            )
    perf = dict(
        subject=sub,
        events=len(e),
        seconds=time.monotonic() - started,
        peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    write_json(folder / (sub + ".json"), perf)
    save_npz_atomic(dest, **payload)
    return perf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["synthetic", "pilot", "extract", "score"])
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    if args.stage == "synthetic":
        from score_nback_function_screen import synthetic

        synthetic()
        return
    if not (OUT / "synthetic.json").exists() or read(PARENT / "verification.json")["status"] != "passed":
        raise ValueError("preflight required")
    meta = pd.read_csv(PARENT / "panel.csv")
    check_panel(meta)
    if args.stage == "score":
        from score_nback_function_screen import score

        score(meta)
        return
    if args.stage == "pilot":
        print(extract(meta[meta.split.eq("train")].iloc[:3], True), flush=True)
        return
    groups = [g for _, g in meta.groupby("subject")]
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(groups)), mp_context=get_context("fork")
    ) as ex:
        futures = [ex.submit(extract, g) for g in groups]
        for f in as_completed(futures):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
