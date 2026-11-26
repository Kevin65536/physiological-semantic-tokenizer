"""Prepare a common four-site, complete-case G0 panel with registered native EOG."""

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import yaml

import run_frozen_science_screen as screen
from run_dual_component_tokenizer import owner_lock
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_training_data
from src.data.downstream_input_cache import DownstreamInputDataset
from src.data.unified_physiology import load_native_eeg_record
from src.data.window_exclusions import paired_window_excluded
from src.data.event_alignment import window_within_alignment_support
from src.utils.io import save_npz_atomic

ROOT = screen.ROOT
CONFIG = screen.CONFIG.with_name("science_ablations_v1.yaml")
CFG = yaml.safe_load(CONFIG.read_text())
OUT = ROOT / CFG["run"]
EEG, HB = CFG["sites"]["eeg"], CFG["sites"]["hb"]
STATE = {}


def check():
    if (
        CFG["scope"]
        != dict(
            dataset=screen.parent.DS,
            record="cnt_nback",
            splits=["train", "validation"],
            test="forbidden",
            protected="forbidden",
        )
        or CFG["parent"] != str(screen.OUT.relative_to(ROOT))
        or CFG["training"]["seeds"] != [17, 43]
        or EEG != ["F1", "F2", "P3", "P4"]
        or HB != ["AFF3h", "AFF4h", "P3h", "P4h"]
    ):
        raise ValueError("unexpected ablation contract")


def donor_indices(frame):
    """Same person, native record and condition. Match closest event phase across disjoint crops."""
    donor = np.full(len(frame), -1)
    for _, idx in frame.groupby(["subject", "split", "condition"]).indices.items():
        for i in idx:
            eligible = idx[np.abs(frame.start.values[idx] - frame.start.values[i]) >= 40]
            if len(eligible):
                donor[i] = eligible[np.argmin(abs(frame.phase.values[eligible] - frame.phase.values[i]))]
    return donor


def load_person(person):
    meta, rows, scales, readers, by, allowed = STATE["inputs"]
    who = meta.query("subject==@person")
    role = who.split.iloc[0]
    reader = readers[role]
    if person not in reader.plan["subject_partitions"][screen.parent.DS][role]:
        raise ValueError("subject outside public roster")
    key = f"{screen.parent.DS}|{person}|cnt_nback"
    rec = reader.records[key]
    native = load_native_eeg_record(ROOT, rec)
    if (
        native.sample_rate_hz != 200
        or native.native_unit != "uV"
        or set(n.upper() for n in native.auxiliary_channel_names) != {"HEOG", "VEOG"}
    ):
        raise ValueError("two measured 200 Hz EOG channels required")
    ocular = native.auxiliary_values[
        :, [list(n.upper() for n in native.auxiliary_channel_names).index(k) for k in ("HEOG", "VEOG")]
    ]
    sites = [native.channel_names.index(n) for n in EEG]
    events = {v["id"]: v["event"] for v in reader.samples if v["subject"] == person and v["join_key"] == key}
    sessions = [s for s in reader.samples if s["join_key"] == key]
    marker = np.array([s["event"]["eeg_time_ms"] / 1000 for s in sessions])
    units = np.array([scales["positions"]["eeg"][screen.parent.DS][k]["scale"] for k in EEG])
    cache = OrderedDict()
    records = []
    blocks = []
    omitted = []
    for source in by[person]:
        ei, hi = source
        e, h = rows[ei], rows[hi]
        ep = [e["positions"].index(n) for n in EEG]
        hp = [h["positions"].index(n) for n in HB]
        hm = {v["start_s"]: v["positions"] for v in h["inputs"]["score"]}
        for entry in e["inputs"]["score"]:
            t = entry["start_s"]
            loc = dict(subject=person, split=role, parent=e["parent"], offset=t)
            if not set(ep) <= set(entry["positions"]) or not set(hp) <= set(hm.get(t, [])):
                omitted.append(dict(**loc, reason="four-site G0 QC"))
                continue
            if role == "train" and (
                not set(ep) <= allowed.get((ei, t), set()) or not set(hp) <= allowed.get((hi, t), set())
            ):
                omitted.append(dict(**loc, reason="outside optimization support"))
                continue
            b = base_tokenizer_batch(rows, [(ei, t, p) for p in ep], scales, array_cache=cache)
            a = base_tokenizer_batch(rows, [(hi, t, p) for p in hp], scales, array_cache=cache)
            start = e["native_start_sample"] + t * 200
            eye = ocular[start : start + 6000]
            if (
                not b["mask"].all()
                or not a["mask"].all()
                or eye.shape != (6000, 2)
                or not np.isfinite(eye).all()
            ):
                omitted.append(dict(**loc, reason="incomplete mask/EOG"))
                continue
            phase = start / 200 + 15 - marker
            ix = np.argmin(abs(phase))
            np.testing.assert_allclose(b["x"], b["s"] + b["o"], atol=1e-4, rtol=2e-5)
            records.append(
                dict(**loc, start=start / 200, condition=sessions[ix]["condition"], phase=phase[ix])
            )
            blocks.append(dict(x=b["x"], s=b["s"], h=a["x"], eog=eye.T.reshape(2, 15, 400)))
    if not blocks:
        raise ValueError(f"no eligible crop for {person}")
    folder = OUT / "data"
    folder.mkdir(exist_ok=True)
    frame = pd.DataFrame(records)
    save_npz_atomic(
        folder / (person + ".npz"),
        **{k: np.stack([v[k] for v in blocks]).astype("float32") for k in blocks[0]},
    )
    frame.to_csv(folder / (person + ".csv"), index=False)
    pd.DataFrame(omitted, columns=["subject", "split", "parent", "offset", "reason"]).to_csv(
        folder / (person + "_exclusions.csv"), index=False
    )
    # One pair of 30s views per event for the same middle patch [-16,-14) before cutoff.
    temporal = []
    history = []
    eog_events = []
    for r in who.itertuples():
        stop = round(r.eeg_stop * 200)
        eog_events.append(ocular[stop - 6000 : stop].T.reshape(2, 15, 400))
        begin = stop - 8800
        valid = (
            begin >= 0
            and window_within_alignment_support(events[r.id], -40.1, 44.2)
            and not paired_window_excluded(
                screen.parent.DS, role, person, "cnt_nback", r.eeg_start - 14, r.hb_start - 14, 44.1
            )
        )
        part = native.values[begin : stop - 2800, sites] if valid else np.empty((0, 4))
        valid = (
            valid and part.shape == (6000, 4) and np.isfinite(part).all() and (np.ptp(part, axis=0) > 0).all()
        )
        temporal.append(dict(id=r.id, valid=valid))
        if valid:
            history.append((part.T.reshape(4, 15, 400) / units[:, None, None]).astype("float32"))
    pd.DataFrame(temporal).to_csv(folder / (person + "_history.csv"), index=False)
    save_npz_atomic(
        folder / (person + "_views.npz"),
        history=np.asarray(history),
        event_ids=who.id.to_numpy(dtype=str),
        eog=np.asarray(eog_events, dtype="float32"),
    )
    return dict(person=person, kept=len(frame), excluded=len(omitted), history=len(history))


def prepare():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, required=True)
    args = p.parse_args()
    check()
    with owner_lock(OUT, "prepare"):
        resolved = OUT / "resolved.yaml"
        if resolved.exists() and yaml.safe_load(resolved.read_text()) != CFG:
            raise ValueError("contract drift")
        resolved.write_text(CONFIG.read_text())
        if (OUT / "prepared.json").is_file():
            return
        m, _, _ = screen.panel()
        if not (screen.parent.read(screen.OUT / "summary.json")["status"] == "completed" and len(m) == 400):
            raise ValueError("incomplete screening parent")
        cfg = screen.audit.aligned.settings("eeg")[0]
        manifest, scales, _, _ = base_tokenizer_training_data(
            cfg, ROOT, recorded_root=screen.spectral.PLAN["recorded_root"]
        )
        rows = manifest["rows"]
        lookup = {}
        allowed = {}
        for i, r in enumerate(rows):
            if (
                r["dataset"] == screen.parent.DS
                and r["record"] == "cnt_nback"
                and r["subject"] in set(m.subject)
            ):
                lookup.setdefault((r["subject"], r["parent"]), {})[r["modality"]] = i
                for v in r["inputs"].get("train", []):
                    allowed[i, v["start_s"]] = set(v["positions"])
        by = {
            s: [(v["eeg"], v["hb"]) for (who, _), v in lookup.items() if who == s and set(v) == {"eeg", "hb"}]
            for s in m.subject.unique()
        }
        readers = {
            s: DownstreamInputDataset(screen.parent.DATA, split=s, task="nback")
            for s in ("train", "validation")
        }
        STATE["inputs"] = m, rows, scales, readers, by, allowed
        start = time.monotonic()
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("fork")) as pool:
            results = list(pool.map(load_person, m.subject.unique()))
        frame = pd.concat(
            [pd.read_csv(OUT / "data" / (s + ".csv")) for s in m.subject.unique()], ignore_index=True
        )
        donor = donor_indices(frame)
        frame["donor"] = donor
        frame.to_csv(OUT / "crops.csv", index=False)
        refs = [scales["positions"]["eeg"][screen.parent.DS][k] for k in EEG]
        screen.parent.save(OUT / "scales.json", refs)
        screen.parent.save(
            OUT / "prepared.json",
            dict(
                status="completed",
                source=cfg["data"]["run"],
                subjects=results,
                crops=len(frame),
                unpaired=int(sum(donor < 0)),
                seconds=time.monotonic() - start,
            ),
        )


if __name__ == "__main__":
    prepare()
