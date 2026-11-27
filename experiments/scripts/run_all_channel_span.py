"""All native-site EEG/fNIRS masked reader training on the public paired panel."""
import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from src.data.channel_geometry import project_head_topview
from src.data.downstream_input_cache import DownstreamInputDataset
from src.data.event_alignment import window_within_alignment_support
from src.foundation.all_channel_span import NativeSpanReader, hidden_sites, masked_wave
from src.foundation.typed_span import view_codes
from run_dual_component_tokenizer import owner_lock
import run_typed_span as base

ROOT = base.ROOT
CONFIG = base.CONFIG.with_name("all_channel_span_v1.yaml")
CFG = yaml.safe_load(CONFIG.read_text())
OUT = ROOT / CFG["output"]
PARENT = ROOT / CFG["parent"]
CACHE = ROOT / CFG["scope"]["cache"]
TR = CFG["training"]
FAMILIES = ("span", "region", "whole")


def check():
    if (CFG["scope"]["splits"] != ["train", "validation"]
            or any(CFG["scope"][s] != "forbidden" for s in ("test", "protected"))
            or CFG["tensor"] != dict(sites="all_native", slots=15, seconds_per_slot=2,
                                      samples=[400, 40], vocab=base.SIZES, width=96)
            or TR["steps"] % 6 or TR["seeds"] != [17, 43]):
        raise ValueError("invalid all-channel contract")


def coordinates(rows):
    xyz, xy = [], []
    for row in rows:
        pos = np.asarray([row.get(k) for k in ("x", "y", "z")], dtype=float)
        known = (row.get("coordinate_system") == CFG["geometry"]["coordinate_system"]
                 and np.isfinite(pos).all() and np.linalg.norm(pos) > 0)
        xyz.append(pos / np.linalg.norm(pos) if known else [np.nan] * 3)
        xy.append(project_head_topview(row) if known else [np.nan] * 2)
    return np.asarray(xyz, np.float32), np.asarray(xy, np.float32)


class Panel(Dataset):
    def __init__(self, split, gains):
        if split not in CFG["scope"]["splits"] or base.read(CACHE / "verification.json")["status"] != "passed":
            raise ValueError("unverified public partition")
        self.reader = DownstreamInputDataset(CACHE, split=split)
        ledger = pd.read_csv(PARENT / "events.csv")
        self.meta = ledger[ledger.split == split].reset_index(drop=True)
        if (set(self.meta.task) != set(CFG["tasks"]) or self.meta.id.duplicated().any()
                or len(self.meta) != {"train": 2471, "validation": 706}[split]):
            raise ValueError("paired event panel changed")
        lookup = {s["id"]: i for i, s in enumerate(self.reader.samples)}
        self.index = [lookup[k] for k in self.meta.id]
        self.layouts = {}
        self.gains = gains
        settings = base.read(ROOT / base.read(PARENT / "prepared.json")["source"] / "scales.json")
        if settings["fitting_split"] != "train":
            raise ValueError("site scale provenance")
        self.scales = settings["positions"]
        for i, row in zip(self.index, self.meta.itertuples()):
            s = self.reader.samples[i]
            if (s["subject"] not in self.reader.plan["subject_partitions"][row.dataset][split]
                    or (s["dataset"], s["subject"], s["record"], s["task"]) !=
                    (row.dataset, row.subject, row.record, row.task)
                    or s["event"]["alignment_role"] != "paired_eeg_fnirs_marker"
                    or not window_within_alignment_support(s["event"], -5, 30)
                    or (s["duration_s"], s["event_relative_start_s"]) != (30, -5)
                    or not np.allclose([row.eeg_start, row.hb_start], [s["eeg_start_s"], s["hb_start_s"]], rtol=0, atol=1e-6)):
                raise ValueError("event/clock mismatch")
            rec = self.reader.records[s["join_key"]]
            path = CACHE / "records" / row.dataset / row.subject / row.record
            if rec.npz_path not in (path, Path("/SSD_2/pid-mcm-implementation") / path.relative_to(ROOT)):
                raise ValueError("unexpected cache record location")
            self.reader.records[s["join_key"]] = replace(rec, npz_path=path)
            if s["join_key"] in self.layouts:
                continue
            measurement = rec.manifest["measurement"]
            en, hn = [measurement[k + "_channel_names"] for k in ("eeg", "fnirs")]
            pairs = [n[:-4] for n in hn if n.endswith("_HbO")]
            if (len(set(en)) != len(en) or len(set(hn)) != len(hn)
                    or set(hn) != {n + "_" + c for n in pairs for c in ("HbO", "HbR")}
                    or not pairs or not en):
                raise ValueError("native layout identities")
            hi = [hn.index(n + "_" + c) for n in pairs for c in ("HbO", "HbR")]
            geo = [self.reader.geometry.for_channels(record=rec, modality=m, channel_names=names)
                   for m, names in (("eeg", en), ("fnirs", hn))]
            xyz, xy, scales, fallback = [], [], [], []
            for m, names, rows in (("eeg", en, geo[0]), ("hb", pairs, [geo[1][j] for j in hi[::2]])):
                q, p = coordinates(rows)
                xyz.append(q)
                xy.append(p)
                table = self.scales[m][row.dataset]
                median = np.median([v["scale"] for v in table.values() if np.isfinite(v["scale"]) and v["scale"] > 0])
                missing = [n for n in names if n not in table]
                scale = np.array([median if n in missing else table[n]["scale"] for n in names], np.float32)
                if not (np.isfinite(scale) & (scale > 0)).all():
                    raise ValueError("invalid frozen site scale")
                scales.append(scale)
                fallback.append(missing)
            self.layouts[s["join_key"]] = dict(names=[en, pairs], hi=hi, xyz=xyz, xy=xy, scales=scales, fallback=fallback)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i = self.index[idx]
        s = self.reader[i]
        layout = self.layouts[self.reader.samples[i]["join_key"]]
        hi = layout["hi"]
        if s["unit"] != dict(eeg="uV", fnirs=("uM" if s["dataset"] == "simultaneous_eeg_nirs" else "relative_Hb_repo_approximation")):
            raise ValueError("unexpected native unit")
        raw = [s["eeg"].reshape(-1, 15, 400),
               s["fnirs"][hi].reshape(-1, 2, 15, 20).transpose(0, 2, 3, 1).reshape(-1, 15, 40)]
        valid = [s["valid_mask"]["eeg"].reshape(raw[0].shape).all(-1),
                 s["valid_mask"]["fnirs"][hi].reshape(-1, 2, 15, 20).all((1, 3))]
        for m in range(2):
            valid[m] &= np.isfinite(raw[m]).all(-1)
            raw[m] = np.where(valid[m][..., None], raw[m], 0).astype(np.float32)
            raw[m] /= layout["scales"][m][:, None, None]
        raw[1] *= self.gains[s["task"]]
        return raw, valid, layout["xyz"], layout["xy"], idx

    def census(self):
        return {key: dict(names=v["names"], scale_fallback=v["fallback"],
                          known_geometry=[np.isfinite(g).all(-1).sum().item() for g in v["xyz"]])
                for key, v in self.layouts.items()}


def collate(samples):
    fields = []
    for field in range(4):
        modalities = []
        for m in range(2):
            values = [torch.from_numpy(s[field][m]) for s in samples]
            modalities.append(torch.nn.utils.rnn.pad_sequence(values, batch_first=True,
                              padding_value=float("nan") if field >= 2 else 0))
        fields.append(modalities)
    return fields, [s[-1] for s in samples]


def create(seed, device):
    g = CFG["geometry"]
    previous = base.load_model("joint", seed, "cpu")
    folder = PARENT / "fits" / f"joint_{seed}"
    previous.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu", weights_only=True))
    nets = previous.nets.eval().requires_grad_(False).to(device)
    reader = NativeSpanReader(base.SIZES, 96, base.P["paths"]["support_seconds"],
                              g["neighbors"], g["maximum_unit_chord"])
    missing, extra = reader.load_state_dict(previous.reader.state_dict(), strict=False)
    if set(missing) != {f"spatial.{i}.weight" for i in range(4)} or extra:
        raise ValueError("warm start did not reproduce parent architecture")
    return nets, reader.to(device), base.read(folder / "done.json")["gains"]


def forward_batch(nets, reader, fields, mod, family, gen, control=False):
    raw, valid, xyz, xy = fields
    hidden = hidden_sites(valid[mod], xyz[mod], family, mod, gen)
    with torch.no_grad():
        codes = []
        for m in range(2):
            shown = valid[m] & ~hidden if m == mod else valid[m]
            wave = masked_wave(raw[m], shown, m)
            codes.append(view_codes(nets[m], wave)[0])
        labels = view_codes(nets[mod], masked_wave(raw[mod], valid[mod], mod))[1]
    counts = hidden.sum((1, 2))
    def score(cross):
        vectors = reader.encode_target(codes, xyz, xy, valid, hidden, mod, cross)
        rows = []
        indices = hidden.nonzero(as_tuple=True)[0]
        for b, h in enumerate(vectors):
            logits = reader.output[2 * mod + b](h[hidden])
            ce = F.cross_entropy(logits, labels[b][hidden], reduction="none")
            rows.append(ce.new_zeros(len(h)).scatter_add(0, indices, ce) / counts.clamp_min(1))
        return torch.stack(rows, -1)
    return score(True), score(False) if control else None, counts


def to_device(fields, device):
    return [[v.to(device, non_blocking=True) for v in pair] for pair in fields]


def loader(panel, batches, workers):
    return DataLoader(panel, batch_sampler=batches, collate_fn=collate, num_workers=workers,
                      pin_memory=True, persistent_workers=False)


def train(seed, pilot, workers):
    device = "cuda"
    bus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    import subprocess
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=pci.bus_id,uuid", "--format=csv,noheader"], text=True)
    allowed = [s.split(", ")[1] for s in rows.splitlines() if s.split(", ")[0] != "00000000:17:00.0"]
    if bus not in allowed or torch.cuda.device_count() != 1:
        raise ValueError("expected one permitted GPU UUID")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    nets, model, gains = create(seed, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TR["lr"])
    panel = Panel("train", gains)
    folder = OUT / (f"pilot_{seed}" if pilot else f"seed_{seed}")
    folder.mkdir(exist_ok=True)
    with owner_lock(folder):
        if (folder / "summary.json").exists():
            return
        base.save(folder / "train_layouts.json", panel.census())
        steps = 4 if pilot else TR["steps"]
        rng = np.random.default_rng(seed)
        indices = []
        while len(indices) < steps * TR["batch"]:
            indices.extend(rng.permutation(len(panel)).tolist())
        batches = np.array(indices[:steps * TR["batch"]]).reshape(steps, TR["batch"]).tolist()
        start = 0
        path = folder / "checkpoint.pt"
        if path.exists():
            state = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state["reader"])
            optimizer.load_state_dict(state["optimizer"])
            start = state["step"]
        clock = time.monotonic()
        gen = torch.Generator(device=device)
        if start < steps:
            for step, (fields, ix) in enumerate(loader(panel, batches[start:], workers), start + 1):
                if ix != batches[step - 1]:
                    raise ValueError("event order drift")
                gen.manual_seed(seed * 100000 + step)
                mod, family = (step - 1) % 2, FAMILIES[((step - 1) // 2) % 3]
                loss, _, counts = forward_batch(nets, model, to_device(fields, device), mod, family, gen)
                if not (counts > 0).any():
                    raise ValueError("empty training batch")
                loss = loss[counts > 0].mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if step <= start + 4 or step % 8 == 0 or step == steps:
                    elapsed = time.monotonic() - clock
                    status = dict(step=step, steps=steps, seed=seed, loss_nats=float(loss.detach()), mode=family,
                                  target=("eeg","hb")[mod], elapsed_s=elapsed, steps_s=(step-start)/elapsed,
                                  masked_siteslots=int(counts.sum()), input_sites=[p.shape[1] for p in fields[0]],
                                  valid_sites=[[int(n) for n in p.any(-1).sum(-1)] for p in fields[1]],
                                  peak_mib=torch.cuda.max_memory_allocated()/2**20)
                    base.save(folder / "progress.json", status)
                    print(json.dumps(status), flush=True)
                if not pilot and (step == 1 or step % TR["checkpoint_interval"] == 0 or step == steps):
                    tmp = path.with_suffix(".partial")
                    torch.save(dict(reader=model.state_dict(), optimizer=optimizer.state_dict(), step=step, seed=seed), tmp)
                    tmp.replace(path)
        if pilot:
            base.save(folder / "summary.json", status)
            return
        del panel
        evaluate(nets, model.eval(), gains, seed, folder, workers)


@torch.no_grad()
def evaluate(nets, model, gains, seed, folder, workers):
    panel = Panel("validation", gains)
    base.save(folder / "development_layouts.json", panel.census())
    batches = [list(range(i, min(i + TR["batch"], len(panel)))) for i in range(0, len(panel), TR["batch"])]
    gen = torch.Generator(device="cuda")
    rows = []
    for batch, (fields, ix) in enumerate(loader(panel, batches, workers)):
        fields = to_device(fields, "cuda")
        for mod in range(2):
            for j, family in enumerate(FAMILIES):
                gen.manual_seed(617031 + batch * 6 + mod * 3 + j)
                a, b, n = forward_batch(nets, model, fields, mod, family, gen, True)
                for k, event in enumerate(ix):
                    r = panel.meta.iloc[event]
                    for branch in range(2):
                        rows.append(dict(id=r.id, dataset=r.dataset, person=r.subject, task=r.task, mod=mod,
                                         branch=branch, family=family, count=int(n[k]),
                                         bits=float(a[k, branch]) / math.log(2), control_bits=float(b[k, branch]) / math.log(2)))
    df = pd.DataFrame(rows)
    df.to_csv(folder / "development.csv.gz", index=False)
    if len(df) != 12 * len(panel):
        raise ValueError("incomplete development replay")
    people = df[df["count"] > 0].groupby(["dataset", "person", "mod", "family", "branch"])[["bits", "control_bits"]].mean()
    people.to_csv(folder / "people.csv")
    mean = people.groupby(["dataset", "person"]).mean().mean()
    base.save(folder / "summary.json", dict(status="completed", steps=TR["steps"], seed=seed, development_events=len(panel),
              bits=mean.bits, no_cross_bits=mean.control_bits, no_cross_minus_full_bits=mean.control_bits-mean.bits,
              missing_event_branch_masks=int((df["count"] == 0).sum())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["pilot", "train"])
    parser.add_argument("--seed", type=int, required=True, choices=TR["seeds"])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    check()
    train(args.seed, args.stage == "pilot", args.workers)
