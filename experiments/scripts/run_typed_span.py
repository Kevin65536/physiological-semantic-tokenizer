"""Public, subject-paired test of joint typed-lag span pretraining."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from src.analysis.dual_component_tokenizer import eeg_spectral_attributes
from src.foundation.aligned_masked import FrozenCodeReader
from src.foundation.typed_span import TypedSpanReader, observed_view, span_mask, view_codes

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/configs/physiology_semantic_tokenizer/typed_span_v1.yaml"
P = yaml.safe_load(CONFIG.read_text())
TR = P["training"]
OUT = ROOT / P["run"]
SIZES = P["shape"]["vocab"]
BRANCHES = ["EEG-S", "EEG-O", "Hb-S", "Hb-O"]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def validate():
    if (
        P["scope"]
        != dict(
            dataset="simultaneous_eeg_nirs",
            record="cnt_nback",
            train=15,
            validation=4,
            test="forbidden",
            protected="forbidden",
        )
        or P["shape"]
        != dict(
            sites=4,
            slots=15,
            seconds_per_slot=2,
            per_slot=[400, 40],
            vocab=[8192, 8192, 1024, 1024],
            dimension=128,
        )
        or P["mask"]["span_slots_eeg_hb"] != [3, 6]
        or P["mask"]["whole_fraction"] != 0.25
        or TR["steps"] % 24
        or TR["calibrate_steps"] % 24
        or TR["seeds"] != [17, 43]
        or P["arms"] != ["joint", "no_lag", "ss_only", "no_anchor", "frozen"]
    ):
        raise ValueError("invalid joint span protocol")


def assignment(meta):
    """Same phase relative to the event marker; bijective within split and condition."""
    donor = np.full(len(meta), -1)
    for _, idx in meta.groupby(["split", "condition"]).indices.items():
        who = meta.subject.to_numpy()[idx]
        rng = np.random.default_rng(717)
        cost = rng.random((len(idx), len(idx))) + (who[:, None] == who) * (len(idx) + 1)
        a, b = linear_sum_assignment(cost)
        if np.any(who[a] == who[b]):
            raise ValueError("no cross-record condition-matched permutation")
        donor[idx[a]] = idx[b]
    if sorted(donor) != list(range(len(meta))):
        raise ValueError("incomplete donor ledger")
    return donor


def task(step):
    mod = step % 2
    length = 15 if step // 6 % 4 == 3 else P["mask"]["span_slots_eeg_hb"][mod]
    visible = ((0,), (1,), (0, 1))[step // 2 % 3]
    return mod, length, visible


def panel(device):
    import run_frozen_science_screen as screen

    anchor = ROOT / P["anchors"]
    ev = ROOT / P["events"]
    if (
        ev != screen.OUT
        or read(ev / "summary.json")["status"] != "completed"
        or read(anchor / "prepared.json")["status"] != "completed"
    ):
        raise ValueError("incomplete public parent")
    m, train, dev = screen.panel()
    crops = pd.read_csv(anchor / "crops.csv")
    if (
        (len(train), len(dev)) != (309, 91)
        or m.id.duplicated().any()
        or set(m.dataset) != {P["scope"]["dataset"]}
        or set(m.record) != {"cnt_nback"}
    ):
        raise ValueError("unexpected event identities")
    for frame in (m, crops):
        if frame.groupby("subject").split.nunique().max() != 1 or frame.groupby(
            "split"
        ).subject.nunique().to_dict() != {"train": 15, "validation": 4}:
            raise ValueError("split boundary")
    if set(zip(m.subject, m.split)) != set(zip(crops.subject, crops.split)):
        raise ValueError("event/anchor roster drift")
    if not all(
        np.allclose(m[k + "_stop"] - m[k + "_start"], 30)
        and np.all((m[k + "_stop"] - m[k + "_marker"]).between(3.89999, 4.00001))
        for k in ("hb", "eeg")
    ):
        raise ValueError("event phase mismatch")

    def tensor(x):
        return torch.as_tensor(np.array(x), device=device, dtype=torch.float32)

    hb_info = read(anchor / "hb_replication/prepared.json")
    if hb_info["status"] != "completed" or hb_info["crops"] != len(crops):
        raise ValueError("incomplete Hb anchors")
    refs = [read(anchor / "scales.json"), hb_info["scales"]]
    eeg_x, eeg_s = [], []
    ledger = []
    for person, group in crops.groupby("subject", sort=False):
        if person not in set(m.subject):
            raise ValueError("unexpected person")
        frame = pd.read_csv(anchor / "data" / (person + ".csv"))
        pd.testing.assert_frame_equal(frame, group[frame.columns].reset_index(drop=True))
        with np.load(anchor / "data" / (person + ".npz")) as z:
            eeg_x.append(z["x"])
            eeg_s.append(z["s"])
        ledger.extend(group.index)
    if ledger != list(range(len(crops))):
        raise ValueError("Hb anchor row order")
    with np.load(anchor / "hb_replication/data.npz") as z:
        anc = [
            (tensor(np.concatenate(eeg_x)), tensor(np.concatenate(eeg_s))),
            (tensor(z["x"]), tensor(z["s"])),
        ]
    raw = [tensor(screen.array("wave")), tensor(screen.array("hb")).reshape(len(m), 4, 15, 40)]
    xy = tensor(screen.array("xy"))
    coord = read(screen.BASE / "strict_common/coordinates.json")
    scale = tensor([r["scale"] for r in refs[0]])
    if coord["site_scales"] != [r["scale"] for r in refs[1]] or xy.shape != (len(m), 2, 4, 2):
        raise ValueError("site coordinates changed")
    eligible = np.flatnonzero(crops.donor.to_numpy() >= 0)
    if crops.iloc[eligible].groupby("split").size().to_dict() != {"train": 463, "validation": 124}:
        raise ValueError("teacher support")
    for mod, width in enumerate((400, 40)):
        if any(
            tuple(x.shape) != (n, 4, 15, width) or not torch.isfinite(x).all()
            for x, n in [(raw[mod], len(m)), *((x, len(crops)) for x in anc[mod])]
        ):
            raise ValueError("wave shape/support")
    variances = [
        [
            tensor([[r[field][b] for r in rs] for b in "xso"])
            for field in ("target_variance", "difference_variance")
        ]
        for rs in refs
    ]
    return dict(
        m=m,
        crops=crops,
        eligible=eligible,
        train=train,
        dev=dev,
        raw=raw,
        xy=xy,
        anc=anc,
        var=variances,
        donor=assignment(m),
        scale=scale,
        gain=tensor(coord["gain"] / np.array(coord["site_scales"])),
    )


def features(wave, mod, scale):
    if mod == 0:
        return eeg_spectral_attributes(wave.flatten(0, 1), scale.repeat(len(wave))).reshape(
            *wave.shape[:3], 4
        )
    shape = wave.reshape(*wave.shape[:3], 20, 2)
    return (shape - shape.mean(dim=-2, keepdim=True)).flatten(-2)


def moments(data):
    import retrain_single_vq_aligned as aligned

    ix = data["eligible"][data["crops"].split.to_numpy()[data["eligible"]] == "train"]
    stats = []
    for mod, (x, s) in enumerate(data["anc"]):
        for j, target in enumerate((s[ix], x[ix] - s[ix])):
            ft = features(target, mod, data["scale"])
            mean, sd = ft.mean((0, 1, 2)), ft.std((0, 1, 2), unbiased=False).clamp_min(1e-6)
            if mod == j == 0:
                orig = aligned.objective()["calibration"][P["scope"]["dataset"]]
                mean, sd = [s.new_tensor(orig[k]) for k in ("mean", "sd")]
            stats.append((mean, sd))
    return stats


class Model(nn.Module):
    def __init__(self, arm, nets):
        super().__init__()
        self.arm = arm
        self.nets = nn.ModuleList(nets)
        width = TR["reader_width"]
        self.local = nn.ModuleList([nn.Identity(), nn.Linear(128, 4), nn.Linear(128, 40), nn.Linear(128, 40)])
        if arm == "frozen":
            books = [net.branches[b]["quantizer"].codebook for net in nets for b in "SO"]
            self.reader = FrozenCodeReader(
                books, range(4), dict(width=width, heads=4, feedforward=2 * width, layers=1, dropout=0.0)
            )
        else:
            self.reader = TypedSpanReader(SIZES, 128, width, P["paths"]["support_seconds"])

    def predict(self, source, target, xy, mask, mod, visible):
        if self.arm != "frozen":
            return self.reader(source[0], target[0], xy, mask, mod, visible, self.arm)
        ordered = (source, target) if mod else (target, source)
        ids = torch.stack([v for part in ordered for v in part[1]], 1)
        hide = torch.zeros_like(ids, dtype=torch.bool)
        hide[:, 2 * mod : 2 * mod + 2] = mask[:, None]
        for j in set((0, 1)).difference(visible):
            hide[:, 2 * (1 - mod) + j] = True
        z = self.reader.encode(ids, xy, None, stream_hidden=hide)
        return [self.reader.output[j](z[:, j]) for j in range(2 * mod, 2 * mod + 2)]


def codes(net, data, mod, idx, mask):
    wave = observed_view(data["raw"][mod][idx], mask, mod, data["gain"])
    return view_codes(net, wave)


def anchor_loss(model, data, ix, stats, update):
    costs, nmse = [], []
    for mod, net in enumerate(model.nets):
        # Only this complete G0 view updates EMA; assignment views remain read-only.
        net.train(update)
        x, s = (v[ix] for v in data["anc"][mod])
        out = net(x.flatten(0, 1), return_codes=True)
        net.eval()
        wave = out["commitment"]
        local = []
        point = []
        ch = 1 if mod == 0 else 2
        for j, (key, tar) in enumerate(zip(("sum", "S", "O"), (x, s, x - s))):
            err = (out[key].reshape_as(x) - tar).reshape(len(x), 4, -1, ch)
            v, dv = (mat[j][None, :, None] for mat in data["var"][mod])
            nm = (err.square() / v).mean((1, 2, 3))
            point.append(nm.detach())
            if j == 0 or model.arm != "no_anchor":
                wave = wave + 0.5 * nm.mean() + (0.1 / 3) * (err.diff(dim=2).square() / dv).mean()
            if j and model.arm not in ("no_anchor", "frozen"):
                branch = 2 * mod + j - 1
                mu, sd = stats[branch]
                measured = (features(tar, mod, data["scale"]) - mu) / sd
                head = net.spectral_head if branch == 0 else model.local[branch]
                pred = head(out["codes"][key]).reshape_as(measured)
                local.append(F.mse_loss(pred, measured))
        costs.append(wave + (TR["local_weight"] * torch.stack(local).mean() if local else 0.0))
        nmse.append(torch.stack(point, 1))
    return torch.stack(costs).mean(), torch.cat(nmse, 1)


def load_model(arm, seed, device):
    import audit_aligned_tokenizer as audit

    if audit.PARENT != ROOT / P["checkpoint"]:
        raise ValueError("checkpoint contract")
    nets = [copy.deepcopy(audit.network(mod, size, 17)).eval() for mod, size in (("eeg", 8192), ("hb", 1024))]
    torch.manual_seed(seed)
    return Model(arm, nets).to(device)


def train(model, data, seed, stats, folder, pilot):
    rng = np.random.default_rng(seed)
    groups = [
        (g.index.to_numpy(), data["eligible"][data["crops"].subject.to_numpy()[data["eligible"]] == person])
        for person, g in data["m"].iloc[data["train"]].groupby("subject")
    ]
    params = [
        {"params": list(model.reader.parameters()) + list(model.local.parameters()), "lr": TR["reader_lr"]},
        {"params": model.nets.parameters(), "lr": TR["tokenizer_lr"]},
    ]
    opt = torch.optim.AdamW(params)
    start = time.monotonic()
    steps = 6 if pilot else TR["steps"]
    for i in range(steps):
        idx, anc = zip(
            *[
                (rng.choice(groups[k][0]), rng.choice(groups[k][1]))
                for k in rng.integers(len(groups), size=TR["batch"])
            ]
        )
        ix = np.array(idx)
        mod, length, visible = task(i)
        update = i >= (2 if pilot else TR["calibrate_steps"]) and model.arm != "frozen"
        model.nets.requires_grad_(update)
        mask = span_mask(
            torch.tensor(rng.integers(16 - length, size=len(ix)), device=data["xy"].device), length, 4
        )
        empty = torch.zeros_like(mask)
        src = codes(model.nets[1 - mod], data, 1 - mod, ix, empty)
        tgt = codes(model.nets[mod], data, mod, ix, mask)
        with torch.no_grad():
            _, labels = codes(model.nets[mod], data, mod, ix, empty)
        logits = model.predict(src, tgt, data["xy"][ix], mask, mod, visible)
        span = torch.stack([F.cross_entropy(z[mask], y[mask]) for z, y in zip(logits, labels)]).mean()
        cost = span * TR["span_weight"]
        if model.arm != "frozen":
            auxiliary, _ = anchor_loss(model, data, np.array(anc), stats, update)
            cost = cost + auxiliary
        opt.zero_grad(set_to_none=True)
        cost.backward()
        nn.utils.clip_grad_norm_(model.parameters(), TR["clip"], error_if_nonfinite=True)
        opt.step()
        if (i + 1) % 24 == 0 or i + 1 == steps:
            print(f"{model.arm} {seed} step {i+1}/{steps}, loss {cost.item():.5f}", flush=True)
    return dict(
        steps=steps,
        train_seconds=time.monotonic() - start,
        peak_gpu_MiB=torch.cuda.max_memory_reserved() / 2**20,
    )


@torch.no_grad()
def evaluate(model, reference, data, stats, folder):
    model.eval()
    m = data["m"]
    rows, guard, usage = [], [], {}
    for ix in np.array_split(data["dev"], math.ceil(len(data["dev"]) / 8)):
        for mod in (0, 1):
            raw = data["raw"][mod][ix]
            empty = torch.zeros(raw.shape[:3], dtype=torch.bool, device=raw.device)
            true = codes(model.nets[1 - mod], data, 1 - mod, ix, empty)
            wrong = codes(model.nets[1 - mod], data, 1 - mod, data["donor"][ix], empty)
            reverse = [[v.flip(2) for v in part] for part in true]
            _, labels = codes(model.nets[mod], data, mod, ix, empty)
            _, fixed = codes(reference[mod], data, mod, ix, empty)
            for j, y in enumerate(labels):
                for idx, sequence in zip(ix, y.cpu().numpy()):
                    key = (m.subject.iloc[idx], BRANCHES[2 * mod + j])
                    usage.setdefault(key, np.zeros(SIZES[2 * mod + j], int))
                    usage[key] += np.bincount(sequence.ravel(), minlength=SIZES[2 * mod + j])
            for length in [P["mask"]["span_slots_eeg_hb"][mod], 15]:
                # Same centrally scheduled mask in every seed, arm and source control.
                mask = score_mask(ix, mod, length, raw.device)
                starts = mask[:, 0].long().argmax(1).cpu().tolist()
                tgt = codes(model.nets[mod], data, mod, ix, mask)
                for ctrl, source, visible in [
                    ("true", true, (0, 1)),
                    ("wrong", wrong, (0, 1)),
                    ("reverse", reverse, (0, 1)),
                    ("none", true, ()),
                    ("S", true, (0,)),
                    ("O", true, (1,)),
                ]:
                    logits = model.predict(source, tgt, data["xy"][ix], mask, mod, visible)
                    for j, z in enumerate(logits):

                        def bits(y):
                            return F.cross_entropy(z[mask], y[mask], reduction="none").reshape(
                                len(ix), -1
                            ).mean(1).cpu().numpy() / math.log(2)

                        values = bits(labels[j])
                        fixed_bits = bits(fixed[j]) if ctrl == "true" else [None] * len(ix)
                        for idx, b, fb, lo in zip(ix, values, fixed_bits, starts):
                            rows.append(
                                dict(
                                    event=m.id.iloc[idx],
                                    subject=m.subject.iloc[idx],
                                    donor=m.id.iloc[data["donor"][idx]],
                                    branch=BRANCHES[2 * mod + j],
                                    slots=length,
                                    start=lo,
                                    control=ctrl,
                                    bits=b,
                                    parent_bits=fb,
                                )
                            )
    kept = data["eligible"][data["crops"].split.to_numpy()[data["eligible"]] == "validation"]
    for ix in np.array_split(kept, math.ceil(len(kept) / 8)):
        _, nmse = anchor_loss(model, data, ix, stats, False)
        for i, values in zip(ix, nmse.cpu().tolist()):
            guard.append(
                dict(
                    crop=i,
                    subject=data["crops"].subject.iloc[i],
                    **dict(zip(["EEG-X", "EEG-S", "EEG-O", "Hb-X", "Hb-S", "Hb-O"], values)),
                )
            )
    counts = []
    for (person, branch), n in usage.items():
        p = n[n > 0] / n.sum()
        counts.append(
            dict(
                subject=person,
                branch=branch,
                used=len(p),
                tokens=n.sum(),
                entropy_bits=-(p * np.log2(p)).sum(),
            )
        )
    for name, records in [("masked", rows), ("anchors", guard), ("usage", counts)]:
        pd.DataFrame(records).to_csv(folder / (name + ".csv"), index=False)


@torch.inference_mode()
def decode_screen(model, data, folder):
    """Hard completion at hidden slots. Visible slots use only the corrupted-view IDs."""
    rows, examples = [], {}
    dev = data["dev"]
    # Common scale fixed by training events, separately for each site and color.
    scales = []
    for mod in (0, 1):
        raw = data["raw"][mod][data["train"]]
        mask = torch.zeros(raw.shape[:3], dtype=torch.bool, device=raw.device)
        colors = 1 if mod == 0 else 2
        complete = observed_view(raw, mask, mod, data["gain"]).reshape(len(raw), 4, -1, colors)
        scales.append(complete.square().mean((0, 2)).clamp_min(1e-12))
    example_ids = set(data["m"].iloc[dev].groupby("subject", sort=False).head(1).index)
    for ix in np.array_split(dev, math.ceil(len(dev) / 8)):
        for mod in (0, 1):
            raw = data["raw"][mod][ix]
            empty = torch.zeros(raw.shape[:3], dtype=torch.bool, device=raw.device)
            obs = observed_view(raw, empty, mod, data["gain"])
            true = codes(model.nets[1 - mod], data, 1 - mod, ix, empty)
            wrong = codes(model.nets[1 - mod], data, 1 - mod, data["donor"][ix], empty)
            colors = 1 if mod == 0 else 2
            for length in (P["mask"]["span_slots_eeg_hb"][mod], 15):
                mask = score_mask(ix, mod, length, raw.device)
                target = codes(model.nets[mod], data, mod, ix, mask)
                for ctrl, src in (("true", true), ("wrong", wrong)):
                    logits = model.predict(src, target, data["xy"][ix], mask, mod, (0, 1))
                    ids = {
                        b: torch.where(mask, z.argmax(-1), t).flatten(0, 1)
                        for b, z, t in zip("SO", logits, target[1])
                    }
                    decoded = model.nets[mod].decode_ids(ids)
                    prediction = decoded["sum"].reshape_as(obs)
                    error = (prediction - obs).reshape(len(ix), 4, 15, -1, colors)
                    error = error.square() / scales[mod][None, :, None, None, :]
                    score = error.mean((-1, -2))[mask].reshape(len(ix), -1).mean(-1)
                    for n, (idx, value) in enumerate(zip(ix, score.cpu().tolist())):
                        row = data["m"].iloc[idx]
                        rows.append(
                            dict(
                                event=row.id,
                                subject=row.subject,
                                modality=("EEG", "Hb")[mod],
                                slots=length,
                                control=ctrl,
                                nmse=value,
                            )
                        )
                        if model.arm == "joint" and ctrl == "true" and idx in example_ids and length < 15:
                            prefix = f"{idx}_{mod}_"
                            examples[prefix + "event"] = np.asarray(row.id)
                            examples[prefix + "mask"] = mask[n].cpu().numpy()
                            examples[prefix + "observed"] = obs[n].cpu().numpy()
                            for j, b in enumerate("SO"):
                                examples[prefix + "source_id_" + b] = src[1][j][n].cpu().numpy()
                                examples[prefix + "completed_id_" + b] = (
                                    ids[b].reshape_as(mask)[n].cpu().numpy()
                                )
                                examples[prefix + "decoded_" + b] = (
                                    decoded[b].reshape_as(obs)[n].cpu().numpy()
                                )
    pd.DataFrame(rows).to_csv(folder / "completion_wave.csv", index=False)
    if examples:
        np.savez_compressed(folder / "completion_examples.npz", **examples)


def score_mask(ix, mod, length, device):
    starts = [np.random.default_rng(7901 + int(i) * 2 + mod).integers(16 - length) for i in ix]
    return span_mask(torch.as_tensor(starts, device=device), length, 4)


def run(arm, seed, pilot, replay=False):
    from run_dual_component_tokenizer import owner_lock

    validate()
    if arm not in P["arms"] or seed not in TR["seeds"]:
        raise ValueError("unregistered cell")
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_gpu.startswith("GPU-") or "," in visible_gpu:
        raise ValueError("one preselected GPU UUID required")
    import subprocess

    cards = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"], text=True
    )
    if visible_gpu not in [
        line.split(", ")[0] for line in cards.splitlines() if line.split(", ")[1] != "00000000:17:00.0"
    ]:
        raise ValueError("excluded GPU")
    folder = OUT / ("pilot" if pilot else "fits") / f"{arm}_{seed}"
    with owner_lock(OUT, "contract", blocking=True):
        resolved = OUT / "resolved.yaml"
        if resolved.exists() and yaml.safe_load(resolved.read_text()) != P:
            raise ValueError("existing run contract differs")
        if not resolved.exists():
            resolved.write_text(CONFIG.read_text())
    with owner_lock(folder):
        if replay:
            done = read(folder / "done.json")
            if done["steps"] != TR["steps"] or done["status"] != "completed":
                raise ValueError("terminal checkpoint required")
            model = load_model(arm, seed, "cuda").eval()
            model.load_state_dict(torch.load(folder / "weights.pt", weights_only=True, map_location="cpu"))
            decode_screen(model, panel("cuda"), folder)
            return
        if (folder / "done.json").is_file():
            return
        data = panel("cuda")
        model = load_model(arm, seed, "cuda")
        stats = moments(data)
        fit = train(model, data, seed, stats, folder, pilot)
        if not pilot:
            ref = load_model("joint", seed, "cuda").nets.eval()
            evaluate(model, ref, data, stats, folder)
            tmp = folder / "weights.partial"
            torch.save(model.state_dict(), tmp)
            tmp.replace(folder / "weights.pt")
        save(
            folder / "done.json",
            dict(status="completed", seed=seed, arm=arm, pilot=pilot, gpu=visible_gpu, **fit),
        )


def summarize():
    tables = {k: [] for k in ("masked", "anchors", "usage", "completion_wave")}
    for arm in P["arms"]:
        for seed in TR["seeds"]:
            folder = OUT / "fits" / f"{arm}_{seed}"
            done = read(folder / "done.json")
            if (done["arm"], done["seed"], done["steps"], done["status"]) != (
                arm,
                seed,
                TR["steps"],
                "completed",
            ) or not (folder / "weights.pt").is_file():
                raise ValueError("incomplete fit")
            for key in tables:
                df = pd.read_csv(folder / (key + ".csv"))
                expected = {
                    "masked": 91 * 4 * 2 * 6,
                    "anchors": 124,
                    "usage": 16,
                    "completion_wave": 91 * 2 * 2 * 2,
                }[key]
                if len(df) != expected or df.subject.nunique() != 4:
                    raise ValueError("incomplete evaluation")
                identities = {
                    "masked": ["event", "branch", "slots", "control"],
                    "completion_wave": ["event", "modality", "slots", "control"],
                    "anchors": ["crop"],
                    "usage": ["subject", "branch"],
                }[key]
                if df.duplicated(identities).any():
                    raise ValueError("duplicate evaluation units")
                numeric = df.select_dtypes("number").drop(columns="parent_bits", errors="ignore")
                if not np.isfinite(numeric.to_numpy()).all():
                    raise ValueError("nonfinite evaluation")
                if key == "masked":
                    actual = df.control == "true"
                    if not np.isfinite(df.loc[actual, "parent_bits"]).all():
                        raise ValueError("missing fixed-vocabulary score")
                    if arm == "ss_only":
                        residual = df[df.branch.isin(["EEG-O", "Hb-O"])]
                        if residual.groupby(["event", "branch", "slots"]).bits.nunique().max() != 1:
                            raise ValueError("source affected disconnected O target")
                tables[key].append(df.assign(arm=arm, seed=seed))
    wave = (
        pd.concat(tables["completion_wave"])
        .groupby(["arm", "seed", "subject", "modality", "slots", "control"])
        .nmse.mean()
        .unstack()
    )
    wave["wrong_minus_true"] = wave.wrong - wave.true
    wave.to_csv(OUT / "completion_wave_by_person.csv")
    mask = pd.concat(tables["masked"])
    unit = ["arm", "seed", "subject", "branch", "slots"]
    person = mask.groupby(unit + ["control"]).bits.mean().unstack()
    for c in ("wrong", "reverse", "none", "S", "O"):
        person[c + "_minus_true"] = person[c] - person.true
    person["parent_bits"] = mask[mask.control == "true"].groupby(unit).parent_bits.mean()
    person.reset_index().to_csv(OUT / "masked_by_person.csv", index=False)
    mean = person.groupby(["arm", "subject"]).mean()
    mean.to_csv(OUT / "paired_people.csv")
    anchor = (
        pd.concat(tables["anchors"])
        .drop(columns="crop")
        .groupby(["arm", "seed", "subject"])
        .mean()
        .groupby("arm")
        .mean()
    )
    anchor.to_csv(OUT / "anchor_nmse.csv")
    pd.concat(tables["usage"]).groupby(["arm", "branch"])[["used", "entropy_bits"]].mean().to_csv(
        OUT / "code_usage.csv"
    )
    save(
        OUT / "summary.json",
        dict(
            status="completed",
            fits=len(tables["masked"]),
            seeds=TR["seeds"],
            train_events=309,
            dev_events=91,
            train_people=15,
            dev_people=4,
            train_anchors=463,
            dev_anchors=124,
            masked_bits=mean.groupby("arm").mean().T.to_dict(),
            anchor_nmse=anchor.T.to_dict(),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["pilot", "run", "decode", "summarize"])
    parser.add_argument("--arm", choices=P["arms"], default="joint")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    validate()
    if args.stage == "summarize":
        summarize()
    else:
        run(args.arm, args.seed, args.stage == "pilot", args.stage == "decode")
