"""Two-seed formal screening of all independent tokenizer interventions."""

import argparse
from collections import defaultdict
import gc
import os
import time

import numpy as np
import pandas as pd
import torch

import run_frozen_science_screen as screen
from prepare_science_ablations import CFG, OUT, check
from run_dual_component_tokenizer import owner_lock
from src.tokenizers.dual_component_tokenizer import build_base_tokenizer
from src.tokenizers.science_ablations import ScienceTokenizer, initialize, objective
from src.utils.io import save_npz_atomic

ARMS = list(dict.fromkeys([*CFG["E2"], *CFG["E3"], *CFG["E5"], *CFG["E6"]]))
TR = CFG["training"]
read = screen.parent.read
save = screen.parent.save


def arrays():
    if read(OUT / "prepared.json")["status"] != "completed":
        raise ValueError("missing preparation")
    meta = pd.read_csv(OUT / "crops.csv")
    parts = defaultdict(list)
    for person, g in meta.groupby("subject", sort=False):
        frame = pd.read_csv(OUT / "data" / (person + ".csv"))
        np.testing.assert_array_equal(frame.start, g.start)
        with np.load(OUT / "data" / (person + ".npz")) as z:
            for key in z.files:
                parts[key].append(z[key])
    data = {k: np.concatenate(v) for k, v in parts.items()}
    # Transfer once and index small read-only arrays on the selected device.
    return meta, data


def references():
    ref = read(OUT / "scales.json")
    cal = screen.audit.aligned.objective()["calibration"][screen.parent.DS]
    return (
        np.array([[v["target_variance"][key][0] for v in ref] for key in "xso"], dtype="float32")[
            :, :, None, None
        ],
        np.array([[v["difference_variance"][key][0] for v in ref] for key in "xso"], dtype="float32")[
            :, :, None, None
        ],
        np.array([v["scale"] for v in ref], dtype="float32"),
        np.array([cal["mean"], cal["sd"]], dtype="float32"),
    )


def controls(meta, data):
    tr = np.flatnonzero((meta.split == "train") & (meta.donor >= 0))
    # Fit one pooled two-channel least squares coefficient per site on training crops, with crop intercept removed.
    xx = np.zeros((2, 2))
    xy = np.zeros((2, 4))
    for i in tr:
        e = data["eog"][i].reshape(2, -1).astype(float)
        e -= e.mean(1, keepdims=True)
        s = data["s"][i].reshape(4, -1).astype(float)
        s -= s.mean(1, keepdims=True)
        xx += e @ e.T
        xy += e @ s.T
    beta = np.linalg.solve(xx + np.eye(2) * max(xx.trace() * 1e-9, 1e-9), xy).T
    power = np.zeros(3001)
    target = np.zeros(2)
    for i in tr:
        x = data["x"][i].reshape(4, 6000).astype(float)
        s = data["s"][i].reshape(4, 6000).astype(float)
        power += np.mean(np.abs(np.fft.rfft(x)) ** 2, axis=0)
        target += [np.mean(s * s), np.mean(np.diff(s, axis=1) ** 2)]
    hz = np.fft.rfftfreq(6000, 1 / 200)
    p = power.copy()
    p[1:-1] *= 2
    candidates = []
    for f in (2, 4, 8, 12, 20, 35):
        q = p * (hz <= f) / 6000**2
        gain = np.sqrt(target[0] / max(q.sum(), 1e-12))
        score = abs(np.log(max((q * 4 * np.sin(np.pi * hz / 200) ** 2).sum() * gain**2, 1e-12) / target[1]))
        candidates.append((float(score), f, float(gain)))
    _, cutoff, gain = min(candidates)
    return dict(beta=beta.tolist(), cutoff=cutoff, gain=gain, training_crops=len(tr))


def target_for(arm, data, ix, donor, control):
    x = data["x"][ix]
    s = data["s"][ix]
    if arm == "generic":
        flat = x.flatten(2)
        f = torch.fft.rfftfreq(flat.shape[-1], 1 / 200, device=x.device)
        s = (
            torch.fft.irfft(torch.fft.rfft(flat) * (f <= control["cutoff"]), n=6000).reshape_as(x)
            * control["gain"]
        )
    elif arm == "sham_teacher":
        s = data["s"][donor[ix]]
    elif arm in ("eog_true", "eog_wrong"):
        eye = data["eog"][ix if arm == "eog_true" else donor[ix]]
        eye = eye - eye.mean((2, 3), keepdim=True)
        beta = x.new_tensor(control["beta"])
        s = s - torch.einsum("sc,bctf->bstf", beta, eye)
    return s


def create(arm, seed, device):
    if arm not in ARMS or seed not in TR["seeds"]:
        raise ValueError("unregistered cell")
    torch.manual_seed(seed)
    cfg, task = screen.audit.aligned.settings("eeg")
    base = build_base_tokenizer(cfg, task, single_id_spectrum=True)
    return ScienceTokenizer(base, arm, CFG["E3"].get(arm, [8192])).to(device)


def schedule(meta, seed):
    train = np.flatnonzero((meta.split == "train") & (meta.donor >= 0))
    groups = [g.index.to_numpy() for _, g in meta.iloc[train].groupby("subject")]
    if len(groups) != 15 or set(meta.iloc[meta.donor.iloc[train]].split) != {"train"}:
        raise ValueError("training boundary")
    batch = TR["positions_per_batch"] // 4
    rng = np.random.default_rng(seed)

    def draw():
        return np.array([rng.choice(groups[j]) for j in rng.integers(len(groups), size=batch)])

    order = np.stack([draw() for _ in range(TR["warmup"] + TR["hard"])])
    init = np.concatenate([draw() for _ in range(18)])  # 288 crops x 4 sites x 15 slots > 16384
    return order, init


def batch_h(arm, data, ix, donor):
    return data["h"][donor[ix] if arm == "cross_wrong" else ix]


@torch.no_grad()
def init_books(model, data, init, donor):
    model.eval()
    pieces = defaultdict(list)
    for ix in init.split(12):
        for b, z in model.encode(data["x"][ix], batch_h(model.arm, data, ix, donor)).items():
            pieces[b].append(z)
    for b, parts in pieces.items():
        z = torch.cat(parts)
        for book in model.books(b):
            initialize(book, z)
            lookup = torch.cat(
                [
                    book.get_embedding(book._assignment_logits(q, book.codebook).argmax(1))
                    for q in z.reshape(-1, z.shape[-1]).split(256)
                ]
            ).reshape_as(z)
            z = z - lookup
    model.train()


def atomic_torch(path, obj):
    tmp = path.with_suffix(".partial")
    torch.save(obj, tmp)
    tmp.replace(path)


def training(
    arm,
    seed,
    device,
    meta,
    data,
    control,
    folder,
    pilot=False,
    *,
    factory=create,
    moments=references,
    targets=target_for,
    loss_function=objective,
    protocol=CFG,
    default_attr=0.01,
):
    cfg = TR
    model = factory(arm, seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.0001)
    order, init = schedule(meta, seed)
    donor = torch.as_tensor(meta.donor.values, device=device)
    var, dvar, scale, cal = [torch.as_tensor(x, device=device) for x in moments()]
    weights = CFG["E2"].get(arm, [1, 1])
    attr = {"attr_full": 0.01, "attr_half": 0.005, "attr_off": 0.0}.get(arm, default_attr)
    warm = cfg["warmup"] if not pilot else 2
    end = len(order) if not pilot else 8
    path = folder / "state.pt"
    start = 0
    history = []
    elapsed = 0.0
    if path.exists() and not pilot:
        state = torch.load(path, map_location=device, weights_only=False)
        if (state["arm"], state["seed"], state["contract"]) != (arm, seed, protocol):
            raise ValueError("resume identity mismatch")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
        start = state["step"]
        elapsed = state["seconds"]
        history = state["history"]
        del state
    began = time.monotonic()
    for step in range(start, end):
        if step == warm:
            init_books(model, data, torch.as_tensor(init, device=device), donor)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.0001)
        ix = torch.as_tensor(order[step], device=device)
        x = data["x"][ix]
        h = batch_h(arm, data, ix, donor)
        s = targets(arm, data, ix, donor, control)
        output = model(x, h, step >= warm)
        loss = loss_function(output, x, s, var, dvar, scale, cal, weights, attr if step >= warm else 0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
        opt.step()
        if (step + 1) % cfg["check_every"] == 0 or step + 1 == end:
            seconds = elapsed + time.monotonic() - began
            history.append(dict(step=step + 1, loss=float(loss.detach()), seconds=seconds))
            if not pilot:
                atomic_torch(
                    path,
                    dict(
                        contract=protocol,
                        arm=arm,
                        seed=seed,
                        step=step + 1,
                        model=model.state_dict(),
                        optimizer=opt.state_dict(),
                        rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(device),
                        seconds=seconds,
                        history=history,
                    ),
                )
            save(folder / "progress.json", dict(step=step + 1, total=end, seconds=seconds, history=history))
    model.eval()
    return model


@torch.inference_mode()
def extract(model, x, h, device, metrics=False):
    results = defaultdict(list)
    for i in range(0, len(x), 8):
        xb = torch.as_tensor(x[i : i + 8], device=device)
        hb = torch.as_tensor(h[i : i + 8], device=device)
        out = model(xb, hb)
        looked = model.lookup(out["ids"])
        for b in "SO":
            torch.testing.assert_close(looked[b], out["vectors"][b], atol=0, rtol=0)
        for level in range(len(model.books("S"))):
            parts = torch.stack([out["parts"][b][level] for b in "SO"], 1)
            results["level" + str(level)].append(parts.reshape(-1, 4, 2, 15, 128).cpu().numpy())
            ids = torch.stack([out["ids"][b][level] for b in "SO"], 1)
            results["ids" + str(level)].append(ids.reshape(-1, 4, 2, 15).cpu().numpy().astype("uint16"))
        if metrics:
            for b in "SO":
                results[b].append(out[b].reshape_as(xb).cpu().numpy())
            if len(model.books("S")) == 2:
                partial = model.decode({b: out["parts"][b][0] for b in "SO"})
                results["first_wave"].append((partial["S"] + partial["O"]).reshape_as(xb).cpu().numpy())
    return {k: np.concatenate(v) for k, v in results.items()}


def diagnostics(model, meta, data, control, device, folder):
    dev = np.flatnonzero((meta.split == "validation") & (meta.donor >= 0))
    donor = torch.as_tensor(meta.donor.values, device=device)
    ix = torch.as_tensor(dev, device=device)
    h = batch_h(model.arm, data, ix, donor)
    z = extract(model, data["x"][ix], h, device, True)
    s = target_for(model.arm, data, ix, donor, control).cpu().numpy()
    x = data["x"][ix].cpu().numpy()
    o = x - s
    ps, po = z["S"], z["O"]
    eye = data["eog"][ix].cpu().numpy().reshape(len(ix), 2, -1)
    eye -= eye.mean(-1, keepdims=True)

    def corr(v):
        v = v.reshape(len(ix), 4, -1)
        v = v - v.mean(-1, keepdims=True)
        return np.abs(
            np.einsum("bsf,bcf->bsc", v, eye)
            / (np.linalg.norm(v, axis=2)[:, :, None] * np.linalg.norm(eye, axis=2)[:, None, :] + 1e-9)
        ).mean((1, 2))

    energy = lambda v: (v**2).mean((1, 2, 3))
    values = dict(
        x_nmse=energy(ps + po - x) / energy(x),
        s_nmse=energy(ps - s) / np.maximum(energy(s), 1e-9),
        o_nmse=energy(po - o) / np.maximum(energy(o), 1e-9),
        s_energy=energy(ps),
        o_energy=energy(po),
        cancellation=(energy(ps) + energy(po)) / np.maximum(energy(ps + po), 1e-9),
        eye_s=corr(ps),
        target_s_energy=energy(s),
        target_o_energy=energy(o),
        target_s_derivative=energy(np.diff(s, axis=3)),
        target_eye_s=corr(s),
        physical_teacher_nmse=energy(ps - data["s"][ix].cpu().numpy())
        / np.maximum(energy(data["s"][ix].cpu().numpy()), 1e-9),
    )
    if "first_wave" in z:
        values["first_only_x_nmse"] = energy(z["first_wave"] - x) / energy(x)
    pd.DataFrame(dict(subject=meta.subject.values[dev], **values)).groupby("subject").mean().to_csv(
        folder / "guardrails.csv"
    )
    usage = []
    for j in range(len(model.books("S"))):
        for b, branch in enumerate("SO"):
            book = model.books(branch)[j]
            ids = z["ids" + str(j)][:, :, b]
            n = np.bincount(ids.ravel(), minlength=book.codebook_size)
            p = n[n > 0] / n.sum()
            usage.append(
                dict(
                    branch=branch,
                    level=j + 1,
                    k=len(n),
                    active=int((n > 0).sum()),
                    entropy_bits=float(-(p * np.log2(p)).sum()),
                )
            )
    save(folder / "usage.json", usage)


def run_cell(arm, seed, device, meta, data, control, pilot):
    folder = OUT / ("pilot" if pilot else "models") / f"{arm}_{seed}"
    with owner_lock(folder):
        if (folder / "done.json").exists():
            return
        start = time.monotonic()
        model = training(arm, seed, device, meta, data, control, folder, pilot)
        if not pilot:
            diagnostics(model, meta, data, control, device, folder)
            x = screen.array("wave")
            h = screen.array("hb").reshape(len(x), 4, 15, 40)
            m, _, _ = screen.panel()
            z = extract(model, x, h, device)
            if arm.startswith("cross_"):
                z["exchange_wrong"] = sum(
                    v
                    for k, v in extract(model, x, h[m.donor.to_numpy()], device).items()
                    if k.startswith("level")
                )
                # Both absence interventions happen before any exchange or quantization.
                for mode, xe, he in [
                    ("without_eeg", np.zeros_like(x), h),
                    ("without_hb", x, np.zeros_like(h)),
                ]:
                    z[mode] = sum(
                        v for k, v in extract(model, xe, he, device).items() if k.startswith("level")
                    )
            save_npz_atomic(folder / "science.npz", **z)
        save(
            folder / "done.json",
            dict(
                status="completed",
                arm=arm,
                seed=seed,
                experimental_steps=8 if pilot else len(schedule(meta, seed)[0]),
                parameters=sum(p.numel() for p in model.parameters()),
                seconds=time.monotonic() - start,
                peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20,
                device=os.environ["CUDA_VISIBLE_DEVICES"],
            ),
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["pilot", "train"])
    p.add_argument("--arms", nargs="+", choices=ARMS, required=True)
    p.add_argument("--seeds", nargs="+", type=int, choices=TR["seeds"], default=TR["seeds"])
    args = p.parse_args()
    check()
    torch.set_num_threads(1)
    if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 1 or not os.environ.get(
        "CUDA_VISIBLE_DEVICES", ""
    ).startswith("GPU-"):
        raise ValueError("launch via the UUID-bound unit")
    xy = screen.array("xy")[:, 0]
    distances = ((xy[:, :, None] - xy[:, None, :]) ** 2).sum(-1)
    distances[:, np.arange(4), np.arange(4)] = np.inf
    if not np.all(distances.argmin(2) == [1, 0, 3, 2]):
        raise ValueError("registered four-site nearest neighbors changed")
    device = "cuda:0"
    meta, raw = arrays()
    control = controls(meta, raw)
    if args.stage == "pilot":
        with owner_lock(OUT, "controls", blocking=True):
            save(OUT / "target_controls.json", control)
    else:
        if control != read(OUT / "target_controls.json"):
            raise ValueError("training controls changed")
    data = {k: torch.as_tensor(v, device=device) for k, v in raw.items()}
    del raw
    for arm in args.arms:
        for seed in args.seeds:
            run_cell(arm, seed, device, meta, data, control, args.stage == "pilot")


if __name__ == "__main__":
    main()
