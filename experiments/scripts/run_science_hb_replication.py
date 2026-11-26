"""Independent Hb-side replication of rate, local sharing and teacher-source contrasts."""

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
from scipy.linalg import solve
import torch
import yaml

import run_frozen_science_screen as screen
from prepare_science_ablations import OUT, CONFIG, HB, ROOT, check
from run_science_ablations import TR, read, save, training, extract
from score_science_ablations import paired_ci, branch_kernels
from run_dual_component_tokenizer import owner_lock
from src.tokenizers.dual_component_tokenizer import build_base_tokenizer
from src.tokenizers.science_ablations import ScienceTokenizer
from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_batch
from src.analysis.frozen_science_screen import slot_vectors, gram, interaction_blocks
from src.inference.conditional_slot_training import fit, forward
from src.utils.io import save_npz_atomic

PROTOCOL = CONFIG.with_name("science_hb_replication_v1.yaml")
PLAN = yaml.safe_load(PROTOCOL.read_text())
DEST = OUT / PLAN["output_child"]
ARMS = [*PLAN["sizes"], *PLAN["network"], *PLAN["teacher"]]


def prepare():
    if PLAN["parent"] != CONFIG.name or PLAN["scope"]["protected"] != "forbidden" or len(ARMS) != 12:
        raise ValueError("Hb scope")
    with owner_lock(DEST, "prepare"):
        if (DEST / "prepared.json").exists():
            return
        (DEST / "resolved.yaml").write_text(PROTOCOL.read_text())
        m = pd.read_csv(OUT / "crops.csv")
        cfg, _ = screen.audit.aligned.settings("hb")
        manifest, scales, _, _ = base_tokenizer_training_data(
            cfg, ROOT, recorded_root=screen.spectral.PLAN["recorded_root"]
        )
        by = {
            r["parent"]: i
            for i, r in enumerate(manifest["rows"])
            if r["modality"] == "hb" and r["dataset"] == screen.parent.DS and r["record"] == "cnt_nback"
        }
        raw = []
        ss = []
        cache = OrderedDict()
        for who, g in m.groupby("subject", sort=False):
            with np.load(OUT / "data" / (who + ".npz")) as z:
                old = z["h"]
            for j, row in enumerate(g.itertuples()):
                i = by[row.parent]
                r = manifest["rows"][i]
                if (r["split"], r["subject"]) != (row.split, who):
                    raise ValueError("Hb split mismatch")
                batch = base_tokenizer_batch(
                    manifest["rows"],
                    [(i, row.offset, r["positions"].index(s)) for s in HB],
                    scales,
                    array_cache=cache,
                )
                np.testing.assert_array_equal(old[j], batch["x"].astype("float32"))
                if not batch["mask"].all():
                    raise ValueError("incomplete Hb target")
                raw.append(batch["x"])
                ss.append(batch["s"])
        x = np.asarray(raw, dtype="float32")
        s = np.asarray(ss, dtype="float32")
        tr = np.flatnonzero((m.split == "train") & (m.donor >= 0))
        t = (np.arange(20) - 9.5) / 10
        t = t / (t @ t)
        slopes = np.einsum("nbsic,i->nbsc", s[tr].reshape(-1, 4, 15, 20, 2), t)
        freq = np.fft.rfftfreq(300, 0.1)
        spec = np.fft.rfft(x[tr].reshape(-1, 4, 300, 2), axis=2)
        tar = s[tr].reshape(-1, 4, 300, 2)
        power = lambda a: np.mean(a**2, axis=(0, 1, 2))
        options = []
        for f in (0.0334, 0.067, 0.1, 0.167, 0.233, 0.4):
            smooth = np.fft.irfft(spec * (freq[None, None, :, None] <= f), n=300, axis=2)
            gain = np.sqrt(power(tar) / power(smooth))
            match = np.abs(np.log(power(np.diff(smooth * gain, axis=2)) / power(np.diff(tar, axis=2)))).mean()
            options.append((match, f, gain.tolist()))
        _, cutoff, gains = min(options)
        save_npz_atomic(DEST / "data.npz", x=x, s=s)
        save(
            DEST / "prepared.json",
            dict(
                status="completed",
                crops=len(m),
                train=len(tr),
                calibration=[slopes.mean((0, 1, 2)).tolist(), slopes.std((0, 1, 2)).tolist()],
                cutoff=cutoff,
                gains=gains,
                scales=[scales["positions"]["hb"][screen.parent.DS][site] for site in HB],
            ),
        )


def create(arm, seed, device):
    torch.manual_seed(seed)
    cfg, task = screen.audit.aligned.settings("hb")
    base = build_base_tokenizer(cfg, task)
    base.spectral_head = torch.nn.Linear(128, 2)
    return ScienceTokenizer(base, arm, PLAN["sizes"].get(arm, [1024])).to(device)


def moments():
    ref = read(DEST / "prepared.json")
    sc = ref["scales"]
    v = lambda field: np.array([[a[field][key] for a in sc] for key in "xso"], dtype="float32")[
        :, :, None, None, :
    ]
    return (
        v("target_variance"),
        v("difference_variance"),
        np.ones(4, dtype="float32"),
        np.array(ref["calibration"], dtype="float32"),
    )


def targets(arm, data, ix, donor, control):
    if arm == "sham_teacher":
        return data["s"][donor[ix]]
    if arm == "generic":
        x = data["x"][ix].reshape(-1, 4, 300, 2)
        freq = torch.fft.rfftfreq(300, 0.1, device=x.device)
        sm = torch.fft.irfft(
            torch.fft.rfft(x, dim=2) * (freq[None, None, :, None] <= control["cutoff"]), n=300, dim=2
        )
        return (sm * x.new_tensor(control["gains"])).reshape(len(ix), 4, 15, 40)
    return data["s"][ix]


def loss_fn(out, x, s, var, dvar, scale, cal, weights, attr):
    val = out["commitment"]
    for j, (b, t) in enumerate(zip(("sum", "S", "O"), (x, s, x - s))):
        e = (out[b].reshape_as(t) - t).reshape(-1, 4, 15, 20, 2)
        val = (
            val
            + 0.5 * (e.square() / var[j]).mean()
            + (0.1 / 3) * (e.flatten(2, 3).diff(dim=2).square() / dvar[j, :, 0]).mean()
        )
    if attr:
        t = (torch.arange(20, device=x.device, dtype=x.dtype) - 9.5) / 10
        slopes = torch.einsum("bslic,i->bslc", s.reshape(-1, 4, 15, 20, 2), t / (t @ t))
        val += attr * (out["spectrum"].reshape_as(slopes) - (slopes - cal[0]) / cal[1]).square().mean()
    return val


def strict_input():
    h = screen.array("hb")
    sc = read(screen.BASE / "strict_common/coordinates.json")
    wave = (
        (h - h[:, :, :50].mean(2, keepdims=True))
        * sc["gain"]
        / np.array(sc["site_scales"])[None, :, None, None]
    )
    return wave.astype("float32").reshape(len(h), 4, 15, 40)


def train(arm, seed, meta, data, control, pilot):
    f = DEST / ("pilot" if pilot else "models") / f"{arm}_{seed}"
    with owner_lock(f):
        if (f / "done.json").exists():
            return
        start = time.monotonic()
        model = training(
            arm,
            seed,
            "cuda:0",
            meta,
            data,
            control,
            f,
            pilot,
            factory=create,
            moments=moments,
            targets=targets,
            loss_function=loss_fn,
            protocol=PLAN,
            default_attr=0.0,
        )
        if not pilot:
            x = strict_input()
            z = extract(model, x, x, "cuda:0")
            save_npz_atomic(f / "science.npz", **z)
            dev = np.flatnonzero((meta.split == "validation") & (meta.donor >= 0))
            ix = torch.as_tensor(dev, device="cuda:0")
            donor = torch.as_tensor(meta.donor.values, device="cuda:0")
            xx = data["x"][ix]
            sy = targets(arm, data, ix, donor, control).cpu().numpy()
            orig = xx.cpu().numpy()
            dec = extract(model, xx, xx, "cuda:0", True)
            norm = lambda a, t: np.mean((a - t) ** 2, axis=(1, 2, 3)) / np.maximum(
                np.mean(t**2, axis=(1, 2, 3)), 1e-9
            )
            vals = dict(
                x_nmse=norm(dec["S"] + dec["O"], orig),
                s_nmse=norm(dec["S"], sy),
                o_nmse=norm(dec["O"], orig - sy),
            )
            if "first_wave" in dec:
                vals["first_only_x_nmse"] = norm(dec["first_wave"], orig)
            pd.DataFrame(dict(subject=meta.subject.values[dev], **vals)).groupby("subject").mean().to_csv(
                f / "guardrails.csv"
            )
        save(
            f / "done.json",
            dict(
                status="completed",
                seed=seed,
                arm=arm,
                steps=8 if pilot else 1280,
                seconds=time.monotonic() - start,
                parameters=sum(v.numel() for v in model.parameters()),
                peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20,
            ),
        )


def paired_reader(m, tr, ev, y, hb, eeg, xy, base, r):
    sd = y[tr].std(0)
    t = dict(
        kind="A",
        people=m.subject.values,
        donor=m.donor.values,
        y=torch.tensor(y, dtype=torch.float32),
        xy=torch.tensor(xy, dtype=torch.float32),
        v=dict(hb=torch.from_numpy(hb), eeg=torch.from_numpy(eeg)),
        curve=[],
    )
    rows = []
    tar = torch.tensor(r / sd, dtype=torch.float32)
    for rs in screen.SEEDS:
        for way in ("anchor", "capacity", "true", "wrong"):
            if way != "anchor":
                model, _ = fit(t, way, rs, tr, tar, screen.parent.TR["residual_steps"], screen.parent.TR)
            for swap in [False] if way in ("anchor", "capacity") else [False, True]:
                with torch.no_grad():
                    pred = base if way == "anchor" else base + forward(model, t, ev, way, swap).numpy() * sd
                err = (((pred - y[ev]) / sd) ** 2).reshape(len(ev), -1).mean(1)
                for who, ix in m.iloc[ev].reset_index(drop=True).groupby("subject").indices.items():
                    rows.append(
                        dict(
                            subject=who,
                            reader_seed=rs,
                            arm=way,
                            input="mismatched" if swap else "aligned",
                            nmse=err[ix].mean(),
                        )
                    )
    return pd.DataFrame(rows)


def score(task):
    arm, seed = task
    f = DEST / "models" / f"{arm}_{seed}"
    with owner_lock(f, "score"):
        if (f / "score.json").exists():
            return
        with np.load(f / "science.npz") as z:
            x = sum(z[k] for k in z.files if k.startswith("level"))
            ids = z["ids0"].astype("int64")
            if "ids1" in z:
                ids = ids * PLAN["sizes"][arm][1] + z["ids1"]
        m, tr, ev = screen.panel()
        hb = slot_vectors(x, tr)
        eeg = screen.source("embedding", tr)
        y = screen.array("target").astype(float)
        with np.load(screen.OUT / "anchor.npz") as z:
            base = z["base"].reshape(y[ev].shape)
            r = y[tr] - z["oof"].reshape(y[tr].shape)
        paired_reader(m, tr, ev, y, hb, eeg, screen.array("xy"), base, r).to_csv(
            f / "people.csv", index=False
        )
        # Additive and nonadditive joint predictions of strictly future Hb, without EEG.
        ks, ko = branch_kernels(x, tr)
        by = []
        target = y.reshape(len(m), -1)
        ym = target[tr].mean(0)
        for name, (a, b) in interaction_blocks(ks, ko, tr, ev, m.donor.to_numpy()).items():
            pred = ym + b @ solve(a + np.eye(len(tr)), target[tr] - ym, assume_a="pos")
            err = (((pred - target[ev]) / target[tr].std(0)) ** 2).mean(1)
            for who, ix in m.iloc[ev].reset_index(drop=True).groupby("subject").indices.items():
                by.append(dict(subject=who, arm=name, nmse=err[ix].mean()))
        pd.DataFrame(by).to_csv(f / "so.csv", index=False)
        h = screen.array("hb").reshape(len(m), 4, 15, 20, 2)
        time_axis = (np.arange(20) - 9.5) / 10
        slope = np.einsum("bslic,i->bslc", h, time_axis / (time_axis @ time_axis))
        attrs = []
        for j in range(2):
            u = x[:, :, j]
            a = u[tr].reshape(-1, 128).astype(float)
            mu = a.mean(0)
            a -= mu
            obs = slope[tr].reshape(-1, 2)
            center = obs.mean(0)
            cov = a.T @ a
            coef = solve(
                cov + np.eye(128) * max(cov.trace() / 128 * 0.01, 1e-8), a.T @ (obs - center), assume_a="pos"
            )
            pred = (u[ev] - mu) @ coef + center
            for who, ix in m.iloc[ev].reset_index(drop=True).groupby("subject").indices.items():
                err = (((pred[ix] - slope[ev[ix]]) / obs.std(0)) ** 2).mean((0, 1, 2))
                _, n = np.unique(ids[ev[ix], :, j], return_counts=True)
                p = n / n.sum()
                attrs.append(
                    dict(
                        subject=who,
                        branch="SO"[j],
                        hbo_slope_nmse=err[0],
                        hbr_slope_nmse=err[1],
                        active_joint_codes=len(n),
                        joint_entropy_bits=-(p * np.log2(p)).sum(),
                    )
                )
        pd.DataFrame(attrs).to_csv(f / "attributes.csv", index=False)
        pd.DataFrame(screen.task_scores(arm, gram(hb), gram(eeg))).to_csv(f / "task.csv", index=False)
        save(f / "score.json", dict(status="completed", events=len(ev)))


def transfer_features():
    p = DEST / "wg_eeg.npz"
    if p.exists():
        return
    with np.load(OUT / "wg/panel.npz") as z:
        wave = z["x"]
    e = screen.audit.network("eeg", 8192, 17)
    ids, _ = screen.encode(e, wave)
    wg = np.stack(
        [e.branches[b]["quantizer"].codebook.numpy()[ids[:, :, i]] for i, b in enumerate("SO")], axis=2
    )
    save_npz_atomic(p, vectors=np.concatenate([screen.array("embedding"), wg]))


def wg_score(task):
    arm, seed = task
    folder = DEST / "models" / f"{arm}_{seed}"
    with owner_lock(folder, "wg"):
        if (folder / "wg_done.json").exists():
            return
        n, tr, _ = screen.panel()
        w = pd.read_csv(OUT / "wg/wg_panel.csv")
        comb = pd.concat([n, w.assign(donor=w.donor + len(n))], ignore_index=True)
        ev = np.arange(len(n), len(comb))
        with np.load(OUT / "wg/panel.npz") as z:
            h, y, xy, base, oof = [z[k] for k in ("h", "y", "xy", "base", "oof")]
        scales = read(DEST / "prepared.json")["scales"]
        original = screen.parent.PARENT / screen.parent.CFG["parents"]["cached"]
        gain = read(original / "input_coordinates.json")["training_hb_gain"]
        h = (
            (
                (h - h[:, :, :50].mean(2, keepdims=True))
                * gain
                / np.array([q["scale"] for q in scales])[None, :, None, None]
            )
            .astype("float32")
            .reshape(len(comb), 4, 15, 40)
        )
        model = create(arm, seed, "cpu").eval()
        state = torch.load(folder / "state.pt", map_location="cpu", weights_only=False)
        if state["contract"] != PLAN or state["step"] != 1280:
            raise ValueError("Hb transfer source")
        model.load_state_dict(state["model"])
        result = extract(model, h, h, "cpu")
        hb = slot_vectors(sum(v for k, v in result.items() if k.startswith("level")), tr)
        with np.load(DEST / "wg_eeg.npz") as z:
            eeg = slot_vectors(z["vectors"], tr)
        r = y[tr] - oof.reshape(y[tr].shape)
        paired_reader(comb, tr, ev, y.astype(float), hb, eeg, xy, base.reshape(y[ev].shape), r).to_csv(
            folder / "wg_people.csv", index=False
        )
        save(folder / "wg_done.json", dict(status="completed", events=len(ev), subjects=w.subject.nunique()))


def finalize():
    from src.analysis.conditional_evidence import effect_tables

    folders = [(a, s, DEST / "models" / f"{a}_{s}") for a in ARMS for s in TR["seeds"]]
    for a, s, f in folders:
        if (
            read(f / "done.json")["steps"] != 1280
            or read(f / "wg_done.json")["events"] != 237
            or read(f / "score.json")["events"] != 91
        ):
            raise ValueError("incomplete Hb cell")
    for name in ("people", "wg_people", "so", "attributes", "task", "guardrails"):
        df = pd.concat(
            [
                pd.read_csv(f / (name + ".csv"), dtype={"arm": str}).assign(model=a, tokenizer_seed=s)
                for a, s, f in folders
            ]
        )
        df.to_csv(DEST / (name + ".csv"), index=False)
    stats = []
    for name in ("people", "wg_people"):
        d = (
            pd.read_csv(DEST / (name + ".csv"), dtype={"arm": str})
            .query("input=='aligned'")
            .groupby(["model", "subject", "arm"], as_index=False)
            .nmse.mean()
        )
        eff = effect_tables(
            d,
            "nmse",
            [
                ("over_capacity", "capacity", "true"),
                ("over_full_Hb", "anchor", "true"),
                ("over_wrong", "wrong", "true"),
            ],
            ["model"],
        )
        eff.to_csv(DEST / (name + "_increments.csv"), index=False)
        raw = d.pivot(index="subject", columns=["model", "arm"], values="nmse")
        delta = {a: raw[a].capacity - raw[a].true for a in ARMS}
        for a in ARMS:
            stats.append(dict(panel=name, model=a, **paired_ci(delta[a] - delta["B0"])))
        for a, b in [
            ("independent_size", "shared"),
            ("shuffled_neighbor", "neighbor"),
            ("single11", "dual11"),
            ("dual10", "dual11"),
            ("attr_full", "attr_half"),
        ]:
            stats.append(dict(panel=name, model=b + "_minus_" + a, **paired_ci(delta[b] - delta[a])))
        stats.append(
            dict(
                panel=name,
                model="rate_by_organization",
                **paired_ci(delta["dual11"] - delta["dual10"] - delta["single11"] + delta["B0"]),
            )
        )
    pd.DataFrame(stats).to_csv(DEST / "paired.csv", index=False)
    d = pd.read_csv(DEST / "so.csv").groupby(["model", "subject", "arm"], as_index=False).nmse.mean()
    effect_tables(
        d,
        "nmse",
        [
            ("O_given_S", "S", "additive"),
            ("S_given_O", "O", "additive"),
            ("above_additive", "additive", "interaction_true_aligned"),
            ("above_same_budget", "additive_expansion", "interaction_true_aligned"),
            ("above_wrong", "interaction_wrong_aligned", "interaction_true_aligned"),
        ],
        ["model"],
    ).to_csv(DEST / "so_contrasts.csv", index=False)
    save(
        DEST / "summary.json",
        dict(
            status="completed",
            planned_fits=len(folders),
            completed_fits=len(folders),
            scope=PLAN["scope"],
            nback_train=309,
            nback_development=91,
            WG_validation=237,
            primary=PLAN["primary"],
            endpoint_table="paired.csv",
            additional_attribute="Hb B0 has no explicit slope head loss. attr_full and attr_half use 0.01 and 0.005.",
            frozen_EEG="original aligned EEG tokenizer",
            scientific_scope="exploratory, four n-back and five WG subjects",
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "pilot", "train", "score", "transfer", "summary"])
    p.add_argument("--seed", type=int, choices=[17, 43], default=17)
    a = p.parse_args()
    check()
    torch.set_num_threads(1)
    if a.stage == "prepare":
        prepare()
        return
    if a.stage == "summary":
        with owner_lock(DEST, "summary"):
            finalize()
        return
    if a.stage in ("score", "transfer"):
        if a.stage == "transfer":
            transfer_features()
        jobs = {(arm, s) for arm in ARMS for s in TR["seeds"]}
        with owner_lock(DEST, a.stage), ProcessPoolExecutor(4, mp_context=mp.get_context("fork")) as pool:
            while jobs:
                ready = [
                    t for t in sorted(jobs) if (DEST / "models" / f"{t[0]}_{t[1]}" / "done.json").is_file()
                ]
                if ready:
                    list(pool.map(score if a.stage == "score" else wg_score, ready))
                    jobs.difference_update(ready)
                else:
                    time.sleep(10)
        return
    import os

    if (
        not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-")
        or "," in os.environ["CUDA_VISIBLE_DEVICES"]
    ):
        raise ValueError("use UUID-bound unit")
    ctrl = read(DEST / "prepared.json")
    m = pd.read_csv(OUT / "crops.csv")
    with np.load(DEST / "data.npz") as z:
        data = {k: torch.as_tensor(z[k], device="cuda:0") for k in z.files}
    data["h"] = data["x"]  # all registered Hb-only cells receive no cross-modal input
    for arm in ["B0"] if a.stage == "pilot" else ARMS:
        train(arm, a.seed, m, data, ctrl, a.stage == "pilot")


if __name__ == "__main__":
    main()
