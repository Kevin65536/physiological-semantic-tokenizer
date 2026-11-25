"""Small paired G0 waveform screen with local VQNSP and the authors' EEG checkpoint."""

import argparse
from collections import defaultdict
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apply_joint_tokenizer import checked_path, load_branch_pair
from retrain_joint_qc_tokenizer import configuration, dump
from src.data.dual_component_tokenizer import (
    base_tokenizer_batch,
    base_tokenizer_joint_eeg,
    base_tokenizer_training_data,
)
from src.tokenizers.labram_vqnsp import LaBraMVQNSP_EEG, LaBraMVQNSP_fNIRS

CFG = (
    ROOT
    / "experiments/configs/physiology_semantic_tokenizer/labram_waveform_screen_v1.yaml"
)
C = yaml.safe_load(CFG.read_text())
PARENT = ROOT / yaml.safe_load((ROOT / C["parent"]).read_text())["run"]


def panel(rows, split):
    by_crop = defaultdict(dict)
    for i, r in enumerate(rows):
        mod = r["modality"]
        if (
            r["dataset"] != C["dataset"]
            or r["split"] != split
            or C["positions"][mod] not in r["positions"]
        ):
            continue
        pos = r["positions"].index(C["positions"][mod])
        for g in r["inputs"]["train" if split == "train" else "score"]:
            if pos in g["positions"]:
                by_crop[r["parent"], g["start_s"]][mod] = (i, g["start_s"], pos)
    groups = defaultdict(list)
    for pair in by_crop.values():
        if set(pair) == {"eeg", "hb"}:
            row, start, _ = pair["eeg"]
            r = rows[row]
            origin = r["native_start_sample"] + start * r["native_rate_hz"]
            groups[r["subject"], r["record"]].append((origin, row, pair))
    selected = []
    for key in sorted(groups):
        unique = []
        end = -1
        for origin, row, pair in sorted(groups[key], key=lambda item: item[:2]):
            if origin >= end:
                unique.append(pair)
                end = origin + 30 * rows[row]["native_rate_hz"]
        n = min(len(unique), C["per_subject_record"][split])
        selected.extend(
            unique[j] for j in np.linspace(0, len(unique) - 1, n, dtype=int)
        )
    return selected


def local_model(mod):
    eeg = mod == "eeg"
    cls = LaBraMVQNSP_EEG if eeg else LaBraMVQNSP_fNIRS
    return cls(
        seq_length=6000 if eeg else 300, patch_size=C["patch"][mod], **C["local_model"]
    )


def fit_local(mod, model, train, out):
    opt = C["training"]
    file = out / (mod + "_local.pt")
    if file.exists():
        model.load_state_dict(torch.load(file, weights_only=True))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt["lr"])
        values = train.transpose(1, 2).reshape(-1, train.shape[1]).contiguous()
        rng = torch.Generator().manual_seed(opt["seed"])
        for step in range(opt["steps"]):
            take = torch.randint(len(values), (opt["batch"],), generator=rng)
            loss = model(values[take])["loss"]
            if not loss.isfinite():
                raise ValueError(f"{mod} nonfinite training loss at {step}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        torch.save(model.state_dict(), file.with_suffix(".tmp"))
        file.with_suffix(".tmp").replace(file)
    return model.eval()


def official_model(assets):
    path = assets / C["official"]["name"]
    identity = hashlib.sha1(
        f"blob {path.stat().st_size}\0".encode() + path.read_bytes()
    ).hexdigest()
    if identity != C["official"]["checkpoint_git_blob"]:
        raise ValueError("official checkpoint differs from pinned Git blob")
    sys.path.insert(0, str(assets.resolve()))
    upstream = importlib.import_module("modeling_vqnsp")
    with torch.serialization.safe_globals(
        [
            argparse.Namespace,
            (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
            np.dtype,
            type(np.dtype("float64")),
        ]
    ):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = upstream.vqnsp_encoder_base_decoder_3x200x12(n_code=8192, code_dim=64)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), checkpoint["epoch"]


def chunks(x):
    return x.reshape(-1, 1, C["official"]["chunk_seconds"], 200)


def calibration(train):
    fft = torch.fft.fft(chunks(train))
    return [
        (v.mean((1, 2, 3)).mean().item(), v.std((1, 2, 3)).mean().item())
        for v in (fft.abs(), fft.angle())
    ]


@torch.inference_mode()
def official_wave(model, x, moments):
    shape = x.shape
    ch = C["official"]["channel_indices"]
    outputs = []
    for block in chunks(x).split(32):
        _, ids, _ = model.encode(block, ch)
        q = model.quantize.embedding.weight[ids.reshape(len(block), 1, -1)].permute(
            0, 3, 1, 2
        )
        amp, phase = model.decode(q, ch)
        amp = (amp * moments[0][1] + moments[0][0]).clamp_min(0)
        phase = phase * moments[1][1] + moments[1][0]
        outputs.append(torch.fft.ifft(torch.polar(amp, phase)).real)
    return torch.cat(outputs).reshape(shape)


def score(x, y, var):
    if not np.isfinite(y).all():
        raise ValueError("nonfinite prediction")
    a = x - x.mean(1, keepdims=True)
    b = y - y.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    corr = np.divide(
        (a * b).sum(1), den, out=np.full_like(den, np.nan), where=den > 1e-10
    )
    return ((x - y) ** 2).mean(1) / var, corr


def synthetic(assets):
    torch.manual_seed(17)
    for mod in C["positions"]:
        model = local_model(mod)
        x = torch.randn(2, model.seq_length)
        model(x)["loss"].backward()
        model.eval()
        with torch.no_grad():
            direct = model(x)
            wave = model.decode_from_tokens(direct["indices"])
            assert wave.shape == x.shape and direct["indices"].shape == (
                2,
                model.n_patches,
            )
            torch.testing.assert_close(wave, direct["x_rec"], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(
                model._reconstruct_time(
                    *model._compute_fft_targets(model._split_to_patches(x))
                ),
                x,
            )
    model, epoch = official_model(assets)
    x = torch.randn(2, 6000, 1)
    y = official_wave(model, x, calibration(x))
    assert y.shape == x.shape and y.isfinite().all() and epoch == 99
    print("synthetic checks passed", flush=True)


def baseline(mod, rows, scales, examples, batch):
    spec = json.loads((PARENT / "bundle.json").read_text())[mod]
    model = load_branch_pair(spec["kind"], checked_path(PARENT / "bundle.json", spec))
    ds = C["dataset"]
    if mod == "hb":
        return model.decode_ids(model.encode(batch["x"]))["sum"].reshape(-1, 300, 2)
    names = model.metadata["positions"][ds]
    channel = [n.upper() for n in names].index(C["positions"][mod].upper())
    sums = []
    for i, start, pos in examples:
        fields, seen, _ = base_tokenizer_joint_eeg(
            rows,
            dict(row=i, start_s=start, positions=[pos]),
            scales,
            names,
            allowed_splits=("validation",),
        )
        ids = model.encode(
            fields["x"].T.reshape(1, len(names), 15, 400), ds, seen[None]
        )
        sums.append(model.decode_ids(ids, ds)["sum"][0, channel].reshape(6000, 1))
    return np.stack(sums)


def run(out, assets):
    cfg = configuration()
    manifest, scales, _, _ = base_tokenizer_training_data(
        cfg, ROOT, recorded_root=C["recorded_root"]
    )
    rows = manifest["rows"]
    admitted = set(map(tuple, cfg["data"]["validation_subjects"]))
    batches, examples, journal = {}, {}, []
    for split in ("train", "validation"):
        pairs = panel(rows, split)
        if not pairs:
            raise ValueError("empty planned panel")
        for p in pairs:
            for i, start, pos in p.values():
                r = rows[i]
                if r["split"] != split or (
                    split == "validation"
                    and (r["dataset"], r["subject"]) not in admitted
                ):
                    raise ValueError("split boundary")
                if r["modality"] == "eeg" and r["source_unit"] != "uV":
                    raise ValueError("official input requires documented microvolts")
        data = {
            m: base_tokenizer_batch(rows, [p[m] for p in pairs], scales)
            for m in C["positions"]
        }
        keep = np.logical_and.reduce(
            [b["mask"].all(axis=(1, 2)) for b in data.values()]
        )
        for p, ok in zip(pairs, keep, strict=True):
            r = rows[p["eeg"][0]]
            journal.append(
                dict(
                    split=split,
                    subject=r["subject"],
                    record=r["record"],
                    parent=r["parent"],
                    eeg=p["eeg"],
                    hb=p["hb"],
                    start=r["native_start_sample"] / r["native_rate_hz"] + p["eeg"][1],
                    complete=bool(ok),
                )
            )
        for m in data:
            batches[split, m] = {k: v[keep] for k, v in data[m].items()}
            examples[split, m] = [p[m] for p, ok in zip(pairs, keep, strict=True) if ok]
    dump(out / "panel.json", journal)
    # Separate processes isolate training RNG and numerical-library threads.
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    with ProcessPoolExecutor(
        2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                modality_run,
                m,
                batches["train", m],
                batches["validation", m],
                examples["validation", m],
                rows_for_baseline(rows, examples["validation", m]),
                scales,
                out,
                assets,
            )
            for m in C["positions"]
        ]
        for future in futures:
            future.result()
    scores = pd.concat([pd.read_csv(out / f"{m}_windows.csv") for m in C["positions"]])
    grouped = scores.groupby(["model", "signal", "subject", "record"])[
        ["nmse", "correlation"]
    ].mean()
    subjects = grouped.groupby(["model", "signal", "subject"]).mean()
    subjects.to_csv(out / "subjects.csv")
    summary = subjects.groupby(["model", "signal"]).mean()
    summary.to_csv(out / "comparison.csv")
    plot_example(
        out, next(r for r in journal if r["complete"] and r["split"] == "validation")
    )
    dump(
        out / "summary.json",
        dict(
            status="completed",
            experiment_id=C["experiment_id"],
            planned=len(journal),
            completed=sum(r["complete"] for r in journal),
            validation_subjects=sorted(
                {
                    r["subject"]
                    for r in journal
                    if r["complete"] and r["split"] == "validation"
                }
            ),
            by_split={
                s: dict(
                    planned=sum(r["split"] == s for r in journal),
                    complete=sum(r["split"] == s and r["complete"] for r in journal),
                )
                for s in ("train", "validation")
            },
            finite_correlations=scores.groupby(["model", "signal"])
            .correlation.count()
            .rename("windows")
            .reset_index()
            .to_dict("records"),
        ),
    )
    print(summary, flush=True)


def plot_example(out, case):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), layout="constrained")
    styles = {
        "x": ("Observed X", "#222222", "-"),
        "so": ("S/O sum", "#087f68", "--"),
        "local": ("Local VQNSP", "#ba4e21", "-."),
        "official": ("Official EEG + calibration", "#345ab4", ":"),
    }
    for ax, (m, color, label, hz, seconds) in zip(
        axes,
        [
            ("eeg", 0, "Fp1 EEG", 200, 2),
            ("hb", 0, "AF7 HbO", 10, 30),
            ("hb", 1, "AF7 HbR", 10, 30),
        ],
        strict=True,
    ):
        with np.load(out / (m + "_example.npz")) as waves:
            for k, (title, ink, ls) in styles.items():
                if k in waves:
                    values = waves[k][: seconds * hz, color]
                    ax.plot(
                        np.arange(len(values)) / hz,
                        values,
                        color=ink,
                        ls=ls,
                        lw=1.4,
                        label=title,
                    )
        ax.set(ylabel="Training-scale units", xlabel="Time in window (s)", title=label)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.suptitle(
        f"First complete development window | {case['subject']} | {case['record']} | onset {case['start']:.3f} s"
    )
    fig.savefig(out / "waveforms.png", dpi=200)
    plt.close(fig)


def rows_for_baseline(rows, examples):
    return {i: rows[i] for i, _, _ in examples}


def modality_run(mod, train_batch, batch, examples, rows, scales, out, assets):
    torch.set_num_threads(1)
    torch.manual_seed(C["training"]["seed"])
    colors = 1 if mod == "eeg" else 2

    def waveform(b):
        return torch.from_numpy(
            b["x"].reshape(len(b["x"]), -1, colors).astype("float32")
        )

    train, x = waveform(train_batch), waveform(batch)
    model = fit_local(mod, local_model(mod), train, out)
    with torch.inference_mode():
        inputs = x.transpose(1, 2).reshape(-1, x.shape[1]).contiguous()
        local = (
            model.decode_from_tokens(model.get_tokens(inputs))
            .view(len(x), colors, -1)
            .transpose(1, 2)
        )
    predictions = dict(
        local=local.numpy(),
        zero=np.zeros_like(x.numpy()),
        so=baseline(mod, rows, scales, examples, batch),
    )
    if mod == "eeg":
        # Fixed FP1 site and training SD, undo G0 scaling to microvolts then use upstream /100.
        scale = (
            scales["positions"][mod][C["dataset"]][C["positions"][mod]]["scale"] / 100.0
        )
        upstream, epoch = official_model(assets)
        moments = calibration(train * scale)
        predictions["official"] = (
            official_wave(upstream, x * scale, moments).numpy() / scale
        )
        dump(
            out / "calibration.json",
            dict(
                epoch=epoch,
                revision=C["official"]["revision"],
                moments=moments,
                scaled_input_factor=scale,
            ),
        )
    scores = []
    target = batch["x"].reshape(x.shape).astype(float)
    var = batch["vx"][:, 0, :colors]
    np.savez_compressed(
        out / (mod + "_example.npz"),
        x=target[0],
        **{k: v[0] for k, v in predictions.items() if k != "zero"},
    )
    for name, values in predictions.items():
        nmse, corr = score(target, values.astype(float), var)
        if name == "so":
            with np.load(
                PARENT / "evaluation/retrained" / f"{mod}_{C['dataset']}_validation.npz"
            ) as saved:
                lookup = {tuple(v): j for j, v in enumerate(saved["examples"])}
                indices = [lookup[t] for t in examples]
                np.testing.assert_allclose(
                    nmse, saved["nmse"][indices, 0], atol=2e-6, rtol=2e-5
                )
                np.testing.assert_allclose(
                    corr, saved["correlation"][indices, 0], atol=2e-6, rtol=2e-5
                )
        for j, (i, start, pos) in enumerate(examples):
            for c, signal in enumerate(["EEG"] if mod == "eeg" else ["HbO", "HbR"]):
                scores.append(
                    dict(
                        model=name,
                        signal=signal,
                        subject=rows[i]["subject"],
                        record=rows[i]["record"],
                        parent=rows[i]["parent"],
                        start_s=start,
                        position=rows[i]["positions"][pos],
                        nmse=nmse[j, c],
                        correlation=corr[j, c],
                    )
                )
    pd.DataFrame(scores).to_csv(out / f"{mod}_windows.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--assets",
        type=Path,
        required=True,
        help="pinned upstream sources and vqnsp.pth",
    )
    p.add_argument("--output", type=Path, default=ROOT / C["output"])
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.synthetic:
        synthetic(args.assets)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        resolved = args.output / "resolved_config.json"
        if resolved.exists() and json.loads(resolved.read_text()) != C:
            raise ValueError("run config changed")
        dump(resolved, C)
        run(args.output, args.assets)
