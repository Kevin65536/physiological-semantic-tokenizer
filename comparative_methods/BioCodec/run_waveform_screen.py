"""Score the released BioCodec EEG tokenizer on the retained LaBraM screen's exact X panel."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.signal import resample_poly

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
sys.path.insert(0, str(ROOT))
from retrain_joint_qc_tokenizer import configuration
from src.data.dual_component_tokenizer import (
    base_tokenizer_batch,
    base_tokenizer_training_data,
)
from run_labram_waveform_screen import score

CFG = yaml.safe_load((OWNER / "waveform_screen_v1.yaml").read_text())
KEY = ["subject", "record", "parent", "start_s", "position", "signal"]


def load_model():
    upstream = OWNER / "upstream"
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    path = upstream / "biocodec_ckpt.pt"
    with path.open("rb") as f:
        if (
            revision != CFG["revision"]
            or hashlib.file_digest(f, "sha256").hexdigest() != CFG["checkpoint_sha256"]
        ):
            raise ValueError("BioCodec release identity changed")
    sys.path.insert(0, str(upstream))
    from biocodec.model import BioCodecModel

    model = BioCodecModel._get_optimized_model(signal_normalize=False)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["epoch"] != 2:
        raise ValueError("expected released EEG checkpoint")
    model.load_state_dict(
        {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()},
        strict=True,
    )
    return model.eval()


def resample(x, src, dst):
    factor = math.gcd(src, dst)
    return resample_poly(x, dst // factor, src // factor, axis=-1, padtype="line")


@torch.inference_mode()
def reconstruct(model, x, hz):
    """One row per scalar waveform. Only uint8 codes cross the decoder boundary."""
    if x.ndim != 2 or x.shape[1] != 30 * hz or not np.isfinite(x).all():
        raise ValueError("complete native 30-second observations required")
    grid = resample(x, hz, 250)
    segments = np.clip(grid, -10, 10).astype("float32").reshape(-1, 1, 1250)
    codes = []
    for block in torch.from_numpy(segments).split(16):
        [(words, scale)] = model.encode(block)
        if (
            scale is not None
            or words.dtype != torch.long
            or words.shape[1:] != (6, 105)
            or words.min() < 0
            or words.max() > 255
        ):
            raise ValueError(
                "expected six integer codebooks and no sample-dependent scale"
            )
        codes.append(words.to(torch.uint8))
    codes = torch.cat(codes)
    decoded = [
        model.decode([(block.long(), None)])[:, 0, :1250] for block in codes.split(16)
    ]
    reconstructed = torch.cat(decoded).reshape(len(x), 7500).numpy()
    return (
        resample(reconstructed, 250, hz),
        codes.numpy().reshape(len(x), 6, 6, 105),
        int((abs(grid) > 10).sum()),
    )


def check_synthetic(model):
    for hz in (10, 200):
        t = np.arange(30 * hz) / hz
        x = np.stack([np.sin(2 * np.pi * t), 0.7 * np.cos(2 * np.pi * t)])
        pred, ids, clipped = reconstruct(model, x, hz)
        assert pred.shape == x.shape and np.isfinite(pred).all() and clipped == 0
        assert ids.shape == (2, 6, 6, 105) and ids.dtype == np.uint8
        with torch.inference_mode():
            raw = torch.from_numpy(
                resample(x, hz, 250).astype("float32").reshape(-1, 1, 1250)
            )
            direct = model(raw).reshape(2, 7500).numpy()
        np.testing.assert_allclose(
            pred, resample(direct, 250, hz), atol=1e-5, rtol=1e-5
        )


def plot(out, case, waves):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), layout="constrained")
    for ax, (signal, hz, seconds) in zip(
        axes, [("EEG", 200, 2), ("HbO", 10, 30), ("HbR", 10, 30)], strict=True
    ):
        for label, wave, ink, style in zip(
            ["Observed X", "BioCodec", "S/O sum"],
            waves[signal],
            ["#20262c", "#ab482b", "#187572"],
            ["-", "--", ":"],
            strict=True,
        ):
            ax.plot(
                np.arange(hz * seconds) / hz,
                wave[: hz * seconds],
                style,
                color=ink,
                label=label,
                lw=1.45,
            )
        ax.set(xlabel="Time in window (s)", ylabel="Training-scale units", title=signal)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.suptitle(
        f"{case['subject']}  {case['record']}  |  onset {case['start']:.3f} s  |  Fp1 EEG and AF7 Hb"
    )
    fig.savefig(out / "waveforms.png", dpi=220)
    plt.close(fig)


def run(model, out):
    parent = ROOT / CFG["panel"]
    if json.loads((parent / "summary.json").read_text())["status"] != "completed":
        raise ValueError("incomplete parent screen")
    planned = [
        r
        for r in json.loads((parent / "panel.json").read_text())
        if r["split"] == "validation"
    ]
    cases = [r for r in planned if r["complete"]]
    population = CFG["population"]
    if (len(planned), len(cases), len({r["subject"] for r in cases})) != (
        population["planned_validation_windows"],
        population["common_complete_windows"],
        population["independent_subjects"],
    ):
        raise ValueError("paired validation denominator changed")
    cfg = configuration()
    admitted = set(map(tuple, cfg["data"]["validation_subjects"]))
    manifest, scales, _, _ = base_tokenizer_training_data(
        cfg, ROOT, recorded_root=CFG["g0_recorded_root"]
    )
    rows = manifest["rows"]
    frames, examples, clipped = [], {}, {}
    for mod, labels, hz in [("eeg", ["EEG"], 200), ("hb", ["HbO", "HbR"], 10)]:
        triples = [c[mod] for c in cases]
        for case, (i, start, pos) in zip(cases, triples, strict=True):
            r = rows[i]
            if (
                r["split"] != "validation"
                or r["dataset"] != CFG["dataset"]
                or (r["dataset"], r["subject"]) not in admitted
                or r["rate"] != hz
                or r["positions"][pos] != CFG["sites"][mod]
                or any(r[k] != case[k] for k in ("subject", "record", "parent"))
                or r["native_start_sample"] / r["native_rate_hz"] + start
                != case["start"]
                or not any(
                    g["start_s"] == start and pos in g["positions"]
                    for g in r["inputs"]["score"]
                )
            ):
                raise ValueError(
                    "frozen crop identity or development authorization changed"
                )
        batch = base_tokenizer_batch(rows, triples, scales)
        if not batch["mask"].all():
            raise ValueError("incomplete scoring support")
        x = batch["x"].reshape(len(cases), 30 * hz, len(labels)).astype(float)
        var = batch["vx"][:, 0, : len(labels)]
        bio = np.empty_like(x)
        with np.load(parent / f"{mod}_example.npz") as previous:
            np.testing.assert_allclose(x[0], previous["x"])
            for j, signal in enumerate(labels):
                bio[:, :, j], ids, clipped[signal] = reconstruct(model, x[:, :, j], hz)
                examples[signal] = (x[0, :, j], bio[0, :, j], previous["so"][:, j])
                np.savez_compressed(out / f"{signal}_codes.npz", ids=ids)
        errors, correlations = score(x, bio, var)
        zero, _ = score(x, np.zeros_like(x), var)
        records = []
        for i, (case, (_, start, _)) in enumerate(zip(cases, triples, strict=True)):
            for j, signal in enumerate(labels):
                records.append(
                    {
                        **{k: case[k] for k in ("subject", "record", "parent")},
                        "position": CFG["sites"][mod],
                        "start_s": start,
                        "signal": signal,
                        "nmse": errors[i, j],
                        "correlation": correlations[i, j],
                        "zero_nmse": zero[i, j],
                    }
                )
        fresh = pd.DataFrame(records)
        old = pd.read_csv(parent / f"{mod}_windows.csv")
        for method in ("zero", "so"):
            ref = old.loc[old.model == method, KEY + ["nmse", "correlation"]]
            merged = fresh.merge(
                ref,
                on=KEY,
                how="outer",
                validate="one_to_one",
                indicator=True,
                suffixes=("", "_ref"),
            )
            if not merged["_merge"].eq("both").all():
                raise ValueError("paired rows differ from retained controls")
            if method == "zero":
                np.testing.assert_allclose(
                    merged.zero_nmse, merged.nmse_ref, rtol=1e-7, atol=1e-9
                )
            frames.append(ref.assign(model=method))
        frames.append(fresh.drop(columns="zero_nmse").assign(model="BioCodec"))
    windows = pd.concat(frames, ignore_index=True)
    windows.to_csv(out / "windows.csv", index=False)
    subjects = (
        windows.groupby(["signal", "model", "subject", "record"])[
            ["nmse", "correlation"]
        ]
        .mean()
        .groupby(level=[0, 1, 2])
        .mean()
    )
    subjects.to_csv(out / "subjects.csv")
    summary = subjects.groupby(level=[0, 1]).mean()
    summary.to_csv(out / "comparison.csv")
    plot(out, cases[0], examples)
    result = dict(
        status="completed",
        config=CFG,
        paired_cases=cases,
        incomplete=[r for r in planned if not r["complete"]],
        encoder_clipped_samples_at_250hz=clipped,
        ids_per_scalar_per_30s=6 * 6 * 105,
        integer_bits_per_second=6 * 6 * 105 * 8 / 30,
        failures=0,
        smoke="passed",
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, default=OWNER / CFG["output"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = load_model()
    check_synthetic(model)
    if args.synthetic:
        print("EEG and Hb rate adapters and integer decoder passed")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        run(model, args.output)
