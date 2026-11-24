"""Bounded multi-ID rate diagnostic on already frozen EEG event slots."""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib, json, multiprocessing, os
from pathlib import Path
import subprocess, sys, time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/scripts")]
from src.analysis.dual_component_tokenizer import (
    base_training_aggregate,
    joint_reconstruction_metrics,
)
from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_batch
from src.tokenizers.adaptive_band_allocation import synthesize
from src.tokenizers.band_event_dictionary import (
    ORIGINAL_SIZE,
    add_residual_words,
    decode_event_coordinates,
    encode_residual_stages,
    fit_residual_stages,
    word_roles,
)

STATE = {}
LAYERS = (1, 2, 4)


def write(path, value):
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)

    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=convert) + "\n")


def save(path, **fields):
    tmp = path.with_suffix(".pending.npz")
    np.savez_compressed(tmp, **fields)
    tmp.replace(path)


def budget():
    unit = os.environ.get("EVENT_RVQ_UNIT")
    if unit:
        used = (
            int(
                subprocess.check_output(
                    ["systemctl", "--user", "show", unit, "-p", "CPUUsageNSec", "--value"], text=True
                )
            )
            / 1e9
        )
        if used > STATE["cfg"]["resources"]["CPU_seconds"] - 15:
            raise RuntimeError("registered event-RVQ budget reached")


@lru_cache(maxsize=8)
def primary_books(fold, b):
    result = []
    for role in range(5):
        with np.load(STATE["source"] / "books" / f"fold_{fold}_{b}_{role}.npz") as z:
            result.append({k: z[k].copy() for k in z if k != "metadata"})
    return result


@lru_cache(maxsize=8)
def residual_books(fold, b):
    result = []
    for role in range(5):
        with np.load(STATE["out"] / "books" / f"fold_{fold}_{b}_{role}.npz") as z:
            result.append({k: z[k].copy() for k in z if k != "metadata"})
    return result


@lru_cache(maxsize=8)
def old_books(fold, b):
    result = []
    for role in range(5):
        with np.load(
            STATE["run"]
            / "component_source_transfer/band_exchange_words_v1/books"
            / f"fold_{fold}_{b}_{role}.npz"
        ) as z:
            result.append((z["balanced_atoms"].astype(float), z["balanced_gain"].copy()))
    return result


def fit_book(task):
    fold, b, role = task
    dest = STATE["out"] / "books" / f"fold_{fold}_{b}_{role}.npz"
    if dest.exists():
        return
    budget()
    waves = []
    weights = []
    primary = primary_books(fold, b)[role]
    for dataset in STATE["cases"]:
        with np.load(STATE["source"] / dataset / f"fold_{fold}_{b}_corpus.npz") as z:
            chosen = z["fit"] & (z["role"] == role)
            flat = z["flat"][chosen]
            source = z["wave"][chosen].copy()
            weights.append(z["weight"][chosen])
        with np.load(STATE["source"] / dataset / f"fold_{fold}_ids.npz") as z:
            words = z["wave_" + b].ravel()[flat]
        if np.any(words < ORIGINAL_SIZE):
            raise ValueError("residual events require a full primary event word")
        index = (words.astype(np.int32) - ORIGINAL_SIZE) % 2047
        waves.append(source - primary["wave_atoms"][index])
    residual = np.concatenate(waves)
    weight = np.concatenate(weights)
    seed = STATE["cfg"]["training"]["seed"] + 100 * fold + 10 * (b == "O") + role
    start = time.process_time()
    fitted = fit_residual_stages(residual, weight, primary["wave_basis"], seed)
    save(
        dest,
        **fitted,
        metadata=np.array(
            json.dumps(
                dict(
                    fold=fold,
                    branch=b,
                    role=role,
                    seed=seed,
                    fit_events=len(residual),
                    CPU_seconds=time.process_time() - start,
                )
            )
        ),
    )


def encode_dataset(dataset):
    for fold in range(4):
        budget()
        dest = STATE["out"] / dataset
        dest.mkdir(exist_ok=True)
        path = dest / f"fold_{fold}_ids.npz"
        if path.exists():
            continue
        with np.load(STATE["source"] / dataset / f"fold_{fold}_ids.npz") as z:
            fields = {k: z[k].copy() for k in ("examples", "fold")}
            fields.update({b: z["wave_" + b].copy() for b in "SO"})
        for b in "SO":
            model = primary_books(fold, b)
            residual = residual_books(fold, b)
            with np.load(STATE["source"] / dataset / f"fold_{fold}_{b}_corpus.npz") as z:
                wave = z["wave"].copy()
                roles = z["role"].copy()
                flat = z["flat"].copy()
            codes = np.zeros((*fields[b].shape, 3), np.uint8)
            for role in range(5):
                selected = roles == role
                ids = fields[b].ravel()[flat[selected]]
                atom = (ids.astype(np.int32) - ORIGINAL_SIZE) % 2047
                delta = wave[selected] - model[role]["wave_atoms"][atom]
                codes.reshape(-1, 3)[flat[selected]] = encode_residual_stages(delta, residual[role])
            np.testing.assert_array_equal(np.flatnonzero(fields[b].ravel() >= ORIGINAL_SIZE), flat)
            fields["extra_" + b] = codes
        save(path, **fields)
    return dataset


def unpack_primary(blob, shape):
    n = int(np.prod(shape))
    bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8), bitorder="little")[: 14 * n]
    return (bits.reshape(n, 14) @ (1 << np.arange(14))).astype(np.uint16).reshape(shape)


def decode(words, extras, old, new, residual, cov, layers):
    result = {}
    for b in "SO":
        roles, coordinates = decode_event_coordinates(words[b], old[b], new[b], "wave", False)
        events = words[b] >= ORIGINAL_SIZE
        corrected = add_residual_words(coordinates, words[b], extras[b][events], residual[b], layers)
        result[b] = synthesize(corrected, roles, cov[b])
    result["sum"] = result["S"] + result["O"]
    return result


def score_dataset(task):
    dataset, replay = task
    dest = STATE["out"] / dataset
    cache = OrderedDict()
    resource = []
    for fold in range(4):
        budget()
        path = dest / f"fold_{fold}_metrics.npz"
        if path.exists() and not replay:
            continue
        old = {b: old_books(fold, b) for b in "SO"}
        primary = {b: primary_books(fold, b) for b in "SO"}
        residual = {b: residual_books(fold, b) for b in "SO"}
        with np.load(
            STATE["run"]
            / "component_source_transfer/adaptive_band_v1"
            / dataset
            / f"fold_{fold}_model.npz"
        ) as z:
            cov = {b: z[b].copy() for b in "SO"}
        with np.load(dest / f"fold_{fold}_ids.npz") as z:
            ex = z["examples"].copy()
            words = {b: z[b].copy() for b in "SO"}
            extra = {b: z["extra_" + b].copy() for b in "SO"}
        if replay:
            with np.load(path) as z:
                expected = {k: z[k].copy() for k in z}
        with np.load(STATE["source"] / dataset / f"fold_{fold}_metrics.npz") as z:
            baseline = z["wave_mean__nmse"]
        stats = defaultdict(list)
        gallery = {}
        offset = 0
        held = 0
        records = []
        for c in STATE["cases"][dataset]:
            take = np.arange(offset, offset + c["count"])
            offset += c["count"]
            if c["fold"] != fold:
                continue
            budget()
            native = np.asarray(c["native"])
            ix = take[native.argsort()]
            paid = {b: words[b][ix] for b in "SO"}
            sent = {b: extra[b][ix] for b in "SO"}
            for b in "SO":
                stem = f"{c['crop']['row']}_{c['crop']['start_s']}_{b}"
                file = dest / (stem + ".extra8")
                primary_path = (
                    STATE["source"]
                    / dataset
                    / f"{c['crop']['row']}_{c['crop']['start_s']}_wave_{b}.14bit"
                )
                read = unpack_primary(primary_path.read_bytes(), paid[b].shape)
                np.testing.assert_array_equal(read, paid[b])
                paid[b] = read
                events = paid[b] >= ORIGINAL_SIZE
                if not replay:
                    file.write_bytes(sent[b][events].tobytes())
                codes = np.frombuffer(file.read_bytes(), np.uint8).reshape(int(events.sum()), 3)
                np.testing.assert_array_equal(codes, sent[b][events])
                sent[b] = np.zeros_like(sent[b])
                sent[b][events] = codes
                records.append(
                    dict(
                        row=c["crop"]["row"],
                        start_s=c["crop"]["start_s"],
                        branch=b,
                        primary_integers=paid[b].size,
                        event_integers_per_extra_stage=len(codes),
                        primary_path=str(primary_path),
                        extra_path=file.name,
                    )
                )
            if any(STATE["rows"][i]["split"] != "train" for i, t, p in ex[take]):
                raise ValueError("train before target reads")
            # Integer-only decoding completes before signal targets are read.
            predictions = {}
            for layers in LAYERS:
                saved_loader = np.load
                if replay:

                    def reject(*args, **kwargs):
                        raise AssertionError("decoder attempted another file read")

                    np.load = reject
                try:
                    values = decode(paid, sent, old, primary, residual, cov, layers)
                finally:
                    np.load = saved_loader
                predictions[layers] = {k: v[native] for k, v in values.items()}
            examples = ex[take]
            batch = base_tokenizer_batch(
                STATE["rows"], examples, STATE["scales"], array_cache=cache, cache_limit=8
            )
            factors = np.array(
                [
                    STATE["scales"]["positions"]["eeg"][dataset][STATE["rows"][i]["positions"][p]][
                        "scale"
                    ]
                    for i, t, p in examples
                ]
            )
            interval = slice(held, held + c["count"])
            stats["examples"].append(examples)
            for layers, pred in predictions.items():
                actual = joint_reconstruction_metrics(pred, batch, 1, factors)
                if actual["support"]["nonfinite_prediction_points"].any():
                    raise ValueError("nonfinite RVQ reconstruction")
                for key, value in actual["metrics"].items():
                    if replay:
                        np.testing.assert_array_equal(value, expected[f"L{layers}__" + key][interval])
                    else:
                        stats[f"L{layers}__" + key].append(value)
                if layers == 1:
                    np.testing.assert_array_equal(actual["metrics"]["nmse"], baseline[interval])
                for j, e in enumerate(examples):
                    if tuple(e) in STATE["selected"]:
                        label = STATE["selected"][tuple(e)]["audit_index"]
                        for k, value in pred.items():
                            gallery[f"{label}_L{layers}_{k}"] = value[j]
                        if layers == 1:
                            for k in ("x", "s", "o", "mask"):
                                gallery[f"{label}_{k}"] = batch[k][j]
            held += c["count"]
        if replay:
            resource.append(
                dict(
                    fold=fold,
                    positions=held,
                    all3levels_all_metric_arrays_exact=True,
                    decoder_additional_file_reads=0,
                )
            )
        else:
            save(path, **{k: np.concatenate(v) for k, v in stats.items()})
            save(dest / f"fold_{fold}_gallery.npz", **gallery)
            write(dest / f"fold_{fold}_payloads.json", records)
    if replay:
        write(dest / "integer_replay.json", dict(status="passed", folds=resource))
    return dataset


def tuple_support():
    with np.load(
        STATE["run"]
        / "component_source_transfer/hard_as_v1/local_information_v1/training_attributes.npz"
    ) as z:
        bank = {k: z[k].copy() for k in ("examples", "fold", "subject", "valid")}
    result = {}
    for b in "SO":
        result[b] = {}
        for level in LAYERS:
            total = qualified = event_total = event_qualified = 0
            for fold in range(4):
                arrays = []
                codes = []
                exs = []
                for dataset in STATE["cases"]:
                    with np.load(STATE["out"] / dataset / f"fold_{fold}_ids.npz") as z:
                        arrays.append(z[b])
                        codes.append(z["extra_" + b])
                        exs.append(z["examples"])
                np.testing.assert_array_equal(np.concatenate(exs), bank["examples"])
                primary = np.concatenate(arrays)
                extra = np.concatenate(codes)
                local = np.column_stack((primary.ravel(), extra.reshape(-1, 3)[:, : level - 1]))
                _, inverse = np.unique(local, axis=0, return_inverse=True)
                inverse = inverse.reshape(primary.shape)
                count = np.bincount(inverse[bank["valid"] & (bank["fold"] != fold)[:, None]])
                people = np.zeros(len(count), np.int32)
                for p in np.unique(bank["subject"][bank["fold"] != fold]):
                    ids = np.unique(inverse[bank["valid"] & (bank["subject"][:, None] == p)])
                    ids = ids[ids < len(count)]
                    people[ids] += 1
                held = bank["valid"] & (bank["fold"] == fold)[:, None]
                key = inverse[held]
                events = primary[held] >= ORIGINAL_SIZE
                good = key < len(count)
                supported = np.zeros(len(key), bool)
                supported[good] = (count[key[good]] >= 8) & (people[key[good]] >= 2)
                total += len(key)
                qualified += int(supported.sum())
                event_total += int(events.sum())
                event_qualified += int((events & supported).sum())
            result[b][f"L{level}"] = dict(
                complete_patches=total,
                supported_patches=qualified,
                support=qualified / total,
                event_complete=event_total,
                event_supported=event_qualified,
                event_support=event_qualified / event_total,
                requirement="the complete emitted local tuple needs >=8 fit patches and >=2 fit people",
            )
    write(STATE["out"] / "tuple_support.json", result)


def summarize():
    data = defaultdict(list)
    primary_count = event_count = 0
    for dataset in STATE["cases"]:
        for fold in range(4):
            with np.load(STATE["out"] / dataset / f"fold_{fold}_metrics.npz") as z:
                for key in z:
                    data[key].append(z[key])
            for r in json.loads((STATE["out"] / dataset / f"fold_{fold}_payloads.json").read_text()):
                primary_count += r["primary_integers"]
                event_count += r["event_integers_per_extra_stage"]
    ex = np.concatenate(data.pop("examples"))
    assert len(ex) == 17040 and primary_count == 511200
    reference = json.loads(
        (STATE["run"] / "component_source_transfer/integer_sum_source_v1/summary.json").read_text()
    )["arms"]["high_transient_source"]["metrics"]
    tuples = json.loads((STATE["out"] / "tuple_support.json").read_text())
    result = {}
    for level in LAYERS:
        metrics = {
            k.split("__", 1)[1]: np.concatenate(v)
            for k, v in data.items()
            if k.startswith(f"L{level}__")
        }
        summary = base_training_aggregate(metrics, ex, STATE["rows"])
        m = summary["metrics"]
        nm = np.ravel(m["nmse"]["macro"])
        ref = np.ravel(reference["nmse"]["macro"])
        gain = np.ravel(m["rms_gain"]["macro"])
        checks = dict(
            total=nm[0] <= 1.05 * ref[0],
            components=np.all(nm[1:] <= 1.1 * ref[1:]),
            datasets=all(
                np.all(np.asarray(v) <= 1.1 * np.asarray(reference["nmse"]["by_dataset"][d]))
                for d, v in m["nmse"]["by_dataset"].items()
            ),
            RMS=np.all((gain >= 0.75) & (gain <= 1.25)),
            bands=all(
                np.max(abs(np.asarray(m[k]["macro"]))) <= 3.0103
                for k in ("band_4_8_db", "band_8_13_db", "band_4_8_abs_db", "band_8_13_abs_db")
            ),
            correlation=min(np.ravel(m["correlation"]["macro"])) > 0,
            envelope=min(np.ravel(m["local_rms_correlation"]["macro"])) >= 0.5,
        )
        bits = 14 * primary_count + 8 * (level - 1) * event_count
        result[f"L{level}"] = dict(
            summary=summary,
            checks=checks,
            waveform_pass=all(checks.values()),
            integers=primary_count + (level - 1) * event_count,
            extra_integers=(level - 1) * event_count,
            paid_bits=bits,
            EEG_pair_bits_per_native_position_second=bits / (17040 * 30),
            local_tuple_support={b: tuples[b][f"L{level}"] for b in "SO"},
        )
    passing = [level for level in LAYERS if result[f"L{level}"]["waveform_pass"]]
    write(
        STATE["out"] / "summary.json",
        dict(
            status="bounded_multi_ID_rate_diagnostic_complete",
            arms=result,
            first_observed_waveform_passing_layer=min(passing) if passing else None,
            formal_single_ID_candidate=False,
            original_arrays=400,
            original_positions=17040,
            original_people=60,
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
        ),
    )


def review():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    import csv

    out = STATE["out"]
    destination = out / "visual_review_v1"
    destination.mkdir(exist_ok=True)
    snapshots = {}
    for dataset in STATE["cases"]:
        for fold in range(4):
            with np.load(out / dataset / f"fold_{fold}_gallery.npz") as z:
                for key in z:
                    snapshots[key] = z[key].copy()
            source = (
                STATE["run"]
                / "component_source_transfer/band_exchange_event_slots_v1"
                / dataset
                / f"fold_{fold}_gallery.npz"
            )
            with np.load(source) as z:
                for identity, c in STATE["selected"].items():
                    index = c["audit_index"]
                    for k in ("S", "O", "sum"):
                        key = f"{index}_slots25_{k}"
                        if key in z:
                            snapshots[f"{index}_control25_{k}"] = z[key].copy()
    save(destination / "display_arrays.npz", **snapshots)
    names = {
        "target": "target",
        "L1": "1 total layer",
        "L2": "2 total layers",
        "L4": "4 total layers",
        "control25": "25% continuous",
    }
    color = {
        "target": "#203d52",
        "L1": "#be823d",
        "L2": "#557fb9",
        "L4": "#197b64",
        "control25": "#8a6da8",
    }
    styles = {"target": "-", "L1": "--", "L2": "-.", "L4": ":", "control25": "-"}
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    figure_records = []
    numbers = []
    hz = np.fft.rfftfreq(400, 1 / 200)
    taxis = np.arange(6000) / 200
    for identity, case in sorted(STATE["selected"].items(), key=lambda z: z[1]["audit_index"]):
        label = case["audit_index"]
        scale = float(case["scale"])
        unit = case["source_unit"]
        waves = {
            "target": {
                b: snapshots[f"{label}_{k}"].reshape(6000) * scale
                for b, k in (("X", "x"), ("S", "s"), ("O", "o"))
            }
        }
        for arm in ("L1", "L2", "L4", "control25"):
            waves[arm] = {
                b: snapshots[f"{label}_{arm}_{k}"].reshape(6000) * scale
                for b, k in (("X", "sum"), ("S", "S"), ("O", "O"))
            }
        valid = snapshots[f"{label}_mask"].reshape(6000).astype(bool)
        peak_time = np.argmax(np.where(valid, np.abs(waves["target"]["X"]), -1)) / 200
        local = (
            min(28, max(0, 2 * np.floor(peak_time / 2))),
            min(30, max(2, 2 * np.floor(peak_time / 2) + 2)),
        )
        fig, axes = plt.subplots(3, 3, figsize=(15, 8.6), layout="constrained")
        for row, branch in enumerate(("X", "S", "O")):
            values = np.array([waves[arm][branch] for arm in waves])
            lo, hi = values.min(), values.max()
            margin = 0.06 * (hi - lo or 1)
            for col, (left, right) in enumerate(((0, 30), local, (13.5, 14.5))):
                ax = axes[row, col]
                for arm in waves:
                    ax.plot(
                        taxis,
                        waves[arm][branch],
                        c=color[arm],
                        ls=styles[arm],
                        lw=0.7 if arm != "target" else 0.85,
                        label=names[arm],
                        alpha=0.75 if arm == "control25" else 1,
                    )
                for edge in np.arange(0, 31, 2):
                    ax.axvline(edge, c="#b8c1c6", ls=":", lw=0.45)
                ax.fill_between(
                    taxis, lo - margin, hi + margin, where=~valid, color="#bababa", alpha=0.35
                )
                ax.set(
                    xlim=(left, right),
                    ylim=(lo - margin, hi + margin),
                    xlabel="Time within 30-s context (s)",
                )
                if col == 0:
                    ax.set_ylabel(branch + " / " + unit)
                if row == 0:
                    ax.set_title(
                        (
                            "Full 30 s",
                            f"2 s at largest |X| event ({peak_time:.3f} s)",
                            "13.5–14.5 s token boundary",
                        )[col]
                    )
            for arm in ("L1", "L2", "L4", "control25"):
                y, pred = waves["target"][branch], waves[arm][branch]
                j = int(np.argmax(np.where(valid, np.abs(y), -1)))
                near = np.arange(max(0, j - 100), min(6000, j + 101))
                near = near[valid[near]]
                k = int(near[np.argmax(np.abs(pred[near]))])
                numbers.append(
                    dict(
                        case=label,
                        dataset=case["dataset"],
                        subject=case["subject"],
                        record=case["record"],
                        position=case["position_name"],
                        arm=arm,
                        branch=branch,
                        RMSE_native=float(np.sqrt(np.mean(np.square(pred[valid] - y[valid])))),
                        target_peak_time_s=j / 200,
                        target_peak=float(y[j]),
                        prediction_same_time=float(pred[j]),
                        prediction_local_abs_peak=float(pred[k]),
                        prediction_local_peak_time_s=k / 200,
                        selected_fraction=None,
                        unit=unit,
                    )
                )
        axes[0, 0].legend(fontsize=7, ncol=2)
        identity_text = f"{case['dataset']} | {case['subject']} | {case['record']} | {case['position_name']} | crop +{case['start_s']}s"
        fig.suptitle(identity_text + " | multi-ID event residual rate diagnostic", fontsize=10)
        stem = f"{label:03d}_{case['dataset']}_{case['subject']}_{case['position_name']}"
        voltage = destination / (stem + "_voltage.png")
        fig.savefig(voltage, dpi=220, facecolor="white")
        plt.close(fig)
        fig, axes = plt.subplots(3, 2, figsize=(14, 8.4), layout="constrained")
        block_valid = valid.reshape(120, 50).all(1)
        patch_valid = valid.reshape(15, 400).all(1)
        for row, branch in enumerate(("X", "S", "O")):
            for j, arm in enumerate(waves):
                y = waves[arm][branch]
                envelope = np.sqrt(np.square(y.reshape(120, 50)).mean(1))
                envelope[~block_valid] = np.nan
                axes[row, 0].plot(
                    np.arange(120) * 0.25 + 0.125,
                    envelope,
                    c=color[arm],
                    ls=styles[arm],
                    lw=0.9,
                    label=names[arm],
                )
                patches = y.reshape(15, 400)[patch_valid]
                power = (
                    np.abs(
                        np.fft.rfft(
                            (patches - patches.mean(1, keepdims=True)) * np.hanning(400), axis=1
                        )
                    )
                    ** 2
                    * 2
                    / (400 * np.square(np.hanning(400)).sum())
                )
                energy = [
                    power[:, (hz >= a) & ((hz <= b) if b == 45 else (hz < b))].sum(1).mean()
                    for a, b in ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
                ]
                axes[row, 1].bar(
                    np.arange(5) + (j - 2) * 0.16,
                    np.maximum(energy, 1e-25),
                    0.15,
                    color=color[arm],
                    label=names[arm],
                )
            axes[row, 0].set(
                xlim=(0, 30),
                xlabel="Time within 30-s context (s)",
                ylabel=branch + " quarter-second RMS / " + unit,
            )
            axes[row, 1].set(
                xticks=np.arange(5),
                xticklabels=["1–4", "4–8", "8–13", "13–30", "30–45"],
                yscale="log",
                xlabel="Frequency band (Hz)",
                ylabel=branch + " mean complete-patch power / unit²",
            )
        axes[0, 0].legend(fontsize=7, ncol=2)
        axes[0, 1].legend(fontsize=7, ncol=2)
        fig.suptitle(identity_text + " | complete-patch bands and quarter-second RMS", fontsize=10)
        dynamics = destination / (stem + "_dynamics.png")
        fig.savefig(dynamics, dpi=220, facecolor="white")
        plt.close(fig)
        figure_records.append(
            dict(
                identity=case,
                voltage=voltage.name,
                dynamics=dynamics.name,
                mask_valid_points=int(valid.sum()),
                status="rendered_pending_actual_view",
            )
        )
    with (destination / "example_metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(numbers[0]))
        writer.writeheader()
        writer.writerows(numbers)
    write(
        destination / "figure_manifest.json",
        dict(
            examples=figure_records,
            figure_count=2 * len(figure_records),
            status="rendered_pending_actual_view",
            actual_interface="1/2/4 total layers add 0/1/3 eight-bit integers only at fixed event slots; continuous is a capacity control",
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
        ),
    )
    for dataset in STATE["cases"]:
        chosen = [r for r in figure_records if r["identity"]["dataset"] == dataset]
        for kind in ("voltage", "dynamics"):
            thumbnails = []
            for r in chosen:
                im = Image.open(destination / r[kind])
                im.thumbnail((1000, 600))
                canvas = Image.new("RGB", (1020, 630), "white")
                canvas.paste(im, ((1020 - im.width) // 2, 25))
                ImageDraw.Draw(canvas).text((10, 5), str(r["identity"]["audit_index"]), fill="black")
                thumbnails.append(canvas)
            canvas = Image.new("RGB", (1020, 630 * len(thumbnails)), "white")
            for n, im in enumerate(thumbnails):
                canvas.paste(im, (0, 630 * n))
            canvas.save(destination / (dataset + "_" + kind + "_contact.jpg"), quality=90)
    html = "<html><head><meta charset='utf-8'><title>固定事件槽的多ID码率诊断</title></head><body><h1>固定事件槽的多ID码率诊断</h1>"
    html += (
        "<p>每个事件槽发送14bit主字，并按总层数追加0、1或3个8bit残差整数。同轴比较原17例的1、2、4层与25%连续事件控制。</p>"
    )
    for r in figure_records:
        c = r["identity"]
        html += f"<h2>{c['audit_index']} {c['dataset']} {c['subject']} {c['position_name']}</h2><p>{' / '.join(c['selection'])}</p>"
        html += "".join(
            f'<a href="{r[k]}"><img width="100%" src="{r[k]}"></a>' for k in ("voltage", "dynamics")
        )
    (destination / "index.html").write_text(html + "</body></html>")


def initialize(args):
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["schema"] != "band_exchange_event_rvq_v1" or cfg["data"]["validation"] != "forbidden":
        raise ValueError("train-only rate diagnostic required")
    run = ROOT / cfg["joint_run"]
    source = run / cfg["source"]
    population = json.loads(
        (
            run / "component_source_transfer/hard_as_v1/local_information_v1/population.json"
        ).read_text()
    )
    manifest, scales, _, _ = base_tokenizer_training_data(
        yaml.safe_load(
            (
                ROOT
                / "experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml"
            ).read_text()
        ),
        ROOT,
    )
    if (manifest["cfg_identity"], manifest["scales_identity"]) != (
        population["cfg_identity"],
        population["scales_identity"],
    ):
        raise ValueError("training identity changed")
    cases = {
        d: [c for c in population["cases"] if c["dataset"] == d]
        for d in sorted(population["native_positions"])
    }
    for cc in cases.values():
        for c in cc:
            if manifest["rows"][c["crop"]["row"]]["split"] != "train":
                raise ValueError("train required")
    out = args.output.resolve()
    out.mkdir(exist_ok=True, parents=True)
    (out / "books").mkdir(exist_ok=True)
    identity = dict(
        config=cfg,
        source_contract_sha256=hashlib.sha256(
            (source / "resolved_config.json").read_bytes()
        ).hexdigest(),
        input_identity=manifest["cfg_identity"],
        scales_identity=manifest["scales_identity"],
    )
    file = out / "resolved_config.json"
    if file.exists() and json.loads(file.read_text()) != identity:
        raise ValueError("different retained rate contract")
    write(file, identity)
    draws = json.loads(
        (
            run
            / "component_source_transfer/integer_sum_source_v1/training_examples/waveform_examples.json"
        ).read_text()
    )["examples"]
    chosen = {
        tuple(c[k] for k in ("row", "start_s", "position")): dict(c, audit_index=i)
        for i, c in enumerate(draws)
    }
    STATE.update(
        cfg=cfg,
        run=run,
        source=source,
        out=out,
        cases=cases,
        rows=manifest["rows"],
        scales=scales,
        selected=chosen,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--stage",
        choices=("fit", "encode", "score", "support", "summarize", "replay", "review", "all"),
        default="all",
    )
    args = p.parse_args()
    initialize(args)
    stages = (
        ("fit", "encode", "score", "support", "summarize") if args.stage == "all" else (args.stage,)
    )
    for stage in stages:
        start = time.monotonic()
        if stage in ("fit", "encode", "score", "replay"):
            function = {
                "fit": fit_book,
                "encode": encode_dataset,
                "score": score_dataset,
                "replay": score_dataset,
            }[stage]
            tasks = (
                [(f, b, r) for f in range(4) for b in "SO" for r in range(5)]
                if stage == "fit"
                else (
                    [(d, stage == "replay") for d in STATE["cases"]]
                    if stage in ("score", "replay")
                    else list(STATE["cases"])
                )
            )
            with ProcessPoolExecutor(
                max_workers=4, mp_context=multiprocessing.get_context("fork")
            ) as pool:
                list(pool.map(function, tasks))
        elif stage == "support":
            tuple_support()
        elif stage == "review":
            review()
        else:
            summarize()
        write(
            STATE["out"] / (stage + "_complete.json"),
            dict(stage=stage, wall_seconds=time.monotonic() - start),
        )


if __name__ == "__main__":
    main()
