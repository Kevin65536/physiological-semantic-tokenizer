"""Measure sparse X-selected continuous band replacements at frozen EEG integer slots."""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/scripts")]
from src.analysis.dual_component_tokenizer import (
    base_training_aggregate,
    joint_reconstruction_metrics,
)
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_training_data
from src.tokenizers.adaptive_band_allocation import synthesize
from src.tokenizers.band_event_dictionary import propagation_scores, select_slots

STATE = {}
ARMS = {"hard": 0.0, "slots10": 0.1, "slots25": 0.25, "continuous": 1.0}


def write(path, value):
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)

    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=convert) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, **fields):
    temporary = path.with_suffix(".pending.npz")
    np.savez_compressed(temporary, **fields)
    temporary.replace(path)


def budget():
    unit = os.environ.get("BAND_EVENT_UNIT")
    if unit:
        used = (
            int(
                subprocess.check_output(
                    ["systemctl", "--user", "show", unit, "-p", "CPUUsageNSec", "--value"], text=True
                )
            )
            / 1e9
        )
        if used >= STATE["cfg"]["resources"]["CPU_seconds"] - 15:
            raise RuntimeError("registered event-slot CPU budget reached")


def frozen_case(dataset, fold):
    words, covroot = STATE["words"], STATE["covariance"]
    folder = words / "prepared" / dataset / f"fold_{fold}"
    ready = folder / "ready.json"
    if not ready.is_file():
        raise ValueError("complete prepared source fold required")
    metadata = json.loads((folder / "metadata.json").read_text())
    fitting = {c["subject"] for c in STATE["cases"][dataset] if c["fold"] != fold}
    held = {c["subject"] for c in STATE["cases"][dataset] if c["fold"] == fold}
    if set(metadata["fit_subjects"]) != fitting or set(metadata["held_subjects"]) != held:
        raise ValueError("prepared source fitted on different people")
    if fitting & held:
        raise ValueError("subject overlap")

    def load(name):
        return np.load(folder / (name + ".npy"), mmap_mode="r", allow_pickle=False)

    data = {key: load(key) for key in ("examples", "fold", "native", "seen")}
    for b in "SO":
        data[b] = {key: load(b + "_" + key) for key in ("fixed", "role", "changed_rows", "changed")}
    expected = STATE["example_map"][dataset]
    np.testing.assert_array_equal(data["examples"], expected)
    with np.load(words / dataset / f"fold_{fold}_ids.npz") as z:
        np.testing.assert_array_equal(z["examples"], expected)
        ids = {b: z["exchange_balanced_" + b] for b in "SO"}
    atom_tables = {}
    hashes = {str(ready): digest(ready)}
    for b in "SO":
        tables = []
        for band in range(5):
            path = words / "books" / f"fold_{fold}_{b}_{band}.npz"
            hashes[str(path)] = digest(path)
            with np.load(path) as z:
                meta = json.loads(str(z["metadata"]))
                if (meta["fold"], meta["branch"], meta["role"]) != (fold, b, band):
                    raise ValueError("wrong atomic book")
                tables.append((z["balanced_atoms"].astype(float), z["balanced_gain"].copy()))
        atom_tables[b] = tables
    path = covroot / dataset / f"fold_{fold}_model.npz"
    hashes[str(path)] = digest(path)
    with np.load(path) as z:
        if set(z["fit_people"]) != fitting:
            raise ValueError("wrong covariance people")
        covariance = {b: z[b].copy() for b in "SO"}
    return data, ids, atom_tables, covariance, hashes


def selected_coordinates(data, native_rows):
    flat = (15 * native_rows[:, None] + np.arange(15)[None, :]).ravel()
    values = data["fixed"].reshape(-1, 400)[flat].copy()
    changed = data["changed_rows"]
    if len(changed):
        place = np.searchsorted(changed, flat)
        good = changed[np.minimum(place, len(changed) - 1)] == flat
        values[good] = data["changed"][place[good]]
    return values.reshape(len(native_rows), 15, 400).astype(float)


def atom_coordinates(ids, atom_tables):
    if ids.dtype.kind not in "ui" or ids.min() < 0 or ids.max() > 5115:
        raise ValueError("actual paid tagged word required")
    roles = np.where(ids > 0, (ids.astype(np.int32) - 1) // 1023, -1).astype(np.int8)
    wave = np.zeros((*ids.shape, 400))
    for role, (atoms, gain) in enumerate(atom_tables):
        match = roles == role
        idx = (ids[match] - 1) % 1023
        wave[match] = atoms[idx] * gain[idx, None]
    return roles, wave


def reconstruct(dataset, c, indices, data, ids, atoms, covariance):
    native = np.asarray(c["native"])
    native_rows = indices[native.argsort()]
    seen = np.asarray(data["seen"])[native_rows]
    if seen.ndim != 1 or np.any(data["native"][native_rows] != np.arange(len(native))):
        raise ValueError("native layout mismatch")
    names = STATE["population"]["native_positions"][dataset]
    table = {k.upper(): v for k, v in STATE["scales"]["positions"]["eeg"][dataset].items()}
    predictions = {a: {} for a in ARMS}
    counts = []
    for b in "SO":
        roles, atomic = atom_coordinates(ids[b][native_rows], atoms[b])
        np.testing.assert_array_equal(roles, data[b]["role"][native_rows])
        continuous = selected_coordinates(data[b], native_rows)
        continuous[roles < 0] = 0
        weights = np.array(
            [
                1 / float(np.ravel(table[name.upper()]["target_variance"][b.lower()])[0])
                for name in names
            ]
        )
        weights *= seen
        scores = propagation_scores(continuous, atomic, roles, covariance[b], weights)
        per_arm = {}
        for arm, fraction in ARMS.items():
            changed = select_slots(scores, roles, fraction)
            selected = np.where(changed[..., None], continuous, atomic)
            predictions[arm][b] = synthesize(selected, roles, covariance[b])[native]
            per_arm[arm] = int(changed.sum())
        counts.append(
            dict(
                branch=b,
                available=int((roles >= 0).sum()),
                changed=per_arm,
                total_error_score=float(scores.sum()),
            )
        )
    for arm in ARMS:
        predictions[arm]["sum"] = predictions[arm]["S"] + predictions[arm]["O"]
    return predictions, counts


def dataset_worker(dataset):
    start = time.process_time()
    out = STATE["output"] / dataset
    out.mkdir(exist_ok=True)
    cases = STATE["cases"][dataset]
    cache = OrderedDict()
    identities = []
    for fold in range(4):
        budget()
        output = out / f"fold_{fold}_metrics.npz"
        if output.exists():
            continue
        data, ids, atoms, covariance, hashes = frozen_case(dataset, fold)
        with np.load(STATE["words"] / dataset / f"fold_{fold}_metrics.npz") as f:
            original = f["exchange_balanced_energy__nmse"]
            original_ex = f["examples"]
        with np.load(STATE["continuous"] / dataset / f"fold_{fold}_metrics.npz") as f:
            original_continuous = f["exchange__nmse"]
            continuous_ex = f["examples"]
        np.testing.assert_array_equal(original_ex, continuous_ex)
        stats = defaultdict(list)
        counts = []
        offset = 0
        held_offset = 0
        gallery = {}
        first_start = time.process_time()
        for c in cases:
            take = np.arange(offset, offset + c["count"])
            offset += c["count"]
            if c["fold"] != fold:
                continue
            budget()
            ex = np.asarray(data["examples"][take])
            if any(STATE["rows"][int(i)]["split"] != "train" for i, t, p in ex):
                raise ValueError("train split required before target read")
            predictions, changed = reconstruct(dataset, c, take, data, ids, atoms, covariance)
            batch = base_tokenizer_batch(
                STATE["rows"], ex, STATE["scales"], array_cache=cache, cache_limit=8
            )
            factors = np.array(
                [
                    STATE["scales"]["positions"]["eeg"][dataset][
                        STATE["rows"][int(i)]["positions"][int(p)]
                    ]["scale"]
                    for i, t, p in ex
                ]
            )
            stats["examples"].append(ex)
            for arm, values in predictions.items():
                measured = joint_reconstruction_metrics(values, batch, 1, factors)
                if measured["support"]["nonfinite_prediction_points"].any():
                    raise ValueError("nonfinite waveform")
                for key, value in measured["metrics"].items():
                    stats[arm + "__" + key].append(value)
                interval = slice(held_offset, held_offset + c["count"])
                if arm == "hard":
                    np.testing.assert_array_equal(measured["metrics"]["nmse"], original[interval])
                if arm == "continuous":
                    np.testing.assert_allclose(
                        measured["metrics"]["nmse"],
                        original_continuous[interval],
                        rtol=5e-6,
                        atol=5e-7,
                    )
                for j, e in enumerate(ex):
                    if tuple(e) in STATE["selected"]:
                        label = STATE["selected"][tuple(e)]["audit_index"]
                        for k, w in values.items():
                            gallery[f"{label}_{arm}_{k}"] = w[j]
                        if arm == "hard":
                            for k in ("x", "s", "o", "mask"):
                                gallery[f"{label}_{k}"] = batch[k][j]
            counts.append(
                dict(
                    row=c["crop"]["row"],
                    start_s=c["crop"]["start_s"],
                    subject=c["subject"],
                    branches=changed,
                )
            )
            held_offset += c["count"]
            if held_offset == c["count"]:
                write(
                    out / f"fold_{fold}_benchmark.json",
                    dict(
                        first_native_case_CPU_seconds=time.process_time() - first_start,
                        positions=c["count"],
                        arms=4,
                    ),
                )
            write(
                out / "progress.json",
                dict(
                    fold=fold,
                    positions_done=held_offset,
                    worker_CPU_seconds=time.process_time() - start,
                ),
            )
        np.testing.assert_array_equal(np.concatenate(stats["examples"]), original_ex)
        save(out / f"fold_{fold}_gallery.npz", **gallery)
        save(output, **{k: np.concatenate(v) for k, v in stats.items()})
        write(out / f"fold_{fold}_selection.json", counts)
        identities.append(dict(fold=fold, hashes=hashes, positions=held_offset))
    write(
        out / "completion.json",
        dict(dataset=dataset, source_identities=identities, CPU_seconds=time.process_time() - start),
    )
    return dataset


def summarize():
    arrays = defaultdict(list)
    for dataset in STATE["cases"]:
        for f in range(4):
            with np.load(STATE["output"] / dataset / f"fold_{f}_metrics.npz") as z:
                for key in z:
                    arrays[key].append(z[key])
    examples = np.concatenate(arrays.pop("examples"))
    if sorted(map(tuple, examples)) != sorted(
        map(tuple, np.concatenate(list(STATE["example_map"].values())))
    ):
        raise ValueError("population changed")
    ref = json.loads((STATE["reference"] / "summary.json").read_text())["arms"][
        "high_transient_source"
    ]["metrics"]
    records = {}
    for arm in ARMS:
        metrics = {
            k.split("__", 1)[1]: np.concatenate(v)
            for k, v in arrays.items()
            if k.startswith(arm + "__")
        }
        values = base_training_aggregate(metrics, examples, STATE["rows"])
        m = values["metrics"]
        nm = np.ravel(m["nmse"]["macro"])
        rn = np.ravel(ref["nmse"]["macro"])
        gain = np.ravel(m["rms_gain"]["macro"])
        checks = dict(
            total=nm[0] <= 1.05 * rn[0],
            components=bool(np.all(nm[1:] <= 1.1 * rn[1:])),
            datasets=all(
                np.all(np.asarray(v) <= 1.1 * np.asarray(ref["nmse"]["by_dataset"][d]))
                for d, v in m["nmse"]["by_dataset"].items()
            ),
            RMS=bool(np.all((gain >= 0.75) & (gain <= 1.25))),
            bands=all(
                np.max(abs(np.asarray(m[k]["macro"]))) <= 3.0103
                for k in ("band_4_8_db", "band_8_13_db", "band_4_8_abs_db", "band_8_13_abs_db")
            ),
            correlation=min(np.ravel(m["correlation"]["macro"])) > 0,
            envelope=min(np.ravel(m["local_rms_correlation"]["macro"])) >= 0.5,
        )
        records[arm] = dict(summary=values, checks=checks, waveform_pass=all(checks.values()))
    positive = [a for a in ("slots10", "slots25") if records[a]["waveform_pass"]]
    write(
        STATE["output"] / "summary.json",
        dict(
            status="capacity_diagnostic_complete",
            arms=records,
            selected_sparse_arms=positive,
            finite_word_codec=False,
            decision="review_finite_dictionary_design"
            if positive
            else "stop_sparse_continuous_slot_mechanism",
            original_positions=len(examples),
            original_arrays=400,
            original_people=60,
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
        ),
    )


def initialize(args):
    cfg = yaml.safe_load(args.config.read_text())
    if (
        cfg["schema"] != "band_exchange_event_slots_v1"
        or cfg["data"]["validation_access"] != "forbidden"
    ):
        raise ValueError("registered train-only diagnostic required")
    run = ROOT / cfg["source"]["joint_run"]
    poppath = run / cfg["source"]["population"]
    population = json.loads(poppath.read_text())
    g0 = yaml.safe_load(
        (
            ROOT / "experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml"
        ).read_text()
    )
    manifest, scales, _, _ = base_tokenizer_training_data(g0, ROOT)
    if (manifest["cfg_identity"], manifest["scales_identity"]) != (
        population["cfg_identity"],
        population["scales_identity"],
    ):
        raise ValueError("training coordinate changed")
    cases = {
        d: [c for c in population["cases"] if c["dataset"] == d]
        for d in sorted(population["native_positions"])
    }
    all_ex = {
        d: np.array(
            [(c["crop"]["row"], c["crop"]["start_s"], p) for c in cc for p in c["crop"]["positions"]]
        )
        for d, cc in cases.items()
    }
    if sum(map(len, all_ex.values())) != 17040 or any(
        manifest["rows"][i]["split"] != "train" for ex in all_ex.values() for i, t, p in ex
    ):
        raise ValueError("registered 17040 train positions required")
    words = run / cfg["source"]["words"]
    info = json.loads((words / "single_word_information.json").read_text())["exchange_balanced"]
    if not all(info[b]["word"]["passed"] for b in "SO"):
        raise ValueError("frozen words lack their original meaning prerequisite")
    original = json.loads(
        (
            run
            / "component_source_transfer/integer_sum_source_v1/training_examples/waveform_examples.json"
        ).read_text()
    )["examples"]
    selected = {
        tuple(e[k] for k in ("row", "start_s", "position")): dict(e, audit_index=j)
        for j, e in enumerate(original)
    }
    out = args.output.resolve()
    out.mkdir(exist_ok=True, parents=True)
    identity = dict(
        config=cfg,
        population_sha256=digest(poppath),
        codebook_contract_sha256=digest(words / "word_contract.json"),
        input_identity=manifest["cfg_identity"],
        scales_identity=manifest["scales_identity"],
    )
    identity_path = out / "resolved_config.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("different existing event-slot contract")
    write(identity_path, identity)
    STATE.update(
        cfg=cfg,
        run=run,
        population=population,
        cases=cases,
        example_map=all_ex,
        rows=manifest["rows"],
        scales=scales,
        words=words,
        covariance=run / cfg["source"]["covariance"],
        continuous=run / cfg["source"]["continuous_reference"],
        reference=run / cfg["source"]["waveform_reference"],
        output=out,
        selected=selected,
    )


def review():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    import csv

    out = STATE["output"]
    destination = out / "visual_review_v1"
    destination.mkdir(exist_ok=True)
    snapshots = {}
    for dataset, cases in STATE["cases"].items():
        for fold in range(4):
            with np.load(out / dataset / f"fold_{fold}_gallery.npz") as z:
                for key in z:
                    snapshots[key] = z[key].copy()
        candidates = []
        scores = []
        for fold in range(4):
            with np.load(out / dataset / f"fold_{fold}_metrics.npz") as z:
                candidates.extend(z["examples"])
                scores.extend(z["slots25__nmse"].mean((1, 2)))
        ex = tuple(map(int, candidates[int(np.argmax(scores))]))
        if ex not in STATE["selected"]:
            i, t, pos = ex
            row = STATE["rows"][i]
            STATE["selected"][ex] = dict(
                modality="eeg",
                dataset=dataset,
                subject=row["subject"],
                record=row["record"],
                parent=row["parent"],
                row=i,
                start_s=t,
                position=pos,
                position_name=row["positions"][pos],
                source_unit=row["source_unit"],
                scale=STATE["scales"]["positions"]["eeg"][dataset][row["positions"][pos]]["scale"],
                selection=["new_slots25_joint_worst"],
                audit_index=len(STATE["selected"]),
            )
    grouped = defaultdict(list)
    for identity, case in STATE["selected"].items():
        if f"{case['audit_index']}_x" not in snapshots:
            grouped[case["dataset"]].append((identity, case))
    for dataset, chosen in grouped.items():
        cases = STATE["cases"][dataset]
        offsets = {}
        offset = 0
        for c in cases:
            offsets[c["crop"]["row"], c["crop"]["start_s"]] = offset
            offset += c["count"]
        for identity, case in chosen:
            i, t, pos = identity
            c = next(c for c in cases if (c["crop"]["row"], c["crop"]["start_s"]) == (i, t))
            data, ids, atoms, cov, _ = frozen_case(dataset, c["fold"])
            start = offsets[i, t]
            take = np.arange(start, start + c["count"])
            predictions, _ = reconstruct(dataset, c, take, data, ids, atoms, cov)
            ex = np.array(data["examples"][take])
            j = np.flatnonzero(ex[:, 2] == pos).item()
            batch = base_tokenizer_batch(STATE["rows"], ex, STATE["scales"])
            label = case["audit_index"]
            for k in ("x", "s", "o", "mask"):
                snapshots[f"{label}_{k}"] = batch[k][j]
            for arm, waves in predictions.items():
                for k, v in waves.items():
                    snapshots[f"{label}_{arm}_{k}"] = v[j]
    save(destination / "display_arrays.npz", **snapshots)
    names = {
        "hard": "0% hard",
        "slots10": "10% continuous",
        "slots25": "25% continuous",
        "continuous": "100% continuous",
        "target": "target",
    }
    color = {
        "hard": "#be823d",
        "slots10": "#557fb9",
        "slots25": "#197b64",
        "continuous": "#8a6da8",
        "target": "#203d52",
    }
    styles = {"hard": "--", "slots10": "-.", "slots25": ":", "continuous": "-", "target": "-"}
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
        for arm in ARMS:
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
                        alpha=0.75 if arm == "continuous" else 1,
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
            for arm in ARMS:
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
                        selected_fraction=ARMS[arm],
                        unit=unit,
                    )
                )
        axes[0, 0].legend(fontsize=7, ncol=2)
        identity_text = f"{case['dataset']} | {case['subject']} | {case['record']} | {case['position_name']} | crop +{case['start_s']}s"
        fig.suptitle(identity_text + " | X-selected continuous replacement diagnostic", fontsize=10)
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
            actual_interface="continuous replacements beyond original paid IDs, not a finite-word model",
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
    html = "<html><head><meta charset='utf-8'><title>事件槽容量诊断</title></head><body><h1>仅训练被试，固定连续替换预算</h1>"
    html += "<p>目标、0%、10%、25%、100%在相同坐标，保留原17例及各库25%最差。连续替换为容量诊断。</p>"
    for r in figure_records:
        c = r["identity"]
        html += f"<h2>{c['audit_index']} {c['dataset']} {c['subject']} {c['position_name']}</h2><p>{' / '.join(c['selection'])}</p>"
        html += "".join(
            f'<a href="{r[k]}"><img width="100%" src="{r[k]}"></a>' for k in ("voltage", "dynamics")
        )
    (destination / "index.html").write_text(html + "</body></html>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review", action="store_true")
    args = parser.parse_args()
    initialize(args)
    if args.review:
        review()
        return
    start = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=STATE["cfg"]["resources"]["workers"],
        mp_context=multiprocessing.get_context("fork"),
    ) as pool:
        list(pool.map(dataset_worker, STATE["cases"]))
    summarize()
    write(
        STATE["output"] / "completion.json",
        dict(
            status="numerical_complete",
            wall_seconds=time.monotonic() - start,
            fractions=ARMS,
            workers=4,
            numerical_threads=1,
            seed="no_randomness",
            stage="sparse_continuous_capacity_diagnostic",
        ),
    )


if __name__ == "__main__":
    main()
