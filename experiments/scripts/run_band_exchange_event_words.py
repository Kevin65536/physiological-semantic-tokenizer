"""Train finite EEG event words on the frozen X-selected source slots."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
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
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit, feature_means
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_training_data
from src.tokenizers.adaptive_band_allocation import synthesize
from src.tokenizers.band_event_dictionary import (
    EVENT_CAPACITY,
    ORIGINAL_SIZE,
    VOCABULARY,
    decode_event_coordinates,
    encode_events,
    event_word_ids,
    fit_event_dictionary,
    propagation_scores,
    select_slots,
    word_roles,
)
from src.tokenizers.band_exchange_words import nearest

STATE = {}
MODES = ("wave", "shape")
ARMS = ("wave_mean", "wave_energy", "shape_mean", "shape_energy", "event_nn", "event_pca")
FIELDS = {"theta": (0, 1), "alpha": (1, 2), "RMS": (2, 10)}


def write(path, value):
    def convert(a):
        if isinstance(a, np.ndarray):
            return a.tolist()
        if isinstance(a, np.generic):
            return a.item()
        raise TypeError(type(a).__name__)

    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=convert) + "\n")


def save(path, **fields):
    temporary = path.with_suffix(".pending.npz")
    np.savez_compressed(temporary, **fields)
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def budget():
    unit = os.environ.get("EVENT_WORD_UNIT")
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
            raise RuntimeError("registered finite-event CPU budget reached")


@lru_cache(maxsize=8)
def books(fold, branch):
    old = []
    for role in range(5):
        with np.load(STATE["old"] / "books" / f"fold_{fold}_{branch}_{role}.npz") as z:
            m = json.loads(str(z["metadata"]))
            if (m["fold"], m["branch"], m["role"]) != (fold, branch, role):
                raise ValueError("wrong old book")
            old.append((z["balanced_atoms"].astype(float), z["balanced_gain"].copy()))
    return old


@lru_cache(maxsize=8)
def new_books(fold, branch):
    result = []
    for role in range(5):
        with np.load(STATE["out"] / "books" / f"fold_{fold}_{branch}_{role}.npz") as z:
            result.append({k: z[k].copy() for k in z if k != "metadata"})
    return result


def prepared(folder, branch, indices):
    fixed = np.load(folder / (branch + "_fixed.npy"), mmap_mode="r")
    rows = np.load(folder / (branch + "_changed_rows.npy"), mmap_mode="r")
    changed = np.load(folder / (branch + "_changed.npy"), mmap_mode="r")
    flat = (15 * indices[:, None] + np.arange(15)[None, :]).ravel()
    wave = fixed.reshape(-1, 400)[flat].copy()
    if len(rows):
        location = np.searchsorted(rows, flat)
        good = rows[np.minimum(location, len(rows) - 1)] == flat
        wave[good] = changed[location[good]]
    return wave.reshape(len(indices), 15, 400).astype(float)


def corpus_dataset(dataset):
    cases = STATE["cases"][dataset]
    dest = STATE["out"] / dataset
    dest.mkdir(exist_ok=True)
    names = STATE["population"]["native_positions"][dataset]
    references = {k.upper(): v for k, v in STATE["scales"]["positions"]["eeg"][dataset].items()}
    expected = STATE["examples"][dataset]
    start = time.process_time()
    for fold in range(4):
        budget()
        if all((dest / f"fold_{fold}_{b}_corpus.npz").exists() for b in "SO"):
            continue
        fit = [c for cc in STATE["cases"].values() for c in cc if c["fold"] != fold]
        record = Counter(
            (c["dataset"], c["subject"], STATE["rows"][c["crop"]["row"]]["record"]) for c in fit
        )
        per_subject = Counter(k[:2] for k in record)
        per_dataset = Counter(k[0] for k in per_subject)
        folder = STATE["old"] / "prepared" / dataset / f"fold_{fold}"
        if not (folder / "ready.json").exists():
            raise ValueError("incomplete source prep")
        meta = json.loads((folder / "metadata.json").read_text())
        fitting = {c["subject"] for c in cases if c["fold"] != fold}
        held = {c["subject"] for c in cases if c["fold"] == fold}
        if (
            fitting & held
            or set(meta["fit_subjects"]) != fitting
            or set(meta["held_subjects"]) != held
        ):
            raise ValueError("source-fold subject boundary mismatch")
        np.testing.assert_array_equal(np.load(folder / "examples.npy"), expected)
        seen = np.load(folder / "seen.npy", mmap_mode="r")
        with np.load(STATE["covariance"] / dataset / f"fold_{fold}_model.npz") as z:
            if set(z["fit_people"]) != fitting:
                raise ValueError("wrong covariance fit")
            cov = {b: z[b].copy() for b in "SO"}
        with np.load(STATE["old"] / dataset / f"fold_{fold}_ids.npz") as z:
            np.testing.assert_array_equal(z["examples"], expected)
            old_ids = {b: z["exchange_balanced_" + b].copy() for b in "SO"}
        branch_data = {b: defaultdict(list) for b in "SO"}
        offset = 0
        for c in cases:
            take = np.arange(offset, offset + c["count"])
            offset += c["count"]
            native = np.asarray(c["native"])
            native_rows = take[native.argsort()]
            available = seen[native_rows]
            identity = (dataset, c["subject"], STATE["rows"][c["crop"]["row"]]["record"])
            weight = (
                (1 / (per_dataset[dataset] * per_subject[identity[:2]] * record[identity]))
                if c["fold"] != fold
                else 0.0
            )
            for b in "SO":
                code = old_ids[b][native_rows]
                roles = np.where(code > 0, (code.astype(np.int32) - 1) // 1023, -1).astype(np.int8)
                source = prepared(folder, b, native_rows)
                source[roles < 0] = 0
                atomic = np.zeros(source.shape)
                for role, (atom, gain) in enumerate(books(fold, b)):
                    keep = roles == role
                    j = (code[keep] - 1) % 1023
                    atomic[keep] = atom[j] * gain[j, None]
                weights = (
                    np.array(
                        [
                            1
                            / float(
                                np.ravel(references[name.upper()]["target_variance"][b.lower()])[0]
                            )
                            for name in names
                        ]
                    )
                    * available
                )
                score = propagation_scores(source, atomic, roles, cov[b], weights)
                chosen = select_slots(score, roles, 0.25)
                native_slot = np.argwhere(chosen)
                scalar = 15 * native_rows[native_slot[:, 0]] + native_slot[:, 1]
                fields = branch_data[b]
                fields["wave"].append(source[chosen].astype(np.float32))
                fields["role"].append(roles[chosen])
                fields["flat"].append(scalar)
                fields["fit"].append(np.full(len(scalar), c["fold"] != fold))
                fields["weight"].append(
                    weight * weights[native_slot[:, 0]] / max(int(available.sum()) * 15, 1)
                )
        for b in "SO":
            arrays = {k: np.concatenate(v) for k, v in branch_data[b].items()}
            order = np.argsort(arrays["flat"])
            arrays = {k: v[order] for k, v in arrays.items()}
            if np.any(np.diff(arrays["flat"]) <= 0):
                raise ValueError("duplicated event slot")
            save(dest / f"fold_{fold}_{b}_corpus.npz", **arrays)
    write(dest / "corpus_complete.json", dict(CPU_seconds=time.process_time() - start))
    return dataset


def fit_book(task):
    fold, branch, role = task
    path = STATE["out"] / "books" / f"fold_{fold}_{branch}_{role}.npz"
    if path.exists():
        return
    budget()
    waves = []
    weights = []
    people = set()
    for dataset in STATE["cases"]:
        with np.load(STATE["out"] / dataset / f"fold_{fold}_{branch}_corpus.npz") as z:
            choose = z["fit"] & (z["role"] == role)
            waves.append(z["wave"][choose])
            weights.append(z["weight"][choose])
            ix = z["flat"][choose] // 15
            people.update(
                (dataset, STATE["rows"][STATE["examples"][dataset][i, 0]]["subject"])
                for i in np.unique(ix)
            )
    wave = np.concatenate(waves)
    weight = np.concatenate(weights)
    if not len(wave):
        raise ValueError("no observed fit event for role")
    seed = STATE["cfg"]["training"]["seed"] + 100 * fold + 10 * (branch == "O") + role
    start = time.process_time()
    fitted = fit_event_dictionary(wave, weight, role, seed)
    save(
        path,
        **fitted,
        metadata=np.array(
            json.dumps(
                dict(
                    fold=fold,
                    branch=branch,
                    role=role,
                    seed=seed,
                    fit_events=len(wave),
                    fit_people=sorted(people),
                    CPU_seconds=time.process_time() - start,
                )
            )
        ),
    )
    save(STATE["out"] / "banks" / f"fold_{fold}_{branch}_{role}.npz", wave=wave)
    return (fold, branch, role, len(wave))


def encode_dataset(dataset):
    start = time.process_time()
    for fold in range(4):
        budget()
        path = STATE["out"] / dataset / f"fold_{fold}_ids.npz"
        if path.exists():
            continue
        with np.load(STATE["old"] / dataset / f"fold_{fold}_ids.npz") as z:
            ex = z["examples"].copy()
            groups = z["fold"].copy()
            old = {b: z["exchange_balanced_" + b].copy() for b in "SO"}
        fields = {}
        for b in "SO":
            fitted = new_books(fold, b)
            with np.load(STATE["out"] / dataset / f"fold_{fold}_{b}_corpus.npz") as z:
                wave = z["wave"].copy()
                role = z["role"].copy()
                flat = z["flat"].copy()
            nn = np.empty_like(wave)
            pca = np.empty_like(wave)
            paid = {mode: old[b].copy() for mode in MODES}
            for band in range(5):
                choose = role == band
                model = fitted[band]
                for mode in MODES:
                    value = encode_events(wave[choose], model, mode)
                    paid[mode].ravel()[flat[choose]] = event_word_ids(band, value)
                coordinates = (wave[choose] - model["wave_mean"]) @ model["wave_basis"].T
                pca[choose] = coordinates @ model["wave_basis"] + model["wave_mean"]
                # All-fit nearest full-waveform atoms are a favorable high-rate diagnostic.
                with np.load(STATE["out"] / "banks" / f"fold_{fold}_{b}_{band}.npz") as z:
                    bank = z["wave"].copy()
                for begin in range(0, int(choose.sum()), 512):
                    local = wave[choose][begin : begin + 512]
                    locations = nearest(local, bank, block_size=512)
                    index = np.flatnonzero(choose)[begin : begin + 512]
                    nn[index] = bank[locations]
            for mode in MODES:
                fields[mode + "_" + b] = paid[mode]
            save(STATE["out"] / dataset / f"fold_{fold}_{b}_controls.npz", flat=flat, nn=nn, pca=pca)
        save(path, examples=ex, fold=groups, **fields)
        write(
            STATE["out"] / dataset / "encode_progress.json",
            dict(fold=fold, CPU_seconds=time.process_time() - start),
        )
    return dataset


def pack(words):
    bits = ((words.astype(np.uint32).ravel()[:, None] >> np.arange(14)) & 1).astype(np.uint8).ravel()
    return np.packbits(bits, bitorder="little").tobytes()


def unpack(blob, shape):
    values = np.unpackbits(np.frombuffer(blob, dtype=np.uint8), bitorder="little")
    n = int(np.prod(shape))
    return (values[: 14 * n].reshape(n, 14) @ (1 << np.arange(14))).astype(np.uint16).reshape(shape)


def decode_finite(ids, old, new, cov, mode, energy):
    result = {}
    for b in "SO":
        roles, coordinates = decode_event_coordinates(ids[b], old[b], new[b], mode, energy)
        result[b] = synthesize(coordinates, roles, cov[b])
    result["sum"] = result["S"] + result["O"]
    return result


def score_dataset(dataset):
    start = time.process_time()
    out = STATE["out"] / dataset
    cache = OrderedDict()
    cases = STATE["cases"][dataset]
    for fold in range(4):
        budget()
        path = out / f"fold_{fold}_metrics.npz"
        if path.exists():
            continue
        old = {b: books(fold, b) for b in "SO"}
        new = {b: new_books(fold, b) for b in "SO"}
        with np.load(STATE["covariance"] / dataset / f"fold_{fold}_model.npz") as z:
            cov = {b: z[b] for b in "SO"}
        with np.load(out / f"fold_{fold}_ids.npz") as z:
            ex = z["examples"].copy()
            streams = {mode: {b: z[mode + "_" + b].copy() for b in "SO"} for mode in MODES}
        controls = {}
        for b in "SO":
            with np.load(out / f"fold_{fold}_{b}_controls.npz") as z:
                controls[b] = {k: z[k].copy() for k in z}
        stats = defaultdict(list)
        gallery = {}
        offset = 0
        transports = []
        for c in cases:
            take = np.arange(offset, offset + c["count"])
            offset += c["count"]
            if c["fold"] != fold:
                continue
            budget()
            examples = ex[take]
            if any(STATE["rows"][i]["split"] != "train" for i, t, p in examples):
                raise ValueError("train only before signal reads")
            native = np.asarray(c["native"])
            native_rows = take[native.argsort()]
            stats["examples"].append(examples)
            batch = base_tokenizer_batch(
                STATE["rows"], examples, STATE["scales"], array_cache=cache, cache_limit=8
            )
            factor = np.array(
                [
                    STATE["scales"]["positions"]["eeg"][dataset][STATE["rows"][i]["positions"][p]][
                        "scale"
                    ]
                    for i, t, p in examples
                ]
            )
            predictions = {}
            for mode in MODES:
                sent = {}
                for b in "SO":
                    original = streams[mode][b][native_rows]
                    blob = pack(original)
                    sent[b] = unpack(blob, original.shape)
                    np.testing.assert_array_equal(sent[b], original)
                    fname = f"{c['crop']['row']}_{c['crop']['start_s']}_{mode}_{b}.14bit"
                    (out / fname).write_bytes(blob)
                    transports.append(
                        dict(
                            file=fname,
                            shape=list(original.shape),
                            integers=original.size,
                            bytes=len(blob),
                            sha256=hashlib.sha256(blob).hexdigest(),
                        )
                    )
                for energy in (False, True):
                    arm = mode + ("_energy" if energy else "_mean")
                    pred = decode_finite(sent, old, new, cov, mode, energy)
                    predictions[arm] = {k: v[native] for k, v in pred.items()}
            for control in ("nn", "pca"):
                branch = {}
                for b in "SO":
                    base = streams["wave"][b][native_rows].copy()
                    original = base.copy()
                    # Recover the original unselected atoms from the original source stream.
                    with np.load(STATE["old"] / dataset / f"fold_{fold}_ids.npz") as z:
                        original = z["exchange_balanced_" + b][native_rows]
                    roles, coordinates = decode_event_coordinates(
                        original, old[b], new[b], "wave", True
                    )
                    flat = (15 * native_rows[:, None] + np.arange(15)[None, :]).ravel()
                    stored = controls[b]["flat"]
                    locations = np.searchsorted(stored, flat)
                    good = stored[np.minimum(locations, len(stored) - 1)] == flat
                    coordinates.reshape(-1, 400)[good] = controls[b][control][locations[good]]
                    branch[b] = synthesize(coordinates, roles, cov[b])[native]
                branch["sum"] = branch["S"] + branch["O"]
                predictions["event_" + control] = branch
            for arm, pred in predictions.items():
                scored = joint_reconstruction_metrics(pred, batch, 1, factor)
                if scored["support"]["nonfinite_prediction_points"].any():
                    raise ValueError("nonfinite event reconstruction")
                for key, value in scored["metrics"].items():
                    stats[arm + "__" + key].append(value)
                for j, e in enumerate(examples):
                    if tuple(e) in STATE["selected"]:
                        label = STATE["selected"][tuple(e)]["audit_index"]
                        for k, v in pred.items():
                            gallery[f"{label}_{arm}_{k}"] = v[j]
                        if arm == "wave_mean":
                            for k in ("x", "s", "o", "mask"):
                                gallery[f"{label}_{k}"] = batch[k][j]
            write(
                out / "score_progress.json",
                dict(fold=fold, case=c["crop"], CPU_seconds=time.process_time() - start),
            )
        save(out / f"fold_{fold}_metrics.npz", **{k: np.concatenate(v) for k, v in stats.items()})
        save(out / f"fold_{fold}_gallery.npz", **gallery)
        write(out / f"fold_{fold}_transport.json", transports)
    write(out / "score_complete.json", dict(CPU_seconds=time.process_time() - start))
    return dataset


def readout(codes, background, attributes, bank, fold, capacity, bg_size):
    fit = bank["fold"] != fold
    held = ~fit
    valid = bank["valid"]
    if set(bank["subject"][fit]) & set(bank["subject"][held]):
        raise ValueError("pattern people overlap")
    fitmask = valid & fit[:, None]
    heldmask = valid & held[:, None]
    audit = SparsePatternAudit(capacity, bg_size, [], 10, FIELDS)
    audit.fit(codes[fitmask], background[fitmask], attributes[fitmask])
    persons = np.zeros(capacity, np.int32)
    for p in np.unique(bank["subject"][fit]):
        seen = valid & (bank["subject"][:, None] == p)
        persons[np.unique(codes[seen])] += 1
    background_prediction, _, prediction = audit.read(codes[heldmask], background[heldmask])
    target = attributes[heldmask]
    error = np.zeros((len(codes), 15, 2, 3))
    error[heldmask] = np.stack(
        [feature_means(np.square(q - target), FIELDS) for q in (background_prediction, prediction)],
        axis=1,
    )
    count = heldmask.sum(1)
    perwindow = error.sum(1) / count.clip(1)[:, None, None]
    perwindow[count == 0] = np.nan
    supported = (audit.local.count[codes[heldmask]] >= 8) & (persons[codes[heldmask]] >= 2)
    return (
        perwindow[held],
        dict(complete=int(heldmask.sum()), qualified=int(supported.sum())),
        audit.local.count,
        persons,
    )


def information():
    source = STATE["run"] / "component_source_transfer/hard_as_v1/local_information_v1"
    with np.load(source / "training_attributes.npz") as z:
        bank = {k: z[k].copy() for k in z}
    expected = np.concatenate(list(STATE["examples"].values()))
    np.testing.assert_array_equal(expected, bank["examples"])
    backgrounds = len(STATE["population"]["background_sites"]) * 4
    result = {}
    for mode in MODES:
        result[mode] = {}
        for b in "SO":
            errors = {
                k: np.empty((len(expected), 2, 3))
                for k in ("word", "role", "role_event", "within_role_event")
            }
            supports = {k: dict(complete=0, qualified=0) for k in errors}
            usage = {}
            for fold in range(4):
                budget()
                paid = []
                for dataset in STATE["cases"]:
                    with np.load(STATE["out"] / dataset / f"fold_{fold}_ids.npz") as z:
                        paid.append(z[mode + "_" + b].copy())
                words = np.concatenate(paid)
                roles, events = word_roles(words)
                tag = roles + 1
                event_role = tag + 6 * events
                cases = (
                    ("word", words, bank["background"], VOCABULARY, backgrounds),
                    ("role", tag, bank["background"], 6, backgrounds),
                    ("role_event", event_role, bank["background"], 12, backgrounds),
                    (
                        "within_role_event",
                        words,
                        12 * bank["background"] + event_role,
                        VOCABULARY,
                        12 * backgrounds,
                    ),
                )
                for label, codes, background, k, n in cases:
                    error, support, count, persons = readout(
                        codes, background, bank[b + "_attributes"], bank, fold, k, n
                    )
                    errors[label][bank["fold"] == fold] = error
                    for key in support:
                        supports[label][key] += support[key]
                    if label == "word":
                        usage[f"fit_count_{fold}"] = count
                        usage[f"fit_people_{fold}"] = persons
                        mask = bank["valid"] & (bank["fold"] == fold)[:, None]
                        usage[f"held_count_{fold}"] = np.bincount(words[mask], minlength=VOCABULARY)
            record = {}
            for label, values in errors.items():
                hierarchy = base_training_aggregate({"nmse": values}, expected, STATE["rows"])[
                    "metrics"
                ]["nmse"]
                m = np.asarray(hierarchy["macro"])
                gain = 1 - m[1] / m[0]
                positive = np.sum(
                    [
                        np.asarray(v["value"])[1] < np.asarray(v["value"])[0]
                        for v in hierarchy["by_subject"]
                    ],
                    axis=0,
                )
                support = supports[label]["qualified"] / supports[label]["complete"]
                record[label] = dict(
                    gain=gain,
                    positive_people=positive,
                    support=support,
                    hierarchy=hierarchy,
                    passed=bool(
                        min(gain[:2]) > 0
                        and max(gain) >= 0.05
                        and min(positive[:2]) >= 30
                        and support >= 0.9
                    ),
                )
            result[mode][b] = record
            save(
                STATE["out"] / f"{mode}_{b}_word_information.npz",
                examples=expected,
                **errors,
                **usage,
            )
    write(STATE["out"] / "single_word_information.json", result)


def summarize():
    all_data = defaultdict(list)
    for dataset in STATE["cases"]:
        for fold in range(4):
            with np.load(STATE["out"] / dataset / f"fold_{fold}_metrics.npz") as z:
                for key in z:
                    all_data[key].append(z[key])
    ex = np.concatenate(all_data.pop("examples"))
    if len(ex) != 17040 or sorted(map(tuple, ex)) != sorted(
        map(tuple, np.concatenate(list(STATE["examples"].values())))
    ):
        raise ValueError("event score population changed")
    reference = json.loads(
        (STATE["run"] / "component_source_transfer/integer_sum_source_v1/summary.json").read_text()
    )["arms"]["high_transient_source"]["metrics"]
    info = json.loads((STATE["out"] / "single_word_information.json").read_text())
    results = {}
    for arm in ARMS:
        metrics = {
            k.split("__", 1)[1]: np.concatenate(v)
            for k, v in all_data.items()
            if k.startswith(arm + "__")
        }
        stats = base_training_aggregate(metrics, ex, STATE["rows"])
        m = stats["metrics"]
        nm = np.ravel(m["nmse"]["macro"])
        ref = np.ravel(reference["nmse"]["macro"])
        rms = np.ravel(m["rms_gain"]["macro"])
        checks = dict(
            total=nm[0] <= 1.05 * ref[0],
            components=np.all(nm[1:] <= 1.1 * ref[1:]),
            datasets=all(
                np.all(np.asarray(v) <= 1.1 * np.asarray(reference["nmse"]["by_dataset"][d]))
                for d, v in m["nmse"]["by_dataset"].items()
            ),
            RMS=np.all((rms >= 0.75) & (rms <= 1.25)),
            bands=all(
                np.max(abs(np.asarray(m[k]["macro"]))) <= 3.0103
                for k in ("band_4_8_db", "band_8_13_db", "band_4_8_abs_db", "band_8_13_abs_db")
            ),
            correlation=min(np.ravel(m["correlation"]["macro"])) > 0,
            envelope=min(np.ravel(m["local_rms_correlation"]["macro"])) >= 0.5,
        )
        finite = not arm.startswith("event_")
        if finite:
            checks.update({f"word_{b}": info[arm.split("_")[0]][b]["word"]["passed"] for b in "SO"})
        results[arm] = dict(
            summary=stats,
            checks=checks,
            finite_candidate=finite,
            passed=finite and all(checks.values()),
        )
    selected = [a for a, v in results.items() if v["passed"]]
    write(
        STATE["out"] / "summary.json",
        dict(
            status="train_numerical_complete",
            arms=results,
            selected=selected,
            original_arrays=400,
            original_people=60,
            original_positions=17040,
            declared_single_word_K=VOCABULARY,
            fixed_bits_per_word=14,
            integers_per_branch_native_context=15,
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
        exs = []
        scores = []
        for fold in range(4):
            with np.load(out / dataset / f"fold_{fold}_gallery.npz") as z:
                for key in z:
                    snapshots[key] = z[key].copy()
            with np.load(out / dataset / f"fold_{fold}_metrics.npz") as z:
                exs.append(z["examples"])
                scores.append(np.maximum(z["wave_energy__nmse"], z["shape_energy__nmse"]))
        exs = np.concatenate(exs)
        scores = np.concatenate(scores)
        for value, label in (
            (scores.mean((1, 2)), "new_finite_event_joint_worst"),
            (scores[:, 1, 0], "new_finite_event_S_worst"),
        ):
            identity = tuple(map(int, exs[np.argmax(value)]))
            if identity in STATE["selected"]:
                continue
            i, t, p = identity
            row = STATE["rows"][i]
            STATE["selected"][identity] = dict(
                modality="eeg",
                dataset=dataset,
                subject=row["subject"],
                record=row["record"],
                parent=row["parent"],
                row=i,
                start_s=t,
                position=p,
                position_name=row["positions"][p],
                source_unit=row["source_unit"],
                scale=STATE["scales"]["positions"]["eeg"][dataset][row["positions"][p]]["scale"],
                selection=[label],
                audit_index=len(STATE["selected"]),
            )
    for identity, case in STATE["selected"].items():
        index = case["audit_index"]
        if f"{index}_x" in snapshots:
            continue
        dataset = case["dataset"]
        all_cases = STATE["cases"][dataset]
        c = next(c for c in all_cases if (c["crop"]["row"], c["crop"]["start_s"]) == identity[:2])
        fold = c["fold"]
        offset = 0
        for one in all_cases:
            if one is c:
                break
            offset += one["count"]
        take = np.arange(offset, offset + c["count"])
        native = np.array(c["native"])
        native_rows = take[native.argsort()]
        old = {b: books(fold, b) for b in "SO"}
        new = {b: new_books(fold, b) for b in "SO"}
        with np.load(STATE["covariance"] / dataset / f"fold_{fold}_model.npz") as z:
            cov = {b: z[b].copy() for b in "SO"}
        predictions = {}
        with np.load(out / dataset / f"fold_{fold}_ids.npz") as z:
            examples = z["examples"][take]
            j = int(np.flatnonzero(examples[:, 2] == identity[2])[0])
            for mode in MODES:
                ids = {b: z[mode + "_" + b][native_rows] for b in "SO"}
                values = decode_finite(ids, old, new, cov, mode, True)
                predictions[mode + "_energy"] = {k: v[native] for k, v in values.items()}
        with np.load(STATE["old"] / dataset / f"fold_{fold}_ids.npz") as z:
            original = {b: z["exchange_balanced_" + b][native_rows] for b in "SO"}
        for control in ("nn", "pca"):
            waves = {}
            for b in "SO":
                roles, coordinates = decode_event_coordinates(
                    original[b], old[b], new[b], "wave", True
                )
                flat = (15 * native_rows[:, None] + np.arange(15)[None, :]).ravel()
                with np.load(out / dataset / f"fold_{fold}_{b}_controls.npz") as z:
                    stored = z["flat"]
                    location = np.searchsorted(stored, flat)
                    good = stored[np.minimum(location, len(stored) - 1)] == flat
                    coordinates.reshape(-1, 400)[good] = z[control][location[good]]
                waves[b] = synthesize(coordinates, roles, cov[b])[native]
            waves["sum"] = waves["S"] + waves["O"]
            predictions["event_" + control] = waves
        batch = base_tokenizer_batch(STATE["rows"], examples, STATE["scales"])
        for k in ("x", "s", "o", "mask"):
            snapshots[f"{index}_{k}"] = batch[k][j]
        for arm, values in predictions.items():
            for k, v in values.items():
                snapshots[f"{index}_{arm}_{k}"] = v[j]
    save(destination / "display_arrays.npz", **snapshots)
    names = {
        "target": "target",
        "wave_energy": "finite waveform",
        "shape_energy": "finite amplitude-shape",
        "event_nn": "fit-event NN",
        "event_pca": "continuous event PCA",
    }
    color = {
        "target": "#203d52",
        "wave_energy": "#be823d",
        "shape_energy": "#197b64",
        "event_nn": "#557fb9",
        "event_pca": "#8a6da8",
    }
    styles = {
        "target": "-",
        "wave_energy": "--",
        "shape_energy": ":",
        "event_nn": "-.",
        "event_pca": "-",
    }
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
        for arm in ("wave_energy", "shape_energy", "event_nn", "event_pca"):
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
                        alpha=0.75 if arm == "event_pca" else 1,
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
            for arm in ("wave_energy", "shape_energy", "event_nn", "event_pca"):
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
        fig.suptitle(
            identity_text + " | finite K15351 event words and two capacity controls", fontsize=10
        )
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
            actual_interface="finite waveform/amplitude-shape each emit one 14bit word; NN and PCA are capacity diagnostics",
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
    html = "<html><head><meta charset='utf-8'><title>有限事件词的训练留人审计</title></head><body><h1>有限事件词的训练留人审计</h1>"
    html += (
        "<p>普通波形词与幅度分层词均发送一个14bit整数。训练事件最近邻与连续PCA为容量对照。原17例及各库新增最差保留。</p>"
    )
    for r in figure_records:
        c = r["identity"]
        html += f"<h2>{c['audit_index']} {c['dataset']} {c['subject']} {c['position_name']}</h2><p>{' / '.join(c['selection'])}</p>"
        html += "".join(
            f'<a href="{r[k]}"><img width="100%" src="{r[k]}"></a>' for k in ("voltage", "dynamics")
        )
    (destination / "index.html").write_text(html + "</body></html>")


def replay_dataset(dataset):
    dest = STATE["out"] / dataset
    cache = OrderedDict()
    checks = []
    for fold in range(4):
        old = {b: books(fold, b) for b in "SO"}
        new = {b: new_books(fold, b) for b in "SO"}
        with np.load(STATE["covariance"] / dataset / f"fold_{fold}_model.npz") as z:
            cov = {b: z[b].copy() for b in "SO"}
        with np.load(dest / f"fold_{fold}_metrics.npz") as z:
            expected = {k: z[k].copy() for k in z}
        with np.load(dest / f"fold_{fold}_ids.npz") as z:
            ex = z["examples"].copy()
            original = {mode: {b: z[mode + "_" + b].copy() for b in "SO"} for mode in MODES}
        offset = 0
        held = 0
        for c in STATE["cases"][dataset]:
            take = np.arange(offset, offset + c["count"])
            offset += c["count"]
            if c["fold"] != fold:
                continue
            budget()
            native = np.array(c["native"])
            native_rows = take[native.argsort()]
            predictions = {}
            for mode in MODES:
                sent = {}
                for b in "SO":
                    path = dest / f"{c['crop']['row']}_{c['crop']['start_s']}_{mode}_{b}.14bit"
                    sent[b] = unpack(path.read_bytes(), (len(native), 15))
                    np.testing.assert_array_equal(sent[b], original[mode][b][native_rows])
                for energy in (False, True):
                    # Loaded integer arrays and frozen numeric state are sufficient.
                    # Block any attempt to open another NumPy file inside the decoder.
                    original_load = np.load

                    def reject_source(*args, **kwargs):
                        raise AssertionError("integer decoder attempted a file read")

                    np.load = reject_source
                    try:
                        values = decode_finite(sent, old, new, cov, mode, energy)
                    finally:
                        np.load = original_load
                    predictions[mode + ("_energy" if energy else "_mean")] = {
                        k: v[native] for k, v in values.items()
                    }
            examples = ex[take]
            if any(STATE["rows"][i]["split"] != "train" for i, t, p in examples):
                raise ValueError("train only scoring")
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
            np.testing.assert_array_equal(examples, expected["examples"][interval])
            for arm, pred in predictions.items():
                actual = joint_reconstruction_metrics(pred, batch, 1, factors)
                for name, value in actual["metrics"].items():
                    np.testing.assert_array_equal(value, expected[arm + "__" + name][interval])
            held += c["count"]
        checks.append(
            dict(
                fold=fold,
                native_arrays=sum(c["fold"] == fold for c in STATE["cases"][dataset]),
                positions=held,
                arms=4,
                every_metric_exact=True,
                decoder_file_reads=0,
                decoder_inputs="14bit paid words and frozen waveform/covariance state only",
            )
        )
    write(
        dest / "integer_replay.json",
        dict(status="passed", folds=checks, CPU_seconds=time.process_time()),
    )
    return dataset


def initialize(args):
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["schema"] != "band_exchange_event_words_v1" or cfg["data"]["allowed_splits"] != ["train"]:
        raise ValueError("train-only event contract required")
    run = ROOT / cfg["source"]["joint_run"]
    old = run / cfg["source"]["original_words"]
    diagnostic = json.loads((run / cfg["source"]["capacity_diagnostic"] / "summary.json").read_text())
    if "slots25" not in diagnostic["selected_sparse_arms"]:
        raise ValueError("registered sparse mechanism did not pass")
    poppath = run / "component_source_transfer/hard_as_v1/local_information_v1/population.json"
    pop = json.loads(poppath.read_text())
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
        pop["cfg_identity"],
        pop["scales_identity"],
    ):
        raise ValueError("G0 coordinate changed")
    cases = {
        d: [c for c in pop["cases"] if c["dataset"] == d] for d in sorted(pop["native_positions"])
    }
    examples = {
        d: np.array(
            [(c["crop"]["row"], c["crop"]["start_s"], p) for c in cc for p in c["crop"]["positions"]]
        )
        for d, cc in cases.items()
    }
    if sum(map(len, examples.values())) != 17040 or any(
        manifest["rows"][i]["split"] != "train" for ex in examples.values() for i, t, p in ex
    ):
        raise ValueError("train before data reads")
    out = args.output.resolve()
    out.mkdir(exist_ok=True, parents=True)
    (out / "books").mkdir(exist_ok=True)
    (out / "banks").mkdir(exist_ok=True)
    identity = dict(
        config=cfg,
        population_sha256=sha(poppath),
        old_contract_sha256=sha(old / "word_contract.json"),
        input_identity=manifest["cfg_identity"],
        scales_identity=manifest["scales_identity"],
    )
    ip = out / "resolved_config.json"
    if ip.exists() and json.loads(ip.read_text()) != identity:
        raise ValueError("different event model contract")
    write(ip, identity)
    gallery = json.loads(
        (
            run
            / "component_source_transfer/integer_sum_source_v1/training_examples/waveform_examples.json"
        ).read_text()
    )["examples"]
    selected = {
        tuple(c[k] for k in ("row", "start_s", "position")): dict(c, audit_index=i)
        for i, c in enumerate(gallery)
    }
    STATE.update(
        cfg=cfg,
        run=run,
        old=old,
        out=out,
        covariance=run / "component_source_transfer/adaptive_band_v1",
        population=pop,
        cases=cases,
        examples=examples,
        rows=manifest["rows"],
        scales=scales,
        selected=selected,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--stage",
        choices=(
            "corpus",
            "fit",
            "encode",
            "score",
            "information",
            "summarize",
            "replay",
            "review",
            "all",
        ),
        default="all",
    )
    args = p.parse_args()
    initialize(args)
    stages = (
        ("corpus", "fit", "encode", "score", "information", "summarize")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        start = time.monotonic()
        if stage in ("corpus", "fit", "encode", "score", "replay"):
            function = {
                "corpus": corpus_dataset,
                "fit": fit_book,
                "encode": encode_dataset,
                "score": score_dataset,
                "replay": replay_dataset,
            }[stage]
            tasks = (
                [(f, b, r) for f in range(4) for b in "SO" for r in range(5)]
                if stage == "fit"
                else list(STATE["cases"])
            )
            with ProcessPoolExecutor(
                max_workers=4, mp_context=multiprocessing.get_context("fork")
            ) as pool:
                list(pool.map(function, tasks))
        elif stage == "information":
            information()
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
