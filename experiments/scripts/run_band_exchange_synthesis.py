"""Train-only matched cross-spectral synthesis of unchanged paid EEG integers."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/scripts")]
from evaluate_joint_tokenizer import input_scales
from src.analysis.dual_component_tokenizer import base_training_aggregate, joint_reconstruction_metrics
from src.data.dual_component_tokenizer import (
    base_tokenizer_batch, base_tokenizer_joint_eeg, base_tokenizer_training_data,
)
from src.tokenizers.adaptive_band_allocation import band_waves, extract_coordinates, synthesize
from src.tokenizers.band_exchange_words import BandExchangeEEG, decode_native, load_fold_books
from src.tokenizers.band_exchange_synthesis import (
    BandExchangeSynthesis, apply_synthesis, fit_grouped_heads, frequency_indices, load_model,
)
from src.tokenizers.eeg_transient_gain import load_model as load_source

STATE = {}
ARMS = ("hard", "continuous", "hard_montage", "hard_same_site",
        "continuous_montage", "continuous_same_site")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    def scalar(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=scalar) + "\n")
    temporary.replace(path)


def budget():
    unit = os.environ.get("BAND_SYNTHESIS_UNIT")
    if unit:
        used = int(subprocess.check_output(
            ["systemctl", "--user", "show", unit, "-p", "CPUUsageNSec", "--value"], text=True
        )) / 1e9
        used += float(os.environ.get("BAND_SYNTHESIS_PRIOR_CPU", "0"))
        if used > 2050:
            raise RuntimeError("35 CPU-minute synthesis budget reached; retained checkpoints resumable")


def assets(dataset, fold):
    run, parent = STATE["run"], STATE["parent"]
    cases, names = STATE["cases"][dataset], STATE["names"][dataset]
    fit = sorted({c["subject"] for c in cases if c["fold"] != fold})
    held = sorted({c["subject"] for c in cases if c["fold"] == fold})
    if set(fit) & set(held):
        raise ValueError("subject fold overlap")
    source_path = run / "component_source_transfer/transient_gain_v1" / f"{dataset}_fold{fold}_transient.npz"
    cov_path = run / "component_source_transfer/adaptive_band_v1" / dataset / f"fold_{fold}_model.npz"
    source = load_source(source_path)
    if (set(source.metadata["fit_subjects"]) != set(fit)
            or source.metadata["held_fold"] != fold
            or source.metadata["positions"][dataset] != names):
        raise ValueError("wrong outer-fold frozen source")
    with np.load(cov_path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["names"], names)
        if set(z["fit_people"]) != set(fit):
            raise ValueError("wrong outer-fold spatial covariance")
        covariance = {b: z[b].copy() for b in "SO"}
    books = {b: load_fold_books(parent / "books", fold, b) for b in "SO"}
    prepared = parent / "prepared" / dataset / f"fold_{fold}"
    metadata = json.loads((prepared / "metadata.json").read_text())
    if (metadata["source_sha256"] != sha(source_path)
            or metadata["covariance_sha256"] != sha(cov_path)
            or metadata["outer_source_fold"] != fold
            or set(metadata["fit_subjects"]) != set(fit)):
        raise ValueError("same-fold prepared provenance changed")
    ids_path = parent / dataset / f"fold_{fold}_ids.npz"
    with np.load(ids_path, allow_pickle=False) as z:
        ids = {b: z["exchange_balanced_" + b].copy() for b in "SO"}
        examples, folds = z["examples"].copy(), z["fold"].copy()
    for b in "SO":
        tags = np.where(ids[b] == 0, -1, (ids[b].astype(np.int32) - 1) // 1023)
        np.testing.assert_array_equal(tags, np.load(prepared / (b + "_role.npy")))
    scales = STATE["scales"]["positions"]["eeg"][dataset]
    table = {k.upper(): v for k, v in scales.items()}
    weights = {b: np.array([1 / float(np.ravel(table[n.upper()]["target_variance"][b.lower()])[0])
                           for n in names]) for b in "SO"}
    metadata_base = dict(schema="band_exchange_eeg_v1", layout="exchange", assignment="balanced",
                         synthesis="energy", codebook_size=5116, positions={dataset: names},
                         held_fold=fold, fit_subjects=fit, held_subjects=held,
                         input_identity=STATE["population"]["cfg_identity"],
                         scales_identity=STATE["population"]["scales_identity"])
    base = BandExchangeEEG(metadata_base, books, {dataset: covariance}, {dataset: source}, {dataset: weights})
    frozen_paths = [source_path, cov_path, ids_path]
    frozen_paths += [parent / "books" / f"fold_{fold}_{b}_{role}.npz" for b in "SO" for role in range(5)]
    hashes = {str(p): sha(p) for p in frozen_paths}
    return base, ids, examples, folds, metadata, hashes


def load_cases(dataset, fold, benchmark=False):
    base, ids, examples, folds, metadata, hashes = assets(dataset, fold)
    native_names = STATE["names"][dataset]
    cases, blocks, offset = STATE["cases"][dataset], [], 0
    selected = None
    if benchmark:
        selected = {next(i for i, c in enumerate(cases) if c["fold"] == fold),
                    next(i for i, c in enumerate(cases) if c["fold"] != fold)}
    cache = OrderedDict()
    fresh = set()
    timers = Counter()
    for index, case in enumerate(cases):
        take = np.arange(offset, offset + case["count"])
        offset += case["count"]
        if selected is not None and index not in selected:
            continue
        budget()
        crop = case["crop"]
        row = STATE["rows"][crop["row"]]
        if (row["split"], row["modality"], row["dataset"], row["subject"]) != (
                "train", "eeg", dataset, case["subject"]):
            raise ValueError("train boundary before array read")
        ex = np.array([(crop["row"], crop["start_s"], p) for p in crop["positions"]])
        np.testing.assert_array_equal(examples[take], ex)
        np.testing.assert_array_equal(folds[take], np.full(len(take), case["fold"]))
        native = np.asarray(case["native"])
        np.testing.assert_array_equal(np.sort(native), np.arange(len(native_names)))
        order = native.argsort()
        sent = {b: ids[b][take][order] for b in "SO"}
        start = time.process_time()
        hard = decode_native(sent, dataset, base.books, base.covariances[dataset], base.sources[dataset],
                             "balanced", "energy")
        timers["hard_decode"] += time.process_time() - start
        start = time.process_time()
        fields, seen, _ = base_tokenizer_joint_eeg(
            STATE["rows"], dict(crop, positions=[]), STATE["scales"], native_names,
            allowed_splits=("train",),
        )
        x = fields["x"].T.reshape(1, len(native), 15, 400).astype(np.float32)
        provenance = metadata["case_inputs"][index]
        if (hashlib.sha256(x.tobytes()).hexdigest() != provenance["X_sha256"]
                or hashlib.sha256(seen.tobytes()).hexdigest() != provenance["observed_sha256"]):
            raise ValueError("historical encoder X or observation mask changed")
        np.testing.assert_array_equal((sent["S"] != 0).any(1), seen)
        dense = base.sources[dataset].predict_continuous(x, dataset, seen[None])
        continuous = {}
        for b in "SO":
            role = np.where(sent[b] == 0, -1, (sent[b].astype(np.int32) - 1) // 1023)
            fields_b = band_waves(np.fft.rfft(dense[b][0].reshape(len(native), 6000)))
            continuous[b] = synthesize(extract_coordinates(fields_b, role), role, base.covariances[dataset][b])
        continuous["sum"] = continuous["S"] + continuous["O"]
        timers["source_and_continuous"] += time.process_time() - start
        fitting = case["fold"] != fold
        if fitting not in fresh:
            start = time.process_time()
            actual = base.encode(x, dataset, seen[None])
            for b in "SO":
                np.testing.assert_array_equal(actual[b][0], sent[b])
            timers["fresh_encode"] += time.process_time() - start
            fresh.add(fitting)
        start = time.process_time()
        original = base_tokenizer_batch(STATE["rows"], ex, STATE["scales"], array_cache=cache, cache_limit=4)
        batch = {}
        for k, value in original.items():
            value = np.asarray(value)
            if k.startswith("v") or k.startswith("dv"):
                batch[k] = np.broadcast_to(value[:, :1, :1].copy(), value.shape)
            else:
                batch[k] = value.copy()
        del original
        timers["targets"] += time.process_time() - start
        blocks.append(dict(case=case, ex=ex, batch=batch, sent=sent, native=native,
                           hard=hard, continuous=continuous,
                           scales=input_scales(STATE["rows"], ex, STATE["scales"])))
    return base, blocks, timers, hashes


def fit_heads(blocks, fold):
    fit = [v for v in blocks if v["case"]["fold"] != fold]
    sites = len(fit[0]["native"])
    complete = np.stack([v["batch"]["mask"][v["native"].argsort()].all(-1) for v in fit], axis=1)
    patterns, inverse = np.unique(complete.reshape(sites, -1), axis=0, return_inverse=True)
    groups = len(patterns)
    masks = patterns.reshape(groups, len(fit), 15)
    members = [np.r_[np.flatnonzero(inverse == g), sites + np.flatnonzero(inverse == g)] for g in range(groups)]
    weights = np.zeros((groups, len(fit)))
    for g in range(groups):
        available = np.flatnonzero(masks[g].any(-1))
        identities = [(fit[i]["case"]["subject"], STATE["rows"][fit[i]["case"]["crop"]["row"]]["record"])
                      for i in available]
        crop_count = Counter(identities)
        subject_records = Counter(k[0] for k in crop_count)
        for i, key in zip(available, identities, strict=True):
            weights[g, i] = 1 / (len(subject_records) * subject_records[key[0]] * crop_count[key])
    gram = {kind: np.zeros((5, groups, 2 * sites, 2 * sites), complex) for kind in ("hard", "continuous")}
    cross = {kind: np.zeros((5, 2 * sites, 2 * sites), complex) for kind in gram}
    window = np.hanning(400)
    bands = frequency_indices(400)
    for i, block in enumerate(fit):
        budget()
        order = block["native"].argsort()
        targets = np.concatenate([block["batch"][b.lower()][order] for b in "SO"])
        target_fft = np.fft.rfft(targets * window, axis=-1, norm="ortho")
        local = defaultdict(list)
        for g in range(groups):
            if weights[g, i]:
                local[masks[g, i].tobytes()].append(g)
        for kind in gram:
            tracks = np.concatenate([block[kind][b] for b in "SO"])
            spec = np.fft.rfft(tracks * window, axis=-1, norm="ortho")
            for mask_bytes, mask_groups in local.items():
                valid = np.frombuffer(mask_bytes, dtype=bool)
                for band, freq in enumerate(bands):
                    x = spec[:, valid][:, :, freq].reshape(2 * sites, -1).T
                    y = target_fft[:, valid][:, :, freq].reshape(2 * sites, -1).T
                    xx = x.conj().T @ x / len(x)
                    xy = (x.conj().T @ y / len(x)).T
                    for g in mask_groups:
                        gram[kind][band, g] += weights[g, i] * xx
                        cross[kind][band, members[g]] += weights[g, i] * xy[members[g]]
    heads = {}
    for kind in gram:
        fitted = fit_grouped_heads(gram[kind], cross[kind], np.r_[inverse, inverse], sites,
                                   STATE["cfg"]["model"]["ridge_fraction"])
        heads.update({kind + "_" + name: coef for name, coef in fitted.items()})
    diagnostics = dict(fit_crops=len(fit), support_groups=groups,
                       fit_complete_patches=int(complete.sum()), fit_people=sorted({b["case"]["subject"] for b in fit}),
                       partial_target_patches_excluded_from_fit_only=int(complete.size - complete.sum()),
                       ridge_fraction=STATE["cfg"]["model"]["ridge_fraction"])
    return heads, diagnostics


def metrics(block, prediction):
    selected = {b: prediction[b][block["native"]] for b in ("sum", "S", "O")}
    result = joint_reconstruction_metrics(selected, block["batch"], 1, block["scales"])
    if result["support"]["nonfinite_prediction_points"].any():
        raise ValueError("nonfinite synthesis output")
    return result


def dataset_worker(dataset):
    with threadpool_limits(limits=1):
        return run_dataset(dataset)


def run_dataset(dataset):
    start = time.process_time()
    here = STATE["out"] / dataset
    here.mkdir(exist_ok=True)
    benchmark = STATE["stage"] == "benchmark"
    folds = [0] if benchmark else range(4)
    records = []
    for fold in folds:
        budget()
        completion = here / f"fold_{fold}_completion.json"
        if not benchmark and completion.exists():
            records.append(json.loads(completion.read_text()))
            continue
        before = time.process_time()
        base, blocks, timings, hashes = load_cases(dataset, fold, benchmark)
        load_seconds = time.process_time() - before
        before = time.process_time()
        heads, support = fit_heads(blocks, fold)
        fit_seconds = time.process_time() - before
        stats = defaultdict(list)
        gallery = {}
        model_dir = here / f"fold_{fold}"
        model_dir.mkdir(exist_ok=True)
        base_path = model_dir / "frozen_base.npz"
        if not base_path.exists():
            base.save(base_path)
        deploy = {}
        for arm in ("hard_montage", "hard_same_site"):
            model = BandExchangeSynthesis(base, dict(schema="band_exchange_synthesis_v1",
                    positions=base.metadata["positions"], held_fold=fold, arm=arm,
                    fit_subjects=support["fit_people"], extra_bitrate=0), heads[arm])
            path = model_dir / (arm + ".npz")
            if not path.exists():
                model.save(path, base_path)
            deploy[arm] = load_model(path)
        np.savez_compressed(model_dir / "analysis_heads.npz", **heads)
        before = time.process_time()
        held = [v for v in blocks if v["case"]["fold"] == fold]
        replay_cases = replay_points = 0
        baseline = {}
        if not benchmark:
            for kind, directory, prefix in (
                ("hard", STATE["parent"], "exchange_balanced_energy"),
                ("continuous", STATE["run"] / "component_source_transfer/adaptive_band_exchange_v1", "exchange"),
            ):
                with np.load(directory / dataset / f"fold_{fold}_metrics.npz", allow_pickle=False) as z:
                    baseline[kind] = {k: z[prefix + "__" + k].copy() for k in ("nmse", "rms_gain", "band_4_8_db", "band_8_13_db")}
                    np.testing.assert_array_equal(z["examples"], np.concatenate([v["ex"] for v in held]))
        for block in held:
            budget()
            stats["examples"].append(block["ex"])
            for arm in ARMS:
                if arm in ("hard", "continuous"):
                    prediction = block[arm]
                else:
                    prediction = apply_synthesis(block[arm.split("_")[0]], heads[arm])
                result = metrics(block, prediction)
                for k, v in result["metrics"].items():
                    stats[arm + "__" + k].append(v)
                if arm.startswith("hard_"):
                    independent = deploy[arm].decode_ids({b: block["sent"][b][None] for b in "SO"}, dataset)
                    for b in ("S", "O", "sum"):
                        np.testing.assert_array_equal(independent[b][0], prediction[b])
                    repeated = metrics(block, {b: v[0] for b, v in independent.items()})
                    for k in result["metrics"]:
                        np.testing.assert_array_equal(result["metrics"][k], repeated["metrics"][k])
                    replay_cases += 1
                    replay_points += len(block["ex"])
                for j, ex in enumerate(block["ex"]):
                    if tuple(ex) in STATE["display"]:
                        label = STATE["display"][tuple(ex)]
                        for b in ("sum", "S", "O"):
                            gallery[f"{label}_{arm}_{b}"] = prediction[b][block["native"][j]]
                        gallery[f"{label}_mask"] = block["batch"]["mask"][j]
                        gallery[f"{label}_scale"] = block["scales"][j]
                        for b in ("x", "s", "o"):
                            gallery[f"{label}_target_{b}"] = block["batch"][b][j]
        joined = {k: np.concatenate(v) for k, v in stats.items()}
        reproduction = {}
        for kind, expected in baseline.items():
            reproduction[kind] = {}
            for name, reference in expected.items():
                actual = joined[kind + "__" + name]
                gap = float(np.nanmax(abs(actual - reference)))
                np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-10, equal_nan=True)
                reproduction[kind][name] = gap
        for path, expected in hashes.items():
            if sha(path) != expected:
                raise ValueError("original frozen input changed")
        score_seconds = time.process_time() - before
        np.savez_compressed(here / f"fold_{fold}_metrics.npz", **joined)
        np.savez_compressed(here / f"fold_{fold}_gallery.npz", **gallery)
        np.savez_compressed(here / f"fold_{fold}_paid_ids.npz",
                            examples=np.concatenate([v["ex"] for v in held]),
                            **{b: np.concatenate([v["sent"][b][v["native"]] for v in held]) for b in "SO"})
        record = dict(dataset=dataset, fold=fold, cases=len(blocks), held_cases=len(held),
                      support=support, load_CPU_seconds=load_seconds, fit_CPU_seconds=fit_seconds,
                      score_replay_CPU_seconds=score_seconds, details_CPU_seconds=timings,
                      peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      CPU_seconds=time.process_time() - start, baseline_reproduction=reproduction,
                      integer_replay_model_crop_pairs=replay_cases, integer_replay_position_pairs=replay_points,
                      array_equal_replay=True, paid_IDs_unchanged=True, source_files=hashes,
                      validation_reads=0, confirmation_reads=0, test_reads=0)
        write(completion, record)
        records.append(record)
        del base, blocks, deploy, heads
    write(here / ("benchmark.json" if benchmark else "completion.json"), records)
    return dict(dataset=dataset, CPU_seconds=time.process_time() - start, folds=records)


def summarize():
    stats = defaultdict(list)
    for dataset in STATE["cases"]:
        for fold in range(4):
            with np.load(STATE["out"] / dataset / f"fold_{fold}_metrics.npz", allow_pickle=False) as z:
                for k in z:
                    stats[k].append(z[k])
    examples = np.concatenate(stats.pop("examples"))
    if len(examples) != 17040 or len(set(map(tuple, examples))) != 17040:
        raise ValueError("original full scoring support changed")
    reference = json.loads((STATE["run"] / "component_source_transfer/integer_sum_source_v1/summary.json").read_text())["arms"]["high_transient_source"]["metrics"]["nmse"]
    results = {}
    for arm in ARMS:
        values = {k.split("__", 1)[1]: np.concatenate(v) for k, v in stats.items() if k.startswith(arm + "__")}
        summary = base_training_aggregate(values, examples, STATE["rows"])
        m = summary["metrics"]
        nmse, old, gain = (np.ravel(v) for v in (m["nmse"]["macro"], reference["macro"], m["rms_gain"]["macro"]))
        checks = dict(total=nmse[0] <= 1.05 * old[0], components=bool(np.all(nmse[1:] <= 1.1 * old[1:])),
                      datasets=all(np.all(np.array(v) <= 1.1 * np.array(reference["by_dataset"][d]))
                                   for d, v in m["nmse"]["by_dataset"].items()),
                      gain=bool(np.all((gain >= .75) & (gain <= 1.25))),
                      bands=all(np.max(abs(np.asarray(m[k]["macro"]))) <= 3.0103
                                for k in ("band_4_8_db", "band_8_13_db", "band_4_8_abs_db", "band_8_13_abs_db")),
                      correlation=bool(min(np.ravel(m["correlation"]["macro"])) > 0),
                      envelope=bool(min(np.ravel(m["local_rms_correlation"]["macro"])) >= .5))
        results[arm] = dict(summary=summary, checks=checks, passed=all(checks.values()),
                            codec_eligible=arm.startswith("hard_"))
    write(STATE["out"] / "summary.json", dict(arms=results,
          qualified=[a for a, v in results.items() if v["codec_eligible"] and v["passed"]],
          paid_IDs="unchanged own-fold exchange_balanced actual S/O", extra_bitrate=0,
          pattern_evidence=str(STATE["parent"] / "single_word_information.json"),
          original_arrays=400, positions=17040, subjects=60, development_reads=0,
          confirmation_reads=0, test_reads=0))


def independent_replay(dataset):
    """A fresh worker opens stored integers and model state before scoring arrays."""
    here = STATE["out"] / dataset
    records = []
    for fold in range(4):
        budget()
        with np.load(here / f"fold_{fold}_paid_ids.npz", allow_pickle=False) as z:
            payload = {b: z[b].copy() for b in "SO"}
            examples = z["examples"].copy()
        cases = [c for c in STATE["cases"][dataset] if c["fold"] == fold]
        expected = np.concatenate([np.array([(c["crop"]["row"], c["crop"]["start_s"], p)
                                             for p in c["crop"]["positions"]]) for c in cases])
        np.testing.assert_array_equal(examples, expected)
        source_ids = STATE["parent"] / dataset / f"fold_{fold}_ids.npz"
        with np.load(source_ids, allow_pickle=False) as z:
            take = z["fold"] == fold
            np.testing.assert_array_equal(z["examples"][take], examples)
            for b in "SO":
                np.testing.assert_array_equal(z["exchange_balanced_" + b][take], payload[b])
        decoded = {}
        for arm in ("hard_montage", "hard_same_site"):
            model = load_model(here / f"fold_{fold}" / (arm + ".npz"))
            values = {b: [] for b in ("S", "O", "sum")}
            offset = 0
            for c in cases:
                take = np.arange(offset, offset + c["count"])
                offset += c["count"]
                native = np.asarray(c["native"])
                sent = {b: payload[b][take][native.argsort()][None] for b in "SO"}
                wave = model.decode_ids(sent, dataset)
                np.testing.assert_array_equal(wave["sum"], wave["S"] + wave["O"])
                for b in values:
                    values[b].append(wave[b][0][native])
            decoded[arm] = {b: np.concatenate(v) for b, v in values.items()}
        # Decode above has no raw observations, targets, scales or encoder calls.
        cache = OrderedDict()
        observed_stats = defaultdict(list)
        offset = 0
        for c in cases:
            take = np.arange(offset, offset + c["count"])
            offset += c["count"]
            ex = examples[take]
            row = STATE["rows"][c["crop"]["row"]]
            if row["split"] != "train" or row["subject"] != c["subject"]:
                raise ValueError("independent score train boundary")
            batch = base_tokenizer_batch(STATE["rows"], ex, STATE["scales"], array_cache=cache, cache_limit=4)
            scale = input_scales(STATE["rows"], ex, STATE["scales"])
            for arm, waveform in decoded.items():
                result = joint_reconstruction_metrics({b: v[take] for b, v in waveform.items()}, batch, 1, scale)
                for k, v in result["metrics"].items():
                    observed_stats[arm + "__" + k].append(v)
        with np.load(here / f"fold_{fold}_metrics.npz", allow_pickle=False) as z:
            for k, values in observed_stats.items():
                np.testing.assert_array_equal(np.concatenate(values), z[k])
        records.append(dict(dataset=dataset, fold=fold, cases=len(cases), positions=len(examples),
                            model_crop_pairs=2 * len(cases), position_model_pairs=2 * len(examples),
                            metric_arrays_exact=len(observed_stats), all_metrics_array_equal=True,
                            original_paid_ID_arrays_equal=True, original_ids_sha256=sha(source_ids),
                            decoder_raw_X_reads=0, decoder_target_reads=0, encoder_calls=0,
                            payload_sha256=sha(here / f"fold_{fold}_paid_ids.npz")))
        del decoded
    write(here / "independent_integer_replay.json", records)
    return records


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = STATE["out"] / "visual_review_v3"
    destination.mkdir(exist_ok=True)
    selections = json.loads((STATE["parent"] / "visual_review_v1/selected_examples.json").read_text())
    gallery = {}
    for dataset in STATE["cases"]:
        for fold in range(4):
            with np.load(STATE["out"] / dataset / f"fold_{fold}_gallery.npz", allow_pickle=False) as z:
                gallery.update({k: z[k].copy() for k in z})
    colors = {"hard": "#999999", "hard_montage": "#c83828", "hard_same_site": "#3276b5",
              "continuous_montage": "#38914b"}
    manifests = []
    rendered = set()
    for example in selections["examples"]:
        identity = (example["row"], example["start_s"], example["position"])
        label = STATE["display"][identity]
        if label in rendered:
            continue
        rendered.add(label)
        t = np.arange(6000) / 200
        scale = float(gallery[f"{label}_scale"])
        unit = "REFED export" if example["source_unit"].startswith("unknown") else example["source_unit"]
        mask = gallery[f"{label}_mask"].reshape(-1)
        target_s = gallery[f"{label}_target_s"].reshape(-1)
        local = 7
        if "worst" in ",".join(example.get("audit_selection", [])) or "max" in ",".join(example.get("selection", [])):
            local = int(np.argmax([np.mean(target_s[i * 400:(i + 1) * 400] ** 2) for i in range(15)]))
        fig, axes = plt.subplots(3, 3, figsize=(15, 8.3))
        for row, (output, key) in enumerate((("sum", "x"), ("S", "s"), ("O", "o"))):
            truth = gallery[f"{label}_target_{key}"].reshape(-1) * scale
            a = local * 2
            boundary = float(np.clip(a, 2, 28))
            ranges = [(0, 30), (a, a + 2), (boundary - .2, boundary + .2)]
            for column, (lo, hi) in enumerate(ranges):
                ax = axes[row, column]
                use = (t >= lo) & (t < hi)
                ax.plot(t[use], np.where(mask[use], truth[use], np.nan), color="black", lw=.8, label="Target")
                for arm, color in colors.items():
                    wave = gallery[f"{label}_{arm}_{output}"].reshape(-1) * scale
                    ax.plot(t[use], np.where(mask[use], wave[use], np.nan), color=color,
                            lw=.7, alpha=.9, ls="--" if arm.startswith("continuous") else "-", label=arm)
                for boundary in np.arange(0, 31, 2):
                    if lo <= boundary <= hi:
                        ax.axvline(boundary, color="#bdbdbd", lw=.5, ls=":")
                if not mask[use].all():
                    ax.fill_between(t[use], 0, 1, where=~mask[use], transform=ax.get_xaxis_transform(),
                                    color="#f5ca6b", alpha=.3, label="masked")
                ax.set_xlim(lo, hi)
                ax.set_ylabel(f"{output} ({unit})")
                ax.set_xlabel("Time (s); dotted: token boundaries")
                ax.grid(alpha=.15)
        axes[0, 0].legend(fontsize=7, ncol=2)
        title = (f"{example['dataset']} | {example['subject']} | {example['record']} | {example['position_name']} | "
                 f"row={example['row']} start={example['start_s']}s\n"
                 "Same emitted S/O integers; fixed five-band train-only heads. Target/prediction share axes; no display smoothing.")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .93))
        path = destination / f"{label:02d}_{example['dataset']}_{example['position_name']}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        manifests.append(dict(path=str(path.relative_to(ROOT)), example=example, selection_inherited=True,
                              local_patch=local, output_order=["X", "S", "O"], kind="voltage",
                              source_unit=example["source_unit"], displayed_unit=unit, actually_opened=False))
    for dataset in STATE["cases"]:
        candidates = [e for e in selections["examples"] if e["dataset"] == dataset]
        example = next((e for e in candidates if "new_hard_joint_worst" in e.get("audit_selection", [])), candidates[0])
        label = STATE["display"][(example["row"], example["start_s"], example["position"])]
        scale = float(gallery[f"{label}_scale"])
        unit = "REFED export" if example["source_unit"].startswith("unknown") else example["source_unit"]
        mask = gallery[f"{label}_mask"].reshape(15, 400)
        fig, axes = plt.subplots(3, 3, figsize=(13, 8))
        frequency = np.fft.rfftfreq(400, 1 / 200)
        window = np.hanning(400)
        for row, (output, key) in enumerate((("sum", "x"), ("S", "s"), ("O", "o"))):
            traces = {"Target": gallery[f"{label}_target_{key}"]}
            traces.update({arm: gallery[f"{label}_{arm}_{output}"] for arm in colors})
            for name, raw in traces.items():
                raw = raw.reshape(15, 400) * scale
                color = "black" if name == "Target" else colors[name]
                spectrum = abs(np.fft.rfft((raw - raw.mean(-1, keepdims=True)) * window)) ** 2
                power = spectrum * (2 / (400 * np.sum(window ** 2)))
                for column, (low, high) in enumerate(((4, 8), (8, 13))):
                    use = (frequency >= low) & (frequency < high)
                    energy = power[:, use].sum(-1)
                    axes[row, column].plot(np.arange(15) * 2 + 1,
                            np.where(mask.all(-1), energy, np.nan), color=color, marker=".", lw=1, label=name)
                    axes[row, column].set_ylabel(f"{output} {low}-{high}Hz\nenergy ({unit}^2)", fontsize=9)
                rms = np.sqrt(np.mean(raw.reshape(-1, 50) ** 2, axis=-1))
                axes[row, 2].plot(np.arange(120) / 4 + .125,
                        np.where(mask.reshape(-1, 50).all(-1), rms, np.nan), color=color, lw=1, label=name)
                axes[row, 2].set_ylabel(f"{output} local RMS\n({unit})", fontsize=9)
        for ax in axes.flat:
            ax.set_xlabel("Time (s)")
            ax.set_xlim(0, 30)
            ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=6)
        fig.suptitle(f"{dataset} | {example['subject']} | {example['position_name']} | inherited hard event\n"
                     "Theta/alpha: complete 2s Hann energy; RMS: complete 0.25s blocks. Original masked support.", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .93))
        path = destination / f"dynamics_{dataset}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        manifests.append(dict(path=str(path.relative_to(ROOT)), example=example,
                              selection_inherited=True, kind="dynamics", source_unit=example["source_unit"],
                              displayed_unit=unit, actually_opened=False))
    write(destination / "visual_audit.json", dict(status="pending_actual_open", figures=manifests,
          inherited_selection_entries=len(selections["examples"]), unique_physical_examples=len(rendered),
          same_coordinate=True, target_mask_visible=True, baseline_same_ID=True,
          original_event_and_failure_selection=str(STATE["parent"] / "visual_review_v1/selected_examples.json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("benchmark", "train", "summary", "replay", "render", "review"), required=True)
    args = parser.parse_args()
    cfg = json.loads(json.dumps(yaml.safe_load(args.config.read_text())))
    if cfg["schema"] != "band_exchange_synthesis_v1" or any(cfg["data"][k] != "forbidden" for k in ("validation", "confirmation", "test")):
        raise ValueError("fixed train-only synthesis contract required")
    run = ROOT / cfg["joint_run"]
    parent = run / cfg["source"]
    out = run / cfg["output"]
    out.mkdir(exist_ok=True)
    lock = (out / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    population = json.loads((run / cfg["population"]).read_text())
    g0 = yaml.safe_load((ROOT / "experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml").read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(g0, ROOT)
    if (manifest["cfg_identity"], manifest["scales_identity"]) != (population["cfg_identity"], population["scales_identity"]):
        raise ValueError("canonical G0 data or scale identity changed")
    information = json.loads((parent / "single_word_information.json").read_text())["exchange_balanced"]
    if not all(information[b]["word"]["passed"] for b in "SO"):
        raise ValueError("frozen original actual-ID semantics not qualified")
    draws = json.loads((parent / "visual_review_v1/selected_examples.json").read_text())["examples"]
    display = {(e["row"], e["start_s"], e["position"]): i for i, e in enumerate(draws)}
    cases = {d: [c for c in population["cases"] if c["dataset"] == d] for d in sorted(population["native_positions"])}
    identity = dict(config=cfg, population_sha256=sha(run / cfg["population"]),
                    original_word_contract_sha256=sha(parent / "word_contract.json"))
    resolved = out / "resolved_config.json"
    if resolved.exists() and json.loads(resolved.read_text()) != identity:
        raise ValueError("frozen synthesis contract changed")
    write(resolved, identity)
    STATE.update(cfg=cfg, run=run, parent=parent, out=out / "benchmark" if args.stage == "benchmark" else out,
                 cases=cases, names=population["native_positions"], population=population,
                 rows=manifest["rows"], scales=scales, display=display, stage=args.stage)
    STATE["out"].mkdir(exist_ok=True)
    if args.stage == "summary":
        summarize()
        return
    if args.stage == "render":
        render()
        return
    if args.stage in ("replay", "review"):
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
            records = list(pool.map(independent_replay, cases))
        flat = [r for dataset in records for r in dataset]
        write(out / "independent_integer_replay.json", dict(records=flat,
              model_crop_pairs=sum(r["model_crop_pairs"] for r in flat),
              position_model_pairs=sum(r["position_model_pairs"] for r in flat),
              all_metrics_array_equal=True, decoding_inputs=["saved paid integer file", "frozen base", "frozen head", "fixed native order"]))
        if args.stage == "review":
            render()
        return
    before = time.time()
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
        accounting = list(pool.map(dataset_worker, cases))
    write(STATE["out"] / "execution.json", dict(stage=args.stage, workers=accounting,
          wall_seconds=time.time() - before, complete=True))
    if args.stage == "train":
        summarize()


if __name__ == "__main__":
    main()
