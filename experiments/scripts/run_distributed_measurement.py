"""Four-fold train-only signed, distributed, single-word EEG measurement test."""
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
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/scripts")]
from evaluate_joint_tokenizer import input_scales
from src.analysis.dual_component_tokenizer import base_training_aggregate, joint_reconstruction_metrics
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit, feature_means
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_joint_eeg, base_tokenizer_training_data
from src.tokenizers.band_exchange_words import source_properties
from src.tokenizers.distributed_measurement import (
    CAPACITY, HARD_ARMS, DistributedMeasurement, atomic_ids, infer, load_model,
    measure, probes, spatial_directions, word_features,
)
from src.tokenizers.eeg_transient_gain import load_model as load_source

STATE = {}
FIELDS = dict(power_4_8=(0, 1), power_8_13=(1, 2), rms_envelope=(2, 10))
ARMS = ("source",) + HARD_ARMS + tuple("continuous_" + a for a in HARD_ARMS)


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
    unit = os.environ.get("DISTRIBUTED_MEASUREMENT_UNIT")
    if unit:
        used = int(subprocess.check_output(["systemctl", "--user", "show", unit,
                                           "-p", "CPUUsageNSec", "--value"], text=True)) / 1e9
        used += float(os.environ.get("DISTRIBUTED_MEASUREMENT_PRIOR_CPU", "0"))
        if used > 2050:
            raise RuntimeError("35 CPU-minute distributed measurement budget reached; completed folds retained")


def data(dataset, fold, benchmark):
    cases = STATE["cases"][dataset]
    selected = None
    if benchmark:
        selected = {next(i for i, c in enumerate(cases) if c["fold"] != fold),
                    next(i for i, c in enumerate(cases) if c["fold"] == fold)}
    source_path = STATE["run"] / STATE["cfg"]["source"] / f"{dataset}_fold{fold}_transient.npz"
    source = load_source(source_path)
    fit_people = sorted({c["subject"] for c in cases if c["fold"] != fold})
    held_people = sorted({c["subject"] for c in cases if c["fold"] == fold})
    names = STATE["names"][dataset]
    if (set(source.metadata["fit_subjects"]) != set(fit_people)
            or set(fit_people) & set(held_people) or source.metadata["held_fold"] != fold
            or source.metadata["positions"][dataset] != names):
        raise ValueError("same-fold train-only source required")
    prepared = STATE["run"] / "component_source_transfer/band_exchange_words_v1/prepared" / dataset / f"fold_{fold}/metadata.json"
    provenance = json.loads(prepared.read_text())
    if provenance["source_sha256"] != sha(source_path):
        raise ValueError("frozen source identity changed")
    source_hash = sha(source_path)
    cache, blocks = OrderedDict(), []
    timing = Counter()
    for index, case in enumerate(cases):
        if selected is not None and index not in selected:
            continue
        budget()
        crop, native = case["crop"], np.asarray(case["native"])
        row = STATE["rows"][crop["row"]]
        if (row["split"], row["modality"], row["dataset"], row["subject"]) != ("train", "eeg", dataset, case["subject"]):
            raise ValueError("train boundary before measured array reads")
        np.testing.assert_array_equal(np.sort(native), np.arange(len(names)))
        start = time.process_time()
        fields, seen, _ = base_tokenizer_joint_eeg(STATE["rows"], dict(crop, positions=[]),
                            STATE["scales"], names, array_cache=cache, allowed_splits=("train",))
        x = fields["x"].T.reshape(1, len(names), 15, 400).astype(np.float32)
        proof = provenance["case_inputs"][index]
        if (hashlib.sha256(x.tobytes()).hexdigest() != proof["X_sha256"]
                or hashlib.sha256(seen.tobytes()).hexdigest() != proof["observed_sha256"]):
            raise ValueError("original native X or observation mask changed")
        dense = source.predict_continuous(x, dataset, seen[None])
        dense = {b: dense[b][0].astype(np.float32) for b in ("S", "O", "sum")}
        reference = None
        if case["fold"] == fold:
            original_x = fields["x"].T.reshape(1, len(names), 15, 400)
            reference = {b: value[0] for b, value in source.predict_continuous(original_x, dataset, seen[None]).items()}
            for branch in ("S", "O", "sum"):
                tolerance = 2e-6 + 1e-5 * abs(reference[branch])
                if np.any(abs(dense[branch] - reference[branch]) > tolerance):
                    raise ValueError("declared float32 source path exceeds existing pointwise replay tolerance")
        timing["observation_source"] += time.process_time() - start
        start = time.process_time()
        examples = np.array([(crop["row"], crop["start_s"], p) for p in crop["positions"]])
        original = base_tokenizer_batch(STATE["rows"], examples, STATE["scales"], array_cache=cache, cache_limit=4)
        batch = {}
        for key, value in original.items():
            value = np.asarray(value)
            batch[key] = np.broadcast_to(value[:, :1, :1].copy(), value.shape) if key.startswith(("v", "dv")) else value.copy()
        del original
        complete = batch["mask"][native.argsort()].all(-1)
        if not np.all(complete == complete[:1]):
            raise ValueError("native target support differs across sites; no silent complete-case restriction permitted")
        timing["target_load"] += time.process_time() - start
        blocks.append(dict(case=case, ex=examples, native=native, seen=seen, x=x,
                           source=dense, source_reference=reference, batch=batch, complete=complete[0],
                           scales=input_scales(STATE["rows"], examples, STATE["scales"])))
    return source, blocks, timing, dict(source_path=str(source_path), source_sha256=source_hash,
             fit_people=fit_people, held_people=held_people)


def case_weights(blocks):
    keys = [(b["case"]["subject"], STATE["rows"][b["case"]["crop"]["row"]]["record"]) for b in blocks]
    crops = Counter(keys)
    records = Counter(k[0] for k in crops)
    return np.array([1 / (len(records) * records[k[0]] * crops[k]) for k in keys])


def make_states(dataset, blocks, fold):
    fit = [b for b in blocks if b["case"]["fold"] != fold and b["complete"].any()]
    weight = case_weights(fit)
    count = len(STATE["names"][dataset])
    table = {k.upper(): v for k, v in STATE["scales"]["positions"]["eeg"][dataset].items()}
    temporal = probes(count, 107 + list(STATE["cases"]).index(dataset))
    states = {}
    for branch in "SO":
        sd = np.sqrt([float(np.ravel(table[n.upper()]["target_variance"][branch.lower()])[0]) for n in STATE["names"][dataset]])
        covariance = np.zeros((count, count))
        for w, block in zip(weight, fit, strict=True):
            values = block["source"][branch][:, block["complete"]].reshape(count, -1) / sd[:, None]
            values = np.where(block["seen"][:, None], values, 0)
            covariance += w * (values @ values.T / values.shape[1])
        state = dict(SD=sd, temporal=temporal, spatial=spatial_directions(covariance))
        local, spatial, properties, weights = [], [], [], []
        for w, block in zip(weight, fit, strict=True):
            valid = block["complete"]
            local.append(measure(block["source"][branch], block["seen"], state, "local")[:, valid])
            spatial.append(measure(block["source"][branch], block["seen"], state, "spatial")[:, valid])
            properties.append(source_properties(block["source"][branch])[:, valid])
            weights.append(np.full(valid.sum(), w / valid.sum()))
        prop, sample_weight = np.concatenate(properties, axis=1), np.concatenate(weights)
        state["local_gain"] = np.einsum("p,cpj->cj", sample_weight, abs(np.concatenate(local, axis=1)))
        state["spatial_gain"] = np.einsum("p,cpj->cj", sample_weight, abs(np.concatenate(spatial, axis=1)))
        medians, levels = np.zeros((count, 3)), np.zeros((count, 3, 2))
        for site in range(count):
            for attribute in range(3):
                value = prop[site, :, attribute]
                order = np.argsort(value, kind="stable")
                medians[site, attribute] = value[order[np.searchsorted(np.cumsum(sample_weight[order]), .5)]]
                for cell in range(2):
                    take = (value > medians[site, attribute]) == bool(cell)
                    levels[site, attribute, cell] = np.sum(sample_weight[take] * value[take]) / max(sample_weight[take].sum(), 1e-20)
        state.update(medians=medians, property_levels=levels)
        states[branch] = state
    return states


def features(block, states):
    received, continuous = {}, {}
    for arm in HARD_ARMS:
        received[arm], continuous[arm] = {}, {}
        layout = "local" if arm.startswith("local") else "spatial"
        for branch in "SO":
            state = states[branch]
            measured = measure(block["source"][branch], block["seen"], state, layout)
            prop = source_properties(block["source"][branch])
            paid = atomic_ids(measured, prop, block["seen"], state["medians"], arm)
            received[arm][branch] = paid
            value = np.concatenate((measured[..., :7], prop), axis=-1) if arm.endswith("attributes") else measured
            continuous[arm][branch] = np.where(block["seen"][:, None, None], value, 0).astype(np.float32)
    hard = {arm: {b: word_features(received[arm][b], states[b][("local" if arm.startswith("local") else "spatial") + "_gain"],
                                           states[b]["property_levels"], arm).astype(np.float32) for b in "SO"} for arm in HARD_ARMS}
    return received, dict(**hard, **{"continuous_" + a: v for a, v in continuous.items()})


def fit_head(x, y, weights):
    start = time.process_time()
    x, y, weights = np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(weights, float)
    weights /= weights.sum()
    xmean = np.einsum("n,nf->f", weights, x).astype(np.float32)
    ymean = np.einsum("n,nf->f", weights, y).astype(np.float32)
    sd = np.sqrt(np.einsum("n,nf->f", weights, (x - xmean) ** 2)).clip(1e-6).astype(np.float32)
    a = ((x - xmean) / sd) * np.sqrt(weights[:, None]).astype(np.float32)
    b = (y - ymean) * np.sqrt(weights[:, None]).astype(np.float32)
    gram = (a.T @ a).astype(np.float64)
    cross = np.asfortranarray(a.T @ b, dtype=np.float64)
    moment_seconds = time.process_time() - start
    start = time.process_time()
    coefficient = cho_solve(cho_factor(gram + STATE["cfg"]["model"]["ridge_fraction"] * np.eye(len(gram)),
                                      check_finite=False), cross, check_finite=False, overwrite_b=True)
    coefficient /= sd[:, None]
    result = dict(xmean=xmean, ymean=ymean, coefficient=coefficient.astype(np.float32))
    return result, moment_seconds, time.process_time() - start


def score(block, prediction):
    result = joint_reconstruction_metrics({b: prediction[b][block["native"]] for b in ("sum", "S", "O")},
                                          block["batch"], 1, block["scales"])
    if result["support"]["nonfinite_prediction_points"].any():
        raise ValueError("nonfinite distributed-measurement voltage output")
    return result["metrics"]


def worker(dataset):
    with threadpool_limits(limits=1):
        try:
            return run_dataset(dataset)
        except Exception:
            import traceback
            write(STATE["out"] / dataset / "failure.json", dict(traceback=traceback.format_exc()))
            raise


def numerical_benchmark(dataset):
    """Synthetic dimensions match the largest FIT fold; no measured fit score."""
    with threadpool_limits(limits=1):
        cases = STATE["cases"][dataset]
        with np.load(STATE["run"] / "component_source_transfer/hard_as_v1/local_information_v1/training_attributes.npz", allow_pickle=False) as z:
            valid = z["valid"]
            fit_counts = [sum(int(valid[c["first"]].sum()) for c in cases if c["fold"] != fold) for fold in range(4)]
        n, sites = max(fit_counts), len(STATE["names"][dataset])
        rng = np.random.default_rng(107)
        start = time.process_time()
        x = rng.normal(size=(n, sites * 10)).astype(np.float32)
        y = rng.normal(size=(n, sites * 400)).astype(np.float32)
        generation = time.process_time() - start
        _, moments, solve = fit_head(x, y, np.full(n, 1 / n))
        result = dict(dataset=dataset, fit_counts=fit_counts, n=n, sites=sites,
                      synthetic_generation_CPU_seconds=generation, moment_CPU_seconds=moments,
                      solve_CPU_seconds=solve, heads_per_fold=12, folds=4,
                      full_fit_CPU_estimate=4 * 12 * (moments + solve),
                      purpose="dimension-matched random fixture throughput only; no model or numerical performance claim",
                      peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        write(STATE["out"] / (dataset + "_matrix_benchmark.json"), result)
        return result


def run_dataset(dataset):
    here = STATE["out"] / dataset
    here.mkdir(exist_ok=True)
    benchmark = STATE["stage"] == "benchmark"
    start = time.process_time()
    records = []
    for fold in ([0] if benchmark else range(4)):
        completion = here / f"fold_{fold}_completion.json"
        if completion.exists():
            records.append(json.loads(completion.read_text()))
            continue
        budget()
        before = time.process_time()
        source, blocks, timings, identity = data(dataset, fold, benchmark)
        load_seconds = time.process_time() - before
        before = time.process_time()
        paths = {a: here / f"fold_{fold}_{a}.npz" for a in HARD_ARMS}
        diagnostic_path = here / f"fold_{fold}_continuous_heads.npz"
        resume_heads = all(p.exists() for p in paths.values()) and diagnostic_path.exists()
        cached_models = {a: load_model(p) for a, p in paths.items()} if resume_heads else {}
        states = cached_models["spatial_attributes"].states if resume_heads else make_states(dataset, blocks, fold)
        for block in blocks:
            block["ids"], block["features"] = features(block, states)
        feature_seconds = time.process_time() - before
        fitting = [b for b in blocks if b["case"]["fold"] != fold and b["complete"].any()]
        weights = case_weights(fitting)
        weights = np.concatenate([np.full(b["complete"].sum(), w / b["complete"].sum()) for w, b in zip(weights, fitting, strict=True)])
        target = {} if resume_heads else {b: np.concatenate([v["batch"][b.lower()][v["native"].argsort()][:, v["complete"]].transpose(1, 0, 2).reshape(v["complete"].sum(), -1) for v in fitting]) for b in "SO"}
        heads, moment_seconds, solve_seconds = {}, 0., 0.
        if resume_heads:
            heads.update({a: model.heads for a, model in cached_models.items()})
            with np.load(diagnostic_path, allow_pickle=False) as z:
                for arm in ARMS:
                    if arm.startswith("continuous_"):
                        heads[arm] = {b: {name: z[f"{arm}_{b}_{name}"].copy()
                                             for name in ("xmean", "ymean", "coefficient")} for b in "SO"}
        else:
            for arm in ARMS[1:]:
                heads[arm] = {}
                for branch in "SO":
                    budget()
                    x = np.concatenate([b["features"][arm][branch][:, b["complete"]].transpose(1, 0, 2).reshape(b["complete"].sum(), -1) for b in fitting])
                    heads[arm][branch], moments, solving = fit_head(x, target[branch], weights)
                    moment_seconds += moments
                    solve_seconds += solving
        del target
        before = time.process_time()
        models = {}
        for arm in HARD_ARMS:
            metadata = dict(schema="distributed_measurement_v1", arm=arm, dataset=dataset,
                            names=STATE["names"][dataset], fold=fold, identity=identity,
                            K=1024, paid_bits=10, fit_people=identity["fit_people"])
            model = DistributedMeasurement(metadata, source, states, heads[arm])
            path = here / f"fold_{fold}_{arm}.npz"
            if not path.exists():
                model.save(path)
            models[arm] = cached_models[arm] if resume_heads else load_model(path)
        diagnostics = {f"{arm}_{branch}_{name}": value
                       for arm in ARMS if arm.startswith("continuous_")
                       for branch in "SO" for name, value in heads[arm][branch].items()}
        if not diagnostic_path.exists():
            np.savez(diagnostic_path, **diagnostics)
        fit_probe = fitting[0]
        for arm, model in models.items():
            fresh_ids = model.encode(fit_probe["x"], dataset, fit_probe["seen"][None])
            for branch in "SO":
                np.testing.assert_array_equal(fresh_ids[branch][0], fit_probe["ids"][arm][branch])
        serialization_seconds = time.process_time() - before
        before = time.process_time()
        stat, ids, all_examples, all_folds = defaultdict(list), defaultdict(list), [], []
        reference_metrics = defaultdict(list)
        replay_cases = fresh = 0
        for block in blocks:
            all_examples.append(block["ex"])
            all_folds.append(np.full(len(block["ex"]), block["case"]["fold"]))
            for arm in HARD_ARMS:
                for b in "SO":
                    ids[arm + "_" + b].append(block["ids"][arm][b][block["native"]])
            if block["case"]["fold"] != fold:
                continue
            budget()
            stat["examples"].append(block["ex"])
            original_reference = score(block, block["source_reference"])
            for key in ("nmse", "rms_gain", "band_4_8_db", "band_8_13_db"):
                reference_metrics[key].append(original_reference[key])
            for arm in ARMS:
                prediction = block["source"] if arm == "source" else {b: infer(block["features"][arm][b], heads[arm][b]) for b in "SO"}
                if arm != "source":
                    prediction["sum"] = prediction["S"] + prediction["O"]
                result = score(block, prediction)
                for key, value in result.items():
                    stat[arm + "__" + key].append(value)
                if arm in HARD_ARMS:
                    repeated = models[arm].decode_ids({b: block["ids"][arm][b][None] for b in "SO"}, dataset)
                    for b in ("S", "O", "sum"):
                        np.testing.assert_array_equal(repeated[b][0], prediction[b])
                    replay_cases += 1
            if not fresh:
                for arm, model in models.items():
                    actual = model.encode(block["x"], dataset, block["seen"][None])
                    for b in "SO":
                        np.testing.assert_array_equal(actual[b][0], block["ids"][arm][b])
                fresh += 1
        score_seconds = time.process_time() - before
        joined = {k: np.concatenate(v) for k, v in stat.items()}
        baseline = {}
        if not benchmark:
            path = STATE["run"] / STATE["cfg"]["source"] / (dataset + "_metrics.npz")
            with np.load(path, allow_pickle=False) as z:
                lookup = {tuple(e): i for i, e in enumerate(z["examples"])}
                selected = np.array([lookup[tuple(e)] for e in joined["examples"]])
                np.testing.assert_array_equal(z["examples"][selected], joined["examples"])
                np.testing.assert_array_equal(z["fold"][selected], np.full(len(selected), fold))
                for key in ("nmse", "rms_gain", "band_4_8_db", "band_8_13_db"):
                    expected = z["transient__" + key][selected]
                    actual = joined["source__" + key]
                    np.testing.assert_array_equal(np.concatenate(reference_metrics[key]), expected)
                    baseline[key] = float(np.nanmax(abs(actual - expected)))
        np.savez_compressed(here / f"fold_{fold}_metrics.npz", **joined)
        np.savez_compressed(here / f"fold_{fold}_ids.npz", examples=np.concatenate(all_examples), fold=np.concatenate(all_folds),
                            **{k: np.concatenate(v) for k, v in ids.items()})
        if sha(identity["source_path"]) != identity["source_sha256"]:
            raise ValueError("original source model changed")
        record = dict(dataset=dataset, fold=fold, cases=len(blocks), fit_patches=len(weights),
                      load_CPU_seconds=load_seconds, feature_CPU_seconds=feature_seconds,
                      moment_CPU_seconds=moment_seconds, solve_CPU_seconds=solve_seconds,
                      serialization_CPU_seconds=serialization_seconds, score_CPU_seconds=score_seconds,
                      details_CPU_seconds=timings, worker_CPU_seconds=time.process_time() - start,
                      peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      source=identity, integer_model_crop_pairs=replay_cases, integer_replay_exact=True,
                      resumed_frozen_heads_without_refitting=resume_heads,
                      fresh_encoder_roundtrip_FIT_and_held=True, source_float32_reproduction_maxgap=baseline,
                      historical_source_float64_original_input_metrics_array_equal=True,
                      validation_reads=0, confirmation_reads=0, test_reads=0)
        write(completion, record)
        records.append(record)
        del models, heads, states, blocks, source
    write(here / "completion.json", records)
    return dict(dataset=dataset, CPU_seconds=time.process_time() - start, records=records)


def word_information():
    with np.load(STATE["run"] / "component_source_transfer/hard_as_v1/local_information_v1/training_attributes.npz", allow_pickle=False) as z:
        bank = {k: z[k].copy() for k in z}
    result = {}
    backgrounds = len(STATE["population"]["background_sites"]) * 4
    for arm in HARD_ARMS:
        result[arm] = {}
        for branch in "SO":
            errors = np.full((len(bank["examples"]), 2, 3), np.nan)
            qualified = total = 0
            usage = {}
            for fold in range(4):
                parts, ex = [], []
                for dataset in STATE["cases"]:
                    with np.load(STATE["out"] / dataset / f"fold_{fold}_ids.npz", allow_pickle=False) as z:
                        parts.append(z[arm + "_" + branch])
                        ex.append(z["examples"])
                paid = np.concatenate(parts)
                np.testing.assert_array_equal(np.concatenate(ex), bank["examples"])
                fit, held = bank["fold"] != fold, bank["fold"] == fold
                if set(bank["subject"][fit]) & set(bank["subject"][held]):
                    raise ValueError("single-word crossfit subject overlap")
                audit = SparsePatternAudit(CAPACITY, backgrounds, [], 10, FIELDS)
                valid = bank["valid"] & fit[:, None]
                audit.fit(paid[valid], bank["background"][valid], bank[branch + "_attributes"][valid])
                people = np.zeros(CAPACITY, int)
                for person in np.unique(bank["subject"][fit]):
                    mask = bank["valid"] & (bank["subject"][:, None] == person)
                    people[np.unique(paid[mask])] += 1
                hv = bank["valid"][held]
                ids, bg = paid[held][hv], bank["background"][held][hv]
                background, _, augmented = audit.read(ids, bg)
                target = bank[branch + "_attributes"][held][hv]
                partial = np.zeros((*hv.shape, 2, 3))
                partial[hv] = np.stack([feature_means((v - target) ** 2, FIELDS) for v in (background, augmented)], axis=1)
                averaged = partial.sum(1) / hv.sum(1).clip(1)[:, None, None]
                averaged[~hv.any(1)] = np.nan
                errors[held] = averaged
                good = (audit.local.count[ids] >= 8) & (people[ids] >= 2)
                qualified += int(good.sum())
                total += len(ids)
                usage[f"fit_{fold}"] = audit.local.count
                usage[f"held_{fold}"] = np.bincount(ids, minlength=CAPACITY)
                usage[f"fit_people_{fold}"] = people
            hierarchy = base_training_aggregate({"nmse": errors}, bank["examples"], STATE["rows"])["metrics"]["nmse"]
            macro = np.asarray(hierarchy["macro"])
            gain = 1 - macro[1] / macro[0]
            fraction = np.mean([np.asarray(v["value"])[1] < np.asarray(v["value"])[0] for v in hierarchy["by_subject"]], axis=0)
            result[arm][branch] = dict(gain=gain, positive_subject_fraction=fraction, support=qualified / total,
                  hierarchy=hierarchy, passed=bool(min(gain[:2]) > 0 and max(gain) >= .05 and min(fraction[:2]) >= .5 and qualified / total >= .9))
            np.savez_compressed(STATE["out"] / f"{arm}_{branch}_word_information.npz", errors=errors, **usage)
    write(STATE["out"] / "single_word_information.json", result)
    return result


def summarize():
    vectors = defaultdict(list)
    for dataset in STATE["cases"]:
        for fold in range(4):
            with np.load(STATE["out"] / dataset / f"fold_{fold}_metrics.npz", allow_pickle=False) as z:
                for key in z:
                    vectors[key].append(z[key])
    examples = np.concatenate(vectors.pop("examples"))
    if len(examples) != 17040 or len(set(map(tuple, examples))) != 17040:
        raise ValueError("original 17040 positions required")
    reference = json.loads((STATE["run"] / "component_source_transfer/integer_sum_source_v1/summary.json").read_text())["arms"]["high_transient_source"]["metrics"]["nmse"]
    information = word_information()
    results = {}
    for arm in ARMS:
        metrics = {k.split("__", 1)[1]: np.concatenate(v) for k, v in vectors.items() if k.startswith(arm + "__")}
        summary = base_training_aggregate(metrics, examples, STATE["rows"])
        m = summary["metrics"]
        nmse, old, gain = (np.ravel(v) for v in (m["nmse"]["macro"], reference["macro"], m["rms_gain"]["macro"]))
        checks = dict(total=nmse[0] <= 1.05 * old[0], components=bool(np.all(nmse[1:] <= 1.1 * old[1:])),
              datasets=all(np.all(np.asarray(v) <= 1.1 * np.asarray(reference["by_dataset"][d])) for d, v in m["nmse"]["by_dataset"].items()),
              gain=bool(np.all((gain >= .75) & (gain <= 1.25))),
              bands=all(np.max(abs(np.asarray(m[k]["macro"]))) <= 3.0103 for k in ("band_4_8_db", "band_8_13_db", "band_4_8_abs_db", "band_8_13_abs_db")),
              correlation=bool(min(np.ravel(m["correlation"]["macro"])) > 0),
              envelope=bool(min(np.ravel(m["local_rms_correlation"]["macro"])) >= .5))
        results[arm] = dict(summary=summary, checks=checks, passed=all(checks.values()),
              single_word_both_pass=all(v["passed"] for v in information.get(arm, {}).values()) if arm in HARD_ARMS else None)
    write(STATE["out"] / "summary.json", dict(arms=results,
          qualified=[a for a in HARD_ARMS if results[a]["passed"] and results[a]["single_word_both_pass"]],
          original_arrays=400, positions=17040, people=60, folds=4, actual_K=1024,
          IDs_per_site_branch_30s=15, bits_per_ID=10, EEG_bits_per_second_per_site=10,
          validation_reads=0, confirmation_reads=0, test_reads=0))


def select_panel():
    retained = json.loads((STATE["run"] / "component_source_transfer/band_exchange_words_v1/visual_review_v1/selected_examples.json").read_text())["examples"]
    chosen = {}
    for index in (0, 4, 9, 13, 17, 18, 19, 20):
        e = dict(retained[index])
        e["selection"] = ["inherited_fixed_or_strong_component_event"]
        chosen[(e["row"], e["start_s"], e["position"])] = e
    for dataset in STATE["cases"]:
        ex, error = [], []
        for fold in range(4):
            with np.load(STATE["out"] / dataset / f"fold_{fold}_metrics.npz", allow_pickle=False) as z:
                ex.extend(z["examples"])
                error.extend(np.mean(z["spatial_attributes__nmse"], axis=(1, 2)))
        ex, error = np.array(ex), np.array(error)
        rng = np.random.default_rng(107)
        random_index = int(rng.integers(len(ex)))
        ordered = sorted(range(len(ex)), key=lambda i: (float(error[i]), tuple(ex[i])))
        for index, rule in ((random_index, "fixed_seed107_random"),
                            (ordered[int(.95 * (len(ordered) - 1))], "all_position_joint_error_P95"),
                            (ordered[-1], "all_position_joint_error_maximum")):
            row_index, start, position = map(int, ex[index])
            row = STATE["rows"][row_index]
            key = (row_index, start, position)
            if key in chosen:
                chosen[key]["selection"].append(rule)
            else:
                chosen[key] = dict(row=row_index, start_s=start, position=position,
                     dataset=dataset, subject=row["subject"], record=row["record"],
                     position_name=row["positions"][position], selection=[rule],
                     source_unit="unknown_REFED_EEG_export" if dataset == "refed" else "uV")
    panel = list(chosen.values())
    for label, example in enumerate(panel):
        example["label"] = label
    write(STATE["out"] / "selected_examples.json", dict(examples=panel,
          rule="old fixed/strong-component event, seed107 random, entire-panel candidate P95 and maximum; no qualification selection"))
    return panel


def replay_dataset(dataset):
    with threadpool_limits(limits=1):
        here = STATE["out"] / dataset
        requested = {(e["row"], e["start_s"], e["position"]): e for e in STATE["panel"] if e["dataset"] == dataset}
        records, gallery = [], {}
        cache = OrderedDict()
        for fold in range(4):
            budget()
            with np.load(here / f"fold_{fold}_ids.npz", allow_pickle=False) as z:
                paid = {a: {b: z[a + "_" + b].copy() for b in "SO"} for a in HARD_ARMS}
                all_ex = z["examples"].copy()
            models = {a: load_model(here / f"fold_{fold}_{a}.npz") for a in HARD_ARMS}
            rows = defaultdict(list)
            predictions = {a: {b: [] for b in ("S", "O", "sum")} for a in HARD_ARMS}
            offset = 0
            held = []
            for case in STATE["cases"][dataset]:
                take = np.arange(offset, offset + case["count"])
                offset += case["count"]
                if case["fold"] != fold:
                    continue
                native = np.asarray(case["native"])
                for arm, model in models.items():
                    decoded = model.decode_ids({b: paid[arm][b][take][native.argsort()][None] for b in "SO"}, dataset)
                    np.testing.assert_array_equal(decoded["sum"], decoded["S"] + decoded["O"])
                    for b in decoded:
                        predictions[arm][b].append(decoded[b][0][native])
                held.append((case, all_ex[take]))
            # All decoding above is from saved integers/model/fixed native order.
            predictions = {a: {b: np.concatenate(v) for b, v in d.items()} for a, d in predictions.items()}
            selection = "continuous_spatial_attributes"
            with np.load(here / f"fold_{fold}_continuous_heads.npz", allow_pickle=False) as z:
                head = {b: {n: z[f"{selection}_{b}_{n}"].copy() for n in ("xmean", "ymean", "coefficient")} for b in "SO"}
            offset = 0
            for case, examples in held:
                budget()
                take = np.arange(offset, offset + len(examples))
                offset += len(examples)
                batch = base_tokenizer_batch(STATE["rows"], examples, STATE["scales"], array_cache=cache, cache_limit=4)
                scales = input_scales(STATE["rows"], examples, STATE["scales"])
                for arm, decoded in predictions.items():
                    result = joint_reconstruction_metrics({b: v[take] for b, v in decoded.items()}, batch, 1, scales)
                    for k, v in result["metrics"].items():
                        rows[arm + "__" + k].append(v)
                chosen = [(j, requested[tuple(e)]) for j, e in enumerate(examples) if tuple(e) in requested]
                if chosen:
                    fields, seen, _ = base_tokenizer_joint_eeg(STATE["rows"], dict(case["crop"], positions=[]),
                                            STATE["scales"], STATE["names"][dataset], allowed_splits=("train",))
                    x = fields["x"].T.reshape(1, len(seen), 15, 400).astype(np.float32)
                    model = models["spatial_attributes"]
                    dense = model.source.predict_continuous(x, dataset, seen[None])
                    continuous = {}
                    for b in "SO":
                        wave = dense[b][0].astype(np.float32)
                        measured = measure(wave, seen, model.states[b], "spatial")
                        feature = np.concatenate((measured[..., :7], source_properties(wave)), axis=-1)
                        feature = np.where(seen[:, None, None], feature, 0).astype(np.float32)
                        continuous[b] = infer(feature, head[b])
                    continuous["sum"] = continuous["S"] + continuous["O"]
                    for j, e in chosen:
                        label = e["label"]
                        for b in ("x", "s", "o", "mask"):
                            gallery[f"{label}_target_{b}"] = batch[b][j]
                        gallery[f"{label}_scale"] = scales[j]
                        for arm in HARD_ARMS:
                            for b in ("S", "O", "sum"):
                                gallery[f"{label}_{arm}_{b}"] = predictions[arm][b][take[j]]
                        for b in ("S", "O", "sum"):
                            gallery[f"{label}_continuous_{b}"] = continuous[b][case["native"][j]]
            with np.load(here / f"fold_{fold}_metrics.npz", allow_pickle=False) as z:
                for k, values in rows.items():
                    np.testing.assert_array_equal(np.concatenate(values), z[k])
            records.append(dict(dataset=dataset, fold=fold, cases=len(held), positions=offset,
                   model_crop_pairs=3 * len(held), model_position_pairs=3 * offset,
                   all_metric_arrays_equal=True, decoder_X_reads=0, decoder_target_reads=0,
                   decoder_continuous_latent_reads=0, encoder_calls_in_decoder=0,
                   paid_IDs_sha256=sha(here / f"fold_{fold}_ids.npz")))
        np.savez_compressed(here / "review_gallery.npz", **gallery)
        write(here / "independent_integer_replay.json", records)
        return records


def render_review():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = STATE["out"] / "visual_review_v1"
    out.mkdir(exist_ok=True)
    gallery = {}
    for dataset in STATE["cases"]:
        with np.load(STATE["out"] / dataset / "review_gallery.npz", allow_pickle=False) as z:
            gallery.update({k: z[k].copy() for k in z})
    colors = dict(spatial_attributes="#c33b2b", local_attributes="#3774b8", spatial_carrier="#b58b24", continuous="#36884a")
    figures = []
    for e in STATE["panel"]:
        label = e["label"]
        scale = float(gallery[f"{label}_scale"])
        unit = "REFED export" if e["dataset"] == "refed" else "uV"
        mask = gallery[f"{label}_target_mask"].reshape(-1)
        t = np.arange(6000) / 200
        target_s = gallery[f"{label}_target_s"].reshape(15, 400)
        patch = int(np.argmax(np.mean(target_s ** 2, axis=-1))) if any("maximum" in a or "event" in a for a in e["selection"]) else 7
        boundary = float(np.clip(patch * 2, 2, 28))
        fig, axes = plt.subplots(3, 3, figsize=(15, 8.3))
        for row, (name, key) in enumerate((("sum", "x"), ("S", "s"), ("O", "o"))):
            target = gallery[f"{label}_target_{key}"].reshape(-1) * scale
            for column, (low, high) in enumerate(((0, 30), (patch * 2, patch * 2 + 2), (boundary - .2, boundary + .2))):
                ax = axes[row, column]
                take = (t >= low) & (t < high)
                ax.plot(t[take], np.where(mask[take], target[take], np.nan), color="black", lw=.8, label="Target")
                for arm, color in colors.items():
                    prediction = gallery[f"{label}_{arm}_{name}"].reshape(-1) * scale
                    ax.plot(t[take], np.where(mask[take], prediction[take], np.nan), color=color, lw=.7,
                            ls="--" if arm == "continuous" else "-", label=arm)
                for edge in np.arange(0, 31, 2):
                    if low <= edge <= high:
                        ax.axvline(edge, color="#aaaaaa", lw=.5, ls=":")
                if not mask[take].all():
                    ax.fill_between(t[take], 0, 1, where=~mask[take], transform=ax.get_xaxis_transform(), color="orange", alpha=.2)
                ax.set_xlim(low, high)
                ax.set_ylabel(f"{name} ({unit})")
                ax.set_xlabel("Time (s); dotted: token boundaries")
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=6, ncol=2)
        fig.suptitle(f"{e['dataset']} | {e['subject']} | {e['record']} | {e['position_name']} | row={e['row']} start={e['start_s']}s\n"
                     "Same actual 10-bit per-site per-2s budget; true targets and all predictions share native coordinates.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, .93))
        path = out / f"{label:02d}_{e['dataset']}_{e['position_name']}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        figures.append(dict(path=str(path.relative_to(ROOT)), example=e, actual_opened=False, kind="voltage"))
    information = json.loads((STATE["out"] / "single_word_information.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for row, branch in enumerate("SO"):
        for column, name in enumerate(("theta", "alpha", "RMS")):
            values = [information[a][branch]["gain"][column] * 100 for a in HARD_ARMS]
            axes[row, column].bar(np.arange(3), values, color=[colors[a] for a in HARD_ARMS])
            axes[row, column].axhline(0, color="black", lw=.7)
            axes[row, column].set_xticks(np.arange(3), ["spatial7+3", "local7+3", "spatial10"], rotation=15)
            axes[row, column].set_ylabel(f"{branch}: {name} conditional MSE gain (%)")
    fig.suptitle("Actual entire single ID versus dataset/site/X-RMS background; four held-person folds")
    fig.tight_layout(rect=(0, 0, 1, .94))
    path = out / "actual_single_word_increment.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    figures.append(dict(path=str(path.relative_to(ROOT)), kind="single_word", actual_opened=False))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for row, branch in enumerate("SO"):
        for arm in HARD_ARMS:
            with np.load(STATE["out"] / f"{arm}_{branch}_word_information.npz", allow_pickle=False) as z:
                count = sum(z[f"held_{f}"] for f in range(4))
                ordered = np.sort(count[1:])[::-1]
                axes[row, 0].plot(np.arange(1, 1024), ordered, label=arm)
                axes[row, 1].plot(np.arange(1, 1024), np.cumsum(ordered) / max(ordered.sum(), 1), label=arm)
        axes[row, 0].set_yscale("symlog", linthresh=1)
        axes[row, 0].set_ylabel(f"{branch}: held complete patches / actual ID")
        axes[row, 1].set_ylabel(f"{branch}: cumulative actual-ID held coverage")
        for ax in axes[row]:
            ax.set_xlabel("All 1023 normal IDs ranked by held occupancy")
            ax.legend(fontsize=7)
    fig.tight_layout()
    path = out / "all_ID_occupancy.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    figures.append(dict(path=str(path.relative_to(ROOT)), kind="occupancy", actual_opened=False))
    write(out / "visual_audit.json", dict(status="pending_actual_open", figures=figures,
          no_display_smoothing=True, all_original_scoring_positions_retained=True,
          physical_units="REFED unknown native export; other datasets uV"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("benchmark", "train", "summary", "numerical_benchmark", "review", "render"), required=True)
    args = parser.parse_args()
    cfg = json.loads(json.dumps(yaml.safe_load(args.config.read_text())))
    if cfg["schema"] != "distributed_measurement_v1" or any(cfg["data"][k] != "forbidden" for k in ("validation", "confirmation", "test", "reserved_four_people")):
        raise ValueError("new train-only atomic measurement contract required")
    run, out = ROOT / cfg["joint_run"], ROOT / cfg["joint_run"] / cfg["output"]
    out.mkdir(exist_ok=True)
    lock = (out / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    population = json.loads((run / cfg["population"]).read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(yaml.safe_load((ROOT / "experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml").read_text()), ROOT)
    if (manifest["cfg_identity"], manifest["scales_identity"]) != (population["cfg_identity"], population["scales_identity"]):
        raise ValueError("original G0 identity changed")
    identity = dict(config=cfg, population_sha256=sha(run / cfg["population"]),
                    data_identity=manifest["cfg_identity"], scales_identity=manifest["scales_identity"])
    if (out / "resolved_config.json").exists() and json.loads((out / "resolved_config.json").read_text()) != identity:
        raise ValueError("frozen distributed measurement contract changed")
    write(out / "resolved_config.json", identity)
    cases = {d: [c for c in population["cases"] if c["dataset"] == d] for d in sorted(population["native_positions"])}
    STATE.update(run=run, out=out / "benchmark" if args.stage == "benchmark" else out, cfg=cfg,
                 cases=cases, names=population["native_positions"], population=population,
                 rows=manifest["rows"], scales=scales, stage=args.stage)
    STATE["out"].mkdir(exist_ok=True)
    if args.stage == "summary":
        summarize()
        return
    if args.stage in ("review", "render"):
        STATE["panel"] = select_panel()
        if args.stage == "review":
            with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
                result = list(pool.map(replay_dataset, cases))
            flat = [v for group in result for v in group]
            write(out / "independent_integer_replay.json", dict(records=flat,
                  model_crop_pairs=sum(v["model_crop_pairs"] for v in flat),
                  model_position_pairs=sum(v["model_position_pairs"] for v in flat), all_metrics_array_equal=True))
        render_review()
        return
    if args.stage == "numerical_benchmark":
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
            results = list(pool.map(numerical_benchmark, cases))
        write(out / "numerical_benchmark.json", results)
        return
    start = time.time()
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
        accounting = list(pool.map(worker, cases))
    write(STATE["out"] / "execution.json", dict(workers=accounting, wall_seconds=time.time() - start, complete=True))
    if args.stage == "train":
        summarize()


if __name__ == "__main__":
    main()
