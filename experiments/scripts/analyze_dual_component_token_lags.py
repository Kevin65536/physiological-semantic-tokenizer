#!/usr/bin/env python3
"""Frozen S-token replay and exploratory, subject-stratified lag diagnostics.

Only the named completed public voltage generation is consumed. No target
array, native dataset, fitting step, or O-token statistic is needed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml

DEFAULT = ROOT / "experiments/configs/physiology_semantic_tokenizer/dual_component_token_lags_v1.yaml"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def table(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_parent(parent):
    summary = read(parent / "voltage_summary.json")
    selection = read(parent / "voltage_selection.json")
    if summary.get("status") != "completed" or summary.get("experiment_id") != "TK-DUAL-VOLTAGE-v2":
        raise ValueError("completed public voltage v2 source required")
    if (selection.get("eeg_candidate"), selection.get("hb_candidate"), selection.get("selection_split")) != (
        "eeg_voltage", "hb_blocks", "validation"
    ) or selection.get("test_used") is not False:
        raise ValueError("frozen validation selection mismatch")
    manifests = {}
    subject_splits = {}
    for phase, folder, allowed in (("development", "voltage_cache", {"train", "validation"}),
                                   ("test", "voltage_test_cache", {"test"})):
        manifest = read(parent / folder / "manifest.json")
        if manifest.get("schema") != "dual_component_voltage_cache_v1" or manifest.get("protected_data") != "forbidden":
            raise ValueError("public voltage manifest required before array access")
        for modality in ("eeg", "hb"):
            identities = set()
            for row in manifest["rows"][modality]:
                if row["dataset"] not in {"eeg_fnirs_single_trial", "simultaneous_eeg_nirs"} or row["split"] not in allowed:
                    raise ValueError("dataset/split outside frozen public source")
                key = (row["dataset"], row["subject"])
                if key in subject_splits and subject_splits[key] != row["split"]:
                    raise ValueError("cross-split subject")
                subject_splits[key] = row["split"]
                if row["id"] in identities:
                    raise ValueError("duplicate native identity")
                identities.add(row["id"])
        manifests[phase] = manifest
    return manifests


def export_one(payload):
    parent, out, seed, modality, limit = payload
    parent, out = Path(parent), Path(out)
    destination = out / f"tokens_seed{seed}_{modality}.npz"
    if destination.exists() and limit is None:
        return read(destination.with_suffix(".json"))
    started = time.monotonic()
    manifests = validate_parent(parent)
    # Use the retained model implementation, not later working-tree changes.
    sys.path.insert(0, str(parent / "source"))
    import torch
    from src.tokenizers.dual_component_tokenizer import DualComponentTokenizer, WaveformDictionary, waveform_windows
    model_source = Path(sys.modules[DualComponentTokenizer.__module__].__file__).resolve()
    if not model_source.is_relative_to((parent / "source").resolve()):
        raise ValueError("frozen model implementation was not imported")
    torch.set_num_threads(1)
    cfg = read(parent / "resolved_config.json")["config"]
    name = "eeg_voltage" if modality == "eeg" else "hb_blocks"
    job = parent / "voltage_jobs" / name / f"seed_{seed}"
    training = read(job / "summary.json")
    if training["status"] != "completed" or training["step"] != cfg["voltage"]["steps"]:
        raise ValueError("completed fixed-budget model required")
    saved = torch.load(job / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = DualComponentTokenizer(saved["model_config"], "continuous_student")
    model.load_state_dict(saved["model_state"])
    model.eval()
    with np.load(job / "dictionaries.npz", allow_pickle=False) as z:
        book = WaveformDictionary.from_state({k: z["S_" + k] for k in
            ("codewords", "value_scale", "delta_scale", "channels", "derivative_weight", "dimension")})
    spec = cfg["voltage"][modality]
    tokens, identities, phases, row_indices = [], [], [], []
    replay_error = 0.0
    read_seconds = 0.0
    for phase, manifest in manifests.items():
        rows = manifest["rows"][modality]
        take = np.array([i for i, r in enumerate(rows) if r["split"] in {"validation", "test"}])
        if limit is not None:
            take = take[:limit]
        folder = "voltage_cache" if phase == "development" else "voltage_test_cache"
        array = np.load(parent / folder / manifest["arrays"][modality + "_u"], mmap_mode="r", allow_pickle=False)
        if list(array.shape) != manifest["shapes"][modality + "_u"]:
            raise ValueError("observation shape drift")
        for begin in range(0, len(take), 128):
            ix = take[begin:begin + 128]
            t = time.monotonic()
            values = torch.from_numpy(np.asarray(array[ix]).copy())
            read_seconds += time.monotonic() - t
            inputs, masks = waveform_windows(values, hop=spec["hop_samples"], span=spec["context_samples"])
            coordinate = np.array([rows[i]["coordinate_index"] for i in ix])
            metadata = torch.from_numpy(np.eye(len(manifest["coordinates"][modality]), dtype="float32")[coordinate])
            metadata = metadata[:, None].expand(-1, inputs.shape[1], -1)
            inputs = torch.cat((inputs, metadata), -1)
            masks = torch.cat((masks, torch.ones_like(metadata, dtype=torch.bool)), -1)
            with torch.inference_mode():
                words = model({modality: inputs}, {modality: masks})[modality]["S"].numpy()
            ids = book.encode(words)
            # Integer round-trip is the same dictionary operation as original evaluation.
            decoded = book.decode(ids)
            manual = sum(book.codewords[level][ids[..., level]] for level in range(book.levels))
            replay_error = max(replay_error, float(np.max(np.abs(decoded - manual))))
            tokens.append(ids)
            identities.extend(rows[i]["id"] for i in ix)
            row_indices.extend(ix.tolist())
            phases.extend([phase] * len(ix))
            if limit is None and begin % 2048 == 0:
                write(out / f"export_progress_{seed}_{modality}.json",
                      dict(phase=phase, completed_cases=begin + len(ix), phase_cases=len(take),
                           elapsed_seconds=time.monotonic() - started))
        if limit is not None:
            break
    ids = np.concatenate(tokens)
    result = dict(status="completed", seed=seed, modality=modality, cases=len(ids), shape=list(ids.shape),
                  elapsed_seconds=time.monotonic() - started, observation_read_seconds=read_seconds,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  integer_decode_max_difference=replay_error, checkpoint=str(job / "checkpoint.pt"),
                  source=str(model_source),
                  input="same-modality observed u only; no S/O targets", benchmark=limit is not None)
    if limit is None:
        np.savez_compressed(destination, ids=ids, row_id=np.array(identities), phase=np.array(phases),
                            row_index=np.array(row_indices))
        write(destination.with_suffix(".json"), result)
    return result


def mi_bits(counts):
    """Plug-in mutual information; bias is compared to whole-trial permutations."""
    c = np.asarray(counts, dtype=np.float64)
    total = c.sum(axis=(-2, -1), keepdims=True)
    marginal = c.sum(-1, keepdims=True) * c.sum(-2, keepdims=True)
    ratio = np.divide(c * total, marginal, out=np.ones_like(c), where=(c > 0) & (marginal > 0))
    return np.sum(c * np.log2(ratio), axis=(-2, -1)) / np.maximum(total[..., 0, 0], 1)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    result = np.empty_like(p)
    result[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return result


def permute_trials(groups, rng):
    order = np.arange(sum(map(len, groups)))
    for group in groups:
        order[group] = rng.permutation(group)
    return order


def sparse_lag(eeg, hb, lag):
    """EEG histograms [trial,ROI,10,K], Hb IDs [trial,ROI,10].

    Exclude cells 0 and 9 because their encoding context is incomplete.
    A positive integer lag means Hb follows EEG by 2*lag seconds.
    """
    if eeg.ndim != 4 or hb.shape != eeg.shape[:3] or eeg.shape[2] != 10:
        raise ValueError("ten aligned 2 s cells per native trial/ROI required")
    positions = np.array([t for t in range(1, 9) if 1 <= t + lag <= 8])
    if not len(positions):
        raise ValueError("lag has no full-context support")
    trial, roi, time_index, code = np.nonzero(eeg[:, :, positions, :])
    time_index = positions[time_index]
    weights = eeg[trial, roi, time_index, code]
    return trial, roi, time_index + lag, code, weights


def lag_counts(sparse, hb, order, size):
    trial, roi, target, code, weights = sparse
    ids = hb[order[trial], roi, target]
    return np.bincount(code * size + ids, weights=weights, minlength=size * size).reshape(size, size)


def expected_counts(eeg, hb, groups, lag, size):
    """Exact expectation under within-stratum random trial pairings."""
    answer = np.zeros((size, size), float)
    eye = np.eye(size)
    for group in groups:
        for t in range(1, 9):
            if not 1 <= t + lag <= 8:
                continue
            for roi in range(eeg.shape[1]):
                left = eeg[group, roi, t].sum(0)
                right = eye[hb[group, roi, t + lag]].mean(0)
                answer += np.outer(left, right)
    return answer


def subject_inputs(parent, out, seed):
    """Join by retained native IDs, then use the recorded six-electrode ROI map."""
    manifests = validate_parent(parent)
    plans = {}
    for phase in ("development", "test"):
        p = read(parent / f"voltage_{phase}_plan.json")
        if p.get("protected_data") != "forbidden":
            raise ValueError("public plan required")
        for row in p["rows"]:
            if row["split"] in {"validation", "test"}:
                plans[row["id"]] = row
    lookup = {}
    for modality in ("eeg", "hb"):
        lookup[modality] = {}
        with np.load(out / f"tokens_seed{seed}_{modality}.npz", allow_pickle=False) as z:
            identities, phases, indices, tokens = (z[k] for k in ("row_id", "phase", "row_index", "ids"))
            for i, identity in enumerate(identities):
                row = manifests[str(phases[i])]["rows"][modality][int(indices[i])]
                if row["id"] != identity or row["split"] not in {"validation", "test"}:
                    raise ValueError("token export native-identity mismatch")
                if identity in lookup[modality]:
                    raise ValueError("duplicate exported identity")
                lookup[modality][str(identity)] = (row, tokens[i].copy())
    subjects = defaultdict(lambda: defaultdict(list))
    for identity, (row, ids) in lookup["hb"].items():
        plan = plans[identity]
        for key in ("dataset", "subject", "record", "split", "trial_id", "coordinate"):
            if plan[key] != row[key]:
                raise ValueError("Hb plan/cache identity mismatch")
        eeg = []
        for channel in plan["anchor"]["eeg_channels"]:
            erow, eids = lookup["eeg"][row["trial_id"] + "__electrode_" + channel]
            if any(erow[k] != row[k] for k in ("dataset", "subject", "record", "split", "trial_id")):
                raise ValueError("EEG/Hb identity mismatch")
            eeg.append(eids)
        if len(eeg) != 6 or ids.shape != (10, 2) or any(e.shape != (200, 4) for e in eeg):
            raise ValueError("retained six-electrode, 100 ms EEG and 2 s Hb contract required")
        # All 20 EEG words in each 2 s cell are retained; this does not interpolate IDs.
        e = np.stack(eeg).reshape(6, 10, 20, 4)
        histogram = np.zeros((10, 128, 4), dtype=np.int16)
        for level in range(4):
            for t in range(10):
                histogram[t, :, level] = np.bincount(e[:, t, :, level].ravel(), minlength=128)
        subjects[(row["dataset"], row["subject"], row["split"])][row["trial_id"]].append((plan, histogram, ids))
    result = []
    for key, trials in sorted(subjects.items()):
        ee, hh, metadata = [], [], []
        for trial_id, rois in sorted(trials.items()):
            rois = sorted(rois, key=lambda x: x[0]["anchor"]["region"])
            if len(rois) != 3 or len({r[0]["anchor"]["region"] for r in rois}) != 3:
                raise ValueError("complete three-ROI trial required")
            ee.append(np.stack([r[1] for r in rois]))
            hh.append(np.stack([r[2] for r in rois]))
            metadata.append(rois[0][0])
        result.append((key, np.stack(ee), np.stack(hh), metadata))
    return result


def analyze_subject(payload):
    key, eeg_all, hb_all, metadata, cfg, seed, out = payload
    started = time.monotonic()
    out = Path(out)
    destination = out / f"subject_{seed}_{key[0]}_{key[1]}.json"
    existing = read(destination) if destination.exists() else None
    if existing and existing.get("pair_support_version") == 2:
        return existing
    group_map = defaultdict(list)
    for i, row in enumerate(metadata):
        group_map[(row["record"], row["condition"], row["session"])].append(i)
    groups = [np.array(v, dtype=int) for v in group_map.values()]
    if any(len(g) < 2 for g in groups):
        raise ValueError("unpermutable singleton stratum; report explicitly rather than silently pooling")
    lags = np.array(cfg["lags_seconds"], int) // 2
    size = eeg_all.shape[3]
    rows, pairs, arrays = [], [], {}
    if existing:
        rows = existing["rows"]
        with np.load(destination.with_suffix(".npz"), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    primary_counts, primary_expected, primary_null_counts = None, None, None
    for el in range(eeg_all.shape[-1]):
        for hl in range(hb_all.shape[-1]):
            primary = [el, hl] == cfg["primary_levels"]
            if existing and not primary:
                continue
            permutations = cfg["primary_permutations"] if primary else cfg["secondary_permutations"]
            eeg, hb = eeg_all[..., el], hb_all[..., hl]
            sparse = [sparse_lag(eeg, hb, lag) for lag in lags]
            observed = np.stack([lag_counts(s, hb, np.arange(len(hb)), size) for s in sparse])
            expected = np.stack([expected_counts(eeg, hb, groups, lag, size) for lag in lags])
            null_mi = np.empty((permutations, len(lags)))
            null_counts = np.empty((permutations, len(lags), size, size), dtype=np.float32) if primary else None
            sequence = np.random.SeedSequence([cfg["permutation_seed"], seed, *"|".join(key).encode()])
            rng = np.random.default_rng(sequence)
            for b in range(permutations):
                order = permute_trials(groups, rng)
                counts = np.stack([lag_counts(s, hb, order, size) for s in sparse])
                null_mi[b] = mi_bits(counts)
                if primary:
                    null_counts[b] = counts
            obs_mi = mi_bits(observed)
            mean = null_mi.mean(0)
            combined = np.concatenate((obs_mi[None], null_mi))
            center = combined.mean(0)
            sd = np.maximum(combined.std(0, ddof=1), 1e-12)
            z = (obs_mi - center) / sd
            null_z = (null_mi - center) / sd
            maximum = null_z.max(1)
            pmax = (1 + np.sum(maximum >= z.max() - 1e-12)) / (permutations + 1)
            for j, lag in enumerate(lags):
                if existing:
                    continue
                rows.append(dict(dataset=key[0], subject=key[1], split=key[2], seed=seed, eeg_level=el, hb_level=hl,
                    lag_s=int(lag * 2), trials=len(metadata), token_pairs=int(observed[j].sum()),
                    mi_bits=float(obs_mi[j]), null_mi_bits=float(mean[j]), excess_mi_bits=float(obs_mi[j] - mean[j]),
                    null_sd=float(sd[j]), p_pointwise=float((1 + np.sum(null_mi[:, j] >= obs_mi[j] - 1e-12)) / (permutations + 1)),
                    p_max_over_lags=float(pmax), permutations=permutations))
            arrays[f"observed_mi_{el}_{hl}"] = obs_mi
            arrays[f"null_mi_{el}_{hl}"] = null_mi
            write(out / f"progress_{seed}_{key[0]}_{key[1]}.json",
                  dict(completed_level_pair=[el, hl], elapsed_seconds=time.monotonic() - started))
            if primary:
                primary_counts, primary_expected, primary_null_counts = observed, expected, null_counts
    observed, expected, null_counts = primary_counts, primary_expected, primary_null_counts
    # Use a label-invariant center and scale, including the actual pairing.
    center = (null_counts.sum(0, dtype=np.float64) + observed) / (len(null_counts) + 1)
    variance = ((np.square(null_counts, dtype=np.float64).sum(0) + observed ** 2)
                - (len(null_counts) + 1) * center ** 2) / len(null_counts)
    sd = np.sqrt(np.maximum(variance, 1.))
    eligible = (observed.sum(-1)[..., None] >= cfg["pair_min_eeg_occurrences"]) & (expected >= cfg["pair_min_expected_count"])
    for j, lag in enumerate(lags):
        positions = [t for t in range(1, 9) if 1 <= t + lag <= 8]
        eeg_present = (eeg_all[..., 0][:, :, positions, :].sum((1, 2)) > 0).sum(0)
        hb_present = np.array([np.any(hb_all[:, :, np.array(positions) + lag, 0] == k, axis=(1, 2)).sum() for k in range(size)])
        eligible[j] &= (eeg_present[:, None] >= cfg["pair_min_trials"]) & (hb_present[None, :] >= cfg["pair_min_trials"])
    eligible &= (lags > 0)[:, None, None]
    z = np.where(eligible, (observed - center) / sd, -np.inf)
    if np.any(eligible):
        maxima = np.full(len(null_counts), -np.inf)
        for b in range(len(null_counts)):
            maxima[b] = np.max(np.where(eligible, (null_counts[b] - center) / sd, -np.inf))
        indices = np.argsort(z.ravel())[-5:][::-1]
        for rank, ix in enumerate(indices, 1):
            j, a, b = np.unravel_index(ix, z.shape)
            if not eligible[j, a, b]:
                continue
            n = observed[j, a].sum()
            p = observed[j, a, b] / n
            p0 = expected[j, a, b] / n
            pairs.append(dict(dataset=key[0], subject=key[1], split=key[2], seed=seed, rank=rank,
                eeg_level=0, hb_level=0, eeg_token=int(a), hb_token=int(b), lag_s=int(2 * lags[j]),
                trials=len(metadata), eeg_occurrences=int(n), pair_occurrences=int(observed[j, a, b]),
                conditional_probability=float(p), matched_null_probability=float(p0),
                marginal_hb_probability=float(observed[j, :, b].sum() / observed[j].sum()),
                probability_difference=float(p - p0), lift=float(p / p0), z=float(z[j, a, b]),
                p_max_pairs_positive_lags=float((1 + np.sum(maxima >= z[j, a, b] - 1e-12)) / (len(null_counts) + 1))))
    arrays["primary_counts"] = observed
    arrays["primary_expected"] = expected
    if not existing:
        np.savez_compressed(destination.with_suffix(".npz"), **arrays)
    else:
        retained_failure = out / "pre_pair_support_fix" / destination.name
        retained_failure.parent.mkdir(exist_ok=True)
        if not retained_failure.exists():
            shutil.copy2(destination, retained_failure)
    result = dict(status="completed", dataset=key[0], subject=key[1], split=key[2], seed=seed,
                  trials=len(metadata), strata=len(groups), stratum_sizes=[len(g) for g in groups],
                  elapsed_seconds=time.monotonic() - started, pair_support_version=2, rows=rows, pairs=pairs)
    if existing:
        result["pair_support_repair"] = "recomputed primary null only; retained MI/count arrays unchanged"
    write(destination, result)
    return result


def run(cfg, parent, out, stage):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "resolved_analysis.json").exists() and read(out / "resolved_analysis.json") != cfg:
        raise ValueError("analysis contract drift; use a new versioned export")
    write(out / "resolved_analysis.json", cfg)
    validate_parent(parent)
    if stage == "benchmark":
        benchmarks = []
        for workers in (1, cfg["resources"]["export_workers"]):
            started = time.monotonic()
            jobs = [(str(parent), str(out), seed, "eeg", 512) for seed in cfg["seeds"]]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(export_one, jobs))
            benchmarks.append(dict(workers=workers, elapsed_seconds=time.monotonic() - started, jobs=results))
        write(out / "benchmark.json", benchmarks)
        print(json.dumps(benchmarks), flush=True)
        return
    if stage in {"export", "all"}:
        jobs = [(str(parent), str(out), seed, modality, None) for seed in cfg["seeds"] for modality in ("eeg", "hb")]
        with ProcessPoolExecutor(max_workers=cfg["resources"]["export_workers"]) as pool:
            for job in as_completed([pool.submit(export_one, x) for x in jobs]):
                print(json.dumps(job.result()), flush=True)
    if stage in {"analyze", "all"}:
        for seed in cfg["seeds"]:
            inputs = subject_inputs(parent, out, seed)
            jobs = [(*x, cfg, seed, str(out)) for x in inputs]
            with ProcessPoolExecutor(max_workers=cfg["resources"]["analysis_workers"]) as pool:
                for job in as_completed([pool.submit(analyze_subject, x) for x in jobs]):
                    result = job.result()
                    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "pairs"}}), flush=True)
        summarize(cfg, parent, out)


def summarize(cfg, parent, out):
    results = [read(p) for p in sorted(out.glob("subject_*.json"))]
    if len(results) != 23 * len(cfg["seeds"]):
        raise ValueError("expected all 23 frozen holdout subjects at every seed")
    rows = [r for result in results for r in result["rows"]]
    pairs = [r for result in results for r in result["pairs"]]
    first = [r for r in pairs if r["seed"] == cfg["primary_seed"] and r["rank"] == 1]
    all_subjects = sorted((r["dataset"], r["subject"]) for r in results if r["seed"] == cfg["primary_seed"])
    lookup = {(r["dataset"], r["subject"]): r for r in first}
    adjusted = holm([lookup[k]["p_max_pairs_positive_lags"] if k in lookup else 1. for k in all_subjects])
    for key, p in zip(all_subjects, adjusted):
        if key in lookup:
            lookup[key]["p_holm_subjects"] = float(p)
    table(out / "subject_lags.csv", rows)
    table(out / "subject_best_pairs.csv", first)
    table(out / "all_pair_examples.csv", [{k: v for k, v in r.items() if k != "p_holm_subjects"} for r in pairs])
    group_rows = []
    for seed in cfg["seeds"]:
        for dataset in sorted({r["dataset"] for r in results}):
            selected = [r for r in results if r["seed"] == seed and r["dataset"] == dataset]
            files = [np.load(out / f"subject_{seed}_{dataset}_{r['subject']}.npz", allow_pickle=False) for r in selected]
            for el in range(4):
                for hl in range(2):
                    observed = np.stack([z[f"observed_mi_{el}_{hl}"] for z in files]).mean(0)
                    null = np.stack([z[f"null_mi_{el}_{hl}"] for z in files]).mean(0)
                    mean = null.mean(0)
                    combined = np.concatenate((observed[None], null))
                    center = combined.mean(0)
                    sd = np.maximum(combined.std(0, ddof=1), 1e-12)
                    maximum = ((null - center) / sd).max(1)
                    z = (observed - center) / sd
                    pmax = (1 + np.sum(maximum >= z.max() - 1e-12)) / (len(null) + 1)
                    for j, lag in enumerate(cfg["lags_seconds"]):
                        group_rows.append(dict(dataset=dataset, seed=seed, eeg_level=el, hb_level=hl,
                            subjects=len(selected), lag_s=lag, mi_bits=float(observed[j]),
                            null_mi_bits=float(mean[j]), excess_mi_bits=float(observed[j] - mean[j]),
                            p_pointwise=float((1 + np.sum(null[:, j] >= observed[j] - 1e-12)) / (len(null) + 1)),
                            p_max_over_lags=float(pmax)))
            for z in files:
                z.close()
    table(out / "group_lags.csv", group_rows)
    write(out / "summary.json", dict(status="completed", experiment_id=cfg["experiment_id"], parent=str(parent),
        subjects=23, trials=sum(r["trials"] for r in results if r["seed"] == cfg["primary_seed"]),
        seeds=cfg["seeds"], primary_pairs=first, group_results=group_rows,
        analysis="exploratory frozen S-token association, not new training or physiological causality",
        completed_parent_denominators={k: read(parent / "voltage_summary.json")[k] for k in ("development_targets", "test_targets")}))


def maximum_test(observed, null):
    """One-sided max statistic with a symmetric studentization across pairings."""
    combined = np.concatenate((np.asarray(observed)[None], null))
    center = combined.mean(0)
    scale = np.maximum(combined.std(0, ddof=1), 1e-12)
    observed_z = (observed - center) / scale
    maxima = ((null - center) / scale).max(1)
    adjusted = (1 + (maxima[:, None] >= observed_z[None] - 1e-12).sum(0)) / (len(null) + 1)
    return float(adjusted.min()), adjusted


def report(cfg, parent, out):
    """Derive direction controls and readable figures from retained count arrays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False})
    results = [read(p) for p in sorted(out.glob("subject_*.json"))]
    if len(results) != 23 * len(cfg["seeds"]):
        raise ValueError("complete declared reanalysis required before reporting")
    lags = np.array(cfg["lags_seconds"])
    positive = lags > 0
    negative_matching = np.array([int(np.flatnonzero(lags == -lag)[0]) for lag in lags[positive]])
    zero = int(np.flatnonzero(lags == 0)[0])
    datasets = sorted({r["dataset"] for r in results})
    names = {"eeg_fnirs_single_trial": "Single-Trial", "simultaneous_eeg_nirs": "Simultaneous"}
    group, subjects, common_rows = [], [], []
    retained = {}
    support_by_subject = {}
    for key, eeg, hb, _ in subject_inputs(parent, out, cfg["primary_seed"]):
        support = []
        for lag in lags // 2:
            positions = np.array([t for t in range(1, 9) if 1 <= t + lag <= 8])
            ep = (eeg[..., 0][:, :, positions, :].sum((1, 2)) > 0).sum(0)
            hp = np.array([np.any(hb[..., 0][:, :, positions + lag] == k, axis=(1, 2)).sum() for k in range(128)])
            support.append((ep[:, None] >= cfg["pair_min_trials"]) & (hp[None] >= cfg["pair_min_trials"]))
        support_by_subject[key[:2]] = np.stack(support)
    for seed in cfg["seeds"]:
        for dataset in datasets:
            selected = [r for r in results if r["seed"] == seed and r["dataset"] == dataset]
            data = []
            for r in selected:
                with np.load(out / f"subject_{seed}_{dataset}_{r['subject']}.npz", allow_pickle=False) as z:
                    data.append({k: z[k] for k in z.files})
            observed = np.stack([z["observed_mi_0_0"] for z in data])
            null = np.stack([z["null_mi_0_0"] for z in data])
            mean_obs, mean_null = observed.mean(0), null.mean(0)
            excess = mean_obs - mean_null.mean(0)
            ppos, adj = maximum_test(mean_obs[positive], mean_null[:, positive])
            pall, _ = maximum_test(mean_obs, mean_null)
            pdir, _ = maximum_test(mean_obs[positive] - mean_obs[negative_matching],
                mean_null[:, positive] - mean_null[:, negative_matching])
            pzero, _ = maximum_test(mean_obs[positive] - mean_obs[zero],
                mean_null[:, positive] - mean_null[:, zero, None])
            secondary_obs = np.stack([np.stack([z[f"observed_mi_{el}_{hl}"] for el in range(4) for hl in range(2)]) for z in data]).mean(0)
            b = cfg["secondary_permutations"]
            secondary_null = np.stack([np.stack([z[f"null_mi_{el}_{hl}"][:b] for el in range(4) for hl in range(2)], axis=1) for z in data]).mean(0)
            p_all_levels, _ = maximum_test(secondary_obs[:, positive].ravel(), secondary_null[:, :, positive].reshape(b, -1))
            p_all_levels_zero, _ = maximum_test((secondary_obs[:, positive] - secondary_obs[:, zero, None]).ravel(),
                (secondary_null[:, :, positive] - secondary_null[:, :, zero, None]).reshape(b, -1))
            p_all_levels_reverse, _ = maximum_test((secondary_obs[:, positive] - secondary_obs[:, negative_matching]).ravel(),
                (secondary_null[:, :, positive] - secondary_null[:, :, negative_matching]).reshape(b, -1))
            best = np.flatnonzero(positive)[np.argmax(excess[positive])]
            group.append(dict(dataset=dataset, seed=seed, subjects=len(selected), trials=sum(r["trials"] for r in selected),
                peak_positive_lag_s=int(lags[best]), peak_excess_mi_bits=float(excess[best]),
                p_positive_lags=ppos, p_all_lags=pall, p_positive_vs_negative=pdir, p_positive_vs_zero=pzero,
                p_all_levels_positive_lags=p_all_levels,
                p_all_levels_positive_vs_zero=p_all_levels_zero, p_all_levels_positive_vs_negative=p_all_levels_reverse,
                positive_adjusted_p=adj.tolist(), lag_s=lags.tolist(), excess_mi_bits=excess.tolist(),
                observed_mi_bits=mean_obs.tolist(), null_mi_bits=mean_null.mean(0).tolist()))
            if seed != cfg["primary_seed"]:
                continue
            retained[dataset] = (selected, data)
            for r, o, n in zip(selected, observed, null):
                p, _ = maximum_test(o[positive], n[:, positive])
                e = o - n.mean(0)
                ix = np.flatnonzero(positive)[np.argmax(e[positive])]
                subjects.append(dict(dataset=dataset, subject=r["subject"], split=r["split"], trials=r["trials"],
                    best_positive_lag_s=int(lags[ix]), peak_excess_mi_bits=float(e[ix]), p_positive_lags=p,
                    excess_mi_bits=e.tolist()))
            # A shared pair is for descriptive person-to-person comparison only.
            counts = np.stack([z["primary_counts"] for z in data])
            expected = np.stack([z["primary_expected"] for z in data])
            denominator = counts.sum(-1)[..., None]
            eligible = (denominator >= cfg["pair_min_eeg_occurrences"]) & (expected >= cfg["pair_min_expected_count"])
            eligible &= positive[None, :, None, None]
            eligible &= np.stack([support_by_subject[(dataset, r["subject"])] for r in selected])
            differences = np.divide(counts - expected, denominator, out=np.zeros_like(counts), where=denominator > 0)
            support = eligible.sum(0)
            avg = np.divide(np.where(eligible, differences, 0).sum(0), support, out=np.full_like(counts[0], -np.inf), where=support >= (len(selected) + 1) // 2)
            j, a, h = np.unravel_index(np.argmax(avg), avg.shape)
            for i, r in enumerate(selected):
                n = denominator[i, j, a, 0]
                if n <= 0:
                    continue
                p, p0 = counts[i, j, a, h] / n, expected[i, j, a, h] / n
                common_rows.append(dict(dataset=dataset, subject=r["subject"], split=r["split"], seed=seed,
                    eeg_level=0, hb_level=0, eeg_token=int(a), hb_token=int(h), lag_s=int(lags[j]),
                    trials=r["trials"], eeg_occurrences=int(n), conditional_probability=float(p),
                    matched_null_probability=float(p0), probability_difference=float(p - p0),
                    sufficient_support=bool(eligible[i, j, a, h]),
                    selection="largest subject-equal positive difference with support in at least half the dataset; descriptive"))
    for r, p in zip(subjects, holm([r["p_positive_lags"] for r in subjects])):
        r["p_holm_subjects"] = float(p)
    for seed in cfg["seeds"]:
        subset = [r for r in group if r["seed"] == seed]
        for field in ("p_positive_lags", "p_positive_vs_negative", "p_positive_vs_zero", "p_all_levels_positive_lags"):
            for r, p in zip(subset, holm([r[field] for r in subset])):
                r[field + "_holm_datasets"] = float(p)
    for field in ("p_all_levels_positive_lags", "p_all_levels_positive_vs_zero", "p_all_levels_positive_vs_negative"):
        for r, p in zip(group, holm([r[field] for r in group])):
            r[field + "_holm_six_panels"] = float(p)
    seed_pair_evidence = []
    for seed in cfg["seeds"]:
        subset = [r for r in results if r["seed"] == seed]
        ps = [r["pairs"][0]["p_max_pairs_positive_lags"] if r["pairs"] else 1. for r in subset]
        adjusted = holm(ps)
        seed_pair_evidence.append(dict(seed=seed, subjects=len(subset),
            significant_subjects=int(np.sum(adjusted <= .05)), minimum_holm_p=float(adjusted.min())))
    table(out / "subject_mi_summary.csv", [{k: v for k, v in r.items() if k != "excess_mi_bits"} for r in subjects])
    table(out / "common_pair_subjects.csv", common_rows)
    table(out / "group_direction_tests.csv", [{k: v for k, v in r.items() if not isinstance(v, list)} for r in group])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, datasets):
        for seed, color, style in zip(cfg["seeds"], ("#005f99", "#cd6419", "#3d7a42"), ("o-", "s--", "^-.")):
            r = next(r for r in group if r["dataset"] == dataset and r["seed"] == seed)
            ax.plot(lags, r["excess_mi_bits"], style, label=f"seed {seed}", color=color, markersize=4)
        ax.axhline(0, color="black", lw=.8)
        ax.axvline(0, color="gray", ls="--", lw=.8)
        ax.set(title=names[dataset], xlabel="fNIRS 相对 EEG 的时间格延迟（秒）", xticks=lags)
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_ylabel("互信息 − 同被试条件随机配对均值（bit）")
    fig.suptitle("S 码本第 1 层：先在被试内计数，再对被试等权平均")
    fig.savefig(out / "group_lag_curves.png", dpi=220)
    plt.close(fig)

    matrix = np.array([r["excess_mi_bits"] for r in subjects])
    bound = max(float(np.max(np.abs(matrix))), 1e-6)
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    labels = [f"{'ST' if r['dataset']==datasets[0] else 'SIM'} {r['subject']}  ({r['split']}, n={r['trials']})" for r in subjects]
    ax.set(xticks=np.arange(len(lags)), xticklabels=lags, yticks=np.arange(len(subjects)), yticklabels=labels,
        xlabel="fNIRS 相对 EEG 的时间格延迟（秒）", title="seed 17，第 1 层 S token：被试差异（统一色标）")
    fig.colorbar(im, ax=ax, label="互信息 − 条件随机配对均值（bit）", shrink=.75)
    fig.savefig(out / "subject_lag_heatmap.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True, sharex=True)
    for ax, dataset in zip(axes, datasets):
        rr = [r for r in common_rows if r["dataset"] == dataset]
        y = np.arange(len(rr))
        ax.barh(y, [100*r["probability_difference"] for r in rr], color=["#005f99" if r["sufficient_support"] else "#b8b8b8" for r in rr])
        ax.axvline(0, color="black", lw=.8)
        ax.set(yticks=y, yticklabels=[r["subject"] for r in rr], xlabel="条件概率 − 条件随机配对概率（百分点）",
               title=f"{names[dataset]}: EEG {rr[0]['eeg_token']} → Hb {rr[0]['hb_token']}, +{rr[0]['lag_s']} s")
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("同一 token 对在不同被试中的表现；数据内选例，灰色为支持不足")
    fig.savefig(out / "common_pair_subjects.png", dpi=220)
    plt.close(fig)

    summary = read(out / "summary.json")
    summary["direction_tests"] = group
    summary["subject_mi_tests"] = subjects
    summary["common_pair_examples"] = common_rows
    summary["pair_evidence_by_seed"] = seed_pair_evidence
    summary["interpretation_limits"] = cfg["claim_limits"]
    summary["figures"] = ["group_lag_curves.png", "subject_lag_heatmap.png", "common_pair_subjects.png"]
    write(out / "summary.json", summary)
    main = [r for r in group if r["seed"] == cfg["primary_seed"]]
    pairs = summary["primary_pairs"]
    significant = sum(r["p_holm_subjects"] <= .05 for r in pairs)
    lines = ["# EEG-S → fNIRS-S token 延迟关联：2026-10-29 探索性检查", "",
        "来源：已完成的 TK-DUAL-VOLTAGE-v2，后续 EEG-O 改进未改变本次 S 分支。只重放冻结模型，不训练或改动 teacher，不评价 EEG-O。",
        f"范围：23 名未参与 tokenizer 训练的被试（验证 11、公开测试 12）；保留 {summary['trials']} / 2091 个计划试次。原 teacher 的 11 个失败不补齐。三个优化种子共享同一被试，不能充当独立复现。", "",
        "**时间和 token 定义。** EEG 每电极每 0.1 秒产生 4 个残差 VQ 层的 S 编号；fNIRS 每 2 秒产生 2 层 S 编号。主分析为两者第 1 层（数组层号 0），其余 7 组层组合分开检查。编号只在相同种子、模态、层内有意义。",
        "每个既有 ROI 的 6 个 EEG 电极共 120 个 token 出现计入一个 2 秒格，并与同试次、同 ROI 的一个 Hb token 配对。延迟是两格中心差，单个 EEG token 的实际偏移相差至多 0.95 秒。各试次不跨窗连接；只用第 2–9 格，保证 EEG 的 2 秒和 Hb 的 6 秒编码输入完整。20 秒试次窗口没有重叠。ROI 可共享电极，时间点、电极、ROI 不作为独立被试。", "",
        "**概率与对照。** 条件概率为指定 EEG token 与指定延迟 Hb token 的共同出现次数 / 该 EEG token 的可配对出现次数。边际 Hb 概率只作描述。主要对照在同被试、记录、条件和会话内随机重配整段 Hb 试次，三个 ROI 一起移动，保留试次内自相关和事件相对时间。随机配对允许原配对；精确期望按每个条件组计算，不按 token 独立打乱。",
        "互信息按每名被试全部配对的 128×128 表计算，以 bit 表示；再对被试等权平均。其有限样本偏差由同样本数的随机配对对照评估。主层用 999 次、其余层用 199 次置换；单侧 p=(1+不小于实测的随机统计量数)/(1+置换次数)。主层的时间扫描用最大统计量校正；具体 token 对扫描同时校正所有满足支持要求的 token 对与正延迟。个体检验再对 23 名被试作 Holm 校正，数据集检验对两库作 Holm 校正。", "",
        "支持要求：EEG 至少 200 次可配对出现、随机配对期望共同出现至少 20 次，两个编号各出现于至少 10 个试次。对应概率的分母不是独立样本数。选出的共享 token 对仅用于描述个体差异；不提供未经校正的生理或人格解释。", "",
        "| 数据集 | 被试 / 试次 | 主层最大正延迟超额 MI | 对应延迟 | 正延迟关联 p（两库校正） | 正延迟胜过反向 p | 正延迟胜过零延迟 p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in main:
        lines.append(f"| {names[r['dataset']]} | {r['subjects']} / {r['trials']} | {r['peak_excess_mi_bits']:.6f} bit | {r['peak_positive_lag_s']} s | {r['p_positive_lags_holm_datasets']:.3f} | {r['p_positive_vs_negative_holm_datasets']:.3f} | {r['p_positive_vs_zero_holm_datasets']:.3f} |")
    lines += ["", "细分层和种子重复属于次要探索。下表的扫描同时覆盖 8 种码本层组合和 4 个正延迟；最后两列还覆盖三种子×两库的六个面板。", "",
        "| 数据集 | 种子 | 全层正延迟扫描 p | 再校正六面板 p | 正延迟优于零延迟 p（六面板） |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for r in group:
        lines.append(f"| {names[r['dataset']]} | {r['seed']} | {r['p_all_levels_positive_lags']:.3f} | {r['p_all_levels_positive_lags_holm_six_panels']:.3f} | {r['p_all_levels_positive_vs_zero_holm_six_panels']:.3f} |")
    lines += ["", "![群体延迟曲线](group_lag_curves.png)", "",
        f"主层具体 token 对在全部正延迟和编号扫描、再经 23 名被试校正后，有 {significant}/23 名被试达到 p≤0.05。完整计数与概率见 [被试最强 token 对](subject_best_pairs.csv)，其余层及优化种子见 [群体方向检验](group_direction_tests.csv) 和 [逐被试逐延迟表](subject_lags.csv)。", "",
        "![被试延迟图](subject_lag_heatmap.png)", "", "![共享 token 对](common_pair_subjects.png)", "",
        "**解释边界。** 高条件概率可能来自 Hb token 本来就常见，因此以同条件随机配对的增量为主。出现正延迟关联，不等于关联只在正延迟存在；须同时看零延迟和反向对照。两个编码器使用窗口上下文，Hb 还来自既有离线预处理，S 目标来自条件模型，因此这些结果不能确定生理传导延迟或因果方向。置换保持试次内相关，仍依赖同条件试次近似可交换；慢漂移可能残留。既有公开测试曾被探索，不是新确认性证据。",
        "同一 token 对的被试间差异是当前数据的描述，尚不能当成稳定个体特征。若要进一步验证，先固定 token 对、方向和延迟，再用新的会话或被试检查。", "",
        "数值事实源：[summary.json](summary.json)；冻结分析合同：[resolved_analysis.json](resolved_analysis.json)；模型、输入来源和资源记录：[launch.json](launch.json)。置换 p 值的加一约定参见 [SciPy 官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)。"]
    lines += ["", "按指定样式输出的 [三联热力图集](heatmaps/index.html) 包含两库和全部 23 名被试的 24 词概览及完整 128 词图。左图为延迟等权边缘化后的条件概率，中图为与同条件随机配对分布的 Jensen–Shannon 差异，右图为完整 fNIRS 词表的固定排序秩期望。中图是描述性分布差异，显著性仍以整段试次置换为准。排序来自训练码字的解码 EEG 电压均值和 HbO 均值。", "",
        "**导出核验。** 原训练未保存逐时刻 ID，本次以冻结模型在 CPU 重放。原始身份及全部验证/测试行一致；108 组码本占用比较中 Hb 完全一致，EEG 单码字计数最大差 6 次，不能据聚合直方图声称逐位置精确复现。差异保留在 [占用核验](token_export_verification.json)。另对两库各一名被试、负/零/正三个延迟直接逐 token 重算，共六组计数与分析数组完全一致，见 [独立计数核验](independent_lag_count_verification.json)。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def conditional_heatmap_values(counts, expected, hb_rank):
    """Uniform-lag joint marginal, matched-null JS and expected ordered Hb rank.

    The input is [lag, EEG word, Hb word]. Normalization within each lag avoids
    giving short lags greater weight merely because more windows can pair.
    """
    counts, expected = np.asarray(counts, float), np.asarray(expected, float)
    if counts.shape != expected.shape or counts.ndim != 3 or np.any(counts < 0) or np.any(expected < 0):
        raise ValueError("matching nonnegative lag contingency arrays required")
    rank = np.asarray(hb_rank, float)
    if rank.shape != (counts.shape[-1],):
        raise ValueError("one fixed rank per Hb word required")
    def normalize(value, axes):
        denominator = value.sum(axis=axes, keepdims=True)
        return np.divide(value, denominator, out=np.zeros_like(value), where=denominator > 0)
    joint = normalize(counts, (-2, -1))
    null_joint = normalize(expected, (-2, -1))
    marginal_joint = joint.mean(0)
    marginal = normalize(marginal_joint, -1)
    p, q = normalize(joint, -1), normalize(null_joint, -1)
    middle = (p + q) / 2
    def kl(a):
        ratio = np.divide(a, middle, out=np.ones_like(a), where=(a > 0) & (middle > 0))
        return np.sum(a * np.log2(ratio), axis=-1)
    js = .5 * (kl(p) + kl(q))
    expectation = p @ rank
    no_support = counts.sum(-1) <= 0
    js[no_support] = np.nan
    expectation[no_support] = np.nan
    marginal[counts.sum((0, 2)) <= 0] = np.nan
    return marginal, js.T, expectation.T, marginal_joint


def heatmaps(cfg, parent, out):
    """Reference-style triptychs, with one training-derived order for all people."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import html
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False})
    seed = cfg["primary_seed"]
    results = [read(p) for p in sorted(out.glob(f"subject_{seed}_*.json"))]
    if len(results) != 23:
        raise ValueError("complete 23-subject primary seed required")
    order, selected, ranking, order_rows = {}, {}, {}, []
    for modality, candidate in (("eeg", "eeg_voltage"), ("hb", "hb_blocks")):
        job = parent / "voltage_jobs" / candidate / f"seed_{seed}"
        health = read(job / "evaluation/summary.json")["codebook_health"]
        occupancy = np.array(next(h["code_counts"] for h in health if
            h["split"] == "train" and h["dataset"] == "pooled" and h["component"] == "S" and h["level"] == 0))
        with np.load(job / "dictionaries.npz", allow_pickle=False) as z:
            words = z["S_codewords"][0]
        coordinate = words.mean(-1) if modality == "eeg" else words.reshape(128, 20, 2)[:, :, 0].mean(-1)
        order[modality] = np.argsort(coordinate, kind="stable")
        ranking[modality] = np.argsort(order[modality])
        frequent = np.argsort(-occupancy, kind="stable")[:24]
        selected[modality] = order[modality][np.isin(order[modality], frequent)]
        for rank, token in enumerate(order[modality]):
            order_rows.append(dict(modality=modality, seed=seed, level=0, original_token_id=int(token),
                full_rank=rank, selected_top24=bool(token in frequent), training_occurrences=int(occupancy[token]),
                decoded_sort_coordinate=float(coordinate[token]),
                sort_meaning="mean EEG voltage in training-RMS coordinate" if modality == "eeg" else "mean HbO in training-SD coordinate"))
    folder = out / "heatmaps"
    folder.mkdir(exist_ok=True)
    table(folder / "token_order.csv", order_rows)
    names = {"eeg_fnirs_single_trial": "Single-Trial", "simultaneous_eeg_nirs": "Simultaneous"}
    panels = []
    by_dataset = defaultdict(list)
    for result in results:
        with np.load(out / f"subject_{seed}_{result['dataset']}_{result['subject']}.npz", allow_pickle=False) as z:
            counts, expected = z["primary_counts"], z["primary_expected"]
        item = dict(dataset=result["dataset"], subject=result["subject"], split=result["split"],
            trials=result["trials"], counts=counts, expected=expected, raw_counts=counts,
            title=f"{names[result['dataset']]} · {result['subject']} · {result['split']} · {result['trials']} 试次")
        by_dataset[result["dataset"]].append(item)
        panels.append(item)
    for dataset, people in by_dataset.items():
        # Equal subjects in the joint distribution, before conditioning on a word.
        counts = np.stack([p["counts"] / p["counts"].sum((-2, -1), keepdims=True) for p in people]).mean(0)
        expected = np.stack([p["expected"] / p["expected"].sum((-2, -1), keepdims=True) for p in people]).mean(0)
        panels.insert(0, dict(dataset=dataset, subject="group", split="validation+test",
            trials=sum(p["trials"] for p in people), counts=counts, expected=expected,
            raw_counts=sum(p["raw_counts"] for p in people),
            title=f"{names[dataset]} · {len(people)} 名被试等权 · {sum(p['trials'] for p in people)} 试次"))
    for p in panels:
        a, b, c, joint = conditional_heatmap_values(p["counts"], p["expected"], ranking["hb"])
        support = p["raw_counts"].sum(-1).T
        b[support < cfg["pair_min_eeg_occurrences"]] = np.nan
        c[support < cfg["pair_min_eeg_occurrences"]] = np.nan
        a[p["raw_counts"].sum((0, 2)) < cfg["pair_min_eeg_occurrences"]] = np.nan
        p["values"] = (a, b, c)
        p["coverage"] = dict(eeg=float(joint.sum(-1)[selected["eeg"]].sum()),
            hb=float(joint.sum(0)[selected["hb"]].sum()))
    # Population means and individual distributions have different dynamic
    # ranges. Every comparable panel shares a scale; no values are clipped.
    scales = {}
    for p in panels:
        for full in (False, True):
            category = ("group" if p["subject"] == "group" else "subject") + ("_all128" if full else "_top24")
            ee, hh = (order["eeg"], order["hb"]) if full else (selected["eeg"], selected["hb"])
            a, b, c = p["values"]
            local = [float(np.nanmax(a[np.ix_(ee, hh)])), float(np.nanmax(b[ee])),
                     float(np.nanmin(c[ee])), float(np.nanmax(c[ee]))]
            if category not in scales:
                scales[category] = local
            else:
                old = scales[category]
                scales[category] = [max(old[0], local[0]), max(old[1], local[1]), min(old[2], local[2]), max(old[3], local[3])]
    artifacts = []
    for p in panels:
        for full in (False, True):
            ee = order["eeg"] if full else selected["eeg"]
            hh = order["hb"] if full else selected["hb"]
            a, b, c = p["values"]
            matrices = [a[np.ix_(ee, hh)], b[ee], c[ee]]
            category = ("group" if p["subject"] == "group" else "subject") + ("_all128" if full else "_top24")
            scale = scales[category]
            scale_label = {"group_all128": "群体128词图", "group_top24": "群体24词图",
                "subject_top24": "个体24词图", "subject_all128": "个体128词图"}[category]
            count = len(ee)
            fig, axes = plt.subplots(1, 3, figsize=(17.5, 7.7 if not full else 13),
                gridspec_kw={"width_ratios": [1.35, 1, 1]})
            fig.subplots_adjust(left=.055, right=.965, top=.81 if not full else .87,
                bottom=.20 if not full else .14, wspace=.35)
            titles = ["EEG × fNIRS：条件概率", "EEG × 延迟：与随机配对的分布差异", "给定 EEG 与延迟的 fNIRS 排序位置期望"]
            cmaps = ["viridis", "magma", "cividis"]
            vmin = [0., 0., float(np.floor(scale[2]))]
            vmax = [scale[0], scale[1], float(np.ceil(scale[3]))]
            labels = ["概率（分母含全部 128 个 fNIRS 词）", "Jensen–Shannon 差异（bit）", "fNIRS 排序秩（0–127）"]
            step = 8 if full else 1
            for i, (ax, values) in enumerate(zip(axes, matrices)):
                cmap = plt.get_cmap(cmaps[i]).copy()
                cmap.set_bad("#cccccc")
                im = ax.imshow(values, origin="upper", aspect="auto", interpolation="nearest", vmin=vmin[i], vmax=vmax[i], cmap=cmap)
                ax.set_title(titles[i], fontsize=12, pad=14)
                ax.set_yticks(np.arange(0, count, step))
                ax.set_yticklabels([str(v) for v in ee[::step]], fontsize=8 if not full else 9)
                if i == 0:
                    ax.set_xticks(np.arange(0, len(hh), step))
                    ax.set_xticklabels([str(v) for v in hh[::step]], rotation=90, fontsize=8 if not full else 9)
                    ax.set_xlabel("fNIRS token ID（按解码 HbO 均值排序）", labelpad=10)
                    ax.set_ylabel("EEG token ID（按解码电压均值排序）", labelpad=10)
                else:
                    ax.set_xticks(np.arange(len(cfg["lags_seconds"])))
                    ax.set_xticklabels(cfg["lags_seconds"])
                    ax.set_xlabel("fNIRS 相对 EEG 的延迟（秒）", labelpad=10)
                bar = fig.colorbar(im, ax=ax, fraction=.047, pad=.04)
                bar.set_label(labels[i], fontsize=10)
                bar.ax.tick_params(labelsize=9)
            shown = "全部 128 个码字" if full else "训练期最常见的 24 / 128 个码字"
            fig.suptitle(p["title"], fontsize=19, y=.965)
            fig.text(.5, .91 if not full else .93,
                f"seed 17 · S 码本第 1 层 · {shown} · 固定排序，{scale_label}内统一色标",
                ha="center", fontsize=12)
            coverage = "全部词表" if full else f"展示码字概率质量：EEG {p['coverage']['eeg']:.1%}，fNIRS {p['coverage']['hb']:.1%}"
            footer = ("左图对延迟等权边缘化后按 EEG 词归一化；中图比较实测配对与同被试/记录/条件/会话随机配对，颜色不表示显著性。\n"
                      "右图对完整 128 词的固定 HbO 排序秩取条件期望。灰色表示 EEG 支持不足（<200 次）；正延迟表示 fNIRS 时间格在后。\n"
                      f"{coverage}。时间格 2 秒，格内偏移 ±0.95 秒。群体/个体、24/128词四组分别使用各自统一色标。")
            fig.text(.055, .045 if not full else .025, footer, fontsize=10, va="bottom", linespacing=1.6)
            name = f"{p['dataset']}_{p['subject']}_{'all128' if full else 'top24'}.png"
            fig.savefig(folder / name, dpi=220)
            plt.close(fig)
            artifacts.append(dict(dataset=p["dataset"], subject=p["subject"], split=p["split"], trials=p["trials"],
                vocabulary_display="all128" if full else "top24_training_frequency", path=name,
                eeg_display_probability_mass=1. if full else p["coverage"]["eeg"],
                hb_display_probability_mass=1. if full else p["coverage"]["hb"]))
    manifest = dict(schema="dual_S_token_lag_triptych_v1", seed=seed, levels=[0, 0], lags_seconds=cfg["lags_seconds"],
        ordering="training codeword mean EEG voltage and mean HbO; not fitted to heldout cooccurrence",
        selected_ids={k: v.tolist() for k, v in selected.items()}, full_order={k: v.tolist() for k, v in order.items()},
        shared_color_scales={k: dict(conditional_probability=[0, v[0]], matched_null_JS_bits=[0, v[1]],
            expected_Hb_rank=[float(np.floor(v[2])), float(np.ceil(v[3]))]) for k, v in scales.items()},
        middle_panel="descriptive JS divergence of conditional distributions, not P(lag|EEG) or a significance map",
        uniform_lag_prior=True, subject_weighting="equal subject joint distributions before row conditioning",
        parent_counts="retained subject_17_* primary_counts and primary_expected; no new model inference", figures=artifacts)
    write(folder / "manifest.json", manifest)
    body = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>S token 延迟三联热力图</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;line-height:1.6;color:#17232c}img{width:100%;height:auto}details{border:1px solid #ccd6dc;margin:15px 0;padding:12px}summary{cursor:pointer;font-weight:600}a{color:#005f99}</style>",
        "<h1>EEG-S 与 fNIRS-S 延迟三联热力图</h1>",
        "<p>左：对不同延迟等权汇总后的 fNIRS 条件概率。中：每个 EEG 词在各延迟下，与同条件随机配对的 fNIRS 分布差异（JS，bit）。右：完整 fNIRS 词表按解码 HbO 均值排序后的位置期望。码字为类别；第三图展示固定排序的秩。</p>",
        "<p>所有图固定 seed 17、第 1 层 S 码本，采用同一训练期排序。群体/个体、24/128词四组分别共用各自的色标，以显示群体均值中的小幅变化；同组各数据集/被试可直接比较。24 词版按训练频率选词；左图仍按完整 128 词归一化。每个被试都有完整 128 词版。中图属于描述性差异，检验结果见<a href='../REPORT.md'>分析报告</a>。"
        "原始 ID 与排序坐标见<a href='token_order.csv'>排序表</a>；范围与数值定义见<a href='manifest.json'>图表记录</a>。</p>"]
    for a in artifacts:
        label = f"{names[a['dataset']]} · {a['subject']} · {a['vocabulary_display']} · {a['trials']} 试次"
        if a["subject"] == "group" and a["vocabulary_display"] != "all128":
            body.append(f"<h2>{html.escape(label)}</h2><a href='{a['path']}'><img src='{a['path']}' alt='{html.escape(label)}'></a>")
        else:
            body.append(f"<details><summary>{html.escape(label)}</summary><a href='{a['path']}'><img loading='lazy' src='{a['path']}' alt='{html.escape(label)}'></a></details>")
    body.append("</html>")
    (folder / "index.html").write_text("\n".join(body))
    summary = read(out / "summary.json")
    summary["requested_heatmaps"] = "heatmaps/manifest.json"
    write(out / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--stage", choices=("benchmark", "export", "analyze", "all", "summarize", "report", "heatmaps"), default="all")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["schema"] != "dual_component_token_lags_v1" or cfg["protected_data"] != "forbidden":
        raise ValueError("named public reanalysis contract required")
    parent, out = ROOT / cfg["parent_run"], ROOT / cfg["output"]
    if args.stage == "heatmaps":
        heatmaps(cfg, parent, out)
    elif args.stage == "report":
        report(cfg, parent, out)
    elif args.stage == "summarize":
        summarize(cfg, parent, out)
    else:
        run(cfg, parent, out, args.stage)


if __name__ == "__main__":
    main()
