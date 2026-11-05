#!/usr/bin/env python3
"""Bounded train/validation diagnostics on the frozen downstream input cache.

The central cache owns signals, labels and partitions. The existing scheduler
owns durable execution; this entry owns only the declared diagnostic arms. Old
production outputs are read-only. No new test-partition arrays are opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler
import yaml

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))
from src.data.downstream_benchmark import (
    EEG_ONLY_CHANNELS, TASKS, VISUAL_LEGACY_PHYSICAL_PANEL, FrozenInputBenchmarkDataset,
)
from src.metrics.downstream_benchmark import classification_baselines, regression_train_mean_baseline, score_predictions
from comparative_methods.downstream_rebench_worker_v1 import ResourceMonitor, fit_input_scaler, fit_neural
from comparative_methods.rebench_baselines_v1 import SignalFeatures
from comparative_methods.run_downstream_rebench_v1 import atomic_json

SCHEMA = "downstream_input_audit_v1"
METHOD_DIR = dict(biot="BIOT", cbramod="CBraMod", reve="REVE", efrm="EFRM-PyTorch", sta_net="STA-Net-PyTorch")
ARRAY_NAMES = ("eeg", "fnirs", "eeg_mask", "fnirs_mask", "target", "mask", "subject_id", "record_id", "sample_id", "partition")


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write(path, value):
    atomic_json(Path(path), serial(value))


def read_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if config.get("schema") == "downstream_input_audit_followup_v1":
        if config.get("parent_config") != "downstream_input_audit_v1.yaml":
            raise ValueError("follow-up must retain the named original diagnostic contract")
        parent = read_config(Path(path).parent / config["parent_config"])
        return {**parent, "experiment_id": config["experiment_id"], "followup": config}
    if (config.get("schema") != SCHEMA or config.get("partitions") != ["train", "validation"]
            or set(config.get("tasks", [])) != set(TASKS)):
        raise ValueError("diagnostic contract must keep all seven tasks and train/validation only")
    if config["stopping"] != "declared_arms_terminal_no_score_driven_extra_search_or_full_production_launch":
        raise ValueError("unknown diagnostic stopping contract")
    return config


def plan(config, project, run):
    if (run / "plan.json").exists():
        previous = json.loads((run / "resolved_config.json").read_text())
        if previous != config:
            raise ValueError("diagnostic contract changed; use a fresh versioned run")
        return json.loads((run / "plan.json").read_text())
    tasks = {}
    for task in config["tasks"]:
        dataset = FrozenInputBenchmarkDataset(project / config["input_cache"], task=task,
                                             splits=config["partitions"], eeg_panel="native")
        records = []
        for row, group in zip(dataset.rows, dataset.groups, strict=True):
            records.append(dict(subject_id=row["subject"], partition=row["split"],
                                dependency_id=row["dependency_id"],
                                source_sample_ids=[dataset.reader.samples[i]["id"] for i in group]))
        partition = dataset.reader.plan["subject_partitions"][dataset.dataset_id]
        if set(partition["train"]) & set(partition["validation"]) or (set(partition["train"]) | set(partition["validation"])) & set(partition["test"]):
            raise ValueError("frozen subject partitions overlap")
        actual = {name: sorted({row["subject_id"] for row in records if row["partition"] == name}) for name in config["partitions"]}
        cache_subjects = {name: sorted({row["subject"] for row in dataset.reader.samples if row["split"] == name}) for name in config["partitions"]}
        if actual != cache_subjects:
            raise ValueError(f"input adapter dropped an admitted task subject: {task}")
        tasks[task] = dict(profile=dataset.profile, samples=records, subjects=actual,
                           counts=dict(Counter(row["partition"] for row in records)),
                           probe_view_count=sum(len(group) for group in dataset.groups),
                           assigned_subjects_without_admitted_task_samples={name: sorted(set(partition[name]) - set(actual[name])) for name in config["partitions"]},
                           test_subjects_not_read=partition["test"])
    result = dict(schema=SCHEMA, input_cache=str((project / config["input_cache"]).resolve()),
                  tasks=tasks, test_access="none; only partition identity metadata inspected",
                  scope=config["scope"], config=str(run / "resolved_config.json"))
    write(run / "resolved_config.json", config)
    write(run / "plan.json", result)
    return result


def prepare(config, project, run, task, output):
    retained = json.loads((run / "plan.json").read_text())["tasks"][task]
    dataset = FrozenInputBenchmarkDataset(project / config["input_cache"], task=task,
                                         splits=config["partitions"], eeg_panel="native")
    if dataset.profile != retained["profile"] or len(dataset) != len(retained["samples"]):
        raise ValueError("frozen input metadata changed after planning")
    arrays = {}
    identities = {key: [] for key in ("subject_id", "record_id", "sample_id", "partition")}
    extrema = {key: [] for key in ("eeg", "fnirs")}
    mask_counts = Counter()
    started = time.monotonic()
    for i, sample in enumerate(dataset):
        if sample["split"] not in config["partitions"]:
            raise ValueError("diagnostic attempted to read an excluded partition")
        expected = retained["samples"][i]
        if sample["source_sample_ids"] != expected["source_sample_ids"] or sample["subject_id"] != expected["subject_id"]:
            raise ValueError("sample identity changed after planning")
        values = dict(eeg=sample["eeg"], fnirs=sample["fnirs"],
                      eeg_mask=sample["channel_valid_mask"]["eeg"], fnirs_mask=sample["channel_valid_mask"]["fnirs"],
                      target=sample["target"], mask=sample["target_valid_mask"])
        for name, value in values.items():
            dtype = (np.bool_ if name.endswith("mask") else
                     (np.int64 if task != "refed_regression" else np.float64) if name == "target" else np.float32)
            value = np.asarray(value, dtype=dtype)
            if name not in arrays:
                arrays[name] = np.lib.format.open_memmap(output / f"{name}.partial.npy", mode="w+", dtype=dtype, shape=(len(dataset), *value.shape))
            arrays[name][i] = value
        for modality in ("eeg", "fnirs"):
            valid = values[modality + "_mask"]
            if not np.isfinite(values[modality][valid]).all():
                raise ValueError("nonfinite measurement on recorded support")
            extrema[modality].append(float(np.max(np.abs(values[modality][valid]))) if valid.any() else None)
            mask_counts[modality + "_missing_points"] += int((~valid).sum())
            mask_counts[modality + "_partial_windows"] += int(not valid.all())
        identities["partition"].append(sample["split"])
        for name in ("subject_id", "record_id", "sample_id"):
            identities[name].append(sample[name])
        if (i + 1) % 256 == 0:
            write(output / "status.json", dict(status="running", samples=i + 1, planned=len(dataset)))
    if len(set(identities["sample_id"])) != len(dataset):
        raise ValueError("duplicate event predictions after probe join")
    if task != "refed_regression" and not bool(arrays["eeg_mask"].all()):
        raise ValueError("classification requires complete EEG support for the named encoders")
    for name, array in arrays.items():
        array.flush()
        os.replace(output / f"{name}.partial.npy", output / f"{name}.npy")
    for name, values in identities.items():
        np.save(output / f"{name}.npy", np.asarray(values), allow_pickle=False)
    elapsed = time.monotonic() - started
    quality = {}
    for modality, values in extrema.items():
        defined = np.asarray([value for value in values if value is not None])
        quality[modality] = dict(unit=dataset.reader.records[dataset.rows[0]["join_key"]].manifest["measurement"][modality + "_preprocessing_state"]["canonical_unit"],
                                 maximum_abs_quantiles=np.quantile(defined, [0, .5, .9, .99, 1]).tolist(),
                                 windows_without_input=sum(value is None for value in values))
    result = dict(task=task, samples=len(dataset), source_probe_views=retained["probe_view_count"],
                  profile=retained["profile"], counts=retained["counts"], subjects=retained["subjects"],
                  masks=dict(mask_counts), quality=quality, arrays={name: name + ".npy" for name in ARRAY_NAMES},
                  seconds=elapsed, samples_per_second=len(dataset) / elapsed,
                  bytes_written=sum((output / (name + ".npy")).stat().st_size for name in ARRAY_NAMES))
    write(output / "array_manifest.json", result)
    return result


def load_arrays(run, task):
    root = run / task / "prepare"
    manifest = json.loads((root / "array_manifest.json").read_text())
    arrays = {name: np.load(root / path, mmap_mode="r", allow_pickle=False) for name, path in manifest["arrays"].items()}
    if set(np.unique(arrays["partition"])) != {"train", "validation"}:
        raise ValueError("diagnostic arrays must contain exactly train/validation")
    return arrays, manifest["profile"]


def view_profile(native, task, panel="matched16", window="full"):
    profile = deepcopy(native)
    requested = (native["channel_names"] if panel == "native" else
                 VISUAL_LEGACY_PHYSICAL_PANEL if panel == "legacy_visual16" and task == "visual" else
                 EEG_ONLY_CHANNELS[TASKS[task]["dataset_id"]] if panel == "matched16" else None)
    if requested is None:
        raise ValueError("unknown diagnostic sensor panel")
    lookup = {name.casefold(): i for i, name in enumerate(native["channel_names"])}
    indices = [lookup[name.casefold()] for name in requested]
    profile.update(channel_names=[native["channel_names"][i] for i in indices],
                   eeg_geometry=[native["eeg_geometry"][i] for i in indices], eeg_panel=panel)
    if window == "event" and task != "refed_regression":
        duration = 2 if task == "dsr" else 8
        profile.update(input_window_schema="downstream_input_event_v1", window_start_s=0.,
                       duration_s=duration, input_samples=duration * 200, fnirs_samples=duration * 10)
    elif window != "full":
        raise ValueError("unexpected diagnostic time window")
    return profile, indices


class ArrayDataset:
    def __init__(self, arrays, native, profile, indices):
        self.arrays, self.profile = arrays, profile
        self.indices = list(map(int, indices))
        lookup = {name.casefold(): i for i, name in enumerate(native["channel_names"])}
        self.channels = [lookup[name.casefold()] for name in profile["channel_names"]]
        offset = profile["window_start_s"] - native["window_start_s"]
        self.slices = {name: slice(round(offset * rate), round((offset + profile["duration_s"]) * rate)) for name, rate in (("eeg", 200), ("fnirs", 10))}
        if offset < 0 or offset + profile["duration_s"] > native["duration_s"]:
            raise ValueError("diagnostic view leaves the frozen input support")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row = self.indices[item]; a = self.arrays
        values = {m: np.asarray(a[m][row, :, self.slices[m]]) for m in ("eeg", "fnirs")}
        masks = {m: np.asarray(a[m + "_mask"][row, :, self.slices[m]]) for m in ("eeg", "fnirs")}
        values["eeg"], masks["eeg"] = values["eeg"][self.channels], masks["eeg"][self.channels]
        return dict(**values, channel_valid_mask=masks, valid_mask={m: mask.any(0) for m, mask in masks.items()},
                    target=a["target"][row], target_valid_mask=a["mask"][row],
                    **{name: str(a[name][row]) for name in ("sample_id", "subject_id", "record_id")})


def batches(dataset, scaler, device, batch_size=16, scale="train_sd"):
    for start in range(0, len(dataset), batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        batch = {}
        masks = {}
        for modality in ("eeg", "fnirs"):
            value = torch.as_tensor(np.stack([s[modality] for s in items]), dtype=torch.float32, device=device)
            mask = torch.as_tensor(np.stack([s["channel_valid_mask"][modality] for s in items]), dtype=torch.bool, device=device)
            stats = scaler[modality]
            value = (value - torch.as_tensor(stats["mean"], device=device, dtype=value.dtype)) / torch.as_tensor(stats["scale"], device=device, dtype=value.dtype)
            if scale == "window_sd" and modality == "eeg":
                count = mask.sum(-1, keepdim=True).clamp_min(1)
                mean = torch.where(mask, value, 0.).sum(-1, keepdim=True) / count
                delta = torch.where(mask, value - mean, 0.)
                sd = (delta.square().sum(-1, keepdim=True) / count).sqrt().clamp_min(1e-6)
                value = delta / sd
            batch[modality] = torch.where(mask, value, 0.)
            masks[modality] = mask
        batch["channel_valid_mask"] = masks
        yield np.asarray(dataset.indices[start:start + batch_size]), batch


def scores(a, indices, prediction, task):
    indices = np.asarray(indices, dtype=int)
    target, mask = a["target"][indices], a["mask"][indices]
    subjects, records = a["subject_id"][indices], a["record_id"][indices]
    result = score_predictions(target, prediction, mask, subjects, task, len(TASKS[task]["classes"]), records)
    if task != "refed_regression":
        classes = np.arange(len(TASKS[task]["classes"]))
        result.update(accuracy=float(accuracy_score(target, prediction)),
                      recall=recall_score(target, prediction, labels=classes, average=None, zero_division=0).tolist(),
                      confusion_matrix=confusion_matrix(target, prediction, labels=classes).tolist(),
                      prediction_counts=np.bincount(prediction, minlength=len(classes)).tolist(),
                      collapsed_subjects=[str(subject) for subject in np.unique(subjects) if len(np.unique(prediction[subjects == subject])) == 1])
    return result


def baseline(a, train, validation, task):
    values = {name: a[name][validation] for name in ("target", "mask", "subject_id", "record_id")}
    if task != "refed_regression":
        return classification_baselines(a["target"][train], values, len(TASKS[task]["classes"]))
    target, mask = a["target"][train], a["mask"][train]
    counts = mask.sum((0, 2))
    mean = np.where(mask, target, 0.).sum((0, 2), dtype=np.float64) / counts
    return regression_train_mean_baseline(mean, counts, values)


def linear_row_coordinates(train, evaluation):
    """Equivalent L2 linear coordinates in the training row space.

    Ordered token features can exceed 100,000 columns. Keeping the nonzero
    training singular directions preserves dot products, predictions and the
    L2 penalty, without fitting thousands of unidentifiable null directions.
    The evaluation matrix never determines the basis or numerical rank.
    """
    x = np.asarray(train, dtype=np.float64)
    if x.shape[1] <= max(4096, 2 * len(x)):
        return train, evaluation, dict(kind="original_features", dimension=x.shape[1])
    z = np.asarray(evaluation, dtype=np.float64)
    gram = x @ x.T
    eigenvalues, vectors = np.linalg.eigh(gram)
    threshold = np.finfo(np.float64).eps * max(x.shape) * max(float(eigenvalues[-1]), 1.)
    retained = eigenvalues > threshold
    if not retained.any():
        return np.zeros((len(x), 1)), np.zeros((len(z), 1)), dict(kind="constant_training_features", dimension=1)
    values, vectors = eigenvalues[retained], vectors[:, retained]
    train_coordinates = vectors * np.sqrt(values)
    eval_coordinates = ((z @ x.T) @ vectors) / np.sqrt(values)
    return train_coordinates, eval_coordinates, dict(kind="training_row_space_equivalent_L2", dimension=len(values),
                                                      source_dimension=x.shape[1], eigenvalue_threshold=threshold)


def linear_grid(features, a, task, config, output, name, *, train=None, validation=None,
                permute=False, fixed=False, feature_mask=None):
    train = np.flatnonzero(a["partition"] == "train") if train is None else np.asarray(train)
    validation = np.flatnonzero(a["partition"] == "validation") if validation is None else np.asarray(validation)
    if set(a["subject_id"][train]) & set(a["subject_id"][validation]):
        raise ValueError("linear diagnostic subject overlap")
    temporal = features.ndim == 3
    x, z = np.asarray(features[train]), np.asarray(features[validation])
    support = np.ones(x.shape[:-1], bool) if feature_mask is None else feature_mask[train]
    scaler = StandardScaler().fit(x.reshape(-1, x.shape[-1])[support.reshape(-1)])
    x = scaler.transform(x.reshape(-1, x.shape[-1])); z = scaler.transform(z.reshape(-1, z.shape[-1]))
    x, z, coordinates = linear_row_coordinates(x, z)
    target = np.asarray(a["target"][train]).copy()
    if permute:
        rng = np.random.default_rng(17)
        for subject in np.unique(a["subject_id"][train]):
            rows = np.flatnonzero(a["subject_id"][train] == subject)
            target[rows] = target[rng.permutation(rows)]
    candidates = []
    best = None
    if task == "refed_regression":
        if not temporal or features.shape[1] != 30:
            raise ValueError("REFED diagnostics require thirty actual one-second feature rows")
        y = target.transpose(0, 2, 1).reshape(-1, 2)
        mask = a["mask"][train].transpose(0, 2, 1).reshape(-1, 2)
        settings = [dict(alpha=value) for value in config["linear"]["ridge_alpha"]]
    else:
        settings = [dict(C=c, class_weight=w) for c in ([1.] if fixed else config["linear"]["C"])
                    for w in ([None] if fixed else [None if w == "none" else w for w in config["linear"]["class_weight"]])]
    for setting in settings:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            if task == "refed_regression":
                predictions, train_predictions = [], []
                for dimension in range(2):
                    valid = mask[:, dimension] & support.reshape(-1)
                    values = y[valid, dimension]; mean = float(values.mean()); scale = max(float(values.std()), 1e-6)
                    model = Ridge(alpha=float(setting["alpha"])).fit(x[valid], (values - mean) / scale)
                    predictions.append(model.predict(z) * scale + mean)
                    train_predictions.append(model.predict(x) * scale + mean)
                pred = np.stack(predictions, -1).reshape(len(validation), 30, 2).transpose(0, 2, 1)
                train_pred = np.stack(train_predictions, -1).reshape(len(train), 30, 2).transpose(0, 2, 1)
            else:
                model = LogisticRegression(**setting, solver="lbfgs", max_iter=config["linear"]["max_iter"], random_state=17).fit(x, target)
                pred, train_pred = model.predict(z), model.predict(x)
        row = dict(setting=setting, train=scores(a, train, train_pred, task), validation=scores(a, validation, pred, task),
                   convergence_warnings=[str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)])
        candidates.append(row)
        score = row["validation"]["primary"]
        if score is not None and (best is None or score > best[0]):
            best = (score, len(candidates) - 1, pred.copy(), train_pred.copy())
    if best is None:
        raise ValueError("no defined validation endpoint; retain failure without dropping a record")
    np.savez(output / f"{name}_predictions.npz", prediction=best[2], train_prediction=best[3],
             target=a["target"][validation], mask=a["mask"][validation],
             sample_id=a["sample_id"][validation], subject_id=a["subject_id"][validation], record_id=a["record_id"][validation],
             train_sample_id=a["sample_id"][train])
    result = dict(selected=candidates[best[1]], candidates=candidates, feature_dim=features.shape[-1], linear_coordinates=coordinates,
                  selected_index=best[1], B0=baseline(a, train, validation, task), training_labels_permuted=permute,
                  train_subjects=sorted(set(a["subject_id"][train])), validation_subjects=sorted(set(a["subject_id"][validation])),
                  training_count=len(train), validation_count=len(validation),
                  decision="development_selected_not_table_ready", prediction_file=f"{name}_predictions.npz")
    write(output / f"{name}.json", result)
    return result


def event_voltage_features(dataset, scaler):
    """Ten measured 100 ms voltage bins at 0–1 s; no learned signal transform."""
    p = dataset.profile
    anchor = round(-p["window_start_s"] * 200)
    output = []
    for _, batch in batches(dataset, scaler, "cpu", batch_size=32):
        eeg = batch["eeg"]
        mask = batch["channel_valid_mask"]["eeg"]
        if not bool(mask.all()):
            raise ValueError("ERP control requires the common complete classification EEG")
        baseline_mean = eeg[..., :anchor].mean(-1, keepdim=True) if anchor else 0.
        bins = (eeg[..., anchor:anchor + 200] - baseline_mean).reshape(len(eeg), eeg.shape[1], 10, 20).mean(-1)
        output.append(bins.flatten(1).numpy())
    return np.concatenate(output)


def run_baselines(config, project, run, task, output):
    a, native = load_arrays(run, task)
    tr = np.flatnonzero(a["partition"] == "train")
    va = np.flatnonzero(a["partition"] == "validation")
    results = {}
    modes = [("whole_window", "matched16", "full", 1), ("native", "native", "full", 1)]
    if task not in {"dsr", "refed_regression"}:
        modes += [("event_window", "matched16", "event", 1), ("temporal", "matched16", "full", 6), ("native_temporal", "native", "full", 6)]
    if task == "visual":
        modes.append(("legacy_visual16", "legacy_visual16", "full", 1))
    for mode, panel, window, segments in modes:
        profile, _ = view_profile(native, task, panel, window)
        profile["classification_segments"] = segments
        dataset = ArrayDataset(a, native, profile, np.arange(len(a["target"])))
        scaler = fit_input_scaler(ArrayDataset(a, native, profile, tr))
        write(output / f"{mode}_input_scaler.json", scaler)
        for modality in ("eeg", "fnirs", "multimodal"):
            # Hb-only is identical across EEG panel controls; no duplicate fits.
            if panel != "matched16" and modality == "fnirs":
                continue
            model = SignalFeatures(task, {**profile, "baseline_modality": modality})
            chunks, supports = [], []
            with torch.no_grad():
                for _, b in batches(dataset, scaler, "cpu", batch_size=32):
                    kwargs = dict(eeg_mask=b["channel_valid_mask"]["eeg"], fnirs_mask=b["channel_valid_mask"]["fnirs"])
                    chunks.append(model(b["eeg"], b["fnirs"], **kwargs).numpy())
                    if task == "refed_regression":
                        supports.append(model.sequence_feature_mask(b["eeg"], b["fnirs"], **kwargs).numpy())
            x = np.concatenate(chunks); support = np.concatenate(supports) if supports else None
            name = f"{mode}_{modality}"
            results[name] = linear_grid(x, a, task, config, output, name, feature_mask=support)
            if (mode == "temporal" or (mode == "whole_window" and task == "dsr")) and modality == "eeg":
                results["permuted_eeg"] = linear_grid(x, a, task, config, output, "permuted_eeg", permute=True, fixed=True)
            if mode == "whole_window" and task != "refed_regression":
                legacy = json.loads((project / config["legacy_splits"]).read_text())["tasks"][task]["folds"][0]["inner_folds"][0]
                old_tr = np.flatnonzero(np.isin(a["subject_id"], legacy["train_subjects"]))
                old_va = np.flatnonzero(np.isin(a["subject_id"], legacy["validation_subjects"]))
                split_name = f"legacy_development_split_{modality}"
                if len(old_tr) and len(old_va):
                    old_scaler = fit_input_scaler(ArrayDataset(a, native, profile, old_tr))
                    old_chunks = []
                    with torch.no_grad():
                        for _, b in batches(dataset, old_scaler, "cpu", batch_size=32):
                            old_chunks.append(model(b["eeg"], b["fnirs"], eeg_mask=b["channel_valid_mask"]["eeg"],
                                                    fnirs_mask=b["channel_valid_mask"]["fnirs"]).numpy())
                    results[split_name] = linear_grid(np.concatenate(old_chunks), a, task, config, output, split_name, train=old_tr, validation=old_va, fixed=True)
                    results[split_name]["interpretation"] = "different_subject_denominators_not_paired_gain_or_legacy_reproduction_all_transforms_refit_on_this_training_subset"
                    write(output / f"{split_name}.json", results[split_name])
                    del old_chunks
            del chunks, x
        if mode == "whole_window" and task != "refed_regression":
            results["event_ERP"] = linear_grid(event_voltage_features(dataset, scaler), a, task, config, output, "event_ERP")
    if task != "refed_regression":
        coverage = np.concatenate([a[modality + "_mask"].mean(-1) for modality in ("eeg", "fnirs")], axis=1)
        results["coverage_only"] = linear_grid(coverage, a, task, config, output, "coverage_only", fixed=True)
    return dict(task=task, results=results, scope=config["scope"])


def efrm_checkpoint(config, project, task):
    manifest = json.loads((project / config["production_manifest"]).read_text())
    expected = f"efrm-pretrain-exclude_{TASKS[task]['dataset_id']}-seed17"
    entry = next(row for row in manifest["jobs"] if row["id"] == expected)
    root = Path(entry["output_dir"])
    result = json.loads((root / "result.json").read_text())
    if result["status"] != "completed":
        raise ValueError("EFRM source checkpoint is not complete")
    checkpoint = root / "pretrained_checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return str(checkpoint)


def run_encoder(config, project, run, task, method, output, device, smoke=False):
    a, native = load_arrays(run, task)
    tr = np.flatnonzero(a["partition"] == "train")
    indices = np.arange(len(a["target"]))
    panels = ["matched16"]
    if method == "cbramod" and task == "visual" and not smoke:
        panels += ["legacy_visual16", "native"]
    results = {}; encoders = {}
    for panel in panels:
        profile, _ = view_profile(native, task, panel)
        profile.update(method_root=str(project / "comparative_methods" / METHOD_DIR[method]))
        scaler = fit_input_scaler(ArrayDataset(a, native, profile, tr))
        write(output / f"{panel}_input_scaler.json", scaler)
        dataset = ArrayDataset(a, native, profile, indices[:32] if smoke else indices)
        scales = ["train_sd"] if smoke else list(config["encoders"]["scales"])
        if method == "cbramod" and task in {"motor_imagery", "mental_arithmetic", "visual"} and panel == "matched16" and not smoke:
            scales += ["physical_100uv"]
        if panel != "matched16":
            scales = ["train_sd"]  # Isolate the physical panel under one scale.
        for scale in scales:
            p = dict(profile)
            if method == "cbramod":
                from comparative_methods.CBraMod.rebench_v1 import create_audit_model
                p.update(audit_readout="flatten", audit_head="none", audit_training_eeg_scaler=scaler["eeg"],
                         audit_scale="physical_100uv" if scale == "physical_100uv" else "train_sd")
                model = create_audit_model(task, p, device, False).eval()
            else:
                module = importlib.import_module(f"comparative_methods.{METHOD_DIR[method]}.rebench_v1")
                if method == "efrm":
                    p["pretrained_checkpoint"] = efrm_checkpoint(config, project, task)
                model = module.create_model(task, p, device, trainable=False).eval()
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise ValueError("linear probe must keep all encoder parameters frozen")
            name = f"{panel}_{scale}"
            features = None; support_chunks = []
            offset = 0; started = time.monotonic()
            with torch.inference_mode():
                for rows, b in batches(dataset, scaler, device, batch_size=8, scale=scale):
                    if method == "efrm":
                        values = model(b["eeg"], b["fnirs"], eeg_mask=b["channel_valid_mask"]["eeg"], fnirs_mask=b["channel_valid_mask"]["fnirs"])
                        if task == "refed_regression":
                            support_chunks.append(model.sequence_feature_mask(b["eeg"], b["fnirs"], eeg_mask=b["channel_valid_mask"]["eeg"], fnirs_mask=b["channel_valid_mask"]["fnirs"]).cpu().numpy())
                    else:
                        values = model(b["eeg"], eeg_mask=b["channel_valid_mask"]["eeg"])
                    values = values.float().cpu().numpy()
                    if not np.isfinite(values).all():
                        raise ValueError("nonfinite encoder features")
                    if features is None:
                        features = np.lib.format.open_memmap(output / f"{name}_features.npy", mode="w+", dtype=np.float32, shape=(len(dataset), *values.shape[1:]))
                    features[offset:offset + len(rows)] = values; offset += len(rows)
                    if offset % 256 == 0:
                        write(output / "status.json", dict(status="running", stage=name, features=offset, planned=len(dataset)))
            features.flush()
            elapsed = time.monotonic() - started
            encoders[name] = dict(profile=p, method=method, input_scale=scale,
                                  extraction_seconds=elapsed, samples_per_second=len(dataset) / elapsed,
                                  feature_shape=list(features.shape), frozen_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                                  provenance=getattr(model, "pretraining_provenance", getattr(model, "metadata", None)))
            if smoke:
                results[name] = dict(scope="throughput_smoke_only", shape=list(features.shape))
            elif method == "cbramod":
                tokens = features.reshape(len(features), len(p["channel_names"]), int(p["duration_s"]), 200)
                readouts = {"mean": tokens.mean((1, 2)), "channel_mean": tokens.mean(2).reshape(len(tokens), -1), "ordered_flatten": features}
                for readout, x in readouts.items():
                    result_name = f"{name}_{readout}"
                    results[result_name] = linear_grid(x, a, task, config, output, result_name)
                del tokens, readouts
            else:
                support = np.concatenate(support_chunks) if support_chunks else None
                results[name] = linear_grid(features, a, task, config, output, name, feature_mask=support)
            del model, features
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return dict(task=task, method=method, results=results, encoders=encoders,
                scope="throughput_smoke_only" if smoke else config["scope"],
                transfer_note="v5_target_excluded_pretraining_to_new_raw_input_domain_not_retrained_cache" if method == "efrm" else "official_frozen_weights")


def run_neural(config, project, run, task, arm, output, device, smoke=False):
    a, native = load_arrays(run, task)
    p, _ = view_profile(native, task)
    tr = np.flatnonzero(a["partition"] == "train"); va = np.flatnonzero(a["partition"] == "validation")
    if smoke:
        # Span the declared subject population: a prefix may contain only one
        # REFED subject with wholly unsupported Hb channels.
        tr = tr[np.linspace(0, len(tr) - 1, min(64, len(tr)), dtype=int)]
        va = va[np.linspace(0, len(va) - 1, min(32, len(va)), dtype=int)]
    train = ArrayDataset(a, native, p, tr); validation = ArrayDataset(a, native, p, va)
    if arm.startswith("cbramod"):
        from comparative_methods.CBraMod.rebench_v1 import create_audit_model
        scaler = fit_input_scaler(train)
        p.update(method_root=str(project / "comparative_methods/CBraMod"), audit_training_eeg_scaler=scaler["eeg"],
                 audit_readout="flatten" if arm == "cbramod_ordered" else "mean",
                 audit_head="source_mlp" if arm == "cbramod_ordered" else "linear")
        module = SimpleNamespace(create_model=create_audit_model, OUTPUT_KIND="predictions")
        method = "cbramod"
    else:
        module = importlib.import_module("comparative_methods.STA-Net-PyTorch.rebench_v1")
        method = "sta_net"
    cfg = config["neural"]
    job = dict(job_id=f"{task}_{arm}", kind="diagnostic", task=task, method=method, seed=cfg["seed"],
               modality_profile="eeg_only" if method == "cbramod" else "multimodal", data_workers=0,
               precision="bf16", batch_size=cfg["batch_size"], microbatch_size=cfg["microbatch_size"],
               gradient_accumulation=cfg["batch_size"] // cfg["microbatch_size"],
               training=dict(max_epochs=2 if smoke else cfg["max_epochs"], min_epochs=2 if smoke else cfg["min_epochs"], patience=cfg["patience"]))
    model, target_scaler, training = fit_neural(module, p, train, validation, job, device, output, arm, cfg[method])
    model.eval(); prediction = []; indices = np.concatenate([tr, va])
    dataset = ArrayDataset(a, native, p, indices)
    with torch.inference_mode():
        for _, batch in batches(dataset, model.input_scaler, device, batch_size=cfg["microbatch_size"]):
            values = model(batch, job)
            prediction.append(target_scaler.inverse(values.float().cpu().numpy()) if task == "refed_regression"
                              else values.argmax(-1).cpu().numpy())
    prediction = np.concatenate(prediction)
    result = dict(train=scores(a, tr, prediction[:len(tr)], task), validation=scores(a, va, prediction[len(tr):], task),
                  training=training, B0=baseline(a, tr, va, task), profile=p, job=job,
                  scope="throughput_smoke_only" if smoke else config["scope"], decision="development_selected_not_table_ready")
    np.savez(output / "predictions.npz", prediction=prediction, sample_id=a["sample_id"][indices],
             subject_id=a["subject_id"][indices], record_id=a["record_id"][indices],
             target=a["target"][indices], mask=a["mask"][indices], partition=a["partition"][indices])
    torch.save(dict(model=model.state_dict(), profile=p, job=job, input_scaler=model.input_scaler,
                    target_scaler=target_scaler.state(), best_epoch=training["best_epoch"]), output / "checkpoint.pt")
    return result


def run_brainfusion(config, project, run, task, output, device, physical=False):
    a, native = load_arrays(run, task); p, _ = view_profile(native, task)
    p["task"] = task
    p["nvc_hrf_grid"] = "physical_common_grid_v1" if physical else "retained_sample_index_v1"
    module = importlib.import_module("comparative_methods.BrainFusion-NVC-CSP-Stacking.rebench_v1")
    tr = np.flatnonzero(a["partition"] == "train"); va = np.flatnonzero(a["partition"] == "validation")
    model = module.create_estimator(p, seed=17, device=device)
    dataset = ArrayDataset(a, native, p, range(len(a["target"])))
    channels = dataset.channels
    def values(rows):
        return [torch.as_tensor(np.asarray(a["eeg"][rows])[:, channels]), torch.as_tensor(np.asarray(a["fnirs"][rows]))]
    def masks(rows):
        return dict(eeg_mask=torch.as_tensor(np.asarray(a["eeg_mask"][rows])[:, channels]), fnirs_mask=torch.as_tensor(np.asarray(a["fnirs_mask"][rows])))
    model.fit(*values(tr), a["target"][tr], subjects=a["subject_id"][tr], sample_ids=a["sample_id"][tr], **masks(tr))
    results = dict(B0=baseline(a, tr, va, task))
    for name, rows in (("train", tr), ("validation", va)):
        prediction = model.predict(*values(rows), **masks(rows))
        results[name] = scores(a, rows, prediction, task)
        # Diagnostic use of the already fitted model, never validation fitting.
        inputs = model._prediction_inputs(*values(rows), **masks(rows))
        features = model.pipeline.features.transform(*inputs)
        components = {}
        for view in module.VIEW_ORDER:
            fitted = model.pipeline.stacking
            x = features[view].cpu().numpy()
            pred = fitted.base_estimators_[view].predict(fitted.view_scalers_[view].transform(x))
            components[view] = scores(a, rows, pred, task)
        results[name + "_components"] = components
        np.savez(output / f"{name}_predictions.npz", sample_id=a["sample_id"][rows], subject_id=a["subject_id"][rows],
                 prediction=prediction, target=a["target"][rows], mask=a["mask"][rows], record_id=a["record_id"][rows])
    results.update(fit_audit=model.audit_state(), prediction_degeneracy=model.last_prediction_degeneracy_, profile=p,
                   scope=config["scope"], decision="development_diagnostic_not_table_ready")
    model.save(output / "checkpoint")
    return results


def build_schedule(config, project, run, phase, source):
    """Resolve a finite DAG for the existing resource/resume controller."""
    jobs = []
    def add(task, arm, gpu, ram, dependencies=(), smoke=False):
        name = ("smoke_" if smoke else "") + arm
        job_id = f"{task}_{name}"
        command = [str(project / ".venv/bin/python"), str(source / "comparative_methods/performance_analysis/audit_downstream_inputs_v1.py"),
                   "--config", str(source / "comparative_methods/downstream_input_audit_v1.yaml"), "--project-root", str(project),
                   "--run", str(run), "--stage", arm, "--task", task, "--device", "{device}"]
        if smoke:
            command.append("--smoke")
        jobs.append(dict(id=job_id, kind="diagnostic", method=arm, task=task, command=command, cwd=str(source),
                         output_dir=str(run / task / name), dependencies=list(dependencies),
                         resources=dict(cpu_threads=config["resources"]["threads_per_job"], ram_gib=ram,
                                        gpu=gpu, gpu_memory_mib=5000 if gpu else 0), timeout_seconds=14400,
                         timing_kind=arm))
    if phase == "prepare":
        for task in config["tasks"]:
            add(task, "prepare", False, 8)
    elif phase == "smoke":
        for method in config["encoders"]["methods"]:
            add("motor_imagery", method + "_lp", True, 8, smoke=True)
        for arm in config["neural"]["methods"]:
            add("motor_imagery", arm, True, 12, smoke=True)
    elif phase == "screen":
        for task in config["tasks"]:
            add(task, "baselines", False, 12)
        for method in config["encoders"]["methods"]:
            for task in config["tasks"]:
                if task != "refed_regression" or method == "efrm":
                    add(task, method + "_lp", True, 14)
        for task in config["brainfusion"]["tasks"]:
            add(task, "brainfusion", True, 12)
        for task in config["neural"]["tasks"]:
            for arm in config["neural"]["methods"]:
                add(task, arm, True, 14)
    elif phase == "followup":
        if "followup" not in config:
            raise ValueError("follow-up schedule needs its versioned contract")
        add("refed_regression", "sta_net", True, 14, smoke=True)
        for task in config["followup"]["brainfusion"]["tasks"]:
            add(task, "brainfusion_physical", True, 12)
        add("refed_regression", "sta_net", True, 14, dependencies=["refed_regression_smoke_sta_net"])
    else:
        raise ValueError("unknown execution phase")
    cfg = config["resources"]
    # Each phase has its own controller status and lock; artifacts remain under
    # this one owning run. Phases are launched sequentially after the prior gate.
    controller_root = run / phase
    controller_root.mkdir(exist_ok=True)
    manifest = dict(schema="downstream_rebench_schedule_v1", run_id=f"{SCHEMA}_{phase}", run_root=str(controller_root),
                    source_root=str(source), scope=config["scope"], jobs=jobs,
                    limits=dict(cpu_threads=cfg["cpu_thread_budget"], ram_gib=cfg["ram_budget_gib"], gpu_ids=cfg["gpus"] if phase != "prepare" else [],
                                max_gpu_jobs_per_device=cfg["max_gpu_jobs_per_device"], gpu_reserve_mib={str(k): v for k, v in cfg["gpu_reserve_mib"].items()} if phase != "prepare" else {}))
    if phase == "followup":
        for job in manifest["jobs"]:
            position = job["command"].index("--config") + 1
            job["command"][position] = str(source / "comparative_methods/downstream_input_followup_v1.yaml")
    destination = controller_root / "jobs.json"
    if destination.exists() and json.loads(destination.read_text()) != manifest:
        raise ValueError("execution DAG changed; retain the old manifest and use a versioned phase")
    write(destination, manifest)
    return dict(manifest=str(destination), jobs=len(jobs), source=str(source))


def aggregate(config, run):
    manifests = [run / phase / "jobs.json" for phase in ("screen", "followup") if (run / phase / "jobs.json").exists()]
    scheduled = [job for path in manifests for job in json.loads(path.read_text())["jobs"]]
    jobs = [job for job in scheduled if "_smoke_" not in job["id"]]
    completed = {}; failures = []
    for job in jobs:
        path = Path(job["output_dir"]) / "result.json"
        if not path.exists():
            failures.append(dict(job=job["id"], status="missing", result=str(path)))
            continue
        row = json.loads(path.read_text())
        if row["status"] != "completed":
            failures.append(dict(job=job["id"], status=row["status"], error=row.get("error")))
        else:
            completed[job["id"]] = dict(result_path=str(path), result=row["result"], resources=row.get("resources"))
    summary = dict(schema=SCHEMA, status="completed" if not failures else "incomplete_or_failed", planned_jobs=len(jobs),
                   completed_jobs=len(completed), failures=failures, results=completed,
                   schedules=[str(path) for path in manifests],
                   scope=config["scope"], interpretation="Validation-selected diagnostics; preserve every declared method/task/seed and failed arm. No final test scores or paper admission.")
    write(run / "summary.json", summary)
    rows = []
    for job_id, item in completed.items():
        result = item["result"]
        task = next(job["task"] for job in jobs if job["id"] == job_id)
        cases = result.get("results", {"model": result})
        for name, value in cases.items():
            selected = value.get("selected", value)
            metric = selected["validation"]; b0 = value["B0"]["b0"]
            row = dict(job=job_id, task=task, arm=name, metric=metric["primary_name"],
                       train_value=selected["train"]["primary"], validation_value=metric["primary"], B0=b0,
                       delta_from_B0=metric["primary"] - b0, subject_count=metric["subject_count"],
                       validation_samples=metric["sample_count"],
                       selected_setting=json.dumps(selected.get("setting", {"best_epoch": result.get("training", {}).get("best_epoch")}), sort_keys=True),
                       mae=metric.get("mae"), train_mean_mae=value["B0"].get("mae"),
                       rmse=metric.get("rmse"), train_mean_rmse=value["B0"].get("rmse"),
                       convergence_warnings=len(selected.get("convergence_warnings", [])),
                       source_result=item["result_path"], decision="development_selected_not_final_test")
            if name.startswith("legacy_development_split"):
                row["decision"] = "split_sensitivity_different_subject_denominators"
            elif name.startswith("legacy_visual16"):
                row["decision"] = "legacy_electrode_selection_control_not_production"
            if job_id.endswith("_brainfusion"):
                row["decision"] = "legacy_HRF_time_grid_control_not_production"
            rows.append(row)
    if rows:
        with (run / "cells.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    controls = paired_controls(completed)
    write(run / "paired_controls.json", controls)
    return dict(status=summary["status"], planned_jobs=summary["planned_jobs"], completed_jobs=len(completed), failures=failures)


def paired_controls(completed):
    """Prespecified same-setting contrasts; never compare two selected maxima."""
    rows = []
    def candidate(case, weight=None):
        return next(row["validation"] for row in case["candidates"] if row["setting"] == {"C": 1., "class_weight": weight})
    def pair(task, name, left, right, setting):
        a = {row["subject_id"]: row["score"] for row in left["subjects"]}
        b = {row["subject_id"]: row["score"] for row in right["subjects"]}
        if set(a) != set(b):
            raise ValueError("paired control changed the validation subjects")
        subjects = sorted(a); delta = np.asarray([b[s] - a[s] for s in subjects])
        draws = np.random.default_rng(20261105).integers(0, len(delta), size=(2000, len(delta)))
        rows.append(dict(task=task, control=name, setting=setting, before=left["primary"], after=right["primary"],
                         delta=float(delta.mean()), subject_count=len(subjects),
                         subject_deltas=dict(zip(subjects, delta.tolist())),
                         conditional_subject_bootstrap_95pct=np.quantile(delta[draws].mean(1), [.025, .975]).tolist()))
    for task in TASKS:
        item = completed.get(task + "_cbramod_lp")
        if item:
            cases = item["result"]["results"]
            for name, before, after in (
                ("CBraMod_mean_to_ordered_same_train_scale", "matched16_train_sd_mean", "matched16_train_sd_ordered_flatten"),
                ("CBraMod_mean_to_channel_mean_same_train_scale", "matched16_train_sd_mean", "matched16_train_sd_channel_mean"),
                ("CBraMod_ordered_train_scale_to_window_scale", "matched16_train_sd_ordered_flatten", "matched16_window_sd_ordered_flatten"),
            ):
                pair(task, name, candidate(cases[before]), candidate(cases[after]), dict(C=1., class_weight=None))
        legacy = completed.get(task + "_brainfusion")
        physical = completed.get(task + "_brainfusion_physical")
        if legacy and physical:
            pair(task, "BrainFusion_HRF_time_grid_only", legacy["result"]["validation"], physical["result"]["validation"],
                 dict(seed=17, changed="HRF_kernel_sampling_period_1_to_0p1_seconds", all_other_hyperparameters="same"))
    for job_id, pairs in (
        ("visual_baselines", [("EEG_statistics", "legacy_visual16_eeg", "whole_window_eeg"),
                              ("multimodal_statistics", "legacy_visual16_multimodal", "whole_window_multimodal")]),
        ("visual_cbramod_lp", [("CBraMod_" + name, "legacy_visual16_train_sd_" + name, "matched16_train_sd_" + name)
                              for name in ("mean", "channel_mean", "ordered_flatten")]),
    ):
        if job_id not in completed:
            continue
        cases = completed[job_id]["result"]["results"]
        for name, before, after in pairs:
            for weight in (None, "balanced"):
                pair("visual", "physical_electrode_correction_" + name, candidate(cases[before], weight), candidate(cases[after], weight),
                     dict(C=1., class_weight=weight, changed_physical_electrodes=6, same_input_events_Hb_and_partitions=True))
    return dict(scope="descriptive_paired_development_controls_not_final_test", controls=rows,
                interval="2000 subject-block percentile resamples, seed20261105, conditional on the fitted models and fixed C1; no refitting, no multiple-comparison claim, small subject denominators retained")


def production_diagnostics(run):
    """Inspect only the completed public predictions named by the snapshot."""
    snapshot = run / "production_snapshot_v1/summary.json"
    summary = json.loads(snapshot.read_text())
    groups = {}
    for source in summary["source_inputs"]:
        if not source["valid"]:
            continue
        result = json.loads(Path(source["result_path"]).read_text())
        job = json.loads(Path(source["job_path"]).read_text())
        modality = job.get("method_profile", {}).get("baseline_modality", "eeg") if result["method"] == "signal_linear" else "not_applicable"
        key = (result["method"], result["track"], result["task"], modality)
        rows = groups.setdefault(key, [])
        train = result.get("training", {})
        selection = result.get("selection", [])
        row = dict(job=source["schedule_job_id"], result_path=source["result_path"],
                   primary=result["metrics"]["primary"], best_epoch=train.get("best_epoch"),
                   selected_candidate=result.get("selected_candidate"),
                   final_training_loss=train.get("history", [{}])[-1].get("train_loss") if train.get("history") else None,
                   highest_inner_primary=max([s["mean_primary"] for s in selection if isinstance(s, dict) and s.get("mean_primary") is not None], default=None))
        if result["method"] == "brainfusion":
            nvc = result.get("audit", {}).get("spec", {}).get("nvc_config", {})
            row["nvc_config"] = nvc
            row["hrf_grid_matches_signal"] = bool(nvc) and math.isclose(
                nvc["hrf_tr"] / nvc["hrf_oversampling"], 1. / nvc["fnirs_sampling_rate_hz"])
        if result["task"] != "refed_regression":
            with np.load(source["predictions_path"], allow_pickle=False) as a:
                labels = np.asarray(a["target"]); prediction = np.asarray(a["prediction"]); subjects = np.asarray(a["subject_id"])
                nclasses = len(TASKS[result["task"]]["classes"])
                row.update(confusion_matrix=confusion_matrix(labels, prediction, labels=np.arange(nclasses)).tolist(),
                           recall=recall_score(labels, prediction, labels=np.arange(nclasses), average=None, zero_division=0).tolist(),
                           prediction_counts=np.bincount(prediction.astype(int), minlength=nclasses).tolist(),
                           collapsed_subject_count=sum(len(np.unique(prediction[subjects == s])) == 1 for s in np.unique(subjects)),
                           subject_count=len(np.unique(subjects)), pooled_balanced_accuracy=result["metrics"].get("pooled_balanced_accuracy"))
        rows.append(row)
    cells = []
    for cell in summary["cells"]:
        key = (cell["method"], cell["track"], cell["task"], cell["baseline_modality"])
        jobs = groups.get(key, [])
        row = {k: cell[k] for k in ("method", "task", "track", "baseline_modality", "value", "subject_count", "completed_job_count", "planned_job_count", "coverage_complete", "decision")}
        row.update(B0=cell["B0"]["value"], jobs=jobs)
        if jobs and "confusion_matrix" in jobs[0]:
            confusion = np.sum([j["confusion_matrix"] for j in jobs], axis=0)
            row.update(pooled_confusion_across_seed_repeats=confusion,
                       pooled_recall=np.divide(np.diag(confusion), confusion.sum(1), out=np.zeros(len(confusion), float), where=confusion.sum(1) > 0),
                       collapsed_subject_seed_count=sum(j["collapsed_subject_count"] for j in jobs),
                       subject_seed_count=sum(j["subject_count"] for j in jobs))
        if cell["task"] == "visual" and cell["baseline_modality"] != "fnirs":
            row["input_contract_decision"] = "FAILURE_RESULT_known_legacy_physical_electrode_mismatch"
        else:
            row["input_contract_decision"] = "retained_input_identity_no_new_claim_of_method_capacity"
        if cell["method"] == "brainfusion" and any(not j.get("hrf_grid_matches_signal", True) for j in jobs):
            row["method_adaptation_decision"] = "FAILURE_RESULT_HRF_kernel_sample_period_differs_from_common_signal_grid"
        cells.append(row)
    full = [c for c in cells if c["coverage_complete"]]
    result = dict(schema=SCHEMA, source=str(snapshot), scope="read_only_public_production_snapshot",
                  completed_cells=len(full), planned_cells=len(cells),
                  completed_cells_at_or_below_B0=sum(c["value"] <= c["B0"] for c in full),
                  completed_cells_below_admission_line=sum(c["decision"] == "REJECTED_VALUE" for c in full),
                  cells=cells, warning="Numerical admission in the historical aggregator does not validate the later-discovered Visual physical-channel defect; all such EEG-dependent values remain failure evidence.")
    write(run / "production_diagnostics.json", result)
    return {k: result[k] for k in ("completed_cells", "planned_cells", "completed_cells_at_or_below_B0", "completed_cells_below_admission_line")}


def independent_primary(target, prediction, mask, subjects, records, task):
    """Independent arithmetic check of retained prediction reductions."""
    values = []
    for subject in np.unique(subjects):
        chosen = subjects == subject
        if task != "refed_regression":
            keep = chosen & mask
            y, p = target[keep], prediction[keep]
            scores = []
            for label in range(len(TASKS[task]["classes"])):
                tp = ((y == label) & (p == label)).sum()
                denominator = (y == label).sum() + (p == label).sum()
                scores.append(2 * tp / denominator if denominator else 0.)
            values.append(float(np.mean(scores)))
        else:
            record_scores = []
            for record in np.unique(records[chosen]):
                selected = chosen & (records == record)
                dimensions = []
                for coordinate in range(2):
                    keep = mask[selected, coordinate]
                    y = np.asarray(target[selected, coordinate][keep], dtype=np.float64)
                    p = np.asarray(prediction[selected, coordinate][keep], dtype=np.float64)
                    denominator = y.var() + p.var() + (y.mean() - p.mean()) ** 2
                    if len(y) < 2 or denominator <= 0:
                        raise ValueError("undefined retained REFED endpoint")
                    dimensions.append(2 * np.mean((y - y.mean()) * (p - p.mean())) / denominator)
                record_scores.append(np.mean(dimensions))
            values.append(float(np.mean(record_scores)))
    return float(np.mean(values))


def verify_predictions(run):
    """Recheck complete job outputs against the single prepared identity owner."""
    manifests = [run / phase / "jobs.json" for phase in ("screen", "followup") if (run / phase / "jobs.json").exists()]
    jobs = [job for path in manifests for job in json.loads(path.read_text())["jobs"] if "_smoke_" not in job["id"]]
    checked, pending, errors = [], [], []
    truth = {}
    for job in jobs:
        task = job["task"]; directory = Path(job["output_dir"])
        result_path = directory / "result.json"
        if not result_path.exists() or json.loads(result_path.read_text()).get("status") != "completed":
            pending.append(job["id"]); continue
        if task not in truth:
            root = run / task / "prepare"
            names = ("target", "mask", "sample_id", "subject_id", "record_id", "partition")
            a = {key: np.load(root / f"{key}.npy", mmap_mode="r", allow_pickle=False) for key in names}
            truth[task] = (a, {str(identity): i for i, identity in enumerate(a["sample_id"])})
        a, lookup = truth[task]
        result = json.loads(result_path.read_text())["result"]
        if "results" in result:
            cases = [(name, directory / value["prediction_file"], value["selected"]["validation"]["primary"],
                      not name.startswith("legacy_development_split")) for name, value in result["results"].items()]
        elif "validation" in result:
            name = "validation_predictions.npz" if job["method"].startswith("brainfusion") else "predictions.npz"
            cases = [(job["method"], directory / name, result["validation"]["primary"], True)]
        else:
            errors.append(dict(job=job["id"], error="unrecognized completed endpoint")); continue
        for name, path, expected, common_validation in cases:
            try:
                with np.load(path, allow_pickle=False) as saved:
                    prediction = np.asarray(saved["prediction"]); ids = np.asarray(saved["sample_id"]).astype(str)
                    indices = np.asarray([lookup[value] for value in ids])
                    if "partition" in saved:
                        select = np.asarray(saved["partition"]) == "validation"
                        ids, indices, prediction = ids[select], indices[select], prediction[select]
                        target, mask = np.asarray(saved["target"])[select], np.asarray(saved["mask"])[select]
                    else:
                        target, mask = np.asarray(saved["target"]), np.asarray(saved["mask"])
                    if len(set(ids)) != len(ids):
                        raise ValueError("duplicate predicted event identity")
                    if common_validation and set(ids) != set(a["sample_id"][a["partition"] == "validation"]):
                        raise ValueError("method changed the common validation denominator")
                    if not np.array_equal(target, a["target"][indices], equal_nan=True) or not np.array_equal(mask, a["mask"][indices]):
                        raise ValueError("retained target/mask differs from the frozen prepared owner")
                    if prediction.shape != target.shape or not np.isfinite(prediction[mask]).all():
                        raise ValueError("invalid prediction shape or scored values")
                    actual = independent_primary(target, prediction, mask, a["subject_id"][indices], a["record_id"][indices], task)
                    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8):
                        raise ValueError(f"primary disagrees with predictions: {actual} vs {expected}")
                    if "train_sample_id" in saved:
                        training = np.asarray([lookup[str(value)] for value in saved["train_sample_id"]])
                        if set(a["subject_id"][training]) & set(a["subject_id"][indices]):
                            raise ValueError("training and validation subjects overlap")
                    checked.append(dict(job=job["id"], arm=name, samples=len(ids), subjects=len(set(a["subject_id"][indices])),
                                        recomputed_primary=actual, result_path=str(result_path), predictions=str(path)))
            except Exception as error:
                errors.append(dict(job=job["id"], arm=name, error=repr(error)))
    result = dict(schema=SCHEMA, status="failed" if errors else "passed_completed_outputs", planned_jobs=len(jobs),
                  checked_endpoints=len(checked), pending_or_failed_jobs=pending, errors=errors, checks=checked,
                  scope="independent_metric_arithmetic_and_exact_target_mask_subject_event_denominators; no test arrays")
    write(run / "verification.json", result)
    return {k: result[k] for k in ("status", "planned_jobs", "checked_endpoints", "pending_or_failed_jobs", "errors")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--task", choices=list(TASKS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase", choices=["prepare", "smoke", "screen", "followup"])
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args(); config = read_config(args.config)
    run, project = args.run.resolve(), args.project_root.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if args.stage == "plan":
        result = plan(config, project, run)
        print(json.dumps({task: v["counts"] for task, v in result["tasks"].items()})); return
    if args.stage == "schedule":
        print(json.dumps(build_schedule(config, project, run, args.phase, args.source_root.resolve()))); return
    if args.stage == "aggregate":
        print(json.dumps(aggregate(config, run))); return
    if args.stage == "production":
        print(json.dumps(production_diagnostics(run))); return
    if args.stage == "verify":
        print(json.dumps(verify_predictions(run))); return
    arms = {"prepare", "baselines", "brainfusion", *config["neural"]["methods"], *[m + "_lp" for m in config["encoders"]["methods"]]}
    if "followup" in config:
        arms.add("brainfusion_physical")
    if args.stage not in arms or args.task is None:
        raise ValueError("known task and diagnostic stage required")
    if args.task == "refed_regression" and args.stage not in {"prepare", "baselines", "efrm_lp"} and not ("followup" in config and args.stage == "sta_net"):
        raise ValueError("unsupported temporal output; do not fabricate a REFED head")
    output = run / args.task / (("smoke_" if args.smoke else "") + args.stage)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists() and json.loads((output / "result.json").read_text())["status"] == "completed":
        return
    torch.set_num_threads(config["resources"]["threads_per_job"])
    torch.manual_seed(17); np.random.seed(17)
    write(output / "status.json", dict(status="running", task=args.task, arm=args.stage, pid=os.getpid(), started_unix=time.time(), source=str(Path(__file__).resolve())))
    try:
        with ResourceMonitor(args.device) as monitor:
            if args.stage == "prepare":
                result = prepare(config, project, run, args.task, output)
            elif args.stage == "baselines":
                result = run_baselines(config, project, run, args.task, output)
            elif args.stage.endswith("_lp"):
                result = run_encoder(config, project, run, args.task, args.stage[:-3], output, args.device, args.smoke)
            elif args.stage in {"brainfusion", "brainfusion_physical"}:
                result = run_brainfusion(config, project, run, args.task, output, args.device, args.stage == "brainfusion_physical")
            else:
                result = run_neural(config, project, run, args.task, args.stage, output, args.device, args.smoke)
            resources = monitor.snapshot()
        write(output / "resource_usage.json", resources)
        write(output / "result.json", dict(schema=SCHEMA, status="completed", task=args.task, arm=args.stage, result=result, resources=resources,
                                          source=str(Path(__file__).resolve()), config=str(args.config.resolve())))
        write(output / "status.json", dict(status="completed", task=args.task, arm=args.stage))
    except Exception as error:
        failure = dict(schema=SCHEMA, status="failed", task=args.task, arm=args.stage, error=repr(error), source=str(Path(__file__).resolve()))
        write(output / "result.json", failure); write(output / "status.json", failure)
        raise


if __name__ == "__main__":
    main()
