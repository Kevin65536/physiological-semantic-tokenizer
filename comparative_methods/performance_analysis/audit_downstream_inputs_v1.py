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
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
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
    if config.get("schema") == "downstream_sta_normalization_v1":
        if (config.get("parent_contract") not in {"downstream_sta_optimization_v1.yaml", "downstream_sta_refed_source_v3.yaml"}
                or config.get("scope") != "retained_public_fixed_validation_development_only"
                or config.get("initialization") != "retained_parent_parameters_no_gradient_updates"
                or config.get("normalization") != "two_training_only_layerwise_passes_dropout_disabled_frozen_parameters"
                or config.get("selection") != "keep_parent_seed17_arm_choice_and_exact_seed_set_no_new_selection"):
            raise ValueError("STA normalization must preserve the parent population, frozen weights and selection")
        parent = read_config(Path(path).parent / config["parent_contract"])
        return {**parent, **config}
    if config.get("schema") == "downstream_sta_optimization_v1":
        if (config.get("source_partitions") != ["train", "validation"]
                or config.get("scope") != "retained_public_fixed_validation_development_only"
                or config.get("tasks") not in [list(TASKS), ["refed_regression"]]
                or config.get("refed_label") != "valid_30second_arousal_mean_le128_low_gt128_high"
                or config.get("initialization") != "random_supervised_STA_only_no_pretrained_model_finetuning"
                or config.get("seeds") != {"screen":17,"confirmation":[42,73]}):
            raise ValueError("STA optimization must preserve the fixed development population and supervised classification contract")
        return config
    if config.get("schema") == "downstream_frozen_classification_v1":
        if (config.get("source_partitions") != ["train", "validation"]
                or config.get("scope") != "retained_public_fixed_validation_development_only"
                or config.get("tasks") != list(TASKS)
                or config.get("methods") != ["cbramod", "sta_net", "efrm", "brainfusion", "mtrenirs"]
                or config.get("encoder_fitting") != "none"
                or config.get("refed_label") != "valid_30second_arousal_mean_le128_low_gt128_high"):
            raise ValueError("frozen classification requires the fixed development and midpoint label contract")
        return config
    if config.get("schema") == "downstream_convergence_probe_v1":
        if (config.get("scope") != "retained_public_train_validation_fixed_objective_replay"
                or config.get("source_partitions") != ["train", "validation"]
                or config.get("methods") != ["biot_lp", "reve_lp", "efrm_lp"]):
            raise ValueError("fixed-objective convergence replay requires the retained public development scope")
        return config
    if config.get("schema") in {"downstream_repair_probe_v1", "downstream_fnirs_probe_v1"}:
        allowed = set(TASKS) - {"refed_regression"} if config["schema"] == "downstream_fnirs_probe_v1" else {"visual", "refed_regression"}
        if (config.get("source_partitions") != ["train", "validation"]
                or not config.get("tasks") or not set(config["tasks"]).issubset(allowed)
                or config.get("scope") != "retained_public_train_validation_development_only"
                or config.get("evaluation") != "three_subject_folds_on_original_development_pool"):
            raise ValueError("repair probe requires the retained development population only")
        if config["schema"] == "downstream_fnirs_probe_v1" and not config.get("fnirs_probe"):
            raise ValueError("fNIRS probe must explicitly select its fNIRS-only implementation")
        return config
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
        self.derivations = profile.get("eeg_derivations")
        if self.derivations is not None:
            if len(self.derivations) != len(profile["channel_names"]) or any(len(pair) != 2 for pair in self.derivations):
                raise ValueError("bipolar derivations must identify exactly two measured electrodes")
            if ["-".join(pair).casefold() for pair in self.derivations] != [name.casefold() for name in profile["channel_names"]]:
                raise ValueError("bipolar names disagree with their physical derivations")
            self.channels = [[lookup[str(name).casefold()] for name in pair] for pair in self.derivations]
        else:
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
        if self.derivations is not None:
            left, right = np.asarray(self.channels).T
            values["eeg"] = values["eeg"][left] - values["eeg"][right]
            masks["eeg"] = masks["eeg"][left] & masks["eeg"][right]
        else:
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


def probe_inputs(config, project, task, *, signals=False):
    """Check retained identities and partitions before opening signal arrays."""
    parent = project / config["parent_run"]
    parent_plan = json.loads((parent / "plan.json").read_text())
    if parent_plan.get("schema") != SCHEMA or parent_plan["scope"] != "public_train_validation_diagnostic_not_final_paper_ranking":
        raise ValueError("repair parent is not the completed public development diagnostic")
    if json.loads((parent / "summary.json").read_text())["status"] != "completed":
        raise ValueError("complete retained parent evidence required")
    root = parent / task / "prepare"
    a = {name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
         for name in ("partition", "sample_id", "subject_id", "record_id")}
    retained = parent_plan["tasks"][task]
    if set(np.unique(a["partition"])) != {"train", "validation"} or len(a["sample_id"]) != sum(retained["counts"].values()):
        raise ValueError("repair arrays leave the original development support")
    for name in config["source_partitions"]:
        if set(a["subject_id"][a["partition"] == name]) != set(retained["subjects"][name]):
            raise ValueError("repair subject identities changed")
    if len(set(a["sample_id"])) != len(a["sample_id"]):
        raise ValueError("duplicate retained event identity")
    for name in ("target", "mask"):
        a[name] = np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    if signals:
        for name in (("fnirs", "fnirs_mask") if signals == "fnirs" else ("eeg", "fnirs", "eeg_mask", "fnirs_mask")):
            a[name] = np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    return a, retained["profile"]


def probe_plan(config, project, run, source, config_name="downstream_repair_probe_v1.yaml"):
    tasks = {}
    for task in config["tasks"]:
        a, profile = probe_inputs(config, project, task)
        subjects = sorted(set(a["subject_id"]))
        shuffled = np.random.default_rng(config["fold_seed"]).permutation(subjects)
        folds = {str(subject): int(index % 3) for index, subject in enumerate(shuffled)}
        tasks[task] = dict(subject_folds=folds, subjects=subjects, samples=len(a["target"]),
                           sample_ids=a["sample_id"].tolist(), profile=profile)
    plan_value = dict(schema=config["schema"], scope=config["scope"], parent_run=config["parent_run"],
                      tasks=tasks, metric_version="v2", test_access="none", config=config)
    for name, value in (("plan.json", plan_value), ("resolved_config.json", config)):
        path = run / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("probe population or contract changed; use a new version")
        write(path, value)
    jobs = []
    def add(identity, task, stage, destination, *, arm=None, fold=None, gpu=False, dependencies=()):
        command = [str(project / ".venv/bin/python"), str(source / "comparative_methods/performance_analysis/audit_downstream_inputs_v1.py"),
                   "--config", str(source / "comparative_methods" / config_name), "--run", str(run),
                   "--project-root", str(project), "--stage", stage, "--device", "{device}"]
        if task:
            command += ["--task", task]
        if arm is not None:command += ["--arm", arm]
        if fold is not None:command += ["--fold", str(fold)]
        threads = config["resources"]["threads_per_job"]
        if stage == "probe-features" and (config.get("native_hb_audit") or config.get("native_eeg_audit")):
            threads *= (config.get("native_hb_audit") or config["native_eeg_audit"])["workers"]
        jobs.append(dict(id=identity, kind="diagnostic", task=task or "aggregate", method=arm or stage,
                         command=command, cwd=str(source), output_dir=str(destination), dependencies=list(dependencies),
                         resources=dict(cpu_threads=threads, ram_gib=16 if stage=="probe-features" and config.get("native_eeg_audit") else 10,
                                        gpu=gpu, gpu_memory_mib=(6500 if config.get("efrm_ft") or config.get("frozen_sequence_encoder") == "efrm" and stage in {"probe-smoke", "probe-fold-encoder"} else 4096 if config.get("encoder_ft") else 2500) if gpu else 0), timeout_seconds=7200,
                         timing_kind=arm or stage))
    fitting = []
    for task in config["tasks"]:
        smoke_job = task + "_smoke"
        add(smoke_job, task, "probe-smoke", run / "smoke" / task, gpu=config.get("smoke_device", "cuda") != "cpu")
        signal_job = task + "_signal_features"
        if config.get("frozen_sequence_encoder"):
            for fold in range(3):
                add(f'{task}_encoder_fold{fold}',task,'probe-fold-encoder',run/task/'encoder'/f'fold_{fold}',fold=fold,gpu=True,dependencies=[smoke_job])
        elif not config.get("retained_feature_run"):
            add(signal_job, task, "probe-features", run / task / "features",
                dependencies=[smoke_job] if (config.get("native_hb_audit") or config.get("native_eeg_audit")) else [])
        uses_encoder = any(spec["features"].startswith("encoder_") for spec in config["models"][task].values())
        if uses_encoder and not config.get("retained_feature_run"):
            add(task + "_encoder_features", task, "probe-encoder", run / task / "encoder", gpu=True, dependencies=[smoke_job])
        for arm, spec in config["models"][task].items():
            for fold in range(3):
                identity = f"{task}_{arm}_fold{fold}"
                dependencies = [smoke_job]
                if config.get("frozen_sequence_encoder"):
                    dependencies.append(f'{task}_encoder_fold{fold}')
                elif spec["features"].startswith("encoder_") and not config.get("retained_feature_run"):
                    dependencies.append(task + "_encoder_features")
                elif not config.get("retained_feature_run") and spec["estimator"] != "cbramod_ft":
                    dependencies.append(signal_job)
                add(identity, task, "probe-fit", run / task / arm / f"fold_{fold}", arm=arm, fold=fold,
                    gpu=spec["estimator"] in {"mlp", "temporal_conv", "cbramod_ft", "efrm_ft", "event_conv"}, dependencies=dependencies)
                fitting.append(identity)
        if config.get("record_controls"):
            identity = task + "_record_control"
            add(identity, task, "probe-record-control", run / task / "record_control", dependencies=[signal_job, smoke_job])
            fitting.append(identity)
    add("aggregate", None, "probe-aggregate", run / "aggregate", dependencies=fitting)
    if config.get("fnirs_probe"):
        jobs[-1].update(kind="aggregate", dependency_policy="terminal")
    if config.get("feature_version", 1) >= 2 or config.get("encoder_ft"):
        add("verification", None, "probe-verify", run / "verification", dependencies=["aggregate"])
    cfg = config["resources"]
    manifest = dict(schema="downstream_rebench_schedule_v1", run_id=config["experiment_id"], run_root=str(run),
                    source_root=str(source), scope=config["scope"], jobs=jobs,
                    limits=dict(cpu_threads=cfg["cpu_thread_budget"], ram_gib=cfg["ram_budget_gib"], gpu_ids=cfg["gpus"],
                                max_gpu_jobs_per_device=cfg["max_gpu_jobs_per_device"],
                                gpu_reserve_mib={str(k): v for k, v in cfg["gpu_reserve_mib"].items() if int(k) in cfg["gpus"]}))
    if (run / "jobs.json").exists() and json.loads((run / "jobs.json").read_text()) != manifest:
        raise ValueError("probe execution matrix changed")
    write(run / "jobs.json", manifest)
    return dict(jobs=len(jobs), fitting_jobs=len(fitting), subjects={k: len(v["subjects"]) for k, v in tasks.items()})


def probe_frontal_projection(eeg, names):
    """Declared within-window scalp-proxy projection, not an EOG measurement."""
    from scipy.signal import butter, sosfiltfilt
    lookup = {str(name).casefold(): i for i, name in enumerate(names)}
    if not {'fp1', 'fp2'}.issubset(lookup) or eeg.ndim != 3 or not np.isfinite(eeg).all():
        raise ValueError("finite complete EEG with physical Fp1/Fp2 required")
    x = np.asarray(eeg, np.float64); delta = x - x.mean(-1, keepdims=True)
    proxy = sosfiltfilt(butter(4, 7., btype='lowpass', fs=200., output='sos'),
                       delta[:, [lookup['fp1'], lookup['fp2']]], axis=-1)
    proxy -= proxy.mean(-1, keepdims=True)
    covariance = proxy @ proxy.swapaxes(-1, -2) / x.shape[-1]
    ridge = 1e-3 * np.maximum(np.trace(covariance, axis1=-2, axis2=-1) / 2, 1e-12)
    cross = delta @ proxy.swapaxes(-1, -2) / x.shape[-1]
    coefficients = np.linalg.solve(covariance + ridge[:, None, None] * np.eye(2), cross.swapaxes(-1, -2)).swapaxes(-1, -2)
    removed = coefficients @ proxy
    ratio = np.sqrt(np.mean(removed ** 2, -1) / np.maximum(np.mean(delta ** 2, -1), 1e-12))
    return (x - removed).astype(np.float32), ratio


def fnirs_probe_extract(config, project, run, task, output, device, *, encoder=False, sample_indices=None):
    """Extract frozen fNIRS features without opening any EEG signal array."""
    from comparative_methods.rebench_baselines_v1 import fnirs_block_covariances
    a, profile = probe_inputs(config, project, task, signals="fnirs")
    output.mkdir(parents=True, exist_ok=True)
    indices = np.arange(len(a["target"])) if sample_indices is None else np.asarray(sample_indices, int)
    stores, masks, descriptors = {}, {}, {}
    metadata = {}; maximum_equivalence_error = None
    if encoder:
        from comparative_methods.MtReNIRS.rebench_v1 import FrozenMtReNIRS, normalize_window
        model = FrozenMtReNIRS(profile, device,
            upstream_root=project / "comparative_methods/MtReNIRS/upstream").eval()
        metadata = model.metadata
    else:
        model = SignalFeatures(task, {**profile, "baseline_modality": "fnirs",
                                      "classification_segments": 1 if task == "dsr" else 6})
        components = [str(g["component"]).lower() for g in profile["fnirs_geometry"]]
    started = time.monotonic()
    batch_size = int(config["fnirs_probe"]["batch_size"])
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        x = torch.as_tensor(np.asarray(a["fnirs"][selected]), device=device if encoder else "cpu")
        mask = torch.as_tensor(np.asarray(a["fnirs_mask"][selected]), dtype=torch.bool, device=x.device)
        if encoder:
            encoded, valid = model.tokens(x, mask)
            ordered = encoded.flatten(1).cpu().numpy()
            mean = encoded.sum(1) / valid.sum(1).clamp_min(1).unsqueeze(-1)
            batch_features = {"encoder_ordered": (ordered, valid.unsqueeze(-1).expand_as(encoded).flatten(1).cpu().numpy()),
                              "encoder_mean": (mean.flatten(1).cpu().numpy(), valid.any(1).unsqueeze(-1).expand_as(mean).flatten(1).cpu().numpy())}
            if start == 0:
                complete = torch.nonzero(mask.flatten(1).all(1), as_tuple=True)[0][:2]
                if len(complete):
                    official = model.official.embed(normalize_window(x[complete], mask[complete]), pool="none", remainder="pad")
                    maximum_equivalence_error = float((official - encoded[complete]).abs().max())
                    if maximum_equivalence_error > 2e-5:
                        raise ValueError("adapter does not reproduce the official complete-input encoder")
        else:
            stats = model(None, x, fnirs_mask=mask).numpy()
            stats_mask = model.feature_mask(None, x, fnirs_mask=mask).numpy()
            covariances, coverage = fnirs_block_covariances(x.numpy(), mask.numpy(), components)
            batch_features = {"fnirs_statistics": (stats, stats_mask),
                              "fnirs_covariance": (covariances, np.ones(covariances.shape, bool)),
                              "fnirs_coverage": (coverage, np.ones(coverage.shape, bool))}
        for name, (values, support) in batch_features.items():
            if not np.isfinite(values[support]).all():
                raise ValueError("nonfinite fNIRS feature on retained support")
            if name not in stores:
                shape = (len(indices), *values.shape[1:])
                stores[name] = np.lib.format.open_memmap(output / (name + ".npy"), mode="w+", dtype=values.dtype, shape=shape)
                masks[name] = np.lib.format.open_memmap(output / (name + "_mask.npy"), mode="w+", dtype=bool, shape=shape)
                descriptors[name] = dict(shape=list(shape), blocks=[int(np.prod(shape[1:]))])
            stores[name][start:start + len(selected)] = values
            masks[name][start:start + len(selected)] = support
        write(output / "status.json", dict(status="running", samples=min(start + batch_size, len(indices)), planned=len(indices)))
    for array in [*stores.values(), *masks.values()]:
        array.flush()
    elapsed = time.monotonic() - started
    value = dict(features=descriptors, metadata=metadata, samples=len(indices), seconds=elapsed,
                 samples_per_second=len(indices) / max(elapsed, 1e-9),
                 official_encoder_max_absolute_difference=maximum_equivalence_error,
                 signal_arrays_read=["fnirs", "fnirs_mask"], input_profile=profile,
                 source_indices=indices.tolist() if sample_indices is not None else "all_retained_development_samples")
    write(output / "features.json", value)
    return value


def probe_extract(config, project, run, task, output, device, *, encoder=False, sample_indices=None):
    if config.get("fnirs_probe"):
        return fnirs_probe_extract(config, project, run, task, output, device, encoder=encoder, sample_indices=sample_indices)
    if config.get("frozen_sequence_encoder"):
        return probe_fold_encoder(config,project,run,task,output,device,fold=0,sample_indices=sample_indices)
    if config.get("native_hb_audit") and not encoder:
        return probe_native_signal_extract(config, project, task, output, sample_indices=sample_indices)
    if config.get("native_eeg_audit") and not encoder:
        return probe_native_signal_extract(config, project, task, output, sample_indices=sample_indices, modality="eeg")
    from comparative_methods.rebench_baselines_v1 import repair_probe_features
    a, native = probe_inputs(config, project, task, signals=True)
    output.mkdir(parents=True, exist_ok=True)
    indices = np.arange(len(a["target"])) if sample_indices is None else np.asarray(sample_indices, dtype=int)
    profile, channels = view_profile(native, task)
    stores = {}; masks = {}; blocks = {}; removed_ratios = []
    model = None
    if encoder:
        from comparative_methods.CBraMod.rebench_v1 import create_model
        profile["method_root"] = str(project / "comparative_methods/CBraMod")
        model = create_model(task, profile, device, trainable=False).eval()
    batch_size = 8 if encoder else 16
    started = time.monotonic()
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices)); selected = indices[start:stop]
        eeg = np.asarray(a["eeg"][selected])[:, channels]
        em = np.asarray(a["eeg_mask"][selected])[:, channels]
        if config.get("event_input_projection"):
            if config["event_input_projection"] != 'frontal_lowfrequency_ols' or task != 'visual' or not em.all():
                raise ValueError("known complete Visual scalp-proxy diagnostic required")
            eeg, removed = probe_frontal_projection(eeg, profile['channel_names'])
            removed_ratios.append(removed)
        if encoder:
            if not em.all():
                raise ValueError("frozen CBraMod probe requires complete EEG")
            x = torch.as_tensor(eeg, device=device, dtype=torch.float32)
            x = x - x.mean(-1, keepdim=True)
            x = x / x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            with torch.inference_mode():
                tokens = model.encoder(x.reshape(len(x), len(channels), 30, 200))
                if config.get("event_convolution"):
                    value = tokens[:, :, 5:17].permute(0, 3, 1, 2).float().cpu().numpy()
                else:
                    event = torch.stack([tokens[:, :, left:right].mean(2) for left, right in ((0, 5), (5, 8), (8, 17), (17, 30))], 2)
                    value = event.flatten(1).float().cpu().numpy()
            name = "encoder_sequence" if config.get("event_convolution") else "encoder_event"
            features = {name: (value, np.ones_like(value, bool), [value.shape[-1]])}
        elif config.get("event_convolution"):
            if task != "visual" or not em.all():
                raise ValueError("event convolution requires complete Visual EEG")
            x = eeg.astype(np.float32) - eeg.mean(-1, keepdims=True)
            x /= np.maximum(np.sqrt(np.square(x).mean(-1, keepdims=True)), 1e-6)
            value = x[:, None, :, 1000:3400]
            features = {"raw_event": (value, np.ones_like(value, bool), [value.shape[-1]])}
        else:
            features = repair_probe_features(eeg, np.asarray(a["fnirs"][selected]), em,
                                             np.asarray(a["fnirs_mask"][selected]), task=task,
                                             version=config.get("feature_version", 1))
        for name, (value, mask, lengths) in features.items():
            if name not in stores:
                shape = (len(indices), *value.shape[1:])
                stores[name] = np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=np.float32, shape=shape)
                masks[name] = np.lib.format.open_memmap(output / f"{name}_mask.npy", mode="w+", dtype=np.bool_, shape=shape)
                blocks[name] = lengths
            stores[name][start:stop], masks[name][start:stop] = value, mask
        write(output / "status.json", dict(status="running", prepared=stop, planned=len(indices)))
    for array in [*stores.values(), *masks.values()]:
        array.flush()
    descriptor = dict(features={k: dict(shape=list(v.shape), blocks=blocks[k]) for k, v in stores.items()},
                      samples=len(indices), source_indices=indices, seconds=time.monotonic() - started,
                      input_profile=profile, fitting="none; fixed transforms of recorded inputs only",
                      checkpoint_metadata=getattr(model, "metadata", None))
    if removed_ratios:
        ratios = np.concatenate(removed_ratios)
        np.save(output / 'projection_removed_rms_ratio.npy', ratios)
        descriptor['input_projection'] = dict(method=config['event_input_projection'], proxy_channels=['Fp1', 'Fp2'],
            proxy_lowpass_hz=7., ridge_trace_fraction=.001, fitting='each_same30second_window_without_labels',
            removed_rms_over_centered_input_rms_quantiles=np.quantile(ratios, [0, .5, .9, .99, 1]),
            per_window_channel_ratios='projection_removed_rms_ratio.npy',
            interpretation='scalp_proxies_may_include_neural_signal;not_measured_EOG_or_verified_clean_neural_truth')
    write(output / "features.json", descriptor)
    return descriptor


def probe_fold_biot(config, project, run, task, output, device, *, fold, sample_indices=None):
    from comparative_methods.BIOT.rebench_v1 import create_model, prest_bipolar_profile
    if task != 'visual' or fold not in range(3):
        raise ValueError('BIOT montage probe requires the admitted Visual panel and fold')
    a, native = probe_inputs(config, project, task, signals=True)
    retained = json.loads((run / 'plan.json').read_text())['tasks'][task]
    if a['sample_id'].tolist() != retained['sample_ids']:
        raise ValueError('BIOT source population changed')
    indices = np.arange(len(a['target'])) if sample_indices is None else np.asarray(sample_indices, int)
    training = (np.flatnonzero([retained['subject_folds'][str(s)] != fold for s in a['subject_id']])
                if sample_indices is None else indices[:12])
    profile, _ = view_profile(native, task)
    profile['method_root'] = str(project / 'comparative_methods/BIOT')
    views = {'biot_native': profile, 'biot_prest8': prest_bipolar_profile(profile)}
    output.mkdir(parents=True, exist_ok=True); started = time.monotonic(); descriptors = {}; metadata = {}
    for name, view in views.items():
        scaler = fit_input_scaler(ArrayDataset(a, native, view, training))
        write(output / (name + '_input_scaler.json'), scaler)
        model = create_model(task, view, device, trainable=False).eval()
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('BIOT linear probe encoder must remain frozen')
        values = np.lib.format.open_memmap(output / (name + '.npy'), mode='w+', dtype=np.float32, shape=(len(indices), 256))
        mask = np.lib.format.open_memmap(output / (name + '_mask.npy'), mode='w+', dtype=np.bool_, shape=values.shape)
        offset = 0
        with torch.inference_mode():
            for rows, batch in batches(ArrayDataset(a, native, view, indices), scaler, device, batch_size=8):
                result = model(batch['eeg'], eeg_mask=batch['channel_valid_mask']['eeg']).float().cpu().numpy()
                if result.shape != (len(rows), 256) or not np.isfinite(result).all():
                    raise ValueError('invalid BIOT montage feature output')
                values[offset:offset + len(rows)] = result; mask[offset:offset + len(rows)] = True; offset += len(rows)
                if offset % 128 == 0 or offset == len(indices):
                    write(output / 'status.json', dict(status='running', input_view=name, features=offset, planned=len(indices), seconds=time.monotonic()-started))
        values.flush(); mask.flush()
        descriptors[name] = dict(shape=list(values.shape), blocks=[256])
        metadata[name] = dict(input_profile=view, checkpoint=model.metadata, input_scaler=name+'_input_scaler.json')
        del model
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
    descriptor = dict(features=descriptors, samples=len(indices), source_indices=indices,
        training_subjects=sorted(set(a['subject_id'][training])), seconds=time.monotonic()-started,
        input_views=metadata, fitting='physical_bipolar_difference_before_current_training_people_input_scaling',
        interpretation='same_matched16_budget;exact_PREST8of16_subset_compared_with_native_positional_transfer')
    write(output / 'features.json', descriptor)
    return descriptor


def probe_fold_encoder(config,project,run,task,output,device,*,fold,sample_indices=None):
    if config.get('frozen_sequence_encoder') == 'biot':
        return probe_fold_biot(config, project, run, task, output, device, fold=fold, sample_indices=sample_indices)
    if task!='refed_regression' or config.get('frozen_sequence_encoder')!='efrm' or fold not in range(3):
        raise ValueError('known REFED frozen temporal encoder required')
    a,native=probe_inputs(config,project,task,signals=True)
    retained=json.loads((run/'plan.json').read_text())['tasks'][task]
    if a['sample_id'].tolist()!=retained['sample_ids']:raise ValueError('encoder source population changed')
    indices=np.arange(len(a['target'])) if sample_indices is None else np.asarray(sample_indices,int)
    training=np.flatnonzero(np.array([retained['subject_folds'][str(s)]!=fold for s in a['subject_id']])) if sample_indices is None else indices[:12]
    profile,_=view_profile(native,task);profile['pretrained_checkpoint']=efrm_checkpoint(config,project,task)
    scaler=fit_input_scaler(ArrayDataset(a,native,profile,training))
    output.mkdir(parents=True,exist_ok=True);write(output/'input_scaler.json',scaler)
    module=importlib.import_module('comparative_methods.EFRM-PyTorch.rebench_v1')
    model=module.create_model(task,profile,device,trainable=False).eval()
    if any(p.requires_grad for p in model.parameters()):raise ValueError('frozen EFRM feature extraction became trainable')
    values=np.lib.format.open_memmap(output/'efrm_sequence.npy',mode='w+',dtype=np.float32,shape=(len(indices),30,model.feature_dim))
    masks=np.lib.format.open_memmap(output/'efrm_sequence_mask.npy',mode='w+',dtype=np.bool_,shape=values.shape)
    started=time.monotonic();offset=0
    with torch.inference_mode():
        for rows,batch in batches(ArrayDataset(a,native,profile,indices),scaler,device,batch_size=1):
            with torch.autocast(device_type='cuda' if str(device).startswith('cuda') else 'cpu',dtype=torch.bfloat16,enabled=str(device).startswith('cuda') and config.get('encoder_precision','fp32')=='bf16'):
                result=model(batch['eeg'],batch['fnirs'],eeg_mask=batch['channel_valid_mask']['eeg'],fnirs_mask=batch['channel_valid_mask']['fnirs'])
                supported=model.sequence_feature_mask(batch['eeg'],batch['fnirs'],eeg_mask=batch['channel_valid_mask']['eeg'],fnirs_mask=batch['channel_valid_mask']['fnirs'])
            value=result.float().cpu().numpy();support=supported.cpu().numpy()
            if value.shape!=(len(rows),30,model.feature_dim) or support.shape!=(len(rows),30) or not np.isfinite(value).all():raise ValueError('invalid EFRM physical-time feature output')
            values[offset:offset+len(rows)]=value;masks[offset:offset+len(rows)]=support[...,None];offset+=len(rows)
            if offset%32==0 or offset==len(indices):write(output/'status.json',dict(status='running',features=offset,planned=len(indices),seconds=time.monotonic()-started))
    values.flush();masks.flush()
    descriptor=dict(features=dict(efrm_sequence=dict(shape=list(values.shape),blocks=[model.feature_dim])),samples=len(indices),source_indices=indices,
        seconds=time.monotonic()-started,training_subjects=sorted(set(a['subject_id'][training])),input_profile=profile,
        fitting='frozen_target_excluded_encoder;input_scale_fitted_only_current_training_people',pretraining_provenance=model.pretraining_provenance,
        interpretation='retained_measurement_v5_target_excluded_checkpoint_transfer_not_new_input_pretraining')
    write(output/'features.json',descriptor);return descriptor


def probe_native_hb_record(arguments):
    """One registered native record; called only after the population check."""
    from scipy.signal import butter, sosfiltfilt
    from src.data.unified_physiology import load_native_fnirs_record, _resample, _slice_window, _interpolate_nonfinite
    from src.data.homer2_preprocessing import bandpass_fnirs
    project, record, requests = arguments
    started = time.monotonic(); native = load_native_fnirs_record(project, record)
    values = native["values"].reshape(len(native["values"]), -1)
    values, repaired = _interpolate_nonfinite(values)
    rate = native["sample_rate_hz"]
    clock = "uniform_native_rate"
    if record.dataset_id == "visual_cognitive_motivation":
        # The cache producer resamples the actual ETG Time column BEFORE its
        # temporal filter. Row number / 10 is not the published sample clock.
        times = np.asarray(native["time_s"], float)
        if times.shape != (len(values),) or np.any(np.diff(times) <= 0) or abs(times[0]) > 1e-9:
            raise ValueError("invalid native ETG clock")
        grid = np.arange(int(np.floor(times[-1] * rate)) + 1) / rate
        values = np.column_stack([np.interp(grid, times, channel) for channel in values.T])
        clock = "native_Time_column_resampled_before_filtering_as_owning_cache_producer"
    views = {}; quality = {}
    for name, high in (("replay", .2), ("bandpass", .7)):
        transformed, state = bandpass_fnirs(values, sample_rate_hz=rate, low_hz=.01, high_hz=high, order=3)
        if state["status"] != "applied":
            raise ValueError("native audit filter did not apply")
        views[name] = _resample(transformed, rate, 10.)
        quality[name] = state
    views["lowpass"] = _resample(sosfiltfilt(butter(3, .7, btype="lowpass", fs=rate, output="sos"), values, axis=0), rate, 10.)
    output = []
    for index, start, positions in requests:
        windows = {}; mask = None
        for name, value in views.items():
            windows[name], actual = _slice_window(value, start * 1000., 30., 10.)
            if mask is not None and not np.array_equal(actual, mask):
                raise ValueError("native views changed the sample clock")
            mask = actual
        output.append((index, positions, windows, mask))
    return output, dict(join_key=record.join_key, seconds=time.monotonic() - started,
        source=native["provenance"], native_shape=list(native["values"].shape), native_rate_hz=rate,
        source_nonfinite_interpolated=repaired, filters=quality, lowpass_hz=.7, lowpass_order=3,
        native_clock_processing=clock,
        resampling="existing_unified_physiology_polyphase_to_10Hz", windows=len(requests))


def probe_native_eeg_record(arguments):
    from dataclasses import replace
    from src.data.unified_physiology import load_native_eeg_record, preprocess_eeg_record_with_quality, _interpolate_nonfinite, _bandpass, _resample, _slice_window
    project, record, requests, names = arguments
    started = time.monotonic();native = load_native_eeg_record(project,record)
    lookup={name.casefold():i for i,name in enumerate(native.channel_names)}
    columns=[lookup[name.casefold()] for name in names]
    native=replace(native,values=native.values[:,columns],channel_names=tuple(names),
        channel_units=tuple(native.channel_units[i] for i in columns) if native.channel_units else ())
    baseline,state,_=preprocess_eeg_record_with_quality(native,signal_branch="raw_with_ocular_artifact",output_coordinate="measurement")
    factors=np.array([item['factor'] for item in state['unit_conversions']])
    finite,repaired=_interpolate_nonfinite(native.values*factors)
    views={'replay':baseline}
    for name,low in [('hp_0p1',.1),('hp_0p01',.01)]:
        views[name]=_resample(_bandpass(finite,native.sample_rate_hz,(low,45.)),native.sample_rate_hz,200.)
    windows=[]
    for index,start,positions in requests:
        values={};support=None
        for name,view in views.items():
            values[name],actual=_slice_window(view,start*1000.,30.,200.)
            if support is not None and not np.array_equal(actual,support):raise ValueError('EEG high-pass views changed support')
            support=actual
        windows.append((index,positions,values,support))
    return windows,dict(join_key=record.join_key,seconds=time.monotonic()-started,source=str(native.source_path),
        source_unit=native.native_unit,output_unit=state['canonical_unit'],channels=names,
        source_nonfinite_interpolated=repaired,baseline_processing=state,
        bands_hz={'replay':[1.,45.],'hp_0p1':[.1,45.],'hp_0p01':[.01,45.]},
        information_support='offline_full_record_filter_before_same30s_window;ocular_artifacts_retained',windows=len(requests))


def probe_native_signal_extract(config, project, task, output, *, sample_indices=None, modality="fnirs"):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    import re
    from comparative_methods.rebench_baselines_v1 import repair_hb_probe_features,repair_voltage_probe_features
    # This first check precedes construction of a native-reader request.
    a, profile = probe_inputs(config, project, task, signals=True)
    spec = config["native_eeg_audit" if modality=="eeg" else "native_hb_audit"]
    expected_scope="same_retained_development_subjects_and_windows_native_EEG_only" if modality=="eeg" else "same_retained_development_subjects_and_windows_native_Hb_only"
    expected_views=["cached","hp_0p1","hp_0p01"] if modality=="eeg" else ["cached","bandpass","lowpass"]
    if spec["scope"] != expected_scope or spec["views"] != expected_views:
        raise ValueError("unknown native signal audit scope")
    dataset = FrozenInputBenchmarkDataset(project / spec["input_cache"], task=task,
        splits=config["source_partitions"], eeg_panel="native")
    identities = [row["dependency_id"] + "|offset=-5|duration=30|joined_probes" if task == "visual" else row["id"] for row in dataset.rows]
    if identities != a["sample_id"].tolist() or dataset.profile["fnirs_channel_names"] != profile["fnirs_channel_names"]:
        raise ValueError("native audit must preserve the exact retained sample/channel population")
    indices = np.arange(len(identities)) if sample_indices is None else np.asarray(sample_indices, int)
    output.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    if modality=="eeg":
        profile,eeg_columns=view_profile(profile,task);positions={name:i for i,name in enumerate(profile['channel_names'])}
    else:
        positions = {name: i for i, name in enumerate(profile["fnirs_channel_names"])}
    work = {}
    for index, source_index in enumerate(indices):
        for member in (dataset.groups[source_index][:1] if modality=="eeg" else dataset.groups[source_index]):
            row = dataset.reader.samples[member]; record = dataset.reader.records[row["join_key"]]
            if (row["split"] not in config["source_partitions"] or row["subject"] != a["subject_id"][source_index]
                    or record.canonical_subject_id != row["subject"]):
                raise ValueError("native request leaves retained development subjects")
            if modality=="eeg":
                key=(str(a['subject_id'][source_index]),str(a['record_id'][source_index]));columns=list(range(len(positions)))
                if key not in work:work[key]=(project,record,[],profile['channel_names'])
                work[key][2].append((index,row['eeg_start_s'],columns))
            else:
                probe = re.search(r"_Probe([12])$", record.base_record_id) if task == "visual" else None
                prefix = f"Probe{probe.group(1)}:" if probe else ""
                columns = [positions[prefix + name] for name in record.manifest["measurement"]["fnirs_channel_names"]]
                if record.join_key not in work:work[record.join_key] = (project, record, [])
                work[record.join_key][2].append((index, row["hb_start_s"], columns))
    mask = np.asarray(a[modality+"_mask"][indices], bool)
    cached = np.asarray(a[modality][indices],np.float32 if modality=="eeg" else float)
    if modality=="eeg":mask,cached=mask[:,eeg_columns],cached[:,eeg_columns]
    views = {name: np.zeros(cached.shape,cached.dtype) for name in ["replay",*expected_views[1:]]}
    filled = np.zeros(mask.shape, bool); records = []
    workers = 1 if sample_indices is not None else int(spec["workers"])
    jobs = [work[key] for key in sorted(work)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        # Bound queued source records and returned arrays to two per worker.
        for offset in range(0, len(jobs), 2 * workers):
            for windows, provenance in pool.map(probe_native_eeg_record if modality=="eeg" else probe_native_hb_record, jobs[offset:offset + 2 * workers]):
                records.append(provenance)
                for index, columns, values, support in windows:
                    if np.any(mask[index, columns] & ~support[None, :]):
                        raise ValueError("native source is shorter than retained valid clock support")
                    for name in views:
                        views[name][index, columns] = values[name]
                    filled[index, columns] = support[None, :]
                write(output / "status.json", dict(status="running", records_completed=len(records), records_planned=len(jobs)))
    if np.any(mask & ~filled):
        raise ValueError("missing native input on the retained valid support")
    error = (views.pop("replay") - cached)[mask]
    scale = float(np.sqrt(np.mean(cached[mask] ** 2)))
    replay = dict(max_abs=float(np.abs(error).max()), rms=float(np.sqrt(np.mean(error ** 2))),
                  signal_rms=scale, relative_rms=float(np.sqrt(np.mean(error ** 2)) / max(scale, 1e-30)))
    write(output / "native_replay.json", dict(replay=replay, records=records, population="original_train_validation_only"))
    if replay["relative_rms"] > spec["replay_relative_rms_tolerance"]:
        raise ValueError("native original-band replay disagrees with frozen signal; inspect before model fitting")
    views["cached"] = cached
    features = {}
    for name, value in views.items():
        x, m, blocks = (repair_voltage_probe_features if modality=="eeg" else repair_hb_probe_features)(value, mask, task=task)
        key = ("voltage_" if modality=="eeg" else "native_") + name
        np.save(output / (key + ".npy"), x); np.save(output / (key + "_mask.npy"), m)
        features[key] = dict(shape=list(x.shape), blocks=blocks)
    descriptor = dict(features=features, samples=len(indices), source_indices=indices,
        seconds=time.monotonic() - started, input_profile=profile, fitting="none; registered native preprocessing audit",
        native_source_records=len(records), native_workers=workers, replay=replay,
        information_support="same30swindows_and_original_full_record_offline_filter_support;no_rest_baseline_or_extra_targets")
    write(output / "features.json", descriptor)
    return descriptor


def probe_smoke(config, project, run, task, output, device):
    if config.get("fnirs_probe"):
        a, _ = probe_inputs(config, project, task)
        indices = np.linspace(0, len(a["target"]) - 1, min(32, len(a["target"])), dtype=int)
        results = {}
        for encoder, name in ((False, "features"), (True, "encoder")):
            results[name] = fnirs_probe_extract(config, project, run, task, output / name, device,
                                                 encoder=encoder, sample_indices=indices)
        return dict(scope="fNIRS_encoder_and_feature_throughput_only_not_evaluation", checks=results)
    a, _ = probe_inputs(config, project, task)
    if config.get("record_objective_probe"):
        keys = list(zip(a["subject_id"].tolist(), a["record_id"].tolist()))
        selected = sorted(set(keys))[:4]
        if len(selected) != 4:
            raise ValueError("record-objective smoke requires four complete retained records")
        indices = np.concatenate([np.flatnonzero([key == chosen for key in keys]) for chosen in selected])
        small = {key: value[indices] for key, value in a.items()}
        train = np.flatnonzero([keys[i] in selected[:3] for i in indices])
        spec = dict(max((row for row in config["models"][task].values() if row["features"] in {"eeg", "fnirs", "multimodal"}),
                        key=lambda row: row["record_ccc_weight"]), epochs=2)
        parent = project / spec["feature_source_run"]
        source_plan = json.loads((parent / "plan.json").read_text())
        if source_plan["parent_run"] != config["parent_run"] or source_plan["tasks"][task]["sample_ids"] != a["sample_id"].tolist():
            raise ValueError("record-objective smoke feature population changed")
        directory = parent / task / "features"
        name = spec["features"]
        descriptor = json.loads((directory / "features.json").read_text())["features"][name]
        values = np.load(directory / (name + ".npy"), mmap_mode="r")[indices]
        mask = np.load(directory / (name + "_mask.npy"), mmap_mode="r")[indices]
        x, _ = probe_transform(values, mask, train, descriptor["blocks"])
        prediction, fitted = probe_neural(x, small, train, probe_weights(small, train, task), task, spec, config, device, output)
        if prediction.shape != small["target"].shape or not np.isfinite(prediction).all():
            raise ValueError("record-objective smoke output is invalid")
        return dict(scope="complete_record_objective_gradient_and_throughput_not_evaluation",
                    source_indices=indices, training=fitted, specification=spec)
    if config.get("efrm_ft"):
        a, native = probe_inputs(config, project, task, signals=True)
        plan = json.loads((run / "plan.json").read_text())["tasks"][task]
        spec = dict(max(config["models"][task].values(), key=lambda row: row["encoder_learning_rate"]), epochs=1, effective_batch_size=4)
        training = np.flatnonzero([plan["subject_folds"][str(s)] != 0 for s in a["subject_id"]])[:12]
        evaluation = np.flatnonzero([plan["subject_folds"][str(s)] == 0 for s in a["subject_id"]])[:4]
        if 'record_ccc_weight' in spec:
            keys = list(zip(a['subject_id'].tolist(), a['record_id'].tolist()))
            groups = {key: np.flatnonzero([value == key for value in keys]) for key in sorted(set(keys))}
            training_keys = sorted((key for key in groups if plan['subject_folds'][key[0]] != 0), key=lambda key: (-len(groups[key]), key))[:3]
            evaluation_key = next(key for key in groups if plan['subject_folds'][key[0]] == 0)
            training = np.concatenate([groups[key] for key in training_keys]); evaluation = groups[evaluation_key]
            spec['effective_record_batch_size'] = 1
        indices = np.concatenate((training, evaluation))
        prediction, fitted = probe_efrm_ft(a, native, training, probe_weights(a, training, task), spec,
            config, project, device, output, fold=0, evaluation_indices=indices)
        if prediction.shape != (len(indices), 2, 30) or not np.isfinite(prediction).all():
            raise ValueError("EFRM gradient smoke has invalid output")
        return dict(scope="encoder_gradient_and_throughput_only_not_evaluation", source_indices=indices, specification=spec, training=fitted)
    if config.get("event_convolution"):
        indices = np.linspace(0, len(a["target"]) - 1, 16, dtype=int)
        small = {key: value[indices] for key, value in a.items()}
        cfg = deepcopy(config); cfg["fit"]["neural_epochs"] = 2; checks = {}
        for encoder, name in ((False, "raw_event"), (True, "encoder_sequence")):
            destination = output / name; destination.mkdir(parents=True, exist_ok=True)
            descriptor = probe_extract(config, project, run, task, destination, device, encoder=encoder, sample_indices=indices)
            x, _ = probe_event_transform(np.load(destination / (name + ".npy")), np.load(destination / (name + "_mask.npy")), np.arange(12))
            prediction, fitted = probe_neural(x, small, np.arange(12), np.ones(12), task,
                dict(estimator="event_conv", auxiliary_weight=.5), cfg, device, destination)
            if prediction.shape != small["target"].shape or not np.isfinite(prediction).all():
                raise ValueError("event convolution smoke has invalid output")
            checks[name] = dict(features=descriptor["features"], seconds=descriptor["seconds"], training=fitted)
        return dict(scope="throughput_and_gradient_only_not_evaluation", source_indices=indices, checks=checks)
    indices = np.arange(16) if (config.get("native_hb_audit") or config.get("native_eeg_audit")) else np.linspace(0, len(a["target"]) - 1, 16, dtype=int)
    ft = [spec for spec in config["models"][task].values() if spec["estimator"] == "cbramod_ft"]
    if ft:
        a, native = probe_inputs(config, project, task, signals=True)
        small = {key: value[indices] for key, value in a.items()}
        prediction, training = probe_cbramod_ft(small, native, np.arange(12), np.ones(12),
            dict(ft[0], epochs=config.get("encoder_smoke_epochs", 2)), config, project, device, output)
        train_accuracy = float(np.mean(prediction[:12] == small["target"][:12]))
        write(output / "optimization_control.json", dict(train_accuracy=train_accuracy,
            training_predictions=prediction[:12], training_targets=small["target"][:12], training=training,
            scope="training_only_small_batch_optimization_control_not_validation"))
        if config.get("encoder_smoke_min_train_accuracy") is not None and train_accuracy < config["encoder_smoke_min_train_accuracy"]:
            raise ValueError("encoder small-batch optimization control did not learn its training labels")
        return dict(scope="throughput_and_encoder_gradient_only_not_evaluation", source_indices=indices,
                    prediction_shape=list(prediction.shape), train_accuracy=train_accuracy, training=training)
    features = output / "features"
    encoder = task == "visual" and any(spec["features"].startswith("encoder_") for spec in config["models"][task].values())
    descriptor = probe_extract(config, project, run, task, features, device, encoder=encoder, sample_indices=indices)
    name = "biot_prest8" if config.get("frozen_sequence_encoder") == "biot" else "efrm_sequence" if config.get("frozen_sequence_encoder") else "voltage_cached" if config.get("native_eeg_audit") else "native_cached" if config.get("native_hb_audit") else "encoder_event" if encoder else "event" if task == "visual" else "multimodal"
    x, _ = probe_transform(np.load(features / (name + ".npy")), np.load(features / (name + "_mask.npy")),
                           np.arange(12), descriptor["features"][name]["blocks"])
    small = {key: value[indices] for key, value in a.items()}
    cfg = deepcopy(config); cfg["fit"]["neural_epochs"] = 2
    weights = np.ones(12) if task == "visual" else small["mask"][:12].transpose(0, 2, 1).astype(float)
    prediction, training = probe_neural(x, small, np.arange(12), weights, task, dict(auxiliary_weight=.5, dropout=.1), cfg, device, output)
    if prediction.shape != small["target"].shape or not np.isfinite(prediction).all():
        raise ValueError("smoke prediction shape or numerical support failed")
    return dict(scope="throughput_and_gradient_only_not_evaluation", source_indices=indices,
                prediction_shape=list(prediction.shape), feature_seconds=descriptor["seconds"], training=training)


def probe_transform(values, support, train, blocks):
    """Masked training-only imputation and scale; equal feature-block norm."""
    values, support = np.asarray(values, float), np.asarray(support, bool)
    x, mask = values[train].reshape(-1, values.shape[-1]), support[train].reshape(-1, values.shape[-1])
    counts = mask.sum(0); mean = np.where(mask, x, 0.).sum(0) / np.maximum(counts, 1)
    scale = np.sqrt(np.square(np.where(mask, x - mean, 0.)).sum(0) / np.maximum(counts, 1))
    scale[scale == 0] = 1.
    normalizer = np.concatenate([np.full(size, np.sqrt(size)) for size in blocks])
    transformed = np.where(support & (counts > 0), (values - mean) / scale / normalizer, 0.)
    if not np.isfinite(transformed).all():
        raise ValueError("nonfinite fold-transformed feature")
    return transformed, dict(mean=mean, scale=scale, counts=counts, block_normalizer=normalizer)


def probe_subject_calibration(values, support, subjects):
    """Declared transductive control: each person's own unlabelled input only."""
    values, support, subjects = np.asarray(values, float), np.asarray(support, bool), np.asarray(subjects)
    transformed = np.zeros(values.shape, float); names = sorted(set(subjects)); means = []; scales = []; counts = []
    for subject in names:
        rows = subjects == subject
        x, mask = values[rows].reshape(-1, values.shape[-1]), support[rows].reshape(-1, values.shape[-1])
        count = mask.sum(0); mean = np.where(mask, x, 0.).sum(0) / np.maximum(count, 1)
        scale = np.sqrt(np.where(mask, (x - mean) ** 2, 0.).sum(0) / np.maximum(count, 1))
        scale[scale <= 1e-8] = 1.
        transformed[rows] = np.where(support[rows], (values[rows] - mean) / scale, 0.)
        means.append(mean); scales.append(scale); counts.append(count)
    return transformed, dict(subjects=np.array(names), mean=np.array(means), scale=np.array(scales), count=np.array(counts))


def probe_weights(a, train, task, *, class_balance="inverse_prior"):
    subjects, records = a["subject_id"][train], a["record_id"][train]
    if class_balance not in {"inverse_prior", "none"} or task == "refed_regression" and class_balance != "inverse_prior":
        raise ValueError("class weighting contrast is defined only for Visual")
    if task != "refed_regression":
        weights = np.asarray([1 / np.sum(subjects == s) for s in subjects], float)
        labels = a["target"][train]
        frequencies = np.bincount(labels, weights=weights, minlength=len(TASKS[task]["classes"]))
        if (frequencies == 0).any():
            raise ValueError("all declared classes must appear in training")
        if class_balance == "inverse_prior":
            weights = weights / frequencies[labels]
    else:
        mask = a["mask"][train].transpose(0, 2, 1)
        weights = np.zeros(mask.shape, float)
        for subject in np.unique(subjects):
            subject_rows = subjects == subject
            record_ids = np.unique(records[subject_rows])
            for record in record_ids:
                selected = subject_rows & (records == record)
                for c in range(2):
                    count = mask[selected, :, c].sum()
                    if count == 0:
                        raise ValueError("training record has no supported regression coordinate")
                    weights[selected, :, c] = mask[selected, :, c] / (len(record_ids) * count)
    return weights * np.count_nonzero(weights) / weights.sum()


def probe_event_transform(values, support, train):
    """Scale each raw/latent component using only current training examples."""
    if values.ndim != 4 or support.shape != values.shape or not support.all():
        raise ValueError("event convolution needs complete component/electrode/time support")
    selected = np.asarray(values[train], np.float64)
    mean = selected.mean((0, 2, 3)); scale = selected.std((0, 2, 3)); scale[scale == 0] = 1.
    transformed = (np.asarray(values, np.float32) - mean.astype(np.float32)[None, :, None, None]) / scale.astype(np.float32)[None, :, None, None]
    if not np.isfinite(transformed).all():
        raise ValueError("nonfinite event convolution input")
    return transformed, dict(mean=mean, scale=scale, fitting="current_training_people_only_component_scale")


def probe_event_head(components, electrodes, dropout):
    """Small shared temporal/spatial filters; retain six ordered event bins."""
    from torch import nn
    raw = components == 1
    class EventHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.filters = nn.Sequential(
                nn.Conv2d(components, 8, (1, 65 if raw else 1), stride=(1, 2 if raw else 1), padding=(0, 32 if raw else 0), bias=False),
                nn.GroupNorm(4, 8), nn.Conv2d(8, 16, (electrodes, 1), groups=8, bias=False), nn.GroupNorm(4, 16), nn.ELU(),
                nn.AvgPool2d((1, 4 if raw else 1)), nn.Dropout(dropout),
                nn.Conv2d(16, 16, (1, 17 if raw else 3), padding=(0, 8 if raw else 1), groups=16, bias=False),
                nn.Conv2d(16, 16, 1, bias=False), nn.GroupNorm(4, 16), nn.ELU(),
                nn.AvgPool2d((1, 5 if raw else 1)), nn.Dropout(dropout))
            self.mean = nn.AdaptiveAvgPool2d((1, 6)); self.maximum = nn.AdaptiveMaxPool2d((1, 6))
            self.output = nn.Linear(192, 8)

        def forward(self, value):
            value = self.filters(value)
            return self.output(torch.cat((self.mean(value), self.maximum(value)), -1).flatten(1))
    return EventHead()


def probe_temporal_head(feature_dim, width, dropout):
    """The same physical-time readout for frozen features and encoder FT."""
    from torch import nn
    class TemporalHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, width), nn.GELU())
            self.blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(width, width, 3, padding=d, dilation=d),
                nn.GELU(), nn.Dropout(dropout)) for d in (1, 2, 4)])
            self.output = nn.Linear(width, 2)

        def forward(self, value):
            value = self.input(value).transpose(1, 2)
            for block in self.blocks:
                value = value + block(value)
            return self.output(value.transpose(1, 2)).sigmoid()
    return TemporalHead()


def probe_record_ccc_loss(prediction, target, valid, groups, coordinate_weights):
    """CCC within each complete subject/video record, never across videos."""
    terms = []; masses = []
    for group, weights in zip(groups, coordinate_weights):
        for coordinate in range(2):
            if float(weights[coordinate]) <= 0:
                continue
            keep = valid[group, :, coordinate]
            observed = target[group, :, coordinate][keep]
            estimated = prediction[group, :, coordinate][keep]
            if not len(observed) or bool(observed.max() == observed.min()):
                raise ValueError("dynamic CCC weight requires a nonconstant supported record")
            centered_y = observed - observed.mean(); centered_p = estimated - estimated.mean()
            denominator = centered_y.square().mean() + centered_p.square().mean() + (observed.mean() - estimated.mean()).square()
            terms.append(1. - 2. * (centered_y * centered_p).mean() / denominator)
            masses.append(float(weights[coordinate]))
    if not terms:
        return prediction.sum() * 0.
    weight = prediction.new_tensor(masses)
    return (torch.stack(terms) * weight).sum() / weight.sum()


def probe_training_records(a, train):
    """Complete training records and subject-equal dynamic-coordinate weights."""
    keys = list(zip(a["subject_id"][train].tolist(), a["record_id"][train].tolist()))
    unique = sorted(set(keys)); groups = [np.array([i for i, key in enumerate(keys) if key == group]) for group in unique]
    for key, rows in zip(unique, groups):
        if len(rows) != np.sum((a["subject_id"] == key[0]) & (a["record_id"] == key[1])):
            raise ValueError("record CCC requires complete training records")
    dynamic = np.array([[np.ptp(np.asarray(a["target"][train][rows, c])[np.asarray(a["mask"][train][rows, c])]) > 0
                         for c in range(2)] for rows in groups])
    weights = dynamic.astype(float)
    for subject in sorted(set(key[0] for key in unique)):
        rows = np.array([key[0] == subject for key in unique])
        weights[rows] /= np.maximum(dynamic[rows].sum(0), 1)
    return groups, weights, (~dynamic).sum(0)


def probe_neural(x, a, train, weights, task, spec, config, device, output):
    """Fixed-epoch nonlinear/temporal readout; held-fold labels are never read."""
    from torch import nn
    cfg = config["fit"]; torch.manual_seed(config["model_seed"])
    width = int(cfg["neural_width"])
    if task == "visual":
        if spec.get("estimator") == "event_conv":
            model = probe_event_head(x.shape[1], x.shape[2], cfg["neural_dropout"]).to(device)
        else:
            model = nn.Sequential(nn.LayerNorm(x.shape[-1]), nn.Linear(x.shape[-1], width), nn.GELU(),
                                  nn.Dropout(cfg["neural_dropout"]), nn.Linear(width, 8)).to(device)
    else:
        model = probe_temporal_head(x.shape[-1], width, float(spec.get("dropout", .1))).to(device)
    values = torch.as_tensor(np.asarray(x[train], np.float32), device=device)
    target = torch.as_tensor(np.asarray(a["target"][train]), device=device)
    w = torch.as_tensor(weights, device=device, dtype=torch.float32)
    if task != "visual":
        target = (target.transpose(1, 2).float() - 1.) / 254.
        valid = torch.as_tensor(np.asarray(a["mask"][train]).transpose(0, 2, 1), device=device)
        target = torch.where(valid, target, 0.)
    attributes = torch.tensor([[1, 1], [1, 0], [0, 0], [0, 1]], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["neural_learning_rate"], weight_decay=cfg["neural_weight_decay"])
    rng = np.random.default_rng(config["model_seed"]); history = []
    epochs = int(spec.get("epochs", cfg["neural_epochs"]))
    record_groups = None; objective = {}
    if "record_ccc_weight" in spec:
        if task != "refed_regression" or float(spec["record_ccc_weight"]) < 0:
            raise ValueError("nonnegative record CCC weight is defined only for REFED")
        record_groups, coordinate_weights, constant_counts = probe_training_records(a, train)
        objective = dict(loss="weighted_MSE_plus_complete_record_CCC", record_ccc_weight=spec["record_ccc_weight"],
            training_records=len(record_groups), constant_coordinates_excluded_from_CCC=constant_counts,
            constant_coordinates_retained_in_MSE=True, record_batch_size=spec["record_batch_size"])
    for epoch in range(epochs):
        model.train(); losses = []; mse_losses = []; ccc_losses = []
        if record_groups is None:
            batches = [(rows, None, None) for rows in np.array_split(rng.permutation(len(train)), math.ceil(len(train) / cfg["neural_batch_size"]))]
        else:
            batches = []
            for selected in np.array_split(rng.permutation(len(record_groups)), math.ceil(len(record_groups) / spec["record_batch_size"])):
                lengths = [len(record_groups[i]) for i in selected]; ends = np.cumsum([0, *lengths])
                batches.append((np.concatenate([record_groups[i] for i in selected]),
                    [np.arange(ends[i], ends[i + 1]) for i in range(len(selected))], coordinate_weights[selected]))
        for rows, groups, mass in batches:
            if not bool((w[rows] > 0).any()):
                continue
            optimizer.zero_grad(set_to_none=True); predicted = model(values[rows])
            if task == "visual":
                loss = nn.functional.cross_entropy(predicted[:, :4], target[rows], reduction="none")
                for c in range(2):
                    loss = loss + float(spec.get("auxiliary_weight", 0.)) / 2 * nn.functional.cross_entropy(
                        predicted[:, 4 + 2*c:6 + 2*c], attributes[target[rows], c], reduction="none")
            else:
                loss = (predicted - target[rows]).square()
            loss = (loss * w[rows]).sum() / w[rows].sum().clamp_min(1.)
            if groups is not None:
                mse_losses.append(loss.item())
                dynamic_loss = probe_record_ccc_loss(predicted, target[rows], valid[rows], groups, mass)
                ccc_losses.append(dynamic_loss.item())
                loss = loss + float(spec["record_ccc_weight"]) * dynamic_loss
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite neural probe loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); losses.append(loss.item())
        if not losses:
            raise ValueError("probe epoch has no supervised target points")
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses))))
        if record_groups is not None:
            history[-1].update(train_MSE=float(np.mean(mse_losses)), train_record_CCC_loss=float(np.mean(ccc_losses)))
        write(output / "status.json", dict(status="running", epoch=epoch + 1, max_epochs=epochs))
    model.eval(); chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), cfg["neural_batch_size"]):
            value = model(torch.as_tensor(np.asarray(x[start:start + cfg["neural_batch_size"]], np.float32), device=device))
            chunks.append(value[:, :4].argmax(-1).cpu().numpy() if task == "visual" else
                          (1. + 254. * value).transpose(1, 2).cpu().numpy())
    torch.save(dict(state_dict=model.state_dict(), config=cfg, specification=spec, feature_dim=x.shape[-1], input_shape=list(x.shape[1:])), output / "model.pt")
    return np.concatenate(chunks), dict(history=history, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), selection="fixed_final_epoch_no_held_fold_selection", **objective)


def probe_cbramod_ft(a, native, train, weights, spec, config, project, device, output):
    """Fixed-budget true encoder FT; held-fold targets never select a state."""
    from comparative_methods.CBraMod.rebench_v1 import CBraModEventProbe
    from torch import nn
    torch.manual_seed(config["model_seed"])
    profile, channels = view_profile(native, "visual")
    profile["method_root"] = str(project / "comparative_methods/CBraMod")
    model = CBraModEventProbe("visual", profile, device).to(device)
    before = {name: p.detach().cpu().clone() for name, p in model.base.encoder.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.base.encoder.parameters() if p.requires_grad], lr=float(spec["encoder_lr"])),
        dict(params=model.head.parameters(), lr=float(spec["head_lr"]))], weight_decay=float(spec["weight_decay"]))
    labels = torch.as_tensor(np.asarray(a["target"][train]), dtype=torch.long, device=device)
    weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    attributes = torch.tensor([[1, 1], [1, 0], [0, 0], [0, 1]], device=device)
    rng = np.random.default_rng(config["model_seed"]); history = []; batch_size = int(spec["batch_size"])
    def inputs(rows):
        x = torch.as_tensor(np.asarray(a["eeg"][rows])[:, channels], dtype=torch.float32, device=device)
        m = torch.as_tensor(np.asarray(a["eeg_mask"][rows])[:, channels], dtype=torch.bool, device=device)
        return x, m
    calibration = None
    if spec.get("feature_scaling") == "training_latent_columns":
        root = project / config["retained_feature_run"]
        descriptor = json.loads((root / "visual/encoder/features.json").read_text())
        identities = json.loads((root / "plan.json").read_text())["tasks"]["visual"]["sample_ids"]
        lookup = {identity: i for i, identity in enumerate(identities)}
        if descriptor["input_profile"]["channel_names"] != profile["channel_names"]:
            raise ValueError("calibration source changed real electrode identity")
        selected = np.array([lookup[str(a["sample_id"][row])] for row in train])
        features = np.load(root / "visual/encoder/encoder_event.npy", mmap_mode="r")
        model.eval()
        with torch.inference_mode():
            x, m = inputs(train[:4]); reproduced = model.event_features(x, eeg_mask=m).float().cpu().numpy()
        expected = np.asarray(features[selected[:4]])
        if not np.allclose(reproduced, expected, rtol=5e-5, atol=5e-5):
            raise ValueError("calibration features do not reproduce the declared frozen encoder and input transform")
        values = np.asarray(features[selected], float); mean = values.mean(0); scale = values.std(0)
        scale[scale <= 1e-6] = 1.
        model.feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float32, device=device))
        model.feature_scale.copy_(torch.as_tensor(scale, dtype=torch.float32, device=device))
        calibration = dict(source=str(root / "visual/encoder/features.json"), training_samples=len(train),
            training_subjects=sorted(set(a["subject_id"][train])), replay_max_abs=float(np.max(np.abs(reproduced - expected))),
            fitting="only_current_training_subjects_no_held_feature_statistics")
        model.metadata["feature_scaling"] = calibration
        del values
    for epoch in range(int(spec["epochs"])):
        model.train(); losses = []; started = time.monotonic()
        warmup = epoch < spec.get("head_warmup_epochs", 0)
        for name, parameter in model.base.encoder.named_parameters():
            if name in before:
                parameter.requires_grad_(not warmup)
        if warmup or spec.get("encoder_training_mode") == "deterministic":
            model.base.encoder.eval()
        for rows in np.array_split(rng.permutation(len(train)), math.ceil(len(train) / batch_size)):
            x, m = inputs(train[rows]); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu",
                                dtype=torch.bfloat16, enabled=str(device).startswith("cuda") and spec.get("precision", "bf16") == "bf16"):
                logits = model(x, eeg_mask=m)
                loss = nn.functional.cross_entropy(logits[:, :4].float(), labels[rows], reduction="none")
                for c in range(2):
                    loss = loss + float(spec["auxiliary_weight"]) / 2 * nn.functional.cross_entropy(
                        logits[:, 4 + 2*c:6 + 2*c].float(), attributes[labels[rows], c], reduction="none")
                loss = (loss * weights[rows]).sum() / weights[rows].sum()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite CBraMod event FT loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); losses.append(loss.item())
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), seconds=time.monotonic() - started, head_only_warmup=warmup))
        write(output / "status.json", dict(status="running", epoch=epoch + 1, max_epochs=spec["epochs"], latest=history[-1]))
    delta = math.sqrt(sum(float((p.detach().cpu() - before[name]).double().square().sum())
        for name, p in model.base.encoder.named_parameters() if name in before))
    if not delta > 0:
        raise ValueError("declared FT failed to update the encoder")
    model.eval(); predictions = []
    with torch.inference_mode():
        for start in range(0, len(a["target"]), batch_size):
            x, m = inputs(np.arange(start, min(start + batch_size, len(a["target"]))))
            with torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu",
                                dtype=torch.bfloat16, enabled=str(device).startswith("cuda") and spec.get("precision", "bf16") == "bf16"):
                predictions.append(model(x, eeg_mask=m)[:, :4].argmax(-1).cpu().numpy())
    torch.save(dict(state_dict=model.state_dict(), profile=profile, specification=spec, metadata=model.metadata), output / "model.pt")
    return np.concatenate(predictions), dict(history=history, encoder_parameter_change_l2=delta,
        trainable_encoder_parameters=sum(p.numel() for p in model.base.encoder.parameters() if p.requires_grad),
        metadata=model.metadata, calibration=calibration,
        encoder_training_mode=spec.get("encoder_training_mode", "source_dropout"),
        selection="fixed_final_epoch_no_held_fold_selection")


def probe_stimulus_student(x, a, train, weights, spec, config):
    """Predict training-stimulus templates from signals; no held identities in inference."""
    from sklearn.kernel_approximation import Nystroem
    def key(index):
        fields = dict(field.split("=", 1) for field in str(a["sample_id"][index]).split("|") if "=" in field)
        return str(a["record_id"][index]), float(fields["start"])
    keys = [key(int(index)) for index in train]
    classes = sorted(set(keys)); lookup = {value: i for i, value in enumerate(classes)}
    labels = np.array([lookup[value] for value in keys])
    mean = np.where(a["mask"][train], a["target"][train], 0.).sum((0, 2)) / a["mask"][train].sum((0, 2))
    templates = np.broadcast_to(mean[None, :, None], (len(classes), 2, 30)).copy()
    for c in range(len(classes)):
        rows = train[labels == c]; mask = a["mask"][rows]; counts = mask.sum(0)
        total = np.where(mask, a["target"][rows], 0.).sum(0)
        templates[c] = np.divide(total, counts, out=templates[c], where=counts > 0)
    mapper = Nystroem(kernel="rbf", gamma=spec["gamma"], n_components=min(spec["landmarks"], len(train)), random_state=config["model_seed"])
    mapper.fit(x[train]); values = mapper.transform(x)
    model = LogisticRegression(C=spec["C"], max_iter=config["fit"]["logistic_max_iter"], tol=config["fit"]["logistic_tol"])
    w = weights.sum((1, 2)); w *= len(w) / w.sum()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(values[train], labels, sample_weight=w)
    probabilities = model.predict_proba(values)
    prediction = (probabilities @ templates[model.classes_].reshape(len(classes), -1)).reshape(len(x), 2, 30)
    return prediction, dict(classes=classes, training_stimulus_accuracy=float(np.mean(model.predict(values[train]) == labels)),
        inferred_class_indices=model.predict(values),
        training_template_shape=list(templates.shape), iterations=model.n_iter_,
        convergence_warnings=[str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)],
        inference="signal_features_only;held_record_id_start_and_targets_never_used",
        interpretation="training_stimulus_supervised_signal_baseline;not_named_method_or_individual_emotion_evidence")


def probe_efrm_ft(a, native, train, weights, spec, config, project, device, output, *, fold, evaluation_indices=None):
    """Continue a fold's frozen temporal head while updating both EFRM encoders."""
    from torch import nn
    torch.manual_seed(config["model_seed"])
    parent = project / config["retained_feature_run"]
    parent_plan = json.loads((parent / "plan.json").read_text())
    plan = parent_plan["tasks"]["refed_regression"]
    if parent_plan["parent_run"] != config["parent_run"] or plan["sample_ids"] != a["sample_id"].tolist():
        raise ValueError("EFRM FT initialization changed source population")
    full_train = np.flatnonzero([plan["subject_folds"][str(s)] != fold for s in a["subject_id"]])
    if not set(map(int, train)).issubset(set(full_train)):
        raise ValueError("EFRM FT training crosses its retained fold")
    features = parent / "refed_regression/encoder" / f"fold_{fold}"
    descriptor = json.loads((features / "features.json").read_text())
    expected_subjects = sorted(set(a["subject_id"][full_train]))
    if descriptor["training_subjects"] != expected_subjects:
        raise ValueError("EFRM FT input scale crosses its retained fold")
    head_parent = project / config.get("initial_head_run", config["retained_feature_run"])
    head_plan = json.loads((head_parent / "plan.json").read_text())
    if head_plan["parent_run"] != config["parent_run"] or head_plan["tasks"]["refed_regression"]["sample_ids"] != a["sample_id"].tolist():
        raise ValueError("EFRM initial head changed source population")
    head_root = head_parent / "refed_regression" / spec["initial_head"] / f"fold_{fold}"
    result = json.loads((head_root / "result.json").read_text())
    if result["status"] != "completed" or result["result"]["training_subjects"] != expected_subjects:
        raise ValueError("EFRM FT requires a complete same-fold temporal head")
    profile, _ = view_profile(native, "refed_regression")
    profile["pretrained_checkpoint"] = efrm_checkpoint(config, project, "refed_regression")
    scaler = json.loads((features / "input_scaler.json").read_text())
    module = importlib.import_module("comparative_methods.EFRM-PyTorch.rebench_v1")
    train_encoder = not spec.get("freeze_encoder", False)
    encoder = module.create_model("refed_regression", profile, device, trainable=train_encoder)
    saved = torch.load(head_root / "model.pt", map_location="cpu", weights_only=True)
    if saved["feature_dim"] != encoder.feature_dim:
        raise ValueError("EFRM temporal head dimension changed")
    head = probe_temporal_head(encoder.feature_dim, int(saved["config"]["neural_width"]),
                               float(saved["specification"].get("dropout", .1))).to(device)
    head.load_state_dict(saved["state_dict"], strict=True)
    head_before = [p.detach().cpu().clone() for p in head.parameters()]
    with np.load(head_root / "transform.npz") as state:
        transform = {key: torch.as_tensor(state[key], device=device, dtype=torch.float32)
                     for key in ("mean", "scale", "counts", "block_normalizer")}
    observed = {name: getattr(encoder.backbone, name + "_model").patch_embed.proj.weight
                for name in ("eeg", "fnirs")}
    before = {name: value.detach().cpu().clone() for name, value in observed.items()}
    encoder_parameters = [p for p in encoder.parameters() if p.requires_grad]
    if train_encoder and (not encoder_parameters or not all(value.requires_grad for value in observed.values())):
        raise ValueError("EFRM fine-tuning requires both trainable signal encoders")
    if not train_encoder and encoder_parameters:
        raise ValueError("frozen encoder control has trainable encoder parameters")
    parameter_groups = [dict(params=encoder_parameters, lr=spec["encoder_learning_rate"])] if encoder_parameters else []
    parameter_groups.append(dict(params=head.parameters(), lr=spec["head_learning_rate"]))
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config["fit"]["neural_weight_decay"])
    def forward(batch):
        value = encoder(batch["eeg"], batch["fnirs"], eeg_mask=batch["channel_valid_mask"]["eeg"],
                        fnirs_mask=batch["channel_valid_mask"]["fnirs"])
        mask = encoder.sequence_feature_mask(batch["eeg"], batch["fnirs"],
            eeg_mask=batch["channel_valid_mask"]["eeg"], fnirs_mask=batch["channel_valid_mask"]["fnirs"])
        value = torch.where(mask[..., None] & (transform["counts"] > 0),
            (value - transform["mean"]) / transform["scale"] / transform["block_normalizer"], 0.)
        return head(value)
    target = torch.as_tensor(np.asarray(a["target"][train]).transpose(0, 2, 1), device=device, dtype=torch.float32)
    valid = torch.as_tensor(np.asarray(a["mask"][train]).transpose(0, 2, 1), device=device)
    target = torch.where(valid, (target - 1.) / 254., 0.)
    w = torch.as_tensor(weights, device=device, dtype=torch.float32)
    rng = np.random.default_rng(config["model_seed"]); history = []; started = time.monotonic(); updates = 0
    record_groups = None; objective = {}
    if "record_ccc_weight" in spec:
        if float(spec['record_ccc_weight']) < 0 or int(spec['effective_record_batch_size']) < 1:
            raise ValueError('nonnegative record loss and positive complete-record batch required')
        record_groups, coordinate_weights, constant_counts = probe_training_records(a, train)
        objective = dict(loss="weighted_MSE_plus_complete_record_CCC", record_ccc_weight=spec["record_ccc_weight"],
            training_records=len(record_groups), constant_coordinates_excluded_from_CCC=constant_counts,
            effective_record_batch_size=spec["effective_record_batch_size"])
    for epoch in range(int(spec["epochs"])):
        encoder.train(train_encoder); head.train(); losses = []; mse_losses = []; ccc_losses = []
        if record_groups is not None:
            order = rng.permutation(len(record_groups)); completed_windows = 0
            for first in range(0, len(order), spec["effective_record_batch_size"]):
                selected = order[first:first + spec["effective_record_batch_size"]]
                combined = np.concatenate([record_groups[i] for i in selected])
                denominator = w[combined].sum().clamp_min(1.)
                dynamic_denominator = max(float(coordinate_weights[selected].sum()), 1e-30)
                optimizer.zero_grad(set_to_none=True); total_loss = 0.; total_mse = 0.; total_ccc = 0.
                for group in selected:
                    rows = record_groups[group]
                    dataset = ArrayDataset(a, native, profile, np.asarray(train)[rows])
                    prediction = torch.cat([forward(batch) for _, batch in batches(dataset, scaler, device, batch_size=1)])
                    mse = ((prediction - target[rows]).square() * w[rows]).sum() / denominator
                    dynamic = probe_record_ccc_loss(prediction, target[rows], valid[rows], [np.arange(len(rows))], coordinate_weights[[group]])
                    dynamic = dynamic * float(coordinate_weights[group].sum()) / dynamic_denominator
                    loss = mse + float(spec["record_ccc_weight"]) * dynamic
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("nonfinite record-objective EFRM loss")
                    loss.backward(); total_loss += loss.item(); total_mse += mse.item(); total_ccc += dynamic.item()
                    del prediction, loss, mse, dynamic
                nn.utils.clip_grad_norm_([*encoder_parameters, *head.parameters()], 1., error_if_nonfinite=True)
                optimizer.step(); updates += 1; losses.append(total_loss); mse_losses.append(total_mse); ccc_losses.append(total_ccc)
                completed_windows += len(combined)
                write(output / "status.json", dict(status="running", epoch=epoch + 1, max_epochs=spec["epochs"],
                    trained_windows=completed_windows, windows_per_epoch=len(train), updates=updates, seconds=time.monotonic() - started))
            history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), train_MSE=float(np.mean(mse_losses)),
                                train_record_CCC_loss=float(np.mean(ccc_losses))))
            continue
        order = rng.permutation(len(train))
        for start in range(0, len(train), int(spec["effective_batch_size"])):
            rows = order[start:start + int(spec["effective_batch_size"])]
            denominator = w[rows].sum().clamp_min(1.)
            optimizer.zero_grad(set_to_none=True); total_loss = 0.
            dataset = ArrayDataset(a, native, profile, np.asarray(train)[rows])
            for offset, (_, batch) in enumerate(batches(dataset, scaler, device, batch_size=1)):
                prediction = forward(batch)
                loss = ((prediction[0] - target[rows[offset]]).square() * w[rows[offset]]).sum() / denominator
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite EFRM fine-tuning loss")
                loss.backward(); total_loss += loss.item()
            nn.utils.clip_grad_norm_([*encoder_parameters, *head.parameters()], 1., error_if_nonfinite=True)
            optimizer.step(); updates += 1; losses.append(total_loss)
            write(output / "status.json", dict(status="running", epoch=epoch + 1, max_epochs=spec["epochs"],
                trained_windows=min(start + len(rows), len(train)), windows_per_epoch=len(train), updates=updates,
                seconds=time.monotonic() - started))
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses))))
    changes = {name: float(torch.linalg.vector_norm(value.detach().cpu() - before[name])) for name, value in observed.items()}
    if train_encoder and not all(value > 0 for value in changes.values()):
        raise ValueError("EFRM encoder parameters did not update")
    if not train_encoder and any(value != 0 for value in changes.values()):
        raise ValueError("frozen EFRM control changed encoder parameters")
    head_change = float(torch.sqrt(sum((p.detach().cpu() - before).square().sum() for p, before in zip(head.parameters(), head_before))))
    if head_change <= 0:
        raise ValueError("EFRM temporal head did not update")
    encoder.eval(); head.eval(); predictions = []
    indices = np.arange(len(a["target"])) if evaluation_indices is None else np.asarray(evaluation_indices)
    with torch.inference_mode():
        for _, batch in batches(ArrayDataset(a, native, profile, indices), scaler, device, batch_size=1):
            predictions.append((1. + 254. * forward(batch)).transpose(1, 2).cpu().numpy())
    if evaluation_indices is None:
        torch.save(dict(encoder=encoder.state_dict() if train_encoder else None, encoder_source=profile["pretrained_checkpoint"], head=head.state_dict(), specification=spec,
            initial_head=str(head_root), input_profile=profile, training_subjects=expected_subjects), output / "model.pt")
    return np.concatenate(predictions), dict(history=history, updates=updates, encoder_change_l2_by_modality=changes, head_change_l2=head_change,
        encoder_trainable_parameters=sum(p.numel() for p in encoder_parameters), initial_head=str(head_root),
        input_scaler_source=str(features / "input_scaler.json"), feature_scaler_source=str(head_root / "transform.npz"),
        pretraining_provenance=encoder.pretraining_provenance, seconds=time.monotonic() - started,
        selection="fixed_final_epoch_no_held_fold_selection", **objective,
        interpretation="old_domain_target_excluded_pretraining_then_new_input_encoder_FT" if train_encoder else "frozen_encoder_continued_temporal_head_control")


def probe_fit(config, project, run, task, arm, fold, output, device):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.kernel_approximation import Nystroem
    from sklearn.svm import SVC
    spec = config["models"][task][arm]
    a, native = probe_inputs(config, project, task, signals=spec["estimator"] in {"cbramod_ft", "efrm_ft"})
    retained = json.loads((run / "plan.json").read_text())["tasks"][task]
    if a["sample_id"].tolist() != retained["sample_ids"]:
        raise ValueError("probe sample order changed")
    assignment = np.asarray([retained["subject_folds"][str(s)] for s in a["subject_id"]])
    train, evaluation = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
    if not len(evaluation) or set(a["subject_id"][train]) & set(a["subject_id"][evaluation]):
        raise ValueError("nonempty disjoint probe subjects required")
    weights = probe_weights(a, train, task, class_balance=spec.get("class_balance", "inverse_prior"))
    fitted = {}; caught = []
    if spec["estimator"] == "cbramod_ft":
        prediction, fitted = probe_cbramod_ft(a, native, train, weights, spec, config, project, device, output)
    elif spec["estimator"] == "efrm_ft":
        if task != "refed_regression":
            raise ValueError("EFRM temporal FT is defined only for REFED")
        prediction, fitted = probe_efrm_ft(a, native, train, weights, spec, config, project, device, output, fold=fold)
    elif spec["estimator"] == "template":
        # These are canonical retained artifact IDs, not a new raw-data parser.
        starts = np.asarray([float(dict(field.split("=", 1) for field in str(value).split("|") if "=" in field)["start"])
                             for value in a["sample_id"]])
        if not np.all(starts % 30 == 0):
            raise ValueError("REFED canonical starts must preserve the retained 30-second grid")
        mean = np.where(a["mask"][train], a["target"][train], 0.).sum((0, 2)) / a["mask"][train].sum((0, 2))
        prediction = np.broadcast_to(mean[None, :, None], a["target"].shape).copy()
        for record in np.unique(a["record_id"]):
            training = train[a["record_id"][train] == record]
            if not len(training):
                continue
            positions = starts[training, None].astype(int) + np.arange(30)
            length = int(positions.max()) + 1; sums = np.zeros((length, 2)); counts = np.zeros((length, 2))
            for c in range(2):
                valid = a["mask"][training, c]
                np.add.at(sums[:, c], positions[valid], a["target"][training, c][valid])
                np.add.at(counts[:, c], positions[valid], 1)
            for row in np.flatnonzero(a["record_id"] == record):
                time_points = starts[row].astype(int) + np.arange(30)
                keep = time_points < length
                for c in range(2):
                    supported = keep.copy(); supported[keep] &= counts[time_points[keep], c] > 0
                    prediction[row, c, supported] = sums[time_points[supported], c] / counts[time_points[supported], c]
        fitted["interpretation"] = "uses_video_identity_and_time; stimulus-only diagnostic, not signal repair"
    else:
        feature_run_name = spec.get("feature_source_run", config.get("retained_feature_run"))
        feature_run = project / feature_run_name if feature_run_name else run
        if feature_run != run:
            other = json.loads((feature_run / "plan.json").read_text())
            if other["parent_run"] != config["parent_run"] or other["tasks"][task]["sample_ids"] != retained["sample_ids"]:
                raise ValueError("retained input-only features changed identity")
        directory = feature_run / task / ("encoder" if spec["features"].startswith("encoder_") else "features")
        if config.get('frozen_sequence_encoder') or spec['features'] == 'efrm_sequence':
            directory=feature_run/task/'encoder'/f'fold_{fold}'
            descriptor=json.loads((directory/'features.json').read_text())
            if descriptor['training_subjects']!=sorted(set(a['subject_id'][train])):raise ValueError('frozen encoder input scaler crosses held fold')
        descriptor = json.loads((directory / "features.json").read_text())["features"][spec["features"]]
        if descriptor["shape"][0] != len(a["target"]):
            raise ValueError("incomplete feature population; a smoke cache cannot enter evaluation")
        feature_values = np.load(directory / (spec["features"] + ".npy"), mmap_mode="r")
        feature_mask = np.load(directory / (spec["features"] + "_mask.npy"), mmap_mode="r")
        if spec["estimator"] == "riemann_logistic":
            from comparative_methods.rebench_baselines_v1 import fnirs_tangent_features
            feature_values, references = fnirs_tangent_features(feature_values, train)
            feature_mask = np.ones(feature_values.shape, bool)
            descriptor = dict(blocks=[feature_values.shape[-1] // 2] * 2)
            np.save(output / "training_riemannian_references.npy", references, allow_pickle=False)
        if spec.get("input_view") == "mask_only":
            feature_values, feature_mask = np.asarray(feature_mask, float), np.ones(feature_mask.shape, bool)
        if spec.get("input_adaptation") == "unlabelled_subject":
            feature_values, individual = probe_subject_calibration(feature_values, feature_mask, a["subject_id"])
            np.savez(output / "subject_calibration.npz", **individual)
        x, transform = (probe_event_transform(feature_values, feature_mask, train) if spec["estimator"] == "event_conv"
                        else probe_transform(feature_values, feature_mask, train, descriptor["blocks"]))
        np.savez(output / "transform.npz", **transform)
        if spec["estimator"] in {"mlp", "temporal_conv", "event_conv"}:
            prediction, fitted = probe_neural(x, a, train, weights, task, spec, config, device, output)
        elif spec["estimator"] == "stimulus_student":
            prediction, fitted = probe_stimulus_student(x, a, train, weights, spec, config)
            inferred = fitted.pop("inferred_class_indices")[evaluation]
            pairs = [fitted["classes"][int(c)] for c in inferred]
            starts = [float(dict(field.split("=", 1) for field in str(a["sample_id"][i]).split("|") if "=" in field)["start"]) for i in evaluation]
            fitted["evaluation_video_accuracy"] = float(np.mean([p[0] == str(a["record_id"][i]) for p, i in zip(pairs, evaluation)]))
            fitted["evaluation_stimulus_accuracy"] = float(np.mean([p == (str(a["record_id"][i]), start) for p, i, start in zip(pairs, evaluation, starts)]))
            fitted["input_view"] = spec.get("input_view", "signals")
            np.savez(output / "stimulus_predictions.npz", inferred_class=inferred, sample_id=a["sample_id"][evaluation],
                class_record=np.array([p[0] for p in fitted["classes"]]), class_start=np.array([p[1] for p in fitted["classes"]]))
        elif task == "visual" and spec["estimator"] == "attribute_logistic":
            # Four-class output remains the primary target. Two independently
            # trained behavioral attributes are an explicitly named readout.
            attributes = np.array([[1, 1], [1, 0], [0, 0], [0, 1]])
            probabilities = []
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                for c in range(2):
                    labels = attributes[a["target"][train], c]
                    w = np.asarray([1 / np.sum(a["subject_id"][train] == s) for s in a["subject_id"][train]])
                    w /= np.bincount(labels, weights=w, minlength=2)[labels]
                    w *= len(w) / w.sum()
                    model = LogisticRegression(C=spec["C"], max_iter=config["fit"]["logistic_max_iter"], tol=config["fit"]["logistic_tol"])
                    model.fit(x[train], labels, sample_weight=w)
                    probabilities.append(model.predict_proba(x))
            joint = np.stack([probabilities[0][:, first] * probabilities[1][:, second] for first, second in attributes], -1)
            prediction = joint.argmax(-1)
            fitted = dict(training_samples=len(train), interpretation="factorized_attributes_with_original_four_class_scoring")
        elif task != "refed_regression":
            if spec["estimator"] == "ridge_classifier":
                model = RidgeClassifier(alpha=spec["alpha"], solver="cholesky")
            elif spec["estimator"] in {"logistic", "riemann_logistic"}:
                model = LogisticRegression(C=spec["C"], solver=spec.get("solver", "lbfgs"), tol=config["fit"]["logistic_tol"],
                    max_iter=config["fit"]["logistic_max_iter"], random_state=config["model_seed"])
            else:
                model = SVC(C=spec["C"], gamma=spec["gamma"], cache_size=1024, random_state=config["model_seed"])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x[train], a["target"][train], sample_weight=weights)
            prediction = model.predict(x)
            fitted = dict(iterations=getattr(model, "n_iter_", None), training_samples=len(train))
        elif spec["estimator"] == "trajectory_rbf":
            # The feature basis and every output coefficient use only the
            # training fold; each target second retains its own support mask.
            mapper = Nystroem(kernel="rbf", gamma=spec["gamma"], n_components=min(spec["landmarks"], len(train)), random_state=config["model_seed"])
            mapper.fit(x[train]); matrix = mapper.transform(x)
            prediction = np.empty(a["target"].shape, float)
            for c in range(2):
                for t in range(30):
                    w = weights[:, t, c]; valid = w > 0
                    if not valid.any():
                        raise ValueError("trajectory output has no training support")
                    model = Ridge(alpha=spec["alpha"])
                    model.fit(matrix[train[valid]], a["target"][train[valid], c, t], sample_weight=w[valid])
                    prediction[:, c, t] = model.predict(matrix)
            fitted = dict(landmarks=min(spec["landmarks"], len(train)), output_support="independently_masked_native_30_seconds")
        else:
            matrix = x.reshape(-1, x.shape[-1]); training_rows = (train[:, None] * 30 + np.arange(30)).reshape(-1)
            if spec["estimator"] == "nystroem_ridge":
                mapper = Nystroem(kernel="rbf", gamma=spec["gamma"], n_components=spec["landmarks"], random_state=config["model_seed"])
                mapper.fit(matrix[training_rows]); matrix = mapper.transform(matrix)
            prediction_columns = []
            y = a["target"][train].transpose(0, 2, 1).reshape(-1, 2)
            w = weights.reshape(-1, 2)
            for c in range(2):
                valid = w[:, c] > 0
                if spec["estimator"] == "extra_trees":
                    model = ExtraTreesRegressor(n_estimators=spec["trees"], min_samples_leaf=spec["min_leaf"],
                        max_depth=spec["max_depth"], n_jobs=config["resources"]["threads_per_job"], random_state=config["model_seed"])
                else:
                    model = Ridge(alpha=spec["alpha"])
                model.fit(matrix[training_rows[valid]], y[valid, c], sample_weight=w[valid, c])
                prediction_columns.append(model.predict(matrix))
            prediction = np.stack(prediction_columns, -1).reshape(len(x), 30, 2).transpose(0, 2, 1)
    if not np.isfinite(prediction[a["mask"]]).all():
        raise ValueError("nonfinite probe predictions on scored support")
    def metric(indices):
        return score_predictions(a["target"][indices], prediction[indices], a["mask"][indices], a["subject_id"][indices],
                                 task, len(TASKS[task]["classes"]), a["record_id"][indices], metric_version="v2")
    np.savez(output / "predictions.npz", prediction=prediction[evaluation], indices=evaluation,
             **{name: a[name][evaluation] for name in ("target", "mask", "sample_id", "subject_id", "record_id")})
    return dict(task=task, arm=arm, fold=fold, specification=spec, train=metric(train), evaluation=metric(evaluation),
                B0=baseline(a, train, evaluation, task), training_subjects=sorted(set(a["subject_id"][train])),
                evaluation_subjects=sorted(set(a["subject_id"][evaluation])), fit=fitted,
                convergence_warnings=[str(value.message) for value in caught if issubclass(value.category, ConvergenceWarning)] + fitted.get("convergence_warnings", []),
                input_adaptation=spec.get("input_adaptation", "none"),
                source=str(Path(__file__).resolve()), scope=config["scope"])


def probe_record_control(config, project, run, task, output):
    """Within-person independent-record control; never a cross-subject result."""
    a, _ = probe_inputs(config, project, task)
    name = config["record_controls"][task]["features"]
    directory = run / task / "features"
    descriptor = json.loads((directory / "features.json").read_text())["features"][name]
    values = np.load(directory / (name + ".npy"), mmap_mode="r")
    support = np.load(directory / (name + "_mask.npy"), mmap_mode="r")
    prediction = np.full(a["target"].shape, np.nan); null = prediction.copy(); seen = np.zeros(len(prediction), bool)
    fitted = []; excluded = []
    for subject in sorted(set(a["subject_id"])):
        rows = np.flatnonzero(a["subject_id"] == subject)
        records = sorted(set(a["record_id"][rows]))
        fold_count = 2 if task == "visual" else 3
        if len(records) < fold_count:
            excluded.append(dict(subject=subject, reason="fewer_than_required_independent_records", records=records)); continue
        shuffled = np.random.default_rng(config["fold_seed"]).permutation(records)
        assignments = {str(r): i % fold_count for i, r in enumerate(shuffled)}
        for fold in range(fold_count):
            test = rows[np.array([assignments[str(r)] == fold for r in a["record_id"][rows]])]
            train = rows[np.array([assignments[str(r)] != fold for r in a["record_id"][rows]])]
            if set(a["record_id"][train]) & set(a["record_id"][test]):
                raise ValueError("record-control leakage")
            x, _ = probe_transform(values, support, train, descriptor["blocks"])
            if task == "visual":
                labels = a["target"][train]; classes = np.unique(labels)
                if len(classes) == 1:
                    p = np.full(len(test), classes[0]); caught = []
                else:
                    model = LogisticRegression(C=1., class_weight="balanced", max_iter=config["fit"]["logistic_max_iter"], tol=config["fit"]["logistic_tol"])
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(x[train], labels)
                    p = model.predict(x[test])
                null[test] = np.bincount(labels, minlength=4).argmax()
            else:
                matrix = x[train].reshape(-1, x.shape[-1]); y = a["target"][train].transpose(0, 2, 1).reshape(-1, 2)
                weights = probe_weights(a, train, task).reshape(-1, 2)
                columns = []; caught = []
                for c in range(2):
                    valid = weights[:, c] > 0
                    model = Ridge(alpha=float(config["record_controls"][task].get("alpha", 100.)))
                    model.fit(matrix[valid], y[valid, c], sample_weight=weights[valid, c])
                    columns.append(model.predict(x[test].reshape(-1, x.shape[-1])))
                    null[test, c] = y[valid, c].mean()
                p = np.stack(columns, -1).reshape(len(test), 30, 2).transpose(0, 2, 1)
            prediction[test] = p; seen[test] = True
            fitted.append(dict(subject=subject, fold=fold, train_records=sorted(set(a["record_id"][train])),
                evaluation_records=sorted(set(a["record_id"][test])), train_samples=len(train), evaluation_samples=len(test),
                convergence_warnings=[str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]))
            write(output / "status.json", dict(status="running", subject=subject, fold=fold, fits_completed=len(fitted)))
    indices = np.flatnonzero(seen)
    if not len(indices) or not np.isfinite(prediction[indices][a["mask"][indices]]).all():
        raise ValueError("incomplete within-record diagnostic predictions")
    np.savez(output / "predictions.npz", prediction=prediction[indices], baseline=null[indices], indices=indices,
             **{key: a[key][indices] for key in ("target", "mask", "sample_id", "subject_id", "record_id")})
    def metric(p):
        return score_predictions(a["target"][indices], p[indices], a["mask"][indices], a["subject_id"][indices],
            task, 4 if task == "visual" else 0, a["record_id"][indices], metric_version="v2")
    return dict(metric=metric(prediction), baseline=metric(null), fits=fitted, excluded=excluded,
                planned_samples=len(a["target"]), evaluated_samples=len(indices),
                interpretation="within_person_independent_record_control_not_cross_subject_repair")


def probe_aggregate(config, project, run):
    cases = []; failed = []
    for task in config["tasks"]:
        a, _ = probe_inputs(config, project, task)
        for arm, spec in config["models"][task].items():
            prediction = np.full(a["target"].shape, np.nan); seen = np.zeros(len(prediction), bool)
            null_subjects = []; sources = []; warnings_count = 0
            for fold in range(3):
                directory = run / task / arm / f"fold_{fold}"; path = directory / "result.json"
                if not path.exists() or json.loads(path.read_text()).get("status") != "completed":
                    failed.append(dict(task=task, arm=arm, fold=fold, result=str(path))); continue
                result = json.loads(path.read_text())["result"]
                with np.load(directory / "predictions.npz", allow_pickle=False) as data:
                    indices = data["indices"]
                    if seen[indices].any() or not np.array_equal(data["sample_id"], a["sample_id"][indices]):
                        raise ValueError("duplicate or misidentified fold prediction")
                    if not np.array_equal(data["target"], a["target"][indices], equal_nan=True) or not np.array_equal(data["mask"], a["mask"][indices]):
                        raise ValueError("prediction target/support differs from retained owner")
                    prediction[indices] = data["prediction"]; seen[indices] = True
                null_subjects += result["B0"]["subjects"]; sources.append(str(path))
                warnings_count += len(result["convergence_warnings"])
            if not seen.all():
                continue
            if len(null_subjects) != len(set(a["subject_id"])) or {x["subject_id"] for x in null_subjects} != set(a["subject_id"]):
                raise ValueError("null baseline changed the subject denominator")
            metric = score_predictions(a["target"], prediction, a["mask"], a["subject_id"], task,
                                       len(TASKS[task]["classes"]), a["record_id"], metric_version="v2")
            if task != "refed_regression":
                b0 = max(np.mean([x["majority_macro_f1"] for x in null_subjects]), np.mean([x["prior_random_expected_macro_f1"] for x in null_subjects]))
                null = dict(f1=float(b0), accuracy=float(np.mean([x["majority_accuracy"] for x in null_subjects])))
                counts = confusion_matrix(a["target"], prediction.astype(int), labels=np.arange(len(TASKS[task]["classes"])))
                metric["confusion"] = counts.tolist()
                passes = metric["primary"] >= b0 + .03 and metric["subject_equal_accuracy"] >= null["accuracy"] + .02 and bool((np.diag(counts) > 0).all())
                if config.get("fnirs_probe"):
                    passes = metric["primary"] > b0
            else:
                null = {error + "_by_dimension": [float(np.mean([np.mean([record[error + "_by_dimension"][c] for record in subject["records"]])
                    for subject in null_subjects])) for c in range(2)] for error in ("mae", "rmse")}
                passes = metric["coverage_complete"] and all(metric["ccc_by_dimension"][c] >= .1
                    and metric["mae_by_dimension"][c] <= .95 * null["mae_by_dimension"][c]
                    and metric["rmse_by_dimension"][c] <= .95 * null["rmse_by_dimension"][c] for c in range(2))
            cases.append(dict(task=task, arm=arm, specification=spec, metric=metric, B0=null,
                              meets_point_targets=bool(passes), convergence_warnings=warnings_count, sources=sources,
                              interpretation=spec.get("interpretation", "development_probe_not_final_evaluation")))
    value = dict(schema=config["schema"], scope=config["scope"], status="completed" if not failed else "incomplete_or_failed",
                 planned_cells=sum(len(x) for x in config["models"].values()), completed_cells=len(cases), failures=failed, cases=cases)
    write(run / "summary.json", value)
    return {k: value[k] for k in ("status", "planned_cells", "completed_cells", "failures")}


def probe_verify(config, project, run):
    """Independent arithmetic on retained predictions; no signal reads."""
    from scipy.stats import binom
    summary = json.loads((run / "summary.json").read_text())
    if not summary["cases"]:
        raise ValueError("no complete probe cells to verify")
    checks = []
    for case in summary["cases"]:
        task, arm = case["task"], case["arm"]
        a, _ = probe_inputs(config, project, task)
        pred = np.full(a["target"].shape, np.nan); seen = np.zeros(len(pred), bool)
        majority_values = []; prior_values = []; accuracy_values = []; null_errors = {"mae": [], "rmse": []}
        subjects = a["subject_id"]; plan = json.loads((run / "plan.json").read_text())["tasks"][task]
        for fold in range(3):
            expected = np.array([plan["subject_folds"][str(s)] == fold for s in subjects])
            train = np.flatnonzero(~expected)
            path = run / task / arm / f"fold_{fold}" / "predictions.npz"
            with np.load(path, allow_pickle=False) as saved:
                rows = saved["indices"]
                np.testing.assert_array_equal(rows, np.flatnonzero(expected))
                for name in ("target", "mask", "sample_id", "subject_id", "record_id"):
                    np.testing.assert_array_equal(saved[name], a[name][rows])
                assert not seen[rows].any()
                pred[rows] = saved["prediction"]; seen[rows] = True
            if task != "refed_regression":
                counts = np.bincount(a["target"][train], minlength=len(TASKS[task]["classes"])); prior = counts / counts.sum(); majority = counts.argmax()
                for subject in np.unique(subjects[expected]):
                    y = a["target"][(subjects == subject) & a["mask"]]; n = len(y); number = np.bincount(y, minlength=len(TASKS[task]["classes"]))
                    majority_values.append(float(2 * number[majority] / (n + number[majority]) / len(TASKS[task]["classes"])))
                    accuracy_values.append(float(number[majority] / n))
                    values = []
                    for c in range(len(TASKS[task]["classes"])):
                        m = np.arange(n + 1)
                        score = np.divide(2 * m * number[c] / n, m + number[c], out=np.zeros(n + 1), where=m + number[c] > 0)
                        values.append(float((score * binom.pmf(m, n, prior[c])).sum()))
                    prior_values.append(float(np.mean(values)))
            else:
                mean = np.where(a["mask"][train], a["target"][train], 0.).sum((0, 2), dtype=float) / a["mask"][train].sum((0, 2))
                for subject in np.unique(subjects[expected]):
                    records = {key: [] for key in null_errors}
                    for record in np.unique(a["record_id"][subjects == subject]):
                        chosen = (subjects == subject) & (a["record_id"] == record)
                        residuals = [a["target"][chosen, c][a["mask"][chosen, c]].astype(float) - mean[c] for c in range(2)]
                        records["mae"].append([np.abs(x).mean() for x in residuals])
                        records["rmse"].append([np.sqrt((x * x).mean()) for x in residuals])
                    for key in null_errors:
                        null_errors[key].append(np.mean(records[key], axis=0))
        assert seen.all() and np.isfinite(pred[a["mask"]]).all()
        actual = case["metric"]; per_subject = []
        for subject in np.unique(subjects):
            if task != "refed_regression":
                selected = (subjects == subject) & a["mask"]
                y, p = a["target"][selected], pred[selected].astype(int)
                matrix = np.zeros((len(TASKS[task]["classes"]),) * 2, int); np.add.at(matrix, (y, p), 1)
                denominator = matrix.sum(0) + matrix.sum(1)
                f1 = np.divide(2 * matrix.diagonal(), denominator, out=np.zeros(len(TASKS[task]["classes"])), where=denominator > 0).mean()
                per_subject.append([f1, matrix.trace() / matrix.sum()])
            else:
                values = {key: [[], []] for key in ("ccc", "mae", "rmse")}
                for record in np.unique(a["record_id"][subjects == subject]):
                    selected = (subjects == subject) & (a["record_id"] == record)
                    for c in range(2):
                        mask = a["mask"][selected, c]
                        y, p = a["target"][selected, c][mask].astype(float), pred[selected, c][mask]
                        yc, pc = y - y.mean(), p - p.mean()
                        if np.ptp(y) > 0:
                            values["ccc"][c].append(float(2 * np.dot(yc, pc) / (np.dot(yc, yc) + np.dot(pc, pc) + len(y) * (y.mean() - p.mean()) ** 2)))
                        values["mae"][c].append(float(np.abs(y - p).mean()))
                        values["rmse"][c].append(float(np.sqrt(np.mean((y - p) ** 2))))
                per_subject.append([[np.mean(values[key][c]) for c in range(2)] for key in ("ccc", "mae", "rmse")])
        values = np.array(per_subject)
        if task != "refed_regression":
            np.testing.assert_allclose(values.mean(0), [actual["primary"], actual["subject_equal_accuracy"]], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(values.std(0, ddof=1), [actual["subject_sd"], actual["accuracy_subject_sd"]], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose([max(np.mean(majority_values), np.mean(prior_values)), np.mean(accuracy_values)],
                [case["B0"]["f1"], case["B0"]["accuracy"]], atol=1e-10, rtol=1e-10)
        else:
            for i, key in enumerate(("ccc", "mae", "rmse")):
                np.testing.assert_allclose(values[:, i].mean(0), actual[key + "_by_dimension"], atol=1e-9, rtol=1e-9)
                np.testing.assert_allclose(values[:, i].std(0, ddof=1), actual[key + "_subject_sd_by_dimension"], atol=1e-9, rtol=1e-9)
            for key in null_errors:
                np.testing.assert_allclose(np.mean(null_errors[key], 0), case["B0"][key + "_by_dimension"], atol=1e-9, rtol=1e-9)
        checks.append(dict(task=task, arm=arm, folds=3, subjects=len(values), samples=int(seen.sum())))
    value = dict(status="passed" if summary["status"] == "completed" else "passed_completed_outputs",
        suite_status=summary["status"], planned_cells=summary["planned_cells"],
        incomplete_or_failed_cells=summary["planned_cells"] - len(checks),
        cells=len(checks), fold_predictions=3 * len(checks), checks=checks,
        scope="independent_ACC_F1_CCC_MAE_RMSE_subject_SD_B0_and_exact_sample_target_mask_identity", raw_signal_reads=0)
    write(run / "verification.json", value)
    return value


def probe_report(run, evidence_runs):
    """Dated communication export; each run summary remains its numeric owner."""
    import html
    import shutil
    roots = list(dict.fromkeys([Path(p).resolve() for p in evidence_runs] + [run]))
    path = run / "iteration_report_v1.html"
    if path.exists():
        raise FileExistsError("use a new report export version; retain the earlier report")
    rows = []; controls = []; convergence = []; inventory = []
    for root in roots:
        summary_path = root / "summary.json"
        if not summary_path.exists():
            raise ValueError(f"missing retained summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        progress = json.loads((root / "progress.json").read_text()) if (root / "progress.json").exists() else {}
        inventory.append(dict(run=root.name, status=summary["status"], controller_status=progress.get("status"),
            counts=progress.get("progress", {}).get("counts", {}), source=str(summary_path),
            planned_cells=summary.get("planned_cells"), completed_cells=summary.get("completed_cells"),
            missing_fold_results=len(summary.get("failures", []))))
        if "planned_heads" in summary:
            for case in summary["cases"]:
                m = case["evaluation"]
                convergence.append(dict(method=case["method"], task=case["task"], old_f1=case["old_evaluation"]["primary"],
                    f1=m["primary"], f1_sd=m["subject_sd"], accuracy=m["subject_equal_accuracy"], accuracy_sd=m["accuracy_subject_sd"],
                    delta_f1=case["delta_macro_f1"], closed=case["numeric_checks"]["numerically_closed"], source=case["source_result"]))
            continue
        for case in summary["cases"]:
            m = case["metric"]; b = case["B0"]; regression = case["task"] == "refed_regression"
            row = dict(run=root.name, task=case["task"], arm=case["arm"], subjects=m["subject_count"], samples=m["sample_count"],
                point_targets=case["meets_point_targets"], warnings=case["convergence_warnings"],
                interpretation=case["interpretation"], source=str(summary_path))
            if not regression:
                row.update(f1=m["primary"], f1_sd=m["subject_sd"], accuracy=m["subject_equal_accuracy"],
                    accuracy_sd=m["accuracy_subject_sd"], B0_f1=b["f1"], B0_accuracy=b["accuracy"])
            else:
                row['records'] = sum(s['record_count'] for s in m['subjects'])
                for coordinate, label in enumerate(("valence", "arousal")):
                    row[label + '_constant_records'] = m['constant_target_record_counts'][coordinate]
                    row[label + '_dynamic_records'] = sum(s['dynamic_ccc_record_counts'][coordinate] for s in m['subjects'])
                    for key in ("ccc", "mae", "rmse"):
                        row[label + "_" + key] = m[key + "_by_dimension"][coordinate]
                        row[label + "_" + key + "_sd"] = m[key + "_subject_sd_by_dimension"][coordinate]
                        if key != "ccc":
                            row[label + "_B0_" + key] = b[key + "_by_dimension"][coordinate]
            rows.append(row)
        plan = json.loads((root / "plan.json").read_text())
        for task in plan.get("config", {}).get("record_controls", {}):
            control = root / task / "record_control/result.json"
            if control.exists():
                value = json.loads(control.read_text())
                controls.append(dict(run=root.name, task=task, source=str(control), **value))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (run / "probe_cells_v1.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with (run / "convergence_cells_v1.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(convergence[0]) if convergence else ["method"])
        writer.writeheader(); writer.writerows(convergence)
    write(run / "report_sources_v1.json", dict(runs=inventory, controls=controls,
        interpretation="dated_communication_only;owning_run_summaries_and_registry_remain_authoritative"))
    def escape(value):
        return html.escape(str(value))
    def number(value):
        return "—" if value is None else f"{float(value):.4f}"
    def pair(row, key):
        return number(row.get(key)) + " ± " + number(row.get(key + "_sd"))
    def method_label(name):
        return {'efrm_record_encoder_ft':'EFRM 编码器更新','efrm_record_frozen_control':'EFRM 冻结编码器',
            'eeg_record_ccc_0p05':'EEG 单模态时序读出','fnirs_record_ccc_0p05':'Hb 单模态时序读出',
            'event_subject_equal_rbf':'事件电位核模型（被试等权）','covariance_subject_equal_rbf':'协方差核模型（被试等权）',
            'relative_band_subject_equal_rbf':'相对频带核模型（被试等权）','training_mean':'训练均值 B0'}.get(name,name)
    def table(headers, body):
        return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+escape(h)+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+escape(v)+'</td>' for v in row)+'</tr>' for row in body) + '</tbody></table></div>'
    def section(title, selected, task):
        if task == "visual":
            headers = ['运行版本','方案','被试数','事件数','F1 ± SD','ACC ± SD','B0 F1','B0 ACC','数值点目标','收敛警告']
            body = [[r['run'].split('probe_')[-1],method_label(r['arm']),r['subjects'],r['samples'],pair(r,'f1'),pair(r,'accuracy'),
                number(r['B0_f1']),number(r['B0_accuracy']),r['point_targets'],r['warnings']] for r in selected]
        else:
            headers = ['运行版本','方案','被试数','窗口数','记录数（效价/唤醒动态可用）','效价 CCC ± SD','唤醒 CCC ± SD','效价 MAE ± SD','唤醒 MAE ± SD','效价 RMSE ± SD','唤醒 RMSE ± SD','数值点目标']
            body = [[r['run'].split('probe_')[-1],method_label(r['arm']),r['subjects'],r['samples'],f"{r['records']}（{r['valence_dynamic_records']}/{r['arousal_dynamic_records']}）",*[pair(r,k) for k in ('valence_ccc','arousal_ccc','valence_mae','arousal_mae','valence_rmse','arousal_rmse')],r['point_targets']] for r in selected]
        return '<h2>'+escape(title)+'</h2>'+table(headers,body)
    text = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>下游任务修复 probe 迭代报告</title>
<style>body{font:15px/1.65 system-ui,"Noto Sans CJK SC",sans-serif;color:#183345;margin:30px auto;padding:0 24px;max-width:1450px}h1{font-size:27px}h2{margin-top:36px;font-size:21px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}th,td{border:1px solid #d7e1e7;padding:7px;text-align:left;white-space:nowrap}th{background:#edf3f7}a{color:#136198}li{margin:6px 0}.note{background:#f3f6f8;padding:16px}</style>
<h1>下游任务修复 probe 迭代报告</h1>
<p>本报告汇总 2026-11-05 的开发性诊断。反复使用原训练／验证人口，Visual 为 12 名被试、2,805 个合并事件，REFED 为 25 名被试、1,475 个原生窗口。
所有主要候选在三个被试开发折上评分，训练种子为 17；这些数字不是最终五折结果，也不提供训练种子标准差。主指标表的“± SD”是被试间样本标准差（ddof=1）；波形误差分解中的记录内 SD 另行定义。</p>
<p>Visual 的四类标签是“编码时是否想记住 × 随后是否记住”的联合行为结果；REFED 的目标为观看视频时的连续效价／唤醒自报标注。二者都不等同于可直接从当前信号读出的生理真值。</p>
<div class="note"><b>如何理解结果：</b>程序与数值修复、开发集上的数值点目标、最终方法修复是不同事项。
刺激身份已知的模板、被试内控制、mask-only 控制及额外个体上下文诊断均单列，不能替代严格单窗信号模型的结果。
全套候选均保留；在这些开发结果中选择最高分会产生选择偏差，不能当作独立确认。</div>
<h2>计算与比较</h2><ul>
<li>Visual：每名被试先按固定四类计算 macro-F1 和 ACC，再对被试等权平均。F1 无信息参照为训练多数类与训练先验随机预测精确期望的较强者；ACC 参照为训练多数类实际表现。</li>
<li>REFED：分别计算效价、唤醒的原生误差；有效时点→记录→被试等权汇总。动态 CCC 对所有方法统一排除真实目标恒定的坐标记录，MAE/RMSE 仍覆盖其全部目标。原生标签范围为 1–255。</li>
<li>三个开发折使用同一被试分配。重复的窗口、训练方案和折数都不是新的独立被试。独立复算核对预测、目标、mask、样本身份和指标算术。</li>
<li>开发数值点目标：Visual F1 至少高于 B0 0.03、ACC 至少高 0.02，四类均有正确预测；REFED 两坐标 CCC 至少 0.10，MAE/RMSE 均较训练均值降低至少 5%。正式修复另需预指定命名参照、强基线和最终不确定性检查。</li></ul>
<h2>已定位的问题与限制</h2><ul>
<li>同一固定目标的线性头复算闭合了原收敛警告；分数变化不足以解释与强简单基线的主要差距。</li>
<li>原生 Hb 频带对照保留了完整人口与 mask。r3 Visual 重放遗漏原生 Time 网格插值，拟合被阻止；r4 版本修正后重放一致，原失败保留。</li>
<li>r6 小批量控制中，同一模型训练模式与评价模式差异很大。只读对照定位到编码器 Dropout；r7 关闭其随机性且继续更新编码器权重，小批量控制通过。该优化修复不能保证跨被试收益。</li>
<li>r8 原启动因 CPU-only 清单含无效 GPU 预留而被拒绝，未开始拟合；保留原清单，使用 source_v2/jobs_v2/launch_v2 的执行迁移。</li></ul>'''
    text += '<p>时间核对的边界：REFED 保持发布的 1 Hz 标签网格及原生视频身份，没有按信号时长拉伸标签。数据合同仍将双模态／标签共同起点列为发布假设；硬件时钟偏移与抖动缺少原始触发证据，不能由处理链重放一致性宣称已独立核实。</p>'
    latest = {}
    for row in rows:
        if row['task'] not in latest or latest[row['task']][0]['run'] != row['run']:
            latest[row['task']] = []
        latest[row['task']].append(row)
    paired_arms = ('efrm_record_encoder_ft','efrm_record_frozen_control')
    named_rows = [item for item in rows if item['task']=='refed_regression' and item['arm'] in paired_arms]
    named_rows = [item for item in named_rows if item['run']==named_rows[-1]['run']] if named_rows else []
    component_cases = list({(item['run'],item['arm']):item for item in [*latest.get('refed_regression', []),*named_rows]}.values())
    components = []; component_summary = []
    for case_index, case in enumerate(component_cases):
        root = Path(case['source']).parent
        for fold in range(3):
            folder = root / 'refed_regression' / case['arm'] / f'fold_{fold}'
            result = json.loads((folder / 'result.json').read_text())['result']
            with np.load(folder / 'predictions.npz') as saved:
                estimates = [(case['arm'], saved['prediction'])]
                if case_index == 0:
                    estimates.append(('training_mean', np.broadcast_to(np.asarray(result['B0']['train_mean'])[None, :, None], saved['target'].shape)))
                for method, predicted in estimates:
                    for subject in sorted(set(saved['subject_id'])):
                        for record in sorted(set(saved['record_id'][saved['subject_id'] == subject])):
                            selected = (saved['subject_id'] == subject) & (saved['record_id'] == record)
                            for c, coordinate in enumerate(('valence', 'arousal')):
                                support = saved['mask'][selected, c]
                                y = np.asarray(saved['target'][selected, c][support], float)
                                p = np.asarray(predicted[selected, c][support], float)
                                if not len(y) or not np.isfinite(p).all():
                                    raise ValueError('error decomposition lost scored record support')
                                centered_y = y - y.mean(); centered_p = p - p.mean()
                                mse = float(np.mean((p - y) ** 2)); bias = float((p.mean() - y.mean()) ** 2)
                                centered = float(np.mean((centered_p - centered_y) ** 2))
                                if not np.isclose(mse, bias + centered, rtol=1e-12, atol=1e-9):
                                    raise ValueError('record error decomposition is not additive')
                                sy = float(y.std()); sp = float(p.std())
                                correlation = float(np.mean(centered_y * centered_p) / (sy * sp)) if np.ptp(y) > 0 and np.ptp(p) > 0 else None
                                components.append(dict(run=case['run'], arm=method, fold=fold, subject=subject, record=record,
                                    coordinate=coordinate, points=len(y), mse=mse, mean_bias_mse=bias, centered_mse=centered,
                                    mae=float(np.mean(np.abs(p-y))), rmse=float(np.sqrt(mse)),
                                    target_sd=sy, prediction_sd=sp, pearson=correlation, prediction_source=str(folder/'predictions.npz'),
                                    baseline_source=str(folder/'result.json') if method=='training_mean' else ''))
    for arm in dict.fromkeys(item['arm'] for item in components):
        for coordinate in ('valence', 'arousal'):
            selected = [item for item in components if item['arm']==arm and item['coordinate']==coordinate]
            people = sorted(set(item['subject'] for item in selected)); means = {}
            for key in ('mse','mean_bias_mse','centered_mse','mae','rmse','target_sd','prediction_sd','pearson'):
                person_values = [[item[key] for item in selected if item['subject']==person and item[key] is not None] for person in people]
                means[key] = float(np.mean([np.mean(value) for value in person_values if value])) if any(person_values) else None
            owner = component_cases[0] if arm=='training_mean' else next(item for item in component_cases if item['arm']==arm)
            for key in ('mae','rmse'):
                expected = owner[coordinate+('_B0_' if arm=='training_mean' else '_')+key]
                if not np.isclose(means[key],expected,rtol=1e-12,atol=1e-9):
                    raise ValueError('error decomposition does not reproduce the owning error metric')
            component_summary.append(dict(arm=arm,coordinate=coordinate,subjects=len(people),records=len(selected),
                pearson_defined_records=sum(item['pearson'] is not None for item in selected),**means,
                mean_bias_fraction=means['mean_bias_mse']/means['mse']))
    if components:
        with (run/'refed_error_components_v1.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(components[0]));writer.writeheader();writer.writerows(components)
        write(run/'refed_error_components_summary_v1.json',dict(cases=component_summary,
            definition='Within each complete subject/video/coordinate: MSE = squared prediction-target mean difference + centered residual MSE. Native valid target points; all records, including constant targets. Average records within subject, then subjects equally. Diagnostic mean record MSE differs from the square of the reported mean record RMSE; no target-based recalibration is performed.',
            source='retained_fold_predictions_and_training_mean;dated_communication_derivation_not_current_state'))
    paired_effects = []
    if set(paired_arms).issubset({item['arm'] for item in named_rows}):
        pair_source = Path(named_rows[0]['source'])
        pair_cases = {item['arm']:item for item in json.loads(pair_source.read_text())['cases']}
        people = [{item['subject_id']:item for item in pair_cases[arm]['metric']['subjects']} for arm in paired_arms]
        if people[0].keys()!=people[1].keys():
            raise ValueError('paired encoder comparison changed participant denominator')
        identities = sorted(people[0]); rng = np.random.default_rng(20261105)
        draws = rng.integers(0,len(identities),size=(5000,len(identities)))
        for c, coordinate in enumerate(('valence','arousal')):
            for metric in ('ccc','mae','rmse'):
                difference = np.array([people[0][person][metric+'_by_dimension'][c]-people[1][person][metric+'_by_dimension'][c] for person in identities])
                difference *= 1 if metric=='ccc' else -1
                low,high = np.quantile(difference[draws].mean(1),[.025,.975])
                paired_effects.append(dict(coordinate=coordinate,metric=metric,subjects=len(identities),
                    effect_higher_is_better=float(difference.mean()),lower95=float(low),upper95=float(high)))
        write(run/'refed_paired_encoder_effect_v1.json',dict(cases=paired_effects,source=str(pair_source),
            methods=list(paired_arms),subjects=identities,bootstrap_draws=5000,bootstrap_seed=20261105,
            interpretation='exploratory paired subject percentile intervals conditional on fitted models/folds; no multiplicity adjustment; CCC=FT-control, errors=control-FT; not final five-fold repair inference'))
    strict_rows = [r for r in rows if not any(word in r['interpretation'] for word in ('additional_unlabelled', 'stimulus', 'mask_only'))]
    passed = sum(bool(r['point_targets']) for r in strict_rows)
    overview = '<h2>本次导出的结果判断</h2><p>纳入 '+str(len(rows))+' 个完整三折候选单元；其中单窗信号及输入处理对照 '+str(len(strict_rows))+' 个，满足全部开发数值点目标 '+str(passed)+' 个。'
    overview += ('本导出仍未得到通过预检的信号候选，不能写成 Visual／REFED 已修复。' if not passed else '点目标通过仍需按预指定命名参照、强基线及最终五折要求继续核验。')
    overview += '重复尝试不增加独立被试数；未完成的折和小样本失败列在下方，不进入成功子集的均值。</p>'
    for task, selected in latest.items():
        overview += section('最近一轮：'+('Visual' if task=='visual' else 'REFED'),selected,task)
    if named_rows and named_rows[0]['run']!=latest['refed_regression'][0]['run']:
        overview += section('最近的 EFRM 编码器配对对照',named_rows,'refed_regression')
    if component_summary:
        overview += '<h2>REFED 近期候选：误差来自哪里</h2><p>每个完整被试－视频－坐标先计算：均方误差＝预测与目标均值差的平方＋去各自均值后的波形均方差，再先记录、后被试等权。此处均方误差的单位是原生标注坐标的平方；它不是主表平均 RMSE 的平方。均值项占比是两项汇总量之比。目标 SD 和预测 SD 在各记录的有效时点上计算，以有效时点数为分母（ddof=0），再按同样顺序平均，单位为原生标注坐标；它们不是被试间 SD。波形项同时含幅度和时序差异；Pearson 只在真实目标和预测均有变化时定义，未定义记录仍保留全部误差。仅分解已有误差，没有用真实留出均值重新校准预测。</p>'
        overview += table(['方案','坐标','平均记录 MSE','均值偏差占比','中心化波形 MSE','目标 SD','预测 SD','Pearson','Pearson 可用记录/全部记录'],
            [[method_label(r['arm']),r['coordinate'],number(r['mse']),f"{100*r['mean_bias_fraction']:.1f}%",number(r['centered_mse']),number(r['target_sd']),number(r['prediction_sd']),number(r['pearson']),f"{r['pearson_defined_records']}/{r['records']}"] for r in component_summary])
        overview += '<p><a href="refed_error_components_v1.csv">逐记录分解与预测来源</a> · <a href="refed_error_components_summary_v1.json">完整汇总与计算定义</a></p>'
    if paired_effects:
        overview += '<h2>编码器更新相对冻结控制的配对变化</h2><p>两组使用相同被试、初始化头和继续训练预算。正值统一表示真实编码器更新较好：CCC 为更新组减冻结组，MAE/RMSE 为冻结组减更新组。95% percentile 区间以被试整体配对重采样 5,000 次、seed=20261105，条件于本次已拟合模型与开发折，未作多重比较校正；这是探索性变化分析，不代替最终五折修复检验。</p>'
        overview += table(['坐标','指标','被试数','配对改善','95% 区间下界','95% 区间上界'],[[r['coordinate'],r['metric'],r['subjects'],number(r['effect_higher_is_better']),number(r['lower95']),number(r['upper95'])] for r in paired_effects])
        overview += '<p><a href="refed_paired_encoder_effect_v1.json">配对分析与来源</a></p>'
    text = text.replace('<h2>计算与比较</h2>',overview+'<h2>计算与比较</h2>',1)
    text += '<p>r11 的 EFRM 时序表示按折重拟合训练侧尺度，使用全 REFED 排除的旧 v5 预训练权重；r12 在同折固定 5 轮头上继续更新两路编码器。两者均不等于新输入重新预训练。r13 的 Visual 读出使用共享时空卷积和六个有序平均／最大池化区间，保留原四分类；CBraMod 编码器在该对照中冻结。r14 仅增加同窗 Fp1/Fp2 的 7 Hz 低频投影相减，保持 r13 模型和训练设置；该投影可能同时移除神经活动，不等于实测 EOG 清理。</p>'
    text += '<p>r15 保持 r12 的两轮训练预算、初始化及头部学习率，只改变编码器学习率。r16 在相同物理电极预算内，比较 BIOT 的原单电极迁移与源 PREST 第 8–15 槽对应的八条真实双极导联；它同时改变参考和槽位语义，不是完整源蒙太奇复现。r17 保持原事件、协方差及相对频带特征，只检验更小 C 和更宽 RBF 核的固定网格；全部候选都列出，开发最高分不具有独立确认效力。训练拟合改善与未见被试预测改善分别判断。</p>'
    text += '<p>r18 的两种时序读出输入分别为原信号特征和同折 EFRM 冻结表示；损失为 MSE 加 0／0.01／0.05 倍记录内 CCC 损失，均训练 15 轮。每批包含完整被试－视频记录，零权重对照使用相同批次。CCC 不混合不同视频，真实恒定坐标从动态损失中排除但仍参与 MSE 与全部误差评分。该组只训练读出；更高的 CCC 若伴随更差原生误差，仍不能判为修复。</p>'
    text += '<p>r19 从同折 r18 的动态损失权重 0.05 读出初始化，使用同一完整记录损失继续两轮；真实更新两路编码器与保持编码器冻结的控制共享头部学习率、尺度和目标支持。r20 复用 r1／r2 的三种核分类器，仅去掉训练类别逆频率权重，保留被试等权；类别先验只来自训练人口，原四分类评分与全部事件不变。</p>'
    text += '<p>r21 使用与 r18 联合信号读出相同的 15 轮训练和权重 0.05 记录内损失，分别只输入 EEG 频带与 Hb 水平／斜率。它检查两模态联合是否影响泛化；网络宽度、原生窗口、被试折和目标支持保持一致。</p>'
    for task, label in (("visual", "Visual"), ("refed_regression", "REFED")):
        strict = [r for r in rows if r['task'] == task and 'additional_unlabelled' not in r['interpretation'] and 'stimulus' not in r['interpretation'] and 'mask_only' not in r['interpretation']]
        extra = [r for r in rows if r['task'] == task and r not in strict]
        if task == "refed_regression" and strict:
            b = strict[0]
            text += '<p>REFED 三开发折的训练均值参照：效价 MAE '+number(b['valence_B0_mae'])+'、RMSE '+number(b['valence_B0_rmse'])+'；唤醒 MAE '+number(b['arousal_B0_mae'])+'、RMSE '+number(b['arousal_B0_rmse'])+'。单位均为原生标注坐标。</p>'
            text += '<p>该人口共 '+str(b['records'])+' 个被试－视频记录，其中效价 '+str(b['valence_constant_records'])+' 个、唤醒 '+str(b['arousal_constant_records'])+' 个坐标记录的真实目标恒定。动态 CCC 的适用记录数分别为 '+str(b['valence_dynamic_records'])+'／'+str(b['arousal_dynamic_records'])+'；全部记录继续参与 MAE/RMSE。窗口、记录与被试三种计数不可互换。</p>'
        text += section(label+'：单窗信号及输入处理对照',strict,task)
        if extra:
            text += section(label+'：模板、刺激监督、掩码或额外上下文诊断',extra,task)
    text += '<h2>原固定训练／验证划分的数值复算</h2><p>此表保留原选中 C、类别权重、特征和划分，不与上述三开发折直接作差。Newton 求解后检查解析梯度；L-BFGS 从该解启动核对目标和概率一致性。</p>'
    text += table(['方法','任务','原 F1','收敛后 F1 ± SD','ACC ± SD','F1 差值','数值闭合'],[[r['method'],r['task'],number(r['old_f1']),pair(r,'f1'),pair(r,'accuracy'),number(r['delta_f1']),r['closed']] for r in convergence])
    text += '<h2>被试内独立记录控制</h2><p>按同一人的独立 Part 或视频留出，属于另一信息条件。单 Part 被试只在这一控制中标记不可用；主跨被试分母不因此删人。完整数值与排除见 report_sources_v1.json。</p>'
    text += table(['版本','任务','状态','评价被试','计划样本','实际样本'],[[r['run'],r['task'],r['status'],r.get('result',{}).get('metric',{}).get('subject_count'),r.get('result',{}).get('planned_samples'),r.get('result',{}).get('evaluated_samples')] for r in controls])
    text += '<h2>完整性与失败</h2><p>候选单元须具备全部三个开发折才进入上表。缺少折结果包括被前置检查阻止的拟合，不将其补成成功。</p>'
    text += table(['运行','汇总状态','计划候选单元','完整候选单元','缺少折结果','任务状态计数'],[[r['run'],r['status'],r['planned_cells'],r['completed_cells'],r['missing_fold_results'],r['counts']] for r in inventory])
    text += '<h2>保留证据</h2><p><a href="probe_cells_v1.csv">完整 probe 数值 CSV</a> · <a href="convergence_cells_v1.csv">数值复算 CSV</a> · <a href="report_sources_v1.json">来源与控制</a></p><ul>'
    for item in inventory:
        text += '<li><a href="'+escape(os.path.relpath(item['source'],run))+'">'+escape(item['run'])+'</a>：'+escape(item['status'])+'；控制器 '+escape(item['controller_status'])+'；'+escape(item['counts'])+'</li>'
    text += '</ul><p>各轮冻结合同、源码、服务启动和完整预测保留在对应目录；当前执行与科学状态仍由项目注册表持有。本 HTML/CSV 是沟通导出，不是第二份状态源。</p></html>'
    path.write_text(text,encoding="utf-8");shutil.copy2(Path(__file__),run/'report_renderer_source_v1.py')
    return dict(report=str(path),probe_cells=len(rows),convergence_heads=len(convergence),controls=len(controls))


def run_probe(args, config, project, run):
    if args.stage == "probe-plan":
        return probe_plan(config, project, run, args.source_root.resolve(), args.config.name)
    if args.stage == "probe-aggregate":
        result = probe_aggregate(config, project, run)
        write(run / "aggregate/result.json", dict(status="completed", result=result))
        return result
    if args.stage == "probe-verify":
        result = probe_verify(config, project, run)
        write(run / "verification/result.json", dict(status="completed", result=result))
        return result
    if args.stage == "probe-report":
        return probe_report(run,args.evidence_runs or [])
    if args.task not in config["tasks"]:
        raise ValueError("known repair task required")
    if args.stage == "probe-smoke":
        output = run / "smoke" / args.task
    elif args.stage in {"probe-features", "probe-encoder"}:
        output = run / args.task / ("encoder" if args.stage == "probe-encoder" else "features")
    elif args.stage=='probe-fold-encoder' and config.get('frozen_sequence_encoder') and args.fold in range(3):
        output=run/args.task/'encoder'/f'fold_{args.fold}'
    elif args.stage == "probe-record-control" and config.get("record_controls"):
        output = run / args.task / "record_control"
    elif args.stage == "probe-fit" and args.arm in config["models"][args.task] and args.fold in range(3):
        output = run / args.task / args.arm / f"fold_{args.fold}"
    else:
        raise ValueError("unknown fixed probe stage or arm")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists() and json.loads((output / "result.json").read_text())["status"] == "completed":
        return dict(status="already_completed", output=str(output))
    torch.set_num_threads(config["resources"]["threads_per_job"])
    write(output / "status.json", dict(status="running", pid=os.getpid(), stage=args.stage))
    try:
        with ResourceMonitor(args.device) as monitor:
            if args.stage == "probe-fit":
                result = probe_fit(config, project, run, args.task, args.arm, args.fold, output, args.device)
            elif args.stage == "probe-smoke":
                result = probe_smoke(config, project, run, args.task, output, args.device)
            elif args.stage == "probe-record-control":
                result = probe_record_control(config, project, run, args.task, output)
            elif args.stage == 'probe-fold-encoder':
                result=probe_fold_encoder(config,project,run,args.task,output,args.device,fold=args.fold)
            else:
                result = probe_extract(config, project, run, args.task, output, args.device, encoder=args.stage == "probe-encoder")
            resources = monitor.snapshot()
        write(output / "result.json", dict(schema=config["schema"], status="completed", result=result, resources=resources))
        write(output / "status.json", dict(status="completed"))
        return dict(status="completed", output=str(output))
    except Exception as error:
        write(output / "result.json", dict(status="failed", error=repr(error), stage=args.stage))
        raise


def logistic_objective_gradient(model, x, y, sample_weights):
    """Independent mean weighted log-loss + sklearn's L2/C convention."""
    from scipy.special import expit, logsumexp, softmax
    x = np.asarray(x, float); y = np.asarray(y, int); w = np.asarray(sample_weights, float)
    penalty = 1. / (float(model.C) * w.sum())
    logits = x @ model.coef_.T + model.intercept_
    if model.coef_.shape[0] == 1:
        z = logits[:, 0]; residual = (expit(z) - y) * w / w.sum()
        loss = float(np.dot(w, np.logaddexp(0, z) - y * z) / w.sum())
        coefficient_gradient = residual @ x + penalty * model.coef_[0]
        intercept_gradient = np.array([residual.sum()])
    else:
        residual = softmax(logits, axis=1); residual[np.arange(len(y)), y] -= 1
        residual *= w[:, None] / w.sum()
        loss = float(np.dot(w, logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]) / w.sum())
        coefficient_gradient = residual.T @ x + penalty * model.coef_
        intercept_gradient = residual.sum(0)
    objective = loss + .5 * penalty * float(np.square(model.coef_).sum())
    return objective, float(max(np.max(np.abs(coefficient_gradient)), np.max(np.abs(intercept_gradient))))


def fit_converged_logistic(x, y, z, setting, config):
    """Fix the objective; use Newton, gradient arithmetic and a second solver."""
    from sklearn.utils.class_weight import compute_sample_weight
    weights = compute_sample_weight(setting.get("class_weight"), y)
    base = dict(C=float(setting["C"]), class_weight=setting.get("class_weight"), random_state=17)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = LogisticRegression(**base, solver="newton-cholesky", max_iter=200, tol=1e-10).fit(x, y)
        second = LogisticRegression(**base, solver="lbfgs", warm_start=True, max_iter=20000, tol=1e-9)
        second.coef_, second.intercept_ = first.coef_.copy(), first.intercept_.copy()
        second.fit(x, y)
    checks = [logistic_objective_gradient(m, x, y, weights) for m in (first, second)]
    probabilities = [m.predict_proba(z) for m in (first, second)]
    probability_difference = float(np.max(np.abs(probabilities[0] - probabilities[1])))
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    passed = (not convergence and max(g for _, g in checks) <= config["gradient_tolerance"]
              and abs(checks[0][0] - checks[1][0]) <= config["objective_tolerance"]
              and probability_difference <= config["probability_tolerance"])
    return first, dict(numerically_closed=passed, objectives=[o for o, _ in checks], gradients_inf=[g for _, g in checks],
        evaluation_probability_max_difference=probability_difference, iterations=[m.n_iter_ for m in (first, second)],
        convergence_warnings=convergence, all_warnings=[str(w.message) for w in caught],
        second_solver_initialization="Newton solution; independent analytic gradient also checked")


def convergence_cases(config, project):
    parent = project / config["parent_run"]
    with (parent / "final_plan_gap_cases_v1.csv").open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["method_key"] in config["methods"]]
    if len(rows) != 18 or len({(r["method_key"], r["task"]) for r in rows}) != 18:
        raise ValueError("the fixed eighteen selected LP heads changed")
    for row in rows:
        path = Path(row["model_result"]).resolve()
        if not path.is_relative_to(parent.resolve()) or row["task"] == "refed_regression":
            raise ValueError("head source leaves the retained classification diagnostic")
    return rows


def run_convergence(args, config, project, run):
    rows = convergence_cases(config, project)
    if args.stage == "convergence-plan":
        jobs = []
        for row in rows:
            identity = row["task"] + "_" + row["method_key"]
            command = [str(project / ".venv/bin/python"), str(args.source_root.resolve() / "comparative_methods/performance_analysis/audit_downstream_inputs_v1.py"),
                "--config", str(args.source_root.resolve() / "comparative_methods" / args.config.name), "--project-root", str(project),
                "--run", str(run), "--stage", "convergence-fit", "--task", row["task"], "--arm", row["method_key"]]
            jobs.append(dict(id=identity, kind="diagnostic", task=row["task"], method=row["method_key"],
                command=command, cwd=str(args.source_root.resolve()), output_dir=str(run / identity), dependencies=[],
                resources=dict(cpu_threads=2, ram_gib=6, gpu=False, gpu_memory_mib=0), timeout_seconds=7200, timing_kind="fixed_logistic"))
        aggregate_command = list(jobs[0]["command"][:-4]); aggregate_command[aggregate_command.index("convergence-fit")] = "convergence-aggregate"
        jobs.append(dict(id="aggregate", kind="diagnostic", task="aggregate", method="independent_scalar_checks", command=aggregate_command,
            cwd=str(args.source_root.resolve()), output_dir=str(run / "aggregate"), dependencies=[j["id"] for j in jobs],
            resources=dict(cpu_threads=2, ram_gib=4, gpu=False, gpu_memory_mib=0), timeout_seconds=7200, timing_kind="aggregate"))
        manifest = dict(schema="downstream_rebench_schedule_v1", run_id=config["experiment_id"], run_root=str(run), source_root=str(args.source_root.resolve()),
            scope=config["scope"], jobs=jobs, limits=dict(cpu_threads=6, ram_gib=24, gpu_ids=[], max_gpu_jobs_per_device=1, gpu_reserve_mib={}))
        for name, value in (("jobs.json", manifest), ("resolved_config.json", config), ("plan.json", dict(cases=rows, scope=config["scope"], config=config))):
            if (run / name).exists() and json.loads((run / name).read_text()) != value:
                raise ValueError("convergence replay contract changed")
            write(run / name, value)
        return dict(jobs=len(jobs), heads=len(rows))
    if args.stage == "convergence-aggregate":
        cases = []; missing = []
        for row in rows:
            directory = run / (row["task"] + "_" + row["method_key"]); path = directory / "result.json"
            if not path.exists() or json.loads(path.read_text())["status"] != "completed":
                missing.append(str(path)); continue
            value = json.loads(path.read_text())["result"]
            with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
                f1 = independent_primary(saved["target"], saved["prediction"], saved["mask"], saved["subject_id"], saved["record_id"], row["task"])
                accuracies = [np.mean(saved["prediction"][saved["subject_id"] == s] == saved["target"][saved["subject_id"] == s]) for s in np.unique(saved["subject_id"])]
                np.testing.assert_allclose([f1, np.mean(accuracies), np.std(accuracies, ddof=1)],
                    [value["evaluation"]["primary"], value["evaluation"]["subject_equal_accuracy"], value["evaluation"]["accuracy_subject_sd"]], atol=1e-10)
            cases.append(value)
        summary = dict(status="completed" if not missing else "incomplete_or_failed", planned_heads=18, completed_heads=len(cases),
            numerically_closed=sum(c["numeric_checks"]["numerically_closed"] for c in cases), cases=cases, failures=missing,
            scope=config["scope"], interpretation="same_original_selected_objective_and_original_development_split;not_new_model_selection")
        write(run / "summary.json", summary); write(run / "aggregate/result.json", dict(status="completed", result=summary))
        return {k: summary[k] for k in ("status", "planned_heads", "completed_heads", "numerically_closed")}
    if args.stage != "convergence-fit":
        raise ValueError("unknown convergence stage")
    row = next(r for r in rows if r["task"] == args.task and r["method_key"] == args.arm)
    directory = run / (args.task + "_" + args.arm); directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists() and json.loads((directory / "result.json").read_text())["status"] == "completed":
        return dict(status="already_completed")
    torch.set_num_threads(2)
    with ResourceMonitor("cpu") as monitor:
        a, _ = probe_inputs(config, project, args.task)
        train, evaluation = np.flatnonzero(a["partition"] == "train"), np.flatnonzero(a["partition"] == "validation")
        source = Path(row["model_result"]).parent; old = json.loads((source / (row["model_arm"] + ".json")).read_text())
        if old["train_subjects"] != sorted(set(a["subject_id"][train])) or old["validation_subjects"] != sorted(set(a["subject_id"][evaluation])):
            raise ValueError("original LP split identity changed")
        features = np.load(source / (row["model_arm"] + "_features.npy"), mmap_mode="r")
        if features.ndim != 2 or len(features) != len(a["target"]):
            raise ValueError("LP features no longer match the retained sample population")
        # Retain the original float32 StandardScaler transform, then improve
        # solver precision without changing features, C or class weights.
        scaler = StandardScaler().fit(np.asarray(features[train]))
        x, z = scaler.transform(np.asarray(features[train])), scaler.transform(np.asarray(features[evaluation]))
        x, z, coordinates = linear_row_coordinates(x, z); x, z = np.asarray(x, float), np.asarray(z, float)
        model, checks = fit_converged_logistic(x, a["target"][train], z, old["selected"]["setting"], config)
        prediction = model.predict(z)
        metric = score_predictions(a["target"][evaluation], prediction, a["mask"][evaluation], a["subject_id"][evaluation],
            args.task, len(TASKS[args.task]["classes"]), a["record_id"][evaluation], metric_version="v2")
        np.savez(directory / "predictions.npz", prediction=prediction, probability=model.predict_proba(z),
            **{name:a[name][evaluation] for name in ("target", "mask", "sample_id", "subject_id", "record_id")})
        np.savez(directory / "head.npz", coefficient=model.coef_, intercept=model.intercept_, classes=model.classes_, mean=scaler.mean_, scale=scaler.scale_)
        value = dict(task=args.task, method=args.arm, old_arm=row["model_arm"], setting=old["selected"]["setting"],
            old_evaluation=old["selected"]["validation"], evaluation=metric, old_convergence_warnings=old["selected"]["convergence_warnings"],
            numeric_checks=checks, B0=baseline(a, train, evaluation, args.task), coordinates=coordinates,
            delta_macro_f1=metric["primary"] - old["selected"]["validation"]["primary"],
            source_result=str(source / (row["model_arm"] + ".json")), source_features=str(source / (row["model_arm"] + "_features.npy")),
            train_subjects=sorted(set(a["subject_id"][train])), evaluation_subjects=sorted(set(a["subject_id"][evaluation])))
        resources = monitor.snapshot()
    write(directory / "result.json", dict(status="completed", result=value, resources=resources))
    return dict(status="completed", output=str(directory), numerically_closed=checks["numerically_closed"])


def frozen_classification_inputs(config, project, task, *, signals=False):
    a, native = probe_inputs(config, project, task, signals=signals)
    if task == "refed_regression":
        from src.data.downstream_benchmark import refed_arousal_binary_targets
        target, mask, mean = refed_arousal_binary_targets(a["target"], a["mask"])
        a = {**a, "target": target, "mask": mask, "arousal_mean": mean}
    return a, native


def frozen_classification_prediction(path, a, indices):
    """Identity join before using a retained prediction; reject missing rows."""
    with np.load(path, allow_pickle=False) as saved:
        values = {key: saved[key] for key in ("target", "prediction", "mask", "sample_id", "subject_id", "record_id")}
        if "partition" in saved:
            values = {key: value[saved["partition"] == "validation"] for key, value in values.items()}
    ids = list(map(str, values["sample_id"]))
    if len(ids) != len(set(ids)) or set(ids) != set(a["sample_id"][indices]):
        raise ValueError("retained predictions do not cover exactly the fixed validation sample identities")
    lookup = {identity: i for i, identity in enumerate(ids)}
    order = [lookup[str(identity)] for identity in a["sample_id"][indices]]
    for key in ("target", "mask", "sample_id", "subject_id", "record_id"):
        np.testing.assert_array_equal(values[key][order], a[key][indices])
    return values["prediction"][order]


def frozen_classification_reuse(config, project, task, method, a, validation):
    parent = project / config["parent_run"]
    candidates = []
    if method in {"cbramod", "efrm"}:
        folder = parent / task / (method + "_lp")
        source = folder / "result.json"
        doc = json.loads(source.read_text())
        if doc["status"] != "completed":
            raise ValueError("complete source fit required")
        replay = None
        if method == "efrm":
            replay = next(row for row in json.loads((project / config["convergence_run"] / "summary.json").read_text())["cases"]
                          if row["task"] == task and row["method"] == "efrm_lp")
        for arm, row in doc["result"]["results"].items():
            if not arm.startswith("matched16_"):
                continue
            if replay is not None and arm == replay["old_arm"]:
                if not replay["numeric_checks"]["numerically_closed"]:
                    continue
                root = project / config["convergence_run"] / (task + "_efrm_lp")
                candidates.append((arm, root / "predictions.npz", root / "result.json"))
            elif not row["selected"]["convergence_warnings"]:
                candidates.append((arm, folder / row["prediction_file"], source))
    else:
        root = project / config["sta_completion_run"] if method == "sta_net" and task in {"wg", "nback", "dsr"} else parent
        folder = root / task / ("sta_net" if method == "sta_net" else "brainfusion_physical")
        source = folder / "result.json"
        doc = json.loads(source.read_text())
        if doc["status"] != "completed":
            raise ValueError("complete supervised source required")
        if method == "brainfusion" and doc["result"]["profile"]["nvc_hrf_grid"] != "physical_common_grid_v1":
            raise ValueError("retained BrainFusion must use the physical HRF grid")
        candidates.append((method, folder / ("predictions.npz" if method == "sta_net" else "validation_predictions.npz"), source))
    scored = []
    for arm, path, source in candidates:
        prediction = frozen_classification_prediction(path, a, validation)
        metric = score_predictions(a["target"][validation], prediction, a["mask"][validation],
            a["subject_id"][validation], task, len(TASKS[task]["classes"]), metric_version="v2")
        scored.append((metric["primary"], metric["subject_equal_accuracy"], arm, prediction, path, source))
    if not scored:
        raise ValueError("no eligible frozen or supervised retained prediction")
    best = max(scored, key=lambda row: row[:3])
    return best[3], dict(track="retained_frozen_LP" if method in {"cbramod", "efrm"} else "retained_supervised_fit",
        arm=best[2], prediction_source=str(best[4]), result_source=str(best[5]),
        selection="highest_retained_validation_macroF1_same_candidate_for_ACC_and_SD",
        candidates=[dict(arm=row[2], f1=row[0], accuracy=row[1], source=str(row[4])) for row in scored],
        encoder_updates=0)


def frozen_classification_features(config, project, run, task, method, a, native, output, device, *, smoke=False):
    """Only REFED needs new inference; other optical tasks use retained tokens."""
    if method == "efrm":
        root = project / config["parent_run"] / task / "efrm_lp"
        source = root / "matched16_window_sd_features.npy"
        result = json.loads((root / "result.json").read_text())
        if result["status"] != "completed":
            raise ValueError("incomplete EFRM features")
        x = np.load(source, mmap_mode="r", allow_pickle=False)
        if x.shape[:2] != (len(a["target"]), 30):
            raise ValueError("EFRM retained feature identity/shape changed")
        return x.reshape(len(x), -1), None, dict(feature_source=str(source),
            track="frozen_target_excluded_EFRM_old_input_domain_transfer_linear_readout", encoder_updates=0)
    if method == "mtrenirs":
        root = project / config["fnirs_feature_run"] / task / "encoder" if task != "refed_regression" else output
        if task == "refed_regression":
            fnirs_probe_extract(config, project, run, task, output, device, encoder=True,
                                sample_indices=np.arange(min(16, len(a["target"]))) if smoke else None)
        else:
            plan = json.loads((project / config["fnirs_feature_run"] / "plan.json").read_text())
            if plan["tasks"][task]["sample_ids"] != a["sample_id"].tolist():
                raise ValueError("MtReNIRS feature identities changed")
        x = np.load(root / "encoder_ordered.npy", mmap_mode="r", allow_pickle=False)
        support = np.load(root / "encoder_ordered_mask.npy", mmap_mode="r", allow_pickle=False)
        return x, support, dict(feature_source=str(root / "encoder_ordered.npy"), track="official_frozen_MtReNIRS_linear_readout",
                               encoder_updates=0, encoder_metadata=json.loads((root / "features.json").read_text())["metadata"],
                               pretraining_overlap="allowed_and_reported;Simultaneous_overlap")
    a, native = frozen_classification_inputs(config, project, task, signals=True)
    p, _ = view_profile(native, task)
    train = np.flatnonzero(a["partition"] == "train")
    indices = np.arange(min(16, len(a["target"]))) if smoke else np.arange(len(a["target"]))
    provenance = dict(encoder_updates=0)
    if method == "cbramod":
        from comparative_methods.CBraMod.rebench_v1 import create_audit_model
        scaler = fit_input_scaler(ArrayDataset(a, native, p, train))
        p.update(method_root=str(project / "comparative_methods/CBraMod"), num_classes=2,
                 classification_adaptation="refed_arousal_window_mean_midpoint_v1", audit_readout="flatten",
                 audit_head="none", audit_scale="train_sd", audit_training_eeg_scaler=scaler["eeg"])
        model = create_audit_model("refed_arousal_binary", p, device, False).eval()
        provenance.update(track="official_frozen_CBraMod_linear_readout",
            padding="unsupported_input_zero_before_encoder;fully_unsupported_one_second_tokens_zero_after_encoder",
            input_scale="same_window_eeg_SD_no_new_subject_calibration")
        scale = "window_sd"
    elif method == "sta_net":
        from comparative_methods.downstream_rebench_worker_v1 import DownstreamModel
        checkpoint_path = project / config["parent_run"] / task / "sta_net/checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        module = importlib.import_module("comparative_methods.STA-Net-PyTorch.rebench_v1")
        encoder = module.create_model(task, checkpoint["profile"], device, trainable=False)
        model = DownstreamModel(encoder, output_kind=module.OUTPUT_KIND, regression=True, classes=2).to(device)
        model.load_state_dict(checkpoint["model"])
        model.requires_grad_(False).eval()
        scaler = checkpoint["input_scaler"]
        provenance.update(track="retained_supervised_regression_STA_backbone_frozen_linear_readout",
                          checkpoint_source=str(checkpoint_path),
                          limitation="backbone_selected_on_original_validation_regression;development_only")
        scale = "train_sd"
    else:
        raise ValueError("unknown frozen feature method")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("encoder updates are forbidden")
    chunks = []
    with torch.inference_mode():
        for rows, b in batches(ArrayDataset(a, native, p, indices), scaler, device, batch_size=8, scale=scale):
            if method == "cbramod":
                # The pinned backbone has no attention-mask API. Explicitly
                # zero-fill input support, then mask unavailable output tokens.
                value = model(b["eeg"]).reshape(len(rows), len(p["channel_names"]), 30, 200)
                valid = b["channel_valid_mask"]["eeg"].reshape(len(rows), len(p["channel_names"]), 30, 200).any(-1)
                value = torch.where(valid[..., None], value, 0.).flatten(1)
            else:
                hidden = []
                hooks = [layer.register_forward_pre_hook(lambda module, args: hidden.append(args[0]))
                         for layer in (model.encoder.backbone.fusion_head, model.encoder.backbone.fnirs_head)]
                try:
                    model(b, checkpoint["job"])
                finally:
                    for hook in hooks: hook.remove()
                value = torch.cat(hidden, -1)
            chunks.append(value.float().cpu().numpy())
            write(output / "status.json", dict(status="running", stage="frozen_feature_inference", samples=sum(len(x) for x in chunks), planned=len(indices)))
    x = np.concatenate(chunks)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite frozen representations")
    np.save(output / "features.npy", x, allow_pickle=False)
    return x, None, provenance


def frozen_classification_fit(config, project, run, task, method, output, device, *, smoke=False):
    a, native = frozen_classification_inputs(config, project, task)
    train = np.flatnonzero((a["partition"] == "train") & a["mask"])
    validation = np.flatnonzero((a["partition"] == "validation") & a["mask"])
    if set(a["subject_id"][train]) & set(a["subject_id"][validation]):
        raise ValueError("fixed train/validation subjects overlap")
    classes = 2 if task == "refed_regression" else len(TASKS[task]["classes"])
    output_task = "refed_arousal_binary" if task == "refed_regression" else task
    reuse = task != "refed_regression" and method != "mtrenirs" and not (method == "brainfusion" and task == "dsr")
    if reuse:
        prediction, provenance = frozen_classification_reuse(config, project, task, method, a, validation)
    elif method == "brainfusion":
        a, native = frozen_classification_inputs(config, project, task, signals=True)
        p, _ = view_profile(native, task)
        p.update(task=output_task, num_classes=classes, nvc_hrf_grid="physical_common_grid_v1",
            classification_adaptation="refed_arousal_window_mean_midpoint_v1" if task == "refed_regression" else "dsr_same_two_second_classification_v1")
        module = importlib.import_module("comparative_methods.BrainFusion-NVC-CSP-Stacking.rebench_v1")
        model = module.create_estimator(p, seed=config["seed"], device=device)
        channels = ArrayDataset(a, native, p, []).channels
        def values(rows):
            return [torch.as_tensor(np.asarray(a["eeg"][rows])[:, channels]), torch.as_tensor(np.asarray(a["fnirs"][rows]))]
        def masks(rows):
            return dict(eeg_mask=torch.as_tensor(np.asarray(a["eeg_mask"][rows])[:, channels]),
                        fnirs_mask=torch.as_tensor(np.asarray(a["fnirs_mask"][rows])))
        if smoke:
            # Real shape/throughput smoke preserves subject grouping and classes.
            selected = []
            for subject in np.unique(a["subject_id"][train])[:6]:
                for label in range(classes):
                    selected.extend(train[(a["subject_id"][train] == subject) & (a["target"][train] == label)][:4])
            train = np.asarray(selected, int); validation = validation[:16]
        model.fit(*values(train), a["target"][train], subjects=a["subject_id"][train], sample_ids=a["sample_id"][train], **masks(train))
        prediction = model.predict(*values(validation), **masks(validation))
        provenance = dict(track="supervised_BrainFusion_classification_adaptation", encoder_updates=0,
                          fit_audit=model.audit_state(), profile=p, selection="retained_fixed_default_subject_OOF_stacking")
        model.save(output / "checkpoint")
    else:
        x, support, provenance = frozen_classification_features(config, project, run, task, method, a, native, output, device, smoke=smoke)
        if smoke:
            return dict(status="smoke_complete", shape=list(x.shape), finite=bool(np.isfinite(x).all()), provenance=provenance)
        if len(x) != len(a["target"]):
            raise ValueError("frozen features lost samples")
        if support is None: support = np.ones(x.shape, bool)
        x, state = probe_transform(x, support, train, [x.shape[-1]])
        subjects = a["subject_id"][train]
        weights = np.asarray([1. / np.sum(subjects == s) for s in subjects])
        frequency = np.bincount(a["target"][train], weights=weights, minlength=classes)
        if (frequency == 0).any(): raise ValueError("training population misses a class")
        weights /= frequency[a["target"][train]]; weights *= len(weights) / weights.sum()
        model = RidgeClassifier(alpha=float(config["linear"]["alpha"]), solver="cholesky").fit(x[train], a["target"][train], sample_weight=weights)
        prediction = model.predict(x[validation])
        np.savez(output / "readout.npz", coefficient=model.coef_, intercept=model.intercept_, classes=model.classes_, **state)
        provenance.update(readout="fixed_ridge_classifier_alpha1", feature_dimension=x.shape[-1],
                          training_subjects=sorted(set(subjects)), feature_scaler_fit="training_only")
    metric = score_predictions(a["target"][validation], prediction, a["mask"][validation], a["subject_id"][validation],
                               output_task, classes, a["record_id"][validation], metric_version="v2")
    arrays = {name: a[name][validation] for name in ("target", "mask", "subject_id", "record_id", "sample_id")}
    np.savez(output / "predictions.npz", prediction=prediction, **arrays)
    b0 = classification_baselines(a["target"][train], arrays, classes)
    return dict(task=output_task, source_task=task, method=method, metric=metric, B0=b0, provenance=provenance,
                training_samples=len(train), validation_samples=len(validation), scope=config["scope"],
                interpretation="development_validation_only_not_final_fivefold_or_new_confirmatory_test")


def frozen_classification_aggregate(config, project, run):
    cases, failures, persons, checks = [], [], [], []
    for task in config["tasks"]:
        a, _ = frozen_classification_inputs(config, project, task)
        validation = np.flatnonzero((a["partition"] == "validation") & a["mask"])
        classes = 2 if task == "refed_regression" else len(TASKS[task]["classes"])
        for method in config["methods"]:
            root = run / task / method
            result_path = root / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {"status": "missing"}
            if result["status"] != "completed":
                failures.append(dict(task=task, method=method, result=result)); continue
            row = result["result"]
            pred = frozen_classification_prediction(root / "predictions.npz", a, validation)
            independent = []
            for subject in sorted(set(a["subject_id"][validation])):
                keep = a["subject_id"][validation] == subject
                y, p = a["target"][validation][keep], pred[keep]
                matrix = np.zeros((classes, classes), int)
                np.add.at(matrix, (y, p), 1)
                denominator = matrix.sum(0) + matrix.sum(1)
                f1 = np.divide(2 * matrix.diagonal(), denominator, out=np.zeros(classes), where=denominator > 0).mean()
                item = dict(task=row["task"], method=method, subject_id=subject, samples=len(y), accuracy=float(np.mean(y == p)), f1=float(f1))
                independent.append(item); persons.append(item)
            expected = dict(subject_equal_accuracy=np.mean([r["accuracy"] for r in independent]),
                accuracy_subject_sd=np.std([r["accuracy"] for r in independent], ddof=1),
                primary=np.mean([r["f1"] for r in independent]), subject_sd=np.std([r["f1"] for r in independent], ddof=1))
            for key, value in expected.items():
                np.testing.assert_allclose(value, row["metric"][key], rtol=0, atol=1e-12)
            checks.append(dict(task=task, method=method, metrics_checked=list(expected), identities="exact"))
            row["result_source"] = str(result_path); cases.append(row)
    result = dict(schema=config["schema"], status="completed" if not failures else "incomplete", planned_cells=35,
                  completed_cells=len(cases), failures=failures, cases=cases, scope=config["scope"],
                  standard_deviation="between_validation_subject_sample_SD_ddof1_not_fold_or_seed_SD")
    write(run / "summary.json", result)
    write(run / "verification.json", dict(status="passed" if not failures else "incomplete", checks=checks, failures=failures,
        encoder_updates=0, original_test_access="none", independent_scoring="per_subject_confusion_matrix_ACC_macroF1_sample_SD"))
    rows = [dict(method=r["method"], task=r["task"], ACC=r["metric"]["subject_equal_accuracy"],
        ACC_STD=r["metric"]["accuracy_subject_sd"], F1=r["metric"]["primary"], F1_STD=r["metric"]["subject_sd"],
        subjects=r["metric"]["subject_count"], samples=r["metric"]["sample_count"],
        B0_ACC=np.mean([s["majority_accuracy"] for s in r["B0"]["subjects"]]), B0_F1=r["B0"]["b0"],
        track=r["provenance"]["track"], source=r["result_source"]) for r in cases]
    for name, values in (("metrics.csv", rows), ("per_subject.csv", persons)):
        if values:
            with (run / name).open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    return dict(status=result["status"], completed_cells=len(cases), failures=failures)


def run_frozen_classification(args, config, project, run):
    if args.stage == "plan":
        tasks = {}
        for task in config["tasks"]:
            a, profile = frozen_classification_inputs(config, project, task)
            tasks[task] = dict(profile=profile, sample_ids=a["sample_id"].tolist(), partitions={})
            for partition in config["source_partitions"]:
                chosen = a["partition"] == partition; valid = chosen & a["mask"]
                tasks[task]["partitions"][partition] = dict(subjects=sorted(set(a["subject_id"][chosen])),
                    total_windows=int(chosen.sum()), valid_windows=int(valid.sum()), class_counts=np.bincount(a["target"][valid]).tolist())
            if task == "refed_regression":
                tasks[task]["label_rule"] = config["refed_label"]
                tasks[task]["midpoint_windows"] = int(np.sum(a["arousal_mean"] == 128))
                np.savez(run / "refed_label_view.npz", mean_arousal=a["arousal_mean"], target=a["target"], mask=a["mask"], sample_id=a["sample_id"], partition=a["partition"])
        plan = dict(schema=config["schema"], tasks=tasks, scope=config["scope"], original_test_access="none", config=config)
        if (run / "plan.json").exists() and json.loads((run / "plan.json").read_text()) != serial(plan):
            raise ValueError("frozen classification contract changed; choose a new run")
        write(run / "plan.json", plan); write(run / "resolved_config.json", config)
        jobs = []
        for task in config["tasks"]:
            for method in config["methods"]:
                gpu = task == "refed_regression" and method in {"cbramod", "sta_net", "mtrenirs"}
                command = [str(project / ".venv/bin/python"), str(args.source_root.resolve() / "comparative_methods/performance_analysis/audit_downstream_inputs_v1.py"),
                    "--config", str(args.config.resolve()), "--project-root", str(project), "--run", str(run),
                    "--stage", "fit", "--task", task, "--arm", method, "--device", "{device}"]
                jobs.append(dict(id=f"{task}_{method}", task=task, method=method, kind="diagnostic", command=command,
                    cwd=str(args.source_root.resolve()), output_dir=str(run / task / method), dependencies=[],
                    resources=dict(cpu_threads=config["resources"]["threads_per_job"], ram_gib=8, gpu=gpu, gpu_memory_mib=3000 if gpu else 0), timeout_seconds=7200))
        command = [str(project / ".venv/bin/python"), str(args.source_root.resolve() / "comparative_methods/performance_analysis/audit_downstream_inputs_v1.py"),
            "--config", str(args.config.resolve()), "--project-root", str(project), "--run", str(run), "--stage", "aggregate"]
        jobs.append(dict(id="aggregate", kind="aggregate", task="aggregate", method="aggregate", command=command,
            cwd=str(args.source_root.resolve()), output_dir=str(run / "aggregate"), dependencies=[j["id"] for j in jobs], dependency_policy="terminal",
            resources=dict(cpu_threads=2, ram_gib=2, gpu=False, gpu_memory_mib=0), timeout_seconds=600))
        cfg = config["resources"]
        write(run / "jobs.json", dict(schema="downstream_rebench_schedule_v1", run_id=config["experiment_id"], run_root=str(run),
            source_root=str(args.source_root.resolve()), scope=config["scope"], jobs=jobs,
            limits=dict(cpu_threads=cfg["cpu_thread_budget"], ram_gib=cfg["ram_budget_gib"], gpu_ids=cfg["gpus"],
                max_gpu_jobs_per_device=cfg["max_gpu_jobs_per_device"], gpu_reserve_mib=cfg["gpu_reserve_mib"])))
        return dict(jobs=len(jobs), planned_cells=35, tasks={k:v["partitions"] for k,v in tasks.items()})
    if args.stage not in {"fit", "aggregate"}:
        raise ValueError("unknown frozen classification stage")
    output = run / "aggregate" if args.stage == "aggregate" else run / ("smoke" if args.smoke else "") / args.task / args.arm
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists() and json.loads((output / "result.json").read_text())["status"] == "completed":
        return dict(status="already_completed", output=str(output))
    torch.set_num_threads(config["resources"]["threads_per_job"])
    torch.manual_seed(config["seed"]); np.random.seed(config["seed"])
    try:
        with ResourceMonitor(args.device) as monitor:
            result = frozen_classification_aggregate(config, project, run) if args.stage == "aggregate" else frozen_classification_fit(
                config, project, run, args.task, args.arm, output, args.device, smoke=args.smoke)
            resources = monitor.snapshot()
        write(output / "result.json", dict(status="completed", result=result, resources=resources))
        write(output / "resource_usage.json", resources)
        write(output / "status.json", dict(status="completed"))
        return dict(status="completed", output=str(output), resources=resources)
    except Exception as error:
        write(output / "result.json", dict(status="failed", error=repr(error)))
        raise


def sta_optimization_arm(config, run, task, arm):
    if arm == "selected":
        choices = [name for name,spec in config["arms"].items() if task in spec["tasks"]]
        if len(choices) == 1:
            arm = choices[0]  # A single predeclared recipe needs no validation selection.
        else:
            selected = json.loads((run / "screen_summary.json").read_text())
            if selected["status"] != "completed":
                raise ValueError("complete screen required before locked seed repeats")
            arm = selected["selected_arms"][task]
    if arm not in config["arms"] or task not in config["arms"][arm]["tasks"]:
        raise ValueError("STA task/arm is outside the fixed comparison matrix")
    return arm, config["arms"][arm]


def sta_optimization_fit(config, project, run, task, requested_arm, seed, output, device, *, smoke=False):
    arm, spec = sta_optimization_arm(config, run, task, requested_arm)
    a, native = frozen_classification_inputs(config, project, task, signals=True)
    p, _ = view_profile(native, task)
    train = np.flatnonzero((a["partition"] == "train") & a["mask"])
    validation = np.flatnonzero((a["partition"] == "validation") & a["mask"])
    if set(a["subject_id"][train]) & set(a["subject_id"][validation]):
        raise ValueError("STA fixed subject partitions overlap")
    output_task = "refed_arousal_binary" if task == "refed_regression" else task
    classes = 2 if task == "refed_regression" else len(TASKS[task]["classes"])
    cfg = config["training"]
    support = "event_3s_eeg_13s_hb" if (spec["input_support"] != "full_admitted_window"
        and task not in {"dsr", "refed_regression"}) else "full_admitted_window"
    p.update(task=output_task, num_classes=classes, sta_implementation_version=spec["implementation"],
        sta_input_support=support, alignment_weight=spec["alignment_weight"],
        eeg_aux_weight=cfg["eeg_aux_weight"], dropout=cfg["dropout"])
    if task == "refed_regression":
        p.update(classes=["low_arousal", "high_arousal"], classification_adaptation="refed_arousal_window_mean_midpoint_v1")
    if spec["class_weight"] == "inverse_training_prior":
        counts = np.bincount(a["target"][train], minlength=classes)
        if (counts == 0).any(): raise ValueError("STA training partition lacks a declared class")
        p["sta_class_weights"] = (len(train) / (classes * counts)).tolist()
    if smoke:
        train = train[np.linspace(0, len(train)-1, min(64, len(train)), dtype=int)]
        validation = validation[np.linspace(0, len(validation)-1, min(32, len(validation)), dtype=int)]
    train_dataset, validation_dataset = [ArrayDataset(a, native, p, rows) for rows in (train, validation)]
    module = importlib.import_module("comparative_methods.STA-Net-PyTorch.rebench_v1")
    job = dict(job_id=f"sta_{task}_{requested_arm}_seed{seed}", kind="diagnostic", method="sta_net", task=output_task,
        seed=seed, track="supervised", modality_profile="multimodal", data_workers=0, precision=cfg["precision"],
        batch_size=cfg["batch_size"], microbatch_size=cfg["microbatch_size"],
        gradient_accumulation=cfg["batch_size"] // cfg["microbatch_size"],
        training=dict(max_epochs=2 if smoke else cfg["max_epochs"], min_epochs=2 if smoke else cfg["min_epochs"],
                      patience=cfg["patience"]))
    candidate = {key:cfg[key] for key in ("optimizer", "learning_rate", "weight_decay", "eps", "lr_schedule", "gradient_clip_norm")}
    model, scaler, training = fit_neural(module, p, train_dataset, validation_dataset, job, device, output, "sta_classification", candidate)
    model.eval()
    indices = np.concatenate([train, validation]); parts = []; branch_parts = {name:[] for name in ("eeg", "fusion", "fnirs")}
    gates = []; probabilities = []; loss_totals = {}
    with torch.inference_mode():
        for rows, b in batches(ArrayDataset(a, native, p, indices), model.input_scaler, device, batch_size=cfg["microbatch_size"]):
            details = model.encoder(b["eeg"], b["fnirs"], eeg_mask=b["channel_valid_mask"]["eeg"],
                                    fnirs_mask=b["channel_valid_mask"]["fnirs"], return_details=True)
            probabilities.append(details["prediction"].float().cpu().numpy())
            parts.append(details["prediction"].argmax(-1).cpu().numpy())
            gates.append(details["fusion_weights"].cpu().numpy())
            for name in branch_parts: branch_parts[name].append(details[name+"_prediction"].argmax(-1).cpu().numpy())
            terms = model.encoder.objective(details, torch.as_tensor(a["target"][rows], dtype=torch.long, device=device))
            for key, value in terms.items(): loss_totals[key] = loss_totals.get(key,0.) + float(value) * len(rows)
    prediction = np.concatenate(parts)
    np.savez(output / "predictions.npz", prediction=prediction, probability=np.concatenate(probabilities),
        fusion_weights=np.concatenate(gates), **{name+"_prediction":np.concatenate(values) for name,values in branch_parts.items()},
        **{key:a[key][indices] for key in ("target", "mask", "sample_id", "subject_id", "record_id", "partition")})
    def metric(rows, pred):
        return score_predictions(a["target"][rows], pred, a["mask"][rows], a["subject_id"][rows], output_task,
                                 classes, a["record_id"][rows], metric_version="v2")
    split = len(train)
    result = dict(task=output_task, source_task=task, arm=arm, requested_arm=requested_arm, seed=seed,
        metric=metric(validation, prediction[split:]), train=metric(train, prediction[:split]),
        branch_validation={name:metric(validation,np.concatenate(values)[split:]) for name,values in branch_parts.items()},
        mean_loss_components={key:value/len(indices) for key,value in loss_totals.items()},
        mean_fusion_weights=np.concatenate(gates)[split:].mean(0),
        B0=classification_baselines(a["target"][train], {key:a[key][validation] for key in ("target", "mask", "subject_id")}, classes),
        profile=p, model_spec=model.encoder.spec, job=job, training=training,
        initialization="random_no_retained_weights_loaded", scope="throughput_smoke_only" if smoke else config["scope"],
        interpretation="development_selection;branch_scores_are_jointly_trained_components_not_independent_unimodal_models")
    torch.save(dict(model=model.state_dict(), profile=p, job=job, input_scaler=model.input_scaler,
                    target_scaler=scaler.state(), best_epoch=training["best_epoch"]), output / "checkpoint.pt")
    return result


def sta_optimization_aggregate(config, project, run, *, final=False):
    cases = []; checks = []
    for task in config["tasks"]:
        a, _ = frozen_classification_inputs(config, project, task)
        validation = np.flatnonzero((a["partition"] == "validation") & a["mask"])
        entries = [(arm,config["seeds"]["screen"]) for arm,spec in config["arms"].items() if task in spec["tasks"]]
        if final and task in config["confirmation_tasks"]:
            entries += [("selected",seed) for seed in config["seeds"]["confirmation"]]
        for arm,seed in entries:
            root = run / task / arm / f"seed_{seed}"
            document = json.loads((root / "result.json").read_text())
            if document["status"] != "completed": raise ValueError("STA matrix has an incomplete or failed fit")
            row = document["result"]
            prediction = frozen_classification_prediction(root / "predictions.npz", a, validation)
            classes = 2 if task == "refed_regression" else len(TASKS[task]["classes"])
            person = []
            for subject in sorted(set(a["subject_id"][validation])):
                chosen = a["subject_id"][validation] == subject
                truth, pred = a["target"][validation][chosen], prediction[chosen]
                confusion = np.zeros((classes,classes),int); np.add.at(confusion,(truth,pred),1)
                denominator = confusion.sum(0)+confusion.sum(1)
                f1 = np.divide(2*confusion.diagonal(),denominator,out=np.zeros(classes),where=denominator>0).mean()
                person.append(dict(subject_id=subject, accuracy=float((truth==pred).mean()), score=float(f1)))
            for field,key,reduce in (("subject_equal_accuracy","accuracy",np.mean),("primary","score",np.mean),
                                    ("accuracy_subject_sd","accuracy",lambda x:np.std(x,ddof=1)),("subject_sd","score",lambda x:np.std(x,ddof=1))):
                np.testing.assert_allclose(row["metric"][field],reduce([p[key] for p in person]),rtol=0,atol=1e-12)
            checks.append(dict(task=task,arm=arm,seed=seed,samples=len(validation),independent_metrics=4))
            row["result_source"] = str(root / "result.json"); cases.append(row)
    selected = {}
    for task in config["tasks"]:
        eligible = [row for row in cases if row["source_task"]==task and row["requested_arm"]!="selected"]
        selected[task] = max(eligible,key=lambda row:(row["metric"]["primary"],row["metric"]["subject_equal_accuracy"],row["arm"]))["arm"]
    normalization_parent = None
    if config["schema"] == "downstream_sta_normalization_v1":
        normalization_parent = json.loads((project / config["fit_run"] / "summary.json").read_text())
        selected = normalization_parent["selected_arms"]
    value = dict(schema=config["schema"],status="completed",cases=cases,completed_fits=len(cases),
                 selected_arms=selected,scope=config["scope"],checks=checks)
    if final:
        previous = normalization_parent["selected_arms"] if normalization_parent else json.loads((run / "screen_summary.json").read_text())["selected_arms"]
        if selected != previous: raise ValueError("confirmation results must not reselect the screen arm")
        baseline = json.loads((project / config["baseline_run"] / "summary.json").read_text())
        old = {row["source_task"]:row for row in baseline["cases"] if row["method"]=="sta_net"}
        comparison = []; subject_rows = []
        for task in config["tasks"]:
            chosen = [row for row in cases if row["source_task"]==task and row["arm"]==selected[task]]
            expected_seeds = [17,42,73] if task in config["confirmation_tasks"] else [17]
            if sorted(r["seed"] for r in chosen)!=expected_seeds: raise ValueError("selected STA seed coverage is incomplete")
            subjects = sorted(p["subject_id"] for p in old[task]["metric"]["subjects"])
            if any(sorted(p["subject_id"] for p in r["metric"]["subjects"])!=subjects for r in chosen):
                raise ValueError("STA comparison changed the validation subjects")
            persons = []
            for subject in subjects:
                points = [next(p for p in r["metric"]["subjects"] if p["subject_id"]==subject) for r in chosen]
                point = dict(task=task,arm=selected[task],subject_id=subject,seeds=len(chosen),
                             accuracy=float(np.mean([p["accuracy"] for p in points])), f1=float(np.mean([p["score"] for p in points])))
                persons.append(point); subject_rows.append(point)
            b = old[task]["metric"]
            comparison.append(dict(task="refed_arousal_binary" if task=="refed_regression" else task,
                arm=selected[task],seeds=len(chosen),subjects=len(subjects),
                old_ACC=b["subject_equal_accuracy"],old_ACC_STD=b["accuracy_subject_sd"],old_F1=b["primary"],old_F1_STD=b["subject_sd"],
                ACC=float(np.mean([p["accuracy"] for p in persons])),ACC_STD=float(np.std([p["accuracy"] for p in persons],ddof=1)),
                F1=float(np.mean([p["f1"] for p in persons])),F1_STD=float(np.std([p["f1"] for p in persons],ddof=1)),
                ACC_seed_SD=float(np.std([r["metric"]["subject_equal_accuracy"] for r in chosen],ddof=1)) if len(chosen)>1 else None,
                F1_seed_SD=float(np.std([r["metric"]["primary"] for r in chosen],ddof=1)) if len(chosen)>1 else None,
                seed_F1={r["seed"]:r["metric"]["primary"] for r in chosen},
                B0_ACC=np.mean([r["majority_accuracy"] for r in chosen[0]["B0"]["subjects"]]),B0_F1=chosen[0]["B0"]["b0"]))
        value.update(comparison=comparison,standard_deviation="sample_SD_across_subjects_after_seed_mean;seed_SD_separate",baseline_seed_count=1)
        if normalization_parent:
            value.update(uncalibrated_comparison=normalization_parent["comparison"],parent_fit_run=config["fit_run"],
                normalization=config["normalization"],parameter_updates=0,selection="locked_parent_choice_no_reselection_after_calibration")
        write(run / "summary.json",value)
        write(run / "verification.json",dict(status="passed",checks=checks,comparison_subject_rows=subject_rows,original_test_access="none"))
        for name,rows in (("comparison.csv",comparison),("per_subject.csv",subject_rows)):
            with (run/name).open("w",newline="",encoding="utf-8-sig") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    else:
        write(run / "screen_summary.json",value)
    return dict(status="completed",completed_fits=len(cases),selected_arms=selected,
                comparison=value.get("comparison"))


def run_sta_optimization(args, config, project, run):
    if args.stage == "sta-plan":
        tasks = {}
        for task in config["tasks"]:
            a, _ = frozen_classification_inputs(config,project,task)
            tasks[task] = dict(sample_ids=a["sample_id"].tolist(),subjects={part:sorted(set(a["subject_id"][a["partition"]==part])) for part in config["source_partitions"]},
                counts={part:int(np.sum(a["partition"]==part)) for part in config["source_partitions"]})
        value = dict(schema=config["schema"],scope=config["scope"],tasks=tasks,config=config,original_test_access="none")
        if (run/"plan.json").exists() and json.loads((run/"plan.json").read_text())!=value:
            raise ValueError("STA contract changed; use a fresh version")
        write(run/"plan.json",value);write(run/"resolved_config.json",config)
        jobs=[]
        def add(identity,stage,task=None,arm=None,seed=17,dependencies=()):
            command=[str(project/".venv/bin/python"),str(args.source_root.resolve()/"comparative_methods/performance_analysis/audit_downstream_inputs_v1.py"),
                "--config",str(args.config.resolve()),"--project-root",str(project),"--run",str(run),"--stage",stage,"--device","{device}","--seed",str(seed)]
            if task:command += ["--task",task,"--arm",arm]
            gpu=task is not None
            output=run/task/arm/f"seed_{seed}" if task else run/identity
            long_input=task=="refed_regression" or arm=="source_full_window"
            jobs.append(dict(id=identity,kind="diagnostic" if task else "aggregate",method="sta_net" if task else stage,task=task or "aggregate",
                command=command,cwd=str(args.source_root.resolve()),output_dir=str(output),dependencies=list(dependencies),
                resources=dict(cpu_threads=2,ram_gib=12 if long_input else 8,gpu=gpu,gpu_memory_mib=(6500 if long_input else 3500) if gpu else 0),timeout_seconds=10800))
        fitting=[]
        for task in config["tasks"]:
            for arm,spec in config["arms"].items():
                if task not in spec["tasks"]:continue
                identity=f"{task}_{arm}_seed17";add(identity,"sta-fit",task,arm);fitting.append(identity)
        add("screen_aggregate","sta-screen-aggregate",dependencies=fitting)
        repeat=[]
        for task in config["confirmation_tasks"]:
            for seed in config["seeds"]["confirmation"]:
                choices=[name for name,spec in config["arms"].items() if task in spec["tasks"]]
                identity=f"{task}_selected_seed{seed}";add(identity,"sta-fit",task,"selected",seed,dependencies=[] if len(choices)==1 else ["screen_aggregate"]);repeat.append(identity)
        add("aggregate","sta-aggregate",dependencies=[*fitting,*repeat,"screen_aggregate"])
        cfg=config["resources"]
        write(run/"jobs.json",dict(schema="downstream_rebench_schedule_v1",run_id=config["experiment_id"],run_root=str(run),source_root=str(args.source_root.resolve()),jobs=jobs,scope=config["scope"],
            limits=dict(cpu_threads=cfg["cpu_thread_budget"],ram_gib=cfg["ram_budget_gib"],gpu_ids=cfg["gpus"],max_gpu_jobs_per_device=cfg["max_gpu_jobs_per_device"],gpu_reserve_mib=cfg["gpu_reserve_mib"])))
        return dict(screen_fits=len(fitting),seed_repeats=len(repeat),total_jobs=len(jobs))
    if args.stage not in {"sta-fit","sta-screen-aggregate","sta-aggregate"}:
        raise ValueError("unknown STA optimization stage")
    output = (run/("smoke" if args.smoke else "")/args.task/args.arm/f"seed_{args.seed}") if args.stage=="sta-fit" else run/("screen_aggregate" if args.stage=="sta-screen-aggregate" else "aggregate")
    output.mkdir(parents=True,exist_ok=True)
    if (output/"result.json").exists() and json.loads((output/"result.json").read_text())["status"]=="completed":
        return dict(status="already_completed",output=str(output))
    torch.set_num_threads(config["resources"]["threads_per_job"])
    try:
        with ResourceMonitor(args.device) as monitor:
            if args.stage=="sta-fit":
                result=sta_optimization_fit(config,project,run,args.task,args.arm,args.seed,output,args.device,smoke=args.smoke)
            else:result=sta_optimization_aggregate(config,project,run,final=args.stage=="sta-aggregate")
            resources=monitor.snapshot()
        write(output/"result.json",dict(status="completed",result=result,resources=resources))
        write(output/"resource_usage.json",resources);write(output/"status.json",dict(status="completed"))
        return dict(status="completed",output=str(output),resources=resources)
    except Exception as error:
        write(output/"result.json",dict(status="failed",error=repr(error)))
        raise


def sta_normalization_fit(config, project, task, arm, seed, output, device):
    parent = project / config["fit_run"]
    parent_summary = json.loads((parent / "summary.json").read_text())
    if parent_summary["status"] != "completed":
        raise ValueError("complete STA fit matrix required before statistics revision")
    selected = [row for row in parent_summary["cases"] if row["source_task"]==task and row["requested_arm"]==arm and row["seed"]==seed]
    if len(selected)!=1: raise ValueError("normalization source is not one exact retained fit")
    old = selected[0]
    source = parent / task / arm / f"seed_{seed}"
    if Path(old["result_source"]).resolve() != (source/"result.json").resolve() or old["scope"]!=config["scope"]:
        raise ValueError("normalization source leaves the retained public fit identity")
    a, native = frozen_classification_inputs(config,project,task,signals=True)
    train=np.flatnonzero((a["partition"]=="train") & a["mask"])
    validation=np.flatnonzero((a["partition"]=="validation") & a["mask"])
    checkpoint=torch.load(source/"checkpoint.pt",map_location="cpu",weights_only=False)
    profile=deepcopy(checkpoint["profile"])
    if profile["sta_implementation_version"]=="source_aligned_v2": profile["sta_implementation_version"]="source_aligned_v3"
    module=importlib.import_module("comparative_methods.STA-Net-PyTorch.rebench_v1")
    from comparative_methods.downstream_rebench_worker_v1 import DownstreamModel
    from sta_net_pytorch.source_aligned_v3 import calibrate_training_batchnorm
    encoder=module.create_model(old["task"],profile,device,trainable=False)
    model=DownstreamModel(encoder,output_kind="predictions",regression=False,classes=profile["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model"]);model.requires_grad_(False).eval()
    before={name:value.detach().cpu().clone() for name,value in model.named_parameters()}
    scaler=checkpoint["input_scaler"]
    dataset=ArrayDataset(a,native,profile,train)
    def forward_batches():
        for _,b in batches(dataset,scaler,device,batch_size=config["training"]["batch_size"]):
            yield dict(eeg=b["eeg"],fnirs=b["fnirs"],eeg_mask=b["channel_valid_mask"]["eeg"],fnirs_mask=b["channel_valid_mask"]["fnirs"])
    calibration=calibrate_training_batchnorm(model.encoder,forward_batches)
    for name,value in model.named_parameters():
        if not torch.equal(value.detach().cpu(),before[name]): raise ValueError("normalization changed a fitted parameter")
    indices=np.concatenate([train,validation]);probabilities=[]
    with torch.inference_mode():
        for _,b in batches(ArrayDataset(a,native,profile,indices),scaler,device,batch_size=config["training"]["batch_size"]):
            probabilities.append(model(b,checkpoint["job"]).softmax(-1).cpu().numpy())
    probability=np.concatenate(probabilities);prediction=probability.argmax(1)
    np.savez(output/"predictions.npz",prediction=prediction,probability=probability,
             **{key:a[key][indices] for key in ("target","mask","subject_id","record_id","sample_id","partition")})
    def score(rows,pred):
        return score_predictions(a["target"][rows],pred,a["mask"][rows],a["subject_id"][rows],old["task"],
            profile["num_classes"],a["record_id"][rows],metric_version="v2")
    result=dict(task=old["task"],source_task=task,arm=old["arm"],requested_arm=arm,seed=seed,
        metric=score(validation,prediction[len(train):]),train=score(train,prediction[:len(train)]),B0=old["B0"],
        uncalibrated_metric=old["metric"],profile=profile,job=checkpoint["job"],training=old["training"],
        calibration=calibration,source_result=str(source/"result.json"),source_checkpoint=str(source/"checkpoint.pt"),
        calibration_subjects=sorted(set(a["subject_id"][train])),calibration_samples=len(train),
        parameter_updates=0,parameter_invariance="every_parameter_exactly_equal",scope=config["scope"],
        interpretation="training_only_inference_statistics_revision;not_upstream_training_trajectory;parent_candidate_choice_locked")
    torch.save(dict(model=model.state_dict(),profile=profile,job=checkpoint["job"],input_scaler=scaler,
        target_scaler=checkpoint["target_scaler"],best_epoch=checkpoint["best_epoch"],calibration=calibration),output/"checkpoint.pt")
    return result


def run_sta_normalization(args,config,project,run):
    if args.stage=="sta-normalization-plan":
        parent=project/config["fit_run"]
        summary=json.loads((parent/"summary.json").read_text())
        expected_fits=sum(len(spec["tasks"]) for spec in config["arms"].values())+len(config["confirmation_tasks"])*len(config["seeds"]["confirmation"])
        if summary["status"]!="completed" or summary["completed_fits"]!=expected_fits:
            raise ValueError("the complete declared fit matrix is required before normalization")
        plan=dict(schema=config["schema"],scope=config["scope"],config=config,parent_plan=str(parent/"plan.json"),
            source_cases=[dict(task=row["source_task"],arm=row["requested_arm"],seed=row["seed"],source=row["result_source"]) for row in summary["cases"]],
            selected_arms=summary["selected_arms"],parameter_updates=0,original_test_access="none")
        if (run/"plan.json").exists() and json.loads((run/"plan.json").read_text())!=plan:
            raise ValueError("normalization contract changed; use a new version")
        write(run/"plan.json",plan);write(run/"resolved_config.json",config)
        jobs=[]
        for row in plan["source_cases"]:
            task,arm,seed=row["task"],row["arm"],row["seed"]
            command=[str(project/".venv/bin/python"),str(args.source_root.resolve()/"comparative_methods/performance_analysis/audit_downstream_inputs_v1.py"),
                "--config",str(args.config.resolve()),"--project-root",str(project),"--run",str(run),"--stage","sta-normalize",
                "--task",task,"--arm",arm,"--seed",str(seed),"--device","{device}"]
            jobs.append(dict(id=f"{task}_{arm}_seed{seed}",kind="diagnostic",task=task,method="sta_net",command=command,
                cwd=str(args.source_root.resolve()),output_dir=str(run/task/arm/f"seed_{seed}"),dependencies=[],
                resources=dict(cpu_threads=2,ram_gib=8,gpu=True,gpu_memory_mib=3500),timeout_seconds=1800))
        command=[str(project/".venv/bin/python"),str(args.source_root.resolve()/"comparative_methods/performance_analysis/audit_downstream_inputs_v1.py"),
            "--config",str(args.config.resolve()),"--project-root",str(project),"--run",str(run),"--stage","sta-normalization-aggregate"]
        jobs.append(dict(id="aggregate",kind="aggregate",task="aggregate",method="aggregate",command=command,
            cwd=str(args.source_root.resolve()),output_dir=str(run/"aggregate"),dependencies=[j["id"] for j in jobs],
            resources=dict(cpu_threads=2,ram_gib=2,gpu=False,gpu_memory_mib=0),timeout_seconds=600))
        cfg=config["resources"]
        write(run/"jobs.json",dict(schema="downstream_rebench_schedule_v1",run_id=config["experiment_id"],run_root=str(run),source_root=str(args.source_root.resolve()),
            scope=config["scope"],jobs=jobs,limits=dict(cpu_threads=cfg["cpu_thread_budget"],ram_gib=cfg["ram_budget_gib"],gpu_ids=cfg["gpus"],
            max_gpu_jobs_per_device=cfg["max_gpu_jobs_per_device"],gpu_reserve_mib=cfg["gpu_reserve_mib"])))
        return dict(calibration_jobs=len(jobs)-1)
    if args.stage not in {"sta-normalize","sta-normalization-aggregate"}:raise ValueError("unknown normalization stage")
    output=run/args.task/args.arm/f"seed_{args.seed}" if args.stage=="sta-normalize" else run/"aggregate"
    output.mkdir(parents=True,exist_ok=True)
    if (output/"result.json").exists() and json.loads((output/"result.json").read_text())["status"]=="completed":
        return dict(status="already_completed")
    torch.set_num_threads(config["resources"]["threads_per_job"])
    try:
        with ResourceMonitor(args.device) as monitor:
            result=sta_normalization_fit(config,project,args.task,args.arm,args.seed,output,args.device) if args.stage=="sta-normalize" else sta_optimization_aggregate(config,project,run,final=True)
            resources=monitor.snapshot()
        write(output/"result.json",dict(status="completed",result=result,resources=resources))
        write(output/"resource_usage.json",resources);write(output/"status.json",dict(status="completed"))
        return dict(status="completed",output=str(output),resources=resources)
    except Exception as error:
        write(output/"result.json",dict(status="failed",error=repr(error)))
        raise


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
    parser.add_argument("--arm")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--evidence-runs", nargs="*", type=Path)
    args = parser.parse_args(); config = read_config(args.config)
    run, project = args.run.resolve(), args.project_root.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if config.get("schema") == "downstream_sta_normalization_v1":
        print(json.dumps(serial(run_sta_normalization(args,config,project,run)))); return
    if config.get("schema") == "downstream_sta_optimization_v1":
        print(json.dumps(serial(run_sta_optimization(args, config, project, run)))); return
    if config.get("schema") == "downstream_frozen_classification_v1":
        print(json.dumps(serial(run_frozen_classification(args, config, project, run)))); return
    if config.get("schema") == "downstream_convergence_probe_v1":
        print(json.dumps(serial(run_convergence(args, config, project, run)))); return
    if config.get("schema") in {"downstream_repair_probe_v1", "downstream_fnirs_probe_v1"}:
        print(json.dumps(serial(run_probe(args, config, project, run)))); return
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
