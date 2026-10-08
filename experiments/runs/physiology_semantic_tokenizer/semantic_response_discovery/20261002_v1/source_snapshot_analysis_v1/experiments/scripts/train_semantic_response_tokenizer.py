#!/usr/bin/env python3
"""Finite synthetic semantic-response tokenizer training and frozen evaluation.

Launch substantial prepare/train/evaluate work under the owning durable
supervisor. This entrypoint never reads measured data. ``export`` additionally
accepts an explicit prepared NPZ with EEG/Hb arrays and applies the synthetic
training coordinate unchanged; such exports do not qualify measured physiology.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizers.semantic_response_tokenizer import (  # noqa: E402
    SemanticResponseTokenizer, gaussian_crps, gaussian_nll, response_loss,
)

ARMS = ("observation_only_capacity_matched", "point_restricted_generator",
        "point_randomized_generator", "distribution_randomized_generator")
ARM_ALIASES = dict(zip(("observation_only", "point_restricted", "point_randomized",
                        "distribution_randomized"), ARMS))
TRAIN_SCENARIOS = ("baseline", "true_a_amplitude", "true_a_shape", "feature_gain_only",
                   "Hb_gain_only", "common_Hb", "tau_change", "initial_change")
EVALUATION_SCENARIOS = TRAIN_SCENARIOS + (
    "non_dct_impulse", "ou_drive", "time_varying_feature_gain", "rotated_spectral_mixture",
    "correlated_Hb_component", "input_lowpass", "viscoelastic_outflow",
    "independent_Hb_drive", "no_coupling", "alternative_hrf", "exact_eeg_alias", "exact_joint_alias",
)
NORMAL_QUANTILES = {"50": 0.6744897501960817, "80": 1.2815515655446004,
                    "95": 1.959963984540054}


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_contract(path):
    raw = Path(path).read_bytes()
    contract = yaml.safe_load(raw)
    if not isinstance(contract, dict):
        raise ValueError("configuration must be a mapping")
    options = dict(contract.get("tokenizer", {}))
    if "observation_dim" in options:
        options.setdefault("observation_dimensions", options["observation_dim"])
    if "base_counts" in options:
        options.setdefault("data_counts", options["base_counts"])
    if "arms" in options and tuple(ARM_ALIASES.get(arm, arm) for arm in options["arms"]) != ARMS:
        raise ValueError("the finite experiment requires the four frozen arms in their declared order")
    defaults = dict(width=64, observation_dimensions=8, learning_rate=0.001, epochs=160,
                    patience=20, batch_size=64, seeds=[101, 202, 303],
                    reconstruction_weight=1.0, source_weight=1.0, amplitude_weight=0.25,
                    ridge_alpha=1.0, reference_steps=20, patches=12,
                    train_scenarios=list(TRAIN_SCENARIOS),
                    evaluation_scenarios=list(EVALUATION_SCENARIOS))
    options = {**defaults, **options}
    options["generator_seed"] = int(contract.get("seed", 20261002))
    if not 1 <= int(options["epochs"]) <= 160 or not 1 <= int(options["patience"]) <= 20:
        raise ValueError("frozen budget requires 1..160 epochs and 1..20 patience")
    if int(options["batch_size"]) < 1 or float(options["ridge_alpha"]) <= 0:
        raise ValueError("batch size and ridge penalty must be positive")
    return contract, options, hashlib.sha256(raw).hexdigest()


def family_for_arm(arm):
    return "restricted" if arm == "point_restricted_generator" else "randomized"


def _mask(array, shape):
    array = np.asarray(array, dtype=bool)
    if array.ndim == 1:
        array = array[:, None]
    return np.broadcast_to(array, shape).copy()


def _generate_case_batch(specs):
    """Worker returns a result or a retained failure for every requested spec."""
    from src.inference.semantic_response_synthetic import generate_response_case
    results = []
    for spec in specs:
        try:
            results.append(dict(spec=spec, case=generate_response_case(**spec)))
        except Exception as exc:
            results.append(dict(spec=spec, failure=dict(type=type(exc).__name__, error=str(exc))))
    return results


def _bounded_generated_cases(specs, workers):
    """Deterministic ordered results with at most 32 * workers cases in flight."""
    chunks = [specs[start:start + 16] for start in range(0, len(specs), 16)]
    if workers == 1:
        for chunk in chunks:
            yield from _generate_case_batch(chunk)
        return
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as executor:
        pending = deque()
        iterator = iter(chunks)
        for _ in range(min(2 * workers, len(chunks))):
            pending.append(executor.submit(_generate_case_batch, next(iterator)))
        while pending:
            yield from pending.popleft().result()
            chunk = next(iterator, None)
            if chunk is not None:
                pending.append(executor.submit(_generate_case_batch, chunk))


def prepare(output, options, identity, workers=1, family_filter=None, partition_filter=None):
    from src.inference.semantic_response_synthetic import response_case_index
    if workers < 1:
        raise ValueError("prepare workers must be positive")
    counts = {"train": 512, "selection": 128, "evaluation": 256}
    # Reduced counts are solely an explicitly declared software/pilot fixture.
    counts.update(options.get("data_counts", {}))
    for family in ("restricted", "randomized"):
        if family_filter is not None and family != family_filter:
            continue
        for partition, count in counts.items():
            if partition_filter is not None and partition != partition_filter:
                continue
            if family == "restricted" and partition == "evaluation":
                continue  # The same randomized evaluation is scored for all arms.
            destination = output / "prepared" / family / f"{partition}.npz"
            manifest_path = destination.with_suffix(".json")
            scenarios = options["evaluation_scenarios"] if partition == "evaluation" else options["train_scenarios"]
            expected = int(count) * len(scenarios) * 2
            if destination.exists() and manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if manifest["config_identity"] != identity:
                    raise ValueError("existing preparation has another configuration identity")
                if manifest["status"] == "complete":
                    if manifest["expected_records"] != expected or manifest["completed_records"] != expected:
                        raise ValueError("completed preparation count differs from the frozen exposure")
                    continue
            columns = {key: [] for key in ("eeg", "hb", "eeg_mask", "hb_mask", "source_shape",
                        "source_mask", "log_amplitude", "a", "base_id", "scenario", "variant",
                        "metadata_json")}
            failures = []
            started = time.monotonic()
            specs = response_case_index(partition, family=family, scenarios=tuple(scenarios),
                                        seed=int(options.get("generator_seed", 20261002)),
                                        indices=range(int(count)))
            iterator = _bounded_generated_cases(specs, workers)
            for ordinal in range(len(specs)):
                try:
                    generated = next(iterator)
                except Exception as exc:
                    failures.append(dict(ordinal=ordinal, type=type(exc).__name__, error=str(exc),
                        unresolved_cases=len(specs) - ordinal, failure_boundary="generation_worker_pool"))
                    break
                if "failure" in generated:
                    failures.append(dict(ordinal=ordinal, spec=generated["spec"], **generated["failure"]))
                    continue
                case = generated["case"]
                try:
                    metadata = case["metadata"]
                    base_id = str(metadata.get("base_id", ordinal // len(scenarios)))
                    scenario = str(metadata.get("scenario", scenarios[ordinal % len(scenarios)]))
                    for variant in ("control", "intervention"):
                        record = case[variant]
                        eeg = np.asarray(record["eeg"], dtype=np.float32)
                        hb = np.asarray(record["hb"], dtype=np.float32)
                        if eeg.shape != (120, 30) or hb.shape != (120, 2):
                            raise ValueError("generator violated EEG[120,30]/Hb[120,2]")
                        source = np.asarray(record["source_shape"], dtype=np.float32)
                        if source.shape != (120,) or not np.isfinite(source).all():
                            raise ValueError("generator violated finite continuous source shape")
                        eeg_mask, hb_mask = _mask(record["eeg_mask"], eeg.shape), _mask(record["hb_mask"], hb.shape)
                        if not np.isfinite(eeg[eeg_mask]).all() or not np.isfinite(hb[hb_mask]).all():
                            raise ValueError("generator returned nonfinite visible observations")
                        values = dict(eeg=eeg, hb=hb, eeg_mask=eeg_mask, hb_mask=hb_mask,
                            source_shape=source, source_mask=np.asarray(record["source_mask"], bool),
                            log_amplitude=float(record["log_amplitude"]),
                            a=np.asarray(record["a"], np.float32), base_id=base_id,
                            scenario=scenario, variant=variant,
                            metadata_json=json.dumps(record.get("metadata", metadata), sort_keys=True))
                        for key, value in values.items():
                            columns[key].append(value)
                except Exception as exc:
                    failures.append({"ordinal": ordinal, "type": type(exc).__name__, "error": str(exc)})
                if ordinal % 64 == 0:
                    atomic_json(manifest_path, dict(status="running", config_identity=identity,
                        family=family, partition=partition, expected_records=expected,
                        completed_records=len(columns["eeg"]), failures=failures,
                        workers=workers, bounded_cases_in_flight=32 * workers,
                        generator_seed=int(options.get("generator_seed", 20261002)),
                        elapsed_s=time.monotonic() - started))
            arrays = {key: np.asarray(value) for key, value in columns.items()}
            complete = not failures and len(columns["eeg"]) == expected
            atomic_npz(destination, **arrays)
            atomic_json(manifest_path, dict(status="complete" if complete else "failed",
                config_identity=identity, family=family, partition=partition,
                base_ids=int(count), scenarios=scenarios, expected_records=expected,
                completed_records=len(columns["eeg"]), failures=failures,
                workers=workers, bounded_cases_in_flight=32 * workers,
                generator_seed=int(options.get("generator_seed", 20261002)),
                elapsed_s=time.monotonic() - started,
                coordinate="generator_reference_centered_RMS_shape_and_log_amplitude"))
            if not complete:
                raise RuntimeError(f"incomplete generation retained at {manifest_path}")


def load_prepared(output, arm, partition, identity, path=None):
    source = Path(path) if path else output / "prepared" / family_for_arm(arm) / f"{partition}.npz"
    if path is None:
        manifest = json.loads(source.with_suffix(".json").read_text())
        if manifest["status"] != "complete" or manifest["config_identity"] != identity:
            raise ValueError("preparation must be complete with the same frozen contract")
    with np.load(source, allow_pickle=False) as handle:
        result = {key: handle[key] for key in handle.files}
    if len(result["eeg"]) == 0 or result["eeg"].shape[1:] != (120, 30) or result["hb"].shape[1:] != (120, 2):
        raise ValueError("prepared arrays must be nonempty EEG[N,120,30]/Hb[N,120,2]")
    if path is None and len(result["eeg"]) != manifest["expected_records"]:
        raise ValueError("prepared arrays do not match their complete-record denominator")
    for modality in ("eeg", "hb"):
        result.setdefault(f"{modality}_mask", np.ones_like(result[modality], dtype=bool))
    return result


def make_model(arm, options):
    return SemanticResponseTokenizer(width=int(options["width"]),
        observation_dimensions=int(options["observation_dimensions"]),
        reference_steps=int(options["reference_steps"]), patches=int(options["patches"]),
        distribution=arm == "distribution_randomized_generator",
        observation_only=arm == "observation_only_capacity_matched")


def tensor_batch(data, indices, device, amplitude_mean, amplitude_scale):
    result = {}
    for key in ("eeg", "hb", "eeg_mask", "hb_mask", "source_shape", "source_mask", "log_amplitude"):
        dtype = torch.bool if key.endswith("mask") else torch.float32
        result[key] = torch.as_tensor(data[key][indices], dtype=dtype, device=device)
    result["log_amplitude"] = (result["log_amplitude"] - amplitude_mean) / amplitude_scale
    return result


def batch_loss(model, batch, options):
    outputs = model(batch["eeg"], batch["hb"], batch["eeg_mask"], batch["hb_mask"])
    losses = response_loss(model, outputs, batch["eeg"], batch["hb"],
        batch["source_shape"], batch["log_amplitude"], batch["eeg_mask"], batch["hb_mask"],
        batch["source_mask"], reconstruction_weight=float(options["reconstruction_weight"]),
        amplitude_weight=float(options["amplitude_weight"]),
        source_weight=float(options["source_weight"]))
    return losses


@torch.no_grad()
def predictions(model, data, device, batch_size, amplitude_mean, amplitude_scale):
    model.eval()
    columns = {modality: {} for modality in ("eeg", "hb")}
    for start in range(0, len(data["eeg"]), batch_size):
        end = min(start + batch_size, len(data["eeg"]))
        eeg = torch.as_tensor(data["eeg"][start:end], dtype=torch.float32, device=device)
        hb = torch.as_tensor(data["hb"][start:end], dtype=torch.float32, device=device)
        outputs = model(eeg, hb,
            torch.as_tensor(data["eeg_mask"][start:end], device=device),
            torch.as_tensor(data["hb_mask"][start:end], device=device))
        for modality, values in outputs.items():
            for key, value in values.items():
                columns[modality].setdefault(key, []).append(value.cpu().numpy())
    result = {modality: {key: np.concatenate(value) for key, value in values.items()}
              for modality, values in columns.items()}
    for values in result.values():
        values["log_amplitude_standardized_mean"] = values["log_amplitude_mean"].copy()
        values["log_amplitude_mean"] = values["log_amplitude_mean"] * amplitude_scale + amplitude_mean
        values["log_amplitude_log_sd"] = values["log_amplitude_log_sd"] + math.log(amplitude_scale)
    return result


def all_code_features(values):
    """Identical effective mean bottleneck: full own observation + source codes."""
    return np.concatenate((values["observation_code"].reshape(len(values["shape_mean"]), -1),
        values["shape_mean"], values["log_amplitude_standardized_mean"][:, None]), axis=1)


def fit_training_probe(features, target, *, partition="train", alpha=1.0):
    if partition != "train":
        raise ValueError("semantic probes fit only training targets")
    features, target = np.asarray(features, np.float64), np.asarray(target, np.float64)
    mean = features.mean(0)
    scale = features.std(0).clip(1e-5)
    standardized = (features - mean) / scale
    target_mean = target.mean(0)
    centered = target - target_mean
    if standardized.shape[1] > standardized.shape[0]:
        coefficients = standardized.T @ np.linalg.solve(
            standardized @ standardized.T + alpha * np.eye(len(standardized)), centered)
    else:
        coefficients = np.linalg.solve(standardized.T @ standardized +
            alpha * np.eye(standardized.shape[1]), standardized.T @ centered)
    return dict(mean=mean, scale=scale, coefficients=coefficients, target_mean=target_mean,
                partition="train", alpha=float(alpha), training_records=len(features))


def apply_probe(probe, features):
    return (features - probe["mean"]) / probe["scale"] @ probe["coefficients"] + probe["target_mean"]


def semantic_predictions(arm, values, probe):
    if arm != "observation_only_capacity_matched":
        return values["shape_mean"], values["log_amplitude_mean"]
    target = apply_probe(probe, all_code_features(values))
    return target[:, :-1], target[:, -1]


def calibrate(arm, predicted, selection, probes):
    """Fit scales on selection only; freeze before evaluation and export."""
    result = {}
    for modality, values in predicted.items():
        shape, amplitude = semantic_predictions(arm, values, probes.get(modality))
        residual = shape - selection["source_shape"]
        amplitude_residual = amplitude - selection["log_amplitude"]
        support = selection["source_mask"].astype(bool)
        if arm == "distribution_randomized_generator":
            ratio = residual / np.exp(values["shape_log_sd"])
            amp_ratio = amplitude_residual / np.exp(values["log_amplitude_log_sd"])
            shape_scale = float(np.sqrt(np.mean(ratio[support] ** 2)).clip(0.1, 10.0))
            amplitude_scale = float(np.sqrt(np.mean(amp_ratio ** 2)).clip(0.1, 10.0))
            kind = "selection_marginal_temperature"
        else:
            shape_scale = float(np.sqrt(np.mean(residual[support] ** 2)).clip(1e-5))
            amplitude_scale = float(np.sqrt(np.mean(amplitude_residual ** 2)).clip(1e-5))
            kind = "selection_constant_residual_scale_not_network_uncertainty"
        result[modality] = dict(shape=shape_scale, amplitude=amplitude_scale, kind=kind,
                                partition="selection", records=len(shape))
    return result


def uncertainty(arm, values, calibration):
    if arm == "distribution_randomized_generator":
        return (np.exp(values["shape_log_sd"]) * calibration["shape"],
                np.exp(values["log_amplitude_log_sd"]) * calibration["amplitude"])
    return (np.full_like(values["shape_mean"], calibration["shape"]),
            np.full_like(values["log_amplitude_mean"], calibration["amplitude"]))


def fit(output, arm, seed, options, identity, device, pilot=False):
    destination = output / ("pilots" if pilot else "fits") / arm / f"seed_{seed}"
    record_path, checkpoint_path = destination / "record.json", destination / "checkpoint.pt"
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous["config_identity"] != identity:
            raise ValueError("fit configuration identity changed")
        if previous["status"] == "complete":
            return previous
    train = load_prepared(output, arm, "train", identity)
    selection = load_prepared(output, arm, "selection", identity)
    if set(train["base_id"]) & set(selection["base_id"]):
        raise ValueError("training and selection base IDs overlap")
    if pilot:
        for dataset in (train, selection):
            keep = np.isin(dataset["base_id"], np.unique(dataset["base_id"])[:16])
            for key in dataset:
                dataset[key] = dataset[key][keep]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(arm, options).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(options["learning_rate"]))
    amplitude_mean = float(train["log_amplitude"].mean())
    amplitude_scale = float(train["log_amplitude"].std().clip(1e-5))
    for modality, branch in (("eeg", model.eeg_branch), ("hb", model.hb_branch)):
        branch.normalizer.fit(torch.as_tensor(train[modality], device=device),
                              torch.as_tensor(train[f"{modality}_mask"], device=device), partition="train")
    start_epoch, best_loss, stale, history, best_state = 0, float("inf"), 0, [], None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["config_identity"] != identity:
            raise ValueError("checkpoint configuration identity changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["next_epoch"]
        best_loss, stale = checkpoint["best_loss"], checkpoint["stale"]
        history, best_state = checkpoint["history"], checkpoint["best_model"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng"):
            # map_location=device also moves RNG byte tensors. CUDA restores
            # require CPU byte tensors even when the model lives on CUDA.
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    epochs = min(3, int(options["epochs"])) if pilot else int(options["epochs"])
    started = time.monotonic()
    completed_examples = 0
    record = dict(status="running", arm=arm, seed=seed, pilot=pilot, config_identity=identity,
        device=str(device), train_records=len(train["eeg"]), selection_records=len(selection["eeg"]),
        train_base_ids=len(np.unique(train["base_id"])), selection_base_ids=len(np.unique(selection["base_id"])),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        mean_bottleneck_per_modality=120 * int(options["observation_dimensions"]) + 120 + 1,
        uncertainty_head_parameters_present_all_arms=True, epochs_budget=epochs,
        amplitude_normalization=dict(mean=amplitude_mean, scale=amplitude_scale, partition="train"),
        metadata=model.token_metadata(), selection_endpoint="frozen_semantic_proper_loss_plus_own_reconstruction")
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            indices = np.random.default_rng(seed + epoch).permutation(len(train["eeg"]))
            running_loss = 0.0
            epoch_start = time.monotonic()
            for start in range(0, len(indices), int(options["batch_size"])):
                take = indices[start:start + int(options["batch_size"])]
                batch = tensor_batch(train, take, device, amplitude_mean, amplitude_scale)
                optimizer.zero_grad(set_to_none=True)
                loss = batch_loss(model, batch, options)["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at epoch {epoch}, batch {start}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                running_loss += float(loss.detach()) * len(take)
                completed_examples += len(take)
            model.eval()
            selection_loss = 0.0
            with torch.no_grad():
                for start in range(0, len(selection["eeg"]), int(options["batch_size"])):
                    take = np.arange(start, min(start + int(options["batch_size"]), len(selection["eeg"])))
                    loss = batch_loss(model, tensor_batch(selection, take, device,
                        amplitude_mean, amplitude_scale), options)["total"]
                    selection_loss += float(loss) * len(take)
            selection_loss /= len(selection["eeg"])
            if not math.isfinite(selection_loss):
                raise FloatingPointError("nonfinite selection endpoint")
            improved = selection_loss < best_loss
            if improved:
                best_loss, stale = selection_loss, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            history.append(dict(epoch=epoch + 1, train_loss=running_loss / len(train["eeg"]),
                selection_loss=selection_loss, elapsed_s=time.monotonic() - epoch_start))
            record.update(completed_epochs=epoch + 1, best_selection_loss=best_loss,
                          elapsed_s=time.monotonic() - started, history=history)
            atomic_checkpoint(checkpoint_path, dict(config_identity=identity,
                model=model.state_dict(), optimizer=optimizer.state_dict(), best_model=best_state,
                next_epoch=epoch + 1, best_loss=best_loss, stale=stale, history=history,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
            atomic_json(record_path, record)
            print(json.dumps(dict(arm=arm, seed=seed, epoch=epoch + 1, selection_loss=selection_loss,
                                  elapsed_s=record["elapsed_s"])), flush=True)
            if stale >= int(options["patience"]):
                break
        if best_state is None:
            raise RuntimeError("no finite selected model")
        model.load_state_dict(best_state)
        selected = predictions(model, selection, device, int(options["batch_size"]), amplitude_mean, amplitude_scale)
        probes = {}
        if model.observation_only:
            training = predictions(model, train, device, int(options["batch_size"]), amplitude_mean, amplitude_scale)
            targets = np.concatenate((train["source_shape"], train["log_amplitude"][:, None]), axis=1)
            probes = {modality: fit_training_probe(all_code_features(values), targets,
                alpha=float(options["ridge_alpha"])) for modality, values in training.items()}
        calibration = calibrate(arm, selected, selection, probes)
        atomic_checkpoint(destination / "selected.pt", dict(config_identity=identity, arm=arm,
            seed=seed, options=options, model=best_state, probes=probes, calibration=calibration,
            amplitude_mean=amplitude_mean, amplitude_scale=amplitude_scale,
            training_base_ids=np.unique(train["base_id"]).tolist(),
            selection_base_ids=np.unique(selection["base_id"]).tolist()))
        record.update(status="complete", calibration=calibration,
            best_epoch=min(history, key=lambda item: item["selection_loss"])["epoch"],
            elapsed_s=time.monotonic() - started,
            examples_per_second=completed_examples / max(time.monotonic() - started, 1e-6),
            selected_checkpoint=str(destination / "selected.pt"))
        atomic_json(record_path, record)
        return record
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      elapsed_s=time.monotonic() - started, history=history)
        atomic_json(record_path, record)
        raise


def selected_model(output, arm, seed, options, identity, device):
    directory = output / "fits" / arm / f"seed_{seed}"
    record = json.loads((directory / "record.json").read_text())
    if record["status"] != "complete" or record["config_identity"] != identity:
        raise ValueError("evaluation requires a complete fit with frozen identity")
    checkpoint = torch.load(directory / "selected.pt", map_location=device, weights_only=False)
    if checkpoint["config_identity"] != identity:
        raise ValueError("selected checkpoint identity differs")
    model = make_model(arm, options).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def _average(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if np.isfinite(values).all() and len(values) else None


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def evaluation_records(arm, predicted, data, checkpoint, model):
    rows, pair_rows, group_rows, risk_rows = [], [], [], []
    for modality, values in predicted.items():
        shape, amp = semantic_predictions(arm, values, checkpoint["probes"].get(modality))
        shape_sd, amp_sd = uncertainty(arm, values, checkpoint["calibration"][modality])
        source_mask = data["source_mask"].astype(bool)
        shape_error = shape - data["source_shape"]
        amp_error = amp - data["log_amplitude"]
        actual = shape * np.exp(np.clip(amp, -20, 20))[:, None]
        truth = data["source_shape"] * np.exp(data["log_amplitude"])[:, None]
        branch = getattr(model, f"{modality}_branch")
        mean, scale = branch.normalizer.mean.cpu().numpy(), branch.normalizer.scale.cpu().numpy()
        visible = data[f"{modality}_mask"]
        target = np.where(visible, (np.where(visible, data[modality], 0) - mean) / scale, 0)
        reconstruction_error = values["reconstruction"] - target
        crps = gaussian_crps(torch.from_numpy(shape), torch.from_numpy(np.log(shape_sd)),
                             torch.from_numpy(data["source_shape"])).numpy()
        nll = gaussian_nll(torch.from_numpy(shape), torch.from_numpy(np.log(shape_sd)),
                           torch.from_numpy(data["source_shape"])).numpy()
        amp_crps = gaussian_crps(torch.from_numpy(amp), torch.from_numpy(np.log(amp_sd)),
                                 torch.from_numpy(data["log_amplitude"])).numpy()
        amp_nll = gaussian_nll(torch.from_numpy(amp), torch.from_numpy(np.log(amp_sd)),
                               torch.from_numpy(data["log_amplitude"])).numpy()
        modality_rows = []
        for index in range(len(shape)):
            support = source_mask[index]
            row = dict(modality=modality, base_id=str(data["base_id"][index]),
                scenario=str(data["scenario"][index]), variant=str(data["variant"][index]),
                shape_rmse=_average(shape_error[index, support] ** 2),
                log_amplitude_squared_error=float(amp_error[index] ** 2),
                source_rmse=_average((actual[index, support] - truth[index, support]) ** 2),
                shape_crps=_average(crps[index, support]), shape_nll=_average(nll[index, support]),
                log_amplitude_crps=float(amp_crps[index]), log_amplitude_nll=float(amp_nll[index]),
                reconstruction_mse=_average(reconstruction_error[index][visible[index]] ** 2),
                reconstruction_native_mse=_average((reconstruction_error[index] * scale)[visible[index]] ** 2),
                source_support=int(support.sum()), observation_support=int(visible[index].sum()),
                uncertainty_kind=checkpoint["calibration"][modality]["kind"],
                uncertainty_score=float(shape_sd[index].mean() + amp_sd[index]),
                finite=bool(np.isfinite(shape[index]).all() and np.isfinite(amp[index])))
            for key in ("shape_rmse", "source_rmse"):
                if row[key] is not None:
                    row[key] = math.sqrt(row[key])
            for level, quantile in NORMAL_QUANTILES.items():
                row[f"shape_coverage_{level}"] = _average(np.abs(shape_error[index, support]) <= quantile * shape_sd[index, support])
                row[f"shape_width_{level}"] = _average(2 * quantile * shape_sd[index, support])
                row[f"log_amplitude_coverage_{level}"] = float(abs(amp_error[index]) <= quantile * amp_sd[index])
                row[f"log_amplitude_width_{level}"] = float(2 * quantile * amp_sd[index])
            rows.append(row)
            modality_rows.append(row)
        pairs = {}
        for index in range(len(shape)):
            pairs.setdefault((str(data["base_id"][index]), str(data["scenario"][index])), {})[str(data["variant"][index])] = index
        for (base_id, scenario), variants in pairs.items():
            if set(variants) != {"control", "intervention"}:
                pair_rows.append(dict(modality=modality, base_id=base_id, scenario=scenario,
                    complete_pair=False, true_change_rms=None, predicted_change_rms=None,
                    change_rmse=None, false_change_rms=None, identical_visible_inputs=None,
                    shape_output_max_difference=None, log_amplitude_output_difference=None,
                    alias_control_confident_amplitude_error=None,
                    alias_intervention_confident_amplitude_error=None,
                    alias_confident_amplitude_error=None))
                continue
            left, right = variants["control"], variants["intervention"]
            true_change, predicted_change = truth[right] - truth[left], actual[right] - actual[left]
            identical = bool(np.array_equal(visible[left], visible[right]) and np.array_equal(
                np.where(visible[left], data[modality][left], 0),
                np.where(visible[right], data[modality][right], 0)))
            unchanged = np.max(np.abs(true_change)) <= 1e-6
            pair_rows.append(dict(modality=modality, base_id=base_id, scenario=scenario,
                complete_pair=True, true_change_rms=float(np.sqrt(np.mean(true_change ** 2))),
                predicted_change_rms=float(np.sqrt(np.mean(predicted_change ** 2))),
                change_rmse=float(np.sqrt(np.mean((predicted_change - true_change) ** 2))),
                false_change_rms=float(np.sqrt(np.mean(predicted_change ** 2))) if unchanged else None,
                identical_visible_inputs=identical,
                shape_output_max_difference=float(np.max(np.abs(shape[right] - shape[left]))),
                log_amplitude_output_difference=float(abs(amp[right] - amp[left])),
                alias_control_confident_amplitude_error=bool(abs(amp_error[left]) > NORMAL_QUANTILES["95"] * amp_sd[left]) if identical else None,
                alias_intervention_confident_amplitude_error=bool(abs(amp_error[right]) > NORMAL_QUANTILES["95"] * amp_sd[right]) if identical else None,
                alias_confident_amplitude_error=bool(
                    abs(amp_error[left]) > NORMAL_QUANTILES["95"] * amp_sd[left] or
                    abs(amp_error[right]) > NORMAL_QUANTILES["95"] * amp_sd[right]) if identical else None))
        metric_keys = [key for key in modality_rows[0] if key not in (
            "modality", "base_id", "scenario", "variant", "uncertainty_kind", "finite")]
        for scenario in np.unique(data["scenario"]):
            selected = [row for row in modality_rows if row["scenario"] == scenario]
            group = dict(modality=modality, scenario=str(scenario), records=len(selected),
                         finite_records=sum(row["finite"] for row in selected))
            for key in metric_keys:
                group[key] = _average([row[key] if row[key] is not None else float("nan") for row in selected])
            group_rows.append(group)
            ordered = sorted(selected, key=lambda row: row["uncertainty_score"])
            for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
                cutoff = ordered[max(1, math.ceil(fraction * len(ordered))) - 1]["uncertainty_score"]
                # Constant post-hoc point scales provide no rejection ranking.
                # Keep all ties instead of manufacturing a curve by base-ID order.
                retained = [row for row in ordered if row["uncertainty_score"] <= cutoff]
                risk_rows.append(dict(modality=modality, scenario=str(scenario),
                    requested_retained_fraction=fraction,
                    retained_fraction=len(retained) / len(ordered),
                    uncertainty_threshold=cutoff,
                    retained_records=len(retained), total_records=len(ordered),
                    shape_rmse=_average([row["shape_rmse"] for row in retained]),
                    source_rmse=_average([row["source_rmse"] for row in retained]),
                    score_kind=checkpoint["calibration"][modality]["kind"],
                    excluded_records_still_in_full_evaluation=len(ordered) - len(retained)))
    return rows, pair_rows, group_rows, risk_rows


def evaluate(output, arm, seed, options, identity, device):
    destination = output / "evaluation" / arm / f"seed_{seed}"
    record_path = destination / "record.json"
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous["config_identity"] != identity:
            raise ValueError("evaluation identity changed")
        if previous["status"] == "complete":
            return previous
    started = time.monotonic()
    model, checkpoint = selected_model(output, arm, seed, options, identity, device)
    # All arms face the same randomized evaluation, including the restricted arm.
    data = load_prepared(output, "point_randomized_generator", "evaluation", identity)
    if set(data["base_id"]) & set(checkpoint["training_base_ids"] + checkpoint["selection_base_ids"]):
        raise ValueError("evaluation overlaps fit base IDs")
    predicted = predictions(model, data, device, int(options["batch_size"]),
                            checkpoint["amplitude_mean"], checkpoint["amplitude_scale"])
    rows, pairs, groups, risk = evaluation_records(arm, predicted, data, checkpoint, model)
    _write_csv(destination / "records.csv", rows)
    _write_csv(destination / "paired_changes.csv", pairs)
    _write_csv(destination / "scenario_metrics.csv", groups)
    _write_csv(destination / "selective_risk.csv", risk)
    record = dict(status="complete", config_identity=identity, arm=arm, seed=seed,
        expected_records=len(data["eeg"]) * 2, completed_records=len(rows),
        finite_records=sum(row["finite"] for row in rows),
        expected_pairs=len(data["eeg"]), completed_pairs=sum(row["complete_pair"] for row in pairs),
        evaluation_family="randomized_same_all_arms", elapsed_s=time.monotonic() - started,
        calibration=checkpoint["calibration"], metrics=groups,
        source_status="known_synthetic_generator_truth_no_measured_physiology_claim")
    atomic_json(record_path, record)
    return record


def export(output, arm, seed, options, identity, device, prepared=None):
    model, checkpoint = selected_model(output, arm, seed, options, identity, device)
    data = load_prepared(output, "point_randomized_generator", "evaluation", identity, path=prepared)
    predicted = predictions(model, data, device, int(options["batch_size"]),
                            checkpoint["amplitude_mean"], checkpoint["amplitude_scale"])
    arrays = {key: data[key] for key in ("base_id", "scenario", "variant") if key in data}
    for modality, values in predicted.items():
        shape, amp = semantic_predictions(arm, values, checkpoint["probes"].get(modality))
        shape_sd, amp_sd = uncertainty(arm, values, checkpoint["calibration"][modality])
        tensor_values = {key: torch.as_tensor(value) for key, value in values.items()}
        tensor_values["shape_mean"] = torch.as_tensor(shape)
        tensor_values["shape_log_sd"] = torch.as_tensor(np.log(shape_sd))
        tensor_values["log_amplitude_mean"] = torch.as_tensor(amp)
        tensor_values["log_amplitude_log_sd"] = torch.as_tensor(np.log(amp_sd))
        tokens = model.export_tokens(tensor_values)
        for key, value in tokens.items():
            if value is not None:
                arrays[f"{modality}_{key}"] = value.numpy()
        arrays[f"{modality}_continuous_shape_sd_calibrated"] = shape_sd
        arrays[f"{modality}_log_amplitude_sd_calibrated"] = amp_sd
        arrays[f"{modality}_reconstruction"] = values["reconstruction"]
        arrays[f"{modality}_all_mean_codes"] = all_code_features(values)
    label = Path(prepared).stem if prepared else "synthetic_evaluation"
    destination = output / "exports" / arm / f"seed_{seed}" / f"{label}.npz"
    atomic_npz(destination, **arrays)
    atomic_json(destination.with_suffix(".json"), dict(status="complete", config_identity=identity,
        arm=arm, seed=seed, records=len(data["eeg"]), source_path=str(prepared) if prepared else None,
        metadata=model.token_metadata(), calibration=checkpoint["calibration"],
        input_normalization="synthetic_training_only_unchanged_for_external_inputs",
        exported_log_amplitude_coordinate="natural_log_generator_reference_RMS_in_native_source_units",
        reconstruction_coordinate="synthetic_training_channel_standardized",
        external_export_does_not_qualify_measured_physiology=prepared is not None))


def load_public_feature_batches(feature_root):
    """Read the existing public feature export, never an original dataset.

    ``hb_target`` is available only to probe fitting/scoring. The encoding view
    is assembled separately from its prefix-masked ``hb`` and its own mask.
    """
    feature_root = Path(feature_root)
    batches, contexts = {}, {}
    for split in ("train", "selection", "evaluation"):
        path = feature_root / f"{split}.npz"
        manifest = json.loads(path.with_suffix(".json").read_text())
        if manifest["schema"] != "semantic_response_discovery_feature_batch_v2" or manifest["split"] != split:
            raise ValueError("unexpected public feature export schema/split")
        with np.load(path, allow_pickle=False) as handle:
            data = {key: handle[key] for key in handle.files}
        with np.load(feature_root / f"{split}__full_spatial_context.npz", allow_pickle=False) as handle:
            spatial = {key: handle[key] for key in handle.files}
        required = ("eeg", "hb", "hb_target", "eeg_mask", "hb_mask", "base_id", "dataset", "subject", "region", "native_id", "sd_hb", "hb_factor")
        if any(key not in data for key in required) or len(data["eeg"]) != manifest["rows"]:
            raise ValueError("incomplete public feature arrays or row denominator")
        if (data["eeg"].shape[1:] != (120, 30) or data["hb"].shape[1:] != (120, 2)
                or data["hb_target"].shape != data["hb"].shape or data["sd_hb"].shape != (len(data["eeg"]), 2)):
            raise ValueError("public tensor contract mismatch")
        if len(set(data["base_id"])) != len(data["base_id"]):
            raise ValueError("public regional IDs must be unique")
        if (data["hb_mask"][:, 40:].any() or not np.isfinite(data["hb_target"]).all()
                or not np.isfinite(data["sd_hb"]).all() or np.any(data["sd_hb"] <= 0)):
            raise ValueError("public Hb input boundary or training SD is invalid")
        if data["hb_factor"].shape != (len(data["eeg"]),) or not np.isfinite(data["hb_factor"]).all() or np.any(data["hb_factor"] <= 0):
            raise ValueError("public training gain must be finite and positive")
        if not np.array_equal(data["base_id"], spatial["base_id"]):
            raise ValueError("spatial context rows differ from primary feature identities")
        if len(manifest["records"]) != len(data["base_id"]):
            raise ValueError("public identity sidecar row mismatch")
        for index, record in enumerate(manifest["records"]):
            if (str(record["ref"]["id"]) != str(data["base_id"][index])
                    or str(record["native_id"]) != str(data["native_id"][index])):
                raise ValueError("public native/regional identity disagrees with sidecar")
        batches[split], contexts[split] = data, spatial
    subject_sets = {split: set(zip(data["dataset"], data["subject"])) for split, data in batches.items()}
    if any(subject_sets[left] & subject_sets[right] for left, right in (
        ("train", "selection"), ("train", "evaluation"), ("selection", "evaluation"))):
        raise ValueError("public feature splits overlap subjects")
    return batches, contexts


def public_encoding_view(data, prefix_steps):
    """Only own observations enter encoders; full target is never copied here."""
    if prefix_steps not in (0, 40):
        raise ValueError("registered public Hb visibility is 0 or 40 steps")
    visible = np.asarray(data["hb_mask"], bool).copy()
    visible[:, prefix_steps:] = False
    hb = np.where(visible, data["hb"], np.nan).astype(np.float32)
    return dict(eeg=data["eeg"], eeg_mask=data["eeg_mask"], hb=hb, hb_mask=visible)


def public_raw_context(data, spatial, prefix_steps):
    """Shared raw EEG + other regional Hb + own visible Hb prefix.

    The full target regional Hb is excluded by its regional identity before
    constructing each row. Other regions are joined only by the exact native
    window identity and ordered by their region names, never by their values.
    """
    lookup = {}
    for index in range(len(spatial["hb"])):
        key = (str(spatial["dataset"][index]), str(spatial["subject"][index]),
               str(spatial["native_id"][index]), str(spatial["region"][index]))
        if key in lookup:
            raise ValueError("duplicate spatial native/region identity")
        lookup[key] = index
    regions = sorted(set(str(value) for value in data["region"]))
    if len(regions) != 3:
        raise ValueError("the registered regional context has exactly three regions")
    features = []
    donors = []
    view = public_encoding_view(data, prefix_steps)
    for index in range(len(data["eeg"])):
        own_region = str(data["region"][index])
        key = (str(data["dataset"][index]), str(data["subject"][index]), str(data["native_id"][index]))
        other_indices = [lookup[(*key, region)] for region in regions if region != own_region]
        if any(str(spatial["base_id"][donor]) == str(data["base_id"][index]) for donor in other_indices):
            raise ValueError("target regional Hb reached spatial context")
        other_hb = spatial["hb"][other_indices].reshape(-1)
        eeg = np.where(data["eeg_mask"][index], data["eeg"][index], 0).reshape(-1)
        own_prefix = np.where(view["hb_mask"][index, :prefix_steps],
                              view["hb"][index, :prefix_steps], 0).reshape(-1)
        row = np.concatenate((eeg, other_hb, own_prefix))
        if not np.isfinite(row).all():
            raise ValueError("public context contains nonfinite visible values")
        features.append(row)
        donors.append([str(spatial["base_id"][donor]) for donor in other_indices])
    return np.asarray(features), donors


def public_code_features(values, prefix_steps, *, include_source, include_uncertainty=False):
    modalities = ("eeg", "hb") if prefix_steps else ("eeg",)
    features = []
    for modality in modalities:
        output = values[modality]
        features.append(output["observation_code"].reshape(len(output["shape_mean"]), -1))
        if include_source:
            features.extend((output["shape_mean"], output["log_amplitude_mean"][:, None]))
        if include_uncertainty:
            features.extend((np.exp(output["shape_log_sd"]),
                             np.exp(output["log_amplitude_log_sd"])[:, None]))
    return np.concatenate(features, axis=1)


def _subject_dataset_average(rows, metric):
    if not rows or any(row.get(metric) is None or not math.isfinite(float(row[metric])) for row in rows):
        return None
    subjects = {}
    for row in rows:
        subjects.setdefault((row["dataset"], row["subject"]), []).append(float(row[metric]))
    datasets = {}
    for (dataset, _), values in subjects.items():
        datasets.setdefault(dataset, []).append(float(np.mean(values)))
    return float(np.mean([np.mean(values) for values in datasets.values()]))


def _selection_probe_score(prediction, target, subjects):
    per_record = np.sqrt(np.mean((prediction - target) ** 2, axis=1))
    return float(np.mean([np.mean(per_record[subjects == subject]) for subject in np.unique(subjects)]))


def fit_public_probe(training_x, training_y, selection_x, selection_y, selection_subjects,
                     candidates, *, partition="train", minimum_sd_fraction=0.05):
    """Public probe weights fit train; alpha and predictive SD use selection.

    Public alpha penalizes the *mean* training squared-error objective, matching
    the existing observation ridge convention. Synthetic decoder probes retain
    their earlier fixed sum-objective alpha and never select on evaluation.
    """
    if partition != "train":
        raise ValueError("public probe weights may only fit training targets")
    rankings, models = [], {}
    for alpha in candidates:
        model = fit_training_probe(training_x, training_y, alpha=float(alpha) * len(training_x))
        prediction = apply_probe(model, selection_x)
        score = _selection_probe_score(prediction, selection_y, np.asarray(selection_subjects))
        if not math.isfinite(score):
            raise FloatingPointError("nonfinite public selection endpoint")
        models[float(alpha)] = model
        rankings.append(dict(alpha=float(alpha), selection_subject_equal_NRMSE=score))
    rankings.sort(key=lambda row: (row["selection_subject_equal_NRMSE"], -row["alpha"]))
    chosen = models[rankings[0]["alpha"]]
    residual = (apply_probe(chosen, selection_x) - selection_y).reshape(len(selection_y), -1, 2)
    # Equal subject weights in the residual second moment, matching the endpoint.
    squared_by_subject = [np.mean(residual[np.asarray(selection_subjects) == subject] ** 2, axis=(0, 1))
                          for subject in np.unique(selection_subjects)]
    raw_sd = np.sqrt(np.mean(squared_by_subject, axis=0))
    calibrated_sd = np.maximum(raw_sd, float(minimum_sd_fraction))
    return chosen, dict(alpha=rankings[0]["alpha"], rankings=rankings,
        predictive_sd_training_units=calibrated_sd.tolist(), selection_residual_sd_before_floor=raw_sd.tolist(),
        floor_fraction_of_training_SD=float(minimum_sd_fraction),
        floor_triggered=(raw_sd < minimum_sd_fraction).tolist(),
        fitting_partition="train", alpha_selection_partition="selection",
        uncertainty_calibration_partition="selection",
        density="Gaussian_predictive_residual_marginals_not_physiological_posterior")


def public_probe_rows(prediction, target, data, indices, predictive_sd, prefix_steps, method):
    prediction = np.asarray(prediction).reshape(len(indices), -1, 2)
    target = np.asarray(target).reshape(prediction.shape)
    error = prediction - target
    sd = np.broadcast_to(np.asarray(predictive_sd)[None, None, :], prediction.shape)
    crps = gaussian_crps(torch.as_tensor(prediction), torch.as_tensor(np.log(sd)), torch.as_tensor(target)).numpy()
    nll = gaussian_nll(torch.as_tensor(prediction), torch.as_tensor(np.log(sd)), torch.as_tensor(target)).numpy()
    rows = []
    for local, index in enumerate(indices):
        training_gain = float(data["hb_factor"][index]) if "hb_factor" in data else 1.0
        native_sd = data["sd_hb"][index] / training_gain
        row = dict(id=str(data["base_id"][index]), native_id=str(data["native_id"][index]),
            dataset=str(data["dataset"][index]), subject=str(data["subject"][index]),
            region=str(data["region"][index]), prefix_steps=prefix_steps, method=method, success=True,
            NRMSE=float(np.sqrt(np.mean(error[local] ** 2))),
            CRPS_training_SD=float(np.mean(crps[local])),
            Gaussian_NLL_training_SD=float(np.mean(nll[local])),
            RMSE_parent_prepared=float(np.sqrt(np.mean((error[local] * native_sd) ** 2))),
            CRPS_parent_prepared=float(np.mean(crps[local] * native_sd)),
            Gaussian_NLL_parent_prepared=float(np.mean(nll[local] + np.log(native_sd))),
            Hb_training_gain=training_gain,
            score_points=int(np.prod(prediction.shape[1:])))
        for component, name in enumerate(("HbO", "HbR")):
            row[f"{name}_NRMSE"] = float(np.sqrt(np.mean(error[local, :, component] ** 2)))
            row[f"{name}_CRPS_training_SD"] = float(np.mean(crps[local, :, component]))
            row[f"{name}_CRPS_parent_prepared"] = float(np.mean(crps[local, :, component]) * native_sd[component])
            row[f"{name}_Gaussian_NLL_training_SD"] = float(np.mean(nll[local, :, component]))
            row[f"{name}_Gaussian_NLL_parent_prepared"] = float(np.mean(nll[local, :, component]) + np.log(native_sd[component]))
            for level, quantile in NORMAL_QUANTILES.items():
                row[f"{name}_coverage_{level}"] = float(np.mean(np.abs(error[local, :, component]) <= quantile * sd[local, :, component]))
                row[f"{name}_width_training_SD_{level}"] = float(2 * quantile * predictive_sd[component])
                row[f"{name}_width_parent_prepared_{level}"] = float(2 * quantile * predictive_sd[component] * native_sd[component])
        rows.append(row)
    return rows


def _probe_contrasts(rows, prefixes):
    contrasts = []
    for prefix in prefixes:
        subset = [row for row in rows if row["prefix_steps"] == prefix]
        lookup = {(row["method"], row["id"]): row for row in subset}
        ids = sorted(set(row["id"] for row in subset))
        for reference in ("context_plus_O", "context_plus_capacity_matched_observation_only", "context_raw"):
            pairs = []
            for identity in ids:
                left = lookup.get(("context_plus_O_plus_S", identity))
                right = lookup.get((reference, identity))
                if left and right and left["success"] and right["success"]:
                    pairs.append(dict(dataset=left["dataset"], subject=left["subject"],
                        NRMSE=left["NRMSE"] - right["NRMSE"],
                        CRPS_training_SD=left["CRPS_training_SD"] - right["CRPS_training_SD"]))
            contrasts.append(dict(prefix_steps=prefix, contrast=f"O_plus_S_minus_{reference}",
                planned_windows=len(ids), common_success_windows=len(pairs),
                delta_NRMSE=_subject_dataset_average(pairs, "NRMSE"),
                delta_CRPS_training_SD=_subject_dataset_average(pairs, "CRPS_training_SD"),
                negative_delta_favors_semantic_plus_observation=True))
    return contrasts


def probe(output, arm, seed, options, identity, device, feature_root, contract):
    """Frozen public conditional probe, with all data boundaries fixed before fit.

    Primary context is identical raw local EEG, the two other regional full Hb
    windows, and own Hb prefix. The three code comparisons add O, O+S, or all
    mean codes from the independent capacity-matched observation-only arm.
    Target Hb suffix never enters any encoder or predictor input. Full-Hb
    synthetic source recovery remains a different endpoint.
    """
    destination = output / "public_probes" / arm / f"seed_{seed}"
    record_path = destination / "record.json"
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous["config_identity"] != identity or previous["feature_root"] != str(Path(feature_root).resolve()):
            raise ValueError("public probe frozen identity or feature source changed")
        if previous["status"] == "complete":
            return previous
    started = time.monotonic()
    batches, spatial = load_public_feature_batches(feature_root)
    prefixes = contract.get("measured", {}).get("prefix_steps", [40, 0])
    if any(prefix not in (0, 40) for prefix in prefixes):
        raise ValueError("public probe prefix exceeds the registered visibility")
    score_start = int(contract.get("measured", {}).get("score_start", 80))
    score_stop = int(contract.get("measured", {}).get("score_stop", 120))
    if not 40 <= score_start < score_stop <= 120:
        raise ValueError("public score support must be strictly after the allowed prefix")
    candidates = contract.get("baselines", {}).get("ridge_candidates", [0.01, 0.1, 1.0, 10.0])
    minimum_sd = contract.get("uncertainty", {}).get("minimum_training_sd_fraction", 0.05)
    methods = ["context_raw", "context_plus_O", "context_plus_O_plus_S",
               "context_plus_capacity_matched_observation_only"]
    if arm == "distribution_randomized_generator":
        methods.append("context_plus_O_plus_S_plus_uncertainty_secondary")
    record = dict(status="running", config_identity=identity, arm=arm, seed=seed,
        feature_root=str(Path(feature_root).resolve()), prefixes=prefixes,
        score_support=[score_start, score_stop], methods=methods,
        planned_rows=len(batches["evaluation"]["eeg"]) * len(methods) * len(prefixes),
        encoder_fit="synthetic_training_only_frozen_no_public_gradient",
        public_probe_fit="train_only_weights_selection_only_alpha_and_predictive_SD",
        common_context="raw_local_EEG_plus_other_region_full_Hb_plus_own_visible_Hb_prefix",
        target_Hb_boundary="own_prefix_only_never_target_suffix; no_target_Hb_codes_at_zero_prefix",
        primary_capacity="all_own_mean_codes_same_dimension_supervised_vs_independent_observation_only",
        uncertainty_features="secondary_only_distribution_arm",
        primary_score_coordinate="component_training_SD_normalized",
        restored_score_coordinate="parent_prepared_Hb_units_after_inverse_new_training_gain_not_raw_native_waveform",
        aggregation="windows_and_regions_within_subject_then_subject_equal_within_dataset_then_dataset_equal",
        physiological_semantics="unqualified_conditional_processed_observation_completion")
    atomic_json(record_path, record)
    all_rows, selections = [], []
    try:
        model, checkpoint = selected_model(output, arm, seed, options, identity, device)
        capacity_model, capacity_checkpoint = selected_model(output, ARMS[0], seed, options, identity, device)
        for prefix in prefixes:
            context, features, donor_records = {}, {method: {} for method in methods}, {}
            for split, data in batches.items():
                context[split], donors = public_raw_context(data, spatial[split], prefix)
                donor_records[split] = {str(identity): values for identity, values in zip(data["base_id"], donors)}
                encoding = public_encoding_view(data, prefix)
                own = predictions(model, encoding, device, int(options["batch_size"]),
                    checkpoint["amplitude_mean"], checkpoint["amplitude_scale"])
                capacity = own if arm == ARMS[0] else predictions(capacity_model, encoding, device,
                    int(options["batch_size"]), capacity_checkpoint["amplitude_mean"], capacity_checkpoint["amplitude_scale"])
                features["context_raw"][split] = context[split]
                features["context_plus_O"][split] = np.concatenate((context[split],
                    public_code_features(own, prefix, include_source=False)), axis=1)
                features["context_plus_O_plus_S"][split] = np.concatenate((context[split],
                    public_code_features(own, prefix, include_source=True)), axis=1)
                features["context_plus_capacity_matched_observation_only"][split] = np.concatenate((context[split],
                    public_code_features(capacity, prefix, include_source=True)), axis=1)
                if len(methods) == 5:
                    features[methods[-1]][split] = np.concatenate((context[split],
                        public_code_features(own, prefix, include_source=True, include_uncertainty=True)), axis=1)
            atomic_json(destination / f"spatial_donors_prefix_{prefix}.json", donor_records)
            groups = sorted(set(zip(batches["train"]["dataset"], batches["train"]["region"])))
            for dataset, region in groups:
                group_indices = {split: np.flatnonzero((data["dataset"] == dataset) & (data["region"] == region))
                                 for split, data in batches.items()}
                if any(len(indices) == 0 for indices in group_indices.values()):
                    raise ValueError("public dataset/region lacks a declared train/selection/evaluation split")
                group_sd = batches["train"]["sd_hb"][group_indices["train"][0]]
                for split, indices in group_indices.items():
                    if not np.allclose(batches[split]["sd_hb"][indices], group_sd, rtol=1e-7, atol=1e-12):
                        raise ValueError("public target SD must be the same training coordinate in every split")
                targets = {split: (data["hb_target"][group_indices[split], score_start:score_stop] /
                                   data["sd_hb"][group_indices[split], None, :]).reshape(len(group_indices[split]), -1)
                           for split, data in batches.items()}
                for method in methods:
                    try:
                        selected, calibration = fit_public_probe(
                            features[method]["train"][group_indices["train"]], targets["train"],
                            features[method]["selection"][group_indices["selection"]], targets["selection"],
                            batches["selection"]["subject"][group_indices["selection"]], candidates,
                            minimum_sd_fraction=minimum_sd)
                        prediction = apply_probe(selected, features[method]["evaluation"][group_indices["evaluation"]])
                        if not np.isfinite(prediction).all():
                            raise FloatingPointError("nonfinite public frozen prediction")
                        rows = public_probe_rows(prediction, targets["evaluation"], batches["evaluation"],
                            group_indices["evaluation"], calibration["predictive_sd_training_units"], prefix, method)
                        all_rows.extend(rows)
                        selections.append(dict(dataset=str(dataset), region=str(region), prefix_steps=prefix,
                            method=method, status="complete", feature_dimensions=features[method]["train"].shape[1],
                            training_rows=len(group_indices["train"]), selection_rows=len(group_indices["selection"]),
                            evaluation_rows=len(group_indices["evaluation"]), training_SD=group_sd.tolist(),
                            **calibration))
                    except Exception as exc:
                        selections.append(dict(dataset=str(dataset), region=str(region), prefix_steps=prefix,
                            method=method, status="failed", type=type(exc).__name__, error=str(exc)))
                        for index in group_indices["evaluation"]:
                            all_rows.append(dict(id=str(batches["evaluation"]["base_id"][index]),
                                native_id=str(batches["evaluation"]["native_id"][index]), dataset=str(dataset),
                                subject=str(batches["evaluation"]["subject"][index]), region=str(region),
                                prefix_steps=prefix, method=method, success=False, NRMSE=None,
                                CRPS_training_SD=None, Gaussian_NLL_training_SD=None, RMSE_parent_prepared=None,
                                CRPS_parent_prepared=None, Gaussian_NLL_parent_prepared=None, score_points=2 * (score_stop - score_start)))
            atomic_json(destination / "selection.json", selections)
            record.update(completed_rows=len(all_rows), failed_rows=sum(not row["success"] for row in all_rows),
                          elapsed_s=time.monotonic() - started)
            atomic_json(record_path, record)
        # A failed method remains in the complete denominator and the CSV.
        fields = list(dict.fromkeys(key for row in all_rows for key in row))
        _write_csv(destination / "records.csv", [{key: row.get(key) for key in fields} for row in all_rows])
        summaries = []
        for prefix in prefixes:
            for method in methods:
                rows = [row for row in all_rows if row["prefix_steps"] == prefix and row["method"] == method]
                summaries.append(dict(prefix_steps=prefix, method=method, planned_windows=len(rows),
                    successful_windows=sum(row["success"] for row in rows),
                    NRMSE=_subject_dataset_average(rows, "NRMSE"),
                    CRPS_training_SD=_subject_dataset_average(rows, "CRPS_training_SD"),
                    Gaussian_NLL_training_SD=_subject_dataset_average(rows, "Gaussian_NLL_training_SD")))
        record.update(status="complete", completed_rows=len(all_rows),
            failed_rows=sum(not row["success"] for row in all_rows), summaries=summaries,
            contrasts=_probe_contrasts(all_rows, prefixes), elapsed_s=time.monotonic() - started,
            selections=selections)
        if len(all_rows) != record["planned_rows"]:
            raise RuntimeError("public probe did not account for every planned regional window")
        atomic_json(record_path, record)
        return record
    except Exception as exc:
        record.update(status="failed", completed_rows=len(all_rows), error_type=type(exc).__name__,
                      error=str(exc), elapsed_s=time.monotonic() - started)
        atomic_json(record_path, record)
        raise


def scenario_variant_tables(output, options):
    """Summarize retained evaluation rows without mixing paired variants.

    Optimization seeds reuse the same evaluation bases. Their mean/range is a
    training-repeat description, never 3 * 256 independent scientific cases.
    Existing completed per-fit evidence files are only read, never rewritten.
    """
    per_seed, grouped_seeds = [], {}
    for arm in ARMS:
        for seed in options["seeds"]:
            path = output / "evaluation" / arm / f"seed_{seed}" / "records.csv"
            if not path.exists():
                continue
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                numeric = [key for key in reader.fieldnames if key not in (
                    "modality", "base_id", "scenario", "variant", "uncertainty_kind", "finite")]
                groups = {}
                for row in reader:
                    groups.setdefault((row["modality"], row["scenario"], row["variant"]), []).append(row)
            for (modality, scenario, variant), records in sorted(groups.items()):
                bases = set(row["base_id"] for row in records)
                row = dict(arm=arm, optimization_seed=seed, modality=modality, scenario=scenario,
                    variant=variant, records=len(records), unique_evaluation_base_ids=len(bases),
                    finite_records=sum(record["finite"].lower() == "true" for record in records))
                for key in numeric:
                    row[key] = _average([float(record[key]) if record[key] else float("nan") for record in records])
                per_seed.append(row)
                grouped_seeds.setdefault((arm, modality, scenario, variant), []).append((row, bases))
    if per_seed:
        _write_csv(output / "scenario_variant_metrics.csv", per_seed)
    repeat_summary = []
    for (arm, modality, scenario, variant), repetitions in sorted(grouped_seeds.items()):
        row = dict(arm=arm, modality=modality, scenario=scenario, variant=variant,
            optimization_seed_count=len(repetitions),
            optimization_seeds="|".join(str(item[0]["optimization_seed"]) for item in repetitions),
            unique_evaluation_base_ids=len(set.union(*(item[1] for item in repetitions))),
            cases_per_seed_min=min(item[0]["unique_evaluation_base_ids"] for item in repetitions),
            cases_per_seed_max=max(item[0]["unique_evaluation_base_ids"] for item in repetitions),
            independence_unit="shared_base_identity_not_optimization_seed")
        for key in per_seed[0]:
            if key in ("arm", "optimization_seed", "modality", "scenario", "variant", "records",
                       "unique_evaluation_base_ids", "finite_records"):
                continue
            values = [item[0][key] for item in repetitions]
            valid = all(value is not None and math.isfinite(value) for value in values)
            row[f"{key}_seed_mean"] = float(np.mean(values)) if valid else None
            row[f"{key}_seed_min"] = float(np.min(values)) if valid else None
            row[f"{key}_seed_max"] = float(np.max(values)) if valid else None
        repeat_summary.append(row)
    if repeat_summary:
        _write_csv(output / "scenario_variant_seed_summary.csv", repeat_summary)
    return dict(per_seed_table="scenario_variant_metrics.csv", seed_repeat_table="scenario_variant_seed_summary.csv",
        per_seed_groups=len(per_seed), seed_repeat_groups=len(repeat_summary),
        aggregation="control_and_intervention_separate; metric_mean_within_base_set; optimization_seed_mean_and_range",
        independence="base_IDs_shared_across_seeds_and_counterfactual_variants")


def summarize(output, options, identity, *, require_public_probes=True):
    fits, evaluations, public_probes, missing = [], [], [], []
    stage_collections = (("fits", fits), ("evaluation", evaluations), ("public_probes", public_probes))
    for arm in ARMS:
        for seed in options["seeds"]:
            for stage, collection in stage_collections:
                path = output / stage / arm / f"seed_{seed}" / "record.json"
                if path.exists():
                    record = json.loads(path.read_text())
                    if record["config_identity"] != identity:
                        raise ValueError("summary found another configuration identity")
                    collection.append(record)
                elif stage != "public_probes" or require_public_probes:
                    missing.append(dict(stage=stage, arm=arm, seed=seed))
    expected = len(ARMS) * len(options["seeds"])
    public_complete = (len(public_probes) == expected and all(
        item["status"] == "complete" and item.get("completed_rows") == item.get("planned_rows")
        for item in public_probes))
    synthetic_complete = (len(fits) == len(evaluations) == expected and
        all(item["status"] == "complete" for item in fits + evaluations) and
        all(item.get("completed_records") == item.get("expected_records") and
            item.get("completed_pairs") == item.get("expected_pairs") for item in evaluations))
    stage_statuses = {}
    for stage, records in stage_collections:
        required = stage != "public_probes" or require_public_probes
        stage_statuses[stage] = dict(expected=expected if required else 0, found=len(records),
            complete=sum(item["status"] == "complete" for item in records),
            fatal_failed=sum(item["status"] == "failed" for item in records),
            missing=expected - len(records) if required else 0,
            required_for_this_summary=required)
    atomic_json(output / "summary.json", dict(schema="semantic_response_tokenizer_suite_v1",
        config_identity=identity, expected_fits=expected, fits=fits,
        evaluations=evaluations, public_probes=public_probes, missing=missing,
        scope="synthetic_and_registered_public_probes" if require_public_probes else "synthetic_training_and_recovery_only",
        public_probe_status="complete" if public_complete else "incomplete",
        synthetic_status="complete" if synthetic_complete else "incomplete",
        expected_public_probes=expected if require_public_probes else 0,
        public_probe_planned_rows=sum(item.get("planned_rows", 0) for item in public_probes),
        public_probe_completed_rows=sum(item.get("completed_rows", 0) for item in public_probes),
        public_probe_failed_rows=sum(item.get("failed_rows", 0) for item in public_probes),
        synthetic_expected_records=sum(item.get("expected_records", 0) for item in evaluations),
        synthetic_completed_records=sum(item.get("completed_records", 0) for item in evaluations),
        synthetic_nonfinite_records=sum(item.get("completed_records", 0) - item.get("finite_records", 0) for item in evaluations),
        stage_statuses=stage_statuses,
        scenario_variant_metrics=scenario_variant_tables(output, options),
        status="complete" if synthetic_complete and (public_complete or not require_public_probes) and not missing else "incomplete",
        physiological_semantic_qualification="not_established_by_synthetic_fit_or_reconstruction"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "pilot", "train", "evaluate", "export", "probe", "summarize"))
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arm", choices=ARMS + tuple(ARM_ALIASES))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prepared", type=Path, help="explicit external EEG/Hb NPZ for frozen export only")
    parser.add_argument("--feature-root", type=Path, help="owning public feature export directory for frozen probe")
    parser.add_argument("--workers", type=int, default=1, help="bounded prepare generation workers; size from measured throughput")
    parser.add_argument("--family", choices=("restricted", "randomized"), help="prepare one family")
    parser.add_argument("--partition", choices=("train", "selection", "evaluation"), help="prepare one partition")
    args = parser.parse_args(argv)
    args.arm = ARM_ALIASES.get(args.arm, args.arm)
    if not args.project_root.is_dir():
        raise ValueError("project root must be an existing directory")
    contract, options, identity = load_contract(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    resolved = args.output / "resolved_tokenizer_config.json"
    if resolved.exists() and json.loads(resolved.read_text())["config_identity"] != identity:
        raise ValueError("output namespace already belongs to another contract")
    if not resolved.exists():
        atomic_json(resolved, dict(config_identity=identity, contract_path=str(args.config.resolve()),
            project_root=str(args.project_root.resolve()), implementation_root=str(PROJECT_ROOT),
            options=options, arms=list(ARMS), finite_fits=len(ARMS) * len(options["seeds"])))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    if args.seed is not None and args.seed not in options["seeds"]:
        raise ValueError("seed is outside the frozen optimization seed list")
    if args.prepared is not None and args.stage != "export":
        raise ValueError("external prepared arrays are only allowed for frozen export")
    if (args.feature_root is None) == (args.stage == "probe"):
        raise ValueError("probe stage requires --feature-root; other stages do not accept it")
    if (args.family or args.partition) and args.stage != "prepare":
        raise ValueError("family/partition selection is only valid for preparation")
    if args.family == "restricted" and args.partition == "evaluation":
        raise ValueError("evaluation is common randomized data for every arm")
    if args.stage == "prepare":
        prepare(args.output, options, identity, args.workers, args.family, args.partition)
    elif args.stage == "summarize":
        summarize(args.output, options, identity, require_public_probes="data_boundary" in contract)
    else:
        for arm in ([args.arm] if args.arm else ARMS):
            for seed in ([args.seed] if args.seed is not None else options["seeds"]):
                if args.stage in ("train", "pilot"):
                    fit(args.output, arm, seed, options, identity, device, pilot=args.stage == "pilot")
                elif args.stage == "evaluate":
                    evaluate(args.output, arm, seed, options, identity, device)
                elif args.stage == "export":
                    export(args.output, arm, seed, options, identity, device, args.prepared)
                elif args.stage == "probe":
                    probe(args.output, arm, seed, options, identity, device, args.feature_root, contract)


if __name__ == "__main__":
    main()
