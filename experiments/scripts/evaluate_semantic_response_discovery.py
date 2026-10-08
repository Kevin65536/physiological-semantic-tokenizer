#!/usr/bin/env python3
"""Exploratory fixed-driver response discovery on exact public parent features.

EEG states are inferred once without Hb. Hb completion consumes an explicit
prefix, while train-only coordinates and subject-disjoint selection objects are
frozen before evaluation. Inputs were processed over full windows: none of the
endpoints in this runner is a native-signal causal forecast.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.optimize import linear_sum_assignment
import yaml

from src.inference.observation_baselines import native_feature_operators
from src.inference.shared_driver_attribution import (
    equal_capacity_hb_basis, fit_standardized_ridge, predict_standardized_ridge,
)
from src.inference.shared_driver_modes import spectral_loadings, temporal_mode_basis
from src.inference.t3a_balloon_robust_ssm import (
    BalloonFixedParameters, BalloonFreeParameters, BalloonParameters,
)

DEFAULT_CONFIG = CODE_ROOT / "experiments/configs/physiology_semantic_tokenizer/semantic_response_discovery_v1.yaml"
SOURCE_RUN = "experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1"
SOURCE_CONFIG = "experiments/configs/physiology_semantic_tokenizer/shared_driver_physiology_semantics_v1.yaml"
ARTIFACT_ROOT = "experiments/runs/physiology_semantic_tokenizer/semantic_response_discovery"
DATASETS = ("eeg_fnirs_single_trial", "simultaneous_eeg_nirs", "visual_cognitive_motivation")
REGIONS = ("prefrontal", "motor", "posterior")
ARMS = ("broad_a", "gamma_selected", "gamma_attenuation", "gain_selected", "zero_driver",
        "random_seed_0", "random_seed_1", "random_seed_2", "input_lowpass", "viscoelastic_outflow")
BASELINES = ("training_template", "task_condition_template_oracle", "own_Hb_prefix_ridge",
             "full_EEG_ridge", "spatial_Hb_ridge", "spatial_Hb_local_EEG_ridge",
             "spatial_Hb_filterbank_EEG_ridge", "spatial_filterbank_train_permutation")
PAIRINGS = ("real", "wrong_training_subject", "real_shift_support",
            "nonwrapping_shift_12s", "other_region_EEG")
SPLIT_COUNTS = {
    "eeg_fnirs_single_trial": {"train": 18, "selection": 5, "evaluation": 6},
    "simultaneous_eeg_nirs": {"train": 17, "selection": 4, "evaluation": 5},
    "visual_cognitive_motivation": {"train": 9, "selection": 3, "evaluation": 4},
}
WINDOW_CAPS = {"train": 4, "selection": 4, "evaluation": 8}
METRICS = ("Hb_nrmse", "HbO_nrmse", "HbR_nrmse", "Gaussian_NLL", "Gaussian_CRPS",
           "Gaussian_CRPS_normalized", "coverage_50", "coverage_80", "coverage_95")


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serial(v) for v in value]
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **{k: np.asarray(v) for k, v in arrays.items()})
    temporary.replace(path)


def read_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    try:
        if (cfg["schema"] != "semantic_response_discovery_v1" or cfg["source_run"] != SOURCE_RUN
                or cfg["source_config"] != SOURCE_CONFIG or cfg["artifact_root"] != ARTIFACT_ROOT
                or cfg["protected_data"] != "forbidden"
                or cfg["data_boundary"] != "exact_public_parent_prepared_regions_frozen_subject_disjoint_metadata_plan"
                or cfg["metadata_plan"] != "metadata_plan_proposal.json"):
            raise ValueError("wrong public source or protected-data boundary")
        t, m, b, measured = cfg["tensor"], cfg["model"], cfg["baselines"], cfg["measured"]
        if (tuple(cfg["data"]["datasets"]) != DATASETS or tuple(cfg["data"]["region_names"]) != REGIONS
                or t["steps"] != 120 or t["dt_s"] != .25 or t["eeg_channels"] != 6 or t["eeg_features"] != 30
                or t["eeg_bands_hz"] != [[1, 4], [4, 8], [8, 13], [13, 30], [30, 45]]
                or t["feature_order"] != "channel_major_band_minor" or t["reference_steps"] != 20
                or t["Hb_components"] != ["HbO", "HbR"] or t["temporal_modes"] != 24
                or tuple(m["arms"]) != ARMS or m["gamma_candidates"] != [-1., -.5, 0., .5, 1.]
                or not np.allclose(m["gain_candidates"], [0., .5, 1 / np.sqrt(2), 1 / np.sqrt(1.25), 1.], rtol=0, atol=1e-14)
                or m["tau_n_candidates"] != [0., .5, 1., 2.]
                or m["tau_v_candidates"] != [0., 1., 3., 6.]
                or len(m["random_seeds"]) != 3 or len(set(m["random_seeds"])) != 3
                or m["state_sd"] != .025 or m["ar_phi"] != .95 or m["ar_weight"] != .1
                or m["eeg_mode_training_RMS_target"] != .025
                or m["common_Hb_modes"] != 4 or m["common_Hb_rho"] != .35
                or m["initial_mode"] != "tied_pv_logprior"
                or measured["prefix_steps"] != [40, 0] or measured["score_start"] != 80 or measured["score_stop"] != 120
                or measured["shift_steps"] != 48 or tuple(measured["pairings"]) != PAIRINGS
                or measured["null_arms"] != ["gamma_selected", "gain_selected", "best_stable"]
                or measured["initial_ablation_modes"] != ["rest", "legacy_free_initial"]
                or b["ridge_candidates"] != [.01, .1, 1., 10.] or b["time_modes"] != 4
                or b["filter_time_constants"] != [.5, 1., 2., 4., 8.]
                or cfg["uncertainty"]["minimum_training_sd_fraction"] != .05
                or cfg["resources"]["numerical_threads"] != 1
                or not 1 <= cfg["resources"]["max_workers"] <= 48):
            raise ValueError("tensor, split, information, model or resource contract mismatch")
        if (len(m["initial_prior_sd"]) != 4 or len(m["legacy_prior_sd"]) != 5
                or min(m["initial_prior_sd"] + m["legacy_prior_sd"]) <= 0
                or not np.isfinite(m["initial_prior_sd"] + m["legacy_prior_sd"]).all()
                or not np.isfinite(m["initial_weight"]) or m["initial_weight"] <= 0
                or cfg["solver"]["max_nfev"] < 1 or cfg["solver"]["substeps"] < 1
                or cfg["solver"]["gradient_tolerance"] <= 0
                or cfg["solver"]["numerical_backend"] not in ("numba", "python")
                or cfg["calibration"]["failure_penalty"] <= 0 or cfg["synthetic"]["repeats"] < 1):
            raise ValueError("positive priors and solver budgets required")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"incomplete semantic response contract: {exc}") from exc
    return cfg


def parameters(cfg, project_root=CODE_ROOT):
    source = yaml.safe_load((Path(project_root) / cfg["source_config"]).read_text())
    fixed = {k: v for k, v in source["fixed"].items() if k not in ("tau", "kappa")}
    p = BalloonParameters(fixed=BalloonFixedParameters(**fixed),
                          free=BalloonFreeParameters(tau=source["fixed"]["tau"], kappa=source["fixed"]["kappa"]))
    p.validate()
    return p


@lru_cache(maxsize=2)
def operators(steps=120):
    native = native_feature_operators(steps)
    eeg, hb = native["eeg"], native["fnirs"] @ native["native_interpolation"]
    block = np.zeros((3 * steps, 3 * steps))
    block[0::3, 0::3], block[1::3, 1::3], block[2::3, 2::3] = eeg, hb, hb
    common = equal_capacity_hb_basis(block, modes=4, rho=.35).reshape(steps, 3, 4)[:, 1:]
    return dict(eeg=eeg, hb=hb, common=common, temporal=temporal_mode_basis(steps, 24),
                time=temporal_mode_basis(steps, 5)[:, 1:])


def validate_ref(ref, parent):
    """Reject dataset, geometry, path and identity drift before any NPZ access."""
    if ref.get("dataset") not in DATASETS:
        raise ValueError("unsupported or protected dataset before array read")
    if ref["dataset"] == "visual_cognitive_motivation" and ref.get("subject") == "S06" and "Part1" in ref.get("record", ""):
        raise ValueError("excluded Visual S06 Part1 before array read")
    region, site = ref.get("region"), ref.get("site")
    allowed_sites = (region, region + "__montage0") if ref["dataset"] == "visual_cognitive_motivation" and region else (region,)
    if region not in REGIONS or site not in allowed_sites:
        raise ValueError("unregistered region identity")
    key = f'{ref["dataset"]}__{ref["subject"]}__{ref["record"]}'
    expected = (Path(parent) / "prepared" / key / (ref["region"] + ".npz")).resolve()
    path = Path(ref["array_path"]).resolve()
    if (path != expected or not path.is_relative_to(Path(parent).resolve() / "prepared")
            or ref.get("key") != key or ref.get("id") != key + f'__{site}__w{ref["window"]}'):
        raise ValueError("array or identity outside exact public parent regional namespace")
    index = ref.get("array_index")
    if (isinstance(index, bool) or not isinstance(index, int) or index < 0 or index != ref.get("window")
            or len(ref.get("eeg_channels", [])) != 6 or len(set(ref["eeg_channels"])) != 6
            or not ref.get("hb_channel") or not all(np.isfinite(ref[k]) for k in ("eeg_start_s", "hb_start_s"))):
        raise ValueError("invalid feature, channel or native window identity")
    return path


@lru_cache(maxsize=12)
def prepared_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        eeg, hb = arrays["eeg_features"].copy(), arrays["hb"].copy()
    if eeg.ndim != 3 or eeg.shape[1:] != (120, 30) or hb.shape != (len(eeg), 120, 2):
        raise ValueError("parent prepared array shape mismatch")
    return eeg, hb


def raw_target(ref, parent):
    path = validate_ref(ref, parent)
    eeg, hb = prepared_arrays(str(path))
    index = ref["array_index"]
    if index >= len(eeg):
        raise ValueError("native window array index out of range")
    eeg, hb = eeg[index], hb[index]
    if not np.isfinite(eeg).all() or not np.isfinite(hb).all():
        raise ValueError("parent prepared window lacks complete finite support")
    return eeg, hb


def parent_plan(cfg, run_root, project_root=CODE_ROOT):
    """Validate metadata membership only; parent fitted coordinates are never read."""
    parent = (Path(project_root) / cfg["source_run"]).resolve()
    if read_json(parent / "summary.json").get("execution") != "completed":
        raise ValueError("exact public parent evidence is not terminal")
    proposal = read_json(Path(run_root) / cfg["metadata_plan"])
    if (proposal.get("schema") != "semantic_response_discovery_metadata_plan_proposal_v1"
            or Path(proposal.get("parent_run", "")).resolve() != parent
            or proposal.get("source_scope") != "exact_public_parent_prepared_regional_EEG30_Hb2"
            or proposal.get("seed") != cfg["seed"]):
        raise ValueError("metadata proposal source or seed mismatch")
    parent_refs = read_json(parent / "cohort.json")["refs"]
    lookup = {r["id"]: r for r in parent_refs}
    if len(lookup) != len(parent_refs):
        raise ValueError("duplicate public parent identity")
    old_panel = read_json(parent / "diagnostic_plan.json")["windows"]
    previous = {dataset: {r["subject"] for r in old_panel if r["dataset"] == dataset} for dataset in DATASETS}
    for dataset in DATASETS:
        split = proposal["splits"][dataset]
        groups = [set(split[k]) for k in WINDOW_CAPS]
        if (any(len(groups[i]) != SPLIT_COUNTS[dataset][name] for i, name in enumerate(WINDOW_CAPS))
                or any(groups[i] & groups[j] for i in range(3) for j in range(i))
                or groups[2] & previous[dataset]):
            raise ValueError("subject split or previous-panel evaluation exclusion mismatch")
        admitted = {r["subject"] for r in parent_refs if r["dataset"] == dataset}
        if set.union(*groups) != admitted:
            raise ValueError("proposal does not exactly partition admitted public subjects")
    items, seen, subject_windows = [], set(), {}
    for window in proposal["windows"]:
        dataset, subject, split = window["dataset"], window["subject"], window["split"]
        if dataset not in DATASETS or split not in WINDOW_CAPS or subject not in proposal["splits"][dataset][split]:
            raise ValueError("window subject outside declared split")
        if window["native_id"] in seen:
            raise ValueError("duplicate native window identity")
        seen.add(window["native_id"])
        subject_windows[(dataset, subject, split)] = subject_windows.get((dataset, subject, split), 0) + 1
        refs = window["regional_refs"]
        if len(refs) != 3 or {r["region"] for r in refs} != set(REGIONS):
            raise ValueError("native window requires three synchronous regions")
        for ref in refs:
            validate_ref(ref, parent)
            if lookup.get(ref["id"]) != ref:
                raise ValueError("regional ref differs from exact public parent cohort")
            if (any(ref[k] != window[k] for k in ("dataset", "subject", "task", "condition", "window"))
                    or abs(ref["eeg_start_s"] - window["eeg_start_s"]) > 1e-6
                    or abs(ref["hb_start_s"] - window["hb_start_s"]) > 1e-6):
                raise ValueError("regional native identity or timing mismatch")
            items.append(dict(ref=ref, split=split, native_id=window["native_id"],
                              calibration_key=f'{dataset}__{ref["region"]}',
                              synchronous_region_ids=sorted(r["id"] for r in refs if r["id"] != ref["id"])))
    if (len(items) != 1032 or len(seen) != 344
            or any(n > WINDOW_CAPS[key[2]] for key, n in subject_windows.items())
            or len(subject_windows) != sum(sum(v.values()) for v in SPLIT_COUNTS.values())):
        raise ValueError("registered subject/window caps or full split inventory mismatch")
    calibration = []
    for dataset in DATASETS:
        for region in REGIONS:
            selected = [x for x in items if x["ref"]["dataset"] == dataset and x["ref"]["region"] == region]
            calibration.append(dict(key=f"{dataset}__{region}", dataset=dataset, region=region,
                                    **{f"{split}_ids": [x["ref"]["id"] for x in selected if x["split"] == split]
                                       for split in WINDOW_CAPS},
                                    **{f"{split}_subjects": proposal["splits"][dataset][split] for split in WINDOW_CAPS}))
    return dict(schema="semantic_response_discovery_public_plan_v1", parent=str(parent),
                proposal_path=str(Path(run_root) / cfg["metadata_plan"]), windows=items, calibration=calibration,
                refs=[x["ref"] for x in items], counts=proposal["counts"], total_native_windows=344,
                total_regional_rows=1032, split_algorithm=proposal["split_algorithm"],
                historical_scope=proposal["historical_scope"], limitations=proposal["limitations"])


def fit_coordinates(eeg, hb, cfg, source_cfg):
    """One new coordinate fit using only explicitly selected training payloads."""
    eeg, hb = np.asarray(eeg, float), np.asarray(hb, float)
    if (eeg.ndim != 3 or eeg.shape[1:] != (120, 30) or hb.shape != (len(eeg), 120, 2)
            or not len(eeg) or not np.isfinite(eeg).all() or not np.isfinite(hb).all()):
        raise ValueError("finite aligned training arrays required")
    x, y = eeg.reshape(-1, 30), hb.reshape(-1, 2)
    projection_rms = np.sqrt(np.mean((x @ spectral_loadings()) ** 2, axis=0))
    variance_hb = y.var(axis=0)
    covariance = np.cov(x, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    pc = eigenvectors[:, -1]
    if pc[np.argmax(np.abs(pc))] < 0:
        pc = -pc
    if projection_rms[0] <= 1e-12 or min(variance_hb) <= 1e-24 or eigenvalues[-1] <= 1e-16:
        raise ValueError("degenerate train-only EEG/Hb coordinate")
    eeg_factor = cfg["model"]["eeg_mode_training_RMS_target"] / projection_rms[0]
    hb_factor = source_cfg["coordinate"]["common_Hb_training_sd_target"] / np.sqrt(variance_hb.mean())
    sd_eeg = np.maximum((x * eeg_factor).std(axis=0), 1e-8)
    sd_hb = np.maximum((y * hb_factor).std(axis=0), 1e-8)
    return dict(eeg_factor=eeg_factor, hb_factor=hb_factor, sd_eeg=sd_eeg, sd_hb=sd_hb,
                pca_loading=pc, pca_eigenvalues=eigenvalues, training_eeg_mean=x.mean(axis=0),
                training_hb_mean=y.mean(axis=0), training_projection_RMS_raw=projection_rms,
                training_projection_RMS_scaled=projection_rms * eeg_factor,
                fit_eeg_sd=sd_eeg, fit_hb_sd=sd_hb,
                modal_loss="separate_objectives_EEG_coordinate_average_library_owned; Hb_only_standardized_SSE_with_proper_priors",
                centering="parent_first_5s_reference_retained; PCA covariance mean not subtracted from observations",
                coordinate_fit="all_and_only_new_train_prepared_windows; no_parent_fold_coordinates",
                SD_interpretation="training_feature_variability_for_loss; predictive_SD_calibrated_separately")


def prefix_mask(prefix, steps=120):
    if isinstance(prefix, bool) or not isinstance(prefix, (int, np.integer)) or not 0 <= prefix <= steps:
        raise ValueError("explicit valid Hb prefix required")
    mask = np.zeros((steps, 2), bool)
    mask[:prefix] = True
    return mask


def endpoint_mask(cfg, support=None):
    mask = np.zeros((cfg["tensor"]["steps"], 2), bool)
    mask[cfg["measured"]["score_start"]:cfg["measured"]["score_stop"]] = True
    if support is not None:
        support = np.asarray(support, bool)
        if support.shape != (len(mask),):
            raise ValueError("endpoint support must be [T]")
        mask &= support[:, None]
    return mask


def visible_hb(hb, prefix):
    """Construct the fitter input from the prefix, never hidden suffix values."""
    hb = np.asarray(hb)
    if hb.shape != (120, 2):
        raise ValueError("Hb coordinate must be [120,2]")
    mask = prefix_mask(prefix)
    out = np.full(hb.shape, np.nan)
    out[mask] = hb[mask]
    if not np.isfinite(out[mask]).all():
        raise ValueError("observed prefix must be finite")
    return out, mask


def random_loadings(seed):
    fixed = spectral_loadings()
    rng = np.random.default_rng(seed)
    contrast = rng.normal(size=len(fixed))
    contrast -= fixed @ (fixed.T @ contrast)
    contrast /= np.linalg.norm(contrast)
    return np.column_stack((fixed[:, 0], contrast))


def infer_modes(eeg, cal, cfg, seed=None, visible=None):
    from src.inference.semantic_response_ssm import infer_eeg_modes
    return infer_eeg_modes(eeg, random_loadings(seed) if seed is not None else spectral_loadings(),
                           np.asarray(cal["fit_eeg_sd"]), temporal_basis=operators()["temporal"],
                           eeg_operator=operators()["eeg"], visible=visible, state_sd=cfg["model"]["state_sd"],
                           ar_phi=cfg["model"]["ar_phi"], ar_weight=cfg["model"]["ar_weight"])


def model_spec(arm, cal, prefix, *, initial_mode=None):
    if arm == "best_stable":
        arm = cal["selected"][str(prefix)]["best_stable"]
    selected = cal["selected"][str(prefix)]
    spec = dict(arm=arm, gamma=0., gain=1., tau_n=0., tau_v=0., random_seed=None,
                initial_mode=initial_mode or "tied_pv_logprior")
    if arm == "gamma_selected":
        spec["gamma"] = selected["gamma"]
    elif arm == "gamma_attenuation":
        spec["gain"] = 1 / np.sqrt(1 + selected["gamma"] ** 2)
    elif arm in ("gain_selected", "input_lowpass", "viscoelastic_outflow"):
        spec["gain"] = selected["gain"]
        if arm == "input_lowpass":
            spec["tau_n"] = selected["tau_n"]
        if arm == "viscoelastic_outflow":
            spec["tau_v"] = selected["tau_v"]
    elif arm == "zero_driver":
        spec["gain"] = 0.
    elif arm.startswith("random_seed_"):
        index = int(arm.rsplit("_", 1)[1])
        spec["random_seed"] = cal["random_seeds"][index]
        spec["gamma"] = selected["random_gamma"][str(index)]
    elif arm != "broad_a":
        raise ValueError("unregistered fixed-driver arm")
    return spec


def driver_from_modes(modes, spec):
    if spec["gamma"]:
        return (np.asarray(modes["a"]) + spec["gamma"] * np.asarray(modes["b"])) / np.sqrt(1 + spec["gamma"] ** 2)
    return spec["gain"] * np.asarray(modes["a"])


def fit_response(hb, modes, spec, cal, cfg, p, prefix):
    from src.inference.semantic_response_ssm import fit_fixed_driver_response
    observed, mask = visible_hb(hb, prefix)
    initial = spec["initial_mode"]
    if initial == "rest":
        prior_mean, prior_sd = None, None
    elif initial == "legacy_free_initial":
        prior_mean, prior_sd = [0., 1., 1., 1., 1.], cfg["model"]["legacy_prior_sd"]
    else:
        prior_mean, prior_sd = np.zeros(4), cfg["model"]["initial_prior_sd"]
    library_initial = "legacy_free_physicalprior" if initial == "legacy_free_initial" else initial
    return fit_fixed_driver_response(observed, driver_from_modes(modes, spec), p, cfg["tensor"]["dt_s"],
        sd=np.asarray(cal["fit_hb_sd"]), visible=mask, hb_operator=operators()["hb"],
        hb_basis=operators()["common"], initial_mode=library_initial, initial_prior_mean=prior_mean,
        initial_prior_sd=prior_sd, initial_weight=cfg["model"]["initial_weight"],
        component_sd=np.asarray(cal["component_sd"]), max_nfev=cfg["solver"]["max_nfev"],
        gradient_tolerance=cfg["solver"]["gradient_tolerance"], tau_n=spec["tau_n"], tau_v=spec["tau_v"],
        substeps=cfg["solver"]["substeps"], numerical_backend=cfg["solver"]["numerical_backend"])


def prediction_scores(prediction, target, training_sd, endpoint, predictive_sd=None):
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    mask, training_sd = np.asarray(endpoint, bool), np.asarray(training_sd, float)
    if prediction.shape != (120, 2) or target.shape != prediction.shape or mask.shape != target.shape or training_sd.shape != (2,):
        raise ValueError("Hb score coordinate/shape mismatch")
    if np.any(training_sd <= 0) or not np.isfinite(training_sd).all():
        raise ValueError("positive training SD required")
    if not np.isfinite(target[mask]).all() or not np.isfinite(prediction[mask]).all():
        raise ValueError("finite prediction and target required on scored support")
    error = prediction - target
    result = dict(scored_HbO_points=int(mask[:, 0].sum()), scored_HbR_points=int(mask[:, 1].sum()))
    for name, columns in (("Hb", [0, 1]), ("HbO", [0]), ("HbR", [1])):
        supported = mask[:, columns]
        result[name + "_nrmse"] = float(np.sqrt(np.mean((error[:, columns] / training_sd[columns])[supported] ** 2))) if supported.any() else None
    if predictive_sd is not None:
        sigma = np.broadcast_to(np.asarray(predictive_sd, float), target.shape)
        if np.any(sigma <= 0) or not np.isfinite(sigma).all():
            raise ValueError("positive frozen predictive SD required")
        z = error / sigma
        nll = .5 * np.log(2 * np.pi) + np.log(sigma) + .5 * z ** 2
        crps = sigma * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-.5 * z ** 2) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))
        result["Gaussian_NLL"] = float(nll[mask].mean()) if mask.any() else None
        result["Gaussian_CRPS"] = float(crps[mask].mean()) if mask.any() else None
        result["Gaussian_CRPS_normalized"] = float((crps / training_sd)[mask].mean()) if mask.any() else None
        for j, label in enumerate(("HbO", "HbR")):
            if mask[:, j].any():
                result[label + "_Gaussian_CRPS"] = float(crps[mask[:, j], j].mean())
                result[label + "_Gaussian_CRPS_normalized"] = float((crps[:, j] / training_sd[j])[mask[:, j]].mean())
                result[label + "_Gaussian_NLL"] = float(nll[mask[:, j], j].mean())
        for label, quantile in (("50", .6744897501960817), ("80", 1.2815515655446004), ("95", 1.959963984540054)):
            result["coverage_" + label] = float((np.abs(z[mask]) <= quantile).mean()) if mask.any() else None
        result["predictive_sd"] = np.asarray(predictive_sd)
    return result


def compact_fit(fit):
    keys = ("success", "converged", "status", "fit_performed", "objective", "evaluations", "nfev",
            "failure_log", "physical_check", "scaled_gradient_inf_norm", "convergence_reason")
    result = {key: fit[key] for key in keys if key in fit}
    result.setdefault("success", False)
    result.setdefault("converged", False)
    result.setdefault("fit_performed", True)
    result.setdefault("status", "missing_status")
    return result


def subject_equal_ranking(rows, candidates, key, penalty):
    ranking = []
    for candidate in candidates:
        cells = [r for r in rows if r.get(key) == candidate]
        subjects = {}
        for row in cells:
            value = row.get("Hb_nrmse")
            loss = float(value) if row.get("success") and value is not None and np.isfinite(value) else penalty
            subjects.setdefault(row["subject"], []).append(loss)
        ranking.append(dict(**{key: candidate}, score=float(np.mean([np.mean(v) for v in subjects.values()])) if subjects else penalty,
                            planned=len(cells), successful=sum(bool(r.get("success")) for r in cells), subjects=len(subjects)))
    ranking.sort(key=lambda r: (r["score"], abs(r[key]) if isinstance(r[key], (float, int)) else 0, str(r[key])))
    return ranking


def calibrate_predictive_sd(records, training_sd, floor_fraction=.05):
    """Selection residual RMS, subject balanced, with a training-only SD floor."""
    moments = {}
    for row, residual, mask in records:
        if not row.get("success"):
            continue
        error, support = np.asarray(residual, float), np.asarray(mask, bool)
        if error.shape != (120, 2) or support.shape != error.shape:
            raise ValueError("predictive calibration residual/support shape mismatch")
        values = []
        for j in range(2):
            values.append(np.mean(error[support[:, j], j] ** 2) if support[:, j].any() and np.isfinite(error[support[:, j], j]).all() else np.nan)
        if np.isfinite(values).all():
            moments.setdefault(row["subject"], []).append(values)
    if not moments:
        return dict(status="selection_residuals_unavailable", predictive_sd=None, subjects=0, successful_windows=0)
    mse = np.mean([np.mean(v, axis=0) for v in moments.values()], axis=0)
    sd = np.maximum(np.sqrt(mse), np.asarray(training_sd) * floor_fraction)
    return dict(status="completed", predictive_sd=sd, subjects=len(moments),
                successful_windows=sum(map(len, moments.values())), fit_partition="selection_only",
                convention="subject_equal_endpoint_residual_RMS_no_evaluation_refit",
                floor_training_sd_fraction=floor_fraction)


def context_features(eeg, own_hb, spatial_hb, prefix, kind, *, dt=.25):
    """All own-target Hb inputs come from the declared prefix only.

    Spatial predictors contain only the other two regions. Time features and
    prefix history are common to every fitted linear arm, including EEG-only.
    """
    eeg, own_hb = np.asarray(eeg, float), np.asarray(own_hb, float)
    if eeg.shape != (120, 30) or own_hb.shape != (120, 2):
        raise ValueError("baseline local coordinates must be EEG30 and Hb2")
    if kind not in BASELINES[2:]:
        raise ValueError("unregistered baseline feature family")
    history, _ = visible_hb(own_hb, prefix)
    features = [operators()["time"]]
    if prefix:
        features.append(np.broadcast_to(history[:prefix].reshape(1, -1), (120, 2 * prefix)))
    if kind != "own_Hb_prefix_ridge":
        if not np.isfinite(eeg).all():
            raise ValueError("finite EEG input required")
    if kind.startswith("spatial"):
        spatial = np.asarray(spatial_hb, float)
        if spatial.shape != (120, 4) or not np.isfinite(spatial).all():
            raise ValueError("exactly two other-region Hb pairs required")
        features.append(spatial)
    if kind in ("full_EEG_ridge", "spatial_Hb_local_EEG_ridge"):
        features.append(eeg)
    elif kind in ("spatial_Hb_filterbank_EEG_ridge", "spatial_filterbank_train_permutation"):
        from src.inference.semantic_response_ssm import cascade_response_features
        features.append(cascade_response_features(eeg, dt, time_constants=(.5, 1., 2., 4., 8.)))
    return np.column_stack(features)


def select_donor(ref, training_refs):
    donors = sorted((r for r in training_refs if r["subject"] != ref["subject"]
                     and r["task"] == ref["task"] and r["condition"] == ref["condition"]), key=lambda r: r["id"])
    if not donors:
        return None
    rank = int(hashlib.sha256(ref["id"].encode()).hexdigest(), 16)
    return donors[rank % len(donors)]


def training_permutation(refs, seed):
    """Bijective task/condition assignment preserves the EEG training marginal."""
    lookup, missing, groups = {}, [], {}
    for ref in refs:
        key = (ref["dataset"], ref["region"], ref["task"], ref["condition"])
        groups.setdefault(key, []).append(ref)
    for key, members in sorted(groups.items()):
        members.sort(key=lambda r: r["id"])
        cost = np.full((len(members), len(members)), np.inf)
        for i, ref in enumerate(members):
            for j, donor in enumerate(members):
                if ref["subject"] != donor["subject"]:
                    digest = hashlib.sha256(f'{seed}|{ref["id"]}|{donor["id"]}'.encode()).digest()
                    cost[i, j] = int.from_bytes(digest[:8], "big") / 2 ** 64
        try:
            rows, columns = linear_sum_assignment(cost)
            feasible = len(rows) == len(members) and np.isfinite(cost[rows, columns]).all()
        except ValueError:
            feasible = False
        if not feasible:
            missing.extend(dict(id=r["id"], group=list(key), reason="matched_bijective_cross_subject_permutation_unavailable") for r in members)
            continue
        lookup.update({members[i]["id"]: members[j]["id"] for i, j in zip(rows, columns)})
    return lookup, missing


def pairing_eeg(eeg, cfg, pairing, donor_eeg=None):
    eeg = np.asarray(eeg, float)
    if eeg.shape != (120, 30):
        raise ValueError("pairing EEG must be [120,30]")
    out, visible, support = eeg.copy(), np.ones_like(eeg, bool), np.ones(120, bool)
    if pairing in ("real_shift_support", "nonwrapping_shift_12s"):
        shift = cfg["measured"]["shift_steps"]
        support[:shift], visible[:shift] = False, False
        if pairing == "nonwrapping_shift_12s":
            out[shift:] = eeg[:-shift]
        out[:shift] = np.nan
    elif pairing in ("wrong_training_subject", "other_region_EEG"):
        if donor_eeg is None or np.asarray(donor_eeg).shape != eeg.shape or not np.isfinite(donor_eeg).all():
            raise ValueError("registered supported EEG donor required")
        out = np.asarray(donor_eeg, float).copy()
    elif pairing != "real":
        raise ValueError("unregistered pairing")
    return out, visible, support


def scaled_window(ref, parent, cal):
    eeg, hb = raw_target(ref, parent)
    return eeg * cal["eeg_factor"], hb * cal["hb_factor"]


def spatial_target(ref, refs, parent, cal):
    donors = sorted((r for r in refs if r["dataset"] == ref["dataset"] and r["subject"] == ref["subject"]
                     and r["key"] == ref["key"] and r["window"] == ref["window"] and r["region"] != ref["region"]
                     and abs(r["eeg_start_s"] - ref["eeg_start_s"]) < 1e-6
                     and abs(r["hb_start_s"] - ref["hb_start_s"]) < 1e-6), key=lambda r: r["region"])
    if len(donors) != 2 or ref["hb_channel"] in {d["hb_channel"] for d in donors}:
        raise ValueError("two synchronous other-region Hb pairs required; target cannot be a spatial input")
    return np.column_stack([raw_target(d, parent)[1] * cal["hb_factor"] for d in donors]), donors


def mode_cache(ref, eeg, cal, cfg, run_root, *, seed=None, pairing="real", visible=None):
    label = "fixed" if seed is None else f"random_{seed}"
    path = Path(run_root) / "eeg_modes" / cal["key"] / ref["id"] / (pairing + "__" + label + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if path.exists():
            with np.load(path, allow_pickle=False) as arrays:
                return {k: arrays[k].copy() for k in arrays.files}
        fit = infer_modes(eeg, cal, cfg, seed=seed, visible=visible)
        result = {k: fit[k] for k in ("r", "a", "b", "prediction", "coefficients") if k in fit}
        result["support_coordinate_counts"] = fit["support"]["coordinate_counts"]
        result.setdefault("r", np.column_stack((result["a"], result["b"])))
        save_arrays(path, **result)
        return result


def parameter_slug(spec):
    selected = {k: spec[k] for k in ("gamma", "gain", "tau_n", "tau_v", "random_seed", "initial_mode")}
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()[:20]


def selection_cell(ref, parent, cal, cfg, p, run_root, spec, prefix):
    path = Path(run_root) / "selection_cells" / cal["key"] / parameter_slug(spec) / ref["id"] / f"prefix_{prefix}.json"
    if path.exists():
        row = read_json(path)
        if row["spec"] != serial({k: spec[k] for k in spec if k != "arm"}):
            raise ValueError("selection resume parameter mismatch")
        return row
    started = time.monotonic()
    eeg, hb = scaled_window(ref, parent, cal)
    modes = mode_cache(ref, eeg, cal, cfg, run_root, seed=spec["random_seed"])
    fit = fit_response(hb, modes, spec, cal, cfg, p, prefix)
    row = dict(id=ref["id"], subject=ref["subject"], prefix_steps=prefix,
               spec={k: spec[k] for k in spec if k != "arm"}, **compact_fit(fit))
    if "prediction" in fit:
        row.update(prediction_scores(fit["prediction"], hb, cal["sd_hb"], endpoint_mask(cfg)))
        save_arrays(path.with_suffix(".npz"), prediction=fit["prediction"],
                    physical_prediction=fit.get("physical_prediction", fit["prediction"]),
                    observation_component=fit.get("observation_component", np.zeros_like(hb)),
                    initial_state=fit.get("initial_state", np.array([])),
                    component_coefficients=fit.get("component_coefficients", np.array([])))
        row["arrays"] = str(path.with_suffix(".npz").relative_to(run_root))
    row["seconds"] = time.monotonic() - started
    write_json(path, row)
    return row


def calibrate_arm_uncertainty(refs, parent, cal, cfg, p, run_root, arm, prefix, initial_mode=None):
    spec = model_spec(arm, cal, prefix, initial_mode=initial_mode)
    records = []
    for ref in refs:
        row = selection_cell(ref, parent, cal, cfg, p, run_root, spec, prefix)
        if row.get("arrays"):
            with np.load(Path(run_root) / row["arrays"], allow_pickle=False) as arrays:
                prediction = arrays["prediction"].copy()
            target = scaled_window(ref, parent, cal)[1]
            records.append((row, prediction - target, endpoint_mask(cfg)))
    return calibrate_predictive_sd(records, cal["sd_hb"], cfg["uncertainty"]["minimum_training_sd_fraction"])


def export_features(ref, split, eeg, hb, modes, cal, run_root):
    path = Path(run_root) / "features" / split / (ref["id"] + ".npz")
    if path.exists():
        return str(path.relative_to(run_root))
    centered = np.asarray(modes["r"]) - np.asarray(modes["r"])[:20].mean(axis=0)
    save_arrays(path, eeg=eeg, hb=hb, eeg_modes=modes["r"], eeg_modes_reference_centered=centered,
                sd_eeg=cal["sd_eeg"], sd_hb=cal["sd_hb"], time_s=np.arange(120) * .25,
                observation_valid_eeg=np.ones_like(eeg, bool), observation_valid_hb=np.ones_like(hb, bool))
    write_json(path.with_suffix(".json"), dict(ref=ref, split=split, calibration_key=cal["key"],
               coordinates="new_train_scaled_relative_parent_prepared_EEG30_Hb2", EEG_teacher="EEG_only_fixed_modes",
               interpretation="public_exploratory_features_not_physiological_ground_truth",
               eeg_factor=cal["eeg_factor"], hb_factor=cal["hb_factor"]))
    return str(path.relative_to(run_root))


def baseline_prediction(kind, ref, eeg, hb, spatial, prefix, cal, model=None):
    if kind == "training_template":
        return np.asarray(cal["training_template"])
    if kind == "task_condition_template_oracle":
        key = json.dumps([ref["task"], ref["condition"]], separators=(",", ":"))
        return np.asarray(cal["task_condition_templates"].get(key, cal["training_template"]))
    if model is None:
        model = cal["ridge_models"][str(prefix)][kind]
    model = {k: np.asarray(v) for k, v in model.items()}
    features = context_features(eeg, hb, spatial, prefix, kind)
    return predict_standardized_ridge(model, features)


def calibration_worker(payload):
    cfg, project_root, run_root, spec, refs, parent = payload
    path = Path(run_root) / "calibration" / (spec["key"] + ".json")
    if path.exists():
        return read_json(path)
    started = time.monotonic()
    lookup = {r["id"]: r for r in refs}
    train = [lookup[i] for i in spec["train_ids"]]
    select = [lookup[i] for i in spec["selection_ids"]]
    if (set(spec["train_subjects"]) & set(spec["selection_subjects"])
            or (set(spec["train_subjects"]) | set(spec["selection_subjects"])) & set(spec["evaluation_subjects"])):
        raise ValueError("training/selection/evaluation subjects overlap")
    raw = [raw_target(r, parent) for r in train]
    e, h = np.array([x[0] for x in raw]), np.array([x[1] for x in raw])
    source_cfg = yaml.safe_load((Path(project_root) / cfg["source_config"]).read_text())
    cal = dict(spec, **fit_coordinates(e, h, cfg, source_cfg), random_seeds=cfg["model"]["random_seeds"])
    e, h = e * cal["eeg_factor"], h * cal["hb_factor"]
    cal["training_template"] = h.mean(axis=0)
    conditions = {}
    for ref, target in zip(train, h):
        key = json.dumps([ref["task"], ref["condition"]], separators=(",", ":"))
        conditions.setdefault(key, []).append(target)
    cal["task_condition_templates"] = {k: np.mean(v, axis=0) for k, v in conditions.items()}
    cal["task_template_role"] = "oracle_nuisance_baseline_only_labels_never_enter_SSM_or_tokenizer_teacher"
    p = parameters(cfg, project_root)
    from src.inference.semantic_response_ssm import fixed_driver_response
    coefficients = []
    design = operators()["common"].reshape(240, 4) / np.tile(cal["fit_hb_sd"], 120)[:, None]
    for ref, eeg, hb in zip(train, e, h):
        modes = mode_cache(ref, eeg, cal, cfg, run_root)
        physical = fixed_driver_response(modes["a"], p, .25, hb_operator=operators()["hb"],
                    substeps=cfg["solver"]["substeps"], numerical_backend=cfg["solver"]["numerical_backend"])
        if not np.isfinite(physical["prediction"]).all():
            raise ValueError("training fixed response lacks finite support")
        residual = (hb - physical["prediction"]) / cal["fit_hb_sd"]
        coefficients.append(np.linalg.lstsq(design, residual.ravel(), rcond=1e-10)[0])
        export_features(ref, "train", eeg, hb, modes, cal, run_root)
    cal["component_sd"] = np.maximum(np.sqrt(np.mean(np.asarray(coefficients) ** 2, axis=0)), np.mean(cal["sd_hb"]) * 1e-6)
    cal["component_prior_fit"] = "training_only_residual_projection_of_fixed_broad_rest_response; same_prior_for_all_arms"
    cal["selected"] = {str(prefix): dict(gamma=0., gain=1., tau_n=0., tau_v=0., random_gamma={}, best_stable="gain_selected")
                       for prefix in cfg["measured"]["prefix_steps"]}
    prefix = 40
    base = dict(arm="selection_candidate", gamma=0., gain=1., tau_n=0., tau_v=0., random_seed=None,
                initial_mode=cfg["model"]["initial_mode"])
    rankings = {}
    for key, candidates in (("gamma", cfg["model"]["gamma_candidates"]), ("gain", cfg["model"]["gain_candidates"])):
        rows = []
        for value in candidates:
            candidate = dict(base, **{key: value})
            for ref in select:
                row = selection_cell(ref, parent, cal, cfg, p, run_root, candidate, prefix)
                rows.append(dict(row, **{key: value}))
        rankings[key] = subject_equal_ranking(rows, candidates, key, cfg["calibration"]["failure_penalty"])
        cal["selected"]["40"][key] = rankings[key][0][key]
    for index, seed in enumerate(cfg["model"]["random_seeds"]):
        rows = []
        for gamma in cfg["model"]["gamma_candidates"]:
            candidate = dict(base, gamma=gamma, random_seed=seed)
            for ref in select:
                row = selection_cell(ref, parent, cal, cfg, p, run_root, candidate, prefix)
                rows.append(dict(row, gamma=gamma))
        ranking = subject_equal_ranking(rows, cfg["model"]["gamma_candidates"], "gamma", cfg["calibration"]["failure_penalty"])
        rankings[f"random_{index}_gamma"] = ranking
        cal["selected"]["40"]["random_gamma"][str(index)] = ranking[0]["gamma"]
    for key in ("tau_n", "tau_v"):
        rows = []
        candidates = cfg["model"][key + "_candidates"]
        for value in candidates:
            candidate = dict(base, gain=cal["selected"]["40"]["gain"], **{key: value})
            for ref in select:
                row = selection_cell(ref, parent, cal, cfg, p, run_root, candidate, prefix)
                rows.append(dict(row, **{key: value}))
        rankings[key] = subject_equal_ranking(rows, candidates, key, cfg["calibration"]["failure_penalty"])
        cal["selected"]["40"][key] = rankings[key][0][key]
    stable = [(rankings["tau_n"][0]["score"], "input_lowpass"),
              (rankings["tau_v"][0]["score"], "viscoelastic_outflow")]
    stable.sort()
    cal["selected"]["40"]["best_stable"] = stable[0][1]
    cal["best_stable_scope"] = "selection40_only_input_lowpass_vs_viscoelastic_outflow; ties_input_lowpass"
    cal["selected"]["0"] = dict(cal["selected"]["40"])
    cal["selection_rankings"] = rankings
    cal["selection_criterion"] = "subject_equal_training_SD_NRMSE_on_selection_prefix40_endpoint80to120"
    cal["zero_prefix_policy"] = "reuse_prefix40_hyperparameters; separate_training_baseline_fit_and_selection_residual_SD"
    cal["uncertainty"], cal["ridge_models"], cal["ridge_selection"] = {}, {}, {}
    for ref in select:
        eeg, hb = scaled_window(ref, parent, cal)
        modes = mode_cache(ref, eeg, cal, cfg, run_root)
        export_features(ref, "selection", eeg, hb, modes, cal, run_root)
    for prefix in cfg["measured"]["prefix_steps"]:
        cal["uncertainty"][str(prefix)] = {}
        for arm in ARMS:
            cal["uncertainty"][str(prefix)][arm] = calibrate_arm_uncertainty(select, parent, cal, cfg, p, run_root, arm, prefix)
        for initial in cfg["measured"]["initial_ablation_modes"]:
            arm = "gamma_selected__" + initial
            cal["uncertainty"][str(prefix)][arm] = calibrate_arm_uncertainty(select, parent, cal, cfg, p, run_root,
                                                                                   "gamma_selected", prefix, initial)
    train_spatial = [spatial_target(ref, refs, parent, cal)[0] for ref in train]
    selection_data = [(ref, *scaled_window(ref, parent, cal), spatial_target(ref, refs, parent, cal)[0]) for ref in select]
    permutation, missing = training_permutation(train, cfg["seed"] + 503)
    cal["training_pair_permutation"] = dict(seed=cfg["seed"] + 503, donors=permutation, missing=missing,
                                            policy="bijective_same_dataset_region_task_condition_different_training_subject_preserve_source_marginal")
    train_index = {ref["id"]: i for i, ref in enumerate(train)}
    selected_weights = {}
    for prefix in cfg["measured"]["prefix_steps"]:
        cal["ridge_models"][str(prefix)] = {}
        for kind in BASELINES:
            model = None
            if kind in BASELINES[2:]:
                if kind == "spatial_filterbank_train_permutation" and missing:
                    cal["uncertainty"][str(prefix)][kind] = dict(status="matched_training_permutation_donor_unavailable", predictive_sd=None)
                    continue
                feature_kind = "spatial_Hb_filterbank_EEG_ridge" if kind == "spatial_filterbank_train_permutation" else kind
                train_features = []
                for i, ref in enumerate(train):
                    input_eeg = e[train_index[permutation[ref["id"]]]] if kind == "spatial_filterbank_train_permutation" else e[i]
                    train_features.append(context_features(input_eeg, h[i], train_spatial[i], prefix, feature_kind))
                x, target = np.concatenate(train_features), h.reshape(-1, 2)
                if prefix == 40 and kind != "spatial_filterbank_train_permutation":
                    rows, models = [], {}
                    for weight in cfg["baselines"]["ridge_candidates"]:
                        candidate = fit_standardized_ridge(x, target, weight)
                        models[weight] = candidate
                        for ref, eeg, hb, spatial in selection_data:
                            prediction = baseline_prediction(kind, ref, eeg, hb, spatial, prefix, cal, candidate)
                            rows.append(dict(id=ref["id"], subject=ref["subject"], weight=weight, success=True,
                                             **prediction_scores(prediction, hb, cal["sd_hb"], endpoint_mask(cfg))))
                    ranking = subject_equal_ranking(rows, cfg["baselines"]["ridge_candidates"], "weight", cfg["calibration"]["failure_penalty"])
                    selected_weights[kind] = ranking[0]["weight"]
                    cal["ridge_selection"][kind] = ranking
                    model = models[selected_weights[kind]]
                else:
                    if kind == "spatial_filterbank_train_permutation":
                        selected_weights[kind] = selected_weights["spatial_Hb_filterbank_EEG_ridge"]
                    model = fit_standardized_ridge(x, target, selected_weights[kind])
                cal["ridge_models"][str(prefix)][kind] = model
            records = []
            for ref, eeg, hb, spatial in selection_data:
                prediction = baseline_prediction(kind, ref, eeg, hb, spatial, prefix, cal, model)
                records.append((dict(subject=ref["subject"], success=True), prediction - hb, endpoint_mask(cfg)))
            cal["uncertainty"][str(prefix)][kind] = calibrate_predictive_sd(records, cal["sd_hb"], cfg["uncertainty"]["minimum_training_sd_fraction"])
    cal["ridge_selected_weights"] = selected_weights
    cal.update(status="completed", success=True, seconds=time.monotonic() - started,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    write_json(path, cal)
    return cal


def measured_cells(cfg, plan):
    cells = []
    for item in plan["windows"]:
        if item["split"] != "evaluation":
            continue
        for prefix in cfg["measured"]["prefix_steps"]:
            for arm in ARMS + BASELINES + ("best_stable",):
                cells.append(dict(item, arm=arm, prefix_steps=prefix, pairing="real", initial_mode="tied_pv_logprior"))
            for arm in cfg["measured"]["null_arms"]:
                for pairing in PAIRINGS[1:]:
                    cells.append(dict(item, arm=arm, prefix_steps=prefix, pairing=pairing, initial_mode="tied_pv_logprior"))
        for initial in cfg["measured"]["initial_ablation_modes"]:
            cells.append(dict(item, arm="gamma_selected__" + initial, prefix_steps=40, pairing="real", initial_mode=initial))
    return cells


def cell_path(run_root, cell):
    return Path(run_root) / "measured" / cell["ref"]["id"] / cell["arm"] / cell["pairing"] / f'prefix_{cell["prefix_steps"]}.json'


@lru_cache(maxsize=12)
def retained_calibration(path):
    return read_json(path)


def measured_cell(cfg, project_root, run_root, cell, refs, parent):
    path = cell_path(run_root, cell)
    if path.exists():
        return read_json(path)
    started = time.monotonic()
    ref, prefix, arm, pairing = cell["ref"], cell["prefix_steps"], cell["arm"], cell["pairing"]
    identity = {k: ref[k] for k in ("id", "dataset", "subject", "record", "site", "region", "task", "condition", "window")}
    row = dict(identity, native_id=cell["native_id"], calibration_key=cell["calibration_key"],
               arm=arm, prefix_steps=prefix, pairing=pairing, initial_mode=cell["initial_mode"],
               eeg_channels=ref["eeg_channels"], hb_channel=ref["hb_channel"],
               temporal_interpretation="offline_processed_feature_completion_not_native_causal_forecast")
    cal = retained_calibration(str(Path(run_root) / "calibration" / (cell["calibration_key"] + ".json")))
    if cal.get("status") != "completed":
        row.update(success=False, converged=False, fit_performed=False, status="calibration_unavailable")
        write_json(path, row)
        return row
    if arm == "best_stable" and pairing == "real":
        chosen = cal["selected"]["40"]["best_stable"]
        retained_cell = dict(cell, arm=chosen)
        retained = measured_cell(cfg, project_root, run_root, retained_cell, refs, parent)
        row = dict(retained, arm="best_stable", chosen_arm=chosen,
                   selection_scope=cal["best_stable_scope"])
        write_json(path, row)
        return row
    eeg, hb = scaled_window(ref, parent, cal)
    spatial, neighbors = spatial_target(ref, refs, parent, cal)
    row["spatial_context_ids"] = [r["id"] for r in neighbors]
    modes = mode_cache(ref, eeg, cal, cfg, run_root)
    export_features(ref, "evaluation", eeg, hb, modes, cal, run_root)
    donor = None
    lookup = {r["id"]: r for r in refs}
    if pairing == "wrong_training_subject":
        donor = select_donor(ref, [lookup[i] for i in cal["train_ids"]])
    elif pairing == "other_region_EEG":
        donor = neighbors[0] if neighbors else None
    if pairing in ("wrong_training_subject", "other_region_EEG") and donor is None:
        row.update(success=False, converged=False, fit_performed=False, status="matched_donor_unavailable",
                   seconds=time.monotonic() - started)
        write_json(path, row)
        return row
    donor_eeg = raw_target(donor, parent)[0] * cal["eeg_factor"] if donor else None
    if donor:
        row.update(donor_id=donor["id"], donor_subject=donor["subject"], donor_region=donor["region"],
                   shared_eeg_channels=sorted(set(ref["eeg_channels"]) & set(donor["eeg_channels"])))
    fitted_eeg, eeg_visible, support = pairing_eeg(eeg, cfg, pairing, donor_eeg)
    endpoint = endpoint_mask(cfg, support)
    uncertainty_key = arm
    if arm in BASELINES:
        if arm not in cal["uncertainty"][str(prefix)] or cal["uncertainty"][str(prefix)][arm].get("status") != "completed":
            row.update(success=False, converged=False, fit_performed=False, status="baseline_calibration_unavailable")
            write_json(path, row)
            return row
        prediction = baseline_prediction(arm, ref, fitted_eeg, hb, spatial, prefix, cal)
        fit = dict(prediction=prediction, success=bool(np.isfinite(prediction).all()), converged=False,
                   fit_performed=False, status="frozen_training_prediction", failure_log=[])
        row["label_use"] = arm == "task_condition_template_oracle"
    else:
        real_arm = "gamma_selected" if "__" in arm else arm
        spec = model_spec(real_arm, cal, prefix, initial_mode=cell["initial_mode"])
        if pairing != "real" or spec["random_seed"] is not None:
            fitted_modes = mode_cache(ref, fitted_eeg, cal, cfg, run_root, seed=spec["random_seed"],
                                      pairing=pairing, visible=eeg_visible)
        else:
            fitted_modes = modes
        fit = fit_response(hb, fitted_modes, spec, cal, cfg, parameters(cfg, project_root), prefix)
        row["selected_spec"] = spec
        if arm == "best_stable":
            uncertainty_key = spec["arm"]
        row["selected_gamma"] = cal["selected"]["40"]["gamma"]
        row["selected_gain"] = cal["selected"]["40"]["gain"]
    row.update(compact_fit(fit))
    uncertainty = cal["uncertainty"][str(prefix)].get(uncertainty_key, {})
    row["predictive_distribution_status"] = uncertainty.get("status", "unavailable")
    if "prediction" in fit and np.isfinite(fit["prediction"][endpoint]).all():
        row.update(prediction_scores(fit["prediction"], hb, cal["sd_hb"], endpoint,
                                    uncertainty.get("predictive_sd") if uncertainty.get("status") == "completed" else None))
        arrays = {k: fit[k] for k in ("prediction", "physical_prediction", "observation_component", "driver",
                                     "states", "initial_state", "component_coefficients") if k in fit}
        save_arrays(path.with_suffix(".npz"), endpoint_mask=endpoint, **arrays)
        row["arrays"] = str(path.with_suffix(".npz").relative_to(run_root))
    row.update(seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    write_json(path, row)
    return row


def measured_worker(payload):
    cfg, project_root, run_root, item, refs, parent = payload
    mini_plan = dict(windows=[item])
    rows = []
    for cell in measured_cells(cfg, mini_plan):
        try:
            rows.append(measured_cell(cfg, project_root, run_root, cell, refs, parent))
        except Exception as exc:
            path = cell_path(run_root, cell)
            identity = {k: cell["ref"][k] for k in ("id", "dataset", "subject", "record", "site", "region", "task", "condition", "window")}
            row = dict(identity, native_id=item["native_id"], arm=cell["arm"], prefix_steps=cell["prefix_steps"],
                       pairing=cell["pairing"], initial_mode=cell["initial_mode"], success=False, converged=False,
                       fit_performed=False, status="failed_exception", error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc(), evaluations=0)
            write_json(path, row)
            rows.append(row)
    return dict(id=item["ref"]["id"], success=True, status="completed", planned_cells=len(rows),
                successful_cells=sum(bool(r.get("success")) for r in rows),
                failed_cells=sum(not bool(r.get("success")) for r in rows),
                fitted_cells=sum(bool(r.get("fit_performed")) for r in rows),
                converged_fitted_cells=sum(bool(r.get("fit_performed")) and bool(r.get("converged")) for r in rows))


def export_batched_features(plan, run_root):
    results = {}
    batch_root = Path(run_root) / "features" / "batch_v2"
    for split in WINDOW_CAPS:
        items = [x for x in plan["windows"] if x["split"] == split]
        records, e, h, scales, eeg_factors, hb_factors = [], [], [], [], [], []
        for item in items:
            path = Path(run_root) / "features" / split / (item["ref"]["id"] + ".npz")
            if not path.exists():
                raise ValueError("feature export missing: " + item["ref"]["id"])
            with np.load(path, allow_pickle=False) as arrays:
                e.append(arrays["eeg"].copy())
                h.append(arrays["hb"].copy())
                scales.append(arrays["sd_hb"].copy())
            metadata = read_json(path.with_suffix(".json"))
            eeg_factors.append(metadata["eeg_factor"])
            hb_factors.append(metadata["hb_factor"])
            records.append(dict(ref=item["ref"], split=split, native_id=item["native_id"],
                                calibration_key=item["calibration_key"]))
        eeg, target = np.asarray(e), np.asarray(h)
        input_hb, mask_hb = target.copy(), np.ones_like(target, bool)
        input_hb[:, 40:], mask_hb[:, 40:] = np.nan, False
        identity = {"base_id": np.array([x["ref"]["id"] for x in records]),
                    "native_id": np.array([x["native_id"] for x in records]),
                    "dataset": np.array([x["ref"]["dataset"] for x in records]),
                    "subject": np.array([x["ref"]["subject"] for x in records]),
                    "region": np.array([x["ref"]["region"] for x in records]),
                    "eeg_factor": np.asarray(eeg_factors), "hb_factor": np.asarray(hb_factors)}
        path = batch_root / (split + ".npz")
        save_arrays(path, eeg=eeg, hb=input_hb, hb_target=target, sd_hb=np.asarray(scales),
                    eeg_mask=np.ones_like(eeg, bool), hb_mask=mask_hb, **identity)
        full_path = path.with_name(split + "__full_spatial_context.npz")
        save_arrays(full_path, eeg=eeg, hb=target, sd_hb=np.asarray(scales),
                    eeg_mask=np.ones_like(eeg, bool), hb_mask=np.ones_like(target, bool), **identity)
        manifest = dict(schema="semantic_response_discovery_feature_batch_v2", split=split, rows=len(records), records=records,
                        primary_encoder_file=str(path.relative_to(run_root)),
                        spatial_context_encoder_file=str(full_path.relative_to(run_root)),
                        hb_target_role="scoring_only_never_encoder_input", primary_Hb_visibility="first_40_steps_only_suffix_NaN",
                        spatial_boundary="full_Hb_encodings_may_enter_only_other_region_context_not_target_region",
                        normalization="dataset_region_new_train_only_scalar_gains; no_external_or_per_window_refit")
        write_json(path.with_suffix(".json"), manifest)
        results[split] = dict(rows=len(records), path=str(path), full_spatial_context_path=str(full_path))
    write_json(batch_root / "manifest.json", dict(schema="semantic_response_discovery_feature_exports_v2", splits=results,
               source="retained_per_row_features; v1_batches_preserved; per_row_scalar_factors_added"))
    return results


def synthetic_case(cfg, scenario, family, repeat, partition, project_root=CODE_ROOT):
    from src.inference.semantic_response_synthetic import generate_response_case
    stream = cfg["synthetic"]["evaluation_seed_stream"] if partition == "evaluation" else cfg["synthetic"]["calibration_seed_stream"]
    amplitudes, durations = cfg["synthetic"]["amplitudes"], cfg["synthetic"]["durations_s"]
    return generate_response_case(cfg["seed"] + stream, scenario, family, partition=partition,
             base_id=f"response_mechanism_{repeat:03d}", parameters=parameters(cfg, project_root),
             amplitude=amplitudes[repeat % len(amplitudes)],
             duration=durations[(repeat // len(amplitudes)) % len(durations)], hb_operator=operators()["hb"])


def synthetic_records(cfg, family, partition, project_root=CODE_ROOT):
    count = cfg["synthetic"]["repeats"] if partition == "evaluation" else cfg["synthetic"]["calibration_repeats"]
    records = []
    for repeat in range(count):
        for scenario in cfg["synthetic"]["scenarios"]:
            case = synthetic_case(cfg, scenario, family, repeat, partition, project_root)
            for variant in ("control", "intervention"):
                truth = case[variant]
                records.append(dict(id=f"{scenario}__r{repeat:03d}__{variant}", subject=f"base_{repeat:03d}",
                                    scenario=scenario, repeat=repeat, variant=variant, truth=truth))
    return records


def synthetic_selection_cell(record, spec, prefix, cal, cfg, p, run_root):
    path = Path(run_root) / "synthetic_selection" / cal["family"] / parameter_slug(spec) / record["id"] / f"prefix_{prefix}.json"
    if path.exists():
        return read_json(path)
    truth = record["truth"]
    modes = infer_modes(truth["eeg"], cal, cfg, seed=spec["random_seed"])
    fit = fit_response(truth["hb"], modes, spec, cal, cfg, p, prefix)
    row = dict(id=record["id"], subject=record["subject"], scenario=record["scenario"], prefix_steps=prefix,
               **compact_fit(fit))
    if "prediction" in fit:
        row.update(prediction_scores(fit["prediction"], truth["hb"], cal["sd_hb"], endpoint_mask(cfg)))
        save_arrays(path.with_suffix(".npz"), prediction=fit["prediction"])
        row["arrays"] = str(path.with_suffix(".npz").relative_to(run_root))
    write_json(path, row)
    return row


def synthetic_calibration_worker(payload):
    cfg, project_root, run_root, family = payload
    path = Path(run_root) / "synthetic_calibration" / (family + ".json")
    if path.exists():
        return read_json(path)
    started = time.monotonic()
    training = synthetic_records(cfg, family, "train", project_root)
    selection = synthetic_records(cfg, family, "selection", project_root)
    p = parameters(cfg, project_root)
    from src.inference.semantic_response_ssm import fixed_driver_response
    eeg, hb = np.array([x["truth"]["eeg"] for x in training]), np.array([x["truth"]["hb"] for x in training])
    sd_eeg, sd_hb = np.maximum(eeg.std(axis=(0, 1)), 1e-8), np.maximum(hb.std(axis=(0, 1)), 1e-8)
    cal = dict(family=family, key="synthetic_" + family, eeg_factor=1., hb_factor=1., sd_eeg=sd_eeg,
               sd_hb=sd_hb, fit_eeg_sd=sd_eeg, fit_hb_sd=sd_hb, random_seeds=cfg["model"]["random_seeds"],
               component_sd=np.full(4, .02), selected={"40": dict(gamma=0., gain=1., tau_n=0., tau_v=0., random_gamma={})},
               training_ids=[r["id"] for r in training], selection_ids=[r["id"] for r in selection],
               coordinate="known_generator_model_coordinates_no_data_dependent_amplitude_rescaling")
    coefficients = []
    design = operators()["common"].reshape(240, 4) / np.tile(sd_hb, 120)[:, None]
    for record in training:
        truth = record["truth"]
        modes = infer_modes(truth["eeg"], cal, cfg)
        physical = fixed_driver_response(modes["a"], p, .25, hb_operator=operators()["hb"],
                    substeps=cfg["solver"]["substeps"], numerical_backend=cfg["solver"]["numerical_backend"])
        coefficients.append(np.linalg.lstsq(design, ((truth["hb"] - physical["prediction"]) / sd_hb).ravel(), rcond=1e-10)[0])
    cal["component_sd"] = np.maximum(np.sqrt(np.mean(np.asarray(coefficients) ** 2, axis=0)), sd_hb.mean() * 1e-6)
    base = dict(arm="selection_candidate", gamma=0., gain=1., tau_n=0., tau_v=0., random_seed=None, initial_mode="tied_pv_logprior")
    rankings = {}
    candidate_sets = [("gamma", cfg["model"]["gamma_candidates"], None),
                      ("gain", cfg["model"]["gain_candidates"], None)]
    candidate_sets += [("gamma", cfg["model"]["gamma_candidates"], seed) for seed in cfg["model"]["random_seeds"]]
    candidate_sets += [(key, cfg["model"][key + "_candidates"], None) for key in ("tau_n", "tau_v")]
    for key, candidates, random_seed in candidate_sets:
        rows = []
        for candidate in candidates:
            spec = dict(base, random_seed=random_seed, **{key: candidate})
            if key in ("tau_n", "tau_v"):
                spec["gain"] = cal["selected"]["40"]["gain"]
            for record in selection:
                row = synthetic_selection_cell(record, spec, 40, cal, cfg, p, run_root)
                rows.append(dict(row, **{key: candidate}))
        ranking = subject_equal_ranking(rows, candidates, key, cfg["calibration"]["failure_penalty"])
        name = key if random_seed is None else f"random_{cfg['model']['random_seeds'].index(random_seed)}_gamma"
        rankings[name] = ranking
        if random_seed is None:
            cal["selected"]["40"][key] = ranking[0][key]
        else:
            cal["selected"]["40"]["random_gamma"][str(cfg["model"]["random_seeds"].index(random_seed))] = ranking[0][key]
    cal["selected"]["40"]["best_stable"] = ("input_lowpass" if rankings["tau_n"][0]["score"] <= rankings["tau_v"][0]["score"] else "viscoelastic_outflow")
    cal["selected"]["0"] = dict(cal["selected"]["40"])
    cal["selection_rankings"], cal["uncertainty"] = rankings, {}
    for prefix in cfg["measured"]["prefix_steps"]:
        cal["uncertainty"][str(prefix)] = {}
        for arm in ARMS:
            spec = model_spec(arm, cal, prefix)
            residuals = []
            for record in selection:
                row = synthetic_selection_cell(record, spec, prefix, cal, cfg, p, run_root)
                if row.get("arrays"):
                    with np.load(Path(run_root) / row["arrays"], allow_pickle=False) as arrays:
                        prediction = arrays["prediction"].copy()
                    residuals.append((row, prediction - record["truth"]["hb"], endpoint_mask(cfg)))
            cal["uncertainty"][str(prefix)][arm] = calibrate_predictive_sd(residuals, sd_hb, .05)
    cal.update(status="completed", success=True, seconds=time.monotonic() - started,
               training_base_identities=cfg["synthetic"]["calibration_repeats"],
               selection_base_identities=cfg["synthetic"]["calibration_repeats"],
               selection_criterion="base_identity_equal_selection_prefix40_training_SD_NRMSE",)
    write_json(path, cal)
    return cal


def response_function_metrics(fit, truth, spec, cfg, p):
    """Fixed probes compare baseline-subtracted functional response, not a label.

    The same probe amplitudes and durations enter every mechanism. Unknown
    observation gain and non-rest initials remain explicitly conditional.
    """
    from src.inference.semantic_response_ssm import fixed_driver_response
    metadata = truth["metadata"]
    true_p = BalloonParameters(fixed=BalloonFixedParameters(**metadata["parameters"]["fixed"]),
                              free=BalloonFreeParameters(**metadata["parameters"]["free"]))
    truth_initial = np.asarray(metadata["initial_state"])
    fitted_initial = np.asarray(fit.get("initial_state", [0., 1., 1., 1., 1.]))
    zeros = np.zeros(120)
    forward_kwargs = dict(hb_operator=operators()["hb"], substeps=cfg["solver"]["substeps"],
                          numerical_backend=cfg["solver"]["numerical_backend"])
    truth_null = fixed_driver_response(zeros, true_p, .25, initial_state=truth_initial,
                                      tau_n=metadata["tau_n"], tau_v=metadata["tau_v"], **forward_kwargs)["prediction"]
    fitted_null = fixed_driver_response(zeros, p, .25, initial_state=fitted_initial,
                                       tau_n=spec["tau_n"], tau_v=spec["tau_v"], **forward_kwargs)["prediction"]
    predicted, targets, probes = [], [], []
    for amplitude, duration in zip(cfg["synthetic"]["amplitudes"], cfg["synthetic"]["durations_s"]):
        probe = np.zeros(120)
        time_s = np.arange(120) * .25
        probe[(time_s >= 5.) & (time_s < 5. + duration)] = amplitude
        target_input = zeros if metadata["true_hb_driver_route"].startswith("zero") else probe
        target = fixed_driver_response(target_input, true_p, .25, initial_state=truth_initial,
                                      tau_n=metadata["tau_n"], tau_v=metadata["tau_v"], **forward_kwargs)["prediction"] - truth_null
        if spec["gamma"]:
            input_probe = probe / np.sqrt(1 + spec["gamma"] ** 2)
        else:
            input_probe = spec["gain"] * probe
        prediction = fixed_driver_response(input_probe, p, .25, initial_state=fitted_initial,
                                          tau_n=spec["tau_n"], tau_v=spec["tau_v"], **forward_kwargs)["prediction"] - fitted_null
        predicted.append(prediction)
        targets.append(target * metadata["hb_coordinate_gain"])
        probes.append(dict(amplitude=amplitude, duration_s=duration, onset_s=5.))
    prediction, target = np.asarray(predicted), np.asarray(targets)
    return dict(response_function_RMSE=float(np.sqrt(np.mean((prediction - target) ** 2))),
                response_function_relative_RMSE=float(np.sqrt(np.mean((prediction - target) ** 2)) / max(np.sqrt(np.mean(target ** 2)), 1e-8)),
                response_function_probes=probes,
                response_function_convention="increment_from_no_input_same_initial_condition; conditional_model_function_not_real_intervention",
                truth_response_mechanism=dict(tau=metadata["parameters"]["free"]["tau"],
                     kappa=metadata["parameters"]["free"]["kappa"], tau_n=metadata["tau_n"], tau_v=metadata["tau_v"])), prediction, target


def synthetic_specs(cfg):
    return [dict(family=family, scenario=scenario, repeat=repeat, variant=variant, arm=arm, prefix_steps=prefix)
            for family in cfg["synthetic"]["families"] for scenario in cfg["synthetic"]["scenarios"]
            for repeat in range(cfg["synthetic"]["repeats"]) for variant in ("control", "intervention")
            for arm in ARMS for prefix in cfg["measured"]["prefix_steps"]]


def synthetic_path(run_root, spec):
    return (Path(run_root) / "synthetic" / spec["family"] / spec["scenario"] / f'r{spec["repeat"]:03d}'
            / spec["variant"] / spec["arm"] / f'prefix_{spec["prefix_steps"]}.json')


def synthetic_worker(payload):
    cfg, project_root, run_root, case_spec = payload
    family, scenario, repeat = case_spec["family"], case_spec["scenario"], case_spec["repeat"]
    cal = read_json(Path(run_root) / "synthetic_calibration" / (family + ".json"))
    case = synthetic_case(cfg, scenario, family, repeat, "evaluation", project_root)
    p = parameters(cfg, project_root)
    rows = []
    for variant in ("control", "intervention"):
        truth = case[variant]
        modes = {None: infer_modes(truth["eeg"], cal, cfg)}
        for seed in cfg["model"]["random_seeds"]:
            modes[seed] = infer_modes(truth["eeg"], cal, cfg, seed=seed)
        for arm in ARMS:
            for prefix in cfg["measured"]["prefix_steps"]:
                spec = dict(case_spec, variant=variant, arm=arm, prefix_steps=prefix)
                path = synthetic_path(run_root, spec)
                if path.exists():
                    rows.append(read_json(path))
                    continue
                started = time.monotonic()
                selected = model_spec(arm, cal, prefix)
                inferred = modes[selected["random_seed"]]
                fit = fit_response(truth["hb"], inferred, selected, cal, cfg, p, prefix)
                row = dict(spec, base_id=truth["metadata"]["base_id"], truth_metadata=truth["metadata"],
                           selected_spec=selected, generated_valid=bool(truth["valid"]), **compact_fit(fit))
                centered = np.asarray(inferred["a"]) - np.asarray(inferred["a"])[:20].mean()
                row["a_reference_centered_RMSE"] = float(np.sqrt(np.mean((centered - truth["a_reference_centered"]) ** 2)))
                row["a_relative_shape_RMSE"] = float(np.sqrt(np.mean((centered / max(np.sqrt(np.mean(centered ** 2)), 1e-12) - truth["source_shape"]) ** 2)))
                row["log_amplitude_error"] = float(np.log(max(np.sqrt(np.mean(centered ** 2)), 1e-12)) - truth["log_amplitude"])
                if "prediction" in fit:
                    uncertainty = cal["uncertainty"][str(prefix)][arm]
                    row.update(prediction_scores(fit["prediction"], truth["hb"], cal["sd_hb"], endpoint_mask(cfg), uncertainty.get("predictive_sd")))
                    arrays = {k: fit[k] for k in ("prediction", "physical_prediction", "observation_component", "states", "driver", "initial_state") if k in fit}
                    if fit.get("success") and arm in ("gain_selected", "input_lowpass", "viscoelastic_outflow"):
                        try:
                            metrics, predicted_response, true_response = response_function_metrics(fit, truth, selected, cfg, p)
                            row.update(metrics)
                            arrays.update(response_function_prediction=predicted_response, response_function_truth=true_response)
                        except Exception as exc:
                            row["response_function_status"] = "failed_exception"
                            row["response_function_error"] = f"{type(exc).__name__}: {exc}"
                    if "states" in fit and truth["metadata"]["state_truth_available"]:
                        row["volume_RMSE"] = float(np.sqrt(np.mean((fit["states"][:, 3] - truth["states"][:, 3]) ** 2)))
                        row["deoxy_RMSE"] = float(np.sqrt(np.mean((fit["states"][:, 5] - truth["states"][:, 5]) ** 2)))
                    save_arrays(path.with_suffix(".npz"), a_inferred=inferred["a"], b_inferred=inferred["b"],
                                a_truth=truth["a"], Hb_truth=truth["hb"], **arrays)
                    row["arrays"] = str(path.with_suffix(".npz").relative_to(run_root))
                row["seconds"] = time.monotonic() - started
                write_json(path, row)
                rows.append(row)
    return dict(success=True, status="completed", **case_spec, planned_cells=len(rows),
                successful_cells=sum(bool(r.get("success")) for r in rows),
                failed_cells=sum(not bool(r.get("success")) for r in rows),
                generated_valid=all(r.get("generated_valid") for r in rows))


def summary_table(frame, groups, metrics=METRICS):
    rows = []
    for keys, group in frame.groupby(groups, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        success = group["success"].fillna(False).astype(bool)
        fitted = group.get("fit_performed", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        converged = group.get("converged", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        row = dict(zip(groups, keys), planned=len(group), successful=int(success.sum()), failed=int((~success).sum()),
                   fitted=int(fitted.sum()), converged_fitted=int((fitted & converged).sum()),
                   prior_or_frozen_predictions=int((success & ~fitted).sum()))
        good = group[success].copy()
        owner = "subject" if "subject" in good else "base_id"
        for metric in metrics:
            if metric not in good:
                continue
            good["_value"] = pd.to_numeric(good[metric], errors="coerce")
            values = good.groupby(owner)._value.mean().dropna() if owner in good else good._value.dropna()
            row[metric] = float(values.mean()) if len(values) else None
            row[metric + "_independent_units"] = len(values)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_rows(frame, *, controls, groups, metric="Hb_nrmse", seed=20261002, bootstrap_repeats=1000):
    """All planned identities remain denominators; only common successes score."""
    results = []
    rng = np.random.default_rng(seed)
    for keys, group in frame.groupby(groups, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for control in controls:
            c = group[group.arm == control].copy()
            if c.empty:
                continue
            for arm in sorted(set(group.arm) - {control}):
                a = group[group.arm == arm].copy()
                merged = a.merge(c, on="id", how="outer", suffixes=("_arm", "_control"), validate="one_to_one")
                av = pd.to_numeric(merged.get(metric + "_arm", pd.Series(np.nan, index=merged.index)), errors="coerce")
                cv = pd.to_numeric(merged.get(metric + "_control", pd.Series(np.nan, index=merged.index)), errors="coerce")
                success = merged.success_arm.eq(True) & merged.success_control.eq(True)
                good = merged[success & np.isfinite(av) & np.isfinite(cv)].copy()
                good["gain"] = cv.loc[good.index] - av.loc[good.index]
                units = good.groupby("subject_arm").gain.mean().dropna()
                bootstrap = rng.choice(units.to_numpy(), (bootstrap_repeats, len(units)), replace=True).mean(axis=1) if len(units) else np.array([])
                interval = np.quantile(bootstrap, [.025, .975]) if len(bootstrap) else [np.nan, np.nan]
                results.append(dict(zip(groups, keys), arm=arm, control=control, metric=metric,
                                    planned_pairs=len(merged), common_success_pairs=len(good),
                                    failed_or_missing_pairs=len(merged) - len(good), subjects=len(units),
                                    gain=float(units.mean()) if len(units) else None, ci_low=interval[0], ci_high=interval[1],
                                    positive_gain="arm_better", interval="subject_block_1000_fixed_training_selection_exploratory"))
    return results


def null_rows(frame, cfg):
    result = []
    for arm in cfg["measured"]["null_arms"]:
        for pairing in ("wrong_training_subject", "nonwrapping_shift_12s", "other_region_EEG"):
            null = frame[(frame.arm == arm) & (frame.pairing == pairing)].copy()
            control_pairing = "real_shift_support" if pairing == "nonwrapping_shift_12s" else "real"
            matched_real = frame[(frame.arm == arm) & (frame.pairing == control_pairing)].copy()
            if null.empty or matched_real.empty:
                continue
            null["arm"], matched_real["arm"] = "null", "real_control"
            compared = paired_rows(pd.concat([null, matched_real]), controls=["real_control"],
                                  groups=["dataset", "prefix_steps"], metric="Gaussian_CRPS", seed=cfg["seed"])
            for row in compared:
                row.update(arm=arm, null_pairing=pairing, control_pairing=control_pairing,
                           positive_gain="null_better_negative_gain_supports_real_pairing")
                result.append(row)
    return result


def primary_null_specificity(frame, cfg):
    """Frozen nine-test exploratory NRMSE family; full support is needed to pass.

    Subject signs exchange under a symmetric-difference null conditional on
    fixed training/selection. Dataset means receive equal weights. All regions
    and windows of a subject remain in one sign block.
    """
    rows = []
    subset = frame[frame.prefix_steps == 40]
    expected_pairs = sum(SPLIT_COUNTS[d]["evaluation"] for d in DATASETS) * 8 * len(REGIONS)
    for arm in cfg["measured"]["null_arms"]:
        for pairing in ("wrong_training_subject", "nonwrapping_shift_12s", "other_region_EEG"):
            control_pairing = "real_shift_support" if pairing == "nonwrapping_shift_12s" else "real"
            a = subset[(subset.arm == arm) & (subset.pairing == pairing)].copy()
            c = subset[(subset.arm == arm) & (subset.pairing == control_pairing)].copy()
            if a.empty and c.empty:
                merged = pd.DataFrame()
            else:
                merged = a.merge(c, on="id", how="outer", suffixes=("_null", "_real"), validate="one_to_one")
            units, valid_count = [], 0
            counts = {dataset: 0 for dataset in DATASETS}
            if len(merged):
                av = pd.to_numeric(merged.get("Hb_nrmse_null", pd.Series(np.nan, index=merged.index)), errors="coerce")
                cv = pd.to_numeric(merged.get("Hb_nrmse_real", pd.Series(np.nan, index=merged.index)), errors="coerce")
                valid = merged.success_null.eq(True) & merged.success_real.eq(True) & np.isfinite(av) & np.isfinite(cv)
                good = merged[valid].copy()
                good["difference"] = av.loc[good.index] - cv.loc[good.index]
                valid_count = len(good)
                for dataset in DATASETS:
                    subject = good[good.dataset_real == dataset].groupby("subject_real").difference.mean().dropna()
                    counts[dataset] = len(subject)
                    for identity, value in subject.items():
                        units.append(dict(dataset=dataset, subject=identity, difference=float(value), weight=1 / (3 * len(subject))))
            all_datasets = all(counts.values())
            observed = p_value = None
            if units and all_datasets:
                weighted = np.array([u["difference"] * u["weight"] for u in units])
                signs = 1 - 2 * ((np.arange(2 ** len(units), dtype=np.uint64)[:, None]
                                 >> np.arange(len(units), dtype=np.uint64)) & 1).astype(np.int8)
                statistics = signs @ weighted
                observed = float(statistics[0])
                p_value = float(np.mean(statistics >= statistics[0]))
            full_inventory = len(units) == 15 and counts == {d: SPLIT_COUNTS[d]["evaluation"] for d in DATASETS}
            eligible = bool(full_inventory and valid_count == expected_pairs and len(merged) == expected_pairs)
            rows.append(dict(arm=arm, null_pairing=pairing, real_pairing=control_pairing, prefix_steps=40,
                             metric="Hb_nrmse", planned_pairs=len(merged), expected_pairs=expected_pairs,
                             common_success_pairs=valid_count, failed_or_missing_pairs=len(merged) - valid_count,
                             subjects=len(units), subjects_by_dataset=counts, dataset_equal_gain=observed,
                             positive_gain="real_better_null_minus_real", p_one_sided_sign_flip=p_value,
                             sign_patterns=2 ** len(units) if all_datasets else 0,
                             pass_eligible=eligible, eligibility_reason=None if eligible else "incomplete_successful_paired_support_or_subject_inventory",
                             family="prefix40_three_arms_times_three_pairings_nine_tests",
                             inference="exact_subject_sign_flip_under_symmetric_difference_null_conditional_fixed_training_selection_exploratory",
                             subject_differences=units))
    ordered = sorted(range(len(rows)), key=lambda i: (1. if rows[i]["p_one_sided_sign_flip"] is None else rows[i]["p_one_sided_sign_flip"], i))
    previous = 0.
    for rank, i in enumerate(ordered):
        p_value = rows[i]["p_one_sided_sign_flip"]
        adjusted = max(previous, min(1., (len(rows) - rank) * (1. if p_value is None else p_value)))
        previous = adjusted
        rows[i]["p_Holm_nine"] = adjusted if p_value is not None else None
        rows[i]["null_specificity_pass"] = bool(rows[i]["pass_eligible"] and p_value is not None
                                               and adjusted <= .05 and rows[i]["dataset_equal_gain"] > 0)
    return rows


def dataset_equal_pairs(frame, *, controls, metric, seed):
    rows, rng = [], np.random.default_rng(seed)
    for prefix in sorted(set(frame.prefix_steps)):
        group = frame[frame.prefix_steps == prefix]
        for control in controls:
            for arm in sorted(set(group.arm) - {control}):
                a, c = group[group.arm == arm], group[group.arm == control]
                if a.empty or c.empty:
                    continue
                merged = a.merge(c, on="id", how="outer", suffixes=("_arm", "_control"), validate="one_to_one")
                av = pd.to_numeric(merged.get(metric + "_arm", pd.Series(np.nan, index=merged.index)), errors="coerce")
                cv = pd.to_numeric(merged.get(metric + "_control", pd.Series(np.nan, index=merged.index)), errors="coerce")
                valid = merged.success_arm.eq(True) & merged.success_control.eq(True) & np.isfinite(av) & np.isfinite(cv)
                good = merged[valid].copy()
                good["gain"] = cv.loc[good.index] - av.loc[good.index]
                dataset_means, distributions, subjects = [], [], 0
                for dataset in DATASETS:
                    units = good[good.dataset_arm == dataset].groupby("subject_arm").gain.mean().dropna()
                    if len(units):
                        dataset_means.append(float(units.mean()))
                        distributions.append(rng.choice(units.to_numpy(), (1000, len(units)), replace=True).mean(axis=1))
                        subjects += len(units)
                full = len(dataset_means) == len(DATASETS)
                bootstrap = np.mean(distributions, axis=0) if full else np.array([])
                interval = np.quantile(bootstrap, [.025, .975]) if len(bootstrap) else [np.nan, np.nan]
                rows.append(dict(prefix_steps=prefix, arm=arm, control=control, metric=metric,
                                 planned_pairs=len(merged), common_success_pairs=len(good), failed_or_missing_pairs=len(merged) - len(good),
                                 subjects=subjects, datasets=len(dataset_means), gain=float(np.mean(dataset_means)) if full else None,
                                 ci_low=interval[0], ci_high=interval[1], positive_gain="arm_better",
                                 interval="dataset_equal_subject_block_1000_fixed_objects_exploratory"))
    return rows


def summarize_synthetic(cfg, run_root):
    rows, missing = [], []
    specs = synthetic_specs(cfg)
    for spec in specs:
        path = synthetic_path(run_root, spec)
        if path.exists():
            rows.append(read_json(path))
        else:
            missing.append(spec)
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(Path(run_root) / "synthetic_cells.csv", index=False)
        summary_table(frame, ["family", "scenario", "variant", "prefix_steps", "arm"],
                      METRICS + ("a_reference_centered_RMSE", "a_relative_shape_RMSE", "log_amplitude_error",
                                 "volume_RMSE", "deoxy_RMSE", "response_function_RMSE", "response_function_relative_RMSE")).to_csv(Path(run_root) / "synthetic_summary.csv", index=False)
        mechanism = frame[(frame.variant == "intervention") & frame.arm.isin(["gain_selected", "input_lowpass", "viscoelastic_outflow"])]
        summary_table(mechanism, ["family", "scenario", "prefix_steps", "arm"],
                      ("Hb_nrmse", "Gaussian_CRPS", "response_function_RMSE", "response_function_relative_RMSE")).to_csv(Path(run_root) / "cross_mechanism_fit_matrix.csv", index=False)
        paired = []
        for keys, group in mechanism.groupby(["family", "prefix_steps", "arm"]):
            control = frame[(frame.family == keys[0]) & (frame.prefix_steps == keys[1]) & (frame.arm == keys[2]) & (frame.variant == "control")]
            for _, row in group.iterrows():
                original = control[(control["scenario"] == row["scenario"]) & (control["repeat"] == row["repeat"])]
                if len(original) != 1 or not row.get("arrays") or not original.iloc[0].get("arrays"):
                    continue
                with np.load(Path(run_root) / row["arrays"], allow_pickle=False) as after, np.load(Path(run_root) / original.iloc[0]["arrays"], allow_pickle=False) as before:
                    change = after["prediction"] - before["prediction"]
                    true_change = after["Hb_truth"] - before["Hb_truth"]
                paired.append(dict(family=keys[0], prefix_steps=keys[1], arm=keys[2], scenario=row["scenario"], repeat=int(row["repeat"]),
                                   counterfactual_response_change_RMSE=float(np.sqrt(np.mean((change - true_change) ** 2))),
                                   true_observation_change_RMS=float(np.sqrt(np.mean(true_change ** 2))),
                                   predicted_observation_change_RMS=float(np.sqrt(np.mean(change ** 2))),
                                   common_success=bool(row["success"]) and bool(original.iloc[0]["success"])))
        pd.DataFrame(paired).to_csv(Path(run_root) / "synthetic_counterfactual_response.csv", index=False)
    result = dict(schema="semantic_response_discovery_synthetic_summary_v1", planned=len(specs), recorded=len(rows),
                  missing=len(missing), successful=sum(bool(r.get("success")) for r in rows),
                  fitted=sum(bool(r.get("fit_performed")) for r in rows),
                  converged_fitted=sum(bool(r.get("fit_performed")) and bool(r.get("converged")) for r in rows),
                  generated_valid=sum(bool(r.get("generated_valid")) for r in rows),
                  execution="completed" if not missing else "incomplete")
    write_json(Path(run_root) / "synthetic_summary.json", result)
    return result


def overall_execution_status(ssm_execution, run_root):
    """Reference the tokenizer owner; do not duplicate its counts or verdict."""
    path = Path(run_root) / "tokenizer" / "summary.json"
    tokenizer_completed = False
    if path.exists():
        tokenizer = read_json(path)
        tokenizer_completed = (tokenizer.get("schema") == "semantic_response_tokenizer_suite_v1"
                               and tokenizer.get("status") == "complete"
                               and tokenizer.get("public_probe_status") == "complete")
    return dict(ssm_execution=ssm_execution, tokenizer_summary_path="tokenizer/summary.json",
                tokenizer_execution="completed" if tokenizer_completed else "incomplete",
                execution="completed" if ssm_execution == "completed" and tokenizer_completed else "incomplete")


def summarize(cfg, run_root):
    root = Path(run_root)
    result = dict(schema="semantic_response_discovery_summary_v1", experiment_id=cfg["experiment_id"],
                  completed_at=datetime.now(timezone.utc).isoformat(), synthetic=summarize_synthetic(cfg, root),
                  scope="previously_exposed_public_cohort_new_exploratory_endpoint; conditional_generator_functions",
                  limitations=["offline_full_window_processing_and_DCT_not_causal_native_prediction",
                               "three_regions_not_independent; coarse_geometry_and_shared_EEG_electrodes",
                               "new_subject_split_is_not_an_independent_confirmation_cohort",
                               "predictive_SD_is_selection_calibrated_not_physiological_posterior",
                               "Simultaneous_selection_has_no_2back_and_evaluation_DSR_is_five_windows_two_subjects"])
    rows, expected = [], []
    if (root / "measured_plan.json").exists():
        plan = read_json(root / "measured_plan.json")
        expected = measured_cells(cfg, plan)
        for cell in expected:
            path = cell_path(root, cell)
            if path.exists():
                rows.append(read_json(path))
            else:
                ref = cell["ref"]
                rows.append(dict(id=ref["id"], dataset=ref["dataset"], subject=ref["subject"], region=ref["region"],
                                 arm=cell["arm"], pairing=cell["pairing"], prefix_steps=cell["prefix_steps"],
                                 success=False, converged=False, fit_performed=False, status="not_recorded"))
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(root / "measured_cells.csv", index=False)
        summary_table(frame, ["dataset", "prefix_steps", "pairing", "arm"]).to_csv(root / "measured_summary.csv", index=False)
        summary_table(frame, ["dataset", "region", "prefix_steps", "pairing", "arm"]).to_csv(root / "regional_summary.csv", index=False)
        real = frame[frame.pairing == "real"]
        controls = ["gamma_attenuation", "gain_selected", "own_Hb_prefix_ridge", "full_EEG_ridge", "spatial_Hb_ridge", "spatial_filterbank_train_permutation"]
        comparisons, overall = [], []
        for metric in ("Hb_nrmse", "Gaussian_CRPS", "Gaussian_NLL", "Gaussian_CRPS_normalized"):
            comparisons.extend(paired_rows(real, controls=controls, groups=["dataset", "prefix_steps"], metric=metric, seed=cfg["seed"]))
            overall.extend(dataset_equal_pairs(real, controls=controls, metric=metric, seed=cfg["seed"]))
        pd.DataFrame(comparisons).to_csv(root / "paired_comparisons.csv", index=False)
        pd.DataFrame(overall).to_csv(root / "dataset_equal_comparisons.csv", index=False)
        pd.DataFrame(null_rows(frame, cfg)).to_csv(root / "paired_nulls.csv", index=False)
        primary_nulls = primary_null_specificity(frame, cfg)
        pd.DataFrame(primary_nulls).to_csv(root / "primary_null_specificity.csv", index=False)
        write_json(root / "primary_null_specificity.json", dict(schema="semantic_response_primary_null_specificity_v1",
                   family_size=9, rows=primary_nulls, metric="Hb_nrmse", prefix_steps=40,
                   full_paired_support_required=True, exploratory=True))
        random = real[real.arm.isin(["random_seed_0", "random_seed_1", "random_seed_2"])].copy()
        if len(random):
            averaged = []
            for keys, group in random.groupby(["id", "dataset", "subject", "prefix_steps"]):
                successful = len(group) == 3 and bool(group.success.all())
                row = dict(zip(["id", "dataset", "subject", "prefix_steps"], keys), arm="predefined_random_seed_average",
                           success=successful, seeds_planned=3, seeds_successful=int(group.success.sum()))
                for metric in METRICS:
                    if metric in group:
                        values = pd.to_numeric(group[metric], errors="coerce")
                        row[metric] = float(values.mean()) if successful and values.notna().all() else None
                averaged.append(row)
            summary_table(pd.DataFrame(averaged), ["dataset", "prefix_steps", "arm"]).to_csv(root / "random_direction_average.csv", index=False)
        result["measured"] = dict(planned=len(expected), recorded=int((frame.status != "not_recorded").sum()),
                                   successful=int(frame.success.sum()), failed=int((~frame.success.astype(bool)).sum()),
                                   fitted=int(frame.fit_performed.sum()), converged_fitted=int((frame.fit_performed & frame.converged).sum()),
                                   subjects=15, native_windows=120, regional_rows=360)
    calibrations = [root / "calibration" / (f"{d}__{r}.json") for d in DATASETS for r in REGIONS]
    result["calibration"] = dict(planned=9, recorded=sum(p.exists() for p in calibrations),
                                 completed=sum(p.exists() and read_json(p).get("status") == "completed" for p in calibrations))
    ssm_execution = ("completed" if result["synthetic"]["execution"] == "completed"
                     and result["calibration"]["recorded"] == 9 and result.get("measured", {}).get("recorded") == len(expected)
                     and bool(expected) else "incomplete")
    result.update(overall_execution_status(ssm_execution, root))
    write_json(root / "summary.json", result)
    return result


def dry_run(cfg, run_root, project_root=CODE_ROOT):
    """Small synthetic validity check, without any public measured payload read."""
    from src.inference.semantic_response_synthetic import generate_response_case
    p = parameters(cfg, project_root)
    case = generate_response_case(cfg["seed"] + 31, "standard", "restricted", partition="train",
                                  base_id="runner_tensor_fixture", parameters=p, amplitude=.025, duration=8.,
                                  hb_operator=operators()["hb"])
    truth = case["control"]
    sd_eeg = np.maximum(truth["eeg"].std(axis=0), 1e-8)
    sd_hb = np.maximum(truth["hb"].std(axis=0), 1e-8)
    cal = dict(sd_eeg=sd_eeg, sd_hb=sd_hb, fit_eeg_sd=sd_eeg, fit_hb_sd=sd_hb,
               component_sd=np.full(4, .005), random_seeds=cfg["model"]["random_seeds"],
               selected={str(prefix): dict(gamma=0., gain=1., tau_n=0., tau_v=0.,
                         random_gamma={str(i): 0. for i in range(3)}, best_stable="input_lowpass")
                         for prefix in cfg["measured"]["prefix_steps"]})
    modes = infer_modes(truth["eeg"], cal, cfg)
    checks = []
    for prefix in cfg["measured"]["prefix_steps"]:
        for arm in ("broad_a", "zero_driver", "input_lowpass", "viscoelastic_outflow"):
            spec = model_spec(arm, cal, prefix)
            fit = fit_response(truth["hb"], modes, spec, cal, cfg, p, prefix)
            poisoned = truth["hb"].copy()
            poisoned[prefix:] = np.nan
            alternative = fit_response(poisoned, modes, spec, cal, cfg, p, prefix)
            same = ("prediction" in fit and "prediction" in alternative
                    and np.array_equal(fit["prediction"], alternative["prediction"]))
            checks.append(dict(arm=arm, prefix_steps=prefix, hidden_suffix_invariant=same, **compact_fit(fit)))
    valid = (bool(truth["valid"]) and modes["r"].shape == (120, 2) and np.isfinite(modes["r"]).all()
             and all(r["hidden_suffix_invariant"] for r in checks)
             and all(r["success"] for r in checks if r["prefix_steps"] == 0))
    result = dict(schema="semantic_response_discovery_dry_run_v1", execution="completed" if valid else "failed",
                  fixture_valid=valid, public_payloads_read=0, tensor=dict(EEG=[120, 30], Hb=[120, 2], modes=[120, 2]),
                  checks=checks, interpretation="ordinary_synthetic_contract_validation_not_data_authorization")
    write_json(Path(run_root) / "dry_run.json", result)
    if not valid:
        raise ValueError("synthetic tensor or hidden-value invariance check failed")
    return result


def require_synthetic_fixture(run_root):
    result = read_json(Path(run_root) / "dry_run.json")
    if (result.get("schema") != "semantic_response_discovery_dry_run_v1"
            or result.get("execution") != "completed" or not result.get("fixture_valid")
            or result.get("public_payloads_read") != 0):
        raise ValueError("completed valid synthetic fixture required before public payload work")


def pilot_worker(payload):
    cfg, project_root, run_root, spec, refs, parent = payload
    path = Path(run_root) / "pilot" / (spec["key"] + ".json")
    if path.exists():
        return read_json(path)
    started = time.monotonic()
    lookup = {r["id"]: r for r in refs}
    train = [lookup[i] for i in spec["train_ids"]]
    raw = [raw_target(ref, parent) for ref in train]
    source_cfg = yaml.safe_load((Path(project_root) / cfg["source_config"]).read_text())
    cal = dict(key=spec["key"], **fit_coordinates(np.array([x[0] for x in raw]), np.array([x[1] for x in raw]), cfg, source_cfg),
               component_sd=np.full(4, .02), pilot_common_prior="provisional_fixed_scale_for_runtime_only_not_evidence_calibration")
    ref = lookup[spec["selection_ids"][0]]
    eeg, hb = scaled_window(ref, parent, cal)
    modes = infer_modes(eeg, cal, cfg)
    p, rows = parameters(cfg, project_root), []
    for name, gamma, gain, tau_n, tau_v in (("gamma", .5, 1., 0., 0.), ("gain", 0., .5, 0., 0.),
                                          ("input_lowpass", 0., .5, 1., 0.), ("viscoelastic_outflow", 0., .5, 0., 3.)):
        candidate = dict(arm=name, gamma=gamma, gain=gain, tau_n=tau_n, tau_v=tau_v,
                         random_seed=None, initial_mode="tied_pv_logprior")
        before = time.monotonic()
        fit = fit_response(hb, modes, candidate, cal, cfg, p, 40)
        rows.append(dict(arm=name, seconds=time.monotonic() - before, **compact_fit(fit)))
    result = dict(key=spec["key"], success=True, status="completed", source_id=ref["id"],
                  fits=rows, fit_count=len(rows), warm_fit_seconds=sum(r["seconds"] for r in rows),
                  seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  training_windows_read=len(train), evaluation_windows_read=0,
                  purpose="resource_throughput_only; not_model_selection_or_formal_endpoint")
    write_json(path, result)
    return result


def safe_worker(payload):
    kind, args = payload
    functions = {"calibrate": calibration_worker, "measured": measured_worker, "pilot": pilot_worker,
                 "synthetic_calibrate": synthetic_calibration_worker, "synthetic": synthetic_worker}
    started = time.monotonic()
    try:
        return functions[kind](args)
    except Exception as exc:
        cfg, project_root, run_root, spec = args[:4]
        identity = dict(spec) if isinstance(spec, dict) else dict(family=spec)
        if kind == "calibrate":
            path = Path(run_root) / "calibration" / (spec["key"] + ".json")
        elif kind == "pilot":
            path = Path(run_root) / "pilot" / (spec["key"] + ".json")
        elif kind == "synthetic_calibrate":
            path = Path(run_root) / "synthetic_calibration" / (spec + ".json")
        else:
            label = spec["ref"]["id"] if kind == "measured" else f'{spec["family"]}__{spec["scenario"]}__r{spec["repeat"]:03d}'
            path = Path(run_root) / "worker_failures" / kind / (label + ".json")
        row = dict(identity, success=False, converged=False, fit_performed=False, status="failed_exception",
                   error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(), seconds=time.monotonic() - started,
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        write_json(path, row)
        return row


def parallel_stage(kind, payloads, workers, run_root):
    """Bound in-flight work; the external systemd owner supplies durability."""
    payloads, pending = iter(payloads), {}
    done = failed = planned_cells = completed_cells = failed_cells = 0
    started, exhausted = time.monotonic(), False
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or not exhausted:
            while not exhausted and len(pending) < 2 * workers:
                try:
                    payload = next(payloads)
                except StopIteration:
                    exhausted = True
                    break
                pending[pool.submit(safe_worker, (kind, payload))] = None
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                pending.pop(future)
                row = future.result()
                done += 1
                failed += int(not bool(row.get("success")))
                planned_cells += int(row.get("planned_cells", 0))
                completed_cells += int(row.get("successful_cells", 0))
                failed_cells += int(row.get("failed_cells", 0))
            write_json(Path(run_root) / f"{kind}_progress.json", dict(stage=kind, completed_workers=done, failed_workers=failed,
                       recorded_cells=planned_cells, successful_cells=completed_cells, failed_cells=failed_cells,
                       in_flight=len(pending), workers=workers, seconds=time.monotonic() - started))
    result = dict(stage=kind, status="completed" if not failed else "completed_with_worker_failures", workers=workers,
                  completed_workers=done, failed_workers=failed, recorded_cells=planned_cells,
                  successful_cells=completed_cells, failed_cells=failed_cells, seconds=time.monotonic() - started)
    write_json(Path(run_root) / f"{kind}_progress.json", result)
    return result


def initialize_run(cfg, args, run_root):
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "resolved_config.json"
    if frozen.exists() and read_json(frozen) != serial(cfg):
        raise ValueError("resolved configuration differs from retained run; use a versioned run")
    if not frozen.exists():
        write_json(frozen, cfg)
    execution_path = root / "execution.json"
    execution = read_json(execution_path) if execution_path.exists() else dict(schema="semantic_response_discovery_execution_v1", stages={})
    execution["current_stage"] = args.stage
    execution["stages"][args.stage] = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(), command=sys.argv, cwd=str(Path.cwd()), python=sys.executable, workers=args.workers,
        supervisor_invocation=os.environ.get("INVOCATION_ID"), systemd_unit=os.environ.get("SYSTEMD_UNIT"),
        numerical_threads={name: os.environ[name] for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    write_json(execution_path, execution)
    return execution


def run_stage(cfg, args, run_root):
    project_root = Path(args.project_root)
    if args.stage == "dryrun":
        return dry_run(cfg, run_root, project_root)
    if args.stage == "synthetic":
        require_synthetic_fixture(run_root)
        calibration = parallel_stage("synthetic_calibrate", [(cfg, str(project_root), str(run_root), f) for f in cfg["synthetic"]["families"]],
                                     min(args.workers, len(cfg["synthetic"]["families"])), run_root)
        if calibration["failed_workers"]:
            raise ValueError("synthetic calibration worker failed; retained evidence must be inspected")
        specs = [dict(family=family, scenario=scenario, repeat=repeat) for family in cfg["synthetic"]["families"]
                 for scenario in cfg["synthetic"]["scenarios"] for repeat in range(cfg["synthetic"]["repeats"])]
        result = parallel_stage("synthetic", [(cfg, str(project_root), str(run_root), spec) for spec in specs], args.workers, run_root)
        result["summary"] = summarize_synthetic(cfg, run_root)
        return result
    if args.stage == "summarize":
        return summarize(cfg, run_root)
    require_synthetic_fixture(run_root)
    plan = parent_plan(cfg, run_root, project_root)
    plan_path = Path(run_root) / "measured_plan.json"
    if plan_path.exists() and read_json(plan_path) != serial(plan):
        raise ValueError("public plan differs from retained run")
    if not plan_path.exists():
        write_json(plan_path, plan)
    if args.stage == "pilot":
        return parallel_stage("pilot", [(cfg, str(project_root), str(run_root), spec, plan["refs"], plan["parent"])
                                        for spec in plan["calibration"]], min(args.workers, 9), run_root)
    if args.stage == "calibrate":
        return parallel_stage("calibrate", [(cfg, str(project_root), str(run_root), spec, plan["refs"], plan["parent"])
                                            for spec in plan["calibration"]], min(args.workers, 9), run_root)
    if args.stage == "measured":
        for spec in plan["calibration"]:
            if not (Path(run_root) / "calibration" / (spec["key"] + ".json")).exists():
                raise ValueError("all terminal calibration records required before evaluation")
        items = [item for item in plan["windows"] if item["split"] == "evaluation"]
        return parallel_stage("measured", [(cfg, str(project_root), str(run_root), item, plan["refs"], plan["parent"])
                                           for item in items], args.workers, run_root)
    if args.stage == "export-features":
        return export_batched_features(plan, run_root)
    raise ValueError("unregistered stage")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=CODE_ROOT)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("dryrun", "pilot", "synthetic", "calibrate", "measured", "summarize", "export-features"), required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    args.workers = cfg["resources"]["max_workers"] if args.workers is None else args.workers
    if not 1 <= args.workers <= cfg["resources"]["max_workers"]:
        parser.error("workers exceed the bounded configured resource scope")
    run_root = args.run_root.resolve()
    permitted = (args.project_root / cfg["artifact_root"]).resolve()
    if not run_root.is_relative_to(permitted) or run_root == permitted:
        parser.error("run root must name one versioned run below the configured artifact root")
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / "producer.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execution = initialize_run(cfg, args, run_root)
        try:
            result = run_stage(cfg, args, run_root)
            execution["stages"][args.stage].update(status="completed", completed_at=datetime.now(timezone.utc).isoformat(), result=result)
        except BaseException as exc:
            execution["stages"][args.stage].update(status="failed", completed_at=datetime.now(timezone.utc).isoformat(),
                error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            write_json(run_root / "execution.json", execution)
            raise
        write_json(run_root / "execution.json", execution)
    print(json.dumps(serial(result), ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
