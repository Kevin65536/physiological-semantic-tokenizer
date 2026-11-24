#!/usr/bin/env python3
"""Train-only Fourier-spatial or staggered-time carrier reachability on one EEG panel."""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
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

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/scripts"))

import numpy as np
from threadpoolctl import threadpool_limits
import yaml

from evaluate_joint_tokenizer import input_scales, write
from src.analysis.dual_component_tokenizer import (
    base_training_aggregate,
    joint_reconstruction_metrics,
    joint_reconstruction_summary,
)
from src.analysis.tokenizer_pattern_audit import attributes, feature_means
from src.data.dual_component_tokenizer import (
    base_tokenizer_batch,
    base_tokenizer_joint_eeg,
    base_tokenizer_training_data,
)
from src.tokenizers.eeg_transient_gain import load_model as load_transient_source
from src.tokenizers.anchored_spatial_carrier import AnchoredSpatialCarrier
from src.tokenizers.frequency_spatial_carrier import (
    FREQUENCY_INDICES,
    FrequencySpatialCarrier,
    fit_modes,
)
from src.tokenizers.spatial_waveform_dictionary import load_model
from src.tokenizers.staggered_time_carrier import (
    StaggeredTimeCarrier,
    fit_interpolator,
    sample_layout,
    stationary_moments,
)

STATE = {}
FIELDS = {"power_4_8": (0, 1), "power_8_13": (1, 2), "rms_envelope": (2, 10)}
METRICS = [
    "nmse",
    "nrmse",
    "rms_gain",
    "correlation",
    "rmse_native",
    "bias_native",
    "local_rms_correlation",
    "band_4_8_db",
    "band_8_13_db",
    "band_4_8_abs_db",
    "band_8_13_abs_db",
    "band_13_30_abs_db",
    "band_30_45_abs_db",
    "boundary_jump_difference_nmse",
]


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def ranked(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def budget():
    if STATE["config"]["schema"] in ["empirical_temporal_spatial_rate_v1", "temporal_local_information_diagnosis_v1"]:
        now = time.process_time()
        if now - STATE.get("last_budget_check", -10.0) >= 2.0:
            unit = os.environ["TOKENIZER_TEMPORAL_RATE_UNIT"]
            used = int(subprocess.check_output(
                ["systemctl", "--user", "show", unit, "-p", "CPUUsageNSec", "--value"], text=True,
            )) / 1e9 + STATE["config"]["resources"].get("prior_CPU_seconds", 0.0)
            if used >= STATE["config"]["resources"]["CPU_budget_seconds"] - 60:
                raise RuntimeError("temporal-spatial supervised CPU budget reached")
            STATE["last_budget_check"] = now
        return
    if time.process_time() > STATE.get(
        "CPU_limit", STATE["config"]["resources"]["CPU_budget_seconds"] / 4 - 30
    ):
        raise RuntimeError("per-dataset share of declared CPU budget reached")


def selected_attributes(values, complete):
    result = np.zeros((*complete.shape, 10))
    if complete.any():
        raw, columns = attributes(values[complete], "eeg")
        result[complete] = np.column_stack(
            [raw[:, a:b] for name, (a, b) in columns.items() if name in FIELDS]
        )
    return result


def prepare(dataset, *, frequency_covariance=True):
    rows, scales, cfg, source = (
        STATE[k] for k in ["rows", "scales", "config", "source"]
    )
    plan = STATE["plan"][dataset]
    names = source.metadata["positions"][dataset]
    native = {n.upper(): i for i, n in enumerate(names)}
    references = {
        name.upper(): record
        for name, record in scales["positions"]["eeg"][dataset].items()
    }
    sd = {
        b: np.sqrt(
            [references[n.upper()]["target_variance"][b.lower()][0] for n in names]
        )
        for b in "SO"
    }
    cases = []
    subject_cov = {}
    for subject, crops in sorted(plan.items()):
        records = defaultdict(list)
        record_sizes = {
            rows[c["row"]]["record"]: sum(
                rows[v["row"]]["record"] == rows[c["row"]]["record"] for v in crops
            )
            for c in crops
        }
        for crop in crops:
            budget()
            row = rows[crop["row"]]
            if (
                row["split"] != "train"
                or row["modality"] != "eeg"
                or row["dataset"] != dataset
                or row["subject"] != subject
            ):
                raise ValueError("exact train subject boundary before cached waveforms")
            file = (
                ROOT
                / cfg["source"]["waveform_cache"]
                / dataset
                / "cache"
                / (ranked([dataset, subject, crop])[:20] + ".npz")
            )
            ex = np.array(
                [(crop["row"], crop["start_s"], p) for p in crop["positions"]], np.int32
            )
            with np.load(file, allow_pickle=False) as f:
                np.testing.assert_array_equal(f["examples"], ex)
                carrier = f["carrier"].reshape(2, len(names), 15, 400)
                observed = f["observed"]
                selected = f["selected"]
                np.testing.assert_array_equal(
                    selected,
                    [native[row["positions"][p].upper()] for p in crop["positions"]],
                )
                batch = {key[6:]: f[key] for key in f if key.startswith("batch_")}
            # Keep variance arrays as broadcast views while retaining all original target samples.
            for key in ["vx", "vs", "vo", "dvx", "dvs", "dvo"]:
                batch[key] = np.broadcast_to(
                    batch[key][:, :1, :1].copy(), batch["x"].shape
                )
            complete = batch["mask"].all(-1)
            y = {b: selected_attributes(batch[b.lower()], complete) for b in "SO"}
            covariance = {}
            for j, b in enumerate("SO") if frequency_covariance else []:
                values = carrier[j] / sd[b][:, None, None]
                spectra = (np.fft.rfft(values, axis=-1) / 400)[
                    :, :, FREQUENCY_INDICES
                ].transpose(1, 2, 0)
                covariance[b] = (
                    np.einsum("qfp,qfr->fpr", spectra, spectra.conj(), optimize=True)
                    / 15
                )
            records[row["record"]].append(covariance)
            x_rms = np.sqrt(np.mean(np.square(batch["x"]), axis=-1))
            bg = selected[:, None] * 4 + np.searchsorted(np.array([0.5, 1, 2]), x_rms)
            cases.append(
                dict(
                    dataset=dataset,
                    subject=subject,
                    record=row["record"],
                    crop=crop,
                    examples=ex,
                    selected=selected,
                    carrier={b: carrier[j][None] for j, b in enumerate("SO")},
                    observed=observed[None],
                    batch=batch,
                    complete=complete,
                    targets=y,
                    background=bg,
                    weight=1
                    / (
                        len(plan)
                        * len(record_sizes)
                        * record_sizes[row["record"]]
                        * len(ex)
                    ),
                    scales=input_scales(rows, ex, scales),
                    fold=STATE["folds"][dataset][subject],
                )
            )
        subject_cov[subject] = {
            b: np.mean(
                [
                    np.mean([x[b] for x in values], axis=0)
                    for values in records.values()
                ],
                axis=0,
            )
            for b in "SO"
            if frequency_covariance
        }
        write(
            STATE["output"] / (dataset + "_progress.json"),
            dict(
                stage="source_statistics",
                subjects=len(subject_cov),
                planned_subjects=len(plan),
                crops=len(cases),
                CPU_seconds=time.process_time(),
            ),
        )
    return cases, subject_cov, sd


def fit_model(dataset, covariances, subjects, sd, coordinates):
    source = STATE["source"]
    cfg = STATE["config"]
    modes = {}
    scales = {}
    for branch in "SO":
        cov = np.mean([covariances[s][branch] for s in subjects], axis=0)
        modes[dataset, branch] = fit_modes(cov, coordinates)
        scales[dataset, branch] = sd[branch]
    meta = dict(
        schema="frequency_spatial_carrier_v1",
        positions={dataset: source.metadata["positions"][dataset]},
        complex_coordinates_per_slot=coordinates,
        source_input_identity=source.metadata["input_identity"],
        scales_identity=source.metadata["scales_identity"],
        fitting_subjects=subjects,
        frequencies_hz=[1, 45],
        covariance="complex_E_z_z_conjugate_per_2s_frequency_in_G0_branch_SD",
        context_s=30,
        patch_s=2,
        continuous_diagnostic=True,
        source_config_sha256=digest(STATE["config_path"]),
    )
    return FrequencySpatialCarrier(meta, scales, modes)


def coordinate_features(value, basis="native"):
    if basis == "power_only":
        return np.abs(value) ** 2
    if basis == "identity":
        return np.asarray(value, dtype=np.float64)
    if basis == "sample_rfft":
        value = np.fft.rfft(value, axis=-1) / np.sqrt(value.shape[-1])
    return np.concatenate((value.real, value.imag, np.abs(value) ** 2), axis=-1)


def fit_local_readout(cases, model, dataset, branch, coordinates=None, *, feature_basis=None):
    groups = defaultdict(list)
    count = len(model.metadata["positions"][dataset])
    basis = model.metadata.get("attribute_basis", "native") if feature_basis is None else feature_basis
    for ordinal, case in enumerate(cases):
        latent = (
            model.analyze(case["carrier"], dataset, case["observed"])
            if coordinates is None else coordinates[ordinal]
        )
        z = latent[branch][0]
        for j, site in enumerate(case["selected"]):
            complete = case["complete"][j]
            if not complete.any():
                continue
            groups[int(site)].append(
                (
                    case["background"][j, complete] % 4,
                    coordinate_features(z[site, complete], basis),
                    case["targets"][branch][j, complete],
                    np.full(complete.sum(), case["weight"] / complete.sum()),
                )
            )
    fitted = {}
    for site, parts in groups.items():
        bg = np.concatenate([p[0] for p in parts])
        x = np.concatenate([p[1] for p in parts])
        y = np.concatenate([p[2] for p in parts])
        w = np.concatenate([p[3] for p in parts])
        w /= w.sum()
        average = np.einsum("n,nf->f", w, y)
        means = np.tile(average, (4, 1))
        seen = np.zeros(4)
        for b in range(4):
            keep = bg == b
            if keep.any():
                means[b] = np.einsum("n,nf->f", w[keep] / w[keep].sum(), y[keep])
                seen[b] = int(keep.sum())
        center = np.einsum("n,nf->f", w, x)
        scale = np.sqrt(np.einsum("n,nf->f", w, np.square(x - center))).clip(1e-10)
        standard = (x - center) / scale
        residual = y - means[bg]
        gram = np.einsum("n,ni,nj->ij", w, standard, standard, optimize=True)
        right = np.einsum("n,ni,nj->ij", w, standard, residual, optimize=True)
        strength = STATE["config"]["selection"][
            "coordinate_readout_ridge_trace_fraction"
        ] * max(np.trace(gram) / len(gram), 1e-12)
        coef = np.linalg.solve(gram + strength * np.eye(len(gram)), right)
        fitted[site] = dict(
            background=means,
            seen=seen,
            center=center,
            scale=scale,
            coefficient=coef,
            basis=basis,
        )
    return fitted


def attribute_errors(case, coordinates, fitted, branch):
    result = np.full((len(case["examples"]), 2, 3), np.nan)
    z = coordinates[branch][0]
    for j, site in enumerate(case["selected"]):
        valid = case["complete"][j]
        if not valid.any():
            continue
        state = fitted[int(site)]
        background = state["background"][case["background"][j, valid] % 4]
        y = case["targets"][branch][j, valid]
        f = (
            coordinate_features(z[site, valid], state.get("basis", "native"))
            - state["center"]
        ) / state["scale"]
        augmented = background + f @ state["coefficient"]
        errors = np.stack(
            [feature_means((p - y) ** 2, FIELDS) for p in (background, augmented)],
            axis=1,
        )
        result[j] = errors.mean(0)
    return result


def projected_source(carrier, frequency_indices=FREQUENCY_INDICES):
    result = {}
    freq = np.fft.rfftfreq(6000, 1 / 200)
    keep = (freq >= 1) & (freq < 45)
    for b, wave in carrier.items():
        spec = np.fft.rfft(wave, axis=-1)
        mask = np.zeros(201, bool)
        mask[frequency_indices] = True
        compact = np.fft.irfft(spec * mask, n=400, axis=-1).reshape(
            *wave.shape[:2], 6000
        )
        result[b] = np.fft.irfft(
            np.fft.rfft(compact, axis=-1) * keep, n=6000, axis=-1
        ).reshape(wave.shape)
    result["sum"] = result["S"] + result["O"]
    return result


def score_case(case, prediction, *, full=False):
    selected = {k: v[0, case["selected"]] for k, v in prediction.items()}
    audit = joint_reconstruction_metrics(selected, case["batch"], 1, case["scales"])
    if audit["support"]["nonfinite_prediction_points"].any():
        raise ValueError("nonfinite retained waveform")
    metrics = audit["metrics"] if full else {k: audit["metrics"][k] for k in METRICS}
    return metrics, audit["support"]


def dataset_worker(dataset):
    with threadpool_limits(1):
        started = time.monotonic()
        out = STATE["output"]
        cfg = STATE["config"]
        rows = STATE["rows"]
        directory = out / dataset
        directory.mkdir(exist_ok=True)
        cases, covariances, sd = prepare(dataset)
        covariance_arrays = {
            f"{b}_{n}": value[b]
            for n, (s, value) in enumerate(covariances.items())
            for b in "SO"
        }
        np.savez_compressed(
            directory / "training_covariances.npz",
            subjects=np.array(list(covariances)),
            **covariance_arrays,
        )
        parts = defaultdict(list)
        examples = []
        fold_values = []
        latent_store = {}
        basis_reports = []
        for fold in range(4):
            train = [case for case in cases if case["fold"] != fold]
            held = [case for case in cases if case["fold"] == fold]
            train_subjects = sorted({case["subject"] for case in train})
            held_subjects = sorted({case["subject"] for case in held})
            if set(train_subjects) & set(held_subjects):
                raise ValueError("held subject in fitted covariance")
            baseline = AnchoredSpatialCarrier.load(
                ROOT / cfg["source"]["anchored_models"] / f"{dataset}_fold_{fold}.npz"
            )
            models = {
                m: fit_model(dataset, covariances, train_subjects, sd, m)
                for m in cfg["model"]["complex_coordinates_per_slot"]
            }
            readouts = {}
            for m, model in models.items():
                budget()
                file = directory / f"fold{fold}_m{m}.npz"
                model.save(file)
                reloaded = FrequencySpatialCarrier.load(file)
                # The decoder replay receives fixed basis and already computed coordinates.
                probe = held[0]
                coordinates = model.analyze(
                    probe["carrier"], dataset, probe["observed"]
                )
                for b, value in reloaded.synthesize(coordinates, dataset).items():
                    np.testing.assert_array_equal(
                        value, model.synthesize(coordinates, dataset)[b]
                    )
                readouts[m] = {
                    b: fit_local_readout(train, model, dataset, b) for b in "SO"
                }
                for b in "SO":
                    modes = model.modes[dataset, b]
                    frequencies, counts = np.unique(
                        modes["frequency"], return_counts=True
                    )
                    local_load = np.array(
                        [
                            abs(modes["vectors"][p, j, p]) ** 2
                            for p in range(len(sd[b]))
                            for j in range(m)
                        ]
                    )
                    basis_reports.append(
                        dict(
                            fold=fold,
                            branch=b,
                            coordinates_per_slot=m,
                            selected_complex_modes=len(sd[b]) * m,
                            train_subjects=train_subjects,
                            held_subjects=held_subjects,
                            train_energy_fraction=float(
                                modes["training_selected_energy"]
                                / max(modes["training_total_energy"], 1e-20)
                            ),
                            frequency_counts={
                                str(f / 2): int(n) for f, n in zip(frequencies, counts)
                            },
                            local_squared_loading_quantiles=np.quantile(
                                local_load, [0, 0.25, 0.5, 0.75, 1]
                            ).tolist(),
                        )
                    )
            for index, case in enumerate(held):
                budget()
                source = case["carrier"]
                anchored = baseline.synthesize(
                    baseline.analyze(source, dataset, case["observed"]), dataset
                )
                predictions = {
                    "source_Wiener": dict(source, sum=source["S"] + source["O"]),
                    "patch_projection": projected_source(source),
                    "anchored": anchored,
                }
                for m, model in models.items():
                    coordinates = model.analyze(source, dataset, case["observed"])
                    name = f"m{m}"
                    predictions[name] = model.synthesize(coordinates, dataset)
                    for b in "SO":
                        parts[f"{name}_attribute_{b}"].append(
                            attribute_errors(case, coordinates, readouts[m][b], b)
                        )
                        latent_store[f"{name}_{b}_{len(examples)}"] = coordinates[b]
                for arm, prediction in predictions.items():
                    met, support = score_case(case, prediction)
                    for k, v in met.items():
                        parts[arm + "__" + k].append(v)
                    if arm == "anchored":
                        parts["valid_points"].append(support["valid_points"])
                        parts["full_patches"].append(support["full_patches"])
                examples.extend(case["examples"])
                fold_values.extend([fold] * len(case["examples"]))
            write(
                out / (dataset + "_progress.json"),
                dict(
                    stage="completed_fold",
                    fold=fold,
                    folds=4,
                    CPU_seconds=time.process_time(),
                ),
            )
        values = {k: np.concatenate(v) for k, v in parts.items()}
        np.savez_compressed(
            out / (dataset + "_held.npz"),
            examples=np.asarray(examples, np.int32),
            fold=np.asarray(fold_values, np.int8),
            **values,
        )
        np.savez_compressed(
            directory / "held_continuous_coordinates.npz", **latent_store
        )
        write(directory / "basis_audit.json", basis_reports)
        write(
            directory / "summary.json",
            dict(
                status="completed",
                cases=len(cases),
                position_windows=len(examples),
                subjects=len(covariances),
                CPU_seconds=time.process_time(),
                elapsed_seconds=time.monotonic() - started,
                continuous_model_reload="exact",
                validation_reads=0,
            ),
        )
        return dataset, time.process_time()


def summarize(out):
    arrays = defaultdict(list)
    for d in sorted(STATE["plan"]):
        with np.load(out / (d + "_held.npz"), allow_pickle=False) as f:
            for k, v in f.items():
                arrays[k].append(v)
    values = {k: np.concatenate(v) for k, v in arrays.items()}
    examples = values["examples"]
    rows = STATE["rows"]
    expected = sorted(
        (c["row"], c["start_s"], p)
        for subjects in STATE["plan"].values()
        for crops in subjects.values()
        for c in crops
        for p in c["positions"]
    )
    if sorted(map(tuple, examples.tolist())) != expected:
        raise ValueError("complete training example population changed")
    arms = {}
    for arm in ["source_Wiener", "patch_projection", "anchored", "m1", "m2", "m4"]:
        met = {
            k.split("__", 1)[1]: v
            for k, v in values.items()
            if k.startswith(arm + "__")
        }
        arms[arm] = base_training_aggregate(met, examples, rows)
    pattern = {}
    for m in [1, 2, 4]:
        pattern[m] = {}
        for b in "SO":
            errors = values[f"m{m}_attribute_{b}"]
            summary = base_training_aggregate({"nmse": errors}, examples, rows)[
                "metrics"
            ]["nmse"]
            mse = np.asarray(summary["macro"])
            inc = 1 - mse[1] / mse[0]
            fraction = np.mean(
                [
                    np.asarray(v["value"])[1] < np.asarray(v["value"])[0]
                    for v in summary["by_subject"]
                ],
                axis=0,
            )
            pattern[m][b] = dict(
                gain=dict(zip(FIELDS, inc)),
                positive_subject_fraction=dict(zip(FIELDS, fraction)),
                hierarchical_attribute_MSE=summary,
                passed=bool(
                    min(inc[:2]) > 0 and max(inc) >= 0.05 and min(fraction[:2]) >= 0.5
                ),
            )
    reference = arms["anchored"]["metrics"]["nmse"]
    limits = STATE["config"]["selection"]
    decisions = {}
    for m in [1, 2, 4]:
        stats = arms[f"m{m}"]["metrics"]
        nm = np.asarray(stats["nmse"]["macro"])
        base = np.asarray(reference["macro"])
        checks = dict(
            component_NMSE=bool(
                np.all(
                    nm
                    <= limits[
                        "waveform_component_max_ratio_to_matched_anchored_continuous"
                    ]
                    * base
                )
            ),
            dataset_NMSE=all(
                np.all(
                    np.asarray(v)
                    <= limits[
                        "waveform_dataset_component_max_ratio_to_matched_anchored_continuous"
                    ]
                    * np.asarray(reference["by_dataset"][d])
                )
                for d, v in stats["nmse"]["by_dataset"].items()
            ),
            RMS=bool(
                np.all(
                    (
                        np.asarray(stats["rms_gain"]["macro"])
                        >= limits["RMS_gain_interval"][0]
                    )
                    & (
                        np.asarray(stats["rms_gain"]["macro"])
                        <= limits["RMS_gain_interval"][1]
                    )
                )
            ),
            positive_correlations=bool(
                np.all(np.asarray(stats["correlation"]["macro"]) > 0)
            ),
            local_RMS=bool(
                np.all(
                    np.asarray(stats["local_rms_correlation"]["macro"])
                    >= limits["local_RMS_correlation_min"]
                )
            ),
            theta_alpha=all(
                np.all(
                    abs(np.asarray(stats[key]["macro"]))
                    <= limits["theta_alpha_signed_and_mean_absolute_dB_max"]
                )
                for key in [
                    "band_4_8_db",
                    "band_8_13_db",
                    "band_4_8_abs_db",
                    "band_8_13_abs_db",
                ]
            ),
            local_coordinate_information=all(pattern[m][b]["passed"] for b in "SO"),
        )
        decisions[m] = dict(
            checks=checks,
            reachable=all(checks.values()),
            eligible_for_single_ID=m in [1, 2] and all(checks.values()),
        )
    selected = next((m for m in [1, 2] if decisions[m]["eligible_for_single_ID"]), None)
    # Match exactly the preceding continuous reference using the same fold and G0 triples.
    largest = 0.0
    for d in sorted(STATE["plan"]):
        with (
            np.load(out / (d + "_held.npz")) as new,
            np.load(
                ROOT / STATE["config"]["source"]["anchored_models"] / (d + "_held.npz")
            ) as old,
        ):
            table = {tuple(v): j for j, v in enumerate(old["examples"])}
            take = [table[tuple(v)] for v in new["examples"]]
            for key in [
                "nmse",
                "rms_gain",
                "correlation",
                "local_rms_correlation",
                "band_4_8_abs_db",
                "band_8_13_abs_db",
            ]:
                a, b = new["anchored__" + key], old["anchored__" + key][take]
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)
                finite = np.isfinite(a) & np.isfinite(b)
                largest = max(
                    largest, float(np.max(abs(a[finite] - b[finite]), initial=0))
                )
    write(
        out / "summary.json",
        dict(
            status="continuous_reachable"
            if selected
            else "continuous_unreachable_under_declared_budgets",
            arms=arms,
            continuous_coordinate_attribute_readout=pattern,
            decisions=decisions,
            selected_for_VQ=selected,
            matched_anchored_reference_max_abs_difference=largest,
            position_windows=len(examples),
            physical_windows=400,
            training_subjects=60,
            exposure="Only frequency covariance, mode selection, Hungarian mapping and local readouts exclude held training subjects. Frozen Wiener and G0 SD had seen all train subjects.",
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
            integer_model_trained=False,
        ),
    )
    return decisions


def staggered_dataset_worker(dataset):
    """Same physical cases/folds/scorer; only the continuous carrier map changes."""
    with threadpool_limits(1):
        started = time.monotonic()
        cfg, out = STATE["config"], STATE["output"]
        directory = out / dataset
        directory.mkdir(exist_ok=True)
        if (directory / "summary.json").exists() and (
            out / (dataset + "_held.npz")
        ).exists():
            prior = json.loads((directory / "summary.json").read_text())
            if prior["status"] == "completed":
                return dataset, prior["CPU_seconds"]
        cases, _, sd = prepare(dataset, frequency_covariance=False)
        moments, weights = {}, np.zeros(4)
        for case in cases:
            budget()
            fold = case["fold"]
            weight = case["weight"] * len(case["examples"])
            weights[fold] += weight
            for b in "SO":
                wave = case["carrier"][b][0].reshape(len(sd[b]), 6000)
                values = stationary_moments(
                    wave / sd[b][:, None], case["observed"][0], 400
                )
                for kind, value in values.items():
                    key = fold, b, kind
                    if key not in moments:
                        moments[key] = np.zeros_like(value)
                    moments[key] += weight * value
        write(
            directory / "covariance_audit.json",
            dict(
                weighting="equal subject then record then crop",
                fold_weights=weights,
                covariance_shape=list(next(iter(moments.values())).shape),
                maximum_lag_samples=400,
                centering=False,
                missing_inputs="zero observation carrier; target remains full Wiener",
            ),
        )
        parts, examples, fold_values, latent_store = defaultdict(list), [], [], {}
        for fold in range(4):
            train = [c for c in cases if c["fold"] != fold]
            held = [c for c in cases if c["fold"] == fold]
            fit_subjects = sorted({c["subject"] for c in train})
            held_subjects = sorted({c["subject"] for c in held})
            if set(fit_subjects) & set(held_subjects):
                raise ValueError("held subject in interpolation covariance")
            covariance = {
                b: {
                    kind: sum(moments[f, b, kind] for f in range(4) if f != fold)
                    / (weights.sum() - weights[fold])
                    for kind in ["observation", "target_observation"]
                }
                for b in "SO"
            }
            models, readouts = {}, {}
            for m in cfg["model"]["samples_per_slot"]:
                for layout in cfg["model"]["layouts"]:
                    budget()
                    name = f"{layout}_m{m}"
                    indices, phase = sample_layout(
                        len(sd["S"]), m, layout == "staggered"
                    )
                    meta = dict(
                        schema="staggered_time_carrier_v1",
                        real_samples_per_slot=m,
                        positions={
                            dataset: STATE["source"].metadata["positions"][dataset]
                        },
                        layout=layout,
                        phase_samples=phase.tolist(),
                        fitting_subjects=fit_subjects,
                        excluded_subjects=held_subjects,
                        context_s=30,
                        sample_rate_hz=200,
                        attribute_basis="sample_rfft",
                        source_sha256=digest(ROOT / cfg["source"]["Wiener"]),
                        config_sha256=digest(STATE["config_path"]),
                        emitted_integers=0,
                        continuous_diagnostic=True,
                    )
                    model = StaggeredTimeCarrier(
                        meta,
                        {(dataset, b): sd[b] for b in "SO"},
                        {dataset: indices},
                        {
                            (dataset, b): fit_interpolator(
                                covariance[b],
                                indices,
                                cfg["model"]["ridge_trace_fraction"],
                            )
                            for b in "SO"
                        },
                    )
                    file = directory / f"fold{fold}_{name}.npz"
                    model.save(file)
                    loaded = StaggeredTimeCarrier.load(file)
                    probe = held[0]
                    coords = model.analyze(probe["carrier"], dataset, probe["observed"])
                    replay = loaded.synthesize(coords, dataset)
                    for b, value in model.synthesize(coords, dataset).items():
                        np.testing.assert_array_equal(value, replay[b])
                    models[name] = model
                    readouts[name] = {
                        b: fit_local_readout(train, model, dataset, b) for b in "SO"
                    }
            baseline = AnchoredSpatialCarrier.load(
                ROOT / cfg["source"]["anchored_models"] / f"{dataset}_fold_{fold}.npz"
            )
            for case in held:
                budget()
                source = case["carrier"]
                predictions = {
                    "source_Wiener": dict(source, sum=source["S"] + source["O"]),
                    "anchored": baseline.synthesize(
                        baseline.analyze(source, dataset, case["observed"]), dataset
                    ),
                }
                for name, model in models.items():
                    coords = model.analyze(source, dataset, case["observed"])
                    predictions[name] = model.synthesize(coords, dataset)
                    for b in "SO":
                        parts[f"{name}_attribute_{b}"].append(
                            attribute_errors(case, coords, readouts[name][b], b)
                        )
                        latent_store[f"{name}_{b}_{len(examples)}"] = coords[b]
                    rms = []
                    for b in ["sum", "S", "O"]:
                        target = source["S"] + source["O"] if b == "sum" else source[b]
                        selected = case["selected"]
                        ratio = np.sqrt(
                            np.sum(
                                predictions[name][b][0, selected] ** 2
                                * case["batch"]["mask"],
                                axis=(1, 2),
                            )
                            / np.maximum(
                                np.sum(
                                    target[0, selected] ** 2 * case["batch"]["mask"],
                                    axis=(1, 2),
                                ),
                                1e-20,
                            )
                        )
                        rms.append(ratio)
                    parts[name + "__source_rms_retention"].append(
                        np.stack(rms, axis=1)[..., None]
                    )
                for arm, prediction in predictions.items():
                    metrics, support = score_case(case, prediction)
                    for key, value in metrics.items():
                        parts[arm + "__" + key].append(value)
                    if arm == "anchored":
                        parts["valid_points"].append(support["valid_points"])
                        parts["full_patches"].append(support["full_patches"])
                examples.extend(case["examples"])
                fold_values.extend([fold] * len(case["examples"]))
            write(
                out / (dataset + "_progress.json"),
                dict(
                    stage="completed_fold",
                    fold=fold,
                    folds=4,
                    CPU_seconds=time.process_time(),
                ),
            )
        np.savez_compressed(
            out / (dataset + "_held.npz"),
            examples=np.asarray(examples, np.int32),
            fold=np.asarray(fold_values, np.int8),
            **{key: np.concatenate(value) for key, value in parts.items()},
        )
        np.savez_compressed(
            directory / "held_continuous_coordinates.npz", **latent_store
        )
        write(
            directory / "summary.json",
            dict(
                status="completed",
                cases=len(cases),
                position_windows=len(examples),
                subjects=len({c["subject"] for c in cases}),
                CPU_seconds=time.process_time(),
                elapsed_seconds=time.monotonic() - started,
                continuous_model_reload="exact",
                validation_reads=0,
            ),
        )
        return dataset, time.process_time()


def summarize_staggered(out):
    arrays = defaultdict(list)
    for d in sorted(STATE["plan"]):
        with np.load(out / (d + "_held.npz"), allow_pickle=False) as f:
            for key, value in f.items():
                arrays[key].append(value)
    values = {k: np.concatenate(v) for k, v in arrays.items()}
    examples, rows = values["examples"], STATE["rows"]
    expected = sorted(
        (c["row"], c["start_s"], p)
        for subjects in STATE["plan"].values()
        for crops in subjects.values()
        for c in crops
        for p in c["positions"]
    )
    if sorted(map(tuple, examples.tolist())) != expected:
        raise ValueError("physical train population changed")
    names = [
        f"{layout}_m{m}" for m in [4, 16, 32] for layout in ["aligned", "staggered"]
    ]
    arms = {
        arm: base_training_aggregate(
            {
                k.split("__", 1)[1]: v
                for k, v in values.items()
                if k.startswith(arm + "__")
            },
            examples,
            rows,
        )
        for arm in ["source_Wiener", "anchored"] + names
    }
    pattern, decisions = {}, {}
    limits = STATE["config"]["selection"]
    for name in names:
        pattern[name] = {}
        for b in "SO":
            summary = base_training_aggregate(
                {"nmse": values[f"{name}_attribute_{b}"]}, examples, rows
            )["metrics"]["nmse"]
            mse = np.asarray(summary["macro"])
            inc = 1 - mse[1] / mse[0]
            fraction = np.mean(
                [
                    np.asarray(v["value"])[1] < np.asarray(v["value"])[0]
                    for v in summary["by_subject"]
                ],
                axis=0,
            )
            pattern[name][b] = dict(
                gain=dict(zip(FIELDS, inc)),
                positive_subject_fraction=dict(zip(FIELDS, fraction)),
                hierarchical_attribute_MSE=summary,
                passed=bool(
                    min(inc[:2]) > 0 and max(inc) >= 0.05 and min(fraction[:2]) >= 0.5
                ),
            )
    for m in [4, 16, 32]:
        name = f"staggered_m{m}"
        stats = arms[name]["metrics"]
        nm = np.asarray(stats["nmse"]["macro"])
        aligned = np.asarray(arms[f"aligned_m{m}"]["metrics"]["nmse"]["macro"])
        anchor = arms["anchored"]["metrics"]["nmse"]
        fold_gains = []
        for fold in range(4):
            take = values["fold"] == fold
            a = base_training_aggregate(
                {"nmse": values[f"aligned_m{m}__nmse"][take]}, examples[take], rows
            )["metrics"]["nmse"]["macro"]
            s = base_training_aggregate(
                {"nmse": values[f"{name}__nmse"][take]}, examples[take], rows
            )["metrics"]["nmse"]["macro"]
            fold_gains.append(float(1 - np.asarray(s)[1, 0] / np.asarray(a)[1, 0]))
        dataset_gains = {
            d: float(
                1
                - np.asarray(v)[1, 0]
                / np.asarray(arms[f"aligned_m{m}"]["metrics"]["nmse"]["by_dataset"][d])[
                    1, 0
                ]
            )
            for d, v in stats["nmse"]["by_dataset"].items()
        }
        phase = dict(
            macro_XSO=float(1 - nm.mean() / aligned.mean()) >= 0.05,
            S=float(1 - nm[1, 0] / aligned[1, 0]) >= 0.05,
            datasets_S=sum(v > 0 for v in dataset_gains.values()) >= 3,
            folds_S=sum(v > 0 for v in fold_gains) >= 3,
            no_component_harm=bool(np.all(nm <= 1.05 * aligned)),
        )
        checks = dict(
            component_NMSE=bool(np.all(nm <= 1.35 * np.asarray(anchor["macro"]))),
            dataset_NMSE=all(
                np.all(np.asarray(v) <= 1.5 * np.asarray(anchor["by_dataset"][d]))
                for d, v in stats["nmse"]["by_dataset"].items()
            ),
            source_RMS=bool(
                np.all(np.asarray(stats["source_rms_retention"]["macro"]) >= 0.9)
            ),
            RMS=bool(
                np.all(
                    (np.asarray(stats["rms_gain"]["macro"]) >= 0.75)
                    & (np.asarray(stats["rms_gain"]["macro"]) <= 1.25)
                )
            ),
            local_RMS=bool(
                np.all(np.asarray(stats["local_rms_correlation"]["macro"]) >= 0.5)
            ),
            theta_alpha=all(
                np.all(abs(np.asarray(stats[key]["macro"])) <= 3.0103)
                for key in [
                    "band_4_8_db",
                    "band_8_13_db",
                    "band_4_8_abs_db",
                    "band_8_13_abs_db",
                ]
            ),
            local_coordinate_information=all(pattern[name][b]["passed"] for b in "SO"),
        )
        decisions[m] = dict(
            phase_hypothesis=phase,
            waveform_and_local_information=checks,
            relative_XSO_improvement=float(1 - nm.mean() / aligned.mean()),
            relative_S_improvement=float(1 - nm[1, 0] / aligned[1, 0]),
            dataset_S_improvement=dataset_gains,
            fold_S_improvement=fold_gains,
            numerical_eligible_for_VQ=all(phase.values()) and all(checks.values()),
        )
    largest = 0.0
    for d in sorted(STATE["plan"]):
        with (
            np.load(out / (d + "_held.npz")) as new,
            np.load(
                ROOT / STATE["config"]["source"]["anchored_models"] / (d + "_held.npz")
            ) as old,
        ):
            lookup = {tuple(v): j for j, v in enumerate(old["examples"])}
            take = [lookup[tuple(v)] for v in new["examples"]]
            for key in [
                "nmse",
                "rms_gain",
                "correlation",
                "local_rms_correlation",
                "band_4_8_abs_db",
                "band_8_13_abs_db",
            ]:
                a, b = new["anchored__" + key], old["anchored__" + key][take]
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)
                finite = np.isfinite(a) & np.isfinite(b)
                largest = max(
                    largest, float(np.max(abs(a[finite] - b[finite]), initial=0))
                )
    selected = next(
        (m for m, state in decisions.items() if state["numerical_eligible_for_VQ"]),
        None,
    )
    write(
        out / "summary.json",
        dict(
            status="continuous_complete_pending_visual_audit",
            arms=arms,
            continuous_coordinate_attribute_readout=pattern,
            decisions=decisions,
            selected_for_VQ_pending_visual=selected,
            matched_anchored_reference_max_abs_difference=largest,
            position_windows=len(examples),
            physical_windows=400,
            training_subjects=60,
            exposure="Wiener/scales saw all train; interpolation and local attribute heads exclude held train subjects",
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
            integer_model_trained=False,
        ),
    )
    return decisions


def review_staggered(out):
    """File-only continuous replay and frozen fixed/random/hard visual panel."""
    import csv
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.process_time()
    summary = json.loads((out / "summary.json").read_text())
    names = [
        f"{layout}_m{m}" for m in [4, 16, 32] for layout in ["aligned", "staggered"]
    ]
    replay_max, figures, densities = 0.0, [], []
    colors = {
        "target": "#172f40",
        "source_Wiener": "#8d979f",
        "aligned_m32": "#3579a8",
        "staggered_m32": "#b55d28",
    }
    for dataset in sorted(STATE["plan"]):
        cases, _, _ = prepare(dataset, frequency_covariance=False)
        directory = out / dataset
        dynamics, denominator, dynamic_examples = defaultdict(list), [], []
        with (
            np.load(out / (dataset + "_held.npz")) as raw,
            np.load(directory / "held_continuous_coordinates.npz") as latents,
        ):
            retained = {k: raw[k] for k in raw}
            lookup = {tuple(e): j for j, e in enumerate(retained["examples"])}
            ex = retained["examples"]
            fixed = min(map(tuple, ex.tolist()))
            random = min(
                map(tuple, ex.tolist()), key=lambda e: ranked([17, dataset, e])
            )
            errors = retained["staggered_m32__nmse"][:, 1, 0]
            hard = tuple(ex[np.nanargmin(abs(errors - np.nanquantile(errors, 0.95)))])
            selections = [
                ("fixed", fixed),
                ("random_seed17", random),
                ("p95_S_error", hard),
            ]
            models = {}
            for case in cases:
                first = tuple(case["examples"][0])
                offset = lookup[first]
                take = [lookup[tuple(e)] for e in case["examples"]]
                fold = case["fold"]
                if fold not in models:
                    models[fold] = {
                        name: StaggeredTimeCarrier.load(
                            directory / f"fold{fold}_{name}.npz"
                        )
                        for name in names
                    }
                predictions = {}
                for name, model in models[fold].items():
                    coordinates = {b: latents[f"{name}_{b}_{offset}"] for b in "SO"}
                    predictions[name] = model.synthesize(coordinates, dataset)
                    metrics, _ = score_case(case, predictions[name], full=True)
                    for key in METRICS:
                        value = metrics[key]
                        expected = retained[name + "__" + key][take]
                        np.testing.assert_array_equal(value, expected)
                        finite = np.isfinite(value) & np.isfinite(expected)
                        replay_max = max(
                            replay_max,
                            float(
                                np.max(abs(value[finite] - expected[finite]), initial=0)
                            ),
                        )
                    for key, value in metrics.items():
                        dynamics[name + "__" + key].append(value)
                predictions["source_Wiener"] = dict(
                    case["carrier"], sum=case["carrier"]["S"] + case["carrier"]["O"]
                )
                reference_metrics, _ = score_case(
                    case, predictions["source_Wiener"], full=True
                )
                for key, value in reference_metrics.items():
                    dynamics["source_Wiener__" + key].append(value)
                denominator.append(
                    np.stack(
                        [case["batch"]["v" + key][:, 0, 0] for key in ["x", "s", "o"]],
                        axis=1,
                    )
                    * case["scales"][:, None] ** 2
                )
                dynamic_examples.extend(case["examples"])
                for rule, identity in selections:
                    match = [
                        j
                        for j, example in enumerate(case["examples"])
                        if tuple(example) == identity
                    ]
                    if not match:
                        continue
                    j = match[0]
                    native = case["selected"][j]
                    factor = case["scales"][j]
                    mask = case["batch"]["mask"][j].ravel().astype(bool)
                    time_axis = np.arange(6000) / 200
                    fig, axes = plt.subplots(
                        3, 3, figsize=(18, 12), layout="constrained"
                    )
                    rmse = {}
                    for row, (target, branch) in enumerate(
                        [("x", "sum"), ("s", "S"), ("o", "O")]
                    ):
                        waves = {"target": case["batch"][target][j].ravel() * factor}
                        waves.update(
                            {
                                arm: predictions[arm][branch][0, native].ravel()
                                * factor
                                for arm in colors
                                if arm != "target"
                            }
                        )
                        lo = min(np.min(w[mask]) for w in waves.values())
                        hi = max(np.max(w[mask]) for w in waves.values())
                        padding = max((hi - lo) * 0.04, 1e-12)
                        rmse[branch] = {
                            arm: float(
                                np.sqrt(
                                    np.mean((wave[mask] - waves["target"][mask]) ** 2)
                                )
                            )
                            for arm, wave in waves.items()
                            if arm != "target"
                        }
                        for col, (left, right) in enumerate(
                            [(0, 30), (14, 16), (15.7, 16.3)]
                        ):
                            ax = axes[row, col]
                            for arm, wave in waves.items():
                                ax.plot(
                                    time_axis,
                                    np.where(mask, wave, np.nan),
                                    color=colors[arm],
                                    lw=0.75 if col == 0 else 1.0,
                                    label=arm,
                                )
                            for boundary in range(0, 31, 2):
                                ax.axvline(boundary, color="#bdcad1", lw=0.5, zorder=0)
                            ax.set(
                                xlim=(left, right),
                                ylim=(lo - padding, hi + padding),
                                xlabel="Time within visible window (s)",
                                ylabel=f"{branch} / "
                                + (
                                    "native export unit"
                                    if dataset == "refed"
                                    else STATE["rows"][identity[0]]["source_unit"]
                                ),
                            )
                            if row == 0:
                                ax.set_title(
                                    [
                                        "Full 30 s",
                                        "Fixed 2 s",
                                        "Across integer boundary",
                                    ][col]
                                )
                            if row == col == 0:
                                ax.legend(fontsize=8, ncol=2)
                    record = STATE["rows"][identity[0]]
                    fig.suptitle(
                        f"{dataset} / {case['subject']} / {record['record']} / {record['positions'][identity[2]]} / +{identity[1]}s / {rule}\nHeld training subject, fold {fold + 1}; continuous actual samples; mask gaps and original amplitudes retained",
                        fontsize=13,
                    )
                    path = out / f"{dataset}_{rule}_waveform.png"
                    fig.savefig(path, dpi=120)
                    plt.close(fig)
                    figures.append(
                        dict(
                            path=path.name,
                            selection=rule,
                            example=list(identity),
                            subject=case["subject"],
                            record=record["record"],
                            fold=fold,
                            valid_samples=int(mask.sum()),
                            rmse_native=rmse,
                        )
                    )
                    if rule == "fixed":
                        fig, axes = plt.subplots(
                            2, 3, figsize=(18, 8), layout="constrained"
                        )
                        for bi, branch in enumerate("SO"):
                            truth = case["batch"][branch.lower()][j].ravel() * factor
                            window = (time_axis >= 14) & (time_axis < 16)
                            maximum = max(
                                np.max(
                                    abs(
                                        predictions[n][branch][0, native].ravel()[
                                            window
                                        ]
                                        * factor
                                    )
                                )
                                for n in names
                            )
                            maximum = max(maximum, np.max(abs(truth[window]))) * 1.05
                            for col, m in enumerate([4, 16, 32]):
                                ax = axes[bi, col]
                                ax.plot(
                                    time_axis,
                                    np.where(mask, truth, np.nan),
                                    color=colors["target"],
                                    label="target",
                                    lw=1,
                                )
                                for layout, color in [
                                    ("aligned", "#3579a8"),
                                    ("staggered", "#b55d28"),
                                ]:
                                    arm = f"{layout}_m{m}"
                                    wave = (
                                        predictions[arm][branch][0, native].ravel()
                                        * factor
                                    )
                                    ax.plot(
                                        time_axis,
                                        np.where(mask, wave, np.nan),
                                        color=color,
                                        lw=1,
                                        label=layout,
                                    )
                                    indices = models[fold][arm].indices[dataset][native]
                                    actual = (
                                        case["carrier"][branch][0, native, 7, indices]
                                        * factor
                                    )
                                    ax.scatter(
                                        14 + indices / 200,
                                        actual,
                                        c=color,
                                        s=14,
                                        marker="o" if layout == "staggered" else "x",
                                        label=layout + " encoded points",
                                    )
                                ax.set(
                                    xlim=(14, 16),
                                    ylim=(-maximum, maximum),
                                    xlabel="Time within window (s)",
                                    ylabel=branch + " / native voltage",
                                    title=f"{m} actual real samples / slot",
                                )
                                if bi == 0 and col == 0:
                                    ax.legend(fontsize=7)
                        fig.suptitle(
                            f"{dataset} / same fixed sample; local actual carriers and array interpolation\nMarkers are encoded Wiener samples, not true supervised S/O; matched physical amplitude limits",
                            fontsize=12,
                        )
                        path = out / f"{dataset}_fixed_sampling_density.png"
                        fig.savefig(path, dpi=120)
                        plt.close(fig)
                        densities.append(dict(path=path.name, example=list(identity)))
        np.savez_compressed(
            directory / "dynamic_replay_audit.npz",
            examples=np.asarray(dynamic_examples),
            target_variance_native=np.concatenate(denominator),
            **{key: np.concatenate(parts) for key, parts in dynamics.items()},
        )
        write(
            directory / "dynamic_replay_summary.json",
            {
                arm: base_training_aggregate(
                    {
                        key.split("__", 1)[1]: np.concatenate(parts)
                        for key, parts in dynamics.items()
                        if key.startswith(arm + "__")
                    },
                    np.asarray(dynamic_examples),
                    STATE["rows"],
                )
                for arm in names + ["source_Wiener"]
            },
        )
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for bi, branch in enumerate(["S", "O"]):
        for layout, color in [("aligned", "#3579a8"), ("staggered", "#b55d28")]:
            axes[bi, 0].plot(
                [4, 16, 32],
                [
                    summary["arms"][f"{layout}_m{m}"]["metrics"]["nmse"]["macro"][
                        bi + 1
                    ][0]
                    for m in [4, 16, 32]
                ],
                "o-",
                color=color,
                label=layout,
            )
        axes[bi, 0].axhline(
            summary["arms"]["source_Wiener"]["metrics"]["nmse"]["macro"][bi + 1][0],
            color="#8d979f",
            label="full Wiener",
        )
        axes[bi, 0].set(
            xlabel="Real samples per local vector",
            ylabel=branch + " NMSE (lower better)",
        )
        axes[bi, 0].legend()
        for fi, (field, label) in enumerate(
            [
                ("power_4_8", "theta"),
                ("power_8_13", "alpha"),
                ("rms_envelope", "local RMS"),
            ]
        ):
            axes[bi, 1].plot(
                [4, 16, 32],
                [
                    summary["continuous_coordinate_attribute_readout"][
                        f"staggered_m{m}"
                    ][branch]["gain"][field]
                    for m in [4, 16, 32]
                ],
                "o-",
                label=label,
            )
        axes[bi, 1].axhline(0, color="grey", lw=0.6)
        axes[bi, 1].set(
            xlabel="Staggered real samples per local vector",
            ylabel=branch + " local attribute MSE reduction",
        )
        axes[bi, 1].legend()
        for key, label in [("band_4_8_abs_db", "theta"), ("band_8_13_abs_db", "alpha")]:
            axes[bi, 2].plot(
                [4, 16, 32],
                [
                    summary["arms"][f"staggered_m{m}"]["metrics"][key]["macro"][bi + 1][
                        0
                    ]
                    for m in [4, 16, 32]
                ],
                "o-",
                label=label,
            )
        axes[bi, 2].axhline(3.0103, color="grey", ls="--", label="fixed threshold")
        axes[bi, 2].set(
            xlabel="Staggered real samples per local vector",
            ylabel=branch + " mean absolute band error / dB",
        )
        axes[bi, 2].legend()
    fig.suptitle(
        "Train-only subject crossfit: original dataset/subject hierarchy, fixed variance and masks",
        fontsize=13,
    )
    fig.savefig(out / "reachability.png", dpi=140)
    plt.close(fig)
    with (out / "per_subject_metrics.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["arm", "metric", "dataset", "subject", "X", "S", "O"])
        for arm, result in summary["arms"].items():
            for key, metric in result["metrics"].items():
                for row in metric["by_subject"]:
                    writer.writerow(
                        [arm, key, row["dataset"], row["subject"]]
                        + np.asarray(row["value"]).ravel().tolist()
                    )
    with (out / "per_subject_local_attributes.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "arm",
                "branch",
                "dataset",
                "subject",
                "attribute",
                "background_MSE",
                "plus_local_vector_MSE",
                "increment",
            ]
        )
        for arm, branches in summary["continuous_coordinate_attribute_readout"].items():
            for branch, result in branches.items():
                for row in result["hierarchical_attribute_MSE"]["by_subject"]:
                    error = np.asarray(row["value"])
                    for j, attribute in enumerate(FIELDS):
                        writer.writerow(
                            [
                                arm,
                                branch,
                                row["dataset"],
                                row["subject"],
                                attribute,
                                error[0, j],
                                error[1, j],
                                1 - error[1, j] / max(error[0, j], 1e-20),
                            ]
                        )
    quantiles = {}
    for dataset in sorted(STATE["plan"]):
        with np.load(out / (dataset + "_held.npz")) as raw:
            quantiles[dataset] = {
                arm: {
                    key: np.nanquantile(
                        raw[arm + "__" + key], [0, 0.1, 0.5, 0.9, 0.95, 1], axis=0
                    )
                    for key in METRICS
                }
                for arm in ["source_Wiener", "anchored"] + names
            }
    write(
        out / "distribution_quantiles.json",
        dict(
            unit="position-window (descriptive; primary endpoints use equal dataset/subject/record hierarchy)",
            quantiles=[0, 0.1, 0.5, 0.9, 0.95, 1],
            datasets=quantiles,
        ),
    )
    write(
        out / "figure_manifest.json",
        dict(
            waveforms=figures, density_diagnostics=densities, summary="reachability.png"
        ),
    )
    write(
        out / "continuous_replay.json",
        dict(
            status="exact",
            arms=names,
            physical_windows=400,
            position_windows=17040,
            max_abs_metric_difference=replay_max,
            replay_input="saved actual real sample vectors plus frozen model files; no original waveforms passed to decoder",
            CPU_seconds=time.process_time() - started,
            validation_reads=0,
        ),
    )


def temporal_rate_worker(dataset):
    """Matched transient-source dimension curve, never an integer qualification."""
    with threadpool_limits(1):
        cfg, out = STATE["config"], STATE["output"]
        directory = out / dataset
        directory.mkdir(exist_ok=True)
        cases, _, sd = prepare(dataset, frequency_covariance=False)
        names = STATE["source"].metadata["positions"][dataset]
        cache = OrderedDict()
        for case in cases:
            fields, observed, _ = base_tokenizer_joint_eeg(
                STATE["rows"],
                dict(case["crop"], positions=[]),
                STATE["scales"],
                names,
                cache,
                allowed_splits=("train",),
            )
            np.testing.assert_array_equal(observed[None], case["observed"])
            case["native_x"] = fields["x"].T.reshape(1, len(names), 15, 400)
        measures, attr, examples, folds = defaultdict(list), defaultdict(list), [], []
        coordinate_store = {}
        supports = {
            "old_local_support": FREQUENCY_INDICES,
            "boundary_complete_support": np.arange(90, dtype=np.int64),
        }
        for fold in range(4):
            budget()
            path = (
                ROOT
                / cfg["source"]["transient_run"]
                / f"{dataset}_fold{fold}_transient.npz"
            )
            source = load_transient_source(path)
            fit = [case for case in cases if case["fold"] != fold]
            held = [case for case in cases if case["fold"] == fold]
            people = sorted({case["subject"] for case in fit})
            if set(source.metadata["fit_subjects"]) != set(people):
                raise ValueError("same source_f training people required")
            covariances = defaultdict(lambda: defaultdict(list))
            for case in cases:
                budget()
                predicted = source.predict_continuous(
                    case["native_x"], dataset, case["observed"]
                )
                case["carrier"] = {branch: predicted[branch] for branch in "SO"}
                if case["fold"] == fold:
                    continue
                item = {}
                for branch in "SO":
                    value = np.where(
                        case["observed"][..., None, None], predicted[branch], 0
                    )
                    spectrum = (
                        np.fft.rfft(value[0] / sd[branch][:, None, None], axis=-1) / 400
                    )
                    spectrum = spectrum[:, :, :90].transpose(1, 2, 0)
                    item[branch] = (
                        np.einsum(
                            "qfp,qfr->fpr", spectrum, spectrum.conj(), optimize=True
                        )
                        / 15
                    )
                covariances[case["subject"]][case["record"]].append(item)
            if fold == 0:
                probe = held[0]
                prediction = dict(
                    probe["carrier"], sum=probe["carrier"]["S"] + probe["carrier"]["O"]
                )
                metrics, _ = score_case(probe, prediction)
                with np.load(
                    ROOT / cfg["source"]["transient_run"] / f"{dataset}_metrics.npz"
                ) as prior:
                    lookup = {
                        tuple(value): n for n, value in enumerate(prior["examples"])
                    }
                    expected = prior["transient__nmse"][
                        [lookup[tuple(value)] for value in probe["examples"]]
                    ]
                    np.testing.assert_allclose(
                        metrics["nmse"], expected, rtol=1e-10, atol=1e-11
                    )
                write(
                    directory / "source_smoke.json",
                    dict(
                        status="matched",
                        fold=fold,
                        row=probe["crop"]["row"],
                        source_sha256=digest(path),
                    ),
                )
            average_covariance = {}
            for branch in "SO":
                average_covariance[branch] = np.mean(
                    [
                        np.mean(
                            [
                                np.mean([item[branch] for item in values], axis=0)
                                for values in records.values()
                            ],
                            axis=0,
                        )
                        for records in covariances.values()
                    ],
                    axis=0,
                )
            for case in held:
                predictions = {
                    "source_transient": dict(
                        case["carrier"], sum=case["carrier"]["S"] + case["carrier"]["O"]
                    )
                }
                for support, indices in supports.items():
                    predictions[support + "_full"] = projected_source(
                        case["carrier"], indices
                    )
                for arm, prediction in predictions.items():
                    metrics, _ = score_case(case, prediction, full=True)
                    for key, value in metrics.items():
                        measures[arm + "__" + key].append(value)
                examples.extend(case["examples"])
                folds.extend([fold] * len(case["examples"]))
            for support, indices in supports.items():
                for count in cfg["model"]["complex_coordinates_per_slot"]:
                    budget()
                    modes = {
                        (dataset, branch): fit_modes(
                            average_covariance[branch][indices],
                            count,
                            frequency_indices=indices,
                        )
                        for branch in "SO"
                    }
                    metadata = dict(
                        schema="frequency_spatial_carrier_v2",
                        positions={dataset: names},
                        complex_coordinates_per_slot=count,
                        local_frequency_indices=indices.tolist(),
                        source_input_identity=source.metadata["input_identity"],
                        scales_identity=source.metadata["scales_identity"],
                        fitting_subjects=people,
                        source_checkpoint=str(path),
                        source_sha256=digest(path),
                        held_fold=fold,
                        continuous_diagnostic=True,
                        context_s=30,
                        patch_s=2,
                        sample_rate_hz=200,
                        output_support_hz=[1, 45],
                        DC_energy_weight=1,
                        positive_frequency_energy_weight=2,
                        no_unencoded_mean_or_scale=True,
                    )
                    model = FrequencySpatialCarrier(
                        metadata, {(dataset, b): sd[b] for b in "SO"}, modes
                    )
                    arm = support + f"_m{count}"
                    model_file = directory / f"{arm}_fold{fold}.npz"
                    model.save(model_file)
                    model = FrequencySpatialCarrier.load(model_file)
                    fitting_coordinates = [
                        model.analyze(case["carrier"], dataset, case["observed"])
                        for case in fit
                    ]
                    heads = {
                        b: fit_local_readout(
                            fit, model, dataset, b, fitting_coordinates
                        )
                        for b in "SO"
                    }
                    head_arrays = {
                        f"{branch}_{site}_{key}": value
                        for branch, sites in heads.items()
                        for site, state in sites.items()
                        for key, value in state.items()
                    }
                    np.savez_compressed(
                        directory / f"{arm}_fold{fold}_local_heads.npz", **head_arrays
                    )
                    del fitting_coordinates
                    for ordinal, case in enumerate(held):
                        budget()
                        coordinates = model.analyze(
                            case["carrier"], dataset, case["observed"]
                        )
                        prediction = model.synthesize(coordinates, dataset)
                        metrics, _ = score_case(case, prediction, full=True)
                        for key, value in metrics.items():
                            measures[arm + "__" + key].append(value)
                        for branch in "SO":
                            attr[arm + "__" + branch].append(
                                attribute_errors(
                                    case, coordinates, heads[branch], branch
                                )
                            )
                            coordinate_store[f"{arm}_{branch}_{fold}_{ordinal}"] = (
                                coordinates[branch]
                            )
                        if ordinal == 0:
                            replay = FrequencySpatialCarrier.load(
                                model_file
                            ).synthesize(coordinates, dataset)
                            for branch in ["S", "O", "sum"]:
                                np.testing.assert_array_equal(
                                    replay[branch], prediction[branch]
                                )
                    write(
                        directory / "progress.json",
                        dict(fold=fold, arm=arm, CPU_seconds=time.process_time()),
                    )
        values = {key: np.concatenate(value) for key, value in measures.items()}
        ex = np.asarray(examples, dtype=np.int32)
        values["examples"] = ex
        values["fold"] = np.asarray(folds)
        np.savez_compressed(directory / "metrics.npz", **values)
        np.savez_compressed(
            directory / "attribute_errors.npz",
            examples=ex,
            **{k: np.concatenate(v) for k, v in attr.items()},
        )
        np.savez_compressed(
            directory / "held_continuous_coordinates.npz", **coordinate_store
        )
        with np.load(
            ROOT / cfg["source"]["transient_run"] / f"{dataset}_metrics.npz"
        ) as prior:
            lookup = {tuple(value): n for n, value in enumerate(prior["examples"])}
            expected = prior["transient__nmse"][[lookup[tuple(value)] for value in ex]]
            actual = values["source_transient__nmse"]
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
            difference = float(np.max(abs(actual - expected)))
        write(
            directory / "completion.json",
            dict(
                status="complete",
                crops=len(cases),
                positions=len(ex),
                source_reference_NMSE_max_abs=difference,
                CPU_seconds=time.process_time(),
                validation_reads=0,
            ),
        )
        return dataset, time.process_time()


def summarize_temporal_rate(out):
    joined, errors = defaultdict(list), defaultdict(list)
    for dataset in sorted(STATE["plan"]):
        with np.load(out / dataset / "metrics.npz") as f:
            for key in f:
                joined[key].append(f[key])
        with np.load(out / dataset / "attribute_errors.npz") as f:
            for key in f:
                if key != "examples":
                    errors[key].append(f[key])
    values = {key: np.concatenate(parts) for key, parts in joined.items()}
    examples = values["examples"]
    arms = sorted({key.split("__")[0] for key in values if "__" in key})
    summaries = {}
    for arm in arms:
        summaries[arm] = base_training_aggregate(
            {
                key.split("__", 1)[1]: v
                for key, v in values.items()
                if key.startswith(arm + "__")
            },
            examples,
            STATE["rows"],
        )
    write(
        out / "summary.json",
        dict(
            status="continuous_dimension_curve_complete_pending_review",
            arms=summaries,
            position_windows=len(examples),
            physical_crops=400,
            training_subjects=60,
            interpretation="empirical held distortion in declared linear transform family; coordinates are not bits; no universal rate-distortion theorem",
            integer_model=False,
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
        ),
    )
    np.savez_compressed(
        out / "attribute_errors.npz",
        examples=examples,
        **{k: np.concatenate(v) for k, v in errors.items()},
    )
    return dict(status="continuous_dimension_curve_complete_pending_review", arms=arms)


def temporal_projection_diagnostics(source, batch, selected):
    """Separate transform losses and the full-support, fixed-subspace target floor."""
    shape = source["S"].shape
    count = len(batch["x"])
    frequency = np.fft.rfftfreq(6000, 1 / 200)
    keep = (frequency >= 1) & (frequency < 45)
    global_only = {
        b: np.fft.irfft(
            np.fft.rfft(source[b].reshape(*shape[:2], 6000), axis=-1) * keep,
            n=6000,
            axis=-1,
        ).reshape(shape)
        for b in "SO"
    }
    global_only["sum"] = global_only["S"] + global_only["O"]
    predictions = {"source_full30_projection_only": global_only}
    for label, indices in [("old", FREQUENCY_INDICES), ("complete", np.arange(90))]:
        local_keep = np.zeros(201, bool)
        local_keep[indices] = True
        value = {
            b: np.fft.irfft(
                np.fft.rfft(source[b], axis=-1) * local_keep, n=400, axis=-1
            )
            for b in "SO"
        }
        value["sum"] = value["S"] + value["O"]
        predictions[label + "_patch_projection_only"] = value
    oracle = {branch: np.zeros_like(source["S"]) for branch in ["S", "O", "sum"]}
    complete = batch["mask"].reshape(count, 6000).all(1)
    below = np.full((count, 3, 1), np.nan)
    above = below.copy()
    for component, (field, branch) in enumerate([("x", "sum"), ("s", "S"), ("o", "O")]):
        truth = np.asarray(
            np.where(
                batch["mask"].reshape(count, 6000),
                batch[field].reshape(count, 6000),
                0.0,
            ),
            dtype=np.float64,
        )
        spectrum = np.fft.rfft(truth, axis=-1)
        projected = np.fft.irfft(spectrum * keep, n=6000, axis=-1)
        oracle[branch][0, selected] = projected.reshape(count, 15, 400)
        energy = abs(spectrum) ** 2
        energy[:, 1:-1] *= 2
        variance = batch["v" + field][:, 0, 0]
        below[:, component, 0] = np.where(
            complete, energy[:, frequency < 1].sum(1) / 6000**2 / variance, np.nan
        )
        above[:, component, 0] = np.where(
            complete, energy[:, frequency >= 45].sum(1) / 6000**2 / variance, np.nan
        )
    predictions["independent_target_projection_oracle"] = oracle
    return predictions, complete, below, above


def temporal_rate_review_worker(dataset):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.tokenizers.role_mixed_spatial import load_role_mixed_decoder

    with threadpool_limits(1):
        cfg, out = STATE["config"], STATE["output"]
        directory = out / dataset
        cases, _, _ = prepare(dataset, frequency_covariance=False)
        ordered = [case for fold in range(4) for case in cases if case["fold"] == fold]
        with np.load(directory / "metrics.npz") as f:
            retained = {key: f[key] for key in f}
        with np.load(directory / "attribute_errors.npz") as f:
            errors = {key: f[key] for key in f if key != "examples"}
        examples = retained["examples"]
        lookup = {tuple(value): n for n, value in enumerate(examples)}
        arms = [
            support + f"_m{m}"
            for support in ["old_local_support", "boundary_complete_support"]
            for m in cfg["model"]["complex_coordinates_per_slot"]
        ]
        joint = np.mean([retained[arm + "__nmse"].mean((1, 2)) for arm in arms], axis=0)
        identities = [
            ("fixed", min(map(tuple, examples.tolist()))),
            (
                "random_seed17",
                min(
                    map(tuple, examples.tolist()),
                    key=lambda value: ranked([17, dataset, value]),
                ),
            ),
            (
                "P95_all_declared_continuous_arms",
                tuple(examples[np.nanargmin(abs(joint - np.nanquantile(joint, 0.95)))]),
            ),
        ]
        models, heads, sources, figures = {}, {}, {}, []
        array_cache, decoder_cache = OrderedDict(), {}
        replayed, coordinate_replays = 0, 0
        diagnosis, complete_support = defaultdict(list), []
        ordinal = defaultdict(int)
        with np.load(directory / "held_continuous_coordinates.npz") as latents:
            for case in ordered:
                budget()
                fold = case["fold"]
                local_ordinal = ordinal[fold]
                ordinal[fold] += 1
                take = [lookup[tuple(value)] for value in case["examples"]]
                if fold not in sources:
                    path = (
                        ROOT
                        / cfg["source"]["transient_run"]
                        / f"{dataset}_fold{fold}_transient.npz"
                    )
                    sources[fold] = load_transient_source(path)
                names = sources[fold].metadata["positions"][dataset]
                fields, observed, _ = base_tokenizer_joint_eeg(
                    STATE["rows"],
                    dict(case["crop"], positions=[]),
                    STATE["scales"],
                    names,
                    array_cache,
                    allowed_splits=("train",),
                )
                np.testing.assert_array_equal(observed[None], case["observed"])
                source = sources[fold].predict_continuous(
                    fields["x"].T.reshape(1, len(names), 15, 400),
                    dataset,
                    observed[None],
                )
                source = {branch: source[branch] for branch in "SO"}
                predictions = {
                    "source_transient": dict(source, sum=source["S"] + source["O"])
                }
                diagnostic, complete, below, above = temporal_projection_diagnostics(
                    source, case["batch"], case["selected"]
                )
                complete_support.extend(complete)
                for diagnostic_arm, prediction in diagnostic.items():
                    actual, _ = score_case(case, prediction, full=True)
                    for name, value in actual.items():
                        diagnosis[diagnostic_arm + "__" + name].append(value)
                    if diagnostic_arm == "independent_target_projection_oracle":
                        np.testing.assert_allclose(
                            actual["nmse"][complete],
                            (below + above)[complete],
                            rtol=1e-12,
                            atol=1e-12,
                        )
                diagnosis["target_projection_below1_floor_nmse"].append(below)
                diagnosis["target_projection_above45_floor_nmse"].append(above)
                for arm in arms:
                    key = arm, fold
                    if key not in models:
                        models[key] = FrequencySpatialCarrier.load(
                            directory / f"{arm}_fold{fold}.npz"
                        )
                        head = {b: {} for b in "SO"}
                        with np.load(
                            directory / f"{arm}_fold{fold}_local_heads.npz"
                        ) as f:
                            for name in f:
                                branch, site, field = name.split("_", 2)
                                head[branch].setdefault(int(site), {})[field] = f[name]
                        heads[key] = head
                    saved = {
                        b: latents[f"{arm}_{b}_{fold}_{local_ordinal}"] for b in "SO"
                    }
                    encoded = models[key].analyze(source, dataset, observed[None])
                    for branch in "SO":
                        np.testing.assert_array_equal(saved[branch], encoded[branch])
                        actual = attribute_errors(
                            case, saved, heads[key][branch], branch
                        )
                        np.testing.assert_array_equal(
                            actual, errors[arm + "__" + branch][take]
                        )
                        coordinate_replays += 1
                    prediction = models[key].synthesize(saved, dataset)
                    actual, _ = score_case(case, prediction, full=True)
                    for name, value in actual.items():
                        np.testing.assert_array_equal(
                            value, retained[arm + "__" + name][take]
                        )
                    predictions[arm] = prediction
                    replayed += 1
                for rule, identity in identities:
                    match = [
                        j
                        for j, value in enumerate(case["examples"])
                        if tuple(value) == identity
                    ]
                    if not match:
                        continue
                    j = match[0]
                    hard_root = ROOT / cfg["source"]["matched_hard_reference"] / dataset
                    sent_ids = None
                    for file in sorted(
                        (hard_root / "integer_ids").glob("s31_o31_*.npz")
                    ):
                        with np.load(file) as f:
                            found = np.flatnonzero(
                                (f["keys"][:, 0] == identity[0])
                                & (f["keys"][:, 1] == identity[1])
                            )
                            if len(found):
                                sent_ids = {
                                    b: f[b][int(found[0]) : int(found[0]) + 1]
                                    for b in "SO"
                                }
                                assert int(f["keys"][int(found[0]), 2]) == fold
                                break
                    if sent_ids is None:
                        raise ValueError("matched integer frontier crop absent")
                    hard = load_role_mixed_decoder(
                        hard_root / "models" / f"s31_o31_fold{fold}_gain_source.npz",
                        frozen_cache=decoder_cache,
                    )
                    predictions["frozen_high_integer"] = hard.decode_ids(
                        sent_ids, dataset
                    )
                    display = [
                        "source_transient",
                        "frozen_high_integer",
                        "old_local_support_m4",
                        "boundary_complete_support_m4",
                        "boundary_complete_support_m16",
                    ]
                    palette = ["#8d979f", "#6b5595", "#397daf", "#c47925", "#328a70"]
                    native, factor = case["selected"][j], case["scales"][j]
                    mask = case["batch"]["mask"][j].ravel()
                    time_axis = np.arange(6000) / 200
                    fig, axes = plt.subplots(
                        3, 3, figsize=(18, 12), layout="constrained"
                    )
                    for line, (field, branch) in enumerate(
                        [("x", "sum"), ("s", "S"), ("o", "O")]
                    ):
                        waves = {"target": case["batch"][field][j].ravel() * factor}
                        waves.update(
                            {
                                arm: predictions[arm][branch][0, native].ravel()
                                * factor
                                for arm in display
                            }
                        )
                        low = min(v[mask].min() for v in waves.values())
                        high = max(v[mask].max() for v in waves.values())
                        margin = max((high - low) * 0.04, 1e-12)
                        for column, bounds in enumerate(
                            [(0, 30), (14, 16), (13.7, 14.3)]
                        ):
                            axis = axes[line, column]
                            for arm, color in [("target", "#172f40")] + list(
                                zip(display, palette)
                            ):
                                axis.plot(
                                    time_axis,
                                    np.where(mask, waves[arm], np.nan),
                                    color=color,
                                    lw=0.8,
                                    label=arm,
                                )
                            for boundary in range(0, 31, 2):
                                axis.axvline(
                                    boundary, color="#bac7cf", lw=0.4, zorder=0
                                )
                            unit = (
                                "native export unit"
                                if dataset == "refed"
                                else STATE["rows"][identity[0]]["source_unit"]
                            )
                            axis.set(
                                xlim=bounds,
                                ylim=(low - margin, high + margin),
                                xlabel="Visible time (s)",
                                ylabel=branch + " / " + unit,
                            )
                            if line == 0:
                                axis.set_title(
                                    ["Full30s", "Fixed2s", "Token boundary"][column]
                                )
                            if line == column == 0:
                                axis.legend(fontsize=7, ncol=2)
                    row = STATE["rows"][identity[0]]
                    fig.suptitle(
                        f"{dataset}/{row['subject']}/{row['record']}/{row['positions'][identity[2]]}/+{identity[1]}s/{rule}\nContinuous coordinate diagnostics; integer frontier is a different frozen transform; mask {mask.sum()}/6000",
                        fontsize=11,
                    )
                    file = out / f"{dataset}_{rule}.png"
                    fig.savefig(file, dpi=130)
                    plt.close(fig)
                    figures.append(
                        dict(
                            path=file.name,
                            example=list(identity),
                            subject=row["subject"],
                            record=row["record"],
                            site=row["positions"][identity[2]],
                            rule=rule,
                            fold=fold,
                            valid_points=int(mask.sum()),
                            full_context_points=6000,
                        )
                    )
        result = dict(
            status="exact",
            replayed_crop_model_pairs=replayed,
            branch_coordinate_and_attribute_replays=coordinate_replays,
            all_original_metrics_array_equal=True,
            all_X_only_source_coordinate_arrays_equal=True,
            figures=figures,
            CPU_seconds=time.process_time(),
            validation_reads=0,
        )
        write(directory / "independent_continuous_replay.json", result)
        np.savez_compressed(
            directory / "projection_failure_diagnosis.npz",
            examples=examples,
            complete_6000_point_support=np.asarray(complete_support),
            **{key: np.concatenate(value) for key, value in diagnosis.items()},
        )
        return result


def review_temporal_rate(out):
    frozen = json.loads((out / "resolved_config.json").read_text())["config"]
    if frozen != json.loads(json.dumps(STATE["config"])):
        raise ValueError("temporal-coordinate review contract changed")
    prior = float(os.environ["TOKENIZER_TEMPORAL_RATE_PRIOR_CPU"])
    STATE["config"]["resources"]["prior_CPU_seconds"] = prior
    directories = [out/dataset for dataset in sorted(STATE["plan"])]
    if all((directory/"independent_continuous_replay.json").exists() and
           (directory/"projection_failure_diagnosis.npz").exists() for directory in directories):
        reports = [json.loads((directory/"independent_continuous_replay.json").read_text()) for directory in directories]
        if any(record["status"] != "exact" for record in reports):
            raise ValueError("only complete exact retained replay may resume summary")
    else:
        with ProcessPoolExecutor(
            max_workers=4, mp_context=multiprocessing.get_context("fork")
        ) as pool:
            reports = list(pool.map(temporal_rate_review_worker, sorted(STATE["plan"])))
    write(
        out / "independent_continuous_replay.json",
        dict(
            status="exact",
            datasets=reports,
            replayed_crop_model_pairs=sum(
                r["replayed_crop_model_pairs"] for r in reports
            ),
            branch_coordinate_and_attribute_replays=sum(
                r["branch_coordinate_and_attribute_replays"] for r in reports
            ),
            validation_reads=0,
        ),
    )
    write(
        out / "figure_manifest.json",
        dict(
            status="rendered_pending_actual_open",
            figures=[f for r in reports for f in r["figures"]],
        ),
    )
    summary = json.loads((out / "summary.json").read_text())
    reference = json.loads(
        (ROOT / frozen["source"]["matched_hard_reference"] / "summary.json").read_text()
    )["arms"]["s31_o31_gain_source"]["metrics"]
    pattern, decisions = {}, {}
    with np.load(out / "attribute_errors.npz") as f:
        examples = f["examples"]
        for arm, record in summary["arms"].items():
            metrics = record["metrics"]
            nmse = np.asarray(metrics["nmse"]["macro"])
            ref_nmse = np.asarray(reference["nmse"]["macro"])
            gain = np.asarray(metrics["rms_gain"]["macro"])
            checks = dict(
                X_NMSE=bool(nmse[0, 0] <= 1.05 * ref_nmse[0, 0]),
                S_O_NMSE=bool(np.all(nmse[1:] <= 1.10 * ref_nmse[1:])),
                dataset=all(
                    np.all(
                        np.asarray(value)
                        <= 1.10 * np.asarray(reference["nmse"]["by_dataset"][d])
                    )
                    for d, value in metrics["nmse"]["by_dataset"].items()
                ),
                amplitude=bool(np.all((gain >= 0.75) & (gain <= 1.25))),
                theta_alpha=all(
                    np.all(abs(np.asarray(metrics[k]["macro"])) <= 3.0103)
                    for k in [
                        "band_4_8_db",
                        "band_8_13_db",
                        "band_4_8_abs_db",
                        "band_8_13_abs_db",
                    ]
                ),
                envelope=bool(
                    np.all(np.asarray(metrics["local_rms_correlation"]["macro"]) >= 0.5)
                ),
                correlation=bool(
                    np.all(np.asarray(metrics["correlation"]["macro"]) > 0)
                ),
            )
            if "_m" in arm:
                pattern[arm] = {}
                for branch in "SO":
                    aggregation = base_training_aggregate(
                        {"nmse": f[arm + "__" + branch]}, examples, STATE["rows"]
                    )["metrics"]["nmse"]
                    mse = np.asarray(aggregation["macro"])
                    increment = 1 - mse[1] / mse[0]
                    fraction = np.mean(
                        [
                            np.asarray(v["value"])[1] < np.asarray(v["value"])[0]
                            for v in aggregation["by_subject"]
                        ],
                        axis=0,
                    )
                    pattern[arm][branch] = dict(
                        gain=dict(zip(FIELDS, increment)),
                        positive_subject_fraction=dict(zip(FIELDS, fraction)),
                        hierarchical_attribute_MSE=aggregation,
                        passed=bool(
                            min(increment[:2]) > 0
                            and max(increment) >= 0.05
                            and min(fraction[:2]) >= 0.5
                        ),
                    )
                checks["local_coordinate_information"] = all(
                    pattern[arm][branch]["passed"] for branch in "SO"
                )
            decisions[arm] = dict(
                checks=checks,
                waveform_pass=all(
                    value
                    for key, value in checks.items()
                    if key != "local_coordinate_information"
                ),
                joint_continuous_pass=all(checks.values()) if "_m" in arm else None,
            )
    summary["continuous_coordinate_information"] = pattern
    summary["decisions"] = decisions
    summary["smallest_supported_complex_coordinates_per_slot"] = {
        support: next(
            (
                m
                for m in frozen["model"]["complex_coordinates_per_slot"]
                if decisions[support + f"_m{m}"]["joint_continuous_pass"]
            ),
            None,
        )
        for support in ["old_local_support", "boundary_complete_support"]
    }
    summary["status"] = "numerical_replay_complete_pending_actual_image_open"
    summary["interpretation"] = (
        "Matched empirical continuous-coordinate reachability; local attributes are not discrete-ID proof and dimension is not bitrate. Separate quantizer and untouched evaluation would be required."
    )
    diagnostic_arrays = defaultdict(list)
    for dataset in sorted(STATE["plan"]):
        with np.load(out / dataset / "projection_failure_diagnosis.npz") as f:
            for key in f:
                diagnostic_arrays[key].append(f[key])
    diagnostic_values = {
        key: np.concatenate(parts) for key, parts in diagnostic_arrays.items()
    }
    diagnostic_examples = diagnostic_values["examples"]
    complete = diagnostic_values["complete_6000_point_support"]
    diagnostic_names = sorted(
        {key.split("__")[0] for key in diagnostic_values if "__" in key}
    )
    diagnostic_summaries = {
        name: base_training_aggregate(
            {
                key.split("__", 1)[1]: value
                for key, value in diagnostic_values.items()
                if key.startswith(name + "__")
            },
            diagnostic_examples,
            STATE["rows"],
        )
        for name in diagnostic_names
    }
    for dataset in sorted(STATE["plan"]):
        with (
            np.load(out / dataset / "projection_failure_diagnosis.npz") as d,
            np.load(out / dataset / "metrics.npz") as m,
        ):
            take = d["complete_6000_point_support"]
            floor = d["independent_target_projection_oracle__nmse"][take]
            for arm in pattern:
                if np.any(m[arm + "__nmse"][take] < floor - 1e-10):
                    raise AssertionError(
                        "full-mask fixed-subspace oracle floor violated"
                    )
    floor_parts = base_training_aggregate(
        {
            "nmse": diagnostic_values["target_projection_below1_floor_nmse"] + diagnostic_values["target_projection_above45_floor_nmse"],
            "below1_nmse": diagnostic_values["target_projection_below1_floor_nmse"],
            "above45_nmse": diagnostic_values["target_projection_above45_floor_nmse"],
        },
        diagnostic_examples,
        STATE["rows"],
    )
    write(
        out / "projection_failure_diagnosis.json",
        dict(
            status="complete_posthoc_deterministic_diagnosis",
            arms=diagnostic_summaries,
            complete_support_positions=int(complete.sum()),
            partial_support_positions=int((~complete).sum()),
            fixed_subspace_floor_parts=floor_parts,
            floor_scope="exact orthogonal-projection NMSE floor only on all6000-valid-point windows for any full30s DFT output restricted to 1<=f<45Hz; not a general bitrate theorem",
            partial_scope="partial masks retained in all numerical reports; target oracle zero-fills invalid labels and is diagnostic rather than a bound there",
            oracle="independent projection of measured X,S,O, not an encoder and not a closed-component candidate",
            selection_unchanged=True,
            validation_reads=0,
        ),
    )
    summary["projection_failure_diagnosis"] = "projection_failure_diagnosis.json"
    write(out / "summary.json", summary)
    render_temporal_rate_curves(out, summary, reference)
    write(
        out / "review_completion.json",
        dict(
            status="numerical_replay_and_render_complete_pending_actual_open",
            replayed_crop_model_pairs=3200,
            branch_coordinate_and_attribute_replays=6400,
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
        ),
    )


def render_temporal_rate_curves(out, summary, reference):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    supports = [
        ("old_local_support", "#397daf"),
        ("boundary_complete_support", "#c47925"),
    ]
    counts = STATE["config"]["model"]["complex_coordinates_per_slot"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), layout="constrained")
    for column, component in enumerate(["X", "S", "O"]):
        for row, metric in enumerate(["nmse", "rms_gain", "band_8_13_abs_db"]):
            axis = axes[row, column]
            for support, color in supports:
                values = [
                    summary["arms"][support + f"_m{m}"]["metrics"][metric]["macro"][
                        column
                    ][0]
                    for m in counts
                ]
                axis.plot(counts, values, color=color, marker="o", label=support)
                floor = summary["arms"][support + "_full"]["metrics"][metric]["macro"][
                    column
                ][0]
                axis.axhline(
                    floor,
                    color=color,
                    ls=":",
                    lw=0.8,
                    label=support + " full projection",
                )
            axis.axhline(
                reference[metric]["macro"][column][0],
                color="#6b5595",
                ls="--",
                label="frozen high integer frontier (different transform)",
            )
            axis.axhline(
                summary["arms"]["source_transient"]["metrics"][metric]["macro"][column][
                    0
                ],
                color="#8d979f",
                lw=0.8,
                label="continuous source_f",
            )
            if metric == "band_8_13_abs_db":
                axis.axhline(3.0103, color="black", ls=":", lw=0.7)
            if metric == "rms_gain":
                for value in [0.75, 1.25]:
                    axis.axhline(value, color="black", ls=":", lw=0.7)
            axis.set(
                xscale="log",
                xticks=counts,
                xticklabels=counts,
                xlabel="Complex coefficients per physical 2s slot (not bits)",
                ylabel=metric,
                title=component + " / " + metric,
            )
            if row == column == 0:
                axis.legend(fontsize=6)
    fig.suptitle(
        "Matched 400-array / 60-person / 4-fold continuous dimension curve; all original masks and hierarchy",
        fontsize=11,
    )
    fig.savefig(out / "effective_dimension_curves.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for row, branch in enumerate("SO"):
        for column, field in enumerate(FIELDS):
            axis = axes[row, column]
            for support, color in supports:
                values = [
                    summary["continuous_coordinate_information"][support + f"_m{m}"][
                        branch
                    ]["gain"][field]
                    for m in counts
                ]
                axis.plot(counts, values, color=color, marker="o", label=support)
            axis.axhline(0, color="black", lw=0.6)
            axis.axhline(0.05, color="grey", ls=":", lw=0.7)
            axis.set(
                xscale="log",
                xticks=counts,
                xticklabels=counts,
                xlabel="Complex coefficients per physical 2s slot",
                ylabel="Held MSE reduction over position/amplitude background",
                title=branch + " / " + field,
            )
            axis.legend(fontsize=7)
    fig.suptitle(
        "Actual local continuous coordinates; train-fitted readouts on held people, not single-ID evidence",
        fontsize=11,
    )
    fig.savefig(out / "local_coordinate_information_curves.png", dpi=150)
    plt.close(fig)
    allocation = []
    for dataset in sorted(STATE["plan"]):
        for support, _ in supports:
            for m in counts:
                for fold in range(4):
                    model = FrequencySpatialCarrier.load(
                        out / dataset / f"{support}_m{m}_fold{fold}.npz"
                    )
                    for branch in "SO":
                        modes = model.modes[dataset, branch]
                        frequency = modes["frequency"] / 2
                        rank = (
                            len(model.metadata["positions"][dataset])
                            - modes["selected_mode_number"]
                        )
                        real_dimensions = np.where(frequency == 0, 1, 2)
                        allocation.append(
                            dict(
                                dataset=dataset,
                                support=support,
                                m=m,
                                fold=fold,
                                branch=branch,
                                effective_real_dimensions_by_physical_slot=real_dimensions.sum(
                                    1
                                ),
                                spatial_rank_counts=dict(
                                    rank1=int(np.sum(rank == 1)),
                                    rank2_to4=int(np.sum((rank >= 2) & (rank <= 4))),
                                    rank5_plus=int(np.sum(rank >= 5)),
                                ),
                                short_window_auxiliary_complex_count=int(
                                    np.sum(frequency < 1)
                                ),
                                total_complex_coordinates_per_30s=int(
                                    frequency.size * 15
                                ),
                                total_effective_real_coordinates_per_30s=int(
                                    real_dimensions.sum() * 15
                                ),
                            )
                        )
    write(
        out / "mode_allocation.json",
        dict(
            records=allocation,
            interpretation="Frequency-specific covariance rank allocation; rank1 is leading shared mode, higher ranks are remaining spatial variation. Counts are diagnostic coordinate dimensions, not bitrate.",
        ),
    )


def temporal_local_diagnosis_worker(dataset):
    with threadpool_limits(1):
        cfg, out = STATE["config"], STATE["output"]
        directory = out / dataset
        directory.mkdir(exist_ok=True)
        basis = ROOT / cfg["source"]["basis_run"]
        cases, _, _ = prepare(dataset, frequency_covariance=False)
        names = STATE["source"].metadata["positions"][dataset]
        cache = OrderedDict()
        spectra = []
        spectral_examples = []
        captures = {}
        for case in cases:
            budget()
            fields, observed, _ = base_tokenizer_joint_eeg(
                STATE["rows"],
                dict(case["crop"], positions=[]),
                STATE["scales"],
                names,
                cache,
                allowed_splits=("train",),
            )
            np.testing.assert_array_equal(observed[None], case["observed"])
            case["native_x"] = fields["x"].T.reshape(1, len(names), 15, 400)
            x = case["native_x"][0, case["selected"]].reshape(
                len(case["examples"]), 6000
            )
            valid = observed[case["selected"]]
            record = {}
            for label, window in [
                ("rectangular", np.ones(6000)),
                ("Hann", np.hanning(6000)),
            ]:
                p = abs(np.fft.rfft(x * window, axis=-1)) ** 2
                p[:, 1:-1] *= 2
                f = np.fft.rfftfreq(6000, 1 / 200)
                den = p.sum(1)
                for name, keep in [
                    ("above45", f >= 45),
                    ("line49_51", (f >= 49) & (f <= 51)),
                ]:
                    record[label + "_" + name] = np.where(
                        valid,
                        np.divide(
                            p[:, keep].sum(1),
                            den,
                            out=np.full(len(x), np.nan),
                            where=den > 0,
                        ),
                        np.nan,
                    )
                if label == "Hann":
                    record["Hann_peak_above45_Hz"] = f[f >= 45][
                        np.argmax(p[:, f >= 45], axis=1)
                    ]
                    j = int(np.nanargmax(record["Hann_above45"]))
                    candidate = dict(
                        rank=float(record["Hann_above45"][j]),
                        example=case["examples"][j],
                        x=x[j],
                        scale=case["scales"][j],
                        subject=case["subject"],
                        record=case["record"],
                        site=names[case["selected"][j]],
                    )
                    if (
                        not captures
                        or candidate["rank"]
                        > captures["maximum_Hann_above45_fraction"]["rank"]
                    ):
                        captures["maximum_Hann_above45_fraction"] = candidate
            spectra.append(record)
            spectral_examples.extend(case["examples"])
        spectral = {
            key: np.concatenate([r[key] for r in spectra]) for key in spectra[0]
        }
        np.savez_compressed(
            directory / "observation_spectral_audit.npz",
            examples=np.asarray(spectral_examples),
            **spectral,
        )
        np.savez_compressed(
            directory / "maximum_Hann_above45_fraction.npz",
            **captures["maximum_Hann_above45_fraction"],
        )
        errors, examples = defaultdict(list), []
        with np.load(basis / dataset / "attribute_errors.npz") as prior:
            lookup = {tuple(e): n for n, e in enumerate(prior["examples"])}
            for fold in range(4):
                source = load_transient_source(
                    ROOT
                    / cfg["source"]["transient_run"]
                    / f"{dataset}_fold{fold}_transient.npz"
                )
                fit = [case for case in cases if case["fold"] != fold]
                held = [case for case in cases if case["fold"] == fold]
                if set(source.metadata["fit_subjects"]) != {c["subject"] for c in fit}:
                    raise ValueError("same held source_f required")
                for case in cases:
                    budget()
                    p = source.predict_continuous(
                        case["native_x"], dataset, case["observed"]
                    )
                    case["carrier"] = {b: p[b] for b in "SO"}
                    case["source_attributes"] = {
                        b: selected_attributes(
                            p[b][0], np.ones((len(names), 15), bool)
                        )[None]
                        for b in "SO"
                    }
                for label, m, feature_basis in [
                    ("native_source_fitted", 2, "identity")
                ] + [
                    (f"power_m{m}", m, "power_only")
                    for m in cfg["model"]["complex_coordinates_per_slot"]
                ]:
                    model = FrequencySpatialCarrier.load(
                        basis
                        / dataset
                        / f"boundary_complete_support_m{m}_fold{fold}.npz"
                    )

                    def coordinates(case):
                        return (
                            case["source_attributes"]
                            if feature_basis == "identity"
                            else model.analyze(
                                case["carrier"], dataset, case["observed"]
                            )
                        )

                    fitted_coordinates = [coordinates(case) for case in fit]
                    heads = {
                        b: fit_local_readout(
                            fit,
                            model,
                            dataset,
                            b,
                            fitted_coordinates,
                            feature_basis=feature_basis,
                        )
                        for b in "SO"
                    }
                    del fitted_coordinates
                    np.savez_compressed(
                        directory / f"{label}_fold{fold}_heads.npz",
                        **{
                            f"{b}_{site}_{key}": value
                            for b, states in heads.items()
                            for site, state in states.items()
                            for key, value in state.items()
                        },
                    )
                    for case in held:
                        z = coordinates(case)
                        take = [lookup[tuple(e)] for e in case["examples"]]
                        for branch in "SO":
                            result = attribute_errors(case, z, heads[branch], branch)
                            original = prior[
                                f"boundary_complete_support_m{m}__{branch}"
                            ][take]
                            np.testing.assert_array_equal(result[:, 0], original[:, 0])
                            errors[label + "__" + branch].append(result)
                            if feature_basis == "identity":
                                direct = result.copy()
                                for j, site in enumerate(case["selected"]):
                                    valid = case["complete"][j]
                                    if valid.any():
                                        direct[j, 1] = feature_means(
                                            (
                                                z[branch][0, site, valid]
                                                - case["targets"][branch][j, valid]
                                            )
                                            ** 2,
                                            FIELDS,
                                        ).mean(0)
                                errors["native_source_direct__" + branch].append(direct)
                    write(
                        directory / "progress.json",
                        dict(fold=fold, readout=label, CPU_seconds=time.process_time()),
                    )
                for case in held:
                    examples.extend(case["examples"])
        np.savez_compressed(
            directory / "attribute_errors.npz",
            examples=np.asarray(examples),
            **{k: np.concatenate(v) for k, v in errors.items()},
        )
        write(
            directory / "completion.json",
            dict(
                status="complete_posthoc_diagnosis",
                CPU_seconds=time.process_time(),
                model_or_original_selection_change=False,
                validation_reads=0,
            ),
        )
        return dataset, time.process_time()


def summarize_temporal_local_diagnosis(out):
    data, ex = defaultdict(list), []
    spectra, sex = defaultdict(list), []
    for dataset in sorted(STATE["plan"]):
        with np.load(out / dataset / "attribute_errors.npz") as f:
            ex.append(f["examples"])
            for key in f:
                if key != "examples":
                    data[key].append(f[key])
        with np.load(out / dataset / "observation_spectral_audit.npz") as f:
            sex.append(f["examples"])
            for key in f:
                if key != "examples":
                    spectra[key].append(f[key])
    ex = np.concatenate(ex)
    sex = np.concatenate(sex)
    results = {}
    for key, parts in data.items():
        stats = base_training_aggregate(
            {"nmse": np.concatenate(parts)}, ex, STATE["rows"]
        )["metrics"]["nmse"]
        mse = np.asarray(stats["macro"])
        increment = 1 - mse[1] / mse[0]
        fraction = np.mean(
            [
                np.asarray(v["value"])[1] < np.asarray(v["value"])[0]
                for v in stats["by_subject"]
            ],
            axis=0,
        )
        results[key] = dict(
            gain=dict(zip(FIELDS, increment)),
            positive_subject_fraction=dict(zip(FIELDS, fraction)),
            hierarchy=stats,
        )
    spectral_summary = {
        key: base_training_aggregate(
            {"nmse": np.concatenate(v)[:, None, None]}, sex, STATE["rows"]
        )["metrics"]["nmse"]
        for key, v in spectra.items()
    }
    write(
        out / "summary.json",
        dict(
            status="complete_posthoc_readout_and_spectrum_diagnosis",
            local_information=results,
            observation_spectrum=spectral_summary,
            original_candidate_gates_changed=False,
            discrete_model_trained=False,
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
            scope="diagnose phase-feature overfitting vs packet-local information with one fixed power-only readout and source-native positive controls; inspect observed X Hann spectrum for crop leakage vs persistent high-frequency support",
        ),
    )
    return dict(status="complete_posthoc_diagnosis", readouts=list(results))


def render_temporal_local_diagnosis(out):
    import csv
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = STATE["config"]
    if json.loads((out / "resolved_config.json").read_text())["config"] != json.loads(
        json.dumps(cfg)
    ):
        raise ValueError("local diagnostic render contract changed")
    summary = json.loads((out / "summary.json").read_text())
    basis = ROOT / cfg["source"]["basis_run"]
    figures = []
    samples = []
    cache = OrderedDict()
    for dataset in sorted(STATE["plan"]):
        with np.load(out / dataset / "maximum_Hann_above45_fraction.npz") as f:
            capture = {key: f[key] for key in f}
        example = np.asarray(capture["example"], int)
        row = STATE["rows"][int(example[0])]
        batch = base_tokenizer_batch(
            STATE["rows"],
            example[None],
            STATE["scales"],
            array_cache=cache,
            cache_limit=8,
        )
        mask = batch["mask"][0].ravel()
        window = np.hanning(6000) * mask
        traces = {
            "X": np.asarray(capture["x"], float),
            "S": np.asarray(batch["s"][0], float).ravel(),
            "O": np.asarray(batch["o"][0], float).ravel(),
        }
        frequency = np.fft.rfftfreq(6000, 1 / 200)
        spectra = {name: np.fft.rfft(value * window) for name, value in traces.items()}
        psd = {
            name: abs(value) ** 2 / (200 * np.sum(window**2))
            for name, value in spectra.items()
        }
        for value in psd.values():
            value[1:-1] *= 2
        hi = {name: float(value[frequency >= 45].sum()) for name, value in psd.items()}
        high_ratio = {name: value / hi["X"] for name, value in hi.items()}
        closure = float(
            np.sqrt(
                np.mean(
                    (traces["X"][mask] - traces["S"][mask] - traces["O"][mask]) ** 2
                )
            )
        )
        projection = np.fft.irfft(
            np.fft.rfft(traces["X"]) * ((frequency >= 1) & (frequency < 45)), n=6000
        )
        colors = {"X": "#172f40", "S": "#348a73", "O": "#c47925"}
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
        for axis, bounds in zip(axes[0], [(0, 100), (1, 45)]):
            for name, value in psd.items():
                axis.semilogy(
                    frequency, np.maximum(value, 1e-24), color=colors[name], label=name
                )
            axis.axvline(
                45, color="#a74242", ls="--", lw=0.8, label="hard decoder cutoff45Hz"
            )
            axis.axvline(
                50, color="#6b5595", ls=":", lw=0.8, label="50Hz artifact audit"
            )
            axis.set(
                xlim=bounds,
                xlabel="Frequency (Hz)",
                ylabel="PSD / frozen input SD squared per Hz",
                title="Hann PSD on the same target-valid points",
            )
            axis.legend(fontsize=8)
        factor = float(capture["scale"])
        time_axis = np.arange(6000) / 200
        unit = "native export unit" if dataset == "refed" else row["source_unit"]
        for axis, bounds in zip(axes[1], [(14, 16), (14, 14.1)]):
            for name, value in traces.items():
                axis.plot(
                    time_axis,
                    np.where(mask, value * factor, np.nan),
                    color=colors[name],
                    lw=0.9,
                    label=name + " measured target",
                )
            axis.plot(
                time_axis,
                np.where(mask, projection * factor, np.nan),
                color="#a74242",
                ls="--",
                lw=0.9,
                label="ideal 30s target projection; diagnostic",
            )
            axis.axvline(14, color="grey", lw=0.6)
            selected = (time_axis >= bounds[0]) & (time_axis <= bounds[1]) & mask
            if selected.any():
                low = min((v * factor)[selected].min() for v in traces.values())
                high = max((v * factor)[selected].max() for v in traces.values())
            else:
                low, high = -factor, factor
                axis.text(
                    0.5,
                    0.5,
                    "No target-valid points in this fixed zoom",
                    transform=axis.transAxes,
                    ha="center",
                )
            margin = max((high - low) * 0.07, 1e-12)
            axis.set(
                xlim=bounds,
                ylim=(low - margin, high + margin),
                xlabel="Visible time (s)",
                ylabel=unit,
                title="Same physical position: X and its original S/O targets",
            )
            axis.legend(fontsize=7)
        fig.suptitle(
            f"{dataset}/{row['subject']}/{row['record']}/{row['positions'][example[2]]}/+{example[1]}s\nMaximum observed-X Hann >=45Hz fraction: {float(capture['rank']):.4%}; target mask {int(mask.sum())}/6000. Band endpoints remain1–45Hz; 50Hz is not semantic gain.",
            fontsize=11,
        )
        file = out / f"{dataset}_maximum_Hann_above45_spectrum.png"
        fig.savefig(file, dpi=150)
        plt.close(fig)
        record = dict(
            path=file.name,
            example=example,
            subject=row["subject"],
            record=row["record"],
            site=row["positions"][example[2]],
            observation_Hann_above45_fraction=float(capture["rank"]),
            matched_target_mask_points=int(mask.sum()),
            S_and_O_high_power_divided_by_X_high_power=high_ratio,
            closure_RMSE_input_SD=closure,
            allocation_note="ratios are separate powers, not additive fractions; all same native position/scale/mask",
        )
        figures.append(record)
        samples.append(record)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    original = json.loads((basis / "summary.json").read_text())
    counts = cfg["model"]["complex_coordinates_per_slot"]
    for row, branch in enumerate("SO"):
        for column, field in enumerate(FIELDS):
            axis = axes[row, column]
            axis.plot(
                counts,
                [
                    original["continuous_coordinate_information"][
                        f"boundary_complete_support_m{m}"
                    ][branch]["gain"][field]
                    for m in counts
                ],
                color="#397daf",
                marker="o",
                label="original phase+power",
            )
            axis.plot(
                counts,
                [
                    summary["local_information"][f"power_m{m}__{branch}"]["gain"][field]
                    for m in counts
                ],
                color="#c47925",
                marker="o",
                label="power-only posthoc diagnostic",
            )
            for label, color in [
                ("native_source_fitted", "#348a73"),
                ("native_source_direct", "#6b5595"),
            ]:
                axis.axhline(
                    summary["local_information"][label + "__" + branch]["gain"][field],
                    color=color,
                    ls="--",
                    label=label,
                )
            axis.axhline(0, color="black", lw=0.6)
            axis.axhline(0.05, color="grey", ls=":", lw=0.6)
            axis.set(
                xticks=counts,
                xlabel="Complex coefficients per physical slot",
                ylabel="Held MSE reduction over same background",
                title=branch + " / " + field,
            )
            axis.legend(fontsize=6)
    fig.suptitle(
        "Native source attributes are informative; phase-invariant readout does not rescue the frozen nonlocal packets. Original eligibility unchanged.",
        fontsize=11,
    )
    fig.savefig(out / "local_information_positive_controls.png", dpi=150)
    plt.close(fig)
    figures.append(
        dict(
            path="local_information_positive_controls.png",
            selection="all predefined readouts",
        )
    )
    with (out / "same_position_high_frequency_allocation.csv").open(
        "w", newline=""
    ) as h:
        writer = csv.writer(h)
        writer.writerow(
            [
                "dataset",
                "subject",
                "record",
                "row",
                "start_s",
                "position",
                "full6000_mask",
                "X_HF_NMSE",
                "S_HF_NMSE",
                "O_HF_NMSE",
                "X_HF_RMS_native",
                "S_HF_RMS_native",
                "O_HF_RMS_native",
                "S_power_over_X_power",
                "O_power_over_X_power",
            ]
        )
        allocation = {}
        for dataset in sorted(STATE["plan"]):
            with np.load(basis / dataset / "projection_failure_diagnosis.npz") as f:
                examples = f["examples"]
                full = f["complete_6000_point_support"]
                high = f["target_projection_above45_floor_nmse"]
                low = f["target_projection_below1_floor_nmse"]
                energy = f[
                    "independent_target_projection_oracle__rmse_native"
                ] ** 2 * np.divide(
                    high,
                    high + low,
                    out=np.full_like(high, np.nan),
                    where=(high + low) > 0,
                )
                ratios = energy[:, 1:, 0] / energy[:, :1, 0]
                for i, (r, start, position) in enumerate(examples):
                    metadata = STATE["rows"][r]
                    writer.writerow(
                        [
                            dataset,
                            metadata["subject"],
                            metadata["record"],
                            int(r),
                            int(start),
                            int(position),
                            bool(full[i]),
                        ]
                        + high[i, :, 0].tolist()
                        + np.sqrt(energy[i, :, 0]).tolist()
                        + ratios[i].tolist()
                    )
                allocation[dataset] = {
                    name: base_training_aggregate(
                        {"nmse": value}, examples, STATE["rows"]
                    )["metrics"]["nmse"]
                    for name, value in [
                        ("true_above45_NMSE", high),
                        ("true_above45_RMS_native", np.sqrt(energy)),
                        ("separate_S_O_power_ratios", ratios[:, None, :]),
                    ]
                }
    write(
        out / "same_position_high_frequency_allocation.json",
        dict(
            scope="all original positions retained; exact above45 fixed-subspace floor and native RMS allocation are reported only on full6000 target masks",
            by_dataset=allocation,
            selected_observation_tail_samples=samples,
            oracle_is_not_encoder=True,
            frequency_semantic_endpoints_unchanged=[1, 45],
            target_arrays_changed=False,
        ),
    )
    write(
        out / "figure_manifest.json",
        dict(status="rendered_pending_actual_open", figures=figures),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--temporal-rate-review", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with (out / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "summary.json").exists() and not (
            args.review or args.temporal_rate_review
        ):
            raise FileExistsError(
                "retained completed compact frequency-spatial experiment"
            )
        if (
            cfg["schema"]
            not in [
                "frequency_spatial_carrier_v1",
                "staggered_time_carrier_v1",
                "empirical_temporal_spatial_rate_v1",
                "temporal_local_information_diagnosis_v1",
            ]
            or cfg["data"]["test_access"] != "forbidden"
            or cfg["data"]["scoring_splits"] != ["train"]
        ):
            raise ValueError(
                "versioned train-only compact coordinate contract required"
            )
        g0 = yaml.safe_load((ROOT / cfg["source"]["g0_config"]).read_text())
        manifest, scales, _, _ = base_tokenizer_training_data(g0, ROOT)
        plan = json.loads((ROOT / cfg["source"]["plan"]).read_text())["plan"]
        folds = json.loads((ROOT / cfg["source"]["folds"]).read_text())
        source = load_model(ROOT / cfg["source"]["Wiener"])
        old = json.loads(
            (
                ROOT / cfg["source"]["waveform_cache"] / "resolved_config.json"
            ).read_text()
        )
        if (
            old["input_identity"] != manifest["cfg_identity"]
            or old["scales_identity"] != manifest["scales_identity"]
            or old["source_sha256"] != digest(ROOT / cfg["source"]["Wiener"])
        ):
            raise ValueError("frozen cache/source/G0 identity mismatch before arrays")
        for d, subjects in plan.items():
            if set(folds[d]) != set(subjects):
                raise ValueError("same source subject folds required")
            for s, crops in subjects.items():
                if any(
                    manifest["rows"][c["row"]]["split"] != "train"
                    or manifest["rows"][c["row"]]["subject"] != s
                    for c in crops
                ):
                    raise ValueError("G0 training scope before arrays")
        if not (args.review or args.temporal_rate_review):
            write(
                out / "resolved_config.json",
                dict(
                    config=cfg,
                    plan=plan,
                    folds=folds,
                    source_sha256=digest(ROOT / cfg["source"]["Wiener"]),
                    source_cache_identity=old,
                ),
            )
        STATE.update(
            config=cfg,
            config_path=args.config,
            output=out,
            rows=manifest["rows"],
            scales=scales,
            plan=plan,
            folds=folds,
            source=source,
        )
        if args.temporal_rate_review:
            if cfg["schema"] == "temporal_local_information_diagnosis_v1":
                render_temporal_local_diagnosis(out)
                return
            if cfg["schema"] != "empirical_temporal_spatial_rate_v1":
                raise ValueError("temporal rate review requires its frozen contract")
            review_temporal_rate(out)
            return
        if args.review:
            if cfg["schema"] != "staggered_time_carrier_v1":
                raise ValueError("review mode owns the staggered diagnostic only")
            spent = sum(
                json.loads((out / "completion.json").read_text())[
                    "usage_CPU_seconds_by_dataset"
                ].values()
            )
            STATE["CPU_limit"] = cfg["resources"]["CPU_budget_seconds"] - spent - 60
            if STATE["CPU_limit"] <= 0:
                raise RuntimeError(
                    "declared total CPU budget exhausted before visual review"
                )
            review_staggered(out)
            return
        started = time.monotonic()
        with ProcessPoolExecutor(
            max_workers=cfg["resources"]["dataset_workers"],
            mp_context=multiprocessing.get_context("fork"),
        ) as pool:
            if cfg["schema"] == "temporal_local_information_diagnosis_v1":
                worker = temporal_local_diagnosis_worker
            elif cfg["schema"] == "empirical_temporal_spatial_rate_v1":
                worker = temporal_rate_worker
            elif cfg["schema"] == "staggered_time_carrier_v1":
                worker = staggered_dataset_worker
            else:
                worker = dataset_worker
            usage = dict(pool.map(worker, sorted(plan)))
        if cfg["schema"] == "temporal_local_information_diagnosis_v1":
            result = summarize_temporal_local_diagnosis(out)
        elif cfg["schema"] == "empirical_temporal_spatial_rate_v1":
            result = summarize_temporal_rate(out)
        elif cfg["schema"] == "staggered_time_carrier_v1":
            result = summarize_staggered(out)
        else:
            result = summarize(out)
        write(
            out / "completion.json",
            dict(
                status="completed",
                usage_CPU_seconds_by_dataset=usage,
                elapsed_seconds=time.monotonic() - started,
                decisions=result,
                validation_reads=0,
            ),
        )
        print(
            json.dumps(
                result, default=lambda x: x.item() if isinstance(x, np.generic) else x
            )
        )


if __name__ == "__main__":
    main()
