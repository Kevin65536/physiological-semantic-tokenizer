#!/usr/bin/env python3
"""Resolve frozen spatial EEG reconstruction error into additive frequency contributions."""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
import yaml

from src.data.dual_component_tokenizer import (
    base_tokenizer_batch,
    base_tokenizer_training_data,
)
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.tokenizers.spatial_waveform_dictionary import (
    SpatialWaveformDictionary,
    load_model,
)

RANGES = ((0, 1), (1, 4), (4, 8), (8, 13), (13, 30), (30, 45), (45, 101))


def aggregate(values, examples, rows):
    return base_training_aggregate({"nmse": values}, examples, rows)["metrics"]["nmse"][
        "by_dataset"
    ]


def projected_continuous(prediction):
    # Match the frozen decoder's rectangular DFT support for a 30-second record.
    frequency = np.fft.rfftfreq(6000, 1 / 200)
    keep = (frequency >= 1) & (frequency < 45)
    result = {}
    for branch in ("S", "O"):
        values = prediction[branch]
        flat = values.reshape(len(values), 6000)
        result[branch] = np.fft.irfft(
            np.fft.rfft(flat, axis=1) * keep, n=6000, axis=1
        ).reshape(values.shape)
    result["sum"] = result["S"] + result["O"]
    return result


def matched_source_worker(dataset):
    from evaluate_joint_tokenizer import input_scales, write
    from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics
    import time

    rows, scales, examples, model, out = MATCHED
    positions = np.array(
        [i for i, (row, _, _) in enumerate(examples) if rows[row]["dataset"] == dataset]
    )
    cache = OrderedDict()
    result = defaultdict(list)
    started = time.monotonic()
    for step in range(0, len(positions), 48):
        selected = examples[positions[step : step + 48]]
        batch = base_tokenizer_batch(
            rows, selected, scales, array_cache=cache, cache_limit=16
        )
        continuous = {
            k: v[:, 0] for k, v in model.predict_continuous(batch["x"][:, None]).items()
        }
        factor = input_scales(rows, selected, scales)
        for name, inputs in [
            ("continuous", continuous),
            ("projected", projected_continuous(continuous)),
        ]:
            audited = joint_reconstruction_metrics(inputs, batch, 1, factor)
            for field in (
                "nmse",
                "nrmse",
                "rms_gain",
                "correlation",
                "rmse_native",
                "bias_native",
                "local_rms_correlation",
                "band_4_8_abs_db",
                "band_8_13_abs_db",
                "band_13_30_abs_db",
                "band_30_45_abs_db",
            ):
                result[name + "__" + field].append(audited["metrics"][field])
            if name == "continuous":
                result["valid_points"].append(audited["support"]["valid_points"])
            if np.any(audited["support"]["nonfinite_prediction_points"]):
                raise ValueError("source prediction nonfinite on retained support")
        if step == 0 or step % 960 == 0:
            write(
                out / (dataset + "_progress.json"),
                dict(
                    scored=step + len(selected),
                    planned=len(positions),
                    seconds=time.monotonic() - started,
                ),
            )
    finished = out / (dataset + ".npz")
    np.savez_compressed(
        finished,
        ordinals=positions,
        **{k: np.concatenate(v) for k, v in result.items()},
    )
    return dataset, time.monotonic() - started


def temporary_wave_bank(source, dataset, waves, available):
    """Pass full real waveforms through the existing spatial synthesis without quantization."""
    tables, address = {}, {}
    for branch in ("S", "O"):
        roles = source.roles(dataset, branch)
        local = np.zeros((1, len(roles), 15), np.int32)
        for band in range(5):
            sites = np.flatnonzero(roles == band)
            atoms = waves[branch][0, sites].reshape(-1, 400)
            tables[branch, band] = np.concatenate((np.zeros((1, 400)), atoms))
            local[0, sites] = np.arange(1, len(atoms) + 1).reshape(-1, 15)
        local[:, ~available] = 0
        address[branch] = local
    temporary = SpatialWaveformDictionary(
        source.metadata,
        source.transfer,
        source.centers,
        tables,
        source.weights,
        source.readout,
        source.feature_weights,
    )
    return temporary, address


def spatial_stage_worker(dataset):
    """Replace the integer atoms with the continuous carrier at the same native sites."""
    from evaluate_joint_tokenizer import input_scales, write
    from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics
    from src.data.dual_component_tokenizer import base_tokenizer_joint_eeg
    from copy import copy
    import time

    rows, scales, examples, source, out = MATCHED
    local, hard_root, source_signature = ROLE_CONTROL
    ordinals = defaultdict(list)
    for j, (idx, start, pos) in enumerate(examples):
        if rows[idx]["dataset"] == dataset:
            ordinals[int(idx), int(start)].append(j)
    metric_fields = (
        "nmse",
        "nrmse",
        "rms_gain",
        "correlation",
        "rmse_native",
        "bias_native",
        "local_rms_correlation",
        "band_4_8_abs_db",
        "band_8_13_abs_db",
        "band_13_30_abs_db",
        "band_30_45_abs_db",
    )
    arrays = defaultdict(list)
    cache = OrderedDict()
    used = set()
    names = source.metadata["positions"][dataset]
    by_name = {name.upper(): i for i, name in enumerate(names)}
    started = time.monotonic()
    for path in sorted((hard_root / dataset).glob("forward_*.npz")):
        with np.load(path, allow_pickle=False) as block:
            if json.loads(str(block["signature"])) != source_signature:
                raise ValueError("mismatched frozen integer source")
            values = {key: block["ids_" + key] for key in ("S", "O")}
            for ix, (idx, start) in enumerate(block["crop_row_start"]):
                key = (int(idx), int(start))
                if key not in ordinals or key in used:
                    raise ValueError(
                        "native integer block outside fixed development scope"
                    )
                used.add(key)
                selected = examples[ordinals[key]]
                w, available, _ = base_tokenizer_joint_eeg(
                    rows,
                    dict(row=idx, start_s=start, positions=[]),
                    scales,
                    names,
                    array_cache=cache,
                    allowed_splits=("validation",),
                )
                codes = {b: v[ix : ix + 1] for b, v in values.items()}
                for code in codes.values():
                    np.testing.assert_array_equal(
                        np.any(code[0] != 0, axis=1), available
                    )
                estimates = source.predict_continuous(
                    w["x"].T.reshape(1, len(names), 15, 400)
                )
                temp, addresses = temporary_wave_bank(
                    source, dataset, estimates, available
                )
                continuous_gain = copy(local)
                continuous_gain.source = temp
                stages = {
                    "spatial_continuous": temp.decode_ids(addresses, dataset),
                    "spatial_gain_continuous": continuous_gain.decode_ids(
                        addresses, dataset
                    ),
                    "spatial_hard": source.decode_ids(codes, dataset),
                }
                batch = base_tokenizer_batch(
                    rows, selected, scales, array_cache=cache, cache_limit=16
                )
                positions = [
                    by_name[rows[idx]["positions"][p].upper()] for _, _, p in selected
                ]
                factors = input_scales(rows, selected, scales)
                for stage, estimate in stages.items():
                    audited = joint_reconstruction_metrics(
                        {k: v[0, positions] for k, v in estimate.items()},
                        batch,
                        1,
                        factors,
                    )
                    for field in metric_fields:
                        arrays[stage + "__" + field].append(audited["metrics"][field])
                    if stage == "spatial_continuous":
                        arrays["valid_points"].append(
                            audited["support"]["valid_points"]
                        )
                    if np.any(audited["support"]["nonfinite_prediction_points"]):
                        raise ValueError("nonfinite stage control")
                arrays["ordinals"].append(ordinals[key])
        write(
            out / (dataset + "_progress.json"),
            dict(
                crops=len(used),
                planned=len(ordinals),
                seconds=time.monotonic() - started,
            ),
        )
    if used != set(ordinals):
        raise ValueError("incomplete development coverage")
    np.savez_compressed(
        out / (dataset + ".npz"), **{k: np.concatenate(v) for k, v in arrays.items()}
    )
    return dataset, time.monotonic() - started


def band_limited_bounds(targets, mask, variances):
    """Bracket the least MSE of a 30-s band-limited signal, on the actual scoring mask."""
    mask = np.asarray(mask, bool)
    n = mask.shape[-1]
    if n != 6000 or targets.shape != (len(mask), 3, n):
        raise ValueError("expected 30-second voltage and X/S/O targets")
    if not mask.any(1).all() or np.any(variances <= 0):
        raise ValueError("nonempty mask and positive training variances required")
    safe = np.where(mask[:, None], targets, 0.0)
    if not np.isfinite(safe).all():
        raise ValueError("nonfinite target on scoring support")
    frequency = np.fft.rfftfreq(n, 1 / 200)
    keep = (frequency >= 1) & (frequency < 45)
    projected = np.fft.irfft(np.fft.rfft(safe, axis=-1) * keep, n=n, axis=-1)
    err = np.where(mask[:, None], safe - projected, 0.0)
    upper = np.square(err).sum(2) / mask.sum(1)[:, None] / variances
    lower = np.where(mask.all(1)[:, None], upper, 0.0)
    return lower, upper


def band_floor_worker(dataset):
    import time

    rows, scales, examples, _, out = MATCHED
    which = np.array(
        [j for j, (i, _, _) in enumerate(examples) if rows[i]["dataset"] == dataset]
    )
    groups = defaultdict(list)
    cache = OrderedDict()
    started = time.monotonic()
    for start in range(0, len(which), 48):
        selected = examples[which[start : start + 48]]
        batch = base_tokenizer_batch(
            rows, selected, scales, array_cache=cache, cache_limit=16
        )
        mask = batch["mask"].reshape(len(selected), 6000)
        targets = np.stack(
            [batch[k].reshape(len(selected), 6000) for k in ("x", "s", "o")], axis=1
        )
        variances = np.column_stack([batch["v" + k][:, 0, 0] for k in ("x", "s", "o")])
        lower, upper = band_limited_bounds(targets, mask, variances)
        groups["floor_lower__nmse"].append(lower[:, :, None])
        groups["floor_upper__nmse"].append(upper[:, :, None])
        groups["valid_points"].append(np.repeat(mask.sum(1)[:, None, None], 3, axis=1))
    np.savez_compressed(
        out / (dataset + ".npz"),
        ordinals=which,
        **{k: np.concatenate(v) for k, v in groups.items()},
    )
    return dataset, time.monotonic() - started


def matched_source(args):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    import time
    from types import SimpleNamespace
    from evaluate_joint_tokenizer import load_source, write

    global MATCHED, ROLE_CONTROL
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "matched_summary.json").exists():
        raise FileExistsError("completed matched audit requires new output")
    rows, scales, examples, manifest = load_source(
        SimpleNamespace(panel=args.panel, modality="eeg", role="development")
    )
    panel = json.loads(args.panel.with_name("panel.json").read_text())
    if any(
        rows[i]["subject"] == panel["confirmation_subjects"][rows[i]["dataset"]]
        for i, _, _ in examples
    ):
        raise ValueError("reserved subject in development panel before waveform access")
    model = load_model(args.run / "model.npz")
    if (
        model.metadata.get("output_projection_hz") != [1, 45]
        or model.metadata["input_identity"] != manifest["cfg_identity"]
        or model.metadata["scales_identity"] != manifest["scales_identity"]
    ):
        raise ValueError("matching frozen G0 source and decoder support required")
    hard_root = args.matched_hard or args.matched_roles or args.band_limited_floor
    ref = json.loads((hard_root / "summary.json").read_text())
    if (
        ref["role"] != "development"
        or ref["modality"] != "eeg"
        or ref["integer_replay"]["status"] != "passed"
    ):
        raise ValueError("replayed EEG development reference required")
    import hashlib

    with Path(ref["checkpoint"]).open("rb") as stream:
        if (
            ref["checkpoint_sha256"]
            != hashlib.file_digest(stream, "sha256").hexdigest()
        ):
            raise ValueError("frozen hard checkpoint changed")
    with np.load(ref["checkpoint"], allow_pickle=False) as f:
        frozen = json.loads(str(f["metadata"]))
    with (args.run / "model.npz").open("rb") as stream:
        if (
            frozen["source_decoder_sha256"]
            != hashlib.file_digest(stream, "sha256").hexdigest()
        ):
            raise ValueError(
                "hard and continuous arms must share the frozen spatial source"
            )
    with np.load(hard_root / "metrics.npz") as hard:
        np.testing.assert_array_equal(hard["examples"], examples)
    MATCHED = rows, scales, examples, model, args.output
    datasets = sorted({rows[i]["dataset"] for i, _, _ in examples})
    stage_worker = matched_source_worker
    stages = {
        "continuous": "frozen per-position Wiener estimate from X, before the integer encoder",
        "projected": "same estimate with exact output 1<=f<45Hz Fourier support on full 30s",
    }
    if args.matched_roles:
        from src.tokenizers.spatial_local_gain import load_decoder

        signature = json.loads((hard_root / "independent_review_v1.json").read_text())[
            "frozen_signature"
        ]
        ROLE_CONTROL = load_decoder(ref["checkpoint"]), hard_root, signature
        stage_worker = spatial_stage_worker
        stages = {
            "spatial_continuous": "exact frozen spatial synthesis on continuous native Wiener carriers",
            "spatial_gain_continuous": "same continuous native wave bank and same frozen gain rule as integer path",
            "spatial_hard": "saved real AS codes, same spatial synthesis before final gain",
        }
    if args.band_limited_floor:
        stage_worker = band_floor_worker
        stages = {
            "floor_lower": "Full mask: exact orthogonal residual. Partial mask: zero lower bound.",
            "floor_upper": "Score the feasible rectangular projection of zero-filled targets on the original mask. Equal to lower bound on full masks.",
        }
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=len(datasets), mp_context=multiprocessing.get_context("fork")
    ) as pool:
        seconds = dict(pool.map(stage_worker, datasets))
    arrays = defaultdict(list)
    for dataset in datasets:
        with np.load(args.output / (dataset + ".npz")) as saved:
            for field in saved.files:
                arrays[field].append(saved[field])
    assembled = {key: np.concatenate(values) for key, values in arrays.items()}
    ordinals = assembled.pop("ordinals")
    order = np.argsort(ordinals)
    np.testing.assert_array_equal(ordinals[order], np.arange(len(examples)))
    full = {k: v[order] for k, v in assembled.items()}
    with np.load(hard_root / "metrics.npz") as hard:
        np.testing.assert_array_equal(
            full["valid_points"], hard["support_valid_points"]
        )
    scores = {}
    for arm in stages:
        fields = {
            key.split("__", 1)[1]: value
            for key, value in full.items()
            if key.startswith(arm + "__")
        }
        scores[arm] = base_training_aggregate(fields, examples, rows)
    scores["integer"] = {
        "metrics": {
            key: ref["metrics"][key] for key in scores[next(iter(stages))]["metrics"]
        },
        "selection_score": ref["selection_score"],
    }
    description = "Projected vs integer includes band-role assignment, spatial mixing, quantization and postprocessing."
    if args.matched_roles:
        description = "Fixed spatial operators, varying the carrier source and application of the frozen gain."
    if args.band_limited_floor:
        description = "Target-assisted analytic bounds on attainable masked MSE for a 30-second real [1,45) Hz Fourier reconstruction. These controls are diagnostics only."
    write(
        args.output / "matched_summary.json",
        dict(
            status="completed",
            inputs=len(examples),
            subjects=ref["subjects"],
            windows=ref["windows"],
            records=ref["records"],
            panel_sha256=panel["panel_sha256"],
            seconds_by_dataset=seconds,
            elapsed_seconds=time.monotonic() - started,
            scoring="Original masks, G0 target-specific variances and per-position train scales. Equal hierarchy through datasets.",
            arms=dict(
                stages,
                integer="same source, role allocation, saved discrete search and local-energy synthesis",
            ),
            scope=description,
            summaries=scores,
        ),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--integers", type=Path)
    mode.add_argument(
        "--matched-hard",
        type=Path,
        help="full development hard metric folder for frozen continuous and matched projection comparison",
    )
    mode.add_argument(
        "--matched-roles",
        type=Path,
        help="full development hard metric folder for continuous vs hard carrier controls in the exact spatial decoder",
    )
    mode.add_argument(
        "--band-limited-floor",
        type=Path,
        help="bracket target-assisted projection floor on all actual EEG development scoring masks",
    )
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.matched_hard is not None
        or args.matched_roles is not None
        or args.band_limited_floor is not None
    ):
        return matched_source(args)
    args.output.mkdir(exist_ok=True, parents=True)
    if (args.output / "summary.json").exists():
        raise FileExistsError("retained completed distortion audit")
    cfg = yaml.safe_load(
        (
            ROOT
            / "experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml"
        ).read_text()
    )
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    rows = manifest["rows"]
    model = load_model(args.run / "model.npz")
    if model.metadata["input_identity"] != manifest["cfg_identity"]:
        raise ValueError("matching G0 model required")
    with np.load(args.panel) as stored:
        examples = stored["eeg_screen"]
    groups = defaultdict(list)
    for ordinal, (index, start, position) in enumerate(examples):
        if rows[index]["split"] != "validation":
            raise ValueError("fixed validation screen boundary before arrays")
        groups[int(index), int(start)].append((ordinal, int(position)))
    metadata = json.loads(args.integers.with_name("integer_metadata.json").read_text())
    cache = OrderedDict()
    errors = np.zeros((len(examples), 3, len(RANGES)))
    target_energies = np.zeros_like(errors)
    predicted_energies = np.zeros_like(errors)
    frequency = np.fft.rfftfreq(6000, 1 / 200)
    fourier_weight = np.full(len(frequency), 2.0)
    fourier_weight[[0, -1]] = 1
    with threadpool_limits(8), np.load(args.integers) as stored:
        for crop_number, ((index, start), selected) in enumerate(groups.items()):
            row = rows[index]
            dataset = row["dataset"]
            names = model.metadata["positions"][dataset]
            if (
                metadata[crop_number]["row"] != index
                or metadata[crop_number]["start_s"] != start
            ):
                raise ValueError("integer physical crop mismatch")
            ids = {branch: stored[f"{branch}_{crop_number}"] for branch in ("S", "O")}
            prediction = model.decode_ids(ids, dataset)
            batch = base_tokenizer_batch(
                rows,
                [(index, start, position) for _, position in selected],
                scales,
                array_cache=cache,
                cache_limit=8,
            )
            positions = [
                next(
                    i
                    for i, name in enumerate(names)
                    if name.upper() == row["positions"][position].upper()
                )
                for _, position in selected
            ]
            mask = batch["mask"].reshape(len(selected), -1)
            count = mask.sum(axis=1)
            for component, (branch, field) in enumerate(
                (("sum", "x"), ("S", "s"), ("O", "o"))
            ):
                truth = batch[field].reshape(len(selected), -1)
                value = prediction[branch][0, positions].reshape(truth.shape)
                variance = batch["v" + field].reshape(truth.shape)
                target = np.where(mask, truth / np.sqrt(variance), 0.0)
                reconstructed = np.where(mask, value / np.sqrt(variance), 0.0)
                spectra = [
                    np.fft.rfft(array, norm="ortho", axis=1)
                    for array in (reconstructed - target, target, reconstructed)
                ]
                energy = [
                    np.abs(spectrum) ** 2 * fourier_weight for spectrum in spectra
                ]
                for band, (lower, upper) in enumerate(RANGES):
                    take = (frequency >= lower) & (frequency < upper)
                    for destination, values in zip(
                        (errors, target_energies, predicted_energies), energy
                    ):
                        destination[
                            [ordinal for ordinal, _ in selected], component, band
                        ] = values[:, take].sum(axis=1) / np.maximum(count, 1)
            (args.output / "progress.json").write_text(
                json.dumps(dict(completed=crop_number + 1, planned=len(groups))) + "\n"
            )
    np.savez_compressed(
        args.output / "frequency_contributions.npz",
        examples=examples,
        nmse_contribution=errors,
        target_G0_variance_normalized_energy=target_energies,
        prediction_G0_variance_normalized_energy=predicted_energies,
    )
    summary = dict(
        status="completed",
        components=["X", "S", "O"],
        frequency_ranges_hz=RANGES,
        error_contribution_by_dataset=aggregate(errors, examples, rows),
        target_normalized_energy_by_dataset=aggregate(target_energies, examples, rows),
        predicted_normalized_energy_by_dataset=aggregate(
            predicted_energies, examples, rows
        ),
        denominator="G0 frozen component training variance; Fourier Parseval error contributions sum to full waveform NMSE",
        masking="target and prediction set to zero on the same invalid points before full-window FFT; exact additive error decomposition",
        fixed_examples=len(examples),
        source_ids=str(args.integers),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
