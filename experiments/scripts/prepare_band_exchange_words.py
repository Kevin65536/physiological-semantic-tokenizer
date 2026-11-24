"""Prepare fixed and exchanged continuous band samples from the frozen X-only source."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
from pathlib import Path
import hashlib
import json
import multiprocessing
import os
import sys
import time

import numpy as np
import yaml
from numpy.lib.format import open_memmap
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.dual_component_tokenizer import (
    base_tokenizer_training_data,
    base_tokenizer_joint_eeg,
)
from src.tokenizers.adaptive_band_allocation import (
    band_waves,
    choose_exchanges,
    exchange_costs,
    extract_coordinates,
    synthesize,
)
from src.tokenizers.eeg_transient_gain import load_model as load_source
from src.analysis.tokenizer_pattern_audit import attributes

STATE = {}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def check_budget():
    unit = os.environ.get("BAND_WORD_PREP_UNIT")
    if unit:
        import subprocess

        used = (
            int(
                subprocess.check_output(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        unit,
                        "-p",
                        "CPUUsageNSec",
                        "--value",
                    ],
                    text=True,
                )
            )
            / 1e9
        )
        if used > 1770:
            raise RuntimeError("bounded preparation CPU budget reached")


def source_features(wave):
    shape = wave.shape
    flat = wave.reshape(-1, 400)
    values, fields = attributes(flat, "eeg")
    theta = values[:, fields["power_4_8"][0]]
    alpha = values[:, fields["power_8_13"][0]]
    rms = np.sqrt(np.mean(np.square(flat), axis=1))
    return (
        np.log1p(np.stack([theta, alpha, rms], axis=1))
        .reshape(*shape[:-1], 3)
        .astype(np.float32)
    )


def dataset_worker(dataset):
    run, out, rows, scales, population = (
        STATE[k] for k in ("run", "out", "rows", "scales", "population")
    )
    cases = [c for c in population["cases"] if c["dataset"] == dataset]
    names = population["native_positions"][dataset]
    C = len(names)
    N = sum(c["count"] for c in cases)
    case_rows = np.concatenate(
        [
            np.array(
                [
                    (c["crop"]["row"], c["crop"]["start_s"], p)
                    for p in c["crop"]["positions"]
                ],
                np.int64,
            )
            for c in cases
        ]
    )
    native = np.concatenate([np.asarray(c["native"], np.int32) for c in cases])
    folds = np.concatenate([np.full(c["count"], c["fold"], np.int8) for c in cases])
    if np.any(native < 0) or np.any(native >= C):
        raise ValueError("invalid native electrode identity")
    for fold in range(4):
        check_budget()
        directory = out / dataset / f"fold_{fold}"
        directory.mkdir(parents=True, exist_ok=True)
        ready = directory / "ready.json"
        if ready.exists():
            saved = json.loads(ready.read_text())
            for name, record in saved["files"].items():
                if digest(directory / name) != record["sha256"]:
                    raise ValueError("retained prepared file identity changed")
            continue
        source_path = (
            run
            / "component_source_transfer/transient_gain_v1"
            / f"{dataset}_fold{fold}_transient.npz"
        )
        covariance_path = (
            run
            / "component_source_transfer/adaptive_band_v1"
            / dataset
            / f"fold_{fold}_model.npz"
        )
        reference_path = (
            run
            / "component_source_transfer/adaptive_band_exchange_v1"
            / dataset
            / f"fold_{fold}_roles.npz"
        )
        metric_path = (
            run
            / "component_source_transfer/adaptive_band_exchange_v1"
            / dataset
            / f"fold_{fold}_metrics.npz"
        )
        source = load_source(source_path)
        fit_people = sorted({c["subject"] for c in cases if c["fold"] != fold})
        held_people = sorted({c["subject"] for c in cases if c["fold"] == fold})
        if (
            set(source.metadata["fit_subjects"]) != set(fit_people)
            or source.metadata["held_fold"] != fold
            or source.metadata["positions"][dataset] != names
        ):
            raise ValueError(
                "same-fold fitted source and native electrode identity required"
            )
        if (
            source.metadata["input_identity"] != STATE["input_identity"]
            or source.metadata["scales_identity"] != STATE["scale_identity"]
        ):
            raise ValueError("G0 source identities changed")
        with np.load(covariance_path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["names"], names)
            if set(z["fit_people"]) != set(fit_people):
                raise ValueError("covariance fitted on wrong subject population")
            covariance = {b: z[b].copy() for b in "SO"}
        with np.load(reference_path, allow_pickle=False) as z:
            expected_roles = {b: z["exchange_" + b].copy() for b in "SO"}
        with np.load(metric_path, allow_pickle=False) as z:
            expected_examples = z["examples"].copy()
        actual_held = np.concatenate(
            [
                np.array(
                    [
                        (c["crop"]["row"], c["crop"]["start_s"], p)
                        for p in c["crop"]["positions"]
                    ],
                    np.int64,
                )
                for c in cases
                if c["fold"] == fold
            ]
        )
        np.testing.assert_array_equal(actual_held, expected_examples)
        if any(
            expected_roles[b].shape != (sum(c["fold"] == fold for c in cases), C, 15)
            for b in "SO"
        ):
            raise ValueError("legal held role tensor changed")
        fixed = {
            b: open_memmap(
                directory / (b + "_fixed.pending.npy"),
                mode="w+",
                dtype=np.float32,
                shape=(N, 15, 400),
            )
            for b in "SO"
        }
        role = {b: np.empty((N, 15), np.int8) for b in "SO"}
        features = {b: np.empty((N, 15, 3), np.float32) for b in "SO"}
        changed_rows = {b: [] for b in "SO"}
        changed_values = {b: [] for b in "SO"}
        all_seen = np.empty(N, bool)
        operators = {}
        offset = 0
        held_index = 0
        replay = []
        input_hashes = []
        for case_index, c in enumerate(cases):
            check_budget()
            crop = c["crop"]
            row = rows[crop["row"]]
            if (row["dataset"], row["subject"], row["split"], row["modality"]) != (
                dataset,
                c["subject"],
                "train",
                "eeg",
            ):
                raise ValueError("prepared train observation boundary before arrays")
            selected = np.asarray(c["native"])
            if len(selected) != c["count"] or c["count"] != len(crop["positions"]):
                raise ValueError("declared case cardinality changed")
            for original, native_position in zip(
                crop["positions"], selected, strict=True
            ):
                if row["positions"][original].upper() != names[native_position].upper():
                    raise ValueError("native electrode name permutation changed")
            fields, seen, _ = base_tokenizer_joint_eeg(
                rows, dict(crop, positions=[]), scales, names, allowed_splits=("train",)
            )
            # The legal continuous arm used this exact historical float32 input conversion.
            x = fields["x"].T.reshape(1, C, 15, 400).astype(np.float32)
            dense = source.predict_continuous(x, dataset, seen[None])
            n = c["count"]
            block = slice(offset, offset + n)
            all_seen[block] = seen[selected]
            mask_key = seen.tobytes()
            input_hashes.append(
                dict(
                    case=case_index,
                    row=crop["row"],
                    start_s=crop["start_s"],
                    X_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                    observed_sha256=hashlib.sha256(seen.tobytes()).hexdigest(),
                )
            )
            for b in "SO":
                waves = band_waves(np.fft.rfft(dense[b][0].reshape(C, 6000)))
                if b not in operators or operators[b][0] != mask_key:
                    scale_table = {
                        name.upper(): v
                        for name, v in scales["positions"]["eeg"][dataset].items()
                    }
                    weights = np.array(
                        [
                            1
                            / float(
                                np.ravel(
                                    scale_table[name.upper()]["target_variance"][
                                        b.lower()
                                    ]
                                )[0]
                            )
                            for name in names
                        ]
                    )
                    weights[~seen] = 0
                    operators[b] = (
                        mask_key,
                        exchange_costs(covariance[b], seen, weights),
                    )
                choices, _ = choose_exchanges(waves, seen, operators[b][1])
                if c["fold"] == fold:
                    np.testing.assert_array_equal(
                        choices, expected_roles[b][held_index]
                    )
                native_role = np.arange(C) % 5
                fixed_full = waves[native_role, np.arange(C)].copy()
                fixed_full[~seen] = 0
                cached = fixed_full[selected].astype(np.float32)
                fixed[b][block] = cached
                role[b][block] = choices[selected]
                feat = source_features(dense[b][0])
                feat[~seen] = 0
                features[b][block] = feat[selected]
                changed = (choices[selected] != native_role[selected, None]) & seen[
                    selected, None
                ]
                flat = np.flatnonzero(changed) + offset * 15
                if len(flat):
                    selected_coords = extract_coordinates(waves, choices)[selected]
                    changed_rows[b].append(flat.astype(np.int64))
                    changed_values[b].append(
                        selected_coords[changed].astype(np.float32)
                    )
                if c["fold"] == fold and held_index == 0:
                    # Restore one held full array from exactly the float32 sparse cache schema.
                    actual = extract_coordinates(waves, choices)
                    reconstructed = fixed_full.astype(np.float32)
                    change_full = (choices != native_role[:, None]) & seen[:, None]
                    reconstructed[change_full] = actual[change_full].astype(np.float32)
                    error = float(np.max(abs(reconstructed - actual)))
                    prediction = synthesize(actual, choices, covariance[b])
                    reloaded = synthesize(reconstructed, choices, covariance[b])
                    gap = float(np.max(abs(prediction - reloaded)))
                    rms = float(np.sqrt(np.mean(np.square(actual))))
                    tolerance = 2e-6 + 1e-5 * np.abs(actual)
                    if np.any(np.abs(reconstructed - actual) > tolerance):
                        raise ValueError(
                            "prepared selected coordinates exceed original replay tolerance"
                        )
                    np.testing.assert_allclose(
                        reloaded, prediction, atol=2e-6, rtol=1e-5
                    )
                    replay.append(
                        dict(
                            branch=b,
                            coordinate_max_abs_error=error,
                            source_RMS=rms,
                            full_synthesis_max_abs_error=gap,
                            original_atol=2e-6,
                            original_rtol=1e-5,
                        )
                    )
            offset += n
            if c["fold"] == fold:
                held_index += 1
            if case_index % 16 == 0:
                write(
                    directory / "progress.json",
                    dict(
                        cases=case_index + 1,
                        planned=len(cases),
                        rows=offset,
                        CPU_seconds=time.process_time(),
                    ),
                )
        assert offset == N
        np.save(directory / "examples.npy", case_rows)
        np.save(directory / "native.npy", native)
        np.save(directory / "fold.npy", folds)
        np.save(directory / "seen.npy", all_seen)
        for b in "SO":
            fixed[b].flush()
            del fixed[b]
            (directory / (b + "_fixed.pending.npy")).replace(
                directory / (b + "_fixed.npy")
            )
            indices = (
                np.concatenate(changed_rows[b])
                if changed_rows[b]
                else np.empty(0, np.int64)
            )
            values = (
                np.concatenate(changed_values[b])
                if changed_values[b]
                else np.empty((0, 400), np.float32)
            )
            if len(indices) > 1 and np.any(np.diff(indices) <= 0):
                raise ValueError(
                    "sparse changed patch order must be strictly increasing"
                )
            np.save(directory / (b + "_changed_rows.npy"), indices)
            np.save(directory / (b + "_changed.npy"), values)
            np.save(directory / (b + "_role.npy"), role[b])
            np.save(directory / (b + "_features.npy"), features[b])
        metadata = dict(
            schema="band_exchange_word_preparation_v1",
            dataset=dataset,
            outer_source_fold=fold,
            native_positions=names,
            rows=N,
            cases=len(cases),
            fit_subjects=fit_people,
            held_subjects=held_people,
            row_order="population.json dataset cases then crop.positions order; native.npy is actual frozen layout index",
            fixed_role="native%5",
            role_missing=-1,
            sparse_changed="flat row*15+patch strictly increasing; overlay changed.npy on fixed.npy after reshape(-1,400)",
            features="log1p Hann theta power, log1p Hann alpha power, log1p whole-two-second RMS of unencoded X-only source_f in G0 train coordinates",
            fixed_and_changed_dtype="float32",
            source_input_conversion="float64 G0 joint X then same float32 as original continuous arm",
            covariance_sha256=digest(covariance_path),
            source_sha256=digest(source_path),
            legal_held_roles_sha256=digest(reference_path),
            held_role_arrays_exact=True,
            first_held_replay=replay,
            case_inputs=input_hashes,
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
        )
        write(directory / "metadata.json", metadata)
        records = {
            p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
            for p in sorted(directory.glob("*.npy"))
        }
        records["metadata.json"] = dict(
            bytes=(directory / "metadata.json").stat().st_size,
            sha256=digest(directory / "metadata.json"),
        )
        write(
            ready,
            dict(
                status="ready",
                schema=metadata["schema"],
                dataset=dataset,
                fold=fold,
                rows=N,
                files=records,
                CPU_seconds=time.process_time(),
            ),
        )
    return dict(
        dataset=dataset, folds=4, rows_per_fold=N, CPU_seconds=time.process_time()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    controller = (out / "controller.lock").open("a+")
    fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "completion.json").exists():
        raise FileExistsError("completed preparation retained")
    base = yaml.safe_load(
        (
            ROOT
            / "experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml"
        ).read_text()
    )
    manifest, scales, _, _ = base_tokenizer_training_data(base, ROOT)
    population_path = (
        run
        / "component_source_transfer/hard_as_v1/local_information_v1/population.json"
    )
    population = json.loads(population_path.read_text())
    if (
        population["cfg_identity"] != manifest["cfg_identity"]
        or population["scales_identity"] != manifest["scales_identity"]
        or population["scoring_positions"] != 17040
        or len(population["cases"]) != 400
    ):
        raise ValueError(
            "same400/60/17040 and original G0 identities required before arrays"
        )
    if any(
        manifest["rows"][c["crop"]["row"]]["split"] != "train"
        for c in population["cases"]
    ):
        raise ValueError("all cases must be train")
    STATE.update(
        run=run,
        out=out,
        rows=manifest["rows"],
        scales=scales,
        population=population,
        input_identity=manifest["cfg_identity"],
        scale_identity=manifest["scales_identity"],
    )
    with threadpool_limits(1), ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        records = list(pool.map(dataset_worker, sorted(population["native_positions"])))
    write(
        out / "completion.json",
        dict(
            status="ready",
            schema="band_exchange_word_preparation_v1",
            datasets=records,
            population_sha256=digest(population_path),
            all_scoring_positions=17040,
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
        ),
    )


if __name__ == "__main__":
    main()
