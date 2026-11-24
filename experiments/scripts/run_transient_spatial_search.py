"""Matched actual single-ID searches using frozen cross-fitted EEG source estimates."""

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import copy
import hashlib, json, multiprocessing, sys
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate_joint_tokenizer import input_scales, write
from src.data.dual_component_tokenizer import (
    base_tokenizer_joint_eeg,
    base_tokenizer_training_data,
    base_tokenizer_batch,
)
from src.analysis.dual_component_tokenizer import (
    joint_reconstruction_metrics,
    base_training_aggregate,
)
from src.tokenizers.spatial_analysis_synthesis import load_encoder
from src.tokenizers.spatial_waveform_dictionary import load_model as load_decoder
from src.tokenizers.eeg_transient_gain import load_model as load_source
from src.tokenizers.role_mixed_spatial import FixedRoleMixedDecoder, load_role_mixed_decoder

STATE = {}
ARMS = ("legacy", "transient")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def role_wave_budget():
    cfg = STATE["cfg"]
    unit = os.environ.get("TOKENIZER_ROLE_WAVE_UNIT")
    used = (
        time.process_time()
        if not unit
        else int(
            subprocess.check_output(
                ["systemctl", "--user", "show", unit, "-p", "CPUUsageNSec", "--value"],
                text=True,
            )
        )
        / 1e9
    )
    total = used + cfg["resources"]["prior_CPU_seconds"]
    if total >= cfg["resources"]["total_CPU_budget_seconds"] - 60:
        raise RuntimeError(
            "total role-mixture CPU budget reached; completed blocks retained"
        )
    return total


def role_mixed_worker(dataset):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        cfg, run, rows, scales, out, hard = (
            STATE[k] for k in ["cfg", "run", "rows", "scales", "out", "hard"]
        )
        root = ROOT / cfg["source"]["hard_run"]
        high_path, low_path = [
            run / hard["families"][f]["decoder"] for f in ["high", "semantic"]
        ]
        high, low = load_decoder(high_path), load_decoder(low_path)
        names = high.metadata["positions"][dataset]
        if low.metadata["positions"][dataset] != names:
            raise ValueError("pure dictionaries have different native support")
        native = {n.upper(): j for j, n in enumerate(names)}
        source_paths = {
            fold: run
            / hard["data"]["source_model_run"]
            / f"{dataset}_fold{fold}_transient.npz"
            for fold in range(4)
        }
        sources = {fold: load_source(file) for fold, file in source_paths.items()}
        pairs = [tuple(v) for v in cfg["selection"]["fixed_waveform_pairs"]] + [
            (31, 31)
        ]
        dest = out / dataset
        dest.mkdir(exist_ok=True)
        model_dir, ids_dir = dest / "models", dest / "integer_ids"
        model_dir.mkdir(exist_ok=True)
        ids_dir.mkdir(exist_ok=True)
        emitters, models, frozen_cache = {}, {}, {}
        for pair in pairs:
            masks = dict(zip("SO", pair))
            tag = f"s{pair[0]}_o{pair[1]}"
            emitters[pair] = FixedRoleMixedDecoder(high, low, masks)
            for fold in range(4):
                for mode in ["raw", "gain", "gain_source"] + (
                    ["raw_source"] if pair == (0, 0) else []
                ):
                    correction = mode.replace("_source", "")
                    estimator = sources[fold] if mode.endswith("_source") else None
                    model = FixedRoleMixedDecoder(
                        high, low, masks, mode=correction, estimator=estimator
                    )
                    file = model_dir / f"{tag}_fold{fold}_{mode}.npz"
                    model.save(
                        file,
                        high_path=high_path,
                        low_path=low_path,
                        source_path=source_paths[fold] if estimator else None,
                    )
                    models[pair, fold, mode] = load_role_mixed_decoder(
                        file, frozen_cache=frozen_cache
                    )
        write(
            dest / "integer_rate.json",
            {f"s{a}_o{b}": emitters[a, b].rate(dataset) for a, b in pairs},
        )
        cache = OrderedDict()
        probes = []
        identity_rows = []
        arms = [
            f"s{a}_o{b}_{mode}"
            for a, b in pairs
            for mode in ["raw", "gain", "gain_source"]
        ]
        arms += ["s0_o0_raw_source"]
        block_paths = []
        for path in sorted((root / "high" / dataset).glob("ids_*.npz")):
            role_wave_budget()
            suffix = path.stem.split("_", 1)[1]
            block_path = dest / f"block_{suffix}.npz"
            block_paths.append(block_path)
            if block_path.exists():
                continue
            hashes, parts, examples = [], defaultdict(list), []
            with (
                np.load(path) as high_ids,
                np.load(root / "semantic" / dataset / path.name) as low_ids,
                np.load(
                    path.with_name(path.name.replace("ids_", "block_"))
                ) as high_scores,
                np.load(
                    root / "semantic" / dataset / path.name.replace("ids_", "block_")
                ) as low_scores,
            ):
                np.testing.assert_array_equal(high_ids["keys"], low_ids["keys"])
                np.testing.assert_array_equal(
                    high_scores["examples"], low_scores["examples"]
                )
                keys = high_ids["keys"]
                for family, archive in [("high", high_ids), ("semantic", low_ids)]:
                    np.testing.assert_array_equal(
                        archive["identity"],
                        [
                            sha(run / hard["families"][family]["encoder"]),
                            sha(run / hard["families"][family]["decoder"]),
                        ],
                    )
                for pair in pairs:
                    tag = f"s{pair[0]}_o{pair[1]}"
                    sent = emitters[pair].emit_ids(
                        {b: high_ids["transient_" + b] for b in "SO"},
                        {b: low_ids["transient_" + b] for b in "SO"},
                        dataset,
                    )
                    np.savez_compressed(
                        ids_dir / f"{tag}_{suffix}.npz", keys=keys, **sent
                    )
                    if pair in [(0, 0), (31, 31)]:
                        original = low_ids if pair == (0, 0) else high_ids
                        for b in "SO":
                            np.testing.assert_array_equal(
                                sent[b], original["transient_" + b]
                            )
                for j, (row_index, start, fold) in enumerate(keys):
                    role_wave_budget()
                    row_index, start, fold = map(int, [row_index, start, fold])
                    row = rows[row_index]
                    if (
                        row["split"] != "train"
                        or row["dataset"] != dataset
                        or row["subject"] in sources[fold].metadata["fit_subjects"]
                    ):
                        raise ValueError(
                            "same held source_f and original train people required before targets"
                        )
                    take = (high_scores["examples"][:, 0] == row_index) & (
                        high_scores["examples"][:, 1] == start
                    )
                    ex = high_scores["examples"][take]
                    sites = [native[row["positions"][p].upper()] for _, _, p in ex]
                    if not probes:
                        fields, observed, _ = base_tokenizer_joint_eeg(
                            rows,
                            dict(row=row_index, start_s=start, positions=[]),
                            scales,
                            names,
                            cache,
                            allowed_splits=("train",),
                        )
                        x = fields["x"].T.reshape(1, len(names), 15, 400)
                        for family, archive in [
                            ("high", high_ids),
                            ("semantic", low_ids),
                        ]:
                            encoder = load_encoder(
                                run / hard["families"][family]["encoder"],
                                component_source=sources[fold],
                            )
                            fresh = encoder.encode(x, dataset, observed[None])
                            for b in "SO":
                                np.testing.assert_array_equal(
                                    fresh[b], archive["transient_" + b][j : j + 1]
                                )
                            probes.append(
                                dict(
                                    dataset=dataset,
                                    family=family,
                                    row=row_index,
                                    start_s=start,
                                    fold=fold,
                                    all_native_IDs_exact=True,
                                    X_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                                )
                            )
                            del encoder
                    batch = base_tokenizer_batch(
                        rows, ex, scales, array_cache=cache, cache_limit=16
                    )
                    factors = input_scales(rows, ex, scales)
                    predictions = {}
                    for pair in pairs:
                        tag = f"s{pair[0]}_o{pair[1]}"
                        with np.load(ids_dir / f"{tag}_{suffix}.npz") as sent:
                            codes = {b: sent[b][j : j + 1] for b in "SO"}
                        raw = models[pair, fold, "raw"].decode_ids(codes, dataset)
                        gain = models[pair, fold, "gain"].decode_ids(codes, dataset)
                        allocated = models[pair, fold, "gain_source"].decoder.allocate(
                            gain, codes, dataset
                        )
                        predictions.update(
                            {
                                tag + "_raw": raw,
                                tag + "_gain": gain,
                                tag + "_gain_source": allocated,
                            }
                        )
                        if pair == (0, 0):
                            predictions[tag + "_raw_source"] = models[
                                pair, fold, "raw_source"
                            ].decoder.allocate(raw, codes, dataset)
                        if j == 0:
                            independent = models[pair, fold, "gain_source"].decode_ids(
                                codes, dataset
                            )
                            for b in independent:
                                np.testing.assert_array_equal(
                                    independent[b], allocated[b]
                                )
                    for arm, prediction in predictions.items():
                        closure = float(
                            np.max(
                                abs(
                                    prediction["S"]
                                    + prediction["O"]
                                    - prediction["sum"]
                                )
                            )
                        )
                        if closure > 1e-10:
                            raise ValueError("integer component sum closure changed")
                        audit = joint_reconstruction_metrics(
                            {b: value[0, sites] for b, value in prediction.items()},
                            batch,
                            1,
                            factors,
                        )
                        if audit["support"]["nonfinite_prediction_points"].any():
                            raise ValueError("nonfinite mixed integer waveform")
                        for metric, value in audit["metrics"].items():
                            parts[arm + "__" + metric].append(value)
                        hashes.append(
                            dict(
                                row=row_index,
                                start_s=start,
                                fold=fold,
                                arm=arm,
                                closure_max_abs=closure,
                                voltage_SHA256={
                                    b: hashlib.sha256(
                                        np.ascontiguousarray(value).tobytes()
                                    ).hexdigest()
                                    for b, value in prediction.items()
                                },
                            )
                        )
                        if arm in ["s31_o31_gain", "s0_o0_raw"]:
                            expected = (
                                high_scores if arm.startswith("s31") else low_scores
                            )
                            for metric in [
                                "nmse",
                                "rms_gain",
                                "correlation",
                                "local_rms_correlation",
                                "band_4_8_abs_db",
                                "band_8_13_abs_db",
                            ]:
                                np.testing.assert_array_equal(
                                    audit["metrics"][metric],
                                    expected["transient__" + metric][take],
                                )
                    examples.extend(ex)
                    identity_rows.append(
                        dict(
                            dataset=dataset,
                            subject=row["subject"],
                            row=row_index,
                            start_s=start,
                            fold=fold,
                            native_positions=names,
                            scored_positions=ex[:, 2].tolist(),
                        )
                    )
            np.savez_compressed(
                block_path,
                examples=np.asarray(examples),
                keys=keys,
                **{k: np.concatenate(v) for k, v in parts.items()},
            )
            write(dest / f"voltage_hashes_{suffix}.json", hashes)
            write(
                dest / "progress.json",
                dict(
                    last_block=suffix,
                    CPU_seconds=time.process_time(),
                    total_CPU_seconds=role_wave_budget(),
                ),
            )
        joined = defaultdict(list)
        for path in block_paths:
            with np.load(path) as f:
                for key in f:
                    joined[key].append(f[key])
        values = {k: np.concatenate(v) for k, v in joined.items() if k != "keys"}
        np.savez_compressed(out / (dataset + "_metrics.npz"), **values)
        write(dest / "source_and_reencoding_probes.json", probes)
        write(dest / "integer_metadata.json", identity_rows)
        write(
            dest / "summary.json",
            dict(
                status="completed",
                position_windows=len(values["examples"]),
                CPU_seconds=time.process_time(),
                model_params_fitted=0,
                arrays_from_saved_final_integers_only=True,
                validation_reads=0,
            ),
        )
        return dataset, time.process_time()


def run_role_mixed_waveform(args):
    import fcntl

    cfg = yaml.safe_load(args.config.read_text())
    if cfg["schema"] != "fixed_role_mixed_waveform_v1" or any(
        cfg["data"][key] != "forbidden"
        for key in ["validation_access", "confirmation_access", "test_access"]
    ):
        raise ValueError("train-only fixed-role waveform falsification required")
    info = ROOT / cfg["source"]["information_run"]
    if (
        sha(info / "information_summary.json")
        != cfg["source"]["information_summary_sha256"]
    ):
        raise ValueError("prior frozen information feasibility changed")
    information = json.loads((info / "information_summary.json").read_text())
    expected = [
        [s, o]
        for s in information["feasible_masks"]["transient"]["S"]
        for o in information["feasible_masks"]["transient"]["O"]
    ]
    if cfg["selection"]["fixed_waveform_pairs"] != expected:
        raise ValueError(
            "only all previously information-admissible fixed pairs may enter waveform analysis"
        )
    hard = yaml.safe_load((ROOT / cfg["source"]["hard_contract"]).read_text())
    run = ROOT / hard["data"]["run"]
    basis = yaml.safe_load((ROOT / hard["data"]["basis_config"]).read_text())["source"]
    manifest, scales, _, _ = base_tokenizer_training_data(
        yaml.safe_load((ROOT / basis["g0_config"]).read_text()), ROOT
    )
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "summary.json").exists():
            raise FileExistsError("retained completed mixed waveform result")
        STATE.update(
            cfg=cfg, run=run, rows=manifest["rows"], scales=scales, out=out, hard=hard
        )
        write(out / "resolved_config.json", cfg)
        with ProcessPoolExecutor(
            max_workers=cfg["resources"]["workers"],
            mp_context=multiprocessing.get_context("fork"),
        ) as pool:
            usage = dict(
                pool.map(role_mixed_worker, sorted(scales["positions"]["eeg"]))
            )
        arrays = defaultdict(list)
        for dataset in sorted(scales["positions"]["eeg"]):
            with np.load(out / (dataset + "_metrics.npz")) as f:
                for key in f:
                    arrays[key].append(f[key])
        values = {k: np.concatenate(v) for k, v in arrays.items()}
        ex = values["examples"]
        with np.load(
            ROOT / cfg["source"]["local_information"] / "training_attributes.npz"
        ) as bank:
            if sorted(map(tuple, ex.tolist())) != sorted(
                map(tuple, bank["examples"].tolist())
            ):
                raise ValueError("complete 17040 training population changed")
        arms = sorted({key.rsplit("__", 1)[0] for key in values if "__" in key})
        summaries = {
            arm: base_training_aggregate(
                {
                    key[len(arm) + 2 :]: value
                    for key, value in values.items()
                    if key.startswith(arm + "__")
                },
                ex,
                manifest["rows"],
            )
            for arm in arms
        }
        reference = summaries["s31_o31_gain_source"]["metrics"]
        for arm, summary in summaries.items():
            metric = summary["metrics"]
            nmse = np.asarray(metric["nmse"]["macro"])
            ref = np.asarray(reference["nmse"]["macro"])
            gain = np.asarray(metric["rms_gain"]["macro"])
            gates = dict(
                X_NMSE=bool(nmse[0, 0] <= 1.05 * ref[0, 0]),
                S_O_NMSE=bool(np.all(nmse[1:] <= 1.1 * ref[1:])),
                dataset=all(
                    np.all(
                        np.asarray(v)
                        <= 1.1 * np.asarray(reference["nmse"]["by_dataset"][d])
                    )
                    for d, v in metric["nmse"]["by_dataset"].items()
                ),
                amplitude=bool(np.all((gain >= 0.75) & (gain <= 1.25))),
                bands=all(
                    np.all(abs(np.asarray(metric[k]["macro"])) <= 3.0103)
                    for k in [
                        "band_4_8_db",
                        "band_8_13_db",
                        "band_4_8_abs_db",
                        "band_8_13_abs_db",
                    ]
                ),
                envelope=bool(
                    np.all(np.asarray(metric["local_rms_correlation"]["macro"]) >= 0.5)
                ),
                correlation=bool(
                    np.all(np.asarray(metric["correlation"]["macro"]) > 0)
                ),
            )
            summary["waveform_checks"] = gates
            summary["waveform_reachability_pass"] = all(gates.values())
        prior = json.loads(
            (ROOT / cfg["source"]["integer_sum_reference"] / "summary.json").read_text()
        )
        for arm, old in [
            ("s31_o31_gain_source", "high_transient_source"),
            ("s0_o0_raw_source", "semantic_transient_source"),
        ]:
            for key in [
                "nmse",
                "rms_gain",
                "correlation",
                "local_rms_correlation",
                "band_4_8_abs_db",
                "band_8_13_abs_db",
            ]:
                np.testing.assert_allclose(
                    summaries[arm]["metrics"][key]["macro"],
                    prior["arms"][old]["metrics"][key]["macro"],
                    rtol=1e-12,
                    atol=1e-12,
                )
        write(
            out / "summary.json",
            dict(
                status="complete_waveform_pending_independent_replay_and_image_open",
                arms=summaries,
                role_pairs=cfg["selection"]["fixed_waveform_pairs"],
                controls="pure high raw/gain/gain_source and pure low original raw/gain/gain_source/raw_source",
                qualified_waveform_arms=[
                    arm
                    for arm in arms
                    if summaries[arm]["waveform_reachability_pass"]
                    and not arm.startswith("s31")
                ],
                position_windows=len(ex),
                physical_crops=400,
                source_subjects=60,
                CPU_seconds=role_wave_budget(),
                worker_usage=usage,
                independent_nested_selection_supported=information[
                    "outer_fit_selection_supported"
                ],
                can_claim_training_feasibility=False,
                original_main_acceptance="unchanged",
                validation_reads=0,
                confirmation_reads=0,
                test_reads=0,
            ),
        )


def role_mixed_review_worker(dataset):
    """Cold frozen-file decoders receive only the emitted integer arrays."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        cfg = STATE["cfg"]
        out = STATE["out"]
        rows = STATE["rows"]
        scales = STATE["scales"]
        directory = out / dataset
        pairs = [tuple(pair) for pair in cfg["selection"]["fixed_waveform_pairs"]]
        pairs.append((31, 31))
        frozen_cache = {}
        models = {}
        array_cache = OrderedDict()
        replayed = 0
        largest_nmse = 0.0
        for block_path in sorted(directory.glob("block_*.npz")):
            role_wave_budget()
            suffix = block_path.stem.split("_", 1)[1]
            hashes = json.loads(
                (directory / f"voltage_hashes_{suffix}.json").read_text()
            )
            expected_hash = {
                (record["row"], record["start_s"], record["arm"]): record[
                    "voltage_SHA256"
                ]
                for record in hashes
            }
            with np.load(block_path) as scores:
                for pair in pairs:
                    tag = f"s{pair[0]}_o{pair[1]}"
                    with np.load(
                        directory / "integer_ids" / f"{tag}_{suffix}.npz"
                    ) as sent:
                        for j, (index, start, fold) in enumerate(sent["keys"]):
                            role_wave_budget()
                            index, start, fold = map(int, [index, start, fold])
                            row = rows[index]
                            if row["split"] != "train" or row["dataset"] != dataset:
                                raise ValueError(
                                    "independent replay outside original train scope"
                                )
                            modes = ["raw", "gain", "gain_source"]
                            if pair == (0, 0):
                                modes.append("raw_source")
                            codes = {branch: sent[branch][j : j + 1] for branch in "SO"}
                            take = (scores["examples"][:, 0] == index) & (
                                scores["examples"][:, 1] == start
                            )
                            examples = scores["examples"][take]
                            model_key = pair, fold, "raw"
                            if model_key not in models:
                                for mode in modes:
                                    file = (
                                        directory
                                        / "models"
                                        / f"{tag}_fold{fold}_{mode}.npz"
                                    )
                                    models[pair, fold, mode] = load_role_mixed_decoder(
                                        file, frozen_cache=frozen_cache
                                    )
                            names = models[model_key].metadata["positions"][dataset]
                            lookup = {name.upper(): n for n, name in enumerate(names)}
                            sites = [
                                lookup[row["positions"][p].upper()]
                                for _, _, p in examples
                            ]
                            # Targets are read after the file-only decoder has received its IDs.
                            predictions = {
                                mode: models[pair, fold, mode].decode_ids(
                                    codes, dataset
                                )
                                for mode in modes
                            }
                            batch = base_tokenizer_batch(
                                rows,
                                examples,
                                scales,
                                array_cache=array_cache,
                                cache_limit=16,
                            )
                            mask = batch["mask"].reshape(len(examples), 6000, 1)
                            count = mask.sum(1)
                            for mode, prediction in predictions.items():
                                arm = f"{tag}_{mode}"
                                for branch, value in prediction.items():
                                    actual_hash = hashlib.sha256(
                                        np.ascontiguousarray(value).tobytes()
                                    ).hexdigest()
                                    if (
                                        actual_hash
                                        != expected_hash[index, start, arm][branch]
                                    ):
                                        raise AssertionError(
                                            "independent integer voltage hash changed"
                                        )
                                normalized = []
                                for branch, field in [
                                    ("sum", "x"),
                                    ("S", "s"),
                                    ("O", "o"),
                                ]:
                                    target = batch[field].reshape(
                                        len(examples), 6000, 1
                                    )
                                    value = prediction[branch][0, sites].reshape(
                                        target.shape
                                    )
                                    squared = np.where(mask, (value - target) ** 2, 0.0)
                                    mse = squared.sum(1) / np.maximum(count, 1)
                                    variance = batch["v" + field][:, 0, :1]
                                    normalized.append(
                                        np.where(count > 0, mse / variance, np.nan)
                                    )
                                actual_nmse = np.stack(normalized, axis=1)
                                expected_nmse = scores[arm + "__nmse"][take]
                                np.testing.assert_array_equal(
                                    actual_nmse, expected_nmse
                                )
                                finite = np.isfinite(actual_nmse)
                                largest_nmse = max(
                                    largest_nmse,
                                    float(
                                        np.max(
                                            abs(
                                                actual_nmse[finite]
                                                - expected_nmse[finite]
                                            ),
                                            initial=0.0,
                                        )
                                    ),
                                )
                                replayed += 1
        report = dict(
            status="exact",
            replayed_crop_arm_pairs=replayed,
            full_native_voltage_hashes="all S/O/sum values exact",
            independent_original_denominator_NMSE_max_abs=largest_nmse,
            input="saved final integers, frozen file references and fixed native metadata only",
            CPU_seconds=time.process_time(),
            validation_reads=0,
        )
        write(directory / "independent_integer_replay.json", report)
        return report


def render_role_mixed_waveforms(args, cfg, rows, scales):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.output.resolve()
    pairs = [tuple(pair) for pair in cfg["selection"]["fixed_waveform_pairs"]]
    mixed_pairs = [pair for pair in pairs if pair != (0, 0)]
    colors = {
        "target": "#172f40",
        "s31_o31_gain_source": "#7f8e98",
        "s0_o0_raw": "#4884b0",
        "s0_o2_gain_source": "#c17024",
        "s0_o4_gain_source": "#348a73",
        "s0_o16_gain_source": "#98669c",
    }
    figures = []
    array_cache = OrderedDict()
    frozen_cache = {}
    for dataset in sorted(scales["positions"]["eeg"]):
        directory = out / dataset
        with np.load(out / (dataset + "_metrics.npz")) as f:
            examples = f["examples"]
            joint_error = np.mean(
                [
                    f[f"s{a}_o{b}_gain_source__nmse"].mean((1, 2))
                    for a, b in mixed_pairs
                ],
                axis=0,
            )
        order = sorted(range(len(examples)), key=lambda n: tuple(examples[n]))
        fixed = order[0]
        random = min(
            order,
            key=lambda n: hashlib.sha256(
                json.dumps([17, dataset, examples[n].tolist()]).encode()
            ).hexdigest(),
        )
        difficult = int(
            np.nanargmin(abs(joint_error - np.nanquantile(joint_error, 0.95)))
        )
        for rule, ordinal in [
            ("fixed", fixed),
            ("random_seed17", random),
            ("P95_family_joint_error", difficult),
        ]:
            index, start, position = map(int, examples[ordinal])
            row = rows[index]
            selected_pair = mixed_pairs[0]
            tag = f"s{selected_pair[0]}_o{selected_pair[1]}"
            match = None
            for file in sorted((directory / "integer_ids").glob(tag + "_*.npz")):
                with np.load(file) as sent:
                    found = np.flatnonzero(
                        (sent["keys"][:, 0] == index) & (sent["keys"][:, 1] == start)
                    )
                    if len(found):
                        j = int(found[0])
                        match = file.stem[len(tag) + 1 :], j, int(sent["keys"][j, 2])
                        break
            if match is None:
                raise ValueError("fixed visual example missing final integers")
            suffix, j, fold = match
            predictions = {}
            native = None
            integer_vectors = {}
            for arm in colors:
                if arm == "target":
                    continue
                # Pair tags contain no ambiguity; decoder suffix is retained verbatim.
                pieces = arm.split("_")
                tag = "_".join(pieces[:2])
                mode = "_".join(pieces[2:])
                model = load_role_mixed_decoder(
                    directory / "models" / f"{tag}_fold{fold}_{mode}.npz",
                    frozen_cache=frozen_cache,
                )
                with np.load(directory / "integer_ids" / f"{tag}_{suffix}.npz") as sent:
                    codes = {branch: sent[branch][j : j + 1] for branch in "SO"}
                predictions[arm] = model.decode_ids(codes, dataset)
                names = model.metadata["positions"][dataset]
                native = next(
                    n
                    for n, name in enumerate(names)
                    if name.upper() == row["positions"][position].upper()
                )
                integer_vectors[arm] = {
                    branch: codes[branch][0, native].tolist() for branch in "SO"
                }
            batch = base_tokenizer_batch(
                rows,
                np.array([[index, start, position]]),
                scales,
                array_cache=array_cache,
                cache_limit=16,
            )
            factor = scales["positions"]["eeg"][dataset][row["positions"][position]][
                "scale"
            ]
            valid = batch["mask"][0].ravel().astype(bool)
            time_axis = np.arange(6000) / 200
            fig, axes = plt.subplots(3, 3, figsize=(18, 12), layout="constrained")
            errors = {}
            for line, (field, branch) in enumerate(
                [("x", "sum"), ("s", "S"), ("o", "O")]
            ):
                truth = batch[field][0].ravel() * factor
                waves = {"target": truth}
                waves.update(
                    {
                        arm: prediction[branch][0, native].ravel() * factor
                        for arm, prediction in predictions.items()
                    }
                )
                low = min(value[valid].min() for value in waves.values())
                high = max(value[valid].max() for value in waves.values())
                margin = max((high - low) * 0.04, 1e-12)
                errors[branch] = {
                    arm: float(np.sqrt(np.mean((value[valid] - truth[valid]) ** 2)))
                    for arm, value in waves.items()
                    if arm != "target"
                }
                for column, bounds in enumerate([(0, 30), (14, 16), (13.7, 14.3)]):
                    axis = axes[line, column]
                    for arm, value in waves.items():
                        axis.plot(
                            time_axis,
                            np.where(valid, value, np.nan),
                            color=colors[arm],
                            linewidth=0.8,
                            label=arm,
                        )
                    for boundary in range(0, 31, 2):
                        axis.axvline(boundary, color="#bac7cf", linewidth=0.4, zorder=0)
                    unit = (
                        "native export unit"
                        if dataset == "refed"
                        else row["source_unit"]
                    )
                    axis.set(
                        xlim=bounds,
                        ylim=(low - margin, high + margin),
                        xlabel="Time within visible 30s (s)",
                        ylabel=branch + " / " + unit,
                    )
                    if line == 0:
                        axis.set_title(
                            ["Full 30 s", "Fixed 2 s", "Integer boundary"][column]
                        )
                    if line == column == 0:
                        axis.legend(fontsize=7, ncol=2)
            fig.suptitle(
                f"{dataset} / {row['subject']} / {row['record']} / {row['positions'][position]} / +{start}s / {rule}\n"
                "Full-sequence integer-only voltage; S all K512; O theta/alpha/gamma role K8192; original low raw and high gain_source controls",
                fontsize=12,
            )
            path = out / f"{dataset}_{rule}.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            figures.append(
                dict(
                    path=path.name,
                    selection=rule,
                    example=[index, start, position],
                    subject=row["subject"],
                    record=row["record"],
                    fold=fold,
                    source_unit=row["source_unit"],
                    rmse_native=errors,
                    emitted_integer_vectors=integer_vectors,
                )
            )
    write(
        out / "figure_manifest.json",
        dict(
            status="rendered_pending_actual_open",
            figures=figures,
            P95_rule="nearest P95 of mean X/S/O NMSE across the three fixed hybrid gain_source arms; no best-window selection",
        ),
    )


def render_role_mixed_information(args, cfg, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.analysis.role_mixed_information import RoleInformationPanel

    out = args.output.resolve()
    canonical = ROOT / cfg["source"]["local_information"]
    information = ROOT / cfg["source"]["information_run"]
    complete = json.loads((information / "full_OOF_information_sets.json").read_text())
    records = complete["branch_records"]
    fields = ["power_4_8", "power_8_13", "rms_envelope"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for line, branch in enumerate("SO"):
        for arm, color in [("legacy", "#b47742"), ("transient", "#24897a")]:
            for column, field in enumerate(fields[:2]):
                values = [
                    records[arm][branch][str(mask)]["gain"][field] for mask in range(32)
                ]
                axes[line, column].plot(
                    range(32), values, color=color, marker=".", label=arm
                )
                axes[line, column].axhline(0.0, color="grey", linewidth=0.5)
                axes[line, column].axhline(
                    0.05, color="grey", linestyle=":", linewidth=0.6
                )
                axes[line, column].set(
                    title=branch + " " + field,
                    xlabel="Fixed high-role five-bit mask",
                    ylabel="Held single-ID MSE reduction",
                )
            mass = [
                records[arm][branch][str(mask)]["qualified_mass"] for mask in range(32)
            ]
            axes[line, 2].plot(range(32), mass, color=color, marker=".", label=arm)
        axes[line, 2].axhline(0.9, color="grey", linestyle="--")
        axes[line, 2].set(
            title=branch + " training support",
            xlabel="Fixed high-role five-bit mask",
            ylabel="Qualified held patch mass",
        )
        for axis in axes[line]:
            axis.set_xticks([0, 1, 2, 4, 8, 16, 31])
            axis.legend(fontsize=8)
    fig.suptitle(
        "All fixed role assignments: descriptive OOF information, not an independently selected model\nBit0 delta, bit1 theta, bit2 alpha, bit3 beta, bit4 gamma; complete 60-person/400-array train panel",
        fontsize=12,
    )
    fig.savefig(out / "all_role_information.png", dpi=150)
    plt.close(fig)
    with np.load(canonical / "training_attributes.npz") as f:
        bank = {key: f[key] for key in ["examples", "fixed_band", "valid"]}
    cached = {}
    for family in ["high", "semantic"]:
        with np.load(canonical / f"{family}_transient_readout_errors.npz") as f:
            cached[family] = {key: f[key] for key in f if key != "examples"}
    strata = {}
    figure_names = ["all_role_information.png"]
    for pair in cfg["selection"]["fixed_waveform_pairs"]:
        tag = f"s{pair[0]}_o{pair[1]}"
        strata[tag] = {}
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for line, (branch, mask) in enumerate(zip("SO", pair)):
            high, low = cached["high"], cached["semantic"]
            roles = []
            for role in range(5):
                panel = RoleInformationPanel(
                    high[branch + "_error"],
                    low[branch + "_error"],
                    high[branch + "_mass"],
                    low[branch + "_mass"],
                    bank,
                    rows,
                    bank["fixed_band"] == role,
                )
                roles.append(panel.evaluate(mask))
            strata[tag][branch] = roles
            colors = [
                "#bd7b32" if (mask >> role) & 1 else "#4686a7" for role in range(5)
            ]
            for column, field in enumerate(fields):
                values = [record["gain"][field] for record in roles]
                axes[line, column].bar(range(5), values, color=colors)
                axes[line, column].axhline(0.0, color="grey", linewidth=0.6)
                axes[line, column].set(
                    xticks=range(5),
                    xticklabels=["delta", "theta", "alpha", "beta", "gamma"],
                    ylabel="Held MSE reduction over same background",
                    title=branch + " " + field,
                )
        fig.suptitle(
            f"{tag}: complete emitted single-ID attributes by fixed frequency role\nBlue K512, orange K8192. Role strata are diagnostic; acceptance stays at the whole-branch level.",
            fontsize=12,
        )
        file = tag + "_role_information.png"
        fig.savefig(out / file, dpi=150)
        plt.close(fig)
        figure_names.append(file)
    write(out / "fixed_pair_role_information.json", strata)
    write(
        out / "information_figure_manifest.json",
        dict(
            status="rendered_pending_actual_open",
            figures=figure_names,
            source="canonical pure-fold held raw background/full-ID errors; exact fixed-role stitching",
        ),
    )


def review_role_mixed_waveform(args):
    cfg = yaml.safe_load(args.config.read_text())
    out = args.output.resolve()
    frozen_config = json.loads((out / "resolved_config.json").read_text())
    if cfg != frozen_config:
        raise ValueError(
            "review contract differs from the executed fixed waveform experiment"
        )
    summary = json.loads((out / "summary.json").read_text())
    stage = os.environ.get("TOKENIZER_ROLE_WAVE_EXECUTED_CPU")
    if stage is None:
        raise ValueError("review requires the measured preceding-stage CPU ledger")
    cfg["resources"] = dict(cfg["resources"], prior_CPU_seconds=float(stage))
    hard = yaml.safe_load((ROOT / cfg["source"]["hard_contract"]).read_text())
    basis = yaml.safe_load((ROOT / hard["data"]["basis_config"]).read_text())["source"]
    manifest, scales, _, _ = base_tokenizer_training_data(
        yaml.safe_load((ROOT / basis["g0_config"]).read_text()), ROOT
    )
    STATE.update(cfg=cfg, out=out, rows=manifest["rows"], scales=scales)
    with ProcessPoolExecutor(
        max_workers=cfg["resources"]["workers"],
        mp_context=multiprocessing.get_context("fork"),
    ) as pool:
        reports = list(
            pool.map(role_mixed_review_worker, sorted(scales["positions"]["eeg"]))
        )
    write(
        out / "independent_integer_replay.json",
        dict(
            status="exact",
            datasets=reports,
            original_and_all_mixed_endpoints="only saved final integers and frozen files",
            all_original_native_voltage_hashes_exact=True,
            original_denominator_NMSE_max_abs=0.0,
        ),
    )
    render_role_mixed_waveforms(args, cfg, manifest["rows"], scales)
    render_role_mixed_information(args, cfg, manifest["rows"])
    write(
        out / "review_completion.json",
        dict(
            status="replay_and_render_completed_pending_actual_open",
            total_campaign_CPU_seconds=role_wave_budget(),
            prior_nested_selection_supported=summary[
                "independent_nested_selection_supported"
            ],
            validation_reads=0,
        ),
    )


def worker(task):
    family, dataset = task
    cfg, run, rows, scales, plan, folds, out = (
        STATE[k] for k in ("cfg", "run", "rows", "scales", "plan", "folds", "out")
    )
    branch = cfg["families"][family]
    enc_path = run / branch["encoder"]
    dec_path = run / branch["decoder"]
    source_root = run / cfg["data"]["source_model_run"]
    priors = [
        load_source(source_root / f"{dataset}_fold{f}_transient.npz") for f in range(4)
    ]
    baseline = load_encoder(enc_path)
    if baseline.codebook_size != branch["size"]:
        raise ValueError("vocabulary changed")
    model = load_decoder(dec_path)
    replay = load_decoder(dec_path)
    names = baseline.metadata["positions"][dataset]
    if model.metadata["positions"][dataset] != names:
        raise ValueError("decoder layout mismatch")
    by_name = {v.upper(): i for i, v in enumerate(names)}
    # Both encoders share immutable atom transforms, but not search histories.
    candidate = copy(baseline)
    candidate.search_history = []
    jobs = sorted(
        [(s, c) for s, l in plan[dataset].items() for c in l],
        key=lambda e: (e[1]["row"], e[1]["start_s"]),
    )
    dest = out / family / dataset
    dest.mkdir(parents=True, exist_ok=True)
    cache = OrderedDict()
    expected = (sha(enc_path), sha(dec_path))
    for start in range(0, len(jobs), 12):
        block = dest / f"block_{start:04d}.npz"
        id_path = dest / f"ids_{start:04d}.npz"
        pairs = jobs[start : start + 12]
        keys = np.array(
            [[c["row"], c["start_s"], folds[dataset][s]] for s, c in pairs], np.int64
        )
        if block.exists():
            with np.load(block) as f:
                np.testing.assert_array_equal(f["keys"], keys)
            continue
        indices = defaultdict(list)
        examples = []
        metrics = defaultdict(list)
        disagreement = []
        for subject, crop in pairs:
            fold = folds[dataset][subject]
            cal = priors[fold]
            meta = cal.metadata
            if (
                subject in meta["fit_subjects"]
                or meta["held_fold"] != fold
                or meta["input_identity"] != baseline.metadata["input_identity"]
                or meta["scales_identity"] != baseline.metadata["scales_identity"]
            ):
                raise ValueError(
                    "held source ancestry mismatch before observation read"
                )
            native_source, seen, _ = base_tokenizer_joint_eeg(
                rows,
                dict(crop, positions=[]),
                scales,
                names,
                cache,
                allowed_splits=("train",),
            )
            x = native_source["x"].T.reshape(1, len(names), 15, 400)
            candidate.component_source = cal
            both = {
                arm: enc.encode(x, dataset, seen[None])
                for arm, enc in zip(ARMS, (baseline, candidate))
            }
            for arm in ARMS:
                for b in "SO":
                    np.testing.assert_array_equal(
                        np.any(both[arm][b][0] != 0, axis=1), seen
                    )
                    indices[arm + "_" + b].append(both[arm][b][0])
            disagreement.append(
                [
                    np.count_nonzero(both["legacy"][b] != both["transient"][b])
                    for b in "SO"
                ]
            )
        np.savez_compressed(id_path, keys=keys, identity=expected, **indices)
        with np.load(id_path) as ids:
            for j, (_, c) in enumerate(pairs):
                ex = [(c["row"], c["start_s"], p) for p in c["positions"]]
                sites = [by_name[rows[i]["positions"][p].upper()] for i, t, p in ex]
                data = base_tokenizer_batch(
                    rows, ex, scales, array_cache=cache, cache_limit=16
                )
                factors = input_scales(rows, ex, scales)
                for arm in ARMS:
                    sent = {b: ids[arm + "_" + b][j : j + 1] for b in "SO"}
                    one = model.decode_ids(sent, dataset)
                    again = replay.decode_ids(sent, dataset)
                    for k in one:
                        np.testing.assert_array_equal(one[k], again[k])
                    audit = joint_reconstruction_metrics(
                        {k: v[0, sites] for k, v in again.items()}, data, 1, factors
                    )
                    if audit["support"]["nonfinite_prediction_points"].any():
                        raise ValueError("nonfinite decoder output")
                    for k, v in audit["metrics"].items():
                        metrics[arm + "__" + k].append(v)
                examples.extend(ex)
        tmp = block.with_suffix(".pending.npz")
        np.savez_compressed(
            tmp,
            keys=keys,
            examples=examples,
            changed=disagreement,
            **{k: np.concatenate(v) for k, v in metrics.items()},
        )
        tmp.replace(block)
        write(
            dest / "progress.json",
            dict(crops=min(start + 12, len(jobs)), total=len(jobs)),
        )
    return family, dataset


def allocate_task(task):
    from src.tokenizers.eeg_component_transfer import IntegerSumSourceDecoder

    family, dataset = task
    cfg, run, rows, scales, out, prior, root = (
        STATE[k] for k in ("cfg", "run", "rows", "scales", "out", "prior", "root")
    )
    raw = load_decoder(run / prior["families"][family]["decoder"])
    site_lookup = {
        s.upper(): i for i, s in enumerate(raw.metadata["positions"][dataset])
    }
    source_root = run / prior["data"]["source_model_run"]
    models = {}
    for fold in range(4):
        source = load_source(source_root / f"{dataset}_fold{fold}_transient.npz")
        models[fold] = [
            IntegerSumSourceDecoder(raw, source, legacy=legacy, strength=strength)
            for legacy, strength in [(True, 1), (False, 1), (False, 0.5)]
        ]
    prefix = root / family / dataset
    cache = OrderedDict()
    parts = defaultdict(list)
    examples = []
    replay = set()
    gaps = []
    selected_fields = (
        "nmse",
        "nrmse",
        "rmse_native",
        "bias_native",
        "rms_gain",
        "correlation",
        "local_rms_correlation",
        "band_4_8_abs_db",
        "band_8_13_abs_db",
        "band_13_30_abs_db",
        "band_30_45_abs_db",
        "band_4_8_db",
        "band_8_13_db",
    )
    for path in sorted(prefix.glob("ids_*.npz")):
        with (
            np.load(path) as ids,
            np.load(path.with_name(path.name.replace("ids_", "block_"))) as matched,
        ):
            exblock = matched["examples"]
            for j, (row, start, fold) in enumerate(ids["keys"]):
                keep = (exblock[:, 0] == row) & (exblock[:, 1] == start)
                ex = exblock[keep]
                if (
                    rows[row]["split"] != "train"
                    or rows[row]["dataset"] != dataset
                    or rows[row]["subject"]
                    in models[fold][0].estimator.metadata["fit_subjects"]
                ):
                    raise ValueError(
                        "integer block is outside matching held training fold"
                    )
                sites = [site_lookup[rows[i]["positions"][p].upper()] for i, t, p in ex]
                batch = base_tokenizer_batch(
                    rows, ex, scales, array_cache=cache, cache_limit=16
                )
                sizes = input_scales(rows, ex, scales)
                examples.extend(ex)
                for arm in ARMS:
                    codes = {b: ids[arm + "_" + b][j : j + 1] for b in "SO"}
                    base = raw.decode_ids(codes, dataset)
                    replay_key = (fold, arm)
                    for name, model in zip(("wiener", "source", "half"), models[fold]):
                        if name == "half":
                            s = base["S"] + 0.5 * (full["S"] - base["S"])
                            pred = dict(sum=base["sum"], S=s, O=base["sum"] - s)
                        else:
                            pred = model.allocate(base, codes, dataset)
                            if name == "source":
                                full = pred
                        if replay_key not in replay:
                            fresh = model.decode_ids(codes, dataset)
                            for b, value in pred.items():
                                np.testing.assert_allclose(
                                    value, fresh[b], rtol=1e-13, atol=1e-13
                                )
                        np.testing.assert_array_equal(pred["sum"], base["sum"])
                        gaps.append(
                            float(np.abs(pred["S"] + pred["O"] - pred["sum"]).max())
                        )
                        audit = joint_reconstruction_metrics(
                            {b: v[0, sites] for b, v in pred.items()}, batch, 1, sizes
                        )
                        if audit["support"]["nonfinite_prediction_points"].any():
                            raise ValueError("nonfinite allocation")
                        for k in selected_fields:
                            parts[arm + "_" + name + "__" + k].append(
                                audit["metrics"][k]
                            )
                    replay.add(replay_key)
        write(
            out / f"{family}_{dataset}_progress.json",
            dict(scored=len(examples), block=path.name),
        )
    np.savez_compressed(
        out / f"{family}_{dataset}.npz",
        examples=examples,
        **{k: np.concatenate(v) for k, v in parts.items()},
    )
    return dict(
        family=family,
        dataset=dataset,
        windows=len(examples),
        max_abs_sum_closure=max(gaps),
        independent_entry_checks=len(replay) * 3,
    )


def run_allocation(args, cfg):
    out = args.output
    root = ROOT / cfg["comparison"]
    prior_result = json.loads((root / "summary.json").read_text())
    prior = prior_result["config"]
    run = ROOT / prior["data"]["run"]
    base = yaml.safe_load((ROOT / prior["data"]["basis_config"]).read_text())["source"]
    manifest, scales, _, _ = base_tokenizer_training_data(
        yaml.safe_load((ROOT / base["g0_config"]).read_text()), ROOT
    )
    plan = json.loads((ROOT / base["native_train_plan"]).read_text())["plan"]
    expected = sorted(
        (c["row"], c["start_s"], p)
        for g in plan.values()
        for crops in g.values()
        for c in crops
        for p in c["positions"]
    )
    if any(manifest["rows"][i]["split"] != "train" for i in {e[0] for e in expected}):
        raise ValueError("train only")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError("retain completed evaluation")
    STATE.update(
        cfg=cfg,
        run=run,
        rows=manifest["rows"],
        scales=scales,
        out=out,
        root=root,
        prior=prior,
    )
    tasks = [(family, d) for family in prior["families"] for d in sorted(plan)]
    with ProcessPoolExecutor(
        max_workers=cfg["resources"]["workers"],
        mp_context=multiprocessing.get_context("fork"),
    ) as pool:
        details = list(pool.map(allocate_task, tasks))
    records = {}
    for family in prior["families"]:
        pieces = defaultdict(list)
        for d in sorted(plan):
            with np.load(out / f"{family}_{d}.npz") as archive:
                for key in archive:
                    pieces[key].append(archive[key])
        data = {k: np.concatenate(v) for k, v in pieces.items()}
        ex = data["examples"]
        if sorted(map(tuple, ex.tolist())) != expected:
            raise ValueError("unmatched development of training arm")
        for enc in ARMS:
            original = prior_result["families"][family]["arms"][enc]["metrics"]["nmse"]
            for dec in ("wiener", "source", "half"):
                prefix = enc + "_" + dec + "__"
                a = base_training_aggregate(
                    {
                        k[len(prefix) :]: v
                        for k, v in data.items()
                        if k.startswith(prefix)
                    },
                    ex,
                    manifest["rows"],
                )
                metric = a["metrics"]
                np.testing.assert_array_equal(
                    metric["nmse"]["macro"][0], original["macro"][0]
                )
                n = np.ravel(metric["nmse"]["macro"])[1:]
                ref = np.ravel(original["macro"])[1:]
                ds = np.array(
                    [
                        np.ravel(metric["nmse"]["by_dataset"][d])[1:]
                        for d in sorted(plan)
                    ]
                )
                ds0 = np.array(
                    [np.ravel(original["by_dataset"][d])[1:] for d in sorted(plan)]
                )
                gain = np.ravel(metric["rms_gain"]["macro"])
                gates = dict(
                    joint=bool(n.sum() <= 0.95 * ref.sum() and np.all(n <= ref)),
                    dataset=bool(
                        np.sum(ds.sum(1) < ds0.sum(1)) >= 3 and np.all(ds <= 1.1 * ds0)
                    ),
                    amplitude=bool(np.all((0.75 <= gain) & (gain <= 1.25))),
                    bands=all(
                        np.all(np.abs(metric[k]["macro"]) <= 3.0103)
                        for k in (
                            "band_4_8_abs_db",
                            "band_8_13_abs_db",
                            "band_4_8_db",
                            "band_8_13_db",
                        )
                    ),
                    envelope=bool(
                        np.all(
                            np.asarray(metric["local_rms_correlation"]["macro"]) >= 0.5
                        )
                    ),
                )
                a.update(
                    gates=gates,
                    waveform_screen_candidate=bool(
                        dec != "wiener" and all(gates.values())
                    ),
                )
                records[family + "_" + enc + "_" + dec] = a
    passed = [k for k, a in records.items() if a["waveform_screen_candidate"]]
    selected = (
        min(
            passed,
            key=lambda k: (
                sum(np.ravel(records[k]["metrics"]["nmse"]["macro"])[1:]),
                0 if k.endswith("source") else 1,
            ),
        )
        if passed
        else None
    )
    write(
        out / "summary.json",
        dict(
            arms=records,
            selected_training_waveform=selected,
            needs_word_audit=True,
            source=cfg,
            details=details,
            validation_reads=0,
        ),
    )
    print("training eligibility", passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--role-mix-review', action='store_true')
    a = parser.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    if cfg.get('schema') == 'fixed_role_mixed_waveform_v1':
        if a.role_mix_review:
            return review_role_mixed_waveform(a)
        return run_role_mixed_waveform(a)
    if cfg["experiment_id"] == "TK-EEG-ACTUAL-INTEGER-SUM-ALLOCATION-v1":
        return run_allocation(a, cfg)
    data = cfg["data"]
    if data["allowed_splits"] != ["train"]:
        raise ValueError("train only")
    run = ROOT / data["run"]
    source = run / data["source_model_run"]
    if json.loads((source / "summary.json").read_text())["selected"] != "transient":
        raise ValueError("required held-domain empirical admission missing")
    base = yaml.safe_load((ROOT / data["basis_config"]).read_text())["source"]
    manifest, scales, _, _ = base_tokenizer_training_data(
        yaml.safe_load((ROOT / base["g0_config"]).read_text()), ROOT
    )
    rows = manifest["rows"]
    folds = json.loads((run / data["folds"]).read_text())
    plan = json.loads((ROOT / base["native_train_plan"]).read_text())["plan"]
    ex_all = []
    for d, people in plan.items():
        for s, crops in people.items():
            for c in crops:
                row = rows[c["row"]]
                if (row["split"], row["dataset"], row["subject"], row["modality"]) != (
                    "train",
                    d,
                    s,
                    "eeg",
                ):
                    raise ValueError("unexpected physical input")
                ex_all.extend((c["row"], c["start_s"], p) for p in c["positions"])
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "summary.json").exists():
        raise FileExistsError("retain completed run")
    STATE.update(
        cfg=cfg, run=run, rows=rows, scales=scales, folds=folds, plan=plan, out=a.output
    )
    tasks = [(family, d) for family in cfg["families"] for d in sorted(plan)]
    with ProcessPoolExecutor(
        max_workers=cfg["resources"]["workers"],
        mp_context=multiprocessing.get_context("fork"),
    ) as pool:
        list(pool.map(worker, tasks))
    summary = {}
    for family in cfg["families"]:
        data = defaultdict(list)
        all_ids = 0
        nchanged = np.zeros(2, np.int64)
        for dataset in sorted(plan):
            for path in sorted((a.output / family / dataset).glob("block_*.npz")):
                with np.load(path) as f:
                    for key in f.files:
                        if key in ("examples", "changed") or "__" in key:
                            data[key].append(f[key])
                    nchanged += f["changed"].sum(0)
                with np.load(path.with_name(path.name.replace("block_", "ids_"))) as f:
                    all_ids += f["legacy_S"].size
        joined = {k: np.concatenate(v) for k, v in data.items()}
        ex = joined["examples"]
        if sorted(map(tuple, ex.tolist())) != sorted(ex_all):
            raise ValueError("matched coverage differs")
        arms = {
            arm: base_training_aggregate(
                {
                    k.split("__")[1]: v
                    for k, v in joined.items()
                    if k.startswith(arm + "__")
                },
                ex,
                rows,
            )
            for arm in ARMS
        }
        rules = cfg["admission"]
        src = arms["legacy"]["metrics"]["nmse"]
        dst = arms["transient"]["metrics"]["nmse"]
        ratio = np.ravel(dst["macro"]) / np.ravel(src["macro"])
        ds = np.array(
            [
                np.ravel(dst["by_dataset"][d]) / np.ravel(src["by_dataset"][d])
                for d in sorted(plan)
            ]
        )
        gates = dict(
            S=bool(ratio[1] <= 1 - rules["S_relative_NMSE_improvement_min"]),
            O=bool(ratio[2] <= 1 - rules["O_relative_NMSE_improvement_min"]),
            X=bool(ratio[0] <= rules["maximum_X_relative_NMSE"]),
            libraries=int(np.all(ds[:, 1:] < 1, axis=1).sum())
            >= rules["required_datasets_both_improved"],
            tail=bool(np.all(ds <= rules["per_dataset_component_ratio_max"])),
        )
        summary[family] = dict(
            arms=arms,
            gates=gates,
            train_waveform_eligible=all(gates.values()),
            changed_ID_fraction=(nchanged / all_ids).tolist(),
            integers_per_arm=all_ids * 2,
        )
    write(
        a.output / "summary.json",
        dict(status="complete", families=summary, config=cfg, validation_reads=0),
    )
    print({k: v["gates"] for k, v in summary.items()})


if __name__ == "__main__":
    main()
