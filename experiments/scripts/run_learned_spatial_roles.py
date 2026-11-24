#!/usr/bin/env python3
"""Train balanced EEG electrode-band roles and evaluate disjoint training subjects."""

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

from experiments.scripts.evaluate_joint_tokenizer import input_scales, write
from src.analysis.dual_component_tokenizer import (
    joint_reconstruction_metrics,
    joint_reconstruction_summary,
)
from src.data.dual_component_tokenizer import (
    base_tokenizer_batch,
    base_tokenizer_joint_eeg,
    base_tokenizer_training_data,
)
from src.tokenizers.learned_role_dictionary import (
    balanced_roles,
    fit_roles,
)
from src.tokenizers.spatial_waveform_dictionary import BANDS, band_filter, load_model

STATE = {}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ranked(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checkpoint(path, **values):
    temp = path.with_suffix(".pending.npz")
    np.savez(temp, **values)
    temp.replace(path)


def cpu_limit():
    current = resource.getrusage(resource.RUSAGE_SELF)
    used = current.ru_utime + current.ru_stime
    if used > STATE["config"]["resources"]["total_CPU_limit_seconds"] / 4 - 10:
        raise RuntimeError(
            "per-dataset CPU allocation exhausted; retained outputs are resumable"
        )


def moments(carrier, targets, mask, variances):
    count = len(carrier[0])
    gram = np.zeros((2, 5, count, count, count))
    cross = np.zeros((2, 5, count, count))
    energy = np.zeros((2, 5, count))
    valid = mask.sum(1) > 0
    for bi in range(2):
        truth = targets[bi]
        for band in range(5):
            source = band_filter(carrier[bi], band).T
            y = band_filter(np.where(np.isfinite(truth), truth, 0.0), band).T
            common = source.T @ source
            for target in np.flatnonzero(valid):
                use = mask[target]
                bad = source[~use]
                denominator = use.sum() * variances[bi, target]
                gram[bi, band, target] = (common - bad.T @ bad) / denominator
                cross[bi, band, target] = source[use].T @ y[use, target] / denominator
                energy[bi, band, target] = np.sum(y[use, target] ** 2) / denominator
    return dict(gram=gram, cross=cross, energy=energy, valid=valid)


def average_stats(collection):
    # Each source item is already an equal-record summary for a single subject.
    counts = np.sum([item["valid"] for item in collection], axis=0)
    result = {}
    for key, trailing in [("gram", (None, None)), ("cross", (None,)), ("energy", ())]:
        sums = np.sum([item[key] for item in collection], axis=0)
        shape = (None, None, slice(None), *trailing)
        result[key] = sums / np.maximum(counts, 1)[shape]
    result["valid"] = counts > 0
    return result


def prepare_dataset(dataset):
    rows, scales, source, cfg, output = (
        STATE[key] for key in ("rows", "scales", "source", "config", "output")
    )
    subjects = STATE["plan"][dataset]
    names = source.metadata["positions"][dataset]
    native = {n.upper(): j for j, n in enumerate(names)}
    cache = OrderedDict()
    folder = output / dataset
    folder.mkdir(exist_ok=True)
    (folder / "cache").mkdir(exist_ok=True)
    statistics = {}
    files = {}
    counter = 0
    for subject, crops in subjects.items():
        records = defaultdict(list)
        files[subject] = []
        for crop in crops:
            cpu_limit()
            row = rows[crop["row"]]
            if (
                row["split"] != "train"
                or row["dataset"] != dataset
                or row["subject"] != subject
            ):
                raise ValueError("training subject boundary required before arrays")
            examples = np.array(
                [(crop["row"], crop["start_s"], p) for p in crop["positions"]],
                dtype=np.int32,
            )
            stored = folder / "cache" / (ranked([dataset, subject, crop])[:20] + ".npz")
            if stored.is_file():
                with np.load(stored, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["examples"], examples)
                    carrier = saved["carrier"]
                    target = saved["target"]
                    masks = saved["target_mask"]
                    variance = saved["target_variance"]
            else:
                fields, observed, _ = base_tokenizer_joint_eeg(
                    rows,
                    dict(crop, positions=[]),
                    scales,
                    names,
                    cache,
                    allowed_splits=("train",),
                )
                observed_values = fields["x"].T.reshape(1, len(names), 15, 400)
                cont = source.predict_continuous(observed_values)
                carrier = np.stack(
                    [cont[b][0].reshape(len(names), 6000) for b in ("S", "O")]
                )
                carrier[:, ~observed] = 0.0
                batch = base_tokenizer_batch(
                    rows, examples, scales, array_cache=cache, cache_limit=8
                )
                selected = np.array(
                    [native[row["positions"][p].upper()] for p in crop["positions"]]
                )
                target = np.zeros_like(carrier)
                masks = np.zeros((len(names), 6000), bool)
                variance = np.ones((2, len(names)))
                for bi, b in enumerate(("s", "o")):
                    target[bi, selected] = batch[b].reshape(len(examples), 6000)
                    variance[bi, selected] = batch["v" + b][:, 0, 0]
                masks[selected] = batch["mask"].reshape(len(examples), 6000)
                checkpoint(
                    stored,
                    examples=examples,
                    carrier=carrier,
                    target=target,
                    target_mask=masks,
                    target_variance=variance,
                    selected=selected,
                    observed=observed,
                    **{"batch_" + key: value for key, value in batch.items()},
                )
            record = moments(carrier, target, masks, variance)
            records[row["record"]].append(record)
            files[subject].append(stored)
            counter += 1
            write(
                folder / "progress.json",
                dict(
                    stage="training_statistics",
                    completed_crops=counter,
                    planned_crops=sum(map(len, subjects.values())),
                    subject=subject,
                    CPU_seconds=time.process_time(),
                ),
            )
        statistics[subject] = average_stats(
            [average_stats(v) for v in records.values()]
        )
    return statistics, files


def continuous_predict(carrier, roles, weights):
    result = {}
    for bi, branch in enumerate(("S", "O")):
        reconstructed = np.zeros_like(carrier[bi])
        for band in range(5):
            selected = np.flatnonzero(roles[branch] == band)
            filtered = band_filter(carrier[bi], band)
            reconstructed += (filtered[selected].T @ weights[branch][band]).T
        result[branch] = reconstructed[None].reshape(1, len(carrier[bi]), 15, 400)
    result["sum"] = result["S"] + result["O"]
    return result


def score_files(files, roles, weights, dataset):
    rows, scales = STATE["rows"], STATE["scales"]
    metrics, support = defaultdict(list), defaultdict(list)
    all_examples = []
    for file in files:
        cpu_limit()
        with np.load(file, allow_pickle=False) as stored:
            examples = stored["examples"]
            batch = {key[6:]: stored[key] for key in stored if key.startswith("batch_")}
            prediction = continuous_predict(stored["carrier"], roles, weights)
            selected = stored["selected"]
        audited = joint_reconstruction_metrics(
            {key: value[0, selected] for key, value in prediction.items()},
            batch,
            1,
            input_scales(rows, examples, scales),
        )
        for key, value in audited["metrics"].items():
            metrics[key].append(value)
        for key, value in audited["support"].items():
            support[key].append(value)
        all_examples.extend(examples)
    return (
        np.array(all_examples, np.int32),
        {k: np.concatenate(v) for k, v in metrics.items()},
        {k: np.concatenate(v) for k, v in support.items()},
    )


def dataset_experiment(dataset):
    with threadpool_limits(1):
        started = time.monotonic()
        stats, files = prepare_dataset(dataset)
        cfg, source, output = STATE["config"], STATE["source"], STATE["output"]
        setting = cfg["training"]
        original = np.arange(len(source.metadata["positions"][dataset])) % 5
        subjects = sorted(stats, key=lambda s: ranked([setting["seed"], dataset, s]))
        folds = {s: j % setting["folds"] for j, s in enumerate(subjects)}
        folder = output / dataset
        captures = {
            arm: dict(examples=[], metrics=defaultdict(list), support=defaultdict(list))
            for arm in ("fixed", "learned")
        }
        fitted = []
        for fold in range(setting["folds"]):
            cpu_limit()
            training = [stats[s] for s in subjects if folds[s] != fold]
            if not training:
                raise ValueError(
                    "held subject fold requires disjoint training subjects"
                )
            combined = average_stats(training)
            models = {arm: dict(roles={}, weights={}) for arm in captures}
            trace = {}
            for bi, branch in enumerate(("S", "O")):
                branch_stats = {
                    key: combined[key][bi] for key in ("gram", "cross", "energy")
                }
                fixed, _ = fit_roles(branch_stats, original, setting["ridge_fraction"])
                roles, learned, history = balanced_roles(
                    branch_stats,
                    original,
                    setting["ridge_fraction"],
                    setting["maximum_accepted_swaps_per_branch_dataset"],
                    setting["minimum_relative_training_objective_improvement_per_swap"],
                )
                models["fixed"]["roles"][branch] = original
                models["fixed"]["weights"][branch] = fixed
                models["learned"]["roles"][branch] = roles
                models["learned"]["weights"][branch] = learned
                trace[branch] = history
            held = [s for s in subjects if folds[s] == fold]
            record = dict(
                fold=fold,
                train_subjects=[s for s in subjects if folds[s] != fold],
                held_subjects=held,
                trace=trace,
                roles={b: v.tolist() for b, v in models["learned"]["roles"].items()},
            )
            fitted.append(record)
            write(
                folder / "fold_progress.json",
                dict(
                    stage="held_training_waveforms",
                    fold=fold,
                    folds=setting["folds"],
                    CPU_seconds=time.process_time(),
                    models=fitted,
                ),
            )
            for arm, model in models.items():
                ex, met, sup = score_files(
                    [p for s in held for p in files[s]],
                    model["roles"],
                    model["weights"],
                    dataset,
                )
                captures[arm]["examples"].append(ex)
                for key, value in met.items():
                    captures[arm]["metrics"][key].append(value)
                for key, value in sup.items():
                    captures[arm]["support"][key].append(value)
            write(
                folder / "fold_progress.json",
                dict(
                    stage="completed_fold",
                    fold=fold,
                    folds=setting["folds"],
                    CPU_seconds=time.process_time(),
                    models=fitted,
                ),
            )
        result = {}
        for arm, capture in captures.items():
            examples = np.concatenate(capture["examples"])
            metrics = {key: np.concatenate(v) for key, v in capture["metrics"].items()}
            support = {key: np.concatenate(v) for key, v in capture["support"].items()}
            checkpoint(
                folder / (arm + "_held_metrics.npz"),
                examples=examples,
                **metrics,
                **{"support_" + k: v for k, v in support.items()},
            )
            result[arm] = joint_reconstruction_summary(
                metrics, support, examples.tolist(), STATE["rows"]
            )
        write(
            folder / "summary.json",
            dict(
                status="completed",
                summaries=result,
                folds=fitted,
                seconds=time.monotonic() - started,
                CPU_seconds=time.process_time(),
                validation_reads=0,
            ),
        )
        # Keep all-data fitting statistics local to this process until the global admission decision.
        combined = average_stats(list(stats.values()))
        checkpoint(folder / "training_moments.npz", **combined)
        return dataset, result


def global_summary(output, results):
    rows = STATE["rows"]
    summaries = {}
    for arm in ("fixed", "learned"):
        examples = []
        metrics = defaultdict(list)
        support = defaultdict(list)
        for dataset in sorted(results):
            with np.load(
                output / dataset / (arm + "_held_metrics.npz"), allow_pickle=False
            ) as saved:
                examples.extend(saved["examples"])
                for key in saved:
                    if key == "examples":
                        continue
                    (support if key.startswith("support_") else metrics)[
                        key[8:] if key.startswith("support_") else key
                    ].append(saved[key])
        met = {k: np.concatenate(v) for k, v in metrics.items()}
        sup = {k: np.concatenate(v) for k, v in support.items()}
        checkpoint(
            output / (arm + "_held_metrics.npz"),
            examples=np.asarray(examples, np.int32),
            **met,
            **{"support_" + k: v for k, v in sup.items()},
        )
        summaries[arm] = joint_reconstruction_summary(met, sup, examples, rows)
    a = summaries["learned"]["metrics"]["nmse"]
    b = summaries["fixed"]["metrics"]["nmse"]
    selection = STATE["config"]["selection"]
    an = np.array(a["macro"]).ravel()
    bn = np.array(b["macro"]).ravel()
    base_subject = {
        (x["dataset"], x["subject"]): np.array(x["value"]).ravel()
        for x in b["by_subject"]
    }
    gains = [
        (base_subject[x["dataset"], x["subject"]][1] - np.array(x["value"]).ravel()[1])
        for x in a["by_subject"]
    ]
    datasets = {
        d: float(
            (np.array(b["by_dataset"][d]).ravel()[1] - np.array(v).ravel()[1])
            / np.array(b["by_dataset"][d]).ravel()[1]
        )
        for d, v in a["by_dataset"].items()
    }
    checks = dict(
        S_improvement=float((bn[1] - an[1]) / bn[1])
        >= selection["required_macro_S_NMSE_relative_improvement"],
        X_O_guard=bool(
            np.all(
                an[[0, 2]]
                <= bn[[0, 2]] * selection["required_macro_X_O_NMSE_max_ratio"]
            )
        ),
        datasets=sum(v > 0 for v in datasets.values())
        >= selection["required_dataset_count_S_improved"],
        dataset_S_guard=all(
            v >= 1 - selection["maximum_dataset_S_NMSE_ratio"]
            for v in datasets.values()
        ),
        subjects=float(np.mean(np.array(gains) > 0))
        >= selection["required_subject_fraction_S_improved"],
    )
    decision = dict(
        eligible=all(checks.values()),
        checks=checks,
        relative_S_improvement=float((bn[1] - an[1]) / bn[1]),
        improved_subjects=int(np.sum(np.array(gains) > 0)),
        subjects=len(gains),
        dataset_relative_S_improvement=datasets,
    )
    write(
        output / "continuous_crossfit.json",
        dict(
            status="completed",
            summaries=summaries,
            decision=decision,
            boundary="All source waveforms are G0 train. Only role selection and spatial coefficients are crossfit; source Wiener/atoms already saw these subjects.",
            validation_reads=0,
        ),
    )
    return decision


def adaptive_dataset(dataset):
    from collections import Counter
    from src.tokenizers.eeg_transient_gain import load_model as load_source
    from src.tokenizers.adaptive_band_allocation import (
        covariance_from_source,
        band_waves,
        regular_covariance,
        choose_roles,
        extract_coordinates,
        synthesize,
    )

    population = STATE["dynamic_population"]
    cases = [c for c in population["cases"] if c["dataset"] == dataset]
    names = population["native_positions"][dataset]
    rows, scales, out = STATE["rows"], STATE["scales"], STATE["out"] / dataset
    out.mkdir(exist_ok=True)
    source_dir = STATE["dynamic_source"]
    n = len(names)
    samples = []
    for c in cases:
        i = c["crop"]["row"]
        if rows[i]["split"] != "train" or rows[i]["subject"] != c["subject"]:
            raise ValueError("train-only source before observation access")
        signals, seen, _ = base_tokenizer_joint_eeg(
            rows, dict(c["crop"], positions=[]), scales, names, allowed_splits=("train",)
        )
        samples.append((signals["x"].T.reshape(1, n, 15, 400).astype(np.float32), seen))
    strategies = ("fixed", "energy", "innovation")
    cache = OrderedDict()
    for fold in range(4):
        completed = out / f"fold_{fold}_metrics.npz"
        if completed.is_file():
            continue
        source = load_source(source_dir / f"{dataset}_fold{fold}_transient.npz")
        fit_people = set(c["subject"] for c in cases if c["fold"] != fold)
        if fit_people != set(source.metadata["fit_subjects"]) or source.metadata["held_fold"] != fold:
            raise ValueError("source fold changed")
        fit = [(c, s) for c, s in zip(cases, samples) if c["fold"] != fold]
        frequencies = Counter((c["subject"], rows[c["crop"]["row"]]["record"]) for c, s in fit)
        records_per_person = Counter(person for person, record in frequencies)
        exchange_parent = STATE.get("dynamic_exchange_parent")
        if exchange_parent is not None:
            with np.load(exchange_parent / dataset / f"fold_{fold}_model.npz") as z:
                np.testing.assert_array_equal(z["names"], names)
                assert set(z["fit_people"]) == fit_people
                cov = {b: z[b].copy() for b in "SO"}
            strategies = ("fixed", "exchange")
        else:
            grams = {b: np.zeros((5, n, n)) for b in "SO"}
            mass = np.zeros((n, n))
            for c, (x, seen) in fit:
                prediction = source.predict_continuous(x, dataset, seen[None])
                person = c["subject"]
                weight = 1 / (
                    frequencies[person, rows[c["crop"]["row"]]["record"]] * records_per_person[person]
                )
                for b in "SO":
                    spec = np.fft.rfft(prediction[b][0].reshape(n, 6000))
                    grams[b] += weight * covariance_from_source(spec)
                mass += weight * (seen[:, None] & seen[None])
            cov = {
                b: np.stack([regular_covariance(g / np.maximum(mass, 1e-12)) for g in grams[b]])
                for b in "SO"
            }
            checkpoint(
                out / f"fold_{fold}_model.npz",
                **cov,
                names=np.array(names),
                fit_people=np.array(sorted(fit_people)),
            )
        operator_cache = {}
        sum_proxy_change = []
        stats = defaultdict(list)
        gallery = {}
        coords_all = defaultdict(list)
        example_ids = []
        for c, (x, seen) in zip(cases, samples):
            if c["fold"] != fold:
                continue
            crop = c["crop"]
            ex = np.asarray([(crop["row"], crop["start_s"], p) for p in crop["positions"]])
            sites = np.asarray(c["native"])
            example_ids.extend(ex)
            dense = source.predict_continuous(x, dataset, seen[None])
            waves = {b: band_waves(np.fft.rfft(dense[b][0].reshape(n, 6000))) for b in "SO"}
            outputs = {}
            for strategy in strategies:
                pred = {}
                for b in "SO":
                    if strategy == "exchange":
                        from src.tokenizers.adaptive_band_allocation import (
                            choose_exchanges,
                            exchange_costs,
                        )

                        mask_key = seen.tobytes()
                        if b not in operator_cache or operator_cache[b][0] != mask_key:
                            scale_table = {
                                site.upper(): v
                                for site, v in scales["positions"]["eeg"][dataset].items()
                            }
                            weights = np.array(
                                [
                                    1
                                    / float(
                                        np.ravel(
                                            scale_table[name.upper()]["target_variance"][b.lower()]
                                        )[0]
                                    )
                                    for name in names
                                ]
                            )
                            weights[~seen] = 0
                            operator_cache[b] = (mask_key, exchange_costs(cov[b], seen, weights))
                        roles, change = choose_exchanges(waves[b], seen, operator_cache[b][1])
                        sum_proxy_change.append(change)
                    else:
                        roles = choose_roles(waves[b], cov[b], seen, strategy)
                    values = extract_coordinates(waves[b], roles)
                    pred[b] = synthesize(values, roles, cov[b])[sites]
                    coords_all[strategy + "_" + b].append(roles)
                pred["sum"] = pred["S"] + pred["O"]
                outputs[strategy] = pred
            batch = base_tokenizer_batch(rows, ex, scales, array_cache=cache, cache_limit=8)
            for strategy, pred in outputs.items():
                result = joint_reconstruction_metrics(pred, batch, 1, input_scales(rows, ex, scales))
                if result["support"]["nonfinite_prediction_points"].any():
                    raise ValueError("nonfinite reconstructed voltage")
                for k, v in result["metrics"].items():
                    stats[strategy + "__" + k].append(v)
                truth = batch["s"]
                critical = batch["mask"] & (np.abs(truth) >= 2 * np.sqrt(batch["vs"]))
                count = critical.sum((1, 2))
                ss = (((pred["S"] - truth) ** 2 / batch["vs"]) * critical).sum((1, 2))
                stats[strategy + "__S_true_high"].append(
                    np.divide(ss, count, out=np.full(len(count), np.nan), where=count > 0)[
                        :, None, None
                    ]
                )
            for j, e in enumerate(ex):
                which = STATE["dynamic_examples"].get(tuple(e))
                if which is not None:
                    for strategy, pred in outputs.items():
                        for b, v in pred.items():
                            gallery[f"{which}_{strategy}_{b}"] = v[j]
        checkpoint(out / f"fold_{fold}_roles.npz", **{k: np.stack(v) for k, v in coords_all.items()})
        if sum_proxy_change:
            checkpoint(out / f"fold_{fold}_proxy.npz", changes=np.stack(sum_proxy_change))
            if np.max(sum_proxy_change) > 1e-12:
                raise ValueError("proxy failed exact exchange objective")
        checkpoint(out / f"fold_{fold}_gallery.npz", **gallery)
        checkpoint(
            completed,
            examples=np.asarray(example_ids),
            **{k: np.concatenate(v) for k, v in stats.items()},
        )
        write(out / "progress.json", dict(completed_fold=fold, CPU_seconds=time.process_time()))
    return dataset


def run_adaptive_band(args, cfg):
    from src.analysis.dual_component_tokenizer import base_training_aggregate

    if cfg["data"] != dict(
        allowed_splits=["train"],
        validation_access="forbidden",
        confirmation_access="forbidden",
        test_access="forbidden",
    ):
        raise ValueError("train-only dynamic frequency contract required")
    old = yaml.safe_load((ROOT / cfg["source"]["word_config"]).read_text())
    hard = yaml.safe_load((ROOT / old["source"]["hard_contract"]).read_text())
    basis = yaml.safe_load((ROOT / hard["data"]["basis_config"]).read_text())["source"]
    manifest, scales, _, _ = base_tokenizer_training_data(
        yaml.safe_load((ROOT / basis["g0_config"]).read_text()), ROOT
    )
    source = ROOT / old["source"]["output"]
    pop = json.loads((source / "population.json").read_text())
    if (
        len(pop["cases"]) != 400
        or len(pop["subjects"]) != 60
        or pop["cfg_identity"] != manifest["cfg_identity"]
        or pop["scales_identity"] != manifest["scales_identity"]
    ):
        raise ValueError("original training population required")
    run = ROOT / hard["data"]["run"]
    gallery = json.loads(
        (
            run
            / "component_source_transfer/integer_sum_source_v1/training_examples/waveform_examples.json"
        ).read_text()
    )["examples"]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "summary.json").exists():
            raise FileExistsError("retain completed dynamic feature result")
        write(
            out / "resolved_config.json",
            dict(
                config=cfg,
                population=sha(source / "population.json"),
                input=manifest["cfg_identity"],
                scales=manifest["scales_identity"],
            ),
        )
        STATE.update(
            rows=manifest["rows"],
            scales=scales,
            out=out,
            dynamic_population=pop,
            dynamic_source=run / cfg["source"]["source_run"],
            dynamic_examples={
                (c["row"], c["start_s"], c["position"]): i for i, c in enumerate(gallery)
            },
        )
        strategies = ("fixed", "energy", "innovation")
        if cfg["schema"] == "adaptive_band_exchange_v1":
            STATE["dynamic_exchange_parent"] = run / cfg["source"]["initial_run"]
            strategies = ("fixed", "exchange")
        with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("fork")) as pool:
            done = list(pool.map(adaptive_dataset, sorted(pop["native_positions"])))
        values = defaultdict(list)
        for dataset in done:
            for fold in range(4):
                with np.load(out / dataset / f"fold_{fold}_metrics.npz") as z:
                    for key in z:
                        values[key].append(z[key])
        arrays = {k: np.concatenate(v) for k, v in values.items()}
        ex = arrays["examples"]
        expected = [
            (c["crop"]["row"], c["crop"]["start_s"], p)
            for c in pop["cases"]
            for p in c["crop"]["positions"]
        ]
        assert len(ex) == 17040 and sorted(map(tuple, ex)) == sorted(expected)
        arms = {}
        for name in strategies:
            arms[name] = base_training_aggregate(
                {k.split("__")[1]: v for k, v in arrays.items() if k.startswith(name + "__")},
                ex,
                manifest["rows"],
            )
        ref = arms["fixed"]["metrics"]
        if "exchange" in strategies:
            registered = json.loads((STATE["dynamic_exchange_parent"] / "summary.json").read_text())[
                "arms"
            ]["fixed"]["metrics"]
            for k in (
                "nmse",
                "rms_gain",
                "band_4_8_abs_db",
                "band_8_13_abs_db",
                "local_rms_correlation",
            ):
                np.testing.assert_array_equal(ref[k]["macro"], registered[k]["macro"])
        for name in strategies[1:]:
            m = arms[name]["metrics"]
            nm, oldnm = [np.ravel(q["nmse"]["macro"]) for q in (m, ref)]
            ratio = np.stack(
                [
                    np.ravel(v) / np.ravel(ref["nmse"]["by_dataset"][ds])
                    for ds, v in m["nmse"]["by_dataset"].items()
                ]
            )
            improved = sum(
                np.ravel(v)[1:].sum() < np.ravel(ref["nmse"]["by_dataset"][ds])[1:].sum()
                for ds, v in m["nmse"]["by_dataset"].items()
            )
            rms = np.ravel(m["rms_gain"]["macro"])
            checks = dict(
                total=nm[0] <= 0.98 * oldnm[0],
                joint=nm[1:].sum() <= 0.95 * oldnm[1:].sum(),
                components=np.all(nm[1:] <= oldnm[1:]),
                datasets=improved >= 3 and ratio.max() <= 1.10,
                RMS=np.all((rms >= 0.75) & (rms <= 1.25)),
                bands=all(
                    np.max(abs(np.asarray(m[k]["macro"]))) <= 3.0103
                    for k in ("band_4_8_db", "band_8_13_db", "band_4_8_abs_db", "band_8_13_abs_db")
                ),
                envelope=min(np.ravel(m["local_rms_correlation"]["macro"])) >= 0.5,
                high_S=np.ravel(m["S_true_high"]["macro"])[0]
                <= 1.05 * np.ravel(ref["S_true_high"]["macro"])[0],
            )
            arms[name]["checks"] = checks
            arms[name]["pass"] = all(checks.values())
        write(
            out / "summary.json",
            dict(
                arms=arms,
                selected=[k for k in strategies[1:] if arms[k]["pass"]],
                status="continuous_only",
                original_training_positions=len(ex),
                validation_reads=0,
                confirmation_reads=0,
                test_reads=0,
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "experiments/configs/physiology_semantic_tokenizer/dual_component_learned_roles_v1.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("schema") in ("adaptive_band_allocation_v1", "adaptive_band_exchange_v1"):
        return run_adaptive_band(args, cfg)
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    with (out / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (
            cfg["schema"] != "dual_component_learned_roles_v1"
            or cfg["data"]["test_access"] != "forbidden"
        ):
            raise ValueError("versioned G0 train-only balanced role contract required")
        config = yaml.safe_load(
            (
                ROOT
                / "experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml"
            ).read_text()
        )
        manifest, scales, _, _ = base_tokenizer_training_data(config, ROOT)
        plan = json.loads((ROOT / cfg["source"]["plan"]).read_text())["plan"]
        for dataset, subjects in plan.items():
            for subject, crops in subjects.items():
                if any(
                    manifest["rows"][v["row"]]["split"] != "train"
                    or manifest["rows"][v["row"]]["subject"] != subject
                    for v in crops
                ):
                    raise ValueError("train scope required before any arrays")
        source = load_model(ROOT / cfg["source"]["model"])
        if (
            source.metadata["input_identity"] != manifest["cfg_identity"]
            or source.metadata["scales_identity"] != manifest["scales_identity"]
        ):
            raise ValueError("frozen source G0 identity mismatch")
        resolved = dict(
            config=cfg,
            source_sha256=sha(ROOT / cfg["source"]["model"]),
            input_identity=manifest["cfg_identity"],
            scales_identity=manifest["scales_identity"],
            plan=plan,
        )
        # Compare the persisted JSON representation, including YAML null-key spelling.
        resolved = json.loads(json.dumps(resolved))
        recorded = out / "resolved_config.json"
        if recorded.exists() and json.loads(recorded.read_text()) != resolved:
            raise ValueError("retained role experiment configuration or source changed")
        if (out / "summary.json").exists():
            return
        write(recorded, resolved)
        STATE.update(
            rows=manifest["rows"],
            scales=scales,
            source=source,
            config=cfg,
            output=out,
            plan=plan,
        )
        started = time.monotonic()
        with ProcessPoolExecutor(
            max_workers=cfg["resources"]["dataset_workers"],
            mp_context=multiprocessing.get_context("fork"),
        ) as pool:
            result = dict(pool.map(dataset_experiment, sorted(plan)))
        decision = global_summary(out, result)
        write(
            out / "summary.json",
            dict(
                status="continuous_admitted"
                if decision["eligible"]
                else "continuous_crossfit_rejected",
                decision=decision,
                elapsed_seconds=time.monotonic() - started,
                validation_reads=0,
                confirmation_reads=0,
                test_reads=0,
            ),
        )
        print(json.dumps(decision))


if __name__ == "__main__":
    main()
