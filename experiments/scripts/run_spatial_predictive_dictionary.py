"""Train and replay matched direct and closed-loop spatial innovation dictionaries."""

import argparse
from collections import OrderedDict, defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/scripts"))
from src.tokenizers.spatial_predictive_dictionary import (
    SpatialPredictiveDictionary,
    band_limit,
    fit_book,
    fit_tree,
    load_model,
)
from src.tokenizers.spatial_waveform_dictionary import load_model as load_prior
from src.tokenizers.eeg_transient_gain import load_model as load_source
from src.data.dual_component_tokenizer import (
    base_tokenizer_training_data,
    base_tokenizer_joint_eeg,
    base_tokenizer_batch,
)
from src.analysis.dual_component_tokenizer import (
    joint_reconstruction_metrics,
    joint_reconstruction_summary,
    base_training_aggregate,
)
from src.analysis.tokenizer_pattern_audit import (
    SparsePatternAudit,
    attributes,
    feature_means,
)
from evaluate_joint_tokenizer import input_scales, write

FIELDS = {"power_4_8": (0, 1), "power_8_13": (1, 2), "rms_envelope": (2, 10)}
STATE = {}
ARMS = ("direct", "predictive")


def cpu_guard():
    unit = os.environ.get("SPATIAL_PREDICTIVE_UNIT")
    used = (
        int(
            subprocess.check_output(
                ["systemctl", "--user", "show", unit, "-p", "CPUUsageNSec", "--value"],
                text=True,
            )
        )
        / 1e9
        if unit
        else time.process_time()
    )
    if used > STATE["cfg"]["resources"]["CPU_budget_seconds"] - 30:
        raise RuntimeError("CPU budget reached; completed folds retained")
    return used


def sample_rows(cases, fit, count, seed):
    rng = np.random.default_rng(seed)
    grouped = defaultdict(list)
    for i in fit:
        c = cases[i]
        for site in np.flatnonzero(c["seen"]):
            grouped[c["subject"]].extend((i, int(site), t) for t in range(15))
    selected = []
    for subject, items in sorted(grouped.items()):
        which = rng.choice(len(items), min(count, len(items)), replace=False)
        selected.extend(items[j] for j in which)
    if len(selected) > 8192:
        selected = [selected[i] for i in rng.choice(len(selected), 8192, replace=False)]
    return selected


def full_source(cases, fold, dataset):
    model = load_source(
        ROOT
        / STATE["cfg"]["source"]["gain_run"]
        / f"{dataset}_fold{fold}_transient.npz"
    )
    expected = {c["subject"] for c in cases if c["fold"] != fold}
    if (
        model.metadata["held_fold"] != fold
        or set(model.metadata["fit_subjects"]) != expected
    ):
        raise ValueError("fold-clean source required")
    output = {b: [] for b in "SO"}
    for c in cases:
        p = model.predict_continuous(c["x"][None], dataset, c["seen"][None])
        for b in "SO":
            output[b].append(band_limit(p[b])[0].astype(np.float32))
    return {b: np.stack(v) for b, v in output.items()}


def training_tree(cases, physical, fit, branch):
    channels = physical[branch].shape[1]
    covariance = np.zeros((channels, channels))
    mass = np.zeros_like(covariance)
    records = Counter((cases[i]["subject"], cases[i]["record"]) for i in fit)
    subject_records = Counter(s for s, r in records)
    for i in fit:
        c = cases[i]
        weight = 1 / (
            subject_records[c["subject"]] * records[c["subject"], c["record"]]
        )
        x = physical[branch][i].reshape(channels, 6000).astype(float)
        valid = c["seen"]
        pair = valid[:, None] & valid[None]
        covariance += weight * (x @ x.T / 6000) * pair
        mass += weight * pair
    covariance /= np.maximum(mass, 1e-12)
    order, parent, gain, scale = fit_tree(covariance)
    return dict(order=order, parent=parent, gain=gain, scale=scale), covariance


def innovations(physical, branch, tree, selected, previous=None):
    result = []
    for i, site, t in selected:
        parent = tree["parent"][site]
        reference = (
            0.0
            if parent < 0
            else (
                physical[branch][i, parent, t]
                if previous is None
                else previous[branch][i, parent, t]
            )
        )
        value = (physical[branch][i, site, t] - tree["gain"][site] * reference) / tree[
            "scale"
        ][site]
        result.append(value)
    return np.array(result, np.float32)


def pattern(cases, codes, fit, held, branch, backgrounds):
    audit = SparsePatternAudit(512, backgrounds, [], 10, FIELDS)
    support = defaultdict(set)
    for i in fit:
        c = cases[i]
        valid = c["complete"]
        words = codes[branch][i, c["native"]][valid]
        audit.fit(words, c["background"][valid], c["target"][branch][valid])
        for code in np.unique(words):
            support[int(code)].add(c["subject"])
    result = []
    qualified = 0
    total = 0
    for i in held:
        c = cases[i]
        valid = c["complete"]
        words = codes[branch][i, c["native"]][valid]
        background, _, conditional = audit.read(words, c["background"][valid])
        y = c["target"][branch][valid]
        error = np.zeros((*valid.shape, 2, 3))
        error[valid] = np.stack(
            [feature_means((v - y) ** 2, FIELDS) for v in [background, conditional]],
            axis=1,
        )
        result.append(error.sum(1) / np.maximum(valid.sum(1), 1)[:, None, None])
        qualified += int(
            np.sum(
                (audit.local.count[words] >= 8)
                & np.array([len(support[int(w)]) >= 2 for w in words])
            )
        )
        total += len(words)
    return np.concatenate(result), qualified, total


def dataset_worker(dataset):
    cfg = STATE["cfg"]
    rows = STATE["rows"]
    scales = STATE["scales"]
    plan = STATE["plan"][dataset]
    folds = STATE["folds"][dataset]
    directory = STATE["out"] / dataset
    directory.mkdir(exist_ok=True)
    prior = load_prior(ROOT / STATE["base"]["waveform_dictionary"])
    names = prior.metadata["positions"][dataset]
    lookup = {n.upper(): i for i, n in enumerate(names)}
    cache = OrderedDict()
    cases = []
    for subject, crops in sorted(plan.items()):
        for crop in crops:
            fields, seen, _ = base_tokenizer_joint_eeg(
                rows,
                dict(crop, positions=[]),
                scales,
                names,
                cache,
                allowed_splits=("train",),
            )
            ex = np.array(
                [(crop["row"], crop["start_s"], p) for p in crop["positions"]]
            )
            batch = base_tokenizer_batch(
                rows, ex, scales, array_cache=cache, cache_limit=8
            )
            # Scalar variances remain broadcast views, without retaining expanded temporal arrays.
            for k in list(batch):
                if k.startswith(("v", "dv")):
                    batch[k] = np.broadcast_to(
                        batch[k][:, :1, :1].copy(), batch["x"].shape
                    )
            native = np.array(
                [lookup[rows[i]["positions"][p].upper()] for i, t, p in ex]
            )
            complete = batch["mask"].all(-1)
            target = {}
            for b in "SO":
                a, slices = attributes(batch[b.lower()][complete], "eeg")
                wide = np.zeros((*complete.shape, 10))
                wide[complete] = np.concatenate(
                    [a[:, slice(*slices[k])] for k in FIELDS], axis=1
                )
                target[b] = wide
            bg = (
                np.array(
                    [STATE["sites"][dataset, rows[i]["positions"][p]] for i, t, p in ex]
                )[:, None]
                * 4
            )
            bg = bg + np.searchsorted(
                [0.5, 1, 2], np.sqrt((batch["x"] ** 2).mean(-1)), side="right"
            )
            cases.append(
                dict(
                    subject=subject,
                    record=rows[crop["row"]]["record"],
                    fold=folds[subject],
                    crop=crop,
                    examples=ex,
                    seen=seen,
                    x=fields["x"].T.reshape(len(names), 15, 400).astype(float),
                    native=native,
                    batch=batch,
                    complete=complete,
                    target=target,
                    background=bg,
                    scale=input_scales(rows, ex, scales),
                )
            )
    observed = np.stack([c["seen"] for c in cases])
    wave_records = {a: [] for a in (*ARMS, "source")}
    examples = []
    patterns = {a: {b: [] for b in "SO"} for a in ARMS}
    support = {a: {b: [0, 0] for b in "SO"} for a in ARMS}
    peak_records = []
    for fold in range(4):
        cpu_guard()
        fold_dir = directory / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        fit = [i for i, c in enumerate(cases) if c["fold"] != fold]
        held = [i for i, c in enumerate(cases) if c["fold"] == fold]
        physical = full_source(cases, fold, dataset)
        selected = sample_rows(
            cases,
            fit,
            cfg["training"]["sample_per_fit_person"],
            cfg["training"]["seed"] + fold,
        )
        trees = {}
        direct = {}
        books = {a: {} for a in ARMS}
        reports = {}
        for j, b in enumerate("SO"):
            trees[b], cov = training_tree(cases, physical, fit, b)
            direct[b] = dict(
                order=np.arange(len(names)),
                parent=np.full(len(names), -1, np.int32),
                gain=np.zeros(len(names)),
                scale=np.sqrt(np.diag(cov).clip(1e-10)),
            )
            for arm, tree in [("direct", direct[b]), ("predictive", trees[b])]:
                cpu_guard()
                before = time.process_time()
                books[arm][b] = fit_book(
                    innovations(physical, b, tree, selected),
                    cfg["training"]["seed"] + 100 * fold + j,
                )
                reports[arm + b] = dict(
                    samples=len(selected),
                    seconds=time.process_time() - before,
                    iterations=books[arm][b]["iterations"],
                )
            reports[b + "_true_prefix_relative_innovation_energy"] = float(
                np.mean(trees[b]["scale"] ** 2 / direct[b]["scale"] ** 2)
            )
        metadata = dict(
            schema="spatial_predictive_dictionary_v1",
            dataset=dataset,
            positions=names,
            held_fold=fold,
            fit_subjects=sorted({cases[i]["subject"] for i in fit}),
            context_seconds=30,
            token_seconds=2,
            codebook_size=512,
            IDs_per_branch_native_position=15,
            bits_per_ID=9,
        )
        initial = SpatialPredictiveDictionary(metadata, trees, books["predictive"])
        first_codes = initial.encode_source(physical, observed)
        previous = initial.decode_raw_ids(first_codes)
        for j, b in enumerate("SO"):
            books["predictive"][b] = fit_book(
                innovations(physical, b, trees[b], selected, previous),
                cfg["training"]["seed"] + 100 * fold + j,
            )
        del previous, first_codes, initial
        models = {
            a: SpatialPredictiveDictionary(
                dict(metadata, arm=a), direct if a == "direct" else trees, books[a]
            )
            for a in ARMS
        }
        codes = {a: m.encode_source(physical, observed) for a, m in models.items()}
        write(
            fold_dir / "training.json",
            dict(
                books=reports,
                model_parameters={
                    a: sum(
                        sum(v.size for v in item.values() if isinstance(v, np.ndarray))
                        for item in m.books.values()
                    )
                    for a, m in models.items()
                },
                CPU_seconds=time.process_time(),
                fit_subjects=metadata["fit_subjects"],
                held_subjects=sorted({cases[i]["subject"] for i in held}),
            ),
        )
        for arm, model in models.items():
            model.save(fold_dir / (arm + ".npz"))
            np.savez_compressed(
                fold_dir / (arm + "_integers.npz"),
                indices=np.array(held),
                **{b: v[held] for b, v in codes[arm].items()},
            )
            reloaded = load_model(fold_dir / (arm + ".npz"))
            with np.load(fold_dir / (arm + "_integers.npz")) as f:
                pred = reloaded.decode_ids({b: f[b] for b in "SO"})
            expected = model.decode_ids({b: v[held] for b, v in codes[arm].items()})
            for b in expected:
                np.testing.assert_array_equal(pred[b], expected[b])
            for k, i in enumerate(held):
                c = cases[i]
                wave_records[arm].append(
                    joint_reconstruction_metrics(
                        {b: v[k, c["native"]] for b, v in pred.items()},
                        c["batch"],
                        1,
                        c["scale"],
                    )
                )
                if arm == "predictive":
                    nmse = wave_records[arm][-1]["metrics"]["nmse"].mean((1, 2))
                    for p in [int(np.argmax(nmse))]:
                        path = fold_dir / f"case_{i:03d}_hard.npz"
                        n = c["native"][p]
                        np.savez_compressed(
                            path,
                            **{q: c["batch"][q][p] for q in ("x", "s", "o", "mask")},
                            **{b: v[k, n] for b, v in pred.items()},
                        )
                        peak_records.append(
                            dict(
                                path=str(path.relative_to(directory)),
                                fold=fold,
                                case=i,
                                score=float(nmse[p]),
                                identity=tuple(map(int, c["examples"][p])),
                                subject=c["subject"],
                                position=rows[c["examples"][p, 0]]["positions"][
                                    c["examples"][p, 2]
                                ],
                                scale=float(c["scale"][p]),
                            )
                        )
            for b in "SO":
                errors, q, total = pattern(
                    cases, codes[arm], fit, held, b, 4 * len(STATE["sites"])
                )
                patterns[arm][b].append(errors)
                support[arm][b][0] += q
                support[arm][b][1] += total
        for i in held:
            c = cases[i]
            examples.extend(c["examples"])
            raw = {b: physical[b][i, c["native"]] for b in "SO"}
            raw["sum"] = raw["S"] + raw["O"]
            wave_records["source"].append(
                joint_reconstruction_metrics(raw, c["batch"], 1, c["scale"])
            )
        write(
            directory / "progress.json",
            dict(
                fold=fold + 1,
                CPU_seconds=time.process_time(),
                campaign_CPU_seconds=cpu_guard(),
            ),
        )
        del physical, codes, models, books
    ex = np.array(examples)
    summaries = {}
    for arm, records in wave_records.items():
        metric = {
            k: np.concatenate([v["metrics"][k] for v in records])
            for k in records[0]["metrics"]
        }
        masks = {
            k: np.concatenate([v["support"][k] for v in records])
            for k in records[0]["support"]
        }
        summaries[arm] = joint_reconstruction_summary(metric, masks, ex, rows)
        np.savez_compressed(
            directory / (arm + "_metrics.npz"),
            examples=ex,
            **metric,
            **{"support_" + k: v for k, v in masks.items()},
        )
    word = {}
    for arm in ARMS:
        word[arm] = {}
        for b in "SO":
            errors = np.concatenate(patterns[arm][b])
            stats = base_training_aggregate({"nmse": errors}, ex, rows)["metrics"][
                "nmse"
            ]
            m = np.array(stats["macro"])
            gains = 1 - m[1] / m[0]
            positive = np.mean(
                [
                    np.array(v["value"])[1] < np.array(v["value"])[0]
                    for v in stats["by_subject"]
                ],
                axis=0,
            )
            q, n = support[arm][b]
            word[arm][b] = dict(
                gain=gains,
                positive_subject_fraction=positive,
                support_mass=q / n,
                qualified=q,
                total=n,
                hierarchy=stats,
            )
            np.savez_compressed(
                directory / (arm + "_" + b + "_word_errors.npz"),
                examples=ex,
                error=errors,
            )
    write(directory / "word_summary.json", word)
    write(directory / "wave_summary.json", summaries)
    write(directory / "hard_cases.json", peak_records)
    return dataset, dict(
        CPU_seconds=time.process_time(), crops=len(cases), positions=len(ex)
    )


def review(args, cfg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.tokenizers.stratified_source_dictionary import pack_ids, unpack_ids

    out = args.output
    summary = json.loads((out / "summary.json").read_text())
    suffix = "" if args.review_version == "v1" else "_" + args.review_version
    if (out / ("review" + suffix + ".json")).exists():
        raise FileExistsError("retained review remains immutable")
    manifest, scales, _, _ = base_tokenizer_training_data(
        yaml.safe_load((ROOT / cfg["source"]["g0_config"]).read_text()), ROOT
    )
    rows = manifest["rows"]
    base = yaml.safe_load((ROOT / cfg["source"]["base"]).read_text())["source"]
    plan = json.loads((ROOT / base["native_train_plan"]).read_text())["plan"]
    folds = json.loads((ROOT / cfg["source"]["folds"]).read_text())
    figures = out / "waveforms"
    figures.mkdir(exist_ok=True)
    gallery = []
    transports = []
    maximum = 0.0
    scored = 0
    for dataset in sorted(plan):
        directory = out / dataset
        crop_cases = [
            (subject, crop)
            for subject, crops in sorted(plan[dataset].items())
            for crop in crops
        ]
        decoded_cache = {}
        for fold in range(4):
            fold_dir = directory / f"fold_{fold}"
            for arm in ARMS:
                file = fold_dir / (arm + "_integers.npz")
                with np.load(file) as z:
                    codes = {b: z[b] for b in "SO"}
                    indices = z["indices"].copy()
                flat = np.concatenate([codes[b].reshape(-1) for b in "SO"])
                payload = pack_ids(flat)
                binary = fold_dir / (arm + "_payload9.bin")
                if binary.exists():
                    if binary.read_bytes() != payload:
                        raise ValueError("retained payload changed")
                else:
                    binary.write_bytes(payload)
                restored = unpack_ids(binary.read_bytes(), len(flat))
                np.testing.assert_array_equal(restored, flat)
                transports.append(
                    dict(
                        dataset=dataset,
                        fold=fold,
                        arm=arm,
                        integers=len(flat),
                        packed_bytes=len(payload),
                        fixed_bits_per_ID=9,
                        padding_bits=len(payload) * 8 - 9 * len(flat),
                    )
                )
                widths = codes["S"].size
                recovered = {
                    b: restored[j * widths : (j + 1) * widths].reshape(codes[b].shape)
                    for j, b in enumerate("SO")
                }
                decoder = load_model(fold_dir / (arm + ".npz"))
                expected_people = sorted(
                    {s for s, c in crop_cases if folds[dataset][s] != fold}
                )
                if decoder.metadata["fit_subjects"] != expected_people:
                    raise ValueError("source training people changed")
                wave = decoder.decode_ids(recovered)
                for j, index in enumerate(indices):
                    decoded_cache[arm, int(index)] = {b: v[j] for b, v in wave.items()}
        with np.load(directory / "predictive_metrics.npz") as f:
            ex = f["examples"]
            errors = f["nmse"]
            support_keys = [k for k in f.files if k.startswith("support_")]
        lookup = {tuple(v): j for j, v in enumerate(ex)}
        joint = errors.mean((1, 2))
        ordered = sorted(range(len(ex)), key=lambda j: tuple(ex[j]))
        rng = np.random.default_rng(81319)
        selected = {ordered[0]: ["fixed_first"]}
        random_index = ordered[int(rng.integers(len(ordered)))]
        selected.setdefault(random_index, []).append("fixed_random_81319")
        p95 = int(np.argmin(abs(joint - np.quantile(joint, 0.95))))
        selected.setdefault(p95, []).append("nearest_candidate_P95")
        worst = int(np.argmax(joint))
        selected.setdefault(worst, []).append("candidate_worst")
        cache = OrderedDict()
        cases_by_identity = {
            (c["row"], c["start_s"]): i for i, (s, c) in enumerate(crop_cases)
        }
        names = load_model(directory / "fold_0/predictive.npz").metadata["positions"]
        native_lookup = {n.upper(): i for i, n in enumerate(names)}
        # Independent metrics over every held position from saved packed integers.
        for case_index, (subject, crop) in enumerate(crop_cases):
            examples = np.array(
                [(crop["row"], crop["start_s"], p) for p in crop["positions"]]
            )
            batch = base_tokenizer_batch(
                rows, examples, scales, array_cache=cache, cache_limit=8
            )
            native = np.array(
                [native_lookup[rows[i]["positions"][p].upper()] for i, t, p in examples]
            )
            scale = input_scales(rows, examples, scales)
            pred = {
                b: v[native] for b, v in decoded_cache["predictive", case_index].items()
            }
            actual = joint_reconstruction_metrics(pred, batch, 1, scale)["metrics"][
                "nmse"
            ]
            expected = errors[[lookup[tuple(v)] for v in examples]]
            np.testing.assert_array_equal(actual, expected)
            maximum = max(maximum, float(abs(actual - expected).max()))
            scored += len(examples)
        for j, rules in selected.items():
            row, start, position = ex[j]
            row = int(row)
            start = int(start)
            position = int(position)
            case_index = cases_by_identity[row, start]
            native = native_lookup[rows[row]["positions"][position].upper()]
            batch = base_tokenizer_batch(
                rows, ex[j : j + 1], scales, array_cache=cache, cache_limit=8
            )
            factor = float(input_scales(rows, ex[j : j + 1], scales)[0])
            unit = rows[row]["source_unit"]
            t = np.arange(6000) / 200
            wave = {
                arm: {
                    b: v[native].reshape(6000) * factor
                    for b, v in decoded_cache[arm, case_index].items()
                }
                for arm in ARMS
            }
            fig, axes = plt.subplots(3, 4, figsize=(16, 9.5))
            for r, (field, target) in enumerate([("sum", "x"), ("S", "s"), ("O", "o")]):
                y = batch[target][0].reshape(6000) * factor
                mask = batch["mask"][0].reshape(6000)
                for col, window in enumerate([(0, 30), (14, 16), (13.5, 14.5)]):
                    ax = axes[r, col]
                    take = (t >= window[0]) & (t < window[1])
                    ax.plot(t[take], y[take], lw=0.8, color="#193c50", label="target")
                    for arm, color in [
                        ("direct", "#b9612d"),
                        ("predictive", "#188c82"),
                    ]:
                        ax.plot(
                            t[take],
                            wave[arm][field][take],
                            lw=0.8,
                            color=color,
                            label=arm,
                        )
                    for boundary in np.arange(0, 31, 2):
                        if window[0] <= boundary <= window[1]:
                            ax.axvline(boundary, lw=0.4, color="gray", alpha=0.5)
                    ax.fill_between(
                        t[take],
                        0,
                        1,
                        where=~mask[take],
                        transform=ax.get_xaxis_transform(),
                        color="gray",
                        alpha=0.25,
                    )
                    ax.set(
                        xlim=window,
                        xlabel="Time in 30s crop (s)",
                        ylabel=field + " / " + unit,
                    )
                    if r == 0:
                        ax.set_title(
                            ["Full 30s", "Fixed central 2s", "Fixed token boundary"][
                                col
                            ]
                        )
                extent = np.concatenate([y, *[wave[a][field] for a in ARMS]])
                lower, upper = np.min(extent), np.max(extent)
                margin = max((upper - lower) * 0.04, 1e-10)
                for col in range(3):
                    axes[r, col].set_ylim(lower - margin, upper + margin)
                ax = axes[r, 3]
                valid = mask.reshape(120, 50).all(1)
                tt = (np.arange(120) + 0.5) * 0.25
                for value, label, color in [
                    (y, "target", "#193c50"),
                    (wave["direct"][field], "direct", "#b9612d"),
                    (wave["predictive"][field], "predictive", "#188c82"),
                ]:
                    rms = np.sqrt(np.mean(value.reshape(120, 50) ** 2, axis=1))
                    ax.plot(
                        tt,
                        np.where(valid, rms, np.nan),
                        lw=0.9,
                        label=label,
                        color=color,
                    )
                ax.set(
                    xlim=(0, 30),
                    xlabel="Time in 30s crop (s)",
                    ylabel="0.25s RMS / " + unit,
                )
                if r == 0:
                    ax.set_title("Local amplitude")
                    ax.legend(fontsize=8)
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(
                f"{dataset} | {rows[row]['subject']}/{rows[row]['record']}/{rows[row]['positions'][position]} | +{start}s\n"
                f"{'+'.join(rules)} | One K512 integer per branch/site/2s\n"
                "Frozen train coordinates; all voltage panels in each row share the same scale",
                fontsize=12,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            file = figures / f"{dataset}_{j:05d}{suffix}.png"
            fig.savefig(file, dpi=190)
            plt.close(fig)
            gallery.append(
                dict(
                    path=str(file.relative_to(out)),
                    dataset=dataset,
                    subject=rows[row]["subject"],
                    record=rows[row]["record"],
                    position=rows[row]["positions"][position],
                    triple=[row, start, position],
                    selection=rules,
                    NMSE=errors[j].tolist(),
                    actual_image_open=False,
                )
            )
    assert scored == 17040
    (out / ("waveform_replay" + suffix + ".json")).write_text(
        json.dumps(
            dict(
                status="exact",
                positions=scored,
                maximum_original_NMSE_difference=maximum,
                packed_transport=transports,
                integer_sum_per_arm=sum(
                    r["integers"] for r in transports if r["arm"] == "direct"
                ),
                decode_inputs="packed integers + frozen model and fixed native montage only",
            ),
            indent=2,
        )
        + "\n"
    )
    write(
        out / ("figure_manifest" + suffix + ".json"),
        dict(figures=gallery, all_untrimmed_cases=True),
    )
    reference = json.loads(
        (
            ROOT
            / "experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261123_joint_baseline_v1/component_source_transfer/integer_sum_source_v1/summary.json"
        ).read_text()
    )["arms"]["high_transient_source"]["metrics"]
    gates = {}
    for arm in ARMS:
        m = summary["waveform"][arm]["metrics"]
        nmse = np.array(m["nmse"]["macro"])
        ref = np.array(reference["nmse"]["macro"])
        rms = np.array(m["rms_gain"]["macro"])
        gates[arm] = dict(
            X_NMSE=bool(np.all(nmse[0] <= 1.05 * ref[0])),
            SO_NMSE=bool(np.all(nmse[1:] <= 1.1 * ref[1:])),
            datasets=all(
                np.all(
                    np.array(v) <= 1.1 * np.array(reference["nmse"]["by_dataset"][d])
                )
                for d, v in m["nmse"]["by_dataset"].items()
            ),
            RMS=bool(np.all((rms >= 0.75) & (rms <= 1.25))),
            theta_alpha=all(
                np.all(np.array(m[k]["macro"]) <= 3.0103)
                for k in ("band_4_8_abs_db", "band_8_13_abs_db")
            ),
            envelope=bool(np.all(np.array(m["local_rms_correlation"]["macro"]) >= 0.5)),
            pattern=all(summary["word"][arm][b]["passed"] for b in "SO"),
        )
    write(
        out / ("review" + suffix + ".json"),
        dict(
            status="replayed_and_rendered_actual_view_pending",
            gates=gates,
            eligible={a: all(v.values()) for a, v in gates.items()},
            CPU_seconds=time.process_time(),
            development_reads=0,
        ),
    )


def pca_diagnostic(args, cfg):
    out = args.output
    manifest, scales, _, _ = base_tokenizer_training_data(
        yaml.safe_load((ROOT / cfg["source"]["g0_config"]).read_text()), ROOT
    )
    rows = manifest["rows"]
    base = yaml.safe_load((ROOT / cfg["source"]["base"]).read_text())["source"]
    plan = json.loads((ROOT / base["native_train_plan"]).read_text())["plan"]
    folds = json.loads((ROOT / cfg["source"]["folds"]).read_text())
    records = {a: [] for a in ARMS}
    examples = []
    cache = OrderedDict()
    models = {}
    sources = {}
    for dataset, people in sorted(plan.items()):
        for fold in range(4):
            sources[dataset, fold] = load_source(
                ROOT / cfg["source"]["gain_run"] / f"{dataset}_fold{fold}_transient.npz"
            )
            for arm in ARMS:
                models[dataset, fold, arm] = load_model(
                    out / dataset / f"fold_{fold}" / (arm + ".npz")
                )
        names = models[dataset, 0, "direct"].metadata["positions"]
        lookup = {n.upper(): i for i, n in enumerate(names)}
        for subject, crops in sorted(people.items()):
            fold = folds[dataset][subject]
            if subject in sources[dataset, fold].metadata["fit_subjects"]:
                raise ValueError("held source was fitted on this person")
            for crop in crops:
                fields, seen, _ = base_tokenizer_joint_eeg(
                    rows,
                    dict(crop, positions=[]),
                    scales,
                    names,
                    cache,
                    allowed_splits=("train",),
                )
                x = fields["x"].T.reshape(1, len(names), 15, 400)
                p = sources[dataset, fold].predict_continuous(x, dataset, seen[None])
                physical = {b: band_limit(p[b]).astype(np.float32) for b in "SO"}
                ex = np.array(
                    [(crop["row"], crop["start_s"], pos) for pos in crop["positions"]]
                )
                examples.extend(ex)
                batch = base_tokenizer_batch(
                    rows, ex, scales, array_cache=cache, cache_limit=8
                )
                native = np.array(
                    [lookup[rows[i]["positions"][pos].upper()] for i, t, pos in ex]
                )
                scale = input_scales(rows, ex, scales)
                for arm in ARMS:
                    prediction = models[dataset, fold, arm].project_source(
                        physical, seen[None]
                    )
                    records[arm].append(
                        joint_reconstruction_metrics(
                            {b: v[0, native] for b, v in prediction.items()},
                            batch,
                            1,
                            scale,
                        )
                    )
    result = {}
    for arm, items in records.items():
        metric = {
            k: np.concatenate([item["metrics"][k] for item in items])
            for k in items[0]["metrics"]
        }
        support = {
            k: np.concatenate([item["support"][k] for item in items])
            for k in items[0]["support"]
        }
        result[arm] = joint_reconstruction_summary(
            metric, support, np.array(examples), rows
        )
        np.savez_compressed(
            out / (arm + "_continuous_PCA64_metrics.npz"),
            examples=np.array(examples),
            **metric,
        )
    write(
        out / "matched_continuous_diagnostic.json",
        dict(
            status="completed",
            arms=result,
            positions=len(examples),
            continuous_per_patch_coordinates=64,
            scope="post-training matched projection diagnostic, no discrete-model claim or refitting",
            CPU_seconds=time.process_time(),
            validation_reads=0,
            test_reads=0,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--pca-diagnostic", action="store_true")
    parser.add_argument("--review-version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if (
        cfg["schema"] != "spatial_predictive_dictionary_experiment_v1"
        or cfg["data"]["allowed_splits"] != ["train"]
        or cfg["model"]["vocabulary"] != 512
        or cfg["model"]["context_seconds"] != 30
        or cfg["model"]["token_seconds"] != 2
        or cfg["model"]["EEG_rate_hz"] != 200
        or cfg["training"]["maximum_book_samples"] != 8192
        or cfg["training"]["kmeans_max_iter"] != 16
        or cfg["training"]["kmeans_batch"] != 512
        or any(
            cfg["data"][s] != "forbidden"
            for s in ["validation_access", "confirmation_access", "test_access"]
        )
    ):
        raise ValueError("train-only original population")
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    if args.review:
        return review(args, cfg)
    if args.pca_diagnostic:
        return pca_diagnostic(args, cfg)
    with (out / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "summary.json").exists():
            raise FileExistsError("completed run retained")
        manifest, scales, _, _ = base_tokenizer_training_data(
            yaml.safe_load((ROOT / cfg["source"]["g0_config"]).read_text()), ROOT
        )
        base = yaml.safe_load((ROOT / cfg["source"]["base"]).read_text())["source"]
        plan = json.loads((ROOT / base["native_train_plan"]).read_text())["plan"]
        folds = json.loads((ROOT / cfg["source"]["folds"]).read_text())
        rows = manifest["rows"]
        assert sum(len(c) for p in plan.values() for c in p.values()) == 400
        assert (
            sum(
                len(c["positions"])
                for p in plan.values()
                for group in p.values()
                for c in group
            )
            == 17040
        )
        assert all(
            rows[c["row"]]["split"] == "train"
            for p in plan.values()
            for group in p.values()
            for c in group
        )
        sites = sorted(
            {
                (d, rows[c["row"]]["positions"][p])
                for d, people in plan.items()
                for group in people.values()
                for c in group
                for p in c["positions"]
            }
        )
        STATE.update(
            cfg=cfg,
            out=out,
            base=base,
            plan=plan,
            folds=folds,
            rows=rows,
            scales=scales,
            sites={k: i for i, k in enumerate(sites)},
        )
        write(
            out / "resolved_config.json",
            dict(
                config=cfg,
                source_input_identity=manifest["cfg_identity"],
                scale_identity=manifest["scales_identity"],
            ),
        )
        with threadpool_limits(1), ProcessPoolExecutor(
            max_workers=4, mp_context=multiprocessing.get_context("fork")
        ) as pool:
            result = dict(pool.map(dataset_worker, sorted(plan)))
        aggregate = {}
        word_summary = {}
        for arm in (*ARMS, "source"):
            pieces = []
            supports = []
            examples = []
            for dataset in sorted(plan):
                with np.load(out / dataset / (arm + "_metrics.npz")) as f:
                    examples.append(f["examples"])
                    pieces.append(
                        {
                            k: f[k]
                            for k in f.files
                            if k != "examples" and not k.startswith("support_")
                        }
                    )
                    supports.append(
                        {k[8:]: f[k] for k in f.files if k.startswith("support_")}
                    )
            metrics = {k: np.concatenate([v[k] for v in pieces]) for k in pieces[0]}
            sup = {k: np.concatenate([v[k] for v in supports]) for k in supports[0]}
            aggregate[arm] = joint_reconstruction_summary(
                metrics, sup, np.concatenate(examples), rows
            )
            np.savez_compressed(
                out / (arm + "_all_metrics.npz"),
                examples=np.concatenate(examples),
                **metrics,
            )
        for arm in ARMS:
            word_summary[arm] = {}
            for b in "SO":
                pieces = []
                examples = []
                q = 0
                n = 0
                for dataset in sorted(plan):
                    with np.load(
                        out / dataset / (arm + "_" + b + "_word_errors.npz")
                    ) as f:
                        pieces.append(f["error"])
                        examples.append(f["examples"])
                    v = json.loads((out / dataset / "word_summary.json").read_text())[
                        arm
                    ][b]
                    q += v["qualified"]
                    n += v["total"]
                scores = base_training_aggregate(
                    {"nmse": np.concatenate(pieces)}, np.concatenate(examples), rows
                )["metrics"]["nmse"]
                m = np.array(scores["macro"])
                gain = 1 - m[1] / m[0]
                positive = np.mean(
                    [
                        np.array(v["value"])[1] < np.array(v["value"])[0]
                        for v in scores["by_subject"]
                    ],
                    axis=0,
                )
                word_summary[arm][b] = dict(
                    gain=gain,
                    positive_people_fraction=positive,
                    support=q / n,
                    hierarchy=scores,
                    passed=bool(
                        min(gain[:2]) > 0
                        and max(gain) >= 0.05
                        and min(positive[:2]) >= 0.5
                        and q / n >= 0.9
                    ),
                )
        write(
            out / "summary.json",
            dict(
                status="numerical_completed_requires_actual_plot_review",
                workers=result,
                waveform=aggregate,
                word=word_summary,
                validation_reads=0,
                test_reads=0,
                confirmation_reads=0,
                integer_replay="every held array reload exact before metrics",
            ),
        )


if __name__ == "__main__":
    main()
