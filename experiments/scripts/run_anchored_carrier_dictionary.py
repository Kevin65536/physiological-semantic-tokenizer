"""Fit single-ID virtual EEG carrier books after the continuous train-subject review."""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.dual_component_tokenizer import (
    base_training_aggregate,
    joint_reconstruction_metrics,
    joint_reconstruction_summary,
)
from src.analysis.tokenizer_pattern_audit import (
    SparsePatternAudit,
    attributes,
    feature_means,
)
from src.data.dual_component_tokenizer import (
    base_tokenizer_training_data,
    base_tokenizer_joint_eeg,
    base_tokenizer_batch,
)
from src.tokenizers.anchored_spatial_carrier import AnchoredSpatialCarrier
from src.tokenizers.anchored_carrier_dictionary import (
    AnchoredCarrierDictionary,
    fit_book,
)
from evaluate_joint_tokenizer import input_scales, write

FIELDS = {"power_4_8": (0, 1), "power_8_13": (1, 2), "rms_envelope": (2, 10)}


def rank(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def budget(cfg):
    if time.process_time() > cfg["resources"]["CPU_budget_seconds"] - 30:
        raise RuntimeError("predeclared virtual-carrier CPU budget reached")


def make_cases(source, rows, scales, plan, folds, site_index, cfg, out, label):
    cases, cache = [], OrderedDict()
    for dataset, subjects in sorted(plan.items()):
        names = source.metadata["positions"][dataset]
        lookup = {name.upper(): i for i, name in enumerate(names)}
        roles = source.roles(dataset)
        for subject, crops in sorted(subjects.items()):
            for crop in crops:
                budget(cfg)
                index, start = crop["row"], crop["start_s"]
                fields, observed, _ = base_tokenizer_joint_eeg(
                    rows, crop, scales, names, cache, allowed_splits=("train",)
                )
                x = fields["x"].T.reshape(1, len(names), 15, 400)
                physical = source.continuous_source(x)
                carriers = source.analyze(physical, dataset, observed[None])
                ex = np.array([(index, start, p) for p in crop["positions"]], np.int32)
                batch = base_tokenizer_batch(
                    rows, ex, scales, array_cache=cache, cache_limit=16
                )
                native = np.array(
                    [
                        lookup[rows[index]["positions"][p].upper()]
                        for p in crop["positions"]
                    ]
                )
                complete = batch["mask"].all(-1)
                target = {}
                for branch in "SO":
                    raw, slices = attributes(batch[branch.lower()][complete], "eeg")
                    full = np.zeros((*complete.shape, 10))
                    full[complete] = np.concatenate(
                        [raw[:, slice(*slices[k])] for k in FIELDS], axis=1
                    )
                    target[branch] = full
                background_site = np.array(
                    [
                        site_index[dataset, rows[index]["positions"][p]]
                        for p in crop["positions"]
                    ]
                )
                background = 4 * background_site[:, None] + np.searchsorted(
                    [0.5, 1.0, 2.0],
                    np.sqrt(np.mean(batch["x"] ** 2, axis=-1)),
                    side="right",
                )
                cases.append(
                    dict(
                        dataset=dataset,
                        subject=subject,
                        fold=folds[dataset][subject],
                        crop=crop,
                        physical={b: physical[b] for b in "SO"},
                        carriers={b: carriers[b] for b in "SO"},
                        observed=observed[None],
                        examples=ex,
                        native=native,
                        complete=complete,
                        target=target,
                        background=background,
                        batch=batch,
                        scales=input_scales(rows, ex, scales),
                        roles=roles,
                    )
                )
                write(
                    out / "progress.json",
                    dict(
                        stage="continuous_training_carriers",
                        source=label,
                        physical_crops=len(cases),
                        planned=sum(len(v) for g in plan.values() for v in g.values()),
                        CPU_seconds=time.process_time(),
                        validation_reads=0,
                    ),
                )
    return cases


def books_for(cases, selected, cfg, fraction):
    positive = cfg["model"]["emitted_positive_atoms"]
    cap = cfg["training"]["book_fit_max_patches_per_subject_branch_band"]
    rng = np.random.default_rng(cfg["training"]["seed"])
    result = {}
    for branch in "SO":
        for band in range(5):
            budget(cfg)
            grouped = defaultdict(list)
            for i in selected:
                case = cases[i]
                positions = np.flatnonzero(case["roles"][case["native"]] == band)
                sample, patch = np.nonzero(case["complete"][positions])
                for s, t in zip(positions[sample], patch):
                    grouped[case["dataset"], case["subject"]].append(
                        (i, int(case["native"][s]), int(t))
                    )
            picked = []
            for key, candidates in sorted(grouped.items()):
                take = rng.choice(
                    len(candidates), min(cap, len(candidates)), replace=False
                )
                picked.extend(candidates[j] for j in take)
            carrier = np.stack(
                [
                    cases[i]["carriers"][branch][0, site, patch]
                    for i, site, patch in picked
                ]
            )
            physical = np.stack(
                [
                    cases[i]["physical"][branch][0, site, patch]
                    for i, site, patch in picked
                ]
            )
            result[branch, band] = fit_book(
                carrier,
                physical,
                band,
                positive_atoms=positive,
                semantic_fraction=fraction,
                seed=cfg["training"]["seed"] + band,
                max_iter=cfg["training"]["kmeans_max_iter"],
                batch_size=cfg["training"]["kmeans_batch_size"],
            )
    return result


def all_ids(model, cases):
    return [
        model.assign(c["carriers"], c["physical"], c["dataset"], c["observed"])
        for c in cases
    ]


def pattern_fold(model, cases, codes, fit_indices, held_indices, backgrounds, datasets):
    audits = {
        b: SparsePatternAudit(
            5 * model.codebook_size, backgrounds, datasets, 10, FIELDS
        )
        for b in "SO"
    }
    subject_support = {b: defaultdict(set) for b in "SO"}
    for i in fit_indices:
        case = cases[i]
        keep = case["complete"]
        band = case["roles"][case["native"]][:, None]
        for branch in "SO":
            ids = (codes[i][branch][0, case["native"]] + band * model.codebook_size)[
                keep
            ]
            audits[branch].fit(
                ids, case["background"][keep], case["target"][branch][keep]
            )
            for code in np.unique(ids):
                subject_support[branch][int(code)].add(
                    (case["dataset"], case["subject"])
                )
    records = {b: [] for b in "SO"}
    supported = {b: [0, 0] for b in "SO"}
    examples = []
    for i in held_indices:
        case = cases[i]
        keep = case["complete"]
        band = case["roles"][case["native"]][:, None]
        examples.extend(case["examples"])
        for branch in "SO":
            ids = (codes[i][branch][0, case["native"]] + band * model.codebook_size)[
                keep
            ]
            background, _, conditional = audits[branch].read(
                ids, case["background"][keep]
            )
            target = case["target"][branch][keep]
            error = np.full((*keep.shape, 2, 3), np.nan)
            error[keep] = np.stack(
                [
                    feature_means((v - target) ** 2, FIELDS)
                    for v in [background, conditional]
                ],
                axis=1,
            )
            count = keep.sum(1)
            mean = np.divide(
                np.nansum(error, axis=1),
                count[:, None, None],
                out=np.full((len(keep), 2, 3), np.nan),
                where=count[:, None, None] > 0,
            )
            records[branch].append(mean)
            ok = (audits[branch].local.count[ids] >= 8) & np.array(
                [len(subject_support[branch][int(code)]) >= 2 for code in ids]
            )
            supported[branch][0] += int(ok.sum())
            supported[branch][1] += len(ids)
    return (
        np.asarray(examples),
        {b: np.concatenate(v) for b, v in records.items()},
        supported,
    )


def waveform_fold(model, cases, codes, held_indices, rows, cfg):
    chosen = {}
    for i in held_indices:
        case = cases[i]
        unit = case["dataset"], case["subject"]
        if unit not in chosen or rank(case["crop"]) < rank(cases[chosen[unit]]["crop"]):
            chosen[unit] = i
    actual, continuous, examples = [], [], []
    for i in chosen.values():
        case = cases[i]
        count = len(case["examples"])
        picked = sorted(
            range(count),
            key=lambda k: rank(
                [cfg["training"]["seed"], *case["examples"][k].tolist()]
            ),
        )[:4]
        batch = {k: v[picked] for k, v in case["batch"].items()}
        native = case["native"][picked]
        factor = case["scales"][picked]
        hard = model.decode_ids(codes[i], case["dataset"])
        soft = model.source.synthesize(case["carriers"], case["dataset"])
        for pred, record in [(hard, actual), (soft, continuous)]:
            record.append(
                joint_reconstruction_metrics(
                    {k: v[0, native] for k, v in pred.items()}, batch, 1, factor
                )
            )
        examples.extend(case["examples"][picked])

    def aggregate(records):
        metrics = {
            k: np.concatenate([r["metrics"][k] for r in records])
            for k in records[0]["metrics"]
        }
        support = {
            k: np.concatenate([r["support"][k] for r in records])
            for k in records[0]["support"]
        }
        summary = joint_reconstruction_summary(metrics, support, examples, rows)
        return dict(summary=summary, metrics=metrics, support=support)

    return np.asarray(examples), aggregate(actual), aggregate(continuous)


def pattern_summary(errors, examples, rows, supported):
    out = {}
    for branch in "SO":
        stats = base_training_aggregate({"nmse": errors[branch]}, examples, rows)[
            "metrics"
        ]["nmse"]
        values = np.asarray(stats["macro"])
        gains = 1 - values[1] / values[0]
        positive = np.mean(
            [
                1 - np.asarray(v["value"])[1] / np.asarray(v["value"])[0] > 0
                for v in stats["by_subject"]
            ],
            axis=0,
        )
        out[branch] = dict(
            gain=dict(zip(FIELDS, gains.tolist())),
            positive_subject_fraction=dict(zip(FIELDS, positive.tolist())),
            support_mass=supported[branch][0] / supported[branch][1],
            hierarchy=stats,
        )
    return out


def run(args, cfg):
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    lock = (out / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "summary.json").exists():
        raise FileExistsError("retained completed anchored dictionary run")
    limit = cfg["resources"]["CPU_budget_seconds"]
    resource.setrlimit(resource.RLIMIT_CPU, (limit - 15, limit))
    basis_run = ROOT / cfg["source"]["basis_run"]
    review = json.loads((basis_run / "summary.json").read_text())
    if not review["admitted_to_hard_training"]:
        write(
            out / "summary.json",
            dict(
                status="continuous_basis_not_admitted",
                measured_dictionary_fits=0,
                source_review=review,
            ),
        )
        return
    base = yaml.safe_load((ROOT / cfg["source"]["g0_config"]).read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(base, ROOT)
    rows = manifest["rows"]
    basis_cfg = yaml.safe_load((ROOT / cfg["source"]["basis_config"]).read_text())
    plan = json.loads((ROOT / basis_cfg["source"]["native_train_plan"]).read_text())[
        "plan"
    ]
    folds = json.loads((basis_run / "fold_subjects.json").read_text())
    for dataset, subjects in plan.items():
        for subject, crops in subjects.items():
            if any(
                rows[c["row"]]["split"] != "train"
                or rows[c["row"]]["dataset"] != dataset
                or rows[c["row"]]["subject"] != subject
                for c in crops
            ):
                raise ValueError("source training plan violates subject/split identity")
    sites = sorted(
        {
            (r["dataset"], p)
            for r in rows
            if r["split"] == "train" and r["modality"] == "eeg"
            for p in r["positions"]
        }
    )
    site_index = {key: i for i, key in enumerate(sites)}
    write(out / "resolved_config.json", cfg)
    if (
        cfg["data"]["test_access"] != "forbidden"
        or cfg["data"]["protected_data"] != "forbidden"
        or cfg["data"]["confirmation_access"] != "forbidden"
    ):
        raise ValueError("train and fixed development boundary required")
    aggregate = {
        fraction: dict(
            pattern={b: [] for b in "SO"},
            examples=[],
            support={b: [0, 0] for b in "SO"},
            wave=[],
            continuous=[],
            wave_examples=[],
        )
        for fraction in cfg["model"]["semantic_distance_fractions"]
    }
    for fold in range(4):
        source_path = basis_run / f"fold_model_{fold}.npz"
        source = AnchoredSpatialCarrier.load(source_path)
        if (
            source.metadata["input_identity"] != manifest["cfg_identity"]
            or source.metadata["scales_identity"] != manifest["scales_identity"]
        ):
            raise ValueError("frozen source and G0 coordinate identity mismatch")
        cases = make_cases(
            source, rows, scales, plan, folds, site_index, cfg, out, f"fold_{fold}"
        )
        fit = [i for i, c in enumerate(cases) if c["fold"] != fold]
        held = [i for i, c in enumerate(cases) if c["fold"] == fold]
        for fraction in aggregate:
            budget(cfg)
            books = books_for(cases, fit, cfg, fraction)
            metadata = dict(
                schema="anchored_carrier_dictionary_v1",
                codebook_size=cfg["model"]["vocabulary_including_missing"],
                semantic_fraction=fraction,
                seed=cfg["training"]["seed"],
                input_identity=manifest["cfg_identity"],
                scales_identity=manifest["scales_identity"],
                held_fold=fold,
                fitting_split="train",
                continuous_decoder_side_information=0,
                integers_per_branch_position_context=15,
            )
            model = AnchoredCarrierDictionary(source, books, metadata)
            key = f"semantic{fraction:g}_fold{fold}"
            model.save(out / (key + ".npz"), source_path)
            ids = all_ids(model, cases)
            ex, err, support = pattern_fold(
                model, cases, ids, fit, held, 4 * len(sites), sorted(plan)
            )
            wave_examples, wave, soft = waveform_fold(
                model, cases, ids, held, rows, cfg
            )
            rec = aggregate[fraction]
            rec["examples"].append(ex)
            for b in "SO":
                rec["pattern"][b].append(err[b])
                rec["support"][b] = [
                    x + y for x, y in zip(rec["support"][b], support[b])
                ]
            rec["wave_examples"].append(wave_examples)
            rec["wave"].append(wave)
            rec["continuous"].append(soft)
            write(
                out / "progress.json",
                dict(
                    stage="dictionary_crossfit",
                    completed_fold=fold + 1,
                    semantic_fraction=fraction,
                    CPU_seconds=time.process_time(),
                    validation_reads=0,
                ),
            )
    # All error arrays use the original per-position hierarchy; folds are not independent subjects.
    candidates = []
    for fraction, rec in aggregate.items():
        examples = np.concatenate(rec["examples"])
        errors = {b: np.concatenate(rec["pattern"][b]) for b in "SO"}
        pattern = pattern_summary(errors, examples, rows, rec["support"])
        wave_examples = np.concatenate(rec["wave_examples"])

        def pooled(records):
            metrics = {
                k: np.concatenate([r["metrics"][k] for r in records])
                for k in records[0]["metrics"]
            }
            support = {
                k: np.concatenate([r["support"][k] for r in records])
                for k in records[0]["support"]
            }
            return joint_reconstruction_summary(
                metrics, support, wave_examples.tolist(), rows
            )

        wave_summary, continuous_summary = (
            pooled(rec["wave"]),
            pooled(rec["continuous"]),
        )
        waveform = np.asarray(wave_summary["metrics"]["nmse"]["macro"])
        continuous = np.asarray(continuous_summary["metrics"]["nmse"]["macro"])
        selection = cfg["selection"]
        ppass = all(
            v["support_mass"] >= selection["support_mass_min"]
            and min(v["gain"][k] for k in ("power_4_8", "power_8_13")) > 0
            and max(v["gain"].values()) >= selection["one_gain_min"]
            and min(
                v["positive_subject_fraction"][k] for k in ("power_4_8", "power_8_13")
            )
            >= 0.5
            for v in pattern.values()
        )
        wpass = bool(
            np.all(
                waveform
                <= selection["waveform_component_max_ratio_to_matched_continuous"]
                * continuous
            )
        )
        wpass &= all(
            np.all(
                np.asarray(value)
                <= selection[
                    "waveform_dataset_component_max_ratio_to_matched_continuous"
                ]
                * np.asarray(
                    continuous_summary["metrics"]["nmse"]["by_dataset"][dataset]
                )
            )
            for dataset, value in wave_summary["metrics"]["nmse"]["by_dataset"].items()
        )
        candidates.append(
            dict(
                semantic_fraction=fraction,
                pattern=pattern,
                waveform_nmse=waveform,
                waveform_summary=wave_summary,
                continuous_nmse=continuous,
                continuous_summary=continuous_summary,
                pattern_pass=ppass,
                waveform_pass=wpass,
                eligible=ppass and wpass,
                fold_waveforms=[v["summary"] for v in rec["wave"]],
                fold_continuous=[v["summary"] for v in rec["continuous"]],
            )
        )
        np.savez_compressed(
            out / f"semantic{fraction:g}_attribute_errors.npz",
            examples=examples,
            **errors,
        )
    eligible = [v for v in candidates if v["eligible"]]
    selected = (
        min(
            eligible,
            key=lambda v: (
                np.asarray(v["waveform_nmse"]).mean(),
                v["semantic_fraction"],
            ),
        )
        if eligible
        else None
    )
    write(
        out / "training_crossfit.json",
        dict(
            status="completed",
            candidates=candidates,
            selected=selected,
            validation_reads=0,
            frozen_Wiener_exposure="all permitted train subjects; carrier basis, books and categorical readouts cross-fitted by subject",
        ),
    )
    write(
        out / "summary.json",
        dict(
            status="train_crossfit_passed" if selected else "train_crossfit_rejected",
            selected_fraction=selected["semantic_fraction"] if selected else None,
            CPU_seconds=time.process_time(),
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
        ),
    )


def render_waveforms(plots, folder):
    import matplotlib.pyplot as plt

    for dataset in sorted({v["dataset"] for v in plots}):
        items = [v for v in plots if v["dataset"] == dataset]
        base = items[0]
        t = np.arange(6000) / 200
        mask = base["batch"]["mask"].reshape(-1)
        fig, axes = plt.subplots(
            3, 3, figsize=(13, 8), sharey="row", layout="constrained"
        )
        for row, (field, b) in enumerate([("x", "sum"), ("s", "S"), ("o", "O")]):
            target = base["batch"][field].reshape(-1) * base["scale"]
            for col, limits in enumerate([(0, 30), (14, 16), (13.5, 14.5)]):
                axis = axes[row, col]
                axis.plot(
                    t,
                    np.where(mask, target, np.nan),
                    color="black",
                    lw=1,
                    label="Target",
                )
                axis.plot(
                    t,
                    base["continuous"][b].reshape(-1) * base["scale"],
                    color="#777777",
                    lw=1,
                    label="Continuous carrier",
                    alpha=0.8,
                )
                for item, color in zip(items, ["#236f95", "#c76029"]):
                    y = item["predicted"][b].reshape(-1) * item["scale"]
                    axis.plot(
                        t,
                        y,
                        color=color,
                        lw=1,
                        label=f"Integer semantic={item['fraction']:g}",
                        alpha=0.9,
                    )
                for edge in np.arange(0, 30, 2):
                    axis.axvline(edge, color="#d7dfe3", lw=0.4, zorder=0)
                edges = np.diff(np.r_[False, ~mask, False].astype(np.int8))
                for first, last in zip(
                    np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
                ):
                    axis.axvspan(
                        first / 200, last / 200, color="#b9bfc6", alpha=0.35, zorder=0
                    )
                axis.set_xlim(*limits)
                axis.set_xlabel("Crop time (s)")
                if col == 0:
                    axis.set_ylabel(field.upper() + " (" + base["unit"] + ")")
                if row == 0:
                    axis.set_title(
                        ["Full 30 s", "Fixed 2 s patch", "Fixed token boundary"][col]
                    )
            axes[row, 0].legend(fontsize=7, loc="upper left", ncol=2)
        fig.suptitle(
            f"Held train {dataset} / {base['subject']} / {base['position']} / crop +{base['crop']['start_s']}s"
        )
        path = folder / (dataset + "_waveforms.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)


def review(args, cfg):
    from src.tokenizers.anchored_carrier_dictionary import load_model
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.output
    if (out / "integer_review_v1.json").exists():
        raise FileExistsError("retained completed carrier dictionary review")
    cv = json.loads((out / "training_crossfit.json").read_text())
    basis_run = ROOT / cfg["source"]["basis_run"]
    basis_cfg = yaml.safe_load((ROOT / cfg["source"]["basis_config"]).read_text())
    plan = json.loads((ROOT / basis_cfg["source"]["native_train_plan"]).read_text())[
        "plan"
    ]
    folds = json.loads((basis_run / "fold_subjects.json").read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(
        yaml.safe_load((ROOT / cfg["source"]["g0_config"]).read_text()), ROOT
    )
    rows = manifest["rows"]
    cache, checks, plots = OrderedDict(), [], []
    folder = out / "review_v1"
    folder.mkdir(exist_ok=True)
    fixed = {}
    for fold in range(4):
        for dataset, subjects in sorted(plan.items()):
            subject = next(s for s in sorted(subjects) if folds[dataset][s] == fold)
            crop = min(subjects[subject], key=rank)
            fixed[fold, dataset] = subject, crop
    for fold in range(4):
        for fraction in cfg["model"]["semantic_distance_fractions"]:
            checkpoint = out / f"semantic{fraction:g}_fold{fold}.npz"
            model = load_model(checkpoint)
            # A second full frozen model load is used only by the integer decoder.
            replay = load_model(checkpoint)
            replay.source.continuous_source = lambda *_: (_ for _ in ()).throw(
                AssertionError("integer replay requested X")
            )
            replay.source.analyze = lambda *_: (_ for _ in ()).throw(
                AssertionError("integer replay requested continuous carriers")
            )
            archive = {}
            for dataset in sorted(plan):
                subject, crop = fixed[fold, dataset]
                if (
                    rows[crop["row"]]["split"] != "train"
                    or rows[crop["row"]]["subject"] != subject
                ):
                    raise ValueError(
                        "registered held training subject required before arrays"
                    )
                names = model.metadata["positions"][dataset]
                fields, observed, _ = base_tokenizer_joint_eeg(
                    rows, crop, scales, names, cache, allowed_splits=("train",)
                )
                x = fields["x"].T.reshape(1, len(names), 15, 400)
                codes = model.encode(x, dataset, observed[None])
                predicted = model.decode_ids(codes, dataset)
                for branch in "SO":
                    archive[dataset + "_" + branch] = codes[branch]
                code_path = (
                    folder / f"codes_fraction{fraction:g}_fold{fold}_{dataset}.npz"
                )
                np.savez_compressed(code_path, **codes)
                with np.load(code_path) as f:
                    disk = {b: f[b].copy() for b in "SO"}
                recovered = replay.decode_ids(disk, dataset)
                maximum = 0.0
                for b in ("S", "O", "sum"):
                    np.testing.assert_array_equal(predicted[b], recovered[b])
                    maximum = max(
                        maximum, float(np.abs(predicted[b] - recovered[b]).max())
                    )
                checks.append(
                    dict(
                        fold=fold,
                        semantic_fraction=fraction,
                        dataset=dataset,
                        subject=subject,
                        crop=crop,
                        integer_count=sum(v.size for v in disk.values()),
                        max_abs=maximum,
                        IDs=code_path.name,
                        decoder_X_or_continuous_calls=0,
                    )
                )
                if fold == 0:
                    p = min(
                        crop["positions"],
                        key=lambda p: rank(
                            [cfg["training"]["seed"], crop["row"], crop["start_s"], p]
                        ),
                    )
                    native = next(
                        i
                        for i, n in enumerate(names)
                        if n.upper() == rows[crop["row"]]["positions"][p].upper()
                    )
                    batch = base_tokenizer_batch(
                        rows,
                        [(crop["row"], crop["start_s"], p)],
                        scales,
                        array_cache=cache,
                        cache_limit=16,
                    )
                    factor = input_scales(
                        rows, [(crop["row"], crop["start_s"], p)], scales
                    )[0]
                    continuous = model.predict_continuous(x, dataset, observed[None])
                    plots.append(
                        dict(
                            dataset=dataset,
                            subject=subject,
                            crop=crop,
                            position=rows[crop["row"]]["positions"][p],
                            unit=rows[crop["row"]]["source_unit"],
                            fraction=fraction,
                            batch={k: batch[k][0] for k in ("x", "s", "o", "mask")},
                            scale=factor,
                            continuous={
                                b: continuous[b][0, native] for b in ("S", "O", "sum")
                            },
                            predicted={
                                b: predicted[b][0, native] for b in ("S", "O", "sum")
                            },
                        )
                    )
    render_waveforms(plots, folder)
    candidates = cv["candidates"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    names = [f"semantic {v['semantic_fraction']:g}" for v in candidates]
    x = np.arange(len(names))
    for index, branch in enumerate("SO"):
        for metric, color in zip(FIELDS, ["#276784", "#bc6429", "#568367"]):
            axes[0, index].plot(
                x,
                [v["pattern"][branch]["gain"][metric] for v in candidates],
                marker="o",
                label=metric,
                color=color,
            )
        axes[0, index].axhline(0, color="#333333", lw=0.7)
        axes[0, index].set_title(
            branch + " local information on held training subjects"
        )
        axes[0, index].set_xticks(x, names)
        axes[0, index].set_ylabel("1 - MSE(background + ID) / MSE(background)")
        axes[0, index].legend(fontsize=8)
    for i, b in enumerate(["X", "S", "O"]):
        axes[1, 0].plot(
            x,
            [np.asarray(v["waveform_nmse"])[i, 0] for v in candidates],
            marker="o",
            label=b,
        )
        axes[1, 0].axhline(
            np.asarray(candidates[0]["continuous_nmse"])[i, 0],
            lw=0.7,
            ls="--",
            alpha=0.6,
        )
    axes[1, 0].set_xticks(x, names)
    axes[1, 0].set_ylabel("G0 target-variance NMSE")
    axes[1, 0].legend()
    for b in "SO":
        axes[1, 1].plot(
            x,
            [v["pattern"][b]["support_mass"] for v in candidates],
            marker="o",
            label=b,
        )
    axes[1, 1].axhline(0.9, color="black", ls="--", lw=0.8)
    axes[1, 1].set_xticks(x, names)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_ylabel("Held patch mass with 8 train patches and 2 subjects")
    axes[1, 1].legend()
    fig.savefig(folder / "training_comparison.png", dpi=220)
    plt.close(fig)
    write(
        out / "integer_review_v1.json",
        dict(
            status="passed",
            scope="all eight frozen fold/semantic models on one registered held-train crop per dataset; saved integers only",
            checks=checks,
            source_window_values_in_decoder=0,
            continuous_path_calls_in_decoder=0,
            validation_reads=0,
            observed_window_examples="review_v1/*_waveforms.png",
            training_figure="review_v1/training_comparison.png",
        ),
    )


def freeze_selected(args, cfg):
    from src.tokenizers.anchored_carrier_dictionary import load_model
    from experiments.scripts.review_joint_tokenizer import score

    out = args.output
    prior = json.loads((out / "summary.json").read_text())
    if prior["selected_fraction"] is None:
        raise ValueError("train crossfit did not identify an eligible dictionary")
    final = out / "selected_full_train_v1"
    if (final / "screen_summary.json").exists():
        raise FileExistsError("completed frozen screen retained")
    final.mkdir(exist_ok=True)
    budget_limit = cfg["resources"]["CPU_budget_seconds"]
    resource.setrlimit(resource.RLIMIT_CPU, (budget_limit - 15, budget_limit))
    basis_run = ROOT / cfg["source"]["basis_run"]
    source_path = basis_run / "model.npz"
    source = AnchoredSpatialCarrier.load(source_path)
    manifest, scales, _, _ = base_tokenizer_training_data(
        yaml.safe_load((ROOT / cfg["source"]["g0_config"]).read_text()), ROOT
    )
    rows = manifest["rows"]
    basis_cfg = yaml.safe_load((ROOT / cfg["source"]["basis_config"]).read_text())
    plan = json.loads((ROOT / basis_cfg["source"]["native_train_plan"]).read_text())[
        "plan"
    ]
    folds = json.loads((basis_run / "fold_subjects.json").read_text())
    sites = sorted(
        {
            (r["dataset"], p)
            for r in rows
            if r["split"] == "train" and r["modality"] == "eeg"
            for p in r["positions"]
        }
    )
    checkpoint = final / "model.npz"
    if not checkpoint.exists():
        cases = make_cases(
            source,
            rows,
            scales,
            plan,
            folds,
            {s: i for i, s in enumerate(sites)},
            cfg,
            final,
            "full_train",
        )
        fraction = prior["selected_fraction"]
        books = books_for(cases, list(range(len(cases))), cfg, fraction)
        model = AnchoredCarrierDictionary(
            source,
            books,
            dict(
                schema="anchored_carrier_dictionary_v1",
                modality="eeg",
                codebook_size=cfg["model"]["vocabulary_including_missing"],
                semantic_fraction=fraction,
                seed=cfg["training"]["seed"],
                input_identity=manifest["cfg_identity"],
                scales_identity=manifest["scales_identity"],
                fitting_split="train",
                integers_per_branch_position_context=15,
                continuous_decoder_side_information=0,
            ),
        )
        model.save(checkpoint, source_path)
        del cases
    model = load_model(checkpoint)
    panel = ROOT / cfg["source"]["panel"]
    panel_meta = json.loads(panel.with_name("panel.json").read_text())
    if panel_meta["panel_sha256"] != hashlib.sha256(panel.read_bytes()).hexdigest():
        raise ValueError("frozen screen changed")
    with np.load(panel) as a:
        examples = a["eeg_screen"].copy()
    groups = defaultdict(list)
    for ordinal, (i, start, p) in enumerate(examples):
        if (
            rows[i]["split"] != "validation"
            or rows[i]["subject"]
            == panel_meta["confirmation_subjects"][rows[i]["dataset"]]
        ):
            raise ValueError(
                "predeclared development-only screen required before arrays"
            )
        groups[int(i), int(start)].append((ordinal, int(p)))
    cache, metadata, saved, metrics, support = (
        OrderedDict(),
        [],
        {},
        defaultdict(list),
        defaultdict(list),
    )
    order = []
    for crop_number, ((i, start), selected) in enumerate(groups.items()):
        names = model.metadata["positions"][rows[i]["dataset"]]
        crop = dict(
            row=i,
            start_s=start,
            positions=next(
                v["positions"]
                for v in rows[i]["inputs"]["score"]
                if v["start_s"] == start
            ),
        )
        fields, observed, _ = base_tokenizer_joint_eeg(
            rows, crop, scales, names, cache, allowed_splits=("validation",)
        )
        ids = model.encode(
            fields["x"].T.reshape(1, len(names), 15, 400),
            rows[i]["dataset"],
            observed[None],
        )
        for branch in "SO":
            saved[f"{branch}_{crop_number}"] = ids[branch]
        metadata.append(
            dict(
                crop=crop_number,
                dataset=rows[i]["dataset"],
                subject=rows[i]["subject"],
                record=rows[i]["record"],
                row=i,
                start_s=start,
                native_positions=names,
            )
        )
    np.savez_compressed(final / "integer_ids.npz", **saved)
    write(final / "integer_metadata.json", metadata)
    for pass_index in range(2):
        model = load_model(checkpoint)
        # Both score passes use persisted integers. Encoding has finished and its X path is disabled.
        model.source.continuous_source = lambda *_: (_ for _ in ()).throw(
            AssertionError("decode requested X")
        )
        model.source.analyze = lambda *_: (_ for _ in ()).throw(
            AssertionError("decode requested carriers")
        )
        values, counts = defaultdict(list), defaultdict(list)
        with np.load(final / "integer_ids.npz") as encoded:
            for crop_number, ((i, start), selected) in enumerate(groups.items()):
                dataset = rows[i]["dataset"]
                names = model.metadata["positions"][dataset]
                take = np.array([(i, start, p) for _, p in selected])
                native = [
                    next(
                        j
                        for j, n in enumerate(names)
                        if n.upper() == rows[i]["positions"][p].upper()
                    )
                    for _, p in selected
                ]
                ids = {b: encoded[f"{b}_{crop_number}"] for b in "SO"}
                pred = model.decode_ids(ids, dataset)
                batch = base_tokenizer_batch(
                    rows, take, scales, array_cache=cache, cache_limit=16
                )
                audit = joint_reconstruction_metrics(
                    {k: v[0, native] for k, v in pred.items()},
                    batch,
                    1,
                    input_scales(rows, take, scales),
                )
                for k, v in audit["metrics"].items():
                    values[k].append(v)
                for k, v in audit["support"].items():
                    counts[k].append(v)
                if pass_index == 0:
                    order.extend(o for o, _ in selected)
        ordering = np.argsort(order)
        got = {k: np.concatenate(v)[ordering] for k, v in values.items()}
        supported = {k: np.concatenate(v)[ordering] for k, v in counts.items()}
        if pass_index == 0:
            metrics, support = got, supported
        else:
            for k, v in got.items():
                np.testing.assert_array_equal(v, metrics[k])
            for k, v in supported.items():
                np.testing.assert_array_equal(v, support[k])
    np.savez_compressed(
        final / "metrics.npz",
        examples=examples,
        **metrics,
        **{"support_" + k: v for k, v in support.items()},
    )
    summary = joint_reconstruction_summary(metrics, support, examples.tolist(), rows)
    summary.update(
        status="completed",
        modality="eeg",
        role="screen",
        checkpoint=str(checkpoint),
        integer_replay=dict(
            status="passed",
            independent_model_loads=2,
            source="integer_ids.npz",
            metric_max_abs_difference=0.0,
            decoder_observation_calls=0,
        ),
    )
    write(final / "screen_summary.json", summary)
    base_root = panel.parent / "baseline_audit/eeg"
    refs = []
    for k in ["vq128", "continuous"]:
        folder = base_root / k
        path = folder / "summary_v2.json"
        refs.append(
            json.loads((path if path.exists() else folder / "summary.json").read_text())
        )
    contract = yaml.safe_load((ROOT / cfg["source"]["contract"]).read_text())
    write(final / "reconstruction_decision.json", score(summary, *refs, contract))


def full_training_worker(dataset):
    from src.tokenizers.anchored_carrier_dictionary import load_model

    out, cfg, rows, scales, plan, folds = FULL_TRAINING
    fractions = cfg["model"]["semantic_distance_fractions"]
    models = {
        (fold, frac): load_model(out / f"semantic{frac:g}_fold{fold}.npz")
        for fold in range(4)
        for frac in fractions
    }
    cache = OrderedDict()
    examples = []
    case_meta = []
    integers = {}
    metrics = {}
    supports = {}
    directory = out / "full_training_review_v2"
    directory.mkdir(exist_ok=True)
    cases = [
        (subject, crop)
        for subject, crops in sorted(plan[dataset].items())
        for crop in crops
    ]
    started = time.monotonic()
    for number, (subject, crop) in enumerate(cases):
        fold = folds[dataset][subject]
        source = models[fold, fractions[0]].source
        names = source.metadata["positions"][dataset]
        fields, seen, _ = base_tokenizer_joint_eeg(
            rows, crop, scales, names, cache, allowed_splits=("train",)
        )
        x = fields["x"].T.reshape(1, len(names), 15, 400)
        physical = source.continuous_source(x)
        carriers = source.analyze(physical, dataset, seen[None])
        continuous = source.synthesize(carriers, dataset)
        ex = np.array(
            [(crop["row"], crop["start_s"], p) for p in crop["positions"]], np.int32
        )
        native = np.array(
            [
                next(
                    i
                    for i, n in enumerate(names)
                    if n.upper() == rows[crop["row"]]["positions"][p].upper()
                )
                for p in crop["positions"]
            ]
        )
        batch = base_tokenizer_batch(
            rows, ex, scales, array_cache=cache, cache_limit=16
        )
        factor = input_scales(rows, ex, scales)
        predictions = {"continuous": continuous}
        for fraction in fractions:
            model = models[fold, fraction]
            codes = model.assign(carriers, physical, dataset, seen[None])
            if number == 0:
                direct = model.encode(x, dataset, seen[None])
                for b in "SO":
                    np.testing.assert_array_equal(codes[b], direct[b])
            for b in "SO":
                integers[f"semantic{fraction:g}_{b}_{number}"] = codes[b]
            predictions[f"semantic{fraction:g}"] = model.decode_ids(codes, dataset)
        for arm, pred in predictions.items():
            audited = joint_reconstruction_metrics(
                {k: v[0, native] for k, v in pred.items()}, batch, 1, factor
            )
            if audited["support"]["nonfinite_prediction_points"].any():
                raise ValueError("nonfinite frozen fold output")
            metrics.setdefault(arm, defaultdict(list))
            supports.setdefault(arm, defaultdict(list))
            for k, v in audited["metrics"].items():
                metrics[arm][k].append(v)
            for k, v in audited["support"].items():
                supports[arm][k].append(v)
        case_meta.append(
            dict(
                number=number,
                subject=subject,
                fold=fold,
                crop=crop,
                native_positions=names,
            )
        )
        examples.extend(ex)
        write(
            directory / (dataset + "_progress.json"),
            dict(
                stage="forward",
                crops=number + 1,
                planned=len(cases),
                seconds=time.monotonic() - started,
            ),
        )
    code_file = directory / (dataset + "_integer_ids.npz")
    np.savez_compressed(code_file, **integers)
    write(directory / (dataset + "_integer_metadata.json"), case_meta)
    combined = {
        arm: {k: np.concatenate(v) for k, v in records.items()}
        for arm, records in metrics.items()
    }
    counts = {
        arm: {k: np.concatenate(v) for k, v in records.items()}
        for arm, records in supports.items()
    }
    del models, integers
    models = {
        (fold, frac): load_model(out / f"semantic{frac:g}_fold{fold}.npz")
        for fold in range(4)
        for frac in fractions
    }
    for model in models.values():
        model.source.continuous_source = lambda *_: (_ for _ in ()).throw(
            AssertionError("integer replay asked for X")
        )
        model.source.analyze = lambda *_: (_ for _ in ()).throw(
            AssertionError("integer replay asked for carrier")
        )
    replay = {f"semantic{fraction:g}": defaultdict(list) for fraction in fractions}
    replay_support = {
        f"semantic{fraction:g}": defaultdict(list) for fraction in fractions
    }
    with np.load(code_file, allow_pickle=False) as stored:
        for item in case_meta:
            crop, fold, number = item["crop"], item["fold"], item["number"]
            names = item["native_positions"]
            ex = np.array(
                [(crop["row"], crop["start_s"], p) for p in crop["positions"]], np.int32
            )
            native = np.array(
                [
                    next(
                        i
                        for i, n in enumerate(names)
                        if n.upper() == rows[crop["row"]]["positions"][p].upper()
                    )
                    for p in crop["positions"]
                ]
            )
            batch = base_tokenizer_batch(
                rows, ex, scales, array_cache=cache, cache_limit=16
            )
            factor = input_scales(rows, ex, scales)
            for fraction in fractions:
                arm = f"semantic{fraction:g}"
                codes = {b: stored[f"{arm}_{b}_{number}"] for b in "SO"}
                prediction = models[fold, fraction].decode_ids(codes, dataset)
                audited = joint_reconstruction_metrics(
                    {k: v[0, native] for k, v in prediction.items()}, batch, 1, factor
                )
                for k, v in audited["metrics"].items():
                    replay[arm][k].append(v)
                for k, v in audited["support"].items():
                    replay_support[arm][k].append(v)
            write(
                directory / (dataset + "_progress.json"),
                dict(
                    stage="integer_replay",
                    crops=number + 1,
                    planned=len(cases),
                    seconds=time.monotonic() - started,
                ),
            )
    for arm, records in replay.items():
        for k, v in records.items():
            np.testing.assert_array_equal(np.concatenate(v), combined[arm][k])
        for k, v in replay_support[arm].items():
            np.testing.assert_array_equal(np.concatenate(v), counts[arm][k])
    for arm in combined:
        np.savez_compressed(
            directory / (dataset + "_" + arm + "_metrics.npz"),
            examples=np.asarray(examples),
            **combined[arm],
            **{"support_" + k: v for k, v in counts[arm].items()},
        )
    return dict(
        dataset=dataset,
        crops=len(cases),
        positions=len(examples),
        seconds=time.monotonic() - started,
        integer_replay_max_abs=0.0,
    )


def full_training_review(args, cfg):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    out = args.output
    dest = out / "full_training_review_v2"
    dest.mkdir(exist_ok=True)
    if (dest / "summary.json").exists():
        raise FileExistsError("retained complete full-training review")
    budget_limit = cfg["resources"]["CPU_budget_seconds"]
    resource.setrlimit(resource.RLIMIT_CPU, (budget_limit - 15, budget_limit))
    manifest, scales, _, _ = base_tokenizer_training_data(
        yaml.safe_load((ROOT / cfg["source"]["g0_config"]).read_text()), ROOT
    )
    rows = manifest["rows"]
    basis_cfg = yaml.safe_load((ROOT / cfg["source"]["basis_config"]).read_text())
    plan = json.loads((ROOT / basis_cfg["source"]["native_train_plan"]).read_text())[
        "plan"
    ]
    folds = json.loads(
        (ROOT / cfg["source"]["basis_run"] / "fold_subjects.json").read_text()
    )
    for dataset, subjects in plan.items():
        for subject, crops in subjects.items():
            if any(
                rows[c["row"]]["split"] != "train"
                or rows[c["row"]]["dataset"] != dataset
                or rows[c["row"]]["subject"] != subject
                for c in crops
            ):
                raise ValueError(
                    "registered train-only full review required before arrays"
                )
    global FULL_TRAINING
    FULL_TRAINING = out, cfg, rows, scales, plan, folds
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        usage = list(pool.map(full_training_worker, sorted(plan)))
    arms = ["continuous"] + [
        f"semantic{v:g}" for v in cfg["model"]["semantic_distance_fractions"]
    ]
    full = {}
    for arm in arms:
        metrics, support = defaultdict(list), defaultdict(list)
        examples = []
        for dataset in sorted(plan):
            with np.load(dest / (dataset + "_" + arm + "_metrics.npz")) as f:
                examples.extend(f["examples"])
                for k in f.files:
                    if k == "examples":
                        continue
                    if k.startswith("support_"):
                        support[k[8:]].append(f[k])
                    else:
                        metrics[k].append(f[k])
        expected = sorted(
            (c["row"], c["start_s"], p)
            for subjects in plan.values()
            for crops in subjects.values()
            for c in crops
            for p in c["positions"]
        )
        if sorted(map(tuple, examples)) != expected:
            raise ValueError("full training triples differ from source plan")
        values = {k: np.concatenate(v) for k, v in metrics.items()}
        counts = {k: np.concatenate(v) for k, v in support.items()}
        full[arm] = joint_reconstruction_summary(values, counts, examples, rows)
        np.savez_compressed(
            dest / (arm + "_metrics.npz"),
            examples=np.array(examples),
            **values,
            **{"support_" + k: v for k, v in counts.items()},
        )
    proxy = json.loads((out / "training_crossfit.json").read_text())
    proxy_check = {}
    for arm in arms:
        samples = []
        for dataset, subjects in sorted(plan.items()):
            for subject, crops in sorted(subjects.items()):
                crop = min(crops, key=rank)
                ex = [(crop["row"], crop["start_s"], p) for p in crop["positions"]]
                selected = sorted(
                    range(len(ex)),
                    key=lambda i: rank([cfg["training"]["seed"], *ex[i]]),
                )[:4]
                samples.extend(ex[i] for i in selected)
        with np.load(dest / (arm + "_metrics.npz")) as f:
            lookup = {tuple(x): i for i, x in enumerate(f["examples"])}
            take = [lookup[tuple(v)] for v in samples]
            metrics = {
                k: f[k][take]
                for k in f.files
                if k != "examples" and not k.startswith("support_")
            }
            counts = {k[8:]: f[k][take] for k in f.files if k.startswith("support_")}
        summ = joint_reconstruction_summary(metrics, counts, samples, rows)
        original = (
            proxy["candidates"][0]["continuous_summary"]
            if arm == "continuous"
            else next(
                v["waveform_summary"]
                for v in proxy["candidates"]
                if arm == f"semantic{v['semantic_fraction']:g}"
            )
        )
        differences = {
            k: float(
                np.max(
                    np.abs(
                        np.asarray(summ["metrics"][k]["macro"])
                        - np.asarray(original["metrics"][k]["macro"])
                    )
                )
            )
            for k in summ["metrics"]
        }
        np.testing.assert_allclose(
            summ["metrics"]["nmse"]["macro"],
            original["metrics"]["nmse"]["macro"],
            rtol=0,
            atol=0,
        )
        proxy_check[arm] = dict(
            triples=len(samples),
            NMSE_max_abs=differences["nmse"],
            macro_differences=differences,
        )
    # The parent's continuous audit used each held fold over these same cases.
    source_root = ROOT / cfg["source"]["basis_run"]
    source_match = {}
    with np.load(dest / "continuous_metrics.npz") as f:
        lookup = {tuple(x): i for i, x in enumerate(f["examples"])}
        for dataset in sorted(plan):
            with np.load(source_root / (dataset + "_held.npz")) as held:
                take = [lookup[tuple(x)] for x in held["examples"]]
                maximum = float(np.max(abs(f["nmse"][take] - held["anchored__nmse"])))
                np.testing.assert_array_equal(f["nmse"][take], held["anchored__nmse"])
                source_match[dataset] = dict(triples=len(take), NMSE_max_abs=maximum)
    write(
        dest / "summary.json",
        dict(
            status="completed",
            explanatory=True,
            training_admission_decision_unchanged=True,
            summaries=full,
            usage=usage,
            elapsed_seconds=time.monotonic() - started,
            proxy_reproduction=proxy_check,
            parent_continuous_reproduction=source_match,
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
            integer_decoder_X_calls=0,
        ),
    )


def full_training_figures(args, cfg):
    from src.tokenizers.anchored_carrier_dictionary import load_model
    import matplotlib

    matplotlib.use("Agg")

    out = args.output
    full = out / "full_training_review_v2"
    dest = full / "hard_waveforms_v2"
    dest.mkdir(exist_ok=True)
    if (dest / "examples.json").exists():
        raise FileExistsError("retained completed hard waveform export")
    manifest, scales, _, _ = base_tokenizer_training_data(
        yaml.safe_load((ROOT / cfg["source"]["g0_config"]).read_text()), ROOT
    )
    rows = manifest["rows"]
    fractions = cfg["model"]["semantic_distance_fractions"]
    selection = defaultdict(list)
    for arm in ["continuous"] + [f"semantic{f:g}" for f in fractions]:
        with np.load(full / (arm + "_metrics.npz")) as archive:
            all_examples = archive["examples"]
            all_scores = np.mean(archive["nmse"], axis=(1, 2))
        for dataset in sorted({rows[i]["dataset"] for i, _, _ in all_examples}):
            candidates = [
                n
                for n, (i, _, _) in enumerate(all_examples)
                if rows[i]["dataset"] == dataset
            ]
            chosen = max(candidates, key=lambda n: float(all_scores[n]))
            selection[tuple(all_examples[chosen])].append(arm + "_max")
    plots, identity, cache = [], [], OrderedDict()
    for number, ((i, start, p), rules) in enumerate(sorted(selection.items())):
        row = rows[i]
        dataset = row["dataset"]
        if row["split"] != "train":
            raise ValueError("training-only hard waveform export before arrays")
        metadata = json.loads((full / (dataset + "_integer_metadata.json")).read_text())
        case = next(
            c
            for c in metadata
            if c["crop"]["row"] == i and c["crop"]["start_s"] == start
        )
        fold = case["fold"]
        names = case["native_positions"]
        native = next(
            j for j, n in enumerate(names) if n.upper() == row["positions"][p].upper()
        )
        batch = base_tokenizer_batch(
            rows, [(i, start, p)], scales, array_cache=cache, cache_limit=16
        )
        factor = input_scales(rows, [(i, start, p)], scales)[0]
        fields, seen, _ = base_tokenizer_joint_eeg(
            rows,
            dict(case["crop"], positions=[]),
            scales,
            names,
            cache,
            allowed_splits=("train",),
        )
        first = load_model(out / f"semantic{fractions[0]:g}_fold{fold}.npz")
        continuous = first.predict_continuous(
            fields["x"].T.reshape(1, len(names), 15, 400), dataset, seen[None]
        )
        group = dataset + "_" + str(number)
        with np.load(full / (dataset + "_integer_ids.npz")) as ids:
            for fraction in fractions:
                model = load_model(out / f"semantic{fraction:g}_fold{fold}.npz")
                codes = {
                    b: ids[f"semantic{fraction:g}_{b}_{case['number']}"] for b in "SO"
                }
                pred = model.decode_ids(codes, dataset)
                plots.append(
                    dict(
                        dataset=group,
                        subject=row["subject"],
                        crop=case["crop"],
                        position=row["positions"][p],
                        unit=row["source_unit"],
                        fraction=fraction,
                        batch={k: batch[k][0] for k in ("x", "s", "o", "mask")},
                        scale=factor,
                        continuous={
                            b: continuous[b][0, native] for b in ("S", "O", "sum")
                        },
                        predicted={b: pred[b][0, native] for b in ("S", "O", "sum")},
                    )
                )
        identity.append(
            dict(
                group=group,
                dataset=dataset,
                subject=row["subject"],
                record=row["record"],
                parent=row["parent"],
                row=i,
                start_s=start,
                position=p,
                native_position=row["positions"][p],
                fold=fold,
                selection=rules,
                scale=factor,
                unit=row["source_unit"],
                figure=group + "_waveforms.png",
            )
        )
    render_waveforms(plots, dest)
    write(
        dest / "examples.json",
        dict(
            status="rendered_pending_actual_open",
            sampling="per dataset maxima of mean X/S/O NMSE from each of the three matching full training arms; all selected examples retained",
            examples=identity,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[
            "train",
            "review",
            "freeze-selected",
            "full-training-review",
            "full-training-figures",
        ],
        default="train",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    threads = (
        2
        if args.stage == "full-training-review"
        else cfg["resources"]["numerical_threads"]
    )
    with threadpool_limits(threads):
        {
            "train": run,
            "review": review,
            "freeze-selected": freeze_selected,
            "full-training-review": full_training_review,
            "full-training-figures": full_training_figures,
        }[args.stage](args, cfg)


if __name__ == "__main__":
    main()
