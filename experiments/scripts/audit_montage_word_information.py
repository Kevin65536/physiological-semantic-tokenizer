"""Audit actual integer words after decoder-aware montage search on original training folds."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
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

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/scripts")]
from evaluate_joint_tokenizer import write
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit, feature_means
from src.data.dual_component_tokenizer import base_tokenizer_training_data
from src.utils.io import save_npz_atomic

STATE = {}
FIELDS = {"theta": (0, 1), "alpha": (1, 2), "RMS": (2, 10)}
CAPACITY = 15351
ARMS = ("frozen", "semantic64", "waveform")
READOUTS = ("word", "type_role", "within_type_role")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def budget():
    unit = os.environ.get("MONTAGE_INFORMATION_UNIT")
    if unit:
        used = (
            int(
                subprocess.check_output(
                    ["systemctl", "--user", "show", unit, "-p", "CPUUsageNSec", "--value"], text=True
                )
            )
            / 1e9
        )
        if used >= 1170:
            raise RuntimeError("1200 CPU-second information audit budget reached")


def restore_montage_streams(
    examples,
    folds,
    fold,
    fit,
    held,
    *,
    stream_fields=("semantic64_S", "semantic64_O", "waveform_S", "waveform_O"),
    capacity=CAPACITY,
):
    """Restore the same population order from two disjoint physical-identity streams."""
    identity = {tuple(e): i for i, e in enumerate(examples)}
    if len(identity) != len(examples):
        raise ValueError("duplicate original physical example")
    output = {k: np.zeros((len(examples), 15), np.uint16) for k in stream_fields}
    seen = np.zeros(len(examples), bool)
    for partition, archive in (("fit", fit), ("held", held)):
        if set(archive) != {"examples", *output}:
            raise ValueError("unexpected search integer fields")
        keys = [tuple(e) for e in archive["examples"]]
        if len(set(keys)) != len(keys) or any(e not in identity for e in keys):
            raise ValueError("duplicate or out-of-population search example")
        ix = np.array([identity[e] for e in keys], np.int64)
        expected = folds != fold if partition == "fit" else folds == fold
        if np.any(seen[ix]) or not np.all(expected[ix]) or set(ix) != set(np.flatnonzero(expected)):
            raise ValueError("search fit/held membership differs from original person fold")
        seen[ix] = True
        for key in output:
            values = archive[key]
            if (
                values.shape != (len(ix), 15)
                or values.dtype.kind not in "ui"
                or np.any(values < 0)
                or np.any(values >= capacity)
            ):
                raise ValueError("actual paid K15351 integer matrix required")
            output[key][ix] = values
    if not seen.all():
        raise ValueError("missing search population member")
    return output


def type_role(words):
    role = np.where(words > 0, (words.astype(np.int32) - 1) // 1023 + 1, 0)
    second = words >= 5116
    role[second] = 6 + (words[second].astype(np.int32) - 5116) // 2047
    if np.any(role < 0) or np.any(role >= 11):
        raise ValueError("paid role outside frozen ten-role namespace")
    return role


def read_streams(fold):
    arrays = {a: {b: [] for b in "SO"} for a in ARMS}
    proof = []
    for d in STATE["names"]:
        old_path = STATE["frozen"] / d / f"fold_{fold}_ids.npz"
        with np.load(old_path) as z:
            expected = z["examples"]
            folds = z["fold"]
            baseline = {b: z["innovation_" + b] for b in "SO"}
        np.testing.assert_array_equal(expected, STATE["examples"][d])
        fit_path = STATE["source"] / d / f"fit_fold{fold}_ids.npz"
        held_path = STATE["source"] / d / f"held_fold{fold}_ids.npz"
        with np.load(fit_path) as fit, np.load(held_path) as held:
            merged = restore_montage_streams(expected, folds, fold, fit, held)
        for b in "SO":
            arrays["frozen"][b].append(baseline[b])
            for a in ARMS[1:]:
                values = merged[a + "_" + b]
                np.testing.assert_array_equal(type_role(values), type_role(baseline[b]))
                arrays[a][b].append(values)
        proof.append(
            dict(
                dataset=d,
                fold=fold,
                full_examples=len(expected),
                fit_examples=int(np.sum(folds != fold)),
                held_examples=int(np.sum(folds == fold)),
                frozen_ID_sha256=sha(old_path),
                fit_ID_sha256=sha(fit_path),
                held_ID_sha256=sha(held_path),
                paid_type_role_exact=True,
            )
        )
    return {
        a: {b: np.concatenate(v) for b, v in branches.items()} for a, branches in arrays.items()
    }, proof


def local_counts(words, fit, held):
    bank = STATE["bank"]
    fm = bank["valid"] & fit[:, None]
    hm = bank["valid"] & held[:, None]
    counts = np.bincount(words[fm], minlength=CAPACITY)
    held_counts = np.bincount(words[hm], minlength=CAPACITY)
    persons = np.zeros(CAPACITY, np.int32)
    dataset_coverage = np.zeros(CAPACITY, np.int16)
    for person in np.unique(bank["subject"][fit]):
        mask = bank["valid"] & (bank["subject"][:, None] == person)
        persons[np.unique(words[mask])] += 1
    for d in STATE["names"]:
        mask = fm & (STATE["dataset"][:, None] == d)
        dataset_coverage[np.unique(words[mask])] += 1
    good = (counts[words[hm]] >= 8) & (persons[words[hm]] >= 2)
    return (
        counts,
        held_counts,
        persons,
        dataset_coverage,
        dict(
            complete_held_patches=int(hm.sum()),
            qualified_held_patches=int(good.sum()),
            qualified_fraction=float(good.mean()),
            used_positive_fit=int(np.sum(counts[1:] > 0)),
            dead_positive_paid=int(np.sum(counts[1:] == 0)),
            dead_positive_paid_fraction=float(np.mean(counts[1:] == 0)),
            held_only_positive=int(np.sum((held_counts[1:] > 0) & (counts[1:] == 0))),
            rare_positive_fit=int(np.sum((counts[1:] > 0) & ((counts[1:] < 8) | (persons[1:] < 2)))),
        ),
    )


def support_fold(fold):
    budget()
    arrays, proof = read_streams(fold)
    bank = STATE["bank"]
    fit = bank["fold"] != fold
    held = ~fit
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for b in "SO":
            result[arm][b] = local_counts(arrays[arm][b], fit, held)[-1]
    write(
        STATE["out"] / f"fold_{fold}_support_preliminary.json",
        dict(
            fold=fold,
            covered_libraries=STATE["names"],
            original_held_people=int(len(np.unique(bank["subject"][held]))),
            status="one_complete_fold_only",
            arms=result,
            input_identity=proof,
        ),
    )
    return result


def score_fold(fold):
    budget()
    before = time.process_time()
    arrays, proof = read_streams(fold)
    bank = STATE["bank"]
    fit = bank["fold"] != fold
    held = ~fit
    valid = bank["valid"]
    fm = valid & fit[:, None]
    hv = valid[held]
    if set(bank["subject"][fit]) & set(bank["subject"][held]):
        raise ValueError("person overlap in single-ID templates")
    output = dict(examples=bank["examples"][held], held_indices=np.flatnonzero(held))
    result = {}
    csv_path = STATE["out"] / f"fold_{fold}_all_ID_usage.csv"
    with csv_path.open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "fold",
                "arm",
                "branch",
                "ID",
                "paid_type_role",
                "fit_complete_patches",
                "fit_people",
                "fit_datasets",
                "held_complete_patches",
                "fit_background_ID_cells",
                "held_background_ID_cells",
                "held_patches_in_seen_background_ID",
                "held_patches_in_8patch2people_background_ID",
                "rare_or_dead_fit",
            ]
        )
        for arm in ARMS:
            result[arm] = {}
            for b in "SO":
                budget()
                words = arrays[arm][b]
                role = type_role(words)
                counts, hcounts, persons, datasets, support = local_counts(words, fit, held)
                summaries = {}
                for readout, codes, background, k, nbg in (
                    ("word", words, bank["background"], CAPACITY, STATE["backgrounds"]),
                    ("type_role", role, bank["background"], 11, STATE["backgrounds"]),
                    (
                        "within_type_role",
                        words,
                        11 * bank["background"] + role,
                        CAPACITY,
                        11 * STATE["backgrounds"],
                    ),
                ):
                    model = SparsePatternAudit(k, nbg, [], 10, FIELDS)
                    model.fit(codes[fm], background[fm], bank[b + "_attributes"][fm])
                    key = codes[held][hv]
                    bg = background[held][hv]
                    base, _, prediction = model.read(key, bg)
                    target = bank[b + "_attributes"][held][hv]
                    err = np.zeros((*hv.shape, 2, 3))
                    err[hv] = np.stack(
                        [feature_means(np.square(v - target), FIELDS) for v in (base, prediction)],
                        axis=1,
                    )
                    window = err.sum(1) / hv.sum(1).clip(1)[:, None, None]
                    window[hv.sum(1) == 0] = np.nan
                    output[arm + "_" + b + "_" + readout] = window
                    unit = np.zeros(k, np.int32)
                    for person in np.unique(bank["subject"][fit]):
                        mask = valid & (bank["subject"][:, None] == person)
                        unit[np.unique(codes[mask])] += 1
                    qualified = (model.local.count[key] >= 8) & (unit[key] >= 2)
                    summaries[readout] = dict(
                        complete_held_patches=len(key),
                        qualified_held_patches=int(qualified.sum()),
                        support=float(qualified.mean()),
                    )
                    if arm == "frozen":
                        with np.load(
                            STATE["frozen"] / f"innovation_{b}_word_metrics.npz"
                        ) as retained:
                            np.testing.assert_array_equal(retained["examples"], bank["examples"])
                            np.testing.assert_allclose(
                                window,
                                retained[readout][held],
                                rtol=2e-14,
                                atol=1e-14,
                                equal_nan=True,
                            )
                            if readout == "word":
                                for saved, now in [
                                    ("fit_count_", counts),
                                    ("fit_people_", persons),
                                    ("held_count_", hcounts),
                                ]:
                                    np.testing.assert_array_equal(retained[saved + str(fold)], now)
                    if readout == "word":
                        np.testing.assert_array_equal(model.local.count, counts)
                        fit_joint = np.array(list(model.joint.count), np.int64)
                        fit_cells = np.bincount(fit_joint % k, minlength=k)
                        testjoint = bg * k + key
                        cell, number = np.unique(testjoint, return_counts=True)
                        held_cells = np.bincount(cell % k, minlength=k)
                        present = np.isin(testjoint, fit_joint)
                        seen_count = np.bincount(key[present], minlength=k)
                        joint_people = Counter()
                        for person in np.unique(bank["subject"][fit]):
                            mask = valid & (bank["subject"][:, None] == person)
                            joint_people.update(
                                map(int, np.unique(background[mask] * k + codes[mask]))
                            )
                        qualcells = np.array(
                            [
                                model.joint.count.get(int(c), 0) >= 8 and joint_people[int(c)] >= 2
                                for c in cell
                            ]
                        )
                        occupied = qualcells[np.searchsorted(cell, testjoint)]
                        qualified_cells = np.bincount(key[occupied], minlength=k)
                        conditional = dict(
                            held_seen_background_ID_fraction=float(present.mean()),
                            held_8patch2people_background_ID_fraction=float(occupied.mean()),
                            fit_observed_background_ID_cells=len(fit_joint),
                            held_observed_background_ID_cells=len(cell),
                            held_unseen_background_ID_cells=int(np.sum(~np.isin(cell, fit_joint))),
                        )
                        for code in range(CAPACITY):
                            writer.writerow(
                                (
                                    fold,
                                    arm,
                                    b,
                                    code,
                                    int(type_role(np.array([code]))[0]),
                                    int(counts[code]),
                                    int(persons[code]),
                                    int(datasets[code]),
                                    int(hcounts[code]),
                                    int(fit_cells[code]),
                                    int(held_cells[code]),
                                    int(seen_count[code]),
                                    int(qualified_cells[code]),
                                    bool(counts[code] < 8 or persons[code] < 2),
                                )
                            )
                        for name, v in [
                            ("fit_count", counts),
                            ("held_count", hcounts),
                            ("fit_people", persons),
                            ("fit_datasets", datasets),
                            ("fit_background_ID_cells", fit_cells),
                            ("held_background_ID_cells", held_cells),
                            ("held_seen_background_ID_count", seen_count),
                            ("held_conditional_qualified_count", qualified_cells),
                        ]:
                            output[arm + "_" + b + "_" + name] = v
                result[arm][b] = dict(readouts=summaries, usage=support, conditional=conditional)
    save_npz_atomic(STATE["out"] / f"fold_{fold}_metrics.npz", **output)
    write(
        STATE["out"] / f"fold_{fold}_review.json",
        dict(
            fold=fold,
            arms=result,
            input_identity=proof,
            positive_control="frozen v8 every readout and all fit/held code and person counts independently reproduced",
            CPU_seconds=time.process_time() - before,
            fit_people=np.unique(bank["subject"][fit]),
            held_people=np.unique(bank["subject"][held]),
        ),
    )
    return fold


def summarize(folds):
    bank = STATE["bank"]
    reviews = {f: json.loads((STATE["out"] / f"fold_{f}_review.json").read_text()) for f in folds}
    global_indices = np.sort(np.concatenate([np.flatnonzero(bank["fold"] == f) for f in folds]))
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for b in "SO":
            branch = {}
            for readout in READOUTS:
                allerr = np.full((len(bank["examples"]), 2, 3), np.nan)
                for f in folds:
                    with np.load(STATE["out"] / f"fold_{f}_metrics.npz") as z:
                        allerr[z["held_indices"]] = z[arm + "_" + b + "_" + readout]
                h = base_training_aggregate(
                    {"nmse": allerr[global_indices]}, bank["examples"][global_indices], STATE["rows"]
                )["metrics"]["nmse"]
                m = np.asarray(h["macro"])
                gain = 1 - m[1] / m[0]
                positive = np.sum(
                    [np.array(v["value"])[1] < np.array(v["value"])[0] for v in h["by_subject"]],
                    axis=0,
                )
                masses = [reviews[f]["arms"][arm][b]["readouts"][readout] for f in folds]
                n = sum(v["complete_held_patches"] for v in masses)
                good = sum(v["qualified_held_patches"] for v in masses)
                branch[readout] = dict(
                    gain=gain,
                    positive_people=positive,
                    support=good / n,
                    hierarchy=h,
                    complete_held_patches=n,
                    qualified_held_patches=good,
                    passed=bool(
                        len(folds) == 4
                        and min(gain[:2]) > 0
                        and max(gain) >= 0.05
                        and min(positive[:2]) >= 30
                        and good / n >= 0.9
                    ),
                )
                if arm == "frozen" and len(folds) == 4:
                    reference = json.loads(
                        (STATE["frozen"] / "single_word_information.json").read_text()
                    )["innovation"][b][readout]
                    np.testing.assert_allclose(gain, reference["gain"], rtol=5e-12, atol=1e-13)
                    np.testing.assert_array_equal(positive, reference["positive_people"])
                    assert good / n == reference["support"]
            branch["conditional_occupancy"] = {
                key: sum(
                    reviews[f]["arms"][arm][b]["conditional"][key]
                    * reviews[f]["arms"][arm][b]["usage"]["complete_held_patches"]
                    for f in folds
                )
                / sum(reviews[f]["arms"][arm][b]["usage"]["complete_held_patches"] for f in folds)
                for key in (
                    "held_seen_background_ID_fraction",
                    "held_8patch2people_background_ID_fraction",
                )
            }
            result[arm][b] = branch
    record = dict(
        status="complete" if len(folds) == 4 else "partial_completed_folds_only",
        completed_folds=folds,
        original_positions=len(global_indices),
        people=len(np.unique(bank["subject"][global_indices])),
        original_arrays=400 if len(folds) == 4 else None,
        arms=result,
        positive_control="v8 frozen actual integers and original readouts match",
        validation_reads=0,
        confirmation_reads=0,
        test_reads=0,
    )
    write(STATE["out"] / ("summary.json" if len(folds) == 4 else "partial_summary.json"), record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--support-only", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["data"] != {
        "fit": "train",
        "score": "train_held_people",
        "validation": "forbidden",
        "confirmation": "forbidden",
        "test": "forbidden",
    }:
        raise ValueError("original train-only boundary required")
    run = ROOT / cfg["run"]
    source = run / "component_source_transfer/montage_word_search_v1"
    frozen = run / cfg["frozen"]
    out = args.output.resolve()
    out.mkdir(exist_ok=True, parents=True)
    pop = json.loads((run / cfg["population"]).read_text())
    with np.load(
        run / "component_source_transfer/hard_as_v1/local_information_v1/training_attributes.npz"
    ) as z:
        bank = dict(z)
    g0 = yaml.safe_load(
        (
            ROOT / "experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml"
        ).read_text()
    )
    manifest, _, _, _ = base_tokenizer_training_data(g0, ROOT)
    rows = manifest["rows"]
    if (pop["cfg_identity"], pop["scales_identity"]) != (
        manifest["cfg_identity"],
        manifest["scales_identity"],
    ):
        raise ValueError("frozen scale or target version changed")
    assert all(rows[i]["split"] == "train" for i, t, p in bank["examples"])
    names = sorted(pop["native_positions"])
    examples = {
        d: np.array(
            [
                (c["crop"]["row"], c["crop"]["start_s"], p)
                for c in pop["cases"]
                if c["dataset"] == d
                for p in c["crop"]["positions"]
            ]
        )
        for d in names
    }
    np.testing.assert_array_equal(bank["examples"], np.concatenate(list(examples.values())))
    STATE.update(
        run=run,
        source=source,
        frozen=frozen,
        out=out,
        pop=pop,
        bank=bank,
        rows=rows,
        names=names,
        examples=examples,
        dataset=np.array([rows[i]["dataset"] for i, t, p in bank["examples"]]),
        backgrounds=4 * len(pop["background_sites"]),
    )
    ready = [
        f
        for f in range(4)
        if all(
            (source / d / f"{part}_fold{f}_ids.npz").is_file()
            for d in names
            for part in ("fit", "held")
        )
    ]
    if not ready:
        raise RuntimeError("no full four-library fit+held fold ready; keep waiting on producer")
    write(
        out / "input_contract.json",
        dict(
            config=cfg,
            ready_folds=ready,
            training_attributes_sha256=sha(
                run
                / "component_source_transfer/hard_as_v1/local_information_v1/training_attributes.npz"
            ),
            population_sha256=sha(run / cfg["population"]),
            audit_source_sha256=sha(Path(__file__)),
            config_sha256=sha(args.config),
        ),
    )
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
        if args.support_only:
            result = list(pool.map(support_fold, ready))
            write(
                out / "support_preliminary.json",
                dict(
                    completed_folds=ready,
                    per_fold=dict(zip(ready, result)),
                    status="all_fold_support_only"
                    if len(ready) == 4
                    else "partial_folds_support_only",
                ),
            )
        else:
            pending = [f for f in ready if not (out / f"fold_{f}_metrics.npz").is_file()]
            completed = [f for f in ready if f not in pending]
            jobs = {pool.submit(score_fold, f): f for f in pending}
            for job in as_completed(jobs):
                completed.append(job.result())
                summarize(sorted(completed))
            summarize(ready)


if __name__ == "__main__":
    main()
