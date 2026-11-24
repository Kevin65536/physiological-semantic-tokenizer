"""Fit supervised local waveform corrections while preserving every emitted EEG integer."""

from __future__ import annotations

import argparse
import fcntl
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/scripts")]
from src.utils.io import file_sha256, save_npz_atomic
from evaluate_joint_tokenizer import input_scales, write
from src.analysis.dual_component_tokenizer import (
    base_training_aggregate,
    joint_reconstruction_metrics,
)
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_training_data
from src.tokenizers.band_exchange_words import (
    RESIDUAL_SLOTS,
    apply_residual_tables,
    decode_branch,
    neighbor_words,
    load_fold_books,
    residual_table,
)


STATE = {}


def crops(fold, fitting):
    state = STATE
    cache = OrderedDict()
    books = {b: load_fold_books(state["out"] / "books", fold, b) for b in "SO"}
    for dataset, cases in state["cases"].items():
        with np.load(state["cov"] / dataset / f"fold_{fold}_model.npz") as archive:
            covariance = {b: archive[b] for b in "SO"}
        with np.load(state["out"] / dataset / f"fold_{fold}_ids.npz") as archive:
            encoded = {b: archive["exchange_balanced_" + b] for b in "SO"}
            examples, folds = archive["examples"], archive["fold"]
        offset = 0
        for c in cases:
            take = np.arange(offset, offset + c["count"])
            offset += c["count"]
            if (c["fold"] != fold) != fitting:
                continue
            ex = examples[take]
            expected = np.array(
                [(c["crop"]["row"], c["crop"]["start_s"], i) for i in c["crop"]["positions"]]
            )
            np.testing.assert_array_equal(ex, expected)
            assert np.all(folds[take] == c["fold"]) and state["rows"][ex[0, 0]]["split"] == "train"
            site = np.asarray(c["native"])
            words = {b: encoded[b][take] for b in "SO"}
            base = {
                b: decode_branch(words[b][site.argsort()], books[b], covariance[b], "balanced", True)[
                    site
                ]
                for b in "SO"
            }
            base["sum"] = base["S"] + base["O"]
            batch = base_tokenizer_batch(
                state["rows"], ex, state["scales"], array_cache=cache, cache_limit=8
            )
            yield c, ex, batch, base, words


def build_model(fold):
    state = STATE
    out = state["destination"]
    dest = out / f"fold_{fold}_tables.npz"
    if dest.exists():
        return
    fit = [c for cc in state["cases"].values() for c in cc if c["fold"] != fold]

    def identity(c):
        return c["dataset"], c["subject"], state["rows"][c["crop"]["row"]]["record"]

    record_counts = Counter(identity(c) for c in fit)
    subject_records = Counter(k[:2] for k in record_counts)
    dataset_subjects = Counter(k[0] for k in subject_records)
    fit_subjects = set(subject_records)
    held_subjects = {
        (c["dataset"], c["subject"])
        for cc in state["cases"].values()
        for c in cc
        if c["fold"] == fold
    }
    assert fit_subjects.isdisjoint(held_subjects)
    residuals = {b: [] for b in "SO"}
    weights = {b: [] for b in "SO"}
    keys = defaultdict(list)
    for c, ex, batch, base, words in crops(fold, True):
        complete = batch["mask"].all(-1)
        if not complete.any():
            continue
        ident = identity(c)
        crop_weight = 1 / (
            dataset_subjects[ident[0]] * subject_records[ident[:2]] * record_counts[ident]
        )
        for b in "SO":
            residuals[b].append((batch[b.lower()][complete] - base[b][complete]).astype(np.float32))
            var = np.broadcast_to(batch["v" + b.lower()][:, 0, 0, None], complete.shape)
            weights[b].append((crop_weight / complete.sum()) / var[complete].astype(float))
            for offset in (-1, 0, 1):
                keys[b, offset].append(neighbor_words(words[b], offset)[complete])
    keys = {k: np.concatenate(v) for k, v in keys.items()}
    fields = {}
    for b, other in (("S", "O"), ("O", "S")):
        remaining = np.concatenate(residuals.pop(b))
        weight = np.concatenate(weights.pop(b))
        weight /= weight.mean()
        for i, (who, offset) in enumerate(RESIDUAL_SLOTS):
            code = keys[b if who == "own" else other, offset]
            table, mass = residual_table(code, remaining, weight, 5116, 16)
            fields[f"{b}_{i}"] = table
            fields[f"{b}_{i}_mass"] = mass
            for start in range(0, len(code), 4096):
                selection = slice(start, start + 4096)
                remaining[selection] -= table[code[selection]]
    save_npz_atomic(
        dest,
        **fields,
        fit_people=np.array(sorted(fit_subjects)),
        held_people=np.array(sorted(held_subjects)),
        complete_patches=np.array(len(weight)),
    )


def score_fold(fold):
    state = STATE
    dest = state["destination"]
    p = dest / f"fold_{fold}_metrics.npz"
    if p.exists():
        return
    with np.load(dest / f"fold_{fold}_tables.npz") as z:
        tables = {b: [z[f"{b}_{i}"] for i in range(len(RESIDUAL_SLOTS))] for b in "SO"}
        held_people = set(map(tuple, z["held_people"]))
    stats = defaultdict(list)
    gallery = {}
    for c, ex, batch, base, words in crops(fold, False):
        assert (c["dataset"], c["subject"]) in held_people
        stats["examples"].append(ex)
        for arm, steps in [("own", 1), ("context", len(RESIDUAL_SLOTS))]:
            wave = apply_residual_tables(base, words, tables, steps)
            result = joint_reconstruction_metrics(
                wave, batch, 1, input_scales(state["rows"], ex, state["scales"])
            )
            if result["support"]["nonfinite_prediction_points"].any():
                raise ValueError("nonfinite EEG output")
            for k, v in result["metrics"].items():
                stats[arm + "__" + k].append(v)
            for j, e in enumerate(ex):
                if tuple(e) in state["display"]:
                    label = state["display"][tuple(e)]
                    for name in ("sum", "S", "O"):
                        gallery[f"{label}_{arm}_{name}"] = wave[name][j]
    save_npz_atomic(dest / f"fold_{fold}_gallery.npz", **gallery)
    save_npz_atomic(p, **{k: np.concatenate(v) for k, v in stats.items()})


def run_fold(fold):
    start = time.process_time()
    build_model(fold)
    score_fold(fold)
    return (fold, time.process_time() - start)


def summarize():
    state = STATE
    dest = state["destination"]
    vectors = defaultdict(list)
    for f in range(4):
        with np.load(dest / f"fold_{f}_metrics.npz") as z:
            for k in z:
                vectors[k].append(z[k])
    ex = np.concatenate(vectors.pop("examples"))
    if sorted(map(tuple, ex)) != sorted(
        (c["crop"]["row"], c["crop"]["start_s"], p)
        for cc in state["cases"].values()
        for c in cc
        for p in c["crop"]["positions"]
    ):
        raise ValueError("position support changed")
    results = {}
    reference = state["reference"]["arms"]["high_transient_source"]["metrics"]["nmse"]
    for arm in ("own", "context"):
        metrics = {
            k.split("__")[1]: np.concatenate(v)
            for k, v in vectors.items()
            if k.startswith(arm + "__")
        }
        s = base_training_aggregate(metrics, ex, state["rows"])
        m = s["metrics"]
        nm = np.ravel(m["nmse"]["macro"])
        old = np.ravel(reference["macro"])
        gain = np.ravel(m["rms_gain"]["macro"])
        checks = dict(
            total=nm[0] <= 1.05 * old[0],
            components=np.all(nm[1:] <= 1.1 * old[1:]),
            datasets=all(
                np.all(np.array(v) <= 1.1 * np.array(reference["by_dataset"][d]))
                for d, v in m["nmse"]["by_dataset"].items()
            ),
            gain=np.all((gain >= 0.75) & (gain <= 1.25)),
            bands=all(
                np.max(abs(np.asarray(m[k]["macro"]))) <= 3.0103
                for k in ("band_4_8_db", "band_8_13_db", "band_4_8_abs_db", "band_8_13_abs_db")
            ),
            correlation=min(np.ravel(m["correlation"]["macro"])) > 0,
            envelope=min(np.ravel(m["local_rms_correlation"]["macro"])) >= 0.5,
        )
        results[arm] = dict(summary=s, checks=checks, passed=all(checks.values()))
    write(
        dest / "summary.json",
        dict(
            arms=results,
            passed=[k for k, v in results.items() if v["passed"]],
            paid_ids="unchanged exchange_balanced",
            source_word_information=str(state["out"] / "single_word_information.json"),
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if (
        cfg["schema"] != "band_word_supervised_residual_v1"
        or cfg["data"]["development"] != "forbidden"
    ):
        raise ValueError("train-only contract required")
    run = ROOT / cfg["joint_run"]
    out = run / cfg["frozen"]
    information = json.loads((out / "single_word_information.json").read_text())["exchange_balanced"]
    assert all(information[b]["word"]["passed"] for b in "SO")
    pop = json.loads(
        (
            run / "component_source_transfer/hard_as_v1/local_information_v1/population.json"
        ).read_text()
    )
    g0 = yaml.safe_load(
        (
            ROOT / "experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml"
        ).read_text()
    )
    manifest, scales, _, _ = base_tokenizer_training_data(g0, ROOT)
    assert (pop["cfg_identity"], pop["scales_identity"]) == (
        manifest["cfg_identity"],
        manifest["scales_identity"],
    )
    cases = {
        d: [c for c in pop["cases"] if c["dataset"] == d] for d in sorted(pop["native_positions"])
    }
    draws = json.loads((out / "visual_review_v1/selected_examples.json").read_text())
    examples = draws if isinstance(draws, list) else draws["examples"]

    def example(e):
        return (e["row"], e["start_s"], e["position"])

    chosen = {example(e): i for i, e in enumerate(examples)}
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    lock = (dest / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(
        cfg=cfg,
        source_contract_sha256=file_sha256(out / "word_contract.json"),
        population_sha256=file_sha256(
            run / "component_source_transfer/hard_as_v1/local_information_v1/population.json"
        ),
    )
    identity_file = dest / "resolved_config.json"
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError("different existing model contract")
    write(identity_file, identity)
    ref = json.loads(
        (run / "component_source_transfer/integer_sum_source_v1/summary.json").read_text()
    )
    STATE.update(
        out=out,
        destination=dest,
        cases=cases,
        rows=manifest["rows"],
        scales=scales,
        display=chosen,
        cov=run / "component_source_transfer/adaptive_band_v1",
        reference=ref,
    )
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("fork")
    ) as workers:
        accounting = list(workers.map(run_fold, range(4)))
    summarize()
    write(
        dest / "completion.json",
        dict(worker_CPU_seconds=accounting, folds=4, original_arrays=400, positions=17040),
    )


if __name__ == "__main__":
    main()
