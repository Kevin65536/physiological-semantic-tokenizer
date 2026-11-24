"""Train and audit paid EEG frequency-tagged words on subject-disjoint source folds."""

from __future__ import annotations

import argparse
import fcntl
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/scripts"))
from evaluate_joint_tokenizer import input_scales, write
from src.analysis.dual_component_tokenizer import (
    base_training_aggregate,
    joint_reconstruction_metrics,
)
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_training_data
from src.tokenizers.band_exchange_words import ATOMS, fit_books, encode_slots, decode_branch, load_fold_books
from src.utils.io import file_sha256 as digest, save_npz_atomic as save
from src.tokenizers.eeg_transient_gain import load_model as load_estimator

STATE = {}
LAYOUTS = ("fixed", "exchange")
OBJECTIVES = ("wave", "balanced")


def read_arms():
    return [
        f"{layout}_{mode}_{method}"
        for layout in LAYOUTS
        for mode in OBJECTIVES
        for method in (("mean", "energy", "allocated") if mode == "balanced" else ("mean", "energy"))
    ]


@lru_cache(maxsize=12)
def folder_data(dataset, fold, branch):
    path = STATE["out"] / "prepared" / dataset / f"fold_{fold}"
    # Every numerical file must belong to a completed same-source-fold preparation.
    if not (path / "ready.json").is_file():
        raise FileNotFoundError(f"incomplete prepared source: {path}")

    def load(name):
        return np.load(path / (name + ".npy"), mmap_mode="r", allow_pickle=False)

    result = {k: load(k) for k in ("examples", "fold", "native", "seen")}
    for k in ("fixed", "role", "changed_rows", "changed", "features"):
        result["branch_" + k] = load(branch + "_" + k)
    expected = STATE["case_examples"][dataset]
    np.testing.assert_array_equal(result["examples"], expected)
    np.testing.assert_array_equal(result["fold"], STATE["case_folds"][dataset])
    np.testing.assert_array_equal(result["native"], STATE["case_sites"][dataset])
    return result


def tags(data, layout):
    if layout == "exchange":
        return np.asarray(data["branch_role"])
    return np.broadcast_to(
        np.where(data["seen"], data["native"] % 5, -1)[:, None], data["branch_role"].shape
    )


def chosen_waves(data, flat_indices, layout):
    flat = np.asarray(flat_indices)
    waves = data["branch_fixed"].reshape(-1, 400)[flat].copy()
    changed = data["branch_changed_rows"]
    if layout == "exchange" and len(changed):
        location = np.searchsorted(changed, flat)
        matches = changed[np.minimum(location, len(changed) - 1)] == flat
        waves[matches] = data["branch_changed"][location[matches]]
    return waves


def balanced_sample(indices, people, maximum, rng):
    if len(indices) <= maximum:
        return indices
    # Equalize represented training people, exhausting smaller groups first.
    groups = [
        rng.permutation(indices[people[indices // 15] == u]) for u in np.unique(people[indices // 15])
    ]
    chosen = []
    length = 0
    while length < maximum:
        available = [g for g in groups if len(g)]
        quota = max(1, (maximum - length) // len(available))
        groups = []
        for g in available:
            n = min(quota, len(g), maximum - length)
            chosen.append(g[:n])
            length += n
            if len(g) > n:
                groups.append(g[n:])
    return np.concatenate(chosen)


def fit_task(task):
    fold, branch, role = task
    out = STATE["out"] / "books"
    path = out / f"fold_{fold}_{branch}_{role}.npz"
    if path.is_file():
        return
    seed = STATE["cfg"]["training"]["seed"] + 100 * fold + 10 * ("O" == branch) + role
    rng = np.random.default_rng(seed)
    waves, properties, counts = [], [], []
    for dataset in STATE["names"]:
        data = folder_data(dataset, fold, branch)
        fit = np.asarray(data["fold"]) != fold
        persons = STATE["people"][dataset]
        assert set(persons[fit]).isdisjoint(persons[~fit])
        for layout in LAYOUTS:
            ix = np.flatnonzero((tags(data, layout) == role) & fit[:, None])
            keep = balanced_sample(ix, persons, 2048, rng)
            if len(keep):
                waves.append(chosen_waves(data, keep, layout))
                properties.append(data["branch_features"].reshape(-1, 3)[keep].copy())
            counts.append(
                dict(
                    dataset=dataset,
                    layout=layout,
                    available=len(ix),
                    used=len(keep),
                    subjects=len(np.unique(persons[keep // 15])),
                )
            )
    y, x = np.concatenate(waves), np.concatenate(properties)
    start = time.process_time()
    model = fit_books(y, x, role, seed)
    save(
        path,
        **model,
        metadata=np.array(
            json.dumps(
                dict(
                    fold=fold,
                    branch=branch,
                    role=role,
                    seed=seed,
                    support=counts,
                    training_CPU_seconds=time.process_time() - start,
                )
            )
        ),
    )
    return dict(
        fold=fold,
        branch=branch,
        role=role,
        training_examples=len(y),
        retained_variance=float(model["pca_variance_fraction"]),
    )


def load_books(fold, branch):
    return load_fold_books(STATE["out"] / "books", fold, branch)


def encode_task(task):
    fold, dataset = task
    out = STATE["out"] / dataset
    out.mkdir(exist_ok=True)
    output = out / f"fold_{fold}_ids.npz"
    if output.exists():
        return
    fields = {}
    for branch in "SO":
        data = folder_data(dataset, fold, branch)
        books = load_books(fold, branch)
        for layout in LAYOUTS:
            role_map = tags(data, layout)
            encoded = {mode: np.zeros(role_map.shape, np.uint16) for mode in OBJECTIVES}
            for role in range(5):
                places = np.flatnonzero(role_map == role)
                for start in range(0, len(places), 2048):
                    take = places[start : start + 2048]
                    waves = chosen_waves(data, take, layout)
                    properties = data["branch_features"].reshape(-1, 3)[take]
                    for mode in OBJECTIVES:
                        encoded[mode].flat[take] = encode_slots(
                            waves, properties, role, books[role], mode
                        )
            for mode, value in encoded.items():
                received = np.where(value > 0, (value.astype(np.int32) - 1) // ATOMS, -1)
                np.testing.assert_array_equal(role_map, received)
                fields[f"{layout}_{mode}_{branch}"] = value
    save(output, examples=data["examples"], fold=data["fold"], **fields)


def word_information():
    from audit_transient_actual_words import raw_readout

    s = STATE
    with np.load(s["bank_dir"] / "training_attributes.npz") as z:
        bank = {k: z[k] for k in z}
    # Each actual integer now identifies its frequency. Pool the same paid word across native sites.
    bank["fixed_band"] = np.zeros_like(bank["fixed_band"])
    bg = len(s["population"]["background_sites"]) * 4
    results = {}
    for layout in LAYOUTS:
        for mode in OBJECTIVES:
            key = f"{layout}_{mode}"
            results[key] = {}
            for b in "SO":
                out_errors = {
                    a: np.empty((len(bank["fold"]), 2, 3)) for a in ("word", "role", "within_role")
                }
                supports = defaultdict(lambda: [0, 0])
                usage = {}
                for fold in range(4):
                    parts = []
                    ordering = []
                    for d in s["names"]:
                        with np.load(s["out"] / d / f"fold_{fold}_ids.npz") as encoded:
                            parts.append(encoded[f"{key}_{b}"])
                            ordering.append(encoded["examples"])
                    paid = np.concatenate(parts)
                    np.testing.assert_array_equal(bank["examples"], np.concatenate(ordering))
                    tag = np.where(paid > 0, (paid.astype(np.int32) - 1) // ATOMS + 1, 0)
                    for name, vocab, words, environment, nbg in [
                        ("word", 5116, paid, bank, bg),
                        ("role", 6, tag, bank, bg),
                        (
                            "within_role",
                            5116,
                            paid,
                            dict(bank, background=6 * bank["background"] + tag),
                            bg * 6,
                        ),
                    ]:
                        take, err, mass, _, count = raw_readout(
                            environment, {b: words}, fold, b, vocab, nbg, include_usage=False
                        )
                        out_errors[name][take] = err
                        supports[name][0] += count["qualified_held_patches"]
                        supports[name][1] += count["complete_held_patches"]
                    for partition, rows in [
                        ("fit", bank["fold"] != fold),
                        ("held", bank["fold"] == fold),
                    ]:
                        complete = bank["valid"] & rows[:, None]
                        usage[f"{partition}_{fold}"] = np.bincount(paid[complete], minlength=5116)
                        unit = np.zeros(5116, dtype=np.int32)
                        for p in np.unique(bank["subject"][rows]):
                            unit[np.unique(paid[complete & (bank["subject"][:, None] == p)])] += 1
                        usage[f"{partition}_people_{fold}"] = unit
                for name, err in out_errors.items():
                    h = base_training_aggregate({"nmse": err}, bank["examples"], s["rows"])[
                        "metrics"
                    ]["nmse"]
                    m = np.asarray(h["macro"])
                    gain = 1 - m[1] / m[0]
                    people = np.sum(
                        [
                            np.asarray(v["value"])[1] < np.asarray(v["value"])[0]
                            for v in h["by_subject"]
                        ],
                        axis=0,
                    )
                    a, t = supports[name]
                    results[key].setdefault(b, {})[name] = dict(
                        gain=gain,
                        support=a / t,
                        people=people,
                        hierarchy=h,
                        passed=bool(
                            min(gain[:2]) > 0
                            and max(gain) >= 0.05
                            and min(people[:2]) >= 30
                            and a / t >= 0.9
                        ),
                    )
                save(
                    s["out"] / f"{key}_{b}_word_metrics.npz",
                    examples=bank["examples"],
                    **out_errors,
                    **usage,
                )
    write(s["out"] / "single_word_information.json", results)


def score_task(dataset):
    out = STATE["out"] / dataset
    cases = STATE["dataset_cases"][dataset]
    cache = OrderedDict()
    for fold in range(4):
        dest = out / f"fold_{fold}_metrics.npz"
        if dest.exists():
            continue
        books = {b: load_books(fold, b) for b in "SO"}
        with np.load(STATE["cov"] / dataset / f"fold_{fold}_model.npz") as z:
            cov = {b: z[b] for b in "SO"}
        source = load_estimator(STATE["estimator"] / f"{dataset}_fold{fold}_transient.npz")
        if set(source.metadata["fit_subjects"]) != {c["subject"] for c in cases if c["fold"] != fold}:
            raise ValueError("outer source leakage")
        metrics = defaultdict(list)
        examples = []
        pick = {}
        with np.load(out / f"fold_{fold}_ids.npz") as z:
            streams = {k: z[k] for k in z if k not in ("examples", "fold")}
        offset = 0
        for c in cases:
            take = np.arange(offset, offset + c["count"])
            offset += c["count"]
            if c["fold"] != fold:
                continue
            ex = STATE["case_examples"][dataset][take]
            assert all(STATE["rows"][i]["split"] == "train" for i, _, _ in ex)
            site = np.asarray(c["native"])
            inv = site.argsort()
            batch = base_tokenizer_batch(
                STATE["rows"], ex, STATE["scales"], array_cache=cache, cache_limit=8
            )
            factors = input_scales(STATE["rows"], ex, STATE["scales"])
            examples.extend(ex)
            for layout in LAYOUTS:
                for mode in OBJECTIVES:
                    tag = f"{layout}_{mode}"
                    versions = {}
                    for method in ("mean", "energy"):
                        branch = {}
                        for b in "SO":
                            ids = streams[tag + "_" + b][take[inv]]
                            branch[b] = decode_branch(ids, books[b], cov[b], mode, method == "energy")
                        versions[method] = dict(**branch, sum=branch["S"] + branch["O"])
                    if mode == "balanced":
                        total = versions["energy"]["sum"]
                        seen = (streams[tag + "_S"][take[inv]] != 0).any(axis=1)
                        candidate = source.predict_continuous(total[None], dataset, seen[None])["S"][
                            0
                        ]
                        selected = np.where(seen[:, None, None], candidate, versions["energy"]["S"])
                        versions["allocated"] = dict(S=selected, O=total - selected, sum=total)
                    for method, full in versions.items():
                        y = {k: v[site] for k, v in full.items()}
                        result = joint_reconstruction_metrics(y, batch, 1, factors)
                        if result["support"]["nonfinite_prediction_points"].any():
                            raise ValueError("nonfinite hard reconstruction")
                        name = tag + "_" + method
                        for k, v in result["metrics"].items():
                            metrics[name + "__" + k].append(v)
                        for j, e in enumerate(ex):
                            n = STATE["display"].get(tuple(e))
                            if n is not None and layout == "exchange" and mode == "balanced":
                                for k, v in y.items():
                                    pick[f"{n}_{name}_{k}"] = v[j]
            if offset % 300 == 0:
                write(out / "scoring_progress.json", dict(fold=fold, offset=offset))
        save(out / f"fold_{fold}_gallery.npz", **pick)
        save(dest, examples=np.array(examples), **{k: np.concatenate(v) for k, v in metrics.items()})
    return dataset


def summarize():
    out = STATE["out"]
    ref = json.loads((STATE["high"] / "summary.json").read_text())["arms"]["high_transient_source"][
        "metrics"
    ]
    info = json.loads((out / "single_word_information.json").read_text())
    arms = {}
    for arm in read_arms():
        buf = defaultdict(list)
        for dataset in STATE["names"]:
            for f in range(4):
                with np.load(out / dataset / f"fold_{f}_metrics.npz") as z:
                    buf["examples"].append(z["examples"])
                    for key in z:
                        if key.startswith(arm + "__"):
                            buf[key.split("__", 1)[1]].append(z[key])
        ex = np.concatenate(buf.pop("examples"))
        if sorted(map(tuple, ex)) != sorted(
            map(tuple, np.concatenate(list(STATE["case_examples"].values())))
        ):
            raise ValueError("score population changed")
        arrays = {k: np.concatenate(v) for k, v in buf.items()}
        s = base_training_aggregate(arrays, ex, STATE["rows"])
        m = s["metrics"]
        nm = np.ravel(m["nmse"]["macro"])
        base = np.ravel(ref["nmse"]["macro"])
        rms = np.ravel(m["rms_gain"]["macro"])
        key = arm.rsplit("_", 1)[0]
        checks = dict(
            X_NMSE=nm[0] <= 1.05 * base[0],
            components=np.all(nm[1:] <= 1.1 * base[1:]),
            dataset=all(
                np.all(np.asarray(v) <= 1.1 * np.asarray(ref["nmse"]["by_dataset"][d]))
                for d, v in m["nmse"]["by_dataset"].items()
            ),
            gain=np.all((rms >= 0.75) & (rms <= 1.25)),
            bands=all(
                np.max(abs(np.asarray(m[k]["macro"]))) <= 3.0103
                for k in ("band_4_8_db", "band_8_13_db", "band_4_8_abs_db", "band_8_13_abs_db")
            ),
            corr=min(np.ravel(m["correlation"]["macro"])) > 0,
            envelope=min(np.ravel(m["local_rms_correlation"]["macro"])) >= 0.5,
            actual_ID_S=info[key]["S"]["word"]["passed"],
            actual_ID_O=info[key]["O"]["word"]["passed"],
        )
        arms[arm] = dict(summary=s, checks=checks, passed=all(checks.values()))
    good = [k for k, v in arms.items() if v["passed"]]
    winner = min(good, key=lambda k: arms[k]["summary"]["selection_score"]) if good else None
    write(
        out / "summary.json",
        dict(
            arms=arms,
            numerically_eligible=good,
            selected=winner,
            all_people=60,
            all_native_crops=400,
            all_training_positions=17040,
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--stage",
        choices=("fit", "encode", "information", "score", "summarize", "all"),
        default="all",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if (
        cfg["schema"] != "band_exchange_waveform_words_v1"
        or cfg["data"]["test_access"] != "forbidden"
        or cfg["data"]["confirmation_access"] != "forbidden"
    ):
        raise ValueError("frozen train-only hard-word contract required")
    run = ROOT / cfg["source"]["joint_run"]
    prior = json.loads((run / cfg["source"]["eligibility"]).read_text())
    if prior["selected"] != ["exchange"]:
        raise ValueError("continuous predecessor not eligible")
    bank = run / cfg["source"]["same_source_bank"]
    pop = json.loads((bank / "population.json").read_text())
    g0 = yaml.safe_load(
        (
            ROOT / "experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml"
        ).read_text()
    )
    manifest, scales, _, _ = base_tokenizer_training_data(g0, ROOT)
    rows = manifest["rows"]
    if (pop["cfg_identity"], pop["scales_identity"]) != (
        manifest["cfg_identity"],
        manifest["scales_identity"],
    ):
        raise ValueError("population and training coordinate changed")
    cases = {
        d: [c for c in pop["cases"] if c["dataset"] == d] for d in sorted(pop["native_positions"])
    }
    ex = {
        d: np.array(
            [(c["crop"]["row"], c["crop"]["start_s"], p) for c in cc for p in c["crop"]["positions"]]
        )
        for d, cc in cases.items()
    }
    if sum(map(len, ex.values())) != 17040 or any(
        rows[i]["split"] != "train" for e in ex.values() for i, t, p in e
    ):
        raise ValueError("training support mismatch")
    high = run / cfg["source"]["high_reference"]
    gallery = json.loads((high / "training_examples/waveform_examples.json").read_text())["examples"]
    out = args.output.resolve()
    out.mkdir(exist_ok=True, parents=True)
    lock = (out / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (out / "books").mkdir(exist_ok=True)
    config_info = dict(
        config=cfg,
        input_identity=pop["cfg_identity"],
        scales_identity=pop["scales_identity"],
        population_sha256=digest(bank / "population.json"),
    )
    identity_file = out / "word_contract.json"
    if identity_file.exists() and json.loads(identity_file.read_text()) != config_info:
        raise ValueError("existing hard-word directory has a different contract")
    write(identity_file, config_info)
    STATE.update(
        cfg=cfg,
        out=out,
        rows=rows,
        scales=scales,
        bank_dir=bank,
        population=pop,
        names=sorted(cases),
        high=high,
        dataset_cases=cases,
        case_examples=ex,
        cov=run / cfg["source"]["covariance"],
        estimator=run / cfg["source"]["estimators"],
        case_folds={
            d: np.array([c["fold"] for c in cc for p in c["native"]]) for d, cc in cases.items()
        },
        case_sites={d: np.concatenate([c["native"] for c in cc]) for d, cc in cases.items()},
        people={d: np.array([rows[i]["subject"] for i, t, p in e]) for d, e in ex.items()},
        display={(v["row"], v["start_s"], v["position"]): i for i, v in enumerate(gallery)},
    )
    stages = (
        ["fit", "encode", "score", "information", "summarize"]
        if args.stage == "all"
        else [args.stage]
    )
    for stage in stages:
        begin = time.monotonic()
        if stage in ("fit", "encode", "score"):
            with ProcessPoolExecutor(
                max_workers=cfg["resources"]["workers"],
                mp_context=multiprocessing.get_context("fork"),
            ) as pool:
                if stage == "fit":
                    list(
                        pool.map(
                            fit_task, [(f, b, j) for f in range(4) for b in "SO" for j in range(5)]
                        )
                    )
                elif stage == "encode":
                    list(pool.map(encode_task, [(f, d) for f in range(4) for d in cases]))
                else:
                    list(pool.map(score_task, sorted(cases)))
        elif stage == "information":
            word_information()
        else:
            summarize()
        write(
            out / f"{stage}_complete.json", dict(stage=stage, wall_seconds=time.monotonic() - begin)
        )


if __name__ == "__main__":
    main()
