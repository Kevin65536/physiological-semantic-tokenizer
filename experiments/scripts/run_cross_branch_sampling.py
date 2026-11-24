"""Fit paired spatial covariances and evaluate aligned versus staggered S/O frequency samples."""

import argparse
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/scripts")]
from evaluate_joint_tokenizer import input_scales, write
from src.analysis.dual_component_tokenizer import (
    base_training_aggregate,
    joint_reconstruction_metrics,
)
from src.data.dual_component_tokenizer import (
    base_tokenizer_training_data,
    base_tokenizer_joint_eeg,
    base_tokenizer_batch,
)
from src.tokenizers.adaptive_band_allocation import band_waves
from src.tokenizers.band_exchange_words import load_fold_books, paid_coordinates, source_properties
from src.tokenizers.cross_branch_sampling import (
    layouts,
    encode_role,
    covariance_blocks,
    calibration,
    reconstruct,
)
from src.tokenizers.eeg_transient_gain import band_indices, load_model
from src.utils.io import file_sha256, save_npz_atomic

STATE = {}
LAYOUTS = ("aligned", "staggered")
READOUTS = (
    "separate_continuous",
    "joint_continuous",
    "separate_hard",
    "joint_hard",
    "separate_calibrated",
    "joint_calibrated",
)


def normalized_observations(cases, names):
    result = []
    for c in cases:
        row = STATE["rows"][c["crop"]["row"]]
        if (row["dataset"], row["subject"], row["split"], row["modality"]) != (
            c["dataset"],
            c["subject"],
            "train",
            "eeg",
        ):
            raise ValueError("invalid original training array before data access")
        values, seen, _ = base_tokenizer_joint_eeg(
            STATE["rows"],
            dict(c["crop"], positions=[]),
            STATE["scales"],
            names,
            allowed_splits=("train",),
        )
        result.append((values["x"].T.reshape(1, len(names), 15, 400).astype(np.float32), seen))
    return result


def assignment(prediction, seen, books):
    geometry = layouts(seen)
    spectra = {}
    waves = {}
    sent = {}
    hard = {}
    for b in "SO":
        spectra[b] = np.fft.rfft(prediction[b][0].reshape(len(seen), 6000))
        f = band_waves(spectra[b])
        features = source_properties(prediction[b][0])
        for layout in LAYOUTS:
            key = layout, b
            if layout == "staggered" and b == "S":
                for table in (waves, sent, hard):
                    table[key] = table["aligned", b]
                continue
            sent[key], waves[key] = encode_role(f, geometry[layout][b], features, books[b])
            hard[key], _ = paid_coordinates(sent[key], books[b], "balanced", True)
    return geometry, spectra, waves, sent, hard


def fit_fold(ds, fold, cases, xvalues, names, folder):
    s = STATE
    n = len(names)
    source_path = s["source"] / f"{ds}_fold{fold}_transient.npz"
    source = load_model(source_path)
    fit = [c for c in cases if c["fold"] != fold]
    people = {c["subject"] for c in fit}
    assert set(source.metadata["fit_subjects"]) == people and source.metadata["held_fold"] == fold
    covfile = s["cov"] / ds / f"fold_{fold}_model.npz"
    with np.load(covfile) as z:
        np.testing.assert_array_equal(z["names"], names)
        assert set(z["fit_people"]) == people
        marginal = {b: z[b] for b in "SO"}
    books = {b: load_fold_books(s["dict"] / "books", fold, b) for b in "SO"}
    rec = Counter((c["subject"], s["rows"][c["crop"]["row"]]["record"]) for c in fit)
    nrec = Counter(p for p, record in rec)
    cross = np.zeros((5, n, n))
    mass = np.zeros((n, n))
    moments = {l: np.zeros((4, 5, 2 * n)) for l in LAYOUTS}
    streams = defaultdict(list)
    examples = []
    folds = []
    held = []
    for c, (x, seen) in zip(cases, xvalues):
        geom, spec, waves, words, decoded = assignment(
            source.predict_continuous(x, ds, seen[None]), seen, books
        )
        site = np.asarray(c["native"])
        np.testing.assert_array_equal(np.sort(site), np.arange(n))
        ex = np.array([(c["crop"]["row"], c["crop"]["start_s"], p) for p in c["crop"]["positions"]])
        examples.append(ex)
        folds.extend([c["fold"]] * len(ex))
        for (layout, b), v in words.items():
            streams[layout + "_" + b].append(v[site])
        if c["fold"] == fold:
            held.append((ex, site, geom, waves, decoded))
            continue
        subject = c["subject"]
        record = s["rows"][ex[0, 0]]["record"]
        w = 1 / (rec[subject, record] * nrec[subject])
        for j, indices in enumerate(band_indices()):
            cross[j] += (
                w * 2 * (spec["S"][:, indices] @ spec["O"][:, indices].conj().T).real / 6000**2
            )
        mass += w * (seen[:, None] & seen[None])
        for l in LAYOUTS:
            for bi, b in enumerate("SO"):
                y = waves[l, b].reshape(n, 6000)
                q = decoded[l, b].reshape(n, 6000)
                stats = np.stack([(y * y).mean(1), (y * q).mean(1), (q * q).mean(1), np.ones(n)])
                sites = np.flatnonzero(seen)
                moments[l][:, geom[l][b][sites], sites + bi * n] += w * stats[:, sites]
    words = {k: np.concatenate(v) for k, v in streams.items()}
    with np.load(s["dict"] / ds / f"fold_{fold}_ids.npz") as prior:
        np.testing.assert_array_equal(np.concatenate(examples), prior["examples"])
        np.testing.assert_array_equal(folds, prior["fold"])
        for b in "SO":
            np.testing.assert_array_equal(words["aligned_" + b], prior["fixed_balanced_" + b])
    save_npz_atomic(
        folder / f"fold_{fold}_ids.npz", examples=np.concatenate(examples), fold=folds, **words
    )
    independent, joint, cor = covariance_blocks(
        marginal["S"], marginal["O"], cross / np.maximum(mass, 1e-12)
    )
    state = dict(
        independent=independent,
        joint=joint,
        canonical_correlations_before_clip=cor,
        fit_people=np.array(sorted(people)),
        held_people=np.array(sorted({c["subject"] for c in cases if c["fold"] == fold})),
        source_sha256=file_sha256(source_path),
        marginal_sha256=file_sha256(covfile),
    )
    for layout in LAYOUTS:
        state[layout + "_gain"], state[layout + "_noise"] = calibration(moments[layout])
    save_npz_atomic(folder / f"fold_{fold}_model.npz", **state)
    return state, held


def score_fold(ds, fold, state, held, folder):
    s = STATE
    positions = []
    metrics = defaultdict(list)
    cache = OrderedDict()
    for ex, sites, geom, dense, hard in held:
        batch = base_tokenizer_batch(s["rows"], ex, s["scales"], array_cache=cache, cache_limit=6)
        scal = input_scales(s["rows"], ex, s["scales"])
        positions.append(ex)
        for layout in LAYOUTS:
            role = np.concatenate([geom[layout][b] for b in "SO"])
            unit = np.ones((5, len(role)))
            zero = np.zeros_like(unit)
            for arm in READOUTS:
                spatial, kind = arm.split("_")
                values = dense if kind == "continuous" else hard
                coord = np.concatenate([values[layout, b] for b in "SO"])
                g = state[layout + "_gain"] if kind == "calibrated" else unit
                noise = state[layout + "_noise"] if kind == "calibrated" else zero
                cov = state["joint" if spatial == "joint" else "independent"]
                y = reconstruct(coord, role, cov, g, noise)
                scored = joint_reconstruction_metrics(
                    {k: v[sites] for k, v in y.items()}, batch, 1, scal
                )
                if scored["support"]["nonfinite_prediction_points"].any():
                    raise ValueError("nonfinite voltage")
                for k, v in scored["metrics"].items():
                    metrics[layout + "_" + arm + "__" + k].append(v)
    values = {k: np.concatenate(v) for k, v in metrics.items()}
    positions = np.concatenate(positions)
    priors = [
        (
            s["dict"] / ds / f"fold_{fold}_metrics.npz",
            "fixed_balanced_energy",
            "aligned_separate_hard",
        ),
        (s["cov"] / ds / f"fold_{fold}_metrics.npz", "fixed", "aligned_separate_continuous"),
    ]
    checks = {}
    for path, old, new in priors:
        with np.load(path) as prior:
            np.testing.assert_array_equal(prior["examples"], positions)
            expected = prior[old + "__nmse"]
            actual = values[new + "__nmse"]
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
            checks[new] = float(np.max(abs(actual - expected)))
    save_npz_atomic(folder / f"fold_{fold}_metrics.npz", examples=positions, **values)
    write(folder / f"fold_{fold}_reproduction.json", checks)


def dataset(ds):
    cases = STATE["cases"][ds]
    names = STATE["pop"]["native_positions"][ds]
    out = STATE["out"] / ds
    out.mkdir(exist_ok=True)
    x = normalized_observations(cases, names)
    start = time.process_time()
    for fold in range(4):
        if (out / f"fold_{fold}_metrics.npz").exists():
            continue
        state, held = fit_fold(ds, fold, cases, x, names, out)
        score_fold(ds, fold, state, held, out)
        write(
            out / "progress.json", dict(completed_fold=fold, CPU_seconds=time.process_time() - start)
        )
    return ds


def summarize():
    s = STATE
    reference = json.loads((s["ref"] / "summary.json").read_text())["arms"]["high_transient_source"][
        "metrics"
    ]["nmse"]
    results = {}
    for layout in LAYOUTS:
        for arm in READOUTS:
            name = layout + "_" + arm
            stats = defaultdict(list)
            ex = []
            for ds in s["cases"]:
                for fold in range(4):
                    with np.load(s["out"] / ds / f"fold_{fold}_metrics.npz") as z:
                        ex.append(z["examples"])
                        for key in z:
                            if key.startswith(name + "__"):
                                stats[key.split("__")[1]].append(z[key])
            full = base_training_aggregate(
                {k: np.concatenate(v) for k, v in stats.items()}, np.concatenate(ex), s["rows"]
            )
            assert full["inputs"] == 17040 and full["windows"] == 400 and full["subjects"] == 60
            m = full["metrics"]
            n = np.ravel(m["nmse"]["macro"])
            ref = np.ravel(reference["macro"])
            gain = np.ravel(m["rms_gain"]["macro"])
            checks = dict(
                X=n[0] <= 1.05 * ref[0],
                components=np.all(n[1:] <= 1.1 * ref[1:]),
                datasets=all(
                    np.all(np.asarray(v) <= 1.1 * np.asarray(reference["by_dataset"][d]))
                    for d, v in m["nmse"]["by_dataset"].items()
                ),
                gain=np.all((gain >= 0.75) & (gain <= 1.25)),
                bands=all(
                    np.max(abs(np.array(m[k]["macro"]))) <= 3.0103
                    for k in ("band_4_8_db", "band_8_13_db", "band_4_8_abs_db", "band_8_13_abs_db")
                ),
                correlation=min(np.ravel(m["correlation"]["macro"])) > 0,
                envelope=min(np.ravel(m["local_rms_correlation"]["macro"])) >= 0.5,
            )
            results[name] = dict(summary=full, checks=checks, waveform_pass=all(checks.values()))
    write(
        s["out"] / "summary.json",
        dict(
            arms=results,
            waveform_eligible=[
                k for k, v in results.items() if "continuous" not in k and v["waveform_pass"]
            ],
            original_positions=17040,
            people=60,
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["schema"] != "cross_branch_sampling_v1" or cfg["data"] != {
        "fit": "train",
        "score": "train_held_people",
        "development": "forbidden",
        "confirmation": "forbidden",
        "test": "forbidden",
    }:
        raise ValueError("train-only scope required")
    run = ROOT / cfg["joint_run"]
    out = args.output.resolve()
    out.mkdir(exist_ok=True, parents=True)
    lock = (out / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pop = json.loads((run / cfg["population"]).read_text())
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
    identity = dict(
        config=cfg,
        population_sha256=file_sha256(run / cfg["population"]),
        input_identity=pop["cfg_identity"],
        scales_identity=pop["scales_identity"],
    )
    saved = out / "resolved_config.json"
    if saved.exists() and json.loads(saved.read_text()) != identity:
        raise ValueError("changed experiment identity")
    write(saved, identity)
    STATE.update(
        pop=pop,
        cases={
            d: [c for c in pop["cases"] if c["dataset"] == d] for d in sorted(pop["native_positions"])
        },
        rows=manifest["rows"],
        scales=scales,
        out=out,
        source=run / cfg["source"],
        cov=run / cfg["marginal_covariances"],
        dict=run / cfg["dictionaries"],
        ref=run / cfg["waveform_reference"],
    )
    with ProcessPoolExecutor(
        max_workers=cfg["resources"]["workers"], mp_context=multiprocessing.get_context("fork")
    ) as pool:
        list(pool.map(dataset, STATE["cases"]))
    summarize()


if __name__ == "__main__":
    main()
