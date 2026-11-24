"""Cross-fit a single EEG word's background-relative attribute profile on the frozen AS training bank."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.analysis.tokenizer_pattern_audit import feature_means
from src.data.dual_component_tokenizer import base_tokenizer_training_data

FIELDS = {"theta": (0, 1), "alpha": (1, 2), "local_rms": (2, 10)}
STATE = {}


def relative_word_readout(
    fit_words,
    background,
    values,
    held_words,
    held_background,
    *,
    vocabulary,
    prior,
    subjects,
    seed,
    loss="relative",
):
    """Fit a background mean and one residual profile per distinct (role, word)."""
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("finite nonnegative power and RMS fields required")
    nbg = max(int(background.max()), int(held_background.max())) + 1
    counts = np.bincount(background, minlength=nbg)
    means = np.zeros((nbg, values.shape[1]))
    np.add.at(means, background, values)
    global_mean = values.mean(axis=0)
    means = np.divide(
        means,
        counts[:, None],
        out=np.tile(global_mean, (nbg, 1)),
        where=counts[:, None] > 0,
    )
    scale = np.maximum(means, np.maximum(1e-6 * global_mean, 1e-20))
    if loss not in {"relative", "absolute"}:
        raise ValueError("word loss must be relative or absolute")
    residual = values - means[background]
    base = means[held_background]
    rng = np.random.default_rng(seed)
    # One permutation per subject and background retains each group's original word histogram.
    group = subjects.astype(np.int64) * nbg + background
    order = np.argsort(group, kind="stable")
    shuffled = fit_words.copy()
    for chunk in np.split(order, np.flatnonzero(np.diff(group[order])) + 1):
        shuffled[chunk] = rng.permutation(fit_words[chunk])
    outputs = [base]
    for labels in (fit_words, shuffled):
        word_count = np.bincount(labels, minlength=vocabulary)
        totals = np.zeros((vocabulary, values.shape[1]))
        design = scale[background]
        if loss == "absolute":
            # Least-squares multiplier in the original attribute coordinate.
            np.add.at(totals, labels, residual * design)
            power = np.zeros_like(totals)
            np.add.at(power, labels, design**2)
            correction = np.divide(
                totals, power, out=np.zeros_like(totals), where=power > 0
            )
            correction *= word_count[:, None] / (word_count[:, None] + prior)
        else:
            np.add.at(totals, labels, residual / design)
            correction = totals / (word_count[:, None] + prior)
        correction[::8192] = 0.0
        outputs.append(
            np.maximum(0, base + scale[held_background] * correction[held_words])
        )
    return outputs


def run_fold(task):
    branch, fold = task
    bank, spec = STATE["bank"], STATE["spec"]
    n = spec["data"]["original_score_windows"]
    valid = bank["valid"]
    fit = (bank["fold"][:, None] != fold) & valid
    held = np.flatnonzero(bank["fold"][:n] == fold)
    if set(bank["subject"][held]) & set(bank["subject"][np.any(fit, axis=1)]):
        raise ValueError("subject leakage")
    k = spec["data"]["vocabulary"]
    words = k * bank["fixed_band"][:, None] + bank[branch]
    if np.any((bank[branch] < 0) | (bank[branch] >= k)) or np.any(
        (bank["fixed_band"] < 0) | (bank["fixed_band"] > 4)
    ):
        raise ValueError("word outside declared role vocabulary")
    held_mask = valid[held]
    targets = bank[branch + "_attributes"]
    predictions = relative_word_readout(
        words[fit],
        bank["background"][fit],
        targets[fit],
        words[held][held_mask],
        bank["background"][held][held_mask],
        vocabulary=5 * k,
        prior=spec["method"]["prior_zero_residual_patches"],
        subjects=np.broadcast_to(bank["subject"][:, None], valid.shape)[fit],
        seed=spec["method"]["seed"] + fold,
        loss=spec["method"].get("loss", "relative"),
    )
    errors = []
    for prediction in predictions:
        losses = np.zeros((*held_mask.shape, 3), float)
        losses[held_mask] = feature_means(
            (prediction - targets[held][held_mask]) ** 2, FIELDS
        )
        err = losses.sum(1) / held_mask.sum(1).clip(1)[:, None]
        err[~held_mask.any(1)] = np.nan
        errors.append(err)
    return branch, held, np.stack(errors, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    src = cfg["data"]
    if src["allowed_splits"] != ["train"]:
        raise ValueError("this diagnostic is training-only")
    for path, sha in [("input", "sha256"), ("matched_errors", "matched_sha256")]:
        with (ROOT / src[path]).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != src[sha]:
                raise ValueError("versioned training bank identity changed")
    manifest, _, _, _ = base_tokenizer_training_data(
        yaml.safe_load((ROOT / src["g0_config"]).read_text()), ROOT
    )
    rows = manifest["rows"]
    with np.load(ROOT / src["input"]) as saved:
        examples = saved["examples"]
        if len(examples) != src["fit_bank_windows"] or any(
            rows[i]["split"] != "train" or rows[i]["modality"] != "eeg"
            for i in np.unique(examples[:, 0])
        ):
            raise ValueError("expected only frozen EEG training rows")
        bank = {k: saved[k] for k in saved.files if k != "examples"}
    n = src["original_score_windows"]
    with np.load(ROOT / src["matched_errors"]) as prior:
        np.testing.assert_array_equal(prior["examples"], examples[:n])
        raw = {b: prior[b][:, 1] for b in "SO"}
    if len(np.unique(bank["subject"])) != src["subjects"] or not np.array_equal(
        np.unique(bank["fold"]), np.arange(src["subject_folds"])
    ):
        raise ValueError("original fold support changed")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise FileExistsError("finished diagnostic exists")
    STATE.update(bank=bank, spec=cfg)
    results = {b: np.full((n, 4, 3), np.nan) for b in "SO"}
    for b in "SO":
        results[b][:, :2] = raw[b]
    with ProcessPoolExecutor(
        max_workers=cfg["resources"]["dataset_fitting_workers"],
        mp_context=multiprocessing.get_context("fork"),
    ) as pool:
        for b, idx, err in pool.map(
            run_fold, [(b, f) for b in "SO" for f in range(src["subject_folds"])]
        ):
            np.testing.assert_allclose(
                err[:, 0], raw[b][idx, 0], rtol=2e-12, atol=1e-14
            )
            results[b][idx, 2:] = err[:, 1:]
    np.savez_compressed(
        args.output / "paired_training_errors.npz",
        examples=examples[:n],
        role=bank["fixed_band"][:n],
        **results,
    )
    report = {}
    for b in "SO":
        record = {}
        for k in ("all", 0, 1, 2, 3, 4):
            use = np.ones(n, bool) if k == "all" else bank["fixed_band"][:n] == k
            stats = base_training_aggregate(
                {"nmse": results[b][use]}, examples[:n][use], rows
            )["metrics"]["nmse"]
            macro = np.array(stats["macro"])
            people = np.array([item["value"] for item in stats["by_subject"]])
            record[str(k)] = dict(
                n=int(use.sum()),
                people=len(people),
                gain=(1 - macro[1:] / macro[:1]).tolist(),
                positive_people=np.sum(people[:, 1:] < people[:, :1], axis=0).tolist(),
                hierarchy=stats,
            )
        report[b] = record
    output = dict(
        frozen_config=cfg,
        arm_order=[
            "background",
            "matched_raw_cell",
            "normalized_pooled_word",
            "within_subject_background_shuffle",
        ],
        attributes=list(FIELDS),
        bands_hz=[[1, 4], [4, 8], [8, 13], [13, 30], [30, 45]],
        branches=report,
    )
    (args.output / "summary.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print({k: v["all"]["gain"] for k, v in report.items()})


if __name__ == "__main__":
    main()
