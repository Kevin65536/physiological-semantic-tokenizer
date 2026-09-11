#!/usr/bin/env python3
"""Render the reproducible visual-review package for an E0-v2 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, run_dir: Path, name: str) -> list[str]:
    figure_dir = run_dir / "figures"
    svg = figure_dir / f"{name}.svg"
    png = figure_dir / f"{name}.png"
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [str(svg.relative_to(run_dir)), str(png.relative_to(run_dir))]


def measurement_alignment(run_dir: Path) -> list[str]:
    data = _load(run_dir / "figure_data" / "measurement_audit.json")
    rows = [row for row in data["summary_rows"] if row["split"] == "validation"]
    datasets = list(data["representative_traces"])
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    for column, space in enumerate(("raw", "canonical")):
        selected = [row for row in rows if row["space"] == space]
        labels = [f"{row['dataset']}\n{row['channel']}" for row in selected]
        values = [row["robust_scale"] for row in selected]
        if space == "canonical":
            joint = data["validation_joint_canonical_scales"]
            labels = list(joint)
            values = list(joint.values())
        axes[0, column].bar(np.arange(len(values)), values, color=[COLORS[i % 4] for i in range(len(values))])
        axes[0, column].set_xticks(np.arange(len(values)), labels, rotation=35, ha="right", fontsize=7)
        axes[0, column].set_yscale("log" if space == "raw" else "linear")
        axes[0, column].set_ylabel("Robust scale" + (" (original unit)" if space == "raw" else " (train robust SD, pooled pair)"))
        axes[0, column].set_title(f"{space.capitalize()} validation fluctuation scale")
        axes[0, column].grid(axis="y", alpha=0.25)
    for index, dataset in enumerate(datasets):
        trace = data["representative_traces"][dataset]
        time = np.asarray(trace["time_s"])
        canonical = np.asarray(trace["canonical"])
        for channel in range(2):
            axes[1, channel].plot(
                time, canonical[:, channel] + index * 8.0,
                color=COLORS[index % len(COLORS)], linewidth=0.8,
                label=dataset if channel == 0 else None,
            )
        axes[1, 0].text(time[0], index * 8.0, dataset, fontsize=7, va="bottom")
    for channel in range(2):
        axes[1, channel].set_xlabel("Time (s)")
        axes[1, channel].set_ylabel(f"Canonical channel {channel} + offset")
        axes[1, channel].set_title(f"Representative canonical traces: channel {channel}")
        axes[1, channel].grid(alpha=0.2)
    return _save(fig, run_dir, "measurement_alignment")


def physical_overlay(run_dir: Path) -> list[str]:
    data = _load(run_dir / "figure_data" / "physical_teacher_overlay.json")
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), constrained_layout=True)
    eeg_time = np.asarray(data["eeg_time_s"])
    axes[0].plot(eeg_time, data["eeg_observed_envelope"], color="#777777", linewidth=0.7, label="Observed EEG envelope")
    axes[0].plot(eeg_time, data["eeg_clean_envelope"], color=COLORS[0], linewidth=1.0, label="Teacher clean envelope")
    axes[0].set_ylabel("Normalized amplitude")
    axes[0].set_title(f"EEG physical observation, subject {data['subject']}")
    axes[0].legend(frameon=False, ncol=2)
    fnirs_time = np.asarray(data["fnirs_time_s"])
    observed = np.asarray(data["fnirs_observed"])
    clean = np.asarray(data["fnirs_clean"])
    for channel in range(2):
        axes[1].plot(fnirs_time, observed[:, channel], color=("#777777", "#BBBBBB")[channel], linewidth=0.8, label=f"Observed ch{channel}")
        axes[1].plot(fnirs_time, clean[:, channel], color=COLORS[channel], linewidth=1.1, label=f"Teacher clean ch{channel}")
    axes[1].set_ylabel("Normalized measurement")
    axes[1].set_title("fNIRS observed and corrected physical prediction")
    axes[1].legend(frameon=False, ncol=4, fontsize=8)
    state = np.asarray(data["state_mean"])
    labels = ("s", "delta_f", "delta_HbO", "delta_Hb", "r")
    for index, label in enumerate(labels):
        values = state[:, index]
        robust = np.median(np.abs(values - np.median(values))) * 1.4826
        normalized = (values - np.median(values)) / max(robust, 1e-8)
        axes[2].plot(fnirs_time, normalized + index * 5.0, color=COLORS[index], linewidth=0.9, label=label)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Robust z-score + offset")
    axes[2].set_title("Extracted shared neural/hemodynamic state")
    axes[2].legend(frameon=False, ncol=5)
    for axis in axes:
        axis.grid(alpha=0.2)
    return _save(fig, run_dir, "physical_teacher_overlay")


def target_observability(run_dir: Path) -> list[str]:
    data = _load(run_dir / "figure_data" / "target_observability.json")
    rows = data["rows"]
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9), constrained_layout=True)
    labels = [f"{row['modality']}:{row['coordinate']}" for row in rows]
    values = np.asarray([row["validation_r2"] for row in rows])
    threshold = np.asarray([row["permutation_q"] for row in rows])
    x = np.arange(len(rows))
    axes[0].bar(x, values, color=[COLORS[0] if row["admitted_local_target"] else "#999999" for row in rows], label="Validation R2")
    axes[0].scatter(x, threshold, color=COLORS[4], marker="_", s=180, label="Permutation q95")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, labels, rotation=40, ha="right", fontsize=8)
    axes[0].set_ylabel("R2 against train-mean baseline")
    axes[0].set_title("Patch-local teacher target observability")
    axes[0].legend(frameon=False)
    admitted_fnirs = next(
        (row for row in rows if row["admitted_local_target"] and row["modality"] == "fnirs"),
        None,
    )
    trace_key = (
        f"{admitted_fnirs['modality']}:{admitted_fnirs['coordinate']}"
        if admitted_fnirs is not None
        else next(iter(data["traces"]))
    )
    trace = data["traces"][trace_key]
    target = np.asarray(trace["target"])
    prediction = np.asarray(trace["prediction"])
    axes[1].scatter(target, prediction, s=12, alpha=0.35, color=COLORS[2], edgecolors="none")
    lower = float(min(target.min(), prediction.min()))
    upper = float(max(target.max(), prediction.max()))
    axes[1].plot([lower, upper], [lower, upper], color="#555555", linestyle="--", linewidth=1, label="Identity")
    axes[1].set_xlabel(f"Teacher target: {trace_key}")
    axes[1].set_ylabel("Patch-local prediction")
    axes[1].set_title("Held-out target versus prediction")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    return _save(fig, run_dir, "target_observability")


def gauge_alignment(run_dir: Path) -> list[str]:
    data = _load(run_dir / "figure_data" / "gauge_alignment.json")
    rows = [row for row in data["gain_rows"] if row["modality"] == "fnirs"]
    folds = data["fold_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    labels = [row["coordinate"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    axes[0].bar(x - width / 2, [row["r2_pre_gauge"] for row in rows], width, color="#999999", label="Raw latent")
    axes[0].bar(x + width / 2, [row["r2_post_gauge"] for row in rows], width, color=COLORS[2], label="Gauge corrected")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("Held-out patch-local R2")
    axes[0].set_title("Target observability before and after gauge")
    axes[0].legend(frameon=False)

    for offset, (coordinate, color) in enumerate((("hbo", COLORS[0]), ("hbr", COLORS[1]))):
        scales = np.asarray([row[f"{coordinate}_scale"] for row in folds], dtype=float)
        index = np.arange(len(scales), dtype=float) + (offset - 0.5) * 0.18
        axes[1].scatter(
            index, np.log10(np.maximum(np.abs(scales), 1e-12)),
            c=np.where(scales >= 0.0, color, COLORS[4]), s=13, alpha=0.65,
            label=f"{coordinate.upper()} abs scale; red=negative",
        )
    axes[1].set_xlabel("Train/validation leave-one-trial fold")
    axes[1].set_ylabel("log10 |state-to-measurement scale|")
    axes[1].set_title("Train-fold gauge scales and sign orientation")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)
    axes[0].grid(alpha=0.2)
    fig.suptitle(data.get("mode", "target gauge alignment"), fontsize=12)
    return _save(fig, run_dir, "gauge_alignment")


def uncertainty_calibration(run_dir: Path) -> list[str]:
    rows = _load(run_dir / "figure_data" / "target_observability.json")["rows"]
    synthetic = _load(run_dir / "figure_data" / "posterior_calibration.json")["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    labels = [row["coordinate"] for row in synthetic]
    unscaled = [row["unscaled_90_coverage"] for row in synthetic]
    coverage = [row["scaled_90_coverage"] for row in synthetic]
    y = np.arange(len(labels))
    axes[0].barh(y - 0.18, unscaled, height=0.34, color="#999999", label="Unscaled")
    axes[0].barh(
        y + 0.18, coverage, height=0.34,
        color=[COLORS[2] if row["calibrated_coverage_pass"] else COLORS[4] for row in synthetic],
        label="Calibrated",
    )
    axes[0].axvline(0.9, color=COLORS[4], linestyle="--", label="Nominal 90%")
    tolerance = synthetic[0]["coverage_tolerance_95"]
    axes[0].axvspan(0.9 - tolerance, 0.9 + tolerance, color=COLORS[4], alpha=0.08, label="Binomial 95% band")
    axes[0].set_yticks(y, labels, fontsize=7)
    axes[0].set_xlabel("Empirical interval coverage")
    axes[0].set_title("Synthetic-truth posterior calibration")
    axes[0].legend(frameon=False)
    diagnostic_labels = [f"{row['modality']}:{row['coordinate']}" for row in rows]
    standardized = [row["standardized_rmse"] for row in rows]
    y = np.arange(len(diagnostic_labels))
    axes[1].barh(y, standardized, color=COLORS[1])
    axes[1].set_yticks(y, diagnostic_labels, fontsize=7)
    axes[1].set_xlabel("RMSE / posterior SD")
    axes[1].set_title("Student error / teacher SD (diagnostic only)")
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
    return _save(fig, run_dir, "uncertainty_calibration")


def vocabulary(run_dir: Path) -> list[str]:
    data = _load(run_dir / "figure_data" / "vocabulary_transmissibility.json")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    for row_index, modality in enumerate(("eeg", "fnirs")):
        plot = data["plot"][modality]
        coordinates = np.asarray(plot["pca"])
        codes = np.asarray(plot["codes"])
        axes[row_index, 0].scatter(coordinates[:, 0], coordinates[:, 1], c=codes, cmap="turbo", s=5, alpha=0.55, rasterized=True)
        axes[row_index, 0].set_xlabel("Teacher target PC1")
        axes[row_index, 0].set_ylabel("Teacher target PC2")
        axes[row_index, 0].set_title(f"{modality.upper()} K=128 Voronoi assignment")
        occupancy = np.asarray(plot["occupancy"])
        axes[row_index, 1].bar(np.arange(len(occupancy)), occupancy, color=COLORS[row_index])
        axes[row_index, 1].set_xlabel("Code index")
        axes[row_index, 1].set_ylabel("Validation assignments")
        axes[row_index, 1].set_title(f"{modality.upper()} occupancy")
        axes[row_index, 1].grid(axis="y", alpha=0.2)
    return _save(fig, run_dir, "vocabulary_transmissibility")


def coupling(run_dir: Path) -> list[str]:
    data = _load(run_dir / "figure_data" / "continuous_coupling_upper_bound.json")
    rows = [row for row in data["rows"] if row["coordinate"] != "joint_logdet"]
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.5), constrained_layout=True)
    target_labels = {"fnirs_innovation": "predictive residual"}
    labels = [f"{target_labels.get(row['target'], row['target'].replace('fnirs_', ''))}:{row['coordinate']}" for row in rows]
    values = [row["incremental_r2"] for row in rows]
    axes[0].bar(np.arange(len(rows)), values, color=[COLORS[0] if value >= 0 else COLORS[4] for value in values])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(np.arange(len(rows)), labels, rotation=30, ha="right")
    axes[0].set_ylabel("Incremental R2 from EEG history")
    axes[0].set_title("Continuous coupling upper bound by target")
    trace = data["traces"]["fnirs_innovation"]
    target = np.asarray(trace["target"])
    baseline = np.asarray(trace["fnirs_history_prediction"])
    full = np.asarray(trace["plus_eeg_history_prediction"])
    axes[1].scatter(target[:, 0], baseline[:, 0], color=COLORS[1], s=10, alpha=0.3, label="fNIRS history")
    axes[1].scatter(target[:, 0], full[:, 0], color=COLORS[0], s=10, alpha=0.3, label="fNIRS + EEG history")
    lower = float(min(target[:, 0].min(), baseline[:, 0].min(), full[:, 0].min()))
    upper = float(max(target[:, 0].max(), baseline[:, 0].max(), full[:, 0].max()))
    axes[1].plot([lower, upper], [lower, upper], color="#555555", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Teacher delta_f predictive residual")
    axes[1].set_ylabel("Held-out prediction")
    axes[1].set_title("Conditional prediction with and without EEG history")
    axes[1].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.2)
    return _save(fig, run_dir, "continuous_coupling_upper_bound")


def mask_coverage(run_dir: Path) -> list[str]:
    with (run_dir / "teacher_mask_coverage.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    subjects = [int(row["subject"]) for row in rows]
    local = [float(row["local_coverage"]) for row in rows]
    context = [float(row["context_coverage"]) for row in rows]
    fig, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    axis.scatter(np.asarray(subjects) - 0.06, local, color=COLORS[2], s=22, alpha=0.75, label="Local state")
    axis.scatter(np.asarray(subjects) + 0.06, context, color=COLORS[1], s=22, alpha=0.75, label="10 s context")
    axis.set_xlabel("Validation subject")
    axis.set_ylabel("Valid patch fraction")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Teacher support mask coverage")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    return _save(fig, run_dir, "teacher_mask_coverage")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    summary = _load(run_dir / "summary.json")
    review_path = run_dir / "visual_review.yaml"
    review = yaml.safe_load(review_path.read_text(encoding="utf-8")) if review_path.is_file() else {}
    builders: list[tuple[str, Callable[[Path], list[str]], str]] = [
        ("measurement_alignment", measurement_alignment, "figure_data/measurement_audit.json"),
        ("physical_teacher_overlay", physical_overlay, "figure_data/physical_teacher_overlay.json"),
        ("target_observability", target_observability, "figure_data/target_observability.json"),
        ("uncertainty_calibration", uncertainty_calibration, "figure_data/posterior_calibration.json"),
        ("vocabulary_transmissibility", vocabulary, "figure_data/vocabulary_transmissibility.json"),
        ("continuous_coupling_upper_bound", coupling, "figure_data/continuous_coupling_upper_bound.json"),
        ("teacher_mask_coverage", mask_coverage, "teacher_mask_coverage.csv"),
    ]
    if (run_dir / "figure_data" / "gauge_alignment.json").is_file():
        builders.insert(3, ("gauge_alignment", gauge_alignment, "figure_data/gauge_alignment.json"))
    entries = []
    for name, builder, source in builders:
        paths = builder(run_dir)
        review_status = review.get("figures", {}).get(name, {}).get("status", "pending")
        entries.append({"name": name, "source_data": source, "artifacts": paths, "review_status": review_status})
    for entry in entries:
        entry["sha256"] = {
            path: hashlib.sha256((run_dir / path).read_bytes()).hexdigest() for path in entry["artifacts"]
        }
    audit = {
        "schema": f"{summary.get('schema', 'physiology_semantic_e0_v2')}_visual_audit",
        "figures": entries,
        "review_checklist": [
            "No unit or semantics label contradicts source metadata.",
            "Canonical scaling does not hide clipping, discontinuities, or failed channels.",
            "Teacher clean traces track plausible observed structure without copying residual noise.",
            "Shared neural/hemodynamic states show no solver discontinuity or boundary artifact.",
            "Observability is not driven by a single subject or monotonic ordering artifact.",
            "Any target gauge uses training-fold parameters, is finite/non-singular, and leaves physical reconstruction invariant.",
            "Vocabulary geometry has no severe collapse hidden by aggregate R2.",
            "EEG-history coupling gain is visible in transition traces and not only a logdet scalar.",
            "Teacher masks exclude unsupported intervals and do not erase a specific subject.",
        ],
        "protected_test_may_open": bool(
            review.get("overall_pass", False)
            and summary.get("validation", {}).get("machine_validation_pass", False)
        ),
    }
    (run_dir / "visual_review.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
