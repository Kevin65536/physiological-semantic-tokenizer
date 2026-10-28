#!/usr/bin/env python3
"""Render retained Tak mismatch evidence without opening measured datasets.

Use ``--run <completed-run> --output <fresh-export>``. The report reads only
summary.json, metrics.csv, subject_summary.csv, examples.json and the explicit
example/weak array paths listed there, plus an optional owning
weak_verification.json and terminal_status_audit.json. Scientific scores and fitting remain
owned by the evaluator and ``src.inference.tak_mismatch``. All figures and
LaTeX equations are bitmap images in both PDFs; Chinese prose, tables and
captions remain searchable through the shared report renderer.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.scripts.render_source_diagnostics_report import render_report

SCHEMA = "tak_mismatch_report_v1"
DATASET_LABELS = {
    "eeg_fnirs_single_trial": "单试次 EEG–fNIRS",
    "simultaneous_eeg_nirs": "同时 EEG–NIRS",
    "visual_cognitive_motivation": "视觉认知与动机",
    "refed": "ReFEd",
    "shin2017": "Shin 2017", "shin2018": "Shin 2018",
    "hbn": "HBN", "bci2023": "BCI 2023", "bci2024": "BCI 2024",
}
MODE_LABELS = {"fixed": "固定参数", "current_dof": "组级 κ、τ 自由度"}
METRICS = {
    "D1": ("Hb-only 非线性失配", "训练 SD", "低值表示不看 EEG 时更容易重建 Hb。"),
    "D_bal_10s": ("10 秒弱动态平衡偏离", "无量纲", "低值表示小信号动态关系更一致；不是失配概率。"),
    "D_bal_20s": ("20 秒弱动态平衡偏离", "无量纲", "低值表示小信号动态关系更一致；不是失配概率。"),
    "D_bal_40s": ("40 秒弱动态平衡偏离", "无量纲", "低值表示小信号动态关系更一致；不是失配概率。"),
    "D_EH": ("EEG 与 Hb 条件驱动读出差异", "EEG 训练 SD", "低值表示 Hb-only 驱动的 EEG 读出更接近 EEG 代理。"),
    "D_obs": ("Hb 可观测驱动差异", "Hb 训练 SD", "固定量与组级解处的逐窗放松量使用不同 nuisance 空间；下降不解释为组共享模型改善。依赖预设噪声情景，零保留秩不可用，不证明生理驱动相同。"),
    "D_drive_H": ("共同初态的非线性驱动回放差异", "Hb 训练 SD", "低值表示两种驱动在共同初态下产生更接近的 Hb。"),
    "Delta_spatial": ("替代六通道相对当前近邻的预测收益", "Hb 训练 SD 平方", "正值表示预先选定的替代六通道降低 Hb 均方误差；负值表示当前近邻更好。"),
    "D4": ("联合模型的三坐标相容性", "训练 SD", "低值表示三种观测的等权均方根残差较小，仍须核对三个分量。"),
    "D_J_E": ("联合 EEG 残差", "EEG 训练 SD", "低值表示 EEG 代理重建更好。"),
    "D_J_O": ("联合 HbO 残差", "HbO 训练 SD", "低值表示 HbO 重建更好。"),
    "D_J_R": ("联合 HbR 残差", "HbR 训练 SD", "低值表示 HbR 重建更好。"),
}
COLORS = {"target": "#202c39", "hb": "#226ea8", "joint": "#ac4470", "eeg": "#d17d25",
          "spatial": "#278579", "wrong": "#8d8d8d", "free_hb": "#269bb2", "free_joint": "#d272a1", "fixed": "#246eab", "current_dof": "#af4571"}
LINESTYLES = {"target": "-", "hb": "--", "joint": "-", "eeg": ":", "free_hb": "-.",
              "free_joint": (0, (7, 2, 2, 2)), "spatial": "-", "wrong": ":",
              "fixed": "-", "current_dof": (0, (6, 2)), "time_fixed": ":", "time_current_dof": "-."}
ALIASES = {"parameter_mode": ("parameter_mode", "mode"), "task_id": ("task_id", "id"),
           "weight_s": ("time_weight_s", "duration_weight_s", "nonoverlap_duration_s", "weight_s"),
           "D_obs": ("D_obs", "D_obs_noise_scenario")}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _num(value, digits=3):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "不可用"
    return f"{value:.{digits}f}" if math.isfinite(value) else "不可用"


def _path(run, value):
    """Resolve one declared artifact without allowing archive/data traversal."""
    path = (run / str(value)).resolve()
    if not path.is_relative_to(run):
        raise ValueError(f"report evidence must stay inside the completed run: {value}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _rename(frame, default_mode="fixed"):
    for target, names in ALIASES.items():
        if target not in frame:
            source = next((name for name in names if name in frame), None)
            if source:
                frame = frame.rename(columns={source: target})
    if "parameter_mode" not in frame:
        frame["parameter_mode"] = default_mode
    return frame


def load_evidence(run):
    """Read explicitly named, terminal evidence; never discover raw/cache roots."""
    import pandas as pd

    run = Path(run).resolve()
    summary = _json(run / "summary.json")
    if summary.get("status") != "completed":
        raise ValueError("report requires a completed summary; partial runs cannot be presented as full evidence")
    files = summary.get("files", {})
    metric_path = _path(run, files.get("metrics", "metrics.csv"))
    with metric_path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    keep = {"dataset", "subject", "record", "pair", "window", "status", "success", "reason",
            "failure_reason", "parameter_mode", "mode", "task_id", "id", "start_s", "end_s", "fold",
            "HbO_RMSE_native", "HbR_RMSE_native", "D_obs_rank", "numerical_status", "spatial_status",
            "hb_only_success", "eeg_only_success", "joint_success", "record_id", "duration_s",
            "hb_only_status", "eeg_only_status", "joint_status",
            "prediction_NMSE", "selected_prediction_NMSE", "zero_NMSE", "template_NMSE", "wrong_NMSE",
            "selected_wrong_NMSE", "pairing_gain", "selected_pairing_gain", "selected_candidate",
            "array_path", "array_index", "group_id", "fixed_task_id", "weak_pair_index",
            "hb_only_parameter_boundary", "joint_parameter_boundary"}
    keep.update(METRICS)
    keep.update(name for names in ALIASES.values() for name in names)
    keep.update(name for name in header if name.startswith("time_weight_") and name.endswith("_s"))
    keep.update(name for name in header if name.startswith("D_obs") or name.startswith("observable_")
                or name in ("retained_rank", "numerical_rank"))
    metric_columns = [name for name in header if name in keep]
    rows = _rename(pd.read_csv(metric_path, usecols=metric_columns, low_memory=False), summary.get("parameter_mode", "fixed"))
    if not {"dataset", "subject", "D1"} <= set(rows):
        raise ValueError("metrics.csv requires dataset, subject and D1")
    for name in METRICS:
        if name in rows:
            rows[name] = pd.to_numeric(rows[name], errors="coerce")
    for weight in [name for name in rows if name == "weight_s" or (name.startswith("time_weight_") and name.endswith("_s"))]:
        rows[weight] = pd.to_numeric(rows[weight], errors="coerce")
        if ((rows[weight] < 0) | ~np.isfinite(rows[weight])).any():
            raise ValueError("temporal distribution weights must be finite and nonnegative")
    subject_path = _path(run, files.get("subjects", "subject_summary.csv"))
    subjects = _rename(pd.read_csv(subject_path), summary.get("parameter_mode", "fixed"))
    if not {"dataset", "subject", "metric", "value"} <= set(subjects):
        raise ValueError("subject_summary.csv requires dataset, subject, metric and value")
    subjects["value"] = pd.to_numeric(subjects["value"], errors="coerce")
    if subjects.duplicated(["dataset", "subject", "parameter_mode", "metric"]).any():
        raise ValueError("subject_summary.csv must contain one value per subject, metric and parameter mode")
    current_rows = pd.DataFrame()
    current_metric_name = files.get("metrics_current_dof", files.get("current_dof_metrics", "metrics_current_dof.csv"))
    current_subject_name = files.get("subjects_current_dof", files.get("current_dof_subjects", "subject_summary_current_dof.csv"))
    required_current = summary.get("schema") == "tak_mismatch_summary_v1"
    has_current = (run/current_metric_name).is_file() or (run/current_subject_name).is_file()
    if required_current or has_current:
        current_summary = summary.get("current_dof") or _json(_path(run, files.get("current_dof_summary", "current_dof/summary.json")))
        if (current_summary.get("status") != "completed" or
                current_summary.get("planned_groups") != current_summary.get("terminal_groups") or
                current_summary.get("planned_windows") != current_summary.get("terminal_windows")):
            raise ValueError("full report requires completed current_dof group evidence")
        current_rows = pd.read_csv(_path(run, current_metric_name), low_memory=False)
        for metric in METRICS:
            if metric+"_free" in current_rows:
                current_rows[metric] = pd.to_numeric(current_rows[metric+"_free"], errors="coerce")
        for name in list(current_rows):
            if name.startswith("time_weight_") and name.endswith("_free_s"):
                current_rows[name.removesuffix("_free_s")+"_s"] = current_rows[name]
        current_rows = _rename(current_rows, "current_dof")
        current_rows["parameter_mode"] = "current_dof"
        if "fixed_task_id" in current_rows:
            current_rows["task_id"] = current_rows["fixed_task_id"]
        current_subjects = _rename(pd.read_csv(_path(run, current_subject_name)), "current_dof")
        current_subjects["parameter_mode"] = "current_dof"
        current_subjects["metric"] = current_subjects.metric.str.removesuffix("_free")
        current_subjects["value"] = pd.to_numeric(current_subjects["value"], errors="coerce")
        subjects = pd.concat((subjects, current_subjects), ignore_index=True)
        rows = pd.concat((rows, current_rows), ignore_index=True)
        if subjects.duplicated(["dataset", "subject", "parameter_mode", "metric"]).any():
            raise ValueError("duplicate current_dof subject statistic")
    example_path = _path(run, files.get("examples", "examples.json"))
    example_index = _json(example_path)
    examples = example_index.get("examples", [])
    if not isinstance(examples, list):
        raise ValueError("examples.json examples must be a list")
    for example in examples:
        for name in ("dataset", "subject", "record", "pair", "window", "role", "array_path"):
            if name not in example:
                raise ValueError(f"example missing {name}")
        _path(run, example["array_path"])
        if example.get("weak_path"):
            _path(run, example["weak_path"])
    planned = summary.get("planned")
    terminal = summary.get("terminal")
    if isinstance(planned, int) and isinstance(terminal, int) and terminal != planned:
        raise ValueError("completed summary must retain all planned terminal states")
    plan = _json(_path(run, files["plan"])) if files.get("plan") else {}
    config_name = files.get("config", "resolved_config.json")
    config = _json(_path(run, config_name)) if (run/config_name).is_file() else {}
    verification_name = files.get("weak_verification", "weak_verification.json")
    verification_path = _path(run, verification_name) if (run/verification_name).is_file() else None
    verification = _json(verification_path) if verification_path else {}
    if verification and verification.get("schema") != "tak_mismatch_weak_independent_verification_v1":
        raise ValueError("unsupported weak verification evidence schema")
    hb_provenance = []
    for record in verification.get("records", []):
        if not record.get("prepared_metadata_path"):
            continue
        metadata_path = _path(run, record["prepared_metadata_path"])
        provenance = _json(metadata_path).get("metadata", {}).get("hb_provenance", {})
        if provenance:
            hb_provenance.append(dict(dataset=record["dataset"], record=record.get("record"),
                                      metadata_path=str(metadata_path), provenance=provenance))
    terminal_name = files.get("terminal_status_audit", "terminal_status_audit.json")
    terminal_path = _path(run, terminal_name) if (run/terminal_name).is_file() else None
    terminal_audit = _json(terminal_path) if terminal_path else {}
    if terminal_audit and terminal_audit.get("schema") != "tak_mismatch_terminal_status_audit_v1":
        raise ValueError("unsupported terminal status audit evidence schema")
    return dict(run=run, summary=summary, rows=rows, subjects=subjects, examples=examples,
                example_index=example_index, metric_path=metric_path, subject_path=subject_path,
                example_path=example_path, inventory=plan.get("summary", {}), current_rows=current_rows, config=config,
                plan_record_reasons={str(record["id"]): record["excluded_reason"] for record in plan.get("records", [])
                                     if record.get("id") and record.get("excluded_reason")},
                plan_path=_path(run, files["plan"]) if files.get("plan") else None,
                weak_verification=verification, weak_verification_path=verification_path,
                terminal_audit=terminal_audit, terminal_audit_path=terminal_path, hb_provenance=hb_provenance)


def weighted_ecdf(values, weights=None):
    """Plotting transform only; scientific aggregation is read from the run."""
    values = np.asarray(values, float)
    weights = np.ones(values.shape) if weights is None else np.asarray(weights, float)
    if values.shape != weights.shape:
        raise ValueError("ECDF values and weights must share shape")
    if np.any(np.isfinite(weights) & (weights < 0)):
        raise ValueError("ECDF weights must be nonnegative")
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.array([]), np.array([])
    order = np.argsort(values[mask], kind="stable")
    x, w = values[mask][order], weights[mask][order]
    return x, np.cumsum(w) / w.sum()


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import font_manager
    candidates = ("Noto Sans CJK SC", "WenQuanYi Micro Hei", "Noto Sans CJK JP", "DejaVu Sans")
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in installed), "DejaVu Sans")
    plt.rcParams.update({"font.family": selected, "axes.unicode_minus": False,
                         "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _save(fig, path):
    fig.savefig(path, dpi=250, facecolor="white", bbox_inches="tight")
    _plotting().close(fig)
    return str(path)


def _datasets(evidence):
    return sorted(set(evidence["rows"]["dataset"].astype(str)) | set(evidence["subjects"]["dataset"].astype(str)))


def _subject_values(evidence, dataset, metric, mode="fixed"):
    frame = evidence["subjects"]
    part = frame[(frame.dataset.astype(str) == dataset) & (frame.metric == metric) &
                 (frame.parameter_mode == mode)]
    return part.loc[np.isfinite(part.value), "value"].to_numpy(float)


def distribution_figure(evidence, metric, destination):
    """Compare datasets on one shared x scale; subjects are the main ECDF."""
    plt = _plotting()
    datasets = _datasets(evidence)
    count = len(datasets)
    fig, axes = plt.subplots(math.ceil(count/2), 2, figsize=(8, 2.8*math.ceil(count/2)),
                             squeeze=False, sharex=True, sharey=True)
    all_values = []
    for ax, dataset in zip(axes.flat, datasets):
        for mode in ("fixed", "current_dof"):
            values = _subject_values(evidence, dataset, metric, mode)
            x, y = weighted_ecdf(values)
            mode_label = "组级解处逐窗放松" if metric == "D_obs" and mode == "current_dof" else MODE_LABELS[mode]
            if len(x):
                ax.step(x, y, where="post", color=COLORS[mode], lw=1.8, ls=LINESTYLES[mode],
                        label=mode_label+f"；被试 n={len(x)}")
                all_values.extend(x.tolist())
            frame = evidence["rows"]
            weight = "time_weight_"+metric+"_s"
            if weight not in frame and metric == "D1":
                weight = "weight_s"
            # A repeated record-level weak score is never a local-time sample.
            if metric in frame and weight in frame and not metric.startswith("D_bal_"):
                part = frame[(frame.dataset.astype(str) == dataset) & (frame.parameter_mode == mode)]
                if part.empty:
                    continue
                x, y = weighted_ecdf(part[metric].to_numpy(float), part[weight].to_numpy(float))
                if len(x):
                    ax.step(x, y, where="post", color=COLORS[mode], lw=1.0, ls=LINESTYLES["time_"+mode], alpha=.75,
                            label=mode_label+"；有效时间加权")
                    all_values.extend(x.tolist())
        if not ax.lines:
            ax.text(.5, .5, "本指标无有效支持\n缺失保留为不可用", ha="center", va="center", transform=ax.transAxes)
        if metric == "Delta_spatial":
            ax.axvline(0, color="#555", lw=.8)
        ax.set(title=DATASET_LABELS.get(dataset, dataset), xlabel=METRICS[metric][1], ylabel="累计比例")
        ax.set_ylim(0, 1.04)
        ax.grid(alpha=.18)
        if ax.lines and ax.get_legend_handles_labels()[1]:
            ax.legend(loc="lower right", frameon=False)
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    if all_values:
        lower, upper = min(all_values), max(all_values)
        margin = max((upper-lower)*.03, abs(upper)*.02, .001)
        axes.flat[0].set_xlim(lower-margin, upper+margin)
    fig.suptitle(METRICS[metric][0]+"：被试等权为主，时间加权为辅", fontsize=12)
    fig.tight_layout()
    return _save(fig, destination)


def _array_data(evidence, example):
    with np.load(_path(evidence["run"], example["array_path"]), allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = ("time_s", "target", "hb_only_prediction", "eeg_only_prediction", "joint_prediction", "score_mask", "sd")
    if any(key not in arrays for key in required):
        raise ValueError(f"example arrays missing required fields: {example['array_path']}")
    time = arrays["time_s"]
    shape = (len(time), 3)
    if time.ndim != 1 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("example time_s must be finite and strictly increasing")
    for key in ("target", "hb_only_prediction", "eeg_only_prediction", "joint_prediction", "score_mask"):
        if arrays[key].shape != shape:
            raise ValueError(f"{key} must have shape [T,3]")
    if (arrays["score_mask"].dtype != bool or arrays["sd"].shape != (3,)
            or not np.isfinite(arrays["sd"]).all() or np.any(arrays["sd"] <= 0)):
        raise ValueError("example requires boolean score mask and positive [3] frozen SD")
    if not np.isfinite(arrays["target"][arrays["score_mask"]]).all():
        raise ValueError("scored example targets must be finite")
    if "hb_factor" in arrays:
        factor = arrays["hb_factor"]
        if factor.ndim != 0 or not np.isfinite(factor) or factor <= 0:
            raise ValueError("hb_factor must be one positive finite common factor")
    return arrays


def _attach_current_example(evidence, example, arrays):
    rows = evidence.get("current_rows")
    if rows is None or rows.empty:
        return arrays, {}
    identity = str(example.get("task_id", example.get("id", "")))
    part = rows[rows.task_id.astype(str) == identity]
    if len(part) != 1:
        raise ValueError(f"fixed example requires one current_dof terminal row: {identity}")
    row = part.iloc[0].to_dict()
    if not isinstance(row.get("array_path"), str) or not row["array_path"]:
        return arrays, row
    index = int(row["array_index"])
    names = {"hb_only_prediction": "free_hb_only_prediction", "joint_prediction": "free_joint_prediction",
             "hb_only_driver": "free_driver_h", "joint_driver": "free_driver_j"}
    with np.load(_path(evidence["run"], row["array_path"]), allow_pickle=False) as archive:
        target = np.asarray(archive["target"])[index]
        if "window_ids" in archive and str(archive["window_ids"][index]) != identity:
            raise ValueError("current_dof array_index identifies a different window")
        if "time_s" in archive and not np.array_equal(archive["time_s"], arrays["time_s"]):
            raise ValueError("fixed/current_dof example time grids differ")
        if "hb_factor" in archive and not np.array_equal(archive["hb_factor"], arrays.get("hb_factor", 1.)):
            raise ValueError("fixed/current_dof example native Hb factors differ")
        if not np.allclose(target, arrays["target"], equal_nan=True, rtol=1e-10, atol=1e-12):
            raise ValueError("fixed/current_dof example targets do not share identical observations")
        if not np.allclose(archive["sd"], arrays["sd"], rtol=1e-10, atol=1e-12):
            raise ValueError("fixed/current_dof example scoring scales differ")
        arrays = dict(arrays)
        for source, destination in names.items():
            if source in archive:
                value = np.asarray(archive[source])[index]
                expected = arrays["target"].shape if source.endswith("prediction") else arrays["time_s"].shape
                if value.shape != expected:
                    raise ValueError("current_dof trajectory shape does not match fixed example")
                arrays[destination] = value
        weak_prefix = f"pair_{example.get('weak_pair_index', example['pair'])}_"
        free_weak = {key: np.asarray(archive[key]) for key in archive.files
                     if key.startswith(weak_prefix) or key == "continuous_hb_time_s"}
        if free_weak:
            arrays["free_weak_arrays"] = free_weak
    return arrays, row


def _masked(values, mask):
    return np.where(np.asarray(mask, bool) & np.isfinite(values), values, np.nan)


def _residual_series(arrays, prediction, component):
    mask = (np.asarray(arrays["score_mask"], bool)[:, component] if component == 0 else
            np.asarray(arrays["score_mask"], bool)[:, 1:].all(axis=1))
    if prediction not in arrays:
        return np.full(len(arrays["time_s"]), np.nan), mask
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value = (arrays["target"][:, component]-arrays[prediction][:, component])/arrays["sd"][component]
    return _masked(value, mask), mask


def _uniform_time_step(time):
    time = np.asarray(time, float)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all():
        raise ValueError("time diagnostics require at least two finite retained grid times")
    step = float(round(float(np.median(np.diff(time))), 12))
    if step <= 0 or not np.allclose(np.diff(time), step, rtol=1e-8, atol=1e-10):
        raise ValueError("time diagnostics require the retained uniform scoring grid")
    return step


def _time_series_diagnostics(time, values, score_mask):
    """Describe one retained residual, preserving missing prediction support."""
    time, values, score_mask = np.asarray(time, float), np.asarray(values, float), np.asarray(score_mask, bool)
    if values.shape != time.shape or score_mask.shape != time.shape:
        raise ValueError("time diagnostic values and score mask must match time_s")
    step = _uniform_time_step(time)
    finite = score_mask & np.isfinite(values)
    count, finite_count = int(score_mask.sum()), int(finite.sum())
    result = dict(scored_bins=count, finite_bins=finite_count, step_s=step,
        effective_scored_seconds=count*step, complete_support=bool(count and count == finite_count),
        rms=None, mean_squared_residual=None, signed_peak=None, absolute_peak=None,
        peak_time_s=None, peak_ties=0, thirds=[], dominant_thirds=[])
    if not count:
        return dict(result, status="unavailable_score_support")
    if finite_count:
        indices = np.flatnonzero(finite)
        largest = int(np.argmax(np.abs(values[finite])))
        peak = int(indices[largest]); amplitude = float(abs(values[peak]))
        result.update(signed_peak=float(values[peak]), absolute_peak=amplitude,
            peak_time_s=float(time[peak]) if amplitude else None,
            peak_ties=int(np.sum(np.abs(values[finite]) == amplitude)))
    start, stop = float(time[score_mask][0]), float(time[score_mask][-1]+step)
    boundaries = np.linspace(start, stop, 4)
    thirds = np.searchsorted(boundaries[1:-1], time+step*1e-9, side="right")
    complete = result["complete_support"]
    amplitude = float(np.max(np.abs(values[finite]))) if finite_count else 0.
    normalized_square = np.zeros(len(values))
    if complete and amplitude:
        normalized_square[finite] = (values[finite]/amplitude)**2
    total = float(normalized_square.sum())
    for index in range(3):
        selected = score_mask & (thirds == index)
        n, available = int(selected.sum()), int((finite & selected).sum())
        energy = float(normalized_square[selected].sum())
        rms = amplitude*math.sqrt(energy/n) if complete and n else None
        result["thirds"].append(dict(start_s=float(boundaries[index]), end_s=float(boundaries[index+1]),
            scored_bins=n, finite_bins=available, effective_scored_seconds=n*step, rms=rms,
            standardized_square_share=energy/total if complete and total else None))
    if not complete:
        return dict(result, status="incomplete_prediction_support")
    rms = amplitude*math.sqrt(total/count) if amplitude else 0.
    mean_square = rms*rms if rms <= math.sqrt(np.finfo(float).max) else None
    result.update(status="completed", rms=float(rms), mean_squared_residual=mean_square)
    if total:
        shares = [item["standardized_square_share"] for item in result["thirds"]]
        result["dominant_thirds"] = [index for index, share in enumerate(shares) if share == max(shares)]
    return result


def example_time_diagnostics(arrays, validation, spatial):
    """Communication diagnostics for selected examples only, without refitting."""
    components = []
    labels = {"hb_only_prediction": "固定 Hb-only", "joint_prediction": "固定联合",
              "free_hb_only_prediction": "组级κτ Hb-only", "free_joint_prediction": "组级κτ 联合"}
    for key, label in labels.items():
        if key not in arrays and key not in validation:
            continue
        verdict = validation.get(key, dict(passed=None, label="数值状态未核实", raw_status="未记录"))
        for component, name in enumerate(("EEG", "HbO", "HbR")):
            values, mask = _residual_series(arrays, key, component)
            row = _time_series_diagnostics(arrays["time_s"], values, mask)
            row.update(prediction=key, view=label, component=name, validated=verdict["passed"],
                       view_status=verdict["label"], raw_view_status=verdict["raw_status"])
            components.append(row)
    by_key = {(item["prediction"], item["component"]): item for item in components}
    comparisons = []
    for prefix, mode in (("", "fixed"), ("free_", "current_dof")):
        for name in ("EEG", "HbO", "HbR"):
            h, j = by_key.get((prefix+"hb_only_prediction", name)), by_key.get((prefix+"joint_prediction", name))
            if h is None or j is None:
                continue
            comparable = all(item["validated"] is True and item["complete_support"]
                             and item["mean_squared_residual"] is not None for item in (h, j))
            item = dict(parameter_mode=mode, component=name, comparable=comparable,
                        joint_minus_hb_rms=None, joint_minus_hb_mean_square=None)
            if comparable:
                item.update(joint_minus_hb_rms=j["rms"]-h["rms"],
                            joint_minus_hb_mean_square=j["mean_squared_residual"]-h["mean_squared_residual"])
            comparisons.append(item)
    space = dict(status=spatial["status"], reason=spatial.get("reason", "未记录"))
    if spatial["status"] == "completed":
        time = np.asarray(arrays["time_s"], float); values = np.asarray(spatial["values"], float)
        finite = np.isfinite(values); count = int(finite.sum()); step = _uniform_time_step(time)
        if not count:
            raise ValueError("completed spatial curve requires finite scored bins")
        positive, negative, zero = finite & (values > 0), finite & (values < 0), finite & (values == 0)
        space.update(scored_bins=count, step_s=step, effective_scored_seconds=count*step,
            mean=float(spatial["mean"]), retained_Delta_spatial=float(spatial["Delta_spatial"]),
            selected_candidate=spatial.get("selected_candidate"), positive_bins=int(positive.sum()),
            negative_bins=int(negative.sum()), zero_bins=int(zero.sum()),
            positive_time_fraction=float(positive.sum()/count), negative_time_fraction=float(negative.sum()/count),
            zero_time_fraction=float(zero.sum()/count), maximum_gain=None, maximum_cost=None)
        for mask, key, largest in ((positive, "maximum_gain", True), (negative, "maximum_cost", False)):
            if mask.any():
                indices = np.flatnonzero(mask)
                pick = int(indices[np.argmax(values[mask]) if largest else np.argmin(values[mask])])
                space[key] = dict(value=float(values[pick]), time_s=float(time[pick]))
    return dict(scope="selected_example_retained_arrays_only", residual_sign="target_minus_prediction",
                time_coordinate="window_relative_seconds", component_units="frozen_training_SD",
                components=components, paired_view_comparisons=comparisons, spatial=space)


def example_time_blocks(diagnostics, source):
    """Visible, case-specific interpretation plus an auditable compact table."""
    def number(value, signed=False):
        return "不可用" if value is None else (f"{value:+.3g}" if signed else f"{value:.3g}")
    def peak(row):
        if row["signed_peak"] is None:
            return "无有限残差"
        if row["absolute_peak"] == 0:
            return "0；全段零误差，无唯一峰时刻" if row["complete_support"] else "有限子集残差为0；完整峰不可用"
        scope = "有限子集：" if not row["complete_support"] else ""
        return scope+number(row["signed_peak"], True)+f" @ {row['peak_time_s']:.1f}s"
    def dominant(row):
        indices = row["dominant_thirds"]
        if row["rms"] == 0:
            return "全评分段零误差，无主导区间"
        if not indices:
            return "无完整预测支持，未计算区间份额"
        if len(indices) == 3:
            thirds = row["thirds"]
            return (f"三段平方和相同，各 {100*thirds[0]['standardized_square_share']:.1f}%；"
                    +"有效点数 "+"/".join(str(item["scored_bins"]) for item in thirds)+"；段 RMS "
                    +"/".join(number(item["rms"]) for item in thirds))
        pieces = []
        for index in indices:
            item = row["thirds"][index]
            pieces.append(f"[{item['start_s']:.1f},{item['end_s']:.1f})s："
                          f"{100*item['standardized_square_share']:.1f}%；n={item['scored_bins']}，段 RMS={number(item['rms'])}")
        return ("并列：" if len(indices) > 1 else "")+"；".join(pieces)
    components = diagnostics["components"]
    lookup = {(item["prediction"], item["component"]): item for item in components}
    comparisons = {(item["parameter_mode"], item["component"]): item for item in diagnostics["paired_view_comparisons"]}
    paragraphs = []
    for prefix, mode, name in (("", "fixed", "固定参数"), ("free_", "current_dof", "组级 κ、τ")):
        h = {c: lookup.get((prefix+"hb_only_prediction", c)) for c in ("EEG", "HbO", "HbR")}
        j = {c: lookup.get((prefix+"joint_prediction", c)) for c in ("EEG", "HbO", "HbR")}
        if h["HbO"] is None and j["HbO"] is None:
            continue
        text = name+"："
        both_hb_valid = all(comparisons.get((mode, c), {}).get("comparable") for c in ("HbO", "HbR"))
        for rows, label in ((h, "Hb-only"), (j, "联合")):
            if rows["HbO"] is None or both_hb_valid:
                continue
            valid = all(rows[c] is not None and rows[c]["validated"] is True and rows[c]["complete_support"] for c in ("HbO", "HbR"))
            text += f"{label} 的 HbO/HbR 残差 RMS 为 {number(rows['HbO']['rms'])}/{number(rows['HbR']['rms'])} 个训练 SD"
            if not valid:
                text += "（该视图未通过或支持不完整，仅作失败／未认证诊断）"
            text += "；"
        changes = []
        for component in ("HbO", "HbR", "EEG"):
            comparison = comparisons.get((mode, component), {})
            if comparison.get("comparable"):
                difference = comparison["joint_minus_hb_rms"]
                direction = "数值降低" if difference < 0 else "数值增大" if difference > 0 else "相同"
                label = "EEG 读出" if component == "EEG" else component
                changes.append(f"{label}从 {number(h[component]['rms'])} 到 {number(j[component]['rms'])}（{direction}，联合减 Hb-only 为 {number(difference, True)}）")
        if changes:
            text += "同支持且两视图均通过时，联合相对 Hb-only："+"、".join(changes)+"。"
        else:
            text += "没有两视图均通过且支持完整的成对比较，不作联合改善结论。"
        candidates, candidate_scope = [], ""
        for rows, label, require_pass in ((h, "该 Hb-only 视图", True), (j, "该联合视图", True),
                                           (h, "该 Hb-only 失败诊断", False), (j, "该联合失败诊断", False)):
            selected = [item for item in (rows.get("HbO"), rows.get("HbR")) if item is not None
                        and item["rms"] is not None and (not require_pass or item["validated"] is True)]
            if selected:
                candidates = selected
                candidate_scope = label+"的两个 Hb 分量中" if len(selected) == 2 else label+"中当前可完整评分的 Hb 分量中"
                break
        if candidates:
            focus = max(candidates, key=lambda item: item["rms"])
            scope = "失败诊断中" if focus["validated"] is not True else "通过条件中"
            ties = sum(item["rms"] == focus["rms"] for item in candidates)
            largest = "并列最大（此处列出其中一个）" if ties > 1 else "最大"
            text += f"{scope}，在{candidate_scope}，{focus['component']} RMS {largest}；最大绝对偏差处的有符号残差为 {peak(focus)}。"
            if focus["signed_peak"]:
                text += "该点预测偏低。" if focus["signed_peak"] > 0 else "该点预测偏高。"
            text += "其残差平方和最多的评分时段："+dominant(focus)+"。"
        for rows, label in ((h, "Hb-only EEG 读出差 d_EH"), (j, "联合 EEG 残差")):
            row = rows.get("EEG")
            if row is not None and row["signed_peak"] is not None:
                rms_text = "" if comparisons.get((mode, "EEG"), {}).get("comparable") else f"RMS={number(row['rms'])}，"
                text += f"{label}的{rms_text}最大绝对偏差处残差 {peak(row)}"
                text += "（失败／支持不完整诊断）。" if row["validated"] is not True or not row["complete_support"] else "。"
        joint_valid = [item for item in j.values() if item is not None and item["validated"] is True
                       and item["complete_support"] and item["rms"] is not None]
        if len(joint_valid) == 3:
            most = max(item["rms"] for item in joint_valid)
            names = "/".join(item["component"] for item in joint_valid if item["rms"] == most)
            if most == 0:
                text += "联合的三个评分坐标均为零残差，没有主导误差坐标。"
            else:
                text += f"联合的三个标准化坐标中，{names} 的 RMS 最大（{number(most)}）；这是观测误差比较，不是生理成分占比。"
        paragraphs.append(_paragraph(text))
    table_rows = []
    for row in components:
        status = "通过" if row["validated"] is True and row["complete_support"] else "失败诊断" if row["validated"] is False else "未认证／缺支持"
        table_rows.append([row["view"], row["component"], status+
                          f"；{row['finite_bins']}/{row['scored_bins']} bins，{row['effective_scored_seconds']:.1f}s",
                          number(row["rms"]), peak(row), dominant(row)])
    table = _table(["参数／视图", "观测", "数值状态与有限／评分支持", "RMS / 训练 SD", "最大绝对值处的有符号残差 @ 相对时间", "残差平方和最多的分段"], table_rows,
        "仅由本例保留数组计算。残差为实测减预测并除以冻结训练 SD；正值表示预测偏低，负值表示预测偏高。EEG 用其 score_mask，两个 Hb 用共同 mask；RMS 只在完整评分预测存在时计算，缺失预测不缩减支持冒充成功。最大绝对值并列时取最早评分格点，全零误差不指定唯一峰。分段份额为标准化残差平方和份额，不是信号未解释方差；各段有效点数与段 RMS 一起判断，不把大份额直接解释为单位时间误差更高。完整三段数字保留在 content.json 的本例 time_diagnostics 中。", [1.2, .6, 1.5, .75, 1.45, 2.5])
    table["source"] = source
    space = diagnostics["spatial"]
    if space["status"] != "completed":
        spatial_text = "空间时间诊断不可用："+space["reason"]+"；不补零，也不据有限失败轨迹给出空间改善结论。"
    elif space.get("selected_candidate") == "nearest6":
        spatial_text = (f"空间选择保留 nearest6，同一预测作自比较；{space['scored_bins']} 个有效评分 bins × {space['step_s']:.3g}s="
            f"{space['effective_scored_seconds']:.1f}s 中零差值占 {100*space['zero_time_fraction']:.1f}%。"
            "这个零点不证明空间映射无误差；没有正／负收益时段，因此不指定收益或代价峰时刻。")
    else:
        spatial_text = (f"空间平均误差代价 Δ_spatial={number(space['mean'])}（已核对保留值）；有效评分时间为 "
            f"{space['scored_bins']} bins × {space['step_s']:.3g}s={space['effective_scored_seconds']:.1f}s。"
            f"正／负／严格数值零时段分别占 {100*space['positive_time_fraction']:.1f}%／{100*space['negative_time_fraction']:.1f}%／{100*space['zero_time_fraction']:.1f}%。")
        for key, label in (("maximum_gain", "最大单点收益"), ("maximum_cost", "最大单点代价")):
            peak_value = space[key]
            spatial_text += (f"{label}为 {number(peak_value['value'], True)} 个训练 SD 平方，出现在相对 {peak_value['time_s']:.1f}s。"
                             if peak_value is not None else f"没有{'正收益' if key == 'maximum_gain' else '负代价'}时段。")
        spatial_text += "这些比例是依赖时间序列上有效评分时长的占比，不是独立样本胜率；单点极值时刻不解释为生理时延。"
    return [*paragraphs, table, _paragraph(spatial_text)]


def _view_validation(record, view):
    """Use a view's retained verdict, never finiteness or its parent outcome."""
    nested = record.get("view_statuses", {}).get(view, {}) if isinstance(record.get("view_statuses"), dict) else {}
    raw = record.get(view+"_status")
    if not isinstance(raw, str) or not raw:
        raw = nested.get("status", "未保存 view 状态")
    value = record.get(view+"_success", nested.get("success"))
    if value is True or isinstance(value, (np.bool_, bool)) and value:
        passed = True
    elif value is False or isinstance(value, (np.bool_, bool)) and not value:
        passed = False
    elif isinstance(value, str) and value.lower() in ("true", "1", "false", "0"):
        passed = value.lower() in ("true", "1")
    elif isinstance(value, (int, float)) and np.isfinite(value) and value in (0, 1):
        passed = bool(value)
    else:
        passed = None
    if passed is None:
        if raw == "completed":
            passed = True
        elif isinstance(raw, str) and raw.startswith(("failed_", "unavailable_")):
            passed = False
    unavailable = isinstance(raw, str) and raw.startswith("unavailable_")
    label = "通过数值检查" if passed is True else "不可用／无支持" if unavailable else "未通过数值检查" if passed is False else "数值状态未核实"
    return dict(passed=passed, label=label, raw_status=raw)


def _example_validation(example, row, current_row):
    fixed = dict(row, **example)
    validation = {view+"_prediction": _view_validation(fixed, view) for view in ("hb_only", "eeg_only", "joint")}
    if current_row:
        validation.update({"free_"+view+"_prediction": _view_validation(current_row, view) for view in ("hb_only", "joint")})
    return validation


def _validation_label(label, key, validation):
    verdict = (validation or {}).get(key)
    if verdict is None:
        return label+"［状态未核实］"
    short = "通过" if verdict["passed"] is True else "不可用" if verdict["label"] == "不可用／无支持" else "未通过" if verdict["passed"] is False else "状态未核实"
    return label+"［"+short+"］"


def spatial_time_curve(arrays, record):
    """Instantaneous paired loss difference; verify its retained window mean."""
    empty = np.full(len(arrays["time_s"]), np.nan)
    result = dict(status="unavailable", values=empty, count=0, mean=None, Delta_spatial=None)
    try:
        delta = float(record.get("Delta_spatial"))
    except (ValueError, TypeError):
        delta = np.nan
    if not np.isfinite(delta):
        return dict(result, reason="无有效保留 Delta_spatial；不可用")
    result["Delta_spatial"] = delta
    if _view_validation(record, "eeg_only")["passed"] is not True:
        return dict(result, reason="当前近邻 EEG-only 未通过数值检查或状态未核实；不可用")
    selected = record.get("selected_candidate")
    validation = record.get("spatial_status", {})
    if (isinstance(validation, dict) and "success" in validation and validation["success"] is not True
            or selected != "nearest6" and (not isinstance(validation, dict) or validation.get("success") is not True)):
        return dict(result, reason="所选空间预测未通过数值检查或状态未核实；不可用")
    if "spatial_prediction" not in arrays:
        return dict(result, reason="所选空间预测数组未保存；不可用")
    target = np.asarray(arrays["target"], float)
    nearest = np.asarray(arrays["eeg_only_prediction"], float)
    chosen = np.asarray(arrays["spatial_prediction"], float)
    if nearest.shape != target.shape or chosen.shape != target.shape:
        raise ValueError("spatial predictions must match target [T,3]")
    support = np.asarray(arrays["score_mask"], bool)[:, 1:].all(axis=1)
    if not support.any():
        return dict(result, reason="两个 Hb 色团无共同评分支持；不可用")
    if not all(np.isfinite(value[support, 1:]).all() for value in (target, nearest, chosen)):
        return dict(result, reason="共同评分支持内存在缺失预测；不可缩减支持冒充主指标")
    sd = np.asarray(arrays["sd"], float)[1:]
    first = ((target[support, 1:]-nearest[support, 1:])/sd)**2
    second = ((target[support, 1:]-chosen[support, 1:])/sd)**2
    value = np.mean(first-second, axis=1)
    mean = float(np.mean(value))
    tolerance = 64*np.finfo(float).eps*max(1., float(np.mean(first)), float(np.mean(second)))
    if not np.isclose(mean, delta, rtol=1e-8, atol=tolerance):
        raise ValueError(f"spatial time-curve mean disagrees with retained Delta_spatial: {mean} != {delta}")
    if selected == "nearest6" and (not np.allclose(nearest[support, 1:], chosen[support, 1:], rtol=0, atol=0) or not np.isclose(delta, 0, rtol=0, atol=tolerance)):
        raise ValueError("nearest6 selection must reuse the same retained prediction and have zero spatial difference")
    values = empty.copy()
    values[support] = value
    return dict(result, status="completed", reason="时间均值已核对保留 Delta_spatial", values=values,
                count=int(support.sum()), mean=mean, selected_candidate=selected)


def _mark_unscored(axis, time, visible, scored):
    excluded = np.asarray(visible, bool) & ~np.asarray(scored, bool)
    edges = np.diff(np.r_[False, excluded, False].astype(int))
    for index, (start, stop) in enumerate(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))):
        step = float(np.median(np.diff(time))) if len(time) > 1 else 0.
        axis.axvspan(time[start], min(time[-1], time[stop-1]+step), color="#a6adb5", alpha=.16,
                     label="可见但未评分支持" if index == 0 else None)


def example_observation_figure(arrays, destination, *, title="", hb_unit="共同 Hb 观测坐标", limits=None, validation=None):
    plt = _plotting()
    fig, axes = plt.subplots(4, 1, figsize=(8, 7.2), sharex=True)
    t, y, mask, sd = (arrays[key] for key in ("time_s", "target", "score_mask", "sd"))
    hb_factor = float(arrays.get("hb_factor", 1.))
    if not math.isfinite(hb_factor) or hb_factor <= 0:
        raise ValueError("hb_factor must be a finite positive common factor")
    views = (("hb_only_prediction", "hb", "Hb-only"), ("joint_prediction", "joint", "联合"),
             ("eeg_only_prediction", "eeg", "EEG-only 独立 Hb 预测"))
    for axis, component in zip(axes[:2], (1, 2)):
        shared = np.isfinite(y[:, 1:]).all(axis=1)
        _mark_unscored(axis, t, shared, mask[:, 1:].all(axis=1))
        axis.plot(t, _masked(y[:, component]/hb_factor, shared), color=COLORS["target"], lw=1.5, ls=LINESTYLES["target"], label="观测：处理后 Hb / EEG 代理")
        for key, color, label in views:
            axis.plot(t, _masked(arrays[key][:, component]/hb_factor, shared), color=COLORS[color], lw=1.2,
                      ls=LINESTYLES[color], label=_validation_label(label, key, validation))
        for key, color, label in (("free_hb_only_prediction", "free_hb", "Hb-only：组级 κ、τ"),
                                  ("free_joint_prediction", "free_joint", "联合：组级 κ、τ")):
            if key in arrays:
                axis.plot(t, _masked(arrays[key][:, component]/hb_factor, shared), color=COLORS[color],
                          lw=1.2, ls=LINESTYLES[color], label=_validation_label(label, key, validation))
        axis.set_ylabel(("HbO" if component == 1 else "HbR")+"\n"+hb_unit)
        if limits is not None:
            axis.set_ylim(limits[component-1])
        axis.grid(alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, .955))
    _mark_unscored(axes[2], t, np.isfinite(y[:, 0]), mask[:, 0])
    axes[2].plot(t, _masked(y[:, 0]/sd[0], np.isfinite(y[:, 0])), color=COLORS["target"], lw=1.4, ls=LINESTYLES["target"], label="EEG 代理")
    for key, color, label in views:
        readout_label = {"hb": "Hb-only 驱动读出", "joint": "联合驱动读出", "eeg": "EEG-only 驱动读出"}[color]
        axes[2].plot(t, _masked(arrays[key][:, 0]/sd[0], np.isfinite(y[:, 0])), color=COLORS[color], lw=1.1,
                     ls=LINESTYLES[color], label=readout_label)
    for key, color, label in (("free_hb_only_prediction", "free_hb", "Hb-only：组级 κ、τ 读出"),
                              ("free_joint_prediction", "free_joint", "联合：组级 κ、τ 读出")):
        if key in arrays:
            axes[2].plot(t, _masked(arrays[key][:, 0]/sd[0], np.isfinite(y[:, 0])),
                         color=COLORS[color], lw=1.1, ls=LINESTYLES[color], label=label)
    axes[2].set_ylabel("EEG 观测\n训练 SD")
    for key, color, label in (("driver_h", "hb", "Hb 条件驱动"), ("driver_e", "eeg", "EEG-only 驱动"),
                              ("driver_j", "joint", "联合驱动")):
        if key in arrays:
            value = np.asarray(arrays[key]).reshape(-1)
            if len(value) != len(t):
                raise ValueError(f"{key} must align with time_s")
            axes[3].plot(t, value, color=COLORS[color], lw=1.2, ls=LINESTYLES[color], label=label)
    for key, color, label in (("free_driver_h", "free_hb", "Hb 驱动：组级 κ、τ"),
                              ("free_driver_j", "free_joint", "联合驱动：组级 κ、τ")):
        if key in arrays:
            axes[3].plot(t, arrays[key], color=COLORS[color], lw=1.1, ls=LINESTYLES[color], label=label)
    if not axes[3].lines:
        axes[3].text(.5, .5, "驱动数组未保存；不可用", ha="center", transform=axes[3].transAxes)
    axes[3].set_ylabel("驱动模型坐标")
    axes[3].set_xlabel("窗口相对时间 / s")
    for axis in axes[2:]:
        axis.grid(alpha=.15)
    fig.suptitle(title or "观测重建与条件驱动", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .89))
    return _save(fig, destination)


def example_residual_figure(arrays, destination, *, validation=None):
    plt = _plotting()
    fig, axes = plt.subplots(3, 1, figsize=(8, 4.8), sharex=True)
    views = (("hb_only_prediction", "hb", "Hb-only 固定"),
             ("joint_prediction", "joint", "联合固定"),
             ("free_hb_only_prediction", "free_hb", "Hb-only 组级κτ"),
             ("free_joint_prediction", "free_joint", "联合组级κτ"))
    for c, (axis, name) in enumerate(zip(axes, ("EEG", "HbO", "HbR"))):
        for key, color, label in views:
            if key not in arrays:
                continue
            value, _ = _residual_series(arrays, key, c)
            axis.plot(arrays["time_s"], value, color=COLORS[color], lw=1.2, ls=LINESTYLES[color],
                      label=_validation_label(label, key, validation))
        axis.axhline(0, color="#555", lw=.7)
        axis.set_ylabel(name+"\n训练 SD")
        axis.grid(alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, .945))
    axes[-1].set_xlabel("窗口相对时间 / s")
    fig.suptitle("Hb-only 与联合有符号残差：实测 − 对应预测", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .86))
    return _save(fig, destination)


def example_observable_figure(arrays, destination):
    """Display the retained local Hb effect, not a measured physiological truth."""
    key = "observable_standardized_hb_difference"
    if key not in arrays:
        return None
    values = np.asarray(arrays[key], float)
    if values.shape != (len(arrays["time_s"]), 2):
        raise ValueError("observable standardized Hb difference must be [T,2]")
    plt = _plotting()
    fig, axes = plt.subplots(2, 1, figsize=(8, 3.5), sharex=True)
    common = arrays["score_mask"][:, 1:].all(axis=1)
    for component, axis in enumerate(axes):
        axis.plot(arrays["time_s"], _masked(values[:, component], common), color=COLORS["spatial"], lw=1.25)
        axis.axhline(0, color="#555", lw=.7)
        axis.set_ylabel(("HbO" if component == 0 else "HbR")+" 局部差异\n训练 SD")
        axis.grid(alpha=.15)
        if not np.isfinite(values[common, component]).any():
            axis.text(.5, .5, "无保留模式或数值不可用", ha="center", transform=axis.transAxes)
    axes[-1].set_xlabel("窗口相对时间 / s")
    fig.suptitle("预设合成噪声情景下的 Hb 可观测驱动差异（局部线性）", fontsize=11)
    fig.tight_layout()
    return _save(fig, destination)


def _weak_series(weak, pair, width):
    """Support the evaluator's explicit pair/scale NPZ prefix convention."""
    prefixes = (f"pair_{pair}_{width}s_", f"pair{pair}_{width}s_", f"p{pair}_w{width}_", f"pair_{pair}_scale_{width}_")
    for prefix in prefixes:
        time_key = next((prefix+key for key in ("center_s", "center_time_s") if prefix+key in weak), None)
        if time_key and prefix+"w" in weak:
            time = np.asarray(weak[time_key], float)
            if prefix+"center_index" in weak and "continuous_hb_time_s" in weak:
                index = np.asarray(weak[prefix+"center_index"], int)
                clock = np.asarray(weak["continuous_hb_time_s"], float)
                if np.any(index < 0) or np.any(index >= len(clock)):
                    raise ValueError("weak center indices exceed the retained native Hb clock")
                time = clock[index]
            value = np.asarray(weak[prefix+"w"], float)
            valid = np.asarray(weak.get(prefix+"valid", np.isfinite(value)), bool)
            return time, _masked(value, valid)
    return None


def example_weak_spatial_figure(evidence, example, arrays, destination, *, spatial=None):
    plt = _plotting()
    fig, axes = plt.subplots(4, 1, figsize=(8, 6.6))
    weak = {}
    if example.get("weak_path"):
        with np.load(_path(evidence["run"], example["weak_path"]), allow_pickle=False) as archive:
            weak = {key: np.asarray(archive[key]) for key in archive.files}
    start = float(example.get("hb_start_s", example.get("start_s",
        arrays.get("hb_absolute_start_s", arrays.get("absolute_start_s", 0.)))))
    end = start + float(arrays["time_s"][-1]-arrays["time_s"][0])
    for axis, width in zip(axes[:3], (10, 20, 40)):
        series = _weak_series(weak, example.get("weak_pair_index", example["pair"]), width)
        if series is not None:
            time, value = series
            context = (time >= start-20) & (time <= end+20)
            time, value = time[context], value[context]
            if len(time) and np.isfinite(value).any():
                axis.plot(time-start, value, color=COLORS["hb"], lw=1.2, ls=LINESTYLES["hb"], label="固定参数")
                axis.axhline(0, color="#555", lw=.7)
                axis.axvspan(0, end-start, color="#dbe7f0", alpha=.6, label="所选拟合窗口")
                axis.set_xlabel("相对所选窗口起点 / s；阴影为拟合窗口")
                axis.legend(loc="upper right", frameon=False)
            else:
                axis.text(.5, .5, "此处无完整有效核支持；不可用", ha="center", transform=axis.transAxes)
        else:
            axis.text(.5, .5, f"{width} 秒完整支持曲线未保存或不可用；不填零", ha="center", transform=axis.transAxes)
        free_series = _weak_series(arrays.get("free_weak_arrays", {}),
                                   example.get("weak_pair_index", example["pair"]), width)
        if free_series is not None:
            free_time, free_value = free_series
            context = (free_time >= start-20) & (free_time <= end+20)
            if np.isfinite(free_value[context]).any():
                axis.plot(free_time[context]-start, free_value[context], color=COLORS["free_hb"],
                          lw=1.2, ls=LINESTYLES["free_hb"], label="组级 τ 条件")
                axis.legend(loc="upper right", frameon=False)
        axis.set_ylabel(f"{width}s 弱残差\n原缓存 Hb单位 × s")
        axis.grid(alpha=.15)
    spatial = spatial if spatial is not None else spatial_time_curve(arrays, example)
    if spatial["status"] == "completed":
        axes[3].plot(start+arrays["time_s"], spatial["values"], color=COLORS["spatial"], lw=1.25,
                     label="当前近邻平方误差 − 冻结所选方案平方误差")
        axes[3].axhline(0, color="#555", lw=.7)
        axes[3].axhline(spatial["Delta_spatial"], color=COLORS["wrong"], ls="--", lw=.9,
                       label="保留窗口平均 Δ_spatial="+_num(spatial["Delta_spatial"]))
        axes[3].legend(frameon=False, fontsize=7)
        if spatial.get("selected_candidate") == "nearest6":
            axes[3].text(.02, .85, "选择保留 nearest6：同一预测自比较", transform=axes[3].transAxes, fontsize=8)
    else:
        axes[3].text(.5, .5, spatial["reason"], ha="center", va="center", wrap=True, transform=axes[3].transAxes)
    axes[3].set(xlabel="原生记录评分网格时间 / s", ylabel="空间瞬时误差代价\n训练 SD 平方")
    axes[3].grid(alpha=.15)
    fig.suptitle("全记录弱残差上下文与空间误差代价随时间变化", fontsize=11)
    fig.tight_layout()
    return _save(fig, destination)


def _paragraph(text):
    return {"type": "paragraph", "text": text}


def _table(columns, rows, caption="", widths=None):
    result = {"type": "table", "columns": columns, "rows": rows, "caption": caption}
    if widths:
        result["widths"] = widths
    return result


def _figure(path, caption, source):
    return {"type": "figure", "path": path, "caption": caption, "source": source}


def _metric_interpretation(evidence, metric):
    pieces = []
    for dataset in _datasets(evidence):
        values = _subject_values(evidence, dataset, metric)
        if not len(values):
            pieces.append(f"{DATASET_LABELS.get(dataset, dataset)} 无有效被试统计")
            continue
        median, q90 = np.quantile(values, (.5, .9))
        text = f"{DATASET_LABELS.get(dataset, dataset)}：{len(values)} 名有效被试，中位数 {_num(median)}，P90 {_num(q90)}"
        current = _subject_values(evidence, dataset, metric, "current_dof")
        if len(current):
            if metric == "D_obs":
                text += f"；组级解处逐窗放松量中位数 {_num(np.median(current))}（nuisance 空间不同，不作为组共享模型改善）"
            else:
                text += f"；组级自由度中位数 {_num(np.median(current))}（两组中位数之差不等于配对效应）"
        paired = evidence["summary"].get("paired_current_dof", {}).get(dataset, {}).get(metric, {})
        if paired:
            delta = paired.get("fixed_minus_free_subject_mean")
            interval = paired.get("subject_bootstrap_95pct")
            contrast = "固定量减逐窗放松量的被试等权数值差" if metric == "D_obs" else "固定减组级自由度的被试等权差"
            text += f"；共同支持 {paired.get('paired_windows', '未记录')} 个窗口对、{paired.get('paired_subjects', '未记录')} 名被试，{contrast} {_num(delta)}"
            if interval is not None:
                text += f"，95% 被试区间 [{_num(interval[0])}, {_num(interval[1])}]"
                if interval[0] <= 0 <= interval[1]:
                    text += "，区间跨零，数值差方向不确定，不作改善判定" if metric == "D_obs" else "，区间跨零，改善方向仍不确定"
        pieces.append(text)
    return "；".join(pieces)+"。"+METRICS[metric][2]


def _coverage(evidence):
    rows, summary = evidence["rows"], evidence["summary"]
    result = []
    for dataset in _datasets(evidence):
        part = rows[rows.dataset.astype(str) == dataset]
        fixed = part[part.parameter_mode == "fixed"]
        if "success" in fixed:
            success = fixed.success.astype(str).str.lower().isin(("true", "1", "1.0"))
        elif "status" in fixed:
            success = fixed.status.isin(("completed", "success", "succeeded"))
        else:
            raise ValueError("metrics must retain status or success for coverage")
        result.append([DATASET_LABELS.get(dataset, dataset), int(fixed.subject.nunique()), len(fixed),
                       int(success.sum()), f"{100*(1-success.mean()):.2f}%" if len(fixed) else "不可用",
                       int(np.isfinite(fixed.D1).sum()), int(np.isfinite(fixed.Delta_spatial).sum()) if "Delta_spatial" in fixed else 0])
    return _table(["数据集", "计划被试", "计划窗口对", "Hb-only 成功", "未成功合计", "D1 有限值", "空间有效值"], result,
                  "计划窗口对包含所有终态。未成功合计包含数值未通过、无支持、执行异常及缺失，不称为数值失败率；各条件的分类与明确分母见下表。有限值不替代成功状态。",
                  [1.25, .9, 1.1, 1.1, 1.1, 1.1, 1.1])


def _metric_table(evidence, metrics):
    table = []
    for dataset in _datasets(evidence):
        for mode in ("fixed", "current_dof"):
            for metric in metrics:
                values = _subject_values(evidence, dataset, metric, mode)
                if mode == "current_dof" and not len(values):
                    continue
                quantiles = np.quantile(values, (.25, .5, .75, .9, .95)) if len(values) else [np.nan]*5
                table.append([DATASET_LABELS.get(dataset, dataset), "固定" if mode == "fixed" else "逐窗放松" if metric == "D_obs" else "组级κτ",
                              metric, len(values), *[_num(value) for value in quantiles]])
    return _table(["数据集", "参数", "指标", "有效被试", "Q25", "中位数", "Q75", "P90", "P95"], table,
                  "分位数针对已保存的被试内汇总值；被试等权。表中有效被试不是窗口数。",
                  [1.2, .85, 1.2, .8, .8, .8, .8, .8, .8])


def _example_row(evidence, example):
    rows = evidence["rows"]
    if "task_id" in rows and example.get("task_id") is not None:
        part = rows[rows.task_id.astype(str) == str(example["task_id"])]
    else:
        part = rows
        for key in ("dataset", "subject", "record", "pair", "window"):
            if key in rows:
                part = part[part[key].astype(str) == str(example[key])]
    if "parameter_mode" in part:
        part = part[part.parameter_mode == example.get("parameter_mode", "fixed")]
    if len(part) != 1:
        raise ValueError(f"example must identify one metric row, found {len(part)}: {example}")
    return part.iloc[0].to_dict()


def hb_unit_label(evidence, dataset):
    units = {item["provenance"].get("output_unit") for item in evidence.get("hb_provenance", [])
             if item["dataset"] == dataset}
    units.discard(None)
    if len(units) != 1:
        return "原缓存 Hb 单位"
    unit = str(next(iter(units))).replace("\u00b5", "\u03bc")
    if unit in ("uM", "μM", "μmol/L"):
        return "μM"
    if unit == "relative_Hb_repo_approximation":
        return "近似 MBLL 相对 Hb"
    if unit.startswith("relative:"):
        return "发布相对单位"
    return unit


def hb_unit_blocks(evidence):
    rows = []
    explanations = {
        "eeg_fnirs_single_trial": "光强经近似 MBLL 形成相对 Hb；尚非标定 μM。原生光学 V 不是血红蛋白浓度单位。",
        "refed": "发布 Hb 的绝对单位未知，保留相对单位；不能根据幅值补设浓度。",
        "visual_cognitive_motivation": "发布 Oxy/Deoxy 的绝对单位未知，保留相对单位；完整 Probe 身份不合并。",
        "simultaneous_eeg_nirs": "缓存已把已知 mmol/L 转为 μM。图示为窗口参考后的浓度变化，未恢复绝对静息浓度。",
    }
    for item in evidence.get("hb_provenance", []):
        p, dataset = item["provenance"], item["dataset"]
        rows.append([DATASET_LABELS.get(dataset, dataset), str(p.get("native_unit", "未记录")).replace("\u00b5", "\u03bc"),
                     str(p.get("output_unit", "未记录")).replace("\u00b5", "\u03bc"), explanations.get(dataset, "按本次保留的 Hb provenance 解释，未知单位不猜测。")])
    blocks = [_paragraph("Hb 波形已除以训练冻结的共同 Hb factor，回到原缓存单位；它仍是带通缓存、前 5 秒窗口参考后的信号。本报告不再做 MBLL、带通或单位换算。弱残差 w、U、V 的量纲为原缓存 Hb 单位 × 秒（核 phi 无量纲），D_bal 则无量纲，不能共用纵轴解释。")]
    if rows:
        blocks.append(_table(["数据集", "原生／发布输入单位", "缓存输出单位", "报告中的含义"], rows,
            "单位与处理来源直接核对本 run 独立验证样例所指向的 prepared metadata / hb_provenance；表列的是保留坐标的含义，不代表跨库绝对浓度可比较。", [1.1, 1.2, 1.7, 3.1]))
        blocks[-1]["source"] = "; ".join(item["metadata_path"] for item in evidence.get("hb_provenance", []))+"；docs/DATA_CONTRACT.md"
    else:
        blocks.append(_paragraph("本次导出没有指向 prepared Hb provenance 的保留样例，图中仅标原缓存单位，不推断 μM 或其他绝对单位。"))
    blocks.append(_paragraph("P0、Q0 是 Tak 模型观测坐标的基准尺度；参考模型的数值 1 和 0.35 不是上述原缓存单位下已知的绝对 Hb 浓度。各库的未知绝对单位、带通与参考限制仍须保留。"))
    return blocks


def spatial_selection_table(evidence):
    """Count retained frozen selections once per outer fold and Hb site."""
    frame = evidence["rows"]
    keys = ["dataset", "pair", "fold"]
    if not all(key in frame for key in (*keys, "selected_candidate")):
        return None
    rows = frame[frame.parameter_mode == "fixed"][keys+["selected_candidate"]].dropna().drop_duplicates()
    if rows.empty:
        return None
    if rows.duplicated(keys).any():
        raise ValueError("one outer fold/Hb site has conflicting retained spatial candidates")
    counts = rows.groupby(["dataset", "selected_candidate"]).size()
    totals = rows.groupby("dataset").size()
    table = _table(["数据集", "训练冻结的所选方案", "外折 × Hb 站点数", "有保留选择的总单元数", "占比"],
        [[DATASET_LABELS.get(dataset, dataset), candidate, int(count), int(totals[dataset]),
          f"{100*count/totals[dataset]:.2f}%"] for (dataset, candidate), count in counts.items()],
        "按 dataset、完整 Hb pair、outer fold 去重；每个单元只计一次。单位为外折 × Hb 站点，多个窗口不会重复增加选择次数；这些单元也不是独立被试。nearest6 表示训练选择保留原方案，对同一预测作自比较，其 Δ_spatial=0 不证明空间映射无误差。", [1.1, 1.7, 1.1, 1.4, .8])
    table["source"] = str(evidence["metric_path"])
    return table


def terminal_status_blocks(evidence):
    """Render mutually exclusive owner classifications and explicit rate bases."""
    audit = evidence.get("terminal_audit", {})
    if not audit:
        return []
    count_rows, rate_rows = [], []
    for dataset, modes in audit.get("by_dataset", {}).items():
        for mode in ("fixed", "current_dof"):
            for view, item in modes.get(mode, {}).get("views", {}).items():
                names = ("planned", "completed", "numerical_failure", "unavailable_support", "execution_failure", "missing", "fitted")
                if any(not isinstance(item.get(name), int) or isinstance(item.get(name), bool) or item[name] < 0 for name in names):
                    raise ValueError("terminal audit requires nonnegative integer category and fitted counts")
                if sum(item[name] for name in names[1:6]) != item["planned"]:
                    raise ValueError("terminal audit categories must partition the planned denominator")
                if item["fitted"] > item["planned"]:
                    raise ValueError("terminal audit fitted count cannot exceed planned windows")
                rates = item.get("rates", {})
                bases = {"support_rate": (item["fitted"], item["planned"]),
                         "numerical_failure_rate": (item["numerical_failure"], item["completed"]+item["numerical_failure"]),
                         "completion_rate": (item["completed"], item["planned"])}
                cells = []
                for rate, expected in bases.items():
                    declared = rates.get(rate, {})
                    if (declared.get("numerator"), declared.get("denominator")) != expected:
                        raise ValueError(f"terminal audit {rate} denominator disagrees with its declared category counts")
                    numerator, denominator = expected
                    cells.append(f"{numerator}/{denominator} ({100*numerator/denominator:.2f}%)" if denominator else "0/0（不可用）")
                identity = [DATASET_LABELS.get(dataset, dataset), "固定" if mode == "fixed" else "组级κτ",
                            {"hb_only": "Hb-only", "eeg_only": "EEG-only", "joint": "联合"}.get(view, view)]
                count_rows.append(identity+[item[name] for name in names[:6]])
                rate_rows.append(identity+cells)
    if not count_rows:
        raise ValueError("terminal audit must retain at least one planned dataset/view classification")
    source = str(evidence["terminal_audit_path"])
    blocks = [
        _paragraph("拟合终态的只读审计状态："+str(audit.get("status", "未记录"))+"。成功、数值未通过、无支持、执行异常和缺失为互斥分类，五类数量合计为全计划分母。无支持没有进入数值拟合，不计作数值失败；执行异常与缺失也单独列出。"),
        _paragraph("固定条件按每个原始 view 的保留状态分类。组级 κ、τ 条件使用各组保留的 fitted 数确定进入该条件拟合的窗口数；组计划数减成功数不能作为数值失败数，未进入拟合的无支持窗口仍保留在全计划分母。分组参数共享不会把窗口数改称独立参数组或独立被试。"),
        _table(["数据集", "参数", "条件", "全计划", "成功", "数值未通过", "无支持", "执行异常", "缺失"], count_rows,
               "每行单位为该条件下的窗口内 Hb 对；五类数量互斥。原始 solver 状态与其他 Dobs／空间／错配状态保留在 owning 审计文件中。", [1.1, .7, .9, .85, .85, .9, .85, .85, .7]),
        _table(["数据集", "参数", "条件", "支持率：进入拟合／全计划", "数值未通过率：未通过／有数值判定", "完成率：成功／全计划"], rate_rows,
               "支持率直接沿用审计 fitted 数；数值未通过率的分母只含成功与数值未通过，不含无支持、执行异常或缺失。0/0 标为不可用，不能写成 0% 失败。每格同时给出分子、分母与百分比。", [1.05, .7, .85, 1.65, 1.95, 1.65]),
    ]
    for block in blocks:
        if block["type"] == "table":
            block["source"] = source
    return blocks


def weak_verification_blocks(evidence):
    """Summarize the one retained audit; never reopen its measured inputs."""
    from collections import Counter

    verification = evidence.get("weak_verification", {})
    if not verification:
        return []
    source = str(evidence["weak_verification_path"])
    summary = verification.get("summary", {})
    def scientific(value):
        try:
            number = float(value)
        except (ValueError, TypeError):
            return "未记录"
        return f"{number:.3g}" if math.isfinite(number) else "不可用"
    blocks = [
        _paragraph("独立数值复核与全量状态核验是两种证据。前者只对预先选定样例逐数组复算；后者读取全部记录的保留 metadata，核对计划分母、完成状态和缺失原因。不能把样例复算范围说成全部数据已逐数组复核。"),
        _paragraph(f"独立数值复核状态：{verification.get('status', '未记录')}。覆盖 {summary.get('record_pairs', '未记录')} 个记录内 Hb 对、{summary.get('width_checks', '未记录')} 个核宽度条件；{summary.get('saved_array_comparisons', '未记录')} 项保留数组比较，{summary.get('saved_D_bal_exact_matches', '未记录')} 项 D_bal 与保留值完全相同。另在 {summary.get('independent_quadrature_center_checks', '未记录')} 个中心做独立 Gauss–Legendre 积分，最大相对误差 {scientific(summary.get('maximum_independent_quadrature_relative_error'))}。"),
        _paragraph("Gauss–Legendre 复核在每个 100 ms 的分段线性 Hb 区间上用 8 个积分点，检查各条件首个、中间、末个有效核中心；相对误差的分母是同一宽度所有有效中心的最大绝对 U 或 V。该核验检查数值积分，不能证明真实数据符合精确生理守恒。"),
        _paragraph("常数参考平移后的 D_bal 最大差异为 "+scientific(summary.get("maximum_reference_D_bal_difference"))+"；共同正尺度变换后的最大差异为 "+scientific(summary.get("maximum_scale_D_bal_difference"))+f"。上述样例中跨越原生连续段边界的有效核数为 {summary.get('valid_cross_segment_kernels', '未记录')}。边界检查按冻结清单的原生 Hb 连续支持执行，不把不连续会话拼接为一个有效核。"),
    ]
    matches = verification.get("source_function_exact_match", {})
    if matches:
        blocks.append(_paragraph(f"生产快照与复核代码的弱核相关函数逐字核对：{sum(value is True for value in matches.values())}/{len(matches)} 一致；函数身份与生产快照位置见本次验证记录。"))
    sample_rows = []
    for record in verification.get("records", []):
        for row in record.get("rows", []):
            sample_rows.append([DATASET_LABELS.get(record.get("dataset"), record.get("dataset")),
                str(record.get("record", "未记录")), str(record.get("pair", "未记录")),
                _num(row.get("width_s"), 0), f"{row.get('valid_centers', '未记录')}/{row.get('planned_centers', '未记录')}",
                _num(row.get("D_bal"), 6)])
    if sample_rows:
        table = _table(["数据集", "样例记录", "完整 Hb 对身份", "核宽 / s", "有效／计划中心", "D_bal"], sample_rows,
            "这些行是独立数值复核的样例，不是整个数据集的分布，也不是独立被试重复。Probe 与原生记录身份完整保留。", [1.1, 2.3, 1, .7, 1.1, .9])
        table["source"] = source
        blocks.append(table)
    audit = verification.get("record_pair_status_audit_20s", {})
    if audit:
        blocks.append(_paragraph("全量 20 秒核的状态核验读取全部计划 record-pair 的 metadata，以下计数与上面的样例数组复算分开。一个单元是原生记录内的 Hb 对，不是一个被试，也不是一个核中心。"))
        status_rows, reason_rows = [], []
        for dataset, item in audit.get("by_dataset", {}).items():
            status_rows.append([DATASET_LABELS.get(dataset, dataset), item.get("planned_record_pairs", "未记录"),
                item.get("completed_record_pairs", "未记录"),
                "; ".join(f"{key}: {value}" for key, value in item.get("status_counts", {}).items()) or "未记录"])
            reasons = {}
            for row in item.get("noncomplete", []):
                reasons.setdefault(row.get("status", "未记录"), []).append(row)
            for status, rows in reasons.items():
                pairs = sorted({str(row.get("pair", "未记录")) for row in rows})
                records = {str(row.get("record", "未记录")) for row in rows}
                zero_denominator = sum(row.get("mean_denominator") == 0 for row in rows)
                zero_support = sum(row.get("valid_centers") == 0 for row in rows)
                explanations = Counter(str(reason) for row in rows
                    if (reason := row.get("plan_record_reason") or evidence.get("plan_record_reasons", {}).get(str(row.get("record")))))
                first = rows[0]
                description = f"记录数 {len(records)}；Hb 对：{', '.join(pairs)}；零有效支持 {zero_support}，mean_denominator 精确为零 {zero_denominator}。"
                if explanations:
                    description += "冻结清单原因："+"; ".join(f"{key} ({count})" for key, count in explanations.items())+"。"
                description += f"首项完整身份：{first.get('record', '未记录')} / {first.get('pair', '未记录')}；其余完整身份保留在本次验证 JSON。"
                reason_rows.append([DATASET_LABELS.get(dataset, dataset), status, len(rows), description])
        for table in (_table(["数据集", "计划 record-pair", "完成 record-pair", "各终态数量"], status_rows,
                            "状态来自全量 20 秒弱残差 metadata 审计；未完成项保留在分母，不填成零失配。", [1.2, 1.1, 1.1, 2.9]),
                      _table(["数据集", "非完成状态", "数量", "支持、能量与冻结清单原因"], reason_rows,
                            "零动态能量、无有效支持和既定排除含义不同；不能统称为数值求解失败。绝对能量仅在同库同单位同核宽度内解释。", [1.1, 1.6, .5, 3.2])):
            table["source"] = source+("；"+str(evidence["plan_path"]) if evidence.get("plan_path") else "")
            blocks.append(table)
    blocks.append(_paragraph("这些核验支持已保存弱残差的数值与支持处理，并不校准真实噪声或证明生理模型有效。全量分布仍由独立 record-pair 与被试汇总给出，跨库排序保持描述性。"))
    return blocks


def build_content(evidence, directory):
    """Build an authored communication export from the evaluator's evidence."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    run, summary = evidence["run"], evidence["summary"]
    aggregation = summary.get("aggregation", "以 subject_summary.csv 中冻结的被试内汇总为准")
    if isinstance(aggregation, (dict, list)):
        aggregation = json.dumps(aggregation, ensure_ascii=False)
    content = {
        "title": "Tak 单驱动模型：四数据集四轴失配分析",
        "subtitle": "Hb 动态相容性、EEG/Hb 驱动差异、空间预测代价与联合残差",
        "date": str(summary.get("completed_at", summary.get("date", ""))),
        "provenance": [f"只读证据目录：{run}", "本报告是保留证据的通信导出，不改变拟合、数据访问或实验状态。"],
        "summary": [
            f"全计划 {summary.get('planned', len(evidence['rows']))} 个窗口对，终态 {summary.get('terminal', len(evidence['rows']))}。四类指标回答不同条件下的问题，不能相加为总失配。",
            "主结果对被试等权；有效时间占比作为辅助分布。窗口、通道对和重叠弱核不是独立被试。未成功、缺失支持和动态激发不足均保留，不填为零。",
            "本轮检查当前缓存坐标与当前 Tak 实现的相容性。带通、参考后的 Hb 不是已知绝对基线的生理状态；EEG PC1 是工程代理，低误差不证明唯一生理过程。",
        ],
        "sections": [],
        "sources": ["终态、区间和配对统计："+str(run/"summary.json"), "冻结清单："+str(run/"plan.json"),
                    "窗口指标："+str(evidence["metric_path"]), "被试汇总："+str(evidence["subject_path"]),
                    "示例规则与数组："+str(evidence["example_path"]),
                    "计算 owner：src/inference/tak_mismatch.py；执行 owner：experiments/scripts/evaluate_tak_mismatch.py"],
    }
    blocks = [
        _paragraph("每个计划项是数据集 × 被试 × 原生记录 × HbO/HbR 通道对 × 时间窗口。Hb-only 仅看 Hb，EEG-only 仅看 EEG，联合模型同时看三坐标。Hb-only 和联合模型均在当前评价窗口内拟合，因此是条件相容性分析；EEG-only 的 Hb 输出才用于独立 EEG→Hb 预测比较。"),
        _coverage(evidence),
        _paragraph("被试内聚合规则："+str(aggregation)+"。主 ECDF 每名有支持的被试一个已保存汇总值，累计比例表示被试比例；较细的时间加权 ECDF 使用窗口表按指标冻结的有效时间权重，表示保留支持上窗口分数的时间占比，不是逐点残差分布。未提供反重复时间权重时不绘制虚线。"),
        _paragraph("数值失败率与指标支持率分开解释。失败不进入成功子集分位数，但仍进入全计划分母；不同指标的有效支持可能不同，不能把非配对子集的中位数差当作模型改善。当前单个 Hb 对的自由空间投影不可检验，不能描述成空间匹配完美。"),
    ]
    inventory_rows = []
    for dataset, item in evidence.get("inventory", {}).items():
        inventory_rows.append([DATASET_LABELS.get(dataset, dataset), item.get("subjects", "未记录"),
            item.get("registered_records", "未记录"), item.get("planned_pair_records", "未记录"),
            item.get("available_pair_windows", "未记录"), _num(item.get("paired_support_seconds"), 1),
            _num(item.get("tail_uncovered_seconds"), 1)])
    if inventory_rows:
        blocks.insert(1, _table(["数据集", "清单被试", "原生记录", "记录内 Hb 对", "配对可用窗口对", "配对支持秒", "未覆盖尾段秒"], inventory_rows,
            "完整冻结清单包括没有可用窗口的原生记录；可用与计划分母不能互相替代。此表直接读取 plan.json 的清单汇总。", [1.2, .8, .9, 1.05, 1.2, 1.2, 1.2]))
    deferred = summary.get("D_obs")
    if deferred:
        blocks.append(_paragraph("Hb 可观测驱动差异的保留情景／状态："+str(deferred)+"。这里的阈值由预设合成噪声情景生成，不是从实测残差估计噪声；不能称为已校准的实测生理识别。"))
    if evidence.get("terminal_audit"):
        blocks.extend(terminal_status_blocks(evidence))
    else:
        view_rows = []
        for dataset in _datasets(evidence):
            for mode in ("fixed", "current_dof"):
                part = evidence["rows"][(evidence["rows"].dataset.astype(str) == dataset) &
                                        (evidence["rows"].parameter_mode == mode)]
                for view, label in (("hb_only", "Hb-only"), ("eeg_only", "EEG-only"), ("joint", "联合")):
                    column = view+"_success"
                    if column in part and part[column].notna().any():
                        successes = part[column].astype(str).str.lower().isin(("true", "1", "1.0"))
                        view_rows.append([DATASET_LABELS.get(dataset, dataset), MODE_LABELS[mode], label,
                                          len(part), int(successes.sum()),
                                          f"{100*(1-successes.mean()):.2f}%" if len(part) else "不可用"])
        if view_rows:
            blocks.append(_table(["数据集", "参数条件", "条件拟合", "全计划", "通过数值检查", "未通过／不可用"], view_rows,
                                 "未提供独立终态审计时，此表保留未通过／不可用合计；该比例含无支持，不称为数值失败率。原始原因见保留终态。"))
    missing = summary.get("missing", summary.get("unavailable", {}))
    if missing:
        blocks.append(_paragraph("保留的缺失／不可检验状态："+json.dumps(missing, ensure_ascii=False)))
    content["sections"].append({"title": "覆盖、信息条件与解释边界", "blocks": blocks})
    content["sections"].append({"title": "评分尺度和四轴的计算含义", "blocks": [
        _paragraph("保留目标使用前 5 秒作窗口参考，该参考段不评分；最后 0.1 秒按合同同时从拟合与评分排除。保存 target 的末 bin 为 NaN 不表示原缓存一定未测量。评分时长按有效 bin 数乘 0.1 秒计算：50:299 共 249 bins 为 24.9 秒，不能用首末评分时刻差 24.8 秒替代。"),
        _paragraph("模型计算路径为标量驱动 r → Tak 血流与血容量状态 → HbO/HbR 观测，同时通过冻结 EEG 读出生成代理观测。当前独立血流状态是 s、f、v、q；输出列 p 恒等于 v，不是额外生理自由度。Hb-only、EEG-only、联合条件改变可见观测，参考、尺度与评分支持保持同合同。"),
        _paragraph("EEG 代理的实际路径为六个近邻通道 × 五频带（1–4、4–8、8–13、13–30、30–45 Hz）→ 对数功率 → 前 5 秒参考 → 训练分区 PC1 → 冻结尺度，输出网格为 10 Hz。同一被试所有会话在同一外折，PCA、尺度、候选空间选择只看训练／内部选择被试。PC1 的数学符号规则不提供正向神经活动的生理定标。"),
        _paragraph("固定参数为主条件。当前自由度条件仅在同一原生连续支持／会话／Hb 对组内共享 κ、τ，Hb-only 与联合条件分别估计其组参数；各窗口仍有各自驱动与初态。空间干预仍使用固定参数的独立 EEG-only 预测，不增加单独 EEG-only 生理参数拟合。Hb-only 的 κ 受驱动先验影响，不视为从 Hb 独立识别的生理值。"),
        {"type": "equation", "latex": r"D_1^2=\frac{1}{2|I_H|}\sum_{t\in I_H}\left[\frac{(H_O-\widehat H_{H,O})^2}{\sigma_O^2}+\frac{(H_R-\widehat H_{H,R})^2}{\sigma_R^2}\right]", "caption": "Hb-only 数据误差；共同有效 Hb 时间支持上两个色团等权。"},
        _paragraph("H 为处理后的实测 Hb；下标 H 表示只用 Hb 拟合的预测。I_H 是 HbO/HbR 共同有效的评分时刻，sigma 是训练分区冻结 SD，不是当前窗口 SD，也不自动等于传感器噪声。训练尺度用于评分，不等于把输入两个色团分别重新归一化。"),
        {"type": "equation", "latex": r"\begin{aligned}w_j&=U_j-V_j,&D_{\mathrm{bal}}&=\sqrt{\frac{\sum_jw_j^2}{2\sum_j(U_j^2+V_j^2)}}\\D_{\mathrm{bal}}^2&=\frac12-\frac{\sum_j U_jV_j}{\sum_j(U_j^2+V_j^2)}\\D_{EH}&=\sqrt{\operatorname{mean}_{I_E}\left[(e-B_E\widehat r_H)/\sigma_E\right]^2}\end{aligned}", "caption": "弱动态平衡相对偏离，以及 Hb 条件驱动的 EEG 读出误差。"},
        _paragraph("D_bal 取决于两个核积分序列的相对内积与能量。接近 0.707 既可能来自 U 与 V 近正交，也可能来自其中一侧的能量远弱于另一侧；它绝不是“70% 未解释方差”，也不是失配概率。跨库高低排序只描述当前观测坐标和核尺度下的动态相容关系，不据此判断哪个数据集的生理过程更好。"),
        _paragraph("弱残差用完整 10、20、40 秒核，不对实测 Hb 求导；U、V 是消驱动线性关系两侧的核积分。缺失核支持不可用；分母数值为零时不可用。目前没有经噪声校准的低动态阈值，近零但非零分母的比值可能不稳定。D_bal 是描述性小信号比值，共同正尺度变化不改变该比值；分母 mean_denominator 的绝对大小只在同库、同原生单位和同核宽度内解释，不能跨库直接比较。B_E 包含 EEG 观测参考，e 为 EEG 代理，I_E 是有效 EEG 评分时刻；不逐窗翻转符号、重调尺度或搜索时移。"),
        {"type": "equation", "latex": r"\begin{aligned}\Delta_{\mathrm{spatial}}&=L_H(\widehat H_{g_0,E})-L_H(\widehat H_{g_\star,E})\\D_4&=\sqrt{(D_{J,E}^2+D_{J,O}^2+D_{J,R}^2)/3}\end{aligned}", "caption": "空间指标为平方误差差值；联合指标为三个标准化残差的等权均方根。"},
        _paragraph("g0 是当前最近六通道，g* 是训练／内部选择后冻结的同容量替代六通道；均仅用 EEG 生成 Hb，评分支持相同。空间正值表示替代方案改善，保留负值；它不是毫米距离。J 表示联合拟合，E/O/R 分别是 EEG/HbO/HbR。联合驱动已使用 Hb，不能据此证明 EEG 具有独立预测价值。"),
        {"type": "equation", "latex": r"D_{\mathrm{obs}}=\frac{\left\|\Sigma_K V_K^\top(\widehat r_E-\widehat r_H)\right\|_2}{\sqrt{N_H}}", "caption": "固定条件剔除初态与参考 DC；组级解处逐窗放松量还按窗口剔除 κ、τ 方向。两者均按情景阈值保留模式。"},
        _paragraph("Sigma、V 来自训练尺度加权的 Tak 驱动雅可比奇异分解；组级参数条件下的逐窗放松可观测差在共享参数拟合解处，逐窗额外剔除 κ、τ 局部方向；数学上允许每个窗口独立局部参数补偿，比实际整组共享参数的补偿集合更宽，不等于全组共享约束下的可检测差。保留秩可与固定条件不同，K 是超过预设合成噪声阈值的模式，N_H 是共同支持上的 Hb 标量观测数（两个色团合计）。零保留秩不可用，不记作零失配。固定驱动范数扰动、噪声 SD、自相关和色团间相关均属于情景假设。"),
        _paragraph("D_obs 的两参数口径并非相同 nuisance 约束下的同义分数：逐窗放松量下降不能称为实际组共享模型改善，直接观测残差改善主要看同支持的 D1、D4。阈值导致保留模式集改变，也不笼统称严格数值下界。D_obs 如有输出，必须结合保存的噪声情景、有效秩与共同支持解释。未做实测噪声校准时，局部雅可比阈值只能是情景诊断，不能称为实测已识别驱动。D_drive_H 是共同参数和共同初态下的非线性回放差异；它与各自重新拟合初态后的 Hb 残差不同。当前自由度条件保留固定 EEG-only 驱动，与组级 Hb-only 驱动在该 Hb-only 组参数及共同静息初态下回放；不冒充重新拟合生理参数的 EEG 独立预测。"),
    ]})
    method_blocks = content["sections"][-1]["blocks"]
    method_blocks[0:0] = hb_unit_blocks(evidence)
    method_blocks.extend([
        {"type": "equation", "latex": r"\begin{aligned}e_{m,c}(t)&=\frac{y_c(t)-\widehat y_{m,c}(t)}{\sigma_c},&\operatorname{RMS}_{m,c}&=\sqrt{\frac1{|I_c|}\sum_{t\in I_c}e_{m,c}(t)^2}\\F_{m,c,k}&=\frac{\sum_{t\in I_c\cap B_k}e_{m,c}(t)^2}{\sum_{t\in I_c}e_{m,c}(t)^2}\end{aligned}", "caption": "仅为典型样例读图计算的标准化残差 RMS 与三等分评分时段平方和份额；不改变全数据 owning 指标。"},
        _paragraph("m 表示固定／组级参数下的 Hb-only 或联合视图，c 表示 EEG、HbO、HbR，sigma 是同一训练冻结 SD。I_c 对 EEG 是自身 score_mask，对 Hb 是两个色团共同 mask。时间皆相对所选窗口起点；把首个评分 bin 到最后评分 bin 末端的物理跨度三等分为 B_k，逐段同时保留有效 bin 数、段 RMS 与平方和份额。评分时长用 bin 数乘采样步长，不用端点差；缺失预测不缩减后当成完整支持。全残差零时没有主导区间；残差或空间极值若并列，均取最早评分格点，不把其时刻或边界效应解读为独立生理时延。空间正、负、零比例按共同有效评分 bins 的时长计，零指严格数值零，不另设调参阈值。"),
    ])
    fixed = summary.get("fixed_parameters", {})
    if fixed:
        method_blocks.insert(2, _table(["固定参数", "数值", "解释坐标"],
            [[name, _num(value, 4), "秒" if name in ("tau", "viscoelastic_s") else "当前 Tak 参考坐标"]
             for name, value in fixed.items() if isinstance(value, (float, int))],
            "固定值直接读取本次 summary；P0、Q0 为模型观测尺度，不直接解释为绝对 Hb 浓度。"))
    settings = evidence.get("config", {}).get("current_dof", {})
    if settings:
        method_blocks.append(_paragraph("组参数冻结合同："+
            f"名称={settings.get('parameter_names')}，边界={settings.get('parameter_bounds')}，参考={settings.get('reference')}，对数先验 SD={settings.get('log_prior_sd')}，起始参数={settings.get('starts')}；每组先验仅计一次。触边阈值为对数参数边界相对距离 {settings.get('boundary_fraction')}。初值来自同可见性的固定条件或零驱动静息态，不由隐藏模态热启动。"))
    noise = summary.get("observability", {})
    if noise:
        noise_rows = [[item.get("id", "未记录"), _num(item.get("noise_sd")), _num(item.get("ar1")),
                       _num(item.get("cross_correlation"))] for item in noise.get("scenarios", [])]
        if noise_rows:
            method_blocks.append(_table(["噪声情景（首行为主）", "processed-Hb 标准化 SD", "AR1", "HbO/HbR 相关"], noise_rows,
                f"每情景 {noise.get('draws', '未记录')} 次合成抽样，模式投影噪声分位数 {noise.get('quantile', '未记录')}，固定种子 {noise.get('seed', '未记录')}。驱动扰动尺度使用冻结训练 EEG SD。"))
            method_blocks.append(_paragraph("阈值比较的是预设范数驱动扰动在各奇异模式的 Hb 响应与该模式的合成噪声投影分位数，不是对当前实测残差做噪声拟合。敏感性条件保留同一数据、坐标和拟合轨迹，仅改变噪声情景；D_obs 的低值须同时看保留秩、失败支持和非线性回放差异。"))
    metrics = [name for name in METRICS if name in set(evidence["subjects"].metric)]
    axes = [
        ("第一轴：不看 EEG 的 Hb 相容性", [m for m in ("D1", "D_bal_10s", "D_bal_20s", "D_bal_40s") if m in metrics]),
        ("第二轴：EEG 与 Hb 条件驱动的差异", [m for m in ("D_EH", "D_obs", "D_drive_H") if m in metrics]),
        ("第三轴：当前近邻规则的留出预测代价", [m for m in ("Delta_spatial",) if m in metrics]),
        ("第四轴：联合模型的误差与折中", [m for m in ("D4", "D_J_E", "D_J_O", "D_J_R") if m in metrics]),
    ]
    for title, names in axes:
        blocks = []
        if title.startswith("第三"):
            blocks.extend([
                {"type": "equation", "latex": r"\begin{aligned}d_{\mathrm{spatial}}(t)&=\frac12\sum_{c\in\{O,R\}}\left[\frac{(H_c-\widehat H_{g_0,E,c})^2}{\sigma_c^2}-\frac{(H_c-\widehat H_{g_\star,E,c})^2}{\sigma_c^2}\right]\\\Delta_{\mathrm{spatial}}&=\operatorname{mean}_{t\in I_H}d_{\mathrm{spatial}}(t)\end{aligned}", "caption": "同一评分时刻上两个色团等权的瞬时平方误差差，时间均值对应保留窗口指标。"},
                _paragraph("空间时间曲线使用保留的 target、当前最近六通道 EEG-only 预测和训练选定六通道预测，除以同一冻结 Hb SD；只在共同 score_mask、两种预测均通过数值检查且保留 Δ_spatial 可用时展示，并核验时间均值。正值表示当时替代方案误差更低，负值表示当前方案更低。这是随记录时间的误差代价，不是 EEG 时移／lag 扫描，本报告不开展新扫描。"),
                _paragraph("所选方案允许等于原最近六通道。selected_candidate=nearest6 时，训练选择保留原方案，两侧复用同一预测，Δ_spatial=0 是自比较结果；不能解释为两个不同近邻方案相同，更不能解释为空间映射没有误差。"),
            ])
            selection_table = spatial_selection_table(evidence)
            if selection_table:
                blocks.append(selection_table)
        if title.startswith("第一"):
            blocks.extend(weak_verification_blocks(evidence))
            blocks.append(_paragraph("弱动态平衡偏离是两侧核积分的相对关系分数。约 0.707 不能读作约 70% 的未解释方差；近正交和一侧能量弱都可产生该值。跨库排序为描述性比较，不构成生理优劣判断。"))
        if not names:
            blocks.append(_paragraph("该轴没有保留可用指标，不能由其他轴替代；请参照覆盖和缺失状态。"))
        for metric in names:
            blocks.append(_paragraph(_metric_interpretation(evidence, metric)))
            path = distribution_figure(evidence, metric, directory/f"distribution_{metric}.png")
            caption = f"{METRICS[metric][0]}。各数据集共享横轴刻度；粗线是被试汇总 ECDF，细线为有明确反重复权重时的时间 ECDF，具体模型和参数线型见图例。非有限值不填零；有效人数见图例和统计表。时间曲线是按持续时间加权的窗口分数；Dbal 的主统计来自独立 record-pair，而非每窗重复附件。"
            if metric == "D_obs":
                caption += "组级解曲线是逐窗剔除 κ、τ 方向的放松诊断，比实际整组共享参数的补偿集合更宽；其下降不作组共享模型改善结论，也不笼统称严格数值下界。"
            blocks.append(_figure(path, caption, str(evidence["subject_path"])+"；"+str(evidence["metric_path"])))
        if names:
            blocks.append(_metric_table(evidence, names))
            intervals = []
            supports = []
            for dataset in _datasets(evidence):
                for mode, owner in (("fixed", "by_dataset"), ("current_dof", "by_dataset_current_dof")):
                    source = summary.get(owner, {}).get(dataset, {}).get("metrics", {})
                    for name in names:
                        key = name+"_free" if mode == "current_dof" else name
                        item = source.get(key, {})
                        if "subject_equal_mean" not in item:
                            continue
                        interval = item.get("subject_bootstrap_95pct")
                        interval_label = "不可用" if interval is None else f"[{_num(interval[0])}, {_num(interval[1])}]"
                        mode_label = "组级解逐窗放松" if mode == "current_dof" and name == "D_obs" else MODE_LABELS[mode]
                        intervals.append([DATASET_LABELS.get(dataset, dataset), mode_label, name,
                                          _num(item.get("subject_equal_mean")), interval_label,
                                          item.get("subjects", "未记录")])
                        supports.append([DATASET_LABELS.get(dataset, dataset), mode_label, name,
                                         item.get("planned", "未记录"), item.get("available", "未记录"),
                                         "record-pair" if name.startswith("D_bal_") else "window-pair"])
            if intervals:
                blocks.append(_table(["数据集", "参数条件", "指标", "被试等权汇总", "95% 区间", "有效被试"], intervals,
                    "区间直接取自 owning summary：整个被试为重采样单位，记录和窗口随被试一起抽取，训练坐标与选择冻结；描述性区间不作多重检验显著性声明。"))
            if supports:
                blocks.append(_table(["数据集", "参数条件", "指标", "全计划", "有效支持", "观测单元"], supports,
                    "每项指标独立保留计划分母和有效支持；Dbal 用独立记录内 Hb 对，其他指标用非重叠窗口内 Hb 对。分位数仅描述有效子集，失败仍保留在计划分母。"))
        if title.startswith("第二"):
            noise_rows = []
            for dataset in _datasets(evidence):
                for mode in ("fixed", "current_dof"):
                    for name in ("D_obs", "D_obs_ar1_sd0p05", "D_obs_ar1_sd0p2", "D_obs_iid_sd0p1"):
                        values = _subject_values(evidence, dataset, name, mode)
                        if len(values):
                            noise_rows.append([DATASET_LABELS.get(dataset, dataset), "固定" if mode == "fixed" else "逐窗放松",
                                               name, len(values), _num(np.median(values)), _num(np.quantile(values, .9))])
            if noise_rows:
                blocks.append(_table(["数据集", "参数", "预设噪声情景", "有效被试", "中位数", "P90"], noise_rows,
                    "D_obs 为主 AR1/SD0.1 情景，其余为噪声假设敏感性；组级列表示共享解处逐窗放松量，不与固定量作模型改善解释。它们使用相同保留拟合轨迹，但保留模式与有效支持可变，不作实测生理可识别的结论。", [1.1, .8, 2.1, .8, .8, .8]))
        if title.startswith("第三"):
            baseline_names = ("prediction_NMSE", "selected_prediction_NMSE", "zero_NMSE", "template_NMSE", "pairing_gain", "selected_pairing_gain")
            baseline_rows = []
            for dataset in _datasets(evidence):
                for name in baseline_names:
                    values = _subject_values(evidence, dataset, name)
                    if len(values):
                        baseline_rows.append([DATASET_LABELS.get(dataset, dataset), name, len(values),
                                              _num(np.median(values)), _num(np.quantile(values, .9))])
            if baseline_rows:
                blocks.append(_table(["数据集", "预测／对照指标", "有效被试", "中位数", "P90"], baseline_rows,
                    "prediction 为当前六近邻仅用 EEG 的 Hb 预测；selected 为冻结替代六通道；zero 为零变化；template 为训练模板。pairing_gain=错配 EEG 的 Hb 均方误差减正确 EEG 的误差，正值才支持正确配对更好；selected_pairing_gain 同理。均方误差用 Hb 训练 SD 平方。", [1.1, 2.2, .9, .9, .9]))
                blocks.append(_paragraph("替代近邻的绝对误差降低还需结合正确与错配 EEG 的优势判断；单纯更接近零或训练模板也可能降低误差。表中各指标支持人数可能不同，未声明共同支持的跨行中位数差只作描述，不能当成配对效果。"))
        if title.startswith("第四"):
            blocks.append(_paragraph("比较 Hb-only、EEG-only 和联合轨迹时，应同时检查 HbO/HbR 与 EEG 的残差。联合总分降低可能伴随某个观测分量恶化；组级 κ、τ 自由度的结果只在保留证据存在时并列，未运行的参数条件不补写数值。数值数据项与优化器含先验的完整目标不同，数据项差值允许为负。"))
        content["sections"].append({"title": title, "blocks": blocks})
    current = evidence.get("current_rows")
    if current is not None and not current.empty and "group_id" in current:
        parameter_rows = []
        for dataset in _datasets(evidence):
            part = current[current.dataset.astype(str) == dataset].groupby("group_id", dropna=False).first()
            for view, label in (("hb", "Hb-only"), ("joint", "联合")):
                kappa_key, tau_key = "kappa_"+view, "tau_"+view
                boundary_key = ("hb_only" if view == "hb" else "joint")+"_parameter_boundary"
                if kappa_key not in part or tau_key not in part:
                    continue
                kappa = np.asarray(part[kappa_key], float)
                tau = np.asarray(part[tau_key], float)
                valid = np.isfinite(kappa) & np.isfinite(tau)
                boundary = part[boundary_key].astype(str).str.lower().isin(("true", "1", "1.0")) if boundary_key in part else None
                parameter_rows.append([DATASET_LABELS.get(dataset, dataset), label, len(part), int(valid.sum()),
                    _num(np.median(kappa[valid])) if valid.any() else "不可用",
                    _num(np.median(tau[valid])) if valid.any() else "不可用",
                    int(boundary[valid].sum()) if boundary is not None else "未记录"])
        if parameter_rows:
            content["sections"].append({"title": "组级参数能吸收失配，但不等于恢复生理真值", "blocks": [
                _table(["数据集", "条件", "计划组", "参数有限组", "κ 中位数", "τ 中位数 / s", "触边组数"], parameter_rows,
                    "每个参数组只计一次，未把重复窗口当作独立参数估计。触边定义直接沿用运行合同；参数有限不等于优化收敛，成功率见覆盖表。"),
                _paragraph("误差下降表示在声明的驱动、初态和组参数先验下相容性改善。大量触边或先验代价改变提示模型用较极端设置吸收失配；Hb-only 中 κ 可由驱动补偿，不能把这些条件估计称为独立生理测量。"),
            ]})
    selection = evidence["example_index"].get("selection", evidence["example_index"].get("rule", "以 examples.json 保存规则为准"))
    content["sections"].append({"title": "典型窗口如何选择", "blocks": [
        _paragraph("选择规则："+str(selection)+"。示例用于解释，不用于调参或确认性比较。各数据集在 D1 有效候选中选接近中位数、P90 和最高有效失配的窗口；尽量使用不同被试，并保留实际排序与分母。不能把单个典型窗口当成整个数据集。"),
        _table(["数据集", "角色", "被试／记录／通道对／窗口", "排序／候选分母"], [
            [DATASET_LABELS.get(e["dataset"], e["dataset"]), str(e["role"]),
             f"{e['subject']} / {e['record']} / {e['pair']} / {e['window']}",
             f"{e.get('rank', '未记录')} / {e.get('selection_denominator', e.get('candidate_denominator', e.get('denominator', '未记录')))}"] for e in evidence["examples"]],
            "精确身份与选择目标以 examples.json 为准。", [1.2, 1, 3, 1.4]),
    ]})
    # Fix Hb axis ranges across the selected examples within each dataset;
    # keep HbO and HbR separate because their native amplitudes differ.
    attached = [_attach_current_example(evidence, example, _array_data(evidence, example))
                for example in evidence["examples"]]
    example_arrays = [item[0] for item in attached]
    example_current_rows = [item[1] for item in attached]
    hb_limits = {}
    for dataset in _datasets(evidence):
        limits = []
        for component in (1, 2):
            values = []
            for example, arrays in zip(evidence["examples"], example_arrays):
                if example["dataset"] != dataset:
                    continue
                factor = float(arrays.get("hb_factor", 1.))
                for key in ("target", "hb_only_prediction", "eeg_only_prediction", "joint_prediction", "free_hb_only_prediction", "free_joint_prediction"):
                    if key not in arrays:
                        continue
                    value = arrays[key][:, component]/factor
                    valid = np.isfinite(arrays["target"][:, 1:]).all(axis=1) & np.isfinite(value)
                    values.extend(value[valid].tolist())
            if values:
                lower, upper = min(values), max(values)
                margin = max((upper-lower)*.06, abs(upper)*.01, 1e-12)
                limits.append((lower-margin, upper+margin))
        if len(limits) == 2:
            hb_limits[dataset] = limits
    for index, (example, arrays, current_row) in enumerate(zip(evidence["examples"], example_arrays, example_current_rows)):
        row = _example_row(evidence, example)
        title = f"{DATASET_LABELS.get(example['dataset'], example['dataset'])}／{example['role']}：{example['subject']}，{example['record']}，Hb 对 {example['pair']}"
        source = str(_path(run, example["array_path"]))
        validation = _example_validation(example, row, current_row)
        fixed_record = dict(row, **example)
        fixed_record["Delta_spatial"] = row.get("Delta_spatial", example.get("Delta_spatial"))
        spatial = spatial_time_curve(arrays, fixed_record)
        main = example_observation_figure(arrays, directory/f"example_{index:02}_observations.png",
                                          title=f"{DATASET_LABELS.get(example['dataset'], example['dataset'])} / {example['role']} · {example['subject']} · Hb 对 {example['pair']}",
                                          hb_unit=example.get("hb_unit", hb_unit_label(evidence, example["dataset"])),
                                          limits=hb_limits.get(example["dataset"]), validation=validation)
        weak = example_weak_spatial_figure(evidence, example, arrays, directory/f"example_{index:02}_weak_spatial.png", spatial=spatial)
        residual = example_residual_figure(arrays, directory/f"example_{index:02}_residual.png", validation=validation)
        scores = "; ".join(f"{name}={_num(row.get(name))}" for name in ("D1", "D_EH", "D_obs", "D_drive_H", "Delta_spatial", "D4"))
        joint = "; ".join(f"{name}={_num(row.get(name))}" for name in ("D_J_E", "D_J_O", "D_J_R"))
        time_diagnostics = example_time_diagnostics(arrays, validation, spatial)
        time_blocks = example_time_blocks(time_diagnostics, source)
        weak_text = "; ".join(f"{width}s D_bal={_num(row.get('D_bal_'+str(width)+'s'))}" for width in (10, 20, 40))
        prediction_text = "; ".join(f"{name}={_num(row.get(name))}" for name in
            ("prediction_NMSE", "selected_prediction_NMSE", "zero_NMSE", "template_NMSE", "wrong_NMSE", "pairing_gain", "selected_pairing_gain"))
        observable = example_observable_figure(arrays, directory/f"example_{index:02}_observable.png")
        validation_names = {"hb_only_prediction": "固定 Hb-only", "eeg_only_prediction": "固定 EEG-only", "joint_prediction": "固定联合", "free_hb_only_prediction": "组级κτ Hb-only", "free_joint_prediction": "组级κτ 联合"}
        status_text = "; ".join(f"{validation_names[key]}：{verdict['label']}（{verdict['raw_status']}）" for key, verdict in validation.items())
        spatial_text = (f"空间时间曲线的 {spatial['count']} 个共同评分时刻均值为 {_num(spatial['mean'], 6)}，已核对保留 Δ_spatial={_num(spatial['Delta_spatial'], 6)}。" if spatial["status"] == "completed" else "空间时间曲线："+spatial["reason"]+"。")
        example_blocks = [
            _paragraph(f"身份：window={example['window']}；task={example.get('task_id', example.get('id', '见清单'))}；参数口径={MODE_LABELS.get(example.get('parameter_mode', 'fixed'), example.get('parameter_mode'))}。本页所有线来自保留数组；灰阴影表示可见但不计分支持（含参考段），断线表示未显示／未评分支持（含真实缺失或合同排除），不跨不可用段连线。"),
            _paragraph("各条件的保留数值状态："+status_text+"。未通过或状态未核实的有限轨迹仅作诊断，不作为有效拟合或成功指标。"),
            _paragraph(scores+"。"+joint+"。"),
            _paragraph("独立预测与基线（Hb 均方误差；越低越好）："+prediction_text+"。所选空间方案："+str(row.get("selected_candidate", "未记录"))+"。配对优势为错配误差减正确配对误差，正值才表示正确 EEG 更好。"),
            _figure(main, "上两幅是带通缓存、前 5 秒窗口参考后的实测 HbO/HbR、Hb-only、联合重建及仅用 EEG 的 Hb 预测；第三幅比较 EEG 代理与三种模型的 EEG 读出，统一用冻结 EEG SD。第四幅为三种条件驱动，驱动不是血流状态 s。各面板统一沿用顶部模型图例及线型；Hb-only 虚线、联合实线、EEG-only 点线、组级 Hb-only 点划线、组级联合长短划线，颜色辅助区分。同一数据集的三个示例共享各自 HbO/HbR 纵轴；Hb 已除以训练共同因子回到原缓存单位（若因子已保存），不把各曲线逐条标准化。", source),
            *time_blocks,
            _figure(weak, "前三幅是完整记录中以 10、20、40 秒完整支持核计算的弱残差；阴影是所选拟合窗口，窗口外曲线只用于上下文。支持不足或未保存时明确显示不可用。末幅为两个已通过数值检查预测在共同评分时刻的空间瞬时平方误差差；横轴是记录时间，正值表示所选方案更好，灰虚线为窗口均值。它不是时移／lag 扫描。", str(_path(run, example["weak_path"])) if example.get("weak_path") else source),
            _paragraph("所附独立完整记录的固定参数汇总："+weak_text+"。这些 D_bal 不是仅对所选 30 秒窗口计算；组级 τ 条件的支持可能因连续组边界与尾段而不同，未声明共同支持时不作配对改善结论。弱残差的纵轴为有符号核积分，量纲是原缓存 Hb 单位 × 秒；不能直接与无量纲 D_bal 比较或共轴解释；不同核宽度也不按同一阈值解释。核间重叠不构成独立重复。"+spatial_text+"没有有效 Δ_spatial、共同支持或通过状态时不显示空间代价曲线，也不填零。"),
            _figure(residual, "每条线为实测减对应 Hb-only／联合预测，再除以相同的训练 SD；EEG 行的 Hb-only 曲线就是 d_EH(t)，两个 Hb 行的 Hb-only 曲线对应 D1 的有符号分量。联合曲线对应 D4 的三个分量。零线代表该时刻无差，图例标明数值状态；EEG 用自身支持，Hb 用共同支持。", source),
            _paragraph("联合三分量的已保存 RMS 为："+joint+"。正残差表示预测偏低，负残差表示预测偏高；孤立尖峰与持续偏差对 RMS 的贡献不同。此例是按 D1 排序选择，不能据此推断独立 EEG 预测有效、驱动唯一或参数真实。"),
        ]
        if current_row:
            current_scores = "; ".join(f"{'D_obs逐窗放松' if name == 'D_obs' else name}={_num(current_row.get(name))}" for name in
                ("D1", "D_EH", "D_obs", "D_drive_H", "D4", "D_J_E", "D_J_O", "D_J_R"))
            example_blocks.insert(3, _paragraph("同一窗口的组级 κ、τ 条件："+current_scores+"。κ、τ 在原生连续记录／会话／Hb 对组内共享，各条件独立拟合；驱动和初态仍按窗口拟合，未增加单独 EEG-only 参数条件。"+f"该组计划 {current_row.get('group_planned_windows', '未记录')} 窗；Hb-only κ={_num(current_row.get('kappa_hb'))}、τ={_num(current_row.get('tau_hb'))} s；联合 κ={_num(current_row.get('kappa_joint'))}、τ={_num(current_row.get('tau_joint'))} s。"))
        if observable:
            example_blocks.extend([
                _figure(observable, "曲线为保留模式上 EEG-only 驱动减 Hb-only 驱动的局部 Hb 影响，已去除允许初态／参考方向并除以训练 Hb SD；它不是实测减预测残差，也不是生理真值。", source),
                _paragraph(f"本例主情景 D_obs={_num(row.get('D_obs'))}；保留秩={example.get('observable_retained_rank', row.get('observable_retained_rank', '未记录'))}，数值秩={example.get('observable_numerical_rank', row.get('observable_numerical_rank', '未记录'))}。应与共同初态非线性回放 D_drive_H={_num(row.get('D_drive_H'))} 交叉解释；局部阈值敏感性不能直接认定实测驱动已识别。"),
            ])
        scenario_rows = []
        for name, item in example.get("observable_scenarios", {}).items():
            scenario_rows.append([name, _num(item.get("noise_sd")), _num(item.get("ar1")),
                                  _num(item.get("cross_correlation")), item.get("retained_rank", "未记录"),
                                  _num(item.get("D_obs"))])
        if scenario_rows:
            example_blocks.append(_table(["合成噪声情景", "Hb 标准化 SD", "AR1", "色团相关", "保留秩", "D_obs"], scenario_rows,
                "这些 SD 是预先规定的 processed-Hb 噪声假设，不由当前实测拟合残差估计；有效秩和分数随假设变化是敏感性结果。", [1.8, 1.1, .7, .8, .8, .8]))
        content["sections"].append({"title": title, "page_break": True, "blocks": example_blocks,
                                    "time_diagnostics": time_diagnostics})
    if evidence.get("weak_verification_path"):
        content["sources"].append("弱残差独立数值与全量状态核验："+str(evidence["weak_verification_path"]))
    if evidence.get("terminal_audit_path"):
        content["sources"].append("拟合支持与数值终态审计："+str(evidence["terminal_audit_path"]))
    for item in evidence.get("hb_provenance", []):
        content["sources"].append("Hb 单位与缓存变换来源："+item["metadata_path"])
    for dataset in _datasets(evidence):
        values = []
        for name in ("D1", "D_EH", "D_obs", "Delta_spatial", "D4"):
            score = _subject_values(evidence, dataset, name)
            values.append(f"{name} 中位数={_num(np.median(score)) if len(score) else '不可用'}")
        paired = summary.get("paired_current_dof", {}).get(dataset, {}).get("D4", {})
        text = DATASET_LABELS.get(dataset, dataset)+"："+"，".join(values)+"。"
        if paired:
            text += f"在 {paired.get('paired_subjects', '未记录')} 名共同支持被试上，固定减组级 κ、τ 的 D4 平均差为 {_num(paired.get('fixed_minus_free_subject_mean'))}；正值表示组级参数条件残差更低。"
        content["summary"].append(text)
    return content


def render_tak_mismatch(run, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"report output must be fresh; retained exports are never overwritten: {output}")
    evidence = load_evidence(run)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tak_mismatch_report_", dir=output.parent) as temporary:
        content = build_content(evidence, Path(temporary))
        content_path = Path(temporary)/"report_content.json"
        content_path.write_text(json.dumps(content, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        checks = render_report(content_path, output)
    checks["content_source"] = str(output/"content.json")
    (output/"delivery_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    index = {"schema": SCHEMA, "run": str(evidence["run"]), "summary": str(evidence["run"]/"summary.json"),
             "metrics": str(evidence["metric_path"]), "subjects": str(evidence["subject_path"]),
             "examples": str(evidence["example_path"]), "selection": evidence["example_index"],
             "scope": "read_only_completed_evidence", "metric_recomputation": False,
             "display_consistency_checks": ["spatial_time_curve_mean_equals_retained_Delta_spatial_on_common_validated_support"],
             "selected_example_diagnostics": "content.json sections[].time_diagnostics; residual RMS/peaks/thirds and spatial sign durations from selected retained arrays only; cohort metrics stay owned by run tables",
             "weak_verification": str(evidence["weak_verification_path"]) if evidence.get("weak_verification_path") else None,
             "terminal_status_audit": str(evidence["terminal_audit_path"]) if evidence.get("terminal_audit_path") else None}
    (output/"report_evidence.json").write_text(json.dumps(index, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    checks = render_tak_mismatch(args.run, args.output)
    print(json.dumps({"output": checks["output"], "pages": checks["exports"]["TECHNICAL_REPORT.pdf"]["pages"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
