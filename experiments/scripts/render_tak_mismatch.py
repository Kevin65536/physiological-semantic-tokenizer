#!/usr/bin/env python3
"""Render retained Tak mismatch evidence without opening measured datasets.

Use ``--run <completed-run> --output <fresh-export>``. The report reads only
summary.json, metrics.csv, subject_summary.csv, examples.json and the explicit
example/weak array paths listed there, plus an optional owning
weak_verification.json and terminal_status_audit.json. Scientific scores and fitting remain
owned by the evaluator and ``src.inference.tak_mismatch``. All figures and
LaTeX equations are bitmap images in both PDFs; Chinese prose, tables and
captions remain searchable through the shared report renderer.

``--fixed-analysis <run/fixed_physiology_v1>`` instead reads the explicitly
declared completed fixed-parameter analysis tables. This branch does not open
the current_dof outputs; scalar census and bounded shape/EOG panel
supports stay separately labelled in the report.

``--revise-report <preserved-export> --visual-channel-audit <owning-audit>``
creates a fresh interpretation-only revision from portable report assets.
It neither opens signal arrays nor recomputes numerical evidence.

``--spatial-analysis <completed-spatial-patterns>`` plots registered Hb-site
maps and every selected subject's retained records with MNE-style topomaps.
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
FIXED_ANALYSIS_SCHEMA = "tak_mismatch_fixed_physiology_analysis_v1"
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
    "D_EH": ("Hb-only 解的 EEG 读出误差", "EEG 训练 SD", "低值表示 Hb-only 驱动的 EEG 读出更接近 EEG 代理。"),
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
FIXED_METRICS = {
    **METRICS,
    "D_obs": ("固定参数局部 Hb 可观测驱动差异", "Hb 训练 SD",
              "剔除初态与参考 DC 方向后，在情景阈值保留的局部 Hb 可观测模式中评分；零保留秩不可用，低值不证明真实生理驱动相同。"),
    "Hb_residual_to_zero_MSE_ratio": (
        "Hb 残差相对零预测的均方误差比", "同支持 MSE 比值",
        "先在同一窗口支持聚合 Hb-only 与零变化预测的均方误差，再相除；可大于 1，不是未解释方差、R² 或生理能量份额。"),
}
FIXED_MAIN_METRICS = ("D1", "Hb_residual_to_zero_MSE_ratio", "D_EH", "D_drive_H",
                      "D_bal_20s", "D_obs", "D4", "Delta_spatial", "D_J_E", "D_J_O", "D_J_R")
OCULAR_LABELS = {
    "EEG_proxy_amplitude": "EEG 代理幅度", "Hb_readout_disagreement": "Hb 解读出差",
    "independent_drive_disagreement": "独立驱动差", "D_EH_squared": "EEG 读出均方误差",
    "EOG_metric_rank_correlation": "EOG/失配秩相关",
    "rank_correlation_given_EEG_and_Hb_amplitude": "控幅后秩残余相关",
    "rho0_minus_mean_nonwrapping_pm5s": "零位移 − ±5 秒均值",
    "cross_window": "窗间秩相关", "cross_window_partial": "控幅后相关",
    "clean_minus_raw": "clean − raw", "clean_minus_amplitude": "clean − amp",
    "amplitude_minus_raw": "amp − raw",
}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _num(value, digits=3):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "不可用"
    if not math.isfinite(value):
        return "不可用"
    if value and abs(value) < 10.**(-digits):
        return f"{value:.{digits}g}"
    return f"{value:.{digits}f}"


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
    return {"type": "paragraph", "text": str(text).replace("−", "-")}


def _table(columns, rows, caption="", widths=None):
    result = {"type": "table", "columns": [column.replace("−", "-") for column in columns],
              "rows": [[value.replace("−", "-") if isinstance(value, str) else value for value in row] for row in rows],
              "caption": caption.replace("−", "-")}
    if widths:
        result["widths"] = widths
    return result


def _figure(path, caption, source):
    return {"type": "figure", "path": str(path), "caption": caption.replace("−", "-"), "source": str(source)}


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


def load_fixed_analysis(run, analysis):
    """Read only the fixed-parameter analysis manifest and its declared files."""
    import pandas as pd

    run, analysis = Path(run).resolve(), Path(analysis).resolve()
    if not analysis.is_relative_to(run):
        raise ValueError("fixed analysis must belong to the declared retained run")
    summary = _json(analysis/"summary.json")
    if summary.get("schema") != FIXED_ANALYSIS_SCHEMA:
        raise ValueError("unsupported fixed physiology analysis schema")
    if summary.get("status") != "completed" or summary.get("parameter_mode") != "fixed":
        raise ValueError("fixed report requires completed fixed-parameter analysis")
    files = summary.get("files", {})
    frames, paths = {}, {}
    table_keys = {"subjects", "subject_tasks", "subject_descriptives", "task_descriptives", "task_comparisons",
                  "repeat_pairs", "repeat_summary", "within_record_pairs", "within_record_summary",
                  "shape_subjects", "ocular_associations", "ocular_examples", "ocular_counterfactual_subjects"}
    for key, name in files.items():
        path = _path(analysis, name)
        if "current_dof" in path.relative_to(run).parts:
            raise ValueError("fixed report cannot consume current_dof artifacts")
        paths[key] = path
        if key in table_keys and path.suffix == ".csv":
            try:
                frames[key] = pd.read_csv(path, low_memory=False, dtype={name: str for name in
                    ("dataset", "subject", "task_label", "record_id", "record", "pair", "repeat_id", "task_a", "task_b")})
            except pd.errors.EmptyDataError:
                frames[key] = pd.DataFrame()
    for key, identity in (("subjects", ["dataset", "subject", "metric"]),
                          ("subject_tasks", ["dataset", "subject", "task_label", "metric"])):
        if key not in frames or not set(identity+["value", "planned_windows", "available_windows"]) <= set(frames[key]):
            raise ValueError(f"fixed analysis requires declared {key} long table and support denominators")
        frame = frames[key]
        if frame.duplicated(identity).any():
            raise ValueError(f"duplicate fixed analysis statistic in {key}")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        for column in ("planned_windows", "available_windows"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        counts = frame[["planned_windows", "available_windows"]].to_numpy(float)
        if (not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any()
                or (frame.planned_windows < frame.available_windows).any()):
            raise ValueError(f"invalid fixed analysis support denominator in {key}")
        if "parameter_mode" in frame and not frame.parameter_mode.eq("fixed").all():
            raise ValueError("fixed analysis contains a non-fixed parameter mode")
    example_index = _json(paths["examples"]) if "examples" in paths else {"examples": []}
    if not isinstance(example_index.get("examples"), list):
        raise ValueError("fixed examples must be an explicitly declared list")
    for example in example_index["examples"]:
        if not {"dataset", "subject", "pair", "window", "role", "array_path"} <= set(example):
            raise ValueError("fixed examples require explicit identity and selection role")
        if "record" not in example and "record_id" not in example:
            raise ValueError("fixed example requires its original record identity")
        if "record" not in example:
            example["record"] = example["record_id"]
        if example.get("parameter_mode", "fixed") != "fixed":
            raise ValueError("fixed examples contain a non-fixed parameter mode")
        path = _path(run, example["array_path"])
        if "current_dof" in path.relative_to(run).parts:
            raise ValueError("fixed example cannot use current_dof arrays")
    counterfactual_stage = analysis/"counterfactual_summary.json"
    if counterfactual_stage.is_file():
        paths["counterfactual_stage"] = _path(analysis, "counterfactual_summary.json")
    if (analysis/"verification.json").is_file():
        paths["verification"] = _path(analysis, "verification.json")
    return dict(run=run, analysis=analysis, summary=summary, tables=frames, paths=paths,
                example_index=example_index, examples=example_index["examples"])


def _fixed_metric_label(metric):
    return FIXED_METRICS.get(metric, (metric, "见指标定义", ""))


def _fixed_metric_code(metric):
    return "残差/零 MSE 比" if metric == "Hb_residual_to_zero_MSE_ratio" else metric


def _fixed_task_label(task):
    return {"mental_arithmetic": "心算", "motor_imagery": "运动想象",
            "dsr": "DSR", "nback": "n-back", "wg": "WG",
            "visual_cognitive_motivation": "本轮 30 秒窗口"}.get(
                str(task), str(task).replace("emotion_video/video_", "视频 "))


def _fixed_task_order(values):
    import re
    return sorted(set(map(str, values)), key=lambda value: [int(part) if part.isdigit() else part
                  for part in re.split(r"(\d+)", value)])


def fixed_distribution_figure(evidence, metrics, destination):
    """Display retained subject statistics; no cohort metric is recomputed."""
    plt = _plotting()
    frame = evidence["tables"]["subjects"]
    datasets = sorted(frame.dataset.astype(str).unique())
    fig, axes = plt.subplots(math.ceil(len(metrics)/2), 2,
                             figsize=(8, 2.9*math.ceil(len(metrics)/2)), squeeze=False)
    palette = ("#246eab", "#af4571", "#278579", "#d17d25")
    for axis, metric in zip(axes.flat, metrics):
        for color, dataset in zip(palette, datasets):
            values = frame.loc[(frame.dataset.astype(str) == dataset) & (frame.metric == metric), "value"].to_numpy(float)
            x, y = weighted_ecdf(values)
            if len(x):
                axis.step(x, y, where="post", color=color, lw=1.5,
                          label=DATASET_LABELS.get(dataset, dataset)+f" n={len(x)}")
        name, unit, _ = _fixed_metric_label(metric)
        axis.set(title=name, xlabel=unit, ylabel="被试累计比例", ylim=(0, 1.04))
        if metric == "Delta_spatial":
            axis.axvline(0, color="#777", ls="--", lw=.8)
        axis.grid(alpha=.17)
        if axis.lines:
            axis.legend(fontsize=7, loc="lower right", frameon=False)
        else:
            axis.text(.5, .5, "无有效被试支持", ha="center", transform=axis.transAxes)
    for axis in list(axes.flat)[len(metrics):]:
        axis.set_visible(False)
    fig.tight_layout()
    return _save(fig, destination)


def fixed_task_distribution_figure(evidence, dataset, metrics, destination):
    """One point per subject/task, with the retained missing denominator."""
    plt = _plotting()
    frame = evidence["tables"]["subject_tasks"]
    frame = frame[frame.dataset.astype(str) == dataset]
    tasks = _fixed_task_order(frame.task_label)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, max(2.6, 2.6*len(metrics))), squeeze=False)
    for axis, metric in zip(axes.flat, metrics):
        counts = []
        for index, task in enumerate(tasks):
            part = frame[(frame.metric == metric) & (frame.task_label.astype(str) == task)].sort_values("subject")
            values = part.value.to_numpy(float)
            values = values[np.isfinite(values)]
            counts.append(f"{_fixed_task_label(task)}\n{len(values)}/{len(part)}")
            if not len(values):
                continue
            # Deterministic offsets expose repeated values without random jitter.
            axis.scatter(index+np.linspace(-.14, .14, len(values)), values, s=13,
                         alpha=.65, color=COLORS["fixed"], edgecolors="none")
            axis.boxplot(values, positions=[index], widths=.45, showfliers=False,
                         patch_artist=False, medianprops={"color": "#202c39", "lw": 1.4})
        name, unit, _ = _fixed_metric_label(metric)
        axis.set(title=name, ylabel=unit)
        axis.set_xticks(range(len(tasks)), counts, rotation=45 if len(tasks) > 4 else 0,
                        ha="right" if len(tasks) > 4 else "center")
        axis.grid(axis="y", alpha=.17)
    fig.suptitle(DATASET_LABELS.get(dataset, dataset)+"：同一被试在各任务分别显示", fontsize=11)
    fig.tight_layout()
    return _save(fig, destination)


def fixed_subject_task_heatmap(evidence, dataset, metric, destination, *, maximum):
    """Keep every planned subject ID visible instead of only showing ECDFs."""
    plt = _plotting()
    frame = evidence["tables"]["subject_tasks"]
    part = frame[(frame.dataset.astype(str) == dataset) & (frame.metric == metric)]
    subjects = _fixed_task_order(part.subject)
    tasks = _fixed_task_order(part.task_label)
    matrix = np.full((len(subjects), len(tasks)), np.nan)
    for row in part.itertuples():
        matrix[subjects.index(str(row.subject)), tasks.index(str(row.task_label))] = row.value
    fig, axis = plt.subplots(figsize=(8, max(4.5, .22*len(subjects)+1.7)))
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#dddddd")
    im = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap,
                     vmin=0, vmax=max(float(maximum), .001))
    axis.set_xticks(range(len(tasks)), [_fixed_task_label(task) for task in tasks],
                      rotation=45 if len(tasks)>4 else 0, ha="right" if len(tasks)>4 else "center")
    axis.set_yticks(range(len(subjects)), subjects, fontsize=8)
    axis.set(xlabel="原生任务／具体视频", ylabel="被试 ID（每行一人）",
             title=DATASET_LABELS.get(dataset, dataset)+" / "+_fixed_metric_code(metric))
    fig.colorbar(im, ax=axis, label=_fixed_metric_label(metric)[1], pad=.025)
    fig.tight_layout()
    return _save(fig, destination)


def fixed_task_contrast_figure(frame, metric, destination, *, maximum=None):
    """Show every retained task contrast, including untestable support."""
    plt = _plotting()
    part = frame[frame.metric == metric]
    datasets = sorted(part.dataset.astype(str).unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(5, 4*len(datasets)), 4.5), squeeze=False)
    finite = np.asarray(part.mean_difference, float)
    maximum = (float(np.max(np.abs(finite[np.isfinite(finite)]))) if np.isfinite(finite).any() else 1.) if maximum is None else maximum
    maximum = max(maximum, .001)
    for axis, dataset in zip(axes.flat, datasets):
        rows = part[part.dataset.astype(str) == dataset]
        tasks = _fixed_task_order(list(rows.task_a)+list(rows.task_b))
        matrix = np.full((len(tasks), len(tasks)), np.nan)
        np.fill_diagonal(matrix, 0.)
        marked = []
        for row in rows.itertuples():
            a, b = tasks.index(str(row.task_a)), tasks.index(str(row.task_b))
            if np.isfinite(row.mean_difference):
                matrix[a, b], matrix[b, a] = row.mean_difference, -row.mean_difference
            if np.isfinite(row.q_value) and row.q_value < .05:
                marked.extend(((a, b), (b, a)))
        cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#dddddd")
        im = axis.imshow(matrix, cmap=cmap, vmin=-maximum, vmax=maximum, aspect="equal")
        if marked:
            axis.scatter([b for a, b in marked], [a for a, b in marked], marker="o", s=8, c="#202020")
        labels = [_fixed_task_label(task) for task in tasks]
        axis.set_xticks(range(len(tasks)), labels, rotation=60, ha="right", fontsize=7)
        axis.set_yticks(range(len(tasks)), labels, fontsize=7)
        axis.set(title=DATASET_LABELS.get(dataset, dataset), xlabel="任务 B", ylabel="任务 A")
        fig.colorbar(im, ax=axis, shrink=.7, label="同被试差 B − A / "+_fixed_metric_label(metric)[1])
    fig.suptitle(_fixed_metric_label(metric)[0]+"：同被试任务差", fontsize=11)
    fig.tight_layout()
    return _save(fig, destination)


def fixed_stability_figure(frame, metrics, destination, *, title):
    """A/B scatter from owning paired records, with shared axes per metric."""
    plt = _plotting()
    fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 3.6), squeeze=False)
    datasets = sorted(frame.dataset.astype(str).unique())
    palette = ("#246eab", "#af4571", "#278579", "#d17d25")
    for axis, metric in zip(axes.flat, metrics):
        selected = frame[frame.metric == metric]
        values = []
        for color, dataset in zip(palette, datasets):
            part = selected[selected.dataset.astype(str) == dataset]
            a, b = np.asarray(part.value_a, float), np.asarray(part.value_b, float)
            finite = np.isfinite(a) & np.isfinite(b)
            values.extend(a[finite]); values.extend(b[finite])
            axis.scatter(a[finite], b[finite], s=16, alpha=.65, color=color,
                         label=DATASET_LABELS.get(dataset, dataset)+f" 配对={int(finite.sum())}")
        low, high = (min(values), max(values)) if values else (0., 1.)
        margin = max((high-low)*.06, .001)
        axis.plot([low-margin, high+margin], [low-margin, high+margin], color="#777", ls="--", lw=.8)
        axis.set(xlim=(low-margin, high+margin), ylim=(low-margin, high+margin),
                 title=_fixed_metric_label(metric)[0], xlabel="A / "+_fixed_metric_label(metric)[1],
                 ylabel="B / "+_fixed_metric_label(metric)[1])
        axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.17)
        axis.legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _save(fig, destination)


def _fixed_method_blocks(evidence):
    blocks = [
        _paragraph("本分析固定 Tak 生理参数，本报告不纳入自由参数条件。每个窗口的 Hb-only 条件可见该窗口的处理后 HbO/HbR，并拟合驱动和初态；联合条件可见 EEG 代理及两个 Hb 坐标。两者都是离线条件拟合，不能作为隐藏 Hb 预测证据。EEG-only 可见 EEG，其 Hb 输出用于独立预测比较。"),
        {"type": "equation", "latex": r"\begin{aligned}r(t)&\longrightarrow [s(t),f(t),v(t),q(t)]\longrightarrow [\widehat H_O(t),\widehat H_R(t)]\\r(t)&\xrightarrow{B_E}\widehat e(t)\end{aligned}",
         "caption": "同一标量条件驱动通过固定 Tak 动力学与观测算子产生 Hb 和 EEG 代理读出。"},
        _paragraph("s 是血流调节信号，f 是归一化流入，v 是归一化静脉血容量，q 是归一化脱氧血红蛋白含量；它们是模型状态，不是直接测量。当前 Tak 实现只有这四个独立血管状态；保留输出 p 恒等于 v。生理参数、EEG 读出及观测变换冻结，驱动和初态按当前可见窗口估计；同一条件的预测接受原缓存与窗口参考算子后再评分。"),
        _paragraph("本轮 EEG 原生分支的路径为六近邻通道×五频带（1–4、4–8、8–13、13–30、30–45 Hz）→100 ms 功率→对数→前 5 秒参考→训练分区冻结第一主成分（PC1）与尺度，输出 10 Hz 代理。同一被试的会话在同一外折；PCA、尺度及空间候选选择仅用训练／内部选择被试，不逐窗翻转符号、重定标或搜索时移。固定 Tak 原生 EEG 分支未应用项目通用缓存中的 EOG 回归步骤；供应商更早处理按原始文档保留。EEG 代理不是神经活动的已定标真值。Hb 使用带通缓存与窗口参考坐标，不是已知绝对基线的生理状态。"),
        {"type": "equation", "latex": r"D_1=\sqrt{\frac{1}{2|I_H|}\sum_{t\in I_H}\sum_{c\in\{O,R\}}\left(\frac{H_c(t)-\widehat H_{H,c}(t)}{\sigma_c}\right)^2}",
         "caption": "Hb-only 相容性误差：两个 Hb 色团在共同评分时间上等权。"},
        _paragraph("H 为处理后实测 Hb，H 下标的预测表示只以 Hb 条件拟合；O/R 为 HbO/HbR。I_H 为二者共同有效评分支持；30 秒窗口的前 5 秒仅作参考，末箱排除，名义评分为 5.0–29.8 秒的 249 个 0.1 秒箱，即 24.9 秒。实际缺失按共同 mask 保留。sigma 是训练分区冻结 SD，不是该窗口 SD、局部 RMS 或已知传感器噪声。D1 单位是训练 SD。"),
        {"type": "equation", "latex": r"\begin{aligned}D_{EH}&=\sqrt{\operatorname{mean}_{I_E}\left[(e-B_E\widehat r_H)/\sigma_E\right]^2}\\D_4&=\sqrt{(D_{J,E}^2+D_{J,O}^2+D_{J,R}^2)/3}\end{aligned}",
         "caption": "EEG 读出误差和联合三坐标误差，均使用对应训练冻结 SD。"},
        _paragraph("D_EH 比较 Hb-only 解的 EEG 读出与实测 EEG 代理，是观测误差，不是两个驱动之差。D4 中 J 表示联合拟合；三个分量等权，合计降低仍可能伴随某个分量恶化。D_drive_H 才比较 EEG-only 与 Hb-only 各自独立正则化拟合所得、通过原数值接纳的条件驱动：固定同一生理参数、同一静息血管初态与保留 DC 规范重放后，以 Hb 输出差的训练 SD RMS 衡量。不承诺非凸拟合的全局最优或唯一性；这些驱动仍是模型条件估计，不是分离出的神经真值。"),
        _paragraph("D_obs 只保留局部 Hb 可观测驱动方向，保留秩依赖预设合成噪声阈值；零秩不可用，低值不证明真实生理驱动相同。D_bal 是完整原生记录上 10/20/40 秒弱核两侧积分的相对平衡偏离，具有独立 record-pair 支持；重复附在窗口的数值不能重复计样本。D_bal 约 0.707 可能来自近正交或一侧能量较弱，不是“70% 未解释方差”。"),
        _paragraph("补充空间指标 Delta_spatial 是当前六近邻减训练／内部选择后冻结的同容量替代六通道的 EEG-only Hb 均方误差，单位是 Hb 训练 SD 平方；正值有利于替代方案，负值有利于当前方案。替代方案允许保留原六近邻，此时零是同一预测的自比较，不说明空间映射没有误差。典型窗口分别列出联合 EEG/HbO/HbR 三分量，避免 D4 总分遮蔽观测间的误差折中。"),
        {"type": "equation", "latex": r"R_{H/0}=\frac{\operatorname{Agg}_{\mathrm{matched}}(D_1^2)}{\operatorname{Agg}_{\mathrm{matched}}(L_{H,0})}",
         "caption": "Hb-only 残差与零变化预测的同支持均方误差比。"},
        _paragraph("分母 L_H,0 是相同处理坐标、训练 SD 和匹配窗口上的零变化 Hb 预测均方误差；先按 owning 分层规则分别聚合分子和分母，再取比。该比值可大于 1，不能称为未解释方差、R² 或生理能量份额。D1 残差仅是模型与观测的差，尚未被分离为真实生理源。"),
        _paragraph("窗口级 RMS 指标先平方为均方误差，在原生记录内对有效窗口与所选通道对等权，随后在被试内对原生记录等权，最后开根。任务单元在同一被试×任务内执行相同顺序；比值指标沿用分子、分母各自聚合后的比。一个被试可贡献多个任务点，窗口、通道、核中心都不是独立被试。非有限指标不填零，完整计划和有效支持分开保留。具体原生记录及任务身份由分析表持有。"),
    ]
    parameters = evidence["summary"].get("fixed_parameters", {})
    if parameters:
        units = dict(tau="秒", viscoelastic_s="秒", kappa="每秒", gamma="每秒平方", alpha="无量纲", E0="无量纲",
                     P0="Tak 观测参考尺度", Q0="Tak 观测参考尺度", neurovascular_gain="工程驱动增益", eeg_loading="工程读出尺度")
        blocks.insert(1, _table(["固定参数", "取值", "单位／坐标"],
            [[key, _num(value, 5), units.get(key, "Tak 参考坐标")]
             for key, value in parameters.items() if isinstance(value, (int, float))],
            "直接读取 owning analysis 的固定参数，不能据此断言个体生理参数已恢复。"))
        blocks[2:2] = [dict(type="equation", latex=r"\dot f=s,\qquad \dot s=g r-\kappa s-\gamma(f-1)",
            caption="当前 Tak 血流响应的驱动关系；f 是相对静息、无量纲的流量。", source=str(ROOT/"docs/ARCHITECTURE.md")),
            _paragraph("因此 s 与 kappa 的单位是每秒，gamma 是每秒平方；驱动 r 保留工程坐标，g 与 r 的乘积具有每秒平方量纲，不为两者分别添加未经标定的神经单位。P0/Q0 是模型观测参考尺度，不冒认混合数据集的绝对 Hb 浓度。")]
    return blocks


def build_fixed_content(evidence, directory):
    """Author the fixed-parameter report from completed analysis artifacts."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    summary, tables, paths = evidence["summary"], evidence["tables"], evidence["paths"]
    subjects, cells = tables["subjects"], tables["subject_tasks"]
    descriptions = {}
    descriptive_frame = tables.get("subject_descriptives")
    if descriptive_frame is not None:
        if descriptive_frame.duplicated(["dataset", "metric"]).any():
            raise ValueError("subject descriptives must contain one owning row per dataset and metric")
        descriptions = {(str(row["dataset"]), str(row["metric"])): row
                        for row in descriptive_frame.to_dict("records")}
    datasets = sorted(subjects.dataset.astype(str).unique())
    metrics = [metric for metric in FIXED_MAIN_METRICS if subjects.metric.eq(metric).any()]
    content = dict(title="固定 Tak 参数下的生理失配：分布、稳定性与观测锚点",
        subtitle="四套公开数据的只读探索；条件相容性、任务配对和来源归属分别判断",
        date=str(summary.get("completed_at", summary.get("created_utc", ""))),
        provenance=["Owning 分析证据："+str(evidence["analysis"]),
                    "本报告汇总已完成的固定参数证据，不改变拟合、原始数据或实验状态。"],
        summary=[str(summary.get("scope_description", "范围以分析 summary 和冻结面板计划为准。")),
            "问题是固定动力学在当前处理后的 EEG/Hb 坐标中留下多少失配、这些失配能否在被试内复现，以及是否伴随任务或可观测生理锚点变化。不同指标回答不同问题，不合成为一个生理总分。",
            "Hb-only 拟合使用当前窗口 Hb，因此低残差说明条件相容性；它不证明独立预测、参数准确或来源可识别。EOG 关联只提供眼源污染候选，不能解释为眼电造成的失配比例。"],
        sections=[], sources=[str(evidence["analysis"]/"summary.json")]+[str(path) for path in paths.values()])
    for text in summary.get("report_findings", []):
        content["summary"].append(str(text).replace("−", "-"))
    dataset_summaries = []
    for dataset in datasets:
        pieces = []
        for metric in ("D1", "Hb_residual_to_zero_MSE_ratio", "D_EH", "D_drive_H"):
            part = subjects[(subjects.dataset.astype(str) == dataset) & (subjects.metric == metric)]
            values = part.value.to_numpy(float); values = values[np.isfinite(values)]
            if len(values):
                median = descriptions.get((dataset, metric), {}).get("median", np.median(values))
                pieces.append(_fixed_metric_code(metric)+"="+_num(median, 4))
        if pieces:
            dataset_summaries.append(DATASET_LABELS.get(dataset, dataset)+"："+"；".join(pieces))
    if dataset_summaries:
        content["summary"].append("四库被试中位数（单位及匹配支持不同）："+"。".join(dataset_summaries)+"。")
    ocular_effects = tables.get("ocular_associations")
    if ocular_effects is not None:
        c = ocular_effects[(ocular_effects.analysis == "counterfactual") & (ocular_effects.contrast == "clean_minus_amplitude")]
        if len(c) and (c.q_value >= .05).all():
            content["summary"].append("EOG 参考回归后，EEG 目标对固定 Hb 读出的误差降低；但相对匹配均值/尺度的原代理，未检出额外改善。该负对照限制了来源归因：不能把 raw 到 clean 的收益解释成眼电造成的失配份额，也不能将未显著写为严格等效。")
    findings = []
    repeated = tables.get("repeat_summary")
    if repeated is not None:
        single = repeated[(repeated.dataset == "eeg_fnirs_single_trial") & (repeated.subjects >= 6)]
        pieces = []
        for task in _fixed_task_order(single.task_label):
            group = single[single.task_label == task]
            d1, ratio = group[group.metric == "D1"], group[group.metric == "Hb_residual_to_zero_MSE_ratio"]
            if len(d1) == len(ratio) == 1:
                a, b = d1.iloc[0], ratio.iloc[0]
                pieces.append(_fixed_task_label(task)+f"：D1 rho={_num(a.spearman_rho)} [{_num(a.rho_ci_low)},{_num(a.rho_ci_high)}]；残差/零 MSE 比 rho={_num(b.spearman_rho)} [{_num(b.rho_ci_low)},{_num(b.rho_ci_high)}]，n={int(a.subjects)}")
        if pieces:
            findings.append("单试次数据的同任务跨原生记录排序可复现："+"；".join(pieces)+"。这是短期记录一致性，不是长期神经血管特质证明。")
    compared = tables.get("task_comparisons")
    if compared is not None:
        chosen = compared[(compared.dataset == "simultaneous_eeg_nirs") & (compared.task_a == "nback") & (compared.task_b == "wg")]
        pieces = []
        for metric in ("D1", "D_EH", "D_drive_H", "Hb_residual_to_zero_MSE_ratio"):
            row = chosen[chosen.metric == metric]
            if len(row) == 1:
                row = row.iloc[0]; pieces.append(_fixed_metric_code(metric)+f" 差={_num(row.mean_difference, 4)}，q={_num(row.q_value, 4)}")
        if pieces:
            findings.append("WG 减 n-back 的同被试对比："+"；".join(pieces)+"。绝对训练尺度残差与相对零基线比例分别判断；固定任务顺序与疲劳混杂，不能归因于任务类型。")
    if ocular_effects is not None:
        chosen = ocular_effects[(ocular_effects.analysis == "counterfactual") & (ocular_effects.contrast == "clean_minus_amplitude")]
        if len(chosen):
            findings.append("EOG 的关键负对照："+"；".join(DATASET_LABELS.get(str(row.dataset), str(row.dataset))
                +f" clean-amp 均差={_num(row.mean, 4)}，95% 区间 [{_num(row.ci_low, 4)},{_num(row.ci_high, 4)}]，q={_num(row.q_value, 4)}"
                for row in chosen.itertuples())+"。均值／尺度匹配已在数值上复现主要平均下降；未检出额外改善不是等效检验，也不能量化眼电造成的失配。")
    shapes = tables.get("shape_subjects")
    if shapes is not None:
        rho = shapes[shapes.feature == "Hb_residual_OR_correlation"].groupby("dataset").value.median()
        common = shapes[shapes.feature == "Hb_common_direction_fraction"].groupby("dataset").value.median()
        if len(rho) and len(common):
            findings.append(f"Hb 残差同向：四库相关中位数范围 {_num(rho.min())}–{_num(rho.max())}，共同方向平方比 {_num(common.min())}–{_num(common.max())}。"
                +"解析投影反例说明，即使输入只有独立噪声，受限拟合也能留下完全共同的残差；同向结构尚不能指认为系统／浅表血流源。")
    content["summary"][3:3] = [f"发现{number}："+text.replace("−", "-") for number, text in enumerate(findings, 1)]
    coverage = []
    for dataset in datasets:
        for metric in metrics:
            part = subjects[(subjects.dataset.astype(str) == dataset) & (subjects.metric == metric)]
            if part.empty:
                continue
            valid = np.isfinite(part.value)
            weak = metric.startswith("D_bal_")
            coverage.append([DATASET_LABELS.get(dataset, dataset), _fixed_metric_code(metric), len(part), int(valid.sum()),
                             "完整记录附件" if weak else int(part.planned_windows.sum()),
                             "不重复计窗" if weak else int(part.available_windows.sum())])
    coverage_table = _table(["数据集", "指标", "计划被试", "有效被试", "计划窗口对", "有效窗口对"], coverage,
        "按指标独立列出支持；同一窗口会出现在多个指标行中，不能跨行相加。分布只画有效被试，失败与无支持仍在计划分母。窗口对不是独立被试。")
    coverage_table["source"] = str(paths["subjects"])
    status_blocks = []
    if "view_coverage" in paths:
        audit = _json(paths["view_coverage"])
        if audit.get("schema") != "tak_mismatch_fixed_physiology_view_coverage_v1":
            raise ValueError("unsupported fixed view coverage schema")
        status_blocks = terminal_status_blocks(dict(terminal_audit=dict(status="completed",
            by_dataset={dataset: dict(fixed=dict(views=views)) for dataset, views in audit["by_dataset"].items()}),
            terminal_audit_path=paths["view_coverage"]))
        status_blocks[1] = _paragraph("本表只读取固定参数各 view 的原始终态；条件拟合通过、数值不收敛、无支持、执行失败与缺失分别保留。一个条件通过不替代其他条件通过，不用总计划减成功数推断数值失败率。")
    verification_blocks = []
    if "verification" in paths:
        verification = _json(paths["verification"])
        if verification.get("status") != "passed":
            raise ValueError("retained independent numeric verification has not passed")
        rows = [["完整固定标量行", verification["fixed_rows"]], ["预定形态面板行", verification["panel_rows"]],
                ["有效 Hb 残差恒等式行", verification["validated_Hb_shape_rows"]],
                ["偏置平方加去均值平方：最大绝对差", _num(verification["bias_centered_energy_max_error"], 4)],
                ["共同／相反正交方向平方和：最大绝对差", _num(verification["orthogonal_direction_energy_max_error"], 4)],
                ["固定 Hb 读出下 C 实际评分窗口对", verification["counterfactual_scored_rows"]],
                ["raw 代理复现最大绝对差", _num(verification["raw_proxy_max_absolute_error"], 4)],
                ["C 每窗共同评分箱数", verification["counterfactual_common_score_bins"]]]
        table = _table(["独立数值核查", "保留结果"], rows,
            "核查仅验证支持、原代理复现与代数平方恒等式；通过不证明生理来源可识别。")
        table["source"] = str(paths["verification"]); verification_blocks.append(table)
    content["sections"].append(dict(title="阅读路径", blocks=[_paragraph(
        "PDF 可按书签跳转，HTML 可查找同名标题。先读摘要，再看“残差形态与时间结构”和“共同残差不唯一指向共同生理源”；眼电结论看“EOG A/B/C”，其中 C 的幅度对照是核心负结果。逐被试 ID 看各库“任务分布”热图，任务效应看“完整配对矩阵与数字解释”，重复性看“同任务原生记录”和“同记录早晚半段”。最后看典型波形；完整重复性诊断长表在末尾附录，所有完整数值继续由 owning CSV 持有。")]))
    content["sections"].append(dict(title="范围、分母与固定条件", page_break=True, blocks=[*status_blocks, coverage_table, *verification_blocks,
        _paragraph("主标量使用完整固定参数运行的保留结果；时序形态、典型波形与 EOG 锚点使用预先按空间几何选定的三通道面板，两者分母不同。每节分别注明来源和有效支持，不把全运行终态数解释为面板数，也不把终态解释为成功拟合。面板不按误差或收敛筛选。"),
        *_fixed_method_blocks(evidence)]))
    _append_fixed_shapes(content, evidence, directory)
    _append_fixed_ocular(content, evidence, directory)
    for offset in range(0, len(metrics), 4):
        names = metrics[offset:offset+4]
        path = fixed_distribution_figure(evidence, names, directory/f"fixed_subjects_{offset//4+1}.png")
        blocks = [_figure(path, "每条阶梯线由同一数据集的被试汇总构成，每位有效被试等权；纵轴为累计被试比例，图例 n 是有效人数。曲线越靠左表示该指标较低；Delta_spatial 的灰虚线是零，正值才有利于替代邻域。每个指标横轴独立，跨库在同一指标内共用横轴；训练 SD 坐标不等于跨库相同绝对浓度。", paths["subjects"])]
        rows = []
        for dataset in datasets:
            for metric in names:
                values = subjects.loc[(subjects.dataset.astype(str) == dataset) & (subjects.metric == metric), "value"].to_numpy(float)
                values = values[np.isfinite(values)]
                if len(values):
                    desc = descriptions.get((dataset, metric), {})
                    if desc and int(desc["available_subjects"]) != len(values):
                        raise ValueError("subject descriptive count disagrees with retained subject rows")
                    rows.append([DATASET_LABELS.get(dataset, dataset), _fixed_metric_code(metric), len(values),
                                 _num(desc.get("median", np.median(values))), _num(desc.get("q25", np.quantile(values, .25))),
                                 _num(desc.get("q75", np.quantile(values, .75))), _num(desc.get("p90", np.quantile(values, .9)))])
        for dataset in datasets:
            pieces = []
            for metric in names:
                part = subjects[(subjects.dataset.astype(str) == dataset) & (subjects.metric == metric)]
                values = part.value.to_numpy(float); values = values[np.isfinite(values)]
                if not len(values):
                    continue
                desc = descriptions.get((dataset, metric), {})
                median = float(desc.get("median", np.median(values)))
                pieces.append(_fixed_metric_code(metric)+f" 中位数 {_num(median)}（有效被试 {len(values)}/{len(part)}）")
                if metric == "Hb_residual_to_zero_MSE_ratio":
                    relation = "低于" if median < 1 else "高于" if median > 1 else "等于"
                    pieces[-1] += f"，其有效子集中位比值{relation}零变化基线的 1"
            if pieces:
                blocks.append(_paragraph(DATASET_LABELS.get(dataset, dataset)+"："+"；".join(pieces)+"。这些是各自有效支持的分布摘要，不能将非配对子集的数值差当成效果。"))
        table = _table(["数据集", "指标", "有效被试", "中位数", "Q25", "Q75", "P90"], rows,
            "分位数优先直接引用 owning subject_descriptives；未提供时仅对已保留被试行作展示性汇总。不重做拟合或显著性检验；不同有效子集的中位数之差不是配对效果。")
        table["source"] = str(paths.get("subject_descriptives", paths["subjects"])); blocks.append(table)
        blocks.extend(_paragraph(metric+"："+_fixed_metric_label(metric)[2]) for metric in names)
        content["sections"].append(dict(title="被试等权分布"+f"（{offset//4+1}）", blocks=blocks, page_break=True))
    task_metrics = [metric for metric in ("D1", "D_EH", "D_drive_H") if cells.metric.eq(metric).any()]
    for dataset in datasets:
        blocks = []
        for metric in ("D1", "Hb_residual_to_zero_MSE_ratio", "D_drive_H"):
            if not cells.metric.eq(metric).any():
                continue
            values = cells.loc[cells.metric == metric, "value"].to_numpy(float)
            maximum = float(np.max(values[np.isfinite(values)])) if np.isfinite(values).any() else 1.
            heatmap = fixed_subject_task_heatmap(evidence, dataset, metric,
                directory/f"{dataset}_{metric}_subject_task.png", maximum=maximum)
            blocks.append(_figure(heatmap, "每行是一名被试（完整 ID 见纵轴），每列为原生任务／具体视频，每格是该被试×任务的保留指标。相同指标跨四库使用同一从 0 开始的完整色标，不裁去极值；灰色为不可用，不是零。行和列不会增加独立被试数；具体数值与支持分母在 owning subject_tasks 表。", paths["subject_tasks"]))
        for offset in range(0, len(task_metrics), 2):
            path = fixed_task_distribution_figure(evidence, dataset, task_metrics[offset:offset+2],
                                                 directory/f"{dataset}_fixed_tasks_{offset//2+1}.png")
            blocks.append(_figure(path, "每个点是一名被试在一个任务／具体视频上的保留汇总；同被试多点有依赖。标签给出有效被试／计划被试；箱体为 Q25–Q75，中线为中位数，须为 1.5 IQR 内极值，全部有限点均显示，横向偏移只为避免遮挡。各指标独立纵轴，任务间在同一面板内共用尺度。", paths["subject_tasks"]))
        task_descriptives = tables.get("task_descriptives")
        if task_descriptives is not None:
            dd = task_descriptives[(task_descriptives.dataset.astype(str) == dataset) & (task_descriptives.metric == "D1")]
            rows = [[_fixed_task_label(row.task_label), f"{int(row.available_subjects)}/{int(row.planned_subjects)}",
                     _num(row.median), f"{_num(row.q25)}–{_num(row.q75)}", _num(row.p90),
                     f"{int(row.available_windows)}/{int(row.planned_windows)}"] for row in dd.itertuples()]
            if rows:
                table = _table(["任务／具体视频", "有效／计划被试", "D1 中位数", "Q25–Q75", "P90", "有效／计划窗口对"], rows,
                    "D1 单位为冻结 Hb 训练 SD，直接读取 owning task_descriptives。每行被试内先对原生记录等权；跨行有重复被试，任务效果使用下一节的配对统计判断。")
                table["source"] = str(paths["task_descriptives"]); blocks.append(table)
        if dataset == "simultaneous_eeg_nirs":
            blocks.append(_paragraph("本数据同天固定按 n-back→WG→DSR 顺序采集，任务差与顺序、疲劳和缓慢状态变化混杂；不能仅凭任务配对差归因为任务类型。"))
        elif dataset == "refed":
            blocks.append(_paragraph("每一列是具体发布视频。不同视频、刺激内容和播放顺序共变，视频差不能直接升格为情绪类别效应；跨视频差也不称为同条件重测。"))
        elif dataset == "visual_cognitive_motivation":
            blocks.append(_paragraph("本轮冻结的 30 秒窗口表未提供可审计的四条件标签，因此仅呈现本轮窗口标签层级分布，不补写四条件任务效应。两个同步 Probe 不构成独立重复。"))
        content["sections"].append(dict(title=DATASET_LABELS.get(dataset, dataset)+"：任务分布", blocks=blocks, page_break=True))
    _append_fixed_contrasts(content, evidence, directory)
    _append_fixed_stability(content, evidence, directory)
    _append_fixed_examples(content, evidence, directory)
    content["sections"].append(dict(title="这些结果能支持的解释与用途", blocks=[
        _paragraph("固定模型残差可以作为处理后观测坐标中的失配表型，用于检查拟合困难时段、比较被试内任务差和筛选需要进一步区分的观测链因素。稳定的误差排序仍可能来自稳定的通道、头皮和系统性血流差异；重复性本身不证明神经生理特异性。"),
        _paragraph("分别判断数值状态、波形恢复、驱动一致性和来源归属。较低 D1 不保证独立 EEG→Hb 预测更好；较高 EOG 相关不证明眼电导致 Hb 残差；HbO 主导或慢变残差与浅表／全身血流解释相容，但不具有来源特异性。未直接校正或干预的对照不能量化因果占比。"),
        _paragraph("下一步优先围绕已有失配表型设计可区分的对照：固定同一片段和评价支持，比较明确的观测处理或独立生理锚点；以被试为独立单元保留失败。本报告未新增 SSM 拟合或训练，仅复用保留拟合并进行已授权的离线观测诊断；后续对照是研究建议。"),
    ]))
    content["sections"].extend(content.pop("_appendix_sections", []))
    return content


def _append_fixed_contrasts(content, evidence, directory):
    frame = evidence["tables"].get("task_comparisons")
    if frame is None or frame.empty:
        content["sections"].append(dict(title="同被试任务比较", blocks=[
            _paragraph("未保留可用任务配对统计，分布图不能替代同被试的任务检验。")]))
        return
    needed = {"dataset", "metric", "task_a", "task_b", "paired_subjects", "mean_difference",
              "ci_low", "ci_high", "p_value", "q_value", "status"}
    if not needed <= set(frame):
        raise ValueError("task contrast table is missing its paired inference contract")
    source = evidence["paths"]["task_comparisons"]
    blocks = [
        _paragraph("每对任务只纳入两个任务均有有效统计的同一被试。差值定义为任务 B 减任务 A；正值表示 B 的该失配指标较大，负值表示较小。95% 百分位区间由整个被试为单位的配对 bootstrap 获得；p 值来自被试差值的双侧符号翻转，BH q 值沿用 owning 表中完整 primary family，不把矩阵的对称显示重复计为两次检验。D1、D_EH、D_drive_H 是预先声明的 primary family；少于 6 对不作检验。"),
        _paragraph("矩阵中每个非对角格的色值=列任务减行任务。灰格表示没有有效效应数值；有颜色但无黑点仍是有效差值，不等于零效应。黑点仅标记 owning q<0.05；对角线的零是自比较参考，不是检验结果。所有任务对均保留，具体可用被试数、区间和未检验状态见相邻数字表及 owning CSV。"),
    ]
    if {"fdr_family", "fdr_family_size"} <= set(frame):
        families = "; ".join(str(family)+"："+str(int(group.fdr_family_size.max()))+" 项可检验比较"
                             for family, group in frame.groupby("fdr_family", sort=True))
        blocks.append(_paragraph("本次完整多重比较分母为 "+families+"；图中仅展示 primary 指标，不缩小 owning 检验家族。"))
    displayed = []
    for metric in ("D1", "D_EH", "D_drive_H"):
        part = frame[frame.metric == metric]
        if part.empty:
            continue
        finite = np.asarray(part.mean_difference, float)
        maximum = max(float(np.max(np.abs(finite[np.isfinite(finite)]))) if np.isfinite(finite).any() else 1., .001)
        for dataset in sorted(part.dataset.astype(str).unique()):
            rows = part[part.dataset.astype(str) == dataset]
            path = fixed_task_contrast_figure(rows, metric,
                directory/f"{dataset}_{metric}_task_contrasts.png", maximum=maximum)
            blocks.append(_figure(path, "每格是已保存的同被试平均差 B−A；同一指标跨数据集使用相同对称色标，单位为对应训练 SD。对称两格是同一对比的正负显示，不是独立结果；灰色和黑点定义见正文。", source))
            if dataset == "refed":
                good = rows[np.isfinite(np.asarray(rows.mean_difference, float))]
                if not good.empty:
                    displayed.append((good.loc[good.mean_difference.abs().idxmax()], "描述性绝对差极值"))
            else:
                displayed.extend((row, "全部可读对比") for _, row in rows.iterrows())
    rows = []
    for row, rule in displayed:
        rows.append([DATASET_LABELS.get(str(row.dataset), str(row.dataset)), row.metric,
            _fixed_task_label(row.task_b)+" − "+_fixed_task_label(row.task_a), int(row.paired_subjects),
            _num(row.mean_difference, 4), f"[{_num(row.ci_low, 4)}, {_num(row.ci_high, 4)}]",
            _num(row.p_value, 4), _num(row.q_value, 4), rule])
    table = _table(["数据集", "指标", "B − A", "配对被试", "均值差", "95% 区间", "p", "BH q", "显示规则"], rows,
        "Single-Trial 与 Simultaneous 显示全部对比；REFED 每指标仅在数字表显示绝对平均差最大的一对，完整矩阵与 CSV 仍保留全部视频对。描述性极值是事后展示，可能夸大可重复效应，不是预设重点对比；区间跨零应保留统计不确定性。",
        [1.1, .9, 1.4, .6, .7, 1.2, .65, .65, 1.1])
    table["source"] = str(source); blocks.append(table)
    ratio = frame[(frame.metric == "Hb_residual_to_zero_MSE_ratio") & (frame.dataset != "refed")]
    if not ratio.empty:
        rows = [[DATASET_LABELS.get(str(row.dataset), str(row.dataset)),
                 _fixed_task_label(row.task_b)+" - "+_fixed_task_label(row.task_a), int(row.paired_subjects),
                 _num(row.mean_difference, 4), f"[{_num(row.ci_low, 4)}, {_num(row.ci_high, 4)}]", _num(row.q_value, 4)]
                for row in ratio.itertuples()]
        block = _table(["数据集", "任务 B - A", "配对被试", "残差/零 MSE 比均值差", "95% 区间", "诊断 family BH q"], rows,
            "相对零预测的同支持误差比是诊断终点，沿用自己的完整 diagnostic family；不能把绝对训练尺度残差增加直接解释为相对当前观测拟合更差。")
        block["source"] = str(source); blocks.append(block)
    sim = frame[(frame.dataset == "simultaneous_eeg_nirs") & (frame.task_a == "nback") & (frame.task_b == "wg")]
    if not sim.empty:
        pieces = []
        for metric in ("D1", "D_EH", "D_drive_H", "Hb_residual_to_zero_MSE_ratio"):
            row = sim[sim.metric == metric]
            if len(row) == 1:
                row = row.iloc[0]
                pieces.append(_fixed_metric_code(metric)+f" 差 {_num(row.mean_difference, 5)}，95% 区间 [{_num(row.ci_low, 5)}, {_num(row.ci_high, 5)}]，q={_num(row.q_value, 5)}，n={int(row.paired_subjects)}")
        blocks.append(_paragraph("同时 EEG-NIRS 的 WG 减 n-back："+"；".join(pieces)+"。D1 的训练尺度残差与相对零预测的比值须分别判断；该任务差同时受到固定任务顺序、疲劳和状态漂移混杂，不能作任务类型的随机化因果解释。"))
    single = frame[(frame.dataset == "eeg_fnirs_single_trial") & frame.metric.isin(("D1", "D_EH", "D_drive_H"))]
    if len(single) == 3 and (single.q_value >= .05).all():
        blocks.append(_paragraph("单试次数据的心算与运动想象对照在三个 primary 终点均未通过 BH q<0.05 阈值；这是未检出明确任务差，不是等效性证明。完整差值和区间见上表。"))
    content["sections"].append(dict(title="同被试任务差：完整配对矩阵与数字解释", blocks=blocks, page_break=True))


def _append_fixed_stability(content, evidence, directory):
    for key, title, explanation in (
        ("repeat", "同任务原生记录的一致性",
         "每点是一名被试在一个任务的成对值。A/B 是按原生 metadata 顺序声明的最早／最晚重复支持，只在同一物理通道对均有效时配对；失败端点不以中间成功记录替换。Single-Trial 的短间隔记录重复不能升级为长期特质；Visual Part1/2 属同次访问跨阶段一致性，两个同步 Probe 不作为重复；REFED 跨视频不能称为同条件重测。"),
        ("within_record", "同记录早晚半段的一致性",
         "A 是同一原生记录的较早一半窗口，B 是较晚一半；按全部计划窗口时间固定边界，奇数个窗口时丢弃中间一窗，不按成功结果移动边界。每半至少有两个不同有效窗口，且只匹配两半均有效的同一物理通道对；先在记录内配对，再对原生记录等权汇总为被试×任务点。两半共享预处理、训练尺度、记录状态及时间依赖，不能称为 test–retest 或独立重复。")):
        pairs = evidence["tables"].get(key+"_pairs")
        summary = evidence["tables"].get(key+"_summary")
        blocks = [_paragraph(explanation)]
        if pairs is None or summary is None or summary.empty:
            blocks.append(_paragraph("没有保留可估计的配对支持，不从相邻窗口或同步通道补造重复。"))
            content["sections"].append(dict(title=title, blocks=blocks)); continue
        required = {"dataset", "subject", "task_label", "metric", "value_a", "value_b"}
        if not required <= set(pairs):
            raise ValueError("fixed stability pairs lack explicit A/B identity")
        scopes = evidence["summary"].get("repeatability_scope", {})
        for kind in summary.repeat_kind.astype(str).unique():
            if kind in scopes:
                blocks.append(_paragraph("本表重复范围："+str(scopes[kind])+"。"))
        if key == "within_record":
            refed = summary[(summary.dataset == "refed") & (summary.subjects >= 6)]
            first = refed[refed.metric == "D1"]
            if not first.empty:
                text = (f"REFED 只有 {first.task_label.nunique()} 个视频满足早晚各半至少两个有效窗口；"
                        +f"每视频有效被试为 {int(first.subjects.min())}–{int(first.subjects.max())}，同一批被试跨视频复用。"
                        +f"D1 排名相关范围 {_num(first.spearman_rho.min())}–{_num(first.spearman_rho.max())}，"
                        +f"同时后半减前半的 D1 中位差为 [{_num(first.median_signed_difference.min(), 4)}, {_num(first.median_signed_difference.max(), 4)}] 个训练 SD。")
                for metric in ("D_drive_H", "Hb_residual_to_zero_MSE_ratio"):
                    chosen = refed[refed.metric == metric]
                    if chosen.empty:
                        continue
                    text += (_fixed_metric_code(metric)+f" 的后半减前半中位差范围 [{_num(chosen.median_signed_difference.min(), 4)}, {_num(chosen.median_signed_difference.max(), 4)}]；")
                    if metric == "Hb_residual_to_zero_MSE_ratio":
                        crossing = int(((chosen.rho_ci_low <= 0) & (chosen.rho_ci_high >= 0)).sum())
                        text += (f"其排名相关 {_num(chosen.spearman_rho.min())}–{_num(chosen.spearman_rho.max())}，{crossing}/{len(chosen)} 个相关区间跨零。")
                blocks.append(_paragraph(text+"排名一致并不等于绝对值稳定；D1 与相对零基线比值的方向也不可混同。"))
        if key == "repeat":
            visual = summary[(summary.dataset == "visual_cognitive_motivation") & (summary.metric == "D1")]
            main = visual[visual.repeat_kind == "same_visit_cross_part"]
            all_pairs = pairs[(pairs.dataset == "visual_cognitive_motivation") & (pairs.metric == "D1")]
            if len(main) == 1:
                row = main.iloc[0]; full_n = int(all_pairs.subject.nunique())
                unknown = int(all_pairs[all_pairs.repeat_kind == "unavailable"].subject.nunique())
                blocks.append(_paragraph(f"Visual 的 D1 有效跨 Part 配对为 {int(row.subjects)}/{full_n} 名完整计划被试；"
                    +f"其中 {unknown} 人缺少重复元数据，元数据可用的 {int(row.planned_subjects)} 人中另有 {int(row.planned_subjects-row.subjects)} 人缺少所选端点共同支持。"
                    +f"有效配对的 Spearman rho={_num(row.spearman_rho)}，95% 区间 [{_num(row.rho_ci_low)}, {_num(row.rho_ci_high)}]。"
                    +"不能把元数据子集当作完整被试分母；本结果属于同次访问跨阶段一致性。"))
        plot_metrics = [metric for metric in ("D1", "Hb_residual_to_zero_MSE_ratio", "D_EH", "D_drive_H") if pairs.metric.eq(metric).any()]
        for metric in plot_metrics:
            path = fixed_stability_figure(pairs, [metric], directory/f"{key}_{metric}.png", title=title)
            blocks.append(_figure(path, "每点为一个被试×任务的保留 A/B 值；同被试可有多个任务点。虚线是相等线；同一指标的 A/B 轴范围相同，颜色区分数据集。每个任务的排序一致性和绝对差由相邻表分别报告，散点图不能将全部任务点当独立被试作合并检验。", evidence["paths"][key+"_pairs"]))
        selected = summary[summary.metric.isin(plot_metrics)]
        available = selected[(selected.subjects >= 6) & np.isfinite(selected.spearman_rho)]
        kinds = dict(same_task_native_record="同任务原生记录", same_visit_cross_part="同次访问跨 Part",
                     within_record_early_late="同记录早晚", unavailable_different_video_stimuli="不同视频不可重测",
                     unavailable="缺少重复元数据")
        def stability_table(part, caption):
            rows = []
            for row in part.itertuples():
                rows.append([DATASET_LABELS.get(str(row.dataset), str(row.dataset)), _fixed_task_label(row.task_label)+" / "+kinds.get(row.repeat_kind, row.repeat_kind),
                    _fixed_metric_code(row.metric), f"{int(row.subjects)}/{int(row.planned_subjects)}", _num(row.spearman_rho),
                    f"[{_num(row.rho_ci_low)}, {_num(row.rho_ci_high)}]",
                    _num(row.median_signed_difference), _num(row.median_absolute_difference), str(row.status)])
            result = _table(["数据集", "任务", "指标", "有效／计划被试", "Spearman ρ", "ρ 95% 区间", "差值中位数 B−A", "绝对差中位数", "状态"], rows,
                caption, [1.1, 1.1, .9, .8, .7, 1.1, .8, .8, 1.2])
            result["source"] = str(evidence["paths"][key+"_summary"])
            return result
        blocks.append(stability_table(available,
            "正文只显示可估计的 primary 指标及同支持残差/零 MSE 比。Spearman ρ 衡量被试排序，绝对差衡量数值接近程度；95% 百分位区间按整个被试重采样。完整诊断、无支持及不足样本状态在末尾附录和 owning CSV。差值与指标同单位。"))
        if key == "repeat":
            absent = [DATASET_LABELS.get(str(dataset), str(dataset)) for dataset, group in summary.groupby("dataset", sort=True)
                      if not (group.subjects > 0).any()]
            if absent:
                blocks.append(_paragraph("、".join(absent)+"没有可用的同条件原生记录重测；不同视频或任务不补作重复，完整不可用分母见附录。"))
        appendix = stability_table(summary,
            "完整保留所有指标、任务、重复范围、计划分母与不可用原因；这些行不作为正文有效配对子集的独立新样本。CSV 是数值 owner，附录仅为只读展示。")
        content.setdefault("_appendix_sections", []).append(dict(title="附录："+title+"完整诊断表", page_break=True, blocks=[appendix]))
        blocks.append(_paragraph("稳定的失配排序可能来自稳定的信号幅度或测量链。D1 与同支持残差／零预测 MSE 比应并列解释；即使排序可重复，也不能直接称为神经血管耦合特质或生理参数准确性。"))
        content["sections"].append(dict(title=title, blocks=blocks, page_break=True))


def _append_fixed_examples(content, evidence, directory):
    examples = evidence["examples"]
    if not examples:
        content["sections"].append(dict(title="典型波形", blocks=[_paragraph("未保留声明选择规则的示例数组，本报告不按视觉效果补选案例。")]))
        return
    index = evidence["example_index"]
    content["sections"].append(dict(title="典型波形的选择与信息条件", page_break=True, blocks=[
        _paragraph("选择规则："+str(index.get("selection", index.get("rule", "以 examples.json 为准")))+"。按误差排序选择仅用于解释，不用于调参或确认性比较；角色、候选分母与重复使用的被试身份见下表。驱动差示例与 D1 示例分开识别，多个示例不是独立重复。"),
        _table(["数据集", "选择角色", "被试／任务／记录／通道对／窗口", "候选分母"], [
            [DATASET_LABELS.get(e["dataset"], e["dataset"]), str(e.get("role", "未声明")),
             " / ".join(str(e.get(key, "未声明")) for key in ("subject", "task_label", "record", "pair", "window")),
             e.get("selection_denominator", e.get("candidate_denominator", "未声明"))] for e in examples],
            "精确身份、选择目标与候选分母以 owning examples.json 为准。"),
    ]))
    retained = [_array_data(evidence, example) for example in examples]
    if any(any(key.startswith("free_") for key in arrays) for arrays in retained):
        raise ValueError("fixed examples must not contain free-parameter comparison trajectories")
    limits = {}
    for dataset in sorted({example["dataset"] for example in examples}):
        component_limits = []
        for component in (1, 2):
            values = []
            for example, arrays in zip(examples, retained):
                if example["dataset"] != dataset:
                    continue
                factor = float(arrays.get("hb_factor", 1.))
                for key in ("target", "hb_only_prediction", "eeg_only_prediction", "joint_prediction"):
                    value = arrays[key][:, component]/factor
                    values.extend(value[np.isfinite(value)].tolist())
            lower, upper = (min(values), max(values)) if values else (0., 1.)
            margin = max((upper-lower)*.05, abs(upper)*.01, 1e-12)
            component_limits.append((lower-margin, upper+margin))
        limits[dataset] = component_limits
    for number, (example, arrays) in enumerate(zip(examples, retained)):
        validation = _example_validation(example, {}, {})
        title = (DATASET_LABELS.get(example["dataset"], example["dataset"])+" / "+str(example.get("role", ""))
                 +" / "+str(example["subject"])+" / "+_fixed_task_label(example.get("task_label", "")))
        source = str(_path(evidence["run"], example["array_path"]))
        path = example_observation_figure(arrays, directory/f"fixed_example_{number:02}_observations.png",
            title=title, hb_unit=example.get("hb_unit", "保留缓存 Hb 坐标"), validation=validation,
            limits=limits[example["dataset"]])
        residual_path = example_residual_figure(arrays, directory/f"fixed_example_{number:02}_residual.png", validation=validation)
        diagnostics = example_time_diagnostics(arrays, validation, dict(status="unavailable", reason="本页不开展空间时移或额外拟合"))
        lookup = {(row["prediction"], row["component"]): row for row in diagnostics["components"]}
        component_rows = []
        for view, label in (("hb_only_prediction", "Hb-only"), ("joint_prediction", "联合")):
            values = [lookup[(view, component)] for component in ("HbO", "HbR", "EEG")]
            component_rows.append([label, *[_num(row["rms"], 4) for row in values],
                "/".join(str(row["scored_bins"]) for row in values), validation[view]["label"]])
        component_table = _table(["条件", "HbO RMS", "HbR RMS", "EEG RMS", "评分 bins O/R/E", "数值状态"], component_rows,
            "RMS 单位为各自冻结训练 SD。Hb 用共同支持，EEG 用自身支持；未通过的有限轨迹仅为失败诊断。完整峰值、分段及支持数字保留在本例 content.json 的 time_diagnostics。")
        component_table["source"] = source
        hb = [lookup[("hb_only_prediction", component)] for component in ("HbO", "HbR")]
        finite_hb = [row for row in hb if row["rms"] is not None]
        reading = ""
        if finite_hb:
            focus = max(finite_hb, key=lambda row: row["rms"])
            reading = (f"本例 Hb-only 的 {focus['component']} RMS 较大；最大绝对残差的有符号值为 {_num(focus['signed_peak'], 4)}，"
                       +("全段零残差无唯一峰时刻。" if focus["peak_time_s"] is None else f"位于窗口内 {focus['peak_time_s']:.1f} 秒。"))
        score = "；".join(metric+"="+_num(example.get(metric), 4) for metric in ("D1", "D_EH", "D_drive_H", "D4"))
        blocks = [_paragraph("身份："+" / ".join(str(example.get(key, "未声明")) for key in ("subject", "task_label", "record", "pair", "window"))+"。"+score+"。"),
            _figure(path, "上两行：处理后实测 Hb、Hb-only 条件重建、联合条件重建及 EEG-only 的独立 Hb 预测。第三行：实测 EEG 代理与各条件驱动的 EEG 读出，以冻结 EEG SD 计；第四行：各自独立正则化拟合的条件驱动，不是神经真值。图例和线型同时区分模型并标明数值接纳状态。灰阴影为可见不评分支持，缺失段断线；Hb 两行使用原缓存单位（通过保留共同因子换回），同库所有样例共享各自 HbO/HbR 纵轴；EEG、驱动和残差图按各例独立范围，不逐条标准化。", source),
            component_table,
            _figure(residual_path, "每条残差为实测减预测，再除以对应冻结训练 SD；HbO/HbR 共用评分 mask，EEG 使用自身支持。正值表示预测偏低，负值表示预测偏高；零线表示该时刻无差。Hb-only 的双 Hb 残差形成 D1，EEG 残差形成 D_EH；联合残差形成 D4 的三个分量。", source),
            _paragraph(reading+"此页残差时段与峰值仅定位模型条件偏差；持续偏置和孤立尖峰都会贡献均方误差。曲线相近不证明驱动唯一或生理来源已分离；驱动间差异还受代理、观测算子、正则化和 DC 规范影响。"),
        ]
        content["sections"].append(dict(title=title, blocks=blocks, page_break=True, time_diagnostics=diagnostics))


def _append_fixed_shapes(content, evidence, directory):
    path = evidence["paths"].get("shape_summary")
    frame = evidence["tables"].get("shape_subjects")
    if path is None or frame is None:
        content["sections"].append(dict(title="残差形态与时间结构", blocks=[
            _paragraph("本导出未收到已完成的时序形态产物；标量 RMS 不能替代残差持续性、符号或时序结构的直接证据。")]))
        return
    summary = _json(path)
    if summary.get("status") != "completed":
        raise ValueError("shape section requires completed analysis evidence")
    if not {"dataset", "subject", "feature", "value", "planned_windows", "available_windows"} <= set(frame):
        raise ValueError("shape_subjects requires explicit features and support denominators")
    if frame.duplicated(["dataset", "subject", "feature"]).any():
        raise ValueError("shape subject distribution must contain one value per subject and feature")
    counts = frame[["planned_windows", "available_windows"]].to_numpy(float)
    if (not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any()
            or (counts[:, 1] > counts[:, 0]).any()):
        raise ValueError("invalid shape support denominator")
    definitions = summary.get("feature_definitions", {})
    if not isinstance(definitions, dict):
        raise ValueError("shape feature definitions must be keyed by feature")
    blocks = [_paragraph(str(summary.get("scope_description", "形态只来自预先几何选择的三通道面板，不代表全窗口总体。"))),
        _paragraph("形态描述使用保留的完整数值数组和原评分 mask；每个残差是处理后实测减 Hb-only 条件重建，再除以冻结训练 SD。时间支持、分母和聚合顺序按每个特征定义保留。被试汇总是主要独立单元；连续时间箱和通道不是独立样本。")]
    for feature in frame.feature.astype(str).drop_duplicates():
        definition = definitions.get(feature)
        if not isinstance(definition, dict) or not all(key in definition for key in ("label", "unit", "description")):
            raise ValueError(f"shape feature lacks a visible definition: {feature}")
        blocks.append(_paragraph(str(definition["label"])+"："+str(definition["description"])+"；单位："+str(definition["unit"])+"。"))
        if definition.get("latex"):
            blocks.append(dict(type="equation", latex=definition["latex"], caption=definition["label"], source=str(path)))
        plt = _plotting(); fig, axis = plt.subplots(figsize=(7, 3.5))
        rows = []
        selected = frame[frame.feature.astype(str) == feature]
        for color, (dataset, part) in zip(("#246eab", "#af4571", "#278579", "#d17d25"), selected.groupby("dataset", sort=True)):
            values = np.asarray(part.value, float)
            x, y = weighted_ecdf(values)
            if len(x):
                axis.step(x, y, where="post", color=color, lw=1.5,
                          label=DATASET_LABELS.get(str(dataset), str(dataset))+f" n={len(x)}")
            good = values[np.isfinite(values)]
            rows.append([DATASET_LABELS.get(str(dataset), str(dataset)), len(part), len(good),
                _num(np.median(good)) if len(good) else "不可用",
                int(part.planned_windows.sum()), int(part.available_windows.sum())])
        axis.set(title=str(definition["label"]), xlabel=str(definition["unit"]), ylabel="被试累计比例", ylim=(0, 1.04))
        axis.grid(alpha=.17); axis.legend(fontsize=8, frameon=False); fig.tight_layout()
        figure_path = _save(fig, directory/("shape_"+feature+".png"))
        blocks.append(_figure(figure_path, "每条曲线由同一数据集的有效被试汇总组成；纵轴为累计被试比例，n 为有效被试数。各数据集在同一特征内共享尺度；不同特征不共用阈值。曲线高低只解释定义的形态量，不推断来源成分。", evidence["paths"]["shape_subjects"]))
        table = _table(["数据集", "计划被试", "有效被试", "中位数", "计划窗口对", "有效窗口对"], rows,
            "分位数只描述有效子集；完整计划和失败仍由 owning 形态表保留。")
        table["source"] = str(evidence["paths"]["shape_subjects"]); blocks.append(table)
    blocks.extend(_paragraph(str(text)) for text in summary.get("report_findings", []))
    for figure in summary.get("report_figures", []):
        blocks.append(_figure(_path(evidence["analysis"], figure["path"]), figure["caption"], str(path)))
    content["sections"].append(dict(title="残差形态与时间结构：预定面板", blocks=blocks, page_break=True))
    figure_path, geometry = projection_counterexample_figure(directory/"independent_noise_projection.png")
    content["sections"].append(dict(title="共同残差不唯一指向共同生理源", page_break=True,
        geometry_illustration=geometry, blocks=[
            _paragraph("残差的高 HbO/HbR 相关与共同方向平方比不能直接用于来源归因。一个无需拟合实测数据的解析反例是：两个观测仅含独立、同方差、零均值噪声，但模型只能拟合反向的一维方向。把观测投到该方向后，两色团留下完全相同的残差。"),
            {"type": "equation", "latex": r"\begin{aligned}\boldsymbol y&=(\epsilon_O,\epsilon_R)^\top,\quad \epsilon_O\perp\epsilon_R,\quad \operatorname{Var}(\epsilon_O)=\operatorname{Var}(\epsilon_R)>0,\quad\boldsymbol v=(1,-1)^\top/\sqrt{2}\\\widehat{\boldsymbol y}&=\boldsymbol v\boldsymbol v^\top\boldsymbol y,\qquad\boldsymbol r=(I-\boldsymbol v\boldsymbol v^\top)\boldsymbol y=\tfrac12(\epsilon_O+\epsilon_R,\epsilon_O+\epsilon_R)^\top\\r_O&=r_R,\qquad\rho(r_O,r_R)=1,\qquad F_c=1\end{aligned}",
             "caption": "线性投影几何的解析反例；独立观测噪声也能在受约束拟合后产生完全共同的残差。"},
            _figure(figure_path, f"每点是一组独立标准正态噪声生成样本；n={geometry['samples']}，固定 seed={geometry['seed']}，与实测被试或窗口无关。左图显示两个独立观测，蓝虚线是模型可拟合的反向方向；右图是投影残差，绿色对角线表示两分量相等。两图共享合成单位和坐标范围。", "报告内解析几何示意；experiments/scripts/render_tak_mismatch.py::projection_counterexample_figure"),
            _paragraph("该反例说明受约束拟合本身就可能把独立噪声投到共同残差方向。它不是实际 Tak 模型的秩证明，也不表示本轮所有残差由几何造成；它排除了“共同残差必有共同生理源”的推断。浅表／全身血流仍是候选机制，需要独立观测或干预才能与拟合几何、预处理和其他共同扰动区分。"),
        ]))


def projection_counterexample_figure(destination, *, seed=20261029, samples=2000):
    """A reproducible algebra illustration, never a fitted measured result."""
    plt = _plotting()
    y = np.random.default_rng(seed).normal(size=(samples, 2))
    v = np.array([1., -1.])/np.sqrt(2.)
    residual = y-y@np.outer(v, v)
    if not np.allclose(residual[:, 0], residual[:, 1], rtol=0, atol=1e-12):
        raise ValueError("linear projection illustration failed its analytic identity")
    common = (residual[:, 0]+residual[:, 1])/np.sqrt(2.)
    difference = (residual[:, 0]-residual[:, 1])/np.sqrt(2.)
    fraction = float(np.sum(common**2)/np.sum(common**2+difference**2))
    if not np.isclose(fraction, 1., rtol=0, atol=1e-14):
        raise ValueError("linear projection illustration lost its common direction identity")
    limit = float(max(np.max(np.abs(y)), np.max(np.abs(residual))))*1.04
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.1), sharex=True, sharey=True)
    for axis, values, title, color in ((axes[0], y, "独立观测噪声", COLORS["target"]),
                                       (axes[1], residual, "受约束投影后的残差", COLORS["spatial"])):
        axis.scatter(values[:, 0], values[:, 1], s=3, alpha=.2, color=color, edgecolors="none", rasterized=True)
        axis.axhline(0, color="#aaa", lw=.5); axis.axvline(0, color="#aaa", lw=.5)
        axis.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="第一分量 / 合成单位", ylabel="第二分量 / 合成单位", title=title)
        axis.set_aspect("equal", adjustable="box")
    axes[0].plot([-limit, limit], [limit, -limit], color=COLORS["hb"], ls="--", lw=1., label="模型可拟合方向")
    axes[1].plot([-limit, limit], [-limit, limit], color=COLORS["spatial"], lw=.8, label="两残差相等")
    for axis in axes: axis.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    return _save(fig, destination), dict(seed=seed, samples=samples, scope="analytic_linear_geometry_illustration_not_Tak_rank_or_source_identification",
        input_sample_correlation=float(np.corrcoef(y.T)[0, 1]), residual_sample_correlation=float(np.corrcoef(residual.T)[0, 1]),
        common_fraction=fraction, analytic_residual_correlation=1, analytic_common_fraction=1)


def _append_fixed_ocular(content, evidence, directory):
    path = evidence["paths"].get("ocular_summary")
    blocks = [
        _paragraph("眼电（EOG）只在有可审计同步支持的数据中作为观测锚点。固定 Tak 原生 EEG 分支未应用通用缓存的 EOG 回归，因此 EOG 关联提示需检查眼源污染；任务共同驱动、姿势或头动也可同时改变 EOG、EEG 代理和 Hb，不能据此认定眼电造成失配或报告因果占比。"),
        _paragraph("浅表／全身血流也可能参与处理后 Hb 波形。HbO 主导或慢变残差与这类解释相容但不具来源特异性；没有独立浅层、循环或干预对照时，不能把 D1 残差命名为已经分离出的非神经生理成分。"),
        _paragraph("EOG 活动来自未滤波的原生水平／垂直眼电（HEOG/VEOG）。每窗在 5–29.9 秒原生样本支持内对每通道去均值，再取两个通道 RMS 平方的等权平均并开根。100 ms 活动曲线沿用同一个评分窗均值，不逐 bin 去均值或归一化；原生 EOG 单位与 EEG 训练 SD 分开标示。"),
    ]
    if path:
        ocular = _json(path)
        if ocular.get("status") != "completed":
            raise ValueError("ocular report section requires completed analysis evidence")
        if ocular.get("scope_description"):
            blocks.insert(0, _paragraph(str(ocular["scope_description"])))
        blocks.extend(_paragraph(str(text)) for text in ocular.get("report_findings", []))
        for figure in ocular.get("report_figures", []):
            declared = _path(evidence["analysis"], figure["path"])
            blocks.append(_figure(declared, str(figure["caption"]), str(path)))
        for table in ocular.get("report_tables", []):
            block = _table(table["columns"], table["rows"], table.get("caption", ""), table.get("widths"))
            block["source"] = str(path); blocks.append(block)
    else:
        blocks.append(_paragraph("本导出未收到已完成、声明支持分母的 EOG 关联产物；本节仅保留解释边界，不填补未经核验的关联数值。"))
    blocks.append(_paragraph("文献背景：Plöchl 等（2012，EEG 与眼追实验）显示眼源成分具有多个来源并依赖参考设置；Kirilina 等（2012）直接测得任务诱发头皮血流对 fNIRS 的贡献。这些证据说明候选机制存在，不替代本数据的来源鉴别。"))
    content["sections"].append(dict(title="生理锚点：相关证据与来源归属分开", blocks=blocks, page_break=True))
    if path:
        _append_ocular_results(content, evidence, directory, ocular)
    content["sources"].extend([
        "Plöchl et al., 2012: https://pmc.ncbi.nlm.nih.gov/articles/PMC3466435/",
        "Kirilina et al., 2012: https://pmc.ncbi.nlm.nih.gov/articles/PMC3348501/",
        "Shin et al., 2018（原数据与任务顺序）: https://www.nature.com/articles/sdata20183",
    ])


def ocular_association_figure(frame, destination, *, title, unit):
    """Forest display of retained subject-level effects and confidence limits."""
    plt = _plotting()
    frame = frame.sort_values(["dataset", "endpoint", "contrast", "analysis"])
    fig, axis = plt.subplots(figsize=(8, max(3.4, .35*len(frame)+1.3)))
    labels = []
    palette = {dataset: color for dataset, color in zip(sorted(frame.dataset.unique()),
               ("#246eab", "#af4571", "#278579", "#d17d25"))}
    for index, row in enumerate(frame.itertuples()):
        label = (DATASET_LABELS.get(str(row.dataset), str(row.dataset))+" / "+OCULAR_LABELS.get(row.endpoint, row.endpoint))
        label += " / "+OCULAR_LABELS.get(row.contrast, OCULAR_LABELS.get(row.analysis, row.analysis))
        labels.append(label+f" ({int(row.subjects)}/{int(row.planned_subjects)})")
        if np.isfinite(row.mean):
            color = palette[row.dataset]
            if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
                axis.plot([row.ci_low, row.ci_high], [index, index], color=color, lw=1.5)
            axis.scatter([row.mean], [index], color=color, marker="o", s=27, zorder=3)
    axis.axvline(0, color="#777", ls="--", lw=.9)
    axis.set_yticks(range(len(labels)), labels, fontsize=8)
    axis.invert_yaxis(); axis.set(xlabel=unit, title=title)
    axis.grid(axis="x", alpha=.17); fig.tight_layout()
    return _save(fig, destination)


def ocular_counterfactual_figure(frame, destination):
    """Each subject remains paired across all three observed EEG coordinates."""
    plt = _plotting()
    datasets = sorted(frame.dataset.astype(str).unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(8, 4), squeeze=False, sharey=True)
    columns = ["eog_"+name+"_DEH_mse" for name in ("raw", "clean", "amplitude")]
    for axis, dataset in zip(axes.flat, datasets):
        part = frame[frame.dataset.astype(str) == dataset].sort_values("subject")
        for row in part.itertuples(index=False):
            values = np.asarray([getattr(row, column) for column in columns], float)
            if np.isfinite(values).all():
                axis.plot(range(3), values, color="#8b9baa", alpha=.45, lw=.7, marker="o", markersize=2.8)
        axis.set_xticks(range(3), ["原代理\nraw", "EOG 回归后\nclean", "原代理匹配均值/SD\namp"])
        axis.set(title=DATASET_LABELS.get(dataset, dataset), ylabel="固定 Hb 解读出误差 / EEG 训练 SD 平方")
        axis.grid(axis="y", alpha=.17)
    fig.tight_layout()
    return _save(fig, destination)


def _append_ocular_results(content, evidence, directory, ocular):
    frame = evidence["tables"].get("ocular_associations")
    if frame is None or frame.empty:
        content["sections"].append(dict(title="EOG 关联的有效支持", blocks=[
            _paragraph("没有保留可估计的被试层 EOG 关联；缺失不记为无关联或零效应。")]))
        return
    required = {"analysis", "dataset", "endpoint", "contrast", "subjects", "planned_subjects", "mean",
                "median", "ci_low", "ci_high", "p_value", "q_value", "fdr_family", "fdr_family_size"}
    if not required <= set(frame):
        raise ValueError("ocular associations lack the subject-level inference contract")
    source = evidence["paths"]["ocular_associations"]
    methods = ocular.get("method", {})
    for analyses, title, method_key, unit in (
        (("cross_window", "cross_window_partial"), "EOG A：活动强度与失配的记录内关联", "A", "被试等权的记录内秩相关"),
        (("lag_specificity",), "EOG B：零位移是否强于非循环 ±5 秒对照", "B", "零位移相关减两侧位移相关均值"),
        (("counterfactual",), "EOG C：回归与幅度对照改变了 EEG 评分目标", "C", "新目标误差减参照误差 / EEG 训练 SD 平方")):
        selected = frame[frame.analysis.isin(analyses)]
        if selected.empty:
            continue
        blocks = [_paragraph(str(methods.get(method_key, "计算步骤与支持以 owning ocular_summary 为准。"))),
            _paragraph("先折叠同一窗口的三个 Hb 对，再对原生记录等权得到每名被试的效应；图中点为被试等权平均效应，横线是以整个被试为重采样单元的 95% bootstrap 百分位区间。p 来自双侧符号翻转，q 直接引用各声明 family 的 BH 校正；不足 6 名有效被试时保留描述量但不检验。标签括号是有效／计划被试。")]
        if method_key == "A":
            blocks.append(_paragraph("窗间关联使用 EOG 活动 RMS 与同窗失配均方误差的记录内 Spearman 相关；每条记录至少 6 个共同有效窗口。控幅诊断至少 10 窗，对 EOG 和失配的秩同时回归 EEG 代理幅度及 Hb 零变化预测幅度后，再相关残余。控幅后的统计关联仍受未测量共同因素影响，不是 EOG 的因果贡献。"))
            for dataset, values in selected.groupby("dataset", sort=True):
                raw = values[values.analysis == "cross_window"]
                partial = values[values.analysis == "cross_window_partial"]
                if not raw.empty:
                    text = DATASET_LABELS.get(str(dataset), str(dataset))+" 的记录内原始秩相关被试均值为："+"；".join(
                        row.endpoint+" "+_num(row.mean, 4)+f"（q={_num(row.q_value, 4)}）" for row in raw.itertuples())+"。"
                    text += "控幅后："+"；".join(row.endpoint+" "+_num(row.mean, 4)+f"（q={_num(row.q_value, 4)}）" for row in partial.itertuples())+"。"
                    blocks.append(_paragraph(text+"原始关联与控制幅度后的剩余关联分别判断；它们与共同活动或测量扰动假设相容，不特指眨眼造成 Hb 残差。"))
        elif method_key == "B":
            blocks.append(_paragraph("三条系列分别是去均值 EEG 代理的平方、Hb 解 EEG 读出误差的平方，以及两个条件驱动去 DC 后差的平方。零与 ±5 秒都使用同一 149 个中心时间箱（10.0–24.8 秒）；位移不首尾循环，也不搜索最优 lag。先在每窗取零相关减两侧相关均值，再按记录和被试等权。正值仅表示零位移关联较强，不把它当作传导时延或来源鉴别。"))
            if "positive_fraction" in selected and selected.positive_fraction.eq(1).all():
                blocks.append(_paragraph(f"本组 {len(selected)} 项对比的每一名有效被试均为正方向；平均增量范围 {_num(selected['mean'].min(), 4)}–{_num(selected['mean'].max(), 4)}。这支持当前代理中的时间锁定关联，但三个系列共享同一 EEG 代理，不能当作三次独立的机制复现。"))
        else:
            if "counterfactual_stage" in evidence["paths"]:
                stage = _json(evidence["paths"]["counterfactual_stage"])
                if stage.get("status") != "completed":
                    raise ValueError("counterfactual stage must be complete for final reporting")
                text = f"C 计划 {stage['planned_window_pairs']} 个 EOG 两库面板窗口对，三种 EEG 代理本身共同有效 {stage['available_window_pairs']}；"
                if "verification" in evidence["paths"]:
                    verified = _json(evidence["paths"]["verification"])
                    text += f"再要求固定 Hb-only 读出有效后，实际评分 {verified['counterfactual_scored_rows']} 个窗口对，每窗共同 {verified['counterfactual_common_score_bins']} 个箱；"
                blocks.append(_paragraph(text
                    +f"{stage['terminal_records']}/{stage['planned_records']} 条原生记录为终态，实际读取原生信号 {stage['measured_signal_reads']} 条。窗口对和记录不是独立被试；效应 n 仍为下方实际配对被试支持。"))
            blocks.extend([
                {"type": "equation", "latex": r"\begin{aligned}L_k&=\operatorname{mean}_{t\in I_E}\left[\frac{e_k(t)-B_E\widehat r_H(t)}{\sigma_E}\right]^2,\quad k\in\{\mathrm{raw},\mathrm{clean},\mathrm{amp}\}\\e_{\mathrm{amp}}(t)&=\mu_{\mathrm{clean}}+\frac{a_{\mathrm{clean}}}{a_{\mathrm{raw}}}\left[e_{\mathrm{raw}}(t)-\mu_{\mathrm{raw}}\right]\end{aligned}",
                 "caption": "三种 EEG 目标都与同一个固定 Hb-only 读出比较；幅度对照仅匹配 clean 的评分支持内均值与标准差。", "source": str(source)},
                _paragraph("raw 是原 EEG 代理；clean 由整条原生记录的 EOG nuisance 回归后，沿原功率、参考、冻结 PCA/符号/训练因子重算；amp 保留 raw 的时间形状，仅在共同评分支持上匹配 clean 的均值 mu 与标准差 a。这里 a 用当前窗 ddof=0 SD，评分分母 sigma_E 仍是原训练冻结 SD；amp 不读取 Hb。原代理尺度接近零时 amp 不可用，clean 尺度为零时保留常量并标记。Tak、Hb-only 驱动、初态与 Hb 读出均不重拟合。"),
                _paragraph("负的 clean−raw 表示回归后的 EEG 目标更接近固定 Hb 读出；clean−amp 的差值及区间用于检查超出均值／尺度匹配的额外收益。本次 amp 在数值上复现主要平均下降，而 clean−amp 未检出额外改善，这不是等效检验。即使个例 clean 优于 amp，也只显示目标时间结构改变，仍可能删去与眼动相关的真实神经活动。整记录处理是离线敏感性分析，不是独立训练的去噪预测，更不是恢复神经真值。Hb 轨迹及 D1 全部保持不变，因此 C 不能用于把 Hb 残差归因于眼电。"),
            ])
            conditions = evidence["tables"].get("ocular_counterfactual_subjects")
            if conditions is not None:
                conditions = conditions[conditions.dataset.isin(ocular.get("datasets", selected.dataset.unique()))]
            if conditions is not None and not conditions.empty:
                path = ocular_counterfactual_figure(conditions, directory/"ocular_C_three_conditions.png")
                blocks.append(_figure(path, "每条线是一名被试的三个同支持均方误差，三条件均有效才连接；每名被试先对记录等权。纵轴在数据集间统一且不截断。下降表示目标坐标更接近同一 Hb 读出，不表示 Tak 拟合改善或清理后神经信号更准确。", evidence["paths"]["ocular_counterfactual_subjects"]))
                rows = []
                for dataset, values in conditions.groupby("dataset", sort=True):
                    for condition in ("raw", "clean", "amplitude"):
                        v = values["eog_"+condition+"_DEH_mse"].to_numpy(float)
                        v = v[np.isfinite(v)]
                        rows.append([DATASET_LABELS.get(str(dataset), str(dataset)), condition, len(v),
                                     _num(np.mean(v)) if len(v) else "不可用", _num(np.median(v)) if len(v) else "不可用"])
                table = _table(["数据集", "EEG 目标", "有效被试", "误差均值", "误差中位数"], rows,
                    "单位为 EEG 训练 SD 平方；此表只展示 owning counterfactual_subjects 的被试值。效应推断使用下方同被试差值，不能用非配对中位数差替代。")
                table["source"] = str(evidence["paths"]["ocular_counterfactual_subjects"]); blocks.append(table)
                for dataset, values in conditions.groupby("dataset", sort=True):
                    columns = ["eog_"+key+"_DEH_mse" for key in ("raw", "clean", "amplitude")]
                    good = values[np.isfinite(values[columns].to_numpy(float)).all(axis=1)]
                    if good.empty:
                        continue
                    raw_mean, clean_mean, amp_mean = good[columns].mean().to_numpy(float)
                    reduction = f"，按被试均值之比下降 {100*(1-clean_mean/raw_mean):.2f}%" if raw_mean > 0 else ""
                    lower_raw = int((good[columns[1]] < good[columns[0]]).sum())
                    lower_amp = int((good[columns[1]] < good[columns[2]]).sum())
                    blocks.append(_paragraph(DATASET_LABELS.get(str(dataset), str(dataset))
                        +f"：raw→clean 误差均值 {_num(raw_mean, 4)}→{_num(clean_mean, 4)}，amp={_num(amp_mean, 4)}（EEG 训练 SD 平方）"
                        +reduction+f"。clean 优于 raw 的被试 {lower_raw}/{len(good)}，优于 amp 的被试 {lower_amp}/{len(good)}；"
                        +"百分比是条件均值之比，不是被试内相对下降的中位数。"))
                extra = selected[selected.contrast == "clean_minus_amplitude"]
                if len(extra) and (extra.q_value >= .05).all() and ((extra.ci_low <= 0) & (extra.ci_high >= 0)).all():
                    blocks.append(_paragraph("核心负结果是 clean 相对 amp 的额外改善未获支持，95% 区间跨零。原生 EOG 活动与代理幅度、失配有关；参考回归改变代理后跨模态差减小，但同等均值／尺度调整已复现主要收益。未检出额外收益不构成严格等效证明，也不量化眼电造成的失配。"))
        path = ocular_association_figure(selected, directory/("ocular_"+method_key+"_effects.png"), title=title, unit=unit)
        blocks.append(_figure(path, "点是已保存的被试等权平均效应，线为 95% 区间，灰虚线为零。区间跨零表示方向仍不确定；有限点没有区间时只作描述。相同分析内各数据集共用横轴；不同分析的横轴单位不同。", source))
        rows = []
        for row in selected.itertuples():
            rows.append([DATASET_LABELS.get(str(row.dataset), str(row.dataset)), OCULAR_LABELS.get(row.endpoint, str(row.endpoint)),
                OCULAR_LABELS.get(row.contrast, str(row.contrast)), f"{int(row.subjects)}/{int(row.planned_subjects)}", _num(row.mean, 4),
                f"[{_num(row.ci_low, 4)}, {_num(row.ci_high, 4)}]", _num(row.p_value, 4), _num(row.q_value, 4)])
        families = "; ".join(str(family)+"（"+str(int(family_rows.fdr_family_size.max()))+" 项可检验比较）"
                             for family, family_rows in selected.groupby("fdr_family", sort=True))
        table = _table(["数据集", "指标／系列", "对比", "有效／计划被试", "均值效应", "95% 区间", "p", "BH q"], rows,
            "各 family 分开校正："+families+"。"+("C 的 clean−raw 与 clean−amp 为 primary；amp−raw 是尺度诊断，不与 primary 混成一个结论。" if method_key == "C" else "primary 与控幅诊断分别解释，沿用完整 family 分母。"), [1.1, 1.1, 1.8, .8, .8, 1.2, .7, .7])
        table["source"] = str(source); blocks.append(table)
        content["sections"].append(dict(title=title, blocks=blocks, page_break=True))
    _append_ocular_examples(content, evidence, directory)


def ocular_example_arrays(evidence, example):
    """Resolve explicit compressed-array indices and verify the fixed target."""
    fixed = _array_data(evidence, dict(example, array_path=example["fixed_array_path"]))
    window = int(example["window_index"])
    oi, ci, pi = (int(example[key]) for key in
                  ("ocular_array_index", "counterfactual_window_index", "counterfactual_pair_index"))
    with np.load(_path(evidence["run"], example["ocular_array_path"]), allow_pickle=False) as archive:
        if int(archive["window_indices"][oi]) != window:
            raise ValueError("ocular compressed index identifies a different original window")
        activity = np.asarray(archive["rms_10hz"])[oi].copy()
        valid = np.asarray(archive["valid_mask"])[oi].copy() if "valid_mask" in archive else np.isfinite(activity)
    with np.load(_path(evidence["run"], example["counterfactual_array_path"]), allow_pickle=False) as archive:
        if int(archive["window_indices"][ci]) != window or str(archive["pair_ids"][pi]) != str(example["pair"]):
            raise ValueError("counterfactual compressed index identifies a different window or physical pair")
        proxies = {name: np.asarray(archive[name+"_proxy"])[ci, pi].copy() for name in ("raw", "clean", "amplitude")}
    shape = (len(fixed["time_s"]),)
    if activity.shape != (shape[0], 2) or valid.shape != activity.shape or any(value.shape != shape for value in proxies.values()):
        raise ValueError("ocular example arrays must share the retained time grid")
    score = np.asarray(fixed["score_mask"], bool)[:, 0]
    if (not score.any() or not np.isfinite(fixed["hb_only_prediction"][score, 0]).all()
            or any(not np.isfinite(value[score]).all() for value in proxies.values())):
        raise ValueError("ocular example requires complete common EEG scoring support")
    if not np.allclose(proxies["raw"][score], fixed["target"][score, 0], rtol=0, atol=1e-10):
        raise ValueError("ocular raw proxy does not reproduce the retained fixed EEG target")
    metrics = {}
    for name, value in proxies.items():
        mse = float(np.mean(((value[score]-fixed["hb_only_prediction"][score, 0])/fixed["sd"][0])**2))
        if not np.isclose(mse, float(example["eog_"+name+"_DEH_mse"]), rtol=1e-8, atol=1e-10):
            raise ValueError("ocular example MSE disagrees with the retained feature row")
        metrics[name] = mse
    return dict(fixed=fixed, proxies=proxies, activity=np.where(valid, activity, np.nan),
                score_mask=score, metrics=metrics)


def ocular_example_figure(arrays, example, destination):
    plt = _plotting()
    fixed, proxies = arrays["fixed"], arrays["proxies"]
    t, sd = fixed["time_s"], float(fixed["sd"][0])
    readout = fixed["hb_only_prediction"][:, 0]
    colors = dict(raw=COLORS["target"], clean=COLORS["spatial"], amplitude=COLORS["current_dof"])
    labels = dict(raw="raw 原代理", clean="clean EOG 回归后", amplitude="amp 匹配均值/SD")
    styles = dict(raw="-", clean="--", amplitude="-.")
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.4), sharex=True)
    for name, value in proxies.items():
        axes[0].plot(t, value/sd, color=colors[name], ls=styles[name], lw=1.2, label=labels[name])
        axes[1].plot(t, _masked((value-readout)/sd, arrays["score_mask"]),
                     color=colors[name], ls=styles[name], lw=1.1, label=labels[name])
    axes[0].plot(t, readout/sd, color=COLORS["hb"], ls=":", lw=1.6, label="固定 Hb-only EEG 读出")
    axes[0].set_ylabel("EEG 代理／读出\n训练 SD")
    axes[0].legend(ncol=2, fontsize=8, frameon=False)
    axes[1].axhline(0, color="#777", lw=.7)
    axes[1].set_ylabel("目标 − Hb 读出\n训练 SD")
    unit = str(example.get("source_units", "原生单位未记录"))
    if unit.lower().replace("μ", "u").replace("µ", "u") in ("uv", "microvolt", "microvolts"):
        unit = "μV"
    for component, label, color, style in ((0, "HEOG 水平", "#bf7d27", "-"), (1, "VEOG 垂直", "#696969", "--")):
        axes[2].plot(t, arrays["activity"][:, component], label=label, color=color, ls=style, lw=1.1)
    axes[2].set_ylabel("100 ms EOG RMS\n"+unit)
    axes[2].legend(ncol=2, fontsize=8, frameon=False)
    for axis in axes:
        _mark_unscored(axis, t, np.isfinite(proxies["raw"]), arrays["score_mask"])
        axis.grid(alpha=.17)
    axes[2].set_xlabel("窗口内时间 / s；灰区可见但不评分")
    roles = {"high_eog_p90": "EOG 活动 P90", "improvement_p10": "clean 相对 amp 改善例", "nonimprovement_median": "未改善反例"}
    fig.suptitle(DATASET_LABELS.get(example["dataset"], example["dataset"])+" / "+str(example["subject"])+" / "+roles.get(example["role"], example["role"]), fontsize=11)
    fig.tight_layout()
    return _save(fig, destination)


def _append_ocular_examples(content, evidence, directory):
    path = evidence["paths"].get("ocular_examples")
    if path is None or path.suffix != ".json":
        return
    index = _json(path)
    if index.get("schema") != "tak_mismatch_fixed_ocular_examples_v1":
        raise ValueError("unsupported ocular example index")
    roles = {"high_eog_p90": "有效 EOG 活动 P90", "improvement_p10": "clean−amp 负差 P10",
             "nonimprovement_median": "clean−amp 非负差中位数"}
    rows = []
    for selection in index.get("selections", []):
        rows.append([DATASET_LABELS.get(selection["dataset"], selection["dataset"]), roles.get(selection["role"], selection["role"]),
            selection["planned_panel_denominator"], selection["valid_counterfactual_denominator"],
            selection["candidate_denominator"], "已选示例" if selection["status"] == "selected" else "无候选，未补图"])
    table = _table(["数据集", "角色", "计划面板窗口对", "三条件有效候选", "角色内候选", "状态"], rows,
        "先完成总体统计，再从三条件有效子集选择：高活动 P90；clean−amp<0 子集的差值 P10；clean−amp≥0 子集的中位数。优先不同被试；无候选时明确不画。选择用于解释，不估计总体改善比例。")
    table["source"] = str(path)
    content["sections"].append(dict(title="EOG 典型时序：改善和反例的选择分母", blocks=[table], page_break=True))
    for number, example in enumerate(index.get("examples", [])):
        arrays = ocular_example_arrays(evidence, example)
        path_image = ocular_example_figure(arrays, example, directory/f"ocular_example_{number:02}.png")
        metrics = arrays["metrics"]
        delta = metrics["clean"]-metrics["amplitude"]
        title = (DATASET_LABELS.get(example["dataset"], example["dataset"])+" / "+roles.get(example["role"], example["role"])).replace("−", "-")
        unit = "EEG 训练 SD 平方"
        text = ("固定 Hb 读出下，raw/clean/amp 三目标的均方误差为 "
                +" / ".join(_num(metrics[key], 4) for key in ("raw", "clean", "amplitude"))+" "+unit
                +f"；clean−amp={delta:+.4g}。")
        text += "此例 clean 目标较 amp 更接近读出，提示超出均值／尺度匹配的时间结构改变。" if delta < 0 else "此例 clean 未比 amp 更接近读出，不能将 raw 到 clean 的变化解释为超出幅度匹配的收益。"
        content["sections"].append(dict(title=title, page_break=True, blocks=[
            _paragraph("身份："+" / ".join(str(example.get(key, "未记录")) for key in ("subject", "record_id", "task_label", "pair", "window_index"))
                       +"；角色内候选分母 "+str(example["candidate_denominator"])+"，三条件有效候选 "+str(example["valid_counterfactual_denominator"])+"。"),
            _figure(path_image, "上图以同一冻结训练 SD 显示三种 EEG 目标及保持不变的 Hb-only EEG 读出；中图为相应目标减该固定读出的有符号残差。下图是原生 HEOG/VEOG 的 100 ms RMS 活动，按整评分窗通道均值中心化，保留原生单位；不同单位分面，时间轴相同。条件颜色和线型全套一致，灰区不计分；缺失断线。", str(path)),
            _paragraph(text+"变化时段用于定位处理敏感性；本图不证明 EOG 造成该失配，也不把 clean 作为神经真值。个例按差值事后选择，不能反推总体额外收益；总体结论以 C 的配对效应与跨零区间为准。"),
        ]))


def render_fixed_analysis(run, analysis, output):
    """Export fixed evidence through the existing bitmap/searchable renderer."""
    import shlex
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"report output must be fresh; retained exports are never overwritten: {output}")
    evidence = load_fixed_analysis(run, analysis)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tak_fixed_report_", dir=output.parent) as temporary:
        content = build_fixed_content(evidence, temporary)
        content_path = Path(temporary)/"report_content.json"
        content_path.write_text(json.dumps(content, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        checks = render_report(content_path, output)
    checks["content_source"] = str(output/"content.json")
    if "verification" in evidence["paths"]:
        checks["retained_numeric_verification"] = dict(source=str(evidence["paths"]["verification"]), status="passed")
    (output/"delivery_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    command = [str(ROOT/".venv/bin/python"), str(Path(__file__).resolve()), "--run", str(evidence["run"]),
               "--fixed-analysis", str(evidence["analysis"]), "--output", str(output)]
    (output/"source_manifest.json").write_text(json.dumps(dict(schema=SCHEMA, parameter_mode="fixed",
        run=str(evidence["run"]), analysis=str(evidence["analysis"]),
        owning_summary=str(evidence["analysis"]/"summary.json"),
        verification=str(evidence["paths"]["verification"]) if "verification" in evidence["paths"] else None,
        renderer_source=str(Path(__file__).resolve()), shared_renderer_source=str(ROOT/"experiments/scripts/render_source_diagnostics_report.py"),
        statistical_source=str(evidence["analysis"]/"source_pipeline_v1/src/analysis/tak_mismatch.py"),
        working_directory=str(ROOT), argv=command, command=shlex.join(command),
        files={key: str(value) for key, value in evidence["paths"].items()},
        scope="read_only_completed_fixed_analysis", current_dof_used=False,
        inference_recomputation=False, selected_example_display_diagnostics=True,
        synthetic_illustration=[section["geometry_illustration"] for section in content["sections"] if "geometry_illustration" in section],
        selection=evidence["example_index"]), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return checks


def revise_visual_mapping_content(content, audit, audit_path):
    """Apply one confirmed interpretation boundary without changing evidence."""
    if audit.get("schema") != "tak_mismatch_visual_channel_impact_audit_v1" or audit.get("status") != "confirmed":
        raise ValueError("Visual revision requires the confirmed owning Tak impact audit")
    if (audit.get("scope", {}).get("dataset") != "visual_cognitive_motivation"
            or audit.get("unaltered_numerical_evidence", {}).get("status") != "unchanged"):
        raise ValueError("Visual interpretation revision must retain the existing numerical evidence")
    revised = json.loads(json.dumps(content, ensure_ascii=False))
    preserved_tables = [(b["columns"], b["rows"]) for s in content["sections"] for b in s["blocks"] if b["type"] == "table"]
    preserved_latex = [b["latex"] for s in content["sections"] for b in s["blocks"] if b["type"] == "equation"]
    scope = audit["scope"]
    panels = audit["nearest6_panels"]
    if len(panels) != scope["distinct_selected_sites"] or len({p["pair_id"] for p in panels}) != len(panels):
        raise ValueError("Visual impact audit must retain the distinct Tak panel identities")
    panel_rows = []
    for panel in panels:
        slots = panel["nearest6"]
        if len(slots) != 6 or sum(bool(slot["mismatch"]) for slot in slots) != panel["mismatch_count"]:
            raise ValueError("Visual audit mismatch counts disagree with the declared nearest6 slots")
        wrong = [f"{slot['declared']} -> {slot['actual']} ({slot['raw_channel']})" for slot in slots if slot["mismatch"]]
        panel_rows.append([panel["pair_id"], f"{panel['mismatch_count']}/6", "；".join(wrong)])
    hb_notice = ("此次 EEG 编号错误未发现进入保留 Hb-only 目标、Hb 尺度、正则、独立初态、flow 先验或接纳检查的依赖。"
        "D1、同支持残差/零预测 MSE 比、Hb-only 波形、Hb 残差形态、D_bal 及这些端点按自身 Hb 可用性的分布／稳定性，仍限原 Hb 窗口、通道和处理条件保留。"
        "这不是新的 Hb 数据质量认证、修复后复测或生理真值证明，也不承诺未来改变预处理或支持后数值不变。")
    eeg_notice = ("Visual 的 EEG 代理、D_EH、D_drive_H、D_obs、Delta_spatial、D4、联合 Hb 分量及 EEG/驱动形态，"
        "仅为冻结错误映射下的工程诊断；不能支持原声明物理电极、近邻空间或局部跨模态生理结论，也不能据其较高误差推断 Visual 固有生理不相容。")
    subset_notice = ("即使展示 D1，若子集按 EEG／joint 成功或其他受影响量筛选，仍须保留该选择依赖，不能称纯 Hb 全分母。"
        "Visual 的 D1 选例自身 Hb-only 数值可保留，同页 EEG、驱动比较及联合曲线均受上述边界约束。Visual 未纳入 EOG A/B/C，另两库的该组结果不因本次修订改变。")
    revised["title"] = "固定 Tak 生理失配：Visual 通道身份边界修订"
    revised.setdefault("provenance", []).append("解释边界 owner："+str(audit_path))
    revised["summary"].insert(0, "本版解释修订："+audit["report_notice"])
    revised["summary"] = [text.replace("视觉认知与动机：", "视觉认知与动机（含 EEG 读数限旧映射工程诊断）：") for text in revised["summary"]]
    table = _table(["Tak Hb 通道对", "错配槽数", "错误槽：声明电极 -> 实际文档电极（原 A 编号）"], panel_rows,
        "这里是本次 Tak 的六种 nearest6 面板，不能套用比较方法固定 16 通道面板的 6/16 计数。FP/Fp 大小写差异不算错配。", [1.1, .65, 4.2])
    table.update(source=str(audit_path), revision_audit_table=True)
    audit_section = dict(title="解释修订审计：Visual 原生 EEG 电极身份", page_break=True, blocks=[
        _paragraph(audit["report_notice"]),
        _paragraph(f"冻结 Tak 范围为 {scope['native_records']} 条 Visual 原生记录、{scope['subjects']} 名被试，"
            +f"{scope['affected_selected_sites']}/{scope['distinct_selected_sites']} 种预定近邻面板全部受影响。"
            +"中央读取将 Location.ced 行号误当 raw A 编号；命中的是实际电极选择与几何关系，不能用重命名恢复既有结果的原声明身份。"),
        table, _paragraph(eeg_notice), _paragraph(hb_notice), _paragraph(subset_notice),
        _paragraph("本版复用修订前报告的保留数值、表格、PNG 和 LaTeX，仅修订解释范围；没有修复 loader、重建 cache、重估 PCA、重新拟合 Tak 或重算指标。修订前成稿按映射问题确认前的历史版本保留，当前应使用本修订版。"),
    ])
    notice = "【Visual 解释边界】含 EEG／空间／联合条件的读数仅为冻结错误映射下工程诊断；纯 Hb-only 按原 Hb 支持保留，不能据此作来源或声明电极位置归因。"
    for section in revised["sections"]:
        title = section["title"]
        for block in section["blocks"]:
            if block["type"] == "figure":
                block["preserve_bitmap"] = True
        visual_section = "视觉认知与动机" in title
        mixed_section = any(key in title for key in ("被试等权分布", "残差形态", "同任务原生", "同记录早晚"))
        if visual_section:
            section["blocks"].insert(0, _paragraph(eeg_notice+" "+hb_notice))
        if title == "范围、分母与固定条件":
            section["blocks"].insert(0, _paragraph("Visual 的六近邻在本轮仅为按错误标签选出的名义近邻；其 EEG/空间解释以新增审计为准。"))
        if title == "典型波形的选择与信息条件":
            section["blocks"].insert(0, _paragraph(subset_notice))
        if title == "这些结果能支持的解释与用途":
            section["blocks"].insert(0, _paragraph(eeg_notice+" "+hb_notice+" "+subset_notice))
        if visual_section or mixed_section:
            if mixed_section:
                section["blocks"].insert(0, _paragraph(notice))
            for block in section["blocks"]:
                if block["type"] in ("figure", "table"):
                    block["caption"] = block.get("caption", "")+" "+notice
    position = next((i for i,s in enumerate(revised["sections"]) if s["title"] == "范围、分母与固定条件"), 0)
    revised["sections"].insert(position, audit_section)
    revised.setdefault("sources", []).append(str(audit_path))
    revised["sources"].append(str(audit["comparison_audit"]["path"]))
    after_tables = [(b["columns"], b["rows"]) for s in revised["sections"] for b in s["blocks"]
                    if b["type"] == "table" and not b.get("revision_audit_table")]
    after_latex = [b["latex"] for s in revised["sections"] for b in s["blocks"] if b["type"] == "equation"]
    if after_tables != preserved_tables or after_latex != preserved_latex:
        raise ValueError("interpretation revision changed retained tables or equations")
    revised["interpretation_revision"] = dict(owner=str(audit_path), scope="Visual_channel_identity_boundary_only",
        retained_table_cells_unchanged=True, retained_latex_unchanged=True, numerical_recomputation=False)
    return revised


def revise_fixed_visual_report(run, previous, audit_path, output):
    """Version an existing report using its portable assets and one impact audit."""
    import shlex
    from PIL import Image
    run, previous, audit_path, output = (Path(value).resolve() for value in (run, previous, audit_path, output))
    if output.exists():
        raise FileExistsError("interpretation revision requires a fresh report directory")
    if not audit_path.is_relative_to(run):
        raise ValueError("Tak impact audit must belong to the declared retained run")
    original_manifest = _json(previous/"source_manifest.json")
    if Path(original_manifest["run"]).resolve() != run:
        raise ValueError("previous report belongs to a different retained run")
    audit = _json(audit_path)
    if Path(audit["retained_run"]).resolve() != run:
        raise ValueError("Visual impact audit belongs to a different retained run")
    original = _json(previous/"content.json")
    revised = revise_visual_mapping_content(original, audit, audit_path)
    for section in revised["sections"]:
        for block in section["blocks"]:
            if block["type"] == "figure":
                block["path"] = str(_path(previous, block["path"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tak_mapping_revision_", dir=output.parent) as temporary:
        content_path = Path(temporary)/"report_content.json"
        content_path.write_text(json.dumps(revised, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        checks = render_report(content_path, output)
    identical = []
    for folder in ("figures", "formulas"):
        for path in sorted((previous/folder).glob("*.png")):
            new = output/folder/path.name
            with Image.open(path) as first, Image.open(new) as second:
                if first.size != second.size or not np.array_equal(np.asarray(first.convert("RGB")), np.asarray(second.convert("RGB"))):
                    raise ValueError(f"interpretation-only revision changed a retained bitmap: {path.name}")
            identical.append(str(path.relative_to(previous)))
    checks.update(content_source=str(output/"content.json"), interpretation_revision=dict(
        owner=str(audit_path), previous_report=str(previous), tables_and_latex_unchanged=True,
        identical_bitmap_count=len(identical), unchanged_bitmap_paths=identical, numerical_recomputation=False))
    checks["retained_numeric_verification"] = original_manifest.get("verification")
    (output/"delivery_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    command = [str(ROOT/".venv/bin/python"), str(Path(__file__).resolve()), "--run", str(run),
        "--revise-report", str(previous), "--visual-channel-audit", str(audit_path), "--output", str(output)]
    manifest = dict(original_manifest, renderer_source=str(Path(__file__).resolve()), working_directory=str(ROOT),
        argv=command, command=shlex.join(command), supersedes=str(previous/"source_manifest.json"),
        visual_channel_impact_audit=str(audit_path), interpretation_revision=revised["interpretation_revision"],
        image_replotting=False, metrics_recomputed=False, previous_report_preserved=True)
    (output/"source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return checks


def spatial_topomap(ax, frame, *, limits, title="", cmap="viridis", names=False):
    """MNE outlines/linear topomap, masked to nearby measured-node triangles."""
    import mne
    from scipy.spatial import Delaunay, QhullError
    from scipy.spatial.distance import cdist
    from matplotlib.colors import Normalize
    from matplotlib.patches import Circle, Ellipse
    from matplotlib.cm import ScalarMappable
    good = frame[np.isfinite(frame.display_value) & np.isfinite(frame.x) & np.isfinite(frame.y)]
    positions = frame[np.isfinite(frame.x) & np.isfinite(frame.y)]
    missing = positions[~np.isfinite(positions.display_value)]
    xy = good[["x","y"]].to_numpy(float)
    values = good.display_value.to_numpy(float)
    norm = Normalize(*limits)
    image = ScalarMappable(norm=norm,cmap=cmap)
    interpolated = False
    if len(good) >= 3 and np.linalg.matrix_rank(xy-xy.mean(axis=0)) == 2:
        try:
            image, _ = mne.viz.plot_topomap(values,xy,axes=ax,show=False,sensors=False,
                names=good.pair_members.tolist() if names else None,contours=0,
                image_interp="linear",extrapolate="local",sphere=(0.,0.,0.,1.),
                cmap=cmap,vlim=limits,res=160,border="mean")
            field = np.asarray(image.get_array(),float)
            x0,x1,y0,y1 = image.get_extent()
            xx,yy = np.meshgrid(np.linspace(x0,x1,field.shape[1]),np.linspace(y0,y1,field.shape[0]))
            triangulation = Delaunay(xy)
            triangles = xy[triangulation.simplices]
            edge = np.stack([np.linalg.norm(triangles[:,i]-triangles[:,j],axis=1) for i,j in ((0,1),(1,2),(2,0))],axis=1).max(axis=1)
            all_xy = positions[["x","y"]].to_numpy(float)
            distances = cdist(all_xy,all_xy);np.fill_diagonal(distances,np.inf)
            spacing = float(np.median(distances.min(axis=1)))
            simplex = triangulation.find_simplex(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
            support = (simplex >= 0) & (xx*xx+yy*yy <= 1.)
            support &= edge[np.maximum(simplex,0)] <= 2.5*spacing
            for point in missing[["x","y"]].to_numpy(float):
                radius = .45*float(np.linalg.norm(xy-point,axis=1).min())
                support &= (xx-point[0])**2+(yy-point[1])**2 > radius*radius
            field = np.ma.array(field,mask=~support | ~np.isfinite(field))
            if field.count():
                if field.min()<values.min()-1e-8 or field.max()>values.max()+1e-8:
                    raise ValueError("linear display interpolation exceeded retained node values")
                image.set_data(field)
                levels = np.linspace(*limits,8)[1:-1]
                levels = levels[(levels>field.min()) & (levels<field.max())]
                if len(levels):ax.contour(xx,yy,field,levels=levels,colors="#273847",linewidths=.45,alpha=.6)
            else:
                image.set_data(np.ma.masked_all(field.shape))
            interpolated = True
        except QhullError:
            ax.clear()
    if not interpolated:
        ax.add_patch(Circle((0,0),1,fill=False,edgecolor="#303c46",lw=1.2))
        for sign in (-1,1):ax.add_patch(Ellipse((1.015*sign,0),.1,.38,fill=False,edgecolor="#303c46",lw=1))
        ax.plot([-.1,0,.1],[.995,1.12,.995],color="#303c46",lw=1)
        ax.text(0,0,"支持不足" if not len(good) else "仅测量节点",ha="center",va="center",fontsize=9,color="#66717b")
    if len(good):ax.scatter(good.x,good.y,c=values,cmap=cmap,norm=norm,s=18,edgecolor="#1d2a33",lw=.55,zorder=7)
    if len(missing):ax.scatter(missing.x,missing.y,marker="x",c="#737d86",s=22,lw=1,zorder=8)
    ax.set(xlim=(-1.16,1.16),ylim=(-1.08,1.2),aspect="equal");ax.axis("off")
    ax.set_title(title,fontsize=10,pad=5)
    ax.text(.5,-.035,f"可用位置 {len(good)}/{len(positions)}",transform=ax.transAxes,ha="center",fontsize=7.5,color="#455560")
    if len(good) and "period" in good:
        if good.period.eq("subject_mean").all():
            low,high=good.available_records.min(),good.available_records.max()
            count=str(int(high)) if low==high else f"{int(low)}–{int(high)}"
            note=f"可用记录 {count}/{int(good.planned_records.max())}"
        elif "metric" in good and not good.metric.str.startswith("D_bal").all():
            fraction=good.available_windows/good.planned_windows
            note=f"窗口覆盖中位 {100*fraction.median():.0f}%"
        else:note="完整记录的弱核指标"
        ax.text(.5,-.095,note,transform=ax.transAxes,ha="center",fontsize=7,color="#455560")
    return image


def _spatial_record_figure(payload):
    """One retained subject/endpoint, all acquisitions, on one shared scale."""
    import re
    plt = _plotting()
    dataset,subject,metric,frame,output = payload
    output=Path(output)
    key=lambda value:[int(x) if x.isdigit() else x for x in re.split(r"(\d+)",str(value))]
    ids=sorted(frame.loc[frame.period.eq("full"),"map_id"].unique(),key=key)
    panels=[("subject_mean","记录等权图")]+[(identity,identity.split("|")[-1]) for identity in ids]
    finite=frame.display_value.to_numpy(float);finite=finite[np.isfinite(finite)]
    signed=metric=="Delta_spatial"
    extent=max(float(np.max(np.abs(finite))) if len(finite) else 1.,1e-6)
    limits=(-extent,extent) if signed else (0.,extent)
    cmap="RdBu_r" if signed else "viridis"
    label,unit,_=FIXED_METRICS[metric]
    results=[]
    for start in range(0,len(panels),9):
        page=panels[start:start+9];columns=min(3,len(page));rows=math.ceil(len(page)/columns)
        fig,axes=plt.subplots(rows,columns,figsize=(8.5,2.55*rows+.65),squeeze=False)
        fig.subplots_adjust(left=.025,right=.89,bottom=.09,top=.86 if rows==1 else .92,wspace=.13,hspace=.26)
        for ax,(identity,title) in zip(axes.ravel(),page):
            part=frame[(frame.period.eq("subject_mean") if identity=="subject_mean" else frame.period.eq("full") & frame.map_id.eq(identity))]
            image=spatial_topomap(ax,part,limits=limits,title=title,cmap=cmap)
        for ax in axes.ravel()[len(page):]:ax.set_visible(False)
        bar=fig.colorbar(image,cax=fig.add_axes([.925,.2,.022,.56]));bar.set_label(unit,fontsize=9)
        fig.suptitle(f"{DATASET_LABELS[dataset]} · {subject} · {label}",fontsize=13,y=.985)
        fig.text(.46,.018,"同一被试／指标的全部记录共用线性色标；黑边圆点保留位置值，灰叉为缺失，空白不等于零。",ha="center",fontsize=8)
        filename=f"records_{dataset}_{subject}_{metric}_{start//9+1:02}.png"
        _save(fig,output/filename)
        results.append(dict(dataset=dataset,subject=subject,metric=metric,file=filename,
            record_ids=[identity for identity,_ in page],limits=list(limits),panel_count=len(page)))
    return results


def _spatial_overview_figure(maps, selected, dataset, metrics, output, *, centered=False):
    plt=_plotting();subjects=selected.subject.tolist()
    fig,axes=plt.subplots(len(metrics),len(subjects),figsize=(8.6,2.55*len(metrics)+.65),squeeze=False)
    fig.subplots_adjust(left=.025,right=.88,bottom=.065,top=.92,wspace=.13,hspace=.32)
    scales=[]
    for row,metric in enumerate(metrics):
        part=maps[maps.dataset.eq(dataset)&maps.subject.isin(subjects)&maps.metric.eq(metric)&maps.period.eq("subject_mean")].copy()
        if centered:
            part["display_value"]-=part.groupby("subject").display_value.transform("median")
        values=part.display_value.to_numpy(float);values=values[np.isfinite(values)]
        signed=centered or metric=="Delta_spatial"
        maximum=max(float(np.max(np.abs(values))) if len(values) else 1.,1e-6)
        limits=(-maximum,maximum) if signed else (0.,maximum)
        label,unit,_=FIXED_METRICS[metric]
        for col,subject in enumerate(subjects):
            current=part[part.subject.eq(subject)]
            image=spatial_topomap(axes[row,col],current,limits=limits,cmap="RdBu_r" if signed else "viridis",
                title=f"{subject} · {metric}" if metric!="Hb_residual_to_zero_MSE_ratio" else f"{subject} · Hb残差比")
        y=(axes[row,0].get_position().y0+axes[row,0].get_position().y1)/2
        bar=fig.colorbar(image,cax=fig.add_axes([.925,y-.087,.018,.174]))
        bar.set_label(("中位数差 / " if centered else "")+unit,fontsize=8)
        scales.append(dict(metric=metric,limits=list(limits),centered=centered))
    fig.suptitle(DATASET_LABELS[dataset]+("：去除各人整体水平后的空间形状" if centered else "：各通道的记录等权偏离"),fontsize=14,y=.985)
    fig.text(.46,.016,"同一行、同一指标共用线性色标；黑边圆点保留位置值，灰叉为无有效值，空白为未插值区域。",ha="center",fontsize=8)
    filename=dataset+"_"+("centered_" if centered else "overview_")+"_".join(metrics)+".png"
    _save(fig,Path(output)/filename)
    return dict(file=filename,dataset=dataset,metrics=list(metrics),subjects=subjects,scales=scales,kind="overview")


def _spatial_repeat_figure(maps, selected, scores, dataset, output):
    plt=_plotting();fig,axes=plt.subplots(len(selected),3,figsize=(8.6,2.7*len(selected)+.6),squeeze=False)
    fig.subplots_adjust(left=.025,right=.88,bottom=.065,top=.92,wspace=.13,hspace=.35)
    cases=[]
    for row,(_,subject_row) in enumerate(selected.iterrows()):
        eligible=scores[scores.dataset.eq(dataset)&scores.subject.eq(subject_row.subject)&scores.eligible]
        best=eligible.sort_values(["score","record_unit"],ascending=[False,True]).iloc[0]
        data=maps[maps.dataset.eq(dataset)&maps.subject.eq(subject_row.subject)&maps.record_unit.eq(best.record_unit)&maps.metric.eq("D1")]
        values=data.display_value.to_numpy(float);values=values[np.isfinite(values)]
        maximum=max(float(values.max()) if len(values) else 1.,1e-6)
        for col,(period,label) in enumerate((("full","整记录"),("early","前半"),("late","后半"))):
            image=spatial_topomap(axes[row,col],data[data.period.eq(period)],limits=(0,maximum),
                title=f"{subject_row.subject} · {label}")
        y=(axes[row,0].get_position().y0+axes[row,0].get_position().y1)/2
        bar=fig.colorbar(image,cax=fig.add_axes([.925,y-.09,.018,.18]));bar.set_label("D1 / 通道训练SD",fontsize=8)
        axes[row,1].text(.5,-.16,f"{best.record_unit.split('|')[-1]}：早晚空间ρ={best.early_late_rho:.3f}",transform=axes[row,1].transAxes,ha="center",fontsize=8)
        cases.append(dict(subject=subject_row.subject,record_unit=best.record_unit,early_late_rho=float(best.early_late_rho),limits=[0,maximum]))
    fig.suptitle(DATASET_LABELS[dataset]+"：同一记录的前后半空间图",fontsize=14,y=.985)
    fig.text(.46,.014,"每行独立色标，行内三图可直接比较；此一致性参与了选例，不是独立验证。",ha="center",fontsize=8)
    filename=dataset+"_early_late.png";_save(fig,Path(output)/filename)
    return dict(file=filename,dataset=dataset,kind="early_late",cases=cases)


def _spatial_layout_figure(maps,selected,dataset,output):
    from matplotlib.patches import Circle,Ellipse
    plt=_plotting();subject=selected.subject.iloc[0]
    points=maps[maps.dataset.eq(dataset)&maps.subject.eq(subject)&maps.metric.eq("D1")&maps.period.eq("subject_mean")]
    points=points[np.isfinite(points.x)&np.isfinite(points.y)].sort_values(["y","x"],ascending=[False,True])
    fig=plt.figure(figsize=(8.6,6.1));ax=fig.add_axes([.025,.12,.52,.76])
    ax.add_patch(Circle((0,0),1,fill=False,color="#394550",lw=1.3))
    for sign in (-1,1):ax.add_patch(Ellipse((sign*1.015,0),.1,.38,fill=False,color="#394550",lw=1.1))
    ax.plot([-.1,0,.1],[.995,1.12,.995],color="#394550",lw=1.2)
    ax.scatter(points.x,points.y,s=24,facecolor="white",edgecolor="#29465d",lw=.8)
    entries=[]
    for index,(_,point) in enumerate(points.iterrows(),1):
        ax.text(point.x,point.y+.036,str(index),ha="center",va="bottom",fontsize=7)
        entries.append(f"{index:2d}  {point.pair_members}")
    midpoint=math.ceil(len(entries)/2)
    for col,text in enumerate((entries[:midpoint],entries[midpoint:])):
        fig.text(.56+.215*col,.86,"\n".join(text),va="top",fontsize=7.4,linespacing=1.35)
    ax.text(0,1.23,"前",ha="center",fontsize=11);ax.text(-1.17,0,"左",ha="center",fontsize=10);ax.text(1.17,0,"右",ha="center",fontsize=10)
    ax.set(xlim=(-1.3,1.3),ylim=(-1.15,1.3),aspect="equal");ax.axis("off")
    fig.suptitle(DATASET_LABELS[dataset]+"：保留测量位置与通道名称",fontsize=14,y=.97)
    fig.text(.5,.04,"图内数字仅用于右侧名称查询；投影不是个体皮层定位。",ha="center",fontsize=9)
    filename=dataset+"_channel_layout.png";_save(fig,Path(output)/filename)
    return dict(file=filename,dataset=dataset,kind="layout",source_subject=subject,locations=len(points))


def _spatial_overview_batch(payload):
    from src.analysis.tak_mismatch import HB_RATIO
    maps,chosen,scores,dataset,plots=payload
    figures=[_spatial_layout_figure(maps,chosen,dataset,plots)]
    groups=[("D1",HB_RATIO,"D_bal_20s")]
    if dataset!="visual_cognitive_motivation":
        groups += [("D_EH","D_drive_H","D4"),("D_J_E","D_J_O","D_J_R"),("D_obs","Delta_spatial","D_bal_10s","D_bal_40s")]
    else:groups += [("D_bal_10s","D_bal_40s")]
    for group in groups:figures.append(_spatial_overview_figure(maps,chosen,dataset,group,plots))
    figures.append(_spatial_overview_figure(maps,chosen,dataset,("D1",HB_RATIO),plots,centered=True))
    figures.append(_spatial_repeat_figure(maps,chosen,scores,dataset,plots))
    return figures


def render_fixed_topography(run, analysis, output, *, workers=2):
    """Render selected spatial cases and every retained record/endpoint map."""
    import os
    import shutil
    import time
    import pandas as pd
    import mne
    from html import escape
    from concurrent.futures import ProcessPoolExecutor
    from scipy.stats import rankdata
    from src.analysis.tak_mismatch import SPATIAL_HB_METRICS,HB_RATIO
    run,analysis,output=(Path(p).resolve() for p in (run,analysis,output))
    if shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex must be on the rendering service PATH before generating spatial figures")
    summary=_json(analysis/"summary.json")
    if (summary.get("schema")!="tak_mismatch_spatial_patterns_v1" or summary.get("status")!="completed"
            or Path(summary["parent_run"]).resolve()!=run or not analysis.is_relative_to(run)):
        raise ValueError("completed spatial analysis inside the declared retained run required")
    if output.exists():raise FileExistsError("spatial report requires a fresh export")
    if not 1<=workers<=8:raise ValueError("one to eight rendering workers required")
    impact=_json(Path(summary["visual_impact_audit"]))
    if impact.get("status")!="confirmed":raise ValueError("confirmed Visual impact boundary required")
    maps=pd.read_csv(_path(analysis,summary["files"]["map_values"]),dtype={"subject":str})
    ranking=pd.read_csv(_path(analysis,summary["files"]["subject_ranking"]),dtype={"subject":str})
    scores=pd.read_csv(_path(analysis,summary["files"]["record_pattern_scores"]),dtype={"subject":str})
    similarities=pd.read_csv(_path(analysis,summary["files"]["record_similarities"]),dtype={"subject":str})
    selected=ranking[ranking.selected].copy()
    if len(selected)!=len(summary["selected_subjects"]):raise ValueError("selected subject denominator changed")
    output.mkdir(parents=True);plots=output/"plots";plots.mkdir()
    started=time.monotonic();overview=[];jobs=[];overview_jobs=[]
    for dataset in DATASET_LABELS:
        chosen=selected[selected.dataset.eq(dataset)].sort_values("rank")
        if not len(chosen):continue
        overview_jobs.append((maps[maps.dataset.eq(dataset)&maps.subject.isin(chosen.subject)].copy(),chosen,
            scores[scores.dataset.eq(dataset)&scores.subject.isin(chosen.subject)].copy(),dataset,str(plots)))
        for subject in chosen.subject:
            for metric in summary["metrics"]:
                if dataset=="visual_cognitive_motivation" and metric not in SPATIAL_HB_METRICS:continue
                part=maps[maps.dataset.eq(dataset)&maps.subject.eq(subject)&maps.metric.eq(metric)&maps.period.isin(("full","subject_mean"))].copy()
                jobs.append((dataset,subject,metric,part,str(plots)))
    records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for figures in pool.map(_spatial_overview_batch,overview_jobs):overview.extend(figures)
        for start in range(0,len(jobs),2*workers):
            for result in pool.map(_spatial_record_figure,jobs[start:start+2*workers]):records.extend(result)
            (output/"render_progress.json").write_text(json.dumps(dict(completed_jobs=min(start+2*workers,len(jobs)),planned_jobs=len(jobs),elapsed_s=time.monotonic()-started)))
    content=dict(schema="source_diagnostics_report_v1",title="固定 Tak 模型偏离的通道地形图",date=summary["created_utc"],
        subtitle="每库三名探索性选例；完整原生记录、通道、缺失与选择分母均保留。",
        provenance=["数值来源："+str(analysis),"仅重汇总已完成固定参数结果；未重新拟合模型，也未读取原始信号或保护材料。"],
        summary=[f"候选覆盖{summary['subjects']}名被试、{summary['native_records']}条有固定结果的原生记录、{summary['planned_window_pairs']}个计划窗口×通道对。按D1空间反差、邻近一致性与早晚空间相关选择{len(selected)}人；所有候选及失败均在subject_ranking和record_pattern_scores中。",
            "图中的channel是HbO/HbR成对测量位置。D_EH等跨模态量对应该位置的六通道EEG代理，不是单个EEG电极的偏离。Visual只绘制经审计可保留的纯Hb指标。",
            "选择利用了要展示的数据，因此清楚的图案和较高的早晚相关不能作为总体或独立确认性证据。完整逐记录、逐指标PNG位于配套index.html；正文保留总体地图、选择依据与D1逐记录对照。"],sections=[])
    method=[_paragraph("沿用父实验的固定Tak参数（κ=0.64、τ=2秒等），各30秒窗口及Hb通道对的条件驱动与四个初态仍由原拟合分别估计。Hb-only只看双Hb，EEG-only只看该位置的六通道EEG代理，联合拟合同时看三种观测；PCA、观察尺度及空间替代方案在外训练被试中冻结。D1是可见Hb的条件重建偏离，不能当作隐藏Hb预测误差或生理真值恢复。"),
        _paragraph("D1先在各30秒窗口内，按各Hb色团的冻结外训练SD标准化残差并计算RMS；整记录先平均平方量再开方。零预测指窗口参考后的零变化预测，Hb残差比在共同支持上分别汇总残差与零预测MSE后相除。它不是R²或生理来源占比。"),
        dict(type="equation",latex=r"D_{1,r,c}=\sqrt{\frac{1}{|W_{r,c}|}\sum_{w\in W_{r,c}}D_{1,w,c}^{2}},\qquad R_{r,c}=\frac{\sum_{w\in W^*_{r,c}}D_{1,w,c}^{2}}{\sum_{w\in W^*_{r,c}}Z_{w,c}}",caption="r为原生记录，c为Hb通道对，w为窗口；Z是同支持零预测的标准化MSE，W*是两项均有效的窗口集合。"),
        _paragraph("被试地图在每个位置对可用原生记录等权汇总；RMS类指标仍先合并平方，残差比仍分别合并分子与分母。不同长度记录不按窗口数量加权。Visual同次采集的两Probe合成一张左右联合图，不能算两次重复。投影位置重合的通道先合并分子、分母并保留成员身份。"),
        _paragraph("所有指标沿用每个Hb通道对和每个外训练折的冻结尺度；同色表示该指标的标准化偏离相同，不表示原生浓度或绝对生理活动相同。D_bal是完整连续记录的弱平衡诊断，10/20/40秒表示弱核宽度；不把在窗口表中重复出现的同一数值重复计数。"),
        _table(["指标","计算含义","方向／单位"],[[m,FIXED_METRICS[m][2],FIXED_METRICS[m][1]] for m in summary["metrics"]],
            "除Delta_spatial为有符号的近邻−替代邻域Hb MSE外，其余偏离通常以较低为好；这些量不能相加。",[1.3,4.7,1.8]),
        _paragraph("用于选例的每个记录图要求至少8个位置，且至少80%的计划位置在整记录和前后半都可用；各半至少两个有效窗口，并达到80%计划窗口覆盖。按整记录的计划时间排序二分，奇数中间窗丢弃。显示地图则保留全部可计算的有限值，包括部分时间支持，并显示覆盖；没有值的点画灰叉，不能填零。"),
        dict(type="equation",latex=r"s_r=\frac{P_{90}(D_1)-P_{10}(D_1)}{\operatorname{median}(D_1)}\,[I_r]_{0}^{1}\,[\rho_{\mathrm{early,late}}]_{0}^{1},\qquad S=\operatorname{median}_{r\in\mathcal E}(s_r)\frac{|\mathcal E|}{|\mathcal R|}",caption="空间反差×正的邻近一致性×正的早晚排序相关；被试得分再乘合格记录覆盖率。此分数只排序展示候选，没有生理阈值。"),
        _paragraph("I是描述性Moran空间一致性：每个位置连接投影中最近四点，双向取并集，删去超过2.5倍典型最近邻间距的边。ρ为共同位置的Spearman秩相关。通道有空间依赖，不把它们当独立被试检验显著性；早晚一致性参与选择，不能称独立验证。"),
        _paragraph("地图使用MNE的头轮廓和线性插值，黑边有色圆点保留各测量位置的原值，等值线仅帮助读图。仅显示测量点凸包内、最长边不超过2.5倍典型邻距的小三角形；无测量的大间隙留白，缺失位置周围留孔。边缘或近共线位置即使无法形成色面，仍保留有色圆点。插值不产生新增测量或空间分辨率。地图朝上为前、图左为左侧；模板／论文布局不是个体MRI或皮层源定位。"),
        _paragraph("总体图同一行共用完整线性色标。中心化图仅减去各被试地图的空间中位数，红色高于、蓝色低于自身水平；不用于重新选择。逐记录图在同一被试和指标的所有页共用色标；不同被试细节页的色标可不同，须读取颜色条。")]
    content["sections"].append(dict(title="如何计算、挑选和阅读这些地图",page_break=True,blocks=method))
    observations=[]
    for dataset in DATASET_LABELS:
        chosen=selected[selected.dataset.eq(dataset)].sort_values("rank")
        if not len(chosen):continue
        rows=[]
        for _,candidate in chosen.iterrows():
            rows.append([candidate.subject,f"{int(candidate.eligible_records)}/{int(candidate.planned_records)}",_num(candidate.median_spatial_contrast),_num(candidate.median_moran_I),_num(candidate.median_early_late_rho),_num(candidate.score)])
        blocks=[_table(["被试","合格／计划记录","空间反差","邻近I","早晚ρ","选例分数"],rows,
            "表中间三项为合格记录中位数；最终分数是每记录乘积的中位数再乘覆盖率，不等于表内中位数相乘。",[1.2,1.2,1,1,1,1]),
            _paragraph("Single-Trial同任务有跨记录重复；Simultaneous三个任务不同，存在固定任务顺序混杂；REFED不同视频为不同刺激；Visual的Part才是记录重复，两Probe同步。跨任务／视频图相似只说明这些观测条件下的模式关联。")]
        if dataset=="visual_cognitive_motivation":blocks.append(_paragraph(impact["report_notice"]))
        for figure in [f for f in overview if f["dataset"]==dataset]:
            if figure["kind"]=="layout":caption="数字对应右侧通道名称；布局取本库第一名选例的保留坐标，其余病例继续使用各自记录坐标。"
            elif figure["kind"]=="early_late":caption="每行选该被试选例分数最高的合格记录，整记录／前半／后半依次展示；行内共用色标、行间独立。高相关已经参与挑选，不能再把它当独立证据。"
            elif figure["scales"][0]["centered"]:caption="每列一名被试，每行一个指标；减去各自空间中位数后显示位置间差异。红／蓝只表示高／低于该人的中位水平。"
            else:caption="每列一名被试，每行一个指标；同一行共用色标。每个点对应保留的Hb位置，颜色为原记录等权汇总；各指标使用各自有效支持，不能把非配对差值解释为效果。"
            blocks.append(_figure(str(plots/figure["file"]),caption,str(analysis/summary["files"]["map_values"])))
        for _,candidate in chosen.iterrows():
            part=maps[maps.dataset.eq(dataset)&maps.subject.eq(candidate.subject)&maps.period.eq("subject_mean")]
            d1=part[part.metric.eq("D1")].set_index("location_id");ratio=part[part.metric.eq(HB_RATIO)].set_index("location_id")
            common=d1.index.intersection(ratio.index)
            common=[x for x in common if np.isfinite(d1.loc[x,"display_value"]) and np.isfinite(ratio.loc[x,"display_value"])]
            rho=float(np.corrcoef(rankdata(d1.loc[common,"display_value"]),rankdata(ratio.loc[common,"display_value"]))[0,1]) if len(common)>=3 else np.nan
            finite=d1[np.isfinite(d1.display_value)];top=finite.nlargest(3,"display_value")
            pair=similarities[similarities.dataset.eq(dataset)&similarities.subject.eq(candidate.subject)&similarities.metric.eq("D1")]
            same=pair[pair.same_task & np.isfinite(pair.spearman_rho)];cross=pair[~pair.same_task & np.isfinite(pair.spearman_rho)]
            text=(f"{candidate.subject}：D1位置P10–P90为{_num(finite.display_value.quantile(.1))}–{_num(finite.display_value.quantile(.9))}；"
                +"最高三个投影位置为"+"、".join(f"{r.pair_members}（{r.display_value:.3f}）" for _,r in top.iterrows())+"。"
                +f"D1与相对零预测残差比在{len(common)}个共同位置上的排序相关为{_num(rho)}；若相关较低，两种失配口径强调的位置不同，不能只看标准化绝对误差图。"
                +(f"同任务跨记录D1图相关中位数{_num(same.spearman_rho.median())}，{len(same)}个记录对（有共享记录，非独立样本）。" if len(same) else "没有可估计的同任务跨记录空间重测。")
                +(f"不同任务／刺激记录图相关中位数{_num(cross.spearman_rho.median())}，{len(cross)}个记录对；这是跨条件关联。" if len(cross) else ""))
            observations.append(dict(dataset=dataset,subject=candidate.subject,text=text,D1_ratio_spearman=rho,
                same_task_median_rho=float(same.spearman_rho.median()) if len(same) else None,
                cross_task_median_rho=float(cross.spearman_rho.median()) if len(cross) else None))
            blocks.append(_paragraph(text))
            for figure in [f for f in records if f["dataset"]==dataset and f["subject"]==candidate.subject and f["metric"]=="D1"]:
                blocks.append(_figure(str(plots/figure["file"]),"该被试全部原记录的D1图；第一幅为记录等权图。部分支持仍展示实际可用窗口的结果，灰叉为没有有效数值。其余指标的同构逐记录图见配套index.html。",str(analysis/summary["files"]["map_values"])))
        content["sections"].append(dict(title=DATASET_LABELS[dataset]+"：候选排序、空间形状与逐记录核对",page_break=True,blocks=blocks))
    content["sections"].append(dict(title="这些图能支持什么",page_break=True,blocks=[
        _paragraph("空间模式表明，固定模型对不同测量位置的相容程度并不一致；它可以帮助定位需要检查的通道、任务条件和模型观测假设。形状在记录内重复也可能来自稳定的传感器位置、光学敏感度、尺度和预处理特征，不自动识别脑区生理来源。"),
        _paragraph("本组选择有意偏向明显且较一致的D1图。群体结论必须回到全部候选，而不是把这12人的漂亮图案当作总体比例。D1、相对零预测残差比、弱平衡和跨模态偏离描述不同问题；在一张图中接近模型，不保证其他指标也接近。"),
        _paragraph("Visual仅支持本次审计限定的Hb-only通道描述。其EEG通道身份问题未在本组修复；不能据旧EEG代理绘制或解释局部跨模态生理地形。后续机制分析应先核验输入身份和各位置测量链，再使用未参与选例的记录及独立生理对照。")]))
    content["sources"]=[str(analysis/"summary.json"),str(analysis/summary["files"]["channel_metrics"]),str(analysis/summary["files"]["subject_ranking"]),str(summary["visual_impact_audit"]),"https://mne.tools/1.12/generated/mne.viz.plot_topomap.html"]
    (output/"report_content.json").write_text(json.dumps(content,ensure_ascii=False,indent=2)+"\n")
    checks=render_report(output/"report_content.json",output/"report")
    html=['<!doctype html><meta charset="utf-8"><title>固定Tak通道偏离地形图</title>',
        '<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:#233747;line-height:1.6}img{max-width:100%;height:auto}summary{cursor:pointer;font-weight:600;padding:10px;background:#edf2f6}details{margin:14px 0}a{color:#12648b}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}figure{margin:0}figcaption{font-size:14px}table{border-collapse:collapse}td,th{border:1px solid #cbd6de;padding:7px}</style>',
        '<h1>固定Tak模型：通道偏离与空间模式</h1><p>每库3名探索性选例，全部原记录逐指标保留。黑边有色圆点为保留位置值，灰叉为无有效值，空白为未插值区域。完整数值由原分析目录持有。</p>',
        '<p><a href="report/TECHNICAL_REPORT.pdf">中文报告 PDF</a> · <a href="report/TECHNICAL_REPORT.html">完整正文 HTML</a> · <a href="report/FIGURE_ATLAS.pdf">正文位图图册</a></p>',
        '<p>总体图同一行统一色标；逐记录图在同一被试／指标内统一色标。跨模态指标画在Hb位置，不能当单个EEG电极误差。Visual仅显示Hb-only。早晚一致性参与选择，不是独立验证。</p>']
    for dataset in DATASET_LABELS:
        chosen=selected[selected.dataset.eq(dataset)].sort_values("rank")
        if not len(chosen):continue
        html.append('<h2>'+escape(DATASET_LABELS[dataset])+'</h2>')
        for figure in [f for f in overview if f["dataset"]==dataset and f["kind"]!="layout"]:
            html.append('<figure><a href="plots/'+escape(figure["file"])+'"><img loading="lazy" src="plots/'+escape(figure["file"])+'"></a></figure>')
        for subject in chosen.subject:
            text=next(o["text"] for o in observations if o["dataset"]==dataset and o["subject"]==subject)
            html.append('<h3>'+escape(subject)+'</h3><p>'+escape(text)+'</p>')
            for metric in summary["metrics"]:
                figures=[f for f in records if f["dataset"]==dataset and f["subject"]==subject and f["metric"]==metric]
                if not figures:continue
                html.append('<details'+(' open' if metric=="D1" else '')+'><summary>'+escape(FIXED_METRICS[metric][0])+'：全部原记录</summary>')
                for figure in figures:html.append('<a href="plots/'+escape(figure["file"])+'"><img loading="lazy" src="plots/'+escape(figure["file"])+'"></a>')
                html.append('</details>')
    html.append('<h2>数值与选择来源</h2>')
    for label,path in summary["files"].items():
        html.append('<p><a href="'+escape(os.path.relpath(analysis/path,output))+'">'+escape(label)+'</a></p>')
    (output/"index.html").write_text("\n".join(html)+"\n")
    shutil.copy2(Path(__file__),output/"renderer_source.py")
    manifest=dict(schema="tak_mismatch_spatial_report_v1",source_analysis=str(analysis),owning_summary=str(analysis/"summary.json"),
        visual_impact_audit=summary["visual_impact_audit"],overview_figures=overview,record_figures=records,observations=observations,
        selected_subjects=summary["selected_subjects"],mne_version=mne.__version__,workers=workers,
        interpolation="MNE linear, local; mask to real-node triangles with longest edge <=2.5 median planned nearest-neighbour spacing; missing-node holes",
        numerical_values_clipped=False,model_fits=0,raw_signal_reads=0,protected_reads=0,elapsed_s=time.monotonic()-started,
        command=[str(ROOT/".venv/bin/python"),str(Path(__file__).resolve()),"--run",str(run),"--spatial-analysis",str(analysis),"--output",str(output),"--workers",str(workers)],
        validation=dict(figures=len(overview)+len(records),report_pages=checks["exports"]["TECHNICAL_REPORT.pdf"]["pages"],
            bitmap_figures=checks["bitmap_figures"],body_and_tables_selectable=checks["body_and_tables_selectable"]))
    (output/"source_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    return manifest["validation"]


def priority_panel_tables(run):
    """Read the explicit plotting panel, retaining pending and failed groups."""
    import pandas as pd
    run = Path(run); panel = _json(run/'plot_priority_v1'/'plan.json')
    rows = []; groups = []
    for group in panel['groups']:
        path = run/'current_dof'/'tasks'/f'{group["id"]}.json'
        saved = _json(path) if path.exists() else {}
        current = {r['id']: r for r in saved.get('rows', [])}
        groups.append(dict(group_id=group['id'], dataset=group['dataset'], subject=group['subject'],
            task_label=group['task_label'], record_id=group['record_id'], pair=group['pair'],
            status=saved.get('status', 'pending'), terminal=bool(saved),
            planned_windows=group['planned_windows'], planned_paired_windows=group.get('paired_windows',0)))
        for identity in group['window_ids']:
            fixed = _json(run/'tasks'/f'{identity}.json')
            free = current.get(identity, {})
            row = dict(id=identity, group_id=group['id'], dataset=group['dataset'], subject=group['subject'],
                task_label=group['task_label'], record_id=group['record_id'], pair=group['pair'],
                window=fixed.get('window'), fixed_status=fixed.get('status'),
                current_status=free.get('status', 'pending'), current_terminal=bool(saved),
                fixed_array_path=fixed.get('array_path'), current_array_path=free.get('array_path'),
                current_array_index=free.get('array_index'),
                hb_only_success=free.get('hb_only_success', False), joint_success=free.get('joint_success', False))
            for metric in ('D1', 'D_EH', 'D_obs', 'Delta_spatial', 'D4', 'D_bal_20s'):
                row[metric] = fixed.get(metric)
                row[metric+'_free'] = free.get(metric+'_free')
            for view in ('hb_only', 'joint'):
                parameters = free.get(view+'_parameters', {})
                for name in ('kappa', 'tau'): row[view+'_'+name] = parameters.get(name)
            rows.append(row)
    windows = pd.DataFrame(rows); group_frame = pd.DataFrame(groups)
    cells = []
    keys = ['dataset', 'subject', 'task_label']
    for key, part in windows.groupby(keys, sort=True):
        gg = group_frame
        for field, value in zip(keys, key): gg = gg[gg[field] == value]
        cell = dict(zip(keys, key)); cell.update(planned_groups=len(gg), terminal_groups=int(gg.terminal.sum()),
            planned_windows=len(part), hb_success_windows=int(part.hb_only_success.sum()),
            joint_success_windows=int(part.joint_success.sum()),
            planned_paired_windows=int(gg.planned_paired_windows.sum()))
        for metric in ('D1', 'D_EH', 'D_obs', 'Delta_spatial', 'D4', 'D_bal_20s'):
            values=pd.to_numeric(part[metric],errors='coerce');valid=np.isfinite(values)
            root_metric=metric not in ('Delta_spatial','D_bal_20s')
            data=pd.DataFrame({'group':part.loc[valid,'group_id'],
                'value':np.square(values[valid]) if root_metric else values[valid]})
            mean=float(data.groupby('group').value.mean().mean()) if valid.any() else None
            cell[metric+'_fixed_available_windows']=int(valid.sum())
            cell[metric+'_fixed_all']=float(np.sqrt(mean)) if root_metric and mean is not None else mean
        for metric in ('D1', 'D_EH', 'D_obs', 'D4'):
            a = pd.to_numeric(part[metric], errors='coerce'); b = pd.to_numeric(part[metric+'_free'], errors='coerce')
            common = np.isfinite(a)&np.isfinite(b)
            cell[metric+'_paired_windows'] = int(common.sum())
            for label, values in (('fixed', a), ('free', b)):
                squared = pd.DataFrame({'group': part.loc[common, 'group_id'], 'value': np.square(values[common])})
                cell[metric+'_'+label+'_paired'] = float(np.sqrt(squared.groupby('group').value.mean().mean())) if common.any() else None
            cell[metric+'_free_minus_fixed'] = (cell[metric+'_free_paired']-cell[metric+'_fixed_paired']) if common.any() else None
        cells.append(cell)
    return panel, windows, group_frame, pd.DataFrame(cells)


def render_priority_panel(run, output):
    """Fresh interim plots; never presents a partial panel as full evidence."""
    from datetime import datetime, timezone
    import pandas as pd
    output = Path(output).resolve(); run = Path(run).resolve()
    if output.exists(): raise FileExistsError('plot export must be fresh')
    panel, windows, groups, cells = priority_panel_tables(run)
    output.mkdir(parents=True)
    windows.to_csv(output/'window_metrics.csv', index=False)
    groups.to_csv(output/'group_status.csv', index=False)
    cells.to_csv(output/'subject_task_metrics.csv', index=False)
    plt = _plotting(); figures = output/'figures'; figures.mkdir()
    content = dict(title='Tak：每名被试与任务的绘图优先面板',
        subtitle='阶段快照：固定参数基线、共享参数配对结果及完整缺失分母',
        date=datetime.now(timezone.utc).isoformat(),
        summary=[f"计划 {panel['planned_cells']} 个被试×任务单元、{panel['planned_groups']} 个共享参数组；本快照已有 {int(groups.terminal.sum())} 组终态。",
            '按每条原生记录的登记空间几何选三个分散 Hb 对；保留全部连续支持及组内窗口。REFED逐视频分列，其他数据按原生任务分列。未按误差或收敛结果选择分析面板。',
            '每点是被试×任务单元，窗口和通道不是独立被试。固定/共享参数只在同一窗口两者指标均有效时配对；先平均组内平方误差，再对组等权平均并开方。',
            'D1是Hb-only的HbO/HbR标准化残差RMS；D4是joint的EEG/HbO/HbR等权标准化残差RMS。分母为冻结训练观察SD；前5秒参考，评分5.0–29.8秒，末箱排除。',
            '自由参数D_obs逐窗放松参数方向，与固定参数的投影空间不同，不能以其下降论证组共享模型改善。数值失败和缺失支持均保留；终态数量不等于成功数量。'],
        sections=[], sources=[str(run/'plot_priority_v1'/'plan.json'),str(output/'window_metrics.csv'),str(output/'group_status.csv')])
    overall = []
    for ds, values in panel['by_dataset'].items():
        cc=cells[cells.dataset==ds]; gg=groups[groups.dataset==ds]
        overall.append([DATASET_LABELS.get(ds,ds),values['subjects'],len(cc),len(gg),int(gg.terminal.sum()),
            int((cc.D1_paired_windows>0).sum()),int((cc.D4_paired_windows>0).sum())])
    content['sections'].append(dict(title='覆盖与指标支持',blocks=[_table(
        ['数据集','被试','任务单元','计划组','终态组','D1配对单元','D4配对单元'],overall,
        '终态包括失败。任务单元的指标有效数与完整计划分开。', [2,1,1,1,1,1,1]),
        _paragraph(f"另有 {len(panel['excluded_support_groups'])} 个所选通道组无完整原生Hb支持，{len(panel['excluded_records'])} 条记录没有可安排窗口；身份和原因保留在冻结面板计划中。")]))
    unavailable=cells[cells.planned_paired_windows==0]
    if len(unavailable):
        content['sections'][0]['blocks'].append(_paragraph(
            f"其中{len(unavailable)}个任务单元没有登记允许的EEG/Hb配对时钟支持，只安排Hb-only结果；等待或增加通道不会恢复joint指标。"
            +' '.join(f'{r.dataset}/{r.subject}/{r.task_label}' for r in unavailable.itertuples())+'。'))
    example_manifest=[]
    for ds in panel['by_dataset']:
        cc=cells[cells.dataset==ds].copy(); subjects=sorted(cc.subject.unique()); tasks=sorted(cc.task_label.unique())
        matrix=np.full((len(subjects),len(tasks)),np.nan)
        for _,c in cc.iterrows(): matrix[subjects.index(c.subject),tasks.index(c.task_label)]=100*c.terminal_groups/c.planned_groups
        fig,ax=plt.subplots(figsize=(max(7, len(tasks)*.43),max(5,len(subjects)*.19)))
        im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=100,cmap='YlGnBu')
        ax.set_xticks(range(len(tasks)),[t.replace('emotion_video/','') for t in tasks],rotation=45,ha='right')
        ax.set_yticks(range(len(subjects)),subjects); ax.set_xlabel('原生任务 / REFED视频');ax.set_ylabel('被试')
        for i in range(len(subjects)):
            for j in range(len(tasks)):
                if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',fontsize=6,color='white' if matrix[i,j]>60 else 'black')
        fig.colorbar(im,ax=ax,label='所选组终态比例（%，含失败）');fig.tight_layout()
        path=_save(fig,figures/(ds+'_coverage.png'))
        blocks=[_figure(path,'每格一个被试×任务；数字及颜色为终态比例，固定0–100%刻度；零表示尚无终态，不能解释为误差为零。',str(output/'group_status.csv'))]
        fig,axes=plt.subplots(2,3,figsize=(11,6))
        for ax,metric in zip(axes.flat,('D1','D_bal_20s','D_EH','D_obs','Delta_spatial','D4')):
            values=pd.to_numeric(cc[metric+'_fixed_all'],errors='coerce').dropna().to_numpy()
            if len(values):
                ax.boxplot(values,positions=[0],widths=.35,showfliers=False)
                ax.scatter(np.linspace(-.12,.12,len(values)),np.sort(values),s=9,alpha=.6)
            ax.set_xticks([0],[f'{len(values)}/{len(cc)}任务单元']);ax.set_title(metric)
            ax.set_ylabel('无量纲' if metric=='D_bal_20s' else '训练SD平方' if metric=='Delta_spatial' else '训练SD')
            if metric=='Delta_spatial':ax.axhline(0,color='gray',ls='--',lw=.8)
        fig.tight_layout();distribution=_save(fig,figures/(ds+'_fixed_four_axes.png'))
        blocks.append(_figure(distribution,'固定参数四轴分布：每点一个被试×任务；箱体为四分位区间、中线为中位数、须为1.5倍IQR内极值，所有点均保留。各图独立刻度。D1与弱平衡为第一轴，EEG/Hb差异为第二轴，空间收益为第三轴，D4为第四轴。',str(output/'subject_task_metrics.csv')))
        blocks.append(_paragraph('D_EH是Hb-only读出相对EEG代理的标准化RMS；D_obs是预设噪声情景下的局部可检测驱动差，不是生理参数准确性。20秒弱平衡来自完整原生记录的保留核积分，越低越相容；Delta_spatial为当前近邻减训练预选替代邻域的独立EEG→Hb均方误差，正值有利于替代邻域。四轴不可相加。'))
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,metric in zip(axes,('D1','D4')):
            a=pd.to_numeric(cc[metric+'_fixed_paired'],errors='coerce');b=pd.to_numeric(cc[metric+'_free_paired'],errors='coerce');mask=np.isfinite(a)&np.isfinite(b)
            ax.scatter(a[mask],b[mask],s=16,alpha=.7)
            lim=max(float(a[mask].max()),float(b[mask].max()),.01)*1.05 if mask.any() else 1.
            ax.plot([0,lim],[0,lim],'k--',lw=.8);ax.set_xlim(0,lim);ax.set_ylim(0,lim);ax.set_aspect('equal')
            ax.set_xlabel('固定参数（训练SD）');ax.set_ylabel('共享参数（训练SD）');ax.set_title(metric+f'：{int(mask.sum())}/{len(cc)} 单元有配对')
        fig.tight_layout();scatter=_save(fig,figures/(ds+'_paired.png'))
        blocks.append(_figure(scatter,'每点一个被试×任务，属于同一被试的点不独立。对角线下方表示共享参数误差较低；不同面板独立刻度。此图不提供显著性检验。',str(output/'subject_task_metrics.csv')))
        counts=[]
        for metric in ('D1','D4'):
            delta=pd.to_numeric(cc[metric+'_free_minus_fixed'],errors='coerce').dropna()
            counts.append(f"{metric}可配对单元{len(delta)}/{len(cc)}；其中共享参数较低{int((delta<0).sum())}个"+(f"，单元差值中位数{delta.median():.4g}训练SD。" if len(delta) else '。'))
        blocks.append(_paragraph(' '.join(counts)+'这些是当前成功配对子集的描述，覆盖未完成时不作为总体结论。'))
        content['sections'].append(dict(title=DATASET_LABELS.get(ds,ds)+'：覆盖和配对误差',page_break=True,blocks=blocks))
        # One retained example per native task/video for a compact PDF; every
        # subject/task remains available in the CSV and explicit array paths.
        for task in tasks:
            candidates=windows[(windows.dataset==ds)&(windows.task_label==task)].copy()
            valid=(candidates.hb_only_success&candidates.joint_success&candidates.fixed_array_path.notna()
                &candidates.current_array_path.notna()&candidates.D1.notna()&candidates.D4.notna())
            candidates=candidates[valid].sort_values('id')
            if candidates.empty:continue
            row=candidates.iloc[0];fixed_path=Path(row.fixed_array_path);free_path=Path(row.current_array_path)
            with np.load(fixed_path,allow_pickle=False) as a: arrays={k:a[k].copy() for k in a.files}
            with np.load(free_path,allow_pickle=False) as b:
                index=int(row.current_array_index)
                for view in ('hb_only','joint'):arrays['free_'+view+'_prediction']=b[view+'_prediction'][index].copy()
            t=np.asarray(arrays.get('time_s',np.arange(300)*.1));target=arrays['target'];scale=np.asarray(arrays['sd'])
            fig,axes=plt.subplots(3,1,figsize=(10,6),sharex=True)
            for component,ax in enumerate(axes):
                ax.plot(t,target[:,component]/scale[component],color=COLORS['target'],label='处理后观测')
                for name,label,color,style in (('hb_only_prediction','固定Hb-only','hb','--'),('joint_prediction','固定joint','joint',':'),('free_hb_only_prediction','共享Hb-only','free_hb','-.'),('free_joint_prediction','共享joint','free_joint','-')):
                    if name in arrays:ax.plot(t,arrays[name][:,component]/scale[component],label=label,color=COLORS[color],ls=style,lw=.9)
                ax.axvspan(0,5,color='gray',alpha=.1);ax.set_ylabel(('EEG代理','HbO','HbR')[component]+' / 训练SD')
            axes[0].legend(ncol=3,fontsize=7);axes[-1].set_xlabel('窗口内时间（秒）；灰区为前5秒参考')
            fig.suptitle(f'{DATASET_LABELS.get(ds,ds)} | {row.subject} | {task} | {row.pair}',fontsize=10);fig.tight_layout()
            png=_save(fig,figures/(str(row.id)+'_curve.png'))
            temporal=[]
            for view,label in (('hb_only','Hb-only'),('joint','joint')):
                components=slice(1,3) if view=='hb_only' else slice(0,3)
                prediction=arrays['free_'+view+'_prediction']
                residual=(prediction[:,components]-target[:,components])/scale[components]
                valid=np.asarray(arrays['score_mask'])[:,components].all(axis=1)&np.isfinite(residual).all(axis=1)
                indices=np.flatnonzero(valid)
                if len(indices):
                    squared=np.square(residual[indices]).sum(axis=1);total=float(squared.sum())
                    peak=int(np.argmax(squared));parts=np.array_split(np.arange(len(indices)),3)
                    contributions=[float(squared[p].sum()/total) if total>0 else 0. for p in parts]
                    dominant=int(np.argmax(contributions));segment=indices[parts[dominant]]
                    temporal.append(f"共享{label}残差：最大逐时刻标准化平方和位于{t[indices[peak]]:.1f}秒；评分支持按时间等分三段后，{t[segment[0]]:.1f}–{t[segment[-1]]:.1f}秒段贡献最大，占该视图平方误差{100*contributions[dominant]:.1f}%。")
            evidence=dict(id=row.id,dataset=ds,subject=row.subject,task=task,pair=row.pair,fixed_array=str(fixed_path),current_array=str(free_path),current_index=int(row.current_array_index))
            example_manifest.append(evidence)
            content['sections'].append(dict(title='保留时间曲线：'+task,page_break=True,blocks=[
                _figure(png,'每条曲线是一种观测或完整条件预测。各窗独立拟合驱动和初态；共享参数在整段原生支持的窗口间相同。仅用于描述，非独立预测或生理真值恢复。',str(free_path)),
                _paragraph(f"身份：{row.subject} / {row.record_id} / {row.pair} / {row.window}。每任务选择当前两种共享视图均通过数值检查的最小窗口ID，仅作示例，不用于选择面板。D1：固定{row.D1:.4g}，共享{row.D1_free:.4g}；D4：固定{row.D4:.4g}，共享{row.D4_free:.4g}（训练SD）。"),
                _paragraph(' '.join(temporal)+'误差集中时段用于定位条件拟合偏差，不证明对应时段的生理机制变化。')]))
    (output/'examples.json').write_text(json.dumps(example_manifest,ensure_ascii=False,indent=2)+'\n')
    content_path=output/'report_content.json';content_path.write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n')
    checks=render_report(content_path,output/'report')
    summary=dict(schema='tak_mismatch_plot_snapshot_v1',created_utc=content['date'],planned_groups=len(groups),
        terminal_groups=int(groups.terminal.sum()),planned_cells=len(cells),
        D1_paired_cells=int((cells.D1_paired_windows>0).sum()),D4_paired_cells=int((cells.D4_paired_windows>0).sum()),
        examples=len(example_manifest),full_campaign_completed=False,report=str(output/'report'/'TECHNICAL_REPORT.pdf'))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'delivery_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--priority-panel", action="store_true", help="Read the frozen bounded plotting panel, including pending groups")
    parser.add_argument("--fixed-analysis", type=Path, help="Read an explicitly completed fixed-parameter physiology analysis; current_dof is not opened")
    parser.add_argument("--spatial-analysis", type=Path, help="Read the completed spatial-pattern analysis and render MNE channel maps")
    parser.add_argument("--workers", type=int, default=2, help="Independent subject/metric plot workers in spatial mode")
    parser.add_argument("--revise-report", type=Path, help="Preserve an existing fixed report's numbers and bitmaps while applying the Visual channel impact audit")
    parser.add_argument("--visual-channel-audit", type=Path, help="The single confirmed Tak Visual interpretation-boundary audit")
    args = parser.parse_args(argv)
    if args.spatial_analysis:
        if args.revise_report or args.fixed_analysis or args.priority_panel or args.visual_channel_audit:
            parser.error("--spatial-analysis has its own completed evidence scope")
        print(json.dumps(render_fixed_topography(args.run,args.spatial_analysis,args.output,workers=args.workers),ensure_ascii=False));return
    if args.revise_report:
        if not args.visual_channel_audit or args.fixed_analysis or args.priority_panel:
            parser.error("--revise-report requires --visual-channel-audit and its own evidence scope")
        checks = revise_fixed_visual_report(args.run, args.revise_report, args.visual_channel_audit, args.output)
        print(json.dumps({"output": checks["output"], "pages": checks["exports"]["TECHNICAL_REPORT.pdf"]["pages"]}, ensure_ascii=False)); return
    if args.visual_channel_audit:
        parser.error("--visual-channel-audit requires --revise-report")
    if args.fixed_analysis:
        if args.priority_panel:
            parser.error("--fixed-analysis and --priority-panel describe different evidence scopes")
        checks = render_fixed_analysis(args.run, args.fixed_analysis, args.output)
        print(json.dumps({"output": checks["output"], "pages": checks["exports"]["TECHNICAL_REPORT.pdf"]["pages"]}, ensure_ascii=False)); return
    if args.priority_panel:
        print(json.dumps(render_priority_panel(args.run,args.output),ensure_ascii=False));return
    checks = render_tak_mismatch(args.run, args.output)
    print(json.dumps({"output": checks["output"], "pages": checks["exports"]["TECHNICAL_REPORT.pdf"]["pages"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
