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
Add ``--spatial-style-preview`` for just two old/new display comparisons using
vertex-centred angular projection and full-head display extrapolation.
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


def spatial_topomap(ax, frame, *, limits, title="", cmap="viridis", names=False,
                    coverage="local", annotate_support=True):
    """MNE linear map with local support, or explicit full-head display fill."""
    import mne
    from scipy.spatial import ConvexHull, Delaunay, QhullError
    from scipy.spatial.distance import cdist
    from matplotlib.colors import Normalize
    from matplotlib.patches import Circle, Ellipse
    from matplotlib.cm import ScalarMappable
    if coverage not in ("local", "head"):
        raise ValueError("spatial coverage must be local or head")
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
                image_interp="linear",extrapolate=coverage,sphere=(0.,0.,0.,1.),
                cmap=cmap,vlim=limits,res=512 if coverage == "head" else 160,border="mean")
            field = np.asarray(image.get_array(),float)
            x0,x1,y0,y1 = image.get_extent()
            xx,yy = np.meshgrid(np.linspace(x0,x1,field.shape[1]),np.linspace(y0,y1,field.shape[0]))
            support = xx*xx+yy*yy <= 1.
            if coverage == "local":
                triangulation = Delaunay(xy)
                triangles = xy[triangulation.simplices]
                edge = np.stack([np.linalg.norm(triangles[:,i]-triangles[:,j],axis=1) for i,j in ((0,1),(1,2),(2,0))],axis=1).max(axis=1)
                all_xy = positions[["x","y"]].to_numpy(float)
                distances = cdist(all_xy,all_xy);np.fill_diagonal(distances,np.inf)
                spacing = float(np.median(distances.min(axis=1)))
                simplex = triangulation.find_simplex(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
                support &= simplex >= 0
                support &= edge[np.maximum(simplex,0)] <= 2.5*spacing
                for point in missing[["x","y"]].to_numpy(float):
                    radius = .45*float(np.linalg.norm(xy-point,axis=1).min())
                    support &= (xx-point[0])**2+(yy-point[1])**2 > radius*radius
            else:
                hull = xy[ConvexHull(xy).vertices]
                hull = np.vstack([hull, hull[0]])
                ax.plot(hull[:,0], hull[:,1], color="#344452", ls=(0,(4,3)),
                        lw=.8, alpha=.8, zorder=6)
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
    if annotate_support:
        ax.text(.5,-.035,f"可用位置 {len(good)}/{len(positions)}",transform=ax.transAxes,ha="center",fontsize=7.5,color="#455560")
    if annotate_support and len(good) and "period" in good:
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


def _spatial_preview_coordinates(frame, plan, *, equator_radius=.9):
    """Reproject a retained subject map without changing any scalar or grouping."""
    from src.data.channel_geometry import project_head_topview
    identities = frame[["dataset", "subject"]].drop_duplicates()
    if len(identities) != 1 or not frame.period.eq("subject_mean").all():
        raise ValueError("one retained subject-mean map required for display preview")
    if not 0 < equator_radius <= 1:
        raise ValueError("display equator radius must be in (0, 1]")
    dataset, subject = identities.iloc[0]
    recorded_ids = {identity for value in frame.record_ids for identity in str(value).split(";")}
    points = {}
    found_records = set()
    for record in plan["records"]:
        if record["id"] not in recorded_ids:
            continue
        if record["dataset"] != dataset or str(record["subject"]) != str(subject):
            raise ValueError("retained map and geometry subject identity differ")
        found_records.add(record["id"])
        geometry = record["geometry_identity"]["fnirs"]
        for pair in record["pairs"]:
            oxy, deoxy = [geometry[i] for i in pair["component_indices"]]
            if oxy.get("component") != "HbO" or deoxy.get("component") != "HbR":
                raise ValueError("ordered HbO/HbR geometry required")
            old = project_head_topview(oxy)
            new = project_head_topview(oxy, projection="azimuthal_equidistant") * equator_radius
            for component in (deoxy,):
                if (not np.allclose(old, project_head_topview(component), atol=1e-9, rtol=0)
                        or not np.allclose(new, project_head_topview(component,
                            projection="azimuthal_equidistant") * equator_radius, atol=1e-9, rtol=0)):
                    raise ValueError("paired Hb geometry differs")
            point = np.r_[old, new]
            if pair["id"] in points and not np.allclose(points[pair["id"]], point, atol=1e-9, rtol=0):
                raise ValueError("record-varying geometry cannot reproject a retained subject map")
            points[pair["id"]] = point
    if found_records != recorded_ids:
        raise ValueError("retained map references missing plan records")
    projected = frame.copy()
    projected["x_orthographic"] = projected.x
    projected["y_orthographic"] = projected.y
    for index, row in projected.iterrows():
        members = np.array([points[pair] for pair in row.pair_members.split(";")])
        if not np.allclose(members[:, :2], [row.x, row.y], atol=1e-7, rtol=0):
            raise ValueError("retained map positions differ from the declared frozen geometry")
        if not np.allclose(members[:, 2:], members[0, 2:], atol=1e-9, rtol=0):
            raise ValueError("formerly coincident channels separate; reaggregation required")
        projected.loc[index, ["x", "y"]] = members[0, 2:]
    xy = projected[["x", "y"]].to_numpy(float)
    if not np.isfinite(xy).all() or (np.linalg.norm(xy, axis=1) >= 1).any():
        raise ValueError("preview requires finite registered positions inside the head")
    if len(np.unique(np.round(xy, 9), axis=0)) != len(xy):
        raise ValueError("display projection creates coincident locations")
    return projected


def render_spatial_style_preview(run, analysis, output):
    """Two display-only tests, preserving retained values, cases and old exports."""
    import shlex
    from datetime import datetime, timezone
    import pandas as pd
    from matplotlib import patheffects

    run, analysis, output = (Path(p).resolve() for p in (run, analysis, output))
    summary = _json(analysis / "summary.json")
    if (summary.get("schema") != "tak_mismatch_spatial_patterns_v1"
            or summary.get("status") != "completed"
            or Path(summary["parent_run"]).resolve() != run or not analysis.is_relative_to(run)):
        raise ValueError("completed spatial analysis inside the declared retained run required")
    if output.exists():
        raise FileExistsError("spatial preview requires a fresh export")
    cases = (("eeg_fnirs_single_trial", "subject_06", "Single-Trial · 06号"),
             ("simultaneous_eeg_nirs", "VP022", "Simultaneous · VP022"))
    selected = {(row["dataset"], row["subject"]) for row in summary["selected_subjects"]}
    if not all((dataset, subject) in selected for dataset, subject, _ in cases):
        raise ValueError("preview cases must belong to the retained illustrative selection")
    source = _path(analysis, summary["files"]["map_values"])
    maps = pd.read_csv(source, dtype={"subject": str})
    plan = _json(run / "plan.json")
    prepared = []
    for dataset, subject, title in cases:
        original = maps[maps.dataset.eq(dataset) & maps.subject.eq(subject)
                        & maps.metric.eq("D1") & maps.period.eq("subject_mean")].copy()
        if original.empty:
            raise ValueError("retained D1 subject map is missing")
        current = _spatial_preview_coordinates(original, plan)
        current = current.assign(_label_y=current.y.round(4)).sort_values(
            ["_label_y", "x"], ascending=[False, True]).drop(columns="_label_y")
        current["display_number"] = np.arange(1, len(current) + 1)
        if not np.array_equal(current.display_value, original.loc[current.index].display_value, equal_nan=True):
            raise ValueError("display preview changed retained values")
        prepared.append((dataset, subject, title, original, current))
    maximum = max(float(item[4].display_value.max()) for item in prepared)
    limits = (0., math.ceil(maximum * 10) / 10)
    plt = _plotting()
    output.mkdir(parents=True)
    figures = []
    for number, (dataset, subject, title, original, current) in enumerate(prepared, 1):
        fig = plt.figure(figsize=(13.2, 9.8))
        left = fig.add_axes([.025, .36, .415, .51])
        right = fig.add_axes([.475, .36, .415, .51])
        spatial_topomap(left, original, limits=limits, title="原图：x-y 正投影／局部着色")
        im = spatial_topomap(right, current, limits=limits, coverage="head",
                            title="测试图：球面等距展开／整头着色")
        for _, row in current.iterrows():
            right.annotate(str(int(row.display_number)), (row.x, row.y), xytext=(0, 6),
                textcoords="offset points", ha="center", va="bottom", fontsize=8,
                color="#172c3c", zorder=9,
                path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])
        for ax in (left, right):
            ax.text(0, 1.17, "前", ha="center", va="bottom", fontsize=10)
            ax.text(-1.13, -.15, "左", ha="right", fontsize=10)
            ax.text(1.13, -.15, "右", ha="left", fontsize=10)
        bar = fig.colorbar(im, cax=fig.add_axes([.935, .405, .018, .36]))
        bar.set_label("Hb 重建失配 D1 / 冻结训练 SD", fontsize=10)
        bar.ax.tick_params(labelsize=9)
        fig.suptitle(f"坐标与着色验收 {number}/2　{title}", fontsize=18, y=.976)
        fig.text(.5, .933, "同一份被试记录等权结果；两图、两个样例共用色标。数值越高，条件重建偏离越大。",
                 ha="center", fontsize=11, color="#42536a")
        fig.text(.5, .265, "黑边圆点：Hb 测量位置　·　灰叉：无有效值　·　右图虚线：测点凸包，线外颜色为外推",
                 ha="center", fontsize=10)
        entries = [f"{int(row.display_number):2d}  {row.pair_members}" for _, row in current.iterrows()]
        columns = 4
        per_column = math.ceil(len(entries) / columns)
        for column in range(columns):
            fig.text(.065 + .235 * column, .229,
                     "\n".join(entries[column * per_column:(column + 1) * per_column]),
                     va="top", fontsize=9, linespacing=1.35, color="#263b4f")
        fig.text(.5, .052, "右图按头顶到通道的球面角度展开，赤道显示半径为头部轮廓的90%；保留左右与前后方向。",
                 ha="center", fontsize=10)
        fig.text(.5, .025, "整头颜色用于显示插值与外推，不表示全脑均有测量；本次仅测试画法，不更新空间统计或正式报告。",
                 ha="center", fontsize=10, color="#42536a")
        filename = f"preview_{number:02d}_{dataset}_{subject}.png"
        _save(fig, output / filename)
        current.to_csv(output / (Path(filename).stem + "_coordinates.csv"), index=False)
        front = current.y_orthographic > .8
        old_span = float(np.ptp(current.loc[front, "y_orthographic"]))
        new_span = float(np.ptp(current.loc[front, "y"]))
        figures.append(dict(file=filename, dataset=dataset, subject=subject, metric="D1",
            points=len(current), available_points=int(current.display_value.notna().sum()),
            available_records=[int(current.available_records.min()), int(current.available_records.max())],
            planned_records=int(current.planned_records.max()), limits=list(limits),
            retained_values_unchanged=True, frontal_rule="retained orthographic y > 0.8",
            frontal_points=int(front.sum()), frontal_y_span_before=old_span,
            frontal_y_span_after=new_span, frontal_y_span_ratio=new_span / old_span))
    command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), "--run", str(run),
               "--spatial-analysis", str(analysis), "--spatial-style-preview", "--output", str(output)]
    manifest = dict(schema="tak_mismatch_spatial_style_preview_v1",
        created_utc=datetime.now(timezone.utc).isoformat(), sources=[str(source), str(run / "plan.json")],
        command=shlex.join(command), working_directory=str(ROOT), renderer=str(Path(__file__).resolve()),
        projection="vertex-centred azimuthal equidistant: r=0.9*atan2(hypot(x,y),z)/(pi/2)",
        frame="right +x, anterior +y; registered template origin; display transformation only",
        interpolation="MNE linear; extrapolate=head; border=mean; clipped to unit head circle",
        documentation="https://mne.tools/1.12/generated/mne.viz.plot_topomap.html",
        selection="two previously selected frontal-pattern illustrations, no new ranking",
        figures=figures, model_fits=0, raw_signal_reads=0, spatial_statistics_recomputed=False,
        previous_exports_preserved=True)
    (output / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


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


def _mechanism_pattern_figures(tables, maps, plan, similarities, shapes, ocular, output):
    """Figures use retained/derived tables; no signal arrays or model fits."""
    import pandas as pd
    from matplotlib.lines import Line2D
    from src.analysis.tak_mismatch import HB_RATIO
    plt = _plotting()
    output = Path(output);output.mkdir()
    figures = {}
    datasets = ("eeg_fnirs_single_trial", "simultaneous_eeg_nirs")
    region_names = ("frontal", "central", "posterior")
    region_labels = ("前额", "中央", "后部")
    region_colors = ("#0072B2", "#009E73", "#D55E00")
    main_cases = ((datasets[0], "subject_25", "Single-Trial 25 · 左中央"),
                  (datasets[0], "subject_06", "Single-Trial 06 · 前额"),
                  (datasets[1], "VP023", "Simultaneous VP023 · 后部"))

    def save(key, fig, caption, source):
        path = _save(fig, output/(key+".png"))
        figures[key] = dict(path=path, caption=caption, source=source, preserve_bitmap=True)

    coordinate_cache = {}
    def display(frame, dataset, subject):
        identity = (dataset, subject)
        if identity not in coordinate_cache:
            reference = maps[maps.dataset.eq(dataset) & maps.subject.eq(subject)
                & maps.period.eq("subject_mean") & maps.metric.eq("D1")]
            coordinate_cache[identity] = _spatial_preview_coordinates(reference, plan).set_index("pair_members")
        coordinates = coordinate_cache[identity].loc[frame.pair_members]
        if not np.allclose(frame[["x","y"]],coordinates[["x_orthographic","y_orthographic"]],atol=1e-7):
            raise ValueError("display template differs from retained map positions")
        result = frame.copy();result[["x","y"]] = coordinates[["x","y"]].to_numpy()
        return result

    # Paired people, rather than pooled channels/windows, show population support.
    fig, axes = plt.subplots(2,2,figsize=(8.7,6.6),squeeze=False)
    regional = tables['regional_values']
    for row, metric in enumerate((HB_RATIO,"D_bal_20s")):
        all_values = regional[regional.metric.eq(metric)&regional.period.eq('subject_mean')].value
        for col,dataset in enumerate(datasets):
            ax = axes[row,col]
            part = regional[regional.dataset.eq(dataset)&regional.metric.eq(metric)&regional.period.eq('subject_mean')]
            wide = part.pivot(index='subject',columns='region',values='value').reindex(columns=region_names)
            complete = wide.dropna()
            for j,(_,values) in enumerate(complete.iterrows()):
                jitter = ((j%7)-3)*.018
                ax.plot(np.arange(3)+jitter,values.to_numpy(),color='#abb4bd',lw=.55,alpha=.6,zorder=1)
                for i,marker in enumerate(('o','s','^')):
                    ax.scatter(i+jitter,values.iloc[i],s=14,marker=marker,color=region_colors[i],alpha=.65,zorder=2)
            for i,name in enumerate(region_names):
                q = complete[name].quantile([.25,.5,.75]).to_numpy()
                ax.plot([i,i],[q[0],q[2]],color='black',lw=2.2,zorder=4)
                ax.scatter(i,q[1],s=40,facecolor='white',edgecolor='black',zorder=5)
            ax.set(xticks=range(3),xticklabels=region_labels,xlim=(-.3,2.3),
                   ylim=(0,max(float(all_values.max())*1.08,.1)))
            ax.set_title(f"{DATASET_LABELS[dataset]} · {len(complete)}/{len(wide)}名被试",fontsize=10)
            ax.set_ylabel("Hb残差／零预测MSE" if row==0 else "20秒弱平衡偏离（无量纲）")
            ax.grid(axis='y',alpha=.18)
    fig.suptitle("全被试区域比较：每条细线连接同一名被试",fontsize=14)
    fig.tight_layout(rect=[0,.06,1,.94])
    fig.text(.5,.012,"白心点与黑竖线为中位数和四分位距；同一行共用纵轴。区域按测量位置粗分，不是皮层定位。",ha='center',fontsize=9)
    save('01_subject_regions',fig,"每条线是一名被试的三个区域值。先用既有记录等权通道图，再取区域内通道中位数。只有三个区域的全部计划位置均有值者进入配对；Single-Trial保留28/29名，Simultaneous为26/26名。区域及被试等权，不把通道或记录当独立被试。",'regional_values.csv')

    def map_grid(key, cases, metrics, population=False):
        two_rows=len(metrics)==2
        fig,axes = plt.subplots(len(metrics),len(cases),figsize=(9,7.0 if two_rows else 2.7*len(metrics)+.7),squeeze=False)
        fig.subplots_adjust(left=.025,right=.87,bottom=.11 if two_rows else .08,
                            top=.86 if two_rows else .91,wspace=.15,hspace=.45 if two_rows else .4)
        anchors={(single,subject):pair for single,subject,pair in (
            ('eeg_fnirs_single_trial','subject_25','C1FC1'),('eeg_fnirs_single_trial','subject_06','FpzFp1'),
            ('simultaneous_eeg_nirs','VP023','P4h'),('eeg_fnirs_single_trial','subject_28','OzPOz'),
            ('simultaneous_eeg_nirs','VP022','AF8'),('refed','30','CH27'))}
        for row,metric in enumerate(metrics):
            panels=[]
            for dataset,subject,title in cases:
                source = tables['population_maps'] if population else maps
                take = source.dataset.eq(dataset)&source.metric.eq(metric)
                if not population:take &= source.subject.eq(subject)&source.period.eq('subject_mean')
                panels.append(display(source[take].copy(),dataset,subject))
            maximum = max(float(p.display_value.max()) for p in panels)
            maximum = math.ceil(maximum*10)/10
            for col,(panel,(_,_,title)) in enumerate(zip(panels,cases)):
                label = 'Hb残差比' if metric==HB_RATIO else metric
                if population:
                    lo,hi=int(panel.available_subjects.min()),int(panel.available_subjects.max())
                    title=title.split(' · ')[0]+f" · 每点{lo if lo==hi else str(lo)+'–'+str(hi)}人"
                image = spatial_topomap(axes[row,col],panel,limits=(0,maximum),coverage='head',title=title+'\n'+label)
                for text in axes[row,col].texts:
                    if text.get_fontsize()<9:text.set_fontsize(9)
                if not population:
                    dataset,subject,_=cases[col];pair=anchors[(dataset,subject)]
                    point=panel[panel.pair_members.eq(pair)].iloc[0]
                    axes[row,col].scatter(point.x,point.y,s=55,facecolor='none',edgecolor='white',lw=1.1,zorder=8)
                    axes[row,col].annotate(pair,(point.x,point.y),xytext=(.03,.85 if point.y>=0 else .12),
                        textcoords='axes fraction',fontsize=10,color='#233747',zorder=9,
                        bbox=dict(boxstyle='round,pad=.15',fc='white',ec='none',alpha=.88),
                        arrowprops=dict(arrowstyle='-',color='#45596b',lw=.7))
            y = (axes[row,0].get_position().y0+axes[row,0].get_position().y1)/2
            bar = fig.colorbar(image,cax=fig.add_axes([.915,y-.085,.018,.17]))
            bar.set_label(FIXED_METRICS[metric][1],fontsize=8)
        fig.suptitle("被试中位数地形图" if population else "典型个体：同一位置在不同指标中的表现",fontsize=14,y=.985)
        fig.text(.46,.025,"行内共用色标；黑边点为通道值，灰叉为缺失。虚线外为外推，点间颜色为插值。",ha='center',fontsize=9)
        save(key,fig,("每点先在被试内按原合同汇总，再取该位置的被试中位数；它不是某名被试的地图。" if population else
            "每列为一名已选病例，每行一个指标；通道值为原有记录等权汇总，病例不是独立确认集。")+
            "采用从头顶按球面角度展开的显示坐标；整头填色不增加测量覆盖。行间单位和色标不同，不能把颜色强度跨指标比较。"+
            ("白圈与名称标出同一代表通道，便于跨指标对照；其余黑边点仍保留全部计划位置。" if not population else ""),
            'population_maps.csv' if population else 'spatial map_values.csv.gz')
    map_grid('02_population_maps',[(datasets[0],'subject_25','Single-Trial · 29名'),
                                 (datasets[1],'VP023','Simultaneous · 26名')],(HB_RATIO,'D_bal_20s','D1'),True)

    contrasts = tables['regional_contrasts']
    fig,axes = plt.subplots(1,2,figsize=(8.8,4.1))
    order = ('nback','dsr','wg')
    for ax,metric in zip(axes,(HB_RATIO,'D_bal_20s')):
        part = contrasts[contrasts.dataset.eq(datasets[1])&contrasts.level.eq('task')
            &contrasts.metric.eq(metric)&contrasts.contrast.eq('posterior_minus_central')]
        wide = part.pivot(index='subject',columns='task_label',values='value').reindex(columns=order)
        for _,values in wide.iterrows():ax.plot(range(3),values,marker='o',ms=2.5,lw=.6,color='#9aa8b5',alpha=.6)
        for i,task in enumerate(order):
            values=wide[task].dropna();q=values.quantile([.25,.5,.75]).to_numpy()
            ax.plot([i,i],[q[0],q[2]],color='#0072B2',lw=3)
            ax.scatter(i,q[1],c='#0072B2',s=35,zorder=5)
            ax.text(i,.96,f"后部较高 {int((values>0).sum())}/{len(wide)}",ha=('left','center','right')[i],va='top',transform=ax.get_xaxis_transform(),fontsize=8)
        ax.axhline(0,color='black',ls='--',lw=.8)
        ax.set(xticks=range(3),xticklabels=('n-back','DSR','词生成'),ylabel='后部 − 中央')
        ax.set_title('Hb残差／零预测MSE比' if metric==HB_RATIO else '20秒弱平衡偏离')
        ax.grid(axis='y',alpha=.15);ax.margins(y=.22)
    fig.suptitle('Simultaneous：后部偏离在三个任务中保持',fontsize=14)
    fig.tight_layout(rect=[0,.08,1,.92])
    fig.text(.5,.015,'每条灰线连接同一被试；蓝点／竖线为中位数／四分位距。连线不代表生理时间。',ha='center',fontsize=9)
    save('03_task_consistency',fig,'每个点是一名被试的区域差值，正值表示后部高于中央。三个任务各有26名被试。任务采用固定顺序，任务效应与时间／疲劳仍混杂；本图只描述空间方向是否保持，没有新增检验。','regional_contrasts.csv')
    map_grid('04_case_maps',main_cases,('D1',HB_RATIO,'D_bal_20s'))
    map_grid('05_amplitude_counterexamples',[(datasets[0],'subject_28','Single-Trial 28'),
        (datasets[1],'VP022','Simultaneous VP022'),('refed','30','REFED 30')],('D1',HB_RATIO))

    fig,axes=plt.subplots(2,3,figsize=(9.2,6.8),squeeze=False)
    for col,(dataset,subject,title) in enumerate(main_cases):
        for row,metric in enumerate(('D1',HB_RATIO)):
            ax=axes[row,col];part=similarities[similarities.dataset.eq(dataset)&similarities.subject.eq(subject)&similarities.metric.eq(metric)]
            identities=sorted(set(part.record_a)|set(part.record_b));matrix=np.eye(len(identities))
            matrix[matrix==0]=np.nan
            for item in part.itertuples():
                i,j=identities.index(item.record_a),identities.index(item.record_b);matrix[i,j]=matrix[j,i]=item.spearman_rho
            image=ax.imshow(matrix,cmap='RdBu_r',vmin=-1,vmax=1)
            labels=[v.split('|')[-1].replace('session_','S').replace('cnt_','') for v in identities]
            ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels)
            ax.tick_params(labelsize=9);plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
            for i in range(len(identities)):
                for j in range(len(identities)):
                    if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if abs(matrix[i,j])>.6 else 'black')
            ax.set_title(title+('\nD1' if metric=='D1' else '\nHb残差比'),fontsize=9)
    fig.subplots_adjust(left=.07,right=.89,bottom=.12,top=.87,hspace=.6,wspace=.4)
    fig.colorbar(image,cax=fig.add_axes([.93,.25,.015,.5])).set_label('共同通道的 Spearman ρ')
    fig.suptitle('跨记录空间形状：绝对误差与相对误差的稳定性不同',fontsize=13)
    fig.text(.48,.015,'S00/02/04为运动想象，S01/03/05为心算；Simultaneous三条记录对应三个不同任务。',ha='center',fontsize=8.5)
    save('06_record_stability',fig,'每个格子是两条原生记录在共同有效通道上的空间排序相关；对角线是自身。记录对会共享记录，不能作为独立样本计数。病例按既有D1空间反差、邻近一致性和记录内早晚相关选择，当前高相关属于探索性证据。','record_similarities.csv')

    parameters=tables['parameter_groups'];pair_fits=tables['fit_pair_groups']
    parameter_cases=((datasets[1],'VP022','VP022'),(datasets[1],'VP023','VP023'),(datasets[0],'subject_06','Single-Trial 06'))
    fig,axes=plt.subplots(3,3,figsize=(9,8.5),squeeze=False)
    for row,(dataset,subject,title) in enumerate(parameter_cases):
        pairs=('AFpz','C5h','POOz') if dataset==datasets[1] else ('FpzFp2','C6CP6','OzO1')
        for col,pair in enumerate(pairs):
            ax=axes[row,col];part=parameters[parameters.dataset.eq(dataset)&parameters.subject.eq(subject)&parameters.pair.eq(pair)]
            ids=sorted(part.group_id.unique());counts=[]
            for i,gid in enumerate(ids):
                group=part[part.group_id.eq(gid)].set_index('view');jitter=(i-(len(ids)-1)/2)*.015
                if group.valid_estimate.all():ax.plot([jitter,1+jitter],group.loc[['hb_only','joint'],'tau'],color='#a9b1b9',lw=.7,zorder=1)
                for j,view in enumerate(('hb_only','joint')):
                    item=group.loc[view]
                    if np.isfinite(item.tau):ax.scatter(j+jitter,item.tau,marker=('o' if j==0 else '^') if item.valid_estimate else 'x',
                        s=22,color=('#246eab' if j==0 else '#c77722') if item.valid_estimate else '#777777',zorder=4)
            for view in ('hb_only','joint'):counts.append(f"{int(part[part.view.eq(view)].valid_estimate.sum())}/{len(ids)}")
            ax.axhline(2,color='#37936b',ls='--',lw=.8);ax.axhline(12,color='#8d3c4a',ls=':',lw=1)
            ax.set(xticks=[0,1],xticklabels=['Hb-only','联合'],ylim=(0,13.2),xlim=(-.3,1.3))
            ax.set_title(f'{title} · {pair}\n有效组：Hb {counts[0]}，联合 {counts[1]}',fontsize=9)
            if col==0:ax.set_ylabel('τ（秒）')
            ax.grid(axis='y',alpha=.15)
    fig.suptitle('时间常数取决于位置、输入条件与数值可用性',fontsize=14)
    fig.tight_layout(rect=[0,.09,1,.94])
    fig.text(.5,.043,'每点一个共享参数组；圆点为Hb-only，三角为联合拟合，灰叉为未通过检查的末次值。',ha='center',fontsize=9)
    fig.text(.5,.015,'连线仅连接同组两种均通过的解；绿虚线为固定τ=2秒，紫点线为工程上界12秒。',ha='center',fontsize=9)
    save('07_tau_and_failures',fig,'每组为一条原生记录中的一个连续Hb片段和一个通道，参数在组内窗口共享。有效表示至少一个窗口通过原有收敛与积分检查，完整窗口覆盖另存表中。两视图各用自身允许的输入窗口，同组连线不保证拟合时间支持相同；参数差异反映观察条件与支持的联合敏感性。灰叉保留失败末次参数，不能作为生理估计；达到12秒上界也不是正常生理参考。','parameter_groups.csv')

    fig,axes=plt.subplots(1,3,figsize=(9.2,4.2),squeeze=False)
    selected_fits=pair_fits[pair_fits.metric.eq('D1') & pair_fits.subject.isin([c[1] for c in parameter_cases])]
    fit_upper=math.ceil(float(selected_fits[['fixed','free']].max().max())*11)/10
    for ax,(dataset,subject,title) in zip(axes[0],parameter_cases):
        pairs=('AFpz','C5h','POOz') if dataset==datasets[1] else ('FpzFp2','C6CP6','OzO1')
        part=pair_fits[pair_fits.dataset.eq(dataset)&pair_fits.subject.eq(subject)&pair_fits.metric.eq('D1')]
        for j,pair in enumerate(pairs):
            group=part[part.pair.eq(pair)];good=group[group.paired_windows>0].sort_values('group_id')
            for i,item in enumerate(good.itertuples()):
                jitter=(i-(len(good)-1)/2)*.015
                ax.plot([j-.12+jitter,j+.12+jitter],[item.fixed,item.free],color='#9da7b1',lw=.7)
                ax.scatter(j-.12+jitter,item.fixed,color='#246eab',marker='o',s=18)
                ax.scatter(j+.12+jitter,item.free,color='#c77722',marker='^',s=20)
            ax.text(j,.98,f'{len(good)}/{len(group)}组',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
        ax.set(xticks=range(3),xticklabels=pairs,ylim=(0,fit_upper),title=title)
        ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.15)
    axes[0,0].set_ylabel('同窗配对 D1 / 冻结训练SD')
    fig.suptitle('参数适配改变哪些位置：保留相同成功窗口再比较',fontsize=13)
    fig.legend(handles=[Line2D([],[],color='#246eab',marker='o',ls='',label='固定参数'),Line2D([],[],color='#c77722',marker='^',ls='',label='共享κ、τ')],loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=[0,.09,1,.91])
    save('08_same_window_adaptation',fig,'每条短线是一组在完全相同窗口支持上的固定与自由参数RMS；组内先平均窗口平方误差再开方。未成功组进入上方计划分母，但不填零。点数不等于被试数。本图定位参数可以吸收的失配，不按误差最小给方法作生理资格排名。','fit_pair_groups.csv')

    fig,axes=plt.subplots(1,3,figsize=(9.1,4))
    ds_order=(*datasets,'refed','visual_cognitive_motivation');ds_labels=('Single\nTrial','Simul.','REFED','Visual\n仅Hb')
    features=(('Hb_common_direction_fraction','标准化共同方向平方比'),('Hb_bias_fraction','偏置平方比'),('Hb_residual_OR_correlation','HbO/HbR残差相关'))
    for ax,(feature,label) in zip(axes,features):
        for i,dataset in enumerate(ds_order):
            values=shapes[shapes.dataset.eq(dataset)&shapes.feature.eq(feature)].value.dropna().to_numpy()
            jitter=((np.arange(len(values))%9)-4)*.018
            ax.scatter(i+jitter,values,s=13,alpha=.55,c='#246eab' if i<3 else '#777777')
            q=np.quantile(values,[.25,.5,.75]);ax.plot([i,i],[q[0],q[2]],color='black',lw=2)
            ax.scatter(i,q[1],s=30,facecolor='white',edgecolor='black',zorder=5)
        ax.set(xticks=range(4),xticklabels=ds_labels,title=label,ylim=(0,1.05));ax.grid(axis='y',alpha=.15)
    fig.suptitle('残差的方向结构与静态偏置分开观察',fontsize=14)
    fig.tight_layout(rect=[0,.07,1,.92])
    fig.text(.5,.01,'每点一名被试；白心点／黑线为中位数／四分位距。共同方向比不是血容量或生理来源份额。',ha='center',fontsize=8.5)
    save('09_residual_directions',fig,'这是预定三通道面板的被试汇总，人数依次29、26、32、16；计划38268个窗口通道对中37577项具有有效Hb形态。双Hb残差在各自训练SD坐标中分解，前两项为平方量比值，第三项为相关；不是三份独立证据，也不能定位到全通道热点。Visual仅保留纯Hb解释。','shape_subjects.csv')

    fig,axes=plt.subplots(1,2,figsize=(8.5,3.5))
    for ax,contrast,label,limits in zip(axes,('clean_minus_raw','clean_minus_amp'),
            ('EOG回归 − 原代理','EOG回归 − 幅度匹配对照'),((-.55,.05),(-.025,.025))):
        part=ocular[ocular.analysis.eq('counterfactual')&ocular.contrast.eq(contrast)]
        if part.empty and contrast=='clean_minus_amp':part=ocular[ocular.analysis.eq('counterfactual')&ocular.contrast.eq('clean_minus_amplitude')]
        for i,dataset in enumerate(datasets):
            item=part[part.dataset.eq(dataset)].iloc[0]
            ax.errorbar(item['mean'],i,xerr=[[item['mean']-item.ci_low],[item.ci_high-item['mean']]],fmt='o',color='#246eab',capsize=3)
        ax.axvline(0,color='black',ls='--',lw=.8)
        ax.set(xlim=limits,yticks=[0,1],yticklabels=['Single-Trial','Simultaneous'],ylim=(-.6,1.6),
            title=label,xlabel='EEG目标对固定Hb读出的MSE差')
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('EOG参照：未检出超出幅度对照的额外变化',fontsize=13)
    fig.tight_layout(rect=[0,.08,1,.9]);fig.text(.5,.01,'两面板横轴独立缩放；单位为EEG训练SD平方，负值表示目标与保留读出更接近。',ha='center',fontsize=8.5)
    save('10_eog_control',fig,'点为既有被试等权均值，横线为整被试重采样95%区间。Single-Trial为29/29名、Simultaneous为26/26名；计划23952个窗口通道对中19949项可评分。左图相对原代理下降，右图相对只匹配均值／SD的对照区间跨零。处理改变了EEG评分目标，Hb模型未重拟合；这不是眼源贡献比例，也不证明清理恢复了神经真值。','ocular_associations.csv')
    return figures


def render_mechanism_patterns(run, analysis, output):
    """Mechanism-oriented synthesis, keeping its numerical owner separate."""
    import shutil
    import pandas as pd
    from datetime import datetime, timezone
    from src.analysis.tak_mismatch import HB_RATIO
    run,analysis,output=(Path(p).resolve() for p in (run,analysis,output))
    summary=_json(analysis/'summary.json')
    if (summary.get('schema')!='tak_mismatch_mechanism_patterns_v1' or summary.get('status')!='completed'
            or Path(summary['parent_run']).resolve()!=run or not analysis.is_relative_to(run)):
        raise ValueError('completed mechanism-pattern analysis in the declared retained run required')
    if output.exists():raise FileExistsError('mechanism report requires a fresh export')
    tables={name:pd.read_csv(_path(analysis,path),dtype={'subject':str}) for name,path in summary['files'].items()}
    sources=summary['sources']
    maps=pd.read_csv(sources['maps'],dtype={'subject':str})
    similarities=pd.read_csv(sources['similarities'],dtype={'subject':str})
    shapes=pd.read_csv(sources['shape_subjects'],dtype={'subject':str})
    ocular=pd.read_csv(sources['ocular_associations'])
    output.mkdir(parents=True)
    figures=_mechanism_pattern_figures(tables,maps,_json(Path(sources['plan'])),similarities,shapes,ocular,output/'plots')
    def figure(key):return dict(type='figure',**figures[key])
    reg=tables['regional_summary'];contrasts=tables['regional_contrast_summary']
    sim='simultaneous_eeg_nirs';single='eeg_fnirs_single_trial'
    rows=[]
    for metric,label in ((HB_RATIO,'Hb残差／零预测MSE比'),('D_bal_20s','20秒弱动态平衡偏离'),('D1','D1（训练SD）')):
        part=reg[reg.dataset.eq(sim)&reg.level.eq('subject')&reg.metric.eq(metric)].set_index('region')
        c=contrasts[contrasts.dataset.eq(sim)&contrasts.level.eq('subject')&contrasts.metric.eq(metric)&contrasts.contrast.eq('posterior_minus_central')].iloc[0]
        rows.append([label,*[f"{part.loc[r,'median']:.3f}" for r in ('frontal','central','posterior')],
                     f'{int(c.positive_subjects)}/{int(c.planned_subjects)}',f"{c['median']:+.4f}"])
    scale_rows=[]
    for dataset in (single,sim):
        for metric in (HB_RATIO,'D_bal_10s','D_bal_20s','D_bal_40s'):
            r=contrasts[contrasts.dataset.eq(dataset)&contrasts.level.eq('subject')&contrasts.metric.eq(metric)&contrasts.contrast.eq('posterior_minus_central')].iloc[0]
            scale_rows.append([DATASET_LABELS[dataset], 'Hb残差比' if metric==HB_RATIO else metric,
                f'{int(r.available_subjects)}/{int(r.planned_subjects)}',f'{int(r.positive_subjects)}/{int(r.available_subjects)}',f"{r['median']:+.4f}"])
    tau_rows=[]
    site=tables['parameter_site_summary']
    for dataset,subject,pair in ((sim,'VP022','AFpz'),(sim,'VP022','C5h'),(sim,'VP022','POOz'),
                               (sim,'VP023','POOz'),(single,'subject_06','FpzFp2')):
        r=site[site.dataset.eq(dataset)&site.subject.eq(subject)&site.pair.eq(pair)&site.view.eq('hb_only')].iloc[0]
        tau_rows.append([subject,pair,f'{int(r.valid_groups)}/{int(r.planned_groups)}',f'{r.tau_median:.2f}',
            f'{int(r.boundary_valid_groups)}/{int(r.valid_groups)}',f'{int(r.valid_windows)}/{int(r.planned_windows)}'])
    same_rows=[]
    for dataset,subject in ((single,'subject_25'),(single,'subject_06'),(sim,'VP023')):
        for metric in ('D1',HB_RATIO):
            s=similarities[similarities.dataset.eq(dataset)&similarities.subject.eq(subject)&similarities.metric.eq(metric)]
            a=s[s.same_task].spearman_rho.dropna();b=s[~s.same_task].spearman_rho.dropna()
            same_rows.append([subject,'D1' if metric=='D1' else 'Hb残差比',
                f'{a.median():.3f}（{len(a)}对）' if len(a) else '无同任务重复',f'{b.median():.3f}（{len(b)}对）'])
    shape_rows=[]
    for dataset in (single,sim,'refed','visual_cognitive_motivation'):
        p=shapes[shapes.dataset.eq(dataset)]
        shape_rows.append([DATASET_LABELS[dataset],str(p.subject.nunique()),
            *[f'{p[p.feature.eq(feature)].value.median():.3f}' for feature in ('Hb_common_direction_fraction','Hb_residual_OR_correlation','Hb_bias_fraction')]])
    content=dict(schema='source_diagnostics_report_v1',title='Tak失配的空间模式与生理机制线索',
        subtitle='统一参照、个体稳定性与参数适配：基于保留证据的机制分析',
        date='2026-11-04；拟合证据：2026-10-28至2026-10-29；本次仅新增描述性汇总与显示',
        provenance=['固定全通道图、预定三通道共享参数面板和EOG辅助分析各自保留原支持。',
            '新汇总数值由机制分析目录持有；本报告只呈现这些表及其来源。未读取原始信号、未重拟合、未新增显著性检验。'],
        summary=['最突出的人群模式是Simultaneous后部测量位置的相对Hb残差高于中央：23/26名被试同向；20秒弱动态平衡偏离为22/26。该方向在三个任务中保持，但未据此定位皮层来源。',
            '个体空间模式比单次误差高低更有机制价值。Single-Trial 25号左中央、06号前额及Simultaneous VP023后部的模式可重复；绝对误差与相对误差仍需分开观察。',
            '自由参数揭示不同位置对模型时间尺度的调整需求。VP022前额Hb-only τ中位数9.61秒，中央与后部约3.16、3.33秒；VP023后部7个有效组全部触及12秒工程上界。它们是条件模型线索，尚非生理时间常数测量。',
            '双Hb残差具有强同向动态结构。EOG虽有关联，清理却未显示超出幅度匹配对照的额外收益，不能据此解释主要空间模式。'],sections=[])
    content['sections'].append(dict(title='1. 分析问题、信息条件与证据范围',blocks=[
        _paragraph('本报告依据空间模式能否帮助区分机制来组织证据。固定参数给所有位置一个共同动力学参照；共享参数允许局部时间响应改变。比较的重点是空间方向、重复性、输入条件依赖和失败位置，而非按平均误差给两种设置排名。'),
        _table(['证据','范围','本报告用途'],[
            ['固定参数全通道','103名被试、473088个计划窗口×Hb通道对','区域分布与个体图谱；失败沿用原分母。'],
            ['区域汇总','Single-Trial 29人；Simultaneous 26人','通道名称可明确粗分为前额／中央／后部。'],
            ['共享κ、τ面板','103人、2826组、38268窗对；每记录3个预选位置','参数适配与输入条件敏感性；不插值成自由参数全脑图。'],
            ['残差形态／EOG','预定三通道面板；EOG仅两库有原生参照','方向分解及独立辅助证据；不扩展成全通道因果归因。'],
            ['Visual','已有EEG通道身份错误','仅保留经审计的纯Hb形态；不解释其跨模态参数。']],
            '组、窗口、通道和任务内重复均不增加独立被试数。完整共享参数全量计划仍未完成。',[1.3,2.4,3.7]),
        _paragraph('固定组采用κ=0.64/秒、τ=2秒；共享组在原生记录、连续Hb支持和Hb通道内拟合κ、τ。两组均逐窗估计驱动和四初态。Hb-only只看双Hb，联合拟合同时看EEG代理与双Hb；训练尺度、观察算子和评分窗口沿用原合同。'),
        _paragraph('区域是传感器位置的粗分：AF/Fp前缀为前额，C/FC前缀为中央，P/O前缀为后部。Single-Trial依次9、24、3个位置；Simultaneous依次16、10、10个位置。两套布局不等同于同一皮层分区；REFED另保留个体图作口径对照，不强行并入区域统计。')]))
    content['sections'].append(dict(title='2. 指标与图表应怎样读取',blocks=[
        _paragraph('D1是处理后双Hb对Hb-only条件重建的训练SD标准化RMS。R_H是同有效支持上的Hb残差MSE／零变化预测MSE，小于1表示优于零变化基线。它不是R²、病变程度或非神经成分百分比。各通道及外层折有自己的训练SD。'),
        dict(type='equation',latex=r'\begin{aligned}D_1&=\sqrt{\operatorname{mean}_{t,c}\!\left[\left(\frac{H_c(t)-\widehat H_c(t)}{\sigma_c}\right)^2\right]}\\R_H&=\frac{\operatorname{mean}_{t,c}\!\left[\left(\frac{H_c(t)-\widehat H_c(t)}{\sigma_c}\right)^2\right]}{\operatorname{mean}_{t,c}\!\left[\left(\frac{H_c(t)}{\sigma_c}\right)^2\right]}\end{aligned}',
            caption='H为参考后的处理Hb，c为HbO/HbR，σ为冻结训练SD，主要评分为各30秒窗的5.0至29.8秒。原记录／被试汇总继续分别合并平方量或比值分子分母。'),
        _paragraph('弱动态平衡D_bal直接检验连续HbT/HbR在当前Tak小信号关系下的弱积分残差。10、20、40秒表示积分核宽度；分母使用同支持的两侧动态量平方和。它不依赖当前窗口的神经驱动拟合，但依赖模型系数、处理坐标和有效支持，没有正常生理阈值。'),
        dict(type='equation',latex=r'm_{s,R}=\operatorname{median}_{c\in R}M_{s,c},\qquad \Delta_s=m_{s,\mathrm{posterior}}-m_{s,\mathrm{central}}',
            caption='M为原有记录等权被试通道值，R为本报告的测量区域。区域内全部计划位置均有值、三个区域均完整才进入配对；最后以被试等权呈现中位数、四分位距和方向人数。'),
        _paragraph('地形图从头顶按球面角度展开三维模板，赤道显示半径为头部轮廓的90%。REFED保留既有平面模板。线性插值／边缘外推填满头部，黑边点保留原通道数值，灰叉表示缺失，虚线标出测点凸包。空白区域被着色只是显示变化，不代表新增测量或更高空间分辨率。')]))
    content['sections'].append(dict(title='3. 后部相对Hb失配在Simultaneous中具有较广覆盖',page_break=True,blocks=[
        figure('01_subject_regions'),
        _table(['指标','前额','中央','后部','后部较高','配对差中位数'],rows,
            '前三个数值列为各区域被试中位数，最后一列先按同被试相减再取中位数，因此不一定等于前面两个中位数相减。方向人数比较后部与中央。',[2.2, .7,.7,.7,1,1.4]),
        _paragraph('Simultaneous后部的相对Hb残差比在23/26名被试中高于中央，配对差中位数为+0.0823；20秒弱平衡为22/26、+0.0558。D1只有18/26同向。因此，值得研究的是归一化后和动态关系中仍保留的区域差异，不能仅用绝对误差热点替代。'),
        _paragraph('Single-Trial的后部方向较弱且采样更少，不能把Simultaneous的模式提升为四库共同生理规律。一名Single-Trial被试未满足完整区域支持，仍在计划29人的分母中。'),
        _table(['数据集','指标','可用／计划人','后部较高／可用人','后部−中央中位数'],scale_rows,
            '核宽敏感性仍是同一批被试与相关指标的描述，不是多份独立确认或新增显著性检验。',[1.6,1.7,1.1,1.3,1.7])]))
    content['sections'].append(dict(title='4. 总体空间图与跨任务保持性',page_break=True,blocks=[
        figure('02_population_maps'),
        _paragraph('总体图每点为该通道可用被试的中位数。它与区域图的“被试内区域中位数，再跨被试中位数”汇总次序不同，两者各自保留数值，不能互相替代。Single-Trial有个别位置缺值，各点可用人数在population_maps.csv中明确保留。'),
        figure('03_task_consistency'),
        _paragraph('在DSR、n-back和词生成中，后部相对残差高于中央的被试分别为24/26、23/26、23/26；20秒弱平衡分别为23/26、22/26、22/26。三个任务共享被试与测量装置，不能当作三次独立人群重复。'),
        _paragraph('该保持性使单一任务的瞬时神经活动不足以独自解释现象。候选解释包括区域动力学、静脉／浅表成分混合及稳定的光学采样差异。当前图谱尚不能在这些机制间作唯一归因。')]))
    content['sections'].append(dict(title='5. 个体热点稳定，但不同指标保留的模式并不相同',page_break=True,blocks=[
        figure('04_case_maps'),
        _paragraph('Single-Trial 25号的C1FC1、C1C3和CP3C3为左中央高D1位置；相对残差也强调左中央偏后部。06号的FpzFp1、FpzFp2及相邻前额位置在D1、相对残差和弱平衡中均值得跟踪。Simultaneous VP023的P4h、PPOz、P3h形成后部高值，D1与相对残差空间相关为0.855。'),
        figure('06_record_stability'),
        _table(['被试','指标','同任务记录相关中位数','跨任务记录相关中位数'],same_rows,
            '括号内是记录对数，不是独立被试数。相关为同一指标在共同通道上的Spearman排序相关。',[1.3,1.2,2.4,2.5]),
        _paragraph('D1稳定并不保证相对残差同样稳定。06号同任务D1相关中位数0.941，而相对残差为0.488；25号分别0.918和0.676。VP023跨任务两项均较高，分别0.958和0.874。因此，病例的机制价值应结合多个读数，而不是只看一张稳定的亮色图。')]))
    content['sections'].append(dict(title='6. 反例帮助区分幅度热点与相对不相容',page_break=True,blocks=[
        figure('05_amplitude_counterexamples'),
        _paragraph('Single-Trial 28号OzPOz的D1约1.984，但相对残差更强调左中央位置。VP022的前额D1热点与相对残差地图相关仅0.028；REFED 30号只有0.108。它们说明标准化绝对误差与相对基线误差可以指向不同位置。'),
        _paragraph('这些图适合检查局部信号幅度、观察尺度与动力学关系各自的贡献。REFED 30号D1跨视频空间相关中位数0.776，相对残差仅0.184；不同视频不是同条件重测，不能把稳定D1图直接解释为稳定生理异常。')]))
    content['sections'].append(dict(title='7. 自由参数揭示局部适配需求，也暴露解释边界',page_break=True,blocks=[
        figure('07_tau_and_failures'),
        _table(['被试','位置','有效／计划组','Hb-only τ中位数／秒','有效组触上界','有效／计划窗'],tau_rows,
            '每组参数只计一次。有效组与有效窗口分母分别保留；12秒是预设工程上界。',[1.4,1.1,1.1,1.5,1.2,1.3]),
        _paragraph('VP022的前额AFpz在可用组中需要更长的τ：中位数9.61秒，中央C5h为3.16秒、后部POOz为3.33秒。三种任务的前额可用结果方向一致，但前额只有6/9组通过，不能忽略其余组。'),
        _paragraph('VP023后部POOz的7/9个有效Hb-only组均达到12秒；相同7组的联合拟合τ中位数约7.38秒。两视图的允许拟合窗口可能不同，差异不能唯一归因于增加EEG信息。这更适合作为“模型需要不同时间响应、且估计受观察及支持条件影响”的线索，不能直接命名为真实血流通过时间。06号FpzFp2只有1/6组通过，六次末值触界不能作为六次成功生理重测。'),
        _paragraph('κ与自由驱动可以在血管舒张信号方程中补偿，参数先验也会参与选择条件解。拟合参数的稳定、边界和输入条件依赖应与残差一起读，不能只看参数是否有解或图案是否清楚。'),
        figure('08_same_window_adaptation'),
        _paragraph('配对图只回答参数适配吸收了哪些位置的误差。每条短线保留同一组的相同窗口支持，成功子集的下降不代表失败窗口上的收益。自由面板仅三个位置，本报告没有将其外推成完整自由参数地形图。')]))
    content['sections'].append(dict(title='8. 双Hb的共同残差方向与眼电参照',page_break=True,blocks=[
        figure('09_residual_directions'),
        _table(['数据集','人数','共同方向平方比','HbO/HbR残差相关','偏置平方比'],shape_rows,
            '均为既有三通道面板的被试中位数。共同方向采用各色团训练SD坐标，不是原生HbT。',[1.8,.6,1.5,1.7,1.4]),
        _paragraph('残差相关中位数约0.95，偏置平方比中位数约0.04至0.09，说明应保留去均值动态结构。共同血流、血容量／静脉混合是值得检查的候选，但受限模型的投影几何也可形成同向残差，不能由同向性直接确定共同生理源。'),
        figure('10_eog_control'),
        _paragraph('EOG活动与失配有约0.10至0.21的被试等权窗间秩相关。EOG回归相对原代理降低MSE约0.400和0.405，但相对均值／SD匹配对照的差值为+0.00627与−0.00152，95%区间均跨零。它不支持将主要空间热点直接解释为眼源贡献。'),
        _paragraph('Kirilina等通过fNIRS、fMRI、血管成像和外周生理记录显示，前额任务相关浅表血流可以污染fNIRS。这提供候选机制依据，不能替代本实验的来源识别。https://pmc.ncbi.nlm.nih.gov/articles/PMC3348501/')]))
    content['sections'].append(dict(title='9. 如何用图谱区分候选机制',page_break=True,blocks=[
        _table(['候选机制','现有相关线索','仍需区分的解释','有用的后续判据'],[
            ['局部血管动力学差异','区域方向及个体模式保持；局部τ适配不同','驱动补偿、固定其他参数、带通与观察算子','同通道未参与拟合的记录上，参数与残余形状能否共同复现。'],
            ['共同血流／浅表或静脉混合','双Hb同向残差，部分前额模式','投影几何、尺度及模型结构','独立短间距／外周生理锚点能否解释该方向及空间分布。'],
            ['稳定测量链差异','跨任务模式、绝对与相对误差地图分离','真实个体血管特征','重新放置传感器、独立光学质量与头部配准后，模式随位置还是随个体保持。'],
            ['眼源或共同状态影响EEG','原生EOG关联及清理敏感性','幅度改变及与眼动相关的神经活动','超出幅度对照的局部、同支持收益；不能用额区EEG替代EOG。']],
            '表中机制均为待区分假说，不是本报告确认的来源。后续判据是建议，本次没有启动新实验。',[1.4,2.0,1.9,2.7]),
        _paragraph('Tak原研究把神经动力学、血流动力学和光学观测分开，并包含空间敏感度与静脉混合。当前采用其血流核心的条件拟合不等于完整的个体光学逆模型。https://pmc.ncbi.nlm.nih.gov/articles/PMC4401444/'),
        _paragraph('当前适合以固定参数全通道图作为统一参照，以自由参数点位图作为机制区分对照。优先跟踪Simultaneous后部的总体方向，以及VP023后部、Single-Trial 25号左中央和06号前额的不同模式。选择依据是方向覆盖、重复性、指标一致性与独立参照，而非失配最小。')]))
    source_rows=[[name,str(Path(path).relative_to(run)) if Path(path).is_relative_to(run) else path] for name,path in sources.items()]
    content['sections'].append(dict(title='来源与可复核导出',page_break=True,blocks=[
        _paragraph('R为保留运行 experiments/runs/physiology_semantic_tokenizer/tak_mismatch/20261028_v1。下表相对路径均从R起算。新派生表位于fixed_physiology_v1/mechanism_patterns_v1；其summary.json记录汇总顺序、范围与来源，报告不替代该数值来源。'),
        _table(['来源键','保留路径'],source_rows,'完整可解析路径见同目录source_manifest.json。',[1.5,5.9]),
        _paragraph('新表包括regional_values、regional_summary、regional_contrasts、population_maps、parameter_groups、fit_pair_groups及parameter_site_summary。图中涉及的失败、缺值、输入视图和工程边界均保留。')]))
    # Keep the short coverage table together, rather than leaving its final
    # failure row alone on the next page after the nine-panel parameter plot.
    position=next(i for i,s in enumerate(content['sections']) if s['title'].startswith('7. '))
    parameter_section=content['sections'][position]
    following=parameter_section['blocks'][1:]
    parameter_section['blocks']=parameter_section['blocks'][:1]
    content['sections'].insert(position+1,dict(title='参数覆盖、条件解释与同窗对照',page_break=True,blocks=following))
    content_path=output/'report_content.json';content_path.write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n')
    checks=render_report(content_path,output/'report')
    shutil.copy2(Path(__file__),output/'renderer_source.py')
    manifest=dict(schema='tak_mismatch_mechanism_report_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        analysis=str(analysis),sources=sources,figures=figures,report=str(output/'report/TECHNICAL_REPORT.pdf'),
        command=[str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--run',str(run),'--mechanism-analysis',str(analysis),'--output',str(output)],
        bitmap_figures=True,projection='azimuthal_equidistant_equator_radius_0.9; planar REFED retained',
        full_head_display='linear interpolation/head extrapolation; measured convex hull shown',
        source_parameters_unchanged=True,model_fits=0,raw_signal_reads=0,new_statistical_tests=0,
        numerical_verification=str(analysis/'verification.json') if (analysis/'verification.json').exists() else None,
        pages=checks['exports']['TECHNICAL_REPORT.pdf']['pages'],figure_count=len(figures))
    (output/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return dict(output=str(output),pages=manifest['pages'],figures=len(figures))


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


def render_four_dataset_fixed_patterns(run, analysis, output):
    """Physiological hypotheses from the complete fixed census, with counterchecks."""
    import shutil
    import pandas as pd
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from src.analysis.tak_mismatch import HB_RATIO
    run,analysis,output=(Path(p).resolve() for p in (run,analysis,output))
    summary=_json(analysis/'summary.json')
    if (summary.get('schema')!='tak_mismatch_four_dataset_fixed_patterns_v1' or summary.get('status')!='completed'
            or Path(summary['parent_run']).resolve()!=run or not analysis.is_relative_to(run)):
        raise ValueError('Completed fixed-only four-dataset evidence required')
    if output.exists():raise FileExistsError('Use a fresh fixed-pattern report export')
    tables={k:pd.read_csv(analysis/v,dtype={'subject':str}) for k,v in summary['files'].items()}
    sources=summary['sources'];visual='visual_cognitive_motivation'
    old=pd.read_csv(sources['reference_maps'],dtype={'subject':str},low_memory=False)
    new=pd.read_csv(sources['corrected_maps'],dtype={'subject':str},low_memory=False)
    maps=pd.concat([old[~old.dataset.eq(visual)],new],ignore_index=True)
    old_plan,new_plan=(_json(Path(sources[k])) for k in ('reference_plan','corrected_plan'))
    plan=dict(records=[r for r in old_plan['records'] if r['dataset']!=visual]+new_plan['records'])
    display_cache={}
    def display(frame,dataset,subject=None):
        if subject is None:
            subject=sorted(maps[maps.dataset.eq(dataset)&maps.period.eq('subject_mean')&maps.metric.eq('D1')].subject.unique())[0]
        key=(dataset,subject)
        if key not in display_cache:
            template=maps[maps.dataset.eq(dataset)&maps.subject.eq(subject)&maps.period.eq('subject_mean')&maps.metric.eq('D1')]
            display_cache[key]=_spatial_preview_coordinates(template,plan).set_index('pair_members',drop=False)
        positions=display_cache[key].loc[frame.pair_members]
        if not np.allclose(frame[['x','y']],positions[['x_orthographic','y_orthographic']],atol=1e-7):
            raise ValueError('Fixed report geometry differs from retained registered map')
        result=frame.copy();result[['x','y']]=positions[['x','y']].to_numpy()
        if not np.array_equal(result.display_value,frame.display_value,equal_nan=True):
            raise ValueError('Display projection changed a retained value')
        return result
    datasets=('eeg_fnirs_single_trial','simultaneous_eeg_nirs',visual,'refed')
    short=dict(zip(datasets,('Single-Trial','Simultaneous','Visual','REFED')))
    colors=dict(zip(datasets,('#2272ac','#d17924','#27816d','#9c4b90')))
    output.mkdir(parents=True);plots=output/'plots';plots.mkdir();plt=_plotting();figures={}
    def save(key,fig,caption,source):
        if key in ('03_population_maps','06_subject_examples') or key.startswith('hotspot_'):
            note=('REFED保留登记平面布局' if key=='hotspot_refed' else
                  '三维测点按头顶球面等距展开；REFED保留平面布局' if key=='03_population_maps' else '头顶球面等距展开')
            fig.text(.5,.005,note+'；虚线为测点凸包，线外颜色为外推。',ha='center',fontsize=8,color='#455560')
            if key in ('03_population_maps','hotspot_refed'):
                if key=='hotspot_refed':
                    caption=caption.replace('坐标采用头顶球面等距展开，','采用登记平面测点布局，')
                caption+=' REFED只有登记平面布局，按原布局缩至头部显示范围，不虚构三维球面位置。'
        figures[key]=dict(type='figure',path=_save(fig,plots/(key+'.png')),caption=caption.replace('−','-'),source=source,preserve_bitmap=True)
    def dots(ax,frame,label,limits=None):
        for i,ds in enumerate(datasets):
            g=frame[frame.dataset.eq(ds)].sort_values('subject');v=g.value.dropna().to_numpy(float)
            ax.scatter(i+.16*np.sin(np.arange(len(v))*2.399963),v,s=21,c=colors[ds],alpha=.7)
            if len(v):
                q=np.quantile(v,[.25,.5,.75]);ax.plot([i,i],q[[0,2]],c='black',lw=2)
                ax.scatter([i],[q[1]],s=40,facecolor='white',edgecolor='black',zorder=4)
            ax.text(i,1.02,f'{len(v)}/{len(g)}',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
        ax.set(xticks=range(4),xticklabels=[short[d] for d in datasets],ylabel=label,xlim=(-.45,3.45))
        ax.tick_params(axis='x',rotation=15,labelsize=8);ax.grid(axis='y',alpha=.18)
        if limits:ax.set_ylim(*limits)
    def value(x):return f'{x:.3f}' if np.isfinite(x) else '不可用'
    def section(title,blocks):return dict(title=title,page_break=True,blocks=blocks)
    subj=tables['subjects'];shapes=tables['shape_subjects'];repeat=tables['spatial_repeat_summary']
    regional=tables['regional_contrasts'];tests=tables['regional_tests']
    fig,axes=plt.subplots(1,3,figsize=(10,3.4))
    for ax,m,label in zip(axes,('D1',HB_RATIO,'D_bal_20s'),('Hb残差 / 训练SD','残差 / 零预测MSE','20秒弱动态偏离')):
        dots(ax,subj[subj.metric.eq(m)],label)
    fig.tight_layout();save('01_census',fig,'每点一名被试；白点为被试中位数，黑线为四分位距，顶部为可用／计划人数。纵轴独立；训练SD和单位化误差不代表跨库相同物理浓度。','subjects.csv')
    fig,axes=plt.subplots(1,3,figsize=(10,3.6))
    for ax,m,label in zip(axes,('D1',HB_RATIO,'D_bal_40s'),('D1后部减前额 / 训练SD','相对残差后部减前额','40秒弱动态偏离后部减前额')):
        d=regional[regional.level.eq('subject')&regional.contrast.eq('posterior_minus_frontal')&regional.metric.eq(m)]
        for i,ds in enumerate(datasets[:2]):
            g=d[d.dataset.eq(ds)];v=g.value.dropna().to_numpy(float)
            ax.scatter(i+.15*np.sin(np.arange(len(v))*2.399963),v,c=colors[ds],s=26,alpha=.75)
            if len(v):ax.plot([i-.22,i+.22],[np.median(v)]*2,c='black',lw=2)
            ax.text(i,1.02,f'{int((v>0).sum())}/{len(v)} 正差',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
        ax.axhline(0,c='black',ls='--',lw=.9);ax.set(xticks=[0,1],xticklabels=['Single-Trial','Simultaneous'],ylabel=label,xlim=(-.5,1.5));ax.grid(alpha=.15)
    fig.tight_layout();save('02_regional_contrasts',fig,'每点为一人：先取该人每区域测点的中位数，再作后部减前额。三个区域全部位置可用才纳入；Single-Trial 28/29人，Simultaneous 26/26人。黑横线为配对差中位数；正值表示后部失配更高。','regional_contrasts.csv; regional_tests.csv')
    population=tables['all_population_maps'];fig,axes=plt.subplots(2,4,figsize=(10,5.8))
    fig.subplots_adjust(left=.02,right=.9,bottom=.06,top=.94,wspace=.10,hspace=.26)
    for row,(metric,label) in enumerate((('D1','D1 / 训练SD'),(HB_RATIO,'残差 / 零预测MSE'))):
        frame=population[population.metric.eq(metric)];upper=float(frame.display_value.max())
        for j,ds in enumerate(datasets):
            g=display(frame[frame.dataset.eq(ds)],ds)
            im=spatial_topomap(axes[row,j],g,limits=(0,upper),title=short[ds],coverage='head')
        cax=fig.add_axes([.93,.57 if row==0 else .12,.017,.30]);fig.colorbar(im,cax=cax).set_label(label)
    save('03_population_maps',fig,'每个测点取被试中位数；两行分别是D1与相对残差，每行四库共用色标。坐标从头顶按球面角度等距展开，鼻尖朝上；黑边点为测点，MNE整头外推填色。虚线为测点凸包，线外颜色是显示外推。四库覆盖不同，不能解释为同一皮层脑区；填色不增加测量分辨率。','all_population_maps.csv; display_coordinates.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for ax,m,label in zip(axes,('D1',HB_RATIO),('D1空间排序：早段 vs 晚段','相对残差空间排序：早段 vs 晚段')):
        d=tables['early_late_subjects'];dots(ax,d[d.metric.eq(m)],label,(-1.02,1.02));ax.axhline(0,c='grey',ls='--',lw=.8)
    fig.tight_layout();save('04_early_late',fig,'全体被试均进入分母，未按稳定性选例。每个记录要求每半至少2个有效窗口、至少8且80%的共同测点；每点为一人的记录相关系数中位数。它是记录内稳定性，不是跨日重测。','early_late_subjects.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,6.4))
    for row,m in enumerate(('D1',HB_RATIO)):
        for col,endpoint in enumerate(('raw_rho','centered_rho')):
            d=tables['spatial_repeat_subjects'];g=d[d.metric.eq(m)&d.endpoint.eq(endpoint)]
            dots(axes[row,col],g,('D1' if row==0 else '相对残差')+'空间相关',(-1.02,1.02))
            axes[row,col].set_title('原空间图' if col==0 else '减去其他被试的共同位置模式',pad=27)
            axes[row,col].axhline(0,c='grey',ls='--',lw=.8)
    fig.tight_layout();save('05_record_stability',fig,'每点一人，先计算记录对的空间Spearman相关，再取该人的中位数。右列逐记录减去其他被试同位置中位数。Single-Trial仅同任务记录；Simultaneous跨任务；Visual跨Part；REFED跨视频。后两种跨条件相似不能称为同条件重测。','spatial_repeat_subjects.csv; spatial_repeat_summary.csv')
    fig,axes=plt.subplots(2,3,figsize=(9.8,6.1));fig.subplots_adjust(left=.02,right=.9,top=.93,bottom=.06,wspace=.12,hspace=.25)
    for row,metric in enumerate(('D1',HB_RATIO)):
        d=maps[maps.dataset.eq('simultaneous_eeg_nirs')&maps.period.eq('subject_mean')&maps.metric.eq(metric)&maps.subject.isin(['VP008','VP023','VP022'])]
        upper=float(d.display_value.max())
        for col,subject in enumerate(('VP008','VP023','VP022')):
            frame=display(d[d.subject.eq(subject)],'simultaneous_eeg_nirs',subject)
            im=spatial_topomap(axes[row,col],frame,limits=(0,upper),title=subject+(' · D1' if row==0 else ' · 相对残差'),coverage='head')
        cax=fig.add_axes([.93,.57 if row==0 else .12,.017,.30]);fig.colorbar(im,cax=cax)
    save('06_subject_examples',fig,'沿用旧报告VP008、VP023、VP022三个选例，不作为新增独立验证。每行共用色标；前两人偏后部，VP022可偏前额。采用头顶球面等距展开和整头外推，虚线外为显示外推。与全体被试的区域差和稳定性一起解释。','retained spatial maps; display_coordinates.csv; previous fixed spatial selection')
    fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    for ax,f,label,limits in zip(axes,('Hb_residual_OR_correlation','Hb_common_direction_fraction','Hb_bias_fraction'),('HbO/HbR残差相关','标准化共同方向平方比','残差偏置平方比'),((0,1.03),(0,1.03),(0,.20))):
        dots(ax,shapes[shapes.feature.eq(f)],label,limits)
    fig.tight_layout();save('07_residual_geometry',fig,'来源是预先冻结的三位置波形面板，而非全通道波形。先合并窗口残差矩，再按记录等权，最后取比值或相关。共同方向不是HbT或某类生理来源比例。','shape_subjects.csv')
    fig,axes=plt.subplots(1,2,figsize=(9.8,3.4));t=np.linspace(0,8,500);y=np.column_stack([np.sin(2*t),np.cos(3.1*t)]);h=np.array([1.,-1.]);fit=np.outer(y@h/(h@h),h);res=y-fit
    axes[0].scatter(y[:,0],y[:,1],s=7,c='#8798a7',alpha=.45,label='两个不同输入分量');axes[0].plot([-1.2,1.2],[1.2,-1.2],c='#2272ac',label='允许拟合方向');axes[0].set(xlabel='输入1',ylabel='输入2',title='解析投影反例，无生理来源变量');axes[0].legend(fontsize=8)
    axes[1].plot(t,res[:,0],c='#2272ac',lw=2,label='残差1');axes[1].plot(t,res[:,1],c='#d17924',ls='--',label='残差2');axes[1].set(xlabel='示意坐标',ylabel='任意单位',title=f'两残差相关={np.corrcoef(res.T)[0,1]:.3f}');axes[1].legend()
    fig.tight_layout();save('08_projection_counterexample',fig,'纯数学示意：把二维输入投影到方向(1,−1)，残差自然落到(1,1)，即使输入没有同一个生理来源，两残差也完全相关。这不是Tak仿真或实测结果，只说明“同向残差⇒共同生理源”的推理不成立。','analytic orthogonal projection; no model refit')
    fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    dots(axes[0],subj[subj.metric.eq('D_EH')],'Hb解读出EEG误差 / 训练SD')
    dots(axes[1],shapes[shapes.feature.eq('EEG_Hb_readout_correlation')],'EEG代理与Hb解读出相关',(-.15,.2))
    dots(axes[2],subj[subj.metric.eq('D_drive_H')],'驱动共同初态回放差 / Hb训练SD')
    fig.tight_layout();save('09_crossmodal',fig,'两侧为全通道标量，中间为冻结三位置波形面板。低形态相关说明当前功率代理与Hb解读出不一致；不是对所有频带、时滞或真实神经血管耦合的否定。','subjects.csv; shape_subjects.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    ind=tables['independent_prediction_subjects']
    for ax,m,label in zip(axes,('prediction_gain_over_zero','prediction_gain_over_template'),('零预测MSE − EEG-only MSE','训练模板MSE − EEG-only MSE')):
        dots(ax,ind[ind.metric.eq(m)],label);ax.axhline(0,c='black',ls='--',lw=.8)
    fig.tight_layout();save('10_independent_prediction',fig,'每点一人；先在相同有效窗口作基线误差减EEG-only误差，再按记录和被试汇总。正值有利于EEG-only，单位为Hb训练SD平方。当前窗口Hb在拟合和初态中隐藏；该对照不同于看得到Hb的条件重建。','independent_prediction_subjects.csv')
    result=_finish_fixed_pattern_report(run,analysis,output,summary,tables,maps,figures,save,plt,short,datasets,value,section,display)
    coordinates=pd.concat([v.reset_index(drop=True) for v in display_cache.values()],ignore_index=True)
    coordinates.to_csv(output/'display_coordinates.csv',index=False)
    spans=[]
    for (ds,sub),frame in display_cache.items():
        front=frame[frame.y_orthographic>.8]
        before=float(np.ptp(front.y_orthographic)) if len(front) else 0.
        after=float(np.ptp(front.y)) if len(front) else 0.
        if before>0:spans.append(dict(dataset=ds,subject=sub,frontal_rule='retained orthographic y >0.8',
            frontal_positions=len(front),old_y_span=before,new_y_span=after,ratio=after/before))
    import mne
    (output/'display_projection.json').write_text(json.dumps(dict(projection='vertex-centered azimuthal equidistant',equator_radius=.9,
        planar_exception='REFED preserves registered paper 2D layout, scaled by0.9 for display; no inferred depth or spherical position',
        library='MNE',library_version=mne.__version__,extrapolate='head',image_interp='linear',convex_hull='dashed; outside colors are extrapolation',
        scalar_changes=0,selection_changes=0,spatial_figures=6,frontal_spans=spans,reference='https://mne.tools/1.12/generated/mne.viz.plot_topomap.html'),ensure_ascii=False,indent=2)+'\n')
    return result


def _finish_fixed_pattern_report(run,analysis,output,summary,tables,maps,figures,save,plt,short,datasets,value,section,display):
    import shutil
    import pandas as pd
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from src.analysis.tak_mismatch import HB_RATIO
    subjects=tables['subjects'];shapes=tables['shape_subjects'];tests=tables['regional_tests'];repeat=tables['spatial_repeat_summary']
    task=tables['task_comparisons'];chosen=task[task.dataset.eq('simultaneous_eeg_nirs')&task.task_a.eq('nback')&task.task_b.eq('wg')]
    fig,axes=plt.subplots(1,4,figsize=(10,3.3))
    for ax,m,label in zip(axes,('D1','D_EH','D_drive_H',HB_RATIO),('Hb残差D1','Hb解的EEG读出误差','共同初态驱动回放差','相对零预测残差比')):
        r=chosen[chosen.metric.eq(m)].iloc[0]
        ax.errorbar([0],[r.mean_difference],yerr=[[r.mean_difference-r.ci_low],[r.ci_high-r.mean_difference]],fmt='o',capsize=5,color='#d17924')
        ax.axhline(0,c='black',ls='--',lw=.8);ax.set(xticks=[],ylabel='WG − n-back',title=label,xlim=(-.6,.6));ax.grid(alpha=.15)
        ax.text(.5,.98,f'n={r.paired_subjects}\nBH q={r.q_value:.4f}',transform=ax.transAxes,ha='center',va='top',fontsize=8)
    fig.tight_layout();save('11_task_contrast',fig,'点为26人配对差均值，线为整被试重采样95%区间；纵轴独立。前三项来自原327项主检验BH家族，相对残差来自1542项诊断家族。差为词生成减n-back；没有按最小p值删减这四项。','task_comparisons.csv; retained statistical family')
    ocular=tables['ocular_associations'];cf=ocular[ocular.analysis.eq('counterfactual')]
    fig,axes=plt.subplots(1,2,figsize=(9.6,3.5))
    for ax,ds in zip(axes,datasets[:2]):
        for i,(c,label) in enumerate((('clean_minus_raw','EOG回归 − 原代理'),('amplitude_minus_raw','均值/尺度匹配 − 原代理'),('clean_minus_amplitude','EOG回归 − 匹配对照'))):
            r=cf[cf.dataset.eq(ds)&cf.contrast.eq(c)].iloc[0]
            ax.errorbar(r['mean'],i,xerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],fmt='o',capsize=4,color='#2272ac')
        ax.axvline(0,c='black',ls='--',lw=.8);ax.set(yticks=range(3),yticklabels=['EOG回归 − 原代理','匹配对照 − 原代理','EOG回归 − 匹配对照'],xlabel='EEG读出差平方的变化 / 训练SD平方',title=short[ds]);ax.invert_yaxis();ax.grid(alpha=.15)
    fig.tight_layout();save('12_ocular_controls',fig,'点为被试等权均值，线为95%被试自助区间；负差表示新的EEG代理更接近固定Hb-only读出。EOG回归不重新拟合SSM；匹配对照只调整代理均值和SD。清理相对原代理有收益，但相对匹配对照区间跨零，不能归为眼源移除的独有贡献。','ocular_associations.csv; retained counterfactual analysis')
    references=[
        ('Kirilina et al., 2012, NeuroImage','https://pmc.ncbi.nlm.nih.gov/articles/PMC3348501/','多模态及外周测量发现头皮血管信号可与任务相关，并可呈个体化、重复的空间分布；这支持候选解释的合理性，不证明本数据的后部模式同源。'),
        ('Yücel et al., 2015, Neurophotonics','https://pmc.ncbi.nlm.nih.gov/articles/PMC4717232/','不同自主神经反应任务中，短源探距回归有助于区分浅表与脑内血流；本轮未使用此类独立分离证据。'),
        ('Shin et al., 2018, Scientific Data','https://www.nature.com/articles/sdata20183','Simultaneous来自同次访问的三类任务，任务顺序未随机化，不能据此确认跨日特质或纯任务因果效应。'),
        ('Tak et al., 2015, NeuroImage','https://www.fil.ion.ucl.ac.uk/~wpenny/publications/tak-penny15.pdf','血流模型将隐藏状态映射到光学观测。本项目使用约简固定模型；条件拟合参数、真实神经驱动与观测混合的识别应分别检验。')]
    median=lambda frame,field,name,ds:float(frame[frame.dataset.eq(ds)&frame[field].eq(name)].value.median())
    regional_rows=[]
    for _,r in tests.iterrows():
        regional_rows.append([short[r.dataset],{'D1':'D1',HB_RATIO:'相对残差','D_bal_40s':'40秒弱残差'}[r.endpoint],f'{r.subjects}/{r.planned_subjects}',
            value(r['mean']),f'[{value(r.ci_low)}, {value(r.ci_high)}]',f'{r.q_value:.5f}'])
    coverage=[];failure_rows=[]
    for ds in datasets:
        c=summary['coverage'][ds];p=c['plan'];coverage.append([short[ds],p['subjects'],p['Hb_admitted_records'],p['planned_pair_windows'],c['views']['hb_only']['completed']])
        for v,label in (('hb_only','Hb-only'),('eeg_only','EEG-only'),('joint','joint')):
            a=c['views'][v];failure_rows.append([short[ds],label,a['planned'],a['completed'],a['numerical_failure'],a['unavailable_support'],a['execution_failure']+a['missing']])
    content=dict(schema='source_diagnostics_report_v1',title='四数据集固定参数失配：最值得追踪的生理相关模式',
        subtitle='完整指标筛查、空间稳定性、竞争机制与证据边界',date=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y-%m-%d'),
        provenance=['103名被试、473088个计划窗口×Hb位置；前三库保留原结果，Visual完整替换为原生通道更正版本。',
            '本轮只读既有数值，新增全被试空间统计与可视化；未读取原始实测信号、未重拟合、未恢复停止的可变参数实验。'],
        summary=['最值得优先解释的是稳定而具有空间结构的Hb模型失配；Simultaneous的后部模式同时得到相对残差和弱动态关系支持。',
            'Simultaneous后部相对残差高于前额：22/26人同向，平均差0.069，95%区间0.047–0.091。它是有独立基线核对的描述性表型，不等于已定位的生理来源。',
            '减去其他被试的共同空间模板后，相对残差跨记录相关中位数：Single-Trial 0.592、Simultaneous 0.722、Visual 0.726；REFED仅0.079。数据集差异是结论的一部分。',
            '四库最一致的统计现象是双Hb残差高度同向、Hb解与EEG代理形状相关很低；拟合几何、代理与观测混合仍是竞争解释。'],sections=[],sources=[str(analysis/'summary.json')]+[r[1] for r in references])
    content['sections'].append(section('1. 本轮分析了什么，以及什么没有被当成独立样本',[
        _table(['数据集','被试','允许Hb记录','计划窗对','Hb-only成功'],coverage,'一窗对为30秒目标窗口×一个HbO/HbR测量位置。重复窗口、通道、同步Probe和视频均不增加独立被试数。'),
        _paragraph('固定参考参数在所有位置相同：血流时间参数为2秒、信号衰减参数为0.64/秒、神经血管有效增益为1；驱动轨迹和四个初态仍然按窗口拟合。Hb-only只看双Hb，EEG-only只看功率代理，joint同时看两模态。本文比较已有固定模型的失配分布，不把自由驱动拟合称为参数完全固定的前向预测。'),
        _paragraph('目标窗首5秒为参考，评分5.0–29.8秒，共249个10Hz时刻；EEG代理由六个近邻电极、五频带对数功率的训练折第一主成分组成，计算可使用更长的原生上下文。Hb沿用离线滤波观测。训练尺度按外层训练被试拟合，不按当前窗口重新归一化。'),
        _paragraph('标量覆盖全量计划窗对；残差形状与眼电分析使用事先冻结的三位置面板。Visual只用更正后EEG通道身份，纯Hb端点与旧版本逐被试复现。模型真值、皮层源位置和未观测生理过程均未在本轮获得。')]))
    content['sections'].append(section('2. 绝对失配、相对失配和稳定性分别回答什么',[
        dict(type='equation',latex=r'D_1=\sqrt{\langle (H-\widehat H_H)^2/\sigma_H^2\rangle},\qquad R_H=\frac{\langle(H-\widehat H_H)^2/\sigma_H^2\rangle}{\langle H^2/\sigma_H^2\rangle}',caption='H为窗口参考后的处理Hb，σ为对应色团的冻结训练SD；帽号为Hb-only重建。平均先在同一有效时间和色团上进行。R_H为残差与零变化预测的MSE之比，不是R²或能量来源比例。'),
        dict(type='equation',latex=r'\Delta_s=\operatorname{median}_{p\in P}M_{sp}-\operatorname{median}_{p\in F}M_{sp},\qquad \rho_s=\operatorname{median}_{a<b}\operatorname{Spearman}_p(M_{sap},M_{sbp})',caption='s为被试，p为测点，P/F为后部/前额，a/b为记录；M是指定失配指标。区域比较要求三个区域完整，稳定性比较要求至少8个且80%的共同位置；这两种可用性规则不同。'),
        _paragraph('全库标量按窗口平方量、记录、被试逐级汇总；图中标明被试中位数或均值。区域统计先取各区域位置中位数，再在同被试内作差；新区域检验仅包括两库×三指标共6项，5000次整被试自助法区间、100000次符号置换与BH校正。它属于已看过结果后的探索，校正不覆盖整个历史探索过程。'),
        _paragraph('空间稳定性的区间是5000次被试自助法对“被试记录对相关中位数”的条件区间；拟合、尺度和去群体模板均冻结，未传播模板估计不确定性。记录内早晚与跨记录、跨任务、跨视频的语义分别呈现。')]))
    content['sections'].append(section('3. 全量分布显示Hb可拟合性不同，但不能据此给生理可信度排名',[
        figures['01_census'],_table(['数据集','D1中位数','相对残差中位数','20秒弱偏离中位数'],[[short[ds],*[value(median(subjects,'metric',m,ds)) for m in ('D1',HB_RATIO,'D_bal_20s')]] for ds in datasets]),
        _paragraph('REFED的Hb相对残差较小，但后续空间相对残差的跨视频稳定性也更弱。低条件重建误差不等于更稳定的个体特征，更不等于更准确的神经驱动。D1受训练尺度与信号幅度影响，应与同支持的零预测比值并看。')]))
    content['sections'].append(section('4. 最清楚的区域模式：Simultaneous后部偏离同时出现在三种口径',[
        figures['02_regional_contrasts'],_table(['数据集','指标','可用/计划人','平均差','95%区间','BH q'],regional_rows),
        _paragraph('Simultaneous中，后部高于前额的有D1 19/26人、相对残差22/26人、40秒弱偏离20/26人；三项均在这个探索性6检验家族内保留。相对残差平均增加0.069是比值的绝对差，不是误差增加6.9%。Single-Trial三个区间均跨零，未发现同样清晰的后部趋势。'),
        _paragraph('这使“Simultaneous后部测量位置与统一Tak关系更不相容”成为当前最值得优先追踪的区域假设。弱残差独立于EEG驱动拟合，但仍共享Hb观测和处理链；三指标一致不是三份独立生理验证。后部/前额是测点前缀分区，不是个体皮层分区。')]))
    content['sections'].append(section('5. 群体空间图保留覆盖差异；个体热点并不都在后部',[
        figures['03_population_maps'],_paragraph('群体图把每位置的被试中位数画在登记模板上。Simultaneous后部趋势与区域统计一致；不同库的光路、覆盖、归一化和观测单位不同，不把跨库色值当作相同物理量或疾病风险。')]))
    content['sections'].append(section('6. 全被试时间稳定性成立到什么程度',[
        figures['04_early_late'],_paragraph('本页没有使用按空间稳定性挑出的病例：每个被试先汇总所有可用记录，再进入总体分布。Simultaneous只有23/26人满足早晚地图的共同位置标准；缺少支持者仍保留在分母。'),
        _paragraph('固定观测链能同时产生稳定的测点偏差和生理差异。因而较高的早晚相关说明“该模式可重复描述”，不能单独排除光耦合、头皮血流、头发、固定训练尺度或长程滤波的影响。')]))
    rr=[]
    for ds in datasets:
        row=[]
        for m in ('D1',HB_RATIO):
            for ep in ('raw_rho','centered_rho'):
                r=repeat[repeat.dataset.eq(ds)&repeat.metric.eq(m)&repeat.endpoint.eq(ep)].iloc[0];row.append(value(r['median']))
        r=repeat[repeat.dataset.eq(ds)&repeat.metric.eq(HB_RATIO)&repeat.endpoint.eq('centered_rho')].iloc[0]
        rr.append([short[ds],f'{r.subjects}/{r.planned_subjects}',*row,f'[{value(r.ci_low)}, {value(r.ci_high)}]'])
    content['sections'].append(section('7. 减去群体共有图形后，前三库仍保留个体空间结构；REFED明显减弱',[
        figures['05_record_stability'],_table(['数据集','可用/计划人','原D1ρ','去模板D1ρ','原相对ρ','去模板相对ρ','末列95%区间'],rr),
        _paragraph('去模板时，每个人在每条记录、每个位置减去其他人的中位数，再比较自己的不同记录。Simultaneous相对残差的相关中位数仍为0.722，24/26人高于0.5；Single-Trial为0.592，Visual为0.726。说明这些模式不全由所有人共有的头部模板造成。'),
        _paragraph('REFED原D1跨视频相关较高，但相对残差从0.158降到去模板后的0.079，没人超过0.5。这是重要反例：原D1热点可能较多反映稳定幅度/尺度或视频条件；目前不支持同样强的相对失配指纹。'),
        _paragraph('去群体模板仍保留每人的固定测量链和尺度偏差，不能解释为已排除所有观测因素。Visual只有11/16人具有可用跨Part地图；Simultaneous与REFED均非同任务跨日重测。')]))
    content['sections'].append(section('8. 保留原病例作为读图示例，群体统计承担主要证据',[
        figures['06_subject_examples'],_paragraph('VP008和VP023后部热点较突出，VP022的D1更偏前额。相对残差图会改变部分热点排序，说明绝对误差与相对可解释性回答不同问题。上述病例原先经过反差和稳定性筛选，因此只是示例；第4、6、7节的全被试结果避免用这些病例代替总体证据。')]))
    if 'hotspot_persistence' in tables:
        import re
        hot=tables['hotspot_subjects'];persistent=tables['hotspot_persistence'];sp=tables['spatial_repeat_subjects']
        statements=[]
        for ds in datasets:
            g=hot[hot.dataset.eq(ds)&hot.metric.eq(HB_RATIO)];good=g.eligible_locations.gt(0)
            statements.append(f"{short[ds]} {int((good&g.stable_locations.gt(0)).sum())}/{int(good.sum())}人")
        content['summary'].insert(1,'四库都可观察到持续的个体高失配位置：相对残差口径下，'+
            '、'.join(statements)+'。这里指至少2条、80%的计划记录可评价，且某位置在至少80%可评价记录进入本人最高四分之一；不是生理诊断阈值。')
        fig,axes=plt.subplots(1,2,figsize=(10,3.5));hotrows=[]
        for ax,m,label in zip(axes,('D1',HB_RATIO),('D1高失配位置重合率','相对残差高失配位置重合率')):
            for i,ds in enumerate(datasets):
                g=sp[sp.dataset.eq(ds)&sp.metric.eq(m)&sp.endpoint.eq('hotspot_overlap')]
                v=g.value.dropna().to_numpy(float)
                ax.scatter(i+.15*np.sin(np.arange(len(v))*2.399963),v,s=24,alpha=.7,
                    color=('#2272ac','#d17924','#27816d','#9c4b90')[i])
                if len(v):ax.plot([i-.2,i+.2],[np.median(v)]*2,c='black',lw=2)
                hp=hot[hot.dataset.eq(ds)&hot.metric.eq(m)];eligible=hp.eligible_locations.gt(0)
                hotrows.append([short[ds],'D1' if m=='D1' else '相对残差',f'{len(v)}/{len(g)}',value(np.median(v)) if len(v) else '不可用',
                    f'{int((hp.stable_locations.gt(0)&eligible).sum())}/{int(eligible.sum())}'])
            ax.axhline(.25,c='grey',ls='--',lw=.8);ax.set(xticks=range(4),xticklabels=[short[d] for d in datasets],ylim=(-.03,1.05),ylabel=label)
            ax.tick_params(axis='x',rotation=15);ax.grid(axis='y',alpha=.15)
        fig.tight_layout();save('13_hotspot_overlap',fig,'每点为一人的记录对重合率中位数：在共同测点上选各图最高四分之一，交集数除以所选位置数。灰线约0.25是等大小随机位置集合的期望参考，实际参考随取整略变；它不是显著性阈值，未假定相邻通道独立。','spatial_record_pairs.csv; hotspot_subjects.csv')
        content['sections'].append(section('8A. 单个被试的持续热点本身就是发现，不要求群体热点重合',[
            figures['13_hotspot_overlap'],_table(['数据集','指标','配对/计划人','热点重合中位数','有持续热点/可评人'],hotrows),
            _paragraph('这里直接检查高失配位置，而不只检查整张图的相关。持续热点定义为：该位置在至少80%的可评价记录中进入本人的最高四分之一，且至少2条、80%的计划记录可评价。这是明确的描述阈值，不是正常生理阈值；完整频率和并列截点保留在表中。'),
            _paragraph('某个被试反复出现一个或几个高失配点，即使不与其他人的热点重合，也支持可复核的个体空间结构。不同数据集中出现这种性质可形成跨数据集的研究线索，但不能说它们位于同一皮层区域或由同一种生理过程造成。'),
            _paragraph('REFED的整体相对误差地图相关较弱，并不逻辑上排除少数持续热点；下面同时给出全体统计和数据选择的病例，避免把“全图不稳定”误写成“局部没有稳定结构”。')]))
        selected=[]
        for ds in datasets:
            candidates=hot[hot.dataset.eq(ds)&hot.metric.eq(HB_RATIO)&hot.eligible_locations.gt(0)].copy()
            overlap=sp[sp.dataset.eq(ds)&sp.metric.eq(HB_RATIO)&sp.endpoint.eq('hotspot_overlap')][['subject','value']]
            candidates=candidates.merge(overlap,on='subject',validate='one_to_one')
            candidates=candidates.sort_values(['stable_locations','maximum_persistence','value','subject'],ascending=[False,False,False,True])
            if candidates.empty:continue
            subject=candidates.iloc[0].subject
            full=maps[maps.dataset.eq(ds)&maps.subject.eq(subject)&maps.metric.eq(HB_RATIO)&maps.period.eq('full')]
            if ds=='eeg_fnirs_single_trial':full=full[full.task_label.eq('motor_imagery')]
            if ds=='simultaneous_eeg_nirs':full=full[full.task_label.isin(['nback','wg'])]
            records=sorted(full.record_unit.unique(),key=lambda x:[int(q) if q.isdigit() else q for q in re.split(r'(\d+)',x)])
            if len(records)<2:continue
            records=[records[0],records[-1]]
            pp=persistent[persistent.dataset.eq(ds)&persistent.subject.eq(subject)&persistent.metric.eq(HB_RATIO)].copy()
            fig,axes=plt.subplots(1,3,figsize=(10,3.6));fig.subplots_adjust(left=.02,right=.97,bottom=.22,top=.85,wspace=.35)
            common=full[full.record_unit.isin(records)];upper=float(common.display_value.max())
            for j,record in enumerate(records):
                frame=full[full.record_unit.eq(record)]
                frame=display(frame,ds,subject)
                im=spatial_topomap(axes[j],frame.drop(columns=['period']),limits=(0,upper),title=record.rsplit('|',1)[-1],coverage='head')
            cb=fig.colorbar(im,ax=list(axes[:2]),orientation='horizontal',fraction=.07,pad=.06);cb.set_label('相对零预测残差比，两记录共用色标',fontsize=8)
            pp['display_value']=pp.fraction.where(pp.eligible)
            pp=display(pp,ds,subject)
            im=spatial_topomap(axes[2],pp,limits=(0,1),title='各位置进入高四分位的频率',coverage='head')
            stable=pp[pp.eligible&pp.fraction.ge(.8)]
            axes[2].scatter(stable.x,stable.y,s=72,facecolors='none',edgecolors='#dc3d35',lw=1.4,zorder=10)
            for _,r in stable.sort_values(['fraction','pair_members'],ascending=[False,True]).head(3).iterrows():
                axes[2].annotate(r.pair_members,(r.x,r.y),xytext=(3,4),textcoords='offset points',fontsize=7,zorder=11)
            cb=fig.colorbar(im,ax=axes[2],orientation='horizontal',fraction=.07,pad=.06);cb.set_label('0至1；红圈为频率≥0.8',fontsize=8)
            fig.suptitle(short[ds]+' · '+subject+'：不同记录中的局部高失配',fontsize=12)
            key='hotspot_'+ds;save(key,fig,'左、中为两条原生记录的相对失配图，右为该人全部可评价记录中的逐位置高四分位频率；红圈为持续热点。坐标采用头顶球面等距展开，虚线外的整头着色为外推。此病例按持续热点数、最高频率、重合率排序选择，不能当作独立验证。','hotspot_persistence.csv; hotspot_record_locations.csv; display_coordinates.csv')
            locations=stable.sort_values(['fraction','pair_members'],ascending=[False,True])
            locationrows=[[r.pair_members,f'{r.hotspot_records}/{r.available_records}',r.planned_records,value(r.fraction)] for _,r in locations.iterrows()]
            if not locationrows:locationrows=[['没有位置达到80%阈值','—','—','—']]
            content['sections'].append(section('持续热点病例 · '+short[ds]+' / '+subject,[
                figures[key],_table(['位置','高四分位/可评记录','计划记录','出现频率'],locationrows),
                _paragraph('候选分母为'+str(len(candidates))+'名可评价被试。两张记录图的窗口覆盖中位数分别为'+
                    '、'.join(f"{100*(full[full.record_unit.eq(r)].available_windows/full[full.record_unit.eq(r)].planned_windows).median():.0f}%" for r in records)+
                    '。同一位置每条记录只计一次；Visual的同步Probe已按采集合并。图中局部热点的持续性与整体地图相关是两个互补结论，来源仍需独立观测区分。')]))
            selected.append(dict(dataset=ds,subject=subject,candidate_subjects=len(candidates),records=records,
                selection='stable site count, maximum persistence, median pair overlap, then subject identity',
                stable_locations=locations[['pair_members','hotspot_records','available_records','planned_records','fraction']].to_dict('records')))
        (output/'hotspot_examples.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
    content['sections'].append(section('9. 双Hb残差高度同向，是强统计现象但不是独立的生理来源证据',[
        figures['07_residual_geometry'],_table(['数据集','残差O/R相关中位数','共同方向平方比','偏置平方比'],[[short[ds],*[value(median(shapes,'feature',f,ds)) for f in ('Hb_residual_OR_correlation','Hb_common_direction_fraction','Hb_bias_fraction')]] for ds in datasets]),
        _paragraph('四库的双Hb残差相关中位数约0.946–0.957；偏置平方只占较小部分，主要还有去均值动态。这个模式值得作为模型结构诊断，但不能把标准化共同方向直接叫作血容量、HbT、呼吸或系统血流成分。')]))
    content['sections'].append(section('10. 拟合几何本身就能产生完全同向的双通道残差',[
        figures['08_projection_counterexample'],dict(type='equation',latex=r'h=(1,-1)^\top,\quad \widehat y=h(h^\top h)^{-1}h^\top y,\quad r=y-\widehat y=\tfrac{y_1+y_2}{2}(1,1)^\top',caption='解析示意中的y为任意二维输入，h为允许方向；两个残差必然相同。该公式只提供归因反例，不声称Tak残差严格等于这个线性模型。'),
        _paragraph('因此，最稳妥的解释是模型在某些观测方向上持续留下结构化残差。独立血流来源、共享光学变化与当前模型可表达方向都可能贡献；现有结果尚不能给出它们的比例。')]))
    content['sections'].append(section('11. Hb重建较好时，当前EEG代理仍未提供一致的驱动解释',[
        figures['09_crossmodal'],_paragraph('Hb-only解的EEG读出与实际功率代理的形状相关中位数，四库均接近零。Hb重建、代理一致性、隐藏Hb预测和真实驱动准确性必须分开判断；这一结果直接限制把Hb拟合驱动称为已恢复的神经活动。'),
        _paragraph('可检验的竞争解释包括：六电极×五频带第一主成分没有保留相关神经信息；观测、时滞或血流映射过于受限；Hb包含未被此EEG代理反映的血流与测量成分。这些是推断，不是本轮完成的机制分离。')]))
    gainrows=[]
    for ds in datasets:
        for m,label in (('prediction_gain_over_zero','对零预测'),('prediction_gain_over_template','对训练模板')):
            g=tables['independent_prediction_subjects'];g=g[g.dataset.eq(ds)&g.metric.eq(m)]
            gainrows.append([short[ds],label,value(g.value.median()),f'{int((g.value>0).sum())}/{g.value.notna().sum()}'])
    content['sections'].append(section('12. 隐藏Hb预测提供另一条证据，不能用条件重建代替',[
        figures['10_independent_prediction'],_table(['数据集','比较','收益中位数','正收益/可用人'],gainrows),
        _paragraph('收益为基线误差减独立EEG-only误差。正收益只能说明在该冻结预测规则和支持下优于该基线；仍须结合错配EEG供体、事件结构及训练尺度解释，不直接证明生理驱动已识别。')]))
    content['sections'].append(section('13. Simultaneous任务差异主要见于绝对量，相对残差没有同样变化',[
        figures['11_task_contrast'],_paragraph('词生成相对n-back的D1、EEG读出误差和驱动回放差均增大；相对残差差值为−0.00094，95%区间−0.01080至0.00859，未检出明确差异。说明任务变化中的幅度/背景尺度值得优先核对，而不能只凭绝对误差上升宣称耦合变差。'),
        _paragraph('原作者说明任务顺序和长时间记录可能影响状态，且数据来自同一次访问。因此任务标签、记录顺序、疲劳和测量状态不能在本分析中完全分开。[Shin等，2018]')]))
    content['sections'].append(section('14. 眼电活动有关联，但清理收益未超出均值/尺度匹配对照',[
        figures['12_ocular_controls'],_paragraph('两库均有EOG活动与失配的时间关联；但EOG回归相对匹配对照的差，Single-Trial为0.0063［−0.0044，0.0175］、Simultaneous为−0.0015［−0.0104，0.0073］。两个区间均跨零，不能把相对原代理的改善单独归因于去除了眼电来源。'),
        _paragraph('Visual和REFED没有本流程声明的原生EOG参照，不用前额EEG替代眼电。现有证据支持代理幅度与观察条件敏感性，尚不支持一个统一眼电机制解释四库失配。')]))
    content['sections'].append(section('15. 生理机制的优先级：先解释稳定的测量位置差异',[
        _table(['候选解释','当前支持','关键反证或缺口','后续可区分的证据'],[
            ['位置/个体相关的血流与光学观测混合','Simultaneous后部三口径一致；相对热点去模板后稳定','没有短距离或外周参照；固定光路和尺度也会稳定','短源探距、血压/呼吸/CO₂或光耦合指标与热点联合分析'],
            ['统一动力学或观测关系过于受限','相对残差与弱平衡同时偏离；残差方向结构明显','当前仅固定模型，残差几何可制造同向性','在独立支持上比较模型扩展与隐藏目标预测'],
            ['EEG功率代理与血流驱动不匹配','四库形状相关低，独立预测提供直接对照','不能据此否定真实神经血管耦合','预先规定频带/空间代理，在训练内选择并留出比较'],
            ['任务引发自主神经或状态变化','两库EOG关联；任务绝对误差变化','相对误差及幅度对照削弱单一归因；顺序混杂','独立外周指标、随机任务顺序或重复访问']]),
        _paragraph('优先报告的发现是“Simultaneous存在稳定、偏后部的固定模型失配表型”，机制假设优先考虑位置/个体相关的血流—光学观测混合与模型限制。现有数据不能在浅表血流、皮层血流、光学耦合和动力学差异之间作唯一归因。'),
        _paragraph('Kirilina等的独立多模态研究说明头皮血管信号也可能产生个体化、跨记录稳定的图形；Yücel等说明短距离参照能帮助分离浅表贡献。这些文献支持候选机制与验证路径，不能把其因果结论移植为本数据已证实的机制。')]))
    # Four frozen median examples, each with independently checked D1 readout.
    diagnostics=[]
    for example in _json(analysis/'examples.json')['examples']:
        with np.load(example['array_path'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
        t=np.arange(len(a['target']))*.1;target=a['target']/a['sd'];pred=a['hb_only_prediction']/a['sd']
        mask=np.asarray(a['score_mask'],bool)
        if mask.ndim==1:mask=np.broadcast_to(mask[:,None],target.shape)
        fig,axes=plt.subplots(2,1,figsize=(9,4.5),sharex=True);errs=[];stats=[]
        for ax,c,name in zip(axes,(1,2),('HbO','HbR')):
            good=mask[:,c].copy();good[:50]=False;good[-1]=False;good&=np.isfinite(target[:,c])&np.isfinite(pred[:,c])
            error=float(np.sqrt(np.mean((target[good,c]-pred[good,c])**2)));corr=float(np.corrcoef(target[good,c],pred[good,c])[0,1]);errs.append(error)
            ax.plot(t,target[:,c],c='#202c39',lw=1.25,label='处理实测');ax.plot(t,pred[:,c],c='#2272ac',ls='--',lw=1.2,label='Hb-only重建')
            ax.axvspan(0,5,color='#eeeeee');ax.set(ylabel=name+' / 训练SD',title=f'RMS误差={error:.3f}；形状相关={corr:.3f}');ax.grid(alpha=.15)
            stats.append(dict(component=name,rmse=error,correlation=corr))
        recovered=float(np.sqrt(np.mean(np.square(errs))))
        if not np.isclose(recovered,example['D1'],rtol=1e-8,atol=1e-10):raise ValueError('Example D1 replay failed')
        axes[0].legend(ncol=2);axes[1].set_xlabel('目标窗口内时间 / 秒');fig.tight_layout()
        key='example_'+example['dataset'];save(key,fig,'每行一个色团；灰区为不评分的5秒参考段。各分量和各示例纵轴独立，图上数值来自完整有效数组。重建只看Hb，因此不能据此证明跨模态驱动准确。',example['array_path'])
        content['sections'].append(section('波形核验 · '+short[example['dataset']]+'：预定面板中位失配示例',[
            figures[key],_paragraph(f"{example['subject']} / {example['record']} / {example['pair']}，Hb窗起点{example['hb_start_s']:.1f}秒；原面板候选{example['candidate_denominator']}／计划{example['planned_panel_denominator']}窗对，按D1接近中位数选择。图中D1独立复算为{recovered:.4f}。"),
            _paragraph('同时看峰谷方向和RMS误差：相关高可以仍有幅度误差，误差小也可能来自目标幅度较小。这个示例帮助阅读失配形态，不用于估计总体频率或选模型。')]))
        diagnostics.append(dict(id=example['id'],D1=recovered,source_D1=example['D1'],components=stats))
    content['sections'].append(section('附录：失败分母、独立验证与证据来源',[
        _table(['数据集','视图','计划窗对','完成','数值失败','支持不足','异常/缺失'],failure_rows),
        _paragraph('同一窗对可以有Hb-only结果而没有EEG联合支持；各指标使用自己的有效分母，不把缺失填零。Visual修正后纯Hb逐被试端点差小于1e−9；分析表保留全部103人的计划分母。'),
        _paragraph('全部派生表与方法定义：'+str(analysis/'summary.json')),
        _paragraph('本报告的所有图与公式均嵌入PNG位图，正文与表格保持可检索；详细源码和数组路径保留在交付清单。')]))
    content['sections'].append(section('文献与机制推断的适用范围',[
        *[_paragraph(f'{name}。{description} {url}') for name,url,description in references],
        _paragraph('文献提供生理上可能的解释和区分办法；本报告的数值只来自项目保留证据。没有新的生理干预、外周参照或独立参数真值，因果归因保持未定。')]))
    (output/'example_diagnostics.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2)+'\n')
    (output/'references.json').write_text(json.dumps([dict(title=a,url=b,scope=c) for a,b,c in references],ensure_ascii=False,indent=2)+'\n')
    path=output/'report_content.json';path.write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n')
    checks=render_report(path,output/'report');shutil.copy2(Path(__file__),output/'renderer_source.py')
    (output/'source_manifest.json').write_text(json.dumps(dict(schema='tak_mismatch_fixed_pattern_report_v1',owning_analysis=str(analysis/'summary.json'),
        source_runs=[summary['sources']['reference_run'],summary['sources']['corrected_run']],figures=figures,
        bitmap_figures=True,formula_source='report_content.json',display_projection='display_projection.json',display_coordinates='display_coordinates.csv',
        raw_signal_reads=0,model_fits=0,shared_parameter_reads=0),ensure_ascii=False,indent=2)+'\n')
    return dict(output=str(output),pages=checks['exports']['TECHNICAL_REPORT.pdf']['pages'],figures=len(figures))


def render_disagreement_comparison(run, analysis, output):
    """All-subject paired comparison; figures are PNG and tables stay selectable."""
    import shutil
    import pandas as pd
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from src.analysis.tak_mismatch import HB_RATIO, DRIVE_RATIO, DISAGREEMENT_METRICS
    run,analysis,output=(Path(p).resolve() for p in (run,analysis,output))
    summary=_json(analysis/'summary.json')
    if (summary.get('schema')!='tak_mismatch_disagreement_comparison_v1' or summary.get('status')!='completed'
            or Path(summary['parent_run']).resolve()!=run or not analysis.is_relative_to(run)):
        raise ValueError('Completed versioned disagreement analysis required')
    if output.exists():raise FileExistsError('Use a fresh disagreement report export')
    output.mkdir(parents=True);plots=output/'plots';plots.mkdir()
    tables={name:pd.read_csv(analysis/path,dtype={'subject':str}) for name,path in summary['files'].items()}
    plt=_plotting()
    plt.rcParams.update({'font.size':11,'axes.titlesize':12,'xtick.labelsize':11,'ytick.labelsize':10})
    datasets=summary['config']['scope']['datasets']
    labels=dict(zip(datasets,('Single-Trial','Simultaneous','Visual','REFED')))
    metric_labels={'D1':'D1','D_drive_H':'Drive',HB_RATIO:'RH',DRIVE_RATIO:'RD','D_EH':'DEH'}
    colors={'D1':'#236fa4','D_drive_H':'#c65d31',HB_RATIO:'#26918b',DRIVE_RATIO:'#a64a82','D_EH':'#79716e'}
    shown=['D1','D_drive_H',HB_RATIO,DRIVE_RATIO]
    fmt=lambda x:f'{x:.3f}' if pd.notna(x) and np.isfinite(float(x)) else '不可用'
    def section(title,blocks):return dict(title=title,page_break=True,blocks=blocks)
    def table(columns,rows):return _table(columns,rows)
    def source(name):return str(analysis/summary['files'][name])
    figures={}
    def save(name,fig,caption,evidence):
        path=plots/(name+'.png');fig.savefig(path,dpi=250,bbox_inches='tight');plt.close(fig)
        block=dict(type='figure',path=str(path),caption=caption,source=source(evidence),preserve_bitmap=True)
        figures[name]=block;return block
    def points(name,frame,caption,evidence,limits=(-1.05,1.15)):
        fig,axes=plt.subplots(2,2,figsize=(8,5.4),sharey=True)
        for ax,ds in zip(axes.ravel(),datasets):
            for i,metric in enumerate(shown):
                g=frame[frame.dataset.eq(ds)&frame.metric.eq(metric)].sort_values('subject')
                v=g.value.dropna().to_numpy(float)
                ax.scatter(i+.13*np.sin(np.arange(len(v))*2.4),v,c=colors[metric],s=17,alpha=.72)
                if len(v):
                    q=np.quantile(v,[.25,.5,.75]);ax.plot([i,i],q[[0,2]],c='black',lw=2)
                    ax.scatter([i],[q[1]],facecolor='white',edgecolor='black',s=35,zorder=5)
                ax.text(i,limits[1]-.08,f'{len(v)}/{len(g)}',ha='center',fontsize=10)
            ax.set(title=labels[ds],xticks=range(4),xticklabels=[metric_labels[m] for m in shown],ylim=limits)
            ax.set_ylabel('热点重合率 − k/N' if name=='hotspot_excess' else '空间 Spearman ρ')
            ax.axhline(0,c='#999999',lw=.7);ax.grid(axis='y',alpha=.15)
        fig.tight_layout()
        return save(name,fig,caption,evidence)
    repeated=tables['spatial_repeat_subjects'];paired=tables['paired_stability_tests']
    primary=paired[paired.endpoint.eq('centered_rho')]
    decisions=[]
    for contrast in ('drive_minus_Hb','normalized_drive_minus_Hb'):
        group=primary[primary.contrast.eq(contrast)]
        decisions.append(dict(contrast=contrast,drive_favors=int(((group['mean']>0)&group.q_value.lt(.05)).sum()),
            hb_favors=int(((group['mean']<0)&group.q_value.lt(.05)).sum()),uncertain=int(group.q_value.ge(.05).sum())))
    content=dict(schema='source_diagnostics_report_v1',title='EEG–Hb驱动分歧与Hb重建失配：全被试配对比较',
        subtitle='重复性、持续热点、幅度对照与论文解释边界',date=datetime.now(ZoneInfo('Asia/Hong_Kong')).date().isoformat(),
        provenance=[f"沿用{summary['planned_subjects']}名被试、{summary['planned_windows']}个计划窗口×Hb位置。Visual只使用更正后的原生EEG身份。",
            '本次只重汇总保留标量与位置；没有新读取原始信号，没有重拟合模型，没有新增受保护评价。'],
        summary=[
            '核心问题是哪个分歧更能重复描述个体空间结构。误差较大或检验p值较小，本身不能证明生理来源更明确。',
            '论文建议：若主线是个体血流模型相容性，以Hb重建失配为主，联合呈现D1与R_H；驱动分歧作为跨模态代理一致性的次要分析。REFED的反向结果及严格覆盖损失必须保留。',
            f"主端点为去群体模板后的跨记录空间相关。原尺度比较中，驱动分歧显著更高{decisions[0]['drive_favors']}库，Hb失配显著更高{decisions[0]['hb_favors']}库；共同幅度归一化后分别为{decisions[1]['drive_favors']}库、{decisions[1]['hb_favors']}库（32项配对检验BH校正）。",
            'REFED是重要反例：原尺度Hb误差更稳定，但共同幅度归一化后驱动分歧更稳定；此时两者的空间重复性均低于前三库的主要Hb结果。不能宣称一个指标在四库全面胜出。',
            '所有方法先匹配窗口，再比较相同记录对；不把窗口、通道或记录对当作独立被试。负结果与不可用分母保留。'],sections=[],sources=[str(analysis/'summary.json')])
    sections=content['sections']
    coverage=[];support=tables['window_support']
    for ds in datasets:
        g=support[support.dataset.eq(ds)]
        coverage.append([labels[ds],g.subject.nunique(),len(g),int(g.D1_available.sum()),int(g.D_drive_H_available.sum()),int(g.common_available.sum())])
    sections.append(section('1. 同一批观测上的比较及覆盖损失',[
        table(['数据集','计划人','计划窗对','Hb失配可用','驱动回放可用','共同窗对'],coverage),
        _paragraph('一窗对为一个30秒目标窗×一个HbO/HbR测点。共同窗要求D1、Drive、DEH均有限，且同窗零预测MSE为正；先把其余窗标为不可用，再聚合全部指标。只有Hb拟合而无EEG或回放成功的窗不进入配对比较，仍保留在计划分母。'),
        _paragraph('主地图要求每个物理成员至少80%计划窗口可用；记录对要求至少8个且80%共同测点。早晚每半至少2个有效窗口。Visual的同步Probe按采集合并；不能增加独立样本量。'),
        _paragraph('跨记录含义：Single-Trial只比较同任务记录，Simultaneous跨任务，Visual同次访问跨Part，REFED跨视频；后三者不称为同条件跨日重测。'),
        _paragraph('来源：'+source('window_support'))]))
    sections.append(section('2. 两种分歧的计算路径不同',[
        dict(type='equation',latex=r'D_1=\sqrt{\left\langle\frac{(H-\widehat H_H)^2}{\sigma_H^2}\right\rangle},\qquad D_{\mathrm{drive},H}=\sqrt{\left\langle\frac{[\mathcal G(r_E;x_{\rm rest})-\mathcal G(r_H;x_{\rm rest})]^2}{\sigma_H^2}\right\rangle}',
             caption='H为处理后并扣除窗首5秒参考的双Hb；σ_H为冻结训练SD。平均在5.0–29.8秒的249个有效时刻及两个色团上计算。'),
        _paragraph('D1：Hb-only拟合看到当前窗口双Hb，估计驱动与四个初态；比较处理观测与该条件重建。其余血流参数固定。它度量模型对当前Hb的描述误差。'),
        _paragraph('Drive：EEG-only和Hb-only分别估计驱动；将两个驱动输入相同固定模型G、相同静息初态，再比较双Hb回放。驱动保留拟合正则化所选择的直流水平，不等于直接的去均值驱动差，也不是独立Hb预测误差。'),
        _paragraph('EEG目标是六个近邻电极×五频带对数功率的训练折第一主成分。当前窗首5秒参考，90秒原生上下文用于EEG处理；Hb来自既有离线滤波缓存。“原始Hb与重建”在本报告中具体指处理观测，不是未经处理的光强或绝对浓度。'),
        dict(type='equation',latex=r'R_H=\frac{\langle (H-\widehat H_H)^2/\sigma_H^2\rangle}{\langle H^2/\sigma_H^2\rangle},\qquad R_D=\frac{\langle[\mathcal G(r_E)-\mathcal G(r_H)]^2/\sigma_H^2\rangle}{\langle H^2/\sigma_H^2\rangle}',
             caption='图表缩写RH和RD分别对应上述两个共同分母比值；Drive对应D_drive,H。比值在窗口平方量、记录等权聚合后形成，不是先平均窗口比值。'),
        _paragraph('DEH是Hb-only解的EEG读出与实测EEG代理之间的RMS差，以训练EEG SD计，仅作辅助。D1和Drive虽同用Hb SD，误差对象和初态不同；不能把数值更大解释为某方法更差或某生理成分更多。')]))
    sections.append(section('3. 统计判据：先比较稳定性，再讨论机制',[
        _paragraph('主端点为去模板跨记录空间Spearman相关：在每条记录、每个位置减去其他被试同位置中位数，再比较本人的两张地图。每人对可用记录对取中位数。模板固定于当前证据；区间不传播模板估计不确定性。'),
        _paragraph('次端点为原地图相关、记录内早晚相关与热点重合超额。热点取共同位置中最高ceil(N/4)个；重合率为交集数/k，超额为重合率减k/N。k/N仅为等大小随机集合参考，不是空间独立性显著性检验。'),
        _paragraph('每个方法对的比较先限制到双方均有效的相同记录对，再分别取被试中位数并作“驱动−Hb”差。平均被试差为推断对象；95%区间为5000次整被试自助法，双侧符号置换在≤16人时穷举，其余100000次加一校正。'),
        _paragraph('四库×四个稳定性端点×原尺度/共同幅度分母两种比较，计划32项构成同一BH家族，可检验项统一校正，不可用项保留。独立预测基线和区域探索各自另成完整家族。没有测试“非负误差大于零”来替代生理证据。'),
        _paragraph('当前为看过既有报告之后的探索性分析；名义区间与本轮BH不覆盖历史探索或后续择优选入论文。稳定性只说明可重复描述，不能区分固定测量链、个体生理和模型/代理限制。')]))
    for endpoint,title in [('centered_rho','4. 主端点：去群体模板后的个体地图'),('raw_rho','5. 原空间地图与测点排序'),('hotspot_excess','6. 反复高分歧位置的重合')]:
        frame=repeated[repeated.endpoint.eq(endpoint)]
        limits=(-.3,.85) if endpoint=='hotspot_excess' else (-1.05,1.15)
        figure=points(endpoint,frame,'每点一名被试，白点为被试中位数，黑线为四分位距；上方为可用/计划人数。D1与Drive为原尺度，RH与RD为共同目标幅度分母对照。横轴按固定方法顺序，纵轴所有数据集共用。','spatial_repeat_subjects',limits)
        rows=[]
        for ds in datasets:
            row=[labels[ds]]
            for m in shown:
                v=frame[frame.dataset.eq(ds)&frame.metric.eq(m)].value
                row.append(f'{fmt(v.median())} ({v.notna().sum()}/{len(v)})')
            rows.append(row)
        caption='数值为被试中位数；括号为可用/计划人数。'
        sections.append(section(title,[figure,table(['数据集','D1','Drive','RH','RD'],rows),_paragraph(caption),
            _paragraph('原地图和去模板地图一起判断：群体共有测点图形可能提高原相关。去模板仍保留个体固定光路、训练尺度与传感器条件。热点超额为描述量，未把相邻测点当独立随机位置进行检验。')]))
    early=tables['early_late_subjects']
    sections.append(section('7. 同一记录内的早晚空间一致性',[
        points('early_late',early,'每点为一人：先对同一记录前后半段计算地图相关，再取该人的记录中位数。使用原始计划时间划分，不因失败而移动切分点；奇数个窗口时省略中间窗。','early_late_subjects'),
        _paragraph('这里衡量空间排序是否在记录内持续，和“跨被试误差大小的早晚相关”不是同一统计量。短记录若每半不足2个有效窗，则该记录不可用；全体被试与计划记录分母保留。')]))
    for contrast,title in [('drive_minus_Hb','8. 原尺度配对检验：驱动分歧是否更稳定'),('normalized_drive_minus_Hb','9. 共同幅度分母后的配对检验')]:
        g=paired[paired.contrast.eq(contrast)]
        endpoint_labels={'centered_rho':'去模板相关','raw_rho':'原相关','early_late_rho':'早晚相关','hotspot_excess':'热点超额'}
        rows=[[labels[r.dataset],endpoint_labels[r.endpoint],f'{r.subjects}/{r.planned_subjects}',fmt(r.mean),f'[{fmt(r.ci_low)}, {fmt(r.ci_high)}]',f'{r.q_value:.4g}' if pd.notna(r.q_value) else '不可用'] for r in g.itertuples()]
        sections.append(section(title,[table(['数据集','端点','配对人','平均差','95%区间','BH q'],rows),
            _paragraph('差值方向为驱动指标减Hb重建指标，正值表示驱动分歧的空间重复性更高；不是误差大小或生理正确率的差。区间跨零或q≥0.05时，当前证据不足以判定优势。所有比较共用32项BH家族。'),
            _paragraph('来源：'+source('paired_stability_tests'))]))
    relaxed_repeat=tables['relaxed_spatial_repeat_subjects']
    relaxed_center=relaxed_repeat[relaxed_repeat.endpoint.eq('centered_rho')]
    rows=[]
    for ds in datasets:
        row=[labels[ds]]
        for metric in shown:
            v=relaxed_center[relaxed_center.dataset.eq(ds)&relaxed_center.metric.eq(metric)].value
            row.append(f'{fmt(v.median())} ({v.notna().sum()}/{len(v)})')
        rows.append(row)
    sections.append(section('9A. 沿用原报告时间覆盖规则的敏感性结果',[
        points('relaxed_centered',relaxed_center,'仍匹配相同窗口与相同记录对，但允许由部分有效时间计算全记录地图；共同测点仍要求至少8个且80%。各点和分母定义与主分析相同。','relaxed_spatial_repeat_subjects'),
        table(['数据集','D1','Drive','RH','RD'],rows),
        _paragraph('严格80%窗口门槛下，Simultaneous仅1/26人有原地图配对，去群体模板后为0/26，无法检验该主端点。保留这一不可评价结果，同时使用预设的宽松覆盖敏感性检验原报告的范围；没有补齐缺失时间，没有改变任何观测值或拟合。'),
        _paragraph('宽松结果不能替代严格结果证明完整时间覆盖；它回答的是“可用时间片段汇总得到的地图是否重复”。')]))
    if 'relaxed_paired_stability_tests' in tables:
        g=tables['relaxed_paired_stability_tests'];rows=[]
        for r in g[g.endpoint.eq('centered_rho')].itertuples():
            rows.append([labels[r.dataset],'RD−RH' if r.contrast.startswith('normalized') else 'Drive−D1',
                f'{r.subjects}/{r.planned_subjects}',fmt(r.mean),f'[{fmt(r.ci_low)}, {fmt(r.ci_high)}]',f'{r.q_value:.4g}' if pd.notna(r.q_value) else '不可用'])
        sections.append(section('9B. 宽松覆盖下的配对比较仍保留独立检验家族',[
            table(['数据集','比较','人','平均差','95%区间','BH q'],rows),
            _paragraph('完整32项宽松覆盖检验另成一个BH家族，本页只展示事先定义的去模板主端点。补充计算在看到严格覆盖损失后开展，保留v1严格分析和全部数值，并独立复现原宽松被试统计；这是探索性敏感性分析。'),
            _paragraph('来源：'+source('relaxed_paired_stability_tests'))]))
    hot=tables['hotspot_subjects'];relaxed=tables['relaxed_hotspot_subjects'];rows=[]
    for ds in datasets:
        for m in shown:
            g=hot[hot.dataset.eq(ds)&hot.metric.eq(m)];r=relaxed[relaxed.dataset.eq(ds)&relaxed.metric.eq(m)]
            rows.append([labels[ds],metric_labels[m],f'{(g.stable_locations>0).sum()}/{(g.eligible_locations>0).sum()}/{len(g)}',
                fmt(g.loc[g.eligible_locations>0,'stable_locations'].median()),f'{(r.stable_locations>0).sum()}/{(r.eligible_locations>0).sum()}/{len(r)}'])
    sections.append(section('10. 持续热点必须同时报告可评价分母',[
        table(['数据集','指标','持续/可评/计划人','可评者热点数中位','宽松持续/可评/计划'],rows),
        _paragraph('持续热点：同一位置至少在80%可评价记录中进入该人的最高四分之一，同时至少2条且80%的计划记录可评价。主分析要求每条记录至少80%计划窗口支持；宽松列允许任意有限全记录值，只用于覆盖敏感性。'),
        _paragraph('不能把可评价子集的高频热点比例推广到所有计划被试；没有足够记录的被试不等于没有热点。并列截点按固定位置顺序处理，完整并列频率保留在逐位置表中。')]))
    repeat=tables['repeat_summary'];rows=[]
    task_names={'mental_arithmetic':'心算','motor_imagery':'运动想象','visual_cognitive_motivation':'视觉任务'}
    for r in repeat[repeat.metric.isin(shown)&repeat.status.eq('available')].itertuples():
        rows.append([labels[r.dataset],task_names.get(r.task_label,r.task_label),metric_labels[r.metric],r.subjects,fmt(r.spearman_rho),f'[{fmt(r.rho_ci_low)}, {fmt(r.rho_ci_high)}]'])
    sections.append(section('11. 被试差异大小的重复性与空间稳定性分开解释',[
        table(['数据集','任务','指标','人','重复ρ','95%区间'],rows),
        _paragraph('每人先汇总最早、最晚同任务记录的共同测点，再在被试之间计算Spearman相关；Visual仅同次访问跨Part。相关高表示“第一次分歧较大的人第二次也较大”，不是地图相同、数值相等或波形复现。'),
        _paragraph('两个候选指标的重复相关差，另以同一被试的四个值一起重采样，保留在repeat_metric_comparison.csv.gz；这是点式探索区间，未再解释为校正后的优胜检验。Simultaneous没有原生同任务独立重复；REFED不同视频不替代同条件重测。')]))
    assoc=tables['association_subjects'];rows=[]
    for ds in datasets:
        for other in ('zero_NMSE','D_bal_40s'):
            row=[labels[ds],'目标幅度' if other=='zero_NMSE' else '40秒弱平衡']
            for m in shown:
                v=assoc[assoc.dataset.eq(ds)&assoc['first'].eq(m)&assoc['second'].eq(other)].value
                row.append(f'{fmt(v.median())} ({v.notna().sum()}/{len(v)})')
            rows.append(row)
    sections.append(section('12. 空间模式与幅度、模型平衡偏离的关系',[
        table(['数据集','对照变量','D1','Drive','RH','RD'],rows),
        _paragraph('每条记录在共同测点上计算空间Spearman相关，再取被试记录中位数，表中再取被试中位数。括号为可用/计划人。目标幅度是同窗零预测MSE；弱平衡是保留的小信号近似、整记录40秒核统计，时间支持不同，只作辅助；并非精确非线性脱氧Hb平衡残差。'),
        _paragraph('较强幅度相关提示误差排序可能部分反映信号幅值或尺度。共同幅度分母可减弱这一路径，但小分母也可能放大不稳定性。与弱平衡相关表示两种模型失配描述在空间上同向，仍共享Hb观测和模型假设，不能充当独立生理验证。')]))
    baseline=tables['prediction_baseline_tests'];rows=[]
    for r in baseline.itertuples():
        base=r.endpoint.replace('EEG_prediction_gain_over_','')
        rows.append([labels[r.dataset],{'zero':'零变化','template':'训练模板','wrong':'错配EEG'}.get(base,base),f'{r.subjects}/{r.planned_subjects}',fmt(r.mean),f'[{fmt(r.ci_low)}, {fmt(r.ci_high)}]',f'{r.q_value:.4g}' if pd.notna(r.q_value) else '不可用'])
    sections.append(section('13. EEG分歧的下游含义由隐藏Hb预测核对',[
        table(['数据集','基线','人','基线−EEG MSE','95%区间','BH q'],rows),
        _paragraph('四库相对零变化预测的平均收益均为负，约−0.110至−0.153训练Hb SD平方；当前EEG-only预测在这组共同支持上均劣于零基线。相对错误供体也没有在本轮完整基线家族内获得显著优势。'),
        _paragraph('预测时当前窗口Hb隐藏，EEG-only单独拟合；上表正值有利于EEG预测。zero为零变化预测，template为训练折模板，wrong为原协议中同被试同条件、实际滤波支持不重叠的EEG供体。每种比较单独匹配有效窗口，且限制在本轮共同支持内。'),
        _paragraph('单位为冻结训练Hb SD平方。先窗口平方量，再原生记录、被试等权；计划12项基线比较，11项可检验项统一BH，REFED错误供体不可用。Drive大并不自动意味着EEG有可用的跨模态预测信息，也不代表Hb反演驱动是真实神经活动。')]))
    regions=tables['regional_tests'];rows=[]
    for r in regions.itertuples():
        rows.append([labels[r.dataset],metric_labels[r.endpoint],f'{r.subjects}/{r.planned_subjects}',fmt(r.mean),f'[{fmt(r.ci_low)}, {fmt(r.ci_high)}]',f'{r.q_value:.4g}' if pd.notna(r.q_value) else '不可用'])
    sections.append(section('14. 区域方向作为辅助，不替代个体重复性',[
        table(['数据集','指标','人','后部−前额','95%区间','BH q'],rows),
        _paragraph('仅对Single-Trial和Simultaneous已有的前额、中央、后部前缀分区计算；要求三个区域完整，先区域位置中位数，再作被试内差。十项区域检验共一BH家族。它是测点区分，不能等同于个体皮层定位；跨指标的原始差值量纲不同。')]))
    ocular=tables['retained_ocular_associations'];rows=[]
    for r in ocular[ocular.analysis.eq('cross_window_partial')&ocular.endpoint.isin(['D1','D_drive_H'])].itertuples():
        rows.append([labels[r.dataset],metric_labels[r.endpoint],r.subjects,fmt(r.mean),
            f'[{fmt(r.ci_low)}, {fmt(r.ci_high)}]',f'{r.q_value:.4g}'])
    sections.append(section('14A. 眼电对照限制“生理解释更清楚”的说法',[
        table(['数据集','指标','人','偏秩相关均值','95%区间','原BH q'],rows),
        _paragraph('这是保留的三测点面板结果：同一记录跨窗口计算眼电RMS与失配平方量的秩关联，并控制EEG代理与Hb目标幅度；记录等权，再被试等权。这里不是空间热点的眼电回归，也不是本轮共同窗的重新估计。'),
        _paragraph('D1仍有眼电相关性，Simultaneous尤其明显；不能因为它更稳定就称其更纯粹或已定位生理来源。Drive的关联较小也不能单独证明它更接近神经活动。两列未做新的配对眼电效应差检验，不能用两列p值大小给方法排序。'),
        _paragraph('R_H并未在这份保留眼电面板中完成同样控制，不能将D1或Drive的眼电结论直接套用给比值指标。原反事实中清理EEG相对原代理有收益，但相对均值/SD匹配对照的区间跨零，尚不能归为眼源去除的独有贡献。')]))
    rows=[];shapes=tables['retained_shape_subjects']
    for ds in datasets:
        def median_metric(metric,endpoint):
            v=relaxed_repeat[relaxed_repeat.dataset.eq(ds)&relaxed_repeat.metric.eq(metric)&relaxed_repeat.endpoint.eq(endpoint)].value
            return f'{fmt(v.median())} ({v.notna().sum()}/{len(v)})'
        shape=shapes[shapes.dataset.eq(ds)&shapes.feature.eq('drive_E_H_correlation')].value
        rows.append([labels[ds],median_metric('D_EH','raw_rho'),median_metric('D_EH','centered_rho'),
            median_metric('D_drive_H','centered_rho'),fmt(shape.median())])
    sections.append(section('14B. 辅助EEG读出误差与直接驱动形态',[
        table(['数据集','DEH原空间ρ','DEH去模板ρ','Drive去模板ρ','直接驱动形态ρ'],rows),
        _paragraph('前三列沿用宽松覆盖的全被试地图，括号为可用/计划人。DEH是Hb解的EEG读出与功率代理之间的差；它可以有重复的空间分布，而两条驱动在窗口内仍几乎不相关。'),
        _paragraph('最后一列来自原三测点面板：每窗在评分支持去均值后计算EEG-only与Hb-only驱动的Pearson相关，先窗口、记录、被试汇总，再取被试中位数。它没有覆盖全通道波形，不能被当作全被试空间相关的同一端点。'),
        _paragraph('低驱动形态相关是当前代理与条件模型的一致性限制；没有真实神经驱动测量，无法单凭此项判断哪条驱动正确。')]))
    sections.append(section('15. 论文选择应对应可证明的科学问题',[
        table(['判断维度','Hb观测−重建失配','EEG−Hb驱动分歧'],[
            ['直接回答','当前固定模型能否描述处理Hb','当前EEG代理与Hb条件驱动是否相容'],
            ['额外依赖','Hb处理、模型、驱动/初态正则化','另加EEG频带/PCA/空间代理、两次拟合与回放初态'],
            ['可支持表述','可重复的模型失配空间表型','可重复的跨模态代理不一致'],
            ['不能直接支持','异常生理来源或病理定位','真实神经血管解耦或神经驱动真值']]),
        _paragraph('若论文主张是“稳定的个体血流模型失配”，应优先选择去模板后仍稳定、共同幅度对照仍成立、覆盖充分的Hb指标。若主张是“当前EEG代理不足以解释Hb驱动”，则驱动分歧可作为独立问题或次要分析，并同时报告隐藏Hb预测与代理/眼电敏感性。'),
        _paragraph('依据本轮结果，更合适的论文安排是以Hb模型失配为主要分析，同时报告D1与R_H，避免仅靠原尺度稳定性；驱动分歧放在跨模态一致性的次要分析。REFED中R_H稳定性弱、归一化驱动指标相对更高的结果必须保留，不能将前三库叙述外推为普遍规律。'),
        _paragraph('R_H与弱平衡偏离的空间关系比归一化驱动差更一致，因而“模型相容性不足”的解释更集中；但二者共享Hb与模型假设，这仅支持解释的一致性，不证明某个生理过程或脑区异常。'),
        _paragraph('本轮可比较统计稳健性，不能仅凭同一批数据决定哪个指标更接近真实生理机制。保留眼电分析来自预先冻结的三位置面板，且只有部分数据集；相关、幅度匹配控制和回归敏感性均不能完成因果分离。'),
        _paragraph('最终正文应公开指标选择发生在本次探索之后；避免选择单个优势数据集、单个热点或最小p值作为确认性结论。REFED与Visual的覆盖和条件差异应作为结果呈现。')]))
    sections.append(section('附录：证据、复现和方法限制',[
        _paragraph('Owning evidence：'+str(analysis/'summary.json')),
        _paragraph('summary.json的files字段明确定位每张数值表：v1目录保留完整配对分析与四库分区，v2目录保留补充覆盖检验并引用原表；各版本的resolved_config.json固定口径，source保存执行源码，execution.log和launch.json记录独立后台服务。报告只汇总这些事实。'),
        _paragraph('独立复核见v2目录verification_v2.json：逐窗共同掩码、618个被试级指标、70个记录地图/热点样本、2060个宽松被试端点和完整BH家族已复算通过；未读取原始信号。'),
        _paragraph('这里没有重读受保护数据，没有重拟合或新增干预，也没有给生理解释打一个未经验证的综合分数。失败支持与不可评价人数不能视作零失配。'),
        _paragraph('模型依据：Tak等，2015，Dynamic causal modelling for functional near-infrared spectroscopy，https://www.fil.ion.ucl.ac.uk/~wpenny/publications/tak-penny15.pdf 。光学混合和非皮层贡献仍是解释边界；原报告的机制文献不构成本轮因果证据。')]))
    sections[-1]['blocks'].extend([
        _paragraph('Kirilina等，2012，https://pmc.ncbi.nlm.nih.gov/articles/PMC3348501/ ：独立多模态测量支持头皮血流可贡献任务相关fNIRS变化，不能据此认定本研究热点来自头皮。'),
        _paragraph('Yücel等，2015，https://pmc.ncbi.nlm.nih.gov/articles/PMC4717232/ ：短距离回归可改善不同自主反应任务的血流响应定位，提示需要独立浅表参照；本轮没有此项干预。')])
    path=output/'report_content.json';path.write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n')
    checks=render_report(path,output/'report')
    shutil.copy2(Path(__file__),output/'renderer_source.py')
    (output/'source_manifest.json').write_text(json.dumps(dict(schema='tak_disagreement_report_v1',owning_analysis=str(analysis/'summary.json'),
        figures=figures,bitmap_figures=True,raw_signal_reads=0,model_fits=0),ensure_ascii=False,indent=2)+'\n')
    return dict(output=str(output),pages=checks['exports']['TECHNICAL_REPORT.pdf']['pages'],figures=len(figures))


def render_four_dataset_patterns(run, analysis, output):
    """Four-dataset synthesis using the corrected Visual numerical owner."""
    import shutil
    import pandas as pd
    from datetime import datetime, timezone
    from src.analysis.tak_mismatch import HB_RATIO
    run, analysis, output = (Path(p).resolve() for p in (run, analysis, output))
    summary = _json(analysis/'summary.json')
    if summary.get('schema')=='tak_mismatch_disagreement_comparison_v1':
        return render_disagreement_comparison(run,analysis,output)
    if summary.get('schema')=='tak_mismatch_four_dataset_fixed_patterns_v1':
        return render_four_dataset_fixed_patterns(run,analysis,output)
    if (summary.get('schema') != 'tak_mismatch_four_dataset_patterns_v1'
            or summary.get('status') != 'completed' or Path(summary['parent_run']).resolve() != run
            or not analysis.is_relative_to(run)):
        raise ValueError('Completed four-dataset corrected-Visual analysis required')
    if output.exists():
        raise FileExistsError('Four-dataset report requires a new versioned export')
    tables = {k:pd.read_csv(analysis/v, dtype={'subject':str}) for k,v in summary['files'].items() if v.endswith('.csv')}
    sources = summary['sources']; visual = 'visual_cognitive_motivation'
    old = pd.read_csv(sources['reference_maps'], dtype={'subject':str})
    new = pd.read_csv(sources['corrected_maps'], dtype={'subject':str})
    maps = pd.concat([old[~old.dataset.eq(visual)], new], ignore_index=True)
    old_plan, new_plan = (_json(Path(sources[k])) for k in ('reference_plan','corrected_plan'))
    plan = dict(records=[r for r in old_plan['records'] if r['dataset'] != visual]+new_plan['records'])
    examples = _json(analysis/summary['files']['examples'])['examples']
    datasets = ('eeg_fnirs_single_trial','simultaneous_eeg_nirs',visual,'refed')
    short = dict(zip(datasets, ('Single-Trial','Simultaneous','Visual','REFED')))
    colors = dict(zip(datasets, ('#2272ac','#d17924','#27816d','#9c4b90')))
    output.mkdir(parents=True); plots = output/'plots'; plots.mkdir()
    plt = _plotting(); figures = {}
    def save(key, fig, caption, source):
        figures[key] = dict(type='figure', path=_save(fig,plots/(key+'.png')),
                            caption=caption,source=source,preserve_bitmap=True)
    def dotplot(ax, frame, value='value', label='', limits=None):
        for i, dataset in enumerate(datasets):
            data = frame[frame.dataset.eq(dataset)].sort_values('subject')
            values = pd.to_numeric(data[value],errors='coerce').dropna().to_numpy()
            jitter = .16*np.sin(np.arange(len(values))*2.399963)
            ax.scatter(i+jitter,values,s=19,color=colors[dataset],alpha=.65,edgecolors='none')
            if len(values):
                q = np.quantile(values,[.25,.5,.75]);ax.plot([i,i],[q[0],q[2]],color='black',lw=2)
                ax.scatter([i],[q[1]],s=36,facecolor='white',edgecolor='black',zorder=5)
            ax.text(i,.98,f'{len(values)}/{len(data)}',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
        ax.set(xticks=range(4),xticklabels=[short[d] for d in datasets],ylabel=label,xlim=(-.45,3.45))
        ax.tick_params(axis='x',labelsize=8,rotation=15);ax.grid(axis='y',alpha=.16)
        if limits is not None:ax.set_ylim(*limits)

    subjects=tables['subject_values']; shapes=tables['shape_subjects']
    fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    for ax,(metric,label) in zip(axes,(('D1','Hb残差 RMS / 训练SD'),(HB_RATIO,'Hb残差 / 零预测 MSE'),('D_bal_20s','20秒弱平衡偏离'))):
        dotplot(ax,subjects[subjects.metric.eq(metric)],label=label)
        if metric==HB_RATIO:ax.axhline(1,color='grey',ls='--',lw=.8)
    fig.tight_layout();save('01_hb_compatibility',fig,
        '每点为一名被试；白心点与竖线为被试中位数和四分位距。顶部为可用／计划人数。三个面板单位与纵轴独立，不能比较颜色深浅或直接相加。比值1为零变化预测基线。','subject_values.csv')
    fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    for ax,(feature,label,limits) in zip(axes,(
        ('Hb_common_direction_fraction','双Hb残差共同方向平方占比',(-.03,1.12)),
        ('Hb_residual_OR_correlation','HbO/HbR去均值残差相关',(-1.05,1.17)),
        ('Hb_bias_fraction','残差中偏置平方占比',(-.03,1.12)))):
        dotplot(ax,shapes[shapes.feature.eq(feature)],label=label,limits=limits)
    fig.tight_layout();save('02_residual_directions',fig,
        '每点为三通道面板的被试汇总；先在记录内合并残差矩，再按记录等权，最后计算比值或相关。共同方向来自训练SD标准化的双Hb残差，不等于HbT，也不代表系统性生理成分的百分比。','shape_subjects.csv')
    fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    dotplot(axes[0],subjects[subjects.metric.eq('D_EH')],label='Hb-only读出的EEG误差 / 训练SD')
    dotplot(axes[1],shapes[shapes.feature.eq('EEG_Hb_readout_correlation')],label='EEG代理与Hb-only读出的相关',limits=(-1.05,1.17))
    dotplot(axes[2],subjects[subjects.metric.eq('D_drive_H')],label='共同初态驱动回放差 / Hb训练SD')
    fig.tight_layout();save('03_crossmodal_disagreement',fig,
        '左、右使用固定参数全通道标量；中间使用预定三通道波形面板。有符号相关与标准化误差分别描述形状和数值差；没有把低Hb误差当作驱动恢复正确的证明。','subject_values.csv; shape_subjects.csv')
    independent=tables['independent_prediction_subjects']
    fig,axes=plt.subplots(1,3,figsize=(10,3.7))
    for ax,metric,label in zip(axes[:2],('prediction_gain_over_zero','prediction_gain_over_template'),
        ('零变化预测MSE 减 EEG-only MSE','训练模板MSE 减 EEG-only MSE')):
        dotplot(ax,independent[independent.metric.eq(metric)],label=label)
        ax.axhline(0,color='black',ls='--',lw=.8)
    dotplot(axes[2],subjects[subjects.metric.eq('Delta_spatial')],label='近邻MSE 减 预选邻域MSE')
    axes[2].axhline(0,color='black',ls='--',lw=.8);fig.tight_layout()
    save('08_independent_prediction',fig,
        '每点为一名被试，单位均为Hb训练SD平方；纵轴各自独立。正值分别有利于EEG-only或预先选择的替代六通道。当前窗口Hb在EEG-only拟合与初始化中隐藏；模板、PCA、尺度和空间候选选择来自训练被试。每个差值先在同有效窗口计算，再按记录及被试汇总。','independent_prediction_subjects.csv; subject_values.csv')

    coordinate_cache={}
    def display(frame,dataset,subject):
        key=(dataset,subject)
        if key not in coordinate_cache:
            template=maps[maps.dataset.eq(dataset)&maps.subject.eq(subject)&maps.period.eq('subject_mean')&maps.metric.eq('D1')]
            coordinate_cache[key]=_spatial_preview_coordinates(template,plan).set_index('pair_members')
        positions=coordinate_cache[key].loc[frame.pair_members]
        if not np.allclose(frame[['x','y']],positions[['x_orthographic','y_orthographic']],atol=1e-7):
            raise ValueError('Four-dataset map does not match its retained geometry')
        frame=frame.copy();frame[['x','y']]=positions[['x','y']].to_numpy();return frame
    population=tables['population_maps']
    fig,axes=plt.subplots(2,4,figsize=(10,6.1));fig.subplots_adjust(left=.02,right=.89,top=.94,bottom=.08,hspace=.25,wspace=.08)
    for row,(metric,label) in enumerate(((HB_RATIO,'Hb残差 / 零预测MSE'),('D_EH','EEG读出误差 / 训练SD'))):
        p=population[population.metric.eq(metric)];maximum=max(float(p.display_value.max()),1e-6)
        for col,dataset in enumerate(datasets):
            reference=maps[maps.dataset.eq(dataset)&maps.period.eq('subject_mean')&maps.metric.eq('D1')].subject.sort_values().iloc[0]
            frame=display(p[p.dataset.eq(dataset)],dataset,reference).drop(columns=['period'],errors='ignore')
            low,high=int(frame.available_subjects.min()),int(frame.available_subjects.max())
            support=str(low) if low==high else f'{low}–{high}'
            im=spatial_topomap(axes[row,col],frame,limits=(0,maximum),coverage='head',
                title=short[dataset]+f' · 每点{support}人')
        pos=axes[row,-1].get_position();bar=fig.colorbar(im,cax=fig.add_axes([.925,pos.y0+.02,.018,pos.height-.04]));bar.set_label(label,fontsize=9)
    fig.text(.47,.02,'每个测点为该位置的被试中位数；每行同色标。虚线外着色为显示外推，未增加测量覆盖。',ha='center',fontsize=8.5)
    save('04_population_maps',fig,'三维模板按从头顶的极角展开，REFED保持登记平面布局。黑边点为实际测量位置，虚线为测点凸包；颜色只在同一行可比较。各库空间覆盖和训练尺度不同，地图不是个体皮层定位。','population_maps.csv')
    spatial=tables['spatial_subjects']
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for ax,(endpoint,label) in zip(axes,(('spatial_contrast','相对Hb残差的空间反差'),('early_late_rho','相对Hb残差早晚空间 Spearman ρ'))):
        frame=spatial[spatial.metric.eq(HB_RATIO)&spatial.endpoint.eq(endpoint)]
        dotplot(ax,frame,label=label,limits=(-1.05,1.15) if endpoint=='early_late_rho' else None)
    fig.tight_layout();save('05_spatial_stability',fig,
        '每点为被试的记录中位数。反差为测点P90减P10，再除以测点中位数。早晚按原记录时间等分，奇数中心窗丢弃，每半至少2个有效窗口；至少8个且80%的位置共同可用。短REFED视频可能无法进入早晚比较，顶部保留完整人数。','spatial_subjects.csv')
    fits=tables['fit_subjects']
    fig,axes=plt.subplots(1,3,figsize=(10,3.7))
    for ax,metric in zip(axes,('D1','D_EH','D4')):
        part=fits[fits.metric.eq(metric)];valid=part[['fixed','free']].notna().all(axis=1)
        maximum=max(float(part.loc[valid,['fixed','free']].max().max()),.01)*1.05
        for dataset,marker in zip(datasets,('o','s','^','D')):
            block=part[part.dataset.eq(dataset)&valid]
            ax.scatter(block.fixed,block.free,s=25,marker=marker,color=colors[dataset],label=short[dataset],alpha=.75)
        ax.plot([0,maximum],[0,maximum],color='grey',ls='--',lw=.8)
        ax.set(xlim=(0,maximum),ylim=(0,maximum),xlabel='固定参数 / 训练SD',ylabel='共享参数 / 训练SD',title=metric)
        ax.grid(alpha=.13)
    axes[0].legend(fontsize=7);fig.tight_layout();save('06_fixed_vs_shared',fig,
        '每点为同一被试、相同成功窗口的配对结果；对每组窗口先合并平方误差，再按组、记录和被试的声明顺序汇总并开方。对角线下方表示共享参数误差较小。共享κ/τ与驱动、初态仍为条件拟合，误差下降不是生理机制恢复的证据。','fit_subjects.csv')
    parameters=tables['parameter_subjects']
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for ax,name,reference_value in zip(axes,('tau','kappa'),(2.,.64)):
        for i,dataset in enumerate(datasets):
            part=parameters[parameters.dataset.eq(dataset)]
            wide=part.pivot(index='subject',columns='view',values='paired_'+name).dropna()
            for j,(_,r) in enumerate(wide.iterrows()):
                jitter=.1*np.sin(j*2.399963)
                ax.plot([i-.14+jitter,i+.14+jitter],[r.hb_only,r.joint],color='#b7bcc2',lw=.6)
                ax.scatter(i-.14+jitter,r.hb_only,color=COLORS['hb'],s=18,marker='o')
                ax.scatter(i+.14+jitter,r.joint,color=COLORS['joint'],s=18,marker='s')
        ax.axhline(reference_value,color='black',ls='--',lw=.8)
        ax.set(yscale='log',xticks=range(4),xticklabels=[short[d] for d in datasets],
            ylabel=(r'$\tau$ / s' if name=='tau' else r'$\kappa$ / s$^{-1}$'))
        ax.tick_params(axis='x',labelsize=8,rotation=15);ax.grid(axis='y',alpha=.15)
    fig.tight_layout();save('07_parameter_visibility',fig,
        '每条细线连接同被试在Hb-only（蓝圆）与joint（紫方）条件下的估计；仅使用两种视图都有效的同一组，先对组取log、记录内及记录间等权平均，再指数变换。纵轴为对数轴；虚线为固定参考τ=2秒、κ=0.64/秒。参数边界与失败另列，不把条件变化当作真实生理参数变化。','parameter_subjects.csv')

    scalar=tables['subject_summary'];shape=tables['shape_summary']
    def scalar_row(ds,metric):return scalar[scalar.dataset.eq(ds)&scalar.endpoint.eq(metric)].iloc[0]
    def shape_row(ds,feature):return shape[shape.dataset.eq(ds)&shape.endpoint.eq(feature)].iloc[0]
    def value_string(value):return f'{value:.3f}' if np.isfinite(value) else '不可用'
    common_text='、'.join(f"{short[d]} {shape_row(d,'Hb_common_direction_fraction')['median']:.3f}" for d in datasets)
    corr_text='、'.join(f"{short[d]} {shape_row(d,'EEG_Hb_readout_correlation')['median']:.3f}" for d in datasets)
    content=dict(schema='source_diagnostics_report_v1',title='四数据集Tak失配的共同模式',
        subtitle='Visual原生通道更正后：残差方向、跨模态不一致、空间稳定性与参数适配',
        date=datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        provenance=['Visual使用更正后31通道原生EEG重新拟合；其余三库沿用经核验的保留数值。',
            '固定参数为全通道标量；波形与共享参数比较采用原有三通道面板。各图保留计划、可用和失败分母。'],
        summary=[f'双Hb残差共同方向平方占比的被试中位数：{common_text}。它描述模型残差结构，不识别其生理来源。',
            f'Hb-only读出与EEG代理的有符号相关中位数：{corr_text}。Hb重建和跨模态驱动一致性需要分别判断。',
            '空间反差与同记录早晚一致性用于判断模式是否稳定；不同数据集的传感器覆盖不等同于相同皮层区域。',
            '共享参数只在相同窗口上与固定模型比较；视图依赖、失败和工程边界共同限定参数的生理解释。'],sections=[],sources=list(sources.values()))
    coverage_rows=[];failure_rows=[]
    for dataset in datasets:
        cov=summary['coverage'][dataset];p=cov['plan']
        coverage_rows.append([short[dataset],p['subjects'],p['Hb_admitted_records'],p['planned_pair_windows'],
            cov['fixed_views']['hb_only']['completed'],cov['fixed_views']['joint']['completed']])
        for view,label in (('hb_only','Hb-only'),('eeg_only','EEG-only'),('joint','joint')):
            v=cov['fixed_views'][view];failure_rows.append([short[dataset],label,v['planned'],v['fitted'],v['numerical_failure'],v['unavailable_support'],v['execution_failure']+v['missing']])
    content['sections'].append(dict(title='1. 输入、拟合条件与四库覆盖',blocks=[
        _paragraph('研究问题是哪些模型失配形态在四库均可检查，并帮助区分观测限制、跨模态代理限制和动力学适配。全部结果为公开数据的探索性条件重建，不是前瞻预测或真实神经驱动恢复。'),
        _table(['数据集','计划被试','允许记录','计划窗对','Hb-only成功','joint成功'],coverage_rows,
            '窗对为一个30秒窗口×一个HbO/HbR测量位置。Visual同步Probe合并为一次采集解释，但保留各自原生记录身份。重复窗、通道和视频不增加独立被试数。'),
        _paragraph('固定参考κ=0.64/秒、τ=2秒；共享组在原生记录、连续Hb支持与Hb通道内估计κ/τ，各窗口仍独立估计驱动和四初态。Hb-only只看双Hb，EEG-only只看EEG代理，joint同时看三坐标。训练PCA与尺度使用外层训练被试。'),
        _paragraph('每个目标窗30秒，首5秒为参考，评分为5.0至29.8秒；EEG功率由最多90秒原生上下文计算，短REFED视频使用真实完整支持。Hb来自既有离线带通缓存。本次不改变90秒上下文规则，不读取下游新窗口缓存。'),
        _paragraph('Visual以Raw_EEG_channel_reference.xls连接A*到电极名称，再按名称连接Location.ced。A11/A12为CP3/CP4，A17/A18为O1/O2，A64为FCz。旧错误结果保留；更正后重新拟合训练折坐标与空间选择，未对旧数组改名。')]))
    content['sections'].append(dict(title='2. 指标含义与汇总顺序',blocks=[
        dict(type='equation',latex=r'D_1=\sqrt{\left\langle (H-\widehat H_H)^2/\sigma_H^2\right\rangle},\qquad R_H=\frac{\left\langle(H-\widehat H_H)^2/\sigma_H^2\right\rangle}{\left\langle H^2/\sigma_H^2\right\rangle}',caption='H为参考后的处理Hb，σ为冻结训练观察SD；H下标表示只由Hb拟合的读出。平均在相同有效时间与两个色团上进行；R_H的分子、分母分别汇总后相除。'),
        dict(type='equation',latex=r'D_{EH}=\sqrt{\left\langle(E-\widehat E_H)^2/\sigma_E^2\right\rangle},\quad c=\frac{r_O+r_R}{\sqrt2},\quad d=\frac{r_O-r_R}{\sqrt2},\quad F_c=\frac{\langle c^2\rangle}{\langle c^2\rangle+\langle d^2\rangle}',caption='E为训练折PC1功率代理；r_O、r_R为各自训练SD标准化的Hb残差。F_c为正交统计坐标的平方占比，不能称为HbT或某类生理来源比例。'),
        _paragraph('D_drive_H将独立EEG驱动与Hb驱动放入同一个固定模型、共同静息初态回放，再计算双Hb训练SD标准化RMS差。D_bal为连续Hb在Tak小信号关系下的弱积分偏离，10/20/40秒为核宽，分母来自同支持两侧动态量平方和；这些尺度不是正常生理阈值。'),
        _paragraph('主标量按窗口平方量、原生记录、被试逐级汇总；每库单列，不把通道训练SD当跨库绝对浓度。残差形态来自三通道面板，先合并矩再取比例。图中被试中位数、四分位距、P90与均值各自标明，本报告不新增显著性检验。')]))
    content['sections'].append(dict(title='3. 四库的Hb相容性及共同残差方向',page_break=True,blocks=[
        figures['01_hb_compatibility'],
        _table(['数据集','D1中位数','R_H中位数','D_bal20中位数'],[[short[d],*[value_string(scalar_row(d,m)['median']) for m in ('D1',HB_RATIO,'D_bal_20s')]] for d in datasets]),
        _paragraph('D1受信号幅度及训练尺度影响，R_H加入同支持零变化基线，弱平衡直接描述双Hb动态关系。三者共同用于区分“数值小”与“相对背景容易解释”；它们不能相加成一个生理可信度分数。'),
        figures['02_residual_directions'],
        _table(['数据集','共同方向占比','残差O/R相关','偏置平方占比'],[[short[d],*[value_string(shape_row(d,f)['median']) for f in ('Hb_common_direction_fraction','Hb_residual_OR_correlation','Hb_bias_fraction')]] for d in datasets]),
        _paragraph('共同方向结构是四库可以共同追踪的机制线索。可能解释包括未建模的共同血流/观测变化或当前Hb关系过于刚性；这是候选解释，现有残差分解不能区分它们。偏置与去均值动态应分开，不能把整段共同变化称为独立生理成分。')]))
    content['sections'].append(dict(title='4. Hb重建与EEG代理一致性分开判断',page_break=True,blocks=[
        figures['03_crossmodal_disagreement'],
        _table(['数据集','D_EH中位数','读出相关中位数','D_drive_H中位数'],[[short[d],value_string(scalar_row(d,'D_EH')['median']),value_string(shape_row(d,'EEG_Hb_readout_correlation')['median']),value_string(scalar_row(d,'D_drive_H')['median'])] for d in datasets]),
        _paragraph('Hb-only条件下的低残差并不要求驱动与EEG功率代理接近；驱动、初态和观测关系可共同吸收变化。误差和有符号相关提供互补信息：小数值残差不能代替波形方向、幅度与时序的检查。'),
        _paragraph('该模式支持继续审查功率代理、跨模态时钟与观测映射，不能直接判定Hb驱动错误、EEG不是神经活动，或残差全为伪迹。Visual更正解决物理通道身份，不能单独解决这种识别问题。')]))
    independent_rows=[]
    for d in datasets:
        for metric,label in (('prediction_gain_over_zero','相对零变化'),('prediction_gain_over_template','相对训练模板')):
            part=independent[independent.dataset.eq(d)&independent.metric.eq(metric)];good=part.value.notna()
            independent_rows.append([short[d],label,f'{good.sum()}/{len(part)}',value_string(part.value.median()),
                int((part.value>0).sum()),int(part.available_windows.sum()),int(part.planned_windows.sum())])
    content['sections'].append(dict(title='4.1 独立预测与空间替代对照',page_break=True,blocks=[
        figures['08_independent_prediction'],
        _table(['数据集','比较','可用／计划人','收益中位数','正收益人数','配对窗对','计划窗对'],independent_rows,
            '收益为基线误差减独立EEG-only误差，单位为Hb训练SD平方；不是百分比。缺失EEG或数值失败不填零。'),
        _paragraph('这组对照检验EEG代理能否提供超出简单Hb预测的价值，与当前Hb参与拟合的条件重建区分。空间替代始终保持六通道和相同频带容量，仅在训练被试选择，再由完整非线性模型评价；正收益不表示已定位到真实神经来源。'),
        _paragraph('即使优于零变化或训练模板，也不能单独排除共同事件结构。错配EEG必须有同条件且真实滤波支持不重叠的供体；短视频或缺少可用供体时，该对照不可用，应保留为限制，而不以循环移位制造伪独立证据。')]))
    content['sections'].append(dict(title='5. 空间异质性与记录内稳定性',page_break=True,blocks=[figures['04_population_maps'],figures['05_spatial_stability'],
        _table(['数据集','相对残差空间反差','早晚ρ中位数','早晚可用／计划人'],[
            [short[d],value_string(spatial[spatial.dataset.eq(d)&spatial.metric.eq(HB_RATIO)&spatial.endpoint.eq('spatial_contrast')].value.median()),
             value_string(spatial[spatial.dataset.eq(d)&spatial.metric.eq(HB_RATIO)&spatial.endpoint.eq('early_late_rho')].value.median()),
             f"{spatial[spatial.dataset.eq(d)&spatial.metric.eq(HB_RATIO)&spatial.endpoint.eq('early_late_rho')].value.notna().sum()}/{summary['coverage'][d]['plan']['subjects']}"] for d in datasets]),
        _paragraph('稳定的空间偏离比单次误差热点更适合形成机制假设；早晚相似仍可能来自持续的观测因素，不能据此定位皮层来源。四库可共同检查空间异质性，但不能强行把不同覆盖的测点称为同一脑区。REFED不同视频是不同刺激，Visual同步Probe也不是重复测量。')]))
    parameter_groups=tables['parameter_groups'];parameter_rows=[]
    for d in datasets:
        for view,label in (('hb_only','Hb-only'),('joint','joint')):
            g=parameter_groups[parameter_groups.dataset.eq(d)&parameter_groups.view.eq(view)]
            valid=g.valid_estimate.astype(bool)
            p=parameters[parameters.dataset.eq(d)&parameters.view.eq(view)]
            parameter_rows.append([short[d],label,f'{valid.sum()}/{len(g)}',int(g.full_planned_support.sum()),
                int(g.loc[valid,'tau_at_upper_bound'].sum()),value_string(p.paired_tau.median()),value_string(p.paired_kappa.median())])
    paired_rows=[]
    for d in datasets:
        for metric in ('D1','D_EH','D4'):
            part=fits[fits.dataset.eq(d)&fits.metric.eq(metric)]
            good=part[['fixed','free']].notna().all(axis=1);paired=part[good]
            paired_rows.append([short[d],metric,f'{len(paired)}/{len(part)}',
                value_string(paired.fixed.median()),value_string(paired.free.median()),
                value_string(paired.free_minus_fixed.median()),int((paired.free_minus_fixed<0).sum())])
    content['sections'].append(dict(title='6. 参数适配的价值取决于模式和稳定性',page_break=True,blocks=[figures['06_fixed_vs_shared'],
        _table(['数据集','指标','配对／计划人','固定中位数','共享中位数','配对差中位数','共享较低人数'],paired_rows,
            '差为共享值减固定值，单位为该指标的训练SD。差先在同被试内计算再取中位数，不一定等于前两列中位数相减。成功配对子集以外的失败保留在后表。'),
        figures['07_parameter_visibility'],
        _table(['数据集','视图','有效／计划组','全窗有效组','τ触上界组','配对τ中位数','配对κ中位数'],parameter_rows,
            '有效组至少有一窗通过数值验证，并不等于全部窗口成功。τ上界12秒是工程约束。末两列先按同组配对、记录等权得到被试几何均值，再取被试中位数。'),
        _paragraph('固定参数为所有位置提供同一个动力学参照，适合描述失配在哪里、是否重复。共享参数用于检验哪些位置需要不同响应，并检查这种适配是否依赖Hb-only或joint的信息条件。只有重建误差降低不足以选择更可解释的模型。'),
        _paragraph('两视图按原生组身份配对，但Hb-only可以使用缺少EEG的窗口，joint不能。因此参数差也可能包含实际可用窗口的差异，不能将其纯粹归因于加入EEG。固定与共享误差的配对图则严格使用相同成功窗口。'),
        _paragraph('参数触边、对输入视图敏感、同被试记录不稳定都限制生理解释；κ尤其可与自由驱动补偿。后续应针对稳定图形设计独立预测或观测对照，不能把当前条件估计当成个体血流参数测量。')]))
    revision=tables['visual_revision_subjects'];revision_rows=[]
    for metric in ('D1','D_EH','D_drive_H','D4','D_bal_20s'):
        r=revision[revision.metric.eq(metric)]
        revision_rows.append([metric,value_string(r.value_old.median()),value_string(r.value_corrected.median()),
            value_string(r.corrected_minus_old.median()),value_string(r.corrected_minus_old.abs().max())])
    content['sections'].append(dict(title='7. Visual修复影响与不能覆盖的机制',blocks=[
        _table(['指标','旧值中位数','更正中位数','配对差中位数','最大绝对差'],revision_rows,
            '按同一被试配对；差为更正值减旧值。EEG目标、PCA和空间邻域已改变，因此EEG相关指标之差是数据修复影响，不能称为预测收益。纯Hb指标应复现。'),
        _paragraph('四库共同支持残差方向、跨模态代理差异、空间异质性和条件参数适配的描述。眼电机制只在具备原生HEOG/VEOG的两库补充；其他两库不得用前额EEG冒充EOG。不同视频或固定任务顺序也不足以提供相同条件的跨日重测。'),
        _paragraph('建议保留固定模型作为共同参照，并把共享参数当作局部适配诊断。优先追踪在相对残差、弱动态关系及时间稳定性上同时出现的结构；再用独立传感器、明确的观测变换或隐藏测量预测检验候选机制。')]))
    full_visual=summary.get('visual_full_shared')
    if full_visual is not None:
        if full_visual.get('status')!='completed':
            raise ValueError('Full Visual parameter maps require their complete numerical source')
        content['provenance'].append(f"Visual另外完成全部Hb位置的共享参数扩展：{full_visual['groups']}组、{full_visual['windows']}窗对。它单列于第8节，未与其他三库较窄的面板混为同覆盖比较。")
        full_groups=tables['visual_full_parameter_groups'];full_maps=tables['visual_full_parameter_maps']
        extension_rows=[]
        for view,label in (('hb_only','Hb-only'),('joint','joint')):
            block=full_groups[full_groups.view.eq(view)];valid=block[block.valid_estimate.astype(bool)]
            equal_subject=valid.groupby('subject')[['tau','kappa']].median()
            extension_rows.append([label,len(block),len(valid),int(block.full_planned_support.sum()),
                int(valid.tau_at_upper_bound.sum()),len(equal_subject),value_string(equal_subject.tau.median()),value_string(equal_subject.kappa.median())])
        content['sections'].append(dict(title='8. Visual全通道共享参数扩展',page_break=True,blocks=[
            _paragraph('本节覆盖Visual每条允许记录的全部24个Hb位置，两侧Probe保持原生身份。前面的四库主比较继续使用每记录三个位置的共同面板；扩大位置分母后的汇总变化不被当作方法改善。'),
            _table(['视图','计划组','有效组','全窗有效组','τ触上界组','有效被试','被试τ中位数','被试κ中位数'],extension_rows,
                '有效组至少一窗通过收敛与积分检查；有限失败参数不进入有效估计。末两列先取每被试有效组中位数，再对被试取中位数。τ以秒、κ以每秒计，τ上界12秒为工程边界。'),
            _paragraph('下面按被试身份展示全部被试，不按参数极值选择。每个测点先在同一位置选Hb-only与joint都有效的同一组，求该通道的参数中位数；重合模板位置再合并通道中位数。两视图按组配对，但实际可拟合窗口可不同；缺失位置以灰叉保留。')]))
        people=sorted(full_maps.subject.unique())
        for offset in range(0,len(people),4):
            chosen=people[offset:offset+4]
            fig,axes=plt.subplots(2,len(chosen),figsize=(10,6.2),squeeze=False)
            fig.subplots_adjust(left=.025,right=.9,top=.92,bottom=.08,wspace=.12,hspace=.26)
            for row,(view,label) in enumerate((('hb_only','Hb-only'),('joint','joint'))):
                for col,subject in enumerate(chosen):
                    frame=full_maps[full_maps.subject.eq(subject)&full_maps.view.eq(view)]
                    frame=display(frame,visual,subject)
                    im=spatial_topomap(axes[row,col],frame,limits=(.25,12.),coverage='head',title=f'{subject} · {label}')
            cax=fig.add_axes([.93,.20,.017,.56]);fig.colorbar(im,cax=cax).set_label(r'配对组 $\tau$ 中位数 / s',fontsize=10)
            fig.suptitle('Visual全通道：同一被试在两种可见条件下的时间尺度',fontsize=13)
            fig.text(.48,.015,'所有页面共用0.25–12秒色标；黑边点为测点，灰叉为缺失，虚线外填色为显示外推。',ha='center',fontsize=8.5)
            key=f'visual_full_tau_{offset//4+1:02}'
            save(key,fig,'上下两行分别为Hb-only和joint，同一列同一被试。每点使用相同的两视图有效组；参数图是条件模型对数据的适配，不能作为真实血流时间常数、个体MRI定位或因果机制图。',str(analysis/'visual_full_parameter_maps.csv'))
            block=full_maps[full_maps.subject.isin(chosen)&full_maps.view.eq('hb_only')]
            entries=[[subject,int((block[block.subject.eq(subject)].available_members>0).sum()),len(block[block.subject.eq(subject)])] for subject in chosen]
            content['sections'].append(dict(title='Visual全通道参数图：'+', '.join(chosen),page_break=True,blocks=[figures[key],
                _table(['被试','配对参数可用位置','计划模板位置'],entries,
                    '模板重合位置已合并，所以位置数可小于两个Probe的48个Hb通道。它没有减少原始拟合和失败分母。')]))
    example_intro=_paragraph('以下示例从原有三通道面板按固定D1选择最接近中位数及P90的窗口，尽量来自不同被试。选择仅用于读图，不是代表性比例、调参或确认性证据。黑线为处理后的实测目标，蓝虚线为Hb-only重建，紫线为joint重建；两种拟合均使用当前窗口的声明可见输入。')
    example_diagnostics=[]
    for number,example in enumerate(sorted(examples,key=lambda r:(datasets.index(r['dataset']),r['role']))):
        with np.load(example['array_path'],allow_pickle=False) as saved:
            arrays={k:saved[k] for k in saved.files}
        time=np.arange(len(arrays['target']))*.1;sd=np.asarray(arrays['sd'],float)
        target=arrays['target']/sd;mask=np.asarray(arrays.get('score_mask',np.isfinite(target)),bool)
        if mask.shape!=target.shape:
            mask=np.broadcast_to(mask[:,None],target.shape)
        fig,axes=plt.subplots(3,1,figsize=(9,5.7),sharex=True);stats=[]
        for c,ax in enumerate(axes):
            ax.plot(time,target[:,c],color=COLORS['target'],lw=1.1,label='处理实测')
            messages=[]
            for key,label,color,style in (('hb_only_prediction','Hb-only',COLORS['hb'],'--'),('joint_prediction','joint',COLORS['joint'],'-')):
                if key not in arrays:continue
                success_key='hb_only_success' if label=='Hb-only' else 'joint_success'
                if not example.get(success_key,False):
                    messages.append(label+'未通过数值检查')
                    continue
                prediction=arrays[key]/sd
                ax.plot(time,prediction[:,c],color=color,ls=style,lw=1.,label=label)
                good=mask[:,c]&np.isfinite(target[:,c])&np.isfinite(prediction[:,c]);good[:50]=False;good[-1]=False
                rmse=float(np.sqrt(np.mean((prediction[good,c]-target[good,c])**2))) if good.any() else np.nan
                corr=float(np.corrcoef(prediction[good,c],target[good,c])[0,1]) if good.sum()>1 and np.std(prediction[good,c])>0 and np.std(target[good,c])>0 else np.nan
                messages.append(f'{label} RMS={rmse:.3f}, r={corr:.3f}')
                stats.append(dict(component=('EEG','HbO','HbR')[c],view=label,
                    rmse=rmse if np.isfinite(rmse) else None,correlation=corr if np.isfinite(corr) else None))
            ax.axvspan(0,5,color='#e8e8e8',alpha=.65);ax.set_ylabel(('EEG代理','HbO','HbR')[c]+' / 训练SD')
            ax.set_title('；'.join(messages),fontsize=8.5);ax.grid(alpha=.14)
        axes[0].legend(ncol=3,fontsize=8);axes[-1].set_xlabel('目标窗口内时间 / 秒')
        fig.suptitle(f"{short[example['dataset']]} · {example['subject']} · {example['pair']} · {example['role']}",fontsize=12)
        fig.tight_layout(rect=[0,0,1,.96]);key=f'example_{number+1:02}'
        save(key,fig,'灰区为5秒参考段，不评分；最后一个驱动箱不评分。各分量和各例使用独立纵轴，单位为对应冻结训练SD；上方给出同评分支持的RMS误差与有符号相关，不能只凭外观判断恢复。',example['array_path'])
        hb=[r for r in stats if r['component'] in ('HbO','HbR') and r['view']=='Hb-only']
        if len(hb)==2 and all(r['rmse'] is not None for r in hb):
            recovered=float(np.sqrt(np.mean([r['rmse']**2 for r in hb])))
            if not np.isclose(recovered,example['D1'],rtol=1e-8,atol=1e-10):
                raise ValueError('Example bitmap statistics do not reproduce the owning D1 metric')
        text='；'.join(f"{r['component']}的Hb-only误差{value_string(r['rmse'] if r['rmse'] is not None else np.nan)}训练SD、相关{value_string(r['correlation'] if r['correlation'] is not None else np.nan)}" for r in hb)
        content['sections'].append(dict(title=(('9' if full_visual else '8')+'. 波形示例 · ' if number==0 else '')+f"{short[example['dataset']]}：{'中位' if example['role']=='median' else 'P90'}失配窗口",page_break=True,blocks=([example_intro] if number==0 else [])+[
            figures[key],_paragraph(f"{example['subject']} / {example['record']} / {example['pair']}；Hb窗起点{example['hb_start_s']:.1f}秒，原生EEG上下文{example['actual_context_s']:.1f}秒。候选{example['candidate_denominator']}／计划{example['planned_panel_denominator']}窗对，选择排序{example['selection_rank']}。"),
            _paragraph(text+'。这些值分别核验幅值误差与波形方向；本例不能判定潜在驱动或残差的真实生理来源。')]))
        example_diagnostics.append(dict(id=example['id'],statistics=stats))
    content['sections'].append(dict(title='附录：完整失败分母与来源',page_break=True,blocks=[
        _table(['数据集','视图','计划窗对','进入拟合','数值失败','支持不足','异常／缺失'],failure_rows,
            '支持不足、未收敛／积分未通过和执行异常分别列出；成功子集的中位数不能替代这些分母。'),
        _paragraph('数值来源：'+str(analysis/'summary.json')),
        _paragraph('Visual原生身份核验：'+sources['input_repair']),
        _paragraph('其他三库保留来源：'+sources['reference_run']),
        _paragraph('所有图为位图嵌入；正文、表格和图注保持可选择。报告只汇总已有数值，不替代各运行及派生分析的事实来源。')]))
    (output/'example_diagnostics.json').write_text(json.dumps(example_diagnostics,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    path=output/'report_content.json';path.write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n')
    checks=render_report(path,output/'report')
    shutil.copy2(Path(__file__),output/'renderer_source.py')
    (output/'source_manifest.json').write_text(json.dumps(dict(schema='tak_mismatch_four_dataset_report_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),owning_analysis=str(analysis/'summary.json'),
        source_runs=[sources['reference_run'],sources['corrected_run']],sources=sources,
        bitmap_figures=True,model_fits=0,raw_signal_reads=0),ensure_ascii=False,indent=2)+'\n')
    return dict(output=str(output),pages=checks['exports']['TECHNICAL_REPORT.pdf']['pages'],figures=len(figures))


def _hb_atlas_subject_selection(maps, ranking):
    """Choose explanatory D1 maps using the retained display ranking.

    Each candidate must have all planned acquisition maps, at least two
    acquisitions, >=8 and 80% shared locations for every pair, and every pair's
    rank correlation >=.5. The peak of the retained subject map must exceed
    the spatial mean in every acquisition. These are display selection rules,
    not calibrated physiological cutoffs or a new inferential endpoint.
    """
    import pandas as pd
    from itertools import combinations
    from scipy.stats import spearmanr
    selected=[];candidates=[];record_checks=[]
    for dataset, ranks in ranking.groupby('dataset',sort=True):
        chosen=False
        for row in ranks.sort_values(['rank','subject']).itertuples():
            group=maps[maps.dataset.eq(dataset)&maps.subject.eq(str(row.subject))&maps.metric.eq('D1')]
            full=group[group.period.eq('full')].copy()
            full['display_value']=full.display_value.where(full.display_valid)
            records=full.pivot(index='record_unit',columns='location_id',values='display_value')
            mean=group[group.period.eq('subject_mean')].set_index('location_id')
            n=len(records);minimum=max(8,.8*records.shape[1]);correlations=[]
            sufficient=n>=2 and records.notna().sum(axis=1).ge(minimum).all()
            for a,b in combinations(range(n),2):
                pair=records.iloc[[a,b]].dropna(axis=1).to_numpy(float)
                if pair.shape[1]<minimum or np.ptp(pair[0])<=1e-12 or np.ptp(pair[1])<=1e-12:
                    sufficient=False;continue
                correlations.append(float(spearmanr(pair[0],pair[1]).statistic))
            peak=mean.display_value.idxmax() if mean.display_value.notna().any() else None
            margins=records[peak]-records.mean(axis=1) if peak in records else pd.Series(np.nan,index=records.index)
            eligible=bool(sufficient and len(correlations)==n*(n-1)//2 and min(correlations)>=.5
                          and margins.notna().all() and margins.gt(0).all())
            record=dict(dataset=dataset,subject=str(row.subject),retained_rank=int(row.rank),
                planned_records=n,record_pairs=len(correlations),
                median_record_correlation=float(np.median(correlations)) if correlations else np.nan,
                minimum_record_correlation=float(min(correlations)) if correlations else np.nan,
                peak_location=peak,peak_pair=mean.loc[peak,'pair_members'] if peak is not None else '',
                peak_D1=float(mean.loc[peak,'display_value']) if peak is not None else np.nan,
                minimum_peak_above_record_mean=float(margins.min()),eligible=eligible,selected=eligible and not chosen)
            candidates.append(record)
            if record['selected']:
                chosen=True;selected.append(record)
                for record_id,values in records.iterrows():
                    record_checks.append(dict(dataset=dataset,subject=str(row.subject),record_unit=record_id,
                        peak_pair=record['peak_pair'],peak_D1=float(values[peak]),
                        spatial_mean_D1=float(values.mean()),peak_above_spatial_mean=float(margins.loc[record_id])))
        if not chosen:raise ValueError(f'No fully supported persistent D1 example for {dataset}')
    return pd.DataFrame(selected),pd.DataFrame(candidates),pd.DataFrame(record_checks)


def render_hb_atlas_design(run, analysis, output):
    """Three clean Hb-only figure sets from completed public scalar evidence."""
    import html
    import shutil
    import pandas as pd
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    run,analysis,output=(Path(p).resolve() for p in (run,analysis,output))
    summary=_json(analysis/'summary.json')
    if (summary.get('schema')!='tak_mismatch_four_dataset_fixed_patterns_v1'
            or summary.get('status')!='completed' or Path(summary['parent_run']).resolve()!=run):
        raise ValueError('Completed fixed-only four-dataset scalar evidence required')
    if output.exists():raise FileExistsError('Use a fresh versioned atlas directory')
    sources=summary['sources'];visual='visual_cognitive_motivation'
    columns=['dataset','subject','task_label','record_unit','period','location_id','metric','x','y','pair_members',
        'record_ids','display_valid','selection_valid','display_value','planned_records','available_records',
        'planned_windows','available_windows']
    maps=[];ranks=[];records=[]
    for map_key,plan_key,corrected in [('reference_maps','reference_plan',False),('corrected_maps','corrected_plan',True)]:
        path=Path(sources[map_key]);frame=pd.read_csv(path,usecols=columns,dtype={'subject':str})
        take=frame.dataset.eq(visual);maps.append(frame[(take if corrected else ~take)&frame.metric.eq('D1')])
        rank=pd.read_csv(path.parent/'subject_ranking.csv',dtype={'subject':str});take=rank.dataset.eq(visual)
        ranks.append(rank[take if corrected else ~take])
        plan=_json(Path(sources[plan_key]))
        records.extend(r for r in plan['records'] if (r['dataset']==visual)==corrected)
    maps=pd.concat(maps,ignore_index=True);ranking=pd.concat(ranks,ignore_index=True);plan=dict(records=records)
    selected,candidates,record_checks=_hb_atlas_subject_selection(maps,ranking)
    population=pd.read_csv(analysis/summary['files']['all_population_maps'])
    population=population[population.metric.eq('D1')].copy()
    subjects=pd.read_csv(analysis/summary['files']['subjects'],dtype={'subject':str})
    subjects=subjects[subjects.metric.eq('D1')].copy()
    persistence=pd.read_csv(analysis/summary['files']['hotspot_persistence'],dtype={'subject':str})
    persistence=persistence[persistence.metric.eq('D1')].copy()
    datasets=['eeg_fnirs_single_trial','simultaneous_eeg_nirs',visual,'refed']
    labels=dict(zip(datasets,['Single-Trial','Simultaneous','Visual','REFED']))
    if set(selected.dataset)!=set(datasets) or len(subjects)!=summary['planned_subjects']:
        raise ValueError('Planned dataset/subject denominator drift')
    def project(frame,dataset,subject):
        template=maps[maps.dataset.eq(dataset)&maps.subject.eq(subject)&maps.period.eq('subject_mean')]
        coords=_spatial_preview_coordinates(template,plan).set_index('pair_members')
        xy=coords.loc[frame.pair_members]
        if not np.allclose(frame[['x','y']],xy[['x_orthographic','y_orthographic']],atol=1e-7):
            raise ValueError('Frozen geometry identity mismatch')
        result=frame.copy();result[['x','y']]=xy[['x','y']].to_numpy()
        assert np.array_equal(frame.display_value,result.display_value,equal_nan=True)
        return result
    group_maps=[];person_maps=[]
    for ds in datasets:
        first=sorted(maps[maps.dataset.eq(ds)].subject.unique())[0]
        group_maps.append(project(population[population.dataset.eq(ds)],ds,first))
        subject=selected[selected.dataset.eq(ds)].subject.iloc[0]
        frame=persistence[persistence.dataset.eq(ds)&persistence.subject.eq(subject)].copy()
        frame['display_value']=frame.fraction.where(frame.eligible)
        person_maps.append(project(frame,ds,subject))
    limits=(0.,1.);cmap='viridis';label='信号的模型偏离度'
    output.mkdir(parents=True);images=output/'images';images.mkdir();plt=_plotting()
    plt.rcParams.update({'font.size':13,'axes.labelsize':15,'xtick.labelsize':12,'ytick.labelsize':12})
    figures=[];checks=[]
    def topography(frames,filename,individual):
        fig,axes=plt.subplots(1,4,figsize=(14.4,4.35))
        fig.subplots_adjust(left=.025,right=.975,top=.89,bottom=.25 if not individual else .22,wspace=.13)
        for ax,ds,frame in zip(axes,datasets,frames):
            title=labels[ds]+(' · '+selected[selected.dataset.eq(ds)].subject.iloc[0] if individual else '')
            panel_limits=limits if individual else (float(np.floor(frame.display_value.min()*100)/100),
                float(np.ceil(frame.display_value.max()*100)/100))
            if panel_limits[1]<=panel_limits[0]:panel_limits=(panel_limits[0],panel_limits[0]+.01)
            image=spatial_topomap(ax,frame,limits=panel_limits,title=title,cmap=cmap,coverage='head',annotate_support=False)
            ax.set_title(title,fontsize=15,pad=12)
            assert image.get_clim()==panel_limits
            for text_item in ax.texts:
                if any(word in text_item.get_text() for word in ('可用位置','窗口覆盖','可用记录','四分位','零预测')):
                    raise ValueError('Unrequested annotation on clean atlas')
            checks.append(dict(figure=filename,dataset=ds,clim=list(image.get_clim()),
                xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),annotations=[t.get_text() for t in ax.texts]))
            if not individual:
                pos=ax.get_position();cax=fig.add_axes([pos.x0+.025,.15,pos.width-.05,.025])
                bar=fig.colorbar(image,cax=cax,orientation='horizontal',ticks=np.linspace(*panel_limits,3),format='%.2f')
                bar.ax.tick_params(labelsize=11)
        if individual:
            cax=fig.add_axes([.27,.13,.46,.035])
            colorbar=fig.colorbar(ScalarMappable(norm=Normalize(*limits),cmap=cmap),cax=cax,
                orientation='horizontal',ticks=np.linspace(0,1,6))
            colorbar.set_label(label,fontsize=14,labelpad=6);colorbar.ax.tick_params(labelsize=11)
        else:fig.text(.5,.035,label,ha='center',fontsize=14)
        path=images/filename;fig.savefig(path,dpi=300,facecolor='white');plt.close(fig)
        figures.append(dict(file='images/'+filename,kind='individual_topography' if individual else 'population_topography',label=label))
    topography(group_maps,'01_population_topography.png',False)
    fig,ax=plt.subplots(figsize=(8.2,5.3));colors=['#2272ac','#d17924','#27816d','#9c4b90'];distribution_summary=[]
    for i,ds in enumerate(datasets):
        frame=subjects[subjects.dataset.eq(ds)].sort_values('subject');values=frame.value.to_numpy(float)
        jitter=.055*np.sin(np.arange(len(values))*2.399963)
        ax.scatter(i+jitter,values,s=44,c=colors[i],alpha=.65,edgecolors='white',linewidths=.45,zorder=2)
        q1,median,q3=np.quantile(values[np.isfinite(values)],[.25,.5,.75])
        ax.plot([i,i],[q1,q3],c='#243442',lw=4,zorder=3)
        ax.hlines([q1,q3],i-.065,i+.065,colors='#243442',linewidth=1.5,zorder=3)
        ax.scatter([i],[median],s=82,facecolor='white',edgecolor='#243442',linewidth=1.7,zorder=4)
        distribution_summary.append(dict(dataset=ds,subjects=int(np.isfinite(values).sum()),q25=q1,median=median,q75=q3))
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],marker='o',markersize=7,markerfacecolor='white',markeredgecolor='#243442',
        linestyle='none',label='中位数'),Line2D([],[],color='#243442',lw=4,label='四分位区间')],frameon=False,loc='upper right',fontsize=11)
    ax.set(xticks=range(4),xticklabels=[labels[ds] for ds in datasets],ylabel=label,xlim=(-.55,3.55))
    ax.set_ylim(0,float(np.ceil(subjects.value.max()*10)/10));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(images/'02_subject_distribution.png',dpi=300,facecolor='white');plt.close(fig)
    figures.append(dict(file='images/02_subject_distribution.png',kind='subject_scatter',label=label))
    topography(person_maps,'03_individual_topography.png',True)
    pd.concat(group_maps,ignore_index=True).to_csv(output/'population_plot_values.csv',index=False)
    pd.concat(person_maps,ignore_index=True).to_csv(output/'individual_plot_values.csv',index=False)
    subjects.to_csv(output/'subject_plot_values.csv',index=False)
    pd.DataFrame(distribution_summary).to_csv(output/'subject_distribution_summary.csv',index=False)
    selected.to_csv(output/'example_selection.csv',index=False);candidates.to_csv(output/'example_candidates.csv',index=False)
    record_checks.to_csv(output/'example_record_checks.csv',index=False)
    def mdtable(columns,rows):
        return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |',
            *['| '+' | '.join(str(x) for x in row)+' |' for row in rows]])
    rows=[[labels[r.dataset],r.subject,r.planned_records,r.record_pairs,f'{r.median_record_correlation:.3f}',
        f'{r.minimum_record_correlation:.3f}',r.peak_pair,f'{r.minimum_peak_above_record_mean:.3f}'] for r in selected.itertuples()]
    guide=f'''# Hb观测图像修订版

## 本版三套图

所有图统一使用“信号的模型偏离度”。图中只保留数据集、必要的被试标识、坐标与色条，不添加位置数量、热点圈选或说明性图注。

1. [数据集总体空间分布](images/01_population_topography.png)：在每个测点取被试级D₁的中位数。它描述该数据集的典型偏离分布。
2. [被试分布散点图](images/02_subject_distribution.png)：每个彩色点是一名被试的整体D₁，保留原失配值，不减去群体模板。横向扰动缩窄，纵坐标完全保留；白色中心点是中位数，深色竖线和两端短线为第25—75百分位区间。纵轴是偏离大小，数值越高表示当前模型对该观测的贴合程度越低。
3. [四名被试的空间示例](images/03_individual_topography.png)：每个数据集一名被试，每点颜色表示该位置在可评价记录中进入本人D₁最高四分之一的经验概率。沿用原四名被试及“信号的模型偏离度”caption，不圈出单独的热点。

## 数值定义与显示范围

D₁沿用已有固定参数Hb-only拟合的标准化残差RMS：在相同评分时刻将观测与重建之差除以各色团冻结训练SD，平方后对时间和两色团平均，再开根。总体地图沿用原记录等权的平方量聚合后取被试中位数；散点图使用保留的整体被试值。

个体概率图直接使用已有hotspot_persistence.csv中的D₁频率：每条记录在本人有效测点中按D₁从高到低选取向上取整的25%位置，再将某点入选记录数除以该点可评价记录数，各记录等权。记录须有至少8个且80%的计划测点有效，并有非零空间反差；位置须有至少2条且80%的计划记录可评价。缺失和失败不计为未入选，分母与并列截点记录保留在individual_plot_values.csv。截点并列时按location_id稳定排序选满指定数量。该量是跨记录的经验频率，不是未来失配的校准概率；数值1表示每条可评价记录均入选，0表示从未入选。具体记录入选标记由来源表hotspot_record_locations.csv持有。

目标窗为30秒，前5秒均值作为参考，评分5.0—29.8秒。模型看到当前窗口Hb，因此该量是条件重建的观测—模型偏离；不是独立预测误差、数据质量诊断或对偏离来源的唯一判断。未重新拟合模型，未读取原始信号。

总体脑地形图每个数据集采用独立颜色范围，按该库保留测点值的最小、最大值向外取整至0.01，并在各图下方显示独立色条，用于观察库内空间差异；颜色深浅不可跨库直接比较。数值没有截断。个体概率图共用 **0—1** 色标，其颜色量与总体图的D₁幅值不同。全部脑地形图保持蓝—黄配色和相同头部坐标范围。独立色标只改变显示对比度，不能据此断言每个数据集均后部高于前部；既有区域配对检验在Simultaneous中支持该方向，Single-Trial未检出同样明确的差异。

坐标为保留的头顶球面等距显示投影；REFED沿用登记平面布局，不推断三维位置。测点覆盖在各库间不同，统一显示坐标不表示同一皮层区域。黑边小点是测点，虚线为测点凸包，边界外颜色属于显示外推；填色不增加测量分辨率。

## 例图的跨记录支持

示例沿用原D₁空间示例排序，优先考虑空间差异、邻近位置连贯性和早晚一致性；再核对全部计划记录地图可评价、每个记录对至少8个且80%共同测点、所有记录对空间相关不低于0.5，以及汇总图峰值位置在每条记录中均高于该记录的测点平均偏离。每库选满足条件的最高原排序者。0.5仅是本次展示选择规则，不是生理或显著性阈值。

{mdtable(['数据集','被试','记录数','记录对','相关中位数','最小相关','汇总峰值测点','该点高于记录均值的最小差'],rows)}

该选择用于展示突出且重复的空间模式，不构成独立验证。Single-Trial的示例核对包括不同任务记录；Simultaneous跨任务，Visual跨Part，REFED跨视频。Visual仅两条采集，跨记录证据更有限。每条记录的连续偏离值核对见example_record_checks.csv，完整候选及排除保留在example_candidates.csv。

## 样本覆盖与溯源

总体图与散点图沿用全量Hb有效支持，共{summary['planned_subjects']}名计划被试；失败和缺失不填零。不同位置的被试覆盖及各被试记录覆盖保留在CSV中，不标在图像上。各库原生单位和处理链不同，颜色或数值不直接给原生生理质量排序。

数值来源：{analysis/'summary.json'}。逐图绘图值、示例选择、色标与坐标检查记录在本目录CSV和selection_manifest.json。图像均为300 dpi PNG。此前各版图集保留在原目录。
'''
    (output/'FIGURE_GUIDE.md').write_text(guide,encoding='utf-8')
    headings=['各数据集总体空间分布','各数据集被试分布','四个数据集的被试示例']
    ordered=[figures[0],figures[1],figures[2]]
    sections=''.join(f'<section><h2>{i+1}　{title}</h2><a href="{f["file"]}"><img src="{f["file"]}" alt="{html.escape(title)}"></a></section>' for i,(title,f) in enumerate(zip(headings,ordered)))
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Hb观测图像</title><style>body{margin:0;background:#f4f6f8;color:#1b3040;font:16px/1.7 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1400px;margin:24px auto;padding:30px;background:white}h1{font-size:26px}h2{font-size:20px;margin-top:36px}img{width:100%;height:auto}section:nth-of-type(2) img{max-width:900px;display:block;margin:auto}a{color:#21668e}nav{font-size:14px}@media(max-width:700px){main{padding:16px;margin:0}}</style><main><h1>Hb观测图像</h1><nav><a href="FIGURE_GUIDE.md">制作说明</a> · <a href="selection_manifest.json">来源记录</a></nav>'+sections+'</main></html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    manifest=dict(schema='hb_atlas_design_v3',owning_analysis=str(analysis/'summary.json'),sources=sources,
        metric='D1',display_label=label,individual_color_limits=list(limits),population_color_limits='per_dataset_full_range',colormap=cmap,figures=figures,
        individual_metric='D1_top_quarter_record_fraction',
        individual_source=str(analysis/summary['files']['hotspot_persistence']),
        individual_record_source=str(analysis/summary['files']['hotspot_record_locations']),
        individual_probability='hotspot_records / available_records; eligible sites only; records equally weighted',
        topography_checks=checks,selection_method='retained_D1_ranking_then_all_record_consistency_and_continuous_peak_margin',
        selected_subjects=selected[['dataset','subject']].to_dict('records'),raw_signal_reads=0,model_fits=0,
        numerical_values_changed=False,bitmap_dpi=300,source_renderer=str(Path(__file__).resolve()))
    (output/'selection_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    shutil.copy2(Path(__file__),output/'renderer_source.py')
    return dict(output=str(output),figures=3,color_limits=list(limits),selected_subjects=manifest['selected_subjects'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--priority-panel", action="store_true", help="Read the frozen bounded plotting panel, including pending groups")
    parser.add_argument("--fixed-analysis", type=Path, help="Read an explicitly completed fixed-parameter physiology analysis; current_dof is not opened")
    parser.add_argument("--spatial-analysis", type=Path, help="Read the completed spatial-pattern analysis and render MNE channel maps")
    parser.add_argument("--spatial-style-preview", action="store_true", help="Two angular-projection/full-head display tests from retained spatial maps")
    parser.add_argument("--mechanism-analysis", type=Path, help="Completed mechanism-pattern scalar analysis for the full physiological-pattern report")
    parser.add_argument("--four-dataset-analysis", type=Path, help="Completed four-dataset synthesis with corrected Visual evidence")
    parser.add_argument('--hb-atlas-design',action='store_true',help='D1 population/distribution figures and individual top-quarter probabilities from the full fixed Hb analysis')
    parser.add_argument("--workers", type=int, default=2, help="Independent subject/metric plot workers in spatial mode")
    parser.add_argument("--revise-report", type=Path, help="Preserve an existing fixed report's numbers and bitmaps while applying the Visual channel impact audit")
    parser.add_argument("--visual-channel-audit", type=Path, help="The single confirmed Tak Visual interpretation-boundary audit")
    args = parser.parse_args(argv)
    if args.four_dataset_analysis:
        if any((args.mechanism_analysis,args.spatial_analysis,args.spatial_style_preview,args.fixed_analysis,args.priority_panel,args.revise_report,args.visual_channel_audit)):
            parser.error('--four-dataset-analysis has its own corrected completed evidence scope')
        render=render_hb_atlas_design if args.hb_atlas_design else render_four_dataset_patterns
        print(json.dumps(render(args.run,args.four_dataset_analysis,args.output),ensure_ascii=False));return
    if args.hb_atlas_design:parser.error('--hb-atlas-design requires --four-dataset-analysis')
    if args.mechanism_analysis:
        if any((args.spatial_analysis,args.spatial_style_preview,args.fixed_analysis,args.priority_panel,args.revise_report,args.visual_channel_audit)):
            parser.error("--mechanism-analysis has its own completed scalar evidence scope")
        print(json.dumps(render_mechanism_patterns(args.run,args.mechanism_analysis,args.output),ensure_ascii=False));return
    if args.spatial_analysis:
        if args.revise_report or args.fixed_analysis or args.priority_panel or args.visual_channel_audit:
            parser.error("--spatial-analysis has its own completed evidence scope")
        if args.spatial_style_preview:
            print(json.dumps(render_spatial_style_preview(args.run,args.spatial_analysis,args.output),ensure_ascii=False));return
        print(json.dumps(render_fixed_topography(args.run,args.spatial_analysis,args.output,workers=args.workers),ensure_ascii=False));return
    if args.spatial_style_preview:
        parser.error("--spatial-style-preview requires --spatial-analysis")
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
