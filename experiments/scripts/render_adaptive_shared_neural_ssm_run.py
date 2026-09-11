#!/usr/bin/env python3
"""Render Chinese post-hoc figures from a completed adaptive Croce-like SSM run.

The renderer consumes only the run's exported ``trajectories.csv``,
``subject_metrics.csv`` and ``fit_parameters.csv``.  It never re-fits the
model and deliberately does not invent channel-wise spatial maps: the legacy
adaptive export contains selected-channel aggregates, not a full geometry
field.  Existing files are never replaced; a suffixed output directory is
used on repeated renders.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

try:
    from src.visualization.token_physiology_plots import save_figure_atomic
except ModuleNotFoundError:  # direct ``python experiments/scripts/...`` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.visualization.token_physiology_plots import save_figure_atomic


MODEL_LABELS = {
    "adaptive_joint": "联合观测（固定区间后验）",
    "adaptive_eeg_only": "仅 EEG（跨模态对照）",
    "adaptive_fnirs_only": "仅 fNIRS",
}
MODEL_COLORS = {
    "adaptive_joint": "#D55E00",
    "adaptive_eeg_only": "#0072B2",
    "adaptive_fnirs_only": "#009E73",
}
SPATIAL_LABELS = {"local": "局部", "global": "全局"}
OBS_COLOR = "#333333"
GRID_COLOR = "#D9D9D9"
MISSING_COLOR = "#8A8A8A"
STATE_LABELS = {
    "vasodilation_s": "血管舒张状态 s",
    "flow_delta": "相对血流变化 Δf",
    "hbo_state": "HbO 状态",
    "hbr_state": "HbR 状态",
    "shared_driver": "共享神经驱动 r(t)",
}
SIGNALS = (
    ("eeg_observation", "eeg_reconstruction", "eeg_predictive_std", "eeg_valid", "EEG 代理"),
    ("hbo_truth", "hbo_estimate", "hbo_predictive_std", "fnirs_valid", "HbO"),
    ("hbr_truth", "hbr_estimate", "hbr_predictive_std", "fnirs_valid", "HbR"),
)

CJK_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "AR PL UMing CN",
        "DejaVu Sans",
    ],
    "axes.unicode_minus": True,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


def _read_csv(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少 adaptive run 导出文件：{path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fields = set(rows[0]) if rows else set()
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"{path.name} 缺少字段：{', '.join(missing)}")
    if not rows:
        raise ValueError(f"{path} 没有数据行")
    return rows


def _float(row: Mapping[str, Any], key: str, default: float = np.nan) -> float:
    try:
        value = row.get(key, "")
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _array(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_float(row, key) for row in rows], dtype=float)


def _bool_array(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    if not rows or key not in rows[0]:
        # Older adaptive exports predate explicit masks; finite values are the
        # only support information available in that contract.
        return np.ones(len(rows), dtype=bool)
    values = []
    for row in rows:
        raw = str(row.get(key, "")).strip().lower()
        values.append(raw in {"1", "true", "t", "yes", "y"})
    return np.asarray(values, dtype=bool)


def _group_trajectories(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int, str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str, int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["condition_id"]),
            str(row["subject"]),
            int(float(row["heldout_trial"])),
            str(row["model"]),
            str(row["spatial_mode"]),
        )
        groups[key].append(dict(row))
    for values in groups.values():
        values.sort(key=lambda row: _float(row, "time_s"))
    return dict(groups)


def _sample_keys(
    groups: Mapping[tuple[str, str, int, str, str], Sequence[Mapping[str, Any]]],
    *,
    max_samples: int,
) -> list[tuple[str, str, int, str, str]]:
    preferred = [key for key in groups if key[3] == "adaptive_joint" and key[4] == "local"]
    candidates = sorted(preferred or groups)
    # Deterministic round-robin over condition/subject prevents a small sample
    # budget from silently selecting every trajectory from subject_01.
    by_subject: dict[tuple[str, str], list[tuple[str, str, int, str, str]]] = defaultdict(list)
    for key in candidates:
        by_subject[(key[0], key[1])].append(key)
    ordered_subjects = sorted(by_subject)
    selected: list[tuple[str, str, int, str, str]] = []
    for fold_index in range(max(len(values) for values in by_subject.values())):
        for subject in ordered_subjects:
            values = by_subject[subject]
            if fold_index < len(values):
                selected.append(values[fold_index])
                if len(selected) >= max(1, int(max_samples)):
                    return selected
    return selected


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " "))


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#CC79A7")


def _spatial_label(mode: str) -> str:
    return SPATIAL_LABELS.get(mode, mode)


def _masked(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    mask = np.isfinite(result)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    result[~mask] = np.nan
    return result


def _plot_band(
    ax: Any,
    time: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    valid: np.ndarray | None,
    *,
    color: str,
    label: str,
    band_label: str = "95% 后验预测带",
) -> None:
    mean = _masked(mean, valid)
    std = _masked(std, valid)
    band_valid = np.isfinite(mean) & np.isfinite(std) & (std >= 0)
    if np.any(band_valid):
        low = np.where(band_valid, mean - 1.96 * std, np.nan)
        high = np.where(band_valid, mean + 1.96 * std, np.nan)
        ax.fill_between(time, low, high, color=color, alpha=0.14, linewidth=0, label=band_label)
    ax.plot(time, mean, color=color, linewidth=1.25, label=label)


def _style_axes(axes: Iterable[Any]) -> None:
    for ax in axes:
        ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.75)
        ax.axvline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)


def _save(fig: Any, output_dir: Path, stem: str) -> list[Path]:
    artifacts = save_figure_atomic(
        fig,
        output_dir / stem,
        formats="png",
        dpi=240,
    )
    plt.close(fig)
    return list(artifacts.figure_paths)


def _plot_reconstruction(
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    key: tuple[str, str, int, str, str],
    output_dir: Path,
    index: int,
) -> list[Path]:
    models = sorted(selected, key=lambda model: (model != "adaptive_joint", model))
    fig, axes = plt.subplots(6, len(models), figsize=(6.0 * len(models), 14.5), sharex="col", squeeze=False, constrained_layout=True)
    for column, model in enumerate(models):
        rows = selected[model]
        time = _array(rows, "time_s")
        for signal_index, (observed_key, estimate_key, std_key, valid_key, label) in enumerate(SIGNALS):
            valid = _bool_array(rows, valid_key)
            observed = _masked(_array(rows, observed_key), valid)
            estimate = _masked(_array(rows, estimate_key), valid)
            std = _masked(_array(rows, std_key), valid)
            color = _model_color(model)
            ax = axes[signal_index, column]
            ax.plot(time, observed, color=OBS_COLOR, linewidth=1.0, label="原始观测")
            _plot_band(ax, time, estimate, std, valid, color=color, label="状态空间重建")
            ax.set_ylabel(label)
            ax.set_title(_model_label(model) if signal_index == 0 else "")
            residual = observed - estimate
            axes[signal_index + 3, column].plot(time, residual, color="#AA3377", linewidth=0.9, label="观测−重建")
            axes[signal_index + 3, column].axhline(0.0, color="#555555", linewidth=0.7)
            axes[signal_index + 3, column].set_ylabel(f"{label}\n残差")
        axes[-1, column].set_xlabel("事件相对时间（秒）")
        for row_index in range(3):
            axes[row_index, column].legend(loc="best")
            axes[row_index + 3, column].legend(loc="best")
    _style_axes(axes.flat)
    fig.suptitle(
        f"原始观测与状态空间重建对比（{key[0]} / {key[1]} / 留出试次 {key[2]} / {key[4]}）\n"
        "固定排序样本；联合观测为同点平滑诊断，不能视为跨模态预测",
        fontsize=13,
    )
    return _save(fig, output_dir, f"重建对比_{index:02d}")


def _plot_states(
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    key: tuple[str, str, int, str, str],
    output_dir: Path,
    index: int,
) -> list[Path]:
    models = sorted(selected, key=lambda model: (model != "adaptive_joint", model))
    state_names = [name for name in STATE_LABELS if f"{name}_std" in selected[models[0]][0]]
    fig, axes = plt.subplots(len(state_names), 1, figsize=(10.0, 2.25 * len(state_names)), sharex=True, squeeze=False, constrained_layout=True)
    axes = axes[:, 0]
    for model in models:
        rows = selected[model]
        time = _array(rows, "time_s")
        color = _model_color(model)
        for ax, name in zip(axes, state_names, strict=True):
            measurement_gauge = name in {"hbo_state", "hbr_state"} and f"target_{name}" in rows[0]
            value_key = f"target_{name}" if measurement_gauge else name
            std_key = f"target_{name}_std" if measurement_gauge else f"{name}_std"
            state = _array(rows, value_key)
            std = _array(rows, std_key)
            _plot_band(
                ax,
                time,
                state,
                std,
                None,
                color=color,
                label=_model_label(model),
                band_label="95% 状态后验区间",
            )
    for ax, name in zip(axes, state_names, strict=True):
        suffix = "（测量坐标）" if name in {"hbo_state", "hbr_state"} else ""
        ax.set_ylabel(f"{STATE_LABELS[name]}{suffix}")
        ax.legend(loc="best", ncol=2)
    axes[-1].set_xlabel("事件相对时间（秒）")
    _style_axes(axes)
    fig.suptitle(
        f"共享神经状态与血流动力学状态轨迹（{key[0]} / {key[1]} / 试次 {key[2]}）\n"
        "HbO/HbR 映射到测量坐标；其余为模型状态坐标，不能解释为唯一生理量",
        fontsize=13,
    )
    return _save(fig, output_dir, f"共享状态_{index:02d}")


def _plot_uncertainty(
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    key: tuple[str, str, int, str, str],
    output_dir: Path,
    index: int,
) -> list[Path]:
    models = sorted(selected, key=lambda model: (model != "adaptive_joint", model))
    fig = plt.figure(figsize=(14.0, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.15))
    axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    state_axis = fig.add_subplot(grid[1, :])
    signal_std_specs = (
        ("eeg_predictive_std", "EEG 代理", "-"),
        ("hbo_predictive_std", "HbO", "--"),
        ("hbr_predictive_std", "HbR", ":"),
    )
    observed_std = [False, False, False]
    for model in models:
        rows = selected[model]
        time = _array(rows, "time_s")
        color = _model_color(model)
        for axis_index, (axis, (std_key, label, linestyle)) in enumerate(zip(axes, signal_std_specs, strict=True)):
            values = _masked(_array(rows, std_key))
            observed_std[axis_index] |= bool(np.any(np.isfinite(values)))
            axis.plot(
                time,
                values,
                color=color,
                linestyle=linestyle,
                linewidth=1.1,
                alpha=0.82,
                label=_model_label(model),
            )
        driver_std = _array(rows, "shared_driver_std")
        state_axis.plot(time, driver_std, color=color, linewidth=1.2, label=_model_label(model))
    for axis, (_std_key, label, _linestyle) in zip(axes, signal_std_specs, strict=True):
        axis.set_ylabel("预测标准差")
        axis.set_title(f"{label}（线型区分坐标）")
    for axis_index, axis in enumerate(axes):
        if observed_std[axis_index]:
            axis.legend(loc="best")
        else:
            axis.text(0.5, 0.5, "当前 CSV 未导出该标准差", transform=axis.transAxes, ha="center", va="center", color=MISSING_COLOR)
    state_axis.set_ylabel("状态后验标准差")
    state_axis.set_xlabel("事件相对时间（秒）")
    state_axis.set_title("共享神经驱动的后验不确定性")
    state_axis.legend(loc="best", ncol=2)
    for ax in [*axes, state_axis]:
        _style_axes([ax])
    fig.suptitle(
        f"SSM 不确定性诊断（{key[0]} / {key[1]} / 试次 {key[2]}）\n"
        "现有 CSV 仅提供观测预测/状态后验标准差；未提供偶然—模型—总不确定性分解",
        fontsize=13,
    )
    return _save(fig, output_dir, f"不确定性_{index:02d}")


def _finite_segments(values: np.ndarray, minimum: int = 16) -> list[np.ndarray]:
    finite = np.isfinite(values)
    segments: list[np.ndarray] = []
    start: int | None = None
    for index, present in enumerate(np.r_[finite, False]):
        if present and start is None:
            start = index
        elif not present and start is not None:
            if index - start >= minimum:
                segments.append(values[start:index])
            start = None
    return segments


def _psd(values: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray] | None:
    spectra = []
    frequencies = None
    for segment in _finite_segments(values):
        centred = segment - np.mean(segment)
        window = np.hanning(len(centred))
        scale = max(float(np.sum(window * window)), 1e-12)
        spectrum = np.abs(np.fft.rfft(centred * window)) ** 2 / scale
        current_f = np.fft.rfftfreq(len(centred), d=dt)
        if frequencies is None:
            frequencies = current_f
            spectra.append(spectrum)
        else:
            common = min(len(frequencies), len(current_f))
            frequencies = frequencies[:common]
            spectra = [item[:common] for item in spectra]
            spectra.append(spectrum[:common])
    if not spectra or frequencies is None:
        return None
    return frequencies, np.nanmean(np.vstack(spectra), axis=0)


def _plot_spectrum(
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    key: tuple[str, str, int, str, str],
    output_dir: Path,
    index: int,
) -> list[Path]:
    models = sorted(selected, key=lambda model: (model != "adaptive_joint", model))
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 9.0), sharex=True, constrained_layout=True)
    dt = 0.1
    for signal_index, (observed_key, estimate_key, _std_key, valid_key, label) in enumerate(SIGNALS):
        ax = axes[signal_index]
        for model in models:
            rows = selected[model]
            valid = _bool_array(rows, valid_key)
            observed = _masked(_array(rows, observed_key), valid)
            estimate = _masked(_array(rows, estimate_key), valid)
            for values, linestyle, suffix in ((observed, "-", "原始"), (estimate, "--", "重建")):
                result = _psd(values, dt)
                if result is None:
                    continue
                frequency, power = result
                positive = frequency > 0
                ax.loglog(frequency[positive], np.maximum(power[positive], 1e-14), linestyle=linestyle, color=_model_color(model), linewidth=1.0, label=f"{_model_label(model)}：{suffix}")
        ax.set_ylabel(f"{label}\nPSD")
        ax.legend(loc="best", ncol=2)
    axes[-1].set_xlabel("频率（Hz，对数坐标）")
    _style_axes(axes)
    fig.suptitle(
        f"原始信号与重建信号的功率谱密度（{key[0]} / {key[1]} / 试次 {key[2]}）\n"
        "仅使用完整连续片段，缺口未做插值；频谱形状不等于物理可辨识性",
        fontsize=13,
    )
    return _save(fig, output_dir, f"频谱对比_{index:02d}")


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    value = _float(row, metric)
    return value if np.isfinite(value) else np.nan


def _plot_subject_metrics(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    groups = sorted({(str(row.get("model", "")), str(row.get("spatial_mode", ""))) for row in rows})
    metric_families = (
        (
            "留出 NRMSE（越低越好）",
            ("trajectory_deviation_nrmse", "hbr_trajectory_deviation_nrmse", "eeg_trajectory_deviation_nrmse"),
            ("HbO", "HbR", "EEG 代理"),
        ),
        (
            "95% 预测覆盖率",
            ("predictive_95_coverage", "hbr_predictive_95_coverage", "eeg_predictive_95_coverage"),
            ("HbO", "HbR", "EEG 代理"),
        ),
        (
            "重建/观测时间标准差比",
            ("temporal_sd_ratio", "hbr_temporal_sd_ratio", "eeg_temporal_sd_ratio"),
            ("HbO", "HbR", "EEG 代理"),
        ),
    )
    available = [
        family for family in metric_families
        if any(metric in rows[0] for metric in family[1])
    ]
    if not available:
        available = [("旧导出 R²/相关/方差比", ("r2", "hbr_r2", "eeg_r2"), ("HbO", "HbR", "EEG 代理")), ("HbO 方差比", ("variance_ratio",), ("HbO",))]
    panel_count = sum(max(1, sum(metric in rows[0] for metric in metrics)) for _title, metrics, _labels in available)
    columns = 3
    figure_rows = int(np.ceil(panel_count / columns))
    fig, axes = plt.subplots(figure_rows, columns, figsize=(15.0, 4.0 * figure_rows), squeeze=False, constrained_layout=True)
    axes_flat = list(axes.flat)
    x = np.arange(len(groups), dtype=float)
    panel_index = 0
    for family_title, family_metrics, modality_labels in available:
        for metric, modality in zip(family_metrics, modality_labels, strict=True):
            if metric not in rows[0]:
                continue
            ax = axes_flat[panel_index]
            panel_index += 1
            for position, (model, spatial) in enumerate(groups):
                values = np.asarray([_metric_value(row, metric) for row in rows if str(row.get("model")) == model and str(row.get("spatial_mode")) == spatial], dtype=float)
                values = values[np.isfinite(values)]
                if not len(values):
                    continue
                jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.zeros(1)
                ax.scatter(np.full(len(values), position) + jitter, values, color=_model_color(model), alpha=0.72, s=22, label=_model_label(model) if position == 0 else None)
                ax.plot([position - 0.16, position + 0.16], [np.median(values)] * 2, color="#111111", linewidth=1.5)
            ax.set_title(f"{family_title}\n{modality}")
            ax.set_xticks(x, [f"{_model_label(model)}\n{_spatial_label(spatial)}" for model, spatial in groups], rotation=12, ha="right")
            if "coverage" in metric:
                ax.axhline(0.95, color="#555555", linestyle="--", linewidth=0.9, label="名义覆盖率 = 95%")
            elif "sd_ratio" in metric or "variance_ratio" in metric:
                ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.9, label="理想比例 = 1")
            ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    for ax in axes_flat[panel_index:]:
        ax.set_visible(False)
    unique_models = sorted({model for model, _spatial in groups}, key=lambda model: model != "adaptive_joint")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=_model_color(model), markersize=6, label=_model_label(model)) for model in unique_models]
    handles.append(Line2D([0], [0], color="#111111", linewidth=1.5, label="受试者中位数"))
    axes_flat[0].legend(handles=handles, loc="best", fontsize=8)
    fig.suptitle("受试者等权的 adaptive SSM 性能分布\n点为受试者，横线为中位数；优先显示留出可靠性指标", fontsize=13)
    return _save(fig, output_dir, "受试者性能")


def _plot_support_parameters(
    trajectory_rows: Sequence[Mapping[str, Any]],
    fit_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    models = sorted({str(row.get("model", "")) for row in trajectory_rows})
    support_items = [(key, label, valid) for key, _estimate, _std, valid, label in SIGNALS]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), gridspec_kw={"width_ratios": [1.0, 1.8]}, constrained_layout=True)
    support = np.full((len(models), len(support_items)), np.nan)
    for i, model in enumerate(models):
        model_rows = [row for row in trajectory_rows if str(row.get("model")) == model]
        for j, (_key, _label, valid_key) in enumerate(support_items):
            flags = _bool_array(model_rows, valid_key)
            support[i, j] = float(np.mean(flags)) if len(flags) else np.nan
    image = axes[0].imshow(support, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(len(support_items)), [item[1] for item in support_items], rotation=20, ha="right")
    axes[0].set_yticks(range(len(models)), [_model_label(model) for model in models])
    axes[0].set_title("导出轨迹有效率（非模型输入使用率）")
    axes[0].set_xlabel("观测量")
    axes[0].set_ylabel("模型")
    for i in range(len(models)):
        for j in range(len(support_items)):
            if np.isfinite(support[i, j]):
                axes[0].text(j, i, f"{support[i, j]:.0%}", ha="center", va="center", color="white" if support[i, j] < 0.65 else "black", fontsize=9)
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04, label="有效比例")

    parameters = [name for name in ("kas", "kaf", "tau0", "alpha", "e0", "phi", "q_driver", "q_scale", "fnirs_noise_scale") if name in fit_rows[0]]
    parameter_values = []
    labels = []
    for name in parameters:
        values = np.asarray([_float(row, name) for row in fit_rows], dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            parameter_values.append(values)
            labels.append(name)
    axes[1].boxplot(parameter_values, tick_labels=labels, showfliers=True, vert=False, patch_artist=True, boxprops={"facecolor": "#BFD7EA", "alpha": 0.85}, medianprops={"color": "#111111"})
    axes[1].set_title("旧 adaptive SSM 拟合参数分布")
    axes[1].set_xlabel("参数值（仅作拟合诊断）")
    axes[1].grid(axis="x", color=GRID_COLOR, linewidth=0.6)
    fig.suptitle("导出支持与参数稳定性诊断\n有效率描述输出轨迹，不表示相应模态进入了模型；旧 CSV 只有选中通道聚合", fontsize=13)
    return _save(fig, output_dir, "支持与参数")


def _plot_trajectory_heatmaps(trajectory_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    groups = _group_trajectories(trajectory_rows)
    keys = sorted(key for key in groups if key[3] == "adaptive_joint" and key[4] == "local")
    if not keys:
        keys = sorted(key for key in groups if key[3] == "adaptive_joint")
    if not keys:
        return []
    time_values = sorted({_float(row, "time_s") for key in keys for row in groups[key] if np.isfinite(_float(row, "time_s"))})
    time_index = {value: index for index, value in enumerate(time_values)}
    matrices = [np.full((len(keys), len(time_values)), np.nan, dtype=float) for _ in range(3)]
    for row_index, key in enumerate(keys):
        for row in groups[key]:
            time = _float(row, "time_s")
            column = time_index.get(time)
            if column is None:
                continue
            driver = _float(row, "shared_driver")
            fnirs_valid = _bool_array([row], "fnirs_valid")[0]
            hbo_residual = _float(row, "hbo_truth") - _float(row, "hbo_estimate") if fnirs_valid else np.nan
            hbr_residual = _float(row, "hbr_truth") - _float(row, "hbr_estimate") if fnirs_valid else np.nan
            matrices[0][row_index, column] = driver
            matrices[1][row_index, column] = hbo_residual
            matrices[2][row_index, column] = hbr_residual
    if not any(np.any(np.isfinite(matrix)) for matrix in matrices):
        return []
    cmap = mpl.colormaps.get_cmap("RdBu_r").copy()
    cmap.set_bad(MISSING_COLOR)
    fig, axes = plt.subplots(3, 1, figsize=(13.0, 10.5), sharex=True, constrained_layout=True)
    labels = ("共享神经驱动 r(t)", "HbO 残差（观测−重建）", "HbR 残差（观测−重建）")
    images = []
    for axis, matrix, label in zip(axes, matrices, labels, strict=True):
        finite = np.abs(matrix[np.isfinite(matrix)])
        limit = max(float(np.quantile(finite, 0.98)) if len(finite) else 1.0, 1e-12)
        image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", origin="lower", extent=(time_values[0], time_values[-1], 0, len(keys)), cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest")
        images.append(image)
        axis.set_ylabel(f"{label}\n轨迹索引")
        axis.axvline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        axis.set_yticks([0, max(0, len(keys) - 1)] if len(keys) > 1 else [0], ["1", str(len(keys))] if len(keys) > 1 else ["1"])
        fig.colorbar(image, ax=axis, fraction=0.018, pad=0.02, label="面板原始量纲")
    axes[-1].set_xlabel("事件相对时间（秒）")
    fig.suptitle(
        f"共享神经驱动与波形残差的时间×轨迹热图（n={len(keys)}）\n"
        "仅 adaptive_joint；行按 condition/subject/trial 固定排序；不同量纲面板分别使用对称色阶",
        fontsize=13,
    )
    return _save(fig, output_dir, "共享驱动与残差热力图")


def _plot_channel_selection(fit_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    fields = (("selected_eeg_channels", "EEG"), ("selected_fnirs_channels", "fNIRS"))
    if not fit_rows or not any(field in fit_rows[0] for field, _label in fields):
        return []
    groups = sorted({str(row.get("model") or row.get("spatial_mode") or "拟合") for row in fit_rows})
    matrices: list[np.ndarray] = []
    labels: list[list[str]] = []
    for field, _field_label in fields:
        counters = {group: Counter() for group in groups}
        for row in fit_rows:
            group = str(row.get("model") or row.get("spatial_mode") or "拟合")
            value = str(row.get(field, ""))
            counters[group].update(channel for channel in value.split("|") if channel)
        all_counts: Counter[str] = Counter()
        for counter in counters.values():
            all_counts.update(counter)
        top_channels = [channel for channel, _count in all_counts.most_common(15)]
        if not top_channels:
            matrices.append(np.empty((len(groups), 0)))
            labels.append([])
            continue
        matrix = np.asarray([[counters[group][channel] for channel in top_channels] for group in groups], dtype=float)
        matrices.append(matrix)
        labels.append(top_channels)
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.0), squeeze=False, constrained_layout=True)
    for axis, matrix, channel_labels, (_field, field_label) in zip(axes[0], matrices, labels, fields, strict=True):
        if not channel_labels:
            axis.text(0.5, 0.5, "未导出选中通道", transform=axis.transAxes, ha="center", va="center")
            axis.set_axis_off()
            continue
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
        axis.set_xticks(range(len(channel_labels)), channel_labels, rotation=55, ha="right")
        axis.set_yticks(range(len(groups)), [_spatial_label(group) for group in groups])
        axis.set_title(f"{field_label} 选中通道频次")
        axis.set_xlabel("通道名称")
        axis.set_ylabel("模型/空间模式")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(column_index, row_index, f"{int(matrix[row_index, column_index])}", ha="center", va="center", color="white" if matrix[row_index, column_index] < np.nanmax(matrix) * 0.55 else "black", fontsize=7)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="拟合折次数")
    fig.suptitle("选中通道分布（非活动热力图）\n仅表示 adaptive SSM 拟合时的通道选择频次，不表示神经活动强度", fontsize=13)
    return _save(fig, output_dir, "选中通道分布")


def _plot_driver_agreement(
    groups: Mapping[tuple[str, str, int, str, str], Sequence[Mapping[str, Any]]],
    output_dir: Path,
) -> list[Path]:
    by_subject: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for key, joint_rows in groups.items():
        condition, subject, trial, model, spatial = key
        if model != "adaptive_joint":
            continue
        eeg_key = (condition, subject, trial, "adaptive_eeg_only", spatial)
        if eeg_key not in groups:
            continue
        joint = _array(joint_rows, "shared_driver")
        eeg_only = _array(groups[eeg_key], "shared_driver")
        valid = np.isfinite(joint) & np.isfinite(eeg_only)
        if np.count_nonzero(valid) < 3:
            continue
        joint = joint[valid]
        eeg_only = eeg_only[valid]
        denominator = float(np.std(joint))
        correlation = float(np.corrcoef(joint, eeg_only)[0, 1]) if np.std(eeg_only) > 0 and denominator > 0 else np.nan
        nrmse = float(np.sqrt(np.mean((joint - eeg_only) ** 2)) / denominator) if denominator > 1e-8 else np.nan
        by_subject[(condition, subject, spatial)].append((correlation, nrmse))
    subject_rows = [
        (condition, subject, spatial, float(np.nanmean([value[0] for value in values])), float(np.nanmean([value[1] for value in values])))
        for (condition, subject, spatial), values in sorted(by_subject.items())
    ]
    comparison_groups = sorted({(condition, spatial) for condition, _subject, spatial, _corr, _nrmse in subject_rows})
    if not subject_rows or not comparison_groups:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5), constrained_layout=True)
    for axis, value_index, title, reference in (
        (axes[0], 3, "joint 与仅 EEG 共享驱动的时间相关", 1.0),
        (axes[1], 4, "joint 与仅 EEG 共享驱动的归一化差异", 0.0),
    ):
        for position, (condition, spatial) in enumerate(comparison_groups):
            values = np.asarray([
                row[value_index] for row in subject_rows if row[0] == condition and row[2] == spatial
            ], dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            jitter = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else np.zeros(1)
            axis.scatter(np.full(len(values), position) + jitter, values, color="#CC79A7", s=30, alpha=0.8)
            axis.plot([position - 0.15, position + 0.15], [np.median(values)] * 2, color="#111111", linewidth=1.5)
        axis.axhline(reference, color="#555555", linestyle="--", linewidth=0.9)
        axis.set_xticks(
            range(len(comparison_groups)),
            [f"{condition}\n{_spatial_label(spatial)}" for condition, spatial in comparison_groups],
            rotation=12,
            ha="right",
        )
        axis.set_title(title)
        axis.set_ylabel("受试者内试次均值")
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    fig.suptitle(
        "共享神经驱动对观测模态的敏感性\n"
        "点为受试者；高相关和低归一化差异表示 joint 与仅 EEG 状态更一致，但不证明生理真值",
        fontsize=13,
    )
    return _save(fig, output_dir, "共享驱动模态一致性")


def _render(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    max_samples: int = 1,
) -> Path:
    trajectories = _read_csv(run_dir / "trajectories.csv", ("condition_id", "subject", "heldout_trial", "model", "spatial_mode", "time_s"))
    subject_metrics = _read_csv(run_dir / "subject_metrics.csv", ("model", "spatial_mode", "subject"))
    fit_parameters = _read_csv(run_dir / "fit_parameters.csv", ("condition_id",))
    groups = _group_trajectories(trajectories)
    keys = _sample_keys(groups, max_samples=max_samples)
    if output_dir is None:
        base = run_dir / "figures" / "adaptive_cn"
        output_dir = base
        suffix = 2
        while output_dir.exists() and any(output_dir.iterdir()):
            output_dir = base.with_name(f"{base.name}_v{suffix}")
            suffix += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index, sample_key in enumerate(keys, start=1):
        sample_models = {
            key[3]: values
            for key, values in groups.items()
            if key[:3] == sample_key[:3] and key[4] == sample_key[4]
        }
        outputs.extend(_plot_reconstruction(sample_models, sample_key, output_dir, index))
        outputs.extend(_plot_states(sample_models, sample_key, output_dir, index))
        outputs.extend(_plot_uncertainty(sample_models, sample_key, output_dir, index))
        outputs.extend(_plot_spectrum(sample_models, sample_key, output_dir, index))
    outputs.extend(_plot_subject_metrics(subject_metrics, output_dir))
    outputs.extend(_plot_support_parameters(trajectories, fit_parameters, output_dir))
    outputs.extend(_plot_trajectory_heatmaps(trajectories, output_dir))
    outputs.extend(_plot_channel_selection(fit_parameters, output_dir))
    outputs.extend(_plot_driver_agreement(groups, output_dir))
    summary = [
        "# Adaptive Croce-like SSM 中文可视化",
        "",
        f"输入目录：`{run_dir}`",
        f"轨迹样本数：{len(keys)}（按 condition/subject/trial/model/spatial_mode 固定排序选择，未按性能挑选）",
        "",
        "## 已生成",
        "",
        *[f"- `{path.name}`" for path in outputs if path.suffix == ".png"],
        "",
        "## 边界",
        "",
        "- `adaptive_joint` 是联合 EEG/fNIRS 固定区间平滑后验；`adaptive_eeg_only` 是跨模态对照。",
        "- 旧 CSV 没有一步预测残差、PIT、逐通道全脑坐标、偶然/模型不确定性分量，因此这些图未伪造。",
        "- EEG 图中的 EEG 是旧实验的 log-power PCA 代理，不是原始 EEG 波形。",
        "- 当前空间热力图只能在后续导出逐通道值和坐标后实现。",
        "- 通道选择频率只表示训练折支持，不表示神经活动幅度。",
    ]
    (output_dir / "可视化说明.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return output_dir


def _synthetic_self_check() -> None:
    """Render a tiny temporary run, exercising missing gaps and CJK labels."""
    with tempfile.TemporaryDirectory(prefix="adaptive_ssm_visual_check_") as temp:
        run_dir = Path(temp)
        rows = []
        for model in ("adaptive_joint", "adaptive_eeg_only"):
            for index in range(32):
                time = index / 10.0 - 1.6
                missing = index in {8, 9}
                row: dict[str, Any] = {
                    "condition_id": "synthetic",
                    "subject": "subject_01",
                    "heldout_trial": "0",
                    "model": model,
                    "spatial_mode": "local",
                    "time_s": str(time),
                    "eeg_observation": str(np.sin(time)),
                    "eeg_reconstruction": str(np.sin(time) * 0.9),
                    "eeg_valid": "False" if missing else "True",
                    "eeg_predictive_std": "0.1",
                    "hbo_truth": str(np.cos(time)),
                    "hbo_estimate": str(np.cos(time) * 0.9),
                    "hbo_predictive_std": "0.1",
                    "hbr_truth": str(np.sin(time / 2.0)),
                    "hbr_estimate": str(np.sin(time / 2.0) * 0.9),
                    "hbr_predictive_std": "0.1",
                    "fnirs_valid": "False" if missing else "True",
                }
                for state_index, name in enumerate(STATE_LABELS):
                    row[name] = str(np.sin(time + state_index / 3.0))
                    row[f"{name}_std"] = "0.1"
                    row[f"target_{name}"] = str(np.sin(time + state_index / 3.0))
                rows.append(row)
        fields = list(rows[0])
        with (run_dir / "trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        metric_fields = ["condition_id", "subject", "model", "spatial_mode", "r2", "pcc", "variance_ratio", "eeg_r2"]
        with (run_dir / "subject_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=metric_fields)
            writer.writeheader()
            for model in ("adaptive_joint", "adaptive_eeg_only"):
                writer.writerow({"condition_id": "synthetic", "subject": "subject_01", "model": model, "spatial_mode": "local", "r2": "0.5", "pcc": "0.7", "variance_ratio": "1.0", "eeg_r2": "0.6"})
        parameter_fields = ["condition_id", "subject", "heldout_trial", "spatial_mode", "selected_eeg_channels", "selected_fnirs_channels", "kas", "kaf", "tau0", "alpha", "e0", "phi", "q_driver", "q_scale", "fnirs_noise_scale"]
        with (run_dir / "fit_parameters.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=parameter_fields)
            writer.writeheader()
            row = {field: "0.5" for field in parameter_fields}
            row.update({"condition_id": "synthetic", "subject": "subject_01", "heldout_trial": "0", "spatial_mode": "local", "selected_eeg_channels": "C3|C4", "selected_fnirs_channels": "S1D1_HbO|S1D1_HbR"})
            writer.writerow(row)
        output = _render(run_dir)
        assert (output / "重建对比_01.png").exists()
        assert (output / "共享驱动与残差热力图.png").exists()
        assert (output / "选中通道分布.png").exists()
        assert (output / "共享驱动模态一致性.png").exists()
        assert not list(output.glob("*.svg"))
        assert (output / "可视化说明.md").exists()
    print("synthetic self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, help="completed adaptive run directory")
    parser.add_argument("--output-dir", type=Path, help="figure output directory; existing files are never replaced")
    parser.add_argument("--max-samples", type=int, default=1, help="number of deterministic trajectory samples (default: 1)")
    parser.add_argument("--self-check", action="store_true", help="render a temporary synthetic run and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        with mpl.rc_context(CJK_STYLE):
            _synthetic_self_check()
        return
    if args.run_dir is None:
        raise SystemExit("需要提供 run_dir，或使用 --self-check")
    with mpl.rc_context(CJK_STYLE):
        output = _render(args.run_dir.resolve(), output_dir=args.output_dir, max_samples=args.max_samples)
    print(output)


if __name__ == "__main__":
    main()
