#!/usr/bin/env python3
"""Render retained measured model-compatibility evidence as a Chinese report.

The report is a communication export, not a new experiment or an authorization
controller. All fitting stays in the owning evaluator. Only explicitly selected
native Hb windows may be replayed through that evaluator's existing reader for
their explanatory figures; their scores must reproduce the retained rows.
Figures and formulas are PNG images; body text and tables remain PDF text.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "model_compatibility_measured_report_v1"
DATASET_NAMES = {
    "eeg_fnirs_single_trial": "Single-Trial",
    "simultaneous_eeg_nirs": "Simultaneous",
    "visual_cognitive_motivation": "Visual",
    "refed": "REFED",
}
DATASETS = tuple(DATASET_NAMES)
COLORS = {"nominal": "#b54e40", "best": "#187969", "HbO": "#c95743",
          "HbR": "#287bb5", "EEG": "#6856a6", "residual": "#ac3970",
          "gray": "#71808e"}
DPI = 250


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(value, digits=3):
    """Use an explicit N/A label instead of inventing a numerical result."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(value):
        return "N/A"
    if value == 0:
        return "0"
    return f"{value:.{digits}g}"


def label(dataset):
    return DATASET_NAMES.get(str(dataset), str(dataset))


def ecdf(values, weights=None):
    """Exact weighted ECDF, retaining zero values and tied observations."""
    x = np.asarray(values, float)
    w = np.ones(len(x), float) if weights is None else np.asarray(weights, float)
    if x.ndim != 1 or w.shape != x.shape or np.any(w < 0):
        raise ValueError("ECDF needs a vector and nonnegative matching weights")
    keep = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[keep], w[keep]
    if not len(x):
        return np.array([]), np.array([])
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    unique, starts = np.unique(x, return_index=True)
    return unique, np.cumsum(np.add.reduceat(w, starts)) / w.sum()


def subject_values(frame, metric):
    """Pair/window -> record -> subject means, never pooled-window inference."""
    needed = {"dataset_id", "subject_id", "record_id", "length_s", metric}
    if not needed.issubset(frame.columns):
        raise ValueError(f"missing hierarchical summary columns: {sorted(needed-set(frame.columns))}")
    valid = frame.loc[np.isfinite(pd.to_numeric(frame[metric], errors="coerce"))]
    levels = ["dataset_id", "subject_id", "record_id", "length_s"]
    if "window_id" in valid:
        window = valid.groupby(levels + ["window_id"], observed=True)[metric].mean()
        record = window.groupby(level=levels, observed=True).mean()
    else:
        record = valid.groupby(levels, observed=True)[metric].mean()
    return record.groupby(level=["dataset_id", "subject_id", "length_s"], observed=True).mean().reset_index()


def subject_equal_weights(frame):
    """Weights for the full distribution: pair -> window -> record -> subject."""
    levels = ["dataset_id", "subject_id", "record_id", "length_s"]
    window_key = "window_id" if "window_id" in frame else "start_s"
    windows = levels + [window_key]
    per_window = frame.groupby(windows, observed=True)[window_key].transform("size")
    per_record = frame.groupby(levels, observed=True)[window_key].transform("nunique")
    per_subject = frame.groupby(["dataset_id", "subject_id", "length_s"], observed=True)["record_id"].transform("nunique")
    return np.asarray(1. / (per_window * per_record * per_subject), float)


def select_exemplars(frame, *, length_s=90, quantiles=(.1, .5, .9)):
    """Nearest fixed-score quantiles; choose distinct subjects when available.

    The requested quantile is calculated on the full successful pair/window
    denominator. Subject diversity only resolves the explanatory example choice;
    no score, threshold, or model parameter is selected from these examples.
    """
    needed = {"dataset_id", "subject_id", "record_id", "length_s", "nominal_D"}
    if not needed.issubset(frame.columns):
        raise ValueError("exemplar candidates lack identity or nominal_D")
    rows = []
    selected = frame.loc[(frame.length_s == length_s)
                         & np.isfinite(pd.to_numeric(frame.nominal_D, errors="coerce"))].copy()
    if "status" in selected:
        selected = selected.loc[selected.status == "ok"]
    keys = [c for c in ("subject_id", "record_id", "start_s", "pair_index") if c in selected]
    for dataset in DATASETS:
        candidates = selected.loc[selected.dataset_id == dataset].sort_values(keys, kind="stable")
        used = set()
        for q in quantiles:
            if candidates.empty:
                continue
            target = float(candidates.nominal_D.quantile(q))
            available = candidates.loc[~candidates.subject_id.astype(str).isin(used)]
            if available.empty:
                available = candidates
            distance = abs(available.nominal_D - target)
            row = available.loc[distance.idxmin()].to_dict()
            row.update(selection_quantile=float(q), selection_target_D=target,
                       selection_denominator=int(len(candidates)))
            rows.append(row)
            used.add(str(row["subject_id"]))
    return rows


def configure_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
                         "ytick.labelsize": 8.5, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18,
                         "lines.linewidth": 1.6, "savefig.facecolor": "white",
                         "figure.facecolor": "white"})
    return plt


def save_figure(fig, directory, name):
    import matplotlib.pyplot as plt
    path = Path(directory) / f"{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


def score_axis(ax, *, axis="x", linthresh=1e-6):
    """Logarithmic outside a labelled neighbourhood of zero; no zero dropping."""
    getattr(ax, f"set_{axis}scale")("symlog", linthresh=linthresh, linscale=.6)
    getattr(ax, f"set_{axis}lim")(**({"left": 0} if axis == "x" else {"bottom": 0}))


def paragraph(text):
    return {"type": "paragraph", "text": text}


def figure(path, caption, source):
    return {"type": "figure", "path": str(path), "caption": caption, "source": str(source)}


def table(columns, rows, *, caption="", source="", widths=None):
    result = {"type": "table", "columns": columns, "rows": rows, "caption": caption, "source": source}
    if widths:
        result["widths"] = widths
    return result


def equation(latex, explanation, *, source="src/inference/model_compatibility.py"):
    return {"type": "equation", "latex": latex, "explanation": explanation, "source": source}


def _read_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required retained evidence is missing: {path}")
    try:
        return pd.read_csv(path, keep_default_na=True)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_census(run):
    """Read only the explicit completed census record list, never discover raw data."""
    run = Path(run)
    inventory_path = run / "record_inventory.jsonl"
    if not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)
    records = [json.loads(line) for line in inventory_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, metas = [], {}
    for record in records:
        token = record.get("record_token", record.get("token"))
        if not token:
            raise ValueError("each census record must retain its explicit record_token")
        csv_path = run / "records" / f"{token}.csv.gz"
        meta_path = csv_path.with_name(f"{token}.meta.json")
        meta = read_json(meta_path)
        metas[str(token)] = meta
        if meta.get("execution") not in ("completed", "failed"):
            raise ValueError(f"census record has no terminal state: {meta_path}")
        if meta.get("execution") == "failed" and not csv_path.exists():
            continue
        frame = _read_csv(csv_path)
        required = {"dataset_id", "subject_id", "record_id", "record_token", "length_s", "status", "nominal_D", "best_D"}
        if not required.issubset(frame):
            raise ValueError(f"incomplete census CSV contract: {csv_path}: {sorted(required-set(frame))}")
        rows.append(frame)
    if not rows:
        raise ValueError("empty census inventory")
    return pd.concat(rows, ignore_index=True), records, metas


def coverage_blocks(frame, inventory, metas, source, *, cached=False):
    counts, support_counts, record_failures, units = [], [], [], []
    for dataset in DATASETS:
        for length in sorted(frame.length_s.unique()):
            part = frame.loc[(frame.dataset_id == dataset) & (frame.length_s == length)]
            statuses = Counter(part.status)
            admitted = [r for r in inventory if r["dataset_id"] == dataset]
            counts.append([label(dataset), int(length), len({r["subject_id"] for r in admitted}), len(admitted),
                           len(part), statuses.get("ok", 0),
                           "; ".join(f"{key}: {value}" for key, value in sorted(statuses.items()) if key != "ok") or "0"])
            full_tokens = set(part.record_token.astype(str))
            valid_tokens = set(part.loc[part.status == "ok", "record_token"].astype(str))
            read_tokens = {str(m["record_token"]) for m in metas.values()
                           if m["dataset_id"] == dataset and m.get("execution") == "completed"}
            support_counts.append([label(dataset), int(length), len(admitted), len(full_tokens),
                                   len(valid_tokens), len(read_tokens-full_tokens)])
        failed = [m for m in metas.values() if m["dataset_id"] == dataset and m.get("execution") == "failed"]
        record_failures.append([label(dataset), len(failed), "; ".join(f"{m['subject_id']}/{m['record_id']}: {m.get('error_type', 'failure')}" for m in failed) or "无"])
        completed = [m for m in metas.values() if m["dataset_id"] == dataset and m.get("execution") == "completed"]
        field = 'measurement_provenance' if cached else 'native_provenance'
        native_units = sorted({str(m.get(field, {}).get('output_unit' if cached else 'native_unit', 'unknown')) for m in completed})
        transforms = sorted({str(m.get(field, {}).get('transform', 'unknown')) for m in completed})
        rate_field = 'sample_rate_hz' if cached else 'native_sample_rate_hz'
        rates = sorted({float(m[rate_field]) for m in completed if rate_field in m})
        units.append([label(dataset), "; ".join(native_units) or "N/A", "; ".join(transforms) or "N/A", "; ".join(number(v) for v in rates) or "N/A"])
    intro = ("本报告只汇总统一 v5/none 缓存中的 Hb 测量。输入已在完整记录上作 0.01–0.2 Hz 带通并重采样到10 Hz，不重复 MBLL、单位换算或带通。统计与曲线使用相同缓存数组；旧原生输入的拟合面板不混入本轮结果。" if cached else
             "本报告先给出全登记公开库存的 Hb 配对闭合分布，再在保留冻结坐标、训练尺度和观测算子的子面板检验 EEG–Hb、共同参数及条件非线性重建。两类分母分别计算。空间映射仅适用于其 Single-Trial 冻结评价面板。全库存主分析使用原生 Hb 读取器，不沿用 v5 的额外滤波轨迹；原始发布处理和单位保留在逐记录 provenance。")
    return [paragraph(intro),
            table(["数据集", "窗长 / 秒", "计划被试", "计划记录", "原生配对窗", "有效评分", "其余状态（完整计数）"], counts,
                  caption="一个配对窗是一个记录中的一个时间窗与一个物理 HbO/HbR 对。窗口、通道和重复记录不构成独立被试。库内被试身份只在各数据集内部计数。", source=source,
                  widths=[1.1, .7, .75, .65, 1.1, 1, 2.2]),
            table(["数据集", "窗长/秒", "计划记录", "有整窗记录", "有有效评分记录", "已读但无整窗"], support_counts,
                  caption="有整窗记录按该窗长 CSV 中不同 record_token 计数，包含随后被排除或不可评分的整窗；有有效评分记录要求至少一个 status=ok 配对窗。已成功读取但长度不足一个完整窗口的记录只进入计划记录，不冒充该窗长的实测评分记录。这类支持不足与整记录读取失败分别保留。", source=source + "/records/*.csv.gz",
                  widths=[1.2, .7, .9, 1, 1.2, 1.1]),
            paragraph("30秒与90秒的非重叠窗口并非相同观测单元，且可用记录集合可能不同。因此两种窗长的分布差异仅作描述，不能归因为窗长的因果效应。合法初态贡献、弱核边缘支持或激励频率可能参与这些差异，但本组实验没有隔离这些机制，相关解释只能作为推断。"),
            table(["数据集", "整记录读取/计算失败", "保留失败身份和类型"], record_failures,
                  caption="原生配对窗数在实际读取原生时钟后确定。若整条记录读取失败，其真实窗口分母未知，不能用缓存时长估计冒充有效实测分母；计划记录仍保留。末尾不足整窗部分不补齐，拼接边界跨窗排除。", source=source),
            table(["数据集", "缓存输出单位" if cached else "metadata原生单位标注", "实际测量处理", "实际采样率/Hz"], units,
                  caption=("单位与处理来自缓存 provenance。Simultaneous 的 mmol/L→µM 已由缓存完成；Single-Trial 仍为相对 MBLL，未知单位保持未知。完整记录带通先于窗口划分，不能解释为未滤波瞬态或原始传感器隔离。" if cached else "单位和变换逐字来自保留 provenance。Single-Trial 从正双波长强度经自然对数光密度和相对 MBLL 得到 Hb；原生测量单位标注不构成变换后 Hb 的绝对标定。Visual/REFED 的物理单位未知时保留未知，跨库不汇集原幅值。普查无新增项目时间滤波；冻结面板另用原父算子。"), source=source + "/records/*.meta.json",
                  widths=[1, 1.6, 3, 1])]


def census_distribution_blocks(frame, directory, source, summary):
    plt = configure_plotting()
    blocks, stats = [], []
    common_max = max(.1, float(frame.loc[frame.status == "ok", ["nominal_D", "best_D"]].max().max()))*1.05
    for length in sorted(frame.length_s.unique()):
        part = frame.loc[(frame.length_s == length) & (frame.status == "ok")]
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.3), layout="constrained")
        for col, metric in enumerate(("nominal_D", "best_D")):
            for dataset in DATASETS:
                values = part.loc[part.dataset_id == dataset]
                x, p = ecdf(values[metric])
                axes[0, col].step(x, p, where="post", label=f"{label(dataset)}，n={len(values):,}配对窗")
                means = subject_values(values, metric)
                x, p = ecdf(values[metric], subject_equal_weights(values))
                axes[1, col].step(x, p, where="post", label=f"{label(dataset)}，n={len(means)}人")
                retained = summary.loc[(summary.dataset_id == dataset) & (summary.length_s == length) & (summary.metric == metric)]
                if len(retained) != 1:
                    raise ValueError(f"owning census summary must contain exactly one {dataset}/{length}/{metric} row")
                retained = retained.iloc[0]
                stats.append([label(dataset), int(length), "固定参数" if metric == "nominal_D" else "窗内最优",
                              int(retained.valid_pair_windows), int(retained.valid_subjects), number(retained.raw_median),
                              number(retained["mean"]), f"[{number(retained.mean_ci_low)}, {number(retained.mean_ci_high)}]"])
            for row in range(2):
                axes[row, col].set(xlabel="闭合能量比 D（0至0.1线性，其余对数）", ylabel="累计比例", ylim=(0, 1.02))
                score_axis(axes[row, col], linthresh=.1)
                axes[row, col].set_xlim(0, common_max)
                axes[row, col].legend(loc="lower right", fontsize=8.5)
            axes[0, col].set_title(("固定 τ = 2 秒" if metric == "nominal_D" else "同窗网格最小值") + "：原始配对窗")
            axes[1, col].set_title("记录内汇总，再被试等权")
        path = save_figure(fig, directory, f"census_ecdf_{int(length)}s")
        blocks.append(figure(path, f"{int(length)} 秒全库存 Hb 闭合分布。上排每个配对窗等权；下排仍显示逐窗 D，但将同窗通道、同记录窗口、同被试记录分别等权，再使每名被试总权重相同。这样长记录和通道多的记录不会主导下排分布。零值保留，横轴0至0.1线性、其余对数；0.1只是显示转换点，不是接收阈值。30/90秒与固定/最优图统一完整横轴范围，不删尾部。最优值是在同一观测窗口重新选择 τ，因而可小于固定参数值，不能解释为未来泛化收益。30/90秒使用不同窗口及可能不同的可用记录集合，跨窗长差异不作因果解释。", source))
    median_rows = []
    for dataset in DATASETS:
        for length in sorted(frame.length_s.unique()):
            part = summary.loc[(summary.dataset_id == dataset) & (summary.length_s == length)]
            by_metric = {r["metric"]: r for r in part.to_dict("records")}
            nominal, best, change = (by_metric[key] for key in ("nominal_D", "best_D", "improvement"))
            median_rows.append([label(dataset), int(length), number(nominal["q50"]), number(best["q50"]), number(change["q50"]), int(nominal["valid_subjects"])])
    blocks.append(table(["数据集", "秒", "固定D中位", "最优D中位", "配对改善中位", "有效被试"], median_rows,
                        caption="均为通道→窗口→记录→被试等权混合分布的精确中位数，直接引用 summary.csv q50。配对改善先对同一配对窗算固定D-最优D，再按相同层级汇总；它不同于两列中位数之差。30/90秒是不同支持分布。", source=source))
    blocks.append(table(["数据集", "秒", "参数", "有效配对窗", "被试", "原始中位 D", "被试等权均值", "均值 95% 区间"], stats,
                        caption="数值直接来自 owning summary.csv。95% 区间为保留合同定义的被试整簇百分位 bootstrap。通道→窗口→记录→被试的汇总顺序先固定，然后重采样被试；区间不刻画模型参数不确定性或传感器噪声。原始中位数与被试等权均值是不同统计量。", source=source,
                        widths=[1.05, .4, .75, 1, .5, .95, 1.05, 1.35]))
    return blocks, stats


def parameter_blocks(frame, directory, source):
    plt = configure_plotting()
    blocks = []
    good = frame.loc[frame.status == "ok"]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), layout="constrained")
    boundary_rows = []
    for ax, dataset in zip(axes.flat, DATASETS):
        for length, style in zip(sorted(good.length_s.unique()), ("-", "--")):
            part = good.loc[(good.dataset_id == dataset) & (good.length_s == length)]
            counts = part.best_tau.value_counts(normalize=True).sort_index()
            ax.plot(counts.index, counts.values, style, marker=".", label=f"{int(length)}秒，n={len(part):,}")
            if len(part):
                low, high = float(part.best_tau.min()), float(part.best_tau.max())
                boundary_rows.append([label(dataset), int(length), len(part),
                                      int((part.best_tau == .5).sum()), int((part.best_tau == 8).sum()),
                                      number(float(part.best_tau.median()))])
        ax.set(title=label(dataset), xlabel="配对窗最优 τ / 秒", ylabel="原始配对窗比例", xlim=(.4, 8.1))
        ax.legend()
    blocks.append(figure(save_figure(fig, directory, "tau_distribution"), "每个有效配对窗在同一离散网格内选择最小 D 对应的 τ。该值是相对 Hb 条件闭合的拟合参数，不是生理真值估计。边界集中只说明工程网格内的最优节点靠边；它既可能来自观测失配，也可能来自低激励或弱识别，不能直接当作异常生理。", source))
    blocks.append(table(["数据集", "秒", "有效配对窗", "τ=0.5秒", "τ=8秒", "τ中位数/秒"], boundary_rows,
                        caption="最优参数触边计数的分母包含全部有效配对窗；不可评分窗不纳入最优参数分布，已在覆盖表保留。", source=source))
    energy_key = next((c for c in ("denominator", "dynamic_energy", "energy") if c in good), None)
    if energy_key:
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), layout="constrained")
        for ax, dataset in zip(axes.flat, DATASETS):
            part = good.loc[good.dataset_id == dataset]
            x, y = np.asarray(part[energy_key], float), np.asarray(part.best_D, float)
            keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y >= 0)
            if keep.any():
                chart = ax.hexbin(np.log10(x[keep]), np.arcsinh(y[keep]/1e-3), gridsize=28, mincnt=1, bins="log", cmap="viridis")
                fig.colorbar(chart, ax=ax, label="配对窗数（对数色阶）")
            ax.set(title=label(dataset), xlabel="log10 固定动态能量分母（库内原生单位平方）", ylabel="asinh(最优 D / 0.001)")
        blocks.append(figure(save_figure(fig, directory, "dynamic_energy_relation"), "固定动态能量分母与最优闭合比。每个六边形汇总有效配对窗，颜色表示数量。纵轴 asinh 显示保留零值；横轴保留各库原生单位，跨库横轴量值不可直接比较。30/90 秒合并仅作能量关联诊断，完整窗长分布见前图。分母很小时较小或较大 D 都不能单独证明识别充分；不可评分的零能量另列状态。", source))
    return blocks


def stratification_blocks(frame, directory, source):
    """Descriptive strata use retained identities, never inferred task labels."""
    plt = configure_plotting()
    good = frame.loc[(frame.status == "ok") & (frame.length_s == 90)]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), layout="constrained")
    rows = []
    for ax, dataset in zip(axes.flat, DATASETS):
        part = good.loc[good.dataset_id == dataset]
        grouped = part.groupby("pair_index", observed=True)
        median = grouped[["nominal_D", "best_D"]].median()
        ax.plot(median.index + 1, median.nominal_D, ".-", color=COLORS["nominal"], label="固定参数")
        ax.plot(median.index + 1, median.best_D, ".--", color=COLORS["best"], label="逐窗最优")
        score_axis(ax, axis="y")
        ax.set(title=label(dataset), xlabel="原读出物理 Hb 对序号（1 起）", ylabel="90秒配对窗原始中位 D")
        ax.legend()
        for dimension, cn in (("subject_id", "被试"), ("record_id", "记录"), ("pair", "原始通道身份"), ("task_id", "原始任务标签")):
            if dimension not in part:
                rows.append([label(dataset), cn, "N/A", "未保留该标签；不由记录名推断"])
                continue
            values = part.groupby(dimension, observed=True).nominal_D.median()
            rows.append([label(dataset), cn, len(values), f"各层中位 D 的范围 [{number(values.min())}, {number(values.max())}]；各层分母不同"])
    blocks = [figure(save_figure(fig, directory, "channel_strata"), "每条线上的点对应原始物理 Hb 对序号，而非脑区。中位数在该通道全部有效 90 秒窗上计算，不把不同通道当作独立被试。跨库的同序号不表示同一空间位置；曲线高低可能同时受记录组成、幅度和闭合失配影响。", source),
              table(["数据集", "原生分层变量", "有效层数", "固定参数描述"], rows,
                    caption="只使用保留 metadata 中已有的身份或标签。层内中位数仅描述异质性，未进行通道、被试或任务间的统计检验。任务未保留时明确 N/A。", source=source)]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), layout="constrained")
    for ax, dataset in zip(axes.flat, DATASETS):
        part = good.loc[good.dataset_id == dataset]
        levels = part.groupby("start_s", observed=True).nominal_D.agg(["median", "count"])
        ax.plot(levels.index, levels["median"], ".-", color=COLORS["nominal"])
        score_axis(ax, axis="y")
        ax.set(title=label(dataset), xlabel="原生记录起点后的窗开始时间 / 秒", ylabel="有效配对窗原始中位 D")
    blocks.append(figure(save_figure(fig, directory, "record_time_strata"), "每个点汇总相同原生窗起点的有效 90 秒配对窗；各点包含的记录数可因记录长短而变化。时间零点是各记录起点，未统一成刺激或生理事件，因而这些线只描述记录内时间位置，不能称为任务诱发轨迹或时间因果效应。", source))
    return blocks


def prepare_hb_examples(run, frame, inventory):
    """Replay only 12 predeclared explanatory windows and retain exact arrays."""
    from experiments.scripts import evaluate_model_compatibility as evaluator
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    from src.inference.model_compatibility import weak_hb_profiles

    run = Path(run)
    example_dir = run / "examples"
    selection_path = example_dir / "hb_selection.json"
    choices = select_exemplars(frame)
    by_token = {str(r["record_token"]): r for r in inventory}
    if selection_path.exists():
        retained = read_json(selection_path)
        expected = [(r["record_token"], int(r["window_id"]), int(r["pair_index"]), r["selection_quantile"]) for r in choices]
        actual = [(r["record_token"], int(r["window_id"]), int(r["pair_index"]), r["selection_quantile"]) for r in retained["examples"]]
        if actual != expected:
            raise ValueError("retained illustrative selection differs; do not overwrite dated evidence")
        for item in retained["examples"]:
            if not (example_dir / item["arrays"]).is_file():
                raise FileNotFoundError(example_dir / item["arrays"])
        return retained["examples"]
    config = evaluator.load_config(run / "resolved_config.yaml")
    cache_root = Path(config["census"]["metadata_cache"])
    index = CleanPhysiologyCacheIndex(cache_root if cache_root.is_absolute() else ROOT / cache_root)
    example_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for row in choices:
        token = str(row["record_token"])
        admitted = by_token[token]
        if admitted["join_key"] != row["join_key"]:
            raise ValueError("example identity is outside the admitted census inventory")
        found = index.records_by_join_key[admitted["join_key"]]
        if len(found) != 1:
            raise ValueError("selected example must resolve to one native record")
        native = evaluator.replay_window(found[0], row, config=config, project_root=ROOT)
        pair_index = int(row["pair_index"])
        observed = np.asarray(native["values"], float)[:, pair_index, :]
        times = np.asarray(native["time_s"], float)
        result = weak_hb_profiles(observed, times=times, return_details=True, **evaluator.metric_kwargs(config))
        nominal = float(result["nominal_D"][0])
        best = float(result["best_D"][0])
        if not np.isclose(nominal, float(row["nominal_D"]), rtol=2e-8, atol=1e-11) or not np.isclose(best, float(row["best_D"]), rtol=2e-8, atol=1e-11):
            raise ValueError(f"native replay does not reproduce retained D: {admitted['join_key']}")
        arrays = dict(observed=observed, time_s=times, taus=result["taus"], profiles=result["profiles"][0],
                      kernel_positions=result["kernel_positions"], denominator=result["denominator"],
                      nominal_D=result["nominal_D"], best_D=result["best_D"], best_tau=result["best_tau"])
        for prefix in ("nominal", "best"):
            details = result[prefix + "_details"]
            for key in ("left", "right", "raw_right", "residual", "fitted_nuisance", "nuisance_coefficients"):
                arrays[prefix + "_" + key] = np.asarray(details[key])[:, 0]
            recalculated = float(np.sum(arrays[prefix + "_residual"]**2) / float(result["denominator"][0]))
            if not np.isclose(recalculated, float(result[prefix + "_D"][0]), rtol=1e-10, atol=1e-13):
                raise ValueError("explanatory closure residual does not reproduce D")
        filename = f"{token}_w{int(row['window_id'])}_p{pair_index}.npz"
        destination = example_dir / filename
        if destination.exists():
            with np.load(destination, allow_pickle=False) as retained:
                if set(retained.files) != set(arrays) or any(not np.array_equal(retained[key], value, equal_nan=True) for key, value in arrays.items()):
                    raise ValueError(f"retained explanatory arrays differ; use a versioned export: {destination}")
        else:
            temporary = destination.with_name(destination.name + ".partial.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(destination)
        item = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()}
        item.update(arrays=filename, measurement_provenance=native["provenance"],
                    channel_names=list(native["channel_names"])[2*pair_index:2*pair_index+2],
                    replay_nominal_D=nominal, replay_best_D=best, score_reproduced=True)
        selected.append(item)
    evaluator.write_json(selection_path, dict(schema=SCHEMA, purpose="explanation only; no parameter or threshold tuning",
        selection="90s fixed nominal D, nearest q=.1/.5/.9 within dataset; distinct subjects where possible",
        examples=selected))
    return selected


def hb_example_blocks(run, examples, directory, *, cached=False):
    plt = configure_plotting()
    units_text = '缓存输出单位' if cached else '发布相对单位'
    blocks = [paragraph(f"以下每库三例按全库 90 秒有效配对窗的固定参数 D，分别选择最接近第 10、50、90 百分位的窗口，尽量来自不同被试。选择仅用于解释，未使用最优参数误差挑例，也未把样例用于调参。每例的记录身份和精确数值数组保留在 examples/。Hb 纵轴保留该记录的{units_text}；闭合纵轴是中心化后的弱积分量，两者不可直接比较幅值。")]
    selection_rows = []
    for index, item in enumerate(examples):
        source = Path(run) / "examples" / item["arrays"]
        with np.load(source, allow_pickle=False) as loaded:
            a = {k: loaded[k] for k in loaded.files}
        t = a["time_s"] - a["time_s"][0]
        positions = a["kernel_positions"]
        fig = plt.figure(figsize=(7.4, 7.0), layout="constrained")
        grid = fig.add_gridspec(3, 2, height_ratios=(1.05, 1, 1))
        top = fig.add_subplot(grid[0, :])
        top.plot(t, a["observed"][:, 0], color=COLORS["HbO"], label="带通缓存 HbO" if cached else "原生 HbO")
        top.plot(t, a["observed"][:, 1], color=COLORS["HbR"], label="带通缓存 HbR" if cached else "原生 HbR")
        top.set(xlabel="窗内时间 / 秒", ylabel=units_text, title=f"{label(item['dataset_id'])}｜{item['subject_id']}｜{item['record_id']}｜{item['pair']}")
        top.legend(ncol=2)
        for col, width in enumerate(sorted(np.unique(positions[:, 1]))):
            ax = fig.add_subplot(grid[1, col])
            mask = np.isclose(positions[:, 1], width)
            centers = positions[mask, 0] - a["time_s"][0]
            ax.plot(centers, a["nominal_left"][mask], color=COLORS["HbR"], label="闭合左侧")
            ax.plot(centers, a["nominal_right"][mask], color=COLORS["best"], label="右侧＋常数/初态")
            ax.set(title=f"固定 τ=2秒；核半宽{number(width)}秒", xlabel="核中心的窗内时间 / 秒", ylabel="弱积分闭合量")
            ax.legend(fontsize=8.5)
        residual = fig.add_subplot(grid[2, 0])
        for width in sorted(np.unique(positions[:, 1])):
            mask = np.isclose(positions[:, 1], width)
            residual.plot(positions[mask, 0]-a["time_s"][0], a["nominal_residual"][mask], label=f"固定/{number(width)}秒核")
            residual.plot(positions[mask, 0]-a["time_s"][0], a["best_residual"][mask], "--", alpha=.8, label=f"最优/{number(width)}秒核")
        residual.axhline(0, color=COLORS["gray"], lw=.7)
        residual.set(xlabel="核中心的窗内时间 / 秒", ylabel="左侧-拟合右侧", title="局部残差（两种核分别连线）")
        residual.legend(fontsize=8.5, ncol=2)
        profile = fig.add_subplot(grid[2, 1])
        profile.plot(a["taus"], a["profiles"], color=COLORS["residual"])
        profile.axvline(2, color=COLORS["nominal"], label="固定 τ=2秒")
        profile.axvline(float(a["best_tau"][0]), color=COLORS["best"], ls="--", label="该窗网格最优")
        profile.set(xlabel="τ / 秒", ylabel="固定分母闭合 D", title="同一窗口参数 profile")
        profile.set_ylim(bottom=0)
        profile.legend(fontsize=8.5)
        path = save_figure(fig, directory, f"hb_example_{index+1:02d}")
        nominal, best = float(a["nominal_D"][0]), float(a["best_D"][0])
        caption = (f"固定评分 q={item['selection_quantile']:.1f} 附近的解释样例，候选分母 {item['selection_denominator']:,} 个配对窗。原生支持 [{number(item['start_s'])}, {number(item['stop_s'])}) 秒；第 {int(item['pair_index'])+1} 个 Hb 对；{int(item['samples'])} 个时间点。"
                   f"固定 D={number(nominal)}，网格最优 D={number(best)}、τ={number(a['best_tau'][0])} 秒，绝对差={number(nominal-best)}。"
                   "闭合两侧与局部残差均来自该精确窗重放；残差平方和/固定动态能量分母已复现保留 D。观测幅值、局部偏差和 profile 平坦/触边应一起判断；此例不代表独立被试统计或生成器真值。")
        worst = int(np.argmax(abs(a["nominal_residual"])))
        local_time = float(positions[worst, 0]-a["time_s"][0])
        local_change = a["nominal_residual"]**2-a["best_residual"]**2
        improved, worsened = int((local_change > 0).sum()), int((local_change < 0).sum())
        best_tau = float(a["best_tau"][0])
        where = "下界" if best_tau == float(a["taus"].min()) else "上界" if best_tau == float(a["taus"].max()) else "内部节点"
        caption += (f" 本例HbO/HbR峰峰幅分别为 {number(np.ptp(a['observed'][:, 0]))}/{number(np.ptp(a['observed'][:, 1]))} 个{units_text}。"
                    f"固定参数最大绝对弱残差位于窗内 {number(local_time)} 秒、半宽 {number(positions[worst, 1])} 秒的核，左-右={number(a['nominal_residual'][worst])}。"
                    f"选择τ后 {improved}/{len(local_change)} 个弱核的残差平方下降、{worsened} 个上升；核互相重叠，计数只描述局部变化。最佳τ位于网格{where}，右下profile采用线性D轴。")
        blocks.append(figure(path, caption, str(source)))
        selection_rows.append([label(item["dataset_id"]), item["subject_id"], item["record_id"], item["pair"],
                               f"[{number(item['start_s'])},{number(item['stop_s'])})", item["selection_quantile"], number(nominal), number(best)])
    blocks.append(table(["数据集", "被试", "记录", "Hb对", "原生秒支持", "选择分位", "固定 D", "最优 D"], selection_rows,
                        caption="完整 join key、源文件、原生通道名、采样点边界、选择目标分数与数值核验见 examples/hb_selection.json。", source=str(Path(run) / "examples" / "hb_selection.json")))
    return blocks


def read_panels(run):
    directory = Path(run) / "panels"
    plan, summary = read_json(directory / "plan.json"), read_json(directory / "summary.json")
    if summary.get("execution") != "completed":
        raise ValueError("full measured report requires every planned panel task to be terminal")
    result = {name: _read_csv(directory / f"{name}.csv") for name in
              ("linear_profile", "linear_summary", "compatibility_sets", "nonlinear", "space")}
    if len(result["linear_summary"]) != int(summary["planned_linear"]) or len(result["nonlinear"]) != int(summary["planned_nonlinear"]) or len(result["space"]) != int(summary["planned_space"]):
        raise ValueError("panel table denominator differs from the owning terminal summary")
    # Terminal failures may have no numerical columns. Keep them in counts and
    # show unavailable scores instead of requiring a successful subset to exist.
    for name, columns in {
        "linear_summary": ("low_signal", "nominal_joint_nmse", "best_joint_joint_nmse", "best_joint_hb_nmse", "best_joint_joint_hb_nmse", "best_joint_cross_increment_nmse"),
        "compatibility_sets": ("dataset", "length_s", "tolerance", "id", "n_nodes", "tau_width", "kappa_width", "hit_boundary", "all_nodes_on_boundary"),
        "nonlinear": ("legal_witness", "optimizer_converged", "D_Hb_NL_recon_nmse", "integration_difference"),
        "space": ("K", "D_rank_joint", "D_space_fraction_raw", "weighted_nmse", "weighted_lower_bound_nmse"),
    }.items():
        for column in columns:
            if column not in result[name]:
                result[name][column] = np.nan
    migration_path = Path(run) / "derived_spatial_stability_v1" / "migration.json"
    if migration_path.is_file():
        migration = read_json(migration_path)
        verification = read_json(migration_path.parent / "verification.json")
        if (migration.get("schema") != "model_compatibility_spatial_stability_migration_v1"
                or migration.get("status") != "completed" or verification.get("status") != "passed"):
            raise ValueError("spatial stability migration is not complete")
        result["space"] = apply_spatial_followup(result["space"], _read_csv(migration_path.parent / "space_stable.csv"))
        result["space_numerical_followup"] = migration_path
    result.update(plan=plan, summary=summary, directory=directory)
    return result


def apply_spatial_followup(space, followup):
    """Use a versioned derived projection without rewriting retained task rows."""
    rows = followup.to_dict("records")
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(space.id):
        raise ValueError("spatial followup task denominator differs from retained space table")
    result = space.copy()
    result["retained_D_space_fraction_raw"] = result.D_space_fraction_raw
    for i, original in result.iterrows():
        row = indexed[original.id]
        if row["record_id"] != original.record_id or row["arm"] != original.arm:
            raise ValueError("spatial numerical followup identity differs")
        stable = float(row["D_space_fraction_raw"])
        if not np.isfinite(stable) or not 0 <= stable <= 1+1e-10:
            raise ValueError("invalid stable spatial projection fraction")
        if not np.isclose(float(row["legacy_D_space_fraction_raw"]), float(original.D_space_fraction_raw), rtol=1e-12, atol=1e-14):
            raise ValueError("spatial followup does not refer to the retained raw score")
        result.loc[i, "D_space_fraction_raw"] = stable
        for name in ("HbO", "HbR"):
            result.loc[i, name+"_D_space_fraction_raw"] = row[name+"_D_space_fraction_raw"]
    return result


def _bool_series(series):
    return series.astype(str).str.lower().isin(("true", "1"))


def panel_distribution_blocks(panels, directory):
    plt = configure_plotting()
    linear, feasible = panels["linear_summary"], panels["compatibility_sets"]
    source = str(panels["directory"])
    good = linear.loc[linear.status == "completed"]
    rows = []
    for dataset in DATASETS:
        for length in (30, 90):
            part = linear.loc[(linear.dataset == dataset) & (linear.length_s == length)]
            complete = part.loc[part.status == "completed"]
            rows.append([label(dataset), length, len(part), len(complete), len(part)-len(complete),
                         int(_bool_series(complete.low_signal).sum()),
                         number(complete.nominal_joint_nmse.median()), number(complete.best_joint_joint_nmse.median()),
                         number(complete.best_joint_hb_nmse.median()), number(complete.best_joint_cross_increment_nmse.median())])
    blocks = [paragraph("四库冻结面板来自已有 output-support 公开评价身份：每库 12 条 90 秒记录、6 名被试，总计 48 条记录、24 个库内被试身份。30 秒支持取原 90 秒记录的中部；90 秒支持使用完整记录，共 96 个记录×支持任务。它们是已选定且拥有冻结读出的面板，不能代表全库存分布。原父 run 的滤波/参考算子被显式裁剪并保留，不能把面板分数与原生普查 D 直接相减。"),
              table(["数据集", "秒", "计划", "完成", "失败", "低幅标记", "固定 joint NMSE", "最优 joint NMSE", "同θ Hb-only NMSE", "同θ cross增量 NMSE"], rows,
                    caption="各误差格为该组完整任务的原始中位数。低幅标记：任一观测分量的中心化 RMS 小于 0.1 个冻结训练 SD；它是工程幅度标记，不是估计噪声。失败不以零值补齐。最优 θ 仅按同窗 joint NMSE 选择，后两列都在这个同一节点计算。", source=source + "/linear_summary.csv",
                    widths=[1.1, .4, .5, .5, .5, .7, 1, 1, 1.05, 1.1])]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.1), layout="constrained")
    for ax, dataset in zip(axes.flat, DATASETS):
        part = good.loc[good.dataset == dataset]
        for length, marker in ((30, "o"), (90, "^")):
            subset = part.loc[part.length_s == length]
            ax.scatter(subset.best_joint_hb_nmse, subset.best_joint_joint_hb_nmse, marker=marker,
                       s=26, alpha=.75, label=f"{length}秒，n={len(subset)}")
        limit = max(float(part.best_joint_hb_nmse.max()), float(part.best_joint_joint_hb_nmse.max()), 1e-5) if len(part) else 1.
        ax.plot([0, limit], [0, limit], color=COLORS["gray"], ls="--", lw=.8)
        score_axis(ax); score_axis(ax, axis="y")
        ax.set(title=label(dataset), xlabel="同一最优joint θ 的 Hb-only NMSE", ylabel="同一 θ 的 joint Hb分量 NMSE")
        ax.legend()
    blocks.append(figure(save_figure(fig, directory, "panel_hb_joint_tradeoff"), "每个点是一个记录×支持任务；同一被试和同一记录的点有依赖。横轴仅拟合 Hb，纵轴在同一 θ 下同时拟合 EEG 与 Hb。点在对角线上方说明兼容 EEG 后 Hb 拟合有折中；这不是独立因果效应或参数估计偏差。两个坐标均为训练 SD 标准化平方误差，零值保留。", source + "/linear_summary.csv"))
    set_rows = []
    for dataset in DATASETS:
        for length in (30, 90):
            part = feasible.loc[(feasible.dataset == dataset) & (feasible.length_s == length)]
            for tolerance in sorted(part.tolerance.unique()):
                subset = part.loc[part.tolerance == tolerance]
                nonempty = subset.loc[subset.n_nodes > 0]
                set_rows.append([label(dataset), length, number(tolerance), len(subset),
                                 int((subset.n_nodes == 0).sum()), number(subset.n_nodes.median()),
                                 number(nonempty.tau_width.median()), number(nonempty.kappa_width.median()),
                                 int(_bool_series(nonempty.hit_boundary).sum()), int(_bool_series(nonempty.all_nodes_on_boundary).sum())])
    blocks.append(table(["数据集", "秒", "阈值", "可评任务", "空集合", "节点中位数", "非空τ宽/秒", "非空κ宽", "非空触边", "仅边界"], set_rows,
                        caption="每个集合都在相同 τ、κ 节点、相同 joint 系数解上同时要求 EEG/HbO/HbR 三个 NRMSE 不超过阈值。τ/κ 宽度是离散节点投影的最大值减最小值，可能跨越不连续空洞，不是置信区间。空集合保留；宽度和触边只在非空集合中定义，不能用这个成功子集掩盖空集合。线性任务失败时共同集不可评，完整失败分母见前表。", source=source + "/compatibility_sets.csv"))
    return blocks


def nonlinear_blocks(panels, directory):
    plt = configure_plotting()
    linear, nonlinear = panels["linear_summary"], panels["nonlinear"]
    paired = nonlinear.merge(linear[["id", "best_joint_hb_nmse", "best_joint_joint_hb_nmse"]], on="id", how="left", validate="one_to_one")
    source = str(panels["directory"])
    legal_count = int(_bool_series(nonlinear.legal_witness).sum())
    converged_count = int(_bool_series(nonlinear.optimizer_converged).sum())
    comparable = paired.loc[_bool_series(paired.legal_witness)
                            & np.isfinite(paired.D_Hb_NL_recon_nmse) & np.isfinite(paired.best_joint_hb_nmse)]
    paired_difference = comparable.best_joint_hb_nmse - comparable.D_Hb_NL_recon_nmse
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8), layout="constrained")
    for ax, dataset in zip(axes.flat, DATASETS):
        part = paired.loc[paired.dataset == dataset]
        legal = part.loc[_bool_series(part.legal_witness) & np.isfinite(part.D_Hb_NL_recon_nmse)]
        for length, marker in ((30, "o"), (90, "^")):
            sub = legal.loc[legal.length_s == length]
            ax.scatter(sub.best_joint_hb_nmse, sub.D_Hb_NL_recon_nmse, marker=marker, s=25, alpha=.8, label=f"{length}秒，合法n={len(sub)}")
        limit = max(float(legal.best_joint_hb_nmse.max()) if len(legal) else 0,
                    float(legal.D_Hb_NL_recon_nmse.max()) if len(legal) else 0, 1e-5)
        ax.plot([0, limit], [0, limit], "--", color=COLORS["gray"], lw=.8)
        score_axis(ax); score_axis(ax, axis="y")
        ax.set(title=label(dataset), xlabel="同θ线性 Hb-only NMSE", ylabel="条件非线性 Hb重建 NMSE")
        ax.legend(fontsize=8.5)
        for length in (30, 90):
            sub = part.loc[part.length_s == length]
            ok = sub.loc[_bool_series(sub.legal_witness) & np.isfinite(sub.D_Hb_NL_recon_nmse)]
            difference = ok.best_joint_hb_nmse - ok.D_Hb_NL_recon_nmse
            rows.append([label(dataset), length, len(sub), len(ok), len(sub)-len(ok),
                         int(_bool_series(sub.optimizer_converged).sum()), number(ok.best_joint_hb_nmse.median()),
                         number(ok.D_Hb_NL_recon_nmse.median()), number(difference.median()),
                         int((difference > 0).sum()), number(ok.integration_difference.max())])
    blocks = [paragraph(f"本次 {len(nonlinear)} 个非线性任务中，{legal_count} 个保留合法轨迹，优化收敛 {converged_count} 个、未收敛 {len(nonlinear)-converged_count} 个。未收敛任务的误差仅为本次预算内找到的合法残差上界，不能作为最优解；较高误差也不能据此排除非线性模型。合法轨迹与优化收敛是两个独立状态，以下全部比较均保留此限制。"),
              paragraph(f"与同θ线性Hb-only逐任务配对后，条件NL误差较低 {int((paired_difference > 0).sum())}/{len(comparable)}，较高 {int((paired_difference < 0).sum())}/{len(comparable)}，完全相同 {int((paired_difference == 0).sum())}/{len(comparable)}。这反映有限预算下不同函数族找到的误差，不能判定线性或非线性模型整体优劣。"),
              paragraph("非线性阶段对全部 96 个记录×支持任务，在各自线性 joint 最小 NMSE 所选 θ 上寻找合法 Hb 前向轨迹。每个保留模型时间点有一个自由 log-flow 系数；流量与初态受既有数值边界约束。这个放松不要求上游 EEG 驱动动力学，κ 不作用于此血管子系统，因此它不能构成 EEG–Hb 联合见证。"),
              figure(save_figure(fig, directory, "nonlinear_linear_comparison"), "横纵坐标使用同一 Hb 目标、mask、训练 SD 和同一 θ，但允许的轨迹族不同：横轴是线性 Hb-only 投影；纵轴是条件非线性自由流量前向拟合。低于对角线表示已找到的非线性残差更小，不代表非线性全局最小值或参数真值。图只画合法见证，完整失败分母在下表。", source + "/nonlinear.csv"),
              table(["数据集", "秒", "计划", "合法轨迹", "无合法/失败", "优化收敛", "线性中位", "NL中位", "配对差中位", "NL较小", "最大积分差"], rows,
                    caption="误差列均为 NMSE；配对差=每任务线性 Hb-only NMSE-非线性 NMSE，再取中位，不能由两组中位数相减替代。优化收敛与轨迹合法分别计数；未收敛但合法的轨迹仍仅提供误差上界。积分差按保留 4/8 子步核验，单位为训练 SD。无合法轨迹不补零。", source=source + "/nonlinear.csv",
                    widths=[1, .35, .45, .65, .8, .65, .75, .7, .8, .55, .8])]
    return blocks, paired


def spatial_blocks(panels, directory):
    plt = configure_plotting()
    space = panels["space"]
    good = space.loc[space.status == "completed"]
    source = str(panels["directory"] / "space.csv")
    followup_note = ""
    if "space_numerical_followup" in panels:
        followup_note = (f" 原raw拟合的latent回放受时间算子病态性影响；本图rawD采用冻结loading的稳健QR空间投影，允许自由时间曲线并放松时间约束，"
                         f"相对原值最大绝对差为{number(abs(good.D_space_fraction_raw-good.retained_D_space_fraction_raw).max())}，rank下界与weighted量不改。")
        source += "; rawD数值复核：" + str(panels["space_numerical_followup"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), layout="constrained")
    rows = []
    for k, color in ((1, COLORS["HbO"]), (2, COLORS["HbR"])):
        part = good.loc[good.K == k]
        axes[0].scatter(part.D_rank_joint, part.D_space_fraction_raw, s=25, color=color, alpha=.7, label=f"K={k}，n={len(part)}窗")
        x, p = ecdf(part.weighted_nmse)
        axes[1].step(x, p, where="post", color=color, label=f"固定读出 K={k}")
        rows.append([k, 36, len(part), 36-len(part), number(part.D_space_fraction_raw.median()),
                     number(part.D_rank_joint.median()), number(part.weighted_nmse.median()),
                     number(part.weighted_lower_bound_nmse.median())])
    axes[0].plot([0, 1], [0, 1], "--", color=COLORS["gray"], lw=.8)
    axes[0].set(xlabel="同窗自由共同 rank-K 原始能量残差比", ylabel="原冻结空间读出残差能量比", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="固定读出的训练SD标准化 NMSE", ylabel="窗口累计比例", ylim=(0, 1.02))
    score_axis(axes[1])
    for ax in axes:
        ax.legend(fontsize=8.5)
    return [paragraph("空间量只覆盖 Single-Trial 原冻结 K1/K2 面板的 36 个评价窗、6 名被试、4 个局部 Hb 对。固定 loading 来自原训练校准，不在当前窗重新学习。rawD允许各色团自由时间曲线并放松时间与血管动力学约束；评价支持保留原mask。weighted误差与其下界沿用原实际算子。其他三库没有这个冻结空间映射，空间结论为 N/A。"),
            figure(save_figure(fig, directory, "spatial_fixed_rank"), "左图每点为一个窗口×K。横轴是将 HbO/HbR 上下堆叠后在当前窗自由学习共同 rank-K 子空间的最优残差比；纵轴是固定loading下的稳健空间投影残差。横轴更自由，不能用来替换纵轴。两种比值的分母为原始观测能量，不是生理成分占比；右图另示训练 SD 标准化 NMSE，两套尺度不混合。" + followup_note, source),
            table(["K", "计划窗", "完成", "失败", "固定读出中位比", "自由rank中位比", "标准化NMSE中位", "放松下界NMSE中位"], rows,
                  caption="72 个 arm×窗口来自同一组 36 窗而非 72 个独立样本。原始能量比、训练 SD 误差和放松下界分别保留其定义；下界不是已实现动力学重建的误差。", source=source)]


def select_panel_examples(panels, paired):
    """Outcome-selected explanations, explicitly separate from primary estimates."""
    linear = panels["linear_summary"]
    chosen = {}
    good = linear.loc[(linear.status == "completed") & (linear.length_s == 90)]
    for dataset in DATASETS:
        part = good.loc[good.dataset == dataset]
        if len(part):
            row = part.sort_values(["best_joint_joint_nmse", "id"], ascending=[False, True]).iloc[0]
            chosen[str(row.id)] = "本库90秒面板中最优joint NMSE最大，用于解释失配"
    sets = panels["compatibility_sets"]
    empty = sets.loc[(sets.tolerance == .5) & (sets.n_nodes == 0)]
    candidates = good.loc[good.id.isin(empty.id)]
    if len(candidates):
        row = candidates.sort_values(["best_joint_joint_nmse", "id"], ascending=[False, True]).iloc[0]
        chosen[str(row.id)] = chosen.get(str(row.id), "") + "；0.5阈值未发现共同节点"
    legal = paired.loc[_bool_series(paired.legal_witness) & np.isfinite(paired.D_Hb_NL_recon_nmse)].copy()
    if len(legal):
        legal["absolute_improvement"] = legal.best_joint_hb_nmse - legal.D_Hb_NL_recon_nmse
        row = legal.sort_values(["absolute_improvement", "id"], ascending=[False, True]).iloc[0]
        chosen[str(row.id)] = chosen.get(str(row.id), "") + "；合法NL相对同θ线性Hb-only的绝对NMSE改善最大"
    return chosen


def audit_prediction(target, prediction, sd, mask, expected, name):
    """Verify displayed retained arrays use the exact scored support and scale."""
    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    scale, selected = np.asarray(sd, float), np.asarray(mask)
    if target.shape != prediction.shape or selected.shape != target.shape or selected.dtype != bool:
        raise ValueError(f"{name} prediction or score mask shape differs")
    if scale.shape != (target.shape[-1],) or np.any(scale <= 0) or not np.isfinite(scale).all() or not selected.any():
        raise ValueError(f"{name} scale or scoring support is invalid")
    residual = (target-prediction)/scale
    actual = float(np.mean(residual[selected]**2))
    if not np.isfinite(actual) or not np.isclose(actual, float(expected), rtol=1e-7, atol=1e-10):
        raise ValueError(f"{name} prediction does not reproduce its retained NMSE")
    return residual


def peak_local_rms(time, residual, mask, *, width_s=5.):
    """Largest fixed five-second segment RMS, using only original scored points."""
    time, error, selected = np.asarray(time, float), np.asarray(residual, float), np.asarray(mask)
    if time.ndim != 1 or error.shape != time.shape or selected.shape != time.shape or selected.dtype != bool:
        raise ValueError("local RMS requires matching time, residual and boolean score mask")
    if len(time) < 2 or np.any(np.diff(time) <= 0) or width_s <= 0:
        raise ValueError("local RMS requires increasing times and positive width")
    candidates = []
    stop = time[-1]+float(np.median(np.diff(time)))
    for start in np.arange(time[0], stop-1e-9, width_s):
        keep = selected & (time >= start) & (time < start+width_s)
        if keep.any():
            values = error[keep]
            if not np.isfinite(values).all():
                raise ValueError("local RMS encountered a nonfinite scored value")
            candidates.append(dict(start_s=float(start), stop_s=float(min(start+width_s, stop)),
                                   rms=float(np.sqrt(np.mean(values**2))), count=int(keep.sum())))
    return max(candidates, key=lambda row: row["rms"]) if candidates else None


def panel_example_blocks(panels, paired, directory):
    plt = configure_plotting()
    folder = panels["directory"]
    choices = select_panel_examples(panels, paired)
    blocks = [paragraph("本节按明示结果规则选取失配、空共同集与非线性改善样例，以解释总体结果；这些挑例不构成额外独立评价，也没有用来改变参数网格或阈值。每库先选 90 秒面板中最优 joint NMSE 最大的一例，再补充 0.5 阈值空共同集和非线性绝对改善最大的例子；同一身份只展示一次。")]
    for index, (task_id, reason) in enumerate(choices.items()):
        record = read_json(folder / "tasks" / "linear" / (task_id + ".json"))
        with np.load(folder / "tasks" / "linear" / (task_id + ".npz"), allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        target, sd, mask = a["target"], a["sd"], a["score_mask"]
        prediction = a["best_joint_joint_prediction"]
        hb_only = a.get("best_joint_hb_prediction")
        nominal = a["nominal_joint_prediction"]
        nl_path = folder / "tasks" / "nonlinear" / (task_id + ".json")
        nl_meta = read_json(nl_path)
        nl_prediction = None
        if nl_meta.get("legal_witness"):
            with np.load(nl_path.with_suffix(".npz"), allow_pickle=False) as z:
                if "prediction" in z:
                    nl_prediction = z["prediction"].copy()
        t = np.arange(len(target))*.25
        offset = 30. if record["length_s"] == 30 else 0.
        t += offset
        # Recompute engineering NMSE on retained mask before drawing a curve.
        residual = audit_prediction(target, prediction, sd, mask, record["best_joint"]["joint_nmse"], "linear panel")
        if nl_prediction is not None:
            audit_prediction(target[:, 1:], nl_prediction, sd[1:], mask[:, 1:], nl_meta["D_Hb_NL_recon_nmse"], "nonlinear panel")
        fig, axes = plt.subplots(3, 2, figsize=(7.4, 7.1), sharex=True, layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0, 1, .91))
        for j, name in enumerate(("EEG", "HbO", "HbR")):
            ax, er = axes[j]
            scaled = target[:, j]/sd[j]
            scored = lambda values: np.where(mask[:, j], values, np.nan)
            ax.plot(t, scored(scaled), color="#242d36", lw=1.8, label="观测目标（左列）")
            ax.plot(t, scored(nominal[:, j]/sd[j]), color=COLORS["gray"], ls=":", label="固定θ joint")
            ax.plot(t, scored(prediction[:, j]/sd[j]), color=COLORS["best"], label="最优θ joint")
            er.plot(t, scored((target[:, j]-nominal[:, j])/sd[j]), color=COLORS["gray"], ls=":", label="目标-固定θjoint")
            er.plot(t, scored(residual[:, j]), color=COLORS["best"], label="目标-joint")
            if j and hb_only is not None:
                ax.plot(t, scored(hb_only[:, j]/sd[j]), color="#bd8c22", ls="--", label="同θ Hb-only")
                er.plot(t, scored((target[:, j]-hb_only[:, j])/sd[j]), color="#bd8c22", ls="--", label="目标-Hb-only")
            if j and nl_prediction is not None:
                ax.plot(t, scored(nl_prediction[:, j-1]/sd[j]), color=COLORS["residual"], ls="-.", label="同θ条件NL")
                er.plot(t, scored((target[:, j]-nl_prediction[:, j-1])/sd[j]), color=COLORS["residual"], ls="-.", label="目标-条件NL")
            er.axhline(0, color=COLORS["gray"], lw=.7)
            ax.set(title=name + "：目标与完整预测", ylabel="原训练 SD 单位")
            er.set(title=name + "：逐时点残差", ylabel="原训练 SD 单位")
            for item in (ax, er):
                for time in t[~mask[:, j]]:
                    item.axvspan(time, time+.25, color="#eeeeee", zorder=-1)
        handles, legend_labels = axes[1, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .995), ncol=3, fontsize=8.5)
        for ax in axes[-1]:
            ax.set_xlabel("相对原90秒面板起点的时间 / 秒")
        best = record["best_joint"]
        caption = (f"{label(record['dataset'])}｜{record['subject']}｜{record['record_id']}｜{record['length_s']}秒。选择理由：{reason.strip('；')}。"
                   f"固定θ joint NMSE={number(record['nominal']['joint_nmse'])}；最优θ=(τ={number(best['tau'])}秒，κ={number(best['kappa'])})，joint NMSE={number(best['joint_nmse'])}。"
                   f"该θ EEG/HbO/HbR NRMSE={number(np.sqrt(best['joint_eeg_nmse']))}/{number(np.sqrt(best['joint_hbo_nmse']))}/{number(np.sqrt(best['joint_hbr_nmse']))}。"
                   "所有目标与预测经过相同父观测算子/参考；训练尺度固定。灰底为不评分末点，主图不展示该点的目标与预测，完整数组仍保留，未评分尾值不决定主轴。完整预测包含允许初态与常数项，残差是目标减预测。")
        local_descriptions = []
        for j, name in enumerate(("EEG", "HbO", "HbR")):
            peak = peak_local_rms(t, residual[:, j], mask[:, j])
            observed = target[mask[:, j], j]/sd[j]
            local_descriptions.append(f"{name}评分观测峰峰幅{number(np.ptp(observed))}个训练SD；joint最大5秒段RMS在[{number(peak['start_s'])},{number(peak['stop_s'])})秒，为{number(peak['rms'])}")
        caption += " 本例逐模态读图：" + "；".join(local_descriptions) + "。局部RMS使用原评分mask，在相对原90秒起点的5秒非重叠段内计算，不是任务条件标签。"
        if nl_prediction is not None:
            difference = float(best["hb_nmse"])-float(nl_meta["D_Hb_NL_recon_nmse"])
            comparison = "较小" if difference > 0 else "较大" if difference < 0 else "相同"
            caption += (f" 条件NL Hb NMSE={number(nl_meta['D_Hb_NL_recon_nmse'])}，同θ线性Hb-only NMSE={number(best['hb_nmse'])}；"
                        f"本例NL误差{comparison}，线性-NL绝对差={number(difference)}。")
            nl_residual = (target[:, 1:]-nl_prediction)/sd[1:]
            for j, name in enumerate(("HbO", "HbR")):
                peak = peak_local_rms(t, nl_residual[:, j], mask[:, j+1])
                caption += f" NL的{name}最大局部RMS在[{number(peak['start_s'])},{number(peak['stop_s'])})秒，为{number(peak['rms'])}。"
            caption += " NL未收敛误差只作已找到上界，不验证上游EEG驱动约束。"
        else:
            caption += " 条件NL无保留合法见证，未画成功预测；具体失败见任务JSON。"
        blocks.append(figure(save_figure(fig, directory, f"panel_waveform_{index+1:02d}"), caption,
                             str(folder / "tasks" / "linear" / (task_id + ".json"))))
        profiles = pd.DataFrame(record["profiles"])
        component_max = np.sqrt(profiles[["joint_eeg_nmse", "joint_hbo_nmse", "joint_hbr_nmse"]].max(axis=1))
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0, 1, .85))
        chart = axes[0].scatter(profiles.tau, profiles.kappa, c=component_max, marker="s", s=75,
                                cmap="viridis", vmin=0)
        fig.colorbar(chart, ax=axes[0], label="最大分量 NRMSE（训练 SD单位）")
        axes[1].scatter(profiles.tau, profiles.kappa, c=np.where(component_max <= .5, "#187969", "#dfdfe3"), marker="s", s=75)
        for ax in axes:
            ax.scatter([2], [.64], marker="x", color=COLORS["nominal"], s=65, label="固定θ")
            ax.scatter([best["tau"]], [best["kappa"]], marker="*", color="#bd8c22", edgecolors="black", s=110, label="最优joint θ")
            ax.set(xlabel="τ / 秒", ylabel="κ / 1/秒", xlim=(.2, 8.3), ylim=(.1, 1.65))
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .995), ncol=2, fontsize=8.5)
        axes[0].set_title("每个节点使用同一joint系数解")
        subset = next(s for s in record["compatibility_sets"] if s["tolerance"] == .5)
        axes[1].set_title(f"阈值0.5：绿=共同节点，n={subset['n_nodes']}/80")
        blocks.append(figure(save_figure(fig, directory, f"panel_nodes_{index+1:02d}"),
                             "每个方块对应实际求解的离散 τ/κ 节点，没有插值。左图颜色是该节点 EEG/HbO/HbR 三个标准化 RMS 的最大值；右图要求三者同时≤0.5。灰色是未满足节点，绿色是共同节点。星号按最小 joint NMSE 选取，未必位于共同集合。投影跨度与触边不能当作置信区间；没有绿色节点不证明连续参数域全局无解。", str(nl_path.parent.parent / "linear" / (task_id + ".json"))))
    return blocks, choices


def methods_section():
    return {"title": "这些量如何计算，以及它们能回答什么", "blocks": [
        paragraph("这份报告使用实测保留结果。此前教学图中存在生成器状态和无噪声已知真值；本报告的 HbO/HbR 是数据集发布或既有原生读出得到的相对观测，EEG 是冻结模型所用的观测特征。两者的数值大小和可识别含义不能直接互换。所有当前窗拟合都使用完整可见窗口，属于离线相容性检查。"),
        equation(r"R=\mathrm{HbR},\quad H=\mathrm{HbO}+\mathrm{HbR},\qquad \bar R=R-\langle R\rangle_t,\quad \bar H=H-\langle H\rangle_t", "每个物理 Hb 对共享其原始相对幅度。去除的是各量的窗口时间均值；没有分别把 HbO/HbR 除以自身标准差。H 与 R 的单位来自原生读出。"),
        equation(r"\ell_j(\tau)=\int\phi_j\bar R\,dt-\tau\int\phi'_j\bar R\,dt,\qquad r_j(\tau)=\rho\!\left[\zeta\!\int\phi_j\bar H\,dt-\xi\tau\!\int\phi'_j\bar H\,dt\right]", "φ 是支撑半宽为 4 或 8 秒的平滑紧支撑核；中心间距 2 秒。导数作用于测试函数而非实测 Hb。ρ、ξ、ζ 来自当前 Balloon 核心固定参数；τ 以秒计。同一窗口内核互相重叠，不能当作独立样本。"),
        equation(r"D_{\mathrm{Hb}}(\tau)=\frac{\min_{a,b}\sum_j[\ell_j-r_j-a\!\int\phi_jdt-b\!\int\phi_je^{-(t-t_0)/\tau}dt]^2}{\sum_j[(\int\phi_j\bar Rdt)^2+\rho^2(\int\phi_j\bar Hdt)^2]}", "分子为独立 profile 的未知常数零点与合法指数初态项之后的残差能量。分母是整个窗口固定的中心化动态能量，在 τ 扫描中保持不变。这个动态闭合能量比可以大于1；它不是残差百分比或解释方差，不能读成超过100%的噪声。分母不是噪声方差，也没有 χ² 或显著性含义。固定参数 τ=2 秒；网格为 0.5 到 8 秒、间隔 0.25 秒。"),
        paragraph("跨模态实测面板采用离散前向设计的子空间投影与 joint-Hb-only 平方残差增量，以准确保留父实验的真实时间算子。这是离散观测空间的工程距离，与教学示例中的连续时间三阶弱闭合 D_EEG-Hb 分开命名；本报告没有将实测投影增量称为三阶弱闭合值。"),
        paragraph("具体而言，保留面板的EEG使用参考算子，Hb使用参考、原生分箱和线性插值的复合算子，裁窗后还排除末评分点。直接对两种已处理目标套用连续三阶弱式会混入时间算子差。在没有额外逆算子或插值假设时，原连续弱式在本面板记N/A；本轮报告与真实算子一致的离散前向距离，两者数值不等同。"),
        equation(r"\widehat y_\theta=L\!\left[c_\theta+G_\theta r+H_\theta z_0+(\mathbf1_T\otimes I_3)b\right],\qquad r\in\mathbb R^T,\ z_0\in\mathbb R^5,\ b\in\mathbb R^3", "θ=(τ,κ)。T 是该面板模型时间点数（间隔0.25秒）。r 为每步分段常数的共享驱动，z0 为五个自由线性初始坐标，b 为三个观测分量的未知常数；它们在完整当前窗重拟合。G、H 和静息偏置 c 由冻结模型生成，L 是原父实验保留的观测/参考算子。所有分量按时间交错排列，先经过 L 再选评分 mask。没有为血管状态增加逐点创新项。", source="src/inference/shared_driver_reconstruction.py:build_shared_driver_design; src/inference/model_compatibility.py:prepare_linear_panel_profiles"),
        equation(r"\mathrm{NMSE}_m=\frac1{N_m}\sum_{i\in\Omega_m}\left(\frac{y_i-\hat y_i}{s_m^{\rm train}}\right)^2,\qquad \mathrm{NRMSE}_m=\sqrt{\mathrm{NMSE}_m}", "冻结面板中的 s 是原训练分区 SD，Ω 是保留评分 mask，m 为 EEG、HbO 或 HbR。分母既不是当前窗 SD，也不是传感器噪声。每个参数节点重拟合原线性初态/允许驱动及经同一观测算子传播的常数项；训练坐标、增益、算子和训练尺度固定。"),
        paragraph("Hb-only 与 joint 在同一 θ=(τ,κ) 节点评分。joint 同时承担 Hb 与 EEG 约束；cross increment 是 joint 总标准化平方残差减去 Hb-only 平方残差，再除以 EEG 评分点数，包含为兼容 EEG 产生的 Hb 折中，不能解释为独立的 EEG 似然。共同集合要求三个分量 NRMSE 同时低于同一工程阈值。网格为空只表示没有发现符合条件的离散节点。阈值 0.25、0.5、1 不是统计拒绝线或生理范围。"),
        paragraph("条件非线性前向重建复用已有血管流量松弛拟合器，在选定同一 θ 下寻找 Hb 观测重建。它使用相同冻结读出、观测算子和评分尺度，是已找到的残差上界；既不是相对 Hb 的绝对状态反演，也不保证全局最优。优化状态、数值积分差异、失败和分母必须与误差一起阅读。空间量将固定空间读出距离与自由 rank-K 下界分别报告。")
    ]}


def build_report(run, output):
    """Build all communication assets from one explicit owning run."""
    from experiments.scripts.render_source_diagnostics_report import render_report

    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"report output must be fresh: {output}")
    plan = read_json(run / "plan.json")
    terminal = read_json(run / "summary.json")
    if terminal.get("execution") != "completed":
        raise ValueError("full measured report requires the completed owning census summary")
    retained_summary = _read_csv(run / "summary.csv")
    census, inventory, metas = read_census(run)
    if terminal.get('schema') == 'model_compatibility_census_cached_v2':
        examples = prepare_hb_examples(run, census, inventory)
        with tempfile.TemporaryDirectory(prefix='cached_Hb_report_') as staging:
            staging = Path(staging)
            distribution, _ = census_distribution_blocks(census, staging, str(run / 'summary.csv'), retained_summary)
            # Reuse the metric equations, but describe their filtered-coordinate
            # meaning explicitly instead of claiming unfiltered ODE closure.
            methods = methods_section()
            methods['blocks'] = methods['blocks'][:4]
            methods['blocks'][0] = paragraph('输入为完整记录已带通的缓存 Hb。D 是在这个测量坐标上计算的描述性闭合残差；指数项仅作为工程拟合项，不把缓存滤波后的瞬态解释为原始血管初态。窗口参数剖面不提供生理参数识别。')
            for block in methods['blocks']:
                for key in ('text', 'caption'):
                    if isinstance(block.get(key), str):
                        block[key] = block[key].replace('原生读出', '缓存输出').replace('合法指数初态项', '工程指数项')
            good = census.loc[census.status == 'ok']
            content = dict(schema='source_diagnostics_report_v1',
                title='统一带通 Hb 缓存的描述性相容性', subtitle='缓存输入、完整分母与同源示例曲线',
                date=str(plan.get('created_at', ''))[:10],
                summary=[f'计划配对窗 {len(census):,} 个，有效评分 {len(good):,} 个；其余状态全部保留。',
                    '输入为 v5/none、0.01–0.2 Hz 带通、10 Hz 缓存。未重新拟合历史模型面板，旧原生目标与本轮数值不直接比较。'],
                provenance=[str(run), '数值与图线使用同一带通缓存；完整记录预处理先于遮挡。'],
                sections=[dict(title='覆盖范围、状态和分母', blocks=coverage_blocks(census, inventory, metas, str(run), cached=True)),
                    methods, dict(title='固定参数与同窗剖面分布', blocks=distribution),
                    dict(title='较好、中部与困难的带通缓存波形', blocks=hb_example_blocks(run, examples, staging, cached=True))],
                sources=[str(run / 'summary.json'), str(run / 'resolved_config.yaml')])
            content_path = staging / 'report_content.json'
            content_path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            checks = render_report(content_path, output)
        (output / 'report_generation.json').write_text(json.dumps(dict(schema=SCHEMA, owning_run=str(run),
            fnirs_input='unified_cached_fnirs_v1', figure_dpi=DPI, historical_panels_included=False), indent=2) + '\n')
        return checks
    panels = read_panels(run)
    examples = prepare_hb_examples(run, census, inventory)
    source = str(run)
    with tempfile.TemporaryDirectory(prefix="model_compatibility_report_") as staging:
        staging = Path(staging)
        distribution, stats = census_distribution_blocks(census, staging, source + "/summary.csv", retained_summary)
        nonlinear, paired = nonlinear_blocks(panels, staging)
        panel_examples, panel_choices = panel_example_blocks(panels, paired, staging)
        content = {"schema": "source_diagnostics_report_v1", "title": "实测模型相容性：全库存分布与条件面板复核",
                   "subtitle": "相对 Hb 闭合、冻结 EEG–Hb 约束、共同参数节点、空间读出和条件非线性重建",
                   "date": str(plan.get("date", plan.get("created_at", "")))[:10],
                   "summary": [],
                   "provenance": [f"唯一数值输入：{run}", "报告只汇总保留证据；不新增实测拟合、不改变冻结参数或样本分区。"],
                   "sections": [{"title": "覆盖范围、状态和分母", "blocks": coverage_blocks(census, inventory, metas, source)},
                                methods_section(), {"title": "全库存固定参数与逐窗最优闭合分布", "blocks": distribution},
                                {"title": "参数边界与动态激励", "blocks": parameter_blocks(census, staging, source)},
                                {"title": "原始通道身份和记录内时间异质性", "blocks": stratification_blocks(census, staging, source)},
                                {"title": "每库较好、中部与困难的精确原生波形", "blocks": hb_example_blocks(run, examples, staging)},
                                {"title": "冻结面板的同参数联合约束", "blocks": panel_distribution_blocks(panels, staging)},
                                {"title": "条件非线性前向重建与线性对照", "blocks": nonlinear},
                                {"title": "固定空间读出与自由秩下界", "blocks": spatial_blocks(panels, staging)},
                                {"title": "面板失配、空共同集和非线性改善的具体曲线", "blocks": panel_examples}],
                   "sources": [str(run / "plan.json"), str(run / "record_inventory.jsonl"),
                               "src/inference/model_compatibility.py", "experiments/scripts/render_model_compatibility.py"]}
        good = census.loc[census.status == "ok"]
        content["summary"] = [f"Hb 普查保留 {len(inventory)} 条登记记录，实际原生时钟上共枚举 {len(census):,} 个计划配对窗；其中 {len(good):,} 个得到有效闭合评分，其余 {len(census)-len(good):,} 个保留具体状态。各窗长有整窗记录数另列，计划记录不等于该窗长的评分记录。分布必须结合被试等权汇总和支持分母阅读。",
                              "较低的闭合残差仅支持指定读出、支持和尺度下的观测相容性。它不验证潜变量真实性，不证明参数唯一，也不提供噪声校准的显著性结论。"]
        nl = panels["nonlinear"]
        legal_n = int(_bool_series(nl.legal_witness).sum())
        converged_n = int(_bool_series(nl.optimizer_converged).sum())
        content["summary"].insert(1, f"非线性限制：{len(nl)} 个条件Hb前向拟合中 {legal_n} 个保留合法轨迹，但优化收敛仅 {converged_n} 个、未收敛 {len(nl)-converged_n} 个。这些误差是本次预算内找到的上界，不能称最优解，也不能用高误差排除模型；该放松没有同时满足上游EEG动力学的要求。")
        common = panels["compatibility_sets"].loc[panels["compatibility_sets"].tolerance == .5]
        content["summary"].insert(-1, f"冻结四库面板共96个记录×支持任务。按各分量NRMSE≤0.5个训练SD的工程阈值，{int((common.n_nodes > 0).sum())}/{len(common)} 个任务发现同θ共同见证节点，{int((common.n_nodes == 0).sum())} 个任务未发现节点；其中 {int((common.n_nodes == 80).sum())} 个任务的全部80节点均通过，不能据此声称参数已识别。此结果限定于离散工程节点，不是连续参数域排除或噪声校准检验。")
        for dataset in DATASETS:
            selected = retained_summary.loc[(retained_summary.dataset_id == dataset) & (retained_summary.length_s == 90)]
            by_metric = {r["metric"]: r for r in selected.to_dict("records")}
            if "nominal_D" in by_metric and "best_D" in by_metric:
                nominal, best = by_metric["nominal_D"], by_metric["best_D"]
                improvement = by_metric.get("improvement", {})
                content["summary"].insert(-1, f"{label(dataset)} 的90秒原生Hb：被试等权混合分布中位 D 从固定参数 {number(nominal['q50'])} 到逐窗网格最优 {number(best['q50'])}；配对绝对改善 D固定-D最优的中位数为 {number(improvement.get('q50'))}。有效配对窗 {int(nominal['valid_pair_windows']):,} 个，覆盖 {int(nominal['valid_subjects'])} 名库内被试。这是同窗离线拟合收益。")
        median_descriptions = []
        for dataset in DATASETS:
            entries = []
            for length in (30, 90):
                by_metric = {r["metric"]: r for r in retained_summary.loc[(retained_summary.dataset_id == dataset) & (retained_summary.length_s == length)].to_dict("records")}
                entries.append(f"{length}秒 {number(by_metric['nominal_D']['q50'])}→{number(by_metric['best_D']['q50'])}")
            median_descriptions.append(label(dataset) + "：" + "，".join(entries))
        spatial = panels["space"].loc[panels["space"].status == "completed"]
        spatial_pairs = spatial.pivot(index=["dataset", "subject", "record_id"], columns="K", values="D_space_fraction_raw").dropna()
        k1, k2 = spatial.loc[spatial.K == 1], spatial.loc[spatial.K == 2]
        outcome_blocks = [
            paragraph("全库存按被试等权混合分布汇总的固定→逐窗最优D中位数为：" + "；".join(median_descriptions) + "。原始配对窗等权分布另保留，不能与本段被试等权中位数混用。最优网格包含固定节点，较低残差是同窗参数profile的收益。30/90秒窗口和可用记录组成不同，跨窗长差异不作因果解释。"),
            paragraph(f"冻结面板在0.5工程阈值下，{int((common.n_nodes > 0).sum())}/{len(common)} 个任务发现共同节点，{int((common.n_nodes == 0).sum())} 个未发现；{int((common.n_nodes == 80).sum())} 个任务全80节点均可行。这支持部分观测在指定尺度下可由当前离散模型容纳，同时说明宽可行集合不能用于声称τ/κ已识别。REFED与其他库的具体差异见同分母表，不把跨库成功率当作相同观测难度。"),
            paragraph(f"空间Single-Trial面板中，固定读出残差能量比中位数K1={number(k1.D_space_fraction_raw.median(), 4)}、K2={number(k2.D_space_fraction_raw.median(), 4)}；{int((spatial_pairs[2] < spatial_pairs[1]).sum())}/{len(spatial_pairs)} 个配对窗口在K2下降。自由共同rank下界中位数分别为{number(k1.D_rank_joint.median(), 4)}/{number(k2.D_rank_joint.median(), 4)}，仍与固定映射有差。该比较只涉及6人36窗的表示维数，不能把K直接等同生理成分数。"),
            paragraph(f"非线性 {legal_n}/{len(nl)} 个任务找到合法前向轨迹，但优化收敛 {converged_n}/{len(nl)}。因此本轮非线性误差统一作为预算内的上界，既不宣布优于线性，也不用高误差排除非线性模型；它没有提供联合EEG-Hb动力学见证。")
        ]
        content["sections"].append({"title": "本组证据的结论与使用边界", "blocks": outcome_blocks + [
            paragraph("全库存分布回答相对 Hb 观测在当前线性闭合下的描述性相容程度；冻结面板回答指定 EEG–Hb 读出和训练尺度下的共同约束；空间面板回答局部冻结 loading 的表示能力。三者具有不同分母、算子和归一化，不能拼成一个生理解释百分比。"),
            paragraph("对每条记录分别最小化不同指标不能替代共同 θ 的节点检验。较小 D、宽参数 profile 或满格共同节点都不能单独证明参数可识别；低激励和自由 nuisance 会降低排除能力。空共同集只表示工程网格未发现见证。非线性较小残差仍只提供已找到的条件重建上界，不能据此恢复绝对血容量、去氧量或神经驱动真值。"),
            paragraph("这些结果可用于定位需要复核的观测通道、时间支持和模型限制，或在既有冻结面板上比较读出。任何新的数据范围、映射训练、参数选择或正式保护评价，均需另按 owning protocol 执行；本报告没有进行此类后续实验。")
        ]})
        content_path = staging / "report_content.json"
        content_path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        checks = render_report(content_path, output)
    (output / "report_generation.json").write_text(json.dumps({"schema": SCHEMA, "owning_run": str(run),
        "census_rows": int(len(census)), "census_ok": int(len(good)), "selection_rule": "fixed nominal score nearest q=.1,.5,.9; distinct subjects where possible",
        "panel_example_rules": panel_choices, "owning_summary": str(run / "summary.json"),
        "space_raw_distance_source": str(panels.get("space_numerical_followup", panels["directory"] / "space.csv")),
        "bootstrap_unit": terminal["bootstrap_unit"], "bootstrap_repeats": terminal["bootstrap_replicates"],
        "figure_dpi": DPI, "protected_data_read": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    checks = build_report(args.run, args.output or args.run / "report_v1")
    print(json.dumps({"output": checks["output"], "pages": checks["exports"]["TECHNICAL_REPORT.pdf"]["pages"],
                      "figures": checks["block_counts"].get("figure", 0)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
