#!/usr/bin/env python3
"""Export retained mode-driver evidence to a searchable PDF with PNG figures.

This command reads only the named run's summary/metric tables. It never loads
prepared signals, fits a model, chooses a parameter, or computes new intervals.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from fontTools.ttLib import TTFont


FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
ARMS = ["R0_PCA_refit", "R1_broadband", "R2_zero", "R2_selected", "independent"]
ARM_LABELS = {"R0_PCA_refit": "R0 PCA重拟合", "R1_broadband": "R1 宽频", "R2_zero": "R2 γ=0", "R2_selected": "R2 训练选γ", "independent": "独立第三轨迹", "training_template": "训练模板", "own_context": "自身上下文", "ridge_cross": "跨模态岭回归"}
DATASET_LABELS = {"eeg_fnirs_single_trial": "单次任务", "simultaneous_eeg_nirs": "同步 EEG/NIRS", "visual_cognitive_motivation": "视觉认知动机"}
MODE_LABELS = {"full": "全观测", "Hb_hidden": "Hb全部隐藏", "center_Hb": "Hb中心隐藏", "HbO_hidden": "HbO隐藏", "HbR_hidden": "HbR隐藏", "center_EEG": "EEG中心隐藏", "EEG_channels_hidden": "EEG通道隐藏"}
SCENARIO_LABELS = {"a_only": "仅宽频a", "b_only_gamma0": "仅谱对比b，γ=0", "both": "a/b共同改变", "common_Hb": "共同Hb成分", "independent_Hb": "独立Hb驱动", "wrong_spectral": "谱结构失配", "lag": "时间错位"}
NULL_LABELS = {"wrong_training_subject": "训练被试错配", "nonwrapping_shift_12s": "12秒非循环移位", "other_region_EEG": "另一脑区EEG"}
COLORS = ["#64748b", "#2563eb", "#14b8a6", "#ea580c", "#9333ea", "#a3a3a3", "#71717a", "#475569"]


def number(value, digits=3):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{value:.{digits}f}" if np.isfinite(value) else "—"


def label(value, labels):
    return labels.get(str(value), str(value))


def read_table(run, name, columns):
    path = run / name
    if not path.is_file():
        raise ValueError(f"Missing retained evidence table: {path}")
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame(columns=columns)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    return frame


class Report:
    """Small fixed-page compositor; body text and tables remain PDF text."""

    def __init__(self, out):
        self.out = out
        self.pdf = fitz.open()
        # The collection's default member is Japanese; select its Simplified
        # Chinese member to preserve normal Chinese Unicode during extraction.
        font_buffer = BytesIO()
        font = TTFont(str(FONT_PATH), fontNumber=2)
        # MuPDF chooses the compatibility alias for some shared glyphs (理,
        # 不, 量). The report uses normal Unicode, so remove those unused aliases
        # before embedding to make exact Chinese text search work as written.
        for cmap in font["cmap"].tables:
            if cmap.isUnicode():
                cmap.cmap = {code: glyph for code, glyph in cmap.cmap.items()
                             if not (0xF900 <= code <= 0xFAFF or 0x2F800 <= code <= 0x2FA1F)}
        font.save(font_buffer)
        self.font = fitz.Font(fontbuffer=font_buffer.getvalue())
        self.page = None
        self.y = 0.
        self.figures = []
        self.text_boxes = []

    def new_page(self, title):
        self.page = self.pdf.new_page(width=595.28, height=841.89)
        self.y = 35.
        self.text(title, size=17, gap=11, color=(.08, .18, .31))
        self.page.draw_line((36, self.y), (559, self.y), color=(.7, .76, .82), width=.5)
        self.y += 13

    def wrap(self, text, width, size):
        lines = []
        for paragraph in str(text).split("\n"):
            if not paragraph:
                lines.append("")
                continue
            current = ""
            for char in paragraph:
                if current and self.font.text_length(current + char, fontsize=size) > width:
                    lines.append(current)
                    current = char
                else:
                    current += char
            lines.append(current)
        return lines

    def text(self, text, *, size=9.5, gap=7, color=(.12, .15, .2)):
        lines = self.wrap(text, 523, size)
        height = len(lines) * size * 1.5
        if self.y + height > 795:
            raise ValueError("Report text overflows a page; shorten the conclusion or split this section")
        writer = fitz.TextWriter(self.page.rect)
        for line in lines:
            if line:
                writer.append((36, self.y + size), line, font=self.font, fontsize=size)
            self.y += size * 1.5
        writer.write_text(self.page, color=color)
        self.text_boxes.append((len(self.pdf), tuple(writer.text_rect)))
        self.y += gap

    def table(self, headers, rows, widths, *, size=8.2, row_height=19):
        if abs(sum(widths) - 523) > .1:
            raise ValueError("Table widths must sum to the report's 523pt body")
        if self.y + (len(rows)+1)*row_height > 795:
            raise ValueError("Report table overflows a page")
        for index, row in enumerate([headers, *rows]):
            self.page.draw_rect(fitz.Rect(36, self.y, 559, self.y+row_height), color=None,
                                fill=(.91, .94, .97) if index == 0 else ((.97, .98, .99) if index % 2 else (1, 1, 1)))
            x = 36
            writer = fitz.TextWriter(self.page.rect)
            for value, width in zip(row, widths):
                text = str(value)
                if self.font.text_length(text, fontsize=size) > width-8:
                    raise ValueError(f"Table cell too wide: {text!r}")
                writer.append((x+4, self.y+row_height/2+size*.35), text, font=self.font, fontsize=size)
                x += width
            writer.write_text(self.page, color=(.12, .15, .2))
            self.y += row_height
        self.y += 10

    def figure(self, path, caption, *, max_height):
        with fitz.open(path) as source:
            ratio = source[0].rect.height/source[0].rect.width
        pixels = fitz.Pixmap(str(path))
        width, height = 523., 523.*ratio
        if height > max_height:
            height = max_height
            width = height/ratio
        if self.y+height > 795:
            raise ValueError("Report figure overflows a page")
        x = 36 + (523-width)/2
        self.page.insert_image(fitz.Rect(x, self.y, x+width, self.y+height), filename=str(path))
        effective_dpi = pixels.width/(width/72.)
        if not 200 <= effective_dpi <= 300:
            raise ValueError(f"Figure resolution is {effective_dpi:.1f} dpi at its displayed size: {path}")
        self.figures.append(dict(file=str(path.relative_to(self.out)), page=len(self.pdf), display_width_pt=width, display_height_pt=height, effective_dpi=effective_dpi, caption=caption))
        self.y += height+6
        self.text(caption, size=8.2, gap=10, color=(.3, .35, .4))

    def save(self, run):
        if not 3 <= len(self.pdf) <= 6:
            raise ValueError(f"Expected 3–6 report pages, got {len(self.pdf)}")
        for index, page in enumerate(self.pdf):
            writer = fitz.TextWriter(page.rect)
            writer.append((36, 815), "共享驱动模式 v1 | 公开开发面板条件诊断", font=self.font, fontsize=7)
            writer.append((525, 815), f"{index+1}/{len(self.pdf)}", font=self.font, fontsize=7)
            writer.write_text(page, color=(.4, .45, .5))
        self.pdf.set_metadata(dict(title="共享驱动 r(t)：区域谱模式开发面板诊断", author="PID-MCM project", subject="Frozen shared_driver_modes_v1 evidence; conditional reconstruction diagnostic"))
        # MuPDF's current native subsetter reports CFF errors for this Noto TTC.
        # Its documented fontTools path retains selectable text and is reliable.
        self.pdf.subset_fonts(fallback=True)
        path = self.out / "REPORT.pdf"
        self.pdf.save(path, garbage=4, deflate=True)
        self.pdf.close()
        with fitz.open(path) as check:
            pages = []
            for index, page in enumerate(check):
                text = page.get_text()
                if len(text) < 30:
                    raise ValueError(f"Page {index+1} has no searchable body text")
                page.get_pixmap(dpi=120, alpha=False).save(self.out/f"page_{index+1:02d}.png")
                pages.append(dict(page=index+1, image_objects=len(page.get_images(full=True)), searchable_text_chars=len(text)))
            if sum(p["image_objects"] for p in pages) != len(self.figures):
                raise ValueError("Every figure must be embedded once as a bitmap image object")
            if "共享驱动" not in "".join(p.get_text() for p in check):
                raise ValueError("Chinese PDF text extraction failed")
        validation = dict(source_run=str(run), report="REPORT.pdf", bitmap_dpi=240,
                          pages=pages, figures=self.figures,
                          searchable_body_and_tables=True, vector_figures=False,
                          rendered_preview_files=[f"page_{index+1:02d}.png" for index in range(len(pages))],
                          wps_checked=False)
        (self.out/"render_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2)+"\n")
        return validation


def save_figure(fig, out, name, *, max_height=None):
    fig.tight_layout(rect=(0, 0, 1, .95))
    path = out/"figures"/(name+".png")
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = bbox.width+.28, bbox.height+.28
    displayed_width = min(523./72., (max_height/72.)*width/height) if max_height else 523./72.
    # Match 240 dpi to the final PDF display size, including tight-bbox padding.
    fig.savefig(path, dpi=240.*displayed_width/width, facecolor="white", bbox_inches="tight", pad_inches=.14)
    plt.close(fig)
    return path


def component_plot(frame, out, *, dataset, regions, name):
    modes = ["full", "Hb_hidden", "center_Hb"]
    fig, axes = plt.subplots(3, len(regions), figsize=(7.2, 6.4), squeeze=False)
    arms = [arm for arm in ARMS if arm in set(frame.arm)]
    for row_index, mode in enumerate(modes):
        metrics = [("EEG", "EEG_nrmse"), ("HbO", "HbO_nrmse"), ("HbR", "HbR_nrmse")] if mode == "full" else [("HbO", "HbO_nrmse"), ("HbR", "HbR_nrmse")]
        width = .8/len(metrics)
        for column_index, region in enumerate(regions):
            ax = axes[row_index, column_index]
            part = frame[(frame.dataset == dataset) & (frame.region == region) & (frame["mode"] == mode)].set_index("arm")
            if part.index.duplicated().any():
                raise ValueError(f"Duplicate owning summary rows for {dataset}/{region}/{mode}; no implicit reaggregation")
            x = np.arange(len(arms))
            for index, (component, metric) in enumerate(metrics):
                values = pd.to_numeric(part.reindex(arms)[metric], errors="coerce").to_numpy(float)
                bars = ax.bar(x+(index-(len(metrics)-1)/2)*width, values, width, label=component,
                              color={"EEG": "#64748b", "HbO": "#ea580c", "HbR": "#2563eb"}[component])
                if mode == "full":
                    for bar, value in zip(bars, values):
                        if np.isfinite(value):
                            ax.annotate(number(value, 2), (bar.get_x()+bar.get_width()/2, value),
                                        xytext=(0, 3), textcoords="offset points", fontsize=6.5,
                                        ha="center", va="bottom", rotation=90)
                    ax.margins(y=.3)
            short = {"R0_PCA_refit": "R0", "R1_broadband": "R1", "R2_zero": "R2 γ0", "R2_selected": "R2 γ选", "independent": "独立"}
            ax.set_xticks(x, [short[arm] for arm in arms], rotation=35, ha="right", fontsize=8)
            ax.set_title(f"区{column_index+1} / {label(mode, MODE_LABELS)}", fontsize=9)
            ax.set_ylabel("训练SD NRMSE", fontsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(metrics), fontsize=8, frameon=False)
    fig.subplots_adjust(top=.94, hspace=.75, wspace=.4)
    return save_figure(fig, out, name, max_height=455)


def interval_plot(frame, out, *, gain, name, labels, max_height):
    """Plot the owning paired differences and intervals, without recomputation."""
    fig, ax = plt.subplots(figsize=(7.2, max(2.4, .22*len(frame)+.8)))
    values = pd.to_numeric(frame[gain], errors="coerce").to_numpy(float)
    low = pd.to_numeric(frame.ci_low, errors="coerce").to_numpy(float)
    high = pd.to_numeric(frame.ci_high, errors="coerce").to_numpy(float)
    finite = np.isfinite(values) & np.isfinite(low) & np.isfinite(high)
    positions = np.arange(len(frame))
    for index in positions[finite]:
        ax.plot([low[index], high[index]], [index, index], lw=1.6, color="#2563eb")
        ax.plot(values[index], index, "o", ms=4, color="#ea580c")
    for index in positions[~finite]:
        ax.text(0, index, "无共同成功或区间不可用", va="center", fontsize=7, color="#71717a")
    ax.axvline(0, color="#64748b", lw=.8)
    ax.set_yticks(positions, labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("NRMSE增益（正值为候选／真实配对更好）", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", alpha=.2)
    return save_figure(fig, out, name, max_height=max_height)


def synthetic_plot(frame, out):
    scenarios = list(frame.scenario.drop_duplicates())
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.7), sharex=True)
    arms = ["R1_broadband", "R2_selected", "independent"]
    x, width = np.arange(len(scenarios)), .8/len(arms)
    for ax, mode in zip(axes, ["a", "b"]):
        for index, arm in enumerate(arms):
            part = frame[frame.arm == arm].set_index("scenario")
            if part.index.duplicated().any():
                raise ValueError("Synthetic state figure expects one retained row per scenario/arm")
            values = pd.to_numeric(part.reindex(scenarios)[mode+"_reference_centered_RMSE"], errors="coerce").to_numpy(float)
            ax.bar(x+(index-(len(arms)-1)/2)*width, values, width,
                   label=label(arm, ARM_LABELS), color=["#2563eb", "#ea580c", "#9333ea"][index])
        ax.set_ylabel(f"{mode} 参考后RMSE", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[-1].set_xticks(x, [label(scenario, SCENARIO_LABELS) for scenario in scenarios], rotation=25, ha="right", fontsize=8)
    axes[0].legend(fontsize=8, ncol=3, frameon=False)
    return save_figure(fig, out, "synthetic_state_recovery", max_height=210)


def render(run, out, conclusions):
    run, out = Path(run).resolve(), Path(out).resolve()
    if out == run or (out.exists() and any(out.iterdir())):
        raise ValueError("Export destination must be a new, empty versioned directory")
    if not FONT_PATH.is_file():
        raise ValueError(f"Required Chinese font is missing: {FONT_PATH}")
    summary = json.loads((run/"summary.json").read_text())
    for kind in ["synthetic", "measured"]:
        if summary[kind]["recorded"] != summary[kind]["planned"]:
            raise ValueError(f"{kind} evidence is incomplete; retained planned cells must all be recorded")
    measured = read_table(run, "measured_summary.csv", ["dataset", "region", "mode", "pairing", "arm", "planned", "converged", "EEG_nrmse", "HbO_nrmse", "HbR_nrmse"])
    synthetic = read_table(run, "synthetic_summary.csv", ["scenario", "mode", "arm", "planned", "converged", "EEG_nrmse", "HbO_nrmse", "HbR_nrmse", "a_reference_centered_RMSE", "b_reference_centered_RMSE"])
    paired = read_table(run, "measured_paired.csv", ["dataset", "region", "mode", "arm", "control", "planned_pairs", "common_success_pairs", "subjects", "gain", "ci_low", "ci_high", "metric"])
    nulls = read_table(run, "pairing_nulls.csv", ["dataset", "region", "mode", "arm", "control", "planned_pairs", "common_success_pairs", "subjects", "gain", "ci_low", "ci_high", "metric"])
    # Cell tables are retained as the audit denominator. Read no cell arrays,
    # predictions, prepared caches, historical run, or dataset inputs.
    measured_cells = read_table(run, "measured_cells.csv", ["dataset", "region", "mode", "arm", "converged", "status"])
    read_table(run, "synthetic_cells.csv", ["scenario", "mode", "arm", "converged"])
    measured = measured[measured.pairing == "real"].copy()
    datasets = list(measured.dataset.drop_duplicates())
    if not 1 <= len(datasets) <= 3:
        raise ValueError("A short report requires evidence from one to three declared datasets")
    regions_by_dataset = {dataset: list(measured[measured.dataset == dataset].region.drop_duplicates()) for dataset in datasets}
    region_codes = {(dataset, region): f"区{index+1}" for dataset, regions in regions_by_dataset.items() for index, region in enumerate(regions)}
    out.mkdir(parents=True, exist_ok=True)
    (out/"figures").mkdir()
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(FONT_PATH)).get_name(),
                         "font.size": 9, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    report = Report(out)
    report.new_page("共享驱动 r(t)：区域谱模式开发面板诊断")
    report.text(f"实验：{summary.get('experiment_id', 'SSM-SHARED-DRIVER-MODES-v1')}\n证据：{run.name}；所有数值来自本次 run 的冻结汇总表。", size=10)
    measured_state = summary["measured"]
    report.text(f"公开开发面板：{measured_state['subjects']}被试，{measured_state['original_windows']}原生窗口，{measured_state['expanded_windows']}区域窗。区域窗重复使用同一原生记录，不能作为独立样本；区域排除{len(measured_state.get('regional_exclusions', []))}项。未使用受保护测试数据。", size=10)
    report.table(["模型", "状态与读出"], [
        ["R0 PCA重拟合", "父PCA固定读出；同24 DCT约束的新基线"],
        ["R1 宽频", "固定同向宽频谱读出；一个状态a"],
        ["R2 γ=0", "宽频a＋α/β谱对比b；血管仅读取a"],
        ["R2 训练选γ", "血管读取(a+γb)/√(1+γ²)，γ仅训练选择"],
        ["独立第三轨迹", "EEG读取a/b；Hb另用一条独立驱动"],
    ], [135, 388], size=9.2, row_height=23)
    report.text("全部臂固定原 Balloon 核心、24 DCT含DC、AR(1)与共同Hb项。4 Hz状态表示慢变包络；本轮DCT最高频率约0.383 Hz。R0不等于历史逐点M0；独立臂是解耦参照，额外先验惩罚使其并非严格误差上界。", size=9.5)
    report.table(["证据", "计划", "已记录", "成功", "未成功"], [
        ["合成", str(summary['synthetic']['planned']), str(summary['synthetic']['recorded']), str(summary['synthetic']['converged']), str(summary['synthetic']['recorded']-summary['synthetic']['converged'])],
        ["实测", str(summary['measured']['planned']), str(summary['measured']['recorded']), str(summary['measured']['converged']), str(summary['measured']['recorded']-summary['measured']['converged'])],
    ], [147, 94, 94, 94, 94], size=9)
    unsuccessful = measured_cells[measured_cells.arm.isin(ARMS) & ~measured_cells.converged]
    numerical = int((unsuccessful.status == "failed_numerical").sum())
    donor_missing = int((unsuccessful.status == "matched_donor_unavailable").sum())
    report.text(f"实测未成功包括数值失败{numerical}条、匹配供体不可用{donor_missing}条；后者未进入优化，不能称为未收敛。", size=8.7)
    for conclusion in conclusions[:1] or ["报告保留重建、隐藏补全、配对优势与失败分母。结果仅支持开发面板上的条件诊断；模式名称和较低重建误差不能单独证明生理语义已恢复。"]:
        report.text(conclusion, size=10)
    simple = paired[(paired["mode"] == "Hb_hidden") & (paired.arm == "R2_selected") & paired.control.isin(["training_template", "own_context", "ridge_cross"]) & (paired.metric == "Hb_nrmse")]
    baseline_rows = []
    for dataset in datasets:
        for region in regions_by_dataset[dataset]:
            part = simple[(simple.dataset == dataset) & (simple.region == region)].set_index("control")
            if part.index.duplicated().any():
                raise ValueError("Duplicate owning simple-baseline paired rows")
            values = [f"{float(part.loc[control, 'gain']):+.3f}" if control in part.index and np.isfinite(part.loc[control, 'gain']) else "—" for control in ["training_template", "own_context", "ridge_cross"]]
            baseline_rows.append([label(dataset, DATASET_LABELS), region_codes[(dataset, region)], *values])
    if baseline_rows:
        report.text("Hb全部隐藏：简单基线−R2的配对NRMSE增益（正值R2更好）", size=9.2)
        report.table(["数据集", "区域", "训练模板", "自身上下文", "跨模态岭回归"], baseline_rows, [190, 48, 95, 95, 95], size=8, row_height=16)
        report.text("直接读取共同成功配对增益，未重聚合；此表仅点估计，区间见原measured_paired.csv。负值表明R2尚未胜过该简单基线。", size=8.3)

    for dataset in datasets:
        regions = regions_by_dataset[dataset]
        if not 1 <= len(regions) <= 3:
            raise ValueError("A dataset figure requires one to three declared regions")
        report.new_page(f"{label(dataset, DATASET_LABELS)}：全观测与隐藏Hb分项")
        report.text("；".join(f"区{index+1}={region}" for index, region in enumerate(regions))+"。区域沿用父prepared记录，不表示源空间定位。", size=8.5, gap=5)
        path = component_plot(measured, out, dataset=dataset, regions=regions, name=f"measured_{dataset}")
        report.figure(path, "图为各臂成功样本的非配对描述汇总：先被试均值再等权。成功样本因臂而异，不能据此认定重建改进；比较参照共同成功配对证据（第5页）。未成功保留在计划分母。", max_height=455)
        rows = []
        for region in regions:
            part = measured[(measured.dataset == dataset) & (measured.region == region) & (measured["mode"] == "Hb_hidden")].set_index("arm")
            for arm in ["R1_broadband", "R2_selected", "independent"]:
                if arm not in part.index:
                    continue
                row = part.loc[arm]
                rows.append([region_codes[(dataset, region)], label(arm, ARM_LABELS), number(row.HbO_nrmse), number(row.HbR_nrmse), f"{int(row.planned-row.converged)}/{int(row.planned)}"])
        report.table(["区域", "Hb全部隐藏", "HbO", "HbR", "未成功/计划"], rows, [58, 208, 70, 70, 117], size=8.3, row_height=17)
        report.text("离线处理后的特征补全不是原始传感器遮挡或因果未来预测。仅较小残差不足以确认唯一共享神经来源。", size=8.6)

    report.new_page("配对差与匹配null：按被试整体bootstrap")
    primary = paired[(paired["mode"] == "Hb_hidden") & (paired.arm == "R2_selected") & paired.control.isin(["R1_broadband", "independent"]) & (paired.metric == "Hb_nrmse")].copy()
    primary = primary.sort_values(["dataset", "region", "control"], kind="stable")
    primary_labels = [f"{label(row.dataset, DATASET_LABELS)}·{region_codes[(row.dataset, row.region)]} / {label(row.control, ARM_LABELS)} [{int(row.common_success_pairs)}/{int(row.planned_pairs)}]" for row in primary.itertuples()]
    report.text("上图：R2训练选γ相对R1或独立臂的隐藏Hb NRMSE增益；下图：真实配对相对错配的增益。正值更好；中括号为共同成功/计划配对。", size=9.2)
    if len(primary):
        path = interval_plot(primary, out, gain="gain", name="paired_Hb_hidden", labels=primary_labels, max_height=245)
        report.figure(path, "点和区间直接读取 owning measured_paired.csv；区间为1000次被试块bootstrap，条件于冻结坐标与共同成功记录。", max_height=245)
    bad = nulls[(nulls["mode"] == "Hb_hidden") & (nulls.metric == "Hb_nrmse")].copy()
    bad = bad.sort_values(["dataset", "region", "control"], kind="stable")
    null_labels = [f"{label(row.dataset, DATASET_LABELS)}·{region_codes[(row.dataset, row.region)]} / {label(row.control, NULL_LABELS)} [{int(row.common_success_pairs)}/{int(row.planned_pairs)}]" for row in bad.itertuples()]
    if len(bad):
        path = interval_plot(bad, out, gain="gain", name="matched_pair_nulls", labels=null_labels, max_height=330)
        report.figure(path, "12秒非循环移位与real_shift_support使用相同评分支持；错被试和另一脑区null沿用已冻结身份。未把跨区域重复当作独立n。", max_height=330)
    if not len(primary) or not len(bad):
        report.text("当前表中某类Hb_hidden配对/区间证据缺失；不能据此宣布真实共享结构成立。", size=9)
    report.text("区域内区间使用表中subjects字段所计被试；本轮总体最多18被试。区间为探索性开发面板估计，不能视为总体确认或teacher资格。其他遮挡模式及完整配对表保留在原run。", size=8.6)

    report.new_page("合成恢复、失配与生理语义边界")
    syn = synthetic[synthetic["mode"] == "full"].copy()
    path = synthetic_plot(syn, out)
    report.figure(path, "a/b误差是首5秒参考后RMSE，单位为固定signed log-power模式坐标；不是NRMSE。R1没有b坐标，缺项不作为零误差。DC误差另列在原汇总；参考后恢复不能证明绝对静息水平。", max_height=210)
    rows = []
    for scenario in syn.scenario.drop_duplicates():
        part = syn[syn.scenario == scenario].set_index("arm")
        for arm in ["R1_broadband", "R2_selected", "independent"]:
            if arm not in part.index:
                continue
            row = part.loc[arm]
            rows.append([label(scenario, SCENARIO_LABELS), {"R1_broadband": "R1", "R2_selected": "R2 γ选", "independent": "独立"}[arm], number(row.EEG_nrmse), number(row.HbO_nrmse), number(row.HbR_nrmse), f"{int(row.planned-row.converged)}/{int(row.planned)}"])
    report.table(["合成情景", "臂", "EEG", "HbO", "HbR", "未成功/计划"], rows, [165, 65, 67, 67, 67, 92], size=7.8, row_height=14)
    for conclusion in conclusions[1:] or ["固定宽频/α–β谱对比坐标只定义区域有效活动模式，不等于E/I、微观同步率或代谢量。两维自由度带来的重建收益需由隐藏补全、实配增益和失配情景共同支持。"]:
        report.text(conclusion, size=8.8)
    report.text("单输入Balloon核心仍约束HbO/HbR关系；升维没有消除这一结构性限制。", size=8.6)
    report.text("参考：Hermes等（2017），PLOS Biology，群体活动的不同测量汇集方式；其ECoG/fMRI结果未校准本轮EEG/fNIRS状态。", size=8.3, gap=2)
    link_text = "https://doi.org/10.1371/journal.pbio.2001461"
    link_y = report.y
    report.text(link_text, size=8.1, color=(.1, .25, .55))
    report.page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(36, link_y, 280, report.y), "uri": link_text})
    return report.save(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Defaults to RUN/report_v1; an existing nonempty export is refused")
    parser.add_argument("--conclusion", action="append", default=[], help="Reviewed interpretation: first paragraph on page 1, further paragraphs on the final page")
    args = parser.parse_args()
    validation = render(args.run_dir, args.output_dir or args.run_dir/"report_v1", args.conclusion)
    print(json.dumps(dict(report=str((args.output_dir or args.run_dir/"report_v1")/"REPORT.pdf"), pages=len(validation["pages"]), figures=len(validation["figures"])), ensure_ascii=False))


if __name__ == "__main__":
    main()
