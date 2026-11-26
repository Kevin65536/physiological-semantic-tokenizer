"""Make the full-window search report from its audited CSV evidence."""

import argparse
from html import escape
import json
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans", "Noto Sans CJK JP"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#223d4a",
        "axes.titleweight": "bold",
        "axes.unicode_minus": False,
    }
)
BLUE = "#146678"
ORANGE = "#c15e34"
DARK = "#203b4b"
GRAY = "#81929b"
DS = {"eeg_fnirs_single_trial": "Single-Trial", "refed": "REFED", "simultaneous_eeg_nirs": "Simultaneous"}
NAMES = {"eeg": "EEG", "hb": "Hb"}
SHAPES = ["span", "scatter", "site"]
LABEL = {"span": "连续时段", "scatter": "随机时间片", "site": "整个位置"}
STREAM = ["EEG-S", "EEG-O", "Hb-S", "Hb-O"]
ROUTE = {"eeg": "Hb → EEG", "hb": "EEG → Hb"}
KEYS = ["mask", "whole_percent", "width", "head"]


def graph(n=1, m=1, h=4.8):
    return plt.subplots(n, m, figsize=(12.1, h), squeeze=False, layout="constrained")


def display_config(name):
    mask, pct, width, head = name.split("-")
    return f"{LABEL[mask]}／{pct}%／{width}／" + {"prior": "词频残差", "linear": "线性头"}[head]


def heat(ax, vals, xx, yy, fmt=".2f", limit=None, cmap="RdBu", markzero=True):
    limit = limit or np.max(abs(vals))
    norm = TwoSlopeNorm(0, -limit, limit) if markzero else None
    im = ax.imshow(vals, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(xx)), xx)
    ax.set_yticks(range(len(yy)), yy)
    for (i, j), v in np.ndenumerate(vals):
        ax.text(
            j,
            i,
            format(v, fmt),
            ha="center",
            va="center",
            fontsize=10,
            color="white" if abs(v) > limit * 0.65 else "#162d39",
        )
    return im


class Book:
    def __init__(self, out):
        self.doc = fitz.open()
        self.out = out
        self.titles = []
        (out / "figures").mkdir(exist_ok=True)

    def put(self, p, box, content, size, color=DARK):
        rem, _ = p.insert_htmlbox(
            fitz.Rect(box),
            escape(content),
            css=f"*{{font-family:sans-serif;font-size:{size}pt;color:{color};line-height:1.32}}",
            scale_low=1,
        )
        if rem < 0:
            raise ValueError("report overflow: " + content)

    def page(self, slug, title, intro, fig, caption, source):
        self.titles.append(title)
        idx = len(self.titles)
        p = self.doc.new_page(width=940, height=680)
        self.put(
            p,
            [28, 14, 900, 40],
            f"EEG 与 fNIRS　｜　全量非重叠窗口掩码预训练　　2026-11-26　　训练人群两折搜索　　　{idx:02}",
            10,
            BLUE,
        )
        self.put(p, [28, 40, 912, 82], title, 22)
        self.put(p, [28, 89, 912, 161], intro, 13)
        if isinstance(fig, list):
            for x, tab in zip([28, 476], fig):
                space, _ = p.insert_htmlbox(
                    fitz.Rect(x, 165, x + 436, 520),
                    tab,
                    css="*{font-family:sans-serif;font-size:9.6pt;color:#203b4b} table{width:100%;border-collapse:collapse} td,th{padding:5px;border-bottom:1px solid #d5e0e2} th{background:#dcebed}",
                    scale_low=1,
                )
                if space < 0:
                    raise ValueError("candidate table overflow")
        else:
            path = self.out / "figures" / f"{idx:02}_{slug}.png"
            fig.savefig(path, dpi=240, facecolor="white")
            plt.close(fig)
            p.insert_image(fitz.Rect(28, 162, 912, 516), filename=str(path), keep_proportion=True)
        self.put(p, [28, 528, 912, 624], caption, 13)
        self.put(p, [28, 634, 912, 673], source, 9.8, BLUE)

    def finish(self):
        self.doc.set_toc([[1, t, i + 1] for i, t in enumerate(self.titles)])
        self.doc.set_metadata(
            {"title": "全量非重叠窗掩码预训练搜索：训练结果与生理读出", "author": "PID-MCM"}
        )
        self.doc.subset_fonts()
        self.doc.save(self.out / "REPORT.pdf", deflate=True, garbage=4)
        self.doc.close()


def outcomes(t):
    keys = ["candidate", "target"]
    q = t.groupby(keys)[["gain", "prior_gain", "pair_gain"]].min().rename(columns=lambda x: "worst_" + x)
    q["cross_margin"] = q[["worst_prior_gain", "worst_pair_gain"]].min(axis=1)
    return q.reset_index()


def nearest(q, target, what):
    g = q[q.target == target]
    return g.loc[g[what].idxmax(), "candidate"]


def overview(b, t, q):
    fig, axes = graph(1, 2)
    for ax, key, title, unit in zip(
        axes[0],
        ["worst_gain", "cross_margin"],
        ["局部属性读出", "整模态遮蔽"],
        ["归一化误差增益", "bit / 隐藏词元"],
    ):
        for i, mod in enumerate(NAMES):
            vals = q[q.target.eq(mod)][key].sort_values().to_numpy()
            ax.scatter(
                vals, i + np.linspace(-0.14, 0.14, len(vals)), s=29, color=[BLUE, ORANGE][i], alpha=0.75
            )
            ax.annotate(
                f"{max(vals):+.3f}",
                (max(vals), i + 0.14),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
            )
        ax.axvline(0, color=DARK, ls="--")
        ax.set(
            yticks=[0, 1],
            yticklabels=[NAMES[m] if key == "worst_gain" else ROUTE[m] for m in NAMES],
            ylim=(-0.55, 1.55),
            xlabel=unit + "　（正值晋级）",
            title=title,
        )
        ax.grid(axis="x", alpha=0.16)
    b.page(
        "decisions",
        "48 次拟合完成，24 个配置均在两折初筛结束",
        "每点是一个配置。局部读出取两折中较小的生理属性增益；整模态预测取两折、两个对照共四个增益的最小值。虚线为预设晋级边界，各指标须严格大于零。",
        fig,
        "局部读出最接近门槛的是 EEG，相差 0.0340 NMSE。整模态预测最接近的是 EEG 输入补全 Hb，相差 0.4151 bit/词元。两个方向均按计划停止；三种子复现、6 秒模板筛选及随后实测 Hb 增量没有产生本轮结果。",
        "证据　training_summary.json、A_selection.json、tables/fold_scores.csv；每个终点均在拟合时未见的被试上评分。",
    )


def coverage(b, p, op):
    g = (
        p.groupby(["dataset", "fold"])
        .agg(
            windows=("subject", "size"),
            people=("subject", "nunique"),
            strict=("donor", lambda x: (x >= 0).sum()),
        )
        .reset_index()
    )
    fig, axes = graph(1, 2)
    for ax, col, title in zip(
        axes[0], ["windows", "strict"], ["全部局部窗口", "同条件严格配对窗口（左图子集）"]
    ):
        for f in (0, 1):
            d = g[g.fold.eq(f)].set_index("dataset").loc[list(DS)]
            rect = ax.bar(
                np.arange(3) + (f - 0.5) * 0.36,
                d[col],
                width=0.34,
                color=[BLUE, ORANGE][f],
                label=f"人群折 {f}",
                hatch="" if not f else "//",
            )
            ax.bar_label(rect, padding=2)
        ax.set(xticks=range(3), xticklabels=list(DS.values()), ylabel="30 秒窗数", title=title)
        ax.margins(y=0.25)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
    b.page(
        "support",
        "2,827 个局部窗和 156 个严格配对窗均参与了训练",
        "51 名被试按库分为两折。拟合折 0 含 27 人、1,524 窗，在折 1 的 24 人、1,303 窗上评分；拟合折 1 时交换。严格配对组共 32 人，跨模态评分时两折分别留出 79 窗／15 人和 77 窗／17 人。",
        fig,
        "全部 48 个终点检查点都覆盖了所在拟合折的每个局部窗，留出折训练曝光为零。每次拟合的 8,192 次抽窗中，4,096 次用于分模态局部训练，4,096 次来自严格配对组。REFED 仅进入局部任务。770 个普通开发窗预留给 C 阶段。",
        "证据　train/panel.csv、各模型 exposure.csv 与终点检查点；核验汇总见 tables/fit_audit.csv。",
    )
    return g


def local(b, t, q):
    fig, axes = graph(1, 2)
    for ax, m in zip(axes[0], NAMES):
        g = t[t.target.eq(m)]
        name = nearest(q, m, "worst_gain")
        for w, marker in [(128, "o"), (256, "^")]:
            a = g[g.width.eq(w)].pivot(index="candidate", columns="fold", values="gain")
            ax.scatter(a[0], a[1], s=48, marker=marker, color=BLUE if w == 128 else ORANGE, label=f"宽 {w}")
        row = g[g.candidate.eq(name)].set_index("fold")
        ax.scatter(
            row.at[0, "gain"],
            row.at[1, "gain"],
            s=190,
            edgecolors=DARK,
            facecolors="none",
            lw=1.4,
            label="最接近门槛",
        )
        ax.axhline(0, ls="--", color=GRAY)
        ax.axvline(0, ls="--", color=GRAY)
        ax.set(
            title=NAMES[m] + (" 相对 α 功率" if m == "eeg" else "O 局部斜率"),
            xlabel="拟合折 0、评分折 1 的增益",
            ylabel="拟合折 1、评分折 0 的增益",
            xlim=(-0.32, 0.08),
            ylim=(-0.32, 0.08),
        )
        ax.legend(fontsize=8, loc="lower right")
    b.page(
        "local",
        "EEG 有 11 个配置只在一个方向改善，HbO 斜率在两折都落后",
        "每点同一配置，右上象限才晋级。纵横轴都是较强基线的 NMSE 减预训练读出的 NMSE。两个基线分别使用固定码向量和同构随机网络；三者都使用相同时间箱、位置、128 维投影及仅在拟合人群训练的岭读出。",
        fig,
        "EEG 是 2 秒 Hann 窗内 8–12.5 Hz 占 1–45 Hz 的功率比例；HbO 是 2 秒线性斜率。先平均窗内四位置与时间片，再用拟合折目标方差归一化预测平方误差。EEG 两个拟合折方差分别为 0.01068 和 0.00436，所以每折应对照该折自己的基线。",
        "证据　tables/fold_scores.csv、train/tokens.npz。每折先对记录内窗口平均，再逐被试、逐库平均，最后适用库等权。",
    )


def prediction_page(b, pred):
    fig, axes = graph(1, 4)
    for i, m in enumerate(NAMES):
        subset = pred[pred.target.eq(m)]
        low = subset[["observed", "predicted"]].min().min() - 0.01
        high = subset[["observed", "predicted"]].max().max() + 0.01
        for f in (0, 1):
            ax = axes[0, 2 * i + f]
            g = subset[subset.fit_fold.eq(f)]
            for j, (ds, label) in enumerate(DS.items()):
                values = g[g.dataset.eq(ds)]
                ax.scatter(
                    values.observed,
                    values.predicted,
                    s=7,
                    alpha=0.5,
                    linewidths=0,
                    color=[BLUE, ORANGE, "#835b97"][j],
                    marker=["o", "^", "s"][j],
                    label=label,
                )
            ax.plot([low, high], [low, high], color=DARK, lw=1, ls="--")
            ax.set(
                xlim=(low, high),
                ylim=(low, high),
                aspect="equal",
                xlabel="实测属性",
                title=f"{NAMES[m]} · 拟合折 {f}",
            )
            if f == 0:
                ax.set_ylabel("读出预测值")
            ax.text(0.04, 0.94, f"r = {g.observed.corr(g.predicted):.2f}", transform=ax.transAxes, va="top")
    axes[0, 0].legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.20), fontsize=9)
    b.page(
        "observed",
        "逐窗重放显示，EEG 相对 α 功率较高时读出倾向于低估",
        "每点是一名留出被试的一段 30 秒窗。同一被试贡献多点。横轴为四位置、15 片的实测属性均值，纵轴为终点模型冻结后的岭预测。虚线为预测等于观测。EEG 为功率比例，Hb 为训练尺度归一化的 HbO 斜率／秒。相同模态的两图共用尺度。",
        fig,
        "拟合折 1、评分折 0 时，实测 α 功率比例最高一成窗口平均被低估 0.229；反向评分低估 0.067。HbO 斜率的逐窗相关约 0.78–0.79，同时其 NMSE 仍高于固定码。r 是各窗合并的描述性 Pearson 相关，晋级采用前页按库、按人的配对误差。",
        "证据　tables/held_predictions.csv。重载四个终点检查点，按原拟合折重建冻结读出，并逐窗复核两项属性的已存 NMSE。",
    )


def people_page(b, t, q, people):
    fig, axes = graph(1, 2)
    rows = []
    for ax, m in zip(axes[0], NAMES):
        c = nearest(q, m, "worst_gain")
        for f in (0, 1):
            v = t.query("candidate==@c and target==@m and fold==@f").iloc[0]
            ref = "fixed" if v.fixed <= v.random else "random"
            pp = people.query("candidate==@c and fold==@f").copy()
            pp["delta"] = pp[f"{m}_{ref}_nmse"] - pp[f"{m}_trained_nmse"]
            for i, (ds, lab) in enumerate(DS.items()):
                vals = pp.loc[pp.dataset.eq(ds), "delta"].to_numpy()
                yy = i + (f - 0.5) * 0.31
                col = [BLUE, ORANGE][f]
                ax.scatter(
                    vals,
                    yy + np.linspace(-0.06, 0.06, len(vals)),
                    s=13,
                    marker=["o", "^"][f],
                    alpha=0.78,
                    color=col,
                    label=f"拟合折 {f}" if i == 0 else None,
                )
                ax.scatter(vals.mean(), yy, marker="|", s=175, lw=2.4, color=DARK)
                rows.append(
                    dict(
                        candidate=c,
                        fold=f,
                        target=m,
                        dataset=ds,
                        selected_reference=ref,
                        people=len(vals),
                        positive=int((vals > 0).sum()),
                        mean=vals.mean(),
                        median=np.median(vals),
                    )
                )
        ax.axvline(0, color=GRAY, ls="--")
        ax.set(
            yticks=range(3),
            yticklabels=list(DS.values()),
            xlabel="较强基线 − 训练表示（NMSE）",
            title=NAMES[m] + "　" + display_config(c),
            ylim=(-0.55, 2.55),
        )
        ax.legend(fontsize=8)
    b.page(
        "people",
        "按被试看，EEG 的跨库差异明显，Hb 的改善人数很少",
        "每个点是一名留出被试，短竖线为对应库及对应折的被试均值。各终点展示两折最小增益最大的配置，属于筛选后的描述。每折用全折较强的同一个基线比较所有被试。两图横轴独立缩放。",
        fig,
        "EEG 在两个评分方向分别有 13/24、13/27 人改善；Simultaneous 的人均增益均为负（−0.142、−0.135），REFED 均为正（+0.163、+0.039）。Hb 分别只有 4/24、2/27 人改善，三库两折的均值全部为负。单人及单库差异均保留在工作簿中。",
        "证据　tables/person_scores.csv、tables/selected_people_summary.csv。筛选和描述使用同一批训练人群，比较未作确认性推断。",
    )
    return pd.DataFrame(rows)


def cross(b, t):
    fig, axes = graph(1, 2)
    for ax, m in zip(axes[0], NAMES):
        g = t[t.target.eq(m)]
        for f in [0, 1]:
            a = g[g.fold.eq(f)]
            ax.scatter(
                a.prior_gain,
                a.pair_gain,
                marker=["o", "^"][f],
                facecolor=["none", ORANGE][f],
                edgecolor=[BLUE, ORANGE][f],
                s=44,
                label=f"拟合折 {f}",
            )
        ax.axvline(0, color=GRAY, ls="--")
        ax.axhline(0, color=GRAY, ls="--")
        ax.set(
            xlim=(-1.72, 0.1),
            ylim=(-0.107, 0.055),
            xlabel="词频 − 真输入（bit/词元）",
            ylabel="错输入 − 真输入（bit/词元）",
            title=ROUTE[m],
        )
        ax.legend()
    b.page(
        "cross",
        "整模态预测的共同瓶颈是训练词频基线",
        "每点是一个配置的一折。在评分窗中隐藏目标模态的全部 120 个 ID，S/O 分别评分后等权。纵轴仅替换可见模态试次，供体与目标同人、同记录、同条件，双时钟均至少相隔 60 秒。右上象限表示同时胜过词频与错配输入。",
        fig,
        "Hb→EEG 的 48 个配置折全部落后于词频，差距 0.721–1.648 bit/词元；EEG→Hb 的差距为 0.297–1.104。只看真错输入时，两个拟合折的正增益配置数在 Hb→EEG 为 19/24、0/24，在 EEG→Hb 为 1/24、11/24。真配对优势随人群而变。",
        "证据　tables/fold_scores.csv。词频由拟合折全部局部窗逐流统计，每个词加 0.5 平滑；A 阶段只训练真配对模型。",
    )


def streams(b, t, q, pp):
    fig, axes = graph(1, 2)
    rows = []
    for ax, m in zip(axes[0], NAMES):
        c = nearest(q, m, "cross_margin")
        keys = ["prior", "true", "wrong", "reverse"]
        for f in [0, 1]:
            p = pp.query("candidate==@c and fold==@f")
            for j, s in enumerate("SO"):
                nums = [p.groupby("dataset")[f"{m}{s}_{k}_bits"].mean().dropna().mean() for k in keys]
                ax.plot(
                    range(4),
                    nums,
                    marker=["o", "^"][f],
                    linestyle=["-", "--"][f],
                    color=[BLUE, ORANGE][j],
                    label=f"{s} · 拟合折 {f}",
                )
                rows.append(dict(candidate=c, target=m, stream=s, fold=f, **dict(zip(keys, nums))))
        ax.set(
            xticks=range(4),
            xticklabels=["词频", "真输入", "错输入", "反序输入"],
            ylabel="交叉熵（bit/词元，低为好）",
            title=ROUTE[m] + "　" + display_config(c),
        )
        ax.legend(ncol=2, fontsize=8)
    b.page(
        "streams",
        "分开 S/O 评分后，输入错配与反序的影响仍小于词频差距",
        "每条线对应目标模态的一条词流与一个拟合折。四列共享隐藏目标，真输入、错输入和反序输入均用同一套已训练参数。连线用于对照条件，横轴不表示时间。两个方向分别选择最接近跨模态门槛的配置，纵轴独立。",
        fig,
        "两流等权后，Hb→EEG 的最近配置在两折比词频分别多付 0.827、0.858 bit/词元。EEG→Hb 的最近配置分别多付 0.415、0.349 bit/词元。此处目标是同窗的离散 ID；预测稍后实测 HbO/HbR 的任务原定在候选锁定后进行。",
        "证据　tables/selected_streams.csv，逐人 S/O 误差见 tables/person_scores.csv。反序同时反转可见模态的两条词流。",
    )
    return pd.DataFrame(rows)


def local_baselines(b, t, q):
    fig, axes = graph(1, 2)
    for ax, m in zip(axes[0], NAMES):
        c = nearest(q, m, "worst_gain")
        for f, col in enumerate((BLUE, ORANGE)):
            row = t.query("candidate==@c and target==@m and fold==@f").iloc[0]
            bars = ax.bar(
                np.arange(3) + (f - 0.5) * 0.34,
                [row[k] for k in ["fixed", "random", "trained"]],
                width=0.32,
                color=col,
                label=f"拟合折 {f}",
                hatch=["", "//"][f],
            )
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
        ax.set(
            xticks=range(3),
            xticklabels=["固定码向量", "随机网络", "掩码训练"],
            title=NAMES[m] + "　" + display_config(c),
            ylabel="NMSE（低为好）",
        )
        ax.margins(y=0.18)
        ax.legend()
    b.page(
        "baselines",
        "固定词向量和随机网络已经能读出部分同窗属性",
        "两图分别取各局部属性最接近门槛的配置。每一组三根柱中的同色柱使用相同评价人群和训练尺度。评分时关闭跨模态注意力，仅提供本模态有序 S/O、几何位置与时间坐标。两图使用独立纵轴。",
        fig,
        "EEG 在拟合折 1 的训练表示为 1.829，优于固定码的 2.107，但落后于随机网络的 1.795。Hb 的训练表示为 0.306、0.440，两折都落后于固定码的 0.223、0.278。固定码可读出属性，与掩码训练能否进一步改善读出，是两个分别计分的问题。",
        "证据　tables/fold_scores.csv。α 和 HbO 斜率取自当前 30 秒实测窗口；它们只作为冻结模型的读出目标。",
    )


def local_masks(b, cm):
    fig, axes = graph(1, 4)
    cm = cm.assign(gain=cm.prior - cm.bits, relative_markov=cm.markov - cm.bits)
    agg = cm.groupby(["stream", "train_mask", "mask"]).gain.median()
    limit = max(1.0, abs(agg).max())
    for k, ax in enumerate(axes[0]):
        a = agg.loc[k].unstack().loc[SHAPES, SHAPES]
        im = heat(ax, a.to_numpy(), [LABEL[s] for s in SHAPES], [LABEL[s] for s in SHAPES], limit=limit)
        ax.tick_params(axis="x", rotation=32)
        ax.set(title=STREAM[k], xlabel="评分遮蔽")
        if k == 0:
            ax.set_ylabel("训练局部遮蔽")
        else:
            ax.set_yticklabels([])
    fig.colorbar(im, ax=list(axes[0]), shrink=0.68, label="词频 − 模型（bit/词元）")
    b.page(
        "task_transfer",
        "保留本模态上下文时，局部补全的收益集中在部分词流",
        "模型只看本模态可见上下文。行是训练时的局部遮蔽策略，列是统一评分掩码。每格取同一训练形状下 8 个配置 × 2 折的中位数；每个配置先做四次固定掩码重复，再按记录、被试和库汇总。四图共用色标。",
        fig,
        "合并训练形状，在随机时间片遮蔽上，EEG-S、EEG-O、Hb-S、Hb-O 相对词频的中位增益分别为 0.024、0.321、0.980、0.480 bit/词元。完整模态被遮住时没有这些同模态线索，因此必须单独检验跨模态能力。",
        "证据　各拟合 common_masks.csv、train/tokens.npz；汇总 tables/common_mask_scores.csv。先验和模型严格使用同一隐藏位置。",
    )
    fig, axes = graph(1, 2)
    for ax, metric, lab in zip(
        axes[0], ["conditional", "markov"], ["原生位置和时刻先验", "可见相邻词转移基线"]
    ):
        mat = (
            cm.assign(delta=cm[metric] - cm.bits)
            .groupby(["stream", "mask"])
            .delta.median()
            .unstack()
            .loc[:, SHAPES]
        )
        im = heat(ax, mat.to_numpy(), [LABEL[s] for s in SHAPES], STREAM, limit=4.5)
        ax.set_title(lab + " − 模型")
    fig.colorbar(im, ax=list(axes[0]), shrink=0.72, label="bit / 隐藏词元")
    b.page(
        "neighbor",
        "随机遮住 Hb 时间片时，可见邻词给出了更强预测",
        "这项补充分析使用既有藏词评分和拟合折词频。位置先验按库、原生位置及第几个 2 秒片统计；转移基线使用紧邻且可见的前词或后词，两侧都可见时平均其概率。两个条件分布各以 16 个低一级先验的伪计数平滑。每格为 48 次拟合的中位数。",
        fig,
        "随机时间片任务中，相对邻词基线的中位增益在 Hb-S 为 −1.205、Hb-O 为 −0.691 bit/词元，仅 4/48 和 3/48 次拟合超过该基线。遮住整个位置时没有可见的本站邻词，两张图对应列相等。结果支持短时连续性可被简单模型利用，尚不足以赋予具体序列生理含义。",
        "证据　tables/common_mask_scores.csv；补充基线由 analyze_full_masked_search.py 按拟合人群重新估计，未参与晋级。",
    )


def factors(b, t):
    factors = [
        ("mask", "span", "scatter", "散点 − 连段"),
        ("mask", "span", "site", "整位置 − 连段"),
        ("whole_percent", 25, 50, "50% − 25%"),
        ("width", 128, 256, "256宽／4层 − 128宽／2层"),
        ("head", "linear", "prior", "词频残差 − 线性头"),
    ]
    fig, axes = graph(1, 2)
    rows = []
    for ax, metric, label in zip(
        axes[0], ["gain", "prior_gain"], ["生理读出增益（NMSE）", "整模态相对词频增益（bit/词元）"]
    ):
        for i, (key, low, high, description) in enumerate(factors):
            for j, m in enumerate(NAMES):
                df = t[t.target.eq(m)].pivot(
                    index=["fold"] + [k for k in KEYS if k != key], columns=key, values=metric
                )
                delta = df[high] - df[low]
                mean = delta.mean()
                lo, hi = delta.quantile([0.25, 0.75])
                y = i + (j - 0.5) * 0.23
                ax.plot([lo, hi], [y, y], lw=3, alpha=0.7, color=[BLUE, ORANGE][j])
                ax.scatter(
                    mean,
                    y,
                    color=[BLUE, ORANGE][j],
                    marker=["o", "^"][j],
                    s=35,
                    label=NAMES[m] + " 目标" if i == 0 else None,
                    zorder=3,
                )
                rows.append(
                    dict(
                        contrast=description,
                        metric=metric,
                        target=m,
                        n_pairs=len(delta),
                        mean=mean,
                        q25=lo,
                        q75=hi,
                        positive=int((delta > 0).sum()),
                    )
                )
        ax.axvline(0, color=GRAY, ls="--")
        ax.set(
            yticks=range(5),
            yticklabels=[f[3] for f in factors],
            title=label,
            xlabel="左列所示差值，正值代表增益变大",
        )
        ax.invert_yaxis()
        ax.legend(fontsize=8)
    b.page(
        "factors",
        "加大模型容量降低了整模态预测相对词频的表现",
        "逐次只对比一个配置因素，匹配其余因素和拟合折。点为配对差均值，短线为配对差的第 25–75 百分位；每个容量、比例或输出头比较含 24 对，每个遮蔽形状比较含 16 对。它们是设计网格内的描述性差值。",
        fig,
        "宽度和层数同时由 128／2 增至 256／4 后，整模态增益在 EEG、Hb 目标上平均下降约 0.58、0.51 bit/词元。局部 Hb 读出随容量略有改善，仍未越过强基线。固定步数、单种子下的这些差异不足以预测更大数据或不同预算的结果。",
        "证据　tables/factor_contrasts.csv；均值和四分位区间均基于配对配置，区间表示配置差异的分布。",
    )
    return pd.DataFrame(rows)


def costs(b, op, t):
    spec = t.query("target=='eeg'")[["candidate", "fold", "width"]]
    data = op.merge(spec, validate="one_to_one")
    fig, axes = graph(1, 2)
    for ax, (key, title) in zip(
        axes[0],
        [("train_seconds", "实际训练耗时（秒／拟合）"), ("scoring_seconds", "冻结评分耗时（秒／拟合）")],
    ):
        for j, w in enumerate([128, 256]):
            a = data[data.width.eq(w)][key].sort_values().to_numpy()
            ax.scatter(np.linspace(-0.15, 0.15, len(a)) + j, a, color=[BLUE, ORANGE][j], s=25)
            ax.hlines(np.median(a), j - 0.22, j + 0.22, color=DARK, lw=2)
        ax.set(xticks=[0, 1], xticklabels=["128 宽／2 层", "256 宽／4 层"], ylabel=title, title=title)
        ax.grid(axis="y", alpha=0.16)
    b.page(
        "runtime",
        "48 个终点均通过码本、步数和训练曝光核验",
        "每次拟合固定 1,024 步，批量 8，包含局部连续段、随机片段或整位置遮蔽；严格配对批中有 25% 或 50% 的步骤隐藏整模态，双向等频。左、右图每点是一次拟合或评分，横线为中位数。",
        fig,
        "正式训练共 49,152 步，训练耗时 36.4–46.5 秒／拟合，评分 5.7–8.3 秒／拟合，两卡并行。逐项核对 48 个检查点的有限参数、四个冻结码本、严格供体、逐窗评分和晋级表。保留的进度文件仅有最后 32 步平均损失，无法据此判断早期收敛或过拟合时点。",
        "证据　tables/fit_audit.csv、A/*/*/checkpoint.pt、result.json、progress.json。资源用时分别记录于各模型 result.json。",
    )


def appendix(b, q):
    grouped = q.pivot(index="candidate", columns="target", values=["worst_gain", "cross_margin"])
    names = sorted(
        grouped.index,
        key=lambda s: (
            SHAPES.index(s.split("-")[0]),
            int(s.split("-")[1]),
            int(s.split("-")[2]),
            s.split("-")[3],
        ),
    )
    fig = []
    for these in [names[:12], names[12:]]:
        vals = grouped.loc[
            these,
            [("worst_gain", "eeg"), ("worst_gain", "hb"), ("cross_margin", "eeg"), ("cross_margin", "hb")],
        ].to_numpy()
        header = (
            "<tr>"
            + "".join("<th>" + v + "</th>" for v in ["配置", "EEG 局部", "Hb 局部", "预测 EEG", "预测 Hb"])
            + "</tr>"
        )
        lines = [
            "<tr><td>" + escape(name) + "</td>" + "".join(f"<td>{v:+.3f}</td>" for v in values) + "</tr>"
            for name, values in zip(these, vals)
        ]
        fig.append("<table>" + header + "".join(lines) + "</table>")
    b.page(
        "ledger",
        "附表　24 个配置距离晋级边界有多远",
        "表中是逐配置的预设晋级余量，两折取较小值。局部列单位为 NMSE；预测 EEG 和 Hb 两列单位为 bit/词元，并已同时取词频与错配输入两项增益的较小值。四列都以正值作为晋级条件，独立判读每一列。",
        fig,
        "配置名称依次为训练局部掩码、严格批中的整模态百分比、模型宽度和输出头。span 为连续段，scatter 为随机时间片，site 为整位置。prior 为词频加残差，linear 为普通线性输出头。附表作定位用，所有折的未舍入值、各基线和人级表保存在工作簿。",
        "证据　tables/candidate_margins.csv、tables/fold_scores.csv。每列均无正余量，两条路线的实际晋级数均为零。",
    )


def verify_export(out, cover, titles):
    import xml.etree.ElementTree as ET
    from zipfile import ZipFile

    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(out / "evidence.xlsx") as workbook:
        assert workbook.testzip() is None
        names = [
            item.attrib["name"]
            for item in ET.fromstring(workbook.read("xl/workbook.xml")).findall(".//s:sheet", ns)
        ]
        assert names == ["说明", *cover.sheet]
        for j, name in enumerate(names[1:], 2):
            sheet = ET.fromstring(workbook.read(f"xl/worksheets/sheet{j}.xml"))
            assert (
                len(sheet.findall(".//s:row", ns)) == len(pd.read_csv(out / "tables" / (name + ".csv"))) + 1
            )
    pages = []
    with fitz.open(out / "REPORT.pdf") as doc:
        assert len(doc) == len(titles) == len(doc.get_toc())
        for i, page in enumerate(doc):
            images = page.get_image_info()
            assert len(page.get_text()) > 150
            if i == len(titles) - 1:
                assert not images and "span-25-128-linear" in page.get_text()
                dpi = None
            else:
                assert len(images) == 1
                im = images[0]
                rect = fitz.Rect(im["bbox"])
                dpi = min(72 * im["width"] / rect.width, 72 * im["height"] / rect.height)
                assert 200 <= dpi <= 300 and page.rect.contains(rect)
            for block in page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]:
                assert page.rect.contains(fitz.Rect(block["bbox"]))
            pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5), alpha=False)
            assert np.asarray(pix.samples_mv).std() > 10
            pages.append(dict(page=i + 1, title=titles[i], bitmap_dpi=round(dpi, 1) if dpi else None))
    (out / "verification.json").write_text(
        json.dumps(
            dict(
                fits_checked=48,
                optimizer_steps=49152,
                train_windows=2827,
                strict_windows=156,
                local_candidates_promoted=0,
                cross_candidates_promoted=0,
                development_evaluated=False,
                probe_replayed_windows_per_target=2827,
                workbook_sheets=len(cover) + 1,
                pages=pages,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run = args.run
    out = args.out
    td = out / "tables"
    t, cm, pp, op = (
        pd.read_csv(td / (k + ".csv"))
        for k in ("fold_scores", "common_mask_scores", "person_scores", "fit_audit")
    )
    p = pd.read_csv(run / "train/panel.csv")
    b = Book(out)
    q = outcomes(t)
    overview(b, t, q)
    support = coverage(b, p, op)
    local(b, t, q)
    local_baselines(b, t, q)
    prediction_page(b, pd.read_csv(td / "held_predictions.csv"))
    selected = people_page(b, t, q, pp)
    cross(b, t)
    streams_table = streams(b, t, q, pp)
    local_masks(b, cm)
    contrasts = factors(b, t)
    costs(b, op, t)
    appendix(b, q)
    b.finish()
    extras = dict(
        candidate_margins=q,
        selected_people_summary=selected,
        selected_streams=streams_table,
        factor_contrasts=contrasts,
        fold_support=support,
    )
    for name, table in extras.items():
        table.to_csv(td / (name + ".csv"), index=False)
    cover = pd.DataFrame(
        [
            ("fit_audit", "每次拟合的终点与曝光覆盖；strict 是同条件合法错配支持。"),
            ("fold_support", "每库、每折的 30 秒窗数、被试数、严格配对窗数。"),
            ("candidate_margins", "每配置、预测目标的两折最小增益；cross_margin 还取两个对照的较小值。"),
            (
                "fold_scores",
                "每配置、拟合折和目标的局部属性 NMSE 与整模态 S/O 平均交叉熵（bit/词元）。正增益代表改善。",
            ),
            (
                "held_predictions",
                "每个局部终点最接近门槛的模型，按训练折冻结读出后，对全部留出窗口重放的实测值与预测值。",
            ),
            (
                "person_scores",
                "每配置、拟合折的留出被试均值；以记录等权。跨模态空值表示该被试无严格供体支持。",
            ),
            (
                "common_mask_scores",
                "每配置、折、评估掩码、目标流的交叉熵（bit/词元）；四次重复，库等权。stream 0/1/2/3 对应 EEG-S/O、Hb-S/O。",
            ),
            ("selected_people_summary", "每个局部终点最接近晋级的配置；每库的逐人改善数、均值和中位数。"),
            ("selected_streams", "每方向最接近跨模态门槛配置的逐流交叉熵（bit/词元）。"),
            ("factor_contrasts", "每次改变一个配置因素，保持其他因素、目标及拟合折相同。差为增益之差。"),
        ],
        columns=["sheet", "说明"],
    )
    with pd.ExcelWriter(out / "evidence.xlsx", engine="xlsxwriter") as x:
        for name, tab in [("说明", cover)] + [(n, pd.read_csv(td / (n + ".csv"))) for n in cover.sheet]:
            tab.to_excel(x, sheet_name=name, index=False)
            ws = x.sheets[name]
            ws.freeze_panes(1, 1)
            ws.autofilter(0, 0, len(tab), len(tab.columns) - 1)
            for j, col in enumerate(tab):
                ws.set_column(j, j, min(90, max(len(col) + 3, tab[col].astype(str).str.len().max() + 2)))
    verify_export(out, cover, b.titles)


if __name__ == "__main__":
    main()
