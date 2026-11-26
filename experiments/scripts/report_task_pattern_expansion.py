"""Bitmap report of real task blocks, physiology, frozen word patterns and control comparisons."""

import argparse
from html import escape
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from run_task_pattern_expansion import CFG, OUT, read
from src.data.task_pattern_panel import load_task_panel
from src.analysis.task_pattern_expansion import ACTIVE, TAIL, PRE
from src.utils.io import write_json

TASKS = ["wg", "mental_arithmetic", "motor_imagery"]
CN = {"wg": "词生成", "mental_arithmetic": "心算", "motor_imagery": "运动想象"}
CONDS = {"wg": ["基线", "词生成"], "mental_arithmetic": ["基线", "心算"], "motor_imagery": ["左手", "右手"]}
COL = ["#13796e", "#bd7138", "#5c6298"]
MAP = {
    "frontal_theta": "额 θ",
    "posterior_alpha": "顶 α",
    "theta_minus_alpha": "额 θ 减顶 α",
    "frontal_alpha": "额 α",
    "motor_alpha_lr": "运动区 α 左减右",
    "motor_beta_lr": "运动区 β 左减右",
    "posterior_alpha_rebound": "顶 α 晚减早",
    "motor_beta_rebound_lr": "运动区 β 晚减早的左右差",
    "frontal_HbO": "额 HbO",
    "frontal_HbR": "额 HbR",
    "motor_HbO_lr": "运动区 HbO 左减右",
    "motor_HbR_lr": "运动区 HbR 左减右",
    "frontal_HbO_minus_HbR": "额 HbO 减 HbR",
    "frontal_HbO_late_minus_early": "额 HbO 晚减早",
    "frontal_HbR_late_minus_early": "额 HbR 晚减早",
}
GROUP = {"EEG": "实测 EEG", "Hb": "实测 Hb", "EEGword": "EEG 词元", "Hbword": "Hb 词元"}
plt.rcParams.update(
    {
        "font.family": "Noto Sans CJK JP",
        "font.size": 10.5,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def title(name):
    if name in MAP:
        return MAP[name]
    b, region, *end = name.split("_")
    reg = {"frontal": "额部", "posterior": "顶部", "motor": "运动区"}[region]
    return b + " " + reg + " Q4" + (" 晚减早" if "timing" in end else " 左减右" if "lr" in end else "")


def plot(n, m, **kw):
    return plt.subplots(n, m, figsize=(13.0, 5.55), layout="constrained", **kw)


def heat(ax, mat, fmt=".2f", vmin=0.3, vmax=0.75, cmap="PuBuGn", tags=None):
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    for i, j in np.ndindex(mat.shape):
        v = mat[i, j]
        lum = np.dot(im.cmap(im.norm(v))[:3], [0.299, 0.587, 0.114])
        ax.text(
            j,
            i,
            format(v, fmt) + (tags[i, j] if tags is not None else ""),
            fontsize=9,
            ha="center",
            va="center",
            color="white" if lum < 0.46 else "#162d34",
        )
    return im


class Report:
    def __init__(self, path):
        self.path = path
        self.doc = fitz.open()
        self.titles = []
        if (path / "REPORT.pdf").exists():
            raise FileExistsError("Choose a fresh versioned report folder")
        path.mkdir(exist_ok=True, parents=True)

    def page(self, slug, heading, lead, fig, caption, source):
        self.titles.append(heading)
        p = self.doc.new_page(width=1000, height=720)

        def write(rect, content, size, color="#1d3740"):
            h, _ = p.insert_htmlbox(
                fitz.Rect(*rect),
                escape(content),
                css=f"*{{font-family:sans-serif;font-size:{size}pt;color:{color};line-height:1.34}}",
                scale_low=1,
            )
            if h < 0:
                raise ValueError("Text overflow " + heading)

        write(
            (28, 13, 975, 38),
            f"跨任务生理模式筛选　　2026 年 11 月 27 日　　　　　　{len(self.titles):02}",
            11,
            "#186d67",
        )
        write((28, 40, 978, 84), heading, 23)
        write((28, 89, 974, 151), lead, 13)
        file = self.path / (slug + ".png")
        fig.savefig(file, dpi=250, facecolor="white")
        plt.close(fig)
        p.insert_image(fitz.Rect(27, 151, 973, 555), filename=str(file), keep_proportion=True)
        write((28, 565, 975, 675), caption, 12)
        write((28, 682, 975, 712), source, 9, "#266f6b")

    def finish(self):
        self.doc.set_toc([[1, t, i] for i, t in enumerate(self.titles, 1)])
        self.doc.subset_fonts()
        p = self.path / "REPORT.pdf"
        self.doc.save(p, garbage=4, deflate=True)
        self.doc.close()
        stats = []
        with fitz.open(p) as doc:
            for i, page in enumerate(doc):
                (im,) = page.get_image_info()
                dpi = im["width"] * 72 / fitz.Rect(im["bbox"]).width
                if not 200 <= dpi <= 300 or len(page.get_text()) < 180:
                    raise ValueError("PDF figure/text check")
                stats.append(dict(page=i + 1, images=1, dpi=dpi, text_chars=len(page.get_text())))
        write_json(self.path / "verification.json", dict(status="passed", pages=stats))


def signatures(rep, loc):
    fig, ax = plot(1, 2, gridspec_kw={"width_ratios": [1, 1.25]})
    g = loc.query("scope=='all' and name=='beta_lr'")
    for j, t in enumerate(TASKS):
        a = g[g.task == t].sort_values("subject")
        off = np.linspace(-0.17, 0.17, len(a))
        ax[0].scatter(a.raw_auc, j + off, s=49, c=COL[j], marker=["o", "s", "^"][j])
        ax[0].scatter(a.raw_auc.mean(), j, s=115, facecolors="none", edgecolors="black", marker="D", lw=1.5)
    ax[0].set_yticks(range(3), [CN[t] for t in TASKS])
    ax[0].invert_yaxis()
    ax[0].axvline(0.5, color="gray", ls="--")
    ax[0].set_xlim(0.28, 0.87)
    ax[0].set_xlabel("晚段相对早段的左右 β 功率差 · AUC")
    ax[0].set_title("每点一人，空心菱形为被试均值")
    nn = ["beta_left", "beta_right", "beta_lr", "beta_mean", "alpha_lr", "beta_lr_pre"]
    labels = [
        "左 β 时段差",
        "右 β 时段差",
        "左减右 β 时段差",
        "双侧平均 β 时段差",
        "左减右 α 时段差",
        "任务前左减右 β",
    ]
    data = loc[loc.scope == "all"].groupby(["name", "task"]).raw_auc.mean().unstack().loc[nn, TASKS]
    heat(ax[1], data.to_numpy(), vmin=0.35, vmax=0.8)
    ax[1].set_yticks(range(6), labels)
    ax[1].set_xticks(range(3), [CN[t] for t in TASKS])
    ax[1].set_title("同一批事件上拆开左、右及频带")
    rep.page(
        "01_beta",
        "首选复核左右 β 功率随时间的变化",
        "词生成训练人群选出的方向，在词生成开发集得到 AUC 0.623（5/5 人同向）；冻结这个方向后，心算为 0.686（3/4）。两套数据使用不同的人、任务和电极。运动想象的同一指标为 0.542。",
        fig,
        "每块先求 13–30 Hz 对数功率在 17–25 秒相对 1–9 秒的变化，再以左减右。正向 AUC 是同一人、同一记录里任务块高于对照块的概率，平局计半；记录和被试逐层等权。运动想象比较右手与左手。词生成比较 C3/C4，心算与运动想象比较 FCC3h/FCC4h。心算仅左侧达到 0.765，左右差的不可替代性仍待验证。",
        "依据　transfer.csv · localization/people.csv · 原始任务标记为时间零点。位点表示头皮传感器，不对应经配准的皮层源。",
    )


def beta_trials(rep, data, loc):
    fig, axs = plot(2, 2, sharex=True)
    provenance = []
    for j, t in enumerate(TASKS[:2]):
        m, a = data[t]
        lf = pd.read_csv(OUT / "localization/development_features.csv").query("task==@t").set_index("id")
        # First subject/record with both classes; nearest class median within that record; ties by event.
        eligible = m[m.split == "validation"]
        group = next(g for _, g in eligible.groupby(["subject", "record"]) if g.label.nunique() == 2)
        for k in (0, 1):
            g = group[group.label == k]
            vals = lf.loc[g.id, "beta_lr"].to_numpy()
            i = g.index[np.argmin(abs(vals - np.median(vals)))]
            row = m.loc[i]
            ax = axs[k, j]
            db = a["e"][i, 4:6, :, 2]
            db = db - db[:, PRE].mean(1, keepdims=True)
            for s, (ls, c) in enumerate([("-", "#13796e"), ("--", "#a95e38")]):
                ax.plot(
                    np.arange(15) * 2 - 4,
                    db[s],
                    ls=ls,
                    marker=["o", "s"][s],
                    ms=3,
                    color=c,
                    label=CFG["sites"][row.dataset]["eeg"][4 + s],
                )
            ax.axvspan(1, 9, color="#c7ddd7", alpha=0.32)
            ax.axvspan(17, 25, color="#e9c9ac", alpha=0.29)
            ax.axvline(0, color="gray", ls=":")
            ax.set_title(
                f"{CN[t]} · {CONDS[t][k]}　{row.subject} / {row.record} / {row.event}", fontsize=10
            )
            ax.set_ylabel("β 相对任务前（dB）")
            ax.set_xlabel("相对标记（秒）")
            ax.legend(ncol=2, fontsize=8, loc="lower left")
            ax.text(
                0.98,
                0.94,
                f"左右时段差 {lf.loc[row.id,'beta_lr']:+.2f} dB",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )
            provenance.append(dict(task=t, id=row.id, role="beta example"))
    low = min(ax.get_ylim()[0] for ax in axs.flat)
    hi = max(ax.get_ylim()[1] for ax in axs.flat)
    for ax in axs.flat:
        ax.set_ylim(low, hi)
    rep.page(
        "02_real_beta",
        "逐块看，词生成与心算的左右差由不同波形构成",
        "开发被试等权平均后，词生成相对基线的左、右时段变化分别多 +0.12、−0.26 dB；心算分别多 +1.36、+0.60 dB。相同的左右差方向，可以来自右侧减少，也可以来自左侧增加更多。",
        fig,
        "每个面板是一条实测试次。每个点对应两秒 EEG 功率，实线为左、虚线为右；绿带和黄带是计算时段差的早段和晚段。为展示真实波形，按编号取首个有两类条件的开发被试与记录，各取最接近该记录本类中位数的块。四幅图共用纵轴。按上一试次条件分层后，词生成、心算 AUC 分别为 0.630、0.712；样例和组均值都不足以确定神经来源。",
        "依据　localization/development_features.csv · record_controls.csv · records/ 的 200 Hz EEG 所提取的两秒频带功率。",
    )
    return provenance


def word_panel(rep, data):
    t = "mental_arithmetic"
    m, a = data[t]
    g = next(
        g for _, g in m[m.split.eq("validation")].groupby(["subject", "record"]) if g.label.nunique() == 2
    )
    lf = pd.read_csv(OUT / "localization/development_features.csv").query("task==@t").set_index("id")
    fig, axs = plot(2, 2, gridspec_kw={"height_ratios": [1.7, 1]})
    waves = []
    out = []
    for j in (0, 1):
        gc = g[g.label.eq(j)]
        vals = lf.loc[gc.id, "Hb_word_timing"].to_numpy()
        i = gc.index[np.argmin(abs(vals - np.median(vals)))]
        row = m.loc[i]
        out.append(dict(task=t, id=row.id, role="Hb example"))
        h = a["h"][i, :2].mean(0) * 1000
        waves.append(h)
        for k, (c, ls) in enumerate([("#ac553f", "-"), ("#305f91", "--")]):
            axs[0, j].plot(np.arange(300) / 10 - 5, h[:, k], c=c, ls=ls, label=["HbO", "HbR"][k])
        axs[0, j].legend(ncol=2, fontsize=8)
        axs[0, j].set_ylabel("相对 Hb（缓存单位 ×1000）")
        axs[0, j].set_title(f"{CONDS[t][j]} · {row.subject} / {row.record} / {row.event}", fontsize=11)
        ax = axs[1, j]
        q = a["hf"][i, :2, :, 0] + 1
        ax.imshow(
            q,
            extent=(-5, 25, 1.5, -0.5),
            vmin=0.5,
            vmax=4.5,
            cmap=ListedColormap(["#e2ebe9", "#a3c7c1", "#579187", "#205956"]),
            aspect="auto",
        )
        for u, v in np.ndindex(q.shape):
            ax.text(
                -4 + 2 * v,
                u,
                str(q[u, v]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if q[u, v] >= 3 else "#333333",
            )
        ax.set_yticks([0, 1], ["左额", "右额"])
        ax.set_xlabel("相对标记（秒）")
        ax.set_title(
            f"Hb-S 状态族 · Q4 晚减早 {vals[np.argmin(abs(vals-np.median(vals)))]:+.2f}", fontsize=10
        )
        for top in axs[:, j]:
            for x in (1, 9, 17):
                top.axvline(x, color="#8c7772", lw=0.9, ls=":")
            top.set_xlim(-5, 25)
    ll = min(h.min() for h in waves)
    hh = max(h.max() for h in waves)
    for ax in axs[0]:
        ax.set_ylim(ll - (hh - ll) * 0.06, hh + (hh - ll) * 0.06)
    rep.page(
        "03_hb_words",
        "值得保留的词元线索是额部 Hb-S 高斜率族出现的时段",
        "心算相对基线，额部 Q4 占用率的“晚段减早段”平均多 13.8 个百分点，AUC 0.629，3/4 人同向。同窗实测 HbO 斜率构成的阈值规则也达到 0.605。",
        fig,
        "上排为同一事件的实测双侧额部 Hb 均值，均已减去任务前五秒均值；Single-Trial 采用近似换算的相对 Hb 单位。下排每格是 2 秒的 Hb-S 词族，1–4 为训练词典按 HbO 斜率划分的四档。Q4 比较 9–17 秒与 1–9 秒的占用比例，整窗编码可见前后文。各类样例取首个合格开发记录中最接近本类指标中位数的块。一条词族标签不代表一个独立生理状态。",
        "依据　mental_arithmetic/pattern_people.csv · localization/people.csv · hb/ 的冻结词元 · 额部 AF3Fp1 与 AF4Fp2。",
    )
    return out


def bank(rep):
    fig, axs = plot(1, 3, gridspec_kw={"width_ratios": [1, 1, 1.15]})
    tabs = [pd.read_csv(OUT / t / "pattern_selection.csv").set_index(["family", "name"]) for t in TASKS]
    for ax, groups, sub in zip(
        axs, [["EEG"], ["Hb"], ["EEGword", "Hbword"]], ["连续 EEG", "连续 Hb", "冻结词元"]
    ):
        cells = [k for k in tabs[0].index if k[0] in groups]
        cells = sorted(cells)
        mat = np.array([[tab.loc[k, "development"] for tab in tabs] for k in cells])
        tags = np.array(
            [
                ["+" if tab.loc[k, "passes"] else "*" if tab.loc[k, "selected"] else "" for tab in tabs]
                for k in cells
            ]
        )
        heat(ax, mat, vmin=0.30, vmax=0.70, tags=tags)
        ax.set_yticks(
            range(len(cells)),
            [
                ("电 " if k[0] == "EEGword" else "血 " if k[0] == "Hbword" else "") + title(k[1])
                for k in cells
            ],
            fontsize=8,
        )
        ax.set_xticks(range(3), [CN[t] for t in TASKS], fontsize=9)
        ax.set_title(sub)
    rep.page(
        "04_pattern_bank",
        "81 项任务内比较，12 项入围，4 项过开发筛选",
        "每项任务有 27 个预定候选。四项通过者是词生成的 β 左右时段差、额 HbO，以及心算的运动区 HbO 左右差、额部 Hb-S 的 Q4 时段差。运动想象本轮没有通过者。",
        fig,
        "每格为按该任务训练集冻结方向后的开发 AUC。＋表示达到继续门槛；* 表示训练入围但开发未过；无标记为完整候选库的诊断值。每类最多入围两项，训练 AUC ≥0.55 且至少 2/3 人同向；开发 AUC ≥0.58、至少 75% 人同向、五个相邻块号内 AUC ≥0.55，且总体与分层估计各至少三人。方向可能因任务而异。对每任务的入围项做被试符号翻转最大统计量校正，四项通过者的单侧参考 p 值为 0.125–0.50。",
        "依据　各任务 pattern_selection.csv、pattern_people.csv　｜　θ 4–8 Hz，α 8–13 Hz，β 13–30 Hz；单格不代表显著性。",
    )


def blood(rep, data):
    fig, axs = plot(1, 3, gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for j, (task, sites, side) in enumerate([("wg", [0, 1], False), ("mental_arithmetic", [2, 3], True)]):
        m, a = data[task]
        g = m[m.split == "validation"]
        v = a["h"][..., 0][:, sites, :]
        attr = (v[:, 0] - v[:, 1]) * 1000 if side else v.mean(1)
        person = []
        for _, p in g.groupby("subject"):
            rec = []
            for _, q in p.groupby("record"):
                if q.label.nunique() == 2:
                    rec.append(attr[q.index[q.label == 1]].mean(0) - attr[q.index[q.label == 0]].mean(0))
            if rec:
                person.append(np.mean(rec, 0))
        ax = axs[j]
        time = np.arange(300) / 10 - 5
        for k, v in enumerate(person):
            ax.plot(time, v, lw=0.9, alpha=0.7, label=None)
        ax.plot(time, np.mean(person, 0), color="#193c43", lw=2.8, label="被试均值")
        ax.axvspan(9, 17, color="#e5c2a0", alpha=0.33)
        ax.axhline(0, color="gray", ls=":")
        ax.set_xlim(-5, 25)
        ax.set_title(["词生成 · 额部 HbO", "心算 · 运动区 HbO 左减右"][j])
        ax.set_xlabel("相对标记（秒）")
        ax.set_ylabel(["任务减基线（μM）", "任务减基线（相对 Hb ×1000）"][j])
        ax.legend(fontsize=8)
    p = pd.read_csv(OUT / "localization/model_people.csv").groupby(["task", "model"]).ba.mean().unstack()
    keys = ["Hb", "Hb_plus_slope", "Hb_plus_word"]
    heat(axs[2], p.loc[TASKS, keys].to_numpy() * 100, ".1f", vmin=43, vmax=68)
    axs[2].set_yticks(range(3), [CN[t] for t in TASKS])
    axs[2].set_xticks(range(3), ["Hb", "加实测斜率", "加词族时段差"], rotation=14)
    axs[2].set_title("小读出 · 平衡准确率（%）")
    rep.page(
        "05_blood",
        "血流方向随任务和位置改变，词族增益也因人而异",
        "词生成 9–17 秒的额部 HbO 比基线低 0.576 μM（4/5 人排序同向）；心算的运动区 HbO 左减右比基线高 0.00117 个相对 Hb 单位（3/4）。词生成的额部 HbO 方向迁移到心算时，AUC 降至 0.345。",
        fig,
        "细线是每个人的同记录条件差，粗线为被试等权均值，色带是评分时段。两图量纲不同。右表为追加诊断，给同一连续 Hb 读出分别加一项 Q4 词族时段差或实测 HbO 高斜率时段差，后者以训练期各额部位点的两秒斜率上四分位为阈值。词生成加入词族后从 55.1% 到 60.5%，3/5 人提高，仅 2/5 人胜过加入实测斜率。心算为 63.1% 到 64.6%，2/4 人提高。",
        "依据　各任务 pattern_people.csv · localization/model_people.csv、slope_thresholds.csv　｜　实测 Hb 系统及单位按数据集分别保留。",
    )


def readouts(rep):
    keys = [
        "ordinal",
        "pre_only",
        "EEG",
        "Hb",
        "EH",
        "E_family",
        "H_family",
        "families",
        "ordered",
        "hybrid",
    ]
    zh = [
        "块号查表",
        "任务前信息",
        "连续 EEG",
        "连续 Hb",
        "连续联合",
        "EEG 词族占用",
        "Hb 词族占用",
        "四路词族占用",
        "占用＋转移",
        "连续联合＋占用",
    ]
    fig, axs = plot(1, 2, gridspec_kw={"width_ratios": [1.25, 1]})
    ds = read(OUT / "summary.json")["tasks"]
    heat(axs[0], np.array([[ds[t]["BA"][k] * 100 for t in TASKS] for k in keys]), ".1f", vmin=42, vmax=76)
    axs[0].set_title("三种任务的被试等权平衡准确率（%）")
    axs[0].set_yticks(range(len(keys)), zh)
    axs[0].set_xticks(range(3), [CN[t] for t in TASKS])
    for j, t in enumerate(TASKS):
        p = pd.read_csv(OUT / t / "model_people.csv").pivot(index="subject", columns="model", values="ba")
        dif = 100 * (p.hybrid - p.EH)
        axs[1].scatter(
            dif, np.linspace(j - 0.17, j + 0.17, len(dif)), c=COL[j], marker=["o", "s", "^"][j], s=48
        )
        axs[1].scatter(dif.mean(), j, s=105, facecolors="none", edgecolors="black", marker="D")
    axs[1].axvline(0, ls="--", c="gray")
    axs[1].set_yticks(range(3), [CN[t] for t in TASKS])
    axs[1].invert_yaxis()
    axs[1].set_title("加入词族后，哪些人改善？")
    axs[1].set_xlabel("混合读出减连续联合（百分点）")
    rep.page(
        "06_readouts",
        "心算读出最强；全部词元加入后，三项任务都未过增益门槛",
        "相同时间支持下，心算连续联合读出为 75.4%，四路词族读出为 57.4%。运动想象的混合读出均值虽由 45.8% 到 56.7%，改善人数不足，块号查表已达 60.0%。",
        fig,
        "每个信号模型用相同训练人群的三折分组交叉验证选取岭系数。连续输入含六个 EEG 位置的三频带或四个 Hb 位置的双色时段均值；词元比较 S/O 四档状态族占用及相邻转移。主要时段为 −5 至 −1、1–9、9–17、17–25 秒。词元编码和连续缓存均为离线输入。平衡准确率先平均个人内两类召回率，再平均被试；右图每点一人、菱形为均值。预设增益需至少 2 个百分点、至少 75% 的人改善。",
        "依据　各任务 predictions.csv.gz、model_people.csv · summary.json　｜　未知块序查表采用训练集多数类，各任务均为二分类。",
    )


def pairs(rep, data):
    fig, axs = plot(1, 3, gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    tab = pd.read_csv(OUT / "event_pairs/contrasts.csv")
    pairs = ["theta_HbO", "alpha_HbO", "theta_HbR", "motor_beta_HbO", "word_alpha_HbO"]
    zh = [
        "额 θ → 额 HbO",
        "顶 α 下降 → 额 HbO",
        "额 θ → 额 HbR 下降",
        "β 左右差 → HbO 左右差",
        "顶 S-Q4 下降 → 额 HbO",
    ]
    for j, (metric, tit) in enumerate([("dev_effect", "同一试次"), ("wrong", "错配试次")]):
        mat = 100 * tab.pivot(index="pair", columns="task", values=metric).loc[pairs, TASKS].to_numpy()
        heat(axs[j], mat, "+.0f", vmin=-20, vmax=20, cmap="RdBu_r")
        axs[j].set_xticks(range(3), [CN[t] for t in TASKS])
        axs[j].set_yticks(range(5), zh if j == 0 else [])
        axs[j].set_title(tit + " · 概率差（百分点）")
    ds = read(OUT / "summary.json")["tasks"]
    mat = np.array([[ds[t]["coupling_nmse"][k] for k in ("H", "HE", "wrong")] for t in TASKS])
    heat(axs[2], mat, vmin=0.2, vmax=2.7, cmap="YlOrBr")
    axs[2].set_yticks(range(3), [CN[t] for t in TASKS])
    axs[2].set_xticks(range(3), ["早段 Hb", "加 EEG", "加错配 EEG"], rotation=15)
    axs[2].set_title("预测晚段 Hb · NMSE")
    rep.page(
        "07_pairing",
        "15 个早期 EEG 事件与晚段 Hb 响应配对均未通过筛选",
        "例如心算中，早段额 θ 较高的试次出现较大额 HbO 增量的频率为 41.3%，其余试次为 57.1%。开发差为 −15.8 个百分点，训练差为 −9.2 个百分点，未过预设的 10 个百分点训练门槛。",
        fig,
        "左中图是在同一人、记录和条件下，有早段事件减无事件的晚段响应概率。阈值为相应有符号指标的训练中位数，每组至少各两块；对照轮换同条件且相隔至少 30 秒的试次。右图按训练标准差归一化晚段减早段 Hb 的预测平方误差，越低越好，三种模型均知任务条件。词生成真实 EEG 优于单用 Hb，却落后于错配 EEG。整窗离线处理限制了方向性与因果解释。",
        "依据　event_pairs/record_condition.csv、thresholds.csv、support.csv · 各任务 coupling_people.csv；不同配对的开发支持为 3–5 人。",
    )


def transfer_page(rep):
    df = pd.read_csv(OUT / "transfer.csv")
    fig, axs = plot(1, 2)
    for ax, source in zip(axs, ["wg", "mental_arithmetic"]):
        g = df[df.source == source].copy().sort_values("target_dev_auc", ascending=False)
        mat = g[["source_dev_auc", "target_dev_auc", "target_stratified"]].to_numpy()
        heat(ax, mat, vmin=0.3, vmax=0.7)
        ax.set_yticks(
            range(len(g)),
            [
                ("电 " if v.family == "EEGword" else "血 " if v.family == "Hbword" else "") + title(v.name)
                for v in g.itertuples()
            ],
            fontsize=9,
        )
        ax.set_xticks(range(3), ["来源开发", "目标开发", "目标块序分层"])
        ax.set_title(CN[source] + "冻结方向 → " + CN[g.target.iloc[0]])
    rep.page(
        "08_transfer",
        "跨任务迁移的九个方向中，只有 β 左右时段差两侧都过门槛",
        "词生成训练中另一个候选“顶部 α 晚减早”迁到心算后 AUC 为 0.650，但它在来源开发集仅为 0.498，因此不计为跨任务复现。上轮 n-back 的“额 θ 升、顶 α 降”固定组合在心算、词生成分别为 0.505、0.468。",
        fig,
        "每行只在来源训练人群中选择候选和正负方向。保持方向后，在两套数据的独立开发分区分别计算同记录 AUC。最后一列把比较限制到五个相邻原始事件序号的小组。迁移的是单一指标的排序方向，无目标数据重拟合，也不搬用两台设备的 Hb 数值阈值。一项跨任务通过还要求来源通过且目标 AUC ≥0.58、至少 75% 同向、分层 AUC ≥0.55。此处按同一公开开发集开展探索。",
        "依据　transfer.csv · 两任务 pattern_people.csv　｜　原 n-back 经验组合只作预定外推检查。",
    )


def support_page(rep):
    ledger = pd.read_csv(OUT / "support.csv")
    fig, axs = plot(1, 2, gridspec_kw={"width_ratios": [1, 1.15]})
    for j, split in enumerate(["train", "validation"]):
        planned = np.array([len(ledger.query("task==@t and split==@split")) for t in TASKS])
        used = np.array(
            [len(ledger.query("task==@t and split==@split and status=='complete'")) for t in TASKS]
        )
        pos = np.arange(3) + (j - 0.5) * 0.34
        axs[0].bar(pos, planned, 0.32, color="#e2e6e4")
        bars = axs[0].bar(pos, used, 0.32, color=["#277a78", "#ba753d"][j], label=["训练", "开发"][j])
        axs[0].bar_label(
            bars, [str(i) + "/" + str(n) for i, n in zip(used, planned)], fontsize=8, padding=2
        )
    axs[0].set_title("保留试次 / 缓存计划试次")
    axs[0].set_ylabel("事件数")
    axs[0].set_xticks(range(3), [CN[t] for t in TASKS])
    axs[0].legend()
    axs[0].set_ylim(0, 1180)
    for j, t in enumerate(TASKS):
        v = (
            ledger.query("task==@t and split=='validation' and status=='complete'")
            .groupby("subject")
            .size()
        )
        axs[1].scatter(v, np.linspace(j - 0.2, j + 0.2, len(v)), c=COL[j], s=38, marker=["o", "s", "^"][j])
        for offs, (sub, n) in zip(np.linspace(j - 0.2, j + 0.2, len(v)), v.items()):
            if n < 18:
                axs[1].annotate(sub, (n, offs), xytext=(5, 2), textcoords="offset points", fontsize=8)
    axs[1].set_yticks(range(3), [CN[t] for t in TASKS])
    axs[1].invert_yaxis()
    axs[1].set_xlim(0, 68)
    axs[1].set_xlabel("每个开发被试保留的试次")
    axs[1].set_title("少量试次的被试也保留在分母内")
    rep.page(
        "09_scope",
        "后续优先复核 β 的时空变化，其次验证单个 Hb 词族是否有增量",
        "词生成和心算优先做任务顺序平衡、监测肌电或动作的复核，分别检查左右站点。模型方向优先验证额部 Hb-S 的出现时段，并与同维度实测斜率读出配对。运动想象、四路词元整体增益和跨模态事件预测本轮暂缓。",
        fig,
        "两套数据共 3,880 个公开训练/开发计划窗口，中央质量筛选及完整性检查后保留 2,777 个，其中开发 615 个。另有 80 个词生成事件在上游因时钟支持不完整未进窗口计划。心算和运动想象用相同四名开发被试，词生成五名。心算剔除任一类少于五次的被试后，β 指标 AUC 为 0.651（三人）。数据不含本次验证所需的独立动作或皮层真值。",
        "依据　plan.csv、upstream_exclusions.csv、support.csv · localization/people.csv · summary.json；所有门槛均为探索性优先级标准。",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=OUT / "report_v1")
    args = p.parse_args()
    rep = Report(args.output)
    data = {}
    for t in TASKS:
        _, m, a = load_task_panel(OUT, t)
        data[t] = (m, a)
    loc = pd.read_csv(OUT / "localization/people.csv")
    signatures(rep, loc)
    ex = beta_trials(rep, data, loc)
    ex += word_panel(rep, data)
    bank(rep)
    blood(rep, data)
    readouts(rep)
    pairs(rep, data)
    transfer_page(rep)
    support_page(rep)
    catalog = []
    for t in TASKS:
        tab = pd.read_csv(OUT / t / "pattern_selection.csv")
        tab.insert(0, "任务", CN[t])
        tab.insert(1, "类别", tab.family.map(GROUP))
        tab.insert(2, "模式", tab.name.map(title))
        tab["计量单位"] = [
            (
                "dB"
                if f == "EEG"
                else CFG["sites"][CFG["tasks"][t]["dataset"]]["hb_unit"] if f == "Hb" else "占用比例差"
            )
            for f in tab.family
        ]
        catalog.append(tab)
    pd.concat(catalog, ignore_index=True).to_csv(
        rep.path / "pattern_catalog.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(ex).to_csv(rep.path / "examples.csv", index=False)
    rep.finish()


if __name__ == "__main__":
    main()
