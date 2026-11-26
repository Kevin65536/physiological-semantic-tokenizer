"""Export the D128 frozen-token audit as searchable Chinese PDF, HTML and evidence workbook."""

import argparse
from html import escape
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlsxwriter

import audit_aligned_tokenizer as audit

D = audit.OUT
P = audit.PARENT
ROOT = audit.aligned.ROOT
DS = {
    "eeg_fnirs_single_trial": "Single-Trial",
    "refed": "REFED",
    "simultaneous_eeg_nirs": "Simultaneous",
    "visual_cognitive_motivation": "Visual",
}
MOD = {"eeg": "EEG", "hb": "Hb"}
BLUE = "#185d81"
ORANGE = "#ba582e"
GREEN = "#337d6c"
GRAY = "#859198"
COL = [BLUE, ORANGE, GREEN, "#79567c", "#ab8e29", "#495d6c"]
M = ["eeg", "hb"]


def csv(n):
    return pd.read_csv(D / (n + ".csv"))


def obj(n):
    return json.loads((D / n).read_text())


def panel(n=1, m=1):
    return plt.subplots(n, m, figsize=(12.4, 4.8), layout="constrained", squeeze=False)


def both(n):
    return pd.concat([csv(n.format(m=m)).assign(mod=MOD[m]) for m in M], ignore_index=True)


def macro(d, keys, cols):
    return (
        d.groupby([*keys, "dataset", "subject"])[cols]
        .mean()
        .groupby([*keys, "dataset"])
        .mean()
        .groupby(keys)
        .mean()
    )


def heat(ax, df, title, fmt=".2f", cmap="YlGnBu"):
    z = df.to_numpy(float)
    p = ax.imshow(z, aspect="auto", cmap=cmap)
    ax.set(
        xticks=range(len(df.columns)),
        xticklabels=df.columns,
        yticks=range(len(df)),
        yticklabels=df.index,
        title=title,
    )
    for (y, x), v in np.ndenumerate(z):
        if np.isfinite(v):
            rgba = p.cmap(p.norm(v))
            luma = np.dot(rgba[:3], [0.299, 0.587, 0.114])
            ax.text(
                x,
                y,
                format(v, fmt),
                ha="center",
                va="center",
                fontsize=10,
                color="white" if luma < 0.52 else "#182933",
            )
    return p


def bars(ax, df, title, ylabel, rotation=0):
    df.plot.bar(ax=ax, color=COL[: len(df.columns)], width=0.76, rot=rotation)
    ax.axhline(0, color="#536670", lw=0.6)
    ax.set(xlabel="", title=title, ylabel=ylabel)
    ax.legend(fontsize=9, loc="best")
    ax.grid(axis="y", alpha=0.15)


class Report:
    def __init__(self, base):
        self.out = base
        (base / "figures").mkdir()
        self.doc = fitz.open()
        self.html = []
        self.heads = []
        self.routes = {}

    def text(self, p, rect, markup, size):
        css = f"*{{font-family:sans-serif;color:#213944;font-size:{size}pt;line-height:1.38}}"
        rem, scale = p.insert_htmlbox(fitz.Rect(rect), markup, css=css, scale_low=1)
        if rem < 0:
            raise ValueError(f"Text overflow {self.heads[-1]} {markup[:90]}")

    def start(self, key, title):
        self.routes.setdefault(key, len(self.heads) + 1)
        self.heads.append(title)
        p = self.doc.new_page(width=936, height=690)
        self.text(
            p,
            (26, 11, 909, 37),
            f"v3 / EEG 与 fNIRS 冻结词元审计　　　2026-11-26　　　普通开发集　　　{len(self.heads):02}",
            10,
        )
        self.text(p, (25, 39, 912, 80), escape(title), 22)
        return p

    def page(self, key, title, intro, fig, caption, src):
        p = self.start(key, title)
        self.text(p, (26, 83, 910, 151), escape(intro), 13)
        name = f"{len(self.heads):02}_{key}.png"
        dest = self.out / "figures" / name
        fig.savefig(dest, dpi=235, facecolor="white")
        plt.close(fig)
        p.insert_image(fitz.Rect(25, 152, 911, 510), filename=str(dest), keep_proportion=True)
        self.text(p, (26, 519, 910, 639), escape(caption), 13)
        self.text(p, (26, 643, 910, 687), "数据表　" + escape(src), 9.4)
        self.html.append(
            f'<section id="p{len(self.heads)}"><h2>{escape(title)}</h2><p>{escape(intro)}</p><a href="figures/{name}"><img src="figures/{name}" alt="{escape(title)}"></a><p>{escape(caption)}</p><small>数据表　{escape(src)}</small></section>'
        )

    def table(self, key, title, intro, columns, rows, foot):
        p = self.start(key, title)
        self.text(p, (26, 83, 910, 145), escape(intro), 13)
        tab = (
            '<table style="border-collapse:collapse;width:100%"><tr>'
            + "".join(f'<th style="background:#dae8eb;padding:5px">{escape(c)}</th>' for c in columns)
            + "</tr>"
        )
        for i, row in enumerate(rows):
            tab += (
                "<tr>"
                + "".join(
                    f'<td style="border-bottom:1px solid #cbd9db;background:{"#f2f6f6" if i%2==0 else "#ffffff"};padding:5px">{escape(str(c))}</td>'
                    for c in row
                )
                + "</tr>"
            )
        tab += "</table>"
        self.text(p, (26, 148, 910, 605), tab, 12)
        self.text(p, (26, 609, 910, 683), escape(foot), 11.5)
        self.html.append(
            f'<section id="p{len(self.heads)}"><h2>{escape(title)}</h2><p>{escape(intro)}</p>{tab}<p>{escape(foot)}</p></section>'
        )

    def finish(self):
        self.doc.set_toc([[1, h, i] for i, h in enumerate(self.heads, 1)])
        self.doc.set_metadata(
            {"title": "v3 冻结 EEG 与 fNIRS 词元分析", "subject": "双模态 128 维单级 VQ 普通开发集审计"}
        )
        self.doc.subset_fonts()
        self.doc.save(self.out / "REPORT.pdf", garbage=4, deflate=True)
        self.doc.close()
        nav = "".join(f'<li><a href="#p{i}">{escape(s)}</a></li>' for i, s in enumerate(self.heads, 1))
        (self.out / "index.html").write_text(
            '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>v3 冻结词元分析</title><style>body{font:17px/1.65 sans-serif;max-width:1160px;padding:24px;margin:auto;color:#213944}img,table{width:100%}section{border-top:1px solid #b8caca;margin-top:24px}a{color:#185d81}small{overflow-wrap:anywhere}ol{columns:2}td,th{padding:8px;text-align:left}</style><h1>v3 冻结 EEG 与 fNIRS 词元分析</h1><p><a href="REPORT.pdf">下载 PDF</a>　<a href="evidence.xlsx">64 项证据工作簿</a>　<a href="tables/">原始评分表</a></p><ol>'
            + nav
            + "</ol>"
            + "".join(self.html)
            + "</html>"
        )


def overview(r):
    r.table(
        "design",
        "128 维冻结词元能保留什么信息？",
        "EEG 采用 C 候选，每支 8192 个词；fNIRS 的双颜色 Hb tokenizer 每支 1024 个词。四个码本向量宽度均为 128，使用终点第 10,000 步、种子 17。",
        ["问题", "本轮发现", "评价单位"],
        [
            (
                "词能否还原波形",
                "整数复放逐元素一致。EEG 的宏 NMSE 为 0.5062，Hb 为 0.1791。",
                "四库，16 位普通开发人",
            ),
            (
                "哪些词满足可读规则",
                "EEG-S/O 覆盖 45.39% / 30.04% 的片段；Hb-S/O 为 87.78% / 78.77%。",
                "按有效片段加权；开发参与筛选",
            ),
            (
                "两支是否互补",
                "已有 S 时，增加 O 对实测 X 属性的误差收益为 EEG 2.41%、Hb 43.48%。",
                "四库等权，人内配对",
            ),
            (
                "过去的顺序是否有用",
                "预测相邻窗时，完整转移加驻留对词袋为 EEG-S −2.40%、Hb-S +4.60%；四维对照下 Hb-S 约 +0.14%。",
                "下一独立窗首个 2 秒",
            ),
            (
                "词与真实事件是否对应",
                "后部高 alpha 词族活动期占用的 MA−BL 变化为 +0.167；精确双侧 p=0.25。",
                "有完整条件配对的 3 人",
            ),
            (
                "EEG 能否改善实测 Hb 预测",
                "Hb 自身历史 NMSE 0.18861；加入对齐 EEG 词元为 0.22834。3 人的误差均上升。",
                "107 个开发试次，3 人",
            ),
        ],
        "X 是观测波形。S 和 O 是 v3 教师分解的配对目标。报告中的读出、词族和检索均在冻结模型上计算；机制推断受限于教师定义、重复开发及外部测量。",
    )
    r.table(
        "design",
        "从完整 30 秒输入到独立被试评分",
        "每次模型只接收本模态、单一原生位置的归一化 X。EEG 每窗 6000 个样点；Hb 每窗 300 个双颜色时间点。每支编码器输出 15 个 2 秒位置对应的整数。",
        ["评价面板", "可见信息及评分量", "样本来源与汇总"],
        [
            (
                "完整读出",
                "量化前向量、码向量或 ID；预测单支真实目标属性。背景为数据集、空间位置、可见 X 的 RMS 档。",
                "60 人拟合；四库 16 人评分。先合并窗内位置，按项目层级汇总至被试、库等权。",
            ),
            (
                "全词表",
                "训练属性均值收缩后，将实际发出的词按训练分位数分成四族；开发仅用于评价及 readable 描述。",
                "EEG 开发有效 780,210 个 2 秒片段；Hb 为 838,583 个。",
            ),
            (
                "局部干预",
                "每人抽最多 32 个完整窗，干预中心信号或整数；背景供体同人、同记录、同位置且时间不重叠。",
                "每模态 1,920 个训练窗、512 个开发窗。窗口在被试内重复。",
            ),
            (
                "成对多位置",
                "在两个原生时钟上对齐同一父记录。四个固定前后左右锚点就近分配到四个不同有效位置。",
                "1,398 个候选，有效 619 个训练窗、164 个开发窗；该面板覆盖三库、14 位开发人。",
            ),
            (
                "事件主端点",
                "Single-Trial 心算 MA 与显式基线 BL。分别分析后部词族占用和实测前额双颜色 Hb 变化。",
                "数据集开发候选 5 人；按各端点可用试次与会话平衡后主检验 3 人。",
            ),
            (
                "统计口径",
                "增益为 1−候选误差/参照误差；差值为参照误差−候选误差。95% 区间按库内被试重采样 2,000 次。",
                "报告明确标出采用不同权重的统计量。仅预指定事件检验给确认方向的精确 p，其余探索性。",
            ),
        ],
        "EEG 的 C 训练目标含 theta/alpha 谱属性；这些属性的可读性有定向监督。四个库为 Single-Trial、REFED、Simultaneous、Visual。Visual 的单模态结果保留，未进入本次共同四位置配对面板。",
    )


def reconstruction(r):
    info = json.loads((P / "summary.json").read_text())["results"]
    fig, ax = panel(1, 3)
    for i, m in enumerate(M):
        v = np.array(info[m]["component_nmse"])
        bars(
            ax[0, i],
            pd.DataFrame(v, index=list("XSO"), columns=["EEG"] if m == "eeg" else ["HbO", "HbR"]),
            MOD[m] + " 全开发面板",
            "训练方差归一化 MSE（越低越好）",
        )
    d = both("{m}_wave_errors")
    e = d.query("mod=='EEG' and metric.str.endswith('Hz')", engine="python")
    e = e.groupby(["dataset", "subject", "component", "metric"])[["error", "target"]].sum()
    e["ratio"] = e.error / e.target
    s = (
        e.ratio.groupby(["dataset", "component", "metric"])
        .mean()
        .groupby(["component", "metric"])
        .mean()
        .unstack(0)
    )
    bars(
        ax[0, 2],
        s.loc[["1-4Hz", "4-8Hz", "8-13Hz", "13-30Hz", "30-45Hz"]],
        "EEG 原网格频带残差",
        "残差能量 / 对应目标能量",
        32,
    )
    r.page(
        "reconstruction",
        "Hb 的 X 重建误差低，EEG 的高频残差接近目标能量",
        "左、中图使用完整重训开发面板。右图使用局部审计的 16 人、每人 32 窗，比较同一原生采样网格上 Hann 加窗后的残差和目标频带能量；1 为零波形预测对应的比值。",
        fig,
        "EEG 的 S 波形 NMSE 为 0.674，高于 O 的 0.436。Hb 则是 O 较难重建，HbO/HbR 的 O 误差为 0.489/0.333。右图按人先合并误差与目标能量，再在库内和库间取均值。极低能量频带的比值易放大；源表同时给出两种能量、导数和分色结果。",
        "single_vq_aligned/20261126_v1/summary.json；eeg_wave_errors.csv；hb_wave_errors.csv",
    )
    ev = P / "eeg/models/eeg__waveform__vq8192/evaluation"
    for case in ["median", "difficult"]:
        z = np.load(ev / f"{case}_waveforms.npz")
        fig, ax = panel(3, 1)
        for a, orig, rec in zip(ax[:, 0], "xso", ["sum", "S", "O"]):
            y = z[orig].flatten()
            p = z["reconstructed_" + rec].flatten()
            mask = z["mask"].flatten()
            t = np.arange(len(y)) / 200
            y = np.where(mask, y, np.nan)
            p = np.where(mask, p, np.nan)
            a.plot(t, y, color=GRAY, lw=0.7, label="观测 X" if orig == "x" else "教师 " + orig.upper())
            a.plot(t, p, color=BLUE, lw=0.7, label="整数解码")
            a.set(ylabel=orig.upper() + " / SD", xlim=(0, 30))
            a.legend(loc="upper right", ncol=2, fontsize=8)
        ax[-1, 0].set_xlabel("窗内时间（秒）")
        ids = []
        for f in ev.glob("*_fixed_windows.json"):
            d = json.loads(f.read_text())
            if case in d:
                ids.append(d[case])
        meta = ids[0]
        nm = meta["nmse"]
        label = "中位示例" if case == "median" else "困难示例"
        r.page(
            "reconstruction",
            "固定 EEG " + label + "的 X、S、O 逐样点复放",
            f'重训评估继承已选窗口。{DS[meta["dataset"]]}，被试 {meta["subject"]}，记录 {meta["record"]}，位置 {meta["position"]}。三行按本行幅度独立取轴限，实线叠加比较相位和幅度。',
            fig,
            f"同一组 S/O 整数产生三行预测，X 为两支解码之和。目标和预测共享训练输入标准差。此窗 X/S/O 的 NMSE 分别为 {nm[0]:.3f}、{nm[1]:.3f}、{nm[2]:.3f}；缺失支持保留为空白。两例沿用父评估坐标，避免从新结果中挑选波形。",
            f"single_vq_aligned/20261126_v1/eeg/models/eeg__waveform__vq8192/evaluation/{case}_waveforms.npz",
        )


def members(r):
    for m in M:
        for b in "SO":
            root = D / f"{m}_17"
            data = np.load(root / f"{b}_member_waves.npz")
            table = csv(f"{m}_17/{b}_members")
            fig, ax = panel(2, 4)
            j = 1 if b == "S" else 2
            for i, tag in enumerate(["low", "middle", "high", "heldout_failure"]):
                sel = table[table.group == tag]
                word = int(sel.word.iloc[0])
                key = tag + "_" + str(word)
                mem = data[key + "_validation_member"]
                tr = data[key + "_train_member"]
                bg = data[key + "_validation_background"]
                channels = 1 if m == "eeg" else 2
                rate = 200 if channels == 1 else 10
                for row, component in enumerate([0, j]):
                    a = ax[row, i]
                    y = mem[:, component].reshape(len(mem), -1, channels)
                    target = tr[:, component].reshape(len(tr), -1, channels).mean(0)
                    ctrl = bg[:, component].reshape(len(bg), -1, channels).mean(0)
                    # Representative nearest to the centroid in squared distance is the squared-Euclidean medoid.
                    med = np.argmin(((y - y.mean(0)) ** 2).mean((1, 2)))
                    t = np.arange(y.shape[1]) / rate
                    for ch in range(channels):
                        col = COL[ch]
                        a.plot(
                            t,
                            y[med, :, ch],
                            c=col,
                            lw=1,
                            label=["EEG", "HbR"][ch] if m == "eeg" else ["HbO", "HbR"][ch],
                        )
                        a.plot(t, target[:, ch], c=col, ls="--", lw=1)
                        a.plot(t, ctrl[:, ch], c=col, ls=":", lw=1)
                    if row == 0:
                        a.set_title(
                            f'{["低频","中频","高频","开发失败"][i]} · ID {word}\nn={len(mem)}', fontsize=11
                        )
                    if row == 1:
                        a.set_xlabel("秒")
                    if i == 0:
                        a.set_ylabel(("实测 X" if row == 0 else "教师 " + b) + " / SD")
            r.page(
                "members",
                f"{MOD[m]}-{b} 的真实成员，从低频词到开发失败词",
                "前三列取训练合格词频的第 10、50、90 百分位；末列取开发人均属性改善最差的合格词。上排为真实 X，下排为同片段教师分量。每格有独立纵轴，标出的 n 为开发匹配成员数。",
                fig,
                "实线是该格开发成员中距成员均值最近的真实波形，虚线是训练成员均值，点线是同人、同位置、同幅度档其他词的开发均值。"
                + ("蓝线表示 EEG。" if m == "eeg" else "蓝为 HbO，橙为 HbR。")
                + "平均模板会抵消相位，词义应结合分布判断。配套数据另有每次配对、随机抽取的真实成员、失败词和按颜色的模板误差。",
                f"{m}_17/{b}_members.csv、{b}_member_waves.npz、{b}_wave_template.csv",
            )


def vocabulary(r):
    fig, ax = panel(1, 2)
    occ = csv("occupancy").query("split=='dev'").copy()
    occ["stream"] = occ["mod"].map(MOD) + "-" + occ.branch
    occ = occ.set_index("stream")
    bars(
        ax[0, 0],
        occ[["capacity", "used", "perplexity"]].rename(
            columns={"capacity": "总词数", "used": "出现词数", "perplexity": "有效词数"}
        ),
        "出现频数与熵的有效词数",
        "词数",
    )
    d = both("{m}_17/coverage").query("dataset!='pooled'")
    d["stream"] = d["mod"] + "-" + d.branch
    v = d.pivot(index="stream", columns="dataset", values="readable_fraction").rename(columns=DS)
    heat(ax[0, 1], 100 * v, "可读词覆盖的开发片段（%）", ".1f")
    r.page(
        "coverage",
        "常用词覆盖了大量片段，覆盖率随模态和数据集变化",
        "训练至少 20 个片段、至少 3 人才进入候选。可读词另须在至少 3 位开发人出现，人均条件属性 MSE 改善且过半开发人改善。误差相对训练背景预测。",
        fig,
        "EEG-S 的 8,192 个词中有 2,560 个满足规则，覆盖开发片段 45.39%；Hb-S 的 1,024 个词中有 509 个，覆盖 87.78%。有效词数为按开发频率计算的熵的指数。四流逐词表同时保留训练支持、开发支持、属性均值和失败；按词频层拆解的词内和词间残差方差见附表。",
        "occupancy.csv；{eeg,hb}_17/coverage.csv、S_words.csv、O_words.csv；{eeg,hb}_variance.csv",
    )
    for m in M:
        fig, ax = panel(1, 2)
        for i, b in enumerate("SO"):
            q = (
                csv(f"readout_{m}_17_{b}/scores")
                .query("dataset=='macro'")
                .pivot(index="method", columns="attribute", values="gain")
            )
            q = q.loc[["hard_z", "embedding", "shrunk", "balanced_shrunk", "decoded"]]
            q.index = ["量化前 z", "实际码向量", "收缩 ID", "人均衡收缩 ID", "解码波形"]
            q.columns = [
                {
                    "power_4_8": "θ 功率",
                    "power_8_13": "α 功率",
                    "patch_rms": "2秒 RMS",
                    "rms_envelope": "250ms 包络",
                    "HbO_centered": "HbO 形态",
                    "HbR_centered": "HbR 形态",
                    "HbO_slope": "HbO 斜率",
                    "HbR_slope": "HbR 斜率",
                }[k]
                for k in q
            ]
            heat(ax[0, i], 100 * q, b + " 支误差增益（%）", ".1f", "PuOr")
        r.page(
            "readouts",
            MOD[m] + " 同支持读出比较：量化前、单个词与整数解码",
            "每个数是 1−预测 MSE/对应背景 MSE，以百分数显示；负值表示预测差于背景。背景固定数据集、位置与可见 X 的 RMS 档。收缩和岭系数在训练被试三折选择。各格保留精确数值，色阶在左右两图内独立。",
            fig,
            (
                "EEG-S 的包络从量化前 50.9% 降到码向量 40.7%，收缩 ID 为 11.9%；α 绝对功率的码向量收益只有约 1.5%。开发 S 的背景×ID 有 70.3% 在训练未见，稀疏组合限制查表表现。"
                if m == "eeg"
                else "Hb-S 码向量对 HbO 和 HbR 斜率的增益分别约 56.8% 和 38.6%；整数解码后从完整 15 词恢复的斜率分别为 72.9% 和 60.2%。该差异包含解码器获得的上下文。"
            )
            + "量化前 z 来自同一终点模型。SD 为训练观测标准差。",
            f"readout_{m}_17_S/、readout_{m}_17_O/ 的 scores.csv、support.json、selection.json",
        )
    fig, ax = panel(1, 2)
    sp = csv("spectral_strata")
    q = macro(sp, ["attribute", "stratum"], ["mse"]).mse.unstack()
    q.columns = ["≤P10", "P10–50", "P50–90", ">P90"]
    heat(ax[0, 0], q, "EEG-S 标准化谱 MSE（人、库等权）", ".1f")
    for j, attr in enumerate(["log_alpha", "log_theta"]):
        q1 = sp[sp.attribute.eq(attr)].groupby(["dataset", "stratum"]).mse.mean().unstack().rename(index=DS)
        for label, row in q1.iterrows():
            if j == 0:
                ax[0, 1].plot(range(4), row, "o-", label=label)
    ax[0, 1].set(
        xticks=range(4), xticklabels=q.columns, title="log α 的分库误差", ylabel="训练谱 SD 单位的 MSE"
    )
    ax[0, 1].legend()
    r.page(
        "spectrum",
        "EEG-S 的绝对 log α 功率误差贯穿多个功率层",
        "以本库训练目标的 10%、50%、90% 分位数划分每个属性；开发数据固定用同一阈值。log 功率由 G0 归一化带能量加地板后，再恢复原输入尺度并以训练均值和 SD 标准化。",
        fig,
        "在 780,210 个完整开发片段中，模型输出和教师目标的 θ、α 绝对功率触及 10⁻¹² 地板的计数均为 0（按恢复后 log 阈值容差 10⁻⁴ 判断）。各功率层均保留较大 log α 误差。频谱目标可从向量回归出一部分，同时波形解码仍会丢失高频能量。",
        "spectral_strata.csv；eeg_wave_errors.csv；readout_eeg_17_S/scores.csv",
    )


def local_pages(r):
    for m in M:
        df = csv(m + "_attribute_probes")
        ran = (
            csv(m + "_random_readout")
            .query("method=='random_geometry'")
            .rename(columns={"attribute": "attr"})
        )
        d = pd.concat([df, ran])
        mse = macro(d, ["branch", "attr", "method"], ["mse", "baseline"])
        score = (100 * (1 - mse.mse / mse.baseline)).unstack()
        attrs = (
            [
                "delta_frac",
                "theta_frac",
                "alpha_frac",
                "beta_frac",
                "low_gamma_frac",
                "peak_hz",
                "halfpower_width",
                "aperiodic_slope",
                "burst_seconds",
                "curvature",
                "zero_cross",
            ]
            if m == "eeg"
            else [
                "HbO_level",
                "HbR_level",
                "HbO_slope",
                "HbR_slope",
                "HbO_peak_time",
                "HbR_peak_time",
                "peak_lag",
                "color_cov",
                "HbO_curvature",
                "HbR_release",
            ]
        )
        labels = (
            [
                "δ 占比",
                "θ 占比",
                "α 占比",
                "β 占比",
                "低 γ 占比",
                "峰频",
                "峰宽",
                "非周期斜率",
                "高包络时长",
                "曲率",
                "过零率",
            ]
            if m == "eeg"
            else [
                "HbO 水平",
                "HbR 水平",
                "HbO 斜率",
                "HbR 斜率",
                "HbO 峰时",
                "HbR 峰时",
                "双色峰时差",
                "双色协方差",
                "HbO 曲率",
                "HbR 峰后差",
            ]
        )
        fig, ax = panel(1, 2)
        for j, b in enumerate("SO"):
            z = score.loc[b].loc[attrs, ["observed", "latent", "embedding", "random_geometry"]]
            z.index = labels
            z.columns = ["实测摘要", "量化前", "码向量", "置乱码向量"]
            heat(ax[0, j], z, b + " 对教师属性的条件增益（%）", ".0f", "PuOr")
        r.page(
            "features",
            MOD[m] + " 的哪些局部属性能由同一读出器恢复？",
            "训练每人约 32 个中心片段拟合带相同背景的岭回归；在 16 位开发人的独立中心上评分。四列依次为 X 的手工摘要、量化前 z、实际码向量、随机重排 ID 与向量对应后得到的同维对照。",
            fig,
            (
                "EEG 的局部峰从 6–14 Hz 内相对两侧 log–log 背景的最大点获得，半功率带宽取连续 3 dB 范围。各窗仅 2 秒，0.5 Hz 分辨率限制了峰宽解释。高包络时长取 8 个 250 ms 子段内超本窗中位数 1.5 倍的总时长。"
                if m == "eeg"
                else "HbO/HbR 共享一个位置与一个 ID。斜率为 2 秒内线性趋势，局部峰时由离散峰样点给出，峰后差取末点减峰值；色间协方差在去均值后计算。"
            )
            + "每格是相对背景的误差百分比；颜色在各图内独立，负值保留。",
            f"{m}_attribute_probes.csv；{m}_random_readout.csv；{m}_local_panel.csv",
        )
    fig, ax = panel(1, 2)
    hs = csv("hb_signs").query("method=='embedding' or readout=='model'")
    hs["model"] = np.where(
        hs.readout == "background",
        "背景",
        hs.method.map({"embedding": "码向量", "random_geometry": "置乱几何", "latent": "量化前"}),
    )
    bars(
        ax[0, 0],
        macro(hs, ["model", "branch"], ["correct_both"]).correct_both.unstack(),
        "中心 2 秒的 HbO 与 HbR 斜率符号同时正确",
        "按人及库平均的比例",
    )
    z = csv("hb_response").query("split=='validation'")
    z = z.groupby(["subject", "color", "kind"])[["peak_s", "recovery"]].mean().reset_index()
    bars(
        ax[0, 1],
        z.groupby(["color", "kind"])
        .peak_s.mean()
        .unstack()
        .rename(columns={"observed": "实测 X", "decoded": "解码 S+O"}),
        "30 秒前部双位置平均波形的峰时",
        "距窗口起点的秒数",
    )
    r.page(
        "hbshape",
        "Hb 码向量可以同时保留两种颜色的趋势符号",
        "左图逐窗判断两个斜率的符号是否同时命中。右图使用双时钟纳入的 MA/BL 窗，前部两位置取均值，以窗首 2 秒校正基线后，取整窗峰时；每位被试先聚合，样本量与历史主端点独立。",
        fig,
        "双色在同一个码中联合编码，S 的码向量能比背景更好地判断联合方向。峰时对局部起伏敏感，平均峰时相近不能保证每个试次对齐。源表保留两色的峰高、峰时、前 10 秒平均斜率及末段相对峰值的差。",
        "hb_signs.csv；hb_response.csv；hb_attribute_probes.csv",
    )
    fig, ax = panel(1, 2)
    d = csv("detector")
    d["precision"] = d.intersection / d.selected
    d["recall"] = d.intersection / d.positives
    d["coverage"] = d.selected / d.n
    d["prevalence"] = d.positives / d.n
    bars(
        ax[0, 0],
        d.groupby("branch")[["precision", "recall", "coverage", "prevalence"]]
        .mean()
        .rename(
            columns={
                "precision": "精确率",
                "recall": "召回率",
                "coverage": "Q4 覆盖率",
                "prevalence": "事件率",
            }
        ),
        "以训练定义的 Q4 识别真实片段中的 α burst",
        "8 位开发人等权比例",
    )
    for i, b in enumerate("SO"):
        v = d[d.branch == b]
        ax[0, 1].scatter(v.matched_control, v.matched_member, label=b, marker=["o", "^"][i], color=COL[i])
    ax[0, 1].plot([0, 1], [0, 1], "--", c=GRAY)
    ax[0, 1].set(xlabel="匹配非 Q4 的事件率", ylabel="Q4 成员的事件率", title="每个点为一名被试和一个分支")
    ax[0, 1].legend()
    r.page(
        "detection",
        "高 α 词族对信号检测器的命中因人而异",
        "独立算法在实测 X 上做 8–13 Hz 带通、Hilbert 包络及 RMS 归一化；每库训练样点第 90 百分位定阈值，排除两端各 0.2 秒，持续至少 0.25 秒算命中。判定规则没有使用开发 ID。",
        fig,
        "8 人共有 12,414 个片段，检测到 2,196 个 burst；EEG-S 的 Q4 命中 624 个。眼电另以同人同额区电极、RMS 比不超过 1.4 匹配，S 慢词仅 2 人 11 对，O 慢词无合格对。独立专家盲评包包含真实成员和对照，当前未产生人工评分。",
        "detector.csv；eye_contrasts.csv；eye_people.csv；blind/blinded_waveforms.pdf",
    )
    fig, ax = panel(1, 2)
    stab = both("{m}_family_stability")
    geo = both("{m}_geometry")
    geo["流"] = geo["mod"] + "-" + geo.branch
    z = stab.groupby(["mod", "branch", "historical_seed"])[["agreement", "ARI"]].mean().reset_index()
    z["流"] = z["mod"] + "-" + z.branch
    bars(
        ax[0, 0],
        z.pivot(index="流", columns="historical_seed", values="agreement").rename(
            columns={17: "历史 seed 17", 43: "历史 seed 43"}
        ),
        "同一片段被分到相同四分位族",
        "留出被试平均一致率",
    )
    bars(
        ax[0, 1],
        geo.set_index("流")[["rho"]].rename(columns={"rho": "Spearman ρ"}),
        "码本距离与开发真实属性距离",
        "按支持合格词组成的词对",
    )
    r.page(
        "geometry",
        "历史词族可部分复现，码本相近的词常有相近属性",
        "左图先在各自训练集定义有序四分位，再对相同的开发片段比较。右图用训练合格且开发至少 3 人的词，EEG 每支随机取 1,500 个，Hb 使用全部；计算码向量余弦距离与开发人均分支属性绝对差的秩相关。",
        fig,
        "EEG-S 的词对相关为 0.340，最近邻属性差 0.0194，置乱属性为 0.0366。词对共享词，不能作为独立重复。历史 EEG C 的宽度为 64，本轮为 128；历史比较同时包含架构宽度和初始化差别。Hb 历史两种子保持 128 维。ARI 是调整随机一致性的聚类指标，详见源表。",
        "{eeg,hb}_historical.csv、_family_stability.csv、_geometry.csv",
    )
    fig, ax = panel(1, 2)
    for i, m in enumerate(M):
        d = csv(m + "_perturb")
        q = macro(d, ["branch", "kind", "dose"], ["same"]).reset_index()
        for (b, kind), v in q[q.kind.isin(["amplitude", "noise"])].groupby(["branch", "kind"]):
            ax[0, i].plot(
                100 * v.dose,
                v.same * 100,
                marker="o" if b == "S" else "^",
                ls="-" if kind == "noise" else "--",
                color=COL["SO".index(b)],
                label=b + (" 噪声" if kind == "noise" else " 幅度"),
            )
        ax[0, i].set(
            title=MOD[m] + " 的中心整数保持率",
            xlabel="相对中心波动幅度的剂量（%）",
            ylabel="中心整数相同比例（%）",
            ylim=(0, 104),
        )
        ax[0, i].legend(ncol=2, fontsize=9)
    r.page(
        "context",
        "同一中心片段的 ID 对窗口起点和信号扰动均敏感",
        "每条曲线比较成对的原窗与单一扰动窗，剂量 5%、20%、50%；噪声乘中心波动 SD，幅度变化围绕原片段均值施加。保持全窗不变的身份对照对四个分支均为 100%。",
        fig,
        "固定中心 2 秒只替换同记录的真实前后文，EEG-S 约一半 ID 保持，Hb-S 约 13%。将完整窗口起点平移 2 秒后，Hb-S 内部相同片段 ID 保持约 69%，两端约 33%。EEG 的等能量 10 Hz 短 burst 和宽带噪声对 S 的 ID 保持约为 97% 和 83%。这表明干预响应依赖内容与上下文。",
        "{eeg,hb}_perturb.csv、_window_shift.csv、_artifact_controls.csv；boundary_shift.csv",
    )
    fig, ax = panel(1, 2)
    for i, m in enumerate(M):
        q = macro(csv(m + "_decode_controls"), ["kind", "branch"], ["central_rms", "outside_rms"])
        q = q.stack().unstack("branch")
        q.index = [f"{a}·" + ("中心" if c == "central_rms" else "外围") for a, c in q.index]
        heat(ax[0, i], q, MOD[m] + " 解码变化 RMS（训练输入 SD）")
    r.page(
        "edits",
        "改一个整数主要改变本地波形；固定中心整数仍允许上下文改变输出",
        "swap_far 从同人同位置的远时段取中心 ID；family_near 从供体取同词族中另一个 ID，未找到则保留原 ID。neighbor 仅换周围 14 个词。reverse、shuffle 分别反转和随机重排整条序列。",
        fig,
        "每个条件都使用冻结整数解码器。中心为第 8 个位置，外围是其余 14 个位置。Hb-S 固定中心而换邻居，中心变化约 0.475 个训练 SD；EEG-S 约 0.049。单支干预核验另一支的每个输出元素不变。逐样本表记录实际是否换词，真实目标 MSE 的变化另列。",
        "{eeg,hb}_decode_controls.csv、_edit.csv",
    )
    fig, ax = panel(1, 2)
    for i, m in enumerate(M):
        q = csv(m + "_17/conditional").query("target=='X' and candidate!='SO_interaction'").copy()
        q["比较"] = q.reference + " → " + q.candidate
        bars(
            ax[0, i],
            (100 * q.set_index("比较")[["gain"]]).rename(columns={"gain": "误差下降"}),
            MOD[m] + " 对实测 X 属性的条件增益",
            "相对前一模型 MSE 的下降（%）",
            15,
        )
    r.page(
        "complement",
        "Hb 的 S/O 两支对实测斜率有明显互补",
        "目标为同位置实测 X 的 EEG α 占比或 HbO 斜率。训练背景与岭读出固定。每次只在指定参照上加入相应分支，先被试内配对再按库汇总。",
        fig,
        "在 S 上增加 O，EEG 增益 2.41%（14/16 人改善），Hb 增益 43.48%（16/16 人改善）。对照包含两支词族主效应后，加入词族交互的 X 误差收益为 EEG −0.007%、Hb +0.228%。这组指标直接量化每个模态内部的两支互补。",
        "{eeg,hb}_17/conditional.csv、conditional_subject.csv、interaction_vs_family_main.csv",
    )
    fig, ax = panel(1, 2)
    for i, m in enumerate(M):
        d = csv(m + "_dynamics")
        z = macro(d, ["branch", "method"], ["dwell_s", "return_s", "transition_prob"])
        bars(
            ax[0, i],
            z.dwell_s.unstack().rename(
                columns={"bag_null": "词袋乱序", "observed": "实测属性分位", "token": "真实词族"}
            ),
            MOD[m] + " 的平均连续驻留",
            "秒",
        )
    r.page(
        "dynamics",
        "词族驻留与原信号分位、同词袋乱序可以直接比较",
        "每个完整 30 秒窗由 15 个词组成。驻留为连续同族片段长度乘 2 秒，再于窗内求均值。实测序列用训练 X 属性分位数离散。乱序保持窗内每一类的总词数。",
        fig,
        "驻留只计当前窗口内可观察的连续段，边界会截短跨窗长段。逐人表还给出同族再访间隔和相邻位置的转移概率。关系建模另有精确保留转移或驻留结构的替代对照，同一词袋的不同顺序有不同误差。",
        "{eeg,hb}_dynamics.csv；{eeg,hb}_17/sequence.csv",
    )
    fig, ax = panel(1, 2)
    for i, m in enumerate(M):
        v = csv(m + "_17/sequence")
        v = v[v.reference.isin(["bag", "bigram_dwell"])].copy()
        v["实验"] = np.where(v.adjacent, "相邻窗 ", "同窗 ") + v.branch
        z = 100 * v.pivot(index="实验", columns="reference", values="gain")
        z.columns = [{"bag": "二元+驻留 对 词袋", "bigram_dwell": "三元 对 二元+驻留"}[c] for c in z]
        bars(ax[0, i], z, MOD[m] + " 的时序读出增益", "相对误差下降（%）")
    r.page(
        "order",
        "Hb 有低阶顺序收益，EEG 的相邻窗读出未见同样收益",
        "同窗任务用前 14 个词预测最后一个实测属性，但编码看过全部 30 秒。相邻窗任务用前一完整窗的末 14 个词预测下一不重叠窗第一个实测属性。二元计数、驻留与三元 motif 逐级加入。",
        fig,
        "相邻窗 EEG 与 Hb 分别有 629/626 个训练窗对、28,106/30,035 个开发窗对。Hb-S/O 加二元及驻留的增益为 4.60%/8.07%；EEG-S/O 为 −2.40%/−2.50%。Hb-S 再加三元的结果为 −0.98%。在与词袋匹配的四维读出下，Hb-S 的该收益仅约 0.14%；有关对照见读出维度页。",
        "{eeg,hb}_17/sequence.csv、sequence_subject.csv、observed_sequence_reference.csv",
    )


def spatial_event_pages(r):
    for m in M:
        d = csv(m + "_montage")
        fig, ax = panel(1, 4)
        scatter = None
        for a, (ds, label) in zip(ax[0], DS.items()):
            g = d[d.dataset.eq(ds)].copy()
            g["hits"] = g.n * g.S_Q4
            g = g.groupby(["subject", "position", "x", "y"])[["hits", "n"]].sum().reset_index()
            g["rate"] = g.hits / g.n
            g = g.groupby(["position", "x", "y"]).rate.mean().reset_index()
            a.add_patch(plt.Circle((0, 0), 1, fill=False, ec=GRAY, lw=1))
            a.plot([-0.11, 0, 0.11], [0.99, 1.10, 0.99], c=GRAY, lw=1)
            scatter = a.scatter(
                g.x, g.y, c=g.rate, cmap="viridis", s=52, edgecolors="#475966", lw=0.3, vmin=0, vmax=1
            )
            a.set(
                xlim=(-1.12, 1.12),
                ylim=(-1.1, 1.14),
                aspect="equal",
                title=label + "\n" + str(g.position.nunique()) + " 个位置",
            )
            a.axis("off")
        fig.colorbar(
            scatter,
            ax=ax.ravel().tolist(),
            orientation="horizontal",
            label="训练定义 S 支 Q4 的开发占用比例",
            shrink=0.5,
            pad=0.03,
        )
        r.page(
            "spatial",
            MOD[m] + " 的全头原生位置词族分布",
            "每个点代表有坐标的该库原生位置，鼻尖朝上。每人每位置先合并有效片段的 Q4 占用，再对可用被试求均值。输入尺度为该位置训练观测 SD；四图采用同一色标。",
            fig,
            "EEG 的 Q4 按教师 S 的 α 占比定义，Hb 按教师 S 的 HbO 斜率定义。点的疏密反映原生布局和可用坐标。独立配对分析另选前后左右四个区域。无坐标的库或位置在图上留空。源表同时列 S/O 占用、实测属性、有效片段数与候选分母，保留无坐标位置。",
            "{0}_montage.csv；positions.csv；paired_panel.csv".format(m),
        )
    fig, ax = panel(1, 2)
    for i, m in enumerate(M):
        q = csv(m + "_synchrony")
        v = (
            macro(q, ["attribute", "method"], ["mse"])
            .mse.unstack()
            .rename(columns={"decoded": "整数解码", "shift": "位置独立时移"})
        )
        bars(ax[0, i], v, MOD[m] + " 的六个位置对", "连接统计相对实测的均方误差")
    r.page(
        "sync",
        "整数解码保留了部分跨位置的波动关系",
        "同一窗内取四个位置的六个位置对。EEG 计算 8–13 Hz Hilbert 包络相关与相位锁定值，剔除两端各 1 秒；Hb 对每种颜色计算位置间时间相关。比较解码波形与实测 X 的对应统计。",
        fig,
        "对照将各位置的解码波形按不同量平移，再与原始位置对比较。空间位置的 Q4 分数关系另用四位置逐时去均值、控制电极距离的秩相关评价；六个边在同一窗内相关联。共同参考、体积传导与系统性血流仍可影响这些头表相关。",
        "eeg_synchrony.csv；hb_synchrony.csv；spatial.csv",
    )
    fig, ax = panel(1, 2)
    v = (
        both("{m}_dataset_identity")
        .groupby(["dataset", "subject", "mod"])
        .correct.mean()
        .groupby(["dataset", "mod"])
        .mean()
        .unstack()
        .rename(index=DS)
    )
    bars(ax[0, 0], v, "由冻结词族分布识别数据集", "留人正确比例", 15)
    d = csv("direction_transfer")
    q = macro(d, ["fit", "method", "target"], ["mse"]).mse.unstack("method")
    q["收益"] = 100 * (1 - q.both / q.own_early)
    bars(
        ax[0, 1],
        q["收益"].unstack("fit").rename(columns={"within": "本库训练头", "leave_dataset": "其余库训练头"}),
        "自身早期摘要上增加对侧模态词族",
        "末 10 秒属性误差下降（%）",
    )
    r.page(
        "transfer",
        "词族分布可泄露数据集身份，跨库读出需单独解释",
        "左图用 4 个区域、3 个时间段的 S/O 词族占用拟合训练人逻辑回归，开发人独立。右图用自身前 10 秒实测属性，以及对侧全 30 秒词族占用，离线解释最后 10 秒实测属性；两侧可见时间不同。",
        fig,
        "EEG 的数据集身份易读出，设备、任务与布局共同变化。右图仅将读出头留一库；冻结 tokenizer 已见这些库的训练被试。按单区域留出再拟合的结果另列，固定四个位置、按人汇总。Visual 缺少该配对面板，但完整单模态词表包含 Visual。",
        "{eeg,hb}_dataset_identity.csv、_position_transfer.csv；direction_transfer.csv；paired_panel.csv",
    )
    fig, ax = panel(1, 2)
    ep = csv("events_17/event_profile").query("split=='validation' and area=='posterior'")
    ph = ["pre", "entry", "active", "exit", "recovery"]
    for i, cond in enumerate(["MA", "BL"]):
        t = ep[ep.condition == cond].groupby("phase").occupancy.mean().reindex(ph)
        ax[0, 0].plot(range(5), t, "o-" if i == 0 else "s--", label=cond, color=COL[i])
    ax[0, 0].set(
        xticks=range(5),
        xticklabels=["提示前", "进入", "活动", "退出", "恢复"],
        ylabel="后部 EEG-S Q4 占用比例",
        title="真实提示的条件轨迹",
    )
    ax[0, 0].legend()
    q = csv("pseudo_contrasts")
    v = q.pivot(index="offset", columns="subject", values="contrast")
    for sub in v:
        ax[0, 1].plot(v.index, v[sub], marker="o", label=sub)
    ax[0, 1].axhline(0, color=GRAY, lw=0.7)
    ax[0, 1].set(
        xticks=[-13, 0, 13],
        xlabel="伪提示相对真实提示的偏移（秒）",
        ylabel="MA−BL 的活动期占用变化",
        title="共同 116 试次的同人对照",
    )
    ax[0, 1].legend(fontsize=9)
    r.page(
        "events",
        "后部高 α 词族的预指定事件检验只有 3 个独立被试",
        "提示前 [−4,0)、进入 [0,2)、活动 [2,10)、退出 [10,18)、恢复 [18,24) 秒。主端点先算活动减提示前，再在同人同会话内算心算 MA 减显式基线 BL，最后对被试等权。",
        fig,
        "主端点 MA−BL 为 +0.1672，3 人被试 bootstrap 的 95% 区间 [0.0483,0.3333]；枚举三人符号的双侧精确 p=0.25。小样本下 bootstrap 区间与精确检验分辨率不同。124 个开发试次具备事件支持，真/伪提示共同支持 116 个；在这些试次内按会话置换条件得到双侧 p=0.110。",
        "events_17/event_contrasts.csv、event_profile.csv、event_support.csv；pseudo_contrasts.csv；cue_permutation.json",
    )
    fig, ax = panel(1, 2)
    for j, (src, sub) in enumerate([("eeg_Q4", "EEG"), ("hb_Q4", "Hb")]):
        q = csv("dose_people").query("split=='validation'")
        for i, (person, g) in enumerate(q.groupby("subject")):
            ax[0, j].plot(g.load, g[src], marker=["o", "s", "^", "D"][i % 4], label=person)
        ax[0, j].set(
            xticks=[0, 2, 3],
            xlabel="N-back 标注负荷",
            ylabel="S 支 Q4 占用比例",
            title=sub + " 逐人负荷曲线",
            ylim=(0, 0.7),
        )
        ax[0, j].legend(fontsize=9)
    ref = csv("refed_state")
    nr = ref.subject.nunique()
    r.page(
        "dose",
        "N-back 高低负荷的共同被试有限，个体响应方向不同",
        "Simultaneous 的 0、2、3-back 使用同一双时钟规则与固定四位置。每点取一位开发被试在该负荷下全部合格窗口的高词族平均占用。每人出现的有效条件按实际样本展示。",
        fig,
        f"0-back 仅 2 人 2 个窗，2-back 为 4 人 10 窗，3-back 为 4 人 5 窗。顺序、任务和疲劳相互关联。另在 REFED 的 {nr} 位开发人中用独立 1 Hz 摇杆评定计算 10 秒唤醒/效价与词族占用的记录内相关。该评定发生在观看视频期间；无任务自然觉醒状态仍缺少独立标注。",
        "nback_dose.csv；dose_people.csv；refed_state.csv；refed_annotations.csv",
    )
    fig, ax = panel(1, 2)
    d = csv("regional_lags")
    for i, m in enumerate(M):
        q = macro(d[d["mod"] == m], ["alignment", "lag_s"], ["r"]).r.unstack(0)
        for k, name in [("native", "相同区域"), ("region_swap", "前后区对换")]:
            ax[0, i].plot(q.index, q[k], marker="o", label=name)
        ax[0, i].axhline(0, c=GRAY, lw=0.7)
        ax[0, i].set(
            xticks=q.index, xlabel="对侧属性相对滞后（秒）", ylabel="去全局后相关 r", title=MOD[m] + " 在前"
        )
        ax[0, i].legend()
    r.page(
        "lags",
        "时间扫描把同区域关联与前后区域交换放在同一尺度",
        "EEG 取实测 α 占比，Hb 取实测 HbO 斜率。先逐时减四位置均值，再在共同剩余时间算跨模态 Pearson 相关。左右图分别将 EEG、Hb 放在前方；每窗、每人、每库依次汇总。",
        fig,
        "区域交换保持相同时间点和原始波形。冻结词的关联另在 MA/BL 条件内以训练中位数二值化 10 秒占用，在同人、记录、条件内估计经验互信息并与分离 ≥60 秒的错试次比较。每个分层样本很少，互信息有有限样本正偏，这两项均作为描述性诊断。",
        "regional_lags.csv；conditional_MI.csv；paired_panel.csv",
    )
    fig, ax = panel(1, 2)
    h = csv("events_17/coupling_subject")
    ds = h.groupby(["subject", "method"]).nmse.mean().unstack()
    order = ["base", "aligned", "continuous", "mismatch", "shift", "frozen_mismatch", "frozen_shift"]
    labs = ["自身历史", "对齐词", "连续 EEG", "重拟错试次", "重拟错时", "冻结错试次", "冻结错时"]
    for p, row in ds.iterrows():
        ax[0, 0].plot(range(len(order)), row[order], marker="o", label=p)
    ax[0, 0].set(
        xticks=range(len(order)),
        xticklabels=labs,
        ylabel="双颜色 NMSE，越低越好",
        title="同一批开发试次的各人结果",
    )
    ax[0, 0].tick_params(axis="x", rotation=29)
    ax[0, 0].legend(fontsize=8)
    q = (
        csv("events_17/coupling_contrasts")
        .query("color=='two_color'")
        .set_index("reference")
        .loc[["base", "continuous", "mismatch", "shift", "frozen_mismatch", "frozen_shift"]]
    )
    ax[0, 1].errorbar(
        q.difference,
        range(len(q)),
        xerr=np.array([q.difference - q.ci_low, q.ci_high - q.difference]),
        fmt="o",
        color=BLUE,
        capsize=4,
    )
    ax[0, 1].axvline(0, c=GRAY)
    ax[0, 1].set(
        yticks=range(len(q)),
        yticklabels=[labs[i] for i in (0, 2, 3, 4, 5, 6)],
        xlabel="参照 NMSE − 对齐词 NMSE",
        title="同人差值及 95% bootstrap 区间",
    )
    r.page(
        "history",
        "对齐 EEG 词元未降低 Hb 自身历史的预测误差",
        "实测目标为前额两个位置平均的 HbO/HbR 变化。历史含 [−2,0)、[0,2)、[2,4) 秒及条件、会话和序号。EEG 单独从 [−26,4) 秒原信号编码；按训练人三折从 2/4/6 秒选择时延，选中 6 秒，评分目标为 [10,12) 减 [2,4)。",
        fig,
        "训练 14 人 614 试次，开发 3 人 107 试次。每色误差除训练目标方差，会话内平衡 MA/BL，再对会话、被试和颜色平均。历史基线 0.18861，加入对齐词为 0.22834，连续摘要为 0.18963。三位被试在对齐词条件下均差于自身历史。相对错时对照的优势不能独立证明增量预测。",
        "events_17/events_summary.json、coupling_predictions.csv、coupling_subject.csv、coupling_contrasts.csv、coupling_pairs.csv",
    )
    fig, ax = panel(1, 2)
    s = csv("systemic_controls")
    v = s.groupby(["method", "color"]).nmse.mean().unstack()
    for i, clr in enumerate(["HbO", "HbR"]):
        z = v.loc[["base", "aligned", "HRF", "global", "global_EEG", "global_HRF"], [clr]]
        z.index = ["自身历史", "+ 对齐词", "+ HRF", "+ 全局 Hb", "全局 + 词", "全局 + HRF"]
        bars(ax[0, i], z, clr + " 误差", "训练目标方差归一化 MSE", 20)
    r.page(
        "systemic",
        "加入目标外全局 Hb 后，EEG 词元仍未带来增益",
        "在与主端点相同的 614/107 试次上，每个历史段分别取目标两位置以外有效光学位置的双色均值时间摘要，再跨位置取中位数。HRF 对照用固定双 gamma 权重汇总同一 EEG 历史中的 θ、α 和 RMS。",
        fig,
        "双颜色 NMSE 为自身历史 0.18861、历史加全局 Hb 0.18231、再加入 EEG 词元 0.22117。固定 HRF 摘要加历史为 0.18221，配合全局 Hb 为 0.17726。全局 Hb 是代理量；准入缓存未提供短程光学、逐试次呼吸、ECG、血压作独立回归，三人结果只能描述此开发面板。",
        "systemic_controls.csv；events_17/events_summary.json",
    )
    fig, ax = panel(1, 2)
    q = both("{m}_matched_retrieval")
    z = macro(q, ["mod", "branch", "method"], ["p5", "chance"]).reset_index()
    z["流"] = z["mod"] + "-" + z.branch
    p = z.pivot(index="流", columns="method", values="p5")
    p["机会"] = z.groupby("流").chance.mean()
    bars(
        ax[0, 0],
        p.rename(columns={"embedding": "码向量", "amplitude": "幅度"}),
        "同模态跨人：匹配图库中的前 5 名",
        "属性四分位匹配比例",
    )
    v = csv("ranking_people").groupby("source")[["R1", "chance", "MRR", "chance_MRR"]].mean()
    v.columns = ["首位正确率", "首位机会", "倒数排名", "倒数排名机会"]
    bars(ax[0, 1], v, "跨模态：同条件同记录候选的配对检索", "按人平均的分数")
    r.page(
        "ranking",
        "严格匹配后，跨人和跨模态检索都缺乏稳定优势",
        "左图以开发片段查询训练人图库，要求同库、同位置、RMS 比在 1/1.4–1.4 之间且至少 5 候选。命中定义为分支 α 占比或 HbO 斜率位于同一训练四分位。右图须找出同一个双时钟试次；负例同人同会话同条件，两侧均相距至少 60 秒。",
        fig,
        "跨模态两个方向各有 22 个合格查询，来自 5 人。按人等权，EEG→Hb 首位正确率 0.162、Hb→EEG 为 0.262，机会值约 0.324。倒数排名的随机期望按每次图库大小单独计算。源表还给出精确排名、同模态平均精度和放宽位置幅度匹配后的结果。",
        "{eeg,hb}_matched_retrieval.csv、_retrieval.csv；paired_retrieval.csv、ranking_people.csv",
    )
    fig, ax = panel(1, 2)
    f = csv("fusion").query("mode=='clean'")
    order = ["eeg", "hb", "late", "linear", "nonlinear_main", "interaction"]
    names = ["EEG", "Hb", "后期平均", "联合线性", "非线性主效应", "加交互"]
    for i, (metric, title) in enumerate(
        [("ba", "平衡准确率，越高越好"), ("brier", "Brier 概率误差，越低越好")]
    ):
        pv = f.pivot(index="model", columns="subject", values=metric).loc[order]
        for j, (_, row) in enumerate(pv.iterrows()):
            ax[0, i].scatter(np.arange(len(row)) * 0.035 + j - 0.075, row, s=16, color=GRAY)
            ax[0, i].scatter(j, row.mean(), marker="D", s=47, color=BLUE)
        ax[0, i].set(xticks=range(len(names)), xticklabels=names, title=title, ylabel="每人评分")
        ax[0, i].tick_params(axis="x", rotation=20)
    r.page(
        "fusion",
        "心算分类中，联合线性头的点估计高于后期平均，配对区间跨零",
        "同一 50 个开发试次、5 位被试。每个灰点为一人，菱形为人均。MA/BL 两类召回率平均得到平衡准确率；Brier 为分类概率的平方误差。所有简单头在训练人上拟合，仅使用冻结词族的区域和时间分布。",
        fig,
        "线性联合为 0.670，EEG 单模态 0.595，Hb 单模态 0.570，后期平均 0.637。联合相对后期平均的配对差 +0.0325，95% 区间 [−0.0425,0.1200]。等维随机投影非线性主效应上改用跨模态积，差 +0.0158，区间也跨零。后期平均的 Brier 反而较低。",
        "fusion.csv；fusion_contrasts.csv；paired_panel.csv",
    )
    fig, ax = panel(1, 2)
    f = csv("fusion")
    modes = ["clean", "eeg_missing", "hb_missing", "eeg_noise", "hb_noise", "eeg_drop", "hb_drop"]
    models = ["eeg", "hb", "late", "linear", "interaction"]
    for i, metric in enumerate(["ba", "brier"]):
        v = f.groupby(["mode", "model"])[metric].mean().unstack().loc[modes, models]
        v.index = ["完整", "缺 EEG", "缺 Hb", "EEG 噪声", "Hb 噪声", "掉 EEG 后区", "掉 Hb 后区"]
        v.columns = ["EEG", "Hb", "后期", "线性", "交互"]
        heat(ax[0, i], v, ["平衡准确率（高较好）", "Brier（低较好）"][i])
    r.page(
        "robust",
        "冻结分类头对整模态缺失、后区掉落和噪声的反应不同",
        "完整训练后不再拟合。整模态缺失时用训练均值替换该模态所有位置与 S/O；后区掉落只替换两个后部区域的特征。噪声直接加入各模态原始输入后重新编码，尺度为各 patch 波动的 25%。",
        fig,
        "同一批 5 位开发被试在每个条件下配对评分。单模态列提供对应参照；被试较少，表中均值差不能当作设备失效时的精度保证。附表还列出仅 S 或仅 O 的完整输入评分，以及非线性主效应基线。",
        "fusion.csv；fusion_contrasts.csv",
    )


def paired_illustration(r):
    from matplotlib.colors import ListedColormap

    q = csv("paired_example")
    z = np.load(D / "paired_example.npz")
    v = q.iloc[0]
    fig, ax = panel(3, 1)
    for i, (m, clr, lab) in enumerate([("eeg", 0, "EEG"), ("hb", 0, "HbO"), ("hb", 1, "HbR")]):
        for key, c, style, txt in [("X", GRAY, "-", "实测"), ("X_decoded", BLUE, "-", "整数解码")]:
            a = z[m + key].reshape(-1, 1 if m == "eeg" else 2)[:, clr]
            ax[i, 0].plot(
                np.arange(len(a)) / (200 if m == "eeg" else 10), a, lw=1, color=c, ls=style, label=txt
            )
        mark = v.eeg_cue - v.eeg_start if m == "eeg" else v.hb_cue - v.hb_start
        ax[i, 0].axvline(mark, ls=":", c=ORANGE, label="心算提示")
        ax[i, 0].set(xlim=(0, 30), ylabel=lab + " / SD")
        ax[i, 0].legend(ncol=3, loc="upper right", fontsize=8)
    ax[-1, 0].set_xlabel("相对各模态已对齐窗起点（秒）")
    r.page(
        "paired_example",
        "同一心算试次的双时钟原始观测和整数解码",
        "在 19 个共同支持的开发 MA 窗中，按四区实测 EEG α 占比选择最接近中位数的一窗。Single-Trial，subject_03，session_01，事件 17；左前区域的实际位置为 EEG F3、Hb FC3FC1。",
        fig,
        "EEG 与 Hb 分别从记录中 510.055 秒和 567.200 秒起取 30 秒，提示在各自窗内约 0.97 秒。蓝线由完整 S 与 O 整数流解码再相加。三行各用本位置的训练输入 SD，不跨行比较绝对幅值。这个固定实例用于解释配对、时钟与重建，不参与事件主端点选择。",
        "paired_example.csv；paired_example.npz；paired_panel.csv；positions.csv",
    )
    fig, ax = panel()
    a = ax[0, 0]
    flow = ["eegS", "eegO", "hbS", "hbO"]
    fam = np.stack([z[k + "_family"] for k in flow])
    a.imshow(
        fam,
        extent=(0, 30, 3.5, -0.5),
        aspect="auto",
        cmap=ListedColormap(["#e0edf1", "#a9d4c5", "#eac77f", "#ac788e"]),
        vmin=-0.5,
        vmax=3.5,
    )
    for i, k in enumerate(flow):
        for j, idx in enumerate(z[k + "_id"]):
            a.text(j * 2 + 1, i, str(idx), ha="center", va="center", fontsize=9)
    a.set(
        xticks=np.arange(0, 31, 2),
        yticks=range(4),
        yticklabels=["EEG-S", "EEG-O", "Hb-S", "Hb-O"],
        xlabel="窗内时间（秒）",
    )
    a.grid(axis="x", lw=0.7, color="white")
    a.set_title("每格的整数属于对应分支独立的词表")
    r.page(
        "paired_example",
        "一个 30 秒配对窗实际发出的 60 个整数",
        "同一试次、同一左前区域。每行 15 个 2 秒位置。底色由浅蓝、绿、黄、紫分别代表本分支训练定义的 Q1 至 Q4；整数是该分支真实输出的 ID。",
        fig,
        "EEG 各支用自己的 8192×128 码本，Hb 各支用自己的 1024×128 码本。相同底色表示各自标量属性处于相同分位；跨分支数值相同的 ID 没有共同编号语义。每个整数依赖整窗编码，下一节的原信号替换检验直接量化这种影响。",
        "paired_example.csv；paired_example.npz；{eeg,hb}_17/{S,O}_family.json",
    )


def sensitivity_pages(r):
    fig, ax = panel(1, 2)
    for i, m in enumerate(M):
        d = csv(m + "_visibility")
        z = macro(d, ["branch", "distance_s"], ["raw_changed", "idmask_visible_changed"]).reset_index()
        for b in "SO":
            g = z[z.branch.eq(b)]
            ax[0, i].plot(
                g.distance_s, 100 * g.raw_changed, marker="o" if b == "S" else "^", label=b + " 先换原信号"
            )
        ax[0, i].plot([2, 14], [0, 0], ":", c=GRAY, label="编码后仅遮中心 ID")
        ax[0, i].set(xlabel="可见邻居距被替换中心（秒）", ylabel="可见邻居 ID 改变（%）", title=MOD[m])
        ax[0, i].legend(fontsize=9)
    r.page(
        "visibility",
        "先替换原始中心片段，会改变本应仍然可见的邻居 ID",
        "固定同一个真实 30 秒窗，只把中心 2 秒换成同人、同电极或光学位置、同记录的远时段供体。重新编码后逐距离比较左右邻居，配对参照为先用完整窗编码、再只遮掉中心 ID。",
        fig,
        "距中心 2 秒的邻居中，EEG-S 和 Hb-O 约有 30% 和 87% 的整数受影响；后编码遮挡保持其余已生成整数原样。该结果量化冻结输入的可见性，提示掩码设计必须声明是遮原信号还是遮整数。联合模型训练后的实际利用程度仍需其输出。",
        "{eeg,hb}_visibility.csv；{eeg,hb}_window_shift.csv",
    )
    fig, ax = panel(1, 2)
    for i, m in enumerate(M):
        z = (
            macro(csv(m + "_directed_edit"), ["kind", "branch"], ["decoded_delta"])
            .decoded_delta.unstack()
            .reindex(["lowest", "random", "highest"])
        )
        z.index = ["最低属性", "随机供体", "最高属性"]
        bars(
            ax[0, i],
            z,
            MOD[m] + " 换一词后的中心属性变化",
            "α 占比变化" if m == "eeg" else "HbO 斜率变化（SD/秒）",
        )
    r.page(
        "directed",
        "按词族属性定向编辑时，Hb 斜率比 EEG α 占比更容易定向改变",
        "固定原窗及周围 14 个整数，供体取同人、同记录、同位置的远时段实际词，限制训练词频比在 1/2–2 之间。至少有两个不同候选才评分。三组在完全相同的合格中心上选属性最高、随机及最低供体。",
        fig,
        "Hb 两支中，最高分供体平均提高解码 HbO 斜率，最低分供体降低斜率。EEG-S 的解码 α 占比没有遵从相同方向。这里干预的是解码器的输入；真实被试没有接受生理干预。逐例表保留训练属性差、真实整数、目标位置及非中心变化。",
        "{eeg,hb}_directed_edit.csv；{eeg,hb}_edit.csv",
    )
    fig, ax = panel(1, 2)
    for i, m in enumerate(M):
        q = csv(m + "_order_budget").query("adjacent")
        t = macro(q, ["branch", "method"], ["mse"]).mse.unstack()
        reg = csv(m + "_17/sequence").query("adjacent and reference=='bag'").set_index("branch").gain
        df = pd.DataFrame({"完整特征": 100 * reg, "同为四维": 100 * (1 - t.bigram_dwell / t.bag)})
        bars(ax[0, i], df, MOD[m] + " 二元+驻留 相对词袋", "相邻窗预测误差下降（%）")
    r.page(
        "budget",
        "给时序读出相同的四维输入后，Hb 的顺序增益大幅缩小",
        "左、右图分别为 EEG 和 Hb。蓝色保留完整二元及驻留描述符；橙色在训练历史上标准化，并仅保留前四个奇异向量，对齐四维词袋的读出输入宽度。相同训练/开发窗、预测目标及岭强度。",
        fig,
        "Hb-S 的相邻窗增益从完整描述符的 4.60% 缩小至约 0.14%。这个对照同时限制维度并进行无监督投影，因此可确认收益对特征表达方式敏感。S/O 交互另用相同 16 个特征槽比较非线性主效应与跨分支词族积，对实测 X 的误差差异很小。",
        "{eeg,hb}_order_budget.csv、_interaction_budget.csv；{eeg,hb}_17/sequence.csv",
    )
    fig, ax = panel(1, 2)
    f = csv("fusion_pair_controls")
    for i, metric in enumerate(["ba", "brier"]):
        z = (
            f.groupby("method")[metric]
            .mean()
            .reindex(
                [
                    "eeg",
                    "hb",
                    "continuous",
                    "aligned",
                    "refit_mismatch",
                    "frozen_mismatch",
                    "offset_-10s",
                    "offset_10s",
                ]
            )
        )
        z.index = ["EEG", "Hb", "连续", "对齐", "重拟错配", "冻结错配", "−10 秒", "+10 秒"]
        bars(
            ax[0, i],
            z.to_frame("人均评分"),
            "平衡准确率" if i == 0 else "Brier 概率误差",
            "越高越好" if i == 0 else "越低越好",
            25,
        )
    n = f.query("method=='aligned'")
    eligible = int(n.n.sum())
    den = n.subject.nunique()
    r.page(
        "pair_null",
        "在同人同条件内换配对，检验联合分类是否依赖对应试次",
        f"限制到有双时钟分离至少 60 秒的同人、会话、条件供体；开发 {den} 人、{eligible} 个试次。对齐、重拟错配、冻结输入错配、连续摘要与单模态头共享面板。",
        fig,
        "错配保留类别，能检验模型是否还依赖该试次的细粒度对应关系。±10 秒将 Hb 的时段特征沿窗移动一格，以最近端点填充，属于特征时序敏感性分析。平衡准确率只对具备两类的被试有定义，Brier 保留全部，分母见源表。",
        "fusion_pair_controls.csv；fusion_pair_support.csv",
    )


def catalogue():
    matrix = ROOT / "docs/physiology_evidence_matrix.xlsx"
    with ZipFile(matrix) as archive:
        tree = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    ws = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    records = []
    for row in tree.findall(".//" + ws + "row"):
        values = [c.findtext(ws + "v", "") for c in row.findall(ws + "c")]
        if values and re.fullmatch(r"A\d{2}", values[0]):
            records.append(values)
    cols = [
        "ID",
        "类别",
        "原工作簿主张",
        "原分析方法",
        "原对照",
        "原指标与单位",
        "最早阶段",
        "优先级",
        "原历史状态",
        "来源",
    ]
    original = pd.DataFrame(records, columns=cols)
    add = pd.read_csv(audit.CONTRACT.with_name("aligned_token_audit_catalog.tsv"), sep="\t")
    assert original.ID.tolist() == add.ID.tolist() == [f"A{i:02}" for i in range(1, 65)]
    merged = original.merge(add, on="ID", validate="one_to_one")

    def expand(s):
        files = []
        for p in s.split("|"):
            files.extend([p.format(m=m) for m in M] if "{m}" in p else [p])
        assert all((D / p).is_file() for p in files), files
        return "|".join(files)

    merged["证据文件"] = merged["证据文件"].map(expand)
    return merged


def appendix(r, df):
    r.table(
        "scope",
        "哪些结论已经可以使用，哪些需要新的证据？",
        "下列条件与工作簿的阶段相对应。T 为固定 tokenizer 后的统计或简单读出；T→P 的本轮结果只回答冻结基线。P 指联合遮挡模型产生的隐藏表示或补全输出。",
        ["可回答的问题", "本轮判读", "还缺少的主要条件"],
        [
            (
                "真实词义及波形",
                "可度量属性偏好、上下文影响及真实成员差异；Hb 局部斜率与低阶顺序证据强于 EEG。",
                "跨人稳定的生理类别命名仍需外部独立标注，特别是尖波、慢波和自然觉醒。",
            ),
            (
                "跨模态增量",
                "与强自身历史相比，冻结 EEG 词在三人实测 Hb 预测上没有正增益；共同匹配的检索也未超过机会。",
                "需要更多独立被试、可用系统生理协变量，并锁定事件、时窗、评价指标。",
            ),
            (
                "已备盲评材料",
                "32 个匿名呈现项目，含训练分层成员、开发失败成员以及同背景对照；解释键与评分表分开。",
                "两位独立评阅者打分后才可计算重复性。",
            ),
            (
                "联合模型所特有的能力",
                "冻结单模态、后期平均、简单联合头、错配及时间对照构成基线。",
                "A48/49/51/52/53/63 需要联合模型或补全结果；本次没有该类权重或输出。",
            ),
            (
                "外部验证",
                "本轮所有数字均来自已有训练与普通开发人。四库间差异与历史种子作为开发稳定性检查。",
                "A47/55/56/57 需要独立队列、前瞻干预、确认协议或额外测量。",
            ),
        ],
        "随后按原工作簿逐项索引。“已计算”描述已执行的冻结端点，包含负结果；“有限支持”表示已有数值而目标主张的一项或多项对照、样本量或标签仍受限。",
    )
    df["正文页"] = df["报告节"].map(r.routes)
    assert df["正文页"].notna().all()
    for start in range(0, len(df), 6):
        part = df.iloc[start : start + 6]
        rows = [
            (a["ID"], a["原工作簿主张"], a["状态"], a["本轮证据与解释边界"], a["正文页"])
            for _, a in part.iterrows()
        ]
        r.table(
            "index",
            f"逐项证据索引　{part.ID.iloc[0]}–{part.ID.iloc[-1]}",
            "“正文页”指向相关分析的首个说明页。可筛选工作簿保留全部原设计列，新增本轮执行状态、对应评分表和具体限制。",
            ["编号", "待检验主张", "本轮状态", "已测范围、未满足条件", "正文页"],
            rows,
            "同一结果可服务多个彼此不同的主张。阳性、负结果、样本不足及阶段依赖均保留；新增实验不改写原工作簿中的历史结果。",
        )


def bundle(r, df):
    dest = r.out
    df.to_csv(dest / "evidence.csv", index=False, encoding="utf-8-sig")
    folder = dest / "tables"
    folder.mkdir()
    raw = [f for f in D.iterdir() if f.is_file() and f.suffix == ".csv" and f.name != "blinding_key.csv"]
    for m in M:
        raw += list((D / f"{m}_17").glob("*.csv*")) + list((D / f"{m}_17").glob("*_member_waves.npz"))
        for b in "SO":
            raw += list((D / f"readout_{m}_17_{b}").glob("*.csv")) + list(
                (D / f"readout_{m}_17_{b}").glob("*.json")
            )
    raw += list((D / "events_17").glob("*.csv*")) + list((D / "events_17").glob("*.json"))
    raw += list(D.glob("paired_example.npz")) + list(D.glob("*_17/?_family.json"))
    raw += [f for f in D.glob("*_status.json")] + [
        D / "cue_permutation.json",
        D / "resolved.yaml",
        D / "resources.log",
        P / "summary.json",
        P / "pairing.json",
    ]
    ev = P / "eeg/models/eeg__waveform__vq8192/evaluation"
    raw += list(ev.glob("*_waveforms.npz")) + list(ev.glob("*_fixed_windows.json"))
    for f in raw:
        p = (
            (folder / f.relative_to(D))
            if f.is_relative_to(D)
            else folder / "aligned_parent" / f.relative_to(P)
        )
        p.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(f, p)
    shutil.copytree(D / "blind", folder / "blind")
    links = sorted(f.relative_to(folder).as_posix() for f in folder.rglob("*") if f.is_file())
    (folder / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>可复核评分表</title><h1>可复核评分表</h1><ul>'
        + "".join(f'<li><a href="{escape(k)}">{escape(k)}</a></li>' for k in links)
        + "</ul>"
    )
    book = xlsxwriter.Workbook(dest / "evidence.xlsx", {"strings_to_urls": False})
    ws = book.add_worksheet("64项逐项核对")
    fmt = book.add_format({"text_wrap": True, "valign": "vcenter", "font_size": 10})
    header = book.add_format(
        {"bg_color": "#174D63", "font_color": "white", "bold": True, "text_wrap": True}
    )
    c = [
        "ID",
        "类别",
        "原工作簿主张",
        "状态",
        "本轮证据与解释边界",
        "正文页",
        "证据文件",
        "最早阶段",
        "优先级",
        "原分析方法",
        "原对照",
        "原指标与单位",
        "原历史状态",
        "来源",
    ]
    ws.write_row(0, 0, c, header)
    for i, row in df.iterrows():
        ws.write_row(i + 1, 0, [row[k] if k != "证据文件" else row[k].replace("|", "\n") for k in c], fmt)
        ws.set_row(i + 1, 74)
    for i, width in enumerate([7, 15, 33, 15, 69, 10, 60, 12, 10, 40, 42, 37, 50, 45]):
        ws.set_column(i, i, width)
    ws.autofilter(0, 0, len(df), len(c) - 1)
    ws.freeze_panes(1, 3)
    idx = book.add_worksheet("证据文件")
    idx.write_row(0, 0, ["文件路径（解压后可打开）"], header)
    idx.set_column(0, 0, 115)
    idx.freeze_panes(1, 0)
    for i, k in enumerate(links, 1):
        idx.write_url(i, 0, "external:tables/" + k, string=k)
    book.close()
    with ZipFile(dest / "report_bundle.zip", "w", ZIP_DEFLATED, compresslevel=5) as z:
        for f in sorted(dest.rglob("*")):
            if f.is_file() and f.name != "report_bundle.zip":
                z.write(f, f.relative_to(dest))


def main():
    import torch

    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=D / "report_v2")
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.exists():
        raise FileExistsError("Retain completed export and choose a fresh version: " + str(dest))
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
        }
    )
    df = catalogue()
    with TemporaryDirectory(dir=dest.parent, prefix=".building-report-") as tmp:
        path = Path(tmp) / "report"
        path.mkdir()
        r = Report(path)
        for work in [
            overview,
            reconstruction,
            paired_illustration,
            members,
            vocabulary,
            local_pages,
            spatial_event_pages,
            sensitivity_pages,
        ]:
            work(r)
            print(work.__name__, len(r.heads), flush=True)
        appendix(r, df)
        r.finish()
        bundle(r, df)
        path.rename(dest)
    print("Delivered", dest)


if __name__ == "__main__":
    main()
