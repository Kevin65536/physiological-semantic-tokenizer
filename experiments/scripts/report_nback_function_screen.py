"""Render the n-back decisions with native-time examples, family rasters and paired observations."""

from html import escape
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.metrics import confusion_matrix

from run_nback_function_screen import CFG, OUT, alpha_power, filter2, pulse, read
from score_nback_function_screen import family_lut
from src.utils.io import write_json

LABELS = ["0-back", "2-back", "3-back"]
C = ["#257987", "#D59030", "#964C74"]
METHOD = {
    "continuous_E": "连续 EEG",
    "continuous_H": "连续 Hb",
    "continuous_EH": "连续联合",
    "hybrid_LDA": "联合判别分析",
    "token_E": "EEG 词元",
    "token_H": "Hb 词元",
    "token_EH": "联合词元",
    "theta_alpha": "额 θ ＋顶 α",
    "pre_E": "量化前 EEG",
    "pre_EH": "量化前联合",
    "X": "观测 X",
    "HP": "2 Hz 高通",
    "S": "解码 S",
    "O": "解码 O",
    "SO": "解码 S＋O",
}
FIG = OUT / "report_v2"
plt.rcParams.update(
    {
        "font.family": "Noto Sans CJK JP",
        "font.size": 10.5,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "savefig.facecolor": "white",
    }
)


def canvas(n, m, **kwargs):
    return plt.subplots(n, m, figsize=(12.6, 5.25), layout="constrained", **kwargs)


def page(doc, slug, title, lead, fig, caption, source):
    pg = doc.new_page(width=1000, height=706)

    def txt(box, s, size, color="#173743"):
        h, _ = pg.insert_htmlbox(
            fitz.Rect(*box),
            escape(s),
            css=f"*{{font-family:sans-serif;font-size:{size}pt;line-height:1.34;color:{color}}}",
            scale_low=1,
        )
        if h < 0:
            raise ValueError("Text overflow " + s[:35])

    txt(
        (28, 16, 976, 39),
        f"认知负荷　/　模型筛选实验　　　2026 年 11 月 27 日　　　　　　　　　{len(doc):02}",
        11,
        "#176D71",
    )
    txt((28, 42, 974, 85), title, 23)
    txt((28, 92, 972, 149), lead, 13)
    path = FIG / (slug + ".png")
    fig.savefig(path, dpi=250)
    plt.close(fig)
    pg.insert_image(fitz.Rect(28, 151, 972, 545), filename=str(path))
    txt((28, 554, 972, 660), caption, 12)
    txt((28, 671, 972, 698), source, 9, "#276F71")


def heat(ax, data, fmt, *, vmin=0, vmax=1, cmap="Blues"):
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    for i, j in np.ndindex(data.shape):
        rgba = im.cmap(im.norm(data[i, j]))
        lum = np.dot(rgba[:3], [0.299, 0.587, 0.114])
        ax.text(
            j,
            i,
            format(data[i, j], fmt),
            ha="center",
            va="center",
            color="white" if lum < 0.49 else "#153139",
        )
    return im


def material():
    m = (
        pd.read_csv(OUT / "support.csv")
        .query("status=='complete' and split=='validation'")
        .reset_index(drop=True)
    )
    a = {}
    for sub, g in m.groupby("subject", sort=False):
        with np.load(OUT / "subjects" / (sub + ".npz")) as w:
            np.testing.assert_array_equal(g.id, w["id"])
            for k in ("h", "e", "ids", "dirty_ids", "wave_S", "dirty_S", "bands_X"):
                a.setdefault(k, []).append(w[k])
    a = {k: np.concatenate(v) for k, v in a.items()}
    return m, a


def physiology(doc, m, a):
    band = a["bands_X"] * 10 / np.log(10)
    centered = band - band[:, :, :2].mean(2, keepdims=True)
    # Mean of log-power over matched sites/slots; dB relative to the two pre-onset patches.
    meas = np.stack(
        [
            centered[:, :2, 5:, 1].mean((1, 2)),
            centered[:, 2:, 5:, 2].mean((1, 2)),
            a["h"][:, :2, 100:, 0].mean((1, 2)),
        ],
        axis=1,
    )
    fields = ["frontal_theta_db", "parietal_alpha_db", "frontal_HbO_uM"]
    tab = pd.concat([m[["subject", "condition", "id"]], pd.DataFrame(meas, columns=fields)], axis=1)
    tab.to_csv(FIG / "physiology_blocks.csv", index=False)
    g = tab.groupby(["subject", "condition"])[fields].mean()
    diff = g.xs(CFG["classes"][1], level="condition") - g.xs(CFG["classes"][0], level="condition")
    av = diff.mean()
    fig, axs = canvas(1, 3)
    for k, (ax, key, title) in enumerate(
        zip(axs, fields, ["额部 θ 功率变化 · dB", "顶部 α 功率变化 · dB", "额部 HbO 变化 · μM"])
    ):
        q = g[key].unstack().reindex(columns=CFG["classes"])
        for i, (sub, vals) in enumerate(q.iterrows()):
            ax.plot(
                np.arange(3) + (i - 1.5) * 0.035,
                vals,
                marker=["o", "s", "^", "D"][i],
                lw=1,
                color=["#627d95", "#c17f43", "#9994a8", "#397c74"][i],
                label=sub,
            )
        ax.axhline(0, color="#333333", lw=0.7, ls=":")
        ax.set_xticks(range(3), LABELS)
        ax.set_title(title)
        if k == 0:
            ax.legend(fontsize=8, ncol=2)
        ax.set_ylabel("相对任务前参考")
        ax.grid(axis="y", alpha=0.15)
    page(
        doc,
        "01_physiology",
        "最值得复核的线索出现在额部 θ 和顶部 α",
        f"从 0-back 到 2-back，4 人的额部 θ 变化都更高、顶部 α 变化都更低；被试等权差分别为 {av.iloc[0]:+.2f} 和 {av.iloc[1]:+.2f} dB。同期额部 HbO 差为 {av.iloc[2]:+.2f} μM，4 人方向一致。",
        fig,
        "每条线连接同一人的三种实验条件。先对每个块求任务后 5–25 秒均值，再在个人、条件内平均。额部用 F1/F2 与 AFF3h/AFF4h，顶部用 P3/P4。θ 为 4–8 Hz，α 为 8–13 Hz；EEG 以 −5 至 −1 秒对数功率为参考，HbO 以 −5 至 0 秒为参考。仅有四名开发被试，块序与负荷完全对应（见第 4 页）。",
        "依据　physiology_blocks.csv · 各人的记录为 cnt_nback · 传感器区域结果，完整离线预处理。",
    )


def classification(doc, m, a, preds, people):
    keys = ["continuous_E", "continuous_H", "continuous_EH", "hybrid_LDA", "token_E", "token_H", "token_EH"]
    fig, axs = canvas(1, 3, gridspec_kw={"width_ratios": [1.65, 1, 1]})
    ba = (
        people.query("input=='clean'").pivot(index="method", columns="subject", values="ba").loc[keys] * 100
    )
    ba["均值"] = ba.mean(axis=1)
    heat(axs[0], ba.to_numpy(), ".0f", vmax=80, cmap="BuGn")
    axs[0].set_yticks(range(len(keys)), [METHOD[k] for k in keys])
    axs[0].set_xticks(range(5), [v.replace("VP", "") for v in ba.columns])
    axs[0].set_title("各被试平衡准确率（%）")
    axs[0].set_xlabel("开发被试编号")
    for ax, k in zip(axs[1:], ["continuous_EH", "token_EH"]):
        g = preds.query("method==@k and input=='clean'")
        mat = confusion_matrix(g.true, g.pred, labels=[0, 1, 2])
        heat(ax, mat, "d", vmax=max(1, mat.max()))
        ax.set_xticks(range(3), ["0", "2", "3"])
        ax.set_yticks(range(3), ["0", "2", "3"])
        ax.set_title(METHOD[k] + " · 块数")
        ax.set_xlabel("模型判定（back）")
        ax.set_ylabel("原始标记（back）")
    page(
        doc,
        "02_task",
        "连续 EEG 可读出部分负荷，加入 Hb 后平均成绩持平",
        "15 人的 309 块训练读出，另 4 人的 91 块用于评价。连续 EEG 为 51.5%，连续联合为 51.2%。联合词元为 33.7%，四人都低于连续联合；三类均匀猜测参考值为 33.3%。",
        fig,
        "模型输入同为 −5 至 +25 秒。三段连续 10 秒分别汇总四位置的五个 EEG 频带对数功率、双色 Hb 均值，或 S/O 硬词元的 128 维码本向量。以三个训练被试折选择岭系数，另设收缩线性判别分析。左图先在各人内等权平均三类召回率，再等权平均被试；右图汇总全部开发块，斜对角为正确判断。输入不含标签、时钟或块序号。",
        "依据　task_predictions.csv · task_people.csv · selection.csv　｜　上游 437 个候选块保留 400 个，本次切窗 400/400 完整。",
    )


def cases(doc, m, a, preds):
    wide = preds.query("input=='clean'").pivot(index="id", columns="method", values="pred").reindex(m.id)
    true = m.condition.map(CFG["classes"].index).to_numpy()
    tok = wide.token_EH.to_numpy() == true
    cont = wide.continuous_EH.to_numpy() == true
    ids = [np.flatnonzero(v)[0] for v in (tok & ~cont, ~tok & cont)]
    fig, axs = canvas(3, 2, gridspec_kw={"height_ratios": [1.15, 1, 0.7]})
    spec = []
    for i in ids:
        f, p = welch(a["e"][i, 2].reshape(15, 400), fs=200, nperseg=200, noverlap=100)
        spec.append(10 * np.log10(np.maximum(p[:, (f >= 2) & (f <= 30)], 1e-12)).T)
    lo = min(v.min() for v in spec)
    hi = max(v.max() for v in spec)
    families = family_lut()
    for j, i in enumerate(ids):
        r = m.iloc[i]
        c = wide.iloc[i]
        axs[0, j].set_title(
            f"{r.subject} · 块 {r.event} · 实测标签 {LABELS[true[i]]}\n连续联合判 {LABELS[c.continuous_EH]}，联合词元判 {LABELS[c.token_EH]}",
            fontsize=11,
        )
        im = axs[0, j].imshow(
            spec[j],
            origin="lower",
            aspect="auto",
            extent=(-5, 25, 1.5, 30.5),
            cmap="magma",
            vmin=lo,
            vmax=hi,
        )
        axs[0, j].set_ylabel("P3 频率（Hz）")
        h = a["h"][i, :2].mean(0)
        for k, (lab, col, ls) in enumerate([("HbO", "#b85c47", "-"), ("HbR", "#315c90", "--")]):
            axs[1, j].plot(np.arange(300) / 10 - 5, h[:, k], color=col, ls=ls, label=lab)
        axs[1, j].legend(ncol=2, fontsize=8)
        axs[1, j].set_ylabel("额部 ΔHb（μM）")
        fam = families[a["ids"][i, :, 0]]
        axs[2, j].imshow(
            fam,
            cmap=ListedColormap(["#e9f1f0", "#a0c9bf", "#458d83", "#154f55"]),
            extent=(-5, 25, 3.5, -0.5),
            vmin=-0.5,
            vmax=3.5,
            aspect="auto",
        )
        axs[2, j].set_yticks(range(4), CFG["sites"]["eeg"], fontsize=7)
        axs[2, j].set_xlabel("相对任务标记（秒）")
        for ax in axs[:, j]:
            ax.axvline(0, color="gray", ls=":", lw=1)
            ax.set_xlim(-5, 25)
    lim = max(abs(a["h"][ids, :2].mean(1)).max(), 1e-5)
    for ax in axs[1]:
        ax.set_ylim(-lim * 1.03, lim * 1.03)
    fig.colorbar(im, ax=axs[0].tolist(), shrink=0.9, label="dB re 1 μV²/Hz")
    page(
        doc,
        "03_cases",
        "同一批任务块里，两种联合读出都有独自判对的样例",
        f"91 块中，联合词元独自判对 {sum(tok&~cont)} 块，连续联合独自判对 {sum(~tok&cont)} 块；另有 {sum(tok&cont)} 块两者都对、{sum(~tok&~cont)} 块都错。左右各取对应类别中按被试编号、块序号排序的第一例。",
        fig,
        "上排是实际 EEG 的 2 秒窗功率谱，中排是左右额部平均的处理后实测 HbO/HbR，下排是 EEG-S 每 2 秒得到的状态族。四种由浅到深的色阶对应训练词典按 α 占比分出的 Q1 到 Q4；词元读入整段 30 秒。虚线标记任务起点。两例使用共同的频谱色标和血红蛋白纵轴；个案展示模型判断发生在哪种观测上。",
        "依据　subjects/ · task_predictions.csv · four_stream_atlas 的训练词典　｜　Hb 单位为 μM，未将预测分数解释为生理量。",
    )


def blockcheck(doc, m):
    directory = OUT / "order_check_v1"
    counts = pd.read_csv(directory / "ordinal_training_counts.csv", index_col=0)
    b = pd.read_csv(directory / "people.csv").groupby(["scope", "method"]).ba.mean().unstack(0)
    keys = ["theta_alpha", "continuous_E", "continuous_EH", "pre_E", "token_E", "pre_EH", "token_EH"]
    fig, axs = canvas(2, 1, gridspec_kw={"height_ratios": [0.8, 2.5]})
    mat = np.full((3, 27), np.nan)
    mat[:, counts.index.to_numpy()] = counts.to_numpy().T
    cm = plt.get_cmap("Blues").copy()
    cm.set_bad("#d6d7d6")
    axs[0].imshow(mat, aspect="auto", cmap=cm, vmin=0, vmax=15)
    axs[0].set_yticks(range(3), LABELS)
    axs[0].set_xticks(range(0, 27, 2))
    axs[0].set_title("训练集中每个块序号只出现一种负荷（蓝色），灰列为没有入组的序号")
    for boundary in (8.5, 17.5):
        axs[0].axvline(boundary, color="#a94534")
    for j, (scope, c, lab) in enumerate(
        [("people", "#327f83", "仅留出被试"), ("people_and_range", "#b96c40", "同时留出被试与块序区间")]
    ):
        values = b.loc[keys, scope].to_numpy() * 100
        bars = axs[1].bar(np.arange(len(keys)) + (j - 0.5) * 0.36, values, 0.34, color=c, label=lab)
        axs[1].bar_label(bars, fmt="%.1f", fontsize=9, padding=2)
    axs[1].set_xticks(range(len(keys)), [METHOD[k] for k in keys])
    axs[1].axhline(100 / 3, color="#4b4f52", ls=":", lw=1)
    axs[1].set_ylim(0, 64)
    axs[1].set_ylabel("平衡准确率（%）")
    axs[1].legend(ncol=2, loc="upper right")
    page(
        doc,
        "04_order_check",
        "固定块序能完全识别标签，额顶两项特征值得换顺序复核",
        "用训练被试的块序号查表，91 个开发块全部判对。进一步同时留出被试和块序区间，额 θ＋顶 α 两个功率变化量达到 48.5%，连续 EEG 为 49.7%，联合词元为 29.4%。",
        fig,
        "在看到固定块序后追加此诊断。把原始块号分成 0–8、9–17、18–26，轮流用训练被试的另外两段拟合，在开发被试的留出段评分，再合并逐块预测。两项特征的参考段与第 1 页相同。量化前后使用相同的时间聚合、维数和读出。联合量化前为 35.9%，硬词元为 29.4%；单独 EEG 量化前也较弱。实验时序仍固定，负荷与顺序的生理贡献需要另用平衡顺序的数据区分。",
        "依据　order_check_v1/ordinal_training_counts.csv · predictions.csv · people.csv　｜　探索性追加，原筛选门槛保持冻结。",
    )


def sequence(doc, m, a):
    resp = pd.read_csv(OUT / "response_predictions.csv")
    nm = resp.groupby(["subject", "method"]).nmse.mean().unstack()
    fig, axs = canvas(1, 3, gridspec_kw={"width_ratios": [1.25, 1, 1.25]})
    sel = np.flatnonzero(m.condition.eq("2-back session"))
    fam = family_lut()[a["ids"][sel, 0, 0, 3:7]]
    ix = np.lexsort(tuple(fam[:, i] for i in (3, 2, 1, 0)))
    sub = fam[ix]
    region_ids = m.id.iloc[sel[ix]]
    im = axs[0].imshow(
        sub + 1,
        cmap=ListedColormap(["#e1eeed", "#99c6b9", "#439283", "#194d54"]),
        vmin=0.5,
        vmax=4.5,
        aspect="auto",
        interpolation="nearest",
    )
    axs[0].set_xticks(range(4), ["1–3", "3–5", "5–7", "7–9"])
    axs[0].set_xlabel("起点后时间（秒）")
    axs[0].set_ylabel("2-back 块，按实际序列排序")
    axs[0].set_title(f"额左 F1 · {len(sel)} 个块")
    fig.colorbar(im, ax=axs[0], ticks=range(1, 5), shrink=0.55, label="EEG-S 族 Q")
    oo = (
        resp.query("region=='frontal' and method=='order'")
        .pivot(index="id", columns="color", values="observed")
        .loc[region_ids]
    )
    lim = abs(oo.to_numpy()).max()
    hb = axs[1].imshow(
        oo.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest"
    )
    axs[1].set_xticks([0, 1], oo.columns)
    axs[1].set_yticks([])
    axs[1].set_title("同一块的额部实测响应")
    fig.colorbar(hb, ax=axs[1], shrink=0.65, label="晚段减早段（μM）")
    meth = ["continuous_occ", "order", "random_direction"]
    heat(axs[2], nm[meth].to_numpy(), ".2f", vmin=0, vmax=max(2, nm.max().max()), cmap="YlOrBr")
    axs[2].set_yticks(range(4), nm.index)
    axs[2].set_xticks(range(3), ["基线", "加顺序", "随机方向"])
    axs[2].set_title("四人完整面板 · NMSE")
    axs[2].set_xlabel("越低越好，含两区域、两色")
    page(
        doc,
        "05_sequence",
        "额部可以看到重复事件族，加入顺序后 Hb 误差反而增加",
        "用块开始后 1–9 秒的四个事件族估计 9–19 秒相对早段的 Hb 变化。连续信息与占用基线误差为 1.354，加入转移为 1.450，随机翻转序列方向为 1.447。加入顺序仅 1/4 人优于基线。",
        fig,
        "左中图每行对应同一个开发块，Q1–Q4 反映 EEG-S 词元的训练期 α 占比。左图只画 F1，读出使用额顶四位置。基线含负荷、早段 Hb、五频带功率、各族占用及游程长度分布。对照随机翻转一半左右的站点序列，保留计数及游程。每个目标按训练标准差归一化平方误差，再在块、区域和双色上平均，最后等权平均被试。整窗双向编码限制了时间因果解释。",
        "依据　response_predictions.csv · subjects/ · 训练词典　｜　每个晚段目标使用实测 HbO/HbR；开发块只用于评分。",
    )


def noise(doc, m, a, people):
    table = pd.read_csv(OUT / "fidelity.csv")
    f = table.groupby(["method", "subject"])[["transfer", "alpha_ratio"]].median().groupby("method").mean()
    s = table.query("method=='S'").sort_values(["alpha_ratio", "id"])
    case = s.iloc[len(s) // 2]
    i = int(np.flatnonzero(m.id == case.id)[0])
    r = m.iloc[i]
    t = np.arange(6000) / 200 - 5
    sel = (t >= 4.7) & (t <= 7.3)
    e = a["e"][i, 0].ravel()
    d = pulse()[0].ravel()
    s0 = a["wave_S"][i, 0].ravel()
    s1 = a["dirty_S"][i, 0].ravel()
    fig, axs = canvas(1, 3, gridspec_kw={"width_ratios": [1.2, 1.15, 1.15]})
    for y, lab, c, ls in (
        (e, "原观测 X", "#454952", "-"),
        (e + d, "加脉冲 X", "#bd683e", "-"),
        (s0, "原片段解码 S", "#2a8181", "-"),
        (s1, "加脉冲解码 S", "#133a62", "--"),
    ):
        axs[0].plot(t[sel], y[sel], label=lab, color=c, ls=ls, lw=1)
    axs[0].set_title(f"{r.subject} · 块 {r.event} · F1")
    axs[0].set_xlabel("任务标记后（秒）")
    axs[0].set_ylabel("电压（μV）")
    axs[0].legend(fontsize=8)
    for arr, lab, c, ls in [
        (a["e"][i, 2:], "原观测 X", "#454952", "-"),
        (a["wave_S"][i, 2:], "解码 S", "#248273", "-"),
        (filter2(a["e"][i, 2:]), "2 Hz 高通", "#c98939", "--"),
    ]:
        hz, ps = welch(arr.reshape(2, -1), fs=200, nperseg=400, axis=-1)
        mask = (hz >= 1) & (hz <= 40)
        axs[1].semilogy(hz[mask], ps.mean(0)[mask], color=c, label=lab, ls=ls)
    axs[1].axvspan(8, 13, color="#edcf83", alpha=0.35)
    axs[1].set_title("同一块的顶部频谱")
    axs[1].set_xlabel("频率（Hz）")
    axs[1].set_ylabel("功率谱密度（μV²/Hz）")
    axs[1].legend(fontsize=8)
    for marker, k in zip(["o", "^", "s", "D", "P"], ["X", "HP", "S", "O", "SO"]):
        axs[2].scatter(f.loc[k, "transfer"], f.loc[k, "alpha_ratio"], s=70, marker=marker, label=METHOD[k])
    axs[2].axhspan(0.8, 1.25, color="#dbe9e2")
    axs[2].axvline(0.5, color="#60676b", ls=":")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("脉冲 RMS 传入比")
    axs[2].set_ylabel("顶部 α 功率保留比")
    axs[2].set_title("四人汇总 · 左上方为目标")
    axs[2].legend(fontsize=8, loc="center right")
    page(
        doc,
        "06_noise",
        "解码 S 压低了注入脉冲，也大幅压低了实测 α 节律",
        "S 的脉冲传入比为 20.7%，顶部 α 功率保留比为 0.068%。常规 2 Hz 高通分别为 7.8% 和 100.0%（四舍五入）。加干扰后，S 的负荷读出为 44.8%，低于无注入连续 EEG 的 51.5%。",
        fig,
        "向同一片段的 EEG 添加峰在 +6 秒、宽度标准差 0.15 秒的高斯脉冲，额部峰值 60 μV、顶部 12 μV，随后重新编码。脉冲传入比是处理前后两次输出之差的 RMS 除以注入 RMS；α 保留比是无注入时顶部 8–13 Hz 功率相对观测的比值。每人先取块中位数，再等权平均四人。波形示例取 S 保留比居中的开发块。注入为工程扰动，参照为处理后实测 EEG，天然噪声未提供干净真值。",
        "依据　fidelity.csv · denoise_gate.csv · task_people.csv　｜　绿色带为保留比 0.8–1.25，虚线为传入比 0.5。",
    )


def stability(doc, m, a, people):
    fam = family_lut()
    dirty = a["dirty_ids"][:, :, 0]
    clean = a["ids"][:, :, 0]
    delta = np.stack([dirty != clean, fam[dirty] != fam[clean]], axis=1)
    tiles = delta.reshape(len(m), -1)
    per_person = pd.DataFrame(tiles).groupby(m.subject).mean()
    maps = per_person.mean(0).to_numpy().reshape(2, 4, 15)
    rows = []
    for sub, values in per_person.iterrows():
        for j, kind in enumerate(["ID", "family"]):
            for site, cells in zip(CFG["sites"]["eeg"], values.to_numpy().reshape(2, 4, 15)[j]):
                rows.extend(
                    dict(subject=sub, kind=kind, site=site, start_s=-5 + 2 * k, change=value)
                    for k, value in enumerate(cells)
                )
    pd.DataFrame(rows).to_csv(FIG / "pulse_state_changes.csv", index=False)
    fig, axs = canvas(1, 3, gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for j, (ax, title) in enumerate(zip(axs[:2], ["EEG-S 硬词元改变", "EEG-S 状态族改变"])):
        im = ax.imshow(
            maps[j] * 100,
            aspect="auto",
            extent=(-5, 25, 3.5, -0.5),
            vmin=0,
            vmax=100,
            cmap="YlGnBu",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_yticks(range(4), CFG["sites"]["eeg"])
        ax.set_xticks([-5, 0, 5, 10, 15, 20, 25])
        ax.set_xlabel("相对任务起点（秒）")
        for v in [5, 7]:
            ax.axvline(v, color="#ae483c", ls="--", lw=0.9)
    fig.colorbar(im, ax=axs[:2].tolist(), label="改变的块（%）", shrink=0.72)
    ba = people.groupby(["method", "input"]).ba.mean().unstack()
    keys = ["X", "HP", "S", "O", "SO"]
    mat = ba.loc[keys, ["clean", "pulse"]] * 100
    heat(axs[2], mat.to_numpy(), ".1f", vmin=30, vmax=65, cmap="BuGn")
    axs[2].set_yticks(range(5), [METHOD[k] for k in keys])
    axs[2].set_xticks([0, 1], ["原片段", "注入脉冲"])
    axs[2].set_title("EEG 读出 · 平衡准确率（%）")
    outside = np.arange(15) != 5
    page(
        doc,
        "07_state_stability",
        "局部扰动也会改变其他时刻的离线词元",
        f"在脉冲中心所在的 5–7 秒格，四位置平均有 {100*maps[0,:,5].mean():.1f}% 的块改变硬词元，{100*maps[1,:,5].mean():.1f}% 改变状态族。在其余 14 格，两项比例仍为 {100*maps[0][:,outside].mean():.1f}% 和 {100*maps[1][:,outside].mean():.1f}%。",
        fig,
        "左中每格表示同一位置、同一任务块注入前后的配对比较。先在每人内部计算改变比例，再等权平均被试；红虚线标出含脉冲中心的时间格。右表为以无注入训练块拟合、冻结后分别评价两种输入的读出。传播范围与整窗上下文有关；本次固定脉冲不能代表天然眼动或重测。当前 S 未通过保真门槛，应保留观测参照。",
        "依据　pulse_state_changes.csv · task_people.csv · subjects/　｜　S 为教师支持分量，O 为余量；两者都可能携带有效活动。",
    )


def decisions(doc):
    pg = doc.new_page(width=1000, height=706)
    html = """<p class="section">认知负荷　/　实验决策　　　2026 年 11 月 27 日</p>
    <h1>接下来优先验证额顶节律与任务的关联</h1>
    <p>在四名开发被试中，额部 θ 和顶部 α 的两项变化量跨块序区间仍保留了负荷读出能力。复核应使用负荷顺序随被试或会话改变的记录，并保留同一时长的连续 EEG 基线。</p>
    <table><tr><th>研究方向</th><th>继续深入的筛选条件</th><th>本次判断</th></tr>
    <tr><td>连续双模态互补</td><td>联合比两种单模态各高至少 2 个百分点，且各有至少 3 人改善。</td><td>作为对照保留。联合比 EEG 低 0.3 个百分点，仅 2 人改善。</td></tr>
    <tr><td>四路硬词元读出</td><td>联合词元超过两种单模态词元与连续联合至少 2 个百分点，且各有至少 3 人改善。</td><td>暂停扩大码本或网络。联合词元比连续联合低 17.5 个百分点，四人方向一致。量化前联合也较弱。</td></tr>
    <tr><td>区域事件顺序</td><td>相对连续与占用基线、随机方向对照，归一化均方误差均降低至少 0.02，且各有至少 3 人改善。</td><td>暂停把这一事件族转移作为血流机制。相比基线，加入顺序的误差增加 0.096。</td></tr>
    <tr><td>仅保留解码 S 去噪</td><td>传入比低于 0.5，顶部 α 功率比在 0.8–1.25，注入后读出比连续 EEG 原片段最多低 2 个百分点。均值及至少 3 人同时合格。</td><td>放弃本设置下直接用 S 替换 EEG。没有被试同时满足三项；对单一模拟慢脉冲，常规高通取得更好的保真。</td></tr></table>
    <p class="note">评价边界　15 人用于训练，4 人用于开发，未读取测试或保护信号。结果为离线筛选，未核验独立行为、天然眼动、重测可靠性或皮层因果连接。块顺序固定限制了对脑功能的解释。未据此对其他编码器、任务或人群作结论。</p>
    <p class="foot">依据　resolved.yaml · summary.json · denoise_gate.csv · order_check_v1/summary.json</p>"""
    css = """*{font-family:sans-serif;color:#193940}p{font-size:14pt;line-height:1.35}h1{font-size:25pt}
    .section,.foot{font-size:11pt;color:#236d70}table{width:100%;border-collapse:collapse}td,th{font-size:12.5pt;
    border-bottom:1px solid #cbdad7;padding:8px;vertical-align:top}th{background:#e6efec;text-align:left}
    .note{font-size:12.5pt}"""
    h, _ = pg.insert_htmlbox(fitz.Rect(32, 20, 968, 683), html, css=css, scale_low=1)
    if h < 0:
        raise ValueError("Decision table overflow")


def finish(doc):
    doc.subset_fonts()
    dest = FIG / "REPORT.pdf"
    doc.save(dest, deflate=True, garbage=4)
    doc.close()
    rows = []
    with fitz.open(dest) as d:
        for i, p in enumerate(d):
            if i == len(d) - 1:
                assert not p.get_image_info() and len(p.get_text()) > 300
                rows.append(dict(page=i + 1, images=0, text_characters=len(p.get_text())))
                continue
            (img,) = p.get_image_info()
            dpi = img["width"] * 72 / fitz.Rect(img["bbox"]).width
            assert 200 <= dpi <= 300 and len(p.get_text()) >= 150
            rows.append(dict(page=i + 1, dpi=dpi, images=1, text_characters=len(p.get_text())))
    write_json(FIG / "verification.json", dict(status="passed", pages=rows))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=FIG, help="Fresh versioned report directory")
    FIG = parser.parse_args().output
    if (FIG / "REPORT.pdf").exists():
        raise FileExistsError("Choose a fresh versioned report directory")
    FIG.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    m, a = material()
    pred = pd.read_csv(OUT / "task_predictions.csv")
    people = pd.read_csv(OUT / "task_people.csv")
    physiology(doc, m, a)
    classification(doc, m, a, pred, people)
    cases(doc, m, a, pred)
    blockcheck(doc, m)
    sequence(doc, m, a)
    noise(doc, m, a, people)
    stability(doc, m, a, people)
    decisions(doc)
    finish(doc)
