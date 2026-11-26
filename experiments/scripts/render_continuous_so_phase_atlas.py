"""Paired time, spectrum, cross-phase and alpha-phase views of retained EEG targets."""

import argparse
from pathlib import Path
import sys

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts.check_continuous_so_targets import OUT, PRIOR, old, projection
from experiments.scripts.render_temporal_projection_so import DS, table

RATE = 200
INTERVAL = slice(30 * RATE, 60 * RATE)
TIME = np.arange(17980) / RATE
FREQ = np.fft.rfftfreq(400, 1 / RATE)
BAND = (FREQ >= 1) & (FREQ <= 45)
ALPHA = (FREQ >= 8) & (FREQ < 13)
BLUE, ORANGE, GREEN = "#224d70", "#c2571e", "#00776f"
FILTER = signal.butter(4, [8, 13], fs=RATE, btype="bandpass", output="sos")


def load_case(case):
    if case["source"]["split"] != "validation" or case["source"]["rate"] != RATE:
        raise ValueError("Expected registered development EEG at 200 Hz")
    with np.load(old.PARENT / "targets" / str(case["i"]) / "baseline.npz") as f:
        np.testing.assert_array_equal(f["original_channel_index"], case["valid"])
        x, s, o = [f[k][:-20] for k in ("x", "v3", "v3_o")]
    assert x.shape == s.shape == o.shape == (len(TIME), len(case["valid"]))
    np.testing.assert_array_equal(o, x - s)
    new_s = projection(s)
    new_o = x - new_s
    scale = np.max(abs(x))
    np.testing.assert_allclose(s + o, x, rtol=0, atol=scale * 1e-12)
    np.testing.assert_allclose(new_s + new_o, x, rtol=0, atol=scale * 1e-12)
    return x, s, o, new_s, new_o


def select(cases):
    candidates = []
    for c in cases:
        _, s, _, new, _ = load_case(c)
        change = np.linalg.norm((new - s)[INTERVAL], axis=0) / np.linalg.norm(s[INTERVAL], axis=0)
        assert np.isfinite(change).all()
        for col, idx in enumerate(c["valid"]):
            candidates.append(
                dict(
                    dataset=c["dataset"],
                    subject=c["subject"],
                    case=c["i"],
                    record=c["key"],
                    channel=c["channels"][idx],
                    col=col,
                    channel_index=idx,
                    start_sample_200=c["source"]["start_sample"] + INTERVAL.start,
                    stop_sample_200=c["source"]["start_sample"] + INTERVAL.stop,
                    relative_change=change[col],
                )
            )
    frame = pd.DataFrame(candidates)
    frame["selected"] = False
    for _, g in frame.groupby("dataset"):
        # With even N both central observations are equally close to the median.
        idx = g.sort_values(["relative_change", "case", "col"]).index[(len(g) - 1) // 2]
        frame.loc[idx, "selected"] = True
    assert (
        len(frame) == 305 and frame.groupby(["dataset", "subject"]).ngroups == 8 and frame.selected.sum() == 4
    )
    return frame


def spectral_pair(a, b):
    """Time-aligned Welch auto- and cross-spectra with the positive phase convention B minus A."""

    def frames(x):
        z = np.lib.stride_tricks.sliding_window_view(x, 400)[::200]
        return np.fft.rfft((z - z.mean(1, keepdims=True)) * signal.windows.hann(400, sym=False), axis=1)

    left, right = frames(a), frames(b)
    norm = np.full(len(FREQ), 2 / (RATE * np.sum(signal.windows.hann(400, sym=False) ** 2)))
    norm[[0, -1]] /= 2
    pa = (abs(left) ** 2).mean(0) * norm
    pb = (abs(right) ** 2).mean(0) * norm
    cross = (left.conj() * right).mean(0) * norm
    coh = np.clip(np.divide(abs(cross) ** 2, pa * pb, out=np.zeros_like(pa), where=pa * pb > 0), 0, 1)
    threshold = 0.001 * max(pa[BAND].max(), pb[BAND].max())
    good = BAND & (pa >= threshold) & (pb >= threshold) & (coh >= 0.5) & (pa * pb > 0)
    return pa, pb, np.angle(cross, deg=True), coh, good


def alpha_pair(a, b):
    za, zb = (signal.hilbert(signal.sosfiltfilt(FILTER, w)) for w in (a, b))
    threshold = 0.1 * max(np.sqrt(np.mean(abs(z[INTERVAL]) ** 2)) for z in (za, zb))
    good = (abs(za) > threshold) & (abs(zb) > threshold)
    angles = [np.where(good, np.angle(z, deg=True), np.nan) for z in (za, zb)]
    diff = np.where(good, np.angle(zb * za.conj(), deg=True), np.nan)
    return angles, diff, good


def wrapped_line(ax, t, values, **style):
    # Split at the ±180° display wrap; keep the first sample after each break.
    jumps = np.where(abs(np.diff(values)) > 180)[0] + 1
    ax.plot(np.insert(t, jumps, np.nan), np.insert(values, jumps, np.nan), **style)


def label(case):
    return f"{DS[case['dataset']]} · {case['subject']} · {case['channel']}"


def figure(case, signals, path):
    relative = case["dataset"] == "refed"
    factor = 1e6 if relative else 1.0
    unit = r"$10^{-6}$ 发布单位" if relative else "µV"
    wave_t = (TIME >= 44) & (TIME < 46)
    phase_t = (TIME >= 44.5) & (TIME < 45)
    fig, axes = plt.subplots(3, 4, figsize=(15, 8.2))
    fig.subplots_adjust(left=0.065, right=0.984, top=0.836, bottom=0.13, wspace=0.31, hspace=0.37)
    fig.suptitle(label(case), x=0.065, ha="left", y=0.984, fontsize=20, weight="bold")
    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, lw=2, label="原 v3"),
            Line2D([], [], color=ORANGE, lw=1.7, ls="--", label="连续投影版"),
            Line2D([], [], color=GREEN, ls="", marker="o", ms=4, label="两版交叉谱相位差"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.984, 0.984),
        ncol=3,
        frameon=False,
    )
    titles = [
        "电压波形  ·  44–46 s",
        "功率谱密度  ·  30–60 s",
        "频率相位差  ·  新减旧",
        "α 瞬时相位  ·  44.5–45 s",
    ]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, pad=11, fontsize=12, weight="bold")
    x, s, o, new_s, new_o = signals
    rows, spectra = [], []
    for k, (a, b), axs in zip("XSO", [(x, x), (s, new_s), (o, new_o)], axes):
        a, b = a * factor, b * factor
        p0, p1, angle, coh, good = spectral_pair(a[INTERVAL], b[INTERVAL])
        alphas, delta, alpha_good = alpha_pair(a, b)
        for wave, p, phi, co, ls in zip((a, b), (p0, p1), alphas, (BLUE, ORANGE), ("-", "--")):
            axs[0].plot(TIME[wave_t], wave[wave_t], lw=1.2, color=co, ls=ls)
            axs[1].semilogy(FREQ[BAND], p[BAND], lw=1.5, color=co, ls=ls)
            wrapped_line(axs[3], TIME[phase_t], phi[phase_t], color=co, ls=ls, lw=1.5)
        for ax in axs[1:3]:
            ax.axvspan(8, 13, color="#f1d790", alpha=0.31, zorder=0)
            ax.set(xlim=(1, 45), xticks=[1, 8, 13, 25, 35, 45])
        axs[2].axhline(0, color="#9da9b0", lw=0.7, zorder=0)
        axs[2].scatter(FREQ[good], angle[good], s=12, color=GREEN)
        axs[2].text(
            0.98,
            0.96,
            f"{good.sum()}/{BAND.sum()} 频点",
            ha="right",
            va="top",
            transform=axs[2].transAxes,
            fontsize=9,
            color="#34545e",
        )
        for ax in axs[2:]:
            ax.set(ylim=(-190, 190), yticks=[-180, -90, 0, 90, 180])
        axs[0].set(xlim=(44, 46), xticks=[44, 44.5, 45, 45.5, 46], ylabel=f"{k}  ({unit})")
        axs[3].set(xlim=(44.5, 45), xticks=[44.5, 44.75, 45])
        for ax in axs:
            ax.grid(axis="y", alpha=0.17)
            ax.spines[["top", "right"]].set_visible(False)
        z = alpha_good[INTERVAL]
        beta = (FREQ >= 13) & (FREQ < 30)
        rows.append(
            dict(
                component=k,
                alpha_power_ratio=p1[ALPHA].sum() / p0[ALPHA].sum(),
                beta_power_ratio=p1[beta].sum() / p0[beta].sum(),
                reliable_freq_bins=int(good.sum()),
                median_abs_alpha_phase_deg=np.median(abs(delta[INTERVAL][z])),
                alpha_support=z.mean(),
                wave_rms_change_over_original=np.linalg.norm((a - b)[INTERVAL]) / np.linalg.norm(a[INTERVAL]),
            )
        )
        for j in np.flatnonzero(BAND):
            spectra.append(
                dict(
                    component=k,
                    freq_hz=FREQ[j],
                    v3_psd=p0[j] / factor**2,
                    continuous_psd=p1[j] / factor**2,
                    phase_new_minus_old_deg=angle[j] if good[j] else np.nan,
                    coherence=coh[j],
                    phase_shown=good[j],
                )
            )
    for ax, xlabel in zip(
        axes[-1],
        [
            "相对父窗起点 (s)",
            f"Hz  ·  PSD 单位为 ({unit})²/Hz",
            "Hz  ·  相位差 (°)",
            "相对父窗起点 (s)  ·  相位 (°)",
        ],
    ):
        ax.set_xlabel(xlabel)
    fig.text(
        0.065,
        0.047,
        "X 为同一实测电压，两版重合。S、O 为监督目标。频谱阴影为 α 带。相位差仅画双侧功率和相干度均达阈值的频点。",
        fontsize=11,
        color="#2c4452",
    )
    fig.text(
        0.065,
        0.017,
        "电压与功率谱纵轴按成分和样例独立缩放，同一面板两版共用刻度。α 瞬时相位在低包络处留空，在 ±180° 处断线。",
        fontsize=10,
        color="#455e65",
    )
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)
    return rows, spectra


def atlas(folder, panels, result, selection):
    doc = fitz.open()
    css = """body{font-family:sans-serif;font-size:12pt;color:#233d4b;line-height:1.4}
    h1{font-size:22pt;color:#134c63}p{margin:4pt 0 8pt}th,td{padding:4pt;border-bottom:0.5pt solid #bfd0d4}
    th{background:#e6eff2;text-align:left}table{width:100%;border-collapse:collapse;font-size:11pt}"""

    def box(pg, rect, body):
        result = pg.insert_htmlbox(
            fitz.Rect(rect), '<div lang="zh-Hans">' + body + '</div>', css=css, scale_low=1
        )
        if result[0] < 0:
            raise ValueError("Report text overflow")

    page = doc.new_page(width=900, height=650)
    display = []
    for _, c in selection[selection.selected].iterrows():
        r = result[(result["case"] == c["case"])].set_index("component")
        display.append(
            [
                label(c),
                str(len(selection[selection.dataset == c.dataset])),
                f'{c.relative_change:.1%}',
                f'{r.loc["S","alpha_power_ratio"]:.3f}',
                f'{r.loc["S","beta_power_ratio"]:.3f}',
                f'{r.loc["O","beta_power_ratio"]:.3f}',
                f'{r.loc["O","median_abs_alpha_phase_deg"]:.3f}°',
            ]
        )
    box(
        page,
        (34, 20, 866, 628),
        "<h1>原 v3 与连续投影版的 X、S、O</h1>"
        "<p>四个数据集各展示一个来自公开开发被试的真实例子。X 是同一实测 EEG；旧版为 S₀ 与 X−S₀，新版为连续投影后的 S 与 X−S。用全长 17,980 个有效采样点生成候选，再在父窗的 30–60 秒评分。两版均验证 X=S+O。生产基线仍为 v3。</p>"
        + table(
            [
                "数据集 · 被试 · 电极",
                "库内候选数",
                "S 相对改变量",
                "S α 功率比",
                "S β 功率比",
                "O β 功率比",
                "O α 相位差中位绝对值",
            ],
            display,
        )
        + "<p>原面板为 8 人、305 个电极记录。固定时间窗，在各库内按 RMS(新 S−旧 S)/RMS(旧 S) 排序，取居中电极；偶数取靠下的一项。功率比为新版除以 v3，α 为 8–13 Hz，β 为 13–30 Hz，频带上限均不计入。样例用于解释，各库一例无法估计总体效应。</p>"
        + "<p>频域与相位。200 Hz，30 秒内分成 29 个重叠一半的 2 秒 Hann 窗，各窗先减均值。功率谱为单边 Welch 密度，频点间隔 0.5 Hz。相位差取平均交叉谱的角度（新版乘旧版的复共轭）。正值表示按这一约定新版相位超前，角度环绕至 ±180°。每个成分只显示两边功率均达到两版 1–45 Hz 共同峰值的 0.1%、且幅值平方相干度 ≥0.5 的频点；分母为 89 个频点。</p>"
        + "<p>时域 α 相位。对各自完整有效父段做四阶 Butterworth 8–13 Hz 双向滤波和 Hilbert 变换，取解析信号的角度。两版包络均超过该成分两版 30 秒包络 RMS 较大值的 10% 时计分。汇总取 30 秒有效点上环绕相位差的绝对值中位数。图中半秒用于读出周期与相位重合程度。</p>"
        + "<p>电压使用各库原单位，REFED 未核实为 µV，图中将其发布值放大 10⁶ 倍。各组条件为同一基线记录的两种分解，无注入或独立事件标签。这些图显示目标构造保留或重新分配的成分。</p>",
    )
    for c, path in panels:
        page = doc.new_page(width=900, height=650)
        page.insert_image(fitz.Rect(15, 12, 885, 12 + 870 * 8.2 / 15), filename=str(path))
        metrics = result[result["case"] == c["case"]].set_index("component")
        s, o = metrics.loc["S"], metrics.loc["O"]
        box(
            page,
            (34, 502, 866, 638),
            f'<b>{label(c)}</b>　{c["record"]}。同一记录 200 Hz 分析坐标 [{c["start_sample_200"]}, {c["stop_sample_200"]})。'
            f'<p>S 的相对 RMS 改变量为 {c["relative_change"]:.1%}，β 功率变为旧版的 {s.beta_power_ratio:.2f} 倍，O 的 β 功率变为 {o.beta_power_ratio:.2f} 倍。'
            f'S、O 在 30 秒内可判读的 α 相位点分别占 {s.alpha_support:.1%}、{o.alpha_support:.1%}，相位差绝对值中位数分别为 {s.median_abs_alpha_phase_deg:.3f}°、{o.median_abs_alpha_phase_deg:.3f}°。'
            f'在 1–45 Hz 内，S 有 {int(s.reliable_freq_bins)}/89 个频点可比较相位，O 有 {int(o.reliable_freq_bins)}/89 个。</p>',
        )
    doc.subset_fonts()
    doc.save(folder / "XSO_时域频域与相位对比.pdf", garbage=4, deflate=True)
    doc.close()


def render(folder):
    if not old.read(OUT / "summary.json")["status"] == "completed":
        raise ValueError("Incomplete parent experiment")
    cases = old.read(PRIOR / "plan.json")["cases"]
    selected = select(cases)
    folder.mkdir(parents=True, exist_ok=False)
    selected.to_csv(folder / "selection.csv", index=False)
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 10, "axes.unicode_minus": False})
    panels, metrics, spectra = [], [], []
    for c in selected[selected.selected].to_dict("records"):
        case = next(case for case in cases if case["i"] == c["case"])
        data = [arr[:, c["col"]] for arr in load_case(case)]
        path = folder / f'{DS[c["dataset"]]}_{c["subject"]}_{c["channel"]}.png'
        m, f = figure(c, data, path)
        identity = {k: c[k] for k in ("dataset", "subject", "case", "channel")}
        metrics.extend(dict(**identity, **r) for r in m)
        spectra.extend(dict(**identity, **r) for r in f)
        panels.append((c, path))
    result = pd.DataFrame(metrics)
    result.to_csv(folder / "metrics.csv", index=False)
    pd.DataFrame(spectra).to_csv(folder / "spectra.csv", index=False)
    atlas(folder, panels, result, selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    render(parser.parse_args().output)
