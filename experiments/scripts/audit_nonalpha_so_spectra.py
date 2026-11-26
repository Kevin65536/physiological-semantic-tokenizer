"""Search retained validation EEG for non-alpha peaks before opening the paired v3 targets."""

from pathlib import Path
import sys

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts.audit_v3_alpha_peak import spectrum
from experiments.scripts.check_continuous_so_targets import OUT, PRIOR, old
from experiments.scripts.score_soft_power_so import DownstreamInputDataset, eog_basis, parent
from experiments.scripts.render_temporal_projection_so import DS, table

FOLDER = OUT / "nonalpha_peak_audit_v1"
RATE = 200
SECTION = slice(6000, 12000)
# Search centers and distances to the two flanking background intervals (Hz).
BANDS = {
    "delta": (2, 3.75, 0.5, 1),
    "theta": (4, 7.75, 1, 2),
    "beta": (13, 29.75, 2, 4),
    "low_gamma": (30, 40, 2, 4),
}
GREEK = dict(delta="δ", theta="θ", beta="β", low_gamma="低 γ")
COLORS = ["#243d5a", "#b54b24", "#00837b"]
STYLES = ["-", "--", "-."]
NAMES = ["实测 X", "v3 S", "v3 O"]


def contrast(f, p, idx, inner, outer):
    """Peak dB above the line between median log PSDs at median log flank frequencies."""
    hz = f[idx]
    anchors = []
    for low, high in ((hz - outer, hz - inner), (hz + inner, hz + outer)):
        mask = (f >= low) & (f <= high)
        if not mask.any() or f[mask].min() <= 0:
            raise ValueError("Missing flank")
        anchors.append((np.median(np.log(f[mask])), np.median(10 * np.log10(p[mask]))))
    return float(10 * np.log10(p[idx]) - np.interp(np.log(hz), *zip(*anchors)))


def local_peaks(f, p, band):
    lo, hi, inner, outer = band
    loc = signal.find_peaks(p)[0]
    return [(j, contrast(f, p, j, inner, outer)) for j in loc if lo <= f[j] <= hi]


def subharmonics(f, p, hz):
    notes = []
    if hz >= 13:
        for harmonic in (2, 3):
            center = hz / harmonic
            if 4 <= center <= 14:
                cand = [
                    (d, f[k])
                    for k, d in local_peaks(
                        f, p, (max(3.25, center - 0.75 / harmonic), center + 0.75 / harmonic, 1, 2)
                    )
                    if d >= 3
                ]
                if cand:
                    notes.append(f"{harmonic}×{max(cand)[1]:g}")
    return ", ".join(notes)


def scalp_reference(x, case):
    sites = [
        col
        for col, site in enumerate(case["valid"])
        if case["channels"][site].upper() not in {"M1", "M2", "CB1", "CB2"}
    ]
    return x[:, sites].mean(axis=1)


def screen(cases):
    records = []
    for c in cases:
        i = c["i"]
        assert c["source"]["split"] == "validation" and c["source"]["rate"] == RATE
        with np.load(old.PARENT / "targets" / str(i) / "baseline.npz") as z:
            np.testing.assert_array_equal(z["original_channel_index"], c["valid"])
            x = z["x"][SECTION]
        assert len(x) == 6000 and np.isfinite(x).all()
        f, p = spectrum(x, seconds=4)
        _, car = spectrum(x - scalp_reference(x, c)[:, None], seconds=4)
        short = [spectrum(y, seconds=4)[1] for y in np.split(x, 3)]
        for col, site in enumerate(c["valid"]):
            for name, band in BANDS.items():
                peaks = local_peaks(f, p[:, col], band)
                trials = [local_peaks(f, q[:, col], band) for q in short]
                ranked = []
                for j, db in peaks:
                    repeated = sum(
                        any(abs(f[j] - f[k]) <= 0.75 and v >= 3 for k, v in parts) for parts in trials
                    )
                    ranked.append((db >= 3 and repeated >= 2, db, repeated, f[j]))
                ok, db, rep, hz = max(ranked, default=(False, np.nan, 0, np.nan))
                car_peaks = local_peaks(f, car[:, col], band)
                car_ok = any(abs(f[k] - hz) <= 0.75 and d >= 3 for k, d in car_peaks)
                car_db = (
                    contrast(f, car[:, col], int(round(hz * 4)), *band[2:])
                    if np.isfinite(hz)
                    else np.nan
                )
                harmonic = subharmonics(f, p[:, col], hz)
                clock = {
                    "simultaneous_eeg_nirs": 10.4,
                    "eeg_fnirs_single_trial": 12.5,
                    "visual_cognitive_motivation": 10.0,
                    "refed": 47.62,
                }[c["dataset"]]
                clock_match = any(abs(hz - k * clock) <= 0.5 for k in (1, 2, 3, 4))
                records.append(
                    dict(
                        case=i,
                        screen_car_peak=car_ok,
                        screen_car_db=car_db,
                        possible_harmonic=harmonic,
                        optical_clock_match=clock_match,
                        record=c["key"],
                        subject=c["subject"],
                        dataset=c["dataset"],
                        col=col,
                        site=c["channels"][site],
                        band=name,
                        hz=hz,
                        excess_db=db,
                        repeated=rep,
                        eligible=ok,
                    )
                )
    df = pd.DataFrame(records)
    assert len(df) == 305 * 4 and df["case"].nunique() == 8
    return df


def choose(df):
    # One peak per band, plus frontal theta and central beta views when eligible.
    specs = [(k, k, df.site.notna()) for k in BANDS]
    specs.extend(
        [
            (
                "frontal_theta",
                "theta",
                df.site.str.upper().isin(
                    ["FZ", "AFZ", "FCZ", "F1", "F2", "FC1", "FC2", "AFF1H", "AFF2H"]
                ),
            ),
            ("central_beta", "beta", df.site.str.upper().str.match(r"^(C|CP|FC|CCP|FCC)[Z0-9]")),
        ]
    )
    central = df.site.str.upper().str.match(r"^(C|CP|FC|CCP|FCC)[Z0-9]")
    robust = (
        df.eligible
        & df.screen_car_peak
        & (df.screen_car_db >= 3)
        & ~df.optical_clock_match
        & df.possible_harmonic.eq("")
    )
    for band, zone in [
        ("delta", df.site.notna()),
        ("theta", df.site.notna()),
        ("beta", central),
        ("low_gamma", df.site.notna()),
    ]:
        subset = robust & zone & (df.band == band)
        if subset.any():
            specs.append(("supported_" + band, band, subset))
    chosen = []
    for label, b, zone in specs:
        ranked = df[(df.band == b) & zone].sort_values(
            ["eligible", "excess_db", "repeated", "case", "col"],
            ascending=[False, False, False, True, True],
        )
        if ranked.empty or not ranked.eligible.any():
            continue
        row = ranked.iloc[0]
        if not np.isfinite(row.hz):
            continue
        if any(t["band"] == b and t["case"] == row["case"] and t["col"] == row.col for t in chosen):
            continue
        chosen.append(dict(row, selection=label))
    return chosen


def example(row, c, df, folder, q):
    i = str(c["i"])
    si = c["valid"][row["col"]]
    col = row["col"]
    with np.load(old.PARENT / "targets" / i / "baseline.npz") as z:
        x, s, o = (z[k][:17980] for k in ("x", "v3", "v3_o"))
    np.testing.assert_array_equal(x - s, o)
    ident = old.read(old.SSM / "inputs" / i / "identity.json")["group"]
    assert ident["key"] == c["key"] and ident["EEG_channels"] == c["channels"]
    native = np.load(old.SSM / "inputs" / i / "native.npy", mmap_mode="r")
    rate = int(ident["rate"])
    band = BANDS[row["band"]]
    lo, hi, inner, outer = band
    waves = np.stack([a[:, col] for a in (x, s, o)], axis=1)
    seg = waves[SECTION]
    f, p = spectrum(seg, seconds=4)
    hz = row["hz"]
    j = np.flatnonzero(f == hz).item()
    _, cross = signal.csd(seg[:, 1], seg[:, 2], fs=RATE, nperseg=800, noverlap=400, window="hann")
    np.testing.assert_allclose(
        p[:, 0], p[:, 1] + p[:, 2] + 2 * cross.real, rtol=1e-9, atol=p[:, 0].max() * 1e-12
    )
    nf, pn = spectrum(native[30 * rate : 60 * rate, si], rate=rate, seconds=4)
    np.testing.assert_array_equal(nf[: len(f)], f)
    _, pc = spectrum(seg[:, 0] - scalp_reference(x[SECTION], c), seconds=4)
    sub = df[(df["case"] == row["case"]) & (df.band == row["band"])]
    nearby = sub[sub.eligible & ((sub.hz - hz).abs() <= 0.75)]

    def close_peak(freq, density):
        return any(abs(freq[k] - hz) <= 0.75 and db >= 3 for k, db in local_peaks(freq, density, band))

    # Energy of the EEG's zero-lag projection on the simultaneous two-channel EOG.
    filt = signal.butter(4, [hz - 0.75, hz + 0.75], fs=RATE, btype="bandpass", output="sos")
    filtered = signal.sosfiltfilt(filt, waves, axis=0)
    eog_share = np.nan
    if q is not None:
        eye = signal.sosfiltfilt(filt, q, axis=0)[SECTION]
        eye -= eye.mean(axis=0)
        v = filtered[SECTION, 0] - filtered[SECTION, 0].mean()
        fit = eye @ np.linalg.lstsq(eye, v, rcond=None)[0]
        eog_share = float(np.sum(fit**2) / np.sum(v**2))
    stats = dict(
        **row,
        unit=c["unit"],
        start_200=c["source"]["start_sample"] + 6000,
        stop_200=c["source"]["start_sample"] + 12000,
        native_rate=rate,
        native_same_hz_db=contrast(nf, pn, j, inner, outer),
        native_peak=close_peak(nf, pn),
        car_same_hz_db=contrast(f, pc, j, inner, outer),
        car_peak=close_peak(f, pc),
        nearby=len(nearby),
        sites=len(sub),
        nearby_sites=" ".join(nearby.site),
        eog_share=eog_share,
        twice_cross_fraction=2 * cross[j].real / p[j, 0],
    )
    for k, key in enumerate(("X", "S", "O")):
        stats[key + "_psd"] = p[j, k]
        stats[key + "_excess_db"] = contrast(f, p[:, k], j, inner, outer)
        stats[key + "_rms"] = np.sqrt(np.mean(seg[:, k] ** 2))
        stats[key + "_narrow_rms"] = np.sqrt(np.mean(filtered[SECTION, k] ** 2))
        stats[key + "_narrow_corr_x"] = np.corrcoef(filtered[SECTION, k], filtered[SECTION, 0])[0, 1]
    # REFED source units have not been verified. Display a common scale for all three components.
    scale = 1e6 if c["dataset"] == "refed" else 1.0
    unit = r"$10^{-6}$ 发布单位" if c["dataset"] == "refed" else "µV"
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.5), layout="constrained")
    middle = slice(44 * 200, 46 * 200)
    t = np.arange(44 * 200, 46 * 200) / 200
    for k in range(3):
        style = dict(color=COLORS[k], ls=STYLES[k], lw=1.3, label=NAMES[k])
        axes[0, 0].plot(t, waves[middle, k] * scale, **style)
        axes[1, 0].plot(t, filtered[middle, k] * scale, **style)
        for ax, mask in [
            (axes[0, 1], (f >= 1) & (f <= 45)),
            (axes[1, 1], (f >= max(1, hz - outer - 1)) & (f <= min(45, hz + outer + 1))),
        ]:
            ax.semilogy(f[mask], p[mask, k] * scale**2, **style)
    for ax in axes[0, 0], axes[1, 0]:
        ax.set(xlabel="相对父窗起点 (s)", ylabel=f"电压 ({unit})")
        ax.axhline(0, color="grey", lw=0.5)
    axes[0, 0].set_title("同一 2 秒的宽带波形")
    axes[1, 0].set_title(f"同一时段 · {hz-.75:g}–{hz+.75:g} Hz 带通（仅供观察）")
    axes[0, 1].axvspan(lo, hi, color="gold", alpha=0.15)
    for ax in axes[:, 1]:
        ax.axvline(hz, color="#777", ls=":", lw=1)
        ax.set(xlabel="频率 (Hz)", ylabel=f"功率谱密度 [({unit})²/Hz]")
    axes[0, 1].set_title("30–60 s 全谱 · 阴影为搜索范围")
    axes[1, 1].set_title(f"峰附近实测与目标谱 · 虚线标出 {hz:g} Hz")
    axes[0, 1].legend(ncol=3, fontsize=9)
    fig.suptitle(
        f"{DS[c['dataset']]} · {c['subject']} · {row['site']}   |   {GREEK[row['band']]} 频带   {hz:g} Hz"
    )
    fig.savefig(folder / (row["selection"] + ".png"), dpi=210)
    plt.close(fig)
    return stats


def atlas(folder, rows, screen):
    explanations = {
        (
            "31",
            "C5",
            17.5,
        ): "中央及中央顶多枚电极出现同频峰。两分量在窄带中均跟随 X，振幅接近。可作为感觉运动 β 候选，本段为观看视频，没有动作起止对照。",
        (
            "8",
            "FPZ",
            7.25,
        ): "同频电极主要在额极和额前。S 保留的波动大于 O。这里缺少独立眼电，7.25 Hz 又靠近 α 下缘，无法把额部 θ 直接归于认知操作。",
        (
            "8",
            "T7",
            3.0,
        ): "三个 10 秒段均能看到慢峰，但同频电极仅 FT7、T7、M2。颞侧和乳突的分布限制了皮层来源解释，缺少独立眼电。",
        (
            "VP002",
            "P7",
            2.0,
        ): "只有 P7 的入选主峰落在这一频率附近，平均重参考后峰相对背景减弱。可作为慢波动和眼动混合的对照。",
        (
            "S06",
            "F7",
            6.25,
        ): "同频电极分布较广。O 保留多于 S，F7 的峰对参考方式敏感。这一记录没有独立眼电，难以确定节律来源。",
        (
            "S14",
            "Fz",
            5.25,
        ): "额中线位置与 θ 候选相符，但重参考后同频点突出度很低，也缺少其他电极的同频入选主峰。未按任务事件分组。",
        (
            "VP002",
            "CP5",
            20.75,
        ): "同电极有约 10.5 Hz 的强局部峰，目标频率也接近光学采样频率 10.4 Hz 的两倍。O 复现了这个尖峰，仍须考虑共享设备信号和非正弦波谐波。",
        (
            "VP002",
            "T8",
            31.25,
        ): "本峰同时靠近 10.5 Hz 局部峰的三倍和光学采样频率的三倍。颞侧高频还可能含肌电。仅凭窄峰和稳定波形无法指认皮层 γ。",
        (
            "VP002",
            "FC6",
            39.5,
        ): "简单的倍频筛查未标记该峰。窄带集中和重参考后保留，均不足以区分神经节律、设备窄带信号和肌电。没有同步肌电对照。",
    }
    doc = fitz.open()

    def box(page, rect, body):
        spare, _ = page.insert_htmlbox(
            fitz.Rect(rect),
            '<div lang="zh-Hans">' + body + "</div>",
            css="body{font-family:sans-serif;font-size:12pt;color:#22394a;line-height:1.35} h1{font-size:22pt;color:#244c65} p{margin:4pt 0 6pt} table{border-collapse:collapse;width:100%} td,th{padding:3pt;border-bottom:0.5pt solid #afc3cc;text-align:left}",
            scale_low=1,
        )
        if spare < 0:
            raise ValueError("Text does not fit")

    pg = doc.new_page(width=900, height=650)
    counts = []
    for name, (lo, hi, inner, outer) in BANDS.items():
        passed = screen[(screen.band == name) & screen.eligible]
        counts.append(
            [
                GREEK[name],
                f"{lo:g}–{hi:g}",
                f"±{inner:g} 至 ±{outer:g}",
                len(passed),
                passed["case"].nunique(),
            ]
        )
    title = "<h1>α 以外的实测 EEG 频谱与 v3 S/O 目标</h1>"
    box(
        pg,
        (32, 18, 868, 635),
        title
        + "<p>公开验证面板含四个数据集、8 名清醒被试，每人一个既有父窗，共 305 个有效电极记录。所有搜索固定在父窗的 30–60 秒，按频带共作 1,220 次描述性筛查。先按实测 X 定位窄峰，再读取 v3 监督电压 S 与残差 O=X−S。</p>"
        + table(
            ["频带", "搜索峰中心 (Hz)", "两侧背景距峰 (Hz)", "合格电极记录 / 305", "涉及被试 / 8"],
            counts,
        )
        + "<p>200 Hz 共同分析坐标，4 秒 Hann 窗、50% 重叠、逐窗去均值。30 秒谱有 14 个窗，频点间距 0.25 Hz。每侧取区间内功率的中位 dB，在对数频率轴上连接两侧背景；峰的突出度为实测 dB 减去该线。保留突出度 ≥3 dB 且在三个不重叠的 10 秒段中至少两次于 ±0.75 Hz 内再次满足此条件的局部峰。一个电极在一个频带取合格峰中突出度最高的一项。计数受频带宽度、频点数量和背景估计影响。</p>"
        + "<p>图例选取。先取每带突出度最高的合格样例，再补充额中线及邻近 θ、中央区 β，以及各频带经平均重参考后原频点仍 ≥3 dB、附近仍见局部峰的例子（β 限中央区）。后一组排除了近光学采样频率整数倍和与 4–14 Hz 明显局部峰构成二、三倍频的例子。重复样例只绘一次。补充筛查针对首轮观察到的混淆，结论为探索性。重参考使用有效 EEG，排除 M1、M2、CB1、CB2。</p>"
        + "<p>时域固定展示 44–46 秒。窄带图在有效父段上对三路电压施加相同的四阶双向 Butterworth 滤波（峰中心 ±0.75 Hz）。频域展示同一 30 秒的全谱和局部放大。每个坐标内三路共用刻度，跨面板可独立缩放。REFED 使用其发布单位。峰功率比按同一频点分别除以 X，另计两倍交叉谱实部；三者相加为 100%。窄带 RMS 比为各自 30 秒均方根电压除以 X 的窄带均方根。</p>"
        + "<p>后页同时检查对齐的原生采样率电压。独立眼电可用时，给出同窗窄带 EEG 在同步双通道眼电空间上的零时滞线性投影能量占比；它只度量共享波动。本面板没有同步肌电或任务事件锁定的条件对照。</p>",
    )
    for r in rows:
        pg = doc.new_page(width=900, height=650)
        pg.insert_image(
            fitz.Rect(18, 8, 882, 8 + 864 * 6.5 / 11.6), filename=str(folder / (r["selection"] + ".png"))
        )
        key = (str(r["subject"]), r["site"], r["hz"])
        note = explanations[key]
        eye = (
            f'眼电共享窄带能量 {r["eog_share"]:.0%}。'
            if np.isfinite(r["eog_share"])
            else "本记录缺少独立眼电对照。"
        )
        box(
            pg,
            (32, 500, 868, 642),
            f'<b>{note}</b><p>同频入选主峰 {r["nearby"]}/{r["sites"]} 个电极，子段复现 {r["repeated"]}/3。X 同频点高于局部背景 {r["excess_db"]:.2f} dB，原生电压为 {r["native_same_hz_db"]:.2f} dB，平均重参考后 {r["car_same_hz_db"]:.2f} dB。{eye}</p>'
            f'<p>峰频点功率相对 X，S 为 {r["S_psd"]/r["X_psd"]:.1%}，O 为 {r["O_psd"]/r["X_psd"]:.1%}，交叉项为 {r["twice_cross_fraction"]:.1%}。窄带 RMS 相对 X，S 为 {r["S_narrow_rms"]/r["X_narrow_rms"]:.1%}，O 为 {r["O_narrow_rms"]/r["X_narrow_rms"]:.1%}。</p>'
            f'<p>{r["record"]}　·　200 Hz 样点 [{r["start_200"]}, {r["stop_200"]})。</p>',
        )
    doc.subset_fonts()
    pdf = folder / "nonalpha_XSO.pdf"
    doc.save(pdf, garbage=4, deflate=True)
    doc.close()
    with fitz.open(pdf) as check:
        assert len(check) == len(rows) + 1
        assert all(
            page.get_text().strip() and len(page.get_images()) == int(page.number > 0) for page in check
        )


def main():
    # Dry-run the peak and background calculation before opening the EEG panel.
    f = np.arange(0, 100.25, 0.25)
    base = 1 / (np.maximum(f, 0.25) ** 1.5)
    for band in BANDS.values():
        center = np.round(np.mean(band[:2]) * 4) / 4
        p = base * (1 + 6 * np.exp(-(((f - center) / 0.25) ** 2)))
        peak = max(local_peaks(f, p, band), key=lambda v: v[1])
        assert f[peak[0]] == center and peak[1] > 7 and not local_peaks(f, base, band)
    cases = old.read(PRIOR / "plan.json")["cases"]
    df = screen(cases)
    chosen = choose(df)
    FOLDER.mkdir(parents=True, exist_ok=True)
    df.to_csv(FOLDER / "screen.csv", index=False)
    view = DownstreamInputDataset(ROOT / parent.CFG["downstream"], split="validation")
    eyes = {i: eog_basis(view, cases[i]) for i in sorted({r["case"] for r in chosen})}
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False})
    rows = [example(r, cases[r["case"]], df, FOLDER, eyes[r["case"]]) for r in chosen]
    pd.DataFrame(rows).to_csv(FOLDER / "examples.csv", index=False)
    atlas(FOLDER, rows, df)


if __name__ == "__main__":
    main()
