"""Replay one declared Visual example and inspect frozen v3 token-member spectra."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts.check_continuous_so_targets import ORDER, OUT, PRIOR, old
from src.data.dual_component_tokenizer import BANDS, voltage_power_cap_target
from src.data.so_supervision_cache import bind_reference, write_json
from src.data.temporal_projection_so import continuous_indices

FOLDER = OUT / "alpha_peak_audit_v1"
RECORD = "visual_cognitive_motivation/S14/S14_Part1/e0000390003_n0045000"


def spectrum(x, rate=200, seconds=2):
    """Time is the leading dimension, trailing dimensions index observations."""
    n = rate * seconds
    return signal.welch(
        x,
        fs=rate,
        nperseg=n,
        noverlap=n // 2,
        axis=0,
        detrend="constant",
        window="hann",
        scaling="density",
    )


def excess(f, p):
    """Residual dB from a local log-log line on flanking bins; no fit at DC."""
    use = ((f >= 5) & (f <= 7)) | ((f >= 14) & (f <= 18))
    coef = np.polyfit(np.log10(f[use]), 10 * np.log10(p[use]), 1)
    hz = np.where(f > 0, f, np.nan)
    return 10 * np.log10(p) - np.multiply.outer(np.log10(hz), coef[0]) - coef[1]


def alpha_peak(f, p):
    residual = excess(f, p)
    peaks = signal.find_peaks(p)[0]
    peaks = peaks[(f[peaks] >= 8) & (f[peaks] <= 13)]
    if len(peaks) == 0:
        return np.nan, np.nan
    j = peaks[np.argmax(residual[peaks])]
    return f[j], residual[j]


def sanity():
    t = np.arange(6000) / 200
    x = np.random.default_rng(104).normal(size=len(t)) + 2 * np.cos(2 * np.pi * 10 * t)
    f, p = spectrum(x)
    assert alpha_peak(f, p)[0] == 10
    np.testing.assert_allclose(excess(f, p), excess(f, 3 * p), atol=1e-10)
    positive = f > 0
    np.testing.assert_allclose(excess(f[positive], 1 / f[positive] ** 2), 0, atol=1e-10)
    assert np.isnan(alpha_peak(f[positive], 1 / f[positive] ** 2)[0])


def replay(case, folder):
    i = str(case["i"])
    inputs = old.SSM / "inputs" / i
    ident = old.read(inputs / "identity.json")["group"]
    assert ident["key"] == case["key"] == RECORD and ident["split"] == "validation"
    assert ident["EEG_channels"] == case["channels"] and ident["rate"] == 500
    names = [case["channels"][n] for n in case["valid"]]
    x = np.load(inputs / "unified.npy")
    native = np.load(inputs / "native.npy")[15000:30000, case["channels"].index("CP4")]
    with np.load(old.SSM / "references" / (case["dataset"] + ".npz")) as ref:
        model = bind_reference(dict(ref), np.load(inputs / "projection.npy"), 900)
    with np.load(old.SSM / "fits" / i / "unified/baseline.npz") as fit:
        proxy = fit["prediction"][:, :, 0]
    args = dict(audit_final=False, frame_batch_size=8)
    capped = voltage_power_cap_target(x, model, proxy, **args)
    relaxed = voltage_power_cap_target(
        x, model, proxy, cap=2 * capped["local_input_ratio"].max(), **args
    )
    assert relaxed["clipped_directions"].max() == 0
    np.testing.assert_allclose(relaxed["s"], x, atol=1e-10, rtol=0)
    with np.load(old.PARENT / "targets" / i / "baseline.npz") as saved:
        np.testing.assert_array_equal(saved["original_channel_index"], case["valid"])
        np.testing.assert_array_equal(x[:, case["valid"]], saved["x"])
        np.testing.assert_allclose(capped["s"][:, case["valid"]], saved["v3"], atol=1e-10, rtol=0)
    ch = case["channels"].index("CP4")
    waves = {
        "X": x[:, ch],
        "S": capped["s"][:, ch],
        "O": capped["o"][:, ch],
        "cap_relaxed": relaxed["s"][:, ch],
        "X_CAR": x[:, ch] - x[:, case["valid"]].mean(axis=1),
    }
    section = slice(6000, 12000)
    curves = {k: spectrum(v[section])[1] for k, v in waves.items()}
    f = spectrum(waves["X"][section])[0]
    native_f, native_p = spectrum(native, 500)
    np.testing.assert_array_equal(f, native_f[: len(f)])
    curves["native"] = native_p[: len(f)]
    hz, _ = alpha_peak(f, curves["X"])
    index = np.flatnonzero(f == hz).item()
    _, cross = signal.csd(
        waves["S"][section], waves["O"][section], fs=200, window="hann", nperseg=400, noverlap=200
    )
    cross = 2 * cross.real
    np.testing.assert_allclose(curves["X"], curves["S"] + curves["O"] + cross, atol=1e-12)
    rows = []
    for key, p in curves.items():
        top, db = alpha_peak(f, p)
        rows.append(
            dict(
                signal=key,
                peak_hz=top,
                peak_db=db,
                at_x_peak_density=p[index],
                at_x_peak_excess_db=excess(f, p)[index],
            )
        )
    pd.DataFrame(rows).to_csv(folder / "component_peaks.csv", index=False)
    pos = (f >= 1) & (f <= 45)
    pd.DataFrame(
        dict(hz=f[pos], cross_SO_twice=cross[pos], **{k: v[pos] for k, v in curves.items()})
    ).to_csv(folder / "spectra.csv", index=False)

    times = []
    for start, stop in [(30, 40), (40, 50), (50, 60), (30, 60)]:
        for length in (2, 4):
            freq, p = spectrum(waves["X"][200 * start : 200 * stop], seconds=length)
            top, db = alpha_peak(freq, p)
            times.append(dict(start_s=start, stop_s=stop, welch_s=length, peak_hz=top, peak_db=db))
    pd.DataFrame(times).to_csv(folder / "time_checks.csv", index=False)

    sites = []
    for name, channel in zip(names, case["valid"]):
        freq, p = spectrum(x[section, channel])
        top, db = alpha_peak(freq, p)
        region = (
            "顶枕"
            if name.startswith(("P", "O"))
            else (
                "额前与额"
                if name.startswith(("Fp", "AF", "F")) and not name.startswith("FC")
                else "中央与中央顶" if name.startswith(("C", "FC")) else "颞"
            )
        )
        sites.append(
            dict(
                channel=name,
                region=region,
                peak_hz=top,
                peak_db=db,
                at_cp4_peak_db=excess(freq, p)[index],
                at_cp4_peak_density=p[index],
            )
        )
    sites = pd.DataFrame(sites)
    sites.to_csv(folder / "scalp.csv", index=False)
    frames = (capped["frame_time_s"] >= 30) & (capped["frame_time_s"] < 60)
    band = BANDS.index((8, 13))
    kept = continuous_indices(17980, ORDER)
    intervals = np.split(kept, np.flatnonzero(np.diff(kept) > 1) + 1)
    stats = dict(
        record=RECORD,
        unit=case["unit"],
        interval_200=[162001, 168001],
        cp4_peak_hz=hz,
        cap_off_max_abs_uv=float(np.max(abs(relaxed["s"] - x))),
        alpha_n_clipped_frames=int(np.sum(capped["clipped_directions"][frames, band] > 0)),
        alpha_n_frames=int(frames.sum()),
        alpha_median_directions=float(np.median(capped["clipped_directions"][frames, band])),
        peak_cross_fraction=float(cross[index] / curves["X"][index]),
        continuous_kept_hz=[[v[0] * 100 / 17980, v[-1] * 100 / 17980] for v in intervals],
    )
    write_json(folder / "replay.json", stats)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    colors = {"X": "#223e5a", "S": "#ab4820", "O": "#00766d", "cap_relaxed": "#788ca3"}
    for key, ls in [("X", "-"), ("cap_relaxed", ":"), ("O", "--"), ("S", "-.")]:
        axes[0].semilogy(
            f[pos],
            curves[key][pos],
            ls,
            color=colors[key],
            label="放宽上限" if key == "cap_relaxed" else key,
        )
    axes[0].axvspan(8, 13, color="#eac65f", alpha=0.18)
    axes[0].set(xlabel="频率 (Hz)", ylabel="功率谱密度 (µV²/Hz)", title="相同估计器 · 30–60 s")
    axes[0].legend(ncol=2)
    for start, marker in [(30, "o"), (40, "s"), (50, "^")]:
        freq, p = spectrum(waves["X"][200 * start : 200 * (start + 10)])
        mask = (freq >= 5) & (freq <= 18)
        axes[1].plot(
            freq[mask], excess(freq, p)[mask], marker=marker, ms=3, label=f"{start}–{start+10} s"
        )
    axes[1].axhline(0, color="grey", lw=0.7)
    axes[1].set(xlabel="频率 (Hz)", ylabel="高于局部背景 (dB)", title="CP4 · 三个不重叠时段")
    axes[1].legend()
    for j, (label, g) in enumerate(sites.groupby("region", sort=False)):
        axes[2].scatter(j + np.linspace(-0.19, 0.19, len(g)), g.at_cp4_peak_db, color="#288178")
        axes[2].hlines(g.at_cp4_peak_db.median(), j - 0.24, j + 0.24, color="black")
        axes[2].set_xticks(range(j + 1), list(sites.region.drop_duplicates())[: j + 1])
    axes[2].set(ylabel=f"{hz:g} Hz 高于局部背景 (dB)", title="每点一枚电极 · 横线为中位数")
    fig.suptitle("Visual · S14  |  实测 α 峰的时间、空间与 v3 目标分配")
    fig.savefig(folder / "case.png", dpi=210)
    plt.close(fig)


def case_member_plot(x, rows, folder):
    # Display the registered alpha-selected O code's only member in the named crop.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    colors = ["#00766d", "#a95324"]
    records = []
    for (idx, row), role, color, ls in zip(rows.iterrows(), ["成员", "匹配对照"], colors, ["-", "--"]):
        assert row.parent == RECORD and row.site == "CP4"
        wave = x[row.example, row.block].astype(float) * row.scale
        start = row.start + row.block * 2
        label = f"{role} O #{row.O} · {start}–{start+2} s"
        axes[0].plot(np.arange(400) / 200, wave, color=color, ls=ls, lw=1.2, label=label)
        f, p = spectrum(wave)
        pick = (f >= 1) & (f <= 45)
        axes[1].semilogy(f[pick], p[pick], color=color, ls=ls, lw=1.2, label=label)
        hz, db = alpha_peak(f, p)
        records.append(
            dict(
                row=idx,
                role=role,
                start_s=start,
                stop_s=start + 2,
                S=row.S,
                O=row.O,
                alpha_fraction=row.alpha,
                peak_hz=hz,
                peak_db=db,
            )
        )
    axes[0].set(
        xlabel="相对片段起点 (s)", ylabel="实测 X (µV)", title="同一被试、同一电极的两秒实测波形"
    )
    axes[1].axvspan(8, 13, color="#eac65f", alpha=0.18)
    axes[1].set(xlabel="频率 (Hz)", ylabel="功率谱密度 (µV²/Hz)", title="两秒 Hann 窗 · 阴影为 α 频带")
    for ax in axes:
        ax.legend(fontsize=9)
    fig.suptitle("Visual · S14 · CP4  |  种子 43 的 O #4868 与既有对照")
    fig.savefig(folder / "cp4_member.png", dpi=210)
    plt.close(fig)
    pd.DataFrame(records).to_csv(folder / "cp4_matched_pair.csv", index=False)


def token_spectra(folder):
    x = np.load(PRIOR / "vq_screen/x.npy", mmap_mode="r").reshape(-1, 15, 400)
    rows, contrasts = [], []
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, layout="constrained")
    for col, seed in enumerate((17, 43)):
        parent = OUT / "members" / f"formal_v3_{seed}"
        meta = pd.read_csv(parent / "validation_members.csv.gz").set_index("row", verify_integrity=True)
        assert (meta.role == "validation").all()
        crop = meta[(meta.parent == RECORD) & (meta.site == "CP4") & (meta.start == 30)].copy()
        crop["start_s"] = crop.start + 2 * crop.block
        crop["stop_s"] = crop.start_s + 2
        assert sorted(crop.block) == list(range(1, 14))
        crop[["start_s", "stop_s", "S", "O", "alpha"]].to_csv(folder / f"cp4_codes_{seed}.csv")
        matches = pd.read_csv(parent / "matches.csv").query("endpoint == 'alpha'")
        for i, (branch, pairs) in enumerate(matches.groupby("branch", sort=False)):
            token = pairs.token.unique().item()
            spectra = []
            for role, column in [("成员", "member_row"), ("对照", "control_row")]:
                m = meta.loc[pairs[column]]
                assert (m[branch] == token).eq(role == "成员").all()
                if role == "成员":
                    members = m
                else:
                    np.testing.assert_array_equal(members[["person", "site"]], m[["person", "site"]])
                    assert np.all(abs(m.log_rms.values - members.log_rms.values) <= np.log(1.4))
                f, p = spectrum(x[m.example, m.block].T.astype(float))
                db = excess(f, p)
                band = (f >= 8) & (f < 13)
                people = []
                for person in sorted(m.person.unique()):
                    keep = m.person.to_numpy() == person
                    avg = np.mean(db[:, keep], axis=1)
                    people.append(avg)
                    rows.append(
                        dict(
                            seed=seed,
                            branch=branch,
                            token=int(token),
                            role=role,
                            person=person,
                            pairs=int(keep.sum()),
                            mean_alpha_excess_db=float(avg[band].mean()),
                        )
                    )
                spectra.append(np.stack(people))
                plot = (f >= 5) & (f <= 18)
                axes[i, col].plot(
                    f[plot],
                    np.mean(people, axis=0)[plot],
                    ls="-" if role == "成员" else "--",
                    label=role,
                )
            if seed == 43 and branch == "O":
                (hit,) = crop.index[crop[branch] == token]
                pair = pairs[pairs.member_row == hit].iloc[0]
                case_member_plot(x, meta.loc[[pair.member_row, pair.control_row]], folder)
            diff = (spectra[0] - spectra[1])[:, band].mean(axis=1)
            contrasts.append(
                dict(
                    seed=seed,
                    branch=branch,
                    token=int(token),
                    pairs=len(pairs),
                    controls=pairs.control_row.nunique(),
                    people=len(diff),
                    positive=int((diff > 0).sum()),
                    mean_difference_db=float(diff.mean()),
                )
            )
            ax = axes[i, col]
            ax.axhline(0, lw=0.7, color="grey")
            ax.axvspan(8, 13, color="#eac65f", alpha=0.17)
            ax.set_title(f"{branch} #{token} · 种子 {seed} · {len(pairs)} 对 / {len(diff)} 人")
            ax.legend()
    for ax in axes[-1]:
        ax.set_xlabel("频率 (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("实测 X 高于局部背景 (dB)")
    fig.suptitle("冻结 v3 词表  |  训练集选码，验证集同被试、同电极、相近幅度配对")
    fig.savefig(folder / "tokens.png", dpi=210)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(folder / "token_people.csv", index=False)
    pd.DataFrame(contrasts).to_csv(folder / "token_contrasts.csv", index=False)


if __name__ == "__main__":
    sanity()
    assert old.read(OUT / "summary.json")["status"] == "completed"
    (case,) = [c for c in old.read(PRIOR / "plan.json")["cases"] if c["key"] == RECORD]
    FOLDER.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False})
    replay(case, FOLDER)
    token_spectra(FOLDER)
