"""Audit retained stage-2 predictions and export the paired n-back visual analysis."""

import argparse
from html import escape
import json
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

from src.analysis.single_vq_relations import interval

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "experiments/runs/physiology_semantic_tokenizer/four_stream_stage2"
RUNS = {"截止窗": "20261126_v1", "登记缓存": "20261126_cached_v1"}
PEOPLE = ["VP002", "VP005", "VP014", "VP019"]
PALETTE = ["#1d5484", "#ac531b", "#327465", "#795380"]
SEEDS = [17, 43]
INK, TEAL, RUST = "#192f3a", "#15716e", "#a94430"
KEY = ["arm", "seed", "input", "subject", "record", "events", "site", "lag_s", "target"]
COMPARISONS = {
    "A": ("A_H", "A_HE"),
    "Hb B1": ("hb_S", "hb_add"),
    "Hb B2": ("hb_add", "hb_interaction"),
    "EEG B1": ("eeg_S", "eeg_add"),
    "EEG B2": ("eeg_add", "eeg_interaction"),
}


def read(path):
    return json.loads(path.read_text())


def audit_run(path, coord):
    cfg = yaml.safe_load((path / "resolved.yaml").read_text())
    result, replay = [read(path / name) for name in ("summary.json", "verification.json")]
    if result["status"] != "completed" or replay["status"] != "passed" or cfg["training"]["seeds"] != SEEDS:
        raise ValueError("incomplete run or seed mismatch")
    panel = pd.read_csv(path / "panel.csv")
    train = panel.split.eq("train").to_numpy()
    dev = panel.loc[~train].reset_index(drop=True)
    if set(panel.split) != {"train", "validation"} or sorted(dev.subject.unique()) != PEOPLE:
        raise ValueError("unexpected population")
    if panel.id.duplicated().any() or panel.groupby("subject").split.nunique().max() != 1:
        raise ValueError("non-independent split")
    donor = panel.iloc[panel.donor].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        panel[["split", "subject", "record", "condition"]], donor[["split", "subject", "record", "condition"]]
    )
    if sorted(panel.donor) != list(range(len(panel))) or (panel.event == donor.event).any():
        raise ValueError("invalid donor permutation")
    for clock in ("hb", "eeg"):
        if ((panel[clock + "_start"] - donor[clock + "_start"]).abs() < 40 - 1e-7).any():
            raise ValueError("overlapping matched donor")
    normalization = read(path / "target_normalization.json")
    norm = {}
    obs = {}
    refs = []
    with np.load(path / "panel.npz") as z:
        for mod in ("eeg", "hb"):
            y = z[mod + "_target"].astype(float)
            mu = y[train].mean(axis=0)
            sd = y[train].std(axis=0)
            np.testing.assert_allclose(mu, normalization[mod]["mean"], atol=1e-10)
            np.testing.assert_allclose(sd, normalization[mod]["sd"], atol=1e-10)
            obs[mod] = y[~train]
            norm[mod] = (obs[mod] - mu) / sd
            for label, ref in [("训练均值", mu), ("零变化", 0.0)]:
                loss = (((obs[mod] - ref) / sd) ** 2).reshape(len(dev), -1).mean(1)
                for sub, g in dev.groupby("subject"):
                    refs.append(
                        dict(
                            coordinate=coord,
                            modality=mod,
                            reference=label,
                            subject=sub,
                            nmse=loss[g.index].mean(),
                        )
                    )
    saved = pd.read_csv(path / "site_horizon_scores.csv.gz")
    rows = []
    scatter = []
    for (arm, seed), group in saved.groupby(["arm", "seed"]):
        mod = (
            "hb"
            if arm.startswith(("A_", "hb_"))
            or (arm.startswith("continuous") and arm != "continuous_alpha_history")
            else "eeg"
        )
        f = path / f"fits/{arm}_{seed}/dev_predictions.npz" if seed else path / (arm + ".npz")
        y = norm[mod]
        targets = ["HbO", "HbR"] if mod == "hb" else ["alpha_fraction"]
        with np.load(f) as predictions:
            if set(predictions.files) != set(group.input):
                raise ValueError("prediction variant mismatch")
            for variant in predictions.files:
                values = predictions[variant]
                if values.shape != (len(dev), 4, 3 * len(targets)) or not np.isfinite(values).all():
                    raise ValueError("incomplete prediction tensor")
                pred = values.astype(float).reshape(y.shape)
                loss = ((pred - y) ** 2).reshape(len(dev), 4, 3, len(targets))
                for (who, rec), g in dev.groupby(["subject", "record"]):
                    by_cell = loss[g.index].mean(0)
                    for (s, l, c), v in np.ndenumerate(by_cell):
                        rows.append(
                            (
                                arm,
                                seed,
                                variant,
                                who,
                                rec,
                                len(g),
                                cfg[mod + "_sites"][s],
                                (2, 4, 6)[l],
                                targets[c],
                                v,
                            )
                        )
                if (arm, seed, variant) == ("A_HE", 17, "aligned"):
                    physical = pred * np.array(normalization["hb"]["sd"]) + normalization["hb"]["mean"]
                    for c, color in enumerate(targets):
                        scatter.append(
                            dev[["id", "subject", "event", "condition"]].assign(
                                coordinate=coord,
                                site=cfg["hb_sites"][0],
                                lag_s=6,
                                target=color,
                                observed=obs[mod][:, 0, 2, c],
                                predicted=physical[:, 0, 2, c],
                            )
                        )
    scores = pd.DataFrame(rows, columns=KEY + ["nmse"])

    def compare(calc, recorded, keys):
        left = calc.set_index(keys).sort_index()
        right = recorded.set_index(keys).sort_index()
        if left.index.has_duplicates or right.index.has_duplicates or not left.index.equals(right.index):
            raise ValueError("score table identity mismatch")
        np.testing.assert_allclose(left.nmse, right.nmse, rtol=2e-6, atol=1e-8)
        return float((left.nmse - right.nmse).abs().max())

    cell_diff = compare(scores, saved, KEY)
    person = scores.groupby(["arm", "input", "subject", "seed"], as_index=False).nmse.mean()
    subject_diff = compare(
        person, pd.read_csv(path / "subject_scores.csv"), ["arm", "input", "subject", "seed"]
    )
    tab = person.query("input=='aligned'").groupby(["subject", "arm"]).nmse.mean().unstack()
    contrasts = pd.read_csv(path / "contrasts.csv")
    checks = []
    for r in contrasts.itertuples():
        d = tab[r.reference] - tab[r.candidate]
        mean, lo, hi = interval(d, np.repeat("nback", len(d)))
        np.testing.assert_allclose(
            [tab[r.reference].mean(), tab[r.candidate].mean(), mean, lo, hi],
            [r.reference_nmse, r.candidate_nmse, r.difference, r.ci_low, r.ci_high],
            atol=1e-6,
        )
        if int((d > 0).sum()) != r.improved or len(d) != r.people:
            raise ValueError("contrast denominator mismatch")
        checks.append(
            dict(
                coordinate=coord,
                contrast=r.contrast,
                reference=r.reference,
                candidate=r.candidate,
                difference=mean,
                ci_low=lo,
                ci_high=hi,
                people=len(d),
                improved=int((d > 0).sum()),
                reference_nmse=tab[r.reference].mean(),
                candidate_nmse=tab[r.candidate].mean(),
            )
        )
    tasks = scores[["arm", "seed"]].drop_duplicates()
    counts = read(path / "prepared.json")["counts"]
    if (
        counts != {"train": int(train.sum()), "validation": len(dev)}
        or result["fits"] != (tasks.seed != 0).sum()
        or result["continuous_controls"] != (tasks.seed == 0).sum()
    ):
        raise ValueError("run coverage drift")
    led = pd.read_csv(path / "support.csv")
    if set(led.loc[led.status == "complete", "id"]) != set(panel.id) or not led.id.is_unique:
        raise ValueError("window ledger mismatch")
    return dict(
        panel=panel,
        sites=scores.assign(coordinate=coord),
        subjects=person.assign(coordinate=coord),
        contrasts=pd.DataFrame(checks),
        scatter=pd.concat(scatter),
        simple=pd.DataFrame(refs),
        split=led.groupby(["split", "status"]).size().rename("events").reset_index(),
        decomposition=read(path / "input_coordinates.json")["nonphysical_control"],
        qa=dict(
            run=path.name,
            cell_rows=len(scores),
            max_cell_error=cell_diff,
            max_subject_error=subject_diff,
            verified_contrasts=len(checks),
            fits=result["fits"],
            continuous=result["continuous_controls"],
        ),
    )


def add_differences(scores, keys):
    tab = scores.query("input=='aligned' and seed!=0").pivot(index=keys, columns="arm", values="nmse")
    return pd.concat(
        [
            (tab[ref] - tab[arm]).rename("difference").dropna().reset_index().assign(contrast=label)
            for label, (ref, arm) in COMPARISONS.items()
        ],
        ignore_index=True,
    )


def axes(n=1, m=1):
    return plt.subplots(n, m, figsize=(11.8, 4.7), squeeze=False, layout="constrained")


def grid(ax, axis="x"):
    ax.grid(axis=axis, color="#e4ebed", lw=0.8)
    ax.set_axisbelow(True)


def heat(ax, a, xlabels, ylabels, limit, fmt="+.3f"):
    img = ax.imshow(a, aspect="auto", cmap="BrBG", norm=TwoSlopeNorm(0, -limit, limit))
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    for (i, j), v in np.ndenumerate(a):
        ax.text(
            j,
            i,
            format(v, fmt),
            ha="center",
            va="center",
            fontsize=10,
            color="white" if abs(v) > limit * 0.6 else INK,
        )
    return img


def overview(comp):
    fig, ax = axes()
    a = ax[0, 0]
    ticks = []
    labels = []
    for i, name in enumerate(["A", "hb B1", "hb B2", "eeg B1", "eeg B2"]):
        ticks.append(i)
        labels.append(
            {
                "A": "A  跨模态",
                "hb B1": "Hb  加入 O",
                "hb B2": "Hb  交互项",
                "eeg B1": "EEG  加入 O",
                "eeg B2": "EEG  交互项",
            }[name]
        )
        for coord, dy, mark, col in zip(RUNS, [-0.14, 0.14], ["o", "s"], [TEAL, RUST]):
            if name.startswith("eeg") and coord == "登记缓存":
                continue
            r = comp[(comp.coordinate == coord) & (comp.contrast == name)].iloc[0]
            a.errorbar(
                r.difference,
                i + (0 if name.startswith("eeg") else dy),
                xerr=[[r.difference - r.ci_low], [r.ci_high - r.difference]],
                fmt=mark,
                color=col,
                capsize=4,
                ms=7,
            )
    a.axvline(0, color=INK, lw=1)
    grid(a)
    a.set_yticks(ticks, labels)
    a.invert_yaxis()
    a.set_xlabel("配对差（对照 NMSE − 候选 NMSE）　　正值表示候选误差较低")
    a.legend(
        handles=[
            Line2D([], [], marker=m, color=c, label=k) for k, m, c in zip(RUNS, ["o", "s"], [TEAL, RUST])
        ],
        loc="upper right",
    )
    a.margins(y=0.15)
    return fig


def seed_plot(d):
    fig, arr = axes()
    ax = arr[0, 0]
    keys = ["A", "Hb B1", "Hb B2", "EEG B2"]
    pivot = d.pivot(index=["coordinate", "subject"], columns=["contrast", "seed"], values="difference")
    ix = [(c, s) for c in RUNS for s in PEOPLE]
    values = pivot.loc[ix, [(k, n) for k in keys for n in SEEDS]].to_numpy()
    img = heat(
        ax,
        values,
        [f"{k}\n种子 {n}" for k in keys for n in SEEDS],
        [f"{c} · {s}" for c, s in ix],
        np.abs(values).max(),
    )
    ax.axhline(3.5, color="white", lw=3)
    fig.colorbar(img, ax=ax, label="配对 NMSE 差", shrink=0.85)
    return fig


def topo(d):
    sub = (
        d[d.contrast.eq("A")]
        .groupby(["coordinate", "target", "site", "lag_s"], as_index=False)
        .difference.mean()
    )
    fig, arr = axes(2, 2)
    sites = ["AFF3h", "AFF4h", "P3h", "P4h"]
    for row, coord in enumerate(RUNS):
        for col, target in enumerate(["HbO", "HbR"]):
            a = arr[row, col]
            val = (
                sub[(sub.coordinate == coord) & (sub.target == target)]
                .pivot(index="site", columns="lag_s", values="difference")
                .loc[sites, [2, 4, 6]]
            )
            img = heat(a, val.to_numpy(), ["2–4 s", "4–6 s", "6–8 s"], sites, sub.difference.abs().max())
            a.set_title(coord + " · " + target)
    fig.colorbar(img, ax=arr.ravel().tolist(), shrink=0.85, label="Hb 历史减去加入 EEG 后的 NMSE")
    return fig, sub


def swapping(subjects):
    fig, arr = axes(1, 2)
    groups = [
        ("A_HE", "A  真配对训练"),
        ("A_wrong", "A  错配训练"),
        ("hb_interaction", "Hb 交互  真配对训练"),
        ("hb_wrong", "Hb 交互  错配训练"),
        ("eeg_interaction", "EEG 交互  真配对训练"),
        ("eeg_wrong", "EEG 交互  错配训练"),
    ]
    cell = subjects.groupby(["coordinate", "arm", "subject", "input"]).nmse.mean().unstack()
    d = (cell.mismatched - cell.aligned).dropna().rename("difference").reset_index()
    lim = d.difference.abs().max() * 1.15
    for a, coord in zip(arr[0], RUNS):
        sub = d[d.coordinate.eq(coord)]
        for i, (arm, _) in enumerate(groups):
            values = sub[sub.arm.eq(arm)].set_index("subject").loc[PEOPLE].difference
            a.scatter(values, i + np.linspace(-0.13, 0.13, 4), s=28, color=PALETTE)
            a.scatter(values.mean(), i, s=65, marker="|", linewidths=2.5, color=INK)
        a.axvline(0, color="0.55", lw=1)
        grid(a)
        a.set_xlim(-lim, lim)
        a.invert_yaxis()
        a.set_title(coord)
        a.set_yticks(range(len(groups)), [g[1] for g in groups])
        a.set_xlabel("错配输入 NMSE − 真配对输入 NMSE")
    return fig, d


def reference_plot(subjects, simple):
    fig, arr = axes(
        1,
        2,
    )
    cols = ["#335d88", "#177770"]
    for a, coord in zip(arr[0], RUNS):
        mean = (
            subjects[subjects.coordinate.eq(coord) & subjects.input.eq("aligned")]
            .groupby(["arm", "subject"])
            .nmse.mean()
            .groupby("arm")
            .mean()
        )
        for (left, right, tag), color, marker in zip(
            [("A_H", "A_HE", "词元读取器"), ("continuous_H", "continuous_HE", "连续 Hb 岭回归")],
            cols,
            ["o", "s"],
        ):
            y = [mean[left], mean[right]]
            a.plot([0, 1], y, marker=marker, color=color, lw=2.4, ms=8, label=tag)
            for j, v in enumerate(y):
                a.annotate(
                    f"{v:.3f}",
                    (j, v),
                    xytext=(-7 if j == 0 else 7, 6),
                    textcoords="offset points",
                    ha="right" if j == 0 else "left",
                    fontsize=10,
                    color=color,
                )
        level = simple.query("coordinate==@coord and modality=='hb' and reference=='训练均值'").nmse.mean()
        a.axhline(level, ls="--", color="0.5", label="始终预测训练均值")
        a.set(
            xlim=(-0.29, 1.29),
            ylim=(0, 3.55),
            xticks=[0, 1],
            xticklabels=["Hb 历史", "Hb 历史 + EEG 词元"],
            title=coord,
        )
        grid(a, "y")
        a.set_ylabel("开发被试等权 NMSE（越低越好）")
    arr[0, 1].legend(loc="upper right", fontsize=9)
    return fig


def calibration(frame):
    fig, arr = axes(1, 4)
    rows = []
    for col, (target, coord) in enumerate((t, c) for t in ("HbO", "HbR") for c in RUNS):
        sub = frame[(frame.target == target) & (frame.coordinate == coord)]
        allv = frame[frame.target == target][["observed", "predicted"]].to_numpy()
        lo, hi = allv.min(), allv.max()
        pad = (hi - lo) * 0.07
        lo -= pad
        hi += pad
        a = arr[0, col]
        for s, color, marker in zip(PEOPLE, PALETTE, ["o", "s", "^", "D"]):
            g = sub[sub.subject.eq(s)]
            a.scatter(g.observed, g.predicted, c=color, marker=marker, s=16, alpha=0.8, label=s)
        a.plot([lo, hi], [lo, hi], ls="--", color="0.55")
        a.axhline(0, color="0.84", lw=0.8)
        a.set(
            xlim=(lo, hi), ylim=(lo, hi), aspect="equal", xlabel="实测变化（µM）", title=f"{coord} · {target}"
        )
        if col == 0:
            a.set_ylabel("预测变化（µM）")
        corr = sub.observed.corr(sub.predicted)
        ratio = sub.predicted.std() / sub.observed.std()
        rmse = np.mean((sub.predicted - sub.observed) ** 2) ** 0.5
        a.set_xlabel(f"实测变化（µM）\nr = {corr:.2f}   SD 比 = {ratio:.2f}\nRMSE = {rmse:.2f} µM")
        rows.append(
            dict(coordinate=coord, target=target, events=len(sub), r=corr, sd_ratio=ratio, rmse_uM=rmse)
        )
    return fig, pd.DataFrame(rows)


def attribution(data):
    rows = []
    for coord, info in data.items():
        for mod, r in info["decomposition"].items():
            for i, quantity in enumerate(["S 平均平方幅度", "S 一阶差平均平方"]):
                rows.append(
                    (
                        coord,
                        mod,
                        quantity,
                        r["train_physical_S_moments"][i],
                        r["train_alternative_S_moments"][i],
                    )
                )
            rows.append((coord, mod, "X 开发重建 MSE", r["dev_physical_X_mse"], r["dev_alternative_X_mse"]))
    d = pd.DataFrame(rows, columns=["coordinate", "modality", "quantity", "physical", "alternative"])
    d["ratio"] = d.alternative / d.physical
    fig, arr = axes(1, 2)
    for a, coord in zip(arr[0], RUNS):
        for i, label in enumerate(["S 平均平方幅度", "S 一阶差平均平方", "X 开发重建 MSE"]):
            for mod, dy, marker, color in [("eeg", -0.13, "s", "#345f86"), ("hb", 0.13, "o", TEAL)]:
                v = d.query("coordinate==@coord and quantity==@label and modality==@mod").ratio.iat[0]
                a.scatter(v, i + dy, color=color, marker=marker, s=52, label=mod.upper() if i == 0 else None)
                a.annotate(
                    f"{v:.2g}×",
                    (v, i + dy),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                )
        a.axvline(1, ls="--", color="0.4")
        a.set_xscale("log")
        a.set_xlim(0.35, 4000)
        a.set_ylim(2.5, -0.5)
        a.set_yticks(range(3), ["S 平均平方幅度", "S 一阶差平均平方", "X 开发重建 MSE"])
        a.set_xlabel("替代分解 / 物理分解（倍，对数刻度）")
        a.set_title(coord)
        a.legend()
    return fig, d


def timing():
    fig, arr = axes()
    a = arr[0, 0]
    a.barh(2, 30, left=-26, height=0.4, color="#b5d6d0")
    a.barh(1, 2, left=2, height=0.4, color="#438c86")
    for l in [6, 8, 10]:
        a.barh(0, 2, left=l, height=0.4, edgecolor="white", color="#cc805e")
        a.text(l + 1, -0.42, f"{l}–{l+2}", ha="center", fontsize=10)
    a.axvline(0, lw=1, color="0.55", ls=":")
    a.axvline(4, lw=1, color=INK, ls="--")
    a.text(0, 2.42, "任务提示", ha="center")
    a.text(4, 2.42, "预测截止", ha="center")
    a.set(
        yticks=[0, 1, 2],
        yticklabels=["未来实测目标窗", "变化量的参考窗", "四流编码历史"],
        ylim=(-0.8, 2.85),
        xlim=(-27, 13),
        xticks=[-26, -20, -10, 0, 4, 6, 8, 10, 12],
        xlabel="相对各自设备上任务标记的时间（秒）",
    )
    grid(a)
    return fig


class Report:
    def __init__(self, folder):
        self.out = folder
        (folder / "figures").mkdir()
        self.doc = fitz.open()
        self.titles = []

    @staticmethod
    def text(page, box, txt, size, color=INK):
        remaining, _ = page.insert_htmlbox(
            fitz.Rect(*box),
            escape(txt),
            css=f"*{{font-family:sans-serif;font-size:{size}pt;line-height:1.35;color:{color}}}",
            scale_low=1,
        )
        if remaining < 0:
            raise ValueError(f"Text overflow: {txt}")

    def page(self, slug, title, lead, fig, caption, source):
        self.titles.append(title)
        n = len(self.titles)
        p = self.doc.new_page(width=905, height=628)
        self.text(
            p,
            (27, 13, 885, 37),
            f"阶段二实验核查　／　Simultaneous n-back　　·　　2026-11-26　　　{n:02}",
            10,
            TEAL,
        )
        self.text(p, (26, 38, 885, 80), title, 22)
        self.text(p, (27, 83, 880, 145), lead, 12.8)
        path = self.out / "figures" / f"{n:02}_{slug}.png"
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=r"Glyph .* missing from font")
            fig.savefig(path, dpi=245, facecolor="white")
        plt.close(fig)
        p.insert_image(fitz.Rect(26, 145, 879, 484), filename=str(path), keep_proportion=True)
        self.text(p, (27, 493, 880, 587), caption, 12.8)
        self.text(p, (27, 596, 880, 624), source, 9.4, TEAL)

    def finish(self):
        self.doc.set_toc([[1, t, i] for i, t in enumerate(self.titles, 1)])
        self.doc.set_metadata({"title": "阶段二实验核查 · EEG 与血氧未来属性", "author": "PID-MCM"})
        self.doc.subset_fonts()
        self.doc.save(self.out / "REPORT.pdf", garbage=4, deflate=True)
        self.doc.close()


def export(dest):
    plt.rcParams.update(
        {
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "font.size": 11,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titleweight": "bold",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.unicode_minus": False,
            "legend.frameon": False,
        }
    )
    data = {k: audit_run(SUITE / path, k) for k, path in RUNS.items()}
    native, cached = data.values()
    pd.testing.assert_frame_equal(
        native["panel"], cached["panel"][native["panel"].columns], check_exact=False, atol=1e-9, rtol=0
    )
    pd.testing.assert_frame_equal(native["split"], cached["split"])

    def combine(k):
        return pd.concat([v[k] for v in data.values()], ignore_index=True)

    sites, subs, contr, scatter, simple = [
        combine(k) for k in ["sites", "subjects", "contrasts", "scatter", "simple"]
    ]
    for k in ["sites", "subjects"]:
        eeg = native[k].arm.str.startswith("eeg_") | native[k].arm.eq("continuous_alpha_history")
        cols = native[k].columns.drop("coordinate")
        pd.testing.assert_frame_equal(
            native[k].loc[eeg, cols].reset_index(drop=True),
            cached[k].loc[cached[k].arm.isin(native[k].loc[eeg, "arm"]), cols].reset_index(drop=True),
        )
    by_person = add_differences(subs, ["coordinate", "subject", "seed"])
    by_site = add_differences(sites, ["coordinate", "subject", "seed", "site", "target", "lag_s"])
    book = Report(dest)
    book.page(
        "effects",
        "EEG 未降低总体血氧预测误差，S/O 交互也未超过加性模型",
        "A 检验脑电（EEG）能否在血氧（Hb）历史之外降低归一化均方误差（NMSE），两组读出参数量相同。B1 比较只用物理教师解释成分 S 与加上剩余成分 O 的非线性加性读出。B2 在加性模型上加入低秩交互。四名开发被试均参与全部比较。",
        overview(contr),
        "点为四人的平均配对差，横线为固定模型后按被试重抽 2,000 次的 95% 百分位区间。登记缓存的 Hb B1 平均改善 0.0354，3/4 人改善，区间为 [−0.0411, +0.1084]。EEG B2 平均为 −0.0380，四人误差均上升。两套处理共用完全相同的 EEG 结果，图中只绘一次。",
        "来源　两次运行的 contrasts.csv、subject_scores.csv；工作簿「配对比较」。区间为逐项描述，未作多重比较校正。",
    )
    book.page(
        "support",
        "同一批事件在截止后 2、4、6 秒开始评分",
        "每个事件有 EEG 和 Hb 两个原生标记。分别读取截止前 30 秒，每个传感器发出 15 个两秒词。预测目标为未来两秒的 HbO/HbR 均值或 EEG α 功率占比，相对于历史末两秒的变化。A、B 共用四个位置和三个时段。",
        timing(),
        "437 个候选中剔除 33 个上游质量排除、2 个与质量排除区重叠、1 个缺少错配供体和 1 个工频污染事件。15 人 309 事件用于训练，4 人 91 事件用于开发。截止窗的 Hb 历史仅在窗内带通，目标取发布测量；登记缓存的历史与目标均来自整记录离线带通。两者的目标尺度随之改变。",
        "来源　panel.csv、support.csv、resolved.yaml、target_normalization.json。每格平方误差除以同位置、时段和属性的训练方差，再在人内汇总事件与各格，平均种子后被试等权。",
    )
    book.page(
        "seed_subject",
        "加入 EEG 后，两套 Hb 处理各只有一人平均改善",
        "每格是同一被试、同一种子下的对照减候选 NMSE。列内 17 与 43 为两次独立初始化，每行是同一人的重复事件。开发事件数依 VP002、VP005、VP014、VP019 顺序为 23、22、23、23。",
        seed_plot(by_person),
        "截止窗中只有 VP005 的 A 增量在两个种子均为正（+0.031、+0.035）。登记缓存中 VP002 的两次结果一正一负，其余三人两次均为负。EEG B2 在 VP002 上的退化最大，两个种子分别为 −0.134 和 −0.102；种子平均后，其余三人的交互误差也上升。",
        "来源　两次运行的 subject_scores.csv，工作簿「逐人逐种子」。两套处理共享 EEG 数据，因此末两列上下重复。",
    )
    fig, local = topo(by_site)
    book.page(
        "sites",
        "登记缓存中 HbO 与 HbR 的增量方向相反",
        "每格是一个传感器、一个色团（氧合 HbO 或脱氧 HbR）和一个截止后时段的 A 增量，按种子和被试等权平均。AFF3h、AFF4h 为额区传感器，P3h、P4h 为顶区传感器。四图共用以零为中心的色标，正值表示 EEG 降低误差。",
        fig,
        "登记缓存跨位置平均后，HbO 三个时段依次为 +0.024、+0.037、+0.034，HbR 为 −0.019、−0.054、−0.036。截止窗的 HbO 为 +0.003、+0.002、−0.032，HbR 三段均为负。24 格中，登记缓存 12 格为正，截止窗 11 格为正。这些相关格子只用于定位误差来源。",
        "来源　两次运行的 site_horizon_scores.csv.gz，工作簿「A位置色团时段」。时间以预测截止点为零。位置解释限于原生传感器。",
    )
    fig, swap = swapping(subs)
    book.page(
        "mismatch",
        "Hb 交互读出对 S/O 配对敏感，仍未超过加性对照",
        "每行固定一组模型，再把同人、同记录、同任务条件的分离事件作为供体。A 仅替换 EEG；Hb 与 EEG 交互行仅替换 O。真配对和错配训练分别拟合。横坐标为错配输入减真配对输入的误差，正值表示换错后误差上升。",
        fig,
        "每行四点自上而下依次为 VP002、VP005、VP014、VP019，黑色短线为四人平均。真配对训练的 Hb 交互读出换错 O 后，截止窗与登记缓存的 NMSE 分别上升 0.045 和 0.070。A 换错 EEG 后分别变化 −0.004 和 −0.001。配对敏感性和相对强加性基线的增量，由两项不同对照回答。",
        "来源　subject_scores.csv 中的 aligned 与 mismatched 输入，工作簿「换输入差」。错配供体在双时钟上均至少相隔 40 秒。",
    )
    book.page(
        "history_controls",
        "连续 Hb 历史基线的强弱随处理方式改变",
        "蓝线为冻结词元读取器，绿线为连续 Hb 历史岭回归。两条线均比较各自的历史基线与增加 EEG 词元后的误差；岭回归增加冻结词的固定投影特征。灰虚线是逐位置、时段、色团始终预测训练期均值的参考误差。",
        reference_plot(subs, simple),
        "截止窗的连续读出从 3.093 降至 2.738，2/4 人改善，配对差区间跨零，而且两端均高于训练均值参考 1.336。登记缓存的连续 Hb 历史达到 0.596，加入 EEG 后升至 0.760，四人均变差。比较处理方式时，历史与评分目标同时改变；这些数值不足以单独归因于滤波或量化。",
        "来源　continuous_H.npz、continuous_HE.npz、panel.npz、target_normalization.json；工作簿「简单参照」。训练均值参考仅用保留目标计算。",
    )
    fig, cal = calibration(scatter)
    book.page(
        "heldout_predictions",
        "固定位置和时段后，Hb 预测的变化幅度偏小",
        "用合同中的首个 Hb 位置 AFF3h、最远的 6–8 秒目标窗、首个种子 17，展示 A 的 Hb 加 EEG 预测。每个点为一个开发事件，91 个事件全部绘出。圆、方、三角、菱形依次为 VP002、VP005、VP014、VP019。",
        fig,
        "虚线表示预测等于实测。同色团跨处理共用实际单位与轴限，两个色团各用自己的刻度。SD 比为预测标准差除以实测标准差。截止窗的 HbO/HbR 分别为 0.13/0.22；登记缓存为 0.32/0.44。登记缓存示例的 HbO 相关为 0.56，仍低估变化幅度。图内相关及 RMSE 按事件描述，主要比较仍按人汇总。",
        "来源　fits/A_HE_17/dev_predictions.npz、panel.npz、target_normalization.json。图示值均已逆变换为 µM；工作簿「逐事件示例」。",
    )
    fig, ratios = attribution(data)
    book.page(
        "decomposition",
        "替代分解的能量与重建误差限制了物理归因",
        "替代 S 通过历史窗内低通得到，O 由原信号减去 S 得到。按训练历史的能量与平滑度选截止频率，再用同一码本预算编码。前两行比较训练期替代 S 与物理码解码 S 的矩，第三行比较两路在开发历史上重建 X 的均方误差。",
        fig,
        "虚线 1 倍表示该量匹配。EEG 替代 S 的平均平方幅度约为解码 S 的 1,120 倍。Hb 替代分解在截止窗和登记缓存的重建 MSE 分别约为原分解的 9.1 和 22.0 倍。当前证据可报告四名开发被试的配对误差和局部异质性；把增益归因于物理教师仍需重建质量及成分统计相近的对照。",
        "来源　两次运行的 input_coordinates.json；工作簿「分解匹配」。矩与重建误差使用各运行冻结的输入尺度，图中报告无量纲比值。",
    )
    book.finish()
    with pd.ExcelWriter(
        dest / "分析数据.xlsx", engine="xlsxwriter", engine_kwargs={"options": {"strings_to_urls": False}}
    ) as w:
        for name, table in {
            "核验": pd.DataFrame([v["qa"] for v in data.values()]),
            "样本流向": native["split"],
            "配对比较": contr,
            "逐人逐种子": by_person,
            "A位置色团时段": local,
            "分层增量明细": by_site,
            "换输入差": swap,
            "简单参照": simple,
            "原单位描述": cal,
            "逐事件示例": scatter,
            "分解匹配": ratios,
            "逐人误差": subs,
            "分层误差复算": sites,
        }.items():
            table.to_excel(w, sheet_name=name, index=False)
            sh = w.sheets[name]
            sh.freeze_panes(1, 1)
            sh.autofilter(0, 0, len(table), len(table.columns) - 1)
            for j, c in enumerate(table.columns):
                sh.set_column(j, j, min(86, max(11, len(c) + 2, table[c].astype(str).str.len().max() + 1)))
    with fitz.open(dest / "REPORT.pdf") as doc:
        if len(doc) != len(book.titles):
            raise ValueError("unexpected page count")
        for p in doc:
            images = p.get_image_info()
            if len(images) != 1 or len(p.get_text().strip()) < 140:
                raise ValueError("missing selectable text or figure")
            box = fitz.Rect(images[0]["bbox"])
            dpi = images[0]["width"] * 72 / box.width
            if not 200 <= dpi <= 300:
                raise ValueError(f"unexpected figure DPI {dpi}")
    return [v["qa"] for v in data.values()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SUITE / "20261126_cached_v1" / "visual_analysis_v2")
    dest = parser.parse_args().output.resolve()
    if dest.exists():
        parser.error("choose a fresh export directory")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        out = Path(tmp) / "export"
        out.mkdir()
        qa = export(out)
        out.rename(dest)
    print(json.dumps(qa, indent=2))
    print(dest)


if __name__ == "__main__":
    main()
