"""Verify three v3 endpoints and render the six-panel Hb vocabulary figure and report."""

from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import run_four_stream_evidence as run
from src.inference.conditional_slot_training import new_model, setup, forward

OUT = run.OUT
COLORS = ["#4682ab", "#238277", "#d39735", "#af553f"]
SEED_COLORS = ["#1e6970", "#a95739"]
NAMES = {
    "native": "原生目标／窗内带通",
    "cached": "登记带通缓存",
    "strict_common": "严格同坐标",
    "WG_cached": "词生成／缓存",
}


def verify():
    path, meta, data = run.prev.load_parent("strict_common")
    tr = meta.split.eq("train").to_numpy()
    dev = ~tr
    y = data["hb_target"].astype(float)
    scores = pd.read_csv(OUT / "a/people.csv")
    with np.load(OUT / "a/dev_predictions.npz") as z:
        for key in z.files:
            rep, arm, variant = key.rsplit("_", 2)
            v = z[key]
            loss = (((v - y[dev]) / y[tr].std(0)) ** 2).reshape(dev.sum(), -1).mean(1)
            calc = pd.Series(loss).groupby(meta.loc[dev, "subject"].to_numpy()).mean()
            g = scores[
                (scores.representation == rep) & (scores.arm == arm) & (scores.input == variant)
            ].set_index("subject")
            np.testing.assert_allclose(calc, g.nmse.reindex(calc.index), atol=1e-6)
        best = pd.read_csv(OUT / "a/selection.csv").query("selected")
        for r in best.itertuples():
            if r.eta == 0 and r.arm != "H":
                np.testing.assert_allclose(
                    z[r.representation + "_" + r.arm + "_aligned"],
                    z[r.representation + "_" + r.arm + "_mismatched"],
                    atol=1e-9,
                )
    n = 0
    for coord in run.CFG["B"]["coordinates"]:
        _, m, d = run.prev.load_parent(coord)
        idx = np.flatnonzero(m.split.eq("validation"))
        for mod in ["eeg", "hb"] if coord == "native" else ["hb"]:
            t = setup(d, m, "B_" + mod)
            for seed in run.CFG["B"]["seeds"]:
                p = OUT / "b" / f"{coord}_{mod}_{seed}"
                state = torch.load(p / "models.pt", map_location="cpu", weights_only=True)
                with np.load(p / "predictions.npz") as z, torch.inference_mode():
                    for arm in ["anchor", "additive_residual", "true", "wrong"]:
                        mdl = new_model("B_" + mod, arm, run.prev.TR, 0, t["y"].shape[-1])
                        mdl.load_state_dict(state["models"][arm])
                        for swap in [False] if arm == "anchor" else [False, True]:
                            v = forward(mdl, t, idx, arm, swap) * state["sd"]
                            if arm == "anchor":
                                base = v + state["mu"]
                                v = base
                            else:
                                v = v + base
                            np.testing.assert_allclose(
                                v.numpy(), z[arm + ("_mismatched" if swap else "_aligned")], atol=1e-6
                            )
                n += 1
    draws = pd.read_csv(OUT / "words/composition_draws.csv.gz")
    assert (
        draws.draw.nunique() == 100
        and not draws.duplicated(["draw", "branch", "tier", "zone", "weight"]).any()
    )
    np.testing.assert_array_equal(draws[[f"q{k}" for k in range(4)]].sum(1), draws.denominator)
    assert draws.anti.isna().equals(draws.denominator.eq(0))
    for name in ["discovery", "new_record"]:
        pmf = pd.read_csv(OUT / "patterns" / f"{name}_q1_exact_null.csv.gz")
        np.testing.assert_allclose(pmf.groupby(["dataset", "subject"]).probability.sum(), 1)
        transitions = pd.read_csv(OUT / "patterns" / f"{name}_conditional_transitions.csv.gz")
        g = transitions.query("outgoing>0")
        np.testing.assert_allclose(g.groupby(["dataset", "subject", "origin"])["observed"].sum(), 1.0)
    return dict(b_seed_replays=n, verified_A_predictions=True, word_draws=100, verified_surrogate_rows=True)


class Report:
    def __init__(self, out):
        self.doc = fitz.open()
        self.out = out
        self.titles = []

    def page(self, slug, title, lead, fig, caption, source):
        self.titles.append(title)
        p = self.doc.new_page(width=910, height=645)

        def put(box, s, size, color):
            h, _ = p.insert_htmlbox(
                fitz.Rect(*box),
                escape(s),
                css=f"*{{font-family:sans-serif;font-size:{size}pt;line-height:1.32;color:{color}}}",
                scale_low=1,
            )
            if h < 0:
                raise ValueError("overflow " + s)

        put([27, 14, 883, 35], f"四流词元证据复核　·　2026-11-26　　　{len(self.titles):02}", 10, "#226a67")
        put([27, 42, 883, 80], title, 20, "#1b3541")
        put([27, 87, 883, 148], lead, 12, "#1b3541")
        path = self.out / (slug + ".png")
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=r"Glyph .* missing from font")
            fig.savefig(path, dpi=240, facecolor="white")
        plt.close(fig)
        p.insert_image(fitz.Rect(27, 149, 883, 496), filename=str(path), keep_proportion=True)
        put([27, 505, 883, 604], caption, 12, "#1b3541")
        put([27, 613, 883, 640], source, 9, "#226a67")

    def finish(self):
        self.doc.set_toc([[1, t, i] for i, t in enumerate(self.titles, 1)])
        self.doc.subset_fonts()
        self.doc.save(self.out / "REPORT.pdf", deflate=True, garbage=4)
        self.doc.close()
        with fitz.open(self.out / "REPORT.pdf") as d:
            for p in d:
                (im,) = p.get_image_info()
                dpi = im["width"] * 72 / fitz.Rect(im["bbox"]).width
                assert 200 <= dpi <= 300 and len(p.get_text()) > 150


def plot(n=1, m=1):
    return plt.subplots(n, m, figsize=(11.6, 4.7), layout="constrained", squeeze=False)


def export(folder):
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "font.size": 11,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    rep = Report(folder)
    wc = pd.read_csv(OUT / "words/composition.csv")
    tiers = ["named", "majority", "unanimous"]
    fig, axes = plot(2, 3)
    # Coordinates use sign, not chronological class number: horizontal HbO, vertical HbR.
    for i, b in enumerate("SO"):
        for j, tier in enumerate(tiers):
            ax = axes[i, j]
            r = wc.query("branch==@b and tier==@tier and zone=='sign' and weight=='word_type'").iloc[0]
            ax.axhline(0, color=".8", lw=0.8)
            ax.axvline(0, color=".8", lw=0.8)
            for q in range(4):
                v = r[f"q{q}"] / r.denominator
                x = -1 if q < 2 else 1
                y = (-1, 1)[q % 2]
                ax.scatter(x, y, s=2300 * v + 20, color=COLORS[q], alpha=0.45)
                ax.text(x, y, f"{r[f'q{q}']:.0f}  ·  {100*v:.1f}%", ha="center", va="center")
            ax.set(
                xlim=(-2, 2),
                ylim=(-2, 2),
                xticks=[-1, 1],
                xticklabels=["HbO ↓", "HbO ↑"],
                yticks=[-1, 1],
                yticklabels=["HbR ↓", "HbR ↑"],
            )
            ax.set_title(f"Hb-{b}　{['训练可命名','开发多数复现','开发全部复现'][j]}　n={r.denominator}")
    rep.page(
        "hb_quadrants",
        "Hb-S 的可复现词集中于双色反向趋势，Hb-O 同时覆盖同向和反向",
        "每格分母为不同码字数。横轴为原始 X 成员的 HbO 两秒斜率方向，纵轴为 HbR 方向，标注为词数与比例。每人每词至少三片，训练与开发各至少三人；先取人内中位斜率，再用训练多数人的象限命名。",
        fig,
        "训练可命名词中，Hb-S 的反向趋势词为 399/451（88.5%），Hb-O 为 263/489（53.8%）。开发多数复现的词中分别为 381/411（92.7%）和 238/452（52.7%）。这描述的是两个离散词表如何组织实测成员；同向或反向变化的生理来源仍待独立测量。",
        "来源　words/composition.csv、word_census.csv.gz、subject_word_statistics.csv.gz。四库公开面板，每支总码字数均为 1024。",
    )
    fig, axes = plot(1, 2)
    fm = pd.read_csv(OUT / "words/frequency_matched.csv")
    null = pd.read_csv(OUT / "words/null_contrasts.csv")
    for ax, zone, title in zip(axes[0], ["sign", "q10"], ["按斜率正负命名", "将近零斜率计为不确定"]):
        for j, b in enumerate("SO"):
            r = wc.query("tier=='named' and zone==@zone and branch==@b").set_index("weight")
            v = [
                r.loc["word_type", "anti"],
                r.loc["dev_emissions", "anti"],
                fm.query("tier=='named' and zone==@zone")[b + "_anti"].iat[0],
            ]
            ax.bar(
                np.arange(3) + (j - 0.5) * 0.36, np.array(v) * 100, width=0.34, color=SEED_COLORS[j], label=b
            )
            for i, val in enumerate(v):
                ax.text(i + (j - 0.5) * 0.36, 100 * val + 1, f"{val*100:.1f}", ha="center", fontsize=9)
        ax.set_xticks(range(3), ["词型等权", "开发发词次数加权", "频数与人数分层匹配"])
        ax.set_ylim(0, 108)
        ax.set_ylabel("可命名词中反向趋势的比例（%）")
        ax.set_title(title)
        ax.legend()
    rep.page(
        "word_controls",
        "词频、支持人数和近零阈值均会改变分母，S/O 分工仍可见",
        "近零阈值取每库、每色团训练片段绝对斜率的第 10 百分位；任一色团低于或等于阈值时保留为不确定票。频数及支持人数在训练和开发分别按二倍区间分层，两支共同权重取层内较少的词数。发词权重以保留片段的词元机会数计。",
        fig,
        "按正负命名且频数匹配后，每支有效权重 382 词，反向比例分别为 88.3% 和 49.8%。在库、被试、位置和 RMS 档内联合置换双色斜率 99 次，训练可命名词型的 S−O 实测差 34.7 个百分点，替代差的 2.5%–97.5% 分位约 −11.5 至 +19.8 个百分点。全员复现层的替代词数极少，双支分母均非零的次数为 0/99。",
        "来源　frequency_matched.csv、null_contrasts.csv、dataset_background.csv。条件背景保留词频与幅值档，不保持片段时序；区间为替代分布分位数。",
    )
    a = pd.read_csv(OUT / "a/people.csv")
    fig, axes = plot(1, 2)
    for ax, k, title in zip(
        axes[0], ["continuous", "discrete"], ["精确连续 Hb ＋ EEG 多频段", "冻结 Hb 与 EEG 全部码向量"]
    ):
        g = a.query("representation==@k and input=='aligned'").pivot(
            index="subject", columns="arm", values="nmse"
        )[["H", "HE", "wrong"]]
        for who, row in g.iterrows():
            ax.plot(range(3), row, marker=".", lw=0.9, label=who)
        ax.plot(range(3), g.mean(), marker="D", lw=2, color="#232e36", label="四人均值")
        ax.set(
            xticks=range(3),
            xticklabels=["Hb＋条件", "加入配对 EEG", "错配训练"],
            title=title,
            ylabel="开发被试 NMSE",
        )
        ax.set_ylim(0, 2.8)
    axes[0, 1].legend(ncol=2, fontsize=9)
    rep.page(
        "conditional_A",
        "强 Hb 核基线上，连续与离散 EEG 均未通过真实配对检验",
        "n-back 训练 15 人 309 事件、开发 4 人 91 事件。两类表示使用同一严格截止点、目标与供体，四个位置、15 个历史槽全部可见。每个核块在拟合折标准化并归一化，再加入相同的已知任务条件核；三折留人选择岭系数与 EEG 权重，可取零。",
        fig,
        "连续臂选择 EEG 权重 0，Hb 与 Hb+EEG 的 NMSE 同为 1.1643。离散臂选择 0.1，1.2700 降至 1.2697，只有 1/4 人改善；错配训练为 1.2632，四人均优于真配对训练。连续诊断用 1–4、4–8、8–13、13–30、30–45 Hz 两秒对数带功率。评分按站点、时段、色团的训练方差归一，再按人等权。",
        "来源　a/selection.csv、a/people.csv、a/contrasts.csv。预测目标为三个未来两秒窗的 HbO/HbR 均值减历史末两秒均值。",
    )
    fig, axes = plot(1, 2)
    b = pd.concat([pd.read_csv(OUT / "b/contrasts.csv"), pd.read_csv(OUT / "patterns/wg_b/contrasts.csv")])
    for ax, mod in zip(axes[0], ["eeg", "hb"]):
        for i, c in enumerate(NAMES):
            for j, (ref, n) in enumerate([("over_additive", "同预算加性"), ("over_wrong", "错配乘积")]):
                u = b.query("coordinate==@c and modality==@mod and contrast==@ref")
                if u.empty:
                    continue
                r = u.iloc[0]
                ax.errorbar(
                    r.delta,
                    i + (j - 0.5) * 0.20,
                    xerr=[[r.delta - r.ci_low], [r.ci_high - r.delta]],
                    fmt=["o", "s"][j],
                    color=SEED_COLORS[j],
                    label=n if i == 0 else None,
                    capsize=3,
                )
        ax.axvline(0, color=".5", ls="--")
        ax.set_title(mod.upper())
        ax.set_yticks(range(4), list(NAMES.values()))
        ax.set_ylim(3.4, -0.4)
        ax.set_xlabel("对照 NMSE − 真配对乘积 NMSE")
        ax.legend(fontsize=9)
    rep.page(
        "residual_B",
        "乘积项对错配的优势没有稳定超过加性扩容",
        "固定同一折外加性基线，用完全相同的剩余目标、批次、两个种子与 150 步预算训练三个小头。加性头将两支分别非线性变换后求和，乘积头逐元素相乘，均保留相同时槽、相邻时差和参数量。错配仅替换残差头的 O。",
        fig,
        "n-back 的整记录 Hb，乘积相对加性扩容为 −0.0006，1/4 人改善；同坐标严格预测为 −0.0060。冻结到词生成，EEG 乘积胜过错配乘积（5/5 人），但五人均未胜过加性扩容。横线为种子先在人内平均后按人重抽 2,000 次的描述性 95% 区间。词生成复用五人、237 个事件。",
        "来源　b/contrasts.csv、patterns/wg_b/contrasts.csv。EEG 在 n-back 不重复计入三套 Hb 坐标。",
    )
    qp = pd.read_csv(OUT / "patterns/double_q1_people.csv")
    fig, axes = plot(1, 2)
    g = qp.query("panel=='discovery'")
    for i, (ds, tab) in enumerate(g.groupby("dataset")):
        axes[0, 0].scatter(tab.excess * 100, i + np.linspace(-0.13, 0.13, len(tab)), color=COLORS[i])
    axes[0, 0].set_yticks(range(3), ["Single-Trial", "REFED", "Simultaneous"])
    axes[0, 0].axvline(0, color="0.5")
    axes[0, 0].set_xlabel("双 Q1 占用率减精确错位期望（百分点）")
    new = qp.query("panel=='new_record'").reset_index(drop=True)
    for i, r in new.iterrows():
        axes[0, 1].plot(np.array([r.exact_null_p025, r.exact_null_p975]) * 100, [i, i], color=".5", lw=4)
        axes[0, 1].scatter(100 * r.observed, i, color=COLORS[1], marker="D")
    axes[0, 1].set_yticks(range(len(new)), new.subject)
    axes[0, 1].set_ymargin(0.4)
    axes[0, 1].set_xlabel("实测双 Q1 占用率及错位分布中央 95%（%）")
    axes[0, 0].set_title("原发现面板")
    axes[0, 1].set_title("发现面板未含的词生成记录")
    rep.page(
        "double_Q1",
        "EEG 双 Q1 在两条新记录上仍有正超额，新增被试只有一人",
        "在同被试、同记录、同条件、同位置内错位完整 O 窗，起点至少相隔 30 秒，保留窗内词序与边际占用。同组传感器共享偏移。对候选占用次数卷积所有合法偏移的离散分布。左图每点一人；右图菱形为实测，线段为替代区间。",
        fig,
        "原三库共 14/14 人为正。新记录 VP017 超额 +0.67 个百分点（13 个事件），VP019 为 +2.40（56 个事件）；只有 VP017 是发现面板之外的被试。按起始状态归一后，两人双 Q1 自转移实测约 5.6% 和 15.4%，均落在相应 999 次替代的中央 95% 范围内。共同输入与分解结构仍可能解释共现。",
        "来源　patterns/double_q1_people.csv、*_q1_exact_null.csv.gz、*_null_counts.npz、*_conditional_transitions.csv.gz。条件概率无出边时留空。",
    )
    fig, axes = plot(1, 3)
    sp = pd.read_csv(OUT / "patterns/spatial_people.csv")
    for ax, ds, title in zip(
        axes[0],
        ["eeg_fnirs_single_trial", "simultaneous_eeg_nirs", "refed"],
        ["Single-Trial", "Simultaneous", "REFED"],
    ):
        for k, c, label in zip(
            ["alpha", "teacher_alpha", "decoded_alpha", "q4"],
            COLORS,
            ["实测 α", "教师 S 的 α", "解码 S 的 α", "S-Q4 占用"],
        ):
            v = (
                sp.query("dataset==@ds")
                .groupby("analysis")[k]
                .mean()
                .reindex(["all", "matched_unadjusted", "matched_adjusted"])
            )
            ax.plot(range(3), 100 * v, color=c, marker="o", label=label)
        ax.axhline(0, color=".65", lw=0.8)
        ax.set_ylim(-5, 26)
        ax.set_title(title)
        ax.set_xticks(range(3), ["全部", "同支持", "同支持加权"])
        ax.set_ylabel("后部减前部（百分点）")
    axes[0, 2].legend(fontsize=9)
    rep.page(
        "spatial_ladder",
        "限制共同支持与调整权重都会削弱 EEG-S Q4 后部优势",
        "三个点依次为所有位置窗、仅保留前后部可匹配的层、再按层内较少机会数加权。按人汇总后库内等权。分层条件为同记录、已知任务条件、RMS 三分位及工频代理二分位。四条线分开显示占用率与功率占比。",
        fig,
        "Single-Trial 的 Q4 从 +22.4 到 +13.3 再到 −0.1 个百分点，Simultaneous 从 +13.3 到 +2.2 再到 −2.4。调整后两库教师 S 的 α 差为 +5.6、+3.5 点，解码 S 的 α 差约 −0.07、−0.19 点。结果定位到该读图指标与冻结链条，不据此判断 EEG-O 或独立神经来源。",
        "来源　patterns/spatial_people.csv。原始与调整两端已逐人核对上轮数值；全部 164 个开发窗，14 名被试。",
    )
    rep.finish()


def finalize():
    for stage in ["a", "b", "words", "patterns"]:
        if run.prev.read(OUT / (stage + ".json"))["status"] != "completed":
            raise ValueError("unfinished " + stage)
    qa = verify()
    dest = OUT / "analysis_v1"
    if not dest.exists():
        with TemporaryDirectory(dir=OUT) as tmp:
            folder = Path(tmp) / "report"
            folder.mkdir()
            export(folder)
            folder.rename(dest)
    a = pd.read_csv(OUT / "a/contrasts.csv")
    gate = {}
    for label, g in a.groupby("representation"):
        gate[label] = bool(
            ((g.delta > 0) & (g.ci_low > 0) & (g.improved > g.people / 2)).all() and len(g) == 2
        )
    summary = dict(
        status="completed",
        experiment_id=run.CFG["experiment_id"],
        verification=qa,
        primary_A=a.to_dict("records"),
        EEG_bridge_trigger=gate["continuous"] and not gate["discrete"],
        primary_B=pd.read_csv(OUT / "b/contrasts.csv").query("contrast=='over_additive'").to_dict("records"),
        new_records=run.prev.read(OUT / "patterns.json")["result"]["new_records"],
        report="analysis_v1/REPORT.pdf",
        protected_access=False,
        tokenizer_refit=False,
    )
    run.prev.save(OUT / "summary.json", summary)
    return qa
