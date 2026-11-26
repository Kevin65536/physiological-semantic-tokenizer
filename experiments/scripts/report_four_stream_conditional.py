"""Summarize the completed conditional reader sequence and verify raw prediction scores."""

from pathlib import Path
import argparse
from html import escape
from tempfile import TemporaryDirectory

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_four_stream_conditional as suite
from src.analysis.conditional_evidence import verify_reader_scores

OUT = suite.OUT
COORD = {
    "native": "截止窗带通",
    "cached": "整记录带通",
    "strict_common": "发布坐标",
    "WG_cached": "词生成迁移",
}
MOD = {"A": "A 跨模态", "B_hb": "Hb 局部交互", "B_eeg": "EEG 局部交互"}
COLORS = ["#327568", "#a04c35", "#305d85", "#875e91"]


def plot(n=1):
    return plt.subplots(1, n, figsize=(11.5, 4.25), squeeze=False, layout="constrained")


def export(folder):
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    doc = fitz.open()
    titles = []

    def page(name, title, intro, fig, caption, src):
        titles.append(title)
        p = doc.new_page(width=900, height=610)

        def text(rect, s, size, color):
            h, _ = p.insert_htmlbox(
                fitz.Rect(*rect),
                escape(s),
                css=f"*{{font-family:sans-serif;font-size:{size}pt;color:{color};line-height:1.3}}",
                scale_low=1,
            )
            if h < 0:
                raise ValueError("text overflow " + s)

        text([26, 14, 874, 40], f"四流条件关系实验　·　2026-11-26　　{len(titles):02}", 10, "#236c66")
        text([26, 42, 875, 83], title, 20, "#18323c")
        text([26, 88, 875, 156], intro, 12, "#18323c")
        fn = folder / (name + ".png")
        fig.savefig(fn, dpi=240)
        plt.close(fig)
        p.insert_image(fitz.Rect(26, 160, 874, 473), filename=str(fn), keep_proportion=True)
        text([26, 484, 874, 575], caption, 12, "#18323c")
        text([26, 577, 874, 606], src, 9, "#236c66")

    bridge = []
    mismatch = []
    for c in ["native", "cached"]:
        df = pd.read_csv(OUT / c / "bridge_cells.csv.gz")
        subj = df.groupby(["arm", "subject"], as_index=False).nmse.mean()
        bridge.append(subj.assign(coordinate=c))
        a = pd.read_csv(OUT / c / "extra_mismatch_by_person.csv")
        mismatch.append(a.assign(coordinate=c))
    bridge = pd.concat(bridge)
    mismatch = pd.concat(mismatch)
    fig, axes = plot(2)
    order = ["uncentered", "exact", "decoded", "pre_VQ", "ID_embeddings", "historical_A_H"]
    labels = ["未中心化历史", "精确连续输入", "ID 解码总和", "量化前向量", "ID 码向量", "旧 Hb 头"]
    for ax, c in zip(axes[0], ["native", "cached"]):
        g = bridge.query("coordinate==@c").pivot(index="subject", columns="arm", values="nmse")[order]
        for i, (_, row) in enumerate(g.iterrows()):
            ax.plot(range(6), row, marker=".", lw=0.8, color=COLORS[i], alpha=0.65)
        ax.plot(range(6), g.mean(), marker="D", color="#172b37", lw=2.4, label="四人平均")
        ax.set_xticks(range(6), labels, rotation=22, ha="right")
        ax.set_title(COORD[c])
        ax.set_ylim(0, 3.2)
        ax.set_ylabel("NMSE（越低越好）")
        ax.legend()
    page(
        "bridge",
        "Hb 精确输入与 ID 解码重建的预测误差接近",
        "n-back 有 15 名训练人、309 个事件和 4 名开发人、91 个事件。每条细线是一名开发被试。前五臂使用全部四位置和完整历史，以相同三折留人规则在训练集选择岭系数。最后一列复用旧神经读出。",
        fig,
        "精确连续输入与解码总和在截止窗为 1.2769 与 1.2708，在整记录带通为 1.0148 与 1.0146。未中心化历史在本读出下分别为 1.1170、0.8186。量化前后差距随坐标改变。两项任务的 Hb 目标也不同，不能由跨任务的绝对误差推断滤波效益。",
        "来源　各坐标 bridge_cells.csv.gz、ridge_selection.csv；解码仅用 S/O 整数，原词序列逐项重放。",
    )
    fig, axes = plot(2)
    for ax, c in zip(axes[0], ["native", "cached"]):
        g = mismatch.query("coordinate==@c")
        for i, (m, train) in enumerate([("hb", "true"), ("hb", "wrong"), ("eeg", "true"), ("eeg", "wrong")]):
            vals = g[(g.modality == m) & (g.training == train)].extra_mismatch_cost
            ax.scatter(vals, i + np.linspace(-0.13, 0.13, len(vals)), c=COLORS, alpha=0.8)
            ax.scatter(vals.mean(), i, marker="|", c="black", s=120)
        ax.axvline(0, color=".55", ls="--")
        ax.set_yticks(range(4), ["Hb 真配对训练", "Hb 错配训练", "EEG 真配对训练", "EEG 错配训练"])
        ax.set_title(COORD[c])
        ax.set_xlabel("交互模型的换错代价 − 加性模型的换错代价")
        ax.set_xlim(-0.043, 0.043)
    page(
        "mismatch",
        "同一 O 供体也会破坏加性模型的预测",
        "换错代价为同一模型的错配输入 NMSE 减真实输入 NMSE。两类模型使用相同供体；正值表示交互模型对换错更敏感。每点为一名开发被试，竖短线为人均值，两个种子先在人内平均。",
        fig,
        "在真配对训练的 Hb 模型中，扣除加性模型的换错代价后，增量在整记录带通为 +0.0004，截止窗为 +0.0174。历史真配对模型仅重放权重；缺失的错配训练加性臂按原预算补拟合。完整训练与推理四格、物理单位 RMSE、偏差、校准斜率及原 300 步曲线可逐项核查。",
        "来源　pairing_matrix.csv、extra_mismatch_by_person.csv、p0_cells.csv.gz、training_curves.csv.gz。",
    )
    paths = [OUT, OUT / "strict_common", OUT / "transfer"]
    allc = pd.concat(
        [pd.read_csv(p / "p1_contrasts.csv", dtype={"candidate": str}) for p in paths], ignore_index=True
    )
    primary = allc.query(
        "not calibrated and input=='aligned' and ((task=='A' and contrast=='over_Hb_capacity') or (task!='A' and contrast=='over_frozen_anchor'))"
    ).copy()
    fig, axes = plot(3)
    for ax, task in zip(axes[0], ["A", "B_hb", "B_eeg"]):
        for i, (c, color) in enumerate(zip(COORD, COLORS)):
            g = primary[(primary.task == task) & (primary.coordinate == c)]
            if len(g):
                r = g.iloc[0]
                ax.errorbar(
                    r.delta,
                    i,
                    xerr=[[r.delta - r.ci_low], [r.ci_high - r.delta]],
                    fmt="o",
                    capsize=3,
                    color=color,
                )
        ax.set_ylim(3.5, -0.5)
        ax.axvline(0, color="0.5", ls="--")
        ax.set_yticks(range(4), list(COORD.values()))
        ax.set_title(MOD[task])
        ax.set_xlim(-0.1, 0.1)
        ax.set_xlabel("对照 NMSE − 候选 NMSE")
    page(
        "primary",
        "完整 Hb 路径和局部 S/O 残差尚未给出稳定增量",
        "A 的两臂共享全 Hb 位置、分支和 15 个历史槽的冻结预测，比较同参数量的 Hb 扩容与 EEG 残差。B 在冻结加性主效应上，先计算同槽及前后一槽的低秩交互，再对时间聚合。残差输出从零开始。",
        fig,
        "前三行是同一 n-back 面板。发布坐标使用同一未带通发布测量构成历史和未来目标；整记录带通只作离线读出。词生成沿用 n-back 权重、尺度、阈值，237 个事件来自 5 人，其中 4 人也参加了原开发集。横线为固定拟合后的逐人 95% bootstrap 描述区间，正值表示候选改善。",
        "来源　p1_contrasts.csv、strict_common/p1_contrasts.csv、transfer/p1_contrasts.csv。各被试内先平均位置、色团、三个目标时段和种子。",
    )
    fig, axes = plot(2)
    for ax, c in zip(axes[0], ["native", "cached"]):
        df = pd.read_csv(OUT / c / "training_curves.csv.gz").query(
            "arm in ['A_H','A_HE','hb_add','hb_interaction']"
        )
        for a, col in zip(["A_H", "A_HE", "hb_add", "hb_interaction"], COLORS):
            u = df[df.arm == a].groupby("step").loss.mean()
            ax.plot(u.rolling(20, min_periods=20).mean(), color=col, label=a)
        ax.set_title(COORD[c])
        ax.set_ylabel("训练批归一化 MSE")
        ax.set_xlabel("更新步数（20 步移动均值）")
        ax.legend(ncol=2, fontsize=9)
    page(
        "training",
        "训练曲线与留人验证回答不同问题",
        "曲线为旧套件 300 步训练损失，图示四个代表臂，两种子等权平均。A_H 与 A_HE 是旧的残差来源替换实验；hb_add 和 hb_interaction 分别为独立拟合的 Hb 加性与交互臂。",
        fig,
        "新套件的基线步数在 60、150 中由训练被试的折外误差选择，残差以折外基线预测构造训练目标。校准另用外层留人、内层重新交叉拟合的预测估计，再冻结仿射校准系数。各模型的固定预算、曲线和选择表独立保留，训练批损失本身不足以判断跨人泛化。",
        "来源　training_curves.csv.gz、readers/*/selection.json、curves.csv.gz、predictions.npz。",
    )
    d = pd.read_csv(OUT / "transfer/spatial_people.csv")
    ss = pd.read_csv(OUT / "transfer/spatial_contrasts.csv")
    fig, axes = plot(2)
    names = ["eeg_fnirs_single_trial", "simultaneous_eeg_nirs", "refed"]
    for ax, k, title in zip(axes[0], ["q4", "alpha"], ["EEG-S 高 α 词族占用率", "连续实测 α 功率占比"]):
        for i, ds in enumerate(names):
            g = d[d.dataset == ds]
            for j, (field, color) in enumerate([(k, COLORS[0]), ("adjusted_" + k, COLORS[1])]):
                pos = i + (j - 0.5) * 0.3
                ax.scatter(
                    g[field] * 100, pos + np.linspace(-0.05, 0.05, len(g)), s=18, color=color, alpha=0.6
                )
                r = ss.query("dataset==@ds and metric==@field").iloc[0]
                ax.errorbar(
                    r["mean"] * 100,
                    pos,
                    xerr=[[(r["mean"] - r.ci_low) * 100], [(r.ci_high - r["mean"]) * 100]],
                    color=color,
                    fmt="D",
                    ms=4,
                    capsize=3,
                    label=("原始" if j == 0 else "分层匹配") if i == 0 else None,
                )
        ax.set_yticks(range(3), ["Single-Trial", "Simultaneous", "REFED"])
        ax.axvline(0, color=".6", ls="--")
        ax.set_xlabel("后部减前部（百分点）")
        ax.set_title(title)
        ax.legend()
    page(
        "spatial",
        "后部空间偏好随数据集及幅值、质量条件而变",
        "164 个完整验证窗，两个前部、两个后部位置。点为人内差，菱形为库内等权均值，区间按人重抽。原始差先按位置算比例；匹配差在同人、记录、条件及幅值和质量分层内比较，以两侧较少机会数加权。",
        fig,
        "幅值用原生电压 RMS 的人内三分位，质量代理用 49–51 Hz 功率占比的人内中位数。所有窗口已通过原面板质量筛查。匹配处理还改变了可比较窗口及其权重；残留差异的解释限于传感器上的表型。共同图谱的 192 个条目是库、传感器、流和词族的组合。",
        "来源　transfer/spatial_people.csv、spatial_contrasts.csv；同一批实测波形的 α 比和原图谱每十秒词族频率均已核验。",
    )
    tr = pd.read_csv(OUT / "transfer/real_interval_reencode.csv")
    fig, axes = plot(2)
    for lag, col in zip([2, 4, 6], COLORS):
        g = tr[tr.target_lag_s == lag].groupby("start_before_cutoff_s").delta_nmse.mean()
        axes[0, 0].plot(g.index, g, marker=".", label=f"{lag}–{lag+2} 秒", color=col)
    g = tr.groupby("start_before_cutoff_s").other_slot_id_change.mean()
    axes[0, 1].plot(g.index, 100 * g, color=COLORS[2], marker=".")
    for ax in axes[0]:
        ax.set_xlabel("被替换的真实两秒段起始时间（相对截止点，秒）")
    axes[0, 0].axhline(0, c=".5", ls="--")
    axes[0, 0].set_ylabel("替换后减替换前 NMSE")
    axes[0, 0].legend()
    axes[0, 1].set_ylabel("其他槽 ID 改变比例（%）")
    page(
        "perturb",
        "替换一个真实 EEG 时间段也会改变其他槽的离散词",
        "每次仅将四个 EEG 通道同一两秒段换成相同条件供体的对应片段，整窗重新编码。基线词逐项复现后，共评估 91 事件 × 15 替换位置。保持 Hb 历史、事件目标与训练模型固定。左图上升表示替换损害预测。",
        fig,
        "未直接替换处的 ID 也发生变化，反映 tokenizer 的整窗上下文。横轴是人为指定的输入支持；曲线按被试和种子等权。三个输出时段均相对于截止点，预测变化和 ID 迁移不足以定位生理传播时延。",
        "来源　transfer/real_interval_reencode.csv；严格截止父坐标的完整历史和冻结 A 模型。",
    )
    vocab = pd.read_csv(OUT / "transfer/hb_dictionary_replication.csv")
    fig, axes = plot(2)
    for ax, b in zip(axes[0], "SO"):
        v = vocab[vocab.branch == b]
        counts = [len(v), v.supported.sum(), v.replicate_majority.sum(), v.replicate_all.sum()]
        ax.barh(np.arange(4), counts, color=["#9caeb2", "#548898", "#308578", "#226159"])
        for j, val in enumerate(counts):
            ax.text(val + 10, j, str(val), va="center")
        ax.set_yticks(range(4), ["全部码字", "训练、开发均有支持", "开发多数人同向", "开发所有人同向"])
        ax.set_xlim(0, 1150)
        ax.invert_yaxis()
        ax.set_title("Hb-" + b)
        ax.set_xlabel("码字数（逐步筛选）")
    page(
        "vocabulary",
        "Hb-O 中有 452 个有支持的词复现了训练期双色趋势方向",
        "按原冻结单模态审计的四库训练与开发面板统计 Hb 全词表。每人每词至少三个有效成员才计入，训练和开发分别至少三人。每人由实测 X 的 HbO 与 HbR 两秒斜率中位数确定象限，训练多数人的方向用作命名。",
        fig,
        "O 支有 553/1024 个词具备支持，452 个在开发多数人同向，237 个在所有有支持开发人同向。S 支分别为 564、411 和 197。按人、记录、位置、条件、阶段及 RMS 档匹配非成员时，词 704 只有 1 名被试有对照，866 只有 3 人。词族描述的是实测成员的形态，生成机制和相对匹配背景的特异性仍须另验。",
        "来源　transfer/hb_dictionary_replication.csv、hb_word_subjects.csv.gz、hb_condition_matched.csv。词典有 1024 个独立整数／分支。",
    )
    joint = pd.read_csv(OUT / "transfer/joint_family_null.csv.gz")
    joint_mean = joint.groupby(["dataset", "modality", "kind", "cell"], as_index=False).excess.mean()
    occ = joint_mean[joint_mean.kind == "occupancy"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 4.25), layout="constrained")
    bound = 100 * occ.excess.abs().max()
    for i, m in enumerate(["eeg", "hb"]):
        for k, ds in enumerate(names):
            ax = axes[i, k]
            g = occ[(occ.modality == m) & (occ.dataset == ds)].sort_values("cell")
            a = 100 * g.excess.to_numpy().reshape(4, 4)
            im = ax.imshow(a, vmin=-bound, vmax=bound, cmap="BrBG", aspect="auto")
            ax.set_xticks(range(4), ["Q1", "Q2", "Q3", "Q4"])
            ax.set_yticks(range(4), ["Q1", "Q2", "Q3", "Q4"])
            if k == 0:
                ax.set_ylabel(m.upper() + " 的 S")
            if i == 0:
                ax.set_title(["Single-Trial", "Simultaneous", "REFED"][k])
            if i == 1:
                ax.set_xlabel("O 词族")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="实测减替代的联合占用率（百分点）")
    page(
        "cooccurrence",
        "S/O 同时占用某些词族的频率高于整窗错位对照",
        "每格为同槽 S、O 词族组合。将 O 的完整 15 槽历史在同人、同记录、同位置、同条件的其他窗之间轮换，枚举时间相隔至少 30 秒的合法偏移。保持两支边际占用及各自窗内词序。先按机会数在人内求频率差，再库内人等权。",
        fig,
        "每模态 656 个传感器窗中，416 个有合法替代；其余 240 个留在覆盖表。Single-Trial 的 Hb 双 Q1 占用高出约 10.0 个百分点。另以同一替代统计全部 256 种相邻联合状态转换，按 14 个窗内相邻机会计数。此处的单格为探索性描述，未进行逐格多重检验，也不作为预测增益证据。",
        "来源　transfer/joint_family_null.csv.gz、sequence_coverage.csv。行列四分组依据各流自身的训练属性，Q1 为低分组。",
    )
    joint_mean.to_csv(folder / "joint_family_summary.csv", index=False)
    bridge.to_csv(folder / "bridge_subjects.csv", index=False)
    primary.to_csv(folder / "primary.csv", index=False)
    refs = []
    cal_rows = []
    for c in ["native", "cached"]:
        df = pd.read_csv(OUT / c / "p0_cells.csv.gz")
        g = df.groupby(
            ["modality", "arm", "training", "input", "subject", "seed"], as_index=False
        ).nmse.mean()
        refs.append(g.assign(coordinate=c))
    for f in (OUT / "readers").glob("*/cells.csv.gz"):
        tab = pd.read_csv(f)
        cal_rows.append(
            tab.groupby(
                ["coordinate", "task", "arm", "input", "seed", "calibrated", "subject", "target"],
                as_index=False,
            )[["nmse", "rmse", "bias", "calibration_slope"]].mean()
        )
    pd.concat(refs).to_csv(folder / "p0_subject_scores.csv", index=False)
    pd.concat(cal_rows).to_csv(folder / "calibration_by_person_target.csv", index=False)

    doc.set_toc([[1, s, i] for i, s in enumerate(titles, 1)])
    doc.subset_fonts()
    doc.save(folder / "REPORT.pdf", deflate=True, garbage=4)
    doc.close()
    summary = dict(
        status="completed",
        experiment_id=suite.CFG["experiment_id"],
        jobs_verified=verify_reader_scores(OUT, suite.TR["seeds"], suite.load_parent, suite.read),
        bridge_nmse=bridge.groupby(["coordinate", "arm"]).nmse.mean().unstack().to_dict("index"),
        primary=primary.to_dict("records"),
        transfer=suite.read(OUT / "p2.json")["components"],
        decision="retain frozen tokenizer; neither replicated quantization loss nor conditional prediction edge established",
        figures=len(titles),
    )
    suite.save(folder / "summary.json", summary)
    with fitz.open(folder / "REPORT.pdf") as doc:
        for p in doc:
            assert len(p.get_image_info()) == 1 and len(p.get_text()) > 100
            im = p.get_image_info()[0]
            assert 200 <= im["width"] * 72 / fitz.Rect(im["bbox"]).width <= 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT / "analysis_v2")
    dest = ap.parse_args().output.resolve()
    if dest.exists():
        ap.error("fresh output required")
    with TemporaryDirectory(dir=dest.parent) as tmp:
        folder = Path(tmp) / "report"
        folder.mkdir()
        export(folder)
        folder.rename(dest)
    print(dest)


if __name__ == "__main__":
    main()
