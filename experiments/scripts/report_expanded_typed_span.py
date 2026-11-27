"""Subject-level contrasts, seed-aligned motif replication, and bitmap evidence figures."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import run_expanded_typed_span as run
from score_expanded_typed_span import known_delay, PATHS
from src.analysis.typed_span_patterns import transfer_selected, pattern_fields
from report_task_pattern_expansion import Report, plot, heat

OUT = run.OUT
TASK = list(run.CFG["tasks"])
CN = {"mental_arithmetic": "心算", "motor_imagery": "运动想象", "wg": "词语生成", "nback": "n-back"}
STREAM = ["EEG-S", "EEG-O", "Hb-S", "Hb-O"]
LAGS = np.arange(-28, 30, 2)
MASS = [f"m{k}" for k in range(-14, 15)]
N = [f"n{k}" for k in range(-14, 15)]
COL = ["#166963", "#ad562e", "#585985", "#787d86"]


def readfits(name, arms=None):
    tab = []
    for arm in arms or run.TR["arms"]:
        for seed in run.TR["seeds"]:
            f = OUT / "fits" / f"{arm}_{seed}"
            done = run.base.read(f / "scored.json")
            if (done["status"], done["arm"], done["seed"]) != ("completed", arm, seed):
                raise ValueError("missing terminal score")
            tab.append(pd.read_csv(f / name).assign(arm=arm, seed=seed))
    return pd.concat(tab, ignore_index=True)


def counts_and_rates(frame):
    p = frame.copy()
    for name in ("true", "wrong", "reverse"):
        p[name] = p[name + "_count"] / p.opportunities
    p["replicated_person"] = (p.true > p.wrong) & (p.true > p.reverse)
    return p


def tables():
    meta = pd.read_csv(OUT / "events.csv")
    scores = readfits("masked.csv.gz")
    dev = np.flatnonzero((meta.split == "validation") & (meta.donor >= 0))
    required = {"true", "wrong", "reverse", "none", "S", "O"}
    for _, d in scores.groupby(["arm", "seed"]):
        if (
            set(d.event) != set(dev)
            or len(d) != len(dev) * 4 * 2 * len(required)
            or set(d.control) != required
            or d.duplicated(["event", "stream", "slots", "control"]).any()
            or not np.isfinite(d.bits).all()
        ):
            raise ValueError("missing or duplicated held-person contrast")
        np.testing.assert_array_equal(d.subject, meta.subject.to_numpy()[d.event])
        np.testing.assert_array_equal(d.task, meta.task.to_numpy()[d.event])
    person = (
        scores.groupby(["arm", "seed", "task", "subject", "stream", "slots", "control"]).bits.mean().unstack()
    )
    for c in ("wrong", "reverse", "S", "O", "none"):
        person[c + "_minus_true"] = person[c] - person.true
    person.reset_index().to_csv(OUT / "masked_people.csv", index=False)
    exact = (
        readfits("exact_orders.csv", ["joint"])
        .groupby(["task", "subject", "seed", "stream", "control"])[["matched", "changes", "all_pairs"]]
        .sum()
    )
    exact["rate"] = exact.matched / exact.changes
    exact.reset_index().to_csv(OUT / "exact_bigram_people.csv", index=False)
    families = []
    for seed in run.TR["seeds"]:
        f = OUT / "fits" / f"joint_{seed}"
        with np.load(f / "tokens.npz") as z, np.load(f / "families.npz") as maps:
            np.testing.assert_array_equal(z["events"], meta.id)
            ids = z["ids"]
            families.append(np.stack([maps[f"stream{i}"][ids[:, i]] for i in range(4)], axis=1))
    selection = pd.read_csv(OUT / "fits/joint_17/selected.csv")
    source = pd.read_csv(OUT / "fits/joint_17/motifs_by_person.csv").assign(seed=17)
    alt, mappings = transfer_selected(*families, selection, meta)
    motifs = counts_and_rates(pd.concat([source, alt.assign(seed=43)], ignore_index=True))
    motifs.to_csv(OUT / "same_motif_people.csv", index=False)
    stats = (
        motifs.groupby(["task", "path", "seed"])
        .agg(
            people=("subject", "size"),
            positive=("replicated_person", "sum"),
            true=("true", "mean"),
            wrong=("wrong", "mean"),
            reverse=("reverse", "mean"),
            hits=("true_count", "sum"),
            opportunities=("opportunities", "sum"),
        )
        .reset_index()
    )
    stats["pass_seed"] = (
        (stats.positive >= np.ceil(0.75 * stats.people))
        & (stats.true > stats.wrong)
        & (stats.true > stats.reverse)
    )
    rep = stats.groupby(["task", "path"]).pass_seed.all().rename("both_seeds")
    stats = stats.merge(rep, on=["task", "path"])
    stats.to_csv(OUT / "same_motif_summary.csv", index=False)
    profiles = readfits("attention.csv.gz")
    profile = (
        profiles.groupby(["task", "arm", "seed", "target", "slots", "path", "control"])[MASS + N + ["null"]]
        .mean()
        .reset_index()
    )
    profile.to_csv(OUT / "attention_profiles.csv.gz", index=False)
    # Separate task/seed-level profiles and person-level controls from the across-task display.
    pooled = profile.groupby(["arm", "target", "slots", "path", "control"])[MASS + N + ["null"]].mean()
    attention_pairs = readfits("paired_attention.csv.gz")
    if len(attention_pairs) != len(dev) * 4 * 2 * 2 * 4 or not np.isfinite(attention_pairs.query_tv).all():
        raise ValueError("incomplete per-event attention replay")
    attention_pairs["peak_agrees"] = attention_pairs.true_peak_s == attention_pairs.wrong_peak_s
    attention_pairs.groupby(["arm", "seed", "task", "subject", "target", "slots", "path"])[
        ["query_tv", "peak_agrees"]
    ].mean().to_csv(OUT / "paired_attention_people.csv")
    shift = readfits("shift_people.csv")
    shift.to_csv(OUT / "shift_people.csv", index=False)
    diagnostics = []
    for (arm, mod, slots, path), g in pooled.groupby(level=[0, 1, 2, 3]):
        u = g.droplevel([0, 1, 2, 3])
        t = u.loc["true", MASS].to_numpy()
        w = u.loc["wrong", MASS].to_numpy()
        k = u.loc["kernel", MASS].to_numpy()
        null = u.loc["true", "null"]
        diagnostics.append(
            dict(
                arm=arm,
                target=mod,
                slots=slots,
                path=path,
                true_peak_s=int(LAGS[t.argmax()]),
                wrong_peak_s=int(LAGS[w.argmax()]),
                kernel_peak_s=int(LAGS[k.argmax()]),
                true_peak_mass=t.max(),
                true_null=null,
                wrong_tv=(abs(t - w).sum() + abs(null - u.loc["wrong", "null"])) / 2,
                kernel_tv=(abs(t - k).sum() + abs(null - u.loc["kernel", "null"])) / 2,
            )
        )
    diagnostic = pd.DataFrame(diagnostics)
    diagnostic.to_csv(OUT / "attention_peaks.csv", index=False)
    return meta, person, exact, stats, rep, selection, profile, diagnostic, mappings


def overview(rep, meta, person):
    fig, ax = plot(1, 2)
    pop = meta.groupby(["task", "split"]).size().unstack().reindex(TASK)
    pop.rename(index=CN, columns={"train": "训练", "validation": "开发"}).plot.bar(
        ax=ax[0], color=COL[:2], rot=0
    )
    ax[0].set(xlabel="", ylabel="事件数", title="按原生事件及被试划分")
    for j, arm in enumerate(("joint", "no_lag")):
        v = person.xs(arm, level="arm").groupby(["task", "subject"]).wrong_minus_true.mean()
        for i, t in enumerate(TASK):
            y = v.loc[t].to_numpy()
            x = i + (j - 0.5) * 0.27
            ax[1].scatter(
                np.linspace(-0.04, 0.04, len(y)) + x, y, color=COL[j], marker=("o", "^")[j], alpha=0.6
            )
            ax[1].plot(
                x,
                y.mean(),
                marker="D",
                ms=8,
                mfc="white",
                mec=COL[j],
                label=("时滞模型", "取消时滞")[j] if i == 0 else None,
            )
    ax[1].axhline(0, color="#333", lw=0.8)
    ax[1].legend()
    ax[1].set(
        xticks=range(4),
        xticklabels=[CN[t] for t in TASK],
        ylabel="错配减真配对（bit / 隐藏词）",
        title="小点为开发被试，空心菱形为均值",
    )
    n = meta.groupby("split").subject.nunique()
    rep.page(
        "design",
        "四任务扩大训练后，按同一被试内的配对差评价",
        f"训练 {n['train']} 人、2,471 个事件，开发 {n['validation']} 人、706 个事件。四条原码本从同一组权重出发，两种时滞设定各训练两个种子，每次 1,536 步、每批 12 个事件。训练覆盖所有纳入事件。",
        fig,
        "每窗相对事件标记为 −5 至 +25 秒，2 秒一词，每个模态四处传感器。同条件错配来自同人同记录的另一不重叠事件。704 个开发事件有合格错配。每人先平均四支路、局部与整模态掩码及两种子，再按人比较。正值表示真配对降低交叉熵。全记录离线滤波先于掩码，整模态隐藏时目标侧没有可见输入。",
        "数据与评分　events.csv、support.csv、prepared.json、masked_people.csv",
    )


def exact_page(rep, exact):
    v = exact.rate.unstack("control").groupby(["task", "stream"]).mean()
    mats = [v[n].unstack("stream").reindex(TASK).values * 100 for n in ("observed", "shuffled")]
    fig, axes = plot(1, 2)
    for ax, mat, title in zip(axes, mats, ["实测顺序", "同一窗口内随机重排"]):
        heat(ax, mat, fmt=".1f", vmin=0, vmax=100)
        ax.set(
            xticks=range(4),
            xticklabels=STREAM,
            yticks=range(4),
            yticklabels=[CN[t] for t in TASK],
            title=title + "（%）",
        )
    advantage = mats[0] - mats[1]
    rep.page(
        "exact_sequences",
        f"真实 ID 的相邻转移在 {(advantage>0).sum()}/16 个任务–支路组合中比随机顺序更常复现",
        "每个格子给出开发集中不同 ID 之间的一步转移，有多少在同任务、同支路、同一传感器位置的训练序列里至少出现一次。比较保留每窗词频的随机顺序。",
        fig,
        "先在每人、种子和支路内合并四个位置的匹配数与可评分转移数，求覆盖率，再对种子和被试等权平均。相邻同词的驻留已剔除。这个比较检验单模态词序的可复现性；跨模态的配对与时差另作评分。不同支路拥有独立词表，数字 ID 只在各自码本内有意义。",
        "计算记录　exact_bigram_people.csv；每个拟合的 tokens.npz 保留四条实际 ID 序列。",
    )


def motif_page(rep, stats, selected):
    fig, axes = plot(1, 2)
    for ax, seed in zip(axes, [17, 43]):
        df = stats[stats.seed == seed]
        mat = df.pivot(index="task", columns="path", values="positive").reindex(TASK).to_numpy()
        ns = df.pivot(index="task", columns="path", values="people").reindex(TASK).to_numpy()
        lab = np.array([[f"{int(a)}/{int(b)}" for a, b in zip(ar, br)] for ar, br in zip(mat, ns)])
        heat(ax, mat / ns, fmt=".2f", vmin=0, vmax=1, tags=lab)
        # Replace the generic heatmap label with numerator and denominator.
        for text, tag in zip(ax.texts, lab.ravel()):
            text.set_text(tag)
        ax.set(
            xticks=range(4),
            xticklabels=PATHS,
            yticks=range(4),
            yticklabels=[CN[t] for t in TASK],
            title=f"种子 {seed}，胜过两项对照的开发人数 / 总人数",
        )
    passed = stats.groupby(["task", "path"]).both_seeds.first().sum()
    rep.page(
        "paired_patterns",
        f"16 个训练入选的跨模态片段中，{passed} 个通过双种子复现筛查",
        "每支路按训练码向量和词频分为 8 族。候选要求 EEG 与 Hb 各连续两个词族都改变，固定传感器对和 0–8 秒间隔。每任务每路径由种子 17 的训练被试选一个候选。",
        fig,
        "分母是该任务有合格配对的开发被试。每人按可用起点计算出现率，真配对须同时高于同条件错配和源序倒转。种子 43 的词族只用训练位置作标签对应，同一候选冻结后重评。两种子均须有至少四分之三的人通过，且两项人均差为正。16 项为探索筛查；罕见片段及零出现保留。",
        "核对表　same_motif_summary.csv、same_motif_people.csv；选取规则及家族对应见实验合同和 summary.json。",
    )


def warm_start_page(rep, exact, stats, selection):
    f = OUT / "fits/joint_17"
    before = (
        pd.read_csv(f / "warm_start_exact.csv")
        .groupby(["task", "subject", "stream", "control"])[["matched", "changes"]]
        .sum()
    )
    before["rate"] = before.matched / before.changes
    groups = []
    for name, v in (("仅热启动", before), ("联合训练", exact)):
        rate = v.rate.unstack("control")
        groups.append(
            (rate.observed - rate.shuffled).groupby(["task", "stream"]).mean().groupby("stream").mean() * 100
        )
    fig, axes = plot(1, 2)
    for j, g in enumerate(groups):
        axes[0].bar(
            np.arange(4) + (j - 0.5) * 0.35, g, width=0.34, color=COL[j], label=("仅热启动", "联合训练")[j]
        )
    axes[0].set(
        xticks=range(4), xticklabels=STREAM, ylabel="覆盖率差（百分点）", title="真实 ID 顺序减每窗随机重排"
    )
    axes[0].axhline(0, color="#333", lw=0.7)
    axes[0].legend()
    picked = (
        selection.assign(
            priority=lambda d: np.minimum(d.train_true - d.train_wrong, d.train_true - d.train_reverse)
        )
        .sort_values("priority", ascending=False)
        .iloc[0]
    )
    task, path = picked.task, int(picked.path)
    pre = counts_and_rates(pd.read_csv(f / "warm_start_motifs.csv"))
    pre = pre[(pre.task == task) & (pre.path == path)][["true", "wrong", "reverse"]].mean()
    after = stats[(stats.task == task) & (stats.path == path)].sort_values("seed")
    for j, values in enumerate([pre.to_numpy(), *after[["true", "wrong", "reverse"]].to_numpy()]):
        axes[1].bar(
            np.arange(3) + (j - 1) * 0.24,
            1000 * values,
            width=0.23,
            color=COL[j],
            label=["仅热启动", "联合 17", "联合 43"][j],
        )
    axes[1].set(
        xticks=range(3),
        xticklabels=["真配对", "同条件错配", "源序倒转"],
        ylabel="每千个可用起点的出现次数",
        title=CN[task] + " " + PATHS[path] + "，训练优势最高的候选",
    )
    axes[1].legend()
    a, b, c, d = pattern_fields(int(picked.word))
    e, h = run.CFG["sites"][task]
    site = int(picked.site)
    rep.page(
        "warm_start",
        "哪些词序结构在联合训练之前已经存在",
        f"左图按人和任务等权比较训练前后的原始 ID 顺序优势。右图固定传感器 {e[site]} / {h[site]}、EEG 词族 {a}→{b}、Hb 词族 {c}→{d} 及相对起点 {int(picked.lag_seconds)} 秒。",
        fig,
        "原始相邻词覆盖率统一在每种表示自己的训练词表中计算。右图用训练槽位对应关系对齐热启动及两个终点的词族，再用相同开发事件统计。每人以两端均有完整两词的起点为分母。右图示例按训练真配对同时超过两个对照的幅度选择。实际整数 ID 可回溯到每个成员。",
        "依据　warm_start_exact.csv、warm_start_motifs.csv（fits/joint_17/）、same_motif_summary.csv。",
    )


def attention_page(rep, profile, diag, mod, slots):
    fig, axes = plot(1, 4, sharey=True)
    df = profile[(profile.target == mod) & (profile.slots == slots)]
    for j, ax in enumerate(axes):
        for arm, ctrl, label, color, style in [
            ("joint", "true", "真配对", COL[0], "-"),
            ("joint", "wrong", "同条件错配", COL[1], "--"),
            ("joint", "kernel", "仅时滞核", COL[2], ":"),
            ("no_lag", "true", "取消显式时滞", COL[3], "-."),
        ]:
            y = df[(df.arm == arm) & (df.control == ctrl) & (df.path == j)][MASS].mean().to_numpy()
            ax.plot(LAGS, y, ls=style, c=color, lw=2, label=label)
        ax.axvline(0, c="#777", lw=0.6)
        ax.set(title=PATHS[j], xlabel="Hb 时间减 EEG 时间（秒）", xticks=[-24, -12, 0, 12, 24])
    axes[0].set_ylabel("每个隐藏查询的注意力质量")
    axes[0].legend(fontsize=9)
    direction = "EEG 补 Hb" if mod else "Hb 补 EEG"
    masking = "整模态隐藏" if slots == 15 else f"隐藏连续 {2*slots} 秒"
    peaks = diag[(diag.arm == "joint") & (diag.target == mod) & (diag.slots == slots)].sort_values("path")
    pairs = pd.read_csv(OUT / "paired_attention_people.csv")
    individual = (
        pairs[(pairs.arm == "joint") & (pairs.target == mod) & (pairs.slots == slots)]
        .groupby(["task", "path"])
        .query_tv.mean()
        .groupby("path")
        .mean()
    )
    peaks_text = "、".join(str(v) for v in peaks.true_peak_s)
    unchanged = int((peaks.true_peak_s == peaks.wrong_peak_s).sum())
    note = (
        "整模态隐藏时，同人同记录的查询相同，源事件又做双射置换，因此两条总体曲线重合是设计结果。"
        if slots == 15
        else "局部遮掩保留同模态上下文，错配后的峰可相同也可改变。"
    )
    note += f"逐事件先比较真、错两行权重（含弃权）的总变差，再求均值，四路为 {individual.min():.3f}–{individual.max():.3f}。"
    rep.page(
        f"attention_{mod}_{slots}",
        f"{direction}，{masking}时的四条注意力曲线",
        f"四路真配对峰依次为 {peaks_text} 秒，错配后 {unchanged}/4 路仍在原峰位。每点为该时差全部键权重之和，按隐藏查询数归一，再按人、任务和种子等权平均。正值表示 Hb 较晚。",
        fig,
        note
        + " 仅核曲线去掉内容点积，取消时滞曲线来自独立拟合。S–S 初始先验中心 +4 秒，范围 −2 至 +12 秒；其他三路 −6 至 +12 秒。",
        "逐人机会数、弃权质量与时差权重　fits/*/attention.csv.gz；全任务汇总　attention_peaks.csv。",
    )


def shift_page(rep):
    df = pd.read_csv(OUT / "shift_people.csv")
    fig, axes = plot(1, 2, sharey=True)
    for mod, ax in enumerate(axes):
        sub = (
            df[df.target == mod]
            .groupby(["task", "arm", "path"])
            .centroid_change_s.mean()
            .groupby(["arm", "path"])
            .mean()
        )
        for j, arm in enumerate(("joint", "no_lag")):
            off = (j - 0.5) * 0.25
            ax.bar(
                np.arange(4) + off, sub.loc[arm], width=0.23, color=COL[j], label=("显式时滞", "取消时滞")[j]
            )
        ax.axhline(0, ls=":", c="#333")
        ax.axhline(4 if mod == 0 else -4, ls="--", c="#333", label="完整跟随内容的参照")
        ax.set(
            title=("Hb 补 EEG", "EEG 补 Hb")[mod], xticks=np.arange(4), xticklabels=PATHS, ylim=(-4.5, 4.5)
        )
        ax.legend(fontsize=9)
    axes[0].set_ylabel("注意力平均时差的改变量（秒）")
    limit = (
        df[df.arm == "joint"]
        .groupby(["task", "target", "path"])
        .centroid_change_s.mean()
        .groupby(["target", "path"])
        .mean()
        .abs()
        .max()
    )
    rep.page(
        "content_shift",
        "将真实源词序向后移 4 秒，注意力的时差是否随之移动",
        f"源 ID 后移两格，完整模型八种路径与方向的平均时差变化最大为 {limit:.3f} 秒。评分限定相同源第 3 至 15 格与目标第 6 至 10 格。独立合成检查在已知 0、4、8 秒延迟上均找回正确峰位。",
        fig,
        "固定每个查询，在相同源位置上重新归一真实键的权重并计算平均时差，然后求移动后减移动前。柱为人内事件平均后再按任务和种子等权汇总。若平均时差完整跟随源内容，Hb 后移对应 +4 秒，EEG 后移对应 −4 秒；窗口与允许范围可衰减响应。该干预作用于词序，未重新拟合教师。",
        "分支、任务及逐被试结果　shift_people.csv。",
    )


def example(rep, selection):
    f = OUT / "fits/joint_17"
    ex = pd.read_csv(f / "examples.csv")
    ranked = selection.assign(
        priority=lambda x: np.minimum(x.train_true - x.train_wrong, x.train_true - x.train_reverse)
    ).sort_values("priority", ascending=False)
    sel = next(v for v in ranked.itertuples() if ((ex.task == v.task) & (ex.path == v.path)).any())
    r = ex[(ex.task == sel.task) & (ex.path == sel.path)].iloc[0]
    with np.load(f / "examples.npz") as z:
        seq = z[r.key + "_ids"]
        times = np.arange(15) * 2 - 4
        fig = plt.figure(figsize=(13, 5.55), layout="constrained")
        grid = fig.add_gridspec(2, 2, width_ratios=[1.3, 1])
        ax = fig.add_subplot(grid[:, 0])
        ax.axis("off")
        table = ax.table(
            cellText=seq.T.astype(str),
            rowLabels=[f"{t:+d}" for t in times],
            colLabels=STREAM,
            cellLoc="center",
            loc="center",
            bbox=[0.09, 0.01, 0.88, 0.93],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for j, start in [(int(r.path) // 2, int(r.eeg_slot)), (2 + int(r.path) % 2, int(r.hb_slot))]:
            for slot in [start, start + 1]:
                table[(slot + 1, j)].set_facecolor("#b3dcd0")
        ax.set_title("四条实际 ID 路径（左列为相对标记的词中心秒数）")
        for mod in (0, 1):
            ax = fig.add_subplot(grid[mod, 1])
            branch = "SO"[int(r.path) // 2 if mod == 0 else int(r.path) % 2]
            ch = 1 if mod == 0 else 2
            hz = 200 if mod == 0 else 10
            t = np.arange(30 * hz) / hz - 5
            observed = z[r.key + f"_obs{mod}"].reshape(-1, ch)
            decoded = z[r.key + f"_{mod}{branch}"].reshape(-1, ch)
            for j in range(ch):
                role = "EEG" if mod == 0 else ("HbO", "HbR")[j]
                ax.plot(t, observed[:, j], color=COL[j], lw=0.65, alpha=0.65, label=role + " 观测")
                ax.plot(
                    t,
                    decoded[:, j],
                    color=COL[2 + j],
                    linestyle="--",
                    lw=0.9,
                    label=role + " " + branch + " 解码",
                )
            start = 2 * (r.eeg_slot if mod == 0 else r.hb_slot) - 5
            ax.axvspan(start, start + 4, alpha=0.13, color="#2f8478")
            ax.set(xlabel="相对事件标记时间（秒）", ylabel="模型输入尺度")
            ax.legend(ncol=2, fontsize=8)
    rep.page(
        "member",
        f"开发集实例：{CN[r.task]}，{r.eeg_site} 与 {r.hb_site} 的两段实际词序",
        f"{r.subject}，{r.record}。选训练优势最高且有开发成员的候选，再取首个匹配实例。路径 {PATHS[int(r.path)]}，两段起点相差 {int(r.lag_s)} 秒。表中绿色单元和波形阴影标出参与匹配的四个词。",
        fig,
        "相邻两行相距 2 秒。模型用完整 30 秒窗口生成整数词，S 表示固定物理教师分量，O 表示完整剩余。右侧绘出处理后观测及所选支路的整数解码。HbO 与 HbR 共享同一 Hb 词；各模态按训练尺度显示。单个实例只能说明该序列确实出现，其跨人复现率见前表。",
        "逐成员身份及完整四流 ID、观测与 S/O 解码　fits/joint_17/examples.csv、examples.npz。",
    )


def heatmap_page(rep):
    with np.load(OUT / "fits/joint_17/example_attention.npz") as z:
        fig, axes = plot(1, 3, sharey=True)
        mx = max(z[k][:, :15].max() for k in ("true", "wrong", "kernel"))
        for ax, key, name in zip(
            axes, ("true", "wrong", "kernel"), ("真配对", "同记录同条件错配", "仅学习的时滞核")
        ):
            a = z[key]
            img = ax.imshow(
                a[:, :15],
                origin="lower",
                extent=[-5, 25, -5, 25],
                aspect="equal",
                vmin=0,
                vmax=mx,
                cmap="viridis",
            )
            ax.plot([-5, 25], [-5, 25], color="white", lw=0.7, linestyle="--")
            ax.set(xlabel="源 EEG 相对标记时间（秒）", title=name + f"　弃权 {a[:,-1].mean():.2f}")
        axes[0].set_ylabel("目标 Hb 相对标记时间（秒）")
        fig.colorbar(img, ax=axes, label="注意力权重（每个查询归一）", shrink=0.76)
        pair = str(z["event"])
        donor = str(z["donor"])
        path = int(z["path_index"])
        delta = float(abs(z["true"] - z["wrong"]).sum(-1).mean() / 2)
    rep.page(
        "example_attention",
        "同一 Hb 目标更换 EEG 源后，逐时格检查注意力",
        f"沿用实例页的 {PATHS[path]} 路径，完整隐藏 Hb 并使用种子 17。三幅图共用色标，行是 Hb 查询、列是 EEG 键。白色虚线为同步位置；线上方表示 Hb 较晚。",
        fig,
        f"含弃权键的每行权重和为 1。对每行真配对与错配分布取绝对差之和的一半，再平均 15 行，本例为 {delta:.3f}。仅核图保留允许范围和学习偏置，将内容点积设为零。两种事件的原始索引保存在随图数组中。",
        "成员 " + pair + "。注意力数组和错配身份见 fits/joint_17/example_attention.npz。",
    )


def report():
    meta, person, exact, stats, passed, sel, profile, diag, mappings = tables()
    version = 1
    while (OUT / f"report_v{version}" / "REPORT.pdf").exists():
        version += 1
    dest = OUT / f"report_v{version}"
    rep = Report(dest)
    overview(rep, meta, person)
    exact_page(rep, exact)
    motif_page(rep, stats, sel)
    warm_start_page(rep, exact, stats, sel)
    for mod in [1, 0]:
        for slots in (15, [3, 6][mod]):
            attention_page(rep, profile, diag, mod, slots)
    shift_page(rep)
    example(rep, sel)
    heatmap_page(rep)
    rep.finish()
    results = dict(
        status="completed",
        fits=4,
        train_events=int((meta.split == "train").sum()),
        development_events=int((meta.split == "validation").sum()),
        support=meta.groupby(["split", "task"])
        .agg(
            events=("id", "size"),
            people=("subject", "nunique"),
            paired=("donor", lambda x: int((x >= 0).sum())),
        )
        .reset_index()
        .to_dict("records"),
        motif_candidates=len(passed),
        both_seeds=int(passed.sum()),
        family_training_alignment=mappings,
        synthetic_delay=known_delay(),
        primary_masked=person.groupby(["arm", "task"])[
            ["wrong_minus_true", "reverse_minus_true", "none_minus_true"]
        ]
        .mean()
        .reset_index()
        .to_dict("records"),
        report=str((dest / "REPORT.pdf").relative_to(OUT)),
        mask_coordinate="offline filtered observation",
        development_only=True,
    )
    run.base.save(OUT / "summary.json", results)
