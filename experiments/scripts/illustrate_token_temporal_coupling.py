#!/usr/bin/env python3
"""Draw an authored ideal token atlas; no datasets, models or run arrays are read.

The curves and probabilities are analytic illustrations, not Tak simulations,
tokenizer outputs, experimental results, or proposed acceptance thresholds.
Reuse the existing bitmap/searchable report exporter for the companion report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Rectangle
import numpy as np

from render_source_diagnostics_report import render_report


ROOT = Path(__file__).resolve().parents[2]
INK, GRAY = "#16324B", "#657586"
BLUE, RED, GREEN, ORANGE = "#3268B2", "#C54B59", "#147D78", "#BE751A"
PALETTE = ["#DAE1E8", "#93CFC8", "#83AEE0", "#EBC491"]


def illustrative_arrays():
    t = np.linspace(-5, 25, 601)
    dt = float(t[1] - t[0])
    activity = 1 / (1 + np.exp(-(t - .25) / .25)) - 1 / (1 + np.exp(-(t - 4.25) / .35))
    eeg = -.85 * activity + .13 * np.exp(-.5 * ((t - 8.5) / 1.4) ** 2)
    kt = np.arange(0, 30 + dt / 2, dt)

    def response(power, tau):
        kernel = kt ** power * np.exp(-kt / tau)
        kernel /= kernel.sum()
        return np.convolve(activity, kernel)[:len(t)]

    hbo = response(3, 1.25)
    # A single fixed illustration scale; there are no per-window data transforms.
    scale = hbo.max()
    hbo /= scale
    hbr = -.38 * response(3.4, 1.3) / scale
    slope = np.gradient(hbo, dt)
    centers = np.arange(-4, 25, 2)
    e_states = np.zeros(15, dtype=int)
    e_states[(centers >= 0) & (centers < 2)] = 1
    e_states[(centers >= 2) & (centers < 6)] = 2
    e_states[(centers >= 6) & (centers < 10)] = 3
    values, derivatives = np.interp(centers, t, hbo), np.interp(centers, t, slope)
    h_states = np.where(values < .08, 0, np.where(values > .88, 2, np.where(derivatives > 0, 1, 3)))

    lags = np.arange(-8, 13, 2)
    q = np.array([.58, .14, .14, .14])[:, None] * np.ones((1, len(lags)))
    excess = np.array([.22 * np.exp(-.5 * ((lags - 2.3) / 1.3) ** 2),
                       .25 * np.exp(-.5 * ((lags - 5.6) / 1.65) ** 2),
                       .20 * np.exp(-.5 * ((lags - 8.6) / 1.6) ** 2)])
    p = q + np.vstack((-excess.sum(axis=0), excess))
    assert np.all((p >= 0) & (p <= 1)) and np.allclose(p.sum(axis=0), 1)
    assert np.allclose((p - q).sum(axis=0), 0)
    peak = int(hbo.argmax())
    level = .55
    rising_time = np.interp(level, hbo[:peak + 1], t[:peak + 1])
    falling_time = np.interp(level, hbo[peak:][::-1], t[peak:][::-1])
    relative = np.linspace(-2, 6, 161)
    rising_future = np.interp(rising_time + relative, t, hbo)
    falling_future = np.interp(falling_time + relative, t, hbo)
    assert abs(np.interp(0, relative, rising_future) - level) < 1e-10
    assert abs(np.interp(0, relative, falling_future) - level) < 1e-10
    return dict(t=t, eeg=eeg, hbo=hbo, hbr=hbr, slope=slope, centers=centers,
                eeg_states=e_states, hb_states=h_states, lags=lags,
                matched_null=q, paired_probability=p, probability_difference=p - q,
                matched_level=np.array(level), matched_times=np.array([rising_time, falling_time]),
                relative_time=relative, rising_future=rising_future, falling_future=falling_future)


def setup_style():
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": FontProperties(fname=str(font)).get_name(),
                         "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRAY,
                         "xtick.color": GRAY, "ytick.color": GRAY,
                         "mathtext.fontset": "stix", "axes.unicode_minus": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def heading(fig, title, subtitle):
    fig.text(.055, .955, title, fontsize=21, weight="bold", va="top")
    fig.text(.055, .895, subtitle, fontsize=12.5, color=GRAY, va="top")


def save_figure(fig, output, name):
    path = output / f"{name}.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def draw_timeline(d, output):
    fig = plt.figure(figsize=(11, 8.3))
    heading(fig, "理想图 1｜token 序列保留快电活动与慢血流响应的先后关系",
            "全部为解析示意；色块是语义类别占位符，不是已学得的编号；示例采用频带功率下降模式。")
    grid = fig.add_gridspec(4, 1, height_ratios=[1.4, 1.5, .48, .48],
                            left=.13, right=.96, top=.83, bottom=.23, hspace=.40)
    axes = [fig.add_subplot(grid[i]) for i in range(4)]
    for ax in axes:
        ax.set_xlim(-5, 25)
        ax.axvline(0, color=GRAY, ls=":", lw=1.2)
    axes[0].plot(d["t"], d["eeg"], color=BLUE, lw=2.8, label="EEG 频带功率变化")
    axes[0].axhline(0, color="#CED6DE", lw=.8)
    axes[0].set_ylabel("功率变化\n（示意单位）")
    axes[0].set_ylim(-1.02, .45)
    axes[0].axvspan(0, 4, color="#F1F3F5", zorder=0)
    axes[0].text(2, .28, "示例事件", ha="center", fontsize=11, color=GRAY)
    axes[0].legend(frameon=False, loc="lower right", fontsize=12)
    axes[1].plot(d["t"], d["hbo"], color=RED, lw=2.8, label="HbO")
    axes[1].plot(d["t"], d["hbr"], color=BLUE, lw=2.5, ls="--", label="HbR")
    axes[1].axhline(0, color="#CED6DE", lw=.8)
    axes[1].set_ylabel("浓度变化\n（示意单位）")
    axes[1].set_ylim(-.57, 1.2)
    axes[1].legend(frameon=False, loc="upper right", ncol=2, fontsize=12)
    axes[1].annotate("响应延迟、展宽、缓慢恢复", xy=(10, float(np.interp(10, d["t"], d["hbo"]))), xytext=(14, .72),
                     fontsize=12, arrowprops=dict(arrowstyle="->", color=GRAY))
    for ax, states, prefix, label in [(axes[2], d["eeg_states"], "E", "EEG-S"),
                                      (axes[3], d["hb_states"], "H", "fNIRS-S")]:
        for center, state in zip(d["centers"], states):
            ax.add_patch(Rectangle((center - 1, 0), 2, 1, fc=PALETTE[int(state)], ec="white", lw=2))
            ax.text(center, .5, f"${prefix}_{state}$", ha="center", va="center", fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=8)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False, bottom=False)
    axes[-1].set_xticks(np.arange(-5, 26, 5))
    axes[-1].set_xlabel("事件相对时间（秒）；每个色块标记一个 2 秒输出位置")
    fig.text(.13, .135, "$E_0$ 基线　$E_1$ 功率下降开始　$E_2$ 持续变化　$E_3$ 恢复", fontsize=13)
    fig.text(.13, .092, "$H_0$ 基线　$H_1$ 上升　$H_2$ 高位 / 峰附近　$H_3$ 恢复", fontsize=13)
    fig.text(.055, .035, "当前方案：每个 token 可见完整 30 秒。图示先后是离线位置关系，不能据此估计因果延迟。",
             fontsize=11.8, color=GRAY)
    return save_figure(fig, output, "01_temporal_motif")


def draw_lag_atlas(d, output):
    fig = plt.figure(figsize=(11, 7.2))
    heading(fig, "理想图 2｜同任务、同事件时序之外，仍存在阶段性的配对关联",
            "所有概率均手工设定；没有样本、置信区间或显著性检验。横轴峰位只是绘图设定。")
    ax = fig.add_axes([.14, .46, .72, .34])
    image = ax.imshow(d["probability_difference"] * 100, cmap="RdBu_r", vmin=-35, vmax=35,
                      aspect="auto", extent=[-9, 13, 3.5, -.5], interpolation="nearest")
    ax.set_yticks(range(4), ["基线 $H_0$", "上升 $H_1$", "峰附近 $H_2$", "恢复 $H_3$"])
    ax.set_xticks(d["lags"])
    ax.set_title("以 EEG 的“变化开始”短序列为锚点", loc="left", pad=12)
    ax.axvline(0, color=INK, ls="--", lw=1)
    for row in range(1, 4):
        lag = d["lags"][d["probability_difference"][row].argmax()]
        ax.plot(lag, row, "o", ms=12, mfc="none", mec=INK, mew=1.3)
    cax = fig.add_axes([.885, .46, .02, .34])
    cb = fig.colorbar(image, cax=cax)
    cb.set_label("配对概率 − 重配概率（百分点）", fontsize=12)
    ax2 = fig.add_axes([.14, .16, .72, .20])
    ax2.plot(d["lags"], d["paired_probability"][2], "o-", color=GREEN, lw=2.3,
              label="同一记录的配对（理想）", ms=5)
    ax2.plot(d["lags"], d["matched_null"][2], "s--", color=GRAY, lw=1.8,
              label="同条件整试次重配", ms=4)
    ax2.fill_between(d["lags"], d["matched_null"][2], d["paired_probability"][2], color=GREEN, alpha=.12)
    ax2.axvline(0, color=INK, ls="--", lw=1)
    ax2.set_ylabel("峰附近 $H_2$\n出现概率")
    ax2.set_xlim(-9, 13)
    ax2.set_ylim(0, .48)
    ax2.set_xticks(d["lags"])
    ax2.set_xlabel("fNIRS 输出位置相对 EEG 序列起点的偏移（秒）；正值表示 Hb 在后")
    ax2.legend(frameon=False, fontsize=11, loc="upper left")
    fig.text(.055, .045, "红色：超额出现；蓝色：较少出现；圆圈：示意峰位。真实分析须计入全上下文与离线滤波。",
             fontsize=11.5, color=GRAY)
    return save_figure(fig, output, "02_excess_lag_atlas")


def draw_state_memory(d, output):
    fig = plt.figure(figsize=(11, 6.6))
    heading(fig, "理想图 3｜同样的幅度，token 短序列区分上升与恢复",
            "解析曲线的两个时刻；仅配平 HbO 幅度。图示用于解释动态表型，不是生理状态可识别性证据。")
    left = fig.add_axes([.11, .29, .35, .49])
    right = fig.add_axes([.59, .29, .36, .49])
    left.plot(d["hbo"], d["slope"], color="#8D9EAD", lw=2)
    left.axhline(0, color="#D6DEE5", lw=1)
    left.axvline(float(d["matched_level"]), color=GRAY, ls="--", lw=1)
    for t, color, label, text_y in zip(d["matched_times"], [GREEN, ORANGE], ["上升 $H_1$", "恢复 $H_3$"], [.145, -.13]):
        y = np.interp(t, d["t"], d["slope"])
        left.plot(float(d["matched_level"]), y, "o", color=color, ms=9)
        left.annotate(label, xy=(float(d["matched_level"]), y), xytext=(.08, text_y),
                      color=color, fontsize=13, arrowprops=dict(arrowstyle="->", color=color))
    for index in (196, 310):
        left.annotate("", xy=(d["hbo"][index + 8], d["slope"][index + 8]),
                      xytext=(d["hbo"][index], d["slope"][index]),
                      arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    left.set_xlabel("HbO 幅度（示意单位）")
    left.set_ylabel("HbO 变化率（示意单位 / 秒）")
    left.set_title("幅度相同，动态阶段不同", loc="left", fontsize=15)
    left.set_ylim(float(d["slope"].min()) - .035, float(d["slope"].max()) + .05)
    for curve, color, label, style in [(d["rising_future"], GREEN, "$H_0$ → $H_1$：仍在上升", "-"),
                                      (d["falling_future"], ORANGE, "$H_2$ → $H_3$：正在恢复", "--")]:
        right.plot(d["relative_time"], curve, color=color, label=label, lw=2.8, ls=style)
    right.plot(0, float(d["matched_level"]), "o", color=INK, ms=7)
    right.axvline(0, color=GRAY, ls=":", lw=1.2)
    right.axvspan(0, 6, color="#EEF3F6", zorder=0)
    right.text(3, 1.07, "后续轨迹", ha="center", fontsize=12, color=GRAY)
    right.set_ylim(0, 1.17)
    right.set_xlim(-2, 6)
    right.set_ylabel("HbO 幅度（示意单位）")
    right.set_xlabel("距幅度匹配时刻的时间（秒）")
    right.set_title("短序列对应不同后续变化", loc="left", fontsize=15)
    right.legend(frameon=False, loc="lower left", fontsize=11)
    fig.text(.11, .17, "希望保留的是：阶段、变化方向、持续时间与历史依赖。", fontsize=14, color=INK)
    fig.text(.11, .115, "平滑与自回归也可产生这种形态；增量价值需与幅值、斜率和自身历史基线比较。", fontsize=12, color=GRAY)
    fig.text(.055, .04, "当前全局编码器能看到后续波形，因此这里只能作回顾性描述；预测检验须在编码前隔离目标。",
             fontsize=11.5, color=GRAY)
    return save_figure(fig, output, "03_history_dependent_states")


def report_content(figures):
    def para(text):
        return {"type": "paragraph", "text": text}

    def figure(path, caption):
        return {"type": "figure", "path": str(path), "preserve_bitmap": True,
                "caption": caption, "source": "本次解析示意；生成数组见 illustration_arrays.npz，非实测或模型输出。"}

    return {
        "title": "理想的 token 时序耦合图谱",
        "subtitle": "当前 S/O 架构下的科学叙事与证据要求｜概念示意，不是实验结果",
        "date": "2026-11-05",
        "summary": [
            "最有说服力的对象是具有可重复观测表型的短 token 序列：EEG 的频带状态变化与随后 Hb 的上升、峰附近和恢复阶段形成概率性的跨模态关系；关系在控制任务时序与信号历史后仍有增量，并能帮助解释独立测量。",
            "图中波形、编号、概率及峰位全部为构造值；未读取任何原始数据、受保护数组或模型权重，未训练模型或运行实测实验。示意曲线不是 Tak 求解，不代表其独特预测，也不是成功阈值。"],
        "provenance": [
            "架构依据：docs/EXPERIMENT_PLAN.md 的“30 秒全局交互与潜在词表健康度起点”；可执行合同：experiments/configs/physiology_semantic_tokenizer/dual_component_latent_health_v1.yaml。",
            "历史结果只引用原报告和 summary.json；本报告不更新项目状态、todo、实验计划或历史证据。"],
        "sections": [
            {"title": "1. 当前架构适合把什么保留到 token 中", "blocks": [
                para("最新验证起点使用 30 秒序列、15 个 2 秒 patch、同位置同模态的全局交互和潜在硬量化。S/O 分别编码与解码；fNIRS 同一位置的 HbO/HbR 联合输入。每个 ID 可依赖完整 30 秒，解码器也读取同分支全部 ID。首轮实测只覆盖 Hb；EEG 新监督仍在另行验证，不能把旧电压版结果直接迁移到新方案。"),
                para("因此优先分析冻结 tokenizer 输出的短序列及其在未参与训练记录中的观测表型。图中的 E0—E3 与 H0—H3 是方便阅读的类别占位符；真实分析应在训练集定义类别、序列和排序，并保留原始 ID、位置、模态、分支与种子。相同数字没有跨模态含义。全局解码器下，同一 ID 也不必对应唯一局部波形；需检查上下文条件下的观测分布及语义稳定性。"),
                para("主分析放在跨模态预训练模型之前，便于判断生理结构是否已进入离散表示。教师层检查条件分解是否合理；后续模型检查这些结构是否提高独立测量的补全或预测。三层分别回答问题，不用同一张关联图重复证明三个阶段。") ]},
            {"title": "2. 图一：从快电活动到慢血流响应的离散序列", "blocks": [
                figure(figures[0], "上下共用事件时间轴。上方是示意 EEG 频带功率变化，下方为示意 HbO/HbR；色块每 2 秒标记一个输出位置。E 类别表示基线、变化开始、维持和恢复；H 类别表示基线、上升、峰附近和恢复。时间与幅值均为说明性设定。"),
                para("希望看到同一 EEG 短序列在多个独立被试或会话中，反复对应一个延迟、展宽的 Hb 响应族，且反应强度、持续时间和恢复阶段能随观测条件变化。它应是多对多、具有分散性的关系；不要求每次出现同一编号后恰好在固定秒数出现另一个编号。"),
                para("此处选用频带功率下降仅为一个示例。EEG 电压正负、某一频带功率和神经驱动不是同一个量；符号关系须按脑区、任务和频带分别核验。HbO/HbR 同编码器内的关联属于联合形态保留，不能当成独立的 EEG–fNIRS 耦合证据。Tak 的流量、容量及恢复动力学为阶段性响应提供建模动机，但图示并未验证 Tak 特有机制。"),
                para("支持的叙事：离散表示保留了与生理响应时间尺度相容的观测形态。若只在教师解码曲线上出现，证据限于先验传递；需要在独立观测上重现，才能形成描述性生理表型。") ]},
            {"title": "3. 图二：去掉共同任务时钟后，还剩什么关联", "blocks": [
                figure(figures[1], "上图每格为：给定 EEG 变化开始短序列时，某个 Hb 阶段出现的概率，减去同条件整试次重配的概率。红色表示超额出现、蓝色表示不足；四行共同构成互斥阶段，故每列差值之和为零。下图单独展开峰附近阶段的配对概率与重配概率。"),
                {"type": "equation", "latex": r"\Delta P_{a b}(\ell)=P(M^H_{t+\ell}=b\mid M^E_t=a,C_t)-P_{\mathrm{pair\ null}}(M^H_{t+\ell}=b\mid M^E_t=a,C_t)",
                 "caption": "用于描述图谱的超额配对概率。", "explanation": "M 是预先定义的 token 类别或短序列，a/b 是各模态自己的类别；正偏移表示 Hb 输出位置在后。C 包括预定的任务、事件时间及记录条件。概率分母是该 EEG 模式的有效配对次数；独立样本量仍按被试或独立记录计算。"},
                para("理想图出现有宽度、有阶段顺序的超额关联带：上升阶段较早，峰附近与恢复阶段随后。不要按测试集最亮的单格挑选结论。选择规则、延迟范围、频带、ROI 和类别都应先冻结；稀有模式、失败及完整分母应可见。按整被试汇总，并处理扫描多个编号、延迟和脑区的多重比较。"),
                para("整试次重配应保留任务条件、事件相对时间、会话及模态内部自相关。该对照依赖条件可交换性，不能自动消除慢漂移或系统性生理因素。更强的检验还须控制 Hb 自身历史、幅度/斜率、EEG 自身历史以及可用的运动和外周生理记录。Kirilina 等表明任务相关的浅表血流也可产生 fNIRS 变化，所以共同任务锁定不够支持脑神经血流解释。"),
                para("支持的叙事：在明确控制范围内，token 保留了记录特异的跨模态时序关联。当前全局 token 只能支持离线、回顾性的输出位置关联；正偏移、反向不对称或互信息峰本身都不能识别生理传导时间。") ]},
            {"title": "4. 图三：从幅度类别推进到有历史依赖的动态表型", "blocks": [
                figure(figures[2], "同一条解析 HbO 曲线在上升和恢复时经过相同幅度。左图纵轴为变化率；箭头指示时间前进。右图把两个幅度匹配时刻对齐为零，显示此前与此后的轨迹。灰色区域是后续时间，不是隐藏预测区间。这里只配平 HbO，没有配平 HbR、斜率或事件阶段。"),
                para("若 token 只表示“高、中、低”，可能只是幅度压缩。更有价值的短序列能区分启动、维持、恢复和历史积累：当前观测水平相近，却对应不同的后续变化。驻留时间也应随实际响应持续时间变化，不能只复现固定任务段长度。"),
                para("这与动态系统中观测幅度不能完整指定内部状态的思想相容，但平滑、自回归和普通时序网络同样可产生这种形态。真实增量应在配平 HbO/HbR、幅度、斜率、事件条件及自身历史后判断；也应看跨会话稳定性，而不是把示例环形轨迹称为已恢复的血管状态。"),
                para("当前编码器已经看过完整未来片段，图三可用于回顾性语义分析。若声称预测后续变化，必须在所有预处理和编码之前隔离评价目标，重新编码可见支持，或另用经过核验的因果/可见前缀配置。只遮住若干输出 ID，会留下其他 token 携带的目标信息。") ]},
            {"title": "5. 哪些比较能把图谱提升为科学证据", "blocks": [
                {"type": "table", "columns": ["希望支持的主张", "最低限度的增量证据"], "widths": [1, 2.3], "rows": [
                    ["token 有可重复生理表型", "训练集固定类别与图形选择；新被试/会话的实际观测重现形态与时间尺度；报告上下文依赖和失败。"],
                    ["保留跨模态耦合信息", "超出事件时间、模态边际、自相关和自身历史；在共同独立测量终点上有收益。"],
                    ["物理引导提供额外价值", "同输入、总码率、上下文、训练暴露和监督信息，比较直接量化、粗糙度匹配平滑、经验时序教师及可区分的机制近邻。"],
                    ["S/O 有选择性的功能分工", "配对加伪迹时 S 的动态表型稳定、变化主要进入 O；真实神经样变化作为正对照，要求 S 保留。O 可以包含其它真实生理信息。"],
                    ["捕获具体生理机制", "另需独立干预或可识别性证据；上述图形和冻结预测收益不能单独到达该层。"]]},
                para("最有说服力的终点是：在同一 HbO/HbR 测量坐标、同一冻结训练尺度及同一目标隔离条件下，物理引导 token 比同监督经验对照更准确地解释独立观测。若有合格预测分布可比较 proper score；目前没有时，使用明确定义的测量误差。不同词表的 token 交叉熵不可直接排名。"),
                {"type": "equation", "latex": r"G=\mathrm{MSE}_{C}-\mathrm{MSE}_{P}",
                 "caption": "一个简单的共同观测终点：正值表示物理引导方案误差更小。", "explanation": "P 为物理引导方案，C 为同监督、同预算的经验时序对照；先按冻结训练尺度分别计算 HbO/HbR 的有效样本平方误差，再在试次/记录内、被试内汇总，最后被试等权。区间按完整被试重采样，保留失败；若最终使用别的主终点，应在评价前固定。"},
                para("这些图形没有一种能先验保证其它设计无法产生。只有当独立测量增益和选择性扰动响应同时成立，并胜过合适经验对照时，才可将结果归因于本次所检验的物理引导方案；具体 Tak 机制的独特性还要与非线性经验模型和机制近邻比较。") ]},
            {"title": "6. 与保留结果的距离，以及可采用的论文表述", "blocks": [
                para("2026-10-29 旧电压版的冻结 S-token 检验是另一代模型的探索性证据：23 名被试，2080/2091 个计划试次。主层最大正偏移超额互信息分别为 Single-Trial 0.001064 bit（4 秒）和 Simultaneous 0.002563 bit（2 秒），两库校正后的正偏移关联 p 均为 0.596；主层特定 token 对经校正后为 0/23 名被试达到 p≤0.05。该结果尚未建立稳健的正偏移耦合，也不能用于判定新 30 秒方案已经成功或必然失败。"),
                para("完成相应独立验证后，可写：本研究的离散表示在独立记录中保留了与神经血流响应相容的阶段结构和跨模态时间依赖；在控制任务时序及信号历史后，这些结构仍提供额外的测量预测信息。若物理方案胜过同监督经验对照，再补充其增量收益。当前应写为待检验假设，而不是既得结论。"),
                para("优先顺序：先确认新词表的硬重建质量和上下文语义，再做训练期固定、独立记录验证的表型图谱；随后检验条件关联与配对扰动，最后使用严格隔离的共同测量终点。无需为此把三个建模阶段都增加一套时序声明。") ]}
        ],
        "sources": [
            "当前设计与合同：docs/EXPERIMENT_PLAN.md；experiments/configs/physiology_semantic_tokenizer/dual_component_latent_health_v1.yaml。",
            "主张与竞争解释：docs/EXPERIMENT_PLAN.md，物理模型独特贡献与非任务证据设计（2026-11-04）；docs/reports/research_status_20261104_v1/todo.md。",
            "保留数值事实源：experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261029_s_token_lags_v1/summary.json；对应 REPORT.md 及 dual_component_token_lags_v1.yaml。",
            "Tak et al. (2015). Dynamic causal modelling for functional near-infrared spectroscopy. 血流量、容量与恢复动力学的建模依据；不作为本模型效果证据。 https://www.fil.ion.ucl.ac.uk/~wpenny/publications/tak-penny15.pdf",
            "Ritter et al. (2009). Rolandic alpha and beta EEG rhythms’ strengths are inversely related to fMRI-BOLD signal in primary somatosensory and motor cortex. 说明频带/区域依赖的符号关系；其 BOLD 结果不能直接充当本研究 HbO/HbR 结论。 https://pubmed.ncbi.nlm.nih.gov/18465747/",
            "Kirilina et al. (2012). The physiological origin of task-evoked systemic artefacts in functional near infrared spectroscopy. 任务相关浅表生理变化的竞争解释。 https://pubmed.ncbi.nlm.nih.gov/22426347/"
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    setup_style()
    d = illustrative_arrays()
    np.savez_compressed(output / "illustration_arrays.npz", **d)
    figures = [draw_timeline(d, output), draw_lag_atlas(d, output), draw_state_memory(d, output)]
    content = output / "report_content.json"
    content.write_text(json.dumps(report_content(figures), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checks = render_report(content, output / "report")
    provenance = dict(kind="authored_ideal_illustration_not_experiment", date="2026-11-05",
                      data_access="none", construction="analytic convolution and authored categorical probabilities; not Tak or tokenizer output",
                      generator=str(Path(__file__).resolve().relative_to(ROOT)),
                      checks=dict(probability_simplex=True, zero_sum_probability_difference=True,
                                  matched_HbO_amplitude=True),
                      figures=[p.name for p in figures],
                      report="report/TECHNICAL_REPORT.pdf", figure_atlas="report/FIGURE_ATLAS.pdf")
    (output / "illustration_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "figures": len(figures),
                      "report_pages": checks["exports"]["TECHNICAL_REPORT.pdf"]["pages"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
