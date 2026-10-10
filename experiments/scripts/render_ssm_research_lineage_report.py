#!/usr/bin/env python3
"""Dated synthesis of retained post-2026-09-28 SSM evidence; no data or fitting.

Reads only the explicitly named result summaries/tables and implementation
sources. Produces a Chinese technical PDF and companion HTML with PNG figures.
The report is a communication snapshot, never an experiment/state authority.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import html
import json
from pathlib import Path
import subprocess

import fitz
from fontTools.ttLib import TTFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "experiments/runs/physiology_semantic_tokenizer"
DEFAULT_OUT = ROOT / "docs/report/20261010_ssm_research_lineage_v2"
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
DS = ["eeg_fnirs_single_trial", "simultaneous_eeg_nirs", "visual_cognitive_motivation"]
LABEL = dict(zip(DS, ["Single-Trial", "Simultaneous", "Visual"]))
COLORS = ["#215F94", "#168478", "#9C5F9E"]
RUNS = {
    "solver": "shared_driver_reconstruction/20260928_conditional_step_control_panel_v2",
    "wave": "shared_driver_reconstruction/20260928_waveform_diagnostic_v1",
    "volume": "shared_driver_reconstruction/20260928_waveform_volume_fraction_v1",
    "prevalence": "data_quality_audit/20260928_hbo_hbr_prevalence_v1",
    "phys": "shared_driver_reconstruction/20260928_physiology_semantics_v1",
    "robust": "shared_driver_teacher_robustness/20261001_v1",
    "attribution": "shared_driver_attribution/20261002_v1",
    "modes": "shared_driver_modes/20261002_v1",
    "response": "semantic_response_discovery/20261002_v1",
    "strength": "ssm_strengthening/20261003_v3",
    "s3": "ssm_strengthening/20261003_s3_v1",
    "obs": "observation_semantics/20261009_v2",
    "obs2": "observation_semantics/20261009_refinement_v1",
    "rate": "shared_driver_reconstruction/20261010_eeg_proxy_rate_v1",
    "params": "shared_driver_parameter_stability/20261010_v1",
    "output": "ssm_output_support/20261010_v1",
}
INPUTS: set[str] = set()
FIGURES: dict[str, dict] = {}


def read(path):
    path = Path(path)
    INPUTS.add(str(path.relative_to(ROOT)))
    return json.loads(path.read_text())


def js(run, name="summary.json"):
    return read(ART / RUNS[run] / name)


def csv(run, name):
    path = ART / RUNS[run] / name
    INPUTS.add(str(path.relative_to(ROOT)))
    return pd.read_csv(path)


def setup():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({
        "font.family": [font_manager.FontProperties(fname=str(FONT)).get_name(), "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.unicode_minus": False, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .15,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "mathtext.fontset": "stix", "pdf.fonttype": 42,
    })


def save(fig, out, name, sources, method):
    path = out / "figures" / (name + ".png")
    fig.savefig(path, dpi=240)
    plt.close(fig)
    FIGURES[name] = dict(file=str(path.relative_to(out)), sources=sources, calculation=method)
    return path


def arrow(ax, start, end, *, color="#8C9AAD", bend=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
        linewidth=1.1, color=color, connectionstyle=f"arc3,rad={bend}", zorder=1))


def box(ax, x, y, w, h, text, *, face="#EAF1F8", edge="#517CA5", size=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.009,rounding_size=0.014",
        facecolor=face, edgecolor=edge, linewidth=1.0, zorder=2))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=size,
        color="#20354B", linespacing=1.45, zorder=3)


def draw_lineage(out):
    fig, ax = plt.subplots(figsize=(523/72, 6.5))
    fig.subplots_adjust(left=.015, right=.985, bottom=.02, top=.98)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    box(ax, .12, .888, .76, .092,
        "9/28 观测修复与数值基座 [E0–E3]\nv5 none；同目标求解器；困难波形与 Hb 同向性审计", size=9.3)
    box(ax, .025, .711, .27, .11, "9/28 固定结构语义 [E4]\n参数共享／合成恢复／补全\n重建通过；个体参数未通过")
    box(ax, .365, .711, .27, .11, "10/1 四臂稳健性 [E5]\n额外观测项＋驱动先验\n重建改善；稳定仍可能错误")
    box(ax, .705, .711, .27, .11, "10/2 区域谱模式 [E7]\n宽频＋谱对比两维驱动\n表示稍好；配对特异性不足")
    box(ax, .365, .526, .27, .115, "10/2 成分归属 [E6]\n方向／空间／反事实／原型\n真实动力学可被额外项吸收")
    box(ax, .705, .526, .27, .115, "10/2 响应与语义 [E8]\n衰减对照＋独立模态编码\n语义增量及幅值覆盖失败")
    box(ax, .365, .339, .27, .115, "10/3 有限响应＋空间 [E9]\n初态混淆 → S3 连续合成\n连续性有益；来源仍未定")
    box(ax, .705, .152, .27, .115, "10/9 观测语义 [E10]\nEOG／硬件／竞争机制\n电位关联成立；Hb 来源未定")
    box(ax, .025, .152, .27, .115, "10/10 参数稳定性 [E12]\n共同内点／连续／权重／共享\n单 τ 局部候选；联合未达标")
    box(ax, .365, .152, .27, .115, "10/10 输出支持 [E13]\n连续实测＋输出剖面＋EEG\nHbR 改善；SSM/T 可交换")
    box(ax, .025, .339, .27, .115, "10/10 代理采样率 [E11]\n4→10 Hz 与驱动网格对照\n没有一致隐藏收益", face="#F4F5F7", edge="#8997A5")
    for s,e in [((.30,.888),(.16,.821)),((.5,.888),(.5,.821)),
                ((.295,.766),(.365,.766)),((.635,.766),(.705,.766)),
                ((.5,.711),(.5,.641)),((.84,.711),(.84,.641)),
                ((.635,.583),(.705,.583)),((.5,.526),(.5,.454)),
                ((.5,.339),(.5,.267)),((.635,.394),(.76,.267))]:
        arrow(ax,s,e)
    # Long paths stay in empty gutters and mean methodological inheritance.
    ax.plot([.16,.16],[.711,.49], color="#8C9AAD", lw=1.1, zorder=1)
    arrow(ax,(.16,.49),(.16,.454))
    ax.plot([.295,.32,.32,.16],[.755,.755,.29,.29], color="#8C9AAD", lw=1.1, zorder=1)
    arrow(ax,(.16,.29),(.16,.267))
    ax.text(.18,.478,"工程支线",fontsize=7.5,color="#63768C")
    ax.text(.5,.074,"箭头表示问题、方法或证据的承续；不是所有代码都依次合并。",
        ha="center",fontsize=9,color="#445B70")
    ax.text(.5,.030,"E 编号对应正文与证据索引。日期按实验合同身份，非 Git 提交日。",
        ha="center",fontsize=8,color="#63768C")
    return save(fig,out,"01_lineage",["research_state/registry.json", "docs/EXPERIMENT_PLAN.md"],
        "Author synthesis of registered dependencies and scoped implementation contracts; no numerical pooling")


def draw_baseline(out):
    p=js("phys","reconstruction_summary.json")
    rows={x["dataset"]:x for x in p["tables"]["common_later_windows"] if x["method"]=="H0_fixed"}
    fig,ax=plt.subplots(figsize=(523/72,3.3));fig.subplots_adjust(left=.10,right=.98,bottom=.22,top=.9)
    x=np.arange(3);vals=[rows[d]["pass_rate"]*100 for d in DS]
    errors=np.array([[100*(rows[d]["pass_rate"]-rows[d]["pass_rate_subject_CI"][0]) for d in DS],
                     [100*(rows[d]["pass_rate_subject_CI"][1]-rows[d]["pass_rate"]) for d in DS]])
    ax.bar(x,vals,color=COLORS,width=.5,yerr=errors,capsize=4)
    for i,d in enumerate(DS):
        row=rows[d];ax.text(i, vals[i]-18, f"{vals[i]:.2f}%",ha="center",color="white",fontsize=12)
        ax.text(i,3,f"{row['subjects']} 人\n{row['pooled_passes']}/{row['planned_windows']} 窗",ha="center",fontsize=8,color='white')
    ax.axhline(80,color="#9F592D",linestyle="--",label="预定工程门槛 80%")
    ax.set(xticks=x,xticklabels=[LABEL[d] for d in DS],ylim=(0,102),ylabel="三项 NRMSE 均 < 0.5 的窗口比例 (%)")
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.17),frameon=False); ax.set_title("冻结 H0 已达到限定观测重建标准")
    return save(fig,out,"02_reconstruction",[f"{RUNS['phys']}/reconstruction_summary.json"],
        "common_later_windows / H0_fixed; retained pass_rate and subject bootstrap CI; failures retained")


def draw_model(out):
    fig,ax=plt.subplots(figsize=(523/72,4.5));fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.97)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    box(ax,.02,.76,.29,.20,"原生 EEG\n六通道 × 五频带功率\nlog → 训练 PCA → 相对代理")
    box(ax,.69,.76,.29,.20,"原生／发布 HbO、HbR\n声明单位、参考与时间支持\n共同正尺度保留相对幅值")
    box(ax,.365,.75,.27,.21,"当前窗口的可见信息\n联合估计驱动与初态\n隐藏目标只评分",face="#EEF6F2",edge="#4D8873")
    box(ax,.16,.435,.40,.21,"非线性动力学核\n$r(t),s,f,v,p,q$\n四阶 RK 积分＋物理合法域",size=10)
    box(ax,.67,.445,.30,.19,"候选结构化额外项\n固定／训练方向与有限基底\n来源未确定",face="#FFF2E6",edge="#B68149")
    box(ax,.08,.115,.36,.18,"SSM 观测分量\n与数据相同的观测算子\n模型条件下的代表性解释")
    box(ax,.57,.115,.40,.18,"总预测与完整剩余项\n总预测＝SSM＋结构项\n完整剩余＝结构项＋拟合残差",face="#EEF6F2",edge="#4D8873")
    for a,b in [((.31,.86),(.365,.86)),((.69,.86),(.635,.86)),((.45,.75),(.4,.645)),
                ((.62,.75),(.79,.635)),((.31,.435),(.28,.295)),((.44,.205),(.57,.205)),
                ((.82,.445),(.80,.295))]:arrow(ax,a,b)
    ax.text(.5,.025,"图示为共同接口；参数稳定性、两维驱动、响应扩展和输出审计各有独立配置。",
        ha="center",fontsize=8,color="#63768C")
    return save(fig,out,"03_model",["src/inference/shared_driver_rk4.py","src/inference/ssm_output_audit.py"],
        "Actual shared nonlinear core and conditional observation decomposition; optional branches explicitly separated")


def equations(out, name, lines, height, sources):
    fig=plt.figure(figsize=(523/72,height))
    for i,line in enumerate(lines):
        fig.text(.025,1-(i+.72)/len(lines),line,fontsize=13.2,va="center",color="#19374F")
    return save(fig,out,name,sources,"LaTeX mathtext raster at final display width")


def draw_parameters(out):
    p=js("params");a=pd.DataFrame(p["evaluation_assessment"])
    fig,ax=plt.subplots(figsize=(523/72,3.65));fig.subplots_adjust(left=.095,right=.98,bottom=.22,top=.90)
    x=np.arange(4)
    for i,ds in enumerate(DS):
        rows=a[a.dataset==ds].set_index("model").loc[["P1_tau","P1_beta","P2","P3"]]
        vals=100*rows.best_repeat_median.to_numpy();pos=x+(i-1)*.24
        ax.bar(pos,vals,.23,label=LABEL[ds],color=COLORS[i])
        for j,(v,e,n) in enumerate(zip(vals,rows.effective_pairs,rows.planned_pairs)):
            ax.text(pos[j],v+1,f"{v:.1f}",ha="center",fontsize=7.5)
    ax.axhline(20,ls="--",color="#B86B2C",label="必要门槛 20%")
    ax.set(xticks=x,xticklabels=[r"仅 $\tau$",r"仅 $\beta$",r"$\tau,\beta$",r"$\tau,\beta,\kappa$"],
        ylim=(0,120),ylabel="重复差异的被试中位数 (%)")
    ax.legend(ncols=4,loc="upper center",bbox_to_anchor=(.5,-.14),fontsize=8)
    ax.set_title("冻结候选中的描述性最佳结果：12 组均未同时满足数值与重复性门槛")
    return save(fig,out,"04_parameter_repeat",[f"{RUNS['params']}/summary.json#evaluation_assessment"],
        "best_repeat_median: maximum symmetric relative error over released coordinates, site mean then subject median; no evaluation reselection")


def draw_shrinkage(out):
    p=js("params");rows=pd.DataFrame(p["synthetic"]+p["selected_synthetic"]["estimates"])
    variants=["C0a","H0.25_C0a","H1_C0a","H4_C0a"]
    fig,ax=plt.subplots(figsize=(523/72,3.45));fig.subplots_adjust(left=.11,right=.98,bottom=.2,top=.87)
    for par,col,mark,label in zip(["tau","neurovascular_gain","kappa"],COLORS,["o","s","^"],[r"$\tau$",r"$\beta$",r"$\kappa$"]):
        own=rows[(rows.model=="P3")&(rows.level==1)&(rows.parameter==par)&rows.variant.isin(variants)].set_index("variant").loc[variants]
        ax.plot(range(4),own.log_recovery_slope,marker=mark,color=col,label=label,lw=1.8)
    ax.axhspan(.8,1.2,color="#D9EEDC",alpha=.6,label="预定差异响应范围 0.8–1.2")
    ax.set(xticks=range(4),xticklabels=["不共享",r"$\lambda=0.25$",r"$\lambda=1$",r"$\lambda=4$"],
        ylim=(0,1.23),ylabel="估计 log 参数对真值的响应斜率")
    ax.set_title("共享压缩真实个体差异：小差异合成条件")
    ax.legend(ncols=4,loc="upper center",bbox_to_anchor=(.5,-.13))
    return save(fig,out,"05_shrinkage",[f"{RUNS['params']}/summary.json#synthetic",f"{RUNS['params']}/summary.json#selected_synthetic"],
        "P3, heterogeneity_log_sd=.15; separate eight training subjects, 32 planned scoring identities, 31 available; reported log_recovery_slope")


def draw_robustness(out):
    p=js("robust");m=csv("robust","measured_summary.csv")
    fig,(ax,bx)=plt.subplots(1,2,figsize=(523/72,3.4));fig.subplots_adjust(left=.08,right=.98,bottom=.2,top=.86,wspace=.38)
    arms=["M0","M-observation","M-prior","M-combined"]
    vals=[sum(p["full_pass"][d][a] for d in DS) for a in arms]
    ax.bar(range(4),vals,color=["#97A5B5","#D29A57","#8AADC7","#215F94"])
    for i,v in enumerate(vals):ax.text(i,v+1,f"{v}/72",ha="center",fontsize=8)
    ax.set(xticks=range(4),xticklabels=["原模型","仅额外项","仅先验","组合"],ylim=(0,75),ylabel="完整观测达标窗数")
    ax.set_title("全观测拟合收益")
    for i,ds in enumerate(DS):
        rows=m[(m.dataset==ds)&(m['mode']=="HbR_hidden")&(m.pairing=="real")].set_index("arm")
        vals=[rows.loc[a,"total_nrmse_median"] for a in ["M0","M-combined"]]
        bx.plot([0,1],vals,marker=["o","s","^"][i],label=LABEL[ds],color=COLORS[i],lw=1.6)
        for j,v in enumerate(vals):bx.annotate(f"{v:.3f}",(j,v),xytext=(4,4),textcoords="offset points",fontsize=7)
    bx.set(xticks=[0,1],xticklabels=["原模型","组合"],xlim=(-.12,1.36),ylim=(.4,1.25),ylabel="隐藏 HbR NRMSE 中位数（越低越好）")
    bx.set_title("单 Hb 遮蔽反而退化");bx.legend(loc="lower right",fontsize=7)
    return save(fig,out,"06_robustness",[f"{RUNS['robust']}/summary.json#full_pass",f"{RUNS['robust']}/measured_summary.csv"],
        "Full pass sums dataset counts; HbR_hidden/real per-dataset successful-fit median, all displayed HbR rows converged 24/24")


def draw_tokenizer(out):
    p=js("response","tokenizer/summary.json")
    rows=pd.DataFrame([dict(seed=x["seed"],**y) for x in p["public_probes"] if x["arm"]=="distribution_randomized_generator" for y in x["summaries"]])
    methods=["context_raw","context_plus_O","context_plus_O_plus_S","context_plus_capacity_matched_observation_only"]
    means=rows[rows.prefix_steps==40].groupby("method").NRMSE.mean().loc[methods]
    fig,ax=plt.subplots(figsize=(523/72,3.25));fig.subplots_adjust(left=.1,right=.97,bottom=.26,top=.86)
    ax.bar(range(4),means,color=["#97A5B5","#215F94","#BB7842","#168478"],width=.56)
    for i,v in enumerate(means):ax.text(i,v+.015,f"{v:.4f}",ha="center",fontsize=10)
    for seed,g in rows[rows.prefix_steps==40].groupby("seed"):
        ax.scatter(range(4),g.set_index("method").NRMSE.loc[methods],s=18,facecolors="white",edgecolors="#334A5D",zorder=3)
    ax.set(xticks=range(4),xticklabels=["原始上下文","上下文＋O","上下文＋O＋S","等容量纯 O"],ylim=(0,1.36),ylabel="Hb 后缀 NRMSE（越低越好）")
    ax.set_title("观测表征有工程收益；加入语义 S 未提供额外预测收益")
    return save(fig,out,"07_public_probe",[f"{RUNS['response']}/tokenizer/summary.json#public_probes"],
        "distribution_randomized_generator, prefix_steps40; three optimization seed mean; dots are optimization repeats of identical 15 evaluation subjects")


def draw_alias(out):
    d=csv("response","tokenizer/scenario_variant_seed_summary.csv")
    d=d[(d.arm=="distribution_randomized_generator")&(d.variant=="intervention")]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(523/72,3.5));fig.subplots_adjust(left=.09,right=.98,bottom=.23,top=.86,wspace=.37)
    x=np.arange(2)
    for i,(mod,col) in enumerate(zip(["eeg","hb"],[COLORS[0],COLORS[1]])):
        own=d[d.modality==mod].set_index("scenario")
        val=own.loc[["baseline","non_dct_impulse"],"shape_rmse_seed_mean"].to_numpy()
        ax.bar(x+(i-.5)*.34,val,.33,color=col,label=mod.upper())
        for j,v in enumerate(val):ax.text(j+(i-.5)*.34,v+.025,f"{v:.3f}",ha="center",fontsize=8)
        val=100*own.loc[["baseline","exact_joint_alias"],"log_amplitude_coverage_95_seed_mean"].to_numpy()
        bx.bar(x+(i-.5)*.34,val,.33,color=col,label=mod.upper())
        for j,v in enumerate(val):bx.text(j+(i-.5)*.34,v+2,f"{v:.2f}",ha="center",fontsize=8)
    ax.set(xticks=x,xticklabels=["匹配基准","未见脉冲"],ylim=(0,1.25),ylabel="单位 RMS 形状 RMSE")
    bx.set(xticks=x,xticklabels=["匹配基准","精确联合 alias"],ylim=(0,112),ylabel="幅值 95% 区间实际覆盖率 (%)")
    bx.axhline(95,ls="--",color="#AD6F37",lw=1)
    ax.set_title("形状泛化有明确失败");bx.set_title("同观测可对应不同真值幅值")
    ax.legend(ncols=2,loc="upper left")
    return save(fig,out,"08_alias",[f"{RUNS['response']}/tokenizer/scenario_variant_seed_summary.csv","src/inference/semantic_response_synthetic.py"],
        "Distribution arm, intervention rows; mean over three optimization seeds; 256 shared base identities, not three independent cohorts")


def draw_anchors(out):
    d=csv("obs","event_metrics.csv");h=csv("obs2","hardware_comparisons.csv")
    fig,(ax,bx)=plt.subplots(1,2,figsize=(523/72,3.9));fig.subplots_adjust(left=.09,right=.98,bottom=.24,top=.86,wspace=.58)
    own=d[d.endpoint=="EEG_voltage"].groupby(["dataset","subject","arm"]).nrmse.mean().groupby(["dataset","arm"]).mean()
    x=np.arange(2)
    for i,(arm,col,label) in enumerate([("training_mean","#97A5B5","训练模板"),("EOG_real",COLORS[0],"真实 EOG")]):
        vals=[own.loc[ds,arm] for ds in DS[:2]]
        ax.bar(x+(i-.5)*.32,vals,.31,color=col,label=label)
        for j,v in enumerate(vals):ax.text(j+(i-.5)*.32,v+.025,f"{v:.3f}",ha="center",fontsize=8)
    ax.set(xticks=x,xticklabels=["Single-Trial\n14 评价人","Simultaneous\n13 评价人"],ylim=(0,1.02),ylabel="前额原始电位 NRMSE")
    ax.set_title("独立眼电观测的可检验关联");ax.legend(ncols=2,loc="upper right")
    keys=["shared_vs_disjoint","disjoint_distance_matched_vs_template"]
    rows=h[h.analysis_panel=="distance_caliper"].set_index("contrast").loc[keys]
    y=np.arange(2)
    bx.errorbar(rows["mean"],y,xerr=[rows["mean"]-rows.low,rows.high-rows["mean"]],fmt="o",color=COLORS[1],capsize=4)
    bx.axvline(0,color="#8693A2",lw=1);bx.set(yticks=y,yticklabels=["共享相对\n非共享 donor","非共享相对\n训练模板"],xlim=(-.02,.185),ylim=(1.6,-.6),xlabel="配对 NRMSE 改善（正值有利）")
    bx.set_title("空间可预测性与硬件关联")
    for y0,(_,r) in zip(y,rows.iterrows()):bx.text(r['mean'],y0+.30,f"{r['mean']:.4f} [{r.low:.4f}, {r.high:.4f}]",ha="center",fontsize=7)
    return save(fig,out,"09_anchors",[f"{RUNS['obs']}/event_metrics.csv",f"{RUNS['obs2']}/hardware_comparisons.csv"],
        "Voltage: window mean then subject equal; hardware: original distance_caliper paired contrast, 20 targets/14 subjects/2240 pairs and 2000-block percentile CI")


def draw_continuity(out):
    p=js("output");d=pd.DataFrame(p["measured_comparisons"])
    fig,ax=plt.subplots(figsize=(523/72,3.85));fig.subplots_adjust(left=.21,right=.98,bottom=.18,top=.87)
    tasks=["full","middle_Hb","middle_HbR"]
    y=0
    for i,ds in enumerate(DS):
        for mode,mark in zip(tasks,["x","s","o"]):
            r=d[(d.dataset==ds)&(d["mode"]==mode)&(d.comparison=="continuity")&(d.candidate=="chain_matched")&(d.metric=="score_nrmse")].iloc[0]
            ax.errorbar(r.improvement,y,xerr=[[r.improvement-r.lower],[r.upper-r.improvement]],fmt=mark,color=COLORS[i],capsize=3)
            y+=1
        y+=.6
    ax.axvline(0,color="#8F9BA8",lw=1)
    ys=[a+b*3.6 for b in range(3) for a in range(3)]
    ax.set(yticks=ys,yticklabels=[f"{LABEL[d]} · {t}" for d in DS for t in ["完整拟合","隐藏双 Hb","仅隐藏 HbR"]],ylim=(9.9,-.8),xlim=(-.18,.58),xlabel="free − 匹配先验连续：NRMSE 改善（正值有利）")
    ax.set_title("连续状态牺牲可见拟合，改善部分未见观测")
    return save(fig,out,"10_continuity",[f"{RUNS['output']}/summary.json#measured_comparisons"],
        "chain_matched versus free, score_nrmse; 12 records/6 subjects per row; original subject-block bootstrap CI")


def draw_profile(out):
    nodes=csv("output","profile_nodes.csv");ranges=csv("output","profile_ranges.csv")
    kinds=ranges[["key","objective"]].rename(columns={"objective":"objective_kind"})
    d=nodes.merge(kinds,on="key",validate="many_to_one")
    d=d[(d.arm=="chain_matched")&(d.objective_kind=="total")&np.isclose(d.offset,.5)]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(523/72,3.6));fig.subplots_adjust(left=.1,right=.98,bottom=.24,top=.86,wspace=.38)
    x=np.arange(3)
    for i,(name,color,label) in enumerate([("SSM_Hb",COLORS[0],"SSM Hb"),("T","#BD813E","T"),("X",COLORS[1],"X")]):
        vals=[100*d[(d.dataset==ds)&(d.output==name)].iloc[0].objective_fraction for ds in DS]
        ax.bar(x+(i-1)*.24,vals,.23,label=label,color=color)
        for j,v in enumerate(vals):ax.text(j+(i-1)*.24,v+1.4,f"{v:.2f}",ha="center",fontsize=7)
    ax.axhline(5,ls="--",color="#834C37",lw=1,label="5% 容差");ax.set(xticks=x,xticklabels=["Single","Sim","Visual"],ylim=(0,82),ylabel="固定 +0.5 输出偏移后的目标增加 (%)")
    ax.set_title("SSM/T 位移代价低，X 更受约束",fontsize=9.5)
    ax.legend(ncols=2,loc="upper center",bbox_to_anchor=(.5,-.13),frameon=False)
    d=d[d.output=="SSM_Hb"].set_index("dataset").loc[DS]
    for i,(name,color,label) in enumerate([("SSM_Hb_rms_change",COLORS[0],"SSM"),("component_rms_change","#BD813E","额外项"),("total_prediction_rms_change",COLORS[1],"总预测")]):
        vals=d[name].to_numpy();bx.bar(x+(i-1)*.24,vals,.23,label=label,color=color)
        for j,v in enumerate(vals):bx.text(j+(i-1)*.24,v+.012,f"{v:.3f}",ha="center",fontsize=7)
    bx.set(xticks=x,xticklabels=["Single","Sim","Visual"],ylim=(0,.63),ylabel="中央 Hb 轨迹变化 RMS / 训练 SD")
    bx.set_title("大幅分量交换，合计变化很小")
    bx.legend(ncols=3,loc="upper center",bbox_to_anchor=(.5,-.13),frameon=False,fontsize=7)
    return save(fig,out,"11_profiles",[f"{RUNS['output']}/profile_nodes.csv",f"{RUNS['output']}/profile_ranges.csv"],
        "Join objective category by key (nodes.objective is numeric); chain_matched,total,offset+0.5; one prespecified record per dataset")


def draw_eeg_increment(out):
    d=pd.DataFrame(js("output")["measured_comparisons"])
    d=d[(d.comparison=="EEG_increment")&(d.comparator=="wrong_pair")&(d.metric=="score_nrmse")]
    fig,ax=plt.subplots(figsize=(523/72,3.5));fig.subplots_adjust(left=.25,right=.98,bottom=.2,top=.86)
    y=0;labels=[]
    for i,ds in enumerate(DS):
        for mode,mark,title in [("middle_Hb","s","双 Hb"),("middle_HbR","o","HbR")]:
            r=d[(d.dataset==ds)&(d["mode"]==mode)].iloc[0]
            ax.errorbar(r.improvement,y,xerr=[[r.improvement-r.lower],[r.upper-r.improvement]],fmt=mark,color=COLORS[i],capsize=4)
            labels.append(f"{LABEL[ds]} · 隐藏{title}");y+=1
    ax.axvline(0,color="#8F9BA8",lw=1)
    ax.set(yticks=range(6),yticklabels=labels,ylim=(5.6,-.6),xlim=(-.23,.41),xlabel="错配 EEG − 真实 EEG：NRMSE 改善")
    ax.set_title("真实配对 EEG 的六个增量区间全部跨零")
    return save(fig,out,"12_eeg_increment",[f"{RUNS['output']}/summary.json#measured_comparisons"],
        "EEG_increment/wrong_pair; frozen chain_matched; 6 subjects/12 paired records per endpoint; descriptive 95% intervals")


class Report:
    """Fixed-page prose/tables with selectable CJK text and bitmap figures."""
    def __init__(self,out):
        self.out=out;self.pdf=fitz.open();self.page=None;self.y=0
        font=TTFont(str(FONT),fontNumber=2)
        for cmap in font["cmap"].tables:
            if cmap.isUnicode():
                cmap.cmap={c:g for c,g in cmap.cmap.items() if not(0xF900<=c<=0xFAFF or 0x2F800<=c<=0x2FA1F)}
        buf=BytesIO();font.save(buf);self.font=fitz.Font(fontbuffer=buf.getvalue())
        self.sections=[];self.images=[];self.titles=[]

    def wrap(self,text,width,size):
        lines=[]
        for para in str(text).split("\n"):
            line=""
            for ch in para:
                if line and self.font.text_length(line+ch,fontsize=size)>width:
                    lines.append(line);line=ch
                else:line+=ch
            lines.append(line)
        return lines

    def new(self,title,subtitle=None):
        self.page=self.pdf.new_page(width=595.28,height=841.89);self.y=34
        self.titles.append(title);self.sections.append([f'<section id="p{len(self.pdf)}"><h1>{html.escape(title)}</h1>'])
        self.text(title,size=16,gap=12,color=(.08,.20,.32),html_add=False)
        if subtitle:self.text(subtitle,size=8.6,color=(.36,.43,.5),gap=12)
        self.page.draw_line((36,self.y),(559,self.y),color=(.75,.81,.86),width=.5);self.y+=11

    def ensure(self,height):
        if self.y+height>790:
            raise ValueError(f"page {len(self.pdf)} overflow at {self.y:.1f} + {height:.1f}: {self.titles[-1]}")

    def text(self,text,size=10,gap=9,color=(.13,.17,.22),html_add=True):
        lines=self.wrap(text,523,size);height=len(lines)*size*1.5
        self.ensure(height)
        writer=fitz.TextWriter(self.page.rect)
        for line in lines:
            if line:writer.append((36,self.y+size),line,font=self.font,fontsize=size)
            self.y+=size*1.5
        writer.write_text(self.page,color=color)
        self.y+=gap
        if html_add:self.sections[-1].append('<p>'+html.escape(text).replace('\n','<br>')+'</p>')

    def sub(self,text):
        self.text(text,size=11.7,gap=6,color=(.08,.30,.43),html_add=False)
        self.sections[-1].append('<h2>'+html.escape(text)+'</h2>')

    def table(self,headers,rows,widths,size=8.6):
        assert abs(sum(widths)-523)<.1
        output=['<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>']
        for i,row in enumerate([headers,*rows]):
            wrapped=[self.wrap(v,w-10,size) for v,w in zip(row,widths)]
            height=max(map(len,wrapped))*size*1.42+10;self.ensure(height)
            self.page.draw_rect(fitz.Rect(36,self.y,559,self.y+height),color=None,
                fill=(.90,.94,.97) if i==0 else ((.96,.975,.985) if i%2 else (1,1,1)))
            writer=fitz.TextWriter(self.page.rect);x=36
            for lines,w in zip(wrapped,widths):
                for j,line in enumerate(lines):writer.append((x+5,self.y+5+size+j*size*1.42),line,font=self.font,fontsize=size)
                x+=w
            writer.write_text(self.page,color=(.13,.17,.22))
            self.y+=height
            if i:output.append('<tr>'+''.join('<td>'+html.escape(str(x)).replace('\n','<br>')+'</td>' for x in row)+'</tr>')
        self.y+=10;output.append('</tbody></table>');self.sections[-1].append(''.join(output))

    def fig(self,name,caption):
        path=self.out/'figures'/(name+'.png');w,h=Image.open(path).size
        height=523*h/w;self.ensure(height)
        self.page.insert_image(fitz.Rect(36,self.y,559,self.y+height),filename=str(path))
        self.images.append(dict(page=len(self.pdf),name=name,width_pt=523,height_pt=height,dpi=w/(523/72)))
        self.y+=height+5
        self.text(caption,size=8.3,gap=10,color=(.34,.4,.47),html_add=False)
        self.sections[-1].append(f'<figure><img src="figures/{name}.png" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')

    def ref(self,label,path,detail):
        p=ROOT/path
        if not p.exists():raise FileNotFoundError(p)
        INPUTS.add(path)
        self.text(label+'：'+detail,size=9,gap=2)
        start=self.y
        self.text(path,size=7.1,gap=9,color=(.28,.40,.53),html_add=False)
        relative='../../../'+path
        self.page.insert_link({'kind':fitz.LINK_LAUNCH,'from':fitz.Rect(36,start,559,self.y-6),'file':relative})
        self.sections[-1].append(f'<p class="source"><a href="{html.escape(relative)}">{html.escape(path)}</a></p>')

    def finish(self):
        for i,page in enumerate(self.pdf):
            writer=fitz.TextWriter(page.rect)
            writer.append((36,816),'SSM 实验谱系与能力边界 | 2026-10-10 证据快照',font=self.font,fontsize=7)
            writer.append((514,816),f'{i+1} / {len(self.pdf)}',font=self.font,fontsize=7)
            writer.write_text(page,color=(.42,.48,.54))
        self.pdf.set_toc([[1,t,i+1] for i,t in enumerate(self.titles)])
        self.pdf.set_metadata(dict(title='预处理修复后的 SSM 实验谱系、参数恢复与信号语义',author='PID-MCM project',
            subject='2026-09-28 至 2026-10-10；保留证据综合；不构成新实验或当前状态源'))
        self.pdf.subset_fonts(fallback=True)
        output=self.out/'SSM_RESEARCH_LINEAGE.pdf';self.pdf.save(output,garbage=4,deflate=True);self.pdf.close()
        css='''body{font-family:"Noto Sans CJK SC",sans-serif;color:#223448;background:#eaf0f5;margin:0}section{max-width:900px;margin:25px auto;padding:38px 48px;background:white;box-shadow:0 2px 10px #ccd4dd}h1{font-size:26px;color:#163c5d}h2{font-size:19px;color:#215c78}p{line-height:1.8;font-size:16px}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.6}th,td{border-bottom:1px solid #dce4ea;text-align:left;padding:9px;vertical-align:top}th{background:#e6eff6}figure{margin:18px 0}img{width:100%;height:auto}figcaption,.source{font-size:13px;color:#5b6d7d;overflow-wrap:anywhere}a{color:#166498}nav{max-width:900px;margin:25px auto;padding:24px 48px;background:white}nav li{margin:7px 0}@media print{body{background:white}section{box-shadow:none;break-before:page;margin:0;padding:0}nav{display:none}}'''
        toc='<nav><strong>预处理修复后的 SSM 实验谱系、参数恢复与信号语义</strong><ol>'+''.join(f'<li><a href="#p{i+1}">{html.escape(t)}</a></li>' for i,t in enumerate(self.titles))+'</ol></nav>'
        (self.out/'SSM_RESEARCH_LINEAGE.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SSM 实验谱系技术报告</title><style>'+css+'</style>'+toc+''.join('\n'.join(s)+'</section>' for s in self.sections)+'</html>')
        preview=self.out/'preview';preview.mkdir(exist_ok=True)
        pages=[]
        with fitz.open(output) as doc:
            for i,page in enumerate(doc):
                chars=len(page.get_text());images=page.get_images(full=True)
                assert chars>100,(i,chars)
                # TextWriter.text_rect uses the font's conservative glyph box.
                # Check actual emitted text spans instead, including separate
                # limits for body and footer, against the delivered PDF.
                spans=[s for b in page.get_text('dict')['blocks'] if b['type']==0
                       for line in b['lines'] for s in line['spans']]
                for span in spans:
                    x0,y0,x1,y1=span['bbox']
                    assert 35<=x0<=x1<=560 and 25<=y0<=y1<=825,(i+1,span)
                    assert y1<799 or y0>802,(i+1,'body/footer overlap',span)
                page.get_pixmap(dpi=110,alpha=False).save(preview/f'page_{i+1:02}.png')
                pages.append(dict(page=i+1,searchable_characters=chars,image_objects=len(images)))
            assert sum(x['image_objects'] for x in pages)==len(self.images)
            text_all=''.join(p.get_text() for p in doc)
            assert '生理' in text_all and '完整剩余项' in text_all
        assert all(235<=x['dpi']<=245 for x in self.images)
        for start in range(0,len(pages),6):
            sheet=Image.new('RGB',(900,1260),'#DBE4EC')
            for i in range(start,min(start+6,len(pages))):
                im=Image.open(preview/f'page_{i+1:02}.png');im.thumbnail((438,405))
                x=((i-start)%2)*450+(450-im.width)//2;y=((i-start)//2)*420+8
                sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+6,y+im.height+1),str(i+1),fill='#203548')
            sheet.save(preview/f'contact_{start+1:02}_{min(start+6,len(pages)):02}.png')
        validation=dict(report=output.name,pages=pages,figures=self.images,
            body_and_tables_searchable=True,all_figures_bitmap=True,figure_dpi=240,
            text_bounds_checked=True,rendered_every_page=True,wps_checked=False,
            no_model_fits=True,no_raw_data_read=True,no_protected_data_read=True)
        (self.out/'export_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
        (self.out/'report_provenance.json').write_text(json.dumps(dict(
            scope='dated communication synthesis, not an experimental result or current state owner',
            date='2026-10-10',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            state_owner='research_state/registry.json',source_files=sorted(INPUTS),figures=FIGURES,
            figure_run_path_root='experiments/runs/physiology_semantic_tokenizer',
            renderer='experiments/scripts/render_ssm_research_lineage_report.py'),ensure_ascii=False,indent=2)+'\n')
        return validation


def build_results(r,params,support,power,rate):
    r.new('参数恢复：单参数仍不够稳定，联合释放更困难', 'E12｜同任务、同条件起点计数、同 ROI／通道的两条 90 s 记录；分别估计与参数转移')
    r.fig('04_parameter_repeat','图 5｜每个数据集×参数层次在冻结候选中的描述性最佳结果，优先按有效配对率排名。每根柱是区域先在被试内平均后的重复差异中位数；不是平均参数误差，也不是评价后重选模型。')
    r.table(['数据与配对性质','评价身份','单 τ 有效配对','联合参数有效配对'],[
        ['Single：独立原生记录','23 人','8/69 区域配对','τ/β：0/69；三参数：0/69'],
        ['Sim：同记录不重叠块','20 人','12/60 区域配对','τ/β：0/60；三参数：0/60'],
        ['Visual：独立原生记录','7 人（另 3 人为同记录辅助）','7/35 区域配对','τ/β：0/35；三参数：0/35'],
    ],[159,93,102,169],size=8.5)
    r.text('“有效配对”要求两记录数值通过、全部释放参数离 log 搜索边界至少 5%、最差参数重复差异≤20%，并且逐记录／模态的原始拟合及参数转移误差均不比 N0 差超过 0.01。固定值与完全共享值即使差异为零，也不能获得有效个体参数资格。')
    r.text('E1/E2 做完整补偿量剖面与数值优化核验；E3 比较分段／连续状态；E4 用开发时间交叉拟合残差建立工作协方差；E5 由开发记录选择共享中心与强度；E6 才评价冻结候选。该流程把“更稳定”与“牺牲观测保真”分开。[E12]')
    r.text('旧 E4 的 24%–32% 与本轮 42%–52% 使用不同记录支持、配对规则和汇总对象，不能解读成时间上的性能倒退。两轮共同指向：目前实测尚不足以支持稳定个体生理常数。',size=9.4)

    r.new('参数剖面与共享：稳定不能靠压平个体差异获得', 'E12｜有完整补偿的共同内点搜索；合成检验保留真实差异与失败分母')
    r.fig('05_shrinkage','图 6｜小差异生成条件下的 log 参数恢复斜率。32 个计划生成身份中 31 个可用，另用 8 个训练身份拟合共享对象；斜率接近 1 才保留真实差异，接近 0 表示差异被压缩。')
    r.text('共享在实测中能降低某些重复误差，但合成显示，λ=1 时 τ/β/κ 的响应斜率仅约 0.13/0.29/0.11，远低于不共享的 0.94/0.98/0.95。固定或强共享可制造稳定，却不等于恢复更准确。大差异层的共享中心另有数值失败，不能用成功子集掩盖。')
    d=pd.DataFrame(params['profile_intersections'])
    rows=[]
    for model,label in [('P1_tau','τ'),('P1_beta','β'),('P2','τ、β'),('P3','τ、β、κ')]:
        own=d[(d.model==model)&np.isclose(d.inner_fraction,.05)&np.isclose(d.tolerance,.05)]
        counts=[int(own[own.objective==obj]['exists'].sum()) for obj in ['observation_only_legal_domain','current_without_parameter_shrinkage']]
        rows.append([label,f'{counts[0]}/18',f'{counts[1]}/18'])
    r.table(['释放参数','仅数据：找到共同内点','工程目标：找到共同内点'],rows,[111,206,206],size=8.6)
    r.text('上表固定 5% 容差与 5% log 内点条件，18 个预定开发组逐组搜索；每个节点重拟合驱动与初态，容差分母为冻结 P0 目标。联合参数没有找到共同内点，是有限搜索的负证据，不能证明共同内点数学上不存在。')
    r.text('Visual 前额 montage 0 的单 τ 中位差异 13.6% 值得独立复验，但其 P90=93.8%、仅 3/7 有效、2/7 近边界；不能用局部中位数宣布稳定恢复。N2 新求解器及连续长度扩展均未达到原定触发条件，没有追加执行。',size=9.2)

    r.new('额外项减少了失配，却也会吸收真实动力学', 'E5→E6｜比较可见重建、隐藏观测和已知真实变化，避免只挑最漂亮的一条曲线')
    r.fig('06_robustness','图 7｜左：完整重建达标数；右：同一公开开发面板的整条 HbR 隐藏误差中位数，各显示组均 24/24 收敛。额外项＋先验的拟合收益不能外推成缺失恢复或生理真值收益。')
    r.text('四臂实验保留六状态与 H0，只改变同向四余弦观测项、驱动幅度先验或两者。隐藏 EEG 的成功解误差从 8.43/20.37/12.57 降到约 1.10，解决了严重发散；但仍弱于训练模板，整模态的 12 个错配／时移对照无一形成全正增量区间。[E5]')
    r.text('匹配合成中，组合臂完整驱动 NRMSE 从 0.101 变为 0.241；参考后的形状误差只从 0.050 变为 0.055。先验改变了偏移／幅值等弱约束方向，因此“更稳定”和“更真实”分离得很清楚。[E5]')
    r.text('归属实验进一步改变共同方向并使用独立区域预测。训练方向相对固定方向，仅 2/6 个单 Hb 端点区间全正；空间受限共同项仅 Single 小幅获益，直接空间 ridge 在 Single/Sim 更有用。[E6]')
    r.text('已知真值合成中，基底内共同干扰的驱动误差可由 0.304 降至 0.132；但真实 τ 变化在 D-fixed 中约有 20.0% 被分配给额外项，训练方向为 31.6%，交换方向为 51.3%。这些是沿真实变化的有符号投影，不是非负能量比例，也不能据此命名具体来源。[E6]')

    r.new('区域谱模式与响应动态：容量收益需要衰减对照', 'E7→E8｜两维 EEG 表示的改善，没有自动成为共享生理驱动的证据')
    r.text('E7 将每个区域的六通道五频带 EEG 表示为宽频同向模式 a 与谱对比模式 b，血管输入为归一化的 a＋γ_spec b；γ_spec 只在训练侧选择。比较一维、两维不耦合 b、两维选择混合及独立 Hb 驱动，时间基底相同。三个区域属于同一记录，统计按被试汇总。')
    r.table(['共同成功配对：R1→R2','Single','Sim','Visual'],[
        ['完整 EEG NRMSE','1.005→0.968','0.953→0.932','0.910→0.887'],
        ['隐藏 Hb NRMSE','1.031→0.984','0.933→0.883','2.023→1.952'],
        ['三类配对 null 同时稳定通过的区域','0/3','0/3','0/3'],
    ],[190,111,111,111],size=8.8)
    r.text('两维模式有有限表示收益，但隐藏 Hb 仍弱于模板和独立驱动参照；γ_spec 跨折换号。9288 个计划单元中 9111 收敛，135 个预算失败、42 个缺匹配供体；另有同一窗口四份收敛解 q 逼近零。数值收敛不等于生理合理。[E7]')
    r.sub('E8 改变推断制度，专门区分谱信息和削弱错误输入')
    r.text('E8 先只用 EEG 估计两条模式，再以可见 Hb 前缀校准初态／共同项，隐藏 Hb 不反向重估驱动。加入与谱混合相同衰减的对照、独立增益、零驱动及随机谱方向；再竞争输入低通和出流松弛。')
    r.table(['40 点前缀：候选相对对照','Hb NRMSE 改善','95% 描述区间'],[
        ['谱混合 − 匹配纯衰减','+0.00354','[−0.00326, +0.01072]'],
        ['选择的稳定动力学 − 独立增益','+0.01371','[+0.00280, +0.02472]'],
        ['谱混合 − 独立增益','−0.05669','[−0.08081, −0.03236]'],
    ],[265,110,148],size=8.6)
    r.text('每个对比有 360 个配对区域窗口、15 名评价被试（6/5/4 人）；被试内平均后按数据集等权。稳定动力学相对独立增益的约 84% 净改善来自三个零 EEG 增益组；九项主要配对 null 均未通过 Holm。收益可以来自更弱的错误耦合，不能宣称发现了真实响应机制。[E8]')

    r.new('连续语义学习：观测表征有收益，S 尚无独立价值', 'E6 原型→E8 四臂三种子｜语义监督目标来自合成生成器；公开探针冻结编码器')
    r.fig('07_public_probe','图 8｜40 点可见 Hb 前缀（4 Hz 下 10 s），评分最后 40 点（20–30 s）。柱为三次优化种子的均值，白点为各种子；评价被试相同，不能把三个点看作三个人群。所有方法使用相同原始上下文。')
    r.text('E6 的有限 typed-token 原型分别学习 EEG 驱动片段与 Hb 结构项投影，已见监督目标改善，但 EEG 增益、τ 和交换样干预会造成错误语义。E8 因此转向独立模态编码器、配对反事实与随机化生成器，同时要求 observation 分支重建完整本模态，而非仅拟合残差。[E6、E8]')
    r.text('O 为完整观测表征，S 为受合成语义目标约束的表征。公开线性探针的共同上下文包括局部 EEG、其他区域完整 Hb 与目标 Hb 可见前缀；目标 Hb 编码器不读待评分后缀。编码器在合成数据上训练后冻结，线性读出仅在训练／选择身份上建立。')
    r.text('分布原型的 O→O+S 使 NRMSE 从 1.1517 变为 1.1554，差为 +0.0037（误差变大）；等容量纯 O 为 1.1393，比 O+S 低 0.0161。三个优化种子方向一致。观测表征的压缩／重建收益成立，但“语义具有额外预测信息”的主张没有获得支持。[E8]')
    r.text('这不是对所有表征学习的否定：它具体限制当前生成目标、独立模态推理和公开探针制度下的 S。不能因为合成语义监督损失下降，就把它作为实测生理 teacher 训练下一代 tokenizer。',size=9.4)

    r.new('语义原型的两个上限：未见形状与精确幅值歧义', 'E8｜条件生成成功，不等于跨机制泛化或无条件幅值识别')
    r.fig('08_alias','图 9｜分布原型，256 个独立基础生成身份，三次优化种子均值。左为单位 RMS 驱动形状 RMSE；右为 log 幅值区间实际覆盖。点估计与区间均只接收本模态输入，真值只评分。')
    r.text('匹配生成条件下，EEG/Hb 形状 RMSE 为 0.043/0.097；改为未见的非 DCT 脉冲，升至 1.017/1.082。说明模型确实学到了一定条件形状规律，但对训练时间基底以外的机制没有可靠泛化。[E8]')
    r.text('精确联合 alias 先改变驱动幅值，再保持 EEG 观测相同，并用专门构造的 Hb 观测补偿项保持 Hb 相同。两个真实幅值不同的对象具有完全一致的输入；任何只依赖这些输入的推断器必须返回同一预测。这个反例是信息限制，不是换一个回归器就能消除。')
    r.text('基准的名义 95% 幅值区间覆盖约为 EEG 97.79%、Hb 98.44%；alias 干预后仅 9.64%、3.65%。区间学习的是生成先验内的不确定性，未包含未识别幅值的完整集合。可保留条件形状目标，绝对幅值需要独立标定或显式说明先验依赖。[E8]')
    r.text('该 alias 允许人为精确 Hb 补偿，是构造的歧义检验，不是普通随机传感器噪声，也不证明现实中所有窗口都有完全相同的两套生理真值。它反驳的是“仅凭当前输入可以无条件识别该广泛机制族的幅值”。',size=9.6)

    r.new('有限响应和连续性：从补偿方向定位可改进的结构', 'E9｜有限候选、独立空间约束与真正连续状态分别测试，避免把更多自由度当作解释')
    r.text('S1 在训练内选择九个 τ/κ 响应候选，冻结后评价隐藏 Hb；S2 用其他区域当前窗口残差训练独立空间读出，目标完全隐藏；S4 分别考察驱动、初态、响应切向与额外项的重叠，并固定额外项变化后重新拟合。')
    r.table(['阶段','主要发现','结论边界'],[
        ['S1 有限响应','完整 Hb 隐藏误差相对改善约 10.0%／3.5%／0.1%；真实响应变化向额外项的平均绝对投影 0.0549→0.0198','有局部工程改善；其他遮挡退化使三数据集均未达到扩展条件'],
        ['S2 空间与联合','部分真实供体有预测价值；前两数据集仍弱于直接原始 Hb ridge','供体污染、物理分量误差及完整端点条件未同时通过'],
        ['S4 补偿定位','初态／额外项的数据切向余弦中位约 0.988；近等价解中初态贡献突出','是局部方向重叠，不是解释方差；因此进一步测试初态接续'],
        ['S3 独立连续合成','相同 90 s 观测、8 个独立生成身份；纯神经中段隐藏 Hb 误差降约 70%，共同干扰降约 13%','非静息共同干扰区间跨零；受限 profile 改善不等于全空间唯一性'],
    ],[85,253,185],size=9)
    r.text('S3 不是把旧独立窗口拼成一条连续真值，而是单独生成连续轨迹、在同一观测下比较可重置与连续传播。它提供了进入原生连续实测的动机；没有在该阶段完成连续实测确认。[E9]')
    r.text('原第一阶段 decision 的 S3 字段记载当时尚未启动；后来的 20261003_s3_v1 有独立 summary。两者按各自时间和范围解释，不能因旧字段仍在而遗漏实际完成的 S3，也不能把 S3 的合成结果补写成原第一阶段的实测结论。')
    r.text('v3 还修正了原始 donor ridge 错误继承 SSM 数值失败的问题：1096 个目标／尺度逐位一致，其他拟合复用旧数组。修正后仍保留负结果，不通过删除强基线为物理分解争取优势。[E9]')

    r.new('独立观测语义：眼电电位关联成立，Hb 来源未定', 'E10｜原生事件关联、源探硬件对照与已知机制竞争；每项信息来源明确')
    r.fig('09_anchors','图 10｜左：被试等权的原生前额电位预测；右：20 个距离匹配目标、14 人、2240 个配对目标窗口，按被试块 bootstrap 的 95% 区间。共享 donor 的优势不自动等于硬件伪迹。')
    r.text('EOG 读出从“预测 SSM 共同项”转为“预测独立观测的前额原始电位”。奇数被试训练、偶数被试评价，真实 EOG 对模板、错被试和时移均有电位收益；Single/Sim 的 NRMSE 从 0.767/0.818 降为 0.333/0.336。这里可赋予“EOG 相关电位”功能语义。[E10]')
    r.text('但对 HbT、X 和 OD 事件目标没有一致正增量。平方功率含交叉项，去除电位后的 log-power 改变不能直接叫作“眼电功率”。眼电关联本身也不能证明潜驱动就是神经活动，或剩余 Hb 就是头皮血流。')
    r.text('共享源／探测器 donor 优于几何距离匹配的非共享 donor 0.0372 NRMSE［0.0250, 0.0499］；非共享 donor 仍优于模板 0.1339。这证实硬件关系之外也有空间可预测性，但组织采样重叠未排除，不能分离硬件、浅表组织与共同生理源。[E10]')
    r.text('竞争机制的正确锚点合成采用已知驱动／初态 oracle；慢噪声补充的无／正确／错锚点错误率为 0.78%／0.26%／18.82%，六对分类增量均未通过 Holm。错误锚点可明显误导；独立短间距和外周生理的实测 E 阶段因数据缺失未执行。',size=9.4)
    r.text('A 阶段还审查了 72 窗的 T/X 敏感性；旧固定方向的六窗 profile 仅两窗出现非零接纳位移。单解、有限探测或代数零坐标都不能当作来源稳定证明，这促成 E13 的逐输出重新优化。',size=9.2)

    r.new('连续实测带来进展：改善隐藏 HbR，减少切点任意性', 'E13｜同一原生 90 s、同一额外项容量与驱动正则；主对照匹配 0/30/60 s 状态先验')
    r.fig('10_continuity','图 11｜每行 12 条配对记录、6 名被试；线段为 2000 次被试块 bootstrap 的 95% 描述区间。全观测亦仅评分中央 30 s；隐藏双 Hb 与仅隐藏 HbR 的目标不同。')
    r.text('free 允许三组五维血管初态，chain_matched 只保留真正起点的五维初态，后续状态由传播得到；两者在相同三个时点施加同一物理状态先验。共用 360 个驱动值与 12 个额外项系数，未知量由 387 减至 377。这隔离了状态接续与减少先验次数的作用。')
    r.text('中央 HbR 隐藏时，Single/Sim/Visual 的误差分别为 0.594→0.467、1.445→1.267、1.458→1.100，改善 21.3%／12.3%／24.5%。全观测误差反而变大，表明收益来自对未见部分的约束，不是全面增加拟合能力。')
    r.text('移动血管切点 ±5 s，同时固定额外项和驱动正则边界后，T 的变化由 0.291/0.600/0.419 降至 0.008/0.011/0.013 个自身训练 SD。连续性显著压制“换个切点就换种解释”。但缩短 Hb 未来上下文，连续 T 仍改变 0.309/0.881/0.414 SD；切点稳定不等于上下文无关。[E13]')
    r.text('仅首端先验的 chain_first 是另一个候选；它在 Sim 的隐藏 HbR 区间仍跨零，不能借用 chain_matched 的全部正区间。E12 连续臂未达参数门槛与此不矛盾：终点、响应参数和 Hb 观测算子均不同。',size=9.2)

    r.new('输出剖面直接证明：连续性没有消除 SSM/T 交换', 'E13｜每个输出寻找自身不利方向，固定输出投影后重拟合全部驱动、初态与额外项')
    r.fig('11_profiles','图 12｜每数据集一条预定代表记录，chain_matched 的总工程目标剖面，偏移 +0.5。左图 SSM Hb、T、X 各用自身训练尺度，不能互相当成相同物理幅值；右图三列统一按 Hb 通道训练 SD。')
    r.text('SSM Hb 移动约 0.5 SD 时，额外项移动约 0.462–0.463 SD，总预测只动 0.037–0.038 SD；三条记录目标仅增加 4.321%、1.728%、0.665%，全部在 5% 工程容差内。这是当前模型和实际记录下“分解不唯一”的可行替代解见证。[E13]')
    r.text('同级 T 位移代价为 1.570%、1.458%、0.240%；X 的 +0.5 位移代价为 70.590%、15.075%、12.750%，相对更受约束。但 Sim/Visual 的 X 仍可接受 ±0.25；Single 在所测 ±0.25 网格未找到可接受点。不能将有限网格空白画成零不确定性。')
    r.text('126 个剖面参考解全部收敛；504 个节点中 456 个收敛、408 个在容差内。48 个失败全部来自连续臂的 p−v，包含局部固定约束、预算终止与回溯停滞。把这些失败计成“输出已精确确定”会颠倒证据含义。')
    r.text('T/X 是两条 Hb 观测的可逆坐标变换。D-fixed 的 (0.65,0.35) 方向会在 X 中代数消掉，所以 X 较稳定部分来自结构；它仍不是独立测得的代谢量或氧饱和度。现阶段可优先验证 X 的可恢复功能，不能先赋予唯一来源名称。',size=9.5)

    r.new('真实 EEG 的独立增量未建立，弱动态也未被排除', 'E13｜真实 EEG、同被试错记录 EEG、仅 Hb 上下文；再用背景匹配注入校准解释检出能力')
    r.fig('12_eeg_increment','图 13｜同一连续模型、同一 Hb 观测，仅替换 EEG。六个区间均跨零；错记录保留同任务、条件起点计数与通道，但不保证事件顺序／相位相同，属于开发性特异性对照。')
    r.text('真实 EEG 在 Visual 两个隐藏任务均劣于 Hb-only；Sim 的双 Hb 隐藏任务可优于 Hb-only，却未明确优于错配 EEG。因此“加入 EEG 有时好于完全不用”尚不能证明使用了配对特异的共享生理信息。')
    r.text('Sim 的 HbR 隐藏＋Hb-only 有 1 个数值失败，主统计按冻结惩罚 100 记入，因此对照均值达 9.500。其余 11 个成功配对中，Hb-only 与真实 EEG 为 1.208 与 1.228。主分母和成功子集同时保留，不能把失败惩罚造成的巨大差值解释为生理收益。[E13]')
    r.sub('为什么负结果不能写成“数据不存在共享动态”')
    r.text('初版 AR(1) 校准低估部分 HbR 波动并未保持 HbO/HbR 相关结构；背景 v2 因此改用开发记录完整中心化轨迹重组，HbO/HbR 共用权重，EEG 用独立权重。它保持训练均值、协方差和多滞后／谱结构的期望，仍不等于纯仪器噪声。')
    r.text('v2 的 1 SD 已知 SSM 注入在三数据集都优于错配 EEG 与 Hb-only；16 个生成身份及预定最初 6 个身份的匹配分析方向一致。0.25／0.5 SD 的检出依赖数据集和对照。该条件校准支持“强匹配动态可检出”，没有给出实测耦合幅值的严格上限。[E13]')

    r.new('采样率支线与分解合同：怎样避免再次追逐拟合数字', 'E11 与 E13｜工程分辨率、隐藏信息和完整语义输出分别处理')
    r.sub('4→10 Hz 没有提供可推广的生理收益')
    rows=[]
    for ds in DS:
        a=next(x for x in rate['aggregates'] if x['dataset']==ds and x['mode']=='full' and x['arm']=='A4')
        b=next(x for x in rate['aggregates'] if x['dataset']==ds and x['mode']=='full' and x['arm']=='C10')
        rows.append([LABEL[ds],f"{a['common_nrmse'][1]:.5f} → {b['common_nrmse'][1]:.5f}",f"{a['common_nrmse'][2]:.5f} → {b['common_nrmse'][2]:.5f}"])
    r.table(['数据集','完整 HbO NRMSE：4→10 Hz','完整 HbR NRMSE：4→10 Hz'],rows,[101,211,211],size=8.8)
    r.text('18 人、72 窗、432/432 实测拟合收敛，旧 4 Hz 特征与 144 条 H0 基线逐值复现。共同 Hb 目标误差的绝对变化均 <0.002，隐藏 Hb 没有一致改善；完整拟合单窗耗时中位约 1.07→2.89 s。仅加密驱动网格可以改善可见拟合，但那是自由度增加。[E11]')
    r.text('10 Hz 同时把功率估计窗从 250 ms 改成 100 ms，因此自身 EEG 目标已改变；旧 4 Hz 兼容误差变大不能简单认定新信号更差，也不能把更低的自身误差算作生理进步。当前不支持默认升级；若再研究，应先分离估计窗、采样间隔与驱动容量。')
    r.fig('15_decomposition','图 14｜可交付分解的闭合关系。H_O/H_R 表示经过共同线性观测算子和共同尺度的 Hb 坐标；在观测、SSM 或额外项上使用同一线性定义时，需明确对象。')
    r.text('完整剩余项 e_remaining 是原可见观测减去 SSM；它包含结构化额外项与未拟合误差。由于各分量通常不正交，不能将它们的平方和直接宣称为“神经／非神经能量占比”。信息支持、参数策略、代表解及替代解范围应与输出一起保存。[E13]')


def build_conclusions(r):
    r.new('哪些上界已被确认：数学限制与实际反例分开判断', '不能突破的条件必须写在结论中；“未检出”不等于“不存在”')
    r.fig('16_limits','图 15｜由已实现动力学得到的 p−v 衰减关系与代理增益规范变换。最后一行是当 β 和 EEG loading 都允许重标定时的尺度歧义；当前冻结 loading 的单一配置不因此自动失去条件可估性。')
    r.table(['边界类型','现有证据能排除的主张','结论适用条件'],[
        ['信息压缩的硬限制','由 log-power／低维 PCA 反推出完整原始电压相位、所有通道源活动','丢失的信息不在输入中；必须保留更丰富观测或缩小输出定义'],
        ['绝对尺度的硬限制','没有独立代理标定，也能把有效 β／驱动幅值当作绝对个体生理量','固定规范可估有效参数；绝对解释仍依赖未测增益与生成假设'],
        ['动力学结构限制','把 p−v 当成独立持续的额外生理过程','同一核下它只按初值指数型衰减；重置可人为再造该自由度'],
        ['精确 alias 反例','同一观测能无条件确定广泛机制族中的唯一真实幅值','E8 构造允许精确 Hb 补偿；只凭相同输入无法区分这些真值'],
        ['当前模型的近等价反例','连续化后，代表性的单条 SSM/T 曲线就是唯一分解','E13 三条预定记录存在 ≤5% 代价、≥0.5 SD 分量变化的合法解'],
    ],[87,242,194],size=8.7)
    r.text('T/X 的命名和线性坐标变换不会增加观测秩，额外项与 SSM 在 T 方向相同也不会产生排他来源。可用动力学、外部观测或干预进一步约束，但不能从“两个坐标各有名字”推出“两个来源各已确定”。')
    r.text('“无法推进”的准确含义是：在信息边界与模型类都不变时，继续降低同一重建目标不足以突破以上主张。它不排除通过独立标定、新的来源观测、不同可识别功能目标或更合适模型继续前进。',size=9.4)

    r.new('当前能力盘点：有进展，但尚未达到两项核心科学目标', '科学判断分别针对工程恢复、个体参数、输出功能与生理来源')
    r.table(['能力','已确认的进展','当前上限／未确认部分'],[
        ['观测重建与数值','错误处理已移除；匹配合成可恢复；非线性合法域与重积分核验较完整','模型条件下可用，不等于独立预测或个体生理真实'],
        ['稳定生理参数','建立了严格配对、共同内点、信息补偿、转移和共享真值校准；局部单 τ 候选','总体未达标；联合参数零有效配对；更好的参数直方图不是进步证据'],
        ['连续状态功能','隐藏 HbR 改善；人为切点引起的 T/X 波动显著减少','上下文仍改变解释；驱动、流量、容积与 SSM/T 均存在弱约束方向'],
        ['SSM 无法解释的部分','完整剩余项与结构项已区分；观测坐标、支持、有限替代解可以明确交付','结构项不能代表全部剩余；不能把剩余直接叫噪声或非神经源'],
        ['数据约束的语义','EOG 相关电位与硬件／空间关系有实测支持；X 相对 T 更受约束','EOG 不提供 Hb 来源；空间相关不区分硬件与组织；X 不是代谢真值'],
        ['跨模态共享信息','错配／时移／其他区域与 Hb-only 对照更充分，强匹配注入可被检出','旧 null 检验与最新六个区间未建立稳定配对特异增量；弱动态未排除'],
        ['语义学生／tokenizer','独立连续 O 接口、合成条件形状和公开探针工程收益','S 无额外公开收益；OOD 形状与 alias 幅值覆盖失败；无生理晋级'],
        ['更细 EEG 代理','4/10 Hz 与仅加密网格已分离，原 4 Hz 可逐值重放','本面板未见一致隐藏收益；不能据此证明所有高采样率无用'],
    ],[93,215,215],size=8.8)
    r.text('对用户提出的两个目标，答案应分开：稳定个体生理参数恢复尚没有普遍突破；为剩余信号赋予语义已有计算合同和局部观测关联的进展，但还没有数据驱动的唯一生理来源识别。更明确的负结果和边界减少了可行假设，也属于实质科学进展。')
    r.text('这里没有把多个开发面板的“没有通过”累加成一项总体不可能定理；反复使用的人群、共同预处理和共享生成器限制了独立证据量。数学反例、代表记录证据和群体实验各自承担不同强度的结论。',size=9.3)

    r.new('接下来应增加可区分的信息，而不是继续扩大拟合自由度', '以下是本报告依据已有结果提出的研究优先级，不是新的已启动实验或授权记录')
    r.sub('1．把首个可发布目标缩小到带条件的输出功能')
    r.text('保留 SSM 代表性观测分量、完整剩余项、T/X、可见支持和参数策略；对潜变量同时给出可接受替代解见证。优先复验 X 的时间功能和连续 HbR 补全，因为已有相对正面证据；不要提前给 X 命名为代谢或神经特异成分。')
    r.sub('2．参数工作只追踪可证伪的窄候选')
    r.text('Visual 前额 montage 0 的单 τ 可以作为独立复验候选，但必须冻结同一坐标、同任务／通道配对、重复性与观测保真标准，保留 P90、近界率和全部计划分母。应先得到真正独立记录上的相容剖面与个体差异恢复，再谈联合参数扩展。')
    r.sub('3．生理来源命名需要模型外的独立观测')
    r.text('EOG 给出了可行范例：先预测独立测得的观测，再检验配对特异性。Hb 来源问题仍缺独立短间距、局部光学参考或同步外周生理测量；仅重复利用同一 HbO/HbR 的线性组合不能增加来源信息。新增哪一种观测，应由它能排除哪组竞争机制来决定。')
    r.sub('4．语义学习应接受可识别目标和强纯观测对照')
    r.text('保留独立 O 分支和等容量 O 对照。若 S 继续表示合成来源，应明确生成条件，保留精确 alias 与未见时间形状测试；幅值输出需给出识别集合或标定依赖。进入实测前，S 必须在相同输入支持下为 O 提供可重复增量，不能仅降低合成监督损失。')
    r.sub('当前不值得作为主推进指标的方向')
    r.text('只增加共同项容量、只加强参数共享、只增加响应库、只提高代理采样率、或在同一开发人群上继续调低重建 NRMSE，都不能单独回答当前两个科学目标。它们可作为受控诊断，但应以新增信息、反事实正确性与独立增量决定是否保留。')
    r.text('需改变判断的最小证据分别是：参数的独立重复与真差异恢复；分量对独立来源观测的特异响应；语义 S 在完整 O 与等容量 O 之上的冻结增量。它们不是“更好看”的输出，而是可区分竞争解释的结果。',size=9.5)

    r.new('分母、失败与尚未执行的验证', '下表保留各 owning summary 的执行单位；行数、拟合数与独立样本数不能互换或跨行求和')
    r.table(['证据组','计划与完成','失败／不确定性如何进入结论'],[
        ['E0 / E3','787 缓存记录；另有 103 人、786 记录原生审计','生产与分析集合不同；不当作缺失一条的同一统计表'],
        ['E1 / E2','求解器 72 对；困难非线性 12 项中 9 成功','求解器资格仅该面板；3 项困难拟合失败仍保留'],
        ['E4','71 人、22,903 区域窗；197,076 个收敛解重放','完整重建按计划分母；REFED 不进入联合源解释'],
        ['E5','合成 4966/4992；实测 7193/7200；profile 264 完成','隐藏 EEG 中位数是成功子集，不抹去未收敛记录'],
        ['E6','合成 8308/8352；实测记录 4200/4224；空间可用 69/72 窗','来源仍 unresolved；EOG 指向条件共同项的早期测试无明确收益'],
        ['E7','9111/9288；18 人、72 原窗、216 区域窗','135 数值预算失败、42 缺供体；三区域非独立样本'],
        ['E8','SSM 合成 5760、开发 23040 均成功；12 原型拟合；36720 探针行','三优化种子共享评价身份；训练恢复失败日志及 OOD／alias 负结果保留'],
        ['E9 / S3','实测 6885/7056 行；合成 59399/63488 行；S3 8 身份、288 行','factorial alias 不算额外独立拟合；S3 尚属合成'],
        ['E10','55 人、440 原生窗；D 128 身份；补充另 128 身份','无独立短间距／外周生理数据，E 未执行'],
        ['E11','36/36 合成；432/432 实测；18 人、72 窗','无失败不等于有生理收益；功率窗差异保留'],
        ['E12','开发 18 人；冻结评价 53 人，配对层级单列','2 条开发 C0b 未通过重积分精度；评价无该类排除；N2／长度扩展未触发'],
        ['E13','实测 755/756；合成 416/416；初版功效 864/864；v2 1152/1152','1 数值失败有惩罚；剖面 456/504 收敛，48 失败不得记为识别'],
    ],[67,229,227],size=8.15)
    r.text('报告制作只重新读取保留摘要和表格并绘图，没有重新拟合、读原始测量数组、访问其他协议的受保护 split，亦未启动新的确认实验。新身份确认、独立来源观测及未触发扩展均不能由本报告的综合文字替代。',size=9)


def build_sources(r):
    def rp(key,name='summary.json'):
        return 'experiments/runs/physiology_semantic_tokenizer/'+RUNS[key]+'/'+name
    r.new('证据索引一：观测修复、数值与固定结构基座', '每项均指向本地保留 owner；PDF 路径保留原身份，HTML 提供对应相对链接')
    r.ref('[状态／合同]','research_state/registry.json','执行与科学判读的唯一登记源；本报告不改写它。')
    r.ref('[设计]','docs/EXPERIMENT_PLAN.md','各套件范围、信息条件、比较对象与停止规则。')
    r.ref('[E0]','docs/DATA_CONTRACT.md','“无运动校正的通用缓存—2026-09-28”：旧漂移机制与 v5 none 边界。')
    r.ref('[E0]','experiments/runs/physiology_semantic_tokenizer/data_quality_audit/20260928_motion_cache_v5/migration.json','四数据集缓存迁移与验证身份。')
    r.ref('[E1]',rp('solver','measured_paired_summary.json'),'同目标二次插值求解器的 72 对结果。')
    r.ref('[E2a]',rp('wave'),'困难 Hb 波形机制定位；同目录保留重检查和失败记录。')
    r.ref('[E2b]',rp('volume'),'额外 HbR/HbT loading 的训练比例筛查。')
    r.ref('[E3]',rp('prevalence','summary.csv'),'原生／发布 Hb 与 v5 同向性统计；相关不能代替来源测量。')
    r.ref('[E4]',rp('phys'),'固定结构跨数据集语义：索引 training、reconstruction、completion 与独立积分审计。')
    r.text('图 2 使用 E4 的 reconstruction_summary.json → tables.common_later_windows → H0_fixed；早期重复性来自 training_summary.json。所有图的精确字段和筛选在 report_provenance.json 与生成脚本中保留。',size=8.8)

    r.new('证据索引二：成分、驱动、响应与语义学习', '原始拟合、分支版本与评价修正分别保留')
    r.ref('[E5]',rp('robust'),'四臂稳健性；图 7 使用 full_pass 和 measured_summary.csv 的 real/HbR_hidden。')
    r.ref('[E6]',rp('attribution'),'成分归属；counterfactual_summary.csv 给出 physiology_tau/full 的投影。')
    r.ref('[E7]',rp('modes'),'区域谱模式；measured_paired.csv 与 pairing_nulls.csv 保留共同成功与完整分母。')
    r.ref('[E8a]',rp('response'),'响应动态；dataset_equal_comparisons.csv 与 primary_null_specificity.csv 给出配对收益／九项校正。')
    r.ref('[E8b]',rp('response','tokenizer/summary.json'),'独立连续原型；public_probes 的 summaries 定义图 8，三优化种子等权。')
    r.ref('[E8c]',rp('response','tokenizer/scenario_variant_seed_summary.csv'),'图 9：分布原型、intervention、baseline/non_dct_impulse/exact_joint_alias。')
    r.ref('[E9a]',rp('strength'),'有限响应与独立空间结果；v3 correction.json 解释 ridge 失败继承修正。')
    r.ref('[E9b]',rp('strength','decision.json'),'S1/S2/F11 完整端点与合成机制条件；保留原阶段的 S3 未启动快照。')
    r.ref('[E9c]',rp('s3'),'后来单独完成的 S3 连续合成；8 身份和受限剖面，不是连续实测。')

    r.new('证据索引三：独立观测、参数稳定性与输出支持', '以下最新结果分别回答来源关联、个体参数与输出功能，不能互相授予资格')
    r.ref('[E10a]',rp('obs'),'EOG／硬件／语义范围；event_metrics.csv 与 event_comparisons.csv 为原生电位读出证据。')
    r.ref('[E10b]',rp('obs2'),'纯几何 donor 重匹配与独立慢噪声机制校准；图 10 采用 distance_caliper 主面板。')
    r.ref('[E11]',rp('rate'),'4/10 Hz 与驱动网格；aggregates/paired 保留固定 Hb 评分与自身 EEG 目标区别。')
    r.ref('[E12]',rp('params'),'图 5/6 的 evaluation_assessment、profile_intersections、synthetic 与 selected_synthetic；保留 postfit_numerical_screen。')
    r.ref('[E13a]',rp('output'),'图 11/13 的 measured_comparisons；同目录 profile_nodes.csv 与 profile_ranges.csv 联合定义图 12。')
    r.ref('[E13b]',rp('output','power_calibration_v2/summary.json'),'背景匹配功效修订，保留 16 身份与最初 6 身份对照；不覆盖初版。')
    r.ref('[E13c]','docs/report/20261010_ssm_output_support_v1.md','详细输出合同与逐项解释；科学数值仍由上两项运行 owner 持有。')
    r.sub('公式、实现和报告再现')
    r.text('动力学：src/inference/shared_driver_rk4.py；观测算子：src/inference/observation_baselines.py；参数指标与剖面：src/inference/shared_driver_stability.py；输出约束：src/inference/ssm_output_audit.py；alias 构造：src/inference/semantic_response_synthetic.py。',size=8.5)
    r.text('生成：.venv/bin/python experiments/scripts/render_ssm_research_lineage_report.py。PNG 图按最终显示宽度 240 dpi 输出；PDF 正文、表格与图注为可搜索文本。相邻 export_validation.json 记录逐页图像对象、文字与边界检查；preview/ 保留逐页渲染。WPS 本机未做交互检查。',size=8.5)


def build(out):
    read(ROOT/'research_state/registry.json')
    phys=js('phys');params=js('params');support=js('output')
    robustness=js('robust');attribution=js('attribution');strength=js('strength')
    s3=js('s3');response=js('response');rate=js('rate');power=js('output','power_calibration_v2/summary.json')
    # Read every named evidence identity before rendering its account.
    for name in ('wave','volume','modes','obs','obs2'):
        js(name)
    js('solver','measured_paired_summary.json')
    for draw in (draw_lineage,draw_baseline,draw_model,draw_parameters,draw_shrinkage,
                 draw_robustness,draw_tokenizer,draw_alias,draw_anchors,draw_continuity,
                 draw_profile,draw_eeg_increment):
        draw(out)
    equations(out,'13_metrics',[
        r'$\mathrm{NRMSE}_j=\sqrt{\mathrm{mean}_{t\in V}(\hat y_{tj}-y_{tj})^2}\,/\,\sigma_{j,\mathrm{train}}$',
        r'$d(A,B)=\max_{k\in\mathrm{released}}\frac{2|\theta_{Ak}-\theta_{Bk}|}{|\theta_{Ak}|+|\theta_{Bk}|}$',
    ],1.14,['src/inference/shared_driver_stability.py','docs/EXPERIMENT_PLAN.md'])
    equations(out,'14_dynamics',[
        r'$x=(r,s,f,v,p,q),\quad \dot s=\beta r-\kappa s-\gamma_f(f-1),\quad \dot f=s$',
        r'$\dot v=(f-v^{1/\alpha})/\tau,\qquad \dot p=(f-v^{1/\alpha-1}p)/\tau$',
        r'$\dot q=[fE(f)/E_0-v^{1/\alpha-1}q]/\tau,\quad E(f)=1-(1-E_0)^{1/f}$',
        r'$h_E=g_Er,\quad h_O=P_0(p-1)-Q_0(q-1),\quad h_R=Q_0(q-1)$',
    ],1.9,['src/inference/shared_driver_rk4.py'])
    equations(out,'15_decomposition',[
        r'$\hat y_{\mathrm{SSM}}=\mathcal{P}h_\theta(\hat x),\qquad \hat y_{\mathrm{total}}=\hat y_{\mathrm{SSM}}+\hat c$',
        r'$e_{\mathrm{remaining}}=y-\hat y_{\mathrm{SSM}}=\hat c+\epsilon,\qquad \epsilon=y-\hat y_{\mathrm{total}}$',
        r'$T=H_O+H_R,\quad X=0.35H_O-0.65H_R$',
        r'$(H_O,H_R)=(0.65,0.35)T+(1,-1)X$',
    ],1.9,['src/inference/ssm_output_audit.py'])
    equations(out,'16_limits',[
        r'$d=p-v,\qquad \dot d=-v^{1/\alpha-1}d/\tau$',
        r'$d(t)=d(0)\exp\!\left[-\int_0^t v(u)^{1/\alpha-1}/\tau\,\mathrm{d}u\right]$',
        r'$r^*=a r,\quad \beta^*=\beta/a,\quad g_E^*=g_E/a\quad (a>0)$',
    ],1.7,['src/inference/shared_driver_rk4.py','src/inference/semantic_response_synthetic.py'])
    r=Report(out)
    version=out.name.rsplit('_',1)[-1]
    r.new('预处理修复后的 SSM 实验谱系、参数恢复与信号语义',
        f'技术报告 {version}｜证据范围：2026-09-28—2026-10-10｜公开开发实验与已知真值合成检验')
    r.sub('核心判断')
    r.text('预处理修复后，状态空间模型（SSM）已具备限定坐标下的观测重建能力。此后的主要成果，是把“拟合得好”进一步拆解为参数重复性、分量可辨识性、独立信息增量和生成机制检验。稳定个体生理参数与唯一来源语义仍未获得资格，但若干有明确条件的工程输出和观测关联已经成立。')
    r.text('1．参数恢复：10 月 10 日更严格的配对检验中，单参数 τ 的重复差异中位数为 51.5%、45.5%、42.0%；联合 τ/β 及 τ/β/κ 没有合格有效配对。Visual 前额 montage 0 的单 τ 是局部候选，重复差异中位 13.6%，但仅 3/7 配对通过全部条件，P90 为 93.8%。[E12]')
    r.text('2．状态与分量：连续状态改善中央 HbR 补全，三数据集相对改善 21.3%、12.3%、24.5%；同时降低人为切点敏感性。然而，SSM Hb 仍可移动 0.5 个训练 SD，仅付出 0.665%–4.321% 的总目标代价，主要由额外项补偿。[E13]')
    r.text('3．语义进展：EOG 对前额原始电位有独立预测收益，空间 Hb 存在可复现的观测关联；T/X 坐标与完整剩余项已有清楚的计算合同。这些结果支持观测功能语义，尚不支持脑／头皮／全身、代谢或细胞来源标签。[E6、E10、E13]')
    r.text('4．语义学习：连续 observation 表征有工程收益，但加入语义分支 S 未超越 O；精确相同观测、不同幅值真值的反例使幅值区间覆盖严重失效。学习到了生成条件中的形状，并未解决真实来源的识别问题。[E8]')
    r.text('5．上界判断：已能排除若干“靠同一观测继续拟合即可解决”的主张；尚不能证明 EEG–Hb 没有共享生理动态，也不能宣布所有 SSM 改进已无空间。必须区分数学上的信息缺失、当前模型中的替代解反例和特定开发面板上的负结果。')
    r.sub('阅读路径')
    r.text('先看实验谱系与两页总览；模型与累计改动解释各分支的真实关系；随后分参数、成分、语义学习与独立观测呈现结果；末尾给出能力边界、可保留输出和后续优先级。每项实验以 E 编号连接末尾的原始证据入口。',size=9.4)
    r.text('本报告是日期固定的综合解释。运行结果以各 summary／表为准，执行状态以 research_state/registry.json 为准。本文没有新增测量实验或修改既有结论记录。',size=8.6,color=(.36,.43,.5))

    r.new('怎样阅读这些结果：四个层次与不同分母', '“能拟合”“可重复”“可辨识”“有来源语义”分别回答不同问题')
    r.text('术语：HbO/HbR 为氧合／脱氧血红蛋白；EOG 为眼电；SD 为标准差；NRMSE 为归一化均方根误差；PCA 为主成分分析；DCT 为离散余弦基底；teacher 指产生监督目标的推断器，tokenizer 指信号表征编码器。',size=9.2)
    r.table(['层次','验证对象','不能替代什么'],[
        ['观测重建','可见 EEG 功率代理和 Hb 的拟合误差、合法状态与数值重放','不证明隐藏目标、驱动或参数正确'],
        ['稳定性','跨记录、切点、先验与观测扰动的输出变化','固定值或强共享也可很稳定'],
        ['可辨识性','重新优化补偿量后，是否仍有低代价替代参数／分解','有限搜索未找到，不等于不存在'],
        ['来源语义','独立观测、配对 null、真实机制反事实和条件增量','变量名称、相关性、合成监督标签均不够'],
    ],[72,239,212],size=9)
    r.fig('13_metrics','指标定义。训练 SD 为允许训练／开发身份的尺度；重复差异取释放参数中的最差坐标。评分支持与汇总顺序仍以每个实验为准。')
    r.text('NRMSE 是误差除以训练标准差，不是解释方差或生理能量比例。9/28 的“三项 <0.5”是工程验收条件；10/10 参数实验先区域内配对、再被试内汇总，最后给出被试中位数／P90。记录、窗口、区域不能当成独立被试。')
    r.text('最新输出支持实验先计算每条记录的 NRMSE，再记录内配对、被试内平均、被试等权；采样率实验则先平均标准化平方误差，再开方。图表仅作实验内部比较，不把二者的绝对值拼成性能增长曲线。')
    r.text('大多数区间条件于冻结的坐标、训练对象与参数策略，未重新拟合整个开发流程。报告中的 bootstrap 区间是开发性描述；只有原合同规定的检验族使用 Holm 校正。优化随机种子重复的是同一批评价身份，不提供新的独立样本。')
    r.text('“真值”在实测中指被隐藏、仅用于评分的观测；在合成中指生成器状态与无噪声分量。两者均不能自动给实测潜变量提供生理真值。大量后续工作重复使用公开开发人群，不能计作多次独立确认。',size=9.4)

    r.new('实验谱系：三条主线从同一观测基座分出', '参数恢复、成分归属、独立观测与语义学习并行推进')
    r.fig('01_lineage','图 1｜问题承续图。E0–E3 为同期观测与数值基座；E4 固定六状态检验确立“重建通过但参数／来源未通过”的分界。后续箭头表示承续的问题或方法，不表示上一实验通过生理资格。')
    r.text('参数主线追问个体差异是否可重复；归属主线追问额外项是否吸收真实生理变化；语义主线要求独立观测或反事实提供区分信息。10/10 输出支持把连续性、补偿方向和 EEG 配对对照放在同一制度下检验。',size=9.4)

    r.new('实验总览一：从预处理修复到来源竞争', '每行对应一个保留实验或同期基础诊断；不是将所有窗口相加的元分析')
    r.table(['编号／日期','测试什么与目标','承续关系','最终结果'],[
        ['E0\n9/28','移除旧导数抑制，核验缓存身份和观测保真','从旧流程的累积漂移定位问题，提供后续数据基座','v5 none 默认；787 条缓存记录重建；保留真实运动污染与旧证据'],
        ['E1\n9/28','同目标求解器：二次插值是否救回失败、节省调用','只改变共享 β 的步长策略；区分计算问题与科学问题','63/72→72/72，调用减少 30.57%；不构成参数可辨识证据'],
        ['E2\n9/28','困难 Hb 波形：去惩罚、释放响应或增加慢共同方向','诊断已有固定 ROI／光学映射后残留的波形失配','额外方向明显改善重建；系数同窗口拟合，非来源证明'],
        ['E3\n9/28','原生 HbO/HbR 同向性是否普遍、是否只是预处理产物','承接困难波形，不把残差相关直接命名为伪迹','103 人、786 记录；同向性存在且处理会放大；无唯一来源'],
        ['E4\n9/28','固定六状态／单驱动；参数共享、合成恢复与缺失补全','先不加独立慢共同项，检验原结构能承担什么','71 人、22,903 区域窗；固定 H0 重建达标，个体参数／补全不足'],
        ['E5\n10/1','原模型／额外项／驱动先验／组合四臂','承接 E4 的隐藏失稳和观测失配；检验有条件 teacher','完整拟合 53/72→64/72；隐藏 EEG 稳定，但 HbR 退化且 null 不通过'],
        ['E6\n10/2','方向学习、独立空间、反事实、EOG 与 typed-token 原型','追问 E5 的新增项究竟吸收什么','固定共同项吸收约 20% 的 τ 变化投影；来源仍未定'],
    ],[53,161,142,167],size=8.4)
    r.text('E0 的 787 条是缓存生产记录，E3/E4 的 786 条是各审计合同实际纳入的记录；它们不是同一分母。E1 是计算工程资格，不能与后续生理语义判定相互替代。',size=9)

    r.new('实验总览二：增加结构以后，独立证据是否增加', '后续负结果推动问题从“增加容量”转为“增加可区分的信息”')
    r.table(['编号／日期','测试什么与目标','承续关系','最终结果'],[
        ['E7\n10/2','区域宽频＋谱对比模式能否提供更具体的共享驱动','承接 E5；扩展 EEG 表示并冻结血管混合权重','拟合略好；九个数据集×区域均未同时通过三类 null'],
        ['E8\n10/2','响应动态与纯衰减对照；连续语义与完整观测表征','合并 E7 的谱混合疑问和 E6 的反事实错配','谱特异收益不明；S 无条件增量；幅值 alias 覆盖失败'],
        ['E9\n10/3','有限 τ/κ 库、独立空间均值、混淆剖面；另行 S3 连续合成','由额外项吸收响应／初态的证据定位主要补偿量','S1 部分隐藏收益；空间与联合臂不具全面资格；S3 为合成候选'],
        ['E10\n10/9','观测坐标范围、EOG 事件、硬件 donor、竞争机制与锚点','承接 E6/E9，寻找模型外的区分信息','EOG 电位预测成立；硬件关联不能排除组织重叠；独立生理 E 未执行'],
        ['E11\n10/10','4/10 Hz 功率代理与驱动网格控制','沿用固定 H0、原 72 窗面板，排查分辨率瓶颈','Hb 完整拟合绝对差 <0.002；隐藏收益不一致，不支持默认升级'],
        ['E12\n10/10','E1–E6 参数共同内点、数值、连续性、权重、共享及真值恢复','直接回到 E4 的个体参数问题；重新冻结开发坐标和记录配对','单 τ 相对较好但总体未达标；联合参数零有效配对；共享压缩真差异'],
        ['E13\n10/10','原生连续状态、逐输出剖面、真实／错配 EEG 与功效校准','承接 E9-S3 并深化 E10 的有限语义范围','HbR 补全改善；SSM/T 仍可交换；六项 EEG 增量区间跨零'],
    ],[53,172,134,164],size=8.4)
    r.text('E9 v3 是对原空间 ridge 失败继承错误的版本化评价修正，复用旧拟合，不算新训练实验。E10 C/D 补充、E13 背景 v2 均保留初版结果；后续结论只在相应修订的边界内成立。',size=9)

    r.new('9 月 28 日的分水岭：观测重建成为可用基座', '保留观测响应是前提，工程达标仍不是生理解释资格 [E0–E4]')
    r.fig('02_reconstruction','图 2｜共同后续窗口的固定 H0 重建。柱为保留 pass_rate，线为被试 bootstrap 区间；三项 EEG/HbO/HbR 的训练 SD NRMSE 均 <0.5 才算通过，数值失败保留在计划分母。')
    r.text('旧导数抑制对非对称上升／下降赋不同权重，重新积分后引入累积漂移。v5 none 删除的是这一步，不是取消带通、重采样或参考处理。相同身份的旧审计中，HbO 晚期减早期为负从 72/72 降为 47/72；剩余下降不能一概认作伪迹。[E0]')
    r.text('跨数据集固定结构实验采用原生时间与窗口局部算子，因此也不能把它与旧整记录缓存的 NRMSE 直接相减。三数据集联合人群为 29、26、16 人；REFED 的 32 人只做 Hb 审计，不补造跨模态几何。[E4]')
    r.text('在匹配合成中，τ、β、联合 τ/β 与条件 κ 恢复筛查通过；197,076 个收敛解经 8 子步重放仍合法且达标判定不变。说明实现与限定生成条件下的恢复可行，并不说明实测个体真值已被恢复。[E4]')
    r.text('参数策略方面，Visual 的数据集／区域共享 τ 仅提高达标率 1.26 个百分点，95% 配对区间 0.33–2.41，Holm p=0.0048；个体适配没有一致增益。旧被试 τ/β 重复差异约 24%–32%，中央 Hb 补全弱于插值。这推动后续终点离开单纯降低重建误差。[E4]')

    r.new('前置诊断的定位：数值失败、困难波形与 Hb 同向性', 'E1–E3｜分别排查求解器、表达能力与观测来源，不把三者归为同一个问题')
    r.sub('同目标求解器改进是真实的工程收益')
    r.text('二次插值步长在共享 β 的 72 个实测配对起点中，使旧 63 个成功变为 72 个，救回全部 9 个失败且无退步；总优化调用减少 30.57%，CPU 时间减少 28.53%。108 次旧条件重放逐值复现历史。收益并不普遍：24/72 起点增加调用。因此保留为明确可选的数值策略，不改称生理识别改进。[E1]')
    r.sub('困难波形主要不是靠取消惩罚解决')
    r.text('S09 困难身份在 MNE 光学对照的 HbO NRMSE 约 0.937；分别移除曲率、初态或 flow 惩罚仍约 0.937，残差 HbO/HbR 相关约 0.935。增加慢共同观测方向可以明显恢复波形，但 Hb-only 无惩罚曾出现巨大驱动、近零流量及积分失败。[E2]')
    r.table(['S09：EEG / HbO / HbR','原非线性基线','线性模型＋训练 loading 额外项'],[
        ['无运动校正','0.194 / 0.993 / 0.676','0.136 / 0.428 / 0.333'],
        ['MNE 光学对照','0.181 / 0.937 / 0.580','0.115 / 0.371 / 0.303'],
    ],[159,166,198],size=8.4)
    r.text('六个训练组共 18 个唯一验证窗口；两光学路径不是 36 个独立窗口。S09 两路径训练 loading 都选 0.2；三项 <0.5 的窗口数由无校正 14/18、MNE 15/18 升至各 18/18。额外项系数仍在验证窗口同目标拟合，且比较中改变了模型表达，不能当作独立预测或正式非线性模型晋级。',size=9.3)
    r.sub('同向 Hb 在原生数据中存在，但有明显数据集与被试差异')
    r.table(['被试等权：通道窗 r>0.5 比例','Single','Sim','Visual','REFED'],[
        ['原生／发布 Hb','10.54%','5.68%','3.02%','0.088%'],
        ['v5 none 处理后','20.45%','12.45%','6.10%','5.68%'],
    ],[199,81,81,81,81],size=8.2)
    r.text('原审计覆盖 13,316 个不重叠 30 s 窗；S09 困难身份原生相关仍达 0.822。已有验证面板中同向窗更难拟合，但 22 个同向窗有 20 个来自 S09，不能忽略被试混杂。该结果支持容纳共同观测成分，未确定其为脑、头皮、硬件或全身来源。[E3]',size=9.3)

    r.new('当前共同动力学核：六个变量并非六个独立生理源', '实际推断是自由驱动轨迹＋五个常微分状态；不是把所有配置字样都视为已运行方程')
    r.fig('14_dynamics','图 3｜当前单驱动非线性核及其观测读出。公式直接对应 shared_driver_rk4._slope 与 compiled_forward；谱模式实验的混合权重另记 γ_spec，与这里的血管反馈 γ_f 不同。')
    r.table(['量','含义与单位／尺度','当前含义边界'],[
        ['r','4 Hz 分段常值自由驱动；单位依赖 EEG log-power/PCA 坐标','不是绝对神经放电或代谢量；主路径没有另加 OU 漂移方程'],
        ['s、f','s 为相对流量变化率；f 为相对基线流量','f>0；实际数值与先验、代理和初态共同有关'],
        ['v、p、q','相对容积、总 Hb 量、脱氧 Hb 量；基线均为 1','p 与 v 只有在匹配初态等条件下相等；不能混写'],
        ['τ、β、κ','τ：秒；β：有效神经血管增益；κ：每秒衰减率','β 的量纲与代理标定相连；当前不是跨设备绝对生理常数'],
        ['P₀、Q₀','参考总 Hb 与脱氧 Hb 系数；输出支持中为 1 与 0.35','是模型条件，不是本批个体绝对 Hb 的独立测量'],
    ],[58,230,235],size=8.8)
    r.text('最新输出支持固定 τ=2 s、β=1、κ=0.64、γ_f=0.32、α=0.32、E₀=0.32、EEG loading=1。参数不变是设定，不是数据识别。参数稳定性分支才分别释放 τ、β 或联合参数，二者的结果不能混用。[E12、E13]')
    r.text('四阶 Runge–Kutta（RK4）积分在全部子步检查 f/v/p/q 正值及 Q₀q≤P₀p；非法候选被拒绝。编译后端、解析切向量、独立重积分与步长优化强化了数值可信度，没有增加生理状态或来源信息。',size=9.4)

    r.new('模型到信号的接口：必须保留完整剩余项', '当前解释对象主要是处理后 EEG 功率代理与相对 Hb；不是原始 EEG 电压的可逆分解')
    r.fig('03_model','图 4｜共同数据流与候选额外项。额外项是结构化观测表达，不自动等于某种生理组织；拟合残差也必须计入 SSM 无法解释的完整部分。')
    r.text('EEG 的六通道、五频带功率经过对数与训练 PCA 后只保留有限坐标，电压相位与被丢弃方向不在该 SSM 的输出空间。不能把代理上的分解反投成完整原始电压来源；EOG 电位实验另行使用原生电位，因此能回答不同的问题。[E10、E13]')
    r.text('额外项加入后，“总预测拟合得好”可以来自 SSM 与额外项互相补偿。报告必须同时保留 SSM 分量、结构项、总预测及完整剩余项；只展示结构项会漏掉未被结构项解释的信号。隐藏观测处没有可计算的实测剩余项，不能用零填充。[E13]')
    r.text('不同分支有不同时间支持：早期 30 s 局部处理、E12 的三块 30 s 算子、E13 的原生连续 90 s Hb bin 算子并不等价。这里的连续补全含未来上下文；不能称为在线状态估计或未来预测。',size=9.4)

    r.new('累计改动清单：共享基座与实验候选分别保留', '“已实现／已测试”不等于“已成为统一默认模型”')
    r.table(['层面','修复后累积的实现／检验','结论与保留方式'],[
        ['观测链','v5 none；原生时间、局部支持、Hb 共同尺度；E13 原生样本 bin 与可见 5 s 参考','通用 none 已切换；E13 算子是特定版本，不能倒改旧结果'],
        ['数值核','合法域 RK4、编译算术、解析切向、阻尼求解、独立重积分；共享 β 二次插值步长','数值工程进展；原默认 Armijo 保留，步长候选需显式选用'],
        ['参数层级','固定 H0、数据集／区域共享、被试、收缩、记录适配；E12 联合 τ/β/κ','可做条件估计；未获得个体生理参数资格'],
        ['驱动约束','原曲率与 flow 约束；E5 增加覆盖常数／线性方向的幅度先验','控制隐藏条件发散；可能带来真实驱动偏差，未作为普适真值 prior'],
        ['额外观测项','固定同向四余弦模式、训练方向、交换方向；精确系数消元或显式联合剖面','能容纳结构失配；来源未定，固定方向与 T 的方向重叠'],
        ['空间／谱表示','区域宽频＋谱对比两模式、固定范数血管混合、独立驱动对照','条件表示改进；γ_spec 跨折变号，未确认共享来源'],
        ['响应适配','输入低通／出流松弛竞争；训练选择有限 τ/κ 库；p=v 初态消融','各自是实验候选；响应改进不能按效果归因为正确机制'],
        ['独立空间观测','跨拟合 donor 残差读出、低秩 ridge、原始 Hb ridge 强基线','存在空间可预测性；供体污染和来源归属仍限制推广'],
        ['连续性与信息','S3 合成；E12 连续参数拟合；E13 匹配边界先验、逐输出剖面、EEG null','连续性有局部收益；没有消除 SSM/T 的低代价替代解'],
        ['语义接口','T/X、完整剩余项与支持信息；独立连续 O/S 编码器及合成原型','观测表示可用；未做 VQ 晋级，未形成合格实测生理 teacher'],
    ],[66,242,215],size=8.35)
    r.text('这些改动分别由已有 library owner 和版本化 runner/config 持有。最新 E13 仍是固定参考参数、单驱动和 D-fixed 方向，并未同时采用两维谱驱动、全部响应扩展、E12 共享参数与 EOG 读出。',size=9)

    build_results(r,params,support,power,rate)
    build_conclusions(r)
    build_sources(r)
    return r.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();out=args.output_dir.resolve()
    if (out/'SSM_RESEARCH_LINEAGE.pdf').exists():
        raise FileExistsError('Use --output-dir with a fresh versioned export; retained reports are not overwritten')
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    setup();validation=build(out)
    print(json.dumps(dict(pdf=str(out/'SSM_RESEARCH_LINEAGE.pdf'),pages=len(validation['pages']),
        figures=len(validation['figures']),checks='pass'),ensure_ascii=False))


if __name__=='__main__':
    main()
