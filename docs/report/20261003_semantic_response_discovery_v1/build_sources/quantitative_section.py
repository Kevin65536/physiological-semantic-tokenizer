"""Retained-record quantitative figures for the 2026-10-02 response suite.

The callable reads metric/record CSVs and JSONs only. It never loads raw signals,
checkpoints or prediction arrays, and performs no fitting or new inference.
"""


def _quant_json_clean(value):
    import math
    from pathlib import Path
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _quant_json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_quant_json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _quant_json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _quant_json_clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_response_quantitative_figures(repo, run, out):
    """Return ``dict(figures=[...], audit=..., tables=...)`` and write PNGs/JSON.

    ``repo``, ``run`` and ``out`` are paths; relative run/out paths resolve below
    repo. All plotting imports are local so integration in another renderer does
    not import a runner or change global data roots.
    """
    import json
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle
    import numpy as np
    import pandas as pd

    repo = Path(repo).resolve()
    run = Path(run)
    out = Path(out)
    run = (repo / run).resolve() if not run.is_absolute() else run.resolve()
    out = (repo / out).resolve() if not out.is_absolute() else out.resolve()
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    sources = []
    figures = []
    tables = {}
    colors = ["#147D92", "#E69A3B", "#7560A5", "#40546A"]
    datasets = ["eeg_fnirs_single_trial", "simultaneous_eeg_nirs", "visual_cognitive_motivation"]
    ds_names = dict(zip(datasets, ["单试次 EEG-fNIRS", "同时 EEG-NIRS", "视觉认知动机"]))
    ds_short = dict(zip(datasets, ["单试次", "同时", "视觉"]))
    regions = ["prefrontal", "motor", "posterior"]
    region_names = dict(zip(regions, ["前额", "运动", "后部"]))
    arms = ["observation_only_capacity_matched", "point_restricted_generator",
            "point_randomized_generator", "distribution_randomized_generator"]
    arm_names = dict(zip(arms, ["等容量纯观测", "受限点原型", "随机点原型", "随机分布原型"]))
    arm_short = dict(zip(arms, ["纯观测", "受限点", "随机点", "分布"]))
    semantic_arms = arms[1:]
    seeds = [101, 202, 303]
    scenario_order = ["baseline", "true_a_amplitude", "true_a_shape", "feature_gain_only",
                      "Hb_gain_only", "common_Hb", "tau_change", "initial_change",
                      "non_dct_impulse", "ou_drive", "time_varying_feature_gain",
                      "rotated_spectral_mixture", "correlated_Hb_component", "input_lowpass",
                      "viscoelastic_outflow", "alternative_hrf", "independent_Hb_drive",
                      "no_coupling", "exact_eeg_alias", "exact_joint_alias"]
    scenario_names = dict(zip(scenario_order, ["基准", "真实驱动幅值", "真实驱动形状", "EEG特征增益",
                         "Hb坐标增益", "公共Hb成分", "时间常数改变", "初始态改变", "未见脉冲", "OU驱动",
                         "时变EEG增益", "旋转谱混合", "相关Hb成分", "输入低通", "粘弹出流", "替代HRF",
                         "独立Hb驱动", "无耦合", "精确EEG别名", "精确联合别名"]))
    methods = ["context_raw", "context_plus_O", "context_plus_O_plus_S",
               "context_plus_capacity_matched_observation_only"]
    method_names = ["原始上下文", "+观测码O", "+观测O+语义S", "+等容量纯观测"]

    def _quant_read_json(relative):
        path = run / relative
        sources.append(str(path.relative_to(repo)))
        return json.loads(path.read_text())

    def _quant_read_csv(relative):
        path = run / relative
        sources.append(str(path.relative_to(repo)))
        return pd.read_csv(path)

    def _quant_records(frame):
        return _quant_json_clean(frame.to_dict("records"))

    def _quant_one(frame, **matches):
        sub = frame
        for key, value in matches.items():
            sub = sub.loc[sub[key].eq(value)]
        if len(sub) != 1:
            raise ValueError(f"Expected one retained row for {matches}, got {len(sub)}")
        return sub.iloc[0]

    def _quant_new(title, nrows=1, ncols=1, **kwargs):
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, 5), squeeze=False, **kwargs)
        fig.suptitle(title, fontsize=19, color="#17334D", y=.975)
        fig.subplots_adjust(left=.10, right=.97, top=.85, bottom=.18, wspace=.42, hspace=.70)
        return fig, axs

    def _quant_save(fig, key, title, caption, source, stats, note, *, bitmap_notes=True, filename=None):
        if bitmap_notes:
            fig.text(.03, .035, note, fontsize=10, color="#4C5D6F", va="bottom")
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        clipped=[]
        for artist in fig.findobj(matplotlib.text.Text):
            if not artist.get_visible() or not artist.get_text() or artist.get_clip_on():
                continue
            bbox=artist.get_window_extent(renderer)
            if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.width+1 or bbox.y1 > fig.bbox.height+1:
                clipped.append(artist.get_text())
        if clipped:
            raise ValueError(f"Figure text outside bitmap bounds ({key}): {clipped}")
        path = figdir / (filename or f"{key}.png")
        axis_text=[artist for ax in fig.axes for artist in ax.findobj(matplotlib.text.Text)
                   if artist.get_visible() and artist.get_text()]
        minimum_axis_label_pt=min(artist.get_fontsize() for artist in axis_text)
        fig.savefig(path, dpi=220, facecolor="white")
        plt.close(fig)
        figures.append(dict(key=key, path=str(path), title=title, caption=caption,
                            conclusion=caption, source=source, stats=_quant_json_clean(stats),
                            width_inches=12, height_inches=5, dpi=220,
                            text_inside_bitmap_bounds=True,
                            minimum_axis_cell_label_pt=minimum_axis_label_pt,
                            bitmap_notes=bitmap_notes,
                            uncertainty_note=note))

    def _quant_heat(ax, data, row_labels, col_labels, title, vmin=None, vmax=None,
                    cmap="YlGnBu", fmt=".3f", colorbar=True, label_fontsize=10, title_pad=7):
        data = np.asarray(data, dtype=float)
        im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title, fontsize=12, pad=title_pad)
        ax.set_yticks(range(len(row_labels)), row_labels, fontsize=label_fontsize)
        ax.set_xticks(range(len(col_labels)), col_labels, fontsize=label_fontsize)
        ax.tick_params(length=0)
        norm = im.norm
        for iy in range(data.shape[0]):
            for ix in range(data.shape[1]):
                value = data[iy, ix]
                text = format(value, fmt) if np.isfinite(value) else "缺失"
                ax.text(ix, iy, text, ha="center", va="center", fontsize=label_fontsize,
                        color="white" if np.isfinite(value) and norm(value) > .68 else "#102A43")
        if colorbar:
            bar = ax.figure.colorbar(im, ax=ax, fraction=.034, pad=.025)
            bar.ax.tick_params(labelsize=label_fontsize)
            bar.ax.yaxis.get_offset_text().set_fontsize(label_fontsize)
        return im

    def _quant_forest(ax, rows, labels, title, color=colors[0]):
        ys = np.arange(len(rows))
        for y, row in zip(ys, rows):
            ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=color, lw=2)
            ax.scatter(row["gain"], y, color=color, s=48, zorder=3)
        ax.axvline(0, color="#8795A5", lw=1, ls="--")
        ax.set_yticks(ys, labels, fontsize=11)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("对照 − 候选：正值为候选改善", fontsize=11)
        ax.grid(axis="x", alpha=.16)
        ax.margins(y=.18)

    def _quant_subject_aggregate(frame, columns, keys):
        # Exact owner aggregation: regions/windows within subject, then subjects
        # within dataset. Dataset-equal reduction is applied by the caller.
        subject = frame.groupby(keys + ["dataset", "subject"], dropna=False)[columns].mean()
        return subject.groupby(level=list(range(len(keys) + 1))).mean().reset_index()

    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        font_family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        font_family = "DejaVu Sans"
    style = {"font.family": font_family, "font.size": 11, "axes.labelsize": 11,
             "axes.titlesize": 13, "xtick.labelsize": 10, "ytick.labelsize": 10,
             "legend.fontsize": 10, "axes.spines.top": False, "axes.spines.right": False,
             "axes.unicode_minus": False, "figure.facecolor": "white"}

    summary = _quant_read_json("summary.json")
    verified = _quant_read_json("verification.json")
    token = _quant_read_json("tokenizer/summary.json")
    pairs = _quant_read_csv("dataset_equal_comparisons.csv")
    ds_pairs = _quant_read_csv("paired_comparisons.csv")
    measured = _quant_read_csv("measured_cells.csv")
    measured_summary = _quant_read_csv("measured_summary.csv")
    nulls = _quant_read_csv("primary_null_specificity.csv")
    cross = _quant_read_csv("cross_mechanism_fit_matrix.csv")
    synth = _quant_read_csv("synthetic_summary.csv")
    metrics = _quant_read_csv("tokenizer/scenario_variant_seed_summary.csv")
    calibs = [_quant_read_json(f"calibration/{dataset}__{region}.json")
              for dataset in datasets for region in regions]
    audit = dict(retained_summary=summary, verification_passed=verified["passed"],
                 tokenizer_stage_statuses=token["stage_statuses"],
                 tokenizer_synthetic_expected=token["synthetic_expected_records"],
                 tokenizer_synthetic_completed=token["synthetic_completed_records"],
                 tokenizer_synthetic_nonfinite=token["synthetic_nonfinite_records"],
                 public_probe_planned=token["public_probe_planned_rows"],
                 public_probe_completed=token["public_probe_completed_rows"],
                 public_probe_failed=token["public_probe_failed_rows"],
                 evaluation_subjects_by_dataset={dataset: measured.loc[measured.dataset.eq(dataset), "subject"].nunique()
                                                 for dataset in datasets},
                 paired_planned_min=int(pairs.planned_pairs.min()),
                 paired_common_success_min=int(pairs.common_success_pairs.min()),
                 paired_failed_or_missing_max=int(pairs.failed_or_missing_pairs.max()),
                 synthetic_scenario_summary_groups=len(metrics),
                 synthetic_scenario_seed_counts=sorted(metrics.optimization_seed_count.unique().tolist()),
                 synthetic_scenario_unique_base_ids=sorted(metrics.unique_evaluation_base_ids.unique().tolist()),
                 synthetic_scenario_cases_per_seed_min=int(metrics.cases_per_seed_min.min()),
                 synthetic_scenario_cases_per_seed_max=int(metrics.cases_per_seed_max.max()),
                 observation_only_semantic_head="raw S head has no semantic supervision; retained synthetic semantic_predictions use training-truth-fitted linear readouts of frozen all_code_features",
                 observation_only_semantic_readout="frozen reconstruction representation plus train-only supervised ridge for shape/logA; not the raw unsupervised S head",
                 seed_interpretation="optimization repeats on shared base IDs/subjects; no independent-seed CI",
                 historical_exposure="all 71 public subjects previously exposed; evaluation is development evidence",
                 reads="retained metric and record CSV/JSON only; no fitting/raw/checkpoint/NPZ access")
    if not verified["passed"] or summary["execution"] != "completed" or token["status"] != "complete":
        raise ValueError("Retained owning execution/verification is not complete")
    if len(measured) != summary["measured"]["recorded"] or not measured.success.all():
        raise ValueError("Measured CSV denominator or success mismatch")
    if pairs.failed_or_missing_pairs.max() != 0 or nulls.failed_or_missing_pairs.max() != 0:
        raise ValueError("Incomplete paired support requires explicit plotting adaptation")

    with plt.rc_context(style):
        # 01: paired primary response contrasts and observation controls.
        comparisons = [("gamma_selected", "gamma_attenuation", "谱混合 vs 匹配衰减"),
                       ("gamma_selected", "gain_selected", "谱混合 vs 独立增益"),
                       ("gain_selected", "gamma_attenuation", "独立增益 vs 匹配衰减"),
                       ("best_stable", "gain_selected", "稳定动力学 vs 独立增益"),
                       ("best_stable", "own_Hb_prefix_ridge", "稳定动力学 vs 自身Hb历史"),
                       ("best_stable", "spatial_Hb_ridge", "稳定动力学 vs 其他区域Hb"),
                       ("spatial_Hb_local_EEG_ridge", "spatial_Hb_ridge", "其他Hb+局部EEG vs 其他Hb")]
        selected = [_quant_one(pairs, prefix_steps=40, arm=a, control=c, metric="Hb_nrmse") for a,c,_ in comparisons]
        fig, axs = _quant_new("SSM主对照：谱混合未超过匹配衰减，稳定响应仍落后于观测基线", ncols=2)
        fig.subplots_adjust(left=.25, wspace=1.0)
        _quant_forest(axs[0,0], selected[:4], [x[2] for x in comparisons[:4]], "响应扩展对照")
        _quant_forest(axs[0,1], selected[4:], [x[2] for x in comparisons[4:]], "观测条件基线", colors[1])
        for ax, rows in zip(axs[0], [selected[:4], selected[4:]]):
            lim = ax.get_xlim(); ax.set_xlim(lim[0], lim[1] + (lim[1]-lim[0])*.18)
            for y,row in enumerate(rows):
                ax.annotate(f'{row.gain:+.4f}', (row.gain,y), xytext=(7,8), textcoords="offset points", fontsize=10)
        key="quant_01_ssm_primary_forest"
        gamma=selected[0]; stable=selected[3]
        cap=(f"40点前缀下，谱混合对匹配衰减的NRMSE改善为{gamma.gain:.5f}，保留区间[{gamma.ci_low:.5f}, {gamma.ci_high:.5f}]跨零；"
             f"稳定动力学对独立增益改善{stable.gain:.5f}，但对自身Hb历史和其他区域Hb基线的改善分别为{selected[4].gain:.5f}、{selected[5].gain:.5f}，均为负。")
        _quant_save(fig,key,"SSM主对照及观测基线",cap,["dataset_equal_comparisons.csv"],
                    [dict(label=label,**row.to_dict()) for (_,_,label),row in zip(comparisons,selected)],
                    "来源：dataset_equal_comparisons.csv；15被试、360配对区域窗；区间为固定训练/选择下的探索性被试块bootstrap。")

        # 02: dataset stratification preserves the same paired endpoint.
        comps=comparisons[:4]
        fig,axs=_quant_new("数据集分层：响应改善的方向与幅度并不一致",ncols=3)
        fig.subplots_adjust(left=.20,wspace=.70)
        data=[]
        for i,dataset in enumerate(datasets):
            rows=[_quant_one(ds_pairs,dataset=dataset,prefix_steps=40,arm=a,control=c,metric="Hb_nrmse") for a,c,_ in comps]
            _quant_forest(axs[0,i],rows,[x[2] for x in comps] if i==0 else ["谱混合/衰减","谱混合/增益","增益/衰减","稳定/增益"],ds_names[dataset],colors[i])
            data.extend([dict(label=label,**row.to_dict()) for (_,_,label),row in zip(comps,rows)])
        stable_by_dataset=[_quant_one(ds_pairs,dataset=dataset,prefix_steps=40,arm="best_stable",control="gain_selected",metric="Hb_nrmse").gain for dataset in datasets]
        _quant_save(fig,"quant_02_ssm_dataset_strata","SSM配对主对照的数据集分层",
                    "每个数据集先对同一被试的窗口与区域取均值，再对被试等权汇总；三个区域不作为独立样本。"
                    "稳定动力学对独立增益的NRMSE改善在三个数据集依次为"+
                    "、".join(f"{x:+.5f}" for x in stable_by_dataset)+"；跨数据集平均不能代替方向一致性。",
                    ["paired_comparisons.csv"],data,
                    "来源：paired_comparisons.csv；评价被试数依次6、5、4；探索性被试块bootstrap，非优化种子CI。")

        # 03: every candidate score, with selected cells outlined.
        fig,axs=_quant_new("参数选择：6/9区域选择零增益，时间扩展仅在选择集冻结",nrows=2,ncols=2)
        fig.subplots_adjust(left=.14,right=.94,top=.87,bottom=.06,wspace=.62,hspace=.36)
        param_values={"gamma":[-1,-.5,0,.5,1],"gain":[0,.5,2**-.5,(.8)**.5,1],"tau_n":[0,.5,1,2],"tau_v":[0,1,3,6]}
        calibration_table=[]
        for ax,(param,candidates) in zip(axs.flat,param_values.items()):
            mat=[]
            for c in calibs:
                ranks=c["selection_rankings"][param]
                values=[next(float(x["score"]) for x in ranks if np.isclose(x[param],v)) for v in candidates]
                mat.append(np.array(values)-min(values))
                calibration_table.append(dict(dataset=c["dataset"],region=c["region"],parameter=param,
                                              selected=c["selected"]["40"][param],candidates=ranks))
            _quant_heat(ax,mat,[ds_short[c["dataset"]]+"·"+region_names[c["region"]] for c in calibs],
                        [f"{x:.3g}" for x in candidates],f"{param}：选择NRMSE − 本行最小值",0,max(.001,float(np.max(mat))),fmt=".3f",label_fontsize=11.5,title_pad=3)
            for iy,c in enumerate(calibs):
                ix=next(j for j,v in enumerate(candidates) if np.isclose(c["selected"]["40"][param],v))
                ax.add_patch(Rectangle((ix-.47,iy-.47),.94,.94,fill=False,ec="#D56A19",lw=2))
        tables["calibration_candidates"]=calibration_table
        _quant_save(fig,"quant_03_parameter_selection","谱混合、增益与时间常数的选择热图",
                    "9个数据集×区域单元中，6个选择gain=0；其余3个选择gain=0.5。热图是各参数候选的选择集NRMSE相对本行最小值，"
                    "橙框标示冻结选择；0点前缀复用40点选择，未重新选择。",
                    ["calibration/*__*.json"],calibration_table,
                    "来源：9个calibration记录；橙框=冻结候选；分数仅来自选择集；时间常数单位s；不同参数按合同顺序选择。",
                    bitmap_notes=False,filename="quant_03_parameter_selection_readable.png")

        # 04: prefix absence/presence, dataset-equal marginal descriptions.
        display_arms=["broad_a","gamma_selected","gamma_attenuation","gain_selected","best_stable",
                      "own_Hb_prefix_ridge","spatial_Hb_ridge","spatial_Hb_local_EEG_ridge"]
        display_names=["宽带驱动","谱混合","匹配衰减","独立增益","稳定动力学","自身Hb历史","其他区域Hb","其他Hb+局部EEG"]
        desc=measured_summary.query('pairing=="real"').groupby(["prefix_steps","arm"])[["Hb_nrmse","Gaussian_CRPS_normalized"]].mean()
        fig,axs=_quant_new("前缀0 vs 40：可见Hb历史显著改变任务难度与方法排序",ncols=2)
        fig.subplots_adjust(left=.15,wspace=.40)
        prefix_stats=[]
        for ax,metric,label in zip(axs[0],["Hb_nrmse","Gaussian_CRPS_normalized"],["Hb NRMSE","归一化Gaussian CRPS"]):
            for y,arm in enumerate(display_arms):
                zero=desc.loc[(0,arm),metric]; forty=desc.loc[(40,arm),metric]
                ax.plot([zero,forty],[y,y],color="#BBC5CF",lw=2)
                ax.scatter(zero,y,s=40,color=colors[1],label="0点前缀" if y==0 else None)
                ax.scatter(forty,y,s=40,color=colors[0],label="40点前缀" if y==0 else None)
                prefix_stats.append(dict(arm=arm,metric=metric,prefix_0=zero,prefix_40=forty))
            ax.set_yticks(range(len(display_arms)),display_names,fontsize=10);ax.invert_yaxis()
            ax.set_xlabel(label+"（越低越好）");ax.grid(axis="x",alpha=.16);ax.legend(loc="best")
        _quant_save(fig,"quant_04_prefix_zero_vs_forty","有无目标Hb前缀的描述性分数",
                    "0点前缀完全排除目标Hb输入；40点前缀提供10s可见历史，评价仍在20–30s。"
                    "两种前缀使用同一冻结超参数。图为被试等权后再数据集等权的边际分数，不能当作有区间的配对增益。",
                    ["measured_summary.csv"],prefix_stats,
                    "来源：measured_summary.csv；每方法/前缀360成功区域窗、15被试；离线特征补全，非原生信号因果预测。")

        # 05: conditional EEG increment beyond other Hb/history.
        spatial=[("spatial_Hb_local_EEG_ridge","spatial_Hb_ridge","局部EEG | 其他Hb"),
                 ("spatial_Hb_filterbank_EEG_ridge","spatial_Hb_ridge","滤波EEG | 其他Hb"),
                 ("spatial_Hb_filterbank_EEG_ridge","spatial_filterbank_train_permutation","真实EEG vs 训练置换")]
        fig,axs=_quant_new("条件EEG增量：加入EEG未稳定改善其他区域Hb基线",ncols=2)
        fig.subplots_adjust(left=.23,wspace=.82)
        spatial_rows=[]
        for ax,metric,label in zip(axs[0],["Hb_nrmse","Gaussian_CRPS_normalized"],["NRMSE增益","归一化CRPS增益"]):
            rows=[];labels=[]
            for prefix in [40,0]:
                for a,c,l in spatial:
                    row=_quant_one(pairs,prefix_steps=prefix,arm=a,control=c,metric=metric)
                    rows.append(row);labels.append(f"{prefix}点·{l}");spatial_rows.append(dict(label=l,**row.to_dict()))
            _quant_forest(ax,rows,labels,label)
        local=_quant_one(pairs,prefix_steps=40,arm=spatial[0][0],control=spatial[0][1],metric="Hb_nrmse")
        _quant_save(fig,"quant_05_spatial_conditional_increment","其他区域Hb与历史之上的EEG增量",
                    f"40点前缀下，其他区域Hb加入局部EEG后的NRMSE增益为{local.gain:.5f}，区间[{local.ci_low:.5f}, {local.ci_high:.5f}]为负；"
                    "滤波EEG及训练配对置换对照同样未提供可晋级的稳定增量。",
                    ["dataset_equal_comparisons.csv"],spatial_rows,
                    "来源：dataset_equal_comparisons.csv；正值=候选改善；目标Hb只允许可见前缀，空间上下文只能读其他区域Hb。")

        # 06: complete multiplicity family, no selective null reporting.
        fig,axs=_quant_new("九项配对null：Holm校正后0/9通过特异性检验",ncols=2)
        fig.subplots_adjust(left=.25,wspace=.70)
        null_names={"wrong_training_subject":"错误训练被试","nonwrapping_shift_12s":"非环绕12s移位","other_region_EEG":"其他区域EEG"}
        null_arm_names={"gamma_selected":"谱混合","gain_selected":"独立增益","best_stable":"稳定动力学"}
        labels=[null_arm_names[x.arm]+"·"+null_names[x.null_pairing] for _,x in nulls.iterrows()]
        ys=np.arange(9)
        axs[0,0].barh(ys,nulls.dataset_equal_gain,color=[colors[0] if x>0 else colors[1] for x in nulls.dataset_equal_gain])
        axs[0,0].axvline(0,color="#8795A5",ls="--");axs[0,0].set_yticks(ys,labels);axs[0,0].invert_yaxis()
        axs[0,0].set_xlim(nulls.dataset_equal_gain.min()-.010,nulls.dataset_equal_gain.max()+.007)
        axs[0,0].set_xlabel("null − 真实配对 NRMSE（正值支持真实配对）")
        for y,val in enumerate(nulls.dataset_equal_gain):
            axs[0,0].annotate(f"{val:+.4f}",(val,y),xytext=(5 if val>=0 else -5,0),textcoords="offset points",ha="left" if val>=0 else "right",va="center",fontsize=10)
        _quant_heat(axs[0,1],nulls[["p_one_sided_sign_flip","p_Holm_nine"]].values,labels,["单侧原始p","Holm(9) p"],"完整登记null家族",0,1,fmt=".3f")
        _quant_save(fig,"quant_06_nine_nulls_holm","全部九项配对null与Holm校正",
                    f"九项检验全部具有360/360公共成功配对、15个被试，Holm校正后{int(nulls.null_specificity_pass.sum())}/9通过。"
                    f"最小原始p={nulls.p_one_sided_sign_flip.min():.5f}；检验条件于固定训练与选择对象，结论仍为探索性。",
                    ["primary_null_specificity.csv"],_quant_records(nulls.drop(columns=["subject_differences"])),
                    "来源：primary_null_specificity.csv；三臂×三null；精确被试符号翻转32768模式，家族内Holm校正。")

        # 07: proper score plus calibration of observed prediction intervals.
        proper_arms=["gamma_selected","gain_selected","best_stable","own_Hb_prefix_ridge","spatial_Hb_ridge","spatial_Hb_local_EEG_ridge"]
        proper_names=["谱混合","独立增益","稳定动力学","自身Hb历史","其他Hb","其他Hb+EEG"]
        proper=measured_summary.query('pairing=="real" and prefix_steps==40').groupby("arm")[["Gaussian_CRPS_normalized","Gaussian_NLL","coverage_50","coverage_80","coverage_95"]].mean().loc[proper_arms]
        fig,axs=_quant_new("SSM预测分布：proper score与覆盖率必须一起阅读",ncols=2)
        fig.subplots_adjust(left=.09,wspace=.33)
        axs[0,0].bar(range(6),proper.Gaussian_CRPS_normalized,color=[colors[i%4] for i in range(6)])
        axs[0,0].set_xticks(range(6),proper_names,rotation=20,ha="right");axs[0,0].set_ylabel("归一化Gaussian CRPS（越低越好）")
        for i,val in enumerate(proper.Gaussian_CRPS_normalized):axs[0,0].text(i,val+.009,f"{val:.3f}",ha="center",fontsize=10)
        for i,arm in enumerate(proper_arms):
            axs[0,1].plot([.5,.8,.95],proper.loc[arm,["coverage_50","coverage_80","coverage_95"]],marker="o",color=colors[i%4],ls="-" if i<4 else "--",label=proper_names[i])
        axs[0,1].plot([.5,.95],[.5,.95],color="#8795A5",ls=":",label="名义覆盖")
        axs[0,1].set_xticks([.5,.8,.95],["50%","80%","95%"]);axs[0,1].set_ylim(.35,1.01)
        axs[0,1].set_xlabel("名义区间");axs[0,1].set_ylabel("实际覆盖率");axs[0,1].legend(ncols=2,loc="lower right")
        _quant_save(fig,"quant_07_ssm_proper_score_coverage","SSM proper score与50/80/95%覆盖",
                    "CRPS及区间覆盖均在冻结选择集残差尺度下评价。覆盖率接近名义水平只反映处理后Hb的预测校准，"
                    "不能把残差Gaussian分布解释为生理状态后验；原坐标NLL另在数值附件保留。",
                    ["measured_summary.csv"],_quant_records(proper.reset_index()),
                    "来源：measured_summary.csv；40点前缀；被试内窗口/区域→被试等权→数据集等权；残差尺度不是生理后验。")

        # 08: cross-mechanism performance of a frozen, collapsed selection policy.
        synthetic_calibs={family:_quant_read_json(f"synthetic_calibration/{family}.json")
                          for family in ["restricted","randomized"]}
        if any(c["selected"]["40"][parameter]!=0.0 for c in synthetic_calibs.values()
               for parameter in ["tau_n","tau_v"]):
            raise ValueError("Figure 08 collapse interpretation requires both families to select zero time extensions")
        scenes=["standard","nonrest","input_lowpass","viscoelastic_outflow","common_Hb","no_coupling"]
        scene_labels=["标准响应","非静息初态","输入低通","粘弹出流","公共Hb","无耦合"]
        syn_arms=["gain_selected","input_lowpass","viscoelastic_outflow"]
        sub=cross.query('family=="randomized" and prefix_steps==40')
        fig,axs=_quant_new("冻结选择策略的跨机制表现",ncols=2)
        fig.subplots_adjust(left=.15,right=.94,wspace=.48)
        for ax,metric,label in zip(axs[0],["Hb_nrmse","response_function_RMSE"],["Hb NRMSE","响应函数绝对RMSE"]):
            mat=[[_quant_one(sub,scenario=sc,arm=arm)[metric] for arm in syn_arms] for sc in scenes]
            _quant_heat(ax,mat,scene_labels,["独立增益","输入低通","粘弹出流"],label,0,max(np.nanmax(mat),.0001),fmt=".3f" if metric=="Hb_nrmse" else ".4f")
            if metric=="response_function_RMSE":
                fig.axes[-1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
        _quant_save(fig,"quant_08_ssm_synthetic_mechanisms","冻结选择策略的跨机制表现",
                    "restricted与randomized两个family均选择τn=τv=0，三个拟合臂实际为同一核；本图不提供机制识别准确率。"
                    "图中保留随机合成族、40点前缀、六种已知场景的干预评价结果；每格12个生成重复且全部成功。"
                    "Hb NRMSE与响应函数绝对RMSE描述冻结选择策略在不同生成机制下的误差，不是机制分类或非零扩展的比较。"
                    "无耦合truth响应接近零，相对响应误差分母退化，因此图用绝对误差；原始相对值保留在附件。",
                    ["cross_mechanism_fit_matrix.csv","synthetic_calibration/restricted.json","synthetic_calibration/randomized.json"],_quant_records(sub),
                    "来源：cross_mechanism_fit_matrix.csv、synthetic_calibration/{restricted,randomized}.json。\n"
                    "两生成族均选择τn=τv=0，三个拟合臂实际为同一核；随机族40点前缀，每格12/12；不提供机制识别准确率。")

        # 09: losses are arm-specific objectives, not cross-arm semantic scores.
        fig,axs=_quant_new("连续原型训练：四臂×三优化种子，选择终点在各臂内部使用",nrows=2,ncols=2)
        fig.subplots_adjust(left=.08,right=.96,top=.83,bottom=.16,wspace=.28,hspace=.75)
        fit_stats=[]
        for ax,arm in zip(axs.flat,arms):
            for i,seed in enumerate(seeds):
                fit=next(x for x in token["fits"] if x["arm"]==arm and x["seed"]==seed)
                hist=pd.DataFrame(fit["history"])
                ax.plot(hist.epoch,hist.selection_loss,color=colors[i],lw=1.5,label=f"选择·{seed}")
                ax.plot(hist.epoch,hist.train_loss,color=colors[i],lw=.8,ls=":",alpha=.65)
                best=hist.loc[hist.epoch.eq(fit["best_epoch"])].iloc[0]
                ax.scatter(best.epoch,best.selection_loss,color=colors[i],s=22,zorder=3)
                fit_stats.append({k:fit[k] for k in ["arm","seed","completed_epochs","best_epoch","best_selection_loss","parameter_count","train_base_ids","selection_base_ids"]})
            ax.set_title(arm_names[arm],fontsize=12);ax.set_xlabel("epoch");ax.set_ylabel("该臂目标loss")
            ax.grid(alpha=.15);ax.legend(ncols=3,loc="best",fontsize=10)
        _quant_save(fig,"quant_09_tokenizer_training_curves","四臂三种子的训练与选择曲线",
                    "12次拟合全部完成，每模型458168参数，512个训练base与128个选择base。实线为选择loss，虚线为训练loss，点为选定epoch。"
                    "不同臂的损失定义不同，图用于检查训练和选择过程，不据此比较跨臂的语义优越性。",
                    ["tokenizer/summary.json:fits[].history"],fit_stats,
                    "来源：tokenizer/summary.json；101/202/303为优化重复；四臂容量相同；训练上限160 epochs，patience=20。")

        # 10/11: all twenty scenarios, split ten per panel at a common scale.
        heat_stats=[]
        max_shape=float(metrics.loc[metrics.arm.isin(semantic_arms),"shape_rmse_seed_mean"].max())
        for number,variant in [(10,"control"),(11,"intervention")]:
            variant_name="控制" if variant=="control" else "干预"
            fig,axs=_quant_new(f"合成{variant_name}形状恢复：20场景、两个模态、三个监督原型",nrows=2,ncols=2)
            fig.subplots_adjust(left=.15,right=.94,top=.87,bottom=.05,wspace=.65,hspace=.30)
            local_stats=[]
            for row,modality in enumerate(["eeg","hb"]):
                for col,scenes in enumerate([scenario_order[:10],scenario_order[10:]]):
                    mat=[]
                    for sc in scenes:
                        rows=[_quant_one(metrics,arm=arm,modality=modality,scenario=sc,variant=variant) for arm in semantic_arms]
                        mat.append([x.shape_rmse_seed_mean for x in rows]);local_stats.extend([x.to_dict() for x in rows])
                    _quant_heat(axs[row,col],mat,[scenario_names[x] for x in scenes],["受限点","随机点","随机分布"],
                                f'{modality.upper()} · {variant_name} · 场景{1 if col==0 else 11}–{10 if col==0 else 20}',0,max_shape,fmt=".2f",label_fontsize=11.5,title_pad=3)
            cap=("每格是同一256个base身份上三个优化种子的形状RMSE均值；两个变体分开汇总、全图与配套图共用色标。"
                 "图聚焦三个端到端语义监督原型；纯观测臂另以冻结重建表征+训练真值拟合线性readout评价，见四臂图，并非其原始未监督S头。")
            scene_for_claim="baseline" if variant=="control" else "non_dct_impulse"
            vals=[_quant_one(metrics,arm=arms[3],modality=m,scenario=scene_for_claim,variant=variant).shape_rmse_seed_mean for m in ["eeg","hb"]]
            cap += ("随机分布原型的基准EEG/Hb形状RMSE为" if variant=="control" else "随机分布原型的未见脉冲干预EEG/Hb形状RMSE升至")+f"{vals[0]:.4f}/{vals[1]:.4f}。"
            _quant_save(fig,f"quant_{number:02d}_tokenizer_{variant}_heatmaps",f"连续原型20场景{variant_name}形状恢复",cap,
                        ["tokenizer/scenario_variant_seed_summary.csv"],local_stats,
                        "来源：scenario_variant_seed_summary.csv；每格256 base×3优化重复；均值不附独立种子CI；控制/干预图色标相同。",
                        bitmap_notes=False,filename=f"quant_{number:02d}_tokenizer_{variant}_heatmaps_readable.png")
            heat_stats.extend(local_stats)
        tables["all_synthetic_scenario_variants"]=_quant_records(metrics)

        # 12: distribution proper scores against the matched randomized point arm.
        fig,axs=_quant_new("随机分布 vs 随机点：均值恢复与proper score的收益并不等价",nrows=2,ncols=2)
        fig.subplots_adjust(left=.10,right=.96,top=.84,bottom=.17,wspace=.30,hspace=.70)
        dist_comp=[]
        for row,modality in enumerate(["eeg","hb"]):
            for col,metric in enumerate(["shape_rmse_seed_mean","log_amplitude_crps_seed_mean"]):
                ax=axs[row,col];xs=[];ys=[]
                for sc in scenario_order:
                    point=_quant_one(metrics,arm=arms[2],modality=modality,scenario=sc,variant="intervention")
                    dist=_quant_one(metrics,arm=arms[3],modality=modality,scenario=sc,variant="intervention")
                    x=point[metric];y=dist[metric];xs.append(x);ys.append(y)
                    dist_comp.append(dict(modality=modality,scenario=sc,metric=metric,point=x,distribution=y,distribution_minus_point=y-x))
                    ax.scatter(x,y,color=colors[0] if y<x else colors[1],s=35)
                    if sc in ["baseline","non_dct_impulse"] or (col==1 and sc=="exact_joint_alias"):
                        near_right=sc!="baseline"
                        ax.annotate(scenario_names[sc],(x,y),xytext=(-5 if near_right else 5,5),
                                    textcoords="offset points",ha="right" if near_right else "left",fontsize=10)
                low=min(xs+ys)*.85;high=max(xs+ys)*1.12
                ax.plot([low,high],[low,high],ls="--",color="#8795A5");ax.set_xlim(low,high);ax.set_ylim(low,high)
                ax.set_xlabel("随机点原型");ax.set_ylabel("随机分布原型")
                ax.set_title(modality.upper()+" · "+("形状RMSE" if col==0 else "log幅值CRPS"),fontsize=12)
                ax.grid(alpha=.15)
        _quant_save(fig,"quant_12_distribution_vs_random_point","随机分布与同生成族随机点的proper-score比较",
                    "每点为一个干预场景的种子均值；虚线下方表示分布原型分数较低。形状RMSE检验点预测，log幅值CRPS同时惩罚偏差与预测宽度。"
                    "该对照保持随机生成族与网络容量相同，只检验分布目标的作用；所有数值及差值保存在附件。",
                    ["tokenizer/scenario_variant_seed_summary.csv"],dist_comp,
                    "来源：scenario_variant_seed_summary.csv；20干预场景；3优化重复同256 base；颜色仅标示描述性分数方向。")

        # 13: alias coverage and width, control/intervention retained separately.
        alias_groups=[("baseline","control"),("exact_eeg_alias","control"),("exact_eeg_alias","intervention"),
                      ("exact_joint_alias","control"),("exact_joint_alias","intervention")]
        alias_labels=["基准\n控制","EEG别名\n控制","EEG别名\n干预","联合别名\n控制","联合别名\n干预"]
        fig,axs=_quant_new("别名压力测试：区间很窄，联合别名干预幅值95%覆盖崩溃",nrows=2,ncols=2)
        fig.subplots_adjust(left=.08,right=.96,top=.84,bottom=.25,wspace=.30,hspace=.75)
        alias_stats=[]
        for ax,metric,label,iscoverage in zip(axs.flat,["shape_coverage_95","log_amplitude_coverage_95","shape_width_95","log_amplitude_width_95"],
                                              ["形状95%覆盖","log幅值95%覆盖","形状95%区间宽","log幅值95%区间宽"],[True,True,False,False]):
            for i,modality in enumerate(["eeg","hb"]):
                vals=[]
                for sc,variant in alias_groups:
                    x=_quant_one(metrics,arm=arms[3],modality=modality,scenario=sc,variant=variant)
                    vals.append(x[metric+"_seed_mean"])
                    alias_stats.append(dict(modality=modality,scenario=sc,variant=variant,metric=metric,
                                            mean=x[metric+"_seed_mean"],minimum=x[metric+"_seed_min"],maximum=x[metric+"_seed_max"]))
                ax.plot(range(5),vals,marker="o",color=colors[i],label=modality.upper())
            ax.set_xticks(range(5),alias_labels);ax.set_ylabel(label)
            if iscoverage:ax.axhline(.95,color="#8795A5",ls="--");ax.set_ylim(0,1.05)
            ax.grid(axis="y",alpha=.15)
        handles,labels=axs[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,ncols=2,loc="lower center",bbox_to_anchor=(.5,.09),frameon=False)
        covs=[_quant_one(metrics,arm=arms[3],modality=m,scenario="exact_joint_alias",variant="intervention").log_amplitude_coverage_95_seed_mean for m in ["eeg","hb"]]
        _quant_save(fig,"quant_13_alias_coverage_width","精确别名的控制/干预覆盖与区间宽度",
                    f"分布原型在精确联合别名干预下，EEG/Hb的log幅值95%覆盖仅{covs[0]*100:.2f}%/{covs[1]*100:.2f}%。"
                    "控制与干预的可见输入相同，语义幅值truth却不同；窄区间不能消除此不可识别性。图保留两变体及形状、幅值宽度。",
                    ["tokenizer/scenario_variant_seed_summary.csv"],alias_stats,
                    "来源：scenario_variant_seed_summary.csv；仅随机分布原型；95%温度校准边际区间；优化种子均值，范围见附件。")

        # 14: retained fraction is selective evaluation, not a new full denominator.
        risk_parts=[];paired_parts=[];public_parts=[]
        for arm in arms:
            for seed in seeds:
                pp=_quant_read_csv(f"tokenizer/evaluation/{arm}/seed_{seed}/paired_changes.csv")
                pp["arm"]=arm;pp["optimization_seed"]=seed;paired_parts.append(pp)
                pub=_quant_read_csv(f"tokenizer/public_probes/{arm}/seed_{seed}/records.csv")
                pub["arm"]=arm;pub["optimization_seed"]=seed;public_parts.append(pub)
                if arm==arms[3]:
                    risk=_quant_read_csv(f"tokenizer/evaluation/{arm}/seed_{seed}/selective_risk.csv")
                    risk["optimization_seed"]=seed;risk_parts.append(risk)
        risk=pd.concat(risk_parts,ignore_index=True)
        paired=pd.concat(paired_parts,ignore_index=True)
        public=pd.concat(public_parts,ignore_index=True)
        if len(public)!=token["public_probe_completed_rows"] or not public.success.all():
            raise ValueError("Retained public probe CSV denominator mismatch")
        if not paired.complete_pair.all():
            raise ValueError("Incomplete synthetic counterfactual pair")
        risk_scenes=["baseline","non_dct_impulse","ou_drive","rotated_spectral_mixture","exact_joint_alias","no_coupling"]
        risk_summary=risk.groupby(["modality","scenario","requested_retained_fraction"])[["retained_fraction","shape_rmse","retained_records","total_records"]].agg(["mean","min","max"])
        fig,axs=_quant_new("OOD风险—保留曲线：按不确定性筛选不能修复生成族偏移",ncols=2)
        fig.subplots_adjust(left=.09,right=.97,top=.84,bottom=.30,wspace=.30)
        risk_colors=["#147D92","#D56A19","#7560A5","#62A27D","#A64D64","#40546A"]
        risk_stats=[]
        for ax,modality in zip(axs[0],["eeg","hb"]):
            for i,sc in enumerate(risk_scenes):
                sub=risk_summary.loc[(modality,sc)]
                ax.plot(sub["retained_fraction"]["mean"],sub["shape_rmse"]["mean"],marker="o",ms=3,color=risk_colors[i],label=scenario_names[sc])
                for fraction,row in sub.iterrows():
                    risk_stats.append(dict(modality=modality,scenario=sc,requested_retained_fraction=fraction,
                                           retained_fraction=row[("retained_fraction","mean")],shape_rmse_mean=row[("shape_rmse","mean")],
                                           shape_rmse_min=row[("shape_rmse","min")],shape_rmse_max=row[("shape_rmse","max")],
                                           retained_records_mean=row[("retained_records","mean")],total_records=row[("total_records","mean")]))
            ax.set_title(modality.upper());ax.set_xlabel("保留评价记录比例");ax.set_ylabel("保留样本形状RMSE")
            ax.grid(alpha=.15);ax.set_xlim(.08,1.02)
        handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncols=6,loc="lower center",bbox_to_anchor=(.5,.09),frameon=False)
        _quant_save(fig,"quant_14_ood_risk_retention","OOD场景的选择性风险与保留比例",
                    "每条曲线把同场景的控制与干预记录合并，按冻结不确定性排序保留一部分记录，再计算形状RMSE；三个优化重复取均值。"
                    "筛除的记录仍计入完整评价分母。未见脉冲、别名等场景不能仅靠提高拒绝比例获得已证明的语义有效性。",
                    ["tokenizer/evaluation/distribution_randomized_generator/seed_*/selective_risk.csv"],risk_stats,
                    "来源：selective_risk.csv；仅分布原型；每场景/模态512控制+干预记录、256 base；曲线均值非独立种子CI。")

        # 15: synthetic paired true response and nuisance false response.
        paired_means=paired.groupby(["arm","optimization_seed","modality","scenario"])[["true_change_rms","predicted_change_rms","change_rmse","false_change_rms"]].mean().reset_index()
        paired_seeds=paired_means.groupby(["arm","modality","scenario"])[["true_change_rms","predicted_change_rms","change_rmse","false_change_rms"]].mean().reset_index()
        fig,axs=_quant_new("反事实检验：真实驱动变化有欠响应，观测扰动产生假语义变化",nrows=2,ncols=2)
        fig.subplots_adjust(left=.08,right=.96,top=.84,bottom=.25,wspace=.33,hspace=.75)
        true_scenes=["true_a_amplitude","true_a_shape","non_dct_impulse","exact_joint_alias"]
        false_scenes=["feature_gain_only","Hb_gain_only","common_Hb","rotated_spectral_mixture","no_coupling"]
        counter_stats=[]
        for row,modality in enumerate(["eeg","hb"]):
            for col,scenes in enumerate([true_scenes,false_scenes]):
                ax=axs[row,col];x=np.arange(len(scenes));width=.22
                for i,arm in enumerate(semantic_arms):
                    subrows=[_quant_one(paired_seeds,arm=arm,modality=modality,scenario=sc) for sc in scenes]
                    vals=[z.predicted_change_rms if col==0 else z.false_change_rms for z in subrows]
                    ax.bar(x+(i-1)*width,vals,width,color=colors[i],label=arm_short[arm])
                    counter_stats.extend([z.to_dict() for z in subrows])
                if col==0:
                    truth=[_quant_one(paired_seeds,arm=arms[3],modality=modality,scenario=sc).true_change_rms for sc in scenes]
                    ax.plot(x,truth,"k_",ms=18,mew=2,label="真实变化")
                ax.set_xticks(x,[scenario_names[sc] for sc in scenes],fontsize=10)
                ax.set_title(modality.upper()+(" · 真实驱动反事实" if col==0 else " · truth=0的观测扰动"),fontsize=12)
                ax.set_ylabel("驱动变化RMS");ax.grid(axis="y",alpha=.15)
                ax.margins(y=.15)
        handles,labels=axs[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,ncols=4,loc="lower center",bbox_to_anchor=(.5,.09),frameon=False)
        alias_delta=_quant_one(paired_seeds,arm=arms[3],modality="eeg",scenario="exact_joint_alias")
        _quant_save(fig,"quant_15_counterfactual_true_false_changes","真实与假语义变化的反事实响应",
                    f"联合别名的真实驱动变化RMS为{alias_delta.true_change_rms:.5f}，分布原型预测变化为{alias_delta.predicted_change_rms:.5f}。"
                    "对truth不变的坐标增益、公共Hb、谱旋转等扰动，非零预测变化为假语义变化；图只比较三种端到端语义监督的原型。",
                    ["tokenizer/evaluation/*/seed_*/paired_changes.csv"],counter_stats,
                    "来源：paired_changes.csv；同base控制−干预配对，每臂/模态/场景256对×3优化重复；均值非独立种子CI。")
        tables["counterfactual_all_scenarios"]=_quant_records(paired_seeds)

        # Frozen public probes: retain primary methods, all rows are complete.
        public_ds=_quant_subject_aggregate(public,["NRMSE","CRPS_training_SD"],["arm","optimization_seed","prefix_steps","method"])
        public_equal=public_ds.groupby(["arm","optimization_seed","prefix_steps","method"])[["NRMSE","CRPS_training_SD"]].mean().reset_index()
        tables["public_probe_dataset_subject_equal"]=_quant_records(public_ds)
        tables["public_probe_dataset_equal"]=_quant_records(public_equal)
        primary=public_equal.loc[public_equal.arm.eq(arms[3]) & public_equal.method.isin(methods)]
        # Verify this derived retained-record aggregation against its owning record.
        for p in token["public_probes"]:
            for z in p["summaries"]:
                own=_quant_one(public_equal,arm=p["arm"],optimization_seed=p["seed"],prefix_steps=z["prefix_steps"],method=z["method"])
                for col in ["NRMSE","CRPS_training_SD"]:
                    if not np.isclose(own[col],z[col],rtol=1e-10,atol=1e-12):
                        raise ValueError(f"Public aggregation differs from owning record: {p['arm']} {col}")
        audit["public_probe_aggregation_matches_owning_records"]=True

        # 16: seed lines, not a statistical confidence interval.
        fig,axs=_quant_new("公开冻结探针：40点前缀，O有收益，S没有进一步增量",ncols=2)
        fig.subplots_adjust(left=.08,right=.97,top=.84,bottom=.22,wspace=.25)
        public_primary_stats=[]
        for ax,col,label in zip(axs[0],["NRMSE","CRPS_training_SD"],["Hb NRMSE","训练SD归一化CRPS"]):
            for i,seed in enumerate(seeds):
                vals=[_quant_one(primary,optimization_seed=seed,prefix_steps=40,method=m)[col] for m in methods]
                ax.plot(range(4),vals,marker="o",lw=1.2,color=colors[i],alpha=.75,label=f"优化种子{seed}")
            mean=primary.query("prefix_steps==40").groupby("method")[col].mean().loc[methods]
            ax.plot(range(4),mean,color="#17334D",lw=2.5,marker="D",ms=5,label="优化重复均值")
            for x,val in enumerate(mean):ax.annotate(f"{val:.4f}",(x,val),xytext=(0,10),textcoords="offset points",ha="center",fontsize=10)
            ax.set_xticks(range(4),method_names,rotation=12,ha="right");ax.set_ylabel(label+"（低为好）")
            ax.grid(axis="y",alpha=.16);ax.legend(loc="best",fontsize=10)
        mean40=primary.query("prefix_steps==40").groupby("method")[["NRMSE","CRPS_training_SD"]].mean().loc[methods]
        _quant_save(fig,"quant_16_public_probe_seed_lines","公开探针四方法的NRMSE与CRPS优化种子线",
                    f"40点前缀NRMSE种子均值依次为{mean40.NRMSE.iloc[0]:.4f}、{mean40.NRMSE.iloc[1]:.4f}、{mean40.NRMSE.iloc[2]:.4f}、{mean40.NRMSE.iloc[3]:.4f}。"
                    "O改善上下文基线；加入S后NRMSE与CRPS均未进一步改善，等容量纯观测表征更好。三种子共享同15评价被试。",
                    ["tokenizer/public_probes/distribution_randomized_generator/seed_*/records.csv","tokenizer/summary.json:public_probes"],
                    _quant_records(primary.query("prefix_steps==40")),
                    "来源：public probe records.csv/record.json；冻结合成编码器；公开train拟合、selection选ridge/残差SD；种子为优化重复。")

        # 17: each dataset/prefix, common score definitions and four methods.
        fig,axs=_quant_new("公开探针分层：零前缀收益有限，40点收益主要来自观测表征",ncols=2)
        fig.subplots_adjust(left=.15,right=.96,top=.84,bottom=.20,wspace=.45)
        matrix_stats=[]
        for ax,col,label in zip(axs[0],["NRMSE","CRPS_training_SD"],["Hb NRMSE","训练SD归一化CRPS"]):
            mat=[];rowlabels=[]
            for prefix in [0,40]:
                for dataset in datasets:
                    sub=public_ds.loc[public_ds.arm.eq(arms[3]) & public_ds.prefix_steps.eq(prefix) & public_ds.dataset.eq(dataset)]
                    means=sub.groupby("method")[col].mean().loc[methods]
                    mat.append(means.values);rowlabels.append(f"{prefix}点·{ds_short[dataset]}")
                    for method,val in means.items():matrix_stats.append(dict(prefix_steps=prefix,dataset=dataset,metric=col,method=method,optimization_seed_mean=val))
            _quant_heat(ax,mat,rowlabels,["原始","+O","+O+S","纯观测"],label,min(np.ravel(mat)),max(np.ravel(mat)),fmt=".3f")
        _quant_save(fig,"quant_17_public_probe_dataset_prefix","公开四方法的数据集与前缀分层",
                    "每行固定同一数据集与目标Hb前缀；每格为先被试等权后取三优化种子均值。"
                    "0点前缀完全排除目标Hb tokens，40点前缀保留可见目标Hb；表征收益依赖前缀与数据集，不能把总平均推广到所有数据边界。",
                    ["tokenizer/public_probes/distribution_randomized_generator/seed_*/records.csv"],matrix_stats,
                    "来源：public probes records.csv；原始/+O/+O+S/等容量纯观测共用上下文；15开发评价被试，无独立确认。")

        # 18: conditional source increment across all three supervised arms.
        fig,axs=_quant_new("冻结S的条件增量：三个监督原型×两种前缀×三个优化种子",ncols=2)
        fig.subplots_adjust(left=.16,right=.97,top=.84,bottom=.20,wspace=.42)
        increments=[]
        inc_labels=[]
        for arm in semantic_arms:
            for prefix in [40,0]:inc_labels.append(arm_short[arm]+f"·{prefix}点")
        for ax,col,label in zip(axs[0],["NRMSE","CRPS_training_SD"],["Δ NRMSE：O+S − O","Δ CRPS：O+S − O"]):
            y=0
            for arm in semantic_arms:
                for prefix in [40,0]:
                    vals=[]
                    for i,seed in enumerate(seeds):
                        before=_quant_one(public_equal,arm=arm,optimization_seed=seed,prefix_steps=prefix,method="context_plus_O")[col]
                        after=_quant_one(public_equal,arm=arm,optimization_seed=seed,prefix_steps=prefix,method="context_plus_O_plus_S")[col]
                        delta=after-before;vals.append(delta)
                        ax.scatter(delta,y+(i-1)*.13,color=colors[i],s=36,label=str(seed) if y==0 else None)
                        increments.append(dict(arm=arm,prefix_steps=prefix,optimization_seed=seed,metric=col,O=before,O_plus_S=after,delta=delta))
                    ax.plot([min(vals),max(vals)],[y,y],color="#A5B1BC",lw=2)
                    ax.scatter(np.mean(vals),y,marker="D",color="#17334D",s=30,zorder=3)
                    y+=1
            ax.axvline(0,color="#8795A5",ls="--");ax.set_yticks(range(6),inc_labels);ax.invert_yaxis()
            ax.set_xlabel(label+"（负值为S增量）");ax.grid(axis="x",alpha=.16);ax.legend(title="优化种子",loc="best",ncols=3)
        dist40=[x for x in increments if x["arm"]==arms[3] and x["prefix_steps"]==40 and x["metric"]=="NRMSE"]
        _quant_save(fig,"quant_18_public_source_increment_all_arms","全部监督原型的O→O+S条件公开增量",
                    f"随机分布原型40点前缀的O→O+S ΔNRMSE三种子分别为"+
                    "、".join(f'{x["delta"]:+.5f}' for x in dist40)+"，方向一致为变差。图同时保留随机点、受限点及0点前缀，"
                    "横线是优化重复的最小—最大范围，菱形为均值，不是独立统计置信区间。",
                    ["tokenizer/public_probes/*/seed_*/records.csv"],increments,
                    "来源：public probes records.csv；Δ=O+S−O，负为改善；同一冻结监督模型的O与O+S；范围非CI。")

        # 19: observation-only semantics are a trained linear readout of frozen
        # reconstruction features, not the raw unsupervised source head.
        probe_scenes=["baseline","non_dct_impulse","no_coupling","exact_joint_alias"]
        probe_labels=["基准","未见脉冲干预","无耦合干预","精确联合别名干预"]
        probe_arm_labels=["纯观测 + 监督线性readout","受限点（S头）","随机点（S头）","分布（S头）"]
        fig,axs=_quant_new("四臂合成评价：纯观测表征使用监督线性readout",ncols=2)
        fig.subplots_adjust(left=.15,right=.96,top=.86,bottom=.24,wspace=.45)
        probe_stats=[]
        probe_colors=["#768391",colors[0],colors[1],colors[2]]
        for ax,modality in zip(axs[0],["eeg","hb"]):
            largest=0.0
            for group,scene in enumerate(probe_scenes):
                ax.axhspan(group*5-.5,group*5+3.5,color="#EDF2F5" if group%2==0 else "white",zorder=0)
                for index,arm in enumerate(arms):
                    row=_quant_one(metrics,arm=arm,modality=modality,scenario=scene,variant="intervention")
                    mean=float(row.shape_rmse_seed_mean)
                    low=float(row.shape_rmse_seed_min);high=float(row.shape_rmse_seed_max)
                    y=group*5+index
                    ax.barh(y,mean,height=.72,color=probe_colors[index],label=probe_arm_labels[index] if group==0 else None,
                            hatch="///" if index==0 else None,edgecolor="white",linewidth=.5,zorder=2)
                    ax.plot([low,high],[y,y],color="#17334D",lw=1.2,zorder=3)
                    ax.annotate(f"{mean:.3f}",(high,y),xytext=(4,0),textcoords="offset points",va="center",fontsize=11.5,color="#17334D")
                    probe_stats.append(dict(row.to_dict(),semantic_prediction_route="frozen_all_code_features_to_training_truth_ridge" if index==0 else "direct_semantically_supervised_shape_logA_head"))
                    largest=max(largest,high)
            ax.set_yticks([group*5+1.5 for group in range(4)],probe_labels,fontsize=11.5)
            ax.set_ylim(18.8,-.8);ax.set_xlim(0,largest*1.19)
            ax.tick_params(axis="x",labelsize=11.5)
            ax.set_xlabel("形状RMSE（越低越好）",fontsize=11.5)
            ax.set_title(modality.upper()+" · 相同随机评价族",fontsize=13)
            ax.grid(axis="x",alpha=.15,zorder=1)
        handles,labels=axs[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,ncols=4,loc="lower center",bbox_to_anchor=(.5,.08),frameon=False,fontsize=11.5,
                   handlelength=1.5,columnspacing=1.1)
        baseline_obs=[_quant_one(metrics,arm=arms[0],modality=m,scenario="baseline",variant="intervention").shape_rmse_seed_mean for m in ["eeg","hb"]]
        baseline_dist=[_quant_one(metrics,arm=arms[3],modality=m,scenario="baseline",variant="intervention").shape_rmse_seed_mean for m in ["eeg","hb"]]
        uncoupled_obs=_quant_one(metrics,arm=arms[0],modality="hb",scenario="no_coupling",variant="intervention").shape_rmse_seed_mean
        uncoupled_dist=_quant_one(metrics,arm=arms[3],modality="hb",scenario="no_coupling",variant="intervention").shape_rmse_seed_mean
        _quant_save(fig,"quant_19_observation_linear_probe","纯观测监督线性readout与三种语义S头的四臂合成比较",
                    "纯观测臂先冻结重建编码器，再用训练集真值拟合all_code_features→shape/logA ridge（α=1）并随checkpoint保存probes；"
                    "保留评价调用semantic_predictions读取此线性readout，并非原始未监督S头。其余三臂直接评价语义监督的shape/logA头。"
                    "458168等参数量只指基础网络，纯观测臂另附ridge参数。"
                    f"基准EEG/Hb形状RMSE：纯观测线性readout为{baseline_obs[0]:.4f}/{baseline_obs[1]:.4f}，分布S头为{baseline_dist[0]:.4f}/{baseline_dist[1]:.4f}。"
                    f"无耦合干预Hb的误差分别为{uncoupled_obs:.4f}/{uncoupled_dist:.4f}，不能声称S头在所有场景普遍优越。"
                    "柱为三个优化重复均值、横线为种子最小—最大范围，非独立种子CI；四场景统一展示干预变体，log幅值误差另附JSON。",
                    ["tokenizer/scenario_variant_seed_summary.csv",
                     "source_snapshot_training_v3/experiments/scripts/train_semantic_response_tokenizer.py:semantic_predictions/fit"],probe_stats,
                    "来源：scenario_variant_seed_summary.csv及冻结训练源码；256共享base×3优化重复；O-only=冻结重建表征+训练真值线性readout，非原始未监督S头。",
                    bitmap_notes=False)
        figures[-1]["section"]="tokenizer"
        figures[-1]["readout_contract"]="O-only: train-only supervised ridge alpha=1 on frozen reconstruction features, with additional ridge parameters beyond the 458168-parameter base network; other arms: direct semantically supervised shape/logA heads"

    manifest = dict(schema="semantic_response_quantitative_report_v1", figures=figures,
                    audit=audit, tables=tables, source_files=sorted(set(sources)),
                    reproduction="build_response_quantitative_figures(repo, run, out)",
                    output_boundary="bitmap PNG figures and descriptive JSON only; retained evidence unchanged")
    clean = _quant_json_clean(manifest)
    (out / "quantitative_provenance.json").write_text(json.dumps(clean,ensure_ascii=False,indent=2)+"\n")
    return clean
