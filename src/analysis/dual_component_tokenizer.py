"""Train-only controls, subject-level evidence, and frozen-token mask pilot.

The cache is already divided by observation-training SD without centering. MSE
in this space is NMSE against that common frozen denominator, for S/O/sum alike.
Windows and seeds are never treated as independent biological replicates.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
import hashlib
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from src.foundation.dual_component_masked import (
    DualComponentMaskedModel, block_masks, own_context, expected_waveforms,
    fit_token_baselines, baseline_probabilities, nonwrapping_source_shift,
)

MODALITIES = ('eeg', 'hb')
BRANCHES = ('S', 'O')


def voltage_teacher_natural_metrics(x, s, eog, training_scale, low_eog):
    """Natural-data diagnostics; raw EEG is a reference, not clean truth."""
    sl = slice(6000, 10000)
    x, s, e = (np.asarray(v, float)[sl] for v in (x, s, eog))
    scale = np.asarray(training_scale, float)
    low = np.asarray(low_eog, bool)[sl]
    def coupling(value):
        a = value-value.mean(0); b = e-e.mean(0)
        denominator = np.sqrt(np.sum(a*a, axis=0)[:,None]*np.sum(b*b, axis=0)[None])
        cor = np.divide(a.T@b, denominator, out=np.zeros_like(denominator), where=denominator>1e-20)
        return float(np.median(np.max(abs(cor), axis=1)))
    return dict(change_NMSE=float(np.mean(((s-x)/scale)**2)),
        low_EOG_change_NMSE=float(np.mean(((s[low]-x[low])/scale)**2)) if low.any() else None,
        low_EOG_samples=int(low.sum()), EOG_correlation=coupling(s),
        raw_EOG_correlation=coupling(x), RMS_ratio=float(np.sqrt(np.mean(s*s)/np.mean(x*x))),
        raw_waveform_correlation=float(np.corrcoef(s.ravel(),x.ravel())[0,1]))


def voltage_teacher_delta_metrics(base_s, changed_s, disturbance, kind):
    """Only the injected change is known on real data; no natural-clean score."""
    sl = slice(6000,10000)
    delta = (np.asarray(changed_s)-np.asarray(base_s))[sl]
    n = np.asarray(disturbance)[sl]
    energy = float(np.mean(n*n))
    if energy <= 0:
        raise ValueError('nonzero intervention required; zero must be scored separately')
    expected = n if kind == 'neural_alpha' else np.zeros_like(n)
    projection = float(np.mean(delta*n)/energy)
    return dict(delta_error_NMSE=float(np.mean((delta-expected)**2)/energy),
        delta_S_RMS_ratio=float(np.sqrt(np.mean(delta*delta)/energy)),
        signed_projection_S=projection, signed_projection_O=1-projection,
        O_delta_error_NMSE=float(np.mean((n-delta-(np.zeros_like(n) if kind=='neural_alpha' else n))**2)/energy),
        injection_RMS_uV=float(np.sqrt(energy)))


def summarize_voltage_teacher(out, panel, evaluations, fits, cfg):
    """One evidence summary; subjects, windows, calibration and failures separate."""
    import pandas as pd
    out=Path(out)
    complete=[e for e in evaluations if e['status']=='completed']
    natural=[r for e in complete for r in e['natural']]
    interventions=[r for e in complete for r in e['perturbations']]
    pd.DataFrame(natural).to_csv(out/'natural_windows.csv',index=False)
    pd.DataFrame(interventions).to_csv(out/'intervention_windows.csv',index=False)
    df=pd.DataFrame(interventions)
    keys=['subject','method','kind','amplitude']
    subjects=df.groupby(keys,as_index=False)[['delta_error_NMSE','signed_projection_S','signed_projection_O']].mean()
    subjects.to_csv(out/'intervention_subjects.csv',index=False)
    aggregate=[]
    for (method,kind,amplitude), group in subjects.groupby(['method','kind','amplitude']):
        aggregate.append(dict(method=method,kind=kind,amplitude=float(amplitude),subjects=len(group),
            delta_error_NRMSE=float(np.sqrt(group['delta_error_NMSE'].mean())),
            signed_projection_S=float(group['signed_projection_S'].mean()),
            signed_projection_O=float(group['signed_projection_O'].mean())))
    paired=[];rng=np.random.default_rng(cfg['voltage_teacher']['seed'])
    for (kind,amplitude),group in subjects.groupby(['kind','amplitude']):
        table=group.pivot(index='subject',columns='method',values='delta_error_NMSE')
        difference=np.sqrt(table['wiener'].to_numpy())-np.sqrt(table['physics_wiener'].to_numpy())
        boot=difference[rng.integers(0,len(difference),(2000,len(difference)))].mean(1)
        paired.append(dict(kind=kind,amplitude=float(amplitude),subjects=len(difference),
            mean_subject_NRMSE_improvement=float(difference.mean()),
            percentile_95=[float(v) for v in np.quantile(boot,[.025,.975])],
            interpretation='positive_favors_physics; exploratory_subject_bootstrap_no_multiplicity_claim'))
    pd.DataFrame(aggregate).to_csv(out/'intervention_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'physics_paired_comparisons.csv',index=False)
    nd=pd.DataFrame(natural); validation=nd[nd.split=='validation']
    ns=validation.groupby(['subject','method'],as_index=False)[['change_NMSE','low_EOG_change_NMSE','EOG_correlation','RMS_ratio']].mean()
    ns.to_csv(out/'natural_subjects.csv',index=False)
    natural_summary=[]
    for method,g in ns.groupby('method'):
        natural_summary.append(dict(method=method,subjects=len(g),
            change_NRMSE=float(np.sqrt(g.change_NMSE.mean())),
            low_EOG_change_NRMSE=float(np.sqrt(g.low_EOG_change_NMSE.mean())),
            EOG_correlation=float(g.EOG_correlation.mean()), RMS_ratio=float(g.RMS_ratio.mean())))
    synthetic=json.loads((out/'synthetic.json').read_text())
    sd=pd.DataFrame(synthetic['metrics']);sd.to_csv(out/'synthetic_metrics.csv',index=False)
    synthetic_summary=[dict(method=m,clean_NRMSE=float(np.sqrt(g.clean_NMSE.mean())),
        mean_correlation=float(g.correlation.mean()),seeds=len(g)) for m,g in sd.groupby('method')]
    # Display only: pick low/median/high natural EOG coupling, distinct subjects.
    candidates=validation[validation.method=='raw'].sort_values(['EOG_correlation','id']).to_dict('records')
    examples=[];used=set()
    for label,q in [('low',.1),('middle',.5),('high',.9)]:
        target=float(np.quantile([r['EOG_correlation'] for r in candidates],q))
        choices=sorted([r for r in candidates if r['subject'] not in used], key=lambda r:abs(r['EOG_correlation']-target))
        chosen=choices[0];used.add(chosen['subject']);examples.append(dict(label=label,**chosen))
    pairchecks=[r for e in complete for r in e['paired_checks']]
    return dict(schema='voltage_teacher_feasibility_v1',status='completed',experiment_id=cfg['experiment_id'],
        counts=dict(planned_windows=panel['planned_windows'],planned_subjects=panel['planned_subjects'],
            prepared_windows=len(evaluations),evaluated_windows=len(complete),evaluation_failures=len(evaluations)-len(complete),
            validation_subjects=int(validation.subject.nunique()),validation_windows=int(validation.id.nunique()),
            physical_fits=len(fits),physical_successes=sum(r['status']=='completed' for r in fits),
            physical_failures=sum(r['status']!='completed' for r in fits),
            joint_physical_failures=sum(r['status']!='completed' and not r['calibration'] for r in fits),
            calibration_failures=sum(r['status']!='completed' and r['calibration'] for r in fits),
            windows_with_physics_abstention=sum(bool(e['physics_failed_rois']) for e in complete)),
        synthetic=synthetic_summary,natural_validation=natural_summary,interventions=aggregate,physics_paired=paired,
        paired_labels=dict(maximum_S_change=max(r['S_change_max_abs'] for r in pairchecks),
            maximum_O_change_error=max(r['O_change_error_max_abs'] for r in pairchecks),
            maximum_closure=max(r['closure_max_abs'] for r in pairchecks),
            meaning='construction_identity_not_a_robustness_result'),
        examples=examples,example_selection='raw_EOG_coupling_near_P10_P50_P90_distinct_validation_subjects_display_only',
        aggregate='time_channel_NMSE_then_equal_window_mean_within_subject_then_equal_subject_mean_then_square_root',
        claim_limits=cfg['claim_limits'],tokenizer_trained=False)


def verify_voltage_teacher_evidence(out):
    """Audit retained labels and re-score saved interventions independently."""
    from src.data.functional_tokenizer import identity_digest
    from src.data.dual_component_tokenizer import voltage_teacher_plan
    out=Path(out);plan=json.loads((out/'plan.json').read_text())
    resolved=json.loads((out/'resolved_config.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    if (summary['status']!='completed' or plan['protected_data']!='forbidden'
            or any(e['split'] not in ('train','validation') for e in plan['rows'])):
        raise ValueError('completed public development evidence required before audit reads')
    cfg=resolved['config'];root=Path(resolved['project_root'])
    parent='experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261028_v3'
    if (cfg['voltage_teacher']['parent_run']!=parent
            or cfg['voltage_teacher']['legacy_voltage_run']!=parent.replace('20261028_v3','20261029_voltage_v1')):
        raise ValueError('exact public baseline lineage required before audit reads')
    if voltage_teacher_plan(cfg,json.loads((root/parent/'inventory.json').read_text()))!=plan:
        raise ValueError('audit plan differs from canonical development identities')
    train={e['id'] for e in plan['rows'] if e['split']=='train'}
    calibration=json.loads((out/'noise_calibration.json').read_text())
    if set(calibration['training_ids']) != train:
        raise ValueError('noise model calibration does not match the frozen train partition')
    for co in json.loads((out/'coordinates.json').read_text()).values():
        permitted={r['id'] for e in plan['rows'] if e['id'] in train for r in e['rows']}
        if not set(co['training_ids']) <= permitted:raise ValueError('PCA used a nontraining identity')
    old=root/cfg['voltage_teacher']['legacy_voltage_run']
    count=0;checks=0;metric_error=0.;closure=0.;exact=0
    for entry in plan['rows']:
        folder=out/'evaluation'/entry['id']
        status=json.loads((folder/'metrics.json').read_text())
        if status['status']!='completed':continue
        with np.load(folder/'base.npz',allow_pickle=False) as ar:a=dict(ar)
        if a['x'].shape!=(18000,16) or not all(np.isfinite(a[k]).all() for k in ('x','physics_wiener','legacy_target')):
            raise ValueError('complete finite native-voltage label support required')
        with np.load(old/'voltage_prepared'/(identity_digest(entry['id'])[:24]+'.npz'),allow_pickle=False) as ar:
            if (not np.array_equal(a['x'][6000:10000].T,ar['eeg_u'])
                    or not np.array_equal(a['legacy_target'][6000:10000].T,ar['eeg_s'])):
                raise ValueError('legacy actual supervision differs from displayed baseline')
        exact+=1;count+=1
        if entry['split']!='validation':continue
        for kind in ('ocular','unreferenced_muscle','neural_alpha'):
            with np.load(folder/f'{kind}.npz',allow_pickle=False) as ar:b=dict(ar)
            n=b['n'].astype(float);denominator=np.mean(n*n)
            if kind!='neural_alpha':
                if not np.array_equal(a['physics_wiener'][6000:10000],b['paired_s']):
                    raise ValueError('paired S changed')
                closure=max(closure,float(np.max(abs(b['paired_s'].astype(float)+b['paired_o']-b['x']))))
            for method in ('raw','legacy_target','eog_regression','wiener','physics_wiener','shifted_physics'):
                key='x' if method=='raw' else method
                delta=b[key].astype(float)-a[key][6000:10000].astype(float)
                expected=n if kind=='neural_alpha' else 0.
                error=float(np.mean((delta-expected)**2)/denominator)
                projection=float(np.mean(delta*n)/denominator)
                row=next(r for r in status['perturbations'] if r['method']==method and r['kind']==kind and r['amplitude']==1.)
                metric_error=max(metric_error,abs(row['delta_error_NMSE']-error),abs(row['signed_projection_S']-projection));checks+=1
    if metric_error>3e-5 or closure>1e-3 or count!=summary['counts']['evaluated_windows']:
        raise ValueError('retained float32 plot/label arrays do not reproduce numerical evidence')
    result=dict(status='passed',completed_windows=count,legacy_supervision_exact_windows=exact,
        independently_rescored_interventions=checks,maximum_metric_difference_float32=metric_error,
        maximum_pair_closure_float32_uV=closure,source_summary=str(out/'summary.json'),
        training_ids=len(train),test_or_protected_signal_reads=0,
        interpretation='float64 numerical metrics; float32 retained waveform export; no additional fits')
    path=out/'verification.json'
    if path.exists() and json.loads(path.read_text())!=result:
        raise ValueError('retained verification changed; use an explicit versioned audit')
    if not path.exists():_json(path,result)
    return result


def render_voltage_teacher_report(out, export=None):
    """Static Chinese evidence report: bitmap figures, searchable PDF body."""
    import html
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scipy.signal import welch
    import pandas as pd
    import fitz
    out=Path(out).resolve(); export=Path(export).resolve() if export else out/'export_v1'
    if export.exists():
        raise ValueError('use a fresh versioned report export; retained reports are immutable')
    summary=json.loads((out/'summary.json').read_text())
    if summary['status']!='completed':raise ValueError('completed numerical evidence required')
    verify_voltage_teacher_evidence(out)
    cfg=json.loads((out/'resolved_config.json').read_text())['config']
    export.mkdir(parents=True);figures=export/'figures';figures.mkdir()
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
        'axes.unicode_minus':False,'font.size':11,'axes.titlesize':12,'axes.labelsize':11,
        'legend.fontsize':9,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    names=dict(raw='原始输入',legacy_target='当前旧目标',eog_regression='EOG 回归',
        wiener='无物理 Wiener',physics_wiener='物理 Wiener',shifted_physics='错位物理',
        physics_oracle='真功率先验（上界）',wrong_physics='错误功率先验')
    colors=dict(raw='#67717d',legacy_target='#d47a1c',eog_regression='#a65d45',
        wiener='#236eb4',physics_wiener='#168368',shifted_physics='#8b5dba',
        physics_oracle='#168368',wrong_physics='#8b5dba')
    pages=[];gallery=[];formulas={}
    def save(fig,name):
        fig.savefig(figures/(name+'.png'),dpi=240,facecolor='white');plt.close(fig)
        return 'figures/'+name+'.png'
    def add(title,intro,fig,name,caption,evidence):
        pages.append(dict(title=title,intro=intro,figure=save(fig,name),caption=caption,evidence=evidence))
    def read_case(key):
        with np.load(out/'evaluation'/key/'base.npz',allow_pickle=False) as a:return dict(a)
    def metric(method,kind,amplitude=1.):
        return next(r for r in summary['interventions'] if r['method']==method and r['kind']==kind and r['amplitude']==amplitude)
    def natural(method):return next(r for r in summary['natural_validation'] if r['method']==method)
    def axgrid(ax):ax.grid(alpha=.18);ax.axhline(0,color='#999999',lw=.6,zorder=0)
    counts=summary['counts'];m=metric('physics_wiener','ocular');w=metric('wiener','ocular')
    with np.load(out/'noise_model.npz',allow_pickle=False) as model_arrays:
        injection_rms=float(np.sqrt(np.mean(model_arrays['voltage_scale']**2)))
        noise_transfer=model_arrays['transfer'].copy()
    methods=['raw','legacy_target','eog_regression','wiener','physics_wiener','shifted_physics']
    fig,axes=plt.subplots(1,3,figsize=(13,5.2),constrained_layout=True)
    for ax,kind,title in zip(axes[1:],['ocular','neural_alpha'],['眼电注入：S 应保持','神经样变化：S 应保留变化']):
        values=[metric(k,kind)['delta_error_NRMSE'] for k in methods]
        ax.bar(range(len(methods)),values,color=[colors[k] for k in methods]);ax.set_xticks(range(len(methods)),[names[k] for k in methods],rotation=38,ha='right')
        ax.set_title(title);ax.set_ylabel('变化误差 / 已知干预 RMS（越低越好）')
        for i,value in enumerate(values):ax.text(i,value,f'{value:.3f}',ha='center',va='bottom',fontsize=9)
        ax.grid(axis='y',alpha=.2)
    syn=summary['synthetic'];axes[0].bar(range(len(syn)),[r['clean_NRMSE'] for r in syn],color=[colors[r['method']] for r in syn])
    axes[0].set_xticks(range(len(syn)),[names[r['method']] for r in syn],rotation=38,ha='right')
    axes[0].set_title('合成真值：电压恢复');axes[0].set_ylabel('误差 / 干净电压 RMS（越低越好）')
    for i,r in enumerate(syn):axes[0].text(i,r['clean_NRMSE'],f"{r['clean_NRMSE']:.3f}",ha='center',va='bottom',fontsize=9)
    axes[0].grid(axis='y',alpha=.2)
    add('EEG S/O 监督目标：先看波形，再判断物理增量',
        f"固定面板：6 名训练、{counts['validation_subjects']} 名验证被试；计划 {counts['planned_windows']} 段，完成 {counts['evaluated_windows']} 段。"
        f"本次测试监督目标，没有训练 tokenizer。眼电注入为 1 倍训练 RMS（{injection_rms:.1f} µV）时，物理版变化误差 {m['delta_error_NRMSE']:.3f}，无物理版 {w['delta_error_NRMSE']:.3f}。",
        fig,'01_overview','左图的真功率先验使用合成干净真值，仅为机制上界；右两图使用真实验证 EEG 背景及已知新增变化。两类图的分母不同，不能横向比较柱高。各组先按窗口、再按被试等权汇总。',
        ['summary.json','synthetic_metrics.csv','intervention_subjects.csv'])

    fig,ax=plt.subplots(figsize=(13,5.6));ax.set_xlim(0,13);ax.set_ylim(0,5.6);ax.axis('off')
    boxes=[(.1,3.8,2.4,1.,'原生 EEG 电压 X\n200 Hz · µV · TP9 参考'),
        (.1,1.8,2.4,1.,'真实 HEOG / VEOG\n训练估计混合 · 当前 EOG 定位'),
        (3.2,3.8,2.6,1.,'EEG 功率减噪声功率\n训练下限 + 新 PCA / 尺度'),
        (3.2,1.8,2.6,1.,'噪声谱协方差 K_N\n信号背景相关 R_S（满秩）'),
        (6.6,3.8,2.5,1.,'固定 Tak 核心 + Hb\n每窗拟合驱动和初态'),
        (6.6,1.8,2.5,1.,'带置信权重的功率融合\n只修正三个代理约束方向'),
        (10.,2.8,2.8,1.3,'全频谱多通道 Wiener\n逆 STFT → S 电压\nO = X − S')]
    for x,y,width,height,label in boxes:
        ax.add_patch(plt.Rectangle((x,y),width,height,facecolor='#eef4fa',edgecolor='#70859c',lw=1.2));ax.text(x+width/2,y+height/2,label,ha='center',va='center',fontsize=11)
    for start,end in [((2.5,4.3),(3.2,4.3)),((2.5,2.3),(3.2,2.3)),((5.8,4.3),(6.6,4.3)),((7.85,3.8),(7.85,2.8)),((5.8,2.3),(6.6,2.3)),((9.1,2.3),(10.,3.2)),((9.1,4.3),(10.,3.7))]:
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#4a6279',lw=1.8))
    formula=r'$S^*=K_S(K_S+K_N)^{-1}X,\quad s^*=\operatorname{ISTFT}(S^*),\quad o^*=x-s^*$'
    ax.text(6.5,.75,formula,ha='center',fontsize=19);formulas['voltage_teacher']=formula
    fig.tight_layout()
    add('物理模型约束慢功率，快速电压来自观测',
        'S/O 最终都保留原通道、200 Hz 时间轴及 µV 单位。Tak 输出的是 10 Hz EEG 功率代理和 Hb 重建，不能单独称为“物理模型生成的 EEG 电压”。本版伪迹条件均值设为零；EOG 回归单列为对照。',
        fig,'02_method','STFT 使用 1 秒 Hann 窗、250 ms 步幅及完整 0–100 Hz 频谱；逆变换恢复原时间轴。EOG 证据限定为 0.5–15 Hz，其覆盖之外的干扰可能被保留。信号背景作 20% 对角收缩，避免三条代理变成低秩电压空间。',
        ['resolved_config.json','noise_calibration.json','coordinates.json'])

    fig,ax=plt.subplots(figsize=(13,5.5));ax.axis('off')
    fusion=r'$\ell^S=\ell^E+\Sigma_E A^\top(A\Sigma_E A^\top+\Sigma_P)^{-1}(\widehat{u}^{\rm phys}-A\ell^E)$'
    uncertainty=r'$\Sigma_E=\operatorname{diag}(0.05+2\rho),\quad \rho=\operatorname{clip}(P_N/P_X,0,1)$'
    ax.text(.5,.83,fusion,ha='center',fontsize=24)
    ax.text(.5,.62,uncertainty,ha='center',fontsize=21)
    ax.text(.03,.40,'每个 100 ms 时点：\n16 电极 × 5 频带 = 80 个 log-power 特征\n3 个 ROI 代理：A 的维度为 3 × 80\n先融合功率，再调整 K_S；不会从 Hb 合成相位',fontsize=13,va='top',linespacing=1.8)
    ax.text(.55,.40,'ρ 越大，EEG 功率估计越不确定\nΣ_P 使用训练 Hb-only 的跨模态预测误差\nlog-power 改变量工程上限为 ±1.4\n频带之外保留原谱统计和信号空间方向',fontsize=13,va='top',linespacing=1.8)
    fig.tight_layout(pad=2);formulas.update(power_fusion=fusion,power_uncertainty=uncertainty)
    add('物理先验怎样进入电压目标：只修正少数功率方向',
        'ℓE 是扣除噪声功率、使用训练下限并减去前五秒参考的自然对数功率；A 是新训练 PCA 与尺度组成的投影。物理输出使用同一新坐标。ΣE 为 log-power 方差权重，ΣP 为代理坐标平方的误差权重。',
        fig,'02b_power_fusion','Tak 血流核心保持固定：κ=0.64、τ=2 秒、τv=2 秒；每个窗口只重估驱动与初态。权重是正则化约定，不表示两份独立真值。若 K_N=0，滤波应恢复原电压，这项性质已通过数值测试。',
        ['resolved_config.json','coordinates.json','physical_variance.json','teacher_config.json'])

    fig,axes=plt.subplots(2,2,figsize=(13,5.8));
    content=[('哪些量只在训练集确定？','EOG → EEG 混合与 ridge\n低 EOG 背景的空间相关\n频带 / 频谱下限、逐通道电压 RMS\n新噪声扣除功率的 PCA 和 EEG / Hb 尺度'),
        ('哪些量在每个 90 秒窗口估计？','EOG 支持的局部噪声协方差\n观测减噪声的功率\nTak 驱动、四个初始状态坐标\n血流核心参数保持固定'),
        ('物理权重如何避免自证？','训练被试首窗：遮蔽全部 EEG 拟合行\n仅用 Hb 拟合，再预测 EEG 代理\n取跨被试预测 MSE 与训练代理方差较大者\n它是保守正则权重，不是后验置信区间'),
        ('干扰诊断固定了什么？','原始背景 EEG、Hb、训练校准\n基准窗口的物理预测\n重算功率、噪声协方差及电压估计\n没有对污染输入重新拟合 Tak')]
    for ax,(title,body) in zip(axes.flat,content):
        ax.axis('off');ax.set_title(title,loc='left',fontweight='bold');ax.text(0,.82,body,va='top',linespacing=1.8,fontsize=12)
    fig.tight_layout(pad=2)
    confidence=json.loads((out/'physical_variance.json').read_text())
    calibration_counts='、'.join(k.rsplit('__',1)[-1]+f" {len(v['calibration_subjects'])}/{v['planned_subjects']}" for k,v in confidence.items())
    diagnostic_rows=[json.loads(path.read_text())['diagnostics'] for path in (out/'evaluation').glob('*/metrics.json')]
    floor_min=min(r['power_floor_fraction'] for r in diagnostic_rows);floor_max=max(r['power_floor_fraction'] for r in diagnostic_rows)
    add('训练校准、窗口拟合和评价各自使用什么信息',
        '所有目标都是完整窗口离线条件估计；Hb 是特权监督。评分 [30,50) 秒，处理支持 90 秒。Hb-only 权重校准只使用成功拟合，其被试覆盖为 '+calibration_counts+'；失败保留，成功子集不能代表完整校准质量。',
        fig,'03_information',f'选窗只使用元数据中的起止次序，不按拟合/噪声筛选。扣噪功率触及训练下限的比例为 {100*floor_min:.1f}%–{100*floor_max:.1f}%（90 秒×电极×频带位置），不表示这些位置全部是伪迹。物理 log-power 修正未触及 ±1.4 上限。',
        ['plan.json','coordinates.json','physical_variance.json','teacher_config.json'])

    fig,axes=plt.subplots(1,2,figsize=(13,5.4));
    formula1=r'$E_{\Delta S}=\sqrt{\frac{\sum_{c,t}(\Delta S_{ct}-d_{ct})^2}{\sum_{c,t}n_{ct}^2}}$'
    formula2=r'$p_S=\frac{\langle\Delta S,n\rangle}{\langle n,n\rangle},\qquad p_O=1-p_S$'
    for ax in axes:ax.axis('off')
    axes[0].text(0,.83,formula1,fontsize=21);axes[0].text(0,.60,'新增伪迹：d = 0\n神经样变化：d = n\n先计算完整 20 秒数组，最后才绘图\n自然改变量另用固定训练逐通道 RMS',fontsize=13,linespacing=1.8,va='top')
    axes[1].text(0,.83,formula2,fontsize=20);axes[1].text(0,.60,'有符号投影：正负和大于 1 都有可能\n它不是来源能量占比；S/O 不要求正交\n窗口先在被试内平均，再等权汇总被试\n区间按整名被试重采样，不把时间点当样本',fontsize=13,linespacing=1.8,va='top')
    fig.tight_layout(pad=2);formulas.update(delta_error=formula1,signed_projection=formula2)
    add('怎样判定干扰去了哪里，而不是只看波形变平',
        '眼电和高频干扰应主要改变 O；神经样 burst 应由 S 保留。变化误差 0 表示完全符合该干预要求。原始直通输入在伪迹诊断中为 1，在神经保留诊断中为 0。自然数据没有干净 EEG 真值。',
        fig,'04_metrics','总体柱高为：每窗时间/通道归一平方误差 → 同被试窗口均值 → 被试等权均值 → 开方。物理配对差另计算每名被试 NRMSE 之差；2000 次整被试 bootstrap 百分位区间仅为探索性，不宣称多重校正显著性。',
        ['intervention_windows.csv','intervention_subjects.csv','physics_paired_comparisons.csv'])

    with np.load(out/'synthetic_example.npz',allow_pickle=False) as archive:synthetic=dict(archive)
    t=np.arange(18000)/200;take=(t>=36)&(t<=44)
    fig,axes=plt.subplots(4,1,figsize=(13,7),sharex=True,constrained_layout=True)
    ylim=max(np.max(abs(synthetic[k][take,0])) for k in ['x','clean','legacy_target','wiener','physics_oracle'])*1.05
    for ax,key in zip(axes,['x','legacy_target','wiener','physics_oracle']):
        ax.plot(t[take],synthetic['clean'][take,0],color='black',lw=1.,label='隐藏干净电压')
        label='原始受污染输入' if key=='x' else names[key]
        ax.plot(t[take],synthetic[key][take,0],color=colors.get(key,colors['raw']),lw=.9,alpha=.8,label=label)
        ax.set_ylim(-ylim,ylim);ax.set_ylabel('µV');ax.legend(loc='upper right',ncol=2);axgrid(ax)
    axes[-1].set_xlabel('窗口内时间（秒）')
    add('合成真值显示：有噪声抑制，但仍有明显电压误差',
        '显示预先固定的评价种子 200、通道 1。黑线是生成器提供的干净神经样电压，仅用于评分与合成 oracle 对照。各方法使用相同原始输入；纵轴完全一致。真实功率先验并不包含快速相位真值。',
        fig,'05_synthetic_waveform','噪声模型只覆盖带通 EOG 证据，合成眼电含更慢的变化；未被该证据覆盖的偏移可能保留。此页展示幅度、相位及偏移，不能把曲线较平滑等同于恢复正确。总体数字来自 12 个独立生成种子。',
        ['synthetic.json','synthetic_example.npz','synthetic_metrics.csv'])

    fig,axes=plt.subplots(1,2,figsize=(13,5.2),constrained_layout=True)
    sl=slice(6000,10000);bias_rows=[];clean=synthetic['clean'][sl]
    for method in ['raw','legacy_target','wiener','physics_oracle']:
        value=synthetic['x' if method=='raw' else method][sl]-clean
        frequencies,power=welch(value,fs=200,nperseg=2000,noverlap=1000,detrend=False,axis=0)
        axes[0].plot(frequencies,10*np.log10(np.maximum(power.mean(1),1e-12)),color=colors[method],label=names[method])
        bias=float(np.mean(value.mean(0)**2)/np.mean(clean**2))
        residual=float(np.mean((value-value.mean(0))**2)/np.mean(clean**2))
        bias_rows.append((method,bias,residual))
    axes[0].set_xscale('symlog',linthresh=.5);axes[0].set_xlim(0,100);axes[0].axvspan(.5,15,color='#dcebdc',alpha=.5)
    axes[0].set_xlabel('频率（Hz；0–0.5 Hz 为线性段）');axes[0].set_ylabel('误差功率谱密度（dB µV²/Hz）');axes[0].set_title('种子 200：S − 干净真值的频谱');axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
    axes[1].bar(range(4),[r[1] for r in bias_rows],color='#a2afb9',label='窗口均值偏差平方')
    axes[1].bar(range(4),[r[2] for r in bias_rows],bottom=[r[1] for r in bias_rows],color='#236eb4',label='去均值后的误差方差')
    axes[1].set_xticks(range(4),[names[r[0]] for r in bias_rows],rotation=25,ha='right');axes[1].set_ylabel('误差平方 / 干净电压均方');axes[1].set_title('误差分解：均值偏差 + 波形误差');axes[1].legend(fontsize=9)
    wr=next(r for r in bias_rows if r[0]=='wiener');fraction=wr[1]/(wr[1]+wr[2])
    add('合成负结果的定位：慢偏移也属于原电压的一部分',
        f"同一预选种子 200 中，无物理 Wiener 的误差有 {100*fraction:.1f}% 来自 20 秒窗口的均值偏差。绿色频率区为 EOG 参考带通的名义范围。这里保留 DC，没有为评价而额外去均值。",
        fig,'05b_synthetic_error_structure','左图用 10 秒 Welch 窗、50% 重叠、禁止去均值；右图是完整 20 秒数组的精确二阶分解。该例提示 0.5–15 Hz 噪声证据不足以处理全部慢伪迹；这是范围诊断，不据此重新调参或改写本组结果。',
        ['synthetic_example.npz','resolved_config.json'])

    fig,axes=plt.subplots(1,3,figsize=(13,5.2),constrained_layout=True)
    nd=pd.read_csv(out/'natural_subjects.csv')
    for ax,column,title,square in zip(axes,['EOG_correlation','low_EOG_change_NMSE','RMS_ratio'],
            ['S 与 EOG 的相关（仅诊断）','低 EOG 区间的波形改变量','S / 原始 EEG 的 RMS 比'],[False,True,False]):
        for j,method in enumerate(methods):
            vals=nd[nd.method==method][column].to_numpy();vals=np.sqrt(vals) if square else vals
            ax.scatter(j+np.linspace(-.1,.1,len(vals)),vals,color=colors[method],s=36)
            ax.plot([j-.2,j+.2],[np.mean(vals)]*2,color=colors[method],lw=2)
        ax.set_xticks(range(len(methods)),[names[k] for k in methods],rotation=35,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('逐通道 max |r(EEG,EOG)| 的中位数')
    axes[1].set_ylabel('改变量 / 固定训练电压 RMS')
    axes[2].set_ylabel('无量纲幅度比');axes[2].axhline(1,color='#888888',ls='--')
    add('天然眼电相关降低，也必须同时检查波形损伤',
        f"每个点代表一名验证被试（n={counts['validation_subjects']}）；短横线为被试均值。低 EOG 阈值只来自训练窗口的 EOG 活动中位数；低 EOG 不等于完全无伪迹，更不等于神经真值。",
        fig,'06_natural_diagnostics','相关下降可能来自去眼电，也可能来自误删与 EOG 相关的脑活动。幅度变小本身不是收益。低 EOG 改变量用于提示损伤风险，不能把保留原始背景的指标解释为绝对去噪精度。',
        ['natural_windows.csv','natural_subjects.csv','noise_calibration.json'])

    fig,axes=plt.subplots(1,3,figsize=(13,5.2),constrained_layout=True)
    for ax,kind,title in zip(axes,['ocular','unreferenced_muscle','neural_alpha'],
            ['有辅助参考的眼电','无辅助参考的高频干扰','应保留的神经样 α burst']):
        for method in methods:
            rows=sorted([r for r in summary['interventions'] if r['method']==method and r['kind']==kind],key=lambda r:r['amplitude'])
            ax.plot([r['amplitude'] for r in rows],[r['delta_error_NRMSE'] for r in rows],marker='o',color=colors[method],label=names[method])
        ax.set_title(title);ax.set_xlabel('注入 RMS / 固定训练 RMS');ax.set_xticks([.5,1,2]);ax.set_ylabel('变化误差 / 已知干预 RMS');axgrid(ax)
    axes[0].legend(loc='best',fontsize=9)
    add('三类干预检验不同能力：去伪迹与保留神经变化',
        f'0.5、1、2 倍训练 RMS 对应 {injection_rms*.5:.2f}、{injection_rms:.2f}、{injection_rms*2:.2f} µV 的全电极/20 秒注入 RMS，是工程压力测试，不是正常生理幅度范围。眼电取未参与训练的供体，混合有 30% 通道偏差；高频干扰为 25–70 Hz，神经样 burst 为 10 Hz。',
        fig,'07_interventions','左、中两图期望 ΔS=0；右图期望 ΔS=神经样变化，均越低越好。EOG 不变时，固定线性回归本就精确保留加性神经变化；它的零误差不证明 EOG 不含脑信号。每点按窗口和被试汇总，九种干预不是独立被试。',
        ['intervention_windows.csv','intervention_subjects.csv','intervention_summary.csv'])

    fig,axes=plt.subplots(1,2,figsize=(13,5.3),constrained_layout=True)
    paired=summary['physics_paired'];labels=[]
    kindnames=dict(ocular='眼电',unreferenced_muscle='高频干扰',neural_alpha='神经样变化')
    for i,r in enumerate(paired):
        lo,hi=r['percentile_95'];center=r['mean_subject_NRMSE_improvement']
        axes[0].errorbar(center,i,xerr=[[max(center-lo,0)],[max(hi-center,0)]],fmt='o',color='#168368',capsize=3)
        labels.append(f"{kindnames[r['kind']]} · {r['amplitude']:g} 倍")
    axes[0].set_yticks(range(len(labels)),labels);axes[0].axvline(0,color='#444444',ls='--');axes[0].set_xlabel('无物理误差 − 物理误差（正值有利于物理）');axes[0].set_title('配对被试差及 95% 探索性区间')
    for i,kind in enumerate(['ocular','unreferenced_muscle','neural_alpha']):
        rows=[metric(method,kind) for method in ['wiener','physics_wiener','shifted_physics']]
        for j,(method,r) in enumerate(zip(['wiener','physics_wiener','shifted_physics'],rows)):
            axes[1].bar(i+(j-1)*.24,r['delta_error_NRMSE'],width=.22,color=colors[method],label=names[method] if i==0 else None)
    axes[1].set_xticks(range(3),['眼电','高频干扰','神经样变化']);axes[1].set_ylabel('变化误差 / 干预 RMS');axes[1].set_title('1 倍强度：正确与错位物理先验');axes[1].legend();axes[1].grid(axis='y',alpha=.2)
    add('物理增量需要同数据、同噪声估计下的配对证据',
        '无物理、有物理、错位物理三个版本共享噪声模型、训练校准与全部输入。错位版本仅将基准物理预测循环移位 45 秒；它是失配控制，不是新的生理解释。区间跨零时不能宣称稳定增益。',
        fig,'08_physics_increment','区间按整名验证被试重采样 2000 次；5 名验证被试只能提供有限的探索性信息。未做多重比较显著性声明。无参考高频干扰上的接近不变，也不能当成物理模型解决了该伪迹。',
        ['physics_paired_comparisons.csv','intervention_subjects.csv'])

    for number,example in enumerate(summary['examples'],1):
        key=example['id'];a=read_case(key);channels=[str(c) for c in a['channels']];subject=example['subject']
        with (out/'prepared'/f'{key}.json').open() as handle:meta=json.load(handle)
        entry=meta['entry'];start=entry['rows'][0]['eeg_start_s'];tag=f'{number:02d}_{subject}'
        case_metrics=json.loads((out/'evaluation'/key/'metrics.json').read_text())
        failed_rois=case_metrics['physics_failed_rois']
        picked=[channels.index(n) for n in ('Fp1','C3','C4') if n in channels]
        while len(picked)<3:picked.append(next(i for i in range(len(channels)) if i not in picked))
        fig,axes=plt.subplots(3,3,figsize=(13,7),sharex=True,constrained_layout=True)
        proxytime=np.arange(900)/10
        roi_labels=[r['anchor']['region'] for r in entry['rows']]
        for i in range(3):
            axes[i,0].plot(proxytime,a['proxy'][:,i],color='#236eb4',lw=.8,label='噪声扣除后的观测代理')
            if i not in failed_rois:
                axes[i,0].plot(proxytime,a['physical'][:,i],color='#168368',lw=1.4,label='Tak 条件重建')
            else:
                axes[i,0].text(.05,.85,'拟合失败：不显示回退值为物理重建',transform=axes[i,0].transAxes,fontsize=9,color='#b64f55')
            axes[i,0].set_ylabel(roi_labels[i]+'\n训练缩放代理')
            for j in (1,2):
                axes[i,j].plot(proxytime,a['hb'][:,i,j-1],color='#67717d',lw=.9,label='缓存实测')
                axes[i,j].plot(proxytime,a['hb_fit'][:,i,j-1],color='#168368',lw=1.4,label='Tak 条件重建')
                axes[i,j].set_ylabel('µM')
            for ax in axes[i]:ax.axvspan(30,50,color='#dbe4eb',alpha=.55,zorder=-1);axgrid(ax)
        for ax,title in zip(axes[0],['EEG 慢功率代理（非电压）','HbO','HbR']):ax.set_title(title);ax.legend(fontsize=8)
        for ax in axes[-1]:ax.set_xlabel('90 秒窗口内时间（秒）')
        good=[i for i in range(3) if i not in failed_rois]
        error=np.sqrt(np.mean(((a['proxy'][300:500,good]-a['physical'][300:500,good])/.025)**2)) if good else float('nan')
        add(f'样例 {number} · {subject}：物理模型实际重建了什么',
            f"{key}；原记录起点 {start:.3f} 秒。每行一个 ROI，灰色区域为评分 [30,50) 秒。成功的 {len(good)}/3 个 ROI 的 EEG 代理 NRMSE 为 {error:.3f}（固定训练 SD=0.025）。Hb 为带通缓存、减五秒参考后的浓度变化。",
            fig,tag+'_physical','新噪声扣除功率与旧含伪迹代理使用不同的训练坐标，不能混用。Tak 在完整窗口联合使用 EEG 代理与 Hb；这是条件重建，不是跨模态预测。生理状态与神经电压真值均未被这张拟合图验证。',
            [f'evaluation/{key}/base.npz',f'prepared/{key}.json','coordinates.json'])

        from src.data.dual_component_tokenizer import voltage_eog_reference
        filtered_eog=voltage_eog_reference(a['eog'],cfg['voltage_teacher'])
        noise_estimate=filtered_eog@noise_transfer
        sl=slice(6000,10000);times=np.arange(4000)/200;c0=picked[0]
        fig,axes=plt.subplots(3,1,figsize=(13,6.4),sharex=True,constrained_layout=True)
        for i in (0,1):
            axes[i].plot(times,a['eog'][sl,i],color='#aab1b9',lw=.8,label='原生辅助电极')
            axes[i].plot(times,filtered_eog[sl,i],color='#236eb4',lw=.9,label='估计器使用的 0.5–15 Hz 参考')
            axes[i].set_ylabel(meta['eog_names'][i]+' · µV');axes[i].legend(ncol=2,loc='upper right')
        axes[2].plot(times,a['x'][sl,c0],color=colors['raw'],lw=.7,label='原始 '+channels[c0])
        axes[2].plot(times,noise_estimate[sl,c0],color=colors['eog_regression'],lw=.8,label='训练混合预测的眼电（协方差依据）')
        axes[2].plot(times,a['x'][sl,c0]-a['physics_wiener'][sl,c0],color=colors['physics_wiener'],ls='--',lw=.9,label='候选 O')
        axes[2].set_ylabel(channels[c0]+' · µV');axes[2].legend(ncol=3,loc='upper right')
        for ax in axes:axgrid(ax);ax.axvspan(9,11,color='#dbe4eb',alpha=.4)
        axes[-1].set_xlabel('评分片段内时间（秒）')
        before=max(abs(np.corrcoef(a['x'][sl,c0],filtered_eog[sl,j])[0,1]) for j in (0,1))
        after=max(abs(np.corrcoef(a['physics_wiener'][sl,c0],filtered_eog[sl,j])[0,1]) for j in (0,1))
        add(f'样例 {number} · {subject}：O 的变化是否有独立眼电证据',
            f'上两行是真实辅助电极，第三行为同一时段的 {channels[c0]}。该电极与两个带通 EOG 的最大绝对相关由 {before:.3f} 变为 {after:.3f}。灰色时段是后续固定两秒放大范围；各行使用独立的 µV 纵轴。',
            fig,tag+'_eog_evidence','EOG 混合只由训练被试估计。棕线是噪声统计模型的电压预测，不是天然伪迹真值；本版 Wiener 没有直接将它作为条件均值扣除。时间上的共同峰值提供眼电证据，但相关下降不能单独证明脑信号无损。',
            [f'evaluation/{key}/base.npz','noise_model.npz','noise_calibration.json'])

        fig,axes=plt.subplots(3,2,figsize=(13,7),sharex=True,constrained_layout=True)
        sl=slice(6000,10000);times=np.arange(4000)/200
        for i,c in enumerate(picked):
            raw=a['x'][sl,c];old=a['legacy_target'][sl,c];new=a['physics_wiener'][sl,c]
            limit=max(np.max(abs(v)) for v in (raw,old,new,raw-old,raw-new))*1.04
            for value,label,color in [(raw,'原始 EEG',colors['raw']),(old,'当前 S 目标',colors['legacy_target']),(new,'新 S 目标',colors['physics_wiener'])]:
                axes[i,0].plot(times,value,color=color,lw=.65,label=label,alpha=.85)
            axes[i,1].plot(times,raw-old,color=colors['legacy_target'],lw=.8,label='当前 O 目标')
            axes[i,1].plot(times,raw-new,color=colors['physics_wiener'],lw=.8,label='新 O 目标')
            for ax in axes[i]:ax.set_ylim(-limit,limit);ax.set_ylabel(channels[c]+' · µV');axgrid(ax)
        axes[0,0].set_title('S 电压与原始观测');axes[0,1].set_title('O = 原始观测 − S（同一电极使用相同纵轴）')
        axes[0,0].legend(ncol=3,loc='upper right');axes[0,1].legend(ncol=2,loc='upper right')
        for ax in axes[-1]:ax.set_xlabel('评分片段内时间（秒；原窗口 30–50 秒）')
        change=np.sqrt(np.mean(((a['physics_wiener'][sl]-a['x'][sl])/a['voltage_scale'])**2))
        c0=picked[0];r0=a['x'][sl,c0];s0=a['physics_wiener'][sl,c0]
        peak0=float(np.max(abs(r0)));peak1=float(np.max(abs(s0)))
        time0=float(np.argmax(abs(r0))/200);time1=float(np.argmax(abs(s0))/200)
        add(f'样例 {number} · {subject}：原始波形与实际 S/O 监督',
            f"新 S 改变量为 {change:.3f} 倍固定训练逐通道 RMS。{channels[c0]} 的绝对峰值由 {peak0:.1f} 变为 {peak1:.1f} µV，发生时点由 {time0:.2f} 变为 {time1:.2f} 秒。当前目标与历史实际监督完全一致；新目标仅为这次候选。",
            fig,tag+'_targets','每行 S 与 O 使用相同 µV 刻度；不同电极为各自刻度，未裁剪峰值。O 可以包含真实脑活动及估计误差，不能直接称为纯噪声。所有通道、完整数组和训练/验证窗口图集另行保留。',
            [f'evaluation/{key}/base.npz',f'prepared/{key}.json','natural_windows.csv'])

        fig,axes=plt.subplots(3,2,figsize=(13,6.3),sharex=True,constrained_layout=True)
        zoom=(times>=9)&(times<=11)
        for i,c in enumerate(picked):
            raw=a['x'][sl,c][zoom];old=a['legacy_target'][sl,c][zoom];new=a['physics_wiener'][sl,c][zoom]
            limit=max(np.max(abs(v)) for v in (raw,old,new,raw-old,raw-new))*1.06
            for value,label,color in [(raw,'原始 EEG',colors['raw']),(old,'当前 S',colors['legacy_target']),(new,'候选 S',colors['physics_wiener'])]:
                axes[i,0].plot(times[zoom],value,color=color,lw=1.,label=label)
            axes[i,1].plot(times[zoom],raw-old,color=colors['legacy_target'],lw=1.,label='当前 O')
            axes[i,1].plot(times[zoom],raw-new,color=colors['physics_wiener'],lw=1.,label='候选 O')
            for ax in axes[i]:ax.set_ylim(-limit,limit);ax.set_ylabel(channels[c]+' · µV');axgrid(ax)
        axes[0,0].set_title('原电压与 S：固定两秒放大');axes[0,1].set_title('O：与同电极 S 图相同刻度')
        axes[0,0].legend(ncol=3);axes[0,1].legend(ncol=2)
        for ax in axes[-1]:ax.set_xlabel('评分片段内时间（秒）')
        add(f'样例 {number} · {subject}：放大看快速波形与残差',
            '所有样例固定放大片段第 9–11 秒，没有按候选方法效果挑选放大范围。原始输入、旧目标与新目标使用相同采样点和电压单位；曲线的峰谷、极性及偏移可直接比较。',
            fig,tag+'_voltage_zoom','本图只放大显示，前页数值仍来自完整 20 秒；未重新滤波、重新拟合或按放大区间归一化。O 中出现有结构的快速波形，不能仅凭其形状判定它是眼电或神经活动。',
            [f'evaluation/{key}/base.npz','natural_windows.csv'])

        with np.load(out/'evaluation'/key/'ocular.npz',allow_pickle=False) as ar:ocular=dict(ar)
        c=picked[0];view=(times>=6)&(times<=14);vtime=times[view]
        raw=a['x'][sl,c];s0=a['physics_wiener'][sl,c];o0=raw-s0
        fig,axes=plt.subplots(3,1,figsize=(13,6.7),sharex=True,constrained_layout=True)
        axes[0].plot(vtime,raw[view],color=colors['raw'],lw=.8,label='原始基础 EEG')
        axes[0].plot(vtime,ocular['x'][view,c],color='#b64f55',lw=.9,label='加入已知眼电后的输入')
        axes[0].plot(vtime,ocular['n'][view,c],color='#935d9c',lw=1.,ls='--',label='已知新增眼电')
        axes[1].plot(vtime,s0[view],color=colors['physics_wiener'],lw=1.2,label='基础 S 标签')
        axes[1].plot(vtime,ocular['paired_s'][view,c],color='#222222',lw=.8,ls='--',label='配对 S 标签（按定义相同）')
        axes[2].plot(vtime,o0[view],color=colors['raw'],lw=.8,label='基础 O 标签')
        axes[2].plot(vtime,ocular['paired_o'][view,c],color='#b64f55',lw=.9,label='配对 O 标签 = 原 O + 新增眼电')
        for ax in axes:ax.set_ylabel(channels[c]+' · µV');ax.legend(ncol=3,loc='upper right');axgrid(ax)
        axes[-1].set_xlabel('评分片段内时间（秒）')
        add(f'样例 {number} · {subject}：加噪训练标签怎样变化',
            '此处展示真正准备交给 S/O 的配对监督规则：S 与基础片段共享同一条电压目标，O 增加已知眼电。各行按自身物理量幅度显示，纵轴仍为 µV。仅展示评分片段第 6–14 秒，指标使用完整 20 秒。',
            fig,tag+'_paired','S 完全不变是标签构造恒等式，不是网络已经学会去噪。基础 S 中若含天然伪迹，配对标签仍会保留它。没有在加噪后重跑 SSM 来修改这份配对 S 标签。',
            [f'evaluation/{key}/ocular.npz',f'evaluation/{key}/metrics.json'])

        fig,axes=plt.subplots(3,2,figsize=(13,6.8),sharex=True,constrained_layout=True)
        n=ocular['n'][view,c]
        for i,method in enumerate(['legacy_target','wiener','physics_wiener']):
            delta=ocular[method][view,c]-a[method][sl,c][view]
            axes[i,0].plot(vtime,n,color='#888888',ls=':',lw=1.,label='新增眼电（尺度参照）')
            axes[i,0].plot(vtime,delta,color=colors[method],lw=1.,label='估计器重算后的 ΔS')
            axes[i,1].plot(vtime,n,color='black',ls='--',lw=1.,label='期望 ΔO = 已知眼电')
            axes[i,1].plot(vtime,n-delta,color=colors[method],lw=1.,label='实际 ΔO')
            lim=max(np.max(abs(n)),np.max(abs(delta)),np.max(abs(n-delta)))*1.05
            for ax in axes[i]:ax.set_ylim(-lim,lim);ax.set_ylabel(names[method]+'\nµV');axgrid(ax)
        axes[0,0].set_title('ΔS：越接近零越抗新增眼电');axes[0,1].set_title('ΔO：应重现已知新增眼电')
        axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
        for ax in axes[-1]:ax.set_xlabel('评分片段内时间（秒）')
        add(f'样例 {number} · {subject}：重算估计器是否真的抗眼电',
            '对污染输入重算功率、噪声统计及电压目标，再减去基础估计。物理预测和训练校准冻结。完整 20 秒、全电极的变化误差（/注入 RMS）：'+
            '；'.join(f"{names[k]} {np.sqrt(next(r['delta_error_NMSE'] for r in case_metrics['perturbations'] if r['method']==k and r['kind']=='ocular' and r['amplitude']==1.)):.3f}" for k in ['legacy_target','wiener','physics_wiener'])+'。',
            fig,tag+'_reestimate','有色曲线贴近左图零线且贴近右图已知眼电，才表示该估计器将新增干扰分给 O。不能用上一页按定义重合的 S 标签替代这项核验。该实验仍不包含污染后重新拟合 Tak 的响应，也不测 tokenizer 泛化。',
            [f'evaluation/{key}/ocular.npz',f'evaluation/{key}/base.npz','intervention_windows.csv'])

    example=summary['examples'][1];key=example['id'];a=read_case(key)
    # The neural intervention has a central cosine topography: its endpoint
    # electrodes receive zero injection. Show a genuinely injected central
    # channel, while the metric continues to score every electrode.
    channels=[str(value) for value in a['channels']]
    c=channels.index('Cz') if 'Cz' in channels else len(channels)//2
    times=np.arange(4000)/200;view=(times>=6)&(times<=14)
    fig,axes=plt.subplots(2,2,figsize=(13,6.3),sharex=True,constrained_layout=True)
    for i,kind in enumerate(['unreferenced_muscle','neural_alpha']):
        with np.load(out/'evaluation'/key/f'{kind}.npz',allow_pickle=False) as ar:b=dict(ar)
        n=b['n'][:,c];expected=n if kind=='neural_alpha' else np.zeros_like(n)
        for j,method in enumerate(['wiener','physics_wiener']):
            delta=b[method][:,c]-a[method][6000:10000,c]
            axes[i,j].plot(times[view],n[view],color='#999999',ls=':',lw=.9,label='已知干预')
            axes[i,j].plot(times[view],expected[view],color='black',ls='--',lw=1.1,label='期望 ΔS')
            axes[i,j].plot(times[view],delta[view],color=colors[method],lw=.9,label='实际 ΔS')
            axes[i,j].set_ylabel(kindnames[kind]+'\nµV');axes[i,j].set_title(names[method]);axes[i,j].legend(fontsize=8);axgrid(axes[i,j])
    for ax in axes[-1]:ax.set_xlabel('评分片段内时间（秒）')
    add('负对照与保留性：没有证据的干扰，不能假装已分离',
        f"显示中位 EOG 样例 {example['subject']} 的 {str(a['channels'][c])} 电极。上排是没有 EOG 参考的高频干扰；下排是与 Hb 保持不变的神经样 α 变化。原始背景和基准物理预测相同。",
        fig,'09_negative_and_neural','上排期望 ΔS=0；若曲线仍复制已知干扰，则噪声模型没有覆盖它。下排期望 ΔS=已知神经样变化；削弱它是损伤风险，不能以“更符合 Hb/SSM”解释为成功。',
        [f'evaluation/{key}/unreferenced_muscle.npz',f'evaluation/{key}/neural_alpha.npz','intervention_windows.csv'])

    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    expected=[66,18];fail=[counts['joint_physical_failures'],counts['calibration_failures']]
    axes[0].bar([0,1],[expected[i]-fail[i] for i in range(2)],label='完成并通过积分/收敛检查',color='#168368')
    axes[0].bar([0,1],fail,bottom=[expected[i]-fail[i] for i in range(2)],label='失败 / 不可用',color='#b64f55')
    axes[0].set_xticks([0,1],['逐窗联合 Tak（22×3）','训练 Hb-only 校准（6×3）']);axes[0].set_ylabel('拟合数（不是独立被试）');axes[0].legend(fontsize=9)
    axes[1].axis('off');p=summary['paired_labels']
    axes[1].text(0,.95,f"配对 S 最大变化：{p['maximum_S_change']:.3g} µV\n"
        f"配对 O 增量误差：{p['maximum_O_change_error']:.3g} µV\n"
        f"配对闭合最大误差：{p['maximum_closure']:.3g} µV\n\n"
        '数值恒等式通过 ≠ 分离真值通过\n物理失败明确使用无先验回退\n失败、原始分母和回退标记均保留\n未训练 tokenizer，也未检验离散解码',va='top',fontsize=13,linespacing=1.9)
    add('完整分母、数值合同和未解决问题一起交付',
        f"计划 {counts['planned_subjects']} 名被试、{counts['planned_windows']} 个窗口；完成电压评价 {counts['evaluated_windows']} 个。物理拟合失败 {counts['physical_failures']} 个，含联合拟合 {counts['joint_physical_failures']}、校准 {counts['calibration_failures']}；{counts['windows_with_physics_abstention']} 个窗口含物理回退。",
        fig,'10_integrity','初次执行在记录 NumPy 标量时中断；新版本只修复记录接口并复用完成输入。闭合数值使用 float64，保存的显示波形为 float32，另行独立重算核验。所有结果为公开开发面板探索；未读测试或受保护数组。',
        ['plan.json','physical_fits.json','summary.json','input_reuse.json'])

    subject_table=pd.read_csv(out/'intervention_subjects.csv')
    table=subject_table[(subject_table.kind=='ocular')&(subject_table.amplitude==1.)].pivot(index='subject',columns='method',values='delta_error_NMSE')
    fig,axes=plt.subplots(1,2,figsize=(13,5.3),constrained_layout=True)
    palette=['#0072b2','#d55e00','#009e73','#cc79a7','#5c6670'];markers=['o','s','^','D','P']
    for ax,baseline,title in zip(axes,['legacy_target','wiener'],['候选目标与当前旧目标','物理项与同构无物理版本']):
        upper=float(np.sqrt(table[[baseline,'physics_wiener']]).to_numpy().max()*1.12)
        ax.plot([0,upper],[0,upper],color='#777777',ls='--',label='两者相同')
        for i,(subject,row) in enumerate(table.iterrows()):
            ax.scatter(np.sqrt(row[baseline]),np.sqrt(row['physics_wiener']),s=85,color=palette[i],marker=markers[i],label=subject)
        ax.set_xlim(0,upper);ax.set_ylim(0,upper);ax.set_aspect('equal');ax.set_xlabel(names[baseline]+'：眼电变化误差');ax.set_ylabel('物理 Wiener：眼电变化误差');ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=9)
    old=metric('legacy_target','ocular');reg=metric('eog_regression','ocular');neural=metric('physics_wiener','neural_alpha');oldneural=metric('legacy_target','neural_alpha')
    improvement=100*(1-m['delta_error_NRMSE']/old['delta_error_NRMSE'])
    add('结论：电压目标有改进，但物理收益尚未得到支持',
        f"1 倍眼电下，候选误差 {m['delta_error_NRMSE']:.4f}，旧目标 {old['delta_error_NRMSE']:.4f}，降低 {improvement:.1f}%；无物理版 {w['delta_error_NRMSE']:.4f}，EOG 回归 {reg['delta_error_NRMSE']:.4f}。神经样变化误差由旧目标 {oldneural['delta_error_NRMSE']:.4f} 降至 {neural['delta_error_NRMSE']:.4f}；无参考高频干扰约为 1，基本未分离。",
        fig,'11_conclusions','每点是一名被试：先在被试内汇总两窗平方误差再开方；点位于左图对角线下方表示候选优于旧目标。下一步应先验证天然伪迹与脑信号损伤、补足噪声证据，再用匹配信息的先验对照测试物理增量；不据此直接替换正式目标。',
        ['summary.json','intervention_subjects.csv','physics_paired_comparisons.csv'])

    # Complete native-channel gallery, with explicit per-electrode display scales.
    entries=json.loads((out/'plan.json').read_text())['rows']
    for entry in entries:
        path=out/'evaluation'/entry['id']/'base.npz'
        if not path.exists():continue
        a=read_case(entry['id']);channels=[str(c) for c in a['channels']];n=len(channels)
        fig,axes=plt.subplots(int(np.ceil(n/4)),4,figsize=(14,9),sharex=True,constrained_layout=True)
        for c,ax in enumerate(axes.flat):
            if c>=n:ax.axis('off');continue
            x=a['x'][6000:10000,c];s=a['physics_wiener'][6000:10000,c]
            ax.plot(times,x,color=colors['raw'],lw=.5,label='原始 EEG')
            ax.plot(times,s,color=colors['physics_wiener'],lw=.6,label='候选 S')
            ax.plot(times,x-s,color=colors['physics_wiener'],ls='--',lw=.6,label='候选 O')
            ax.set_title(channels[c]);ax.set_ylabel('µV');ax.set_xlabel('秒');axgrid(ax)
        axes.flat[0].legend(fontsize=8,ncol=3,loc='upper left')
        filename=save(fig,'gallery_'+entry['id'])
        gallery.append(dict(title=f"{entry['subject']} · {entry['split']} · 窗口 {entry['slot']+1}",
            intro=f"{entry['id']}；评分范围原窗口 30–50 秒；完整 {n} 电极，每个电极独立纵轴，均保留 µV。",
            figure=filename,caption='灰：原始 EEG；绿实线：候选 S；绿虚线：候选 O。独立纵轴仅用于逐电极阅读，跨电极幅度必须读取坐标，不以视觉高度比较。',evidence=[f'evaluation/{entry["id"]}/base.npz']))

    from fontTools import ttLib, subset
    report_font=ttLib.TTFont(font,fontNumber=2)
    options=subset.Options();options.retain_gids=True
    subsetter=subset.Subsetter(options=options)
    subsetter.populate(text=''.join(p[k] for p in pages+gallery for k in ('title','intro','caption')))
    subsetter.subset(report_font);report_font_path=export/'report_font.otf';report_font.save(report_font_path)
    def pdf_export(destination,records):
        doc=fitz.open()
        for i,p in enumerate(records):
            page=doc.new_page(width=1000,height=700)
            page.insert_font(fontname='ReportNoto',fontfile=str(report_font_path))
            for rect,text,size in [(fitz.Rect(34,22,966,73),p['title'],22),
                    (fitz.Rect(34,80,966,154),p['intro'],12),
                    (fitz.Rect(34,616,966,674),p['caption'],11)]:
                remaining=page.insert_textbox(rect,text,fontname='ReportNoto',fontsize=size,lineheight=1.45)
                if remaining<0:raise ValueError(f'PDF text overflow on page {i+1}: {remaining}')
            page.insert_image(fitz.Rect(34,162,966,607),filename=str(export/p['figure']),keep_proportion=True)
            page.insert_text((35,680),'Evidence: '+', '.join(Path(s).name for s in p['evidence']),fontsize=7,color=(.35,.35,.35))
            page.insert_text((35,691),f'{i+1} / {len(records)}  |  TK-EEG-VOLTAGE-TEACHER-v1  |  2026-11-05',fontsize=8,color=(.4,.4,.4))
        doc.save(destination,deflate=True,garbage=4);doc.close()
    pdf_export(export/'EEG_SO_监督目标可行性报告.pdf',pages)
    pdf_export(export/'全部窗口电压图集.pdf',gallery)
    style='body{font:16px/1.7 system-ui,sans-serif;background:#edf1f5;color:#172739;margin:0}main{max-width:1280px;margin:auto}section{background:white;margin:24px 0;padding:28px;border-radius:10px}img{width:100%;height:auto}h1,h2{line-height:1.35}p{max-width:1100px}small{color:#536879}a{color:#1769a4}'
    blocks=[]
    for p in pages+gallery:
        sources=' · '.join(f'<a href="../{html.escape(s)}">{html.escape(s)}</a>' for s in p['evidence'])
        blocks.append(f'<section><h2>{html.escape(p["title"])}</h2><p>{html.escape(p["intro"])}</p><img src="{p["figure"]}" alt="{html.escape(p["title"])}"><p>{html.escape(p["caption"])}</p><small>数值来源：{sources}</small></section>')
    (export/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EEG S/O 监督目标实验</title><style>'+style+'</style><main><section><h1>EEG S/O 监督目标可行性实验</h1><p>静态详细报告与全部窗口图集。以运行目录 summary.json 和逐窗数值为事实源。</p><p><a href="EEG_SO_监督目标可行性报告.pdf">报告 PDF</a> · <a href="全部窗口电压图集.pdf">全部窗口图集 PDF</a></p></section>'+''.join(blocks)+'</main></html>',encoding='utf-8')
    _json(export/'figure_formulas.json',formulas)
    manifest=dict(source_summary=str(out/'summary.json'),report_pages=pages,gallery=gallery,
        bitmap_dpi=240,body_text='searchable_pdf_text',source='retained_arrays_no_new_fits',
        external_method_sources=['https://github.com/exporl/mwf-artifact-removal','https://proceedings.mlr.press/v80/lehtinen18a.html'])
    _json(export/'figure_index.json',manifest)
    return dict(status='completed',export=str(export),report_pages=len(pages),gallery_pages=len(gallery))


def _json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, default=convert, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def _csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(''); return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)


def _cache(run):
    cache = Path(run) / 'cache'
    manifest = json.loads((cache / 'manifest.json').read_text())
    arrays = {k: np.load(cache / manifest.get('arrays', {}).get(k, k + '.npy'), mmap_mode='r')
              for m in MODALITIES for k in (m + '_u', m + '_s', m + '_e')}
    if any(len(x) != len(manifest['rows']) for x in arrays.values()):
        raise ValueError('cache row identity/array shape mismatch')
    if any(not np.allclose(arrays[m + '_s'] + arrays[m + '_e'], arrays[m + '_u'], atol=2e-5)
           for m in MODALITIES):
        raise ValueError('cached teacher decomposition identity failed')
    return cache, manifest, arrays


def _region(row):
    return str(row.get('roi', row.get('region', row.get('anchor', {}).get('region', ''))))


def normalized_roughness(x, channels=1):
    """Per-channel mean squared 100 ms difference / variance; patch-local."""
    x = np.asarray(x, float).reshape(*np.asarray(x).shape[:-1], -1, channels)
    axes = tuple(range(x.ndim - 1))
    return np.mean(np.diff(x, axis=-2) ** 2, axis=axes) / np.maximum(np.var(x, axis=axes), 1e-12)


def fit_matched_smoothing(observed, physical, training, sigmas=(0., 1., 2., 4., 8.), *, channels=1):
    """Choose each channel's smoothing width using training roughness alone."""
    observed, physical = np.asarray(observed), np.asarray(physical)
    training = np.asarray(training, bool)
    if observed.shape != physical.shape or len(training) != len(observed) or not training.any():
        raise ValueError('same-support observations/teacher and training mask required')
    sigmas = tuple(float(s) for s in sigmas)
    if not sigmas or min(sigmas) < 0:
        raise ValueError('nonnegative candidate Gaussian widths required')
    desired = normalized_roughness(physical[training], channels)
    reshaped = observed.reshape(*observed.shape[:-1], -1, channels)
    candidates = [reshaped if s == 0 else gaussian_filter1d(reshaped, s, axis=-2, mode='nearest')
                  for s in sigmas]
    roughness = np.stack([normalized_roughness(x[training].reshape(observed[training].shape), channels)
                          for x in candidates])
    mismatch = np.abs(np.log(np.maximum(roughness, 1e-12)) - np.log(np.maximum(desired, 1e-12)))
    selected = np.argmin(mismatch, axis=0)
    result = np.stack([candidates[selected[c]][..., c] for c in range(channels)], axis=-1)
    report = dict(sigmas_samples=[sigmas[j] for j in selected],
        sigmas_seconds=[.1 * sigmas[j] for j in selected], candidate_sigmas_samples=sigmas,
        teacher_training_roughness=desired, selected_training_roughness=roughness[selected, np.arange(channels)],
        selection='train-only per-channel absolute log normalized-roughness mismatch',
        support='within each 2-second patch; nearest endpoint; no cross-patch smoothing',
        fit_rows=int(training.sum()), eval_used_for_selection=False)
    return result.reshape(observed.shape).astype('float32'), report


def fit_smooth_targets(cfg, cache_manifest, cache_dir):
    """Return a manifest for fixed matched-smoothing S and its full residual."""
    cache = Path(cache_dir); rows = cache_manifest['rows']
    train = np.array([r['split'] == 'train' for r in rows])
    smoothing = cfg.get('smoothing', cfg.get('analysis', {}).get('smoothing', {}))
    sigmas = smoothing.get('sigma_steps', smoothing.get('sigmas_samples', smoothing.get('sigmas', [0., 1., 2., 4., 8.])))
    result = dict(schema='dual_component_matched_smoothing_v1', arrays={}, modalities={})
    for modality in MODALITIES:
        def read(part):
            key = modality + '_' + part
            return np.load(cache / cache_manifest.get('arrays', {}).get(key, key + '.npy'), mmap_mode='r')
        u, s = read('u'), read('s')
        smooth, report = fit_matched_smoothing(u, s, train, sigmas, channels=2 if modality == 'hb' else 1)
        for name, data in (('s', smooth), ('e', np.asarray(u) - smooth)):
            filename = 'smooth_' + modality + '_' + name + '.npy'
            path = cache / filename
            if path.exists():
                existing = np.load(path, mmap_mode='r')
                if existing.shape != data.shape or not np.array_equal(existing, data):
                    raise ValueError('retained smoothing target changed; use a fresh run namespace')
            else:
                np.save(path, data.astype('float32'))
            result['arrays'][modality + '_' + name] = filename
        result['modalities'][modality] = report
    _json(cache / 'smoothing.json', result)
    return result


def reconstruction_rows(predictions, physical, remaining, observed, rows, *, variant, seed, target_family='Tak_teacher'):
    """Window scores; all three denominators are observation-training SD squared."""
    result = []
    for m in MODALITIES:
        targets = (dict(sum=observed[m]) if variant == 'ordinary_hard'
                   else dict(S=physical[m], O=remaining[m], sum=observed[m]))
        for branch, target in targets.items():
            value = np.asarray(predictions[m + '_' + branch], float)
            target = np.asarray(target, float)
            if value.shape != target.shape or len(value) != len(rows):
                raise ValueError('predictions/targets/identities must share complete support')
            selectors = [('EEG_proxy' if m == 'eeg' else 'Hb_joint', slice(None))]
            if m == 'hb': selectors.extend([('HbO', slice(0, None, 2)), ('HbR', slice(1, None, 2))])
            for signal, selector in selectors:
                difference = value[..., selector] - target[..., selector]
                finite = np.isfinite(difference).all(axis=tuple(range(1, value.ndim)))
                errors = np.mean(difference ** 2, axis=tuple(range(1, value.ndim)))
                for i, row in enumerate(rows):
                    result.append(dict(dataset=row.get('dataset', ''), subject=row['subject'],
                        record=row.get('record', row.get('join_key', '')), split=row['split'],
                        variant=variant, seed=int(seed), modality=m, component=branch, signal=signal,
                        target_family='processed_observation' if branch == 'sum' else target_family,
                        NMSE=float(errors[i]) if finite[i] else None,
                        windows=1, failures=int(not finite[i]),
                        branch_semantics=('arbitrary_additive_streams' if variant == 'ordinary_hard'
                                          else 'matched_smoothing' if variant == 'smooth_hard' else 'Tak_teacher')))
    return result


def subject_aggregate(rows, *, value='NMSE'):
    """Windows -> equal records -> subject; retain failures and observed counts."""
    excluded = {'subject', 'record', value, 'windows', 'failures'}
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple((k, str(v)) for k, v in row.items() if k not in excluded)
        groups[(key, str(row['subject']))][str(row.get('record', ''))].append(row)
    result = []
    for (key, subject), records in groups.items():
        means = []
        total, failures = 0, 0
        for record_rows in records.values():
            values = [float(r[value]) for r in record_rows if r.get(value) is not None and np.isfinite(r[value])]
            total += len(record_rows)
            failures += sum(int(r.get('failures', r.get(value) is None)) for r in record_rows)
            if values: means.append(float(np.mean(values)))
        result.append(dict(key, subject=subject, **{value:float(np.mean(means)) if means else None},
            records=len(records), windows=total, failures=failures))
    return result


def paired_subject_bootstrap(first, second, *, value='NMSE', samples=2000, seed=0):
    """Average seeds within subjects, then paired subject bootstrap; first-second."""
    def reduce(rows):
        values = defaultdict(list)
        for row in rows:
            x = row.get(value)
            if x is not None and np.isfinite(x):
                values[(str(row.get('dataset', '')), str(row['subject']))].append(float(x))
        return {key:float(np.mean(x)) for key, x in values.items()}
    a, b = reduce(first), reduce(second)
    common = sorted(a.keys() & b.keys())
    difference = np.array([a[s] - b[s] for s in common])
    result = dict(subjects=len(common), first_subjects=len(a), second_subjects=len(b),
        unmatched_first=len(a.keys() - b.keys()), unmatched_second=len(b.keys() - a.keys()),
        effect='first minus second; lower NMSE favors first',
        mean_difference=float(difference.mean()) if len(difference) else None,
        interval_type='percentile paired-subject bootstrap, seeds averaged within subject',
        bootstrap_samples=int(samples), CI95=None,
        inference='exploratory; no multiplicity correction; windows/seeds are not independent subjects')
    if len(common) >= 2 and samples >= 1:
        rng = np.random.default_rng(seed)
        draws = difference[rng.integers(len(common), size=(samples, len(common)))].mean(axis=1)
        result['CI95'] = np.quantile(draws, [.025, .975]).tolist()
    return result


def latent_codebook_health(ids, codebook, rows, thresholds):
    """Full-population, per-dataset usage; geometry is latent, never waveform."""
    ids, book = np.asarray(ids), np.asarray(codebook, dtype=float)
    if (ids.ndim != 2 or ids.dtype.kind not in 'iu' or len(ids) != len(rows)
            or book.ndim != 2 or not np.isfinite(book).all()
            or np.any(ids < 0) or np.any(ids >= len(book))
            or any(r['split'] not in ('train', 'validation') for r in rows)):
        raise ValueError('finite latent codebook and complete development integer assignments required')
    centered = book - book.mean(0)
    eigen = np.linalg.eigvalsh(centered.T @ centered / max(len(book), 1)).clip(0)
    rank = float(eigen.sum() ** 2 / max(float(np.square(eigen).sum()), 1e-30))
    distances = np.maximum(np.square(book).sum(1)[:, None] + np.square(book).sum(1)[None] - 2 * book @ book.T, 0)
    np.fill_diagonal(distances, np.inf)
    nearest = np.sqrt(distances.min(1))
    scale = max(float(np.sqrt(np.mean(np.square(centered).sum(1)))), 1e-12)
    geometry = dict(codebook_participation_rank=rank, dimension=book.shape[1],
        nearest_distance_median_over_codebook_RMS=float(np.median(nearest) / scale),
        near_duplicate_fraction=float(np.mean(nearest / scale < .01)),
        geometry='centered latent covariance participation rank; distances are not decoded waveform distances')
    summaries, coverage = [], []
    for dataset in ['pooled', *sorted({r['dataset'] for r in rows})]:
        for split in ('train', 'validation'):
            selected = np.array([r['split'] == split and (dataset == 'pooled' or r['dataset'] == dataset) for r in rows])
            if not selected.any():
                continue
            selected_ids = ids[selected]
            counts = np.bincount(selected_ids.ravel(), minlength=len(book))
            probability = counts / counts.sum()
            entropy = -float(np.sum(probability[probability > 0] * np.log(probability[probability > 0])))
            active, effective = int(np.sum(counts > 0)), float(np.exp(entropy))
            active_floor = thresholds['train_active_fraction' if split == 'train' else 'validation_active_fraction']
            checks = dict(active=active / len(book) >= active_floor,
                effective=effective / len(book) >= thresholds['effective_fraction'],
                concentration=float(probability.max()) <= thresholds['maximum_code_probability'])
            summaries.append(dict(dataset=dataset, split=split, patches=int(counts.sum()),
                windows=int(selected.sum()), subjects=len({(r['dataset'], r['subject']) for i, r in enumerate(rows) if selected[i]}),
                active_codes=active, codebook_size=len(book), active_fraction=active / len(book),
                effective_codes=effective, effective_fraction=effective / len(book),
                maximum_code_probability=float(probability.max()), checks=checks, passed=all(checks.values())))
            selected_rows = [r for i, r in enumerate(rows) if selected[i]]
            for k in range(len(book)):
                member = np.flatnonzero(np.any(selected_ids == k, axis=1))
                coverage.append(dict(dataset=dataset, split=split, code=k, assignments=int(counts[k]),
                    probability=float(probability[k]), windows=len(member),
                    subjects=len({(selected_rows[i]['dataset'], selected_rows[i]['subject']) for i in member}),
                    records=len({(selected_rows[i]['dataset'], selected_rows[i]['subject'], selected_rows[i]['record']) for i in member})))
    return dict(passed=bool(summaries) and all(r['passed'] for r in summaries), summaries=summaries,
                geometry=geometry, coverage=coverage)


def latent_hb_fidelity(predictions, targets, rows):
    """Separate Hb colors and branches using the retained observation-SD units."""
    window_metrics = []
    for branch in ('S', 'O', 'sum'):
        truth = np.asarray(targets['u' if branch == 'sum' else branch], dtype=float)
        estimate = np.asarray(predictions[branch], dtype=float)
        if (truth.shape != estimate.shape or truth.shape != (len(rows), 15, 40)
                or not np.isfinite(truth).all() or not np.isfinite(estimate).all()):
            raise ValueError('complete aligned [window,15,40] Hb targets and predictions required')
        a, b = estimate.reshape(-1, 300, 2), truth.reshape(-1, 300, 2)
        for signal, selector in [('HbO', [0]), ('HbR', [1]), ('joint', [0, 1])]:
            aa, bb = a[:, :, selector], b[:, :, selector]
            ac, bc = aa - aa.mean(1, keepdims=True), bb - bb.mean(1, keepdims=True)
            denominator = np.sqrt(np.square(ac).sum((1, 2)) * np.square(bc).sum((1, 2)))
            values = dict(NMSE=np.square(aa - bb).mean((1, 2)),
                zero_NMSE=np.square(bb).mean((1, 2)),
                centered_correlation=np.divide((ac * bc).sum((1, 2)), denominator,
                    out=np.full(len(rows), np.nan), where=denominator > 1e-12))
            for metric, array in values.items():
                for i, row in enumerate(rows):
                    window_metrics.append(dict(dataset=row['dataset'], split=row['split'], subject=row['subject'],
                        record=row['record'], branch=branch, signal=signal, metric=metric,
                        value=float(array[i]) if np.isfinite(array[i]) else None, failures=int(not np.isfinite(array[i]))))
    subjects = subject_aggregate(window_metrics, value='value')
    grouped = defaultdict(list)
    for row in subjects:
        grouped[tuple(row[k] for k in ('dataset', 'split', 'branch', 'signal', 'metric'))].append(row)
    summary = []
    for key, group in grouped.items():
        values = [r['value'] for r in group if r['value'] is not None]
        summary.append(dict(zip(('dataset', 'split', 'branch', 'signal', 'metric'), key),
            value=float(np.mean(values)) if values else None, subjects=len(values),
            undefined_subjects=len(group) - len(values), undefined_windows=sum(r['failures'] for r in group)))
    return dict(metrics=summary, subjects=subjects,
        units='MSE after division by frozen training observation SD; no window normalization',
        aggregation='time/color then equal windows within record then equal records within subject then equal subjects')


def codebook_health(ids, prototypes, rows, *, modality, branch, variant, seed, observed=None):
    """Split-specific occupancy and independent subject/record coverage per code."""
    ids, prototypes = np.asarray(ids), np.asarray(prototypes)
    if ids.ndim != 2 or prototypes.ndim != 2 or len(ids) != len(rows):
        raise ValueError('codebook health requires [window,time] IDs and [code,waveform] prototypes')
    if ids.dtype.kind not in 'iu': raise ValueError('hard integer IDs required')
    result = []
    for split in ('train', 'validation', 'test'):
        selected = np.array([r['split'] == split for r in rows])
        split_ids = ids[selected]
        valid = split_ids[(split_ids >= 0) & (split_ids < len(prototypes))]
        counts = np.bincount(valid, minlength=len(prototypes))
        p = counts / max(int(counts.sum()), 1)
        entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
        observed_rms = np.sqrt(np.mean(np.asarray(observed, float) ** 2, axis=-1)) if observed is not None else None
        for k, prototype in enumerate(prototypes):
            assignment_mask = selected[:, None] & (ids == k)
            amplitudes = observed_rms[assignment_mask] if observed_rms is not None else np.array([])
            assigned = np.flatnonzero(selected & np.any(ids == k, axis=1))
            result.append(dict(variant=variant, seed=seed, modality=modality, component=branch,
                split=split, code=k, assignments=int(counts[k]), probability=float(p[k]),
                assigned_windows=len(assigned), subjects=len({(rows[i].get('dataset', ''), rows[i]['subject']) for i in assigned}),
                records=len({(rows[i].get('dataset', ''), rows[i]['subject'], rows[i].get('record', rows[i].get('join_key', ''))) for i in assigned}),
                assigned_observation_RMS_P10=float(np.quantile(amplitudes, .1)) if len(amplitudes) else None,
                assigned_observation_RMS_P50=float(np.quantile(amplitudes, .5)) if len(amplitudes) else None,
                assigned_observation_RMS_P90=float(np.quantile(amplitudes, .9)) if len(amplitudes) else None,
                prototype_mean=float(prototype.mean()), prototype_RMS=float(np.sqrt(np.mean(prototype ** 2))),
                prototype_min=float(prototype.min()), prototype_max=float(prototype.max()),
                active_codes=int((counts > 0).sum()), perplexity=float(np.exp(entropy)),
                dead_fraction=float(np.mean(counts == 0))))
    return result


def align_codewords(reference, candidate):
    """Training-frozen waveform correspondence, not equality of arbitrary IDs."""
    a, b = np.asarray(reference, float), np.asarray(candidate, float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError('same code count and decoded waveform dimensions required')
    cost = np.mean((a[:, None] - b[None]) ** 2, axis=-1)
    i, j = linear_sum_assignment(cost)
    mapping = np.empty(len(b), dtype=int); mapping[j] = i
    return mapping, dict(mean_prototype_NMSE=float(cost[i, j].mean()),
                        P90_prototype_NMSE=float(np.quantile(cost[i, j], .9)))


def _job(cfg, run, variant, seed):
    return Path(run) / 'jobs' / variant / ('seed_' + str(seed))


def _subject_plot(path, rows):
    if not rows: return
    import matplotlib; matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    selected = [r for r in rows if r['split'] == 'test' and r['component'] == 'sum' and r['NMSE'] is not None and r.get('signal') in ('EEG_proxy', 'Hb_joint')]
    variants = list(dict.fromkeys(r['variant'] for r in selected))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, m in zip(axes, MODALITIES):
        for j, variant in enumerate(variants):
            values = [r['NMSE'] for r in selected if r['variant'] == variant and r['modality'] == m]
            if values: ax.scatter(np.full(len(values), j), values, s=14, alpha=.45)
        ax.set(xticks=np.arange(len(variants)), xticklabels=variants,
               ylabel='Observation MSE / frozen training SD²', title=m + ': subject-equal record means')
        ax.tick_params(axis='x', labelrotation=35); ax.grid(alpha=.2)
    fig.savefig(path, dpi=180); plt.close(fig)


def analyze_suite(cfg, run):
    """Read only this run's explicit completed job exports; write shared tables."""
    run = Path(run); cache, manifest, arrays = _cache(run); rows = manifest['rows']
    settings = cfg.get('analysis', {})
    variants = cfg.get('variants', cfg.get('training', {}).get('variants', ['continuous_student', 'hard_student',
                'component_compression', 'smooth_hard', 'ordinary_hard']))
    seeds = cfg.get('seeds', cfg.get('training', {}).get('seeds', [17, 29, 43]))
    physical = {m:arrays[m + '_s'] for m in MODALITIES}
    remaining = {m:arrays[m + '_e'] for m in MODALITIES}
    observed = {m:arrays[m + '_u'] for m in MODALITIES}
    subjects, health, alignment, missing = [], [], [], []
    reference = {}
    for variant in variants:
        for seed in seeds:
            job = _job(cfg, run, variant, seed)
            record = json.loads((job / 'summary.json').read_text()) if (job / 'summary.json').exists() else {}
            if (record.get('status') != 'completed' or record.get('step') != int(cfg.get('training', {}).get('steps', 3000))
                    or not (job / 'predictions.npz').exists()):
                missing.append(dict(variant=variant, seed=seed, reason='completed_fixed_budget_job_and_prediction_export_required')); continue
            with np.load(job / 'predictions.npz', allow_pickle=False) as data:
                predictions = {key:data[key] for key in data.files}
            component_physical, component_remaining = physical, remaining
            target_family = 'Tak_teacher'
            if variant == 'smooth_hard':
                smoothing = json.loads((cache / 'smoothing.json').read_text())
                component_physical = {m:np.load(cache / smoothing['arrays'][m + '_s'], mmap_mode='r') for m in MODALITIES}
                component_remaining = {m:np.load(cache / smoothing['arrays'][m + '_e'], mmap_mode='r') for m in MODALITIES}
                if any(not np.allclose(component_physical[m] + component_remaining[m], observed[m], atol=2e-5) for m in MODALITIES):
                    raise ValueError('matched smoothing targets must sum to the same processed observation')
                target_family = 'matched_smoothing'
            windows = reconstruction_rows(predictions, component_physical, component_remaining, observed, rows,
                                           variant=variant, seed=seed, target_family=target_family)
            subjects.extend(subject_aggregate(windows))
            if (job / 'codewords.npz').exists():
                with np.load(job / 'codewords.npz', allow_pickle=False) as data:
                    codewords = {key:data[key] for key in data.files}
                for m in MODALITIES:
                    for branch in BRANCHES:
                        key = m + '_' + branch; ids_key = m + '_ids_' + branch
                        if key not in codewords or ids_key not in predictions: continue
                        health.extend(codebook_health(predictions[ids_key], codewords[key], rows,
                            modality=m, branch=branch, variant=variant, seed=seed, observed=observed[m]))
                        rkey = (variant, m, branch)
                        if rkey not in reference:
                            reference[rkey] = (seed, codewords[key], predictions[ids_key])
                        else:
                            ref_seed, ref_book, ref_ids = reference[rkey]
                            mapping, effect = align_codewords(ref_book, codewords[key])
                            current = predictions[ids_key]; valid = (current >= 0) & (ref_ids >= 0)
                            aligned = mapping[np.maximum(current, 0)]
                            for split in ('train', 'validation', 'test'):
                                selected = np.array([r['split'] == split for r in rows])[:, None] & valid
                                alignment.append(dict(variant=variant, modality=m, component=branch,
                                    reference_seed=ref_seed, seed=seed, split=split, **effect,
                                    aligned_agreement=float(np.mean(aligned[selected] == ref_ids[selected])) if selected.any() else None,
                                    matching='Hungarian distance of fixed decoded waveforms; no eval fitting'))
    # Simple fixed-output controls are fit on train only and evaluated everywhere.
    train = np.array([r['split'] == 'train' for r in rows])
    if train.any():
        for baseline in ('training_mean', 'zero'):
            predictions = {}
            for m in MODALITIES:
                for branch, values in (('S', physical[m]), ('O', remaining[m])):
                    center = np.mean(values[train], axis=(0, 1), keepdims=True) if baseline == 'training_mean' else np.zeros((1, 1, values.shape[-1]))
                    predictions[m + '_' + branch] = np.broadcast_to(center, values.shape)
                predictions[m + '_sum'] = predictions[m + '_S'] + predictions[m + '_O']
            subjects.extend(subject_aggregate(reconstruction_rows(predictions, physical, remaining, observed,
                rows, variant=baseline, seed=0)))
    comparisons = []
    for dataset in sorted({r.get('dataset', '') for r in rows}):
        for split in ('validation', 'test'):
            for m, signal in (('eeg', 'EEG_proxy'), ('hb', 'Hb_joint'), ('hb', 'HbO'), ('hb', 'HbR')):
                for branch in ('S', 'O', 'sum'):
                    def pick(variant):
                        return [r for r in subjects if r['variant'] == variant and r['split'] == split
                                and r['modality'] == m and r['component'] == branch and r['signal'] == signal
                                and r.get('dataset', '') == dataset]
                    for control in ('continuous_student', 'component_compression', 'smooth_hard', 'ordinary_hard', 'training_mean'):
                        if control in ('smooth_hard', 'ordinary_hard') and branch != 'sum': continue
                        effect = paired_subject_bootstrap(pick('hard_student'), pick(control),
                            samples=int(settings.get('bootstrap_samples', cfg.get('evaluation', {}).get('bootstrap_repeats', 1000))), seed=int(cfg.get('seed', 0)))
                        comparisons.append(dict(first='hard_student', second=control, dataset=dataset, split=split,
                            modality=m, component=branch, signal=signal,
                            target_family='processed_observation' if branch == 'sum' else 'Tak_teacher', **effect))
    output = run / 'analysis'; output.mkdir(exist_ok=True)
    _csv(output / 'reconstruction_subjects.csv', subjects)
    _csv(output / 'codebook_health.csv', health)
    _csv(output / 'seed_alignment.csv', alignment)
    _csv(output / 'paired_comparisons.csv', comparisons)
    _subject_plot(output / 'reconstruction_subjects.png', subjects)
    sensitivity = analyze_teacher_sensitivity(cfg, run)
    quality = vocabulary_quality_screen(subjects, cfg)
    masked = summarize_masked_pilots(cfg, run)
    summary = dict(schema='dual_component_analysis_v1', status='completed' if not missing else 'partial',
        missing_jobs=missing, reconstruction_subject_rows=len(subjects), codebook_rows=len(health),
        paired_comparisons=comparisons, teacher_sensitivity=sensitivity, quality_screen=quality, masked=masked,
        physical_compatibility=dict(status='not_evaluated',
            reason='No Tak-compatible whole-trajectory witness was evaluated for decoded masked predictions; 20-second patches retain the 90-second teacher reference coordinate'),
        planned_windows=manifest.get('planned_windows', len(rows)), retained_windows=len(rows),
        failed_trials=manifest.get('failed_trials', []),
        metric='mean squared error in shared frozen observation-training-SD coordinates; no per-window scaling',
        aggregation='equal windows within record; equal records within subject; seeds averaged before paired subject bootstrap',
        scope='processed-observation offline completion; full-record Hb preprocessing retained',
        cross_vocabulary_CE_ranking_allowed=False)
    _json(output / 'summary.json', summary)
    return summary


def _checkpoint(path, model, optimizer, step, rng, losses, signature):
    payload = dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
        step=step, numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
        losses=losses, signature=signature, cuda_rng=torch.cuda.get_rng_state(next(model.parameters()).device) if next(model.parameters()).is_cuda else None)
    tmp = path.with_suffix('.tmp'); torch.save(payload, tmp); tmp.replace(path)


def _fit_reader(ids, rows, config, output, *, device, seed, reader):
    """Fixed-exposure train-only fitting, resumable independently per reader."""
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = DualComponentMaskedModel({m:int(config['vocab_size']) for m in MODALITIES},
        patches=ids.shape[1], width=int(config.get('width', 128)), layers=int(config.get('layers', 2)),
        heads=int(config.get('heads', 4)), dropout=float(config.get('dropout', .1))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get('learning_rate', 3e-4)),
                                  weight_decay=float(config.get('weight_decay', 1e-4)))
    rng = np.random.default_rng(seed); start = 0; losses = []
    path = output / 'checkpoint.pt'
    signature = dict(reader=reader, seed=seed, config=config, id_shape=ids.shape,
        training_identities=[r.get('id', str(i)) for i, r in enumerate(rows) if r['split'] == 'train'])
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved.get('signature') != signature or saved['step'] > int(config.get('steps', 1000)):
            raise ValueError('masked checkpoint identity/config/reader/budget mismatch')
        model.load_state_dict(saved['model_state']); optimizer.load_state_dict(saved['optimizer_state'])
        start = int(saved['step']); rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu()); losses = saved.get('losses', [])
        if saved.get('cuda_rng') is not None and str(device).startswith('cuda'):
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), device)
    datasets = sorted({r.get('dataset', '') for r in rows if r['split'] == 'train'})
    groups = [np.array([i for i, r in enumerate(rows) if r['split'] == 'train' and r.get('dataset', '') == d]) for d in datasets]
    if not groups or any(not len(g) for g in groups): raise ValueError('train data missing')
    steps = int(config.get('steps', 1000)); batch_size = int(config.get('batch_size', 64))
    block = int(config.get('block_patches', 2)); checkpoint_every = int(config.get('checkpoint_every', 250))
    began = time.monotonic(); model.train()
    for step in range(start, steps):
        selected = rng.choice(groups[int(rng.integers(len(groups)))], batch_size, replace=True)
        batch = ids[selected]; target_modality = int(rng.integers(len(MODALITIES)))
        task = 'time' if rng.integers(2) == 0 else 'modality'
        visible, target = block_masks(batch >= 0, rng=rng, target_modality=target_modality,
                                      block_patches=block, task=task)
        if reader == 'own': visible = own_context(visible, target_modality)
        truth = torch.as_tensor(batch, dtype=torch.long, device=device)
        target = torch.as_tensor(target, device=device)
        logits = model(truth, torch.as_tensor(visible, device=device))
        branch_losses = []
        for c in range(2):
            keep = target[..., c]
            if keep.any(): branch_losses.append(F.cross_entropy(logits[..., c, :][keep], truth[..., c][keep]))
        if not branch_losses: raise ValueError('empty masked target support')
        loss = torch.stack(branch_losses).sum()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get('gradient_clip', 5.)))
        optimizer.step()
        if (step + 1) % 50 == 0 or step == 0:
            losses.append(dict(step=step + 1, loss=float(loss.detach()), task=task, target_modality=MODALITIES[target_modality]))
        if (step + 1) % checkpoint_every == 0 or step + 1 == steps:
            _checkpoint(path, model, optimizer, step + 1, rng, losses, signature)
            _json(output / 'progress.json', dict(step=step + 1, steps=steps, reader=reader,
                last_loss=float(loss.detach()), resumed_from=start, elapsed_s=time.monotonic() - began))
    _csv(output / 'training_loss.csv', losses)
    model.eval()
    return model


def _predict(model, ids, visible, *, device, batch_size=128):
    result = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            logits = model(torch.as_tensor(ids[start:start + batch_size], device=device, dtype=torch.long),
                           torch.as_tensor(visible[start:start + batch_size], device=device))
            result.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(result)


def matched_donor_tokens(ids, rows, decoded, *, source_modality, training_ids,
                         training_rows, seed=0):
    """Same dataset/ROI/time, training-amplitude-bin donor, different subject.

    Matching uses only source token IDs and fixed decoded vocabulary, never the
    target, teacher, task/condition labels or eval-fitted amplitude thresholds.
    Donors come from the same evaluation partition and preserve patch index.
    """
    rng = np.random.default_rng(seed); m = int(source_modality)
    def amplitude(x):
        waves = sum(decoded[(m, c)][np.maximum(x[:, :, m, c], 0)] for c in range(2))
        return np.sqrt(np.mean(waves ** 2, axis=-1))
    training_amp = amplitude(training_ids)
    cutoffs = np.quantile(training_amp, [1 / 3, 2 / 3])
    amplitude_bins = np.searchsorted(cutoffs, amplitude(ids))
    result = ids.copy(); available = np.zeros(ids.shape[:2], bool)
    pools = defaultdict(list)
    for j, row in enumerate(rows):
        for t in range(ids.shape[1]):
            if np.all(ids[j, t, m] >= 0):
                pools[(str(row.get('dataset', '')), str(row.get('split', '')), _region(row), t, int(amplitude_bins[j, t]))].append(j)
    for i, row in enumerate(rows):
        for t in range(ids.shape[1]):
            key = (str(row.get('dataset', '')), str(row.get('split', '')), _region(row), t, int(amplitude_bins[i, t]))
            pool = pools.get(key, [])
            donor = None
            # Usually succeeds immediately, while remaining bounded and exact.
            for _ in range(min(8, len(pool))):
                candidate = pool[int(rng.integers(len(pool)))]
                if rows[candidate]['subject'] != row['subject']:
                    donor = candidate; break
            if donor is None:
                candidates = [j for j in pool if rows[j]['subject'] != row['subject']]
                if candidates: donor = int(rng.choice(candidates))
            if donor is not None:
                result[i, t, m] = ids[donor, t, m]
                available[i, t] = True
    return result, available, dict(amplitude_cutoffs_training_SD=cutoffs,
        matching='dataset, ROI, patch phase and train-fixed source decoded-amplitude tertile; other subject',
        eligible_source_positions=int(available.sum()), planned_source_positions=int(available.size),
        missing_donor='source tokens hidden; target scores retained', task_labels_used=False)


def _metric_subjects(window_rows):
    """One row per subject and metric, preserving trial/record dependence."""
    return subject_aggregate(window_rows, value='value')


def _score_probabilities(probability, ids, target, rows, decoded, arrays, marginal, metadata,
                         *, route, seed, task, context, target_modality):
    m = target_modality; modality = MODALITIES[m]; result = []
    predicted = {c:expected_waveforms(probability[:, :, m, c], decoded[(m, c)]) for c in range(2)}
    oracle = {c:decoded[(m, c)][np.maximum(ids[:, :, m, c], 0)] for c in range(2)}
    branch_targets = {0:arrays[modality + '_s'], 1:arrays[modality + '_e']}
    def add(values, selected, component, metric):
        for i, row in enumerate(rows):
            keep = selected[i]
            if not keep.any(): continue
            value = values[i][keep]
            result.append(dict(dataset=row.get('dataset', ''), subject=row['subject'],
                record=row.get('record', row.get('join_key', '')), split=row['split'],
                variant=route, seed=seed, task=task, context=context, modality=modality,
                component=component, metric=metric,
                target_family=('processed_observation' if component == 'sum' else 'arbitrary_additive_streams' if route == 'ordinary_hard' else 'matched_smoothing' if route == 'smooth_hard' else 'Tak_teacher'),
                value=float(np.mean(value)) if np.isfinite(value).all() else None,
                failures=int(not np.isfinite(value).all())))
    for c, branch in enumerate(BRANCHES):
        selected = target[:, :, m, c]
        truth = np.maximum(ids[:, :, m, c], 0)[..., None]
        def ce(p): return -np.log(np.maximum(np.take_along_axis(p[:, :, m, c], truth, axis=-1)[..., 0], 1e-12))
        error, marginal_error, metadata_error = ce(probability), ce(marginal), ce(metadata)
        add(error, selected, branch, 'CE_nats')
        entropy = -np.sum(probability[:, :, m, c] * np.log(np.maximum(probability[:, :, m, c], 1e-12)), axis=-1)
        add(entropy, selected, branch, 'predictive_entropy_nats')
        add(marginal_error - error, selected, branch, 'DeltaCE_marginal_nats')
        add(metadata_error - error, selected, branch, 'DeltaCE_metadata_nats')
        add(np.mean((predicted[c] - oracle[c]) ** 2, axis=-1), selected, branch, 'model_to_tokenizer_NMSE')
        if route != 'ordinary_hard':
            add(np.mean((predicted[c] - branch_targets[c]) ** 2, axis=-1), selected, branch, 'NMSE')
            add(np.mean((oracle[c] - branch_targets[c]) ** 2, axis=-1), selected, branch, 'tokenizer_NMSE')
            if modality == 'hb':
                for channel, name in enumerate(('HbO', 'HbR')):
                    add(np.mean((predicted[c][..., channel::2] - branch_targets[c][..., channel::2]) ** 2, axis=-1), selected, branch, 'NMSE_' + name)
    selected = target[:, :, m].all(-1)
    summed = predicted[0] + predicted[1]; oracle_sum = oracle[0] + oracle[1]
    add(np.mean((summed - arrays[modality + '_u']) ** 2, axis=-1), selected, 'sum', 'NMSE')
    if modality == 'hb':
        for channel, name in enumerate(('HbO', 'HbR')):
            add(np.mean((summed[..., channel::2] - arrays[modality + '_u'][..., channel::2]) ** 2, axis=-1), selected, 'sum', 'NMSE_' + name)
    if route != 'ordinary_hard':
        channels = 2 if modality == 'hb' else 1
        assembled = np.where(target[:, :, m, 0, None], predicted[0], oracle[0])
        segments = assembled.reshape(*assembled.shape[:2], -1, channels)
        teacher_segments = branch_targets[0].reshape(segments.shape)
        signed_jumps = segments[:, 1:, 0] - segments[:, :-1, -1]
        teacher_signed_jumps = teacher_segments[:, 1:, 0] - teacher_segments[:, :-1, -1]
        jumps = np.mean(signed_jumps ** 2, axis=-1)
        true_jumps = np.mean(teacher_signed_jumps ** 2, axis=-1)
        jump_error = np.mean((signed_jumps - teacher_signed_jumps) ** 2, axis=-1)
        boundaries = target[:, 1:, m, 0] | target[:, :-1, m, 0]
        add(np.pad(jumps, ((0, 0), (1, 0))), np.pad(boundaries, ((0, 0), (1, 0))), 'S', 'boundary_jump_squared_trainingSD')
        add(np.pad(jumps - true_jumps, ((0, 0), (1, 0))), np.pad(boundaries, ((0, 0), (1, 0))), 'S', 'boundary_jump_excess_over_teacher_trainingSD')
        add(np.pad(jump_error, ((0, 0), (1, 0))), np.pad(boundaries, ((0, 0), (1, 0))), 'S', 'boundary_jump_error_MSE_trainingSD2')
    add(np.mean((oracle_sum - arrays[modality + '_u']) ** 2, axis=-1), selected, 'sum', 'tokenizer_NMSE')
    add(np.mean((summed - oracle_sum) ** 2, axis=-1), selected, 'sum', 'model_to_tokenizer_NMSE')
    return result, predicted


def _trace_plot(path, rows, arrays, oracle, predicted, target, modality, route):
    if not len(rows): return
    import matplotlib; matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    channels = 2 if modality == 'hb' else 1
    shape = (len(rows), -1, channels)
    fig, axes = plt.subplots(3, channels, figsize=(11, 7), squeeze=False, constrained_layout=True)
    time_axis = np.arange(np.asarray(arrays[modality + '_u'])[0].size // channels) * .1
    s = np.asarray(arrays[modality + '_s']).reshape(shape)
    o = np.asarray(arrays[modality + '_e']).reshape(shape)
    u = np.asarray(arrays[modality + '_u']).reshape(shape)
    mask = np.repeat(target[0], arrays[modality + '_u'].shape[-1] // channels)
    for c in range(channels):
        for a, (teacher, dec, pred, title) in enumerate(((s, oracle[0], predicted[0], 'S'),
                (o, oracle[1], predicted[1], 'O'), (u, oracle[0] + oracle[1], predicted[0] + predicted[1], 'sum'))):
            ax = axes[a, c]
            if route != 'ordinary_hard' or a == 2:
                label = ('processed observation' if a == 2 else 'matched smoothing target' if route == 'smooth_hard' else 'fixed teacher target')
                ax.plot(time_axis, teacher[0, :, c], label=label, color='black')
            ax.plot(time_axis, dec.reshape(shape)[0, :, c], label='true-ID frozen decode', color='#d98b23', ls='--')
            ax.plot(time_axis, pred.reshape(shape)[0, :, c], label='masked expected decode', color='#177db0')
            ax.fill_between(time_axis, 0, 1, where=mask, transform=ax.get_xaxis_transform(), color='#aaaaaa', alpha=.2)
            ax.set(title=f'{modality} {title}' + (' (arbitrary additive stream)' if route == 'ordinary_hard' and a != 2 else '') + (f' channel {c + 1}' if channels > 1 else ''),
                   xlabel='event time (s)', ylabel='frozen training SD')
            ax.legend(fontsize=7); ax.grid(alpha=.2)
    row = rows[0]
    fig.suptitle(f"Offline processed-observation completion | first heldout identity {row.get('id', row['subject'])}\nGrey: jointly hidden S/O blocks; teacher shown for scoring only", fontsize=9)
    fig.savefig(path, dpi=200); plt.close(fig)


def run_masked_pilot(cfg, run, *, device='cpu', seed=17, route='hard_student'):
    """Train paired/own readers, then frozen-context controls at identical masks.

    CE is interpreted only relative to this route's train-fitted baselines.
    Cross-route comparison uses the common observation-space decoded NMSE.
    """
    if route not in ('hard_student', 'smooth_hard', 'ordinary_hard'):
        raise ValueError('masked pilot requires a frozen deployable hard-ID vocabulary')
    device = str(device)
    if device.startswith('cuda'):
        budget_mib = float(cfg.get('resources', {}).get('max_new_gpu_memory_mib', 4096))
        properties = torch.cuda.get_device_properties(device)
        torch.cuda.set_per_process_memory_fraction(min(budget_mib * 1024 ** 2 / properties.total_memory, 1.), device)
        torch.cuda.reset_peak_memory_stats(device)
    run = Path(run); cache, manifest, all_arrays = _cache(run); all_rows = manifest['rows']
    config = dict(cfg.get('masked', {})); config['vocab_size'] = int(cfg['model']['codebook_size'])
    job = _job(cfg, run, route, seed)
    job_record = json.loads((job / 'summary.json').read_text()) if (job / 'summary.json').exists() else {}
    if job_record.get('status') != 'completed' or job_record.get('step') != int(cfg.get('training', {}).get('steps', 3000)):
        raise ValueError('masked reader requires a completed fixed-budget frozen tokenizer job')
    with np.load(job / 'predictions.npz', allow_pickle=False) as export:
        ids = np.stack([np.stack([export[m + '_ids_' + c] for c in BRANCHES], axis=-1) for m in MODALITIES], axis=-2)
    if ids.dtype.kind not in 'iu' or ids.shape != (len(all_rows), int(manifest.get('patches', 10)), 2, 2):
        raise ValueError('exact same-ROI [N,10,2,2] hard ID export required')
    with np.load(job / 'codewords.npz', allow_pickle=False) as books:
        decoded = {(m, c):books[modality + '_' + branch] for m, modality in enumerate(MODALITIES) for c, branch in enumerate(BRANCHES)}
    if any(len(w) != config['vocab_size'] for w in decoded.values()):
        raise ValueError('vocabulary/decoder count mismatch')
    if np.any(ids >= config['vocab_size']):
        raise ValueError('token IDs exceed frozen vocabulary')
    digest = hashlib.sha256(np.ascontiguousarray(ids).tobytes())
    for key in sorted(decoded): digest.update(np.ascontiguousarray(decoded[key]).tobytes())
    config['frozen_representation_signature'] = digest.hexdigest()
    # Only metadata keys permitted in the baseline; condition/task are ignored.
    baseline_rows = [dict(row, roi=_region(row)) for row in all_rows]
    baseline = fit_token_baselines(ids, baseline_rows, config['vocab_size'])
    output = run / 'masked' / route / ('seed_' + str(seed)); output.mkdir(parents=True, exist_ok=True)
    models = {name:_fit_reader(ids, all_rows, config, output / name, device=device,
                seed=seed, reader=name) for name in ('paired', 'own')}
    selected = np.array([r['split'] in ('validation', 'test') for r in all_rows])
    eval_ids = ids[selected]; rows = [r for i, r in enumerate(all_rows) if selected[i]]
    if not len(rows): raise ValueError('heldout scoring support missing')
    arrays = {key:np.asarray(value[selected]) for key, value in all_arrays.items()}
    if route == 'smooth_hard':
        smoothing = json.loads((cache / 'smoothing.json').read_text())
        for m in MODALITIES:
            for name in ('s', 'e'):
                arrays[m + '_' + name] = np.load(cache / smoothing['arrays'][m + '_' + name], mmap_mode='r')[selected]
    baseline_eval_rows = [dict(r, roi=_region(r)) for r in rows]
    marginal = baseline_probabilities(baseline, baseline_eval_rows)
    metadata = baseline_probabilities(baseline, baseline_eval_rows, metadata=True)
    train = np.array([r['split'] == 'train' for r in all_rows])
    scores, donor_reports = [], []
    for target_m, modality in enumerate(MODALITIES):
        source_m = 1 - target_m
        for task in ('time', 'modality'):
            visible, target = block_masks(eval_ids >= 0, rng=np.random.default_rng(seed + target_m * 10 + (task == 'modality')),
                target_modality=target_m, block_patches=int(config.get('block_patches', 2)), task=task)
            contexts = [('paired', models['paired'], eval_ids, visible),
                        ('own_reader', models['own'], eval_ids, own_context(visible, target_m)),
                        ('frozen_remove_source_modality', models['paired'], eval_ids, own_context(visible, target_m))]
            for c, branch in enumerate(BRANCHES):
                ablated = visible.copy(); ablated[:, :, source_m, c] = False
                contexts.append(('frozen_remove_source_' + branch, models['paired'], eval_ids, ablated))
            # Retain only source tokens preceding each sample's first target patch.
            past = visible.copy()
            target_time = target[:, :, target_m].any(-1)
            for i in range(len(rows)):
                locations = np.flatnonzero(target_time[i])
                if len(locations): past[i, locations[0]:] = False
            contexts.append(('frozen_past_only', models['paired'], eval_ids, past))
            donor, available, report = matched_donor_tokens(eval_ids, rows, decoded, source_modality=source_m,
                training_ids=ids[train], training_rows=[r for i, r in enumerate(all_rows) if train[i]], seed=seed)
            donor_visible = visible.copy(); donor_visible[:, :, source_m] &= available[:, :, None]
            contexts.append(('frozen_matched_donor', models['paired'], donor, donor_visible))
            donor_reports.append(dict(target_modality=modality, task=task, **report))
            # Nonwrapping time shift: vacated source tokens are hidden, never wrapped.
            shift = min(int(config.get('shift_patches', 3)), ids.shape[1] - 1)
            shifted, shifted_visible = nonwrapping_source_shift(eval_ids, visible, source_m, shift)
            contexts.append(('frozen_nonwrapping_source_shift', models['paired'], shifted, shifted_visible))
            # Explicit train-fit probability controls, with no contextual input.
            controls = [('training_marginal', marginal), ('training_metadata', metadata)]
            for name, model, context_ids, context_visible in contexts:
                probability = _predict(model, context_ids, context_visible, device=device,
                    batch_size=int(config.get('eval_batch_size', 128)))
                window_scores, predicted = _score_probabilities(probability, eval_ids, target, rows, decoded, arrays,
                    marginal, metadata, route=route, seed=seed, task=task, context=name, target_modality=target_m)
                scores.extend(_metric_subjects(window_scores))
                if name == 'paired' and task == 'modality':
                    oracle = {c:decoded[(target_m, c)][np.maximum(eval_ids[:, :, target_m, c], 0)] for c in range(2)}
                    _trace_plot(output / ('trace_' + modality + '.png'), rows, arrays, oracle, predicted,
                                target[:, :, target_m].all(-1), modality, route)
            for name, probability in controls:
                window_scores, _ = _score_probabilities(probability, eval_ids, target, rows, decoded, arrays,
                    marginal, metadata, route=route, seed=seed, task=task, context=name, target_modality=target_m)
                scores.extend(_metric_subjects(window_scores))
            # Waveform baselines see the same visible token interface as readers.
            oracle_sum = sum(decoded[(target_m, c)][np.maximum(eval_ids[:, :, target_m, c], 0)] for c in range(2))
            mean_wave = np.asarray(all_arrays[modality + '_u'][train]).mean(axis=0)
            selected_target = target[:, :, target_m].all(-1)
            for name in ('training_mean_waveform', 'linear_interpolation_visible_decodes'):
                prediction = np.broadcast_to(mean_wave, oracle_sum.shape).copy()
                if name.startswith('linear'):
                    for i in range(len(rows)):
                        known = np.flatnonzero(visible[i, :, target_m].all(-1))
                        if len(known):
                            for d in range(oracle_sum.shape[-1]):
                                prediction[i, :, d] = np.interp(np.arange(ids.shape[1]), known, oracle_sum[i, known, d])
                error = np.mean((prediction - arrays[modality + '_u']) ** 2, axis=-1)
                window_rows = [dict(dataset=r.get('dataset', ''), subject=r['subject'],
                    record=r.get('record', r.get('join_key', '')), split=r['split'], variant=route,
                    seed=seed, task=task, context=name, modality=modality, component='sum', metric='NMSE', target_family='processed_observation',
                    value=float(error[i][selected_target[i]].mean()), failures=0)
                    for i, r in enumerate(rows) if selected_target[i].any()]
                scores.extend(_metric_subjects(window_rows))
        # A same-position opposite branch remains visible: explicitly diagnostic.
        for c, branch in enumerate(BRANCHES):
            visible, target = block_masks(eval_ids >= 0, rng=np.random.default_rng(seed + 100 + target_m),
                target_modality=target_m, block_patches=int(config.get('block_patches', 2)), task='modality', single_branch=c)
            probability = _predict(models['paired'], eval_ids, visible, device=device)
            window_rows, _ = _score_probabilities(probability, eval_ids, target, rows, decoded, arrays,
                marginal, metadata, route=route, seed=seed, task='single_branch_' + branch + '_diagnostic',
                context='paired', target_modality=target_m)
            scores.extend(_metric_subjects(window_rows))
    _csv(output / 'subject_metrics.csv', scores)
    _json(output / 'donor_controls.json', donor_reports)
    summary = dict(schema='dual_component_masked_pilot_v1', status='completed', route=route, seed=seed,
        reader_steps=int(config.get('steps', 1000)), readers=['paired', 'own'],
        train_rows=int(train.sum()), evaluation_rows=len(rows), subject_metric_rows=len(scores),
        teacher_inputs=False, sample_input='hard integer IDs only, one ROI 10-patch context',
        decoded_expectation='probability-weighted individually decoded fixed codewords',
        cross_vocabulary_CE_ranking_allowed=False, main_comparison='same-observation-space sum NMSE',
        ordinary_branch_semantics='arbitrary additive streams' if route == 'ordinary_hard' else None,
        context_controls='frozen reader ablations are dependencies, not causal physiological effects',
        physical_compatibility=dict(status='not_evaluated',
            reason='Boundary jumps describe morphological continuity only; no Tak whole-trajectory compatibility witness was evaluated'),
        continuity_diagnostics='S patch-boundary jump roughness and signed-jump error MSE in training-SD squared; morphological description, not Tak reachability or compatibility',
        donor_reports=donor_reports, scope='offline processed-observation completion',
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.startswith('cuda') else 0,
        cuda_memory_budget_mib=cfg.get('resources', {}).get('max_new_gpu_memory_mib', 4096) if device.startswith('cuda') else None,
        inference='descriptive heldout public-development evidence; no protected confirmation')
    _json(output / 'summary.json', summary)
    return summary


def vocabulary_quality_screen(subjects, cfg):
    settings = cfg.get('evaluation', {}).get('quality_screen', {})
    minimum_gain = float(settings.get('minimum_relative_gain_over_train_mean', .1))
    maximum_gap = float(settings.get('maximum_hard_minus_continuous_over_mean', .1))
    checks = []
    for dataset in sorted({r.get('dataset', '') for r in subjects}):
        for modality in MODALITIES:
            for branch in BRANCHES:
                values = {}
                for variant in ('hard_student', 'continuous_student', 'training_mean'):
                    bysubject = defaultdict(list)
                    for row in subjects:
                        if (row.get('dataset', '') == dataset and row['modality'] == modality
                                and row['component'] == branch and row.get('signal') in ('EEG_proxy', 'Hb_joint') and row['split'] == 'validation'
                                and row['variant'] == variant and row.get('NMSE') is not None):
                            bysubject[row['subject']].append(row['NMSE'])
                    values[variant] = float(np.mean([np.mean(x) for x in bysubject.values()])) if bysubject else None
                hard, continuous, mean = (values[k] for k in ('hard_student', 'continuous_student', 'training_mean'))
                available = all(x is not None and np.isfinite(x) for x in (hard, continuous, mean)) and mean > 0
                gain = 1 - hard / mean if available else None
                gap = (hard - continuous) / mean if available else None
                checks.append(dict(dataset=dataset, modality=modality, component=branch, validation_NMSE=values,
                    relative_gain_over_train_mean=gain, hard_minus_continuous_over_mean=gap,
                    passed=bool(available and gain >= minimum_gain and gap <= maximum_gap)))
    passed = bool(checks and all(x['passed'] for x in checks))
    return dict(passed=passed, status='passed' if passed else 'diagnostic_only', checks=checks,
        thresholds=dict(minimum_relative_gain_over_train_mean=minimum_gain,
                        maximum_hard_minus_continuous_over_mean=maximum_gap),
        on_failure='continue fixed small masked pilot diagnostic-only; no scale expansion',
        selection_split='validation only; no test-based training or selection')


def analyze_teacher_sensitivity(cfg, run):
    """Paired equal-budget training-identity sensitivity, never a truth audit."""
    root = Path(run) / 'sensitivity'
    declared = json.loads((root / 'summary.json').read_text()) if (root / 'summary.json').exists() else {}
    manifest_path = root / 'manifest.json'
    seed = int(cfg.get('sensitivity', {}).get('seed', 17))
    dirs = {name:root / name / ('seed_' + str(seed)) for name in ('reference', 'alternative')}
    expected_steps = int(cfg.get('sensitivity', {}).get('paired_steps', 500))
    arm_records = [json.loads((path / 'summary.json').read_text()) if (path / 'summary.json').exists() else {} for path in dirs.values()]
    ready = all(record.get('status') == 'completed' and record.get('step') == expected_steps for record in arm_records)
    if not manifest_path.exists() or not ready or any(not (path / 'codewords.npz').exists() for path in dirs.values()):
        return dict(status='unavailable', reason=declared.get('reason', 'No complete legal near-equivalent whole-trajectory witness and paired equal-budget retraining exports'),
                    finite_search=declared, physiological_truth_claim=False)
    manifest = json.loads(manifest_path.read_text()); rows = manifest['rows']
    exports, codewords = {}, {}
    for name, path in dirs.items():
        with np.load(path / 'predictions.npz', allow_pickle=False) as data:
            exports[name] = {key:data[key] for key in data.files}
        with np.load(path / 'codewords.npz', allow_pickle=False) as data:
            codewords[name] = {key:data[key] for key in data.files}
    results = []
    for modality in MODALITIES:
        for branch in BRANCHES:
            key = modality + '_' + branch
            mapping, effect = align_codewords(codewords['reference'][key], codewords['alternative'][key])
            ref = exports['reference'][modality + '_ids_' + branch]
            alt = exports['alternative'][modality + '_ids_' + branch]
            valid = (ref >= 0) & (alt >= 0)
            delta = np.mean((exports['reference'][key] - exports['alternative'][key]) ** 2, axis=-1)
            for dataset in sorted({r.get('dataset', '') for r in rows}):
                select = np.array([r.get('dataset', '') == dataset for r in rows])
                keep = select[:, None] & valid
                results.append(dict(dataset=dataset, modality=modality, component=branch, **effect,
                    assigned_decoded_waveform_change_NMSE=float(delta[keep].mean()) if keep.any() else None,
                    aligned_ID_agreement=float(np.mean(ref[keep] == mapping[alt[keep]])) if keep.any() else None,
                    windows=int(select.sum()), subjects=len({r['subject'] for i, r in enumerate(rows) if select[i]}),
                    interpretation='paired train-identity teacher sensitivity; no heldout-generalization or physiological-truth claim'))
    _csv(root / 'prototype_sensitivity.csv', results)
    result = dict(status='completed', seed=seed, paired_steps=int(cfg.get('sensitivity', {}).get('paired_steps', 500)),
        rows=results, evidence=declared, evaluation='same selected training identities, equal optimizer budget',
        scientific_limit='finite legal whole-trajectory witness, not posterior or complete feasible set',
        physiological_truth_claim=False)
    _json(root / 'analysis.json', result)
    return result


def summarize_masked_pilots(cfg, run):
    run = Path(run); config = cfg.get('masked', {}); subjects, missing = [], []
    for route in config.get('routes', ('hard_student', 'smooth_hard', 'ordinary_hard')):
        for seed in config.get('seeds', cfg.get('seeds', (17, 29, 43))):
            path = run / 'masked' / route / ('seed_' + str(seed)) / 'subject_metrics.csv'
            if not path.exists():
                missing.append(dict(route=route, seed=seed)); continue
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    row['value'] = float(row['value']) if row['value'] else None
                    row['seed'] = int(row['seed']); subjects.append(row)
    if not subjects: return dict(status='not_run', missing=missing)
    comparisons = []; samples = int(cfg.get('evaluation', {}).get('bootstrap_repeats', 1000))
    grouped = defaultdict(list)
    for row in subjects:
        grouped[(row['variant'], row['dataset'], row['split'], row['task'], row['modality'], row['component'], row['metric'])].append(row)
    for (route, dataset, split, task, modality, component, metric), group in grouped.items():
        if metric not in ('NMSE', 'NMSE_HbO', 'NMSE_HbR', 'CE_nats', 'DeltaCE_metadata_nats'): continue
        first = [r for r in group if r['context'] == 'paired']
        for context in sorted({r['context'] for r in group} - {'paired'}):
            second = [r for r in group if r['context'] == context]
            result = paired_subject_bootstrap(first, second, value='value', samples=samples, seed=int(cfg.get('seed', 0)))
            result['effect'] = 'paired minus control; ' + ('higher favors paired' if metric.startswith('DeltaCE') else 'lower favors paired')
            comparisons.append(dict(route=route, dataset=dataset, split=split, task=task, modality=modality,
                component=component, metric=metric, first='paired', second=context, **result))
    # Across vocabularies compare only the shared observation-space sum error.
    for control in ('smooth_hard', 'ordinary_hard'):
        keys = {(r['dataset'], r['split'], r['task'], r['modality']) for r in subjects
                if r['metric'] in ('NMSE', 'NMSE_HbO', 'NMSE_HbR') and r['component'] == 'sum' and r['context'] == 'paired'}
        for dataset, split, task, modality in keys:
            def pick(route):
                return [r for r in subjects if r['variant'] == route and r['dataset'] == dataset
                    and r['split'] == split and r['task'] == task and r['modality'] == modality
                    and r['metric'] == 'NMSE' and r['component'] == 'sum' and r['context'] == 'paired']
            result = paired_subject_bootstrap(pick('hard_student'), pick(control), value='value', samples=samples, seed=int(cfg.get('seed', 0)))
            comparisons.append(dict(route='cross_vocabulary', dataset=dataset, split=split, task=task,
                modality=modality, component='sum', metric='NMSE', first='hard_student', second=control, **result))
    _csv(run / 'analysis' / 'masked_subject_metrics.csv', subjects)
    _csv(run / 'analysis' / 'masked_paired_comparisons.csv', comparisons)
    result = dict(status='completed' if not missing else 'partial', missing=missing, subject_rows=len(subjects),
        comparisons=comparisons, cross_vocabulary_CE_ranking_allowed=False,
        aggregation='equal record means per subject; seeds averaged within subject before resampling',
        inference='exploratory percentile subject bootstrap; no multiple-comparison significance claims')
    _json(run / 'analysis' / 'masked_summary.json', result)
    return result


def refinement_fidelity(predictions, targets, rows, variant):
    """Training/validation waveform metrics; no test rows or per-window scaling."""
    if any(r['split'] not in ('train', 'validation') for r in rows):
        raise ValueError('refinement selection is training/validation only')
    training = np.array([r['split'] == 'train' for r in rows])
    window_rows = []
    for modality, channels in (('eeg', 1), ('hb', 2)):
        truth = dict(S=targets[modality + '_s'], O=targets[modality + '_e'], sum=targets[modality + '_u'])
        for component, target in truth.items():
            estimate = np.asarray(predictions[modality + '_' + component], float)
            target = np.asarray(target, float)
            if estimate.shape != target.shape or len(target) != len(rows):
                raise ValueError('aligned complete waveform support required')
            a = estimate.reshape(len(rows), -1, channels)
            b = target.reshape(a.shape)
            centered_a = a - a.mean(1, keepdims=True); centered_b = b - b.mean(1, keepdims=True)
            norm_a = np.sqrt(np.sum(centered_a ** 2, axis=(1, 2)))
            norm_b = np.sqrt(np.sum(centered_b ** 2, axis=(1, 2)))
            denominator = norm_a * norm_b
            correlation = np.divide(np.sum(centered_a * centered_b, axis=(1, 2)), denominator,
                out=np.full(len(rows), np.nan), where=denominator > 1e-12)
            delta_a, delta_b = np.diff(a, axis=1), np.diff(b, axis=1)
            # Frozen training RMS of differences, not SD or a window denominator.
            delta_scale = max(float(np.mean(delta_b[training] ** 2)), 1e-8)
            metrics = dict(NMSE=np.mean((a - b) ** 2, axis=(1, 2)),
                derivative_NMSE=np.mean((delta_a - delta_b) ** 2, axis=(1, 2)) / delta_scale,
                centered_correlation=correlation,
                centered_amplitude_ratio=np.divide(norm_a, norm_b, out=np.full(len(rows), np.nan), where=norm_b > 1e-12))
            for metric, values in metrics.items():
                for i, row in enumerate(rows):
                    window_rows.append(dict(dataset=row['dataset'], subject=row['subject'], record=row['record'],
                        split=row['split'], variant=variant, modality=modality, component=component,
                        metric=metric, value=float(values[i]) if np.isfinite(values[i]) else None,
                        failures=int(not np.isfinite(values[i]))))
    subjects = subject_aggregate(window_rows, value='value')
    grouped = defaultdict(list)
    for row in subjects:
        grouped[(row['dataset'], row['split'], row['modality'], row['component'], row['metric'])].append(row)
    summaries = []
    for (dataset, split, modality, component, metric), values in grouped.items():
        valid = [r['value'] for r in values if r['value'] is not None]
        summaries.append(dict(dataset=dataset, split=split, modality=modality, component=component,
            metric=metric, value=float(np.mean(valid)) if valid else None, subjects=len(valid),
            undefined_subjects=len(values) - len(valid), undefined_windows=sum(r['failures'] for r in values)))
    return subjects, summaries


def refinement_code_health(ids, words, rows, modality, component, size):
    """Per-level occupancy and correctly identified subject/record coverage."""
    result = []
    for level in range(ids.shape[-1]):
        for split in ('train', 'validation'):
            take = np.array([r['split'] == split for r in rows])
            selected = ids[take, :, level]
            counts = np.bincount(selected.ravel(), minlength=size)
            probabilities = counts / counts.sum()
            nz = probabilities > 0
            per_code_subjects, per_code_records = [], []
            for code in range(size):
                assigned = np.flatnonzero(take & (ids[:, :, level] == code).any(-1))
                per_code_subjects.append(len({(rows[i]['dataset'], rows[i]['subject']) for i in assigned}))
                per_code_records.append(len({(rows[i]['dataset'], rows[i]['subject'], rows[i]['record']) for i in assigned}))
            prototypes = np.asarray(words[level], float)
            distance = np.mean((prototypes[:, None] - prototypes[None]) ** 2, axis=-1)
            np.fill_diagonal(distance, np.inf)
            result.append(dict(modality=modality, component=component, level=level, split=split,
                size=size, active_codes=int(nz.sum()), active_fraction=float(nz.mean()),
                perplexity=float(np.exp(-(probabilities[nz] * np.log(probabilities[nz])).sum())),
                max_probability=float(probabilities.max()), minimum_assignments=int(counts.min()),
                codes_with_three_subjects_fraction=float(np.mean(np.asarray(per_code_subjects) >= 3)),
                per_code_subjects=per_code_subjects, per_code_records=per_code_records,
                nearest_prototype_RMS_median=float(np.sqrt(np.median(distance.min(-1))))))
    return result


def refinement_decision(summary, baseline, health, settings):
    """Predeclared engineering screen: usage cannot substitute for fidelity."""
    reference = {(r['dataset'], r['modality'], r['component']):r['value'] for r in baseline
        if r['split'] == 'validation' and r['metric'] == 'NMSE'}
    fidelity = []
    for row in summary:
        if row['split'] != 'validation' or row['metric'] != 'NMSE': continue
        key = row['dataset'], row['modality'], row['component']; base = reference[key]
        gain = 1 - row['value'] / base if row['value'] is not None and base > 0 else None
        fidelity.append(dict(dataset=key[0], modality=key[1], component=key[2], baseline_NMSE=base,
            NMSE=row['value'], relative_improvement=gain,
            passed=bool(gain is not None and gain >= settings['minimum_waveform_improvement'])))
    health_checks = []
    for row in health:
        required_active = settings['train_active_fraction'] if row['split'] == 'train' else settings['validation_active_fraction']
        health_checks.append(dict(modality=row['modality'], component=row['component'], level=row['level'], split=row['split'],
            passed=bool(row['active_fraction'] >= required_active and row['perplexity'] / row['size'] >= settings['effective_fraction']
                and row['max_probability'] <= settings['maximum_code_probability']),
            active_fraction=row['active_fraction'], effective_fraction=row['perplexity'] / row['size'],
            maximum_code_probability=row['max_probability']))
    return dict(fidelity_pass=all(r['passed'] for r in fidelity),
        health_pass=bool(health_checks and all(r['passed'] for r in health_checks)),
        fidelity=fidelity, health=health_checks,
        selection='validation only; no posthoc change of success thresholds')


def render_refinement(cfg, run, selected, *, export=None):
    """Fixed-rule validation cases: teacher, previous hard decode, new hard decode."""
    if selected is None: return []
    import matplotlib; matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    run = Path(run); cache = run / 'cache'; manifest = json.loads((cache / 'manifest.json').read_text())
    folder = run / 'refinement' / selected['candidate'] / ('seed_' + str(selected['seed']))
    variant = 'hard' if selected['name'] == selected['candidate'] else 'vq' + selected['name'].rsplit('_vq', 1)[1]
    continuous_file = folder / 'continuous/validation_predictions.npz'
    folder = folder / variant
    with np.load(run / 'baseline_hard/validation_predictions.npz') as data:
        baseline = {k:data[k] for k in data.files}
    with np.load(folder / 'validation_predictions.npz') as data:
        updated = {k:data[k] for k in data.files}
    if not np.array_equal(baseline['row_index'], updated['row_index']):
        raise ValueError('comparison plots require identical validation identities')
    continuous = None
    if continuous_file.exists():
        with np.load(continuous_file) as data: continuous = {k:data[k] for k in data.files}
        if not np.array_equal(baseline['row_index'], continuous['row_index']):
            raise ValueError('continuous/hard comparison requires identical identities')
    indices = baseline['row_index']; rows = [manifest['rows'][i] for i in indices]
    targets = {m + '_' + c:np.load(cache / (m + '_' + suffix + '.npy'), mmap_mode='r')[indices]
        for m in MODALITIES for c, suffix in (('S','s'),('O','e'),('sum','u'))}
    baseline_error = np.mean((baseline['hb_sum'] - targets['hb_sum']) ** 2, axis=(1,2))
    destination = run if export is None else Path(export)
    if export is not None: destination.mkdir(parents=True, exist_ok=False)
    examples = []; files = []; output = destination / 'figures'; output.mkdir(exist_ok=True)
    for dataset in sorted({r['dataset'] for r in rows}):
        group = [i for i, row in enumerate(rows) if row['dataset'] == dataset]
        used = set()
        for label, quantile in [('first_identity', None), ('median_parent_error', .5), ('p90_parent_error', .9)]:
            candidates = [i for i in group if rows[i]['subject'] not in used]
            if not candidates: continue
            if quantile is None: i = candidates[0]
            else:
                target = float(np.quantile(baseline_error[group], quantile))
                i = min(candidates, key=lambda j: (abs(baseline_error[j] - target), rows[j]['id']))
            used.add(rows[i]['subject']); examples.append(dict(row=rows[i], selection=label,
                parent_sum_NMSE=float(baseline_error[i]), candidate_used_for_selection=False))
            for modality, channels in [('eeg',1),('hb',2)]:
                fig, axes = plt.subplots(3, channels, figsize=(12 if channels == 2 else 9, 8),
                    squeeze=False, constrained_layout=True)
                time_axis = np.arange(200) * .1
                for row_index, component in enumerate(('S','O','sum')):
                    key = modality + '_' + component
                    truth = targets[key][i].reshape(200, channels)
                    old = baseline[key][i].reshape(200, channels)
                    new = updated[key][i].reshape(200, channels)
                    for channel in range(channels):
                        ax = axes[row_index, channel]
                        ax.plot(time_axis, truth[:,channel], color='black', lw=1.7,
                            label='Observed target' if component == 'sum' else 'Fixed Tak component target')
                        ax.plot(time_axis, old[:,channel], color='#888888', ls='--', lw=1.5, label='Previous hard tokenizer')
                        if continuous is not None:
                            before_vq = continuous[key][i].reshape(200, channels)[:,channel]
                            ax.plot(time_axis, before_vq, color='#D55E00', ls=':', lw=1.4, label='Refined before quantization')
                        ax.plot(time_axis, new[:,channel], color='#0072B2', lw=1.6, label='Refined hard tokenizer')
                        before = np.mean((old[:,channel] - truth[:,channel]) ** 2)
                        after = np.mean((new[:,channel] - truth[:,channel]) ** 2)
                        signal = ('HbO','HbR')[channel] if channels == 2 else 'EEG power proxy'
                        meaning = dict(S='S: Tak output', O='O: fixed residual', sum='total: observed')[component]
                        ax.set_title(f'{signal} | {meaning} | NMSE {before:.4g} -> {after:.4g}', fontsize=10)
                        ax.set_xlabel('Event time (s)'); ax.set_ylabel('Frozen training SD'); ax.grid(alpha=.2)
                        if row_index == 0: ax.legend(fontsize=7, loc='best')
                candidate = next(c for c in cfg['refinement']['candidates'] if c['name'] == selected['candidate'])
                support = 2 * (1 + 2 * candidate['context_radius'])
                fig.suptitle(f'{selected["name"]} | seed {selected["seed"]} | {support}s processed-observation support | '
                    f'14 -> {selected["bits"]:g} bits / modality / 2s\n'
                    f'Validation: {rows[i]["id"]}\n'
                    f'{label}; complete observations visible; no masked-model predictions', fontsize=10)
                path = output / f'{dataset}_{label}_{modality}.png'
                fig.savefig(path, dpi=250); plt.close(fig); files.append(str(path))
    _json(destination / 'examples.json', examples)
    report = json.loads((folder / 'summary.json').read_text())
    health = [x for x in report['codebook_health'] if x['split'] == 'validation']
    fig, axes = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    labels = [f'{x["modality"]}-{x["component"]}-L{x["level"]+1}' for x in health]
    for ax, values, threshold, title in [
        (axes[0],[x['active_fraction'] for x in health],cfg['refinement']['health']['validation_active_fraction'],'Active codes / 128'),
        (axes[1],[x['perplexity']/x['size'] for x in health],cfg['refinement']['health']['effective_fraction'],'Effective codes / 128')]:
        ax.bar(labels, values, color='#0072B2'); ax.axhline(threshold,color='#D55E00',ls='--',label='Predeclared threshold')
        ax.set_ylim(0,1.05); ax.set_title(title); ax.tick_params(axis='x',rotation=35); ax.legend(fontsize=8)
        for i,value in enumerate(values): ax.text(i,value+.01,f'{value:.2f}',ha='center',fontsize=8)
    fig.suptitle(f'Validation codebook health | {selected["name"]} | seed {selected["seed"]} | {selected["bits"]:g} bits / modality / 2s')
    path = output / 'selected_codebook_health.png'; fig.savefig(path,dpi=250);plt.close(fig);files.append(str(path))
    suite_path = run / 'refinement_summary.json'
    if suite_path.exists():
        suite = json.loads(suite_path.read_text())
        matching = [v for r in suite['candidates'] + suite.get('confirmation', [])
            if r.get('status') == 'completed' and r['candidate']['name'] == selected['candidate']
            for v in r['variants'] if v['name'] == selected['name']]
        if matching:
            datasets = sorted({r['dataset'] for r in report['decision']['fidelity']})
            labels = ['EEG S','EEG O','EEG total','Hb S','Hb O','Hb total']
            keys = [(m,c) for m in ('eeg','hb') for c in ('S','O','sum')]
            fig, axes = plt.subplots(1,len(datasets),figsize=(13,5),squeeze=False,constrained_layout=True)
            for ax,dataset in zip(axes[0],datasets):
                values = np.array([[next(r['relative_improvement']*100 for r in v['decision']['fidelity']
                    if (r['dataset'],r['modality'],r['component']) == (dataset,*key)) for key in keys] for v in matching])
                mean = values.mean(0); spread = np.stack((mean-values.min(0),values.max(0)-mean))
                ax.bar(np.arange(6),mean,color=['#0072B2']*3+['#009E73']*3,yerr=spread,capsize=4)
                for i,value in enumerate(mean): ax.text(i,value+3,f'{value:.1f}%',ha='center',fontsize=10)
                ax.axhline(30,color='#D55E00',ls='--',label='Predeclared 30% improvement')
                ax.axhline(0,color='black',lw=.8)
                ax.set_xticks(np.arange(6),labels,rotation=25)
                ax.set_ylim(min(-5,float(values.min())-10),105)
                ax.set_ylabel('NMSE reduction versus previous hard tokenizer (%)')
                ax.set_title(dataset); ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
            fig.suptitle(f'{selected["name"]}: mean and range of {len(matching)} seeds on the same validation subjects\n'
                f'{selected["bits"]:g} versus 14 bits / modality / 2s; fixed training-SD denominator; subjects weighted equally',fontsize=11)
            path=output/'seed_fidelity_comparison.png';fig.savefig(path,dpi=250);plt.close(fig);files.append(str(path))
    return files


VOLTAGE_METRICS=('NMSE','centered_correlation','boundary_error_NMSE','boundary_jump_RMS','derivative_error_NMSE')


def boundary_neighborhood_errors(prediction,target,*,hop=20,spans=(1,3,5,11)):
    """Separate instantaneous seam jumps from persistent offsets across a seam.

    An odd span k compares samples k intervals apart, centred on each seam.
    The target's signed change is subtracted before squaring. Result [case,span]
    averages boundaries and channels, with no sample-specific normalization.
    """
    a=np.asarray(prediction,float);b=np.asarray(target,float)
    if a.shape!=b.shape or a.ndim!=3:raise ValueError('aligned case/time/channel waveforms required')
    errors=[]
    for span in spans:
        if span<1 or span%2!=1:raise ValueError('positive odd boundary span required')
        right=np.arange(hop,a.shape[1],hop)+(span-1)//2;left=right-span
        take=(left>=0)&(right<a.shape[1]);left,right=left[take],right[take]
        if not len(left):raise ValueError('no complete boundary neighborhoods')
        difference=(a[:,right]-a[:,left])-(b[:,right]-b[:,left])
        errors.append(np.mean(difference*difference,axis=(1,2)))
    return np.stack(errors,axis=-1)


def voltage_case_metrics(predictions,targets,*,hop):
    """Per-electrode/ROI-case metrics in frozen observation units.

    The boundary error compares *signed* adjacent-sample differences at hop
    boundaries. A smooth but incorrect curve therefore cannot pass by erasing
    legitimate slopes. HbO/HbR are also scored separately.
    """
    values=np.asarray(targets['sum'],float)
    signals=[('joint',slice(None))] if values.shape[-1]==1 else [('joint',slice(None)),('HbO',slice(0,1)),('HbR',slice(1,2))]
    result=np.full((len(values),3,len(signals),len(VOLTAGE_METRICS)),np.nan)
    for j,component in enumerate(('S','O','sum')):
        if component not in predictions:continue
        a=np.asarray(predictions[component],float);b=np.asarray(targets[component],float)
        if a.shape!=b.shape or a.shape!=values.shape:raise ValueError('same complete continuous support required')
        for k,(_,select) in enumerate(signals):
            x,y=a[:,:,select],b[:,:,select];dx,dy=np.diff(x,axis=1),np.diff(y,axis=1)
            xc=x-x.mean(1,keepdims=True);yc=y-y.mean(1,keepdims=True)
            denominator=np.sqrt(np.sum(xc*xc,axis=(1,2))*np.sum(yc*yc,axis=(1,2)))
            corr=np.divide(np.sum(xc*yc,axis=(1,2)),denominator,out=np.full(len(x),np.nan),where=denominator>1e-12)
            boundary=dx[:,hop-1::hop]-dy[:,hop-1::hop]
            result[:,j,k,:]=np.stack((np.mean((x-y)**2,axis=(1,2)),corr,
                np.mean(boundary**2,axis=(1,2)),np.sqrt(np.mean(dx[:,hop-1::hop]**2,axis=(1,2))),
                np.mean((dx-dy)**2,axis=(1,2))),axis=-1)
    return result,[x[0] for x in signals]


def voltage_aggregate(values,rows,*,modality,variant,signals):
    """Equal records within subject, then equal subjects; all case counts retained."""
    grouped=defaultdict(lambda:defaultdict(list))
    for i,row in enumerate(rows):grouped[(row['dataset'],row['split'],row['subject'])][row['record']].append(i)
    def mean(x,axis=0):
        n=np.isfinite(x).sum(axis=axis)
        return np.divide(np.nansum(x,axis=axis),n,out=np.full(n.shape,np.nan),where=n>0)
    subjects=[];summaries=defaultdict(list)
    for (dataset,split,subject),records in grouped.items():
        average=mean(np.stack([mean(values[index]) for index in records.values()]))
        indices=[i for ids in records.values() for i in ids]
        for c,component in enumerate(('S','O','sum')):
            for s,signal in enumerate(signals):
                for j,metric in enumerate(VOLTAGE_METRICS):
                    value=float(average[c,s,j]);entry=dict(dataset=dataset,split=split,subject=subject,modality=modality,
                        variant=variant,component=component,signal=signal,metric=metric,value=value if np.isfinite(value) else None,
                        records=len(records),cases=len(indices),undefined_cases=int(np.sum(~np.isfinite(values[indices,c,s,j]))))
                    subjects.append(entry);summaries[(dataset,split,component,signal,metric)].append(entry)
    result=[]
    for (dataset,split,component,signal,metric),group in summaries.items():
        valid=[r['value'] for r in group if r['value'] is not None]
        result.append(dict(dataset=dataset,split=split,modality=modality,component=component,signal=signal,metric=metric,
            value=float(np.mean(valid)) if valid else None,subjects=len(valid),planned_subjects=len(group),
            cases=sum(r['cases'] for r in group),undefined_cases=sum(r['undefined_cases'] for r in group)))
    return subjects,result


def token_tuple_catalog(ids,rows,*,size=128):
    """Count complete residual-VQ tuples without merging shared first-level IDs."""
    ids=np.asarray(ids)
    if ids.ndim!=3 or len(ids)!=len(rows) or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=size):
        raise ValueError('Complete integer [case,patch,level] IDs and aligned identities required')
    if size**ids.shape[-1]>np.iinfo(np.int64).max:raise ValueError('Tuple packing would overflow')
    packed=np.zeros(ids.shape[:2],dtype='int64')
    for level in range(ids.shape[-1]):packed=packed*size+ids[...,level]
    keys,counts=np.unique(packed,return_counts=True)
    subjects=sorted({(r['dataset'],r['subject']) for r in rows});lookup={s:i for i,s in enumerate(subjects)}
    sid=np.array([lookup[(r['dataset'],r['subject'])] for r in rows],dtype='int64')
    if size**ids.shape[-1]*len(subjects)>np.iinfo(np.int64).max:raise ValueError('Subject/tuple packing would overflow')
    pairs=np.unique(packed*len(subjects)+sid[:,None]);subject_keys,subject_counts=np.unique(pairs//len(subjects),return_counts=True)
    if not np.array_equal(keys,subject_keys):raise ValueError('Tuple subject-count identity mismatch')
    return dict(keys=keys,counts=counts,subject_counts=subject_counts)


def select_token_tuple_examples(catalog,book,*,levels,count=3,mode='frequency'):
    """Choose frequent prototypes spanning amplitude/shape, never input fit error."""
    eligible=np.flatnonzero((catalog['counts']>=20)&(catalog['subject_counts']>=2))
    frequency_order=sorted(eligible,key=lambda i:(-int(catalog['counts'][i]),int(catalog['keys'][i])))
    if mode not in ('frequency','shape'):raise ValueError('Unknown token illustration selection')
    order=frequency_order if mode=='shape' else frequency_order[:25]
    if len(order)<count:raise ValueError('Too few repeated multi-subject full tuples for the requested examples')
    keys=catalog['keys'][order].copy();codes=np.zeros((len(keys),levels),dtype='int16')
    for level in range(levels-1,-1,-1):codes[:,level]=keys%book.size;keys//=book.size
    if keys.any():raise ValueError('Tuple level count does not match dictionary')
    waves=book.decode(codes).astype(float)
    if mode=='shape':
        shaped=waves.reshape(len(waves),-1,book.channels);centered=shaped-shaped.mean(axis=1,keepdims=True)
        dynamic=np.sqrt(np.mean(centered**2,axis=(1,2)));top=np.argsort(-dynamic,kind='stable')[:25]
        order=[order[i] for i in top];codes=codes[top];waves=centered.reshape(len(waves),-1)[top]
    selected=[0]
    while len(selected)<count:
        distance=np.min(np.mean((waves[:,None,:]-waves[selected][None,:,:])**2,axis=-1),axis=1)
        distance[selected]=-1;selected.append(int(np.argmax(distance)))
    return [dict(key=int(catalog['keys'][order[i]]),ids=codes[i].tolist(),occurrences=int(catalog['counts'][order[i]]),
        subject_count=int(catalog['subject_counts'][order[i]]),candidate_rank=i+1,
        frequency_rank=frequency_order.index(order[i])+1) for i in selected]


def render_token_correspondence(analysis,out):
    """Dark fixed S prototypes against every assigned raw input patch in pale blue."""
    import os
    import shutil
    from html import escape
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    import fitz
    analysis,out=Path(analysis),Path(out);summary=json.loads((analysis/'summary.json').read_text())
    if summary['status']!='completed' or summary['protected_reads']!=0 or summary['new_fits']!=0:raise ValueError('Completed retained-only token analysis required')
    if out.exists():raise ValueError('Preserve previous figure exports; choose a fresh output')
    out.mkdir(parents=True);(out/'figures').mkdir()
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':12,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    figures=[];pages=[];source=escape(os.path.relpath(analysis,out));dark='#14578a';light='#9ec9ef'
    for modality in ['hb','eeg']:
        result=summary['results'][modality];channels=result['channels'];items=[]
        for item in result['selected']:
            with np.load(analysis/modality/item['file'],allow_pickle=False) as a:raw=a['original'];prototype=a['prototype'];codes=a['ids']
            if len(raw)!=item['occurrences'] or raw.shape[1:]!=prototype.shape or codes.tolist()!=item['ids']:
                raise ValueError('Prototype/cloud identity or complete-member count mismatch')
            items.append((item,raw,prototype))
        fig,axes=plt.subplots(channels,len(items),figsize=(14,7.4 if channels==2 else 4.7),squeeze=False,layout='constrained')
        labels=['HbO','HbR'] if modality=='hb' else ['EEG']
        shape_view=summary.get('selection_view')=='shape';limits=[]
        for channel in range(channels):
            low=min(min(float(raw[:,:,channel].min()),float(prototype[:,channel].min())) for _,raw,prototype in items)
            high=max(max(float(raw[:,:,channel].max()),float(prototype[:,channel].max())) for _,raw,prototype in items)
            margin=max((high-low)*.06,1e-4);limits.append((low-margin,high+margin))
        time_axis=np.arange(result['hop_samples'])/result['sample_rate_hz']*(1000 if modality=='eeg' else 1)
        for col,(item,raw,prototype) in enumerate(items):
            code=' · '.join(map(str,item['ids']))
            for channel in range(channels):
                ax=axes[channel,col];coordinates=np.stack([np.broadcast_to(time_axis,raw[:,:,channel].shape),raw[:,:,channel]],axis=-1)
                ax.add_collection(LineCollection(coordinates,colors=light,linewidths=.65,alpha=.13,rasterized=True,zorder=1))
                ax.plot(time_axis,prototype[:,channel],color=dark,lw=2.7,zorder=3)
                low,high=limits[channel]
                if shape_view:
                    low=min(float(raw[:,:,channel].min()),float(prototype[:,channel].min()));high=max(float(raw[:,:,channel].max()),float(prototype[:,channel].max()))
                    margin=max((high-low)*.06,1e-4);low-=margin;high+=margin
                ax.set_xlim(0,result['hop_samples']/result['sample_rate_hz']*(1000 if modality=='eeg' else 1));ax.set_ylim(low,high);ax.grid(alpha=.12)
                ax.set_xlabel('片段内时间 / 毫秒' if modality=='eeg' else '片段内时间 / 秒')
                ax.set_ylabel(labels[channel]+(' / 冻结训练RMS' if modality=='eeg' else ' / 冻结训练SD'))
                ax.set_title('S-token ['+code+f']\n{item["occurrences"]}个原始片段 · {item["subjects"]}名被试',fontsize=11)
        fig.suptitle(('HbO / HbR' if modality=='hb' else 'EEG')+'：S-token解码波形与全部对应原始片段'+('\n各列纵轴独立缩放，保留所有原始幅度' if shape_view else ''),fontsize=17)
        fig.legend(handles=[Line2D([],[],color=dark,lw=2.7,label='S-token解码波形'),Line2D([],[],color=light,lw=1.5,label='对应的全部原始输入片段')],
            loc='outside lower center',ncol=2,fontsize=12)
        file='figures/'+modality+'_S_token_correspondence.png';fig.savefig(out/file,dpi=260,facecolor='white');plt.close(fig)
        figures.append(dict(modality=modality,file=file,examples=len(items),plotted_lines=[item['occurrences'] for item,_,_ in items],all_members=True))
        caption=('每列是一个完整S码组合，深蓝线由所有层的固定码字相加直接得到；浅蓝细线逐条绘制该组合在已评价公开测试集中的全部原始输入片段，没有抽样或按贴合程度筛除。'
            +('同列HbO/HbR属于同一批2秒片段。这里显示单码原型，未应用依赖邻码的100ms拼接后平滑。' if modality=='hb' else '每条EEG片段为100毫秒；编码器使用的同电极2秒上下文未在横轴展开。')
            +('各列使用独立纵轴范围，完整显示原始极值；跨列比较幅度须读刻度。' if shape_view else '同一信号的三列共用完整纵轴范围，保留所有极值。')
            +'浅蓝是观测输入u，深蓝仅为S分支；两者差异还包含O分支，不能直接当作完整S+O重建误差。')
        pages.append('<h1>'+('HbO/HbR' if modality=='hb' else 'EEG')+'：S-token与原始输入对应</h1><img width="750" src="'+file+'"><p>'+caption+'</p><p class="source">来源：<a href="'+source+'/'+modality+'/summary.json">完整分母与选码规则</a>；同目录example_*_members.csv记录每条浅蓝线的身份。</p>')
    counts={m:summary['results'][m]['total_patches'] for m in ['eeg','hb']}
    intro='<h1>S-token重建与原始信号：蓝色对应图</h1><p>来源：已完成的电极电压版tokenizer v2，固定seed 17。模型、词表、训练尺度及测试被试均不改变；本次只回放编码和整数解码，不训练、不重新拟合SSM。</p>'
    intro+=f'<p>统计范围为此前已评价的公开测试集，两库各6名被试：EEG共{counts["eeg"]:,}个100毫秒片段，Hb共{counts["hb"]:,}个2秒双Hb片段。EEG的完整S-token为4层整数码组合，Hb为2层；不能只用第一层ID把不同完整解码波形混为一类。</p>'
    intro+=('<p>选例先要求同一完整组合至少出现20次、覆盖至少2名被试，再按解码原型的时间变化RMS选前25组。第一例取变化最大，另两例在去时间均值的原型上依次取波形距离较大的组合；这里只在选例时去均值，实际图中的原始幅度保持不变。选择不使用原始片段贴合误差，属于形态展示样例，不是频率代表或随机样本。</p>' if summary.get('selection_view')=='shape' else
        '<p>选例先要求同一完整组合至少出现20次、覆盖至少2名被试，再取频次最高的25组：第一例为最高频，另两例按解码波形RMS距离依次选择差异较大的原型。此规则展示常见码的幅度与形状范围，没有按原始片段贴合程度选图；不代表总体最优或随机样本。</p>')
    if summary.get('selection_view')=='shape':
        diagnostic=escape(os.path.relpath(Path(summary['frequency_report']),out))
        intro+='<p>最高频码的幅度饱和和接近平坦的原型保留于<a href="'+diagnostic+'/index.html">高频码完整诊断图</a>，本组形态展示不替代这些负结果。</p>'
    intro+='<p>每个所选码组合的全部匹配片段都画成浅蓝背景；深蓝为该S-token唯一的独立解码原型。原始输入保持缓存的冻结训练尺度，不逐片段去均值、对齐峰值或归一化。EEG单位为各数据集/电极的训练RMS，Hb为继承的训练观测SD；跨坐标的原生μV或浓度单位不能直接合并成一条固定原型。</p>'
    intro+='<p>S是条件模型分量，不是原始观测u本身。Hb的S来自固定Tak输出监督，电压S来自条件幅度修正；它们均不是已知生理真值。本图说明“一组原始片段对应一个离散表示”的关系，完整观测重建仍需S+O。</p>'
    sections=[intro]+pages;css='body{font-family:sans-serif;font-size:10pt;line-height:1.45;color:#223746}h1{font-size:18pt;margin:0 0 10pt}p{margin:8pt 0}.source{font-size:8pt}img{max-width:100%}'
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>S-token对应图</title><style>'+css+'body{max-width:1150px;margin:30px auto;padding:0 24px}section{margin:35px 0}img{width:100%}</style><body><p><a href="FIGURE_ATLAS.pdf">PDF图册</a></p>'+''.join('<section>'+p+'</section>' for p in sections)+'</body></html>')
    pdf=fitz.open()
    def rectfn(number,filled):
        r=fitz.Rect(0,0,842,595);return r,r+(32,25,-32,-25),None
    for page in sections:
        part=fitz.Story(page,user_css=css,archive=fitz.Archive(str(out))).write_with_links(rectfn)
        if len(part)!=1:raise ValueError('Token-atlas page overflow; split the visible explanation')
        pdf.insert_pdf(part);part.close()
    pdf.save(out/'FIGURE_ATLAS.pdf',garbage=4,deflate=True)
    validation=dict(pages=len(pdf),bitmap_images=sum(len(p.get_images()) for p in pdf),searchable_characters=sum(len(p.get_text()) for p in pdf),png_dpi=260)
    if validation['bitmap_images']!=2 or validation['searchable_characters']<800:raise ValueError('Searchable bitmap atlas verification failed')
    pdf.close();_json(out/'source_manifest.json',dict(owning_analysis=str(analysis),figures=figures,validation=validation,refits=0))
    shutil.copy2(Path(__file__),out/'renderer_source.py');return figures


def render_voltage_examples(out,examples,summary):
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    out=Path(out);folder=out/'voltage_figures';folder.mkdir(exist_ok=True);files=[];index=[]
    for item in examples:
        row=item['row'];modality=item['modality'];channels=item['targets']['sum'].shape[-1]
        show=400 if modality=='eeg' else 200
        time_axis=np.arange(show)/item['sample_rate_hz']
        fig,axes=plt.subplots(3,channels,figsize=(12 if channels==2 else 10,8),squeeze=False,constrained_layout=True)
        for j,component in enumerate(['S','O','sum']):
            truth=item['targets'][component]
            for c in range(channels):
                ax=axes[j,c];ax.plot(time_axis,truth[:show,c],color='black',lw=1.4,label='Conditional target' if component!='sum' else 'Observed voltage/Hb')
                for role,color,style,label in [('baseline','#888888','--',item['baseline']),('selected','#0072B2','-',item['candidate']),('continuous','#D55E00',':','Before quantization')]:
                    ax.plot(time_axis,item['predictions'][role][component][:show,c],color=color,ls=style,lw=1.2,label=label)
                error=np.mean((item['predictions']['selected'][component][:,c]-truth[:,c])**2)
                label=row['channel'] if modality=='eeg' else ['HbO','HbR'][c]
                ax.set_title(f'{label} | {component} | complete-20s MSE {error:.4g}',fontsize=10)
                ax.set_xlabel('Event time (s)');ax.set_ylabel(item['units']);ax.grid(alpha=.2)
                if j==0:ax.legend(fontsize=7)
                if modality=='hb':
                    for t in range(2,20,2):ax.axvline(t,color='#bbbbbb',lw=.6,alpha=.5)
        fig.suptitle(f'{modality}: {item["selected_rate"]} | seed 17 | public test\n{row["id"]}\n'
            f'{item["selection"]}; '+('first 2s shown; metrics use all 20s' if modality=='eeg' else 'vertical lines: original 2s patch boundaries'),fontsize=10)
        path=folder/f'{row["dataset"]}_{item["selection"]}_{modality}.png';fig.savefig(path,dpi=250);plt.close(fig)
        files.append(str(path));index.append(dict(row=row,modality=modality,selection=item['selection']))
    metrics=summary['metrics'];datasets=sorted({r['dataset'] for r in metrics})
    fig,axes=plt.subplots(1,len(datasets),figsize=(12,4.5),squeeze=False,constrained_layout=True)
    for ax,dataset in zip(axes[0],datasets):
        labels=[];before=[];after=[]
        for component in ['S','O','sum']:
            for signal in ['HbO','HbR']:
                lookup=lambda model:next(r['value'] for r in metrics if (r['dataset'],r['modality'],r['component'],r['signal'],r['metric'],r['model'])==(dataset,'hb',component,signal,'boundary_error_NMSE',model))
                labels.append(component+' '+signal);before.append(lookup('hb_blocks'));after.append(lookup('selected'))
        x=np.arange(len(labels));ax.bar(x-.18,before,width=.36,color='#888888',label='Independent 2s blocks')
        label=summary['selection']['hb_candidate']+(' + 100ms synthesis' if summary['selection'].get('hb_variant')=='vq2_smoothed' else '')
        ax.bar(x+.18,after,width=.36,color='#0072B2',label=label)
        ax.set_xticks(x,labels,rotation=30);ax.set_ylabel('Signed boundary-difference error (training SD squared)')
        ax.set_title(dataset);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Public test Hb boundary error | seeds averaged within subject, subjects weighted equally',fontsize=11)
    path=folder/'hb_boundary_comparison.png';fig.savefig(path,dpi=250);plt.close(fig);files.append(str(path))
    _json(out/'voltage_figure_index.json',dict(files=files,examples=index,
        selection='first identity, median and P90 baseline Hb error; distinct subjects; EEG first native-order retained electrode of same trial',
        source='voltage_summary.json and frozen checkpoints; no fitted graphical smoothing'))
    return files


def eeg_residual_case_metrics(prediction, target, *, hop=20):
    """Complete-window O errors in the frozen raw-observation training units."""
    x, y = np.asarray(prediction, float), np.asarray(target, float)
    if x.shape != y.shape or x.ndim != 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('aligned finite [case,time] residual waveforms required')
    xc, yc = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    denominator = np.sqrt(np.sum(xc * xc, axis=1) * np.sum(yc * yc, axis=1))
    correlation = np.divide(np.sum(xc * yc, axis=1), denominator,
                            out=np.full(len(x), np.nan), where=denominator > 1e-12)
    difference = np.diff(x - y, axis=1)
    return dict(NMSE=np.mean((x - y) ** 2, axis=1),
                zero_NMSE=np.mean(y * y, axis=1),
                prediction_power=np.mean(x * x, axis=1),
                centered_correlation=correlation,
                signed_projection=np.mean(x * y, axis=1),
                boundary_error_NMSE=np.mean(difference[:, hop-1::hop] ** 2, axis=1),
                derivative_error_NMSE=np.mean(difference ** 2, axis=1))


def eeg_residual_aggregate(metrics, rows):
    """Equal records per subject, then equal subjects; ratios after averaging.

    Undefined correlations remain counted and are excluded only from that
    metric's mean. Zero-prediction MSE and predicted power are retained so a
    small error in raw-EEG units cannot masquerade as accurate residuals.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        groups[(row['dataset'], row['split'], row['subject'])][row['record']].append(i)
    def finite_mean(values):
        values = np.asarray(values, float)
        valid = np.isfinite(values)
        return float(values[valid].mean()) if valid.any() else None
    def ratios(record):
        denominator = record['zero_NMSE']
        record['zero_prediction_skill'] = 1 - record['NMSE'] / denominator if denominator > 0 else None
        record['RMS_ratio'] = float(np.sqrt(record['prediction_power'] / denominator)) if denominator > 0 else None
        record['signed_projection_ratio'] = record['signed_projection'] / denominator if denominator > 0 else None
        return record
    subjects = []
    for (dataset, split, subject), records in sorted(groups.items()):
        indices = [i for group in records.values() for i in group]
        row = dict(dataset=dataset, split=split, subject=subject, records=len(records), cases=len(indices),
                   undefined_correlations=int(np.sum(~np.isfinite(metrics['centered_correlation'][indices]))))
        row.update({name: finite_mean([finite_mean(value[group]) for group in records.values()])
                    for name, value in metrics.items()})
        subjects.append(ratios(row))
    datasets = []
    for dataset, split in sorted({(r['dataset'], r['split']) for r in subjects}):
        selected = [r for r in subjects if (r['dataset'], r['split']) == (dataset, split)]
        row = dict(dataset=dataset, split=split, subjects=len(selected), cases=sum(r['cases'] for r in selected),
                   undefined_correlations=sum(r['undefined_correlations'] for r in selected))
        row.update({name: finite_mean([r[name] for r in selected]) for name in metrics})
        datasets.append(ratios(row))
    return subjects, datasets


def summarize_eeg_o_run(run, output):
    """Read completed subject tables and export paired, descriptive evidence."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    run, output = Path(run), Path(output)
    result = json.loads((run/'eeg_o_summary.json').read_text())
    if result['status'] != 'completed':raise ValueError('complete EEG-O producer required')
    output.mkdir(parents=True, exist_ok=True)
    parent=Path(result['source_parent']);candidate=result['selection']['candidate']
    sources=[];paired=[];seeds=[]
    for job in result['selected_results']:
        seed=job['seed'];seeds.append(seed)
        new_path=run/'eeg_o_jobs'/candidate/f'seed_{seed}'/'subjects.csv'
        old_path=parent/'voltage_jobs/eeg_voltage'/f'seed_{seed}'/'evaluation/vq4_subjects.csv'
        sources.extend([str(new_path),str(old_path)])
        with new_path.open() as stream:new=list(csv.DictReader(stream))
        with old_path.open() as stream:old=list(csv.DictReader(stream))
        lookup={(r['dataset'],r['subject'],r['component'],r['metric']):r for r in old
                if r['split']=='validation' and r['signal']=='joint'}
        for row in new:
            if row['variant']!='hard':continue
            if row['split']!='validation':raise ValueError('non-validation row in EEG-O report')
            key=(row['dataset'],row['subject']);base=lookup[(*key,'O','NMSE')]
            if int(base['cases'])!=int(row['cases']):raise ValueError('unmatched residual case population')
            record=dict(dataset=key[0],subject=key[1],seed=seed,cases=int(row['cases']),records=int(row['records']))
            record.update({k:float(row[k]) for k in ['NMSE','zero_NMSE','prediction_power','centered_correlation',
                'signed_projection','boundary_error_NMSE','derivative_error_NMSE','frozen_S_plus_O_NMSE']})
            record['old_NMSE']=float(base['value'])
            record['old_correlation']=float(lookup[(*key,'O','centered_correlation')]['value'])
            record['old_sum_NMSE']=float(lookup[(*key,'sum','NMSE')]['value'])
            paired.append(record)
    groups=defaultdict(list)
    for row in paired:groups[(row['dataset'],row['subject'])].append(row)
    subjects=[]
    for (dataset,subject),rows in sorted(groups.items()):
        if {r['seed'] for r in rows}!=set(seeds):raise ValueError('missing paired optimization seed')
        record=dict(dataset=dataset,subject=subject,cases=rows[0]['cases'],records=rows[0]['records'],seeds=len(rows))
        record.update({k:float(np.mean([r[k] for r in rows])) for k in rows[0]
                       if k not in ('dataset','subject','seed','cases','records')})
        subjects.append(record)
    comparisons=[]
    for dataset in sorted({r['dataset'] for r in subjects}):
        rows=[r for r in subjects if r['dataset']==dataset]
        average={k:float(np.mean([r[k] for r in rows])) for k in rows[0]
                 if k not in ('dataset','subject','cases','records','seeds')}
        new=[dict(dataset=dataset,subject=r['subject'],NMSE=r['NMSE']) for r in rows]
        old=[dict(dataset=dataset,subject=r['subject'],NMSE=r['old_NMSE']) for r in rows]
        paired_error=paired_subject_bootstrap(new,old,samples=5000,seed=20261029)
        comparisons.append(dict(dataset=dataset,subjects=len(rows),cases=sum(r['cases'] for r in rows),**average,
            old_to_zero_ratio=average['old_NMSE']/average['zero_NMSE'],
            new_to_zero_ratio=average['NMSE']/average['zero_NMSE'],
            relative_error_reduction_percent=100*(1-average['NMSE']/average['old_NMSE']),
            zero_prediction_skill=1-average['NMSE']/average['zero_NMSE'],
            RMS_ratio=float(np.sqrt(average['prediction_power']/average['zero_NMSE'])),
            signed_projection_ratio=average['signed_projection']/average['zero_NMSE'],paired_error=paired_error))
    _csv(output/'paired_subject_seeds.csv',paired);_csv(output/'paired_subjects.csv',subjects)
    cfg=json.loads((run/'resolved_config.json').read_text())['config']
    summary=dict(source_run=str(run),producer_summary=str(run/'eeg_o_summary.json'),candidate=candidate,seeds=seeds,
        source_tables=sources,comparisons=comparisons,
        training_budget=[dict(candidate=r['candidate']['name'],seed=r['seed'],step=r['step'],budget=r['budget'])
                         for r in json.loads((run/'eeg_o_screen_jobs.json').read_text())['jobs']
                         +json.loads((run/'eeg_o_replication_jobs.json').read_text())['jobs']],
        planned_trials=result['planned_trials'],retained_trials=result['retained_trials'],
        input_support=cfg['eeg_o']['raw_input_support'],old_input_support='one electrode, local 2 s raw voltage',
        bits_per_O_second_per_electrode=280,test_used=False,
        aggregation='equal cases within record, equal records within subject, seeds averaged within subject, equal subjects; ratios calculated after averaging',
        inference='paired whole-subject percentile95% interval; validation-selected candidate, fixed scales and selection, no multiplicity correction or independent confirmation')
    _json(output/'summary.json',summary)
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(len(comparisons),3,figsize=(14,7),squeeze=False)
    fig.subplots_adjust(left=.075,right=.985,bottom=.20,top=.82,hspace=.45,wspace=.28)
    titles=['O MSE / zero-prediction MSE (lower)','O centered correlation (higher)','Frozen S + O NMSE (lower)']
    for i,row in enumerate(comparisons):
        names=('old_to_zero_ratio','new_to_zero_ratio'),('old_correlation','centered_correlation'),('old_sum_NMSE','frozen_S_plus_O_NMSE')
        for j,pair in enumerate(names):
            ax=axes[i,j];ax.bar([0,1],[row[k] for k in pair],color=['#888888','#0072B2'],width=.55)
            for seed_index,seed in enumerate(seeds):
                data=[r for r in paired if r['dataset']==row['dataset'] and r['seed']==seed]
                if j==0:vals=[np.mean([r['old_NMSE'] for r in data])/np.mean([r['zero_NMSE'] for r in data]),np.mean([r['NMSE'] for r in data])/np.mean([r['zero_NMSE'] for r in data])]
                else:vals=[np.mean([r[k] for r in data]) for k in pair]
                ax.scatter(np.array([0,1])+(seed_index-1)*.08,vals,color='black',s=20,marker=['o','^','s'][seed_index],zorder=3)
            if j==0:ax.axhline(1,color='#333',ls='--',lw=1,label='Zero prediction')
            ax.set_xticks([0,1],['Original','New']);ax.set_title(titles[j]);ax.grid(axis='y',alpha=.2)
            for x,k in enumerate(pair):ax.annotate(f'{row[k]:.4g}',(x,row[k]),xytext=(0,5),textcoords='offset points',ha='center')
            if j==0:
                short='Single-Trial' if row['dataset']=='eeg_fnirs_single_trial' else 'Simultaneous'
                ax.set_ylabel(f'{short}\n{row["subjects"]} validation subjects')
            ax.margins(y=.20)
            if j==0:ax.set_ylim(0,1.2)
            elif j==1:ax.set_ylim(0,.5)
            else:ax.set_ylim(0,1.2*max(max(r['old_sum_NMSE'],r['frozen_S_plus_O_NMSE']) for r in paired))
    fig.suptitle(f'EEG-O refinement: {candidate} | fixed targets and O bitrate | three optimization seeds',fontsize=15,y=.98)
    fig.text(.075,.90,'Bars: equal-subject means after seed averaging. Circle/triangle/square: seeds 17/29/43 on the same subjects; not confidence intervals.',fontsize=10)
    support='New: retained 90 s EEG reference/filter support; central 20 s observed proxy.' if cfg['eeg_o'].get('architecture')=='projection_gain' else 'New: full 20 s EEG filtering; gain-network context is shorter.'
    fig.text(.075,.115,'Original: single-electrode 2 s input. '+support,fontsize=10)
    fig.text(.075,.075,'O = fixed conditional voltage correction residual. Both hard O decoders use 280 bit/s/electrode; no measured-signal or gain sidecar.',fontsize=10)
    fig.text(.075,.035,'Identical development subjects and targets; input support differs. Paired subject intervals and all source tables are retained in summary.json.',fontsize=10)
    fig.savefig(output/'eeg_o_comparison.png',dpi=250);plt.close(fig)
    return summary
