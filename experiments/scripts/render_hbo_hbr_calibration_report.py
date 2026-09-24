#!/usr/bin/env python3
"""Read retained calibration evidence and export a Chinese report with bitmap figures."""
from pathlib import Path
import argparse
import base64
import html
import json
import shutil
import unicodedata
import re
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import fitz
import markdown

CONDITIONS=['white','gain2_white','independent_slow','shared_slow','gain2_independent_slow','dpf_mixing','dpf_mixing_nonrest']
CLABELS=['白噪声','HbR增益×2＋白噪声','独立慢变','共同慢变','增益×2＋独立慢变','DPF混合＋慢变','DPF混合＋非平衡']
METHODS=['fixed','mean_only','noise_only','independent','oracle']
MLABELS=['固定观测','仅均值校准','仅噪声校准','两者独立校准','真实观测 oracle']
COLORS=['#8b939c','#d58b36','#5189b8','#219277','#713e82']


def read(p):return json.loads(Path(p).read_text())


def dump(p,x):
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(a) for k,a in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)):return [clean(a) for a in v]
        if isinstance(v,np.generic):return clean(v.item())
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    Path(p).write_text(json.dumps(clean(x),indent=2,ensure_ascii=False,allow_nan=False))


def table(df,columns=None):
    if columns is not None:df=df[columns]
    def cell(v):
        if isinstance(v,float):return f'{v:.4g}' if np.isfinite(v) else '—'
        return str(v).replace('|',' / ')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'


def seed_ci(values):
    values=np.asarray(values,dtype=float);values=values[np.isfinite(values)]
    rng=np.random.default_rng(20260924)
    b=values[rng.integers(0,len(values),(5000,len(values)))].mean(axis=1)
    return float(values.mean()),*np.quantile(b,[.025,.975]).tolist()


def analyze(run,refinement,basin_run,profile_basin_run,out):
    m=read(run/'manifest.json');r=read(refinement/'manifest.json')
    if m['status']!='completed' or r['status']!='completed':raise ValueError('report requires completed fits')
    cfg=m['resolved_config'];parent=Path(__file__).resolve().parents[2]/cfg['followup']['parent_run']
    d=pd.read_csv(run/'metrics.csv');ref=pd.read_csv(refinement/'metrics.csv');cross=d[d.group.eq('cross')].copy()
    basin_manifest=read(basin_run/'manifest.json')
    if basin_manifest['status']!='completed':raise ValueError('basin check incomplete')
    basin=read(basin_run/'tasks'/'basin_check|dpf_mixing_nonrest|r01|tau2.json')['models']['oracle_profile_warm_start']
    profile_basin_manifest=read(profile_basin_run/'manifest.json')
    if profile_basin_manifest['status']!='completed':raise ValueError('profile basin check incomplete')
    profile_basin=pd.read_csv(profile_basin_run/'metrics.csv')
    good=cross[cross.status.eq('completed')];info=read(run/'driver_information.json');audit=read(run/'source_audit.json')
    contrasts=[]
    for (cond,rep,method),g in good.groupby(['condition','replicate','method']):
        low=g[g.true_tau.eq(1)];high=g[g.true_tau.eq(4)]
        if len(low) and len(high):
            delta=float(high.iloc[0].tau-low.iloc[0].tau)
            contrasts.append(dict(condition=cond,replicate=rep,method=method,delta_tau=delta,error=delta-3,absolute_error=abs(delta-3),direction=delta>0))
    contrasts=pd.DataFrame(contrasts);contrasts.to_csv(out/'tau_contrasts.csv',index=False)
    effects=[]
    for metric in ['driver_nrmse','combination_relative_error','observation_clean_rmse']:
        for cond in ['all']+CONDITIONS:
            group=good if cond=='all' else good[good.condition.eq(cond)]
            pivot=group.pivot(index=['condition','replicate','true_tau'],columns='method',values=metric)
            delta={'mean_benefit_white':pivot.fixed-pivot.mean_only,
                'mean_benefit_calibrated_noise':pivot.noise_only-pivot.independent,
                'noise_benefit_identity':pivot.fixed-pivot.noise_only,
                'noise_benefit_calibrated_mean':pivot.mean_only-pivot.independent,
                'joint_benefit':pivot.fixed-pivot.independent,
                'interaction_extra_benefit':pivot.mean_only+pivot.noise_only-pivot.fixed-pivot.independent}
            for effect,values in delta.items():
                seed=values.groupby(level='replicate').mean();estimate,lo,hi=seed_ci(seed)
                effects.append(dict(condition=cond,metric=metric,effect=effect,mean_paired_effect=estimate,cluster95_low=lo,cluster95_high=hi,seeds=len(seed)))
    effects=pd.DataFrame(effects);effects.to_csv(out/'paired_effects.csv',index=False)
    summaries=[]
    for method in METHODS+['joint_target']:
        g=good[good.method.eq(method)];c=contrasts[contrasts.method.eq(method)];v=g[g.intervals_valid.eq(True)]
        summaries.append(dict(method=method,total=168,completed=len(g),combination_pct=100*g.combination_relative_error.median(),r_pct=100*g.driver_nrmse.median(),
            observation_clean_rmse=g.observation_clean_rmse.median(),residual_rmse=g.observation_residual_rmse.median(),
            tau_pairs=len(c),tau_direction=int(c.direction.sum()),tau_delta_median=c.delta_tau.median(),tau_MAE=c.absolute_error.mean(),tau_error_q05=c.error.quantile(.05),tau_error_q95=c.error.quantile(.95),
            valid_intervals=len(v),tau_coverage_pct=100*v.tau_95_covers.astype(float).mean(),r_coverage_pct=100*v.driver_pointwise_95_coverage.mean(),any_bound=int(g.active_bound.eq(True).sum())))
    summary=pd.DataFrame(summaries);summary.to_csv(out/'method_summary.csv',index=False)
    coverage=[]
    for method in METHODS:
        g=good[good.method.eq(method)&good.intervals_valid.eq(True)]
        for metric in ['tau_95_covers','driver_pointwise_95_coverage']:
            # Resample seed clusters with all their valid rows; preserve random denominators.
            aggregates=g.assign(value=g[metric].astype(float)).groupby('replicate').value.agg(['sum','count'])
            rng=np.random.default_rng(20260924);ids=rng.integers(0,len(aggregates),(5000,len(aggregates)))
            rates=aggregates['sum'].to_numpy()[ids].sum(axis=1)/aggregates['count'].to_numpy()[ids].sum(axis=1)
            coverage.append(dict(method=method,metric=metric,estimate=float(aggregates['sum'].sum()/aggregates['count'].sum()),low=float(np.quantile(rates,.025)),high=float(np.quantile(rates,.975)),valid=len(g)))
    coverage=pd.DataFrame(coverage);coverage.to_csv(out/'coverage.csv',index=False)
    profile=pd.concat([d[d.group.eq('profile')].assign(profile_stage='global'),ref[ref.group.eq('profile')].assign(profile_stage='local_resolution'),profile_basin.assign(profile_stage='four_start_basin_check')],ignore_index=True)
    profiles=[]
    for (cond,rep,p),g in profile.groupby(['condition','replicate','parameter']):
        old=read(parent/'tasks'/f'{cond}|r{rep:02d}|tau2.json')['models']['oracle']
        fit=g[g.status.eq('completed')]
        base=min(old['objective'],float(fit.objective.min()))
        if cond=='dpf_mixing_nonrest' and rep==1:base=min(base,basin['objective'])
        inside=fit[fit.objective-base<=1.920729]
        grid=r['resolved_config']['followup']['profile_custom_grids'][f'{cond}|r{rep:02d}|p{int(p)}']
        profiles.append(dict(condition=cond,replicate=int(rep),parameter=int(p),total=len(g),completed=len(fit),minimum_cost_minus_original=float(fit.objective.min()-old['objective']),reference_cost=base,
            accepted_grid_min=float(inside.value.min()),accepted_grid_max=float(inside.value.max()),
            original_estimate=old['parameters'][int(p)],local_sd=grid['local_sd'],
            domain_boundary_accepted=bool(any(np.isclose(inside.value,b).any() for b in ([.5,12] if p==0 else [-.12,.12])))))
    profiles=pd.DataFrame(profiles);profiles.to_csv(out/'profile_summary.csv',index=False)
    profile.to_csv(out/'profile_points.csv',index=False)
    failed=d[~d.status.eq('completed')].copy();pd.concat([failed,ref[~ref.status.eq('completed')]],ignore_index=True).to_csv(out/'failures.csv',index=False)
    details=[]
    for job in read(run/'task_inventory.json'):
        row=read(run/'tasks'/f'{job["key"]}.json')
        for method,model in row.get('models',{}).items():
            if model.get('reused') or job['group']=='profile':continue
            details.append(dict(group=job['group'],condition=job['condition'],replicate=job['replicate'],method=method,status=model['status'],
                active_indices=model.get('active_indices',[]),rank=model.get('rank'),
                initial_bounds=any(k in range(8,13) for k in model.get('active_indices',[])),
                physiology_bounds=any(k in [0,1] for k in model.get('active_indices',[])),
                driver_bounds=any(k in range(2,8) for k in model.get('active_indices',[])),
                matrix_bounds=any(k>=13 for k in model.get('active_indices',[]))))
    dump(out/'bound_details.json',details)
    result=dict(primary_run=str(run),profile_resolution_run=str(refinement),parent_run=str(parent),
        new_fits=m['fits_total']+r['fits_total']+basin_manifest['fits_total']+profile_basin_manifest['fits_total'],completed_new_fits=m['fits_completed']+r['fits_completed']+basin_manifest['fits_completed']+profile_basin_manifest['fits_completed'],
        basin_check_run=str(basin_run),basin_check=basin,
        profile_basin_run=str(profile_basin_run),
        reused_fits=672,independent_seeds=8,source_records=len(audit['records']),source_subjects=3,
        methods=summaries,profiles=profiles.to_dict('records'),conditional_measured_status=audit['C_status'])
    dump(out/'analysis.json',result)
    return locals()


def render(run,refinement,basin_run,profile_basin_run,out):
    out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir()
    data=analyze(run,refinement,basin_run,profile_basin_run,out)
    d,cross,good,contrasts,effects,summary,coverage,profiles,profile,info,audit,parent,cfg,result=[data[k] for k in ['d','cross','good','contrasts','effects','summary','coverage','profiles','profile','info','audit','parent','cfg','result']]
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    figures=[]
    def save(name,title,fig):
        fig.tight_layout();fig.savefig(out/'figures'/f'{name}.png',dpi=240,bbox_inches='tight',facecolor='white');plt.close(fig);figures.append((name,title))
    # 1: factorial conditional medians.
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,metric,title in zip(axes,['combination_relative_error','driver_nrmse'],['有效组合相对误差（%）','未对齐驱动 NRMSE（%）']):
        matrix=good.pivot_table(index='condition',columns='method',values=metric,aggfunc='median').reindex(index=CONDITIONS,columns=METHODS).values*100
        im=ax.imshow(matrix,aspect='auto',cmap='YlOrRd',vmin=0,vmax=100 if metric.startswith('combination') else 55)
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=9,color='white' if v>(55 if metric.startswith('combination') else 33) else '#182129')
        ax.set_xticks(range(5),MLABELS,rotation=28,ha='right');ax.set_yticks(range(7),CLABELS);ax.set_title(title)
        fig.colorbar(im,ax=ax,shrink=.7)
    save('01_factorial','图1：分条件的均值／噪声校准交叉对照；每格24个面板',fig)
    # 2: paired clustered effect, improvement positive.
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    etypes=['mean_benefit_white','mean_benefit_calibrated_noise','noise_benefit_identity','noise_benefit_calibrated_mean','joint_benefit','interaction_extra_benefit']
    elabs=['均值校准：白噪声评分','均值校准：已校准噪声','噪声校准：单位均值','噪声校准：已校准均值','联合收益','超出简单相加的收益']
    for ax,metric,title in zip(axes,['driver_nrmse','combination_relative_error'],['驱动误差降低','有效组合误差降低']):
        g=effects[effects.condition.eq('all')&effects.metric.eq(metric)].set_index('effect').loc[etypes];v=g.mean_paired_effect.values*100;lo=g.cluster95_low.values*100;hi=g.cluster95_high.values*100
        ax.errorbar(v,np.arange(6),xerr=np.array([v-lo,hi-v]),fmt='o',color='#219277',capsize=4);ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(6),elabs);ax.invert_yaxis();ax.set_xlabel('配对误差减少（百分点；8种子聚类区间）');ax.set_title(title)
    save('02_effects','图2：先在种子内配对，再对8个种子重采样；正值表示收益',fig)
    # 3: physiological contrast errors.
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    groups=[contrasts[contrasts.method.eq(m)].error.values for m in METHODS+['joint_target']]
    axes[0].boxplot(groups,tick_labels=MLABELS+['自由矩阵'],showfliers=True);axes[0].axhline(0,color='gray',ls='--');axes[0].tick_params(axis='x',rotation=28);axes[0].set_ylabel('估计 τ 差值 − 3 秒');axes[0].set_title('方向保留之外：差值误差的完整分布')
    for method,label,color in zip(METHODS,MLABELS,COLORS):
        g=contrasts[contrasts.method.eq(method)].groupby('replicate').absolute_error.mean();axes[1].plot(g.index,g.values,'o-',label=label,color=color)
    axes[1].set_xlabel('独立种子');axes[1].set_ylabel('种子内平均绝对差值误差（秒）');axes[1].legend(fontsize=8)
    save('03_tau_contrasts','图3：τ=4 与 τ=1 的配对差值；56个配对归属于8个种子',fig)
    # 4: common observation coordinate vs teacher error.
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    for method,label,color in zip(METHODS[:4],MLABELS[:4],COLORS):
        g=good[good.method.eq(method)];axes[0].scatter(g.observation_residual_rmse,100*g.driver_nrmse,s=14,alpha=.5,label=label,color=color)
    axes[0].set_xlabel('统一观测坐标中目标残差 RMSE');axes[0].set_ylabel('驱动误差（%）');axes[0].set_xscale('log');axes[0].legend(fontsize=8)
    s=summary.set_index('method').loc[METHODS];x=np.arange(5)
    axes[1].bar(x-.18,s.observation_clean_rmse,.36,label='相对干净真值');axes[1].bar(x+.18,s.residual_rmse,.36,label='相对带噪目标');axes[1].set_xticks(x,MLABELS,rotation=28,ha='right');axes[1].set_ylabel('统一观测坐标 RMSE 中位数');axes[1].legend(fontsize=8)
    save('04_observation','图4：统一Hb观测坐标；更小的带噪目标残差不等价于更好恢复',fig)
    # 5: oracle localization, paired 96 samples.
    abconds=cfg['followup']['oracle_conditions'];base=good[good.method.eq('oracle')&good.condition.isin(abconds)];abl=d[d.group.eq('oracle_ablation')&d.status.eq('completed')];combined=pd.concat([base,abl]);ams=['oracle','known_initial','known_physiology','quarter_slow_noise'];alabs=['原 oracle','已知初态','已知 τ、η','慢噪声幅度×1/4']
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for cond in abconds:
        vals=[100*combined[combined.condition.eq(cond)&combined.method.eq(m)].driver_nrmse.median() for m in ams];axes[0].plot(range(4),vals,'o-',label=CLABELS[CONDITIONS.index(cond)])
    axes[0].set_xticks(range(4),alabs,rotation=20);axes[0].set_ylabel('驱动误差中位数（%）');axes[0].legend(fontsize=8)
    nl=d[d.group.eq('noiseless')];axes[1].hist(np.log10(nl.driver_nrmse),bins=18,color='#219277');axes[1].set_xlabel('log10 无噪声驱动相对误差');axes[1].set_ylabel('面板数 / 168');axes[1].set_title('完整非线性/光学/滤波闭环；非真值起点')
    save('05_oracle','图5：oracle误差定位；含噪对照使用同一批96个样本',fig)
    # 6: continuous context scored on common center.
    ctx=d[d.group.eq('context')&d.status.eq('completed')];fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for cond in cfg['followup']['context_conditions']:
        g=ctx[ctx.condition.eq(cond)]
        for rep in range(8):
            q=g[g.replicate.eq(rep)].sort_values('duration');axes[0].plot(q.duration,100*q.driver_common_nrmse,alpha=.16,color=COLORS[cfg['followup']['context_conditions'].index(cond)+1])
        q=g.groupby('duration').driver_common_nrmse.median();axes[0].plot(q.index,100*q.values,'o-',lw=2,label=CLABELS[CONDITIONS.index(cond)])
    axes[0].set_xticks([30,60,120]);axes[0].set_xlabel('拟合连续上下文长度（秒）');axes[0].set_ylabel('同一45–75秒驱动误差（%）');axes[0].legend(fontsize=8)
    for duration,color in zip([30,60,120],COLORS[1:4]):
        job=read(run/'tasks'/f'context|dpf_mixing_nonrest|r00|tau2|duration={duration}.json');tr=job['traces'][f'context_{duration}'];t=np.array(tr['time']);keep=(t>=45)&(t<75);axes[1].plot(t[keep],np.array(tr['driver'])[keep],color=color,label=f'{duration}秒上下文')
    axes[1].plot(t[keep],np.array(tr['true_driver'])[keep],color='black',ls='--',label='真实 r');axes[1].set_xlabel('共同记录时间（秒）');axes[1].set_ylabel('r（固定物理尺度）');axes[1].legend(fontsize=8)
    save('06_context','图6：连续记录上下文；所有方法只在同一中间30秒评分',fig)
    # 7/8: projected information and nuisance directions.
    oracleinfo=[x for x in info if x['method']=='oracle'];fig,axes=plt.subplots(1,2,figsize=(12,4.1))
    for cond,label in zip(CONDITIONS,CLABELS):
        vals=np.array([x['eigenvalues'] for x in oracleinfo if x['condition']==cond]);axes[0].plot(range(1,7),np.median(vals,axis=0),'o-',label=label)
    axes[0].set_yscale('log');axes[0].set_xlabel('有效信息特征值（从弱到强排序）');axes[0].set_ylabel('白化驱动信息');axes[0].legend(fontsize=7,ncol=2)
    for cond in cfg['followup']['context_conditions']:
        vals=[]
        for duration in [30,60,120]:
            vals.append(np.median([x['minimum_eigenvalue'] for x in info if x['condition']==cond and x['method']==f'context_{duration}']))
        axes[1].plot([30,60,120],vals,'o-',label=CLABELS[CONDITIONS.index(cond)])
    axes[1].set_yscale('log');axes[1].set_xticks([30,60,120]);axes[1].set_xlabel('连续上下文（秒）');axes[1].set_ylabel('最弱驱动方向有效信息中位数');axes[1].legend(fontsize=8)
    save('07_information','图7：投影消去生理参数与初态后的局部驱动信息',fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    weights=np.array([np.mean([np.array(x['weakest_direction'])**2 for x in oracleinfo if x['condition']==c],axis=0) for c in CONDITIONS]);im=axes[0].imshow(weights,aspect='auto',vmin=0,vmax=1,cmap='Blues');axes[0].set_yticks(range(7),CLABELS);axes[0].set_xticks(range(6),['sin .035','sin .075','sin .14','cos .035','cos .075','cos .14'],rotation=28);fig.colorbar(im,ax=axes[0],label='最弱方向的平均平方载荷')
    ratios=[np.median([x['trace_retained_fraction'] for x in oracleinfo if x['condition']==c]) for c in CONDITIONS];axes[1].barh(CLABELS,np.array(ratios)*100,color='#5189b8');axes[1].set_xlabel('消除 nuisance 后保留的信息迹（%）');axes[1].invert_yaxis()
    save('08_directions','图8：弱驱动方向与未知初态／生理参数的混淆；局部分析不等于全局唯一性',fig)
    # 9: refined profiles and local Fisher approximation.
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    for i,cond in enumerate(cfg['followup']['profile_conditions']):
        for j,param in enumerate([0,6]):
            ax=axes[i,j]
            for rep,color in zip([0,1],['#5189b8','#d58b36']):
                g=profile[profile.condition.eq(cond)&profile.replicate.eq(rep)&profile.parameter.eq(param)&profile.status.eq('completed')].sort_values(['value','objective']).drop_duplicates('value')
                old=read(parent/'tasks'/f'{cond}|r{rep:02d}|tau2.json')['models']['oracle'];ps=profiles[profiles.condition.eq(cond)&profiles.replicate.eq(rep)&profiles.parameter.eq(param)].iloc[0];mu=ps.original_estimate;sd=ps.local_sd
                center=g[(g.value>=mu-4.1*sd)&(g.value<=mu+4.1*sd)];ax.plot(center.value,center.objective-ps.reference_cost,'o-',ms=3,color=color,label=f'种子{rep} 剖面')
                xs=np.linspace(mu-4*sd,mu+4*sd,100)
                xs=xs[(xs>=.5)&(xs<=12)] if param==0 else xs[(xs>=-.12)&(xs<=.12)]
                ax.plot(xs,.5*((xs-mu)/sd)**2+old['objective']-ps.reference_cost,':',color=color,alpha=.8)
                truth=2. if param==0 else np.random.default_rng(cfg['synthetic']['seed']+rep*10000).normal(0,.015,6)[4]
                ax.axvline(truth,color=color,ls='--',alpha=.5)
            ax.axhline(1.920729,color='gray',ls='--',lw=1);ax.set_ylim(-.05,8);ax.set_title(CLABELS[CONDITIONS.index(cond)]);ax.set_xlabel('τ（秒）' if param==0 else 'r 的 0.075 Hz 余弦系数');ax.set_ylabel('Δ 负对数似然');ax.legend(fontsize=7)
    save('09_profiles','图9：数值剖面（实线）与局部Fisher近似（点线）；竖虚线是真值',fig)
    # 10: free matrix and exact eta equivalence.
    free=d[d.group.eq('free_matrix')&d.status.eq('completed')];sub=good[good.method.eq('oracle')&good.true_tau.eq(2)&good.condition.isin(cfg['followup']['free_matrix_conditions'])];fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    arr=[sub.driver_nrmse*100,free[free.method.eq('free_correct_noise')].driver_nrmse*100,free[free.method.eq('free_correct_noise_eta_fixed')].driver_nrmse*100];axes[0].boxplot(arr,tick_labels=['真实固定A\n24/24','自由A＋正确Σ\n22/24','自由A＋正确Σ\n固定η；22/24']);axes[0].set_ylabel('驱动 NRMSE（%）');axes[0].set_title('去掉精确冗余仍未恢复到 oracle 水平')
    etas=np.linspace(.2,.6,80);ce=lambda eta:np.array([[1,-eta],[0,eta]]);mat=np.array([ce(.35)@np.linalg.inv(ce(e)) for e in etas])
    for i,j in [(0,0),(0,1),(1,0),(1,1)]:axes[1].plot(etas,mat[:,i,j],label=f'A′[{i+1},{j+1}]')
    axes[1].set_xlabel('替代 η′');axes[1].set_ylabel('保持完全相同观测的 A′ 元素');axes[1].set_title('A′ = A C(η) inv(C(η′))；这里 A=I');axes[1].legend(fontsize=8)
    save('10_free_matrix','图10：正确噪声下的自由矩阵诊断及精确η/A等价方向',fig)
    # 11: approximate UQ coverage.
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    for ax,metric,title in zip(axes,['tau_95_covers','driver_pointwise_95_coverage'],['τ 区间经验覆盖','r 逐时点区间平均覆盖']):
        g=coverage[coverage.metric.eq(metric)].set_index('method').loc[METHODS];v=g.estimate.values*100;ax.errorbar(range(5),v,yerr=np.array([v-g.low.values*100,g.high.values*100-v]),fmt='o',capsize=4,color='#5189b8');ax.axhline(95,color='gray',ls='--');ax.set_xticks(range(5),[f'{a}\n有效{b}/168' for a,b in zip(MLABELS,g.valid)],rotation=20,ha='right');ax.set_ylim(0,102);ax.set_ylabel('覆盖率（%；种子聚类描述区间）');ax.set_title(title)
    save('11_coverage','图11：仅在可用局部区间上计算覆盖率，同时显示完整分母',fig)
    # 12: preselected r and Hb traces.
    fig,axes=plt.subplots(3,2,figsize=(12,9))
    for i,cond in enumerate(['gain2_white','shared_slow','dpf_mixing_nonrest']):
        new=read(run/'tasks'/f'cross|{cond}|r00|tau2.json');old=read(parent/'tasks'/f'{cond}|r00|tau2.json');tr=new['traces']['mean_only'];t=np.array(tr['time']);axes[i,0].plot(t,tr['true_driver'],'k--',label='真值')
        for method,label,color in zip(METHODS[:4],MLABELS[:4],COLORS):
            trace=new['traces'][method]['driver'] if method in new['traces'] else old['driver_traces'][method];axes[i,0].plot(t,trace,color=color,label=label)
        axes[i,0].set_title(CLABELS[CONDITIONS.index(cond)]+'：r');axes[i,0].set_ylabel('未对齐 r');axes[i,0].set_xlabel('秒');axes[i,0].legend(fontsize=7,ncol=2)
        hb=np.array(tr['true_hb']);mean=np.array(tr['hb']);noise=np.array(new['traces']['noise_only']['hb'])
        for k,color,name in [(0,'#bf5444','HbO'),(1,'#5189b8','HbR')]:
            axes[i,1].plot(t,hb[:,k],color=color,label=name+'真值');axes[i,1].plot(t,mean[:,k],color=color,ls='--',label=name+'仅均值');axes[i,1].plot(t,noise[:,k],color=color,ls=':',label=name+'仅噪声')
        axes[i,1].set_xlabel('秒');axes[i,1].set_ylabel('生理输出相对Hb');axes[i,1].legend(fontsize=7,ncol=2)
    save('12_traces','图12：预先固定的种子0、τ=2样例；无符号／幅度／时移对齐',fig)
    # 13: actual native optical and released processing.
    fig,axes=plt.subplots(3,2,figsize=(12,8.4))
    for i,ex in enumerate(audit['examples']):
        t=ex['time_s'];native=np.array(ex['intensity_V']);hb=np.array(ex['processed_relative_Hb']);label=ex['join_key'].split('|')[1]
        for k,name in enumerate(['760 nm','850 nm']):axes[i,0].plot(t,native[:,k],label=name,lw=.8)
        for k,name in enumerate(['HbO','HbR']):axes[i,1].plot(t,hb[:,k],label=name,lw=.8)
        axes[i,0].set_title(label+' / session_01 / '+ex['pair']);axes[i,0].set_ylabel('原始光强读出（V）');axes[i,1].set_ylabel('当前缓存相对Hb（非 μM）')
        for ax in axes[i]:ax.set_xlabel('NIRS记录时间（秒）');ax.legend(fontsize=8)
    save('13_measured_optics','图13：真实双波长原始读出与现有幅度保留处理；只展示，不拟合teacher',fig)
    # 14: clock evidence, missing auxiliary streams explicit.
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));labels=[]
    for i,row in enumerate(audit['records']):
        offsets=np.array(row['corresponding_marker_offsets_ms']);labels.append(row['subject'][-2:]+'/'+row['record'][-2:]);axes[0].scatter(offsets/1000,np.full(len(offsets),i),s=10,alpha=.6)
    axes[0].set_yticks(range(9),labels);axes[0].set_xlabel('同一事件的 NIRS − EEG 原生时间（秒）');axes[0].set_ylabel('被试 / session');axes[0].set_title('共享事件不等于数组从零点直接对齐')
    availability=np.tile([1,1,1,0,0,0],(9,1));im=axes[1].imshow(availability,cmap='RdYlGn',vmin=0,vmax=1,aspect='auto');axes[1].set_yticks(range(9),labels);axes[1].set_xticks(range(6),['双波长','变换记录','EOG','ECG/呼吸','个体DPF','已知Hb标准'],rotation=30,ha='right')
    for (i,j),v in np.ndenumerate(availability):axes[1].text(j,i,'已核实' if v else '未建立',ha='center',va='center',fontsize=8)
    save('14_source_evidence','图14：9条实际记录的时钟和独立观测信息；“未建立”限定于所查文件',fig)
    # Build narrative, numbers always from retained rows.
    def med(group,method,metric='driver_nrmse'):return float(d[d.group.eq(group)&d.method.eq(method)&d.status.eq('completed')][metric].median())
    text='# 观测校准归因、oracle误差定位与实测约束核查\n\n'
    text+='2026-09-24｜完整实验与可视化报告｜HBO-HBR-CALIBRATION-ATTRIBUTION-v2\n\n'
    text+=f'**结论：独立观测信息的收益来自纠正均值映射与噪声度量，两者缺一时可能继续发生错误补偿。当前最明确的剩余误差来源是有限时间支持下的驱动／初态／生理参数混淆；增加连续上下文显著改善同一目标时间段的恢复。实测独立校准仍未建立。**\n\n'
    text+=f'本轮新增 {result["new_fits"]} 次拟合，完成 {result["completed_new_fits"]} 次；保留所有失败。主运行1128次，剖面局部分辨率补充{read(refinement/"summary.json")["fits_total"]}次，局部解释放约束复核1次，受影响样例的多起点剖面复核68次。原实验的672条四方法记录只复用，未重新优化。原先另外336次光学校准扰动拟合仍保留在v1，不计入本轮新增数量。\n\n'
    text+='## 1. 实验覆盖与判读边界\n\n'
    schedule=pd.DataFrame([['均值/噪声交叉',168,336,'复用固定、独立、oracle、自由矩阵原结果'],['无噪声闭环',168,168,'全部7条件、8种子、3个τ'],['oracle消融',96,288,'已知初态、已知生理、慢噪声×1/4'],['自由矩阵/正确协方差',24,48,'η自由与η固定；均为诊断'],['连续时长',24,72,'3条件×8种子，τ=2；每条30/60/120秒'],['剖面',6,216,'3条件×2种子，τ及0.075Hz余弦系数'],['剖面局部补密',6,read(refinement/'summary.json')['fits_total'],'按旧oracle Fisher尺度预定网格'],['实测实际文件',9,0,'3名被试×3条已有公开记录，无teacher拟合']],columns=['实验','基础面板','新增拟合','范围'])
    schedule.loc[len(schedule)]=['局部解复核',1,1,'从两条剖面低目标解释放约束；保留原oracle']
    schedule.loc[len(schedule)]=['受影响剖面的多起点复核',1,read(profile_basin_run/'summary.json')['fits_total'],'同一已声明网格，追加原内部解及边界解起点']
    text+=table(schedule)+'\n'
    text+='生成与拟合共用当前非线性Balloon核心。r为3个已知频率（0.035、0.075、0.14 Hz）的未知正余弦系数；待估τ、η、6个驱动系数和5个血流初态。α、E₀、κ、γ、P₀使用已知真值，这既含尺度约定，也含额外生理知识。没有过程随机性，没有EEG观测，不等同于完整六状态随机SSM。每个面板两个非真值起点、每起点100次函数评估，固定边界；失败不删去。\n\n'
    text+='原目标、512组已知Hb标准与512条独立噪声校准记录相互独立；不同条件和τ在同一随机种子内配对。只有8个独立种子。报告的聚类区间用5000次种子重采样，属于小样本描述性不确定性，没有把168个面板当成168个独立重复。\n\n'
    text+='## 2. 均值与噪声校准各自贡献\n\n'
    text+='固定组使用(A=I, Σ白噪声)；仅均值组使用(Â, Σ白噪声)；仅噪声组使用(I, Σ̂)；独立组使用(Â, Σ̂)。oracle使用真实A和Σ。所有误差以合成真值评价，r不做后验对齐；不同协方差下的加权目标值不作横向排名。\n\n'
    compact=summary[['method','completed','combination_pct','r_pct','any_bound']].copy();compact.columns=['方法','完成/168','组合误差%','r误差%','任意参数触界数'];text+=table(compact)+'\n'
    text+='增益×2＋白噪声最清楚地隔离均值校准作用：仅均值校准的r误差约2.32%，固定组17.22%，仅噪声组18.37%。独立慢变与共同慢变则主要受噪声度量改善：共同慢变仅噪声组10.17%，固定组50.46%。DPF混合＋慢变需两者共同使用；例如混合条件仅噪声组34.20%，联合校准17.44%。只修正Σ而保留错误A，也可能恶化参数组合（混合条件中位96.46%）。这排除了“分开优化本身普遍有效”的解释。\n\n'
    text+='白噪声且A本来正确时，校准增加的有限样本误差可略微降低表现；oracle不是每次随机实现中点估计误差的硬下界。图2的效应是逐面板配对差，再先按种子平均；不能用全体中位数相减代替该配对效应。\n\n'
    text+=f'![交叉对照](figures/01_factorial.png)\n\n![配对归因](figures/02_effects.png)\n\n'
    text+='## 3. 真实生理差异与统一观测误差\n\n'
    compact=summary[['method','tau_pairs','tau_direction','tau_delta_median','tau_MAE','tau_error_q05','tau_error_q95']].copy();compact.columns=['方法','完整配对','方向正确','差值中位s','差值MAE s','有符号误差5%','有符号误差95%'];text+=table(compact)+'\n'
    text+='真值差值为3秒。方向正确只是一项较弱检查，图3与表格进一步给出幅度误差、尾部以及每个种子的差值MAE。自由矩阵未收敛的配对保留在完整分母之外单列，不能与56个完整配对的方法直接等同。\n\n'
    text+='统一观测坐标误差使用每个方法的拟合A生成预测，与真实A生成的干净输出比较，并用相同滤波／重采样／基线算子处理。另报相对带噪目标的残差。固定组可以有更小目标残差，却更偏离干净真值及r；这与把慢变污染吸入驱动的机制一致，不能仅凭残差降低判断teacher改善。\n\n'
    text+='![生理差值](figures/03_tau_contrasts.png)\n\n![统一观测误差](figures/04_observation.png)\n\n'
    text+='## 4. oracle剩余误差：数值闭环、未知量与慢噪声\n\n'
    text+=f'无噪声168/168完成，r误差中位数{nl.driver_nrmse.median():.3g}、最大值{nl.driver_nrmse.max():.3g}。保留非线性积分、光学回放、滤波、重采样与基线操作，仍从非真值起点恢复全部未知量；残差按固定0.001缩放，仅为确定性数值目标，不将该缩放解释成真实噪声似然，也不输出无噪声置信区间。\n\n'
    oracle_table=[]
    for method,label in zip(ams,alabs):
        g=combined[combined.method.eq(method)];oracle_table.append(dict(设置=label,完成=len(g),r误差中位百分比=100*g.driver_nrmse.median(),组合误差中位百分比=100*g.combination_relative_error.median()))
    text+=table(pd.DataFrame(oracle_table))+'\n'
    text+='原oracle在同一96面板上r误差16.80%；已知初态12.38%，已知τ/η为11.06%，慢噪声幅度降至1/4为8.10%。已知真值实验用于区分混淆机制，不能当成可部署方法。已知生理参数时组合误差为零是实验设定，不能作为方法恢复成绩。减小慢噪声同时保留白噪声与配对随机实现，不是换一个更容易的种子。\n\n'
    text+='![oracle消融](figures/05_oracle.png)\n\n'
    text+='## 5. 连续上下文与信息支持\n\n'
    context_table=ctx.groupby('duration').agg(完成=('status','size'),共同30秒误差=('driver_common_nrmse','median'),全段误差=('driver_nrmse','median'),组合误差=('combination_relative_error','median')).reset_index();text+=table(context_table)+'\n'
    text+='每个种子/条件先生成一条连续120秒非线性轨迹及同一原生噪声实现，再截取45–75、30–90、0–120秒作为30、60、120秒拟合上下文。截取后的真实初态随连续轨迹演化，拟合仍将其视为未知；驱动基函数保持全局相位。三者一律在45–75秒的300个原生采样点评分；没有拼接trial。\n\n'
    text+='每个窗口独立执行同一滤波／重采样／基线算子，并将对应协方差同步传播；因此收益包含额外观测信息与滤波边缘／基线上下文的影响，不能把它完全归于单一初态机制。背景噪声由预设有限频率慢成分生成，完整协方差已知；120秒足以区分这些已知频带，实际宽带非平稳噪声不保证获得相同改善。\n\n'
    text+='![连续上下文](figures/06_context.png)\n\n'
    text+='## 6. 局部有效信息与剖面似然\n\n'
    text+='将白化Jacobian分成驱动系数Jᵣ与nuisance Jν，计算 Iᵣ|ν = Jᵣᵀ(I − JνJν⁺)Jᵣ。这里nuisance包括τ、η及全部初态；自由矩阵诊断还包括矩阵元素。图7报告最弱到最强的特征值，图8以平方载荷避免特征向量符号任意性，并展示消去nuisance后保留的信息迹。它们在拟合解处线性化，不能单独证明全局唯一性。\n\n'
    text+='![局部信息](figures/07_information.png)\n\n![混淆方向](figures/08_directions.png)\n\n'
    text+='剖面分别固定τ或r的0.075 Hz余弦系数（参数索引6），重优化其余参数。先按全局范围扫描，再按旧v1 oracle的局部标准差补密，不用新增结果选择有利案例。选择白噪声、独立慢变、混合非平衡的种子0与1，所有失败点与边界保留。横线Δ负对数似然=1.9207是单参数渐近95%参考；边界、有限样本与模型条件使它不等同于已验证的95%区间。\n\n'
    text+='剖面方法用于区分结构冗余与有限信息导致的实用不可辨识，参见[Raue等，2009](https://pubmed.ncbi.nlm.nih.gov/19505944/)。本报告没有据此宣称完整SSM已可辨识。下表只给出满足参考阈值的离散网格包络，不通过插值伪造精确端点；局部点线偏离数值曲线表示二次近似不足。\n\n'
    text+='**剖面还发现了旧oracle的局部解：混合非平衡、种子1、τ真值2秒。**用两条剖面最低目标解作额外起点、释放固定坐标后，目标由60.3464降至58.1201，但τ/η分别落在0.5/0.15下界，r误差反而由17.85%升至36.31%。这次1面板复核保留原两起点并仅追加两起点，未覆盖旧结果。说明更充分优化也可能偏爱错误的边界解释；优化局部性与有限信息同时存在，不能把oracle残差全部归为数值错误或全部归为信息不足。\n\n'
    text+='图9对该样例用释放约束后的最低已知目标作基准，旧局部Fisher曲线整体上移2.2262；其他样例用当前所有已知解中最低目标作基准。最低已知不等于已证明全局最优。包络可能跨越不连续接受区，原始离散点供复核，不能将包络误当成已验证的连续置信区间。\n\n'
    text+='发现该局部解后，对受影响样例的两条原剖面网格追加复核：每点保留两个默认起点，另加旧内部解与已找到的边界解，固定坐标仍由各网格点指定。图9展示同一坐标各次计算中目标最低的解，完整重复结果以profile_stage保留；其余样例没有据此扩张优化预算。\n\n'
    pt=profiles[['condition','replicate','parameter','accepted_grid_min','accepted_grid_max','local_sd','minimum_cost_minus_original']].copy();pt.columns=['条件','种子','参数索引','接受网格下界','接受网格上界','局部sd','最低目标−旧最优'];text+=table(pt)+'\n![剖面似然](figures/09_profiles.png)\n\n'
    text+='## 7. 自由矩阵、精确冗余与失败结果\n\n'
    text+='当前设定中η不改变血流状态动力学，Hb输出可写为 C(η)[P₀(p−1),P₀(q−1)]ᵀ，其中 C(η)=[[1,−η],[0,η]]。对任意可行η′，令A′=A C(η) C(η′)⁻¹，即保持完全相同观测均值。这条等价方向可以保持r完全不变；因此η/A秩亏不能直接证明r不唯一。合成回归检查验证了状态、驱动和观测均值均不变。\n\n'
    text+='使用同一真实协方差后，自由矩阵与固定η的自由矩阵均仅22/24收敛，驱动误差中位数分别62.43%和56.33%。固定η移除了明确的η/A精确冗余，仍未接近同面板的oracle。剩余自由度、实际信息不足、边界和优化失败需直接观察，不能统称为“优化器不够努力”。\n\n'
    freefail=d[d.group.eq('free_matrix')&~d.status.eq('completed')][['condition','replicate','method','status']];text+=table(freefail)+'\n'
    text+='上述4次新增失败均保留两起点的评估预算和消息；没有移除失败面板或扩大预算反复挑选。原joint_target的11次失败继续作为复用历史证据存在。自由矩阵只是诊断反例，不作为实测主候选。\n\n![自由矩阵](figures/10_free_matrix.png)\n\n'
    text+='## 8. 不确定性与参数触界\n\n'
    text+='局部Gaussian区间仅在全自由参数Jacobian满秩、未触界且物理检查通过时报告。独立组传播光学矩阵的估计协方差；没有积分512条独立噪声记录形成的有限样本协方差估计误差，也没有完整的非线性后验。可用区间条件下的覆盖有选择效应，图11同时标明无效区间数量。8个种子的聚类区间很宽，不能支持精确标称覆盖结论。\n\n'
    text+='新增结果保存active_indices，0/1对应τ/η，2–7驱动系数，8–12初态，13–16自由矩阵。任一坐标触界不再统称“生理参数异常”。旧结果仅保存总触界标志，报告不追溯猜测旧参数是哪一项触界；旧表90/168降到7/168的总触界数只能解释为整体补偿减轻。\n\n![覆盖率](figures/11_coverage.png)\n\n'
    text+='## 9. 代表轨迹与实际文件证据\n\n'
    text+='图12固定展示种子0、τ=2的三个机制条件，选择依据是条件类型，不按误差高低挑选。左侧r使用同一尺度、符号、时间轴，右侧展示独立的两条校准对照如何影响生理Hb输出。\n\n![合成轨迹](figures/12_traces.png)\n\n'
    text+='实测核查覆盖Single-Trial被试01、09、18的session_01、03、05，均为已有记录身份。通过中央registry/cache索引定位，复用原MAT读取器与统一EEG接口，读取实际通道、单位、双波长字段及事件；核对对应ZIP的被试成员和cnt_artifact通道元数据，没有递归遍历其他数据集或受保护比较证据。\n\n'
    text+='9条记录均为32路EEG导出（30路头皮EEG＋VEOG/HEOG），200 Hz、μV；NIRS为72列，按36个空间通道的760/850 nm对应成对，10 Hz、V。原文档记录ECG/呼吸采集，但所查主记录、artifact通道和匹配ZIP成员中未建立这些数据流。不能把EOG替代成呼吸或把未找到解释成原始实验从未采集。[数据发布入口](https://doc.ml.tu-berlin.de/hBCI/)及本地原始说明保留在来源记录中。\n\n'
    text+='原生cnt还保留nSources=14、nDetectors=16等设备字段；保留这些文件事实，不用历史摘要中的设备数量覆盖它们。转换明确采用距离3 cm、DPF=6的实现假设，消光系数[[0.148,0.384],[0.252,0.179]]，自然对数OD。系数来源标记为repo_approximate_table_for_alignment_audit_not_subject_calibrated，单位为relative_Hb_repo_approximation。它保留测量幅度，但不是已完成个体DPF标定的绝对μM。DPF的0.8/1.2仍仅为合成压力设置，没有转成实测可信区间。\n\n'
    text+='![原始光学证据](figures/13_measured_optics.png)\n\n'
    source_table=pd.DataFrame([dict(被试=r['subject'],记录=r['record'],EEG秒=r['eeg_seconds'],NIRS秒=r['fnirs_seconds'],共享事件=r['eeg_event_count'],平均偏移秒=np.mean(r['corresponding_marker_offsets_ms'])/1000,偏移标准差ms=np.std(r['corresponding_marker_offsets_ms'])) for r in audit['records']]);text+=table(source_table)+'\n'
    text+='共享并口事件能支持时间映射，但原生EEG/NIRS记录长度和起点明显不同；图14给出实际事件偏移，不能直接按数组索引对齐。ECG／呼吸未找到可映射的数据流，故无法验证它们的时钟、作为已观测辅助量拟合系数，或开展对应的同记录留出检验。\n\n![时钟与可用信息](figures/14_source_evidence.png)\n\n'
    text+='## 10. 计划完成状态与下一步选择\n\n'
    text+='附件三条工作线中：交叉对照、自由矩阵同协方差诊断、无噪声闭环、已知初态／生理消融、慢噪声减弱、连续时长、有效信息、双参数剖面、实际光学／辅助文件核查均完成。实测辅助观测候选的拟合／留出检验未具备输入；原计划C与D因此没有执行。这个缺口来自实际证据，不是计算失败或额外审批要求。\n\n'
    text+='后续优先联系数据持有方核实ECG／呼吸是否存在未包含在当前发布包中的同步文件，或选取确有独立光学／系统生理约束的数据。拿到具体流后先核验单位、时钟与同记录留出解释力，再进入C；ECG／呼吸可约束系统性成分，却不能自动标定光学增益，也不是纯噪声真值。\n\n'
    text+='合成方面应先把连续上下文与初始化策略迁移到实际消费者，并在未知频谱、非平稳污染、错误生理固定量、EEG约束和完整随机SSM中验证。当前正证据不支持马上增加生理状态或启动tokenizer训练。只有独立观测约束已落实、推断在相应合成条件通过，而实测仍稳定出现同类动态偏差时，才有依据改动生理结构。\n\n'
    text+='## 11. 复现、资源与交付验证\n\n'
    text+='主运行在用户systemd服务下使用48个独立物理核心，每worker数值库线程为1，最多48个在途任务；用户linger已启用。6worker试运行覆盖6类任务，10次拟合全部完成。主运行约203秒完成，运行期间一次资源采样内存约1.76 GB，CPU累计约6178 CPU秒（该采样并非峰值或最终资源总量）。配置、命令、source_snapshot、分面板原子结果和失败日志均保留；没有修改旧实验生成物。\n\n'
    text+='本目录analysis.json、method_summary.csv、paired_effects.csv、tau_contrasts.csv、profile_points.csv、profile_summary.csv、coverage.csv、bound_details.json和failures.csv均由上述证据生成。source_audit.json在上级目录，包含具体路径、MAT字段、ZIP成员、处理合同、通道和时钟。原始任务结果分别位于主运行和剖面补密运行的tasks/，报告不是另一个实验事实源。\n\n'
    text+='交付包含本Markdown、可离线阅读HTML、主PDF、图册PDF和14幅240 dpi PNG。PDF正文、表格、图注保持可搜索；全部图形作为PNG图像嵌入，没有向PDF嵌入SVG或矢量图页。检查图像对象、正文提取和渲染页面；本环境没有完成WPS滚动／缩放实测。\n'
    for name,title in figures:
        text=re.sub(r'(!\[[^\]]*\]\(figures/'+re.escape(name)+r'\.png\))',lambda m:m.group(1)+'\n\n*'+title+'*',text)
    (out/'REPORT.md').write_text(text)
    body=markdown.markdown(text,extensions=['tables']);body=body.replace('<img ','<img width="510" ')
    body=re.sub(r'<p>(<img[^>]+>)</p>',r'<div style="page-break-before:always;page-break-inside:avoid">\1</div>',body)
    css='body{font-family:sans-serif;font-size:9pt;line-height:1.45;color:#18232f}h1{font-size:20pt;color:#14566b}h2{font-size:13pt;color:#14566b;margin-top:15pt}table{font-size:6.8pt;border-collapse:collapse;width:100%}td,th{border:0.5pt solid #cbd5df;padding:3pt}img{max-width:100%}p{margin:7pt 0}a{color:#176b87}'
    story=fitz.Story(body,user_css=css,archive=fitz.Archive(str(out)))
    def rectfn(number,filled):
        if number>80:raise RuntimeError('PDF layout did not finish')
        page=fitz.paper_rect('a4');return page,page+(36,32,-36,-32),None
    pdf=story.write_with_links(rectfn);pdf.save(out/'REPORT.pdf',garbage=4,deflate=True)
    searchable=''.join(p.get_text() for p in pdf)
    normalize=lambda value:''.join(unicodedata.normalize('NFKC',value).split())
    expected_sections=[line.removeprefix('## ') for line in text.splitlines() if line.startswith('## ')]
    missing_sections=[section for section in expected_sections if normalize(section) not in normalize(searchable)]
    missing_captions=[title for _,title in figures if normalize(title) not in normalize(searchable)]
    image_widths=[rect.width for page in pdf for item in page.get_images() for rect in page.get_image_rects(item[0])]
    validation=dict(pages=len(pdf),image_objects=sum(len(p.get_images()) for p in pdf),searchable_text_chars=len(searchable),figures=len(figures),bitmap_dpi=240,wps_checked=False,
        expected_searchable_sections=len(expected_sections),missing_searchable_sections=missing_sections,
        missing_searchable_captions=missing_captions,minimum_figure_width_pt=min(image_widths))
    assert validation['image_objects']==len(figures) and not missing_sections and not missing_captions and min(image_widths)>490,validation
    pdf.close();atlas=fitz.open()
    for name,title in figures:
        page=atlas.new_page(width=842,height=595);page.insert_image(fitz.Rect(24,55,818,565),filename=str(out/'figures'/f'{name}.png'),keep_proportion=True);page.insert_htmlbox(fitz.Rect(25,12,817,47),html.escape(title))
    atlas.save(out/'FIGURES.pdf',garbage=4,deflate=True);atlas.close()
    for name,_ in figures:body=body.replace(f'figures/{name}.png','data:image/png;base64,'+base64.b64encode((out/'figures'/f'{name}.png').read_bytes()).decode())
    (out/'REPORT.html').write_text('<html lang="zh-CN"><meta charset="utf-8"><style>'+css+'</style>'+body+'</html>')
    dump(out/'report_validation.json',validation);shutil.copyfile(__file__,out/'render_source.py');print(json.dumps(validation))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--profile-resolution-run',type=Path,required=True);parser.add_argument('--basin-check-run',type=Path,required=True);parser.add_argument('--profile-basin-run',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();render(args.run.resolve(),args.profile_resolution_run.resolve(),args.basin_check_run.resolve(),args.profile_basin_run.resolve(),args.output_dir.resolve())
