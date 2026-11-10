#!/usr/bin/env python3
"""Read-only paired v1/v3 cache audit; no raw inputs, labels, or refitting.

Full task metadata census and an outcome-independent spectral panel: first,
middle and last window and pair in every record. Outputs live in a fresh child
of the completed v3 run. Subject summaries, not windows, are inference units.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.stats import kurtosis


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def anchors(values):
    values = sorted(set(values))
    return [values[i] for i in sorted({0, len(values)//2, len(values)-1})] if values else []


def select_panel(names):
    windows = anchors(n.split('__p')[0] for n in names)
    return [n for w in windows for n in anchors(n for n in names if n.startswith(w+'__p'))]


def signal_features(x, rate=10.):
    """Constant-detrended Hann periodogram; relative bands sum discrete bins.

    No padding/filtering. Frequencies below three cycles per available support
    are kept separate; the fast band is >0.2 Hz. EEG here is PC1 log-power,
    so these frequencies describe its envelope, not native EEG oscillations.
    """
    x = np.asarray(x, float)
    sd = float(x.std())
    if not np.isfinite(x).all() or sd <= 1e-12:
        return dict(sd=sd, kurtosis=np.nan, mobility=np.nan, centroid=np.nan,
                    slow_fraction=np.nan, core_fraction=np.nan, fast_fraction=np.nan)
    f, p = periodogram(x, fs=rate, window='hann', detrend='constant', scaling='density')
    p = p[1:]; f = f[1:]
    total = p.sum()
    cutoff = 3*rate/len(x)
    return dict(sd=sd, kurtosis=float(kurtosis(x, fisher=True, bias=False)),
                mobility=float(np.diff(x).std()*rate/sd),
                centroid=float(f@p/total),
                slow_fraction=float(p[f<cutoff].sum()/total),
                core_fraction=float(p[(f>=cutoff)&(f<=.2)].sum()/total),
                fast_fraction=float(p[f>.2].sum()/total))


def record_equal(frame):
    return frame.groupby('record').mean(numeric_only=True).mean().to_dict()


def task_values(d):
    fit = d['fits']['tak_tau_beta']
    values = {f'nmse_{k}': v for k,v in fit['metrics_common'].items()}
    values.update(tau=fit['tau'], beta=fit['beta'])
    for w in fit['waveform']:
        c = w['component']
        for k in ['pearson_r', 'centered_amplitude_ratio', 'local_centered_nrmse',
                  'signed_gain', 'bias_training_sd']:
            values[c+'_'+k] = w[k] if w[k] is not None else np.nan
        a = w['centered_amplitude_ratio']
        values[c+'_abs_log_amplitude_ratio'] = abs(np.log(a)) if a>0 else np.nan
        values[c+'_nonpositive_r'] = float(w['pearson_r']<=0) if w['pearson_r'] is not None else np.nan
    values['Hb_not_better_than_zero'] = float(d['Hb_not_better_than_zero'])
    return values


def spectral_pair(old, new):
    with np.load(old, allow_pickle=False) as a, np.load(new, allow_pickle=False) as b:
        if not np.array_equal(a['time_s'], b['time_s']):
            raise ValueError('Different time support')
        if not np.array_equal(a['target'][:,1:], b['target'][:,1:]):
            raise ValueError('Hb targets changed')
        if not np.array_equal(a['fit_mask'], b['fit_mask']):
            raise ValueError('Masks changed')
        use = np.all(a['fit_mask'] & a['prediction_valid'] & b['prediction_valid'], axis=1)
        if use.sum() < 51 or not np.array_equal(np.flatnonzero(use), np.arange(use.sum())):
            raise ValueError('Noncontiguous scored support')
        rows = []
        for version, data in [('v1',a), ('v3',b)]:
            for j, component in enumerate(['EEG', 'HbO', 'HbR']):
                target = signal_features(data['target'][use,j])
                pred = signal_features(data['prediction'][use,j])
                row = dict(version=version, component=component, scored_steps=int(use.sum()))
                row.update({'target_'+k:v for k,v in target.items()})
                row.update({'prediction_'+k:v for k,v in pred.items()})
                rows.append(row)
        for row in rows:
            data = a if row['version']=='v1' else b
            row['target_HbO_HbR_r'] = float(np.corrcoef(data['target'][use,1:].T)[0,1])
            row['prediction_HbO_HbR_r'] = float(np.corrcoef(data['prediction'][use,1:].T)[0,1])
            row['EEG_target_v1_v3_r'] = float(np.corrcoef(a['target'][use,0],b['target'][use,0])[0,1])
        return rows


def subject_worker(payload):
    cfg, subject_key = payload
    started = time.monotonic()
    out = Path(cfg['output'])/'subjects'/subject_key.replace('/', '__')
    done = out.with_suffix('.json')
    if done.exists():
        return json.loads(done.read_text())
    roots = [Path(cfg[v])/'tasks'/subject_key for v in ['v1','v3']]
    files = [{str(p.relative_to(r)) for p in r.glob('*/*.json')} for r in roots]
    if files[0] != files[1]:
        raise ValueError('Task identity inventories differ: '+subject_key)
    by_record = {}
    for f in sorted(files[0]):
        by_record.setdefault(str(Path(f).parent), []).append(Path(f).name)
    panel = [str(Path(rec)/n) for rec,names in by_record.items() for n in select_panel(names)]
    # Freeze selection before reading fit outcomes.
    write_json(out.with_suffix('.selection.json'), dict(subject=subject_key, selected=panel))
    panel = set(panel)
    rows = []; spectral = []; failures = []; matched = 0
    counts = dict(planned=len(files[0]), both_qualified=0, v1_only=0, v3_only=0, neither=0,
                  selected=len(panel), selected_qualified=0)
    for name in sorted(files[0]):
        pair = [json.loads((r/name).read_text()) for r in roots]
        for d, schema in zip(pair, ['shared_driver_tak_subject_cache_v1','shared_driver_tak_subject_cache_v3']):
            if d['schema'] != schema:
                raise ValueError('Wrong task schema')
        # Output coordinate name changes; underlying observation identity must match.
        for k in ['dataset','subject','record','window','pair','split']:
            if pair[0]['identity'][k] != pair[1]['identity'][k]:
                raise ValueError('Identity mismatch: '+name)
        q = [bool(d.get('qualified')) for d in pair]
        state = 'both_qualified' if all(q) else 'v1_only' if q[0] else 'v3_only' if q[1] else 'neither'
        counts[state] += 1
        if not all(q):
            failures.append(dict(task=name, v1=pair[0]['status'], v3=pair[1]['status']))
            continue
        matched += 1
        rec = str(Path(name).parent)
        for version,d in zip(['v1','v3'], pair):
            rows.append(dict(record=rec,version=version,**task_values(d)))
        if name in panel:
            spectral.extend(dict(task=name,record=rec,**r) for r in spectral_pair(
                (roots[0]/name).with_suffix('.npz'), (roots[1]/name).with_suffix('.npz')))
            counts['selected_qualified'] += 1
    frame = pd.DataFrame(rows)
    means = {v:record_equal(g.drop(columns='version')) for v,g in frame.groupby('version')}
    # Tail diagnostics use all common qualified windows; explicitly distinct from subject means.
    tails = {v:g.select_dtypes('number').quantile(.9).to_dict() for v,g in frame.groupby('version')}
    spec = pd.DataFrame(spectral)
    spec.to_csv(out.with_suffix('.spectral.csv'), index=False)
    specmeans = {v+'_'+c:record_equal(g.drop(columns=['version','component','task']))
                 for (v,c),g in spec.groupby(['version','component'])}
    result = dict(subject=subject_key,counts=counts,means=means,p90_windows=tails,
                  spectra=specmeans,failures=failures,seconds=time.monotonic()-started,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(done,result)
    return result


def summarize(cfg):
    out = Path(cfg['output'])
    subjects = [json.loads((out/'subjects'/(s.replace('/','__')+'.json')).read_text()) for s in cfg['subjects']]
    counts = {k:sum(s['counts'][k] for s in subjects) for k in subjects[0]['counts']}
    for version,own in [('v1','v1_only'),('v3','v3_only')]:
        source = json.loads((Path(cfg[version])/'summary.json').read_text())['counts']
        if counts['planned']!=source['planned'] or counts['both_qualified']+counts[own]!=source['qualified']:
            raise ValueError('Census does not reproduce source counts')
    paired=[]; spec=[]
    for s in subjects:
        ds,sub=s['subject'].split('/')
        for metric in s['means']['v1']:
            a=s['means']['v1'][metric];b=s['means']['v3'][metric]
            paired.append(dict(dataset=ds,subject=sub,metric=metric,v1=a,v3=b,delta=b-a))
        for key,values in s['spectra'].items():
            v,c=key.split('_',1)
            for metric,value in values.items():
                spec.append(dict(dataset=ds,subject=sub,version=v,component=c,metric=metric,value=value))
    frame=pd.DataFrame(paired);frame.to_csv(out/'subject_metrics.csv',index=False)
    pd.DataFrame(spec).to_csv(out/'subject_spectra.csv',index=False)
    rng=np.random.default_rng(20261110); results=[]
    for (ds,metric),g in frame.groupby(['dataset','metric']):
        g=g.dropna(subset=['v1','v3']);delta=g.delta.to_numpy()
        if not len(delta):continue
        ci=np.quantile(delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1),[.025,.975])
        results.append(dict(dataset=ds,metric=metric,n=len(delta),v1=g.v1.mean(),v3=g.v3.mean(),
                            delta=g.delta.mean(),ci_low=ci[0],ci_high=ci[1],subjects_increased=int((delta>0).sum())))
    pd.DataFrame(results).to_csv(out/'dataset_metrics.csv',index=False)
    write_json(out/'summary.json',dict(status='completed',counts=counts,subjects=len(subjects),
        inference='pointwise descriptive 95% paired subject bootstrap; 2000 resamples, no multiple-testing correction',
        source_summaries={v:str(Path(cfg[v])/'summary.json') for v in ['v1','v3']},
        spectral_panel='first/middle/last window and pair per record; selection before outcomes; common qualified only',
        aggregation='mean tasks within record, then mean records within subject, then equal subjects within dataset',
        EEG='10Hz PC1 log-power envelope, different targets in v1/v3; not native EEG voltage spectrum'))


def report(cfg):
    import html
    import shutil
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out=Path(cfg['output']);export=out/'report_v1';export.mkdir(exist_ok=False)
    shutil.copyfile(__file__,export/'source.py')
    summary=json.loads((out/'summary.json').read_text())
    full=pd.read_csv(out/'dataset_metrics.csv')
    aliases=dict(eeg_fnirs_single_trial='Single-Trial',simultaneous_eeg_nirs='Simultaneous',
                 refed='REFED',visual_cognitive_motivation='Visual')
    order=list(aliases)
    frames=[]
    for path in sorted((out/'subjects').glob('*.spectral.csv')):
        f=pd.read_csv(path);ds,sub=path.name.removesuffix('.spectral.csv').split('__')
        f['dataset']=ds;f['subject']=sub;frames.append(f)
    spectra=pd.concat(frames,ignore_index=True)
    spectra['sd_ratio']=np.where(spectra.target_sd>1e-12,spectra.prediction_sd/spectra.target_sd,np.nan)
    spectra['HbO_HbR_correlation_error']=abs(spectra.prediction_HbO_HbR_r-spectra.target_HbO_HbR_r)
    # Robust summaries keep near-flat input flags; do not average unbounded ratios.
    robust=spectra.groupby(['dataset','subject','version','component','record']).median(numeric_only=True).groupby(
        ['dataset','subject','version','component']).median()
    robust.to_csv(export/'spectral_subject_medians.csv')
    med=robust.groupby(['dataset','version','component']).median()
    med.to_csv(export/'spectral_dataset_medians.csv')
    counts=spectra.assign(flat=spectra.target_sd<=1e-12).groupby(['dataset','version','component']).agg(
        selected_qualified=('task','size'),flat_target=('flat','sum'))
    counts.to_csv(export/'spectral_denominators.csv')
    errors=[];corr=[];amp=[];freq=[];joint=[];coordinate=[]
    c1=json.loads((Path(cfg['v1'])/'coordinates.json').read_text())
    c3=json.loads((Path(cfg['v3'])/'coordinates.json').read_text())
    for ds in order:
        row={'数据集':aliases[ds]}
        for metric,label in [('nmse_hb_nmse','Hb'),('nmse_eeg_nmse','EEG代理')]:
            r=full[(full.dataset==ds)&(full.metric==metric)].iloc[0]
            a,b=np.sqrt(r.v1),np.sqrt(r.v3)
            row.update({label+' v1':a,label+' v3':b,label+'变化%':100*(b/a-1)})
        errors.append(row)
        for comp in ['EEG','HbO','HbR']:
            r=full[(full.dataset==ds)&(full.metric==comp+'_pearson_r')].iloc[0]
            corr.append({'数据集':aliases[ds],'分量':comp,'v1':r.v1,'v3':r.v3,'差值':r.delta,
                         '区间下限':r.ci_low,'区间上限':r.ci_high,'被试数':r.n})
            a=med.loc[(ds,'v1',comp)];b=med.loc[(ds,'v3',comp)]
            amp.append({'数据集':aliases[ds],'分量':comp,'v1 SD比':a.sd_ratio,'v3 SD比':b.sd_ratio})
            freq.append({'数据集':aliases[ds],'分量':comp,'实测v1快变功率%':100*a.target_fast_fraction,
                         '实测v3快变功率%':100*b.target_fast_fraction,
                         'v1预测快变功率%':100*a.prediction_fast_fraction,'v3预测快变功率%':100*b.prediction_fast_fraction,
                         '实测v1超额峰度':a.target_kurtosis,'实测v3超额峰度':b.target_kurtosis,
                         'v1预测超额峰度':a.prediction_kurtosis,'v3预测超额峰度':b.prediction_kurtosis})
        a=med.loc[(ds,'v1','HbO')];b=med.loc[(ds,'v3','HbO')]
        joint.append({'数据集':aliases[ds],'实测HbO/HbR相关':a.target_HbO_HbR_r,
                      'v1预测相关':a.prediction_HbO_HbR_r,'v3预测相关':b.prediction_HbO_HbR_r,
                      'v1相关绝对误差':a.HbO_HbR_correlation_error,'v3相关绝对误差':b.HbO_HbR_correlation_error})
        for k,v in c1['individual'].items():
            if not k.startswith(ds+'__'):continue
            w=c3['individual'][k];pc=np.array(v['pc']);pc3=np.array(w['pc'])
            coordinate.append(dict(dataset=ds,subject=k.split('__')[1],
                cosine=pc@pc3/np.linalg.norm(pc)/np.linalg.norm(pc3),
                eeg_factor_ratio=w['eeg_factor']/v['eeg_factor'],hb_factor_ratio=w['hb_factor']/v['hb_factor']))
    pd.DataFrame(coordinate).to_csv(export/'coordinate_changes.csv',index=False)
    tables=dict(errors=errors,correlations=corr,amplitudes=amp,frequency=freq,Hb_joint=joint)
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(export/(name+'.csv'),index=False)
    colors=['#3478a4','#d16d32'];x=np.arange(4)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    e=pd.DataFrame(errors)
    for ax,label,title in [(axes[0,0],'Hb','Paired Hb NRMSE (lower is better)'),
                           (axes[0,1],'EEG代理','Paired EEG-proxy NRMSE (lower is better)')]:
        for j,v in enumerate(['v1','v3']):ax.bar(x+(j-.5)*.32,e[label+' '+v],.32,label=v,color=colors[j])
        ax.set_title(title);ax.set_ylabel('Training-reference SD units');ax.legend()
    for j,v in enumerate(['v1','v3']):
        values=[med.loc[(ds,v,'HbO'),'sd_ratio'] for ds in order]
        axes[1,0].bar(x+(j-.5)*.32,values,.32,label=v,color=colors[j])
        values=[100*med.loc[(ds,v,'HbO'),'prediction_fast_fraction'] for ds in order]
        axes[1,1].bar(x+(j-.5)*.32,values,.32,label=v,color=colors[j])
    axes[1,0].axhline(1,color='gray',ls='--');axes[1,0].set_title('HbO predicted SD / observed SD (sample panel)')
    axes[1,0].set_ylabel('Ratio; dashed line = amplitude match');axes[1,0].legend()
    axes[1,1].scatter(x,[100*med.loc[(ds,'v1','HbO'),'target_fast_fraction'] for ds in order],
                      marker='D',color='black',label='Observed',zorder=4)
    axes[1,1].set_title('HbO power above 0.2 Hz (sample panel)');axes[1,1].set_ylabel('% of positive-frequency power');axes[1,1].legend()
    for ax in axes.flat:
        ax.set_xticks(x,list(aliases.values()));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('v3 versus v1: better Hb fit, different statistical trade-offs\nTop: 208,109 paired tasks / 103 subjects; bottom: 5,050 fixed-panel paired tasks',fontsize=14)
    fig.savefig(export/'comparison.png',dpi=220);plt.close(fig)
    def table(rows):return pd.DataFrame(rows).to_html(index=False,float_format=lambda x:f'{x:.4f}')
    body='''<!doctype html><meta charset="utf-8"><title>v3 与 v1 缓存统计比较</title>
<style>body{max-width:1200px;margin:40px auto;font:17px/1.75 sans-serif;color:#182635}h1,h2{line-height:1.4}table{border-collapse:collapse;font-size:14px;margin:18px 0;width:100%}td,th{border:1px solid #ced7df;padding:7px;text-align:right}th{background:#edf3f7}img{width:100%}p{max-width:1050px}</style>
<h1>v3 与 v1 全量拟合缓存：统计特征与配置影响</h1>
<p>结论：v3 在四库全部 103 名被试中降低平均 Hb 误差、提高 HbO/HbR 波形相关性，同时降低 EEG 代理波形相关性。幅度失配总体缓解，但频谱、峰度与 HbO/HbR 联合关系的改善具有数据集差异。</p>
<h2>配置与比较口径</h2>
<p>v1 使用每库训练被试拟合的共同 PC1；v3 每名被试独立 PC1。两版都冻结被试正尺度，Hb 原始目标和 Hb 尺度保持一致。观测权重由 (1,1,1) 改为 (1/3,4/3,4/3)，改变 EEG 与 Hb 误差的优化优先级；τ、β 的 log 先验 SD=0.125、参数范围、动力学及停止条件保留。旧解在 v3 仅作初值，全部变量重新优化。</p>
<p>全量元数据有 209,340 对身份，208,109 对共同数值合格；仅 v1 合格 461 对、仅 v3 合格 481 对、两版均不合格 289 对。全量拟合成功数因此只净增 20。共同成功上的配对统计与失败分母分别保留。</p>
<p>频谱面板在读取结果前按每条记录首、中、末窗口及通道编号选定 5,073 对，其中 5,050 对共同合格；23 对不替换。这里只读保留的公开缓存，不读取原始信号或标签、不重新拟合。</p>
<p>误差沿用源实现：每时点平方残差除以固定训练参考 SD²；Hb 先平均 HbO/HbR 的归一化平方误差，再在记录内平均任务、被试内平均记录、数据集内等权平均被试，最后开方。所有差值为 v3−v1。相关性也按记录、被试、数据集逐层平均。CSV 中 95% 区间为被试配对 bootstrap 2,000 次的描述性点区间，未做多重比较校正。</p>
<p>EEG 是 10 Hz 的 PC1 对数功率代理，不是原生电压。个体 PCA 改变其目标；其误差表示各版相对各自目标的保真度。样本中 Hb 目标、时间与 mask 均逐数组核对相同。</p>
<img src="comparison.png" alt="全量配对误差、样本幅度与频谱比较">
<h2>全量配对误差与相关性</h2>'''+table(errors)+table(corr)+'''
<p>Hb 改善与 EEG 代理保真度下降在全部被试方向一致，符合提高 Hb 相对权重的设计。PCA 与权重同时改变，本次比较给出组合修改的净影响；不能将收益百分比单独分配给某一配置项。</p>
<h2>幅度与信号分布</h2>
<p>下表为预测 SD/实测 SD：每记录内取样本中位数，再取被试内记录中位数、数据集内被试中位数。1 表示幅度相等。采用这种稳健汇总，是因为 REFED/Visual 的部分近恒定 Hb 目标使均值比值极大。频谱样本中 REFED 两分量各有 10 个目标 SD≤1e−12，形状及比值记为缺失并保留计数；未把比值截断。</p>'''+table(amp)+'''
<p>Single-Trial/Simultaneous 的 Hb 幅度低估缓解；REFED/Visual 的 HbO 幅度过大缓解，但 Visual HbO 仍约 2.5 倍。REFED HbR 的典型幅度比从接近 1 变为约 0.76，说明误差改善并不要求每个分布特征都更接近实测。</p>
<h2>频谱、峰度与联合关系</h2>
<p>按实际完整评分支持（末点排除）计算周期 Hann 窗、去常数趋势的 periodogram，10 Hz、不补零、不额外滤波。快变比例为 &gt;0.2 Hz 的功率占所有正频率功率的比例；不足三个周期的低频单独保存在数据表。短 REFED 使用真实长度。超额峰度使用无偏 Fisher 定义，正态分布参考值为 0。表中均采用与幅度相同的三级中位数汇总。</p>'''+table(freq)+'''
<p>Single-Trial/Simultaneous 的预测 Hb 快变功率下降；REFED/Visual 则上升，例如 REFED HbO 从约 1.24% 到 3.93%，实测约 0.55%。REFED HbR 预测超额峰度从约 0.98 到 2.24，实测约 0.06。更低重建误差没有带来统一的频谱和尾部分布匹配。</p>'''+table(joint)+'''
<p>预测 HbO/HbR 仍保持比实测更强的负相关；这项联合结构偏差并未随单分量误差下降而充分消失。幅度、单分量相关和双分量关系应分别判断。</p>
<h2>配置影响与使用建议</h2>
<p>个体 PCA 改变被试 EEG 代理坐标，Hb 输入本身不变。观测权重重分配使拟合更贴合 Hb，代价是 EEG 代理形状相关下降。27 个生产先导案例中，旧解热启动与标准路径的最大相对目标增加仅 1.87e−10，CPU/GPU 最大预测差约 3.57e−14 被试 SD，支持热启动主要承担数值加速作用；它没有复用旧预测作为新结果。</p>
<p>v3 适合本项目 Hb 优先的条件重建用途。后续统计使用完整 v3 的有效性标记，并按数据集同时检查幅度、频谱和 HbO/HbR 联合关系；尤其保留 REFED/Visual 的形状失配。v3/features 是到 v1/features 的链接，旧输入特征依赖应保留。</p>
<h2>证据与复现</h2><p>原始完成状态与数值结果仍由各 run 的 summary.json、verification.json、逐任务 JSON/NPZ 持有。此报告为只读派生统计。父目录保留 analysis_config.json、launch.json、summary.json、dataset_metrics.csv、subject_metrics.csv，以及逐被试选择、失败记录和统计表；本目录保留完整频谱汇总、坐标变化及渲染源码。</p>'''
    body+='<p>源缓存：'+html.escape(cfg['v1'])+'<br>'+html.escape(cfg['v3'])+'</p>'
    (export/'report.html').write_text(body)
    print(json.dumps(dict(report=str(export/'report.html'),summary=summary['counts'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['v1','v3','output']:p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--stage',choices=['plan','pilot','run','summarize','report'],required=True)
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args();out=args.output.resolve()
    if out.parent!=args.v3.resolve():raise ValueError('Output must be a fresh v3 child')
    cfg=dict(schema='tak_cache_version_audit_v1',v1=str(args.v1.resolve()),v3=str(args.v3.resolve()),output=str(out),
             boundary='completed public cache reconstruction statistics only; no raw data, labels, protected data or downstream scoring',
             spectral_selection='first/middle/last window x first/middle/last pair per record before reading outcomes',
             spectral_support='entire scored contiguous support, last bin excluded; 10 Hz; periodic Hann periodogram; no padding/filter',
             failure_policy='full census retained; paired statistics on common numerical successes; selected failures never replaced')
    for v in ['v1','v3']:
        root=Path(cfg[v]);s=json.loads((root/'summary.json').read_text());check=json.loads((root/'verification.json').read_text())
        if s['schema']!='shared_driver_tak_subject_cache_'+v or s['status']!='completed' or check['status']!='passed' or check['issues']:
            raise ValueError('Completed verified source required')
        resolved=json.loads((root/'resolved_config.json').read_text())
        if resolved['data']['protected_data']!='forbidden':raise ValueError('Public cache contract required')
    subjects=[str(s.relative_to(args.v3/'tasks')) for ds in (args.v3/'tasks').iterdir() if ds.is_dir() for s in ds.iterdir() if s.is_dir()]
    cfg['subjects']=sorted(subjects)
    if args.stage=='plan':
        out.mkdir(exist_ok=False);(out/'subjects').mkdir();write_json(out/'analysis_config.json',cfg)
        print(json.dumps(dict(subjects=len(subjects),output=str(out))),flush=True);return
    if json.loads((out/'analysis_config.json').read_text())!=cfg:raise ValueError('Analysis configuration changed')
    if args.stage=='summarize':summarize(cfg);return
    if args.stage=='report':report(cfg);return
    if not os.environ.get('SSM_SYSTEMD_UNIT'):raise ValueError('Use durable user systemd for census')
    selected=cfg['subjects']
    if args.stage=='pilot':selected=[next(s for s in selected if s.startswith(ds+'/')) for ds in sorted({s.split('/')[0] for s in selected})]
    started=time.monotonic();completed=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(subject_worker,[(cfg,s) for s in selected]):
            completed.append(r);print(json.dumps(dict(subject=r['subject'],counts=r['counts'],seconds=r['seconds'],rss=r['peak_rss_mib'])),flush=True)
            write_json(out/(args.stage+'_progress.json'),dict(completed=len(completed),total=len(selected),seconds=time.monotonic()-started))
    write_json(out/(args.stage+'_execution.json'),dict(status='completed',workers=args.workers,seconds=time.monotonic()-started,
        planned=sum(r['counts']['planned'] for r in completed),peak_worker_mib=max(r['peak_rss_mib'] for r in completed),
        supervisor=os.environ['SSM_SYSTEMD_UNIT']))
    if args.stage=='run':summarize(cfg)


def residual_geometry(amplitude, correlation):
    """Centered residual geometry for O=X-S, independent of physical units."""
    a, r = float(amplitude), float(correlation)
    if not np.isfinite([a, r]).all() or a < 0 or abs(r) > 1+1e-8:
        raise ValueError('Invalid amplitude/correlation')
    r = np.clip(r, -1, 1)
    b = np.sqrt(max(0., 1+a*a-2*a*r))
    return dict(s_sd_ratio=a, o_sd_ratio=b, correlation_sx=r,
                correlation_so=(r-a)/b if a*b>1e-12 else np.nan,
                variance_sum_ratio=a*a+b*b)


def fnirs_subject(payload):
    cfg, subject = payload
    out = Path(cfg['output'])/'subjects'/subject.replace('/', '__')
    if out.with_suffix('.json').exists():
        return json.loads(out.with_suffix('.json').read_text())
    start = time.monotonic(); rows=[]; failures=[]; checks=[]; planned=0
    for path in sorted((Path(cfg['run'])/'tasks'/subject).glob('*/*.json')):
        d=json.loads(path.read_text()); planned+=1
        if d['schema']!='shared_driver_tak_subject_cache_v3':
            raise ValueError('Unexpected source schema')
        identity=d['identity']
        if identity['dataset']+'/'+identity['subject']!=subject:
            raise ValueError('Unexpected source identity')
        if not d['qualified']:
            failures.append(dict(task=str(path.relative_to(Path(cfg['run']))),status=d['status']))
            continue
        for w in d['fits']['tak_tau_beta']['waveform']:
            if w['component']=='EEG':continue
            common=dict(dataset=identity['dataset'],subject=identity['subject'],record=identity['record'],
                        window=identity['window']['key'],pair=identity['pair']['id'],split=identity['split'],
                        duration_s=d['duration_s'],component=w['component'])
            if w['pearson_r'] is None:
                rows.append(dict(**common,defined=False));continue
            g=residual_geometry(w['centered_amplitude_ratio'],w['pearson_r'])
            # A valid amplitude denominator must agree with its retained physical SD.
            sd=w['target_centered_rms_in_cache_unit']
            ref=w['absolute_rmse_in_cache_unit']/w['nrmse'] if w['nrmse']>0 else np.nan
            row=dict(**common,defined=True,**g,local_rmse_ratio=w['local_centered_nrmse'],
                     x_sd_cache=sd,x_sd_subject=sd/ref if ref>0 else np.nan,
                     s_sd_subject=g['s_sd_ratio']*sd/ref if ref>0 else np.nan)
            rows.append(row)
            # First qualified task per subject: independently check the algebra on arrays.
            if len(checks)<2:
                with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:
                    j=1 if w['component']=='HbO' else 2
                    use=np.all(a['fit_mask'] & a['prediction_valid'],axis=1)
                    x=a['target'][use,j];s=a['prediction'][use,j];o=x-s
                    actual=[s.std()/x.std(),o.std()/x.std(),np.corrcoef(s,o)[0,1]]
                    expected=[g['s_sd_ratio'],g['o_sd_ratio'],g['correlation_so']]
                    np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-9)
                    checks.append(dict(task=str(path),component=w['component'],max_error=float(np.max(np.abs(np.array(actual)-expected)))))
    pd.DataFrame(rows).to_csv(out.with_suffix('.csv'),index=False)
    result=dict(subject=subject,planned=planned,qualified=planned-len(failures),failed=len(failures),
                failures=failures,array_checks=checks,seconds=time.monotonic()-start,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(out.with_suffix('.json'),result);return result


def fnirs_summarize(cfg):
    out=Path(cfg['output']); frames=[]; status=[]
    for subject in cfg['subjects']:
        path=out/'subjects'/subject.replace('/','__')
        frames.append(pd.read_csv(path.with_suffix('.csv'),dtype={'subject':str}))
        status.append(json.loads(path.with_suffix('.json').read_text()))
    frame=pd.concat(frames,ignore_index=True)
    source=json.loads((Path(cfg['run'])/'summary.json').read_text())['counts']
    counts={k:sum(s[k] for s in status) for k in ['planned','qualified','failed']}
    if counts!={k:source[k] for k in counts}:raise ValueError('Census differs from completed source')
    defined=frame[frame.defined].copy()
    events={}
    for cutoff in [1.5,2.,3.,5.]:
        key='amplified_'+str(cutoff).replace('.','p')
        events[key]=defined.s_sd_ratio>=cutoff
    events.update(severe_cancellation=(defined.s_sd_ratio>=2)&(defined.o_sd_ratio>=1)&(defined.correlation_so<=-.8),
                  residual_ge_signal=defined.o_sd_ratio>=1,poor_shape=defined.correlation_sx<.5,
                  faithful_small_residual=(defined.correlation_sx>=.8)&(defined.o_sd_ratio<=.5))
    for key,value in events.items():defined[key]=value.astype(float)
    metrics=['s_sd_ratio','o_sd_ratio','correlation_sx','correlation_so','variance_sum_ratio','x_sd_subject','s_sd_subject']
    group=['dataset','subject','component']
    records=defined.groupby(group+['record'])[metrics+list(events)].mean()
    subjects=records.groupby(group).mean().reset_index()
    subjects.to_csv(out/'subject_severity.csv',index=False)
    rows=[];rng=np.random.default_rng(20261110)
    for (ds,c),g in defined.groupby(['dataset','component']):
        ss=subjects[(subjects.dataset==ds)&(subjects.component==c)]
        allg=frame[(frame.dataset==ds)&(frame.component==c)]
        dsstatus=[s for s in status if s['subject'].startswith(ds+'/')]
        row=dict(dataset=ds,component=c,subjects=len(ss),planned=sum(s['planned'] for s in dsstatus),
                 qualified=len(allg),defined=len(g),undefined=len(allg)-len(g),failed=sum(s['failed'] for s in dsstatus))
        for key in metrics:
            row[key+'_subject_mean']=ss[key].mean();row[key+'_pooled_median']=g[key].median();row[key+'_pooled_p90']=g[key].quantile(.9)
        for key in events:
            row[key+'_count']=int(g[key].sum());row[key+'_pooled_rate']=g[key].mean()
            row[key+'_subject_rate']=ss[key].mean();row[key+'_subjects_any']=int((g.groupby('subject')[key].max()>0).sum())
            if key=='severe_cancellation':
                values=ss[key].to_numpy()
                ci=np.quantile(values[rng.integers(0,len(values),(2000,len(values)))].mean(1),[.025,.975])
                row[key+'_ci_low'],row[key+'_ci_high']=ci
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'dataset_severity.csv',index=False)
    # Duration and original split are transparent sensitivity strata, not new evaluation gates.
    defined['duration_group']=np.where(defined.duration_s<89.999,'short_REFED','90s')
    for key in ['split','duration_group']:
        defined.groupby(['dataset','component',key])[list(events)].agg(['count','mean']).to_csv(out/(key+'_rates.csv'))
    # Within-subject relative amplitude quartiles isolate a small-denominator association.
    defined['amplitude_quartile']=defined.groupby(group).x_sd_subject.transform(
        lambda x:pd.qcut(x.rank(method='first'),4,labels=[1,2,3,4]).astype(int))
    quartiles=defined.groupby(group+['amplitude_quartile'])[metrics+list(events)].mean().reset_index()
    quartiles.to_csv(out/'subject_amplitude_quartiles.csv',index=False)
    quartiles.groupby(['dataset','component','amplitude_quartile'])[metrics+list(events)].mean().to_csv(out/'amplitude_quartiles.csv')
    # Dataset-native channel reference is metadata, not a fitted grouping.
    import sys
    project=Path(cfg['run']).parents[4]
    sys.path.insert(0,str(project))
    from src.data.channel_geometry import records_from_visual_fnirs_reference
    reference=project/'data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults/fNIRS_to_EEG_channel_reference.xlsx'
    mapping={r.record_id+':'+r.channel_name:r.metadata['nearest_eeg_label'] for r in records_from_visual_fnirs_reference(reference)}
    visual=defined[defined.dataset=='visual_cognitive_motivation'].copy()
    visual['channel_class']=visual.pair.map(lambda p:'EEG_anchor' if mapping[p]!='-' else 'unlabelled_position')
    visual.groupby(['subject','component','pair'])[metrics+list(events)].mean().reset_index().to_csv(out/'visual_subject_channels.csv',index=False)
    channelgroups=visual.groupby(['subject','component','channel_class','record'])[metrics+list(events)].mean().groupby(['subject','component','channel_class']).mean()
    channelgroups.to_csv(out/'visual_subject_channel_classes.csv')
    channelgroups.groupby(['component','channel_class']).mean().to_csv(out/'visual_channel_classes.csv')
    write_json(out/'visual_channel_metadata.json',dict(source=str(reference),mapping=mapping,
        interpretation='Native EEG-reference labels only; not proof of source-detector length, sensor quality or physical cause'))
    # Reuse the frozen outcome-independent spectral panel; each subject remains one unit.
    spectra=pd.read_csv(Path(cfg['comparison'])/'subject_spectra.csv',dtype={'subject':str})
    spectra=spectra[(spectra.version=='v3')&(spectra.component!='EEG')]
    spectra.to_csv(out/'subject_input_spectra.csv',index=False)
    spectra.groupby(['dataset','component','metric']).value.agg(['mean','median','min','max']).to_csv(out/'dataset_input_spectra.csv')
    full=[]
    for subject in cfg['subjects']:
        sample=pd.read_csv(Path(cfg['comparison'])/'subjects'/(subject.replace('/','__')+'.spectral.csv'))
        sample=sample[(sample.version=='v3')&(sample.component!='EEG')&(sample.scored_steps==899)]
        ds,sub=subject.split('/')
        for c,g in sample.groupby('component'):
            values=g.groupby('record').mean(numeric_only=True).mean()
            full.append(dict(dataset=ds,subject=sub,component=c,**values.to_dict()))
    full=pd.DataFrame(full);full.to_csv(out/'subject_input_spectra_90s.csv',index=False)
    full.groupby(['dataset','component']).mean(numeric_only=True).to_csv(out/'dataset_input_spectra_90s.csv')
    # The user's figure: all baseline Hb channels, without the maximum-example selection.
    features=pd.read_csv(Path(cfg['features'])/'features.csv',dtype={'subject':str})
    features=features[(features.role=='evaluation')&(features.condition=='baseline')&(features.modality!='EEG')]
    panel=[]
    for key,g in features.groupby('point_key'):
        a=g.set_index('branch');x=a.loc['X'];s=a.loc['S'];o=a.loc['O']
        r=residual_geometry(s['std']/x['std'],s.correlation_to_X)
        panel.append(dict(point_key=key,dataset=x.dataset,subject=x.subject,component=x.modality,
                          **r,x_std_ref=x.std_ref,s_std_ref=s.std_ref,x_centroid=x.spectral_centroid,
                          s_centroid=s.spectral_centroid,x_relative_slow=x.relative_slow,
                          x_mobility=x.hjorth_mobility,s_mobility=s.hjorth_mobility,
                          x_outband=x.outband_fraction,s_outband=s.outband_fraction,
                          core_power_sum_ratio=(s.core_power_ref+o.core_power_ref)/x.core_power_ref))
    panel=pd.DataFrame(panel)
    panel['severe_cancellation']=(panel.s_sd_ratio>=2)&(panel.o_sd_ratio>=1)&(panel.correlation_so<=-.8)
    panel.to_csv(out/'figure_panel_channels.csv',index=False)
    panel.groupby(['dataset','component']).agg(n=('point_key','size'),severe=('severe_cancellation','sum'),
        subjects=('subject','nunique'),a=('s_sd_ratio','median'),b=('o_sd_ratio','median'),
        corr_sx=('correlation_sx','median'),corr_so=('correlation_so','median'),
        spectral_ratio=('core_power_sum_ratio','median')).to_csv(out/'figure_panel_summary.csv')
    write_json(out/'summary.json',dict(status='completed',counts=counts,subjects=len(status),
        defined_components=len(defined),undefined_components=len(frame)-len(defined),
        independent_unit='subject; record means within subject; equal subjects within dataset',
        array_checks=sum(len(s['array_checks']) for s in status),
        max_array_error=max(c['max_error'] for s in status for c in s['array_checks']),
        severity='SD(S)/SD(X)>=2 AND SD(O)/SD(X)>=1 AND corr(S,O)<=-0.8; engineering description',
        new_fits=0,raw_reads=0,protected_reads=0,
        source_spectral_summary=str(Path(cfg['comparison'])/'summary.json')))


def fnirs_report(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(cfg['output'])
    ds=['eeg_fnirs_single_trial','simultaneous_eeg_nirs','refed','visual_cognitive_motivation']
    labels=['Single','Sim','REFED','Visual'];colors=['#546b83','#248886','#ce9235','#b8444c']
    summary=pd.read_csv(out/'dataset_severity.csv').set_index(['dataset','component'])
    subjects=pd.read_csv(out/'subject_severity.csv')
    spectral=pd.read_csv(out/'subject_input_spectra_90s.csv')
    quartiles=pd.read_csv(out/'amplitude_quartiles.csv')
    classes=pd.read_csv(out/'visual_channel_classes.csv')
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    for j,c in enumerate(['HbO','HbR']):
        for i,d in enumerate(ds):
            g=subjects[(subjects.dataset==d)&(subjects.component==c)]
            axes[0,j].scatter(np.full(len(g),i),100*g.severe_cancellation,color=colors[i],alpha=.6,s=24)
            row=summary.loc[(d,c)];v=100*row.severe_cancellation_subject_rate
            axes[0,j].plot([i-.2,i+.2],[v,v],color='black',lw=2)
            axes[0,j].text(i,v+3,f'{v:.2f}%',ha='center',fontsize=10)
        axes[0,j].set(xticks=range(4),xticklabels=labels,ylim=(-2,105),ylabel='Severe cancellation (% of fits)',title=c+' | each point = one subject')
        for i,d in enumerate(ds):
            q=quartiles[(quartiles.dataset==d)&(quartiles.component==c)]
            axes[1,j].plot(q.amplitude_quartile,q.severe_cancellation*100,'o-',color=colors[i],label=labels[i])
        axes[1,j].set(xticks=[1,2,3,4],xticklabels=['Q1 weak','Q2','Q3','Q4 strong'],ylim=(-2,105),ylabel='Severe cancellation (%)',title=c+' | within-subject input SD quartiles')
        axes[1,j].legend();axes[1,j].grid(alpha=.2)
    fig.savefig(out/'severity.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for j,c in enumerate(['HbO','HbR']):
        for k,(metric,label) in enumerate([('target_centroid','Input spectral centroid (Hz)'),('target_mobility','Input SD(derivative) / SD (1/s)'),('target_fast_fraction','Input power above 0.2 Hz (%)')]):
            ax=axes[j,k]
            for i,d in enumerate(ds):
                g=spectral[(spectral.dataset==d)&(spectral.component==c)][metric]*(100 if k==2 else 1)
                ax.scatter(np.full(len(g),i),g,color=colors[i],alpha=.6,s=24)
                ax.plot([i-.2,i+.2],[g.mean()]*2,color='black',lw=2)
            ax.set(xticks=range(4),xticklabels=labels,ylabel=label,title=c+' | each point = one subject');ax.grid(alpha=.2)
    fig.savefig(out/'input_spectra.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    group=pd.read_csv(out/'visual_subject_channel_classes.csv')
    for ax,(metric,label) in zip(axes,[('x_sd_subject','Input SD / subject calibration SD'),('severe_cancellation','Severe cancellation (%)'),('correlation_sx','Correlation of S with X')]):
        for j,c in enumerate(['HbO','HbR']):
            for k,cl in enumerate(['EEG_anchor','unlabelled_position']):
                v=group[(group.component==c)&(group.channel_class==cl)][metric]*(100 if metric=='severe_cancellation' else 1)
                pos=j*2+k;ax.scatter(np.full(len(v),pos),v,color=['#b8444c','#248886'][k],alpha=.6)
                ax.plot([pos-.2,pos+.2],[v.mean()]*2,color='black',lw=2)
        ax.set(xticks=range(4),xticklabels=['HbO\n14 anchors','HbO\n10 other','HbR\n14 anchors','HbR\n10 other'],ylabel=label);ax.grid(alpha=.2)
    fig.savefig(out/'visual_channel_classes.png',dpi=220);plt.close(fig)
    def table(frame):return frame.to_html(index=False,float_format=lambda x:f'{x:.4g}',border=0)
    overview=[]
    for d,name in zip(ds,labels):
        for c in ['HbO','HbR']:
            r=summary.loc[(d,c)]
            overview.append({'数据集':name,'分量':c,'被试':int(r.subjects),'有效分母':int(r.defined),
                'S/X SD中位数':r.s_sd_ratio_pooled_median,'S/X SD P90':r.s_sd_ratio_pooled_p90,
                'O/X SD中位数':r.o_sd_ratio_pooled_median,'corr(S,X)中位数':r.correlation_sx_pooled_median,
                '严重抵消%（被试等权）':100*r.severe_cancellation_subject_rate,
                '95%区间':f'{100*r.severe_cancellation_ci_low:.2f}–{100*r.severe_cancellation_ci_high:.2f}',
                'corr(S,X)<0.5 %':100*r.poor_shape_subject_rate})
    panel=pd.read_csv(out/'figure_panel_summary.csv')
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>fNIRS 数据集差异与 S/O 抵消普查</title>
<style>body{font-family:sans-serif;max-width:1400px;margin:auto;padding:30px;line-height:1.8;color:#203040}img{width:100%}table{border-collapse:collapse;font-size:14px;width:100%}td,th{padding:7px;border:1px solid #ccd}h1,h2{color:#173952}.lead{background:#eef5f8;padding:18px}code{background:#eee}</style>
<h1>fNIRS 数据集差异与 S/O 抵消普查 · 2026-11-10</h1>
<p class="lead">Visual 存在广泛且跨被试一致的抵消问题；REFED 的问题集中在低幅度窗口。最强关联是局部输入波动相对于统一被试尺度过小，而非两个数据集整体频率都更低。Visual 的弱幅度与失真还对应一个明确的通道分组。以下把原图所属小样本与全缓存分开呈现。</p>
<h2>1. 总体范围、计算与失败</h2>
<p>完整 v3 缓存：103 名被试，Single/Sim/REFED/Visual 分别 29/26/32/16 名；209,340 个窗口×Hb通道对，其中 208,590 个数值合格，750 个失败。失败分布为 302/257/9/182。REFED 另有 31 个合格拟合的 HbO/HbR 相关性未定义，共 62 个分量，保留并排除在需要相关性的指标分母外。所有缓存属于既有公开重建合同，覆盖原 train/validation/test 记录，不是新受保护评测或新的留出性能评估；未读原始信号、未重新拟合。</p>
<p>模型采用个体 PCA、每被试固定尺度与 EEG:HbO:HbR=1:4:4 权重。S 是保存的模型 Hb 预测，O=X−S。评分使用完整有效窗口、排除末格；REFED 短记录保留实际时长。以去均值标准差定义 a=SD(S)/SD(X)，b=SD(O)/SD(X)，r=corr(X,S)。恒等式 b²=1+a²−2ar，corr(S,O)=(r−a)/b；逐被试直接读取一个保存数组独立核验，共 206 个分量，最大差约 1.2×10⁻¹⁴。</p>
<p>严重抵消定义为 a≥2、b≥1、corr(S,O)≤−0.8，是本次描述性工程阈值。对于 O=X−S，a≥2 本身已保证后两项，因此它实质上是“至少两倍的波动放大”，不是三个独立证据。另列 1.5/3/5 倍敏感性和 corr(S,X)&lt;0.5 的形态失真。发生率先在记录内取平均，再记录等权汇总到被试，最后被试等权；表中幅度比和形态相关性中位数/P90来自全部有效窗口×通道。窗口存在重叠，不作为独立被试。95%区间为 2,000 次被试 bootstrap 的逐项描述区间，未作多重比较校正。</p>
'''
    text+=table(pd.DataFrame(overview))
    text+='''<p>Visual 16/16 名被试均出现严重 HbO 抵消，各被试发生率为 42.8%–58.7%；REFED 为 31/32 名，范围约 0%–38.9%。Visual 还有约 29.8% 的 HbO 拟合满足 corr(S,X)≥0.8 且残差波动≤输入一半，说明良好与失真重建并存。Single/Sim 的严重放大罕见，但其残差也并非在所有窗口都很小。</p><img src="severity.png"><p>上排每点是一名被试，黑线为被试均值；下排按每名被试、每种 Hb 的输入 SD 四分位分组，再计算组内发生率并对被试等权。Q1 最弱，Q4 最强；该分层的记录权重与总体率不同，专用于幅度关联。</p>
<h2>2. 弱幅度是明显风险因素，Visual 还存在明确通道分组</h2>
<p>REFED HbO 从最弱四分位到最强四分位，严重抵消率为 41.95%、8.39%、0.38%、0.03%。Visual 对应 99.94%、93.42%、20.08%、0%。因此局部弱幅度有很强关联；这不等于所有小幅度信号都是伪迹或所有大幅度信号都可靠。</p>
<p>按发布者 fNIRS_to_EEG_channel_reference.xlsx 的既有标签，Visual 每侧 24 通道中，14 个有 EEG 对应位置、10 个记为“-”。这不是依据结果拟合出的分组。有 EEG 标签组的 HbO 输入 SD/被试校准 SD 平均仅 0.029，另组为 1.104；输出 S 则为 0.092 与 1.111。严重抵消率为 88.90% 对 3.08%，平均 corr(S,X) 为 0.278 对 0.891。HbR 同方向：71.05% 对 0.16%。该分组均值先记录、后被试等权。</p>
<p>14 个 EEG 对应通道为 CH1、3、4、7、8、10、11、12、13、14、16、18、21、23，两个 Probe 均适用。图中 Probe1:CH13 属于这一弱幅度组。现有读入代码按 CH 数值列读取 Oxy/Deoxy，未见以 EEG 标签给波形重新缩放的分支；几何插值作用于坐标。这些结果定位了下一步应检查的具体通道边界，尚不能确定采集增益、探头布局或上游处理中的哪一步造成幅度差。</p><img src="visual_channel_classes.png"><p>每点是一名被试；颜色和横轴共同区分两类通道，黑线为均值。三个面板分别说明输入大小、抵消发生率和形态保持，坐标各有独立单位。</p>
<h2>3. 频谱与形态：Visual 并非整体更慢</h2>
<p>复用既有、读取结果前已固定的频谱面板：每记录首/中/末窗口×首/中/末位置，共计划 5,073 个、共同数值合格 5,050 个。主比较仅用 90 秒记录，10 Hz、899 个评分点，Hann 周期窗、常数去趋势、无新增滤波或补零。谱重心是全部正频率的功率加权平均；快变比例为&gt;0.2 Hz功率/全部正频率功率；mobility是离散导数SD/信号SD，单位 s⁻¹。先记录内平均，再记录、被试等权。这是覆盖全部被试的固定位置面板，不是全部通道的频谱普查。</p>
<p>HbO 的输入谱重心 Single/Sim/REFED/Visual 为 0.0501/0.0503/0.0512/0.0680 Hz；高于0.2 Hz的功率比例为 0.25%/0.27%/0.96%/2.47%；mobility 为 0.366/0.376/0.415/0.502 s⁻¹。REFED 谱重心接近两个对照，Visual 更高。因此“小幅度”与“低频、平滑”必须分开：数据支持前者与抵消强关联，不支持用整体更平滑统一解释两个数据集。</p>
<p>相同面板中，S 的 HbO 谱重心为 0.0442/0.0476/0.0792/0.0923 Hz，REFED、Visual 的模型预测更偏向快变。实测 HbO/HbR 相关性为 −0.139/−0.041/−0.366/−0.578，模型预测为 −0.758/−0.733/−0.828/−0.935。这是模型输出形态被共同动力学约束的证据，不能据此将实测中不符合反相关的部分全部判为噪声。</p><img src="input_spectra.png"><p>每点为一名被试的固定面板平均，黑线为数据集被试等权均值。HbO/HbR分别成行，三列单位不同；快变比例反映当前缓存输入预处理后的谱，不是未经处理的设备原始谱。</p>
<h2>4. 原图所属 14 名被试：并非只有画出的极端通道</h2>
<p>原图分析使用较早的训练参考 PCA 设置，14 名验证被试每人一个90秒窗口、3个ROI，合计每种 Hb 42个点；频谱和形态分析范围为5–85秒。原图选各数据集被试中位功率比最大的被试，再选其中最大的ROI。下表使用全部42点，严重阈值与上文一致。spectral_ratio 是0.0375–0.2 Hz内 [P(S)+P(O)]/P(X)，不是 P(S+O)/P(X)；相加信号 S+O=X 的功率比恒为1。</p>
'''
    text+=table(panel)
    text+='''<p>原图中 REFED/Visual 的 HbO 各5/9个ROI严重放大，Single/Sim均0/12。该小样本说明现象超出示例，但全数据发生率应使用上面的 v3 普查；两代PCA及时间支持不同，不能将差异直接解释成模型改进效果。</p>
<h2>5. 机制解释与处理建议</h2>
<p><b>实现与代数已经确认：</b>模型联合拟合EEG代理、HbO和HbR，使用全被试固定的分量尺度；Hb的S直接取预测，O取残差。局部输入幅度极小时，按被试尺度计算的绝对误差仍可较小，而相对局部输入的误差和幅度比很大。若S远大于X，O自然趋近−S，因此相加闭合不是分解质量标准。</p>
<p><b>统计支持的解释：</b>REFED低幅度窗口、Visual的14个弱幅度通道组更容易被模型输出压过；与此同时，S的频谱和HbO/HbR耦合向模型共同动力学偏移。这与统一尺度下的联合拟合折中一致。REFED完整90秒的HbO抵消率约13.22%，短记录约10.65%（此处为窗口通道池化率），短记录不是主因。当前分析没有重新拟合，尚未通过固定其它条件的消融确定各机制的因果贡献。</p>
<p><b>建议：</b>优先核查Visual两类通道在发布数据、设备增益/通道定义与缓存中的幅度对应关系；随后在固定窗口、固定模型和PCA下比较现有尺度与带可靠性下限的局部/通道尺度，避免将近常数噪声直接放大。另以单模态Hb拟合对照联合拟合，定位EEG驱动冲突；可比较有幅度/频带约束的Hb监督目标构造。判断应同时看S形态、O大小、局部与统一尺度误差及失败率。现有Visual弱通道的S不宜直接当作“保留原形态、只去除少量噪声”的监督目标。上述比较尚未运行。</p>
<p>跨数据集原始浓度单位未全部核定，因此幅度以同一数据内的局部SD比和被试校准SD比较，不直接比较缓存绝对数值。大比值同时报告P90和发生率；极小分母不会被裁剪或替换为零。</p>
<h2>可复核证据</h2><p><a href="analysis_config.json">分析范围</a> · <a href="summary.json">完成与核验</a> · <a href="dataset_severity.csv">总体幅度/形态/阈值敏感性</a> · <a href="subject_severity.csv">逐被试发生率</a> · <a href="amplitude_quartiles.csv">幅度分层</a> · <a href="visual_channel_metadata.json">原始通道标签来源</a> · <a href="visual_subject_channel_classes.csv">通道分组逐被试结果</a> · <a href="dataset_input_spectra_90s.csv">等时长频谱</a> · <a href="figure_panel_channels.csv">原图完整通道统计</a> · <a href="duration_group_rates.csv">短记录分层</a> · <a href="split_rates.csv">原始分区分层</a>。逐被试CSV与JSON保留确切记录/窗口/通道和失败身份；来源合同及谱面板见配置路径。</p></html>'''
    (out/'report.html').write_text(text,encoding='utf-8')


def fnirs_examples(cfg, export):
    """Four real retained Hb records, in their frozen common-Hb model scale."""
    import csv
    from scipy.io import loadmat, whosmat
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(cfg['output']);run=Path(cfg['run']);export=Path(export).resolve()
    if export.parent!=out or export.exists():
        raise ValueError('Use a fresh example export directly below the audit')
    coordinates=json.loads((run/'coordinates.json').read_text())
    datasets=['eeg_fnirs_single_trial','simultaneous_eeg_nirs','refed','visual_cognitive_motivation']
    names=['Single','Sim','REFED','Visual'];selected=[];signals=[];metrics=[];units=[]
    for ds in datasets:
        frame=pd.concat([pd.read_csv(out/'subjects'/(sub.replace('/','__')+'.csv'),dtype={'subject':str})
                         for sub in cfg['subjects'] if sub.startswith(ds+'/')],ignore_index=True)
        frame=frame[(frame.split=='validation')&(frame.duration_s==90)&frame.defined]
        keys=['dataset','subject','record','window','pair']
        pairs=frame.pivot(index=keys,columns='component',values=['s_sd_ratio','o_sd_ratio','correlation_sx'])
        problematic=ds in ['refed','visual_cognitive_motivation']
        mask=(pairs['s_sd_ratio']>=2).all(axis=1) if problematic else (
            (pairs['correlation_sx']>=.8).all(axis=1)&(pairs['o_sd_ratio']<=.5).all(axis=1))
        candidates=pairs[mask]
        if candidates.empty:raise ValueError('No matched HbO/HbR example: '+ds)
        values=np.log(candidates['s_sd_ratio'])
        score=((values-values.median())**2).sum(axis=1)
        key=score.sort_values(kind='stable').index[0]
        row=dict(zip(keys,key));row.update(candidate_count=len(candidates),validation_complete_pairs=len(pairs),problematic=problematic)
        matches=[]
        for path in (run/'tasks'/ds/row['subject']/row['record']).glob(row['window']+'__p*.json'):
            d=json.loads(path.read_text())
            if d['identity']['pair']['id']==row['pair']:matches.append((path,d))
        if len(matches)!=1:raise ValueError('Ambiguous retained task')
        path,d=matches[0];co=coordinates['individual'][ds+'__'+row['subject']]
        if not d['qualified'] or d['identity']['split']!='validation':raise ValueError('Unexpected task boundary')
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:
            t=a['time_s'].copy();x=a['target'][:,1:].copy();s=a['prediction'][:,1:].copy();o=a['residual'][:,1:].copy()
            scale=a['scale'][1:].copy();reference=a['reference_unscaled'][1:].copy()
            use=np.all(a['fit_mask'] & a['prediction_valid'],axis=1)
        np.testing.assert_allclose(scale,[co['hb_factor']]*2,rtol=0,atol=0)
        np.testing.assert_allclose(x,s+o,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(x[:50].mean(axis=0),0,atol=1e-10)
        with np.load(run/d['feature_source'],allow_pickle=False) as a:
            original=a['hb'][:,d['identity']['pair']['native_index'],:]
            np.testing.assert_allclose(original-original[:50].mean(0),x,rtol=1e-10,atol=1e-10)
        unit=dict(dataset=ds,**d['provenance']['hb'])
        source=[Path(v['path']) for v in unit['source_files']]
        if ds=='eeg_fnirs_single_trial':
            p=next(p for p in source if 'NIRS_01-29' in str(p))
            native=loadmat(p,variable_names=['cnt'],squeeze_me=True,struct_as_record=False)['cnt'][int(row['record'].rsplit('_',1)[1])]
            unit['direct_check']=dict(path=str(p),field='cnt[session].yUnit',value=str(native.yUnit),signal=str(native.signal))
            if str(native.yUnit)!='V':raise ValueError('Unexpected Single native unit')
        elif ds=='simultaneous_eeg_nirs':
            p=next(p for p in source if '-NIRS/' in str(p))
            native=loadmat(p,variable_names=[row['record']],squeeze_me=True,struct_as_record=False)[row['record']]
            unit['direct_check']=dict(path=str(p),oxy=str(native.oxy.yUnit),deoxy=str(native.deoxy.yUnit),signal=str(native.oxy.signal))
            if str(native.oxy.yUnit)!='mmol/L' or str(native.deoxy.yUnit)!='mmol/L':raise ValueError('Unexpected Sim native unit')
        elif ds=='refed':
            p=next(p for p in source if p.name=='fNIRS_videos.mat')
            unit['direct_check']=dict(path=str(p),variables=whosmat(p),conclusion='numeric video tensors; no standalone unit field; publisher README identifies HbO/HbR but gives no concentration unit')
        else:
            headers={}
            for p in source:
                if p.suffix!='.csv':continue
                lines=[]
                with p.open(encoding='utf-8-sig') as f:
                    for line in f:
                        if line.strip()=='Data':break
                        lines.append(line.rstrip('\r\n'))
                headers[str(p)]=lines
            exception_rows={p:next(row[1:] for row in csv.reader(lines) if row and row[0]=='Exception Ch')
                            for p,lines in headers.items()}
            unit['direct_check']=dict(headers=headers,exception_ch=exception_rows,
                conclusion='Oxy/Deoxy exports; inspected CSV headers have no explicit concentration unit; Exception Ch retained as metadata, no new channel exclusion')
        units.append(unit)
        row.update(task=str(path),window_metadata=d['identity']['window'],hb_factor=co['hb_factor'],
                   subject_component_sd=co['sd'][1:],reference_cache_units=reference.tolist(),
                   output_unit=unit['output_unit'],normalization='saved common subject Hb factor; existing first-5s reference; no new fitted scale')
        selected.append(row);signals.append(dict(t=t,x=x*scale,s=s*scale,o=o*scale,use=use))
        for j,c in enumerate(['HbO','HbR']):
            xx=x[use,j];ss=s[use,j];oo=o[use,j]
            m=dict(dataset=ds,subject=row['subject'],record=row['record'],pair=row['pair'],component=c,
                   input_sd_model=float(xx.std()*scale[j]),s_sd_model=float(ss.std()*scale[j]),o_sd_model=float(oo.std()*scale[j]),
                   input_sd_cache=float(xx.std()),correlation_so=float(np.corrcoef(ss,oo)[0,1]),
                   **{k:v for k,v in residual_geometry(ss.std()/xx.std(),np.corrcoef(xx,ss)[0,1]).items() if k!='correlation_so'})
            retained=next(w for w in d['fits']['tak_tau_beta']['waveform'] if w['component']==c)
            np.testing.assert_allclose([m['s_sd_ratio'],m['correlation_sx']],
                [retained['centered_amplitude_ratio'],retained['pearson_r']],rtol=1e-10,atol=1e-10)
            metrics.append(m)
    export.mkdir()
    write_json(export/'selection.json',dict(rule='validation only, 90s, both Hb components; REFED/Visual a>=2; Single/Sim corr>=0.8 and residual SD ratio<=0.5; nearest squared distance to componentwise median log amplitude ratio in each eligible set; stable identity tie break',
        purpose='outcome-selected illustration; not dataset prevalence',rows=selected))
    write_json(export/'unit_checks.json',units)
    pd.DataFrame(metrics).to_csv(export/'metrics.csv',index=False)
    samples=[]
    for row,a in zip(selected,signals):
        for j,c in enumerate(['HbO','HbR']):
            samples.extend(dict(dataset=row['dataset'],subject=row['subject'],record=row['record'],pair=row['pair'],component=c,
                time_s=float(t),record_time_s=float(t+row['window_metadata']['hb_start_s']),scored=bool(use),
                X=float(x),S=float(s),O=float(o)) for t,use,x,s,o in zip(a['t'],a['use'],a['x'][:,j],a['s'][:,j],a['o'][:,j]))
    pd.DataFrame(samples).to_csv(export/'normalized_samples.csv',index=False)
    for full,filename in [(False,'measured_HbO_HbR.png'),(True,'decomposition_HbO_HbR.png')]:
        fig,axes=plt.subplots(4,2,figsize=(15,13),layout='constrained')
        shared=max(abs(a['x'][a['use']]).max() for a in signals)*1.08
        for i,(name,row,a) in enumerate(zip(names,selected,signals)):
            for j,c in enumerate(['HbO','HbR']):
                ax=axes[i,j];use=a['use'];t=a['t'][use]
                if full:
                    ax.plot(t,a['s'][use,j],color='#2479bf',lw=1.3,label='S: model prediction')
                    ax.plot(t,a['o'][use,j],color='#e76f28',lw=1.15,ls='--',label='O: X - S')
                ax.plot(t,a['x'][use,j],color='#262b33',lw=1.6,label='X: measured input',zorder=4)
                ax.axvspan(0,5,color='#b5bdc4',alpha=.18);ax.axhline(0,color='gray',lw=.6)
                m=next(m for m in metrics if m['dataset']==row['dataset'] and m['component']==c)
                info=(f'SD(S)/SD(X)={m["s_sd_ratio"]:.2f}; SD(O)/SD(X)={m["o_sd_ratio"]:.2f}; r(S,X)={m["correlation_sx"]:.2f}' if full else
                      f'Input SD={m["input_sd_model"]:.5f}; saved Hb factor={row["hb_factor"]:.6g}')
                ax.set_title(f'{name} / {row["subject"]} / {row["record"]} / {row["pair"]} | {c}\n{info}',fontsize=10)
                ax.set(xlim=(0,90),xlabel='Time from window start (s)',ylabel='Hb in frozen subject model scale')
                if not full:ax.set_ylim(-shared,shared)
                ax.grid(alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=11)
        fig.suptitle('Same saved normalization; independent panel y limits' if full else 'Measured HbO/HbR | same saved normalization and common y limits',fontsize=14)
        fig.savefig(export/filename,dpi=220);plt.close(fig)
    table=pd.DataFrame(metrics)[['dataset','subject','pair','component','input_sd_model','s_sd_ratio','o_sd_ratio','correlation_sx','correlation_so']]
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>四数据集实际 HbO/HbR 对照</title><style>body{font-family:sans-serif;max-width:1500px;margin:auto;padding:25px;line-height:1.8}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccc;padding:6px}</style>
<h1>REFED、Visual 问题记录与 Single、Sim 良好记录对照</h1>
<p>每个数据集一段真实记录，同一窗口、同一位置同时展示 HbO/HbR。X 是已进入模型的实测 Hb：保留 v5/none 的全记录0.01–0.2 Hz滤波与10 Hz采样；没有新增滤波或重新拟合。S与O来自已完成的v3个体PCA、1:4:4联合拟合。选样用于解释具体波形，不替代总体发生率。</p>
<h2>归一化完全复用已冻结方案</h2><p>使用 coordinates.json 中每个被试保存的 hb_factor，将现有缓存坐标乘回模型坐标；同一被试的HbO/HbR、所有记录及通道共用该因子。此前的定义为 0.025 / sqrt((Var(HbO)+Var(HbR))/2)，方差来自该被试全部允许的校准数据。沿用每窗口前5秒参考，不按当前窗口或单个Hb成分重新标准化。这里的纵轴是模型无量纲坐标，不是统一物理浓度。</p>
<h2>原始单位检查</h2><table><tr><th>数据集</th><th>所选原始文件直接检查</th><th>既有输入转换</th></tr>
<tr><td>Single</td><td>subject 14，NIRS cnt.mat 的 session_04：yUnit=V，signal=NIRS (low wavelength, high wavelength)</td><td>光电电压经原有近似MBLL得到相对Hb，再乘保存的共同Hb因子；V不是Hb浓度单位。</td></tr>
<tr><td>Sim</td><td>VP002-NIRS/cnt_wg.mat：oxy.yUnit与deoxy.yUnit均为mmol/L</td><td>既有读取已转换为µM，继续使用保存的因子；本次不重复乘1000。</td></tr>
<tr><td>REFED</td><td>17/fNIRS_videos.mat仅含视频数值张量；原始README说明HbO/HbR/HbT及3种Abs，未给浓度单位。</td><td>保留relative:refed坐标并复用保存的因子。</td></tr>
<tr><td>Visual</td><td>S06_Part2_Probe1_Oxy/Deoxy.csv头及发布说明未明确浓度单位；Oxy/Deoxy语义明确。</td><td>保留relative:visual坐标并复用保存的因子，不据设备型号假定µM。</td></tr></table>
<p><b>此次原始文件头核查的新线索：</b>所选Visual记录的Oxy/Deoxy头均有Exception Ch字段：CH2、5、6、9、15、17、19、20、22、24为1，其余14通道为0。这个分组与上次统计中的高幅度10通道/低幅度14通道完全对应；所选CH12为0。因此，不能仅凭高幅度组拟合较好就将其当作正常生理标尺，需先核查该字段的设备语义和通道质量。本次保留字段作为元数据，不新增通道排除、不重算校准尺度；该发现目前直接对应这份所选记录。</p>
<h2>先比较实测记录：全部面板相同纵轴</h2><img src="measured_HbO_HbR.png"><p>黑线为实测X；灰色0–5秒区间为已有参考段。横轴是窗口相对时间，四段记录不表示同一事件。所有面板共享纵轴，REFED/Visual在模型尺度下较小的实际波动被保留下来。</p>
<h2>再看同一记录的 S/O 分解</h2><img src="decomposition_HbO_HbR.png"><p>黑色实线X、蓝色实线S、橙色虚线O。为看清弱信号的形态，各面板纵轴独立缩放，但数据值、归一化与上图完全相同。Single/Sim的S较好跟随实测峰谷；REFED/Visual的S出现更大的波动，O反向补偿。幅度比均用完整有效89.9秒的去均值SD计算；最后一格沿用原合同排除。相关性用于核验形态保持。</p>
'''
    body+=table.to_html(index=False,float_format=lambda v:f'{v:.5g}')
    body+='''<h2>可复核身份与选样</h2><p>仅使用验证分区完整90秒、两种Hb指标均有效的保留拟合。REFED/Visual要求两种Hb均SD(S)/SD(X)≥2；Single/Sim要求两种Hb均corr(S,X)≥0.8且SD(O)/SD(X)≤0.5。各集合内选择两种Hb的log幅度比最接近各自中位数的样例，稳定排序处理并列。</p>'''
    body+=pd.DataFrame([{k:r[k] for k in ['dataset','subject','record','window','pair','candidate_count','validation_complete_pairs','hb_factor']} for r in selected]).to_html(index=False)
    body+='''<p><a href="selection.json">精确源路径、记录起点、参考值与选择分母</a> · <a href="unit_checks.json">原始单位字段/CSV头核查</a> · <a href="metrics.csv">逐样例指标</a> · <a href="normalized_samples.csv">同源完整数值</a> · <a href="verification.json">数组重放与归一化核验</a></p></html>'''
    (export/'report.html').write_text(body,encoding='utf-8')
    write_json(export/'verification.json',dict(status='passed',checks=['task qualified and validation split','Hb common scale equals frozen coordinate exactly',
        'target equals prepared measured Hb minus original first-5s mean','S+O closure','original metrics replay'],new_fits=0,new_normalization_fits=0,examples=4))


def fnirs_main():
    import sys
    p=argparse.ArgumentParser(description='Read-only Hb morphology/cancellation census')
    p.add_argument('--fnirs-audit',action='store_true')
    for key in ['run','features','comparison','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stage',choices=['plan','pilot','run','summarize','report','examples'],required=True)
    p.add_argument('--example-export',type=Path)
    p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();out=a.output.resolve()
    if out.parent!=a.run.resolve():raise ValueError('Use a fresh child of the completed v3 cache')
    s=json.loads((a.run/'summary.json').read_text());v=json.loads((a.run/'verification.json').read_text())
    contract=json.loads((a.run/'resolved_config.json').read_text())
    if s['status']!='completed' or s['schema']!='shared_driver_tak_subject_cache_v3' or v['status']!='passed' or v['issues']:
        raise ValueError('Completed verified v3 cache required')
    if contract['data']['protected_data']!='forbidden':raise ValueError('Public reconstruction only')
    parent=json.loads((a.comparison/'analysis_config.json').read_text())
    if Path(parent['v3']).resolve()!=a.run.resolve():raise ValueError('Wrong spectral source')
    cfg=dict(schema='fnirs_cancellation_census_v1',**{k:str(getattr(a,k).resolve()) for k in ['run','features','comparison','output']},
             subjects=parent['subjects'],boundary='completed public cache metadata; one verification array per subject; retained feature/spectral tables; no raw/protected reads or fits',
             support='full scored support; final bin excluded; original 14-subject panel retains 5-85s',
             amplitude='centered SD; O=X-S; correlation and variance reconstructed algebraically from retained waveform metrics',
             spectral='reuse original 5073 selected tasks, 5050 common numerical successes; all records first/middle/last window and site')
    if a.stage=='plan':
        out.mkdir(exist_ok=False);(out/'subjects').mkdir();write_json(out/'analysis_config.json',cfg);return
    if json.loads((out/'analysis_config.json').read_text())!=cfg:raise ValueError('Frozen config changed')
    if a.stage=='examples':
        if a.example_export is None:p.error('--example-export is required for examples')
        fnirs_examples(cfg,a.example_export);return
    if a.stage=='report':fnirs_report(cfg);return
    if a.stage=='summarize':fnirs_summarize(cfg);return
    if not os.environ.get('SSM_SYSTEMD_UNIT'):raise ValueError('Use durable user systemd')
    selected=cfg['subjects']
    if a.stage=='pilot':selected=[next(s for s in selected if s.startswith(ds+'/')) for ds in sorted({s.split('/')[0] for s in selected})]
    start=time.monotonic();done=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(fnirs_subject,[(cfg,s) for s in selected]):
            done.append(result);print(json.dumps({k:result[k] for k in ['subject','planned','qualified','seconds']}),flush=True)
            write_json(out/(a.stage+'_progress.json'),dict(completed=len(done),total=len(selected)))
    write_json(out/(a.stage+'_execution.json'),dict(workers=a.workers,seconds=time.monotonic()-start,
        planned=sum(s['planned'] for s in done),peak_worker_mib=max(s['peak_rss_mib'] for s in done),
        supervisor=os.environ['SSM_SYSTEMD_UNIT']))
    if a.stage=='run':fnirs_summarize(cfg)


if __name__=='__main__':
    import sys
    fnirs_main() if '--fnirs-audit' in sys.argv else main()
