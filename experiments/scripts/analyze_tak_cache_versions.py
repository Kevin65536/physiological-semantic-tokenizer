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


def fnirs_source_audit(cfg, export):
    """Source quality metadata census and bounded calibration-moment replay.

    No production masks/scales change. REFED's undocumented bit polarity is
    deliberately kept as flag0/flag1, unlike the documented Hitachi exclusion.
    """
    import csv
    import sys
    run=Path(cfg['run']);out=Path(cfg['output']);export=Path(export).resolve()
    if export.parent!=out or export.exists():raise ValueError('Fresh audit child required')
    project=run.parents[4];sys.path.insert(0,str(project))
    from src.data.ssm_prepared import fit_feature_coordinate
    plan=json.loads((run/'plan.json').read_text())
    coordinates=json.loads((run/'coordinates.json').read_text())['individual']
    records={r['cache_key']:r for r in plan['records']}
    headers=[];flags={}
    for key,r in records.items():
        if r['dataset']!='visual_cognitive_motivation' or not r['cache_windows']:continue
        files=[];bits=[]
        for source in r['source_identity']:
            p=Path(source['path'])
            if p.suffix!='.csv':continue
            metadata={}
            with p.open(encoding='utf-8-sig') as f:
                for line in f:
                    if line.strip()=='Data':break
                    row=next(csv.reader([line]))
                    if row and row[0] in ['Exception Ch','Analog Gain','Digital Gain','Wave[nm]','File Version']:
                        metadata[row[0]]=row[1:]
            flag=np.array(metadata['Exception Ch'],int)
            if flag.shape!=(24,) or not np.isin(flag,[0,1]).all():raise ValueError('Invalid channel header')
            bits.append(flag);files.append(dict(path=str(p),metadata=metadata))
        if len(bits)!=2:raise ValueError('Both chromophore headers required')
        np.testing.assert_array_equal(bits[0],bits[1]);flags[key]=bits[0]
        admitted=[p['native_index'] for p in r['pairs'] if p['status']=='available']
        headers.append(dict(subject=r['subject'],record=r['record'],cache_key=key,files=files,
                            excluded_channels=(np.flatnonzero(bits[0])+1).tolist(),
                            admitted_excluded_count=int(bits[0][admitted].sum()),admitted_pairs=len(admitted)))
    reservations=pd.read_csv(project/'data/REFED-dataset/fNIRS_reservations.csv').set_index('sub_id')
    if reservations.shape!=(32,51) or not np.isin(reservations,[0,1]).all():raise ValueError('Unexpected REFED table')
    for key,r in records.items():
        if r['dataset']=='refed':flags[key]=reservations.loc[int(r['subject'])].to_numpy(int)
    groups=[]
    for subject in cfg['subjects']:
        ds,sub=subject.split('/')
        if ds not in ['refed','visual_cognitive_motivation']:continue
        f=pd.read_csv(out/'subjects'/(subject.replace('/','__')+'.csv'),dtype={'subject':str})
        f=f[f.defined].copy()
        f['source_flag']=[flags[ds+'/'+sub+'/'+r.record][int(r.pair.rsplit('CH',1)[1])-1] for r in f.itertuples()]
        f['severe']=((f.s_sd_ratio>=2)&(f.o_sd_ratio>=1)&(f.correlation_so<=-.8)).astype(float)
        g=f.groupby(['component','source_flag','record'])[['severe','x_sd_subject','s_sd_subject','correlation_sx']].mean()
        g=g.groupby(['component','source_flag']).mean().reset_index()
        g['dataset']=ds;g['subject']=sub;groups.append(g)
    groups=pd.concat(groups,ignore_index=True)
    # Only the two subjects shown in the immediately preceding examples.
    moments=[]
    for key in ['visual_cognitive_motivation__S06','refed__17']:
        co=coordinates[key];acc={k:[0,np.zeros(2),np.zeros(2)] for k in ['all','flag0','flag1']}
        for source in co['sources']:
            r=records[source['record_key']]
            with np.load(run/'features'/source['record_key']/(source['window_key']+'.npz'),allow_pickle=False) as a:
                for pair in r['pairs']:
                    i=pair['native_index'];channels=pair['channel_sets']['nearest6']
                    if pair['status']!='available' or not a['hb_valid'][i] or not a['eeg_valid'][channels].all():continue
                    h=a['hb'][:,i];flag=flags[source['record_key']][i]
                    for k in ['all','flag'+str(flag)]:
                        v=acc[k];v[0]+=len(h);v[1]+=h.sum(0);v[2]+=(h*h).sum(0)
        result=dict(subject=key,stored_hb_factor=co['hb_factor'],groups={})
        overall_mean=acc['all'][1]/acc['all'][0]
        total_centered=(acc['all'][2]-acc['all'][0]*overall_mean**2).sum()
        for k,(n,s,q) in acc.items():
            variance=q/n-(s/n)**2;sd=np.sqrt(variance.mean())
            result['groups'][k]=dict(samples=n,component_sd=np.sqrt(variance).tolist(),common_sd=float(sd),
                diagnostic_factor=float(.025/sd),factor_over_stored=float(.025/sd/co['hb_factor']),
                contribution_to_existing_pooled_variance=float((q-2*overall_mean*s+n*overall_mean**2).sum()/total_centered))
        np.testing.assert_allclose(result['groups']['all']['diagnostic_factor'],co['hb_factor'],rtol=1e-10)
        assert acc['all'][0]==co['training_samples']
        moments.append(result)
    # A common positive unit conversion cancels against the same fitted Hb scale.
    rng=np.random.default_rng(17);eeg=rng.normal(size=(200,30));hb=rng.normal(size=(200,2))*[2,.5]
    a=fit_feature_coordinate([(eeg,hb)]);b=fit_feature_coordinate([(eeg,hb*1000)])
    error=float(np.max(abs(hb*a['hb_factor']-hb*1000*b['hb_factor'])))
    np.testing.assert_allclose(hb*a['hb_factor'],hb*1000*b['hb_factor'],rtol=1e-12,atol=1e-12)
    export.mkdir()
    write_json(export/'analysis_config.json',dict(schema='fnirs_source_metadata_review_v1',source_run=str(run),
        source_census=str(out),raw_reads='headers only; no new raw waveform reads',
        calibration_replay_subjects=['visual_cognitive_motivation__S06','refed__17'],
        scope='read-only registered public metadata and retained prepared calibration inputs; no modified outputs, normalization or fits'))
    write_json(export/'visual_headers.json',headers)
    groups.to_csv(export/'subject_source_flag_groups.csv',index=False)
    groups.groupby(['dataset','component','source_flag']).mean(numeric_only=True).to_csv(export/'dataset_source_flag_groups.csv')
    reservations.to_csv(export/'refed_reservations_snapshot.csv')
    write_json(export/'calibration_moment_replay.json',moments)
    write_json(export/'summary.json',dict(status='completed',visual_records=len(headers),visual_subjects=len({h['subject'] for h in headers}),
        visual_files=sum(len(h['files']) for h in headers),
        visual_records_with_excluded_channels=sum(bool(h['excluded_channels']) for h in headers),
        visual_excluded_record_channels=sum(len(h['excluded_channels']) for h in headers),
        visual_admitted_excluded_record_channels=sum(h['admitted_excluded_count'] for h in headers),
        refed_flag0_subject_channels=int((reservations==0).to_numpy().sum()),
        refed_flag1_subject_channels=int((reservations==1).to_numpy().sum()),
        refed_bit_polarity='unverified; do not infer from name or magnitude',
        unit_scale_invariance_max_error=error,calibration_replays=2,new_fits=0,production_mutations=0))


def visual_mask_subject(payload):
    cfg,subject=payload
    start=time.monotonic();run=Path(cfg['run']);out=Path(cfg['export'])/'subjects'/subject
    if out.with_suffix('.json').exists():return json.loads(out.with_suffix('.json').read_text())
    co=json.loads((run/'coordinates.json').read_text())['individual']['visual_cognitive_motivation__'+subject]
    records={r['cache_key']:r for r in json.loads((run/'plan.json').read_text())['records'] if r['dataset']=='visual_cognitive_motivation' and r['subject']==subject}
    headers={r['cache_key']:r for r in json.loads(Path(cfg['headers']).read_text()) if r['subject']==subject}
    acc={k:[0,np.zeros(2),np.zeros(2)] for k in ['all','usable','excluded']};counts={};wave=None
    chosen_record=sorted(headers)[0]
    chosen_windows=[v for v in co['sources'] if v['record_key']==chosen_record]
    chosen_window='w00005' if any(v['window_key']=='w00005' for v in chosen_windows) else chosen_windows[len(chosen_windows)//2]['window_key']
    for source in co['sources']:
        r=records[source['record_key']];header=headers[source['record_key']]
        flags=np.zeros(24,bool);flags[np.array(header['excluded_channels'])-1]=True
        with np.load(run/'features'/source['record_key']/(source['window_key']+'.npz'),allow_pickle=False) as a:
            valid_e=a['eeg_valid'];valid_h=a['hb_valid'];h=a['hb']
            if valid_e.shape!=(31,) or valid_h.shape!=(24,):raise ValueError('Visual channel count changed')
            if source['record_key'] not in counts:
                counts[source['record_key']]=dict(subject=subject,record=r['record'],eeg_names=r['native_eeg_channel_identity']['channel_names'],
                    eeg_valid_all_windows=valid_e.copy(),hb_valid_all_windows=valid_h.copy(),windows=0,excluded=np.flatnonzero(flags).tolist())
            cr=counts[source['record_key']];cr['eeg_valid_all_windows'] &= valid_e;cr['hb_valid_all_windows'] &= valid_h;cr['windows']+=1
            for pair in r['pairs']:
                i=pair['native_index'];chs=pair['channel_sets']['nearest6']
                if pair['status']!='available' or not valid_h[i] or not valid_e[chs].all():continue
                x=h[:,i]
                if not np.isfinite(x).all():raise ValueError('Nonfinite admitted Hb')
                for k in ['all','excluded' if flags[i] else 'usable']:
                    v=acc[k];v[0]+=len(x);v[1]+=x.sum(0);v[2]+=(x*x).sum(0)
            if source['record_key']==chosen_record and source['window_key']==chosen_window:
                wave=dict(hb=h.copy(),time_s=a['time_s'].copy(),excluded=flags,valid=valid_h.copy())
    groups={};overall_mean=acc['all'][1]/acc['all'][0]
    total=(acc['all'][2]-acc['all'][0]*overall_mean**2).sum()
    for k,(n,s,q) in acc.items():
        if not n:raise ValueError('Missing channel group')
        variance=q/n-(s/n)**2;common_sd=np.sqrt(variance.mean())
        groups[k]=dict(samples=n,component_sd=np.sqrt(variance).tolist(),common_sd=float(common_sd),factor=float(.025/common_sd),
                       original_variance_fraction=float((q-2*overall_mean*s+n*overall_mean**2).sum()/total))
    np.testing.assert_allclose(groups['all']['factor'],co['hb_factor'],rtol=1e-10)
    assert acc['all'][0]==co['training_samples']
    for c in counts.values():
        c['eeg_available']=int(c['eeg_valid_all_windows'].sum());c['hb_recorded']=int(c['hb_valid_all_windows'].sum())
        keep=c['hb_valid_all_windows'].copy();keep[c['excluded']]=False
        c['hb_usable']=int(keep.sum());c['hb_excluded']=24-c['hb_usable']
        c['usable_channels']=(np.flatnonzero(keep)+1).tolist();c['excluded_channels']=(np.array(c.pop('excluded'))+1).tolist()
        c['eeg_valid_all_windows']=c['eeg_valid_all_windows'].tolist();c['hb_valid_all_windows']=c['hb_valid_all_windows'].tolist()
    if wave is None:raise ValueError('Missing selected window')
    x=wave['hb']-wave['hb'][:50].mean(0)
    # Same physical samples: only the frozen per-subject multiplicative factor differs.
    old=x*groups['all']['factor'];new=x*groups['usable']['factor'];ratio=groups['usable']['factor']/groups['all']['factor']
    np.testing.assert_allclose(new,old*ratio,rtol=1e-12,atol=1e-12)
    np.savez_compressed(out.with_suffix('.npz'),**wave,referenced=x,old=old,new=new)
    result=dict(subject=subject,groups=groups,multiplier=ratio,records=list(counts.values()),
        waveform=dict(record_key=chosen_record,window_key=chosen_window,window=next(w for w in records[chosen_record]['cache_windows'] if w['key']==chosen_window)),
        seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(out.with_suffix('.json'),result);return result


def visual_mask_report(cfg):
    import sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import mne
    sys.path.insert(0,str(Path(cfg['run']).parents[4]))
    from src.data.channel_geometry import project_head_topview
    out=Path(cfg['export']);run=Path(cfg['run'])
    figdir=out/'report_v2'
    figdir.mkdir(exist_ok=False)
    results=[json.loads((out/'subjects'/(s+'.json')).read_text()) for s in cfg['subjects']]
    counts=pd.DataFrame([r for s in results for r in s['records']]);counts.to_json(out/'record_counts.json',orient='records',indent=2)
    flat=counts[['subject','record','eeg_available','hb_recorded','hb_usable','hb_excluded','windows']].copy()
    flat.to_csv(out/'record_counts.csv',index=False)
    flat['session']=flat.record.str.replace(r'_Probe[12]$','',regex=True)
    sessions=flat.groupby(['subject','session']).agg(eeg_available=('eeg_available','min'),fnirs_locations=('hb_usable','sum'),excluded_locations=('hb_excluded','sum'),probes=('record','size')).reset_index()
    if not (sessions.probes==2).all():raise ValueError('Missing opposite probe')
    sessions['Hb_sequences']=2*sessions.fnirs_locations;sessions.to_csv(out/'session_counts.csv',index=False)
    rows=[]
    for s in results:
        row=dict(subject=s['subject'],old_factor=s['groups']['all']['factor'],usable_factor=s['groups']['usable']['factor'],multiplier=s['multiplier'],
                 old_common_sd=s['groups']['all']['common_sd'],usable_common_sd=s['groups']['usable']['common_sd'],
                 excluded_variance_fraction=s['groups']['excluded']['original_variance_fraction'],
                 usable_sd_after_old=s['groups']['usable']['common_sd']*s['groups']['all']['factor'],usable_sd_after_new=.025)
        for j,c in enumerate(['HbO','HbR']):
            row[c+'_usable_sd_old']=s['groups']['usable']['component_sd'][j]*row['old_factor']
            row[c+'_usable_sd_new']=s['groups']['usable']['component_sd'][j]*row['usable_factor']
        rows.append(row)
    scales=pd.DataFrame(rows);scales.to_csv(out/'subject_normalization.csv',index=False)
    records={r['cache_key']:r for r in json.loads((run/'plan.json').read_text())['records']}
    selected=next(s for s in results if s['subject']=='S06')
    example_records=[records['visual_cognitive_motivation/S06/S06_Part2_Probe'+str(i)] for i in [1,2]]
    fig,axes=plt.subplots(1,3,figsize=(17,6),layout='constrained')
    sensor_rows=[]
    def draw(ax,geometry,values,title,names,discrete=False):
        pos=np.array([project_head_topview(r,projection='azimuthal_equidistant') for r in geometry])*.095
        if discrete:
            im,_=mne.viz.plot_topomap(np.zeros(len(pos)),pos,axes=ax,sphere=(0,0,0,.095),sensors=False,
                contours=0,cmap=ListedColormap(['white','white']),vlim=(0,1),show=False)
            for flag,marker,color,label in [(0,'o','#157f78','Usable'),(1,'X','#b83e43','Excluded')]:
                take=np.asarray(values)==flag;ax.scatter(pos[take,0],pos[take,1],s=110,marker=marker,c=color,label=label,zorder=4)
        else:
            im,_=mne.viz.plot_topomap(np.asarray(values),pos,axes=ax,sphere=(0,0,0,.095),sensors=True,
                contours=0,cmap='Reds',vlim=(0,100),image_interp='linear',extrapolate='head',show=False)
        for (px,py),name in zip(pos,names):
            offset=(-12,-17) if name=='9' else (5,13) if name=='16' else (5,5)
            ax.annotate(name,(px,py),xytext=offset,textcoords='offset points',fontsize=8,zorder=6,
                bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.4),
                arrowprops=dict(arrowstyle='-',lw=.45,color='#555') if name in ['9','16'] else None)
        ax.set_title(title,fontsize=12);return im
    geometry=example_records[0]['geometry_identity']['eeg']
    draw(axes[0],geometry,np.zeros(31),'EEG | 31 recorded channels', [r['channel_name'] for r in geometry],True)
    for j,r in enumerate(example_records,1):
        g=[v for v in r['geometry_identity']['fnirs'] if v['component']=='HbO']
        own=counts[(counts.subject=='S06')&(counts.record==r['record'])].iloc[0]
        flags=np.array([int(i+1 in own.excluded_channels) for i in range(24)])
        draw(axes[j],g,flags,f'S06 Part2 / Probe{j} | {int((flags==0).sum())} usable, {int(flags.sum())} excluded',[str(i+1) for i in range(24)],True)
        for i,v in enumerate(g):sensor_rows.append(dict(probe=j,channel=i+1,excluded=int(flags[i]),x=v['x'],y=v['y'],z=v['z'],coordinate_status=v['coordinate_status']))
    handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle('MNE scalp layout | nose up; left hemisphere on left | fNIRS numbers = CH identifiers',fontsize=13)
    fig.savefig(figdir/'mne_channel_map.png',dpi=240);plt.close(fig);pd.DataFrame(sensor_rows).to_csv(out/'example_geometry.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained');freq=[]
    for j,r in enumerate(example_records):
        probe=j+1;rec=counts[counts.record.str.endswith('Probe'+str(probe))]
        rates=[]
        for ch in range(1,25):
            per=rec.groupby('subject').excluded_channels.apply(lambda v:np.mean([ch in x for x in v]))
            rates.append(float(per.mean()*100));freq.append(dict(probe=probe,channel=ch,exclusion_percent=rates[-1],subjects=len(per)))
        g=[v for v in r['geometry_identity']['fnirs'] if v['component']=='HbO']
        im=draw(axes[j],g,rates,f'Probe{probe} | exclusion frequency, 16 subjects',[str(i+1) for i in range(24)])
    fig.colorbar(im,ax=axes,label='Excluded records within subject, then equal subjects (%)',shrink=.75)
    fig.suptitle('MNE topomap of channel-exclusion frequency (not brain activation)',fontsize=13)
    fig.savefig(figdir/'mne_exclusion_frequency.png',dpi=240);plt.close(fig);pd.DataFrame(freq).to_csv(out/'channel_exclusion_frequency.csv',index=False)
    order=scales.sort_values('multiplier').subject.tolist();chosen=list(dict.fromkeys(['S06',order[0],order[-1]]))
    examples=[];sample_rows=[]
    fig,axes=plt.subplots(len(chosen),2,figsize=(15,4*len(chosen)),layout='constrained')
    fig2,axes2=plt.subplots(len(chosen),2,figsize=(15,4*len(chosen)),layout='constrained')
    for i,subject in enumerate(chosen):
        row=next(s for s in results if s['subject']==subject)
        with np.load(out/'subjects'/(subject+'.npz'),allow_pickle=False) as a:
            x=a['referenced'];old=a['old'];new=a['new'];t=a['time_s'];flags=a['excluded'];valid=a['valid']
            std=x[:-1].std(axis=0);strength=np.sqrt((std**2).mean(axis=1))
            def representative(flag):
                indices=np.flatnonzero(valid&(flags==flag));return int(indices[np.argsort(strength[indices],kind='stable')[(len(indices)-1)//2]])
            keep=11 if subject=='S06' and not flags[11] else representative(False);drop=representative(True)
            examples.append(dict(subject=subject,**row['waveform'],usable_channel=keep+1,excluded_channel=drop+1,multiplier=row['multiplier']))
            for j,c in enumerate(['HbO','HbR']):
                ax=axes[i,j];ax.plot(t[:-1],x[:-1,drop,j],color='#b83e43',ls='--',label='Excluded channel')
                ax.plot(t[:-1],x[:-1,keep,j],color='#157f78',lw=1.5,label='Usable channel')
                identity=row['waveform']['record_key'].rsplit('/',1)[-1]+' / '+row['waveform']['window_key']
                ax.set(title=f'{identity} | {c}\nUsable CH{keep+1}; excluded CH{drop+1}',xlabel='Window time (s)',ylabel='Cached relative Hb; first 5s reference')
                ax.grid(alpha=.2)
                ax=axes2[i,j];ax.plot(t[:-1],new[:-1,keep,j],color='#157f78',label='Usable-only calibration')
                ax.plot(t[:-1],old[:-1,keep,j],color='#777',ls='--',label='Past all-channel calibration')
                ax.set(title=f'{subject} / CH{keep+1} / {c} | multiplier={row["multiplier"]:.2f}',xlabel='Window time (s)',ylabel='Normalized Hb in model coordinates')
                ax.grid(alpha=.2)
                corr=float(np.corrcoef(old[:-1,keep,j],new[:-1,keep,j])[0,1]);np.testing.assert_allclose(corr,1,atol=1e-12)
                f,p0=periodogram(old[:-1,keep,j],fs=10);_,p1=periodogram(new[:-1,keep,j],fs=10)
                np.testing.assert_allclose(p1,p0*row['multiplier']**2,rtol=1e-8,atol=1e-15)
                sample_rows.append(dict(subject=subject,component=c,usable=keep+1,excluded=drop+1,
                    usable_sd_cache=float(std[keep,j]),excluded_sd_cache=float(std[drop,j]),
                    usable_sd_old=float(old[:-1,keep,j].std()),usable_sd_new=float(new[:-1,keep,j].std()),waveform_r=corr))
    fig.suptitle('Measured usable vs excluded Hb | each panel has its own y range',fontsize=14)
    fig2.suptitle('Same usable channel | different normalization; independent panel y ranges',fontsize=14)
    for f,ax in [(fig,axes[0,0]),(fig2,axes2[0,0])]:
        h,l=ax.get_legend_handles_labels();f.legend(h,l,loc='outside lower center',ncol=2)
    fig.savefig(figdir/'usable_excluded_waveforms.png',dpi=220);fig2.savefig(figdir/'normalization_waveforms.png',dpi=220);plt.close(fig);plt.close(fig2)
    write_json(out/'example_selection.json',dict(rule='S06 retained example plus minimum/maximum subject scale multiplier; first record, w00005 if present; excluded channel median pooled Hb SD; S06 CH12 retained; other usable channels median pooled Hb SD; descriptive only',rows=examples))
    pd.DataFrame(sample_rows).to_csv(out/'example_metrics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    axes[0].bar(scales.subject,scales.multiplier,color='#157f78');axes[0].set(ylabel='Usable-only factor / old factor',xlabel='Subject');axes[0].tick_params(axis='x',rotation=45)
    for i,r in enumerate(scales.itertuples()):axes[1].plot([0,1],[r.usable_sd_after_old,.025],'-o',alpha=.65)
    axes[1].set(xticks=[0,1],xticklabels=['Past: all channels','Usable channels only'],ylabel='Pooled usable Hb SD after normalization')
    fig.savefig(figdir/'normalization_summary.png',dpi=220);plt.close(fig)
    summary=dict(status='completed',subjects=len(results),probe_records=len(counts),eeg_sessions=len(sessions),eeg_channels=sorted(sessions.eeg_available.unique().tolist()),
        fnirs_locations_min=int(sessions.fnirs_locations.min()),fnirs_locations_max=int(sessions.fnirs_locations.max()),
        session_location_count_distribution={str(k):int(v) for k,v in sessions.fnirs_locations.value_counts().sort_index().items()},
        multiplier_min=float(scales.multiplier.min()),multiplier_median=float(scales.multiplier.median()),multiplier_max=float(scales.multiplier.max()),
        excluded_variance_fraction_median=float(scales.excluded_variance_fraction.median()),
        normalization='same common Hb formula, metadata exclusion before moments; EEG scale/PCA unchanged',new_SSM_fits=0,
        geometry='registered CED/graphical template; interpolated positions retained as uncertain display locations; not individual cortical localization',
        MNE_version=mne.__version__)
    if (out/'summary.json').exists():
        if json.loads((out/'summary.json').read_text())!=summary:raise ValueError('Report changed retained summary')
    else:write_json(out/'summary.json',summary)
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Visual可用通道及归一化对照</title><style>body{font-family:sans-serif;max-width:1450px;margin:auto;padding:28px;line-height:1.8}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccc;padding:6px}</style><h1>Visual：通道有效性、实际波形与共同Hb归一化对照</h1>
<p>本轮读取已完成公开缓存的16名被试，按每记录设备Exception Ch标记排除通道，并与原准备数据的有效性掩码取交集。只比较归一化；原有PCA、EEG尺度、预处理、参考窗口和SSM结果保留。可用表示在发布者排除标记与既有数值支持合同下可纳入，不表示已证明完全没有生理伪迹。</p>
<h2>通道数量与MNE位置图</h2><p>31个EEG通道包含FCz；DC09触发和参考不计入。两侧Probe共享同一EEG记录，不能重复计算成62通道。fNIRS名义48个测量位置，每位置有HbO/HbR两条序列。实际数量按每记录的排除标记变化，下表按同一EEG会话合并双侧Probe；EEG有效性检查覆盖全部参与校准的保存窗口。</p>'''
    body+=sessions.to_html(index=False)
    body+='''<img src="mne_channel_map.png"><p>左为31个EEG位置；中、右为S06 Part2的左右Probe，数字是CH编号。绿色圆点可用、红色叉排除，鼻尖向上。坐标复用已有Location.ced和设备图形模板；缺少直接EEG对应标签的fNIRS位置来自既有几何插值，仅作布局显示，不是逐被试皮层定位。fNIRS排除不意味着同位置EEG必须排除。</p><img src="mne_exclusion_frequency.png"><p>颜色表示每被试内记录排除率、再16人等权平均，色条0–100%。散点为真实通道位置；背景是MNE绘图插值，仅帮助读位置分布，不是脑激活或未测位置的有效性判定。不能用一份固定排除列表替代逐记录标记。</p>
<h2>同一实际记录：可用与排除通道</h2><img src="usable_excluded_waveforms.png"><p>各行是同一被试、同一记录、同一90秒窗口，左右分别HbO/HbR；绿实线为可用通道，红虚线为排除通道。两条线共用面板纵轴，面板间独立刻度。数据为原有全记录0.01–0.2 Hz、10 Hz实测缓存，沿用前5秒参考，没有新增滤波。数值保持既有Visual相对Hb坐标，不擅自标为µM。S06沿用CH12，其余例取可用/排除组幅度中位位置；另两名被试展示尺度变化最小/最大端点，不代表总体发生率。</p>
<h2>只改变参与归一化的通道集合</h2><p>共同因子仍为g=0.025/sqrt((Var(HbO)+Var(HbR))/2)。过去在每被试所有已准备通道/窗口上计算；本次用设备可用且原支持有效的通道计算。相同被试所有记录和HbO/HbR共用一个因子；保留原重叠窗口权重，不按当前窗口或单个通道自归一化。所有16人的旧因子均从保存输入精确重现，未将排除通道置零后加入分母。</p><img src="normalization_summary.png"><p>左为各被试新/旧因子，右为可用Hb集合在两种尺度下的共同SD。新值0.025由定义保证，不是新的拟合成绩；HbO和HbR各自的SD不强制相等。</p>'''
    body+=scales.round(6).to_html(index=False)
    body+='''<img src="normalization_waveforms.png"><p>每个面板是同一可用通道的旧/新归一化波形，灰虚线旧、绿实线新，坐标同尺度。形态相关性为1，所有样本乘同一正数；功率谱绝对幅度乘该倍数平方，归一化谱形、谱重心及HbO/HbR相关性不变。因子在被试间不同，跨被试相对幅度会改变。</p>
<h2>如何解释结果</h2><p>此对照测量了排除通道对幅度尺度的影响。保留同一个共同Hb公式即可恢复可用通道在模型坐标中的幅度，不需要逐窗口标准化，也不需要猜测物理单位。新坐标可能改变联合模型中的有效数据权重及动力学折中，需要重新拟合才能知道S/O改善多少。对旧S/O仅统一乘因子，其SD(S)/SD(X)和相关性不会改变；因此本报告不将重标度曲线称为新重建结果。</p>
<p>EEG通道、PCA及其因子在本对照中固定，以隔离Hb归一化变化。后续完整有效性迁移还需核查按可用位置拟合个体PCA及EEG尺度的影响，不能把本轮Hb尺度对照直接替代完整新缓存。</p>
<p><a href="summary.json">摘要</a> · <a href="record_counts.json">逐Probe通道名与掩码</a> · <a href="session_counts.csv">每EEG会话的双侧数量</a> · <a href="subject_normalization.csv">16人归一化数值</a> · <a href="example_selection.json">示例精确身份</a> · <a href="example_metrics.csv">示例幅度与形态核验</a> · <a href="channel_exclusion_frequency.csv">各通道排除率</a>。subjects目录保留逐被试矩、同源波形、旧/新数值及资源记录。</p></html>'''
    for filename in ['summary.json','record_counts.json','session_counts.csv','subject_normalization.csv','example_selection.json','example_metrics.csv','channel_exclusion_frequency.csv']:
        body=body.replace('href="'+filename+'"','href="../'+filename+'"')
    (figdir/'report.html').write_text(body,encoding='utf-8')


def visual_mask_main(parent,export,stage,workers):
    out=Path(export).resolve()
    if out.parent!=Path(parent['output']):raise ValueError('Use a fresh child of the Hb audit')
    cfg=dict(schema='visual_valid_normalization_v1',run=parent['run'],export=str(out),
        headers=str(Path(parent['output'])/'source_review_20261110_v1/visual_headers.json'),
        subjects=[s.split('/')[1] for s in parent['subjects'] if s.startswith('visual_cognitive_motivation/')],
        boundary='retained public prepared Visual inputs and metadata only; Hb normalization comparison; no raw signal reads or SSM refits',
        frozen='same model, EEG scale/PCA, first-5s reference, preprocessing and window weighting; no production overwrite')
    if stage=='visual-plan':
        out.mkdir(exist_ok=False);(out/'subjects').mkdir();write_json(out/'analysis_config.json',cfg);return
    if json.loads((out/'analysis_config.json').read_text())!=cfg:raise ValueError('Frozen analysis changed')
    if stage=='visual-report':visual_mask_report(cfg);return
    if not os.environ.get('SSM_SYSTEMD_UNIT'):raise ValueError('Use durable user systemd for full calibration replay')
    selected=['S01','S06'] if stage=='visual-pilot' else cfg['subjects'];done=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(visual_mask_subject,[(cfg,s) for s in selected]):
            done.append(r);print(json.dumps({k:r[k] for k in ['subject','multiplier','seconds','peak_rss_mib']}),flush=True)
            write_json(out/(stage+'_progress.json'),dict(completed=len(done),total=len(selected)))
    write_json(out/(stage+'_execution.json'),dict(seconds=time.monotonic()-start,workers=workers,subjects=len(done),
        peak_worker_mib=max(r['peak_rss_mib'] for r in done),supervisor=os.environ['SSM_SYSTEMD_UNIT']))
    if stage=='visual-run':visual_mask_report(cfg)


def refed_flag_worker(payload):
    cfg,subject=payload
    start=time.monotonic();out=Path(cfg['export'])/'subjects'/subject
    if out.with_suffix('.json').exists():return json.loads(out.with_suffix('.json').read_text())
    run=Path(cfg['run']);bits=cfg['flags'][subject];rows=[];failures=[];planned=0
    for path in sorted((run/'tasks/refed'/subject).glob('*/*.json')):
        d=json.loads(path.read_text());identity=d['identity'];planned+=1
        if d['schema']!='shared_driver_tak_subject_cache_v3' or identity['dataset']!='refed' or str(identity['subject'])!=subject:
            raise ValueError('Unexpected task identity')
        ch=int(identity['pair']['id'][2:]);base=dict(subject=subject,record=identity['record'],window=identity['window']['key'],
            channel=ch,flag=bits[ch-1],split=identity['split'],duration_s=d['duration_s'],task=str(path.relative_to(run)))
        if not d['qualified']:
            failures.append(dict(**base,status=d['status']));continue
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:
            use=np.all(a['fit_mask'] & a['prediction_valid'],axis=1)
            x=a['target'][use,1:];s=a['prediction'][use,1:];o=a['residual'][use,1:];scale=a['scale'][1:]
            np.testing.assert_allclose(x,s+o,rtol=1e-10,atol=1e-12)
            if not np.array_equal(np.flatnonzero(use),np.arange(len(x))):raise ValueError('Scored gap')
        for j,c in enumerate(['HbO','HbR']):
            w=next(w for w in d['fits']['tak_tau_beta']['waveform'] if w['component']==c)
            rec=dict(**base,component=c,scored_steps=len(x),correlation_sx=w['pearson_r'],s_sd_ratio=w['centered_amplitude_ratio'],
                local_rmse_ratio=w['local_centered_nrmse'],input_sd_model=float(x[:,j].std()*scale[j]),
                model_sd_model=float(s[:,j].std()*scale[j]),residual_sd_model=float(o[:,j].std()*scale[j]))
            defined=w['pearson_r'] is not None
            rec.update(defined=defined,poor_shape=float(w['pearson_r']<.5) if defined else np.nan,
                severe=float(w['centered_amplitude_ratio']>=2) if defined else np.nan)
            if defined:rec.update(residual_geometry(w['centered_amplitude_ratio'],w['pearson_r']))
            for branch,values in [('x',x),('s',s),('o',o)]:
                feature=signal_features(values[:,j]);rec.update({branch+'_'+k:v for k,v in feature.items() if k!='sd'})
            rec['input_hbo_hbr_r']=float(np.corrcoef(x.T)[0,1]) if min(x.std(0))>1e-12 else np.nan
            rec['s_hbo_hbr_r']=float(np.corrcoef(s.T)[0,1]) if min(s.std(0))>1e-12 else np.nan
            rows.append(rec)
    pd.DataFrame(rows).to_csv(out.with_suffix('.csv'),index=False)
    result=dict(subject=subject,planned=planned,qualified=planned-len(failures),failed=len(failures),failures=failures,
                seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(out.with_suffix('.json'),result);return result


def refed_flag_subjects(frame):
    """Robust descriptive values; rates retain all defined outcomes per record."""
    keys=['subject','component','flag','record']
    metrics=['input_sd_model','model_sd_model','residual_sd_model','s_sd_ratio','o_sd_ratio','correlation_sx','correlation_so',
             'local_rmse_ratio','x_centroid','s_centroid','x_mobility','s_mobility','x_fast_fraction','s_fast_fraction',
             'x_slow_fraction','x_kurtosis','input_hbo_hbr_r','s_hbo_hbr_r']
    numeric=frame.groupby(keys)[metrics].median()
    rates=frame.groupby(keys)[['severe','poor_shape']].mean()
    records=numeric.join(rates)
    return records.groupby(['subject','component','flag']).mean().reset_index(),records.reset_index()


def refed_flag_summary(cfg):
    out=Path(cfg['export']);frames=[];status=[]
    for sub in cfg['subjects']:
        frames.append(pd.read_csv(out/'subjects'/(sub+'.csv'),dtype={'subject':str}))
        status.append(json.loads((out/'subjects'/(sub+'.json')).read_text()))
    frame=pd.concat(frames,ignore_index=True)
    source=json.loads((Path(cfg['run'])/'summary.json').read_text())['datasets']['refed']['counts']
    counts={k:sum(s[k] for s in status) for k in ['planned','qualified','failed']}
    if counts!={k:source[k] for k in counts}:raise ValueError('REFED census mismatch')
    pd.DataFrame([f for s in status for f in s['failures']]).to_csv(out/'failures.csv',index=False)
    rng=np.random.default_rng(20261110);comparisons=[];denominators=[]
    for stratum,part in [('all',frame),('90s',frame[frame.duration_s==90]),('short',frame[frame.duration_s<90]),
                        ('exclude_CH10_12_15',frame[~frame.channel.isin([10,12,15])])]:
        sub,rec=refed_flag_subjects(part)
        sub.to_csv(out/('subject_'+stratum+'.csv'),index=False);rec.to_csv(out/('record_'+stratum+'.csv'),index=False)
        for (c,flag),g in part.groupby(['component','flag']):
            denominators.append(dict(stratum=stratum,component=c,flag=int(flag),qualified=len(g),defined=int(g.defined.sum()),
                undefined=int((~g.defined).sum()),subjects=g.subject.nunique(),subject_channels=len(g[['subject','channel']].drop_duplicates())))
        for c,g in sub.groupby('component'):
            for metric in [x for x in g.columns if x not in ['subject','component','flag']]:
                paired=g.pivot(index='subject',columns='flag',values=metric).dropna()
                if not len(paired):continue
                diff=(paired[1]-paired[0]).to_numpy();ci=np.quantile(diff[rng.integers(0,len(diff),(2000,len(diff)))].mean(1),[.025,.975])
                comparisons.append(dict(stratum=stratum,component=c,metric=metric,n=len(paired),flag0=paired[0].mean(),flag1=paired[1].mean(),
                    difference_1_minus_0=diff.mean(),ci_low=ci[0],ci_high=ci[1],subjects_1_greater=int((diff>0).sum())))
    pd.DataFrame(comparisons).to_csv(out/'group_comparisons.csv',index=False)
    pd.DataFrame(denominators).to_csv(out/'denominators.csv',index=False)
    site=frame.groupby(['subject','channel','component'])[['input_sd_model','s_sd_ratio','correlation_sx','severe']].median().reset_index()
    site['flag']=[cfg['flags'][r.subject][r.channel-1] for r in site.itertuples()]
    site.to_csv(out/'subject_channel_values.csv',index=False)
    channel_rows=[]
    for (ch,c),g in site.groupby(['channel','component']):
        n=g.groupby('flag').size()
        if len(n)!=2 or min(n)<3:continue
        for metric in ['input_sd_model','s_sd_ratio','correlation_sx']:
            channel_rows.append(dict(channel=ch,component=c,metric=metric,n0=int(n[0]),n1=int(n[1]),
                flag0=g[g.flag==0][metric].median(),flag1=g[g.flag==1][metric].median()))
    pd.DataFrame(channel_rows).to_csv(out/'same_channel_comparison.csv',index=False)
    bits=pd.DataFrame.from_dict(cfg['flags'],orient='index',columns=range(1,52)).sort_index(key=lambda x:x.astype(int))
    bits.index.name='subject';bits.to_csv(out/'flag_matrix.csv')
    pd.DataFrame(dict(channel=range(1,52),flag0=(bits==0).sum().to_numpy(),flag1=(bits==1).sum().to_numpy())).to_csv(out/'channel_frequency.csv',index=False)
    pd.DataFrame(dict(subject=bits.index,flag0=(bits==0).sum(axis=1),flag1=(bits==1).sum(axis=1))).to_csv(out/'subject_frequency.csv',index=False)
    write_json(out/'summary.json',dict(status='completed',counts=counts,subjects=len(status),records=frame[['subject','record']].drop_duplicates().shape[0],
        subject_channels=bits.size,flag0=int((bits==0).to_numpy().sum()),flag1=int((bits==1).to_numpy().sum()),
        label_level='published subject x channel; no video, session, time or Hb-species key',
        inference='paired subject bootstrap: mean(flag1-flag0), 2000 resamples, pointwise descriptive 95% intervals, no multiplicity adjustment',
        aggregation='numeric metrics: median channels/windows within video, mean videos within subject, equal subjects; rates: mean defined outcomes within video then same hierarchy',
        units='saved v3 common subject Hb scale; original reference; no new normalization',new_fits=0,raw_signal_reads=0))


def refed_flag_report(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    out=Path(cfg['export']);run=Path(cfg['run'])
    report=out/'report_v2';report.mkdir(exist_ok=False)
    bits=pd.read_csv(out/'flag_matrix.csv',index_col=0)
    freq=pd.read_csv(out/'channel_frequency.csv');subfreq=pd.read_csv(out/'subject_frequency.csv')
    sub=pd.read_csv(out/'subject_all.csv',dtype={'subject':str})
    comp=pd.read_csv(out/'group_comparisons.csv');summary=json.loads((out/'summary.json').read_text())
    colors=['#d98235','#247b9d']
    fig,axes=plt.subplots(2,1,figsize=(17,12),height_ratios=[3,1],layout='constrained')
    im=axes[0].imshow(bits.to_numpy(),aspect='auto',cmap=ListedColormap(colors),vmin=0,vmax=1,interpolation='nearest')
    axes[0].set(xticks=range(51),xticklabels=range(1,52),yticks=range(32),yticklabels=bits.index,xlabel='Channel',ylabel='Subject',title='Published REFED flags: one row per subject, shared across videos')
    axes[0].tick_params(axis='x',labelsize=8)
    cb=fig.colorbar(im,ax=axes[0],ticks=[0,1],fraction=.02);cb.ax.set_yticklabels(['flag 0','flag 1'])
    axes[1].bar(freq.channel,freq.flag0,color=colors[0],label='flag 0');axes[1].bar(freq.channel,freq.flag1,bottom=freq.flag0,color=colors[1],label='flag 1')
    axes[1].set(xticks=range(1,52),ylim=(0,34),xlabel='Channel',ylabel='Number of subjects',title='Each channel denominator = 32 subjects');axes[1].tick_params(axis='x',labelsize=8);axes[1].legend()
    fig.savefig(report/'flag_frequency.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(17,9),layout='constrained')
    metrics=[('input_sd_model','Measured SD / frozen model coordinate'),('correlation_sx','Correlation S vs X'),('severe','Severe amplification (%)'),('x_centroid','Input spectral centroid (Hz)')]
    for i,c in enumerate(['HbO','HbR']):
        for ax,(metric,label) in zip(axes[i],metrics):
            g=sub[sub.component==c].pivot(index='subject',columns='flag',values=metric).dropna();values=g.to_numpy()*(100 if metric=='severe' else 1)
            for row in values:ax.plot([0,1],row,color='gray',alpha=.25,lw=.6)
            for f in [0,1]:ax.scatter(np.full(len(values),f),values[:,f],color=colors[f],s=23,alpha=.7)
            ax.set(xticks=[0,1],xticklabels=['flag 0','flag 1'],ylabel=label,title=c+' | paired subjects')
            if metric=='input_sd_model':ax.set_yscale('log')
            ax.grid(alpha=.2)
    fig.savefig(report/'group_statistics.png',dpi=220);plt.close(fig)
    # Input-only illustration: three subjects at quartiles of the between-flag amplitude ratio.
    ss=pd.read_csv(out/'subject_90s.csv',dtype={'subject':str})
    amp=ss.pivot(index=['subject','component'],columns='flag',values='input_sd_model')
    ratios=np.log(amp[0]/amp[1]).groupby('subject').mean().sort_values(kind='stable')
    chosen=[str(ratios.index[int(round((len(ratios)-1)*q))]) for q in [.25,.5,.75]]
    examples=[];waves=[]
    for subject in chosen:
        data=pd.read_csv(out/'subjects'/(subject+'.csv'),dtype={'subject':str})
        data=data[(data.duration_s==90)&data.defined]
        wide=data.pivot(index=['record','window','channel','flag','task'],columns='component',values='input_sd_model')
        wide=wide[(wide>0).all(axis=1)].dropna();logs=np.log(wide)
        centre=logs.groupby(level='flag').median()
        score=pd.Series([float(((row-centre.loc[key[3]])**2).sum()) for key,row in logs.iterrows()],index=logs.index,name='distance').reset_index()
        winners=score.sort_values(['distance','record','window','channel'],kind='stable').groupby(['record','window','flag'],sort=False).head(1)
        complete=winners.groupby(['record','window']).filter(lambda x:len(x)==2)
        pairscore=complete.groupby(['record','window']).distance.sum().sort_values(kind='stable')
        record,window=pairscore.index[0];pair=complete[(complete.record==record)&(complete.window==window)].sort_values('flag')
        selected=dict(subject=subject,record=record,window=window,input_group_ratio=float(np.exp(ratios.loc[subject])),channels=[],candidate_pairs=len(pairscore))
        signal={}
        for row in pair.itertuples():
            path=run/row.task;meta=json.loads(path.read_text())
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:
                use=np.all(a['fit_mask'] & a['prediction_valid'],axis=1);scale=a['scale'][1:]
                signal[row.flag]=dict(t=a['time_s'][use],X=a['target'][use,1:]*scale,S=a['prediction'][use,1:]*scale,O=a['residual'][use,1:]*scale)
                np.testing.assert_allclose(signal[row.flag]['X'],signal[row.flag]['S']+signal[row.flag]['O'],rtol=1e-10,atol=1e-12)
            selected['channels'].append(dict(flag=int(row.flag),channel=int(row.channel),task=str(path),window_metadata=meta['identity']['window']))
        for item in selected['channels']:
            aa=signal[item['flag']];item['metrics']={}
            for j,c in enumerate(['HbO','HbR']):
                xx,ss,oo=(aa[k][:,j] for k in ['X','S','O'])
                item['metrics'][c]=dict(input_sd=float(xx.std()),s_sd_ratio=float(ss.std()/xx.std()),
                    o_sd_ratio=float(oo.std()/xx.std()),correlation_sx=float(np.corrcoef(xx,ss)[0,1]))
        examples.append(selected);waves.append(signal)
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for flag in [0,1]:
            a=signal[flag];ch=selected['channels'][flag]['channel']
            for j,c in enumerate(['HbO','HbR']):
                ax=axes[flag,j]
                for branch,color,ls in [('X','#242b31','-'),('S','#287abd','-'),('O','#dd7139','--')]:ax.plot(a['t'],a[branch][:,j],color=color,ls=ls,label=branch,lw=1.2)
                aa=a['S'][:,j].std()/a['X'][:,j].std();rr=np.corrcoef(a['X'][:,j],a['S'][:,j])[0,1]
                ax.set(xlabel='Time in window (s)',ylabel='Frozen common-Hb model scale',xlim=(0,90),title=f'flag {flag} / CH{ch} / {c} | SD(S)/SD(X)={aa:.2f}; r={rr:.2f}')
                ax.axvspan(0,5,color='gray',alpha=.12);ax.grid(alpha=.2)
        fig.suptitle(f'REFED {subject} / {record} / {window} | same input interval; independent y limits')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
        fig.savefig(report/f'waveform_subject_{subject}.png',dpi=220);plt.close(fig)
        np.savez_compressed(report/f'waveform_subject_{subject}.npz',**{f'flag{f}_{k}':v for f,a in signal.items() for k,v in a.items()})
    fig,axes=plt.subplots(3,2,figsize=(14,11),layout='constrained')
    for i,(ex,signal) in enumerate(zip(examples,waves)):
        for j,c in enumerate(['HbO','HbR']):
            ax=axes[i,j]
            for f in [0,1]:ax.plot(signal[f]['t'],signal[f]['X'][:,j],color=colors[f],ls=['-','--'][f],label=f'flag {f} / CH{ex["channels"][f]["channel"]}')
            ax.set(xlabel='Time in window (s)',ylabel='Frozen common-Hb model scale',title=f'{ex["subject"]} / {ex["record"]} / {ex["window"]} / {c}',xlim=(0,90));ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Measured X only | same y scale for both flags inside each panel')
    fig.savefig(report/'measured_pairs.png',dpi=220);plt.close(fig)
    write_json(report/'example_selection.json',dict(rule='subjects at 25/50/75% ranks of log input-SD flag0/flag1 ratio; within subject, same 90s window channels nearest group median log HbO/HbR input SD; no fit outcome selection beyond numerical/defined support',examples=examples))
    display=comp[(comp.stratum=='all')&comp.metric.isin(['input_sd_model','s_sd_ratio','o_sd_ratio','correlation_sx','correlation_so','severe','poor_shape'])].copy()
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>REFED 0/1通道标记分析</title><style>body{font-family:sans-serif;max-width:1450px;margin:auto;padding:30px;line-height:1.85;color:#243542}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border:1px solid #ccd}code{background:#eee}</style><h1>REFED：0/1通道标记、统计与同窗波形对比</h1>
<p>发布者fNIRS_reservations.csv只有sub_id与ch-1…ch-51：32名被试各一行，没有视频、时间、Hb种类或重复会话键。因此它是已发布的被试×通道标记，同一人的15段视频共用一行；不能从该表判断设备是否曾逐记录检查并更新。0/1有效性方向仍未得到来源确认，本报告不替任一组命名为正常或坏通道。</p>
<h2>标记分布</h2><p>共1632个被试通道，0为223个（13.66%），1为1409个（86.34%）。每名被试0通道数为2–16，中位数6。CH10全32人为0，CH12与CH15各31人；三者占全部0标记的94/223。其余高频0通道包括CH37、35、25、30、43。完整51通道频次如下；每格代表一个被试通道，不把15个视频重复算作15个标记。</p><img src="flag_frequency.png">'''
    html+=freq.sort_values(['flag0','channel'],ascending=[False,True]).to_html(index=False)
    html+='''<h2>全部记录的分组重建统计</h2><p>覆盖32人、480个视频记录、800个拟合窗口、40,800个窗口×通道任务；40,791个数值合格，9个失败保留于failures.csv。31个合格任务的HbO/HbR相关性不可定义，共62个分量，不以零补齐。归一化复用v3被试共同Hb因子、前5秒参考及原0.01–0.2 Hz/10 Hz输入；S为保存模型预测，O=X−S；没有新增归一化或重新拟合。</p>
<p>数值指标先在每名被试的每个视频内，对窗口/通道取中位数，再视频均值、被试等权；事件率在视频内取有效结果均值，再按相同层级汇总。严重放大为SD(S)/SD(X)≥2；poor_shape为corr(S,X)&lt;0.5；波动幅度为去均值SD。input_sd_model是共同模型坐标的实测SD，不能标成µM。置信区间对同一被试的flag1−flag0差值重采样2000次；独立单位为被试，逐项95%描述区间，未校正多重比较。均值不等于把所有窗口当独立重复。</p><img src="group_statistics.png">'''
    html+=display.round(5).to_html(index=False)
    html+='''<p>上图每条灰线连接同一被试的两组值；幅度轴为对数，其余面板线性；橙色0、蓝色1。区间和完整指标见group_comparisons.csv。两组通道位置和数量不同，因此这是一种标签关联，不是标记对信号的因果效应。</p>
<h2>时间支持、频谱及通道位置敏感性</h2><p>所有合格任务均计算HbO/HbR的X/S/O频谱；没有仅挑选首中末位置。常数去趋势、周期Hann窗、无补零/新增滤波；谱重心为所有正频率功率加权平均，fast_fraction是&gt;0.2 Hz功率占所有正频率功率比例；slow_fraction为少于3周期的低频功率比例，仅描述。mobility为离散导数SD/信号SD，单位s⁻¹。频谱主比较使用完整90秒任务（899个评分点）；短REFED单列，以避免分辨率差异。</p>'''
    html+=comp[(comp.stratum=='90s')&comp.metric.isin(['x_centroid','s_centroid','x_fast_fraction','s_fast_fraction','x_mobility','input_hbo_hbr_r'])].round(5).to_html(index=False)
    html+='''<p>另外排除CH10、12、15后重新计算分组差异；该敏感性中只纳入仍同时有0/1组的被试，实际配对数见n列。same_channel_comparison.csv在同一物理通道的不同被试间比较，要求两组各至少3名被试，作为位置混杂的补充描述，不称为被试内配对。</p>'''
    html+=comp[(comp.stratum=='exclude_CH10_12_15')&comp.metric.isin(['input_sd_model','correlation_sx','severe'])].round(5).to_html(index=False)
    html+='''<h2>相同被试、视频和时间窗的真实波形</h2><p>以完整90秒输入幅度为依据，选择两组幅度比排序约25%、50%、75%处的三名被试；在每人内部选接近组内典型输入幅度的同窗通道对。不是按S/O误差选最好或最坏结果。下图只画实测X；每面板0/1组共用纵轴，可直接比较同一共同归一化下的幅度，面板间独立刻度。灰参考段和时间偏移身份保留在选择文件中。</p><img src="measured_pairs.png">'''
    for ex in examples:
        subject=ex['subject'];html+=f'<h3>被试{subject}，{ex["record"]}，{ex["window"]}</h3><img src="waveform_subject_{subject}.png"><p>上排flag0 CH{ex["channels"][0]["channel"]}，下排flag1 CH{ex["channels"][1]["channel"]}；左HbO、右HbR。X黑实线、S蓝实线、O橙虚线，灰色0–5秒为原参考段。所有曲线沿用保存的共同尺度；这里独立缩放面板纵轴以读清弱波形，不是重新归一化。标题给出幅度比与形态相关性，避免仅凭视觉重叠判断。</p>'
        html+=pd.DataFrame([dict(flag=v['flag'],channel=v['channel'],component=c,**m) for v in ex['channels'] for c,m in v['metrics'].items()]).round(5).to_html(index=False)
    html+='''<h2>本组结果的解释</h2><p>在32名被试中，1组的被试汇总实测幅度均小于0组，HbO/HbR的S与X形态相关性也均更低。HbO严重放大率为0组1.13%、1组14.56%；HbR为0组0.51%、1组4.68%。HbO的1−0组差为13.44个百分点，95%被试配对bootstrap区间9.27–17.77个百分点。这个差异同时出现在15段视频中，逐视频两组HbO发生率分别约0.4–2.1%与9.9–18.9%，不是单一视频造成。</p>
<p>完整90秒记录中，0组实测HbO/HbR谱重心分别0.0623/0.0640 Hz，1组为0.0506/0.0522 Hz：1组不仅幅度小，变化也较慢。模型S的HbO谱重心却在1组升至0.0763 Hz，0组为0.0503 Hz，反映1组的重建同时改变频谱形态。输入HbO/HbR相关性0组约−0.858，1组约−0.611；这是统计结构差异，不能单凭“更反相关”判定测量有效。</p>
<p>排除最集中为0的CH10、12、15后，剩30名被试仍能两组配对：HbO严重放大率为2.45%对15.43%，差12.98个百分点（95%区间8.33–17.62）。在0/1组各至少有3名被试的20个相同通道中，18个通道的0组输入幅度更大；19个通道的0组HbO相关性更高。因此，通道构成影响比较，但上述差异并非只由那三个通道产生。</p>
<p>波形例显示两组有不同的实际幅度和峰谷结构。0组在本模型下更容易被S跟随，并不证明0就是有效通道；反之，1组的小幅度和模型失配也不证明1就是坏通道。当前证据支持在确认发布者0/1语义后按其规则处理，并用同一组视频与时间窗检验有效性/归一化修正后的变化；不依据拟合优度自行反转标记，也不以本次只读统计声称修复后的S/O已经改善。</p>'''
    html+='''<h2>证据入口</h2><p><a href="../summary.json">完成与分母</a> · <a href="../group_comparisons.csv">配对统计与区间</a> · <a href="../denominators.csv">组别/时长分母及未定义值</a> · <a href="../failures.csv">数值失败</a> · <a href="../subject_all.csv">逐被试汇总</a> · <a href="../record_all.csv">逐视频汇总</a> · <a href="../channel_frequency.csv">通道频次</a> · <a href="../same_channel_comparison.csv">同通道敏感性</a> · <a href="example_selection.json">同窗波形身份与选择规则</a>。每名被试的完整窗口指标及波形同源数组均保留；不改变原生产掩码、校准与拟合。</p></html>'''
    (report/'report.html').write_text(html,encoding='utf-8')


def refed_flag_main(parent,export,stage,workers):
    run=Path(parent['run']);out=Path(export).resolve()
    if out.parent!=Path(parent['output']):raise ValueError('Use a fresh child of the Hb audit')
    reference=run.parents[4]/'data/REFED-dataset/fNIRS_reservations.csv'
    bits=pd.read_csv(reference,dtype={'sub_id':str}).set_index('sub_id')
    if bits.shape!=(32,51) or not bits.index.is_unique or not np.isin(bits,[0,1]).all():raise ValueError('Invalid flags')
    cfg=dict(schema='refed_channel_flags_v1',run=str(run),export=str(out),source_table=str(reference),
        subjects=sorted(bits.index,key=int),flags={s:bits.loc[s].astype(int).tolist() for s in bits.index},
        boundary='completed registered public REFED cache and publisher metadata only; no raw/protected reads, labels as targets, changed scales or fits',
        support='full contiguous scored support, last bin excluded; 90s and shorter records separately; 10 Hz Hann constant-detrended PSD without new filter',
        endpoints='flag frequency; paired subject amplitude/morphology/cancellation; channel identity sensitivity; spectrum; same-window 0/1 examples')
    if stage=='plan':
        out.mkdir(exist_ok=False);(out/'subjects').mkdir();write_json(out/'analysis_config.json',cfg);return
    if json.loads((out/'analysis_config.json').read_text())!=cfg:raise ValueError('Frozen analysis changed')
    if stage=='report':refed_flag_report(cfg);return
    if not os.environ.get('SSM_SYSTEMD_UNIT'):raise ValueError('Durable supervisor required')
    subjects=cfg['subjects'] if stage=='run' else ['1','7']
    start=time.monotonic();done=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(refed_flag_worker,[(cfg,s) for s in subjects]):
            done.append(r);print(json.dumps({k:r[k] for k in ['subject','planned','qualified','failed','seconds']}),flush=True)
            write_json(out/(stage+'_progress.json'),dict(completed=len(done),total=len(subjects)))
    write_json(out/(stage+'_execution.json'),dict(seconds=time.monotonic()-start,workers=workers,subjects=len(done),
        planned=sum(d['planned'] for d in done),peak_worker_mib=max(d['peak_rss_mib'] for d in done),supervisor=os.environ['SSM_SYSTEMD_UNIT']))
    if stage=='run':refed_flag_summary(cfg)


def refed_author_probe(export):
    """Execute only reviewed author function definitions on temporary synthetic MATs."""
    import ast
    import contextlib
    import gc
    import io
    import tempfile
    import mne
    from scipy.io import loadmat,savemat
    from torch.utils.data import Dataset
    out=Path(export)
    source=json.loads((out/'probe_sources.json').read_text())
    files={r['path']:r['code'] for r in source['files']}
    opened=[]
    def read_mat(path,*args,**kwargs):
        opened.append(str(path));return loadmat(path,*args,**kwargs)
    def definitions(path,names,extra=None):
        tree=ast.parse(files[path]);nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        if {n.name for n in nodes}!=set(names):raise ValueError('Author definitions changed')
        ns=dict(np=np,os=os,gc=gc,loadmat=read_mat,savemat=savemat,mne=mne,Dataset=Dataset)
        ns.update(extra or {});exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns);return ns
    raw=definitions('load_REFED.py',['load_data'])
    pre=definitions('1_process_sample.ipynb',['preprocess_fNIRS_sub'])
    lower=definitions('7_supervised_learning/utils/load_REFED.py',['load_data'])
    wrapper=definitions('7_supervised_learning/utils/utils_data.py',['load_data','process_data','EEG_fNIRS_Dataset'],
                        dict(load_REFED_data=lower['load_data']))
    checks={};logs=io.StringIO()
    with tempfile.TemporaryDirectory(prefix='refed_author_probe_') as temp,contextlib.redirect_stdout(logs):
        root=Path(temp);native=root/'raw'/'1';native.mkdir(parents=True);processed=root/'processed'/'1';processed.mkdir(parents=True)
        t=np.arange(1000)/(1000/21)
        wave=np.sin(2*np.pi*.13*t)+.25*np.cos(2*np.pi*.7*t)
        values=np.arange(1,7)[:,None,None]*np.arange(1,52)[None,:,None]*wave[None,None,:]
        values+=np.arange(6)[:,None,None]*100
        videos={f'video_{i}':values+i for i in range(1,16)}
        savemat(native/'fNIRS_videos.mat',videos)
        # Contrary bits are intentionally present; a reader must actually consume them to change selection.
        (root/'raw/fNIRS_reservations.csv').write_text('sub_id,'+','.join(f'ch-{c}' for c in range(1,52))+'\n1,'+','.join('0' if c%2 else '1' for c in range(1,52))+'\n')
        loaded=raw['load_data'](str(native.parent),sub_list=['1'],modality=['fNIRS'])['1']['fNIRS']['v1']
        np.testing.assert_array_equal(loaded,videos['video_1']);checks['raw_loader_exact_all_six_types_51_channels']=True
        # Preprocessing is local to the video, linear in channel amplitudes, with no amplitude standardization.
        savemat(native/'fNIRS_videos.mat',{'video_1':values})
        para=dict(fNIRS_fs=1000/21,fNIRS_fs_down=25,fNIRS_filt_band=[.01,10],window=1)
        pre['preprocess_fNIRS_sub'](str(native),para,str(processed))
        first=loadmat(processed/'fNIRS_videos.mat')['video_1'];assert first.shape==(21,6,51,25)
        savemat(native/'fNIRS_videos.mat',{'video_1':values*1000})
        pre['preprocess_fNIRS_sub'](str(native),para,str(processed))
        scaled=loadmat(processed/'fNIRS_videos.mat')['video_1']
        rel=float(np.max(abs(scaled-first*1000))/np.max(abs(first*1000)))
        np.testing.assert_allclose(scaled,first*1000,rtol=1e-7,atol=1e-5)
        checks.update(preprocess_shape=list(first.shape),preprocess_scale_linearity_relative_error=rel)
        # Deliberate unequal channel amplitudes survive the author function.
        ratio=float(np.std(first[:,:,50,:])/np.std(first[:,:,0,:]))
        np.testing.assert_allclose(ratio,51,rtol=1e-8);checks['preserved_channel_51_over_1_SD']=ratio
        processed_videos={f'video_{i}':first+i for i in range(1,16)}
        savemat(processed/'fNIRS_videos.mat',processed_videos)
        savemat(processed/'fNIRS_baselines.mat',{f'video_{i}':np.full_like(first,-9999) for i in range(1,16)})
        data=wrapper['load_data']('1',str(processed.parent),'unused',modality=['fNIRS'],info=['video','baseline'])
        np.testing.assert_array_equal(data['fNIRS'][0],processed_videos['video_1'][:,2])
        np.testing.assert_array_equal(data['fNIRS_baseline'][0],data['fNIRS'][0])
        checks['training_HbT_selection_all_51']=True;checks['baseline_key_reloads_videos_not_baselines']=True
        before=data['fNIRS'][0].copy();data=wrapper['process_data'](data)
        np.testing.assert_array_equal(data['fNIRS'][0][:,0],before)
        dataset=wrapper['EEG_fNIRS_Dataset'](np.zeros((len(before),1,64,200)),data['fNIRS'][0],np.zeros(len(before)),'3c')
        np.testing.assert_array_equal(dataset.fnirs,before[:,None].astype(np.float32))
        checks['process_data_only_adds_axis']=True;checks['dataset_only_casts_fnirs_to_float32']=True
        checks['actual_mat_reads']=sorted(set(Path(p).name for p in opened))
        assert 'fNIRS_baselines.mat' not in checks['actual_mat_reads']
    write_json(out/'synthetic_verification.json',dict(status='passed',author_commit=source['commit'],checks=checks,
        scope='reviewed function/class AST definitions only; no notebook top-level, training, network, real records or labels',measured_data_reads=0))
    (out/'synthetic_probe.log').write_text(logs.getvalue(),encoding='utf-8')


def holm_adjust(values):
    p=np.asarray(values,float);order=np.argsort(p);adjusted=np.empty(len(p))
    adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return adjusted


def paired_signflip(delta, signs):
    d=np.asarray(delta,float)
    if d.ndim!=1 or not np.isfinite(d).all() or signs.shape[1]!=len(d):raise ValueError('Finite paired units required')
    observed=abs(d.mean());null=abs(signs@d/len(d))
    return float((1+np.count_nonzero(null>=observed-1e-15))/(len(null)+1))


def refed_state_worker(payload):
    cfg,subject=payload;start=time.monotonic();out=Path(cfg['export'])/'subjects'/subject
    if out.with_suffix('.json').exists():return json.loads(out.with_suffix('.json').read_text())
    run=Path(cfg['run']);rows=[];failed=0;pv_error=0.
    for path in sorted((run/'tasks/refed'/subject).glob('*/*.json')):
        d=json.loads(path.read_text());ident=d['identity'];ch=int(ident['pair']['id'][2:])
        if d['schema']!='shared_driver_tak_subject_cache_v3' or ident['dataset']!='refed' or str(ident['subject'])!=subject:
            raise ValueError('Wrong retained identity')
        row=dict(subject=subject,record=ident['record'],window=ident['window']['key'],channel=ch,
            flag=cfg['flags'][subject][ch-1],duration_s=d['duration_s'],qualified=d['qualified'],status=d['status'],task=str(path.relative_to(run)))
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:
            use=np.all(a['fit_mask'] & a['observation_valid'],axis=1)
            x=a['target'][use,1:];scale=a['scale'][1:];sd=a['sd_subject'][1:]
            if not np.isfinite(x).all() or not np.array_equal(np.flatnonzero(use),np.arange(len(x))):raise ValueError('Invalid measured support')
            for j,c in enumerate(['HbO','HbR']):
                y=x[:,j];z=y*scale[j];p05,p95=np.quantile(z,[.05,.95])
                row.update({c+'_sd':float(z.std()),c+'_sd_relative_export':float(y.std()),c+'_sd_subject_ratio':float(y.std()/sd[j]),
                    c+'_min':float(z.min()),c+'_max':float(z.max()),c+'_ptp':float(np.ptp(z)),c+'_p95_minus_p05':float(p95-p05)})
            if d['qualified']:
                if not np.all(a['prediction_valid'][use]):raise ValueError('Qualified state support differs')
                st=a['states'][use];np.testing.assert_allclose(st[:,0],a['driver'][use],atol=0,rtol=0)
                pv_error=max(pv_error,float(abs(st[:,3]-st[:,4]).max()))
                np.testing.assert_allclose(st[:,3],st[:,4],rtol=0,atol=0)
                fit=d['fits']['tak_tau_beta'];row.update(tau=fit['tau'],beta=fit['beta'])
                for j,name,rest in [(0,'r',0.),(1,'s',0.),(2,'f',1.),(3,'v',1.),(5,'q',1.)]:
                    v=st[:,j];p05,p95=np.quantile(v,[.05,.95])
                    row.update({name+'_mean':float(v.mean()),name+'_sd':float(v.std()),name+'_rms_rest':float(np.sqrt(np.mean((v-rest)**2))),
                        name+'_min':float(v.min()),name+'_max':float(v.max()),name+'_p05':float(p05),name+'_p95':float(p95)})
            else:failed+=1
        rows.append(row)
    frame=pd.DataFrame(rows)
    for c in ['HbO','HbR']:
        ref=frame[frame.flag==0].groupby(['record','window'])[c+'_sd'].median()
        den=np.array([ref.loc[(r.record,r.window)] for r in frame.itertuples()])
        frame[c+'_relative_flag0_window']=np.divide(frame[c+'_sd'],den,out=np.full(len(frame),np.nan),where=den>0)
    frame.to_csv(out.with_suffix('.csv'),index=False)
    result=dict(subject=subject,planned=len(rows),qualified=len(rows)-failed,failed=failed,p_equals_v_max_error=pv_error,
        seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(out.with_suffix('.json'),result);return result


def refed_state_summary(cfg):
    out=Path(cfg['export']);frames=[pd.read_csv(out/'subjects'/(s+'.csv'),dtype={'subject':str}) for s in cfg['subjects']]
    frame=pd.concat(frames,ignore_index=True);fit=frame[frame.qualified].copy()
    counts=dict(planned=len(frame),qualified=len(fit),failed=len(frame)-len(fit))
    source=json.loads((Path(cfg['run'])/'summary.json').read_text())['datasets']['refed']['counts']
    if counts!={k:source[k] for k in counts}:raise ValueError('Source census changed')
    state_names=['r','s','f','v','q'];primary=['tau','beta']+[s+'_'+m for s in state_names for m in ['mean','sd']]
    metrics=primary+[s+'_rms_rest' for s in state_names]
    rng=np.random.default_rng(20261110);signs=rng.choice([-1.,1.],size=(20000,len(cfg['subjects'])))
    boot=rng.integers(0,len(cfg['subjects']),size=(5000,len(cfg['subjects'])))
    comparisons=[]
    for stratum,g in [('all',fit),('90s',fit[fit.duration_s==90]),('exclude_CH10_12_15',fit[~fit.channel.isin([10,12,15])])]:
        sub=g.groupby(['subject','flag','record'])[metrics].median().groupby(['subject','flag']).mean().reset_index()
        sub.to_csv(out/('subject_states_'+stratum+'.csv'),index=False)
        for metric in metrics:
            a=sub.pivot(index='subject',columns='flag',values=metric).dropna();delta=(a[1]-a[0]).to_numpy()
            ss=signs[:,:len(a)];bb=boot if len(a)==len(cfg['subjects']) else rng.integers(0,len(a),size=(5000,len(a)))
            ci=np.quantile(delta[bb].mean(1),[.025,.975])
            comparisons.append(dict(stratum=stratum,metric=metric,n=len(a),flag0=a[0].mean(),flag1=a[1].mean(),difference_1_minus_0=delta.mean(),
                ci_low=ci[0],ci_high=ci[1],p=paired_signflip(delta,ss),primary=metric in primary,
                paired_effect_dz=delta.mean()/delta.std(ddof=1) if delta.std(ddof=1)>0 else np.nan))
    # Distribution-level comparison: CDFs average within video, then subject.
    cdf_results=[]
    for metric in ['tau','beta']:
        grid=np.unique(np.quantile(fit[metric],np.linspace(0,1,257)));cdf=[];rows=[]
        for subject in cfg['subjects']:
            block=fit[fit.subject==subject];pair=[]
            for flag in [0,1]:
                arrays=[]
                for _,g in block[block.flag==flag].groupby('record'):
                    v=np.sort(g[metric]);arrays.append(np.searchsorted(v,grid,side='right')/len(v))
                value=np.mean(arrays,axis=0);pair.append(value)
                rows.extend(dict(subject=subject,flag=flag,value=float(x),cdf=float(y)) for x,y in zip(grid,value))
            cdf.append(pair[1]-pair[0])
        cdf=np.asarray(cdf);observed=float(abs(cdf.mean(0)).max());null=np.max(abs(signs@cdf/len(cdf)),axis=1)
        cdf_results.append(dict(metric=metric+'_CDF',statistic=observed,p=float((1+(null>=observed-1e-15).sum())/(len(null)+1))))
        pd.DataFrame(rows).to_csv(out/(metric+'_cdf.csv'),index=False)
    comparisons=pd.DataFrame(comparisons)
    family=comparisons[(comparisons.stratum=='all')&comparisons.primary]
    corrected=holm_adjust(list(family.p)+[r['p'] for r in cdf_results])
    comparisons['p_holm']=np.nan;comparisons.loc[family.index,'p_holm']=corrected[:len(family)]
    for r,p in zip(cdf_results,corrected[len(family):]):r['p_holm']=float(p)
    comparisons.to_csv(out/'parameter_state_tests.csv',index=False);write_json(out/'parameter_distribution_tests.json',cdf_results)
    # Empirical ranges include all observed inputs; fitted-state ranges exclude failed solves.
    ranges=[]
    amplitude=[c+'_'+m for c in ['HbO','HbR'] for m in ['sd','sd_relative_export','sd_subject_ratio','min','max','ptp','p95_minus_p05','relative_flag0_window']]
    for metric in amplitude+['tau','beta']+[s+'_'+m for s in state_names for m in ['mean','sd','rms_rest','min','max']]:
        g=frame if metric in amplitude else fit
        for flag in [0,1]:
            v=g.loc[g.flag==flag,metric].dropna();q=v.quantile([0,.01,.05,.25,.5,.75,.95,.99,1])
            ranges.append(dict(metric=metric,flag=flag,n=len(v),**{name:val for name,val in zip(['min','p01','p05','p25','median','p75','p95','p99','max'],q)}))
    pd.DataFrame(ranges).to_csv(out/'empirical_ranges.csv',index=False)
    rates=[]
    for c in ['HbO','HbR']:
        for base,thresholds in [('sd_subject_ratio',[.1,.25,.5]),('relative_flag0_window',[.25,.5,1.])]:
            for threshold in thresholds:
                col=c+'_'+base;df=frame[frame[col].notna()].copy();df['event']=(df[col]<=threshold).astype(float)
                v=df.groupby(['subject','flag','record']).event.mean().groupby(['subject','flag']).mean().reset_index()
                for flag,g in v.groupby('flag'):
                    pool=df[df.flag==flag]
                    rates.append(dict(component=c,reference=base,threshold=threshold,flag=flag,subject_equal_rate=g.event.mean(),
                        subjects=len(g),pooled_count=int(pool.event.sum()),pooled_total=len(pool),pooled_rate=pool.event.mean()))
    pd.DataFrame(rates).to_csv(out/'small_amplitude_rates.csv',index=False)
    write_json(out/'summary.json',dict(status='completed',counts=counts,subjects=32,primary_tests=14,
        inference='20000 paired-subject sign flips; Holm across tau/beta centers, 5 state means/SDs, and 2 CDF tests; 5000 paired subject bootstrap intervals',
        amplitude='all planned finite measured inputs including failed fits; empirical extrema are observed bounds, not physiological limits',
        state_order=['r','s','f','v','p','q'],duplicate_state='p=v exactly; no duplicate p significance test',new_fits=0))


def refed_filter_examples(cfg,report):
    import sys
    import matplotlib.pyplot as plt
    run=Path(cfg['run']);project=run.parents[4];sys.path.insert(0,str(project))
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    from src.data.unified_physiology import load_native_fnirs_record,load_cached_fnirs_record,_resample
    from src.data.homer2_preprocessing import bandpass_fnirs
    index=CleanPhysiologyCacheIndex(project/'data/cache/physiology_semantic_clean_v5')
    examples=json.loads(Path(cfg['native_examples']).read_text())['examples']
    coordinates=json.loads((run/'coordinates.json').read_text())['individual']
    descriptions=[];metrics=[];checks=[]
    for ex in examples:
        subject=str(ex['subject']);record=ex['record'];channels=[r['channel']-1 for r in ex['channels']]
        matches=[r for r in index.records if r.dataset_id=='refed' and r.canonical_subject_id==subject and r.base_record_id==record and r.signal_branch=='hbo_hbr']
        if len(matches)!=1:raise ValueError('Ambiguous registry record')
        source=matches[0];native=load_native_fnirs_record(project,source,record_only=True)
        raw=native['values'][:,channels,:];rate=native['sample_rate_hz'];time_s=native['time_s']
        if not np.isfinite(raw).all():raise ValueError('Selected native channels have missing support')
        high,hq=bandpass_fnirs(raw,sample_rate_hz=rate,low_hz=.01,high_hz=10.,order=3)
        low,lq=bandpass_fnirs(raw,sample_rate_hz=rate,low_hz=.01,high_hz=.2,order=3)
        if any(q['status']!='applied' for q in [hq,lq]):raise ValueError('Filter failed')
        cached=load_cached_fnirs_record(project,source)['values'][:,channels,:]
        replay=_resample(low.reshape(len(low),-1),rate,10.).reshape(-1,2,2)
        factor=coordinates['refed__'+subject]['hb_factor']
        if replay.shape!=cached.shape:raise ValueError('Resampled replay shape mismatch')
        replay_error=float(abs((replay-cached)*factor).max());np.testing.assert_allclose(replay*factor,cached*factor,rtol=1e-8,atol=1e-10)
        start=ex['channels'][0]['window_metadata']['hb_start_s'];take=(time_s>=start)&(time_s<start+90)
        local_t=time_s[take]-start;reference=local_t<5
        display={key:(v[take]-v[take][reference].mean(0))*factor for key,v in [('raw',raw),('0.01-10',high),('0.01-0.2',low)]}
        fig,axes=plt.subplots(2,3,figsize=(18,8),layout='constrained')
        for j,c in enumerate(['HbO','HbR']):
            lo=min(v[:,:,j].min() for v in display.values());hi=max(v[:,:,j].max() for v in display.values());pad=(hi-lo)*.07
            for ax,(name,value) in zip(axes[j],display.items()):
                for flag,color,ls in [(0,'#d98235','-'),(1,'#247b9d','--')]:
                    ax.plot(local_t,value[:,flag,j],color=color,ls=ls,lw=.8,label=f'flag {flag} / CH{channels[flag]+1}')
                    sd=float(value[:,flag,j].std());metrics.append(dict(subject=subject,record=record,component=c,flag=flag,view=name,sd=sd,ptp=float(np.ptp(value[:,flag,j]))))
                ax.set(xlim=(0,90),ylim=(lo-pad,hi+pad),xlabel='Time in window (s)',ylabel=c+' / saved common model scale',title=name+' | '+c)
                ax.axvspan(0,5,color='gray',alpha=.12);ax.grid(alpha=.2)
        fig.suptitle(f'REFED {subject} / {record} / {start:.1f}-{start+90:.1f}s | full-record filtering; shared y limits per Hb')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
        fig.savefig(report/f'filter_comparison_{subject}.png',dpi=220);plt.close(fig)
        np.savez_compressed(report/f'filter_comparison_{subject}.npz',time_s=local_t,channels=np.array(channels)+1,
            raw_export=raw[take],raw_model=display['raw'],band_001_10=display['0.01-10'],band_001_02=display['0.01-0.2'],hb_factor=factor)
        checks.append(dict(subject=subject,record=record,native_rate_hz=rate,cache_replay_max_model_error=replay_error,source=native['provenance']))
        descriptions.append(dict(subject=subject,record=record,start_s=start,channels=[c+1 for c in channels],hb_factor=factor))
    pd.DataFrame(metrics).to_csv(report/'filter_metrics.csv',index=False)
    write_json(report/'filter_verification.json',dict(status='passed',checks=checks,examples=descriptions,
        filtering='both branches independently from full native record, third-order Butterworth zero phase; native sampling retained in plot',
        normalization='same saved subject factor; each view uses same first-5s reference rule; no fitted SD normalization',new_fits=0))


def refed_state_report(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(cfg['export']);report=out/'report_v1';report.mkdir(exist_ok=False)
    tests=pd.read_csv(out/'parameter_state_tests.csv');ranges=pd.read_csv(out/'empirical_ranges.csv');rates=pd.read_csv(out/'small_amplitude_rates.csv')
    sub=pd.read_csv(out/'subject_states_all.csv',dtype={'subject':str});colors=['#d98235','#247b9d']
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for j,param in enumerate(['tau','beta']):
        cdf=pd.read_csv(out/(param+'_cdf.csv'))
        for flag in [0,1]:
            g=cdf[cdf.flag==flag].groupby('value').cdf.mean();axes[0,j].plot(g.index,g.values,color=colors[flag],label=f'flag {flag}')
        axes[0,j].set(xlabel=param+(' (s)' if param=='tau' else ' (effective gain)'),ylabel='Subject/record-weighted CDF',title=param+' distribution');axes[0,j].legend();axes[0,j].grid(alpha=.2)
        wide=sub.pivot(index='subject',columns='flag',values=param)
        for row in wide.to_numpy():axes[1,j].plot([0,1],row,color='gray',alpha=.3,lw=.8)
        for flag in [0,1]:axes[1,j].scatter(np.full(len(wide),flag),wide[flag],color=colors[flag],s=25)
        axes[1,j].set(xticks=[0,1],xticklabels=['flag 0','flag 1'],ylabel=param+(' (s)' if param=='tau' else ''),title='Video medians averaged within each subject')
        axes[1,j].axhline(2 if param=='tau' else 1,color='gray',ls=':',label='prior center');axes[1,j].grid(alpha=.2)
    fig.savefig(report/'parameters.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(18,8),layout='constrained')
    for j,name in enumerate(['r','s','f','v','q']):
        for i,metric in enumerate(['mean','sd']):
            key=name+'_'+metric;wide=sub.pivot(index='subject',columns='flag',values=key);ax=axes[i,j]
            for row in wide.to_numpy():ax.plot([0,1],row,color='gray',alpha=.25,lw=.6)
            for flag in [0,1]:ax.scatter(np.full(len(wide),flag),wide[flag],color=colors[flag],s=20)
            ax.set(xticks=[0,1],xticklabels=['0','1'],xlabel='Flag',ylabel=key,title=name+' / '+metric);ax.grid(alpha=.2)
    fig.suptitle('Internal state values | each connected pair is one subject | p = v exactly')
    fig.savefig(report/'internal_states.png',dpi=220);plt.close(fig)
    # Pooled amplitude distributions are descriptive; no point is treated as an independent subject.
    rows=[]
    for subject in cfg['subjects']:
        f=pd.read_csv(out/'subjects'/(subject+'.csv'),dtype={'subject':str});rows.append(f)
    frame=pd.concat(rows,ignore_index=True)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,c in zip(axes,['HbO','HbR']):
        for flag in [0,1]:
            v=np.sort(frame.loc[frame.flag==flag,c+'_sd']);positive=v>0
            ax.plot(v[positive],np.arange(1,len(v)+1)[positive]/len(v),color=colors[flag],label=f'flag {flag}')
        ax.set(xscale='log',xlabel='Measured SD in saved common-Hb model scale',ylabel='Fraction of window/channel inputs',title=c+' amplitude CDF (all planned inputs)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(report/'amplitude_distribution.png',dpi=220);plt.close(fig)
    refed_filter_examples(cfg,report)
    primary=tests[(tests.stratum=='all')&tests.primary]
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>REFED 参数、内部状态与滤波幅度审计</title><style>body{font-family:sans-serif;max-width:1500px;margin:auto;padding:30px;line-height:1.85;color:#243542}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border:1px solid #ccd}h1,h2{color:#173952}</style><h1>REFED 0/1组：τ、β、内部状态、小幅度普遍性与原始滤波对照</h1>
<p>继续分析已完成v3的32名被试、480个视频、40,800个窗口×通道任务。全部有限实测输入进入幅度分母，包括9个求解失败；参数及状态仅使用40,791个数值合格解。保留原被试共同Hb因子、前5秒参考、1:4:4权重及个体PCA，不重新拟合。标记为被试×通道元数据，0/1方向不由拟合优度决定。</p>
<h2>参数与状态的定义、检验</h2><p>τ（tau，秒）为该Tak模型的血管传输时间参数；β是当前归一化坐标中的有效神经血管增益（neurovascular_gain），不是绝对生理耦合强度。冻结先验中心τ=2、β=1，对数先验SD均0.125；允许边界分别[0.25,12]和[0.1,10]。参数为每个窗口×位置独立拟合；窗口重叠不产生独立被试。</p>
<p>状态数组顺序为[r,s,f,v,p,q]：r为模型驱动（模型坐标），s为血管舒张信号（df/dt，s⁻¹），f为相对血流、v为相对血容量、p为总Hb相关量、q为相对脱氧Hb量。f/v/p/q基准为1，r/s基准为0；当前Tak实现p=v严格成立，所以不把p再检验一次。小写s是内部变量，与输出分支大写S不同。报告均值、去均值SD以及相对基准RMS；这些是条件拟合状态，不是直接实测生理真值。</p>
<p>每个拟合先对有效时间点汇总；然后每视频/标记组内对拟合取中位数、被试内对视频均值，最后对32名被试作配对。主检验为τ/β中心差、5个状态的均值和SD差、τ/β的整体CDF差，共14项；使用20,000次被试级符号翻转，Holm控制这14项的多重比较。CDF先在各视频内计算，再视频/被试等权，用最大CDF差作统计量。区间为5,000次被试配对bootstrap的逐项95%区间。p值描述组间关联，不构成标记导致生理变化的因果检验。</p><img src="parameters.png">'''
    html+=primary.round(7).to_html(index=False)
    html+=pd.DataFrame(json.loads((out/'parameter_distribution_tests.json').read_text())).to_html(index=False)
    html+='''<p>上图上排是被试/视频等权CDF，下排每条线为同一被试两组的参数汇总；虚线为先验中心。统计显著性与实际变化幅度应一起读。另保留完整90秒及排除CH10/12/15的敏感性结果；其p值未加入主14项家族，仅作探索性补充。</p><img src="internal_states.png"><p>各面板单位与范围独立，第一行时间均值，第二行时间SD；每条连线一名被试。接近1的f/v/q均值与其波动SD是不同信息，不能把1作为波动幅度。</p>
<h2>1组究竟有多小：上下界、典型值和覆盖范围</h2><p>主幅度为每个窗口的去均值SD，使用已冻结的共同Hb模型坐标。下表分位数对全部窗口×通道描述，不作为独立统计样本。min/max是本次观察到的极值，不是生理上下界或置信区间；典型范围看P25–P75，尾部看P01–P99。另保留未经物理单位推定的发布数组相对单位SD、参考后最小/最大值、峰峰值及P95−P05。</p><img src="amplitude_distribution.png">'''
    html+=ranges[ranges.metric.isin(['HbO_sd','HbR_sd','HbO_sd_relative_export','HbR_sd_relative_export','HbO_ptp','HbR_ptp'])].to_html(index=False,float_format=lambda x:f'{x:.6g}')
    html+='''<p>“平缓”分两种参考：sd_subject_ratio是窗口SD/同被试原校准分量SD；relative_flag0_window是该窗口SD/同被试、同视频、同窗0组通道SD的中位数。阈值0.1/0.25/0.5等是描述性工程阈值，不是生理异常标准。发生率先记录内平均，再记录/被试等权；pooled_count/total另外保留以核对分母。</p>'''
    html+=rates[rates.flag==1].round(5).to_html(index=False)
    html+='''<h2>原始、0.01–10 Hz、0.01–0.2 Hz：同源波形对比</h2><p>复用前次被试26、2、13的同视频/同窗0/1通道对，不因本次滤波效果重新挑选。原始是发布的HbO/HbR，不是光强；通过统一原始读取器仅解码所需视频变量。两个滤波分支分别从完整原始视频开始，均为三阶零相位Butterworth，然后截取同一90秒；保持原生约47.62 Hz，以免10 Hz成分在10 Hz采样下混叠。这样隔离截止频率差异，不混入滤波阶数差异；作者原示例使用的IIR默认设置不等同于本图的匹配阶数对照。</p>
<p>显示时各视图使用同一被试保存因子，并按同一规则减其窗口前5秒均值；不除以各自SD。NPZ另保留未作显示参考的原始导出值。每行同一Hb的三列共用纵轴，0/1两条线也共用；HbO与HbR不同排。灰段为前5秒参考。低通分支进一步重采样至10 Hz，已与完整保留缓存核验；不将滤波图当作新SSM拟合。</p>'''
    verification=json.loads((report/'filter_verification.json').read_text())
    for ex in verification['examples']:
        html+=f'<h3>被试{ex["subject"]} / {ex["record"]} / {ex["start_s"]:.1f}–{ex["start_s"]+90:.1f}秒；0组CH{ex["channels"][0]}、1组CH{ex["channels"][1]}</h3><img src="filter_comparison_{ex["subject"]}.png">'
    html+=pd.read_csv(report/'filter_metrics.csv').round(6).to_html(index=False)
    html+='''<p><a href="../summary.json">分母及检验合同</a> · <a href="../parameter_state_tests.csv">参数/状态差异与校正p值</a> · <a href="../parameter_distribution_tests.json">整体参数分布检验</a> · <a href="../empirical_ranges.csv">完整幅度/状态范围</a> · <a href="../small_amplitude_rates.csv">小幅度覆盖率</a> · <a href="filter_verification.json">原始读取、同源滤波与缓存重放</a>。subjects目录保留每个任务的参数、状态时间统计及输入幅度；图旁NPZ保留同源曲线。</p></html>'''
    (report/'report.html').write_text(html,encoding='utf-8')


def refed_state_readout(cfg):
    """Versioned reading guide and zooms from the already exported numeric curves."""
    import shutil
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(cfg['export']);old=out/'report_v1';new=out/'report_v2'
    shutil.copytree(old,new)
    frame=pd.concat([pd.read_csv(out/'subjects'/(s+'.csv')) for s in cfg['subjects']],ignore_index=True)
    fits=frame[frame.qualified].copy();fits['tau_upper']=fits.tau>=12-1e-6
    boundaries=[]
    for flag,g in fits.groupby('flag'):
        sub=g.groupby(['subject','record']).tau_upper.mean().groupby('subject').mean()
        boundaries.append(dict(flag=int(flag),qualified=len(g),tau_upper_count=int(g.tau_upper.sum()),pooled_rate=g.tau_upper.mean(),subject_equal_rate=sub.mean()))
    pd.DataFrame(boundaries).to_csv(out/'parameter_boundary_rates.csv',index=False)
    examples=json.loads((new/'filter_verification.json').read_text())['examples']
    for ex in examples:
        with np.load(new/f'filter_comparison_{ex["subject"]}.npz',allow_pickle=False) as a:
            t=a['time_s'];columns=[('Raw Hb (5s referenced)',a['raw_model']),('0.01-10 Hz',a['band_001_10']),('0.01-0.2 Hz',a['band_001_02'])]
            fig,axes=plt.subplots(2,3,figsize=(18,8),layout='constrained')
            for j,c in enumerate(['HbO','HbR']):
                for ax,(name,v) in zip(axes[j],columns):
                    for flag,color,ls in [(0,'#d98235','-'),(1,'#247b9d','--')]:ax.plot(t,v[:,flag,j],color=color,ls=ls,lw=.8,label=f'flag {flag} / CH{ex["channels"][flag]}')
                    ax.set(xlim=(0,90),xlabel='Time in window (s)',ylabel=c+' / saved model scale',title=f'{name} | SD 0/1 = {v[:,0,j].std():.4g} / {v[:,1,j].std():.4g}')
                    ax.axvspan(0,5,color='gray',alpha=.12);ax.grid(alpha=.2)
            fig.suptitle(f'REFED {ex["subject"]} / {ex["record"]} | independent column y limits; identical numeric normalization')
            fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
            fig.savefig(new/f'filter_zoom_{ex["subject"]}.png',dpi=220);plt.close(fig)
    intro='''<h2>主要结果与统计解释</h2><p><b>τ和β都有显著分布差异。</b>按主被试等权汇总，0/1组τ为2.1749/2.0623秒，差−0.1125秒（95%区间−0.1773至−0.0624），β为1.1474/0.9649，差−0.1825（−0.2360至−0.1394）；这两项及两项整体CDF检验的Holm校正p均约0.00070。显著性不意味着变化量都很大，τ典型汇总约下降5.2%，β约15.9%。</p>
<p>τ的关键差异还在分布宽度：全部合格拟合的0组P05–P95为1.578–7.744秒，1组为2.010–2.358秒；池化中位数分别2.043、2.056，与上述视频/被试汇总不是同一统计量。0组τ触及12秒上限97/5569，1组5/35222。0组的长尾说明“波形更容易重建”并不等于参数估计全部稳定，1组更靠近先验的现象也不是参数真值验证。</p>
<p><b>内部状态差异显著，1组血管状态更接近模型静息点。</b>0/1组f均值为0.9457/0.9986，v为0.9819/0.9995，q为1.0302/1.0009。f波动SD为0.03073/0.01085、v为0.009919/0.002841、q为0.02046/0.004665，分别约减少65%、71%、77%；这些均值/SD检验Holm校正p约0.00070。驱动r的SD为0.01975/0.01905，只减少约3.6%，校正p约0.0034。即驱动波动仍相近，而血管响应与输入Hb尺度有明显组间差异。状态是在固定观测参考、先验和坐标下的条件模型量，不能把这些均值直接解释为真实血流下降。</p>
<p><b>1组弱幅度普遍存在，但并非所有记录都接近零。</b>相对同被试同视频同窗0组典型SD，1组HbO有81.9%不超过其一半、54.1%不超过四分之一；HbR分别77.7%、53.6%。这些是视频/被试等权发生率。其模型坐标SD中位数为HbO 0.002806、HbR 0.002018；P25–P75分别0.001660–0.004967、0.001054–0.003919；P01–P99分别0.000547–0.022390、0.000245–0.021587。观测SD最小接近数值零，最大为0.3319和0.2115，保留少量高幅度尾部。</p>
<p>若“上下界”指参考后波形值而非SD，全部1组输入的HbO实测范围为−1.9649至0.3504，HbR为−0.2977至1.2442，单位均为保存的模型坐标。极值受异常片段影响，不代表典型幅度，更不是正常生理区间。</p>
<p><b>同源滤波图显示，小幅度差异在所选原始记录中已经存在，窄频段又进一步降低两组波动。</b>例如被试2的HbO，原始/0.01–10/0.01–0.2 Hz的SD分别为0组0.07457/0.04623/0.006830，1组0.01281/0.008555/0.001198。此处三段都是同一原始片段、同一保存尺度；不能把频段改变当作单位换算。三个例子的低频重放与原缓存最大误差均为0。原始滤波对照只有这三组样例，普遍性统计对应完整的现有0.01–0.2 Hz输入。</p>'''
    path=new/'report.html';body=path.read_text();body=body.replace('<h2>参数与状态的定义、检验</h2>',intro+'<h2>参数与状态的定义、检验</h2>',1)
    for ex in examples:
        needle=f'<img src="filter_comparison_{ex["subject"]}.png">'
        body=body.replace(needle,needle+f'<p>上图统一纵轴读幅度；下图仅独立缩放各列纵轴读形态，数值和归一化完全相同。</p><img src="filter_zoom_{ex["subject"]}.png">')
    path.write_text(body,encoding='utf-8')


def refed_state_main(parent,export,stage,workers):
    out=Path(export).resolve();run=Path(parent['run'])
    if out.parent!=Path(parent['output']):raise ValueError('Fresh Hb audit child required')
    flags=pd.read_csv(run.parents[4]/'data/REFED-dataset/fNIRS_reservations.csv',dtype={'sub_id':str}).set_index('sub_id')
    cfg=dict(schema='refed_state_amplitude_v1',run=str(run),export=str(out),subjects=sorted(flags.index,key=int),
        flags={s:flags.loc[s].astype(int).tolist() for s in flags.index},
        native_examples=str(Path(parent['output'])/'refed_flags_20261110_v1/report_v2/example_selection.json'),
        boundary='completed public REFED fits; no refits or production changes; raw Hb reads limited to the three preceding selected records for requested filter plots',
        primary_tests=['tau_median','beta_median','r_mean','r_SD','s_mean','s_SD','f_mean','f_SD','v_mean','v_SD','q_mean','q_SD','tau_CDF','beta_CDF'],
        aggregation='time summary per fit; median fits within video/flag; mean videos within subject; paired subjects; sensitivity for 90s and excluding CH10/12/15',
        amplitude='SD, peak-to-peak, P95-P05 and referenced extrema in frozen model scale; SD also in relative export units',
        filters='native full-video input; matched third-order zero-phase Butterworth 0.01-10 and 0.01-0.2 Hz; same saved subject factor; no window renormalization')
    if stage=='plan':out.mkdir(exist_ok=False);(out/'subjects').mkdir();write_json(out/'analysis_config.json',cfg);return
    if json.loads((out/'analysis_config.json').read_text())!=cfg:raise ValueError('Frozen analysis changed')
    if stage=='report':refed_state_report(cfg);return
    if stage=='readout':refed_state_readout(cfg);return
    if not os.environ.get('SSM_SYSTEMD_UNIT'):raise ValueError('Use durable supervisor')
    start=time.monotonic();done=[];subjects=cfg['subjects'] if stage=='run' else ['1','7']
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(refed_state_worker,[(cfg,s) for s in subjects]):
            done.append(r);print(json.dumps(r),flush=True);write_json(out/(stage+'_progress.json'),dict(completed=len(done),total=len(subjects)))
    write_json(out/(stage+'_execution.json'),dict(seconds=time.monotonic()-start,workers=workers,subjects=len(done),peak_worker_mib=max(r['peak_rss_mib'] for r in done),supervisor=os.environ['SSM_SYSTEMD_UNIT']))
    if stage=='run':refed_state_summary(cfg)


def fnirs_main():
    import sys
    p=argparse.ArgumentParser(description='Read-only Hb morphology/cancellation census')
    p.add_argument('--fnirs-audit',action='store_true')
    for key in ['run','features','comparison','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stage',choices=['plan','pilot','run','summarize','report','examples','source-audit','visual-plan','visual-pilot','visual-run','visual-report','refed-plan','refed-pilot','refed-run','refed-report','author-check','state-plan','state-pilot','state-run','state-report','state-readout'],required=True)
    p.add_argument('--example-export',type=Path)
    p.add_argument('--source-export',type=Path)
    p.add_argument('--visual-export',type=Path)
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
    if a.stage.startswith('state-'):
        if a.source_export is None:p.error('--source-export is required for state analysis')
        refed_state_main(cfg,a.source_export,a.stage.removeprefix('state-'),a.workers);return
    if a.stage=='author-check':
        if a.source_export is None:p.error('--source-export is required for author-check')
        refed_author_probe(a.source_export);return
    if a.stage.startswith('refed-'):
        if a.source_export is None:p.error('--source-export is required for REFED stages')
        refed_flag_main(cfg,a.source_export,a.stage.removeprefix('refed-'),a.workers);return
    if a.stage.startswith('visual-'):
        if a.visual_export is None:p.error('--visual-export is required for Visual stages')
        visual_mask_main(cfg,a.visual_export,a.stage,a.workers);return
    if a.stage=='source-audit':
        if a.source_export is None:p.error('--source-export is required for source-audit')
        fnirs_source_audit(cfg,a.source_export);return
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


def amplitude_subject(payload):
    """Read frozen prepared Hb, retaining failed-fit inputs and native columns."""
    cfg,key,co=payload;start=time.monotonic();ds,subject=key.split('__');out=Path(cfg['export'])/'subjects'/key
    if out.with_suffix('.json').exists():return json.loads(out.with_suffix('.json').read_text())
    root=Path(cfg['visual_run'] if ds=='visual_cognitive_motivation' else cfg['run'])
    rows=[];replay=0.;windows=0
    for source in co['sources']:
        path=root/'features'/source['record_key']/(source['window_key']+'.npz')
        meta=json.loads(path.with_suffix('.json').read_text());record=source['record_key'].split('/')[-1]
        if meta['status']!='completed' or meta['join_key']!='|'.join([ds,subject,record]):raise ValueError('Prepared identity mismatch')
        with np.load(path,allow_pickle=False) as a:
            hb=a['hb'];valid=a['hb_valid']
        indices=list(range(hb.shape[1]))
        if ds=='visual_cognitive_motivation':
            indices=meta['fnirs_selection']['retained_pair_indices']
            with np.load(meta['source_prepared'],allow_pickle=False) as a:old=a['hb'][:,indices]
            replay=max(replay,float(np.max(np.abs(hb-old))));np.testing.assert_allclose(hb,old,rtol=0,atol=1e-12)
        if hb.shape!=(source['steps'],len(indices),2):raise ValueError('Invalid prepared Hb shape')
        if np.any(valid & ~np.isfinite(hb).all(axis=(0,2))):raise ValueError('Nonfinite admitted Hb')
        # Exactly the retained fit/score support: last 100 ms bin excluded.
        z=hb[:-1]*co['hb_factor'];sd=z.std(0);lo=z.min(0);hi=z.max(0)
        for i,native in enumerate(indices):
            row=dict(dataset=ds,subject=subject,record=record,window=source['window_key'],native_index=native,
                duration_s=meta['window']['duration_s'],hb_factor=co['hb_factor'],input_valid=bool(valid[i]))
            for j,c in enumerate(['HbO','HbR']):row.update({c+'_sd':sd[i,j],c+'_min':lo[i,j],c+'_max':hi[i,j],c+'_ptp':hi[i,j]-lo[i,j]})
            rows.append(row)
        windows+=1
    pd.DataFrame(rows).to_csv(out.with_suffix('.csv'),index=False)
    result=dict(subject=key,windows=windows,pair_windows=len(rows),visual_replay_max_error=replay,
        seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(out.with_suffix('.json'),result);return result


def amplitude_weighted_quantile(values,weights,probabilities):
    """Inverse weighted empirical CDF; no interpolation across zero-amplitude mass."""
    v=np.asarray(values,float);w=np.asarray(weights,float);order=np.argsort(v)
    if not np.isfinite(v).all() or not np.isfinite(w).all() or np.any(w<=0):raise ValueError('Invalid weighted sample')
    c=np.cumsum(w[order]);c/=c[-1]
    return v[order][np.minimum(np.searchsorted(c,probabilities),len(v)-1)]


def amplitude_weights(frame):
    # Equal subjects, then records, then windows, then available channels.
    k=['dataset','subject','record','window'];n=frame.groupby(k).native_index.transform('size')
    wins=frame[k].drop_duplicates().groupby(k[:3]).size()
    recs=frame[k[:3]].drop_duplicates().groupby(k[:2]).size()
    nw=wins.reindex(pd.MultiIndex.from_frame(frame[k[:3]])).to_numpy()
    nr=recs.reindex(pd.MultiIndex.from_frame(frame[k[:2]])).to_numpy()
    return 1/(n.to_numpy()*nw*nr)


def amplitude_summary(cfg):
    from scipy.stats import spearmanr
    out=Path(cfg['export']);keys=cfg['subjects'];frames=[pd.read_csv(out/'subjects'/(k+'.csv'),dtype={'subject':str}) for k in keys]
    frame=pd.concat(frames,ignore_index=True)
    if 'input_valid' not in frame:raise ValueError('Input validity field required')
    # Earlier successful pilot shards contained only finite, admitted Hb.
    frame['input_valid']=frame.input_valid.map(lambda x: True if pd.isna(x) else bool(x))
    counts=frame.groupby('dataset').size().to_dict()
    if counts!=cfg['expected_pair_windows']:raise ValueError(f'Input census mismatch: {counts}')
    unavailable=frame[~frame.input_valid].groupby('dataset').size().to_dict()
    frame=frame[frame.input_valid].copy();frame['weight']=amplitude_weights(frame)
    qprob=[0,.01,.05,.25,.5,.75,.95,.99,1];qnames=['min','p01','p05','p25','median','p75','p95','p99','max']
    summaries=[];spreads=[];ratios=[]
    for scope,f in [('all',frame),('90s',frame[np.isclose(frame.duration_s,90)].copy())]:
        if scope=='90s':f['weight']=amplitude_weights(f)
        for ds,g in f.groupby('dataset'):
            for c in ['HbO','HbR']:
                q=amplitude_weighted_quantile(g[c+'_sd'],g.weight,qprob)
                summaries.append(dict(scope=scope,dataset=ds,component=c,n=len(g),subjects=g.subject.nunique(),
                    **dict(zip(qnames,q)),p99_over_p01=q[7]/q[1],p95_over_p05=q[6]/q[2],
                    below_0025=float(np.average(g[c+'_sd']<.0025,weights=g.weight)),
                    below_005=float(np.average(g[c+'_sd']<.005,weights=g.weight)),
                    near_constant=int((g[c+'_sd']<=1e-12).sum()),waveform_min=g[c+'_min'].min(),waveform_max=g[c+'_max'].max()))
            usable=g[(g.HbO_sd>1e-12)&(g.HbR_sd>1e-12)];ratio=usable.HbO_sd/usable.HbR_sd
            ratios.append(dict(scope=scope,dataset=ds,defined=len(usable),near_constant_excluded=len(g)-len(usable),
                **dict(zip(qnames,amplitude_weighted_quantile(ratio,amplitude_weights(usable),qprob)))))
    for (ds,sub,rec,win),g in frame.groupby(['dataset','subject','record','window']):
        for c in ['HbO','HbR']:
            a=g[c+'_sd'].to_numpy();p05,p95=np.quantile(a,[.05,.95])
            spreads.append(dict(dataset=ds,subject=sub,record=rec,window=win,component=c,channels=len(g),
                duration_s=g.duration_s.iloc[0],p95_over_p05=p95/p05 if p05>0 else np.inf,
                max_over_min=a.max()/a.min() if a.min()>0 else np.inf))
    spread=pd.DataFrame(spreads);spread.to_csv(out/'within_window_spread.csv',index=False)
    ss=spread.groupby(['dataset','subject','component','record'])[['p95_over_p05','max_over_min']].median().groupby(['dataset','subject','component']).mean().reset_index()
    ss.to_csv(out/'subject_channel_spread.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'amplitude_distribution.csv',index=False);pd.DataFrame(ratios).to_csv(out/'hbo_hbr_ratio.csv',index=False)
    # REFED input, fitted parameters/states, and S/O diagnostics share exact identities.
    audit=Path(cfg['audit']);states=pd.concat([pd.read_csv(audit/'refed_state_amplitude_20261110_v1/subjects'/(s+'.csv'),dtype={'subject':str}) for s in sorted(frame[frame.dataset=='refed'].subject.unique())],ignore_index=True)
    amp=frame[frame.dataset=='refed'].copy();amp['channel']=amp.native_index+1
    joined=amp.merge(states,on=['subject','record','window','channel'],validate='one_to_one',suffixes=('','_state'))
    for c in ['HbO','HbR']:np.testing.assert_allclose(joined[c+'_sd'],joined[c+'_sd_state'],rtol=1e-10,atol=1e-13)
    old=pd.concat([pd.read_csv(audit/'subjects'/('refed__'+s+'.csv'),dtype={'subject':str}) for s in sorted(amp.subject.unique())],ignore_index=True)
    old['channel']=old.pair.str[2:].astype(int);join=['subject','record','window','channel'];diag=[]
    for c in ['HbO','HbR']:
        g=joined.merge(old[(old.component==c)&old.defined],on=join,validate='one_to_one',suffixes=('','_fit'))
        g['component']=c;g['input_sd']=g[c+'_sd'];g['severe']=g.s_sd_ratio>=2
        g['output_sd']=g.input_sd*g.s_sd_ratio
        g['posthoc_gain']=np.maximum(0,g.correlation_sx/g.s_sd_ratio)
        g['posthoc_residual_ratio']=np.sqrt(np.maximum(0,1-np.maximum(0,g.correlation_sx)**2))
        g['amplitude_bin']=pd.cut(g.input_sd,[0,.001,.0025,.005,.01,.025,np.inf],right=False).astype(str)
        diag.append(g[['dataset','subject','record','window','native_index','channel','component','flag','duration_s','input_sd','output_sd','beta','tau','s_sd_ratio','o_sd_ratio','correlation_sx','correlation_so','severe','posthoc_gain','posthoc_residual_ratio','amplitude_bin']])
    diag=pd.concat(diag,ignore_index=True);diag.to_csv(out/'refed_cancellation_join.csv',index=False)
    strata=[];beta=[];associations=[]
    def equal_subject(g,cols):return g.groupby(['subject','record'])[cols].mean().groupby('subject').mean()
    for c,g in diag.groupby('component'):
        for label,v in g.groupby('amplitude_bin'):
            rates=equal_subject(v,['severe']);strata.append(dict(component=c,bin=label,n=len(v),subjects=len(rates),
                severe_rate=rates.severe.mean(),input_sd_median=v.input_sd.median(),output_sd_median=v.output_sd.median(),beta_median=v.beta.median()))
        for label,v in g.groupby('severe'):
            beta.append(dict(component=c,severe=bool(label),n=len(v),beta_min=v.beta.min(),beta_p05=v.beta.quantile(.05),beta_median=v.beta.median(),beta_p95=v.beta.quantile(.95),beta_max=v.beta.max(),
                beta_at_lower=int(np.isclose(v.beta,.1,atol=1e-6,rtol=0).sum()),gain_median=v.posthoc_gain.median(),
                negative_correlation=float((v.correlation_sx<=0).mean()),posthoc_residual_median=v.posthoc_residual_ratio.median()))
        for sub,v in g.groupby('subject'):
            associations.append(dict(subject=sub,component=c,rho_amplitude_beta=spearmanr(v.input_sd,v.beta).statistic,
                rho_amplitude_inflation=spearmanr(v.input_sd,v.s_sd_ratio).statistic))
    pd.DataFrame(strata).to_csv(out/'refed_amplitude_strata.csv',index=False);pd.DataFrame(beta).to_csv(out/'refed_beta_by_cancellation.csv',index=False)
    pd.DataFrame(associations).to_csv(out/'refed_subject_associations.csv',index=False)
    # Positive gain applied to both Hb outputs is a waveform-only diagnostic, not a beta refit.
    a,b=[diag[diag.component==c].set_index(join) for c in ['HbO','HbR']]
    both=a.join(b,lsuffix='_o',rsuffix='_r',how='inner');den=both.output_sd_o**2+both.output_sd_r**2
    both['common_gain']=np.maximum(0,(both.correlation_sx_o*both.input_sd_o*both.output_sd_o+both.correlation_sx_r*both.input_sd_r*both.output_sd_r)/den)
    common=both[['common_gain']].copy()
    for c,suffix in [('HbO','o'),('HbR','r')]:
        ar=both['s_sd_ratio_'+suffix];r=both['correlation_sx_'+suffix];k=both.common_gain
        common[c+'_residual_ratio']=np.sqrt(np.maximum(0,1+(k*ar)**2-2*k*ar*r))
        common[c+'_severe']=both['severe_'+suffix]
    common.reset_index().to_csv(out/'refed_common_gain_diagnostic.csv',index=False)
    write_json(out/'summary.json',dict(status='completed',input_counts=counts,unavailable_input_counts=unavailable,subjects=frame.groupby('dataset').subject.nunique().to_dict(),
        amplitude_support='all finite admitted inputs including failed fits; missing/unsupported inputs counted separately; exact scored time support',
        visual='completed usable-channel prepared inputs and frozen v4 coordinates; no running v4 fits read',
        refed_qualified=int(states.qualified.sum()),refed_failed=int((~states.qualified).sum()),refed_defined_component_rows=len(diag),
        refed_undefined_component_rows=2*int(states.qualified.sum())-len(diag),
        weighting='CDF: equal subjects/records/windows/channels; channel-spread: median windows within record then mean records within subject',
        exploratory='descriptive associations and posthoc gains; no causal beta or new fitted-model claim'))
    write_json(out/'verification.json',dict(status='passed',input_census_matches=True,refed_input_replay=True,
        visual_prepared_replay_max_error=max(json.loads((out/'subjects'/(k+'.json')).read_text())['visual_replay_max_error'] for k in keys),
        joins_unique=True,visual_fit_results_read=0,raw_reads=0,model_refits=0))


def amplitude_report(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(cfg['export']);version=1
    while (out/f'report_v{version}').exists():version+=1
    report=out/f'report_v{version}';report.mkdir(exist_ok=False)
    d=pd.read_csv(out/'amplitude_distribution.csv');spread=pd.read_csv(out/'subject_channel_spread.csv')
    beta=pd.read_csv(out/'refed_beta_by_cancellation.csv');strata=pd.read_csv(out/'refed_amplitude_strata.csv')
    ratio=pd.read_csv(out/'hbo_hbr_ratio.csv');diag=pd.read_csv(out/'refed_cancellation_join.csv');summary=json.loads((out/'summary.json').read_text())
    coordinates=json.loads((Path(cfg['run'])/'coordinates.json').read_text())['individual']
    objective=[]
    for (c,severe),g in diag.groupby(['component','severe']):
        j=1 if c=='HbO' else 2
        den=np.array([coordinates['refed__'+str(s)]['sd'][j] for s in g.subject])
        relative=g.input_sd.to_numpy()/den
        objective.append(dict(component=c,severe=bool(severe),n=len(g),input_sd_over_subject_sd_median=np.median(relative),
            residual_sd_over_subject_sd_median=np.median(relative*g.o_sd_ratio),residual_sd_over_input_sd_median=g.o_sd_ratio.median()))
    pd.DataFrame(objective).to_csv(report/'refed_objective_scale.csv',index=False)
    labels={'eeg_fnirs_single_trial':'Single','simultaneous_eeg_nirs':'Sim','refed':'REFED','visual_cognitive_motivation':'Visual usable'}
    order=list(labels);colors=['#5276a7','#4b9b77','#cb6757','#ad7db5']
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,c in zip(axes,['HbO','HbR']):
        for i,ds in enumerate(order):
            v=d[(d.scope=='all')&(d.component==c)&(d.dataset==ds)].iloc[0]
            ax.plot([v.p01,v.p99],[i,i],c=colors[i],lw=2);ax.plot([v.p25,v.p75],[i,i],c=colors[i],lw=9,solid_capstyle='butt');ax.scatter(v['median'],i,c='black',s=24,zorder=3)
        ax.set(xscale='log',yticks=range(4),yticklabels=[labels[x] for x in order],xlabel='Window SD in frozen model coordinates',title=c+' | line P01-P99; bar IQR; dot median')
        ax.set_xlim(d.p01.min()*.8,d.p99.max()*1.2)
        ax.axvline(.0025,c='grey',ls='--',alpha=.6);ax.grid(axis='x',alpha=.2)
    fig.savefig(report/'amplitude_ranges.png',dpi=210);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,c in zip(axes,['HbO','HbR']):
        for i,ds in enumerate(order):
            v=spread[(spread.component==c)&(spread.dataset==ds)].p95_over_p05.to_numpy()
            ax.scatter(i+np.linspace(-.13,.13,len(v)),v,color=colors[i],s=18,alpha=.7)
            ax.plot([i-.22,i+.22],[np.median(v)]*2,c='black',lw=2)
        ax.set(yscale='log',xticks=range(4),xticklabels=[labels[x] for x in order],ylabel='Same-window channel SD: P95 / P05',title=c+' | each dot = one subject')
        ax.set_ylim(spread.p95_over_p05.min()*.8,spread.p95_over_p05.max()*1.2);ax.grid(axis='y',alpha=.2)
    fig.savefig(report/'channel_spread.png',dpi=210);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');bins=np.array([.001,.0025,.005,.01,.025]);flagbins=[]
    for j,c in enumerate(['HbO','HbR']):
        g=diag[diag.component==c].copy();g['bin']=np.digitize(g.input_sd,bins)
        for flag,color in [(0,'#377eb8'),(1,'#e27932')]:
            rows=[]
            for b,v in g[g.flag==flag].groupby('bin'):
                s=v.groupby(['subject','record'])[['severe','beta']].mean().groupby('subject').mean();rows.append((b,s.severe.mean()*100,s.beta.mean()))
                flagbins.append(dict(component=c,flag=flag,bin=int(b),n=len(v),subjects=len(s),severe_percent=s.severe.mean()*100,beta_mean=s.beta.mean()))
            a=np.array(rows);axes[0,j].plot(a[:,0],a[:,1],'o-',c=color,label='flag '+str(flag));axes[1,j].plot(a[:,0],a[:,2],'o-',c=color,label='flag '+str(flag))
        axes[0,j].set(title=c,ylabel='Severe cancellation (%)',ylim=(-3,103));axes[1,j].set(ylabel='Mean fitted beta',xlabel='Input window SD bin',ylim=(.88,1.75))
        for ax in axes[:,j]:ax.set_xticks(range(6),['<.001','.001-.0025','.0025-.005','.005-.01','.01-.025','>=.025'],rotation=25);ax.legend();ax.grid(alpha=.2)
    fig.savefig(report/'refed_amplitude_beta.png',dpi=210);plt.close(fig)
    pd.DataFrame(flagbins).to_csv(report/'flag_amplitude_strata.csv',index=False)
    def table(x):return x.to_html(index=False,float_format=lambda v:f'{v:.5g}',border=0)
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>跨数据集幅度跨度与 REFED 抵消</title><style>body{font:17px/1.7 sans-serif;max-width:1200px;margin:40px auto;padding:0 20px}img{width:100%}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}td,th{padding:7px;border-bottom:1px solid #ddd}h2{margin-top:36px}</style><h1>跨数据集 fNIRS 幅度跨度与 REFED 的 S/O 抵消</h1>'''
    body+='<p>比较对象：全部计划求解输入，Single 29、Sim 26、REFED 32、Visual 16 名被试。输入已完整记录滤波至 0.01–0.2 Hz、重采样 10 Hz，按已冻结的被试共同 Hb 因子缩放；每窗先前已做首 5 秒参考，SD 去均值，不受参考常数影响。保留重叠窗口身份；最后 100 ms 与实际评分一致排除。这里比较模型坐标中的输入幅度，不将不同设备的未知原生单位解释为共同浓度。</p>'
    body+='<p>Visual 使用已完成并验证的 v6 可用通道输入、v4 冻结尺度。设备标记 1 的通道按记录排除，不按小幅度阈值排除；此前被排除通道中的大幅度输入曾影响共同归一化。新版 SSM 尚在运行，因此本报告不使用其部分拟合结果，也不沿用旧版 Visual 抵消率作为新版结果。</p>'
    body+='<h2>总体幅度：上下界、典型值与主体跨度</h2><img src="amplitude_ranges.png"><p>幅度定义为单个通道、单个窗口时间序列的 SD。经验分布按被试→记录→窗口→通道逐层等权。线为 P01–P99，粗条为 P25–P75，点为中位数。最小/最大值是全部输入的经验极值，接近常量会使最大/最小比失真，主体跨度优先看 P99/P01 和 P95/P05。虚线 0.0025 只是共同尺度目标 0.025 的十分之一，不是通道有效性阈值。</p>'
    body+=table(d[d.scope=='all'][['dataset','component','n','min','p01','p05','median','p95','p99','max','p99_over_p01','p95_over_p05','below_0025','near_constant']])
    body+='<h2>同一时间窗内的通道差异</h2><img src="channel_spread.png"><p>每窗先算通道 SD 的 P95/P05，再记录内取窗口中位数、被试内记录等权平均；每个点是一名被试，黑线是被试中位数。此比值在共同正增益下不变，区分了同窗通道异质性与跨被试/跨任务幅度差异。Visual 的每个原生 Probe 记录分别统计，不混合不同采样支持。</p>'
    body+=table(spread.groupby(['dataset','component']).p95_over_p05.agg(['median','min','max']).reset_index())
    body+='<h2>HbO 与 HbR 本身的幅度比例</h2><p>对同一拟合位置/窗口计算 SD(HbO)/SD(HbR)，再按相同层级加权；共同 Hb 因子严格抵消。任一成分 SD≤1e−12 的数值常量另计，不计算不稳定比例。β 不是两个独立的色团幅度系数，因此这一分布也需单独考察。</p>'+table(ratio[ratio.scope=='all'])
    body+='<h2>REFED：小幅度、β 与抵消的关系</h2><img src="refed_amplitude_beta.png"><p>严重抵消沿用 SD(S)/SD(X)≥2；O=X−S。在该条件下残差必然很大且与 S 反向。按固定输入 SD 分箱后，每箱先记录、再被试等权；缺少该箱的被试不补零，因此不同箱的被试构成可能不同。曲线分别显示 flag 0/1，不赋予其有效性极性。表中给出全部标记的分箱分母。</p>'+table(strata)
    body+='<p>上图每个点的样本数与被试数如下，bin 0–5 依次对应图中六个区间；少量被试的点不能解释为总体比例。</p>'+table(pd.DataFrame(flagbins))
    body+='<p>β 的硬边界为 [0.1,10]，先验中心 1、log SD 0.125。下表的分位数为全部已定义拟合的合并描述，不把重复窗口当作独立推断样本。beta_at_lower 是触及 0.1 的数量；gain_median 是保持 S 形态时的最优非负事后缩放，不是重估的 β。</p>'+table(beta)
    body+='<p>实际动力学为 ds/dt=βr−κs−γ(f−1)，df/dt=s；HbR=Q₀(q−1)，HbO=P₀(v−1)−Q₀(q−1)。β 通过共同血流系统改变两条 Hb 输出；驱动、初态、τ、流量约束和参数先验共同影响解。不同通道独立求解，所以不能将数据集 P99/P01 直接与 β 的 100 倍搜索范围比较后宣称“容量不够”。触界与先验牵制也需要分别判断。</p>'
    body+='<p>波形缩放诊断：令 a=SD(S)/SD(X)、r=corr(X,S)，最优非负中心化缩放 k=max(0,r/a)，其残差 SD/X SD 为 sqrt(1−max(0,r)²)。它回答幅度调整后仍剩多少形态误差；不满足重新积分的状态方程，也不代表修改 β 后会得到该波形。另存同时缩放 HbO/HbR 的共同增益结果，按两者中心化平方误差求最优值。以上均未重新拟合模型。</p>'
    body+='<h2>为什么残差相对原始波形很大，求解仍可能接受</h2><p>当前数据项为固定被试分量 SD 归一化的平方误差，HbO/HbR 权重各为 4/3，EEG 为 1/3；它不按当前通道当前窗口的 SD 重新加权。因此局部信号很小时，相对该波形数倍的残差，仍可能只占被试尺度的一小部分。下表给出合并中位数；残差 SD 只描述波动部分，完整目标还计入残差均值、初态、驱动、流量和参数先验。β 先验平方惩罚为 (ln β/0.125)²，在 β=0.5 时为 30.75，β=0.1 时为 339.32；这些惩罚直接进入既有优化目标。</p>'+table(pd.DataFrame(objective))
    body+='<h2>90 秒窗口敏感性与覆盖</h2>'+table(d[d.scope=='90s'][['dataset','component','n','p01','median','p99','p99_over_p01','below_0025']])
    body+=f'<p>总计划输入 {sum(summary["input_counts"].values())} 个位置×窗口，其中不可用输入 {summary["unavailable_input_counts"]}，另计而不作为零幅度。其余有限有效输入全部纳入，包括求解失败。</p>'
    body+=f'<p>REFED：{summary["refed_qualified"]} 个合格拟合、{summary["refed_failed"]} 个失败；输入统计保留失败。参数/抵消关联限于已定义的 {summary["refed_defined_component_rows"]} 条色团结果，另有 {summary["refed_undefined_component_rows"]} 条相关未定义。幅度和比例是描述性统计，未经新的模型消融不能将关联解释为 β 约束的因果效应。</p>'
    body+='<p><a href="../summary.json">汇总与口径</a> · <a href="../verification.json">校验</a> · <a href="../amplitude_distribution.csv">完整幅度表</a> · <a href="../within_window_spread.csv">逐窗通道跨度</a> · <a href="../refed_cancellation_join.csv">REFED 逐拟合关联</a> · <a href="../refed_common_gain_diagnostic.csv">Hb 共同增益诊断</a></p></html>'
    (report/'report.html').write_text(body)


def amplitude_comparison_main():
    p=argparse.ArgumentParser();p.add_argument('--amplitude-comparison',action='store_true')
    for k in ['run','visual-run','audit','export']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--stage',choices=['plan','pilot','run','report'],required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    out=a.export.resolve();run=a.run.resolve();visual=a.visual_run.resolve()
    if out.parent!=a.audit.resolve() or a.audit.resolve().parent!=run:raise ValueError('Use a fresh retained audit child')
    if json.loads((run/'verification.json').read_text())['status']!='passed':raise ValueError('Verified v3 required')
    prep=json.loads((visual/'preparation_execution.json').read_text());checks=json.loads((visual/'migration_checks.json').read_text())
    if prep['status']!='completed' or checks['status']!='passed':raise ValueError('Completed verified Visual preparation required')
    co=json.loads((visual/'coordinates.json').read_text())['individual'];old=json.loads((run/'coordinates.json').read_text())['individual']
    for key,v in co.items():
        if v['dataset']!='visual_cognitive_motivation' and v!=old[key]:raise ValueError('Unchanged coordinate differs')
    expected={ds:v['counts']['planned'] for ds,v in json.loads((run/'summary.json').read_text())['datasets'].items()}
    expected['visual_cognitive_motivation']=checks['new_fits']
    cfg=dict(schema='cross_dataset_amplitude_v1',run=str(run),visual_run=str(visual),audit=str(a.audit.resolve()),export=str(out),subjects=sorted(co),expected_pair_windows=expected,
        support='frozen prepared 0.01-0.2 Hz Hb; remove last bin; preserve all inputs independent of fit outcome',
        normalization='unchanged subject common Hb factor; Visual uses verified v6 usable normalization already adopted by v4',
        boundary='retained public inputs and completed v3 audits only; no raw/protected reads, no v4 fit reads, no refits')
    if a.stage=='plan':
        out.mkdir();(out/'subjects').mkdir();write_json(out/'analysis_config.json',cfg)
        write_json(out/'source_status_at_plan.json',dict(visual_execution=json.loads((visual/'execution.json').read_text()),visual_progress=json.loads((visual/'progress.json').read_text())))
        return
    if json.loads((out/'analysis_config.json').read_text())!=cfg:raise ValueError('Frozen analysis changed')
    if a.stage=='report':amplitude_report(cfg);return
    if not os.environ.get('SSM_SYSTEMD_UNIT'):raise ValueError('Use a durable supervisor')
    selected=cfg['subjects'] if a.stage=='run' else [next(k for k in cfg['subjects'] if co[k]['dataset']==ds) for ds in sorted(expected)]
    start=time.monotonic();done=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(amplitude_subject,[(cfg,k,co[k]) for k in selected]):
            done.append(r);print(json.dumps(r),flush=True);write_json(out/(a.stage+'_progress.json'),dict(completed=len(done),total=len(selected)))
    write_json(out/(a.stage+'_execution.json'),dict(seconds=time.monotonic()-start,subjects=len(done),workers=a.workers,supervisor=os.environ['SSM_SYSTEMD_UNIT'],peak_worker_mib=max(r['peak_rss_mib'] for r in done)))
    if a.stage=='run':amplitude_summary(cfg)


def visual_cache_report(cache_root,export):
    """Audit the completed compact measurement/downstream caches, without fits."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from itertools import combinations
    root=Path(cache_root).resolve();out=Path(export).resolve()
    if out.parent!=root or out.exists():raise ValueError('Fresh direct cache report child required')
    verify=json.loads((root/'verification.json').read_text())
    if verify['status']!='passed' or verify['errors']:raise ValueError('Completed verified cache required')
    plan=json.loads((root/'input_plan.json').read_text());manifest=json.loads((root/'cache_manifest.json').read_text())
    if plan['schema']!='physiology_downstream_input_v2' or manifest['execution']!='completed':raise ValueError('Wrong input generation')
    lower=Path(plan['source_cache']);lm=json.loads((lower/'cache_manifest.json').read_text())
    rows={m['join_key']:m for m in manifest['records'] if m['dataset_id']=='visual_cognitive_motivation'}
    physical={m['join_key']:m for m in lm['records'] if m['dataset_id']=='visual_cognitive_motivation'}
    samples=[s for s in plan['samples'] if s['dataset']=='visual_cognitive_motivation'];admitted={s['join_key'] for s in samples}
    norms={m['canonical_subject_id']:json.loads(Path(m['fnirs_normalization']['source']).read_text()) for m in rows.values()}
    subjects=sorted(norms);channels=[f'Probe{p}:CH{i}' for p in [1,2] for i in range(1,25)]
    records=[];frequency=[];subject_sets={};distributions=[];within=[];moments=[];statistics=[];spectral=[]
    for key,m in rows.items():
        probe=m['base_record_id'].rsplit('_',1)[-1];names=[n.rsplit('_',1)[0] for n in m['measurement']['fnirs_channel_names'][::2]]
        records.append(dict(subject=m['canonical_subject_id'],record=m['base_record_id'],probe=probe,
            admitted_samples=key in admitted,locations=len(names),eeg_channels=len(m['measurement']['eeg_channel_names']),
            usable_channels=names,excluded=m['fnirs_selection']['device_excluded_channels']))
    for sub in subjects:
        r=[r for r in records if r['subject']==sub and r['admitted_samples']]
        stable=set();possible=set();freq={}
        for probe in ['Probe1','Probe2']:
            own=[x for x in r if x['probe']==probe];sets=[{probe+':'+c for c in x['usable_channels']} for x in own]
            stable |= set.intersection(*sets);possible |= set.union(*sets)
            for ch in [c for c in channels if c.startswith(probe+':')]:
                freq[ch]=np.mean([ch in s for s in sets]);frequency.append(dict(subject=sub,channel=ch,usable_fraction=freq[ch],records=len(sets)))
            for (a,ra),(b,rb) in combinations(list(zip(sets,own)),2):
                within.append(dict(subject=sub,probe=probe,first=ra['record'],second=rb['record'],jaccard=len(a&b)/len(a|b),changed_channels=sorted(a^b)))
        subject_sets[sub]=stable
        distributions.append(dict(subject=sub,records=len(r),stable_count=len(stable),ever_usable_count=len(possible),
            stable_channels=sorted(stable),variable_channels=sorted(possible-stable),never_usable_channels=sorted(set(channels)-possible)))
        n=norms[sub];old=n['moments']['old'];new=n['moments']['usable']
        moments.append(dict(subject=sub,old_factor=old['factor'],new_factor=n['factor'],multiplier=n['factor']/old['factor'],
            old_usable_common_sd=new['common_sd']*old['factor'],new_usable_common_sd=new['common_sd']*n['factor'],
            old_excluded_variance_fraction=n['moments']['excluded']['original_variance_fraction']))
    for key,m in rows.items():
        if key not in admitted:continue
        own=[s for s in samples if s['join_key']==key];a=np.load(Path(m['record_npz'])/'fnirs.npy',mmap_mode='r')
        oldfactor=norms[m['canonical_subject_id']]['moments']['old']['factor'];factor=norms[m['canonical_subject_id']]['factor']
        b=np.load(Path(physical[key]['record_npz'])/'fnirs.npy',mmap_mode='r')
        legacy=np.load(Path(physical[key]['migration_source']['measurement'])/'fnirs.npy',mmap_mode='r')
        spectral_ids={own[j]['id'] for j in sorted({0,len(own)//2,len(own)-1})}
        for sample in own:
            start=round(sample['hb_start_s']*10);end=start+round(sample['duration_s']*10)
            x=np.asarray(b[start:end]).reshape(-1,b.shape[1]//2,2);z=np.asarray(a[start:end]).reshape(x.shape)
            np.testing.assert_allclose(z,x*factor,rtol=1e-12,atol=1e-12)
            for j,c in enumerate(['HbO','HbR']):
                sd=x[:,:,j].std(0);med=float(np.median(sd))
                statistics.append(dict(subject=m['canonical_subject_id'],record=m['base_record_id'],component=c,sample=sample['id'],
                    usable_locations=len(sd),physical_sd_median=med,old_formula_sd_median=med*oldfactor,
                    actual_new_sd_median=float(np.median(z[:,:,j].std(0))),physical_sd_p90=float(np.quantile(sd,.9))))
            if sample['id'] in spectral_ids:
                old_all=np.asarray(legacy[start:end]).reshape(len(x),24,2)
                for condition,values in [('old_all_channels',old_all),('usable_channels',x)]:
                    for j,c in enumerate(['HbO','HbR']):
                        feat=[signal_features(values[:,ch,j]) for ch in range(values.shape[1])]
                        spectral.append(dict(subject=m['canonical_subject_id'],record=m['base_record_id'],sample=sample['id'],
                            condition=condition,component=c,channels=len(feat),
                            **{metric:float(np.nanmedian([f[metric] for f in feat])) for metric in ['sd','mobility','centroid','fast_fraction','kurtosis']}))
    cross=[]
    for a,b in combinations(subjects,2):
        sa,sb=subject_sets[a],subject_sets[b];cross.append(dict(first=a,second=b,jaccard=len(sa&sb)/len(sa|sb),shared=len(sa&sb),different=sorted(sa^sb)))
    common=set.intersection(*subject_sets.values());ever=set.union(*(set(x['stable_channels'])|set(x['variable_channels']) for x in distributions))
    sf=pd.DataFrame(statistics);subject_stats=sf.groupby(['subject','component','record']).median(numeric_only=True).groupby(['subject','component']).mean().reset_index()
    out.mkdir();write_json(out/'record_channels.json',records);write_json(out/'subject_channel_sets.json',distributions)
    pd.DataFrame(frequency).to_csv(out/'subject_channel_frequency.csv',index=False)
    write_json(out/'within_subject_changes.json',within);write_json(out/'between_subject_jaccard.json',cross)
    pd.DataFrame(moments).to_csv(out/'normalization.csv',index=False);sf.to_csv(out/'sample_statistics.csv',index=False);subject_stats.to_csv(out/'subject_statistics.csv',index=False)
    spec=pd.DataFrame(spectral);spec.to_csv(out/'spectral_panel.csv',index=False)
    specsub=spec.groupby(['subject','component','condition','record']).mean(numeric_only=True).groupby(['subject','component','condition']).mean().reset_index()
    specsub.to_csv(out/'subject_spectral_statistics.csv',index=False)
    specsummary=specsub.groupby(['component','condition']).mean(numeric_only=True).reset_index();specsummary.to_csv(out/'dataset_spectral_statistics.csv',index=False)
    freq=pd.DataFrame(frequency).pivot(index='subject',columns='channel',values='usable_fraction').reindex(index=subjects,columns=channels)
    fig,ax=plt.subplots(figsize=(19,7),layout='constrained')
    im=ax.imshow(freq.to_numpy(),aspect='auto',vmin=0,vmax=1,cmap='viridis')
    ax.set(yticks=range(len(subjects)),yticklabels=subjects,xticks=range(48),xticklabels=[c.split(':')[1] for c in channels],
        title='Usable channel frequency within subject | left: Probe1; right: Probe2')
    ax.tick_params(axis='x',rotation=90);ax.axvline(23.5,color='white',lw=2);fig.colorbar(im,ax=ax,label='Fraction of admitted records retaining this channel')
    fig.savefig(out/'channel_availability.png',dpi=220);plt.close(fig)
    matrix=np.eye(len(subjects))
    for r in cross:
        i=subjects.index(r['first']);j=subjects.index(r['second']);matrix[i,j]=matrix[j,i]=r['jaccard']
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained');im=axes[0].imshow(matrix,vmin=0,vmax=1,cmap='viridis')
    axes[0].set(xticks=range(16),xticklabels=subjects,yticks=range(16),yticklabels=subjects,title='Between-subject stable-channel Jaccard');axes[0].tick_params(axis='x',rotation=90)
    fig.colorbar(im,ax=axes[0],label='Intersection / union')
    mf=pd.DataFrame(moments);axes[1].bar(mf.subject,mf.multiplier,color='#227f79');axes[1].tick_params(axis='x',rotation=60);axes[1].set(ylabel='New usable-only factor / old all-channel factor',title='Normalization correction, 16 subjects')
    fig.savefig(out/'consistency_and_scaling.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,c in zip(axes,['HbO','HbR']):
        for r in subject_stats[subject_stats.component==c].itertuples():ax.plot([0,1],[r.old_formula_sd_median,r.actual_new_sd_median],'-o',alpha=.7)
        ax.set(xticks=[0,1],xticklabels=['Old shared scale (replay)','New downstream cache'],ylabel='Median usable-channel window SD',yscale='log',title=c+' | each line = one subject');ax.grid(alpha=.2)
    fig.savefig(out/'actual_sample_statistics.png',dpi=220);plt.close(fig)
    summary=dict(status='completed',measurement_version='physiology_semantic_clean_v6',downstream_version='physiology_downstream_input_v2',
        all_records=verify['checked_records'],all_samples=verify['checked_samples'],visual_cached_records=len(records),visual_sample_records=len(admitted),
        visual_samples=len(samples),subjects=len(subjects),universal_stable_channels=sorted(common),universal_stable_count=len(common),
        never_usable_channels=sorted(set(channels)-ever),stable_count_range=[min(x['stable_count'] for x in distributions),max(x['stable_count'] for x in distributions)],
        subjects_with_within_record_changes=sorted({r['subject'] for r in within if r['jaccard']<1}),
        subjects_with_multiple_records=sorted({r['subject'] for r in within}),
        within_jaccard_min=min(r['jaccard'] for r in within),between_jaccard_min=min(r['jaccard'] for r in cross),
        between_jaccard_median=float(np.median([r['jaccard'] for r in cross])),between_jaccard_max=max(r['jaccard'] for r in cross),
        multiplier_min=float(mf.multiplier.min()),multiplier_median=float(mf.multiplier.median()),multiplier_max=float(mf.multiplier.max()),
        excluded_channels_in_new_arrays=0,new_SSM_fits=0,
        improvement='device-excluded channels physically removed; calibration no longer includes excluded variance; remaining waveform values preserved up to explicit subject scale',
        interpretation='normalization target is by construction, not a denoising or SSM-quality score; frequencies and shape unchanged on common channels')
    write_json(out/'summary.json',summary)
    def tab(d):return d.to_html(index=False,float_format=lambda v:f'{v:.5g}')
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Visual输入缓存修复与统计评估</title><style>body{font-family:sans-serif;max-width:1500px;margin:auto;padding:28px;line-height:1.8}img{width:100%}table{border-collapse:collapse;font-size:13px;width:100%}td,th{border:1px solid #ccc;padding:6px}</style><h1>Visual底层测量v6与下游输入v2：完成验证及统计评估</h1>
<p>底层测量缓存物理移除Exception Ch=1及原支持无效的Hb位置，保留原CH编号、时间轴、相对测量单位和既有滤波结果。Visual EEG直接复用下游v1已验证的31通道数据，修复底层旧30通道身份。下游新缓存的Visual Hb明确为测量值乘被试共同因子；连续数组不新增中心化。原始测量值保留在底层v6，归一化因子与逆因子由底层normalization/&lt;subject&gt;.json唯一拥有。另三个数据集及全部EEG信号复用原数组，REFED不作新处理。</p>
<p>每被试因子沿用0.025/sqrt((Var(HbO)+Var(HbR))/2)，在此前已纳入的90秒窗口、前5秒参考和原重叠权重上，只改变可用通道集合。验证/测试被试仍按既有公开无标签个体适配范围计算，不是零样本归一化。旧SSM拟合没有读取或修改。</p>'''
    body+=f'<p>四库{summary["all_records"]}条记录、{summary["all_samples"]}个样本全部通过加载验证。Visual保留{len(records)}条缓存记录，其中{len(admitted)}条具有获准样本，合计{len(samples)}个样本；S06 Part1既有排除保持。分布统计使用这{len(admitted)}条实际有样本的记录，避免将仅保留的排除记录当作可用实验。</p>'
    body+='<h2>统计特征改善的含义</h2><p>已实现的改善是输入有效性和幅度参照：排除通道不再作为信号列、校准样本或监督输入参与处理；可用列与原测量逐值一致，下游数值与其乘新因子逐值一致。旧尺度能从同一测量窗口精确重放。归一化不会改变原可用通道的波形时序、相关性或相对谱形；绝对SD乘因子，功率乘因子平方。共同校准SD为0.025由定义保证，不能当作降噪成绩。</p>'
    body+=f'<p>16人的新/旧因子为{mf.multiplier.min():.2f}–{mf.multiplier.max():.2f}倍，中位数{mf.multiplier.median():.2f}倍。下表使用原全通道共同尺度作重放比较；旧下游v1本身存储的是未归一化测量，不能将其误称为已应用该尺度。</p>'+tab(mf)
    body+='<img src="actual_sample_statistics.png"><p>每个点对是一名被试。指标先对每个30秒下游样本的可用通道取去均值SD中位数，再记录内取样本中位数，最后记录等权汇总到被试；同一可用支持比较旧公式重放与新实际缓存。由于校准使用90秒窗口，这个30秒样本统计不必等于0.025。</p>'+tab(subject_stats)
    body+='<h2>通道集合变化后的频谱与形态统计</h2><p>每记录固定取样本列表首/中/末三个窗口，在同一30秒时间支持上比较旧全部24位置与新可用位置。各窗口先取通道特征中位数，再窗口/记录/被试逐层等权。SD保留同一Visual相对测量单位；mobility为导数SD/信号SD（s⁻¹）；centroid为全部正频率谱重心（Hz）；fast_fraction为&gt;0.2 Hz功率比例；kurtosis为Fisher超额峰度。Hann谱分辨率约0.0333 Hz，仅作处理后波形描述，不解释不足3周期的低频为独立生理节律。集合变化与同一通道重标度是两种不同比较；下表并非滤波前后的降噪实验。</p>'+tab(specsummary)
    body+='<h2>跨被试及同一被试内的可用通道分布</h2><img src="channel_availability.png"><p>每行一名被试，每列一个明确的Probe:CH位置；颜色是该被试该Probe在实际有样本记录中的可用比例。0为均不可用，1为均可用，中间值说明Part间发生变化。两侧编号独立，不能将同编号当成同一位置。</p>'
    body+=f'<p>所有被试、所有纳入记录共同保留的严格公共集合有{len(common)}个位置（{2*len(common)}条Hb序列）：'+', '.join(sorted(common))+'. 每名被试自身所有记录的稳定集合为'+str(summary['stable_count_range'])+'个位置。</p>'
    body+=f'<p>{len(summary["subjects_with_multiple_records"])}名被试有可作同侧跨记录比较的数据，其中{len(summary["subjects_with_within_record_changes"])}名发生通道变化：'+', '.join(summary['subjects_with_within_record_changes'])+f'。同侧记录Jaccard最低{summary["within_jaccard_min"]:.3f}。只有一个记录的被试不提供跨记录稳定性证据。</p>'
    body+='<img src="consistency_and_scaling.png"><p>左图比较每名被试的稳定可用集合：Jaccard=交集大小/并集大小。对角线为自身，左右Probe身份保留；右图展示同一批被试的尺度修正倍数。</p>'
    body+=f'<p>120对被试的Jaccard范围{summary["between_jaccard_min"]:.3f}–{summary["between_jaccard_max"]:.3f}，中位数{summary["between_jaccard_median"]:.3f}。分布存在公共核心，也存在被试/记录特异的可用位置，不能按通道序号截断或把变长数组的第i列直接视为同一空间位置。</p>'
    body+=tab(pd.DataFrame([{k:r[k] for k in ['subject','records','stable_count','ever_usable_count','variable_channels']} for r in distributions]))
    body+='<h2>供评估的建议与边界</h2><p>后续模型应按原始Probe:CH身份和返回几何处理变长可用通道，或在显式公共通道面板上做公平比较。公共集合能保证对齐，但会损失部分有效位置；保留全部可用位置则需要按身份及掩码对齐，不能填零后让缺失列参与统计。本次尚未建立新的SSM求解缓存，因此未把输入幅度修正解释成S/O抵消已解决。</p><p><a href="../verification.json">下游全量验证</a> · <a href="summary.json">分析摘要</a> · <a href="normalization.csv">逐被试尺度</a> · <a href="subject_channel_sets.json">完整通道集合</a> · <a href="within_subject_changes.json">同被试变化</a> · <a href="between_subject_jaccard.json">被试间差异</a> · <a href="subject_statistics.csv">被试统计</a> · <a href="sample_statistics.csv">完整样本统计</a></p></html>'
    (out/'report.html').write_text(body,encoding='utf-8')


def cache_audit_components(x,s,sd):
    """Score referenced observations; centered SD ratios are not energy fractions."""
    x=np.asarray(x,float);s=np.asarray(s,float);o=x-s
    if x.shape!=s.shape or x.ndim!=2 or not np.isfinite([x,s]).all():raise ValueError('Finite matching time/component arrays required')
    xc=x-x.mean(0);sc=s-s.mean(0);oc=o-o.mean(0)
    xs=np.sqrt(np.mean(xc*xc,0));ss=np.sqrt(np.mean(sc*sc,0));osd=np.sqrt(np.mean(oc*oc,0))
    xs=np.where(np.ptp(x,axis=0)>0,xs,0.);ss=np.where(np.ptp(s,axis=0)>0,ss,0.);osd=np.where(np.ptp(o,axis=0)>0,osd,0.)
    def div(a,b):return np.divide(a,b,out=np.full_like(np.asarray(a,float),np.nan),where=np.asarray(b)>0)
    r=div(np.mean(xc*sc,0),xs*ss);so=div(np.mean(sc*oc,0),ss*osd)
    ar=div(ss,xs);br=div(osd,xs)
    rows=[]
    for j,c in enumerate(['EEG','HbO','HbR']):
        defined=bool(xs[j]>0 and ss[j]>0)
        rows.append(dict(component=c,defined=defined,x_sd=xs[j],x_sd_subject=xs[j]/sd[j],s_sd_ratio=ar[j],o_sd_ratio=br[j],correlation_sx=r[j],correlation_so=so[j],
            variance_sum_ratio=ar[j]**2+br[j]**2,local_nrmse=div(np.sqrt(np.mean((s-x)**2,0)),xs)[j],
            subject_nrmse=float(np.sqrt(np.mean((s[:,j]-x[:,j])**2))/sd[j]),bias_subject=float((s[:,j]-x[:,j]).mean()/sd[j]),
            severe_cancellation=float(ar[j]>=2 and br[j]>=1 and so[j]<=-.8) if defined else np.nan,
            faithful_small_residual=float(r[j]>=.8 and br[j]<=.5) if defined else np.nan,
            negative_correlation=float(r[j]<=0) if defined else np.nan,
            residual_ge_signal=float(br[j]>=1) if defined else np.nan))
    return rows


def cache_audit_overlap(previous,current,rate=10.):
    """Compare shared physical times after restoring the recorded window reference."""
    a,b=previous,current
    left=max(a['start'],b['start']);right=min(a['start']+len(a['x'])/rate,b['start']+len(b['x'])/rate)
    n=round((right-left)*rate)
    if n<=0:return None
    ia=round((left-a['start'])*rate);ib=round((left-b['start'])*rate)
    xa=a['x'][ia:ia+n]+a['ref'];xb=b['x'][ib:ib+n]+b['ref']
    sa=a['s'][ia:ia+n]+a['ref'];sb=b['s'][ib:ib+n]+b['ref']
    return dict(seconds=n/rate,target_max_abs=float(np.max(abs(xa-xb))),
        subject_nrmse=np.sqrt(np.mean((sa-sb)**2,0))/b['sd'],
        local_nrmse=np.divide(np.sqrt(np.mean((sa-sb)**2,0)),xb.std(0),out=np.full(xb.shape[1],np.nan),where=xb.std(0)>1e-12),
        state_rms=np.sqrt(np.mean((a['state'][ia:ia+n]-b['state'][ib:ib+n])**2,0)))


def cache_audit_init(input_root):
    global _CACHE_AUDIT_INPUT
    from src.data.downstream_input_cache import DownstreamInputDataset
    _CACHE_AUDIT_INPUT=DownstreamInputDataset(input_root)


def cache_audit_record(payload):
    cfg,r,indices,coordinate=payload;began=time.monotonic();out=Path(cfg['output'])/'records'/r['dataset']/r['subject']/r['record'];out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists():return json.loads((out/'status.json').read_text())
    dataset=_CACHE_AUDIT_INPUT;record=dataset.records[r['join_key']];m=record.manifest;ds=r['dataset'];ident=dict(dataset=ds,subject=r['subject'],record=r['record'],split=r['split'])
    arrays=dataset.index.load_record_arrays(record);checks=[];sample_rows=[];input_stats=[];spectra=[];tasks=[];components=[];overlaps=[];old_components=[];transitions=[]
    for modality in ['eeg','fnirs']:
        a=arrays[modality];support=arrays[modality+'_supported_channels'];names=m['measurement'][modality+'_channel_names']
        if a.dtype!=np.float64 or a.shape[1]!=len(names) or len(set(names))!=len(names) or not np.isfinite(a[:,support]).all():raise ValueError('Invalid supported record array')
    low=Path(cfg['lower_records'][r['join_key']]);lm=json.loads((low/'record.json').read_text());h=np.load(low/lm['arrays']['fnirs'],mmap_mode='r')
    factor=1.
    if m.get('fnirs_normalization'):factor=json.loads(Path(m['fnirs_normalization']['source']).read_text())['factor']
    np.testing.assert_allclose(arrays['fnirs'],h*factor,rtol=1e-12,atol=1e-12,equal_nan=True)
    nominal=len(m['measurement']['fnirs_channel_names'])//2
    available=int(np.asarray(arrays['fnirs_supported_channels']).reshape(-1,2).all(1).sum())
    if ds=='visual_cognitive_motivation':
        if nominal!=available or nominal!=len(r['pairs']):raise ValueError('Visual compact support mismatch')
        if any(p['name'] in m['fnirs_selection']['device_excluded_channels'] for p in r['pairs']):raise ValueError('Excluded Visual pair present')
    spectral_eligible=[i for i in indices if dataset.samples[i]['supported_seconds']>=30]
    spectral_samples={spectral_eligible[i] for i in sorted({0,len(spectral_eligible)//2,len(spectral_eligible)-1})} if spectral_eligible else set()
    for index in indices:
        sample=dataset[index];row=dict(ident,sample_id=sample['id'],dependency_id=sample['dependency_id'],partial=sample['partial'],duration_s=dataset.samples[index]['duration_s'],supported_seconds=dataset.samples[index]['supported_seconds'])
        for modality in ['eeg','fnirs']:
            v=sample[modality];mask=sample['valid_mask'][modality];g=sample['channel_geometry'][modality]
            if v.shape!=mask.shape or not np.isfinite(v[mask]).all() or np.any(v[~mask]!=0):raise ValueError('Consumer validity/padding failure')
            if len(g)!=len(v) or any(not p['position_available'] for p in g):raise ValueError('Missing consumer geometry')
            row[modality+'_valid_fraction']=float(mask.mean());row[modality+'_supported_channels']=int(mask.any(1).sum())
            row[modality+'_rounding_error_s']=sample['alignment'][modality]['rounding_error_s']
            rate=sample['sample_rate_hz'][modality]
            for component,take in ([('EEG',np.arange(len(v)))] if modality=='eeg' else [('HbO',np.arange(0,len(v),2)),('HbR',np.arange(1,len(v),2))]):
                channel=[];sf=[]
                for ch in take:
                    x=v[ch,mask[ch]]
                    if len(x)<2:continue
                    scale=1. if modality=='eeg' else coordinate['hb_factor']/factor
                    channel.append([x.std(),np.ptp(x),np.sqrt(np.mean(x*x)),x.std()*scale,float(x.std()<=1e-12)])
                    if index in spectral_samples:
                        if modality=='fnirs':sf.append(signal_features(x,rate))
                        else:
                            f,p=periodogram(x,fs=rate,window='hann',detrend='constant');den=p[(f>=1)&(f<=45)].sum()
                            sf.append({name:float(p[(f>=lo)&(f<hi)].sum()/den) if den>1e-24 else np.nan for name,lo,hi in [('delta',1,4),('theta',4,8),('alpha',8,13),('beta',13,30),('gamma',30,45)]})
                z=np.array(channel)
                if len(z):input_stats.append(dict(ident,sample_id=sample['id'],component=component,unit=sample['unit'][modality],duration_s=dataset.samples[index]['duration_s'],supported_seconds=dataset.samples[index]['supported_seconds'],channels=len(z),
                    sd=float(np.median(z[:,0])),ptp=float(np.median(z[:,1])),rms=float(np.median(z[:,2])),ssm_scaled_sd=float(np.median(z[:,3])),constant_fraction=float(z[:,4].mean())))
                if sf:
                    spectrum=pd.DataFrame(sf).mean().to_dict();spectra.append(dict(ident,source='input',component=component,sample_id=sample['id'],**spectrum))
        sample_rows.append(row)
    root=Path(cfg['ssm']);previous={};qualified={};expected=[];oldroot=Path(cfg['baseline']);parameter_bounds={'tau':(.25,12.),'beta':(.1,10.)}
    spectral_windows=set(anchors([w['key'] for w in r['cache_windows']]));spectral_pairs=set(anchors([p['native_index'] for p in r['pairs']]))
    for w in r['cache_windows']:
        for pair in r['pairs']:
            name=w['key']+f"__p{pair['native_index']:03d}";expected.append(name+'.json');path=root/'tasks'/r['cache_key']/(name+'.json')
            d=json.loads(path.read_text());it=d['identity'];qualified[(w['key'],pair['native_index'])]=d['qualified']
            expected_schema='shared_driver_tak_subject_cache_v4' if ds=='visual_cognitive_motivation' else 'shared_driver_tak_subject_cache_v3'
            if d['schema']!=expected_schema or it['join_key']!=r['join_key'] or it['pair']['id']!=pair['id'] or it['window']!=w:raise ValueError('Task identity mismatch')
            identity=dict(ident,window=w['key'],pair=pair['id'],native_index=pair['native_index'],duration_s=w['duration_s'])
            task=dict(identity,status=d['status'],qualified=d['qualified'],route=d.get('initialization_route'),array_present=path.with_suffix('.npz').exists())
            old=None
            if ds=='visual_cognitive_motivation':
                old=json.loads((oldroot/'tasks'/r['cache_key']/(name+'.json')).read_text());transitions.append(dict(identity,old_qualified=old['qualified'],new_qualified=d['qualified']))
            if d.get('fits',{}).get('tak_tau_beta'):
                fit=d['fits']['tak_tau_beta'];task.update(tau=fit.get('tau'),beta=fit.get('beta'),converged=fit['solver'].get('converged'))
                for k,(lo,hi) in parameter_bounds.items():
                    v=fit.get(k);v=np.nan if v is None else v;task[k+'_at_bound']=bool(np.isclose(v,lo,rtol=1e-5) or np.isclose(v,hi,rtol=1e-5))
            if not d['qualified']:
                tasks.append(task);previous.pop(pair['native_index'],None);continue
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:
                n=round(w['duration_s']*10);use=np.all(a['fit_mask']&a['observation_valid']&a['prediction_valid'],axis=1)
                if not np.array_equal(np.flatnonzero(use),np.arange(n-1)):raise ValueError('Unexpected fit support')
                x=a['target'][use];s=a['prediction'][use];o=a['residual'][use];sd=a['sd_subject'];scale=a['scale'];state=a['states'][use];reference=a['reference_unscaled']
                if not np.isfinite(state).all() or not np.array_equal(state[:,3],state[:,4]) or np.any(state[:,2:]<=0):raise ValueError('State validity failure')
                np.testing.assert_allclose(x,s+o,rtol=1e-12,atol=1e-12);np.testing.assert_array_equal(state[:,0],a['driver'][use])
                np.testing.assert_allclose(scale,[coordinate['eeg_factor'],coordinate['hb_factor'],coordinate['hb_factor']],rtol=1e-12)
                if not np.array_equal(a['time_s'],np.arange(n)/10.):raise ValueError('Clock mismatch')
                newrows=cache_audit_components(x,s,sd)
                for c in newrows:components.append(dict(identity,**c))
                for j,nm,rest in [(0,'driver',0),(1,'vasodilatory',0),(2,'flow',1),(3,'volume',1),(5,'deoxy',1)]:
                    z=state[:,j];task.update({nm+'_mean':float(z.mean()),nm+'_sd':float(z.std()),nm+'_min':float(z.min()),nm+'_max':float(z.max()),nm+'_rms_rest':float(np.sqrt(np.mean((z-rest)**2)))})
                def corr(a,b):return float(np.corrcoef(a,b)[0,1]) if min(a.std(),b.std())>1e-12 else np.nan
                task.update(x_HbO_HbR_r=corr(x[:,1],x[:,2]),s_HbO_HbR_r=corr(s[:,1],s[:,2]),o_HbO_HbR_r=corr(o[:,1],o[:,2]),flow_volume_r=corr(state[:,2],state[:,3]))
                current=dict(start=w['hb_start_s'],x=x[:,1:],s=s[:,1:],ref=reference[1:],sd=sd[1:],state=state[:,[0,1,2,3,5]])
                if pair['native_index'] in previous:
                    ov=cache_audit_overlap(previous[pair['native_index']],current)
                    if ov is not None:
                        if ov['target_max_abs']>1e-9:raise ValueError('Overlapping Hb targets disagree after reference restoration')
                        overlaps.append(dict(identity,seconds=ov['seconds'],target_max_abs=ov['target_max_abs'],HbO_subject_nrmse=ov['subject_nrmse'][0],HbR_subject_nrmse=ov['subject_nrmse'][1],HbO_local_nrmse=ov['local_nrmse'][0],HbR_local_nrmse=ov['local_nrmse'][1],**dict(zip(['driver_rms','vasodilatory_rms','flow_rms','volume_rms','deoxy_rms'],ov['state_rms']))))
                previous[pair['native_index']]=current
                if w['key'] in spectral_windows and pair['native_index'] in spectral_pairs:
                    for j,c in enumerate(['EEG','HbO','HbR']):
                        for name,z in [('X',x),('S',s),('O',o)]:spectra.append(dict(identity,source='SSM_'+name,component=c,**signal_features(z[:,j])))
                if old is not None and old['qualified']:
                    oldfit=old['fits']['tak_tau_beta']
                    for wf in oldfit['waveform']:
                        if wf['pearson_r'] is None:continue
                        g=residual_geometry(wf['centered_amplitude_ratio'],wf['pearson_r']);a,b,rr=g['s_sd_ratio'],g['o_sd_ratio'],g['correlation_so']
                        old_components.append(dict(identity,component=wf['component'],**g,local_nrmse=wf['local_centered_nrmse'],severe_cancellation=float(a>=2 and b>=1 and rr<=-.8),faithful_small_residual=float(wf['pearson_r']>=.8 and b<=.5)))
            tasks.append(task)
    folder=root/'tasks'/r['cache_key'];actual={p.name for p in folder.glob('*.json')}
    if actual!=set(expected):raise ValueError('Task census mismatch including excluded channels')
    if ds!='visual_cognitive_motivation' and expected and folder.resolve()!=(oldroot/'tasks'/r['cache_key']).resolve():raise ValueError('Unchanged dataset tasks not exact reused source')
    joins={j['sample_id']:j for j in cfg['joins'].get(r['cache_key'],[])}
    for row in sample_rows:
        join=joins[row['sample_id']];row['teacher_join']=join['status'];row['teacher_pairs']=len(r['pairs']);row['teacher_qualified_pairs']=0
        if join['status']=='available':
            row['teacher_qualified_pairs']=sum(qualified[(join['window_key'],p['native_index'])] for p in r['pairs'])
            row['teacher_fraction']=row['teacher_qualified_pairs']/len(r['pairs'])
            row['teacher_hb_offset_s']=join['hb_offset_s'];row['teacher_eeg_offset_s']=join['eeg_offset_s']
            row['teacher_last_bin_in_sample']=join['hb_offset_s']+join['supported_seconds']>join['fit_duration_s']-.1+1e-9
        else:row['teacher_fraction']=0.
    for name,rows in [('samples',sample_rows),('input',input_stats),('tasks',tasks),('components',components),('spectra',spectra),('overlap',overlaps),('baseline',old_components),('transitions',transitions)]:
        if rows:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    result=dict(ident,status='completed',samples=len(indices),tasks=len(tasks),qualified=sum(t['qualified'] for t in tasks),failed=sum(not t['qualified'] for t in tasks),
        EEG_channels=arrays['eeg'].shape[1],EEG_supported_channels=int(arrays['eeg_supported_channels'].sum()),Hb_positions=nominal,Hb_supported_positions=available,
        EEG_unit=m['measurement']['eeg_preprocessing_state']['canonical_unit'],Hb_unit=m['measurement']['fnirs_preprocessing_state']['canonical_unit'],
        hb_factor_in_input=factor,ssm_hb_factor=coordinate['hb_factor'],SSM_scoring_common_reference='retained_v3_training_reference',seconds=time.monotonic()-began,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(out/'status.json',result);return result


def cache_audit_plan(cfg,root,out):
    import yaml
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    from collections import Counter,defaultdict
    for key in ['measurement','input','ssm','baseline']:
        v=json.loads((Path(cfg[key])/'verification.json').read_text())
        if v['status']!='passed' or v.get('issues') or v.get('errors'):raise ValueError('Verified cache required: '+key)
    ssm=json.loads((Path(cfg['ssm'])/'plan.json').read_text());ip=json.loads((Path(cfg['input'])/'input_plan.json').read_text())
    if ssm['schema']!='shared_driver_tak_subject_cache_v4' or ip['schema']!='physiology_downstream_input_v2' or ssm['subject_partitions']!=ip['subject_partitions']:raise ValueError('Version/split mismatch')
    ids=[s['id'] for s in ip['samples']]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate sample identity')
    deps={};sample_indices=defaultdict(list)
    for i,s in enumerate(ip['samples']):
        sample_indices[s['join_key']].append(i)
        key=(s['dataset'],s['subject']);old=deps.setdefault(key,s['split'])
        if old!=s['split']:raise ValueError('Subject leakage')
    lower=CleanPhysiologyCacheIndex(cfg['measurement'])
    if {r.join_key for r in lower.records}!={r['join_key'] for r in ssm['records']}:raise ValueError('Record census mismatch')
    joins=defaultdict(list)
    for j in ssm['downstream_joins']:joins[j['record_key']].append(j)
    if set(ids)!={j['sample_id'] for j in ssm['downstream_joins']}:raise ValueError('Sample join identity mismatch')
    plan=dict(records=ssm['records'],sample_indices=dict(sample_indices),lower_records={r.join_key:str(r.npz_path) for r in lower.records},joins=dict(joins),
        coordinate=str(Path(cfg['ssm'])/'coordinates.json'),sample_count=len(ids),planned_tasks=ssm['planned_pair_windows'],
        split_counts={ds:{s:len(v) for s,v in parts.items()} for ds,parts in ip['subject_partitions'].items()},dependency_counts=dict(Counter(s['dataset'] for s in {s['dependency_id']:s for s in ip['samples']}.values())))
    out.mkdir(parents=True,exist_ok=False);write_json(out/'analysis_config.json',cfg);write_json(out/'plan.json',plan)
    return dict(records=len(plan['records']),samples=len(ids),tasks=plan['planned_tasks'],split_counts=plan['split_counts'])


def cache_audit_aggregate(frame,groups,metrics):
    # Per-record window/pair means, then equal records within each subject.
    record=frame.groupby(['dataset','subject','record']+groups,dropna=False)[metrics].mean()
    subject=record.groupby(['dataset','subject']+groups,dropna=False).mean().reset_index()
    return subject


def cache_audit_summary(cfg,plan):
    from collections import Counter
    out=Path(cfg['output']);statuses=[];frames={}
    for r in plan['records']:
        p=out/'records'/r['dataset']/r['subject']/r['record'];statuses.append(json.loads((p/'status.json').read_text()))
        for name in ['samples','input','tasks','components','spectra','overlap','baseline','transitions']:
            if (p/(name+'.csv')).exists():frames.setdefault(name,[]).append(pd.read_csv(p/(name+'.csv'),dtype={'subject':str}))
    frames={k:pd.concat(v,ignore_index=True) for k,v in frames.items()}
    for k,v in frames.items():v.to_csv(out/(k+'.csv'),index=False)
    status=pd.DataFrame(statuses);status.to_csv(out/'records.csv',index=False)
    expected=json.loads((Path(cfg['ssm'])/'summary.json').read_text())
    counts=dict(records=len(status),samples=len(frames['samples']),planned=len(frames['tasks']),qualified=int(frames['tasks'].qualified.sum()))
    counts['failed']=counts['planned']-counts['qualified']
    if counts['samples']!=plan['sample_count'] or any(counts[k]!=expected['counts'][k] for k in ['planned','qualified','failed']):raise ValueError('Audit census differs from owning completed cache')
    distributions={};subject_tables={}
    selections=dict(input=(['component','unit'],['sd','ssm_scaled_sd','ptp','rms','constant_fraction']),
        components=(['component'],['x_sd_subject','s_sd_ratio','o_sd_ratio','correlation_sx','correlation_so','variance_sum_ratio','local_nrmse','subject_nrmse','severe_cancellation','faithful_small_residual','negative_correlation','residual_ge_signal']),
        spectra=(['source','component'],['centroid','fast_fraction','mobility','kurtosis','delta','theta','alpha','beta','gamma']),
        tasks=([],['tau','beta','tau_at_bound','beta_at_bound','x_HbO_HbR_r','s_HbO_HbR_r','o_HbO_HbR_r','flow_volume_r']+[n+'_'+m for n in ['driver','vasodilatory','flow','volume','deoxy'] for m in ['mean','sd','min','max','rms_rest']]),
        overlap=([],['seconds','HbO_subject_nrmse','HbR_subject_nrmse','HbO_local_nrmse','HbR_local_nrmse','flow_rms','volume_rms','deoxy_rms']))
    for name,(groups,metrics) in selections.items():
        frame=frames[name];frame=frame[frame.qualified] if name=='tasks' else frame
        metrics=[k for k in metrics if k in frame]
        sub=cache_audit_aggregate(frame,groups,metrics);sub.to_csv(out/('subject_'+name+'.csv'),index=False);subject_tables[name]=sub
        rows=[]
        for key,g in sub.groupby(['dataset']+groups,dropna=False):
            key=key if isinstance(key,tuple) else (key,);base=dict(zip(['dataset']+groups,key));base['subjects']=len(g)
            for m in metrics:
                z=g[m].dropna();base[m+'_median']=float(z.median()) if len(z) else None;base[m+'_p10']=float(z.quantile(.1)) if len(z) else None;base[m+'_p90']=float(z.quantile(.9)) if len(z) else None
            rows.append(base)
        distributions[name]=rows
    paired=frames['components'].merge(frames['baseline'],on=['dataset','subject','record','window','pair','native_index','duration_s','split','component'],suffixes=('_new','_old'))
    metrics=['s_sd_ratio','o_sd_ratio','correlation_sx','severe_cancellation','faithful_small_residual']
    for m in metrics:paired[m+'_delta']=paired[m+'_new']-paired[m+'_old']
    sub=cache_audit_aggregate(paired,['component'],[m+s for m in metrics for s in ['_new','_old','_delta']]);sub.to_csv(out/'visual_paired_subjects.csv',index=False)
    rng=np.random.default_rng(20261110);paired_summary=[]
    for component,g in sub.groupby('component'):
        for m in metrics:
            z=g[m+'_delta'].dropna().to_numpy();means=rng.choice(z,(2000,len(z)),replace=True).mean(1)
            paired_summary.append(dict(component=component,metric=m,subjects=len(z),old_subject_equal_mean=float(g[m+'_old'].mean()),new_subject_equal_mean=float(g[m+'_new'].mean()),delta=float(z.mean()),CI95=np.quantile(means,[.025,.975]).tolist()))
    raw_rates=[]
    for (ds,c),g in frames['components'].groupby(['dataset','component']):
        valid=g[g.defined];raw_rates.append(dict(dataset=ds,component=c,qualified=len(g),defined=len(valid),undefined=len(g)-len(valid),severe_cancellation=int(valid.severe_cancellation.sum()),faithful_small_residual=int(valid.faithful_small_residual.sum()),negative_correlation=int(valid.negative_correlation.sum())))
    coverage=[]
    for ds,g in frames['samples'].groupby('dataset'):
        coverage.append(dict(dataset=ds,samples=len(g),dependencies=plan['dependency_counts'][ds],joined=int((g.teacher_join=='available').sum()),any_qualified=int((g.teacher_qualified_pairs>0).sum()),all_qualified=int(((g.teacher_qualified_pairs==g.teacher_pairs)&(g.teacher_join=='available')).sum()),qualified_pair_fraction=float(g.teacher_qualified_pairs.sum()/g.teacher_pairs.sum()),partial=int(g.partial.sum()),touch_last_bin=int(g.teacher_last_bin_in_sample.fillna(False).sum()),mean_EEG_valid=float(g.eeg_valid_fraction.mean()),mean_Hb_valid=float(g.fnirs_valid_fraction.mean())))
    summary=dict(schema=cfg['schema'],status='completed',counts=counts,coverage=coverage,split_counts=plan['split_counts'],raw_component_rates=raw_rates,distributions=distributions,visual_paired=paired_summary,
        transitions=frames['transitions'].groupby(['old_qualified','new_qualified']).size().reset_index(name='count').to_dict('records'),
        per_dataset=status.groupby('dataset')[['samples','tasks','qualified','failed']].sum().reset_index().to_dict('records'),
        input_versions=dict(measurement=cfg['measurement'],input=cfg['input'],ssm=cfg['ssm']),checks=dict(record_array_finite_support=True,measurement_input_scale=True,all_consumer_samples=True,task_identity_and_no_extras=True,all_qualified_array_closure_state_and_scale=True,overlap_target_max_abs=float(frames['overlap'].target_max_abs.max()),subject_split_disjoint=True),
        aggregation=cfg['aggregation'],refits=0,training_runs=0)
    write_json(out/'summary.json',summary)
    return dict(status='completed',counts=counts,coverage=coverage)


def cache_audit_report(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from html import escape
    out=Path(cfg['output']);s=json.loads((out/'summary.json').read_text());labels={'eeg_fnirs_single_trial':'Single','simultaneous_eeg_nirs':'Sim','refed':'REFED','visual_cognitive_motivation':'Visual'};order=list(labels)
    plt.rcParams.update({'font.size':10,'figure.dpi':130})
    records=pd.read_csv(out/'records.csv',dtype={'subject':str});comp=pd.read_csv(out/'subject_components.csv',dtype={'subject':str});task=pd.read_csv(out/'subject_tasks.csv',dtype={'subject':str});ov=pd.read_csv(out/'subject_overlap.csv',dtype={'subject':str});paired=pd.read_csv(out/'visual_paired_subjects.csv',dtype={'subject':str})
    fig,ax=plt.subplots(1,2,figsize=(11,4.4))
    for k,c in enumerate(['HbO','HbR']):
        for j,ds in enumerate(order):
            g=comp[(comp.dataset==ds)&(comp.component==c)]
            ax[0].scatter(np.full(len(g),j)+(k-.5)*.18,g.severe_cancellation*100,s=18,alpha=.6,label=c if j==0 else None)
            ax[1].scatter(np.full(len(g),j)+(k-.5)*.18,g.faithful_small_residual*100,s=18,alpha=.6,label=c if j==0 else None)
    for a,title in zip(ax,['Large opposing S/O components','Shape preserved with small residual']):
        a.set_xticks(range(4),[labels[d] for d in order]);a.set_ylabel('Record-equal rate within subject (%)');a.set_title(title);a.legend();a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'cancellation.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4.2))
    for j,c in enumerate(['HbO','HbR']):
        g=paired[paired.component==c]
        for _,r in g.iterrows():ax[j].plot([0,1],[r.severe_cancellation_old*100,r.severe_cancellation_new*100],'-o',alpha=.6,ms=3)
        ax[j].set_xticks([0,1],['v3 usable pairs','v4 same pairs']);ax[j].set_title(c+' / paired subjects');ax[j].set_ylabel('Severe cancellation rate (%)');ax[j].grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'visual_paired.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    for j,ds in enumerate(order):
        g=task[task.dataset==ds];ax[0].scatter(np.full(len(g),j),g.tau,s=18,alpha=.6);ax[1].scatter(np.full(len(g),j),g.beta,s=18,alpha=.6)
        z=ov[ov.dataset==ds];ax[2].scatter(np.full(len(z),j)-.08,z.HbO_subject_nrmse,s=18,alpha=.6,label='HbO' if j==0 else None);ax[2].scatter(np.full(len(z),j)+.08,z.HbR_subject_nrmse,s=18,alpha=.6,label='HbR' if j==0 else None)
    for a,title,y in zip(ax,['Effective transit time','Effective coupling','Overlapping window disagreement'],['tau (s)','beta (model coordinates)','RMS difference / subject component SD']):
        a.set_xticks(range(4),[labels[d] for d in order]);a.set_title(title);a.set_ylabel(y);a.grid(alpha=.2)
    ax[2].legend();fig.tight_layout();fig.savefig(out/'parameters_overlap.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4.1));inp=pd.read_csv(out/'subject_input.csv',dtype={'subject':str})
    for j,ds in enumerate(order):
        for k,c in enumerate(['HbO','HbR']):
            g=inp[(inp.dataset==ds)&(inp.component==c)];ax[0].scatter(np.full(len(g),j)+(k-.5)*.16,g.ssm_scaled_sd,s=18,alpha=.6,label=c if j==0 else None)
        g=records[records.dataset==ds];ax[1].scatter(np.full(len(g),j),g.Hb_supported_positions,s=13,alpha=.3)
    ax[0].set_yscale('log');ax[0].set_ylabel('SD in frozen subject Hb coordinates');ax[0].set_title('Input-window amplitude / each point = subject');ax[0].legend();ax[1].set_ylabel('Available HbO/HbR locations');ax[1].set_title('Input channel count / each point = record')
    for a in ax:a.set_xticks(range(4),[labels[d] for d in order]);a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'input_amplitude_channels.png',dpi=220);plt.close(fig)
    def table(rows):
        f=rows.copy() if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
        if 'dataset' in f:f['dataset']=f.dataset.replace(labels)
        return f.to_html(index=False,float_format=lambda v:f'{v:.4g}',border=0,escape=True)
    def dist(name,metrics):
        rows=[]
        for d in s['distributions'][name]:
            row={k:d[k] for k in ['dataset','component','source','unit','subjects'] if k in d}
            for m in metrics:
                a=d.get(m+'_median');lo=d.get(m+'_p10');hi=d.get(m+'_p90')
                row[m]='—' if a is None else f'{a:.4g} [{lo:.4g}, {hi:.4g}]'
            rows.append(row)
        return table(rows)
    h='<!doctype html><html lang="zh"><meta charset="utf-8"><title>Tokenizer 缓存完整性与生理统计审计</title><style>body{font-family:Arial,"Noto Sans CJK SC",sans-serif;max-width:1250px;margin:32px auto;line-height:1.65;color:#18232f}table{border-collapse:collapse;display:block;overflow:auto;font-size:13px}th,td{padding:6px 9px;border-bottom:1px solid #ddd;text-align:right}th{background:#eef3f6}img{width:100%;height:auto}code{background:#eef3f6}h2{margin-top:38px}p{max-width:1100px}.note{background:#f1f6fa;padding:14px}</style>'
    h+='<h1>Tokenizer 开发前的缓存完整性与生理统计审计</h1><p>数据快照：2026-11-10。底层测量 v6、下游输入 v2、SSM v4；v4 中非 Visual 任务保留原 v3 身份。全部公开缓存的描述性审计，不重新拟合、训练或访问保护评价。原始数据与历史结果保持原值。</p>'
    h+=f'<p class="note">已核对 {s["counts"]["records"]} 条缓存记录、{s["counts"]["samples"]} 个输入样本与 {s["counts"]["planned"]} 个求解单元；数值合格 {s["counts"]["qualified"]}，失败 {s["counts"]["failed"]}。数值合格表示既定收敛、积分和状态检查通过；波形恢复及分量可用性由下方独立统计描述。</p>'
    h+='<h2>1. 完整性、分区和训练单元</h2>'+table(s['per_dataset'])
    h+=table([dict(dataset=ds,**v) for ds,v in s['split_counts'].items()])
    h+='<p>样本 ID 唯一，被试分区互斥。所有已声明记录的数组形状、类型、支持范围内有限值、几何以及全部实际加载样本的掩码/补零均已检查；Hb 底层测量与下游存储尺度逐记录复现；SSM 任务按计划完整枚举、无额外通道。所有合格数组逐一检查 X=S+O、尺度、时钟和 p=v。失败解不作为有效目标。</p>'
    h+='<h2>2. 下游短窗能获得多少有效 teacher 目标</h2>'+table(s['coverage'])
    h+='<p>joined：存在同时覆盖 EEG/Hb 支持区间的求解窗；any_qualified / all_qualified：至少一个 / 所有候选位置有合格求解；qualified_pair_fraction：合格位置数之和除以所有样本候选位置数之和，包含未匹配样本。dependencies 是事件/记录依赖身份的数量（不是独立被试数），不能把共享 EEG 的双 Probe 样本当作独立事件。partial 是保留的短尾样本；touch_last_bin 表示样本触及 SSM 不参与拟合评分的最后100 ms，需要额外的目标有效掩码。现有输入掩码与 teacher 掩码是不同对象。</p>'
    h+='<h2>3. 通道、单位与归一化</h2>'+table(records.groupby('dataset').agg(records=('record','size'),EEG_min=('EEG_supported_channels','min'),EEG_max=('EEG_supported_channels','max'),Hb_min=('Hb_supported_positions','min'),Hb_max=('Hb_supported_positions','max')).reset_index())
    h+=table(records[['dataset','EEG_unit','Hb_unit']].drop_duplicates())
    h+='<p>一个 Hb 位置包含 HbO/HbR 两条信号。Visual 保留逐记录自身的可用位置，使用 Probe:CH 身份及 compact_index 对齐；不同记录的列号不可直接对应。底层 Hb 保持测量坐标，下游 v2 仅 Visual 已乘被试共同 Hb 因子；其他数据集仍保留各自测量坐标。SSM npz 的 target/prediction/residual 保存未缩放坐标，scale 明确记录进入求解器的乘数。归一化适配器必须根据元数据决定是否乘尺度，而不是对全部缓存统一再乘一次。</p>'
    h+='<img src="input_amplitude_channels.png"><p>左图在已有 SSM 被试尺度下比较各任务原生长度输入的 SD：先对可用通道取中位数，再按记录和被试等权汇总。各被试的总体校准目标相同不意味着每个局部窗口具有相同幅度。右图为逐记录可用位置数，重复点叠加。</p>'+dist('input',['sd','ssm_scaled_sd','constant_fraction'])
    h+='<h2>4. 频谱与形态</h2><p>频谱面板按结果无关的首/中/末样本或窗口与通道预先选择。输入谱统计只使用至少30秒的实际支持；2秒任务窗口无法分辨0.2 Hz以下的频谱，不进入这组输入谱统计。Hb 与 SSM EEG 代理使用10 Hz、常量去趋势 Hann 周期图；fast_fraction 为 >0.2 Hz 的相对功率，centroid 单位 Hz，mobility 为相邻差分 SD×采样率 / 信号 SD。短窗谱泄漏可产生带通范围外功率，不将其直接解释为真实高频生理活动。输入 EEG 的 delta/theta/alpha/beta/gamma 对应1–4/4–8/8–13/13–30/30–45 Hz；SSM 的 EEG 是局部六电极×五频带的 PC1 对数功率，10 Hz 代理频率描述包络变化，不能与原电压振荡频率混为一谈。</p>'
    h+=dist('spectra',['centroid','fast_fraction','mobility','delta','theta','alpha','beta','gamma'])
    h+='<h2>5. 波形恢复与 S/O 反向抵消</h2><p>X 为当前窗口观测，S 为 SSM 完整条件预测，O=X−S。所有比值的 SD 均先在该窗口评分区间去均值；S_SD/X_SD 与 O_SD/X_SD 不属于互斥的能量占比，二者平方和可以远大于1。强抵消沿用既有诊断阈值：S_SD/X_SD≥2、O_SD/X_SD≥1 且 corr(S,O)≤−0.8；形态保留且小残差为 corr(S,X)≥0.8 且 O_SD/X_SD≤0.5。这些是描述性规则，不是自动删样本标准。</p>'
    h+='<img src="cancellation.png">'+table(s['raw_component_rates'])
    h+='<p>上表分母为合格任务的各分量，defined 保留相关/幅度比可定义的部分；数值失败和未定义项分别报告。散点每点为一名被试在该分量上的记录等权发生率。下面表格为被试统计的中位数 [P10,P90]；聚合顺序是窗口×位置在记录内平均、记录在被试内等权平均，再描述被试分布。</p>'+dist('components',['correlation_sx','s_sd_ratio','o_sd_ratio','local_nrmse','subject_nrmse','severe_cancellation','faithful_small_residual'])
    h+='<h2>6. Visual 同一批可用通道的修正前后比较</h2><p>只比较新版本保留的相同位置/窗口，连续指标使用两版本均合格的配对集合；同时保留完整资格转换计数。避免把删除坏通道的集合变化和同一通道的求解改善混在一起。区间为按被试重采样2000次的探索性95% bootstrap 区间，冻结已有拟合；窗口不是独立被试，未进行确认性显著性结论。</p>'+table(s['transitions'])+'<img src="visual_paired.png">'+table(s['visual_paired'])
    h+='<h2>7. 参数、潜状态与重复时间的一致性</h2><p>τ 的单位为秒，β 是当前个体投影/尺度下的有效耦合参数；其先验中心分别为2、1，log先验 SD=0.125，边界为[0.25,12]、[0.1,10]。它们是当前模型条件估计，不是独立测得的个体生理真值。状态依次为 driver、vasodilatory、flow、volume、p、deoxy；p 与 volume 完全相同，不能作为两条独立的生理自由度。flow/volume/deoxy 以静息值1为参考，driver/vasodilatory 的参考为0。</p>'
    h+='<img src="parameters_overlap.png">'+dist('tasks',['tau','beta','tau_at_bound','beta_at_bound','x_HbO_HbR_r','s_HbO_HbR_r','o_HbO_HbR_r','flow_sd','volume_sd','deoxy_sd'])
    h+='<p>重叠统计比较同一记录、同一位置、相邻且均合格的独立求解窗。在共同 Hb 时间上恢复各自记录的窗口参考后，检查 X 一致，再比较 S。两窗都有约束却仍可产生不同的条件估计；不能直接把它们当成一条无缝连续状态轨迹。</p>'+dist('overlap',['seconds','HbO_subject_nrmse','HbR_subject_nrmse','HbO_local_nrmse','HbR_local_nrmse','flow_rms','volume_rms','deoxy_rms'])
    h+='<h2>8. 对 tokenizer 输入和监督接口的要求</h2><ol><li>保留 dataset/subject/session/Probe/channel/time/split/dependency 身份、几何、原始单位和尺度来源，按通道身份处理变长集合；所有损失只在实际有效位置和时间计算。</li><li>区分200 Hz EEG电压、10 Hz EEG对数功率 PC1、10 Hz Hb 和模型潜状态。当前 SSM 不能直接提供原始 EEG 电压域的 S/O 真值；个体 PC1 的方向和电极邻域也不是跨被试固定语义。</li><li>构建独立的 teacher 可用掩码：无包含窗、未收敛、最后未评分时间格及缺失通道均显式保留。数值资格与波形质量分别记录，不能把所有 qualified 目标无差别当作高质量生理标签。</li><li>保留 S/O 抵消与局部幅度质量特征，训练损失的分母来源显式化。Visual 的旧公共参考 SD 受历史不可用通道影响，其公共 NRMSE 很小不能单独证明形态恢复。</li><li>90秒求解窗包含30秒输入窗外的上下文，属于离线、双模态条件 teacher。原有按全部被试自身无标签窗口进行的校准属于个体适配；若要严格零样本或实时部署，需要另定可见信息与校准支持。</li><li>同一物理时间可能对应多个独立窗口的状态估计，使用已保存的唯一 sample-to-fit 规则，先验证重叠一致性再考虑连续状态监督。p=v 的重复自由度应合并。</li><li>数据集任务数、窗口数与有效通道数不平衡，训练采样应明确被试/记录/位置权重；双 Probe 共享的 EEG 和重叠窗口不增加独立样本量。先建立可复现的统一适配器和掩码，再开展 tokenizer 实验。</li></ol>'
    h+='<p>完整证据：'+ ' · '.join(f'<a href="{name}">{name}</a>' for name in ['summary.json','records.csv','samples.csv','tasks.csv','components.csv','spectra.csv','overlap.csv','visual_paired_subjects.csv','analysis_config.json'])+'</p></html>'
    (out/'report.html').write_text(h,encoding='utf-8')


def cache_audit_join_record(payload):
    cfg,r,indices,coordinate=payload;dataset=_CACHE_AUDIT_INPUT
    out=Path(cfg['output'])/'records'/r['dataset']/r['subject']/r['record'];path=out/'join_check.json'
    if path.exists():return json.loads(path.read_text())
    joins={j['sample_id']:j for j in cfg['joins'].get(r['cache_key'],[])};prepared={};count=0;maximum=0.;phase=0.;pairs=0;naive_mismatch=0;declared_clock_mismatch=0;missing_steps=0;examples=[]
    for i in indices:
        sample=dataset[i];j=joins[sample['id']]
        if j['status']!='available':continue
        f=Path(cfg['ssm'])/'features'/r['cache_key']/(j['window_key']+'.npz')
        if f not in prepared:
            with np.load(f,allow_pickle=False) as a:prepared[f]=(a['hb'],a['hb_reference'],a['hb_valid'])
        h,reference,valid=prepared[f]
        w=next(w for w in r['cache_windows'] if w['key']==j['window_key'])
        start=dataset.samples[i]['hb_start_s'];actual_start=round((start*1000)/1000*10);offset=actual_start-round(w['hb_start_s']*10)
        naive=round(j['hb_offset_s']*10);naive_mismatch+=int(offset!=naive)
        declared_clock_mismatch+=int(actual_start!=round(start*10))
        if offset!=naive:examples.append(dict(sample_id=sample['id'],actual_offset=offset,naive_relative_offset=naive,requested_start=start,window_start=w['hb_start_s']))
        phase=max(phase,abs(offset/10-j['hb_offset_s']))
        x=sample['fnirs'].T.reshape(-1,len(r['pairs']),2);mask=sample['valid_mask']['fnirs'].T.reshape(x.shape)
        n=min(len(x),len(h)-offset);factor=1.
        if sample['fnirs_normalization']:factor=json.loads(Path(sample['fnirs_normalization']['source']).read_text())['factor']
        keep=mask[:n]&valid[None,:,None]
        expected=(h[offset:offset+n]+reference)*factor
        if keep.any():
            error=float(np.max(abs(x[:n][keep]-expected[keep])));maximum=max(maximum,error)
            np.testing.assert_allclose(x[:n][keep],expected[keep],rtol=1e-10,atol=1e-10)
        missing_steps+=int(np.any(mask[n:],axis=(1,2)).sum())
        count+=1;pairs+=int(valid.sum())
    result=dict(dataset=r['dataset'],subject=r['subject'],record=r['record'],samples=len(indices),joined=count,checked_pair_samples=pairs,max_abs_error=maximum,max_grid_rounding_s=phase,naive_offset_mismatched_samples=naive_mismatch,declared_clock_mismatched_samples=declared_clock_mismatch,teacher_missing_valid_steps=missing_steps,examples=examples,status='passed')
    write_json(path,result);return result


def cache_audit_join_main(cfg,plan,workers):
    from concurrent.futures import wait,FIRST_COMPLETED
    out=Path(cfg['output']);co=json.loads(Path(plan['coordinate']).read_text())['individual'];runtime=dict(cfg,joins=plan['joins']);start=time.monotonic()
    payloads=iter((runtime,r,plan['sample_indices'].get(r['join_key'],[]),co[r['dataset']+'__'+r['subject']]) for r in plan['records']);rows=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=cache_audit_init,initargs=(cfg['input'],)) as pool:
        active={}
        def submit():
            try:v=next(payloads)
            except StopIteration:return
            active[pool.submit(cache_audit_join_record,v)]=True
        for _ in range(2*workers):submit()
        while active:
            done,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
            for f in done:active.pop(f);rows.append(f.result());submit()
    frame=pd.DataFrame(rows);frame.to_csv(out/'join_check.csv',index=False)
    result=dict(status='passed',records=len(rows),samples=int(frame.samples.sum()),joined=int(frame.joined.sum()),max_abs_error=float(frame.max_abs_error.max()),max_grid_rounding_s=float(frame.max_grid_rounding_s.max()),naive_offset_mismatched_samples=int(frame.naive_offset_mismatched_samples.sum()),declared_clock_mismatched_samples=int(frame.declared_clock_mismatched_samples.sum()),teacher_missing_valid_steps=int(frame.teacher_missing_valid_steps.sum()),seconds=time.monotonic()-start,workers=workers,by_dataset=frame.groupby('dataset')[['samples','joined','checked_pair_samples','naive_offset_mismatched_samples','declared_clock_mismatched_samples','teacher_missing_valid_steps']].sum().reset_index().to_dict('records'))
    write_json(out/'join_check.json',result);print(json.dumps(result))


def cache_audit_review(cfg):
    """Interpret the complete census, retaining low-variance tails and grid findings."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from html import escape
    out=Path(cfg['output']);s=json.loads((out/'summary.json').read_text());join=json.loads((out/'join_check.json').read_text())
    comp=pd.read_csv(out/'components.csv',dtype={'subject':str});samples=pd.read_csv(out/'samples.csv',dtype={'subject':str});baseline=pd.read_csv(out/'baseline.csv',dtype={'subject':str})
    rows=[]
    for (ds,c),g in comp.groupby(['dataset','component']):
        valid=g[g.defined];low=g.x_sd_subject<.001
        rows.append(dict(dataset=ds,component=c,qualified=len(g),undefined=int((~g.defined).sum()),low_variance=int(low.sum()),
            amplitude_ratio_median=float(valid.s_sd_ratio.median()),amplitude_ratio_p95=float(valid.s_sd_ratio.quantile(.95)),
            residual_ratio_median=float(valid.o_sd_ratio.median()),local_nrmse_median=float(valid.local_nrmse.median()),
            severe_cancellation_rate=float(valid.severe_cancellation.mean()),faithful_rate=float(valid.faithful_small_residual.mean())))
    robust=pd.DataFrame(rows);robust.to_csv(out/'robust_component_statistics.csv',index=False)
    low=comp[(comp.x_sd_subject<.001)&(comp.component!='EEG')];low.to_csv(out/'low_variance_windows.csv',index=False)
    durations=samples.groupby(['dataset','duration_s']).agg(samples=('sample_id','size'),supported_min=('supported_seconds','min'),supported_max=('supported_seconds','max')).reset_index()
    durations.to_csv(out/'sample_durations.csv',index=False)
    bad=samples[samples.teacher_join!='available'];bad.to_csv(out/'unmatched_samples.csv',index=False)
    matched=comp.merge(baseline,on=['dataset','subject','record','window','pair','native_index','duration_s','split','component'],suffixes=('_new','_old'))
    paired_robust=[]
    for c,g in matched.groupby('component'):
        for m in ['s_sd_ratio','o_sd_ratio']:
            # Robust location for amplitude ratios; no near-flat rows discarded.
            rr=g.groupby(['subject','record'])[[m+'_old',m+'_new']].median().groupby('subject').median()
            paired_robust.append(dict(component=c,metric=m,subjects=len(rr),old=float(rr[m+'_old'].median()),new=float(rr[m+'_new'].median()),aggregation='window_pair_median_then_record_median_then_subject_median'))
    example=low[low.dataset=='visual_cognitive_motivation'].sort_values('x_sd_subject').iloc[0]
    path=Path(cfg['ssm'])/'tasks'/example.dataset/example.subject/example.record/(example.window+f'__p{int(example.native_index):03d}.npz')
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    with np.load(path) as a:
        for j,ax in enumerate(axes,1):
            for label,key,color in [('X','target','#53657a'),('S','prediction','#2083db'),('O','residual','#e65b38')]:ax.plot(a['time_s'][:-1],a[key][:-1,j]*a['scale'][j],label=label,color=color,lw=1.)
            ax.set_ylabel(['HbO','HbR'][j-1]+' / subject coordinate');ax.legend(ncol=3);ax.grid(alpha=.2)
    axes[0].set_title(f'Lowest-variance Visual qualified example: {example.subject} / {example.record} / {example.pair} / {example.window}')
    axes[1].set_xlabel('Time within fitted window (s)');fig.tight_layout();fig.savefig(out/'low_variance_example.png',dpi=220);plt.close(fig)
    review=dict(status='completed',join_check=join,durations=durations.to_dict('records'),robust_components=rows,robust_visual_paired=paired_robust,
        low_variance_threshold='X centered SD < 0.001 * frozen subject component SD; diagnostic only',low_variance_count=len(low),
        example=example[['dataset','subject','record','window','pair','component','x_sd_subject','s_sd_ratio']].to_dict(),
        unmatched_by_dataset=bad.groupby('dataset').size().to_dict(),meaning='descriptive_cache_readiness_not_physiological_truth_or_tokenizer_training_qualification')
    write_json(out/'review_summary.json',review)
    def table(v):return pd.DataFrame(v).to_html(index=False,float_format=lambda x:f'{x:.5g}',border=0)
    visual={ (r['component'],r['metric']):r for r in s['visual_paired']}
    findings='<h2>主要结论与开发优先级</h2><p>存储完整性与身份检查通过，完整缓存可以作为后续 tokenizer 的可追溯输入与条件监督来源；直接训练前仍需显式的单位/尺度适配、teacher 掩码和时间格连接。现有数值资格不能替代局部波形质量。</p>'
    findings+='<ul>'
    for c in ['HbO','HbR']:
        q=visual[(c,'severe_cancellation')];r=visual[(c,'correlation_sx')]
        findings+=f'<li>Visual {c} 同一可用位置的强抵消发生率由{q["old_subject_equal_mean"]*100:.2f}%降至{q["new_subject_equal_mean"]*100:.3f}%；预测与观测相关由{r["old_subject_equal_mean"]:.3f}升至{r["new_subject_equal_mean"]:.3f}。以上是记录/被试等权配对均值。</li>'
    q=visual[('EEG','correlation_sx')];findings+=f'<li>共同求解存在代价：Visual EEG PC1 的相关由{q["old_subject_equal_mean"]:.3f}降至{q["new_subject_equal_mean"]:.3f}；EEG输入未改变，差异来自新的Hb尺度下重新联合优化。相同可用任务中的数值失败由4增至126，全部保留。</li>'
    findings+=f'<li>全部{join["joined"]}个可连接样本，在绝对采样格上恢复Hb参考后与输入一致，最大差异{join["max_abs_error"]:.3g}。若直接对相对时间差四舍五入，{join["naive_offset_mismatched_samples"]}个样本会选到不同格；另有{join["declared_clock_mismatched_samples"]}个样本的当前返回时间轴与切片运算的半格取整结果不同。应使用记录级整数样本坐标，不能直接round(relative_offset×rate)。本轮仅记录问题，未改写已有缓存。</li>'
    findings+='<li>25个输入样本没有完整包含的求解窗（Sim 21、Visual 4），694个求解任务数值失败。REFED保留416个短尾输入；输入补零、teacher无解和最后未评分格需要不同掩码。</li><li>设备可用标记不足以保证每个时间窗有有效动态。Visual保留少量近常量片段，极小分母仍会产生很大的幅度比；它们没有从结果中删去。HbR及REFED的残差和形态问题仍应单独评估。</li></ul>'
    findings+='<h3>样本实际长度与频谱支持</h3>'+table(durations)
    findings+='<p>输入样本长度不是统一30秒：不同任务的原始长度按既有合同保留。输入频谱面板只使用至少30秒的实际支持，2秒窗口只统计幅度/有效性；跨数据集低频比较主要参考相同90秒SSM支持，REFED的60秒支持保留单独身份。</p>'
    findings+='<h3>稳健幅度统计与近常量尾部</h3>'+table(robust[robust.component!='EEG'])+table(paired_robust)
    findings+='<p>上表第一组给出全体合格任务的分布中位数/P95及原始发生率；第二组按窗口位置中位数→记录中位数→被试中位数比较Visual幅度。所有近常量窗口仍保留。后文原始配对幅度均值会被少数极小分母主导，不用于描述典型幅度。低动态阈值0.001倍被试分量SD是诊断阈值，不是新的自动剔除规则。</p>'
    findings+='<img src="low_variance_example.png"><p>例图按Visual合格任务中最低相对输入SD选择，用于解释剩余失效模式，不代表典型被试。X、S、O在同一被试尺度和同一坐标轴绘制；X近乎平直，而S与O仍有相反波动。设备标记在该位置仍为可用，因此全局设备通道交集也不能替代时间窗质量检查。</p>'
    findings+='<p>补充证据：<a href="join_check.json">逐点连接检查</a> · <a href="robust_component_statistics.csv">稳健分布</a> · <a href="low_variance_windows.csv">完整低动态窗口清单</a> · <a href="unmatched_samples.csv">无teacher匹配的输入清单</a> · <a href="review_summary.json">本页结论数值</a></p>'
    body=(out/'report.html').read_text();body=body.replace('<h2>1. 完整性、分区和训练单元</h2>',findings+'<h2>1. 完整性、分区和训练单元</h2>',1)
    (out/'report_review.html').write_text(body,encoding='utf-8')


def cache_audit_delivery(cfg):
    """Versioned final figures and same-duration spectral sensitivity, no source writes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from html.parser import HTMLParser
    out=Path(cfg['output']);summary=json.loads((out/'summary.json').read_text());review=json.loads((out/'review_summary.json').read_text());labels={'eeg_fnirs_single_trial':'Single','simultaneous_eeg_nirs':'Sim','refed':'REFED','visual_cognitive_motivation':'Visual'};order=list(labels);colors=['#2083db','#e65b38']
    comp=pd.read_csv(out/'subject_components.csv');inp=pd.read_csv(out/'subject_input.csv');records=pd.read_csv(out/'records.csv');task=pd.read_csv(out/'subject_tasks.csv');ov=pd.read_csv(out/'subject_overlap.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for k,c in enumerate(['HbO','HbR']):
        for j,ds in enumerate(order):
            g=comp[(comp.dataset==ds)&(comp.component==c)]
            for a,key in zip(axes,['severe_cancellation','faithful_small_residual']):a.scatter(np.full(len(g),j)+(k-.5)*.18,g[key]*100,s=18,alpha=.6,color=colors[k],label=c if j==0 else None)
    for a,title in zip(axes,['Large opposing S/O components','Shape preserved with small residual']):a.set_xticks(range(4),[labels[d] for d in order]);a.set_ylabel('Record-equal rate within subject (%)');a.set_title(title);a.legend();a.grid(alpha=.2)
    axes[1].legend(loc='upper center')
    fig.tight_layout();fig.savefig(out/'cancellation_final.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for j,ds in enumerate(order):
        for k,c in enumerate(['HbO','HbR']):
            g=inp[(inp.dataset==ds)&(inp.component==c)];axes[0].scatter(np.full(len(g),j)+(k-.5)*.16,g.ssm_scaled_sd,s=18,alpha=.6,color=colors[k],label=c if j==0 else None)
        g=records[records.dataset==ds];axes[1].scatter(np.full(len(g),j),g.Hb_supported_positions,s=13,alpha=.3,color='#53657a')
    axes[0].set_yscale('log');axes[0].set_ylabel('SD in frozen subject Hb coordinates');axes[0].set_title('Input-window amplitude / each point = subject');axes[0].legend();axes[1].set_ylabel('Available HbO/HbR locations');axes[1].set_title('Input channel count / each point = record')
    for a in axes:a.set_xticks(range(4),[labels[d] for d in order]);a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'input_amplitude_channels_final.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for j,ds in enumerate(order):
        g=task[task.dataset==ds];axes[0].scatter(np.full(len(g),j),g.tau,s=18,alpha=.6,color='#53657a');axes[1].scatter(np.full(len(g),j),g.beta,s=18,alpha=.6,color='#53657a')
        g=ov[ov.dataset==ds]
        for k,c in enumerate(['HbO','HbR']):axes[2].scatter(np.full(len(g),j)+(k-.5)*.16,g[c+'_subject_nrmse'],s=18,alpha=.6,color=colors[k],label=c if j==0 else None)
    for a,title,y in zip(axes,['Effective transit time','Effective coupling','Overlapping window disagreement'],['tau (s)','beta (model coordinates)','RMS difference / subject component SD']):a.set_xticks(range(4),[labels[d] for d in order]);a.set_title(title);a.set_ylabel(y);a.grid(alpha=.2)
    axes[2].legend();fig.tight_layout();fig.savefig(out/'parameters_overlap_final.png',dpi=220);plt.close(fig)
    spectra=pd.read_csv(out/'spectra.csv',dtype={'subject':str});panel=spectra[(spectra.source=='SSM_X')&(spectra.duration_s==90)&(spectra.component!='EEG')]
    sub=cache_audit_aggregate(panel,['component'],['centroid','fast_fraction','mobility','kurtosis']);sub.to_csv(out/'spectral_90s_subjects.csv',index=False)
    sensitivity=sub.groupby(['dataset','component'])[['centroid','fast_fraction','mobility','kurtosis']].median().reset_index();sensitivity.to_csv(out/'spectral_90s_summary.csv',index=False)
    raw=pd.read_csv(out/'tasks.csv',usecols=['dataset','qualified','tau_at_bound','beta_at_bound'],dtype={'qualified':'boolean','tau_at_bound':'boolean','beta_at_bound':'boolean'});parameter_counts=raw[raw.qualified].groupby('dataset').agg(qualified=('qualified','size'),tau_at_bound=('tau_at_bound','sum'),beta_at_bound=('beta_at_bound','sum')).reset_index();parameter_counts.to_csv(out/'parameter_bound_counts.csv',index=False)
    coordinates=json.loads((Path(cfg['ssm'])/'coordinates.json').read_text());calibration=[]
    for key,c in coordinates['individual'].items():
        pop=coordinates['population'][c['dataset']]
        calibration.append(dict(dataset=c['dataset'],subject=c['subject'],eeg_factor=c['eeg_factor'],hb_factor=c['hb_factor'],PC_population_cosine=float(np.dot(c['pc'],pop['pc'])),HbO_SD=c['sd'][1],HbR_SD=c['sd'][2]))
    calibration=pd.DataFrame(calibration);calibration.to_csv(out/'calibration_subjects.csv',index=False)
    cal_summary=calibration.groupby('dataset').agg(subjects=('subject','size'),PC_cosine_min=('PC_population_cosine','min'),PC_cosine_median=('PC_population_cosine','median'),hb_factor_min=('hb_factor','min'),hb_factor_median=('hb_factor','median'),hb_factor_max=('hb_factor','max')).reset_index()
    body=(out/'report_review.html').read_text()
    for old,new in [('cancellation.png','cancellation_final.png'),('input_amplitude_channels.png','input_amplitude_channels_final.png'),('parameters_overlap.png','parameters_overlap_final.png')]:body=body.replace(old,new)
    body=body.replace('<h2>4. 频谱与形态</h2>','<h3>被试校准与PCA方向</h3>'+cal_summary.to_html(index=False,float_format=lambda x:f'{x:.5g}',border=0)+'<p>全部被试均有明确的正尺度与PC加载矩阵。相对各自数据集训练参考方向，PC余弦中位数均约0.99，主要方向总体相近；REFED与Visual仍有较明显的个体偏离。原始Hb因子反映不同单位与个体幅度，因子本身不是生理效应量。应保存并使用各被试的实际坐标对象。</p><h2>4. 频谱与形态</h2>')
    extra='<h3>相同90秒观测支持下的频谱敏感性检查</h3><p>下表只用90秒任务的观测X，排除REFED的60秒支持；统计为记录等权后各被试均值的中位数。centroid单位Hz，fast_fraction为比例，mobility单位每秒，kurtosis为超额峰度。</p>'+sensitivity.to_html(index=False,float_format=lambda x:f'{x:.5g}',border=0)
    body=body.replace('<h2>5. 波形恢复与 S/O 反向抵消</h2>',extra+'<h2>5. 波形恢复与 S/O 反向抵消</h2>')
    body=body.replace('<h2>8. 对 tokenizer 输入和监督接口的要求</h2>','<p>全体合格任务的参数触界计数：</p>'+parameter_counts.to_html(index=False,border=0)+'<h2>8. 对 tokenizer 输入和监督接口的要求</h2>')
    if review['join_check']['teacher_missing_valid_steps']:
        body=body.replace('<h3>样本实际长度与频谱支持</h3>',f'<p>绝对采样格核对另发现 {review["join_check"]["teacher_missing_valid_steps"]} 个输入有效时间格超出被选teacher窗的离散支持；这些格必须使用无teacher掩码，不能外推补齐。</p><h3>样本实际长度与频谱支持</h3>')
    body=body.replace('另有0个样本的当前返回时间轴与切片运算的半格取整结果不同。','当前返回时间轴与实际切片一致。')
    ex=review['example'];detail=f"该HbR窗口的X SD/被试分量SD为{ex['x_sd_subject']:.3g}，S SD/被试分量SD为{ex['x_sd_subject']*ex['s_sd_ratio']:.3g}；极大的局部幅度比主要来自极小的分母。"
    body=body.replace('设备标记在该位置仍为可用，因此全局设备通道交集也不能替代时间窗质量检查。',detail+'设备标记在该位置仍为可用，因此全局设备通道交集也不能替代时间窗质量检查。')
    target=out/'report_final.html';target.write_text(body,encoding='utf-8')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ('href','src') and not v.startswith(('http','#')):self.paths.append(v)
    links=Links();links.feed(body);missing=[s for s in links.paths if not (out/s).exists()]
    if missing:raise ValueError('Broken report links: '+str(missing))
    write_json(out/'delivery_checks.json',dict(status='passed',report=target.name,local_links_checked=len(links.paths),missing=missing,figures='PNG_220dpi',spectral_90s_summary=sensitivity.to_dict('records'),parameter_bound_counts=parameter_counts.to_dict('records'),calibration=cal_summary.to_dict('records')))


def tokenizer_cache_audit_main():
    import sys,yaml
    from datetime import datetime,timezone
    from concurrent.futures import wait,FIRST_COMPLETED
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
    p=argparse.ArgumentParser();p.add_argument('--tokenizer-cache-audit',action='store_true');p.add_argument('--stage',choices=['plan','pilot','run','report','join-check','review','delivery'],required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--project-root',type=Path,default=Path.cwd());p.add_argument('--config',type=Path);a=p.parse_args()
    folder=root/'experiments/configs/physiology_semantic_tokenizer'
    cfg=yaml.safe_load((a.config or folder/'tokenizer_cache_audit_v2.yaml').read_text())
    if cfg['schema'] not in ['tokenizer_cache_audit_v1','tokenizer_cache_audit_v2'] or cfg!=yaml.safe_load((folder/(cfg['schema']+'.yaml')).read_text()):raise ValueError('Reviewed audit contract required')
    for key in ['measurement','input','ssm','baseline']:cfg[key]=str((a.project_root/cfg[key]).resolve())
    out=Path(cfg['ssm'])/cfg['output_child'];cfg['output']=str(out)
    if a.stage=='plan':print(json.dumps(cache_audit_plan(cfg,a.project_root,out)));return
    if json.loads((out/'analysis_config.json').read_text())!=cfg:raise ValueError('Frozen audit config changed')
    if a.stage=='report':cache_audit_report(cfg);return
    if a.stage=='review':cache_audit_review(cfg);return
    if a.stage=='delivery':cache_audit_delivery(cfg);return
    if not os.environ.get('SSM_SYSTEMD_UNIT') or not os.environ.get('INVOCATION_ID'):raise ValueError('Durable supervisor required')
    plan=json.loads((out/'plan.json').read_text())
    if a.stage=='join-check':cache_audit_join_main(cfg,plan,a.workers);return
    co=json.loads(Path(plan['coordinate']).read_text())['individual'];records=plan['records']
    if a.stage=='pilot':records=[next(r for r in records if r['dataset']==ds and r['cache_windows']) for ds in sorted({r['dataset'] for r in records})]
    runtime=dict(cfg,lower_records=plan['lower_records'],joins=plan['joins']);payloads=iter((runtime,r,plan['sample_indices'].get(r['join_key'],[]),co[r['dataset']+'__'+r['subject']]) for r in records)
    start=time.monotonic();results=[];write_json(out/'execution.json',dict(status='running',stage=a.stage,workers=a.workers,supervisor=os.environ['SSM_SYSTEMD_UNIT'],started_at=datetime.now(timezone.utc).isoformat()))
    try:
        with ProcessPoolExecutor(max_workers=a.workers,initializer=cache_audit_init,initargs=(cfg['input'],)) as pool:
            active={}
            def submit():
                try:v=next(payloads)
                except StopIteration:return
                active[pool.submit(cache_audit_record,v)]=True
            for _ in range(2*a.workers):submit()
            while active:
                done,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
                for f in done:active.pop(f);results.append(f.result());submit()
                write_json(out/'progress.json',dict(stage=a.stage,completed=len(results),total=len(records),elapsed_s=time.monotonic()-start,workers=a.workers,updated_at=datetime.now(timezone.utc).isoformat()))
        execution=dict(status='completed',stage=a.stage,records=len(results),workers=a.workers,seconds=time.monotonic()-start,peak_worker_mib=max(r['peak_rss_mib'] for r in results),supervisor=os.environ['SSM_SYSTEMD_UNIT'])
        write_json(out/(a.stage+'_execution.json'),execution)
        if a.stage=='run':cache_audit_summary(cfg,plan);cache_audit_report(cfg)
        write_json(out/'execution.json',execution);print(json.dumps(execution))
    except Exception as e:
        import traceback
        write_json(out/'execution.json',dict(status='failed',stage=a.stage,error=traceback.format_exc(),seconds=time.monotonic()-start));raise


if __name__=='__main__':
    import sys
    if '--tokenizer-cache-audit' in sys.argv:
        tokenizer_cache_audit_main()
    elif '--amplitude-comparison' in sys.argv:
        amplitude_comparison_main()
    elif '--visual-cache-review' in sys.argv:
        parser=argparse.ArgumentParser();parser.add_argument('--visual-cache-review',type=Path,required=True);parser.add_argument('--export',type=Path,required=True)
        args=parser.parse_args();visual_cache_report(args.visual_cache_review,args.export)
    else:fnirs_main() if '--fnirs-audit' in sys.argv else main()
